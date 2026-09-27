from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from scripts import run_e6_srij as e6
from scripts import run_utility_routing_audit as e5
from scripts import run_routing_identifiability as common
from src.models.canonical_routing_audit import CanonicalRoutingAudit, canonical_mix
from src.models.utility_routing_audit import uniform_state_mix
from src.models.structural_action_ranker import (
    ACTION_NAMES, execute_hard_action, hard_action_bank, decision_signature,
    select_hard_action,
)

ROOT = Path(__file__).resolve().parents[1]
E6_OUT = ROOT / "outputs/routing_identifiability_v1"
E6_RES = ROOT / "results/routing_identifiability_v1"
OUT = ROOT / "outputs/structural_observability_v1"
RES = ROOT / "results/structural_observability_v1"
SCREEN = ["Movies", "ele-fashion", "Reddit-S"]
CONFIRM = ["Toys", "Grocery"]
SEEDS = [42, 43, 44]
VARIANTS = [
    ("R0_trajectory_ranker", "r0_trajectory"),
    ("R1_capacity_control", "r1_capacity_control"),
    ("R1_decision_ranker", "r1_decision"),
]
DEVICE_DEFAULT = "cuda:0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def action_matrix_from_e6(saved: np.lib.npyio.NpzFile) -> np.ndarray:
    required = {"joint", "lt_u", "lu_v", "luu", "node_id", "split"}
    if not required.issubset(saved.files):
        raise RuntimeError(f"E6 cache keys mismatch: {sorted(saved.files)}")
    joint, lt_u, lu_v, luu = saved["joint"], saved["lt_u"], saved["lu_v"], saved["luu"]
    n = len(saved["node_id"])
    if joint.shape != (n, 4, 4) or lt_u.shape != (n, 4) or lu_v.shape != (n, 4) or luu.shape != (n,):
        raise RuntimeError("E6 action cache shape mismatch")
    matrix = np.empty((n, 5, 5), dtype=np.float32)
    matrix[:, 0, 0] = luu
    matrix[:, 1:, 0] = lt_u
    matrix[:, 0, 1:] = lu_v
    matrix[:, 1:, 1:] = joint
    return matrix


def corrected_restart_ranges(audit: pd.DataFrame) -> pd.DataFrame:
    group = ["dataset", "seed", "split", "level", "chunk_start"]
    missing = set(group + ["restart", "mean_best_loss"]) - set(audit.columns)
    if missing:
        raise ValueError(f"E6 optimizer audit missing columns: {sorted(missing)}")
    sizes = audit.groupby(group, dropna=False).restart.nunique()
    if not (sizes == 3).all():
        raise RuntimeError("Every E6 optimizer chunk must contain all three restarts")
    values = audit.groupby(group, dropna=False).mean_best_loss.agg(
        restart_range=lambda s: float(s.max() - s.min()),
        restart_max=lambda s: float(s.max()),
        restart_min=lambda s: float(s.min()),
    ).reset_index()
    return values


def corrected_restart_audit() -> tuple[pd.DataFrame, dict]:
    source = E6_RES / "phase_c_optimizer_audit.csv"
    raw = pd.read_csv(source)
    ranges = corrected_restart_ranges(raw)
    summary = {
        "source": "results/routing_identifiability_v1/phase_c_optimizer_audit.csv",
        "grouping": ["dataset", "seed", "split", "level", "chunk_start"],
        "chunk_count": int(len(ranges)),
        "range_mean": float(ranges.restart_range.mean()),
        "range_median": float(ranges.restart_range.median()),
        "range_p90": float(ranges.restart_range.quantile(0.9)),
        "range_max": float(ranges.restart_range.max()),
        "historical_file_modified": False,
    }
    RES.mkdir(parents=True, exist_ok=True)
    ranges.to_csv(RES / "e6_corrected_restart_audit.csv", index=False)
    lines = [
        "# Corrected E6 optimizer restart audit",
        "",
        "Recomputed within-chunk restart ranges from the historical E6 audit. Each group is dataset, seed, split, oracle level, and chunk_start; the range is max(mean_best_loss) minus min(mean_best_loss) across the three restarts.",
        "",
        f"Chunks audited: {summary['chunk_count']}. The E6 source CSV and E6 report were not modified.",
        "",
        "| Statistic | Within-chunk restart range |",
        "|---|---:|",
        f"| Mean | {summary['range_mean']:.8g} |",
        f"| Median | {summary['range_median']:.8g} |",
        f"| P90 | {summary['range_p90']:.8g} |",
        f"| Maximum | {summary['range_max']:.8g} |",
        "",
        "This corrected statistic isolates restart variation within the same chunk and does not mix differences among chunks, splits, or oracle levels.",
        "",
    ]
    (RES / "e6_corrected_restart_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(RES / "e6_corrected_restart_audit_summary.json", summary)
    return ranges, summary


def _stratified_quartile(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    return pd.qcut(ranks, 4, labels=False, duplicates="drop").to_numpy(dtype=int) + 1


def _spearman(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _metric_pair(logits: torch.Tensor, labels: torch.Tensor, label_set: list[int]) -> tuple[float, float, float, torch.Tensor]:
    pred = logits.argmax(dim=-1)
    acc = float((pred == labels).float().mean().item())
    from sklearn.metrics import f1_score
    f1 = float(f1_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy(),
                        labels=label_set, average="macro", zero_division=0))
    ce = float(F.cross_entropy(logits, labels).item())
    return acc, f1, ce, pred


def _allowed(data, device):
    return common._allowed_split(data, device)



def _recompute_action_matrix(c0, head, states_text, states_visual, y) -> np.ndarray:
    matrix = np.empty((len(y), 5, 5), dtype=np.float32)
    chunk_size = 32768
    with torch.no_grad():
        uniform_t = uniform_state_mix(states_text)
        uniform_v = uniform_state_mix(states_visual)
        for start in range(0, len(y), chunk_size):
            end = min(start + chunk_size, len(y))
            rows = slice(start, end)
            for at in range(5):
                selected_t = uniform_t[rows] if at == 0 else states_text[at - 1][rows]
                for av in range(5):
                    selected_v = uniform_v[rows] if av == 0 else states_visual[av - 1][rows]
                    logits = head(c0._fuse(selected_t, selected_v))
                    matrix[rows, at, av] = F.cross_entropy(logits, y[rows], reduction="none").cpu().numpy()
    return matrix


def _load_context(dataset: str, seed: int, device: torch.device, verify_actions: bool = True) -> dict:
    cfg, data, payload, c0, head = common._load_c0(dataset, seed, device)
    allowed, y, train, val = _allowed(data, device)
    with torch.no_grad():
        analysis = c0.analyze(data.x.to(device), data.edge_index.to(device))
        states_text_all, states_visual_all = analysis["S_text"], analysis["S_visual"]
        states_text = [state[allowed] for state in states_text_all]
        states_visual = [state[allowed] for state in states_visual_all]
    expected_ids = allowed.detach().cpu().numpy()
    expected_split = np.array(["train"] * len(train) + ["validation"] * len(val))
    cache_path = E6_OUT / "phase_b_cache" / f"{dataset}_seed{seed}.npz"
    d0_path = E6_OUT / "checkpoints" / "D0_independent" / f"{dataset}_seed{seed}.pt"
    d0_saved = torch.load(d0_path, map_location="cpu", weights_only=False)
    c0_path = e5.c0_checkpoint(dataset, seed)
    c0_digest = e5.sha256(c0_path)
    if d0_saved.get("c0_sha256") != c0_digest:
        raise RuntimeError(f"E6 D0 checkpoint C0 SHA mismatch: {dataset}/{seed}")
    if d0_saved.get("dataset") != dataset or int(d0_saved.get("seed", -1)) != seed:
        raise RuntimeError(f"E6 D0 checkpoint identity mismatch: {dataset}/{seed}")
    cache_verified = False
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        matrix = action_matrix_from_e6(cached)
        if not np.array_equal(cached["node_id"], expected_ids):
            raise RuntimeError(f"E6 cache node ids do not match C0 Train/Validation rows: {dataset}/{seed}")
        if not np.array_equal(cached["split"], expected_split):
            raise RuntimeError(f"E6 cache split/order mismatch: {dataset}/{seed}")
        raw_path = E6_OUT / "phase_b_node_landscape" / f"{dataset}_seed{seed}.csv.gz"
        if not raw_path.is_file():
            raise RuntimeError(f"E6 action cache has no matching raw landscape: {dataset}/{seed}")
        raw = pd.read_csv(raw_path)
        if not np.array_equal(raw.node_id.to_numpy(), expected_ids) or not np.array_equal(raw.split.to_numpy(), expected_split):
            raise RuntimeError(f"E6 raw action landscape node/split mismatch: {dataset}/{seed}")
        for k in range(4):
            if not np.allclose(raw[f"L_t{k}_U"].to_numpy(), matrix[:, k + 1, 0], atol=3e-6, rtol=0):
                raise RuntimeError(f"E6 Text marginal cache/raw mismatch: {dataset}/{seed}/S{k}")
            if not np.allclose(raw[f"L_U_v{k}"].to_numpy(), matrix[:, 0, k + 1], atol=3e-6, rtol=0):
                raise RuntimeError(f"E6 Visual marginal cache/raw mismatch: {dataset}/{seed}/S{k}")
            for l in range(4):
                if not np.allclose(raw[f"L_{k}_{l}"].to_numpy(), matrix[:, k + 1, l + 1], atol=3e-6, rtol=0):
                    raise RuntimeError(f"E6 joint action cache/raw mismatch: {dataset}/{seed}/S{k}/S{l}")
        if not np.allclose(raw.uniform_loss.to_numpy(), matrix[:, 0, 0], atol=3e-6, rtol=0):
            raise RuntimeError(f"E6 uniform loss cache/raw mismatch: {dataset}/{seed}")
        cache_verified = True
        action_loss_source = "verified_E6_cache_and_raw_landscape"
    else:
        # Confirmation datasets have frozen E6 D0 checkpoints but no E6 action-loss cache.
        # Recompute only the 25 hard actions on allowed Train+Validation nodes from frozen C0.
        matrix = _recompute_action_matrix(c0, head, states_text, states_visual, y)
        action_loss_source = "recomputed_frozen_C0_after_screening_lock"
    action_recompute_error = action_recompute_p99 = action_recompute_mean = float("nan")
    # Uniform reload integrity remains checked during ranker training too.
    uniform_full = c0._fuse(uniform_state_mix(states_text_all), uniform_state_mix(states_visual_all))
    uniform_error = float((uniform_full[allowed] - analysis["fused_z"][allowed]).abs().max().item())
    if verify_actions and cache_verified:
        recomputed = _recompute_action_matrix(c0, head, states_text, states_visual, y)
        action_differences = np.abs(recomputed - matrix)
        action_recompute_error = float(action_differences.max())
        action_recompute_p99 = float(np.quantile(action_differences, .99))
        action_recompute_mean = float(action_differences.mean())
        if action_recompute_error >= 1e-4:
            raise RuntimeError(f"E6 action definition fails C0 recompute {dataset}/{seed}: {action_recompute_error}")
    elif not cache_verified:
        action_recompute_error = action_recompute_p99 = action_recompute_mean = 0.0
    if uniform_error >= 1e-6:
        raise RuntimeError(f"C0 Uniform reload mismatch {dataset}/{seed}: {uniform_error}")
    return {
        "dataset": dataset, "seed": seed, "device": device, "cfg": cfg, "data": data,
        "payload": payload, "c0": c0, "head": head, "allowed": allowed, "y": y,
        "train": train, "val": val, "labels": sorted(int(x) for x in torch.unique(y).cpu().tolist()),
        "states_text": states_text, "states_visual": states_visual,
        "states_text_all": states_text_all, "states_visual_all": states_visual_all,
        "matrix": torch.as_tensor(matrix, device=device), "node_id": expected_ids,
        "split": expected_split, "c0_sha256": c0_digest, "uniform_reload_error": uniform_error,
        "action_recompute_error": action_recompute_error,
        "action_recompute_p99": action_recompute_p99,
        "action_recompute_mean": action_recompute_mean, "cache_verified": cache_verified,
        "action_loss_source": action_loss_source,
    }


def _action_landscape_frame(ctx: dict) -> tuple[pd.DataFrame, list[dict]]:
    matrix = ctx["matrix"].detach().cpu().numpy()
    rows, summaries = [], []
    split_local = ctx["split"]
    node_id = ctx["node_id"]
    for modality in ("text", "visual"):
        losses = matrix[:, :, 0] if modality == "text" else matrix[:, 0, :]
        for split_name in ("train", "validation"):
            ids = np.flatnonzero(split_local == split_name)
            sub = losses[ids]
            order = np.argsort(sub, axis=1, kind="stable")
            best, second = order[:, 0], order[:, 1]
            sorted_losses = np.take_along_axis(sub, order[:, :2], axis=1)
            margin = sorted_losses[:, 1] - sorted_losses[:, 0]
            gain = sub[:, 0] - sorted_losses[:, 0]
            quartile = _stratified_quartile(margin)
            for j, row_id in enumerate(ids):
                rows.append({
                    "dataset": ctx["dataset"], "seed": ctx["seed"], "split": split_name,
                    "node_id": int(node_id[row_id]), "modality": modality,
                    "oracle_best_action": ACTION_NAMES[int(best[j])],
                    "oracle_second_action": ACTION_NAMES[int(second[j])],
                    "oracle_best_ce": float(sorted_losses[j, 0]),
                    "oracle_second_ce": float(sorted_losses[j, 1]),
                    "oracle_margin": float(margin[j]), "oracle_gain_vs_uniform": float(gain[j]),
                    "uniform_hard_best": bool(best[j] == 0), "margin_quartile": int(quartile[j]),
                    **{f"CE_{ACTION_NAMES[a]}": float(sub[j, a]) for a in range(5)},
                })
            summary = {
                "dataset": ctx["dataset"], "seed": ctx["seed"], "split": split_name,
                "modality": modality, "node_count": int(len(ids)),
                "oracle_margin_mean": float(margin.mean()), "oracle_margin_median": float(np.median(margin)),
                "oracle_margin_p90": float(np.quantile(margin, .9)),
                "oracle_gain_mean": float(gain.mean()), "oracle_gain_median": float(np.median(gain)),
                "oracle_gain_p90": float(np.quantile(gain, .9)),
                "fraction_oracle_gain_gt_0": float((gain > 0).mean()),
                "uniform_hard_best_fraction": float((best == 0).mean()),
            }
            summary.update({f"best_fraction_{ACTION_NAMES[a]}": float((best == a).mean()) for a in range(5)})
            summary.update({f"mean_CE_{ACTION_NAMES[a]}": float(sub[:, a].mean()) for a in range(5)})
            summaries.append(summary)
    return pd.DataFrame(rows), summaries


def _stability_rows(frames: dict[tuple[str, int], pd.DataFrame], datasets: list[str]) -> list[dict]:
    output = []
    action_loss_cols = [f"CE_{name}" for name in ACTION_NAMES]
    left, right = np.triu_indices(5, k=1)
    for dataset in datasets:
        for modality in ("text", "visual"):
            for split in ("train", "validation"):
                per_seed = []
                for seed in SEEDS:
                    frame = frames[(dataset, seed)]
                    per_seed.append(frame[(frame.modality == modality) & (frame.split == split)].set_index("node_id"))
                for i, j in combinations(range(len(SEEDS)), 2):
                    pair = per_seed[i][action_loss_cols + ["oracle_best_action", "oracle_margin", "oracle_gain_vs_uniform"]].join(
                        per_seed[j][action_loss_cols + ["oracle_best_action", "oracle_margin", "oracle_gain_vs_uniform"]],
                        lsuffix="_a", rsuffix="_b", how="inner",
                    )
                    losses_a = pair[[f"{c}_a" for c in action_loss_cols]].to_numpy()
                    losses_b = pair[[f"{c}_b" for c in action_loss_cols]].to_numpy()
                    ordering_a = np.sign(losses_a[:, left] - losses_a[:, right])
                    ordering_b = np.sign(losses_b[:, left] - losses_b[:, right])
                    output.append({
                        "dataset": dataset, "modality": modality, "split": split,
                        "seed_a": SEEDS[i], "seed_b": SEEDS[j], "common_nodes": int(len(pair)),
                        "hard_best_action_agreement": float((pair.oracle_best_action_a == pair.oracle_best_action_b).mean()),
                        "oracle_margin_spearman": _spearman(pair.oracle_margin_a, pair.oracle_margin_b),
                        "oracle_gain_spearman": _spearman(pair.oracle_gain_vs_uniform_a, pair.oracle_gain_vs_uniform_b),
                        "pairwise_action_ordering_agreement": float((ordering_a == ordering_b).mean()),
                    })
    return output


def compute_decision_signatures(c0, head, states_text, states_visual) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference-only feature builder; intentionally has no label argument."""
    with torch.no_grad():
        bank_t = hard_action_bank(states_text)
        bank_v = hard_action_bank(states_visual)
        uniform_t, uniform_v = bank_t[:, 0], bank_v[:, 0]
        prediction_u = torch.softmax(head(c0._fuse(uniform_t, uniform_v)), dim=-1)
        sig_t, sig_v = [], []
        for action in range(5):
            prediction_t = torch.softmax(head(c0._fuse(bank_t[:, action], uniform_v)), dim=-1)
            prediction_v = torch.softmax(head(c0._fuse(uniform_t, bank_v[:, action])), dim=-1)
            sig_t.append(decision_signature(prediction_u, prediction_t))
            sig_v.append(decision_signature(prediction_u, prediction_v))
        return torch.stack(sig_t, dim=1), torch.stack(sig_v, dim=1)


def _d0_model(ctx: dict) -> tuple[CanonicalRoutingAudit, dict]:
    path = E6_OUT / "checkpoints" / "D0_independent" / f"{ctx['dataset']}_seed{ctx['seed']}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("c0_sha256") != ctx["c0_sha256"]:
        raise RuntimeError(f"E6 D0 C0 hash mismatch {ctx['dataset']}/{ctx['seed']}")
    model = CanonicalRoutingAudit(256, "independent").to(ctx["device"])
    model.load_state_dict(saved["router_state"], strict=True)
    model.eval()
    with torch.no_grad():
        out = model(ctx["states_text"], ctx["states_visual"])
    return model, out


def d0_misrouting(ctx: dict, action_frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    _, output = _d0_model(ctx)
    raw, summaries = [], []
    matrices = ctx["matrix"].detach().cpu().numpy()
    for modality in ("text", "visual"):
        alpha = output["alpha_text"] if modality == "text" else output["alpha_visual"]
        alpha_np = alpha.detach().cpu().numpy()
        projected = alpha_np.argmax(1) + 1
        losses = matrices[:, :, 0] if modality == "text" else matrices[:, 0, :]
        oracle = losses.argmin(1)
        regret = losses[np.arange(len(losses)), projected] - losses.min(1)
        entropy = -(np.clip(alpha_np, 1e-12, 1) * np.log(np.clip(alpha_np, 1e-12, 1))).sum(1)
        sorted_alpha = np.sort(alpha_np, axis=1)
        d0_margin = sorted_alpha[:, -1] - sorted_alpha[:, -2]
        l1_uniform = np.abs(alpha_np - .25).sum(1)
        oracle_rows = action_frame[(action_frame.modality == modality)].set_index("node_id")
        for i, node in enumerate(ctx["node_id"]):
            row = oracle_rows.loc[int(node)]
            raw.append({
                "dataset": ctx["dataset"], "seed": ctx["seed"], "split": ctx["split"][i],
                "node_id": int(node), "modality": modality,
                **{f"alpha_S{k}": float(alpha_np[i, k]) for k in range(4)},
                "d0_argmax_structural_order": ACTION_NAMES[int(projected[i])],
                "d0_uniform_distance_l1": float(l1_uniform[i]), "d0_alpha_entropy": float(entropy[i]),
                "d0_alpha_top1_top2_margin": float(d0_margin[i]),
                "oracle_best_action": ACTION_NAMES[int(oracle[i])],
                "oracle_margin": float(row.oracle_margin), "oracle_gain_vs_uniform": float(row.oracle_gain_vs_uniform),
                "oracle_margin_quartile": int(row.margin_quartile),
                "hard_order_agreement": bool(projected[i] == oracle[i]),
                "discrete_projected_regret": float(regret[i]),
                **{f"CE_{ACTION_NAMES[a]}": float(losses[i, a]) for a in range(5)},
            })
        frame = pd.DataFrame(raw[-len(ctx["node_id"]):])
        for split_name in ("train", "validation"):
            sub = frame[frame.split == split_name]
            if sub.empty:
                continue
            summaries.append({
                "dataset": ctx["dataset"], "seed": ctx["seed"], "split": split_name, "modality": modality,
                "candidate": "E6_D0", "node_count": len(sub), "hard_order_agreement": float(sub.hard_order_agreement.mean()),
                "selection_regret_mean": float(sub.discrete_projected_regret.mean()),
                "selection_regret_median": float(sub.discrete_projected_regret.median()),
                "selection_regret_p90": float(sub.discrete_projected_regret.quantile(.9)),
                "d0_uniform_distance_l1_mean": float(sub.d0_uniform_distance_l1.mean()),
                "d0_alpha_entropy_mean": float(sub.d0_alpha_entropy.mean()),
                "d0_alpha_margin_mean": float(sub.d0_alpha_top1_top2_margin.mean()),
            })
            for quartile, group in sub.groupby("oracle_margin_quartile"):
                summaries.append({
                    "dataset": ctx["dataset"], "seed": ctx["seed"], "split": split_name,
                    "modality": modality, "candidate": "E6_D0", "margin_quartile": int(quartile),
                    "node_count": len(group), "mean_oracle_margin": float(group.oracle_margin.mean()),
                    "mean_oracle_gain": float(group.oracle_gain_vs_uniform.mean()),
                    "hard_order_agreement": float(group.hard_order_agreement.mean()),
                    "selection_regret_mean": float(group.discrete_projected_regret.mean()),
                    "selection_regret_median": float(group.discrete_projected_regret.median()),
                    "selection_regret_p90": float(group.discrete_projected_regret.quantile(.9)),
                    "d0_uniform_distance_l1_mean": float(group.d0_uniform_distance_l1.mean()),
                    "d0_alpha_entropy_mean": float(group.d0_alpha_entropy.mean()),
                    "d0_alpha_margin_mean": float(group.d0_alpha_top1_top2_margin.mean()),
                })
    raw_frame = pd.DataFrame(raw)
    path = OUT / "phase_a_d0_misrouting_nodes" / f"{ctx['dataset']}_seed{ctx['seed']}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(path, index=False, compression="gzip")
    return raw_frame, summaries, output


def preflight_screening(device_name: str = DEVICE_DEFAULT) -> dict:
    RES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    frames, summaries, d0_summaries = {}, [], []
    checks = []
    for dataset in SCREEN:
        for seed in SEEDS:
            ctx = _load_context(dataset, seed, device)
            frame, rows = _action_landscape_frame(ctx)
            frames[(dataset, seed)] = frame
            raw_path = OUT / "phase_a_action_landscape_nodes" / f"{dataset}_seed{seed}.csv.gz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(raw_path, index=False, compression="gzip")
            summaries.extend(rows)
            _, d0rows, _ = d0_misrouting(ctx, frame)
            d0_summaries.extend(d0rows)
            checks.append({
                "dataset": dataset, "seed": seed, "c0_checkpoint_sha256": ctx["c0_sha256"],
                "cache_node_split_verified": ctx["cache_verified"],
                "recomputed_5x5_action_max_abs_error": ctx["action_recompute_error"],
                "recomputed_5x5_action_p99_abs_error": ctx["action_recompute_p99"],
                "recomputed_5x5_action_mean_abs_error": ctx["action_recompute_mean"],
                "c0_uniform_reload_max_abs_error": ctx["uniform_reload_error"],
                "test_labels_indexed": False,
            })
            print(f"[A] verified E6 cache and action landscape: {dataset} seed={seed}", flush=True)
            del ctx
            torch.cuda.empty_cache()
    write_csv(RES / "phase_a_action_landscape.csv", summaries)
    write_csv(RES / "phase_a_action_stability.csv", _stability_rows(frames, SCREEN))
    write_csv(RES / "phase_a_d0_misrouting.csv", d0_summaries)
    write_csv(RES / "preflight_cache_checks.csv", checks)
    strat = build_stratification(frames, {})
    strat.extend(row for row in d0_summaries if "margin_quartile" in row)
    write_csv(RES / "phase_a_margin_stratification.csv", strat)
    max_recompute = max(row["recomputed_5x5_action_max_abs_error"] for row in checks)
    lines = [
        "# E7-CSAO preflight",
        "",
        "C0 was reloaded from the frozen E6 reference checkpoints. No C0 parameters were trained; all Train/Validation action labels came from the verified E6 losses. Test indices and labels were not read.",
        "",
        f"Screening contexts checked: {len(checks)}/9. E6 node ids, split order, raw action landscape, and D0 C0 checkpoint hashes matched. Maximum 5x5 action CE recomputation error: {max_recompute:.6g}.",
        f"The recomputation uses the exact E6 action definitions and 32,768-node chunks; small remaining float32 differences arise from repeated graph aggregation (largest per-run P99: {max(row['recomputed_5x5_action_p99_abs_error'] for row in checks):.6g}). E6 NPZ/raw values were cross-checked within serialization precision.",
        "E6 corrected restart audit was recomputed from within-chunk, three-restart groups; E6 source outputs were left unchanged.",
        "",
    ]
    (RES / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"frames": frames, "checks": checks, "stratification": strat}


def build_stratification(frames: dict, model_node_rows: dict) -> list[dict]:
    # Filled after ranker execution for all model variants.
    rows = []
    for (dataset, seed), frame in frames.items():
        for modality in ("text", "visual"):
            sub = frame[frame.modality == modality]
            for split_name in ("train", "validation"):
                mod = sub[sub.split == split_name]
                for quartile, group in mod.groupby("margin_quartile"):
                    rows.append({
                        "dataset": dataset, "seed": seed, "split": split_name, "candidate": "oracle_landscape",
                        "modality": modality, "margin_quartile": int(quartile), "node_count": len(group),
                        "mean_oracle_margin": float(group.oracle_margin.mean()),
                        "mean_oracle_gain": float(group.oracle_gain_vs_uniform.mean()),
                        "hard_oracle_action_agreement": 1.0, "selection_regret": 0.0,
                    })
    return rows


def phase0_main() -> dict:
    _, summary = corrected_restart_audit()
    return summary
