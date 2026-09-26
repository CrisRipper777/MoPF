from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_mag_data
from src.models.routing_audit import Model

OUTPUT = ROOT / "outputs/routing_audit_v1"
RESULTS = ROOT / "results/routing_audit_v1"
SCREEN_DATASETS = ["Movies", "ele-fashion", "Reddit-S"]
ALL_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
UTILITY_NAMES = ["G_T_given_VU", "G_V_given_TU"]


def _read_metric(payload: dict, name: str) -> float:
    value = payload.get("metrics", {}).get(name)
    if isinstance(value, dict):
        return float(value.get("mean", float("nan")))
    return float(value) if value is not None else float("nan")


def _metric_file(directory: Path) -> dict:
    path = directory / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run metrics: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir(phase: str, variant: str, dataset: str, seed: int) -> Path:
    return OUTPUT / "runs" / phase / variant / dataset / f"seed{seed}"


def checkpoint_path(phase: str, variant: str, dataset: str, seed: int) -> Path:
    return OUTPUT / "checkpoints" / phase / variant / f"{dataset}_seed{seed}.pt"


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _utility_metrics(frames: dict[int, pd.DataFrame], utility: str, split_consistent: bool):
    selected = {}
    for seed, frame in frames.items():
        values = frame.loc[frame["split"].isin(["train", "val"]), ["node_id", "split", utility]]
        values = values.drop_duplicates("node_id").set_index("node_id")
        selected[seed] = values
    common = set.intersection(*(set(value.index) for value in selected.values()))
    merged = None
    for seed in SEEDS:
        part = selected[seed].loc[sorted(common)].rename(
            columns={utility: f"u{seed}", "split": f"split{seed}"}
        )
        merged = part if merged is None else merged.join(part, how="inner")
    if merged is None:
        return [], 0
    if split_consistent:
        keep = (merged.split42 == merged.split43) & (merged.split43 == merged.split44)
        merged = merged.loc[keep]
    n = int(len(merged))
    if n == 0:
        return [], n
    records = []

    def add(metric, pair, value):
        records.append({"metric": metric, "seed_pair": pair, "value": float(value)})

    for left, right in ((42, 43), (42, 44), (43, 44)):
        x = merged[f"u{left}"].to_numpy(dtype=np.float64)
        y = merged[f"u{right}"].to_numpy(dtype=np.float64)
        add("pairwise_spearman", f"{left}-{right}", _spearman(x, y))
        add("sign_agreement", f"{left}-{right}", np.mean(np.sign(x) == np.sign(y)))
        k = max(1, int(math.ceil(0.10 * n)))
        for name, order_x, order_y in (
            ("top10_jaccard", np.argsort(-x)[:k], np.argsort(-y)[:k]),
            ("bottom10_jaccard", np.argsort(x)[:k], np.argsort(y)[:k]),
        ):
            a, b = set(order_x.tolist()), set(order_y.tolist())
            add(name, f"{left}-{right}", len(a & b) / max(len(a | b), 1))
    signs = np.sign(merged[["u42", "u43", "u44"]].to_numpy(dtype=np.float64))
    add("three_seed_unanimous_sign_fraction", "42-43-44", np.mean((signs == signs[:, :1]).all(axis=1)))
    node_mean = merged[["u42", "u43", "u44"]].mean(axis=1).to_numpy(dtype=np.float64)
    node_var = merged[["u42", "u43", "u44"]].var(axis=1, ddof=0).to_numpy(dtype=np.float64)
    between = float(np.var(node_mean, ddof=0))
    within = float(np.mean(node_var))
    add("between_node_variance", "42-43-44", between)
    add("within_node_across_seed_variance", "42-43-44", within)
    add("between_within_variance_ratio", "42-43-44", between / within if within > 0 else float("inf"))
    return records, n


def run_a0_utility() -> None:
    source = OUTPUT.parent / "cmrf_discovery_v1" / "node_utility"
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not source.is_dir() or not list(source.glob("*.csv.gz")):
        pd.DataFrame([{"status": "SKIPPED_MISSING_RAW"}]).to_csv(
            RESULTS / "a0_utility_stability.csv", index=False
        )
        (RESULTS / "a0_utility_stability_report.md").write_text(
            "# A0 utility cross-seed stability\n\nSKIPPED_MISSING_RAW: E3 raw tables are absent. No E3 retraining was started.\n",
            encoding="utf-8",
        )
        return
    rows, report_lines = [], []
    for dataset in ALL_DATASETS:
        frames = {}
        for seed in SEEDS:
            path = source / f"{dataset}_seed{seed}.csv.gz"
            if path.is_file():
                frames[seed] = pd.read_csv(path, usecols=["node_id", "split", *UTILITY_NAMES])
        if len(frames) != 3:
            report_lines.append(f"- {dataset}: one or more seed tables are missing.")
            continue
        for utility in UTILITY_NAMES:
            by_scope = {}
            for scope, same_split in (
                ("common_train_val", False),
                ("same_split_intersection", True),
            ):
                metrics, n = _utility_metrics(frames, utility, same_split)
                sufficient = (not same_split) or n >= 100
                by_scope[scope] = n
                for metric in metrics:
                    rows.append({
                        "dataset": dataset,
                        "utility": utility,
                        "scope": scope,
                        "n_nodes": n,
                        "sufficient_for_summary": sufficient,
                        **metric,
                        "value": metric["value"] if sufficient else float("nan"),
                    })
            subset = [
                row for row in rows
                if row["dataset"] == dataset and row["utility"] == utility
                and row["scope"] == "common_train_val"
            ]

            def average(metric):
                values = [r["value"] for r in subset if r["metric"] == metric and np.isfinite(r["value"])]
                return float(np.mean(values)) if values else float("nan")

            unanimous = average("three_seed_unanimous_sign_fraction")
            ratio = average("between_within_variance_ratio")
            report_lines.append(
                f"- {dataset} / {utility}: common Train+Validation n={by_scope['common_train_val']}; "
                f"mean pairwise Spearman={average('pairwise_spearman'):.3f}, "
                f"sign agreement={average('sign_agreement'):.3f}, "
                f"unanimous sign={unanimous:.3f}, "
                f"top-10% Jaccard={average('top10_jaccard'):.3f}, "
                f"bottom-10% Jaccard={average('bottom10_jaccard'):.3f}, "
                f"between/within variance={ratio:.3f}; "
                f"same-split n={by_scope['same_split_intersection']}"
                f"{' (insufficient)' if by_scope['same_split_intersection'] < 100 else ''}."
            )
    pd.DataFrame(rows).to_csv(RESULTS / "a0_utility_stability.csv", index=False)
    report = (
        "# A0 utility cross-seed stability\n\n"
        "E3 node tables were reused directly. Only rows marked train or val were included; "
        "no E3 retraining and no test labels or metrics were used. Same-split summaries "
        "require at least 100 matched nodes.\n\n## Results\n\n"
        + "\n".join(report_lines)
        + "\n"
    )
    (RESULTS / "a0_utility_stability_report.md").write_text(report, encoding="utf-8")


def _compose_cfg(dataset: str, seed: int, variant: str):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}", "task=nc", "model=routing_audit",
                f"seed={seed}", "num_runs=1", "task.evaluate_test=false",
                f"model.routing_variant={variant}",
            ],
        )


def _load_model_data(dataset, seed, variant, checkpoint, device):
    cfg = _compose_cfg(dataset, seed, variant)
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = Model(cfg, info).to(device).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device).eval()
    classifier.load_state_dict(payload["head_state"], strict=True)
    return (
        data, model, classifier, data.x.to(device), data.edge_index.to(device),
        data.train_idx.to(device), data.val_idx.to(device), data.y.to(device),
    )


def _validation_metrics(route, classifier, y, val_idx, labels):
    pred = classifier(route["fused_z"][val_idx]).argmax(dim=-1)
    target = y[val_idx]
    acc = float((pred == target).float().mean().item())
    f1 = float(f1_score(
        target.detach().cpu().numpy(), pred.detach().cpu().numpy(),
        labels=labels, average="macro", zero_division=0,
    ))
    return acc, f1, pred


def _alpha_change(base, changed, val_idx):
    return float(torch.stack([
        (base["alpha_text"][val_idx] - changed["alpha_text"][val_idx]).abs().mean(),
        (base["alpha_visual"][val_idx] - changed["alpha_visual"][val_idx]).abs().mean(),
    ]).mean().item())


def _intervention_row(dataset, seed, variant, name, normal, changed, classifier,
                      y, val_idx, labels, normal_pred, base, shuffle_seed=None):
    acc, f1, pred = _validation_metrics(changed, classifier, y, val_idx, labels)
    return {
        "dataset": dataset, "seed": seed, "variant": variant,
        "intervention": name, "shuffle_seed": shuffle_seed,
        "val_acc": acc, "val_macro_f1": f1,
        "acc_drop": normal[0] - acc, "macro_f1_drop": normal[1] - f1,
        "prediction_flip_rate": float((pred != normal_pred).float().mean().item()),
        "mean_abs_alpha_change": _alpha_change(base, changed, val_idx),
    }


@torch.no_grad()
def analyze_intervention(phase, variant, dataset, seed, checkpoint, device, kind):
    data, model, classifier, x, edge, train_idx, val_idx, y = _load_model_data(
        dataset, seed, variant, checkpoint, device
    )
    analysis = model.analyze(x, edge)
    rho_text_normal = analysis.get("rho_text")
    rho_visual_normal = analysis.get("rho_visual")
    if rho_text_normal is not None:
        rho_text_normal = rho_text_normal.clone()
        rho_visual_normal = rho_visual_normal.clone()
    labels = sorted({
        int(value) for value in y[torch.cat([train_idx, val_idx])].detach().cpu().tolist()
        if 0 <= int(value) < int(data.num_classes)
    })
    normal = _validation_metrics(analysis, classifier, y, val_idx, labels)
    rows = []
    if kind == "global":
        zero = torch.zeros(4, device=device, dtype=x.dtype)
        changed = model.reroute(analysis, global_text_override=zero, global_visual_override=zero)
        rows.append(_intervention_row(
            dataset, seed, variant, "zero_global_prior", normal, changed,
            classifier, y, val_idx, labels, normal[2], analysis,
        ))
    elif kind == "node":
        mean_t = analysis["delta_text"][train_idx].mean(0)
        mean_v = analysis["delta_visual"][train_idx].mean(0)
        delta_t, delta_v = analysis["delta_text"].clone(), analysis["delta_visual"].clone()
        delta_t[val_idx], delta_v[val_idx] = mean_t, mean_v
        changed = model.reroute(analysis, delta_text_override=delta_t, delta_visual_override=delta_v)
        rows.append(_intervention_row(
            dataset, seed, variant, "train_mean_delta", normal, changed,
            classifier, y, val_idx, labels, normal[2], analysis,
        ))
        for index in range(10):
            generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + index)
            permutation = torch.randperm(int(val_idx.numel()), generator=generator).to(device)
            delta_t, delta_v = analysis["delta_text"].clone(), analysis["delta_visual"].clone()
            delta_t[val_idx] = analysis["delta_text"][val_idx[permutation]]
            delta_v[val_idx] = analysis["delta_visual"][val_idx[permutation]]
            changed = model.reroute(analysis, delta_text_override=delta_t, delta_visual_override=delta_v)
            rows.append(_intervention_row(
                dataset, seed, variant, "shuffle_delta", normal, changed,
                classifier, y, val_idx, labels, normal[2], analysis, index,
            ))
    elif kind == "relation":
        mean_t = analysis["relation_text"][train_idx].mean(0)
        mean_v = analysis["relation_visual"][train_idx].mean(0)
        rel_t, rel_v = analysis["relation_text"].clone(), analysis["relation_visual"].clone()
        rel_t[val_idx], rel_v[val_idx] = mean_t, mean_v
        changed = model.reroute(
            analysis, relation_text_override=rel_t, relation_visual_override=rel_v
        )
        rows.append(_intervention_row(
            dataset, seed, variant, "train_mean_relation_context", normal, changed,
            classifier, y, val_idx, labels, normal[2], analysis,
        ))
        for index in range(10):
            generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + index)
            permutation = torch.randperm(int(val_idx.numel()), generator=generator).to(device)
            rel_t, rel_v = analysis["relation_text"].clone(), analysis["relation_visual"].clone()
            rel_t[val_idx] = analysis["relation_text"][val_idx[permutation]]
            rel_v[val_idx] = analysis["relation_visual"][val_idx[permutation]]
            changed = model.reroute(
                analysis, relation_text_override=rel_t, relation_visual_override=rel_v
            )
            rows.append(_intervention_row(
                dataset, seed, variant, "shuffle_relation_context", normal, changed,
                classifier, y, val_idx, labels, normal[2], analysis, index,
            ))
    elif kind == "joint":
        for index in range(10):
            generator = torch.Generator(device="cpu").manual_seed(seed * 1000 + index)
            permutation = torch.randperm(int(val_idx.numel()), generator=generator).to(device)
            joint = analysis["joint_relation"].clone()
            joint[val_idx] = analysis["joint_relation"][val_idx[permutation]]
            changed = model.reroute(analysis, joint_relation_override=joint)
            rows.append(_intervention_row(
                dataset, seed, variant, "shuffle_joint_relation_context", normal, changed,
                classifier, y, val_idx, labels, normal[2], analysis, index,
            ))
    else:
        raise ValueError(kind)

    trval = torch.cat([train_idx, val_idx])
    alpha_t, alpha_v = analysis["alpha_text"], analysis["alpha_visual"]
    entropy_t = -(alpha_t.clamp_min(1e-12) * alpha_t.clamp_min(1e-12).log()).sum(-1)
    entropy_v = -(alpha_v.clamp_min(1e-12) * alpha_v.clamp_min(1e-12).log()).sum(-1)
    summary = {
        "dataset": dataset, "seed": seed, "variant": variant,
        "normal_val_acc": normal[0], "normal_val_macro_f1": normal[1],
        "global_text_logits": None if model.global_text_logits is None else model.global_text_logits.detach().cpu().tolist(),
        "global_visual_logits": None if model.global_visual_logits is None else model.global_visual_logits.detach().cpu().tolist(),
        "alpha_mean_text_train": alpha_t[train_idx].mean(0).cpu().tolist(),
        "alpha_mean_visual_train": alpha_v[train_idx].mean(0).cpu().tolist(),
        "alpha_mean_text_val": alpha_t[val_idx].mean(0).cpu().tolist(),
        "alpha_mean_visual_val": alpha_v[val_idx].mean(0).cpu().tolist(),
        "entropy_mean_text_train": float(entropy_t[train_idx].mean().item()),
        "entropy_mean_visual_train": float(entropy_v[train_idx].mean().item()),
        "entropy_mean_text_val": float(entropy_t[val_idx].mean().item()),
        "entropy_mean_visual_val": float(entropy_v[val_idx].mean().item()),
        "per_node_variance_text": float(alpha_t[trval].var(0, unbiased=False).mean().item()),
        "per_node_variance_visual": float(alpha_v[trval].var(0, unbiased=False).mean().item()),
    }
    if model.variant == "cross_relation":
        summary["rho_text_mean_val"] = float(rho_text_normal[val_idx].mean().item())
        summary["rho_text_std_val"] = float(rho_text_normal[val_idx].std(unbiased=False).item())
        summary["rho_visual_mean_val"] = float(rho_visual_normal[val_idx].mean().item())
        summary["rho_visual_std_val"] = float(rho_visual_normal[val_idx].std(unbiased=False).item())
    raw_dir = OUTPUT / "routing_stats" / phase / variant / dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    ids = trval.detach().cpu().tolist()
    raw_rows = []
    for modality, alpha in (("text", alpha_t), ("visual", alpha_v)):
        values = alpha[trval].detach().cpu()
        entropy = -(values.clamp_min(1e-12) * values.clamp_min(1e-12).log()).sum(-1)
        for i, node_id in enumerate(ids):
            raw_rows.append({
                "node_id": node_id, "modality": modality,
                "alpha0": float(values[i, 0]), "alpha1": float(values[i, 1]),
                "alpha2": float(values[i, 2]), "alpha3": float(values[i, 3]),
                "entropy": float(entropy[i]),
                "node_alpha_component_variance": float(values[i].var(unbiased=False)),
            })
    pd.DataFrame(raw_rows).to_csv(
        raw_dir / f"seed{seed}.csv.gz", index=False, compression="gzip"
    )
    (raw_dir / f"seed{seed}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return rows, summary


def _load_result_row(phase, variant, dataset, seed, label=None):
    run = _metric_file(run_dir(phase, variant, dataset, seed))
    ckpt = checkpoint_path(phase, variant, dataset, seed)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt.is_file() else {}
    return {
        "variant": label or variant, "model_variant": variant, "phase": phase,
        "dataset": dataset, "seed": seed,
        "val_acc": _read_metric(run, "val_acc"),
        "val_macro_f1": _read_metric(run, "val_macro_f1"),
        "best_epoch": run.get("best_epoch", payload.get("epoch")),
        "trainable_parameter_count": sum(v.numel() for v in payload.get("model_state", {}).values())
            + sum(v.numel() for v in payload.get("head_state", {}).values()),
    }


def make_phase_a_results():
    rows = []
    ref = OUTPUT.parent / "cmrf_discovery_v1"
    for dataset in SCREEN_DATASETS:
        for seed in SEEDS:
            payload = _metric_file(ref / "runs" / "uniform" / dataset / f"seed{seed}")
            ckpt = torch.load(
                ref / "reference_uniform" / "checkpoints" / f"{dataset}_seed{seed}.pt",
                map_location="cpu", weights_only=False,
            )
            rows.append({
                "variant": "A0_uniform", "model_variant": "uniform",
                "phase": "reused_e3_c0", "dataset": dataset, "seed": seed,
                "val_acc": _read_metric(payload, "val_acc"),
                "val_macro_f1": _read_metric(payload, "val_macro_f1"),
                "best_epoch": ckpt.get("epoch"),
                "trainable_parameter_count": sum(v.numel() for v in ckpt["model_state"].values())
                    + sum(v.numel() for v in ckpt["head_state"].values()),
                "checkpoint_reused": True,
            })
    aliases = {
        "global_simplex": "A1_global_simplex",
        "node_flat_simplex": "A2_node_flat_simplex",
        "hierarchical_flat_simplex": "A3_hierarchical_flat_simplex",
    }
    for variant, label in aliases.items():
        for dataset in SCREEN_DATASETS:
            for seed in SEEDS:
                row = _load_result_row("phase_a", variant, dataset, seed, label)
                row["checkpoint_reused"] = False
                rows.append(row)
    frame = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "phase_a_results.csv", index=False)
    comparisons = [
        ("A1_global_simplex", "A0_uniform"),
        ("A2_node_flat_simplex", "A0_uniform"),
        ("A3_hierarchical_flat_simplex", "A0_uniform"),
        ("A3_hierarchical_flat_simplex", "A1_global_simplex"),
        ("A3_hierarchical_flat_simplex", "A2_node_flat_simplex"),
    ]
    out = []
    for candidate, parent in comparisons:
        joined = frame[frame.variant == candidate].merge(
            frame[frame.variant == parent], on=["dataset", "seed"], suffixes=("_a", "_b")
        )
        joined["delta_acc"] = joined.val_acc_a - joined.val_acc_b
        joined["delta_macro_f1"] = joined.val_macro_f1_a - joined.val_macro_f1_b
        for dataset, part in joined.groupby("dataset"):
            out.append({
                "candidate": candidate, "parent": parent, "dataset": dataset,
                "paired_seeds": len(part), "mean_delta_acc": part.delta_acc.mean(),
                "mean_delta_macro_f1": part.delta_macro_f1.mean(),
                "positive_acc_pairs": int((part.delta_acc > 0).sum()),
                "positive_f1_pairs": int((part.delta_macro_f1 > 0).sum()),
            })
        out.append({
            "candidate": candidate, "parent": parent, "dataset": "ALL_SCREEN",
            "paired_seeds": len(joined), "mean_delta_acc": joined.delta_acc.mean(),
            "mean_delta_macro_f1": joined.delta_macro_f1.mean(),
            "positive_acc_pairs": int((joined.delta_acc > 0).sum()),
            "positive_f1_pairs": int((joined.delta_macro_f1 > 0).sum()),
        })
    pd.DataFrame(out).to_csv(RESULTS / "phase_a_analysis.csv", index=False)
    return frame


def run_phase_a_interventions(device):
    rows, summaries = [], []
    for variant, kind in (
        ("global_simplex", "global"),
        ("node_flat_simplex", "node"),
        ("hierarchical_flat_simplex", "node"),
    ):
        for dataset in SCREEN_DATASETS:
            for seed in SEEDS:
                recs, summary = analyze_intervention(
                    "phase_a", variant, dataset, seed,
                    checkpoint_path("phase_a", variant, dataset, seed), device, kind,
                )
                rows.extend(recs)
                summaries.append(summary)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "phase_a_intervention.csv", index=False)
    (OUTPUT / "phase_a_routing_summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return frame


def make_phase_b_results():
    rows = []
    for dataset in SCREEN_DATASETS:
        for seed in SEEDS:
            rows.append(_load_result_row(
                "phase_a", "hierarchical_flat_simplex", dataset, seed,
                "B0_hierarchical_flat",
            ))
    for variant, label in (
        ("trajectory", "B1_trajectory"),
        ("relation", "B2_relation"),
        ("capacity_control", "B2_capacity_control"),
    ):
        for dataset in SCREEN_DATASETS:
            for seed in SEEDS:
                rows.append(_load_result_row("phase_b", variant, dataset, seed, label))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "phase_b_results.csv", index=False)
    out = []
    for candidate, parent in (
        ("B1_trajectory", "B0_hierarchical_flat"),
        ("B2_capacity_control", "B1_trajectory"),
        ("B2_relation", "B2_capacity_control"),
        ("B2_relation", "B1_trajectory"),
    ):
        joined = frame[frame.variant == candidate].merge(
            frame[frame.variant == parent], on=["dataset", "seed"], suffixes=("_a", "_b")
        )
        joined["delta_acc"] = joined.val_acc_a - joined.val_acc_b
        joined["delta_macro_f1"] = joined.val_macro_f1_a - joined.val_macro_f1_b
        for dataset, part in joined.groupby("dataset"):
            out.append({
                "candidate": candidate, "parent": parent, "dataset": dataset,
                "paired_seeds": len(part), "mean_delta_acc": part.delta_acc.mean(),
                "mean_delta_macro_f1": part.delta_macro_f1.mean(),
                "positive_acc_pairs": int((part.delta_acc > 0).sum()),
                "positive_f1_pairs": int((part.delta_macro_f1 > 0).sum()),
                "candidate_parameters": float(part.trainable_parameter_count_a.mean()),
                "parent_parameters": float(part.trainable_parameter_count_b.mean()),
                "parameter_diff_pct": float(
                    100.0 * (part.trainable_parameter_count_a.mean() - part.trainable_parameter_count_b.mean())
                    / max(part.trainable_parameter_count_b.mean(), 1.0)
                ),
            })
        out.append({
            "candidate": candidate, "parent": parent, "dataset": "ALL_SCREEN",
            "paired_seeds": len(joined), "mean_delta_acc": joined.delta_acc.mean(),
            "mean_delta_macro_f1": joined.delta_macro_f1.mean(),
            "positive_acc_pairs": int((joined.delta_acc > 0).sum()),
            "positive_f1_pairs": int((joined.delta_macro_f1 > 0).sum()),
            "candidate_parameters": float(joined.trainable_parameter_count_a.mean()),
            "parent_parameters": float(joined.trainable_parameter_count_b.mean()),
            "parameter_diff_pct": float(
                100.0 * (joined.trainable_parameter_count_a.mean() - joined.trainable_parameter_count_b.mean())
                / max(joined.trainable_parameter_count_b.mean(), 1.0)
            ),
        })
    pd.DataFrame(out).to_csv(RESULTS / "phase_b_capacity_control.csv", index=False)
    return frame


def run_phase_b_interventions(device):
    rows, summaries = [], []
    for variant, kind in (("trajectory", "node"), ("relation", "relation")):
        for dataset in SCREEN_DATASETS:
            for seed in SEEDS:
                recs, summary = analyze_intervention(
                    "phase_b", variant, dataset, seed,
                    checkpoint_path("phase_b", variant, dataset, seed), device, kind,
                )
                rows.extend(recs)
                summaries.append(summary)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "phase_b_intervention.csv", index=False)
    (OUTPUT / "phase_b_routing_summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return frame


def _comparison(frame, candidate, parent):
    left = frame[frame.variant == candidate]
    right = frame[frame.variant == parent]
    out = left.merge(right, on=["dataset", "seed"], suffixes=("_a", "_b"))
    out["delta_acc"] = out.val_acc_a - out.val_acc_b
    out["delta_f1"] = out.val_macro_f1_a - out.val_macro_f1_b
    return out


def _functional(interventions, variant, kind):
    subset = interventions[
        (interventions.variant == variant) & (interventions.intervention == kind)
    ]

    def reduce(frame):
        if frame.empty:
            return {"acc_drop": float("nan"), "macro_f1_drop": float("nan"),
                    "mean_abs_alpha_change": float("nan")}
        if frame.intervention.iloc[0].startswith("shuffle_"):
            frame = frame.groupby(["dataset", "seed"], as_index=False).agg(
                acc_drop=("acc_drop", "mean"),
                macro_f1_drop=("macro_f1_drop", "mean"),
                mean_abs_alpha_change=("mean_abs_alpha_change", "mean"),
            )
        return {
            "acc_drop": float(frame.acc_drop.mean()),
            "macro_f1_drop": float(frame.macro_f1_drop.mean()),
            "mean_abs_alpha_change": float(frame.mean_abs_alpha_change.mean()),
        }

    if subset.empty:
        return {
            "functional": False, "mean_acc_drop": float("nan"),
            "mean_f1_drop": float("nan"), "mean_alpha_change": float("nan"),
            "mean_intervention": {},
        }
    main = reduce(subset)
    mean_effect = {}
    if kind == "shuffle_delta":
        mean_subset = interventions[
            (interventions.variant == variant)
            & (interventions.intervention == "train_mean_delta")
        ]
        mean_effect = reduce(mean_subset)
        alpha_effect = min(main["mean_abs_alpha_change"],
                           mean_effect.get("mean_abs_alpha_change", float("nan")))
        functional = bool(
            alpha_effect > 0.001
            and max(main["acc_drop"], main["macro_f1_drop"]) >= 0.0005
            and max(mean_effect.get("acc_drop", float("nan")),
                    mean_effect.get("macro_f1_drop", float("nan"))) >= 0.0005
        )
    else:
        alpha_effect = main["mean_abs_alpha_change"]
        functional = bool(
            alpha_effect > 0.001
            and max(main["acc_drop"], main["macro_f1_drop"]) >= 0.0005
        )
    return {
        "functional": functional,
        "mean_acc_drop": main["acc_drop"],
        "mean_f1_drop": main["macro_f1_drop"],
        "mean_alpha_change": main["mean_abs_alpha_change"],
        "mean_intervention": mean_effect,
    }


def _gate(frame, interventions, candidate, parent, intervention):
    pairs = _comparison(frame, candidate, parent)
    by_dataset = pairs.groupby("dataset").agg(
        acc=("delta_acc", "mean"), f1=("delta_f1", "mean")
    )
    positive_datasets = int((by_dataset.acc > 0).sum())
    positive_seeds = int((pairs.delta_acc > 0).sum())
    f1_mean = float(pairs.delta_f1.mean())
    damaged = int((by_dataset.f1 < -0.005).sum())
    no_f1_damage = f1_mean >= -0.003 and damaged <= 1
    model_variant = {
        "A1_global_simplex": "global_simplex",
        "A2_node_flat_simplex": "node_flat_simplex",
        "A3_hierarchical_flat_simplex": "hierarchical_flat_simplex",
        "B1_trajectory": "trajectory",
        "B2_relation": "relation",
        "C1_cross_relation": "cross_relation",
    }.get(candidate, candidate)
    iv = _functional(interventions, model_variant, intervention)
    return {
        "candidate": candidate, "parent": parent,
        "positive_datasets": positive_datasets,
        "positive_seed_pairs": positive_seeds,
        "mean_delta_acc": float(pairs.delta_acc.mean()),
        "mean_delta_macro_f1": f1_mean,
        "datasets_with_f1_drop_below_-0.005": damaged,
        "no_systematic_macro_f1_damage": bool(no_f1_damage),
        "intervention": iv,
        "candidate_gate_pass": bool(
            positive_datasets >= 2 and positive_seeds >= 6 and no_f1_damage and iv["functional"]
        ),
    }


def _decision_row(frame, hypothesis, candidate, parent, functional, note):
    pairs = _comparison(frame, candidate, parent)
    dataset_means = pairs.groupby("dataset").delta_acc.mean()
    f1_by_dataset = pairs.groupby("dataset").delta_f1.mean()
    supported = int((dataset_means > 0).sum())
    positive = int((pairs.delta_acc > 0).sum())
    mean_acc = float(pairs.delta_acc.mean())
    mean_f1 = float(pairs.delta_f1.mean())
    no_systematic_f1_damage = mean_f1 >= -0.003 and int((f1_by_dataset < -0.005).sum()) <= 1
    if supported >= 2 and positive >= 6 and functional and no_systematic_f1_damage:
        status = "STRONG_SUPPORT"
    elif supported >= 1 or positive >= 1 or functional:
        status = "MIXED_SUPPORT"
    else:
        status = "NO_SUPPORT"
    return {
        "hypothesis": hypothesis, "status": status,
        "datasets_supported": supported,
        "positive_paired_seeds": f"{positive}/{len(pairs)}",
        "mean_delta_val_acc": mean_acc, "mean_delta_macro_f1": mean_f1,
        "functional_intervention": bool(functional),
        "no_systematic_macro_f1_damage": bool(no_systematic_f1_damage),
        "evidence_note": note,
    }


def finalize_decision():
    a = pd.read_csv(RESULTS / "phase_a_results.csv")
    ai = pd.read_csv(RESULTS / "phase_a_intervention.csv")
    b = pd.read_csv(RESULTS / "phase_b_results.csv")
    bi = pd.read_csv(RESULTS / "phase_b_intervention.csv")
    candidates = [
        _gate(a, ai, "A1_global_simplex", "A0_uniform", "zero_global_prior"),
        _gate(a, ai, "A2_node_flat_simplex", "A0_uniform", "shuffle_delta"),
        _gate(a, ai, "A3_hierarchical_flat_simplex", "A1_global_simplex", "shuffle_delta"),
        _gate(b, bi, "B1_trajectory", "B0_hierarchical_flat", "shuffle_delta"),
        _gate(b, bi, "B2_relation", "B2_capacity_control", "shuffle_relation_context"),
    ]
    c_results_path = RESULTS / "phase_c_results.csv"
    c_interventions_path = RESULTS / "phase_c_intervention.csv"
    c_results = pd.read_csv(c_results_path) if c_results_path.is_file() else None
    c_interventions = pd.read_csv(c_interventions_path) if c_interventions_path.is_file() else None
    if c_results is not None and c_interventions is not None:
        candidates.append(_gate(
            pd.concat([b, c_results], ignore_index=True), c_interventions,
            "C1_cross_relation", "B2_relation", "shuffle_joint_relation_context",
        ))
    eligible = sorted(
        [item for item in candidates if item["candidate_gate_pass"]],
        key=lambda item: (item["mean_delta_acc"], item["mean_delta_macro_f1"]),
        reverse=True,
    )
    winner = eligible[0]["candidate"] if eligible else None

    b2_control = _comparison(b, "B2_relation", "B2_capacity_control")
    dataset_means = b2_control.groupby("dataset").delta_acc.mean()
    relation_gain = float(b2_control.delta_acc.mean())
    relation_positive_datasets = int((dataset_means > 0).sum())
    relation_positive_pairs = int((b2_control.delta_acc > 0).sum())
    relation_iv = _functional(bi, "relation", "shuffle_relation_context")
    phase_c_trigger = bool(
        relation_gain > 0 and relation_positive_datasets >= 2 and relation_positive_pairs >= 6
        and max(relation_iv["mean_acc_drop"], relation_iv["mean_f1_drop"])
        >= max(0.5 * abs(relation_gain), 0.0005)
    )
    global_iv = _functional(ai, "global_simplex", "zero_global_prior")
    node_iv = _functional(ai, "hierarchical_flat_simplex", "shuffle_delta")
    trajectory_iv = _functional(bi, "trajectory", "shuffle_delta")

    decision = [
        _decision_row(a, "H1_global_structural_preference", "A1_global_simplex",
                      "A0_uniform", global_iv["functional"], json.dumps(global_iv)),
        _decision_row(a, "H2_node_routing_beyond_global",
                      "A3_hierarchical_flat_simplex", "A1_global_simplex",
                      node_iv["functional"], json.dumps(node_iv)),
        _decision_row(b, "H3_trajectory_context_beyond_flat",
                      "B1_trajectory", "B0_hierarchical_flat",
                      trajectory_iv["functional"], json.dumps(trajectory_iv)),
        _decision_row(b, "H4_relation_context_beyond_trajectory_capacity",
                      "B2_relation", "B2_capacity_control", relation_iv["functional"],
                      json.dumps({
                          "vs_capacity_control_mean_acc": relation_gain,
                          "vs_B1_mean_acc": float(_comparison(b, "B2_relation", "B1_trajectory").delta_acc.mean()),
                          **relation_iv,
                      })),
    ]
    confirmation_analysis_path = RESULTS / "confirmation_analysis.csv"
    confirmation_analysis = (
        pd.read_csv(confirmation_analysis_path)
        if confirmation_analysis_path.is_file() else None
    )
    confirmation_summary = {}
    if confirmation_analysis is not None and not confirmation_analysis.empty:
        for variant, part in confirmation_analysis.groupby("variant"):
            dataset_means = part.groupby("dataset").delta_val_acc.mean()
            confirmation_summary[variant] = {
                "datasets_supported": int((dataset_means > 0).sum()),
                "positive_paired_seeds": int((part.delta_val_acc > 0).sum()),
                "paired_seed_count": int(len(part)),
                "mean_delta_val_acc": float(part.delta_val_acc.mean()),
                "mean_delta_macro_f1": float(part.delta_macro_f1.mean()),
                "dataset_mean_delta_val_acc": {
                    str(dataset): float(value)
                    for dataset, value in dataset_means.items()
                },
            }
        global_confirmation = confirmation_summary.get("global_simplex")
        if global_confirmation is not None:
            h1 = next(row for row in decision if row["hypothesis"] == "H1_global_structural_preference")
            if (global_confirmation["mean_delta_val_acc"] <= 0
                    or global_confirmation["datasets_supported"] < 3):
                h1["status"] = "MIXED_SUPPORT"
            h1["confirmation_datasets_supported"] = global_confirmation["datasets_supported"]
            h1["confirmation_positive_paired_seeds"] = (
                f"{global_confirmation['positive_paired_seeds']}/"
                f"{global_confirmation['paired_seed_count']}"
            )
            h1["confirmation_mean_delta_val_acc"] = global_confirmation["mean_delta_val_acc"]
            h1["confirmation_mean_delta_macro_f1"] = global_confirmation["mean_delta_macro_f1"]
            h1["evidence_note"] = json.dumps({
                "screening": json.loads(h1["evidence_note"]),
                "five_dataset_confirmation": global_confirmation,
            })
        hierarchy_confirmation = confirmation_summary.get("hierarchical_minus_global_simplex")
        if hierarchy_confirmation is not None:
            h2 = next(row for row in decision if row["hypothesis"] == "H2_node_routing_beyond_global")
            h2["confirmation_datasets_supported"] = hierarchy_confirmation["datasets_supported"]
            h2["confirmation_positive_paired_seeds"] = (
                f"{hierarchy_confirmation['positive_paired_seeds']}/"
                f"{hierarchy_confirmation['paired_seed_count']}"
            )
            h2["confirmation_mean_delta_val_acc"] = hierarchy_confirmation["mean_delta_val_acc"]
            h2["confirmation_mean_delta_macro_f1"] = hierarchy_confirmation["mean_delta_macro_f1"]
            h2["evidence_note"] = json.dumps({
                "screening": json.loads(h2["evidence_note"]),
                "five_dataset_confirmation": hierarchy_confirmation,
            })

    shuffle_rows = bi[
        (bi.variant == "relation") & (bi.intervention == "shuffle_relation_context")
    ]
    per_seed_shuffle = shuffle_rows.groupby(["dataset", "seed"]).acc_drop.mean()
    if relation_iv["functional"] and max(relation_iv["mean_acc_drop"], relation_iv["mean_f1_drop"]) >= 0.0005:
        h5_status = "STRONG_SUPPORT"
    elif max(relation_iv["mean_acc_drop"], relation_iv["mean_f1_drop"]) > 0:
        h5_status = "MIXED_SUPPORT"
    else:
        h5_status = "NO_SUPPORT"
    decision.append({
        "hypothesis": "H5_relation_context_functionally_used",
        "status": h5_status,
        "datasets_supported": int((shuffle_rows.groupby("dataset").acc_drop.mean() > 0).sum()),
        "positive_paired_seeds": f"{int((per_seed_shuffle > 0).sum())}/{len(per_seed_shuffle)}",
        "mean_delta_val_acc": -relation_iv["mean_acc_drop"],
        "mean_delta_macro_f1": -relation_iv["mean_f1_drop"],
        "functional_intervention": relation_iv["functional"],
        "evidence_note": json.dumps(relation_iv),
    })

    if c_results is not None and c_interventions is not None:
        c_pairs = _comparison(
            pd.concat([b, c_results], ignore_index=True),
            "C1_cross_relation", "B2_relation",
        )
        c_iv = _functional(c_interventions, "cross_relation", "shuffle_joint_relation_context")
        c_ds = c_pairs.groupby("dataset").delta_acc.mean()
        c_supported = int((c_ds > 0).sum())
        c_positive = int((c_pairs.delta_acc > 0).sum())
        c_gate = next((item for item in candidates if item["candidate"] == "C1_cross_relation"), None)
        h6_status = (
            "STRONG_SUPPORT" if c_gate and c_gate["candidate_gate_pass"]
            else ("MIXED_SUPPORT" if c_supported > 0 or c_iv["functional"] else "NO_SUPPORT")
        )
        decision.append({
            "hypothesis": "H6_paired_cross_modal_relation_context",
            "status": h6_status, "datasets_supported": c_supported,
            "positive_paired_seeds": f"{c_positive}/{len(c_pairs)}",
            "mean_delta_val_acc": float(c_pairs.delta_acc.mean()),
            "mean_delta_macro_f1": float(c_pairs.delta_f1.mean()),
            "functional_intervention": c_iv["functional"],
            "evidence_note": json.dumps({"gate": c_gate, "joint_shuffle": c_iv}),
        })
        phase_c_status = "C1_COMPLETE"
    else:
        decision.append({
            "hypothesis": "H6_paired_cross_modal_relation_context",
            "status": "NOT_YET_EVALUATED" if phase_c_trigger else "NO_SUPPORT",
            "datasets_supported": 0, "positive_paired_seeds": "0/0",
            "mean_delta_val_acc": float("nan"), "mean_delta_macro_f1": float("nan"),
            "functional_intervention": False,
            "evidence_note": "Phase C gate passed; awaiting C1." if phase_c_trigger
                else "OWN_RELATION_CONTEXT_NOT_SUPPORTED; the conditional gate failed, so this design was not tested.",
        })
        phase_c_status = "TRIGGERED" if phase_c_trigger else "OWN_RELATION_CONTEXT_NOT_SUPPORTED"

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decision).to_csv(RESULTS / "routing_audit_decision_matrix.csv", index=False)
    smoke_root = OUTPUT / "smoke"
    nonformal_smokes = len(list(smoke_root.rglob("metrics.json"))) if smoke_root.is_dir() else 0
    manifest_path = OUTPUT / "run_manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        trained = manifest.loc[manifest.state == "trained", ["phase", "variant", "dataset", "seed"]]
        trained_runs = int(trained.drop_duplicates().shape[0])
        failed_attempts = int((manifest.state == "failed").sum())
        latest = manifest.drop_duplicates(
            subset=["phase", "variant", "dataset", "seed"], keep="last"
        )
        failed_runs = int((latest.state == "failed").sum())
        trained_keys = set(map(tuple, manifest.loc[manifest.state == "trained", ["phase", "variant", "dataset", "seed"]].drop_duplicates().to_numpy()))
        failed_keys = manifest.loc[manifest.state == "failed", ["phase", "variant", "dataset", "seed"]].drop_duplicates()
        retried_failures = sum(tuple(row) in trained_keys for row in failed_keys.to_numpy())
        training_attempts = int(manifest.state.isin(["trained", "failed"]).sum())
        trained_by_phase = manifest.loc[manifest.state == "trained"].groupby("phase")[["variant", "dataset", "seed"]].apply(
            lambda part: int(part.drop_duplicates().shape[0])
        ).to_dict()
        reused_jobs = int((manifest.state == "reused_existing").sum())
    else:
        trained_runs, failed_runs, failed_attempts, retried_failures, training_attempts, trained_by_phase, reused_jobs = 0, 0, 0, 0, 0, {}, 0
    summary = {
        "status": "screening_complete",
        "phase_candidates": candidates,
        "winning_candidate": winner,
        "phase_c_triggered": phase_c_trigger,
        "phase_c_status": phase_c_status,
        "phase_c_gate": {
            "B2_minus_capacity_mean_acc": relation_gain,
            "positive_dataset_means": relation_positive_datasets,
            "positive_paired_seeds": relation_positive_pairs,
            "relation_shuffle": relation_iv,
        },
        "run_counts": {
            "new_formal_trainings": trained_runs,
            "failed_trainings_unresolved": failed_runs,
            "failed_attempts_total": failed_attempts,
            "failed_attempts_retried_successfully": retried_failures,
            "training_attempts": training_attempts,
            "successful_formal_trainings_by_phase": trained_by_phase,
            "preexisting_jobs_reused": reused_jobs,
            "screening_uniform_c0_reused": 9,
            "all_dataset_uniform_c0_checkpoints_available": 15,
            "nonformal_debug_smokes": nonformal_smokes,
        },
        "confirmation_analysis": confirmation_summary,
        "decision_thresholds": {
            "candidate_positive_datasets": "at least 2/3",
            "candidate_positive_seed_pairs": "at least 6/9",
            "no_systematic_macro_f1_damage": "overall delta >= -0.003 and at most one dataset mean delta < -0.005",
            "functional_intervention": "mean absolute alpha change > 0.001 and mean Acc or Macro-F1 drop >= 0.0005",
            "node_adaptation": "both train-mean and shuffled delta must each lower Acc or Macro-F1 by at least 0.0005",
            "phase_c_shuffle_commensurate": "relation shuffle drop >= max(50% of B2-capacity Acc gain, 0.0005)",
        },
    }
    if phase_c_status == "C1_COMPLETE":
        summary["status"] = "phase_c_complete"
    (RESULTS / "routing_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


@torch.no_grad()
def analyze_phase_c(device):
    rows, summaries = [], []
    for dataset in SCREEN_DATASETS:
        for seed in SEEDS:
            recs, summary = analyze_intervention(
                "phase_c", "cross_relation", dataset, seed,
                checkpoint_path("phase_c", "cross_relation", dataset, seed), device, "joint",
            )
            rows.extend(recs)
            summaries.append(summary)
    pd.DataFrame(rows).to_csv(RESULTS / "phase_c_intervention.csv", index=False)
    results = [
        _load_result_row("phase_c", "cross_relation", ds, seed, "C1_cross_relation")
        for ds in SCREEN_DATASETS for seed in SEEDS
    ]
    pd.DataFrame(results).to_csv(RESULTS / "phase_c_results.csv", index=False)
    (OUTPUT / "phase_c_routing_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def make_confirmation_results():
    summary = json.loads((RESULTS / "routing_audit_summary.json").read_text(encoding="utf-8"))
    winner = summary.get("winning_candidate")
    chosen = ["global_simplex", "hierarchical_flat_simplex"]
    name_map = {
        "A1_global_simplex": "global_simplex",
        "A2_node_flat_simplex": "node_flat_simplex",
        "A3_hierarchical_flat_simplex": "hierarchical_flat_simplex",
        "B1_trajectory": "trajectory",
        "B2_relation": "relation",
        "C1_cross_relation": "cross_relation",
    }
    if winner:
        chosen.append(name_map.get(winner, winner))
    if winner == "C1_cross_relation":
        chosen.append("relation")
    phases = {
        "global_simplex": "phase_a",
        "node_flat_simplex": "phase_a",
        "hierarchical_flat_simplex": "phase_a",
        "trajectory": "phase_b", "relation": "phase_b",
        "cross_relation": "phase_c",
    }
    rows = []
    reference = OUTPUT.parent / "cmrf_discovery_v1"
    for dataset in ALL_DATASETS:
        for seed in SEEDS:
            metrics = _metric_file(reference / "runs" / "uniform" / dataset / f"seed{seed}")
            ckpt = torch.load(
                reference / "reference_uniform" / "checkpoints" / f"{dataset}_seed{seed}.pt",
                map_location="cpu", weights_only=False,
            )
            rows.append({
                "variant": "uniform", "model_variant": "uniform",
                "phase": "reused_e3_c0", "dataset": dataset, "seed": seed,
                "val_acc": _read_metric(metrics, "val_acc"),
                "val_macro_f1": _read_metric(metrics, "val_macro_f1"),
                "best_epoch": ckpt.get("epoch"),
                "trainable_parameter_count": sum(v.numel() for v in ckpt["model_state"].values())
                    + sum(v.numel() for v in ckpt["head_state"].values()),
                "confirmation_source": "reused_e3_c0",
            })
    for variant in dict.fromkeys(chosen):
        for dataset in ALL_DATASETS:
            for seed in SEEDS:
                if dataset in SCREEN_DATASETS:
                    row = _load_result_row(phases[variant], variant, dataset, seed, variant)
                    row["confirmation_source"] = "screening_checkpoint_reused"
                else:
                    row = _load_result_row("confirmation", variant, dataset, seed, variant)
                    row["confirmation_source"] = "confirmation_training"
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "confirmation_results.csv", index=False)
    uniform = frame.loc[frame.variant == "uniform", [
        "dataset", "seed", "val_acc", "val_macro_f1"
    ]]
    paired = []
    for variant in ("global_simplex", "hierarchical_flat_simplex"):
        candidate = frame.loc[frame.variant == variant]
        merged = candidate.merge(
            uniform, on=["dataset", "seed"], suffixes=("_candidate", "_uniform"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            paired.append({
                "variant": variant,
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "delta_val_acc": float(row["val_acc_candidate"] - row["val_acc_uniform"]),
                "delta_macro_f1": float(row["val_macro_f1_candidate"] - row["val_macro_f1_uniform"]),
                "confirmation_source": row["confirmation_source"],
            })
    hierarchical = frame.loc[frame.variant == "hierarchical_flat_simplex"]
    global_simplex = frame.loc[frame.variant == "global_simplex"]
    hierarchy_pairs = hierarchical.merge(
        global_simplex, on=["dataset", "seed"],
        suffixes=("_hierarchical", "_global"), validate="one_to_one",
    )
    for _, row in hierarchy_pairs.iterrows():
        paired.append({
            "variant": "hierarchical_minus_global_simplex",
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "delta_val_acc": float(row["val_acc_hierarchical"] - row["val_acc_global"]),
            "delta_macro_f1": float(row["val_macro_f1_hierarchical"] - row["val_macro_f1_global"]),
            "confirmation_source": row["confirmation_source_hierarchical"],
        })
    pd.DataFrame(paired).to_csv(RESULTS / "confirmation_analysis.csv", index=False)
    return frame


def write_final_report(summary, confirmation=None):
    decisions = pd.read_csv(RESULTS / "routing_audit_decision_matrix.csv")
    phase_a = pd.read_csv(RESULTS / "phase_a_analysis.csv")
    phase_b = pd.read_csv(RESULTS / "phase_b_capacity_control.csv")
    lines = [
        "# Experiment 4 — Adaptive Structural Routing Sufficiency Audit", "",
        "- Branch/base: routing_audit, starting commit cf938f9fb6cde3b5ba1edd1e2ff5f944e228b15f.",
        "- NC only; all findings use Train + Validation; task.evaluate_test=false.",
        "- Fixed physical topology and E3 C0 unit-weight symmetric normalized operator.",
        f"- Phase C: {summary.get('phase_c_status')}.",
        f"- Screening winner: {summary.get('winning_candidate')}.",
        f"- Successful formal runs: {summary.get('run_counts', {}).get('new_formal_trainings', 0)}; attempts: {summary.get('run_counts', {}).get('training_attempts', 0)}; unresolved failures: {summary.get('run_counts', {}).get('failed_trainings_unresolved', 0)}; retry successes: {summary.get('run_counts', {}).get('failed_attempts_retried_successfully', 0)}; nonformal debug smokes excluded: {summary.get('run_counts', {}).get('nonformal_debug_smokes', 0)}.",
        "",
        "## A0 utility stability", "",
        "E3 raw tables were read directly; see a0_utility_stability_report.md and CSV.", "",
        "## Phase A", "",
        "| Comparison | Dataset | Mean Δ Val Acc | Mean Δ Macro-F1 | Positive seed pairs |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in phase_a.iterrows():
        lines.append(f"| {r['candidate']} vs {r['parent']} | {r['dataset']} | {r['mean_delta_acc']:.4f} | {r['mean_delta_macro_f1']:.4f} | {int(r['positive_acc_pairs'])}/{int(r['paired_seeds'])} |")
    lines += [
        "", "Node/global frozen interventions are in phase_a_intervention.csv. An A3 gain without a functional residual intervention is interpreted as global calibration.",
        "", "## Phase B", "",
        "| Comparison | Dataset | Mean Δ Val Acc | Mean Δ Macro-F1 | Positive seed pairs |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in phase_b.iterrows():
        lines.append(f"| {r['candidate']} vs {r['parent']} | {r['dataset']} | {r['mean_delta_acc']:.4f} | {r['mean_delta_macro_f1']:.4f} | {int(r['positive_acc_pairs'])}/{int(r['paired_seeds'])} |")
    lines += [
        "", "B2 relation mean/shuffle interventions are in phase_b_intervention.csv; parameter counts and paired contrasts are in phase_b_capacity_control.csv.",
        "", "B2 and its capacity-control parameter balance is recorded as parameter_diff_pct in phase_b_capacity_control.csv.",
        "", "## Decision matrix", "",
        "| Hypothesis | Decision | Datasets supported | Paired seeds | Δ Acc | Δ Macro-F1 | Functional intervention | Confirmation (datasets; seeds; Δ Acc; Δ F1) |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for _, r in decisions.iterrows():
        if pd.notna(r.get("confirmation_mean_delta_val_acc", np.nan)):
            confirm = (
                f"{int(r['confirmation_datasets_supported'])}/5; "
                f"{r['confirmation_positive_paired_seeds']}; "
                f"{r['confirmation_mean_delta_val_acc']:.4f}; "
                f"{r['confirmation_mean_delta_macro_f1']:.4f}"
            )
        else:
            confirm = "not run for this hypothesis"
        lines.append(f"| {r['hypothesis']} | {r['status']} | {r['datasets_supported']} | {r['positive_paired_seeds']} | {r['mean_delta_val_acc']:.4f} | {r['mean_delta_macro_f1']:.4f} | {r['functional_intervention']} | {confirm} |")
    if confirmation is not None and not confirmation.empty:
        trained = int((confirmation.confirmation_source == "confirmation_training").sum())
        reused = int((confirmation.confirmation_source == "screening_checkpoint_reused").sum())
        lines += ["", "## Five-dataset confirmation", "", f"New confirmation rows: {trained}; reused screening rows: {reused}; Uniform rows reused from E3 C0: {int((confirmation.confirmation_source == 'reused_e3_c0').sum())}.", "", "| Variant | Dataset | Mean Val Acc | Mean Macro-F1 | Δ Acc vs Uniform | Δ Macro-F1 vs Uniform |", "|---|---|---:|---:|---:|---:|"]
        confirmation_analysis_path = RESULTS / "confirmation_analysis.csv"
        confirmation_analysis = (
            pd.read_csv(confirmation_analysis_path)
            if confirmation_analysis_path.is_file() else pd.DataFrame()
        )
        for (variant, dataset), part in confirmation.groupby(["variant", "dataset"]):
            delta = confirmation_analysis.loc[
                (confirmation_analysis.variant == variant)
                & (confirmation_analysis.dataset == dataset)
            ] if not confirmation_analysis.empty else pd.DataFrame()
            if variant == "uniform" or delta.empty:
                delta_acc = delta_f1 = float("nan")
                delta_acc_text = delta_f1_text = "—"
            else:
                delta_acc = float(delta.delta_val_acc.mean())
                delta_f1 = float(delta.delta_macro_f1.mean())
                delta_acc_text, delta_f1_text = f"{delta_acc:.4f}", f"{delta_f1:.4f}"
            lines.append(f"| {variant} | {dataset} | {part.val_acc.mean():.4f} | {part.val_macro_f1.mean():.4f} | {delta_acc_text} | {delta_f1_text} |")
        if not confirmation_analysis.empty:
            lines += ["", "Paired five-dataset aggregate versus Uniform:"]
            for variant, part in confirmation_analysis.groupby("variant"):
                ds_means = part.groupby("dataset").delta_val_acc.mean()
                lines.append(
                    f"- {variant}: mean Δ Acc={part.delta_val_acc.mean():.4f}, "
                    f"mean Δ Macro-F1={part.delta_macro_f1.mean():.4f}, "
                    f"positive dataset means={int((ds_means > 0).sum())}/5, "
                    f"positive paired seeds={int((part.delta_val_acc > 0).sum())}/{len(part)}."
                )
            lines.append(
                "A1_global_simplex passed the predeclared three-dataset screen, but five-dataset "
                "confirmation has a negative overall mean and gains on only two dataset means. "
                "Treat the screen result as mixed and do not claim a robust cross-dataset improvement."
            )
    lines += [
        "", "## Interpretation boundary", "",
        "The report separates global structural calibration, node adaptation, relation-context-aware adaptation, and paired cross-modal relational conditioning. A skipped Phase C only rejects its predeclared gate, not every possible cross-modal interaction.",
        "", "No expert bank, signed filter, common/private decomposition, task-aware router, or deeper transformer was added.",
    ]
    (RESULTS / "routing_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary["status"] = "complete"
    (RESULTS / "routing_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["a0", "phase_a", "phase_b", "phase_c", "finalize", "confirmation", "report"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "a0":
        run_a0_utility()
    elif args.phase == "phase_a":
        run_phase_a_interventions(torch.device(args.device))
        make_phase_a_results()
    elif args.phase == "phase_b":
        run_phase_b_interventions(torch.device(args.device))
        make_phase_b_results()
    elif args.phase == "phase_c":
        analyze_phase_c(torch.device(args.device))
    elif args.phase == "finalize":
        finalize_decision()
    elif args.phase == "confirmation":
        make_confirmation_results()
    elif args.phase == "report":
        summary = json.loads((RESULTS / "routing_audit_summary.json").read_text(encoding="utf-8"))
        path = RESULTS / "confirmation_results.csv"
        write_final_report(summary, pd.read_csv(path) if path.is_file() else None)


if __name__ == "__main__":
    main()

