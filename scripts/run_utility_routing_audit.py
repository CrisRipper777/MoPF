from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_mag_data
from src.models.cmrf_probe import Model as C0Model
from src.models.utility_routing_audit import (
    UtilityRouter, action_bank, expected_mixture, safe_probabilities, uniform_state_mix,
)

OUTPUT = ROOT / "outputs/utility_routing_audit_v1"
RESULTS = ROOT / "results/utility_routing_audit_v1"
C0_ROOT = ROOT / "outputs/cmrf_discovery_v1"
SCREEN = ["Movies", "ele-fashion", "Reddit-S"]
CONFIRM = ["Toys", "Grocery"]
ALL = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
ACTION_NAMES = ["A0_SELF", "A1_S1", "A2_S2", "A3_S3", "AU_UNIFORM"]
VARIANT_SPECS = {
    "B0_frozen_direct_ce": ("flat", "direct_ce"),
    "B1_flat_utility_kd": ("flat", "utility_kd"),
    "B2_weighted_utility_kd": ("flat", "weighted_utility_kd"),
    "C0_decision_capacity_control": ("decision_capacity", "weighted_utility_kd"),
    "C1_decision_utility_router": ("decision_response", "weighted_utility_kd"),
    "D0_relation_capacity_control": ("relation_capacity", "weighted_utility_kd"),
    "D1_utility_relation_router": ("relation", "weighted_utility_kd"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def c0_checkpoint(dataset: str, seed: int) -> Path:
    return C0_ROOT / "reference_uniform" / "checkpoints" / f"{dataset}_seed{seed}.pt"


def c0_metrics(dataset: str, seed: int) -> Path:
    return C0_ROOT / "runs" / "uniform" / dataset / f"seed{seed}" / "metrics.json"


def cache_path(dataset: str, seed: int) -> Path:
    return OUTPUT / "cache" / f"{dataset}_seed{seed}.pt"


def raw_action_path(dataset: str, seed: int) -> Path:
    return OUTPUT / "action_landscape" / f"{dataset}_seed{seed}.csv.gz"


def run_dir(phase: str, variant: str, dataset: str, seed: int) -> Path:
    return OUTPUT / "runs" / phase / variant / dataset / f"seed{seed}"


def checkpoint_path(phase: str, variant: str, dataset: str, seed: int) -> Path:
    return OUTPUT / "checkpoints" / phase / variant / f"{dataset}_seed{seed}.pt"


def make_cfg(dataset: str, seed: int, device: str):
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.dataset = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    cfg.task = OmegaConf.load(ROOT / "configs/task/nc.yaml")
    cfg.model = OmegaConf.load(ROOT / "configs/model/cmrf_probe.yaml")
    cfg.seed = int(seed)
    cfg.num_runs = 1
    cfg.device = device
    cfg.model.composition_mode = "uniform"
    cfg.task.evaluate_test = False
    cfg.pop("hydra", None)
    OmegaConf.resolve(cfg)
    return cfg


def load_c0(dataset: str, seed: int, device: torch.device):
    cfg = make_cfg(dataset, seed, str(device))
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = C0Model(cfg, info).to(device)
    payload = torch.load(c0_checkpoint(dataset, seed), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()
    for module in (model, classifier):
        for param in module.parameters():
            param.requires_grad_(False)
    return cfg, data, payload, model, classifier


def metric_pair(logits: torch.Tensor, target: torch.Tensor, labels: list[int]):
    pred = logits.argmax(-1)
    return (
        float((pred == target).float().mean().item()),
        float(f1_score(
            target.detach().cpu().numpy(), pred.detach().cpu().numpy(),
            labels=labels, average="macro", zero_division=0,
        )),
        pred,
    )


def select_train_val_labels(labels: torch.Tensor, train_idx: torch.Tensor, val_idx: torch.Tensor):
    """Return only the explicitly allowed Train/Validation labels."""
    train_idx = train_idx.to(labels.device)
    val_idx = val_idx.to(labels.device)
    allowed = torch.cat([train_idx, val_idx])
    return allowed, labels[allowed]


def utility_distillation_loss(
    p_text, p_visual, q_text, q_visual, strength_text, strength_visual,
    train_idx, *, weighted: bool,
):
    """KD objective that indexes teacher targets exclusively at Train nodes."""
    train_idx = train_idx.to(p_text.device)
    wt = strength_text[train_idx] if weighted else None
    wv = strength_visual[train_idx] if weighted else None
    return 0.5 * (
        _weighted_kd(q_text[train_idx], p_text[train_idx], wt)
        + _weighted_kd(q_visual[train_idx], p_visual[train_idx], wv)
    )


def build_eprop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Coalesced symmetric physical edges with all loops removed."""
    edge, _ = remove_self_loops(edge_index.long())
    edge = to_undirected(edge, num_nodes=int(num_nodes))
    edge, _ = remove_self_loops(edge)
    return coalesce(edge, num_nodes=int(num_nodes))


def teacher_from_ce(ce: torch.Tensor):
    q = torch.softmax(-ce, dim=-1)
    qsafe = q.clamp_min(1e-12)
    entropy = -(qsafe * qsafe.log()).sum(-1)
    strength = (1.0 - entropy / math.log(5.0)).clamp(0.0, 1.0)
    order = torch.sort(ce, dim=-1).values
    return q, entropy, strength, ce.argmin(-1), order[:, 1] - order[:, 0]


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), 1e-12, 1.0)
    q = np.clip(q.astype(np.float64), 1e-12, 1.0)
    p /= p.sum(axis=-1, keepdims=True)
    q /= q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m), axis=-1) + np.sum(q * np.log(q / m), axis=-1)).mean())


def cosine_rows(p: np.ndarray, q: np.ndarray) -> float:
    denom = np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1)
    valid = denom > 0
    if not valid.any():
        return float("nan")
    return float(((p[valid] * q[valid]).sum(-1) / denom[valid]).mean())


@torch.no_grad()
def preflight_one(dataset: str, seed: int, device_name: str):
    device = torch.device(device_name)
    cfg, data, payload, c0, classifier = load_c0(dataset, seed, device)
    allowed_cpu, y_allowed_cpu = select_train_val_labels(
        data.y, data.train_idx, data.val_idx
    )
    allowed_cpu = allowed_cpu.cpu()
    n_train = int(data.train_idx.numel())
    train_local = torch.arange(n_train, dtype=torch.long, device=device)
    val_local = torch.arange(n_train, int(allowed_cpu.numel()), dtype=torch.long, device=device)
    allowed = allowed_cpu.to(device)
    # This is the only label access in preflight: explicitly indexed Train and Val rows.
    y_allowed = y_allowed_cpu.to(device).long()
    valid_y = (y_allowed >= 0) & (y_allowed < int(data.num_classes))
    eval_labels = sorted(set(int(v) for v in y_allowed[valid_y].detach().cpu().tolist()))
    if not torch.all(valid_y):
        raise RuntimeError(f"non-class labels found on Train/Val nodes: {dataset}/{seed}")

    x = data.x.to(device)
    edge = data.edge_index.to(device)
    analysis = c0.analyze(x, edge)
    states_t, states_v = analysis["S_text"], analysis["S_visual"]
    au_t = uniform_state_mix(states_t)
    au_v = uniform_state_mix(states_v)
    manual_fused = c0._fuse(au_t, au_v)
    direct_fused = analysis["fused_z"]
    embed_error = float((manual_fused - direct_fused).abs().max().item())
    direct_prob = torch.softmax(classifier(direct_fused[allowed]), dim=-1)
    manual_prob = torch.softmax(classifier(manual_fused[allowed]), dim=-1)
    probability_error = float((direct_prob - manual_prob).abs().max().item())
    val_global = allowed[val_local]
    direct_logits = classifier(direct_fused[val_global])
    manual_logits = classifier(manual_fused[val_global])
    # Local allowed rows preserve Train first, Validation second; never touch test_idx.
    val_y = y_allowed[val_local]
    direct_acc, direct_f1, _ = metric_pair(direct_logits, val_y, eval_labels)
    manual_acc, manual_f1, _ = metric_pair(manual_logits, val_y, eval_labels)

    ckpt_metrics = payload["metrics"]
    def stored_metric(name):
        value = ckpt_metrics[name]
        return float(value["mean"]) if isinstance(value, dict) else float(value)
    stored_acc = stored_metric("val_acc")
    stored_f1 = stored_metric("val_macro_f1")
    check = {
        "dataset": dataset, "seed": seed, "checkpoint": str(c0_checkpoint(dataset, seed)),
        "checkpoint_sha256": sha256(c0_checkpoint(dataset, seed)),
        "stored_val_acc": stored_acc, "reloaded_val_acc": direct_acc,
        "stored_val_macro_f1": stored_f1, "reloaded_val_macro_f1": direct_f1,
        "val_acc_abs_delta": abs(stored_acc - direct_acc),
        "val_macro_f1_abs_delta": abs(stored_f1 - direct_f1),
        "uniform_embedding_max_abs_error": embed_error,
        "uniform_probability_max_abs_error": probability_error,
        "forward_equivalent_lt_1e-6": embed_error < 1e-6 and probability_error < 1e-6,
        "metric_equivalent_1e-6": abs(stored_acc-direct_acc) < 1e-6 and abs(stored_f1-direct_f1) < 1e-6,
        "eval_labels_source": "unique Train+Validation labels only",
        "test_labels_indexed": False,
    }

    logits_by_mod = {}
    state_bank_t = action_bank(states_t)
    state_bank_v = action_bank(states_v)
    for modality in ("text", "visual"):
        action_logits = []
        for action in range(5):
            if modality == "text":
                fused = c0._fuse(state_bank_t[allowed, action], au_v[allowed])
            else:
                fused = c0._fuse(au_t[allowed], state_bank_v[allowed, action])
            action_logits.append(classifier(fused))
        logits_by_mod[modality] = torch.stack(action_logits, dim=1)
    ce_by_mod = {}
    for modality, logits in logits_by_mod.items():
        expanded_y = y_allowed.unsqueeze(1).expand(-1, 5)
        ce_by_mod[modality] = F.cross_entropy(
            logits.reshape(-1, int(data.num_classes)),
            expanded_y.reshape(-1), reduction="none",
        ).reshape(-1, 5)
    q_by_mod = {}
    entropy_by_mod = {}
    strength_by_mod = {}
    best_by_mod = {}
    margin_by_mod = {}
    for modality in ("text", "visual"):
        q, entropy, strength, best, margin = teacher_from_ce(ce_by_mod[modality])
        q_by_mod[modality] = q
        entropy_by_mod[modality] = entropy
        strength_by_mod[modality] = strength
        best_by_mod[modality] = best
        margin_by_mod[modality] = margin

    split_values = np.array(["train"] * n_train + ["validation"] * (len(allowed_cpu)-n_train))
    raw_rows = []
    summary_rows = []
    for modality in ("text", "visual"):
        logits = logits_by_mod[modality]
        prob = torch.softmax(logits, dim=-1)
        ce = ce_by_mod[modality]
        q = q_by_mod[modality]
        ent = entropy_by_mod[modality]
        strength = strength_by_mod[modality]
        best = best_by_mod[modality]
        best_gap = margin_by_mod[modality]
        pred = logits.argmax(-1)
        top2 = torch.topk(prob, k=2, dim=-1).values
        pred_margin = top2[..., 0] - top2[..., 1]
        true_p = prob.gather(-1, y_allowed[:, None, None].expand(-1,5,1)).squeeze(-1)
        gain = ce[:, 4] - ce.min(-1).values
        oracle_correct = (pred == y_allowed[:, None]).any(dim=1)
        uniform_correct = pred[:, 4] == y_allowed
        # Transfer once before row construction; per-scalar CUDA .item() calls
        # otherwise force millions of device synchronizations on large graphs.
        prob_np = prob.detach().cpu().numpy()
        ce_np = ce.detach().cpu().numpy()
        true_p_np = true_p.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        pred_margin_np = pred_margin.detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        ent_np = ent.detach().cpu().numpy()
        strength_np = strength.detach().cpu().numpy()
        best_np = best.detach().cpu().numpy()
        best_gap_np = best_gap.detach().cpu().numpy()
        gain_np = gain.detach().cpu().numpy()
        pred_entropy_np = -(np.clip(prob_np, 1e-12, 1.0) * np.log(np.clip(prob_np, 1e-12, 1.0))).sum(-1)
        for local, node_id in enumerate(allowed_cpu.tolist()):
            for action, action_name in enumerate(ACTION_NAMES):
                raw_rows.append({
                    "dataset": dataset, "seed": seed, "node_id": int(node_id),
                    "split": split_values[local], "modality": modality, "action": action_name,
                    "CE": float(ce_np[local, action]),
                    "true_class_probability": float(true_p_np[local, action]),
                    "predicted_class": int(pred_np[local, action]),
                    "prediction_entropy": float(pred_entropy_np[local, action]),
                    "prediction_margin": float(pred_margin_np[local, action]),
                    "teacher_probability": float(q_np[local, action]),
                    "teacher_entropy": float(ent_np[local]),
                    "preference_strength": float(strength_np[local]),
                    "hard_best_action": ACTION_NAMES[int(best_np[local])],
                    "best_vs_second_CE_margin": float(best_gap_np[local]),
                    "oracle_CE_gain_vs_Uniform": float(gain_np[local]),
                })
        for split_name, idx in (("train", torch.arange(n_train, device=device)),
                                ("validation", torch.arange(n_train, len(allowed_cpu), device=device))):
            if idx.numel() == 0:
                continue
            b = best[idx]
            counts = torch.bincount(b, minlength=5).float() / idx.numel()
            ce_mean = ce[idx].mean(0)
            local_oracle_correct = oracle_correct[idx].float().mean()
            local_uniform_acc = uniform_correct[idx].float().mean()
            max_action_freq = float(counts.max().item())
            mean_gain = float(gain[idx].mean().item())
            summary_rows.append({
                "dataset": dataset, "seed": seed, "split": split_name,
                "modality": modality, "node_count": int(idx.numel()),
                **{f"hard_best_freq_{ACTION_NAMES[a]}": float(counts[a].item()) for a in range(5)},
                "mean_teacher_entropy": float(ent[idx].mean().item()),
                "mean_preference_strength": float(strength[idx].mean().item()),
                "uniform_hard_best_frequency": float(counts[4].item()),
                "self_hard_best_frequency": float(counts[0].item()),
                "mean_CE_AU": float(ce_mean[4].item()),
                "mean_oracle_CE": float(ce[idx].min(-1).values.mean().item()),
                "mean_oracle_CE_gain_vs_Uniform": mean_gain,
                "oracle_accuracy_upper_bound": float(local_oracle_correct.item()),
                "uniform_action_accuracy": float(local_uniform_acc.item()),
                "oracle_accuracy_gain_vs_Uniform": float((local_oracle_correct-local_uniform_acc).item()),
                "max_action_best_frequency": max_action_freq,
                "action_space_status": (
                    "ROUTING_ACTION_SPACE_LOW_HEADROOM"
                    if max_action_freq > 0.95 and mean_gain < 0.01 else "OK"
                ),
            })

    raw_path = raw_action_path(dataset, seed)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False, compression="gzip")
    summary_path = OUTPUT / "action_landscape_summaries" / f"{dataset}_seed{seed}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # E_prop is the coalesced symmetric non-self physical edge set.
    prop = build_eprop(data.edge_index.detach().cpu(), int(data.num_nodes))
    h0_text_all = states_t[0].detach().cpu().float()
    h0_visual_all = states_v[0].detach().cpu().float()
    # Map allowed global node ids back to raw-table order; labels outside Train/Val do not enter cache.
    cache = {
        "dataset": dataset, "seed": seed, "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes), "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]), "hidden_dim": int(states_t[0].size(-1)),
        "allowed_idx": allowed_cpu.long(),
        "train_idx": torch.arange(n_train, dtype=torch.long),
        "val_idx": torch.arange(n_train, len(allowed_cpu), dtype=torch.long),
        "states_text": [s[allowed].detach().cpu().float() for s in states_t],
        "states_visual": [s[allowed].detach().cpu().float() for s in states_v],
        "h0_text_all": h0_text_all,
        "h0_visual_all": h0_visual_all,
        "edge_prop": prop.detach().cpu().long(),
        "y_allowed": y_allowed.detach().cpu().long(),
        "eval_labels": eval_labels,
        "action_ce_text": ce_by_mod["text"].detach().cpu().float(),
        "action_ce_visual": ce_by_mod["visual"].detach().cpu().float(),
        "action_logits_text": logits_by_mod["text"].detach().cpu().float(),
        "action_logits_visual": logits_by_mod["visual"].detach().cpu().float(),
        "teacher_text": q_by_mod["text"].detach().cpu().float(),
        "teacher_visual": q_by_mod["visual"].detach().cpu().float(),
        "teacher_strength_text": strength_by_mod["text"].detach().cpu().float(),
        "teacher_strength_visual": strength_by_mod["visual"].detach().cpu().float(),
        "decision_text": None, "decision_visual": None,
        "c0_baseline": {
            "val_acc": direct_acc, "val_macro_f1": direct_f1,
            "stored_val_acc": stored_acc, "stored_val_macro_f1": stored_f1,
        },
        "c0_checkpoint": str(c0_checkpoint(dataset, seed)),
        "c0_checkpoint_sha256": check["checkpoint_sha256"],
    }
    for modality in ("text", "visual"):
        logits = logits_by_mod[modality]
        probs = torch.softmax(logits, -1).clamp_min(1e-12)
        ent = -(probs * probs.log()).sum(-1, keepdim=True)
        top2p = torch.topk(probs, k=2, dim=-1).values
        margin = top2p[:, :, :1] - top2p[:, :, 1:2]
        pa = probs[:, 4:5, :].clamp_min(1e-12)
        kl = (probs * (probs.log() - pa.log())).sum(-1, keepdim=True)
        delta = logits - logits[:, 4:5, :]
        cache[f"decision_{modality}"] = torch.cat([delta, ent, margin, kl], dim=-1).cpu().float()
    cache_out = cache_path(dataset, seed)
    cache_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_out)

    check["action_landscape_path"] = str(raw_path.relative_to(ROOT))
    check["action_landscape_sha256"] = sha256(raw_path)
    check["action_cache_path"] = str(cache_out.relative_to(ROOT))
    check["action_rows"] = len(raw_rows)
    check["allowed_node_count"] = int(len(allowed_cpu))
    check["train_node_count"] = int(n_train)
    check["validation_node_count"] = int(len(allowed_cpu)-n_train)
    check["test_labels_indexed"] = False
    check_path = OUTPUT / "preflight_checks" / f"{dataset}_seed{seed}.json"
    check_path.parent.mkdir(parents=True, exist_ok=True)
    check_path.write_text(json.dumps(check, indent=2), encoding="utf-8")
    return check, summary_rows


def aggregate_preflight(one_results):
    summary_rows = [row for _, rows in one_results for row in rows]
    pd.DataFrame(summary_rows).to_csv(RESULTS / "action_landscape_summary.csv", index=False)
    stability_rows = []
    for dataset in SCREEN:
        caches = {
            seed: torch.load(cache_path(dataset, seed), map_location="cpu", weights_only=False)
            for seed in SEEDS
        }
        for modality in ("text", "visual"):
            for split_name in ("train", "validation"):
                indices_by_seed = {}
                for seed, cache in caches.items():
                    cache_split = "val" if split_name == "validation" else split_name
                    local = cache[f"{cache_split}_idx"].numpy()
                    global_ids = cache["allowed_idx"][local].numpy()
                    indices_by_seed[seed] = {
                        int(node_id): int(loc) for node_id, loc in zip(global_ids, local)
                    }
                common = set.intersection(*(set(x) for x in indices_by_seed.values()))
                for left, right in ((42, 43), (42, 44), (43, 44)):
                    common_ids = sorted(common)
                    li = torch.tensor([indices_by_seed[left][i] for i in common_ids])
                    ri = torch.tensor([indices_by_seed[right][i] for i in common_ids])
                    ql = caches[left][f"teacher_{modality}"][li].numpy()
                    qr = caches[right][f"teacher_{modality}"][ri].numpy()
                    cel = caches[left][f"action_ce_{modality}"][li].numpy()
                    cer = caches[right][f"action_ce_{modality}"][ri].numpy()
                    wl = caches[left][f"teacher_strength_{modality}"][li].numpy()
                    wr = caches[right][f"teacher_strength_{modality}"][ri].numpy()
                    spearman = (
                        float(spearmanr(wl, wr).statistic)
                        if len(wl) >= 3 and np.std(wl) > 0 and np.std(wr) > 0
                        else float("nan")
                    )
                    stability_rows.append({
                        "dataset": dataset, "modality": modality, "split": split_name,
                        "seed_pair": f"{left}-{right}", "common_nodes": len(common_ids),
                        "hard_best_action_agreement": float((cel.argmin(-1) == cer.argmin(-1)).mean()),
                        "soft_q_mean_js_divergence": js_divergence(ql, qr),
                        "soft_q_mean_cosine_similarity": cosine_rows(ql, qr),
                        "preference_strength_spearman": spearman,
                    })
    pd.DataFrame(stability_rows).to_csv(RESULTS / "action_stability.csv", index=False)

    checks = [
        json.loads(p.read_text())
        for p in sorted((OUTPUT / "preflight_checks").glob("*.json"))
    ]
    report = [
        "# E5 C0 reuse and action-landscape preflight",
        "",
        f"Screening C0 checks completed: {len(checks)}/9. No Toys/Grocery C0 checkpoint or split data was loaded before the winner lock.",
        "",
        "All route supervision/audit rows are Train or Validation only. Test indices/labels were never indexed. Frozen C0 uses the stored projectors, normalized physical propagation, plain fusion, and classifier.",
        "",
        "## C0 reproduction",
        "",
        "| Dataset | Seed | Embedding max abs. error | Probability max abs. error | Stored/reloaded Val Acc | Stored/reloaded Val Macro-F1 | Forward <1e-6 | Metrics <1e-6 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for c in checks:
        report.append(
            f"| {c['dataset']} | {c['seed']} | {c['uniform_embedding_max_abs_error']:.3g} | "
            f"{c['uniform_probability_max_abs_error']:.3g} | "
            f"{c['stored_val_acc']:.6f}/{c['reloaded_val_acc']:.6f} | "
            f"{c['stored_val_macro_f1']:.6f}/{c['reloaded_val_macro_f1']:.6f} | "
            f"{c['forward_equivalent_lt_1e-6']} | {c['metric_equivalent_1e-6']} |"
        )
    metric_mismatches = [c for c in checks if not c["metric_equivalent_1e-6"]]
    if metric_mismatches:
        report += [
            "",
            "Stored C0 Validation Macro-F1 did not reproduce within 1e-6 under the no-Test Train+Validation label protocol for at least one row. The experiment retains the required no-Test rule and reports both values; no Test labels are consulted to force a match.",
        ]
    report += [
        "",
        "## Action-space/headroom summary",
        "",
        "See action_landscape_summary.csv for per-dataset/seed/split/modality action frequencies, entropy, preference strength, oracle CE gain, and label-oracle accuracy upper bound. The upper bound is not an achievable router score.",
        "",
        "See action_stability.csv for split-matched cross-seed hard-action agreement, soft-teacher JS/cosine, and preference-strength Spearman.",
        "",
        "## Data provenance",
        "",
        f"- Frozen C0 source: {C0_ROOT.relative_to(ROOT)}/reference_uniform/checkpoints/.",
        "- The five action embeddings share C0's frozen plain fusion and classifier; modality-specific action landscapes hold the opposite modality at AU.",
        "- Train teachers use softmax(-CE) without temperature and preference strength 1-H(q)/log(5). Validation teachers occur only in audit columns and never enter the training objective.",
        "- Raw action rows and tensor caches are under outputs/utility_routing_audit_v1/; no E3/E4 output was overwritten.",
        "",
    ]
    RESULTS.joinpath("preflight_report.md").write_text("\n".join(report), encoding="utf-8")
    (RESULTS / "preflight_summary.json").write_text(
        json.dumps({
            "checks": checks,
            "all_forward_equivalent": all(c["forward_equivalent_lt_1e-6"] for c in checks),
            "all_metrics_equivalent": all(c["metric_equivalent_1e-6"] for c in checks),
            "metric_mismatch_count": len(metric_mismatches),
            "action_rows": int(sum(c["action_rows"] for c in checks)),
            "test_labels_indexed": False,
        }, indent=2),
        encoding="utf-8",
    )


def _append_manifest(phase: str, variant: str, dataset: str, seed: int, device: str, state: str, code: int = 0):
    import fcntl
    path = OUTPUT / "run_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        writer = csv.DictWriter(f, fieldnames=["phase", "variant", "dataset", "seed", "device", "state", "return_code"])
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "phase": phase, "variant": variant, "dataset": dataset, "seed": seed,
            "device": device, "state": state, "return_code": code,
        })
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def summarize_action_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    cache = torch.load(cache_path(str(frame.dataset.iloc[0]), int(frame.seed.iloc[0])),
                       map_location="cpu", weights_only=False)
    for (dataset, seed, split, modality), part in frame.groupby(
        ["dataset", "seed", "split", "modality"], sort=False
    ):
        ce_pivot = part.pivot(index="node_id", columns="action", values="CE")[ACTION_NAMES]
        q_pivot = part.pivot(index="node_id", columns="action", values="teacher_probability")[ACTION_NAMES]
        pred_pivot = part.pivot(index="node_id", columns="action", values="predicted_class")[ACTION_NAMES]
        ce = ce_pivot.to_numpy()
        q = q_pivot.to_numpy()
        pred = pred_pivot.to_numpy()
        strength = part.groupby("node_id").preference_strength.first().reindex(ce_pivot.index)
        split_idx = cache[f"{split}_idx"]
        ids = cache["allowed_idx"][split_idx].numpy()
        positions = {int(node): int(i) for node, i in zip(ids, split_idx.numpy())}
        selected = torch.tensor([positions[int(i)] for i in ce_pivot.index], dtype=torch.long)
        y = cache["y_allowed"][selected].numpy()
        best = ce.argmin(-1)
        counts = np.bincount(best, minlength=5) / max(len(best), 1)
        oracle_correct = (pred == y[:, None]).any(axis=1).mean()
        uniform_acc = (pred[:, 4] == y).mean()
        ent = -(np.clip(q, 1e-12, 1) * np.log(np.clip(q, 1e-12, 1))).sum(-1)
        gain = ce[:, 4] - ce.min(axis=-1)
        max_freq = float(counts.max())
        mean_gain = float(gain.mean())
        rows.append({
            "dataset": dataset, "seed": int(seed), "split": split, "modality": modality,
            "node_count": len(ce),
            **{f"hard_best_freq_{ACTION_NAMES[a]}": float(counts[a]) for a in range(5)},
            "mean_teacher_entropy": float(ent.mean()),
            "mean_preference_strength": float(strength.mean()),
            "uniform_hard_best_frequency": float(counts[4]),
            "self_hard_best_frequency": float(counts[0]),
            "mean_CE_AU": float(ce[:, 4].mean()),
            "mean_oracle_CE": float(ce.min(-1).mean()),
            "mean_oracle_CE_gain_vs_Uniform": mean_gain,
            "oracle_accuracy_upper_bound": float(oracle_correct),
            "uniform_action_accuracy": float(uniform_acc),
            "oracle_accuracy_gain_vs_Uniform": float(oracle_correct-uniform_acc),
            "max_action_best_frequency": max_freq,
            "action_space_status": (
                "ROUTING_ACTION_SPACE_LOW_HEADROOM"
                if max_freq > 0.95 and mean_gain < 0.01 else "OK"
            ),
        })
    return rows


def _subprocess(args: list[str]):
    cmd = [sys.executable, str(Path(__file__).resolve()), *args]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, check=False)


def preflight_all(devices: list[str], datasets: list[str]):
    jobs = []
    outcomes = []
    for ds in datasets:
        for seed in SEEDS:
            check_path = OUTPUT / "preflight_checks" / f"{ds}_seed{seed}.json"
            summary_path = OUTPUT / "action_landscape_summaries" / f"{ds}_seed{seed}.csv"
            raw_path = OUTPUT / "action_landscape" / f"{ds}_seed{seed}.csv.gz"
            cache_file = cache_path(ds, seed)
            reusable = all(path.is_file() for path in (check_path, summary_path, raw_path, cache_file))
            if reusable:
                check = json.loads(check_path.read_text())
                reusable = (
                    check.get("dataset") == ds
                    and int(check.get("seed", -1)) == seed
                    and check.get("test_labels_indexed") is False
                    and check.get("forward_equivalent_lt_1e-6") is True
                    and check.get("metric_equivalent_1e-6") is True
                    and check.get("checkpoint_sha256") == sha256(c0_checkpoint(ds, seed))
                    and check.get("action_landscape_sha256") == sha256(raw_path)
                )
            if reusable:
                outcomes.append((check, pd.read_csv(summary_path).to_dict("records")))
                print(f"[preflight {ds} seed={seed}] validated cache reused", flush=True)
            else:
                jobs.append((ds, seed))
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        pending = {}
        iterator = iter(jobs)
        for device in devices:
            try:
                ds, seed = next(iterator)
            except StopIteration:
                break
            pending[pool.submit(_subprocess, ["--preflight-one", ds, str(seed), device])] = (ds, seed, device)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                ds, seed, device = pending.pop(future)
                result = future.result()
                if result.returncode:
                    raise RuntimeError(f"preflight failed: {ds}/seed{seed} on {device}\n{result.stdout[-4000:]}")
                print(f"[preflight {ds} seed={seed} {device}] complete", flush=True)
                check = json.loads((OUTPUT / "preflight_checks" / f"{ds}_seed{seed}.json").read_text())
                small_summary = pd.read_csv(
                    OUTPUT / "action_landscape_summaries" / f"{ds}_seed{seed}.csv"
                ).to_dict("records")
                outcomes.append((check, small_summary))
                try:
                    nds, nseed = next(iterator)
                except StopIteration:
                    continue
                pending[pool.submit(_subprocess, ["--preflight-one", nds, str(nseed), device])] = (nds, nseed, device)
    aggregate_preflight(outcomes)


def load_c0_modules(dataset: str, seed: int, device: torch.device):
    cfg = make_cfg(dataset, seed, str(device))
    payload = torch.load(c0_checkpoint(dataset, seed), map_location="cpu", weights_only=False)
    model = C0Model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"])).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval(); classifier.eval()
    for module in (model, classifier):
        for p in module.parameters():
            p.requires_grad_(False)
    return model, classifier, payload


def _move_cache(cache: dict, device: torch.device) -> dict:
    moved = dict(cache)
    for key in ("states_text", "states_visual"):
        moved[key] = [x.to(device) for x in cache[key]]
    for key in (
        "h0_text_all", "h0_visual_all", "edge_prop", "allowed_idx",
        "train_idx", "val_idx", "y_allowed", "action_ce_text",
        "action_ce_visual", "teacher_text", "teacher_visual",
        "teacher_strength_text", "teacher_strength_visual",
        "decision_text", "decision_visual", "action_logits_text",
        "action_logits_visual",
    ):
        moved[key] = cache[key].to(device)
    moved["actions_text"] = action_bank(moved["states_text"])
    moved["actions_visual"] = action_bank(moved["states_visual"])
    return moved


def router_forward(router: UtilityRouter, cache: dict):
    return router(
        cache["states_text"], cache["states_visual"],
        decision_text=cache.get("decision_text"),
        decision_visual=cache.get("decision_visual"),
        edge_index=cache.get("edge_prop"),
        h0_text_all=cache.get("h0_text_all"),
        h0_visual_all=cache.get("h0_visual_all"),
        allowed_idx=cache.get("allowed_idx"),
    )


def _route_logits(c0, classifier, cache, p_text, p_visual):
    bank_t = cache.get("actions_text")
    bank_v = cache.get("actions_visual")
    if bank_t is None: bank_t = action_bank(cache["states_text"])
    if bank_v is None: bank_v = action_bank(cache["states_visual"])
    zt = expected_mixture(bank_t, p_text)
    zv = expected_mixture(bank_v, p_visual)
    return classifier(c0._fuse(zt, zv))


def _single_modality_logits(c0, classifier, cache, p, modality: str):
    bank_t = cache.get("actions_text")
    bank_v = cache.get("actions_visual")
    if bank_t is None: bank_t = action_bank(cache["states_text"])
    if bank_v is None: bank_v = action_bank(cache["states_visual"])
    if modality == "text":
        zt = expected_mixture(bank_t, p)
        zv = bank_v[:, 4]
    else:
        zt = bank_t[:, 4]
        zv = expected_mixture(bank_v, p)
    return classifier(c0._fuse(zt, zv))


def _regret_metrics(cache: dict, p_text: torch.Tensor, p_visual: torch.Tensor,
                    route_logits: torch.Tensor) -> dict:
    val = cache["val_idx"]
    yv = cache["y_allowed"][val]
    ce_t = cache["action_ce_text"][val]
    ce_v = cache["action_ce_visual"][val]
    logits_t = _single_modality_logits(
        _REGRET_C0, _REGRET_HEAD, cache, p_text, "text"
    )[val]
    logits_v = _single_modality_logits(
        _REGRET_C0, _REGRET_HEAD, cache, p_visual, "visual"
    )[val]
    route_ce_t = F.cross_entropy(logits_t, yv, reduction="none")
    route_ce_v = F.cross_entropy(logits_v, yv, reduction="none")
    regret_t = route_ce_t - ce_t.min(-1).values
    regret_v = route_ce_v - ce_v.min(-1).values
    excess_t = route_ce_t - ce_t[:, 4]
    excess_v = route_ce_v - ce_v[:, 4]
    q_t = torch.softmax(-ce_t, -1)
    q_v = torch.softmax(-ce_v, -1)
    q_t = q_t.clamp_min(1e-12); q_v = q_v.clamp_min(1e-12)
    pt = p_text[val].clamp_min(1e-12); pv = p_visual[val].clamp_min(1e-12)
    kl_t = (q_t * (q_t.log() - pt.log())).sum(-1)
    kl_v = (q_v * (q_v.log() - pv.log())).sum(-1)
    route_strength_t = 1.0 - (-(pt * pt.log()).sum(-1) / math.log(5.0))
    route_strength_v = 1.0 - (-(pv * pv.log()).sum(-1) / math.log(5.0))
    teacher_strength_t = cache["teacher_strength_text"][val]
    teacher_strength_v = cache["teacher_strength_visual"][val]

    def corr(a, b):
        aa=a.detach().cpu().numpy(); bb=b.detach().cpu().numpy()
        if len(aa)<3 or np.std(aa)==0 or np.std(bb)==0:
            return float("nan")
        return float(spearmanr(aa,bb).statistic)

    all_regret = torch.cat([regret_t, regret_v])
    all_excess = torch.cat([excess_t, excess_v])
    all_kl = torch.cat([kl_t, kl_v])
    return {
        "mean_routing_regret": float(all_regret.mean().item()),
        "median_routing_regret": float(all_regret.median().item()),
        "p90_routing_regret": float(torch.quantile(all_regret, 0.90).item()),
        "mean_excess_loss_vs_uniform": float(all_excess.mean().item()),
        "text_mean_regret": float(regret_t.mean().item()),
        "visual_mean_regret": float(regret_v.mean().item()),
        "text_mean_excess_loss_vs_uniform": float(excess_t.mean().item()),
        "visual_mean_excess_loss_vs_uniform": float(excess_v.mean().item()),
        "hard_action_agreement": float(0.5 * (
            (p_text[val].argmax(-1) == ce_t.argmin(-1)).float().mean().item()
            + (p_visual[val].argmax(-1) == ce_v.argmin(-1)).float().mean().item()
        )),
        "soft_target_kl": float(all_kl.mean().item()),
        "router_teacher_strength_spearman": float(0.5 * (
            corr(route_strength_t, teacher_strength_t)
            + corr(route_strength_v, teacher_strength_v)
        )),
    }


# The training process is single-run/single-device; these references are scoped
# only during regret calculations to avoid threading a frozen head through many helpers.
_REGRET_C0 = None
_REGRET_HEAD = None


def evaluate_probs(c0, classifier, cache, p_text, p_visual):
    logits = _route_logits(c0, classifier, cache, p_text, p_visual)
    val = cache["val_idx"]
    yv = cache["y_allowed"][val]
    labels = cache["eval_labels"]
    acc, f1, pred = metric_pair(logits[val], yv, labels)
    regret = _regret_metrics(cache, p_text, p_visual, logits)
    return {
        "val_acc": acc, "val_macro_f1": f1,
        "prediction": pred, "logits": logits, **regret,
    }


def _clone_cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weighted_kd(q, p, weights=None):
    q = q.clamp_min(1e-12)
    per_node = (q * (q.log() - p.clamp_min(1e-12).log())).sum(-1)
    if weights is None:
        return per_node.mean()
    weights = weights.clamp_min(0.0)
    return (weights * per_node).sum() / weights.sum().clamp_min(1e-8)


def _run_router_train(
    phase: str, variant: str, dataset: str, seed: int, device_name: str,
    mode: str, objective: str,
):
    device=torch.device(device_name)
    cache_cpu=torch.load(cache_path(dataset,seed),map_location="cpu",weights_only=False)
    cache=_move_cache(cache_cpu,device)
    c0,classifier,payload=load_c0_modules(dataset,seed,device)
    global _REGRET_C0,_REGRET_HEAD
    _REGRET_C0,_REGRET_HEAD=c0,classifier

    checkpoint=checkpoint_path(phase,variant,dataset,seed)
    directory=run_dir(phase,variant,dataset,seed)
    if (checkpoint.is_file() and (directory/"complete.marker").is_file()
            and (directory/"metrics.json").is_file()):
        return "reused_existing"

    base_seed=int(seed)+230701
    _set_seed(base_seed)
    template=UtilityRouter(int(cache["hidden_dim"]),int(cache["num_classes"]),mode)
    initial=_clone_cpu_state(template)
    del template
    router=UtilityRouter(int(cache["hidden_dim"]),int(cache["num_classes"]),mode).to(device)
    router.load_state_dict(initial,strict=True)
    optimizer=torch.optim.AdamW(router.parameters(),lr=1e-3,weight_decay=1e-4)
    train=cache["train_idx"]; val=cache["val_idx"]
    y=cache["y_allowed"]
    q_t=cache["teacher_text"][train]; q_v=cache["teacher_visual"][train]
    w_t=cache["teacher_strength_text"][train]; w_v=cache["teacher_strength_visual"][train]
    c0.eval(); classifier.eval()
    best_acc=-1.0; best_state=None; best_metrics=None; best_epoch=0
    best_regret=float("inf"); best_regret_epoch=0
    patience_left=30; history=[]

    for epoch in range(1,201):
        router.train(); optimizer.zero_grad(set_to_none=True)
        out=router_forward(router,cache)
        p_t,p_v=out["prob_text"],out["prob_visual"]
        if objective=="direct_ce":
            logits=_route_logits(c0,classifier,cache,p_t,p_v)
            loss=F.cross_entropy(logits[train],y[train])
        elif objective=="utility_kd":
            loss=utility_distillation_loss(
                p_t,p_v,cache["teacher_text"],cache["teacher_visual"],
                cache["teacher_strength_text"],cache["teacher_strength_visual"],
                train,weighted=False,
            )
        elif objective=="weighted_utility_kd":
            loss=utility_distillation_loss(
                p_t,p_v,cache["teacher_text"],cache["teacher_visual"],
                cache["teacher_strength_text"],cache["teacher_strength_visual"],
                train,weighted=True,
            )
        else:
            raise ValueError(objective)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(),1.0)
        optimizer.step()

        router.eval()
        with torch.no_grad():
            out=router_forward(router,cache)
            measured=evaluate_probs(c0,classifier,cache,out["prob_text"],out["prob_visual"])
        history.append({
            "epoch":epoch,"train_loss":float(loss.item()),
            "val_acc":measured["val_acc"],"val_macro_f1":measured["val_macro_f1"],
            "mean_routing_regret":measured["mean_routing_regret"],
        })
        if measured["mean_routing_regret"]<best_regret:
            best_regret=measured["mean_routing_regret"]; best_regret_epoch=epoch
        if measured["val_acc"]>best_acc:
            best_acc=measured["val_acc"]; best_epoch=epoch
            best_state=_clone_cpu_state(router)
            best_metrics={k:v for k,v in measured.items() if k not in {"prediction","logits"}}
            patience_left=30
        else:
            patience_left-=1
        if epoch>=30 and patience_left<=0:
            break

    if best_state is None:
        raise RuntimeError(f"no best checkpoint selected for {phase}/{variant}/{dataset}/{seed}")
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "phase":phase,"variant":variant,"mode":mode,"objective":objective,
        "seed":seed,"epoch":best_epoch,"best_regret_epoch":best_regret_epoch,
        "metrics":best_metrics,"router_state":best_state,
        "c0_checkpoint":str(c0_checkpoint(dataset,seed)),
        "c0_checkpoint_sha256":sha256(c0_checkpoint(dataset,seed)),
        "trainable_parameter_count":sum(p.numel() for p in router.parameters() if p.requires_grad),
        "test_metrics":None,
    },checkpoint)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"metrics.json").write_text(json.dumps({
        "metrics":{"val_acc":best_metrics["val_acc"],"val_macro_f1":best_metrics["val_macro_f1"]},
        "best_epoch":best_epoch,"best_regret_epoch":best_regret_epoch,
        "selection":"best_validation_accuracy",
        "routing_regret_epoch_is_analysis_only":True,
        "test_metrics":None,
    },indent=2),encoding="utf-8")
    pd.DataFrame(history).to_csv(directory/"training_history.csv",index=False)
    (directory/"complete.marker").write_text("complete\n",encoding="utf-8")
    return "trained"


def train_group_one(phase: str, dataset: str, seed: int, device: str):
    if phase=="phase_b":
        variants=list(VARIANT_SPECS)[:3]
    elif phase=="phase_c":
        variants=["C0_decision_capacity_control","C1_decision_utility_router"]
    elif phase=="phase_d":
        variants=["D0_relation_capacity_control","D1_utility_relation_router"]
    elif phase=="confirmation":
        lock=json.loads((RESULTS/"screening_winner_lock.json").read_text())
        variants=[lock["winning_variant"]] if lock["winning_variant"]!="C0_Uniform" else []
    else:
        raise ValueError(phase)
    outputs=[]
    for variant in variants:
        mode,objective=VARIANT_SPECS[variant]
        state=_run_router_train(phase,variant,dataset,seed,device,mode,objective)
        _append_manifest(phase,variant,dataset,seed,device,state)
        outputs.append((variant,state))
    return outputs


def train_groups(phase: str, datasets: list[str], devices: list[str]):
    jobs=[(ds,seed) for ds in datasets for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        pending={}
        iterator=iter(jobs)
        for device in devices:
            try: ds,seed=next(iterator)
            except StopIteration: break
            pending[pool.submit(_subprocess,["--train-group",phase,ds,str(seed),device])]=(ds,seed,device)
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                ds,seed,device=pending.pop(future)
                result=future.result()
                if result.returncode:
                    raise RuntimeError(f"training failed {phase}/{ds}/seed{seed} on {device}\n{result.stdout[-5000:]}")
                print(f"[{phase} {ds} seed={seed} {device}] complete",flush=True)
                try: nds,nseed=next(iterator)
                except StopIteration: continue
                pending[pool.submit(_subprocess,["--train-group",phase,nds,str(nseed),device])]=(nds,nseed,device)


def load_router_run(phase: str, variant: str, dataset: str, seed: int, device):
    cache = _move_cache(torch.load(cache_path(dataset, seed), map_location="cpu", weights_only=False), device)
    c0, classifier, payload = load_c0_modules(dataset, seed, device)
    mode, _ = VARIANT_SPECS[variant]
    router = UtilityRouter(int(cache["hidden_dim"]), int(cache["num_classes"]), mode).to(device)
    saved = torch.load(checkpoint_path(phase, variant, dataset, seed), map_location="cpu", weights_only=False)
    router.load_state_dict(saved["router_state"], strict=True)
    router.eval()
    global _REGRET_C0, _REGRET_HEAD
    _REGRET_C0, _REGRET_HEAD = c0, classifier
    with torch.no_grad():
        out = router_forward(router, cache)
    return cache, c0, classifier, router, out, saved


@torch.no_grad()
def analyze_variant_run(
    phase: str, variant: str, dataset: str, seed: int, device_name: str,
    *, make_interventions: bool = False, relation_interventions: bool = False,
):
    device = torch.device(device_name)
    cache, c0, classifier, router, out, saved = load_router_run(
        phase, variant, dataset, seed, device
    )
    p_t, p_v = out["prob_text"].clone(), out["prob_visual"].clone()
    normal = evaluate_probs(c0, classifier, cache, p_t, p_v)
    base_row = {
        "phase": phase, "variant": variant, "dataset": dataset, "seed": seed,
        "val_acc": normal["val_acc"], "val_macro_f1": normal["val_macro_f1"],
        "best_epoch": saved["epoch"], "best_regret_epoch": saved["best_regret_epoch"],
        "trainable_parameter_count": saved["trainable_parameter_count"],
        **{k:v for k,v in normal.items() if k not in {"prediction","logits"}},
    }
    intervention_rows = []
    val = cache["val_idx"]; train=cache["train_idx"]
    norm_pred = normal["prediction"]
    specs = []
    if make_interventions:
        pt=p_t.clone(); pv=p_v.clone()
        pt[val]=p_t[train].mean(0); pv[val]=p_v[train].mean(0)
        specs.append(("train_mean_probability",pt,pv))
        for index in range(10):
            gen=torch.Generator(device="cpu").manual_seed(seed*10007+index)
            perm=torch.randperm(int(val.numel()),generator=gen).to(device)
            pt=p_t.clone(); pv=p_v.clone()
            pt[val]=p_t[val[perm]]; pv[val]=p_v[val[perm]]
            specs.append((f"node_shuffle_{index}",pt,pv))
        pt=p_t.clone(); pv=p_v.clone()
        pt[val]=0; pv[val]=0; pt[val,4]=1; pv[val,4]=1
        specs.append(("uniform_action_fallback",pt,pv))
    if relation_interventions:
        pooled_t=out["pooled_relation_text"]
        pooled_v=out["pooled_relation_visual"]
        allowed=cache["allowed_idx"]
        train_global=allowed[train]
        val_global=allowed[val]
        mean_t=pooled_t[train_global].mean(0)
        mean_v=pooled_v[train_global].mean(0)
        override_t=pooled_t[allowed].clone(); override_v=pooled_v[allowed].clone()
        override_t[val]=mean_t; override_v[val]=mean_v
        out_mean=router(
            cache["states_text"],cache["states_visual"],
            edge_index=cache["edge_prop"],h0_text_all=cache["h0_text_all"],
            h0_visual_all=cache["h0_visual_all"],allowed_idx=allowed,
            relation_override_text=override_t,relation_override_visual=override_v,
        )
        specs.append(("train_mean_relation_context",
                      out_mean["prob_text"],out_mean["prob_visual"]))
        for index in range(10):
            gen=torch.Generator(device="cpu").manual_seed(seed*10007+index)
            perm=torch.randperm(int(val.numel()),generator=gen).to(device)
            override_t=pooled_t[allowed].clone(); override_v=pooled_v[allowed].clone()
            override_t[val]=pooled_t[val_global[perm]]
            override_v[val]=pooled_v[val_global[perm]]
            changed=router(
                cache["states_text"],cache["states_visual"],
                edge_index=cache["edge_prop"],h0_text_all=cache["h0_text_all"],
                h0_visual_all=cache["h0_visual_all"],allowed_idx=allowed,
                relation_override_text=override_t,relation_override_visual=override_v,
            )
            specs.append((f"relation_context_shuffle_{index}",
                          changed["prob_text"],changed["prob_visual"]))
    for name,pt,pv in specs:
        measured=evaluate_probs(c0,classifier,cache,pt,pv)
        delta_t=(p_t[val]-pt[val]).abs().mean()
        delta_v=(p_v[val]-pv[val]).abs().mean()
        intervention_rows.append({
            "phase":phase,"variant":variant,"dataset":dataset,"seed":seed,
            "intervention":name,
            "val_acc":measured["val_acc"],"val_macro_f1":measured["val_macro_f1"],
            "acc_drop":normal["val_acc"]-measured["val_acc"],
            "macro_f1_drop":normal["val_macro_f1"]-measured["val_macro_f1"],
            "prediction_flip_rate":float((norm_pred!=measured["prediction"]).float().mean().item()),
            "mean_abs_probability_change":float(0.5*(delta_t+delta_v).item()),
            "mean_routing_regret":measured["mean_routing_regret"],
            "mean_excess_loss_vs_uniform":measured["mean_excess_loss_vs_uniform"],
        })
    return base_row, intervention_rows


def analyze_phase_b(device_name: str):
    rows=[]; regrets=[]; interventions=[]
    for variant in list(VARIANT_SPECS)[:3]:
        for dataset in SCREEN:
            for seed in SEEDS:
                row, iv=analyze_variant_run(
                    "phase_b",variant,dataset,seed,device_name,make_interventions=True
                )
                rows.append(row); interventions.extend(iv)
                for modality in ("text","visual"):
                    regrets.append({
                        "variant":variant,"dataset":dataset,"seed":seed,
                        "modality":modality,
                        "mean_routing_regret":row[f"{modality}_mean_regret"],
                        "mean_excess_loss_vs_uniform":row[f"{modality}_mean_excess_loss_vs_uniform"],
                        "hard_action_agreement":row["hard_action_agreement"],
                        "soft_target_kl":row["soft_target_kl"],
                        "router_teacher_strength_spearman":row["router_teacher_strength_spearman"],
                        "best_epoch":row["best_epoch"],"best_regret_epoch":row["best_regret_epoch"],
                    })
    pd.DataFrame(rows).to_csv(RESULTS/"phase_b_objective_results.csv",index=False)
    pd.DataFrame(regrets).to_csv(RESULTS/"phase_b_routing_regret.csv",index=False)
    pd.DataFrame(interventions).to_csv(RESULTS/"phase_b_interventions.csv",index=False)
    return rows, interventions


def _paired_delta(rows: list[dict], candidate: str, parent: str, metric: str):
    frame=pd.DataFrame(rows)
    a=frame.loc[frame.variant==candidate,["dataset","seed",metric]].rename(columns={metric:"candidate"})
    b=frame.loc[frame.variant==parent,["dataset","seed",metric]].rename(columns={metric:"parent"})
    merged=a.merge(b,on=["dataset","seed"],validate="one_to_one")
    merged["delta"]=merged.candidate-merged.parent
    return merged


def phase_b_gate(rows: list[dict]):
    summaries=[]
    eligible=[]
    for candidate in ("B1_flat_utility_kd","B2_weighted_utility_kd"):
        regret=_paired_delta(rows,candidate,"B0_frozen_direct_ce","mean_routing_regret")
        acc=_paired_delta(rows,candidate,"B0_frozen_direct_ce","val_acc")
        f1=_paired_delta(rows,candidate,"B0_frozen_direct_ce","val_macro_f1")
        ds_reg=regret.groupby("dataset").delta.mean()
        ds_acc=acc.groupby("dataset").delta.mean()
        positive_ds=int((ds_reg<0).sum())
        positive_seed=int((regret.delta<0).sum())
        mean_acc=float(acc.delta.mean()); mean_f1=float(f1.delta.mean())
        passes=(positive_ds>=2 and positive_seed>=6 and mean_acc>=-0.001 and mean_f1>=-0.003)
        row={
            "candidate":candidate,
            "datasets_lower_regret":positive_ds,
            "positive_regret_seed_pairs":f"{positive_seed}/{len(regret)}",
            "mean_delta_regret":float(regret.delta.mean()),
            "mean_delta_acc":mean_acc,"mean_delta_macro_f1":mean_f1,
            "mean_delta_regret_by_dataset":{str(k):float(v) for k,v in ds_reg.items()},
            "mean_delta_acc_by_dataset":{str(k):float(v) for k,v in ds_acc.items()},
            "utility_aligned_objective_gate":bool(passes),
        }
        summaries.append(row)
        if passes: eligible.append(row)
    supported=bool(eligible)
    # B2 receives a separate exploratory preference-weighting tag only if its
    # Acc is non-negative on two datasets and it reduces Movies harm vs B0.
    b2_acc=_paired_delta(rows,"B2_weighted_utility_kd","B0_frozen_direct_ce","val_acc")
    b1_acc=_paired_delta(rows,"B1_flat_utility_kd","B0_frozen_direct_ce","val_acc")
    movies_b2=float(b2_acc.loc[b2_acc.dataset=="Movies","delta"].mean())
    movies_b1=float(b1_acc.loc[b1_acc.dataset=="Movies","delta"].mean())
    b2_dataset=int((b2_acc.groupby("dataset").delta.mean()>=0).sum())
    preference_promising=bool(b2_dataset>=2 and movies_b2>movies_b1)
    return {
        "status":"UTILITY_ALIGNED_OBJECTIVE_SUPPORTED" if supported else "UTILITY_ALIGNED_OBJECTIVE_NOT_SUPPORTED",
        "supported":supported,"candidate_gates":summaries,
        "preference_weighting_promising":preference_promising,
        "B2_nonnegative_acc_datasets":b2_dataset,
        "Movies_delta_acc_B1":movies_b1,"Movies_delta_acc_B2":movies_b2,
    }


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


def edge_alignment_audit(datasets: list[str], device: str):
    rows=[]
    for dataset in datasets:
        # Edge alignment is graph-level and seed-independent for these datasets.
        cfg=make_cfg(dataset,42,device)
        data=load_mag_data(cfg,"nc",42)
        raw=data.edge_index.detach().cpu().long()
        raw,_=remove_self_loops(raw)
        raw=coalesce(raw,num_nodes=int(data.num_nodes))
        pairs=set(map(tuple,raw.t().tolist()))
        reverse={(v,u) for u,v in pairs}
        eprop=build_eprop(raw,int(data.num_nodes))
        prop_pairs=set(map(tuple,eprop.t().tolist()))
        reciprocal=len(pairs & reverse)/max(len(pairs),1)
        overlap=len(pairs & prop_pairs)/max(len(pairs),1)
        rows.append({
            "dataset":dataset,"raw_unique_nonself_directed_edges":len(pairs),
            "eprop_directed_edges":len(prop_pairs),
            "raw_edge_reciprocity_ratio":reciprocal,
            "raw_relation_edge_vs_eprop_overlap":overlap,
            "edge_sets_equal":pairs==prop_pairs,
            "relation_encoder_uses":"E_prop=coalesce(to_undirected(remove_self_loops(E_raw)))",
        })
    pd.DataFrame(rows).to_csv(RESULTS/"edge_set_alignment_audit.csv",index=False)


def analyze_conditional_phase(phase: str, device: str):
    if phase=="phase_c":
        variants=["C0_decision_capacity_control","C1_decision_utility_router"]
    elif phase=="phase_d":
        variants=["D0_relation_capacity_control","D1_utility_relation_router"]
    else:
        raise ValueError(phase)
    rows=[]; interventions=[]
    for variant in variants:
        for dataset in SCREEN:
            for seed in SEEDS:
                row,iv=analyze_variant_run(
                    phase,variant,dataset,seed,device,
                    make_interventions=True,
                    relation_interventions=(phase=="phase_d" and variant=="D1_utility_relation_router"),
                )
                rows.append(row); interventions.extend(iv)
    table=pd.DataFrame(rows)
    ivframe=pd.DataFrame(interventions)
    if phase=="phase_c":
        table.to_csv(RESULTS/"phase_c_decision_results.csv",index=False)
        ivframe.to_csv(RESULTS/"phase_c_interventions.csv",index=False)
        parent,candidate="C0_decision_capacity_control","C1_decision_utility_router"
    else:
        table.to_csv(RESULTS/"phase_d_relation_results.csv",index=False)
        ivframe.to_csv(RESULTS/"phase_d_interventions.csv",index=False)
        parent,candidate="D0_relation_capacity_control","D1_utility_relation_router"
    acc=_paired_delta(rows,candidate,parent,"val_acc")
    f1=_paired_delta(rows,candidate,parent,"val_macro_f1")
    regret=_paired_delta(rows,candidate,parent,"mean_routing_regret")
    params=table.groupby("variant").trainable_parameter_count.first().to_dict()
    p_diff=100*abs(params[candidate]-params[parent])/max(params[candidate],params[parent],1)
    if p_diff>5.0:
        raise RuntimeError(f"capacity control differs by {p_diff:.3f}% (>5%) in {phase}")
    capacity_rows=[]
    for _,row in acc.iterrows():
        key=(row.dataset,int(row.seed))
        fval=float(f1.loc[(f1.dataset==key[0])&(f1.seed==key[1]),"delta"].iloc[0])
        rval=float(regret.loc[(regret.dataset==key[0])&(regret.seed==key[1]),"delta"].iloc[0])
        capacity_rows.append({
            "dataset":key[0],"seed":key[1],"candidate":candidate,"parent":parent,
            "delta_val_acc":float(row.delta),"delta_macro_f1":fval,
            "delta_routing_regret":rval,
            "candidate_trainable_parameters":params[candidate],
            "control_trainable_parameters":params[parent],
            "parameter_diff_pct":p_diff,
        })
    capacity=pd.DataFrame(capacity_rows)
    if phase=="phase_c":
        capacity.to_csv(RESULTS/"phase_c_capacity_control.csv",index=False)
    else:
        capacity.to_csv(RESULTS/"phase_d_capacity_control.csv",index=False)
    ds_acc=acc.groupby("dataset").delta.mean()
    ds_reg=regret.groupby("dataset").delta.mean()
    pos_seed=int((acc.delta>0).sum())
    positive_ds=int((ds_acc>0).sum())
    regret_ds=int((ds_reg<0).sum())
    regret_seed=int((regret.delta<0).sum())
    mean_acc=float(acc.delta.mean())
    mean_f1=float(f1.delta.mean())
    mean_reg=float(regret.delta.mean())
    # Use the same breadth thresholds as the declared screening gate; the sign
    # and effect are also reported to avoid reducing a conditional claim to a label.
    gate=bool(positive_ds>=2 and pos_seed>=6 and regret_ds>=2 and regret_seed>=6 and mean_acc>0 and mean_reg<0)
    intervention_name="relation_context_shuffle_"
    shuffle=ivframe.loc[ivframe.intervention.astype(str).str.startswith(intervention_name)]
    shuffle_acc=float(shuffle.acc_drop.mean()) if len(shuffle) else float("nan")
    shuffle_f1=float(shuffle.macro_f1_drop.mean()) if len(shuffle) else float("nan")
    if phase=="phase_d":
        gate=gate and (shuffle_acc>0 or shuffle_f1>0)
    gate_summary={
        "phase":phase,"candidate":candidate,"capacity_control":parent,
        "mean_delta_acc":mean_acc,"mean_delta_macro_f1":mean_f1,
        "mean_delta_regret":mean_reg,"positive_acc_datasets":positive_ds,
        "positive_acc_seed_pairs":f"{pos_seed}/{len(acc)}",
        "lower_regret_datasets":regret_ds,
        "lower_regret_seed_pairs":f"{regret_seed}/{len(regret)}",
        "trainable_parameter_diff_pct":p_diff,
        "mean_relation_context_shuffle_acc_drop":shuffle_acc,
        "mean_relation_context_shuffle_macro_f1_drop":shuffle_f1,
        "gate_pass":gate,
    }
    _write_json(RESULTS/f"{phase}_gate.json",gate_summary)
    return rows,gate_summary


def _load_phase_rows(phase: str):
    if phase=="phase_b":
        return pd.read_csv(RESULTS/"phase_b_objective_results.csv").to_dict("records")
    if phase=="phase_c":
        return pd.read_csv(RESULTS/"phase_c_decision_results.csv").to_dict("records")
    if phase=="phase_d":
        return pd.read_csv(RESULTS/"phase_d_relation_results.csv").to_dict("records")
    raise ValueError(phase)


def phase_b_historical_comparison(rows: list[dict]):
    # E4 HierarchicalFlat is a historical screen reference only. In E4's
    # published result schema it lives in phase_b_results.csv as B0_hierarchical_flat.
    e4_path=ROOT/"results/routing_audit_v1/phase_b_results.csv"
    if not e4_path.is_file():
        return {"status":"NOT_TESTED","reason":"E4 screening result table missing"}
    e4=pd.read_csv(e4_path)
    if "model_variant" not in e4.columns or "dataset" not in e4.columns:
        return {"status":"NOT_TESTED","reason":"E4 result schema lacks model_variant/dataset"}
    old=e4.loc[
        (e4.model_variant=="hierarchical_flat_simplex")
        & (e4.dataset.isin(SCREEN))
    ]
    if old.empty or not {"seed","val_acc","val_macro_f1"}.issubset(old.columns):
        return {"status":"NOT_TESTED","reason":"E4 HierarchicalFlat screening rows missing"}
    c0_rows=[]
    for dataset in SCREEN:
        for seed in SEEDS:
            p=c0_metrics(dataset,seed)
            m=json.loads(p.read_text())["metrics"]
            c0_rows.append({
                "dataset":dataset,"seed":seed,
                "val_acc":float(m["val_acc"]["mean"]),
                "val_macro_f1":float(m["val_macro_f1"]["mean"]),
            })
    c0=pd.DataFrame(c0_rows)
    old=old[["dataset","seed","val_acc","val_macro_f1"]].merge(
        c0,on=["dataset","seed"],suffixes=("_E4","_C0"),validate="one_to_one"
    )
    old["E4_delta_acc_vs_C0"]=old.val_acc_E4-old.val_acc_C0
    old["E4_delta_f1_vs_C0"]=old.val_macro_f1_E4-old.val_macro_f1_C0
    old=old[["dataset","seed","E4_delta_acc_vs_C0","E4_delta_f1_vs_C0"]]
    b0=pd.DataFrame([r for r in rows if r["variant"]=="B0_frozen_direct_ce"])
    b0c0=b0.merge(c0,on=["dataset","seed"],suffixes=("_B0","_C0"),validate="one_to_one")
    b0c0["B0_delta_acc_vs_C0"]=b0c0.val_acc_B0-b0c0.val_acc_C0
    b0c0["B0_delta_f1_vs_C0"]=b0c0.val_macro_f1_B0-b0c0.val_macro_f1_C0
    joined=b0c0.merge(old,on=["dataset","seed"],validate="one_to_one")
    joined["B0_minus_E4_delta_acc"]=joined.B0_delta_acc_vs_C0-joined.E4_delta_acc_vs_C0
    joined["B0_minus_E4_delta_f1"]=joined.B0_delta_f1_vs_C0-joined.E4_delta_f1_vs_C0
    joined.to_csv(RESULTS/"phase_b_vs_e4_historical.csv",index=False)
    better=int((joined.B0_minus_E4_delta_acc>0).sum())
    mean_acc=float(joined.B0_minus_E4_delta_acc.mean())
    return {
        "status":"MIXED_SUPPORT" if (better>=6 or mean_acc>0) else "NO_SUPPORT",
        "positive_paired_seeds":f"{better}/{len(joined)}",
        "mean_B0_minus_E4_acc_effect":mean_acc,
        "mean_B0_minus_E4_f1_effect":float(joined.B0_minus_E4_delta_f1.mean()),
        "note":"E4 B0 HierarchicalFlat vs C0 and E5 B0 frozen Direct CE vs C0, paired on the same screening dataset/seed; descriptive historical contrast.",
    }


def run_phase_e(device: str, phase_b_rows: list[dict], b_gate: dict,
                phase_c_gate: dict | None, phase_d_gate: dict | None):
    # The best utility-KD model is selected on screening Validation Accuracy,
    # with routing regret as a tie-break; the safe transform uses no fitted threshold.
    kd=pd.DataFrame([r for r in phase_b_rows if r["variant"] in {
        "B1_flat_utility_kd","B2_weighted_utility_kd"
    }])
    if kd.empty:
        rows=[{"status":"SKIPPED_NO_KD_ROUTER","reason":"No trained utility KD candidate"}]
        pd.DataFrame(rows).to_csv(RESULTS/"phase_e_safe_routing.csv",index=False)
        return None,rows
    ranking=kd.groupby("variant").agg(
        mean_val_acc=("val_acc","mean"),
        mean_routing_regret=("mean_routing_regret","mean"),
    ).reset_index().sort_values(
        ["mean_val_acc","mean_routing_regret"],ascending=[False,True]
    )
    selected=str(ranking.iloc[0].variant)
    variant_candidates=[selected]
    if b_gate["supported"]:
        for gate,variant in ((phase_c_gate,"C1_decision_utility_router"),
                             (phase_d_gate,"D1_utility_relation_router")):
            if gate and gate.get("gate_pass"):
                variant_candidates.append(variant)
    # If C1/D1 was eligible, compare by mean Val Acc among the eligible utility candidates.
    for candidate in variant_candidates[1:]:
        frame=pd.read_csv(RESULTS/(
            "phase_c_decision_results.csv" if candidate.startswith("C1_")
            else "phase_d_relation_results.csv"
        ))
        kd=pd.concat([kd,frame.loc[frame.variant==candidate]],ignore_index=True)
    means=kd.groupby("variant").val_acc.mean()
    selected=str(means.idxmax())

    output=[]
    for dataset in SCREEN:
        for seed in SEEDS:
            phase="phase_b" if selected.startswith("B") else ("phase_c" if selected.startswith("C1") else "phase_d")
            cache,c0,classifier,router,router_out,saved=load_router_run(phase,selected,dataset,seed,torch.device(device))
            p_t=router_out["prob_text"]; p_v=router_out["prob_visual"]
            normal=evaluate_probs(c0,classifier,cache,p_t,p_v)
            safe_t=safe_probabilities(p_t); safe_v=safe_probabilities(p_v)
            safe=evaluate_probs(c0,classifier,cache,safe_t,safe_v)
            val=cache["val_idx"]
            one_t=torch.zeros_like(p_t); one_t[:,4]=1
            one_v=torch.zeros_like(p_v); one_v[:,4]=1
            uniform=evaluate_probs(c0,classifier,cache,one_t,one_v)
            for method,result,pt,pv in (
                ("normal_router",normal,p_t,p_v),
                ("safe_router",safe,safe_t,safe_v),
                ("C0_Uniform",uniform,one_t,one_v),
            ):
                output.append({
                    "variant":selected,"dataset":dataset,"seed":seed,"method":method,
                    "val_acc":result["val_acc"],"val_macro_f1":result["val_macro_f1"],
                    "delta_acc_vs_uniform":result["val_acc"]-uniform["val_acc"],
                    "delta_macro_f1_vs_uniform":result["val_macro_f1"]-uniform["val_macro_f1"],
                    "mean_routing_regret":result["mean_routing_regret"],
                    "mean_excess_loss_vs_uniform":result["mean_excess_loss_vs_uniform"],
                    "mean_abs_probability_change_vs_normal":float(0.5*(
                        (pt[val]-p_t[val]).abs().mean()+(pv[val]-p_v[val]).abs().mean()
                    ).item()),
                })
    frame=pd.DataFrame(output)
    frame.to_csv(RESULTS/"phase_e_safe_routing.csv",index=False)
    table=frame.groupby(["dataset","method"]).agg(
        val_acc=("val_acc","mean"),
        val_macro_f1=("val_macro_f1","mean"),
        delta_acc_vs_uniform=("delta_acc_vs_uniform","mean"),
        delta_macro_f1_vs_uniform=("delta_macro_f1_vs_uniform","mean"),
        mean_routing_regret=("mean_routing_regret","mean"),
    ).reset_index()
    pivot=table.pivot(index="dataset",columns="method",values="delta_acc_vs_uniform")
    movies=bool(
        "Movies" in pivot.index
        and pivot.loc["Movies","safe_router"]>pivot.loc["Movies","normal_router"]
    )
    retained=True
    for dataset in ("ele-fashion","Reddit-S"):
        if dataset in pivot.index:
            norm=float(pivot.loc[dataset,"normal_router"])
            safe=float(pivot.loc[dataset,"safe_router"])
            if norm>0:
                retained=retained and safe>=0.8*norm and safe>=0
    supported=bool(movies and retained)
    metadata={
        "selected_variant":selected,"phase":"phase_b" if selected.startswith("B") else ("phase_c" if selected.startswith("C1") else "phase_d"),
        "conservative_personalization_supported":supported,
        "movies_negative_transfer_reduced":movies,
        "positive_dataset_gains_retained":bool(retained),
        "selected_by":"highest screening mean Val Acc among utility-aligned candidates; regret tie-break",
    }
    _write_json(RESULTS/"phase_e_summary.json",metadata)
    return metadata,output


def screening_decisions(phase_b_rows, b_gate, historical, phase_c_gate, phase_d_gate, phase_e):
    action=pd.read_csv(RESULTS/"action_landscape_summary.csv")
    val=action.loc[action.split=="validation"]
    ds_land=val.groupby("dataset").agg(
        mean_preference_strength=("mean_preference_strength","mean"),
        mean_oracle_ce_gain=("mean_oracle_CE_gain_vs_Uniform","mean"),
        mean_oracle_acc_gain=("oracle_accuracy_gain_vs_Uniform","mean"),
        diverse_action_count=("max_action_best_frequency","count"),
    )
    h1_status=(
        "STRONG_SUPPORT" if (ds_land.mean_preference_strength.mean()>=0.1
                              and int((ds_land.mean_oracle_ce_gain>0.01).sum())>=2)
        else ("MIXED_SUPPORT" if int((ds_land.mean_oracle_ce_gain>0).sum())>=1 else "NO_SUPPORT")
    )
    rows=[{
        "hypothesis":"H1_action_landscape_nontrivial","status":h1_status,
        "evidence":json.dumps(ds_land.mean().to_dict()),
        "counterevidence":"Per-node oracle is label-informed and only an upper bound.",
    },{
        "hypothesis":"H2_frozen_routing_reduces_coadaptation_problem",
        "status":historical["status"],"evidence":json.dumps(historical),
        "counterevidence":"B0 is compared to E4's Validation-selected joint router only on the fixed screening panel.",
    }]
    aligned_status="STRONG_SUPPORT" if b_gate["supported"] else (
        "MIXED_SUPPORT" if any(x["mean_delta_regret"]<0 for x in b_gate["candidate_gates"])
        else "NO_SUPPORT"
    )
    rows.append({
        "hypothesis":"H3_utility_distillation_improves_routing_alignment",
        "status":aligned_status,"evidence":json.dumps(b_gate),
        "counterevidence":"Regret reduction and task metrics are reported separately.",
    })
    rows.append({
        "hypothesis":"H4_preference_weighting_helps_selective_personalization",
        "status":"STRONG_SUPPORT" if b_gate["preference_weighting_promising"] else (
            "MIXED_SUPPORT" if b_gate["B2_nonnegative_acc_datasets"]>0 else "NO_SUPPORT"
        ),
        "evidence":json.dumps({k:v for k,v in b_gate.items() if k!="candidate_gates"}),
        "counterevidence":"Preference weighting is not called useful when regret improves without task benefit.",
    })
    rows.append({
        "hypothesis":"H5_decision_response_improves_utility_routing",
        "status":"NOT_TESTED" if phase_c_gate is None else (
            "STRONG_SUPPORT" if phase_c_gate["gate_pass"] else "NO_SUPPORT"
        ),
        "evidence":json.dumps(phase_c_gate) if phase_c_gate else "Phase B gate failed.",
        "counterevidence":"Capacity-matched C1 vs C0 is required.",
    })
    rows.append({
        "hypothesis":"H6_relation_context_benefits_from_utility_alignment",
        "status":"NOT_TESTED" if phase_d_gate is None else (
            "STRONG_SUPPORT" if phase_d_gate["gate_pass"] else "NO_SUPPORT"
        ),
        "evidence":json.dumps(phase_d_gate) if phase_d_gate else "Phase B gate failed; Phase D is conditional.",
        "counterevidence":"D1 uses E_prop without loops; edge context cannot alter propagation.",
    })
    rows.append({
        "hypothesis":"H7_conservative_personalization_reduces_negative_transfer",
        "status":"NOT_TESTED" if phase_e is None else (
            "STRONG_SUPPORT" if phase_e["conservative_personalization_supported"] else "NO_SUPPORT"
        ),
        "evidence":json.dumps(phase_e) if phase_e else "No utility router available for Phase E.",
        "counterevidence":"Confidence blend has no fitted threshold; report per-dataset results.",
    })
    pd.DataFrame(rows).to_csv(RESULTS/"screening_decision_matrix.csv",index=False)
    return rows


def freeze_screening_winner(phase_b_rows, b_gate, phase_c_gate, phase_d_gate, phase_e):
    eligible=[]
    if b_gate["supported"]:
        for item in b_gate["candidate_gates"]:
            if item["utility_aligned_objective_gate"]:
                eligible.append(item["candidate"])
        for phase_gate,variant in ((phase_c_gate,"C1_decision_utility_router"),
                                   (phase_d_gate,"D1_utility_relation_router")):
            if phase_gate and phase_gate.get("gate_pass"):
                eligible.append(variant)
    if not eligible:
        winner="C0_Uniform"
        use_safe=False
        objective="frozen_reference"
        mode="C0_uniform"
        rationale="No routing candidate met the preregistered objective and performance gates; keep frozen C0 Uniform as incumbent."
    else:
        rows=pd.DataFrame(phase_b_rows)
        phase_map={"B0":"phase_b","B1":"phase_b","B2":"phase_b","C1":"phase_c","D1":"phase_d"}
        metrics=[]
        for variant in eligible:
            if variant.startswith("B"):
                frame=rows.loc[rows.variant==variant]
            else:
                phase="phase_c" if variant.startswith("C1") else "phase_d"
                file="phase_c_decision_results.csv" if phase=="phase_c" else "phase_d_relation_results.csv"
                frame=pd.read_csv(RESULTS/file).loc[lambda f:f.variant==variant]
            metrics.append((variant,float(frame.val_acc.mean()),float(frame.mean_routing_regret.mean())))
        metrics.sort(key=lambda x:(x[1],-x[2]),reverse=True)
        winner=metrics[0][0]
        spec=VARIANT_SPECS[winner]
        mode,objective=spec
        phase="phase_b" if winner.startswith("B") else ("phase_c" if winner.startswith("C1") else "phase_d")
        use_safe=bool(
            phase_e is not None
            and phase_e.get("conservative_personalization_supported")
            and phase_e.get("selected_variant")==winner
        )
        rationale={
            "eligible_candidates":eligible,
            "validation_accuracy_and_regret_ranking":metrics,
            "objective_gate":"passed",
            "safe_routing_selected":use_safe,
        }
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    lock={
        "experiment":"E5 Utility-Aligned Structural Routing Audit",
        "winning_variant":winner,
        "winning_architecture":mode,
        "objective":objective,
        "safe_routing_used":use_safe,
        "fixed_hyperparameters":{
            "screening_datasets":SCREEN,"seeds":SEEDS,
            "hidden_dim":256,"max_order":3,"actions":ACTION_NAMES,
            "flat_state_projection_dim":32,"context_dim":128,
            "router_hidden_dim":64,"router_output_dim":5,
            "optimizer":"AdamW","learning_rate":1e-3,"weight_decay":1e-4,
            "max_epochs":200,"patience":30,"grad_clip":1.0,
            "selection":"best Validation Accuracy",
            "utility_teacher":"softmax(-CE), no temperature",
            "preference_strength":"1-H(q)/log(5)",
        },
        "git_commit":head,
        "confirmation_datasets_not_loaded_before_lock":True,
        "decision_rationale":rationale,
        "phase_b_gate":b_gate,
        "phase_c_gate":phase_c_gate,
        "phase_d_gate":phase_d_gate,
        "phase_e":phase_e,
    }
    _write_json(RESULTS/"screening_winner_lock.json",lock)
    return lock


def _preflight_confirmation(devices: list[str]):
    jobs=[(ds,seed) for ds in CONFIRM for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        pending={}
        iterator=iter(jobs)
        for device in devices:
            try: ds,seed=next(iterator)
            except StopIteration: break
            pending[pool.submit(_subprocess,["--preflight-one",ds,str(seed),device])]=(ds,seed,device)
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                ds,seed,device=pending.pop(future)
                result=future.result()
                if result.returncode:
                    raise RuntimeError(f"confirmation C0 preflight failed: {ds}/seed{seed}\n{result.stdout[-4000:]}")
                print(f"[confirmation C0 {ds} seed={seed}] ready after winner lock",flush=True)
                try: nds,nseed=next(iterator)
                except StopIteration: continue
                pending[pool.submit(_subprocess,["--preflight-one",nds,str(nseed),device])]=(nds,nseed,device)


def run_confirmation(devices: list[str]):
    lock_path=RESULTS/"screening_winner_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("refusing to access Toys/Grocery before screening_winner_lock.json")
    lock=json.loads(lock_path.read_text())
    _preflight_confirmation(devices)
    if lock["winning_variant"]!="C0_Uniform":
        train_groups("confirmation",CONFIRM,devices)
    rows=[]; analysis_rows=[]
    for dataset in CONFIRM:
        for seed in SEEDS:
            check=json.loads((OUTPUT/"preflight_checks"/f"{dataset}_seed{seed}.json").read_text())
            c0row={
                "dataset":dataset,"seed":seed,"variant":"C0_Uniform",
                "source":"reused_E3_C0_checkpoint",
                "val_acc":check["reloaded_val_acc"],
                "val_macro_f1":check["reloaded_val_macro_f1"],
                "mean_routing_regret":float("nan"),
                "best_epoch":None,
            }
            rows.append(c0row)
            if lock["winning_variant"]=="C0_Uniform":
                cand=dict(c0row); cand["variant"]="screening_winner"; cand["source"]="reused_E3_C0_checkpoint"
                rows.append(cand)
                analysis_rows.append({
                    "dataset":dataset,"seed":seed,"candidate":"C0_Uniform",
                    "delta_val_acc":0.0,"delta_macro_f1":0.0,
                    "delta_routing_regret":float("nan"),
                })
                continue
            variant=lock["winning_variant"]
            phase="confirmation"
            device=devices[(CONFIRM.index(dataset)*len(SEEDS)+(seed-42)//1)%len(devices)]
            cache,c0,classifier,router,out,saved=load_router_run(
                phase,variant,dataset,seed,torch.device(device)
            )
            pt,pv=out["prob_text"],out["prob_visual"]
            if lock["safe_routing_used"]:
                pt=safe_probabilities(pt); pv=safe_probabilities(pv)
            measured=evaluate_probs(c0,classifier,cache,pt,pv)
            # Build exact all-node-in-cache AU action distributions.
            p0=torch.zeros_like(out["prob_text"]); p0[:,4]=1.0
            uniform_eval=evaluate_probs(c0,classifier,cache,p0,p0)
            candidate={
                "dataset":dataset,"seed":seed,"variant":"screening_winner",
                "source":"winner_checkpoint","winning_variant":variant,
                "safe_routing_used":lock["safe_routing_used"],
                "val_acc":measured["val_acc"],"val_macro_f1":measured["val_macro_f1"],
                "mean_routing_regret":measured["mean_routing_regret"],
                "mean_excess_loss_vs_uniform":measured["mean_excess_loss_vs_uniform"],
                "best_epoch":saved["epoch"],"best_regret_epoch":saved["best_regret_epoch"],
            }
            rows.append(candidate)
            analysis_rows.append({
                "dataset":dataset,"seed":seed,"candidate":variant,
                "delta_val_acc":measured["val_acc"]-uniform_eval["val_acc"],
                "delta_macro_f1":measured["val_macro_f1"]-uniform_eval["val_macro_f1"],
                "delta_routing_regret":measured["mean_routing_regret"]-uniform_eval["mean_routing_regret"],
            })
    pd.DataFrame(rows).to_csv(RESULTS/"confirmation_results.csv",index=False)
    paired=pd.DataFrame(analysis_rows)
    paired.to_csv(RESULTS/"confirmation_paired_seed_deltas.csv",index=False)
    grouped=[]
    for (dataset,candidate),part in paired.groupby(["dataset","candidate"],sort=True):
        grouped.append({
            "dataset":dataset,"candidate":candidate,
            "paired_seeds":int(part.seed.nunique()),
            "mean_delta_val_acc":float(part.delta_val_acc.mean()),
            "mean_delta_macro_f1":float(part.delta_macro_f1.mean()),
            "mean_delta_routing_regret":float(part.delta_routing_regret.mean()),
            "positive_acc_seed_pairs":int((part.delta_val_acc>0).sum()),
            "lower_regret_seed_pairs":int((part.delta_routing_regret<0).sum()),
            "interpretation":("NOT_TESTED_C0_IDENTITY" if candidate=="C0_Uniform" else "ROUTER_VS_C0"),
        })
    pd.DataFrame(grouped).to_csv(RESULTS/"confirmation_analysis.csv",index=False)


def build_five_dataset_winner_summary(lock):
    winner=lock["winning_variant"]
    screening=pd.read_csv(RESULTS/"phase_e_safe_routing.csv")
    if winner=="C0_Uniform":
        selected=screening.loc[screening.method=="C0_Uniform"].copy()
        selected["source"]="reused_E3_C0_checkpoint"
    else:
        method="safe_router" if lock["safe_routing_used"] else "normal_router"
        selected=screening.loc[(screening.variant==winner)&(screening.method==method)].copy()
        selected["source"]="screening_winner_checkpoint"
    selected=selected.rename(columns={"val_acc":"winner_val_acc","val_macro_f1":"winner_val_macro_f1"})
    selected=selected[["dataset","seed","winner_val_acc","winner_val_macro_f1","source"]]
    confirm=pd.read_csv(RESULTS/"confirmation_results.csv")
    confirm=confirm.loc[confirm.variant=="screening_winner"].copy()
    confirm=confirm.rename(columns={"val_acc":"winner_val_acc","val_macro_f1":"winner_val_macro_f1"})
    confirm=confirm[["dataset","seed","winner_val_acc","winner_val_macro_f1","source"]]
    paired=pd.concat([selected,confirm],ignore_index=True)
    paired.insert(2,"winning_variant",winner)
    paired.to_csv(RESULTS/"five_dataset_winner_seed_results.csv",index=False)
    summary=paired.groupby(["dataset","winning_variant"],as_index=False).agg(
        paired_seeds=("seed","nunique"),
        mean_val_acc=("winner_val_acc","mean"),
        mean_macro_f1=("winner_val_macro_f1","mean"),
        sources=("source",lambda x:";".join(sorted(set(x)))),
    )
    summary.to_csv(RESULTS/"five_dataset_winner_summary.csv",index=False)
    return summary


def finalize_all(phase_b_rows, b_gate, historical, phase_c_gate, phase_d_gate, phase_e):
    matrix=pd.read_csv(RESULTS/"screening_decision_matrix.csv")
    conf_path=RESULTS/"confirmation_analysis.csv"
    paired_path=RESULTS/"confirmation_paired_seed_deltas.csv"
    if paired_path.is_file() or conf_path.is_file():
        conf=pd.read_csv(paired_path if paired_path.is_file() else conf_path)
        if conf.candidate.ne("C0_Uniform").any():
            selected=conf.loc[conf.candidate!="C0_Uniform"]
            ds=selected.groupby("dataset").agg(
                dacc=("delta_val_acc","mean"),df1=("delta_macro_f1","mean"),
                dreg=("delta_routing_regret","mean"),
            )
            pos_seed=int((selected.delta_val_acc>0).sum())
            if len(ds)>=2 and int((ds.dacc>=0).sum())==2 and pos_seed>=4 and bool((ds.df1>=-0.003).all()) and bool((ds.dreg<0).all()):
                h8="STRONG_SUPPORT"
            elif float(selected.delta_val_acc.mean())>=0 or bool((ds.dreg<0).all()):
                h8="MIXED_SUPPORT"
            else:
                h8="NO_SUPPORT"
            evidence={
                "dataset_means":ds.to_dict(orient="index"),
                "positive_seed_pairs":f"{pos_seed}/{len(selected)}",
                "mean_delta_val_acc":float(selected.delta_val_acc.mean()),
                "mean_delta_macro_f1":float(selected.delta_macro_f1.mean()),
                "mean_delta_routing_regret":float(selected.delta_routing_regret.mean()),
            }
        else:
            h8="NOT_TESTED"
            evidence={"reason":"C0 Uniform was locked as incumbent; no learned router confirmation candidate."}
    else:
        h8="NOT_TESTED"; evidence={"reason":"Untouched confirmation has not been run."}
    matrix=matrix.loc[matrix.hypothesis!="H8_untouched_dataset_confirmation"].copy()
    matrix=pd.concat([matrix,pd.DataFrame([{
        "hypothesis":"H8_untouched_dataset_confirmation",
        "status":h8,"evidence":json.dumps(evidence),
        "counterevidence":"Toys and Grocery were accessed only after the winner lock.",
    }])],ignore_index=True)
    matrix.to_csv(RESULTS/"screening_decision_matrix.csv",index=False)
    winner_lock=json.loads((RESULTS/"screening_winner_lock.json").read_text())
    five_dataset_summary=build_five_dataset_winner_summary(winner_lock)
    summary={
        "experiment":"E5 — Utility-Aligned Structural Routing Audit",
        "status":"complete","branch":"utility_routing_audit",
        "starting_commit":"12e52f948aab8944dc71eb342d091ef3266a53f3",
        "final_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "screening_datasets":SCREEN,"confirmation_datasets":CONFIRM,"seeds":SEEDS,
        "phase_b_gate":b_gate,"historical_e4_comparison":historical,
        "phase_c_gate":phase_c_gate,"phase_d_gate":phase_d_gate,
        "phase_e":phase_e,
        "winner_lock":winner_lock,
        "five_dataset_winner_summary":five_dataset_summary.to_dict(orient="records"),
        "decision_matrix":matrix.to_dict(orient="records"),
        "run_counts":{},
        "test_labels_indexed":False,
        "c0_retrained":False,
    }
    manifest_path=OUTPUT/"run_manifest.csv"
    if manifest_path.is_file():
        m=pd.read_csv(manifest_path)
        summary["run_counts"]={
            "trained":int((m.state=="trained").sum()),
            "reused_existing":int((m.state=="reused_existing").sum()),
            "failed":int((m.state=="failed").sum()),
            "attempts":int(m.state.isin(["trained","failed"]).sum()),
            "by_phase":m.groupby(["phase","state"]).size().unstack(fill_value=0).to_dict(orient="index"),
        }
    (RESULTS/"utility_routing_audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    write_final_report(summary)
    return summary


def write_final_report(summary):
    lines=[
        "# Experiment 5 — Utility-Aligned Structural Routing Audit",
        "",
        f"- Branch: utility_routing_audit; HEAD: {summary['final_head']}; starting commit: {summary['starting_commit']}.",
        "- NC only; screen results use Movies, ele-fashion, Reddit-S (seeds 42/43/44).",
        "- No C0 retraining; all routing objectives use Train labels only. Validation is used for selection/audit; Test is not indexed.",
        "- Toys and Grocery were not loaded until screening_winner_lock.json existed.",
        f"- Frozen winner: {summary['winner_lock']['winning_variant']} ({summary['winner_lock']['winning_architecture']}); safe routing={summary['winner_lock']['safe_routing_used']}.",
        f"- Phase B objective gate: {summary['phase_b_gate']['status']}.",
        "",
        "## C0 reuse and action landscape",
        "",
        "See preflight_report.md for all 9 checkpoint reproduction checks. Raw action rows are gzip-compressed under outputs/utility_routing_audit_v1/action_landscape/.",
        "",
        "Action best frequencies, preference entropy/strength, oracle CE gain and label-oracle accuracy upper bound are in action_landscape_summary.csv; cross-seed hard-action agreement, soft-q JS/cosine and preference-strength Spearman are in action_stability.csv.",
        "",
        "The label-oracle accuracy is an upper bound and is not a deployable result.",
        "",
        "## Phase B",
        "",
        "| Variant | Dataset | Mean Val Acc | Mean Macro-F1 | Mean Routing Regret | Excess Loss vs AU |",
        "|---|---|---:|---:|---:|---:|",
    ]
    b=pd.read_csv(RESULTS/"phase_b_objective_results.csv")
    for (variant,dataset),part in b.groupby(["variant","dataset"]):
        lines.append(
            f"| {variant} | {dataset} | {part.val_acc.mean():.4f} | {part.val_macro_f1.mean():.4f} | "
            f"{part.mean_routing_regret.mean():.4f} | {part.mean_excess_loss_vs_uniform.mean():.4f} |"
        )
    lines += [
        "",
        "Paired routing-regret and objective gates are in phase_b_routing_regret.csv and the screening decision matrix. Node shuffles, train-mean probabilities and AU fallback are in phase_b_interventions.csv.",
        "",
        f"B0 vs E4 historical comparison: {json.dumps(summary['historical_e4_comparison'])}.",
        "",
        f"Preference weighting gate: {json.dumps(summary['phase_b_gate']['preference_weighting_promising'])}.",
        "",
    ]
    for phase,key,filename in (
        ("Phase C","phase_c_gate","phase_c_decision_results.csv"),
        ("Phase D","phase_d_gate","phase_d_relation_results.csv"),
    ):
        gate=summary.get(key)
        if gate is None:
            lines += [f"## {phase}", "", "NOT_TESTED because its predeclared Phase B gate did not pass.", ""]
        else:
            intervention_file = "phase_c_interventions.csv" if phase == "Phase C" else "phase_d_interventions.csv"
            lines += [f"## {phase}", "", f"Gate: {json.dumps(gate)}.", f"Per-run/capacity comparisons: {filename}; interventions are in {intervention_file}.", ""]
    if summary.get("phase_d_gate") is not None:
        lines += [
            "Edge reciprocity and raw-to-E_prop overlap are in edge_set_alignment_audit.csv. D1 relation tokens use E_prop without self-loops and never modify propagation.",
            "",
        ]
    if summary.get("phase_e"):
        lines += [
            "## Phase E conservative routing",
            "",
            f"Selection/audit: {json.dumps(summary['phase_e'])}. Per-run normal/safe/C0 values and regret are in phase_e_safe_routing.csv.",
            "",
        ]
    else:
        lines += ["## Phase E conservative routing","","No utility-aligned router passed the Phase B gate; the phase was not used for model promotion.",""]
    lines += [
        "## Screening decision matrix",
        "",
        "| Hypothesis | Status | Evidence |",
        "|---|---|---|",
    ]
    for r in summary["decision_matrix"]:
        ev=str(r.get("evidence",""))
        if len(ev)>500: ev=ev[:497]+"..."
        lines.append(f"| {r['hypothesis']} | {r['status']} | {ev} |")
    five_path=RESULTS/"five_dataset_winner_summary.csv"
    if five_path.is_file():
        five=pd.read_csv(five_path)
        lines += [
            "",
            "## Five-dataset winner summary (after confirmation)",
            "",
            "These are per-dataset Validation metrics for the locked winner; the cross-dataset aggregate did not affect winner selection.",
            "",
            "| Dataset | Paired seeds | Mean Val Acc | Mean Macro-F1 |",
            "|---|---:|---:|---:|",
        ]
        for _,row in five.iterrows():
            lines.append(
                f"| {row.dataset} | {int(row.paired_seeds)} | {row.mean_val_acc:.4f} | {row.mean_macro_f1:.4f} |"
            )
        lines.append("")
    if (RESULTS/"confirmation_analysis.csv").is_file():
        conf=pd.read_csv(RESULTS/"confirmation_analysis.csv")
        lines += [
            "## Untouched Toys/Grocery confirmation",
            "",
            "The winner lock was written before any Toys/Grocery features, labels, checkpoint metrics, or predictions were loaded.",
            "",
            "| Dataset | Paired seeds | Mean Δ Val Acc | Mean Δ Macro-F1 | Mean Δ Routing Regret | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for _,row in conf.iterrows():
            regret=("N/A" if pd.isna(row.mean_delta_routing_regret) else f"{row.mean_delta_routing_regret:.4f}")
            interpretation=("NOT_TESTED: locked winner is C0 identity" if row.candidate=="C0_Uniform" else "paired against C0")
            lines.append(
                f"| {row.dataset} | {int(row.paired_seeds)} | {row.mean_delta_val_acc:.4f} | "
                f"{row.mean_delta_macro_f1:.4f} | {regret} | {interpretation} |"
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "Results distinguish frozen direct task routing, unweighted utility distillation, preference-weighted distillation, decision-response evidence, corrected physical-edge relation evidence, and conservative AU fallback. Conditional phases not run are marked NOT_TESTED in the decision matrix.",
        "",
        "This audit does not implement MoE experts, signed filters, cross-modal routing, common/private decomposition, topology changes, or test-set evaluation.",
        "",
    ]
    (RESULTS/"utility_routing_audit_report.md").write_text("\n".join(lines),encoding="utf-8")


def run_all(devices: list[str]):
    if not torch.cuda.is_available():
        raise RuntimeError("E5 requires CUDA, but CUDA is unavailable")
    for device in devices:
        if not str(device).startswith("cuda"):
            raise RuntimeError(f"invalid requested device {device}")
    OUTPUT.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    preflight_all(devices,SCREEN)
    print("[preflight] screening action landscapes and C0 checks complete",flush=True)
    train_groups("phase_b",SCREEN,devices)
    phase_b_rows,_=analyze_phase_b(devices[0])
    b_gate=phase_b_gate(phase_b_rows)
    _write_json(RESULTS/"phase_b_gate.json",b_gate)
    historical=phase_b_historical_comparison(phase_b_rows)
    phase_c_gate=None
    phase_d_gate=None
    if b_gate["supported"]:
        train_groups("phase_c",SCREEN,devices)
        _,phase_c_gate=analyze_conditional_phase("phase_c",devices[0])
    b2_pass=next((r["utility_aligned_objective_gate"] for r in b_gate["candidate_gates"]
                  if r["candidate"]=="B2_weighted_utility_kd"),False)
    if b2_pass:
        edge_alignment_audit(SCREEN,devices[0])
        train_groups("phase_d",SCREEN,devices)
        _,phase_d_gate=analyze_conditional_phase("phase_d",devices[0])
    phase_e,_=run_phase_e(devices[0],phase_b_rows,b_gate,phase_c_gate,phase_d_gate)
    screening_decisions(phase_b_rows,b_gate,historical,phase_c_gate,phase_d_gate,phase_e)
    lock=freeze_screening_winner(phase_b_rows,b_gate,phase_c_gate,phase_d_gate,phase_e)
    print(f"[winner lock] {lock['winning_variant']} safe={lock['safe_routing_used']}",flush=True)
    run_confirmation(devices)
    finalize_all(phase_b_rows,b_gate,historical,phase_c_gate,phase_d_gate,phase_e)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=[
        "preflight","phase_b","phase_c","phase_d","phase_e","confirmation","all"
    ],default="all")
    parser.add_argument("--devices",nargs="+",default=["cuda:0","cuda:1"])
    parser.add_argument("--preflight-one",nargs=3,metavar=("DATASET","SEED","DEVICE"))
    parser.add_argument("--train-group",nargs=4,metavar=("PHASE","DATASET","SEED","DEVICE"))
    args=parser.parse_args()
    if args.preflight_one:
        dataset,seed,device=args.preflight_one
        check,summary=preflight_one(dataset,int(seed),device)
        print(json.dumps({"check":check,"summary_rows":len(summary)},indent=2))
        return
    if args.train_group:
        phase,dataset,seed,device=args.train_group
        print(train_group_one(phase,dataset,int(seed),device))
        return
    if args.phase=="preflight":
        OUTPUT.mkdir(parents=True,exist_ok=True)
        RESULTS.mkdir(parents=True,exist_ok=True)
        preflight_all(args.devices,SCREEN)
        return
    if args.phase=="phase_b":
        train_groups("phase_b",SCREEN,args.devices)
        rows,_=analyze_phase_b(args.devices[0])
        _write_json(RESULTS/"phase_b_gate.json",phase_b_gate(rows))
        return
    if args.phase=="phase_c":
        train_groups("phase_c",SCREEN,args.devices)
        analyze_conditional_phase("phase_c",args.devices[0])
        return
    if args.phase=="phase_d":
        edge_alignment_audit(SCREEN,args.devices[0])
        train_groups("phase_d",SCREEN,args.devices)
        analyze_conditional_phase("phase_d",args.devices[0])
        return
    if args.phase=="phase_e":
        rows=pd.read_csv(RESULTS/"phase_b_objective_results.csv").to_dict("records")
        gate=json.loads((RESULTS/"phase_b_gate.json").read_text())
        c=json.loads((RESULTS/"phase_c_gate.json").read_text()) if (RESULTS/"phase_c_gate.json").is_file() else None
        d=json.loads((RESULTS/"phase_d_gate.json").read_text()) if (RESULTS/"phase_d_gate.json").is_file() else None
        run_phase_e(args.devices[0],rows,gate,c,d)
        return
    if args.phase=="confirmation":
        run_confirmation(args.devices)
        return
    run_all(args.devices)


if __name__=="__main__":
    main()
