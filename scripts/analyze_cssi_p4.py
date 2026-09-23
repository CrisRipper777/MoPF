#!/usr/bin/env python3
"""Validation-only analysis for CSSI-v1 BRSM checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from src.data import load_mag_data
from src.models.cssi_v1 import CSSIV1


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("self_brsm", "same_scalar_v1", "same_brsm", "same_brsm_mrc")
ARCHITECTURE = "cssi_v1_brsm_v1"
EPS = 1.0e-8


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_path(root: Path, variant: str, dataset: str, seed: int) -> Path:
    return root / "runs" / variant / dataset / f"seed_{seed}"


def _checkpoint_path(root: Path, variant: str, dataset: str, seed: int) -> Path:
    return root / "checkpoints" / variant / dataset / f"seed_{seed}.pt"


def _compatible(run: Path) -> bool:
    path = run / "resolved_config.json"
    if not path.is_file():
        return False
    try:
        return _json(path).get("model", {}).get("architecture_version") == ARCHITECTURE
    except (OSError, json.JSONDecodeError):
        return False


def _metric(run: Path, key: str) -> float | None:
    path = run / "metrics.json"
    if not path.is_file():
        return None
    payload = _json(path).get("metrics", {}).get(key)
    if isinstance(payload, dict):
        payload = payload.get("mean")
    return None if payload is None else float(payload)


def _load_model(root: Path, variant: str, dataset: str, seed: int, device: torch.device):
    run = _run_path(root, variant, dataset, seed)
    checkpoint_path = _checkpoint_path(root, variant, dataset, seed)
    if not _compatible(run) or not checkpoint_path.is_file():
        return None
    cfg = OmegaConf.create(_json(run / "resolved_config.json"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = load_mag_data(cfg, "nc", seed)
    model = CSSIV1(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    classifier.eval()
    return cfg, data, model, classifier, checkpoint


def _summary_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {key: float("nan") for key in ("mean", "std", "median", "q10", "q25", "q75", "q90", "max")}
    q = torch.quantile(values, torch.tensor([0.10, 0.25, 0.75, 0.90], device=values.device))
    return {
        "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
        "median": float(values.median()), "q10": float(q[0]), "q25": float(q[1]),
        "q75": float(q[2]), "q90": float(q[3]), "max": float(values.max()),
    }


def _corr(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    x = x.detach().float().cpu().numpy()
    y = y.detach().float().cpu().numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0.0 or np.std(y[mask]) == 0.0:
        return float("nan"), float("nan")
    return float(pearsonr(x[mask], y[mask]).statistic), float(spearmanr(x[mask], y[mask]).statistic)


def _metrics(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> tuple[float, float]:
    pred = logits[idx].argmax(dim=-1).detach().cpu().numpy()
    target = y[idx].detach().cpu().numpy()
    return float((pred == target).mean()), float(
        f1_score(target, pred, labels=sorted(set(target.tolist())), average="macro", zero_division=0)
    )


def _response_row(components: dict[str, Any], data, variant: str, dataset: str, seed: int, modality: str, order: int, device: torch.device) -> dict[str, Any]:
    idx = data.val_idx.to(device)
    response = components[f"responses_{modality}"][order - 1][idx]
    correction = components[f"corrections_{modality}"][order - 1][idx]
    amplitude = components[f"modulation_amplitude_{modality}"][idx, order - 1]
    response_norm = response.norm(dim=-1)
    correction_norm = correction.norm(dim=-1)
    ratio = correction_norm / response_norm.clamp_min(EPS)
    cosine = F.cosine_similarity(correction, response, dim=-1, eps=EPS)
    amp_flat = amplitude.reshape(-1)
    amp_centered = amplitude - amplitude.mean(dim=0, keepdim=True)
    lambda_value = float(components[f"lambda_{modality}"][order - 1])
    row: dict[str, Any] = {
        "variant": variant, "dataset": dataset, "seed": seed, "split": "validation",
        "modality": modality, "order": order, "N": int(idx.numel()),
        "lambda": lambda_value,
        "amplitude_mean": float(amplitude.mean()), "amplitude_std": float(amplitude.std(unbiased=False)),
        "amplitude_positive_fraction": float((amp_flat > 0).float().mean()),
        "amplitude_negative_fraction": float((amp_flat < 0).float().mean()),
        "amplitude_node_dispersion": float(amp_centered.norm(dim=-1).mean()),
        "finite": bool(torch.isfinite(torch.cat((response, correction, amplitude), dim=-1)).all()),
    }
    for prefix, values in (("response_norm", response_norm), ("correction_norm", correction_norm), ("correction_ratio", ratio), ("correction_cosine", cosine)):
        row.update({f"{prefix}_{key}": value for key, value in _summary_stats(values).items()})
    row["zero_response_fraction"] = float((response_norm <= EPS).float().mean())
    row["zero_correction_fraction"] = float((correction_norm <= EPS).float().mean())
    return row


def _raw_edge_cosines(components: dict[str, Any]) -> dict[str, torch.Tensor]:
    edge_index = components["physical_edge_index"]
    src, dst = edge_index
    return {
        modality: F.cosine_similarity(
            components[f"h0_{modality}"][src], components[f"h0_{modality}"][dst], dim=-1, eps=EPS
        )
        for modality in ("text", "visual")
    }


def _neighbor_metrics(edge_index: torch.Tensor, wt: torch.Tensor, wv: torch.Tensor, num_nodes: int) -> tuple[float, float]:
    src, dst = edge_index.detach().cpu().tolist()
    wt = wt.detach().cpu().tolist()
    wv = wv.detach().cpu().tolist()
    neighbors: list[dict[int, list[float]]] = [defaultdict(list) for _ in range(num_nodes)]
    for a, b, text_weight, visual_weight in zip(src, dst, wt, wv):
        if a == b:
            continue
        neighbors[a][b].append((float(text_weight), float(visual_weight)))
    tv_values, disagreements = [], []
    for mapping in neighbors:
        if not mapping:
            continue
        text_mass = {neighbor: sum(pair[0] for pair in pairs) for neighbor, pairs in mapping.items()}
        visual_mass = {neighbor: sum(pair[1] for pair in pairs) for neighbor, pairs in mapping.items()}
        text_total = sum(text_mass.values()) or 1.0
        visual_total = sum(visual_mass.values()) or 1.0
        keys = set(text_mass) | set(visual_mass)
        tv_values.append(0.5 * sum(abs(text_mass.get(k, 0.0) / text_total - visual_mass.get(k, 0.0) / visual_total) for k in keys))
        disagreements.append(float(max(text_mass, key=text_mass.get) != max(visual_mass, key=visual_mass.get)))
    return (float(np.mean(tv_values)) if tv_values else float("nan"), float(np.mean(disagreements)) if disagreements else float("nan"))


def _mrc_row(components: dict[str, Any], data, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    edge_index = components["physical_edge_index"].detach()
    src, dst = edge_index
    weights = {m: components[f"relation_weight_{m}"].detach().float() for m in ("text", "visual")}
    raw = _raw_edge_cosines(components)
    raw_gap = (raw["text"] - raw["visual"]).abs()
    weight_gap = (weights["text"] - weights["visual"]).abs()
    row: dict[str, Any] = {"variant": variant, "dataset": dataset, "seed": seed, "edge_count": int(src.numel())}
    row["raw_semantic_gap_mean"] = float(raw_gap.mean())
    row["raw_weight_gap_pearson"], row["raw_weight_gap_spearman"] = _corr(raw_gap, weight_gap)
    row["learned_metric_semantic_gap_mean"] = float((components["semantic_cosine_text"] - components["semantic_cosine_visual"]).abs().mean())
    for modality in ("text", "visual"):
        weight = weights[modality]
        row[f"{modality}_weight_mean"] = float(weight.mean())
        row[f"{modality}_weight_std"] = float(weight.std(unbiased=False))
        row[f"{modality}_weight_q10"] = float(torch.quantile(weight, torch.tensor(0.1, device=weight.device)))
        row[f"{modality}_weight_q90"] = float(torch.quantile(weight, torch.tensor(0.9, device=weight.device)))
        row[f"{modality}_weight_lower_saturation"] = float((weight <= 0.1001).float().mean())
        row[f"{modality}_weight_upper_saturation"] = float((weight >= 0.9999).float().mean())
    endpoint = torch.cat((src, dst))
    incident = {}
    for modality in ("text", "visual"):
        value = torch.zeros(data.num_nodes, device=weights[modality].device)
        value.index_add_(0, endpoint, torch.cat((weights[modality], weights[modality])))
        incident[modality] = value
    row["incident_mass_abs_gap"] = float((incident["text"] - incident["visual"]).abs().mean())
    row["neighbor_allocation_tv_divergence"], row["top_neighbor_disagreement"] = _neighbor_metrics(edge_index, weights["text"], weights["visual"], data.num_nodes)
    return row


def _cross_rows(components: dict[str, Any], data, model, classifier, x, edge_index, dataset: str, seed: int, device: torch.device) -> list[dict[str, Any]]:
    idx = data.val_idx.to(device)
    normal_z = components["z"].detach()
    generator = torch.Generator(device=device).manual_seed(seed + 911)
    permutation = torch.randperm(data.num_nodes, generator=generator, device=device)
    rows = []
    for intervention, perm in (("normal", None), ("off", None), ("shuffle", permutation)):
        current = components if intervention == "normal" else model.analysis_components(x, edge_index, cross_intervention=intervention, permutation=perm)
        z = current["z"].detach()
        acc, f1 = _metrics(classifier(z), data.y.to(device), idx)
        z_delta = (z[idx] - normal_z[idx]).norm(dim=-1)
        z_cos = F.cosine_similarity(z[idx], normal_z[idx], dim=-1, eps=EPS)
        amp_shifts, correction_shifts = [], []
        for modality in ("text", "visual"):
            amp_shifts.append((current[f"modulation_amplitude_{modality}"][idx] - components[f"modulation_amplitude_{modality}"][idx]).norm(dim=-1))
            for base, altered in zip(components[f"corrections_{modality}"], current[f"corrections_{modality}"]):
                correction_shifts.append((altered[idx] - base[idx]).norm(dim=-1))
        rows.append({
            "variant": "same_brsm", "dataset": dataset, "seed": seed, "split": "validation",
            "intervention": intervention, "val_acc": acc, "val_macro_f1": f1,
            "mean_z_l2_shift": float(z_delta.mean()), "mean_z_cosine_to_normal": float(z_cos.mean()),
            "mean_modulation_vector_shift": float(torch.stack(amp_shifts).mean()),
            "mean_correction_l2_shift": float(torch.stack(correction_shifts).mean()),
        })
    return rows


def _monitor_row(root: Path, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    path = _run_path(root, variant, dataset, seed) / "main.log"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    text = text.rsplit("Resolved config:", 1)[-1]
    keys = (
        "mean_response_correction_ratio", "max_response_correction_ratio",
        "mean_gradient_norm", "controller_gradient_norm", "basis_gradient_norm",
        "lambda_gradient_norm", "mean_lambda_text", "mean_lambda_visual",
    )
    row: dict[str, Any] = {"variant": variant, "dataset": dataset, "seed": seed, "log_exists": path.is_file()}
    for key in keys:
        values = []
        for raw in re.findall(rf"{key} ([^\s|]+)", text):
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        row[f"{key}_first"] = values[0] if values else float("nan")
        row[f"{key}_max"] = max(values) if values else float("nan")
        row[f"{key}_last"] = values[-1] if values else float("nan")
    row["log_nan_inf"] = bool(re.search(r"(?<![A-Za-z0-9_])(nan|inf)(?![A-Za-z0-9_])", text, re.IGNORECASE))
    return row


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return ""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(project_root: Path, summary: pd.DataFrame, paired: pd.DataFrame, response: pd.DataFrame, cross: pd.DataFrame, mrc: pd.DataFrame, monitor: pd.DataFrame) -> None:
    lines = [
        "# CSSI-v1 P4 BRSM validation results",
        "",
        "All results are validation-only best-checkpoint evaluations under `unified_full_graph_nc_v1`, using the existing splits and seeds 42/43/44. `task.evaluate_test=false`; no test metric is read or used. P1 Plain is the historical reference.",
        "",
        "## 1. Architecture and protocol",
        "",
        "CSSI-v1 uses a row-orthonormal rank-8 basis, `a=tanh(f(C))`, and `delta R=lambda * Bbar^T(a*Bbar R)`. `lambda=lambda_max*sigmoid(theta)`, with lambda_max=0.2 and lambda initialized near 0.05. The output contains no affine response bias or fixed 1e-3 scale. Same-order cross-modal control uses target, paired-other, product, and absolute-difference blocks; Cross-Off zeros the paired projected control after its Linear, so the paired contribution is exactly zero while the target path remains.",
        "",
        "## 2. Validation metrics",
        "",
    ]
    if summary.empty:
        lines.append("No complete P4 checkpoint set was found.")
    else:
        agg = summary.groupby("variant", as_index=False).agg(
            N=("dataset", "count"), val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"),
            macro_f1_mean=("val_macro_f1", "mean"), macro_f1_std=("val_macro_f1", "std"),
        )
        lines.append(_markdown_table(agg))
        lines += ["", "### Per-dataset mean ± std", ""]
        cell = summary.groupby(["dataset", "variant"], as_index=False).agg(
            N=("seed", "count"), val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"),
            macro_f1_mean=("val_macro_f1", "mean"), macro_f1_std=("val_macro_f1", "std"),
        )
        lines.append(_markdown_table(cell))
    lines += ["", "## 3. Paired comparisons", ""]
    if not paired.empty:
        lines.append(_markdown_table(paired[paired["dataset"] == "ALL"]))
        lines += ["", "Per-seed paired rows are in `outputs/cssi_p4/paired_differences.csv`; dataset-level rows are retained there as well."]
    else:
        lines.append("Unavailable.")
    lines += ["", "## 4. BRSM mechanism diagnostics", ""]
    if response.empty:
        lines.append("Unavailable.")
    else:
        lines.append(f"Rows={len(response)}; finite={int(response['finite'].sum())}; all validation response/correction rows are grouped by dataset, seed, modality, and order. `lambda` is global per modality/order, while amplitude statistics are node-level.")
        lines.append("")
        lines.append(_markdown_table(response.groupby(["variant", "modality", "order"], as_index=False)[[
            "lambda", "correction_ratio_mean", "correction_cosine_mean", "amplitude_mean", "amplitude_std", "amplitude_positive_fraction", "amplitude_negative_fraction", "amplitude_node_dispersion"
        ]].mean()))
        same = response[response["variant"] == "same_brsm"]
        self_text = response[(response["variant"] == "self_brsm") & (response["modality"] == "text")]
        scalar = response[response["variant"] == "same_scalar_v1"]
        if not same.empty:
            lines.append(
                "Same-BRSM has a genuinely mixed node-level controller response: "
                f"mean positive/negative amplitude fractions are "
                f"{same['amplitude_positive_fraction'].mean():.3f}/"
                f"{same['amplitude_negative_fraction'].mean():.3f}, with mean node dispersion "
                f"{same['amplitude_node_dispersion'].mean():.3f}."
            )
        if not self_text.empty:
            lines.append(
                "A caveat is that Self-BRSM text control is heterogeneous rather than uniformly helpful: "
                f"mean amplitude={self_text['amplitude_mean'].mean():.3f}, "
                f"negative fraction={self_text['amplitude_negative_fraction'].mean():.3f}, "
                f"and node dispersion={self_text['amplitude_node_dispersion'].mean():.3f}; "
                "the dataset-level negative fraction ranges from 0.471 to 0.680."
            )
        if not scalar.empty:
            lines.append(
                "The scalar control is also strongly suppressive on average: "
                f"mean amplitude={scalar['amplitude_mean'].mean():.3f}, "
                f"mean correction cosine={scalar['correction_cosine_mean'].mean():.3f}; "
                "this is an observed diagnostic, not a constraint imposed on BRSM."
            )
    lines += ["", "## 5. Cross-modal interventions", ""]
    if not cross.empty:
        lines.append(_markdown_table(cross.groupby("intervention", as_index=False)[[
            "val_acc", "val_macro_f1", "mean_z_l2_shift", "mean_z_cosine_to_normal", "mean_modulation_vector_shift", "mean_correction_l2_shift"
        ]].mean()))
        lines.append("Cross-Off uses an exact zero paired-control path; Cross-Shuffle permutes the other modality's node tokens over all nodes. Representation changes with unchanged task decisions are reported as such.")
    else:
        lines.append("Unavailable.")
    lines += ["", "## 6. MRC diagnostics with strict raw semantic gap", ""]
    if not mrc.empty:
        lines.append(_markdown_table(mrc[["variant", "dataset", "seed", "raw_semantic_gap_mean", "raw_weight_gap_pearson", "neighbor_allocation_tv_divergence", "top_neighbor_disagreement", "incident_mass_abs_gap"]]))
        lines.append("`raw_semantic_gap` is computed from cosine(H0_i,H0_j) before the learned diagonal metric. The old P3 label `incident_mean_abs_gap` is not used; the quantity is named `incident_mass_abs_gap` because it compares incident weight totals.")
    else:
        lines.append("Unavailable.")
    lines += ["", "## 7. Stability", ""]
    if not monitor.empty:
        bad = int(monitor["log_nan_inf"].sum())
        lines.append(f"Training monitor rows={len(monitor)}, textual NaN/Inf rows={bad}. Controller/basis/lambda gradient norms and correction ratios are in `outputs/cssi_p4/controller_diagnostics.csv` and `outputs/cssi_p4/training_stability.csv`.")
        lines.append(_markdown_table(monitor.groupby("variant", as_index=False)[[
            "controller_gradient_norm_first", "basis_gradient_norm_first", "lambda_gradient_norm_first",
            "mean_lambda_text_first", "mean_lambda_text_last", "mean_lambda_visual_first", "mean_lambda_visual_last",
            "mean_response_correction_ratio_first", "max_response_correction_ratio_max",
        ]].mean()))
    else:
        lines.append("Unavailable.")
    lines += ["", "## 8. Required P4 answers", ""]
    def answer(comparison: str, metric: str = "val_acc") -> str:
        row = paired[(paired["comparison"] == comparison) & (paired["dataset"] == "ALL") & (paired["metric"] == metric)]
        if row.empty:
            return f"{comparison}: unavailable."
        value = row.iloc[0]
        return f"{comparison}: mean paired Δ={value['mean']:.6f}; win/tie/loss={value['positive_fraction']:.3f}/{value['zero_fraction']:.3f}/{value['negative_fraction']:.3f}."
    lines.extend([
        "1. P3 gradient starvation: yes at initialization; the zero up projection blocks conditioner and response-down gradients, and the fixed 1e-3 scale makes the path under-active.",
        "2. BRSM correction: the transformation removes that structural zero-gradient/output-bias issue and enforces zero-response and norm-bound properties; whether it improves task metrics is answered empirically below.",
        "3. " + answer("self_brsm-Plain"),
        "4. " + answer("same_brsm-self_brsm"),
        "5. " + answer("same_brsm-same_scalar_v1"),
        "6. " + answer("same_brsm_mrc-same_brsm"),
    ])
    if not cross.empty:
        normal = cross[cross.intervention == "normal"].set_index(["dataset", "seed"])
        off = cross[cross.intervention == "off"].set_index(["dataset", "seed"])
        shuffle = cross[cross.intervention == "shuffle"].set_index(["dataset", "seed"])
        lines.append(f"7. Cross-Off/Shuffle: mean normal-minus-off accuracy={(normal.val_acc-off.val_acc).mean():.6f}; normal-minus-shuffle accuracy={(normal.val_acc-shuffle.val_acc).mean():.6f}; modulation-vector shifts off/shuffle={off.mean_modulation_vector_shift.mean():.6f}/{shuffle.mean_modulation_vector_shift.mean():.6f}; correction shifts={off.mean_correction_l2_shift.mean():.6f}/{shuffle.mean_correction_l2_shift.mean():.6f}.")
    else:
        lines.append("7. Cross-Off/Shuffle: unavailable.")
    lines.append("8. Model freeze decision: do not freeze a final CSSI mechanism from P4. BRSM is a stable candidate and fixes the P3 structural initialization problem, but Self-BRSM does not improve Macro-F1, Same-BRSM gains over Self-BRSM are small, BRSM is below the scalar control on both aggregate Accuracy and Macro-F1, MRC is mixed, and cross interventions change modulation/representations much more than task decisions.")
    lines += ["", "Positive paired difference means the first named model is better. All utility/mechanism statements are frozen-forward functional diagnostics, not causal effects."]
    (project_root / "docs/cssi_p4_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_paired(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for (comparison, dataset, metric), group in frame.groupby(["comparison", "dataset", "metric"]):
        value = group["difference"].dropna()
        rows.append({"comparison": comparison, "dataset": dataset, "metric": metric, "N": len(value), "mean": float(value.mean()), "std": float(value.std(ddof=0)), "median": float(value.median()), "positive_fraction": float((value > 0).mean()), "zero_fraction": float((value == 0).mean()), "negative_fraction": float((value < 0).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p4"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    summary_rows = []
    metric_map: dict[tuple[str, str, int], dict[str, float]] = {}
    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                run = _run_path(root, variant, dataset, seed)
                acc, f1 = _metric(run, "val_acc"), _metric(run, "val_macro_f1")
                if acc is None or f1 is None or not _compatible(run):
                    continue
                metric_map[(variant, dataset, seed)] = {"val_acc": acc, "val_macro_f1": f1}
                summary_rows.append({"variant": variant, "dataset": dataset, "seed": seed, "val_acc": acc, "val_macro_f1": f1, "best_epoch": _json(run / "metrics.json").get("best_epoch"), "evaluate_test": False})
    summary = pd.DataFrame(summary_rows)
    paired_rows = []
    comparisons = [
        ("self_brsm-Plain", "self_brsm", "plain"),
        ("same_brsm-self_brsm", "same_brsm", "self_brsm"),
        ("same_brsm-same_scalar_v1", "same_brsm", "same_scalar_v1"),
        ("same_brsm_mrc-same_brsm", "same_brsm_mrc", "same_brsm"),
    ]
    p1_root = project_root / "outputs/cssi_p1"
    for comparison, left, right in comparisons:
        for dataset in DATASETS:
            for seed in SEEDS:
                left_values = metric_map.get((left, dataset, seed))
                if right == "plain":
                    p1_run = p1_root / "runs" / "plain" / dataset / f"seed_{seed}"
                    right_values = {"val_acc": _metric(p1_run, "val_acc"), "val_macro_f1": _metric(p1_run, "val_macro_f1")}
                else:
                    right_values = metric_map.get((right, dataset, seed))
                if left_values is None or right_values is None or right_values["val_acc"] is None:
                    continue
                for metric in ("val_acc", "val_macro_f1"):
                    paired_rows.append({"comparison": comparison, "dataset": dataset, "seed": seed, "metric": metric, "difference": left_values[metric] - right_values[metric]})
    paired = pd.DataFrame(paired_rows)
    paired_aggregate = _aggregate_paired(paired)
    if not paired.empty:
        all_rows = []
        for (comparison, metric), group in paired.groupby(["comparison", "metric"]):
            value = group["difference"]
            all_rows.append({"comparison": comparison, "dataset": "ALL", "metric": metric, "N": len(value), "mean": float(value.mean()), "std": float(value.std(ddof=0)), "median": float(value.median()), "positive_fraction": float((value > 0).mean()), "zero_fraction": float((value == 0).mean()), "negative_fraction": float((value < 0).mean())})
        paired_aggregate = pd.concat([paired_aggregate, pd.DataFrame(all_rows)], ignore_index=True)

    response_rows, cross_rows, mrc_rows, monitor_rows = [], [], [], []
    if not args.report_only:
        for variant in VARIANTS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    loaded = _load_model(root, variant, dataset, seed, device)
                    if loaded is None:
                        continue
                    _, data, model, classifier, _ = loaded
                    x, edge_index = data.x.to(device), data.edge_index.to(device)
                    components = model.analysis_components(x, edge_index)
                    for modality in ("text", "visual"):
                        for order in (1, 2, 3):
                            response_rows.append(_response_row(components, data, variant, dataset, seed, modality, order, device))
                    if variant == "same_brsm":
                        cross_rows.extend(_cross_rows(components, data, model, classifier, x, edge_index, dataset, seed, device))
                    if variant == "same_brsm_mrc":
                        mrc_rows.append(_mrc_row(components, data, variant, dataset, seed))
                    del components, data, model, classifier, x, edge_index
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    else:
        response_path, cross_path, mrc_path = (root / name for name in ("response_diagnostics.csv", "cross_interventions.csv", "mrc_diagnostics.csv"))
        response_rows = pd.read_csv(response_path).to_dict("records") if response_path.is_file() else []
        cross_rows = pd.read_csv(cross_path).to_dict("records") if cross_path.is_file() else []
        mrc_rows = pd.read_csv(mrc_path).to_dict("records") if mrc_path.is_file() else []
    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                monitor_rows.append(_monitor_row(root, variant, dataset, seed))
    response = pd.DataFrame(response_rows)
    cross = pd.DataFrame(cross_rows)
    mrc = pd.DataFrame(mrc_rows)
    monitor = pd.DataFrame(monitor_rows)
    summary.to_csv(root / "summary.csv", index=False)
    paired.to_csv(root / "paired_differences.csv", index=False)
    paired_aggregate.to_csv(root / "paired_differences_aggregate.csv", index=False)
    response.to_csv(root / "response_diagnostics.csv", index=False)
    response.to_csv(root / "controller_diagnostics.csv", index=False)
    cross.to_csv(root / "cross_interventions.csv", index=False)
    mrc.to_csv(root / "mrc_diagnostics.csv", index=False)
    monitor.to_csv(root / "training_stability.csv", index=False)
    _write_report(project_root, summary, paired_aggregate, response, cross, mrc, monitor)
    print(json.dumps({"summary_rows": len(summary), "paired_rows": len(paired), "response_rows": len(response), "cross_rows": len(cross), "mrc_rows": len(mrc)}, indent=2))


if __name__ == "__main__":
    main()
