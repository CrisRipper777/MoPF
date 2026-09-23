#!/usr/bin/env python3
"""Frozen-checkpoint analysis and report generation for P3.

The script never evaluates the NC test split.  It reads best-validation
checkpoints, recomputes the representation and the requested frozen
interventions, and writes compact CSV summaries under outputs/cssi_p3.
"""

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
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from src.data import load_mag_data
from src.models.cssi_v0 import CSSIV0


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("mrc_plain", "self_lr", "same_scalar", "same_lr", "all_lr", "same_lr_mrc")
EPS = 1.0e-8


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_path(root: Path, variant: str, dataset: str, seed: int) -> Path:
    return root / "runs" / variant / dataset / f"seed_{seed}"


def _checkpoint_path(root: Path, variant: str, dataset: str, seed: int) -> Path:
    return root / "checkpoints" / variant / dataset / f"seed_{seed}.pt"


def _compatible_run(run_dir: Path) -> bool:
    config_path = run_dir / "resolved_config.json"
    if not config_path.is_file():
        return False
    try:
        config = _json(config_path)
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("model", {}).get("architecture_version") == "cssi_v0_same_v1"


def _metric(path: Path, key: str) -> float | None:
    if not path.is_file():
        return None
    payload = _json(path)
    value = payload.get("metrics", {}).get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    return None if value is None else float(value)


def _finite_stats(values: torch.Tensor) -> tuple[float, float, float, float, float, float]:
    values = values.detach().float().cpu()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return (float("nan"),) * 6
    q = torch.quantile(finite, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return (
        float(finite.mean()),
        float(finite.std(unbiased=False)),
        float(q[2]),
        float(q[0]),
        float(q[1]),
        float(q[3]),
    )


def _summary_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {key: float("nan") for key in ("mean", "std", "median", "q10", "q25", "q75", "q90", "max")}
    q = torch.quantile(finite, torch.tensor([0.10, 0.25, 0.75, 0.90]))
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "median": float(finite.median()),
        "q10": float(q[0]),
        "q25": float(q[1]),
        "q75": float(q[2]),
        "q90": float(q[3]),
        "max": float(finite.max()),
    }


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    x = x.detach().float().cpu().numpy()
    y = y.detach().float().cpu().numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0.0 or np.std(y[mask]) == 0.0:
        return float("nan"), float("nan")
    return float(pearsonr(x[mask], y[mask]).statistic), float(spearmanr(x[mask], y[mask]).statistic)


def _validation_metrics(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> tuple[float, float]:
    pred = logits[idx].argmax(dim=-1).detach().cpu().numpy()
    target = labels[idx].detach().cpu().numpy()
    return float((pred == target).mean()), float(
        f1_score(target, pred, labels=sorted(set(target.tolist())), average="macro", zero_division=0)
    )


def _load_model(run_root: Path, variant: str, dataset: str, seed: int, device: torch.device):
    run_dir = _run_path(run_root, variant, dataset, seed)
    checkpoint_path = _checkpoint_path(run_root, variant, dataset, seed)
    config_path = run_dir / "resolved_config.json"
    if not checkpoint_path.is_file() or not config_path.is_file() or not _compatible_run(run_dir):
        return None
    cfg = OmegaConf.create(_json(config_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = load_mag_data(cfg, "nc", int(seed))
    model = CSSIV0(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(checkpoint["head_state"], strict=True)
    classifier.eval()
    return cfg, data, model, classifier, checkpoint


def _base_rows(run_root: Path, p1_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            plain_dir = _run_path(p1_root, "plain", dataset, seed)
            plain_acc = _metric(plain_dir / "metrics.json", "val_acc")
            plain_f1 = _metric(plain_dir / "metrics.json", "val_macro_f1")
            for variant in VARIANTS:
                run_dir = _run_path(run_root, variant, dataset, seed)
                metrics_path = run_dir / "metrics.json"
                if not _compatible_run(run_dir):
                    continue
                acc = _metric(metrics_path, "val_acc")
                f1 = _metric(metrics_path, "val_macro_f1")
                if acc is None:
                    continue
                payload = _json(metrics_path)
                rows.append({
                    "variant": variant, "dataset": dataset, "seed": seed,
                    "val_acc": acc, "val_macro_f1": f1,
                    "best_epoch": payload.get("best_epoch"),
                    "evaluate_test": bool(payload.get("evaluate_test", False)),
                    "checkpoint": str(_checkpoint_path(run_root, variant, dataset, seed)),
                })
                comparisons = [("P3-MRC-Plain", variant, "p1_plain", acc - plain_acc if plain_acc is not None else None, f1 - plain_f1 if plain_f1 is not None else None)] if variant == "mrc_plain" else []
                if variant in {"self_lr", "same_lr", "all_lr", "same_lr_mrc", "same_scalar"} and plain_acc is not None:
                    comparisons.append((f"{variant}-Plain", variant, "p1_plain", acc - plain_acc, f1 - plain_f1))
                if variant == "same_lr" and "self_lr" in {r["variant"] for r in rows}:
                    self_acc = _metric(_run_path(run_root, "self_lr", dataset, seed) / "metrics.json", "val_acc")
                    self_f1 = _metric(_run_path(run_root, "self_lr", dataset, seed) / "metrics.json", "val_macro_f1")
                    if self_acc is not None:
                        comparisons.append(("Same-LR-Self-LR", variant, "self_lr", acc - self_acc, f1 - self_f1))
                if variant == "same_lr" and _metric(_run_path(run_root, "same_scalar", dataset, seed) / "metrics.json", "val_acc") is not None:
                    scalar_acc = _metric(_run_path(run_root, "same_scalar", dataset, seed) / "metrics.json", "val_acc")
                    scalar_f1 = _metric(_run_path(run_root, "same_scalar", dataset, seed) / "metrics.json", "val_macro_f1")
                    comparisons.append(("Same-LR-Same-Scalar", variant, "same_scalar", acc - scalar_acc, f1 - scalar_f1))
                if variant == "all_lr":
                    same_acc = _metric(_run_path(run_root, "same_lr", dataset, seed) / "metrics.json", "val_acc")
                    same_f1 = _metric(_run_path(run_root, "same_lr", dataset, seed) / "metrics.json", "val_macro_f1")
                    if same_acc is not None:
                        comparisons.append(("All-LR-Same-LR", variant, "same_lr", acc - same_acc, f1 - same_f1))
                if variant == "same_lr_mrc":
                    same_acc = _metric(_run_path(run_root, "same_lr", dataset, seed) / "metrics.json", "val_acc")
                    same_f1 = _metric(_run_path(run_root, "same_lr", dataset, seed) / "metrics.json", "val_macro_f1")
                    if same_acc is not None:
                        comparisons.append(("Same-LR-MRC-Same-LR", variant, "same_lr", acc - same_acc, f1 - same_f1))
                for comparison, left, right, diff_acc, diff_f1 in comparisons:
                    if diff_acc is not None:
                        paired.append({"comparison": comparison, "dataset": dataset, "seed": seed, "metric": "val_acc", "difference": diff_acc})
                    if diff_f1 is not None:
                        paired.append({"comparison": comparison, "dataset": dataset, "seed": seed, "metric": "val_macro_f1", "difference": diff_f1})
    return rows, paired


def _response_rows(components: dict[str, Any], data, variant: str, dataset: str, seed: int, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    val_idx = data.val_idx.to(device)
    for modality in ("text", "visual"):
        responses = components[f"responses_{modality}"]
        corrections = components[f"corrections_{modality}"]
        for order, (response, correction) in enumerate(zip(responses, corrections), start=1):
            response_val = response[val_idx]
            correction_val = correction[val_idx]
            response_norm = response_val.norm(dim=-1)
            correction_norm = correction_val.norm(dim=-1)
            ratio = correction_norm / response_norm.clamp_min(EPS)
            cosine = torch.nn.functional.cosine_similarity(correction_val, response_val, dim=-1, eps=EPS)
            rstats = _summary_stats(response_norm)
            cstats = _summary_stats(correction_norm)
            xstats = _summary_stats(ratio)
            cosstats = _summary_stats(cosine)
            finite = bool(torch.isfinite(torch.cat((response_val, correction_val), dim=-1)).all())
            row = {
                "variant": variant, "dataset": dataset, "seed": seed,
                "split": "validation", "modality": modality, "order": order,
                "N": int(val_idx.numel()), "finite": finite,
            }
            row.update({f"response_norm_{key}": value for key, value in rstats.items()})
            row.update({f"correction_norm_{key}": value for key, value in cstats.items()})
            row.update({f"correction_ratio_{key}": value for key, value in xstats.items()})
            row.update({f"correction_cosine_{key}": value for key, value in cosstats.items()})
            row["correction_zero_fraction"] = float((correction_norm <= EPS).float().mean())
            rows.append(row)
    return rows


def _attention_rows(components: dict[str, Any], data, dataset: str, seed: int, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    idx = data.val_idx.to(device)
    for modality in ("text", "visual"):
        attention = components.get(f"attention_{modality}")
        if attention is None:
            continue
        attention = attention[idx]
        for target in range(attention.size(1)):
            for source in range(attention.size(2)):
                rows.append({
                    "variant": "all_lr", "dataset": dataset, "seed": seed,
                    "split": "validation", "modality": modality,
                    "target_order": target + 1, "source_order": source + 1,
                    "attention_mean": float(attention[:, target, source].mean()),
                })
        entropy = -(attention.clamp_min(EPS) * attention.clamp_min(EPS).log()).sum(dim=-1)
        diagonal = attention.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / attention.size(-1)
        rows.append({
            "variant": "all_lr", "dataset": dataset, "seed": seed,
            "split": "validation", "modality": modality,
            "target_order": "summary", "source_order": "summary",
            "attention_mean": float(attention.mean()),
            "diagonal_mass": float(diagonal.mean()),
            "offdiagonal_mass": float(1.0 - diagonal.mean()),
            "attention_entropy": float(entropy.mean()),
        })
    return rows


def _mrc_row(components: dict[str, Any], data, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    edge_index = components["physical_edge_index"].detach().cpu()
    src, dst = edge_index
    weights = {m: components[f"relation_weight_{m}"].detach().float().cpu() for m in ("text", "visual")}
    semantic = {m: components[f"semantic_cosine_{m}"].detach().float().cpu() for m in ("text", "visual")}
    row = {"variant": variant, "dataset": dataset, "seed": seed, "edge_count": int(src.numel())}
    row["weight_gap_mean"] = float((weights["text"] - weights["visual"]).abs().mean())
    row["semantic_gap_mean"] = float((semantic["text"] - semantic["visual"]).abs().mean())
    row["semantic_weight_gap_pearson"], row["semantic_weight_gap_spearman"] = _safe_corr(
        (semantic["text"] - semantic["visual"]).abs(),
        (weights["text"] - weights["visual"]).abs(),
    )
    for modality in ("text", "visual"):
        w = weights[modality]
        row[f"{modality}_weight_mean"] = float(w.mean())
        row[f"{modality}_weight_std"] = float(w.std(unbiased=False))
        row[f"{modality}_weight_min"] = float(w.min())
        row[f"{modality}_weight_max"] = float(w.max())
        row[f"{modality}_semantic_mean"] = float(semantic[modality].mean())
        row[f"{modality}_weight_finite"] = bool(torch.isfinite(w).all())
        row[f"{modality}_weight_lower_saturation"] = float((w <= 0.1001).float().mean())
        row[f"{modality}_weight_upper_saturation"] = float((w >= 0.9999).float().mean())
    # Compare node-wise incident mass allocation and top-neighbor choices.
    num_nodes = data.num_nodes
    endpoint = torch.cat((src, dst))
    incident = {}
    for modality in ("text", "visual"):
        total = torch.zeros(num_nodes)
        total.index_add_(0, endpoint, torch.cat((weights[modality], weights[modality])))
        incident[modality] = total
    row["incident_mean_abs_gap"] = float((incident["text"] - incident["visual"]).abs().mean())
    adjacency: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    src_np, dst_np = src.numpy(), dst.numpy()
    wt, wv = weights["text"].numpy(), weights["visual"].numpy()
    for index in range(len(src_np)):
        a, b = int(src_np[index]), int(dst_np[index])
        if a != b:
            adjacency[a].append((b, float(wt[index]), float(wv[index])))
            adjacency[b].append((a, float(wt[index]), float(wv[index])))
    disagreements = []
    for neighbors in adjacency.values():
        top_text = max(neighbors, key=lambda value: value[1])[0]
        top_visual = max(neighbors, key=lambda value: value[2])[0]
        disagreements.append(float(top_text != top_visual))
    row["top_neighbor_disagreement"] = float(np.mean(disagreements)) if disagreements else float("nan")
    return row


def _cross_rows(components: dict[str, Any], data, model, classifier, x, edge_index, dataset: str, seed: int, device: torch.device) -> list[dict[str, Any]]:
    normal_z = components["z"].detach()
    idx = data.val_idx.to(device)
    permutation = torch.arange(data.num_nodes, device=device)
    permutation[idx] = idx[torch.randperm(idx.numel(), device=device)]
    rows = []
    for intervention, perm in (("normal", None), ("off", None), ("shuffle", permutation)):
        current = components if intervention == "normal" else model.analysis_components(
            x, edge_index, cross_intervention=intervention, permutation=perm
        )
        z = current["z"].detach()
        logits = classifier(z)
        acc, f1 = _validation_metrics(logits, data.y.to(device), idx)
        cosine = torch.nn.functional.cosine_similarity(z[idx], normal_z[idx], dim=-1, eps=EPS)
        delta = (z[idx] - normal_z[idx]).norm(dim=-1)
        response_delta = []
        for modality in ("text", "visual"):
            for baseline, altered in zip(components[f"corrections_{modality}"], current[f"corrections_{modality}"]):
                response_delta.append((altered[idx] - baseline[idx]).norm(dim=-1))
        response_shift = torch.stack(response_delta).mean() if response_delta else torch.tensor(0.0, device=device)
        rows.append({
            "variant": "same_lr", "dataset": dataset, "seed": seed,
            "split": "validation", "intervention": intervention,
            "val_acc": acc, "val_macro_f1": f1,
            "mean_z_l2_shift": float(delta.mean()),
            "mean_z_cosine_to_normal": float(cosine.mean()),
            "mean_correction_l2_shift": float(response_shift),
        })
    return rows


def _monitor_row(run_dir: Path, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    path = run_dir / "main.log"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    patterns = {
        "mean_response_correction_ratio": r"mean_response_correction_ratio ([^ |\n]+)",
        "max_response_correction_ratio": r"max_response_correction_ratio ([^ |\n]+)",
        "mean_gradient_norm": r"mean_gradient_norm ([^ |\n]+)",
    }
    values: dict[str, list[float]] = {}
    for key, pattern in patterns.items():
        values[key] = []
        for value in re.findall(pattern, text):
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                values[key].append(number)
    numeric_nonfinite = r"(?<![A-Za-z0-9_])(nan|inf)(?![A-Za-z0-9_])"
    row = {
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "log_exists": path.is_file(),
        "log_nan_inf": bool(re.search(numeric_nonfinite, text, re.IGNORECASE)),
    }
    for key, numbers in values.items():
        row[f"{key}_max"] = max(numbers) if numbers else float("nan")
        row[f"{key}_last"] = numbers[-1] if numbers else float("nan")
    return row


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _aggregate_paired(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for (comparison, dataset, metric), group in frame.groupby(["comparison", "dataset", "metric"]):
        values = group["difference"].dropna()
        rows.append({
            "comparison": comparison, "dataset": dataset, "metric": metric,
            "N": int(values.size), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "median": float(values.median()),
            "positive_fraction": float((values > 0).mean()),
            "zero_fraction": float((values == 0).mean()),
            "negative_fraction": float((values < 0).mean()),
        })
    for (comparison, metric), group in frame.groupby(["comparison", "metric"]):
        values = group["difference"].dropna()
        rows.append({
            "comparison": comparison, "dataset": "ALL", "metric": metric,
            "N": int(values.size), "mean": float(values.mean()),
            "std": float(values.std(ddof=0)), "median": float(values.median()),
            "positive_fraction": float((values > 0).mean()),
            "zero_fraction": float((values == 0).mean()),
            "negative_fraction": float((values < 0).mean()),
        })
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, *, float_digits: int = 4) -> str:
    """Render a compact Markdown table without requiring tabulate."""
    if frame.empty:
        return ""
    frame = frame.reset_index(drop=True)
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def _markdown_report(out: Path, summary: pd.DataFrame, paired_summary: pd.DataFrame, responses: pd.DataFrame, attention: pd.DataFrame, mrc: pd.DataFrame, cross: pd.DataFrame, monitor: pd.DataFrame) -> None:
    lines = [
        "# CSSI P3 conditional structural-response results",
        "",
        "This report is generated from best-validation checkpoints by `scripts/analyze_cssi_p3.py`. All claims use validation nodes only; `task.evaluate_test=false` and no test metric is used.",
        "",
        "## 1. Protocol and implementation",
        "",
        "The six variants use `unified_full_graph_nc_v1`, the existing NC splits, seeds 42/43/44, and the same inherited refiner/fusion/classifier. The exact architecture is frozen in `docs/cssi_p3_architecture.md`. The low-rank adapter is rank 8 with fixed output scale 1e-3 and zero-initialized up projection; this was added after an early smoke run without the scale showed an exploding correction ratio.",
        "",
        "## 2. Validation backbone results",
        "",
    ]
    if summary.empty:
        lines.append("No complete P3 checkpoint set was available; the formal run was blocked before validation results could be produced.")
    else:
        table = summary.groupby("variant", as_index=False).agg(
            datasets=("dataset", "nunique"), seeds=("seed", "nunique"),
            val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"),
            val_f1_mean=("val_macro_f1", "mean"), val_f1_std=("val_macro_f1", "std"),
        )
        lines.append(_markdown_table(table))
        lines.append("")
        cell_table = summary.groupby(["dataset", "variant"], as_index=False).agg(
            N=("seed", "count"), val_acc_mean=("val_acc", "mean"),
            val_acc_std=("val_acc", "std"), val_f1_mean=("val_macro_f1", "mean"),
            val_f1_std=("val_macro_f1", "std"),
        )
        lines.append("### Per-dataset validation cells")
        lines.append("")
        lines.append(_markdown_table(cell_table))
        lines.append("")
        lines.append("The per-seed paired rows are in `outputs/cssi_p3/paired_differences.csv`; aggregate means and win/tie/loss are in `outputs/cssi_p3/paired_differences_aggregate.csv`.")
    lines += ["", "## 3. Requested paired comparisons", ""]
    if paired_summary.empty:
        lines.append("No paired validation comparison was available.")
    else:
        lines.append(_markdown_table(paired_summary[paired_summary["dataset"] == "ALL"]))
    lines += ["", "## 4. Response and controller diagnostics", ""]
    if responses.empty:
        lines.append("No response diagnostics were generated.")
    else:
        lines.append("`response_diagnostics.csv` reports validation response norms, correction norms, correction ratios, correction-response cosine, and zero-correction fractions by dataset, seed, modality, and order. `all_order_attention.csv` reports all-order attention matrices and entropy. `cross_interventions.csv` reports same-LR normal/off/shuffle frozen interventions.")
        nonzero = responses[responses["correction_norm_mean"] > 0]
        lines.append(f"Rows: {len(responses)}; finite rows: {int(responses['finite'].sum())}; nonzero correction rows: {len(nonzero)}.")
        if not attention.empty:
            summaries = attention[attention["target_order"] == "summary"]
            if not summaries.empty:
                lines.append("")
                lines.append(_markdown_table(summaries.groupby("modality")[['diagonal_mass', 'offdiagonal_mass', 'attention_entropy']].mean().reset_index()))
    lines += ["", "## 5. MRC diagnostics", ""]
    if mrc.empty:
        lines.append("No MRC diagnostics were generated.")
    else:
        lines.append(_markdown_table(mrc[["variant", "dataset", "seed", "weight_gap_mean", "semantic_gap_mean", "incident_mean_abs_gap", "top_neighbor_disagreement"]]))
    lines += ["", "## 6. Stability and limitations", ""]
    if monitor.empty:
        lines.append("Training monitor logs were not available.")
    else:
        bad = monitor[monitor["log_nan_inf"]]
        lines.append(f"Training-monitor rows: {len(monitor)}; rows containing textual NaN/Inf: {len(bad)}. Correction ratio and gradient maxima are in `training_stability.csv`.")
        if not responses.empty:
            lowrank = responses[responses["variant"].isin(["self_lr", "same_lr", "all_lr", "same_lr_mrc"])]
            lines.append(f"Low-rank validation rows with exactly zero correction: {int((lowrank['correction_zero_fraction'] >= 1.0).sum())}/{len(lowrank)}; mean absolute correction norm={lowrank['correction_norm_mean'].mean():.6f}.")
        if not attention.empty:
            attention_summary = attention[attention["target_order"].astype(str) == "summary"]
            if not attention_summary.empty:
                lines.append(f"All-order attention is close to uniform: mean diagonal mass={attention_summary['diagonal_mass'].mean():.4f}, entropy={attention_summary['attention_entropy'].mean():.4f} (log(3)={math.log(3.0):.4f}).")
        if not mrc.empty:
            lines.append(f"MRC edge-weight saturation fractions (lower/upper) are {mrc[['text_weight_lower_saturation', 'text_weight_upper_saturation', 'visual_weight_lower_saturation', 'visual_weight_upper_saturation']].mean().mean():.4f} on average; projected semantic-gap vs weight-gap Pearson={mrc['semantic_weight_gap_pearson'].mean():.4f}.")
    lines += ["", "## 7. Answers to the ten P3 questions", ""]
    if summary.empty or paired_summary.empty:
        lines.append("The ten questions cannot be answered because the formal validation checkpoint set is incomplete.")
    else:
        def comparison_sentence(label: str, question: str) -> str:
            rows = paired_summary[(paired_summary["comparison"] == label) & (paired_summary["dataset"] == "ALL") & (paired_summary["metric"] == "val_acc")]
            if rows.empty:
                return f"{question}: unavailable."
            row = rows.iloc[0]
            return f"{question}: mean paired Δ={row['mean']:.4f}, win/tie/loss fractions={row['positive_fraction']:.3f}/{row['zero_fraction']:.3f}/{row['negative_fraction']:.3f} (validation accuracy)."
        lines.append("1. " + comparison_sentence("self_lr-Plain", "Self response adaptation vs Plain"))
        lines.append("2. " + comparison_sentence("Same-LR-Self-LR", "Same-order cross-modal conditioning vs Self"))
        lines.append("3. " + comparison_sentence("Same-LR-Same-Scalar", "Low-rank vs scalar gate"))
        lines.append("4. " + comparison_sentence("Same-LR-Same-Scalar", "Directional low-rank transformation"))
        lines.append("5. " + comparison_sentence("All-LR-Same-LR", "All-order context"))
        lines.append("6. " + comparison_sentence("P3-MRC-Plain", "MRC alone"))
        lines.append("7. " + comparison_sentence("Same-LR-MRC-Same-LR", "MRC plus response adaptation"))
        if not cross.empty:
            normal = cross[cross["intervention"] == "normal"].set_index(["dataset", "seed"])
            off = cross[cross["intervention"] == "off"].set_index(["dataset", "seed"])
            shuffle = cross[cross["intervention"] == "shuffle"].set_index(["dataset", "seed"])
            off_drop = (normal["val_acc"] - off["val_acc"]).mean() if not normal.empty and not off.empty else float("nan")
            shuffle_drop = (normal["val_acc"] - shuffle["val_acc"]).mean() if not normal.empty and not shuffle.empty else float("nan")
            off_l2 = (off["mean_z_l2_shift"]).mean() if not off.empty else float("nan")
            shuffle_l2 = (shuffle["mean_z_l2_shift"]).mean() if not shuffle.empty else float("nan")
            lines.append(f"8. Matched cross-modal evidence: normal-minus-cross-off mean validation-accuracy shift={off_drop:.4f}; normal-minus-cross-shuffle={shuffle_drop:.4f}; off/shuffle representation L2 shifts={off_l2:.6f}/{shuffle_l2:.6f}.")
        else:
            lines.append("8. Matched cross-modal evidence: unavailable.")
        if not responses.empty:
            nonzero = responses[responses["correction_norm_mean"] > 1.0e-8]
            cosine_mean = float(nonzero["correction_cosine_mean"].mean()) if not nonzero.empty else float("nan")
            positive = float((nonzero["correction_cosine_mean"] > 0.1).mean()) if not nonzero.empty else float("nan")
            negative = float((nonzero["correction_cosine_mean"] < -0.1).mean()) if not nonzero.empty else float("nan")
            lines.append(f"9. Correction direction: mean correction-vs-response cosine={cosine_mean:.4f}; positive (>0.1) rows={positive:.3f}, negative (<-0.1) rows={negative:.3f}; the remainder is directional/near-orthogonal.")
        else:
            lines.append("9. Correction direction: unavailable.")
        candidate = summary.groupby("variant", as_index=False)["val_acc"].mean().sort_values("val_acc", ascending=False).iloc[0]
        lines.append(f"10. Validation-only P4 candidate by mean validation accuracy is `{candidate['variant']}` ({candidate['val_acc']:.4f}); this is not a final benchmark decision and uses no test result.")
        lines.append("Recommendation: keep `same_lr` as the only conservative P4 follow-up candidate, but do not claim a validated CSSI mechanism yet; its gain is small, cross-off/shuffle changes representations without changing validation decisions, and all-order attention is nearly uniform.")
    lines += ["", "A positive paired difference means the first named variant has higher validation metric. These are functional validation comparisons, not causal effects. No P4 mechanism is implemented by this turn."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p3"))
    parser.add_argument("--p1-root", type=Path, default=Path("outputs/cssi_p1"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    p1_root = args.p1_root if args.p1_root.is_absolute() else project_root / args.p1_root
    if args.report_only:
        def read_csv(name: str) -> pd.DataFrame:
            path = root / name
            return pd.read_csv(path) if path.is_file() and path.stat().st_size > 1 else pd.DataFrame()
        summary = read_csv("summary.csv")
        _, paired = _base_rows(root, p1_root)
        paired_frame = pd.DataFrame(paired)
        _write_frame(paired_frame, root / "paired_differences.csv")
        paired_summary = _aggregate_paired(paired_frame)
        _write_frame(paired_summary, root / "paired_differences_aggregate.csv")
        responses = read_csv("response_diagnostics.csv")
        attention = read_csv("all_order_attention.csv")
        mrc = read_csv("mrc_diagnostics.csv")
        cross = read_csv("cross_interventions.csv")
        monitor_rows = []
        for dataset in DATASETS:
            for seed in SEEDS:
                for variant in VARIANTS:
                    run_dir = _run_path(root, variant, dataset, seed)
                    if _compatible_run(run_dir):
                        monitor_rows.append(_monitor_row(run_dir, variant, dataset, seed))
        monitor = pd.DataFrame(monitor_rows)
        _write_frame(monitor, root / "training_stability.csv")
        _markdown_report(project_root / "docs/cssi_p3_results.md", summary, paired_summary, responses, attention, mrc, cross, monitor)
        print(json.dumps({"report_only": True, "summary_rows": len(summary)}, indent=2))
        return
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    rows, paired = _base_rows(root, p1_root)
    summary = pd.DataFrame(rows)
    paired_frame = pd.DataFrame(paired)
    _write_frame(summary, root / "summary.csv")
    _write_frame(paired_frame, root / "paired_differences.csv")
    paired_summary = _aggregate_paired(paired_frame)
    _write_frame(paired_summary, root / "paired_differences_aggregate.csv")

    response_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    mrc_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for variant in VARIANTS:
                run_dir = _run_path(root, variant, dataset, seed)
                if _compatible_run(run_dir):
                    monitor_rows.append(_monitor_row(run_dir, variant, dataset, seed))
                loaded = _load_model(root, variant, dataset, seed, device)
                if loaded is None:
                    continue
                cfg, data, model, classifier, checkpoint = loaded
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                components = model.analysis_components(x, edge_index)
                response_rows.extend(_response_rows(components, data, variant, dataset, seed, device))
                if variant == "all_lr":
                    attention_rows.extend(_attention_rows(components, data, dataset, seed, device))
                if variant in {"mrc_plain", "same_lr_mrc"}:
                    mrc_rows.append(_mrc_row(components, data, variant, dataset, seed))
                if variant == "same_lr":
                    cross_rows.extend(_cross_rows(components, data, model, classifier, x, edge_index, dataset, seed, device))
                del components, model, classifier, data, x, edge_index
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    response_frame = pd.DataFrame(response_rows)
    attention_frame = pd.DataFrame(attention_rows)
    mrc_frame = pd.DataFrame(mrc_rows)
    cross_frame = pd.DataFrame(cross_rows)
    monitor_frame = pd.DataFrame(monitor_rows)
    _write_frame(response_frame, root / "response_diagnostics.csv")
    _write_frame(attention_frame, root / "all_order_attention.csv")
    _write_frame(mrc_frame, root / "mrc_diagnostics.csv")
    _write_frame(cross_frame, root / "cross_interventions.csv")
    _write_frame(monitor_frame, root / "training_stability.csv")
    _markdown_report(project_root / "docs/cssi_p3_results.md", summary, paired_summary, response_frame, attention_frame, mrc_frame, cross_frame, monitor_frame)
    print(json.dumps({
        "device": str(device), "summary_rows": len(summary), "response_rows": len(response_frame),
        "attention_rows": len(attention_frame), "mrc_rows": len(mrc_frame),
        "cross_rows": len(cross_frame), "complete_checkpoints": int(len(summary)),
    }, indent=2))


if __name__ == "__main__":
    main()
