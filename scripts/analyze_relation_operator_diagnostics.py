from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import f1_score
from torch_geometric.utils import remove_self_loops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_relation_operator_audit as base  # noqa: E402

OUT = ROOT / "results/relation_operator_diagnostics_v1"
EPS = 1e-6
LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
RANDOM_REPEATS = 20

D1_INTERVENTION_FIELDS = [
    "dataset", "seed", "modality_scope", "intervention", "val_acc", "val_macro_f1",
    "mean_delta_ce", "median_delta_ce", "p10_delta_ce", "p90_delta_ce",
    "harmed_fraction", "improved_fraction", "near_zero_fraction", "mean_margin_utility",
    "median_margin_utility", "prediction_flip_rate", "mean_route_l1_change",
    "test_evaluated", "status",
]
D1_NODE_FIELDS = [
    "dataset", "seed", "modality_scope", "intervention", "node_id", "label",
    "normal_ce", "intervention_ce", "delta_ce", "normal_margin", "intervention_margin",
    "margin_utility", "normal_prediction", "intervention_prediction", "prediction_flip",
    "normal_logits", "intervention_logits",
]
NODE_SUMMARY_FIELDS = [
    "dataset", "seed", "modality", "modality_scope", "intervention", "population",
    "n_nodes", "mean_delta_ce", "median_delta_ce", "p10_delta_ce", "p90_delta_ce",
    "harmed_fraction", "improved_fraction", "near_zero_fraction", "mean_margin_utility",
    "median_margin_utility", "prediction_flip_rate", "val_acc", "val_macro_f1",
]
D2_SHRINK_FIELDS = [
    "dataset", "seed", "modality_scope", "lambda", "val_acc", "val_macro_f1",
    "mean_delta_ce_vs_lambda1", "median_delta_ce_vs_lambda1", "p10_delta_ce_vs_lambda1",
    "p90_delta_ce_vs_lambda1", "harmed_fraction", "improved_fraction", "near_zero_fraction",
    "mean_margin_utility", "median_margin_utility", "prediction_flip_rate",
    "best_lambda_by_mean_ce", "best_lambda_by_val_accuracy", "test_evaluated",
]
D2_SHRINK_NODE_FIELDS = [
    "dataset", "seed", "modality_scope", "lambda", "node_id", "label", "normal_ce",
    "intervention_ce", "delta_ce", "normal_margin", "intervention_margin", "margin_utility",
    "normal_prediction", "intervention_prediction", "prediction_flip",
]
D2_TARGET_FIELDS = [
    "dataset", "seed", "modality", "group_type", "group", "intervention_id", "repeat",
    "is_target", "selected_edges", "candidate_edges", "candidate_scope", "fallback_rate", "fallback_3d_count",
    "fallback_2d_count", "fallback_1d_count", "fallback_unmatched_strata_count",
    "mean_delta_ce_all_val", "median_delta_ce_all_val", "p10_delta_ce_all_val",
    "p90_delta_ce_all_val", "harmed_fraction_all_val", "improved_fraction_all_val",
    "near_zero_fraction_all_val", "mean_margin_utility_all_val", "prediction_flip_rate_all_val",
    "n_validation_val", "n_touched_val", "mean_absolute_delta_ce_all_val",
    "mean_absolute_delta_ce_touched_val", "mean_absolute_margin_utility_all_val",
    "mean_absolute_margin_utility_touched_val",
    "val_acc", "val_macro_f1", "mean_delta_ce_touched_val", "median_delta_ce_touched_val",
    "p10_delta_ce_touched_val", "p90_delta_ce_touched_val", "harmed_fraction_touched_val",
    "improved_fraction_touched_val", "near_zero_fraction_touched_val",
    "mean_margin_utility_touched_val", "prediction_flip_rate_touched_val",
    "mean_message_change_abs", "mean_relative_message_change", "normal_val_acc",
    "normal_val_macro_f1", "delta_val_acc", "delta_val_macro_f1", "test_evaluated",
]
D2_TARGET_NODE_FIELDS = [
    "dataset", "seed", "modality", "group_type", "group", "intervention_id", "repeat",
    "is_target", "node_id", "label", "touched_validation_node", "normal_ce", "intervention_ce",
    "delta_ce", "normal_margin", "intervention_margin", "margin_utility", "normal_prediction",
    "intervention_prediction", "prediction_flip",
]
D2_RANDOM_FIELDS = [
    "dataset", "seed", "modality", "group_type", "group", "population", "n_repeats",
    "requested_repeats", "n_empty_touched_controls",
    "target_mean_utility", "random_mean_utility", "random_std_utility", "excess_utility",
    "target_percentile_in_random", "two_sided_empirical_extremeness", "target_mean_message_change_abs",
    "random_mean_message_change_abs", "excess_utility_per_target_message_change",
    "excess_utility_per_random_message_change", "test_evaluated",
]
MATCH_FIELDS = [
    "dataset", "seed", "modality", "group_type", "group", "repeat", "selected_edges",
    "candidate_edges", "candidate_scope", "fallback_rate", "fallback_3d_count", "fallback_2d_count",
    "fallback_1d_count", "fallback_unmatched_strata_count", "source_degree_mean_diff",
    "source_degree_smd", "target_degree_mean_diff", "target_degree_smd",
    "personalization_magnitude_mean_diff", "personalization_magnitude_smd",
    "source_degree_bin_tvd", "target_degree_bin_tvd", "personalization_bin_tvd",
    "no_overlap", "duplicate_edges", "test_evaluated",
]
D2_CONCENTRATION_FIELDS = [
    "dataset", "seed", "modality", "group_type", "group", "n_validation_val",
    "n_touched_val", "touched_validation_fraction", "mean_absolute_delta_ce_all_val",
    "mean_absolute_delta_ce_touched_val", "harmed_fraction_all_val",
    "improved_fraction_all_val", "near_zero_fraction_all_val",
    "harmed_fraction_touched_val", "improved_fraction_touched_val",
    "near_zero_fraction_touched_val", "mean_absolute_margin_utility_all_val",
    "mean_absolute_margin_utility_touched_val",
]
STRUCT_SUMMARY_FIELDS = [
    "dataset", "seed", "modality", "analysis", "descriptor", "n_edges", "statistic",
    "value", "p_value", "group", "test_evaluated",
]
STRUCT_EDGE_FIELDS = [
    "dataset", "seed", "modality", "target", "source", "semantic_quartile",
    "personalization_quartile", "semantic_similarity", "personalization_deviation_l1",
    "norm_D_target", "norm_D_source", "norm_D_difference", "cos_D_target_source",
    "cos_H0_target_source",
]


class GzipCsvWriter:
    def __init__(self, path: Path, fields: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=6)
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        self.writer.writeheader()

    def writerow(self, row: dict) -> None:
        self.writer.writerow(row)

    def writerows(self, rows) -> None:
        self.writer.writerows(rows)

    def close(self) -> None:
        self.handle.close()


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)



def _read_csv_rows(path: Path) -> list[dict]:
    def parse(value):
        if value == "":
            return None
        if value in {"True", "False"}:
            return value == "True"
        try:
            if value.lstrip("-").isdigit():
                return int(value)
            return float(value)
        except (ValueError, AttributeError):
            return value

    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _target_concentration_from_node_file(path: Path) -> dict[tuple, dict]:
    """Stream target intervention nodes into per-context touched/all absolute utility summaries."""
    keys = ("dataset", "seed", "modality", "group_type", "group")
    stats = defaultdict(lambda: {
        "n_validation_val": 0, "n_touched_val": 0,
        "sum_abs_ce_all": 0.0, "sum_abs_ce_touched": 0.0,
        "sum_abs_margin_all": 0.0, "sum_abs_margin_touched": 0.0,
        "harmed_all": 0, "improved_all": 0, "near_all": 0,
        "harmed_touched": 0, "improved_touched": 0, "near_touched": 0,
    })
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["is_target"]) != 1:
                continue
            key = tuple(row[name] for name in keys)
            item = stats[key]
            delta = float(row["delta_ce"])
            margin = float(row["margin_utility"])
            touched = int(row["touched_validation_node"]) == 1
            item["n_validation_val"] += 1
            item["sum_abs_ce_all"] += abs(delta)
            item["sum_abs_margin_all"] += abs(margin)
            item["harmed_all"] += int(delta > EPS)
            item["improved_all"] += int(delta < -EPS)
            item["near_all"] += int(abs(delta) <= EPS)
            if touched:
                item["n_touched_val"] += 1
                item["sum_abs_ce_touched"] += abs(delta)
                item["sum_abs_margin_touched"] += abs(margin)
                item["harmed_touched"] += int(delta > EPS)
                item["improved_touched"] += int(delta < -EPS)
                item["near_touched"] += int(abs(delta) <= EPS)
    out = {}
    for key, item in stats.items():
        n_all = item["n_validation_val"]
        n_touched = item["n_touched_val"]
        out[key] = {
            "n_validation_val": n_all,
            "n_touched_val": n_touched,
            "mean_absolute_delta_ce_all_val": item["sum_abs_ce_all"] / n_all if n_all else None,
            "mean_absolute_delta_ce_touched_val": item["sum_abs_ce_touched"] / n_touched if n_touched else None,
            "mean_absolute_margin_utility_all_val": item["sum_abs_margin_all"] / n_all if n_all else None,
            "mean_absolute_margin_utility_touched_val": item["sum_abs_margin_touched"] / n_touched if n_touched else None,
            "harmed_fraction_all_val": item["harmed_all"] / n_all if n_all else None,
            "improved_fraction_all_val": item["improved_all"] / n_all if n_all else None,
            "near_zero_fraction_all_val": item["near_all"] / n_all if n_all else None,
            "harmed_fraction_touched_val": item["harmed_touched"] / n_touched if n_touched else None,
            "improved_fraction_touched_val": item["improved_touched"] / n_touched if n_touched else None,
            "near_zero_fraction_touched_val": item["near_touched"] / n_touched if n_touched else None,
        }
    return out


def _target_concentration_rows(target_rows: list[dict]) -> list[dict]:
    rows = []
    for row in target_rows:
        if int(row["is_target"]) != 1:
            continue
        n_all = int(row.get("n_validation_val") or 0)
        n_touched = int(row.get("n_touched_val") or 0)
        out = {name: row.get(name) for name in D2_CONCENTRATION_FIELDS}
        out["touched_validation_fraction"] = n_touched / n_all if n_all else None
        rows.append(out)
    return rows


def _matching_quality_summary(match_rows: list[dict]) -> dict:
    if not match_rows:
        return {"n_repeats": 0}
    selected = sum(int(row["selected_edges"]) for row in match_rows)
    relaxed = sum(
        int(row["fallback_2d_count"]) + int(row["fallback_1d_count"])
        + int(row["fallback_unmatched_strata_count"])
        for row in match_rows
    )
    summary = {
        "n_repeats": len(match_rows),
        "n_relaxed_repeats": sum(float(row["fallback_rate"]) > 0 for row in match_rows),
        "selected_edges": selected,
        "relaxed_edges": relaxed,
        "relaxed_edge_fraction": relaxed / selected if selected else 0.0,
        "no_overlap_failures": sum(int(row["no_overlap"]) != 1 for row in match_rows),
        "duplicate_edges": sum(int(row["duplicate_edges"]) for row in match_rows),
    }
    for name in ("source_degree_smd", "target_degree_smd", "personalization_magnitude_smd",
                 "source_degree_bin_tvd", "target_degree_bin_tvd", "personalization_bin_tvd"):
        values = np.asarray([abs(float(row[name])) for row in match_rows], dtype=np.float64)
        summary[f"mean_abs_{name}"] = float(values.mean())
        summary[f"p95_abs_{name}"] = float(np.quantile(values, 0.95))
        summary[f"max_abs_{name}"] = float(values.max())
    return summary


def _refresh_existing_results() -> None:
    """Rebuild expanded summaries/report from completed artifacts without loading checkpoints."""
    report_path = OUT / "relation_operator_diagnostics_report.md"
    previous_report = report_path.read_text(encoding="utf-8")
    elapsed_match = re.search(r"Elapsed analysis time: ([0-9.]+) s", previous_report)
    peak_match = re.search(r"Peak allocated GPU memory recorded by PyTorch: ([0-9.]+) MiB", previous_report)
    if not elapsed_match or not peak_match:
        raise RuntimeError("existing report must contain runtime and peak-memory records for summary refresh")

    d1_rows = _read_csv_rows(OUT / "d1_global_operator_interventions.csv")
    d1_decisions = _d1_decision(d1_rows)
    shrink_rows = _read_csv_rows(OUT / "d2_shrinkage_sweep.csv")
    shrink_contexts = [row for row in shrink_rows if row["dataset"] != "ALL"]
    shrink_rows = shrink_contexts + _aggregate_shrinkage(shrink_contexts)
    target_rows = _read_csv_rows(OUT / "d2_targeted_personalization.csv")
    aggregates = _target_concentration_from_node_file(OUT / "d2_targeted_node_utility.csv.gz")
    keys = ("dataset", "seed", "modality", "group_type", "group")
    aggregate_lookup = {tuple(str(row[name]) for name in keys): row for row in target_rows if int(row["is_target"]) == 1}
    for key, values in aggregates.items():
        row = aggregate_lookup.get(tuple(str(value) for value in key))
        if row is None:
            raise RuntimeError(f"target node utility summary has no matching target context: {key}")
        row.update(values)
    concentration_rows = _target_concentration_rows(target_rows)
    random_rows = _read_csv_rows(OUT / "d2_random_control_summary.csv")
    structural_rows = _read_csv_rows(OUT / "d2_structural_context_descriptives.csv")
    match_rows = _read_csv_rows(OUT / "d2_matching_quality.csv")
    d2_decisions = _d2_decision(shrink_rows, random_rows, structural_rows, target_rows)
    _write_csv(OUT / "d1_decision.csv", d1_decisions)
    _write_csv(OUT / "d2_shrinkage_sweep.csv", shrink_rows, D2_SHRINK_FIELDS)
    _write_csv(OUT / "d2_targeted_personalization.csv", target_rows, D2_TARGET_FIELDS)
    _write_csv(OUT / "d2_node_utility_concentration.csv", concentration_rows, D2_CONCENTRATION_FIELDS)
    _write_csv(OUT / "d2_decision.csv", d2_decisions)
    _write_report(
        d1_rows, d1_decisions, shrink_rows, target_rows, random_rows, structural_rows,
        match_rows, d2_decisions, RANDOM_REPEATS, float(elapsed_match.group(1)),
        float(peak_match.group(1)),
    )

def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


def _assert_frozen_payload(config: dict, payload: dict, ckpt_path: Path) -> None:
    task = config.get("task", {})
    if task.get("evaluate_test") is not False:
        raise RuntimeError(f"test evaluation was enabled in {ckpt_path.parent.parent / 'resolved_config.json'}")
    for section_name, section in (("checkpoint metrics", payload.get("metrics", {})),):
        if any("test" in str(key).lower() for key in section):
            raise RuntimeError(f"test metric present in {section_name} for {ckpt_path}")


@torch.no_grad()
def _prepare_frozen(model, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
    """Compute the current model's normal H0, physical operator, and routes once."""
    model.eval()
    operator = model._get_operator(edge_index, x.size(0), x.dtype)
    target, source, _, _, _, _ = model._operator_edges(operator)
    h0_text = model.text_projector(x[:, : model.text_dim])
    h0_visual = model.visual_projector(x[:, model.text_dim : model.text_dim + model.visual_dim])
    route_text, _ = model._edge_controls("text", h0_text, target, source)
    route_visual, _ = model._edge_controls("visual", h0_visual, target, source)
    return {
        "operator": operator, "target": target, "source": source,
        "h0_text": h0_text, "h0_visual": h0_visual,
        "route_text": route_text, "route_visual": route_visual,
    }


@torch.no_grad()
def _forward_routes(model, prepared: dict, route_text: torch.Tensor, route_visual: torch.Tensor) -> torch.Tensor:
    states_text, _ = model._propagate_one(
        prepared["h0_text"], prepared["operator"], route_text, modality="text"
    )
    states_visual, _ = model._propagate_one(
        prepared["h0_visual"], prepared["operator"], route_visual, modality="visual"
    )
    z_text = torch.stack(states_text, dim=0).mean(dim=0)
    z_visual = torch.stack(states_visual, dim=0).mean(dim=0)
    return model.plain_fusion(torch.cat([z_text, z_visual], dim=-1))


def _validation_arrays(head, embedding, data, eval_labels: list[int]) -> dict:
    val_idx = data.val_idx.to(embedding.device)
    # Only Validation labels are indexed. The dataset's Test labels are not read.
    labels = data.y[data.val_idx].to(embedding.device).long()
    logits = head(embedding[val_idx])
    predictions = logits.argmax(dim=-1)
    ce = F.cross_entropy(logits, labels, reduction="none")
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    margin = true - masked.max(dim=-1).values
    return {
        "val_idx": val_idx,
        "labels": labels,
        "logits": logits,
        "predictions": predictions,
        "ce": ce,
        "margin": margin,
        "acc": float((predictions == labels).float().mean().item()),
        "macro_f1": float(f1_score(
            labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(),
            labels=eval_labels, average="macro", zero_division=0,
        )),
    }


def _node_deltas(normal: dict, changed: dict) -> dict:
    delta_ce = changed["ce"] - normal["ce"]
    margin_utility = normal["margin"] - changed["margin"]
    return {
        "delta_ce": delta_ce,
        "margin_utility": margin_utility,
        "prediction_flip": changed["predictions"] != normal["predictions"],
    }


def _summary_values(normal: dict, changed: dict, select: torch.Tensor | None = None) -> dict:
    utility = _node_deltas(normal, changed)
    ids = torch.ones_like(utility["delta_ce"], dtype=torch.bool) if select is None else select
    dce = utility["delta_ce"][ids]
    margin = utility["margin_utility"][ids]
    flip = utility["prediction_flip"][ids]
    if not dce.numel():
        return {"n_nodes": 0}
    q = torch.quantile(dce, torch.tensor([0.1, 0.5, 0.9], device=dce.device))
    return {
        "n_nodes": int(dce.numel()),
        "mean_delta_ce": float(dce.mean().item()),
        "mean_absolute_delta_ce": float(dce.abs().mean().item()),
        "median_delta_ce": float(q[1].item()),
        "p10_delta_ce": float(q[0].item()),
        "p90_delta_ce": float(q[2].item()),
        "harmed_fraction": float((dce > EPS).float().mean().item()),
        "improved_fraction": float((dce < -EPS).float().mean().item()),
        "near_zero_fraction": float((dce.abs() <= EPS).float().mean().item()),
        "mean_margin_utility": float(margin.mean().item()),
        "mean_absolute_margin_utility": float(margin.abs().mean().item()),
        "median_margin_utility": float(margin.median().item()),
        "prediction_flip_rate": float(flip.float().mean().item()),
    }


def _metrics_and_raw_rows(
    normal: dict, changed: dict, *, dataset: str, seed: int, scope: str, intervention: str,
) -> tuple[dict, list[dict]]:
    summary = _summary_values(normal, changed)
    utility = _node_deltas(normal, changed)
    rows = []
    ids = normal["val_idx"].detach().cpu().tolist()
    nlog = normal["logits"].detach().cpu().tolist()
    ilog = changed["logits"].detach().cpu().tolist()
    labels = normal["labels"].detach().cpu().tolist()
    nce = normal["ce"].detach().cpu().tolist()
    ice = changed["ce"].detach().cpu().tolist()
    nm = normal["margin"].detach().cpu().tolist()
    im = changed["margin"].detach().cpu().tolist()
    npred = normal["predictions"].detach().cpu().tolist()
    ipred = changed["predictions"].detach().cpu().tolist()
    dce = utility["delta_ce"].detach().cpu().tolist()
    mu = utility["margin_utility"].detach().cpu().tolist()
    for i, node_id in enumerate(ids):
        rows.append({
            "dataset": dataset, "seed": seed, "modality_scope": scope,
            "intervention": intervention, "node_id": node_id, "label": labels[i],
            "normal_ce": nce[i], "intervention_ce": ice[i], "delta_ce": dce[i],
            "normal_margin": nm[i], "intervention_margin": im[i], "margin_utility": mu[i],
            "normal_prediction": npred[i], "intervention_prediction": ipred[i],
            "prediction_flip": int(npred[i] != ipred[i]),
            "normal_logits": json.dumps(nlog[i], separators=(",", ":")),
            "intervention_logits": json.dumps(ilog[i], separators=(",", ":")),
        })
    return {
        **summary, "val_acc": changed["acc"], "val_macro_f1": changed["macro_f1"],
    }, rows


def _route_for_d1(route: torch.Tensor, intervention: str) -> torch.Tensor:
    if intervention == "zero":
        return torch.zeros_like(route)
    if intervention == "uniform":
        return torch.full_like(route, 1.0 / route.numel())
    if intervention == "top1":
        out = torch.zeros_like(route)
        out[route.argmax()] = 1.0
        return out
    if intervention.startswith("single_e"):
        out = torch.zeros_like(route)
        out[int(intervention.removeprefix("single_e"))] = 1.0
        return out
    if intervention.startswith("drop_e"):
        expert = int(intervention.removeprefix("drop_e"))
        out = route.clone()
        out[expert] = 0.0
        total = out.sum()
        return out / total if float(total.item()) > 0 else out
    raise ValueError(f"unknown D1 route intervention {intervention!r}")


def _replace_modality_routes(base_t, base_v, modality_scope: str, intervention: str):
    text, visual = base_t, base_v
    if modality_scope in {"text_only", "both"}:
        text = _route_for_d1(base_t, intervention)
    if modality_scope in {"visual_only", "both"}:
        visual = _route_for_d1(base_v, intervention)
    return text, visual


def _route_change(base_route: torch.Tensor, changed_route: torch.Tensor) -> float:
    return float((base_route - changed_route).abs().mean().item())


def _d1_scenarios(route_text: torch.Tensor, route_visual: torch.Tensor):
    scenarios = []
    for intervention in ("zero", "uniform", "top1"):
        for scope in ("text_only", "visual_only", "both"):
            scenarios.append((scope, intervention))
    for modality in ("text_only", "visual_only"):
        for expert in range(4):
            scenarios.append((modality, f"single_e{expert}"))
            scenarios.append((modality, f"drop_e{expert}"))
    return scenarios


def _make_semantic_bins(a0_model, a0_h0, data, target, source, modality: str):
    train_nodes = torch.zeros(data.num_nodes, dtype=torch.bool, device=target.device)
    val_nodes = torch.zeros_like(train_nodes)
    train_nodes[data.train_idx.to(target.device)] = True
    val_nodes[data.val_idx.to(target.device)] = True
    sims = []
    for start in range(0, target.numel(), a0_model.edge_chunk_size):
        end = min(start + a0_model.edge_chunk_size, target.numel())
        sims.append(F.cosine_similarity(
            a0_h0[target[start:end]], a0_h0[source[start:end]], dim=-1
        ))
    similarity = torch.cat(sims) if sims else a0_h0.new_empty((0,))
    train_edges = train_nodes[target] & train_nodes[source]
    val_related = val_nodes[target] | val_nodes[source]
    if not train_edges.any():
        raise RuntimeError("no Train–Train non-self edges available for semantic quartiles")
    thresholds = torch.quantile(
        similarity[train_edges], torch.tensor([0.25, 0.5, 0.75], device=target.device)
    )
    positions = torch.nonzero(val_related, as_tuple=False).flatten()
    return {
        "thresholds": thresholds,
        "positions": positions,
        "bins": torch.bucketize(similarity[positions], thresholds),
        "similarity": similarity,
        "train_similarity": similarity[train_edges],
        "target": target,
        "source": source,
        "train_edges": train_edges,
        "reference": "plain_A0_H0",
        "modality": modality,
    }


def _quantile_bins(values: torch.Tensor, train_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not train_mask.any():
        raise RuntimeError("no Train–Train values available for quartile thresholds")
    thresholds = torch.quantile(
        values[train_mask], torch.tensor([0.25, 0.5, 0.75], device=values.device)
    )
    return thresholds, torch.bucketize(values, thresholds)


def _edge_degrees(num_nodes: int, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=target.device)
    degree.index_add_(0, target, torch.ones_like(target, dtype=torch.float32))
    return degree.clamp_min(1.0)


def _d2_edge_context(prepared: dict, data, route: torch.Tensor, modality: str, semantic: dict):
    target, source = prepared["target"], prepared["source"]
    train_nodes = torch.zeros(data.num_nodes, dtype=torch.bool, device=target.device)
    val_nodes = torch.zeros_like(train_nodes)
    train_nodes[data.train_idx.to(target.device)] = True
    val_nodes[data.val_idx.to(target.device)] = True
    train_edges = train_nodes[target] & train_nodes[source]
    val_related = val_nodes[target] | val_nodes[source]
    global_route = route.mean(dim=0)
    deviation = (route - global_route.unsqueeze(0)).abs().sum(dim=-1)
    dev_thresholds, dev_bins = _quantile_bins(deviation, train_edges)
    degree = _edge_degrees(data.num_nodes, target, source)
    log_degree = degree.log()
    src_thresholds, src_bins = _quantile_bins(log_degree[source], train_edges)
    dst_thresholds, dst_bins = _quantile_bins(log_degree[target], train_edges)
    val_positions = torch.nonzero(val_related, as_tuple=False).flatten()
    strata = torch.stack((src_bins, dst_bins, dev_bins), dim=-1)
    return {
        "global_route": global_route,
        "deviation": deviation,
        "dev_thresholds": dev_thresholds,
        "dev_bins": dev_bins,
        "train_edges": train_edges,
        "val_positions": val_positions,
        "strata": strata,
        "src_degree": degree[source],
        "dst_degree": degree[target],
        "semantic": semantic,
        "semantic_group_bins": semantic["bins"],
        "target": target,
        "source": source,
    }


def _group_definitions(context: dict):
    positions = context["val_positions"]
    semantic_bins = context["semantic_group_bins"]
    deviation_bins = context["dev_bins"][positions]
    definitions = []
    for q in range(4):
        definitions.append(("semantic_similarity", f"Q{q + 1}", positions[semantic_bins == q]))
        definitions.append(("personalization_deviation", f"D{q + 1}", positions[deviation_bins == q]))
    return definitions


def _tuple_keys(values: np.ndarray, columns: tuple[int, ...]):
    if not columns:
        return [()] * len(values)
    return [tuple(int(row[c]) for c in columns) for row in values]


def _random_candidate_pool(num_edges: int, target_positions: torch.Tensor) -> torch.Tensor:
    """Use the full non-self physical edge population, excluding selected target edges."""
    all_positions = torch.arange(num_edges, dtype=torch.long, device=target_positions.device)
    target_mask = torch.zeros(num_edges, dtype=torch.bool, device=target_positions.device)
    target_mask[target_positions] = True
    return all_positions[~target_mask]


def _sample_stratified_random(
    target_positions: torch.Tensor,
    candidate_positions: torch.Tensor,
    strata: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict]:
    """Match 3D strata first, then relax dimensions without reusing any candidate."""
    tpos = target_positions.detach().cpu().numpy().astype(np.int64, copy=False)
    cpos = candidate_positions.detach().cpu().numpy().astype(np.int64, copy=False)
    sbins = strata.detach().cpu().numpy().astype(np.int8, copy=False)
    if np.intersect1d(tpos, cpos, assume_unique=False).size:
        raise RuntimeError("random candidate pool overlaps target group")
    if cpos.size < tpos.size:
        return torch.empty(0, dtype=torch.long, device=target_positions.device), {
            "ok": False, "candidate_edges": int(cpos.size), "fallback_counts": {},
        }

    unused = np.ones(cpos.size, dtype=bool)
    random_pick = np.full(tpos.size, -1, dtype=np.int64)
    fallback_level = np.full(tpos.size, -1, dtype=np.int8)
    remaining = np.arange(tpos.size, dtype=np.int64)
    candidate_strata = sbins[cpos]
    patterns = [
        ((0, 1, 2), 0),
        ((0, 1), 1), ((0, 2), 1), ((1, 2), 1),
        ((0,), 2), ((1,), 2), ((2,), 2),
        ((), 3),
    ]
    for columns, level in patterns:
        if remaining.size == 0:
            break
        target_keys = _tuple_keys(sbins[tpos[remaining]], columns)
        target_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for local, key in zip(remaining.tolist(), target_keys):
            target_groups[key].append(local)
        cand_keys = _tuple_keys(candidate_strata, columns)
        candidate_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for ci, key in enumerate(cand_keys):
            if unused[ci]:
                candidate_groups[key].append(ci)
        matched_local = []
        for key, requested in target_groups.items():
            available = candidate_groups.get(key, [])
            take = min(len(requested), len(available))
            if not take:
                continue
            req_order = rng.permutation(len(requested))[:take]
            cand_order = rng.permutation(len(available))[:take]
            for ri, ci in zip(req_order.tolist(), cand_order.tolist()):
                local_target = requested[ri]
                candidate_index = available[ci]
                random_pick[local_target] = cpos[candidate_index]
                fallback_level[local_target] = level
                unused[candidate_index] = False
                matched_local.append(local_target)
        if matched_local:
            matched_set = set(matched_local)
            remaining = np.asarray([idx for idx in remaining.tolist() if idx not in matched_set], dtype=np.int64)

    if remaining.size:
        raise RuntimeError("stratified sampler failed despite sufficient non-target candidates")
    if np.any(random_pick < 0) or np.unique(random_pick).size != tpos.size:
        raise RuntimeError("matched sample is incomplete or duplicates candidate edges")
    if np.intersect1d(tpos, random_pick).size:
        raise RuntimeError("matched random edge group overlaps target group")
    fallback_counts = {f"{level}d": int(np.sum(fallback_level == level)) for level in (0, 1, 2, 3)}
    return torch.as_tensor(random_pick, dtype=torch.long, device=target_positions.device), {
        "ok": True, "candidate_edges": int(cpos.size), "fallback_counts": fallback_counts,
        "fallback_rate": float(np.mean(fallback_level > 0)) if tpos.size else 0.0,
    }


def _matching_quality(target_positions, random_positions, strata, source_degree, target_degree, deviation, meta):
    def vals(tensor, positions):
        return tensor[positions].detach().cpu().numpy().astype(np.float64, copy=False)

    target_bins = strata[target_positions].detach().cpu().numpy()
    random_bins = strata[random_positions].detach().cpu().numpy()
    row = {
        "selected_edges": int(target_positions.numel()),
        "candidate_edges": int(meta["candidate_edges"]),
        "candidate_scope": "all_nonself_physical_edges_excluding_target",
        "fallback_rate": meta["fallback_rate"],
        "fallback_3d_count": meta["fallback_counts"]["0d"],
        "fallback_2d_count": meta["fallback_counts"]["1d"],
        "fallback_1d_count": meta["fallback_counts"]["2d"],
        "fallback_unmatched_strata_count": meta["fallback_counts"]["3d"],
        "no_overlap": int(np.intersect1d(
            target_positions.detach().cpu().numpy(), random_positions.detach().cpu().numpy()
        ).size == 0),
        "duplicate_edges": int(random_positions.numel() - random_positions.unique().numel()),
    }
    for name, tensor, col in (
        ("source_degree", source_degree, 0),
        ("target_degree", target_degree, 1),
        ("personalization_magnitude", deviation, 2),
    ):
        a, b = vals(tensor, target_positions), vals(tensor, random_positions)
        pooled = math.sqrt((float(a.var()) + float(b.var())) / 2.0)
        row[f"{name}_mean_diff"] = float(b.mean() - a.mean())
        row[f"{name}_smd"] = float((b.mean() - a.mean()) / (pooled + 1e-12))
        ta = np.bincount(target_bins[:, col], minlength=4).astype(float) / max(len(target_bins), 1)
        rb = np.bincount(random_bins[:, col], minlength=4).astype(float) / max(len(random_bins), 1)
        row[f"{name.replace('_magnitude', '')}_bin_tvd"] = float(0.5 * np.abs(ta - rb).sum())
    return row


def _compute_touched_mask(data, target, source, selected_positions, device):
    touched = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    if selected_positions.numel():
        edges = selected_positions.to(device)
        edge_target, edge_source = target[edges], source[edges]
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        val_mask[data.val_idx.to(device)] = True
        touched[edge_target[val_mask[edge_target]]] = True
        touched[edge_source[val_mask[edge_source]]] = True
    return touched[data.val_idx.to(device)]


def _shrink_route(route: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    global_route = route.mean(dim=0)
    if float(lam) == 1.0:
        return route, global_route
    if float(lam) == 0.0:
        return global_route.unsqueeze(0).expand_as(route), global_route
    return global_route.unsqueeze(0) + float(lam) * (route - global_route.unsqueeze(0)), global_route


def _route_to_global(route: torch.Tensor, global_route: torch.Tensor, selected_positions: torch.Tensor):
    if not selected_positions.numel():
        raise ValueError("cannot intervene on an empty edge group")
    out = route.clone()
    selected_mask = torch.zeros(route.size(0), dtype=torch.bool, device=route.device)
    selected_mask[selected_positions] = True
    out[selected_positions] = global_route.unsqueeze(0).expand(selected_positions.numel(), -1)
    if not torch.equal(out[~selected_mask], route[~selected_mask]):
        raise RuntimeError("unselected routes changed during route-to-global intervention")
    return out


@torch.no_grad()
def _edge_message_change_vectors(model, prepared, route, global_route, modality: str):
    state = prepared[f"h0_{modality}"]
    source = prepared["source"]
    experts = getattr(model, f"experts_{modality}")
    abs_parts, rel_parts = [], []
    for start in range(0, source.numel(), model.edge_chunk_size):
        end = min(start + model.edge_chunk_size, source.numel())
        src_state = state[source[start:end]]
        expert_values = torch.stack([expert(src_state) for expert in experts], dim=1)
        route_delta = route[start:end] - global_route
        residual_delta = (expert_values * route_delta.unsqueeze(-1)).sum(dim=1)
        normal_residual = (expert_values * route[start:end].unsqueeze(-1)).sum(dim=1)
        base_message = src_state + normal_residual
        abs_parts.append(residual_delta.norm(dim=-1))
        rel_parts.append(residual_delta.norm(dim=-1) / (base_message.norm(dim=-1) + 1e-12))
    return torch.cat(abs_parts), torch.cat(rel_parts)


def _structural_rows(
    dataset: str, seed: int, modality: str, h0: torch.Tensor, target: torch.Tensor, source: torch.Tensor,
    operator: torch.Tensor, data, val_positions: torch.Tensor, semantic_bins: torch.Tensor,
    semantic_similarity: torch.Tensor, deviation: torch.Tensor, deviation_bins: torch.Tensor,
    raw_writer: GzipCsvWriter,
) -> list[dict]:
    train_nodes = torch.zeros(data.num_nodes, dtype=torch.bool, device=target.device)
    val_nodes = torch.zeros_like(train_nodes)
    train_nodes[data.train_idx.to(target.device)] = True
    val_nodes[data.val_idx.to(target.device)] = True
    train_edges = train_nodes[target] & train_nodes[source]
    sparse_index = operator.indices()
    sparse_values = operator.values()
    keep = sparse_index[0] != sparse_index[1]
    nbr_operator = torch.sparse_coo_tensor(
        sparse_index[:, keep], sparse_values[keep], operator.shape,
        dtype=operator.dtype, device=operator.device,
    ).coalesce()
    g = torch.sparse.mm(nbr_operator, h0)
    response = g - h0
    edge_t, edge_s = target[val_positions], source[val_positions]
    dev = deviation[val_positions]
    d_t, d_s = response[edge_t], response[edge_s]
    h_t, h_s = h0[edge_t], h0[edge_s]
    n_t, n_s = d_t.norm(dim=-1), d_s.norm(dim=-1)
    n_diff = (d_t - d_s).norm(dim=-1)
    cos_d = F.cosine_similarity(d_t, d_s, dim=-1)
    cos_h = F.cosine_similarity(h_t, h_s, dim=-1)
    arrays = [
        val_positions.detach().cpu().tolist(), edge_t.detach().cpu().tolist(), edge_s.detach().cpu().tolist(),
        semantic_bins.detach().cpu().tolist(), deviation_bins[val_positions].detach().cpu().tolist(),
        semantic_similarity.detach().cpu().tolist(),
        dev.detach().cpu().tolist(), n_t.detach().cpu().tolist(), n_s.detach().cpu().tolist(),
        n_diff.detach().cpu().tolist(), cos_d.detach().cpu().tolist(), cos_h.detach().cpu().tolist(),
    ]
    for i, position in enumerate(arrays[0]):
        raw_writer.writerow({
            "dataset": dataset, "seed": seed, "modality": modality,
            "target": arrays[1][i], "source": arrays[2][i],
            "semantic_quartile": arrays[3][i] + 1,
            "personalization_quartile": arrays[4][i] + 1,
            "semantic_similarity": arrays[5][i], "personalization_deviation_l1": arrays[6][i],
            "norm_D_target": arrays[7][i], "norm_D_source": arrays[8][i],
            "norm_D_difference": arrays[9][i], "cos_D_target_source": arrays[10][i],
            "cos_H0_target_source": arrays[11][i],
        })
    metrics = {
        "norm_D_target": n_t, "norm_D_source": n_s, "norm_D_difference": n_diff,
        "cos_D_target_source": cos_d, "cos_H0_target_source": cos_h,
    }
    rows = []
    deviation_cpu = dev.detach().cpu().numpy()
    for descriptor, tensor in metrics.items():
        values = tensor.detach().cpu().numpy()
        if values.size > 2:
            corr = spearmanr(deviation_cpu, values, nan_policy="omit")
            rows.append({
                "dataset": dataset, "seed": seed, "modality": modality,
                "analysis": "spearman_deviation_vs_structural_response",
                "descriptor": descriptor, "n_edges": int(values.size),
                "statistic": "spearman_rho", "value": _finite(corr.statistic),
                "p_value": _finite(corr.pvalue), "group": "validation_related",
                "test_evaluated": False,
            })
        by_deviation = [tensor[deviation_bins[val_positions] == q] for q in range(4)]
        usable = [part.detach().cpu().numpy() for part in by_deviation if part.numel()]
        if len(usable) >= 2 and all(len(part) for part in usable):
            stat = kruskal(*usable, nan_policy="omit")
            rows.append({
                "dataset": dataset, "seed": seed, "modality": modality,
                "analysis": "kruskal_personalization_quartiles",
                "descriptor": descriptor, "n_edges": int(values.size),
                "statistic": "H", "value": _finite(stat.statistic),
                "p_value": _finite(stat.pvalue), "group": "D1-D4",
                "test_evaluated": False,
            })
        q1 = semantic_bins == 0
        q1_high = tensor[q1 & (deviation_bins[val_positions] == 3)].detach().cpu().numpy()
        q1_low = tensor[q1 & (deviation_bins[val_positions] < 3)].detach().cpu().numpy()
        if q1_high.size and q1_low.size:
            stat = mannwhitneyu(q1_high, q1_low, alternative="two-sided")
            rows.append({
                "dataset": dataset, "seed": seed, "modality": modality,
                "analysis": "q1_high_vs_lower_personalization",
                "descriptor": descriptor, "n_edges": int(values.size),
                "statistic": "mann_whitney_u", "value": _finite(stat.statistic),
                "p_value": _finite(stat.pvalue), "group": "Q1_D4_vs_D1-D3",
                "test_evaluated": False,
            })
    return rows


def _write_node_utility_row(writer: GzipCsvWriter, *, dataset, seed, modality, group_type,
                            group, intervention_id, repeat, is_target, normal, changed, touched):
    util = _node_deltas(normal, changed)
    ids = normal["val_idx"].detach().cpu().tolist()
    vals = [
        normal["labels"].detach().cpu().tolist(), normal["ce"].detach().cpu().tolist(),
        changed["ce"].detach().cpu().tolist(), util["delta_ce"].detach().cpu().tolist(),
        normal["margin"].detach().cpu().tolist(), changed["margin"].detach().cpu().tolist(),
        util["margin_utility"].detach().cpu().tolist(), normal["predictions"].detach().cpu().tolist(),
        changed["predictions"].detach().cpu().tolist(), util["prediction_flip"].detach().cpu().tolist(),
        touched.detach().cpu().tolist(),
    ]
    for i, node_id in enumerate(ids):
        writer.writerow({
            "dataset": dataset, "seed": seed, "modality": modality, "group_type": group_type,
            "group": group, "intervention_id": intervention_id, "repeat": repeat,
            "is_target": int(is_target), "node_id": int(node_id), "label": int(vals[0][i]),
            "touched_validation_node": int(vals[10][i]), "normal_ce": vals[1][i],
            "intervention_ce": vals[2][i], "delta_ce": vals[3][i],
            "normal_margin": vals[4][i], "intervention_margin": vals[5][i],
            "margin_utility": vals[6][i], "normal_prediction": int(vals[7][i]),
            "intervention_prediction": int(vals[8][i]), "prediction_flip": int(vals[9][i]),
        })


def _target_summary_rows(
    dataset, seed, modality, group_type, group, intervention_id, repeat, is_target,
    selected_edges, candidate_edges, fallback_meta, normal, changed, touched,
    mean_message_abs, mean_message_rel,
) -> list[dict]:
    all_stats = _summary_values(normal, changed)
    touched_stats = _summary_values(normal, changed, touched)
    return [{
        "dataset": dataset, "seed": seed, "modality": modality, "group_type": group_type,
        "group": group, "intervention_id": intervention_id, "repeat": repeat,
        "is_target": int(is_target), "selected_edges": selected_edges,
        "candidate_edges": candidate_edges,
        "candidate_scope": "all_nonself_physical_edges_excluding_target",
        "fallback_rate": fallback_meta.get("fallback_rate", 0.0),
        "fallback_3d_count": fallback_meta.get("fallback_counts", {}).get("0d", 0),
        "fallback_2d_count": fallback_meta.get("fallback_counts", {}).get("1d", 0),
        "fallback_1d_count": fallback_meta.get("fallback_counts", {}).get("2d", 0),
        "fallback_unmatched_strata_count": fallback_meta.get("fallback_counts", {}).get("3d", 0),
        "n_validation_val": all_stats.get("n_nodes", 0),
        "n_touched_val": touched_stats.get("n_nodes", 0),
        "mean_absolute_delta_ce_all_val": all_stats.get("mean_absolute_delta_ce"),
        "mean_absolute_delta_ce_touched_val": touched_stats.get("mean_absolute_delta_ce"),
        "mean_absolute_margin_utility_all_val": all_stats.get("mean_absolute_margin_utility"),
        "mean_absolute_margin_utility_touched_val": touched_stats.get("mean_absolute_margin_utility"),
        "mean_delta_ce_all_val": all_stats.get("mean_delta_ce"),
        "median_delta_ce_all_val": all_stats.get("median_delta_ce"),
        "p10_delta_ce_all_val": all_stats.get("p10_delta_ce"),
        "p90_delta_ce_all_val": all_stats.get("p90_delta_ce"),
        "harmed_fraction_all_val": all_stats.get("harmed_fraction"),
        "improved_fraction_all_val": all_stats.get("improved_fraction"),
        "near_zero_fraction_all_val": all_stats.get("near_zero_fraction"),
        "mean_margin_utility_all_val": all_stats.get("mean_margin_utility"),
        "prediction_flip_rate_all_val": all_stats.get("prediction_flip_rate"),
        "val_acc": changed["acc"], "val_macro_f1": changed["macro_f1"],
        "mean_delta_ce_touched_val": touched_stats.get("mean_delta_ce"),
        "median_delta_ce_touched_val": touched_stats.get("median_delta_ce"),
        "p10_delta_ce_touched_val": touched_stats.get("p10_delta_ce"),
        "p90_delta_ce_touched_val": touched_stats.get("p90_delta_ce"),
        "harmed_fraction_touched_val": touched_stats.get("harmed_fraction"),
        "improved_fraction_touched_val": touched_stats.get("improved_fraction"),
        "near_zero_fraction_touched_val": touched_stats.get("near_zero_fraction"),
        "mean_margin_utility_touched_val": touched_stats.get("mean_margin_utility"),
        "prediction_flip_rate_touched_val": touched_stats.get("prediction_flip_rate"),
        "mean_message_change_abs": mean_message_abs,
        "mean_relative_message_change": mean_message_rel,
        "normal_val_acc": normal["acc"], "normal_val_macro_f1": normal["macro_f1"],
        "delta_val_acc": changed["acc"] - normal["acc"],
        "delta_val_macro_f1": changed["macro_f1"] - normal["macro_f1"],
        "test_evaluated": False,
    }]


def _run_d1(dataset, seed, model, data, head, eval_labels, x, edge_index, writer):
    prepared = _prepare_frozen(model, x, edge_index)
    normal_embedding = _forward_routes(model, prepared, prepared["route_text"], prepared["route_visual"])
    normal = _validation_arrays(head, normal_embedding, data, eval_labels)
    rows = []
    route_text, route_visual = prepared["route_text"], prepared["route_visual"]
    # Store a normal row for each modality scope so descriptive decisions can compare
    # modality-specific interventions against the same frozen checkpoint baseline.
    for scope in ("text_only", "visual_only", "both"):
        _, raw = _metrics_and_raw_rows(
            normal, normal, dataset=dataset, seed=seed, scope=scope, intervention="normal"
        )
        writer.writerows(raw)
        rows.append({
            "dataset": dataset, "seed": seed, "modality_scope": scope,
            "intervention": "normal", "val_acc": normal["acc"],
            "val_macro_f1": normal["macro_f1"], "mean_delta_ce": 0.0,
            "median_delta_ce": 0.0, "p10_delta_ce": 0.0, "p90_delta_ce": 0.0,
            "harmed_fraction": 0.0, "improved_fraction": 0.0,
            "near_zero_fraction": 1.0, "mean_margin_utility": 0.0,
            "median_margin_utility": 0.0, "prediction_flip_rate": 0.0,
            "mean_route_l1_change": 0.0, "test_evaluated": False, "status": "complete",
        })
    for scope, intervention in _d1_scenarios(route_text, route_visual):
        rt, rv = _replace_modality_routes(route_text, route_visual, scope, intervention)
        embedding = _forward_routes(model, prepared, rt, rv)
        changed = _validation_arrays(head, embedding, data, eval_labels)
        summary, raw = _metrics_and_raw_rows(
            normal, changed, dataset=dataset, seed=seed, scope=scope,
            intervention=intervention,
        )
        writer.writerows(raw)
        delta_route = []
        if scope in {"text_only", "both"}:
            delta_route.append(_route_change(route_text, rt))
        if scope in {"visual_only", "both"}:
            delta_route.append(_route_change(route_visual, rv))
        rows.append({
            "dataset": dataset, "seed": seed, "modality_scope": scope,
            "intervention": intervention, "val_acc": changed["acc"],
            "val_macro_f1": changed["macro_f1"],
            "mean_delta_ce": summary["mean_delta_ce"],
            "median_delta_ce": summary["median_delta_ce"],
            "p10_delta_ce": summary["p10_delta_ce"], "p90_delta_ce": summary["p90_delta_ce"],
            "harmed_fraction": summary["harmed_fraction"],
            "improved_fraction": summary["improved_fraction"],
            "near_zero_fraction": summary["near_zero_fraction"],
            "mean_margin_utility": summary["mean_margin_utility"],
            "median_margin_utility": summary["median_margin_utility"],
            "prediction_flip_rate": float((changed["predictions"] != normal["predictions"]).float().mean().item()),
            "mean_route_l1_change": float(np.mean(delta_route)) if delta_route else 0.0,
            "test_evaluated": False, "status": "complete",
        })
        print(f"  D1 {dataset}/seed{seed} {scope}:{intervention} ΔCE={summary['mean_delta_ce']:+.6g}", flush=True)
    return rows


def _run_shrinkage(dataset, seed, model, data, head, eval_labels, prepared, normal, node_writer):
    rows = []
    route_t, route_v = prepared["route_text"], prepared["route_visual"]
    route_map = {"text": route_t, "visual": route_v}
    globals_ = {m: r.mean(dim=0) for m, r in route_map.items()}
    normal_output = {"text_only": normal, "visual_only": normal, "both": normal}
    by_scope = defaultdict(list)
    for scope in ("text_only", "visual_only", "both"):
        for lam in LAMBDAS:
            if lam == 1.0:
                changed = normal_output[scope]
            else:
                rt, rv = route_t, route_v
                if scope in {"text_only", "both"}:
                    rt, _ = _shrink_route(route_t, lam)
                if scope in {"visual_only", "both"}:
                    rv, _ = _shrink_route(route_v, lam)
                embedding = _forward_routes(model, prepared, rt, rv)
                changed = _validation_arrays(head, embedding, data, eval_labels)
            summary = _summary_values(normal, changed)
            by_scope[scope].append((lam, changed, summary))
            rows.append({
                "dataset": dataset, "seed": seed, "modality_scope": scope, "lambda": lam,
                "val_acc": changed["acc"], "val_macro_f1": changed["macro_f1"],
                "mean_delta_ce_vs_lambda1": summary["mean_delta_ce"],
                "median_delta_ce_vs_lambda1": summary["median_delta_ce"],
                "p10_delta_ce_vs_lambda1": summary["p10_delta_ce"],
                "p90_delta_ce_vs_lambda1": summary["p90_delta_ce"],
                "harmed_fraction": summary["harmed_fraction"],
                "improved_fraction": summary["improved_fraction"],
                "near_zero_fraction": summary["near_zero_fraction"],
                "mean_margin_utility": summary["mean_margin_utility"],
                "median_margin_utility": summary["median_margin_utility"],
                "prediction_flip_rate": float((changed["predictions"] != normal["predictions"]).float().mean().item()),
                "test_evaluated": False,
            })
            util = _node_deltas(normal, changed)
            for i, node_id in enumerate(normal["val_idx"].detach().cpu().tolist()):
                node_writer.writerow({
                    "dataset": dataset, "seed": seed, "modality_scope": scope, "lambda": lam,
                    "node_id": node_id,
                    "label": int(normal["labels"][i].item()), "normal_ce": float(normal["ce"][i].item()),
                    "intervention_ce": float(changed["ce"][i].item()),
                    "delta_ce": float(util["delta_ce"][i].item()),
                    "normal_margin": float(normal["margin"][i].item()),
                    "intervention_margin": float(changed["margin"][i].item()),
                    "margin_utility": float(util["margin_utility"][i].item()),
                    "normal_prediction": int(normal["predictions"][i].item()),
                    "intervention_prediction": int(changed["predictions"][i].item()),
                    "prediction_flip": int(util["prediction_flip"][i].item()),
                })
    # Context-level descriptive choices are repeated on each row for convenient filtering.
    for scope, values in by_scope.items():
        best_ce = min(values, key=lambda item: item[2]["mean_delta_ce"])[0]
        best_acc = max(values, key=lambda item: item[1]["acc"])[0]
        for row in rows:
            if row["modality_scope"] == scope:
                row["best_lambda_by_mean_ce"] = best_ce
                row["best_lambda_by_val_accuracy"] = best_acc
    return rows


def _run_targeted_group(
    dataset, seed, modality, group_type, group, target_positions, context,
    model, data, head, eval_labels, prepared, normal, route, global_route,
    node_writer, match_rows, message_change_vectors, random_repeats: int,
):
    if target_positions.numel() == 0:
        return [], []
    # Controls come from all non-self physical edges, allowing exact correction-magnitude
    # matching for D1/D4 targets; only the target edge set itself is excluded. No Test labels
    # or metrics are used for selecting controls.
    candidate_positions = _random_candidate_pool(context["target"].numel(), target_positions)
    if candidate_positions.numel() < target_positions.numel():
        raise RuntimeError(
            f"insufficient non-target validation-related edges for matched controls: "
            f"{dataset}/seed{seed}/{modality}/{group}"
        )
    group_rows, control_rows = [], []
    selections = [("target", target_positions, None, True)]
    for repeat in range(random_repeats):
        match_seed = (seed * 1000003 + (0 if modality == "text" else 1) * 10007
                      + (0 if group_type == "semantic_similarity" else 1009)
                      + int(group[-1]) * 101 + repeat)
        rng = np.random.default_rng(match_seed)
        chosen, meta = _sample_stratified_random(
            target_positions, candidate_positions, context["strata"], rng
        )
        if not meta["ok"]:
            raise RuntimeError(f"matched random sample failed for {dataset}/seed{seed}/{modality}/{group}")
        quality = _matching_quality(
            target_positions, chosen, context["strata"], context["src_degree"],
            context["dst_degree"], context["deviation"], meta,
        )
        match_rows.append({
            "dataset": dataset, "seed": seed, "modality": modality, "group_type": group_type,
            "group": group, "repeat": repeat, **quality, "test_evaluated": False,
        })
        selections.append((f"matched_random_{repeat:02d}", chosen, meta, False))

    for intervention_id, selected_positions, meta, is_target in selections:
        changed_route = _route_to_global(route, global_route, selected_positions)
        rt, rv = prepared["route_text"], prepared["route_visual"]
        if modality == "text":
            rt = changed_route
        else:
            rv = changed_route
        embedding = _forward_routes(model, prepared, rt, rv)
        changed = _validation_arrays(head, embedding, data, eval_labels)
        touched = _compute_touched_mask(
            data, context["target"], context["source"], selected_positions, embedding.device
        )
        msg_abs_vector, msg_rel_vector = message_change_vectors[modality]
        msg_abs = float(msg_abs_vector[selected_positions].mean().item())
        msg_rel = float(msg_rel_vector[selected_positions].mean().item())
        group_rows.extend(_target_summary_rows(
            dataset, seed, modality, group_type, group, intervention_id,
            -1 if is_target else int(intervention_id.removeprefix("matched_random_")),
            is_target, int(selected_positions.numel()), int(meta["candidate_edges"]) if meta else int(candidate_positions.numel()),
            meta or {"fallback_rate": 0.0, "fallback_counts": {"0d": target_positions.numel()}},
            normal, changed, touched, msg_abs, msg_rel,
        ))
        _write_node_utility_row(
            node_writer, dataset=dataset, seed=seed, modality=modality, group_type=group_type,
            group=group, intervention_id=intervention_id,
            repeat=-1 if is_target else int(intervention_id.removeprefix("matched_random_")),
            is_target=is_target, normal=normal, changed=changed, touched=touched,
        )
        population_stats = _summary_values(normal, changed)
        group_rows[-1]["target_mean_delta_ce_all_val"] = population_stats.get("mean_delta_ce") if is_target else None
        if is_target:
            print(
                f"  D2-target {dataset}/seed{seed}/{modality}/{group} "
                f"n_edges={selected_positions.numel()} ΔCE={population_stats.get('mean_delta_ce', float('nan')):+.6g}",
                flush=True,
            )
    return group_rows, control_rows


def _d2_random_summaries(dataset, seed, modality, group_type, group, rows):
    out = []
    random_rows = [row for row in rows if not row["is_target"]]
    target = next((row for row in rows if row["is_target"]), None)
    for population, utility_key, message_key in (
        ("all_validation", "mean_delta_ce_all_val", "mean_message_change_abs"),
        ("touched_validation", "mean_delta_ce_touched_val", "mean_message_change_abs"),
    ):
        if target is None:
            continue
        valid_randoms = [row for row in random_rows if row.get(utility_key) is not None]
        random_utilities = np.asarray([float(row[utility_key]) for row in valid_randoms], dtype=float)
        target_value = target.get(utility_key)
        if target_value is not None:
            target_utility = float(target_value)
        else:
            target_utility = None
        if len(random_utilities) and target_utility is not None:
            less = int(np.sum(random_utilities < target_utility))
            equal = int(np.sum(np.isclose(random_utilities, target_utility, atol=1e-12)))
            percentile = (less + 0.5 * equal) / len(random_utilities)
            two_sided = min(1.0, 2.0 * (1 + min(less, len(random_utilities) - less)) / (len(random_utilities) + 1))
            random_message = np.asarray([float(row[message_key]) for row in valid_randoms], dtype=float)
            mean_random_message = float(random_message.mean())
            mean_target_message = float(target[message_key])
            random_mean = float(random_utilities.mean())
            random_std = float(random_utilities.std(ddof=0))
            excess = target_utility - random_mean
            per_target = excess / (mean_target_message + 1e-12)
            per_random = excess / (mean_random_message + 1e-12)
        else:
            percentile = two_sided = mean_random_message = random_mean = random_std = excess = None
            mean_target_message = float(target[message_key])
            per_target = per_random = None
        out.append({
            "dataset": dataset, "seed": seed, "modality": modality, "group_type": group_type,
            "group": group, "population": population, "n_repeats": len(valid_randoms),
            "requested_repeats": len(random_rows),
            "n_empty_touched_controls": len(random_rows) - len(valid_randoms) if population == "touched_validation" else 0,
            "target_mean_utility": target_utility, "random_mean_utility": random_mean,
            "random_std_utility": random_std, "excess_utility": excess,
            "target_percentile_in_random": percentile,
            "two_sided_empirical_extremeness": two_sided,
            "target_mean_message_change_abs": mean_target_message,
            "random_mean_message_change_abs": mean_random_message,
            "excess_utility_per_target_message_change": per_target,
            "excess_utility_per_random_message_change": per_random,
            "test_evaluated": False,
        })
    return out


def _aggregate_shrinkage(rows: list[dict]) -> list[dict]:
    aggregates = []
    keys = sorted({(row["modality_scope"], float(row["lambda"])) for row in rows})
    for scope, lam in keys:
        subset = [row for row in rows if row["modality_scope"] == scope and float(row["lambda"]) == lam]
        aggregates.append({
            "dataset": "ALL", "seed": "ALL", "modality_scope": scope, "lambda": lam,
            **{key: float(np.mean([row[key] for row in subset])) for key in (
                "val_acc", "val_macro_f1", "mean_delta_ce_vs_lambda1", "median_delta_ce_vs_lambda1",
                "harmed_fraction", "improved_fraction", "near_zero_fraction", "mean_margin_utility",
                "median_margin_utility", "prediction_flip_rate",
            )},
            "p10_delta_ce_vs_lambda1": float(np.mean([row["p10_delta_ce_vs_lambda1"] for row in subset])),
            "p90_delta_ce_vs_lambda1": float(np.mean([row["p90_delta_ce_vs_lambda1"] for row in subset])),
            "best_lambda_by_mean_ce": "",
            "best_lambda_by_val_accuracy": "", "test_evaluated": False,
        })
    for scope in sorted({row["modality_scope"] for row in rows}):
        subset = [row for row in aggregates if row["modality_scope"] == scope]
        best_ce = min(subset, key=lambda r: float(r["mean_delta_ce_vs_lambda1"]))["lambda"]
        best_acc = max(subset, key=lambda r: float(r["val_acc"]))["lambda"]
        for row in subset:
            row["best_lambda_by_mean_ce"] = best_ce
            row["best_lambda_by_val_accuracy"] = best_acc
    return aggregates


def _d1_decision(rows: list[dict]) -> list[dict]:
    by_context = defaultdict(dict)
    for row in rows:
        by_context[(row["dataset"], int(row["seed"]), row["modality_scope"])][row["intervention"]] = row
    zero_rows = [row for row in rows if row["intervention"] == "zero"]
    zero_positive = sum(row["mean_delta_ce"] > 0 and row["mean_margin_utility"] > 0 for row in zero_rows)
    # Diagnostic thresholds are declared descriptive and are not statistical tests.
    near_tolerance = 0.01  # nats/node; CE differences below this are treated as close for this summary only.
    metric_tolerance = 0.01  # absolute Val Accuracy/F1.
    modality_contexts = []
    for (dataset, seed, scope), values in by_context.items():
        if scope not in {"text_only", "visual_only"}:
            continue
        singles = [values.get(f"single_e{i}") for i in range(4)]
        singles = [row for row in singles if row is not None]
        if not singles:
            continue
        best = min(singles, key=lambda row: float(row["mean_delta_ce"]))
        top1 = values.get("top1")
        uniform = values.get("uniform")
        # The Normal metric is recovered using the `normal` companion row.
        normal = values.get("normal")
        if normal:
            normal_acc = float(normal["val_acc"])
            normal_f1 = float(normal["val_macro_f1"])
        else:
            normal_acc, normal_f1 = float("nan"), float("nan")
        sufficient = (
            float(best["mean_delta_ce"]) <= near_tolerance
            and float(best["val_acc"]) >= normal_acc - metric_tolerance
            and float(best["val_macro_f1"]) >= normal_f1 - metric_tolerance
            and top1 is not None and uniform is not None
            and float(top1["mean_delta_ce"]) <= near_tolerance
            and float(uniform["mean_delta_ce"]) <= near_tolerance
        )
        modality_contexts.append({
            "dataset": dataset, "seed": seed, "modality": scope,
            "best_single": best["intervention"], "best_single_mean_delta_ce": best["mean_delta_ce"],
            "best_single_delta_accuracy": float(best["val_acc"]) - normal_acc,
            "best_single_delta_macro_f1": float(best["val_macro_f1"]) - normal_f1,
            "top1_mean_delta_ce": None if top1 is None else top1["mean_delta_ce"],
            "uniform_mean_delta_ce": None if uniform is None else uniform["mean_delta_ce"],
            "single_sufficient_context": bool(sufficient),
        })
    sufficient_count = sum(row["single_sufficient_context"] for row in modality_contexts)
    all_count = len(modality_contexts)
    zero_decision = "GLOBAL_TRANSFORMATION_FUNCTIONAL" if zero_rows and zero_positive > len(zero_rows) / 2 else "GLOBAL_TRANSFORMATION_NOT_CONSISTENTLY_NECESSARY"
    mixture_decision = (
        "SINGLE_SHARED_OPERATOR_SUFFICIENT"
        if all_count and sufficient_count > all_count / 2
        else "MULTI_GLOBAL_OPERATOR_MIXTURE_SUPPORTED"
    )
    return [
        {
            "question": "D1_A_global_transformation_necessity", "decision": zero_decision,
            "n_contexts": len(zero_rows), "contexts_mean_delta_ce_and_margin_positive": zero_positive,
            "descriptive_rule": "ZERO has mean ΔCE>0 and mean margin utility>0 in a majority of modality scopes; ΔCE=CE(intervention)-CE(normal).",
            "test_evaluated": False,
        },
        {
            "question": "D1_B_multiple_global_operators", "decision": mixture_decision,
            "n_modality_contexts": all_count, "single_sufficient_contexts": sufficient_count,
            "ce_close_tolerance_nats_per_node": near_tolerance,
            "accuracy_macro_f1_tolerance_absolute": metric_tolerance,
            "descriptive_rule": "A modality-context passes when diagnostic best single, TOP1, and UNIFORM each have mean ΔCE≤0.01 and best-single Validation Accuracy/Macro-F1 are within 0.01 of NORMAL. This is a descriptive discovery rule; expert IDs are modality-specific.",
            "test_evaluated": False,
        },
    ]


def _d2_decision(shrink_rows, random_rows, structural_rows, targeted_rows):
    contexts = {(row["dataset"], row["seed"], row["modality_scope"]): row for row in shrink_rows if row["dataset"] != "ALL"}
    intermediate_better = 0
    comparison_count = 0
    for key in {(r["dataset"], r["seed"], r["modality_scope"]) for r in shrink_rows if r["dataset"] != "ALL"}:
        local = [r for r in shrink_rows if (r["dataset"], r["seed"], r["modality_scope"]) == key]
        normal = next((r for r in local if float(r["lambda"]) == 1.0), None)
        inter = [r for r in local if 0.0 < float(r["lambda"]) < 1.0]
        if normal and inter:
            comparison_count += 1
            intermediate_better += int(min(inter, key=lambda r: r["mean_delta_ce_vs_lambda1"])["mean_delta_ce_vs_lambda1"] < -EPS)
    shrink_decision = "FULL_PERSONALIZATION_APPEARS_OVERSTRONG" if comparison_count and intermediate_better > comparison_count / 2 else "NO_MAJORITY_INTERMEDIATE_SHRINKAGE_ADVANTAGE"
    q1_rows = [r for r in random_rows if r["group_type"] == "semantic_similarity" and r["group"] == "Q1" and r["population"] == "all_validation"]
    d4_rows = [r for r in random_rows if r["group_type"] == "personalization_deviation" and r["group"] == "D4" and r["population"] == "all_validation"]
    q1_positive = sum(float(r["excess_utility"]) > 0 for r in q1_rows)
    d4_positive = sum(float(r["excess_utility"]) > 0 for r in d4_rows)
    target_only = [r for r in targeted_rows if int(r.get("is_target") or 0) == 1]
    d4_targets = [r for r in target_only if r["group_type"] == "personalization_deviation" and r["group"] == "D4"]
    d4_removal_better_ce = sum(float(r["mean_delta_ce_all_val"]) < 0 for r in d4_targets)
    d4_removal_better_acc = sum(float(r.get("delta_val_acc", 0.0)) > 0 for r in d4_targets)
    concentrated = sum(
        float(row.get("mean_absolute_delta_ce_touched_val") or 0.0)
        > float(row.get("mean_absolute_delta_ce_all_val") or 0.0) + EPS
        for row in target_only
    )
    mean_abs_all = float(np.mean([float(row["mean_absolute_delta_ce_all_val"]) for row in target_only])) if target_only else 0.0
    touched_abs_rows = [float(row["mean_absolute_delta_ce_touched_val"]) for row in target_only
                        if row.get("mean_absolute_delta_ce_touched_val") not in (None, "")]
    mean_abs_touched = float(np.mean(touched_abs_rows)) if touched_abs_rows else 0.0
    mean_touched_fraction = float(np.mean([
        int(row.get("n_touched_val") or 0) / max(int(row.get("n_validation_val") or 0), 1)
        for row in target_only
    ])) if target_only else 0.0
    concentration_decision = (
        "UTILITY_STRONGER_ON_DIRECTLY_TOUCHED_NODES"
        if target_only and concentrated > len(target_only) / 2
        else "NO_MAJORITY_TOUCHED_NODE_CONCENTRATION"
    )
    descriptors = [r for r in structural_rows if r["analysis"] == "spearman_deviation_vs_structural_response"]
    corr_positive = sum(float(r["value"]) > 0 for r in descriptors if math.isfinite(float(r["value"])))
    return [
        {
            "question": "D2_Q1_full_personalization_strength", "decision": shrink_decision,
            "contexts_with_intermediate_lambda_better_by_mean_ce": intermediate_better,
            "contexts_compared": comparison_count,
            "description": "Counts modality-scope contexts where at least one intermediate λ has lower mean Validation CE than λ=1.",
            "test_evaluated": False,
        },
        {
            "question": "D2_Q2_strong_deviation_utility", "decision": "DESCRIPTIVE_TARGET_VS_MATCHED_RANDOM",
            "D4_target_excess_utility_positive_contexts": d4_positive, "D4_contexts": len(d4_rows),
            "D4_removal_improves_CE_contexts": d4_removal_better_ce,
            "D4_removal_improves_accuracy_contexts": d4_removal_better_acc,
            "description": "Positive CE excess means removing personalization from D4 is more harmful than removing it from matched random edges; report matched-control percentile alongside this count.",
            "test_evaluated": False,
        },
        {
            "question": "D2_Q3_semantic_Q1_specialized_utility", "decision": "DESCRIPTIVE_TARGET_VS_MATCHED_RANDOM",
            "Q1_target_excess_utility_positive_contexts": q1_positive, "Q1_contexts": len(q1_rows),
            "description": "Q1 route-to-global utility is compared against 20 degree- and deviation-matched control edge sets sampled from the full physical edge population, excluding the target group.",
            "test_evaluated": False,
        },
        {
            "question": "D2_Q4_node_utility_concentration", "decision": concentration_decision,
            "target_interventions_touched_abs_ce_exceeds_all_abs_ce": concentrated,
            "target_interventions": len(target_only), "mean_touched_validation_node_fraction": mean_touched_fraction,
            "mean_absolute_delta_ce_all_validation": mean_abs_all,
            "mean_absolute_delta_ce_touched_validation": mean_abs_touched,
            "touched_to_all_absolute_delta_ce_ratio": mean_abs_touched / mean_abs_all if mean_abs_all else None,
            "description": "Compares mean absolute node ΔCE for directly touched Validation nodes with all Validation nodes; this is a descriptive concentration check.",
            "test_evaluated": False,
        },
        {
            "question": "D2_Q5_structural_response_association", "decision": "DESCRIPTIVE_ONLY",
            "spearman_rows": len(descriptors), "positive_spearman_rows": corr_positive,
            "description": "Edge-level Spearman correlations and exploratory Kruskal–Wallis/Mann–Whitney summaries are descriptive; edges are dependent and these are not causal tests.",
            "test_evaluated": False,
        },
    ]


def _load_context(variant, dataset, seed, device):
    cfg, data, model, head, payload, eval_labels, x, edge_index = base._load_checkpoint_context(
        variant, dataset, seed, device
    )
    config_path = base.run_directory(variant, dataset, seed) / "resolved_config.json"
    config = base.read_json(config_path)
    _assert_frozen_payload(config, payload, base.checkpoint_path(variant, dataset, seed))
    if any("test" in str(key).lower() for key in payload.get("metrics", {})):
        raise RuntimeError(f"test field in frozen checkpoint metrics: {base.checkpoint_path(variant, dataset, seed)}")
    return cfg, data, model, head, eval_labels, x, edge_index


def _run_one_a3_context(dataset, seed, device, random_repeats, shrink_node_writer,
                        target_node_writer, match_rows, structural_raw_writer):
    _, data, model, head, eval_labels, x, edge_index = _load_context("relation_expert", dataset, seed, device)
    prepared = _prepare_frozen(model, x, edge_index)
    normal_embedding = _forward_routes(model, prepared, prepared["route_text"], prepared["route_visual"])
    normal = _validation_arrays(head, normal_embedding, data, eval_labels)
    # Frozen A0 H0 defines the semantic similarity groups.
    _, a0_data, a0_model, _, a0_labels, a0_x, a0_edge_index = _load_context("plain", dataset, seed, device)
    a0_prepared = _prepare_frozen(a0_model, a0_x, a0_edge_index)
    if not torch.equal(prepared["target"], a0_prepared["target"]) or not torch.equal(prepared["source"], a0_prepared["source"]):
        raise RuntimeError(f"A0/A3 canonical edge ordering differs for {dataset}/seed{seed}")
    semantics = {}
    contexts = {}
    structural_rows = []
    message_change_vectors = {
        modality: _edge_message_change_vectors(model, prepared, prepared[f"route_{modality}"],
                                               prepared[f"route_{modality}"].mean(dim=0), modality)
        for modality in ("text", "visual")
    }
    for modality in ("text", "visual"):
        semantic = _make_semantic_bins(
            a0_model, a0_prepared[f"h0_{modality}"], a0_data,
            prepared["target"], prepared["source"], modality,
        )
        route = prepared[f"route_{modality}"]
        context = _d2_edge_context(prepared, data, route, modality, semantic)
        semantics[modality] = semantic
        contexts[modality] = context
        structural_rows.extend(_structural_rows(
            dataset, seed, modality, a0_prepared[f"h0_{modality}"], prepared["target"],
            prepared["source"], prepared["operator"], data, context["val_positions"],
            semantic["bins"], semantic["similarity"][semantic["positions"]],
            context["deviation"], context["dev_bins"], structural_raw_writer,
        ))
    shrink_rows = _run_shrinkage(
        dataset, seed, model, data, head, eval_labels, prepared, normal, shrink_node_writer
    )
    target_rows = []
    for modality in ("text", "visual"):
        context = contexts[modality]
        route = prepared[f"route_{modality}"]
        global_route = context["global_route"]
        for group_type, group, positions in _group_definitions(context):
            rows, _ = _run_targeted_group(
                dataset, seed, modality, group_type, group, positions, context,
                model, data, head, eval_labels, prepared, normal, route, global_route,
                target_node_writer, match_rows, message_change_vectors, random_repeats,
            )
            target_rows.extend(rows)
    del a0_model, a0_data, a0_x, a0_edge_index, a0_prepared, model, head, data, prepared, x, edge_index
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
    return shrink_rows, target_rows, structural_rows


def _write_report(d1_rows, d1_decisions, shrink_rows, target_rows, random_rows, structural_rows,
                  match_rows, d2_decisions, random_repeats, runtime_seconds, peak_gpu_mib):
    zero = [row for row in d1_rows if row["intervention"] == "zero"]
    shrink_contexts = [row for row in shrink_rows if row["dataset"] != "ALL"]
    inter_better = sum(
        min(float(r["mean_delta_ce_vs_lambda1"]) for r in shrink_contexts
            if r["dataset"] == key[0] and r["seed"] == key[1] and r["modality_scope"] == key[2]
            and 0.0 < float(r["lambda"]) < 1.0) < -EPS
        for key in {(r["dataset"], r["seed"], r["modality_scope"]) for r in shrink_contexts}
    )
    d4 = [r for r in random_rows if r["group_type"] == "personalization_deviation" and r["group"] == "D4" and r["population"] == "all_validation"]
    q1 = [r for r in random_rows if r["group_type"] == "semantic_similarity" and r["group"] == "Q1" and r["population"] == "all_validation"]
    target_only = [r for r in target_rows if int(r.get("is_target") or 0) == 1]
    mean_abs_all = float(np.mean([float(r["mean_absolute_delta_ce_all_val"]) for r in target_only])) if target_only else 0.0
    touched_rows = [r for r in target_only if r.get("mean_absolute_delta_ce_touched_val") not in (None, "")]
    mean_abs_touched = float(np.mean([float(r["mean_absolute_delta_ce_touched_val"]) for r in touched_rows])) if touched_rows else 0.0
    touched_fraction = float(np.mean([
        int(r.get("n_touched_val") or 0) / max(int(r.get("n_validation_val") or 0), 1)
        for r in target_only
    ])) if target_only else 0.0
    harmful_all = float(np.mean([float(r["harmed_fraction_all_val"]) for r in target_only])) if target_only else 0.0
    harmful_touched = float(np.mean([float(r["harmed_fraction_touched_val"]) for r in touched_rows])) if touched_rows else 0.0
    improved_all = float(np.mean([float(r["improved_fraction_all_val"]) for r in target_only])) if target_only else 0.0
    improved_touched = float(np.mean([float(r["improved_fraction_touched_val"]) for r in touched_rows])) if touched_rows else 0.0
    matching = _matching_quality_summary(match_rows)
    lines = [
        "# P0-D — Global Prior & Relation Personalization Diagnostics", "",
        "## Scope and data integrity", "",
        "This report analyzes only the 15 frozen A2 `global_expert` and 15 frozen A3 `relation_expert` checkpoints. It does not train or update models. Checkpoints are loaded with `strict=True` by the existing checkpoint loader. All reported task metrics index only Validation labels; fixed Macro-F1 labels come from Train and Validation. Train–Train edges define diagnostic quantile thresholds. No Test metrics or Test labels are evaluated.",
        "A0 semantic quartiles use each frozen plain checkpoint's projected H0 and the existing canonical non-self physical edge ordering. A3 shrinkage λ=0 uses the global-mean route of that A3 checkpoint and must not be interpreted as a separately trained A2 result.",
        f"Matched random controls use {random_repeats} no-replacement samples per target context from all non-self physical edges excluding the target set. Sampling first matches the joint quartile of log source degree, log target degree, and personalization magnitude; it then relaxes to two dimensions, one dimension, and finally unrestricted matching when a stratum lacks enough candidates. Every relaxation level and distribution balance is recorded. Across {matching.get('n_repeats', 0)} matched repeats, {matching.get('n_relaxed_repeats', 0)} used relaxed strata; {matching.get('relaxed_edges', 0)}/{matching.get('selected_edges', 0)} selected control edges ({100 * matching.get('relaxed_edge_fraction', 0.0):.4f}%) were relaxed. Mean absolute SMDs were source degree {matching.get('mean_abs_source_degree_smd', 0.0):.4f}, target degree {matching.get('mean_abs_target_degree_smd', 0.0):.4f}, and personalization magnitude {matching.get('mean_abs_personalization_magnitude_smd', 0.0):.4f}. Empirical percentiles are descriptive, not formal significance tests.",
        "Structural response is `D_i = P_nbr H0_i - H0_i`, where `P_nbr` is the frozen A0 symmetric-normalized physical operator with its self-loop coordinates removed. Structural correlations and rank tests are descriptive; edge dependence prevents causal interpretation.",
        "",
        "## D1 — Global operator necessity", "",
        f"ZERO caused positive mean ΔCE and positive mean margin utility in {sum(r['mean_delta_ce'] > 0 and r['mean_margin_utility'] > 0 for r in zero)}/{len(zero)} modality-scope contexts. Decision: **{d1_decisions[0]['decision']}**.",
        f"Multiple-operator decision: **{d1_decisions[1]['decision']}** ({d1_decisions[1].get('single_sufficient_contexts', 0)}/{d1_decisions[1].get('n_modality_contexts', 0)} modality-contexts pass the declared close-to-normal rule). This is a descriptive discovery decision. See `d1_global_operator_interventions.csv` and node-level logits/utility in `d1_node_utility.csv.gz`.",
        "Expert IDs are reported within Text or Visual modality and are not interpreted as cross-modality semantic matches. `best single` is a diagnostic oracle selected on Validation CE, not a deployable selection policy.",
        "",
        "## D2 — Relation-specific personalization", "",
        f"At least one intermediate shrinkage λ had lower Validation CE than λ=1 in {inter_better}/{len({(r['dataset'],r['seed'],r['modality_scope']) for r in shrink_contexts})} dataset × seed × modality-scope contexts. Decision: **{d2_decisions[0]['decision']}**.",
        f"D4 route-to-global removal improves mean CE in {sum(float(r['mean_delta_ce_all_val']) < 0 for r in target_rows if r['is_target'] and r['group_type'] == 'personalization_deviation' and r['group'] == 'D4')}/{sum(bool(r['is_target']) for r in target_rows if r['group_type'] == 'personalization_deviation' and r['group'] == 'D4')} target contexts and improves Validation Accuracy in {sum(float(r.get('delta_val_acc', 0.0)) > 0 for r in target_rows if r['is_target'] and r['group_type'] == 'personalization_deviation' and r['group'] == 'D4')} contexts. D4 target-vs-matched-random CE excess is positive in {sum(float(r['excess_utility']) > 0 for r in d4)}/{len(d4)} population summaries. Q1 target CE excess is positive in {sum(float(r['excess_utility']) > 0 for r in q1)}/{len(q1)} population summaries. Read target percentile, random spread, and message-normalized effects in `d2_random_control_summary.csv`; these values are descriptive rankings rather than significance tests.",
        f"Node utility is concentrated on directly touched nodes: across {len(target_only)} target interventions, mean absolute ΔCE was {mean_abs_touched:.4f} for touched Validation nodes and {mean_abs_all:.4f} across all Validation nodes; touched-node mean absolute utility was higher in {d2_decisions[3].get('target_interventions_touched_abs_ce_exceeds_all_abs_ce', 0)}/{len(target_only)} cases. Selected edges touched {touched_fraction:.1%} of Validation nodes on average. Harmed fractions (ΔCE>0) were {harmful_touched:.1%} touched vs {harmful_all:.1%} all, while improved fractions (ΔCE<0) were {improved_touched:.1%} vs {improved_all:.1%}. See `d2_node_utility_concentration.csv` and the per-node raw file. When a matched control has no directly touched Validation nodes, the touched-control summary reports zero valid repeats and the empty count explicitly; all-node controls still retain all requested repeats.",
        f"Structural-response edge rows: {sum(r['analysis'] == 'spearman_deviation_vs_structural_response' for r in structural_rows)} Spearman summaries; group and Q1 comparisons are in `d2_structural_context_descriptives.csv`. Decision: **{d2_decisions[-1]['decision']}**. These results support descriptive assessment of structural context only.",
        "",
        "## Explicit decisions", "",
    ]
    for row in d1_decisions + d2_decisions:
        lines.append(f"- **{row['question']}** — `{row['decision']}`. {row.get('description', row.get('descriptive_rule', ''))}")
    lines += [
        "", "## Runtime", "",
        f"Elapsed analysis time: {runtime_seconds:.1f} s. Peak allocated GPU memory recorded by PyTorch: {peak_gpu_mib:.1f} MiB.",
        "", "## Boundaries", "",
        "No new training was performed. Test metrics and Test labels were not evaluated. P0-R, Stage II, architecture changes, and training computation changes were not implemented.",
    ]
    (OUT / "relation_operator_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen P0-D global-operator and relation-personalization diagnostics.")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--random-repeats", type=int, default=RANDOM_REPEATS)
    parser.add_argument("--datasets", nargs="*", default=base.DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=base.SEEDS)
    parser.add_argument("--refresh-existing-results", action="store_true",
                        help="Rebuild summaries/report from completed artifacts without loading checkpoints or using a GPU.")
    args = parser.parse_args()
    if args.refresh_existing_results:
        _refresh_existing_results()
        print(f"Refreshed summaries from existing diagnostics: {OUT}", flush=True)
        return
    if args.random_repeats < 20:
        raise ValueError("the registered diagnostic requires at least 20 matched random repeats")
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    unknown = set(args.datasets) - set(base.DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    OUT.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    import time
    start_time = time.time()

    d1_writer = GzipCsvWriter(OUT / "d1_node_utility.csv.gz", D1_NODE_FIELDS)
    d2_shrink_node_writer = GzipCsvWriter(OUT / "d2_shrinkage_node_utility.csv.gz", D2_SHRINK_NODE_FIELDS)
    d2_target_node_writer = GzipCsvWriter(OUT / "d2_targeted_node_utility.csv.gz", D2_TARGET_NODE_FIELDS)
    structural_raw_writer = GzipCsvWriter(OUT / "d2_structural_context_edge_rows.csv.gz", STRUCT_EDGE_FIELDS)
    d1_rows, shrink_rows, targeted_rows, structural_rows, match_rows, d2_random_rows = [], [], [], [], [], []
    try:
        for dataset in args.datasets:
            for seed in args.seeds:
                print(f"[P0-D] starting {dataset}/seed{seed}", flush=True)
                _, data, model, head, eval_labels, x, edge_index = _load_context(
                    "global_expert", dataset, seed, device
                )
                d1_rows.extend(_run_d1(dataset, seed, model, data, head, eval_labels, x, edge_index, d1_writer))
                del model, head, data, x, edge_index
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.empty_cache()
                shr, targ, struct = _run_one_a3_context(
                    dataset, seed, device, args.random_repeats, d2_shrink_node_writer,
                    d2_target_node_writer, match_rows, structural_raw_writer,
                )
                shrink_rows.extend(shr)
                targeted_rows.extend(targ)
                structural_rows.extend(struct)
                random_rows = []
                for modality in ("text", "visual"):
                    for group_type in ("semantic_similarity", "personalization_deviation"):
                        for group_id in range(1, 5):
                            group = f"Q{group_id}" if group_type == "semantic_similarity" else f"D{group_id}"
                            rows = [r for r in targ if r["modality"] == modality and r["group_type"] == group_type and r["group"] == group]
                            random_rows.extend(_d2_random_summaries(dataset, seed, modality, group_type, group, rows))
                d2_random_rows.extend(random_rows)
                _write_csv(OUT / "d2_random_control_summary.csv", d2_random_rows, D2_RANDOM_FIELDS)
                print(f"[P0-D] completed {dataset}/seed{seed}", flush=True)
    finally:
        d1_writer.close()
        d2_shrink_node_writer.close()
        d2_target_node_writer.close()
        structural_raw_writer.close()

    shrink_rows.extend(_aggregate_shrinkage(shrink_rows))
    d1_decisions = _d1_decision(d1_rows)
    d2_decisions = _d2_decision(shrink_rows, d2_random_rows, structural_rows, targeted_rows)
    concentration_rows = _target_concentration_rows(targeted_rows)
    _write_csv(OUT / "d1_global_operator_interventions.csv", d1_rows, D1_INTERVENTION_FIELDS)
    _write_csv(OUT / "d1_node_utility_summary.csv", [
        {**{key: row.get(key) for key in ("dataset", "seed", "modality_scope", "intervention")},
         "modality": "", "population": "all_validation", **{k: row.get(k) for k in (
             "n_nodes", "mean_delta_ce", "median_delta_ce", "p10_delta_ce", "p90_delta_ce",
             "harmed_fraction", "improved_fraction", "near_zero_fraction", "mean_margin_utility",
             "median_margin_utility", "prediction_flip_rate", "val_acc", "val_macro_f1",
         )}}
        for row in d1_rows
    ], NODE_SUMMARY_FIELDS)
    _write_csv(OUT / "d1_decision.csv", d1_decisions)
    _write_csv(OUT / "d2_shrinkage_sweep.csv", shrink_rows, D2_SHRINK_FIELDS)
    _write_csv(OUT / "d2_targeted_personalization.csv", targeted_rows, D2_TARGET_FIELDS)
    _write_csv(OUT / "d2_node_utility_concentration.csv", concentration_rows, D2_CONCENTRATION_FIELDS)
    _write_csv(OUT / "d2_random_control_summary.csv", d2_random_rows, D2_RANDOM_FIELDS)
    _write_csv(OUT / "d2_matching_quality.csv", match_rows, MATCH_FIELDS)
    _write_csv(OUT / "d2_structural_context_descriptives.csv", structural_rows, STRUCT_SUMMARY_FIELDS)
    _write_csv(OUT / "d2_decision.csv", d2_decisions)
    elapsed = time.time() - start_time
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    _write_report(d1_rows, d1_decisions, shrink_rows, targeted_rows, d2_random_rows,
                  structural_rows, match_rows, d2_decisions, args.random_repeats, elapsed, peak_mib)
    print(f"P0-D diagnostics complete: {OUT}", flush=True)
    print(f"Elapsed {elapsed:.1f}s; peak allocated GPU memory {peak_mib:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
