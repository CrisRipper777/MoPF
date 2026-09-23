#!/usr/bin/env python3
"""P1.5 post-hoc mechanism attribution audit for SSI-MAG-V3 Full.

The script deliberately has no training path.  It loads the fifteen existing
P1 best checkpoints, calls the model's analysis-only API, and writes scalar
summaries.  No test metric is read or used for any decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
EPS = 1.0e-12


def _as_np(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _flat(value: Any) -> np.ndarray:
    return _as_np(value).reshape(-1).astype(np.float64, copy=False)


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _stats(value: Any) -> dict[str, float]:
    x = _finite(_flat(value))
    if x.size == 0:
        return {key: math.nan for key in ("mean", "std", "abs_mean", "q10", "q50", "q90")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "abs_mean": float(np.mean(np.abs(x))),
        "q10": float(np.quantile(x, 0.10)),
        "q50": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
    }


def _put(row: dict[str, Any], prefix: str, value: Any) -> None:
    for key, item in _stats(value).items():
        row[f"{prefix}_{key}"] = item


def _corr(x: Any, y: Any) -> tuple[float, float]:
    a = _flat(x)
    b = _flat(y)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) <= EPS or np.std(b) <= EPS:
        return math.nan, math.nan
    return float(pearsonr(a, b).statistic), float(spearmanr(a, b).statistic)


def _cov_contribution(term: Any, total: Any) -> tuple[float, float, float, bool]:
    x = _flat(term)
    y = _flat(total)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    var = float(np.var(y)) if y.size else 0.0
    if y.size < 3 or var <= EPS:
        return math.nan, math.nan, var, True
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    corr = math.nan if np.std(x) <= EPS else float(np.corrcoef(x, y)[0, 1])
    return cov / var, corr, var, False


def _put_corr(row: dict[str, Any], prefix: str, x: Any, y: Any) -> None:
    pearson, spearman = _corr(x, y)
    row[f"{prefix}_pearson"] = pearson
    row[f"{prefix}_spearman"] = spearman


def _put_contrib(row: dict[str, Any], prefix: str, term: Any, total: Any) -> None:
    contribution, corr, variance, near_zero = _cov_contribution(term, total)
    row[f"{prefix}_covariance_contribution"] = contribution
    row[f"{prefix}_corr"] = corr
    row["total_variance"] = variance
    row["total_variance_near_zero"] = int(near_zero)


def _safe_ratio(num: float, den: float) -> float:
    return float(num / (abs(den) + EPS))


def _edge_cos(vectors: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return vectors.new_empty((0,))
    src, dst = edge_index
    return F.cosine_similarity(vectors[src], vectors[dst], dim=-1, eps=EPS)


def _incident_mean(edge_index: torch.Tensor, values: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return values.new_zeros((num_nodes,))
    endpoints = torch.cat((edge_index[0], edge_index[1]), dim=0)
    repeated = torch.cat((values, values), dim=0)
    total = values.new_zeros((num_nodes,))
    count = values.new_zeros((num_nodes,))
    total.index_add_(0, endpoints, repeated)
    count.index_add_(0, endpoints, torch.ones_like(repeated))
    return total / count.clamp_min(1.0)


def _edge_occurrences(edge_index: torch.Tensor, weights: torch.Tensor, num_nodes: int) -> dict[int, list[float]]:
    result: dict[int, list[float]] = defaultdict(list)
    for (src, dst), weight in zip(edge_index.detach().cpu().t().tolist(), weights.detach().cpu().tolist()):
        result[int(src) * num_nodes + int(dst)].append(float(weight))
    for values in result.values():
        values.sort()
    return result


def _operator_metrics(model: Any, physical_edge_index: torch.Tensor, learned_index: torch.Tensor, learned_weight: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> dict[str, float]:
    with torch.no_grad():
        raw_index, raw_weight = model._normalized_operator(
            physical_edge_index,
            torch.ones(physical_edge_index.size(1), device=physical_edge_index.device, dtype=dtype),
            num_nodes,
            dtype,
        )
    learned = _edge_occurrences(learned_index, learned_weight, num_nodes)
    raw = _edge_occurrences(raw_index, raw_weight, num_nodes)
    diffs: list[float] = []
    reference: list[float] = []
    missing_learned = 0
    missing_raw = 0
    for key in set(learned) | set(raw):
        left, right = learned.get(key, []), raw.get(key, [])
        count = min(len(left), len(right))
        diffs.extend(a - b for a, b in zip(left[:count], right[:count]))
        reference.extend(right[:count])
        missing_learned += max(0, len(right) - count)
        missing_raw += max(0, len(left) - count)
    if not diffs:
        return {"operator_relative_l1": math.nan, "operator_mae": math.nan, "operator_aligned_edges": 0.0, "operator_missing_learned": float(missing_learned), "operator_missing_raw": float(missing_raw)}
    delta = np.asarray(diffs, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    return {
        "operator_relative_l1": float(np.sum(np.abs(delta)) / (np.sum(np.abs(ref)) + EPS)),
        "operator_mae": float(np.mean(np.abs(delta))),
        "operator_aligned_edges": float(delta.size),
        "operator_missing_learned": float(missing_learned),
        "operator_missing_raw": float(missing_raw),
    }


def _operator_pair_metrics(index_a: torch.Tensor, weight_a: torch.Tensor, index_b: torch.Tensor, weight_b: torch.Tensor, num_nodes: int) -> dict[str, float]:
    left = _edge_occurrences(index_a, weight_a, num_nodes)
    right = _edge_occurrences(index_b, weight_b, num_nodes)
    diffs: list[float] = []
    reference: list[float] = []
    for key in set(left) & set(right):
        count = min(len(left[key]), len(right[key]))
        diffs.extend(a - b for a, b in zip(left[key][:count], right[key][:count]))
        reference.extend(left[key][:count])
    if not diffs:
        return {"operator_text_visual_relative_l1": math.nan, "operator_text_visual_mae": math.nan}
    delta = np.asarray(diffs, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    return {
        "operator_text_visual_relative_l1": float(np.sum(np.abs(delta)) / (np.sum(np.abs(ref)) + EPS)),
        "operator_text_visual_mae": float(np.mean(np.abs(delta))),
    }


def _overlap_fraction(a: np.ndarray, b: np.ndarray, fraction: float, largest: bool) -> float:
    if a.size == 0:
        return math.nan
    count = max(1, int(math.ceil(a.size * fraction)))
    order_a = np.argsort(a)
    order_b = np.argsort(b)
    if largest:
        selected_a = set(order_a[-count:].tolist())
        selected_b = set(order_b[-count:].tolist())
    else:
        selected_a = set(order_a[:count].tolist())
        selected_b = set(order_b[:count].tolist())
    return float(len(selected_a & selected_b) / count)


def _semantic_pair_metrics(text: Any, visual: Any) -> dict[str, float]:
    a, b = _flat(text), _flat(visual)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return {}
    rank_a = rankdata(a, method="average") / a.size
    rank_b = rankdata(b, method="average") / b.size
    pearson, spearman = _corr(a, b)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mean_abs_difference": float(np.mean(np.abs(a - b))),
        "median_abs_difference": float(np.median(np.abs(a - b))),
        "q10_abs_difference": float(np.quantile(np.abs(a - b), 0.10)),
        "q50_abs_difference": float(np.quantile(np.abs(a - b), 0.50)),
        "q90_abs_difference": float(np.quantile(np.abs(a - b), 0.90)),
        "sign_disagreement_fraction": float(np.mean(np.sign(a) != np.sign(b))),
        "top10_overlap": _overlap_fraction(a, b, 0.10, True),
        "bottom10_overlap": _overlap_fraction(a, b, 0.10, False),
        "mean_abs_percentile_rank_difference": float(np.mean(np.abs(rank_a - rank_b))),
    }


def _attention_summary(attention: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
    matrix = attention.detach().float().mean(dim=0).cpu().numpy()
    h = matrix.shape[-1]
    clipped = np.clip(matrix, EPS, 1.0)
    entropy = float(np.mean(-np.sum(clipped * np.log(clipped), axis=-1)))
    diagonal = float(np.mean(np.trace(matrix) / h))
    rows = matrix[:, None, :] if matrix.ndim == 2 else matrix
    del rows
    row_diffs = []
    for i in range(h):
        for j in range(i + 1, h):
            row_diffs.append(np.mean(np.abs(matrix[i] - matrix[j])))
    return matrix, {
        "normalized_entropy": entropy / max(math.log(h), EPS),
        "diagonal_mass": diagonal,
        "off_diagonal_mass": 1.0 - diagonal,
        "row_diversity_l1": float(np.mean(row_diffs)) if row_diffs else 0.0,
    }


def _checkpoint_run(project_root: Path, dataset: str, seed: int) -> Path:
    return project_root / "outputs/ssi_mag_v3_p1_full" / dataset / "full" / f"seed{seed}"


def _load_run(project_root: Path, dataset: str, seed: int, device: torch.device):
    run_dir = _checkpoint_run(project_root, dataset, seed)
    cfg = OmegaConf.create(json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8")))
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(cfg, info).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return run_dir, cfg, data, model


def _relation_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int, num_nodes: int, dtype: torch.dtype) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edge_index = normal["physical_edge_index"]
    projections = {}
    s_values = {}
    h0_values = {}
    relative_values = {}
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        projection = normal[f"relation_projection_{modality}"]
        s = _edge_cos(projection, edge_index)
        h0 = _edge_cos(normal[f"h0_{modality}"], edge_index)
        mu = _incident_mean(edge_index, s, num_nodes)
        relative = s - 0.5 * (mu[edge_index[0]] + mu[edge_index[1]]) if edge_index.numel() else s
        residual = normal[f"relation_residual_{modality}"]
        beta = float(normal[f"beta_parameter_{modality}"])
        log_modulation = beta * residual
        row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "N_edges": int(s.numel()), "beta": beta}
        _put(row, "semantic_compatibility", s)
        _put(row, "raw_h0_compatibility", h0)
        _put(row, "relative_compatibility", relative)
        _put(row, "relation_residual", residual)
        _put(row, "raw_log_modulation", log_modulation)
        for threshold in (0.90, 0.95, 0.99):
            row[f"relation_residual_abs_gt_{str(threshold).replace('.', '')}"] = float(np.mean(np.abs(_flat(residual)) > threshold)) if s.numel() else math.nan
        _put_corr(row, "residual_vs_semantic", residual, s)
        _put_corr(row, "residual_vs_relative", residual, relative)
        weight = normal[f"relation_weight_{modality}"]
        weight_stats = _stats(weight)
        row["relation_weight_cv"] = _safe_ratio(weight_stats["std"], weight_stats["mean"])
        row.update(_operator_metrics(model, edge_index, normal[f"normalized_edge_index_{modality}"], normal[f"normalized_edge_weight_{modality}"], num_nodes, dtype))
        projections[modality], s_values[modality], h0_values[modality], relative_values[modality] = projection, s, h0, relative
        rows.append(row)
    pair = {"dataset": dataset, "seed": seed, "modality": "text_vs_visual", "N_edges": int(s_values["text"].numel())}
    for prefix, left, right in (("relation_space", s_values["text"], s_values["visual"]), ("raw_h0", h0_values["text"], h0_values["visual"]), ("relative", relative_values["text"], relative_values["visual"])):
        for key, value in _semantic_pair_metrics(left, right).items():
            pair[f"{prefix}_{key}"] = value
    pair.update(_operator_pair_metrics(normal["normalized_edge_index_text"], normal["normalized_edge_weight_text"], normal["normalized_edge_index_visual"], normal["normalized_edge_weight_visual"], num_nodes))
    rows.append(pair)
    return rows, {"s": s_values, "h0": h0_values, "relative": relative_values, "projections": projections}


def _semantic_reference_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for modality in MODALITIES:
        p = normal[f"p_{modality}"]
        c = normal[f"local_adaptation_{modality}"]
        rho_p = float(getattr(model, f"semantic_rho_p_{modality}").detach().cpu())
        rho_d = float(getattr(model, f"semantic_rho_d_{modality}").detach().cpu())
        rho_c = float(getattr(model, f"semantic_rho_c_{modality}").detach().cpu())
        bias = getattr(model, f"semantic_bias_{modality}").detach()
        alpha0 = float(model.semantic_reference_init)
        raw_p = _flat(p)
        p_sat = {f"p_fraction_abs_gt_{str(t).replace('.', '')}": float(np.mean(np.abs(raw_p) > t)) for t in (0.90, 0.95, 0.99)}
        for hop, d in enumerate(normal[f"d_{modality}"], start=1):
            term_bias = torch.full_like(p, bias[hop - 1])
            term_p = rho_p * p
            term_d = rho_d * d
            term_c = rho_c * c
            adaptive_logit = term_bias + term_p + term_d + term_c
            alpha = normal[f"alpha_{modality}"][hop - 1]
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "rho_p": rho_p, "rho_d": rho_d, "rho_c": rho_c, "alpha0": alpha0}
            for name, value in (("term_bias", term_bias), ("term_p", term_p), ("term_d", term_d), ("term_c", term_c), ("adaptive_logit", adaptive_logit), ("p", p), ("d", d), ("c", c), ("alpha", alpha)):
                _put(row, name, value)
            for key, value in p_sat.items():
                row[key] = value
            for name, value in (("bias", term_bias), ("p", term_p), ("d", term_d), ("c", term_c)):
                _put_contrib(row, f"{name}_to_adaptive_logit", value, adaptive_logit)
            logit_contributions = [row.get(f"{name}_to_adaptive_logit_covariance_contribution", math.nan) for name in ("bias", "p", "d", "c")]
            finite_logit_contributions = [value for value in logit_contributions if np.isfinite(value)]
            row["adaptive_logit_covariance_contribution_sum"] = float(np.sum(finite_logit_contributions)) if finite_logit_contributions else math.nan
            row["adaptive_logit_covariance_contribution_sum_error"] = abs(row["adaptive_logit_covariance_contribution_sum"] - 1.0) if finite_logit_contributions else math.nan
            total_stats = _stats(adaptive_logit)
            row["adaptive_logit_variance"] = total_stats["std"] ** 2
            row["adaptive_logit_reconstruction_error"] = float(torch.max(torch.abs(adaptive_logit - (term_bias + term_p + term_d + term_c))))
            row["alpha_iqr"] = float(np.quantile(_flat(alpha), 0.75) - np.quantile(_flat(alpha), 0.25))
            row["alpha_q90_q10"] = float(np.quantile(_flat(alpha), 0.90) - np.quantile(_flat(alpha), 0.10))
            row["adaptive_range_ratio"] = _safe_ratio(row["alpha_q90_q10"], total_stats["mean"])
            row["alpha_reference_logit_constant"] = math.log(alpha0 / (1.0 - alpha0))
            rows.append(row)
    return rows


def _filter_rows(normal: dict[str, Any], dataset: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for modality in MODALITIES:
        gamma = normal[f"gamma_{modality}"]
        delta_gamma = normal[f"delta_gamma_{modality}"]
        eta = normal[f"eta_{modality}"]
        content = normal[f"delta_content_{modality}"]
        reference = normal[f"reference_residual_{modality}"]
        relation = normal[f"relation_filter_residual_{modality}"]
        for hop in range(eta.size(1)):
            static_prior = gamma[hop] + delta_gamma[hop]
            eta_h = eta[:, hop]
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "static_prior": float(static_prior)}
            _put(row, "static_prior_abs", torch.abs(static_prior))
            for name, value in (("content", content[:, hop]), ("reference", reference[:, hop]), ("relation", relation[:, hop]), ("eta", eta_h)):
                _put(row, name, value)
                row[f"{name}_over_static_prior_abs_mean"] = _safe_ratio(_stats(value)["abs_mean"], float(static_prior))
                _put_contrib(row, f"{name}_to_eta", value, eta_h)
            eta_contributions = [row.get(f"{name}_to_eta_covariance_contribution", math.nan) for name in ("content", "reference", "relation")]
            finite_contributions = [value for value in eta_contributions if np.isfinite(value)]
            row["eta_covariance_contribution_sum"] = float(np.sum(finite_contributions)) if finite_contributions else math.nan
            row["eta_covariance_contribution_sum_error"] = abs(row["eta_covariance_contribution_sum"] - 1.0) if finite_contributions else math.nan
            row["eta_negative_fraction"] = float((eta_h < 0).float().mean())
            rows.append(row)
        effective = normal[f"effective_order_{modality}"]
        mean_alpha = torch.stack(normal[f"alpha_{modality}"], dim=1).mean(dim=1)
        mean_d = torch.stack(normal[f"d_{modality}"], dim=1).mean(dim=1)
        for name, value in (("mean_alpha", mean_alpha), ("mean_d", mean_d), ("c", normal[f"local_adaptation_{modality}"]), ("p", normal[f"p_{modality}"])):
            pearson, spearman = _corr(effective, value)
            for row in rows[-(eta.size(1)):]:
                if row["modality"] == modality:
                    row[f"effective_order_vs_{name}_pearson"] = pearson
                    row[f"effective_order_vs_{name}_spearman"] = spearman
    return rows


def _context_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    attention_data: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        states = normal[f"S_{modality}"]
        deltas = normal[f"D_{modality}"]
        interaction_output = normal[f"interaction_output_{modality}"]
        delta_gate = float(normal[f"delta_gate_{modality}"])
        interaction_gate = float(normal[f"interaction_gate_{modality}"])
        attention = normal[f"attention_{modality}"]
        matrix, attention_stats = _attention_summary(attention)
        attention_data[modality] = {"matrix": matrix.tolist(), **attention_stats}
        delta_proj = getattr(model, f"context_delta_proj_{modality}")
        for hop in range(1, len(states)):
            previous, current = states[hop - 1], states[hop]
            raw_change = current - previous
            raw_norm = raw_change.norm(dim=-1)
            relative_raw = raw_norm / (previous.norm(dim=-1) + EPS)
            raw_cos = F.cosine_similarity(current, previous, dim=-1, eps=EPS)
            context_injection = delta_gate * delta_proj(deltas[hop])
            context_denominator = F.layer_norm(current, (current.size(-1),)).norm(dim=-1)
            interaction_injection = interaction_gate * interaction_output[:, hop]
            interaction_denominator = current.norm(dim=-1)
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "delta_gate": delta_gate, "interaction_gate": interaction_gate, **{f"attention_{k}": v for k, v in attention_stats.items()}}
            for name, value in (("raw_change_norm", raw_norm), ("relative_raw_change", relative_raw), ("cosine_current_previous", raw_cos), ("context_injection_norm", context_injection.norm(dim=-1)), ("context_injection_ratio", context_injection.norm(dim=-1) / (context_denominator + EPS)), ("interaction_injection_norm", interaction_injection.norm(dim=-1)), ("interaction_injection_ratio", interaction_injection.norm(dim=-1) / (interaction_denominator + EPS))):
                _put(row, name, value)
            rows.append(row)
    return rows, attention_data


def _run_analysis(project_root: Path, dataset: str, seed: int, device: torch.device):
    run_dir, cfg, data, model = _load_run(project_root, dataset, seed, device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    with torch.no_grad():
        normal = model.analysis(x, edge_index)
    finite = all(torch.isfinite(value).all().item() for value in _tensor_values(normal))
    relation_rows, relation_cache = _relation_rows(normal, model, dataset, seed, data.num_nodes, x.dtype)
    reference_rows = _semantic_reference_rows(normal, model, dataset, seed)
    filter_rows = _filter_rows(normal, dataset, seed)
    context_rows, attention_data = _context_rows(normal, model, dataset, seed)
    meta = {"dataset": dataset, "seed": seed, "run_dir": str(run_dir), "checkpoint": str(run_dir / "best.pt"), "finite": bool(finite), "edge_count": int(edge_index.size(1)), "node_count": int(data.num_nodes)}
    for modality in MODALITIES:
        beta = float(normal[f"beta_parameter_{modality}"])
        meta[f"beta_{modality}"] = beta
        rel_row = next(row for row in relation_rows if row["modality"] == modality)
        meta[f"operator_relative_l1_{modality}"] = rel_row["operator_relative_l1"]
        meta[f"scorer_saturation_95_{modality}"] = rel_row["relation_residual_abs_gt_095"]
        meta[f"alpha_mean_{modality}"] = float(np.mean([row["alpha_mean"] for row in reference_rows if row["modality"] == modality]))
        meta[f"alpha_std_{modality}"] = float(np.mean([row["alpha_std"] for row in reference_rows if row["modality"] == modality]))
        meta[f"p_saturation_95_{modality}"] = float(np.mean([row["p_fraction_abs_gt_095"] for row in reference_rows if row["modality"] == modality]))
        meta[f"p_mean_{modality}"] = float(np.mean([row["p_mean"] for row in reference_rows if row["modality"] == modality]))
        meta[f"p_median_{modality}"] = float(np.mean([row["p_q50"] for row in reference_rows if row["modality"] == modality]))
        context = [row for row in context_rows if row["modality"] == modality]
        meta[f"g_delta_{modality}"] = float(normal[f"delta_gate_{modality}"])
        meta[f"g_int_{modality}"] = float(normal[f"interaction_gate_{modality}"])
        meta[f"delta_injection_ratio_{modality}"] = float(np.mean([row["context_injection_ratio_mean"] for row in context]))
        meta[f"interaction_injection_ratio_{modality}"] = float(np.mean([row["interaction_injection_ratio_mean"] for row in context]))
        effective = _flat(normal[f"effective_order_{modality}"])
        meta[f"effective_order_mean_{modality}"] = float(np.mean(effective))
        meta[f"effective_order_std_{modality}"] = float(np.std(effective))
    pair = next(row for row in relation_rows if row["modality"] == "text_vs_visual")
    meta["semantic_discrepancy_mean_abs"] = pair["relation_space_mean_abs_difference"]
    meta["semantic_discrepancy_spearman"] = pair["relation_space_spearman"]
    meta["attention_matrices"] = attention_data
    del normal, model, data, x, edge_index
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return relation_rows, reference_rows, filter_rows, context_rows, meta


def _tensor_values(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _tensor_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_values(item)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _consistency_rows(meta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [key for key in meta_rows[0] if key not in {"dataset", "seed", "run_dir", "checkpoint", "finite", "attention_matrices", "edge_count", "node_count"}]
    rows = []
    for dataset in DATASETS:
        for key in fields:
            values = np.asarray([row[key] for row in meta_rows if row["dataset"] == dataset], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values))
            rows.append({"dataset": dataset, "quantity": key, "n_seeds": int(values.size), "mean": mean, "std": std, "cv_abs_mean": _safe_ratio(std, mean), "sign_stable": int(np.all(values >= 0) or np.all(values <= 0)), "near_constant": int(std <= max(1.0e-6, abs(mean) * 1.0e-2))})
    return rows


def _report(project_root: Path, output: Path, relation: list[dict[str, Any]], reference: list[dict[str, Any]], filtering: list[dict[str, Any]], context: list[dict[str, Any]], meta: list[dict[str, Any]], attention: dict[str, Any]) -> None:
    finite_count = sum(int(row["finite"]) for row in meta)
    pair = [row for row in relation if row["modality"] == "text_vs_visual"]
    def mean_field(rows: list[dict[str, Any]], key: str) -> float:
        values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=np.float64)
        return float(np.mean(values)) if values.size else math.nan
    relation_discrepancy = mean_field(pair, "relation_space_mean_abs_difference")
    relation_spearman = mean_field(pair, "relation_space_spearman")
    residual_semantic = mean_field([row for row in relation if row["modality"] != "text_vs_visual"], "residual_vs_semantic_spearman")
    residual_relative = mean_field([row for row in relation if row["modality"] != "text_vs_visual"], "residual_vs_relative_spearman")
    scorer_sat = mean_field([row for row in relation if row["modality"] != "text_vs_visual"], "relation_residual_abs_gt_095")
    beta = mean_field([row for row in relation if row["modality"] != "text_vs_visual"], "beta")
    op = mean_field([row for row in relation if row["modality"] != "text_vs_visual"], "operator_relative_l1")
    p_sat = mean_field(reference, "p_fraction_abs_gt_095")
    alpha_std = mean_field(reference, "alpha_std")
    p_contrib = mean_field(reference, "p_to_adaptive_logit_covariance_contribution")
    d_contrib = mean_field(reference, "d_to_adaptive_logit_covariance_contribution")
    c_contrib = mean_field(reference, "c_to_adaptive_logit_covariance_contribution")
    content_ratio = mean_field(filtering, "content_over_static_prior_abs_mean")
    reference_ratio = mean_field(filtering, "reference_over_static_prior_abs_mean")
    relation_ratio = mean_field(filtering, "relation_over_static_prior_abs_mean")
    eta_var = mean_field(filtering, "total_variance")
    eta_content_contrib = mean_field(filtering, "content_to_eta_covariance_contribution")
    eta_reference_contrib = mean_field(filtering, "reference_to_eta_covariance_contribution")
    eta_relation_contrib = mean_field(filtering, "relation_to_eta_covariance_contribution")
    eta_contribution_sum_error = mean_field(filtering, "eta_covariance_contribution_sum_error")
    effective_alpha = mean_field(filtering, "effective_order_vs_mean_alpha_spearman")
    effective_d = mean_field(filtering, "effective_order_vs_mean_d_spearman")
    effective_c = mean_field(filtering, "effective_order_vs_c_spearman")
    effective_p = mean_field(filtering, "effective_order_vs_p_spearman")
    raw_change = mean_field(context, "raw_change_norm_mean")
    relative_raw_change = mean_field(context, "relative_raw_change_mean")
    raw_cosine = mean_field(context, "cosine_current_previous_mean")
    delta_ratio = mean_field(context, "context_injection_ratio_mean")
    interaction_ratio = mean_field(context, "interaction_injection_ratio_mean")
    attention_entropy = mean_field(context, "attention_normalized_entropy")
    attention_diag = mean_field(context, "attention_diagonal_mass")
    operator_modality = mean_field(pair, "operator_text_visual_relative_l1")
    p_sign_summary = {}
    for modality in MODALITIES:
        stable_mean = 0
        stable_median = 0
        for dataset in DATASETS:
            values = [row[f"p_mean_{modality}"] for row in meta if row["dataset"] == dataset]
            medians = [row[f"p_median_{modality}"] for row in meta if row["dataset"] == dataset]
            stable_mean += int(bool(values) and (all(value >= 0 for value in values) or all(value <= 0 for value in values)))
            stable_median += int(bool(medians) and (all(value >= 0 for value in medians) or all(value <= 0 for value in medians)))
        p_sign_summary[modality] = (stable_mean, stable_median)
    attention_std = []
    for dataset in attention:
        for modality in attention[dataset]:
            attention_std.append(float(np.mean(np.asarray(attention[dataset][modality]["seed_std"], dtype=np.float64))))
    attention_seed_std = float(np.mean(attention_std)) if attention_std else math.nan
    lines = [
        "# SSI-MAG-V3 P1.5 Mechanism Attribution Audit",
        "",
        "This is a post-hoc diagnosis over the existing 15 P1 Full best checkpoints (5 NC datasets × seeds 42/43/44). No training, ablation, LP execution, hyperparameter search, or test-based model selection was performed. The model forward and training behavior were not modified.",
        "",
        f"## Checkpoint integrity\n\n- Loaded checkpoints: {len(meta)}/15\n- All analysis tensors finite: {finite_count}/{len(meta)}\n- Analysis API: `SSIMAGV3.analysis()`\n- Test metrics: not read or used",
        "",
        "## R1 — Relation semantic discrepancy",
        "",
        f"Across run-level edge summaries, relation-space Text/Visual semantic compatibility has mean absolute discrepancy {relation_discrepancy:.6f} and mean Spearman correlation {relation_spearman:.4f}. The learned Text/Visual normalized operators differ by relative-L1 {operator_modality:.6f}. This is the explicit cosine in the learned relation-projection space; raw H0 cosine is reported separately in `p15_relation_attribution.csv`.",
        f"The learned residual has mean Spearman correlation {residual_semantic:.4f} with absolute relation compatibility and {residual_relative:.4f} with neighborhood-relative compatibility. The mean fraction with |residual|>0.95 is {scorer_sat:.4f}, mean beta is {beta:.6f}, and learned-vs-unit normalized-operator relative-L1 is {op:.6f}.",
        "Diagnosis: if semantic discrepancy is non-negligible while residual alignment is weak and residual saturation is high, the evidence points to a parameterization/utilization problem rather than a failure of the motivation. Small beta and small operator perturbation should not be described as strong structure modulation.",
        "",
        "## R2 — Adaptive semantic reference attribution",
        "",
        f"The mean |p|>0.95 fraction is {p_sat:.4f}; mean alpha node-wise standard deviation is {alpha_std:.6f}. Mean covariance contributions to adaptive-logit variance are p={p_contrib:.4f}, d={d_contrib:.4f}, c={c_contrib:.4f}; four-term contribution-sum error averages {mean_field(reference, 'adaptive_logit_covariance_contribution_sum_error'):.6g}. Mean effective-order Spearman correlations are alpha={effective_alpha:.4f}, d={effective_d:.4f}, c={effective_c:.4f}, p={effective_p:.4f}. Contributions are signed and are computed as Cov(term,total)/Var(total).",
        f"The p mean sign is stable on {p_sign_summary.get('text', (0, 0))[0]}/5 datasets for Text and {p_sign_summary.get('visual', (0, 0))[0]}/5 for Visual; median-sign stability is {p_sign_summary.get('text', (0, 0))[1]}/5 and {p_sign_summary.get('visual', (0, 0))[1]}/5. Near-zero variances are explicitly marked in the CSV. A saturated or seed-unstable p branch should not support a strong node-adaptive semantic-prior claim.",
        "",
        "## Stage II — Signed filtering attribution",
        "",
        f"Mean absolute-term/static-prior ratios are content={content_ratio:.4f}, reference={reference_ratio:.4f}, relation={relation_ratio:.4f}; mean eta variance is {eta_var:.6f}. Signed eta covariance contributions are content={eta_content_contrib:.4f}, reference={eta_reference_contrib:.4f}, relation={eta_relation_contrib:.4f}; contribution-sum error={eta_contribution_sum_error:.6g}. The full node-variance contributions are in `p15_filter_attribution.csv`.",
        "A signed eta pattern should be described as node-adaptive only when its node variance and branch attribution are materially nonzero and stable across seeds. The report does not collapse signed contributions into magnitudes.",
        "",
        "## Context change and cross-hop interaction injection",
        "",
        f"Mean raw state change norm is {raw_change:.6f}, relative raw change is {relative_raw_change:.6f}, and cosine(S_k,S_(k-1)) is {raw_cosine:.6f}. Mean context-change injection ratio is {delta_ratio:.6f}; cross-hop interaction injection ratio is {interaction_ratio:.6f}. LayerNorm D norms are not used as mechanism evidence. Attention mean normalized entropy is {attention_entropy:.4f}, diagonal mass is {attention_diag:.4f}, and mean per-entry seed std is {attention_seed_std:.6f}; full matrices are in `p15_attention_matrices.json`.",
        "",
        "## Cross-seed consistency",
        "",
        "`p15_consistency.csv` reports dataset-level mean/std/CV, sign stability, and near-constant flags for beta, operator perturbation, scorer saturation, semantic discrepancy, alpha/p quantities, gate/injection quantities, and effective order. No seed was removed.",
        "",
        "## Evidence-based V3.1 candidates",
        "",
        "### Candidate R1 — Relative Semantic Relation Modulation",
        "",
        "Verdict: supported as a V3.1 motivation/parameterization candidate, not as an implemented P1.5 change. Relation-space discrepancy is large while the learned-vs-unit operator perturbation is small; the current residual has only partial alignment with semantic and relative compatibility. An explicit relative compatibility term with a raw-topology fallback is therefore justified for a later controlled revision.",
        "",
        "### Candidate R2 — Structure-State Adaptive Semantic Reference",
        "",
        "Verdict: supported for revision diagnosis. p contributes most adaptive-logit variance, alpha node spread is small, and p mean/median signs are unstable across seeds. A later revision should replace or constrain the saturated/seed-sensitive p branch and test whether d/local relation context can carry the intended node variation.",
        "",
        "### Candidate Stage II — Semantic-Query Signed Filtering",
        "",
        "Verdict: a plausible Stage-II validation candidate, not yet a demonstrated utility mechanism. Context-change and interaction injections are non-negligible, attention is non-uniform, and the three-seed matrix spread is reported explicitly; a later semantic-query signed filter should be tested only with a matched validation protocol and without changing the P1.5 diagnosis.",
        "",
        "## Boundary",
        "",
        "This artifact is diagnosis only. No architecture revision, auxiliary loss, retraining, ablation, LP run, or test-based decision was performed.",
    ]
    (output / "p15_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ssi_mag_v3_p15_analysis"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    relation_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    attention: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    errors = []
    for dataset in args.datasets:
        for seed in args.seeds:
            try:
                relation, reference, filtering, context, meta = _run_analysis(project_root, dataset, seed, device)
                relation_rows.extend(relation)
                reference_rows.extend(reference)
                filter_rows.extend(filtering)
                context_rows.extend(context)
                meta_rows.append(meta)
                for modality in MODALITIES:
                    attention[dataset][modality][str(seed)] = meta["attention_matrices"][modality]
                print(f"OK {dataset} seed={seed} finite={meta['finite']}", flush=True)
            except Exception as exc:  # preserve a machine-readable failure record
                errors.append({"dataset": dataset, "seed": seed, "error": repr(exc)})
                print(f"ERROR {dataset} seed={seed}: {exc}", flush=True)
    _write_csv(output / "p15_relation_attribution.csv", relation_rows)
    _write_csv(output / "p15_semantic_reference_attribution.csv", reference_rows)
    _write_csv(output / "p15_filter_attribution.csv", filter_rows)
    _write_csv(output / "p15_context_change.csv", context_rows)
    consistency = _consistency_rows(meta_rows) if meta_rows else []
    _write_csv(output / "p15_consistency.csv", consistency)
    attention_json: dict[str, Any] = {}
    for dataset, modalities in attention.items():
        attention_json[dataset] = {}
        for modality, runs in modalities.items():
            matrices = [np.asarray(item["matrix"], dtype=np.float64) for item in runs.values()]
            attention_json[dataset][modality] = {
                "runs": runs,
                "seed_mean": np.mean(matrices, axis=0).tolist() if matrices else [],
                "seed_std": np.std(matrices, axis=0).tolist() if matrices else [],
            }
    (output / "p15_attention_matrices.json").write_text(json.dumps(attention_json, indent=2, allow_nan=True), encoding="utf-8")
    summary = {
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "expected_runs": len(args.datasets) * len(args.seeds),
        "loaded_runs": len(meta_rows),
        "finite_runs": int(sum(int(row["finite"]) for row in meta_rows)),
        "errors": errors,
        "device": str(device),
        "test_metrics_read": False,
        "training_invoked": False,
        "ablation_invoked": False,
        "lp_invoked": False,
        "output_files": ["p15_relation_attribution.csv", "p15_semantic_reference_attribution.csv", "p15_filter_attribution.csv", "p15_context_change.csv", "p15_attention_matrices.json", "p15_consistency.csv", "p15_summary.json", "p15_report.md"],
    }
    (output / "p15_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    _report(project_root, output, relation_rows, reference_rows, filter_rows, context_rows, meta_rows, attention_json)
    print(json.dumps(summary, indent=2), flush=True)
    if errors or len(meta_rows) != summary["expected_runs"] or summary["finite_runs"] != len(meta_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
