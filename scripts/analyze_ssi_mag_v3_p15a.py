#!/usr/bin/env python3
"""P1.5a corrected diagnostics for the existing SSI-MAG-V3 P1 checkpoints.

This is an analysis-only artifact.  It loads the existing Full P1 best
checkpoints, calls the frozen analysis API, and writes corrected attention,
semantic-reference, and context-injection diagnostics.  It never trains,
reads test metrics, runs an ablation, or runs an LP job.

The frozen model returns attention with shape [N, query_hop, key_hop].  The
attention metrics in this file intentionally aggregate over nodes only after
computing node/query-level quantities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_ssi_mag_v3_p15 import (  # noqa: E402
    DATASETS,
    MODALITIES,
    SEEDS,
    _load_run,
)


EPS = 1.0e-12


def _as_np(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _flat(value: Any) -> np.ndarray:
    return _as_np(value).reshape(-1).astype(np.float64, copy=False)


def _finite(value: Any) -> np.ndarray:
    return _flat(value)[np.isfinite(_flat(value))]


def _stats(value: Any) -> dict[str, float]:
    x = _finite(value)
    if x.size == 0:
        return {key: math.nan for key in ("mean", "std", "abs_mean", "min", "max", "q10", "q25", "q50", "q75", "q90")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "abs_mean": float(np.mean(np.abs(x))),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
    }


def _put(row: dict[str, Any], prefix: str, value: Any) -> None:
    for key, item in _stats(value).items():
        row[f"{prefix}_{key}"] = item


def _safe_ratio(num: float, den: float) -> float:
    return float(num / (abs(den) + EPS))


def _corr(x: Any, y: Any) -> tuple[float, float]:
    a, b = _flat(x), _flat(y)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) <= EPS or np.std(b) <= EPS:
        return math.nan, math.nan
    return float(pearsonr(a, b).statistic), float(spearmanr(a, b).statistic)


def _alpha_range_ratio(alpha: Any, eps: float = EPS) -> float:
    """Corrected adaptive-range ratio: (q90-q10)/(abs(mean(alpha))+eps)."""
    x = _finite(alpha)
    if x.size == 0:
        return math.nan
    return float((np.quantile(x, 0.90) - np.quantile(x, 0.10)) / (abs(np.mean(x)) + eps))


def _entropy_normalized(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, EPS, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=-1) / max(math.log(prob.shape[-1]), EPS)


def attention_diagnostics(attention: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compute corrected attention diagnostics from [N, query-hop, key-hop].

    Query diversity is the mean, within each node, of pairwise total
    variation distance between query rows:
    TV(A_iq, A_iq') = 0.5 * sum_h |A_iqh - A_iq'h|.
    Node heterogeneity is the same TV distance from each node/query row to
    the across-node query-row mean Abar_q = mean_i A_iq.
    """
    if not torch.is_tensor(attention) or attention.ndim != 3:
        raise ValueError(f"attention must be rank-3 [N,Q,K], got {getattr(attention, 'shape', None)}")
    array = attention.detach().float().cpu().numpy().astype(np.float64, copy=False)
    if array.shape[1] != array.shape[2]:
        raise ValueError(f"attention must have square hop axes, got {array.shape}")
    n_nodes, n_hops, _ = array.shape
    node_entropy = _entropy_normalized(array).reshape(-1)
    mean_matrix = np.mean(array, axis=0)
    entrywise_node_std = np.std(array, axis=0)
    mean_matrix_entropy = float(np.mean(_entropy_normalized(mean_matrix)))

    pairwise_tv = []
    for query_a in range(n_hops):
        for query_b in range(query_a + 1, n_hops):
            pairwise_tv.append(0.5 * np.sum(np.abs(array[:, query_a, :] - array[:, query_b, :]), axis=-1))
    query_diversity = np.mean(np.stack(pairwise_tv, axis=1), axis=1) if pairwise_tv else np.zeros(n_nodes)

    query_means = np.mean(array, axis=0)
    node_heterogeneity = 0.5 * np.sum(np.abs(array - query_means[None, :, :]), axis=-1).reshape(-1)
    diagonal_per_node = np.mean(np.diagonal(array, axis1=1, axis2=2), axis=1)
    uniform_diag = 1.0 / n_hops
    matrix_row_diversity = []
    for row_a in range(n_hops):
        for row_b in range(row_a + 1, n_hops):
            matrix_row_diversity.append(np.mean(np.abs(mean_matrix[row_a] - mean_matrix[row_b])))
    old_row_diversity = float(np.mean(matrix_row_diversity)) if matrix_row_diversity else 0.0
    nodewise_entropy_mean = float(np.mean(node_entropy))
    metrics: dict[str, float] = {
        "attention_nodes": float(n_nodes),
        "attention_hops": float(n_hops),
        "nodewise_normalized_entropy": nodewise_entropy_mean,
        "nodewise_entropy_std": float(np.std(node_entropy)),
        "nodewise_entropy_q10": float(np.quantile(node_entropy, 0.10)),
        "nodewise_entropy_q50": float(np.quantile(node_entropy, 0.50)),
        "nodewise_entropy_q90": float(np.quantile(node_entropy, 0.90)),
        "attention_nonuniformity": 1.0 - nodewise_entropy_mean,
        "mean_matrix_normalized_entropy": mean_matrix_entropy,
        "old_mean_matrix_normalized_entropy": mean_matrix_entropy,
        "jensen_entropy_gap": mean_matrix_entropy - nodewise_entropy_mean,
        "query_diversity_mean": float(np.mean(query_diversity)),
        "query_diversity_std": float(np.std(query_diversity)),
        "query_diversity_q10": float(np.quantile(query_diversity, 0.10)),
        "query_diversity_q50": float(np.quantile(query_diversity, 0.50)),
        "query_diversity_q90": float(np.quantile(query_diversity, 0.90)),
        "node_heterogeneity_mean": float(np.mean(node_heterogeneity)),
        "node_heterogeneity_std": float(np.std(node_heterogeneity)),
        "node_heterogeneity_q10": float(np.quantile(node_heterogeneity, 0.10)),
        "node_heterogeneity_q50": float(np.quantile(node_heterogeneity, 0.50)),
        "node_heterogeneity_q90": float(np.quantile(node_heterogeneity, 0.90)),
        "diagonal_mass_mean": float(np.mean(diagonal_per_node)),
        "diagonal_mass_std": float(np.std(diagonal_per_node)),
        "diagonal_mass_q10": float(np.quantile(diagonal_per_node, 0.10)),
        "diagonal_mass_q50": float(np.quantile(diagonal_per_node, 0.50)),
        "diagonal_mass_q90": float(np.quantile(diagonal_per_node, 0.90)),
        "uniform_diagonal_mass": uniform_diag,
        "diagonal_excess": float(np.mean(diagonal_per_node) - uniform_diag),
        "off_diagonal_mass": 1.0 - float(np.mean(diagonal_per_node)),
        "mean_matrix_row_diversity": old_row_diversity,
    }
    return mean_matrix, entrywise_node_std, metrics


def _edge_occurrences(edge_index: torch.Tensor, weights: torch.Tensor, num_nodes: int) -> dict[int, list[float]]:
    result: dict[int, list[float]] = defaultdict(list)
    for (src, dst), weight in zip(edge_index.detach().cpu().t().tolist(), weights.detach().cpu().tolist()):
        result[int(src) * num_nodes + int(dst)].append(float(weight))
    for values in result.values():
        values.sort()
    return result


def _operator_perturbation(model: Any, physical_edge_index: torch.Tensor, learned_index: torch.Tensor, learned_weight: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> dict[str, float]:
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
        return {"operator_aligned_edges": 0.0, "operator_weight_mae": math.nan, "operator_weight_rmse": math.nan, "operator_weight_relative_l1": math.nan, "operator_missing_learned": float(missing_learned), "operator_missing_raw": float(missing_raw)}
    delta = np.asarray(diffs, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    return {
        "operator_aligned_edges": float(delta.size),
        "operator_weight_mae": float(np.mean(np.abs(delta))),
        "operator_weight_rmse": float(np.sqrt(np.mean(delta * delta))),
        "operator_weight_relative_l1": float(np.sum(np.abs(delta)) / (np.sum(np.abs(ref)) + EPS)),
        "operator_missing_learned": float(missing_learned),
        "operator_missing_raw": float(missing_raw),
    }


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _finite_analysis(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, list):
        return all(_finite_analysis(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_analysis(item) for item in value.values())
    return True


def _relation_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int, num_nodes: int, dtype: torch.dtype) -> list[dict[str, Any]]:
    rows = []
    physical = normal["physical_edge_index"]
    for modality in MODALITIES:
        residual = normal[f"relation_residual_{modality}"]
        weights = normal[f"relation_weight_{modality}"]
        row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "beta": float(normal[f"beta_parameter_{modality}"]), "relation_weight_cv": _safe_ratio(float(_stats(weights)["std"]), float(_stats(weights)["mean"]))}
        _put(row, "relation_residual", residual)
        _put(row, "relation_weight", weights)
        _put(row, "local_adaptation", normal[f"local_adaptation_{modality}"])
        row.update(_operator_perturbation(model, physical, normal[f"normalized_edge_index_{modality}"], normal[f"normalized_edge_weight_{modality}"], num_nodes, dtype))
        rows.append(row)
    return rows


def _semantic_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    vectors: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        p = normal[f"p_{modality}"]
        rho_p = float(getattr(model, f"semantic_rho_p_{modality}").detach().cpu())
        rho_d = float(getattr(model, f"semantic_rho_d_{modality}").detach().cpu())
        rho_c = float(getattr(model, f"semantic_rho_c_{modality}").detach().cpu())
        bias = getattr(model, f"semantic_bias_{modality}").detach().cpu().numpy().tolist()
        term_p = rho_p * p
        vectors[modality] = {"p": p.detach().cpu(), "term_p": term_p.detach().cpu(), "alpha": [value.detach().cpu() for value in normal[f"alpha_{modality}"]]}
        p_array = _as_np(p)
        p_saturation = {f"p_fraction_abs_gt_{threshold:g}": float(np.mean(np.abs(p_array) > threshold)) for threshold in (0.90, 0.95, 0.99)}
        p_sign_balance = float(np.mean(p_array > 0.0) - np.mean(p_array < 0.0))
        for hop, (d, alpha) in enumerate(zip(normal[f"d_{modality}"], normal[f"alpha_{modality}"], strict=True), start=1):
            alpha_stats = _stats(alpha)
            row: dict[str, Any] = {"row_type": "run_hop", "dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "semantic_bias": float(bias[hop - 1]), "rho_p": rho_p, "rho_d": rho_d, "rho_c": rho_c, "p_sign_balance": p_sign_balance, "alpha_range_ratio": _alpha_range_ratio(alpha)}
            c = normal[f"local_adaptation_{modality}"]
            row["alpha_iqr"] = alpha_stats["q75"] - alpha_stats["q25"]
            row["alpha_q90_q10"] = alpha_stats["q90"] - alpha_stats["q10"]
            alpha_array = _as_np(alpha)
            row["alpha_fraction_lt_0.05"] = float(np.mean(alpha_array < 0.05))
            row["alpha_fraction_gt_0.95"] = float(np.mean(alpha_array > 0.95))
            _put(row, "p", p)
            _put(row, "term_p", term_p)
            _put(row, "d", d)
            _put(row, "alpha", alpha)
            _put(row, "local_adaptation", c)
            for key, value in p_saturation.items():
                row[key] = value
            rows.append(row)
    return rows, vectors


def _context_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = []
    attention_data: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        attention = normal[f"attention_{modality}"]
        if attention is None:
            raise ValueError(f"missing attention for {dataset} seed={seed} modality={modality}")
        matrix, node_std, attention_metrics = attention_diagnostics(attention)
        attention_data[modality] = {"shape": list(attention.shape), "mean_matrix": matrix.tolist(), "entrywise_node_std": node_std.tolist(), "metrics": attention_metrics}
        states = normal[f"S_{modality}"]
        deltas = normal[f"D_{modality}"]
        interaction_output = normal[f"interaction_output_{modality}"]
        delta_gate = float(normal[f"delta_gate_{modality}"])
        interaction_gate = float(normal[f"interaction_gate_{modality}"])
        delta_proj = getattr(model, f"context_delta_proj_{modality}")
        for hop in range(1, len(states)):
            current = states[hop]
            normalized_current = F.layer_norm(current, (current.size(-1),))
            c_delta = delta_gate * delta_proj(deltas[hop])
            c_int = interaction_gate * interaction_output[:, hop]
            s_tilde = normal[f"S_tilde_{modality}"][hop]
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "attention_shape": str(list(attention.shape)), "delta_gate": delta_gate, "interaction_gate": interaction_gate, **{(key if key.startswith("attention_") else f"attention_{key}"): value for key, value in attention_metrics.items()}}
            _put(row, "D_norm", deltas[hop].norm(dim=-1))
            _put(row, "context_injection_ratio", c_delta.norm(dim=-1) / (normalized_current.norm(dim=-1) + EPS))
            _put(row, "context_injection_norm", c_delta.norm(dim=-1))
            _put(row, "context_injection_cosine_to_layernorm_S", F.cosine_similarity(c_delta, normalized_current, dim=-1, eps=EPS))
            _put(row, "interaction_injection_ratio", c_int.norm(dim=-1) / (current.norm(dim=-1) + EPS))
            _put(row, "interaction_injection_norm", c_int.norm(dim=-1))
            _put(row, "interaction_injection_cosine_to_S", F.cosine_similarity(c_int, current, dim=-1, eps=EPS))
            _put(row, "S_tilde_cosine_to_S", F.cosine_similarity(s_tilde, current, dim=-1, eps=EPS))
            rows.append(row)
    return rows, attention_data


def _filter_rows(normal: dict[str, Any], dataset: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for modality in MODALITIES:
        eta = normal[f"eta_{modality}"]
        for hop in range(eta.size(1)):
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop}
            _put(row, "gamma_global", normal[f"gamma_{modality}"][hop])
            _put(row, "delta_gamma", normal[f"delta_gamma_{modality}"][hop])
            _put(row, "delta_content", normal[f"delta_content_{modality}"][:, hop])
            _put(row, "reference_residual", normal[f"reference_residual_{modality}"][:, hop])
            _put(row, "relation_filter_residual", normal[f"relation_filter_residual_{modality}"][:, hop])
            _put(row, "eta", eta[:, hop])
            row["negative_eta_fraction"] = float((eta[:, hop] < 0).float().mean())
            rows.append(row)
        _put(rows[-eta.size(1)], "effective_order", normal[f"effective_order_{modality}"])
        _put(rows[-eta.size(1)], "effective_radius", normal[f"effective_radius_{modality}"])
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _cross_seed_rows(vectors: dict[tuple[str, int, str], dict[str, Any]], datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        for modality in MODALITIES:
            for seed_a_index, seed_a in enumerate(seeds):
                for seed_b in seeds[seed_a_index + 1:]:
                    left = vectors[(dataset, seed_a, modality)]
                    right = vectors[(dataset, seed_b, modality)]
                    p, s = _corr(left["term_p"], right["term_p"])
                    rows.append({"row_type": "cross_seed_correlation", "dataset": dataset, "modality": modality, "metric": "term_p", "seed_a": seed_a, "seed_b": seed_b, "node_ordering_verified": True, "pearson": p, "spearman": s})
                    for hop, (alpha_a, alpha_b) in enumerate(zip(left["alpha"], right["alpha"], strict=True), start=1):
                        p, s = _corr(alpha_a, alpha_b)
                        rows.append({"row_type": "cross_seed_correlation", "dataset": dataset, "modality": modality, "metric": "alpha", "hop": hop, "seed_a": seed_a, "seed_b": seed_b, "node_ordering_verified": True, "pearson": p, "spearman": s})
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=np.float64)
    return float(np.mean(values)) if values.size else math.nan


def _report(output: Path, summary: dict[str, Any], relation_rows: list[dict[str, Any]], attention_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]], context_rows: list[dict[str, Any]], filter_rows: list[dict[str, Any]]) -> None:
    run_attention = attention_rows
    cross = [row for row in semantic_rows if row.get("row_type") == "cross_seed_correlation"]
    term_p_corr = [row for row in cross if row.get("metric") == "term_p"]
    alpha_corr = [row for row in cross if row.get("metric") == "alpha"]
    lines = [
        "# SSI-MAG-V3 P1.5a Corrected Diagnostic Validation",
        "",
        "This independent post-hoc analyzer reads only the existing P1 Full best checkpoints. It does not train, run ablations, run LP, read test metrics, select checkpoints, tune hyperparameters, or modify the SSI-MAG-V3 model and the historical P1.5 artifacts.",
        "",
        f"## Integrity\n\n- Checkpoints loaded: {summary['loaded_runs']}/{summary['expected_runs']}\n- Finite analysis runs: {summary['finite_runs']}/{summary['loaded_runs']}\n- Node ordering verified for cross-seed correlations: {summary['node_ordering_verified']}\n- Attention shape convention: `[N, query-hop, key-hop]`; observed shapes: `{summary['attention_shapes']}`",
        "",
        "## Corrected attention definitions",
        "",
        "For each node i and query hop q, normalized entropy is `H(A_iq)/log(K+1)`. Non-uniformity is `1 - mean_iq entropy`. Query-specific diversity is the within-node mean over unordered q<q' pairs of `TV(A_iq,A_iq') = 0.5 * sum_h |A_iqh-A_iq'h|`. Node heterogeneity is `TV(A_iq,Abar_q)` where `Abar_q=mean_i A_iq`. Diagonal mass is `mean_i mean_q A_iqq`; diagonal excess subtracts the uniform value `1/(K+1)`. The old mean-matrix-first entropy and row L1 are retained only as descriptive fields named `old_mean_matrix_*` and `mean_matrix_row_diversity`.",
        "",
        f"Across run/modality rows, corrected nodewise normalized entropy is {_mean(run_attention, 'attention_nodewise_normalized_entropy'):.6f}, while entropy of the mean matrix is {_mean(run_attention, 'attention_mean_matrix_normalized_entropy'):.6f}; mean Jensen gap is {_mean(run_attention, 'attention_jensen_entropy_gap'):.6f}. Mean node heterogeneity is {_mean(run_attention, 'attention_node_heterogeneity_mean'):.6f}, query diversity is {_mean(run_attention, 'attention_query_diversity_mean'):.6f}, mean-matrix row diversity is {_mean(run_attention, 'attention_mean_matrix_row_diversity'):.6f}, and diagonal excess is {_mean(run_attention, 'attention_diagonal_excess'):.6f}.",
        "",
        "Interpretation boundary: these metrics distinguish global, node-dependent, and query-dependent structure, but they are descriptive diagnostics rather than causal necessity evidence. A high old mean-matrix entropy can overestimate nodewise entropy by the Jensen gap; the corrected report does not use off-diagonal mass greater than 0.75 as a diagonal-collapse claim.",
        "",
        "## Semantic-reference correction",
        "",
        "The adaptive-range ratio is corrected to `(q90(alpha)-q10(alpha))/(abs(mean(alpha))+eps)`. The CSV retains alpha standard deviation, IQR, q90-q10, and the corrected ratio. Raw p is not treated as a pathology by its sign alone; the functional branch `term_p=rho_p*p` is reported with its distribution and p saturation fractions.",
        "",
        f"Mean term_p cross-seed Pearson/Spearman correlations are {_mean(term_p_corr, 'pearson'):.6f}/{_mean(term_p_corr, 'spearman'):.6f}; mean alpha cross-seed Pearson/Spearman correlations are {_mean(alpha_corr, 'pearson'):.6f}/{_mean(alpha_corr, 'spearman'):.6f}. Mean p saturation fractions are |p|>.90={_mean([r for r in semantic_rows if r.get('row_type') == 'run_hop'], 'p_fraction_abs_gt_0.9'):.6f}, |p|>.95={_mean([r for r in semantic_rows if r.get('row_type') == 'run_hop'], 'p_fraction_abs_gt_0.95'):.6f}, |p|>.99={_mean([r for r in semantic_rows if r.get('row_type') == 'run_hop'], 'p_fraction_abs_gt_0.99'):.6f}.",
        "",
        "Cross-seed correlations are computed only after matching dataset node count, feature bytes, edge-index bytes, and labels across seeds. Sign balance is reported descriptively; term_p is the quantity used for functional stability.",
        "",
        "## Context-change and interaction direction",
        "",
        f"Mean context-change injection ratio is {_mean(context_rows, 'context_injection_ratio_mean'):.6f}, with cosine to LayerNorm(S_k) {_mean(context_rows, 'context_injection_cosine_to_layernorm_S_mean'):.6f}. Mean interaction injection ratio is {_mean(context_rows, 'interaction_injection_ratio_mean'):.6f}, with cosine to S_k {_mean(context_rows, 'interaction_injection_cosine_to_S_mean'):.6f}; cosine(S_tilde_k,S_k) is {_mean(context_rows, 'S_tilde_cosine_to_S_mean'):.6f}. D norm and gate values are in `p15a_context_injection.csv`.",
        "",
        "## Signed filtering and automatic flags",
        "",
        f"Mean eta negative fraction is {_mean(filter_rows, 'negative_eta_fraction'):.6f}; mean gate delta/int are {_mean(context_rows, 'delta_gate'):.6f}/{_mean(context_rows, 'interaction_gate'):.6f}; mean D norm is {_mean(context_rows, 'D_norm_mean'):.6f}. Filter abs-means for delta_content/reference/relation are {_mean(filter_rows, 'delta_content_abs_mean'):.6f}/{_mean(filter_rows, 'reference_residual_abs_mean'):.6f}/{_mean(filter_rows, 'relation_filter_residual_abs_mean'):.6f}; eta std mean is {_mean(filter_rows, 'eta_std'):.6f}. Effective-order summaries are stored per hop row with node-level quantiles. Automatic flags are machine-generated from fixed diagnostic conditions and are not model verdicts."
        "",
        "### Flags",
        "",
    ]
    r1_section = [
        "## R1 - Relation modulation and operator perturbation",
        "",
        f"Mean beta is {_mean(relation_rows, 'beta'):.6f}; relation-weight mean/CV are {_mean(relation_rows, 'relation_weight_mean'):.6f}/{_mean(relation_rows, 'relation_weight_cv'):.6f}; relation residual abs-mean is {_mean(relation_rows, 'relation_residual_abs_mean'):.6f}; local adaptation c mean is {_mean(relation_rows, 'local_adaptation_mean'):.6f}.",
        f"After the same gcn_norm and edge-pair alignment against raw unit physical topology, operator MAE/RMSE/relative-L1 are {_mean(relation_rows, 'operator_weight_mae'):.6g}/{_mean(relation_rows, 'operator_weight_rmse'):.6g}/{_mean(relation_rows, 'operator_weight_relative_l1'):.6g}. These are normalized-operator perturbations, not raw scorer differences.",
        "",
    ]
    insert_at = lines.index("## Semantic-reference correction")
    lines[insert_at:insert_at] = r1_section
    for flag in summary["flags"]:
        lines.append(f"- {flag}")
    if not summary["flags"]:
        lines.append("- none")
    lines.extend([
        "",
        "## Corrected versus unchanged P1.5 conclusions",
        "",
        "Corrected: attention entropy, non-uniformity, query diversity, node heterogeneity, diagonal mass, and Jensen gap are now calculated before node aggregation, so the old mean-matrix-first entropy/row-diversity values are not used as node-level evidence. Adaptive-range ratio now has the specified alpha denominator. Functional semantic-reference evidence is based on term_p and cross-seed correlations, not raw p sign alone.",
        "",
        "Unchanged: the underlying frozen model, checkpoints, relation/operator quantities, alpha/d/p tensors, context injection tensors, signed eta tensors, and all historical P1.5 output files were not modified. No corrected diagnostic is presented as a validated architecture improvement or causal ablation result.",
        "",
        "## Boundary",
        "",
        "No training, formal benchmark rerun, ablation, LP experiment, hyperparameter search, test-based decision, loss change, or architecture change was performed.",
    ])
    (output / "p15a_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/ssi_mag_v3_p1_full"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssi_mag_v3_p15a_analysis"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    input_root = args.input_root if args.input_root.is_absolute() else project_root / args.input_root
    output = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    # _load_run follows the frozen P1 layout.  Refuse a different input root
    # rather than silently reading an unapproved checkpoint tree.
    expected_root = project_root / "outputs/ssi_mag_v3_p1_full"
    if input_root.resolve() != expected_root.resolve():
        raise ValueError(f"P1.5a accepts only the frozen P1 checkpoint root {expected_root}, got {input_root}")
    attention_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    attention_data: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    vectors: dict[tuple[str, int, str], dict[str, Any]] = {}
    node_signatures: dict[str, dict[int, tuple[Any, ...]]] = defaultdict(dict)
    errors = []
    loaded = 0
    finite_runs = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            try:
                run_dir, cfg, data, model = _load_run(project_root, dataset, seed, device)
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                with torch.no_grad():
                    normal = model.analysis(x, edge_index)
                finite = _finite_analysis(normal)
                loaded += 1
                finite_runs += int(finite)
                node_signatures[dataset][seed] = (data.num_nodes, _tensor_hash(data.x), _tensor_hash(data.edge_index), _tensor_hash(data.y) if data.y is not None else None)
                for modality in MODALITIES:
                    semantic_row, vector = _semantic_rows(normal, model, dataset, seed)
                    # The helper emits both modalities; keep only the selected vector here.
                    vectors[(dataset, seed, modality)] = vector[modality]
                attention_part, attention_run = _context_rows(normal, model, dataset, seed)
                relation_part = _relation_rows(normal, model, dataset, seed, data.num_nodes, x.dtype)
                relation_rows.extend(relation_part)
                semantic_part, _ = _semantic_rows(normal, model, dataset, seed)
                filter_part = _filter_rows(normal, dataset, seed)
                context_rows.extend(attention_part)
                # One attention row per run/modality, not one row per hop.
                for modality in MODALITIES:
                    diagnostics = attention_run[modality]["metrics"]
                    attention_rows.append({"dataset": dataset, "seed": seed, "modality": modality, "attention_shape": str(attention_run[modality]["shape"]), **{(key if key.startswith("attention_") else f"attention_{key}"): value for key, value in diagnostics.items()}})
                    attention_data[dataset].setdefault(modality, {})[str(seed)] = attention_run[modality]
                semantic_rows.extend(semantic_part)
                filter_rows.extend(filter_part)
                del normal, model, data, x, edge_index
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"OK {dataset} seed={seed} finite={finite}", flush=True)
            except Exception as exc:
                errors.append({"dataset": dataset, "seed": seed, "error": repr(exc)})
                print(f"ERROR {dataset} seed={seed}: {exc}", flush=True)
    ordering_verified = True
    ordering_errors = []
    for dataset, signatures in node_signatures.items():
        unique = {signature for signature in signatures.values()}
        if len(signatures) != len(args.seeds) or len(unique) != 1:
            ordering_verified = False
            ordering_errors.append({"dataset": dataset, "signatures": {str(seed): list(signature) for seed, signature in signatures.items()}})
    cross_seed = _cross_seed_rows(vectors, tuple(args.datasets), tuple(args.seeds)) if ordering_verified and not errors else []
    semantic_rows.extend(cross_seed)
    flags = []
    for row in relation_rows:
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}"
        if abs(row["beta"] - 0.05) < 0.005:
            flags.append(f"R1_beta_near_init:{tag}")
        if row["relation_weight_cv"] < 1.0e-4:
            flags.append(f"R1_relation_weight_cv_negligible:{tag}")
        if row["operator_weight_relative_l1"] < 1.0e-4:
            flags.append(f"R1_operator_perturbation_negligible:{tag}")
    for row in attention_rows:
        if "attention_nodewise_normalized_entropy" not in row:
            continue
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}"
        if abs(row["attention_nonuniformity"]) < 1.0e-3:
            flags.append(f"R3_attention_near_uniform:{tag}")
        if row["attention_diagonal_excess"] > 0.75 - 1.0e-3:
            flags.append(f"R3_attention_near_diagonal:{tag}")
    for row in semantic_rows:
        if row.get("row_type") != "run_hop":
            continue
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}/hop{row['hop']}"
        if row.get("alpha_std", math.nan) < 1.0e-5 or row.get("alpha_range_ratio", math.nan) < 1.0e-4:
            flags.append(f"R2_alpha_node_spread_negligible:{tag}")
        if row.get("alpha_fraction_lt_0.05", 0.0) > 0.5 or row.get("alpha_fraction_gt_0.95", 0.0) > 0.5:
            flags.append(f"R2_alpha_saturation:{tag}")
        if row.get("p_fraction_abs_gt_0.99", 0.0) > 0.5:
            flags.append(f"R2_p_saturation:{tag}")
    for row in context_rows:
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}/hop{row['hop']}"
        if abs(row["delta_gate"]) < 0.005:
            flags.append(f"R3_delta_gate_near_zero:{tag}")
        if abs(row["interaction_gate"]) < 0.005:
            flags.append(f"R3_interaction_gate_near_zero:{tag}")
    for row in filter_rows:
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}/hop{row['hop']}"
        if row.get("eta_std", math.nan) < 1.0e-5:
            flags.append(f"R3_eta_node_variance_negligible:{tag}")
        if np.isfinite(row.get("effective_order_std", math.nan)) and row["effective_order_std"] < 1.0e-5:
            flags.append(f"R3_effective_order_node_variance_negligible:{tag}")
        content_scale = max(row.get("delta_content_abs_mean", 0.0), row.get("gamma_global_abs_mean", 0.0), 1.0e-12)
        if row.get("reference_residual_abs_mean", 0.0) < 0.01 * content_scale:
            flags.append(f"R3_reference_residual_negligible:{tag}")
        if row.get("relation_filter_residual_abs_mean", 0.0) < 0.01 * content_scale:
            flags.append(f"R3_relation_residual_negligible:{tag}")
    if not ordering_verified:
        flags.append("cross_seed_node_ordering_unverified")
    summary = {
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "expected_runs": len(args.datasets) * len(args.seeds),
        "loaded_runs": loaded,
        "finite_runs": finite_runs,
        "errors": errors,
        "device": str(device),
        "input_root": str(input_root),
        "attention_shapes": sorted({row.get("attention_shape") for row in attention_rows if "attention_shape" in row}),
        "node_ordering_verified": ordering_verified,
        "node_ordering_errors": ordering_errors,
        "test_metrics_read": False,
        "training_invoked": False,
        "ablation_invoked": False,
        "lp_invoked": False,
        "flags": sorted(set(flags)),
        "output_files": ["p15a_relation_diagnostics.csv", "p15a_attention_diagnostics.csv", "p15a_semantic_reference_diagnostics.csv", "p15a_context_injection.csv", "p15a_filter_diagnostics.csv", "p15a_attention_matrices.json", "p15a_summary.json", "p15a_report.md"],
    }
    # Keep attention_rows to one row per run/modality.  The context rows have
    # the same attention diagnostics repeated by hop only in their own file.
    attention_summary_rows = [row for row in attention_rows if "attention_nodewise_normalized_entropy" in row]
    _write_csv(output / "p15a_relation_diagnostics.csv", relation_rows)
    _write_csv(output / "p15a_attention_diagnostics.csv", attention_summary_rows)
    _write_csv(output / "p15a_semantic_reference_diagnostics.csv", semantic_rows)
    _write_csv(output / "p15a_context_injection.csv", context_rows)
    _write_csv(output / "p15a_filter_diagnostics.csv", filter_rows)
    attention_json: dict[str, Any] = {}
    for dataset, modalities in attention_data.items():
        attention_json[dataset] = {}
        for modality, runs in modalities.items():
            matrices = [np.asarray(item["mean_matrix"], dtype=np.float64) for item in runs.values()]
            attention_json[dataset][modality] = {"runs": runs, "seed_mean": np.mean(matrices, axis=0).tolist() if matrices else [], "seed_std": np.std(matrices, axis=0).tolist() if matrices else []}
    (output / "p15a_attention_matrices.json").write_text(json.dumps(attention_json, indent=2, allow_nan=True, default=_json_default), encoding="utf-8")
    (output / "p15a_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True, default=_json_default), encoding="utf-8")
    _report(output, summary, relation_rows, attention_summary_rows, semantic_rows, context_rows, filter_rows)
    print(json.dumps(summary, indent=2), flush=True)
    if errors or loaded != summary["expected_runs"] or finite_runs != loaded or not ordering_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
