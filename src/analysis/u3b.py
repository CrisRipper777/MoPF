"""Pure analysis helpers for U3-B Transport-Conditioned Preference Residual.

The functions in this module intentionally do not import the MoPF model.  They
make the U3-A conductance definition and the TCPR centering/identifiability
rules reusable by preflight, training-time diagnostics, and tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


EPS = 1e-12


def mean_incident_conductance(
    edge_index: torch.Tensor,
    conductance: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return U3-A's mean incident conductance and raw incident degree.

    This is deliberately the statistic used by U3-A: conductance is summed
    once for each endpoint of every edge in the original physical support.
    It does not use normalized operator weights, cosine values, or a degree
    normalization.  Isolated nodes receive the graph-level edge-conductance
    mean, as required by U3-B.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}")
    if conductance.ndim != 1 or conductance.numel() != edge_index.size(1):
        raise ValueError(
            "conductance must have shape [E] matching edge_index, got "
            f"{tuple(conductance.shape)} for E={edge_index.size(1)}"
        )
    if num_nodes < 0:
        raise ValueError(f"num_nodes must be non-negative, got {num_nodes}")

    edge_index = edge_index.to(device=conductance.device, dtype=torch.long)
    conductance = conductance.to(dtype=torch.float32)
    degree = torch.zeros(num_nodes, dtype=conductance.dtype, device=conductance.device)
    incident_sum = torch.zeros_like(degree)
    for endpoint in (edge_index[0], edge_index[1]):
        degree.index_add_(0, endpoint, torch.ones_like(conductance))
        incident_sum.index_add_(0, endpoint, conductance)
    mean_conductance = incident_sum / degree.clamp_min(1.0)
    non_isolated = degree > 0.0
    if bool(non_isolated.any()):
        graph_mean = mean_conductance[non_isolated].mean()
    elif conductance.numel():
        graph_mean = conductance.mean()
    else:
        graph_mean = torch.zeros((), dtype=conductance.dtype, device=conductance.device)
    # The fallback is the graph-level mean over non-isolated nodes.  Once it
    # is assigned, the full node mean equals that same value, so the centered
    # context of every isolated node is exactly zero.
    mean_conductance = torch.where(non_isolated, mean_conductance, graph_mean)
    return mean_conductance, degree


def centered_transport_context(
    edge_index: torch.Tensor,
    conductance: torch.Tensor,
    num_nodes: int,
) -> dict[str, torch.Tensor | float]:
    """Build the centered node-level transport context used by TCPR."""
    mean_conductance, degree = mean_incident_conductance(
        edge_index, conductance, num_nodes
    )
    centered = mean_conductance - mean_conductance.mean()
    return {
        "mean_conductance": mean_conductance,
        "centered": centered,
        "degree": degree,
        "centered_mean": float(centered.mean().item()) if centered.numel() else 0.0,
        "graph_mean_conductance": float(mean_conductance.mean().item()) if mean_conductance.numel() else 0.0,
    }


def order_centered_profile(theta: torch.Tensor) -> torch.Tensor:
    """Return beta = theta - mean_order(theta), preserving shape/device."""
    if theta.ndim != 1 or theta.numel() < 1:
        raise ValueError(f"theta must have shape [K+1], got {tuple(theta.shape)}")
    return theta - theta.mean()


def transport_residual(
    centered_context: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return beta and tau for a scalar node/order TCPR interaction."""
    if centered_context.ndim != 1:
        raise ValueError("centered_context must have shape [N]")
    beta = order_centered_profile(theta)
    return beta, centered_context.unsqueeze(-1) * beta.unsqueeze(0)


def shuffle_node_scalar(context: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministically shuffle a complete node-level scalar context."""
    if context.ndim != 1:
        raise ValueError("context must have shape [N]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(context.size(0), generator=generator).to(context.device)
    return context.index_select(0, permutation)


def _safe_pearson(first: np.ndarray, second: np.ndarray) -> tuple[float, bool]:
    if first.size < 3 or second.size != first.size:
        return 0.0, True
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return 0.0, True
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= EPS:
        return 0.0, True
    return float(np.dot(first_centered, second_centered) / denominator), False


def partial_spearman(
    outcome: torch.Tensor | np.ndarray,
    predictor: torch.Tensor | np.ndarray,
    control: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """Compute raw and degree-controlled Spearman association.

    The partial statistic follows the frozen protocol exactly: average-rank
    transform each variable, linearly residualize outcome and predictor on an
    intercept plus the ranked control, then Pearson-correlate the residual
    ranks.  No labels are involved.
    """
    arrays = [
        np.asarray(value.detach().cpu() if isinstance(value, torch.Tensor) else value, dtype=np.float64).reshape(-1)
        for value in (outcome, predictor, control)
    ]
    if not (len(arrays[0]) == len(arrays[1]) == len(arrays[2])):
        raise ValueError("outcome, predictor, and control must have equal length")
    mask = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    outcome_np, predictor_np, control_np = [value[mask] for value in arrays]
    sample_count = int(mask.sum())
    if sample_count < 3:
        return {
            "raw_rho": 0.0,
            "raw_p_value": None,
            "partial_rho": 0.0,
            "sample_count": sample_count,
            "degenerate": True,
            "raw_sign": 0,
            "partial_sign": 0,
        }

    raw = spearmanr(outcome_np, predictor_np)
    ranked_outcome = rankdata(outcome_np, method="average")
    ranked_predictor = rankdata(predictor_np, method="average")
    ranked_control = rankdata(control_np, method="average")
    design = np.column_stack((np.ones(sample_count), ranked_control))
    outcome_residual = ranked_outcome - design @ np.linalg.lstsq(
        design, ranked_outcome, rcond=None
    )[0]
    predictor_residual = ranked_predictor - design @ np.linalg.lstsq(
        design, ranked_predictor, rcond=None
    )[0]
    partial, degenerate = _safe_pearson(outcome_residual, predictor_residual)
    return {
        "raw_rho": float(raw.statistic) if np.isfinite(raw.statistic) else 0.0,
        "raw_p_value": float(raw.pvalue) if np.isfinite(raw.pvalue) else None,
        "partial_rho": partial,
        "sample_count": sample_count,
        "degenerate": bool(degenerate),
        "raw_sign": int(np.sign(raw.statistic)) if np.isfinite(raw.statistic) else 0,
        "partial_sign": int(np.sign(partial)),
    }


def association_gate(
    rows: list[dict[str, Any]],
    *,
    min_overall_sign_fraction: float = 2.0 / 3.0,
    min_dataset_modality_fraction: float = 0.7,
    min_median_abs_partial_rho: float = 0.05,
) -> dict[str, Any]:
    """Apply a sign/effect/dataset-consistency preflight gate.

    The gate intentionally reports p-values but does not use them.  A
    dataset/modality group is direction-consistent when at least two of its
    three seeds share the same nonzero partial-rho sign.
    """
    valid = [row for row in rows if not row.get("degenerate", True)]
    signs = [int(row.get("partial_sign", 0)) for row in valid if int(row.get("partial_sign", 0))]
    positive = sum(sign > 0 for sign in signs)
    negative = sum(sign < 0 for sign in signs)
    overall_fraction = max(positive, negative) / max(len(signs), 1)
    median_abs = float(np.median([abs(float(row["partial_rho"])) for row in valid])) if valid else 0.0

    grouped: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["modality"])), []).append(int(row.get("partial_sign", 0)))
    consistent_groups = 0
    group_payload = []
    for key in sorted(grouped):
        group_signs = [sign for sign in grouped[key] if sign]
        pos = sum(sign > 0 for sign in group_signs)
        neg = sum(sign < 0 for sign in group_signs)
        direction_count = max(pos, neg)
        consistent = direction_count >= 2 and len(group_signs) >= 2
        consistent_groups += int(consistent)
        group_payload.append({
            "dataset": key[0],
            "modality": key[1],
            "partial_signs": group_signs,
            "dominant_sign": 1 if pos >= neg and pos else (-1 if neg else 0),
            "consistent": bool(consistent),
        })
    group_fraction = consistent_groups / max(len(grouped), 1)
    passed = bool(
        valid
        and overall_fraction >= min_overall_sign_fraction
        and group_fraction >= min_dataset_modality_fraction
        and median_abs >= min_median_abs_partial_rho
    )
    return {
        "passed": passed,
        "valid_count": len(valid),
        "positive_count": positive,
        "negative_count": negative,
        "overall_dominant_sign_fraction": overall_fraction,
        "median_abs_partial_rho": median_abs,
        "consistent_dataset_modality_groups": consistent_groups,
        "dataset_modality_group_count": len(grouped),
        "dataset_modality_consistency_fraction": group_fraction,
        "thresholds": {
            "min_overall_sign_fraction": min_overall_sign_fraction,
            "min_dataset_modality_fraction": min_dataset_modality_fraction,
            "min_median_abs_partial_rho": min_median_abs_partial_rho,
        },
        "groups": group_payload,
        "uses_p_values_for_gate": False,
    }
