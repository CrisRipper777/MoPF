"""Analysis-only diagnostics for the frozen U2-A monomial response bank.

This module deliberately does not alter the formal MoPF forward path.  It
reconstructs the frozen projection/conductance/GCN-normalization path, keeps
the unnormalised response bank, and computes diagnostics on detached response
representations.  The small formal-forward capture helper is used only for
equivalence auditing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


EPS = 1e-12
NUMERICAL_TOLERANCE = 1e-8
MAX_CKA_NODES = 20_000
DEFAULT_NODE_SAMPLE_SEED = 20260911


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), EPS))


def _clone_bank(bank: list[torch.Tensor]) -> list[torch.Tensor]:
    return [value.detach().clone() for value in bank]


def _propagation_bank(
    model: torch.nn.Module,
    h0: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> list[torch.Tensor]:
    """Reconstruct H_0,...,H_K with the frozen recurrence, without normalising H."""
    bank = [h0]
    current = h0
    for _ in range(int(model.max_order)):
        current = model._propagate_once(current, edge_index, edge_weight)
        bank.append(current)
    return bank


@torch.no_grad()
def extract_analysis_response_bank(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor | None,
) -> dict[str, Any]:
    """Extract P_text/P_visual and the raw monomial response bank.

    The projections, semantic conductance, GCN normalization, and recurrence
    are the frozen model operations.  No response normalization or filter,
    node conditioner, refinement, fusion, or classifier is applied here.
    """
    was_training = model.training
    model.eval()
    edge_index = model._edge_index_or_empty(edge_index, x.device)
    x_text, x_visual = model._split_features(x)
    h_text = model.text_proj(x_text)
    h_visual = model.visual_proj(x_visual)
    edges = model._semantic_edge_weights(h_text, h_visual, edge_index)
    norm_t_index, norm_t_weight = model._normalized_operator(
        edge_index, edges["w_t"], int(x.size(0)), h_text.dtype
    )
    norm_v_index, norm_v_weight = model._normalized_operator(
        edge_index, edges["w_v"], int(x.size(0)), h_visual.dtype
    )
    bank_text = _propagation_bank(model, h_text, norm_t_index, norm_t_weight)
    bank_visual = _propagation_bank(model, h_visual, norm_v_index, norm_v_weight)
    if was_training:
        model.train()
    return {
        "h_text": h_text,
        "h_visual": h_visual,
        "edges": edges,
        "norm_text_index": norm_t_index,
        "norm_text_weight": norm_t_weight,
        "norm_visual_index": norm_v_index,
        "norm_visual_weight": norm_v_weight,
        "banks": {"text": bank_text, "visual": bank_visual},
        "source_edge_index": edge_index,
    }


@torch.no_grad()
def capture_formal_forward_banks(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor | None,
) -> dict[str, Any]:
    """Capture the two banks actually returned by the formal ``forward`` path."""
    captured: list[list[torch.Tensor]] = []
    operators: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = model._propagation_bank

    def hooked(
        h0: torch.Tensor,
        norm_edge_index: torch.Tensor,
        norm_edge_weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        bank = original(h0, norm_edge_index, norm_edge_weight)
        captured.append(_clone_bank(bank))
        operators.append((norm_edge_index.detach().clone(), norm_edge_weight.detach().clone()))
        return bank

    object.__setattr__(model, "_propagation_bank", hooked)
    try:
        was_training = model.training
        model.eval()
        model(x, edge_index)
        if was_training:
            model.train()
    finally:
        object.__setattr__(model, "_propagation_bank", original)
    if len(captured) != 2:
        raise RuntimeError(f"Expected two formal modality banks, captured {len(captured)}")
    return {"text": captured[0], "visual": captured[1], "operators": operators}


def bank_equivalence(
    analysis_bank: dict[str, list[torch.Tensor]],
    formal_bank: dict[str, list[torch.Tensor]],
) -> dict[str, Any]:
    differences: dict[str, list[float]] = {}
    maximum = 0.0
    for modality in ("text", "visual"):
        modality_diffs = []
        for analysis, formal in zip(analysis_bank[modality], formal_bank[modality]):
            diff = float((analysis - formal).abs().max().item())
            modality_diffs.append(diff)
            maximum = max(maximum, diff)
        differences[modality] = modality_diffs
    return {
        "max_abs_difference": float(maximum),
        "tolerance": 1e-6,
        "passed": bool(maximum < 1e-6),
        "per_modality_per_order": differences,
    }


def deterministic_node_subset(
    num_nodes: int,
    *,
    max_nodes: int = MAX_CKA_NODES,
    seed: int = DEFAULT_NODE_SAMPLE_SEED,
) -> torch.Tensor:
    count = min(int(num_nodes), int(max_nodes))
    if count == int(num_nodes):
        return torch.arange(int(num_nodes), dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(int(num_nodes), generator=generator)[:count].sort().values


def frobenius_cosine_matrix(responses: list[torch.Tensor]) -> torch.Tensor:
    flattened = torch.stack([response.reshape(-1).double() for response in responses], dim=0)
    norms = flattened.norm(dim=1)
    matrix = flattened @ flattened.T
    return torch.clamp(matrix / (norms[:, None] * norms[None, :] + EPS), -1.0, 1.0)


def _linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    cross = first.T @ second
    first_gram = first.T @ first
    second_gram = second.T @ second
    numerator = cross.square().sum()
    denominator = torch.sqrt(first_gram.square().sum() * second_gram.square().sum())
    if float(denominator.item()) <= EPS:
        return 0.0
    return float((numerator / denominator).item())


def linear_cka_matrix(
    responses: list[torch.Tensor],
    node_subset: torch.Tensor,
) -> torch.Tensor:
    subset = node_subset.to(device=responses[0].device)
    selected = [response.index_select(0, subset) for response in responses]
    count = len(selected)
    matrix = torch.zeros((count, count), dtype=torch.float64, device=responses[0].device)
    for i in range(count):
        for j in range(i, count):
            value = _linear_cka(selected[i].double(), selected[j].double())
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _gram_spectrum(gram: torch.Tensor) -> dict[str, Any]:
    raw = np.linalg.eigvalsh(gram.detach().cpu().numpy().astype(np.float64))
    scale = max(1.0, float(np.max(np.abs(raw))))
    tolerance = NUMERICAL_TOLERANCE * scale
    clipped = raw.copy()
    clip_magnitude = 0.0
    for index, value in enumerate(clipped):
        if value < 0.0 and abs(value) <= tolerance:
            clip_magnitude = max(clip_magnitude, abs(float(value)))
            clipped[index] = 0.0
    positive = clipped[clipped > EPS]
    lambda_max = float(np.max(clipped)) if clipped.size else 0.0
    lambda_min_positive = float(np.min(positive)) if positive.size else None
    condition = (
        None
        if lambda_min_positive is None
        else float(lambda_max / max(lambda_min_positive, EPS))
    )
    positive_mass = float(positive.sum())
    if positive_mass <= EPS:
        effective_rank = 0.0
    else:
        probabilities = positive / positive_mass
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + EPS))))
    return {
        "eigenvalues": raw.tolist(),
        "eigenvalues_clipped": clipped.tolist(),
        "clip_tolerance": float(tolerance),
        "clip_magnitude": float(clip_magnitude),
        "psd_within_numerical_tolerance": bool(np.min(raw) >= -tolerance),
        "lambda_max": lambda_max,
        "lambda_min_positive": lambda_min_positive,
        "condition_number": condition,
        "effective_rank": float(effective_rank),
        "normalized_effective_rank": float(effective_rank / len(raw)) if len(raw) else None,
    }


def normalized_response_gram(responses: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
    normalized = []
    for response in responses:
        response = response.double()
        normalized.append(response / (response.norm() + EPS))
    gram = torch.stack([value.reshape(-1) for value in normalized], dim=0)
    gram = gram @ gram.T
    return gram, _gram_spectrum(gram)


def incremental_novelty(responses: list[torch.Tensor]) -> dict[str, Any]:
    basis: list[torch.Tensor] = []
    novelty: list[float] = [1.0]
    explained: list[float] = [0.0]
    for index, response in enumerate(responses[1:], start=1):
        residual = response.reshape(-1).double().clone()
        for vector in basis:
            residual = residual - torch.dot(vector, residual) * vector
        response_norm = response.double().norm()
        novelty_value = float((residual.norm() / (response_norm + EPS)).item())
        novelty_value = max(0.0, novelty_value)
        novelty.append(novelty_value)
        explained.append(float(max(0.0, 1.0 - novelty_value * novelty_value)))
        residual_norm = residual.norm()
        if float(residual_norm.item()) > EPS:
            basis.append(residual / residual_norm)
    return {
        "novelty_ratio": novelty,
        "projection_explained_ratio": explained,
        "basis_rank": len(basis),
    }


def _pair_payload(matrix: torch.Tensor) -> dict[str, Any]:
    values = matrix.detach().cpu().numpy().astype(np.float64)
    order_count = values.shape[0]
    adjacent = {
        f"{index}_{index + 1}": float(values[index, index + 1])
        for index in range(order_count - 1)
    }
    high_order = None if order_count < 2 else float(values[-2, -1])
    return {
        "matrix": values.tolist(),
        "adjacent": adjacent,
        "high_order": high_order,
    }


def _coalesced_entries(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    values = edge_weight.detach().cpu().numpy().astype(np.float64, copy=False)
    keys = dst * int(num_nodes) + src
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_values = values[order]
    if sorted_keys.size == 0:
        return sorted_keys, sorted_values
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    return sorted_keys[starts], np.add.reduceat(sorted_values, starts)


def _lookup(sorted_keys: np.ndarray, sorted_values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    result = np.zeros(keys.shape, dtype=np.float64)
    if sorted_keys.size == 0 or keys.size == 0:
        return result
    positions = np.searchsorted(sorted_keys, keys)
    valid = positions < sorted_keys.size
    valid &= sorted_keys[np.minimum(positions, sorted_keys.size - 1)] == keys
    result[valid] = sorted_values[positions[valid]]
    return result


def sparse_symmetry_audit(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> dict[str, Any]:
    """Audit P-P^T using coalesced sparse entries, never an N x N tensor."""
    keys, values = _coalesced_entries(edge_index, edge_weight, num_nodes)
    transpose_keys = (keys % int(num_nodes)) * int(num_nodes) + (keys // int(num_nodes))
    union = np.union1d(keys, transpose_keys)
    forward = _lookup(keys, values, union)
    reverse = _lookup(keys, values, (union % int(num_nodes)) * int(num_nodes) + (union // int(num_nodes)))
    difference_norm = float(np.linalg.norm(forward - reverse))
    operator_norm = float(np.linalg.norm(values))
    ratio = difference_norm / max(operator_norm, EPS)
    if ratio <= 1e-5:
        classification = "near_symmetric"
    elif ratio <= 1e-3:
        classification = "approximately_symmetric"
    else:
        classification = "materially_asymmetric"
    return {
        "asymmetry_ratio": float(ratio),
        "difference_frobenius_norm": difference_norm,
        "operator_frobenius_norm": operator_norm,
        "classification": classification,
        "nnz_coalesced": int(keys.size),
    }


def _scipy_sparse_operator(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
):
    import scipy.sparse

    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    values = edge_weight.detach().cpu().numpy().astype(np.float64, copy=False)
    return scipy.sparse.csr_matrix((values, (dst, src)), shape=(num_nodes, num_nodes))


def _rayleigh_power(operator, *, sign: float, seed: int, iterations: int = 96) -> float:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(operator.shape[0]).astype(np.float64)
    vector /= max(np.linalg.norm(vector), EPS)
    estimate = 0.0
    for _ in range(iterations):
        updated = sign * operator.dot(vector)
        norm = np.linalg.norm(updated)
        if norm <= EPS:
            return 0.0
        vector = updated / norm
        estimate = float(vector @ operator.dot(vector))
    return estimate


def spectral_range_audit(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    symmetry: dict[str, Any],
    *,
    seed: int = DEFAULT_NODE_SAMPLE_SEED,
) -> dict[str, Any]:
    """Estimate the symmetric spectral range only when the audit permits it."""
    if float(symmetry["asymmetry_ratio"]) > 1e-3:
        return {
            "status": "skipped_due_to_operator_asymmetry",
            "spectral_interpretation_status": "requires_caution",
            "lambda_min": None,
            "lambda_max": None,
            "max_abs_eigenvalue": None,
        }
    operator = _scipy_sparse_operator(edge_index, edge_weight, int(num_nodes))
    symmetric_operator = (operator + operator.T) * 0.5
    try:
        from scipy.sparse.linalg import eigsh

        if int(num_nodes) > 2:
            vector = np.random.default_rng(seed).standard_normal(int(num_nodes))
            lambda_max = float(
                eigsh(symmetric_operator, k=1, which="LA", v0=vector, tol=1e-7, maxiter=500)[0][0]
            )
            lambda_min = float(
                eigsh(symmetric_operator, k=1, which="SA", v0=vector, tol=1e-7, maxiter=500)[0][0]
            )
            return {
                "status": "estimated_eigsh_symmetric_part",
                "spectral_interpretation_status": "supported_with_symmetry_audit",
                "lambda_min": lambda_min,
                "lambda_max": lambda_max,
                "max_abs_eigenvalue": max(abs(lambda_min), abs(lambda_max)),
                "operator_used": "(P+P^T)/2",
            }
    except Exception as error:  # pragma: no cover - depends on scipy convergence
        failure = f"{type(error).__name__}: {error}"
    else:
        failure = "eigsh_requires_num_nodes_gt_2"
    lambda_max = _rayleigh_power(symmetric_operator, sign=1.0, seed=seed)
    lambda_min = _rayleigh_power(symmetric_operator, sign=-1.0, seed=seed + 1)
    return {
        "status": "approximate_rayleigh_power",
        "spectral_interpretation_status": "supported_with_approximate_range",
        "lambda_min": float(lambda_min),
        "lambda_max": float(lambda_max),
        "max_abs_eigenvalue": max(abs(lambda_min), abs(lambda_max)),
        "operator_used": "(P+P^T)/2",
        "eigsh_failure": failure,
    }


def _evolution_payload(
    responses: list[torch.Tensor],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    dirichlet_allowed: bool,
) -> dict[str, Any]:
    norms = [float(response.norm().item()) for response in responses]
    variances = [float(response.var(dim=0, unbiased=False).mean().item()) for response in responses]
    structural_variations = []
    dirichlet = []
    for response in responses:
        propagated = _propagate_response(response, edge_index, edge_weight)
        structural_variations.append(_ratio(float((response - propagated).norm().item()), float(response.norm().item())))
        if dirichlet_allowed:
            energy = float((response * (response - propagated)).sum().item())
            dirichlet.append(energy)
    norm_ratio = [_ratio(value, norms[0]) for value in norms]
    consecutive_norm_ratio = [1.0] + [_ratio(norms[index], norms[index - 1]) for index in range(1, len(norms))]
    variance_ratio = [_ratio(value, variances[0]) for value in variances]
    sv_ratio = [_ratio(value, structural_variations[0]) for value in structural_variations]
    payload: dict[str, Any] = {
        "norm": norms,
        "norm_ratio_to_order_0": norm_ratio,
        "norm_ratio_to_previous": consecutive_norm_ratio,
        "node_feature_variance": variances,
        "node_feature_variance_ratio_to_order_0": variance_ratio,
        "structural_variation": structural_variations,
        "structural_variation_ratio_to_order_0": sv_ratio,
    }
    if dirichlet_allowed:
        squared_norms = [max(value * value, EPS) for value in norms]
        normalized = [_ratio(value, denominator) for value, denominator in zip(dirichlet, squared_norms)]
        payload["dirichlet_energy"] = dirichlet
        payload["dirichlet_energy_normalized"] = normalized
        payload["dirichlet_energy_normalized_ratio_to_order_0"] = [_ratio(value, normalized[0]) for value in normalized]
    else:
        payload["dirichlet_energy_status"] = "skipped_due_to_operator_asymmetry"
    return payload


def _propagate_response(response: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros_like(response)
    src, dst = edge_index
    output = torch.zeros_like(response)
    output.index_add_(0, dst, response[src] * edge_weight.unsqueeze(-1))
    return output


def modality_response_diagnostics(
    responses: list[torch.Tensor],
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    node_subset: torch.Tensor,
    operator_audit: dict[str, Any],
    spectral_audit: dict[str, Any],
) -> dict[str, Any]:
    cosine = frobenius_cosine_matrix(responses)
    cka = linear_cka_matrix(responses, node_subset)
    gram, spectrum = normalized_response_gram(responses)
    novelty = incremental_novelty(responses)
    evolution = _evolution_payload(
        responses,
        edge_index,
        edge_weight,
        dirichlet_allowed=float(operator_audit["asymmetry_ratio"]) <= 1e-3,
    )
    return {
        "pairwise_frobenius_cosine": _pair_payload(cosine),
        "linear_cka": _pair_payload(cka),
        "normalized_response_gram": gram.detach().cpu().numpy().astype(np.float64).tolist(),
        "gram_spectrum": spectrum,
        "incremental_structural_novelty": novelty,
        "response_evolution": evolution,
        "operator_audit": operator_audit,
        "spectral_audit": spectral_audit,
    }
