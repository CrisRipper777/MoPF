"""Analysis-only response variants for the frozen U2-B screening."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


VARIANTS = ("B0", "B1", "B2", "B3")
ALPHAS = (0.05, 0.1, 0.2)
EPS = 1e-12


def build_response_variant(
    h0: torch.Tensor,
    propagate_once: Callable[[torch.Tensor], torch.Tensor],
    *,
    max_hop: int,
    variant: str,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Build cumulative states and response coordinates without model mutation."""
    variant = str(variant).upper()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown U2-B variant: {variant}")
    if int(max_hop) < 0:
        raise ValueError(f"max_hop must be non-negative, got {max_hop}")
    if variant in {"B2", "B3"} and not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    states = [h0]
    for _ in range(int(max_hop)):
        previous = states[-1]
        propagated = propagate_once(previous)
        if variant in {"B2", "B3"}:
            current = (1.0 - float(alpha)) * propagated + float(alpha) * h0
        else:
            current = propagated
        states.append(current)

    if variant in {"B1", "B3"}:
        # The formal states remain untouched float32 tensors.  Innovation
        # coordinates are formed in float64 so the strict reconstruction audit
        # measures the algebraic identity rather than cancellation error from
        # repeatedly summing float32 differences.
        responses = [states[0].double()] + [
            states[index].double() - states[index - 1].double()
            for index in range(1, len(states))
        ]
    else:
        responses = list(states)
    reconstruction: list[float | None] = []
    if variant in {"B1", "B3"}:
        running = torch.zeros_like(h0, dtype=torch.float64)
        for state, response in zip(states, responses):
            running = running + response
            reconstruction.append(float((running - state.double()).abs().max().item()))
    else:
        reconstruction = [None for _ in states]
    return {
        "variant": variant,
        "alpha": float(alpha) if variant in {"B2", "B3"} else None,
        "states": states,
        "responses": responses,
        "reconstruction_error_by_hop": reconstruction,
        "max_reconstruction_error": max(value for value in reconstruction if value is not None)
        if any(value is not None for value in reconstruction)
        else None,
    }


def semantic_retention(
    states: list[torch.Tensor],
    h0: torch.Tensor,
    node_subset: torch.Tensor,
    *,
    computation_dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Measure modality-specific semantic retention relative to the ego state."""
    subset = node_subset.to(device=h0.device)
    reference = h0.index_select(0, subset)
    reference = reference.to(dtype=computation_dtype)
    reference_flat = reference.reshape(-1)
    reference_norm = reference_flat.norm()
    cka_values = []
    cosine_values = []
    for state in states:
        selected = state.index_select(0, subset).to(dtype=computation_dtype)
        first = selected - selected.mean(dim=0, keepdim=True)
        second = reference - reference.mean(dim=0, keepdim=True)
        cross = first.T @ second
        first_gram = first.T @ first
        second_gram = second.T @ second
        denominator = torch.sqrt(first_gram.square().sum() * second_gram.square().sum())
        cka = 0.0 if float(denominator.item()) <= EPS else float((cross.square().sum() / denominator).item())
        state_flat = selected.reshape(-1)
        cosine = float(
            (state_flat @ reference_flat)
            / (state_flat.norm() * reference_norm + EPS)
        )
        cka_values.append(float(np.clip(cka, -1.0, 1.0)))
        cosine_values.append(float(np.clip(cosine, -1.0, 1.0)))
    hops = np.arange(len(states), dtype=np.float64)
    slope_cka = float(np.polyfit(hops, np.asarray(cka_values), 1)[0]) if len(states) > 1 else 0.0
    slope_cosine = float(np.polyfit(hops, np.asarray(cosine_values), 1)[0]) if len(states) > 1 else 0.0
    return {
        "linear_cka_to_h0": cka_values,
        "frobenius_cosine_to_h0": cosine_values,
        "retention_ratio_to_hop0": {
            "linear_cka": [value / max(cka_values[0], EPS) for value in cka_values],
            "frobenius_cosine": [value / max(cosine_values[0], EPS) for value in cosine_values],
        },
        "retention_decay_slope": {
            "linear_cka": slope_cka,
            "frobenius_cosine": slope_cosine,
        },
    }


def response_magnitude_audit(
    responses: list[torch.Tensor],
    h0: torch.Tensor,
    *,
    innovation_channel: bool,
    vanishing_threshold: float = 1e-4,
) -> dict[str, Any]:
    norms = [float(response.norm().item()) for response in responses]
    h0_norm = max(float(h0.norm().item()), EPS)
    to_h0 = [value / h0_norm for value in norms]
    to_previous = [1.0] + [norms[index] / max(norms[index - 1], EPS) for index in range(1, len(norms))]
    return {
        "norm": norms,
        "norm_ratio_to_h0": to_h0,
        "norm_ratio_to_previous": to_previous,
        "innovation_channel": bool(innovation_channel),
        "vanishing_threshold": float(vanishing_threshold),
        "vanishing_by_hop": [bool(innovation_channel and value < vanishing_threshold) for value in to_h0],
        "vanishing_innovation_channel": bool(innovation_channel and any(value < vanishing_threshold for value in to_h0[1:])),
    }
