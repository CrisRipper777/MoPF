"""Analysis-only algebra and metric helpers for the U2-C0 audit.

The helpers in this module do not alter the formal MoPF path.  They make the
state/response split, the anchored differential identities, and the exact
coefficient-coordinate transform explicit and testable in CPU float64.
"""

from __future__ import annotations

from typing import Callable

import torch

from src.models.mopf import (
    anchored_differential_to_monomial_coefficients,
    monomial_to_anchored_differential_coefficients,
)


EPS = 1e-15


def ordinary_monomial_bank(
    h0: torch.Tensor,
    propagate: Callable[[torch.Tensor], torch.Tensor],
    max_order: int,
) -> list[torch.Tensor]:
    """Return ``M_k=P^k H`` without response normalization."""
    bank = [h0]
    current = h0
    for _ in range(int(max_order)):
        current = propagate(current)
        bank.append(current)
    return bank


def ordinary_differential_bank(
    h0: torch.Tensor,
    propagate: Callable[[torch.Tensor], torch.Tensor],
    max_order: int,
) -> list[torch.Tensor]:
    """Return ``D_0=M_0`` and ordinary inter-hop differences."""
    monomial = ordinary_monomial_bank(h0, propagate, max_order)
    return [monomial[0]] + [monomial[k] - monomial[k - 1] for k in range(1, len(monomial))]


def anchored_differential_banks(
    h0: torch.Tensor,
    propagate: Callable[[torch.Tensor], torch.Tensor],
    max_order: int,
    alpha: float,
) -> dict[str, list[torch.Tensor]]:
    """Return anchored cumulative states and their differential responses."""
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError(f"alpha must satisfy 0 <= alpha < 1, got {alpha}")
    states = [h0]
    current = h0
    for _ in range(int(max_order)):
        current = (1.0 - float(alpha)) * propagate(current) + float(alpha) * h0
        states.append(current)
    responses = [states[0]] + [states[k] - states[k - 1] for k in range(1, len(states))]
    return {"states": states, "responses": responses}


def anchored_closed_form_responses(
    h0: torch.Tensor,
    propagate: Callable[[torch.Tensor], torch.Tensor],
    max_order: int,
    alpha: float,
) -> list[torch.Tensor]:
    """Evaluate ``q^k P^(k-1)(P-I)H`` by recurrence, not matrix powers."""
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError(f"alpha must satisfy 0 <= alpha < 1, got {alpha}")
    q = 1.0 - float(alpha)
    first_difference = propagate(h0) - h0
    responses = [h0]
    current = first_difference
    for order in range(1, int(max_order) + 1):
        responses.append((q**order) * current)
        current = propagate(current)
    return responses


def _stack_sum(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(values, dim=0).sum(dim=0)


def relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = (first - second).norm()
    denominator = second.norm().clamp_min(EPS)
    return float((numerator / denominator).item())


def max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).abs().max().item())


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = (first.reshape(-1) * second.reshape(-1)).sum()
    denominator = first.norm() * second.norm()
    if float(denominator.item()) <= EPS:
        return 0.0
    return float((numerator / denominator).clamp(-1.0, 1.0).item())


def algebra_audit(
    h0: torch.Tensor,
    propagate: Callable[[torch.Tensor], torch.Tensor],
    max_order: int,
    alpha: float,
) -> dict[str, object]:
    """Audit reconstruction, closed form, scaling, and alpha-zero limits."""
    anchored = anchored_differential_banks(h0, propagate, max_order, alpha)
    ordinary = ordinary_differential_bank(h0, propagate, max_order)
    closed = anchored_closed_form_responses(h0, propagate, max_order, alpha)
    q = 1.0 - float(alpha)
    reconstruction = []
    closed_form = []
    scaling = []
    for order in range(int(max_order) + 1):
        reconstructed = _stack_sum(anchored["responses"][: order + 1])
        reconstruction.append(
            {"order": order, "max_abs": max_abs(reconstructed, anchored["states"][order]),
             "relative_l2": relative_l2(reconstructed, anchored["states"][order])}
        )
        closed_form.append(
            {"order": order, "max_abs": max_abs(anchored["responses"][order], closed[order]),
             "relative_l2": relative_l2(anchored["responses"][order], closed[order])}
        )
        if order == 0:
            scaling.append({"order": order, "max_abs": 0.0, "relative_l2": 0.0})
        else:
            expected = (q**order) * ordinary[order]
            scaling.append(
                {"order": order, "max_abs": max_abs(anchored["responses"][order], expected),
                 "relative_l2": relative_l2(anchored["responses"][order], expected)}
            )
    alpha_zero = anchored_differential_banks(h0, propagate, max_order, 0.0)
    alpha_zero_limit = [
        {"order": order, "max_abs": max_abs(alpha_zero["responses"][order], ordinary[order]),
         "relative_l2": relative_l2(alpha_zero["responses"][order], ordinary[order])}
        for order in range(int(max_order) + 1)
    ]
    return {
        "state_reconstruction": reconstruction,
        "closed_form_identity": closed_form,
        "b3_b1_scaling_identity": scaling,
        "alpha_zero_limit": alpha_zero_limit,
        "max_state_reconstruction_abs": max(item["max_abs"] for item in reconstruction),
        "max_state_reconstruction_relative_l2": max(item["relative_l2"] for item in reconstruction),
        "max_closed_form_abs": max(item["max_abs"] for item in closed_form),
        "max_closed_form_relative_l2": max(item["relative_l2"] for item in closed_form),
        "max_scaling_abs": max(item["max_abs"] for item in scaling),
        "max_scaling_relative_l2": max(item["relative_l2"] for item in scaling),
        "max_alpha_zero_abs": max(item["max_abs"] for item in alpha_zero_limit),
        "max_alpha_zero_relative_l2": max(item["relative_l2"] for item in alpha_zero_limit),
    }


def pairwise_cosine_matrix(
    responses: list[torch.Tensor],
    *,
    absolute: bool = False,
    squared: bool = False,
) -> torch.Tensor:
    """Return signed, absolute, or squared Frobenius cosine matrices."""
    flattened = torch.stack([response.reshape(-1).double() for response in responses])
    norms = flattened.norm(dim=1).clamp_min(EPS)
    matrix = (flattened @ flattened.T) / (norms[:, None] * norms[None, :])
    matrix = matrix.clamp(-1.0, 1.0)
    if absolute:
        matrix = matrix.abs()
    if squared:
        matrix = matrix.square()
    return matrix


def mean_off_diagonal(matrix: torch.Tensor) -> float:
    count = matrix.size(0)
    if count < 2:
        return 0.0
    mask = ~torch.eye(count, dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].mean().item())


def response_magnitude(responses: list[torch.Tensor], h0: torch.Tensor) -> dict[str, list[float]]:
    h0_norm = h0.norm().clamp_min(EPS)
    norms = [float(value.norm().item()) for value in responses]
    return {
        "norm": norms,
        "norm_ratio_to_h0": [float(value / h0_norm.item()) for value in norms],
    }


def coefficient_transform_audit(
    eta: torch.Tensor,
    alpha: float,
) -> dict[str, float | list[float]]:
    """Report forward/inverse coordinate-transform errors in float64."""
    eta = eta.detach().double()
    differential = monomial_to_anchored_differential_coefficients(eta, alpha)
    reconstructed = anchored_differential_to_monomial_coefficients(differential, alpha)
    return {
        "alpha": float(alpha),
        "q": 1.0 - float(alpha),
        "max_abs_round_trip": max_abs(reconstructed, eta),
        "relative_l2_round_trip": relative_l2(reconstructed, eta),
        "cosine_round_trip": cosine(reconstructed, eta),
        "q_inverse_by_order": [
            float((1.0 - float(alpha)) ** (-order))
            for order in range(int(eta.size(-1)))
        ],
        "differential": differential,
        "reconstructed": reconstructed,
    }

