"""Pure analysis helpers for U3-A node/modality preference diagnosis.

Nothing in this module is imported by the formal training or inference path.
All attribution is expressed in terms of final effective eta coefficients.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


EPS = 1e-12


def canonical_effective_decomposition(
    eta_by_modality: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Decompose final eta into global, modality, and centered-node terms."""
    if set(eta_by_modality) != {"text", "visual"}:
        raise ValueError("eta_by_modality must contain exactly text and visual")
    text = eta_by_modality["text"]
    visual = eta_by_modality["visual"]
    if text.ndim != 2 or visual.ndim != 2 or tuple(text.shape) != tuple(visual.shape):
        raise ValueError("text and visual eta must both have shape [N,K+1]")
    stacked = torch.stack((text, visual), dim=0)
    mu = stacked.mean(dim=(0, 1))
    modality_means = stacked.mean(dim=1)
    nu = modality_means - mu.unsqueeze(0)
    xi = stacked - modality_means.unsqueeze(1)
    reconstructed = mu.view(1, 1, -1) + nu.unsqueeze(1) + xi
    return {
        "eta": {"text": text, "visual": visual},
        "stacked": stacked,
        "mu": mu,
        "nu": {"text": nu[0], "visual": nu[1]},
        "xi": {"text": xi[0], "visual": xi[1]},
        "reconstructed": {"text": reconstructed[0], "visual": reconstructed[1]},
        "reconstruction_max_abs": float((reconstructed - stacked).abs().max().item()),
        "mean_modality_deviation": nu.mean(dim=0),
        "mean_node_centered": {"text": xi[0].mean(dim=0), "visual": xi[1].mean(dim=0)},
    }


def distribution_summary(values: torch.Tensor | np.ndarray) -> dict[str, float | int]:
    """Return the required robust distribution summary for a numeric tensor."""
    array = values.detach().cpu().double().reshape(-1).numpy() if isinstance(values, torch.Tensor) else np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "q10": 0.0, "q25": 0.0, "q75": 0.0, "q90": 0.0, "l2_norm": 0.0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "l2_norm": float(np.linalg.norm(array)),
    }


def coefficient_statistics(
    eta: torch.Tensor,
    *,
    name: str,
) -> dict[str, Any]:
    """Summarize a coefficient tensor per order and globally."""
    if eta.ndim != 2:
        raise ValueError("eta must have shape [N,K+1]")
    values = eta.detach().double()
    return {
        "name": name,
        "per_order": [distribution_summary(values[:, order]) for order in range(values.size(1))],
        "global": distribution_summary(values),
    }


def contribution_profile(
    eta: torch.Tensor,
    states: list[torch.Tensor],
    *,
    eps: float = EPS,
) -> dict[str, Any]:
    """Compute actual state contributions and contribution-weighted order."""
    if eta.ndim != 2 or len(states) != eta.size(1):
        raise ValueError("eta must be [N,K+1] and states must contain K+1 tensors")
    contributions = torch.stack(
        [eta[:, order : order + 1] * states[order] for order in range(eta.size(1))],
        dim=1,
    )
    magnitudes = contributions.norm(dim=-1)
    denominator = magnitudes.sum(dim=1, keepdim=True)
    nonzero = denominator.squeeze(1) > float(eps)
    probabilities = torch.where(
        denominator > float(eps),
        magnitudes / denominator.clamp_min(float(eps)),
        torch.zeros_like(magnitudes),
    )
    orders = torch.arange(eta.size(1), device=eta.device, dtype=eta.dtype)
    response_order = (probabilities * orders.view(1, -1)).sum(dim=1)
    order_count = max(eta.size(1) - 1, 1)
    entropy = -(probabilities * torch.log(probabilities + float(eps))).sum(dim=1)
    entropy_normalized = entropy / float(np.log(eta.size(1)))
    return {
        "contributions": contributions,
        "magnitudes": magnitudes,
        "probabilities": probabilities,
        "probability_sum": probabilities.sum(dim=1),
        "all_zero": ~nonzero,
        "response_order": response_order,
        "normalized_response_order": response_order / float(order_count),
        "entropy": entropy,
        "normalized_entropy": entropy_normalized,
        "summary": {
            "response_order": distribution_summary(response_order),
            "normalized_response_order": distribution_summary(response_order / float(order_count)),
            "entropy": distribution_summary(entropy),
            "normalized_entropy": distribution_summary(entropy_normalized),
            "probability_sum_max_error": float((probabilities[nonzero].sum(dim=1) - 1.0).abs().max().item()) if bool(nonzero.any()) else 0.0,
            "all_zero_count": int((~nonzero).sum().item()),
        },
    }


def shuffle_centered_profile(xi: torch.Tensor, seed: int) -> torch.Tensor:
    """Shuffle complete [K+1] profiles with one node permutation."""
    if xi.ndim != 2:
        raise ValueError("xi must have shape [N,K+1]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(xi.size(0), generator=generator).to(xi.device)
    return xi.index_select(0, permutation)


def counterfactual_eta(
    decomposition: Mapping[str, Any],
    mode: str,
    *,
    shuffle_seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Construct canonical eta interventions without touching model state."""
    mode = str(mode).strip().lower()
    mu = decomposition["mu"]
    nu = decomposition["nu"]
    xi = decomposition["xi"]
    if mode == "full":
        return {modality: decomposition["eta"][modality] for modality in ("text", "visual")}
    if mode == "nonode":
        return {modality: mu.unsqueeze(0) + nu[modality].unsqueeze(0) for modality in ("text", "visual")}
    if mode == "nomodality":
        return {modality: mu.unsqueeze(0) + xi[modality] for modality in ("text", "visual")}
    if mode == "globalonly":
        n = xi["text"].size(0)
        return {modality: mu.unsqueeze(0).expand(n, -1) for modality in ("text", "visual")}
    if mode == "modalityswap":
        return {
            "text": mu.unsqueeze(0) + nu["visual"].unsqueeze(0) + xi["text"],
            "visual": mu.unsqueeze(0) + nu["text"].unsqueeze(0) + xi["visual"],
        }
    if mode == "nodeshuffle":
        if shuffle_seed is None:
            raise ValueError("nodeshuffle requires shuffle_seed")
        return {
            modality: mu.unsqueeze(0) + nu[modality].unsqueeze(0) + shuffle_centered_profile(xi[modality], shuffle_seed)
            for modality in ("text", "visual")
        }
    raise ValueError(f"Unknown U3-A counterfactual mode: {mode!r}")


def profile_pairwise_variability(xi: torch.Tensor, *, max_profiles: int = 512) -> dict[str, Any]:
    """Summarize pairwise node-profile variability without an NxN matrix."""
    if xi.ndim != 2:
        raise ValueError("xi must have shape [N,K+1]")
    count = min(int(max_profiles), int(xi.size(0)))
    subset = xi[:count].double()
    if count < 2:
        distances = subset.new_zeros((0,))
    else:
        distances = torch.pdist(subset, p=2)
    return {"sample_count": count, "pair_count": int(distances.numel()), "distance": distribution_summary(distances)}
