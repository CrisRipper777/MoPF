from __future__ import annotations

import torch

from src.analysis.u3a import (
    canonical_effective_decomposition,
    contribution_profile,
    counterfactual_eta,
    profile_pairwise_variability,
    shuffle_centered_profile,
)


def _eta():
    text = torch.tensor([[1.0, 2.0, 0.5], [1.5, 1.0, 0.0], [0.5, 3.0, 1.5]])
    visual = torch.tensor([[0.5, 1.0, 1.0], [1.0, 1.5, 0.5], [2.0, 0.5, 1.5]])
    return text, visual


def test_canonical_decomposition_reconstructs_and_centers_components():
    text, visual = _eta()
    decomposition = canonical_effective_decomposition({"text": text, "visual": visual})
    assert decomposition["reconstruction_max_abs"] < 1e-7
    assert torch.allclose(decomposition["mean_modality_deviation"], torch.zeros(3), atol=1e-6)
    assert torch.allclose(decomposition["mean_node_centered"]["text"], torch.zeros(3), atol=1e-6)
    assert torch.allclose(decomposition["mean_node_centered"]["visual"], torch.zeros(3), atol=1e-6)


def test_contribution_profile_normalizes_and_has_valid_order_entropy():
    text, _ = _eta()
    states = [torch.ones(3, 4), torch.full((3, 4), 2.0), torch.full((3, 4), 3.0)]
    profile = contribution_profile(text, states)
    assert torch.allclose(profile["probability_sum"], torch.ones(3))
    assert torch.isfinite(profile["entropy"]).all()
    assert ((profile["response_order"] >= 0) & (profile["response_order"] <= 2)).all()
    assert ((profile["normalized_response_order"] >= 0) & (profile["normalized_response_order"] <= 1)).all()
    assert ((profile["normalized_entropy"] >= 0) & (profile["normalized_entropy"] <= 1)).all()


def test_counterfactuals_change_only_the_requested_canonical_components():
    text, visual = _eta()
    decomposition = canonical_effective_decomposition({"text": text, "visual": visual})
    nonode = counterfactual_eta(decomposition, "nonode")
    nomodality = counterfactual_eta(decomposition, "nomodality")
    global_only = counterfactual_eta(decomposition, "globalonly")
    swap = counterfactual_eta(decomposition, "modalityswap")
    assert torch.allclose(nonode["text"], decomposition["mu"] + decomposition["nu"]["text"])
    assert torch.allclose(nomodality["visual"], decomposition["mu"] + decomposition["xi"]["visual"])
    assert torch.allclose(global_only["text"], decomposition["mu"].expand_as(text))
    assert torch.allclose(swap["text"], decomposition["mu"] + decomposition["nu"]["visual"] + decomposition["xi"]["text"])


def test_node_shuffle_preserves_full_profile_marginal_and_is_single_permutation():
    xi = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    shuffled = shuffle_centered_profile(xi, 20260921)
    assert torch.equal(torch.sort(shuffled[:, 0]).values, torch.sort(xi[:, 0]).values)
    assert torch.equal(torch.sort(shuffled[:, 1]).values, torch.sort(xi[:, 1]).values)
    # The same row permutation must apply to every order.
    for row in shuffled:
        source_rows = torch.where((xi == row).all(dim=1))[0]
        assert source_rows.numel() == 1
    assert torch.equal(shuffled, shuffle_centered_profile(xi, 20260921))


def test_profile_pairwise_variability_is_finite():
    xi = torch.randn(12, 4)
    result = profile_pairwise_variability(xi)
    assert result["pair_count"] == 66
    assert result["distance"]["count"] == 66
