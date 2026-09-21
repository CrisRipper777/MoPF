from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.analyze_mechanism_verification import (
    _directional_margins,
    _feature_space_linear_cka,
    _ranked_relation_categories,
    _same_bank_contribution_reallocation,
    _same_operator_state_banks,
)
from src.models.factory import build_model


def test_raw_relation_categories_use_frozen_empirical_rank_cuts():
    sim_text = np.arange(8, dtype=np.float64)
    sim_visual = np.array([6, 7, 2, 3, 4, 5, 0, 1], dtype=np.float64)
    result = _ranked_relation_categories(sim_text, sim_visual)
    assert np.array_equal(np.flatnonzero(result["text_specific"]), np.array([6, 7]))
    assert np.array_equal(np.flatnonzero(result["visual_specific"]), np.array([0, 1]))
    assert not np.any(result["text_specific"] & result["visual_specific"])
    assert np.all(result["rank_text"] >= 0.0) and np.all(result["rank_text"] <= 1.0)


def test_directional_margin_formula_matches_category_means():
    result = _directional_margins(
        np.array([3.0, 4.0, 2.0, 1.0]),
        np.array([1.0, 2.0, 5.0, 3.0]),
        np.array([True, True, False, False]),
        np.array([False, False, True, True]),
    )
    assert result["M_T"] == 2.0
    assert result["M_V"] == 2.5
    assert result["M_overall"] == 2.25
    assert result["mean_w_text_E_T"] == 3.5
    assert result["mean_w_visual_E_V"] == 4.0


def test_feature_space_linear_cka_identity_and_centering():
    torch.manual_seed(18)
    features = torch.randn(31, 7, dtype=torch.float32)
    assert abs(_feature_space_linear_cka(features, features, chunk_rows=6) - 1.0) < 1e-12
    shifted = features + torch.arange(7, dtype=torch.float32)
    assert abs(_feature_space_linear_cka(features, shifted, chunk_rows=5) - 1.0) < 1e-12


def test_anchored_and_ordinary_banks_share_h0_without_mutation():
    h0 = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    original = h0.clone()
    operator = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.5, 0.5]],
        dtype=torch.float64,
    )
    anchored, ordinary = _same_operator_state_banks(h0, lambda value: operator @ value, 3, 0.1)
    assert torch.equal(h0, original)
    assert anchored[0] is h0 and ordinary[0] is h0
    assert torch.allclose(ordinary[1], operator @ h0)
    assert torch.allclose(anchored[1], 0.9 * (operator @ h0) + 0.1 * h0)
    assert not torch.equal(anchored[1], ordinary[1])


def test_same_bank_profiles_normalize_and_reallocation_is_bounded():
    eta = torch.tensor(
        [[0.1, 0.2, 0.3], [0.7, 0.2, 0.1]],
        dtype=torch.float64,
    )
    states = [
        torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 2.0], [0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float64),
    ]
    result = _same_bank_contribution_reallocation(eta, states)
    assert torch.allclose(result["p_adapt"].sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(result["p_uniform"].sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert bool((result["D"] >= 0).all()) and bool((result["D"] <= 1).all())
    assert result["C"].shape == (2,)


def test_uniform_eta_has_zero_reallocation_for_unequal_response_norms():
    # The counterfactual is defined from this exact response bank, so equal
    # coefficients must reproduce its normalized response-norm profile.
    states = [
        torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 3.0], [0.0, 2.0]], dtype=torch.float64),
        torch.tensor([[4.0, 0.0], [0.0, 5.0]], dtype=torch.float64),
    ]
    eta = torch.full((2, 3), 1.0 / 3.0, dtype=torch.float64)
    result = _same_bank_contribution_reallocation(eta, states)
    expected_uniform_profile = torch.stack(
        [torch.linalg.vector_norm(state, dim=1) for state in states], dim=1
    )
    expected_uniform_profile /= expected_uniform_profile.sum(dim=1, keepdim=True)
    assert torch.allclose(result["p_adapt"], expected_uniform_profile, atol=1e-12, rtol=0.0)
    assert torch.allclose(result["p_uniform"], expected_uniform_profile, atol=1e-12, rtol=0.0)
    assert torch.allclose(result["D"], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0.0)


def test_analysis_helpers_do_not_mutate_model_state_dict():
    cfg = OmegaConf.create({
        "ablation": "full",
        "model": {
            "name": "mopf", "version": "mopf", "hidden_dim": 8,
            "dropout": 0.0, "norm": "layernorm", "max_order": 2,
            "num_layers": 2, "map_prior_restart": 0.15, "map_prior_order": 2,
            "diffusion_add_self_loops": True, "edge_weight_mode": "learned_diag_cos",
            "num_metric_perspectives": 4, "metric_init_seed": 20260910,
            "metric_init_noise_std": 0.01, "edge_weight_min": 0.1,
            "edge_weight_temperature": 0.35, "filter_rank": 4,
            "multihop_state_mode": "anchored", "multihop_response_mode": "cumulative",
            "multihop_anchor_alpha": 0.1, "global_filter_trainable": True,
            "use_modality_residual": True, "use_node_residual": True,
            "use_transport_residual": True, "composition_mode": "adaptive",
            "hrc_weight": 0.0, "fusion_mode": "concat_residual_mlp",
            "export_aux_stats": False, "export_node_aux": False,
            "export_edge_aux": False, "lp_pair_operator": None,
        },
        "task": {"num_neighbors": [5, 5]},
    })
    model = build_model(cfg, {"input_dim": 10, "num_nodes": 5, "text_dim": 4, "visual_dim": 6})
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        conductance = model.conductance_stats(x, edge_index)
        components = model._encode_components(x, edge_index)
        analysis = model.analysis_stats(x, edge_index)
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert conductance["conductance_text"].numel() == edge_index.size(1)
    assert components["eta_text"].shape == (5, 3)
    assert analysis["eta_visual"].shape == (5, 3)
