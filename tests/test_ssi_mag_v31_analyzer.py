from __future__ import annotations

import torch
from omegaconf import OmegaConf

from scripts.analyze_ssi_mag_v31_p17b import (
    _alpha_range_ratio,
    _covariance_contribution,
    _cross_seed_rows,
    _operator_metrics,
    _tail_fractions,
    _validate_attention_simplex,
    attention_diagnostics,
)
from src.models.ssi_mag_v31 import SSIMAGV31


class _IdentityOperator:
    @staticmethod
    def _normalized_operator(edge_index, edge_weight, num_nodes, dtype):
        return edge_index, edge_weight


def _model() -> SSIMAGV31:
    cfg = OmegaConf.create(
        {
            "ablation": "full",
            "model": {
                "name": "ssi_mag_v31",
                "hidden_dim": 8,
                "dropout": 0.0,
                "norm": "layernorm",
                "max_order": 3,
                "num_layers": 3,
                "diffusion_add_self_loops": True,
                "relation_rank": 3,
                "relation_strength_init": 0.05,
                "relation_edge_chunk_size": 2,
                "relation_init_seed": 20260923,
                "relation_init_std": 0.02,
                "semantic_reference_init": 0.10,
                "context_change_gate_init": 0.10,
                "hop_interaction_gate_init": 0.10,
                "hop_interaction_layers": 1,
                "hop_interaction_heads": 1,
                "hop_interaction_dropout": 0.0,
                "hop_interaction_order_embedding": True,
                "filter_rank": 2,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "relation_filter_scale_init": 0.10,
                "fusion": "concat_residual_mlp",
                "eps": 1e-8,
            },
        }
    )
    return SSIMAGV31(cfg, {"input_dim": 10, "text_dim": 4, "visual_dim": 6})


def test_operator_identical_has_zero_perturbation_metrics() -> None:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 2]], dtype=torch.long)
    weights = torch.ones(3)
    metrics = _operator_metrics(
        _IdentityOperator(), edge_index, edge_index, weights, 3, torch.float32
    )
    assert metrics["operator_weight_mae"] == 0.0
    assert metrics["operator_weight_rmse"] == 0.0
    assert metrics["operator_weight_relative_l1"] == 0.0


def test_analyzer_preserves_explicit_self_loop_identity_after_nonzero_scorer() -> None:
    model = _model()
    with torch.no_grad():
        model.relation_scorer_text.weight.copy_(torch.tensor([[0.2, -0.1, 0.05]]))
        model.relation_scorer_text.bias.fill_(0.03)
    x = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)
    h0 = model.text_proj(x[:, :4])
    relation = model._relation_edge_outputs(h0, edge_index, "text")
    assert relation["relation_residual"][-1].item() == 0.0
    assert relation["relation_weight"][-1].item() == 1.0
    c = model._local_adaptation(
        edge_index,
        relation["relation_residual"],
        relation["beta"],
        relation["nonself_mask"],
        x.size(0),
    )
    assert c[2].item() == 0.0


def test_eta_variance_attribution_reconstructs_covariance_contribution() -> None:
    content = torch.tensor([1.0, 2.0, 4.0, 8.0])
    relation = torch.tensor([0.0, 1.0, 0.0, -1.0])
    eta = content + relation
    content_part = _covariance_contribution(content, eta)
    relation_part = _covariance_contribution(relation, eta)
    assert content_part["variance_near_zero"] == 0
    assert relation_part["variance_near_zero"] == 0
    total = content_part["covariance_contribution"] + relation_part["covariance_contribution"]
    assert abs(total - 1.0) < 1e-6


def test_alpha_corrected_range_ratio_uses_abs_mean_denominator() -> None:
    alpha = torch.tensor([0.05, 0.10, 0.15, 0.20])
    expected = (torch.quantile(alpha, 0.90) - torch.quantile(alpha, 0.10)) / (alpha.mean().abs() + 1e-12)
    assert abs(_alpha_range_ratio(alpha) - expected.item()) < 1e-6


def test_p_scale_tail_metrics_are_not_tanh_saturation_metrics() -> None:
    tails = _tail_fractions(torch.tensor([-6.0, -4.0, 0.0, 2.0, 5.0]))
    assert tails["fraction_abs_gt_3"] == 0.6
    assert tails["fraction_abs_gt_5"] == 0.2


def test_cross_seed_ordering_guard_blocks_unverified_correlations() -> None:
    vectors = {
        ("Movies", 42, "text"): {"term_p": torch.tensor([1.0, 2.0, 3.0]), "alpha": [torch.tensor([0.1, 0.2, 0.3])]},
        ("Movies", 43, "text"): {"term_p": torch.tensor([1.1, 2.1, 3.1]), "alpha": [torch.tensor([0.1, 0.2, 0.3])]},
        ("Movies", 42, "visual"): {"term_p": torch.tensor([2.0, 3.0, 4.0]), "alpha": [torch.tensor([0.2, 0.3, 0.4])]},
        ("Movies", 43, "visual"): {"term_p": torch.tensor([2.1, 3.1, 4.1]), "alpha": [torch.tensor([0.2, 0.3, 0.4])]},
    }
    matching = {"Movies": {42: (3, "x", "e", "y"), 43: (3, "x", "e", "y")}}
    rows, verified = _cross_seed_rows(vectors, matching, ("Movies",), (42, 43))
    assert verified["Movies"] is True
    assert any(row["metric"] == "term_p" and row["ordering_verified"] for row in rows)
    mismatched = {"Movies": {42: (3, "x", "e", "y"), 43: (3, "different", "e", "y")}}
    rows, verified = _cross_seed_rows(vectors, mismatched, ("Movies",), (42, 43))
    assert verified["Movies"] is False
    assert rows == [{"dataset": "Movies", "ordering_verified": False, "metric": "not_computed"}]


def test_analyzer_reuses_corrected_nodewise_attention_definition() -> None:
    attention = torch.full((2, 4, 4), 0.25)
    _, _, metrics = attention_diagnostics(attention)
    assert metrics["nodewise_normalized_entropy"] == 1.0
    assert metrics["attention_nonuniformity"] == 0.0


def test_attention_simplex_rejects_invalid_rows() -> None:
    import pytest

    valid = torch.full((2, 4, 4), 0.25)
    result = _validate_attention_simplex(valid)
    assert result["attention_max_row_sum_error"] < 1e-5
    with pytest.raises(ValueError, match="simplex-normalized"):
        _validate_attention_simplex(valid * 0.9)
    with pytest.raises(ValueError, match="negative mass"):
        _validate_attention_simplex(torch.tensor([[[1.1, -0.1, 0.0, 0.0]]]))
