from __future__ import annotations

from types import SimpleNamespace

import torch

from src.analysis.u2a import (
    bank_equivalence,
    capture_formal_forward_banks,
    deterministic_node_subset,
    extract_analysis_response_bank,
    frobenius_cosine_matrix,
    incremental_novelty,
    linear_cka_matrix,
    normalized_response_gram,
    sparse_symmetry_audit,
)
from src.models.mopf import MoPF


def _cfg():
    return SimpleNamespace(
        model={
            "hidden_dim": 8,
            "dropout": 0.0,
            "norm": "layernorm",
            "max_order": 3,
            "num_layers": 3,
            "map_prior_restart": 0.15,
            "map_prior_order": 2,
            "diffusion_add_self_loops": True,
            "edge_weight_mode": "learned_diag_cos",
            "num_metric_perspectives": 4,
            "metric_init_seed": 20260910,
            "metric_init_noise_std": 0.01,
            "edge_weight_min": 0.1,
            "edge_weight_temperature": 0.35,
            "filter_rank": 4,
            "global_filter_trainable": True,
            "use_modality_residual": True,
            "use_node_residual": True,
            "hrc_weight": 0.0,
            "ppc_weight": 0.0,
            "fusion_mode": "concat_residual_mlp",
            "node_conditioner_mode": "absolute",
        }
    )


def _model_and_graph():
    torch.manual_seed(123)
    model = MoPF(
        _cfg(),
        {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6},
    )
    model.eval()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0]],
        dtype=torch.long,
    )
    return model, x, edge_index


def test_analysis_bank_matches_formal_forward_bank_and_operator():
    model, x, edge_index = _model_and_graph()
    state_keys = tuple(model.state_dict().keys())
    with torch.no_grad():
        analysis = extract_analysis_response_bank(model, x, edge_index)
        formal = capture_formal_forward_banks(model, x, edge_index)
    equivalence = bank_equivalence(analysis["banks"], formal)
    assert equivalence["passed"]
    assert equivalence["max_abs_difference"] < 1e-6
    assert torch.equal(analysis["norm_text_index"], formal["operators"][0][0])
    assert torch.equal(analysis["norm_visual_index"], formal["operators"][1][0])
    assert tuple(model.state_dict().keys()) == state_keys
    assert model.edge_weight_mode == "learned_diag_cos"
    assert model.edge_weight_temperature == 0.35


def test_response_metrics_have_expected_domains():
    responses = [
        torch.arange(30, dtype=torch.float32).view(6, 5),
        torch.arange(30, dtype=torch.float32).view(6, 5) * 0.5,
        torch.flip(torch.arange(30, dtype=torch.float32).view(6, 5), dims=[0]),
    ]
    cosine = frobenius_cosine_matrix(responses)
    assert torch.allclose(torch.diag(cosine), torch.ones(3, dtype=cosine.dtype), atol=1e-6)
    subset = deterministic_node_subset(6, max_nodes=4, seed=17)
    cka = linear_cka_matrix(responses, subset)
    assert torch.isfinite(cka).all()
    assert float(cka.min()) >= -1e-8
    assert float(cka.max()) <= 1.0 + 1e-6
    gram, spectrum = normalized_response_gram(responses)
    assert torch.allclose(torch.diag(gram), torch.ones(3, dtype=gram.dtype), atol=1e-6)
    assert spectrum["psd_within_numerical_tolerance"]
    assert 1.0 <= spectrum["effective_rank"] <= 3.0 + 1e-6
    novelty = incremental_novelty(responses)
    assert all(torch.isfinite(torch.tensor(value)) for value in novelty["novelty_ratio"])
    assert all(0.0 <= value <= 1.0 + 1e-6 for value in novelty["novelty_ratio"])


def test_sparse_symmetry_audit_matches_dense_toy_calculation():
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 2.0, 0.5, 0.25])
    audit = sparse_symmetry_audit(edge_index, edge_weight, 2)
    expected = ((0.5 - 0.25) ** 2 + (0.25 - 0.5) ** 2) ** 0.5
    expected /= (1.0**2 + 2.0**2 + 0.5**2 + 0.25**2) ** 0.5
    assert abs(audit["asymmetry_ratio"] - expected) < 1e-12
    assert audit["classification"] == "materially_asymmetric"


def test_u1t_relation_config_is_frozen_for_u2a():
    config = ("configs/model/mopf.yaml")
    text = open(config, encoding="utf-8").read()
    assert "edge_weight_mode: learned_diag_cos" in text
    assert "edge_weight_temperature: 0.35" in text
