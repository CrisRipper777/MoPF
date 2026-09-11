from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.analysis.u2c0 import algebra_audit
from src.models.mopf import (
    MoPF,
    anchored_differential_to_monomial_coefficients,
    monomial_to_anchored_differential_coefficients,
)


def _cfg(**updates):
    model = {
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
    model.update(updates)
    return SimpleNamespace(model=model)


def _graph():
    torch.manual_seed(123)
    model = MoPF(
        _cfg(),
        {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6},
    )
    x = torch.randn(6, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0]],
        dtype=torch.long,
    )
    return model, x, edge_index


def test_default_cumulative_mode_delegates_to_historical_bank():
    model, x, edge_index = _graph()
    model.eval()
    with torch.no_grad():
        x_text, _ = model._split_features(x)
        h0 = model.text_proj(x_text)
        edge = model._semantic_edge_weights(h0, h0, edge_index)
        index, weight = model._normalized_operator(edge_index, edge["w_t"], x.size(0), h0.dtype)
        old = model._propagation_bank(h0, index, weight)
        banks = model._build_multihop_banks(h0, index, weight)
    assert model.multihop_mode == "cumulative"
    assert all(torch.equal(first, second) for first, second in zip(old, banks["states"]))
    assert all(torch.equal(first, second) for first, second in zip(old, banks["responses"]))


def test_anchored_state_response_banks_and_gradient_path():
    model, x, edge_index = _graph()
    anchored = MoPF(
        _cfg(multihop_mode="anchored_differential", multihop_anchor_alpha=0.1),
        {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6},
    )
    assert not any(name == "multihop_anchor_alpha" for name, _ in anchored.named_parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel() for parameter in anchored.parameters()
    )
    anchored.node_vector_text.data.normal_()
    anchored.train()
    z, _, _, aux_loss, _ = anchored(x, edge_index)
    loss = z.square().mean() + aux_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in anchored.parameters()
    )
    anchored.eval()
    with torch.no_grad():
        components = anchored._encode_components(x, edge_index)
        eta, _, _ = anchored._node_residuals(
            components["states_text"],
            anchored.node_proj_text,
            anchored.node_vector_text,
            "text",
        )
        eta = anchored._effective_coefficients("text", eta)
        expected = anchored._filter_bases(components["responses_text"], eta)
    assert torch.allclose(components["z_text"], expected)
    assert components["bases_text"] is components["responses_text"]
    assert components["conditioned_bases_text"] is components["conditioned_states_text"]


def test_algebra_identities_hold_in_float64():
    matrix = torch.tensor(
        [[0.2, 0.8, 0.0], [0.1, 0.5, 0.4], [0.0, 0.3, 0.7]],
        dtype=torch.float64,
    )
    h0 = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=torch.float64)
    audit = algebra_audit(h0, lambda value: matrix @ value, 5, 0.1)
    assert audit["max_state_reconstruction_abs"] < 1e-10
    assert audit["max_closed_form_abs"] < 1e-10
    assert audit["max_scaling_abs"] < 1e-10
    assert audit["max_alpha_zero_abs"] < 1e-10


def test_coefficient_transform_supports_vector_and_batch_exactly():
    eta = torch.tensor([0.15, -0.2, 0.4, 0.65], dtype=torch.float64)
    batch = torch.stack((eta, eta.flip(0)))
    for value in (eta, batch):
        differential = monomial_to_anchored_differential_coefficients(value, 0.1)
        reconstructed = anchored_differential_to_monomial_coefficients(differential, 0.1)
        assert torch.max(torch.abs(reconstructed - value)) < 1e-10
    assert torch.allclose(
        monomial_to_anchored_differential_coefficients(eta, 0.0),
        torch.tensor([eta.sum(), eta[1:].sum(), eta[1:].sum() - eta[1], eta[3]], dtype=torch.float64),
    )


def test_pdc_is_explicitly_rejected_for_anchored_differential():
    with pytest.raises(ValueError, match="requires.*absolute"):
        MoPF(
            _cfg(node_conditioner_mode="pdc", multihop_mode="anchored_differential"),
            {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6},
        )


def test_formal_config_relation_and_temperature_remain_frozen():
    text = open("configs/model/mopf.yaml", encoding="utf-8").read()
    assert "edge_weight_mode: learned_diag_cos" in text
    assert "edge_weight_temperature: 0.35" in text
    assert "multihop_mode:" not in text
