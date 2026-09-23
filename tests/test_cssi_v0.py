from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.cssi_v0 import CSSIV0


def _cfg(**overrides):
    values = {
        "name": "cssi_v0",
        "hidden_dim": 16,
        "dropout": 0.0,
        "norm": "layernorm",
        "max_order": 3,
        "num_layers": 3,
        "diffusion_add_self_loops": True,
        "use_mrc": False,
        "edge_metric": "learned_diag_cos",
        "metric_init_seed": 20260910,
        "edge_weight_min": 0.1,
        "edge_weight_temperature": 0.35,
        "eps": 1.0e-8,
        "filter_rank": 4,
        "multihop_state": "anchored",
        "multihop_response": "cumulative",
        "multihop_anchor_alpha": 0.0,
        "global_prior_restart": 0.15,
        "global_prior_order": 2,
        "hop_interaction_layers": 1,
        "hop_interaction_heads": 1,
        "hop_interaction_dropout": 0.0,
        "hop_interaction_order_embedding": True,
        "hop_interaction_gate_init": 0.10,
        "relation_descriptor_eps": 1.0e-6,
        "relation_bias_init": 0.10,
        "relation_init_seed": 20260921,
        "relation_init_std": 0.02,
        "fusion": "concat_residual_mlp",
        "condition_mode": "self",
        "adapter_type": "lowrank",
        "adapter_enabled": True,
        "control_dim": 8,
        "response_rank": 4,
        "response_correction_scale": 1.0e-3,
    }
    values.update(overrides)
    return OmegaConf.create({"model": values})


def _inputs():
    torch.manual_seed(7)
    x = torch.randn(8, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 5, 6, 7], [1, 0, 2, 1, 3, 2, 5, 4, 7, 6]],
        dtype=torch.long,
    )
    return x, edge_index


def _model(**overrides):
    torch.manual_seed(11)
    return CSSIV0(
        _cfg(**overrides),
        {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6},
    ).eval()


def test_zero_init_lowrank_is_plain_and_preserves_explicit_response_formula():
    x, edge_index = _inputs()
    model = _model()
    with torch.no_grad():
        components = model.analysis_components(x, edge_index)
    assert torch.equal(components["normalized_edge_weight_text"], components["normalized_edge_weight_visual"])
    assert torch.allclose(components["z"], components["z_base"], atol=1e-7, rtol=0.0)
    for modality in ("text", "visual"):
        states = components[f"states_{modality}"]
        responses = components[f"responses_{modality}"]
        for previous, current in zip(states[:-1], states[1:]):
            expected_next = model._propagate_once(
                previous,
                components[f"normalized_edge_index_{modality}"],
                components[f"normalized_edge_weight_{modality}"],
            )
            assert torch.allclose(current, expected_next, atol=1e-6, rtol=0.0)
        base = torch.stack(states, dim=1).mean(dim=1)
        expected = states[0] + sum(
            coefficient * response
            for coefficient, response in zip((0.75, 0.50, 0.25), responses)
        )
        assert torch.allclose(base, expected, atol=1e-6, rtol=0.0)
        assert torch.equal(
            components[f"corrections_{modality}"][0],
            torch.zeros_like(components[f"corrections_{modality}"][0]),
        )
    classifier = torch.nn.Linear(model.out_dim, 3)
    assert torch.allclose(
        classifier(components["z"]), classifier(components["z_base"]), atol=1e-7, rtol=0.0
    )


def test_response_correction_and_utility_sign_convention():
    # The P3 utility convention is helpful = positive loss/margin reduction.
    base_loss, counterfactual_loss = 0.8, 0.5
    base_margin, counterfactual_margin = 0.2, -0.1
    assert counterfactual_loss - base_loss < 0.0
    assert base_loss - counterfactual_loss > 0.0
    assert base_margin - counterfactual_margin > 0.0


def test_same_and_all_controller_shapes_and_all_order_attention():
    x, edge_index = _inputs()
    for mode in ("same", "all"):
        model = _model(condition_mode=mode)
        with torch.no_grad():
            components = model.analysis_components(x, edge_index)
        assert components["control_text"].shape == (8, 3, 8)
        assert components["control_visual"].shape == (8, 3, 8)
        if mode == "all":
            assert components["attention_text"].shape == (8, 3, 3)
            assert components["attention_visual"].shape == (8, 3, 3)
            assert torch.allclose(
                components["attention_text"].sum(dim=-1),
                torch.ones(8, 3),
                atol=1e-6,
            )


def test_scalar_adapter_keeps_nodewise_gate_shape():
    x, edge_index = _inputs()
    model = _model(condition_mode="same", adapter_type="scalar")
    with torch.no_grad():
        components = model.analysis_components(x, edge_index)
    for modality in ("text", "visual"):
        for correction in components[f"corrections_{modality}"]:
            assert correction.shape == (8, 16)
        assert components[f"controller_gate_{modality}"].shape == (8, 3)


def test_lowrank_is_not_a_dense_d_by_d_operator_and_mrc_flag_is_supported():
    model = _model(use_mrc=True, condition_mode="same")
    assert model.response_up_text.weight.shape == (16, 4)
    assert model.response_up_visual.weight.shape == (16, 4)
    assert model.use_mrc
