from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.cssi_scalar_probe import CSSIScalarProbe


def _cfg(variant: str):
    values = {
        "name": "cssi_scalar_probe",
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
        "scalar_variant": variant,
        "condition_mode": "none",
        "adapter_type": "none",
        "adapter_enabled": False,
        "control_dim": 8,
        "response_rank": 4,
        "gate_max": 0.1,
        "lambda_max": 0.2,
        "lambda_init": 0.05,
    }
    if variant == "same_scalar_mrc":
        values["use_mrc"] = True
    return OmegaConf.create({"model": values})


def _inputs():
    torch.manual_seed(19)
    x = torch.randn(8, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 5, 6, 7], [1, 0, 2, 1, 3, 2, 5, 4, 7, 6]],
        dtype=torch.long,
    )
    return x, edge_index


def _model(variant: str):
    torch.manual_seed(23)
    return CSSIScalarProbe(
        _cfg(variant),
        {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6},
    ).eval()


def test_static_zero_gate_has_direct_gradient_and_is_bounded():
    x, edge_index = _inputs()
    model = _model("static_modality")
    z, _, _, _, _ = model(x, edge_index)
    z.square().mean().backward()
    assert model.static_gate_theta.grad is not None
    assert float(model.static_gate_theta.grad.abs().sum()) > 0.0
    with torch.no_grad():
        components = model.analysis_components(x, edge_index)
    for modality in ("text", "visual"):
        gate = components[f"scalar_gate_{modality}"]
        assert torch.all(gate.abs() <= 0.1 + 1e-7)
        assert torch.equal(
            components[f"corrections_{modality}"][0],
            gate[:, 0:1] * components[f"responses_{modality}"][0],
        )


def test_static_order_is_node_and_cross_independent():
    x, edge_index = _inputs()
    model = _model("static_order")
    with torch.no_grad():
        normal = model.analysis_components(x, edge_index, cross_intervention="normal")
        off = model.analysis_components(x, edge_index, cross_intervention="off")
    for modality in ("text", "visual"):
        assert torch.allclose(
            normal[f"scalar_gate_{modality}"],
            off[f"scalar_gate_{modality}"],
            atol=0.0,
            rtol=0.0,
        )
        assert torch.allclose(
            normal[f"scalar_gate_{modality}"],
            normal[f"scalar_gate_{modality}"][0:1].expand_as(
                normal[f"scalar_gate_{modality}"]
            ),
            atol=0.0,
            rtol=0.0,
        )


def test_same_scalar_cross_off_removes_paired_projection_bias():
    x, edge_index = _inputs()
    model = _model("same_scalar_mrc")
    with torch.no_grad():
        normal = model.analysis_components(x, edge_index, cross_intervention="normal")
        off = model.analysis_components(x, edge_index, cross_intervention="off")
    assert torch.equal(off["control_q_visual_for_text"], torch.zeros_like(off["control_q_visual_for_text"]))
    assert torch.equal(off["control_q_text_for_visual"], torch.zeros_like(off["control_q_text_for_visual"]))
    assert not torch.allclose(
        normal["scalar_gate_text"], off["scalar_gate_text"], atol=1e-8, rtol=0.0
    )
