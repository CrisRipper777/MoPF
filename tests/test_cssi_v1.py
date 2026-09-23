from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.cssi_v1 import CSSIV1


def _cfg(**overrides):
    values = {
        "name": "cssi_v1",
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
        "condition_mode": "same",
        "adapter_type": "brsm",
        "adapter_enabled": True,
        "control_dim": 8,
        "response_rank": 4,
        "lambda_max": 0.2,
        "lambda_init": 0.05,
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
    return CSSIV1(
        _cfg(**overrides),
        {"input_dim": 10, "num_nodes": 8, "text_dim": 4, "visual_dim": 6},
    ).eval()


def test_basis_is_row_orthonormal_and_zero_response_has_zero_correction():
    x, edge_index = _inputs()
    model = _model()
    with torch.no_grad():
        components = model.analysis_components(x, edge_index)
    for modality in ("text", "visual"):
        basis = components[f"response_basis_{modality}"]
        identity = basis @ basis.transpose(0, 1)
        assert torch.allclose(identity, torch.eye(4), atol=1e-5, rtol=0.0)
        for response, correction in zip(
            components[f"responses_{modality}"], components[f"corrections_{modality}"]
        ):
            zero = model._adapt_responses(
                [torch.zeros_like(response)],
                components[f"control_{modality}"][:, :1],
                modality,
            )[0][0]
            assert torch.equal(zero, torch.zeros_like(zero))
            bound = correction.norm(dim=-1) <= 0.2 * response.norm(dim=-1) + 1e-5
            assert bool(bound.all())


def test_v1_has_no_fixed_output_scale_path_and_exact_cross_off():
    x, edge_index = _inputs()
    model = _model(condition_mode="same")
    with torch.no_grad():
        normal = model.analysis_components(x, edge_index, cross_intervention="normal")
        off = model.analysis_components(x, edge_index, cross_intervention="off")
    assert torch.equal(off["control_q_visual_for_text"], torch.zeros_like(off["control_q_visual_for_text"]))
    assert torch.equal(off["control_q_text_for_visual"], torch.zeros_like(off["control_q_text_for_visual"]))
    assert not torch.allclose(
        normal["corrections_text"][0], off["corrections_text"][0], atol=1e-8, rtol=0.0
    )
    assert not hasattr(model, "lambda_output_scale")


def test_scalar_is_bounded_and_controller_receives_gradient():
    x, edge_index = _inputs()
    model = _model(adapter_type="scalar")
    z, _, _, _, _ = model(x, edge_index)
    loss = z.square().mean()
    loss.backward()
    controller_grads = [
        p.grad for name, p in model.named_parameters()
        if "scalar_controller" in name and p.grad is not None
    ]
    assert controller_grads and any(float(g.abs().sum()) > 0.0 for g in controller_grads)
    with torch.no_grad():
        components = model.analysis_components(x, edge_index)
    for modality in ("text", "visual"):
        for response, correction in zip(
            components[f"responses_{modality}"], components[f"corrections_{modality}"]
        ):
            assert bool((correction.norm(dim=-1) <= 0.2 * response.norm(dim=-1) + 1e-5).all())


def test_brsm_controller_basis_and_lambda_receive_gradient():
    x, edge_index = _inputs()
    model = _model()
    z, _, _, _, _ = model(x, edge_index)
    z.square().mean().backward()
    groups = {
        "controller": [
            parameter for name, parameter in model.named_parameters()
            if "response_condition" in name
        ],
        "basis": [model.response_basis_text, model.response_basis_visual],
        "lambda": [model.lambda_theta_text, model.lambda_theta_visual],
    }
    for parameters in groups.values():
        assert parameters and all(parameter.grad is not None for parameter in parameters)
        assert any(float(parameter.grad.abs().sum()) > 0.0 for parameter in parameters)
