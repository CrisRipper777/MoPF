from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.cosi_mag_final import CoSIMAGFinal
from src.models.factory import build_model
from src.models.ssi_mag_v3 import SSIMAGV3


def _cfg(ablation: str = "full", *, max_order: int = 3, hidden_dim: int = 8):
    return OmegaConf.create(
        {
            "ablation": ablation,
            "model": {
                "name": "ssi_mag_v3",
                "hidden_dim": hidden_dim,
                "dropout": 0.0,
                "norm": "layernorm",
                "max_order": max_order,
                "num_layers": max_order,
                "diffusion_add_self_loops": True,
                "relation_rank": 3,
                "relation_strength_init": 0.05,
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
                "reference_filter_scale_init": 0.10,
                "relation_filter_scale_init": 0.10,
                "fusion": "concat_residual_mlp",
                "eps": 1.0e-8,
                "relation_edge_chunk_size": 2,
            },
        }
    )


def _model(ablation: str = "full", *, max_order: int = 3) -> SSIMAGV3:
    torch.manual_seed(17)
    return SSIMAGV3(
        _cfg(ablation, max_order=max_order),
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    x = torch.randn(6, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    return x, edge_index


def test_beta_zero_restores_unit_weights_on_all_physical_edges() -> None:
    x, edge_index = _inputs()
    model = _model("raw_relation")
    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
    for modality in ("text", "visual"):
        branch = components[modality]
        assert branch["beta"].item() == 0.0
        torch.testing.assert_close(
            branch["relation_weight"], torch.ones(edge_index.size(1)), rtol=0.0, atol=0.0
        )


def test_relation_descriptor_and_residual_are_symmetric_under_edge_reversal() -> None:
    x, _ = _inputs()
    model = _model()
    model.eval()
    h0 = model.text_proj(x[:, :4])
    forward = torch.tensor([[0, 2], [2, 4]], dtype=torch.long)
    reverse = forward.flip(0)
    _, residual_forward = model._relation_descriptor(h0, forward, "text")
    _, residual_reverse = model._relation_descriptor(h0, reverse, "text")
    torch.testing.assert_close(residual_forward, residual_reverse, rtol=0.0, atol=0.0)


def test_zero_initialized_semantic_reference_parameters_give_alpha_point_one() -> None:
    x, edge_index = _inputs()
    model = _model()
    model.eval()
    analysis = model.analysis(x, edge_index)
    for modality in ("text", "visual"):
        for alpha in analysis[f"alpha_{modality}"]:
            torch.testing.assert_close(alpha, torch.full_like(alpha, 0.1), atol=1e-7, rtol=0.0)


def test_isolated_nodes_have_zero_finite_local_adaptation() -> None:
    x, edge_index = _inputs()
    model = _model()
    analysis = model.analysis(x, edge_index)
    for modality in ("text", "visual"):
        c = analysis[f"local_adaptation_{modality}"]
        assert c[4].item() == 0.0
        assert c[5].item() == 0.0
        assert torch.isfinite(c).all()


def test_all_context_banks_have_expected_shapes_and_are_finite() -> None:
    x, edge_index = _inputs()
    model = _model()
    analysis = model.analysis(x, edge_index)
    for modality in ("text", "visual"):
        for key in ("Q", "S", "D", "S_tilde"):
            bank = analysis[f"{key}_{modality}"]
            assert len(bank) == (3 if key == "Q" else 4)
            for value in bank:
                assert value.shape == (6, 8)
                assert torch.isfinite(value).all()
        assert analysis[f"alpha_{modality}"][0].shape == (6,)
        assert analysis[f"eta_{modality}"].shape == (6, 4)
        assert analysis[f"z_{modality}"].shape == (6, 8)
        assert torch.isfinite(analysis[f"eta_{modality}"]).all()
        assert torch.isfinite(analysis[f"z_{modality}"]).all()
        assert torch.isfinite(analysis[f"attention_{modality}"]).all()


def test_interaction_off_is_exactly_identity_for_context_content() -> None:
    x, edge_index = _inputs()
    model = _model()
    analysis = model.analysis_intervention(x, edge_index, interaction="off")
    for modality in ("text", "visual"):
        for s_tilde, state in zip(
            analysis[f"S_tilde_{modality}"], analysis[f"S_{modality}"], strict=True
        ):
            assert torch.equal(s_tilde, state)


def test_final_composition_uses_interacted_context_not_original_context() -> None:
    x, edge_index = _inputs()
    model = _model()
    model.eval()
    analysis = model.analysis(x, edge_index)
    expected = model._compose(
        analysis["S_tilde_text"], analysis["eta_text"]
    )
    original = model._compose(analysis["S_text"], analysis["eta_text"])
    torch.testing.assert_close(expected, analysis["z_text"], rtol=0.0, atol=0.0)
    assert not torch.allclose(expected, original, rtol=1e-6, atol=1e-7)


def test_task_loss_has_finite_gradients_through_all_required_paths() -> None:
    x, edge_index = _inputs()
    model = _model()
    head = nn.Linear(model.out_dim, 3)
    labels = torch.tensor([0, 1, 2, 1, 0, 2])
    z, _, _, aux_loss, _ = model(x, edge_index)
    loss = nn.CrossEntropyLoss()(head(z), labels) + aux_loss
    loss.backward()
    names = (
        "relation_scorer_text.0.weight",
        "theta_beta_text",
        "semantic_bias_text",
        "semantic_rho_p_text",
        "semantic_rho_d_text",
        "semantic_rho_c_text",
        "hop_layers_text.0.query.weight",
        "hop_layers_text.0.key.weight",
        "hop_layers_text.0.value.weight",
        "node_proj_text.0.weight",
    )
    parameters = dict(model.named_parameters())
    for name in names:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum().item() > 0.0, name


def test_text_and_visual_streams_are_independent_until_late_fusion() -> None:
    x, edge_index = _inputs()
    model = _model()
    model.eval()
    first = model.analysis(x, edge_index)
    changed = x.clone()
    changed[:, 4:] += 7.0
    second = model.analysis(changed, edge_index)
    for key in ("z_text", "z_text_refined", "S_text", "S_tilde_text", "eta_text"):
        if isinstance(first[key], list):
            for left, right in zip(first[key], second[key], strict=True):
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(first[key], second[key], rtol=0.0, atol=0.0)
    assert not torch.equal(first["z"], second["z"])


def test_nc_forward_and_inference_interfaces_are_preserved() -> None:
    x, edge_index = _inputs()
    model = _model()
    model.train()
    output = model(x, edge_index)
    assert len(output) == 5
    assert output[0].shape == (6, 8)
    inferred = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)
    assert inferred.shape == (6, 8)
    assert inferred.device.type == "cpu"


def test_num_layers_must_match_max_order() -> None:
    cfg = _cfg(max_order=3)
    cfg.model.num_layers = 2
    with pytest.raises(ValueError, match="num_layers == num_order|max_order"):
        SSIMAGV3(cfg, {"input_dim": 10, "text_dim": 4, "visual_dim": 6})


def test_factory_and_historical_cosi_model_remain_usable() -> None:
    cfg = _cfg()
    built = build_model(cfg, {"input_dim": 10, "text_dim": 4, "visual_dim": 6})
    assert isinstance(built, SSIMAGV3)

    old_cfg = OmegaConf.create(
        {
            "model": {
                "name": "cosi_mag_final",
                "hidden_dim": 8,
                "dropout": 0.0,
                "norm": "layernorm",
                "max_order": 3,
                "num_layers": 3,
                "diffusion_add_self_loops": True,
                "edge_metric": "learned_diag_cos",
                "edge_weight_min": 0.1,
                "edge_weight_temperature": 0.35,
                "eps": 1e-8,
                "filter_rank": 2,
                "multihop_state": "anchored",
                "multihop_response": "cumulative",
                "multihop_anchor_alpha": 0.1,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "hop_interaction_layers": 1,
                "hop_interaction_heads": 1,
                "hop_interaction_dropout": 0.0,
                "hop_interaction_order_embedding": True,
                "hop_interaction_gate_init": 0.1,
                "relation_descriptor_eps": 1e-6,
                "relation_bias_init": 0.1,
                "relation_init_seed": 20260921,
                "relation_init_std": 0.02,
                "fusion": "concat_residual_mlp",
            }
        }
    )
    old_model = CoSIMAGFinal(old_cfg, {"input_dim": 10, "text_dim": 4, "visual_dim": 6})
    x, edge_index = _inputs()
    z, _, _, _, _ = old_model(x, edge_index)
    assert z.shape == (6, 8)
