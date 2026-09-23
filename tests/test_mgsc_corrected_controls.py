"""Integrity tests for strict raw-state and plain-backbone controls."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.mgsc_mag import MGSCMAG
from src.models.mgsc_mag_corrected_controls import MGSCMAGCorrectedControls


DATA_INFO = {
    "input_dim": 10,
    "num_nodes": 7,
    "num_classes": 3,
    "text_dim": 4,
    "visual_dim": 6,
}


def _cfg(control: str = "full"):
    return OmegaConf.create(
        {
            "model": {
                "name": "mgsc_mag_corrected_controls",
                "version": "mgsc_mag_corrected_controls",
                "hidden_dim": 8,
                "dropout": 0.0,
                "norm": "layernorm",
                "max_order": 3,
                "diffusion_add_self_loops": True,
                "edge_metric": "learned_diag_cos",
                "metric_init_seed": 20260910,
                "edge_weight_min": 0.1,
                "edge_weight_temperature": 0.35,
                "eps": 1e-8,
                "filter_rank": 4,
                "multihop_state": "anchored",
                "multihop_response": "cumulative",
                "multihop_anchor_alpha": 0.1,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "hop_interaction_layers": 1,
                "hop_interaction_heads": 1,
                "hop_interaction_dropout": 0.0,
                "hop_interaction_order_embedding": True,
                "hop_interaction_gate_init": 0.10,
                "relation_descriptor_eps": 1e-6,
                "relation_bias_init": 0.10,
                "relation_init_seed": 20260921,
                "relation_init_std": 0.02,
                "fusion": "concat_residual_mlp",
                "adaptive_context_gate": True,
                "context_gate_order_dim": 4,
                "context_gate_hidden_dim": 10,
                "direct_interacted_integration": True,
                "use_legacy_relation_order_bias": True,
                "control": control,
            }
        }
    )


def _graph():
    torch.manual_seed(123)
    x = torch.randn(DATA_INFO["num_nodes"], DATA_INFO["input_dim"])
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long
    )
    return x, edge_index


def test_full_matches_canonical_p2_exactly():
    torch.manual_seed(7)
    reference = MGSCMAG(_cfg("full"), DATA_INFO)
    torch.manual_seed(7)
    candidate = MGSCMAGCorrectedControls(_cfg("full"), DATA_INFO)
    assert set(reference.state_dict()) == set(candidate.state_dict())
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x, edge_index = _graph()
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        z_reference = reference(x, edge_index)[0]
        z_candidate = candidate(x, edge_index)[0]
    assert torch.max(torch.abs(z_reference - z_candidate)).item() <= 1e-6


def test_raw_terminal_is_raw_terminal_and_bypasses_interaction():
    x, edge_index = _graph()
    model = MGSCMAGCorrectedControls(_cfg("raw_terminal_only"), DATA_INFO)
    model._cross_order_interaction = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("interaction called"))
    analysis = model.analysis_multi_order(x, edge_index)
    assert analysis["control_readout_source"] == "raw_terminal"
    assert torch.allclose(analysis["z_text"], analysis["states_text"][-1])
    assert torch.allclose(analysis["z_visual"], analysis["states_visual"][-1])


def test_raw_bank_mean_is_raw_equal_mean_and_does_not_read_eta():
    x, edge_index = _graph()
    model = MGSCMAGCorrectedControls(_cfg("raw_bank_mean"), DATA_INFO)
    model._cross_order_interaction = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("interaction called"))
    with torch.no_grad():
        model.gamma_global.fill_(float("nan"))
        model.delta_gamma_text.fill_(float("nan"))
        model.delta_gamma_visual.fill_(float("nan"))
    analysis = model.analysis_multi_order(x, edge_index)
    assert analysis["control_readout_source"] == "raw_bank_mean"
    assert torch.allclose(
        analysis["z_text"], torch.stack(analysis["states_text"], dim=1).mean(dim=1)
    )
    assert torch.allclose(
        analysis["z_visual"], torch.stack(analysis["states_visual"], dim=1).mean(dim=1)
    )
    assert torch.isfinite(analysis["z"]).all()


def test_plain_backbone_has_uniform_relations_fixed_gate_and_same_order_count():
    x, edge_index = _graph()
    model = MGSCMAGCorrectedControls(_cfg("plain_multi_order_backbone"), DATA_INFO)
    analysis = model.analysis_multi_order(x, edge_index)
    assert torch.allclose(analysis["relation_weight_text"], torch.ones(edge_index.size(1)))
    assert torch.allclose(analysis["relation_weight_visual"], torch.ones(edge_index.size(1)))
    assert len(analysis["states_text"]) == model.max_order + 1
    assert len(analysis["states_visual"]) == model.max_order + 1
    for modality in ("text", "visual"):
        gates = torch.stack(analysis[f"context_gate_{modality}"], dim=1)
        assert torch.allclose(gates, torch.full_like(gates, 0.9), atol=1e-6)


def test_plain_keeps_refinement_fusion_and_has_natural_lower_parameter_count():
    full = MGSCMAGCorrectedControls(_cfg("full"), DATA_INFO)
    plain = MGSCMAGCorrectedControls(_cfg("plain_multi_order_backbone"), DATA_INFO)
    for name in (
        "text_refine_mlp", "visual_refine_mlp", "text_refine_norm", "visual_refine_norm",
        "fusion_skip", "fusion_mlp", "output_norm",
    ):
        assert hasattr(full, name) and hasattr(plain, name)
        assert sum(parameter.numel() for parameter in getattr(full, name).parameters()) == sum(
            parameter.numel() for parameter in getattr(plain, name).parameters()
        )
    assert sum(parameter.numel() for parameter in plain.parameters()) < sum(
        parameter.numel() for parameter in full.parameters()
    )


def test_all_controls_are_finite_backward_safe_and_handle_isolated_nodes():
    x, edge_index = _graph()
    edge_index = edge_index[:, :4]
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    for control in (
        "full", "raw_terminal_only", "raw_bank_mean", "plain_multi_order_backbone"
    ):
        torch.manual_seed(11)
        model = MGSCMAGCorrectedControls(_cfg(control), DATA_INFO)
        classifier = torch.nn.Linear(model.out_dim, DATA_INFO["num_classes"])
        logits = classifier(model(x, edge_index)[0])
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(model.analysis_multi_order(x, edge_index)["z"]).all()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

