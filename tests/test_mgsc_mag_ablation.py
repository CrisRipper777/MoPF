"""Integrity and functional tests for the canonical MGSC ablations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.mgsc_mag import MGSCMAG
from src.models.mgsc_mag_ablation import MGSCMAGFunctionalAblation


DATA_INFO = {
    "input_dim": 10,
    "num_nodes": 7,
    "num_classes": 3,
    "text_dim": 4,
    "visual_dim": 6,
}


def _cfg(ablation: str = "full"):
    return OmegaConf.create(
        {
            "model": {
                "name": "mgsc_mag_ablation",
                "version": "mgsc_mag_functional_ablation",
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
                "ablation": ablation,
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


def test_full_ablation_matches_canonical_p2_parameter_shapes_and_forward():
    torch.manual_seed(7)
    reference = MGSCMAG(_cfg("full"), DATA_INFO)
    torch.manual_seed(7)
    candidate = MGSCMAGFunctionalAblation(_cfg("full"), DATA_INFO)
    assert set(reference.state_dict()) == set(candidate.state_dict())
    assert all(
        reference.state_dict()[name].shape == candidate.state_dict()[name].shape
        for name in reference.state_dict()
    )
    candidate.load_state_dict(reference.state_dict(), strict=True)
    x, edge_index = _graph()
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        z_reference = reference(x, edge_index)[0]
        z_candidate = candidate(x, edge_index)[0]
    assert torch.max(torch.abs(z_reference - z_candidate)).item() <= 1e-6


def test_all_ablation_variants_are_finite_and_backward_safe():
    x, edge_index = _graph()
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    for name in (
        "uniform_relations",
        "global_context_gate",
        "terminal_state_only",
        "uniform_integration",
        "no_cross_order_interaction",
        "attribute_only",
    ):
        torch.manual_seed(11)
        model = MGSCMAGFunctionalAblation(_cfg(name), DATA_INFO)
        classifier = torch.nn.Linear(model.out_dim, DATA_INFO["num_classes"])
        logits = classifier(model(x, edge_index)[0])
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        assert torch.isfinite(model.analysis_multi_order(x, edge_index)["z"]).all()


def test_uniform_relations_are_exactly_uniform_before_normalization():
    model = MGSCMAGFunctionalAblation(_cfg("uniform_relations"), DATA_INFO)
    x, edge_index = _graph()
    analysis = model.analysis_relation_calibration(x, edge_index)
    assert torch.allclose(
        analysis["relation_weight_text"],
        torch.ones_like(analysis["relation_weight_text"]),
    )
    assert torch.allclose(
        analysis["relation_weight_visual"],
        torch.ones_like(analysis["relation_weight_visual"]),
    )


def test_global_gate_is_node_invariant_and_initialized_at_point_nine():
    model = MGSCMAGFunctionalAblation(_cfg("global_context_gate"), DATA_INFO)
    x, edge_index = _graph()
    analysis = model.analysis_multi_order(x, edge_index)
    for modality in ("text", "visual"):
        bank = torch.stack(analysis[f"context_gate_{modality}"], dim=1)
        assert torch.allclose(bank, torch.full_like(bank, 0.9), atol=1e-6)
        assert torch.allclose(
            analysis[f"global_context_gate_{modality}"],
            torch.full((3,), 0.9),
            atol=1e-6,
        )


def test_composition_and_interaction_variants_use_declared_paths():
    x, edge_index = _graph()
    terminal = MGSCMAGFunctionalAblation(_cfg("terminal_state_only"), DATA_INFO)
    terminal_output = terminal.analysis_multi_order(x, edge_index)
    assert torch.allclose(terminal_output["z_text"], terminal_output["interacted_states_text"][-1])
    assert torch.allclose(
        terminal_output["z_visual"], terminal_output["interacted_states_visual"][-1]
    )

    uniform = MGSCMAGFunctionalAblation(_cfg("uniform_integration"), DATA_INFO)
    uniform_output = uniform.analysis_multi_order(x, edge_index)
    assert torch.allclose(
        uniform_output["z_text"],
        torch.stack(uniform_output["interacted_states_text"], dim=1).mean(dim=1),
    )

    no_interaction = MGSCMAGFunctionalAblation(
        _cfg("no_cross_order_interaction"), DATA_INFO
    )
    no_interaction_output = no_interaction.analysis_multi_order(x, edge_index)
    for modality in ("text", "visual"):
        for raw, interacted in zip(
            no_interaction_output[f"states_{modality}"],
            no_interaction_output[f"interacted_states_{modality}"],
        ):
            assert torch.allclose(raw, interacted)
        attention = no_interaction_output[f"attention_{modality}"]
        assert torch.allclose(
            attention,
            torch.eye(4).unsqueeze(0).expand(DATA_INFO["num_nodes"], -1, -1),
        )


def test_isolated_nodes_remain_finite():
    model = MGSCMAGFunctionalAblation(_cfg("uniform_relations"), DATA_INFO)
    x, edge_index = _graph()
    edge_index = edge_index[:, :4]
    output = model.analysis_multi_order(x, edge_index)
    assert torch.isfinite(output["z"]).all()
