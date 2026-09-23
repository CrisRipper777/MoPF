from __future__ import annotations

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.cosi_mag_ablation import CoSIMAGAblation
from src.models.cssi_response_probe import CSSIResponseProbe


def _cfg() -> object:
    return OmegaConf.create(
        {
            "model": {
                "name": "cssi_response_probe",
                "ablation_mode": "all_plain",
                "hidden_dim": 8,
                "dropout": 0.0,
                "norm": "layernorm",
                "max_order": 3,
                "num_layers": 3,
                "diffusion_add_self_loops": True,
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
            }
        }
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(7, 10, generator=generator)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 5], [1, 0, 2, 1, 3, 2, 5, 4]], dtype=torch.long
    )
    return x, edge_index


def _model() -> CSSIResponseProbe:
    torch.manual_seed(17)
    return CSSIResponseProbe(
        _cfg(),
        {"input_dim": 10, "num_nodes": 7, "text_dim": 4, "visual_dim": 6},
    ).eval()


def test_all_plain_forward_path_is_uniform_multi_order_and_preserves_downstream_stack():
    x, edge_index = _inputs()
    model = _model()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
        output = model(x, edge_index)[0]
    for modality in ("text", "visual"):
        states = components[f"states_{modality}"]
        assert len(states) == 4
        assert torch.equal(
            components[f"z_{modality}"], torch.stack(states, dim=1).mean(dim=1)
        )
    assert torch.equal(output, components["z"])
    assert "interacted_states_text" not in components
    assert "eta_text" not in components
    assert torch.equal(
        components["normalized_edge_index_text"],
        components["normalized_edge_index_visual"],
    )
    assert torch.equal(
        components["normalized_edge_weight_text"],
        components["normalized_edge_weight_visual"],
    )


def test_text_visual_normalized_operator_equality_under_unit_physical_weights():
    x, edge_index = _inputs()
    model = _model()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
    assert torch.equal(
        components["relation_weight_text"],
        torch.ones_like(components["relation_weight_text"]),
    )
    assert torch.equal(
        components["relation_weight_visual"],
        torch.ones_like(components["relation_weight_visual"]),
    )
    assert torch.equal(
        components["normalized_edge_index_text"],
        components["normalized_edge_index_visual"],
    )
    assert torch.equal(
        components["normalized_edge_weight_text"],
        components["normalized_edge_weight_visual"],
    )


def test_alpha_zero_recurrence_and_response_reconstruction():
    x, edge_index = _inputs()
    model = _model()
    with torch.no_grad():
        components = model.response_components(x, edge_index)
    checks = components["equivalence_checks"]
    assert checks["operator_index_equal"]
    for modality in ("text", "visual"):
        assert checks[modality]["recurrence_max_abs_error"] <= 1.0e-5
        assert checks[modality]["response_reconstruction_max_abs_error"] <= 1.0e-5
        assert checks[modality]["operator_response_max_abs_error"] <= 1.0e-5
        assert checks[modality]["plain_representation_max_abs_error"] <= 1.0e-5


def test_plain_response_equivalence_and_leave_one_response_out_base_case():
    x, edge_index = _inputs()
    model = _model()
    classifier = torch.nn.Linear(8, 3)
    classifier.eval()
    with torch.no_grad():
        components = model.response_components(x, edge_index)
        base = model.leave_one_response_out_logits(classifier, components)
        ordinary = classifier(components["z"])
        for modality in ("text", "visual"):
            for order in (1, 2, 3):
                counterfactual = model.leave_one_response_out_logits(
                    classifier, components, modality=modality, order=order
                )
                assert counterfactual.shape == base.shape
    assert torch.allclose(base, ordinary, atol=1.0e-5, rtol=0.0)
    assert torch.isfinite(base).all()


def test_utility_sign_convention_is_counterfactual_minus_base():
    label = torch.tensor([0])
    base_logits = torch.tensor([[4.0, 0.0]])
    helpful_counterfactual = torch.tensor([[1.0, 0.0]])
    harmful_counterfactual = torch.tensor([[6.0, 0.0]])
    helpful = F.cross_entropy(helpful_counterfactual, label) - F.cross_entropy(
        base_logits, label
    )
    harmful = F.cross_entropy(harmful_counterfactual, label) - F.cross_entropy(
        base_logits, label
    )
    assert helpful.item() > 0.0
    assert harmful.item() < 0.0


def test_ce_and_margin_helpful_direction_agree_on_toy_example():
    label = torch.tensor([0])
    base_logits = torch.tensor([[4.0, 0.0]])
    removed_logits = torch.tensor([[1.0, 0.0]])
    ce_utility = F.cross_entropy(removed_logits, label) - F.cross_entropy(
        base_logits, label
    )
    base_margin = base_logits[:, 0] - base_logits[:, 1]
    removed_margin = removed_logits[:, 0] - removed_logits[:, 1]
    margin_utility = base_margin - removed_margin
    assert ce_utility.item() > 0.0
    assert margin_utility.item() > 0.0


def test_response_probe_adds_no_parameters_and_matches_canonical_all_plain_state_keys():
    probe = _model()
    canonical_cfg = _cfg()
    canonical_cfg.model.name = "cosi_mag_ablation"
    torch.manual_seed(17)
    canonical = CoSIMAGAblation(
        canonical_cfg,
        {"input_dim": 10, "num_nodes": 7, "text_dim": 4, "visual_dim": 6},
    ).eval()
    assert tuple(probe.state_dict()) == tuple(canonical.state_dict())
