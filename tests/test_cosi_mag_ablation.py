from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.cosi_mag_ablation import CoSIMAGAblation
from src.models.cosi_mag_final import CoSIMAGFinal
from src.tasks.lp import _resolve_lp_num_neighbors


def _cfg(mode: str = "full", alpha: float | None = None):
    if alpha is None:
        alpha = 0.0 if mode == "no_semantic_anchor" else 0.1
    return OmegaConf.create(
        {
            "model": {
                "name": "cosi_mag_ablation",
                "ablation_mode": mode,
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
                "multihop_anchor_alpha": alpha,
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
    generator = torch.Generator().manual_seed(71)
    x = torch.randn(5, 6, generator=generator)
    # Node 4 is isolated; nodes 0..3 have physical edges.
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 0, 2]], dtype=torch.long
    )
    return x, edge_index


def _make_model(mode: str, alpha: float | None = None) -> CoSIMAGAblation:
    return CoSIMAGAblation(
        _cfg(mode, alpha),
        {"input_dim": 6, "num_nodes": 5, "text_dim": 3, "visual_dim": 3},
    )


def _assert_tree_exact(left, right) -> None:
    assert left.keys() == right.keys()
    for key in left:
        left_value, right_value = left[key], right[key]
        if isinstance(left_value, list):
            assert len(left_value) == len(right_value)
            for a, b in zip(left_value, right_value, strict=True):
                assert torch.equal(a, b), key
        elif isinstance(left_value, torch.Tensor):
            assert torch.equal(left_value, right_value), key
        else:
            assert left_value == right_value, key


def test_full_mode_is_machine_exact_with_final_model() -> None:
    x, edge_index = _inputs()
    torch.manual_seed(100)
    final = CoSIMAGFinal(
        _cfg("full"),
        {"input_dim": 6, "num_nodes": 5, "text_dim": 3, "visual_dim": 3},
    ).eval()
    torch.manual_seed(900)
    ablation = _make_model("full").eval()
    ablation.load_state_dict(final.state_dict(), strict=True)

    final_components = final._encode_components(
        x, edge_index, capture_attention=True
    )
    ablation_components = ablation._encode_components(
        x, edge_index, capture_attention=True
    )
    _assert_tree_exact(final_components, ablation_components)
    assert torch.equal(final(x, edge_index)[0], ablation(x, edge_index)[0])
    assert ablation.requires_full_lp_sampler_depth is True


def test_no_mrc_uses_uniform_physical_weights_and_zero_context_including_isolated_nodes() -> None:
    x, edge_index = _inputs()
    model = _make_model("no_mrc").eval()
    components = model._encode_components(x, edge_index, capture_attention=True)

    assert torch.equal(
        components["relation_weight_text"], torch.ones(edge_index.size(1))
    )
    assert torch.equal(
        components["relation_weight_visual"], torch.ones(edge_index.size(1))
    )
    assert torch.equal(
        components["normalized_edge_index_text"],
        components["normalized_edge_index_visual"],
    )
    assert torch.equal(
        components["normalized_edge_weight_text"],
        components["normalized_edge_weight_visual"],
    )
    assert torch.count_nonzero(components["local_relation_context_text"]) == 0
    assert torch.count_nonzero(components["local_relation_context_visual"]) == 0
    assert components["local_relation_context_text"][4].item() == 0.0
    assert components["local_relation_context_visual"][4].item() == 0.0

    original_z = components["z"].detach().clone()
    with torch.no_grad():
        for name in (
            "metric_theta_text",
            "metric_theta_visual",
            "relation_beta_raw_text",
            "relation_beta_raw_visual",
            "theta_relation_scale_text",
            "theta_relation_scale_visual",
        ):
            parameter = dict(model.named_parameters())[name]
            parameter.copy_(torch.randn_like(parameter) * 20.0)
    changed_components = model._encode_components(x, edge_index)
    assert torch.equal(original_z, changed_components["z"])


def test_no_semantic_anchor_sets_recurrence_and_global_prior_basis() -> None:
    x, edge_index = _inputs()
    model = _make_model("no_semantic_anchor").eval()
    full = _make_model("full").eval()
    assert model.multihop_anchor_alpha == 0.0
    prior = model._make_global_prior()
    expected_gamma = model._monomial_to_anchored_cumulative(prior, 0.0)
    assert torch.equal(model.gamma_global.detach(), expected_gamma)
    full_gamma = full._monomial_to_anchored_cumulative(
        full._make_global_prior(), full.multihop_anchor_alpha
    )
    assert not torch.equal(expected_gamma, full_gamma)

    components = model._encode_components(x, edge_index)
    for modality in ("text", "visual"):
        states = components[f"states_{modality}"]
        normalized_index = components[f"normalized_edge_index_{modality}"]
        normalized_weight = components[f"normalized_edge_weight_{modality}"]
        for order in range(1, model.max_order + 1):
            expected = model._propagate_once(
                states[order - 1], normalized_index, normalized_weight
            )
            assert torch.equal(states[order], expected)

    try:
        _make_model("no_semantic_anchor", alpha=0.1)
    except ValueError as error:
        assert "multihop_anchor_alpha=0.0" in str(error)
    else:
        raise AssertionError("no_semantic_anchor accepted alpha=0.1")


def _is_rcmi_parameter(name: str) -> bool:
    return (
        name.startswith("hop_order_embedding_")
        or name.startswith("hop_layers_")
        or name.startswith("theta_hop_gate_")
        or name.startswith("relation_beta_raw_")
        or name.startswith("theta_relation_scale_")
        or name.startswith("node_proj_")
        or name.startswith("node_vector_")
        or name in {"gamma_global", "delta_gamma_text", "delta_gamma_visual"}
    )


def test_no_rcmi_uniform_pool_is_invariant_to_all_rcmi_parameters_and_has_no_grads() -> None:
    x, edge_index = _inputs()
    model = _make_model("no_rcmi").eval()
    components = model._encode_components(x, edge_index)
    for modality in ("text", "visual"):
        expected = torch.stack(components[f"states_{modality}"], dim=1).mean(dim=1)
        assert torch.equal(components[f"z_{modality}"], expected)

    inactive = [(name, parameter) for name, parameter in model.named_parameters() if _is_rcmi_parameter(name)]
    assert inactive
    original_z = components["z"].detach().clone()
    with torch.no_grad():
        for _, parameter in inactive:
            parameter.copy_(torch.randn_like(parameter) * 10.0)
    assert torch.equal(original_z, model._encode_components(x, edge_index)["z"])

    model.train()
    model.zero_grad(set_to_none=True)
    model(x, edge_index)[0].sum().backward()
    for name, parameter in inactive:
        assert parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0, name


def test_lp_sampler_depth_is_extended_to_all_three_orders_for_each_variant() -> None:
    cfg = OmegaConf.create(
        {
            "task": {"num_neighbors": [5, 5]},
            "model": {"num_layers": 3},
        }
    )
    for mode in ("no_mrc", "no_semantic_anchor", "no_rcmi"):
        model = _make_model(mode)
        assert model.requires_full_lp_sampler_depth is True
        assert _resolve_lp_num_neighbors(cfg, model) == [5, 5, 5]


def test_ablation_mode_and_anchor_alpha_are_fail_fast_constraints() -> None:
    try:
        _make_model("no_mrc", alpha=0.0)
    except ValueError as error:
        assert "no_mrc requires model.multihop_anchor_alpha=0.1" in str(error)
    else:
        raise AssertionError("no_mrc accepted an alpha that changes the frozen baseline")

    bad_cfg = _cfg("not_a_variant")
    try:
        CoSIMAGAblation(
            bad_cfg,
            {"input_dim": 6, "num_nodes": 5, "text_dim": 3, "visual_dim": 3},
        )
    except ValueError as error:
        assert "ablation_mode must be one of" in str(error)
    else:
        raise AssertionError("unknown ablation_mode was accepted")
