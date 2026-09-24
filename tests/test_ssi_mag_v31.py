from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.factory import build_model
from src.models.ssi_mag_v31 import SSIMAGV31


def _cfg(*, max_order: int = 3):
    return OmegaConf.create(
        {
            "ablation": "full",
            "model": {
                "name": "ssi_mag_v31",
                "hidden_dim": 8,
                "dropout": 0.0,
                "norm": "layernorm",
                "max_order": max_order,
                "num_layers": max_order,
                "diffusion_add_self_loops": True,
                "relation_rank": 3,
                "relation_strength_init": 0.05,
                "relation_edge_chunk_size": 2,
                "relation_init_seed": 20260923,
                "relation_init_std": 0.02,
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
                "relation_filter_scale_init": 0.10,
                "fusion": "concat_residual_mlp",
                "eps": 1.0e-8,
            },
        }
    )


def _model(*, max_order: int = 3) -> SSIMAGV31:
    torch.manual_seed(17)
    return SSIMAGV31(
        _cfg(max_order=max_order),
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    x = torch.randn(6, 10)
    # Bidirectional physical pairs, one explicit self-loop, and isolated 4/5.
    edge_index = torch.tensor(
        [
            [0, 1, 1, 0, 1, 2, 2, 1, 3],
            [1, 0, 2, 1, 2, 1, 2, 2, 3],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def test_r1_scorer_zero_init_and_initial_unit_weights() -> None:
    x, edge_index = _inputs()
    model = _model()
    for modality in ("text", "visual"):
        scorer = getattr(model, f"relation_scorer_{modality}")
        assert torch.equal(scorer.weight, torch.zeros_like(scorer.weight))
        assert torch.equal(scorer.bias, torch.zeros_like(scorer.bias))
        h0 = getattr(model, f"{modality}_proj")(x[:, :4] if modality == "text" else x[:, 4:])
        relation = model._relation_edge_outputs(h0, edge_index, modality)
        assert torch.equal(relation["relation_residual"], torch.zeros(edge_index.size(1)))
        assert torch.equal(
            relation["relation_weight"], torch.ones(edge_index.size(1))
        )


def test_r1_endpoint_symmetry_and_self_loop_exclusion_from_mu_c() -> None:
    x, edge_index = _inputs()
    model = _model()
    with torch.no_grad():
        model.relation_scorer_text.weight.copy_(torch.tensor([[0.2, -0.1, 0.05]]))
        model.relation_scorer_text.bias.fill_(0.03)
    h0 = model.text_proj(x[:, :4])
    forward = torch.tensor([[0, 1, 1], [1, 2, 2]], dtype=torch.long)
    reverse = forward.flip(0)
    first = model._relation_descriptor(h0, forward, "text")
    second = model._relation_descriptor(h0, reverse, "text")
    torch.testing.assert_close(first["relation_score"], second["relation_score"], rtol=0.0, atol=0.0)

    without_self = edge_index[:, edge_index[0] != edge_index[1]]
    with_self = model._relation_descriptor(h0, edge_index, "text")
    without = model._relation_descriptor(h0, without_self, "text")
    torch.testing.assert_close(with_self["relation_mu"], without["relation_mu"], rtol=0.0, atol=0.0)
    relation = model._relation_edge_outputs(h0, edge_index, "text")
    c = model._local_adaptation(
        edge_index,
        relation["relation_residual"],
        relation["beta"],
        relation["nonself_mask"],
        x.size(0),
    )
    relation_without = model._relation_edge_outputs(h0, without_self, "text")
    c_without = model._local_adaptation(
        without_self,
        relation_without["relation_residual"],
        relation_without["beta"],
        relation_without["nonself_mask"],
        x.size(0),
    )
    torch.testing.assert_close(c, c_without, rtol=0.0, atol=0.0)
    assert c[4].item() == 0.0 and c[5].item() == 0.0
    assert torch.isfinite(c).all()

    self_mask = edge_index[0] == edge_index[1]
    relation = model._relation_edge_outputs(h0, edge_index, "text")
    assert self_mask.any()
    assert torch.equal(
        relation["relation_residual"][self_mask],
        torch.zeros_like(relation["relation_residual"][self_mask]),
    )
    assert torch.equal(
        relation["relation_weight"][self_mask],
        torch.ones_like(relation["relation_weight"][self_mask]),
    )


def test_bidirectional_edges_match_single_undirected_neighborhood_mean() -> None:
    x, _ = _inputs()
    model = _model()
    with torch.no_grad():
        model.relation_scorer_text.weight.copy_(torch.tensor([[0.2, -0.1, 0.05]]))
        model.relation_scorer_text.bias.fill_(0.03)
    h0 = model.text_proj(x[:, :4])
    single = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    bidirectional = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    single_stats = model._relation_descriptor(h0, single, "text")
    bi_stats = model._relation_descriptor(h0, bidirectional, "text")
    torch.testing.assert_close(bi_stats["relation_mu"], single_stats["relation_mu"], rtol=0.0, atol=0.0)
    single_relation = model._relation_edge_outputs(h0, single, "text")
    bi_relation = model._relation_edge_outputs(h0, bidirectional, "text")
    single_c = model._local_adaptation(
        single, single_relation["relation_residual"], single_relation["beta"],
        single_relation["nonself_mask"], x.size(0)
    )
    bi_c = model._local_adaptation(
        bidirectional, bi_relation["relation_residual"], bi_relation["beta"],
        bi_relation["nonself_mask"], x.size(0)
    )
    torch.testing.assert_close(bi_c, single_c, rtol=0.0, atol=0.0)


def test_initial_operator_matches_raw_unit_operator_and_relation_off() -> None:
    x, edge_index = _inputs()
    model = _model()
    model.eval()
    analysis = model.analysis(x, edge_index)
    for modality in ("text", "visual"):
        raw_index, raw_weight = model._normalized_operator(
            edge_index, torch.ones(edge_index.size(1)), x.size(0), x.dtype
        )
        torch.testing.assert_close(
            analysis[f"normalized_edge_index_{modality}"], raw_index, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            analysis[f"normalized_edge_weight_{modality}"], raw_weight, rtol=0.0, atol=0.0
        )
    off = model.analysis_intervention(x, edge_index, relation="off")
    for modality in ("text", "visual"):
        assert torch.equal(off[f"relation_weight_{modality}"], torch.ones(edge_index.size(1)))
        assert torch.equal(off[f"c_{modality}"], torch.zeros(x.size(0)))


def test_stabilized_prior_formula_and_nonzero_norm() -> None:
    x, edge_index = _inputs()
    model = _model()
    analysis = model.analysis(x, edge_index)
    for modality in ("text", "visual"):
        h0 = analysis[f"h0_{modality}"]
        prior_norm = getattr(model, f"semantic_prior_norm_{modality}")
        prior_vector = getattr(model, f"semantic_prior_vector_{modality}")
        expected = (prior_norm(h0) * prior_vector).sum(-1) / prior_vector.norm(p=2).clamp_min(model.eps)
        torch.testing.assert_close(analysis[f"p_{modality}"], expected, rtol=0.0, atol=0.0)
        assert prior_vector.norm().item() > 0.0
        assert torch.isfinite(prior_vector.norm())
        assert not hasattr(model, f"semantic_rho_c_{modality}")
        assert not hasattr(model, f"reference_filter_scale_{modality}")


def test_initial_alpha_and_semantic_state_formula() -> None:
    x, edge_index = _inputs()
    model = _model()
    analysis = model.analysis(x, edge_index)
    for modality in ("text", "visual"):
        for q, s, alpha in zip(
            analysis[f"Q_{modality}"],
            analysis[f"S_{modality}"][1:],
            analysis[f"alpha_{modality}"],
            strict=True,
        ):
            torch.testing.assert_close(alpha, torch.full_like(alpha, 0.1), rtol=0.0, atol=1e-7)
            expected = 0.9 * q + 0.1 * analysis[f"h0_{modality}"]
            torch.testing.assert_close(s, expected, rtol=0.0, atol=1e-7)


def test_context_and_signed_filter_formula_exports_have_expected_shapes() -> None:
    x, edge_index = _inputs()
    model = _model()
    analysis = model.analysis(x, edge_index)
    assert all("reference_residual" not in key for key in analysis)
    for modality in ("text", "visual"):
        assert len(analysis[f"D_{modality}"]) == 4
        assert len(analysis[f"context_change_injection_{modality}"]) == 4
        assert analysis[f"eta_{modality}"].shape == (6, 4)
        expected_eta = (
            analysis[f"gamma_{modality}"].unsqueeze(0)
            + analysis[f"delta_gamma_{modality}"].unsqueeze(0)
            + analysis[f"delta_content_{modality}"]
            + analysis[f"relation_filter_residual_{modality}"]
        )
        torch.testing.assert_close(analysis[f"eta_{modality}"], expected_eta, rtol=0.0, atol=0.0)
        expected_z = model._compose(analysis[f"S_tilde_{modality}"], analysis[f"eta_{modality}"])
        torch.testing.assert_close(expected_z, analysis[f"z_{modality}"], rtol=0.0, atol=0.0)


def test_interaction_off_is_exact_identity_and_final_uses_interacted_context() -> None:
    x, edge_index = _inputs()
    model = _model()
    normal = model.analysis(x, edge_index)
    off = model.analysis_intervention(x, edge_index, interaction="off")
    for modality in ("text", "visual"):
        for interacted, state in zip(
            normal[f"S_tilde_{modality}"], normal[f"S_{modality}"], strict=True
        ):
            assert not torch.equal(interacted, state) or normal[f"interaction_gate_{modality}"].item() == 0.0
        for interacted, state in zip(
            off[f"S_tilde_{modality}"], off[f"S_{modality}"], strict=True
        ):
            assert torch.equal(interacted, state)
        expected = model._compose(normal[f"S_tilde_{modality}"], normal[f"eta_{modality}"])
        torch.testing.assert_close(expected, normal[f"z_{modality}"], rtol=0.0, atol=0.0)
        original = model._compose(normal[f"S_{modality}"], normal[f"eta_{modality}"])
        assert not torch.allclose(expected, original, rtol=1e-6, atol=1e-7)


def test_gradient_flow_and_zero_scorer_leaves_zero_after_optimizer_step() -> None:
    x, edge_index = _inputs()
    model = _model()
    head = nn.Linear(model.out_dim, 3)
    labels = torch.tensor([0, 1, 2, 1, 0, 2])
    z, _, _, aux_loss, _ = model(x, edge_index)
    loss = nn.CrossEntropyLoss()(head(z), labels) + aux_loss
    loss.backward()
    names = (
        "relation_scorer_text.weight",
        "relation_scorer_text.bias",
        "theta_beta_text",
        "semantic_bias_text",
        "semantic_rho_p_text",
        "semantic_rho_d_text",
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
        if name != "theta_beta_text":
            assert gradient.abs().sum().item() > 0.0, name

    # At the exact neutral initialization, d exp(beta*a)/d beta is zero because
    # a=0. Once the scorer moves, theta_beta must receive a live gradient.
    scorer_before = model.relation_scorer_text.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    assert torch.equal(scorer_before, torch.zeros_like(scorer_before))
    assert not torch.equal(model.relation_scorer_text.weight, torch.zeros_like(model.relation_scorer_text.weight))
    optimizer.zero_grad(set_to_none=True)
    z, _, _, aux_loss, _ = model(x, edge_index)
    (nn.CrossEntropyLoss()(head(z), labels) + aux_loss).backward()
    assert model.theta_beta_text.grad is not None
    assert torch.isfinite(model.theta_beta_text.grad).all()
    assert model.theta_beta_text.grad.abs().item() > 0.0
    for parameter in (model.relation_proj_text.weight, model.semantic_prior_vector_text):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_text_visual_paths_are_independent_until_late_fusion() -> None:
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


def test_forward_inference_factory_and_finite_analysis() -> None:
    x, edge_index = _inputs()
    model = _model()
    output = model(x, edge_index)
    assert len(output) == 5 and output[0].shape == (6, 8)
    assert torch.isfinite(output[0]).all()
    inferred = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)
    assert inferred.shape == (6, 8) and inferred.device.type == "cpu"
    built = build_model(_cfg(), {"input_dim": 10, "text_dim": 4, "visual_dim": 6})
    assert isinstance(built, SSIMAGV31)
    analysis = model.analysis(x, edge_index)
    for value in analysis.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
        elif isinstance(value, list):
            for item in value:
                if torch.is_tensor(item):
                    assert torch.isfinite(item).all()
