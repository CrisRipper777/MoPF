from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.cosi_mag_final import CoSIMAGFinal
from src.models.predictor import LinkPredictor
from src.tasks.common import build_optimizer
from src.tasks.lp import (
    _evaluate_split,
    _raise_if_nonfinite,
    _resolve_lp_num_neighbors,
)


def _cfg():
    return OmegaConf.create(
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
                "eps": 1.0e-8,
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
                "relation_descriptor_eps": 1.0e-6,
                "relation_bias_init": 0.1,
                "relation_init_seed": 20260921,
                "relation_init_std": 0.02,
                "fusion": "concat_residual_mlp",
            },
            "task": {
                "optimizer": "adam",
                "lr": 1.0e-3,
                "weight_decay": 1.0e-5,
                "num_neighbors": [5, 5],
            },
        }
    )


def _model() -> CoSIMAGFinal:
    torch.manual_seed(7)
    return CoSIMAGFinal(
        _cfg(),
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 0.0, 0.0, 0.0],
        [-5.0e-23, 2.0e-23, 1.0e-23, -1.0e-23],
    ],
)
def test_relation_beta_zero_and_tiny_forward_backward_are_finite(values) -> None:
    model = _model()
    with torch.no_grad():
        model.relation_beta_raw_text.copy_(torch.tensor(values))

    beta_hat, relation_scale = model._relation_to_order_profile("text")
    objective = (beta_hat * torch.tensor([1.0, -2.0, 3.0, -4.0])).sum()
    objective = objective + relation_scale
    objective.backward()

    assert torch.isfinite(beta_hat).all()
    assert torch.isfinite(relation_scale)
    assert model.relation_beta_raw_text.grad is not None
    assert torch.isfinite(model.relation_beta_raw_text.grad).all()


def test_relation_beta_normal_rms_matches_previous_forward_formula() -> None:
    model = _model()
    values = torch.tensor([0.03, -0.02, 0.015, -0.011])
    with torch.no_grad():
        model.relation_beta_raw_text.copy_(values)

    actual, _ = model._relation_to_order_profile("text")
    centered = values - values.mean()
    previous = centered / centered.square().mean().sqrt().clamp_min(
        model.relation_descriptor_eps
    )

    torch.testing.assert_close(actual, previous, rtol=1.0e-6, atol=1.0e-7)


def test_relation_beta_parameters_are_the_only_zero_weight_decay_parameters() -> None:
    model = _model()
    head = nn.Linear(model.out_dim, 1)
    optimizer = build_optimizer(
        list(model.parameters()) + list(head.parameters()), _cfg(), model=model
    )
    group_by_parameter = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    beta_parameters = {
        id(model.relation_beta_raw_text),
        id(model.relation_beta_raw_visual),
    }

    assert group_by_parameter[id(model.relation_beta_raw_text)]["weight_decay"] == 0.0
    assert group_by_parameter[id(model.relation_beta_raw_visual)]["weight_decay"] == 0.0
    for parameter in list(model.parameters()) + list(head.parameters()):
        if id(parameter) not in beta_parameters:
            assert group_by_parameter[id(parameter)]["weight_decay"] == 1.0e-5


def test_cosi_uses_full_sampler_depth_without_changing_default_two_hop_models() -> None:
    cfg = _cfg()
    assert _resolve_lp_num_neighbors(cfg, _model()) == [5, 5, 5]
    assert _resolve_lp_num_neighbors(cfg, nn.Identity()) == [5, 5]


def test_lp_nonfinite_checks_raise_before_ranking_or_optimization() -> None:
    with pytest.raises(FloatingPointError, match="training logits"):
        _raise_if_nonfinite(torch.tensor([0.0, float("nan")]), "training logits")

    predictor = LinkPredictor(in_dim=2, hidden_dim=2, num_layers=1, dropout=0.0)
    with torch.no_grad():
        predictor.net[0].weight.fill_(float("nan"))
    split = {
        "source_node": torch.tensor([0]),
        "target_node": torch.tensor([1]),
        "target_node_neg": torch.tensor([[2, 3]]),
    }
    with pytest.raises(FloatingPointError, match="evaluation positive scores"):
        _evaluate_split(
            torch.ones(4, 2),
            predictor,
            split,
            torch.device("cpu"),
            batch_size=1,
        )
