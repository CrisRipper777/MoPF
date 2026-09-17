from __future__ import annotations

import subprocess
import types

import torch

from src.ablation import (
    CORE_STORY_ABLATIONS,
    effective_composition_mode,
    effective_edge_weight_mode,
    resolve_ablation,
)
from src.models.mopf import Model

from tests.test_mopf import _cfg, _graph


DATA_INFO = {"input_dim": 10, "num_nodes": 7, "text_dim": 4, "visual_dim": 6}

# Keep the Full-path regression independent of the commit that eventually
# contains the Core Story implementation.  Otherwise reading HEAD here would
# compare the modified model with itself after the user commits the changes.
FULL_REGRESSION_BASELINE_COMMIT = "97b57a02884d7ce3441b5500d000c41042e19fa3"


def _build(name: str = "full", **overrides):
    torch.manual_seed(123)
    cfg = _cfg(
        edge_weight_mode="learned_diag_cos",
        use_transport_residual=True,
        multihop_state_mode="anchored",
        multihop_response_mode="cumulative",
        multihop_anchor_alpha=0.1,
        **overrides,
    )
    cfg["ablation"] = name
    return Model(cfg, DATA_INFO)


def _graph_with_isolated_node() -> tuple[torch.Tensor, torch.Tensor]:
    x, edge_index = _graph()
    return torch.cat([x, torch.full((1, x.size(1)), 0.7)], dim=0), edge_index


def test_core_resolver_is_explicit_and_legacy_variant_is_not_reinterpreted() -> None:
    model_cfg = _cfg().model
    old = resolve_ablation("wo_learned_semantic_calibration")
    relation = resolve_ablation("wo_relation_calibration")
    adaptive = resolve_ablation("wo_adaptive_composition")

    assert effective_edge_weight_mode(model_cfg, old) == "separate_cos"
    assert effective_edge_weight_mode(model_cfg, relation) == "raw_uniform"
    assert effective_composition_mode(model_cfg, adaptive) == "uniform"
    assert effective_composition_mode(model_cfg, resolve_ablation("full")) == "adaptive"
    assert tuple(CORE_STORY_ABLATIONS) == (
        "wo_relation_calibration",
        "wo_semantic_anchor",
        "wo_adaptive_composition",
    )


def test_wo_relation_calibration_uses_uniform_weights_identical_operators_and_zero_context() -> None:
    x, edge_index = _graph_with_isolated_node()
    model = _build("wo_relation_calibration")
    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
        stats = model.conductance_stats(x, edge_index)

    edges = components["edges"]
    assert model.edge_weight_mode == "raw_uniform"
    assert torch.equal(edges["w_t"], torch.ones_like(edges["w_t"]))
    assert torch.equal(edges["w_v"], torch.ones_like(edges["w_v"]))
    assert torch.equal(stats["conductance_text"], torch.ones_like(stats["conductance_text"]))
    assert torch.equal(stats["conductance_visual"], torch.ones_like(stats["conductance_visual"]))
    assert torch.equal(stats["src"], edge_index[0])
    assert torch.equal(stats["dst"], edge_index[1])
    assert torch.equal(components["norm_t_index"], components["norm_v_index"])
    assert torch.equal(components["norm_t_weight"], components["norm_v_weight"])
    for modality in ("text", "visual"):
        context = components[f"transport_context_{modality}"]
        assert float(context.abs().max()) <= 1e-7


def test_wo_semantic_anchor_is_ordinary_state_bank_with_cumulative_responses() -> None:
    x, edge_index = _graph()
    model = _build("wo_semantic_anchor")
    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)

    assert model.multihop_state_mode == "ordinary"
    assert model.multihop_response_mode == "cumulative"
    for modality in ("text", "visual"):
        states = components[f"states_{modality}"]
        responses = components[f"responses_{modality}"]
        expected = model._propagation_bank(
            components[f"h_{modality}"],
            components["norm_t_index" if modality == "text" else "norm_v_index"],
            components["norm_t_weight" if modality == "text" else "norm_v_weight"],
        )
        assert all(torch.equal(actual, target) for actual, target in zip(states, expected))
        assert all(torch.equal(actual, target) for actual, target in zip(responses, states))
        assert float(
            (states[1] - (0.9 * expected[1] + 0.1 * components[f"h_{modality}"]))
            .abs()
            .max()
        ) > 1e-6


def test_wo_adaptive_composition_is_exact_response_bank_mean_and_preserves_stage_i_ii() -> None:
    x, edge_index = _graph()
    adaptive = _build("full", composition_mode="adaptive")
    uniform = _build("wo_adaptive_composition", composition_mode="adaptive")
    uniform.load_state_dict(adaptive.state_dict())
    adaptive.eval()
    uniform.eval()
    with torch.no_grad():
        full = adaptive._encode_components(x, edge_index)
        components = uniform._encode_components(x, edge_index)

    assert uniform.composition_mode == "uniform"
    for key in ("h_text", "h_visual", "norm_t_index", "norm_t_weight", "norm_v_index", "norm_v_weight"):
        assert torch.equal(components[key], full[key])
    for modality in ("text", "visual"):
        assert all(
            torch.equal(left, right)
            for left, right in zip(components[f"states_{modality}"], full[f"states_{modality}"])
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(components[f"responses_{modality}"], full[f"responses_{modality}"])
        )
        expected = torch.stack(components[f"responses_{modality}"], dim=1).mean(dim=1)
        assert torch.allclose(components[f"z_{modality}"], expected, atol=0.0, rtol=0.0)

    # Adaptive coefficients can change while the uniform final composition is
    # invariant, proving that gamma/Delta-gamma/node/tau are bypassed.
    before = {key: components[key].clone() for key in ("z_text", "z_visual", "z")}
    with torch.no_grad():
        uniform.gamma_global.add_(17.0)
        uniform.delta_gamma_text.sub_(9.0)
        uniform.delta_gamma_visual.add_(11.0)
        uniform.node_vector_text.add_(3.0)
        uniform.node_vector_visual.sub_(4.0)
        if hasattr(uniform, "theta_transport_text"):
            uniform.theta_transport_text.add_(5.0)
            uniform.theta_transport_visual.sub_(6.0)
        after = uniform._encode_components(x, edge_index)
    for key, value in before.items():
        assert torch.equal(after[key], value)


def test_core_variants_forward_backward_are_finite_and_shape_compatible() -> None:
    x, edge_index = _graph()
    outputs = {}
    for variant in CORE_STORY_ABLATIONS:
        model = _build(variant)
        model.train()
        z, _, _, aux_loss, _ = model(x, edge_index)
        (z.square().mean() + aux_loss).backward()
        assert z.shape == (6, 8)
        assert torch.isfinite(z).all()
        assert torch.isfinite(aux_loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        outputs[variant] = z.detach()
    assert set(outputs) == set(CORE_STORY_ABLATIONS)


def test_full_path_regression_matches_pre_core_story_baseline_and_logits() -> None:
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{FULL_REGRESSION_BASELINE_COMMIT}:src/models/mopf.py",
        ],
        text=True,
    ).replace("from .common import make_norm", "from src.models.common import make_norm")
    historical_module = types.ModuleType("historical_mopf_core_story_regression")
    historical_module.__package__ = "src.models"
    exec(compile(source, "HEAD:src/models/mopf.py", "exec"), historical_module.__dict__)

    x, edge_index = _graph()
    cfg = _cfg(
        edge_weight_mode="learned_diag_cos",
        use_transport_residual=True,
        multihop_state_mode="anchored",
        multihop_response_mode="cumulative",
        multihop_anchor_alpha=0.1,
        composition_mode="adaptive",
    )
    cfg["ablation"] = "full"
    torch.manual_seed(777)
    historical = historical_module.Model(
        cfg,
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )
    torch.manual_seed(777)
    current = Model(
        cfg,
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )
    assert current.state_dict().keys() == historical.state_dict().keys()
    current.load_state_dict(historical.state_dict())
    historical.eval()
    current.eval()
    with torch.no_grad():
        old = historical._encode_components(x, edge_index)
        new = current._encode_components(x, edge_index)

    for key in ("h_text", "h_visual", "norm_t_index", "norm_t_weight", "norm_v_index", "norm_v_weight", "eta_text", "eta_visual", "z_text", "z_visual", "z"):
        assert torch.equal(new[key], old[key]), key
    for key in ("edges",):
        for edge_key, old_value in old[key].items():
            assert torch.equal(new[key][edge_key], old_value), f"{key}.{edge_key}"
    for modality in ("text", "visual"):
        for bank_key in ("states", "responses"):
            old_bank = old[f"{bank_key}_{modality}"]
            new_bank = new[f"{bank_key}_{modality}"]
            assert all(torch.equal(left, right) for left, right in zip(new_bank, old_bank))

    torch.manual_seed(778)
    classifier = torch.nn.Linear(old["z"].size(-1), 3)
    assert torch.equal(classifier(new["z"]), classifier(old["z"]))
