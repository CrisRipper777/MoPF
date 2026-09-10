from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.mopf import MoPF


def _cfg(temperature: float = 2.0):
    return SimpleNamespace(
        model={
            "hidden_dim": 8,
            "dropout": 0.0,
            "norm": "layernorm",
            "max_order": 3,
            "num_layers": 3,
            "map_prior_restart": 0.15,
            "map_prior_order": 2,
            "diffusion_add_self_loops": True,
            "edge_weight_mode": "learned_diag_cos",
            "num_metric_perspectives": 4,
            "metric_init_seed": 20260910,
            "metric_init_noise_std": 0.01,
            "edge_weight_min": 0.1,
            "edge_weight_temperature": temperature,
            "filter_rank": 4,
            "global_filter_trainable": True,
            "use_modality_residual": True,
            "use_node_residual": True,
            "hrc_weight": 0.0,
            "ppc_weight": 0.0,
            "fusion_mode": "concat_residual_mlp",
            "node_conditioner_mode": "absolute",
        }
    )


def _graph():
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def _model():
    torch.manual_seed(123)
    model = MoPF(
        _cfg(),
        {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6},
    )
    model.eval()
    return model


def _state_copy(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def test_tau2_override_exactly_matches_formal_r1_forward():
    x, edge_index = _graph()
    model = _model()
    with torch.no_grad():
        formal = model._encode_components(x, edge_index)
        override = model.analysis_encode_with_metric_override(
            x, edge_index, temperature_override=2.0
        )
    assert torch.equal(formal["z"], override["z"])
    assert torch.equal(formal["edges"]["w_t"], override["edges"]["w_t"])
    assert torch.equal(formal["edges"]["w_v"], override["edges"]["w_v"])


def test_temperature_override_does_not_mutate_model_state_or_checkpoint_state():
    x, edge_index = _graph()
    model = _model()
    before = _state_copy(model)
    with torch.no_grad():
        model.analysis_encode_with_metric_override(x, edge_index, temperature_override=0.25)
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)
    assert model.edge_weight_temperature == 2.0


def test_raw_semantic_scores_are_invariant_across_temperature_grid():
    x, edge_index = _graph()
    model = _model()
    with torch.no_grad():
        baseline = model.analysis_encode_with_metric_override(x, edge_index, temperature_override=2.0)
        for temperature in (1.5, 1.0, 0.75, 0.5, 0.35, 0.25):
            current = model.analysis_encode_with_metric_override(
                x, edge_index, temperature_override=temperature
            )
            assert torch.equal(baseline["edges"]["cos_t"], current["edges"]["cos_t"])
            assert torch.equal(baseline["edges"]["cos_v"], current["edges"]["cos_v"])


def test_identity_metric_and_learned_metric_temperature_overrides_work():
    x, edge_index = _graph()
    model = _model()
    ones_text = torch.ones_like(model.semantic_metric_weights("text"))
    ones_visual = torch.ones_like(model.semantic_metric_weights("visual"))
    with torch.no_grad():
        learned = model.analysis_encode_with_metric_override(
            x, edge_index, temperature_override=0.5
        )
        identity = model.analysis_encode_with_metric_override(
            x,
            edge_index,
            text_weights=ones_text,
            visual_weights=ones_visual,
            temperature_override=0.5,
        )
    assert learned["edges"]["metric_weights_text"].shape == (1, 8)
    assert torch.equal(identity["edges"]["metric_weights_text"], ones_text)
    assert torch.equal(identity["edges"]["metric_weights_visual"], ones_visual)
    assert torch.isfinite(learned["z"]).all()
    assert torch.isfinite(identity["z"]).all()


def test_edge_support_and_gcn_norm_family_are_unchanged_for_every_tau():
    x, edge_index = _graph()
    model = _model()
    with torch.no_grad():
        baseline = model.analysis_encode_with_metric_override(x, edge_index, temperature_override=2.0)
        for temperature in (1.5, 1.0, 0.75, 0.5, 0.35, 0.25):
            current = model.analysis_encode_with_metric_override(
                x, edge_index, temperature_override=temperature
            )
            assert torch.equal(current["norm_t_index"], baseline["norm_t_index"])
            assert torch.equal(current["norm_v_index"], baseline["norm_v_index"])
            assert current["edges"]["w_t"].numel() == edge_index.size(1)
            assert current["edges"]["w_v"].numel() == edge_index.size(1)


def test_conductance_is_finite_and_within_configured_bounds_for_temperature_grid():
    x, edge_index = _graph()
    model = _model()
    with torch.no_grad():
        for temperature in (2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25):
            components = model.analysis_encode_with_metric_override(
                x, edge_index, temperature_override=temperature
            )
            for key in ("w_t", "w_v"):
                values = components["edges"][key]
                assert torch.isfinite(values).all()
                assert float(values.min()) >= model.edge_weight_min - 1e-7
                assert float(values.max()) <= 1.0 + 1e-7


def test_lower_temperature_changes_mapping_on_controlled_scores():
    model = _model()
    scores = torch.tensor([-1.0, -0.25, 0.0, 0.25, 1.0])
    with torch.no_grad():
        high = model._edge_weight_from_cosine(scores, temperature_override=2.0)
        low = model._edge_weight_from_cosine(scores, temperature_override=0.25)
    assert not torch.equal(high, low)
    assert float((low[1:] - low[:-1]).min()) > 0.0


def test_temperature_is_not_trainable_and_is_shared_by_text_and_visual():
    model = _model()
    assert not any("temperature" in name for name, _ in model.named_parameters())
    scores = torch.tensor([-0.75, 0.0, 0.75])
    with torch.no_grad():
        text = model._edge_weight_from_cosine(scores, temperature_override=0.35)
        visual = model._edge_weight_from_cosine(scores, temperature_override=0.35)
    assert torch.equal(text, visual)


def test_propagation_bank_and_filter_fusion_parameters_are_analysis_only():
    x, edge_index = _graph()
    model = _model()
    before = _state_copy(model)
    with torch.no_grad():
        high = model.analysis_encode_with_metric_override(x, edge_index, temperature_override=2.0)
        low = model.analysis_encode_with_metric_override(x, edge_index, temperature_override=0.25)
    assert len(high["bases_text"]) == len(low["bases_text"]) == 4
    assert len(high["bases_visual"]) == len(low["bases_visual"]) == 4
    assert torch.equal(high["h_text"], low["h_text"])
    assert torch.equal(high["h_visual"], low["h_visual"])
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)


def test_invalid_temperature_override_is_rejected_without_state_change():
    x, edge_index = _graph()
    model = _model()
    before = _state_copy(model)
    for temperature in (0.0, -1.0, float("inf"), float("nan")):
        try:
            model.analysis_encode_with_metric_override(
                x, edge_index, temperature_override=temperature
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"temperature {temperature!r} was accepted")
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value)
