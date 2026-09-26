from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.models.cmrf_probe import Model
from src.tasks.nc import _resolve_nc_eval_labels


def make_model(mode="uniform", dtype=torch.float32):
    cfg = OmegaConf.create(
        {
            "model": {
                "hidden_dim": 8,
                "max_order": 3,
                "dropout": 0.0,
                "composition_mode": mode,
                "controller_latent_dim": 3,
                "controller_hidden_dim": 7,
                "controller_lambda": 0.25,
            }
        }
    )
    return Model(cfg, {"input_dim": 11, "text_dim": 5, "visual_dim": 6}).to(dtype=dtype)


def graph_inputs(dtype=torch.float32):
    torch.manual_seed(21)
    x = torch.randn(7, 11, dtype=dtype)
    # Includes duplicated/reversed links and a self-loop to audit canonicalization.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 0],
         [1, 0, 2, 1, 3, 2, 5, 5, 4, 6, 0]],
        dtype=torch.long,
    )
    return x, edge_index


def test_propagation_operator_matches_baseline_multi_order_bank():
    baseline_path = (
        Path(__file__).resolve().parents[2]
        / "baseline"
        / "src"
        / "models"
        / "multi_order_bank.py"
    )
    spec = importlib.util.spec_from_file_location("cmrf_baseline_multi_order_bank", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    ref = object.__new__(baseline.Model)
    torch.nn.Module.__init__(ref)
    model = make_model()
    _, edge_index = graph_inputs()
    actual = model._build_propagation_operator(edge_index, 7, torch.float32).to_dense()
    expected = ref._build_propagation_operator(edge_index, 7, torch.float32).to_dense()
    assert torch.equal(actual, expected)


def test_uniform_state_and_response_space_equivalence_float_and_double():
    for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-10)):
        model = make_model("uniform", dtype).eval()
        x, edge_index = graph_inputs(dtype)
        analysis = model.analyze(x, edge_index)
        states = analysis["S_text"]
        responses = analysis["R_text"]
        uniform = sum(states) / 4.0
        response_form = states[0] + 0.75 * responses[0] + 0.50 * responses[1] + 0.25 * responses[2]
        assert (uniform - response_form).abs().max().item() < tolerance


def test_adaptive_zero_initialization_matches_uniform_and_keeps_h0_fixed():
    x, edge_index = graph_inputs()
    uniform = make_model("uniform").eval()
    for mode in ("own_soft", "cross_soft", "gated_cross_soft"):
        adaptive = make_model(mode).eval()
        adaptive.load_state_dict(
            {k: v for k, v in uniform.state_dict().items() if k in adaptive.state_dict()},
            strict=False,
        )
        adaptive.load_state_dict(
            {k: v for k, v in adaptive.state_dict().items() if k in uniform.state_dict()},
            strict=False,
        )
        ref = uniform.analyze(x, edge_index)
        got = adaptive.analyze(x, edge_index)
        assert (ref["fused_z"] - got["fused_z"]).abs().max().item() < 1e-6
        assert torch.allclose(got["a_text"], torch.tensor([0.75, 0.50, 0.25]).expand(7, -1))
        assert torch.allclose(got["a_visual"], torch.tensor([0.75, 0.50, 0.25]).expand(7, -1))
        # H0 is inserted with unit coefficient by construction; companion q cannot alter it.
        shuffled = got["q_visual"][torch.randperm(7)]
        changed = adaptive.recompose(got, q_visual=shuffled)
        assert torch.equal(changed["Z_text"], got["Z_text"])
        assert torch.equal(changed["Z_visual"], got["Z_visual"])


def test_cross_shuffle_only_replaces_companion_control_input():
    x, edge_index = graph_inputs()
    model = make_model("cross_soft").eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.delta_text_cross[-1].weight, std=0.05)
        torch.nn.init.normal_(model.delta_text_cross[-1].bias, std=0.05)
    analysis = model.analyze(x, edge_index)
    q_text_before = analysis["q_text"].clone()
    q_visual_shuffled = analysis["q_visual"][torch.tensor([3, 0, 6, 2, 1, 5, 4])]
    intervention = model.intervene_companion(
        analysis, target_modality="Text", companion_q=q_visual_shuffled
    )
    assert torch.equal(analysis["q_text"], q_text_before)
    assert torch.equal(analysis["S_text"][0], model.analyze(x, edge_index)["S_text"][0])
    assert torch.equal(analysis["S_visual"][3], model.analyze(x, edge_index)["S_visual"][3])
    assert torch.equal(analysis["H0_text"], model.analyze(x, edge_index)["H0_text"])
    assert not torch.allclose(analysis["a_text"], intervention["a_text"])
    assert torch.equal(analysis["a_visual"], intervention["a_visual"])


def test_test_labels_are_not_used_when_test_evaluation_is_disabled():
    data = SimpleNamespace(
        y=torch.tensor([0, 1, 2, 0, 1, 2]),
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2, 3]),
        test_idx=torch.tensor([4, 5]),
        num_classes=3,
    )
    first = _resolve_nc_eval_labels(data, include_test=False)
    data.y[4:] = torch.tensor([99, -100])
    second = _resolve_nc_eval_labels(data, include_test=False)
    assert first == second == [0, 1, 2]


def test_operator_has_no_semantic_feature_weighting():
    model = make_model()
    x, edge_index = graph_inputs()
    p0 = model._build_propagation_operator(edge_index, x.size(0), x.dtype).to_dense()
    changed_features = x.roll(1, dims=1) * 17.0
    p1 = model._build_propagation_operator(edge_index, changed_features.size(0), changed_features.dtype).to_dense()
    assert torch.equal(p0, p1)


def test_forward_and_backward_are_finite():
    for mode in ("uniform", "own_soft", "cross_soft", "gated_cross_soft"):
        model = make_model(mode)
        x, edge_index = graph_inputs()
        out = model(x, edge_index)[0]
        assert torch.isfinite(out).all()
        out.square().mean().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()


def test_checkpoint_reload_equivalence():
    model = make_model("gated_cross_soft").eval()
    x, edge_index = graph_inputs()
    expected = model(x, edge_index)[0]
    clone = make_model("gated_cross_soft").eval()
    clone.load_state_dict(model.state_dict())
    actual = clone(x, edge_index)[0]
    assert torch.equal(expected, actual)
