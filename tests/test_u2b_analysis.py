from __future__ import annotations

from types import SimpleNamespace

import torch

from src.analysis.u2b import build_response_variant, response_magnitude_audit, semantic_retention


def _toy_operator():
    matrix = torch.tensor(
        [[0.8, 0.2, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]],
        dtype=torch.float64,
    )
    h0 = torch.arange(12, dtype=torch.float64).view(3, 4) / 10.0

    def propagate(value: torch.Tensor) -> torch.Tensor:
        return matrix @ value

    return h0, propagate


def test_b0_is_the_formal_cumulative_recurrence():
    h0, propagate = _toy_operator()
    result = build_response_variant(h0, propagate, max_hop=3, variant="B0")
    expected = [h0]
    for _ in range(3):
        expected.append(propagate(expected[-1]))
    assert all(torch.equal(left, right) for left, right in zip(result["states"], expected))
    assert all(torch.equal(left, right) for left, right in zip(result["responses"], expected))
    assert result["max_reconstruction_error"] is None


def test_b1_reconstruction_is_exact():
    h0, propagate = _toy_operator()
    result = build_response_variant(h0, propagate, max_hop=6, variant="B1")
    assert result["max_reconstruction_error"] < 1e-12
    for state, response in zip(result["states"], result["responses"]):
        assert torch.isfinite(state).all()
        assert torch.isfinite(response).all()


def test_b3_reconstruction_is_exact_and_alpha_is_fixed():
    h0, propagate = _toy_operator()
    before = h0.clone()
    result = build_response_variant(h0, propagate, max_hop=6, variant="B3", alpha=0.1)
    assert result["alpha"] == 0.1
    assert result["max_reconstruction_error"] < 1e-12
    assert torch.equal(h0, before)
    assert all(not tensor.requires_grad for tensor in result["states"])


def test_semantic_retention_and_magnitude_are_finite():
    h0, propagate = _toy_operator()
    result = build_response_variant(h0, propagate, max_hop=3, variant="B2", alpha=0.1)
    subset = torch.arange(h0.size(0))
    retention = semantic_retention(result["states"], h0, subset)
    magnitude = response_magnitude_audit(result["responses"], h0, innovation_channel=False)
    assert all(torch.isfinite(torch.tensor(value)) for value in retention["linear_cka_to_h0"])
    assert all(torch.isfinite(torch.tensor(value)) for value in retention["frobenius_cosine_to_h0"])
    assert all(torch.isfinite(torch.tensor(value)) for value in magnitude["norm_ratio_to_h0"])
    assert not magnitude["vanishing_innovation_channel"]


def test_alpha_sensitivity_does_not_add_parameters_or_change_model_config():
    h0, propagate = _toy_operator()
    results = [
        build_response_variant(h0, propagate, max_hop=3, variant="B3", alpha=alpha)
        for alpha in (0.05, 0.1, 0.2)
    ]
    assert [result["alpha"] for result in results] == [0.05, 0.1, 0.2]
    assert all(result["max_reconstruction_error"] < 1e-12 for result in results)


def test_k_stress_is_analysis_only_and_does_not_touch_formal_config():
    model_config = open("configs/model/mopf.yaml", encoding="utf-8").read()
    assert "max_order: 3" in model_config
    assert "edge_weight_mode: learned_diag_cos" in model_config
    assert "edge_weight_temperature: 0.35" in model_config
    h0, propagate = _toy_operator()
    stress = [build_response_variant(h0, propagate, max_hop=k, variant="B0") for k in range(1, 7)]
    assert len(stress[-1]["states"]) == 7
    assert "max_order: 6" not in model_config
