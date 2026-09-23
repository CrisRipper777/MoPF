from __future__ import annotations

import math

import torch

from scripts.analyze_ssi_mag_v3_p15a import _alpha_range_ratio, attention_diagnostics


def test_uniform_attention_is_uniform_at_node_and_query_levels() -> None:
    attention = torch.full((5, 4, 4), 0.25)
    _, _, metrics = attention_diagnostics(attention)
    assert math.isclose(metrics["nodewise_normalized_entropy"], 1.0, abs_tol=1e-9)
    assert math.isclose(metrics["attention_nonuniformity"], 0.0, abs_tol=1e-12)
    assert math.isclose(metrics["query_diversity_mean"], 0.0, abs_tol=1e-12)
    assert math.isclose(metrics["node_heterogeneity_mean"], 0.0, abs_tol=1e-12)
    assert math.isclose(metrics["diagonal_excess"], 0.0, abs_tol=1e-12)


def test_identity_attention_is_nonuniform_and_diagonal() -> None:
    attention = torch.eye(4).unsqueeze(0).repeat(5, 1, 1)
    _, _, metrics = attention_diagnostics(attention)
    assert math.isclose(metrics["nodewise_normalized_entropy"], 0.0, abs_tol=1e-9)
    assert math.isclose(metrics["attention_nonuniformity"], 1.0, abs_tol=1e-12)
    assert math.isclose(metrics["diagonal_mass_mean"], 1.0, abs_tol=1e-12)
    assert math.isclose(metrics["diagonal_excess"], 0.75, abs_tol=1e-12)
    assert metrics["query_diversity_mean"] > 0.0


def test_identical_nonuniform_query_rows_have_zero_query_diversity() -> None:
    row = torch.tensor([0.7, 0.1, 0.1, 0.1])
    attention = row.view(1, 1, 4).repeat(5, 4, 1)
    _, _, metrics = attention_diagnostics(attention)
    assert metrics["nodewise_normalized_entropy"] < 1.0
    assert math.isclose(metrics["query_diversity_mean"], 0.0, abs_tol=1e-12)


def test_mean_matrix_can_be_uniform_while_nodes_are_heterogeneous() -> None:
    attention = torch.zeros(4, 4, 4)
    for node in range(4):
        attention[node, :, node] = 1.0
    _, _, metrics = attention_diagnostics(attention)
    assert math.isclose(metrics["mean_matrix_normalized_entropy"], 1.0, abs_tol=1e-12)
    assert metrics["nodewise_normalized_entropy"] < 1.0e-9
    assert metrics["node_heterogeneity_mean"] > 0.0
    assert metrics["jensen_entropy_gap"] > 0.0


def test_p_sign_symmetry_keeps_functional_term_p_unchanged() -> None:
    p_one = torch.tensor([0.2, -0.4, 0.7])
    rho_one = torch.tensor(0.3)
    p_two = -p_one
    rho_two = -rho_one
    torch.testing.assert_close(rho_one * p_one, rho_two * p_two)


def test_alpha_range_ratio_uses_alpha_mean_denominator() -> None:
    alpha = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    expected = (0.46 - 0.14) / 0.30
    assert math.isclose(_alpha_range_ratio(alpha), expected, rel_tol=0.0, abs_tol=1e-7)
