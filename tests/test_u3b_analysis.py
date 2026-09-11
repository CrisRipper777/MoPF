from __future__ import annotations

import torch

from src.analysis.u3b import (
    association_gate,
    centered_transport_context,
    mean_incident_conductance,
    order_centered_profile,
    partial_spearman,
    shuffle_node_scalar,
    transport_residual,
)
from src.models.mopf import MoPF


def test_mean_incident_conductance_matches_raw_endpoint_definition_and_isolates():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    conductance = torch.tensor([0.2, 0.8])
    mean, degree = mean_incident_conductance(edge_index, conductance, 3)
    assert torch.allclose(mean, torch.tensor([0.5, 0.5, 0.5]))
    assert torch.equal(degree, torch.tensor([2.0, 2.0, 0.0]))


def test_centered_context_and_tcp_residual_are_identifiable():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    conductance = torch.tensor([0.2, 0.2, 0.8, 0.8])
    context = centered_transport_context(edge_index, conductance, 4)
    theta = torch.tensor([1.0, 2.0, 4.0])
    beta, tau = transport_residual(context["centered"], theta)
    assert abs(float(context["centered"].mean())) < 1e-7
    assert context["centered"][3].abs() < 1e-7
    assert torch.allclose(beta, torch.tensor([-4.0 / 3.0, -1.0 / 3.0, 5.0 / 3.0]))
    assert torch.allclose(tau.sum(dim=1), torch.zeros(4), atol=1e-7)
    assert abs(float(tau.mean())) < 1e-7


def test_zero_theta_has_zero_centered_profile_and_shuffle_preserves_marginal():
    context = torch.tensor([-0.5, 0.0, 0.5, 1.0])
    assert torch.equal(order_centered_profile(torch.zeros(4)), torch.zeros(4))
    shuffled = shuffle_node_scalar(context, 20260931)
    assert torch.equal(torch.sort(shuffled).values, torch.sort(context).values)
    assert torch.equal(shuffled, shuffle_node_scalar(context, 20260931))


def test_partial_spearman_residualizes_ranked_degree_and_gate_ignores_p_values():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(5)
    degree = torch.rand(30, generator=generator)
    shared_residual = torch.rand(30, generator=generator)
    predictor = degree + 0.2 * shared_residual
    outcome = degree + 0.4 * shared_residual
    result = partial_spearman(outcome, predictor, degree)
    assert result["partial_rho"] > 0.0
    assert result["raw_rho"] > 0.0
    rows = [
        {"dataset": dataset, "modality": modality, "partial_rho": 0.25, "partial_sign": 1, "degenerate": False}
        for dataset in ("A", "B")
        for modality in ("text", "visual")
        for _ in range(3)
    ]
    gate = association_gate(rows)
    assert gate["passed"]
    assert gate["uses_p_values_for_gate"] is False


class _Cfg:
    def __init__(self, use_transport_residual: bool):
        self.model = {
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
            "edge_weight_temperature": 0.35,
            "filter_rank": 4,
            "global_filter_trainable": True,
            "use_modality_residual": True,
            "use_node_residual": True,
            "hrc_weight": 0.0,
            "fusion_mode": "concat_residual_mlp",
            "node_conditioner_mode": "absolute",
            "use_transport_residual": use_transport_residual,
        }


def _model_graph():
    torch.manual_seed(123)
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0]],
        dtype=torch.long,
    )
    return x, edge_index


def test_tcpr_adds_only_two_order_profiles_and_preserves_zero_init_exactly():
    x, edge_index = _model_graph()
    torch.manual_seed(9)
    baseline = MoPF(_Cfg(False), {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6})
    torch.manual_seed(9)
    candidate = MoPF(_Cfg(True), {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6})
    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        base = baseline._encode_components(x, edge_index)
        cand = candidate._encode_components(x, edge_index)
    for name in ("eta_text", "eta_visual", "z_text", "z_visual", "z"):
        assert torch.equal(base[name], cand[name])
    assert set(name for name, _ in candidate.named_parameters() if "theta_transport" in name) == {
        "theta_transport_text",
        "theta_transport_visual",
    }
    assert sum(parameter.numel() for parameter in candidate.parameters()) - sum(parameter.numel() for parameter in baseline.parameters()) == 8


def test_tcpr_context_is_detached_and_identifiability_constraints_hold():
    x, edge_index = _model_graph()
    model = MoPF(_Cfg(True), {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6})
    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
    for modality in ("text", "visual"):
        context = components[f"transport_context_{modality}"]
        beta = components[f"beta_transport_{modality}"]
        tau = components[f"tau_transport_{modality}"]
        assert context.requires_grad is False
        assert abs(float(context.mean())) < 1e-7
        assert abs(float(beta.sum())) < 1e-7
        assert torch.allclose(tau.sum(dim=1), torch.zeros(6), atol=1e-7)
        assert abs(float(tau.mean())) < 1e-7


def test_tcpr_transport_off_removes_only_tau_from_effective_eta():
    x, edge_index = _model_graph()
    model = MoPF(_Cfg(True), {"input_dim": 10, "num_nodes": 6, "num_classes": 3, "text_dim": 4, "visual_dim": 6})
    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
        off = model._effective_coefficients(
            "text",
            components["delta_node_text"],
            torch.zeros(6),
        )
    assert torch.allclose(off, components["eta_text"] - components["tau_transport_text"])
