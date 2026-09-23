from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.cosi_mag_final import CoSIMAGFinal
from src.models.mgsc_mag import MGSCMAG


def _cfg(*, adaptive: bool, direct: bool = False, legacy: bool = True):
    return OmegaConf.create(
        {
            "model": {
                "name": "mgsc_mag",
                "hidden_dim": 8,
                "dropout": 0.0,
                "norm": "layernorm",
                "max_order": 3,
                "diffusion_add_self_loops": True,
                "edge_metric": "learned_diag_cos",
                "metric_init_seed": 20260910,
                "edge_weight_min": 0.1,
                "edge_weight_temperature": 0.35,
                "eps": 1e-8,
                "filter_rank": 4,
                "multihop_state": "anchored",
                "multihop_response": "cumulative",
                "multihop_anchor_alpha": 0.1,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "hop_interaction_layers": 1,
                "hop_interaction_heads": 1,
                "hop_interaction_dropout": 0.0,
                "hop_interaction_order_embedding": True,
                "hop_interaction_gate_init": 0.10,
                "relation_descriptor_eps": 1e-6,
                "relation_bias_init": 0.10,
                "relation_init_seed": 20260921,
                "relation_init_std": 0.02,
                "fusion": "concat_residual_mlp",
                "adaptive_context_gate": adaptive,
                "context_gate_order_dim": 4,
                "context_gate_hidden_dim": 10,
                "direct_interacted_integration": direct,
                "use_legacy_relation_order_bias": legacy,
            }
        }
    )


DATA_INFO = {"input_dim": 10, "num_nodes": 7, "num_classes": 3, "text_dim": 4, "visual_dim": 6}


def _graph():
    torch.manual_seed(123)
    x = torch.randn(DATA_INFO["num_nodes"], DATA_INFO["input_dim"])
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]], dtype=torch.long)
    return x, edge_index


def test_p0_equivalence_with_reference_shared_state_mapping():
    torch.manual_seed(7)
    reference = CoSIMAGFinal(_cfg(adaptive=False), DATA_INFO)
    torch.manual_seed(7)
    candidate = MGSCMAG(_cfg(adaptive=False), DATA_INFO)
    missing, unexpected = candidate.load_state_dict(reference.state_dict(), strict=False)
    assert unexpected == []
    assert all(name.startswith("context_gate_") for name in missing)
    x, edge_index = _graph()
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        z_reference = reference(x, edge_index)[0]
        z_candidate = candidate(x, edge_index)[0]
    assert torch.max(torch.abs(z_reference - z_candidate)).item() <= 1e-6


def test_adaptive_nc_forward_backward_gate_analysis_and_finiteness():
    torch.manual_seed(11)
    model = MGSCMAG(_cfg(adaptive=True), DATA_INFO)
    x, edge_index = _graph()
    classifier = torch.nn.Linear(model.out_dim, DATA_INFO["num_classes"])
    model.train()
    logits = classifier(model(x, edge_index)[0])
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    gate_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("context_gate_") and parameter.grad is not None
    ]
    assert gate_gradients
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gate_gradients)

    model.eval()
    analysis = model.analysis_multi_order(x, edge_index)
    for modality in ("text", "visual"):
        gates = analysis[f"context_gate_{modality}"]
        assert len(gates) == 3
        gate_bank = torch.stack(gates, dim=1)
        assert gate_bank.shape == (DATA_INFO["num_nodes"], 3)
        assert torch.isfinite(gate_bank).all()
        assert torch.allclose(gate_bank, torch.full_like(gate_bank, 0.9), atol=1e-6)

    inference = model.inference(x, edge_index, device=torch.device("cpu"))
    assert torch.isfinite(inference).all()


def test_direct_interacted_integration_is_finite():
    model = MGSCMAG(_cfg(adaptive=True, direct=True), DATA_INFO)
    x, edge_index = _graph()
    output = model(x, edge_index)[0]
    assert output.shape == (DATA_INFO["num_nodes"], model.out_dim)
    assert torch.isfinite(output).all()

    analysis = model.analysis_intervention(
        x,
        edge_index,
        relation="normal",
        interaction="off",
    )
    assert torch.isfinite(analysis["z"]).all()
    assert torch.isfinite(analysis["z_text"]).all()
    assert torch.isfinite(analysis["z_visual"]).all()


def test_gate_audit_interventions_preserve_marginal_or_constant_structure():
    torch.manual_seed(19)
    model = MGSCMAG(_cfg(adaptive=True, direct=True), DATA_INFO)
    x, edge_index = _graph()
    normal = model.analysis_intervention(x, edge_index, gate_intervention="normal")
    globalized = model.analysis_intervention(x, edge_index, gate_intervention="globalized")
    shuffled = model.analysis_intervention(
        x, edge_index, gate_intervention="shuffled", gate_permutation_seed=77
    )
    fixed = model.analysis_intervention(x, edge_index, gate_intervention="fixed_0.9")
    for analysis in (normal, globalized, shuffled, fixed):
        assert torch.isfinite(analysis["z"]).all()
    for modality in ("text", "visual"):
        normal_bank = torch.stack(normal[f"context_gate_{modality}"], dim=1)
        global_bank = torch.stack(globalized[f"context_gate_{modality}"], dim=1)
        shuffled_bank = torch.stack(shuffled[f"context_gate_{modality}"], dim=1)
        fixed_bank = torch.stack(fixed[f"context_gate_{modality}"], dim=1)
        assert torch.allclose(global_bank.std(dim=0), torch.zeros(3), atol=1e-7)
        assert torch.allclose(
            torch.sort(normal_bank, dim=0).values,
            torch.sort(shuffled_bank, dim=0).values,
            atol=1e-7,
        )
        assert torch.allclose(fixed_bank, torch.full_like(fixed_bank, 0.9))


def test_legacy_relation_order_bias_cleanup_switch_is_finite():
    model = MGSCMAG(_cfg(adaptive=True, direct=True, legacy=False), DATA_INFO)
    x, edge_index = _graph()
    output = model.analysis_intervention(x, edge_index)
    assert torch.isfinite(output["z"]).all()
