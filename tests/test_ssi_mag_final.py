from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.ssi_mag_final import SSIMAGFinal
from src.models.ssi_mag_final_ablation import SSIMAGFinalAblation
from src.models.ssi_mag_v31_r1u import SSIMAGV31R1U


def _cfg(ablation: str = "full"):
    base = OmegaConf.load("configs/model/ssi_mag_final.yaml")
    cfg = OmegaConf.create({"ablation": ablation, "model": OmegaConf.to_container(base, resolve=False)})
    cfg.model.hidden_dim = 8
    cfg.model.relation_rank = 3
    cfg.model.filter_rank = 2
    cfg.model.relation_edge_chunk_size = 2
    cfg.model.num_layers = 3
    cfg.model.max_order = 3
    return cfg


def _inputs():
    torch.manual_seed(811)
    x = torch.randn(6, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0, 4], [1, 0, 2, 1, 3, 2, 0, 3, 4]],
        dtype=torch.long,
    )
    return x, edge_index


def _info():
    return {"input_dim": 10, "text_dim": 4, "visual_dim": 6, "num_nodes": 6, "num_classes": 3}


def _assert_tree_finite(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
    elif isinstance(value, list):
        for item in value:
            _assert_tree_finite(item)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_tree_finite(item)


def test_canonical_state_mapping_and_all_analysis_interventions_are_equivalent():
    x, edge_index = _inputs()
    torch.manual_seed(29)
    pilot = SSIMAGV31R1U(_cfg(), _info())
    canonical = SSIMAGFinal(_cfg(), _info())
    canonical.load_state_dict(pilot.state_dict(), strict=True)
    assert not hasattr(canonical, "theta_beta_text")
    assert not hasattr(canonical, "theta_beta_visual")
    assert not hasattr(canonical, "relation_strength_init")
    assert not any("theta_beta" in name for name, _ in canonical.named_parameters())
    for relation, interaction, reference in (("normal", "normal", "normal"), ("off", "normal", "normal"), ("normal", "off", "normal"), ("normal", "normal", "off")):
        left = pilot.analysis_intervention(x, edge_index, relation=relation, interaction=interaction, reference=reference)
        right = canonical.analysis_intervention(x, edge_index, relation=relation, interaction=interaction, reference=reference)
        for key in ("z_text", "z_visual", "z", "a_text", "a_visual", "relation_weight_text", "relation_weight_visual", "c_text", "c_visual", "alpha_text", "alpha_visual", "Q_text", "Q_visual", "S_text", "S_visual", "D_text", "D_visual", "attention_text", "attention_visual", "S_tilde_text", "S_tilde_visual", "reference_residual_text", "reference_residual_visual", "relation_filter_residual_text", "relation_filter_residual_visual", "eta_text", "eta_visual"):
            a, b = left[key], right[key]
            if isinstance(a, list):
                for aa, bb in zip(a, b, strict=True):
                    torch.testing.assert_close(aa, bb, rtol=0.0, atol=0.0)
            else:
                torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
    assert canonical.analysis(x, edge_index)["beta_present"] is False
    assert not hasattr(canonical, "_active_alphas")


def test_canonical_forward_backward_and_analysis_are_finite():
    x, edge_index = _inputs()
    model = SSIMAGFinal(_cfg(), _info())
    z, _, _, aux, _ = model(x, edge_index)
    (z.square().mean() + aux).backward()
    _assert_tree_finite(model.analysis(x, edge_index))
    assert torch.isfinite(z).all()
    assert not hasattr(model, "_active_alphas")


def test_paper_ablation_exact_stage_formulas():
    x, edge_index = _inputs()
    for name in ("no_relation_modulation", "fixed_semantic_reference", "last_context_only", "no_context_change", "no_cross_hop_interaction", "global_filter_only", "no_formation_conditioning"):
        model = SSIMAGFinalAblation(_cfg(name), _info())
        z, _, _, aux, _ = model(x, edge_index)
        (z.square().mean() + aux).backward()
        analysis = model.analysis(x, edge_index)
        _assert_tree_finite(analysis)
        for modality in ("text", "visual"):
            if name == "no_relation_modulation":
                raw_i, raw_w = model._normalized_operator(edge_index, torch.ones(edge_index.size(1)), 6, x.dtype)
                torch.testing.assert_close(analysis[f"normalized_edge_index_{modality}"], raw_i)
                torch.testing.assert_close(analysis[f"normalized_edge_weight_{modality}"], raw_w)
                torch.testing.assert_close(analysis[f"c_{modality}"], torch.zeros_like(analysis[f"c_{modality}"]))
            if name == "fixed_semantic_reference":
                for alpha in analysis[f"alpha_{modality}"]:
                    torch.testing.assert_close(alpha, torch.full_like(alpha, 0.1), rtol=0.0, atol=0.0)
            if name == "last_context_only":
                eta = analysis[f"eta_{modality}"]
                assert torch.count_nonzero(eta[:, :-1]) == 0
                expected = eta[:, -1:].unsqueeze(-1) * analysis[f"S_tilde_{modality}"][-1].unsqueeze(1)
                torch.testing.assert_close(analysis[f"z_{modality}"], expected.squeeze(1))
            if name == "no_context_change":
                for injection in analysis[f"context_change_injection_{modality}"]:
                    torch.testing.assert_close(injection, torch.zeros_like(injection))
            if name == "no_cross_hop_interaction":
                for state, interacted in zip(analysis[f"S_{modality}"], analysis[f"S_tilde_{modality}"], strict=True):
                    torch.testing.assert_close(state, interacted, rtol=0.0, atol=0.0)
            gamma = analysis[f"gamma_{modality}"].unsqueeze(0)
            delta_gamma = analysis[f"delta_gamma_{modality}"].unsqueeze(0)
            if name == "global_filter_only":
                torch.testing.assert_close(analysis[f"eta_{modality}"], gamma + delta_gamma)
            if name == "no_formation_conditioning":
                torch.testing.assert_close(analysis[f"eta_{modality}"], gamma + delta_gamma + analysis[f"delta_content_{modality}"])
