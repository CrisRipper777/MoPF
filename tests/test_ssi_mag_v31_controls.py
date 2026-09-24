from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.ssi_mag_v3 import SSIMAGV3
from src.models.ssi_mag_v31 import SSIMAGV31
from src.models.ssi_mag_v31_controls import SSIMAGV31Controls


def _cfg(control: str = "r2_only"):
    base = OmegaConf.load("configs/model/ssi_mag_v31_controls.yaml")
    cfg = OmegaConf.create({"ablation": "full", "model": OmegaConf.to_container(base, resolve=False)})
    cfg.model.hidden_dim = 8
    cfg.model.relation_rank = 3
    cfg.model.filter_rank = 2
    cfg.model.relation_edge_chunk_size = 2
    cfg.model.num_layers = 3
    cfg.model.max_order = 3
    cfg.model.control = control
    return cfg


def _inputs():
    torch.manual_seed(101)
    x = torch.randn(6, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0, 4], [1, 0, 2, 1, 3, 2, 0, 3, 4]],
        dtype=torch.long,
    )
    return x, edge_index


def _info():
    return {"input_dim": 10, "text_dim": 4, "visual_dim": 6, "num_nodes": 6, "num_classes": 3}


def test_control_forward_backward_and_reference_formula():
    x, edge_index = _inputs()
    for control in ("r2_only", "r1_r2"):
        torch.manual_seed(17)
        model = SSIMAGV31Controls(_cfg(control), _info())
        z, _, _, aux, _ = model(x, edge_index)
        (z.square().mean() + aux).backward()
        analysis = model.analysis(x, edge_index)
        assert torch.isfinite(z).all()
        for modality in ("text", "visual"):
            alpha = torch.cat(
                [torch.ones_like(analysis[f"p_{modality}"]).unsqueeze(-1)]
                + [value.unsqueeze(-1) for value in analysis[f"alpha_{modality}"]],
                dim=-1,
            )
            centered = alpha - alpha.mean(dim=-1, keepdim=True)
            expected_ref = getattr(model, f"reference_filter_scale_{modality}") * centered
            torch.testing.assert_close(analysis[f"reference_residual_{modality}"], expected_ref)
            expected_eta = (
                analysis[f"gamma_{modality}"].unsqueeze(0)
                + analysis[f"delta_gamma_{modality}"].unsqueeze(0)
                + analysis[f"delta_content_{modality}"]
                + analysis[f"reference_residual_{modality}"]
                + analysis[f"relation_filter_residual_{modality}"]
            )
            torch.testing.assert_close(analysis[f"eta_{modality}"], expected_eta)
            expected_z = model._compose(analysis[f"S_tilde_{modality}"], expected_eta)
            torch.testing.assert_close(analysis[f"z_{modality}"], expected_z)


def test_b_relation_path_matches_historical_v3_exactly():
    x, edge_index = _inputs()
    torch.manual_seed(11)
    old = SSIMAGV3(_cfg("r2_only"), _info())
    torch.manual_seed(12)
    control = SSIMAGV31Controls(_cfg("r2_only"), _info())
    for modality in ("text", "visual"):
        for suffix in ("relation_norm", "relation_proj", "relation_scorer"):
            left = getattr(old, f"{suffix}_{modality}")
            right = getattr(control, f"{suffix}_{modality}")
            right.load_state_dict(left.state_dict())
        getattr(control, f"theta_beta_{modality}").data.copy_(getattr(old, f"theta_beta_{modality}").data)
        h0 = getattr(old, f"{modality}_proj")(x[:, :4] if modality == "text" else x[:, 4:])
        old_out = old._relation_edge_outputs(h0, edge_index, modality)
        control_out = control._relation_edge_outputs(h0, edge_index, modality)
        torch.testing.assert_close(control_out["relation_residual"], old_out["relation_residual"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(control_out["relation_weight"], old_out["relation_weight"], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            control._local_adaptation(edge_index, control_out["relation_residual"], control_out["beta"], control_out["nonself_mask"], 6),
            old._local_adaptation(edge_index, old_out["relation_residual"], old_out["beta"], 6),
            rtol=0.0,
            atol=0.0,
        )


def test_b_and_ab_use_the_same_v31_r2_formula():
    x, edge_index = _inputs()
    torch.manual_seed(19)
    b = SSIMAGV31Controls(_cfg("r2_only"), _info())
    torch.manual_seed(20)
    ab = SSIMAGV31Controls(_cfg("r1_r2"), _info())
    for modality in ("text", "visual"):
        for name in (
            f"semantic_prior_norm_{modality}",
            f"semantic_prior_vector_{modality}",
            f"semantic_bias_{modality}",
            f"semantic_rho_p_{modality}",
            f"semantic_rho_d_{modality}",
        ):
            left, right = getattr(b, name), getattr(ab, name)
            if isinstance(left, torch.nn.Module):
                left.load_state_dict(right.state_dict())
            else:
                left.data.copy_(right.data)
    h0 = b.text_proj(x[:, :4])
    raw_index, raw_weight = b._normalized_operator(edge_index, torch.ones(edge_index.size(1)), 6, x.dtype)
    left = b._semantic_states(h0, raw_index, raw_weight, "text")
    right = ab._semantic_states(h0, raw_index, raw_weight, "text")
    torch.testing.assert_close(left["p"], right["p"], rtol=0.0, atol=0.0)
    for key in ("propagated", "changes", "alphas", "states"):
        for left_value, right_value in zip(left[key], right[key], strict=True):
            torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)


def test_ab_reference_off_and_zero_scale_match_frozen_v31():
    x, edge_index = _inputs()
    torch.manual_seed(31)
    frozen = SSIMAGV31(_cfg("r2_only"), _info())
    torch.manual_seed(31)
    ab = SSIMAGV31Controls(_cfg("r1_r2"), _info())
    common = frozen.state_dict()
    ab_state = ab.state_dict()
    for name, value in common.items():
        if name in ab_state:
            torch.testing.assert_close(ab_state[name], value, rtol=0.0, atol=0.0)
    with torch.no_grad():
        ab.reference_filter_scale_text.zero_()
        ab.reference_filter_scale_visual.zero_()
    left = frozen.analysis(x, edge_index)
    right = ab.analysis(x, edge_index)
    torch.testing.assert_close(left["z"], right["z"], rtol=0.0, atol=0.0)
    off = ab.analysis_intervention(x, edge_index, reference="off")
    torch.testing.assert_close(off["reference_residual_text"], torch.zeros_like(off["reference_residual_text"]))
    torch.testing.assert_close(off["reference_residual_visual"], torch.zeros_like(off["reference_residual_visual"]))
