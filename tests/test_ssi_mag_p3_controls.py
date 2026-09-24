from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.ssi_mag_generic_gpr import SSIMAGGenericGPR
from src.models.ssi_mag_generic_ppr import SSIMAGGenericPPR
from src.models.ssi_mag_scd import SSIMAGSCD
from src.models.ssi_mag_scd_ablation import SSIMAGSCDABlation
from src.tasks.lp import _resolve_lp_num_neighbors


def _cfg(variant: str = "full"):
    return OmegaConf.create(
        {
            "ablation": variant,
            "model": {
                "name": "ssi_mag_scd_ablation",
                "ablation": "full",
                "hidden_dim": 8,
                "dropout": 0.0,
                "norm": "layernorm",
                "max_order": 3,
                "num_layers": 3,
                "diffusion_add_self_loops": True,
                "relation_rank": 3,
                "relation_edge_chunk_size": 32,
                "relation_init_seed": 20260923,
                "relation_init_std": 0.02,
                "semantic_reference_init": 0.10,
                "context_change_gate_init": 0.10,
                "hop_interaction_gate_init": 0.10,
                "hop_interaction_layers": 1,
                "hop_interaction_heads": 1,
                "hop_interaction_dropout": 0.0,
                "hop_interaction_order_embedding": True,
                "filter_rank": 2,
                "global_prior_restart": 0.15,
                "global_prior_order": 2,
                "reference_filter_scale_init": 0.10,
                "relation_filter_scale_init": 0.10,
                "fusion": "concat_residual_mlp",
                "eps": 1.0e-8,
            },
        }
    )


def _generic_cfg(name: str):
    cfg = _cfg("full")
    cfg.model.name = name
    cfg.model.control_mode = "ppr_style" if "ppr" in name else "gpr_style"
    return cfg


def _data_info():
    return {"input_dim": 10, "text_dim": 4, "visual_dim": 6}


def _graph():
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
        ],
        dtype=torch.long,
    )
    x = torch.randn(5, 10)
    return x, edge_index


def _assert_finite_backward(model, x, edge_index):
    model.train()
    output = model(x, edge_index)[0]
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_generic_yaml_keeps_fixed_k_as_an_integer():
    root = Path(__file__).resolve().parents[1]
    for name in ("ssi_mag_generic_ppr.yaml", "ssi_mag_generic_gpr.yaml"):
        cfg = OmegaConf.load(root / "configs" / "model" / name)
        assert cfg.max_order == 3
        assert cfg.num_layers == 3


def test_full_s_ablation_is_state_and_output_equivalent_to_frozen_s():
    x, edge_index = _graph()
    torch.manual_seed(123)
    frozen = SSIMAGSCD(_cfg("full"), _data_info()).eval()
    torch.manual_seed(123)
    full = SSIMAGSCDABlation(_cfg("full"), _data_info()).eval()
    full.load_state_dict(frozen.state_dict(), strict=True)
    frozen_analysis = frozen.analysis(x, edge_index)
    full_analysis = full.analysis(x, edge_index)
    assert set(frozen.state_dict()) == set(full.state_dict())
    assert torch.allclose(frozen_analysis["z"], full_analysis["z"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        frozen_analysis["eta_text"], full_analysis["eta_text"], atol=1e-6, rtol=1e-6
    )


def test_s_ablation_formulas_and_all_paths_are_finite():
    x, edge_index = _graph()
    for variant in SSIMAGSCDABlation.ABLATIONS:
        model = SSIMAGSCDABlation(_cfg(variant), _data_info()).eval()
        analysis = model.analysis(x, edge_index)
        for value in (analysis["z"], analysis["z_text"], analysis["z_visual"]):
            assert torch.isfinite(value).all()
        if variant == "no_semantic_conductance":
            for modality in ("text", "visual"):
                branch = analysis[f"relation_weight_{modality}"]
                assert torch.allclose(branch, torch.ones_like(branch))
                assert torch.allclose(
                    analysis[f"c_{modality}"], torch.zeros_like(analysis[f"c_{modality}"])
                )
                assert torch.allclose(
                    analysis[f"relation_filter_residual_{modality}"],
                    torch.zeros_like(analysis[f"relation_filter_residual_{modality}"]),
                )
        elif variant == "fixed_restart":
            for modality in ("text", "visual"):
                assert all(
                    torch.allclose(alpha, torch.full_like(alpha, 0.1))
                    for alpha in analysis[f"alpha_{modality}"]
                )
                xi = torch.cat(
                    [
                        torch.ones_like(analysis[f"p_{modality}"]).unsqueeze(-1),
                        *[
                            alpha.unsqueeze(-1)
                            for alpha in analysis[f"alpha_{modality}"]
                        ],
                    ],
                    dim=-1,
                )
                centered = xi - xi.mean(dim=-1, keepdim=True)
                expected = (
                    model.reference_filter_scale_text
                    * centered
                    if modality == "text"
                    else model.reference_filter_scale_visual * centered
                )
                assert torch.allclose(
                    analysis[f"reference_residual_{modality}"], expected
                )
        elif variant == "terminal_context_only":
            for modality in ("text", "visual"):
                eta = analysis[f"eta_{modality}"]
                assert torch.allclose(eta[:, :-1], torch.zeros_like(eta[:, :-1]))
                assert torch.allclose(
                    analysis[f"reference_residual_{modality}"],
                    torch.zeros_like(analysis[f"reference_residual_{modality}"]),
                )
                expected = eta[:, -1:].mul(analysis[f"S_{modality}"][-1])
                assert torch.allclose(analysis[f"z_{modality}"], expected)
        elif variant == "global_response_only":
            for modality in ("text", "visual"):
                expected = analysis[f"gamma_{modality}"] + analysis[f"delta_gamma_{modality}"]
                assert torch.allclose(analysis[f"eta_{modality}"], expected)
                for key in ("delta_content", "reference_residual", "relation_filter_residual"):
                    value = analysis[f"{key}_{modality}"]
                    assert torch.allclose(value, torch.zeros_like(value))
        elif variant == "no_formation_conditioning":
            for modality in ("text", "visual"):
                expected = (
                    analysis[f"gamma_{modality}"]
                    + analysis[f"delta_gamma_{modality}"]
                    + analysis[f"delta_content_{modality}"]
                )
                assert torch.allclose(analysis[f"eta_{modality}"], expected)
                for key in ("reference_residual", "relation_filter_residual"):
                    value = analysis[f"{key}_{modality}"]
                    assert torch.allclose(value, torch.zeros_like(value))
        _assert_finite_backward(model, x, edge_index)


def test_generic_ppr_and_gpr_are_clean_controls():
    x, edge_index = _graph()
    for model_cls, name in (
        (SSIMAGGenericPPR, "ssi_mag_generic_ppr"),
        (SSIMAGGenericGPR, "ssi_mag_generic_gpr"),
    ):
        model = model_cls(_generic_cfg(name), _data_info()).eval()
        analysis = model.analysis(x, edge_index)
        assert torch.isfinite(analysis["z"]).all()
        forbidden = ("relation", "semantic", "adaptive", "formation", "attention")
        assert not any(
            any(token in key.lower() for token in forbidden)
            for key in model.state_dict()
        )
        for modality in ("text", "visual"):
            raw_index, raw_weight = model._raw_operator(edge_index, x.size(0), analysis[f"h0_{modality}"].dtype)
            assert torch.equal(raw_index, analysis[f"normalized_edge_index_{modality}"])
            assert torch.allclose(raw_weight, analysis[f"normalized_edge_weight_{modality}"])
            states = analysis[f"S_{modality}"]
            current = states[0]
            for order in range(1, 4):
                propagated = model._propagate_once(
                    current,
                    analysis[f"normalized_edge_index_{modality}"],
                    analysis[f"normalized_edge_weight_{modality}"],
                )
                if name.endswith("ppr"):
                    current = 0.9 * propagated + 0.1 * states[0]
                else:
                    current = propagated
                assert torch.allclose(current, states[order], atol=1e-6, rtol=1e-6)
            if name.endswith("ppr"):
                assert torch.allclose(analysis[f"z_{modality}"], states[-1])
            else:
                expected = sum(
                    analysis[f"gamma_{modality}"][order] * states[order]
                    for order in range(4)
                )
                assert torch.allclose(analysis[f"z_{modality}"], expected)
                with torch.no_grad():
                    getattr(model, f"gamma_{modality}").fill_(-0.25)
                signed = model.analysis(x, edge_index)[f"gamma_{modality}"]
                assert torch.all(signed < 0)
        _assert_finite_backward(model, x, edge_index)


def test_modality_streams_remain_independent_until_late_fusion():
    x, edge_index = _graph()
    for model in (
        SSIMAGSCD(_cfg("full"), _data_info()),
        SSIMAGGenericPPR(_generic_cfg("ssi_mag_generic_ppr"), _data_info()),
        SSIMAGGenericGPR(_generic_cfg("ssi_mag_generic_gpr"), _data_info()),
    ):
        model.eval()
        first = model.analysis(x, edge_index)
        changed = x.clone()
        changed[:, :4] += 10.0
        second = model.analysis(changed, edge_index)
        assert torch.allclose(first["z_visual"], second["z_visual"], atol=1e-6, rtol=1e-6)


def test_lp_sampler_depth_matches_k_three_for_all_control_models():
    cfg = OmegaConf.create({"model": {"num_layers": 3}, "task": {"num_neighbors": [5, 5, 5]}})
    x, edge_index = _graph()
    for model in (
        SSIMAGSCD(_cfg("full"), _data_info()),
        SSIMAGGenericPPR(_generic_cfg("ssi_mag_generic_ppr"), _data_info()),
        SSIMAGGenericGPR(_generic_cfg("ssi_mag_generic_gpr"), _data_info()),
    ):
        assert model.max_order == 3
        assert _resolve_lp_num_neighbors(cfg, model) == [5, 5, 5]
        assert len(_resolve_lp_num_neighbors(cfg, model)) == model.max_order
        _assert_finite_backward(model, x, edge_index)
