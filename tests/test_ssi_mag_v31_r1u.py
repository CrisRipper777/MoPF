from __future__ import annotations

import math

import torch
from omegaconf import OmegaConf

from src.models.ssi_mag_v31 import SSIMAGV31
from src.models.ssi_mag_v31_controls import SSIMAGV31Controls
from src.models.ssi_mag_v31_r1u import SSIMAGV31R1U


def _cfg():
    base = OmegaConf.load("configs/model/ssi_mag_v31_r1u.yaml")
    cfg = OmegaConf.create({"ablation": "full", "model": OmegaConf.to_container(base, resolve=False)})
    cfg.model.hidden_dim = 8
    cfg.model.relation_rank = 3
    cfg.model.filter_rank = 2
    cfg.model.relation_edge_chunk_size = 2
    cfg.model.num_layers = 3
    cfg.model.max_order = 3
    return cfg


def _inputs():
    torch.manual_seed(211)
    x = torch.randn(6, 10)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 0, 1, 2, 2, 1, 2, 3, 3, 2, 0, 3, 4],
            [1, 0, 2, 1, 0, 1, 3, 2, 3, 2, 0, 3, 3, 0, 4],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def _info():
    return {"input_dim": 10, "text_dim": 4, "visual_dim": 6, "num_nodes": 6, "num_classes": 3}


def test_zero_init_has_no_beta_and_exact_unit_relation_operator():
    x, edge_index = _inputs()
    model = SSIMAGV31R1U(_cfg(), _info())
    assert not any("theta_beta" in name for name, _ in model.named_parameters())
    assert not hasattr(model, "theta_beta_text")
    assert not hasattr(model, "theta_beta_visual")
    h0 = model.text_proj(x[:, :4])
    out = model._relation_edge_outputs(h0, edge_index, "text")
    assert torch.equal(out["relation_score"], torch.zeros_like(out["relation_score"]))
    assert torch.equal(out["relation_weight"], torch.ones_like(out["relation_weight"]))
    learned_index, learned_weight = model._normalized_operator(edge_index, out["relation_weight"], 6, x.dtype)
    raw_index, raw_weight = model._normalized_operator(edge_index, torch.ones(edge_index.size(1)), 6, x.dtype)
    torch.testing.assert_close(learned_index, raw_index)
    torch.testing.assert_close(learned_weight, raw_weight)


def test_relation_formula_is_symmetric_bounded_and_c_is_mean_abs_score():
    x, edge_index = _inputs()
    model = SSIMAGV31R1U(_cfg(), _info())
    with torch.no_grad():
        model.relation_scorer_text.weight.copy_(torch.tensor([[0.2, -0.3, 0.1]]))
        model.relation_scorer_text.bias.fill_(0.05)
    h0 = model.text_proj(x[:, :4])
    out = model._relation_edge_outputs(h0, edge_index, "text")
    edge_to_score = {(int(a), int(b)): value for (a, b), value in zip(edge_index.t().tolist(), out["relation_score"])}
    for (a, b), value in edge_to_score.items():
        if (b, a) in edge_to_score:
            torch.testing.assert_close(value, edge_to_score[(b, a)], rtol=0.0, atol=1.0e-6)
    nonself = edge_index[0] != edge_index[1]
    assert bool((out["relation_weight"][nonself] > math.exp(-1.0)).all())
    assert bool((out["relation_weight"][nonself] < math.exp(1.0)).all())
    self_mask = ~nonself
    torch.testing.assert_close(out["relation_score"][self_mask], torch.zeros_like(out["relation_score"][self_mask]))
    torch.testing.assert_close(out["relation_weight"][self_mask], torch.ones_like(out["relation_weight"][self_mask]))
    c = model._local_adaptation(edge_index, out["relation_residual"], out["beta"], out["nonself_mask"], 6)
    expected = torch.zeros(6)
    count = torch.zeros(6)
    for idx in torch.where(nonself)[0]:
        src, dst = edge_index[:, idx]
        value = out["relation_score"][idx].abs()
        expected[src] += value
        expected[dst] += value
        count[src] += 1
        count[dst] += 1
    expected = torch.where(count > 0, expected / count.clamp_min(1), torch.zeros_like(expected))
    torch.testing.assert_close(c, expected)
    assert c[5].item() == 0.0


def test_scorer_and_delayed_projection_gradients_are_live():
    x, edge_index = _inputs()
    model = SSIMAGV31R1U(_cfg(), _info())
    h0 = model.text_proj(x[:, :4])
    coeff = torch.arange(edge_index.size(1), dtype=x.dtype)
    out = model._relation_edge_outputs(h0, edge_index, "text")
    loss = (out["relation_weight"] * coeff).sum()
    loss.backward()
    assert model.relation_scorer_text.weight.grad is not None
    assert torch.isfinite(model.relation_scorer_text.weight.grad).all()
    assert model.relation_scorer_text.weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    optimizer.zero_grad()
    h0 = model.text_proj(x[:, :4])
    out = model._relation_edge_outputs(h0, edge_index, "text")
    (out["relation_weight"] * coeff).sum().backward()
    grad = model.relation_proj_text.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_r2_reference_eta_and_composition_match_frozen_formulas():
    x, edge_index = _inputs()
    torch.manual_seed(17)
    model = SSIMAGV31R1U(_cfg(), _info())
    analysis = model.analysis(x, edge_index)
    for modality in ("text", "visual"):
        alpha_bank = torch.cat(
            [torch.ones_like(analysis[f"p_{modality}"]).unsqueeze(-1)]
            + [value.unsqueeze(-1) for value in analysis[f"alpha_{modality}"]], dim=-1
        )
        centered = alpha_bank - alpha_bank.mean(dim=-1, keepdim=True)
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
        torch.testing.assert_close(
            analysis[f"z_{modality}"], model._compose(analysis[f"S_tilde_{modality}"], expected_eta)
        )


def test_r1u_r2_and_stage_two_match_v31_at_zero_relation_score():
    x, edge_index = _inputs()
    torch.manual_seed(23)
    frozen = SSIMAGV31(_cfg(), _info())
    torch.manual_seed(23)
    u = SSIMAGV31R1U(_cfg(), _info())
    frozen_analysis = frozen.analysis(x, edge_index)
    u_analysis = u.analysis(x, edge_index)
    for modality in ("text", "visual"):
        for key in ("p", "alpha", "Q", "S", "D", "S_tilde"):
            left = frozen_analysis[f"{key}_{modality}"]
            right = u_analysis[f"{key}_{modality}"]
            if isinstance(left, list):
                for a, b in zip(left, right, strict=True):
                    torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)
            else:
                torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_forward_backward_and_analysis_are_finite():
    x, edge_index = _inputs()
    model = SSIMAGV31R1U(_cfg(), _info())
    z, _, _, aux, _ = model(x, edge_index)
    (z.square().mean() + aux).backward()
    analysis = model.analysis(x, edge_index)
    assert torch.isfinite(z).all()
    assert all(torch.isfinite(v).all() for v in analysis.values() if torch.is_tensor(v))
    off = model.analysis_intervention(x, edge_index, relation="off", reference="off")
    assert torch.isfinite(off["z"]).all()
