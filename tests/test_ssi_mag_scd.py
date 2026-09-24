from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.ssi_mag_final_ablation import SSIMAGFinalAblation
from src.models.ssi_mag_scd import SSIMAGSCD


def _cfg(name: str = "ssi_mag_scd", ablation: str = "full"):
    base = OmegaConf.load(f"configs/model/{name}.yaml")
    cfg = OmegaConf.create({"ablation": ablation, "model": OmegaConf.to_container(base, resolve=False)})
    cfg.model.hidden_dim = 8
    cfg.model.relation_rank = 3
    cfg.model.filter_rank = 2
    cfg.model.relation_edge_chunk_size = 2
    cfg.model.num_layers = 3
    cfg.model.max_order = 3
    return cfg


def _inputs():
    torch.manual_seed(1811)
    x = torch.randn(6, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0, 4, 5], [1, 0, 2, 1, 3, 2, 0, 3, 4, 5]],
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


def test_scd_deletes_all_dead_modules_and_has_clean_graph_flags():
    model = SSIMAGSCD(_cfg(), _info())
    assert model.context_change_present is False
    assert model.cross_hop_interaction_present is False
    assert all(not hasattr(model, name) for name in model.DEAD_COMPONENT_NAMES)
    names = set(model.state_dict())
    assert not any(
        name.startswith(prefix) or f".{prefix}." in name
        for name in names
        for prefix in model.DEAD_COMPONENT_NAMES
    )
    assert not any(token in name for name in names for token in ("query", "key", "value", "attention"))


def test_scd_constructor_preserves_common_active_initialization():
    torch.manual_seed(29)
    historical = SSIMAGFinalAblation(_cfg("ssi_mag_final_ablation", "no_cross_hop_interaction"), _info())
    torch.manual_seed(29)
    clean = SSIMAGSCD(_cfg(), _info())
    clean_keys = set(clean.state_dict())
    historical_keys = set(historical.state_dict())
    dropped = historical_keys - clean_keys
    assert dropped
    assert all(
        any(key == prefix or key.startswith(prefix + ".") for prefix in clean.DEAD_COMPONENT_NAMES)
        for key in dropped
    )
    assert clean_keys <= historical_keys
    for key in sorted(clean_keys):
        torch.testing.assert_close(clean.state_dict()[key], historical.state_dict()[key], rtol=0.0, atol=0.0)


def test_scd_relation_and_restart_formulas_are_exact():
    x, edge_index = _inputs()
    model = SSIMAGSCD(_cfg(), _info())
    with torch.no_grad():
        model.relation_scorer_text.weight.copy_(torch.tensor([[0.2, -0.3, 0.1]]))
        model.relation_scorer_text.bias.fill_(0.05)
    analysis = model.analysis(x, edge_index)
    for modality in ("text", "visual"):
        relation_nonself = analysis[f"physical_edge_index"][0] != analysis[f"physical_edge_index"][1]
        a = analysis[f"a_{modality}"]
        assert torch.equal(a[~relation_nonself], torch.zeros_like(a[~relation_nonself]))
        expected_weight = torch.exp(a)
        torch.testing.assert_close(analysis[f"relation_weight_{modality}"], expected_weight)
        c = analysis[f"c_{modality}"]
        expected_c = torch.zeros_like(c)
        count = torch.zeros_like(c)
        src, dst = analysis[f"physical_edge_index"][:, relation_nonself]
        value = a[relation_nonself].abs()
        expected_c.index_add_(0, src, value)
        expected_c.index_add_(0, dst, value)
        count.index_add_(0, src, torch.ones_like(value))
        count.index_add_(0, dst, torch.ones_like(value))
        expected_c = torch.where(count > 0, expected_c / count.clamp_min(1), expected_c)
        torch.testing.assert_close(c, expected_c)

        alpha_bank = torch.cat(
            [torch.ones_like(analysis[f"p_{modality}"]).unsqueeze(-1)]
            + [value.unsqueeze(-1) for value in analysis[f"alpha_{modality}"]], dim=-1
        )
        expected_ref = getattr(model, f"reference_filter_scale_{modality}") * (
            alpha_bank - alpha_bank.mean(dim=-1, keepdim=True)
        )
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
            analysis[f"z_{modality}"], model._compose(analysis[f"S_{modality}"], expected_eta)
        )
        torch.testing.assert_close(analysis[f"S_used_{modality}"], analysis[f"S_{modality}"])
        assert torch.isfinite(analysis[f"eta_{modality}"]).all()


def test_scd_has_no_attention_or_context_change_analysis_outputs_and_intervention_is_safe():
    x, edge_index = _inputs()
    model = SSIMAGSCD(_cfg(), _info())
    analysis = model.analysis_intervention(x, edge_index, interaction="off", reference="off")
    _assert_tree_finite(analysis)
    forbidden = ("attention_", "tokens_", "D_", "S_tilde_", "interaction_output_", "context_change_injection_")
    assert not any(key.startswith(forbidden) for key in analysis)
    for modality in ("text", "visual"):
        torch.testing.assert_close(
            analysis[f"reference_residual_{modality}"],
            torch.zeros_like(analysis[f"reference_residual_{modality}"]),
        )
    assert analysis["context_change_present"] is False
    assert analysis["cross_hop_interaction_present"] is False


def test_scd_forward_backward_and_relation_off_are_finite():
    x, edge_index = _inputs()
    model = SSIMAGSCD(_cfg(), _info())
    z, _, _, aux, _ = model(x, edge_index)
    (z.square().mean() + aux).backward()
    assert torch.isfinite(z).all()
    _assert_tree_finite(model.analysis(x, edge_index))
    relation_off = model.analysis_intervention(x, edge_index, relation="off")
    for modality in ("text", "visual"):
        torch.testing.assert_close(relation_off[f"relation_weight_{modality}"], torch.ones_like(relation_off[f"relation_weight_{modality}"]))
        torch.testing.assert_close(relation_off[f"c_{modality}"], torch.zeros_like(relation_off[f"c_{modality}"]))
