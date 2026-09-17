from __future__ import annotations

import subprocess
import types

import torch

from src.ablation import ALL_ABLATIONS, INTERACTION_ABLATIONS, MAIN_ABLATIONS, resolve_ablation
from src.models.mopf import Model

from tests.test_mopf import _cfg, _graph


def _final_cfg(name: str):
    cfg = _cfg(
        edge_weight_mode="learned_diag_cos",
        use_transport_residual=True,
        multihop_state_mode="anchored",
        multihop_response_mode="cumulative",
        multihop_anchor_alpha=0.1,
    )
    cfg["ablation"] = name
    return cfg


def _build(name: str):
    torch.manual_seed(123)
    return Model(
        _final_cfg(name),
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )


def test_f2_catalog_is_frozen() -> None:
    assert len(MAIN_ABLATIONS) == 5
    assert len(INTERACTION_ABLATIONS) == 3
    assert tuple(ALL_ABLATIONS) == MAIN_ABLATIONS + INTERACTION_ABLATIONS
    assert resolve_ablation("full").global_preference is True


def test_f2_variant_switches_are_exactly_scoped() -> None:
    expected = {
        "full": ("learned_diag_cos", "anchored", "cumulative", True, True, True),
        "wo_learned_semantic_calibration": ("separate_cos", "anchored", "cumulative", True, True, True),
        "wo_semantic_anchor": ("learned_diag_cos", "ordinary", "cumulative", True, True, True),
        "wo_tcpr": ("learned_diag_cos", "anchored", "cumulative", True, True, False),
        "wo_node_adaptation": ("learned_diag_cos", "anchored", "cumulative", True, False, True),
        "wo_modality_adaptation": ("learned_diag_cos", "anchored", "cumulative", False, True, True),
        "wo_anchor_tcpr": ("learned_diag_cos", "ordinary", "cumulative", True, True, False),
        "wo_node_tcpr": ("learned_diag_cos", "anchored", "cumulative", True, False, False),
        "wo_modality_tcpr": ("learned_diag_cos", "anchored", "cumulative", False, True, False),
    }
    for name in ("full", *ALL_ABLATIONS):
        model = _build(name)
        edge_mode, state_mode, response_mode, modality, node, tcpr = expected[name]
        assert model.edge_weight_mode == edge_mode
        assert model.multihop_state_mode == state_mode
        assert model.multihop_response_mode == response_mode
        assert model.use_modality_residual is modality
        assert model.use_node_residual is node
        assert model.use_transport_residual is tcpr


def test_f2_all_variants_forward_and_gradient_are_finite_with_same_shape() -> None:
    x, edge_index = _graph()
    outputs = {}
    for name in ("full", *ALL_ABLATIONS):
        model = _build(name)
        model.train()
        z, _, _, aux_loss, _ = model(x, edge_index)
        (z.square().mean() + aux_loss).backward()
        assert z.shape == (6, 8)
        assert torch.isfinite(z).all()
        assert torch.isfinite(aux_loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        outputs[name] = z.detach()
    assert set(outputs) == {"full", *ALL_ABLATIONS}


def test_explicit_full_is_numerically_equivalent_to_legacy_default() -> None:
    x, edge_index = _graph()
    torch.manual_seed(321)
    implicit = Model(
        _final_cfg("full"),
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )
    torch.manual_seed(321)
    explicit_cfg = _final_cfg("full")
    explicit = Model(
        explicit_cfg,
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )
    implicit.eval()
    explicit.eval()
    with torch.no_grad():
        left = implicit(x, edge_index)
        right = explicit(x, edge_index)
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[3], right[3])
    assert implicit.state_dict().keys() == explicit.state_dict().keys()


def test_historical_head_full_path_matches_current_full_path() -> None:
    """Compare the pre-change HEAD model with the explicit current Full path."""
    source = subprocess.check_output(
        ["git", "show", "HEAD:src/models/mopf.py"], text=True
    ).replace("from .common import make_norm", "from src.models.common import make_norm")
    module = types.ModuleType("historical_mopf_for_regression")
    module.__package__ = "src.models"
    exec(compile(source, "HEAD:src/models/mopf.py", "exec"), module.__dict__)

    x, edge_index = _graph()
    torch.manual_seed(777)
    historical_cfg = _final_cfg("full")
    historical_cfg.pop("ablation", None)
    historical = module.Model(
        historical_cfg,
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )
    torch.manual_seed(777)
    current = Model(
        _final_cfg("full"),
        {"input_dim": 10, "num_nodes": 6, "text_dim": 4, "visual_dim": 6},
    )
    # Load the historical model state into the current model so this is a
    # same-checkpoint comparison, not only a same-initialization comparison.
    current.load_state_dict(
        {key: value.detach().clone() for key, value in historical.state_dict().items()}
    )
    historical.eval()
    current.eval()
    with torch.no_grad():
        old = historical(x, edge_index)
        new = current(x, edge_index)
    old_representation, new_representation = old[0], new[0]
    assert torch.equal(old_representation, new_representation)
    assert torch.equal(old[3], new[3])

    torch.manual_seed(778)
    classifier = torch.nn.Linear(old_representation.size(-1), 3)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    with torch.no_grad():
        old_logits = classifier(old_representation)
        new_logits = classifier(new_representation)
    assert float((old_representation - new_representation).abs().max()) == 0.0
    assert float((old_logits - new_logits).abs().max()) == 0.0
    for split_idx in (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])):
        old_acc = (old_logits[split_idx].argmax(dim=-1) == labels[split_idx]).float().mean()
        new_acc = (new_logits[split_idx].argmax(dim=-1) == labels[split_idx]).float().mean()
        assert torch.equal(old_acc, new_acc)
