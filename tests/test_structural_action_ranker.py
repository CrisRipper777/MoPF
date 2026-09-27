from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scripts import run_routing_identifiability as common
from scripts import structural_observability_data as dataio
from scripts import structural_observability_training as trainio
from src.models.structural_action_ranker import (
    ACTION_NAMES,
    StructuralActionRanker,
    decision_signature,
    execute_hard_action,
    hard_action_bank,
    pairwise_ranking_loss,
    pairwise_weights,
    select_hard_action,
)
from src.models.utility_routing_audit import uniform_state_mix


def _states(n=7, hidden=9):
    gen = torch.Generator().manual_seed(71)
    return [torch.randn(n, hidden, generator=gen) for _ in range(4)]


def test_uniform_action_is_exact_frozen_c0_uniform():
    states = _states()
    torch.testing.assert_close(hard_action_bank(states)[:, 0], uniform_state_mix(states), atol=0, rtol=0)


def test_hard_action_execution_selects_one_state_per_node():
    states = _states()
    bank = hard_action_bank(states)
    actions = torch.tensor([0, 1, 2, 3, 4, 2, 1])
    result = execute_hard_action(states, actions)
    torch.testing.assert_close(result, bank[torch.arange(len(actions)), actions], atol=0, rtol=0)
    assert all(torch.equal(result[i], bank[i, actions[i]]) for i in range(len(actions)))


def test_action_execution_is_hard_and_has_no_probability_mixture_interface():
    states = _states()
    probabilities = torch.full((len(states[0]), 5), 0.2)
    with pytest.raises(ValueError):
        execute_hard_action(states, probabilities)
    assert "action probabilities" not in inspect.signature(execute_hard_action).parameters


def test_decision_signature_is_label_free_and_fixed_to_eleven_features():
    assert set(inspect.signature(decision_signature).parameters) == {"prediction_uniform", "prediction_action"}
    assert "label" not in str(inspect.signature(decision_signature)).lower()
    pu = torch.softmax(torch.randn(8, 5, 4), -1)
    pa = torch.softmax(torch.randn(8, 5, 4), -1)
    signature = decision_signature(pu, pa)
    assert signature.shape == (8, 5, 11)
    assert torch.isfinite(signature).all()


def test_pairwise_supervision_uses_train_action_losses_only():
    scores = torch.randn(6, 5, requires_grad=True)
    losses = torch.rand(6, 5)
    train = torch.tensor([0, 1, 2, 3])
    first = pairwise_ranking_loss(scores, losses, train)
    changed = losses.clone()
    changed[4:] = torch.tensor([[1e4, 0, 2e4, 3e4, 4e4], [4e4, 3e4, 2e4, 1e4, 0]])
    second = pairwise_ranking_loss(scores, changed, train)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_pairwise_weights_sum_to_one_for_each_non_degenerate_node():
    losses = torch.tensor([[0.0, 0.1, 0.2, 0.4, 0.9], [1.0, 0.8, 0.6, 0.4, 0.1]])
    _, _, weights = pairwise_weights(losses)
    torch.testing.assert_close(weights.sum(1), torch.ones(2), atol=1e-7, rtol=1e-7)


def test_zero_difference_node_has_zero_pairwise_ranking_loss():
    scores = torch.randn(2, 5, requires_grad=True)
    losses = torch.stack([torch.ones(5), torch.tensor([0., 1., 1., 2., 3.])])
    zero = pairwise_ranking_loss(scores[:1], losses[:1])
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)
    _, _, weights = pairwise_weights(losses)
    torch.testing.assert_close(weights[0].sum(), torch.tensor(0.0), atol=0, rtol=0)
    both = pairwise_ranking_loss(scores, losses)
    one = pairwise_ranking_loss(scores[1:], losses[1:])
    torch.testing.assert_close(both, one / 2, atol=1e-7, rtol=1e-7)


def test_validation_oracle_cannot_change_train_pairwise_loss():
    scores = torch.randn(5, 5)
    losses = torch.rand(5, 5)
    train = torch.tensor([True, True, True, False, False])
    original = pairwise_ranking_loss(scores, losses, train)
    losses[~train] = torch.randn_like(losses[~train]) * 100
    changed = pairwise_ranking_loss(scores, losses, train)
    torch.testing.assert_close(original, changed, atol=0, rtol=0)


def test_frozen_c0_modules_are_eval_only_and_receive_no_gradients():
    c0 = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU())
    head = torch.nn.Linear(8, 3)
    common.freeze_c0(c0, head)
    assert not c0.training and not head.training
    assert all(not p.requires_grad for p in list(c0.parameters()) + list(head.parameters()))
    x = torch.randn(4, 6)
    logits = head(c0(x))
    assert not logits.requires_grad
    assert all(p.grad is None for p in list(c0.parameters()) + list(head.parameters()))


def test_capacity_control_parameter_count_is_within_five_percent_of_r1():
    r1 = StructuralActionRanker(9, "r1_decision")
    cap = StructuralActionRanker(9, "r1_capacity_control")
    assert abs(cap.trainable_parameter_count - r1.trainable_parameter_count) / r1.trainable_parameter_count <= 0.05


def test_global_action_baseline_uses_only_train_cache_rows(tmp_path, monkeypatch):
    root = tmp_path / "e6"
    cache_dir = root / "phase_b_cache"
    cache_dir.mkdir(parents=True)
    joint = np.zeros((3, 4, 4), dtype=np.float32)
    lt_u = np.array([[0.3, 0.4, 0.1, 0.2], [0.2, 0.3, 0.0, 0.1], [-100, 0, 0, 0]], dtype=np.float32)
    lu_v = np.array([[0.1, 0.3, 0.4, 0.2], [0.2, 0.4, 0.3, 0.1], [-100, 0, 0, 0]], dtype=np.float32)
    luu = np.array([0.8, 0.8, -100.0], dtype=np.float32)
    np.savez(cache_dir / "Toy_seed7.npz", joint=joint, lt_u=lt_u, lu_v=lu_v, luu=luu,
             node_id=np.arange(3), split=np.array(["train", "train", "validation"]))
    monkeypatch.setattr(dataio, "E6_OUT", root)
    monkeypatch.setattr(dataio, "RES", tmp_path / "results")
    monkeypatch.setattr(dataio, "SEEDS", [7])
    choices = trainio.global_action_choices(["Toy"])
    assert choices["Toy"]["text"] == 3
    assert choices["Toy"]["visual"] == 1


def test_confidence_thresholds_are_derived_from_train_only():
    train_conf = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    thresholds = trainio.confidence_coverage_thresholds(train_conf)
    assert thresholds[0.50] == pytest.approx(np.quantile(train_conf, 0.5))
    assert thresholds[1.0] == pytest.approx(np.quantile(train_conf, 0.0))
    assert set(inspect.signature(trainio.confidence_coverage_thresholds).parameters) == {"train_confidence", "coverages"}


def test_exact_score_ties_choose_uniform():
    scores = torch.tensor([[1., 1., 0., 0., 0.], [0., 2., 2., 1., 0.], [1., 2., 2., 0., 0.], [0., 1., 2., 3., 0.]])
    assert select_hard_action(scores).tolist() == [0, 0, 0, 3]


def test_allowed_split_never_touches_test_indices():
    class Guard:
        train_idx = torch.tensor([0, 2])
        val_idx = torch.tensor([1])
        y = torch.tensor([4, 5, 6, 999])

        @property
        def test_idx(self):
            raise AssertionError("test_idx must not be accessed")

    allowed, y, train, val = common._allowed_split(Guard(), torch.device("cpu"))
    assert allowed.tolist() == [0, 2, 1]
    assert y.tolist() == [4, 6, 5]
    assert train.tolist() == [0, 1] and val.tolist() == [2]


def test_checkpoint_reload_preserves_ranker_scores():
    states_t, states_v = _states(), _states()
    signature_t, signature_v = torch.randn(7, 5, 11), torch.randn(7, 5, 11)
    model = StructuralActionRanker(9, "r1_decision").eval()
    with torch.no_grad():
        expected = model(states_t, states_v, signature_t, signature_v)
        reloaded = StructuralActionRanker(9, "r1_decision").eval()
        reloaded.load_state_dict(model.state_dict(), strict=True)
        actual = reloaded(states_t, states_v, signature_t, signature_v)
    torch.testing.assert_close(expected["scores_text"], actual["scores_text"], atol=0, rtol=0)
    torch.testing.assert_close(expected["scores_visual"], actual["scores_visual"], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["r0_trajectory", "r1_capacity_control", "r1_decision"])
def test_all_ranker_variants_have_finite_forward_and_backward(mode):
    states_t, states_v = _states(n=5), _states(n=5)
    sig_t, sig_v = torch.randn(5, 5, 11), torch.randn(5, 5, 11)
    model = StructuralActionRanker(9, mode)
    out = model(states_t, states_v, sig_t if mode == "r1_decision" else None,
                sig_v if mode == "r1_decision" else None)
    loss = out["scores_text"].square().mean() + out["scores_visual"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_corrected_restart_audit_groups_by_chunk_before_range():
    frame = __import__("pandas").DataFrame([
        {"dataset": "D", "seed": 1, "split": "train", "level": 2, "chunk_start": 0, "restart": r, "mean_best_loss": v}
        for r, v in enumerate([1.0, 1.1, 1.2])
    ] + [
        {"dataset": "D", "seed": 1, "split": "train", "level": 2, "chunk_start": 8, "restart": r, "mean_best_loss": v}
        for r, v in enumerate([10.0, 10.1, 10.2])
    ])
    ranges = dataio.corrected_restart_ranges(frame)
    assert len(ranges) == 2
    np.testing.assert_allclose(ranges.restart_range.to_numpy(), [0.2, 0.2], atol=1e-12)
