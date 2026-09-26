from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts.run_utility_routing_audit import (
    build_eprop,
    select_train_val_labels,
    teacher_from_ce,
    utility_distillation_loss,
)
from src.models.cmrf_probe import Model as C0Model
from src.models.utility_routing_audit import (
    UtilityRouter,
    action_bank,
    expected_mixture,
    safe_probabilities,
    uniform_state_mix,
)


def _c0(hidden_dim=8, num_classes=3):
    cfg = OmegaConf.create({
        "model": {
            "hidden_dim": hidden_dim,
            "max_order": 3,
            "dropout": 0.0,
            "composition_mode": "uniform",
            "controller_latent_dim": 32,
            "controller_hidden_dim": 64,
            "controller_lambda": 0.25,
        }
    })
    info = {
        "input_dim": 7, "num_nodes": 6, "num_classes": num_classes,
        "text_dim": 3, "visual_dim": 4,
    }
    return C0Model(cfg, info), info


def _states(n=8, hidden=8):
    generator = torch.Generator().manual_seed(19)
    return [torch.randn(n, hidden, generator=generator) for _ in range(4)]


def test_c0_checkpoint_exact_reload(tmp_path):
    torch.manual_seed(4)
    c0, info = _c0()
    head = torch.nn.Linear(c0.out_dim, info["num_classes"])
    path = tmp_path / "c0.pt"
    torch.save({"model_state": c0.state_dict(), "head_state": head.state_dict()}, path)
    c0_copy, _ = _c0()
    head_copy = torch.nn.Linear(c0_copy.out_dim, info["num_classes"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    c0_copy.load_state_dict(payload["model_state"], strict=True)
    head_copy.load_state_dict(payload["head_state"], strict=True)
    x = torch.randn(6, 7)
    edge = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    c0.eval(); c0_copy.eval(); head.eval(); head_copy.eval()
    with torch.no_grad():
        a = c0.analyze(x, edge)
        b = c0_copy.analyze(x, edge)
        torch.testing.assert_close(a["fused_z"], b["fused_z"], atol=0, rtol=0)
        torch.testing.assert_close(
            head(a["fused_z"]).softmax(-1), head_copy(b["fused_z"]).softmax(-1),
            atol=0, rtol=0,
        )


def test_action_bank_contains_each_state():
    states = _states()
    bank = action_bank(states)
    assert bank.shape == (8, 5, 8)
    for k in range(4):
        torch.testing.assert_close(bank[:, k], states[k], atol=0, rtol=0)


def test_au_is_exact_c0_uniform_and_close_to_arithmetic_average():
    states = _states()
    bank = action_bank(states)
    torch.testing.assert_close(bank[:, 4], uniform_state_mix(states), atol=0, rtol=0)
    torch.testing.assert_close(
        bank[:, 4], torch.stack(states, dim=1).mean(dim=1), atol=1e-6, rtol=1e-6
    )


def test_text_action_changes_only_text_representation():
    states_t = _states()
    states_v = _states()
    bt, bv = action_bank(states_t), action_bank(states_v)
    p_t = F.one_hot(torch.full((8,), 2), 5).float()
    p_v = F.one_hot(torch.full((8,), 4), 5).float()
    zt = expected_mixture(bt, p_t)
    zv = expected_mixture(bv, p_v)
    torch.testing.assert_close(zv, bv[:, 4], atol=0, rtol=0)
    assert not torch.equal(zt, bt[:, 4])


def test_visual_action_changes_only_visual_representation():
    states_t = _states()
    states_v = _states()
    bt, bv = action_bank(states_t), action_bank(states_v)
    p_t = F.one_hot(torch.full((8,), 4), 5).float()
    p_v = F.one_hot(torch.full((8,), 1), 5).float()
    zt = expected_mixture(bt, p_t)
    zv = expected_mixture(bv, p_v)
    torch.testing.assert_close(zt, bt[:, 4], atol=0, rtol=0)
    assert not torch.equal(zv, bv[:, 4])


def test_uniform_probability_simplex_is_valid():
    states_t, states_v = _states(), _states()
    router = UtilityRouter(8, 3, "flat")
    out = router(states_t, states_v)
    torch.testing.assert_close(out["prob_text"], torch.full((8, 5), 0.2))
    torch.testing.assert_close(out["prob_visual"], torch.full((8, 5), 0.2))


def test_soft_teacher_is_normalized_and_matches_true_class_probabilities():
    torch.manual_seed(1)
    logits = torch.randn(13, 5, 4)
    y = torch.randint(0, 4, (13,))
    ce = F.cross_entropy(logits.reshape(-1, 4), y[:, None].expand(-1, 5).reshape(-1), reduction="none").reshape(13, 5)
    q, entropy, strength, _, _ = teacher_from_ce(ce)
    true_p = logits.softmax(-1).gather(-1, y[:, None, None].expand(-1, 5, 1)).squeeze(-1)
    expected_q = true_p / true_p.sum(-1, keepdim=True)
    torch.testing.assert_close(q.sum(-1), torch.ones(13))
    torch.testing.assert_close(q, expected_q, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(entropy).all()


def test_preference_strength_is_bounded():
    ce = torch.rand(50, 5)
    _, _, strength, _, _ = teacher_from_ce(ce)
    assert ((strength >= 0) & (strength <= 1)).all()


def test_validation_teacher_values_do_not_enter_training_kd():
    torch.manual_seed(11)
    p_t = torch.softmax(torch.randn(8, 5), -1).requires_grad_()
    p_v = torch.softmax(torch.randn(8, 5), -1).requires_grad_()
    q_t = torch.softmax(torch.randn(8, 5), -1)
    q_v = torch.softmax(torch.randn(8, 5), -1)
    w_t = torch.rand(8); w_v = torch.rand(8)
    train = torch.tensor([0, 1, 2, 3])
    first = utility_distillation_loss(p_t, p_v, q_t, q_v, w_t, w_v, train, weighted=True)
    q_t[4:] = torch.softmax(torch.randn(4, 5) * 100, -1)
    q_v[4:] = torch.softmax(torch.randn(4, 5) * 100, -1)
    w_t[4:] = 0; w_v[4:] = 1e6
    second = utility_distillation_loss(p_t, p_v, q_t, q_v, w_t, w_v, train, weighted=True)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_safe_one_hot_uniform_fallback_is_exact():
    p = torch.zeros(10, 5); p[:, 4] = 1
    safe = safe_probabilities(p)
    torch.testing.assert_close(safe, p, atol=0, rtol=0)
    bank = action_bank(_states(n=10))
    torch.testing.assert_close(expected_mixture(bank, safe), bank[:, 4], atol=0, rtol=0)


def test_train_val_label_selector_never_reads_other_positions():
    labels = torch.tensor([0, 1, 2, 0, 77, 88])
    train = torch.tensor([0, 1, 2])
    val = torch.tensor([3])
    allowed, y = select_train_val_labels(labels, train, val)
    assert allowed.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [0, 1, 2, 0]
    assert 77 not in y.tolist() and 88 not in y.tolist()


def test_frozen_c0_parameters_receive_no_gradients():
    c0, info = _c0()
    for p in c0.parameters():
        p.requires_grad_(False)
    head = torch.nn.Linear(c0.out_dim, info["num_classes"])
    router = UtilityRouter(8, info["num_classes"], "flat")
    states_t, states_v = _states(), _states()
    out = router(states_t, states_v)
    zt = expected_mixture(action_bank(states_t), out["prob_text"])
    zv = expected_mixture(action_bank(states_v), out["prob_visual"])
    logits = head(c0._fuse(zt, zv))
    logits.square().mean().backward()
    assert all(p.grad is None for p in c0.parameters())
    assert any(p.grad is not None for p in router.parameters())


def test_eprop_is_symmetric_coalesced_and_has_no_self_loops():
    edge = torch.tensor([[0, 1, 1, 2, 2, 2], [0, 0, 2, 1, 2, 3]])
    eprop = build_eprop(edge, 4)
    assert not torch.any(eprop[0] == eprop[1])
    pairs = set(map(tuple, eprop.t().tolist()))
    assert all((v, u) in pairs for u, v in pairs)
    assert eprop.size(1) == len(pairs)


def test_relation_context_does_not_mutate_eprop_or_states():
    n, hidden = 7, 8
    states_t, states_v = _states(n, hidden), _states(n, hidden)
    original_t = [s.clone() for s in states_t]
    original_v = [s.clone() for s in states_v]
    edge = build_eprop(torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]), n)
    edge_before = edge.clone()
    router = UtilityRouter(hidden, 3, "relation")
    out = router(
        states_t, states_v, edge_index=edge,
        h0_text_all=torch.randn(n, hidden), h0_visual_all=torch.randn(n, hidden),
        allowed_idx=torch.arange(n),
    )
    assert torch.equal(edge, edge_before)
    for a, b in zip(states_t, original_t):
        assert torch.equal(a, b)
    for a, b in zip(states_v, original_v):
        assert torch.equal(a, b)
    assert out["pooled_relation_text"].shape == (n, 65)
    assert out["prob_text"].shape == (n, 5)


def test_relation_context_never_changes_propagation_states():
    states_t, states_v = _states(), _states()
    before_t = [s.clone() for s in states_t]
    before_v = [s.clone() for s in states_v]
    router = UtilityRouter(8, 3, "relation")
    edge = build_eprop(torch.tensor([[0,1,2,3,4,5,6],[1,2,3,4,5,6,7]]), 8)
    router(states_t, states_v, edge_index=edge, h0_text_all=torch.randn(8,8),
           h0_visual_all=torch.randn(8,8), allowed_idx=torch.arange(8))
    assert all(torch.equal(a,b) for a,b in zip(states_t,before_t))
    assert all(torch.equal(a,b) for a,b in zip(states_v,before_v))


def test_router_checkpoint_reload_equivalence(tmp_path):
    torch.manual_seed(9)
    model = UtilityRouter(8, 3, "decision_response").eval()
    state = _states()
    tokens = torch.randn(8, 5, 6)
    out = model(state, state, decision_text=tokens, decision_visual=tokens)
    path = tmp_path / "router.pt"
    torch.save(model.state_dict(), path)
    restored = UtilityRouter(8, 3, "decision_response").eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    out2 = restored(state, state, decision_text=tokens, decision_visual=tokens)
    torch.testing.assert_close(out["prob_text"], out2["prob_text"], atol=0, rtol=0)
    torch.testing.assert_close(out["prob_visual"], out2["prob_visual"], atol=0, rtol=0)


def test_capacity_controls_match_augmented_context_parameters_within_five_percent():
    counts = {}
    for mode in ("decision_response", "decision_capacity", "relation", "relation_capacity"):
        model = UtilityRouter(256, 6, mode)
        counts[mode] = sum(p.numel() for p in model.parameters())
    for actual, control in (
        ("decision_response", "decision_capacity"),
        ("relation", "relation_capacity"),
    ):
        diff = abs(counts[actual] - counts[control]) / max(counts[actual], counts[control])
        assert diff <= 0.05, f"{actual}={counts[actual]} {control}={counts[control]} diff={diff:.3%}"


def test_all_router_modes_forward_backward_are_finite():
    states_t, states_v = _states(), _states()
    decision = torch.randn(8, 5, 6)
    edge = build_eprop(torch.tensor([[0,1,2,3,4,5,6],[1,2,3,4,5,6,7]]), 8)
    for mode in ("flat", "decision_response", "decision_capacity", "relation", "relation_capacity"):
        model = UtilityRouter(8, 3, mode)
        out = model(
            states_t, states_v,
            decision_text=decision, decision_visual=decision,
            edge_index=edge, h0_text_all=states_t[0],
            h0_visual_all=states_v[0], allowed_idx=torch.arange(8),
        )
        loss = out["prob_text"].square().mean() + out["prob_visual"].square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
