from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts import run_routing_identifiability as common
from scripts.e6_srij_training import train_only_cross_entropy
from src.models.canonical_routing_audit import (
    CanonicalRoutingAudit,
    canonical_mix,
    e5_probabilities_to_alpha,
)


def _states(n=9, hidden=12):
    generator = torch.Generator().manual_seed(711)
    return [torch.randn(n, hidden, generator=generator) for _ in range(4)]


def test_e5_canonical_mapping_and_redundant_examples():
    p_flat = torch.full((5, 5), 0.2)
    p_uniform_action = torch.zeros(5, 5)
    p_uniform_action[:, 4] = 1.0
    alpha_flat = e5_probabilities_to_alpha(p_flat)
    alpha_uniform_action = e5_probabilities_to_alpha(p_uniform_action)
    torch.testing.assert_close(alpha_flat, alpha_uniform_action, atol=0, rtol=0)
    torch.testing.assert_close(alpha_flat, torch.full((5, 4), 0.25), atol=0, rtol=0)
    torch.testing.assert_close(alpha_flat.sum(-1), torch.ones(5), atol=0, rtol=0)


def test_e5_action_embedding_equals_canonical_alpha_embedding():
    torch.manual_seed(4)
    states = _states()
    p = torch.softmax(torch.randn(9, 5), dim=-1)
    alpha = e5_probabilities_to_alpha(p)
    au = canonical_mix(states, torch.full((9, 4), 0.25))
    actions = torch.stack([*states, au], dim=1)
    old_embedding = (actions * p.unsqueeze(-1)).sum(1)
    new_embedding = canonical_mix(states, alpha)
    torch.testing.assert_close(old_embedding, new_embedding, atol=1e-6, rtol=1e-6)


def test_uniform_canonical_alpha_reproduces_c0_uniform_operation():
    states = _states()
    alpha = torch.full((len(states[0]), 4), 0.25)
    expected = states[0]
    for coefficient, low, high in zip((0.75, 0.50, 0.25), states[:-1], states[1:]):
        expected = expected + coefficient * (high - low)
    torch.testing.assert_close(canonical_mix(states, alpha), expected, atol=0, rtol=0)


def test_independent_zero_initialized_router_reproduces_uniform_states():
    states_t, states_v = _states(), _states()
    router = CanonicalRoutingAudit(hidden_dim=12, mode="independent").eval()
    out = router(states_t, states_v)
    torch.testing.assert_close(out["alpha_text"], torch.full((9, 4), 0.25), atol=0, rtol=0)
    torch.testing.assert_close(out["alpha_visual"], torch.full((9, 4), 0.25), atol=0, rtol=0)
    torch.testing.assert_close(canonical_mix(states_t, out["alpha_text"]), canonical_mix(states_t, torch.full((9, 4), .25)), atol=0, rtol=0)


def test_joint_action_matrix_shape_and_marginal_residual_identity():
    losses = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    assert losses.shape == (2, 4, 4)
    marginal_t = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    marginal_v = torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)
    uniform = torch.tensor([2.0, 3.0])
    prediction = marginal_t[:, :, None] + marginal_v[:, None, :] - uniform[:, None, None]
    residual = losses - prediction
    assert residual.shape == (2, 4, 4)
    torch.testing.assert_close(losses, prediction + residual, atol=0, rtol=0)


def test_simplex_mixtures_include_discrete_vertices():
    states = _states(n=4)
    vertices = torch.eye(4)
    mixed = canonical_mix(states, vertices)
    for k in range(4):
        torch.testing.assert_close(mixed[k], states[k][k], atol=0, rtol=0)
    alpha = torch.softmax(torch.randn(4, 4), dim=-1)
    torch.testing.assert_close(alpha.sum(-1), torch.ones(4))
    assert torch.all(alpha >= 0)


def test_frozen_parameters_and_finite_forward_backward():
    states_t, states_v = _states(), _states()
    frozen = torch.nn.Linear(12, 3)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    router = CanonicalRoutingAudit(hidden_dim=12, mode="independent")
    out = router(states_t, states_v)
    zt = canonical_mix(states_t, out["alpha_text"])
    zv = canonical_mix(states_v, out["alpha_visual"])
    loss = F.cross_entropy(frozen(zt + zv), torch.arange(9) % 3)
    loss.backward()
    assert all(parameter.grad is None for parameter in frozen.parameters())
    assert any(parameter.grad is not None for parameter in router.parameters())
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in router.parameters() if parameter.grad is not None)


def test_joint_companion_context_override_only_changes_requested_input():
    states_t, states_v = _states(), _states()
    router = CanonicalRoutingAudit(hidden_dim=12, mode="joint").eval()
    with torch.no_grad():
        torch.nn.init.normal_(router.text_head.layers[-1].weight, std=0.03)
        torch.nn.init.normal_(router.text_head.layers[-1].bias, std=0.03)
    base = router(states_t, states_v)
    perm = torch.tensor([2, 3, 0, 1, 8, 4, 5, 6, 7])
    changed = router(states_t, states_v, companion_visual=base["context_visual"][perm])
    torch.testing.assert_close(changed["context_text"], base["context_text"], atol=0, rtol=0)
    torch.testing.assert_close(changed["context_visual"], base["context_visual"], atol=0, rtol=0)
    torch.testing.assert_close(changed["alpha_visual"], base["alpha_visual"], atol=0, rtol=0)
    assert not torch.allclose(changed["alpha_text"], base["alpha_text"])


def test_checkpoint_reload_equivalence(tmp_path):
    router = CanonicalRoutingAudit(hidden_dim=12, mode="capacity_control").eval()
    states_t, states_v = _states(), _states()
    expected = router(states_t, states_v)
    path = tmp_path / "canonical_router.pt"
    torch.save(router.state_dict(), path)
    clone = CanonicalRoutingAudit(hidden_dim=12, mode="capacity_control").eval()
    clone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    actual = clone(states_t, states_v)
    torch.testing.assert_close(expected["alpha_text"], actual["alpha_text"], atol=0, rtol=0)
    torch.testing.assert_close(expected["alpha_visual"], actual["alpha_visual"], atol=0, rtol=0)


def test_d1_head_parameter_capacity_control_is_within_five_percent():
    joint = CanonicalRoutingAudit(256, "joint")
    control = CanonicalRoutingAudit(256, "capacity_control")
    difference = abs(joint.head_parameter_count - control.head_parameter_count) / joint.head_parameter_count
    assert difference <= 0.05


def test_c0_loader_freezes_backbone_and_classifier():
    backbone = torch.nn.Linear(12, 12)
    classifier = torch.nn.Linear(12, 3)
    common.freeze_c0(backbone, classifier)
    assert not backbone.training and not classifier.training
    assert all(not p.requires_grad for p in (*backbone.parameters(), *classifier.parameters()))


def test_training_objective_uses_train_rows_only():
    logits = torch.randn(5, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1])
    train = torch.tensor([0, 1, 2])
    loss = train_only_cross_entropy(logits, labels, train)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[3:]) == 0


def test_allowed_split_never_reads_test_indices():
    class SplitData:
        train_idx = torch.tensor([0, 2])
        val_idx = torch.tensor([1])
        y = torch.tensor([0, 1, 0])

        @property
        def test_idx(self):
            raise AssertionError("Test indices must not be accessed")

    allowed, y, train, val = common._allowed_split(SplitData(), torch.device("cpu"))
    assert allowed.tolist() == [0, 2, 1]
    assert y.tolist() == [0, 0, 1]
    assert train.tolist() == [0, 1]
    assert val.tolist() == [2]
