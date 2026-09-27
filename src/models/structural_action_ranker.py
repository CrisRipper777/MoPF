from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from src.models.utility_routing_audit import uniform_state_mix

ACTION_NAMES = ("Uniform", "S0", "S1", "S2", "S3")
ACTION_COUNT = 5


def hard_action_bank(states: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return five hard execution embeddings [Uniform, S0, S1, S2, S3]."""
    if len(states) != 4:
        raise ValueError("hard structural actions require exactly S0, S1, S2, S3")
    return torch.stack([uniform_state_mix(list(states)), *states], dim=1)


def execute_hard_action(states: Sequence[torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    """Execute one selected action per node. No expected-action mixture is used."""
    bank = hard_action_bank(states)
    if action.ndim == 0:
        action = action.expand(bank.shape[0])
    if action.shape != (bank.shape[0],):
        raise ValueError("action ids must have shape [nodes]")
    if action.dtype != torch.long:
        action = action.long()
    return bank[torch.arange(bank.shape[0], device=bank.device), action]


def select_hard_action(scores: torch.Tensor) -> torch.Tensor:
    """Argmax five scores; exact ties containing Uniform resolve to Uniform."""
    if scores.ndim < 1 or scores.shape[-1] != ACTION_COUNT:
        raise ValueError("action scores must end in five actions")
    maximum = scores.max(dim=-1, keepdim=True).values
    any_tie = (scores == maximum).sum(dim=-1, keepdim=True) > 1
    non_uniform_best = scores[..., 1:].argmax(dim=-1, keepdim=True) + 1
    return torch.where(any_tie, torch.zeros_like(non_uniform_best),
                       torch.where(scores[..., 0:1] == maximum, torch.zeros_like(non_uniform_best), non_uniform_best)).squeeze(-1)


def decision_signature(
    prediction_uniform: torch.Tensor,
    prediction_action: torch.Tensor,
) -> torch.Tensor:
    """Build the fixed 11-D label-free response signature for one action."""
    if prediction_uniform.shape != prediction_action.shape or prediction_uniform.ndim < 2:
        raise ValueError("paired predictions must have identical [..., classes] shapes")
    pu = prediction_uniform.clamp_min(1e-12)
    pa = prediction_action.clamp_min(1e-12)
    pu = pu / pu.sum(dim=-1, keepdim=True)
    pa = pa / pa.sum(dim=-1, keepdim=True)
    logu, loga = pu.log(), pa.log()
    entropy_u = -(pu * logu).sum(dim=-1)
    entropy_a = -(pa * loga).sum(dim=-1)
    top_u = pu.topk(2, dim=-1).values
    top_a = pa.topk(2, dim=-1).values
    margin_u, margin_a = top_u[..., 0] - top_u[..., 1], top_a[..., 0] - top_a[..., 1]
    max_u, max_a = top_u[..., 0], top_a[..., 0]
    kl_action_uniform = (pa * (loga - logu)).sum(dim=-1)
    midpoint = 0.5 * (pu + pa)
    log_midpoint = midpoint.log()
    js = 0.5 * (
        (pu * (logu - log_midpoint)).sum(dim=-1)
        + (pa * (loga - log_midpoint)).sum(dim=-1)
    )
    l1 = (pa - pu).abs().sum(dim=-1)
    baseline_class = pu.argmax(dim=-1, keepdim=True)
    delta_baseline_probability = (
        pa.gather(-1, baseline_class).squeeze(-1)
        - pu.gather(-1, baseline_class).squeeze(-1)
    )
    top1_changed = (pa.argmax(dim=-1) != baseline_class.squeeze(-1)).to(pu.dtype)
    return torch.stack(
        [
            entropy_u, entropy_a, margin_u, margin_a, max_u, max_a,
            kl_action_uniform, js, l1, delta_baseline_probability, top1_changed,
        ],
        dim=-1,
    )


def pairwise_ranking_loss(
    scores: torch.Tensor,
    action_losses: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized counterfactual pairwise loss from per-node action CE only."""
    if scores.shape != action_losses.shape or scores.ndim != 2 or scores.shape[1] != ACTION_COUNT:
        raise ValueError("scores and action losses must both have shape [nodes, 5]")
    left, right = torch.triu_indices(ACTION_COUNT, ACTION_COUNT, offset=1, device=scores.device)
    delta = action_losses[:, right] - action_losses[:, left]
    magnitude = delta.abs()
    normalizer = magnitude.sum(dim=1, keepdim=True)
    weights = torch.where(normalizer > 0, magnitude / normalizer.clamp_min(1e-12), 0.0)
    target = delta.sign()
    score_difference = scores[:, left] - scores[:, right]
    node_loss = (weights * F.softplus(-target * score_difference)).sum(dim=1)
    if node_mask is not None:
        if node_mask.dtype == torch.bool:
            if node_mask.shape != (scores.shape[0],):
                raise ValueError("boolean node mask must have shape [nodes]")
            node_loss = node_loss[node_mask]
        else:
            node_loss = node_loss[node_mask.long()]
        if node_loss.numel() == 0:
            return scores.sum() * 0
    return node_loss.mean()


def pairwise_weights(action_losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pair indices and per-node normalized weights; degenerate nodes are zero."""
    if action_losses.ndim != 2 or action_losses.shape[1] != ACTION_COUNT:
        raise ValueError("action losses must have shape [nodes, 5]")
    left, right = torch.triu_indices(ACTION_COUNT, ACTION_COUNT, offset=1, device=action_losses.device)
    delta = action_losses[:, right] - action_losses[:, left]
    magnitude = delta.abs()
    total = magnitude.sum(dim=1, keepdim=True)
    weights = torch.where(total > 0, magnitude / total.clamp_min(1e-12), 0.0)
    return left, right, weights


def weighted_pairwise_accuracy(scores: torch.Tensor, action_losses: torch.Tensor) -> torch.Tensor:
    left, right, weights = pairwise_weights(action_losses)
    preference = (action_losses[:, right] - action_losses[:, left]).sign()
    predicted = (scores[:, left] - scores[:, right]).sign()
    correct = (preference == predicted).to(scores.dtype)
    per_node = (weights * correct).sum(dim=1)
    nondegenerate = weights.sum(dim=1) > 0
    if not nondegenerate.any():
        return scores.new_tensor(float("nan"))
    return per_node[nondegenerate].mean()


class _TrajectoryContext(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.state_projection = nn.ModuleList(nn.Linear(hidden_dim, 32) for _ in range(4))
        self.flat = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

    def forward(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(states) != 4:
            raise ValueError("trajectory encoder requires four states")
        projected = [
            torch.relu(layer(state))
            for layer, state in zip(self.state_projection, states)
        ]
        return self.flat(torch.cat(projected, dim=-1))


class _ModalityActionScorer(nn.Module):
    def __init__(self, hidden_dim: int, mode: str):
        super().__init__()
        self.context = _TrajectoryContext(hidden_dim)
        self.action_embedding = nn.Embedding(ACTION_COUNT, 16)
        self.mode = mode
        if mode == "r1_decision":
            self.decision_embedding = nn.Sequential(nn.Linear(11, 32), nn.ReLU())
            self.capacity_embedding = None
        elif mode == "r1_capacity_control":
            self.decision_embedding = None
            self.capacity_embedding = nn.Sequential(
                nn.Linear(144, 4), nn.ReLU(), nn.Linear(4, 32), nn.ReLU()
            )
        elif mode == "r0_trajectory":
            self.decision_embedding = None
            self.capacity_embedding = None
        else:
            raise ValueError(f"unsupported action ranker mode: {mode}")
        scorer_input_dim = 144 if mode == "r0_trajectory" else 176
        self.score_head = nn.Sequential(
            nn.Linear(scorer_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(
        self,
        states: Sequence[torch.Tensor],
        signatures: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context(states)
        n, device = context.shape[0], context.device
        ids = torch.arange(ACTION_COUNT, device=device)
        action_tokens = self.action_embedding(ids).unsqueeze(0).expand(n, -1, -1)
        context_tokens = context.unsqueeze(1).expand(-1, ACTION_COUNT, -1)
        trajectory_action = torch.cat([context_tokens, action_tokens], dim=-1)
        if self.mode == "r1_decision":
            if signatures is None or signatures.shape != (n, ACTION_COUNT, 11):
                raise ValueError("R1 requires decision signatures with shape [nodes, 5, 11]")
            decision_tokens = self.decision_embedding(signatures)
            score_input = torch.cat([trajectory_action, decision_tokens], dim=-1)
        elif self.mode == "r1_capacity_control":
            capacity_tokens = self.capacity_embedding(trajectory_action)
            score_input = torch.cat([trajectory_action, capacity_tokens], dim=-1)
        else:
            score_input = trajectory_action
        scores = self.score_head(score_input).squeeze(-1)
        confidence = scores.topk(2, dim=-1).values.diff(dim=-1).squeeze(-1).abs()
        route_vs_uniform = scores[:, 1:].max(dim=-1).values - scores[:, 0]
        return scores, torch.stack([confidence, route_vs_uniform], dim=-1)


class StructuralActionRanker(nn.Module):
    """Independent per-modality hard structural action scorer."""

    MODES = {"r0_trajectory", "r1_capacity_control", "r1_decision"}

    def __init__(self, hidden_dim: int = 256, mode: str = "r0_trajectory"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported ranker mode: {mode}")
        self.mode = mode
        self.text = _ModalityActionScorer(hidden_dim, mode)
        self.visual = _ModalityActionScorer(hidden_dim, mode)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        states_text: Sequence[torch.Tensor],
        states_visual: Sequence[torch.Tensor],
        signatures_text: torch.Tensor | None = None,
        signatures_visual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scores_text, confidence_text = self.text(states_text, signatures_text)
        scores_visual, confidence_visual = self.visual(states_visual, signatures_visual)
        return {
            "scores_text": scores_text,
            "scores_visual": scores_visual,
            "confidence_text": confidence_text,
            "confidence_visual": confidence_visual,
        }
