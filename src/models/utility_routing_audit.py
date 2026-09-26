from __future__ import annotations

import torch
from torch import nn


def uniform_state_mix(states: list[torch.Tensor]) -> torch.Tensor:
    """C0's four-state average, evaluated through its equivalent response form.

    Algebraically this is (S0+S1+S2+S3)/4. Using C0's fixed telescoping
    coefficients preserves its Float32 operation order and exact checkpoint
    reproduction while remaining the same positive uniform state mixture.
    """
    if len(states) != 4:
        raise ValueError("utility routing requires exactly S0, S1, S2, S3")
    z = states[0]
    for coefficient, low, high in zip((0.75, 0.50, 0.25), states[:-1], states[1:]):
        z = z + coefficient * (high - low)
    return z


def action_bank(states: list[torch.Tensor]) -> torch.Tensor:
    """Return A0..A3 and AU embeddings as [N, 5, H]."""
    if len(states) != 4:
        raise ValueError("utility routing requires exactly S0, S1, S2, S3")
    uniform = uniform_state_mix(states)
    return torch.stack([*states, uniform], dim=1)


class _RelationEvidence(nn.Module):
    """E_prop edge context. Tokens affect router evidence only."""

    CHUNK = 131072

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.reduce = nn.Linear(hidden_dim, 32)
        self.edge_encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.pool = nn.Linear(65, 128)
        self.gate = nn.Linear(128, 128)

    def pool_context(
        self, h0: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.numel() and bool((edge_index[0] == edge_index[1]).any()):
            raise ValueError("E_prop relation context must not include self-loops")
        source, target = edge_index.long()
        u = self.reduce(h0)
        pieces = []
        for start in range(0, source.numel(), self.CHUNK):
            src = source[start : start + self.CHUNK]
            dst = target[start : start + self.CHUNK]
            ui, uj = u[dst], u[src]
            pieces.append(self.edge_encoder(torch.cat([ui, uj, ui - uj, ui * uj], dim=-1)))
        tokens = (
            torch.cat(pieces, dim=0)
            if pieces
            else u.new_zeros((0, 32))
        )
        n = h0.size(0)
        total = h0.new_zeros((n, 32))
        square_total = h0.new_zeros((n, 32))
        degree = h0.new_zeros((n, 1))
        if target.numel():
            total.index_add_(0, target, tokens)
            square_total.index_add_(0, target, tokens.square())
            degree.index_add_(0, target, h0.new_ones((target.numel(), 1)))
        mean = total / degree.clamp_min(1.0)
        variance = (square_total / degree.clamp_min(1.0) - mean.square()).clamp_min(0.0)
        pooled = torch.cat([mean, variance, torch.log1p(degree)], dim=-1)
        return pooled

    def fuse_context(
        self,
        flat_context: torch.Tensor,
        pooled: torch.Tensor,
        allowed_idx: torch.Tensor,
    ) -> torch.Tensor:
        relation = self.pool(pooled[allowed_idx])
        return flat_context + torch.sigmoid(self.gate(flat_context)) * relation


class _Branch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        mode: str,
        capacity_hidden: int | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.state_proj = nn.ModuleList(nn.Linear(hidden_dim, 32) for _ in range(4))
        self.flat = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        if mode == "decision_response":
            self.delta_logits = nn.Linear(num_classes, 32)
            self.scalar_encoder = nn.Sequential(
                nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8)
            )
            self.decision_context = nn.Sequential(nn.Linear(200, 128), nn.ReLU())
            self.capacity = None
            self.relation = None
        elif mode == "decision_capacity":
            self.delta_logits = self.scalar_encoder = self.decision_context = None
            self.capacity = nn.Sequential(
                nn.Linear(128, 101), nn.ReLU(), nn.Linear(101, 128)
            )
            self.relation = None
        elif mode == "relation":
            self.relation = _RelationEvidence(hidden_dim)
            self.capacity = None
            self.delta_logits = self.scalar_encoder = self.decision_context = None
        elif mode == "relation_capacity":
            self.relation = None
            self.capacity = nn.Sequential(
                nn.Linear(128, 169), nn.ReLU(), nn.Linear(169, 128)
            )
            self.delta_logits = self.scalar_encoder = self.decision_context = None
        elif mode == "flat":
            self.relation = None
            self.capacity = None
            self.delta_logits = self.scalar_encoder = self.decision_context = None
        else:
            raise ValueError(f"unknown router evidence mode: {mode}")
        self.router = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def encode_flat(self, states: list[torch.Tensor]) -> torch.Tensor:
        return self.flat(torch.cat([
            torch.relu(proj(state)) for proj, state in zip(self.state_proj, states)
        ], dim=-1))

    def forward(
        self,
        states: list[torch.Tensor],
        *,
        decision_tokens: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        h0_all: torch.Tensor | None = None,
        allowed_idx: torch.Tensor | None = None,
        relation_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        context = self.encode_flat(states)
        pooled_relation = None
        if self.mode == "decision_response":
            if decision_tokens is None:
                raise ValueError("decision-response evidence is required")
            delta = self.delta_logits(decision_tokens[..., :-3])
            scalars = self.scalar_encoder(decision_tokens[..., -3:])
            decision = self.decision_context(
                torch.cat([delta, scalars], dim=-1).flatten(start_dim=1)
            )
            context = context + decision
        elif self.mode == "decision_capacity":
            context = context + self.capacity(context)
        elif self.mode == "relation":
            if edge_index is None or h0_all is None or allowed_idx is None:
                raise ValueError("E_prop and full-graph H0 are required for relation evidence")
            pooled_relation = self.relation.pool_context(h0_all, edge_index)
            if relation_override is not None:
                pooled_relation = pooled_relation.clone()
                pooled_relation[allowed_idx] = relation_override
            context = self.relation.fuse_context(context, pooled_relation, allowed_idx)
        elif self.mode == "relation_capacity":
            context = context + self.capacity(context)
        logits = self.router(context)
        return context, torch.softmax(logits, dim=-1), pooled_relation


class UtilityRouter(nn.Module):
    """Five-action router; all C0 encoders/fusion/classifier live outside it."""

    MODES = {"flat", "decision_response", "decision_capacity", "relation", "relation_capacity"}

    def __init__(self, hidden_dim: int, num_classes: int, mode: str = "flat"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported mode {mode}")
        self.mode = mode
        self.text = _Branch(hidden_dim, num_classes, mode)
        self.visual = _Branch(hidden_dim, num_classes, mode)

    def forward(
        self,
        states_text: list[torch.Tensor],
        states_visual: list[torch.Tensor],
        *,
        decision_text: torch.Tensor | None = None,
        decision_visual: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        h0_text_all: torch.Tensor | None = None,
        h0_visual_all: torch.Tensor | None = None,
        allowed_idx: torch.Tensor | None = None,
        relation_override_text: torch.Tensor | None = None,
        relation_override_visual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ctx_t, p_t, pooled_t = self.text(
            states_text, decision_tokens=decision_text, edge_index=edge_index,
            h0_all=h0_text_all, allowed_idx=allowed_idx,
            relation_override=relation_override_text,
        )
        ctx_v, p_v, pooled_v = self.visual(
            states_visual, decision_tokens=decision_visual, edge_index=edge_index,
            h0_all=h0_visual_all, allowed_idx=allowed_idx,
            relation_override=relation_override_visual,
        )
        return {
            "context_text": ctx_t,
            "context_visual": ctx_v,
            "prob_text": p_t,
            "prob_visual": p_v,
            "pooled_relation_text": pooled_t,
            "pooled_relation_visual": pooled_v,
        }


def expected_mixture(actions: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 3 or probabilities.shape != actions.shape[:2]:
        raise ValueError("expected actions [N,5,H] and probabilities [N,5]")
    return torch.sum(actions * probabilities.unsqueeze(-1), dim=1)


def safe_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Confidence-weight router mass by its entropy; fallback is AU action."""
    if probabilities.size(-1) != 5:
        raise ValueError("safe routing expects a five-action simplex")
    p = probabilities.clamp_min(1e-12)
    confidence = 1.0 - (-(p * p.log()).sum(dim=-1) / torch.log(p.new_tensor(5.0)))
    fallback = torch.zeros_like(probabilities)
    fallback[..., 4] = 1.0
    return (1.0 - confidence.unsqueeze(-1)) * fallback + confidence.unsqueeze(-1) * probabilities
