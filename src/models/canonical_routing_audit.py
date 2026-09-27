from __future__ import annotations

import torch
from torch import nn

from src.models.utility_routing_audit import uniform_state_mix


def e5_probabilities_to_alpha(probabilities: torch.Tensor) -> torch.Tensor:
    """Collapse E5's redundant AU action into the canonical four-state simplex."""
    if probabilities.ndim < 1 or probabilities.shape[-1] != 5:
        raise ValueError("E5 probabilities must end in five actions")
    alpha = probabilities[..., :4] + probabilities[..., 4:5] * 0.25
    return alpha / alpha.sum(dim=-1, keepdim=True)


def canonical_mix(states: list[torch.Tensor], alpha: torch.Tensor) -> torch.Tensor:
    """Execute a four-state mixture, retaining C0's operation order at alpha_U."""
    if len(states) != 4:
        raise ValueError("canonical routing requires exactly S0, S1, S2, S3")
    if alpha.shape[-1] != 4:
        raise ValueError("canonical alpha must end in four orders")
    if alpha.ndim == 1:
        alpha = alpha.unsqueeze(0).expand(states[0].size(0), -1)
    if alpha.shape[0] != states[0].shape[0]:
        raise ValueError("alpha and state node dimensions must match")
    if bool(torch.all(alpha == 0.25)):
        uniform = uniform_state_mix(states)
        if alpha.requires_grad:
            direct = states[0] * alpha[:, 0:1]
            for order in range(1, 4):
                direct = direct + states[order] * alpha[:, order : order + 1]
            # Preserve C0’s exact forward value while keeping the router
            # gradient alive at its zero-initialized uniform starting point.
            return uniform + (direct - direct.detach())
        return uniform
    result = states[0] * alpha[:, 0:1]
    for order in range(1, 4):
        result = result + states[order] * alpha[:, order : order + 1]
    return result


class _TrajectoryContext(nn.Module):
    """E5-sized state-wise evidence encoder for one modality."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.state_projection = nn.ModuleList(
            nn.Linear(hidden_dim, 32) for _ in range(4)
        )
        self.flat = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        if len(states) != 4:
            raise ValueError("trajectory encoder requires four states")
        tokens = [
            torch.relu(projection(state))
            for projection, state in zip(self.state_projection, states)
        ]
        return self.flat(torch.cat(tokens, dim=-1))


class _AlphaHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.layers(context), dim=-1)


class CanonicalRoutingAudit(nn.Module):
    """Independent, joint, or capacity-control four-order router.

    ``independent`` is D0. ``joint`` and ``capacity_control`` are the two D1
    comparators; their modality trajectory encoders have identical capacity.
    """

    MODES = {"independent", "joint", "capacity_control"}

    def __init__(self, hidden_dim: int = 256, mode: str = "independent"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported canonical router mode: {mode}")
        self.mode = mode
        self.text_context = _TrajectoryContext(hidden_dim)
        self.visual_context = _TrajectoryContext(hidden_dim)
        if mode == "independent":
            self.text_head = _AlphaHead(128, 64)
            self.visual_head = _AlphaHead(128, 64)
        elif mode == "joint":
            self.text_head = _AlphaHead(256, 64)
            self.visual_head = _AlphaHead(256, 64)
        else:
            # 128 -> 125 -> 4 is within 0.1% of the joint head's parameter
            # count (256 -> 64 -> 4), while reading only own-modality context.
            self.text_head = _AlphaHead(128, 125)
            self.visual_head = _AlphaHead(128, 125)

    @property
    def head_parameter_count(self) -> int:
        return sum(p.numel() for p in (*self.text_head.parameters(), *self.visual_head.parameters()))

    def forward(
        self,
        states_text: list[torch.Tensor],
        states_visual: list[torch.Tensor],
        *,
        companion_text: torch.Tensor | None = None,
        companion_visual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ctx_t = self.text_context(states_text)
        ctx_v = self.visual_context(states_visual)
        if self.mode == "joint":
            input_t = torch.cat(
                [ctx_t, ctx_v if companion_visual is None else companion_visual], dim=-1
            )
            input_v = torch.cat(
                [ctx_v, ctx_t if companion_text is None else companion_text], dim=-1
            )
        else:
            if companion_text is not None or companion_visual is not None:
                raise ValueError("companion contexts are only valid for the joint router")
            input_t, input_v = ctx_t, ctx_v
        return {
            "context_text": ctx_t,
            "context_visual": ctx_v,
            "alpha_text": self.text_head(input_t),
            "alpha_visual": self.visual_head(input_v),
        }
