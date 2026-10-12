from __future__ import annotations

import math

import torch
import torch.nn as nn


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}")
    return math.log(probability / (1.0 - probability))


class RelationConditionedMultiOrderInteraction(nn.Module):
    """Cross-order attention followed by signed structural residual mixing."""

    def __init__(
        self,
        hidden_dim: int = 256,
        context_dim: int = 32,
        attention_dim: int = 64,
        max_order: int = 3,
        attention_dropout: float = 0.0,
        attention_gate_init: float = 0.1,
        semantic_gate_init: float = 0.1,
        eta_initial: tuple[float, ...] | list[float] = (0.2, 0.0, 0.0),
        refine_dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.max_order = int(max_order)
        self.num_orders = self.max_order + 1
        if self.max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {max_order}")
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")
        if len(eta_initial) != self.max_order:
            raise ValueError(
                f"eta_initial must have max_order={self.max_order} values, got {len(eta_initial)}"
            )
        if any(abs(float(value)) >= 0.5 for value in eta_initial):
            raise ValueError("each eta_initial value must be strictly between -0.5 and 0.5")

        # Q/K and the value transform are shared across modalities and orders.
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.attention_scale = float(attention_dim) ** -0.5

        self.order_embedding_text = nn.Parameter(
            torch.empty(self.num_orders, hidden_dim)
        )
        self.order_embedding_visual = nn.Parameter(
            torch.empty(self.num_orders, hidden_dim)
        )
        nn.init.normal_(self.order_embedding_text, mean=0.0, std=0.02)
        nn.init.normal_(self.order_embedding_visual, mean=0.0, std=0.02)

        self.condition_order_bias = nn.ModuleDict(
            {
                "text": nn.Linear(context_dim, self.num_orders),
                "visual": nn.Linear(context_dim, self.num_orders),
            }
        )
        initial_prior = torch.tensor(
            [math.atanh(2.0 * float(value)) for value in eta_initial],
            dtype=torch.float32,
        )
        self.shared_order_prior = nn.Parameter(initial_prior)
        self.modality_order_bias = nn.ParameterDict(
            {
                "text": nn.Parameter(torch.zeros(self.max_order)),
                "visual": nn.Parameter(torch.zeros(self.max_order)),
            }
        )
        self.conditional_score = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(hidden_dim + context_dim, hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 4, 1),
                )
                for modality in ("text", "visual")
            }
        )
        for scorer in self.conditional_score.values():
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)

        self.attention_gate_logit = nn.ParameterDict(
            {
                "text": nn.Parameter(torch.tensor(_logit(attention_gate_init))),
                "visual": nn.Parameter(torch.tensor(_logit(attention_gate_init))),
            }
        )
        self.semantic_gate = nn.ModuleDict(
            {
                "text": nn.Linear(hidden_dim + context_dim, hidden_dim),
                "visual": nn.Linear(hidden_dim + context_dim, hidden_dim),
            }
        )
        for gate in self.semantic_gate.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, _logit(semantic_gate_init))

        self.refine = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(refine_dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for modality in ("text", "visual")
            }
        )
        self.refine_norm = nn.ModuleDict(
            {"text": nn.LayerNorm(hidden_dim), "visual": nn.LayerNorm(hidden_dim)}
        )

    def forward_modality(
        self,
        h0: torch.Tensor,
        bank: torch.Tensor,
        q: torch.Tensor,
        modality: str,
    ) -> dict[str, torch.Tensor]:
        if modality not in {"text", "visual"}:
            raise ValueError(f"unknown modality {modality!r}")
        if bank.size(1) != self.num_orders:
            raise ValueError(
                f"expected {self.num_orders} order states, got {bank.size(1)}"
            )

        order_embedding = getattr(self, f"order_embedding_{modality}")
        token_input = self.token_norm(bank + order_embedding.unsqueeze(0))
        query = self.query(token_input)
        key = self.key(token_input)
        # Values use the original B_k states; order embeddings never enter V.
        value = self.value(bank)
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.attention_scale
        relation_bias = self.condition_order_bias[modality](q)
        logits = logits + relation_bias.unsqueeze(1)
        attention = torch.softmax(logits, dim=-1)
        attention_output = torch.matmul(self.attention_dropout(attention), value)
        attention_gate = torch.sigmoid(self.attention_gate_logit[modality])
        interacted = bank + attention_gate * attention_output

        differences = interacted[:, 1:, :] - interacted[:, :1, :]
        condition = q.unsqueeze(1).expand(-1, self.max_order, -1)
        conditioned = torch.cat((differences, condition), dim=-1)
        conditional = self.conditional_score[modality](conditioned).squeeze(-1)
        eta = 0.5 * torch.tanh(
            self.shared_order_prior.unsqueeze(0)
            + self.modality_order_bias[modality].unsqueeze(0)
            + conditional
        )
        structural_residual = (eta.unsqueeze(-1) * differences).sum(dim=1)

        semantic_gate = torch.sigmoid(
            self.semantic_gate[modality](torch.cat((h0, q), dim=-1))
        )
        y = h0 + semantic_gate * structural_residual
        z = self.refine_norm[modality](y + self.refine[modality](y))
        return {
            "z": z,
            "interacted_bank": interacted,
            "differences": differences,
            "eta": eta,
            "structural_residual": structural_residual,
            "semantic_gate": semantic_gate,
            "attention": attention,
            "attention_gate": attention_gate,
            "relation_order_bias": relation_bias,
        }


__all__ = ["RelationConditionedMultiOrderInteraction"]
