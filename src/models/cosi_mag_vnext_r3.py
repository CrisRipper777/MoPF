from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from .cosi_filter import chebyshev_bank, sparse_operator_matmul
from .cosi_mag_vnext import Model as VNextModel


def anchored_multi_hop_bank(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    max_order: int,
    anchor_alpha: float,
) -> torch.Tensor:
    """Return B0=x; Bk=alpha*x+(1-alpha)*P B(k-1)."""
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    if not 0.0 <= anchor_alpha <= 1.0:
        raise ValueError(f"anchor_alpha must be in [0, 1], got {anchor_alpha}")
    states = [x]
    current = x
    for _ in range(max_order):
        current = anchor_alpha * x + (1.0 - anchor_alpha) * sparse_operator_matmul(
            edge_index, edge_weight, current
        )
        states.append(current)
    return torch.stack(states, dim=1)


def _uniform_operator(
    physical_edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index, edge_weight = gcn_norm(
        physical_edge_index,
        edge_weight=torch.ones(
            physical_edge_index.size(1), device=physical_edge_index.device, dtype=dtype
        ),
        num_nodes=num_nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=dtype,
    )
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=dtype)
    return edge_index, edge_weight


class NoContextInteraction(nn.Module):
    """RCMI with the node-specific relation context path removed."""

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int,
        max_order: int,
        attention_dropout: float,
        attention_gate_init: float,
        semantic_gate_init: float,
        eta_initial: tuple[float, ...],
        refine_dropout: float,
    ):
        super().__init__()
        if len(eta_initial) != max_order:
            raise ValueError("eta_initial length must equal max_order")
        if not 0.0 < attention_gate_init < 1.0 or not 0.0 < semantic_gate_init < 1.0:
            raise ValueError("gate initializers must be in (0, 1)")
        if any(abs(value) >= 0.5 for value in eta_initial):
            raise ValueError("eta initializers must be strictly between -0.5 and 0.5")
        self.hidden_dim = int(hidden_dim)
        self.max_order = int(max_order)
        self.num_orders = self.max_order + 1
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.attention_scale = float(attention_dim) ** -0.5
        self.order_embedding_text = nn.Parameter(torch.empty(self.num_orders, hidden_dim))
        self.order_embedding_visual = nn.Parameter(torch.empty(self.num_orders, hidden_dim))
        nn.init.normal_(self.order_embedding_text, mean=0.0, std=0.02)
        nn.init.normal_(self.order_embedding_visual, mean=0.0, std=0.02)
        self.shared_order_prior = nn.Parameter(
            torch.tensor([math.atanh(2.0 * value) for value in eta_initial])
        )
        self.modality_order_bias = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(self.max_order)) for m in ("text", "visual")}
        )
        self.conditional_score = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 4, 1),
                )
                for m in ("text", "visual")
            }
        )
        for scorer in self.conditional_score.values():
            nn.init.zeros_(scorer[-1].weight)
            nn.init.zeros_(scorer[-1].bias)
        gate_logit = math.log(attention_gate_init / (1.0 - attention_gate_init))
        self.attention_gate_logit = nn.ParameterDict(
            {m: nn.Parameter(torch.tensor(gate_logit)) for m in ("text", "visual")}
        )
        semantic_logit = math.log(semantic_gate_init / (1.0 - semantic_gate_init))
        self.semantic_gate = nn.ModuleDict(
            {m: nn.Linear(hidden_dim, hidden_dim) for m in ("text", "visual")}
        )
        for gate in self.semantic_gate.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, semantic_logit)
        self.refine = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(refine_dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for m in ("text", "visual")
            }
        )
        self.refine_norm = nn.ModuleDict(
            {m: nn.LayerNorm(hidden_dim) for m in ("text", "visual")}
        )

    def forward_modality(self, h0, bank, modality: str):
        embedding = getattr(self, f"order_embedding_{modality}")
        token = self.token_norm(bank + embedding.unsqueeze(0))
        query, key = self.query(token), self.key(token)
        value = self.value(bank)
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.attention_scale
        attention = torch.softmax(logits, dim=-1)
        attention_output = torch.matmul(self.attention_dropout(attention), value)
        attention_gate = torch.sigmoid(self.attention_gate_logit[modality])
        interacted = bank + attention_gate * attention_output
        differences = interacted[:, 1:, :] - interacted[:, :1, :]
        conditional = self.conditional_score[modality](differences).squeeze(-1)
        eta = 0.5 * torch.tanh(
            self.shared_order_prior.unsqueeze(0)
            + self.modality_order_bias[modality].unsqueeze(0)
            + conditional
        )
        structural_residual = (eta.unsqueeze(-1) * differences).sum(dim=1)
        semantic_gate = torch.sigmoid(self.semantic_gate[modality](h0))
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
        }


class R3Model(VNextModel):
    """Controlled uniform-propagation variants based on the frozen VNext."""

    variant = "u"

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info)
        self.anchor_alpha = float(cfg.model.get("anchor_alpha", 0.1))
        if self.variant == "uq":
            self.crc = None
            self.interaction = NoContextInteraction(
                hidden_dim=self.hidden_dim,
                attention_dim=int(cfg.model.get("attention_dim", 64)),
                max_order=self.max_order,
                attention_dropout=float(cfg.model.get("attention_dropout", 0.0)),
                attention_gate_init=float(cfg.model.get("attention_gate_init", 0.1)),
                semantic_gate_init=float(cfg.model.get("semantic_gate_init", 0.1)),
                eta_initial=tuple(float(v) for v in cfg.model.get("eta_initial", [0.2, 0.0, 0.0][: self.max_order])),
                refine_dropout=float(cfg.model.get("dropout", 0.2)),
            )

    def _encode_components(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        x_text, x_visual = self._split_features(x)
        h0_text, h0_visual = self.text_proj(x_text), self.visual_proj(x_visual)
        edge_index = self._edge_index(edge_index, x.device)
        if self.variant == "uq":
            physical = edge_index[:, edge_index[0] != edge_index[1]].contiguous()
            relation = {"physical_edge_index": physical, "q": None}
        else:
            relation = self.crc(h0_text, h0_visual, edge_index)
            physical = relation["physical_edge_index"]
        uniform_index, uniform_weight = _uniform_operator(
            physical, int(x.size(0)), h0_text.dtype
        )
        if self.variant == "ua":
            bank_fn = lambda h: anchored_multi_hop_bank(
                h, uniform_index, uniform_weight, self.max_order, self.anchor_alpha
            )
        else:
            bank_fn = lambda h: chebyshev_bank(
                h, uniform_index, uniform_weight, self.max_order
            )
        bank_text, bank_visual = bank_fn(h0_text), bank_fn(h0_visual)
        if self.variant == "uq":
            text = self.interaction.forward_modality(h0_text, bank_text, "text")
            visual = self.interaction.forward_modality(h0_visual, bank_visual, "visual")
        else:
            text = self.interaction.forward_modality(h0_text, bank_text, relation["q"], "text")
            visual = self.interaction.forward_modality(h0_visual, bank_visual, relation["q"], "visual")
        fused = torch.cat((text["z"], visual["z"]), dim=-1)
        z = self.output_norm(self.fusion_skip(fused) + self.fusion_mlp(fused))
        return {
            "h0_text": h0_text,
            "h0_visual": h0_visual,
            "relation": relation,
            "uniform_edge_index": uniform_index,
            "uniform_edge_weight": uniform_weight,
            "bank_text": bank_text,
            "bank_visual": bank_visual,
            "text": text,
            "visual": visual,
            "z": z,
        }


__all__ = ["R3Model", "NoContextInteraction", "anchored_multi_hop_bank"]
