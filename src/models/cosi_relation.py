from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class CrossModalRelationContext(nn.Module):
    """Score observed physical edges and derive node-local relation context."""

    def __init__(
        self,
        relation_hidden_dim: int = 32,
        context_dim: int = 32,
        weight_scale: float = 0.3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.weight_scale = float(weight_scale)
        self.eps = float(eps)
        if not 0.0 <= self.weight_scale < 1.0:
            raise ValueError("weight_scale must satisfy 0 <= weight_scale < 1")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if relation_hidden_dim <= 0 or context_dim <= 0:
            raise ValueError("relation_hidden_dim and context_dim must be positive")
        self.relation_scorer = nn.Sequential(
            nn.Linear(4, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 2),
        )
        nn.init.zeros_(self.relation_scorer[-1].weight)
        nn.init.zeros_(self.relation_scorer[-1].bias)
        self.context_mlp = nn.Sequential(
            nn.Linear(8, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
        )
        self.context_dim = int(context_dim)

    @staticmethod
    def _without_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
        return edge_index[:, edge_index[0] != edge_index[1]].contiguous()

    def _edge_evidence(
        self,
        h_text: torch.Tensor,
        h_visual: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        cosine_text = F.cosine_similarity(
            h_text[src], h_text[dst], dim=-1, eps=self.eps
        )
        cosine_visual = F.cosine_similarity(
            h_visual[src], h_visual[dst], dim=-1, eps=self.eps
        )
        difference = (cosine_text - cosine_visual).abs()
        evidence = torch.stack(
            (cosine_text, cosine_visual, difference, cosine_text * cosine_visual),
            dim=-1,
        )
        scores = self.relation_scorer(evidence)
        a, b = scores.unbind(dim=-1)
        weight_text = 1.0 + self.weight_scale * torch.tanh(a + b)
        weight_visual = 1.0 + self.weight_scale * torch.tanh(a - b)
        return cosine_text, cosine_visual, weight_text, weight_visual

    @staticmethod
    def _node_statistics(
        edge_index: torch.Tensor,
        cosine_text: torch.Tensor,
        cosine_visual: torch.Tensor,
        weight_text: torch.Tensor,
        weight_visual: torch.Tensor,
        num_nodes: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the 8-value descriptor and degree, excluding self-loops."""
        device = weight_text.device
        dtype = weight_text.dtype
        src = edge_index[0]
        count = torch.zeros(num_nodes, device=device, dtype=dtype)
        count.index_add_(0, src, torch.ones_like(weight_text))
        safe_count = count.clamp_min(1.0)

        values = torch.stack(
            (
                cosine_text,
                cosine_visual,
                (cosine_text - cosine_visual).abs(),
                weight_text,
                weight_visual,
            ),
            dim=-1,
        )
        sums = torch.zeros((num_nodes, values.size(-1)), device=device, dtype=dtype)
        if src.numel():
            sums.index_add_(0, src, values)
        means = sums / safe_count.unsqueeze(-1)

        # Center before squaring; this avoids cancellation in E[x^2] - E[x]^2.
        centered = values - means[src] if src.numel() else values
        squared_sums = torch.zeros_like(sums)
        if src.numel():
            squared_sums.index_add_(0, src, centered.square())
        variance = (squared_sums / safe_count.unsqueeze(-1)).clamp_min(0.0)
        # clamp_min creates a zero derivative for a perfectly uniform
        # neighborhood; sqrt(0) would otherwise yield an infinite derivative.
        stds = torch.sqrt(variance.clamp_min(float(eps) ** 2))

        # Empty neighborhoods have zero means/stds and log1p(degree)=0.
        means = torch.where(count.unsqueeze(-1) > 0, means, torch.zeros_like(means))
        stds = torch.where(count.unsqueeze(-1) > 0, stds, torch.zeros_like(stds))
        descriptor = torch.cat(
            (means[:, :3], means[:, 3:5], stds[:, 3:5], torch.log1p(count).unsqueeze(-1)),
            dim=-1,
        )
        return descriptor, count

    @staticmethod
    def _normalized_operator(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_index, normalized_weight = gcn_norm(
            edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            improved=False,
            add_self_loops=True,
            flow="source_to_target",
            dtype=dtype,
        )
        if normalized_weight is None:
            normalized_weight = torch.ones(
                normalized_index.size(1), device=normalized_index.device, dtype=dtype
            )
        return normalized_index, normalized_weight

    def forward(
        self,
        h_text: torch.Tensor,
        h_visual: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_nodes = int(h_text.size(0))
        physical_edge_index = self._without_self_loops(edge_index)
        cosine_text, cosine_visual, weight_text, weight_visual = self._edge_evidence(
            h_text, h_visual, physical_edge_index
        )
        descriptor, degree = self._node_statistics(
            physical_edge_index,
            cosine_text,
            cosine_visual,
            weight_text,
            weight_visual,
            num_nodes,
            self.eps,
        )
        q = self.context_mlp(descriptor)
        norm_text_index, norm_text_weight = self._normalized_operator(
            physical_edge_index, weight_text, num_nodes, h_text.dtype
        )
        norm_visual_index, norm_visual_weight = self._normalized_operator(
            physical_edge_index, weight_visual, num_nodes, h_visual.dtype
        )
        return {
            "physical_edge_index": physical_edge_index,
            "cosine_text": cosine_text,
            "cosine_visual": cosine_visual,
            "weight_text": weight_text,
            "weight_visual": weight_visual,
            "descriptor": descriptor,
            "degree": degree,
            "q": q,
            "P_text_edge_index": norm_text_index,
            "P_text_edge_weight": norm_text_weight,
            "P_visual_edge_index": norm_visual_index,
            "P_visual_edge_weight": norm_visual_weight,
        }

    @torch.no_grad()
    def diagnostic_summary(self, relation: dict[str, torch.Tensor]) -> dict[str, float | int]:
        """Small detached summary suitable for reports and run manifests."""
        result: dict[str, float | int] = {
            "physical_directed_edges": int(relation["physical_edge_index"].size(1)),
            "num_nodes": int(relation["q"].size(0)),
        }
        for name in (
            "cosine_text",
            "cosine_visual",
            "weight_text",
            "weight_visual",
            "descriptor",
        ):
            value = relation[name].float()
            result[f"{name}_mean"] = float(value.mean().item()) if value.numel() else 0.0
            result[f"{name}_std"] = float(value.std(unbiased=False).item()) if value.numel() else 0.0
            if name.startswith("weight_") and value.numel():
                result[f"{name}_min"] = float(value.min().item())
                result[f"{name}_max"] = float(value.max().item())
        if relation["cosine_text"].numel():
            result["mean_abs_cosine_gap"] = float(
                (relation["cosine_text"] - relation["cosine_visual"]).abs().mean().item()
            )
        else:
            result["mean_abs_cosine_gap"] = 0.0
        return result


__all__ = ["CrossModalRelationContext"]
