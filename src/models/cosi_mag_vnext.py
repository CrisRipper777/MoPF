from __future__ import annotations

import torch
import torch.nn as nn

from .common import make_norm
from .cosi_filter import chebyshev_bank
from .cosi_interaction import RelationConditionedMultiOrderInteraction
from .cosi_relation import CrossModalRelationContext


class ProjectionMLP(nn.Module):
    """Independent modality projection matching the existing CoSI projector."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        dropout: float,
        norm: str | None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_mlp(
    in_dim: int, hidden_dim: int, out_dim: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class Model(nn.Module):
    """CoSI-MAG VNext: CRC → MMF → RCMI → concat residual late fusion."""

    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("CoSI-MAG VNext requires non-empty text and visual features")
        if self.text_dim + self.visual_dim > input_dim:
            raise ValueError(
                f"text_dim+visual_dim={self.text_dim + self.visual_dim} exceeds input_dim={input_dim}"
            )

        model_cfg = cfg.model
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        dropout = float(model_cfg.get("dropout", 0.2))
        norm = model_cfg.get("norm", "layernorm")
        self.max_order = int(model_cfg.get("max_order", 3))
        if self.max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {self.max_order}")
        if not bool(model_cfg.get("diffusion_add_self_loops", True)):
            raise ValueError("CoSI-MAG VNext requires exactly one self-loop per node in P")
        fusion = str(model_cfg.get("fusion", "concat_residual_mlp")).strip().lower()
        if fusion != "concat_residual_mlp":
            raise ValueError("CoSI-MAG VNext fixes late fusion to concat_residual_mlp")
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim

        self.text_proj = ProjectionMLP(self.text_dim, hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, hidden_dim, dropout, norm)
        self.crc = CrossModalRelationContext(
            relation_hidden_dim=int(model_cfg.get("relation_hidden_dim", 32)),
            context_dim=int(model_cfg.get("relation_context_dim", 32)),
            weight_scale=float(model_cfg.get("relation_weight_scale", 0.3)),
            eps=float(model_cfg.get("eps", 1e-8)),
        )
        self.interaction = RelationConditionedMultiOrderInteraction(
            hidden_dim=hidden_dim,
            context_dim=int(model_cfg.get("relation_context_dim", 32)),
            attention_dim=int(model_cfg.get("attention_dim", 64)),
            max_order=self.max_order,
            attention_dropout=float(model_cfg.get("attention_dropout", 0.0)),
            attention_gate_init=float(model_cfg.get("attention_gate_init", 0.1)),
            semantic_gate_init=float(model_cfg.get("semantic_gate_init", 0.1)),
            eta_initial=tuple(
                float(value)
                for value in model_cfg.get(
                    "eta_initial", [0.2, 0.0, 0.0][: self.max_order]
                )
            ),
            refine_dropout=dropout,
        )
        self.fusion_skip = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fusion_mlp = _make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"x must be [N, F], got shape={tuple(x.shape)}")
        if x.size(1) < self.text_dim + self.visual_dim:
            raise ValueError(
                f"feature width {x.size(1)} is smaller than configured modality width "
                f"{self.text_dim + self.visual_dim}"
            )
        if not torch.is_floating_point(x):
            x = x.float()
        return (
            x[:, : self.text_dim],
            x[:, self.text_dim : self.text_dim + self.visual_dim],
        )

    @staticmethod
    def _edge_index(edge_index: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if edge_index is None:
            return torch.empty((2, 0), device=device, dtype=torch.long)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
        return edge_index.to(device=device, dtype=torch.long).contiguous()

    def _encode_components(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        x_text, x_visual = self._split_features(x)
        h0_text = self.text_proj(x_text)
        h0_visual = self.visual_proj(x_visual)
        edge_index = self._edge_index(edge_index, x.device)

        relation = self.crc(h0_text, h0_visual, edge_index)
        bank_text = chebyshev_bank(
            h0_text,
            relation["P_text_edge_index"],
            relation["P_text_edge_weight"],
            self.max_order,
        )
        bank_visual = chebyshev_bank(
            h0_visual,
            relation["P_visual_edge_index"],
            relation["P_visual_edge_weight"],
            self.max_order,
        )
        text = self.interaction.forward_modality(
            h0_text, bank_text, relation["q"], "text"
        )
        visual = self.interaction.forward_modality(
            h0_visual, bank_visual, relation["q"], "visual"
        )
        fused = torch.cat((text["z"], visual["z"]), dim=-1)
        z = self.output_norm(self.fusion_skip(fused) + self.fusion_mlp(fused))
        return {
            "h0_text": h0_text,
            "h0_visual": h0_visual,
            "relation": relation,
            "bank_text": bank_text,
            "bank_visual": bank_visual,
            "text": text,
            "visual": visual,
            "z": z,
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        components = self._encode_components(x, edge_index)
        z = components["z"]
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        # Relation scoring and degree normalization are global over the supplied
        # graph, so inference is intentionally one exact full-graph forward.
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self(x.to(device), edge_index.to(device) if edge_index is not None else None)
        return z.detach().cpu()

    @torch.no_grad()
    def relation_diagnostics(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, float | int]:
        """Export compact CRC and downstream mechanism summaries."""
        states = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            x_device = x.to(next(self.parameters()).device)
            components = self._encode_components(x_device, edge_index)
            relation = components["relation"]
            summary = self.crc.diagnostic_summary(relation)
            for modality in ("text", "visual"):
                state = components[modality]
                summary[f"eta_{modality}_mean"] = float(state["eta"].mean().item()) if state["eta"].numel() else 0.0
                summary[f"eta_{modality}_std"] = float(state["eta"].std(unbiased=False).item()) if state["eta"].numel() else 0.0
                for order in range(state["eta"].size(1)):
                    summary[f"eta_{modality}_order{order + 1}_mean"] = float(
                        state["eta"][:, order].mean().item()
                    )
                summary[f"gate_{modality}_mean"] = float(state["semantic_gate"].mean().item())
                summary[f"gate_{modality}_std"] = float(state["semantic_gate"].std(unbiased=False).item())
                h0 = components[f"h0_{modality}"]
                effective_residual = state["semantic_gate"] * state["structural_residual"]
                summary[f"structural_residual_{modality}_rms"] = float(
                    state["structural_residual"].square().mean().sqrt().item()
                )
                summary[f"structural_residual_{modality}_relative_rms"] = float(
                    state["structural_residual"].square().mean().sqrt().item()
                    / h0.square().mean().sqrt().clamp_min(1e-12).item()
                )
                summary[f"effective_structural_residual_{modality}_rms"] = float(
                    effective_residual.square().mean().sqrt().item()
                )
                summary[f"effective_structural_residual_{modality}_relative_rms"] = float(
                    effective_residual.square().mean().sqrt().item()
                    / h0.square().mean().sqrt().clamp_min(1e-12).item()
                )
                summary[f"attention_gate_{modality}"] = float(state["attention_gate"].item())
            q = relation["q"]
            summary["q_mean_norm"] = float(q.norm(dim=-1).mean().item()) if q.numel() else 0.0
            return summary
        finally:
            for module, was_training in states:
                module.training = was_training


__all__ = ["Model", "ProjectionMLP"]
