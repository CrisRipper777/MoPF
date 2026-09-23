from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from .common import make_norm


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _inverse_tanh(value: float) -> float:
    value = float(value)
    if not -1.0 < value < 1.0:
        raise ValueError(f"gate initialization must be between -1 and 1, got {value}")
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


def _logit(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {value}")
    return math.log(value / (1.0 - value))


class ProjectionMLP(nn.Module):
    """Project one modality without receiving features from the other modality."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, norm: str | None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossHopAttentionBlock(nn.Module):
    """One-head self-attention over the K+1 hop tokens of each node."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.scale = hidden_dim**-0.5


class SSIMAGV3(nn.Module):
    """SSI-MAG-V3: modality-specific structure-semantic context reasoning.

    Text and visual streams remain separate through relation modulation,
    semantic-reference contextualization, cross-hop attention, signed
    filtering, and residual refinement.  They are concatenated only in the
    final late-fusion block.
    """

    requires_full_graph_training = True
    requires_full_lp_sampler_depth = True
    no_weight_decay_parameter_names = frozenset(
        {"theta_beta_text", "theta_beta_visual"}
    )

    ABLATIONS = {
        "full",
        "raw_relation",
        "fixed_reference",
        "terminal_context",
        "no_context_change",
        "no_cross_hop_interaction",
        "uniform_context",
    }

    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        if self.text_dim + self.visual_dim > input_dim:
            raise ValueError(
                "text_dim+visual_dim="
                f"{self.text_dim + self.visual_dim} exceeds input_dim={input_dim}"
            )

        model_cfg = cfg.model
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.out_dim = self.hidden_dim
        dropout = float(model_cfg.get("dropout", 0.2))
        norm = model_cfg.get("norm", "layernorm")
        self.max_order = int(model_cfg.get("max_order", 3))
        self.num_layers = int(model_cfg.get("num_layers", self.max_order))
        if self.max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {self.max_order}")
        if self.num_layers != self.max_order:
            raise ValueError(
                "SSI-MAG-V3 requires model.num_layers == model.max_order for "
                f"NC/LP framework compatibility, got {self.num_layers} and {self.max_order}"
            )

        self.diffusion_add_self_loops = bool(
            model_cfg.get("diffusion_add_self_loops", True)
        )
        self.relation_rank = int(model_cfg.get("relation_rank", 16))
        self.relation_strength_init = float(
            model_cfg.get("relation_strength_init", 0.05)
        )
        self.semantic_reference_init = float(
            model_cfg.get("semantic_reference_init", 0.10)
        )
        self.context_change_gate_init = float(
            model_cfg.get("context_change_gate_init", 0.10)
        )
        self.hop_interaction_gate_init = float(
            model_cfg.get("hop_interaction_gate_init", 0.10)
        )
        self.filter_rank = int(model_cfg.get("filter_rank", 4))
        self.global_prior_restart = float(model_cfg.get("global_prior_restart", 0.15))
        self.global_prior_order = int(model_cfg.get("global_prior_order", 2))
        self.reference_filter_scale_init = float(
            model_cfg.get("reference_filter_scale_init", 0.10)
        )
        self.relation_filter_scale_init = float(
            model_cfg.get("relation_filter_scale_init", 0.10)
        )
        self.eps = float(model_cfg.get("eps", 1.0e-8))
        self.edge_chunk_size = int(model_cfg.get("relation_edge_chunk_size", 65536))
        self.relation_init_seed = int(model_cfg.get("relation_init_seed", 20260923))
        self.relation_init_std = float(model_cfg.get("relation_init_std", 0.02))
        self.order_embedding_enabled = bool(
            model_cfg.get("hop_interaction_order_embedding", True)
        )
        interaction_layers = int(model_cfg.get("hop_interaction_layers", 1))
        interaction_heads = int(model_cfg.get("hop_interaction_heads", 1))
        interaction_dropout = float(model_cfg.get("hop_interaction_dropout", 0.0))
        fusion = str(model_cfg.get("fusion", "concat_residual_mlp")).lower()
        if self.relation_rank < 1 or self.filter_rank < 1:
            raise ValueError("relation_rank and filter_rank must be positive")
        if self.eps <= 0.0 or self.edge_chunk_size < 1 or self.relation_init_std <= 0.0:
            raise ValueError(
                "eps, relation_edge_chunk_size, and relation_init_std must be positive"
            )
        if not 0.0 < self.relation_strength_init < 1.0:
            raise ValueError("relation_strength_init must be in (0, 1)")
        if not 0.0 < self.semantic_reference_init < 1.0:
            raise ValueError("semantic_reference_init must be in (0, 1)")
        if interaction_layers != 1 or interaction_heads != 1:
            raise ValueError("SSI-MAG-V3 uses exactly one cross-hop attention layer and head")
        if interaction_dropout != 0.0:
            raise ValueError("SSI-MAG-V3 cross-hop attention dropout must be 0.0")
        if not self.order_embedding_enabled:
            raise ValueError("SSI-MAG-V3 requires hop interaction order embeddings")
        if fusion != "concat_residual_mlp":
            raise ValueError("SSI-MAG-V3 only supports fusion=concat_residual_mlp")
        if not 0.0 <= self.global_prior_restart <= 1.0:
            raise ValueError("global_prior_restart must be in [0, 1]")
        if self.global_prior_order < 0:
            raise ValueError("global_prior_order must be non-negative")

        top_level_ablation = cfg.get("ablation", None) if hasattr(cfg, "get") else None
        configured_ablation = (
            top_level_ablation
            if top_level_ablation not in {None, "full"}
            else model_cfg.get("ablation", top_level_ablation or "full")
        )
        self.ablation = str(configured_ablation or "full").strip().lower()
        if self.ablation not in self.ABLATIONS:
            valid = ", ".join(sorted(self.ABLATIONS))
            raise ValueError(f"unknown SSI-MAG-V3 ablation {self.ablation!r}; expected {valid}")

        self.text_proj = ProjectionMLP(self.text_dim, self.hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, self.hidden_dim, dropout, norm)

        descriptor_dim = 2 * self.relation_rank + 1
        self.relation_norm_text = nn.LayerNorm(self.hidden_dim)
        self.relation_norm_visual = nn.LayerNorm(self.hidden_dim)
        self.relation_proj_text = nn.Linear(self.hidden_dim, self.relation_rank)
        self.relation_proj_visual = nn.Linear(self.hidden_dim, self.relation_rank)
        self.relation_scorer_text = _make_mlp(
            descriptor_dim, self.relation_rank, 1, 0.0
        )
        self.relation_scorer_visual = _make_mlp(
            descriptor_dim, self.relation_rank, 1, 0.0
        )
        self.theta_beta_text = nn.Parameter(
            torch.tensor(_logit(self.relation_strength_init))
        )
        self.theta_beta_visual = nn.Parameter(
            torch.tensor(_logit(self.relation_strength_init))
        )

        order_count = self.max_order + 1
        self.semantic_prior_norm_text = nn.LayerNorm(self.hidden_dim)
        self.semantic_prior_norm_visual = nn.LayerNorm(self.hidden_dim)
        init_generator = torch.Generator(device="cpu").manual_seed(self.relation_init_seed)
        self.semantic_prior_vector_text = nn.Parameter(
            torch.randn(self.hidden_dim, generator=init_generator) * self.relation_init_std
        )
        self.semantic_prior_vector_visual = nn.Parameter(
            torch.randn(self.hidden_dim, generator=init_generator) * self.relation_init_std
        )
        self.semantic_bias_text = nn.Parameter(torch.zeros(self.max_order))
        self.semantic_bias_visual = nn.Parameter(torch.zeros(self.max_order))
        self.semantic_rho_p_text = nn.Parameter(torch.zeros(self.max_order))
        self.semantic_rho_p_visual = nn.Parameter(torch.zeros(self.max_order))
        self.semantic_rho_d_text = nn.Parameter(torch.zeros(self.max_order))
        self.semantic_rho_d_visual = nn.Parameter(torch.zeros(self.max_order))
        self.semantic_rho_c_text = nn.Parameter(torch.zeros(self.max_order))
        self.semantic_rho_c_visual = nn.Parameter(torch.zeros(self.max_order))

        self.context_delta_norm_text = nn.LayerNorm(self.hidden_dim)
        self.context_delta_norm_visual = nn.LayerNorm(self.hidden_dim)
        self.context_delta_proj_text = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.context_delta_proj_visual = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=False
        )
        self.theta_delta_text = nn.Parameter(
            torch.tensor(_inverse_tanh(self.context_change_gate_init))
        )
        self.theta_delta_visual = nn.Parameter(
            torch.tensor(_inverse_tanh(self.context_change_gate_init))
        )

        self.hop_order_embedding_text = nn.Parameter(
            torch.zeros(order_count, self.hidden_dim)
        )
        self.hop_order_embedding_visual = nn.Parameter(
            torch.zeros(order_count, self.hidden_dim)
        )
        self.hop_layers_text = nn.ModuleList(
            [CrossHopAttentionBlock(self.hidden_dim, interaction_dropout)]
        )
        self.hop_layers_visual = nn.ModuleList(
            [CrossHopAttentionBlock(self.hidden_dim, interaction_dropout)]
        )
        self.theta_int_text = nn.Parameter(
            torch.tensor(_inverse_tanh(self.hop_interaction_gate_init))
        )
        self.theta_int_visual = nn.Parameter(
            torch.tensor(_inverse_tanh(self.hop_interaction_gate_init))
        )

        self.gamma_global = nn.Parameter(self._make_global_prior())
        self.delta_gamma_text = nn.Parameter(torch.zeros(order_count))
        self.delta_gamma_visual = nn.Parameter(torch.zeros(order_count))
        self.node_proj_text = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.filter_rank) for _ in range(order_count)
        )
        self.node_proj_visual = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.filter_rank) for _ in range(order_count)
        )
        self.node_vector_text = nn.Parameter(
            torch.randn(
                order_count, self.filter_rank, generator=init_generator
            )
            * self.relation_init_std
        )
        self.node_vector_visual = nn.Parameter(
            torch.randn(
                order_count, self.filter_rank, generator=init_generator
            )
            * self.relation_init_std
        )
        self.relation_order_profile_text = nn.Parameter(
            torch.randn(order_count, generator=init_generator) * self.relation_init_std
        )
        self.relation_order_profile_visual = nn.Parameter(
            torch.randn(order_count, generator=init_generator) * self.relation_init_std
        )
        self.reference_filter_scale_text = nn.Parameter(
            torch.tensor(self.reference_filter_scale_init)
        )
        self.reference_filter_scale_visual = nn.Parameter(
            torch.tensor(self.reference_filter_scale_init)
        )
        self.relation_filter_scale_text = nn.Parameter(
            torch.tensor(self.relation_filter_scale_init)
        )
        self.relation_filter_scale_visual = nn.Parameter(
            torch.tensor(self.relation_filter_scale_init)
        )

        self.text_refine_mlp = _make_mlp(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout
        )
        self.visual_refine_mlp = _make_mlp(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout
        )
        self.text_refine_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_refine_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_mlp = _make_mlp(
            2 * self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def _make_global_prior(self) -> torch.Tensor:
        restart = self.global_prior_restart
        prior = torch.zeros(self.max_order + 1, dtype=torch.float32)
        if self.max_order >= self.global_prior_order:
            for order in range(self.global_prior_order):
                prior[order] = restart * (1.0 - restart) ** order
            prior[self.global_prior_order] = (1.0 - restart) ** self.global_prior_order
        else:
            for order in range(self.max_order):
                prior[order] = restart * (1.0 - restart) ** order
            prior[self.max_order] = (1.0 - restart) ** self.max_order
        return prior

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype == torch.long:
            x = x.float()
        return (
            x[:, : self.text_dim],
            x[:, self.text_dim : self.text_dim + self.visual_dim],
        )

    @staticmethod
    def _edge_index_or_empty(
        edge_index: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor:
        if edge_index is None:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index.to(device=device, dtype=torch.long)

    def _relation_descriptor(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        modality: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modality == "text":
            normalized = self.relation_norm_text(h0)
            relation_projection = self.relation_proj_text(normalized)
            scorer = self.relation_scorer_text
        else:
            normalized = self.relation_norm_visual(h0)
            relation_projection = self.relation_proj_visual(normalized)
            scorer = self.relation_scorer_visual

        if edge_index.numel() == 0:
            descriptor = h0.new_empty((0, 2 * self.relation_rank + 1))
            return relation_projection, h0.new_empty((0,))

        src, dst = edge_index
        residuals: list[torch.Tensor] = []
        for start in range(0, edge_index.size(1), self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, edge_index.size(1))
            source = relation_projection[src[start:stop]]
            target = relation_projection[dst[start:stop]]
            descriptor = torch.cat(
                (
                    source * target,
                    (source - target).abs(),
                    F.cosine_similarity(source, target, dim=-1, eps=self.eps).unsqueeze(-1),
                ),
                dim=-1,
            )
            residuals.append(torch.tanh(scorer(descriptor).squeeze(-1)))
        return relation_projection, torch.cat(residuals, dim=0)

    def _relation_edge_outputs(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        modality: str,
        *,
        force_unit: bool = False,
    ) -> dict[str, torch.Tensor]:
        relation_projection, residual = self._relation_descriptor(h0, edge_index, modality)
        theta_beta = getattr(self, f"theta_beta_{modality}")
        beta = torch.sigmoid(theta_beta)
        if force_unit:
            beta_for_weight = torch.zeros_like(beta)
        else:
            beta_for_weight = beta
        relation_weight = torch.exp(beta_for_weight * residual)
        return {
            "relation_projection": relation_projection,
            "relation_residual": residual,
            "beta": beta_for_weight,
            "beta_parameter": beta,
            "relation_weight": relation_weight,
        }

    @staticmethod
    def _local_adaptation(
        edge_index: torch.Tensor,
        relation_residual: torch.Tensor,
        beta: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return relation_residual.new_zeros((num_nodes,))
        endpoints = torch.cat((edge_index[0], edge_index[1]), dim=0)
        incident = torch.cat(
            ((beta * relation_residual).abs(), (beta * relation_residual).abs()), dim=0
        )
        total = relation_residual.new_zeros((num_nodes,))
        count = relation_residual.new_zeros((num_nodes,))
        total.index_add_(0, endpoints, incident)
        count.index_add_(0, endpoints, torch.ones_like(incident))
        return torch.where(
            count > 0,
            total / count.clamp_min(1.0),
            torch.zeros_like(total),
        )

    def _normalized_operator(
        self,
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
            add_self_loops=self.diffusion_add_self_loops,
            flow="source_to_target",
            dtype=dtype,
        )
        if normalized_weight is None:
            normalized_weight = torch.ones(
                normalized_index.size(1), dtype=dtype, device=normalized_index.device
            )
        return normalized_index, torch.nan_to_num(
            normalized_weight, nan=0.0, posinf=0.0, neginf=0.0
        )

    @staticmethod
    def _propagate_once(
        h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index
        return scatter(
            h[src] * edge_weight.unsqueeze(-1),
            dst,
            dim=0,
            dim_size=h.size(0),
            reduce="sum",
        )

    def _semantic_states(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        local_adaptation: torch.Tensor,
        modality: str,
        *,
        fixed_reference: bool,
    ) -> dict[str, Any]:
        prior_norm = getattr(self, f"semantic_prior_norm_{modality}")
        prior_vector = getattr(self, f"semantic_prior_vector_{modality}")
        biases = getattr(self, f"semantic_bias_{modality}")
        rho_p = getattr(self, f"semantic_rho_p_{modality}")
        rho_d = getattr(self, f"semantic_rho_d_{modality}")
        rho_c = getattr(self, f"semantic_rho_c_{modality}")
        p = torch.tanh((prior_norm(h0) * prior_vector).sum(dim=-1))

        states = [h0]
        propagated = []
        changes = []
        alphas = []
        current = h0
        logit_alpha0 = _logit(self.semantic_reference_init)
        for order in range(1, self.max_order + 1):
            q = self._propagate_once(current, edge_index, edge_weight)
            d = 1.0 - F.cosine_similarity(q, h0, dim=-1, eps=self.eps)
            if fixed_reference:
                alpha = torch.full_like(d, self.semantic_reference_init)
            else:
                alpha = torch.sigmoid(
                    torch.as_tensor(logit_alpha0, dtype=h0.dtype, device=h0.device)
                    - biases[order - 1]
                    - rho_p[order - 1] * p
                    - rho_d[order - 1] * d
                    - rho_c[order - 1] * local_adaptation
                )
            s = (1.0 - alpha).unsqueeze(-1) * q - alpha.unsqueeze(-1) * h0
            propagated.append(q)
            changes.append(d)
            alphas.append(alpha)
            states.append(s)
            current = s
        return {
            "states": states,
            "propagated": propagated,
            "changes": changes,
            "alphas": alphas,
            "p": p,
        }

    def _context_tokens(
        self,
        states: list[torch.Tensor],
        modality: str,
        *,
        disable_change: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        delta_norm = getattr(self, f"context_delta_norm_{modality}")
        delta_proj = getattr(self, f"context_delta_proj_{modality}")
        theta_delta = getattr(self, f"theta_delta_{modality}")
        order_embedding = getattr(self, f"hop_order_embedding_{modality}")
        delta_gate = torch.tanh(theta_delta)
        if disable_change:
            delta_gate = torch.zeros_like(delta_gate)
        deltas = [torch.zeros_like(states[0])]
        tokens = [
            F.layer_norm(
                states[0], (self.hidden_dim,)
            ) - order_embedding[0].unsqueeze(0)
        ]
        for order in range(1, len(states)):
            delta = delta_norm(states[order] - states[order - 1])
            if disable_change:
                delta = torch.zeros_like(delta)
            token = (
                F.layer_norm(states[order], (self.hidden_dim,))
                - delta_gate * delta_proj(delta)
                - order_embedding[order].unsqueeze(0)
            )
            deltas.append(delta)
            tokens.append(token)
        return deltas, tokens, delta_gate

    def _cross_hop_interaction(
        self,
        states: list[torch.Tensor],
        tokens: list[torch.Tensor],
        modality: str,
        *,
        interaction_off: bool,
        capture_attention: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        state_bank = torch.stack(states, dim=1)
        token_bank = torch.stack(tokens, dim=1)
        layer = self.hop_layers_text[0] if modality == "text" else self.hop_layers_visual[0]
        theta_int = self.theta_int_text if modality == "text" else self.theta_int_visual
        gate = torch.tanh(theta_int)
        if interaction_off:
            gate = torch.zeros_like(gate)
        normalized = layer.norm(token_bank)
        query = layer.query(normalized)
        key = layer.key(normalized)
        value = layer.value(normalized)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) * layer.scale, dim=-1
        )
        output_bank = torch.matmul(layer.attention_dropout(attention), value)
        if interaction_off:
            s_tilde_bank = state_bank
        else:
            s_tilde_bank = state_bank - gate * output_bank
        return (
            [s_tilde_bank[:, order, :] for order in range(s_tilde_bank.size(1))],
            attention,
            gate,
        )

    def _relation_profile(self, modality: str) -> torch.Tensor:
        raw = getattr(self, f"relation_order_profile_{modality}")
        centered = raw - raw.mean()
        denominator = centered.square().mean().sqrt().clamp_min(self.eps)
        return centered / denominator

    def _node_content_score(
        self, states: list[torch.Tensor], modality: str
    ) -> torch.Tensor:
        projectors = getattr(self, f"node_proj_{modality}")
        node_vector = getattr(self, f"node_vector_{modality}")
        scores = []
        for order, state in enumerate(states):
            projected = torch.tanh(projectors[order](state))
            scores.append((projected * node_vector[order]).sum(dim=-1) / float(self.filter_rank))
        return torch.stack(scores, dim=-1)

    @staticmethod
    def _compose(states: list[torch.Tensor], eta: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(states[0])
        for order, state in enumerate(states):
            result = result + eta[:, order : order + 1] * state
        return result

    @staticmethod
    def _effective_order(eta: torch.Tensor) -> torch.Tensor:
        orders = torch.arange(eta.size(-1), dtype=eta.dtype, device=eta.device)
        weights = eta.abs()
        return (weights * orders.unsqueeze(0)).sum(dim=-1) / weights.sum(dim=-1).clamp_min(
            torch.finfo(eta.dtype).eps
        )

    def _filter_contexts(
        self,
        states: list[torch.Tensor],
        semantic: dict[str, Any],
        local_adaptation: torch.Tensor,
        modality: str,
        *,
        terminal_context: bool,
        uniform_context: bool,
    ) -> dict[str, torch.Tensor]:
        alpha_bank = torch.cat(
            [torch.ones_like(semantic["p"]).unsqueeze(-1)]
            + [alpha.unsqueeze(-1) for alpha in semantic["alphas"]],
            dim=-1,
        )
        alpha_centered = alpha_bank - alpha_bank.mean(dim=-1, keepdim=True)
        relation_profile = self._relation_profile(modality).to(
            device=states[0].device, dtype=states[0].dtype
        )
        reference_scale = getattr(self, f"reference_filter_scale_{modality}")
        relation_scale = getattr(self, f"relation_filter_scale_{modality}")
        reference_residual = reference_scale * alpha_centered
        relation_residual = (
            relation_scale
            * local_adaptation.unsqueeze(-1)
            * relation_profile.unsqueeze(0)
        )
        delta_content = self._node_content_score(states, modality)
        delta = delta_content - reference_residual - relation_residual
        gamma = self.gamma_global.to(dtype=states[0].dtype).unsqueeze(0)
        delta_gamma = getattr(self, f"delta_gamma_{modality}").unsqueeze(0)
        eta_adaptive = gamma - delta_gamma - delta
        if uniform_context:
            eta = torch.full_like(eta_adaptive, 1.0 / float(len(states)))
        elif terminal_context:
            eta = torch.zeros_like(eta_adaptive)
            eta[:, -1] = 1.0
        else:
            eta = eta_adaptive
        return {
            "alpha_bank": alpha_bank,
            "alpha_centered": alpha_centered,
            "relation_profile": relation_profile,
            "delta_content": delta_content,
            "reference_residual": reference_residual,
            "relation_residual": relation_residual,
            "delta": delta,
            "eta_adaptive": eta_adaptive,
            "eta": eta,
        }

    def _encode_modality(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        modality: str,
        *,
        relation_intervention: str,
        interaction_intervention: str,
        capture_attention: bool,
    ) -> dict[str, Any]:
        force_unit = self.ablation == "raw_relation" or relation_intervention == "off"
        relation = self._relation_edge_outputs(
            h0, edge_index, modality, force_unit=force_unit
        )
        beta = relation["beta"]
        local_adaptation = self._local_adaptation(
            edge_index, relation["relation_residual"], beta, h0.size(0)
        )
        norm_index, norm_weight = self._normalized_operator(
            edge_index, relation["relation_weight"], h0.size(0), h0.dtype
        )
        semantic = self._semantic_states(
            h0,
            norm_index,
            norm_weight,
            local_adaptation,
            modality,
            fixed_reference=self.ablation == "fixed_reference",
        )
        disable_change = self.ablation == "no_context_change"
        deltas, tokens, delta_gate = self._context_tokens(
            semantic["states"], modality, disable_change=disable_change
        )
        interaction_off = (
            self.ablation == "no_cross_hop_interaction"
            or interaction_intervention == "off"
        )
        s_tilde, attention, interaction_gate = self._cross_hop_interaction(
            semantic["states"],
            tokens,
            modality,
            interaction_off=interaction_off,
            capture_attention=capture_attention,
        )
        filtering = self._filter_contexts(
            s_tilde,
            semantic,
            local_adaptation,
            modality,
            terminal_context=self.ablation == "terminal_context",
            uniform_context=self.ablation == "uniform_context",
        )
        z = self._compose(s_tilde, filtering["eta"])
        return {
            **relation,
            "normalized_edge_index": norm_index,
            "normalized_edge_weight": norm_weight,
            "local_adaptation": local_adaptation,
            "semantic": semantic,
            "deltas": deltas,
            "tokens": tokens,
            "delta_gate": delta_gate,
            "attention": attention if capture_attention else None,
            "interaction_gate": interaction_gate,
            "s_tilde": s_tilde,
            "filtering": filtering,
            "z": z,
        }

    def _encode_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        relation_intervention: str = "normal",
        interaction_intervention: str = "normal",
        capture_attention: bool = False,
    ) -> dict[str, Any]:
        x_text, x_visual = self._split_features(x)
        h0_text = self.text_proj(x_text)
        h0_visual = self.visual_proj(x_visual)
        text = self._encode_modality(
            h0_text,
            edge_index,
            "text",
            relation_intervention=relation_intervention,
            interaction_intervention=interaction_intervention,
            capture_attention=capture_attention,
        )
        visual = self._encode_modality(
            h0_visual,
            edge_index,
            "visual",
            relation_intervention=relation_intervention,
            interaction_intervention=interaction_intervention,
            capture_attention=capture_attention,
        )
        z_text = text["z"]
        z_visual = visual["z"]
        z_text_refined = self.text_refine_norm(z_text + self.text_refine_mlp(z_text))
        z_visual_refined = self.visual_refine_norm(
            z_visual + self.visual_refine_mlp(z_visual)
        )
        fused_input = torch.cat((z_text_refined, z_visual_refined), dim=-1)
        fused = self.output_norm(self.fusion_skip(fused_input) + self.fusion_mlp(fused_input))
        return {
            "h0_text": h0_text,
            "h0_visual": h0_visual,
            "text": text,
            "visual": visual,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": fused,
            "physical_edge_index": edge_index,
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)
        z = torch.nan_to_num(components["z"], nan=0.0, posinf=1e4, neginf=-1e4)
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        del batch_size
        if device is None:
            device = next(self.parameters()).device
        was_training = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            edge_index = edge_index.to(device) if edge_index is not None else None
            z, _, _, _, _ = self(x.to(device), edge_index)
            return z.detach().cpu()
        finally:
            for module, training in was_training:
                module.training = training

    def _analysis_call(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        relation: str = "normal",
        interaction: str = "normal",
    ) -> dict[str, Any]:
        relation = str(relation).strip().lower()
        interaction = str(interaction).strip().lower()
        if relation not in {"normal", "off"}:
            raise ValueError("relation must be normal|off")
        if interaction not in {"normal", "off"}:
            raise ValueError("interaction must be normal|off")
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        training_states = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self._encode_components(
                x,
                edge_index,
                relation_intervention=relation,
                interaction_intervention=interaction,
                capture_attention=True,
            )
        finally:
            for module, was_training in training_states:
                module.training = was_training

    @staticmethod
    def _clone_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, list):
            return [SSIMAGV3._clone_value(item) for item in value]
        if isinstance(value, dict):
            return {key: SSIMAGV3._clone_value(item) for key, item in value.items()}
        return value

    @staticmethod
    def _edge_summary(value: torch.Tensor) -> dict[str, torch.Tensor]:
        if value.numel() == 0:
            zero = value.new_zeros(())
            return {
                "mean": zero,
                "std": zero,
                "abs_mean": zero,
                "min": zero,
                "max": zero,
            }
        return {
            "mean": value.mean(),
            "std": value.std(unbiased=False),
            "abs_mean": value.abs().mean(),
            "min": value.min(),
            "max": value.max(),
        }

    def _analysis_public(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        relation: str = "normal",
        interaction: str = "normal",
    ) -> dict[str, Any]:
        components = self._analysis_call(
            x, edge_index, relation=relation, interaction=interaction
        )
        text = components["text"]
        visual = components["visual"]
        result: dict[str, Any] = {
            "physical_edge_index": components["physical_edge_index"],
            "ablation": self.ablation,
            "z_text": components["z_text"],
            "z_visual": components["z_visual"],
            "z_text_refined": components["z_text_refined"],
            "z_visual_refined": components["z_visual_refined"],
            "fused_z": components["z"],
            "z": components["z"],
        }
        for modality, branch in (("text", text), ("visual", visual)):
            semantic = branch["semantic"]
            filtering = branch["filtering"]
            result.update(
                {
                    f"beta_{modality}": branch["beta"],
                    f"beta_parameter_{modality}": branch["beta_parameter"],
                    f"relation_projection_{modality}": branch["relation_projection"],
                    f"relation_residual_{modality}": branch["relation_residual"],
                    f"relation_weight_{modality}": branch["relation_weight"],
                    f"relation_residual_mean_{modality}": self._edge_summary(
                        branch["relation_residual"]
                    )["mean"],
                    f"relation_residual_std_{modality}": self._edge_summary(
                        branch["relation_residual"]
                    )["std"],
                    f"relation_residual_abs_mean_{modality}": self._edge_summary(
                        branch["relation_residual"]
                    )["abs_mean"],
                    f"relation_weight_mean_{modality}": self._edge_summary(
                        branch["relation_weight"]
                    )["mean"],
                    f"relation_weight_std_{modality}": self._edge_summary(
                        branch["relation_weight"]
                    )["std"],
                    f"relation_weight_min_{modality}": self._edge_summary(
                        branch["relation_weight"]
                    )["min"],
                    f"relation_weight_max_{modality}": self._edge_summary(
                        branch["relation_weight"]
                    )["max"],
                    f"local_adaptation_{modality}": branch["local_adaptation"],
                    f"p_{modality}": semantic["p"],
                    f"gamma_{modality}": self.gamma_global,
                    f"delta_gamma_{modality}": getattr(self, f"delta_gamma_{modality}"),
                    f"delta_content_{modality}": filtering["delta_content"],
                    f"reference_residual_{modality}": filtering["reference_residual"],
                    f"relation_filter_residual_{modality}": filtering["relation_residual"],
                    f"delta_{modality}": filtering["delta"],
                    f"eta_adaptive_{modality}": filtering["eta_adaptive"],
                    f"eta_{modality}": filtering["eta"],
                    f"effective_order_{modality}": self._effective_order(filtering["eta"]),
                    f"effective_radius_{modality}": self._effective_order(filtering["eta"]),
                    f"attention_{modality}": branch["attention"],
                    f"delta_gate_{modality}": branch["delta_gate"],
                    f"interaction_gate_{modality}": branch["interaction_gate"],
                    f"relation_profile_{modality}": filtering["relation_profile"],
                    f"normalized_edge_index_{modality}": branch["normalized_edge_index"],
                    f"normalized_edge_weight_{modality}": branch["normalized_edge_weight"],
                    f"d_norm_{modality}": [
                        value.norm(dim=-1) for value in branch["deltas"]
                    ],
                    f"Q_{modality}": semantic["propagated"],
                    f"S_{modality}": semantic["states"],
                    f"D_{modality}": branch["deltas"],
                    f"alpha_{modality}": semantic["alphas"],
                    f"S_tilde_{modality}": branch["s_tilde"],
                }
            )
        return self._clone_value(result)

    @torch.no_grad()
    def analysis(self, x: torch.Tensor, edge_index: torch.Tensor | None) -> dict[str, Any]:
        """Export all Stage-I/Stage-II quantities without changing training behavior."""
        return self._analysis_public(x, edge_index)

    @torch.no_grad()
    def analysis_stats(self, x: torch.Tensor, edge_index: torch.Tensor | None) -> dict[str, Any]:
        return self.analysis(x, edge_index)

    @torch.no_grad()
    def analysis_relation_calibration(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, Any]:
        """Return the Stage-I relation quantities using the common analysis API."""
        result = self.analysis(x, edge_index)
        keys = (
            "physical_edge_index",
            "relation_projection_text",
            "relation_projection_visual",
            "relation_residual_text",
            "relation_residual_visual",
            "relation_weight_text",
            "relation_weight_visual",
            "beta_text",
            "beta_visual",
            "local_adaptation_text",
            "local_adaptation_visual",
            "normalized_edge_index_text",
            "normalized_edge_index_visual",
            "normalized_edge_weight_text",
            "normalized_edge_weight_visual",
        )
        return {key: result[key] for key in keys}

    @torch.no_grad()
    def analysis_multi_order(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, Any]:
        return self.analysis(x, edge_index)

    @torch.no_grad()
    def analysis_intervention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        interaction: str = "normal",
        relation: str = "normal",
    ) -> dict[str, Any]:
        result = self._analysis_public(
            x, edge_index, relation=relation, interaction=interaction
        )
        result["interaction_intervention"] = str(interaction)
        result["relation_intervention"] = str(relation)
        return result


Model = SSIMAGV3
