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


class SSIMAGV31(nn.Module):
    """SSI-MAG-V3.1: modality-specific structure-semantic context reasoning.

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

    ABLATIONS = {"full"}

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
                "SSI-MAG-V3.1 requires model.num_layers == model.max_order for "
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
        # Kept local only for compatibility with the copied constructor block;
        # V3.1 intentionally does not create a reference-filter parameter.
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
            raise ValueError("SSI-MAG-V3.1 uses exactly one cross-hop attention layer and head")
        if interaction_dropout != 0.0:
            raise ValueError("SSI-MAG-V3.1 cross-hop attention dropout must be 0.0")
        if not self.order_embedding_enabled:
            raise ValueError("SSI-MAG-V3.1 requires hop interaction order embeddings")
        if fusion != "concat_residual_mlp":
            raise ValueError("SSI-MAG-V3.1 only supports fusion=concat_residual_mlp")
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
            raise ValueError(
                "SSI-MAG-V3.1 P1.7a is frozen to ablation=full; "
                f"got {self.ablation!r}"
            )

        self.text_proj = ProjectionMLP(self.text_dim, self.hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, self.hidden_dim, dropout, norm)

        self.relation_norm_text = nn.LayerNorm(self.hidden_dim)
        self.relation_norm_visual = nn.LayerNorm(self.hidden_dim)
        # R1-B: semantic relation projections are bias-free so relation
        # compatibility is measured in a centered modality representation.
        self.relation_proj_text = nn.Linear(
            self.hidden_dim, self.relation_rank, bias=False
        )
        self.relation_proj_visual = nn.Linear(
            self.hidden_dim, self.relation_rank, bias=False
        )
        self.relation_scorer_text = nn.Linear(3, 1, bias=True)
        self.relation_scorer_visual = nn.Linear(3, 1, bias=True)
        for scorer in (self.relation_scorer_text, self.relation_scorer_visual):
            nn.init.zeros_(scorer.weight)
            nn.init.zeros_(scorer.bias)
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
        # Stabilized semantic reference: p is normalized by the learned
        # prior-vector norm and has no tanh saturation or c branch.
        self.semantic_rho_p_text = nn.Parameter(torch.zeros(()))
        self.semantic_rho_p_visual = nn.Parameter(torch.zeros(()))
        self.semantic_rho_d_text = nn.Parameter(torch.zeros(()))
        self.semantic_rho_d_visual = nn.Parameter(torch.zeros(()))

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
        # V3.1 has no Stage-II reference residual or reference scale.
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
    ) -> dict[str, torch.Tensor]:
        """Compute symmetric semantic compatibility on the original edge support.

        The mu statistics deliberately exclude all self-loops, including
        explicit input self-loops.  Newly inserted gcn_norm self-loops are
        therefore never visible to either mu or c.
        """
        if modality == "text":
            normalized = self.relation_norm_text(h0)
            relation_projection = self.relation_proj_text(normalized)
            scorer = self.relation_scorer_text
        else:
            normalized = self.relation_norm_visual(h0)
            relation_projection = self.relation_proj_visual(normalized)
            scorer = self.relation_scorer_visual

        num_edges = int(edge_index.size(1))
        if num_edges == 0:
            empty = h0.new_empty((0,))
            return {
                "relation_projection": relation_projection,
                "compatibility": empty,
                "relation_mu": h0.new_zeros((h0.size(0),)),
                "relation_centered": empty,
                "relation_score": empty,
                "nonself_mask": torch.zeros(
                    (0,), dtype=torch.bool, device=h0.device
                ),
            }

        src, dst = edge_index
        compatibility_chunks: list[torch.Tensor] = []
        for start in range(0, num_edges, self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, num_edges)
            source = relation_projection[src[start:stop]]
            target = relation_projection[dst[start:stop]]
            compatibility_chunks.append(
                F.cosine_similarity(source, target, dim=-1, eps=self.eps)
            )
        compatibility = torch.cat(compatibility_chunks, dim=0)
        nonself_mask = src.ne(dst)

        relation_mu = h0.new_zeros((h0.size(0),))
        if bool(nonself_mask.any()):
            nonself_src = src[nonself_mask]
            nonself_dst = dst[nonself_mask]
            incident_nodes = torch.cat((nonself_src, nonself_dst), dim=0)
            incident_values = torch.cat(
                (compatibility[nonself_mask], compatibility[nonself_mask]), dim=0
            )
            incident_sum = h0.new_zeros((h0.size(0),))
            incident_count = h0.new_zeros((h0.size(0),))
            incident_sum.index_add_(0, incident_nodes, incident_values)
            incident_count.index_add_(
                0, incident_nodes, torch.ones_like(incident_values)
            )
            relation_mu = torch.where(
                incident_count > 0,
                incident_sum / incident_count.clamp_min(1.0),
                torch.zeros_like(incident_sum),
            )

        relation_centered = compatibility - 0.5 * (
            relation_mu[src] + relation_mu[dst]
        )
        score_chunks: list[torch.Tensor] = []
        for start in range(0, num_edges, self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, num_edges)
            descriptor = torch.stack(
                (
                    compatibility[start:stop],
                    relation_centered[start:stop],
                    relation_centered[start:stop].abs(),
                ),
                dim=-1,
            )
            score_chunks.append(F.softsign(scorer(descriptor).squeeze(-1)))
        relation_score = torch.cat(score_chunks, dim=0)
        return {
            "relation_projection": relation_projection,
            "compatibility": compatibility,
            "relation_mu": relation_mu,
            "relation_centered": relation_centered,
            "relation_score": relation_score,
            "nonself_mask": nonself_mask,
        }

    def _relation_edge_outputs(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        modality: str,
        *,
        force_unit: bool = False,
    ) -> dict[str, torch.Tensor]:
        relation = self._relation_descriptor(h0, edge_index, modality)
        theta_beta = getattr(self, f"theta_beta_{modality}")
        beta_parameter = torch.sigmoid(theta_beta)
        beta = torch.zeros_like(beta_parameter) if force_unit else beta_parameter
        relation_weight = torch.exp(beta * relation["relation_score"])
        return {
            **relation,
            "relation_residual": relation["relation_score"],
            "beta": beta,
            "beta_parameter": beta_parameter,
            "relation_weight": relation_weight,
        }

    @staticmethod
    def _local_adaptation(
        edge_index: torch.Tensor,
        relation_residual: torch.Tensor,
        beta: torch.Tensor,
        nonself_mask: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Mean incident |beta*a| over original non-self physical relations."""
        if edge_index.numel() == 0 or not bool(nonself_mask.any()):
            return relation_residual.new_zeros((num_nodes,))
        physical_index = edge_index[:, nonself_mask]
        incident_values = (beta * relation_residual[nonself_mask]).abs()
        endpoints = torch.cat((physical_index[0], physical_index[1]), dim=0)
        incident = torch.cat((incident_values, incident_values), dim=0)
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
        modality: str,
    ) -> dict[str, Any]:
        prior_norm = getattr(self, f"semantic_prior_norm_{modality}")
        prior_vector = getattr(self, f"semantic_prior_vector_{modality}")
        biases = getattr(self, f"semantic_bias_{modality}")
        rho_p = getattr(self, f"semantic_rho_p_{modality}")
        rho_d = getattr(self, f"semantic_rho_d_{modality}")
        h = prior_norm(h0)
        p = (h * prior_vector).sum(dim=-1) / prior_vector.norm(p=2).clamp_min(
            self.eps
        )

        states = [h0]
        propagated = []
        changes = []
        alphas = []
        current = h0
        logit_alpha0 = _logit(self.semantic_reference_init)
        for order in range(1, self.max_order + 1):
            q = self._propagate_once(current, edge_index, edge_weight)
            d = 1.0 - F.cosine_similarity(q, h0, dim=-1, eps=self.eps)
            alpha = torch.sigmoid(
                torch.as_tensor(logit_alpha0, dtype=h0.dtype, device=h0.device)
                + biases[order - 1]
                + rho_p * p
                + rho_d * d
            )
            s = (1.0 - alpha).unsqueeze(-1) * q + alpha.unsqueeze(-1) * h0
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
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        delta_norm = getattr(self, f"context_delta_norm_{modality}")
        delta_proj = getattr(self, f"context_delta_proj_{modality}")
        theta_delta = getattr(self, f"theta_delta_{modality}")
        order_embedding = getattr(self, f"hop_order_embedding_{modality}")
        delta_gate = torch.tanh(theta_delta)
        deltas = [torch.zeros_like(states[0])]
        injections = [torch.zeros_like(states[0])]
        tokens = [
            F.layer_norm(states[0], (self.hidden_dim,))
            + order_embedding[0].unsqueeze(0)
        ]
        for order in range(1, len(states)):
            delta = delta_norm(states[order] - states[order - 1])
            injection = delta_gate * delta_proj(delta)
            token = (
                F.layer_norm(states[order], (self.hidden_dim,))
                + injection
                + order_embedding[order].unsqueeze(0)
            )
            deltas.append(delta)
            injections.append(injection)
            tokens.append(token)
        return deltas, tokens, delta_gate, injections

    def _cross_hop_interaction(
        self,
        states: list[torch.Tensor],
        tokens: list[torch.Tensor],
        modality: str,
        *,
        interaction_off: bool,
        capture_attention: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
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
            s_tilde_bank = state_bank + gate * output_bank
        return (
            [s_tilde_bank[:, order, :] for order in range(s_tilde_bank.size(1))],
            attention,
            gate,
            output_bank,
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
        local_adaptation: torch.Tensor,
        modality: str,
    ) -> dict[str, torch.Tensor]:
        relation_profile = self._relation_profile(modality).to(
            device=states[0].device, dtype=states[0].dtype
        )
        relation_scale = getattr(self, f"relation_filter_scale_{modality}")
        relation_residual = (
            relation_scale
            * local_adaptation.unsqueeze(-1)
            * relation_profile.unsqueeze(0)
        )
        delta_content = self._node_content_score(states, modality)
        gamma = self.gamma_global.to(dtype=states[0].dtype).unsqueeze(0)
        delta_gamma = getattr(self, f"delta_gamma_{modality}").unsqueeze(0)
        eta = gamma + delta_gamma + delta_content + relation_residual
        return {
            "relation_profile": relation_profile,
            "delta_content": delta_content,
            "relation_residual": relation_residual,
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
        force_unit = relation_intervention == "off"
        relation = self._relation_edge_outputs(
            h0, edge_index, modality, force_unit=force_unit
        )
        local_adaptation = self._local_adaptation(
            edge_index,
            relation["relation_residual"],
            relation["beta"],
            relation["nonself_mask"],
            h0.size(0),
        )
        norm_index, norm_weight = self._normalized_operator(
            edge_index, relation["relation_weight"], h0.size(0), h0.dtype
        )
        semantic = self._semantic_states(
            h0, norm_index, norm_weight, modality
        )
        deltas, tokens, delta_gate, injections = self._context_tokens(
            semantic["states"], modality
        )
        interaction_off = interaction_intervention == "off"
        s_tilde, attention, interaction_gate, interaction_output = self._cross_hop_interaction(
            semantic["states"],
            tokens,
            modality,
            interaction_off=interaction_off,
            capture_attention=capture_attention,
        )
        filtering = self._filter_contexts(
            s_tilde, local_adaptation, modality
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
            "context_change_injection": injections,
            "attention": attention if capture_attention else None,
            "interaction_gate": interaction_gate,
            "interaction_output": interaction_output,
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
            return [SSIMAGV31._clone_value(item) for item in value]
        if isinstance(value, dict):
            return {key: SSIMAGV31._clone_value(item) for key, item in value.items()}
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
            "h0_text": components["h0_text"],
            "h0_visual": components["h0_visual"],
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
            relation_stats = self._edge_summary(branch["relation_residual"])
            weight_stats = self._edge_summary(branch["relation_weight"])
            rho_p = getattr(self, f"semantic_rho_p_{modality}")
            rho_d = getattr(self, f"semantic_rho_d_{modality}")
            result.update(
                {
                    f"relation_z_{modality}": branch["relation_projection"],
                    f"z_relation_{modality}": branch["relation_projection"],
                    f"relation_compatibility_{modality}": branch["compatibility"],
                    f"relation_mu_{modality}": branch["relation_mu"],
                    f"relation_centered_{modality}": branch["relation_centered"],
                    f"a_{modality}": branch["relation_score"],
                    f"relation_residual_{modality}": branch["relation_residual"],
                    f"relation_weight_{modality}": branch["relation_weight"],
                    f"beta_{modality}": branch["beta"],
                    f"beta_parameter_{modality}": branch["beta_parameter"],
                    f"relation_residual_mean_{modality}": relation_stats["mean"],
                    f"relation_residual_std_{modality}": relation_stats["std"],
                    f"relation_residual_abs_mean_{modality}": relation_stats["abs_mean"],
                    f"relation_weight_mean_{modality}": weight_stats["mean"],
                    f"relation_weight_std_{modality}": weight_stats["std"],
                    f"relation_weight_min_{modality}": weight_stats["min"],
                    f"relation_weight_max_{modality}": weight_stats["max"],
                    f"local_adaptation_{modality}": branch["local_adaptation"],
                    f"c_{modality}": branch["local_adaptation"],
                    f"p_{modality}": semantic["p"],
                    f"term_p_{modality}": rho_p * semantic["p"],
                    f"d_{modality}": semantic["changes"],
                    f"semantic_bias_{modality}": getattr(self, f"semantic_bias_{modality}"),
                    f"semantic_rho_p_{modality}": rho_p,
                    f"semantic_rho_d_{modality}": rho_d,
                    f"gamma_{modality}": self.gamma_global,
                    f"delta_gamma_{modality}": getattr(self, f"delta_gamma_{modality}"),
                    f"delta_content_{modality}": filtering["delta_content"],
                    f"relation_filter_residual_{modality}": filtering["relation_residual"],
                    f"relation_residual_eta_{modality}": filtering["relation_residual"],
                    f"eta_{modality}": filtering["eta"],
                    f"effective_order_{modality}": self._effective_order(filtering["eta"]),
                    f"effective_radius_{modality}": self._effective_order(filtering["eta"]),
                    f"attention_{modality}": branch["attention"],
                    f"tokens_{modality}": branch["tokens"],
                    f"interaction_output_{modality}": branch["interaction_output"],
                    f"delta_gate_{modality}": branch["delta_gate"],
                    f"interaction_gate_{modality}": branch["interaction_gate"],
                    f"relation_profile_{modality}": filtering["relation_profile"],
                    f"normalized_edge_index_{modality}": branch["normalized_edge_index"],
                    f"normalized_edge_weight_{modality}": branch["normalized_edge_weight"],
                    f"d_norm_{modality}": [value.norm(dim=-1) for value in branch["deltas"]],
                    f"Q_{modality}": semantic["propagated"],
                    f"S_{modality}": semantic["states"],
                    f"D_{modality}": branch["deltas"],
                    f"context_change_injection_{modality}": branch["context_change_injection"],
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
            "relation_z_text",
            "relation_z_visual",
            "relation_compatibility_text",
            "relation_compatibility_visual",
            "relation_mu_text",
            "relation_mu_visual",
            "relation_centered_text",
            "relation_centered_visual",
            "a_text",
            "a_visual",
            "relation_residual_text",
            "relation_residual_visual",
            "relation_weight_text",
            "relation_weight_visual",
            "beta_text",
            "beta_visual",
            "local_adaptation_text",
            "local_adaptation_visual",
            "c_text",
            "c_visual",
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


Model = SSIMAGV31
