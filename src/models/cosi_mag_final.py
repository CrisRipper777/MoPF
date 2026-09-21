from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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


def _logit(probability: float) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {value}")
    return math.log(value / (1.0 - value))


class ProjectionMLP(nn.Module):
    """Project one modality without mixing it with the other modality."""

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


class CrossOrderAttentionBlock(nn.Module):
    """Single-head attention over one node's multi-hop order tokens."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(0.0)
        self.scale = float(hidden_dim) ** -0.5


class CoSIMAGFinal(nn.Module):
    """Three-stage multimodal encoder for relation-calibrated multi-order learning."""

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

        hidden_dim = int(cfg.model.get("hidden_dim", 256))
        dropout = float(cfg.model.get("dropout", 0.2))
        norm = cfg.model.get("norm", "layernorm")
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.max_order = int(cfg.model.get("max_order", 3))
        if self.max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {self.max_order}")

        self.global_prior_restart = float(cfg.model.get("global_prior_restart", 0.15))
        self.global_prior_order = int(cfg.model.get("global_prior_order", 2))
        if not 0.0 <= self.global_prior_restart <= 1.0:
            raise ValueError("global_prior_restart must be in [0, 1]")
        if self.global_prior_order < 0:
            raise ValueError("global_prior_order must be non-negative")

        self.diffusion_add_self_loops = bool(
            cfg.model.get("diffusion_add_self_loops", True)
        )
        self.edge_weight_min = float(cfg.model.get("edge_weight_min", 0.1))
        self.edge_weight_temperature = float(
            cfg.model.get("edge_weight_temperature", 0.35)
        )
        self.eps = float(cfg.model.get("eps", 1e-8))
        self.metric_init_seed = int(cfg.model.get("metric_init_seed", 20260910))
        self.filter_rank = int(cfg.model.get("filter_rank", 4))
        self.multihop_anchor_alpha = float(
            cfg.model.get("multihop_anchor_alpha", 0.1)
        )
        edge_metric = str(cfg.model.get("edge_metric", "learned_diag_cos")).lower()
        state_mode = str(cfg.model.get("multihop_state", "anchored")).lower()
        response_mode = str(cfg.model.get("multihop_response", "cumulative")).lower()
        fusion = str(cfg.model.get("fusion", "concat_residual_mlp")).lower()
        interaction_layers = int(cfg.model.get("hop_interaction_layers", 1))
        interaction_heads = int(cfg.model.get("hop_interaction_heads", 1))
        interaction_dropout = float(cfg.model.get("hop_interaction_dropout", 0.0))
        order_embedding = bool(
            cfg.model.get("hop_interaction_order_embedding", True)
        )
        self.relation_descriptor_eps = float(
            cfg.model.get("relation_descriptor_eps", 1e-6)
        )
        self.relation_bias_init = float(cfg.model.get("relation_bias_init", 0.10))
        self.relation_init_seed = int(cfg.model.get("relation_init_seed", 20260921))
        self.relation_init_std = float(cfg.model.get("relation_init_std", 0.02))
        self.hop_interaction_gate_init = float(
            cfg.model.get("hop_interaction_gate_init", 0.10)
        )
        if not 0.0 <= self.edge_weight_min <= 1.0:
            raise ValueError("edge_weight_min must be in [0, 1]")
        if self.edge_weight_temperature <= 0.0 or self.eps <= 0.0:
            raise ValueError("edge_weight_temperature and eps must be positive")
        if self.filter_rank < 1:
            raise ValueError("filter_rank must be positive")
        if not 0.0 <= self.multihop_anchor_alpha < 1.0:
            raise ValueError("multihop_anchor_alpha must satisfy 0 <= alpha < 1")
        if self.relation_descriptor_eps <= 0.0 or self.relation_init_std <= 0.0:
            raise ValueError("relation descriptor epsilon and initialization std must be positive")
        if edge_metric != "learned_diag_cos":
            raise ValueError("edge_metric is fixed to learned_diag_cos")
        if state_mode != "anchored" or response_mode != "cumulative":
            raise ValueError("multi-hop state and response are fixed to anchored/cumulative")
        if fusion != "concat_residual_mlp":
            raise ValueError("fusion is fixed to concat_residual_mlp")
        if interaction_layers != 1 or interaction_heads != 1 or interaction_dropout != 0.0:
            raise ValueError("cross-order interaction is fixed to one head/layer with zero dropout")
        if not order_embedding:
            raise ValueError("cross-order interaction requires order embeddings")

        self.text_proj = ProjectionMLP(self.text_dim, hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, hidden_dim, dropout, norm)

        metric_one = torch.tensor(1.0, dtype=torch.float32)
        metric_theta = torch.log(torch.expm1(metric_one))
        self.metric_theta_text = nn.Parameter(metric_theta.expand(hidden_dim).clone())
        self.metric_theta_visual = nn.Parameter(metric_theta.expand(hidden_dim).clone())

        initial_prior = self._make_global_prior()
        prior_in_model_basis = self._monomial_to_anchored_cumulative(
            initial_prior, self.multihop_anchor_alpha
        )
        self.gamma_global = nn.Parameter(prior_in_model_basis)
        self.delta_gamma_text = nn.Parameter(torch.zeros(self.max_order + 1))
        self.delta_gamma_visual = nn.Parameter(torch.zeros(self.max_order + 1))

        self.node_proj_text = nn.ModuleList(
            nn.Linear(hidden_dim, self.filter_rank)
            for _ in range(self.max_order + 1)
        )
        self.node_proj_visual = nn.ModuleList(
            nn.Linear(hidden_dim, self.filter_rank)
            for _ in range(self.max_order + 1)
        )
        self.node_vector_text = nn.Parameter(
            torch.zeros(self.max_order + 1, self.filter_rank)
        )
        self.node_vector_visual = nn.Parameter(
            torch.zeros(self.max_order + 1, self.filter_rank)
        )

        self.text_refine_mlp = _make_mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.visual_refine_mlp = _make_mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.text_refine_norm = nn.LayerNorm(hidden_dim)
        self.visual_refine_norm = nn.LayerNorm(hidden_dim)
        self.fusion_skip = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fusion_mlp = _make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

        order_count = self.max_order + 1
        self.hop_order_embedding_text = nn.Parameter(
            torch.zeros(order_count, hidden_dim)
        )
        self.hop_order_embedding_visual = nn.Parameter(
            torch.zeros(order_count, hidden_dim)
        )
        self.hop_layers_text = nn.ModuleList(
            [CrossOrderAttentionBlock(hidden_dim)]
        )
        self.hop_layers_visual = nn.ModuleList(
            [CrossOrderAttentionBlock(hidden_dim)]
        )
        self.theta_hop_gate_text = nn.Parameter(
            torch.tensor(_inverse_tanh(self.hop_interaction_gate_init))
        )
        self.theta_hop_gate_visual = nn.Parameter(
            torch.tensor(_inverse_tanh(self.hop_interaction_gate_init))
        )

        for offset, modality in enumerate(("text", "visual")):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.relation_init_seed + offset)
            initial_order_profile = torch.randn(
                order_count, generator=generator, dtype=torch.float32
            ) * self.relation_init_std
            setattr(
                self,
                f"relation_beta_raw_{modality}",
                nn.Parameter(initial_order_profile),
            )
            setattr(
                self,
                f"theta_relation_scale_{modality}",
                nn.Parameter(torch.tensor(_logit(self.relation_bias_init))),
            )

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

    @staticmethod
    def _monomial_to_anchored_cumulative(
        coefficients: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        count = int(coefficients.numel())
        matrix = torch.zeros(
            (count, count), dtype=coefficients.dtype, device=coefficients.device
        )
        retention = 1.0 - float(alpha)
        matrix[0, 0] = 1.0
        for order in range(1, count):
            matrix[order, order] = retention**order
            powers = torch.arange(order, dtype=coefficients.dtype, device=coefficients.device)
            matrix[order, :order] = float(alpha) * retention**powers
        return torch.linalg.solve_triangular(
            matrix.transpose(0, 1), coefficients.unsqueeze(-1), upper=True
        ).squeeze(-1)

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

    def _normalized_metric_weights(self, theta: torch.Tensor) -> torch.Tensor:
        raw_weights = F.softplus(theta)
        if raw_weights.dim() == 1:
            raw_weights = raw_weights.unsqueeze(0)
        return raw_weights / (raw_weights.mean(dim=-1, keepdim=True) + self.eps)

    def _edge_cosine_values_with_weights(
        self,
        h: torch.Tensor,
        metric_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if metric_weights.dim() == 1:
            metric_weights = metric_weights.unsqueeze(0)
        if edge_index.numel() == 0:
            return h.new_empty((metric_weights.size(0), 0))
        src, dst = edge_index
        chunk_size = 65536
        scores = []
        for perspective in range(metric_weights.size(0)):
            weighted_hidden = h * metric_weights[perspective].unsqueeze(0)
            chunks = []
            for start in range(0, edge_index.size(1), chunk_size):
                stop = min(start + chunk_size, edge_index.size(1))
                chunk_src = src[start:stop]
                chunk_dst = dst[start:stop]

                def score_chunk(
                    weighted: torch.Tensor,
                    *,
                    chunk_src: torch.Tensor = chunk_src,
                    chunk_dst: torch.Tensor = chunk_dst,
                ) -> torch.Tensor:
                    return F.cosine_similarity(
                        weighted[chunk_src], weighted[chunk_dst], dim=-1, eps=self.eps
                    )

                if torch.is_grad_enabled() and weighted_hidden.requires_grad:
                    score = checkpoint(score_chunk, weighted_hidden, use_reentrant=False)
                else:
                    score = score_chunk(weighted_hidden)
                chunks.append(
                    torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
                        -1.0, 1.0
                    )
                )
            scores.append(torch.cat(chunks, dim=0))
        return torch.stack(scores, dim=0)

    def _edge_weight_from_cosine(self, cosine: torch.Tensor) -> torch.Tensor:
        weight = self.edge_weight_min + (1.0 - self.edge_weight_min) * torch.sigmoid(
            cosine / self.edge_weight_temperature
        )
        return torch.nan_to_num(
            weight, nan=1.0, posinf=1.0, neginf=self.edge_weight_min
        ).clamp(self.edge_weight_min, 1.0)

    def _relation_calibration(
        self, h_text: torch.Tensor, h_visual: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        metric_text = self._normalized_metric_weights(self.metric_theta_text).to(
            device=h_text.device, dtype=h_text.dtype
        )
        metric_visual = self._normalized_metric_weights(self.metric_theta_visual).to(
            device=h_visual.device, dtype=h_visual.dtype
        )
        cosine_text = self._edge_cosine_values_with_weights(
            h_text, metric_text, edge_index
        ).mean(dim=0)
        cosine_visual = self._edge_cosine_values_with_weights(
            h_visual, metric_visual, edge_index
        ).mean(dim=0)
        weight_text = self._edge_weight_from_cosine(cosine_text)
        weight_visual = self._edge_weight_from_cosine(cosine_visual)
        return {
            "semantic_cosine_text": cosine_text,
            "semantic_cosine_visual": cosine_visual,
            "relation_weight_text": weight_text,
            "relation_weight_visual": weight_visual,
            "metric_weights_text": metric_text,
            "metric_weights_visual": metric_visual,
        }

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

    def _multi_hop_states(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        states = [h0]
        current = h0
        for _ in range(self.max_order):
            propagated = self._propagate_once(current, edge_index, edge_weight)
            current = (
                (1.0 - self.multihop_anchor_alpha) * propagated
                + self.multihop_anchor_alpha * h0
            )
            states.append(current)
        return states

    @staticmethod
    def _incident_mean(
        edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        device = edge_weight.device
        dtype = edge_weight.dtype
        endpoint = torch.cat((edge_index[0], edge_index[1]), dim=0)
        incident_weight = torch.cat((edge_weight, edge_weight), dim=0)
        count = torch.zeros(num_nodes, dtype=dtype, device=device)
        first = torch.zeros_like(count)
        if endpoint.numel():
            count.index_add_(0, endpoint, torch.ones_like(incident_weight))
            first.index_add_(0, endpoint, incident_weight)
        mean = first / count.clamp_min(1.0)
        return torch.where(count > 0, mean, torch.zeros_like(mean))

    @staticmethod
    def _incident_moments(
        edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute population incident moments for analysis exports."""
        device = edge_weight.device
        dtype = edge_weight.dtype
        endpoint = torch.cat((edge_index[0], edge_index[1]), dim=0)
        incident_weight = torch.cat((edge_weight, edge_weight), dim=0)
        count = torch.zeros(num_nodes, dtype=dtype, device=device)
        first = torch.zeros_like(count)
        second = torch.zeros_like(count)
        if endpoint.numel():
            count.index_add_(0, endpoint, torch.ones_like(incident_weight))
            first.index_add_(0, endpoint, incident_weight)
            second.index_add_(0, endpoint, incident_weight.square())
        safe_count = count.clamp_min(1.0)
        mean = first / safe_count
        variance = (second / safe_count - mean.square()).clamp_min(0.0)
        std = variance.sqrt()
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        std = torch.where(count > 0, std, torch.zeros_like(std))
        return mean, std

    def _local_relation_context(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mean = self._incident_mean(edge_index, edge_weight.detach(), num_nodes)
            # Preserve the frozen first-column reduction layout without
            # computing or using an incident-variance descriptor.
            reduction_input = torch.stack((mean, torch.zeros_like(mean)), dim=-1)
            graph_mean = reduction_input.mean(dim=0, keepdim=True)
            graph_std = reduction_input.std(dim=0, unbiased=False, keepdim=True)
            local_context = (
                (reduction_input - graph_mean)
                / (graph_std + self.relation_descriptor_eps)
            )[:, 0]
        return local_context.detach(), mean.detach()

    def _relation_to_order_profile(
        self, modality: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = getattr(self, f"relation_beta_raw_{modality}")
        centered = raw - raw.mean()
        beta_hat = centered / centered.square().mean().sqrt().clamp_min(
            self.relation_descriptor_eps
        )
        scale = torch.sigmoid(getattr(self, f"theta_relation_scale_{modality}"))
        return beta_hat, scale

    def _cross_order_interaction(
        self,
        states: list[torch.Tensor],
        modality: str,
        local_context: torch.Tensor,
        *,
        relation_intervention: str,
        interaction_intervention: str,
        capture_attention: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        state_bank = torch.stack(states, dim=1)
        if modality == "text":
            order_embedding = self.hop_order_embedding_text
            layer = self.hop_layers_text[0]
            gate = torch.tanh(self.theta_hop_gate_text)
        else:
            order_embedding = self.hop_order_embedding_visual
            layer = self.hop_layers_visual[0]
            gate = torch.tanh(self.theta_hop_gate_visual)

        beta_hat, relation_scale = self._relation_to_order_profile(modality)
        relation_order_bias = (
            relation_scale
            * local_context.detach().to(dtype=state_bank.dtype).unsqueeze(-1)
            * beta_hat.to(dtype=state_bank.dtype).unsqueeze(0)
        )
        if relation_intervention == "off":
            relation_bias_for_attention = None
        else:
            relation_bias_for_attention = relation_order_bias
        if interaction_intervention == "off":
            gate = torch.zeros_like(gate)

        tokens = state_bank + order_embedding.unsqueeze(0)
        normalized = layer.norm(tokens)
        query = layer.query(normalized)
        key = layer.key(normalized)
        value = layer.value(normalized)
        logits = torch.matmul(query, key.transpose(-1, -2)) * layer.scale
        if relation_bias_for_attention is not None:
            logits = logits + relation_bias_for_attention.unsqueeze(1)
        attention = torch.softmax(logits, dim=-1)
        if interaction_intervention == "uniform":
            attention = torch.full_like(attention, 1.0 / float(attention.size(-1)))
        elif interaction_intervention == "query_collapse":
            key_mass = attention.mean(dim=1, keepdim=True)
            attention = key_mass.expand_as(attention)
        interaction = torch.matmul(layer.attention_dropout(attention), value)
        interacted_bank = state_bank + gate * interaction
        interacted = [
            interacted_bank[:, order, :] for order in range(interacted_bank.size(1))
        ]
        return interacted, attention if capture_attention else None

    def _node_preference(
        self, interacted_states: list[torch.Tensor], modality: str
    ) -> torch.Tensor:
        projectors = getattr(self, f"node_proj_{modality}")
        node_vector = getattr(self, f"node_vector_{modality}")
        responses = []
        for order, state in enumerate(interacted_states):
            projected = torch.tanh(projectors[order](state))
            responses.append(
                (projected * node_vector[order]).sum(dim=-1)
                / float(self.filter_rank)
            )
        return torch.stack(responses, dim=-1)

    @staticmethod
    def _compose(states: list[torch.Tensor], eta: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(states[0])
        for order, state in enumerate(states):
            output = output + eta[:, order : order + 1] * state
        return output

    @staticmethod
    def _effective_order(eta: torch.Tensor) -> torch.Tensor:
        orders = torch.arange(eta.size(-1), device=eta.device, dtype=eta.dtype)
        weights = eta.abs()
        return (weights * orders.unsqueeze(0)).sum(dim=-1) / weights.sum(
            dim=-1
        ).clamp_min(torch.finfo(eta.dtype).eps)

    def _encode_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        relation_intervention: str = "normal",
        interaction_intervention: str = "normal",
        relation_permutation: torch.Tensor | None = None,
        capture_attention: bool = False,
    ) -> dict[str, Any]:
        x_text, x_visual = self._split_features(x)
        h_text = self.text_proj(x_text)
        h_visual = self.visual_proj(x_visual)
        calibrated = self._relation_calibration(h_text, h_visual, edge_index)
        norm_text_index, norm_text_weight = self._normalized_operator(
            edge_index,
            calibrated["relation_weight_text"],
            int(x.size(0)),
            h_text.dtype,
        )
        norm_visual_index, norm_visual_weight = self._normalized_operator(
            edge_index,
            calibrated["relation_weight_visual"],
            int(x.size(0)),
            h_visual.dtype,
        )
        states_text = self._multi_hop_states(
            h_text, norm_text_index, norm_text_weight
        )
        states_visual = self._multi_hop_states(
            h_visual, norm_visual_index, norm_visual_weight
        )
        relation_context_text, incident_mean_text = (
            self._local_relation_context(
                edge_index,
                calibrated["relation_weight_text"],
                int(x.size(0)),
            )
        )
        relation_context_visual, incident_mean_visual = (
            self._local_relation_context(
                edge_index,
                calibrated["relation_weight_visual"],
                int(x.size(0)),
            )
        )
        if relation_permutation is not None:
            relation_context_text = relation_context_text.index_select(
                0, relation_permutation
            )
            relation_context_visual = relation_context_visual.index_select(
                0, relation_permutation
            )

        interacted_text, attention_text = self._cross_order_interaction(
            states_text,
            "text",
            relation_context_text,
            relation_intervention=relation_intervention,
            interaction_intervention=interaction_intervention,
            capture_attention=capture_attention,
        )
        interacted_visual, attention_visual = self._cross_order_interaction(
            states_visual,
            "visual",
            relation_context_visual,
            relation_intervention=relation_intervention,
            interaction_intervention=interaction_intervention,
            capture_attention=capture_attention,
        )
        delta_text = self._node_preference(interacted_text, "text")
        delta_visual = self._node_preference(interacted_visual, "visual")
        eta_text = (
            self.gamma_global.unsqueeze(0)
            + self.delta_gamma_text.unsqueeze(0)
            + delta_text
        )
        eta_visual = (
            self.gamma_global.unsqueeze(0)
            + self.delta_gamma_visual.unsqueeze(0)
            + delta_visual
        )
        z_text = self._compose(states_text, eta_text)
        z_visual = self._compose(states_visual, eta_visual)
        z_text_refined = self.text_refine_norm(z_text + self.text_refine_mlp(z_text))
        z_visual_refined = self.visual_refine_norm(
            z_visual + self.visual_refine_mlp(z_visual)
        )
        fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
        z = self.output_norm(
            self.fusion_skip(fused_input) + self.fusion_mlp(fused_input)
        )

        beta_hat_text, relation_scale_text = self._relation_to_order_profile("text")
        beta_hat_visual, relation_scale_visual = self._relation_to_order_profile("visual")
        output: dict[str, Any] = {
            "h0_text": h_text,
            "h0_visual": h_visual,
            **calibrated,
            "physical_edge_index": edge_index,
            "normalized_edge_index_text": norm_text_index,
            "normalized_edge_weight_text": norm_text_weight,
            "normalized_edge_index_visual": norm_visual_index,
            "normalized_edge_weight_visual": norm_visual_weight,
            "states_text": states_text,
            "states_visual": states_visual,
            "local_relation_context_text": relation_context_text,
            "local_relation_context_visual": relation_context_visual,
            "incident_relation_mean_text": incident_mean_text,
            "incident_relation_mean_visual": incident_mean_visual,
            "beta_hat_text": beta_hat_text,
            "beta_hat_visual": beta_hat_visual,
            "relation_scale_text": relation_scale_text,
            "relation_scale_visual": relation_scale_visual,
            "hop_gate_text": torch.tanh(self.theta_hop_gate_text),
            "hop_gate_visual": torch.tanh(self.theta_hop_gate_visual),
            "interacted_states_text": interacted_text,
            "interacted_states_visual": interacted_visual,
            "delta_text": delta_text,
            "delta_visual": delta_visual,
            "eta_text": eta_text,
            "eta_visual": eta_visual,
            "effective_order_text": self._effective_order(eta_text),
            "effective_order_visual": self._effective_order(eta_visual),
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": z,
        }
        if capture_attention:
            output["attention_text"] = attention_text
            output["attention_visual"] = attention_visual
        return output

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
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        edge_index = edge_index.to(device) if edge_index is not None else None
        z, _, _, _, _ = self(x.to(device), edge_index)
        return z.detach().cpu()

    def _analysis_call(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        relation_intervention: str = "normal",
        interaction_intervention: str = "normal",
        permutation_seed: int = 0,
    ) -> dict[str, Any]:
        relation_intervention = str(relation_intervention).strip().lower()
        interaction_intervention = str(interaction_intervention).strip().lower()
        if relation_intervention not in {"normal", "off", "shuffle"}:
            raise ValueError("relation_intervention must be normal|off|shuffle")
        if interaction_intervention not in {
            "normal",
            "query_collapse",
            "uniform",
            "off",
        }:
            raise ValueError(
                "interaction_intervention must be normal|query_collapse|uniform|off"
            )
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        permutation = None
        if relation_intervention == "shuffle":
            generator = torch.Generator(device="cpu").manual_seed(int(permutation_seed))
            permutation = torch.randperm(x.size(0), generator=generator).to(x.device)

        training_states = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self._encode_components(
                x,
                edge_index,
                relation_intervention=relation_intervention,
                interaction_intervention=interaction_intervention,
                relation_permutation=permutation,
                capture_attention=True,
            )
        finally:
            for module, was_training in training_states:
                module.training = was_training

    @torch.no_grad()
    def analysis_relation_calibration(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        components = self._analysis_call(x, edge_index)
        return {
            key: components[key].detach().clone()
            for key in (
                "h0_text",
                "h0_visual",
                "physical_edge_index",
                "semantic_cosine_text",
                "semantic_cosine_visual",
                "relation_weight_text",
                "relation_weight_visual",
                "metric_weights_text",
                "metric_weights_visual",
                "normalized_edge_index_text",
                "normalized_edge_weight_text",
                "normalized_edge_index_visual",
                "normalized_edge_weight_visual",
            )
        }

    @torch.no_grad()
    def analysis_multi_order(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, Any]:
        components = self._analysis_call(x, edge_index)
        tensor_keys = (
            "h0_text",
            "h0_visual",
            "local_relation_context_text",
            "local_relation_context_visual",
            "incident_relation_mean_text",
            "incident_relation_mean_visual",
            "beta_hat_text",
            "beta_hat_visual",
            "relation_scale_text",
            "relation_scale_visual",
            "hop_gate_text",
            "hop_gate_visual",
            "delta_text",
            "delta_visual",
            "eta_text",
            "eta_visual",
            "effective_order_text",
            "effective_order_visual",
            "attention_text",
            "attention_visual",
            "z_text",
            "z_visual",
            "z_text_refined",
            "z_visual_refined",
            "z",
        )
        result = {
            key: components[key].detach().clone()
            for key in tensor_keys
        }
        result["states_text"] = [value.detach().clone() for value in components["states_text"]]
        result["states_visual"] = [
            value.detach().clone() for value in components["states_visual"]
        ]
        result["interacted_states_text"] = [
            value.detach().clone() for value in components["interacted_states_text"]
        ]
        result["interacted_states_visual"] = [
            value.detach().clone() for value in components["interacted_states_visual"]
        ]
        _, relation_std_text = self._incident_moments(
            components["physical_edge_index"],
            components["relation_weight_text"],
            components["h0_text"].size(0),
        )
        _, relation_std_visual = self._incident_moments(
            components["physical_edge_index"],
            components["relation_weight_visual"],
            components["h0_visual"].size(0),
        )
        result["incident_relation_std_text"] = relation_std_text.detach().clone()
        result["incident_relation_std_visual"] = relation_std_visual.detach().clone()
        result["effective_order_summary_text"] = torch.stack(
            (
                result["effective_order_text"].mean(),
                result["effective_order_text"].std(unbiased=False),
            )
        )
        result["effective_order_summary_visual"] = torch.stack(
            (
                result["effective_order_visual"].mean(),
                result["effective_order_visual"].std(unbiased=False),
            )
        )
        return result

    @torch.no_grad()
    def analysis_intervention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        relation: str = "normal",
        interaction: str = "normal",
        permutation_seed: int = 0,
    ) -> dict[str, Any]:
        components = self._analysis_call(
            x,
            edge_index,
            relation_intervention=relation,
            interaction_intervention=interaction,
            permutation_seed=permutation_seed,
        )
        components["relation_intervention"] = relation
        components["interaction_intervention"] = interaction
        return components


Model = CoSIMAGFinal
