from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops, to_undirected


class _Bottleneck(nn.Module):
    """The unchanged shared/adapted operator from the frozen CRSA path."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(value)))


class _ContextualRelationEncoder(nn.Module):
    """Unchanged P0-R contextual relation query used to compute CRSA routes."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.self_projection = nn.Linear(256, 64)
        self.context_projection = nn.Linear(256, 64)
        self.null_target = nn.Parameter(torch.zeros(256))
        self.null_source = nn.Parameter(torch.zeros(256))
        self.role_embedding = nn.Parameter(torch.empty(2, 64))
        self.kind_embedding = nn.Parameter(torch.empty(2, 64))
        self.relation_query = nn.Parameter(torch.empty(1, 1, 64))
        nn.init.normal_(self.role_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.kind_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.relation_query, mean=0.0, std=0.02)

        self.evidence_norm = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(64)
        self.ffn = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(64)
        self.router = nn.Linear(64, 4)
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.router.bias)
        with torch.no_grad():
            self.router.bias[0] = torch.log(torch.tensor(3.0))

    def forward(
        self,
        h0: torch.Tensor,
        structural: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        h_target, h_source = h0[target], h0[source]
        role_target, role_source = self.role_embedding[0], self.role_embedding[1]
        kind_self, kind_context = self.kind_embedding[0], self.kind_embedding[1]

        token_target_self = self.self_projection(h_target) + role_target + kind_self
        token_source_self = self.self_projection(h_source) + role_source + kind_self
        token_target_context = (
            self.context_projection(structural[target] + self.null_target)
            + role_target + kind_context
        )
        token_source_context = (
            self.context_projection(structural[source] + self.null_source)
            + role_source + kind_context
        )
        evidence = torch.stack(
            (token_target_self, token_source_self, token_target_context, token_source_context),
            dim=1,
        )
        evidence = self.evidence_norm(evidence)
        query = self.relation_query.expand(h_target.size(0), -1, -1)
        # Keep the same general MHA path and route probabilities as CRSA v1.
        attended, _ = self.attention(query, evidence, evidence, need_weights=True)
        state = self.attention_norm(query + attended)
        state = self.ffn_norm(state + self.ffn(state))[:, 0]
        return self.router(state)


class _ICSRModality(nn.Module):
    """One modality-local intrinsic-context interaction adapter shared over hops."""

    def __init__(self):
        super().__init__()
        self.down = nn.Linear(512, 32)
        self.up = nn.Linear(32, 256)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @staticmethod
    def descriptors(h0: torch.Tensor, context: torch.Tensor):
        intrinsic = F.layer_norm(h0, (256,))
        contextual = F.layer_norm(context, (256,))
        agreement = intrinsic * contextual
        deviation = contextual - intrinsic
        return agreement, deviation

    def forward(self, h0: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        agreement, deviation = self.descriptors(h0, context)
        interaction = torch.cat((agreement, deviation), dim=-1)
        return self.up(F.gelu(self.down(interaction)))


class Model(nn.Module):
    """Frozen CRSA followed by modality-local ICSR interaction residuals."""

    HIDDEN_DIM = 256
    MAX_ORDER = 3
    requires_full_graph_training = True

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("crsa_icsr requires positive text_dim and visual_dim")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("crsa_icsr expects concatenated [text, visual] features")

        model_cfg = cfg.model
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.max_order = int(model_cfg.get("max_order", 3))
        self.dropout = float(model_cfg.get("dropout", 0.2))
        self.edge_chunk_size = int(model_cfg.get("edge_chunk_size", 65536))
        self.icsr_node_chunk_size = int(model_cfg.get("icsr_node_chunk_size", 4096))
        self.icsr_bottleneck_dim = int(model_cfg.get("icsr_bottleneck_dim", 32))
        if self.hidden_dim != 256 or self.max_order != 3 or self.dropout != 0.2:
            raise ValueError("CRSA+ICSR fixes hidden_dim=256, max_order=3, dropout=0.2")
        fixed_crsa = {
            "relation_dim": 64,
            "relation_heads": 4,
            "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256,
            "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3,
        }
        if any(int(model_cfg.get(key, value)) != value for key, value in fixed_crsa.items()):
            raise ValueError("CRSA+ICSR fixes the inherited P0-R CRSA dimensions")
        if self.icsr_bottleneck_dim != 32:
            raise ValueError("ICSR fixes its bottleneck dimension at 32")
        if self.edge_chunk_size < 1 or self.icsr_node_chunk_size < 1:
            raise ValueError("edge and ICSR node chunk sizes must be positive")
        self.use_icsr = bool(model_cfg.get("use_icsr", True))

        # This common backbone construction order exactly matches frozen CRSA.
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(self.visual_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.plain_fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 256),
        )

        seed = int(cfg.get("seed", 0))
        self.shared_text = self.shared_visual = None
        self.relation_text = self.relation_visual = None
        self.adapters_text = self.adapters_visual = None
        cpu_rng = torch.get_rng_state()
        torch.random.default_generator.manual_seed(seed + 17011)
        try:
            self.shared_text = _Bottleneck(256, 256)
            self.shared_visual = _Bottleneck(256, 256)
            self.relation_text = _ContextualRelationEncoder(self.dropout)
            self.relation_visual = _ContextualRelationEncoder(self.dropout)
            self.adapters_text = nn.ModuleList(_Bottleneck(256, 32) for _ in range(3))
            self.adapters_visual = nn.ModuleList(_Bottleneck(256, 32) for _ in range(3))
        finally:
            torch.set_rng_state(cpu_rng)

        self.icsr_text = self.icsr_visual = None
        if self.use_icsr:
            cpu_rng = torch.get_rng_state()
            torch.random.default_generator.manual_seed(seed + 49021)
            try:
                self.icsr_text = _ICSRModality()
                self.icsr_visual = _ICSRModality()
            finally:
                torch.set_rng_state(cpu_rng)

        self.out_dim = 256
        self._operator_cache_key = None
        self._operator_cache = None

    def _chunks(self, size: int) -> Iterator[tuple[int, int]]:
        for start in range(0, size, self.edge_chunk_size):
            yield start, min(start + self.edge_chunk_size, size)

    @staticmethod
    def _build_propagation_operator(
        edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        # Exact physical-operator construction from the frozen CRSA implementation.
        edge_index = edge_index.long()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        target, source = edge_index
        values = torch.ones(target.numel(), dtype=dtype, device=edge_index.device)
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, target, values)
        inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
        normalized = inv_sqrt[target] * values * inv_sqrt[source]
        return torch.sparse_coo_tensor(
            edge_index,
            normalized,
            (num_nodes, num_nodes),
            dtype=dtype,
            device=edge_index.device,
        ).coalesce()

    def _get_operator(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
        key = (
            edge_index.data_ptr(),
            int(getattr(edge_index, "_version", 0)),
            tuple(edge_index.shape),
            int(num_nodes),
            edge_index.device,
            dtype,
        )
        if key == self._operator_cache_key and self._operator_cache is not None:
            return self._operator_cache
        operator = self._build_propagation_operator(edge_index, num_nodes, dtype)
        self._operator_cache_key = key
        self._operator_cache = operator
        return operator

    @staticmethod
    def _operator_edges(operator: torch.Tensor):
        indices, weights = operator.indices(), operator.values()
        target, source = indices[0], indices[1]
        self_mask = target == source
        return (
            target[~self_mask],
            source[~self_mask],
            weights[~self_mask],
            target[self_mask],
            source[self_mask],
            weights[self_mask],
        )

    @staticmethod
    def _neighbor_operator(operator: torch.Tensor) -> torch.Tensor:
        indices, weights = operator.indices(), operator.values()
        keep = indices[0] != indices[1]
        return torch.sparse_coo_tensor(
            indices[:, keep],
            weights[keep],
            operator.shape,
            dtype=operator.dtype,
            device=operator.device,
        ).coalesce()

    def _prepare(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        if edge_index is None:
            raise ValueError("crsa_icsr requires a physical graph edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [N,{self.input_dim}], got {tuple(x.shape)}")
        edge_index = edge_index.to(x.device)
        operator = self._get_operator(edge_index, x.size(0), x.dtype)
        target, source, weight, self_target, self_source, self_weight = self._operator_edges(operator)
        h0_text = self.text_projector(x[:, : self.text_dim])
        h0_visual = self.visual_projector(x[:, self.text_dim : self.text_dim + self.visual_dim])
        neighbor_operator = self._neighbor_operator(operator)
        structural_text = torch.sparse.mm(neighbor_operator, h0_text) - h0_text
        structural_visual = torch.sparse.mm(neighbor_operator, h0_visual) - h0_visual
        return {
            "operator": operator,
            "target": target,
            "source": source,
            "weight": weight,
            "self_target": self_target,
            "self_source": self_source,
            "self_weight": self_weight,
            "h0_text": h0_text,
            "h0_visual": h0_visual,
            "structural_text": structural_text,
            "structural_visual": structural_visual,
        }

    def _encode_routes(
        self,
        encoder: _ContextualRelationEncoder,
        h0: torch.Tensor,
        structural: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        chunk_size = min(self.edge_chunk_size, 8192)
        route_parts = []
        for start in range(0, target.numel(), chunk_size):
            end = min(start + chunk_size, target.numel())
            edge_target, edge_source = target[start:end], source[start:end]

            def encode(features, response, dst, src):
                return encoder(features, response, dst, src)

            args = (h0, structural, edge_target, edge_source)
            if torch.is_grad_enabled() and h0.requires_grad:
                logits = checkpoint(encode, *args, use_reentrant=False)
            else:
                logits = encode(*args)
            route_parts.append(torch.softmax(logits, dim=-1))
        if route_parts:
            return torch.cat(route_parts, dim=0)
        return h0.new_empty((0, 4))

    @staticmethod
    def _mix_adapters(adapter_outputs: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        # Route zero is the unchanged no-adaptation route.
        return torch.sum(adapter_outputs * route[..., 1:, None], dim=1)

    def _interaction_chunks(
        self, module: _ICSRModality, h0: torch.Tensor, context: torch.Tensor,
        *, return_descriptors: bool,
    ):
        residual_parts, agreement_parts, deviation_parts = [], [], []
        for start in range(0, h0.size(0), self.icsr_node_chunk_size):
            end = min(start + self.icsr_node_chunk_size, h0.size(0))
            h_part, c_part = h0[start:end], context[start:end]
            args = (h_part, c_part)
            if torch.is_grad_enabled() and any(value.requires_grad for value in args):
                residual = checkpoint(module, *args, use_reentrant=False)
            else:
                residual = module(*args)
            residual_parts.append(residual)
            if return_descriptors:
                agreement, deviation = module.descriptors(h_part, c_part)
                agreement_parts.append(agreement)
                deviation_parts.append(deviation)
        if not residual_parts:
            empty = h0.new_empty((0, 256))
            return empty, empty if return_descriptors else None, empty if return_descriptors else None
        residual = torch.cat(residual_parts, dim=0)
        if return_descriptors:
            return (residual, torch.cat(agreement_parts, dim=0),
                    torch.cat(deviation_parts, dim=0))
        return residual, None, None

    def _encode_modality(self, prepared: dict, modality: str, *, return_analysis: bool) -> dict:
        h0 = prepared[f"h0_{modality}"]
        routes = self._encode_routes(
            getattr(self, f"relation_{modality}"), h0,
            prepared[f"structural_{modality}"], prepared["target"], prepared["source"],
        )
        previous = h0
        state_sum = h0
        interaction_sum = h0.new_zeros(h0.shape)
        keep_rse = return_analysis
        states = [h0] if return_analysis else None
        rse_values, agreements, deviations, interactions = [], [], [], []
        target, source, edge_weight = prepared["target"], prepared["source"], prepared["weight"]

        for hop in range(1, 4):
            current = previous.new_zeros(previous.shape)
            current.index_add_(
                0, prepared["self_target"],
                previous[prepared["self_source"]] * prepared["self_weight"].unsqueeze(-1),
            )
            relation_effect = previous.new_zeros(previous.shape) if keep_rse else None
            for start, end in self._chunks(target.numel()):
                t, s = target[start:end], source[start:end]
                source_state = previous[s]
                route_chunk = routes[start:end]
                weight_chunk = edge_weight[start:end]

                if keep_rse:
                    def adapt_chunk(source_values, route_values, weights):
                        shared = getattr(self, f"shared_{modality}")(source_values)
                        adapter_outputs = torch.stack(
                            [adapter(source_values) for adapter in getattr(self, f"adapters_{modality}")],
                            dim=1,
                        )
                        residual = self._mix_adapters(adapter_outputs, route_values)
                        semantic_effect = shared + residual
                        message = source_values + shared + residual
                        edge_weight_column = weights.unsqueeze(-1)
                        return message * edge_weight_column, semantic_effect * edge_weight_column

                    if torch.is_grad_enabled() and (source_state.requires_grad or route_chunk.requires_grad):
                        weighted_message, weighted_effect = checkpoint(
                            adapt_chunk, source_state, route_chunk, weight_chunk,
                            use_reentrant=False,
                        )
                    else:
                        weighted_message, weighted_effect = adapt_chunk(
                            source_state, route_chunk, weight_chunk
                        )
                else:
                    def adapt_chunk(source_values, route_values, weights):
                        shared = getattr(self, f"shared_{modality}")(source_values)
                        adapter_outputs = torch.stack(
                            [adapter(source_values) for adapter in getattr(self, f"adapters_{modality}")],
                            dim=1,
                        )
                        residual = self._mix_adapters(adapter_outputs, route_values)
                        message = source_values + shared + residual
                        return message * weights.unsqueeze(-1)

                    if torch.is_grad_enabled() and (source_state.requires_grad or route_chunk.requires_grad):
                        weighted_message = checkpoint(
                            adapt_chunk, source_state, route_chunk, weight_chunk,
                            use_reentrant=False,
                        )
                    else:
                        weighted_message = adapt_chunk(source_state, route_chunk, weight_chunk)
                current.index_add_(0, t, weighted_message)
                if keep_rse:
                    relation_effect.index_add_(0, t, weighted_effect)

            state_sum = state_sum + current
            if self.use_icsr:
                residual, agreement, deviation = self._interaction_chunks(
                    getattr(self, f"icsr_{modality}"), h0, current,
                    return_descriptors=return_analysis,
                )
                interaction_sum = interaction_sum + residual
                if return_analysis:
                    interactions.append(residual)
                    agreements.append(agreement)
                    deviations.append(deviation)
            if return_analysis:
                states.append(current)
                rse_values.append(relation_effect)
            previous = current

        base = state_sum / 4.0
        if self.use_icsr:
            z = base + interaction_sum / 3.0
        else:
            z = base
        output = {"base": base, "z": z}
        if return_analysis:
            output.update(
                states=states, rse=rse_values, routes=routes,
                interactions=interactions, agreements=agreements, deviations=deviations,
            )
        return output

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor, *, return_analysis: bool):
        prepared = self._prepare(x, edge_index)
        text = self._encode_modality(prepared, "text", return_analysis=return_analysis)
        visual = self._encode_modality(prepared, "visual", return_analysis=return_analysis)
        fused = self.plain_fusion(torch.cat((text["z"], visual["z"]), dim=-1))
        result = {
            "H0_text": prepared["h0_text"], "H0_visual": prepared["h0_visual"],
            "Z_text": text["z"], "Z_visual": visual["z"],
            "base_text": text["base"], "base_visual": visual["base"],
            "fused_z": fused, "D_text": prepared["structural_text"],
            "D_visual": prepared["structural_visual"],
        }
        if return_analysis:
            for modality, encoded in (("text", text), ("visual", visual)):
                result[f"C_{modality}"] = encoded["states"]
                result[f"rse_{modality}"] = encoded["rse"]
                result[f"routes_{modality}"] = encoded["routes"]
                result[f"agreement_{modality}"] = encoded["agreements"]
                result[f"deviation_{modality}"] = encoded["deviations"]
                result[f"interaction_{modality}"] = encoded["interactions"]
        return result

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        """Return contextual states, ICSR descriptors/residuals, RSE diagnostics, and readouts."""
        return self._forward_impl(x, edge_index, return_analysis=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        result = self._forward_impl(x, edge_index, return_analysis=False)
        z = result["fused_z"]
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None,
        device: torch.device | None = None, batch_size: int = 4096,
    ) -> torch.Tensor:
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        return self(x.to(device), edge_index.to(device))[0].detach().cpu()
