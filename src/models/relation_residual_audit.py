from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops, to_undirected


class _Bottleneck(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(value)))


class _ContextualRelationEncoder(nn.Module):
    """One query attends to intrinsic and context evidence for a directed edge."""

    def __init__(
        self,
        hidden_dim: int = 256,
        relation_dim: int = 64,
        heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if relation_dim % heads:
            raise ValueError("relation_dim must be divisible by relation_heads")
        self.self_projection = nn.Linear(hidden_dim, relation_dim)
        self.context_projection = nn.Linear(hidden_dim, relation_dim)
        self.null_target = nn.Parameter(torch.zeros(hidden_dim))
        self.null_source = nn.Parameter(torch.zeros(hidden_dim))
        self.role_embedding = nn.Parameter(torch.empty(2, relation_dim))
        self.kind_embedding = nn.Parameter(torch.empty(2, relation_dim))
        self.relation_query = nn.Parameter(torch.empty(1, 1, relation_dim))
        nn.init.normal_(self.role_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.kind_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.relation_query, mean=0.0, std=0.02)

        self.evidence_norm = nn.LayerNorm(relation_dim)
        self.attention = nn.MultiheadAttention(
            relation_dim, heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(relation_dim)
        self.ffn = nn.Sequential(
            nn.Linear(relation_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, relation_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(relation_dim)
        self.router = nn.Linear(relation_dim, 4)
        nn.init.normal_(self.router.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.router.bias)
        with torch.no_grad():
            self.router.bias[0] = torch.log(torch.tensor(3.0))

    def forward(
        self,
        h0: torch.Tensor,
        context: torch.Tensor | None,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_target, h_source = h0[target], h0[source]
        role_target, role_source = self.role_embedding[0], self.role_embedding[1]
        kind_self, kind_context = self.kind_embedding[0], self.kind_embedding[1]

        token_target_self = self.self_projection(h_target) + role_target + kind_self
        token_source_self = self.self_projection(h_source) + role_source + kind_self
        if context is None:
            context_target = self.null_target.expand_as(h_target)
            context_source = self.null_source.expand_as(h_source)
        else:
            context_target = context[target] + self.null_target
            context_source = context[source] + self.null_source
        token_target_context = self.context_projection(context_target) + role_target + kind_context
        token_source_context = self.context_projection(context_source) + role_source + kind_context
        evidence = torch.stack(
            (token_target_self, token_source_self, token_target_context, token_source_context), dim=1
        )
        evidence = self.evidence_norm(evidence)
        query = self.relation_query.expand(h_target.size(0), -1, -1)
        # Request weights to select PyTorch's general MHA path; the fused SDPA
        # kernel rejects the large edge-batch configurations used by full graphs.
        attended, _ = self.attention(query, evidence, evidence, need_weights=True)
        state = self.attention_norm(query + attended)
        state = self.ffn_norm(state + self.ffn(state))[:, 0]
        return state, self.router(state)


class Model(nn.Module):
    """P0-R shared-prior and contextual residual relation adaptation.

    The normalized physical graph and three-hop uniform readout are fixed to the
    P0 relation-operator audit. Non-self edge routes are computed from H0 (and,
    for R2, the frozen-formula local structural response) once and reused at
    every hop. Only E x 4 routes persist across hops; evidence tokens are built
    chunk by chunk.
    """

    HIDDEN_DIM = 256
    MAX_ORDER = 3
    RELATION_DIM = 64
    ADAPTER_BOTTLENECK_DIM = 32
    VARIANTS = {"shared_prior", "global_residual", "attribute_residual", "context_residual"}
    requires_full_graph_training = True

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("relation_residual_audit requires positive text_dim and visual_dim")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("relation_residual_audit expects concatenated [text, visual] features")

        self.hidden_dim = int(cfg.model.get("hidden_dim", self.HIDDEN_DIM))
        self.max_order = int(cfg.model.get("max_order", self.MAX_ORDER))
        if self.hidden_dim != self.HIDDEN_DIM or self.max_order != self.MAX_ORDER:
            raise ValueError("P0-R fixes hidden_dim=256 and max_order=3")
        self.num_layers = self.MAX_ORDER
        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.relation_dim = int(cfg.model.get("relation_dim", self.RELATION_DIM))
        self.relation_heads = int(cfg.model.get("relation_heads", 4))
        self.relation_ffn_dim = int(cfg.model.get("relation_ffn_dim", 128))
        self.shared_bottleneck_dim = int(cfg.model.get("shared_bottleneck_dim", 256))
        self.adapter_bottleneck_dim = int(cfg.model.get("adapter_bottleneck_dim", 32))
        self.num_residual_adapters = int(cfg.model.get("num_residual_adapters", 3))
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 65536))
        if (self.dropout != 0.2 or self.relation_dim != self.RELATION_DIM
                or self.relation_heads != 4 or self.relation_ffn_dim != 128
                or self.shared_bottleneck_dim != self.HIDDEN_DIM
                or self.adapter_bottleneck_dim != self.ADAPTER_BOTTLENECK_DIM):
            raise ValueError(
                "P0-R fixes dropout=0.2, relation_dim=64, four heads, FFN=128, "
                "shared bottleneck=256, and adapter bottleneck=32"
            )
        if self.num_residual_adapters != 3:
            raise ValueError("P0-R fixes three residual adapters")
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.residual_variant = str(cfg.model.get("residual_variant", "shared_prior")).lower()
        if self.residual_variant not in self.VARIANTS:
            raise ValueError(f"unknown residual_variant={self.residual_variant!r}")
        self.uses_residual = self.residual_variant != "shared_prior"
        self.uses_context = self.residual_variant == "context_residual"

        # Match P0 Text/Visual projection and plain fusion exactly.
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.plain_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.shared_text = _Bottleneck(self.hidden_dim, self.shared_bottleneck_dim)
        self.shared_visual = _Bottleneck(self.hidden_dim, self.shared_bottleneck_dim)

        self.relation_text = None
        self.relation_visual = None
        self.adapters_text = None
        self.adapters_visual = None
        if self.uses_residual:
            self.relation_text = _ContextualRelationEncoder(
                self.hidden_dim, self.relation_dim, self.relation_heads,
                self.relation_ffn_dim, self.dropout,
            )
            self.relation_visual = _ContextualRelationEncoder(
                self.hidden_dim, self.relation_dim, self.relation_heads,
                self.relation_ffn_dim, self.dropout,
            )
            self.adapters_text = nn.ModuleList(
                _Bottleneck(self.hidden_dim, self.adapter_bottleneck_dim)
                for _ in range(self.num_residual_adapters)
            )
            self.adapters_visual = nn.ModuleList(
                _Bottleneck(self.hidden_dim, self.adapter_bottleneck_dim)
                for _ in range(self.num_residual_adapters)
            )
        self.out_dim = self.hidden_dim
        self._operator_cache_key = None
        self._operator_cache_edge_index = None
        self._operator_cache = None

    def _chunks(self, size: int) -> Iterator[tuple[int, int]]:
        for start in range(0, size, self.edge_chunk_size):
            yield start, min(start + self.edge_chunk_size, size)

    @staticmethod
    def _build_propagation_operator(
        edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
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
            edge_index, normalized, (num_nodes, num_nodes), dtype=dtype, device=edge_index.device
        ).coalesce()

    def _get_operator(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype):
        key = (
            edge_index.data_ptr(), int(getattr(edge_index, "_version", 0)),
            tuple(edge_index.shape), int(num_nodes), edge_index.device, dtype,
        )
        if key == self._operator_cache_key and self._operator_cache is not None:
            return self._operator_cache
        operator = self._build_propagation_operator(edge_index, num_nodes, dtype)
        self._operator_cache_key = key
        self._operator_cache_edge_index = edge_index
        self._operator_cache = operator
        return operator

    @staticmethod
    def _operator_edges(operator: torch.Tensor):
        indices, weights = operator.indices(), operator.values()
        target, source = indices[0], indices[1]
        self_mask = target == source
        return (
            target[~self_mask], source[~self_mask], weights[~self_mask],
            target[self_mask], source[self_mask], weights[self_mask],
        )

    @staticmethod
    def _neighbor_operator(operator: torch.Tensor) -> torch.Tensor:
        indices, weights = operator.indices(), operator.values()
        keep = indices[0] != indices[1]
        return torch.sparse_coo_tensor(
            indices[:, keep], weights[keep], operator.shape,
            dtype=operator.dtype, device=operator.device,
        ).coalesce()

    def _prepare_inputs(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        if edge_index is None:
            raise ValueError("relation_residual_audit requires physical edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [N,{self.input_dim}], got {tuple(x.shape)}")
        edge_index = edge_index.to(x.device)
        operator = self._get_operator(edge_index, x.size(0), x.dtype)
        target, source, weight, self_target, self_source, self_weight = self._operator_edges(operator)
        h0_text = self.text_projector(x[:, :self.text_dim])
        h0_visual = self.visual_projector(x[:, self.text_dim:self.text_dim + self.visual_dim])
        structural = {"text": None, "visual": None}
        if self.uses_context:
            neighbor_operator = self._neighbor_operator(operator)
            structural["text"] = torch.sparse.mm(neighbor_operator, h0_text) - h0_text
            structural["visual"] = torch.sparse.mm(neighbor_operator, h0_visual) - h0_visual
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
            "structural": structural,
        }

    def _encode_route_chunks(
        self,
        encoder: _ContextualRelationEncoder,
        h0: torch.Tensor,
        context: torch.Tensor | None,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        # The configured edge chunk bounds graph work; MHA uses a smaller cap
        # because GPU attention kernels have lower batch-grid limits.
        relation_chunk_size = min(self.edge_chunk_size, 8192)
        for start in range(0, target.numel(), relation_chunk_size):
            end = min(start + relation_chunk_size, target.numel())
            t, s = target[start:end], source[start:end]
            if context is None:
                def encode(h, edge_target, edge_source):
                    return encoder(h, None, edge_target, edge_source)[1]
                args = (h0, t, s)
            else:
                def encode(h, c, edge_target, edge_source):
                    return encoder(h, c, edge_target, edge_source)[1]
                args = (h0, context, t, s)
            if torch.is_grad_enabled() and h0.requires_grad:
                logits = checkpoint(encode, *args, use_reentrant=False)
            else:
                logits = encode(*args)
            parts.append(torch.softmax(logits, dim=-1))
        if parts:
            return torch.cat(parts, dim=0)
        return h0.new_empty((0, 4))

    @staticmethod
    def _shuffle_context(context: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(context.size(0), generator=generator).to(context.device)
        return context[permutation]

    @staticmethod
    def _mix_adapters(adapter_outputs: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        """Index zero has no module; only adapter routes 1..3 produce residuals."""
        if adapter_outputs.size(1) != 3 or routes.size(-1) != 4:
            raise ValueError("expected three adapters and four routes including no-adaptation")
        return torch.sum(adapter_outputs * routes[..., 1:, None], dim=1)

    def _routes_for_modality(
        self,
        modality: str,
        h0: torch.Tensor,
        structural: torch.Tensor | None,
        target: torch.Tensor,
        source: torch.Tensor,
        *,
        context_mode: str = "real",
        context_seed: int = 0,
        route_mode: str = "normal",
        shrinkage: float | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.uses_residual:
            return None, None
        if context_mode not in {"real", "null", "shuffle"}:
            raise ValueError(f"unknown context_mode={context_mode!r}")
        if route_mode not in {"normal", "global"}:
            raise ValueError(f"unknown route_mode={route_mode!r}")
        if shrinkage is not None and not 0.0 <= float(shrinkage) <= 1.0:
            raise ValueError("shrinkage must lie in [0,1]")

        context = structural if self.uses_context and context_mode == "real" else None
        if self.uses_context and context_mode == "shuffle" and structural is not None:
            context = self._shuffle_context(structural, context_seed)
        encoder = getattr(self, f"relation_{modality}")
        routes = self._encode_route_chunks(encoder, h0, context, target, source)
        if not routes.numel():
            return routes, routes
        global_route = routes.mean(dim=0, keepdim=True)
        if route_mode == "global" or (self.residual_variant == "global_residual"):
            used = global_route.expand_as(routes)
        elif shrinkage is not None:
            if float(shrinkage) == 1.0:
                used = routes
            elif float(shrinkage) == 0.0:
                used = global_route.expand_as(routes)
            else:
                used = global_route + float(shrinkage) * (routes - global_route)
        else:
            used = routes
        return used, routes

    def route_analysis(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        context_mode: str = "real",
        context_seed: int = 0,
        route_mode: str = "normal",
        shrinkage: float | None = None,
    ) -> dict:
        prepared = self._prepare_inputs(x, edge_index)
        routes = {}
        raw_routes = {}
        for modality in ("text", "visual"):
            used, raw = self._routes_for_modality(
                modality, prepared[f"h0_{modality}"], prepared["structural"][modality],
                prepared["target"], prepared["source"], context_mode=context_mode,
                context_seed=context_seed + (0 if modality == "text" else 1),
                route_mode=route_mode, shrinkage=shrinkage,
            )
            routes[modality] = used
            raw_routes[modality] = raw
        return {
            "H0_text": prepared["h0_text"], "H0_visual": prepared["h0_visual"],
            "D_text": prepared["structural"]["text"], "D_visual": prepared["structural"]["visual"],
            "routes_text": routes["text"], "routes_visual": routes["visual"],
            "raw_routes_text": raw_routes["text"], "raw_routes_visual": raw_routes["visual"],
            "target": prepared["target"], "source": prepared["source"],
        }

    @staticmethod
    def _distribution_stats(values: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        values = values.detach().float().reshape(-1)
        if not values.numel():
            zero = torch.zeros(())
            return {f"{prefix}_{name}": zero for name in ("mean", "median", "p10", "p90")}
        quantiles = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9]))
        return {
            f"{prefix}_mean": values.mean(),
            f"{prefix}_median": quantiles[1],
            f"{prefix}_p10": quantiles[0],
            f"{prefix}_p90": quantiles[2],
        }

    def _route_statistics(self, route: torch.Tensor | None, modality: str) -> dict[str, torch.Tensor]:
        if route is None or not route.numel():
            return {}
        route = route.detach()
        output = self._distribution_stats(route[:, 0], f"rr_{modality}_pi0")
        output[f"rr_{modality}_pi0_fraction_gt_05"] = (route[:, 0] > 0.5).float().mean()
        output[f"rr_{modality}_pi0_fraction_gt_08"] = (route[:, 0] > 0.8).float().mean()
        for adapter in range(1, 4):
            output[f"rr_{modality}_adapter{adapter}_route_mean"] = route[:, adapter].mean()
        entropy = -(route.clamp_min(1e-12) * route.clamp_min(1e-12).log()).sum(dim=-1)
        output[f"rr_{modality}_routing_entropy"] = entropy.mean()
        top1 = F.one_hot(route.argmax(dim=-1), num_classes=4).float().mean(dim=0)
        for index in range(4):
            output[f"rr_{modality}_top1_fraction{index}"] = top1[index]
        return output

    def _propagate_modality(
        self,
        prepared: dict,
        modality: str,
        route: torch.Tensor | None,
        *,
        shared_enabled: bool = True,
        residual_enabled: bool = True,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        h0 = prepared[f"h0_{modality}"]
        shared_module = getattr(self, f"shared_{modality}")
        adapters = getattr(self, f"adapters_{modality}")
        target, source, edge_weight = prepared["target"], prepared["source"], prepared["weight"]
        states = [h0]
        stats: dict[str, torch.Tensor] = self._route_statistics(route, modality)
        for hop in range(1, self.MAX_ORDER + 1):
            state = states[-1]
            output = state.new_zeros(state.shape)
            output.index_add_(
                0, prepared["self_target"],
                state[prepared["self_source"]] * prepared["self_weight"].unsqueeze(-1),
            )
            shared_ratios, residual_ratios = [], []
            cosine_parts = [[], [], []]
            for start, end in self._chunks(target.numel()):
                t, s, w = target[start:end], source[start:end], edge_weight[start:end]
                source_state = state[s]
                shared_delta = shared_module(source_state)
                shared_effect = shared_delta if shared_enabled else torch.zeros_like(shared_delta)
                if adapters is not None and route is not None:
                    adapter_outputs = torch.stack([adapter(source_state) for adapter in adapters], dim=1)
                    residual_value = self._mix_adapters(adapter_outputs, route[start:end])
                    if residual_enabled:
                        residual_effect = residual_value
                    else:
                        residual_effect = torch.zeros_like(residual_value)
                    for pair, (left, right) in enumerate(((0, 1), (0, 2), (1, 2))):
                        cosine_parts[pair].append(F.cosine_similarity(
                            adapter_outputs[:, left], adapter_outputs[:, right], dim=-1
                        ).detach())
                else:
                    adapter_outputs = None
                    residual_value = torch.zeros_like(source_state)
                    residual_effect = residual_value
                message = source_state + shared_effect + residual_effect
                output.index_add_(0, t, message * w.unsqueeze(-1))
                shared_ratios.append((shared_delta.norm(dim=-1) / (source_state.norm(dim=-1) + 1e-12)).detach())
                residual_ratios.append((residual_effect.norm(dim=-1) /
                                        ((source_state + shared_effect).norm(dim=-1) + 1e-12)).detach())
            states.append(output)
            shared_values = torch.cat(shared_ratios) if shared_ratios else h0.new_empty((0,))
            residual_values = torch.cat(residual_ratios) if residual_ratios else h0.new_empty((0,))
            stats.update(self._distribution_stats(shared_values, f"rr_{modality}_hop{hop}_rho_shared"))
            if self.uses_residual:
                stats.update(self._distribution_stats(residual_values, f"rr_{modality}_hop{hop}_rho_adapt"))
                if residual_values.numel():
                    stats[f"rr_{modality}_hop{hop}_rho_adapt_fraction_gt_05"] = (residual_values > 0.5).float().mean()
                    stats[f"rr_{modality}_hop{hop}_rho_adapt_fraction_gt_1"] = (residual_values > 1.0).float().mean()
                for pair, values in enumerate(cosine_parts):
                    if values:
                        stats[f"rr_{modality}_hop{hop}_adapter_cosine{pair}"] = torch.cat(values).mean()
        return states, stats

    def analyze(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        shared_enabled: bool = True,
        residual_enabled: bool = True,
        context_mode: str = "real",
        context_seed: int = 0,
        route_mode: str = "normal",
        shrinkage: float | None = None,
        shrinkage_text: float | None = None,
        shrinkage_visual: float | None = None,
    ) -> dict:
        prepared = self._prepare_inputs(x, edge_index)
        routes: dict[str, torch.Tensor | None] = {}
        raw_routes: dict[str, torch.Tensor | None] = {}
        for modality in ("text", "visual"):
            route, raw = self._routes_for_modality(
                modality,
                prepared[f"h0_{modality}"],
                prepared["structural"][modality],
                prepared["target"],
                prepared["source"],
                context_mode=context_mode,
                context_seed=context_seed + (0 if modality == "text" else 1),
                route_mode=route_mode,
                shrinkage=(shrinkage_text if modality == "text" and shrinkage_text is not None
                           else shrinkage_visual if modality == "visual" and shrinkage_visual is not None
                           else shrinkage),
            )
            routes[modality], raw_routes[modality] = route, raw
        states_text, stats_text = self._propagate_modality(
            prepared, "text", routes["text"],
            shared_enabled=shared_enabled, residual_enabled=residual_enabled,
        )
        states_visual, stats_visual = self._propagate_modality(
            prepared, "visual", routes["visual"],
            shared_enabled=shared_enabled, residual_enabled=residual_enabled,
        )
        z_text = torch.stack(states_text).mean(dim=0)
        z_visual = torch.stack(states_visual).mean(dim=0)
        fused = self.plain_fusion(torch.cat((z_text, z_visual), dim=-1))
        stats = {**stats_text, **stats_visual}
        return {
            "H0_text": prepared["h0_text"], "H0_visual": prepared["h0_visual"],
            "D_text": prepared["structural"]["text"], "D_visual": prepared["structural"]["visual"],
            "C_text": states_text, "C_visual": states_visual,
            "Z_text": z_text, "Z_visual": z_visual, "fused_z": fused,
            "routes_text": routes["text"], "routes_visual": routes["visual"],
            "raw_routes_text": raw_routes["text"], "raw_routes_visual": raw_routes["visual"],
            "mechanism_stats": stats,
            "num_nonself_edges": int(prepared["target"].numel()),
            "num_self_edges": int(x.size(0)),
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        analysis = self.analyze(x, edge_index)
        z = analysis["fused_z"]
        return z, None, None, z.new_zeros(()), analysis["mechanism_stats"]

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        return self(x.to(device), edge_index.to(device))[0].detach().cpu()
