from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import add_self_loops, coalesce, remove_self_loops, to_undirected


class _Bottleneck(nn.Module):
    """Bottleneck operator copied from the frozen P0-R CRSA implementation."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(value)))


class _ContextualRelationEncoder(nn.Module):
    """Frozen P0-R relation-query block used by CRSA, with no diagnostic variants."""

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
        # Keep the known-safe general MHA path for full-graph edge batches.
        attended, _ = self.attention(query, evidence, evidence, need_weights=True)
        state = self.attention_norm(query + attended)
        state = self.ffn_norm(state + self.ffn(state))[:, 0]
        return self.router(state)


class _IATRModality(nn.Module):
    """Trajectory reconciliation for one modality; nodes are independent batches."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.state_projection = nn.Linear(256, 256)
        self.increment_projection = nn.Linear(256, 256)
        self.order_embedding = nn.Parameter(torch.empty(3, 256))
        nn.init.normal_(self.order_embedding, mean=0.0, std=0.02)
        self.token_norm = nn.LayerNorm(256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )

        self.query_projection = nn.Linear(256, 256)
        self.key_projection = nn.Linear(256, 256)
        self.value_projection = nn.Linear(256, 256)
        self.anchor_attention = nn.MultiheadAttention(
            256, 4, dropout=dropout, batch_first=True
        )
        self.anchor_dropout = nn.Dropout(dropout)
        self.anchor_norm = nn.LayerNorm(256)
        self.anchor_ffn = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(256)

    def forward(
        self,
        h0: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        return_trace: bool = False,
    ):
        states = torch.stack((c1, c2, c3), dim=1)
        increments = torch.stack((c1 - h0, c2 - c1, c3 - c2), dim=1)
        tokens = self.token_norm(
            self.state_projection(states)
            + self.increment_projection(increments)
            + self.order_embedding.unsqueeze(0)
        )
        trajectory = self.trajectory_encoder(tokens)

        query = self.query_projection(h0).unsqueeze(1)
        keys = self.key_projection(trajectory)
        values = self.value_projection(trajectory)
        readout, _ = self.anchor_attention(query, keys, values, need_weights=False)
        anchor = self.anchor_norm(h0 + self.anchor_dropout(readout[:, 0]))
        reconciled = self.output_norm(anchor + self.anchor_ffn(anchor))
        if return_trace:
            return reconciled, tokens, trajectory, anchor
        return reconciled


class Model(nn.Module):
    """Two-stage CRSA+IATR model for full-graph multimodal NC."""

    HIDDEN_DIM = 256
    MAX_ORDER = 3
    VARIANTS = {"base", "crsa", "iatr", "full"}
    requires_full_graph_training = True

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("crsa_iatr requires positive text_dim and visual_dim")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("crsa_iatr expects concatenated [text, visual] features")

        model_cfg = cfg.model
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.max_order = int(model_cfg.get("max_order", 3))
        self.dropout = float(model_cfg.get("dropout", 0.2))
        self.edge_chunk_size = int(model_cfg.get("edge_chunk_size", 65536))
        self.iatr_node_chunk_size = int(model_cfg.get("iatr_node_chunk_size", 4096))
        if self.hidden_dim != 256 or self.max_order != 3 or self.dropout != 0.2:
            raise ValueError("CRSA+IATR v1 fixes hidden_dim=256, max_order=3, dropout=0.2")
        if (int(model_cfg.get("relation_dim", 64)) != 64
                or int(model_cfg.get("relation_heads", 4)) != 4
                or int(model_cfg.get("relation_ffn_dim", 128)) != 128
                or int(model_cfg.get("shared_bottleneck_dim", 256)) != 256
                or int(model_cfg.get("adapter_bottleneck_dim", 32)) != 32
                or int(model_cfg.get("num_residual_adapters", 3)) != 3):
            raise ValueError("CRSA+IATR v1 fixes the inherited P0-R CRSA dimensions")
        if (int(model_cfg.get("trajectory_heads", 4)) != 4
                or int(model_cfg.get("trajectory_layers", 1)) != 1
                or int(model_cfg.get("trajectory_ffn_dim", 512)) != 512
                or int(model_cfg.get("anchor_heads", 4)) != 4
                or int(model_cfg.get("anchor_ffn_dim", 512)) != 512):
            raise ValueError("CRSA+IATR v1 fixes its IATR dimensions")
        if self.edge_chunk_size < 1 or self.iatr_node_chunk_size < 1:
            raise ValueError("edge and IATR node chunk sizes must be positive")

        configured_variant = model_cfg.get("variant", None)
        if configured_variant is not None and str(configured_variant).lower() not in self.VARIANTS:
            raise ValueError(f"unknown CRSA+IATR variant: {configured_variant}")
        self.use_crsa = bool(model_cfg.get("use_crsa", True))
        self.use_iatr = bool(model_cfg.get("use_iatr", True))
        if configured_variant is not None:
            expected = {
                "base": (False, False),
                "crsa": (True, False),
                "iatr": (False, True),
                "full": (True, True),
            }[str(configured_variant).lower()]
            if (self.use_crsa, self.use_iatr) != expected:
                raise ValueError("variant name conflicts with use_crsa/use_iatr flags")
            self.variant = str(configured_variant).lower()
        else:
            self.variant = {
                (False, False): "base",
                (True, False): "crsa",
                (False, True): "iatr",
                (True, True): "full",
            }[(self.use_crsa, self.use_iatr)]

        # Frozen P0/P0-R Text/Visual projectors and plain fusion.
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
        if self.use_crsa:
            # Independent RNG streams keep common backbone and classifier-head
            # initialization aligned across the 2x2 factor variants.
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

        self.iatr_text = self.iatr_visual = None
        if self.use_iatr:
            cpu_rng = torch.get_rng_state()
            torch.random.default_generator.manual_seed(seed + 29009)
            try:
                self.iatr_text = _IATRModality(self.dropout)
                self.iatr_visual = _IATRModality(self.dropout)
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
        # Exact physical-operator construction from the frozen relation_residual_audit R2.
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
            edge_index, normalized, (num_nodes, num_nodes),
            dtype=dtype, device=edge_index.device,
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

    def _prepare(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        if edge_index is None:
            raise ValueError("crsa_iatr requires a physical graph edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [N,{self.input_dim}], got {tuple(x.shape)}")
        edge_index = edge_index.to(x.device)
        operator = self._get_operator(edge_index, x.size(0), x.dtype)
        target, source, weight, self_target, self_source, self_weight = self._operator_edges(operator)
        h0_text = self.text_projector(x[:, :self.text_dim])
        h0_visual = self.visual_projector(x[:, self.text_dim:self.text_dim + self.visual_dim])
        structural = {"text": None, "visual": None}
        if self.use_crsa:
            neighbor_operator = self._neighbor_operator(operator)
            structural["text"] = torch.sparse.mm(neighbor_operator, h0_text) - h0_text
            structural["visual"] = torch.sparse.mm(neighbor_operator, h0_visual) - h0_visual
        return {
            "operator": operator, "target": target, "source": source, "weight": weight,
            "self_target": self_target, "self_source": self_source, "self_weight": self_weight,
            "h0_text": h0_text, "h0_visual": h0_visual, "structural": structural,
        }

    def _encode_routes(
        self, encoder: _ContextualRelationEncoder, h0: torch.Tensor,
        structural: torch.Tensor, target: torch.Tensor, source: torch.Tensor,
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
        # Route zero has no module and is exactly zero adaptation.
        return torch.sum(adapter_outputs * route[..., 1:, None], dim=1)

    def _propagate_one(
        self, prepared: dict, modality: str, routes: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        h0 = prepared[f"h0_{modality}"]
        states = [h0]
        target, source, edge_weight = prepared["target"], prepared["source"], prepared["weight"]
        for hop in range(1, 4):
            previous = states[-1]
            current = previous.new_zeros(previous.shape)
            current.index_add_(
                0, prepared["self_target"],
                previous[prepared["self_source"]] * prepared["self_weight"].unsqueeze(-1),
            )
            for start, end in self._chunks(target.numel()):
                t, s = target[start:end], source[start:end]
                source_state = previous[s]
                message = source_state
                if self.use_crsa:
                    shared = getattr(self, f"shared_{modality}")(source_state)
                    adapter_outputs = torch.stack(
                        [adapter(source_state) for adapter in getattr(self, f"adapters_{modality}")],
                        dim=1,
                    )
                    residual = self._mix_adapters(adapter_outputs, routes[start:end])
                    message = source_state + shared + residual
                current.index_add_(
                    0, t, message * edge_weight[start:end].unsqueeze(-1)
                )
            states.append(current)
        return states

    def _run_iatr(
        self, module: _IATRModality, states: list[torch.Tensor], *, return_trace: bool,
    ):
        h0, c1, c2, c3 = states
        output_parts = []
        trace_parts = {"tokens": [], "trajectory": [], "anchor": []}
        for start in range(0, h0.size(0), self.iatr_node_chunk_size):
            end = min(start + self.iatr_node_chunk_size, h0.size(0))
            args = (h0[start:end], c1[start:end], c2[start:end], c3[start:end])
            if return_trace:
                output, tokens, trajectory, anchor = module(*args, return_trace=True)
                trace_parts["tokens"].append(tokens)
                trace_parts["trajectory"].append(trajectory)
                trace_parts["anchor"].append(anchor)
            elif torch.is_grad_enabled() and h0.requires_grad:
                output = checkpoint(module, *args, use_reentrant=False)
            else:
                output = module(*args)
            output_parts.append(output)
        output = torch.cat(output_parts, dim=0) if output_parts else h0.new_empty((0, 256))
        if not return_trace:
            return output, None
        trace = {key: torch.cat(values, dim=0) for key, values in trace_parts.items()}
        return output, trace

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor, *, return_trace: bool):
        prepared = self._prepare(x, edge_index)
        routes = {}
        if self.use_crsa:
            for modality in ("text", "visual"):
                routes[modality] = self._encode_routes(
                    getattr(self, f"relation_{modality}"),
                    prepared[f"h0_{modality}"],
                    prepared["structural"][modality],
                    prepared["target"], prepared["source"],
                )
        else:
            routes = {"text": None, "visual": None}

        states_text = self._propagate_one(prepared, "text", routes["text"])
        states_visual = self._propagate_one(prepared, "visual", routes["visual"])
        traces = {"text": None, "visual": None}
        if self.use_iatr:
            z_text, traces["text"] = self._run_iatr(
                self.iatr_text, states_text, return_trace=return_trace
            )
            z_visual, traces["visual"] = self._run_iatr(
                self.iatr_visual, states_visual, return_trace=return_trace
            )
        else:
            z_text = torch.stack(states_text).mean(dim=0)
            z_visual = torch.stack(states_visual).mean(dim=0)
        fused = self.plain_fusion(torch.cat((z_text, z_visual), dim=-1))
        result = {
            "H0_text": states_text[0], "H0_visual": states_visual[0],
            "C_text": states_text, "C_visual": states_visual,
            "Z_text": z_text, "Z_visual": z_visual, "fused_z": fused,
            "D_text": prepared["structural"]["text"],
            "D_visual": prepared["structural"]["visual"],
            "routes_text": routes["text"], "routes_visual": routes["visual"],
        }
        if return_trace:
            result["trajectory_tokens_text"] = traces["text"]["tokens"] if traces["text"] else None
            result["trajectory_tokens_visual"] = traces["visual"]["tokens"] if traces["visual"] else None
            result["trajectory_encoded_text"] = traces["text"]["trajectory"] if traces["text"] else None
            result["trajectory_encoded_visual"] = traces["visual"]["trajectory"] if traces["visual"] else None
            result["anchor_text"] = traces["text"]["anchor"] if traces["text"] else None
            result["anchor_visual"] = traces["visual"]["anchor"] if traces["visual"] else None
        return result

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        return self._forward_impl(x, edge_index, return_trace=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        result = self._forward_impl(x, edge_index, return_trace=False)
        z = result["fused_z"]
        return z, None, None, z.new_zeros(()), {}

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
