from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    remove_self_loops,
    to_undirected,
)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _zero_last_layer(module: nn.Module) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


class _FlatTrajectory(nn.Module):
    """E3-sized per-state projections followed by a flat trajectory encoder."""

    def __init__(self, hidden_dim: int, latent_dim: int, context_dim: int):
        super().__init__()
        self.state_proj = nn.ModuleList(nn.Linear(hidden_dim, latent_dim) for _ in range(4))
        self.context = nn.Sequential(
            nn.Linear(4 * latent_dim, context_dim),
            nn.ReLU(),
        )

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        flat = torch.cat(
            [torch.relu(layer(state)) for layer, state in zip(self.state_proj, states)],
            dim=-1,
        )
        return self.context(flat)


class _TrajectoryTransformer(nn.Module):
    # Nodes are independent examples in this per-node sequence encoder. Chunking
    # avoids CUDA attention kernels exceeding grid limits on large NC graphs.
    NODE_CHUNK_SIZE = 8192

    def __init__(self, hidden_dim: int, context_dim: int):
        super().__init__()
        self.state_proj = nn.Linear(hidden_dim, 64)
        self.order_embedding = nn.Parameter(torch.zeros(4, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.pool_proj = nn.Sequential(nn.Linear(64, context_dim), nn.ReLU())

    def forward(self, states: list[torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([self.state_proj(state) for state in states], dim=1)
        tokens = tokens + self.order_embedding.unsqueeze(0)
        encoded = torch.cat(
            [self.encoder(chunk) for chunk in tokens.split(self.NODE_CHUNK_SIZE, dim=0)],
            dim=0,
        )
        return self.pool_proj(encoded.mean(dim=1))


class _RelationContext(nn.Module):
    """Encode directed physical edge pairs for routing context only."""

    RELATION_DIM = 32
    CONTEXT_DIM = 128

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.reduce = nn.Linear(hidden_dim, self.RELATION_DIM)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4 * self.RELATION_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, self.RELATION_DIM),
        )
        self.pool_proj = nn.Linear(2 * self.RELATION_DIM + 1, self.CONTEXT_DIM)
        self.gate = nn.Linear(self.CONTEXT_DIM, self.CONTEXT_DIM)

    def forward(
        self, h0: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index is None or edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("relation routing requires the unchanged physical edge_index")
        # PyG convention: edge_index[0] is the source j and edge_index[1] the
        # target i, so this encodes each directed physical edge j -> i.
        source, target = edge_index.long()
        u = self.reduce(h0)
        u_target, u_source = u[target], u[source]
        token = self.edge_encoder(
            torch.cat(
                [u_target, u_source, u_target - u_source, u_target * u_source],
                dim=-1,
            )
        )
        n = h0.size(0)
        total = h0.new_zeros((n, self.RELATION_DIM))
        square_total = h0.new_zeros((n, self.RELATION_DIM))
        degree = h0.new_zeros((n, 1))
        total.index_add_(0, target, token)
        square_total.index_add_(0, target, token.square())
        degree.index_add_(0, target, h0.new_ones((target.numel(), 1)))
        mean = total / degree.clamp_min(1.0)
        variance = (square_total / degree.clamp_min(1.0) - mean.square()).clamp_min(0.0)
        pooled = torch.cat([mean, variance, torch.log1p(degree)], dim=-1)
        return pooled, token

    def fuse(self, trajectory: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        relation = self.pool_proj(pooled)
        return trajectory + torch.sigmoid(self.gate(trajectory)) * relation


class Model(nn.Module):
    """Fixed-topology state-mixture model for the Adaptive Routing Audit.

    The graph operator is the same unit-weight, symmetrized physical operator
    used by CMRF C0. Edge-relation tokens are consumed only by a routing head;
    they never modify edge_index, edge weights, or propagation messages.
    """

    MAX_ORDER = 3
    MODES = {
        "uniform",
        "global_simplex",
        "node_flat_simplex",
        "hierarchical_flat_simplex",
        "trajectory",
        "relation",
        "capacity_control",
        "cross_relation",
    }
    requires_full_graph_training = True

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("routing_audit requires text and visual features")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("routing_audit expects concatenated [text, visual] features")

        self.hidden_dim = int(cfg.model.get("hidden_dim", 256))
        self.max_order = int(cfg.model.get("max_order", self.MAX_ORDER))
        if self.max_order != self.MAX_ORDER:
            raise ValueError("routing_audit fixes max_order=3")
        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.context_dim = int(cfg.model.get("context_dim", 128))
        if self.context_dim != 128:
            raise ValueError("routing_audit fixes context_dim=128")
        self.variant = str(cfg.model.get("routing_variant", "uniform")).lower()
        if self.variant not in self.MODES:
            raise ValueError(f"unknown routing_variant={self.variant!r}")

        # Keep E3 C0 parameter names and module shapes for checkpoint reuse.
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
        self.out_dim = self.hidden_dim

        has_global = self.variant in {
            "global_simplex",
            "hierarchical_flat_simplex",
            "trajectory",
            "relation",
            "capacity_control",
            "cross_relation",
        }
        self.global_text_logits = nn.Parameter(torch.zeros(4)) if has_global else None
        self.global_visual_logits = nn.Parameter(torch.zeros(4)) if has_global else None

        self.flat_text = self.flat_visual = None
        self.trajectory_text = self.trajectory_visual = None
        self.router_text = self.router_visual = None
        self.relation_text = self.relation_visual = None
        self.joint_relation = None
        self.joint_gate_text = self.joint_gate_visual = None
        self.capacity_text = self.capacity_visual = None

        if self.variant in {"node_flat_simplex", "hierarchical_flat_simplex"}:
            latent = int(cfg.model.get("latent_dim", 32))
            self.flat_text = _FlatTrajectory(self.hidden_dim, latent, self.context_dim)
            self.flat_visual = _FlatTrajectory(self.hidden_dim, latent, self.context_dim)
            self.router_text = _mlp(self.context_dim, 64, 4)
            self.router_visual = _mlp(self.context_dim, 64, 4)
        elif self.variant in {
            "trajectory",
            "relation",
            "capacity_control",
            "cross_relation",
        }:
            self.trajectory_text = _TrajectoryTransformer(self.hidden_dim, self.context_dim)
            self.trajectory_visual = _TrajectoryTransformer(self.hidden_dim, self.context_dim)
            self.router_text = _mlp(self.context_dim, 64, 4)
            self.router_visual = _mlp(self.context_dim, 64, 4)
            if self.variant in {"relation", "cross_relation"}:
                self.relation_text = _RelationContext(self.hidden_dim)
                self.relation_visual = _RelationContext(self.hidden_dim)
            if self.variant == "cross_relation":
                self.joint_relation = nn.Sequential(
                    nn.Linear(4 * _RelationContext.RELATION_DIM, 64),
                    nn.ReLU(),
                    nn.Linear(64, _RelationContext.RELATION_DIM),
                )
                self.joint_pool_proj = nn.Linear(65, self.context_dim)
                self.joint_gate_text = nn.Linear(2 * self.context_dim, 1)
                self.joint_gate_visual = nn.Linear(2 * self.context_dim, 1)
            if self.variant == "capacity_control":
                # Matches each own-relation branch within 1%; it reads only
                # the B1 trajectory context and has no edge-index input.
                self.capacity_text = _mlp(self.context_dim, 168, self.context_dim)
                self.capacity_visual = _mlp(self.context_dim, 168, self.context_dim)
                _zero_last_layer(self.capacity_text)
                _zero_last_layer(self.capacity_visual)

        for router in (self.router_text, self.router_visual):
            if router is not None:
                _zero_last_layer(router)

        self._operator_cache_key = None
        self._operator_cache = None
        self._operator_cache_edge_index = None

    def _build_propagation_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build P=D^-1/2(A+I)D^-1/2 with unit physical edge weights."""
        edge_index = edge_index.long()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        values = torch.ones(row.numel(), dtype=dtype, device=edge_index.device)
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, row, values)
        inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
        normalized = inv_sqrt[row] * values * inv_sqrt[col]
        return torch.sparse_coo_tensor(
            edge_index,
            normalized,
            (num_nodes, num_nodes),
            dtype=dtype,
            device=edge_index.device,
        ).coalesce()

    def _get_operator(self, edge_index, num_nodes, dtype):
        key = (
            edge_index.data_ptr(),
            int(getattr(edge_index, "_version", 0)),
            tuple(edge_index.shape),
            int(num_nodes),
            edge_index.device,
            dtype,
        )
        if self._operator_cache_key == key and self._operator_cache is not None:
            return self._operator_cache
        operator = self._build_propagation_operator(edge_index, num_nodes, dtype)
        self._operator_cache_key = key
        self._operator_cache_edge_index = edge_index
        self._operator_cache = operator
        return operator

    @staticmethod
    def _states(h0: torch.Tensor, operator: torch.Tensor) -> list[torch.Tensor]:
        states = [h0]
        for _ in range(3):
            states.append(torch.sparse.mm(operator, states[-1]))
        return states

    @staticmethod
    def mix_states(states: list[torch.Tensor], alpha: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(states, dim=1)
        return torch.sum(stacked * alpha.unsqueeze(-1), dim=1)

    @staticmethod
    def _pool_joint(tokens: torch.Tensor, target: torch.Tensor, num_nodes: int) -> torch.Tensor:
        dim = tokens.size(-1)
        total = tokens.new_zeros((num_nodes, dim))
        square_total = tokens.new_zeros((num_nodes, dim))
        degree = tokens.new_zeros((num_nodes, 1))
        total.index_add_(0, target, tokens)
        square_total.index_add_(0, target, tokens.square())
        degree.index_add_(0, target, tokens.new_ones((target.numel(), 1)))
        mean = total / degree.clamp_min(1.0)
        var = (square_total / degree.clamp_min(1.0) - mean.square()).clamp_min(0.0)
        return torch.cat([mean, var, torch.log1p(degree)], dim=-1)

    def _contexts(self, analysis, relation_text=None, relation_visual=None, joint=None):
        variant = self.variant
        if variant in {"uniform", "global_simplex"}:
            return None, None
        if variant in {"node_flat_simplex", "hierarchical_flat_simplex"}:
            return analysis["flat_context_text"], analysis["flat_context_visual"]
        text = analysis["trajectory_context_text"]
        visual = analysis["trajectory_context_visual"]
        if variant == "capacity_control":
            return text + self.capacity_text(text), visual + self.capacity_visual(visual)
        if variant in {"relation", "cross_relation"}:
            rel_t = analysis["relation_text"] if relation_text is None else relation_text
            rel_v = analysis["relation_visual"] if relation_visual is None else relation_visual
            ctx_t = self.relation_text.fuse(text, rel_t)
            ctx_v = self.relation_visual.fuse(visual, rel_v)
            if variant == "cross_relation":
                joint = analysis["joint_relation"] if joint is None else joint
                joint_ctx = self.joint_pool_proj(joint)
                rho_t = torch.sigmoid(
                    self.joint_gate_text(torch.cat([ctx_t, joint_ctx], dim=-1))
                )
                rho_v = torch.sigmoid(
                    self.joint_gate_visual(torch.cat([ctx_v, joint_ctx], dim=-1))
                )
                analysis["rho_text"] = rho_t
                analysis["rho_visual"] = rho_v
                ctx_t = ctx_t + rho_t * joint_ctx
                ctx_v = ctx_v + rho_v * joint_ctx
            return ctx_t, ctx_v
        return text, visual

    def _deltas(self, context_text, context_visual, n, device, dtype):
        if self.router_text is None:
            zero = torch.zeros((n, 4), device=device, dtype=dtype)
            return zero, zero
        return self.router_text(context_text), self.router_visual(context_visual)

    def reroute(
        self,
        analysis: dict,
        *,
        delta_text_override: torch.Tensor | None = None,
        delta_visual_override: torch.Tensor | None = None,
        global_text_override: torch.Tensor | None = None,
        global_visual_override: torch.Tensor | None = None,
        relation_text_override: torch.Tensor | None = None,
        relation_visual_override: torch.Tensor | None = None,
        joint_relation_override: torch.Tensor | None = None,
    ) -> dict:
        """Recompose routing while reusing frozen states and all shared layers."""
        context_text, context_visual = self._contexts(
            analysis,
            relation_text=relation_text_override,
            relation_visual=relation_visual_override,
            joint=joint_relation_override,
        )
        n = analysis["S_text"][0].size(0)
        delta_t, delta_v = self._deltas(
            context_text,
            context_visual,
            n,
            analysis["S_text"][0].device,
            analysis["S_text"][0].dtype,
        )
        if delta_text_override is not None:
            delta_t = delta_text_override
        if delta_visual_override is not None:
            delta_v = delta_visual_override
        base_t = (
            torch.zeros((4,), device=delta_t.device, dtype=delta_t.dtype)
            if self.global_text_logits is None
            else self.global_text_logits
        )
        base_v = (
            torch.zeros((4,), device=delta_v.device, dtype=delta_v.dtype)
            if self.global_visual_logits is None
            else self.global_visual_logits
        )
        if global_text_override is not None:
            base_t = global_text_override
        if global_visual_override is not None:
            base_v = global_visual_override
        alpha_t = torch.softmax(base_t.unsqueeze(0) + delta_t, dim=-1)
        alpha_v = torch.softmax(base_v.unsqueeze(0) + delta_v, dim=-1)
        z_t = self.mix_states(analysis["S_text"], alpha_t)
        z_v = self.mix_states(analysis["S_visual"], alpha_v)
        fused = self.plain_fusion(torch.cat([z_t, z_v], dim=-1))
        return {
            "delta_text": delta_t,
            "delta_visual": delta_v,
            "alpha_text": alpha_t,
            "alpha_visual": alpha_v,
            "Z_text": z_t,
            "Z_visual": z_v,
            "fused_z": fused,
            "context_text": context_text,
            "context_visual": context_visual,
        }

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        if edge_index is None:
            raise ValueError("routing_audit requires the physical edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [N,{self.input_dim}], got {tuple(x.shape)}")
        edge_input = edge_index.to(x.device)
        operator = self._get_operator(edge_input, x.size(0), x.dtype)
        h0_t = self.text_projector(x[:, : self.text_dim])
        h0_v = self.visual_projector(x[:, self.text_dim : self.text_dim + self.visual_dim])
        states_t = self._states(h0_t, operator)
        states_v = self._states(h0_v, operator)
        analysis = {
            "H0_text": h0_t,
            "H0_visual": h0_v,
            "S_text": states_t,
            "S_visual": states_v,
        }
        if self.variant in {"node_flat_simplex", "hierarchical_flat_simplex"}:
            analysis["flat_context_text"] = self.flat_text(states_t)
            analysis["flat_context_visual"] = self.flat_visual(states_v)
        elif self.variant in {
            "trajectory",
            "relation",
            "capacity_control",
            "cross_relation",
        }:
            analysis["trajectory_context_text"] = self.trajectory_text(states_t)
            analysis["trajectory_context_visual"] = self.trajectory_visual(states_v)
            if self.variant in {"relation", "cross_relation"}:
                relation_t, edge_t = self.relation_text(h0_t, edge_input)
                relation_v, edge_v = self.relation_visual(h0_v, edge_input)
                analysis["relation_text"] = relation_t
                analysis["relation_visual"] = relation_v
                if self.variant == "cross_relation":
                    source, target = edge_input.long()
                    joint_tokens = self.joint_relation(
                        torch.cat(
                            [edge_t, edge_v, edge_t - edge_v, edge_t * edge_v], dim=-1
                        )
                    )
                    analysis["joint_relation"] = self._pool_joint(
                        joint_tokens, target, x.size(0)
                    )
                    analysis["edge_index_reference"] = edge_input
        routed = self.reroute(analysis)
        analysis.update(routed)
        return analysis

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        fused = self.analyze(x, edge_index)["fused_z"]
        return fused, None, None, fused.new_zeros(()), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        self.eval()
        if edge_index is None:
            raise ValueError("routing_audit requires the physical edge_index")
        if device is None:
            device = next(self.parameters()).device
        return self(x.to(device), edge_index.to(device))[0].detach().cpu()
