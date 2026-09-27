from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    remove_self_loops,
    to_undirected,
)


class _RelationEncoder(nn.Module):
    """Encode intrinsic Text or Visual endpoint attributes for a physical edge."""

    def __init__(self, hidden_dim: int = 256, reduced_dim: int = 64):
        super().__init__()
        self.reduce = nn.Linear(hidden_dim, reduced_dim)
        self.encoder = nn.Sequential(
            nn.Linear(4 * reduced_dim, 128),
            nn.GELU(),
            nn.Linear(128, reduced_dim),
            nn.LayerNorm(reduced_dim),
        )
        self.reduced_dim = reduced_dim

    def forward_reduced(
        self, reduced: torch.Tensor, target: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        hi, hj = reduced[target], reduced[source]
        return self.encoder(torch.cat([hi, hj, hi - hj, hi * hj], dim=-1))


class _ScalarWeightHead(nn.Module):
    def __init__(self, relation_dim: int = 64, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.proj = nn.Linear(relation_dim, heads)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, relation: torch.Tensor) -> torch.Tensor:
        # 2*sigmoid(0) == 1, so this starts at the physical propagation baseline.
        return (2.0 * torch.sigmoid(self.proj(relation))).mean(dim=-1)


class _BottleneckExpert(nn.Module):
    def __init__(self, hidden_dim: int = 256, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(value)))


class Model(nn.Module):
    """P0 audit of semantic scalar weighting versus relation-conditioned operators.

    The topology and normalized physical coefficients are shared by every
    variant. Relation controls are computed once from H0 and reused across all
    three propagation steps. Per-edge controls are stored as E or E x R values;
    no E x R x hidden_dim tensor is retained.
    """

    MAX_ORDER = 3
    HIDDEN_DIM = 256
    RELATION_DIM = 64
    NUM_EXPERTS = 4
    VARIANTS = {"plain", "scalar_weight", "global_expert", "relation_expert"}
    requires_full_graph_training = True

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("relation_operator_audit requires positive text_dim and visual_dim")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("relation_operator_audit expects concatenated [text, visual] features")

        self.hidden_dim = int(cfg.model.get("hidden_dim", self.HIDDEN_DIM))
        if self.hidden_dim != self.HIDDEN_DIM:
            raise ValueError("relation_operator_audit fixes hidden_dim=256")
        self.max_order = int(cfg.model.get("max_order", self.MAX_ORDER))
        if self.max_order != self.MAX_ORDER:
            raise ValueError("relation_operator_audit fixes max_order=3")
        self.num_layers = self.MAX_ORDER
        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 65536))
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")
        self.operator_variant = str(cfg.model.get("operator_variant", "plain")).lower()
        if self.operator_variant not in self.VARIANTS:
            raise ValueError(f"unknown operator_variant={self.operator_variant!r}")

        # These modules intentionally match cmrf_probe's projectors and fusion.
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

        self.relation_text = None
        self.relation_visual = None
        self.scalar_head_text = None
        self.scalar_head_visual = None
        self.query_text = None
        self.query_visual = None
        self.prototypes_text = None
        self.prototypes_visual = None
        self.experts_text = None
        self.experts_visual = None

        if self.operator_variant != "plain":
            self.relation_text = _RelationEncoder(self.hidden_dim, self.RELATION_DIM)
            self.relation_visual = _RelationEncoder(self.hidden_dim, self.RELATION_DIM)
        if self.operator_variant == "scalar_weight":
            self.scalar_head_text = _ScalarWeightHead(self.RELATION_DIM)
            self.scalar_head_visual = _ScalarWeightHead(self.RELATION_DIM)
        elif self.operator_variant in {"global_expert", "relation_expert"}:
            # A2 and A3 instantiate precisely the same trainable modules.
            self.query_text = nn.Linear(self.RELATION_DIM, self.RELATION_DIM, bias=False)
            self.query_visual = nn.Linear(self.RELATION_DIM, self.RELATION_DIM, bias=False)
            self.prototypes_text = nn.Parameter(torch.empty(self.NUM_EXPERTS, self.RELATION_DIM))
            self.prototypes_visual = nn.Parameter(torch.empty(self.NUM_EXPERTS, self.RELATION_DIM))
            nn.init.normal_(self.prototypes_text, mean=0.0, std=0.02)
            nn.init.normal_(self.prototypes_visual, mean=0.0, std=0.02)
            self.experts_text = nn.ModuleList(
                _BottleneckExpert(self.hidden_dim, self.RELATION_DIM)
                for _ in range(self.NUM_EXPERTS)
            )
            self.experts_visual = nn.ModuleList(
                _BottleneckExpert(self.hidden_dim, self.RELATION_DIM)
                for _ in range(self.NUM_EXPERTS)
            )

        self._operator_cache_key = None
        self._operator_cache_edge_index = None
        self._operator_cache = None

    def _build_propagation_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """CMRF's unit-weight P=D^-1/2(A+I)D^-1/2, including canonicalization."""
        edge_index = edge_index.long()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        row, col = edge_index  # Sparse coordinate row=target, col=source.
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

    def _get_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
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
    def _operator_edges(operator: torch.Tensor):
        indices = operator.indices()
        weights = operator.values()
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

    def _chunks(self, size: int) -> Iterator[tuple[int, int]]:
        for start in range(0, size, self.edge_chunk_size):
            yield start, min(start + self.edge_chunk_size, size)

    @staticmethod
    def _route_from_relation(
        relation: torch.Tensor, query: nn.Linear, prototypes: torch.Tensor
    ) -> torch.Tensor:
        q = query(relation)
        logits = q @ prototypes.t() / (relation.size(-1) ** 0.5)
        return torch.softmax(logits, dim=-1)

    def _edge_controls(
        self,
        modality: str,
        h0: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        relation_state_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict]:
        if self.operator_variant == "plain":
            return None, {}
        encoder = getattr(self, f"relation_{modality}")
        reduced = encoder.reduce(h0)
        if relation_state_override is not None and relation_state_override.shape != (
            target.numel(), self.RELATION_DIM
        ):
            raise ValueError("relation_state_override has an invalid shape")

        if self.operator_variant == "scalar_weight":
            head = getattr(self, f"scalar_head_{modality}")
            scalar_parts = []
            for start, end in self._chunks(target.numel()):
                t, s = target[start:end], source[start:end]
                if relation_state_override is not None:
                    scalar_parts.append(head(relation_state_override[start:end]))
                    continue

                def encode_scalar(reduced_features, edge_target, edge_source):
                    relation = encoder.forward_reduced(reduced_features, edge_target, edge_source)
                    return head(relation)

                if torch.is_grad_enabled() and reduced.requires_grad:
                    part = checkpoint(encode_scalar, reduced, t, s, use_reentrant=False)
                else:
                    part = encode_scalar(reduced, t, s)
                scalar_parts.append(part)
            scalar = torch.cat(scalar_parts, dim=0) if scalar_parts else h0.new_empty((0,))
            info = {
                "scalar_weight_mean": scalar.detach().mean() if scalar.numel() else h0.new_zeros(()),
                "scalar_weight_median": scalar.detach().median() if scalar.numel() else h0.new_zeros(()),
            }
            return scalar, info

        query = getattr(self, f"query_{modality}")
        prototypes = getattr(self, f"prototypes_{modality}")
        if self.operator_variant == "global_expert":
            relation_sum = h0.new_zeros((self.RELATION_DIM,))
            if relation_state_override is not None:
                relation_sum = relation_state_override.sum(dim=0)
            else:
                # Pool deterministic chunk sums; never concatenate all E x 64 states.
                for start, end in self._chunks(target.numel()):
                    t, s = target[start:end], source[start:end]

                    def encode_relation(reduced_features, edge_target, edge_source):
                        return encoder.forward_reduced(reduced_features, edge_target, edge_source)

                    if torch.is_grad_enabled() and reduced.requires_grad:
                        part = checkpoint(encode_relation, reduced, t, s, use_reentrant=False)
                    else:
                        part = encode_relation(reduced, t, s)
                    relation_sum = relation_sum + part.sum(dim=0)
            if target.numel():
                mean_relation = (relation_sum / target.numel()).unsqueeze(0)
            else:
                mean_relation = h0.new_zeros((1, self.RELATION_DIM))
            routes = self._route_from_relation(mean_relation, query, prototypes).squeeze(0)
            route_rows = routes.unsqueeze(0)
        else:
            if relation_state_override is not None:
                routes = self._route_from_relation(relation_state_override, query, prototypes)
            else:
                route_parts = []
                for start, end in self._chunks(target.numel()):
                    t, s = target[start:end], source[start:end]

                    def encode_route(reduced_features, edge_target, edge_source):
                        relation = encoder.forward_reduced(reduced_features, edge_target, edge_source)
                        return self._route_from_relation(relation, query, prototypes)

                    if torch.is_grad_enabled() and reduced.requires_grad:
                        part = checkpoint(encode_route, reduced, t, s, use_reentrant=False)
                    else:
                        part = encode_route(reduced, t, s)
                    route_parts.append(part)
                routes = torch.cat(route_parts, dim=0) if route_parts else h0.new_empty((0, self.NUM_EXPERTS))
            route_rows = routes

        if route_rows.numel():
            entropy = -(route_rows.clamp_min(1e-12) * route_rows.clamp_min(1e-12).log()).sum(-1)
            info = {
                "expert_load": route_rows.detach().mean(dim=0),
                "routing_entropy": entropy.detach().mean(),
                "top1_fraction": F.one_hot(route_rows.detach().argmax(-1), self.NUM_EXPERTS).float().mean(0),
            }
        else:
            info = {
                "expert_load": h0.new_full((self.NUM_EXPERTS,), 1.0 / self.NUM_EXPERTS),
                "routing_entropy": h0.new_zeros(()),
                "top1_fraction": h0.new_zeros((self.NUM_EXPERTS,)),
            }
        return routes, info

    def _alter_routes(
        self, routes: torch.Tensor | None, intervention: str | None, seed: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if routes is None or self.operator_variant not in {"global_expert", "relation_expert"}:
            if intervention not in {None, "normal"}:
                raise ValueError(f"intervention {intervention!r} requires expert routing")
            return routes, None
        if intervention in {None, "normal"}:
            return routes, None
        if intervention in {"zero_expert", "mean_expert"}:
            return routes, routes
        if intervention == "global_mean":
            if routes.ndim == 1:
                return routes, routes
            replacement = routes.mean(dim=0, keepdim=True).expand_as(routes)
            return replacement, routes
        if intervention == "shuffle":
            if routes.ndim == 1 or routes.size(0) < 2:
                return routes, routes
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            permutation = torch.randperm(routes.size(0), generator=generator).to(routes.device)
            return routes[permutation], routes
        raise ValueError(f"unknown intervention={intervention!r}")

    def _aggregate_chunk(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        edge_weight: torch.Tensor,
        *,
        modality: str,
        reference_control: torch.Tensor | None = None,
        intervention: str | None = None,
        residual_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_state = state[source]
        if self.operator_variant == "scalar_weight":
            message = source_state * control.unsqueeze(-1)
            reference_message = (
                source_state if reference_control is None else source_state * reference_control.unsqueeze(-1)
            )
        else:
            experts = getattr(self, f"experts_{modality}")
            # Only the current edge chunk has an [E_chunk,R,d] expert tensor.
            expert_values = torch.stack([expert(source_state) for expert in experts], dim=1)
            if intervention == "zero_expert":
                residual = torch.zeros_like(source_state)
                reference_residual = torch.sum(
                    expert_values * reference_control.unsqueeze(-1), dim=1
                )
            elif intervention == "mean_expert":
                residual = expert_values.mean(dim=1)
                reference_residual = torch.sum(
                    expert_values * reference_control.unsqueeze(-1), dim=1
                )
            else:
                residual = torch.sum(expert_values * control.unsqueeze(-1), dim=1)
                reference_residual = (
                    residual
                    if reference_control is None
                    else torch.sum(expert_values * reference_control.unsqueeze(-1), dim=1)
                )
            if residual_scale is not None:
                residual = residual * residual_scale.unsqueeze(-1)
            message = source_state + residual
            reference_message = source_state + reference_residual

        weighted = message * edge_weight.unsqueeze(-1)
        aggregated = state.new_zeros(state.shape)
        aggregated.index_add_(0, target, weighted)
        if reference_control is None:
            change_sum = state.new_zeros(())
            change_count = state.new_zeros(())
        else:
            relative_change = (message - reference_message).norm(dim=-1) / (
                reference_message.norm(dim=-1) + 1e-12
            )
            if residual_scale is not None:
                relative_change = relative_change[residual_scale < 1.0]
            change_sum = relative_change.sum()
            change_count = state.new_tensor(float(relative_change.numel()))
        return aggregated, change_sum, change_count

    def _propagate_semantic(
        self,
        state: torch.Tensor,
        nonself_target: torch.Tensor,
        nonself_source: torch.Tensor,
        nonself_weight: torch.Tensor,
        self_target: torch.Tensor,
        self_source: torch.Tensor,
        self_weight: torch.Tensor,
        controls: torch.Tensor | None,
        *,
        modality: str,
        intervention: str | None = None,
        reference_controls: torch.Tensor | None = None,
        residual_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output = state.new_zeros(state.shape)
        # A physical self-loop always follows the unchanged identity message path.
        output = output.index_add_(
            0, self_target, state[self_source] * self_weight.unsqueeze(-1)
        )
        change_sum = state.new_zeros(())
        change_count = state.new_zeros(())
        for start, end in self._chunks(nonself_target.numel()):
            target = nonself_target[start:end]
            source = nonself_source[start:end]
            weight = nonself_weight[start:end]
            if self.operator_variant == "plain":
                # Plain has no relation controls and can use P@C directly outside this path.
                continue
            if self.operator_variant == "global_expert":
                control = controls
                reference_control = reference_controls
                if control.ndim == 1:
                    control = control.unsqueeze(0).expand(end - start, -1)
                if reference_control is not None and reference_control.ndim == 1:
                    reference_control = reference_control.unsqueeze(0).expand(end - start, -1)
            else:
                control = controls[start:end]
                reference_control = None if reference_controls is None else reference_controls[start:end]
            scale = None if residual_scale is None else residual_scale[start:end]

            has_reference = reference_control is not None
            reference_chunk = control if reference_control is None else reference_control
            has_residual_scale = residual_scale is not None

            def aggregate(
                state_features, edge_control, edge_target, edge_source, edge_weight,
                edge_scale, edge_reference, track_reference=has_reference,
                scale_residual=has_residual_scale,
            ):
                return self._aggregate_chunk(
                    state_features,
                    edge_control,
                    edge_target,
                    edge_source,
                    edge_weight,
                    modality=modality,
                    reference_control=edge_reference if track_reference else None,
                    intervention=intervention,
                    residual_scale=edge_scale if scale_residual else None,
                )

            # Recompute edge activations in backward; only node states and small
            # E_chunk x R controls are retained by autograd.
            call_args = (
                state,
                control,
                target,
                source,
                weight,
                scale if scale is not None else weight.new_ones(weight.shape),
                reference_chunk,
            )
            if torch.is_grad_enabled() and (state.requires_grad or control.requires_grad):
                part, part_change, part_count = checkpoint(
                    aggregate, *call_args, use_reentrant=False
                )
            else:
                part, part_change, part_count = aggregate(*call_args)
            output = output + part
            change_sum = change_sum + part_change
            change_count = change_count + part_count
        message_change = None
        if change_count.item() > 0:
            message_change = change_sum / change_count
        return output, message_change

    def _propagate_one(
        self,
        h0: torch.Tensor,
        operator: torch.Tensor,
        controls: torch.Tensor | None,
        *,
        modality: str,
        intervention: str | None = None,
        reference_controls: torch.Tensor | None = None,
        residual_scale: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
        if self.operator_variant == "plain":
            states = [h0]
            for _ in range(self.MAX_ORDER):
                states.append(torch.sparse.mm(operator, states[-1]))
            return states, [None, None, None]
        edges = self._operator_edges(operator)
        target, source, weights, self_target, self_source, self_weights = edges
        states = [h0]
        changes = []
        for _ in range(self.MAX_ORDER):
            states_next, message_change = self._propagate_semantic(
                states[-1],
                target,
                source,
                weights,
                self_target,
                self_source,
                self_weights,
                controls,
                modality=modality,
                intervention=intervention,
                reference_controls=reference_controls,
                residual_scale=residual_scale,
            )
            states.append(states_next)
            changes.append(message_change)
        return states, changes

    def _controls_for_modality(
        self,
        modality: str,
        h0: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        relation_state_override: torch.Tensor | None,
        intervention: str | None,
        intervention_seed: int,
    ):
        controls, stats = self._edge_controls(
            modality,
            h0,
            target,
            source,
            relation_state_override=relation_state_override,
        )
        active, reference = self._alter_routes(controls, intervention, intervention_seed)
        if self.operator_variant == "global_expert" and active is not None:
            # A2 already uses a modality-global vector, so all routing interventions are null.
            if intervention == "shuffle":
                active = controls
            elif intervention == "global_mean":
                active = controls
        route_change = None
        if active is not None and reference is not None and active.ndim == 2 and active.numel():
            route_change = (active - reference).abs().mean()
        return active, reference, stats, route_change

    def analyze(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        intervention: str | None = None,
        intervention_seed: int = 0,
        relation_state_override_text: torch.Tensor | None = None,
        relation_state_override_visual: torch.Tensor | None = None,
        residual_scale: torch.Tensor | None = None,
        residual_scale_text: torch.Tensor | None = None,
        residual_scale_visual: torch.Tensor | None = None,
    ) -> dict:
        """Return frozen propagation states and compact routing/message diagnostics.

        ``intervention`` supports ``shuffle``, ``global_mean``, ``zero_expert``
        and ``mean_expert`` for validation-only mechanism analysis. A per-edge
        ``residual_scale`` is an offline hook for similarity-stratified tests.
        """
        if edge_index is None:
            raise ValueError("relation_operator_audit requires the physical edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"expected x shape [N,{self.input_dim}], got {tuple(x.shape)}")
        edge_input = edge_index.to(x.device)
        operator = self._get_operator(edge_input, x.size(0), x.dtype)
        nonself_target, nonself_source, _, _, _, _ = self._operator_edges(operator)
        if residual_scale_text is None:
            residual_scale_text = residual_scale
        if residual_scale_visual is None:
            residual_scale_visual = residual_scale
        for scale in (residual_scale_text, residual_scale_visual):
            if scale is not None and scale.shape != nonself_target.shape:
                raise ValueError(f"residual scales must have shape [{nonself_target.numel()}]")

        h0_text = self.text_projector(x[:, : self.text_dim])
        h0_visual = self.visual_projector(x[:, self.text_dim : self.text_dim + self.visual_dim])
        controls_text, ref_text, stats_text, route_change_text = self._controls_for_modality(
            "text", h0_text, nonself_target, nonself_source,
            relation_state_override_text, intervention, intervention_seed,
        )
        controls_visual, ref_visual, stats_visual, route_change_visual = self._controls_for_modality(
            "visual", h0_visual, nonself_target, nonself_source,
            relation_state_override_visual, intervention, intervention_seed + 1,
        )
        if self.operator_variant in {"global_expert", "relation_expert"}:
            if residual_scale_text is not None and ref_text is None:
                ref_text = controls_text
            if residual_scale_visual is not None and ref_visual is None:
                ref_visual = controls_visual
        states_text, message_changes_text = self._propagate_one(
            h0_text, operator, controls_text, modality="text", intervention=intervention,
            reference_controls=ref_text, residual_scale=residual_scale_text,
        )
        states_visual, message_changes_visual = self._propagate_one(
            h0_visual, operator, controls_visual, modality="visual", intervention=intervention,
            reference_controls=ref_visual, residual_scale=residual_scale_visual,
        )
        z_text = torch.stack(states_text, dim=0).mean(dim=0)
        z_visual = torch.stack(states_visual, dim=0).mean(dim=0)
        fused = self.plain_fusion(torch.cat([z_text, z_visual], dim=-1))
        return {
            "H0_text": h0_text,
            "H0_visual": h0_visual,
            "C_text": states_text,
            "C_visual": states_visual,
            "S_text": states_text,
            "S_visual": states_visual,
            "C1_text": states_text[1],
            "C2_text": states_text[2],
            "C3_text": states_text[3],
            "C1_visual": states_visual[1],
            "C2_visual": states_visual[2],
            "C3_visual": states_visual[3],
            "Z_text": z_text,
            "Z_visual": z_visual,
            "fused_z": fused,
            "routing_stats_text": stats_text,
            "routing_stats_visual": stats_visual,
            "intervention_stats": {
                "routing_change_text": route_change_text,
                "routing_change_visual": route_change_visual,
                "mean_message_change_text": self._mean_optional(message_changes_text),
                "mean_message_change_visual": self._mean_optional(message_changes_visual),
            },
            "num_nonself_edges": int(nonself_target.numel()),
            "num_self_edges": int(x.size(0)),
        }

    @staticmethod
    def _mean_optional(values: list[torch.Tensor | None]) -> torch.Tensor | None:
        active = [value for value in values if value is not None]
        if not active:
            return None
        return torch.stack(active).mean()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        z = self.analyze(x, edge_index)["fused_z"]
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
        if edge_index is None:
            raise ValueError("relation_operator_audit requires the physical edge_index")
        if device is None:
            device = next(self.parameters()).device
        return self(x.to(device), edge_index.to(device))[0].detach().cpu()
