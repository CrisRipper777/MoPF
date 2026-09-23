"""MGSC-MAG candidate encoder with adaptive context-state formation.

This module subclasses the frozen CoSI-MAG reference.  The default candidate
path keeps the reference relation calibration, modality-specific propagation,
cross-order interaction, preference, fusion, and task-facing interfaces.  P1
only changes the fixed semantic-anchor recurrence when
``adaptive_context_gate=True``.  P2's direct interacted-state integration is
implemented as a separate opt-in switch and is false by default.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .cosi_mag_final import CoSIMAGFinal


def _logit(probability: float) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {value}")
    return math.log(value / (1.0 - value))


class MGSCMAG(CoSIMAGFinal):
    """CoSI-compatible MGSC-MAG candidate with node/order adaptive gates."""

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info)
        self.adaptive_context_gate = bool(cfg.model.get("adaptive_context_gate", True))
        self.direct_interacted_integration = bool(
            cfg.model.get("direct_interacted_integration", False)
        )
        self.use_legacy_relation_order_bias = bool(
            cfg.model.get("use_legacy_relation_order_bias", True)
        )
        self.context_gate_order_dim = int(cfg.model.get("context_gate_order_dim", 16))
        self.context_gate_hidden_dim = int(cfg.model.get("context_gate_hidden_dim", 64))
        if self.context_gate_order_dim < 1 or self.context_gate_hidden_dim < 1:
            raise ValueError("context gate dimensions must be positive")

        gate_input_dim = 4 * self.hidden_dim + self.context_gate_order_dim
        self.context_gate_text = nn.Sequential(
            nn.Linear(gate_input_dim, self.context_gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.context_gate_hidden_dim, 1),
        )
        self.context_gate_visual = nn.Sequential(
            nn.Linear(gate_input_dim, self.context_gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.context_gate_hidden_dim, 1),
        )
        self.context_gate_order_embedding_text = nn.Parameter(
            torch.zeros(self.max_order + 1, self.context_gate_order_dim)
        )
        self.context_gate_order_embedding_visual = nn.Parameter(
            torch.zeros(self.max_order + 1, self.context_gate_order_dim)
        )
        self._initialize_context_gate(self.context_gate_text)
        self._initialize_context_gate(self.context_gate_visual)

    @staticmethod
    def _initialize_context_gate(module: nn.Sequential) -> None:
        """Initialize the gate head so every order starts at g≈0.9."""
        output_layer = module[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("context gate must end with a Linear layer")
        nn.init.zeros_(output_layer.weight)
        nn.init.constant_(output_layer.bias, _logit(0.9))

    def _adaptive_multi_hop_states(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        modality: str,
        override_gates: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Form one modality's state bank with node/order adaptive gates."""
        if modality == "text":
            gate_mlp = self.context_gate_text
            order_embedding = self.context_gate_order_embedding_text
        elif modality == "visual":
            gate_mlp = self.context_gate_visual
            order_embedding = self.context_gate_order_embedding_visual
        else:
            raise ValueError(f"unknown modality {modality!r}")
        if override_gates is not None and len(override_gates) != self.max_order:
            raise ValueError(
                f"override_gates must contain {self.max_order} orders, got {len(override_gates)}"
            )

        states = [h0]
        gates: list[torch.Tensor] = []
        current = h0
        for order in range(1, self.max_order + 1):
            propagated = self._propagate_once(current, edge_index, edge_weight)
            order_token = order_embedding[order].unsqueeze(0).expand(h0.size(0), -1)
            gate_input = torch.cat(
                (
                    h0,
                    propagated,
                    (h0 - propagated).abs(),
                    h0 * propagated,
                    order_token,
                ),
                dim=-1,
            )
            if override_gates is None:
                gate = torch.sigmoid(gate_mlp(gate_input)).squeeze(-1)
            else:
                gate = override_gates[order - 1].to(
                    device=h0.device, dtype=h0.dtype
                )
            current = (1.0 - gate.unsqueeze(-1)) * h0 + gate.unsqueeze(-1) * propagated
            states.append(current)
            gates.append(gate)
        return states, gates

    @staticmethod
    def _override_gate_bank(
        gate_bank: list[torch.Tensor],
        intervention: str,
        permutations: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Create analysis-only gate values while preserving normal marginals."""
        if intervention == "normal":
            return gate_bank
        if intervention == "globalized":
            return [gate.mean().expand_as(gate) for gate in gate_bank]
        if intervention == "fixed_0.9":
            return [torch.full_like(gate, 0.9) for gate in gate_bank]
        if intervention == "shuffled":
            if permutations is None or len(permutations) != len(gate_bank):
                raise ValueError("shuffled gate intervention requires one permutation per order")
            return [
                gate.index_select(0, permutation.to(device=gate.device))
                for gate, permutation in zip(gate_bank, permutations)
            ]
        raise ValueError(
            "gate_intervention must be normal|globalized|shuffled|fixed_0.9"
        )

    def _encode_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        relation_intervention: str = "normal",
        interaction_intervention: str = "normal",
        relation_permutation: torch.Tensor | None = None,
        capture_attention: bool = False,
        gate_intervention: str = "normal",
        gate_permutations: dict[str, list[torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        # This branch is intentionally delegated to the frozen implementation.
        # It gives the candidate an exact shared-parameter eval path when both
        # P1 and P2 switches are disabled.
        if not self.adaptive_context_gate and not self.direct_interacted_integration:
            return super()._encode_components(
                x,
                edge_index,
                relation_intervention=relation_intervention,
                interaction_intervention=interaction_intervention,
                relation_permutation=relation_permutation,
                capture_attention=capture_attention,
            )

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
        if self.adaptive_context_gate:
            states_text, normal_gate_text = self._adaptive_multi_hop_states(
                h_text, norm_text_index, norm_text_weight, "text"
            )
            states_visual, normal_gate_visual = self._adaptive_multi_hop_states(
                h_visual, norm_visual_index, norm_visual_weight, "visual"
            )
            context_gate_text = self._override_gate_bank(
                normal_gate_text,
                gate_intervention,
                None if gate_permutations is None else gate_permutations.get("text"),
            )
            context_gate_visual = self._override_gate_bank(
                normal_gate_visual,
                gate_intervention,
                None if gate_permutations is None else gate_permutations.get("visual"),
            )
            if gate_intervention != "normal":
                states_text, _ = self._adaptive_multi_hop_states(
                    h_text,
                    norm_text_index,
                    norm_text_weight,
                    "text",
                    override_gates=context_gate_text,
                )
                states_visual, _ = self._adaptive_multi_hop_states(
                    h_visual,
                    norm_visual_index,
                    norm_visual_weight,
                    "visual",
                    override_gates=context_gate_visual,
                )
        else:
            states_text = self._multi_hop_states(h_text, norm_text_index, norm_text_weight)
            states_visual = self._multi_hop_states(
                h_visual, norm_visual_index, norm_visual_weight
            )
            context_gate_text = []
            context_gate_visual = []

        relation_context_text, incident_mean_text = self._local_relation_context(
            edge_index,
            calibrated["relation_weight_text"],
            int(x.size(0)),
        )
        relation_context_visual, incident_mean_visual = self._local_relation_context(
            edge_index,
            calibrated["relation_weight_visual"],
            int(x.size(0)),
        )
        if relation_permutation is not None:
            relation_context_text = relation_context_text.index_select(0, relation_permutation)
            relation_context_visual = relation_context_visual.index_select(0, relation_permutation)

        attention_relation_context_text = (
            relation_context_text
            if self.use_legacy_relation_order_bias
            else torch.zeros_like(relation_context_text)
        )
        attention_relation_context_visual = (
            relation_context_visual
            if self.use_legacy_relation_order_bias
            else torch.zeros_like(relation_context_visual)
        )

        interacted_text, attention_text = self._cross_order_interaction(
            states_text,
            "text",
            attention_relation_context_text,
            relation_intervention=relation_intervention,
            interaction_intervention=interaction_intervention,
            capture_attention=capture_attention,
        )
        interacted_visual, attention_visual = self._cross_order_interaction(
            states_visual,
            "visual",
            attention_relation_context_visual,
            relation_intervention=relation_intervention,
            interaction_intervention=interaction_intervention,
            capture_attention=capture_attention,
        )
        delta_text = self._node_preference(interacted_text, "text")
        delta_visual = self._node_preference(interacted_visual, "visual")
        eta_text = self.gamma_global.unsqueeze(0) + self.delta_gamma_text.unsqueeze(0) + delta_text
        eta_visual = self.gamma_global.unsqueeze(0) + self.delta_gamma_visual.unsqueeze(0) + delta_visual
        compose_text = interacted_text if self.direct_interacted_integration else states_text
        compose_visual = interacted_visual if self.direct_interacted_integration else states_visual
        z_text = self._compose(compose_text, eta_text)
        z_visual = self._compose(compose_visual, eta_visual)
        z_text_refined = self.text_refine_norm(z_text + self.text_refine_mlp(z_text))
        z_visual_refined = self.visual_refine_norm(z_visual + self.visual_refine_mlp(z_visual))
        fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
        z = self.output_norm(self.fusion_skip(fused_input) + self.fusion_mlp(fused_input))

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
            "context_gate_text": context_gate_text,
            "context_gate_visual": context_gate_visual,
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

    @torch.no_grad()
    def analysis_multi_order(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, Any]:
        result = super().analysis_multi_order(x, edge_index)
        components = self._analysis_call(x, self._edge_index_or_empty(edge_index, x.device))
        for modality in ("text", "visual"):
            gates = components.get(f"context_gate_{modality}", [])
            result[f"context_gate_{modality}"] = [gate.detach().clone() for gate in gates]
            if gates:
                gate_bank = torch.stack(gates, dim=1)
                result[f"context_gate_order_std_{modality}"] = gate_bank.std(dim=1, unbiased=False)
                result[f"context_gate_summary_{modality}"] = torch.stack(
                    (gate_bank.mean(), gate_bank.std(unbiased=False))
                )
            else:
                result[f"context_gate_order_std_{modality}"] = x.new_zeros(x.size(0))
                result[f"context_gate_summary_{modality}"] = x.new_zeros(2)
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
        gate_intervention: str = "normal",
        gate_permutation_seed: int = 0,
    ) -> dict[str, Any]:
        """Run inference-only relation, interaction, and gate interventions.

        Gate interventions are audit controls.  They do not add trainable
        parameters and are never used by the task runner's training path.
        Shuffling uses a fixed, independent permutation per modality/order so
        that every intervened gate distribution has the corresponding normal
        marginal distribution.
        """
        relation = str(relation).strip().lower()
        interaction = str(interaction).strip().lower()
        gate_intervention = str(gate_intervention).strip().lower()
        if relation not in {"normal", "off", "shuffle"}:
            raise ValueError("relation must be normal|off|shuffle")
        if interaction not in {"normal", "query_collapse", "uniform", "off"}:
            raise ValueError("interaction must be normal|query_collapse|uniform|off")
        if gate_intervention not in {"normal", "globalized", "shuffled", "fixed_0.9"}:
            raise ValueError(
                "gate_intervention must be normal|globalized|shuffled|fixed_0.9"
            )
        if gate_intervention != "normal" and not self.adaptive_context_gate:
            raise ValueError("gate interventions require adaptive_context_gate=true")

        edge_index = self._edge_index_or_empty(edge_index, x.device)
        relation_permutation = None
        if relation == "shuffle":
            generator = torch.Generator(device="cpu").manual_seed(int(permutation_seed))
            relation_permutation = torch.randperm(x.size(0), generator=generator).to(x.device)

        gate_permutations = None
        if gate_intervention == "shuffled":
            gate_permutations = {}
            for modality_offset, modality in enumerate(("text", "visual")):
                modality_permutations = []
                for order in range(self.max_order):
                    generator = torch.Generator(device="cpu").manual_seed(
                        int(gate_permutation_seed) + modality_offset * 1000 + order
                    )
                    modality_permutations.append(
                        torch.randperm(x.size(0), generator=generator).to(x.device)
                    )
                gate_permutations[modality] = modality_permutations

        training_states = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            return self._encode_components(
                x,
                edge_index,
                relation_intervention=relation,
                interaction_intervention=interaction,
                relation_permutation=relation_permutation,
                capture_attention=True,
                gate_intervention=gate_intervention,
                gate_permutations=gate_permutations,
            )
        finally:
            for module, was_training in training_states:
                module.training = was_training


Model = MGSCMAG
