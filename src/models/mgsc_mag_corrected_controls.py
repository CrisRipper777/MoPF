"""Strict representation-level controls for canonical MGSC-MAG P2.

These controls are deliberately separate from the existing fine-grained
functional ablations.  ``full`` delegates to canonical MGSC-MAG exactly;
``raw_terminal_only`` and ``raw_bank_mean`` retain Stage-I states but bypass
all Stage-II readout; ``plain_multi_order_backbone`` removes learned MRC,
adaptive gates, order interaction, and preference coefficients while keeping
the projection/refinement/late-fusion capacity and NC interface.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .mgsc_mag import MGSCMAG


CONTROLS = (
    "full",
    "raw_terminal_only",
    "raw_bank_mean",
    "plain_multi_order_backbone",
)


class MGSCMAGCorrectedControls(MGSCMAG):
    """Canonical P2 plus strictly defined raw-state controls."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__(cfg, data_info)
        self.control = str(cfg.model.get("control", "full")).strip().lower()
        if self.control not in CONTROLS:
            raise ValueError(f"unknown corrected control {self.control!r}; expected {CONTROLS}")
        if self.control == "plain_multi_order_backbone":
            self._remove_conditional_parameters()

    def _remove_conditional_parameters(self) -> None:
        """Remove conditional modules so plain parameter counts are natural."""
        names = (
            "metric_theta_text",
            "metric_theta_visual",
            "gamma_global",
            "delta_gamma_text",
            "delta_gamma_visual",
            "node_proj_text",
            "node_proj_visual",
            "node_vector_text",
            "node_vector_visual",
            "hop_order_embedding_text",
            "hop_order_embedding_visual",
            "hop_layers_text",
            "hop_layers_visual",
            "theta_hop_gate_text",
            "theta_hop_gate_visual",
            "relation_beta_raw_text",
            "relation_beta_raw_visual",
            "theta_relation_scale_text",
            "theta_relation_scale_visual",
            "context_gate_text",
            "context_gate_visual",
            "context_gate_order_embedding_text",
            "context_gate_order_embedding_visual",
        )
        for name in names:
            if hasattr(self, name):
                delattr(self, name)
        # The inherited optimizer preset refers to relation-profile parameters
        # that no longer exist in the plain backbone.
        self.no_weight_decay_parameter_names = frozenset()

    def _relation_calibration(
        self, h_text: torch.Tensor, h_visual: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.control != "plain_multi_order_backbone":
            return super()._relation_calibration(h_text, h_visual, edge_index)
        text_weight = torch.ones(edge_index.size(1), device=h_text.device, dtype=h_text.dtype)
        visual_weight = torch.ones(edge_index.size(1), device=h_visual.device, dtype=h_visual.dtype)
        return {
            "semantic_cosine_text": torch.zeros_like(text_weight),
            "semantic_cosine_visual": torch.zeros_like(visual_weight),
            "relation_weight_text": text_weight,
            "relation_weight_visual": visual_weight,
            "metric_weights_text": torch.ones_like(h_text[:1]),
            "metric_weights_visual": torch.ones_like(h_visual[:1]),
        }

    def _plain_multi_order_states(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        states = [h0]
        current = h0
        for _ in range(self.max_order):
            proposal = self._propagate_once(current, edge_index, edge_weight)
            current = 0.1 * h0 + 0.9 * proposal
            states.append(current)
        return states

    @staticmethod
    def _identity_attention(states: list[torch.Tensor]) -> torch.Tensor:
        identity = torch.eye(
            len(states), dtype=states[0].dtype, device=states[0].device
        )
        return identity.unsqueeze(0).expand(states[0].size(0), -1, -1)

    def _refine_and_fuse(
        self,
        z_text: torch.Tensor,
        z_visual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_text_refined = self.text_refine_norm(z_text + self.text_refine_mlp(z_text))
        z_visual_refined = self.visual_refine_norm(
            z_visual + self.visual_refine_mlp(z_visual)
        )
        fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
        z = self.output_norm(self.fusion_skip(fused_input) + self.fusion_mlp(fused_input))
        return z_text_refined, z_visual_refined, z

    def _control_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        capture_attention: bool,
    ) -> dict[str, Any]:
        x_text, x_visual = self._split_features(x)
        h_text = self.text_proj(x_text)
        h_visual = self.visual_proj(x_visual)
        calibrated = self._relation_calibration(h_text, h_visual, edge_index)
        norm_text_index, norm_text_weight = self._normalized_operator(
            edge_index, calibrated["relation_weight_text"], int(x.size(0)), h_text.dtype
        )
        norm_visual_index, norm_visual_weight = self._normalized_operator(
            edge_index, calibrated["relation_weight_visual"], int(x.size(0)), h_visual.dtype
        )

        if self.control == "plain_multi_order_backbone":
            states_text = self._plain_multi_order_states(h_text, norm_text_index, norm_text_weight)
            states_visual = self._plain_multi_order_states(h_visual, norm_visual_index, norm_visual_weight)
            gate_text = [h_text.new_full((h_text.size(0),), 0.9) for _ in range(self.max_order)]
            gate_visual = [h_visual.new_full((h_visual.size(0),), 0.9) for _ in range(self.max_order)]
        else:
            states_text, gate_text = self._adaptive_multi_hop_states(
                h_text, norm_text_index, norm_text_weight, "text"
            )
            states_visual, gate_visual = self._adaptive_multi_hop_states(
                h_visual, norm_visual_index, norm_visual_weight, "visual"
            )

        # These controls intentionally do not call _cross_order_interaction.
        if self.control == "raw_terminal_only":
            z_text = states_text[-1]
            z_visual = states_visual[-1]
            readout = "raw_terminal"
        else:
            z_text = torch.stack(states_text, dim=1).mean(dim=1)
            z_visual = torch.stack(states_visual, dim=1).mean(dim=1)
            readout = "raw_bank_mean"

        z_text_refined, z_visual_refined, z = self._refine_and_fuse(z_text, z_visual)
        zeros_text = h_text.new_zeros((h_text.size(0), self.max_order + 1))
        zeros_visual = h_visual.new_zeros((h_visual.size(0), self.max_order + 1))
        relation_context_text = h_text.new_zeros(h_text.size(0))
        relation_context_visual = h_visual.new_zeros(h_visual.size(0))
        incident_text = h_text.new_zeros(h_text.size(0))
        incident_visual = h_visual.new_zeros(h_visual.size(0))
        identity_text = self._identity_attention(states_text)
        identity_visual = self._identity_attention(states_visual)
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
            "context_gate_text": gate_text,
            "context_gate_visual": gate_visual,
            "local_relation_context_text": relation_context_text,
            "local_relation_context_visual": relation_context_visual,
            "incident_relation_mean_text": incident_text,
            "incident_relation_mean_visual": incident_visual,
            "beta_hat_text": h_text.new_zeros(self.max_order + 1),
            "beta_hat_visual": h_visual.new_zeros(self.max_order + 1),
            "relation_scale_text": h_text.new_zeros(()),
            "relation_scale_visual": h_visual.new_zeros(()),
            "hop_gate_text": h_text.new_zeros(()),
            "hop_gate_visual": h_visual.new_zeros(()),
            "interacted_states_text": states_text,
            "interacted_states_visual": states_visual,
            "delta_text": zeros_text,
            "delta_visual": zeros_visual,
            "eta_text": zeros_text,
            "eta_visual": zeros_visual,
            "effective_order_text": h_text.new_zeros(h_text.size(0)),
            "effective_order_visual": h_visual.new_zeros(h_visual.size(0)),
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": z,
            "control_readout_source": readout,
            "control_raw_states_text": states_text,
            "control_raw_states_visual": states_visual,
        }
        if capture_attention:
            output["attention_text"] = identity_text
            output["attention_visual"] = identity_visual
        return output

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
        if self.control == "full":
            return super()._encode_components(
                x,
                edge_index,
                relation_intervention=relation_intervention,
                interaction_intervention=interaction_intervention,
                relation_permutation=relation_permutation,
                capture_attention=capture_attention,
                gate_intervention=gate_intervention,
                gate_permutations=gate_permutations,
            )
        del relation_intervention, interaction_intervention, relation_permutation
        del gate_intervention, gate_permutations
        return self._control_components(x, edge_index, capture_attention=capture_attention)

    @torch.no_grad()
    def analysis_multi_order(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, Any]:
        if self.control == "full":
            return super().analysis_multi_order(x, edge_index)
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._control_components(x, edge_index, capture_attention=True)
        result = {
            key: components[key].detach().clone()
            for key in (
                "h0_text", "h0_visual", "physical_edge_index",
                "semantic_cosine_text", "semantic_cosine_visual",
                "relation_weight_text", "relation_weight_visual",
                "normalized_edge_index_text", "normalized_edge_weight_text",
                "normalized_edge_index_visual", "normalized_edge_weight_visual",
                "eta_text", "eta_visual", "z_text", "z_visual",
                "z_text_refined", "z_visual_refined", "z",
                "attention_text", "attention_visual",
            )
        }
        result["states_text"] = [value.detach().clone() for value in components["states_text"]]
        result["states_visual"] = [value.detach().clone() for value in components["states_visual"]]
        result["interacted_states_text"] = result["states_text"]
        result["interacted_states_visual"] = result["states_visual"]
        result["context_gate_text"] = [value.detach().clone() for value in components["context_gate_text"]]
        result["context_gate_visual"] = [value.detach().clone() for value in components["context_gate_visual"]]
        result["control_readout_source"] = components["control_readout_source"]
        result["control_raw_states_text"] = result["states_text"]
        result["control_raw_states_visual"] = result["states_visual"]
        return result


Model = MGSCMAGCorrectedControls
