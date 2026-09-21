from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .cosi_mag_final import CoSIMAGFinal


ABLATION_MODES = frozenset(
    {"full", "no_mrc", "no_semantic_anchor", "no_rcmi"}
)


class CoSIMAGAblation(CoSIMAGFinal):
    """Framework-level ablations of the frozen CoSI-MAG final encoder."""

    def __init__(self, cfg, data_info: dict):
        model_cfg = cfg.model
        mode = str(model_cfg.get("ablation_mode", "full")).strip().lower()
        if mode not in ABLATION_MODES:
            raise ValueError(
                f"ablation_mode must be one of {sorted(ABLATION_MODES)}, got {mode!r}"
            )

        configured_alpha = float(model_cfg.get("multihop_anchor_alpha", 0.1))
        expected_alpha = 0.0 if mode == "no_semantic_anchor" else 0.1
        if not math.isclose(
            configured_alpha, expected_alpha, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{mode} requires model.multihop_anchor_alpha={expected_alpha}; "
                f"got {configured_alpha}"
            )

        super().__init__(cfg, data_info)
        self.ablation_mode = mode

    def _relation_calibration(
        self, h_text: torch.Tensor, h_visual: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.ablation_mode != "no_mrc":
            return super()._relation_calibration(h_text, h_visual, edge_index)

        # MRC is removed as a stage: physical edges receive unit weights and
        # the learned metrics are not evaluated on the propagation path.
        edge_count = int(edge_index.size(1))
        weight_text = h_text.new_ones(edge_count)
        weight_visual = h_visual.new_ones(edge_count)
        return {
            "semantic_cosine_text": h_text.new_zeros(edge_count),
            "semantic_cosine_visual": h_visual.new_zeros(edge_count),
            "relation_weight_text": weight_text,
            "relation_weight_visual": weight_visual,
            "metric_weights_text": h_text.new_ones((1, h_text.size(-1))),
            "metric_weights_visual": h_visual.new_ones((1, h_visual.size(-1))),
        }

    def _local_relation_context(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ablation_mode != "no_mrc":
            return super()._local_relation_context(edge_index, edge_weight, num_nodes)

        # Do not infer a relation descriptor from unit weights. In particular,
        # isolated nodes must not acquire a nonzero z-score from the graph mean.
        context = edge_weight.new_zeros(num_nodes)
        incident_mean = self._incident_mean(
            edge_index, edge_weight.detach(), num_nodes
        )
        return context, incident_mean.detach()

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
        if self.ablation_mode != "no_mrc":
            return super()._cross_order_interaction(
                states,
                modality,
                local_context,
                relation_intervention=relation_intervention,
                interaction_intervention=interaction_intervention,
                capture_attention=capture_attention,
            )

        # Keep content attention, while excluding the relation-conditioned
        # logit bias as an explicit ablation edge in the dependency graph.
        return super()._cross_order_interaction(
            states,
            modality,
            local_context,
            relation_intervention="off",
            interaction_intervention=interaction_intervention,
            capture_attention=capture_attention,
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
    ) -> dict[str, Any]:
        if self.ablation_mode != "no_rcmi":
            return super()._encode_components(
                x,
                edge_index,
                relation_intervention=relation_intervention,
                interaction_intervention=interaction_intervention,
                relation_permutation=relation_permutation,
                capture_attention=capture_attention,
            )
        return self._encode_without_rcmi(x, edge_index)

    def _encode_without_rcmi(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, Any]:
        """Run projection, MRC, SMP, and fusion without entering RCMI."""
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

        # Uniformly pool the order bank. No order tokens, attention, local
        # relation descriptor, node preference, or learned order coefficients
        # participate in this branch.
        z_text = torch.stack(states_text, dim=1).mean(dim=1)
        z_visual = torch.stack(states_visual, dim=1).mean(dim=1)
        z_text_refined = self.text_refine_norm(
            z_text + self.text_refine_mlp(z_text)
        )
        z_visual_refined = self.visual_refine_norm(
            z_visual + self.visual_refine_mlp(z_visual)
        )
        fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
        z = self.output_norm(
            self.fusion_skip(fused_input) + self.fusion_mlp(fused_input)
        )

        return {
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
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": z,
        }


Model = CoSIMAGAblation
