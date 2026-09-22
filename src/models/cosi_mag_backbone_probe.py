from __future__ import annotations

from typing import Any

import torch

from .cosi_mag_ablation import CoSIMAGAblation


PROBE_MODES = frozenset(
    {
        "no_graph",
        "last_hop",
        "simple_fusion",
        "early_fusion",
    }
)


class CoSIMAGBackboneProbe(CoSIMAGAblation):
    """Architecture probes for the strong plain CoSI-MAG backbone.

    Every probe uses unit edge weights, alpha=0 ordinary propagation and no
    RCMI.  The probes isolate graph use, explicit retention of all propagation
    orders, late multimodal processing, and the residual fusion block.
    """

    def __init__(self, cfg, data_info: dict):
        probe_mode = str(cfg.model.get("probe_mode", "no_graph")).strip().lower()
        if probe_mode not in PROBE_MODES:
            raise ValueError(
                f"probe_mode must be one of {sorted(PROBE_MODES)}, got {probe_mode!r}"
            )
        if str(cfg.model.get("ablation_mode", "")).strip().lower() != "all_plain":
            raise ValueError("backbone probes require ablation_mode=all_plain")
        super().__init__(cfg, data_info)
        self.probe_mode = probe_mode

    def _encode_without_rcmi(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, Any]:
        x_text, x_visual = self._split_features(x)
        h_text = self.text_proj(x_text)
        h_visual = self.visual_proj(x_visual)
        calibrated = self._relation_calibration(h_text, h_visual, edge_index)

        norm_index, norm_weight = self._normalized_operator(
            edge_index,
            calibrated["relation_weight_text"],
            int(x.size(0)),
            h_text.dtype,
        )
        states_text = self._multi_hop_states(h_text, norm_index, norm_weight)
        states_visual = self._multi_hop_states(h_visual, norm_index, norm_weight)

        if self.probe_mode == "no_graph":
            z_text = h_text
            z_visual = h_visual
        elif self.probe_mode == "last_hop":
            z_text = states_text[-1]
            z_visual = states_visual[-1]
        else:
            z_text = torch.stack(states_text, dim=1).mean(dim=1)
            z_visual = torch.stack(states_visual, dim=1).mean(dim=1)

        if self.probe_mode == "simple_fusion":
            z_text_refined = z_text
            z_visual_refined = z_visual
            z = 0.5 * (z_text + z_visual)
        elif self.probe_mode == "early_fusion":
            joint_h0 = 0.5 * (h_text + h_visual)
            joint_states = self._multi_hop_states(joint_h0, norm_index, norm_weight)
            joint = torch.stack(joint_states, dim=1).mean(dim=1)
            z = self.text_refine_norm(joint + self.text_refine_mlp(joint))
            z_text = joint
            z_visual = joint
            z_text_refined = z
            z_visual_refined = z
        else:
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
            "normalized_edge_index_text": norm_index,
            "normalized_edge_weight_text": norm_weight,
            "normalized_edge_index_visual": norm_index,
            "normalized_edge_weight_visual": norm_weight,
            "states_text": states_text,
            "states_visual": states_visual,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": z,
        }


Model = CoSIMAGBackboneProbe
