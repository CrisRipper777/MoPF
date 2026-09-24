from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .ssi_mag_v31 import SSIMAGV31


class SSIMAGV31R1U(SSIMAGV31):
    """P1.8 direct-utilization R1 pilot.

    This class is an independent development model.  It preserves the frozen
    V3.1 R2, context-change, cross-hop attention, signed filtering, historical
    reference residual, late fusion, and NC interfaces.  Only relation
    utilization changes: the semantic relation score is used directly in
    ``exp(a)`` without a learned beta attenuation.
    """

    ABLATIONS = {"full"}
    no_weight_decay_parameter_names = frozenset()

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info)
        # V3.1 creates these for its beta path.  R1U deliberately removes them
        # from the module and state_dict: there is no theta_beta or beta path.
        del self.theta_beta_text
        del self.theta_beta_visual

        model_cfg = cfg.model
        self.reference_filter_scale_init = float(
            model_cfg.get("reference_filter_scale_init", 0.10)
        )
        self.reference_filter_scale_text = nn.Parameter(
            torch.tensor(self.reference_filter_scale_init)
        )
        self.reference_filter_scale_visual = nn.Parameter(
            torch.tensor(self.reference_filter_scale_init)
        )
        self._reference_intervention_off = False

    def _relation_edge_outputs(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        modality: str,
        *,
        force_unit: bool = False,
    ) -> dict[str, torch.Tensor]:
        relation = self._relation_descriptor(h0, edge_index, modality)
        score = relation["relation_score"]
        relation_weight = (
            torch.ones_like(score) if force_unit else torch.exp(score)
        )
        # These zero tensors are compatibility exports for the common analysis
        # schema, not parameters and not a multiplicative beta mechanism.
        zero = h0.new_zeros(())
        return {
            **relation,
            "relation_residual": score,
            "beta": zero,
            "beta_parameter": zero,
            "relation_weight": relation_weight,
        }

    @staticmethod
    def _local_adaptation(
        edge_index: torch.Tensor,
        relation_residual: torch.Tensor,
        beta: torch.Tensor,
        nonself_mask: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        del beta
        if edge_index.numel() == 0 or not bool(nonself_mask.any()):
            return relation_residual.new_zeros((num_nodes,))
        physical_index = edge_index[:, nonself_mask]
        incident_values = relation_residual[nonself_mask].abs()
        endpoints = torch.cat((physical_index[0], physical_index[1]), dim=0)
        incident = torch.cat((incident_values, incident_values), dim=0)
        total = relation_residual.new_zeros((num_nodes,))
        count = relation_residual.new_zeros((num_nodes,))
        total.index_add_(0, endpoints, incident)
        count.index_add_(0, endpoints, torch.ones_like(incident))
        return torch.where(
            count > 0,
            total / count.clamp_min(1.0),
            torch.zeros_like(total),
        )

    def _filter_contexts(
        self,
        states: list[torch.Tensor],
        local_adaptation: torch.Tensor,
        modality: str,
    ) -> dict[str, torch.Tensor]:
        alpha_bank = torch.cat(
            [torch.ones_like(states[0][..., 0]).unsqueeze(-1)]
            + [alpha.unsqueeze(-1) for alpha in self._active_alphas],
            dim=-1,
        )
        alpha_centered = alpha_bank - alpha_bank.mean(dim=-1, keepdim=True)
        relation_profile = self._relation_profile(modality).to(
            device=states[0].device, dtype=states[0].dtype
        )
        reference_scale = getattr(self, f"reference_filter_scale_{modality}")
        relation_scale = getattr(self, f"relation_filter_scale_{modality}")
        reference_residual = reference_scale * alpha_centered
        if self._reference_intervention_off:
            reference_residual = torch.zeros_like(reference_residual)
        relation_residual = (
            relation_scale
            * local_adaptation.unsqueeze(-1)
            * relation_profile.unsqueeze(0)
        )
        delta_content = self._node_content_score(states, modality)
        gamma = self.gamma_global.to(dtype=states[0].dtype).unsqueeze(0)
        delta_gamma = getattr(self, f"delta_gamma_{modality}").unsqueeze(0)
        eta = gamma + delta_gamma + delta_content + reference_residual + relation_residual
        return {
            "alpha_bank": alpha_bank,
            "alpha_centered": alpha_centered,
            "relation_profile": relation_profile,
            "delta_content": delta_content,
            "reference_residual": reference_residual,
            "relation_residual": relation_residual,
            "delta": delta_content + reference_residual + relation_residual,
            "eta_adaptive": eta,
            "eta": eta,
        }

    def _encode_modality(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        modality: str,
        *,
        relation_intervention: str,
        interaction_intervention: str,
        capture_attention: bool,
    ) -> dict[str, Any]:
        force_unit = relation_intervention == "off"
        relation = self._relation_edge_outputs(
            h0, edge_index, modality, force_unit=force_unit
        )
        local_adaptation = self._local_adaptation(
            edge_index,
            relation["relation_residual"],
            relation["beta"],
            relation["nonself_mask"],
            h0.size(0),
        )
        if force_unit:
            local_adaptation = torch.zeros_like(local_adaptation)
        norm_index, norm_weight = self._normalized_operator(
            edge_index, relation["relation_weight"], h0.size(0), h0.dtype
        )
        semantic = self._semantic_states(h0, norm_index, norm_weight, modality)
        self._active_alphas = semantic["alphas"]
        try:
            deltas, tokens, delta_gate, injections = self._context_tokens(
                semantic["states"], modality
            )
            s_tilde, attention, interaction_gate, interaction_output = self._cross_hop_interaction(
                semantic["states"],
                tokens,
                modality,
                interaction_off=interaction_intervention == "off",
                capture_attention=capture_attention,
            )
            filtering = self._filter_contexts(
                s_tilde, local_adaptation, modality
            )
        finally:
            del self._active_alphas
        z = self._compose(s_tilde, filtering["eta"])
        return {
            **relation,
            "normalized_edge_index": norm_index,
            "normalized_edge_weight": norm_weight,
            "local_adaptation": local_adaptation,
            "semantic": semantic,
            "deltas": deltas,
            "tokens": tokens,
            "delta_gate": delta_gate,
            "context_change_injection": injections,
            "attention": attention if capture_attention else None,
            "interaction_gate": interaction_gate,
            "interaction_output": interaction_output,
            "s_tilde": s_tilde,
            "filtering": filtering,
            "z": z,
        }

    def _analysis_public(self, *args, reference: str = "normal", **kwargs):
        reference = str(reference).strip().lower()
        if reference not in {"normal", "off"}:
            raise ValueError("reference must be normal|off")
        self._reference_intervention_off = reference == "off"
        try:
            result = super()._analysis_public(*args, **kwargs)
        finally:
            self._reference_intervention_off = False
        for modality in ("text", "visual"):
            alpha = result[f"alpha_{modality}"]
            alpha_bank = torch.cat(
                [torch.ones_like(result[f"p_{modality}"]).unsqueeze(-1)]
                + [value.unsqueeze(-1) for value in alpha],
                dim=-1,
            )
            centered = alpha_bank - alpha_bank.mean(dim=-1, keepdim=True)
            result[f"alpha_centered_{modality}"] = centered
            result[f"reference_filter_scale_{modality}"] = getattr(
                self, f"reference_filter_scale_{modality}"
            ).detach().clone()
            result[f"reference_residual_{modality}"] = (
                torch.zeros_like(centered)
                if reference == "off"
                else getattr(self, f"reference_filter_scale_{modality}") * centered
            )
            result[f"eta_adaptive_{modality}"] = result[f"eta_{modality}"]
            result[f"delta_{modality}"] = (
                result[f"delta_content_{modality}"]
                + result[f"reference_residual_{modality}"]
                + result[f"relation_filter_residual_{modality}"]
            )
        result["reference_intervention"] = reference
        result["r1u"] = True
        return result

    @torch.no_grad()
    def analysis_intervention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        interaction: str = "normal",
        relation: str = "normal",
        reference: str = "normal",
    ) -> dict[str, Any]:
        result = self._analysis_public(
            x,
            edge_index,
            relation=relation,
            interaction=interaction,
            reference=reference,
        )
        result["interaction_intervention"] = str(interaction)
        result["relation_intervention"] = str(relation)
        result["reference_intervention"] = str(reference)
        return result


Model = SSIMAGV31R1U
