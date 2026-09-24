from __future__ import annotations

from typing import Any

import torch

from .ssi_mag_v31_r1u import SSIMAGV31R1U


class SSIMAGFinal(SSIMAGV31R1U):
    """Canonical frozen SSI-MAG final implementation.

    This is the P1.8-U computation graph with a stateless context/filter API.
    It has no beta parameter: semantic relation scores are used directly as
    ``w_ij = exp(a_ij)``.  The class is separate from the development pilot
    so paper code cannot depend on temporary mutable state.
    """

    ABLATIONS = {"full"}
    no_weight_decay_parameter_names = frozenset()

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info)
        if hasattr(self, "relation_strength_init"):
            delattr(self, "relation_strength_init")

    def _filter_contexts(
        self,
        states: list[torch.Tensor],
        semantic: dict[str, Any],
        local_adaptation: torch.Tensor,
        modality: str,
    ) -> dict[str, torch.Tensor]:
        """Formation-conditioned signed coefficients with explicit inputs."""
        alpha_bank = torch.cat(
            [torch.ones_like(semantic["p"]).unsqueeze(-1)]
            + [alpha.unsqueeze(-1) for alpha in semantic["alphas"]],
            dim=-1,
        )
        alpha_centered = alpha_bank - alpha_bank.mean(dim=-1, keepdim=True)
        relation_profile = self._relation_profile(modality).to(
            device=states[0].device, dtype=states[0].dtype
        )
        reference_scale = getattr(self, f"reference_filter_scale_{modality}")
        relation_scale = getattr(self, f"relation_filter_scale_{modality}")
        reference_residual = reference_scale * alpha_centered
        if bool(getattr(self, "_reference_intervention_off", False)):
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
        relation_off = relation_intervention == "off"
        relation = self._relation_edge_outputs(
            h0, edge_index, modality, force_unit=relation_off
        )
        local_adaptation = self._local_adaptation(
            edge_index, relation["relation_residual"], relation["beta"],
            relation["nonself_mask"], h0.size(0)
        )
        if relation_off:
            local_adaptation = torch.zeros_like(local_adaptation)
        norm_index, norm_weight = self._normalized_operator(
            edge_index, relation["relation_weight"], h0.size(0), h0.dtype
        )
        semantic = self._semantic_states(h0, norm_index, norm_weight, modality)
        deltas, tokens, delta_gate, injections = self._context_tokens(
            semantic["states"], modality
        )
        s_tilde, attention, interaction_gate, interaction_output = self._cross_hop_interaction(
            semantic["states"], tokens, modality,
            interaction_off=interaction_intervention == "off",
            capture_attention=capture_attention,
        )
        filtering = self._filter_contexts(s_tilde, semantic, local_adaptation, modality)
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

    def _analysis_public(self, *args, **kwargs):
        result = super()._analysis_public(*args, **kwargs)
        result["beta_present"] = False
        result["r1u"] = False
        for modality in ("text", "visual"):
            result[f"beta_present_{modality}"] = False
        return result


Model = SSIMAGFinal
