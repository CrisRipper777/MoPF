from __future__ import annotations

from typing import Any

import torch

from .ssi_mag_final import SSIMAGFinal


class SSIMAGSCD(SSIMAGFinal):
    """Clean Semantic-Controlled Diffusion model.

    The R1 relation conductance, R2 semantic-feedback restart, signed
    formation response, and late fusion are inherited from the frozen Final
    implementation.  The two P2 paths that were shown to be unnecessary are
    removed from the instantiated module and from the computation graph:
    context-change encoding and cross-hop interaction.

    The constructor intentionally calls the historical Final constructor
    before deleting the dead modules.  This preserves the historical global
    RNG/constructor order for every retained parameter while leaving no dead
    parameter in ``parameters()`` or ``state_dict()``.
    """

    ABLATIONS = {"full"}
    no_weight_decay_parameter_names = frozenset()
    context_change_present = False
    cross_hop_interaction_present = False

    # These are the exact module/parameter names created by SSIMAGV31 for the
    # two removed paths.  Keep this list explicit for checkpoint audits.
    DEAD_COMPONENT_NAMES = (
        "context_delta_norm_text",
        "context_delta_norm_visual",
        "context_delta_proj_text",
        "context_delta_proj_visual",
        "theta_delta_text",
        "theta_delta_visual",
        "hop_order_embedding_text",
        "hop_order_embedding_visual",
        "hop_layers_text",
        "hop_layers_visual",
        "theta_int_text",
        "theta_int_visual",
    )

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info)
        for name in self.DEAD_COMPONENT_NAMES:
            if not hasattr(self, name):
                raise RuntimeError(f"historical dead component was not constructed: {name}")
            delattr(self, name)

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
        """Encode one modality without constructing tokens or attention."""
        del interaction_intervention, capture_attention
        relation_off = relation_intervention == "off"
        relation = self._relation_edge_outputs(
            h0, edge_index, modality, force_unit=relation_off
        )
        local_adaptation = self._local_adaptation(
            edge_index,
            relation["relation_residual"],
            relation["beta"],
            relation["nonself_mask"],
            h0.size(0),
        )
        if relation_off:
            local_adaptation = torch.zeros_like(local_adaptation)
        norm_index, norm_weight = self._normalized_operator(
            edge_index, relation["relation_weight"], h0.size(0), h0.dtype
        )
        semantic = self._semantic_states(h0, norm_index, norm_weight, modality)
        states = semantic["states"]
        filtering = self._filter_contexts(states, semantic, local_adaptation, modality)
        z = self._compose(states, filtering["eta"])
        return {
            **relation,
            "normalized_edge_index": norm_index,
            "normalized_edge_weight": norm_weight,
            "local_adaptation": local_adaptation,
            "semantic": semantic,
            "filtering": filtering,
            "s_used": states,
            "z": z,
        }

    def _analysis_public(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        relation: str = "normal",
        interaction: str = "normal",
        reference: str = "normal",
    ) -> dict[str, Any]:
        relation = str(relation).strip().lower()
        interaction = str(interaction).strip().lower()
        reference = str(reference).strip().lower()
        if relation not in {"normal", "off"}:
            raise ValueError("relation must be normal|off")
        if interaction not in {"normal", "off"}:
            raise ValueError("interaction must be normal|off")
        if reference not in {"normal", "off"}:
            raise ValueError("reference must be normal|off")

        self._reference_intervention_off = reference == "off"
        try:
            components = self._analysis_call(
                x, edge_index, relation=relation, interaction=interaction
            )
        finally:
            self._reference_intervention_off = False

        result: dict[str, Any] = {
            "physical_edge_index": components["physical_edge_index"],
            "ablation": self.ablation,
            "h0_text": components["h0_text"],
            "h0_visual": components["h0_visual"],
            "z_text": components["z_text"],
            "z_visual": components["z_visual"],
            "z_text_refined": components["z_text_refined"],
            "z_visual_refined": components["z_visual_refined"],
            "fused_z": components["z"],
            "z": components["z"],
            "context_change_present": False,
            "cross_hop_interaction_present": False,
            "reference_intervention": reference,
        }
        for modality in ("text", "visual"):
            branch = components[modality]
            semantic = branch["semantic"]
            filtering = branch["filtering"]
            relation_stats = self._edge_summary(branch["relation_residual"])
            weight_stats = self._edge_summary(branch["relation_weight"])
            rho_p = getattr(self, f"semantic_rho_p_{modality}")
            rho_d = getattr(self, f"semantic_rho_d_{modality}")
            result.update(
                {
                    f"relation_z_{modality}": branch["relation_projection"],
                    f"z_relation_{modality}": branch["relation_projection"],
                    f"relation_compatibility_{modality}": branch["compatibility"],
                    f"relation_mu_{modality}": branch["relation_mu"],
                    f"relation_centered_{modality}": branch["relation_centered"],
                    f"a_{modality}": branch["relation_score"],
                    f"relation_residual_{modality}": branch["relation_residual"],
                    f"relation_weight_{modality}": branch["relation_weight"],
                    f"beta_{modality}": branch["beta"],
                    f"beta_parameter_{modality}": branch["beta_parameter"],
                    f"relation_residual_mean_{modality}": relation_stats["mean"],
                    f"relation_residual_std_{modality}": relation_stats["std"],
                    f"relation_residual_abs_mean_{modality}": relation_stats["abs_mean"],
                    f"relation_weight_mean_{modality}": weight_stats["mean"],
                    f"relation_weight_std_{modality}": weight_stats["std"],
                    f"relation_weight_min_{modality}": weight_stats["min"],
                    f"relation_weight_max_{modality}": weight_stats["max"],
                    f"local_adaptation_{modality}": branch["local_adaptation"],
                    f"c_{modality}": branch["local_adaptation"],
                    f"p_{modality}": semantic["p"],
                    f"term_p_{modality}": rho_p * semantic["p"],
                    f"d_{modality}": semantic["changes"],
                    f"term_d_{modality}": [rho_d * value for value in semantic["changes"]],
                    f"semantic_bias_{modality}": getattr(self, f"semantic_bias_{modality}"),
                    f"semantic_rho_p_{modality}": rho_p,
                    f"semantic_rho_d_{modality}": rho_d,
                    f"gamma_{modality}": self.gamma_global,
                    f"delta_gamma_{modality}": getattr(self, f"delta_gamma_{modality}"),
                    f"delta_content_{modality}": filtering["delta_content"],
                    f"reference_residual_{modality}": filtering["reference_residual"],
                    f"relation_filter_residual_{modality}": filtering["relation_residual"],
                    f"relation_residual_eta_{modality}": filtering["relation_residual"],
                    f"eta_{modality}": filtering["eta"],
                    f"effective_order_{modality}": self._effective_order(filtering["eta"]),
                    f"effective_radius_{modality}": self._effective_order(filtering["eta"]),
                    f"relation_profile_{modality}": filtering["relation_profile"],
                    f"normalized_edge_index_{modality}": branch["normalized_edge_index"],
                    f"normalized_edge_weight_{modality}": branch["normalized_edge_weight"],
                    f"Q_{modality}": semantic["propagated"],
                    f"S_{modality}": semantic["states"],
                    f"S_used_{modality}": branch["s_used"],
                    f"alpha_{modality}": semantic["alphas"],
                }
            )
        result["beta_present"] = False
        result["r1u"] = False
        result["reference_intervention"] = reference
        for modality in ("text", "visual"):
            result[f"beta_present_{modality}"] = False
        return self._clone_value(result)


Model = SSIMAGSCD
