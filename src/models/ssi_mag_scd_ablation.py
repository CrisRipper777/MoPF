from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .ssi_mag_scd import SSIMAGSCD


class SSIMAGSCDABlation(SSIMAGSCD):
    """Final-S formula ablations with the frozen S graph as the reference.

    The constructor temporarily exposes ``ablation=full`` to the historical
    parent constructor because the frozen S base only accepts Full.  The
    S-specific switch is restored immediately after construction and is used
    only by this subclass's stage-local formula overrides.
    """

    ABLATIONS = {
        "full",
        "no_semantic_conductance",
        "fixed_restart",
        "terminal_context_only",
        "global_response_only",
        "no_formation_conditioning",
    }

    def __init__(self, cfg, data_info: dict):
        configured = str(cfg.get("ablation", "full")).strip().lower()
        if configured not in self.ABLATIONS:
            configured = str(cfg.model.get("ablation", "full")).strip().lower()
        if configured not in self.ABLATIONS:
            raise ValueError(f"unknown S-specific ablation: {configured!r}")
        root_value = cfg.get("ablation", "full")
        model_value = cfg.model.get("ablation", "full")
        cfg.ablation = "full"
        cfg.model.ablation = "full"
        try:
            super().__init__(cfg, data_info)
        finally:
            cfg.ablation = root_value
            cfg.model.ablation = model_value
        self.ablation = configured
        if self.ablation == "fixed_restart" and abs(float(self.semantic_reference_init) - 0.10) > 1e-12:
            raise ValueError("fixed_restart requires semantic_reference_init=0.10")

    def _semantic_states(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        modality: str,
    ) -> dict[str, Any]:
        if self.ablation != "fixed_restart":
            return super()._semantic_states(h0, edge_index, edge_weight, modality)
        prior_norm = getattr(self, f"semantic_prior_norm_{modality}")
        prior_vector = getattr(self, f"semantic_prior_vector_{modality}")
        h = prior_norm(h0)
        p = (h * prior_vector).sum(dim=-1) / prior_vector.norm(p=2).clamp_min(self.eps)
        states = [h0]
        propagated: list[torch.Tensor] = []
        changes: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        current = h0
        alpha_value = float(self.semantic_reference_init)
        for _order in range(1, self.max_order + 1):
            q = self._propagate_once(current, edge_index, edge_weight)
            d = 1.0 - F.cosine_similarity(q, h0, dim=-1, eps=self.eps)
            alpha = torch.full_like(d, alpha_value)
            s = (1.0 - alpha).unsqueeze(-1) * q + alpha.unsqueeze(-1) * h0
            propagated.append(q)
            changes.append(d)
            alphas.append(alpha)
            states.append(s)
            current = s
        return {
            "states": states,
            "propagated": propagated,
            "changes": changes,
            "alphas": alphas,
            "p": p,
        }

    def _filter_contexts(
        self,
        states: list[torch.Tensor],
        semantic: dict[str, Any],
        local_adaptation: torch.Tensor,
        modality: str,
    ) -> dict[str, torch.Tensor]:
        result = super()._filter_contexts(states, semantic, local_adaptation, modality)
        gamma = self.gamma_global.to(dtype=states[0].dtype).unsqueeze(0)
        delta_gamma = getattr(self, f"delta_gamma_{modality}").unsqueeze(0)
        if self.ablation == "global_response_only":
            result["delta_content"] = torch.zeros_like(result["delta_content"])
            result["reference_residual"] = torch.zeros_like(result["reference_residual"])
            result["relation_residual"] = torch.zeros_like(result["relation_residual"])
            result["delta"] = torch.zeros_like(result["delta"])
            result["eta"] = gamma + delta_gamma
            result["eta_adaptive"] = result["eta"]
        elif self.ablation == "no_formation_conditioning":
            result["reference_residual"] = torch.zeros_like(result["reference_residual"])
            result["relation_residual"] = torch.zeros_like(result["relation_residual"])
            result["delta"] = result["delta_content"]
            result["eta"] = gamma + delta_gamma + result["delta_content"]
            result["eta_adaptive"] = result["eta"]
        return result

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
        if self.ablation != "terminal_context_only":
            relation_arg = "off" if self.ablation == "no_semantic_conductance" else relation_intervention
            return super()._encode_modality(
                h0,
                edge_index,
                modality,
                relation_intervention=relation_arg,
                interaction_intervention=interaction_intervention,
                capture_attention=capture_attention,
            )

        # Composite baseline: form the complete S0...SK bank, but only the
        # terminal state reaches the final representation. Earlier contexts
        # are retained only as inputs to S_K, never as final summands.
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
        base_filtering = self._filter_contexts(
            states, semantic, local_adaptation, modality
        )
        zero_reference = torch.zeros_like(base_filtering["reference_residual"])
        terminal_eta = torch.zeros_like(base_filtering["eta"])
        terminal_eta[:, -1] = (
            self.gamma_global[-1].to(dtype=h0.dtype)
            + getattr(self, f"delta_gamma_{modality}")[-1].to(dtype=h0.dtype)
            + base_filtering["delta_content"][:, -1]
            + base_filtering["relation_residual"][:, -1]
        )
        filtering = {
            **base_filtering,
            "reference_residual": zero_reference,
            "delta": base_filtering["delta_content"]
            + base_filtering["relation_residual"],
            "eta_adaptive": terminal_eta,
            "eta": terminal_eta,
        }
        z = terminal_eta[:, -1:].mul(states[-1]).squeeze(1)
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
        if self.ablation != "terminal_context_only":
            return super()._analysis_public(
                x, edge_index, relation=relation, interaction=interaction, reference=reference
            )

        relation = str(relation).strip().lower()
        interaction = str(interaction).strip().lower()
        reference = str(reference).strip().lower()
        if relation not in {"normal", "off"} or interaction not in {"normal", "off"}:
            raise ValueError("relation and interaction must be normal|off")
        if reference not in {"normal", "off"}:
            raise ValueError("reference must be normal|off")
        components = self._analysis_call(x, edge_index, relation=relation, interaction=interaction)
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
            "terminal_context_only": True,
        }
        for modality in ("text", "visual"):
            branch = components[modality]
            semantic = branch["semantic"]
            filtering = branch["filtering"]
            relation_stats = self._edge_summary(branch["relation_residual"])
            weight_stats = self._edge_summary(branch["relation_weight"])
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
                    f"term_p_{modality}": getattr(self, f"semantic_rho_p_{modality}") * semantic["p"],
                    f"d_{modality}": semantic["changes"],
                    f"term_d_{modality}": [
                        getattr(self, f"semantic_rho_d_{modality}") * value
                        for value in semantic["changes"]
                    ],
                    f"semantic_bias_{modality}": getattr(self, f"semantic_bias_{modality}"),
                    f"semantic_rho_p_{modality}": getattr(self, f"semantic_rho_p_{modality}"),
                    f"semantic_rho_d_{modality}": getattr(self, f"semantic_rho_d_{modality}"),
                    f"gamma_{modality}": self.gamma_global,
                    f"delta_gamma_{modality}": getattr(self, f"delta_gamma_{modality}"),
                    f"delta_content_{modality}": filtering["delta_content"],
                    f"reference_residual_{modality}": filtering["reference_residual"],
                    f"relation_filter_residual_{modality}": filtering["relation_residual"],
                    f"relation_residual_eta_{modality}": filtering["relation_residual"],
                    f"eta_{modality}": filtering["eta"],
                    f"eta_terminal_{modality}": filtering["eta"][:, -1],
                    f"effective_order_{modality}": self._effective_order(filtering["eta"]),
                    f"effective_radius_{modality}": self._effective_order(filtering["eta"]),
                    f"relation_profile_{modality}": filtering["relation_profile"],
                    f"alpha_{modality}": semantic["alphas"],
                    f"Q_{modality}": semantic["propagated"],
                    f"S_{modality}": semantic["states"],
                    f"S_used_{modality}": branch["s_used"],
                    f"normalized_edge_index_{modality}": branch["normalized_edge_index"],
                    f"normalized_edge_weight_{modality}": branch["normalized_edge_weight"],
                }
            )
        result["beta_present"] = False
        result["r1u"] = False
        for modality in ("text", "visual"):
            result[f"beta_present_{modality}"] = False
        return self._clone_value(result)


Model = SSIMAGSCDABlation
