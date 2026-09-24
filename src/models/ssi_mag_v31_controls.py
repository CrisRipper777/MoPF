from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssi_mag_v3 import SSIMAGV3, _make_mlp
from .ssi_mag_v31 import SSIMAGV31


class SSIMAGV31Controls(SSIMAGV31):
    """Independent P1.7c attribution controls.

    ``r2_only`` is B (historical V3 R1 + V3.1 R2 + old reference residual).
    ``r1_r2`` is AB (V3.1 R1 + V3.1 R2 + old reference residual).

    The frozen V3 and V3.1 implementations are intentionally not modified.
    """

    CONTROLS = {"r2_only", "r1_r2"}

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info)
        configured = cfg.model.get("control", "r2_only")
        self.control = str(configured).strip().lower()
        if self.control not in self.CONTROLS:
            raise ValueError(
                f"unknown SSI-MAG-V3.1 control {self.control!r}; "
                f"expected {', '.join(sorted(self.CONTROLS))}"
            )

        # Both controls restore the historical Stage-II reference residual.
        self.reference_filter_scale_init = float(
            cfg.model.get("reference_filter_scale_init", 0.10)
        )
        self.reference_filter_scale_text = nn.Parameter(
            torch.tensor(self.reference_filter_scale_init)
        )
        self.reference_filter_scale_visual = nn.Parameter(
            torch.tensor(self.reference_filter_scale_init)
        )

        if self.control == "r2_only":
            # Replace the inherited V3.1 R1 modules by the historical V3
            # projection/scorer shapes and initialization path.  The actual
            # relation calculation below delegates to SSIMAGV3 verbatim.
            self.relation_proj_text = nn.Linear(self.hidden_dim, self.relation_rank)
            self.relation_proj_visual = nn.Linear(self.hidden_dim, self.relation_rank)
            descriptor_dim = 2 * self.relation_rank + 1
            self.relation_scorer_text = _make_mlp(
                descriptor_dim, self.relation_rank, 1, 0.0
            )
            self.relation_scorer_visual = _make_mlp(
                descriptor_dim, self.relation_rank, 1, 0.0
            )

    def _relation_edge_outputs(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        modality: str,
        *,
        force_unit: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.control == "r1_r2":
            return super()._relation_edge_outputs(
                h0, edge_index, modality, force_unit=force_unit
            )

        # Historical V3 R1 formula, including its descriptor, scorer, beta,
        # edge weight, and local-adaptation-compatible residual definition.
        relation_projection, residual = SSIMAGV3._relation_descriptor(
            self, h0, edge_index, modality
        )
        theta_beta = getattr(self, f"theta_beta_{modality}")
        beta_parameter = torch.sigmoid(theta_beta)
        beta = torch.zeros_like(beta_parameter) if force_unit else beta_parameter
        relation_weight = torch.exp(beta * residual)
        nonself_mask = (
            edge_index[0].ne(edge_index[1])
            if edge_index.numel()
            else torch.zeros(0, dtype=torch.bool, device=edge_index.device)
        )
        if edge_index.numel():
            src, dst = edge_index
            compatibility = F.cosine_similarity(
                relation_projection[src], relation_projection[dst], dim=-1, eps=self.eps
            )
        else:
            compatibility = h0.new_empty((0,))
        return {
            "relation_projection": relation_projection,
            "compatibility": compatibility,
            "relation_mu": h0.new_zeros((h0.size(0),)),
            "relation_centered": residual,
            "relation_score": residual,
            "relation_residual": residual,
            "beta": beta,
            "beta_parameter": beta_parameter,
            "relation_weight": relation_weight,
            "nonself_mask": nonself_mask,
        }

    @staticmethod
    def _local_adaptation(
        edge_index: torch.Tensor,
        relation_residual: torch.Tensor,
        beta: torch.Tensor,
        nonself_mask: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        del nonself_mask
        # B must retain historical V3 c_i exactly, including the physical
        # edge support supplied to the old implementation.
        return SSIMAGV3._local_adaptation(
            edge_index, relation_residual, beta, num_nodes
        )

    def _filter_contexts(
        self,
        states: list[torch.Tensor],
        semantic: dict[str, Any],
        local_adaptation: torch.Tensor,
        modality: str,
    ) -> dict[str, torch.Tensor]:
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
        eta_adaptive = (
            gamma
            + delta_gamma
            + delta_content
            + reference_residual
            + relation_residual
        )
        return {
            "alpha_bank": alpha_bank,
            "alpha_centered": alpha_centered,
            "relation_profile": relation_profile,
            "delta_content": delta_content,
            "reference_residual": reference_residual,
            "relation_residual": relation_residual,
            "delta": delta_content + reference_residual + relation_residual,
            "eta_adaptive": eta_adaptive,
            "eta": eta_adaptive,
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
        relation = self._relation_edge_outputs(
            h0,
            edge_index,
            modality,
            force_unit=relation_intervention == "off",
        )
        local_adaptation = self._local_adaptation(
            edge_index,
            relation["relation_residual"],
            relation["beta"],
            relation["nonself_mask"],
            h0.size(0),
        )
        norm_index, norm_weight = self._normalized_operator(
            edge_index, relation["relation_weight"], h0.size(0), h0.dtype
        )
        semantic = self._semantic_states(h0, norm_index, norm_weight, modality)
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
            s_tilde, semantic, local_adaptation, modality
        )
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
        result["control"] = self.control
        result["reference_intervention"] = reference
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


Model = SSIMAGV31Controls
