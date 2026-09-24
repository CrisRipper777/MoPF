from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .ssi_mag_final import SSIMAGFinal


class SSIMAGFinalAblation(SSIMAGFinal):
    """Paper-facing NC ablations around the frozen canonical Full graph.

    The canonical class remains untouched. Every switch is implemented as an
    explicit stage-local override and keeps the same initialization, protocol,
    modality separation, and late-fusion head.
    """

    ABLATIONS = {
        "full",
        "no_relation_modulation",
        "fixed_semantic_reference",
        "last_context_only",
        "no_context_change",
        "no_cross_hop_interaction",
        "global_filter_only",
        "no_formation_conditioning",
    }

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info)
        configured = cfg.get("ablation", "full") if hasattr(cfg, "get") else "full"
        if configured in {None, "full"}:
            configured = cfg.model.get("ablation", "full")
        self.ablation = str(configured or "full").strip().lower()
        if self.ablation not in self.ABLATIONS:
            raise ValueError(f"unknown final paper ablation: {self.ablation}")

    def _semantic_states(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        modality: str,
    ) -> dict[str, Any]:
        if self.ablation != "fixed_semantic_reference":
            return super()._semantic_states(h0, edge_index, edge_weight, modality)
        prior_norm = getattr(self, f"semantic_prior_norm_{modality}")
        prior_vector = getattr(self, f"semantic_prior_vector_{modality}")
        h = prior_norm(h0)
        p = (h * prior_vector).sum(dim=-1) / prior_vector.norm(p=2).clamp_min(self.eps)
        states = [h0]
        propagated, changes, alphas = [], [], []
        current = h0
        alpha_value = torch.as_tensor(
            self.semantic_reference_init, dtype=h0.dtype, device=h0.device
        )
        for _order in range(1, self.max_order + 1):
            q = self._propagate_once(current, edge_index, edge_weight)
            d = 1.0 - F.cosine_similarity(q, h0, dim=-1, eps=self.eps)
            alpha = torch.full_like(d, self.semantic_reference_init)
            s = (1.0 - alpha).unsqueeze(-1) * q + alpha.unsqueeze(-1) * h0
            propagated.append(q)
            changes.append(d)
            alphas.append(alpha)
            states.append(s)
            current = s
        return {"states": states, "propagated": propagated, "changes": changes, "alphas": alphas, "p": p}

    def _context_tokens(self, states: list[torch.Tensor], modality: str):
        deltas, tokens, delta_gate, injections = super()._context_tokens(states, modality)
        if self.ablation != "no_context_change":
            return deltas, tokens, delta_gate, injections
        order_embedding = getattr(self, f"hop_order_embedding_{modality}")
        tokens = [
            F.layer_norm(state, (self.hidden_dim,)) + order_embedding[index].unsqueeze(0)
            for index, state in enumerate(states)
        ]
        zeros = [torch.zeros_like(value) for value in injections]
        return deltas, tokens, torch.zeros_like(delta_gate), zeros

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
        if self.ablation == "global_filter_only":
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
        relation_off = relation_intervention == "off" or self.ablation == "no_relation_modulation"
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
        deltas, tokens, delta_gate, injections = self._context_tokens(semantic["states"], modality)
        interaction_off = (
            interaction_intervention == "off"
            or self.ablation in {"no_cross_hop_interaction", "last_context_only"}
        )
        s_tilde, attention, interaction_gate, interaction_output = self._cross_hop_interaction(
            semantic["states"], tokens, modality,
            interaction_off=interaction_off, capture_attention=capture_attention,
        )
        filtering = self._filter_contexts(s_tilde, semantic, local_adaptation, modality)
        if self.ablation == "last_context_only":
            # The terminal branch is recomposed from terminal-hop quantities
            # only. Earlier S/Q/alpha entries cannot enter through attention
            # or the filter; the terminal reference residual is defined as
            # zero because there is no multi-hop centering bank in this mode.
            content = torch.zeros_like(filtering["delta_content"])
            content[:, -1] = filtering["delta_content"][:, -1]
            relation_residual = torch.zeros_like(filtering["relation_residual"])
            relation_residual[:, -1] = filtering["relation_residual"][:, -1]
            reference_residual = torch.zeros_like(filtering["reference_residual"])
            gamma = self.gamma_global.to(dtype=filtering["eta"].dtype).unsqueeze(0)
            delta_gamma = getattr(self, f"delta_gamma_{modality}").unsqueeze(0)
            eta = torch.zeros_like(filtering["eta"])
            eta[:, -1] = gamma[:, -1] + delta_gamma[:, -1] + content[:, -1] + relation_residual[:, -1]
            filtering["delta_content"] = content
            filtering["reference_residual"] = reference_residual
            filtering["relation_residual"] = relation_residual
            filtering["delta"] = content + relation_residual
            filtering["eta"] = eta
            filtering["eta_adaptive"] = eta
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


Model = SSIMAGFinalAblation
