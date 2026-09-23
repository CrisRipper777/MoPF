from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cosi_mag_final import CoSIMAGFinal


RESPONSE_COEFFICIENTS = (0.75, 0.50, 0.25)
CONDITION_MODES = frozenset({"self", "same", "all", "none"})
ADAPTER_TYPES = frozenset({"scalar", "lowrank", "none"})


class CSSIV0(CoSIMAGFinal):
    """Unified P3 conditional structural-response adaptation family.

    The inherited CoSI-MAG modules provide the projectors, GCN normalization,
    propagation, modality refinement, late fusion, and classifier interface.
    This class adds only the controller/adapter path used by P3; it does not
    alter the historical model implementations.
    """

    def __init__(self, cfg, data_info: dict):
        model_cfg = cfg.model
        self.use_mrc = bool(model_cfg.get("use_mrc", False))
        self.condition_mode = str(model_cfg.get("condition_mode", "self")).lower()
        self.adapter_type = str(model_cfg.get("adapter_type", "lowrank")).lower()
        self.control_dim = int(model_cfg.get("control_dim", 64))
        self.response_rank = int(model_cfg.get("response_rank", 8))
        self.response_correction_scale = float(
            model_cfg.get("response_correction_scale", 1.0e-3)
        )
        self.adapter_enabled = bool(model_cfg.get("adapter_enabled", True))
        if self.condition_mode not in CONDITION_MODES:
            raise ValueError(f"condition_mode must be one of {sorted(CONDITION_MODES)}")
        if self.adapter_type not in ADAPTER_TYPES:
            raise ValueError(f"adapter_type must be one of {sorted(ADAPTER_TYPES)}")
        if self.control_dim < 1 or self.control_dim > 128:
            raise ValueError("control_dim must be in [1, 128]")
        if self.response_rank < 1 or self.response_rank > 64:
            raise ValueError("response_rank must be in [1, 64]")
        if self.response_correction_scale <= 0.0:
            raise ValueError("response_correction_scale must be positive")
        if self.adapter_type == "none":
            self.adapter_enabled = False
        if not self.adapter_enabled:
            self.adapter_type = "none"
        if self.condition_mode == "none" and self.adapter_enabled:
            raise ValueError("an enabled adapter requires condition_mode self|same|all")
        configured_alpha = float(model_cfg.get("multihop_anchor_alpha", 0.0))
        if not math.isclose(configured_alpha, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("CSSIV0 requires multihop_anchor_alpha=0.0")
        if int(model_cfg.get("max_order", 3)) != 3:
            raise ValueError("CSSIV0 P3 is fixed to max_order=3")
        super().__init__(cfg, data_info)

        hidden_dim = self.hidden_dim
        self.response_order_embedding_text = nn.Parameter(
            torch.zeros(self.max_order, self.control_dim)
        )
        self.response_order_embedding_visual = nn.Parameter(
            torch.zeros(self.max_order, self.control_dim)
        )
        self.token_h_text = nn.Linear(hidden_dim, self.control_dim)
        self.token_s_text = nn.Linear(hidden_dim, self.control_dim)
        self.token_r_text = nn.Linear(hidden_dim, self.control_dim)
        self.token_h_visual = nn.Linear(hidden_dim, self.control_dim)
        self.token_s_visual = nn.Linear(hidden_dim, self.control_dim)
        self.token_r_visual = nn.Linear(hidden_dim, self.control_dim)
        self.token_norm_text = nn.LayerNorm(self.control_dim)
        self.token_norm_visual = nn.LayerNorm(self.control_dim)

        self.control_proj_text = nn.Linear(self.control_dim, self.control_dim)
        self.control_proj_visual = nn.Linear(self.control_dim, self.control_dim)
        self.self_controller_text = nn.Sequential(
            nn.Linear(self.control_dim, self.control_dim),
            nn.GELU(),
            nn.Linear(self.control_dim, self.control_dim),
        )
        self.self_controller_visual = nn.Sequential(
            nn.Linear(self.control_dim, self.control_dim),
            nn.GELU(),
            nn.Linear(self.control_dim, self.control_dim),
        )
        same_input_dim = 4 * self.control_dim
        self.same_controller_text = nn.Sequential(
            nn.Linear(same_input_dim, self.control_dim),
            nn.GELU(),
            nn.Linear(self.control_dim, self.control_dim),
        )
        self.same_controller_visual = nn.Sequential(
            nn.Linear(same_input_dim, self.control_dim),
            nn.GELU(),
            nn.Linear(self.control_dim, self.control_dim),
        )
        self.all_query_text = nn.Linear(self.control_dim, self.control_dim)
        self.all_key_text = nn.Linear(self.control_dim, self.control_dim)
        self.all_value_text = nn.Linear(self.control_dim, self.control_dim)
        self.all_query_visual = nn.Linear(self.control_dim, self.control_dim)
        self.all_key_visual = nn.Linear(self.control_dim, self.control_dim)
        self.all_value_visual = nn.Linear(self.control_dim, self.control_dim)
        self.all_scale = float(self.control_dim) ** -0.5

        self.scalar_controller_text = nn.Sequential(
            nn.Linear(self.control_dim, self.control_dim),
            nn.GELU(),
            nn.Linear(self.control_dim, 1),
        )
        self.scalar_controller_visual = nn.Sequential(
            nn.Linear(self.control_dim, self.control_dim),
            nn.GELU(),
            nn.Linear(self.control_dim, 1),
        )
        self.scalar_correction_scale = nn.Parameter(torch.zeros(1))

        self.response_down_text = nn.Linear(hidden_dim, self.response_rank, bias=False)
        self.response_down_visual = nn.Linear(hidden_dim, self.response_rank, bias=False)
        self.response_condition_text = nn.Linear(self.control_dim, self.response_rank)
        self.response_condition_visual = nn.Linear(self.control_dim, self.response_rank)
        self.response_up_text = nn.Linear(self.response_rank, hidden_dim)
        self.response_up_visual = nn.Linear(self.response_rank, hidden_dim)
        nn.init.zeros_(self.response_up_text.weight)
        nn.init.zeros_(self.response_up_text.bias)
        nn.init.zeros_(self.response_up_visual.weight)
        nn.init.zeros_(self.response_up_visual.bias)

    def _relation_calibration(
        self, h_text: torch.Tensor, h_visual: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.use_mrc:
            return super()._relation_calibration(h_text, h_visual, edge_index)
        edge_count = int(edge_index.size(1))
        return {
            "semantic_cosine_text": h_text.new_zeros(edge_count),
            "semantic_cosine_visual": h_visual.new_zeros(edge_count),
            "relation_weight_text": h_text.new_ones(edge_count),
            "relation_weight_visual": h_visual.new_ones(edge_count),
            "metric_weights_text": h_text.new_ones((1, h_text.size(-1))),
            "metric_weights_visual": h_visual.new_ones((1, h_visual.size(-1))),
        }

    @staticmethod
    def _responses(states: list[torch.Tensor]) -> list[torch.Tensor]:
        return [states[order] - states[order - 1] for order in range(1, len(states))]

    def _response_tokens(
        self,
        h0: torch.Tensor,
        states: list[torch.Tensor],
        responses: list[torch.Tensor],
        modality: str,
    ) -> torch.Tensor:
        if modality == "text":
            h_layer = self.token_h_text
            s_layer = self.token_s_text
            r_layer = self.token_r_text
            norm = self.token_norm_text
            order_embedding = self.response_order_embedding_text
        else:
            h_layer = self.token_h_visual
            s_layer = self.token_s_visual
            r_layer = self.token_r_visual
            norm = self.token_norm_visual
            order_embedding = self.response_order_embedding_visual
        tokens = []
        for order, response in enumerate(responses):
            token = (
                h_layer(h0)
                + s_layer(states[order])
                + r_layer(response)
                + order_embedding[order].unsqueeze(0)
            )
            tokens.append(norm(token))
        return torch.stack(tokens, dim=1)

    def _condition(
        self,
        target_tokens: torch.Tensor,
        other_tokens: torch.Tensor,
        modality: str,
        *,
        cross_intervention: str,
        cross_permutation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        if cross_intervention not in {"normal", "off", "shuffle"}:
            raise ValueError("cross_intervention must be normal|off|shuffle")
        if cross_intervention == "off":
            other_tokens = torch.zeros_like(other_tokens)
        elif cross_intervention == "shuffle":
            if cross_permutation is None:
                raise ValueError("cross shuffle requires a node permutation")
            other_tokens = other_tokens.index_select(0, cross_permutation)
        if modality == "text":
            target_projection = self.control_proj_text
            other_projection = self.control_proj_visual
            self_controller = self.self_controller_text
            same_controller = self.same_controller_text
            q_layer = self.all_query_text
            k_layer = self.all_key_visual
            v_layer = self.all_value_visual
        else:
            target_projection = self.control_proj_visual
            other_projection = self.control_proj_text
            self_controller = self.self_controller_visual
            same_controller = self.same_controller_visual
            q_layer = self.all_query_visual
            k_layer = self.all_key_text
            v_layer = self.all_value_text
        target_q = target_projection(target_tokens)
        other_q = other_projection(other_tokens)
        if self.condition_mode == "none":
            condition = torch.zeros_like(target_q)
            return condition, None, target_q, other_q
        if self.condition_mode == "self":
            condition = self_controller(target_tokens)
            return condition, None, target_q, other_q
        if self.condition_mode == "same":
            product = target_q * other_q
            difference = (target_q - other_q).abs()
            condition = same_controller(
                torch.cat((target_q, other_q, product, difference), dim=-1)
            )
            return condition, None, target_q, other_q

        query = q_layer(target_q)
        key = k_layer(other_q)
        value = v_layer(other_q)
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.all_scale
        attention = torch.softmax(logits, dim=-1)
        condition = torch.matmul(attention, value)
        return condition, attention, target_q, other_q

    def _adapt_responses(
        self,
        responses: list[torch.Tensor],
        condition: torch.Tensor,
        modality: str,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        correction_list = []
        gate_list = []
        for order, response in enumerate(responses):
            c = condition[:, order, :]
            if self.adapter_type == "scalar" and self.adapter_enabled:
                controller = (
                    self.scalar_controller_text
                    if modality == "text"
                    else self.scalar_controller_visual
                )
                # controller(c) is [N, 1]; keep that singleton feature
                # dimension so multiplication with [N, hidden_dim] is
                # node-wise rather than accidentally expanding to [N, N, d].
                gate = torch.tanh(controller(c))
                correction = (
                    self.response_correction_scale
                    * self.scalar_correction_scale
                    * gate
                    * response
                )
                gate_list.append(gate.squeeze(-1))
            elif self.adapter_type == "lowrank" and self.adapter_enabled:
                down = (
                    self.response_down_text
                    if modality == "text"
                    else self.response_down_visual
                )
                condition_layer = (
                    self.response_condition_text
                    if modality == "text"
                    else self.response_condition_visual
                )
                up = (
                    self.response_up_text
                    if modality == "text"
                    else self.response_up_visual
                )
                latent = down(response)
                amplitude = torch.tanh(condition_layer(c))
                correction = self.response_correction_scale * up(amplitude * latent)
                gate_list.append(amplitude)
            else:
                correction = torch.zeros_like(response)
                gate_list.append(torch.zeros_like(c))
            correction_list.append(correction)
        corrections = torch.stack(correction_list, dim=1)
        gate = torch.stack(gate_list, dim=1)
        return [corrections[:, order, :] for order in range(corrections.size(1))], {
            "controller_gate": gate,
        }

    def _refine_and_fuse(
        self, z_text: torch.Tensor, z_visual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return z_text_refined, z_visual_refined, z

    def _encode_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        cross_intervention: str = "normal",
        cross_permutation: torch.Tensor | None = None,
        capture_attention: bool = False,
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
        states_text = self._multi_hop_states(h_text, norm_text_index, norm_text_weight)
        states_visual = self._multi_hop_states(
            h_visual, norm_visual_index, norm_visual_weight
        )
        responses_text = self._responses(states_text)
        responses_visual = self._responses(states_visual)
        tokens_text = self._response_tokens(
            h_text, states_text, responses_text, "text"
        )
        tokens_visual = self._response_tokens(
            h_visual, states_visual, responses_visual, "visual"
        )
        condition_text, attention_text, q_text, q_visual_for_text = self._condition(
            tokens_text,
            tokens_visual,
            "text",
            cross_intervention=cross_intervention,
            cross_permutation=cross_permutation,
        )
        condition_visual, attention_visual, q_visual, q_text_for_visual = self._condition(
            tokens_visual,
            tokens_text,
            "visual",
            cross_intervention=cross_intervention,
            cross_permutation=cross_permutation,
        )
        corrections_text, adapter_info_text = self._adapt_responses(
            responses_text, condition_text, "text"
        )
        corrections_visual, adapter_info_visual = self._adapt_responses(
            responses_visual, condition_visual, "visual"
        )
        z_base_text = torch.stack(states_text, dim=1).mean(dim=1)
        z_base_visual = torch.stack(states_visual, dim=1).mean(dim=1)
        z_text = z_base_text + sum(
            coefficient * correction
            for coefficient, correction in zip(RESPONSE_COEFFICIENTS, corrections_text)
        )
        z_visual = z_base_visual + sum(
            coefficient * correction
            for coefficient, correction in zip(RESPONSE_COEFFICIENTS, corrections_visual)
        )
        z_base_text_refined, z_base_visual_refined, z_base = self._refine_and_fuse(
            z_base_text, z_base_visual
        )
        z_text_refined, z_visual_refined, z = self._refine_and_fuse(z_text, z_visual)
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
            "responses_text": responses_text,
            "responses_visual": responses_visual,
            "response_tokens_text": tokens_text,
            "response_tokens_visual": tokens_visual,
            "control_text": condition_text,
            "control_visual": condition_visual,
            "control_q_text": q_text,
            "control_q_visual": q_visual,
            "control_q_visual_for_text": q_visual_for_text,
            "control_q_text_for_visual": q_text_for_visual,
            "attention_text": attention_text,
            "attention_visual": attention_visual,
            "corrections_text": corrections_text,
            "corrections_visual": corrections_visual,
            "controller_gate_text": adapter_info_text["controller_gate"],
            "controller_gate_visual": adapter_info_visual["controller_gate"],
            "z_base_text": z_base_text,
            "z_base_visual": z_base_visual,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_base_text_refined": z_base_text_refined,
            "z_base_visual_refined": z_base_visual_refined,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z_base": z_base,
            "z": z,
            "cross_intervention": cross_intervention,
        }

    @staticmethod
    def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            numerator / denominator.clamp_min(torch.finfo(numerator.dtype).eps),
            nan=0.0,
            posinf=1.0e6,
            neginf=-1.0e6,
        )

    def _training_aux_info(self, components: dict[str, Any]) -> dict[str, torch.Tensor]:
        ratios = []
        cosines = []
        for modality in ("text", "visual"):
            for response, correction in zip(
                components[f"responses_{modality}"],
                components[f"corrections_{modality}"],
            ):
                response_norm = response.norm(dim=-1)
                correction_norm = correction.norm(dim=-1)
                ratios.append(self._safe_ratio(correction_norm, response_norm).mean())
                cosines.append(
                    F.cosine_similarity(
                        correction, response, dim=-1, eps=1.0e-6
                    ).mean()
                )
        return {
            "mean_response_correction_ratio": torch.stack(ratios).mean(),
            "max_response_correction_ratio": torch.stack(ratios).max(),
            "mean_response_correction_cosine": torch.stack(cosines).mean(),
        }

    def gradient_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return monitoring-only gradient norm after backward, before clipping."""
        squared = []
        for parameter in self.parameters():
            if parameter.grad is not None:
                squared.append(parameter.grad.detach().float().square().sum())
        if not squared:
            value = torch.zeros((), device=next(self.parameters()).device)
        else:
            value = torch.stack(squared).sum().sqrt()
        return {"mean_gradient_norm": value}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)
        z = torch.nan_to_num(components["z"], nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        return z, None, None, z.new_zeros(()), self._training_aux_info(components)

    @torch.no_grad()
    def analysis_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        cross_intervention: str = "normal",
        permutation: torch.Tensor | None = None,
        capture_attention: bool = True,
    ) -> dict[str, Any]:
        training_states = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            edge_index = self._edge_index_or_empty(edge_index, x.device)
            components = self._encode_components(
                x,
                edge_index,
                cross_intervention=cross_intervention,
                cross_permutation=permutation,
                capture_attention=capture_attention,
            )
            return components
        finally:
            for module, was_training in training_states:
                module.training = was_training


Model = CSSIV0
