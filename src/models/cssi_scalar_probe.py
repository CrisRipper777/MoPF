from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .cssi_v0 import CSSIV0, RESPONSE_COEFFICIENTS


SCALAR_VARIANTS = frozenset(
    {"static_modality", "static_order", "self_scalar", "same_scalar_mrc"}
)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class CSSIScalarProbe(CSSIV0):
    """Minimal scalar-control hierarchy for the P5 conditionality test.

    The encoder, unit-edge relation path, explicit S0...S3 bank, response
    tokens, modality refinement, late fusion, and classifier interface come
    from CSSIV0.  This class replaces only the response correction with one of
    four deliberately small scalar controls.
    """

    def __init__(self, cfg, data_info: dict[str, Any]):
        model_cfg = cfg.model
        variant = str(model_cfg.get("scalar_variant", "static_modality")).lower()
        if variant not in SCALAR_VARIANTS:
            raise ValueError(
                f"scalar_variant must be one of {sorted(SCALAR_VARIANTS)}, got {variant}"
            )
        if variant == "self_scalar":
            condition_mode = "self"
        elif variant == "same_scalar_mrc":
            condition_mode = "same"
        else:
            condition_mode = "none"

        # CSSIV0 supplies the unchanged backbone and response-token modules.
        # Disable its P3 adapter at construction; this class owns all scalar
        # correction parameters and never calls the P3 response adapter.
        base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        base_cfg.model.condition_mode = condition_mode
        base_cfg.model.adapter_type = "none"
        base_cfg.model.adapter_enabled = False
        base_cfg.model.response_correction_scale = 1.0
        super().__init__(base_cfg, data_info)

        self.scalar_variant = variant
        self.condition_mode = condition_mode
        self.gate_max = float(model_cfg.get("gate_max", 0.1))
        self.lambda_max = float(model_cfg.get("lambda_max", 0.2))
        self.lambda_init = float(model_cfg.get("lambda_init", 0.05))
        if self.gate_max <= 0.0:
            raise ValueError("gate_max must be positive")
        if self.lambda_max <= 0.0 or not 0.0 < self.lambda_init < self.lambda_max:
            raise ValueError("lambda_init must lie strictly between zero and lambda_max")

        if variant == "static_modality":
            self.static_gate_theta = nn.Parameter(torch.zeros(2))
        elif variant == "static_order":
            self.static_gate_theta = nn.Parameter(torch.zeros(2, self.max_order))
        else:
            init_theta = _logit(self.lambda_init / self.lambda_max)
            self.lambda_theta_text = nn.Parameter(
                torch.full((self.max_order,), init_theta)
            )
            self.lambda_theta_visual = nn.Parameter(
                torch.full((self.max_order,), init_theta)
            )

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
        if modality == "text":
            target_projection = self.control_proj_text
            other_projection = self.control_proj_visual
            self_controller = self.self_controller_text
            same_controller = self.same_controller_text
        else:
            target_projection = self.control_proj_visual
            other_projection = self.control_proj_text
            self_controller = self.self_controller_visual
            same_controller = self.same_controller_visual

        target_q = target_projection(target_tokens)
        if cross_intervention == "off":
            # Zero after Linear projection so its bias cannot reintroduce
            # paired evidence.
            other_q = torch.zeros_like(target_q)
        else:
            if cross_intervention == "shuffle":
                if cross_permutation is None:
                    raise ValueError("cross shuffle requires a node permutation")
                other_tokens = other_tokens.index_select(0, cross_permutation)
            other_q = other_projection(other_tokens)

        if self.scalar_variant in {"static_modality", "static_order"}:
            condition = torch.zeros_like(target_q)
        elif self.scalar_variant == "self_scalar":
            condition = self_controller(target_tokens)
        else:
            product = target_q * other_q
            difference = (target_q - other_q).abs()
            condition = same_controller(
                torch.cat((target_q, other_q, product, difference), dim=-1)
            )
        return condition, None, target_q, other_q

    def _lambda_strength(self, modality: str) -> torch.Tensor:
        theta = (
            self.lambda_theta_text
            if modality == "text"
            else self.lambda_theta_visual
        )
        return self.lambda_max * torch.sigmoid(theta)

    def _gate_matrix(
        self, condition: torch.Tensor, modality: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = condition.size(0)
        if self.scalar_variant == "static_modality":
            modality_index = 0 if modality == "text" else 1
            value = self.gate_max * torch.tanh(self.static_gate_theta[modality_index])
            gate = value.expand(nodes, self.max_order)
            amplitude = gate
        elif self.scalar_variant == "static_order":
            modality_index = 0 if modality == "text" else 1
            value = self.gate_max * torch.tanh(self.static_gate_theta[modality_index])
            gate = value.unsqueeze(0).expand(nodes, -1)
            amplitude = gate
        else:
            controller = (
                self.scalar_controller_text
                if modality == "text"
                else self.scalar_controller_visual
            )
            amplitude = torch.tanh(controller(condition)).squeeze(-1)
            gate = amplitude * self._lambda_strength(modality).unsqueeze(0)
        return gate, amplitude

    def _adapt_responses(
        self,
        responses: list[torch.Tensor],
        condition: torch.Tensor,
        modality: str,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        gate, amplitude = self._gate_matrix(condition, modality)
        corrections = [
            gate[:, order].unsqueeze(-1) * response
            for order, response in enumerate(responses)
        ]
        if self.scalar_variant in {"static_modality", "static_order"}:
            lambda_value = gate.new_full((self.max_order,), self.gate_max)
        else:
            lambda_value = self._lambda_strength(modality)
        return corrections, {
            "controller_gate": gate,
            "modulation_amplitude": amplitude,
            "lambda": lambda_value,
        }

    def _encode_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        cross_intervention: str = "normal",
        cross_permutation: torch.Tensor | None = None,
        capture_attention: bool = False,
    ) -> dict[str, Any]:
        components = super()._encode_components(
            x,
            edge_index,
            cross_intervention=cross_intervention,
            cross_permutation=cross_permutation,
            capture_attention=capture_attention,
        )
        # CSSIV0 stores the adapter gate under controller_gate.  Give the
        # analysis path an explicit scalar name and retain the unscaled
        # amplitude for diagnostics.
        components["scalar_gate_text"] = components["controller_gate_text"]
        components["scalar_gate_visual"] = components["controller_gate_visual"]
        for modality in ("text", "visual"):
            gate = components[f"scalar_gate_{modality}"]
            if self.scalar_variant in {"static_modality", "static_order"}:
                amplitude = gate
            else:
                amplitude = gate / self._lambda_strength(modality).unsqueeze(0)
            components[f"scalar_amplitude_{modality}"] = amplitude
        return components

    def _training_aux_info(self, components: dict[str, Any]) -> dict[str, torch.Tensor]:
        return super()._training_aux_info(components)


Model = CSSIScalarProbe
