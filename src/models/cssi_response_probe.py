from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .cosi_mag_ablation import CoSIMAGAblation


RESPONSE_COEFFICIENTS = (0.75, 0.50, 0.25)


class CSSIResponseProbe(CoSIMAGAblation):
    """Analysis-only view of the canonical ``all_plain`` backbone.

    This class intentionally adds no trainable parameters and does not alter
    the inherited forward path.  It only exposes the ordinary state bank and
    re-applies the frozen refinement/fusion stack to counterfactual plain
    modality representations.
    """

    def __init__(self, cfg, data_info: dict):
        if str(cfg.model.get("ablation_mode", "")).strip().lower() != "all_plain":
            raise ValueError("CSSI response analysis requires ablation_mode=all_plain")
        super().__init__(cfg, data_info)

    @staticmethod
    def response_bank(states: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(states) != len(RESPONSE_COEFFICIENTS) + 1:
            raise ValueError(
                "response analysis expects S0..S3; got " f"{len(states)} states"
            )
        return [states[order] - states[order - 1] for order in range(1, len(states))]

    @staticmethod
    def _max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left - right).abs().max().item()) if left.numel() else 0.0

    @torch.no_grad()
    def response_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        check_equivalences: bool = True,
        tolerance: float = 2.0e-5,
    ) -> dict[str, Any]:
        """Return frozen plain states and validate their algebraic identities."""
        if self.ablation_mode != "all_plain":
            raise RuntimeError("response_components is defined only for all_plain")
        components = self._encode_components(
            x,
            self._edge_index_or_empty(edge_index, x.device),
        )
        states_text = components["states_text"]
        states_visual = components["states_visual"]
        responses_text = self.response_bank(states_text)
        responses_visual = self.response_bank(states_visual)

        def validate(
            h0: torch.Tensor,
            states: list[torch.Tensor],
            responses: list[torch.Tensor],
            normalized_index: torch.Tensor,
            normalized_weight: torch.Tensor,
        ) -> dict[str, float]:
            recurrence_error = 0.0
            reconstruction_error = 0.0
            operator_response_error = 0.0
            for order, response in enumerate(responses, start=1):
                recurrence = self._propagate_once(
                    states[order - 1], normalized_index, normalized_weight
                )
                recurrence_error = max(
                    recurrence_error,
                    self._max_abs_difference(states[order], recurrence),
                )
                reconstruction_error = max(
                    reconstruction_error,
                    self._max_abs_difference(
                        response, states[order] - states[order - 1]
                    ),
                )
                operator_response_error = max(
                    operator_response_error,
                    self._max_abs_difference(
                        response, recurrence - states[order - 1]
                    ),
                )
            z_from_responses = h0.clone()
            for coefficient, response in zip(RESPONSE_COEFFICIENTS, responses):
                z_from_responses = z_from_responses + coefficient * response
            z_plain = torch.stack(states, dim=1).mean(dim=1)
            plain_error = self._max_abs_difference(z_plain, z_from_responses)
            return {
                "recurrence_max_abs_error": recurrence_error,
                "response_reconstruction_max_abs_error": reconstruction_error,
                "operator_response_max_abs_error": operator_response_error,
                "plain_representation_max_abs_error": plain_error,
            }

        text_checks = validate(
            components["h0_text"],
            states_text,
            responses_text,
            components["normalized_edge_index_text"],
            components["normalized_edge_weight_text"],
        )
        visual_checks = validate(
            components["h0_visual"],
            states_visual,
            responses_visual,
            components["normalized_edge_index_visual"],
            components["normalized_edge_weight_visual"],
        )
        operator_index_equal = bool(
            torch.equal(
                components["normalized_edge_index_text"],
                components["normalized_edge_index_visual"],
            )
        )
        operator_weight_error = self._max_abs_difference(
            components["normalized_edge_weight_text"],
            components["normalized_edge_weight_visual"],
        )
        checks = {
            "operator_index_equal": operator_index_equal,
            "operator_weight_max_abs_error": operator_weight_error,
            "text": text_checks,
            "visual": visual_checks,
        }
        if check_equivalences:
            errors = [
                text_checks["recurrence_max_abs_error"],
                visual_checks["recurrence_max_abs_error"],
                text_checks["response_reconstruction_max_abs_error"],
                visual_checks["response_reconstruction_max_abs_error"],
                text_checks["operator_response_max_abs_error"],
                visual_checks["operator_response_max_abs_error"],
                text_checks["plain_representation_max_abs_error"],
                visual_checks["plain_representation_max_abs_error"],
                operator_weight_error,
            ]
            if not operator_index_equal or max(errors, default=0.0) > tolerance:
                raise AssertionError(
                    "all_plain response equivalence check failed: " f"{checks}"
                )

        components = dict(components)
        components["responses_text"] = responses_text
        components["responses_visual"] = responses_visual
        components["equivalence_checks"] = checks
        components["z_plain_text"] = torch.stack(states_text, dim=1).mean(dim=1)
        components["z_plain_visual"] = torch.stack(states_visual, dim=1).mean(dim=1)
        return components

    def refine_and_fuse_plain(
        self, z_text: torch.Tensor, z_visual: torch.Tensor
    ) -> torch.Tensor:
        """Apply the exact frozen modality refinement and late fusion stack."""
        z_text_refined = self.text_refine_norm(
            z_text + self.text_refine_mlp(z_text)
        )
        z_visual_refined = self.visual_refine_norm(
            z_visual + self.visual_refine_mlp(z_visual)
        )
        fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
        return self.output_norm(
            self.fusion_skip(fused_input) + self.fusion_mlp(fused_input)
        )

    def logits_from_plain_modalities(
        self,
        classifier: nn.Module,
        z_text: torch.Tensor,
        z_visual: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute logits through refinement, late fusion, and the head."""
        return classifier(self.refine_and_fuse_plain(z_text, z_visual))

    @torch.no_grad()
    def leave_one_response_out_logits(
        self,
        classifier: nn.Module,
        components: dict[str, Any],
        modality: str | None = None,
        order: int | None = None,
    ) -> torch.Tensor:
        """Recompute frozen-forward logits after removing one response.

        With ``modality=None`` and ``order=None`` this is the base case and
        must reproduce the normal all_plain logits exactly up to float32
        arithmetic.  No logits are edited directly.
        """
        z_text = components["z_plain_text"]
        z_visual = components["z_plain_visual"]
        if (modality is None) != (order is None):
            raise ValueError("modality and order must be both set or both unset")
        if modality is not None:
            modality = str(modality).strip().lower()
            if modality not in {"text", "visual"}:
                raise ValueError("modality must be text or visual")
            if order not in {1, 2, 3}:
                raise ValueError("order must be one of 1, 2, 3")
            response = components[f"responses_{modality}"][int(order) - 1]
            updated = (z_text if modality == "text" else z_visual) - (
                RESPONSE_COEFFICIENTS[int(order) - 1] * response
            )
            if modality == "text":
                z_text = updated
            else:
                z_visual = updated
        return self.logits_from_plain_modalities(classifier, z_text, z_visual)


Model = CSSIResponseProbe
