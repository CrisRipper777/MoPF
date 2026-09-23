"""Functional ablations of the canonical MGSC-MAG P2 encoder.

The full variant delegates to :class:`MGSCMAG` and has the same parameter
shapes.  Every other variant changes one explicitly named computation while
keeping hidden size, refinement, fusion, task heads, and the NC protocol
unchanged.  This module is an analysis/training control, not a new full-model
architecture.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .mgsc_mag import MGSCMAG


ABLATIONS = (
    "full",
    "uniform_relations",
    "global_context_gate",
    "terminal_state_only",
    "uniform_integration",
    "no_cross_order_interaction",
    "attribute_only",
    "fixed_gate_09",
)


def _logit(probability: float) -> float:
    value = float(probability)
    return math.log(value / (1.0 - value))


class MGSCMAGFunctionalAblation(MGSCMAG):
    """Canonical P2 with one controlled functional ablation."""

    def __init__(self, cfg, data_info: dict[str, Any]):
        super().__init__(cfg, data_info)
        self.ablation = str(cfg.model.get("ablation", "full")).strip().lower()
        if self.ablation not in ABLATIONS:
            raise ValueError(
                f"unknown MGSC functional ablation {self.ablation!r}; "
                f"expected one of {ABLATIONS}"
            )
        if self.ablation in {"global_context_gate", "fixed_gate_09"}:
            initial = _logit(0.9)
            self.global_context_gate_logit_text = nn.Parameter(
                torch.full((self.max_order,), initial)
            )
            self.global_context_gate_logit_visual = nn.Parameter(
                torch.full((self.max_order,), initial)
            )

    def _relation_calibration(
        self, h_text: torch.Tensor, h_visual: torch.Tensor, edge_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        calibrated = super()._relation_calibration(h_text, h_visual, edge_index)
        if self.ablation != "uniform_relations":
            return calibrated
        # Physical support remains exactly edge_index.  Only the learned MRC
        # edge scores are removed before the usual normalized graph operator.
        uniform_text = torch.ones_like(calibrated["relation_weight_text"])
        uniform_visual = torch.ones_like(calibrated["relation_weight_visual"])
        return {
            **calibrated,
            "semantic_cosine_text": torch.zeros_like(
                calibrated["semantic_cosine_text"]
            ),
            "semantic_cosine_visual": torch.zeros_like(
                calibrated["semantic_cosine_visual"]
            ),
            "relation_weight_text": uniform_text,
            "relation_weight_visual": uniform_visual,
        }

    def _adaptive_multi_hop_states(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        modality: str,
        override_gates: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.ablation not in {"global_context_gate", "fixed_gate_09"}:
            return super()._adaptive_multi_hop_states(
                h0, edge_index, edge_weight, modality, override_gates
            )
        if override_gates is not None:
            raise ValueError("global gate ablations do not accept gate overrides")
        if modality == "text":
            logits = self.global_context_gate_logit_text
        elif modality == "visual":
            logits = self.global_context_gate_logit_visual
        else:
            raise ValueError(f"unknown modality {modality!r}")
        states = [h0]
        gates: list[torch.Tensor] = []
        current = h0
        for order in range(1, self.max_order + 1):
            propagated = self._propagate_once(current, edge_index, edge_weight)
            if self.ablation == "fixed_gate_09":
                gate_value = h0.new_tensor(0.9)
            else:
                gate_value = torch.sigmoid(logits[order - 1]).to(
                    device=h0.device, dtype=h0.dtype
                )
            gate = gate_value.expand(h0.size(0))
            current = (1.0 - gate.unsqueeze(-1)) * h0 + gate.unsqueeze(-1) * propagated
            states.append(current)
            gates.append(gate)
        return states, gates

    def _cross_order_interaction(
        self,
        states: list[torch.Tensor],
        modality: str,
        local_context: torch.Tensor,
        *,
        relation_intervention: str,
        interaction_intervention: str,
        capture_attention: bool,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        if self.ablation not in {
            "no_cross_order_interaction",
            "attribute_only",
        }:
            return super()._cross_order_interaction(
                states,
                modality,
                local_context,
                relation_intervention=relation_intervention,
                interaction_intervention=interaction_intervention,
                capture_attention=capture_attention,
            )
        count = len(states)
        attention = None
        if capture_attention:
            identity = torch.eye(
                count, dtype=states[0].dtype, device=states[0].device
            )
            attention = identity.unsqueeze(0).expand(states[0].size(0), -1, -1)
        # This makes tilde S_k = S_k.  For attribute_only, S_0 is exactly H0.
        return states, attention

    def _compose(
        self, states: list[torch.Tensor], eta: torch.Tensor
    ) -> torch.Tensor:
        if self.ablation == "terminal_state_only":
            return states[-1]
        if self.ablation == "uniform_integration":
            return torch.stack(states, dim=1).mean(dim=1)
        if self.ablation == "attribute_only":
            return states[0]
        return super()._compose(states, eta)

    @torch.no_grad()
    def analysis_multi_order(
        self, x: torch.Tensor, edge_index: torch.Tensor | None
    ) -> dict[str, Any]:
        result = super().analysis_multi_order(x, edge_index)
        result["ablation"] = self.ablation
        if self.ablation in {"global_context_gate", "fixed_gate_09"}:
            for modality in ("text", "visual"):
                gates = result[f"context_gate_{modality}"]
                bank = torch.stack(gates, dim=1)
                result[f"global_context_gate_{modality}"] = bank[0].detach().clone()
        return result


Model = MGSCMAGFunctionalAblation
