from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .cssi_v0 import CSSIV0


RESPONSE_COEFFICIENTS = (0.75, 0.50, 0.25)


def _logit(probability: float) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}")
    return math.log(probability / (1.0 - probability))


class CSSIV1(CSSIV0):
    """CSSI-v1 with bounded response subspace modulation (BRSM).

    The inherited P3 encoder supplies the unchanged two-modality projectors,
    unit/MRC relation construction, three propagation steps, response tokens,
    and late refinement/fusion.  Only the response transformation is replaced:
    a row-orthonormal rank-r basis and a bounded, response-proportional
    modulation are used.  The inherited P3 adapter modules remain in the
    state dictionary for checkpoint compatibility but are not referenced by
    this class's forward path.
    """

    def __init__(self, cfg, data_info: dict):
        model_cfg = cfg.model
        condition_mode = str(model_cfg.get("condition_mode", "same")).lower()
        adapter_type = str(model_cfg.get("adapter_type", "brsm")).lower()
        if condition_mode not in {"self", "same"}:
            raise ValueError("CSSIV1 supports condition_mode=self|same only")
        if adapter_type not in {"brsm", "scalar", "none"}:
            raise ValueError("CSSIV1 adapter_type must be brsm|scalar|none")
        self.condition_mode = condition_mode
        self.adapter_type = adapter_type
        self.adapter_enabled = bool(model_cfg.get("adapter_enabled", True))
        self.lambda_max = float(model_cfg.get("lambda_max", 0.2))
        self.lambda_init = float(model_cfg.get("lambda_init", 0.05))
        if not 0.0 < self.lambda_init < self.lambda_max:
            raise ValueError("lambda_init must lie strictly between 0 and lambda_max")
        if self.lambda_max <= 0.0:
            raise ValueError("lambda_max must be positive")
        if adapter_type == "none" or not self.adapter_enabled:
            self.adapter_type = "none"
            self.adapter_enabled = False

        # CSSIV0 owns the unchanged encoder/controller construction.  Use a
        # private config copy only to satisfy its legacy adapter validation;
        # its response-up path is never called by CSSIV1.
        base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        base_cfg.model.adapter_type = "lowrank" if self.adapter_type == "brsm" else self.adapter_type
        base_cfg.model.adapter_enabled = self.adapter_enabled
        base_cfg.model.response_correction_scale = 1.0
        super().__init__(base_cfg, data_info)
        # CSSIV0 normalizes the legacy adapter label during construction;
        # restore the v1 dispatch label after its modules have been built.
        self.adapter_type = adapter_type if self.adapter_enabled else "none"

        # The default Linear initialization makes tanh(f(C)) so small that a
        # rank-8 projection in d=256 produces a nearly universal ~1e-3
        # first-epoch correction ratio.  This is an initialization gain for
        # the controller only, not an output scale: BRSM remains exactly
        # lambda * Bbar^T(a*Bbar R), with |a|<=1 and lambda<=lambda_max.
        if self.adapter_type == "brsm":
            with torch.no_grad():
                self.response_condition_text.weight.mul_(4.0)
                self.response_condition_text.bias.mul_(4.0)
                self.response_condition_visual.weight.mul_(4.0)
                self.response_condition_visual.bias.mul_(4.0)

        self.response_basis_text = nn.Parameter(
            torch.empty(self.response_rank, self.hidden_dim)
        )
        self.response_basis_visual = nn.Parameter(
            torch.empty(self.response_rank, self.hidden_dim)
        )
        nn.init.orthogonal_(self.response_basis_text)
        nn.init.orthogonal_(self.response_basis_visual)
        lambda_theta = _logit(self.lambda_init / self.lambda_max)
        self.lambda_theta_text = nn.Parameter(
            torch.full((self.max_order,), lambda_theta)
        )
        self.lambda_theta_visual = nn.Parameter(
            torch.full((self.max_order,), lambda_theta)
        )

    def _orthonormal_basis(self, modality: str) -> torch.Tensor:
        raw = (
            self.response_basis_text
            if modality == "text"
            else self.response_basis_visual
        )
        # QR on B.T gives Q in R^{d x r}; transpose to obtain row-orthonormal
        # Bbar in R^{r x d}.  This construction is differentiable for the
        # full-row-rank parameter values used here.
        q, _ = torch.linalg.qr(raw.transpose(0, 1), mode="reduced")
        return q.transpose(0, 1)

    def modulation_strength(self, modality: str) -> torch.Tensor:
        theta = (
            self.lambda_theta_text
            if modality == "text"
            else self.lambda_theta_visual
        )
        return self.lambda_max * torch.sigmoid(theta)

    def _condition(
        self,
        target_tokens: torch.Tensor,
        other_tokens: torch.Tensor,
        modality: str,
        *,
        cross_intervention: str,
        cross_permutation: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Build self/same controls with an exact cross-off intervention."""
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
            # Zero after the projection: Linear bias cannot reintroduce a
            # paired contribution.  The target path remains intact.
            other_q = torch.zeros_like(target_q)
        else:
            if cross_intervention == "shuffle":
                if cross_permutation is None:
                    raise ValueError("cross shuffle requires a node permutation")
                other_tokens = other_tokens.index_select(0, cross_permutation)
            other_q = other_projection(other_tokens)

        if self.condition_mode == "self":
            condition = self_controller(target_tokens)
        else:
            product = target_q * other_q
            difference = (target_q - other_q).abs()
            condition = same_controller(
                torch.cat((target_q, other_q, product, difference), dim=-1)
            )
        return condition, None, target_q, other_q

    def _adapt_responses(
        self,
        responses: list[torch.Tensor],
        condition: torch.Tensor,
        modality: str,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        corrections = []
        amplitudes = []
        basis = self._orthonormal_basis(modality)
        strengths = self.modulation_strength(modality)
        for order, response in enumerate(responses):
            c = condition[:, order, :]
            if not self.adapter_enabled or self.adapter_type == "none":
                amplitude = torch.zeros(
                    response.size(0), self.response_rank,
                    dtype=response.dtype, device=response.device,
                )
                correction = torch.zeros_like(response)
            elif self.adapter_type == "brsm":
                condition_layer = (
                    self.response_condition_text
                    if modality == "text"
                    else self.response_condition_visual
                )
                amplitude = torch.tanh(condition_layer(c))
                latent = response @ basis.transpose(0, 1)
                correction = (amplitude * latent) @ basis
                correction = strengths[order] * correction
            elif self.adapter_type == "scalar":
                controller = (
                    self.scalar_controller_text
                    if modality == "text"
                    else self.scalar_controller_visual
                )
                amplitude = torch.tanh(controller(c))
                correction = strengths[order] * amplitude * response
            else:  # pragma: no cover - constructor validation makes this unreachable
                raise RuntimeError(f"unsupported adapter_type={self.adapter_type}")
            corrections.append(correction)
            amplitudes.append(amplitude)
        return corrections, {
            "modulation_amplitude": torch.stack(amplitudes, dim=1),
            "controller_gate": torch.stack(amplitudes, dim=1),
            "lambda": strengths,
            "basis": basis,
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
        # The parent has already run its adapter. Recompute only the response
        # adaptation and downstream refinement using the v1 BRSM path.
        corrections_text, info_text = self._adapt_responses(
            components["responses_text"], components["control_text"], "text"
        )
        corrections_visual, info_visual = self._adapt_responses(
            components["responses_visual"], components["control_visual"], "visual"
        )
        z_base_text = components["z_base_text"]
        z_base_visual = components["z_base_visual"]
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
        components.update(
            {
                "corrections_text": corrections_text,
                "corrections_visual": corrections_visual,
                "modulation_amplitude_text": info_text["modulation_amplitude"],
                "modulation_amplitude_visual": info_visual["modulation_amplitude"],
                "lambda_text": info_text["lambda"],
                "lambda_visual": info_visual["lambda"],
                "response_basis_text": info_text["basis"],
                "response_basis_visual": info_visual["basis"],
                "controller_gate_text": info_text["modulation_amplitude"],
                "controller_gate_visual": info_visual["modulation_amplitude"],
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
        )
        return components

    def _training_aux_info(self, components: dict[str, Any]) -> dict[str, torch.Tensor]:
        info = super()._training_aux_info(components)
        info["mean_lambda_text"] = components["lambda_text"].mean()
        info["mean_lambda_visual"] = components["lambda_visual"].mean()
        return info

    def gradient_diagnostics(self) -> dict[str, torch.Tensor]:
        info = super().gradient_diagnostics()
        controller_grads = []
        for name, parameter in self.named_parameters():
            if parameter.grad is not None and (
                "self_controller" in name
                or "same_controller" in name
                or "response_condition" in name
                or "scalar_controller" in name
            ):
                controller_grads.append(parameter.grad.detach().float().square().sum())
        basis_grads = [
            parameter.grad.detach().float().square().sum()
            for parameter in (self.response_basis_text, self.response_basis_visual)
            if parameter.grad is not None
        ]
        lambda_grads = [
            parameter.grad.detach().float().square().sum()
            for parameter in (self.lambda_theta_text, self.lambda_theta_visual)
            if parameter.grad is not None
        ]
        device = next(self.parameters()).device
        info["controller_gradient_norm"] = (
            torch.stack(controller_grads).sum().sqrt()
            if controller_grads else torch.zeros((), device=device)
        )
        info["basis_gradient_norm"] = (
            torch.stack(basis_grads).sum().sqrt()
            if basis_grads else torch.zeros((), device=device)
        )
        info["lambda_gradient_norm"] = (
            torch.stack(lambda_grads).sum().sqrt()
            if lambda_grads else torch.zeros((), device=device)
        )
        return info


Model = CSSIV1
