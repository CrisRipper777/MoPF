from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .mopf_iamoc import MoPFIAMOC


def _logit(probability: float) -> float:
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {value}")
    return math.log(value / (1.0 - value))


class MoPFIAMOCV2(MoPFIAMOC):
    """IAMOC-v2 relation-state-conditioned hop interaction.

    Stage I and Stage II remain inherited from MoPF. Relation moments are
    computed from their calibrated sparse edge weights and detached before
    they condition Stage-III hop attention.
    """

    def __init__(self, cfg, data_info: dict):
        # V2 deliberately removes relation conditioning from the inherited
        # output path. Relation state enters only through this class's
        # interaction hook.
        super().__init__(cfg, data_info)
        self.relation_state_mode = str(
            cfg.model.get("relation_state_mode", "none")
        ).strip().lower()
        if self.relation_state_mode not in {"none", "rank1", "profile"}:
            raise ValueError(
                "model.relation_state_mode must be none|rank1|profile, got "
                f"{self.relation_state_mode!r}"
            )
        if self.relation_conditioning != "none":
            raise ValueError(
                "IAMOC-v2 requires model.relation_conditioning='none'; "
                "relation state is injected only into hop-attention logits"
            )
        if self.relation_state_mode == "none" and self.ppc_weight != 0.0:
            raise ValueError("R0 requires model.ppc_weight=0")

        self.relation_descriptor_eps = float(
            cfg.model.get("relation_descriptor_eps", 1e-6)
        )
        if self.relation_descriptor_eps <= 0:
            raise ValueError("model.relation_descriptor_eps must be positive")
        self.relation_bias_init = float(cfg.model.get("relation_bias_init", 0.10))
        self.relation_init_seed = int(cfg.model.get("relation_init_seed", 20260921))
        self.relation_init_std = float(cfg.model.get("relation_init_std", 0.02))
        if not 0.0 < self.relation_bias_init < 1.0:
            raise ValueError("model.relation_bias_init must be in (0, 1)")
        if self.relation_init_std <= 0:
            raise ValueError("model.relation_init_std must be positive")

        self._relation_state_original: dict[str, torch.Tensor] = {}
        self._relation_state_used: dict[str, torch.Tensor] = {}
        self._relation_moments: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._relation_diagnostics: dict[str, dict[str, torch.Tensor]] = {}
        self._analysis_relation_bias_off = False
        self._analysis_relation_shuffle_seed: int | None = None

        if self.relation_state_mode == "rank1":
            for offset, modality in enumerate(("text", "visual")):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.relation_init_seed + offset)
                initial = torch.randn(
                    self.max_order + 1,
                    generator=generator,
                    dtype=torch.float32,
                ) * self.relation_init_std
                setattr(self, f"relation_beta_raw_{modality}", nn.Parameter(initial))
                setattr(
                    self,
                    f"theta_relation_scale_{modality}",
                    nn.Parameter(torch.tensor(_logit(self.relation_bias_init))),
                )
        elif self.relation_state_mode == "profile":
            # New relation modules must not shift the seeded initialization of
            # the unchanged NC classifier constructed immediately afterwards.
            with torch.random.fork_rng(devices=[]):
                for modality in ("text", "visual"):
                    setattr(
                        self,
                        f"relation_profile_{modality}",
                        nn.Sequential(
                            nn.Linear(2, 16),
                            nn.ReLU(),
                            nn.Linear(16, self.max_order + 1),
                        ),
                    )
                    setattr(
                        self,
                        f"theta_relation_profile_scale_{modality}",
                        nn.Parameter(torch.tensor(_logit(self.relation_bias_init))),
                    )

    @staticmethod
    def _relation_moments_for_nodes(
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute population E[w], std[w] on both physical edge endpoints."""
        device = edge_weight.device
        dtype = edge_weight.dtype
        endpoint = torch.cat((edge_index[0], edge_index[1]), dim=0)
        incident_weight = torch.cat((edge_weight, edge_weight), dim=0)
        count = torch.zeros(num_nodes, dtype=dtype, device=device)
        first = torch.zeros_like(count)
        second = torch.zeros_like(count)
        if endpoint.numel():
            count.index_add_(0, endpoint, torch.ones_like(incident_weight))
            first.index_add_(0, endpoint, incident_weight)
            second.index_add_(0, endpoint, incident_weight.square())
        safe_count = count.clamp_min(1.0)
        mean = first / safe_count
        variance = (second / safe_count - mean.square()).clamp_min(0.0)
        std = variance.sqrt()
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        std = torch.where(count > 0, std, torch.zeros_like(std))
        return mean, std

    def _relation_descriptors(
        self,
        edge_index: torch.Tensor,
        edges: dict[str, torch.Tensor],
        num_nodes: int,
    ) -> None:
        if self.relation_state_mode == "none":
            self._relation_state_original = {}
            self._relation_state_used = {}
            self._relation_moments = {}
            return

        original: dict[str, torch.Tensor] = {}
        moments: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.no_grad():
            for modality, key in (("text", "w_t"), ("visual", "w_v")):
                mu, sigma = self._relation_moments_for_nodes(
                    edge_index, edges[key].detach(), num_nodes
                )
                raw = torch.stack((mu, sigma), dim=-1)
                graph_mean = raw.mean(dim=0, keepdim=True)
                graph_std = raw.std(dim=0, unbiased=False, keepdim=True)
                standardized = ((raw - graph_mean) / (graph_std + self.relation_descriptor_eps)).detach()
                original[modality] = standardized
                moments[modality] = (mu.detach(), sigma.detach())

        used = dict(original)
        if self._analysis_relation_shuffle_seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(
                int(self._analysis_relation_shuffle_seed)
            )
            permutation = torch.randperm(num_nodes, generator=generator).to(edge_index.device)
            used = {
                modality: value.index_select(0, permutation)
                for modality, value in original.items()
            }
        self._relation_state_original = original
        self._relation_state_used = used
        self._relation_moments = moments

    def _transport_context(
        self,
        edge_index: torch.Tensor,
        edges: dict[str, torch.Tensor],
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        # Keep every inherited transport quantity intact for R0 and for the
        # Stage-I/II outputs. The new descriptor is an additional detached
        # input consumed only by _hop_interaction.
        transport = super()._transport_context(edge_index, edges, num_nodes)
        self._relation_descriptors(edge_index, edges, num_nodes)
        return transport

    def _relation_bias(self, modality: str, state: torch.Tensor) -> torch.Tensor:
        if self.relation_state_mode == "rank1":
            beta = getattr(self, f"relation_beta_raw_{modality}")
            beta = beta - beta.mean()
            beta = beta / beta.square().mean().sqrt().clamp_min(self.relation_descriptor_eps)
            scale = torch.sigmoid(getattr(self, f"theta_relation_scale_{modality}"))
            return scale * state[:, :1] * beta.unsqueeze(0)
        if self.relation_state_mode == "profile":
            raw = getattr(self, f"relation_profile_{modality}")(state)
            centered = raw - raw.mean(dim=-1, keepdim=True)
            normalized = centered / centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
                self.relation_descriptor_eps
            )
            scale = torch.sigmoid(
                getattr(self, f"theta_relation_profile_scale_{modality}")
            )
            return scale * normalized
        raise RuntimeError("relation bias requested for R0")

    def _hop_interaction(
        self,
        states: list[torch.Tensor],
        modality: str,
        relation_context: torch.Tensor,
        beta_order: torch.Tensor,
        *,
        capture_attention: bool,
        relation_bias_off: bool,
        relation_context_override: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        # The R0 branch executes the original implementation, preserving both
        # arithmetic and random-number consumption of IAMOC-v1 V2.
        if self.relation_state_mode == "none":
            return super()._hop_interaction(
                states,
                modality,
                relation_context,
                beta_order,
                capture_attention=capture_attention,
                relation_bias_off=relation_bias_off,
                relation_context_override=relation_context_override,
            )

        del relation_context, beta_order, relation_context_override
        state_bank = torch.stack(states, dim=1)
        if modality == "text":
            embedding = self.hop_order_embedding_text
            layers = self.hop_layers_text
            gate = torch.tanh(self.theta_hop_gate_text)
        else:
            embedding = self.hop_order_embedding_visual
            layers = self.hop_layers_visual
            gate = torch.tanh(self.theta_hop_gate_visual)

        descriptor = self._relation_state_used[modality].to(
            device=state_bank.device, dtype=state_bank.dtype
        ).detach()
        bias = self._relation_bias(modality, descriptor).to(dtype=state_bank.dtype)
        if self._analysis_relation_bias_off or relation_bias_off:
            bias_for_attention = None
        else:
            bias_for_attention = bias

        tokens = state_bank + embedding.unsqueeze(0) if self.hop_interaction_order_embedding else state_bank
        current = state_bank
        captured: list[torch.Tensor] = []
        last_content_rms = state_bank.new_zeros(())
        for layer_index, layer in enumerate(layers):
            normalized = layer.norm(tokens)
            query = layer.query(normalized)
            key = layer.key(normalized)
            value = layer.value(normalized)
            content_logits = torch.matmul(query, key.transpose(-1, -2)) * layer.scale
            logits = content_logits
            if bias_for_attention is not None:
                logits = logits + bias_for_attention.unsqueeze(1)
            attention = torch.softmax(logits, dim=-1)
            interaction = torch.matmul(layer.attention_dropout(attention), value)
            residual_base = state_bank if layer_index == 0 else current
            current = residual_base + gate * interaction
            if capture_attention:
                captured.append(attention)
            last_content_rms = content_logits.detach().float().square().mean().sqrt()
            tokens = current

        effective_bias = (
            torch.zeros_like(bias)
            if self._analysis_relation_bias_off or relation_bias_off
            else bias
        )
        self._relation_diagnostics[modality] = {
            "relation_state": descriptor.detach(),
            "relation_state_original": self._relation_state_original[modality].detach(),
            "relation_mu": self._relation_moments[modality][0].detach(),
            "relation_sigma": self._relation_moments[modality][1].detach(),
            "relation_bias": effective_bias.detach(),
            "relation_bias_rms": effective_bias.detach().float().square().mean().sqrt(),
            "content_logits_rms": last_content_rms,
            "bias_content_ratio": effective_bias.detach().float().square().mean().sqrt()
            / last_content_rms.clamp_min(1e-12),
            "per_order_bias_std": effective_bias.detach().float().std(dim=0, unbiased=False),
            "relation_scale": torch.sigmoid(
                getattr(
                    self,
                    f"theta_relation_scale_{modality}"
                    if self.relation_state_mode == "rank1"
                    else f"theta_relation_profile_scale_{modality}",
                )
            ).detach(),
        }
        last_attention = (
            captured[-1]
            if captured
            else state_bank.new_zeros(
                (state_bank.size(0), state_bank.size(1), state_bank.size(1))
            )
        )
        interacted = [current[:, order, :] for order in range(current.size(1))]
        return interacted, captured, last_attention

    def _encode_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        pdc_rho_text: torch.Tensor | None = None,
        pdc_rho_visual: torch.Tensor | None = None,
        metric_weights_text: torch.Tensor | None = None,
        metric_weights_visual: torch.Tensor | None = None,
        temperature_override: float | None = None,
        _capture_hop_attention: bool = False,
        _relation_bias_off: bool = False,
        _relation_context_override_text: torch.Tensor | None = None,
        _relation_context_override_visual: torch.Tensor | None = None,
        _capture_relation_diagnostics: bool = False,
    ) -> dict[str, Any]:
        components = super()._encode_components(
            x,
            edge_index,
            pdc_rho_text=pdc_rho_text,
            pdc_rho_visual=pdc_rho_visual,
            metric_weights_text=metric_weights_text,
            metric_weights_visual=metric_weights_visual,
            temperature_override=temperature_override,
            _capture_hop_attention=_capture_hop_attention,
            _relation_bias_off=_relation_bias_off,
            _relation_context_override_text=_relation_context_override_text,
            _relation_context_override_visual=_relation_context_override_visual,
        )
        if _capture_relation_diagnostics and self.relation_state_mode != "none":
            for modality in ("text", "visual"):
                for key, value in self._relation_diagnostics[modality].items():
                    components[f"{key}_{modality}"] = value
        return components

    @torch.no_grad()
    def analysis_hop_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        relation_intervention: str = "normal",
        permutation_seed: int = 0,
    ) -> dict[str, Any]:
        intervention = str(relation_intervention).strip().lower()
        if intervention not in {"normal", "off", "shuffle"}:
            raise ValueError("relation_intervention must be normal|off|shuffle")
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        was_training = self.training
        previous_off = self._analysis_relation_bias_off
        previous_shuffle = self._analysis_relation_shuffle_seed
        self.eval()
        self._analysis_relation_bias_off = intervention == "off"
        self._analysis_relation_shuffle_seed = (
            int(permutation_seed) if intervention == "shuffle" else None
        )
        try:
            components = self._encode_components(
                x,
                edge_index,
                _capture_hop_attention=True,
                _relation_bias_off=intervention == "off",
                _capture_relation_diagnostics=True,
            )
        finally:
            self._analysis_relation_bias_off = previous_off
            self._analysis_relation_shuffle_seed = previous_shuffle
            if was_training:
                self.train()

        return {
            **components,
            "relation_intervention": intervention,
            "hop_gate_text": torch.tanh(self.theta_hop_gate_text).detach().clone(),
            "hop_gate_visual": torch.tanh(self.theta_hop_gate_visual).detach().clone(),
        }


Model = MoPFIAMOCV2
