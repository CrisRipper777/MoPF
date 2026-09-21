from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mopf import MoPF


def _inverse_tanh(value: float) -> float:
    value = float(value)
    if not -1.0 < value < 1.0:
        raise ValueError(f"gate initialization must be between -1 and 1, got {value}")
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


class HopInteractionBlock(nn.Module):
    """Single-head, within-node attention over propagation-order tokens."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.scale = float(hidden_dim) ** -0.5

    def forward(
        self,
        tokens: torch.Tensor,
        residual_base: torch.Tensor,
        gate: torch.Tensor,
        relation_bias: torch.Tensor | None = None,
        *,
        capture_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.norm(tokens)
        query = self.query(normalized)
        key = self.key(normalized)
        value = self.value(normalized)
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        if relation_bias is not None:
            logits = logits + relation_bias.unsqueeze(1)
        attention = torch.softmax(logits, dim=-1)
        interaction = torch.matmul(self.attention_dropout(attention), value)
        output = residual_base + gate * interaction
        return output, attention if capture_attention else None


class MoPFIAMOC(MoPF):
    """Interaction-aware Stage-III preference estimation for formal MoPF."""

    def __init__(self, cfg, data_info: dict):
        if str(cfg.model.get("node_conditioner_mode", "absolute")).strip().lower() != "absolute":
            raise ValueError(
                "IAMOC v1 supports only model.node_conditioner_mode='absolute'"
            )
        super().__init__(cfg, data_info)

        self.use_hop_interaction = bool(cfg.model.get("use_hop_interaction", True))
        self.hop_interaction_layers = int(cfg.model.get("hop_interaction_layers", 1))
        if self.hop_interaction_layers < 1:
            raise ValueError("model.hop_interaction_layers must be >= 1")
        self.hop_interaction_heads = int(cfg.model.get("hop_interaction_heads", 1))
        if self.hop_interaction_heads != 1:
            raise ValueError("IAMOC v1 uses one attention head")
        self.hop_interaction_order_embedding = bool(
            cfg.model.get("hop_interaction_order_embedding", True)
        )
        self.hop_interaction_dropout = float(
            cfg.model.get("hop_interaction_dropout", 0.0)
        )
        self.hop_interaction_gate_init = float(
            cfg.model.get("hop_interaction_gate_init", 0.10)
        )
        self.relation_conditioning = str(
            cfg.model.get("relation_conditioning", "output")
        ).strip().lower()
        if self.relation_conditioning not in {"output", "none", "attention"}:
            raise ValueError(
                "model.relation_conditioning must be output|none|attention, got "
                f"{self.relation_conditioning!r}"
            )
        self.relation_bias_gate_init = float(
            cfg.model.get("relation_bias_gate_init", 0.10)
        )
        self.export_hop_attention = bool(cfg.model.get("export_hop_attention", False))

        order_count = self.max_order + 1
        self.hop_order_embedding_text = nn.Parameter(
            torch.zeros(order_count, self.hidden_dim)
        )
        self.hop_order_embedding_visual = nn.Parameter(
            torch.zeros(order_count, self.hidden_dim)
        )
        self.hop_layers_text = nn.ModuleList(
            HopInteractionBlock(self.hidden_dim, self.hop_interaction_dropout)
            for _ in range(self.hop_interaction_layers)
        )
        self.hop_layers_visual = nn.ModuleList(
            HopInteractionBlock(self.hidden_dim, self.hop_interaction_dropout)
            for _ in range(self.hop_interaction_layers)
        )
        self.theta_hop_gate_text = nn.Parameter(
            torch.tensor(_inverse_tanh(self.hop_interaction_gate_init))
        )
        self.theta_hop_gate_visual = nn.Parameter(
            torch.tensor(_inverse_tanh(self.hop_interaction_gate_init))
        )
        self.theta_relation_bias_text = nn.Parameter(
            torch.tensor(_inverse_tanh(self.relation_bias_gate_init))
        )
        self.theta_relation_bias_visual = nn.Parameter(
            torch.tensor(_inverse_tanh(self.relation_bias_gate_init))
        )

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
        state_bank = torch.stack(states, dim=1)
        if modality == "text":
            embedding = self.hop_order_embedding_text
            layers = self.hop_layers_text
            gate = torch.tanh(self.theta_hop_gate_text)
            relation_gate = torch.tanh(self.theta_relation_bias_text)
        else:
            embedding = self.hop_order_embedding_visual
            layers = self.hop_layers_visual
            gate = torch.tanh(self.theta_hop_gate_visual)
            relation_gate = torch.tanh(self.theta_relation_bias_visual)

        effective_context = (
            relation_context
            if relation_context_override is None
            else relation_context_override
        )
        relation_bias = None
        if (
            self.use_hop_interaction
            and self.relation_conditioning == "attention"
            and not relation_bias_off
        ):
            relation_bias = (
                relation_gate
                * effective_context.detach().to(dtype=state_bank.dtype).unsqueeze(-1)
                * beta_order.to(dtype=state_bank.dtype).unsqueeze(0)
            )

        if not self.use_hop_interaction:
            return states, [], state_bank.new_zeros((state_bank.size(0), state_bank.size(1), state_bank.size(1)))

        tokens = state_bank + embedding.unsqueeze(0) if self.hop_interaction_order_embedding else state_bank
        current = state_bank
        captured: list[torch.Tensor] = []
        for layer_index, layer in enumerate(layers):
            residual_base = state_bank if layer_index == 0 else current
            current, attention = layer(
                tokens,
                residual_base,
                gate,
                relation_bias,
                capture_attention=capture_attention,
            )
            if attention is not None:
                captured.append(attention)
            tokens = current
        interacted = [current[:, order, :] for order in range(current.size(1))]
        last_attention = (
            captured[-1]
            if captured
            else state_bank.new_zeros(
                (state_bank.size(0), state_bank.size(1), state_bank.size(1))
            )
        )
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
    ) -> dict[str, Any]:
        x_text, x_visual = self._split_features(x)
        h_text = self.text_proj(x_text)
        h_visual = self.visual_proj(x_visual)

        edges = self._semantic_edge_weights(
            h_text,
            h_visual,
            edge_index,
            metric_weights_text=metric_weights_text,
            metric_weights_visual=metric_weights_visual,
            temperature_override=temperature_override,
        )
        norm_t_index, norm_t_weight = self._normalized_operator(
            edge_index, edges["w_t"], int(x.size(0)), h_text.dtype
        )
        norm_v_index, norm_v_weight = self._normalized_operator(
            edge_index, edges["w_v"], int(x.size(0)), h_visual.dtype
        )
        banks_text = self._build_multihop_banks(h_text, norm_t_index, norm_t_weight)
        banks_visual = self._build_multihop_banks(h_visual, norm_v_index, norm_v_weight)
        states_text = banks_text["states"]
        states_visual = banks_visual["states"]
        responses_text = banks_text["responses"]
        responses_visual = banks_visual["responses"]

        transport = self._transport_context(edge_index, edges, int(x.size(0)))
        beta_text = self.theta_transport_text - self.theta_transport_text.mean()
        beta_visual = self.theta_transport_visual - self.theta_transport_visual.mean()
        capture_attention = _capture_hop_attention or self.export_hop_attention
        interacted_text, attention_layers_text, attention_text = self._hop_interaction(
            states_text,
            "text",
            transport["transport_context_text"],
            beta_text,
            capture_attention=capture_attention,
            relation_bias_off=_relation_bias_off,
            relation_context_override=_relation_context_override_text,
        )
        interacted_visual, attention_layers_visual, attention_visual = self._hop_interaction(
            states_visual,
            "visual",
            transport["transport_context_visual"],
            beta_visual,
            capture_attention=capture_attention,
            relation_bias_off=_relation_bias_off,
            relation_context_override=_relation_context_override_visual,
        )
        delta_node_text, conditioned_bases_text, pdc_aux_text = self._node_residuals(
            interacted_text,
            self.node_proj_text,
            self.node_vector_text,
            "text",
            pdc_rho_text,
        )
        delta_node_visual, conditioned_bases_visual, pdc_aux_visual = self._node_residuals(
            interacted_visual,
            self.node_proj_visual,
            self.node_vector_visual,
            "visual",
            pdc_rho_visual,
        )

        # Only output conditioning retains the formal TCPR term. Relation bias
        # has already influenced interaction for the attention variant.
        if self.relation_conditioning == "output":
            eta_text = self._effective_coefficients(
                "text", delta_node_text, transport["transport_context_text"]
            )
            eta_visual = self._effective_coefficients(
                "visual", delta_node_visual, transport["transport_context_visual"]
            )
        else:
            modality_text = self.delta_gamma_text if self.use_modality_residual else torch.zeros_like(self.delta_gamma_text)
            modality_visual = self.delta_gamma_visual if self.use_modality_residual else torch.zeros_like(self.delta_gamma_visual)
            eta_text = self.gamma_global.unsqueeze(0) + modality_text.unsqueeze(0) + delta_node_text
            eta_visual = self.gamma_global.unsqueeze(0) + modality_visual.unsqueeze(0) + delta_node_visual

        if self.relation_conditioning == "output" and self.use_transport_residual:
            tau_text = transport["transport_context_text"].unsqueeze(-1) * beta_text.unsqueeze(0)
            tau_visual = transport["transport_context_visual"].unsqueeze(-1) * beta_visual.unsqueeze(0)
        else:
            tau_text = delta_node_text.new_zeros(delta_node_text.shape)
            tau_visual = delta_node_visual.new_zeros(delta_node_visual.shape)

        z_text = self._compose_responses(responses_text, eta_text)
        z_visual = self._compose_responses(responses_visual, eta_visual)
        z_text_refined = self.text_refine_norm(z_text + self.text_refine_mlp(z_text))
        z_visual_refined = self.visual_refine_norm(z_visual + self.visual_refine_mlp(z_visual))
        fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
        z = self.output_norm(self.fusion_skip(fused_input) + self.fusion_mlp(fused_input))

        components: dict[str, Any] = {
            "h_text": h_text,
            "h_visual": h_visual,
            "edges": edges,
            "norm_t_index": norm_t_index,
            "norm_t_weight": norm_t_weight,
            "norm_v_index": norm_v_index,
            "norm_v_weight": norm_v_weight,
            "bases_text": responses_text,
            "bases_visual": responses_visual,
            "states_text": states_text,
            "states_visual": states_visual,
            "responses_text": responses_text,
            "responses_visual": responses_visual,
            "conditioned_states_text": conditioned_bases_text,
            "conditioned_states_visual": conditioned_bases_visual,
            "conditioned_bases_text": conditioned_bases_text,
            "conditioned_bases_visual": conditioned_bases_visual,
            "pdc_aux_text": pdc_aux_text,
            "pdc_aux_visual": pdc_aux_visual,
            "delta_node_text": delta_node_text,
            "delta_node_visual": delta_node_visual,
            "mean_conductance_text": transport["mean_conductance_text"],
            "mean_conductance_visual": transport["mean_conductance_visual"],
            "transport_context_text": transport["transport_context_text"],
            "transport_context_visual": transport["transport_context_visual"],
            "transport_degree_text": transport["transport_degree_text"],
            "transport_degree_visual": transport["transport_degree_visual"],
            "beta_transport_text": beta_text,
            "beta_transport_visual": beta_visual,
            "tau_transport_text": tau_text,
            "tau_transport_visual": tau_visual,
            "eta_text": eta_text,
            "eta_visual": eta_visual,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": z,
        }
        if capture_attention:
            components.update(
                {
                    "hop_attention_text": attention_text,
                    "hop_attention_visual": attention_visual,
                    "hop_attention_layers_text": attention_layers_text,
                    "hop_attention_layers_visual": attention_layers_visual,
                }
            )
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
        """Analysis-only attention capture and inference-time relation intervention."""
        intervention = str(relation_intervention).strip().lower()
        if intervention not in {"normal", "off", "shuffle"}:
            raise ValueError("relation_intervention must be normal|off|shuffle")
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        was_training = self.training
        self.eval()

        context_text = context_visual = None
        if intervention == "shuffle":
            with torch.no_grad():
                x_text, x_visual = self._split_features(x)
                h_text, h_visual = self.text_proj(x_text), self.visual_proj(x_visual)
                edges = self._semantic_edge_weights(h_text, h_visual, edge_index)
                contexts = self._transport_context(edge_index, edges, int(x.size(0)))
                generator = torch.Generator(device="cpu").manual_seed(int(permutation_seed))
                permutation = torch.randperm(x.size(0), generator=generator).to(x.device)
                context_text = contexts["transport_context_text"].index_select(0, permutation)
                context_visual = contexts["transport_context_visual"].index_select(0, permutation)

        components = self._encode_components(
            x,
            edge_index,
            _capture_hop_attention=True,
            _relation_bias_off=intervention == "off",
            _relation_context_override_text=context_text,
            _relation_context_override_visual=context_visual,
        )
        if was_training:
            self.train()
        return {
            **components,
            "relation_intervention": intervention,
            "hop_gate_text": torch.tanh(self.theta_hop_gate_text).detach().clone(),
            "hop_gate_visual": torch.tanh(self.theta_hop_gate_visual).detach().clone(),
            "relation_bias_gate_text": torch.tanh(self.theta_relation_bias_text).detach().clone(),
            "relation_bias_gate_visual": torch.tanh(self.theta_relation_bias_visual).detach().clone(),
        }

    @torch.no_grad()
    def analysis_stats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        result = self.analysis_hop_attention(x, edge_index)
        output = {
            "gamma_global": self.gamma_global.detach().clone(),
            "delta_gamma_text": self.delta_gamma_text.detach().clone(),
            "delta_gamma_visual": self.delta_gamma_visual.detach().clone(),
        }
        output.update(
            {
                key: result[key].detach().clone()
                for key in (
                "delta_node_text",
                "delta_node_visual",
                "eta_text",
                "eta_visual",
                "hop_attention_text",
                "hop_attention_visual",
                "hop_gate_text",
                "hop_gate_visual",
                "relation_bias_gate_text",
                "relation_bias_gate_visual",
                )
                if key in result
            }
        )
        output["effective_radius_text"] = self._effective_radius(result["eta_text"]).detach().clone()
        output["effective_radius_visual"] = self._effective_radius(result["eta_visual"]).detach().clone()
        return output


Model = MoPFIAMOC
