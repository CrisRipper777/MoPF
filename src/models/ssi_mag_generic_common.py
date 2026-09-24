from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from .ssi_mag_v31 import ProjectionMLP, _make_mlp


class SSIMAGGenericBase(nn.Module):
    """Shared clean encoder for conceptual PPR/GPR-style controls.

    Only the modality projections, multi-scale diffusion states, modality
    refinement, and late fusion are shared with S. No SCD relation, semantic
    prior, adaptive restart, response residual, or attention parameters are
    constructed here.
    """

    requires_full_graph_training = True
    requires_full_lp_sampler_depth = True

    def __init__(self, cfg, data_info: dict, *, control_mode: str):
        super().__init__()
        if str(cfg.get("ablation", "full")).strip().lower() != "full":
            raise ValueError("generic diffusion controls require top-level ablation=full")
        if control_mode not in {"ppr_style", "gpr_style"}:
            raise ValueError(f"unknown generic control mode: {control_mode}")
        input_dim = int(data_info["input_dim"])
        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        self.hidden_dim = int(cfg.model.get("hidden_dim", 256))
        self.out_dim = self.hidden_dim
        self.max_order = int(cfg.model.get("max_order", 3))
        self.num_layers = int(cfg.model.get("num_layers", self.max_order))
        if self.max_order != 3 or self.num_layers != self.max_order:
            raise ValueError("P3 generic controls require K=num_layers=max_order=3")
        self.diffusion_add_self_loops = bool(cfg.model.get("diffusion_add_self_loops", True))
        self.eps = float(cfg.model.get("eps", 1.0e-8))
        dropout = float(cfg.model.get("dropout", 0.2))
        norm = cfg.model.get("norm", "layernorm")
        self.control_mode = control_mode
        self.text_proj = ProjectionMLP(self.text_dim, self.hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, self.hidden_dim, dropout, norm)
        self.text_refine_mlp = _make_mlp(self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout)
        self.visual_refine_mlp = _make_mlp(self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout)
        self.text_refine_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_refine_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_skip = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.fusion_mlp = _make_mlp(2 * self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        if control_mode == "gpr_style":
            prior = self._make_global_prior(cfg)
            self.gamma_text = nn.Parameter(prior.clone())
            self.gamma_visual = nn.Parameter(prior.clone())

    @staticmethod
    def _make_global_prior(cfg) -> torch.Tensor:
        restart = float(cfg.model.get("global_prior_restart", 0.15))
        prior_order = int(cfg.model.get("global_prior_order", 2))
        prior = torch.zeros(4, dtype=torch.float32)
        if prior_order < 4:
            for order in range(prior_order):
                prior[order] = restart * (1.0 - restart) ** order
            prior[prior_order] = (1.0 - restart) ** prior_order
        return prior

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype == torch.long:
            x = x.float()
        return x[:, : self.text_dim], x[:, self.text_dim : self.text_dim + self.visual_dim]

    @staticmethod
    def _edge_index_or_empty(edge_index: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if edge_index is None:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index.to(device=device, dtype=torch.long)

    def _raw_operator(self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        unit = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)
        normalized_index, normalized_weight = gcn_norm(
            edge_index,
            edge_weight=unit,
            num_nodes=num_nodes,
            improved=False,
            add_self_loops=self.diffusion_add_self_loops,
            flow="source_to_target",
            dtype=dtype,
        )
        if normalized_weight is None:
            normalized_weight = torch.ones(normalized_index.size(1), dtype=dtype, device=normalized_index.device)
        return normalized_index, torch.nan_to_num(normalized_weight, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _propagate_once(h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index
        return scatter(h[src] * edge_weight.unsqueeze(-1), dst, dim=0, dim_size=h.size(0), reduce="sum")

    @staticmethod
    def _compose(states: list[torch.Tensor], gamma: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(states[0])
        for order, state in enumerate(states):
            result = result + gamma[order] * state
        return result

    def _encode_modality(self, h0: torch.Tensor, edge_index: torch.Tensor, modality: str, **_: Any) -> dict[str, Any]:
        normalized_index, normalized_weight = self._raw_operator(edge_index, h0.size(0), h0.dtype)
        states = [h0]
        current = h0
        alpha = 0.10
        for _order in range(1, self.max_order + 1):
            propagated = self._propagate_once(current, normalized_index, normalized_weight)
            if self.control_mode == "ppr_style":
                current = (1.0 - alpha) * propagated + alpha * h0
            else:
                current = propagated
            states.append(current)
        if self.control_mode == "ppr_style":
            z = states[-1]
            gamma = None
        else:
            gamma = getattr(self, f"gamma_{modality}")
            z = self._compose(states, gamma.to(dtype=h0.dtype))
        return {
            "states": states,
            "normalized_edge_index": normalized_index,
            "normalized_edge_weight": normalized_weight,
            "alpha": alpha,
            "gamma": gamma,
            "z": z,
        }

    def _encode_components(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        x_text, x_visual = self._split_features(x)
        h0_text = self.text_proj(x_text)
        h0_visual = self.visual_proj(x_visual)
        text = self._encode_modality(h0_text, edge_index, "text")
        visual = self._encode_modality(h0_visual, edge_index, "visual")
        z_text, z_visual = text["z"], visual["z"]
        z_text_refined = self.text_refine_norm(z_text + self.text_refine_mlp(z_text))
        z_visual_refined = self.visual_refine_norm(z_visual + self.visual_refine_mlp(z_visual))
        fused_input = torch.cat((z_text_refined, z_visual_refined), dim=-1)
        fused = self.output_norm(self.fusion_skip(fused_input) + self.fusion_mlp(fused_input))
        return {
            "h0_text": h0_text,
            "h0_visual": h0_visual,
            "text": text,
            "visual": visual,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": fused,
            "physical_edge_index": edge_index,
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)
        z = torch.nan_to_num(components["z"], nan=0.0, posinf=1e4, neginf=-1e4)
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(self, x: torch.Tensor, edge_index: torch.Tensor | None, device: torch.device | None = None, batch_size: int = 65536) -> torch.Tensor:
        del batch_size
        if device is None:
            device = next(self.parameters()).device
        was_training = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            z, _, _, _, _ = self(x.to(device), edge_index.to(device) if edge_index is not None else None)
            return z.detach().cpu()
        finally:
            for module, training in was_training:
                module.training = training

    @torch.no_grad()
    def analysis(self, x: torch.Tensor, edge_index: torch.Tensor | None) -> dict[str, Any]:
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)
        result: dict[str, Any] = {
            "control_mode": self.control_mode,
            "physical_edge_index": edge_index,
            "h0_text": components["h0_text"],
            "h0_visual": components["h0_visual"],
            "z_text": components["z_text"],
            "z_visual": components["z_visual"],
            "z_text_refined": components["z_text_refined"],
            "z_visual_refined": components["z_visual_refined"],
            "z": components["z"],
        }
        for modality in ("text", "visual"):
            branch = components[modality]
            result[f"S_{modality}"] = branch["states"]
            result[f"normalized_edge_index_{modality}"] = branch["normalized_edge_index"]
            result[f"normalized_edge_weight_{modality}"] = branch["normalized_edge_weight"]
            result[f"alpha_{modality}"] = branch["alpha"]
            if branch["gamma"] is not None:
                result[f"gamma_{modality}"] = branch["gamma"]
        return {key: self._clone_value(value) for key, value in result.items()}

    @staticmethod
    def _clone_value(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, list):
            return [SSIMAGGenericBase._clone_value(item) for item in value]
        if isinstance(value, dict):
            return {key: SSIMAGGenericBase._clone_value(item) for key, item in value.items()}
        return value
