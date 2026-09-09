from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import scatter

from .common import make_norm


class ProjectionMLP(nn.Module):
    """Project one frozen modality without mixing it with the other modality."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, norm: str | None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            make_norm(norm, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """The lightweight two-linear-layer MLP used for refinement and fusion."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class MoPF(nn.Module):
    """Multi-order Personalized Filtering for multimodal attributed graphs.

    MoPF keeps text and visual representations separate through projection,
    semantic graph construction, and polynomial propagation.  It learns a
    hierarchy of unbounded polynomial coefficients: a global prior, a
    modality residual, and a low-rank node-conditioned residual.
    """

    EDGE_WEIGHT_MODES = {"separate_cos", "shared_avg_cos", "raw_uniform"}

    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        hidden_dim = int(cfg.model.get("hidden_dim", 256))
        dropout = float(cfg.model.get("dropout", 0.2))
        norm = cfg.model.get("norm", "layernorm")

        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        if self.text_dim + self.visual_dim > input_dim:
            raise ValueError(
                "text_dim+visual_dim="
                f"{self.text_dim + self.visual_dim} exceeds input_dim={input_dim}"
            )

        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.max_order = int(cfg.model.get("max_order", 3))
        if self.max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {self.max_order}")
        self.num_layers = int(cfg.model.get("num_layers", self.max_order))
        if self.num_layers != self.max_order:
            raise ValueError(
                "MoPF requires model.num_layers == model.max_order so sampled "
                f"LP message passing has the same depth; got {self.num_layers} "
                f"and {self.max_order}"
            )

        self.map_prior_restart = float(cfg.model.get("map_prior_restart", 0.15))
        if not 0.0 <= self.map_prior_restart <= 1.0:
            raise ValueError(
                "map_prior_restart must be in [0, 1], got "
                f"{self.map_prior_restart}"
            )
        self.map_prior_order = int(cfg.model.get("map_prior_order", 2))
        if self.map_prior_order < 0:
            raise ValueError(
                f"map_prior_order must be >= 0, got {self.map_prior_order}"
            )
        self.diffusion_add_self_loops = bool(
            cfg.model.get("diffusion_add_self_loops", True)
        )

        self.edge_weight_mode = str(
            cfg.model.get("edge_weight_mode", "separate_cos")
        ).strip().lower()
        if self.edge_weight_mode not in self.EDGE_WEIGHT_MODES:
            valid = ", ".join(sorted(self.EDGE_WEIGHT_MODES))
            raise ValueError(
                f"model.edge_weight_mode must be one of [{valid}], "
                f"got {self.edge_weight_mode!r}"
            )
        self.edge_weight_min = float(cfg.model.get("edge_weight_min", 0.1))
        if not 0.0 <= self.edge_weight_min <= 1.0:
            raise ValueError(
                "edge_weight_min must be in [0, 1], got "
                f"{self.edge_weight_min}"
            )
        self.edge_weight_temperature = float(
            cfg.model.get("edge_weight_temperature", 2.0)
        )
        if self.edge_weight_temperature <= 0.0:
            raise ValueError(
                "edge_weight_temperature must be positive, got "
                f"{self.edge_weight_temperature}"
            )
        self.eps = float(cfg.model.get("eps", 1e-8))
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}")

        self.filter_rank = int(cfg.model.get("filter_rank", 4))
        if self.filter_rank < 1:
            raise ValueError(f"filter_rank must be >= 1, got {self.filter_rank}")
        self.global_filter_trainable = bool(
            cfg.model.get("global_filter_trainable", True)
        )
        self.use_modality_residual = bool(
            cfg.model.get("use_modality_residual", True)
        )
        self.use_node_residual = bool(cfg.model.get("use_node_residual", True))
        self.hrc_weight = float(cfg.model.get("hrc_weight", 0.0))
        if self.hrc_weight < 0.0:
            raise ValueError(f"hrc_weight must be >= 0, got {self.hrc_weight}")

        fusion_mode = str(
            cfg.model.get("fusion_mode", "concat_residual_mlp")
        ).strip().lower()
        if fusion_mode != "concat_residual_mlp":
            raise ValueError(
                "MoPF only supports fusion_mode='concat_residual_mlp', got "
                f"{fusion_mode!r}"
            )
        self.fusion_mode = fusion_mode

        self.text_proj = ProjectionMLP(self.text_dim, hidden_dim, dropout, norm)
        self.visual_proj = ProjectionMLP(self.visual_dim, hidden_dim, dropout, norm)

        gamma_init = self._make_global_prior()
        self.gamma_global = nn.Parameter(
            gamma_init,
            requires_grad=self.global_filter_trainable,
        )
        self.delta_gamma_text = nn.Parameter(torch.zeros(self.max_order + 1))
        self.delta_gamma_visual = nn.Parameter(torch.zeros(self.max_order + 1))

        self.node_proj_text = nn.ModuleList(
            nn.Linear(hidden_dim, self.filter_rank) for _ in range(self.max_order + 1)
        )
        self.node_proj_visual = nn.ModuleList(
            nn.Linear(hidden_dim, self.filter_rank) for _ in range(self.max_order + 1)
        )
        self.node_vector_text = nn.Parameter(
            torch.zeros(self.max_order + 1, self.filter_rank)
        )
        self.node_vector_visual = nn.Parameter(
            torch.zeros(self.max_order + 1, self.filter_rank)
        )

        self.text_refine_mlp = _make_mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.visual_refine_mlp = _make_mlp(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.text_refine_norm = nn.LayerNorm(hidden_dim)
        self.visual_refine_norm = nn.LayerNorm(hidden_dim)

        self.fusion_skip = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fusion_mlp = _make_mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # MoPF is compatible with full-graph NC and sampled LP.  NC's task
        # configuration remains the authority for choosing full-graph mode.
        self.requires_full_graph_training = False
        # NC sets this to the local training-node indices before a training
        # forward.  Keeping it out of the state dict makes it runtime context,
        # not a learned model component.
        self._hrc_training_idx: torch.Tensor | None = None

    def set_hrc_training_nodes(self, node_idx: torch.Tensor | None) -> None:
        """Set the nodes used by HRC's training-time population mean."""
        self._hrc_training_idx = None if node_idx is None else node_idx.detach()

    def _make_global_prior(self) -> torch.Tensor:
        """Initialize the MAP two-step restart polynomial in a general form."""
        alpha = self.map_prior_restart
        gamma = torch.zeros(self.max_order + 1, dtype=torch.float32)
        if self.max_order >= self.map_prior_order:
            for order in range(self.map_prior_order):
                gamma[order] = alpha * (1.0 - alpha) ** order
            gamma[self.map_prior_order] = (1.0 - alpha) ** self.map_prior_order
        else:
            # Preserve the truncated restart expansion's final residual mass
            # when an ablation chooses fewer orders than the MAP prior.
            for order in range(self.max_order):
                gamma[order] = alpha * (1.0 - alpha) ** order
            gamma[self.max_order] = (1.0 - alpha) ** self.max_order
        return gamma

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype == torch.long:
            x = x.float()
        text_feat = x[:, : self.text_dim]
        visual_feat = x[:, self.text_dim : self.text_dim + self.visual_dim]
        return text_feat, visual_feat

    @staticmethod
    def _edge_index_or_empty(
        edge_index: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if edge_index is None:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return edge_index.to(device=device, dtype=torch.long)

    def _edge_cosine_values(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h.new_empty((0,))
        src, dst = edge_index
        cosine = F.cosine_similarity(h[src], h[dst], dim=-1, eps=self.eps)
        return torch.nan_to_num(cosine, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
            -1.0, 1.0
        )

    def _edge_weight_from_cosine(self, cosine: torch.Tensor) -> torch.Tensor:
        weight = self.edge_weight_min + (1.0 - self.edge_weight_min) * torch.sigmoid(
            cosine / self.edge_weight_temperature
        )
        return torch.nan_to_num(
            weight,
            nan=1.0,
            posinf=1.0,
            neginf=self.edge_weight_min,
        ).clamp(self.edge_weight_min, 1.0)

    def _semantic_edge_weights(
        self,
        h_text: torch.Tensor,
        h_visual: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build two sparse semantic graphs only on the supplied edges."""
        cos_t = self._edge_cosine_values(h_text, edge_index)
        cos_v = self._edge_cosine_values(h_visual, edge_index)
        if self.edge_weight_mode == "raw_uniform":
            w_t = torch.ones_like(cos_t)
            w_v = torch.ones_like(cos_v)
        elif self.edge_weight_mode == "shared_avg_cos":
            shared = self._edge_weight_from_cosine(0.5 * (cos_t + cos_v))
            w_t = shared
            w_v = shared
        elif self.edge_weight_mode == "separate_cos":
            w_t = self._edge_weight_from_cosine(cos_t)
            w_v = self._edge_weight_from_cosine(cos_v)
        else:  # guarded in __init__, retained for type-checker exhaustiveness.
            raise RuntimeError(f"Unhandled edge_weight_mode: {self.edge_weight_mode}")
        return {
            "cos_t": cos_t,
            "cos_v": cos_v,
            "w_t": w_t,
            "w_v": w_v,
            "w_shared": 0.5 * (w_t + w_v),
        }

    def _normalized_operator(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sparse GCN-normalized edges, including optional self loops."""
        norm_edge_index, norm_edge_weight = gcn_norm(
            edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            improved=False,
            add_self_loops=self.diffusion_add_self_loops,
            flow="source_to_target",
            dtype=dtype,
        )
        if norm_edge_weight is None:
            norm_edge_weight = torch.ones(
                norm_edge_index.size(1), dtype=dtype, device=norm_edge_index.device
            )
        return norm_edge_index, torch.nan_to_num(
            norm_edge_weight,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    @staticmethod
    def _propagate_once(
        h: torch.Tensor,
        norm_edge_index: torch.Tensor,
        norm_edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if norm_edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = norm_edge_index
        return scatter(
            h[src] * norm_edge_weight.unsqueeze(-1),
            dst,
            dim=0,
            dim_size=h.size(0),
            reduce="sum",
        )

    def _propagation_bank(
        self,
        h0: torch.Tensor,
        norm_edge_index: torch.Tensor,
        norm_edge_weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Construct H_0, ..., H_K where H_(k+1)=Ahat H_k."""
        bases = [h0]
        h = h0
        for _ in range(self.max_order):
            h = self._propagate_once(h, norm_edge_index, norm_edge_weight)
            bases.append(h)
        return bases

    def _node_residuals(
        self,
        bases: list[torch.Tensor],
        projectors: nn.ModuleList,
        node_vectors: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_node_residual:
            return bases[0].new_zeros((bases[0].size(0), self.max_order + 1))
        residuals = []
        for order, base in enumerate(bases):
            q = torch.tanh(projectors[order](base))
            residuals.append(
                (q * node_vectors[order]).sum(dim=-1) / float(self.filter_rank)
            )
        return torch.stack(residuals, dim=-1)

    @staticmethod
    def _hrc_raw_loss(
        delta_node_text: torch.Tensor,
        delta_node_visual: torch.Tensor,
        node_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the raw hierarchical residual-centering penalty.

        The mean is over the supplied current training nodes.  If no index is
        supplied, all rows in the current model input are used, which keeps
        the helper convenient for direct analysis and unit tests.
        """
        if node_idx is not None:
            node_idx = node_idx.to(device=delta_node_text.device, dtype=torch.long)
            if node_idx.numel() == 0:
                return delta_node_text.new_zeros(())
            delta_node_text = delta_node_text.index_select(0, node_idx)
            delta_node_visual = delta_node_visual.index_select(0, node_idx)
        means = torch.stack(
            (delta_node_text.mean(dim=0), delta_node_visual.mean(dim=0)), dim=0
        )
        return means.square().mean()

    def _effective_coefficients(
        self,
        modality: str,
        node_residual: torch.Tensor,
    ) -> torch.Tensor:
        if modality == "text":
            delta_gamma = self.delta_gamma_text
        elif modality == "visual":
            delta_gamma = self.delta_gamma_visual
        else:
            raise ValueError(f"Unknown modality: {modality!r}")
        if not self.use_modality_residual:
            delta_gamma = torch.zeros_like(delta_gamma)
        return self.gamma_global.unsqueeze(0) + delta_gamma.unsqueeze(0) + node_residual

    @staticmethod
    def _filter_bases(bases: list[torch.Tensor], eta: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(bases[0])
        for order, base in enumerate(bases):
            output = output + eta[:, order : order + 1] * base
        return output

    @staticmethod
    def _effective_radius(eta: torch.Tensor) -> torch.Tensor:
        """Return the absolute-coefficient weighted propagation order.

        This is an analysis-only statistic. Using absolute coefficients keeps
        cancellation between polynomial terms from making a long-range filter
        look artificially short:

            r_i = sum_k k * |eta_i,k| / sum_k |eta_i,k|.
        """
        orders = torch.arange(eta.size(-1), device=eta.device, dtype=eta.dtype)
        weights = eta.abs()
        return (weights * orders.unsqueeze(0)).sum(dim=-1) / weights.sum(
            dim=-1
        ).clamp_min(torch.finfo(eta.dtype).eps)

    @torch.no_grad()
    def analysis_stats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Export learned filter coefficients without changing ``forward``."""
        was_training = self.training
        self.eval()
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)
        if was_training:
            self.train()
        return {
            "gamma_global": self.gamma_global.detach().clone(),
            "delta_gamma_text": self.delta_gamma_text.detach().clone(),
            "delta_gamma_visual": self.delta_gamma_visual.detach().clone(),
            "eta_text": components["eta_text"].detach().clone(),
            "eta_visual": components["eta_visual"].detach().clone(),
            "delta_node_text": components["delta_node_text"].detach().clone(),
            "delta_node_visual": components["delta_node_visual"].detach().clone(),
            "effective_radius_text": self._effective_radius(
                components["eta_text"]
            ).detach().clone(),
            "effective_radius_visual": self._effective_radius(
                components["eta_visual"]
            ).detach().clone(),
        }

    def _encode_components(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor]]:
        x_text, x_visual = self._split_features(x)
        h_text = self.text_proj(x_text)
        h_visual = self.visual_proj(x_visual)

        edges = self._semantic_edge_weights(h_text, h_visual, edge_index)
        norm_t_index, norm_t_weight = self._normalized_operator(
            edge_index,
            edges["w_t"],
            int(x.size(0)),
            h_text.dtype,
        )
        norm_v_index, norm_v_weight = self._normalized_operator(
            edge_index,
            edges["w_v"],
            int(x.size(0)),
            h_visual.dtype,
        )
        bases_text = self._propagation_bank(h_text, norm_t_index, norm_t_weight)
        bases_visual = self._propagation_bank(h_visual, norm_v_index, norm_v_weight)

        delta_node_text = self._node_residuals(
            bases_text, self.node_proj_text, self.node_vector_text
        )
        delta_node_visual = self._node_residuals(
            bases_visual, self.node_proj_visual, self.node_vector_visual
        )
        eta_text = self._effective_coefficients("text", delta_node_text)
        eta_visual = self._effective_coefficients("visual", delta_node_visual)
        z_text = self._filter_bases(bases_text, eta_text)
        z_visual = self._filter_bases(bases_visual, eta_visual)

        z_text_refined = self.text_refine_norm(
            z_text + self.text_refine_mlp(z_text)
        )
        z_visual_refined = self.visual_refine_norm(
            z_visual + self.visual_refine_mlp(z_visual)
        )
        fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
        skip = self.fusion_skip(fused_input)
        fusion = self.fusion_mlp(fused_input)
        z = self.output_norm(skip + fusion)

        return {
            "h_text": h_text,
            "h_visual": h_visual,
            "edges": edges,
            "norm_t_index": norm_t_index,
            "norm_t_weight": norm_t_weight,
            "norm_v_index": norm_v_index,
            "norm_v_weight": norm_v_weight,
            "bases_text": bases_text,
            "bases_visual": bases_visual,
            "delta_node_text": delta_node_text,
            "delta_node_visual": delta_node_visual,
            "eta_text": eta_text,
            "eta_visual": eta_visual,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": z,
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)
        z = torch.nan_to_num(components["z"], nan=0.0, posinf=1e4, neginf=-1e4)
        if self.hrc_weight == 0.0 or not self.use_node_residual:
            aux_loss = z.new_zeros(())
        else:
            aux_loss = self.hrc_weight * self._hrc_raw_loss(
                components["delta_node_text"],
                components["delta_node_visual"],
                self._hrc_training_idx,
            )
        return z, None, None, aux_loss, {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Exact full-graph MoPF inference; batching is intentionally unused."""
        del batch_size
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        edge_index = edge_index.to(device) if edge_index is not None else None
        z, _, _, _, _ = self(x.to(device), edge_index)
        return z.detach().cpu()


Model = MoPF
