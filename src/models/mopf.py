from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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

    EDGE_WEIGHT_MODES = {
        "separate_cos",
        "learned_diag_cos",
        "multi_perspective_cos_broken",
        "multi_perspective_cos",
        "shared_avg_cos",
        "raw_uniform",
    }

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

        self.num_metric_perspectives = int(
            cfg.model.get("num_metric_perspectives", 4)
        )
        if self.num_metric_perspectives != 4:
            raise ValueError(
                "MoPF U1 fixes num_metric_perspectives=4, got "
                f"{self.num_metric_perspectives}"
            )
        self.metric_init_seed = int(cfg.model.get("metric_init_seed", 20260910))
        self.metric_init_noise_std = float(
            cfg.model.get("metric_init_noise_std", 0.01)
        )
        if self.metric_init_noise_std <= 0.0:
            raise ValueError(
                "metric_init_noise_std must be positive, got "
                f"{self.metric_init_noise_std}"
            )

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
        self.node_conditioner_mode = str(
            cfg.model.get("node_conditioner_mode", "absolute")
        ).strip().lower()
        valid_conditioner_modes = {
            "absolute",
            "pdc",
            "pdc_v2_sep",
            "pdc_v2_full",
        }
        if self.node_conditioner_mode not in valid_conditioner_modes:
            raise ValueError(
                "model.node_conditioner_mode must be absolute|pdc|pdc_v2_sep|pdc_v2_full, got "
                f"{self.node_conditioner_mode!r}"
            )
        self.ppc_weight = float(cfg.model.get("ppc_weight", 0.0))
        if self.ppc_weight < 0.0:
            raise ValueError(f"ppc_weight must be >= 0, got {self.ppc_weight}")

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

        # Semantic metric upgrades are opt-in. R0 therefore has exactly the
        # Frozen MoPF-v0 parameter/state path. R1 has one metric vector per
        # modality; R2 has four deterministic, symmetry-broken vectors.
        if self.edge_weight_mode in {"learned_diag_cos", "multi_perspective_cos_broken"}:
            theta_init = self._make_metric_theta_initialization(
                hidden_dim, self.edge_weight_mode
            )
            perspective_count = (
                1
                if self.edge_weight_mode == "learned_diag_cos"
                else self.num_metric_perspectives
            )
            self.metric_theta_text = nn.Parameter(theta_init[:perspective_count].clone())
            self.metric_theta_visual = nn.Parameter(theta_init[:perspective_count].clone())
            if self.edge_weight_mode == "learned_diag_cos":
                self.metric_theta_text = nn.Parameter(self.metric_theta_text[0].clone())
                self.metric_theta_visual = nn.Parameter(self.metric_theta_visual[0].clone())
        elif self.edge_weight_mode == "multi_perspective_cos":
            # Preserve the historical U1 S1 parameterization exactly.
            theta_one = torch.tensor(1.0, dtype=torch.float32)
            theta_init = torch.log(torch.expm1(theta_one))
            self.metric_theta_text = nn.Parameter(
                torch.full(
                    (self.num_metric_perspectives, hidden_dim),
                    float(theta_init.item()),
                )
            )
            self.metric_theta_visual = nn.Parameter(
                torch.full(
                    (self.num_metric_perspectives, hidden_dim),
                    float(theta_init.item()),
                )
            )

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

        # PDC-v1 keeps an explicit unused theta_0 slot for checkpoint
        # compatibility.  PDC-v2-Sep fixes rho_0=0 and therefore stores only
        # orders 1..K; PDC-v2-Full stores a trainable theta_0 initialized to 0.
        theta_size = (
            self.max_order
            if self.node_conditioner_mode == "pdc_v2_sep"
            else self.max_order + 1
        )
        pdc_theta_text = torch.zeros(theta_size)
        pdc_theta_visual = torch.zeros(theta_size)
        if self.node_conditioner_mode in {"pdc", "pdc_v2_sep", "pdc_v2_full"}:
            self.pdc_theta_text = nn.Parameter(pdc_theta_text)
            self.pdc_theta_visual = nn.Parameter(pdc_theta_visual)
        else:
            self.register_buffer("pdc_theta_text", pdc_theta_text)
            self.register_buffer("pdc_theta_visual", pdc_theta_visual)

        if self.node_conditioner_mode in {"pdc_v2_sep", "pdc_v2_full"}:
            discrepancy_orders = (
                self.max_order if self.node_conditioner_mode == "pdc_v2_sep" else self.max_order + 1
            )
            self.node_disc_proj_text = nn.ModuleList(
                nn.Linear(hidden_dim, self.filter_rank)
                for _ in range(discrepancy_orders)
            )
            self.node_disc_proj_visual = nn.ModuleList(
                nn.Linear(hidden_dim, self.filter_rank)
                for _ in range(discrepancy_orders)
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

    def _make_metric_theta_initialization(
        self,
        hidden_dim: int,
        mode: str,
    ) -> torch.Tensor:
        """Create identifiable R1/R2 metric initialization in theta space."""
        theta_one = torch.tensor(1.0, dtype=torch.float32)
        inverse_one = torch.log(torch.expm1(theta_one))
        if mode == "learned_diag_cos":
            return torch.full((1, hidden_dim), float(inverse_one.item()))
        if mode != "multi_perspective_cos_broken":
            raise ValueError(f"Unsupported metric initialization mode: {mode!r}")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.metric_init_seed)
        noise = torch.randn(
            (self.num_metric_perspectives, hidden_dim),
            generator=generator,
            dtype=torch.float32,
        )
        noise = noise - noise.mean(dim=0, keepdim=True)
        current_rms = noise.square().mean().sqrt().clamp_min(torch.finfo(noise.dtype).eps)
        noise = noise * (self.metric_init_noise_std / current_rms)
        target_weights = 1.0 + noise
        if not bool((target_weights > 0.0).all()):
            raise ValueError("R2 metric initialization produced a non-positive target weight")
        return torch.log(torch.expm1(target_weights))

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

    def _edge_cosine_values_with_weights(
        self,
        h: torch.Tensor,
        metric_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return sparse edge scores for one or more positive metric vectors."""
        if metric_weights.dim() == 1:
            metric_weights = metric_weights.unsqueeze(0)
        if edge_index.numel() == 0:
            return h.new_empty((metric_weights.size(0), 0))
        src, dst = edge_index
        # Precompute one weighted hidden matrix per perspective. This avoids
        # repeating O(N*d) multiplication inside every edge chunk while the
        # chunked/checkpointed path keeps large-graph training memory bounded.
        chunk_size = 65536
        scores = []
        for perspective in range(metric_weights.size(0)):
            weighted_hidden = h * metric_weights[perspective].unsqueeze(0)
            chunks = []
            for start in range(0, edge_index.size(1), chunk_size):
                stop = min(start + chunk_size, edge_index.size(1))
                chunk_src = src[start:stop]
                chunk_dst = dst[start:stop]

                def _score_chunk(
                    weighted: torch.Tensor,
                    *,
                    chunk_src: torch.Tensor = chunk_src,
                    chunk_dst: torch.Tensor = chunk_dst,
                ) -> torch.Tensor:
                    return F.cosine_similarity(
                        weighted[chunk_src], weighted[chunk_dst], dim=-1, eps=self.eps
                    )

                if torch.is_grad_enabled() and weighted_hidden.requires_grad:
                    score = checkpoint(
                        _score_chunk,
                        weighted_hidden,
                        use_reentrant=False,
                    )
                else:
                    score = _score_chunk(weighted_hidden)
                chunks.append(
                    torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
                        -1.0, 1.0
                    )
                )
            scores.append(torch.cat(chunks, dim=0))
        return torch.stack(scores, dim=0)

    def _multi_perspective_cosine_values(
        self,
        h: torch.Tensor,
        theta: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return U1's unnormalized positive metric perspective scores."""
        return self._edge_cosine_values_with_weights(
            h, F.softplus(theta), edge_index
        )

    def _normalized_metric_weights(self, theta: torch.Tensor) -> torch.Tensor:
        raw_weights = F.softplus(theta)
        if raw_weights.dim() == 1:
            raw_weights = raw_weights.unsqueeze(0)
        return raw_weights / (
            raw_weights.mean(dim=-1, keepdim=True) + self.eps
        )

    def _resolve_metric_override(
        self,
        modality: str,
        override: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.edge_weight_mode == "learned_diag_cos":
            expected = (1, self.hidden_dim)
        elif self.edge_weight_mode == "multi_perspective_cos_broken":
            expected = (self.num_metric_perspectives, self.hidden_dim)
        else:
            raise ValueError(
                "Metric overrides require learned_diag_cos or "
                "multi_perspective_cos_broken"
            )
        if override is None:
            theta = getattr(self, f"metric_theta_{modality}")
            return self._normalized_metric_weights(theta).to(device=device, dtype=dtype)
        weights = torch.as_tensor(override, device=device, dtype=dtype)
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)
        if tuple(weights.shape) != expected:
            raise ValueError(
                f"{modality} metric override must have shape {expected}, "
                f"got {tuple(weights.shape)}"
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            raise ValueError(f"{modality} metric override must be finite and positive")
        return weights / (weights.mean(dim=-1, keepdim=True) + self.eps)

    def _resolve_edge_weight_temperature(
        self, temperature_override: float | None = None
    ) -> float:
        """Resolve a fixed analysis-time temperature without mutating state."""
        temperature = (
            self.edge_weight_temperature
            if temperature_override is None
            else float(temperature_override)
        )
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                "temperature_override must be a finite positive scalar, got "
                f"{temperature_override!r}"
            )
        return temperature

    def _edge_weight_from_cosine(
        self,
        cosine: torch.Tensor,
        *,
        temperature_override: float | None = None,
    ) -> torch.Tensor:
        temperature = self._resolve_edge_weight_temperature(temperature_override)
        weight = self.edge_weight_min + (1.0 - self.edge_weight_min) * torch.sigmoid(
            cosine / temperature
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
        *,
        metric_weights_text: torch.Tensor | None = None,
        metric_weights_visual: torch.Tensor | None = None,
        temperature_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build two sparse semantic graphs only on the supplied edges."""
        cos_t = self._edge_cosine_values(h_text, edge_index)
        cos_v = self._edge_cosine_values(h_visual, edge_index)
        perspective_cos_t = cos_t.unsqueeze(0)
        perspective_cos_v = cos_v.unsqueeze(0)
        if self.edge_weight_mode == "raw_uniform":
            w_t = torch.ones_like(cos_t)
            w_v = torch.ones_like(cos_v)
        elif self.edge_weight_mode == "shared_avg_cos":
            shared = self._edge_weight_from_cosine(
                0.5 * (cos_t + cos_v),
                temperature_override=temperature_override,
            )
            w_t = shared
            w_v = shared
        elif self.edge_weight_mode == "separate_cos":
            w_t = self._edge_weight_from_cosine(
                cos_t, temperature_override=temperature_override
            )
            w_v = self._edge_weight_from_cosine(
                cos_v, temperature_override=temperature_override
            )
        elif self.edge_weight_mode == "multi_perspective_cos":
            perspective_cos_t = self._multi_perspective_cosine_values(
                h_text, self.metric_theta_text, edge_index
            )
            perspective_cos_v = self._multi_perspective_cosine_values(
                h_visual, self.metric_theta_visual, edge_index
            )
            cos_t = perspective_cos_t.mean(dim=0)
            cos_v = perspective_cos_v.mean(dim=0)
            w_t = self._edge_weight_from_cosine(
                cos_t, temperature_override=temperature_override
            )
            w_v = self._edge_weight_from_cosine(
                cos_v, temperature_override=temperature_override
            )
        elif self.edge_weight_mode in {
            "learned_diag_cos",
            "multi_perspective_cos_broken",
        }:
            weights_t = self._resolve_metric_override(
                "text", metric_weights_text, h_text.device, h_text.dtype
            )
            weights_v = self._resolve_metric_override(
                "visual", metric_weights_visual, h_visual.device, h_visual.dtype
            )
            perspective_cos_t = self._edge_cosine_values_with_weights(
                h_text, weights_t, edge_index
            )
            perspective_cos_v = self._edge_cosine_values_with_weights(
                h_visual, weights_v, edge_index
            )
            cos_t = perspective_cos_t.mean(dim=0)
            cos_v = perspective_cos_v.mean(dim=0)
            w_t = self._edge_weight_from_cosine(
                cos_t, temperature_override=temperature_override
            )
            w_v = self._edge_weight_from_cosine(
                cos_v, temperature_override=temperature_override
            )
        else:  # guarded in __init__, retained for type-checker exhaustiveness.
            raise RuntimeError(f"Unhandled edge_weight_mode: {self.edge_weight_mode}")
        return {
            "cos_t": cos_t,
            "cos_v": cos_v,
            "perspective_cos_t": perspective_cos_t,
            "perspective_cos_v": perspective_cos_v,
            "w_t": w_t,
            "w_v": w_v,
            "w_shared": 0.5 * (w_t + w_v),
            "metric_weights_text": (
                self._resolve_metric_override(
                    "text", metric_weights_text, h_text.device, h_text.dtype
                )
                if self.edge_weight_mode
                in {"learned_diag_cos", "multi_perspective_cos_broken"}
                else h_text.new_empty((0, h_text.size(1)))
            ),
            "metric_weights_visual": (
                self._resolve_metric_override(
                    "visual", metric_weights_visual, h_visual.device, h_visual.dtype
                )
                if self.edge_weight_mode
                in {"learned_diag_cos", "multi_perspective_cos_broken"}
                else h_visual.new_empty((0, h_visual.size(1)))
            ),
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

    @torch.no_grad()
    def conductance_stats(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        temperature_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return sparse conductance/operator diagnostics without propagation.

        This is an analysis-only path. It uses the same projections, edge
        support, conductance transform, and ``gcn_norm`` calls as ``forward``
        but deliberately does not construct the propagation bank or alter any
        training state. The returned perspective score rows are ``[P, E]``;
        S0 is represented by one ordinary-cosine row for uniform diagnostics.
        """
        was_training = self.training
        self.eval()
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        x_text, x_visual = self._split_features(x)
        h_text = self.text_proj(x_text)
        h_visual = self.visual_proj(x_visual)
        edges = self._semantic_edge_weights(
            h_text,
            h_visual,
            edge_index,
            temperature_override=temperature_override,
        )
        norm_t_index, norm_t_weight = self._normalized_operator(
            edge_index, edges["w_t"], int(x.size(0)), h_text.dtype
        )
        norm_v_index, norm_v_weight = self._normalized_operator(
            edge_index, edges["w_v"], int(x.size(0)), h_visual.dtype
        )
        if was_training:
            self.train()
        return {
            "src": edge_index[0].detach().clone(),
            "dst": edge_index[1].detach().clone(),
            "cos_text": edges["cos_t"].detach().clone(),
            "cos_visual": edges["cos_v"].detach().clone(),
            "conductance_text": edges["w_t"].detach().clone(),
            "conductance_visual": edges["w_v"].detach().clone(),
            "perspective_cos_text": edges["perspective_cos_t"].detach().clone(),
            "perspective_cos_visual": edges["perspective_cos_v"].detach().clone(),
            "metric_weights_text": edges["metric_weights_text"].detach().clone(),
            "metric_weights_visual": edges["metric_weights_visual"].detach().clone(),
            "norm_text_index": norm_t_index.detach().clone(),
            "norm_text_weight": norm_t_weight.detach().clone(),
            "norm_visual_index": norm_v_index.detach().clone(),
            "norm_visual_weight": norm_v_weight.detach().clone(),
        }

    @torch.no_grad()
    def metric_perspective_weights(self, modality: str) -> torch.Tensor:
        """Return positive learned metric weights for U1 diagnostics."""
        if self.edge_weight_mode in {
            "learned_diag_cos",
            "multi_perspective_cos_broken",
        }:
            return self.semantic_metric_weights(modality)
        if self.edge_weight_mode != "multi_perspective_cos":
            return torch.ones(
                (1, self.hidden_dim),
                device=next(self.parameters()).device,
                dtype=next(self.parameters()).dtype,
            )
        if modality == "text":
            theta = self.metric_theta_text
        elif modality == "visual":
            theta = self.metric_theta_visual
        else:
            raise ValueError(f"Unknown modality: {modality!r}")
        return F.softplus(theta).detach().clone()

    @torch.no_grad()
    def semantic_metric_weights(self, modality: str) -> torch.Tensor:
        """Return normalized R1/R2 metric weights for analysis."""
        if self.edge_weight_mode not in {
            "learned_diag_cos",
            "multi_perspective_cos_broken",
        }:
            raise ValueError(
                "semantic_metric_weights requires a learned R1/R2 metric mode"
            )
        theta = getattr(self, f"metric_theta_{modality}")
        return self._normalized_metric_weights(theta).detach().clone()

    @torch.no_grad()
    def analysis_encode_with_metric_override(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        text_weights: torch.Tensor | None = None,
        visual_weights: torch.Tensor | None = None,
        temperature_override: float | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor]]:
        """Frozen functional audit path for R1/R2 metric interventions."""
        if self.edge_weight_mode not in {
            "learned_diag_cos",
            "multi_perspective_cos_broken",
        }:
            raise ValueError("Metric interventions require an R1/R2 model")
        was_training = self.training
        self.eval()
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(
            x,
            edge_index,
            metric_weights_text=text_weights,
            metric_weights_visual=visual_weights,
            temperature_override=temperature_override,
        )
        if was_training:
            self.train()
        return components

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

    def pdc_rho(self, modality: str) -> torch.Tensor:
        """Return PDC's bounded scalar calibration profile.

        PDC-v1 and PDC-v2-Sep keep rho_0 fixed at zero.  PDC-v2-Full exposes
        rho_0 as a trainable value initialized at zero.
        """
        if modality == "text":
            theta = self.pdc_theta_text
        elif modality == "visual":
            theta = self.pdc_theta_visual
        else:
            raise ValueError(f"Unknown modality: {modality!r}")
        rho = torch.tanh(theta)
        if self.node_conditioner_mode == "pdc_v2_sep":
            return torch.cat((rho[:1] * 0.0, rho), dim=0)
        if self.node_conditioner_mode == "pdc_v2_full":
            return rho
        return torch.cat((rho[:1] * 0.0, rho[1:]), dim=0)

    def _conditioner_bases(
        self,
        bases: list[torch.Tensor],
        modality: str,
        rho_override: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Build absolute or PDC-conditioned inputs for node residuals."""
        if self.node_conditioner_mode == "absolute":
            return bases
        rho = self.pdc_rho(modality) if rho_override is None else rho_override
        conditioned = [bases[0]]
        for order in range(1, len(bases)):
            discrepancy = bases[order] - bases[order - 1]
            discrepancy_norm = F.layer_norm(
                discrepancy,
                (discrepancy.size(-1),),
                weight=None,
                bias=None,
                eps=self.eps,
            )
            conditioned.append(bases[order] + rho[order] * discrepancy_norm)
        return conditioned

    def _node_residuals(
        self,
        bases: list[torch.Tensor],
        projectors: nn.ModuleList,
        node_vectors: torch.Tensor,
        modality: str,
        rho_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        if not self.use_node_residual:
            conditioned_bases = (
                bases
                if self.node_conditioner_mode in {"pdc_v2_sep", "pdc_v2_full"}
                else self._conditioner_bases(bases, modality, rho_override)
            )
            return (
                bases[0].new_zeros((bases[0].size(0), self.max_order + 1)),
                conditioned_bases,
                {
                    "discrepancy": [torch.zeros_like(base) for base in bases],
                    "discrepancy_norm": [torch.zeros_like(base) for base in bases],
                    "branch": [
                        bases[0].new_zeros((bases[0].size(0), self.filter_rank))
                        for _ in bases
                    ],
                    "state_projection": [
                        bases[0].new_zeros((bases[0].size(0), self.filter_rank))
                        for _ in bases
                    ],
                },
            )

        if self.node_conditioner_mode in {"pdc_v2_sep", "pdc_v2_full"}:
            if modality == "text":
                discrepancy_projectors = self.node_disc_proj_text
            elif modality == "visual":
                discrepancy_projectors = self.node_disc_proj_visual
            else:
                raise ValueError(f"Unknown modality: {modality!r}")
            rho = self.pdc_rho(modality) if rho_override is None else rho_override
            residuals = []
            discrepancies: list[torch.Tensor] = []
            discrepancy_norms: list[torch.Tensor] = []
            branches: list[torch.Tensor] = []
            state_projections: list[torch.Tensor] = []
            for order, base in enumerate(bases):
                state_projection = projectors[order](base)
                if self.node_conditioner_mode == "pdc_v2_full":
                    discrepancy = (
                        bases[1] - bases[0] if order == 0 else bases[order] - bases[order - 1]
                    )
                    discrepancy_norm = F.layer_norm(
                        discrepancy,
                        (discrepancy.size(-1),),
                        weight=None,
                        bias=None,
                        eps=self.eps,
                    )
                    disc_projection = discrepancy_projectors[order](discrepancy_norm)
                elif order == 0:
                    discrepancy = torch.zeros_like(base)
                    discrepancy_norm = torch.zeros_like(base)
                    disc_projection = base.new_zeros(
                        (base.size(0), self.filter_rank)
                    )
                else:
                    discrepancy = bases[order] - bases[order - 1]
                    discrepancy_norm = F.layer_norm(
                        discrepancy,
                        (discrepancy.size(-1),),
                        weight=None,
                        bias=None,
                        eps=self.eps,
                    )
                    disc_projection = discrepancy_projectors[order - 1](discrepancy_norm)
                branch = rho[order] * disc_projection
                q = torch.tanh(state_projection + branch)
                residuals.append(
                    (q * node_vectors[order]).sum(dim=-1) / float(self.filter_rank)
                )
                discrepancies.append(discrepancy)
                discrepancy_norms.append(discrepancy_norm)
                branches.append(branch)
                state_projections.append(state_projection)
            return (
                torch.stack(residuals, dim=-1),
                bases,
                {
                    "discrepancy": discrepancies,
                    "discrepancy_norm": discrepancy_norms,
                    "branch": branches,
                    "state_projection": state_projections,
                },
            )

        conditioner_bases = self._conditioner_bases(
            bases, modality, rho_override
        )
        residuals = []
        rho = self.pdc_rho(modality) if rho_override is None else rho_override
        discrepancies = [torch.zeros_like(base) for base in bases]
        discrepancy_norms = [torch.zeros_like(base) for base in bases]
        branches = [torch.zeros_like(base) for base in bases]
        state_projections = []
        for order, base in enumerate(conditioner_bases):
            state_projection = projectors[order](base)
            q = torch.tanh(state_projection)
            residuals.append(
                (q * node_vectors[order]).sum(dim=-1) / float(self.filter_rank)
            )
            state_projections.append(state_projection)
            if self.node_conditioner_mode == "pdc" and order >= 1:
                discrepancies[order] = bases[order] - bases[order - 1]
                discrepancy_norms[order] = F.layer_norm(
                    discrepancies[order],
                    (discrepancies[order].size(-1),),
                    weight=None,
                    bias=None,
                    eps=self.eps,
                )
                branches[order] = rho[order] * discrepancy_norms[order]
        return (
            torch.stack(residuals, dim=-1),
            conditioner_bases,
            {
                "discrepancy": discrepancies,
                "discrepancy_norm": discrepancy_norms,
                "branch": branches,
                "state_projection": state_projections,
            },
        )

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

    @staticmethod
    def _ppc_raw_loss(
        delta_node_text_1: torch.Tensor,
        delta_node_visual_1: torch.Tensor,
        delta_node_text_2: torch.Tensor,
        delta_node_visual_2: torch.Tensor,
        node_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compare centered node profiles between two stochastic views."""
        if node_idx is not None:
            node_idx = node_idx.to(
                device=delta_node_text_1.device,
                dtype=torch.long,
            )
            if node_idx.numel() == 0:
                return delta_node_text_1.new_zeros(())
            tensors = (
                delta_node_text_1,
                delta_node_visual_1,
                delta_node_text_2,
                delta_node_visual_2,
            )
            delta_node_text_1, delta_node_visual_1, delta_node_text_2, delta_node_visual_2 = (
                tensor.index_select(0, node_idx) for tensor in tensors
            )

        centered_profiles = []
        for first, second in (
            (delta_node_text_1, delta_node_text_2),
            (delta_node_visual_1, delta_node_visual_2),
        ):
            centered_first = first - first.mean(dim=0, keepdim=True)
            centered_second = second - second.mean(dim=0, keepdim=True)
            centered_profiles.append((centered_first - centered_second).square())
        return torch.stack(centered_profiles, dim=0).mean()

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
        *,
        pdc_rho_text: torch.Tensor | None = None,
        pdc_rho_visual: torch.Tensor | None = None,
        metric_weights_text: torch.Tensor | None = None,
        metric_weights_visual: torch.Tensor | None = None,
        temperature_override: float | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor]]:
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

        delta_node_text, conditioned_bases_text, pdc_aux_text = self._node_residuals(
            bases_text,
            self.node_proj_text,
            self.node_vector_text,
            "text",
            pdc_rho_text,
        )
        delta_node_visual, conditioned_bases_visual, pdc_aux_visual = self._node_residuals(
            bases_visual,
            self.node_proj_visual,
            self.node_vector_visual,
            "visual",
            pdc_rho_visual,
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
            "conditioned_bases_text": conditioned_bases_text,
            "conditioned_bases_visual": conditioned_bases_visual,
            "pdc_aux_text": pdc_aux_text,
            "pdc_aux_visual": pdc_aux_visual,
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

    def _analysis_rho_from_mask(
        self,
        modality: str,
        mask: torch.Tensor | list[float] | tuple[float, ...] | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Resolve an analysis-only PDC mask without mutating parameters."""
        if mask is None:
            return None
        if self.node_conditioner_mode not in {
            "pdc",
            "pdc_v2_sep",
            "pdc_v2_full",
        }:
            raise ValueError(
                "PDC analysis masks require model.node_conditioner_mode='pdc'"
            )
        mask_tensor = torch.as_tensor(mask, device=device)
        if mask_tensor.numel() != self.max_order + 1:
            raise ValueError(
                f"PDC mask for {modality} must have {self.max_order + 1} values, "
                f"got {mask_tensor.numel()}"
            )
        mask_tensor = mask_tensor.reshape(self.max_order + 1).to(dtype=self.pdc_theta_text.dtype)
        if not bool(torch.isfinite(mask_tensor).all()):
            raise ValueError(f"PDC mask for {modality} contains non-finite values")
        if bool(((mask_tensor < 0.0) | (mask_tensor > 1.0)).any()):
            raise ValueError(f"PDC mask for {modality} must be in [0, 1]")
        # Preserve bitwise equivalence with the formal path for the common
        # all-ones intervention.  Multiplying by one is mathematically
        # identical but can introduce a small GPU rounding difference.
        if bool(torch.equal(mask_tensor, torch.ones_like(mask_tensor))):
            return None
        # rho_0 is always zero by definition.  Ignore any caller value at k=0
        # so the diagnostic cannot accidentally alter the zero-order path.
        mask_tensor = mask_tensor.clone()
        mask_tensor[0] = 0.0
        rho = self.pdc_rho(modality).to(device=device)
        return rho * mask_tensor

    @torch.no_grad()
    def analysis_encode_with_pdc_mask(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        *,
        text_mask: torch.Tensor | list[float] | tuple[float, ...] | None = None,
        visual_mask: torch.Tensor | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | dict[str, torch.Tensor]]:
        """Encode with frozen PDC masks for mechanism diagnostics only.

        The learned checkpoint parameters are never changed.  ``None`` means
        use the normal learned rho profile; a mask value of zero removes only
        that order's discrepancy correction.  This helper is intentionally
        separate from ``forward``/``inference`` so formal training and default
        inference cannot silently inherit an analysis intervention.
        """
        if self.node_conditioner_mode not in {
            "pdc",
            "pdc_v2_sep",
            "pdc_v2_full",
        }:
            raise ValueError(
                "analysis_encode_with_pdc_mask is only defined for PDC models"
            )
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        rho_text = self._analysis_rho_from_mask("text", text_mask, x.device)
        rho_visual = self._analysis_rho_from_mask("visual", visual_mask, x.device)
        return self._encode_components(
            x,
            edge_index,
            pdc_rho_text=rho_text,
            pdc_rho_visual=rho_visual,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None):
        edge_index = self._edge_index_or_empty(edge_index, x.device)
        components = self._encode_components(x, edge_index)
        z = torch.nan_to_num(components["z"], nan=0.0, posinf=1e4, neginf=-1e4)
        aux_info: dict[str, torch.Tensor] = {}
        ppc_active = (
            self.training
            and self.ppc_weight > 0.0
            and self.use_node_residual
        )
        if ppc_active:
            # View 1 remains the sole task embedding. View 2 exists only for
            # the auxiliary profile-consistency calculation.
            components_2 = self._encode_components(x, edge_index)
            ppc_raw = self._ppc_raw_loss(
                components["delta_node_text"],
                components["delta_node_visual"],
                components_2["delta_node_text"],
                components_2["delta_node_visual"],
                self._hrc_training_idx,
            )
            aux_loss = self.ppc_weight * ppc_raw
            aux_info["ppc_raw"] = ppc_raw.detach()
        elif self.hrc_weight == 0.0 or not self.use_node_residual:
            aux_loss = z.new_zeros(())
        else:
            aux_loss = self.hrc_weight * self._hrc_raw_loss(
                components["delta_node_text"],
                components["delta_node_visual"],
                self._hrc_training_idx,
            )
        return z, None, None, aux_loss, aux_info

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
