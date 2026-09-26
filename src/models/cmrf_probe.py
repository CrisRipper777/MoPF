from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    remove_self_loops,
    to_undirected,
)


class _StateEncoder(nn.Module):
    """Project H0,R1,R2,R3 independently and concatenate the token summaries."""

    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.tokens = nn.ModuleList(nn.Linear(hidden_dim, latent_dim) for _ in range(4))
        self.out_dim = 4 * latent_dim

    def forward(self, h0: torch.Tensor, responses: list[torch.Tensor]) -> torch.Tensor:
        values = [h0, *responses]
        return torch.cat(
            [torch.relu(layer(value)) for layer, value in zip(self.tokens, values)],
            dim=-1,
        )


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _zero_last_layer(module: nn.Module) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


class Model(nn.Module):
    """Independent multi-order structural utility probe and soft controller.

    Propagation uses the fixed unit-weight physical graph operator from the
    baseline Multi-Order Bank. Cross-modal information reaches the embedding
    only through response coefficients; late fusion is the same plain MLP for
    every composition mode.
    """

    MAX_ORDER = 3
    BASE_COEFFICIENTS = (0.75, 0.50, 0.25)
    COMPOSITION_MODES = {"uniform", "own_soft", "cross_soft", "gated_cross_soft"}
    requires_full_graph_training = True

    def __init__(self, cfg, data_info):
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        self.input_dim = int(data_info.get("input_dim", self.text_dim + self.visual_dim))
        if self.text_dim <= 0 or self.visual_dim <= 0:
            raise ValueError("cmrf_probe requires positive text_dim and visual_dim")
        if self.input_dim != self.text_dim + self.visual_dim:
            raise ValueError("cmrf_probe expects concatenated [text, visual] input features")

        self.hidden_dim = int(cfg.model.get("hidden_dim", 256))
        if int(cfg.model.get("max_order", self.MAX_ORDER)) != self.MAX_ORDER:
            raise ValueError("cmrf_probe fixes max_order=3")
        self.max_order = self.MAX_ORDER
        self.num_layers = self.MAX_ORDER
        self.dropout = float(cfg.model.get("dropout", 0.2))
        self.composition_mode = str(cfg.model.get("composition_mode", "uniform")).lower()
        if self.composition_mode not in self.COMPOSITION_MODES:
            raise ValueError(f"unknown composition_mode={self.composition_mode!r}")
        self.controller_lambda = 0.25
        if float(cfg.model.get("controller_lambda", 0.25)) != self.controller_lambda:
            raise ValueError("cmrf_probe fixes controller_lambda at 0.25")

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.visual_projector = nn.Sequential(
            nn.Linear(self.visual_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        latent_dim = int(cfg.model.get("controller_latent_dim", 32))
        controller_hidden = int(cfg.model.get("controller_hidden_dim", 64))
        if self.composition_mode == "uniform":
            self.state_encoder_text = None
            self.state_encoder_visual = None
            qdim = 4 * latent_dim
        else:
            self.state_encoder_text = _StateEncoder(self.hidden_dim, latent_dim)
            self.state_encoder_visual = _StateEncoder(self.hidden_dim, latent_dim)
            qdim = self.state_encoder_text.out_dim

        self.delta_text_own = None
        self.delta_visual_own = None
        self.delta_text_cross = None
        self.delta_visual_cross = None
        self.delta_text_cross_branch = None
        self.delta_visual_cross_branch = None
        self.rho_visual_to_text = None
        self.rho_text_to_visual = None
        if self.composition_mode == "own_soft":
            self.delta_text_own = _mlp(qdim, controller_hidden, 3)
            self.delta_visual_own = _mlp(qdim, controller_hidden, 3)
            _zero_last_layer(self.delta_text_own)
            _zero_last_layer(self.delta_visual_own)
        elif self.composition_mode == "cross_soft":
            self.delta_text_cross = _mlp(2 * qdim, controller_hidden, 3)
            self.delta_visual_cross = _mlp(2 * qdim, controller_hidden, 3)
            _zero_last_layer(self.delta_text_cross)
            _zero_last_layer(self.delta_visual_cross)
        elif self.composition_mode == "gated_cross_soft":
            self.delta_text_own = _mlp(qdim, controller_hidden, 3)
            self.delta_visual_own = _mlp(qdim, controller_hidden, 3)
            self.delta_text_cross_branch = _mlp(2 * qdim, controller_hidden, 3)
            self.delta_visual_cross_branch = _mlp(2 * qdim, controller_hidden, 3)
            self.rho_visual_to_text = _mlp(2 * qdim, controller_hidden, 3)
            self.rho_text_to_visual = _mlp(2 * qdim, controller_hidden, 3)
            for module in (
                self.delta_text_own,
                self.delta_visual_own,
                self.delta_text_cross_branch,
                self.delta_visual_cross_branch,
            ):
                _zero_last_layer(module)

        self.register_buffer(
            "base_coefficients",
            torch.tensor(self.BASE_COEFFICIENTS, dtype=torch.float32),
        )
        self.plain_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.out_dim = self.hidden_dim
        self._operator_cache_key = None
        self._operator_cache_edge_index = None
        self._operator_cache = None

    def _build_propagation_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build P=D_tilde^-1/2(A+I)D_tilde^-1/2 using unit edge weights."""
        edge_index = edge_index.long()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        values = torch.ones(row.numel(), dtype=dtype, device=edge_index.device)
        degree = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        degree.index_add_(0, row, values)
        inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
        normalized = inv_sqrt[row] * values * inv_sqrt[col]
        return torch.sparse_coo_tensor(
            edge_index,
            normalized,
            (num_nodes, num_nodes),
            dtype=dtype,
            device=edge_index.device,
        ).coalesce()

    def _get_operator(
        self, edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (
            edge_index.data_ptr(),
            int(getattr(edge_index, "_version", 0)),
            tuple(edge_index.shape),
            int(num_nodes),
            edge_index.device,
            dtype,
        )
        if self._operator_cache_key == key and self._operator_cache is not None:
            return self._operator_cache
        operator = self._build_propagation_operator(edge_index, num_nodes, dtype)
        self._operator_cache_key = key
        self._operator_cache_edge_index = edge_index
        self._operator_cache = operator
        return operator

    @staticmethod
    def _propagate(h0: torch.Tensor, operator: torch.Tensor) -> list[torch.Tensor]:
        states = [h0]
        for _ in range(3):
            states.append(torch.sparse.mm(operator, states[-1]))
        return states

    @staticmethod
    def _responses(states: list[torch.Tensor]) -> list[torch.Tensor]:
        return [states[k] - states[k - 1] for k in range(1, 4)]

    def _coefficients(self, q_text: torch.Tensor, q_visual: torch.Tensor):
        n = q_text.size(0)
        base = self.base_coefficients.to(
            device=q_text.device, dtype=q_text.dtype
        ).expand(n, -1)
        zero = torch.zeros_like(base)
        rho_v2t = rho_t2v = None
        if self.composition_mode == "uniform":
            delta_t = delta_v = zero
        elif self.composition_mode == "own_soft":
            delta_t = self.controller_lambda * torch.tanh(self.delta_text_own(q_text))
            delta_v = self.controller_lambda * torch.tanh(self.delta_visual_own(q_visual))
        elif self.composition_mode == "cross_soft":
            delta_t = self.controller_lambda * torch.tanh(
                self.delta_text_cross(torch.cat([q_text, q_visual], dim=-1))
            )
            delta_v = self.controller_lambda * torch.tanh(
                self.delta_visual_cross(torch.cat([q_visual, q_text], dim=-1))
            )
        else:
            v2t = torch.cat([q_text, q_visual], dim=-1)
            t2v = torch.cat([q_visual, q_text], dim=-1)
            rho_v2t = torch.sigmoid(self.rho_visual_to_text(v2t))
            rho_t2v = torch.sigmoid(self.rho_text_to_visual(t2v))
            # Half-budget each bounded branch so the combined C3 delta stays
            # within the same fixed +/-0.25 residual range as C1/C2.
            branch_bound = self.controller_lambda * 0.5
            own_t = branch_bound * torch.tanh(self.delta_text_own(q_text))
            own_v = branch_bound * torch.tanh(self.delta_visual_own(q_visual))
            cross_t = branch_bound * torch.tanh(
                self.delta_text_cross_branch(v2t)
            )
            cross_v = branch_bound * torch.tanh(
                self.delta_visual_cross_branch(t2v)
            )
            delta_t = own_t + rho_v2t * cross_t
            delta_v = own_v + rho_t2v * cross_v
        return base + delta_t, base + delta_v, rho_v2t, rho_t2v

    @staticmethod
    def _compose_one(
        h0: torch.Tensor, responses: list[torch.Tensor], coefficients: torch.Tensor
    ) -> torch.Tensor:
        # H0 has a hard-coded unit coefficient; controllers cannot modify it.
        z = h0
        for k, response in enumerate(responses):
            z = z + coefficients[:, k : k + 1] * response
        return z

    def _fuse(self, z_text: torch.Tensor, z_visual: torch.Tensor) -> torch.Tensor:
        return self.plain_fusion(torch.cat([z_text, z_visual], dim=-1))

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
        if edge_index is None:
            raise ValueError("cmrf_probe requires the physical edge_index")
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"expected x shape [N,{self.input_dim}], got {tuple(x.shape)}"
            )
        operator = self._get_operator(edge_index.to(x.device), x.size(0), x.dtype)
        h0_text = self.text_projector(x[:, : self.text_dim])
        h0_visual = self.visual_projector(
            x[:, self.text_dim : self.text_dim + self.visual_dim]
        )
        states_text = self._propagate(h0_text, operator)
        states_visual = self._propagate(h0_visual, operator)
        responses_text = self._responses(states_text)
        responses_visual = self._responses(states_visual)
        if self.state_encoder_text is None:
            # Uniform C0 has no unused controller parameters.
            q_text, q_visual = h0_text, h0_visual
        else:
            q_text = self.state_encoder_text(h0_text, responses_text)
            q_visual = self.state_encoder_visual(h0_visual, responses_visual)
        a_text, a_visual, rho_v2t, rho_t2v = self._coefficients(q_text, q_visual)
        z_text = self._compose_one(h0_text, responses_text, a_text)
        z_visual = self._compose_one(h0_visual, responses_visual, a_visual)
        fused = self._fuse(z_text, z_visual)
        return {
            "H0_text": h0_text,
            "H0_visual": h0_visual,
            "S_text": states_text,
            "S_visual": states_visual,
            "R_text": responses_text,
            "R_visual": responses_visual,
            "q_text": q_text,
            "q_visual": q_visual,
            "a_text": a_text,
            "a_visual": a_visual,
            "rho_V2T": rho_v2t,
            "rho_T2V": rho_t2v,
            "Z_text": z_text,
            "Z_visual": z_visual,
            "fused_z": fused,
        }

    def recompose(
        self,
        analysis: dict,
        *,
        q_text: torch.Tensor | None = None,
        q_visual: torch.Tensor | None = None,
    ) -> dict:
        """Recompute controller/fusion from frozen states and optional control inputs."""
        q_text = analysis["q_text"] if q_text is None else q_text
        q_visual = analysis["q_visual"] if q_visual is None else q_visual
        a_text, a_visual, rho_v2t, rho_t2v = self._coefficients(q_text, q_visual)
        z_text = self._compose_one(analysis["H0_text"], analysis["R_text"], a_text)
        z_visual = self._compose_one(
            analysis["H0_visual"], analysis["R_visual"], a_visual
        )
        return {
            "a_text": a_text,
            "a_visual": a_visual,
            "rho_V2T": rho_v2t,
            "rho_T2V": rho_t2v,
            "Z_text": z_text,
            "Z_visual": z_visual,
            "fused_z": self._fuse(z_text, z_visual),
        }

    def intervene_companion(
        self,
        analysis: dict,
        *,
        target_modality: str,
        companion_q: torch.Tensor,
    ) -> dict:
        """Change one controller's companion input and freeze the other path."""
        q_text = analysis["q_text"]
        q_visual = analysis["q_visual"]
        base_t, base_v, base_rho_v2t, base_rho_t2v = self._coefficients(
            q_text, q_visual
        )
        if target_modality.lower() == "text":
            changed_t, _, changed_rho_v2t, _ = self._coefficients(
                q_text, companion_q
            )
            a_text, a_visual = changed_t, base_v
            rho_v2t, rho_t2v = changed_rho_v2t, base_rho_t2v
        elif target_modality.lower() == "visual":
            _, changed_v, _, changed_rho_t2v = self._coefficients(
                companion_q, q_visual
            )
            a_text, a_visual = base_t, changed_v
            rho_v2t, rho_t2v = base_rho_v2t, changed_rho_t2v
        else:
            raise ValueError("target_modality must be Text or Visual")
        z_text = self._compose_one(analysis["H0_text"], analysis["R_text"], a_text)
        z_visual = self._compose_one(
            analysis["H0_visual"], analysis["R_visual"], a_visual
        )
        return {
            "a_text": a_text,
            "a_visual": a_visual,
            "rho_V2T": rho_v2t,
            "rho_T2V": rho_t2v,
            "Z_text": z_text,
            "Z_visual": z_visual,
            "fused_z": self._fuse(z_text, z_visual),
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        z = self.analyze(x, edge_index)["fused_z"]
        return z, None, None, z.new_zeros(()), {}

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        self.eval()
        if edge_index is None:
            raise ValueError("cmrf_probe requires the physical edge_index")
        if device is None:
            device = next(self.parameters()).device
        z, _, _, _, _ = self.forward(x.to(device), edge_index.to(device))
        return z.detach().cpu()
