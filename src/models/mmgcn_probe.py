from __future__ import annotations

import torch
import torch.nn as nn

from .mmgcn import Model as MMGCN


ID_MODES = frozenset({"reference_fixed", "zero_fixed", "trainable"})


class Model(MMGCN):
    """MMGCN diagnostic that changes only the repeated node-ID residual."""

    def __init__(self, cfg, data_info: dict):
        mode = str(cfg.model.get("id_mode", "reference_fixed")).strip().lower()
        if mode not in ID_MODES:
            raise ValueError(f"id_mode must be one of {sorted(ID_MODES)}, got {mode!r}")
        super().__init__(cfg, data_info)
        self.id_mode = mode
        if mode == "zero_fixed":
            with torch.no_grad():
                self.id_embedding.zero_()
        elif mode == "trainable":
            value = self.id_embedding.detach().clone()
            del self.id_embedding
            self.register_parameter("id_embedding", nn.Parameter(value))

    def _apply(self, fn):
        if getattr(self, "id_mode", None) == "trainable":
            return nn.Module._apply(self, fn)
        return super()._apply(fn)
