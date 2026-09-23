"""Controlled prior-initialization diagnostic for canonical MGSC-MAG P2.

The default ``legacy_anchored`` mode is byte-for-byte equivalent in parameter
initialization to MGSC-MAG. ``direct`` only replaces the initialization of
``gamma_global``; no forward path or task protocol is changed.
"""

from __future__ import annotations

import torch

from .mgsc_mag import MGSCMAG


class MGSCMAGPriorInitControl(MGSCMAG):
    """MGSC-MAG with an explicit, inference/training-controlled prior basis."""

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info)
        self.prior_init_mode = str(cfg.model.get("prior_init_mode", "legacy_anchored")).lower()
        if self.prior_init_mode not in {"legacy_anchored", "direct"}:
            raise ValueError("prior_init_mode must be legacy_anchored or direct")
        if self.prior_init_mode == "direct":
            with torch.no_grad():
                self.gamma_global.copy_(self._make_global_prior().to(self.gamma_global))


Model = MGSCMAGPriorInitControl

