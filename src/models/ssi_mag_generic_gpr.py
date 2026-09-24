from __future__ import annotations

from .ssi_mag_generic_common import SSIMAGGenericBase


class SSIMAGGenericGPR(SSIMAGGenericBase):
    """Generic GPR-style multi-scale diffusion control, not an official GPR-GNN replica."""

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info, control_mode="gpr_style")


Model = SSIMAGGenericGPR
