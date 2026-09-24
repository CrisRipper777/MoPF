from __future__ import annotations

from .ssi_mag_generic_common import SSIMAGGenericBase


class SSIMAGGenericPPR(SSIMAGGenericBase):
    """Generic PPR-style diffusion control, not an official APPNP replica."""

    def __init__(self, cfg, data_info: dict):
        super().__init__(cfg, data_info, control_mode="ppr_style")


Model = SSIMAGGenericPPR
