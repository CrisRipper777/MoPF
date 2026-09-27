"""E6-SRIJ launcher correcting E5's global-to-local cache row convention."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_e6_srij_experiment  # installs the E5 cache compatibility wrapper
from scripts import run_utility_routing_audit as e5
from scripts.e6_srij_finalize import main


_move_cache = e5._move_cache


def _move_cache_with_local_allowed_rows(cache: dict, device: torch.device) -> dict:
    moved = _move_cache(cache, device)
    # E5 cached states, labels, and teacher CE are already Train+Validation
    # rows, while allowed_idx stores their global node identifiers.
    moved["allowed_idx"] = torch.arange(
        moved["y_allowed"].numel(), device=device, dtype=torch.long
    )
    return moved


e5._move_cache = _move_cache_with_local_allowed_rows


if __name__ == "__main__":
    main()
