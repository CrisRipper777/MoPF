"""Launch the preregistered E6-SRIJ phases with the E5 cache adapted safely."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_utility_routing_audit as e5
from scripts.e6_srij_finalize import main


_move_cache = e5._move_cache


def _move_cache_with_global_teacher_rows(cache: dict, device: torch.device) -> dict:
    """E5 teacher CE is allowed-node indexed; expose it globally for the phase-A index map."""
    moved = _move_cache(cache, device)
    n = moved["states_text"][0].size(0)
    allowed = moved["allowed_idx"].long()
    for key in ("action_ce_text", "action_ce_visual"):
        values = moved[key]
        if values.size(0) != n:
            full = values.new_zeros((n, values.size(1)))
            full[allowed] = values
            moved[key] = full
    return moved


e5._move_cache = _move_cache_with_global_teacher_rows


if __name__ == "__main__":
    main()
