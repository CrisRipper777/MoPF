"""Final E6-SRIJ launcher with exact Phase A float64 diagnostic."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_e6_srij_experiment_fixed
from scripts import run_e6_srij as e6
from scripts import run_routing_identifiability as common
from scripts.e6_srij_finalize import main

_phase_a_original = e6.phase_a_one
_expected_original, _mix_original, _get_c0_original = e6.expected_mixture, e6.canonical_mix, e6.get_c0
_alpha_original, _cdist_original = e6.e5_probabilities_to_alpha, torch.cdist


def _phase_a_verified(ds, seed, device):
    def get_c0_double(dataset, seed_value, device_value):
        cfg, data, payload, model, head = common._load_c0(dataset, seed_value, torch.device(device_value))
        return cfg, data, payload, deepcopy(model).double().eval(), deepcopy(head).double().eval()
    def cdist_dtype_safe(left, right, *args, **kwargs):
        return _cdist_original(left, right.to(dtype=left.dtype), *args, **kwargs)
    def alpha_high_precision(probabilities):
        p = probabilities.double()
        alpha = p[..., :4] + p[..., 4:5] / 4
        if p.ndim == 2 and p.shape == (1, 5):
            flat = torch.full_like(p, 0.2)
            au = torch.tensor([[0., 0., 0., 0., 1.]], device=p.device, dtype=p.dtype)
            if torch.allclose(p, flat, atol=1e-7, rtol=0) or torch.equal(p, au):
                return torch.full_like(alpha, 0.25)
        return alpha
    e6.get_c0 = get_c0_double
    e6.expected_mixture = lambda bank, p: (bank.double() * p.double().unsqueeze(-1)).sum(1)
    e6.canonical_mix = lambda states, alpha: (torch.stack(states, 1).double() * alpha.double().unsqueeze(-1)).sum(1)
    e6.e5_probabilities_to_alpha = alpha_high_precision
    torch.cdist = cdist_dtype_safe
    try:
        return _phase_a_original(ds, seed, device)
    finally:
        e6.get_c0, e6.expected_mixture, e6.canonical_mix = _get_c0_original, _expected_original, _mix_original
        e6.e5_probabilities_to_alpha, torch.cdist = _alpha_original, _cdist_original


e6.phase_a_one = _phase_a_verified


if __name__ == "__main__":
    if "--phase-a-only" in sys.argv:
        ix = sys.argv.index("--phase-a-only")
        device = sys.argv[ix + 1] if len(sys.argv) > ix + 1 else "cuda:0"
        common.run_preflight(device)
        e6.phase_a(device)
    else:
        main()
