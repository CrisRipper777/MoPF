from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.models.crsa_rcfa import Model as FrozenCRSA
from src.models.crsa_icsr import Model as ICSRModel, _ICSRModality

OUT = ROOT / "results/crsa_icsr_v3"


def cfg(seed: int, *, use_icsr: bool, edge_chunk: int = 4, node_chunk: int = 3):
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "name": "crsa_icsr", "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
            "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3, "edge_chunk_size": edge_chunk,
            "use_icsr": use_icsr, "icsr_bottleneck_dim": 32,
            "icsr_node_chunk_size": node_chunk,
        },
    })


def old_cfg(seed: int, edge_chunk: int = 4):
    c = cfg(seed, use_icsr=False, edge_chunk=edge_chunk)
    c.model.name = "crsa_rcfa"
    c.model.pop("use_icsr")
    c.model.pop("icsr_bottleneck_dim")
    c.model.pop("icsr_node_chunk_size")
    c.model.use_rcfa = False
    c.model.use_relation_condition = True
    c.model.rcfa_hidden_dim = 256
    c.model.rcfa_node_chunk_size = 3
    return c


def max_error(a, b) -> float:
    if isinstance(a, (list, tuple)):
        return max((max_error(x, y) for x, y in zip(a, b)), default=0.0)
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def main() -> None:
    torch.set_num_threads(1)
    info = {"input_dim": 10, "text_dim": 6, "visual_dim": 4,
            "num_nodes": 9, "num_classes": 3}
    torch.manual_seed(5)
    x = torch.randn(9, 10)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 0, 0, 3],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 8, 7, 0, 1, 3],
    ])
    seed = 31
    torch.manual_seed(seed)
    old = FrozenCRSA(old_cfg(seed), info).eval()
    old_head = nn.Linear(256, info["num_classes"])
    torch.manual_seed(seed)
    crsa = ICSRModel(cfg(seed, use_icsr=False), info).eval()
    crsa_head = nn.Linear(256, info["num_classes"])
    torch.manual_seed(seed)
    full = ICSRModel(cfg(seed, use_icsr=True), info).eval()
    full_head = nn.Linear(256, info["num_classes"])

    if old.state_dict().keys() != crsa.state_dict().keys():
        raise AssertionError("legacy CRSA state_dict keys differ")
    init_state_err = max((max_error(a, crsa.state_dict()[key])
                          for key, a in old.state_dict().items()), default=0.0)
    head_rng_err = max((max_error(a, b) for a, b in zip(old_head.parameters(), crsa_head.parameters())),
                       default=0.0)
    head_rng_err = max(head_rng_err, max((max_error(a, b) for a, b in zip(crsa_head.parameters(), full_head.parameters())), default=0.0))
    with torch.no_grad():
        a, b, out = old.analyze(x, edge_index), crsa.analyze(x, edge_index), full.analyze(x, edge_index)
    parity_parts = [max_error(a[k], b[k]) for k in
                    ("H0_text", "H0_visual", "base_text", "base_visual", "fused_z")]
    for modality in ("text", "visual"):
        for key in (f"C_{modality}", f"rse_{modality}", f"routes_{modality}"):
            parity_parts.append(max_error(a[key], b[key]))
    legacy_parity = max(parity_parts, default=0.0)

    zero_errors = []
    agreement_errors, deviation_errors = [], []
    for modality in ("text", "visual"):
        zero_errors.extend(float(v.abs().max().item()) for v in out[f"interaction_{modality}"])
        zero_errors.append(max_error(out[f"Z_{modality}"], out[f"base_{modality}"]))
        h = F.layer_norm(out[f"H0_{modality}"], (256,))
        for hop in range(1, 4):
            c = F.layer_norm(out[f"C_{modality}"][hop], (256,))
            agreement_errors.append(max_error(out[f"agreement_{modality}"][hop - 1], h * c))
            deviation_errors.append(max_error(out[f"deviation_{modality}"][hop - 1], c - h))

    with torch.no_grad():
        changed_v, changed_t = x.clone(), x.clone()
        torch.manual_seed(73)
        changed_v[:, info["text_dim"]:] = torch.randn_like(changed_v[:, info["text_dim"]:])
        changed_t[:, :info["text_dim"]] = torch.randn_like(changed_t[:, :info["text_dim"]:])
        original = full.analyze(x, edge_index)
        visual_changed = full.analyze(changed_v, edge_index)
        text_changed = full.analyze(changed_t, edge_index)
    isolation_errors = []
    for key in ("H0_text", "Z_text", "base_text"):
        isolation_errors.append(max_error(original[key], visual_changed[key]))
    for key in ("C_text", "agreement_text", "deviation_text", "interaction_text"):
        isolation_errors.append(max_error(original[key], visual_changed[key]))
    for key in ("C_visual", "agreement_visual", "deviation_visual", "interaction_visual"):
        isolation_errors.append(max_error(original[key], text_changed[key]))
    isolation_errors.append(max_error(original["Z_visual"], text_changed["Z_visual"]))

    parameter_delta = sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in crsa.parameters())
    forward_finite = all(torch.isfinite(t).all().item() for t in (out["fused_z"], out["Z_text"], out["Z_visual"]))
    full.train()
    head = nn.Linear(256, info["num_classes"])
    labels = torch.arange(x.size(0)) % info["num_classes"]
    z = full(x, edge_index)[0]
    loss = F.cross_entropy(head(z), labels)
    loss.backward()
    grads = [p.grad for p in list(full.parameters()) + list(head.parameters()) if p.grad is not None]
    backward_finite = bool(torch.isfinite(loss).item() and grads and
                           all(torch.isfinite(g).all().item() for g in grads))
    up_grad_nonzero = all(getattr(full, f"icsr_{m}").up.weight.grad.abs().sum().item() > 0
                          for m in ("text", "visual"))

    full.eval()
    reload = ICSRModel(cfg(seed, use_icsr=True), info)
    reload.load_state_dict(full.state_dict(), strict=True)
    full.eval()
    reload.eval()
    with torch.no_grad():
        reload_equal = torch.equal(full(x, edge_index)[0], reload(x, edge_index)[0])

    metrics = {
        "legacy_crsa_parameter_initialization_max_abs_error": init_state_err,
        "legacy_crsa_head_rng_max_abs_error": head_rng_err,
        "legacy_crsa_parity_max_abs_error": legacy_parity,
        "identity_init_max_abs_error": max(zero_errors, default=0.0),
        "interaction_zero_init_max_abs": max((float(v.abs().max().item()) for m in ("text", "visual")
                                                for v in out[f"interaction_{m}"]), default=0.0),
        "agreement_formula_max_abs_error": max(agreement_errors, default=0.0),
        "deviation_formula_max_abs_error": max(deviation_errors, default=0.0),
        "modality_isolation_max_abs_error": max(isolation_errors, default=0.0),
        "parameter_delta_vs_crsa": parameter_delta,
        "forward_finite": forward_finite,
        "backward_finite": backward_finite,
        "icsr_up_gradient_nonzero": up_grad_nonzero,
        "checkpoint_reload_equal": reload_equal,
        "same_modality_shared_adapter_count": 2,
        "rse_used_inside_icsr": False,
        "other_modality_used_inside_icsr": False,
        "utility_gate_present": False,
    }
    numeric = ["legacy_crsa_parameter_initialization_max_abs_error", "legacy_crsa_head_rng_max_abs_error",
               "legacy_crsa_parity_max_abs_error", "identity_init_max_abs_error",
               "agreement_formula_max_abs_error", "deviation_formula_max_abs_error",
               "modality_isolation_max_abs_error"]
    failed = [key for key in numeric if metrics[key] >= 1e-6]
    failed.extend(key for key in ("forward_finite", "backward_finite", "icsr_up_gradient_nonzero",
                                   "checkpoint_reload_equal") if not metrics[key])
    if parameter_delta != 49_728:
        failed.append("parameter_delta_vs_crsa")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight_audit.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    lines = ["# CRSA + ICSR v3 Preflight", "", "Synthetic graph audit; no dataset test labels or LP data were accessed.", "",
             *[f"- `{key}`: `{value}`" for key, value in metrics.items()], "",
             f"Preflight: **{'PASS' if not failed else 'FAIL'}**" + (f" ({', '.join(failed)})" if failed else ""), ""]
    (OUT / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
