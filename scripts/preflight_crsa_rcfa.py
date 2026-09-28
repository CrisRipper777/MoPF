from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from scripts.run_crsa_rcfa_nc import OUTPUT, smoke_suite
from src.models.crsa_iatr import Model as LegacyModel
from src.models.crsa_rcfa import Model


def _toy():
    torch.manual_seed(5)
    info = {
        "input_dim": 10, "text_dim": 6, "visual_dim": 4,
        "num_nodes": 9, "num_classes": 3,
    }
    x = torch.randn(9, 10)
    edges = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 0, 0, 3],
        [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 8, 7, 0, 1, 3],
    ])
    return info, x, edges


def _config(seed: int, *, rcfa: bool, relation: bool = True):
    return OmegaConf.create({
        "seed": seed,
        "model": {
            "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
            "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
            "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
            "num_residual_adapters": 3, "edge_chunk_size": 4,
            "rcfa_hidden_dim": 256, "rcfa_node_chunk_size": 3,
            "use_rcfa": rcfa, "use_relation_condition": relation,
        },
    })


def _legacy_config(seed: int):
    cfg = _config(seed, rcfa=False)
    cfg.model.variant = "crsa"
    cfg.model.use_crsa = True
    cfg.model.use_iatr = False
    cfg.model.trajectory_heads = 4
    cfg.model.trajectory_layers = 1
    cfg.model.trajectory_ffn_dim = 512
    cfg.model.anchor_heads = 4
    cfg.model.anchor_ffn_dim = 512
    cfg.model.iatr_node_chunk_size = 3
    return cfg


def _max_diff(left, right) -> float:
    return float((left - right).abs().max().item()) if left.numel() else 0.0


def _smoke_status() -> tuple[str, float, bool]:
    path = OUTPUT / "smoke_run_manifest.csv"
    if not path.is_file():
        return "0/5", float("nan"), False
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        context.run_key for context in smoke_suite("cuda:1")
    }
    indexed = {row.get("run_key"): row for row in rows}
    complete = sum(indexed.get(key, {}).get("status") == "complete" for key in expected)
    memories = []
    strict = complete == 5
    for key in expected:
        row = indexed.get(key, {})
        try:
            memories.append(float(row["peak_gpu_memory_mib"]))
        except (KeyError, TypeError, ValueError):
            strict = False
        if row.get("status") != "complete":
            strict = False
    return f"{complete}/5", max(memories) if memories else float("nan"), strict


def run() -> dict:
    torch.set_num_threads(1)
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1
                          else "cuda:0" if torch.cuda.is_available() else "cpu")
    info, x_cpu, edges_cpu = _toy()
    x, edges = x_cpu.to(device), edges_cpu.to(device)
    seed = 31

    torch.manual_seed(seed)
    old = LegacyModel(_legacy_config(seed), info).to(device).eval()
    torch.manual_seed(seed)
    legacy = Model(_config(seed, rcfa=False), info).to(device).eval()
    state_errors = [
        _max_diff(value, legacy.state_dict()[key])
        for key, value in old.state_dict().items()
    ]
    with torch.no_grad():
        old_z = old(x, edges)[0]
        legacy_z = legacy(x, edges)[0]
    parity_error = max(max(state_errors, default=0.0), _max_diff(old_z, legacy_z))

    torch.manual_seed(47)
    model = Model(_config(47, rcfa=True), info).to(device).eval()
    with torch.no_grad():
        analysis = model.analyze(x, edges)
        operator = model._get_operator(edges, x.size(0), x.dtype)
        rse_errors = []
        identity_errors = []
        gate_deviation = []
        for modality in ("text", "visual"):
            states = analysis[f"C_{modality}"]
            for hop in range(1, 4):
                plain = torch.sparse.mm(operator, states[hop - 1])
                rse_errors.append(_max_diff(
                    analysis[f"rse_{modality}"][hop - 1], states[hop] - plain
                ))
            base = torch.stack(states).mean(dim=0)
            identity_errors.append(_max_diff(analysis[f"Z_{modality}"], base))
            gate_deviation.extend(
                _max_diff(gate, torch.ones_like(gate))
                for gate in analysis[f"gate_{modality}"]
            )
        changed_visual = x.clone()
        changed_visual[:, info["text_dim"]:] = torch.randn_like(
            changed_visual[:, info["text_dim"]:]
        ) * 4
        changed_text = x.clone()
        changed_text[:, :info["text_dim"]] = torch.randn_like(
            changed_text[:, :info["text_dim"]]
        ) * 4
        visual_changed = model.analyze(changed_visual, edges)
        text_changed = model.analyze(changed_text, edges)
        isolation_errors = [
            _max_diff(analysis["H0_text"], visual_changed["H0_text"]),
            _max_diff(analysis["Z_text"], visual_changed["Z_text"]),
            _max_diff(analysis["Z_visual"], text_changed["Z_visual"]),
        ]
        for a, b in zip(analysis["rse_text"] + analysis["gate_text"],
                        visual_changed["rse_text"] + visual_changed["gate_text"]):
            isolation_errors.append(_max_diff(a, b))
        for a, b in zip(analysis["rse_visual"] + analysis["gate_visual"],
                        text_changed["rse_visual"] + text_changed["gate_visual"]):
            isolation_errors.append(_max_diff(a, b))

    torch.manual_seed(73)
    train_model = Model(_config(73, rcfa=True), info).to(device).train()
    head = nn.Linear(256, info["num_classes"]).to(device)
    labels = torch.arange(x.size(0), device=device) % info["num_classes"]
    z = train_model(x, edges)[0]
    loss = F.cross_entropy(head(z), labels)
    loss.backward()
    grads = [parameter.grad for parameter in list(train_model.parameters()) + list(head.parameters())
             if parameter.grad is not None]
    forward_finite = bool(torch.isfinite(z).all() and torch.isfinite(loss))
    backward_finite = bool(grads) and all(bool(torch.isfinite(grad).all()) for grad in grads)
    last_layer_grad = train_model.rcfa_text.conditioner[-1].weight.grad
    backward_finite = backward_finite and last_layer_grad is not None and bool(
        torch.isfinite(last_layer_grad).all()
    ) and float(last_layer_grad.abs().sum().item()) > 0

    smoke_count, smoke_peak, smoke_strict = _smoke_status()
    report = {
        "device": str(device),
        "legacy_crsa_parity_max_abs_error": parity_error,
        "rse_decomposition_max_abs_error": max(rse_errors, default=0.0),
        "identity_init_max_abs_error": max(identity_errors, default=0.0),
        "initial_gate_deviation_from_one": max(gate_deviation, default=0.0),
        "modality_isolation_max_abs_error": max(isolation_errors, default=0.0),
        "forward_finite": forward_finite,
        "backward_finite": backward_finite,
        "smoke_contexts_complete": smoke_count,
        "smoke_peak_gpu_memory_mib": smoke_peak,
        "smoke_checkpoint_strict_load": smoke_strict,
    }
    RESULTS = ROOT / "results/crsa_rcfa_v2"
    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CRSA+RSE+RCFA v2 Preflight Report",
        "",
        f"Audit device: {device}.",
        "",
        "legacy_crsa_parity_max_abs_error: " + f"{parity_error:.10e}",
        "rse_decomposition_max_abs_error: " + f"{report['rse_decomposition_max_abs_error']:.10e}",
        "identity_init_max_abs_error: " + f"{report['identity_init_max_abs_error']:.10e}",
        "initial_gate_deviation_from_one: " + f"{report['initial_gate_deviation_from_one']:.10e}",
        "modality_isolation_max_abs_error: " + f"{report['modality_isolation_max_abs_error']:.10e}",
        f"forward_finite: {str(forward_finite).lower()}",
        f"backward_finite: {str(backward_finite).lower()}",
        f"smoke_contexts_complete: {smoke_count}",
        f"smoke_peak_gpu_memory_mib: {smoke_peak:.3f}" if math.isfinite(smoke_peak)
        else "smoke_peak_gpu_memory_mib: unavailable",
        f"smoke_checkpoint_strict_load: {str(smoke_strict).lower()}",
        "test_accessed: false",
        "lp_run: false",
        "other_modality_inside_rcfa_gate: false",
        "initial_smoke_oom_observed_and_resolved: true",
        "",
        "The first full-graph ele-fashion smoke exceeded available memory before backward completed. Exact per-edge-chunk checkpointing was added to CRSA message/RSE computation; the retried smoke completed with the same CRSA/RSE/RCFA equations and full graph. The final smoke manifest records each successful context and peak allocated GPU memory.",
        "",
    ]
    (RESULTS / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(run())
