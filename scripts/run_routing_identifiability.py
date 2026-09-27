from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_utility_routing_audit as e5
from src.models.canonical_routing_audit import (
    CanonicalRoutingAudit,
    canonical_mix,
    e5_probabilities_to_alpha,
)
from src.models.utility_routing_audit import (
    UtilityRouter,
    action_bank,
    expected_mixture,
    safe_probabilities,
    uniform_state_mix,
)

OUTPUT = ROOT / "outputs/routing_identifiability_v1"
RESULTS = ROOT / "results/routing_identifiability_v1"
SCREEN = ["Movies", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
VARIANTS = [
    "B0_frozen_direct_ce",
    "B1_flat_utility_kd",
    "B2_weighted_utility_kd",
]
ALPHA_U = (0.25, 0.25, 0.25, 0.25)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _scalar_metric(value) -> float:
    return float(value["mean"] if isinstance(value, dict) else value)


def _metric_pair(logits: torch.Tensor, y: torch.Tensor, labels: list[int]):
    pred = logits.argmax(-1)
    acc = float((pred == y).float().mean().item())
    f1 = float(f1_score(
        y.detach().cpu().numpy(), pred.detach().cpu().numpy(),
        labels=labels, average="macro", zero_division=0,
    ))
    return acc, f1, pred


def _states_from_analysis(analysis: dict):
    return analysis["S_text"], analysis["S_visual"]


def _allowed_split(data, device):
    """Return Train then Validation only; this helper never indexes test_idx."""
    train_global = data.train_idx.to(device=device, dtype=torch.long)
    val_global = data.val_idx.to(device=device, dtype=torch.long)
    allowed_global = torch.cat([train_global, val_global])
    y = data.y.to(device=device, dtype=torch.long)[allowed_global]
    train = torch.arange(train_global.numel(), device=device)
    val = torch.arange(train_global.numel(), allowed_global.numel(), device=device)
    if y.numel() != train.numel() + val.numel():
        raise RuntimeError("Train/Validation label alignment failed")
    return allowed_global, y, train, val


def freeze_c0(model, classifier):
    """Put the frozen C0 backbone and classifier in evaluation-only mode."""
    model.eval()
    classifier.eval()
    for module in (model, classifier):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return model, classifier


def _load_c0(dataset: str, seed: int, device: torch.device):
    cfg, data, payload, model, classifier = e5.load_c0(dataset, seed, device)
    freeze_c0(model, classifier)
    return cfg, data, payload, model, classifier


@torch.no_grad()
def _c0_preflight_one(dataset: str, seed: int, device_name: str) -> dict:
    device = torch.device(device_name)
    cfg, data, payload, c0, classifier = _load_c0(dataset, seed, device)
    allowed, y, train, val = _allowed_split(data, device)
    analysis = c0.analyze(data.x.to(device), data.edge_index.to(device))
    states_t, states_v = _states_from_analysis(analysis)
    zt, zv = uniform_state_mix(states_t), uniform_state_mix(states_v)
    fused = c0._fuse(zt, zv)
    embedding_error = float((fused[allowed] - analysis["fused_z"][allowed]).abs().max().item())
    prob = torch.softmax(classifier(fused[allowed]), dim=-1)
    stored_prob = torch.softmax(classifier(analysis["fused_z"][allowed]), dim=-1)
    probability_error = float((prob - stored_prob).abs().max().item())
    labels = sorted(int(x) for x in torch.unique(y).detach().cpu().tolist())
    val_acc, val_f1, _ = _metric_pair(classifier(fused[allowed[val]]), y[val], labels)
    historical = e5.c0_metrics(dataset, seed)
    old = json.loads(historical.read_text(encoding="utf-8")) if historical.is_file() else {}
    metrics = old.get("metrics", {})
    stored_acc = _scalar_metric(metrics["val_acc"]) if "val_acc" in metrics else float(payload["metrics"]["val_acc"]["mean"])
    stored_f1 = _scalar_metric(metrics["val_macro_f1"]) if "val_macro_f1" in metrics else float(payload["metrics"]["val_macro_f1"]["mean"])
    row = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint": str(e5.c0_checkpoint(dataset, seed).relative_to(ROOT)),
        "checkpoint_sha256": _sha256(e5.c0_checkpoint(dataset, seed)),
        "embedding_max_abs_error": embedding_error,
        "probability_max_abs_error": probability_error,
        "stored_val_acc": stored_acc,
        "reloaded_val_acc": val_acc,
        "val_acc_abs_delta": abs(stored_acc - val_acc),
        "stored_val_macro_f1": stored_f1,
        "reloaded_val_macro_f1": val_f1,
        "val_macro_f1_abs_delta": abs(stored_f1 - val_f1),
        "evaluate_test": False,
        "test_labels_indexed": False,
    }
    if embedding_error >= 1e-6 or probability_error >= 1e-6:
        raise RuntimeError(f"C0 reload forward mismatch: {dataset}/seed{seed}: {row}")
    if abs(stored_acc - val_acc) >= 1e-6 or abs(stored_f1 - val_f1) >= 1e-6:
        raise RuntimeError(f"C0 stored validation metric mismatch: {dataset}/seed{seed}: {row}")
    return row


def run_preflight(device: str) -> list[dict]:
    rows = [_c0_preflight_one(ds, seed, device) for ds in SCREEN for seed in SEEDS]
    _write_csv(RESULTS / "preflight_c0_checks.csv", rows)
    lines = [
        "# E6-SRIJ preflight",
        "",
        "C0 was reloaded from `outputs/cmrf_discovery_v1/reference_uniform/` and evaluated on Train/Validation only.",
        "Text/visual projectors, physical propagation, plain fusion, and classifier were frozen. No C0 training was performed.",
        "",
        f"Validated checkpoints: {len(rows)}/9.",
        f"Maximum embedding error: {max(row['embedding_max_abs_error'] for row in rows):.8g}.",
        f"Maximum probability error: {max(row['probability_max_abs_error'] for row in rows):.8g}.",
        f"Maximum Val Acc delta: {max(row['val_acc_abs_delta'] for row in rows):.8g}.",
        f"Maximum Val Macro-F1 delta: {max(row['val_macro_f1_abs_delta'] for row in rows):.8g}.",
        "",
        "No Test labels were indexed. Toys/Grocery were not loaded.",
    ]
    (RESULTS / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
