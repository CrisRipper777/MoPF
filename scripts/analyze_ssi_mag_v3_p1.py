#!/usr/bin/env python3
"""Post-hoc mechanism and frozen-intervention analysis for P1 SSI-MAG-V3 NC.

This script never calls the trainer.  It loads the saved NC model and head from
each completed run, reuses the resolved config/data split, and evaluates the
model's no-grad analysis/intervention API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
MODALITIES = ("text", "visual")
EPS = 1.0e-12


def scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def flat(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().float().cpu().reshape(-1)


def has_nonfinite(value: Any) -> bool:
    if torch.is_tensor(value):
        return not bool(torch.isfinite(value).all())
    if isinstance(value, list):
        return any(has_nonfinite(item) for item in value)
    if isinstance(value, dict):
        return any(has_nonfinite(item) for item in value.values())
    return False


def stats(value: Any, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)) -> dict[str, float]:
    x = flat(value)
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                **{f"q{int(q * 100):02d}": 0.0 for q in quantiles}}
    if x.numel() > 1_000_000:
        stride = int(math.ceil(x.numel() / 1_000_000))
        qx = x[::stride]
    else:
        qx = x
    q = torch.quantile(qx, torch.tensor(quantiles, dtype=qx.dtype))
    out = {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
    }
    out.update({f"q{int(p * 100):02d}": float(v) for p, v in zip(quantiles, q)})
    return out


def put_stats(row: dict[str, Any], prefix: str, value: Any, *, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)) -> None:
    for key, value in stats(value, quantiles).items():
        row[f"{prefix}_{key}"] = value


def put_hops(row: dict[str, Any], prefix: str, values: list[Any], start: int = 0, *, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90)) -> None:
    for offset, value in enumerate(values):
        put_stats(row, f"{prefix}_k{start + offset}", value, quantiles=quantiles)


def eval_metrics(logits: torch.Tensor, data: Any, idx: torch.Tensor, labels: list[int]) -> dict[str, float]:
    pred = logits[idx.to(logits.device)].argmax(dim=-1).detach().cpu()
    target = data.y[idx].detach().cpu()
    return {
        "acc": float((pred == target).float().mean()),
        "macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=labels, average="macro", zero_division=0)),
    }


def edge_occurrences(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> dict[int, list[float]]:
    src, dst = edge_index.detach().cpu().long()
    weights = edge_weight.detach().cpu().float()
    result: dict[int, list[float]] = defaultdict(list)
    for u, v, w in zip(src.tolist(), dst.tolist(), weights.tolist()):
        result[int(u) * num_nodes + int(v)].append(float(w))
    return result


def operator_perturbation(model: Any, physical_edge_index: torch.Tensor, full_index: torch.Tensor, full_weight: torch.Tensor, num_nodes: int, dtype: torch.dtype) -> dict[str, float]:
    with torch.no_grad():
        raw_index, raw_weight = model._normalized_operator(
            physical_edge_index,
            torch.ones(physical_edge_index.size(1), device=physical_edge_index.device, dtype=dtype),
            num_nodes,
            dtype,
        )
    full_map = edge_occurrences(full_index, full_weight, num_nodes)
    raw_map = edge_occurrences(raw_index, raw_weight, num_nodes)
    diffs: list[float] = []
    raw_values: list[float] = []
    missing_full = 0
    missing_raw = 0
    for key in set(full_map) | set(raw_map):
        fvals = full_map.get(key, [])
        rvals = raw_map.get(key, [])
        common = min(len(fvals), len(rvals))
        diffs.extend(fvals[i] - rvals[i] for i in range(common))
        raw_values.extend(rvals[:common])
        missing_full += max(0, len(rvals) - common)
        missing_raw += max(0, len(fvals) - common)
    d = torch.tensor(diffs, dtype=torch.float64)
    r = torch.tensor(raw_values, dtype=torch.float64)
    if d.numel() == 0:
        return {"operator_weight_mae": math.nan, "operator_weight_rmse": math.nan,
                "operator_weight_relative_l1": math.nan, "operator_aligned_edges": 0.0,
                "operator_missing_full": float(missing_full), "operator_missing_raw": float(missing_raw)}
    return {
        "operator_weight_mae": float(d.abs().mean()),
        "operator_weight_rmse": float(d.square().mean().sqrt()),
        "operator_weight_relative_l1": float(d.abs().sum() / (r.abs().sum() + EPS)),
        "operator_aligned_edges": float(d.numel()),
        "operator_missing_full": float(missing_full),
        "operator_missing_raw": float(missing_raw),
    }


def attention_stats(attention: torch.Tensor) -> dict[str, float]:
    a = attention.detach().float().cpu()
    if a.numel() == 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0, "diagonal_mass": 0.0,
                "off_diagonal_mass": 0.0, "row_diversity_l1": 0.0, "row_diversity_l2": 0.0}
    h = a.size(-1)
    entropy = -(a.clamp_min(EPS) * a.clamp_min(EPS).log()).sum(dim=-1).mean(dim=-1)
    diagonal = a.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / float(h)
    l1: list[torch.Tensor] = []
    l2: list[torch.Tensor] = []
    for i in range(h):
        for j in range(i + 1, h):
            l1.append((a[:, i, :] - a[:, j, :]).abs().sum(dim=-1))
            l2.append((a[:, i, :] - a[:, j, :]).square().sum(dim=-1).sqrt())
    row_l1 = torch.stack(l1, dim=-1).mean(dim=-1) if l1 else torch.zeros(a.size(0))
    row_l2 = torch.stack(l2, dim=-1).mean(dim=-1) if l2 else torch.zeros(a.size(0))
    return {
        "entropy": float(entropy.mean()),
        "normalized_entropy": float(entropy.mean() / max(math.log(h), EPS)),
        "diagonal_mass": float(diagonal.mean()),
        "off_diagonal_mass": float((1.0 - diagonal).mean()),
        "row_diversity_l1": float(row_l1.mean()),
        "row_diversity_l2": float(row_l2.mean()),
    }


def compare_embeddings(normal: dict[str, Any], changed: dict[str, Any], row: dict[str, Any], tag: str) -> None:
    for name in ("z", "z_text", "z_visual"):
        base = normal[name].detach().float()
        other = changed[name].detach().float()
        diff = other - base
        row[f"{tag}_{name}_mae"] = float(diff.abs().mean())
        row[f"{tag}_{name}_relative_l2"] = float(diff.norm() / base.norm().clamp_min(EPS))
        row[f"{tag}_{name}_cosine"] = float(F.cosine_similarity(base, other, dim=-1).mean())


def compare_intervention(normal: dict[str, Any], changed: dict[str, Any], classifier: nn.Module, data: Any, labels: list[int], row: dict[str, Any], tag: str) -> None:
    compare_embeddings(normal, changed, row, tag)
    normal_logits = classifier(normal["z"])
    changed_logits = classifier(changed["z"])
    normal_pred = normal_logits.argmax(dim=-1)
    changed_pred = changed_logits.argmax(dim=-1)
    row[f"{tag}_prediction_flip_rate"] = float((normal_pred != changed_pred).float().mean())
    for split, idx in (("val", data.val_idx), ("test", data.test_idx)):
        before = eval_metrics(normal_logits, data, idx, labels)
        after = eval_metrics(changed_logits, data, idx, labels)
        row[f"{tag}_{split}_acc_delta"] = after["acc"] - before["acc"]
        row[f"{tag}_{split}_macro_f1_delta"] = after["macro_f1"] - before["macro_f1"]


def add_flags(row: dict[str, Any], flags: list[str]) -> None:
    row["flag_count"] = len(flags)
    row["flags"] = ";".join(flags)


def analyze_run(run_dir: Path, device: torch.device) -> dict[str, Any]:
    cfg = OmegaConf.create(json.loads((run_dir / "resolved_config.json").read_text()))
    seed = int(cfg.seed)
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(cfg, data_info).to(device)
    model.load_state_dict(payload["model_state"])
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"])
    model.eval()
    classifier.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = sorted({int(v) for idx in (data.train_idx, data.val_idx, data.test_idx) for v in data.y[idx].tolist() if 0 <= int(v) < int(data.num_classes)})
    with torch.no_grad():
        normal = model.analysis_intervention(x, edge_index)
        relation_off = model.analysis_intervention(x, edge_index, relation="off")
        interaction_off = model.analysis_intervention(x, edge_index, interaction="off")
    row: dict[str, Any] = {"dataset": str(cfg.dataset.name), "seed": seed, "ablation": str(cfg.ablation), "model": str(cfg.model.name), "checkpoint": str(run_dir / "best.pt")}
    results_path = run_dir / "results.json"
    if results_path.is_file():
        result = json.loads(results_path.read_text())
        for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
            row[key] = float(result.get(key, {}).get("mean", math.nan))
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    row["best_epoch"] = int(payload.get("epoch")) if payload.get("epoch") is not None else math.nan
    row["runtime_seconds"] = float(metrics.get("runtime_seconds", math.nan))
    row["peak_gpu_memory_mib"] = float(metrics.get("peak_gpu_memory_mib", math.nan) or math.nan)
    log_text = (run_dir / "train.log").read_text(errors="ignore") if (run_dir / "train.log").is_file() else ""
    stopped = re.findall(r"Early stopping at epoch (\d+)", log_text)
    row["total_trained_epochs"] = int(stopped[-1]) if stopped else len(re.findall(r"Epoch \d+", log_text))
    train_loss_matches = re.findall(r"Epoch\s+(\d+)\s+\|\s+Train Loss\s+([0-9.eE+-]+)", log_text)
    train_losses = {int(epoch): float(loss) for epoch, loss in train_loss_matches}
    row["train_loss_at_best"] = train_losses.get(int(row["best_epoch"]), math.nan) if math.isfinite(float(row["best_epoch"])) else math.nan
    row["train_loss_last"] = train_losses[max(train_losses)] if train_losses else math.nan
    row["val_loss_logged"] = 0
    row["nan_inf_detected"] = int(has_nonfinite(normal) or has_nonfinite(relation_off) or has_nonfinite(interaction_off))
    flags: list[str] = []
    if row["nan_inf_detected"]:
        flags.append("numerical_nonfinite_detected")
    for modality in MODALITIES:
        beta = scalar(normal[f"beta_parameter_{modality}"])
        row[f"beta_{modality}"] = beta
        row[f"beta_parameter_{modality}"] = beta
        residual = normal[f"relation_residual_{modality}"]
        weights = normal[f"relation_weight_{modality}"]
        put_stats(row, f"relation_residual_{modality}", residual, quantiles=(0.10, 0.50, 0.90))
        put_stats(row, f"relation_weight_{modality}", weights, quantiles=(0.10, 0.50, 0.90))
        weight_stats = stats(weights, quantiles=(0.10, 0.50, 0.90))
        row[f"relation_weight_{modality}_cv"] = weight_stats["std"] / max(abs(weight_stats["mean"]), EPS)
        put_stats(row, f"local_adaptation_{modality}", normal[f"local_adaptation_{modality}"])
        put_stats(row, f"p_{modality}", normal[f"p_{modality}"], quantiles=(0.10, 0.50, 0.90))
        put_hops(row, f"d_{modality}", normal[f"d_{modality}"], start=1, quantiles=(0.10, 0.50, 0.90))
        put_hops(row, f"alpha_{modality}", normal[f"alpha_{modality}"], start=1)
        put_hops(row, f"Q_{modality}", normal[f"Q_{modality}"], start=1, quantiles=(0.10, 0.50, 0.90))
        put_hops(row, f"S_{modality}", normal[f"S_{modality}"], start=0, quantiles=(0.10, 0.50, 0.90))
        put_hops(row, f"D_norm_{modality}", normal[f"d_norm_{modality}"], start=0, quantiles=(0.10, 0.50, 0.90))
        delta_content = normal[f"delta_content_{modality}"]
        put_hops(row, f"delta_content_{modality}", [delta_content[:, k] for k in range(delta_content.size(1))], start=0, quantiles=(0.10, 0.50, 0.90))
        reference_residual = normal[f"reference_residual_{modality}"]
        put_hops(row, f"reference_residual_{modality}", [reference_residual[:, k] for k in range(reference_residual.size(1))], start=0, quantiles=(0.10, 0.50, 0.90))
        relation_residual = normal[f"relation_filter_residual_{modality}"]
        put_hops(row, f"relation_filter_residual_{modality}", [relation_residual[:, k] for k in range(relation_residual.size(1))], start=0, quantiles=(0.10, 0.50, 0.90))
        eta = normal[f"eta_{modality}"]
        put_hops(row, f"eta_{modality}", [eta[:, k] for k in range(eta.size(1))], start=0, quantiles=(0.10, 0.50, 0.90))
        for k in range(eta.size(1)):
            row[f"eta_{modality}_k{k}_negative_fraction"] = float((eta[:, k] < 0).float().mean())
        put_stats(row, f"effective_order_{modality}", normal[f"effective_order_{modality}"])
        put_stats(row, f"effective_radius_{modality}", normal[f"effective_radius_{modality}"])
        row[f"delta_gate_{modality}"] = scalar(normal[f"delta_gate_{modality}"])
        row[f"interaction_gate_{modality}"] = scalar(normal[f"interaction_gate_{modality}"])
        row[f"gamma_{modality}"] = json.dumps([scalar(v) for v in normal[f"gamma_{modality}"]])
        row[f"delta_gamma_{modality}"] = json.dumps([scalar(v) for v in normal[f"delta_gamma_{modality}"]])
        semantic_bias = getattr(model, f"semantic_bias_{modality}").detach().cpu()
        row[f"semantic_bias_{modality}"] = json.dumps([scalar(v) for v in semantic_bias])
        for k, value in enumerate(semantic_bias):
            row[f"semantic_bias_{modality}_k{k + 1}"] = scalar(value)
        for name in ("semantic_rho_p", "semantic_rho_d", "semantic_rho_c"):
            row[f"{name}_{modality}"] = scalar(getattr(model, f"{name}_{modality}"))
        for k, value in enumerate(normal[f"gamma_{modality}"]):
            row[f"gamma_{modality}_k{k}"] = scalar(value)
        for k, value in enumerate(normal[f"delta_gamma_{modality}"]):
            row[f"delta_gamma_{modality}_k{k}"] = scalar(value)
        a = torch.cat([v.reshape(-1) for v in normal[f"alpha_{modality}"]])
        row[f"alpha_{modality}_fraction_lt_0.05"] = float((a < 0.05).float().mean())
        row[f"alpha_{modality}_fraction_gt_0.95"] = float((a > 0.95).float().mean())
        attention = normal[f"attention_{modality}"]
        for key, value in attention_stats(attention).items():
            row[f"attention_{modality}_{key}"] = value
        op = operator_perturbation(model, edge_index, normal[f"normalized_edge_index_{modality}"], normal[f"normalized_edge_weight_{modality}"], data.num_nodes, x.dtype)
        row.update({f"{key}_{modality}": value for key, value in op.items()})
        if abs(beta - 0.05) <= 0.005:
            flags.append(f"R1_{modality}_beta_near_init")
        if row[f"relation_weight_{modality}_cv"] < 1.0e-3:
            flags.append(f"R1_{modality}_weight_cv_negligible")
        if row[f"operator_weight_relative_l1_{modality}"] < 1.0e-3:
            flags.append(f"R1_{modality}_operator_perturbation_negligible")
        alpha_means = [row[f"alpha_{modality}_k{k}_mean"] for k in range(1, len(normal[f"alpha_{modality}"]) + 1)]
        alpha_stds = [row[f"alpha_{modality}_k{k}_std"] for k in range(1, len(normal[f"alpha_{modality}"]) + 1)]
        if max(alpha_stds) < 0.005 and max(abs(value - 0.1) for value in alpha_means) < 0.005:
            flags.append(f"R2_{modality}_alpha_near_fixed")
        if max(row[f"alpha_{modality}_fraction_lt_0.05"], row[f"alpha_{modality}_fraction_gt_0.95"]) > 0.5:
            flags.append(f"R2_{modality}_alpha_saturation")
        if abs(row[f"delta_gate_{modality}"]) < 0.01:
            flags.append(f"R3_{modality}_delta_gate_near_zero")
        if abs(row[f"interaction_gate_{modality}"]) < 0.01:
            flags.append(f"R3_{modality}_interaction_gate_near_zero")
        if row[f"attention_{modality}_normalized_entropy"] > 0.99:
            flags.append(f"R3_{modality}_attention_near_uniform")
        if row[f"attention_{modality}_diagonal_mass"] > 0.95:
            flags.append(f"R3_{modality}_attention_near_diagonal")
        if row[f"effective_order_{modality}_std"] < 1.0e-3:
            flags.append(f"R3_{modality}_effective_order_node_variance_negligible")
        if max(row[f"eta_{modality}_k{k}_std"] for k in range(len(normal[f"eta_{modality}"][0]))) < 1.0e-4:
            flags.append(f"R3_{modality}_eta_node_variance_negligible")
    compare_intervention(normal, relation_off, classifier, data, labels, row, "relation_off")
    compare_intervention(normal, interaction_off, classifier, data, labels, row, "interaction_off")
    add_flags(row, flags)
    del normal, relation_off, interaction_off, model, classifier, x, edge_index, data, payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    out: list[dict[str, Any]] = []
    for group, items in groups.items():
        result: dict[str, Any] = {group_key: group, "n_runs": len(items)}
        keys = sorted({key for item in items for key, value in item.items() if key not in {group_key, "dataset", "checkpoint", "flags", "model", "ablation", "gamma_text", "gamma_visual", "delta_gamma_text", "delta_gamma_visual"} and isinstance(value, (int, float)) and math.isfinite(float(value))})
        for key in keys:
            values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))]
            if values:
                t = torch.tensor(values, dtype=torch.float64)
                result[f"{key}_mean"] = float(t.mean())
                result[f"{key}_std"] = float(t.std(unbiased=False))
        out.append(result)
    return out


def performance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1", "best_epoch", "total_trained_epochs", "train_loss_at_best", "train_loss_last", "runtime_seconds", "peak_gpu_memory_mib")
    out: list[dict[str, Any]] = []
    for group, items in [(d, [r for r in rows if r["dataset"] == d]) for d in DATASETS] + [("ALL", rows)]:
        result = {"dataset": group, "n_runs": len(items)}
        for key in keys:
            vals = [float(r[key]) for r in items if key in r and math.isfinite(float(r[key]))]
            if vals:
                t = torch.tensor(vals, dtype=torch.float64)
                result[f"{key}_mean"] = float(t.mean())
                result[f"{key}_std"] = float(t.std(unbiased=False))
        out.append(result)
    return out


def fmt(value: Any) -> str:
    return "NA" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{float(value):.4f}"


HISTORICAL_ROOTS = {
    "CoSI-MAG Final": ROOT / "outputs/cosi_mag_final_benchmark/nc",
    "All Plain": ROOT / "outputs/cosi_mag_final_joint_ablation/all_plain/nc",
}


def historical_comparison(perf: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    verified: dict[str, dict[str, dict[str, float]]] = {}
    status: dict[str, str] = {}
    metrics = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    for method, root in HISTORICAL_ROOTS.items():
        method_data: dict[str, dict[str, float]] = {}
        ok = True
        for dataset in DATASETS:
            run_dir = root / dataset / "runs_42_43_44"
            result_path = run_dir / "results.json"
            config_path = run_dir / "resolved_config.json"
            manifest_path = run_dir / ("benchmark_manifest.json" if method == "CoSI-MAG Final" else "joint_ablation_manifest.json")
            if not (result_path.is_file() and config_path.is_file() and manifest_path.is_file()):
                ok = False
                break
            config = json.loads(config_path.read_text())
            manifest = json.loads(manifest_path.read_text())
            seeds = manifest.get("run_seeds")
            if (config.get("task", {}).get("name") != "nc"
                    or config.get("task", {}).get("protocol_version") != "unified_full_graph_nc_v1"
                    or config.get("task", {}).get("training_mode") != "full_graph"
                    or int(config.get("num_runs", 0)) != 3
                    or seeds != SEEDS):
                ok = False
                break
            result = json.loads(result_path.read_text())
            if not all(key in result and "mean" in result[key] and "std" in result[key] for key in metrics):
                ok = False
                break
            method_data[dataset] = {f"{key}_mean": float(result[key]["mean"]) for key in metrics}
            method_data[dataset].update({f"{key}_std": float(result[key]["std"]) for key in metrics})
        if ok:
            verified[method] = method_data
            status[method] = "verified: matching NC protocol, full-graph mode, seeds 42/43/44, validation-Accuracy selection"
        else:
            status[method] = "unavailable: provenance checks failed"
    v3 = {item["dataset"]: item for item in perf}
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS + ["ALL"]:
        methods: dict[str, dict[str, float]] = {}
        current = v3.get(dataset)
        if current:
            methods["SSI-MAG-V3 Full"] = {f"{key}_mean": float(current.get(f"{key}_mean", math.nan)) for key in metrics}
            methods["SSI-MAG-V3 Full"].update({f"{key}_std": float(current.get(f"{key}_std", math.nan)) for key in metrics})
        for method, data in verified.items():
            if dataset == "ALL":
                values = [data[d] for d in DATASETS]
                methods[method] = {}
                for key in metrics:
                    means = [item[f"{key}_mean"] for item in values]
                    methods[method][f"{key}_mean"] = sum(means) / len(means)
                    methods[method][f"{key}_std"] = float(torch.tensor(means, dtype=torch.float64).std(unbiased=False))
            else:
                methods[method] = data[dataset]
        v3_values = methods.get("SSI-MAG-V3 Full", {})
        for method, values in methods.items():
            row: dict[str, Any] = {"dataset": dataset, "method": method, "source_status": status.get(method, "current P1")}
            for key in metrics:
                row[f"{key}_mean"] = values.get(f"{key}_mean", math.nan)
                row[f"{key}_std"] = values.get(f"{key}_std", math.nan)
                if method != "SSI-MAG-V3 Full":
                    row[f"delta_vs_v3_{key}"] = values.get(f"{key}_mean", math.nan) - v3_values.get(f"{key}_mean", math.nan)
            rows.append(row)
    return rows, status


def write_report(path: Path, rows: list[dict[str, Any]], perf: list[dict[str, Any]], input_root: Path, comparison: list[dict[str, Any]]) -> None:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=False, capture_output=True, text=True).stdout.strip()
    lines = [
        "# SSI-MAG-V3 P1 Full NC mechanism health check",
        "",
        f"- Repository branch / commit: `{branch}` / `{commit}`",
        f"- Input root: `{input_root}`",
        "- Protocol: `model=ssi_mag_v3`, `ablation=full`, NC only, unified full-graph, seeds 42/43/44.",
        f"- Completed runs analyzed: {len(rows)}/15. No retraining was performed by this analyzer.",
        "- Checkpoint selection used validation Accuracy; test metrics are descriptive final metrics only.",
        "- LP jobs: 0; ablation jobs: 0; hyperparameter search: 0.",
        "",
        "## Performance",
        "",
        "| Dataset | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | best epoch | trained epochs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in perf:
        lines.append("| {dataset} | {va} ± {vas} | {vf} ± {vfs} | {ta} ± {tas} | {tf} ± {tfs} | {be} ± {bes} | {te} ± {tes} |".format(
            dataset=item["dataset"], va=fmt(item.get("val_acc_mean")), vas=fmt(item.get("val_acc_std")),
            vf=fmt(item.get("val_macro_f1_mean")), vfs=fmt(item.get("val_macro_f1_std")),
            ta=fmt(item.get("test_acc_mean")), tas=fmt(item.get("test_acc_std")),
            tf=fmt(item.get("test_macro_f1_mean")), tfs=fmt(item.get("test_macro_f1_std")),
            be=fmt(item.get("best_epoch_mean")), bes=fmt(item.get("best_epoch_std")),
            te=fmt(item.get("total_trained_epochs_mean")), tes=fmt(item.get("total_trained_epochs_std"))))
    lines += [
        "",
        "## Mechanism health snapshot (population mean over available seeds)",
        "",
        "| Dataset | beta T | beta V | operator rel-L1 T | operator rel-L1 V | alpha k1 std T | alpha k1 std V | g_delta T | g_int T | relation-off fused rel-L2 | interaction-off fused rel-L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS + ["ALL"]:
        group = [r for r in rows if dataset == "ALL" or r["dataset"] == dataset]
        def avg(key: str) -> float:
            values = [float(r[key]) for r in group if key in r and math.isfinite(float(r[key]))]
            return sum(values) / len(values) if values else math.nan
        keys = ("beta_text", "beta_visual", "operator_weight_relative_l1_text", "operator_weight_relative_l1_visual", "alpha_text_k1_std", "alpha_visual_k1_std", "delta_gate_text", "interaction_gate_text", "relation_off_z_relative_l2", "interaction_off_z_relative_l2")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(dataset, *(fmt(avg(key)) for key in keys)))
    lines += [
        "",
        "High-dimensional Q/S quantiles use a deterministic uniformly strided sample capped at 1,000,000 values to bound analyzer memory; full-tensor mean/std/min/max are retained.",
        "Train loss is recovered from the existing train log at the validation-selected best epoch; the framework does not log validation loss (val_loss_logged=0).",
        "",
        "## Mechanism evidence",
        "",
        "R1 reports learned beta, relation residual/weight distributions, local adaptation, and normalized-operator perturbation against the same physical topology with unit edge weights and the same self-loop policy.",
        "R2 reports per-hop alpha and d, semantic prior p, and learned bias/rho parameters. Alpha saturation flags use alpha < 0.05 or > 0.95.",
        "R3 reports context-change/interactions gates, D norms, attention entropy, diagonal/off-diagonal mass, and signed eta/effective-order statistics.",
        "Attention row diversity is the mean pairwise distance across query rows of the same node: mean L1/L2 distance over all unordered row pairs.",
        "",
        "## Frozen functional sensitivity",
        "",
        "relation_off and interaction_off are post-hoc frozen interventions evaluated with the same saved NC classifier/head. They are sensitivity diagnostics, not retrained causal ablations.",
        "The per-run CSV contains embedding MAE/relative-L2/cosine changes, prediction flip rates, and validation/test Accuracy and Macro-F1 deltas.",
        "",
        "## Automatic flags",
        "",
    ]
    flagged = [r for r in rows if int(r.get("flag_count", 0)) > 0]
    if flagged:
        for row in flagged:
            lines.append(f"- `{row['dataset']}/seed{row['seed']}`: {row['flags']}")
    else:
        lines.append("- None under the declared diagnostic thresholds.")
    lines += ["", "## Historical comparison", ""]
    if comparison:
        lines += [
            "| Dataset | Method | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for dataset in DATASETS + ["ALL"]:
            for item in comparison:
                if item["dataset"] != dataset:
                    continue
                lines.append("| {} | {} | {} | {} | {} | {} |".format(
                    dataset,
                    item["method"],
                    fmt(item.get("val_acc_mean")),
                    fmt(item.get("val_macro_f1_mean")),
                    fmt(item.get("test_acc_mean")),
                    fmt(item.get("test_macro_f1_mean")),
                ))
        lines.append("")
        lines.append("Historical rows are descriptive only; no baseline was rerun and no test metric was used for model selection.")
    else:
        lines.append("No provenance-verified historical comparison was available.")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Performance evidence, mechanism behavior evidence, and frozen functional sensitivity are reported as separate evidence types. This report does not declare a mechanism failed or propose an unvalidated fix.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT / "outputs/ssi_mag_v3_p1_full")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/ssi_mag_v3_p1_analysis")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset in args.datasets:
        for seed in args.seeds:
            run_dir = args.input_root / dataset / "full" / f"seed{seed}"
            marker = run_dir / "complete.marker"
            checkpoint = run_dir / "best.pt"
            if not marker.is_file() or not checkpoint.is_file():
                missing.append(f"{dataset}/seed{seed}")
                continue
            print(f"[analyze] {dataset}/seed{seed} device={device}", flush=True)
            rows.append(analyze_run(run_dir, device))
    write_csv(args.output_root / "p1_mechanism_per_run.csv", rows)
    mechanism = aggregate(rows, "dataset")
    if rows:
        all_rows = [dict(row, dataset="ALL") for row in rows]
        mechanism.append(aggregate(all_rows, "dataset")[0])
    write_csv(args.output_root / "p1_mechanism_summary.csv", mechanism)
    perf = performance_rows(rows)
    write_csv(args.output_root / "p1_performance_summary.csv", perf)
    comparison, comparison_status = historical_comparison(perf)
    write_csv(args.output_root / "p1_comparison_summary.csv", comparison)
    write_report(args.output_root / "p1_report.md", rows, perf, args.input_root, comparison)
    payload = {"rows": rows, "missing": missing, "device": str(device), "protocol": "unified_full_graph_nc_v1", "lp_jobs": 0, "ablation_jobs": 0, "historical_comparison_status": comparison_status, "historical_comparison": comparison}
    (args.output_root / "p1_analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(f"analysis_complete={len(rows)} missing={len(missing)} output={args.output_root}")
    if missing:
        print("missing_runs=" + ",".join(missing))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
