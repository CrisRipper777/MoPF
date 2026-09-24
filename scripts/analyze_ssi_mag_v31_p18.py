#!/usr/bin/env python3
"""Post-hoc P1.8 R1U mechanism and frozen-sensitivity analyzer.

The analyzer only loads validation-selected NC checkpoints. It never trains,
selects a checkpoint with test metrics, launches LP, or runs retrained
ablations. B/AB are the completed P1.7c controls and U is the single P1.8
direct-utilization pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_ssi_mag_v3_p15a import attention_diagnostics
from scripts.analyze_ssi_mag_v31_p17b import (
    _finite_analysis,
    _operator_metrics,
    _tensor_hash,
    _validate_attention_simplex,
)
from src.data import load_mag_data
from src.models import build_model

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
VARIANTS = ("U", "AB", "B")
EPS = 1.0e-12
PROTOCOL = "unified_full_graph_nc_v1"


def _flat(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _finite(value: Any) -> np.ndarray:
    x = _flat(value)
    return x[np.isfinite(x)]


def _to_cpu_tree(value: Any) -> Any:
    """Release GPU copies between full-graph intervention passes."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    return value


def _stats(value: Any) -> dict[str, float]:
    x = _finite(value)
    keys = ("mean", "std", "abs_mean", "min", "max", "q01", "q10", "q25", "q50", "q75", "q90", "q99")
    if x.size == 0:
        return {key: math.nan for key in keys}
    return {
        "mean": float(x.mean()), "std": float(x.std()), "abs_mean": float(np.abs(x).mean()),
        "min": float(x.min()), "max": float(x.max()),
        "q01": float(np.quantile(x, .01)), "q10": float(np.quantile(x, .10)),
        "q25": float(np.quantile(x, .25)), "q50": float(np.quantile(x, .50)),
        "q75": float(np.quantile(x, .75)), "q90": float(np.quantile(x, .90)),
        "q99": float(np.quantile(x, .99)),
    }


def _put(row: dict[str, Any], prefix: str, value: Any) -> None:
    row.update({f"{prefix}_{key}": item for key, item in _stats(value).items()})


def _ratio(num: float, den: float) -> float:
    return float(num / (abs(den) + EPS))


def _covariance_contribution(term: Any, total: Any) -> dict[str, float | int]:
    x, y = _flat(term), _flat(total)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    variance = float(np.var(y)) if y.size else 0.0
    if y.size < 3 or variance <= EPS:
        return {"covariance_over_variance": math.nan, "variance": variance, "variance_near_zero": 1}
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    return {"covariance_over_variance": covariance / variance, "variance": variance, "variance_near_zero": 0}


def _alpha_range_ratio(alpha: Any) -> float:
    x = _finite(alpha)
    return float((np.quantile(x, .90) - np.quantile(x, .10)) / (abs(x.mean()) + EPS)) if x.size else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _run_dir(root: Path, variant: str, dataset: str, seed: int) -> Path:
    return root / dataset / variant / f"seed{seed}"


def _check_lock(path: Path, *, model: str, variant: str, datasets: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"refusing analysis without provenance lock: {path}")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("task") != "nc" or lock.get("protocol") != PROTOCOL:
        raise SystemExit(f"invalid protocol provenance: {path}")
    if model == "ssi_mag_v31_r1u":
        expected = lock.get("model") == model and lock.get("variant") == variant
    else:
        expected = lock.get("model") == model and variant in lock.get("variants", {}).values()
    if (not expected or tuple(lock.get("datasets", [])) != datasets
            or tuple(int(x) for x in lock.get("seeds", [])) != seeds
            or int(lock.get("lp_jobs", -1)) != 0):
        raise SystemExit(f"provenance lock does not match requested set: {path}")
    return lock


def _load_checkpoint(root: Path, variant: str, dataset: str, seed: int, device: torch.device):
    run = _run_dir(root, variant, dataset, seed)
    required = [run / name for name in ("best.pt", "resolved_config.json", "complete.marker", "metrics.json", "results.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint artifacts: " + ", ".join(missing))
    marker = json.loads((run / "complete.marker").read_text(encoding="utf-8"))
    cfg = OmegaConf.create(json.loads((run / "resolved_config.json").read_text(encoding="utf-8")))
    expected_model = "ssi_mag_v31_r1u" if variant == "U" else "ssi_mag_v31_controls"
    if (marker.get("status") != "complete" or marker.get("task") != "nc"
            or marker.get("dataset") != dataset or int(marker.get("seed", -1)) != seed):
        raise ValueError(f"completion identity mismatch: {run}")
    if (str(cfg.model.name) != expected_model or str(cfg.task.name) != "nc"
            or str(cfg.task.protocol_version) != PROTOCOL
            or str(cfg.task.training_mode) != "full_graph" or str(cfg.ablation) != "full"):
        raise ValueError(f"protocol/model mismatch: {run}")
    if variant == "AB" and str(cfg.model.control) != "r1_r2":
        raise ValueError(f"AB control mismatch: {run}")
    if variant == "B" and str(cfg.model.control) != "r2_only":
        raise ValueError(f"B control mismatch: {run}")
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim, "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    payload = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("selection") != "best_val_accuracy" or "head_state" not in payload:
        raise ValueError(f"checkpoint selection/head schema mismatch: {run}")
    model = build_model(cfg, info).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()
    return run, cfg, data, model, head


def _performance_rows(roots: dict[str, Path], datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, root in roots.items():
        for dataset in datasets:
            for seed in seeds:
                run = _run_dir(root, variant, dataset, seed)
                result = json.loads((run / "results.json").read_text(encoding="utf-8"))
                metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                row: dict[str, Any] = {"row_type": "run", "variant": variant, "dataset": dataset, "seed": seed}
                for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                    row[key] = float(result[key]["mean"])
                row["best_epoch"] = metrics.get("best_epoch", math.nan)
                row["runtime_seconds"] = metrics.get("runtime_seconds", math.nan)
                row["peak_gpu_memory_mib"] = metrics.get("peak_gpu_memory_mib", math.nan)
                log = run / "train.log"
                text = log.read_text(errors="ignore") if log.is_file() else ""
                epochs = [int(value) for value in re.findall(r"Epoch\s+(\d+)", text)]
                row["total_trained_epochs"] = max(epochs) if epochs else math.nan
                row["nan_inf_detected"] = int(any(not math.isfinite(float(row[key])) for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")))
                rows.append(row)
    for variant in roots:
        for dataset in (*datasets, "ALL"):
            run_rows = [
                item for item in rows
                if item.get("row_type") == "run"
                and item["variant"] == variant
                and (dataset == "ALL" or item["dataset"] == dataset)
            ]
            # ALL is an unweighted mean of the five dataset means. Its
            # standard deviation is across dataset means, never pooled.
            if dataset == "ALL":
                grouped = {
                    name: [item for item in run_rows if item["dataset"] == name]
                    for name in datasets
                }
                value_groups = {
                    key: [
                        float(np.mean([float(item[key]) for item in group]))
                        for group in grouped.values() if group
                    ]
                    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
                }
                summary_groups = {
                    key: [
                        float(np.mean([float(item[key]) for item in group if math.isfinite(float(item.get(key, math.nan)))]))
                        for group in grouped.values()
                        if any(math.isfinite(float(item.get(key, math.nan))) for item in group)
                    ]
                    for key in ("best_epoch", "total_trained_epochs", "runtime_seconds", "peak_gpu_memory_mib", "nan_inf_detected")
                }
            else:
                value_groups = {
                    key: [float(item[key]) for item in run_rows]
                    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
                }
                summary_groups = {
                    key: [float(item[key]) for item in run_rows if math.isfinite(float(item.get(key, math.nan)))]
                    for key in ("best_epoch", "total_trained_epochs", "runtime_seconds", "peak_gpu_memory_mib", "nan_inf_detected")
                }
            row = {
                "row_type": "dataset_summary", "variant": variant, "dataset": dataset,
                "seed": "", "seed_count": len(run_rows), "run_count": len(run_rows),
                "dataset_count": len(datasets) if dataset == "ALL" else 1,
                "aggregation": "unweighted_mean_of_dataset_means" if dataset == "ALL" else "seed_mean",
            }
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1", "best_epoch", "total_trained_epochs", "runtime_seconds", "peak_gpu_memory_mib", "nan_inf_detected"):
                values = np.asarray((value_groups if key in value_groups else summary_groups)[key], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size:
                    row[f"{key}_mean"] = float(values.mean())
                    row[f"{key}_population_std"] = float(values.std())
            rows.append(row)
    return rows

def _relation_rows(normal: dict[str, Any], model: Any, variant: str, dataset: str, seed: int, num_nodes: int, dtype: torch.dtype) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edge_index = normal["physical_edge_index"]
    nonself = edge_index[0] != edge_index[1]
    relation_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        score = normal[f"a_{modality}"][nonself]
        weights = normal[f"relation_weight_{modality}"][nonself]
        compatibility = normal[f"relation_compatibility_{modality}"][nonself]
        relative = normal[f"relation_centered_{modality}"][nonself]
        row: dict[str, Any] = {
            "row_type": "relation", "variant": variant, "dataset": dataset, "seed": seed,
            "modality": modality, "nonself_edge_count": int(score.numel()),
            "beta_parameter_present": int(hasattr(model, f"theta_beta_{modality}")),
        }
        beta = normal.get(f"beta_parameter_{modality}")
        row["beta_parameter"] = float(beta.item()) if torch.is_tensor(beta) and beta.numel() == 1 else 0.0
        _put(row, "semantic_compatibility", compatibility)
        _put(row, "relative_compatibility", relative)
        _put(row, "relation_score", score)
        log_weight = weights.log() if weights.numel() else weights
        _put(row, "log_weight_modulation", log_weight)
        _put(row, "relation_weight", weights)
        row["relation_weight_cv"] = _ratio(row["relation_weight_std"], row["relation_weight_mean"])
        row["relation_weight_bounds_ok"] = int(bool(
            weights.numel() == 0 or
            ((weights > math.exp(-1.0) - 1e-6) & (weights < math.exp(1.0) + 1e-6)).all().item()
        ))
        scorer = getattr(model, f"relation_scorer_{modality}")
        if isinstance(scorer, nn.Linear) and int(scorer.in_features) == 3:
            descriptor = torch.stack((compatibility, relative, relative.abs()), dim=-1)
            logit = F.linear(descriptor, scorer.weight, scorer.bias).squeeze(-1)
            row["scorer_w_s"] = float(scorer.weight[0, 0].detach().cpu())
            row["scorer_w_r"] = float(scorer.weight[0, 1].detach().cpu())
            row["scorer_w_abs_r"] = float(scorer.weight[0, 2].detach().cpu())
            row["scorer_bias"] = float(scorer.bias[0].detach().cpu())
            _put(row, "semantic_term", scorer.weight[0, 0] * compatibility)
            _put(row, "relative_term", scorer.weight[0, 1] * relative)
            _put(row, "abs_relative_term", scorer.weight[0, 2] * relative.abs())
            _put(row, "dynamic_logit", logit)
        else:
            for key in ("semantic_term", "relative_term", "abs_relative_term", "dynamic_logit"):
                _put(row, key, torch.empty(0, device=score.device))
        row["score_fraction_abs_gt_0.8"] = float((score.abs() > .8).float().mean().item()) if score.numel() else math.nan
        row["score_fraction_abs_gt_0.9"] = float((score.abs() > .9).float().mean().item()) if score.numel() else math.nan
        c = normal[f"c_{modality}"]
        _put(row, "local_adaptation", c)
        row["isolated_c_zero_fraction"] = math.nan
        if nonself.any():
            degree = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.long)
            endpoints = torch.cat((edge_index[0, nonself], edge_index[1, nonself]))
            degree.index_add_(0, endpoints, torch.ones_like(endpoints))
            isolated = degree == 0
            if isolated.any():
                row["isolated_c_zero_fraction"] = float((c[isolated] == 0).float().mean().item())
        row.update(_operator_metrics(
            model, edge_index, normal[f"normalized_edge_index_{modality}"],
            normal[f"normalized_edge_weight_{modality}"], num_nodes, dtype
        ))
        relation_rows.append(row)

        eta = normal[f"eta_{modality}"]
        content = normal[f"delta_content_{modality}"]
        rel_filter = normal[f"relation_filter_residual_{modality}"]
        reference = normal.get(f"reference_residual_{modality}", torch.zeros_like(rel_filter))
        for hop in range(eta.size(1)):
            filter_row = {
                "row_type": "filter", "variant": variant, "dataset": dataset,
                "seed": seed, "modality": modality, "hop": hop,
                "gamma": float(normal[f"gamma_{modality}"][hop]),
                "delta_gamma": float(normal[f"delta_gamma_{modality}"][hop]),
            }
            _put(filter_row, "delta_content", content[:, hop])
            _put(filter_row, "reference_residual", reference[:, hop])
            _put(filter_row, "relation_filter_residual", rel_filter[:, hop])
            _put(filter_row, "eta", eta[:, hop])
            filter_row["negative_eta_fraction"] = float((eta[:, hop] < 0).float().mean().item())
            filter_row.update({
                f"relation_{key}": value
                for key, value in _covariance_contribution(rel_filter[:, hop], eta[:, hop]).items()
            })
            filter_row["relation_covariance_contribution"] = filter_row.get(
                "relation_covariance_over_variance", math.nan
            )
            filter_rows.append(filter_row)
    return relation_rows, filter_rows


def _stage2_rows(normal: dict[str, Any], model: Any, variant: str, dataset: str, seed: int, interaction_off: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        p = normal[f"p_{modality}"]
        bias = normal[f"semantic_bias_{modality}"]
        rho_p = float(normal[f"semantic_rho_p_{modality}"])
        rho_d = float(normal[f"semantic_rho_d_{modality}"])
        rho_c = (
            float(getattr(model, f"semantic_rho_c_{modality}").detach().cpu())
            if hasattr(model, f"semantic_rho_c_{modality}") else math.nan
        )
        alpha_list = normal[f"alpha_{modality}"]
        d_list = normal[f"d_{modality}"]
        for hop, (alpha, change) in enumerate(zip(alpha_list, d_list, strict=True), start=1):
            row = {
                "row_type": "semantic", "variant": variant, "dataset": dataset,
                "seed": seed, "modality": modality, "hop": hop,
                "semantic_bias": float(bias[hop - 1]), "rho_p": rho_p,
                "rho_d": rho_d, "rho_c": rho_c,
                "alpha_range_ratio": _alpha_range_ratio(alpha),
                "alpha_fraction_lt_0.05": float((alpha < .05).float().mean().item()),
                "alpha_fraction_gt_0.95": float((alpha > .95).float().mean().item()),
            }
            _put(row, "p", p)
            _put(row, "alpha", alpha)
            _put(row, "d", change)
            rows.append(row)
        attention = normal[f"attention_{modality}"]
        simplex = _validate_attention_simplex(attention)
        _, _, attention_metrics = attention_diagnostics(attention)
        rows.append({
            "row_type": "attention", "variant": variant, "dataset": dataset,
            "seed": seed, "modality": modality, **simplex, **attention_metrics,
        })
        states = normal[f"S_{modality}"]
        deltas = normal[f"D_{modality}"]
        s_tilde = normal[f"S_tilde_{modality}"]
        for hop in range(1, len(states)):
            row = {
                "row_type": "context", "variant": variant, "dataset": dataset,
                "seed": seed, "modality": modality, "hop": hop,
                "delta_gate": float(normal[f"delta_gate_{modality}"].item()),
                "interaction_gate": float(normal[f"interaction_gate_{modality}"].item()),
                "interaction_output_norm_mean": float(
                    normal[f"interaction_output_{modality}"][:, hop].norm(dim=-1).mean().item()
                ),
            }
            _put(row, "D_norm", deltas[hop].norm(dim=-1))
            _put(row, "S_tilde_minus_S", s_tilde[hop] - states[hop])
            rows.append(row)
        row = {
            "row_type": "effective_order", "variant": variant, "dataset": dataset,
            "seed": seed, "modality": modality,
        }
        _put(row, "effective_order", normal[f"effective_order_{modality}"])
        _put(row, "effective_radius", normal[f"effective_radius_{modality}"])
        rows.append(row)
        if interaction_off is not None:
            exact = max(
                float((interaction_off[f"S_tilde_{modality}"][hop] - states[hop]).abs().max().item())
                for hop in range(len(states))
            )
            rows.append({
                "row_type": "interaction_off_check", "variant": variant,
                "dataset": dataset, "seed": seed, "modality": modality,
                "max_abs_S_tilde_minus_S": exact, "exact_within_1e-7": int(exact <= 1e-7), "exact_within_1e-5": int(exact <= 1e-5), "exact_within_1e-4": int(exact <= 1e-4),
            })
    return rows


def _sensitivity(normal: dict[str, Any], relation_off: dict[str, Any], interaction_off: dict[str, Any], head: nn.Module, data: Any, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {"variant": variant, "dataset": dataset, "seed": seed}
    base_logits = head(normal["z"])
    base_pred = base_logits.argmax(-1)
    for tag, changed in (("relation_off", relation_off), ("interaction_off", interaction_off)):
        for source, label in (("z", "fused_embedding"), ("z_text", "text_embedding"), ("z_visual", "visual_embedding")):
            diff = changed[source].float() - normal[source].float()
            row[f"{tag}_{label}_mae"] = float(diff.abs().mean().item())
            row[f"{tag}_{label}_relative_l2"] = float(
                diff.norm().item() / (normal[source].float().norm().item() + EPS)
            )
            row[f"{tag}_{label}_cosine"] = float(
                F.cosine_similarity(changed[source].float(), normal[source].float(), dim=-1).mean().item()
            )
        logits = head(changed["z"])
        pred = logits.argmax(-1)
        row[f"{tag}_prediction_flip_rate"] = float((pred != base_pred).float().mean().item())
        for split, index in (("val", data.val_idx), ("test", data.test_idx)):
            target = data.y[index].detach().cpu().numpy()
            before = base_pred[index].detach().cpu().numpy()
            after = pred[index].detach().cpu().numpy()
            labels = list(range(int(data.num_classes)))
            row[f"{tag}_{split}_acc_delta"] = float((after == target).mean() - (before == target).mean())
            row[f"{tag}_{split}_macro_f1_delta"] = float(
                f1_score(target, after, labels=labels, average="macro", zero_division=0)
                - f1_score(target, before, labels=labels, average="macro", zero_division=0)
            )
    return row

def _comparison(performance: list[dict[str, Any]], datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    runs = {(row["variant"], row["dataset"], int(row["seed"])): row for row in performance if row.get("row_type") == "run"}
    rows: list[dict[str, Any]] = []
    for control in ("AB", "B"):
        for dataset in datasets:
            for seed in seeds:
                u = runs[("U", dataset, seed)]
                c = runs[(control, dataset, seed)]
                row: dict[str, Any] = {
                    "row_type": "performance_paired",
                    "comparison": f"U_vs_{control}", "dataset": dataset, "seed": seed,
                }
                for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                    row[f"U_{key}"] = u[key]
                    row[f"{control}_{key}"] = c[key]
                    row[f"delta_{key}"] = u[key] - c[key]
                rows.append(row)
            subset = [
                row for row in rows
                if row.get("row_type") == "performance_paired"
                and row["comparison"] == f"U_vs_{control}" and row["dataset"] == dataset
            ]
            row = {
                "row_type": "performance_dataset_summary",
                "comparison": f"U_vs_{control}", "dataset": dataset, "seed": "",
            }
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                values = np.asarray([item[f"delta_{key}"] for item in subset], dtype=np.float64)
                row[f"delta_{key}_mean"] = float(values.mean())
                row[f"delta_{key}_population_std"] = float(values.std())
            rows.append(row)
        subset = [
            row for row in rows
            if row.get("row_type") == "performance_paired"
            and row["comparison"] == f"U_vs_{control}"
        ]
        row = {
            "row_type": "performance_dataset_summary",
            "comparison": f"U_vs_{control}", "dataset": "ALL", "seed": "",
        }
        for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
            values = np.asarray([item[f"delta_{key}"] for item in subset], dtype=np.float64)
            row[f"delta_{key}_mean"] = float(values.mean())
            row[f"delta_{key}_population_std"] = float(values.std())
        rows.append(row)
    return rows


def _flags(relation_rows: list[dict[str, Any]], stage_rows: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for row in relation_rows:
        if row.get("row_type") != "relation":
            continue
        tag = f"{row['variant']}/{row['dataset']}/seed{row['seed']}/{row['modality']}"
        if row.get("relation_weight_cv", 0.0) < 1e-4:
            flags.append(f"R1_weight_cv_negligible:{tag}")
        if row.get("operator_weight_relative_l1", 0.0) < 1e-4:
            flags.append(f"R1_operator_perturbation_near_zero:{tag}")
        if row.get("score_fraction_abs_gt_0.9", 0.0) > .5:
            flags.append(f"R1_softsign_saturation:{tag}")
        if row.get("relation_score_abs_mean", 0.0) < 1e-4:
            flags.append(f"R1_score_collapse:{tag}")
        if not row.get("relation_weight_bounds_ok", 0):
            flags.append(f"R1_weight_bound_violation:{tag}")
    for row in stage_rows:
        tag = f"{row.get('variant')}/{row.get('dataset')}/seed{row.get('seed')}/{row.get('modality')}"
        if row.get("row_type") == "semantic":
            if row.get("alpha_std", 1.0) < 1e-5:
                flags.append(f"R2_alpha_node_variance_negligible:{tag}/hop{row.get('hop')}")
            if row.get("alpha_fraction_lt_0.05", 0.0) > .5 or row.get("alpha_fraction_gt_0.95", 0.0) > .5:
                flags.append(f"R2_alpha_saturation:{tag}/hop{row.get('hop')}")
        elif row.get("row_type") == "attention":
            if row.get("attention_nonuniformity", 1.0) < 1e-3:
                flags.append(f"R3_attention_near_uniform:{tag}")
            if row.get("diagonal_mass_mean", 0.0) > 1.0 - 1e-3:
                flags.append(f"R3_attention_near_diagonal:{tag}")
        elif row.get("row_type") == "context":
            if abs(row.get("delta_gate", 1.0)) < .005:
                flags.append(f"R3_delta_gate_near_zero:{tag}/hop{row.get('hop')}")
            if abs(row.get("interaction_gate", 1.0)) < .005:
                flags.append(f"R3_interaction_gate_near_zero:{tag}/hop{row.get('hop')}")
        elif row.get("row_type") == "interaction_off_check" and row.get("exact_within_1e-4") != 1:
            flags.append(f"R3_interaction_off_not_exact:{tag}")
    return sorted(set(flags))


def _report(output: Path, summary: dict[str, Any]) -> None:
    guard = summary["performance_guardrail"]
    lines = [
        "# SSI-MAG-V3.1 P1.8 R1U Post-hoc Report", "",
        "This report separates performance evidence, mechanism behavior evidence, and frozen functional sensitivity. U is the only newly trained variant; B and AB are provenance-locked P1.7c controls.", "",
        "## Integrity", "",
        f"- U expected/loaded/finite: {summary['expected_u_runs']}/{summary['loaded_u_runs']}/{summary['finite_u_runs']}.",
        f"- Control runs loaded: {summary['control_loaded_runs']}; attention failures: {summary['attention_failures']}.",
        f"- Device: {summary['device']}; analyzer training invoked: {summary['training_invoked']}; LP jobs: {summary['lp_jobs']}.",
        "- All model selection remains validation Accuracy; test metrics are descriptive only.", "",
        "## Performance evidence", "",
        "See p18_performance.csv for per-run metrics and population-standard-deviation summaries. p18_r1_comparison.csv contains paired U-vs-AB and U-vs-B validation/test deltas.",
        f"- Predefined performance trigger: {guard['status']}; mean U-vs-B validation Accuracy delta {guard['mean_val_acc_delta_pp']:.4f} pp, Macro-F1 delta {guard['mean_val_f1_delta_pp']:.4f} pp; simultaneous dataset declines {guard['simultaneous_declines']}.",
        "- This is a review trigger, not a final paper-model decision.", "",
        "## R1 semantic behavior", "",
        "U reports semantic compatibility, centered relative compatibility, linear scorer terms, bounded softsign score tails, log-weight modulation, weight quantiles/CV/bounds, and local adaptation c in p18_r1_comparison.csv.", "",
        "## R1 propagation utilization", "",
        "Normalized operator MAE/RMSE/relative-L1 are aligned by directed edge-pair multisets against the same raw unit-weight topology; self-loop insertion is not compared by position.", "",
        "## Stage-II sanity", "",
        "Corrected alpha range ratio is (q90-q10)/(abs(mean(alpha))+eps). Attention metrics use nodewise entropy, query-row total-variation diversity, node heterogeneity, diagonal/off-diagonal mass, and simplex validation. Reference residuals, signed eta, effective order, and interaction-off equality (with a 1e-4 repeated-GPU-replay tolerance) are in p18_stage2_sanity.csv.", "",
        "## Frozen sensitivity", "",
        "Relation-off and interaction-off keep the trained model and NC head fixed. They are functional sensitivity diagnostics, not causal necessity claims and not retrained ablations.", "",
        "## Flags", "",
        f"- Automatic descriptive flags: {len(summary['flags'])}. Flags report quantities below fixed diagnostic thresholds; no repair or tuning was performed.",
        "- No final model choice is made here; the open question is whether R1U direct utilization provides enough validated benefit to justify further human review.", "",
        "## Prohibited work confirmation", "",
        "No LP, no additional relation variant, no hyperparameter search, no auxiliary loss, no test-based selection, and no retrained ablation was run in this analysis.", "",
    ]
    (output / "p18_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision_packet(summary: dict[str, Any]) -> None:
    guard = summary["performance_guardrail"]
    text = "\n".join([
        "# SSI-MAG-V3.1 P1.8 Decision Packet", "",
        "Decision status: OPEN REVIEW — no final paper-model decision", "",
        "## Scope", "",
        "P1.8 trains exactly one independent direct-utilization R1U variant (U) under the frozen full-graph NC protocol. P1.7c B/AB are read-only controls.", "",
        "## Evidence separation", "",
        "- Performance evidence: outputs/ssi_mag_v31_p18_analysis/p18_performance.csv and paired rows in p18_r1_comparison.csv.",
        "- Mechanism behavior evidence: R1 semantic scorer/weight/operator diagnostics, Stage-II alpha/attention/signed-filter diagnostics.",
        "- Frozen functional sensitivity: relation-off and interaction-off with the same saved NC head; not retrained causal ablations.", "",
        "## Predefined trigger", "",
        f"U-vs-B validation guardrail status: {guard['status']}; mean Accuracy delta {guard['mean_val_acc_delta_pp']:.4f} pp; mean Macro-F1 delta {guard['mean_val_f1_delta_pp']:.4f} pp; simultaneous declines {guard['simultaneous_declines']}.",
        "This trigger only requests review and does not decide the final paper model.", "",
        "## Boundary", "",
        "No LP, no extra relation variants, no tuning, no auxiliary loss, no test-based selection, and no additional ablation training were performed.", "",
    ]) + "\n"
    (ROOT / "docs/ssi_mag_v31_p18_decision_packet.md").write_text(text, encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/ssi_mag_v31_p18_r1u"))
    parser.add_argument("--p17c-root", type=Path, default=Path("outputs/ssi_mag_v31_p17c_controls"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssi_mag_v31_p18_analysis"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    input_root = (ROOT / args.input_root).resolve() if not args.input_root.is_absolute() else args.input_root.resolve()
    controls_root = (ROOT / args.p17c_root).resolve() if not args.p17c_root.is_absolute() else args.p17c_root.resolve()
    output = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    datasets, seeds = tuple(args.datasets), tuple(args.seeds)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested analysis device is unavailable: {device}")
    u_lock = _check_lock(
        input_root / "provenance.lock.json", model="ssi_mag_v31_r1u",
        variant="U", datasets=datasets, seeds=seeds
    )
    c_lock = _check_lock(
        controls_root / "provenance.lock.json", model="ssi_mag_v31_controls",
        variant="AB", datasets=datasets, seeds=seeds
    )
    roots = {"U": input_root, "AB": controls_root, "B": controls_root}
    performance = _performance_rows(roots, datasets, seeds)
    comparison = _comparison(performance, datasets, seeds)
    relation_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    attention_failures: list[str] = []
    loaded: defaultdict[str, int] = defaultdict(int)
    finite_runs: defaultdict[str, int] = defaultdict(int)
    signatures: dict[str, dict[int, str]] = defaultdict(dict)

    for variant in VARIANTS:
        for dataset in datasets:
            for seed in seeds:
                try:
                    run, cfg, data, model, head = _load_checkpoint(
                        roots[variant], variant, dataset, seed, device
                    )
                    del run, cfg
                    x, edge = data.x.to(device), data.edge_index.to(device)
                    with torch.no_grad():
                        normal = model.analysis(x, edge)
                        if device.type == "cuda":
                            normal = _to_cpu_tree(normal)
                            torch.cuda.empty_cache()
                        relation_off = (
                            model.analysis_intervention(x, edge, relation="off")
                            if variant == "U" else None
                        )
                        if device.type == "cuda" and relation_off is not None:
                            relation_off = _to_cpu_tree(relation_off)
                            torch.cuda.empty_cache()
                        interaction_off = (
                            model.analysis_intervention(x, edge, interaction="off")
                            if variant == "U" else None
                        )
                        if device.type == "cuda" and interaction_off is not None:
                            interaction_off = _to_cpu_tree(interaction_off)
                            torch.cuda.empty_cache()
                    if device.type == "cuda":
                        model = model.cpu()
                        head = head.cpu()
                    finite = _finite_analysis(normal) and (
                        variant != "U" or (
                            _finite_analysis(relation_off)
                            and _finite_analysis(interaction_off)
                        )
                    )
                    if not finite:
                        raise ValueError("analysis contains NaN/Inf")
                    for modality in MODALITIES:
                        try:
                            _validate_attention_simplex(normal[f"attention_{modality}"])
                        except ValueError as exc:
                            attention_failures.append(
                                f"{variant}/{dataset}/seed{seed}/{modality}: {exc}"
                            )
                            raise
                    loaded[variant] += 1
                    finite_runs[variant] += 1
                    signatures[dataset][seed] = _tensor_hash(data.edge_index)
                    rel, filt = _relation_rows(
                        normal, model, variant, dataset, seed,
                        data.num_nodes, x.dtype
                    )
                    relation_rows.extend(rel)
                    relation_rows.extend(filt)
                    if variant == "U":
                        stage_rows.extend(filt)
                        stage_rows.extend(_stage2_rows(
                            normal, model, variant, dataset, seed, interaction_off
                        ))
                        sensitivity_rows.append(_sensitivity(
                            normal, relation_off, interaction_off, head, data,
                            variant, dataset, seed
                        ))
                    del normal, relation_off, interaction_off, model, head, data, x, edge
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"OK {variant} {dataset} seed={seed}", flush=True)
                except Exception as exc:
                    errors.append({
                        "variant": variant, "dataset": dataset, "seed": seed,
                        "error": repr(exc),
                    })
                    print(f"ERROR {variant} {dataset} seed={seed}: {exc}", flush=True)

    guard_rows = [
        row for row in comparison
        if row.get("row_type") == "performance_dataset_summary"
        and row.get("comparison") == "U_vs_B"
        and row.get("dataset") in datasets
    ]
    acc = [float(row["delta_val_acc_mean"]) for row in guard_rows]
    f1 = [float(row["delta_val_macro_f1_mean"]) for row in guard_rows]
    declines = [
        row["dataset"] for row in guard_rows
        if row["delta_val_acc_mean"] < 0 and row["delta_val_macro_f1_mean"] < 0
    ]
    mean_acc, mean_f1 = float(np.mean(acc)), float(np.mean(f1))
    guard_status = (
        "PERFORMANCE_REVIEW_TRIGGER"
        if mean_acc < -.003 or mean_f1 < -.003 or len(declines) >= 3
        else "WITHIN_PREDEFINED_PERFORMANCE_GUARDRAIL"
    )
    guard = {
        "status": guard_status,
        "mean_val_acc_delta": mean_acc,
        "mean_val_f1_delta": mean_f1,
        "mean_val_acc_delta_pp": 100 * mean_acc,
        "mean_val_f1_delta_pp": 100 * mean_f1,
        "simultaneous_declines": declines,
        "threshold_pp": -.30,
        "test_used": False,
    }
    flags = _flags(relation_rows, stage_rows)
    ordering = {
        dataset: (
            len(signatures.get(dataset, {})) == len(seeds)
            and len(set(signatures.get(dataset, {}).values())) == 1
        )
        for dataset in datasets
    }
    summary = {
        "schema": "ssi_mag_v31_p18_r1u_analysis_v1",
        "datasets": list(datasets), "seeds": list(seeds),
        "expected_u_runs": len(datasets) * len(seeds),
        "loaded_u_runs": loaded["U"], "finite_u_runs": finite_runs["U"],
        "control_expected_runs": 2 * len(datasets) * len(seeds),
        "control_loaded_runs": loaded["AB"] + loaded["B"],
        "variant_loaded_runs": dict(loaded), "errors": errors,
        "device": str(device), "input_root": str(input_root),
        "controls_root": str(controls_root),
        "provenance": {"u": u_lock, "p17c": c_lock},
        "node_ordering_verified": ordering,
        "attention_failures": attention_failures,
        "flags": flags, "performance_guardrail": guard,
        "training_invoked": False, "ablation_invoked": False,
        "lp_invoked": False, "lp_jobs": 0,
        "test_metrics_used_for_selection_or_decision": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "p18_performance.csv", performance)
    _write_csv(output / "p18_r1_comparison.csv", comparison + relation_rows)
    _write_csv(output / "p18_r1_sensitivity.csv", sensitivity_rows)
    _write_csv(output / "p18_stage2_sanity.csv", stage_rows)
    (output / "p18_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    _report(output, summary)
    _decision_packet(summary)
    print(json.dumps(summary, indent=2, default=_json_default), flush=True)
    expected = len(datasets) * len(seeds)
    return 0 if (
        loaded["U"] == expected and loaded["AB"] == expected
        and loaded["B"] == expected and not errors
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
