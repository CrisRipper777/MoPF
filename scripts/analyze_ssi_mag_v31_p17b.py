#!/usr/bin/env python3
"""Post-hoc diagnostics for future SSI-MAG-V3.1 Full NC checkpoints.

This script is analysis-only. It loads best validation-selected checkpoints,
NC classifier heads, and NC graph/features in eval/no-grad mode. It never
trains, selects checkpoints, reads test metrics for a decision, runs LP, or
runs ablations. P1.7a.1 only implements and synthetically tests this future
P1.7b analyzer; it must not be pointed at a nonexistent/partial checkpoint
root during this stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_ssi_mag_v3_p15a import attention_diagnostics  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
EPS = 1.0e-12


def _as_np(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _flat(value: Any) -> np.ndarray:
    return _as_np(value).reshape(-1)


def _finite(value: Any) -> np.ndarray:
    x = _flat(value)
    return x[np.isfinite(x)]


def _stats(value: Any) -> dict[str, float]:
    x = _finite(value)
    keys = ("mean", "std", "abs_mean", "q10", "q25", "q50", "q75", "q90")
    if x.size == 0:
        return {key: math.nan for key in keys}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "abs_mean": float(np.mean(np.abs(x))),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
    }


def _put(row: dict[str, Any], prefix: str, value: Any) -> None:
    for key, item in _stats(value).items():
        row[f"{prefix}_{key}"] = item


def _safe_ratio(num: float, den: float) -> float:
    return float(num / (abs(den) + EPS))


def _corr(x: Any, y: Any) -> tuple[float, float]:
    a, b = _flat(x), _flat(y)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) <= EPS or np.std(b) <= EPS:
        return math.nan, math.nan
    return float(pearsonr(a, b).statistic), float(spearmanr(a, b).statistic)


def _covariance_contribution(term: Any, total: Any) -> dict[str, float | int]:
    x, y = _flat(term), _flat(total)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    variance = float(np.var(y)) if y.size else 0.0
    if y.size < 3 or variance <= EPS:
        return {
            "covariance_contribution": math.nan,
            "covariance_correlation": math.nan,
            "total_variance": variance,
            "variance_near_zero": 1,
        }
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    correlation = math.nan if np.std(x) <= EPS else float(np.corrcoef(x, y)[0, 1])
    return {
        "covariance_contribution": covariance / variance,
        "covariance_correlation": correlation,
        "total_variance": variance,
        "variance_near_zero": 0,
    }


def _alpha_range_ratio(alpha: Any, eps: float = EPS) -> float:
    x = _finite(alpha)
    if x.size == 0:
        return math.nan
    return float((np.quantile(x, 0.90) - np.quantile(x, 0.10)) / (abs(np.mean(x)) + eps))


def _tail_fractions(value: Any, thresholds: tuple[float, ...] = (3.0, 5.0)) -> dict[str, float]:
    x = _finite(value)
    return {
        f"fraction_abs_gt_{threshold:g}": float(np.mean(np.abs(x) > threshold)) if x.size else math.nan
        for threshold in thresholds
    }


def _edge_occurrences(
    edge_index: torch.Tensor, weights: torch.Tensor, num_nodes: int
) -> dict[int, list[float]]:
    result: dict[int, list[float]] = defaultdict(list)
    for (src, dst), weight in zip(
        edge_index.detach().cpu().t().tolist(), weights.detach().cpu().tolist()
    ):
        result[int(src) * num_nodes + int(dst)].append(float(weight))
    for values in result.values():
        values.sort()
    return result


def _operator_metrics(
    model: Any,
    physical_edge_index: torch.Tensor,
    learned_index: torch.Tensor,
    learned_weight: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Compare normalized operators after reliable edge-pair multiset alignment."""
    raw_index, raw_weight = model._normalized_operator(
        physical_edge_index,
        torch.ones(physical_edge_index.size(1), device=physical_edge_index.device, dtype=dtype),
        num_nodes,
        dtype,
    )
    learned = _edge_occurrences(learned_index, learned_weight, num_nodes)
    raw = _edge_occurrences(raw_index, raw_weight, num_nodes)
    diffs: list[float] = []
    reference: list[float] = []
    missing_learned = 0
    missing_raw = 0
    for key in set(learned) | set(raw):
        left, right = learned.get(key, []), raw.get(key, [])
        count = min(len(left), len(right))
        diffs.extend(a - b for a, b in zip(left[:count], right[:count]))
        reference.extend(right[:count])
        missing_learned += max(0, len(right) - count)
        missing_raw += max(0, len(left) - count)
    if not diffs:
        return {
            "operator_aligned_edges": 0.0,
            "operator_weight_mae": math.nan,
            "operator_weight_rmse": math.nan,
            "operator_weight_relative_l1": math.nan,
            "operator_missing_learned": float(missing_learned),
            "operator_missing_raw": float(missing_raw),
        }
    delta = np.asarray(diffs, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    return {
        "operator_aligned_edges": float(delta.size),
        "operator_weight_mae": float(np.mean(np.abs(delta))),
        "operator_weight_rmse": float(np.sqrt(np.mean(delta * delta))),
        "operator_weight_relative_l1": float(np.sum(np.abs(delta)) / (np.sum(np.abs(ref)) + EPS)),
        "operator_missing_learned": float(missing_learned),
        "operator_missing_raw": float(missing_raw),
    }


def _semantic_pair_metrics(left: Any, right: Any, prefix: str) -> dict[str, float]:
    a, b = _flat(left), _flat(right)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return {f"{prefix}_{key}": math.nan for key in ("pearson", "spearman", "mean_abs_difference", "q10_abs_difference", "q50_abs_difference", "q90_abs_difference")}
    pearson, spearman = _corr(a, b)
    difference = np.abs(a - b)
    return {
        f"{prefix}_pearson": pearson,
        f"{prefix}_spearman": spearman,
        f"{prefix}_mean_abs_difference": float(np.mean(difference)),
        f"{prefix}_q10_abs_difference": float(np.quantile(difference, 0.10)),
        f"{prefix}_q50_abs_difference": float(np.quantile(difference, 0.50)),
        f"{prefix}_q90_abs_difference": float(np.quantile(difference, 0.90)),
    }


def _tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _finite_analysis(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, list):
        return all(_finite_analysis(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_analysis(item) for item in value.values())
    return True


def _ordering_verified(signatures: dict[int, tuple[Any, ...]], seeds: tuple[int, ...]) -> bool:
    return len(signatures) == len(seeds) and len(set(signatures.values())) == 1


def _cross_seed_rows(
    vectors: dict[tuple[str, int, str], dict[str, Any]],
    signatures: dict[str, dict[int, tuple[Any, ...]]],
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    verified: dict[str, bool] = {}
    for dataset in datasets:
        ok = _ordering_verified(signatures.get(dataset, {}), seeds)
        verified[dataset] = ok
        if not ok:
            rows.append({"dataset": dataset, "ordering_verified": False, "metric": "not_computed"})
            continue
        for modality in MODALITIES:
            for left_index, seed_a in enumerate(seeds):
                for seed_b in seeds[left_index + 1 :]:
                    left = vectors[(dataset, seed_a, modality)]
                    right = vectors[(dataset, seed_b, modality)]
                    pearson, spearman = _corr(left["term_p"], right["term_p"])
                    rows.append({"dataset": dataset, "modality": modality, "metric": "term_p", "seed_a": seed_a, "seed_b": seed_b, "ordering_verified": True, "pearson": pearson, "spearman": spearman})
                    for hop, (alpha_a, alpha_b) in enumerate(zip(left["alpha"], right["alpha"], strict=True), start=1):
                        pearson, spearman = _corr(alpha_a, alpha_b)
                        rows.append({"dataset": dataset, "modality": modality, "metric": "alpha", "hop": hop, "seed_a": seed_a, "seed_b": seed_b, "ordering_verified": True, "pearson": pearson, "spearman": spearman})
    return rows, verified


def _checkpoint_run(input_root: Path, dataset: str, seed: int) -> Path:
    return input_root / dataset / "full" / f"seed{seed}"


def _load_run(input_root: Path, dataset: str, seed: int, device: torch.device):
    run_dir = _checkpoint_run(input_root, dataset, seed)
    required = (run_dir / "best.pt", run_dir / "resolved_config.json", run_dir / "complete.marker")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing explicit-success checkpoint artifacts: " + ", ".join(missing))
    marker = json.loads((run_dir / "complete.marker").read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("task") != "nc" or int(marker.get("seed", -1)) != seed:
        raise ValueError(f"checkpoint is not an explicit successful NC seed={seed}: {marker}")
    cfg = OmegaConf.create(json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8")))
    if str(cfg.model.name) != "ssi_mag_v31" or str(cfg.task.name) != "nc" or str(cfg.ablation) != "full":
        raise ValueError(f"unexpected V3.1 Full NC config in {run_dir}")
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    payload = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"unexpected checkpoint selection schema in {run_dir}")
    model = build_model(cfg, data_info).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()
    return run_dir, cfg, data, model, head


def _relation_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int, num_nodes: int, dtype: torch.dtype) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    edge_index = normal["physical_edge_index"]
    nonself = edge_index[0] != edge_index[1]
    rows: list[dict[str, Any]] = []
    values: dict[str, torch.Tensor] = {}
    for modality in MODALITIES:
        compatibility = normal[f"relation_compatibility_{modality}"][nonself]
        relative = normal[f"relation_centered_{modality}"][nonself]
        score = normal[f"a_{modality}"][nonself]
        weights = normal[f"relation_weight_{modality}"][nonself]
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "modality": modality,
            "nonself_edge_count": int(nonself.sum().item()),
            "beta": float(normal[f"beta_parameter_{modality}"].item()),
            "score_fraction_abs_gt_0.8": float((score.abs() > 0.8).float().mean().item()) if score.numel() else math.nan,
            "score_fraction_abs_gt_0.9": float((score.abs() > 0.9).float().mean().item()) if score.numel() else math.nan,
        }
        _put(row, "semantic_compatibility", compatibility)
        _put(row, "relative_compatibility", relative)
        _put(row, "learned_score", score)
        _put(row, "relation_weight", weights)
        row["relation_weight_cv"] = _safe_ratio(row["relation_weight_std"], row["relation_weight_mean"])
        c = normal[f"c_{modality}"]
        _put(row, "local_adaptation", c)
        degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        if nonself.any():
            endpoints = torch.cat((edge_index[0, nonself], edge_index[1, nonself]))
            degree.index_add_(0, endpoints, torch.ones_like(endpoints))
        isolated = degree == 0
        row["isolated_count"] = int(isolated.sum().item())
        row["isolated_c_zero_fraction"] = float((c[isolated] == 0).float().mean().item()) if isolated.any() else math.nan
        row["isolated_c_max_abs"] = float(c[isolated].abs().max().item()) if isolated.any() else math.nan
        row.update(_operator_metrics(model, edge_index, normal[f"normalized_edge_index_{modality}"], normal[f"normalized_edge_weight_{modality}"], num_nodes, dtype))
        rows.append(row)
        values[modality] = normal[f"relation_compatibility_{modality}"][nonself]
    pair = {"dataset": dataset, "seed": seed, "modality": "text_vs_visual", "nonself_edge_count": int(nonself.sum().item())}
    pair.update(_semantic_pair_metrics(values["text"], values["visual"], "semantic_compatibility"))
    text_relative = normal["relation_centered_text"][nonself]
    visual_relative = normal["relation_centered_visual"][nonself]
    pair.update(_semantic_pair_metrics(text_relative, visual_relative, "relative_compatibility"))
    pair.update(_semantic_pair_metrics(normal["a_text"][nonself], normal["a_visual"][nonself], "learned_score"))
    rows.append(pair)
    return rows, values


def _semantic_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, str], dict[str, Any]] = {}
    for modality in MODALITIES:
        p = normal[f"p_{modality}"]
        rho_p = float(getattr(model, f"semantic_rho_p_{modality}").detach().cpu())
        rho_d = float(getattr(model, f"semantic_rho_d_{modality}").detach().cpu())
        bias = getattr(model, f"semantic_bias_{modality}").detach().cpu()
        term_p = rho_p * p
        vectors[(modality, "vector")] = {"term_p": term_p.detach().cpu(), "alpha": [a.detach().cpu() for a in normal[f"alpha_{modality}"]]}
        p_tail = _tail_fractions(p)
        for hop, (change, alpha) in enumerate(zip(normal[f"d_{modality}"], normal[f"alpha_{modality}"], strict=True), start=1):
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "hop": hop,
                "semantic_bias": float(bias[hop - 1]),
                "rho_p": rho_p,
                "rho_d": rho_d,
                "alpha_iqr": _stats(alpha)["q75"] - _stats(alpha)["q25"],
                "alpha_range_ratio": _alpha_range_ratio(alpha),
                "alpha_fraction_lt_0.05": float((alpha < 0.05).float().mean().item()),
                "alpha_fraction_gt_0.95": float((alpha > 0.95).float().mean().item()),
                **{f"p_{key}": value for key, value in _stats(p).items()},
                **{f"term_p_{key}": value for key, value in _stats(term_p).items()},
                **{f"d_{key}": value for key, value in _stats(change).items()},
                **{f"alpha_{key}": value for key, value in _stats(alpha).items()},
                **{f"p_{key}": value for key, value in p_tail.items()},
            }
            row["alpha_q90_q10"] = row["alpha_q90"] - row["alpha_q10"]
            rows.append(row)
    return rows, vectors


def _attention_rows(normal: dict[str, Any], dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    for modality in MODALITIES:
        attention = normal[f"attention_{modality}"]
        mean_matrix, node_std, metrics = attention_diagnostics(attention)
        row = {"dataset": dataset, "seed": seed, "modality": modality, "attention_shape": str(list(attention.shape)), **metrics}
        _put(row, "entrywise_node_std", node_std)
        rows.append(row)
        matrices[modality] = {"shape": list(attention.shape), "mean_matrix": mean_matrix.tolist(), "entrywise_node_std": node_std.tolist(), "metrics": metrics}
    return rows, matrices


def _context_rows(normal: dict[str, Any], model: Any, dataset: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        states = normal[f"S_{modality}"]
        deltas = normal[f"D_{modality}"]
        injections = normal[f"context_change_injection_{modality}"]
        interaction_output = normal[f"interaction_output_{modality}"]
        delta_gate = float(normal[f"delta_gate_{modality}"].item())
        interaction_gate = float(normal[f"interaction_gate_{modality}"].item())
        for hop in range(1, len(states)):
            previous, current = states[hop - 1], states[hop]
            normalized_current = F.layer_norm(current, (current.size(-1),))
            c_delta = injections[hop]
            c_int = interaction_gate * interaction_output[:, hop]
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "delta_gate": delta_gate, "interaction_gate": interaction_gate}
            _put(row, "raw_change_norm", (current - previous).norm(dim=-1))
            _put(row, "relative_raw_change", (current - previous).norm(dim=-1) / (previous.norm(dim=-1) + EPS))
            _put(row, "cosine_current_previous", F.cosine_similarity(current, previous, dim=-1, eps=EPS))
            _put(row, "D_norm", deltas[hop].norm(dim=-1))
            _put(row, "context_injection_norm", c_delta.norm(dim=-1))
            _put(row, "context_injection_ratio", c_delta.norm(dim=-1) / (normalized_current.norm(dim=-1) + EPS))
            _put(row, "context_injection_cosine_to_layernorm_S", F.cosine_similarity(c_delta, normalized_current, dim=-1, eps=EPS))
            _put(row, "interaction_injection_norm", c_int.norm(dim=-1))
            _put(row, "interaction_injection_ratio", c_int.norm(dim=-1) / (current.norm(dim=-1) + EPS))
            _put(row, "interaction_injection_cosine_to_S", F.cosine_similarity(c_int, current, dim=-1, eps=EPS))
            _put(row, "S_tilde_cosine_to_S", F.cosine_similarity(normal[f"S_tilde_{modality}"][hop], current, dim=-1, eps=EPS))
            rows.append(row)
    return rows


def _filter_rows(normal: dict[str, Any], dataset: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    effective_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        gamma = normal[f"gamma_{modality}"]
        delta_gamma = normal[f"delta_gamma_{modality}"]
        content = normal[f"delta_content_{modality}"]
        relation = normal[f"relation_filter_residual_{modality}"]
        eta = normal[f"eta_{modality}"]
        for hop in range(eta.size(1)):
            eta_h = eta[:, hop]
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "gamma": float(gamma[hop]), "delta_gamma": float(delta_gamma[hop])}
            _put(row, "gamma_distribution", gamma[hop])
            _put(row, "delta_gamma_distribution", delta_gamma[hop])
            _put(row, "delta_content", content[:, hop])
            _put(row, "relation_residual", relation[:, hop])
            _put(row, "eta", eta_h)
            row["negative_eta_fraction"] = float((eta_h < 0).float().mean().item())
            row.update({f"content_{key}": value for key, value in _covariance_contribution(content[:, hop], eta_h).items()})
            row.update({f"relation_{key}": value for key, value in _covariance_contribution(relation[:, hop], eta_h).items()})
            row["eta_variance_near_zero"] = int(row["content_variance_near_zero"] or row["relation_variance_near_zero"])
            scale = max(row["delta_content_abs_mean"], abs(row["gamma"]), EPS)
            row["relation_residual_small_amplitude"] = int(row["relation_residual_abs_mean"] < 0.01 * scale)
            rows.append(row)
        effective_rows.append({"dataset": dataset, "seed": seed, "modality": modality, **{f"effective_order_{key}": value for key, value in _stats(normal[f"effective_order_{modality}"]).items()}, **{f"effective_radius_{key}": value for key, value in _stats(normal[f"effective_radius_{modality}"]).items()}})
    return rows, effective_rows


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


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=np.float64)
    return float(np.mean(values)) if values.size else math.nan


def _flags(relation_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]], attention_rows: list[dict[str, Any]], context_rows: list[dict[str, Any]], filter_rows: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for row in relation_rows:
        if row.get("modality") == "text_vs_visual":
            continue
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}"
        if abs(row["beta"] - 0.05) < 0.005:
            flags.append(f"R1_beta_near_init:{tag}")
        if row["relation_weight_cv"] < 1.0e-4:
            flags.append(f"R1_relation_weight_cv_negligible:{tag}")
        if row["operator_weight_relative_l1"] < 1.0e-4:
            flags.append(f"R1_operator_perturbation_negligible:{tag}")
        if row["score_fraction_abs_gt_0.9"] > 0.5:
            flags.append(f"R1_score_saturation:{tag}")
        if row["relation_residual_abs_mean"] < 1.0e-4:
            flags.append(f"relation_residual_small_amplitude:{tag}")
    for row in semantic_rows:
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}/hop{row['hop']}"
        if row["alpha_std"] < 1.0e-5:
            flags.append(f"R2_alpha_node_spread_negligible:{tag}")
        if row["alpha_fraction_lt_0.05"] > 0.5 or row["alpha_fraction_gt_0.95"] > 0.5:
            flags.append(f"R2_alpha_saturation:{tag}")
        if row["p_fraction_abs_gt_3"] > 0.5 or row["p_fraction_abs_gt_5"] > 0.5:
            flags.append(f"R2_p_scale_tail:{tag}")
    for row in attention_rows:
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}"
        if abs(row["attention_nonuniformity"]) < 1.0e-3:
            flags.append(f"R3_attention_near_uniform:{tag}")
        if row["diagonal_excess"] > 0.75:
            flags.append(f"R3_attention_near_diagonal:{tag}")
    for row in context_rows:
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}/hop{row['hop']}"
        if abs(row["delta_gate"]) < 0.005:
            flags.append(f"R3_delta_gate_near_zero:{tag}")
        if abs(row["interaction_gate"]) < 0.005:
            flags.append(f"R3_interaction_gate_near_zero:{tag}")
    for row in filter_rows:
        tag = f"{row['dataset']}/{row['seed']}/{row['modality']}/hop{row['hop']}"
        if row["eta_variance_near_zero"]:
            flags.append(f"R3_eta_variance_near_zero:{tag}")
        if row["relation_residual_small_amplitude"]:
            flags.append(f"relation_residual_small_amplitude:{tag}")
    return sorted(set(flags))


def _report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# SSI-MAG-V3.1 P1.7b Post-hoc Diagnostic Report",
        "",
        "This report is generated only from validation-selected Full V3.1 NC checkpoints in evaluation/no-grad mode. It does not train, select checkpoints, read test metrics for a decision, run LP, run ablations, or tune hyperparameters.",
        "",
        "## Integrity",
        "",
        f"- Expected/loaded/finite runs: {summary['expected_runs']}/{summary['loaded_runs']}/{summary['finite_runs']}",
        f"- Node ordering verified by dataset: `{summary['node_ordering_verified']}`",
        f"- Attention definition source: imported P1.5a corrected `attention_diagnostics()`; shapes: `{summary['attention_shapes']}`",
        f"- Device: `{summary['device']}`",
        "",
        "## Metric interpretation",
        "",
        "See `docs/ssi_mag_v31_diagnostic_spec.md`. All relation/operator, alpha, entropy, eta-sign, and effective-order quantities are descriptive/pathology checks. No direction such as larger perturbation, larger alpha variance, lower entropy, or more negative eta is treated as inherently better.",
        "",
        "## Diagnostic groups",
        "",
        f"- R1 rows: `{summary['relation_rows']}`; R2 rows: `{summary['semantic_rows']}`; attention rows: `{summary['attention_rows']}`; context rows: `{summary['context_rows']}`; filter rows: `{summary['filter_rows']}`.",
        f"- Cross-seed rows: `{summary['cross_seed_rows']}`; ordering failures: `{summary['ordering_failures']}`.",
        f"- Automatic flags: `{len(summary['flags'])}`. Flags are reported observations, not architecture verdicts.",
        "",
        "## Boundary",
        "",
        "No formal P1.7b benchmark, LP, ablation, auxiliary loss, test-based decision, or architecture modification is performed by this analyzer.",
    ]
    (output / "p17b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/ssi_mag_v31_p17b_full"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssi_mag_v31_p17b_analysis"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    input_root = args.input_root if args.input_root.is_absolute() else project_root / args.input_root
    output = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    relation_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    effective_rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, int, str], dict[str, Any]] = {}
    signatures: dict[str, dict[int, tuple[Any, ...]]] = defaultdict(dict)
    matrices: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    errors: list[dict[str, Any]] = []
    loaded = 0
    finite_runs = 0
    for dataset in args.datasets:
        for seed in args.seeds:
            try:
                run_dir, cfg, data, model, head = _load_run(input_root, dataset, seed, device)
                del cfg, head
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                with torch.no_grad():
                    normal = model.analysis(x, edge_index)
                finite = _finite_analysis(normal)
                loaded += 1
                finite_runs += int(finite)
                signatures[dataset][seed] = (data.num_nodes, _tensor_hash(data.x), _tensor_hash(data.edge_index), _tensor_hash(data.y) if data.y is not None else None)
                relation_part, _ = _relation_rows(normal, model, dataset, seed, data.num_nodes, x.dtype)
                semantic_part, semantic_vectors = _semantic_rows(normal, model, dataset, seed)
                attention_part, attention_matrix = _attention_rows(normal, dataset, seed)
                context_part = _context_rows(normal, model, dataset, seed)
                filter_part, effective_part = _filter_rows(normal, dataset, seed)
                relation_rows.extend(relation_part)
                semantic_rows.extend(semantic_part)
                attention_rows.extend(attention_part)
                context_rows.extend(context_part)
                filter_rows.extend(filter_part)
                effective_rows.extend(effective_part)
                for modality in MODALITIES:
                    vectors[(dataset, seed, modality)] = semantic_vectors[(modality, "vector")]
                    matrices[dataset].setdefault(modality, {})[str(seed)] = attention_matrix[modality]
                del normal, model, data, x, edge_index
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"OK {dataset} seed={seed} finite={finite}", flush=True)
            except Exception as exc:
                errors.append({"dataset": dataset, "seed": seed, "error": repr(exc)})
                print(f"ERROR {dataset} seed={seed}: {exc}", flush=True)

    dataset_tuple = tuple(args.datasets)
    seed_tuple = tuple(args.seeds)
    cross_seed_rows, ordering = _cross_seed_rows(vectors, signatures, dataset_tuple, seed_tuple)
    flags = _flags(relation_rows, semantic_rows, attention_rows, context_rows, filter_rows)
    ordering_failures = sorted(dataset for dataset, verified in ordering.items() if not verified)
    summary = {
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "expected_runs": len(args.datasets) * len(args.seeds),
        "loaded_runs": loaded,
        "finite_runs": finite_runs,
        "errors": errors,
        "device": str(device),
        "input_root": str(input_root),
        "node_ordering_verified": ordering,
        "ordering_failures": ordering_failures,
        "attention_shapes": sorted({row["attention_shape"] for row in attention_rows}),
        "relation_rows": len(relation_rows),
        "semantic_rows": len(semantic_rows),
        "attention_rows": len(attention_rows),
        "context_rows": len(context_rows),
        "filter_rows": len(filter_rows),
        "cross_seed_rows": len(cross_seed_rows),
        "flags": flags,
        "test_metrics_read": False,
        "training_invoked": False,
        "ablation_invoked": False,
        "lp_invoked": False,
    }
    _write_csv(output / "p17b_relation_diagnostics.csv", relation_rows)
    _write_csv(output / "p17b_semantic_diagnostics.csv", semantic_rows)
    _write_csv(output / "p17b_attention_diagnostics.csv", attention_rows)
    _write_csv(output / "p17b_context_diagnostics.csv", context_rows)
    _write_csv(output / "p17b_filter_diagnostics.csv", filter_rows)
    _write_csv(output / "p17b_effective_order.csv", effective_rows)
    _write_csv(output / "p17b_cross_seed_consistency.csv", cross_seed_rows)
    _write_csv(output / "p17b_flags.csv", [{"flag": flag} for flag in flags])
    (output / "p17b_attention_matrices.json").write_text(json.dumps(matrices, indent=2, default=_json_default), encoding="utf-8")
    (output / "p17b_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    _report(output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if loaded == summary["expected_runs"] and finite_runs == loaded and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
