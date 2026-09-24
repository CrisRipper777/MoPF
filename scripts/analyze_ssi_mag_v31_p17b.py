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
import subprocess
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
from sklearn.metrics import f1_score

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


def _validate_attention_simplex(
    attention: torch.Tensor,
    *,
    min_value_tolerance: float = -1.0e-7,
    row_sum_tolerance: float = 1.0e-5,
) -> dict[str, float]:
    """Reject malformed attention before downstream attention statistics."""
    if attention.ndim != 3:
        raise ValueError(f"attention must be rank-3, got shape={tuple(attention.shape)}")
    if not bool(torch.isfinite(attention).all().item()):
        raise ValueError("attention contains NaN/Inf")
    minimum = float(attention.min().item()) if attention.numel() else 0.0
    if minimum < min_value_tolerance:
        raise ValueError(f"attention has negative mass below tolerance: min={minimum}")
    row_sums = attention.sum(dim=-1)
    max_row_error = float((row_sums - 1.0).abs().max().item()) if row_sums.numel() else 0.0
    if max_row_error >= row_sum_tolerance:
        raise ValueError(f"attention rows are not simplex-normalized: max_error={max_row_error}")
    return {"attention_min": minimum, "attention_max_row_sum_error": max_row_error}


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
        scorer = getattr(model, f"relation_scorer_{modality}")
        coefficients = scorer.weight.detach().reshape(-1).cpu()
        bias = float(scorer.bias.detach().reshape(-1)[0].cpu())
        term_s = coefficients[0].to(compatibility.device) * compatibility
        term_r = coefficients[1].to(compatibility.device) * relative
        term_abs = coefficients[2].to(compatibility.device) * relative.abs()
        dynamic_logit = term_s + term_r + term_abs
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "modality": modality,
            "nonself_edge_count": int(nonself.sum().item()),
            "beta": float(normal[f"beta_parameter_{modality}"].item()),
            "scorer_w_s": float(coefficients[0]),
            "scorer_w_r": float(coefficients[1]),
            "scorer_w_abs": float(coefficients[2]),
            "scorer_bias": bias,
            "score_fraction_abs_gt_0.8": float((score.abs() > 0.8).float().mean().item()) if score.numel() else math.nan,
            "score_fraction_abs_gt_0.9": float((score.abs() > 0.9).float().mean().item()) if score.numel() else math.nan,
        }
        _put(row, "semantic_compatibility", compatibility)
        _put(row, "relative_compatibility", relative)
        _put(row, "learned_score", score)
        _put(row, "scorer_term_s", term_s)
        _put(row, "scorer_term_r", term_r)
        _put(row, "scorer_term_abs", term_abs)
        _put(row, "dynamic_logit", dynamic_logit)
        row["dynamic_logit_std_over_bias_abs"] = _safe_ratio(row["dynamic_logit_std"], abs(bias))
        row["dynamic_logit_abs_mean_over_bias_abs"] = _safe_ratio(row["dynamic_logit_abs_mean"], abs(bias))
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
        simplex = _validate_attention_simplex(attention)
        mean_matrix, node_std, metrics = attention_diagnostics(attention)
        row = {"dataset": dataset, "seed": seed, "modality": modality, "attention_shape": str(list(attention.shape)), **simplex, **metrics}
        _put(row, "entrywise_node_std", node_std)
        rows.append(row)
        matrices[modality] = {"shape": list(attention.shape), "mean_matrix": mean_matrix.tolist(), "entrywise_node_std": node_std.tolist(), "simplex": simplex, "metrics": metrics}
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


def _functional_sensitivity(
    normal: dict[str, Any],
    relation_off: dict[str, Any],
    interaction_off: dict[str, Any],
    head: nn.Module,
    data: Any,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    """Frozen representation/prediction sensitivity; never retrains a head."""
    row: dict[str, Any] = {"dataset": dataset, "seed": seed}
    for tag, changed in (("relation_off", relation_off), ("interaction_off", interaction_off)):
        for source, label in (("z", "fused_embedding"), ("z_text", "text_embedding"), ("z_visual", "visual_embedding")):
            base = normal[source].float()
            other = changed[source].float()
            difference = other - base
            row[f"{tag}_{label}_mae"] = float(difference.abs().mean().item())
            row[f"{tag}_{label}_relative_l2"] = float(
                difference.norm().item() / (base.norm().item() + EPS)
            )
            row[f"{tag}_{label}_cosine"] = float(
                F.cosine_similarity(base, other, dim=-1, eps=EPS).mean().item()
            )
        normal_logits = head(normal["z"])
        changed_logits = head(changed["z"])
        normal_pred = normal_logits.argmax(dim=-1)
        changed_pred = changed_logits.argmax(dim=-1)
        row[f"{tag}_prediction_flip_rate"] = float(
            (normal_pred != changed_pred).float().mean().item()
        )
        val_idx = data.val_idx.to(normal_logits.device)
        target = data.y[val_idx].detach().cpu().numpy()
        labels = list(range(int(data.num_classes)))
        before = normal_logits[val_idx].argmax(dim=-1).detach().cpu().numpy()
        after = changed_logits[val_idx].argmax(dim=-1).detach().cpu().numpy()
        row[f"{tag}_val_acc_delta"] = float((after == target).mean() - (before == target).mean())
        row[f"{tag}_val_macro_f1_delta"] = float(
            f1_score(target, after, labels=labels, average="macro", zero_division=0)
            - f1_score(target, before, labels=labels, average="macro", zero_division=0)
        )
    return row


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


def _performance_rows(input_root: Path, datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            run_dir = _checkpoint_run(input_root, dataset, seed)
            results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "model": "ssi_mag_v31", "ablation": "full"}
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                row[key] = float(results[key]["mean"])
            row["best_epoch"] = int(metrics.get("best_epoch")) if metrics.get("best_epoch") is not None else math.nan
            row["runtime_seconds"] = float(metrics.get("runtime_seconds", math.nan))
            row["peak_gpu_memory_mib"] = float(metrics.get("peak_gpu_memory_mib", math.nan))
            log_path = run_dir / "train.log"
            log_text = log_path.read_text(errors="ignore") if log_path.is_file() else ""
            epochs = [int(value) for value in __import__("re").findall(r"Epoch\s+(\d+)", log_text)]
            row["total_trained_epochs"] = max(epochs) if epochs else math.nan
            row["nan_inf_detected"] = int(not all(math.isfinite(float(row[key])) for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")))
            rows.append(row)
    return rows


def _performance_summary(rows: list[dict[str, Any]], datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in (*datasets, "ALL"):
        items = rows if dataset == "ALL" else [row for row in rows if row["dataset"] == dataset]
        result: dict[str, Any] = {"dataset": dataset, "n_runs": len(items)}
        for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1", "best_epoch", "total_trained_epochs", "runtime_seconds", "peak_gpu_memory_mib", "nan_inf_detected"):
            values = np.asarray([float(row[key]) for row in items if key in row and math.isfinite(float(row[key]))], dtype=np.float64)
            if values.size:
                result[f"{key}_mean"] = float(values.mean())
                result[f"{key}_population_std"] = float(values.std())
                result[f"{key}_values"] = ";".join(f"{value:.12g}" for value in values)
        output.append(result)
    return output


def _load_v3_reference() -> tuple[list[dict[str, Any]], str]:
    """Load only provenance-verified historical V3 Full NC runs."""
    root = ROOT / "outputs/ssi_mag_v3_p1_full"
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            run_dir = root / dataset / "full" / f"seed{seed}"
            try:
                marker = json.loads((run_dir / "complete.marker").read_text(encoding="utf-8"))
                cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
                results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
                task = cfg["task"]
                dataset_cfg = cfg["dataset"]
                split_path = str(dataset_cfg.get("nc_split_path") or dataset_cfg.get("node_split_path") or "")
                split_matches = (
                    split_path.endswith(f"seed{seed}_train0.6_val0.2.pt")
                    or (dataset == "ele-fashion" and split_path.endswith("ele-fashion/split.pt"))
                )
                if not (
                    marker.get("status") == "complete"
                    and marker.get("task") == "nc"
                    and marker.get("dataset") == dataset
                    and int(marker.get("seed", -1)) == seed
                    and cfg["model"]["name"] == "ssi_mag_v3"
                    and cfg["ablation"] == "full"
                    and task["name"] == "nc"
                    and task["protocol_version"] == "unified_full_graph_nc_v1"
                    and task["training_mode"] == "full_graph"
                    and int(cfg.get("num_runs", 0)) == 1
                    and int(cfg["seed"]) == seed
                    and split_matches
                ):
                    raise ValueError("identity/protocol/split mismatch")
                row = {"dataset": dataset, "seed": seed, "model": "ssi_mag_v3", "ablation": "full"}
                for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                    row[key] = float(results[key]["mean"])
                rows.append(row)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return [], f"unavailable: provenance checks failed for {dataset}/seed{seed}: {exc}"
    return rows, "verified: matching datasets/seeds, NC, full-graph, unified_full_graph_nc_v1, validation-Accuracy checkpoint selection"


def _paired_performance_rows(v31_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v3_rows, status = _load_v3_reference()
    if not v3_rows:
        return [{"row_type": "status", "source_status": status}], {"status": status, "available": False}
    v3 = {(row["dataset"], int(row["seed"])): row for row in v3_rows}
    paired: list[dict[str, Any]] = []
    metric_keys = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    for row in v31_rows:
        ref = v3[(row["dataset"], int(row["seed"]))]
        out: dict[str, Any] = {"row_type": "paired_seed", "dataset": row["dataset"], "seed": row["seed"], "source_status": status}
        for key in metric_keys:
            out[f"v31_{key}"] = row[key]
            out[f"v3_{key}"] = ref[key]
            out[f"delta_{key}"] = row[key] - ref[key]
        paired.append(out)
    for dataset in (*DATASETS, "ALL"):
        items = v31_rows if dataset == "ALL" else [row for row in v31_rows if row["dataset"] == dataset]
        refs = v3_rows if dataset == "ALL" else [row for row in v3_rows if row["dataset"] == dataset]
        out = {"row_type": "dataset_summary", "dataset": dataset, "seed": "", "source_status": status}
        for key in metric_keys:
            a = np.asarray([row[key] for row in items], dtype=np.float64)
            b = np.asarray([row[key] for row in refs], dtype=np.float64)
            out[f"v31_{key}_mean"] = float(a.mean())
            out[f"v31_{key}_population_std"] = float(a.std())
            out[f"v3_{key}_mean"] = float(b.mean())
            out[f"v3_{key}_population_std"] = float(b.std())
            out[f"delta_{key}_mean"] = float(a.mean() - b.mean())
            out[f"delta_{key}_values"] = ";".join(f"{value:.12g}" for value in (a - b))
        out["v31_better_val_acc_count"] = int(sum(a["val_acc"] > b["val_acc"] for a, b in zip(items, refs)))
        out["v3_better_val_acc_count"] = int(sum(a["val_acc"] < b["val_acc"] for a, b in zip(items, refs)))
        out["val_acc_tie_count"] = int(sum(a["val_acc"] == b["val_acc"] for a, b in zip(items, refs)))
        out["v31_better_val_macro_f1_count"] = int(sum(a["val_macro_f1"] > b["val_macro_f1"] for a, b in zip(items, refs)))
        out["v3_better_val_macro_f1_count"] = int(sum(a["val_macro_f1"] < b["val_macro_f1"] for a, b in zip(items, refs)))
        out["val_macro_f1_tie_count"] = int(sum(a["val_macro_f1"] == b["val_macro_f1"] for a, b in zip(items, refs)))
        paired.append(out)
    return paired, {"status": status, "available": True}


def _numeric_mean(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows if key in row and _is_finite_number(row[key])], dtype=np.float64)
    return (float(values.mean()), float(values.std())) if values.size else (math.nan, math.nan)


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mechanism_comparison(
    relation_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    filter_rows: list[dict[str, Any]],
    effective_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    old_path = ROOT / "outputs/ssi_mag_v3_p1_analysis/p1_mechanism_per_run.csv"
    if not old_path.is_file():
        return [{"status": "unavailable: V3 mechanism summary missing"}]
    with old_path.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    if len(old_rows) != len(DATASETS) * len(SEEDS):
        return [{"status": "unavailable: V3 mechanism run count mismatch"}]
    old_by_key = {(row.get("dataset"), int(row["seed"])): row for row in old_rows}
    rows: list[dict[str, Any]] = []
    def add(quantity: str, dataset: str, modality: str, hop: int | str, new_rows: list[dict[str, Any]], new_key: str, old_key: str, note: str) -> None:
        selected_new = [row for row in new_rows if row.get("dataset") == dataset and row.get("modality") == modality and (hop == "" or str(row.get("hop")) == str(hop))]
        selected_old = [row for row in old_rows if row.get("dataset") == dataset]
        new_mean, new_std = _numeric_mean(selected_new, new_key)
        old_values = [float(row[old_key]) for row in selected_old if old_key in row and _is_finite_number(row[old_key])]
        old_mean = float(np.mean(old_values)) if old_values else math.nan
        old_std = float(np.std(old_values)) if old_values else math.nan
        rows.append({"quantity": quantity, "dataset": dataset, "modality": modality, "hop": hop, "v31_mean": new_mean, "v31_population_std": new_std, "v3_mean": old_mean, "v3_population_std": old_std, "note": note})
    for dataset in DATASETS:
        for modality in MODALITIES:
            add("normalized_operator_relative_l1", dataset, modality, "", relation_rows, "operator_weight_relative_l1", f"operator_weight_relative_l1_{modality}", "same physical support and gcn_norm; descriptive")
            add("beta", dataset, modality, "", relation_rows, "beta", f"beta_{modality}", "scorer definition changed; beta alone is not a utility ranking")
            add("local_adaptation_c_mean", dataset, modality, "", relation_rows, "local_adaptation_mean", f"local_adaptation_{modality}_mean", "same c_i definition family")
            for hop in range(1, 4):
                add("alpha_mean", dataset, modality, hop, semantic_rows, "alpha_mean", f"alpha_{modality}_k{hop}_mean", "same semantic-reference quantity")
                add("alpha_std", dataset, modality, hop, semantic_rows, "alpha_std", f"alpha_{modality}_k{hop}_std", "same semantic-reference quantity")
                add("eta_negative_fraction", dataset, modality, hop - 1, filter_rows, "negative_eta_fraction", f"eta_{modality}_k{hop-1}_negative_fraction", "signed filtering behavior")
            add("attention_normalized_entropy", dataset, modality, "", attention_rows, "nodewise_normalized_entropy", f"attention_{modality}_normalized_entropy", "P1.5a corrected nodewise metric")
            add("attention_node_heterogeneity", dataset, modality, "", attention_rows, "node_heterogeneity_mean", f"attention_{modality}_row_diversity_l1", "V3 column is older row-diversity proxy; interpret descriptively")
            add("context_change_gate", dataset, modality, "", context_rows, "delta_gate", f"delta_gate_{modality}", "learned gate")
            add("cross_hop_interaction_gate", dataset, modality, "", context_rows, "interaction_gate", f"interaction_gate_{modality}", "learned gate")
            add("effective_order_mean", dataset, modality, "", effective_rows, "effective_order_mean", f"effective_order_{modality}_mean", "signed effective order")
            add("relation_off_fused_relative_l2", dataset, modality, "", sensitivity_rows, "relation_off_fused_embedding_relative_l2", "relation_off_z_relative_l2", "frozen sensitivity; V3.1 row is duplicated by modality")
            add("interaction_off_fused_relative_l2", dataset, modality, "", sensitivity_rows, "interaction_off_fused_embedding_relative_l2", "interaction_off_z_relative_l2", "frozen sensitivity; V3.1 row is duplicated by modality")
    return rows


def _guardrail(paired: list[dict[str, Any]], comparison_info: dict[str, Any]) -> dict[str, Any]:
    if not comparison_info.get("available"):
        return {"status": "REVIEW_REQUIRED", "reason": "paired V3 provenance unavailable; validation guardrail cannot be evaluated"}
    summaries = {row.get("dataset"): row for row in paired if row.get("row_type") == "dataset_summary" and row.get("dataset") in DATASETS}
    acc_deltas = [float(summaries[d]["delta_val_acc_mean"]) for d in DATASETS]
    f1_deltas = [float(summaries[d]["delta_val_macro_f1_mean"]) for d in DATASETS]
    mean_acc = float(np.mean(acc_deltas))
    mean_f1 = float(np.mean(f1_deltas))
    simultaneous_declines = [d for d in DATASETS if summaries[d]["delta_val_acc_mean"] < 0.0 and summaries[d]["delta_val_macro_f1_mean"] < 0.0]
    passed = mean_acc >= -0.005 and mean_f1 >= -0.005 and len(simultaneous_declines) < 3
    return {
        "status": "PASS_GUARDRAIL" if passed else "REVIEW_REQUIRED",
        "mean_validation_accuracy_delta": mean_acc,
        "mean_validation_macro_f1_delta": mean_f1,
        "mean_validation_accuracy_delta_pp": 100.0 * mean_acc,
        "mean_validation_macro_f1_delta_pp": 100.0 * mean_f1,
        "simultaneous_validation_declines": simultaneous_declines,
        "simultaneous_decline_count": len(simultaneous_declines),
        "threshold_pp": -0.5,
        "test_used_for_guardrail": False,
    }


def _write_decision_packet(summary: dict[str, Any]) -> None:
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    guard = summary.get("guardrail", {})
    lines = [
        "# SSI-MAG-V3.1 P1.7b Decision Packet",
        "",
        f"Decision status: **{guard.get('status', 'REVIEW_REQUIRED')}**",
        "",
        "This packet separates performance evidence, mechanism behavior evidence, and frozen functional sensitivity. It does not choose a final paper model.",
        "",
        "## 1. Performance evidence",
        "",
        f"- Formal Full NC runs: {summary.get('loaded_runs', 0)}/{summary.get('expected_runs', 0)} loaded; finite: {summary.get('finite_runs', 0)}.",
        "- Checkpoint selection is validation Accuracy; test metrics are final descriptive outputs only.",
        "",
        "## 2. Seed stability",
        "",
        f"- Cross-seed ordering failures: `{summary.get('ordering_failures', [])}`.",
        "- See `p17b_performance_summary.csv` for population standard deviations.",
        "",
        "## 3–5. Mechanism behavior",
        "",
        "- R1/R2/Stage-II quantities are reported in the diagnostic CSVs, including scorer dynamic-logit attribution and attention simplex validation.",
        "- Flags are descriptive review triggers, not automatic mechanism failures.",
        "",
        "## 6. Frozen functional sensitivity",
        "",
        "- `relation=off` and `interaction=off` use unchanged model parameters and the same saved NC head; these are sensitivity diagnostics, not retrained causal ablations.",
        "",
        "## 7. V3 vs V3.1 comparison",
        "",
        f"- Paired performance provenance: `{summary.get('comparison_status', 'unavailable')}`.",
        "- Test deltas are descriptive and were not used for guardrail decisions.",
        "",
        "## 8. Guardrail status",
        "",
        f"- `{guard.get('status', 'REVIEW_REQUIRED')}`.",
        f"- Validation Accuracy mean delta: `{guard.get('mean_validation_accuracy_delta_pp', 'NA')} pp`; Validation Macro-F1 mean delta: `{guard.get('mean_validation_macro_f1_delta_pp', 'NA')} pp`.",
        f"- Simultaneous validation declines: `{guard.get('simultaneous_validation_declines', [])}`.",
        "",
        "## 9. Open questions",
        "",
        "- Any saturation, collapse, or negligible-effect flag requires human audit and is not repaired here.",
        "- No LP, retrained ablation, hyperparameter search, auxiliary loss, or test-based architecture selection was run.",
    ]
    (docs / "ssi_mag_v31_p17b_decision_packet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if guard.get("status") == "PASS_GUARDRAIL":
        plan = [
            "# SSI-MAG-V3.1 P1.7c Control Plan (not executed)",
            "",
            "This is a planning artifact only. No controls are implemented, trained, or launched in P1.7b.",
            "",
            "- A: R1 revision only",
            "- B: R2 stabilization / rho_c removal only",
            "- C: Stage-II reference residual removal only",
            "- AB: A+B",
            "- Historical V3 is the reference; V3.1 Full is the ABC reference.",
        ]
        (docs / "ssi_mag_v31_p17c_control_plan.md").write_text("\n".join(plan) + "\n", encoding="utf-8")


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
            flags.append(f"R1_operator_perturbation_near_zero:{tag}")
        if row["score_fraction_abs_gt_0.9"] > 0.5:
            flags.append(f"R1_score_saturation:{tag}")
        if row["learned_score_abs_mean"] < 1.0e-4:
            flags.append(f"R1_relation_score_small_amplitude:{tag}")
        if row["dynamic_logit_std"] < 1.0e-4:
            flags.append(f"R1_dynamic_logit_negligible:{tag}")
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
        if row["diagonal_mass_mean"] > 1.0 - 1.0e-3:
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
            flags.append(f"R3_relation_residual_small_amplitude:{tag}")
    return sorted(set(flags))


def _report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# SSI-MAG-V3.1 P1.7b Post-hoc Diagnostic Report",
        "",
        "This report uses only validation-selected Full V3.1 NC checkpoints in evaluation/no-grad mode. It does not train, run LP, run retrained ablations, tune hyperparameters, add losses, or use test metrics for a model decision.",
        "",
        "## Integrity",
        "",
        f"- Expected/loaded/finite runs: {summary['expected_runs']}/{summary['loaded_runs']}/{summary['finite_runs']}",
        f"- Provenance lock: `{summary.get('provenance_lock', 'missing')}`",
        f"- Node ordering verified by dataset: `{summary['node_ordering_verified']}`",
        f"- Attention simplex validation failures: `{summary.get('attention_simplex_failures', [])}`",
        f"- Device: `{summary['device']}` (CPU is the default analysis device)",
        "",
        "## Performance evidence",
        "",
        "See `p17b_performance_summary.csv` for validation/test Accuracy and Macro-F1, population standard deviations, best epoch, runtime, and peak GPU memory. Test metrics are descriptive final metrics from the validation-selected checkpoint.",
        "",
        "## Mechanism behavior evidence",
        "",
        f"- R1 rows: `{summary['relation_rows']}`; R2 rows: `{summary['semantic_rows']}`; attention rows: `{summary['attention_rows']}`; context rows: `{summary['context_rows']}`; filter rows: `{summary['filter_rows']}`.",
        f"- Cross-seed rows: `{summary['cross_seed_rows']}`; ordering failures: `{summary['ordering_failures']}`.",
        f"- Automatic flags: `{len(summary['flags'])}`. These are declared pathology/review triggers, not verdicts.",
        "- R1 scorer decomposition reports the linear terms `w_s*s`, `w_r*rrel`, `w_abs*abs(rrel)`, their dynamic logit, and the softsign output on non-self physical edges.",
        "- Attention row diversity is the mean pairwise L1/L2 distance between different query rows for the same node over all unordered row pairs.",
        "",
        "## Frozen functional sensitivity",
        "",
        "`relation=off` and `interaction=off` keep all parameters and the saved NC head fixed. Their embedding and validation prediction changes are sensitivity evidence, not retrained causal ablations.",
        "",
        "## V3 versus V3.1",
        "",
        f"- Paired performance provenance: `{summary.get('comparison_status', 'unavailable')}`.",
        "- `p17b_v31_vs_v3_performance.csv` contains per-seed paired validation/test deltas and dataset summaries.",
        "- `p17b_v31_vs_v3_mechanism.csv` compares only defined comparable quantities; old V3 scorer values are not ranked against new V3.1 scorer values as the same scale.",
        "",
        "## Guardrail",
        "",
        f"- Status: **{summary.get('guardrail', {}).get('status', 'REVIEW_REQUIRED')}**",
        f"- Validation Accuracy mean delta: `{summary.get('guardrail', {}).get('mean_validation_accuracy_delta_pp', 'NA')} pp`.",
        f"- Validation Macro-F1 mean delta: `{summary.get('guardrail', {}).get('mean_validation_macro_f1_delta_pp', 'NA')} pp`.",
        f"- Simultaneous validation declines: `{summary.get('guardrail', {}).get('simultaneous_validation_declines', [])}`.",
        "- This guardrail uses validation evidence and mechanism evidence only; test performance cannot trigger it.",
        "",
        "## Boundary",
        "",
        "No final paper-model decision or unvalidated repair is made. No LP jobs, retrained ablations, hyperparameter search, auxiliary loss, or test-based architecture selection were run.",
    ]
    (output / "p17b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/ssi_mag_v31_p17b_full"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssi_mag_v31_p17b_analysis"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    input_root = args.input_root if args.input_root.is_absolute() else project_root / args.input_root
    output = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested analysis device is unavailable: {device}")
    dataset_tuple = tuple(args.datasets)
    seed_tuple = tuple(args.seeds)
    lock_path = input_root / "provenance.lock.json"
    if not lock_path.is_file():
        raise SystemExit(f"refusing analysis without formal provenance lock: {lock_path}")
    try:
        provenance = json.loads(lock_path.read_text(encoding="utf-8"))
        if not (
            provenance.get("task") == "nc"
            and provenance.get("model") == "ssi_mag_v31"
            and provenance.get("variant") == "full"
            and tuple(provenance.get("datasets", [])) == dataset_tuple
            and tuple(int(seed) for seed in provenance.get("seeds", [])) == seed_tuple
        ):
            raise ValueError("provenance lock does not match requested formal NC set")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid provenance lock: {exc}") from exc

    relation_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    effective_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, int, str], dict[str, Any]] = {}
    signatures: dict[str, dict[int, tuple[Any, ...]]] = defaultdict(dict)
    matrices: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    errors: list[dict[str, Any]] = []
    attention_simplex_failures: list[str] = []
    loaded = 0
    finite_runs = 0
    for dataset in dataset_tuple:
        for seed in seed_tuple:
            try:
                run_dir, cfg, data, model, head = _load_run(input_root, dataset, seed, device)
                del cfg, run_dir
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                with torch.no_grad():
                    normal = model.analysis(x, edge_index)
                    relation_off = model.analysis_intervention(x, edge_index, relation="off")
                    interaction_off = model.analysis_intervention(x, edge_index, interaction="off")
                finite = _finite_analysis(normal) and _finite_analysis(relation_off) and _finite_analysis(interaction_off)
                if not finite:
                    raise ValueError("normal or intervention analysis contains NaN/Inf")
                for modality in MODALITIES:
                    try:
                        _validate_attention_simplex(normal[f"attention_{modality}"])
                    except ValueError as exc:
                        attention_simplex_failures.append(f"{dataset}/seed{seed}/{modality}: {exc}")
                        raise
                loaded += 1
                finite_runs += int(finite)
                signatures[dataset][seed] = (data.num_nodes, _tensor_hash(data.x), _tensor_hash(data.edge_index), _tensor_hash(data.y) if data.y is not None else None)
                relation_part, _ = _relation_rows(normal, model, dataset, seed, data.num_nodes, x.dtype)
                semantic_part, semantic_vectors = _semantic_rows(normal, model, dataset, seed)
                attention_part, attention_matrix = _attention_rows(normal, dataset, seed)
                context_part = _context_rows(normal, model, dataset, seed)
                filter_part, effective_part = _filter_rows(normal, dataset, seed)
                sensitivity_part = _functional_sensitivity(normal, relation_off, interaction_off, head, data, dataset, seed)
                relation_rows.extend(relation_part)
                semantic_rows.extend(semantic_part)
                attention_rows.extend(attention_part)
                context_rows.extend(context_part)
                filter_rows.extend(filter_part)
                effective_rows.extend(effective_part)
                sensitivity_rows.append(sensitivity_part)
                for modality in MODALITIES:
                    vectors[(dataset, seed, modality)] = semantic_vectors[(modality, "vector")]
                    matrices[dataset].setdefault(modality, {})[str(seed)] = attention_matrix[modality]
                del normal, relation_off, interaction_off, model, head, data, x, edge_index
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"OK {dataset} seed={seed} finite={finite}", flush=True)
            except Exception as exc:
                errors.append({"dataset": dataset, "seed": seed, "error": repr(exc)})
                print(f"ERROR {dataset} seed={seed}: {exc}", flush=True)

    cross_seed_rows, ordering = _cross_seed_rows(vectors, signatures, dataset_tuple, seed_tuple)
    flags = _flags(relation_rows, semantic_rows, attention_rows, context_rows, filter_rows)
    ordering_failures = sorted(dataset for dataset, verified in ordering.items() if not verified)
    try:
        performance_run_rows = _performance_rows(input_root, dataset_tuple, seed_tuple)
        performance_summary = _performance_summary(performance_run_rows, dataset_tuple)
    except Exception as exc:
        errors.append({"performance": repr(exc)})
        performance_run_rows = []
        performance_summary = []
    paired_performance, comparison_info = _paired_performance_rows(performance_run_rows) if performance_run_rows else ([{"row_type": "status", "source_status": "unavailable: V3.1 performance rows missing"}], {"available": False, "status": "unavailable: V3.1 performance rows missing"})
    mechanism_comparison = _mechanism_comparison(relation_rows, semantic_rows, attention_rows, context_rows, filter_rows, effective_rows, sensitivity_rows)
    guardrail = _guardrail(paired_performance, comparison_info)
    summary = {
        "datasets": list(dataset_tuple),
        "seeds": list(seed_tuple),
        "expected_runs": len(dataset_tuple) * len(seed_tuple),
        "loaded_runs": loaded,
        "finite_runs": finite_runs,
        "errors": errors,
        "device": str(device),
        "input_root": str(input_root),
        "provenance_lock": str(lock_path),
        "node_ordering_verified": ordering,
        "ordering_failures": ordering_failures,
        "attention_simplex_failures": attention_simplex_failures,
        "attention_shapes": sorted({row["attention_shape"] for row in attention_rows}),
        "relation_rows": len(relation_rows),
        "semantic_rows": len(semantic_rows),
        "attention_rows": len(attention_rows),
        "context_rows": len(context_rows),
        "filter_rows": len(filter_rows),
        "cross_seed_rows": len(cross_seed_rows),
        "sensitivity_rows": len(sensitivity_rows),
        "performance_rows": len(performance_run_rows),
        "performance_summary_rows": len(performance_summary),
        "flags": flags,
        "comparison_status": comparison_info.get("status", "unavailable"),
        "comparison_available": bool(comparison_info.get("available", False)),
        "guardrail": guardrail,
        "test_metrics_used_for_selection_or_decision": False,
        "training_invoked": False,
        "ablation_invoked": False,
        "lp_invoked": False,
    }
    _write_csv(output / "p17b_performance_run.csv", performance_run_rows)
    _write_csv(output / "p17b_performance_summary.csv", performance_summary)
    _write_csv(output / "p17b_relation_diagnostics.csv", relation_rows)
    _write_csv(output / "p17b_semantic_diagnostics.csv", semantic_rows)
    _write_csv(output / "p17b_attention_diagnostics.csv", attention_rows)
    _write_csv(output / "p17b_context_diagnostics.csv", context_rows)
    _write_csv(output / "p17b_filter_diagnostics.csv", filter_rows)
    _write_csv(output / "p17b_effective_order.csv", effective_rows)
    _write_csv(output / "p17b_cross_seed_consistency.csv", cross_seed_rows)
    _write_csv(output / "p17b_intervention_sensitivity.csv", sensitivity_rows)
    _write_csv(output / "p17b_v31_vs_v3_performance.csv", paired_performance)
    _write_csv(output / "p17b_v31_vs_v3_mechanism.csv", mechanism_comparison)
    _write_csv(output / "p17b_flags.csv", [{"flag": flag} for flag in flags])
    (output / "p17b_attention_matrices.json").write_text(json.dumps(matrices, indent=2, default=_json_default), encoding="utf-8")
    (output / "p17b_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    _report(output, summary)
    _write_decision_packet(summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if loaded == summary["expected_runs"] and finite_runs == loaded and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
