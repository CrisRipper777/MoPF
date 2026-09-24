#!/usr/bin/env python3
"""Read-only P2.1a final response-evidence audit.

This audit consumes the existing S and P2 checkpoints only.  It does not train,
launch LP, create an architecture variant, or use test metrics for a decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.data import load_mag_data
from src.models.factory import build_model
from src.models.ssi_mag_scd import SSIMAGSCD

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
DEAD_PREFIXES = tuple(SSIMAGSCD.DEAD_COMPONENT_NAMES)
P2_VARIANTS = (
    "full",
    "no_cross_hop_interaction",
    "global_filter_only",
    "no_formation_conditioning",
)
P2_METHODS = {
    "full": "P2 Full",
    "no_cross_hop_interaction": "P2 no_cross_hop_interaction",
    "global_filter_only": "P2 global_filter_only",
    "no_formation_conditioning": "P2 no_formation_conditioning",
}
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
EPS = 1.0e-12
VAR_EPS = 1.0e-12
COVARIANCE_TOL = 2.0e-4
ATOL = 1.0e-4
COLLAPSE_RATIO_THRESHOLD = 1.0e-3


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["row_type"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_np(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _stats(prefix: str, value: Any) -> dict[str, float]:
    array = _as_np(value).reshape(-1)
    keys = ("mean", "std", "abs_mean", "q01", "q10", "q50", "q90", "q99", "min", "max")
    if array.size == 0:
        return {f"{prefix}_{key}": 0.0 for key in keys}
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(array))),
        f"{prefix}_q01": float(np.quantile(array, 0.01)),
        f"{prefix}_q10": float(np.quantile(array, 0.10)),
        f"{prefix}_q50": float(np.quantile(array, 0.50)),
        f"{prefix}_q90": float(np.quantile(array, 0.90)),
        f"{prefix}_q99": float(np.quantile(array, 0.99)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def _is_dead(key: str) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in DEAD_PREFIXES)


def _cfg(dataset: str, seed: int, model_name: str, device: str):
    base = OmegaConf.load(ROOT / "configs/config.yaml")
    dataset_cfg = OmegaConf.load(ROOT / "configs/dataset" / f"{dataset}.yaml")
    task_cfg = OmegaConf.load(ROOT / "configs/task/nc.yaml")
    model_cfg = OmegaConf.load(ROOT / "configs/model" / f"{model_name}.yaml")
    cfg = OmegaConf.merge(
        base,
        {
            "dataset": dataset_cfg,
            "task": task_cfg,
            "model": model_cfg,
            "seed": int(seed),
            "device": device,
            "ablation": "no_cross_hop_interaction" if model_name == "ssi_mag_final_ablation" else "full",
        },
    )
    cfg.hydra.run.dir = "outputs/p21a_analysis"
    OmegaConf.resolve(cfg)
    return cfg


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def map_checkpoint_state(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    kept = {key: value for key, value in source.items() if not _is_dead(key)}
    missing = sorted(set(target) - set(kept))
    unexpected = sorted(set(kept) - set(target))
    dropped = sorted(set(source) - set(kept))
    if missing:
        raise RuntimeError(f"active checkpoint keys missing in S: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected non-dead checkpoint keys: {unexpected}")
    if not dropped or not all(_is_dead(key) for key in dropped):
        raise RuntimeError(f"checkpoint dropped keys are not exactly dead branch: {dropped}")
    return {key: kept[key] for key in target}, dropped


def _max_diff(left: Any, right: Any) -> tuple[float, float, bool]:
    if isinstance(left, list):
        if not isinstance(right, list) or len(left) != len(right):
            return float("inf"), float("inf"), False
        values = [_max_diff(a, b) for a, b in zip(left, right, strict=True)]
        return max(item[0] for item in values), max(item[1] for item in values), all(item[2] for item in values)
    if not torch.is_tensor(left) or not torch.is_tensor(right) or left.shape != right.shape:
        return float("inf"), float("inf"), False
    a, b = left.detach().float(), right.detach().float()
    if a.numel() == 0:
        return 0.0, 0.0, True
    diff = (a - b).abs()
    scale = torch.maximum(a.abs(), b.abs()).clamp_min(torch.finfo(a.dtype).eps)
    max_abs = float(diff.max().item())
    max_rel = float((diff / scale).max().item())
    return max_abs, max_rel, bool(torch.isfinite(diff).all() and max_abs <= ATOL)


def _compare_pair(old: dict[str, Any], clean: dict[str, Any], head_old: torch.nn.Module, head_clean: torch.nn.Module) -> dict[str, tuple[float, float, bool]]:
    pairs: dict[str, tuple[Any, Any]] = {
        "h0_text": (old["h0_text"], clean["h0_text"]),
        "h0_visual": (old["h0_visual"], clean["h0_visual"]),
        "z_text": (old["z_text"], clean["z_text"]),
        "z_visual": (old["z_visual"], clean["z_visual"]),
        "z_text_refined": (old["z_text_refined"], clean["z_text_refined"]),
        "z_visual_refined": (old["z_visual_refined"], clean["z_visual_refined"]),
        "fused_z": (old["z"], clean["z"]),
    }
    for modality in ("text", "visual"):
        for key in ("relation_z", "relation_compatibility", "relation_mu", "relation_centered", "a", "relation_weight", "c", "normalized_edge_weight", "p", "term_p", "delta_content", "reference_residual", "relation_filter_residual", "eta"):
            pairs[f"{key}_{modality}"] = (old[f"{key}_{modality}"], clean[f"{key}_{modality}"])
        for key in ("d", "alpha", "Q", "S"):
            pairs[f"{key}_{modality}"] = (old[f"{key}_{modality}"], clean[f"{key}_{modality}"])
        pairs[f"S_used_{modality}"] = (old[f"S_tilde_{modality}"], clean[f"S_used_{modality}"])
    pairs["prediction_logits"] = (head_old(old["z"]), head_clean(clean["z"]))
    return {name: _max_diff(left, right) for name, (left, right) in pairs.items()}


def _load_pair(dataset: str, seed: int, p2_root: Path, device: torch.device, data_cache: dict[tuple[str, int], Any]):
    key = (dataset, seed)
    if key not in data_cache:
        data_cache[key] = load_mag_data(_cfg(dataset, seed, "ssi_mag_scd", str(device)), "nc", seed)
    data = data_cache[key]
    info = _data_info(data)
    old_cfg = _cfg(dataset, seed, "ssi_mag_final_ablation", str(device))
    clean_cfg = _cfg(dataset, seed, "ssi_mag_scd", str(device))
    old = build_model(old_cfg, info).to(device)
    clean = build_model(clean_cfg, info).to(device)
    path = p2_root / dataset / "no_cross_hop_interaction" / f"seed{seed}" / "best.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload["model_state"]
    old.load_state_dict(source, strict=True)
    mapped, dropped = map_checkpoint_state(source, clean.state_dict())
    clean.load_state_dict(mapped, strict=True)
    head_old = torch.nn.Linear(clean.out_dim, int(data.num_classes)).to(device)
    head_clean = torch.nn.Linear(clean.out_dim, int(data.num_classes)).to(device)
    head_old.load_state_dict(payload["head_state"], strict=True)
    head_clean.load_state_dict(payload["head_state"], strict=True)
    return data, old, clean, head_old, head_clean, dropped


def _finite_array(value: Any) -> bool:
    array = _as_np(value)
    return bool(array.size == 0 or np.isfinite(array).all())


def _corr(left: Any, right: Any, spearman: bool = False) -> float:
    """Pearson or average-rank Spearman correlation."""
    x = _as_np(left).reshape(-1)
    y = _as_np(right).reshape(-1)
    if x.size != y.size or x.size < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0
    if spearman:
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    return float(np.corrcoef(x, y)[0, 1])


def _cov_ratio(component: Any, eta: Any) -> float:
    x = _as_np(component).reshape(-1)
    y = _as_np(eta).reshape(-1)
    variance = float(np.var(y))
    if variance <= VAR_EPS:
        return 0.0
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / variance)


def _assert_covariance_identity(
    content: Any,
    reference: Any,
    relation: Any,
    eta: Any,
    *,
    label: str,
) -> tuple[float, float, float, float, float]:
    """Return component covariance shares and assert their identity."""
    variance = float(np.var(_as_np(eta).reshape(-1)))
    content_cov = _cov_ratio(content, eta)
    reference_cov = _cov_ratio(reference, eta)
    relation_cov = _cov_ratio(relation, eta)
    formation_cov = reference_cov + relation_cov
    total = content_cov + formation_cov
    if variance > VAR_EPS:
        if not np.isfinite(total) or abs(total - 1.0) > COVARIANCE_TOL:
            raise AssertionError(
                f"covariance identity failed for {label}: "
                f"content={content_cov}, reference={reference_cov}, "
                f"relation={relation_cov}, total={total}"
            )
    return content_cov, reference_cov, relation_cov, formation_cov, total


def _r1_rows(dataset: str, seed: int, analysis: dict[str, Any], model: SSIMAGSCD) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    compat_text = _as_np(analysis["relation_compatibility_text"])
    compat_visual = _as_np(analysis["relation_compatibility_visual"])
    edge_index = analysis["physical_edge_index"]
    nonself = (edge_index[0] != edge_index[1]).detach().cpu().numpy()
    discrepancy = compat_text - compat_visual
    discrepancy_values = discrepancy[nonself]
    for modality in ("text", "visual"):
        a = analysis[f"a_{modality}"]
        weights = analysis[f"relation_weight_{modality}"]
        c = analysis[f"c_{modality}"]
        raw_index, raw_weight = model._normalized_operator(
            edge_index,
            torch.ones(edge_index.size(1), device=edge_index.device),
            int(analysis["h0_text"].size(0)),
            analysis["h0_text"].dtype,
        )
        got_index = analysis[f"normalized_edge_index_{modality}"]
        if not torch.equal(got_index, raw_index):
            raise RuntimeError(f"normalized edge support changed for {dataset}/{seed}/{modality}")
        got_weight = analysis[f"normalized_edge_weight_{modality}"]
        raw_np, got_np = _as_np(raw_weight), _as_np(got_weight)
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "modality": modality,
            "row_type": "run_modality",
        }
        row.update(_stats("a", a))
        row.update(_stats("c", c))
        row.update(_stats("semantic_discrepancy", discrepancy_values))
        row["a_fraction_abs_gt_0_8"] = float(np.mean(np.abs(_as_np(a)) > 0.8))
        row["a_fraction_abs_gt_0_9"] = float(np.mean(np.abs(_as_np(a)) > 0.9))
        row["relation_weight_finite"] = int(_finite_array(weights))
        row.update(_stats("relation_weight", weights))
        w_np = _as_np(weights)
        row["relation_weight_cv"] = float(np.std(w_np) / max(abs(np.mean(w_np)), EPS))
        row["operator_mae_vs_raw"] = float(np.mean(np.abs(got_np - raw_np)))
        row["operator_rmse_vs_raw"] = float(np.sqrt(np.mean((got_np - raw_np) ** 2)))
        row["operator_relative_l1_vs_raw"] = float(np.sum(np.abs(got_np - raw_np)) / max(np.sum(np.abs(raw_np)), EPS))
        row["semantic_discrepancy_finite"] = int(np.isfinite(discrepancy_values).all())
        row["semantic_discrepancy_nondegenerate"] = int(float(np.std(discrepancy_values)) > EPS)
        row["relation_score_finite"] = int(_finite_array(a))
        row["operator_perturbation_finite"] = int(np.isfinite(raw_np).all() and np.isfinite(got_np).all())
        row["c_node_variation_nonzero"] = int(float(np.std(_as_np(c))) > EPS)
        row["same_edge_semantic_compatibility_pearson"] = _corr(compat_text[nonself], compat_visual[nonself])
        row["same_edge_semantic_compatibility_spearman"] = _corr(compat_text[nonself], compat_visual[nonself], spearman=True)
        row["same_edge_semantic_compatibility_mad"] = float(np.mean(np.abs(discrepancy_values)))
        rows.append(row)
    chain = {
        "semantic_discrepancy_mad": float(np.mean(np.abs(discrepancy_values))),
        "semantic_discrepancy_std": float(np.std(discrepancy_values)),
        "conductance_std_text": float(np.std(_as_np(analysis["a_text"])[nonself])),
        "conductance_std_visual": float(np.std(_as_np(analysis["a_visual"])[nonself])),
    }
    return rows, chain


def _r2_rows(dataset: str, seed: int, analysis: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    for modality in ("text", "visual"):
        p, term_p = analysis[f"p_{modality}"], analysis[f"term_p_{modality}"]
        rho_d = float(analysis[f"semantic_rho_d_{modality}"].item())
        rho_p = float(analysis[f"semantic_rho_p_{modality}"].item())
        biases = analysis[f"semantic_bias_{modality}"]
        for hop, (d, alpha, term_d) in enumerate(
            zip(analysis[f"d_{modality}"], analysis[f"alpha_{modality}"], analysis[f"term_d_{modality}"], strict=True),
            start=1,
        ):
            alpha_np = _as_np(alpha)
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "hop": hop,
                "row_type": "run_hop",
                "b": float(biases[hop - 1].item()),
                "rho_p": rho_p,
                "rho_d": rho_d,
                "alpha_finite": int(_finite_array(alpha)),
                "alpha_saturation_fraction": float(np.mean((alpha_np < 0.05) | (alpha_np > 0.95))),
                "alpha_saturated": int(np.any((alpha_np < 0.05) | (alpha_np > 0.95))),
            }
            row.update(_stats("p", p))
            row.update(_stats("term_p", term_p))
            row.update(_stats("d", d))
            row.update(_stats("term_d", term_d))
            row.update(_stats("alpha", alpha))
            row["alpha_iqr"] = float(np.quantile(alpha_np, 0.75) - np.quantile(alpha_np, 0.25))
            row["alpha_adaptive_range"] = float((np.quantile(alpha_np, 0.90) - np.quantile(alpha_np, 0.10)) / (abs(np.mean(alpha_np)) + EPS))
            row["alpha_fraction_lt_0_05"] = float(np.mean(alpha_np < 0.05))
            row["alpha_fraction_gt_0_95"] = float(np.mean(alpha_np > 0.95))
            row["corr_d_alpha_pearson"] = _corr(d, alpha)
            row["corr_d_alpha_spearman"] = _corr(d, alpha, spearman=True)
            row["corr_term_d_alpha_pearson"] = _corr(term_d, alpha)
            row["corr_term_d_alpha_spearman"] = _corr(term_d, alpha, spearman=True)
            rows.append(row)
            chain.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "modality": modality,
                    "hop": hop,
                    "evidence_axis": "R2_d_to_alpha",
                    "d_mean": row["d_mean"],
                    "alpha_mean": row["alpha_mean"],
                    "term_d_mean": row["term_d_mean"],
                    "alpha_finite": row["alpha_finite"],
                    "alpha_saturation_fraction": row["alpha_saturation_fraction"],
                    "corr_d_alpha_pearson": row["corr_d_alpha_pearson"],
                    "corr_d_alpha_spearman": row["corr_d_alpha_spearman"],
                    "corr_term_d_alpha_pearson": row["corr_term_d_alpha_pearson"],
                    "corr_term_d_alpha_spearman": row["corr_term_d_alpha_spearman"],
                }
            )
    return rows, chain


def _r3_rows(dataset: str, seed: int, analysis: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    chain: list[dict[str, Any]] = []
    for modality in ("text", "visual"):
        eta = analysis[f"eta_{modality}"]
        content = analysis[f"delta_content_{modality}"]
        reference = analysis[f"reference_residual_{modality}"]
        relation = analysis[f"relation_filter_residual_{modality}"]
        gamma = analysis[f"gamma_{modality}"] + analysis[f"delta_gamma_{modality}"]
        effective = analysis[f"effective_order_{modality}"]
        for hop in range(eta.size(1)):
            e, c, ref, rel = eta[:, hop], content[:, hop], reference[:, hop], relation[:, hop]
            content_cov, reference_cov, relation_cov, formation_cov, cov_sum = _assert_covariance_identity(
                c,
                ref,
                rel,
                e,
                label=f"{dataset}/{seed}/{modality}/hop{hop}",
            )
            c_abs = float(np.mean(np.abs(_as_np(c))))
            ref_abs = float(np.mean(np.abs(_as_np(ref))))
            rel_abs = float(np.mean(np.abs(_as_np(rel))))
            c_std = float(np.std(_as_np(c)))
            ref_std = float(np.std(_as_np(ref)))
            rel_std = float(np.std(_as_np(rel)))
            base_abs = abs(float(gamma[hop].item())) + EPS
            row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "hop": hop,
                "row_type": "run_hop",
                "gamma_plus_delta_gamma": float(gamma[hop].item()),
                "eta_variance": float(np.var(_as_np(e))),
                "negative_eta_fraction": float((_as_np(e) < 0).mean()),
                "content_covariance_contribution": content_cov,
                "reference_covariance_contribution": reference_cov,
                "relation_covariance_contribution": relation_cov,
                "formation_covariance_contribution": formation_cov,
                "covariance_identity_sum": cov_sum,
                "formation_abs_ratio": (ref_abs + rel_abs) / (c_abs + base_abs),
                "formation_node_std_ratio": (ref_std + rel_std) / (c_std + EPS),
                "formation_collapse_candidate": int(
                    (ref_abs + rel_abs) / (c_abs + base_abs) <= COLLAPSE_RATIO_THRESHOLD
                    and (ref_std + rel_std) / (c_std + EPS) <= COLLAPSE_RATIO_THRESHOLD
                    and abs(formation_cov) <= COLLAPSE_RATIO_THRESHOLD
                ),
            }
            for name, value in (
                ("gamma", gamma[hop]),
                ("delta_content", c),
                ("delta_ref", ref),
                ("delta_rel", rel),
                ("eta", e),
                ("effective_order", effective),
            ):
                row.update(_stats(name, value))
            rows.append(row)
            chain.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "modality": modality,
                    "hop": hop,
                    "evidence_axis": "R3_formation_to_eta",
                    "eta_variance": float(np.var(_as_np(e))),
                    "delta_ref_abs_mean": row["delta_ref_abs_mean"],
                    "delta_rel_abs_mean": row["delta_rel_abs_mean"],
                    "formation_abs_ratio": row["formation_abs_ratio"],
                    "formation_node_std_ratio": row["formation_node_std_ratio"],
                    "content_covariance_contribution": content_cov,
                    "reference_covariance_contribution": reference_cov,
                    "relation_covariance_contribution": relation_cov,
                    "formation_covariance_contribution": formation_cov,
                    "covariance_identity_sum": cov_sum,
                }
            )
    return rows, chain


def _formal_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            run = root / dataset / f"seed{seed}"
            metrics = _load_json(run / "metrics.json")
            results = _load_json(run / "results.json")
            row: dict[str, Any] = {
                "row_type": "run",
                "method": "S",
                "dataset": dataset,
                "seed": seed,
                "best_epoch": metrics.get("best_epoch"),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
            }
            for key in METRICS:
                row[key] = float(results[key]["mean"])
            rows.append(row)
    return rows


def _p2_rows(root: Path) -> list[dict[str, Any]]:
    source = root / "p2_runs.csv"
    if not source.is_file():
        raise RuntimeError(f"missing P2 analysis runs: {source}")
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            variant = raw.get("variant")
            if variant not in P2_VARIANTS or raw.get("row_type") != "run":
                continue
            row: dict[str, Any] = {
                "row_type": "run",
                "method": P2_METHODS[variant],
                "variant": variant,
                "dataset": raw["dataset"],
                "seed": int(raw["seed"]),
                "best_epoch": float(raw["best_epoch"]),
                "runtime_seconds": float(raw["runtime_seconds"]),
                "peak_gpu_memory_mib": float(raw["peak_gpu_memory_mib"]),
            }
            for key in METRICS:
                row[key] = float(raw[key])
            rows.append(row)
    expected = 15 * len(P2_VARIANTS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} P2 evidence rows, found {len(rows)}")
    return rows


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _performance_gate_rows(formal_root: Path, p2_analysis_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p2 = _p2_rows(p2_analysis_root)
    formal = _formal_rows(formal_root)
    rows: list[dict[str, Any]] = []
    lookup = {(row["method"], row["dataset"], int(row["seed"])): row for row in p2 + formal}
    baseline = "P2 no_cross_hop_interaction"
    dataset_means: list[dict[str, Any]] = []
    for dataset in DATASETS:
        run_deltas: list[dict[str, Any]] = []
        for seed in SEEDS:
            clean = lookup[("S", dataset, seed)]
            base = lookup[(baseline, dataset, seed)]
            delta = {
                "row_type": "paired_run",
                "comparison": "S-P2_no_cross_hop_interaction",
                "dataset": dataset,
                "seed": seed,
                "delta_val_acc": clean["val_acc"] - base["val_acc"],
                "delta_val_macro_f1": clean["val_macro_f1"] - base["val_macro_f1"],
                "delta_test_acc": clean["test_acc"] - base["test_acc"],
                "delta_test_macro_f1": clean["test_macro_f1"] - base["test_macro_f1"],
                "s_val_acc": clean["val_acc"],
                "s_val_macro_f1": clean["val_macro_f1"],
                "baseline_val_acc": base["val_acc"],
                "baseline_val_macro_f1": base["val_macro_f1"],
            }
            run_deltas.append(delta)
            rows.append(delta)
        dataset_row = {
            "row_type": "paired_dataset_mean",
            "comparison": "S-P2_no_cross_hop_interaction",
            "dataset": dataset,
            "seed_count": len(run_deltas),
        }
        for key in ("delta_val_acc", "delta_val_macro_f1", "delta_test_acc", "delta_test_macro_f1"):
            values = [float(item[key]) for item in run_deltas]
            dataset_row[f"{key}_mean"] = _mean(values)
            dataset_row[f"{key}_population_std"] = float(statistics.pstdev(values))
        dataset_row["both_validation_metrics_declined"] = int(
            dataset_row["delta_val_acc_mean"] < -EPS and dataset_row["delta_val_macro_f1_mean"] < -EPS
        )
        dataset_row["mean_gate_failed"] = int(
            dataset_row["delta_val_acc_mean"] < -0.003 or dataset_row["delta_val_macro_f1_mean"] < -0.003
        )
        dataset_means.append(dataset_row)
        rows.append(dataset_row)

    all_row: dict[str, Any] = {
        "row_type": "paired_all_dataset_mean",
        "comparison": "S-P2_no_cross_hop_interaction",
        "dataset_count": len(DATASETS),
        "run_count": len(DATASETS) * len(SEEDS),
        "aggregation": "unweighted_mean_of_dataset_paired_means",
    }
    for key in ("delta_val_acc", "delta_val_macro_f1", "delta_test_acc", "delta_test_macro_f1"):
        means = [float(item[f"{key}_mean"]) for item in dataset_means]
        all_row[f"{key}_mean"] = _mean(means)
        all_row[f"{key}_population_std"] = float(statistics.pstdev(means))
    all_row["both_validation_metrics_declined_dataset_count"] = sum(item["both_validation_metrics_declined"] for item in dataset_means)
    all_row["performance_mean_gate_failed"] = int(
        all_row["delta_val_acc_mean"] < -0.003 or all_row["delta_val_macro_f1_mean"] < -0.003
    )
    all_row["performance_count_gate_failed"] = int(all_row["both_validation_metrics_declined_dataset_count"] >= 3)
    all_row["performance_gate_failed"] = int(all_row["performance_mean_gate_failed"] or all_row["performance_count_gate_failed"])
    rows.append(all_row)

    # Existing retrained formation ablations are evidence only; test metrics are
    # retained nowhere in the decision logic below.
    for variant in ("no_formation_conditioning", "global_filter_only"):
        method = P2_METHODS[variant]
        for dataset in DATASETS:
            clean_vals = [lookup[("S", dataset, seed)]["val_acc"] for seed in SEEDS]
            clean_f1 = [lookup[("S", dataset, seed)]["val_macro_f1"] for seed in SEEDS]
            base_vals = [lookup[(method, dataset, seed)]["val_acc"] for seed in SEEDS]
            base_f1 = [lookup[(method, dataset, seed)]["val_macro_f1"] for seed in SEEDS]
            rows.append(
                {
                    "row_type": "formation_baseline_dataset_mean",
                    "comparison": f"S-{variant}",
                    "dataset": dataset,
                    "seed_count": len(SEEDS),
                    "s_val_acc_mean": _mean(clean_vals),
                    "s_val_macro_f1_mean": _mean(clean_f1),
                    "baseline_val_acc_mean": _mean(base_vals),
                    "baseline_val_macro_f1_mean": _mean(base_f1),
                    "delta_val_acc_mean": _mean(clean_vals) - _mean(base_vals),
                    "delta_val_macro_f1_mean": _mean(clean_f1) - _mean(base_f1),
                }
            )
        baseline_dataset_rows = [
            row for row in rows
            if row["row_type"] == "formation_baseline_dataset_mean" and row["comparison"] == f"S-{variant}"
        ]
        rows.append(
            {
                "row_type": "formation_baseline_all_dataset_mean",
                "comparison": f"S-{variant}",
                "dataset_count": len(DATASETS),
                "delta_val_acc_mean": _mean([row["delta_val_acc_mean"] for row in baseline_dataset_rows]),
                "delta_val_macro_f1_mean": _mean([row["delta_val_macro_f1_mean"] for row in baseline_dataset_rows]),
            }
        )

    baseline_summary: dict[str, dict[str, float]] = {}
    for variant in ("no_formation_conditioning", "global_filter_only"):
        record = next(
            row for row in rows
            if row["row_type"] == "formation_baseline_all_dataset_mean" and row["comparison"] == f"S-{variant}"
        )
        baseline_summary[variant] = {
            "delta_val_acc_mean": float(record["delta_val_acc_mean"]),
            "delta_val_macro_f1_mean": float(record["delta_val_macro_f1_mean"]),
        }
    # Positive S-minus-ablated deltas show that the existing formation-retaining
    # S checkpoint is better than the retrained no-formation/global-only controls.
    baseline_support = any(
        value > 0.0
        for item in baseline_summary.values()
        for value in item.values()
    )
    decision = {
        "delta_val_acc_mean": float(all_row["delta_val_acc_mean"]),
        "delta_val_macro_f1_mean": float(all_row["delta_val_macro_f1_mean"]),
        "performance_mean_gate_failed": bool(all_row["performance_mean_gate_failed"]),
        "performance_count_gate_failed": bool(all_row["performance_count_gate_failed"]),
        "performance_gate_failed": bool(all_row["performance_gate_failed"]),
        "both_validation_metrics_declined_dataset_count": int(all_row["both_validation_metrics_declined_dataset_count"]),
        "paired_dataset_means": dataset_means,
        "formation_baseline_evidence": baseline_summary,
        "formation_baseline_supports_conditioning": bool(baseline_support),
    }
    return rows, decision


def _parameter_count(model: torch.nn.Module, requires_grad_only: bool = False) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if not requires_grad_only or parameter.requires_grad))


def _latency(model: torch.nn.Module, data: Any, device: torch.device) -> float:
    model.eval()
    x, edge = data.x.to(device), data.edge_index.to(device)
    with torch.no_grad():
        for _ in range(2):
            model(x, edge)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(5):
            model(x, edge)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return float((time.perf_counter() - start) * 1000.0 / 5.0)


def _efficiency_rows(formal_root: Path, p2_analysis_root: Path, device: torch.device, data_cache: dict[tuple[str, int], Any]) -> list[dict[str, Any]]:
    p2 = _p2_rows(p2_analysis_root)
    formal = _formal_rows(formal_root)
    rows: list[dict[str, Any]] = []
    specs = (
        ("P2 Full", p2, "full", "ssi_mag_final_ablation"),
        ("P2 no_cross_hop_interaction", p2, "no_cross_hop_interaction", "ssi_mag_final_ablation"),
        ("S", formal, "full", "ssi_mag_scd"),
    )
    for method, source, ablation, model_name in specs:
        subset = [item for item in source if item["method"] == method]
        for item in subset:
            dataset, seed = item["dataset"], int(item["seed"])
            if (dataset, seed) not in data_cache:
                data_cache[(dataset, seed)] = load_mag_data(_cfg(dataset, seed, "ssi_mag_scd", str(device)), "nc", seed)
            data = data_cache[(dataset, seed)]
            cfg = _cfg(dataset, seed, model_name, str(device))
            cfg.ablation = ablation
            model = build_model(cfg, _data_info(data)).to(device)
            checkpoint = (
                p2_analysis_root.parent / "ssi_mag_final_nc_ablation" / dataset /
                ("full" if method == "P2 Full" else "no_cross_hop_interaction") /
                f"seed{seed}" / "best.pt"
                if method != "S" else formal_root / dataset / f"seed{seed}" / "best.pt"
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model_state = payload["model_state"]
            if method == "S":
                model.load_state_dict(model_state, strict=True)
            else:
                model.load_state_dict(model_state, strict=True)
            total = _parameter_count(model)
            requires_grad = _parameter_count(model, requires_grad_only=True)
            dead = int(sum(parameter.numel() for name, parameter in model.named_parameters() if method == "P2 no_cross_hop_interaction" and _is_dead(name)))
            active = total - dead
            row = {
                "row_type": "run",
                "method": method,
                "dataset": dataset,
                "seed": seed,
                "total_parameters": total,
                "requires_grad_parameters": requires_grad,
                "functionally_active_parameters": active,
                "dead_interaction_parameters": dead,
                "runtime_seconds": item["runtime_seconds"],
                "peak_gpu_memory_mib": item["peak_gpu_memory_mib"],
                "best_epoch": item["best_epoch"],
            }
            if seed == 42:
                row["eval_forward_latency_ms_seed42"] = _latency(model, data, device)
            rows.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for method in ("P2 Full", "P2 no_cross_hop_interaction", "S"):
        subset = [item for item in rows if item["method"] == method]
        row: dict[str, Any] = {"row_type": "all_dataset_summary", "method": method, "dataset_count": len(DATASETS), "run_count": len(subset)}
        for key in ("total_parameters", "requires_grad_parameters", "functionally_active_parameters", "dead_interaction_parameters", "runtime_seconds", "peak_gpu_memory_mib", "best_epoch", "eval_forward_latency_ms_seed42"):
            values = [float(item[key]) for item in subset if key in item and _is_finite(item[key])]
            if values:
                row[f"{key}_mean"] = _mean(values)
                row[f"{key}_population_std"] = float(statistics.pstdev(values))
        rows.append(row)
    return rows


def _health_summary(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    if kind == "r1":
        return {
            "row_type": "health_summary",
            "scope": "all",
            "run_modality_count": len(rows),
            "semantic_discrepancy_nondegenerate_fraction": _mean([float(row["semantic_discrepancy_nondegenerate"]) for row in rows]),
            "relation_score_finite_fraction": _mean([float(row["relation_score_finite"]) for row in rows]),
            "operator_perturbation_finite_fraction": _mean([float(row["operator_perturbation_finite"]) for row in rows]),
            "c_node_variation_nonzero_fraction": _mean([float(row["c_node_variation_nonzero"]) for row in rows]),
            "semantic_discrepancy_std_mean": _mean([float(row["semantic_discrepancy_std"]) for row in rows]),
            "relation_score_std_mean": _mean([float(row["a_std"]) for row in rows]),
            "operator_perturbation_mae_mean": _mean([float(row["operator_mae_vs_raw"]) for row in rows]),
            "c_node_std_mean": _mean([float(row["c_std"]) for row in rows]),
        }
    if kind == "r2":
        return {
            "row_type": "health_summary",
            "scope": "all",
            "run_hop_count": len(rows),
            "alpha_finite_fraction": _mean([float(row["alpha_finite"]) for row in rows]),
            "alpha_saturation_fraction_mean": _mean([float(row["alpha_saturation_fraction"]) for row in rows]),
            "alpha_std_mean": _mean([float(row["alpha_std"]) for row in rows]),
            "alpha_adaptive_range_mean": _mean([float(row["alpha_adaptive_range"]) for row in rows]),
            "corr_d_alpha_pearson_mean": _mean([float(row["corr_d_alpha_pearson"]) for row in rows]),
            "corr_d_alpha_spearman_mean": _mean([float(row["corr_d_alpha_spearman"]) for row in rows]),
            "corr_term_d_alpha_pearson_mean": _mean([float(row["corr_term_d_alpha_pearson"]) for row in rows]),
            "corr_term_d_alpha_spearman_mean": _mean([float(row["corr_term_d_alpha_spearman"]) for row in rows]),
            "correlation_threshold_applied": False,
        }
    collapse_count = sum(int(row["formation_collapse_candidate"]) for row in rows)
    return {
        "row_type": "health_summary",
        "scope": "all",
        "run_hop_count": len(rows),
        "formation_collapse_candidate_count": collapse_count,
        "formation_collapse_candidate_fraction": collapse_count / max(len(rows), 1),
        "formation_collapse_majority": bool(collapse_count > len(rows) / 2),
        "formation_abs_ratio_mean": _mean([float(row["formation_abs_ratio"]) for row in rows]),
        "formation_node_std_ratio_mean": _mean([float(row["formation_node_std_ratio"]) for row in rows]),
        "formation_covariance_contribution_mean": _mean([float(row["formation_covariance_contribution"]) for row in rows]),
        "formation_covariance_abs_mean": _mean([abs(float(row["formation_covariance_contribution"])) for row in rows]),
        "content_covariance_contribution_mean": _mean([float(row["content_covariance_contribution"]) for row in rows]),
        "covariance_identity_max_abs_error": max(abs(float(row["covariance_identity_sum"]) - 1.0) for row in rows if float(row["eta_variance"]) > VAR_EPS),
        "covariance_identity_asserted": True,
    }


def _decision_doc(decision: dict[str, Any], r1_health: dict[str, Any], r2_health: dict[str, Any], r3_health: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    performance_failed = bool(decision["performance_gate_failed"])
    formation_collapse = bool(r3_health["formation_collapse_majority"])
    baseline_support = bool(decision["formation_baseline_supports_conditioning"])
    trigger_reasons: list[str] = []
    if performance_failed:
        trigger_reasons.append("performance simplification gate failed")
    if formation_collapse and baseline_support:
        trigger_reasons.append("formation-conditioned response collapse with supporting retrained controls")
    status = "P22_RESPONSE_PILOT_TRIGGERED" if trigger_reasons else "FREEZE_S"
    summary = {
        "schema": "ssi_mag_scd_p21a_analysis_v1",
        "decision_status": status,
        "decision_reasons": trigger_reasons,
        "performance_gate": {
            "val_acc_delta_mean": decision["delta_val_acc_mean"],
            "val_macro_f1_delta_mean": decision["delta_val_macro_f1_mean"],
            "mean_gate_failed": decision["performance_mean_gate_failed"],
            "count_gate_failed": decision["performance_count_gate_failed"],
            "both_metrics_declined_dataset_count": decision["both_validation_metrics_declined_dataset_count"],
            "threshold_pp": -0.30,
            "dataset_paired_means": decision["paired_dataset_means"],
        },
        "formation_conditioning_health": {
            "collapse_thresholds": {
                "formation_abs_ratio": COLLAPSE_RATIO_THRESHOLD,
                "formation_node_std_ratio": COLLAPSE_RATIO_THRESHOLD,
                "abs_formation_covariance_contribution": COLLAPSE_RATIO_THRESHOLD,
                "majority_rule": ">50% of dataset/seed/modality/hop rows",
            },
            "r3": r3_health,
            "baseline_evidence": decision["formation_baseline_evidence"],
            "baseline_supports_conditioning": baseline_support,
        },
        "r1_health": r1_health,
        "r2_health": r2_health,
        "formal_nc_jobs": "15/15 existing checkpoints read",
        "training_jobs": 0,
        "lp_jobs": 0,
        "new_architecture_variants": 0,
        "test_used_for_decision": False,
        "next_step": (
            "Final architecture candidate = S; next: generic diffusion baselines, robustness, final paper ablation, LP. No new response mechanism."
            if status == "FREEZE_S" else
            "Evidence is recorded for one P2.2 response pilot; no P2.2 implementation or training is performed."
        ),
    }
    lines = [
        "# P2.1a Final Response Evidence Audit",
        "",
        f"Final decision: **{status}**.",
        "",
        "## Performance gate",
        "",
        f"S minus P2 `no_cross_hop_interaction` five-dataset paired validation mean: Accuracy {100*decision['delta_val_acc_mean']:.4f} pp; Macro-F1 {100*decision['delta_val_macro_f1_mean']:.4f} pp.",
        f"The dataset-level simultaneous-decline count is {decision['both_validation_metrics_declined_dataset_count']}/5; the count trigger requires at least 3/5.",
        "Test metrics were retained as descriptive rows only and were not used for this decision.",
        "",
        "## Formation-conditioned response",
        "",
        f"Formation covariance is reference plus relation only. The audited collapse-candidate fraction is {r3_health['formation_collapse_candidate_fraction']:.4f}; majority collapse = {r3_health['formation_collapse_majority']}.",
        f"Existing retrained controls are present: no_formation_conditioning and global_filter_only, with baseline support = {baseline_support}.",
        "",
        "## Mechanism health",
        "",
        "R1 reports semantic discrepancy nondegeneracy, finite relation scores, finite operator perturbation, and c-node variation without treating larger perturbation as better.",
        "R2 reports alpha finiteness/saturation/adaptive range and d-to-alpha Pearson/Spearman correlations; correlation magnitude has no hard threshold.",
        "R3 covariance identity assertions were applied whenever Var(eta) exceeded epsilon.",
        "",
        "## Boundary",
        "",
        "No training jobs, new architectures, LP, SHAR/FSCC, tuning, or test-based selection were used.",
    ]
    if status == "FREEZE_S":
        lines.extend([
            "",
            "Final architecture candidate = S.",
            "Next: generic diffusion baselines, robustness, final paper ablation, and LP. No new response mechanism is implemented.",
        ])
    else:
        lines.extend([
            "",
            "P2.2 is triggered only as a recorded evidence decision; this audit does not implement P2.2.",
        ])
    return "\n".join(lines) + "\n", summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-root", type=Path, default=Path("outputs/ssi_mag_final_nc_ablation"))
    parser.add_argument("--p2-analysis-root", type=Path, default=Path("outputs/ssi_mag_final_nc_ablation_analysis"))
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/ssi_mag_scd_p21_nc"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssi_mag_scd_p21a_analysis"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    p2_root = (ROOT / args.p2_root).resolve() if not args.p2_root.is_absolute() else args.p2_root.resolve()
    p2_analysis_root = (ROOT / args.p2_analysis_root).resolve() if not args.p2_analysis_root.is_absolute() else args.p2_analysis_root.resolve()
    formal_root = (ROOT / args.formal_root).resolve() if not args.formal_root.is_absolute() else args.formal_root.resolve()
    output = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    output.mkdir(parents=True, exist_ok=True)

    r1_rows: list[dict[str, Any]] = []
    r2_rows: list[dict[str, Any]] = []
    r3_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    data_cache: dict[tuple[str, int], Any] = {}
    equivalence_rows: list[dict[str, Any]] = []
    checkpoint_summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            data, old, clean, head_old, head_clean, dropped = _load_pair(
                dataset, seed, p2_root, device, data_cache
            )
            old.eval(), clean.eval(), head_old.eval(), head_clean.eval()
            with torch.no_grad():
                old_analysis = old.analysis(data.x.to(device), data.edge_index.to(device))
                clean_analysis = clean.analysis(data.x.to(device), data.edge_index.to(device))
            comparisons = _compare_pair(old_analysis, clean_analysis, head_old, head_clean)
            good = all(item[2] for item in comparisons.values())
            for quantity, (max_abs, max_rel, equivalent) in comparisons.items():
                equivalence_rows.append({
                    "row_type": "quantity",
                    "dataset": dataset,
                    "seed": seed,
                    "quantity": quantity,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                    "equivalent_at_1e-4": int(equivalent),
                })
            checkpoint_summary.append({
                "dataset": dataset,
                "seed": seed,
                "active_output_equivalent": int(good),
                "dropped_dead_key_count": len(dropped),
            })
            if not good:
                raise RuntimeError(f"checkpoint equivalence failed for {dataset}/seed{seed}")
            r1, r1_chain = _r1_rows(dataset, seed, clean_analysis, clean)
            r2, r2_chain = _r2_rows(dataset, seed, clean_analysis)
            r3, r3_chain = _r3_rows(dataset, seed, clean_analysis)
            r1_rows.extend(r1)
            r2_rows.extend(r2)
            r3_rows.extend(r3)
            chain_rows.append({"row_type": "R1", "dataset": dataset, "seed": seed, **r1_chain})
            chain_rows.extend({"row_type": "R2", **row} for row in r2_chain)
            chain_rows.extend({"row_type": "R3", **row} for row in r3_chain)
            del old, clean, head_old, head_clean, old_analysis, clean_analysis
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_csv(output / "p21a_checkpoint_equivalence.csv", equivalence_rows + [{"row_type": "checkpoint_summary", **row} for row in checkpoint_summary])
    performance, decision = _performance_gate_rows(formal_root, p2_analysis_root)
    _write_csv(output / "p21a_performance_gate.csv", performance)
    r1_health = _health_summary(r1_rows, "r1")
    r2_health = _health_summary(r2_rows, "r2")
    r3_health = _health_summary(r3_rows, "r3")
    _write_csv(output / "p21a_r1_health.csv", r1_rows + [r1_health])
    _write_csv(output / "p21a_r2_health.csv", r2_rows + [r2_health])
    _write_csv(output / "p21a_response_health.csv", r3_rows + [r3_health])
    efficiency = _efficiency_rows(formal_root, p2_analysis_root, device, data_cache)
    _write_csv(output / "p21a_efficiency_corrected.csv", efficiency)
    decision_text, summary = _decision_doc(decision, r1_health, r2_health, r3_health)
    summary.update({
        "formal_root": str(formal_root),
        "p2_root": str(p2_root),
        "output_root": str(output),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "checkpoint_equivalence": "15/15",
        "checkpoint_max_abs_atol": ATOL,
        "protocol": "unified_full_graph_nc_v1",
    })
    (output / "p21a_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = decision_text + "\n## Formal outputs\n\n" + "\n".join(
        f"- `{name}`" for name in (
            "p21a_performance_gate.csv",
            "p21a_response_health.csv",
            "p21a_r1_health.csv",
            "p21a_r2_health.csv",
            "p21a_efficiency_corrected.csv",
            "p21a_summary.json",
        )
    ) + "\n"
    (output / "p21a_report.md").write_text(report, encoding="utf-8")
    doc_path = ROOT / "docs/p21a_next_decision.md"
    doc_path.write_text(decision_text, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
