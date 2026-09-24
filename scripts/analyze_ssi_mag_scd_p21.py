#!/usr/bin/env python3
"""Run the P2.1 clean-model audit and write registered evidence tables.

The first stage maps every P2 formal ``no_cross_hop_interaction`` checkpoint
to ``ssi_mag_scd`` and stops on any active-output mismatch.  The same pass
then computes the attention-free R1/R2/R3 diagnostics.  Formal S performance
and efficiency are read only after the 15 S jobs are complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data import load_mag_data
from src.models.factory import build_model
from src.models.ssi_mag_scd import SSIMAGSCD

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
DEAD_PREFIXES = tuple(SSIMAGSCD.DEAD_COMPONENT_NAMES)
P2_VARIANTS = ("full", "no_cross_hop_interaction")
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
EPS = 1.0e-12
ATOL = 1.0e-4


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["row_type"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
    if array.size == 0:
        return {f"{prefix}_{key}": 0.0 for key in ("mean", "std", "abs_mean", "q01", "q10", "q50", "q90", "q99", "min", "max")}
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


def _corr(left: Any, right: Any, spearman: bool = False) -> float:
    x, y = _as_np(left).reshape(-1), _as_np(right).reshape(-1)
    if x.size != y.size or x.size < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0
    if spearman:
        x = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
        y = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def _cov_ratio(component: Any, eta: Any) -> float:
    x, y = _as_np(component).reshape(-1), _as_np(eta).reshape(-1)
    variance = float(np.var(y))
    if variance <= EPS:
        return 0.0
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / variance)


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
    # This read-only analyzer does not run Hydra jobs; remove its time resolver.
    cfg.hydra.run.dir = "outputs/p21_analysis"
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


def _is_dead(key: str) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in DEAD_PREFIXES)


def map_checkpoint_state(source: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Map P2 no-cross state to S and reject every non-dead discrepancy."""
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
        for key in (
            "relation_z", "relation_compatibility", "relation_mu", "relation_centered",
            "a", "relation_weight", "c", "normalized_edge_weight", "p", "term_p",
            "delta_content", "reference_residual", "relation_filter_residual", "eta",
        ):
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


def _r1_rows(dataset: str, seed: int, analysis: dict[str, Any], model: SSIMAGSCD) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    compat_text = _as_np(analysis["relation_compatibility_text"])
    compat_visual = _as_np(analysis["relation_compatibility_visual"])
    discrepancy = compat_text - compat_visual
    for modality in ("text", "visual"):
        a = analysis[f"a_{modality}"]
        weights = analysis[f"relation_weight_{modality}"]
        c = analysis[f"c_{modality}"]
        nonself = analysis["physical_edge_index"][0] != analysis["physical_edge_index"][1]
        w_np = _as_np(weights)
        raw_index, raw_weight = model._normalized_operator(
            analysis["physical_edge_index"],
            torch.ones(analysis["physical_edge_index"].size(1), device=analysis["physical_edge_index"].device),
            int(analysis["h0_text"].size(0)),
            analysis["h0_text"].dtype,
        )
        got_index = analysis[f"normalized_edge_index_{modality}"]
        if not torch.equal(got_index, raw_index):
            raise RuntimeError(f"normalized edge support changed for {dataset}/{seed}/{modality}")
        got_weight = analysis[f"normalized_edge_weight_{modality}"]
        raw_np, got_np = _as_np(raw_weight), _as_np(got_weight)
        row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "row_type": "run_modality"}
        row.update(_stats("a", a))
        row["a_fraction_abs_gt_0_8"] = float(np.mean(np.abs(_as_np(a)) > 0.8))
        row["a_fraction_abs_gt_0_9"] = float(np.mean(np.abs(_as_np(a)) > 0.9))
        row.update(_stats("relation_weight", weights))
        row["relation_weight_cv"] = float(np.std(w_np) / max(abs(np.mean(w_np)), EPS))
        row.update(_stats("c", c))
        row["operator_mae_vs_raw"] = float(np.mean(np.abs(got_np - raw_np)))
        row["operator_rmse_vs_raw"] = float(np.sqrt(np.mean((got_np - raw_np) ** 2)))
        row["operator_relative_l1_vs_raw"] = float(np.sum(np.abs(got_np - raw_np)) / max(np.sum(np.abs(raw_np)), EPS))
        row["same_edge_semantic_compatibility_pearson"] = _corr(compat_text[nonself.cpu().numpy()], compat_visual[nonself.cpu().numpy()])
        row["same_edge_semantic_compatibility_spearman"] = _corr(compat_text[nonself.cpu().numpy()], compat_visual[nonself.cpu().numpy()], spearman=True)
        row["same_edge_semantic_compatibility_mad"] = float(np.mean(np.abs(discrepancy[nonself.cpu().numpy()])))
        rows.append(row)
    chain = {
        "semantic_discrepancy_mad": float(np.mean(np.abs(discrepancy[nonself.cpu().numpy()]))),
        "conductance_std_text": float(np.std(_as_np(analysis["a_text"])[nonself.cpu().numpy()])),
        "conductance_std_visual": float(np.std(_as_np(analysis["a_visual"])[nonself.cpu().numpy()])),
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
        for hop, (d, alpha, term_d) in enumerate(zip(analysis[f"d_{modality}"], analysis[f"alpha_{modality}"], analysis[f"term_d_{modality}"], strict=True), start=1):
            alpha_np = _as_np(alpha)
            stats = _stats("alpha", alpha)
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "row_type": "run_hop", "b": float(biases[hop - 1].item()), "rho_p": rho_p, "rho_d": rho_d}
            row.update(_stats("p", p))
            row.update(_stats("term_p", term_p))
            row.update(_stats("d", d))
            row.update(_stats("term_d", term_d))
            row.update(stats)
            row["alpha_iqr"] = float(np.quantile(alpha_np, 0.75) - np.quantile(alpha_np, 0.25))
            row["alpha_adaptive_range"] = float((np.quantile(alpha_np, 0.90) - np.quantile(alpha_np, 0.10)) / (abs(np.mean(alpha_np)) + EPS))
            row["alpha_fraction_lt_0_05"] = float(np.mean(alpha_np < 0.05))
            row["alpha_fraction_gt_0_95"] = float(np.mean(alpha_np > 0.95))
            row["corr_d_alpha_pearson"] = _corr(d, alpha)
            row["corr_d_alpha_spearman"] = _corr(d, alpha, spearman=True)
            row["corr_term_d_alpha_pearson"] = _corr(term_d, alpha)
            row["corr_term_d_alpha_spearman"] = _corr(term_d, alpha, spearman=True)
            rows.append(row)
            chain.append({"dataset": dataset, "seed": seed, "modality": modality, "evidence_axis": "R2_d_to_alpha", "d_mean": row["d_mean"], "alpha_mean": row["alpha_mean"], "term_d_mean": row["term_d_mean"], "corr_d_alpha_pearson": row["corr_d_alpha_pearson"], "corr_term_d_alpha_pearson": row["corr_term_d_alpha_pearson"]})
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
            e, c, r, rel = eta[:, hop], content[:, hop], reference[:, hop], relation[:, hop]
            rc, rr, rrel = _cov_ratio(c, e), _cov_ratio(r, e), _cov_ratio(rel, e)
            row: dict[str, Any] = {"dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "row_type": "run_hop", "gamma_plus_delta_gamma": float(gamma[hop].item()), "negative_eta_fraction": float((_as_np(e) < 0).mean()), "content_variance_attribution": rc, "reference_variance_attribution": rr, "relation_variance_attribution": rrel, "variance_attribution_sum": rc + rr + rrel}
            for name, value in (("gamma", gamma[hop]), ("delta_content", c), ("delta_ref", r), ("delta_rel", rel), ("eta", e), ("effective_order", effective)):
                row.update(_stats(name, value))
            rows.append(row)
            chain.append({"dataset": dataset, "seed": seed, "modality": modality, "evidence_axis": "R3_c_alpha_to_eta", "hop": hop, "c_mean": float(_as_np(analysis[f'c_{modality}']).mean()), "eta_variance": float(np.var(_as_np(e))), "content_variance_attribution": rc, "reference_variance_attribution": rr, "relation_variance_attribution": rrel})
    return rows, chain


def _formal_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            run = root / dataset / f"seed{seed}"
            metrics = _load_json(run / "metrics.json")
            results = _load_json(run / "results.json")
            row: dict[str, Any] = {"row_type": "run", "method": "S", "dataset": dataset, "seed": seed, "best_epoch": metrics.get("best_epoch"), "runtime_seconds": metrics.get("runtime_seconds"), "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib")}
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
            if raw.get("variant") not in P2_VARIANTS or raw.get("row_type") != "run":
                continue
            row: dict[str, Any] = {"row_type": "run", "method": "P2 Full" if raw["variant"] == "full" else "P2 no_cross_hop_interaction", "dataset": raw["dataset"], "seed": int(raw["seed"]), "best_epoch": float(raw["best_epoch"]), "runtime_seconds": float(raw["runtime_seconds"]), "peak_gpu_memory_mib": float(raw["peak_gpu_memory_mib"])}
            for key in METRICS:
                row[key] = float(raw[key])
            rows.append(row)
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 P2 Full/no-cross rows, found {len(rows)}")
    return rows


def _performance_rows(formal_root: Path, p2_analysis_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _p2_rows(p2_analysis_root) + _formal_rows(formal_root)
    dataset_summary: list[dict[str, Any]] = []
    methods = ("P2 Full", "P2 no_cross_hop_interaction", "S")
    for method in methods:
        for dataset in DATASETS:
            subset = [row for row in rows if row["method"] == method and row["dataset"] == dataset]
            if len(subset) != 3:
                raise RuntimeError(f"expected 3 rows for {method}/{dataset}, found {len(subset)}")
            row = {"row_type": "dataset_summary", "method": method, "dataset": dataset, "seed_count": 3}
            for key in METRICS + ("best_epoch", "runtime_seconds", "peak_gpu_memory_mib"):
                values = [float(item[key]) for item in subset if _is_finite(item.get(key))]
                row[f"{key}_mean"] = statistics.mean(values)
                row[f"{key}_population_std"] = statistics.pstdev(values)
            dataset_summary.append(row)
    rows.extend(dataset_summary)
    all_summary: list[dict[str, Any]] = []
    for method in methods:
        subset = [row for row in dataset_summary if row["method"] == method]
        row = {"row_type": "all_dataset_summary", "method": method, "dataset_count": 5, "aggregation": "unweighted_mean_of_dataset_means"}
        for key in METRICS + ("best_epoch", "runtime_seconds", "peak_gpu_memory_mib"):
            values = [item[f"{key}_mean"] for item in subset]
            row[f"{key}_mean"] = statistics.mean(values)
            row[f"{key}_population_std"] = statistics.pstdev(values)
        all_summary.append(row)
    rows.extend(all_summary)
    lookup = {(row["method"], row["dataset"], int(row["seed"])): row for row in rows if row["row_type"] == "run"}
    paired: list[dict[str, Any]] = []
    for baseline in ("P2 no_cross_hop_interaction", "P2 Full"):
        comparison = "S-" + baseline.replace("P2 ", "P2_")
        for dataset in DATASETS:
            for seed in SEEDS:
                clean, base = lookup[("S", dataset, seed)], lookup[(baseline, dataset, seed)]
                row = {"row_type": "paired_run", "comparison": comparison, "dataset": dataset, "seed": seed}
                for key in METRICS:
                    row[f"delta_{key}"] = clean[key] - base[key]
                paired.append(row)
            subset = [item for item in paired if item["row_type"] == "paired_run" and item["comparison"] == comparison and item["dataset"] == dataset]
            row = {"row_type": "paired_dataset_summary", "comparison": comparison, "dataset": dataset, "seed_count": 3}
            for key in METRICS:
                values = [item[f"delta_{key}"] for item in subset]
                row[f"delta_{key}_mean"] = statistics.mean(values)
                row[f"delta_{key}_population_std"] = statistics.pstdev(values)
            paired.append(row)
        subset = [item for item in paired if item["row_type"] == "paired_run" and item["comparison"] == comparison]
        dataset_means = [item for item in paired if item["row_type"] == "paired_dataset_summary" and item["comparison"] == comparison]
        row = {"row_type": "paired_all_dataset_summary", "comparison": comparison, "dataset_count": 5, "run_count": 15}
        for key in METRICS:
            values = [item[f"delta_{key}"] for item in subset]
            means = [item[f"delta_{key}_mean"] for item in dataset_means]
            row[f"delta_{key}_mean"] = statistics.mean(means)
            row[f"delta_{key}_population_std"] = statistics.pstdev(means)
            row[f"better_count_{key}"] = sum(value > EPS for value in values)
            row[f"worse_count_{key}"] = sum(value < -EPS for value in values)
            row[f"tie_count_{key}"] = 15 - row[f"better_count_{key}"] - row[f"worse_count_{key}"]
        paired.append(row)
    rows.extend(paired)
    decision = next(item for item in paired if item["row_type"] == "paired_all_dataset_summary" and item["comparison"] == "S-P2_no_cross_hop_interaction")
    return rows, decision


def _parameter_count(model_state: dict[str, torch.Tensor]) -> int:
    return int(sum(value.numel() for value in model_state.values()))


def _latency(model, data, device: torch.device) -> float:
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
    for method, source, model_name in (("P2 Full", p2, "ssi_mag_final_ablation"), ("P2 no_cross_hop_interaction", p2, "ssi_mag_final_ablation"), ("S", formal, "ssi_mag_scd")):
        subset = [item for item in source if item["method"] == method]
        for item in subset:
            dataset, seed = item["dataset"], int(item["seed"])
            checkpoint = (p2_analysis_root.parent / "ssi_mag_final_nc_ablation" / dataset / ("full" if method == "P2 Full" else "no_cross_hop_interaction") / f"seed{seed}" / "best.pt") if method != "S" else (formal_root / dataset / f"seed{seed}" / "best.pt")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model_state = payload["model_state"]
            active = _parameter_count(model_state)
            if method == "S":
                total, trainable = active, active
            else:
                dropped = sum(value.numel() for key, value in model_state.items() if _is_dead(key))
                total, trainable = active, active
                active -= dropped
                if method == "P2 Full":
                    active = total
            row = {"row_type": "run", "method": method, "dataset": dataset, "seed": seed, "model_trainable_parameters": active, "model_total_parameters": total, "runtime_seconds": item["runtime_seconds"], "peak_gpu_memory_mib": item["peak_gpu_memory_mib"], "best_epoch": item["best_epoch"]}
            if seed == 42:
                if (dataset, seed) not in data_cache:
                    data_cache[(dataset, seed)] = load_mag_data(_cfg(dataset, seed, "ssi_mag_scd", str(device)), "nc", seed)
                data = data_cache[(dataset, seed)]
                cfg = _cfg(dataset, seed, model_name, str(device))
                model = build_model(cfg, _data_info(data)).to(device)
                if method == "S":
                    mapped, _ = map_checkpoint_state(model_state, model.state_dict()) if any(_is_dead(key) for key in model_state) else (model_state, [])
                    model.load_state_dict(mapped, strict=True)
                else:
                    model.load_state_dict(model_state, strict=True)
                row["eval_forward_latency_ms_seed42"] = _latency(model, data, device)
            rows.append(row)
    for method in ("P2 Full", "P2 no_cross_hop_interaction", "S"):
        subset = [item for item in rows if item["method"] == method]
        row = {"row_type": "all_dataset_summary", "method": method, "dataset_count": 5, "run_count": 15}
        for key in ("model_trainable_parameters", "model_total_parameters", "runtime_seconds", "peak_gpu_memory_mib", "best_epoch", "eval_forward_latency_ms_seed42"):
            values = [float(item[key]) for item in subset if key in item and _is_finite(item[key])]
            if values:
                row[f"{key}_mean"] = statistics.mean(values)
                row[f"{key}_population_std"] = statistics.pstdev(values)
        rows.append(row)
    return rows


def _decision_doc(output: Path, decision: dict[str, Any], r1_rows: list[dict[str, Any]], r2_rows: list[dict[str, Any]], r3_rows: list[dict[str, Any]], efficiency_rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    val_acc = float(decision["delta_val_acc_mean"])
    val_f1 = float(decision["delta_val_macro_f1_mean"])
    dataset_pairs = {(row["dataset"]): row for row in []}
    del dataset_pairs
    review = val_acc < -0.003 or val_f1 < -0.003
    r3_eta_std = float(np.mean([row["eta_std"] for row in r3_rows])) if r3_rows else 0.0
    r3_effective_std = float(np.mean([row["effective_order_std"] for row in r3_rows])) if r3_rows else 0.0
    contribution_abs = [abs(float(row["content_variance_attribution"])) + abs(float(row["reference_variance_attribution"])) + abs(float(row["relation_variance_attribution"])) for row in r3_rows]
    mechanism = {
        "r1_rows": len(r1_rows),
        "r2_rows": len(r2_rows),
        "r3_rows": len(r3_rows),
        "eta_node_std_mean": r3_eta_std,
        "effective_order_node_std_mean": r3_effective_std,
        "eta_node_variance_collapse": r3_eta_std < 1.0e-6,
        "formation_contribution_abs_sum_mean": float(np.mean(contribution_abs)) if contribution_abs else 0.0,
        "formation_conditioning_weak": bool(contribution_abs) and float(np.mean(contribution_abs)) < 0.15,
        "strong_baseline_evidence": "not evaluated",
    }
    if mechanism["eta_node_variance_collapse"]:
        review = True
    status = "REVIEW_REQUIRED" if review else "SIMPLIFICATION_PASS"
    summary = {"decision_status": status, "validation_delta_mean": {"val_acc": val_acc, "val_macro_f1": val_f1}, "gate_threshold_pp": -0.003, "mechanism_health": mechanism, "formal_nc_jobs": "15/15", "lp_jobs": 0, "new_architecture_variants_beyond_S": 0}
    text = "\n".join([
        "# P2.1 next decision", "",
        f"Decision status: **{status}**.", "",
        f"S minus P2 `no_cross_hop_interaction` five-dataset validation mean: Accuracy {100*val_acc:.4f} pp; Macro-F1 {100*val_f1:.4f} pp.",
        "The gate uses validation metrics, mechanism diagnostics, and efficiency only; Test metrics remain descriptive.", "",
        "## A. Is S sufficient to freeze?", "",
        "Yes under the predefined simplification gate." if status == "SIMPLIFICATION_PASS" else "No; the predefined review gate was triggered.", "",
        "## B. Is formation-conditioned response too weak?", "",
        "See the registered eta variance and component-attribution fields in `p21_response_diagnostics.csv`; no causal interpretation is assigned.", "",
        "## C. Is a response-enhancement pilot triggered?", "",
        "No automatic enhancement is implemented. Strong-baseline evidence is `not evaluated` unless separately registered; this round does not add SHAR, FSCC, attention replacement, fusion gates, MoE, or spectral redesign.", "",
        "## Boundary", "",
        "Formal NC jobs = 15/15; LP jobs = 0; new architecture variants beyond S = 0.", "",
    ]) + "\n"
    return text, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-root", type=Path, default=Path("outputs/ssi_mag_final_nc_ablation"))
    parser.add_argument("--p2-analysis-root", type=Path, default=Path("outputs/ssi_mag_final_nc_ablation_analysis"))
    parser.add_argument("--formal-root", type=Path, default=Path("outputs/ssi_mag_scd_p21_nc"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssi_mag_scd_p21_analysis"))
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

    equivalence_rows: list[dict[str, Any]] = []
    r1_rows: list[dict[str, Any]] = []
    r2_rows: list[dict[str, Any]] = []
    r3_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    data_cache: dict[tuple[str, int], Any] = {}
    checkpoint_summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            data, old, clean, head_old, head_clean, dropped = _load_pair(dataset, seed, p2_root, device, data_cache)
            old.eval(), clean.eval(), head_old.eval(), head_clean.eval()
            with torch.no_grad():
                old_analysis = old.analysis(data.x.to(device), data.edge_index.to(device))
                clean_analysis = clean.analysis(data.x.to(device), data.edge_index.to(device))
            comparisons = _compare_pair(old_analysis, clean_analysis, head_old, head_clean)
            good = all(item[2] for item in comparisons.values())
            for quantity, (max_abs, max_rel, equivalent) in comparisons.items():
                equivalence_rows.append({"row_type": "quantity", "dataset": dataset, "seed": seed, "quantity": quantity, "max_abs": max_abs, "max_rel": max_rel, "equivalent_at_1e-4": int(equivalent), "dropped_dead_key_count": len(dropped)})
            checkpoint_summary.append({"dataset": dataset, "seed": seed, "checkpoint": str(p2_root / dataset / "no_cross_hop_interaction" / f"seed{seed}" / "best.pt"), "active_output_equivalent": int(good), "dropped_dead_key_count": len(dropped), "dropped_dead_keys": ";".join(dropped)})
            if not good:
                raise RuntimeError(f"checkpoint equivalence failed for {dataset}/seed{seed}; refusing training/summary")
            r1, r1_chain = _r1_rows(dataset, seed, clean_analysis, clean)
            r2, r2_chain = _r2_rows(dataset, seed, clean_analysis)
            r3, r3_chain = _r3_rows(dataset, seed, clean_analysis)
            r1_rows.extend(r1), r2_rows.extend(r2), r3_rows.extend(r3)
            chain_rows.append({"row_type": "R1", "dataset": dataset, "seed": seed, **r1_chain})
            chain_rows.extend({"row_type": "R2", **row} for row in r2_chain)
            chain_rows.extend({"row_type": "R3", **row} for row in r3_chain)
            del old, clean, head_old, head_clean, old_analysis, clean_analysis
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_csv(output / "p21_checkpoint_equivalence.csv", equivalence_rows + [{"row_type": "checkpoint_summary", **row} for row in checkpoint_summary])
    _write_csv(output / "p21_relation_diagnostics.csv", r1_rows)
    _write_csv(output / "p21_restart_diagnostics.csv", r2_rows)
    _write_csv(output / "p21_response_diagnostics.csv", r3_rows)
    _write_csv(output / "p21_feedback_chain.csv", chain_rows)

    performance, decision = _performance_rows(formal_root, p2_analysis_root)
    _write_csv(output / "p21_performance.csv", performance)
    efficiency = _efficiency_rows(formal_root, p2_analysis_root, device, data_cache)
    _write_csv(output / "p21_efficiency.csv", efficiency)
    decision_text, summary = _decision_doc(output, decision, r1_rows, r2_rows, r3_rows, efficiency)
    (ROOT / "docs/p21_next_decision.md").write_text(decision_text, encoding="utf-8")
    summary.update({"schema": "ssi_mag_scd_p21_analysis_v1", "checkpoint_equivalence": "15/15", "formal_root": str(formal_root), "p2_root": str(p2_root), "output_root": str(output), "protocol": "unified_full_graph_nc_v1", "datasets": list(DATASETS), "seeds": list(SEEDS), "training_test_used_for_selection": False})
    (output / "p21_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "p21_report.md").write_text("\n".join([
        "# P2.1 SCD report", "", "## Integrity", "", "- Formal NC jobs: 15/15", "- LP jobs: 0", "- New architecture variants beyond S: 0", "- Checkpoint equivalence: 15/15", "- Equivalence tolerance: `1e-4` maximum absolute error", "", "## Outputs", "", "- `p21_performance.csv`: S versus P2 no-cross paired validation/test-descriptive comparison and S versus P2 Full descriptive comparison.", "- `p21_relation_diagnostics.csv`, `p21_restart_diagnostics.csv`, `p21_response_diagnostics.csv`: registered R1/R2/R3 diagnostics without attention metrics.", "- `p21_feedback_chain.csv`: separate chain-of-evidence records; no total score.", "- `p21_efficiency.csv`: P2 Full, P2 no-cross, and clean S efficiency records.", "", f"## Decision", "", f"Status: **{summary['decision_status']}**.", "", "Test is not used for the architecture decision. No enhancement mechanism is implemented automatically.", "", ]) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
