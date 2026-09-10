"""Run and summarize the MoPF-vNext U1 NC semantic-conductance study.

The script intentionally invokes the unchanged NC runner through
``src.main``. It does not import or modify the LP runner. Each dataset,
variant, and seed is a separate formal full-graph NC job so that timing,
checkpoint, and peak-memory metadata are unambiguous.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402
from src.utils.summary import count_parameters, mean_std  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
VARIANTS = ("S0", "S1")
SEEDS = (42, 43, 44)
K_BY_DATASET = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
}
MAGB_DATASETS = {"Movies", "Toys", "Grocery", "Reddit-S"}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _compose_cfg(
    dataset: str,
    seed: int,
    variant: str,
    device: str,
    k: int,
):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        f"seed={seed}",
        f"device={device}",
        f"model.edge_weight_mode={'separate_cos' if variant == 'S0' else 'multi_perspective_cos'}",
        f"model.max_order={k}",
        f"model.num_layers={k}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _fixed_split_override(dataset: str) -> str | None:
    if dataset not in MAGB_DATASETS:
        return None
    return str(
        Path("/hdd1/DataInHere/YHF/data/MAGB_split")
        / f"{dataset}_nc_seed42_train0.6_val0.2.pt"
    )


def _device_index(device: str) -> str | None:
    value = str(device).strip().lower()
    if value.startswith("cuda:"):
        return value.split(":", 1)[1]
    if value == "cuda":
        return "0"
    return None


def _peak_gpu_memory_mib(pid: int, device: str) -> float | None:
    index = _device_index(device)
    if index is None:
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    peak = None
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or fields[0] != str(pid):
            continue
        value = fields[1].split()[0]
        try:
            current = float(value)
        except ValueError:
            continue
        peak = current if peak is None else max(peak, current)
    return peak


def _run_formal_job(
    *,
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    k = K_BY_DATASET[dataset]
    run_dir = output_root / "runs" / dataset / variant / f"seed{seed}"
    checkpoint = run_dir / "best.pt"
    record_path = run_dir / "run_record.json"
    if resume and record_path.is_file() and checkpoint.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    model_mode = "separate_cos" if variant == "S0" else "multi_perspective_cos"
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        "num_runs=1",
        f"seed={seed}",
        f"device={device}",
        f"model.edge_weight_mode={model_mode}",
        "model.num_metric_perspectives=4",
        f"model.max_order={k}",
        f"model.num_layers={k}",
        "task.training_mode=full_graph",
        "task.optimizer=adamw",
        "task.epochs=300",
        "task.lr=1e-3",
        "task.weight_decay=1e-4",
        "task.patience=30",
        "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4",
        "task.eval_every=1",
        "task.grad_clip=1.0",
        "task.evaluate_test=true",
        "model.export_aux_stats=false",
        "model.export_node_aux=false",
        "model.export_edge_aux=false",
        f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={run_dir}",
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        command.append(f"dataset.nc_split_path={split_override}")

    started = time.monotonic()
    process_log = (run_dir / "process.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=process_log,
            stderr=subprocess.STDOUT,
        )
        peak = None
        while process.poll() is None:
            current = _peak_gpu_memory_mib(process.pid, device)
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(2.0)
        current = _peak_gpu_memory_mib(process.pid, device)
        if current is not None:
            peak = current if peak is None else max(peak, current)
    finally:
        process_log.close()
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"Formal U1 job failed ({dataset}, {variant}, seed={seed}) with "
            f"exit code {process.returncode}: {' '.join(command)}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected best checkpoint was not created: {checkpoint}")

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = dict(checkpoint_payload.get("metrics", {}))
    cfg = _compose_cfg(dataset, seed, variant, device, k)
    model = build_model(cfg, checkpoint_payload["data_info"])
    parameter_count = count_parameters(model) + int(data_info_num_classes(checkpoint_payload["data_info"])) * int(model.out_dim + 1)
    record = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "k": k,
        "device": device,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "downstream": {
            "val_acc": metrics.get("val_acc"),
            "val_macro_f1": metrics.get("val_macro_f1"),
            "test_acc": metrics.get("test_acc"),
            "test_macro_f1": metrics.get("test_macro_f1"),
        },
        "best_epoch": checkpoint_payload.get("epoch"),
        "parameter_count": parameter_count,
        "training_time_seconds": elapsed,
        "peak_gpu_memory_mib": peak,
        "peak_gpu_memory_status": "available" if peak is not None else "unavailable",
    }
    _dump_json(record_path, record)
    return record


def data_info_num_classes(data_info: dict[str, Any]) -> int:
    value = data_info.get("num_classes")
    if value is None:
        raise ValueError("Checkpoint data_info does not contain num_classes")
    return int(value)


def _distribution(values: np.ndarray, *, entropy: bool = False) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "std", "min", "max", "q10", "q25", "q50", "q75", "q90")}
    output = {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    for quantile in (10, 25, 50, 75, 90):
        output[f"q{quantile}"] = float(np.quantile(values, quantile / 100.0))
    return output


def _entropy_summary(src: np.ndarray, conductance: np.ndarray, num_nodes: int) -> dict[str, float]:
    degree_weight = np.bincount(src, weights=conductance, minlength=num_nodes)
    probability = conductance / np.maximum(degree_weight[src], np.finfo(np.float64).tiny)
    entropy = np.bincount(
        src,
        weights=-probability * np.log(np.maximum(probability, np.finfo(np.float64).tiny)),
        minlength=num_nodes,
    )
    return {
        key: value
        for key, value in _distribution(entropy).items()
        if key in {"mean", "std", "q25", "q50", "q75"}
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata

        return np.asarray(rankdata(values, method="average"), dtype=np.float64)
    except ImportError:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = np.arange(values.size, dtype=np.float64)
        return ranks


def _correlation(first: np.ndarray, second: np.ndarray, spearman: bool = False) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return 1.0 if np.array_equal(first, second) else 0.0
    if spearman:
        first = _rankdata(first)
        second = _rankdata(second)
    return float(np.corrcoef(first, second)[0, 1])


def _coalesced_sparse(index: torch.Tensor, weight: torch.Tensor, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    keys = (
        index[0].cpu().numpy().astype(np.int64) * np.int64(num_nodes)
        + index[1].cpu().numpy().astype(np.int64)
    )
    values = weight.cpu().numpy().astype(np.float64)
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    values = values[order]
    if keys.size == 0:
        return keys, values
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    return keys[starts], np.add.reduceat(values, starts)


def _normalized_operator_difference(
    index_a: torch.Tensor,
    weight_a: torch.Tensor,
    index_b: torch.Tensor,
    weight_b: torch.Tensor,
    num_nodes: int,
) -> float:
    keys_a, values_a = _coalesced_sparse(index_a, weight_a, num_nodes)
    keys_b, values_b = _coalesced_sparse(index_b, weight_b, num_nodes)
    norm_a = float(np.sqrt(np.square(values_a).sum()))
    norm_b = float(np.sqrt(np.square(values_b).sum()))
    common_a, positions_a, positions_b = np.intersect1d(
        keys_a, keys_b, assume_unique=True, return_indices=True
    )
    del common_a
    dot = float(np.dot(values_a[positions_a], values_b[positions_b])) if positions_a.size else 0.0
    numerator = math.sqrt(max(norm_a * norm_a + norm_b * norm_b - 2.0 * dot, 0.0))
    return float(numerator / (norm_a + norm_b + 1e-12))


def _label_aware(
    src: np.ndarray,
    dst: np.ndarray,
    labels: np.ndarray,
    conductance: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    valid = (
        (labels[src] >= 0)
        & (labels[src] < num_classes)
        & (labels[dst] >= 0)
        & (labels[dst] < num_classes)
    )
    same = valid & (labels[src] == labels[dst])
    different = valid & ~same
    same_values = conductance[same]
    different_values = conductance[different]
    same_mean = float(same_values.mean()) if same_values.size else float("nan")
    different_mean = float(different_values.mean()) if different_values.size else float("nan")
    pooled = math.sqrt(
        max(
            (
                max(same_values.size - 1, 0) * float(same_values.var(ddof=0) if same_values.size else 0.0)
                + max(different_values.size - 1, 0) * float(different_values.var(ddof=0) if different_values.size else 0.0)
            )
            / max(same_values.size + different_values.size - 2, 1),
            0.0,
        )
    )
    return {
        "valid_edge_count": int(valid.sum()),
        "same_label_edge_count": int(same_values.size),
        "different_label_edge_count": int(different_values.size),
        "same_label_mean": same_mean,
        "same_label_median": float(np.median(same_values)) if same_values.size else float("nan"),
        "different_label_mean": different_mean,
        "different_label_median": float(np.median(different_values)) if different_values.size else float("nan"),
        "effect_size_same_minus_different": float((same_mean - different_mean) / pooled) if pooled > 0.0 else float("nan"),
    }


def _perspective_specialization(model, stats: dict[str, torch.Tensor]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for modality, score_key in (
        ("text", "perspective_cos_text"),
        ("visual", "perspective_cos_visual"),
    ):
        weights = model.metric_perspective_weights(modality).cpu().numpy().astype(np.float64)
        scores = stats[score_key].cpu().numpy().astype(np.float64)
        aggregate = scores.mean(axis=0)
        perspective_count = int(scores.shape[0])
        similarity = np.zeros((perspective_count, perspective_count), dtype=np.float64)
        for left in range(perspective_count):
            for right in range(perspective_count):
                similarity[left, right] = _correlation(weights[left], weights[right])
        mean_by_dimension = weights.mean(axis=0)
        std_by_dimension = weights.std(axis=0, ddof=0)
        coefficient_of_variation = std_by_dimension / np.maximum(np.abs(mean_by_dimension), 1e-12)
        score_correlation = np.ones((perspective_count, perspective_count), dtype=np.float64)
        if perspective_count > 1:
            score_correlation = np.corrcoef(scores)
            score_correlation = np.nan_to_num(score_correlation, nan=0.0)
        output[modality] = {
            "perspective_count": perspective_count,
            "weight_pairwise_cosine_similarity": similarity.tolist(),
            "weight_mean_by_dimension": mean_by_dimension.tolist(),
            "weight_std_by_dimension": std_by_dimension.tolist(),
            "weight_coefficient_of_variation_by_dimension": coefficient_of_variation.tolist(),
            "score_pairwise_correlation": score_correlation.tolist(),
            "score_mean_absolute_deviation_from_aggregate": np.abs(scores - aggregate).mean(axis=1).tolist(),
            "score_mean_absolute_deviation": float(np.abs(scores - aggregate).mean()),
        }
    return output


def _mechanism_diagnostics(
    *,
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    k = K_BY_DATASET[dataset]
    cfg = _compose_cfg(dataset, seed, variant, device, k)
    data = load_mag_data(cfg, "nc", seed)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    stats = model.conductance_stats(x, edge_index)
    src = stats["src"].cpu().numpy().astype(np.int64)
    dst = stats["dst"].cpu().numpy().astype(np.int64)
    conductance_text = stats["conductance_text"].cpu().numpy().astype(np.float64)
    conductance_visual = stats["conductance_visual"].cpu().numpy().astype(np.float64)
    cos_text = stats["cos_text"].cpu().numpy().astype(np.float64)
    cos_visual = stats["cos_visual"].cpu().numpy().astype(np.float64)
    labels = data.y.cpu().numpy().astype(np.int64) if data.y is not None else np.full(data.num_nodes, -1)
    mechanism = {
        "conductance_distributions": {
            "text": _distribution(conductance_text),
            "visual": _distribution(conductance_visual),
        },
        "modality_conductance_gap": {
            "mean_abs": float(np.abs(conductance_text - conductance_visual).mean()),
            "median_abs": float(np.median(np.abs(conductance_text - conductance_visual))),
            "pearson": _correlation(conductance_text, conductance_visual),
            "spearman": _correlation(conductance_text, conductance_visual, spearman=True),
        },
        "normalized_operator_difference": _normalized_operator_difference(
            stats["norm_text_index"],
            stats["norm_text_weight"],
            stats["norm_visual_index"],
            stats["norm_visual_weight"],
            int(data.num_nodes),
        ),
        "neighborhood_entropy": {
            "text": _entropy_summary(src, conductance_text, int(data.num_nodes)),
            "visual": _entropy_summary(src, conductance_visual, int(data.num_nodes)),
        },
        "label_aware_analysis": {
            "text": _label_aware(src, dst, labels, conductance_text, int(data.num_classes)),
            "visual": _label_aware(src, dst, labels, conductance_visual, int(data.num_classes)),
        },
        "perspective_specialization": _perspective_specialization(model, stats),
        "pathology_checks": {
            "all_finite": bool(
                np.isfinite(conductance_text).all()
                and np.isfinite(conductance_visual).all()
                and np.isfinite(cos_text).all()
                and np.isfinite(cos_visual).all()
            ),
            "within_conductance_bounds": bool(
                conductance_text.min(initial=0.0) >= 0.1 - 1e-6
                and conductance_visual.min(initial=0.0) >= 0.1 - 1e-6
                and conductance_text.max(initial=1.0) <= 1.0 + 1e-6
                and conductance_visual.max(initial=1.0) <= 1.0 + 1e-6
            ),
            "text_conductance_collapsed": bool(np.std(conductance_text) < 1e-8),
            "visual_conductance_collapsed": bool(np.std(conductance_visual) < 1e-8),
            "original_edge_count": int(src.size),
            "normalized_text_edge_count": int(stats["norm_text_index"].size(1)),
            "normalized_visual_edge_count": int(stats["norm_visual_index"].size(1)),
        },
    }
    del model, data, x, edge_index, stats
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return mechanism


def _initialization_probe(dataset: str, device: str) -> dict[str, Any]:
    k = K_BY_DATASET[dataset]
    cfg_s0 = _compose_cfg(dataset, 42, "S0", device, k)
    cfg_s1 = _compose_cfg(dataset, 42, "S1", device, k)
    set_seed(42)
    data = load_mag_data(cfg_s0, "nc", 42)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    s0 = build_model(cfg_s0, data_info).to(device)
    set_seed(42)
    s1 = build_model(cfg_s1, data_info).to(device)
    s1.load_state_dict(s0.state_dict(), strict=False)
    s0.eval()
    s1.eval()
    with torch.no_grad():
        ordinary = s0.conductance_stats(data.x.to(device), data.edge_index.to(device))
        multi = s1.conductance_stats(data.x.to(device), data.edge_index.to(device))
    result: dict[str, Any] = {
        "dataset": dataset,
        "seed": 42,
        "k": k,
        "thresholds": {"pearson_gt": 0.99, "spearman_gt": 0.99},
        "modalities": {},
    }
    passed = True
    for modality in ("text", "visual"):
        baseline = ordinary[f"cos_{modality}"].cpu().numpy().astype(np.float64)
        perspective = multi[f"perspective_cos_{modality}"].mean(dim=0).cpu().numpy().astype(np.float64)
        item = {
            "pearson": _correlation(baseline, perspective),
            "spearman": _correlation(baseline, perspective, spearman=True),
            "mean_absolute_difference": float(np.abs(baseline - perspective).mean()),
            "max_absolute_difference": float(np.abs(baseline - perspective).max(initial=0.0)),
        }
        item["passed"] = bool(item["pearson"] > 0.99 and item["spearman"] > 0.99)
        passed = passed and item["passed"]
        result["modalities"][modality] = item
    result["passed"] = passed
    del s0, s1, data, ordinary, multi
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _numeric_summary(values: list[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "population_std": None}
    mean, std = mean_std(clean)
    return {"mean": mean, "population_std": std}


def _aggregates(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    downstream_keys = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    mechanism_keys = (
        ("modality_conductance_gap", "mean_abs"),
        ("modality_conductance_gap", "median_abs"),
        ("modality_conductance_gap", "pearson"),
        ("modality_conductance_gap", "spearman"),
        ("normalized_operator_difference", None),
        ("perspective_specialization", "text", "score_mean_absolute_deviation"),
        ("perspective_specialization", "visual", "score_mean_absolute_deviation"),
    )
    for dataset in DATASETS:
        result[dataset] = {}
        for variant in VARIANTS:
            selected = [row for row in records if row["dataset"] == dataset and row["variant"] == variant]
            downstream = {
                key: _numeric_summary([row["downstream"].get(key) for row in selected])
                for key in downstream_keys
            }
            downstream["best_epoch"] = _numeric_summary([row.get("best_epoch") for row in selected])
            downstream["parameter_count"] = _numeric_summary([row.get("parameter_count") for row in selected])
            mechanism = [row["mechanism"] for row in selected]
            mechanism_summary: dict[str, Any] = {}
            for path in mechanism_keys:
                if path[0] == "modality_conductance_gap":
                    values = [item["modality_conductance_gap"][path[1]] for item in mechanism]
                    mechanism_summary[f"modality_conductance_gap_{path[1]}"] = _numeric_summary(values)
                elif path[0] == "normalized_operator_difference":
                    mechanism_summary["normalized_operator_difference"] = _numeric_summary(
                        [item["normalized_operator_difference"] for item in mechanism]
                    )
                else:
                    modality = path[1]
                    values = [item["perspective_specialization"][modality][path[2]] for item in mechanism]
                    mechanism_summary[f"{modality}_{path[2]}"] = _numeric_summary(values)
            result[dataset][variant] = {
                "seed_count": len(selected),
                "seeds": [row["seed"] for row in selected],
                "downstream": downstream,
                "mechanism": mechanism_summary,
            }
    return result


def _s1_minus_s0(records: list[dict[str, Any]], aggregates: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    for dataset in DATASETS:
        result[dataset] = {"per_seed": [], "aggregate": {}}
        for seed in SEEDS:
            s0 = next(row for row in records if row["dataset"] == dataset and row["variant"] == "S0" and row["seed"] == seed)
            s1 = next(row for row in records if row["dataset"] == dataset and row["variant"] == "S1" and row["seed"] == seed)
            result[dataset]["per_seed"].append(
                {
                    "seed": seed,
                    **{
                        key: float(s1["downstream"][key] - s0["downstream"][key])
                        for key in keys
                    },
                }
            )
        for key in keys:
            values = [item[key] for item in result[dataset]["per_seed"]]
            result[dataset]["aggregate"][key] = _numeric_summary(values)
            result[dataset]["aggregate"][f"{key}_s1_minus_s0_mean"] = float(np.mean(values))
    return result


def _candidate_decision(summary: dict[str, Any]) -> dict[str, Any]:
    aggregates = summary["aggregates"]
    per_run = summary["per_run"]
    specialization_positive = {}
    structural_change = {}
    downstream_ok = {}
    pathology_ok = True
    for dataset in DATASETS:
        s0 = aggregates[dataset]["S0"]
        s1 = aggregates[dataset]["S1"]
        downstream_ok[dataset] = bool(
            s1["downstream"]["val_acc"]["mean"] >= s0["downstream"]["val_acc"]["mean"] - 0.01
        )
        specialization_positive[dataset] = bool(
            all(
                row["mechanism"]["perspective_specialization"][modality]["score_mean_absolute_deviation"] > 1e-5
                for row in per_run
                if row["dataset"] == dataset and row["variant"] == "S1"
                for modality in ("text", "visual")
            )
        )
        structural_change[dataset] = bool(
            abs(
                s1["mechanism"]["modality_conductance_gap_mean_abs"]["mean"]
                - s0["mechanism"]["modality_conductance_gap_mean_abs"]["mean"]
            ) > 1e-5
            or abs(
                s1["mechanism"]["normalized_operator_difference"]["mean"]
                - s0["mechanism"]["normalized_operator_difference"]["mean"]
            ) > 1e-5
        )
        for row in per_run:
            if row["dataset"] == dataset and row["variant"] == "S1":
                checks = row["mechanism"]["pathology_checks"]
                pathology_ok = pathology_ok and checks["all_finite"] and checks["within_conductance_bounds"] and not checks["text_conductance_collapsed"] and not checks["visual_conductance_collapsed"]
    specialized_count = sum(specialization_positive.values())
    structural_count = sum(structural_change.values())
    if pathology_ok and specialized_count >= 3 and all(downstream_ok.values()) and structural_count >= 3:
        decision = "Strong candidate"
    elif pathology_ok and all(downstream_ok.values()):
        decision = "Conditional candidate"
    else:
        decision = "Reject"
    return {
        "decision": decision,
        "criteria": {
            "validation_same_band_tolerance": 0.01,
            "specialization_threshold_mean_abs_score_deviation": 1e-5,
            "specialization_positive_dataset_count": specialized_count,
            "structurally_changed_dataset_count": structural_count,
            "pathology_ok": pathology_ok,
            "downstream_same_band_by_dataset": downstream_ok,
            "specialization_by_dataset": specialization_positive,
            "structural_change_by_dataset": structural_change,
        },
        "rationale": (
            "This classification is a descriptive decision rule over the five "
            "datasets and three seeds. It does not claim statistical significance; "
            "entropy, operator distance, and perspective similarity are interpreted "
            "as mechanism diagnostics rather than as monotone objectives."
        ),
    }


def _write_master_table(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for item in summary["per_run"]:
        row = {
            "dataset": item["dataset"],
            "variant": item["variant"],
            "seed": item["seed"],
            "k": item["k"],
            "val_acc": item["downstream"].get("val_acc"),
            "val_macro_f1": item["downstream"].get("val_macro_f1"),
            "test_acc": item["downstream"].get("test_acc"),
            "test_macro_f1": item["downstream"].get("test_macro_f1"),
            "best_epoch": item.get("best_epoch"),
            "parameter_count": item.get("parameter_count"),
            "training_time_seconds": item.get("training_time_seconds"),
            "peak_gpu_memory_mib": item.get("peak_gpu_memory_mib"),
        }
        for modality in ("text", "visual"):
            row[f"conductance_{modality}_mean"] = item["mechanism"]["conductance_distributions"][modality]["mean"]
            row[f"conductance_{modality}_std"] = item["mechanism"]["conductance_distributions"][modality]["std"]
            row[f"perspective_{modality}_score_mad"] = item["mechanism"]["perspective_specialization"][modality]["score_mean_absolute_deviation"]
        row["conductance_gap_mean_abs"] = item["mechanism"]["modality_conductance_gap"]["mean_abs"]
        row["conductance_gap_median_abs"] = item["mechanism"]["modality_conductance_gap"]["median_abs"]
        row["conductance_gap_pearson"] = item["mechanism"]["modality_conductance_gap"]["pearson"]
        row["conductance_gap_spearman"] = item["mechanism"]["modality_conductance_gap"]["spearman"]
        row["normalized_operator_difference"] = item["mechanism"]["normalized_operator_difference"]
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["candidate_decision"]
    lines = [
        "# MoPF-vNext U1 — Modality-Conditioned Semantic Conductance",
        "",
        "This report covers NC only: S0 Frozen Separate Cosine and S1 Multi-Perspective Semantic Conductance.",
        "LP, sports-copurchase, the decoder, sampler, and LP protocol are intentionally excluded.",
        "",
        f"- Frozen reference behavior commit: `{summary['frozen_reference_commit']}`",
        f"- vNext commit at analysis time: `{summary['vnext_commit']}`",
        f"- Test status: `{summary['git_clean_status']}`",
        f"- Candidate decision: **{decision['decision']}**",
        "",
        "## Downstream comparison",
        "",
        "The authoritative numerical table is `outputs/u1_semantic_conductance/u1_master_table.csv`;",
        "the JSON contains the per-seed records, population standard deviations, and S1-minus-S0 values.",
        "Checkpoint selection is best validation accuracy only. No test metric is used for selection.",
        "",
        "| Dataset | S0 Val Acc | S1 Val Acc | S1-S0 Val Acc | S0 Test Acc | S1 Test Acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        s0 = summary["aggregates"][dataset]["S0"]["downstream"]
        s1 = summary["aggregates"][dataset]["S1"]["downstream"]
        delta = summary["s1_minus_s0"][dataset]["aggregate"]["val_acc_s1_minus_s0_mean"]
        lines.append(
            f"| {dataset} | {s0['val_acc']['mean']:.6f} | {s1['val_acc']['mean']:.6f} | {delta:+.6f} | "
            f"{s0['test_acc']['mean']:.6f} | {s1['test_acc']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism interpretation",
            "",
            "The conductance distributions, modality gaps, normalized operator differences, neighborhood entropy,",
            "label-aware edge summaries, and perspective diagnostics are reported without a preferred direction.",
            "For heterophilous or non-label-homogeneous graphs, same-label conductance is not assumed to be higher.",
            "",
            "## Candidate rationale",
            "",
            decision["rationale"],
            "",
            "The decision is a screening label, not a statistical significance claim.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/u1_semantic_conductance")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    devices = [str(device) for device in args.devices]
    probe = {}
    for index, dataset in enumerate(args.datasets):
        probe[dataset] = _initialization_probe(dataset, devices[index % len(devices)])
        print(f"[U1] initialization probe {dataset}: passed={probe[dataset]['passed']}", flush=True)
    if not all(item["passed"] for item in probe.values()):
        payload = {
            "initialization_equivalence": probe,
            "error": "Initialization equivalence failed; formal U1 training was not started.",
        }
        _dump_json(args.output_root / "initialization_equivalence_failed.json", payload)
        raise RuntimeError(payload["error"])
    if args.probe_only:
        _dump_json(args.output_root / "initialization_equivalence.json", {"initialization_equivalence": probe})
        return

    records: list[dict[str, Any]] = []
    jobs = [
        (dataset, variant, seed)
        for dataset in args.datasets
        for variant in args.variants
        for seed in args.seeds
    ]
    for job_index, (dataset, variant, seed) in enumerate(jobs):
        device = devices[job_index % len(devices)]
        print(
            f"[U1] training {job_index + 1}/{len(jobs)} dataset={dataset} "
            f"variant={variant} seed={seed} device={device}",
            flush=True,
        )
        record = _run_formal_job(
            dataset=dataset,
            variant=variant,
            seed=seed,
            device=device,
            output_root=args.output_root,
            resume=not args.no_resume,
        )
        print(f"[U1] diagnostics dataset={dataset} variant={variant} seed={seed}", flush=True)
        record["mechanism"] = _mechanism_diagnostics(
            dataset=dataset,
            variant=variant,
            seed=seed,
            device=device,
            checkpoint_path=Path(record["checkpoint"]),
        )
        _dump_json(Path(record["run_dir"]) / "mechanism_diagnostics.json", record["mechanism"])
        records.append(record)

    records.sort(key=lambda item: (item["dataset"], item["variant"], item["seed"]))
    summary: dict[str, Any] = {
        "metadata": {
            "stage": "MoPF-vNext U1 Semantic Conductance Study",
            "scope": "NC only",
            "generated_at_unix": time.time(),
            "script": str(Path(__file__).relative_to(ROOT)),
            "no_lp_experiments": True,
            "no_sports_copurchase": True,
        },
        "frozen_reference_commit": _git("rev-parse", "mopf-v0-frozen"),
        "vnext_commit": _git("rev-parse", "HEAD"),
        "git_clean_status": _git("status", "--short") or "clean",
        "variants": list(args.variants),
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "K": {dataset: K_BY_DATASET[dataset] for dataset in args.datasets},
        "protocol": {
            "name": "unified_full_graph_nc_v1",
            "training_mode": "full_graph",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 300,
            "patience": 30,
            "dropout": 0.2,
            "hidden_dim": 256,
            "checkpoint_selection": "best_validation_accuracy",
            "macro_f1": "fixed labels from supervised train/val/test union, zero_division=0",
            "test_metrics_not_used_for_selection": True,
        },
        "initialization_equivalence": probe,
        "per_run": records,
    }
    summary["aggregates"] = _aggregates(records)
    summary["s1_minus_s0"] = _s1_minus_s0(records, summary["aggregates"])
    summary["candidate_decision"] = _candidate_decision(summary)
    summary["artifacts"] = {
        "summary": str(args.output_root / "u1_master_summary.json"),
        "master_table": str(args.output_root / "u1_master_table.csv"),
        "report": str(ROOT / "docs/mopf_u1_semantic_conductance.md"),
    }
    _dump_json(args.output_root / "u1_master_summary.json", summary)
    _write_master_table(args.output_root / "u1_master_table.csv", summary)
    _write_report(ROOT / "docs/mopf_u1_semantic_conductance.md", summary)
    print(f"[U1] summary={args.output_root / 'u1_master_summary.json'}", flush=True)
    print(f"[U1] decision={summary['candidate_decision']['decision']}", flush=True)


if __name__ == "__main__":
    main()
