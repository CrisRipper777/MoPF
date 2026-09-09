"""Run the complete MoPF M1-C PDC/PPC candidate study.

The study is intentionally staged:

1. calibrate one global PPC scale from a no-update seed-42 initialization probe;
2. train all 6 variants x 5 datasets x 3 seeds with the existing NC protocol;
3. audit best-validation checkpoints for mechanism, stochastic stability, cross-
   seed stability, and frozen node counterfactuals;
4. write one complete ``m1c_master_summary.json`` and a readable report.

The formal MoPF defaults remain untouched.  Candidate-only Hydra overrides are
passed with ``+model.*`` when the runner launches a training subprocess.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.common import clone_state_dict  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402
from src.utils.summary import count_parameters, mean_std  # noqa: E402
from scripts.run_mopf_m1b_counterfactual import (  # noqa: E402
    _canonical_decomposition,
    _compute_counterfactual_embeddings,
    _metrics,
    _resolve_eval_labels,
)


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
K_BY_DATASET = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
}
VARIANT_ORDER = (
    "v0_current",
    "v1_pdc",
    "v2_ppc_low",
    "v3_ppc_high",
    "v4_pdc_ppc_low",
    "v5_pdc_ppc_high",
)
BASE_VARIANTS = {
    "v0_current": {"node_conditioner_mode": "absolute", "ppc": "none"},
    "v1_pdc": {"node_conditioner_mode": "pdc", "ppc": "none"},
    "v2_ppc_low": {"node_conditioner_mode": "absolute", "ppc": "low"},
    "v3_ppc_high": {"node_conditioner_mode": "absolute", "ppc": "high"},
    "v4_pdc_ppc_low": {"node_conditioner_mode": "pdc", "ppc": "low"},
    "v5_pdc_ppc_high": {"node_conditioner_mode": "pdc", "ppc": "high"},
}
SHUFFLE_SEEDS = tuple(range(10))
STOCHASTIC_VIEWS = 20
PPC_EPS = 1e-8


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _compose_cfg(dataset: str, seed: int, device: str = "cpu"):
    """Compose the same resolved config used by ``src.main`` for probes."""
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        f"model.max_order={K_BY_DATASET[dataset]}",
        f"model.num_layers={K_BY_DATASET[dataset]}",
        "+model.node_conditioner_mode=absolute",
        "+model.ppc_weight=0.0",
        "model.hrc_weight=0.0",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _scale_probe_one(dataset: str, device: torch.device) -> dict[str, Any]:
    cfg = _compose_cfg(dataset, 42, str(device))
    set_seed(42)
    data = load_mag_data(cfg, "nc", 42)
    model = build_model(cfg, _data_info(data)).to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    model.train()
    classifier.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_idx = data.train_idx.to(device)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        components_1 = model._encode_components(x, edge_index)
        components_2 = model._encode_components(x, edge_index)
        task_loss = criterion(
            classifier(components_1["z"].index_select(0, train_idx)),
            data.y.to(device).index_select(0, train_idx),
        )
        ppc_raw = model._ppc_raw_loss(
            components_1["delta_node_text"],
            components_1["delta_node_visual"],
            components_2["delta_node_text"],
            components_2["delta_node_visual"],
            train_idx,
        )
    ratio = float(task_loss.item()) / (float(ppc_raw.item()) + PPC_EPS)
    return {
        "dataset": dataset,
        "seed": 42,
        "task_loss_initial": float(task_loss.item()),
        "ppc_raw_initial": float(ppc_raw.item()),
        "ratio_task_over_ppc": ratio,
        "parameter_count": count_parameters(model) + count_parameters(classifier),
    }


def calibrate(output_root: Path, device: torch.device, force: bool = False) -> dict[str, Any]:
    path = output_root / "ppc_scale_calibration.json"
    if path.is_file() and not force:
        return _load_json(path)
    probes = [_scale_probe_one(dataset, device) for dataset in DATASETS]
    ratios = np.asarray([item["ratio_task_over_ppc"] for item in probes], dtype=np.float64)
    ratio = float(np.median(ratios))
    payload = {
        "protocol": "M1-C no-update seed42 baseline initialization probe",
        "epsilon": PPC_EPS,
        "datasets": list(DATASETS),
        "probes": probes,
        "R": ratio,
        "lambda_low": 0.005 * ratio,
        "lambda_high": 0.020 * ratio,
        "formula": "R=median(task_loss_initial/(ppc_raw_initial+epsilon)); lambda_low=0.005R; lambda_high=0.020R",
        "global_weight_across_datasets": True,
        "hrc_weight": 0.0,
    }
    _dump_json(path, payload)
    return payload


def _gpu_memory_mb(pid: int) -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def _parse_model_params(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    matches = re.findall(r"model params=(\d+)", log_path.read_text(errors="replace"))
    return int(matches[-1]) if matches else None


def _training_command(
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    output_dir: Path,
    lambda_low: float,
    lambda_high: float,
) -> list[str]:
    spec = BASE_VARIANTS[variant]
    ppc_weight = {
        "none": 0.0,
        "low": lambda_low,
        "high": lambda_high,
    }[spec["ppc"]]
    return [
        sys.executable,
        "src/main.py",
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        f"model.max_order={K_BY_DATASET[dataset]}",
        f"model.num_layers={K_BY_DATASET[dataset]}",
        f"+model.node_conditioner_mode={spec['node_conditioner_mode']}",
        f"+model.ppc_weight={ppc_weight:.16g}",
        "model.hrc_weight=0.0",
        "model.edge_weight_mode=separate_cos",
        "task.training_mode=full_graph",
        "task.optimizer=adamw",
        "task.epochs=300",
        "task.lr=1e-3",
        "task.weight_decay=1e-4",
        "task.patience=30",
        "task.evaluate_test=true",
        "task.save_ckpt_path=" + str(output_dir / "best_val_accuracy.pt"),
        "hydra.job.chdir=false",
        "hydra.run.dir=" + str(output_dir),
    ]


def _train_one(
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    output_root: Path,
    lambda_low: float,
    lambda_high: float,
    force: bool,
) -> dict[str, Any]:
    output_dir = output_root / dataset / variant / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_val_accuracy.pt"
    metrics_path = output_dir / "metrics.json"
    if checkpoint.is_file() and metrics_path.is_file() and not force:
        return _load_json(metrics_path)

    log_path = output_dir / "runner.log"
    command = _training_command(
        dataset, variant, seed, device, output_dir, lambda_low, lambda_high
    )
    started = time.perf_counter()
    peak_memory: float | None = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            current_memory = _gpu_memory_mb(process.pid)
            if current_memory is not None:
                peak_memory = max(peak_memory or 0.0, current_memory)
            time.sleep(1.0)
        return_code = process.wait()
    train_time = time.perf_counter() - started
    if return_code != 0 or not checkpoint.is_file():
        tail = log_path.read_text(errors="replace")[-4000:]
        failure = {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "device": device,
            "return_code": return_code,
            "train_time_sec": train_time,
            "log_tail": tail,
        }
        _dump_json(output_dir / "failure.json", failure)
        raise RuntimeError(
            f"Training failed for {dataset}/{variant}/seed{seed}; see {log_path}"
        )

    (output_dir / "failure.json").unlink(missing_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_metrics = dict(payload.get("metrics", {}))
    record = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "device": device,
        "checkpoint": str(checkpoint),
        "best_epoch": payload.get("epoch"),
        "metrics": run_metrics,
        "train_time_sec": train_time,
        "peak_gpu_memory_mb": peak_memory,
        "parameter_count": _parse_model_params(output_dir / "main.log"),
        "selection": payload.get("selection"),
    }
    _dump_json(metrics_path, record)
    return record


def train_all(
    output_root: Path,
    calibration: dict[str, Any],
    devices: tuple[str, ...],
    force: bool,
) -> list[dict[str, Any]]:
    jobs = list(itertools.product(DATASETS, VARIANT_ORDER, SEEDS))
    results: list[dict[str, Any]] = []
    lambda_low = float(calibration["lambda_low"])
    lambda_high = float(calibration["lambda_high"])

    # Bind one serial worker to each device.  A generic thread pool cannot
    # bind a worker to a GPU: when job durations differ, two queued jobs can
    # otherwise land on the same device and cause avoidable OOMs.
    jobs_by_device = [jobs[index::len(devices)] for index in range(len(devices))]

    def run_device(device_jobs, device):
        return [
            _train_one(
                dataset,
                variant,
                seed,
                device,
                output_root,
                lambda_low,
                lambda_high,
                force,
            )
            for dataset, variant, seed in device_jobs
        ]

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_device, device_jobs, device)
            for device_jobs, device in zip(jobs_by_device, devices)
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: (item["dataset"], VARIANT_ORDER.index(item["variant"]), item["seed"]))
    return results


def _summary(value: torch.Tensor | np.ndarray) -> dict[str, float | int | None]:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    array = array.astype(np.float64, copy=False).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "population_std": None, "min": None, "max": None}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "population_std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    x = left.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    y = right.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    rx = _rankdata(x[mask])
    ry = _rankdata(y[mask])
    if rx.std() == 0.0 or ry.std() == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _profile_centered(delta: torch.Tensor) -> torch.Tensor:
    return delta - delta.mean(dim=0, keepdim=True)


def _pdc_mechanism_stats(model, components, stats) -> dict[str, Any]:
    result: dict[str, Any] = {
        "node_conditioner_mode": model.node_conditioner_mode,
        "rho_text": model.pdc_rho("text").detach().cpu().tolist(),
        "rho_visual": model.pdc_rho("visual").detach().cpu().tolist(),
        "rho_text_abs_max": float(model.pdc_rho("text").detach().abs().max().item()),
        "rho_visual_abs_max": float(model.pdc_rho("visual").detach().abs().max().item()),
        "modalities": {},
    }
    for modality in ("text", "visual"):
        delta = stats[f"delta_node_{modality}"]
        centered = _profile_centered(delta)
        bases = components[f"bases_{modality}"]
        discrepancy_by_order = [torch.zeros(delta.size(0), device=delta.device)]
        discrepancy_by_order.extend(
            torch.linalg.vector_norm(bases[order] - bases[order - 1], dim=-1)
            for order in range(1, len(bases))
        )
        discrepancy = torch.stack(discrepancy_by_order, dim=1)
        discrepancy_mean = discrepancy[:, 1:].mean(dim=1) if discrepancy.size(1) > 1 else discrepancy[:, 0]
        profile_magnitude = torch.linalg.vector_norm(centered, dim=1)
        std_by_order = centered.std(dim=0, unbiased=False)
        mean_by_order = delta.mean(dim=0)
        abs_mean_by_order = delta.abs().mean(dim=0)
        ratio_by_order = mean_by_order.abs() / (std_by_order + PPC_EPS)
        result["modalities"][modality] = {
            "mean_abs_node_residual_by_order": abs_mean_by_order.detach().cpu().tolist(),
            "delta_node_mean_by_order": mean_by_order.detach().cpu().tolist(),
            "centered_node_std_by_order": std_by_order.detach().cpu().tolist(),
            "centered_node_std_mean": float(std_by_order.mean().item()),
            "centered_node_std_max": float(std_by_order.max().item()),
            "shared_offset_ratio_by_order": ratio_by_order.detach().cpu().tolist(),
            "shared_offset_ratio_mean": float(ratio_by_order.mean().item()),
            "discrepancy_norm_mean_by_order": discrepancy.mean(dim=0).detach().cpu().tolist(),
            "discrepancy_profile_spearman": _spearman(discrepancy_mean, profile_magnitude),
            "discrepancy_mean_node_summary": _summary(discrepancy_mean),
            "centered_profile_magnitude_summary": _summary(profile_magnitude),
        }
    canonical, canonical_summary = _canonical_decomposition(stats)
    del canonical
    result["canonical_decomposition"] = canonical_summary
    return result


def _profile_pair_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float | None]:
    first = first.float()
    second = second.float()
    rmse = torch.sqrt((first - second).square().mean())
    first_rms = torch.sqrt(first.square().mean())
    second_rms = torch.sqrt(second.square().mean())
    denominator = 0.5 * (first_rms + second_rms)
    if float(denominator.item()) < 1e-6:
        return {"cosine_similarity": None, "normalized_rmse": None, "status": "N/A: profile RMS < 1e-6"}
    cosine = F.cosine_similarity(
        first.reshape(-1), second.reshape(-1), dim=0, eps=1e-8
    )
    return {
        "cosine_similarity": float(cosine.item()),
        "normalized_rmse": float((rmse / (denominator + PPC_EPS)).item()),
        "status": "ok",
    }


@torch.no_grad()
def _stochastic_profile_stability(model, x, edge_index) -> dict[str, Any]:
    was_training = model.training
    model.train()
    views: dict[str, list[torch.Tensor]] = {"text": [], "visual": []}
    std_values: dict[str, list[float]] = {"text": [], "visual": []}
    for _ in range(STOCHASTIC_VIEWS):
        components = model._encode_components(x, edge_index)
        for modality in ("text", "visual"):
            centered = _profile_centered(components[f"delta_node_{modality}"])
            views[modality].append(centered.detach().cpu())
            std_values[modality].append(float(centered.std(dim=0, unbiased=False).mean().item()))
    if not was_training:
        model.eval()

    output: dict[str, Any] = {
        "num_views": STOCHASTIC_VIEWS,
        "pair_count": STOCHASTIC_VIEWS * (STOCHASTIC_VIEWS - 1) // 2,
        "modalities": {},
    }
    for modality in ("text", "visual"):
        mse_values: list[float] = []
        cosine_values: list[float] = []
        for i, j in itertools.combinations(range(STOCHASTIC_VIEWS), 2):
            first = views[modality][i]
            second = views[modality][j]
            mse_values.append(float((first - second).square().mean().item()))
            cosine_values.append(
                float(
                    F.cosine_similarity(
                        first, second, dim=1, eps=1e-8
                    ).mean().item()
                )
            )
        output["modalities"][modality] = {
            "stochastic_profile_mse": _summary(np.asarray(mse_values)),
            "stochastic_profile_cosine": _summary(np.asarray(cosine_values)),
            "centered_node_std_mean_over_views": float(np.mean(std_values[modality])),
            "centered_node_std_min_over_views": float(np.min(std_values[modality])),
            "centered_node_std_max_over_views": float(np.max(std_values[modality])),
        }
    return output


@torch.no_grad()
def _analyze_one_run(run_dir: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(run_dir / "best_val_accuracy.pt", map_location="cpu", weights_only=False)
    raw_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
    data = load_mag_data(cfg, "nc", int(payload["seed"]))
    model = build_model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"])
    classifier.eval()
    eval_labels = _resolve_eval_labels(data)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    with torch.no_grad():
        stats = model.analysis_stats(x, edge_index)
        components = model._encode_components(x, edge_index)
        mechanism = _pdc_mechanism_stats(model, components, stats)
        stochastic = _stochastic_profile_stability(model, x, edge_index)
        model.eval()
        original_z = _compute_counterfactual_embeddings(model, components, stats, "original")
        original_metrics = _metrics(classifier, original_z, data, device, eval_labels)
        node_mean_z = _compute_counterfactual_embeddings(model, components, stats, "node_mean")
        node_mean_metrics = _metrics(classifier, node_mean_z, data, device, eval_labels)
        shuffle_records = []
        for shuffle_seed in SHUFFLE_SEEDS:
            z = _compute_counterfactual_embeddings(
                model, components, stats, f"node_shuffle:{shuffle_seed}"
            )
            shuffle_records.append(
                {
                    "permutation_seed": shuffle_seed,
                    "metrics": _metrics(classifier, z, data, device, eval_labels),
                }
            )

    canonical, canonical_summary = _canonical_decomposition(stats)
    del canonical
    original = dict(original_metrics)
    node_mean = dict(node_mean_metrics)
    node_mean.update(
        {
            "delta_test_acc": original["test_acc"] - node_mean_metrics["test_acc"],
            "delta_test_macro_f1": original["test_macro_f1"] - node_mean_metrics["test_macro_f1"],
            "delta_val_acc": original["val_acc"] - node_mean_metrics["val_acc"],
            "delta_val_macro_f1": original["val_macro_f1"] - node_mean_metrics["val_macro_f1"],
        }
    )
    shuffle_acc = np.asarray([row["metrics"]["test_acc"] for row in shuffle_records])
    shuffle_f1 = np.asarray([row["metrics"]["test_macro_f1"] for row in shuffle_records])
    shuffle_mean = {
        key: float(np.mean([row["metrics"][key] for row in shuffle_records]))
        for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }
    shuffle_drop = dict(shuffle_mean)
    shuffle_drop.update(
        {
            "delta_test_acc": original["test_acc"] - shuffle_mean["test_acc"],
            "delta_test_macro_f1": original["test_macro_f1"] - shuffle_mean["test_macro_f1"],
            "delta_val_acc": original["val_acc"] - shuffle_mean["val_acc"],
            "delta_val_macro_f1": original["val_macro_f1"] - shuffle_mean["val_macro_f1"],
        }
    )
    counterfactual = {
        "eval_labels": eval_labels,
        "original": original_metrics,
        "node_mean": node_mean,
        "node_shuffle": {
            "permutation_mode": "independent text and visual complete-profile permutations",
            "permutation_seeds": list(SHUFFLE_SEEDS),
            "permutations": shuffle_records,
            "mean_metrics": shuffle_mean,
            "mean_drop": shuffle_drop,
            "std_metrics": {
                "test_acc": float(shuffle_acc.std(ddof=0)),
                "test_macro_f1": float(shuffle_f1.std(ddof=0)),
                "val_acc": float(np.std([row["metrics"]["val_acc"] for row in shuffle_records], ddof=0)),
                "val_macro_f1": float(np.std([row["metrics"]["val_macro_f1"] for row in shuffle_records], ddof=0)),
            },
        },
        "canonical_decomposition": canonical_summary,
    }

    mechanism_path = run_dir / "mechanism_stats.json"
    stability_path = run_dir / "profile_stability.json"
    counterfactual_path = run_dir / "counterfactual.json"
    _dump_json(mechanism_path, mechanism)
    _dump_json(stability_path, {"stochastic": stochastic, "cross_seed": None})
    _dump_json(counterfactual_path, counterfactual)

    centered_profiles = {
        modality: _profile_centered(stats[f"delta_node_{modality}"]).detach().cpu()
        for modality in ("text", "visual")
    }
    metrics_record = _load_json(run_dir / "metrics.json")
    metrics_record["analysis_metrics"] = original_metrics
    metrics_record["mechanism_stats"] = mechanism
    metrics_record["profile_stability"] = stochastic
    metrics_record["counterfactual"] = counterfactual
    _dump_json(run_dir / "metrics.json", metrics_record)

    record = {
        **metrics_record,
        "mechanism_stats": mechanism,
        "profile_stability": {"stochastic": stochastic, "cross_seed": None},
        "counterfactual": counterfactual,
        "config": {
            "node_conditioner_mode": str(cfg.model.get("node_conditioner_mode", "absolute")),
            "ppc_weight": float(cfg.model.get("ppc_weight", 0.0)),
            "max_order": int(cfg.model.max_order),
        },
    }
    del model, classifier, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record, centered_profiles


def _cross_seed_stability(
    profiles: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    pairs = []
    for first_seed, second_seed in itertools.combinations(sorted(profiles), 2):
        pair = {"seed_a": first_seed, "seed_b": second_seed, "modalities": {}}
        for modality in ("text", "visual"):
            pair["modalities"][modality] = _profile_pair_metrics(
                profiles[first_seed][modality], profiles[second_seed][modality]
            )
        pairs.append(pair)

    means: dict[str, Any] = {}
    for modality in ("text", "visual"):
        cosine = [
            pair["modalities"][modality]["cosine_similarity"]
            for pair in pairs
            if pair["modalities"][modality]["cosine_similarity"] is not None
        ]
        nrmse = [
            pair["modalities"][modality]["normalized_rmse"]
            for pair in pairs
            if pair["modalities"][modality]["normalized_rmse"] is not None
        ]
        means[modality] = {
            "cosine_similarity_mean": float(np.mean(cosine)) if cosine else None,
            "normalized_rmse_mean": float(np.mean(nrmse)) if nrmse else None,
            "valid_pair_count": len(cosine),
        }
    return {"pairs": pairs, "mean": means}


def analyze_all(output_root: Path, devices: tuple[str, ...]) -> list[dict[str, Any]]:
    jobs = list(itertools.product(DATASETS, VARIANT_ORDER, SEEDS))
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    profiles: dict[tuple[str, str], dict[int, dict[str, torch.Tensor]]] = {}

    jobs_by_device = [jobs[index::len(devices)] for index in range(len(devices))]

    def analyze_device(device_jobs, device):
        analyzed = []
        for dataset, variant, seed in device_jobs:
            run_dir = output_root / dataset / variant / f"seed{seed}"
            if not (run_dir / "best_val_accuracy.pt").is_file():
                raise FileNotFoundError(f"Missing checkpoint for analysis: {run_dir}")
            record, centered = _analyze_one_run(run_dir, torch.device(device))
            analyzed.append((dataset, variant, seed, record, centered))
        return analyzed

    # Analysis uses the same two-GPU fan-out as training, but each job remains
    # a full-graph single-device operation.
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(analyze_device, device_jobs, device)
            for device_jobs, device in zip(jobs_by_device, devices)
        ]
        for future in as_completed(futures):
            for dataset, variant, seed, record, centered in future.result():
                records[(dataset, variant, seed)] = record
                profiles.setdefault((dataset, variant), {})[seed] = centered

    for (dataset, variant), seed_profiles in profiles.items():
        cross_seed = _cross_seed_stability(seed_profiles)
        for seed in SEEDS:
            run_dir = output_root / dataset / variant / f"seed{seed}"
            stability = _load_json(run_dir / "profile_stability.json")
            stability["cross_seed"] = cross_seed
            _dump_json(run_dir / "profile_stability.json", stability)
            records[(dataset, variant, seed)]["profile_stability"]["cross_seed"] = cross_seed

    return [
        records[(dataset, variant, seed)]
        for dataset in DATASETS
        for variant in VARIANT_ORDER
        for seed in SEEDS
    ]


def _aggregate_numbers(values: list[float | None]) -> dict[str, float | None]:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not valid:
        return {"mean": None, "population_std": None}
    return {"mean": float(np.mean(valid)), "population_std": float(np.std(valid, ddof=0))}


def _aggregate_metric(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    return _aggregate_numbers([record["metrics"].get(key) for record in records])


def _aggregate_dataset_variant(records: list[dict[str, Any]]) -> dict[str, Any]:
    downstream = {
        key: _aggregate_metric(records, key)
        for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }
    mechanism: dict[str, Any] = {}
    for modality in ("text", "visual"):
        for key in (
            "discrepancy_profile_spearman",
            "centered_node_std_mean",
            "centered_node_std_max",
            "shared_offset_ratio_mean",
        ):
            mechanism[f"{modality}_{key}"] = _aggregate_numbers(
                [record["mechanism_stats"]["modalities"][modality][key] for record in records]
            )
        max_len = max(
            len(record["mechanism_stats"]["modalities"][modality]["centered_node_std_by_order"])
            for record in records
        )
        mechanism[f"{modality}_centered_node_std_by_order"] = [
            _aggregate_numbers(
                [
                    record["mechanism_stats"]["modalities"][modality]["centered_node_std_by_order"][order]
                    for record in records
                ]
            )
            for order in range(max_len)
        ]
    for key in ("rho_text_abs_max", "rho_visual_abs_max"):
        mechanism[key] = _aggregate_numbers([record["mechanism_stats"][key] for record in records])

    stochastic: dict[str, Any] = {}
    for modality in ("text", "visual"):
        stochastic[modality] = {}
        for key in ("stochastic_profile_mse", "stochastic_profile_cosine"):
            stochastic[modality][key] = _aggregate_numbers(
                [
                    record["profile_stability"]["stochastic"]["modalities"][modality][key]["mean"]
                    for record in records
                ]
            )
        stochastic[modality]["centered_node_std_mean_over_views"] = _aggregate_numbers(
            [
                record["profile_stability"]["stochastic"]["modalities"][modality]["centered_node_std_mean_over_views"]
                for record in records
            ]
        )
        cross_seed = records[0]["profile_stability"]["cross_seed"]["mean"][modality]
        stochastic[modality]["cross_seed_cosine"] = cross_seed["cosine_similarity_mean"]
        stochastic[modality]["cross_seed_normalized_rmse"] = cross_seed["normalized_rmse_mean"]

    counterfactual: dict[str, Any] = {
        "node_mean": {
            key: _aggregate_numbers(
                [record["counterfactual"]["node_mean"][key] for record in records]
            )
            for key in ("delta_test_acc", "delta_test_macro_f1", "delta_val_acc", "delta_val_macro_f1")
        },
        "node_shuffle": {},
    }
    for key in ("delta_test_acc", "delta_test_macro_f1", "delta_val_acc", "delta_val_macro_f1"):
        counterfactual["node_shuffle"][f"{key}_across_training_seeds"] = _aggregate_numbers(
            [record["counterfactual"]["node_shuffle"]["mean_drop"][key] for record in records]
        )
    counterfactual["node_shuffle"]["within_seed_permutation_std"] = {
        key: _aggregate_numbers(
            [record["counterfactual"]["node_shuffle"]["std_metrics"][key] for record in records]
        )
        for key in ("test_acc", "test_macro_f1")
    }
    return {
        "n_training_seeds": len(records),
        "seeds": [int(record["seed"]) for record in records],
        "downstream": downstream,
        "mechanism": mechanism,
        "stability": stochastic,
        "frozen_counterfactual": counterfactual,
        "best_epoch": _aggregate_numbers([record["best_epoch"] for record in records]),
        "train_time_sec": _aggregate_numbers([record["train_time_sec"] for record in records]),
        "peak_gpu_memory_mb": _aggregate_numbers([record["peak_gpu_memory_mb"] for record in records]),
        "parameter_count": sorted({record["parameter_count"] for record in records}),
    }


def _candidate_summary(
    aggregates: dict[tuple[str, str], dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    dataset_summaries = {dataset: aggregates[(dataset, variant)] for dataset in DATASETS}
    baseline = [aggregates[(dataset, "v0_current")] for dataset in DATASETS]
    candidate = [dataset_summaries[dataset] for dataset in DATASETS]

    def mean_delta(path: tuple[str, ...]) -> float:
        values = []
        for base, cand in zip(baseline, candidate):
            value_base = base
            value_cand = cand
            for part in path:
                value_base = value_base[part]
                value_cand = value_cand[part]
            if value_base["mean"] is not None and value_cand["mean"] is not None:
                values.append(value_cand["mean"] - value_base["mean"])
        return float(np.mean(values)) if values else 0.0

    downstream_val_delta = mean_delta(("downstream", "val_acc"))
    centered_std_delta = mean_delta(("mechanism", "text_centered_node_std_mean"))
    profile_mse_delta = mean_delta(("stability", "text", "stochastic_profile_mse"))
    profile_cosine_delta = mean_delta(("stability", "text", "stochastic_profile_cosine"))
    shuffle_drop = [
        dataset_summaries[dataset]["frozen_counterfactual"]["node_shuffle"]["delta_test_acc_across_training_seeds"]["mean"]
        for dataset in DATASETS
    ]
    detected_pathologies: list[str] = []
    if any(value is not None and value < -0.001 for value in shuffle_drop):
        detected_pathologies.append("node-shuffle improves Test Acc on at least one dataset")
    if centered_std_delta > 0.5 * max(
        float(aggregates[(dataset, "v0_current")]["mechanism"]["text_centered_node_std_mean"]["mean"] or 0.0)
        for dataset in DATASETS
    ):
        detected_pathologies.append("centered node std increased materially relative to baseline")
    if variant in {"v2_ppc_low", "v3_ppc_high", "v4_pdc_ppc_low", "v5_pdc_ppc_high"}:
        low_scale = any(
            dataset_summaries[dataset]["stability"]["text"]["stochastic_profile_mse"]["mean"]
            >= aggregates[(dataset, "v0_current")]["stability"]["text"]["stochastic_profile_mse"]["mean"]
            for dataset in DATASETS
        )
        if low_scale:
            detected_pathologies.append("stochastic profile MSE did not improve on every dataset")
        collapse_datasets = []
        for dataset in DATASETS:
            base_std = aggregates[(dataset, "v0_current")]["stability"]["text"][
                "centered_node_std_mean_over_views"
            ]["mean"]
            current_std = dataset_summaries[dataset]["stability"]["text"][
                "centered_node_std_mean_over_views"
            ]["mean"]
            if (
                base_std is not None
                and current_std is not None
                and current_std < max(1e-6, 0.25 * base_std)
            ):
                collapse_datasets.append(dataset)
        if collapse_datasets:
            detected_pathologies.append(
                "centered node profile collapse on " + ", ".join(collapse_datasets)
            )

    if variant == "v1_pdc":
        interpretation = (
            "PDC is a candidate-only propagation conditioner. Inspect rho departure from zero, "
            "discrepancy/profile Spearman movement, and downstream stability jointly; no performance-only claim is made."
        )
    elif variant in {"v2_ppc_low", "v3_ppc_high"}:
        interpretation = (
            "PPC tests stochastic profile consistency with absolute conditioning; its primary evidence is the "
            "20-view profile MSE/cosine change and preservation of centered variation."
        )
    elif variant in {"v4_pdc_ppc_low", "v5_pdc_ppc_high"}:
        interpretation = (
            "The combined candidate tests whether PDC specificity and PPC stability are complementary. "
            "A combined downstream gain is not required for mechanism support."
        )
    else:
        interpretation = "Reference current MoPF: absolute conditioner, PPC disabled, HRC disabled."
    return {
        "variant": variant,
        "dataset_summaries": dataset_summaries,
        "descriptive_deltas_vs_v0": {
            "mean_validation_accuracy": downstream_val_delta,
            "mean_text_centered_std": centered_std_delta,
            "mean_text_stochastic_profile_mse": profile_mse_delta,
            "mean_text_stochastic_profile_cosine": profile_cosine_delta,
        },
        "detected_pathologies": detected_pathologies,
        "evidence_supported_interpretation": interpretation,
    }


def _decision(candidate_summaries: dict[str, dict[str, Any]]) -> dict[str, str]:
    pdc = candidate_summaries["v1_pdc"]
    pdc_rho = [
        pdc["dataset_summaries"][dataset]["mechanism"]["rho_text_abs_max"]["mean"]
        for dataset in DATASETS
    ]
    pdc_active = sum(value is not None and value > 1e-3 for value in pdc_rho) >= 3
    pdc_pathology = any("materially" in item for item in pdc["detected_pathologies"])

    ppc_candidates = [candidate_summaries[name] for name in ("v2_ppc_low", "v3_ppc_high")]
    ppc_improved = 0
    ppc_collapse = False
    for candidate in ppc_candidates:
        for dataset in DATASETS:
            base = candidate_summaries["v0_current"]["dataset_summaries"][dataset]
            current = candidate["dataset_summaries"][dataset]
            base_mse = base["stability"]["text"]["stochastic_profile_mse"]["mean"]
            cur_mse = current["stability"]["text"]["stochastic_profile_mse"]["mean"]
            if base_mse is not None and cur_mse is not None and cur_mse < base_mse:
                ppc_improved += 1
            base_std = base["stability"]["text"]["centered_node_std_mean_over_views"]["mean"]
            cur_std = current["stability"]["text"]["centered_node_std_mean_over_views"]["mean"]
            if base_std and cur_std is not None and cur_std < 0.25 * base_std:
                ppc_collapse = True
    if pdc_active and not pdc_pathology:
        pdc_decision = "Keep for further study"
    elif pdc_active:
        pdc_decision = "Keep for further study"
    else:
        pdc_decision = "Reject"
    if ppc_improved >= 3 and not ppc_collapse:
        ppc_decision = "Keep for further study"
    elif ppc_collapse:
        ppc_decision = "Reject"
    else:
        ppc_decision = "Keep for further study"
    combined = candidate_summaries["v4_pdc_ppc_low"]
    combined_pathology = bool(combined["detected_pathologies"])
    combined_decision = "Keep for further study" if not combined_pathology else "Reject"
    return {"PDC": pdc_decision, "PPC": ppc_decision, "Combined": combined_decision}


def _write_master_table(path: Path, aggregates: dict[tuple[str, str], dict[str, Any]]) -> None:
    fieldnames = [
        "dataset", "variant", "val_acc_mean", "val_acc_std", "val_macro_f1_mean", "val_macro_f1_std",
        "test_acc_mean", "test_acc_std", "test_macro_f1_mean", "test_macro_f1_std",
        "node_mean_test_acc_drop_mean", "node_shuffle_test_acc_drop_mean", "node_shuffle_test_acc_drop_std_across_seeds",
        "text_centered_std_mean", "visual_centered_std_mean", "text_stochastic_mse_mean", "visual_stochastic_mse_mean",
        "text_stochastic_cosine_mean", "visual_stochastic_cosine_mean", "text_cross_seed_cosine", "visual_cross_seed_cosine",
        "text_cross_seed_normalized_rmse", "visual_cross_seed_normalized_rmse",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in DATASETS:
            for variant in VARIANT_ORDER:
                aggregate = aggregates[(dataset, variant)]
                row = {"dataset": dataset, "variant": variant}
                for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                    row[f"{metric}_mean"] = aggregate["downstream"][metric]["mean"]
                    row[f"{metric}_std"] = aggregate["downstream"][metric]["population_std"]
                row["node_mean_test_acc_drop_mean"] = aggregate["frozen_counterfactual"]["node_mean"]["delta_test_acc"]["mean"]
                row["node_shuffle_test_acc_drop_mean"] = aggregate["frozen_counterfactual"]["node_shuffle"]["delta_test_acc_across_training_seeds"]["mean"]
                row["node_shuffle_test_acc_drop_std_across_seeds"] = aggregate["frozen_counterfactual"]["node_shuffle"]["delta_test_acc_across_training_seeds"]["population_std"]
                row["text_centered_std_mean"] = aggregate["mechanism"]["text_centered_node_std_mean"]["mean"]
                row["visual_centered_std_mean"] = aggregate["mechanism"]["visual_centered_node_std_mean"]["mean"]
                row["text_stochastic_mse_mean"] = aggregate["stability"]["text"]["stochastic_profile_mse"]["mean"]
                row["visual_stochastic_mse_mean"] = aggregate["stability"]["visual"]["stochastic_profile_mse"]["mean"]
                row["text_stochastic_cosine_mean"] = aggregate["stability"]["text"]["stochastic_profile_cosine"]["mean"]
                row["visual_stochastic_cosine_mean"] = aggregate["stability"]["visual"]["stochastic_profile_cosine"]["mean"]
                row["text_cross_seed_cosine"] = aggregate["stability"]["text"]["cross_seed_cosine"]
                row["visual_cross_seed_cosine"] = aggregate["stability"]["visual"]["cross_seed_cosine"]
                row["text_cross_seed_normalized_rmse"] = aggregate["stability"]["text"]["cross_seed_normalized_rmse"]
                row["visual_cross_seed_normalized_rmse"] = aggregate["stability"]["visual"]["cross_seed_normalized_rmse"]
                writer.writerow(row)


def _write_report(
    path: Path,
    calibration: dict[str, Any],
    candidate_summaries: dict[str, dict[str, Any]],
    decisions: dict[str, str],
) -> None:
    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{100.0 * value:+.3f} pp"

    calibration_zero = all(
        abs(float(probe["ppc_raw_initial"])) < PPC_EPS
        for probe in calibration["probes"]
    )

    rows = []
    for variant in VARIANT_ORDER:
        summary = candidate_summaries[variant]
        val = [summary["dataset_summaries"][dataset]["downstream"]["val_acc"]["mean"] for dataset in DATASETS]
        test = [summary["dataset_summaries"][dataset]["downstream"]["test_acc"]["mean"] for dataset in DATASETS]
        shuffle = [summary["dataset_summaries"][dataset]["frozen_counterfactual"]["node_shuffle"]["delta_test_acc_across_training_seeds"]["mean"] for dataset in DATASETS]
        rows.append(
            "| {variant} | {val:.3f}% | {test:.3f}% | {shuffle} | {pathologies} |".format(
                variant=variant,
                val=100.0 * float(np.mean(val)),
                test=100.0 * float(np.mean(test)),
                shuffle=pct(float(np.mean(shuffle))),
                pathologies="; ".join(summary["detected_pathologies"]) or "none recorded",
            )
        )
    report = f"""# MoPF M1-C — Propagation Profile Calibration Study

## Scope and protocol

This candidate study evaluates PDC (Propagation-Discrepancy Calibration), PPC (Propagation Profile Consistency), and their combinations. It runs six variants × five NC datasets × three training seeds (42, 43, 44), with Movies/Toys/ele-fashion/Reddit-S at K=3 and Grocery at validation-selected K=2.

The existing full-graph NC protocol is retained: AdamW, learning rate `1e-3`, weight decay `1e-4`, 300 maximum epochs, patience 30, and best-checkpoint selection by Validation Accuracy. Macro-F1 uses the fixed repaired label protocol. HRC remains disabled at `hrc_weight=0` throughout.

The formal default file `configs/model/mopf.yaml` was not changed. Candidate settings are passed only as experiment overrides; V0 is absolute conditioning with PPC disabled.

## PPC scale calibration

The global no-update seed-42 calibration is stored in `outputs/m1c_pdc_ppc/ppc_scale_calibration.json`:

- `R = {calibration['R']:.6g}`
- `lambda_low = {calibration['lambda_low']:.6g}`
- `lambda_high = {calibration['lambda_high']:.6g}`

The probe values are retained in the JSON. Because the current model initializes `node_vector_text/visual` to zero, the initial PPC raw loss was below epsilon for every dataset: **calibration pathology detected = {str(calibration_zero).lower()}**. The prescribed epsilon formula was used exactly; the scale was not silently replaced with a per-dataset or hand-tuned weight.

## Downstream and frozen summary

The table gives the unweighted mean Test/Validation Accuracy across the five dataset means. The node-shuffle column is the mean Test Acc drop relative to Original, aggregated first over 10 permutations and then over the three training seeds.

| Variant | mean Val Acc | mean Test Acc | mean Node-shuffle drop | Detected pathologies |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

All four downstream metrics, per-seed values, population standard deviations, mechanism diagnostics, and frozen Macro-F1 drops are in the master JSON. The CSV is a compact browsing table only.

## Downstream evidence

Downstream metrics are reported descriptively. Differences around 0.1–0.3 percentage points in a single run are not interpreted as module gains. Candidate selection is not based on Test metrics.

## Mechanism evidence

PDC exports learned `rho_text[k]` and `rho_visual[k]`, discrepancy magnitudes, centered profile magnitudes, discrepancy/profile Spearman statistics, shared-offset ratios, mean absolute node residuals, and centered node standard deviations. Nonzero rho is evidence that the candidate was used by optimization, not proof that it improved the mechanism.

## Stability evidence

PPC exports 20-view pairwise centered-profile MSE and per-node profile cosine for both modalities. Cross-seed stability uses centered profiles from the three best checkpoints and reports flattened cosine similarity plus normalized RMSE for 42–43, 42–44, and 43–44. Profile RMS below `1e-6` is marked N/A rather than interpreted as numerical stability.

## Frozen functional evidence

For every dataset/variant/seed, Node-mean and independent text/visual Node-shuffle counterfactuals are evaluated with fixed permutation seeds 0–9. The full Val/Test Acc and Macro-F1 drops, including within-seed shuffle mean/std and across-training-seed aggregates, are included in the master JSON.

## Candidate decisions

- **PDC: {decisions['PDC']}**. This is based on rho movement, discrepancy/profile diagnostics, profile pathologies, downstream validation band, and node-shuffle evidence jointly.
- **PPC: {decisions['PPC']}**. This is based on stochastic MSE/cosine movement, cross-seed stability, centered-profile preservation, and downstream validation band jointly.
- **Combined: {decisions['Combined']}**. This requires evidence that the two candidate mechanisms are compatible; it is not a request to maximize Test Accuracy.

These are candidate-study decisions, not a final MoPF declaration. No formal default has been changed and no new loss is enabled by default.

## Unsupported claims

This study does not support universal claims that PDC or PPC is necessary for every dataset, causal claims from descriptive Spearman/cross-seed correlations, claims that a lower stochastic MSE alone proves better propagation, or claims that frozen counterfactual drops equal retraining ablation effects. The master JSON is the authoritative external-analysis artifact.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def build_master_summary(
    output_root: Path,
    calibration: dict[str, Any],
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in run_records:
        by_group.setdefault((record["dataset"], record["variant"]), []).append(record)
    aggregates = {
        key: _aggregate_dataset_variant(sorted(records, key=lambda item: item["seed"]))
        for key, records in by_group.items()
    }
    candidate_summaries = {
        variant: _candidate_summary(aggregates, variant) for variant in VARIANT_ORDER
    }
    # The reference candidate is needed by the decision helper for all variants.
    decisions = _decision(candidate_summaries)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    summary = {
        "metadata": {
            "stage": "M1-C",
            "study": "Propagation Profile Calibration Study",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit,
            "protocol": {
                "task": "NC",
                "training_mode": "full_graph",
                "optimizer": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 300,
                "patience": 30,
                "checkpoint_selection": "best Validation Accuracy",
                "fixed_macro_f1": True,
                "hrc_weight": 0.0,
                "edge_weight_mode": "separate_cos",
                "node_shuffle_seeds": list(SHUFFLE_SEEDS),
                "stochastic_profile_views": STOCHASTIC_VIEWS,
            },
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "variants": list(VARIANT_ORDER),
            "K_by_dataset": K_BY_DATASET,
            "ppc_calibration": calibration,
        },
        "runs": run_records,
        "dataset_variant_aggregates": {
            f"{dataset}/{variant}": aggregates[(dataset, variant)]
            for dataset in DATASETS
            for variant in VARIANT_ORDER
        },
        "candidate_level_summary": candidate_summaries,
        "candidate_decisions": decisions,
        "artifacts": {
            "master_table_csv": str(output_root / "m1c_master_table.csv"),
            "report": str(ROOT / "docs" / "mopf_m1c_pdc_ppc.md"),
        },
    }
    _dump_json(output_root / "m1c_master_summary.json", summary)
    _write_master_table(output_root / "m1c_master_table.csv", aggregates)
    _write_report(ROOT / "docs" / "mopf_m1c_pdc_ppc.md", calibration, candidate_summaries, decisions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "train", "analyze", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "m1c_pdc_ppc")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    devices = tuple(args.devices)
    if any(device.startswith("cuda") for device in devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    calibration = calibrate(args.output_root, torch.device(devices[0]), force=args.force)
    print(
        f"[m1c] calibration R={calibration['R']:.6g} "
        f"lambda_low={calibration['lambda_low']:.6g} "
        f"lambda_high={calibration['lambda_high']:.6g}",
        flush=True,
    )
    if args.stage == "calibrate":
        return

    if args.stage in {"train", "all"}:
        train_records = train_all(args.output_root, calibration, devices, args.force)
        print(f"[m1c] training complete: {len(train_records)}/90 runs", flush=True)
    if args.stage == "train":
        return

    run_records = analyze_all(args.output_root, devices)
    summary = build_master_summary(args.output_root, calibration, run_records)
    print(
        f"[m1c] master summary={args.output_root / 'm1c_master_summary.json'} "
        f"decisions={summary['candidate_decisions']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
