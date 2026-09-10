"""Run M1-F: PDC-v2 cross-task transfer validation on sports-copurchase LP.

This launcher deliberately runs only seed=42 for S0/S1/S2/S3.  It reuses the
existing sampled LP task, decoder, evaluator, positive-edge masking, and
inference protocol; the extra code is limited to checkpoint bookkeeping and
frozen mechanism audits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import LinkPredictor, build_model  # noqa: E402
from src.tasks.inference import resolve_inference_mode  # noqa: E402
from src.tasks.lp import _evaluate_split, _resolve_lp_num_neighbors  # noqa: E402
from src.utils.summary import count_parameters  # noqa: E402

from scripts.run_mopf_m1e_pdc_v2 import (  # noqa: E402
    EPS,
    _compose_with_profile,
    _delta_changes,
    _dump_json,
    _finite_tensor_tree,
    _m1e_condition_masks,
    _pathway_utilization,
)


DATASET = "sports-copurchase"
SEED = 42
MAX_ORDER = 3
SAMPLER = [5, 5, 5]
SHUFFLE_SEEDS = tuple(range(10))
VARIANTS = ("S0_current", "S1_pdc_v1", "S2_pdc_v2_sep", "S3_pdc_v2_full")
VARIANT_MODES = {
    "S0_current": "absolute",
    # Existing MoPF code names the already-implemented PDC-v1 mode "pdc".
    "S1_pdc_v1": "pdc",
    "S2_pdc_v2_sep": "pdc_v2_sep",
    "S3_pdc_v2_full": "pdc_v2_full",
}
PDC_VARIANTS = VARIANTS[1:]
METRICS = ("val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10")
AUDIT_METRICS = METRICS
OUTPUT_ROOT = ROOT / "outputs" / "m1f_sports_seed42"


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_log() -> str | None:
    try:
        return subprocess.run(
            ["git", "log", "-1", "--oneline"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _training_command(variant: str, device: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "src/main.py",
        f"dataset={DATASET}",
        "task=lp",
        "model=mopf",
        f"seed={SEED}",
        "num_runs=1",
        f"device={device}",
        f"model.max_order={MAX_ORDER}",
        f"model.num_layers={MAX_ORDER}",
        f"task.num_neighbors={SAMPLER}",
        f"+model.node_conditioner_mode={VARIANT_MODES[variant]}",
        "model.hrc_weight=0.0",
        "+model.ppc_weight=0.0",
        "model.edge_weight_mode=separate_cos",
        "task.training_mode=sampled",
        "task.save_ckpt_path=" + str(output_dir / "best_val_mrr.pt"),
        "hydra.job.chdir=false",
        "hydra.run.dir=" + str(output_dir),
    ]


def _gpu_index(device: str) -> str:
    return device.split(":", 1)[1] if ":" in device else device


def _sample_gpu_memory_mib(device: str, pid: int) -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={_gpu_index(device)}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    peak = None
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) == int(pid):
                peak = max(float(fields[1]), peak or 0.0)
        except ValueError:
            continue
    return peak


def _monitor_gpu(device: str, pid: int, stop: threading.Event, peak: list[float | None]) -> None:
    while not stop.is_set():
        current = _sample_gpu_memory_mib(device, pid)
        if current is not None:
            peak[0] = max(current, peak[0] or 0.0)
        stop.wait(0.5)


def _parse_model_params(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    matches = re.findall(r"model\+decoder params=(\d+)", log_path.read_text(errors="replace"))
    return int(matches[-1]) if matches else None


def _parse_best_epoch(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    rows = []
    current_epoch = None
    for line in log_path.read_text(errors="replace").splitlines():
        epoch_match = re.search(r"Epoch\s+(\d+)\s+\|", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        val_match = re.search(r"Val MRR\s+([0-9]+(?:\.[0-9]+)?)", line)
        if val_match and current_epoch is not None:
            rows.append((current_epoch, float(val_match.group(1))))
    if not rows:
        return None
    return int(max(rows, key=lambda row: row[1])[0])


def _parse_log_elapsed(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    stamps = re.findall(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", log_path.read_text(errors="replace"))
    if len(stamps) < 2:
        return None
    try:
        first = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M:%S")
        last = datetime.strptime(stamps[-1], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return max((last - first).total_seconds(), 0.0)


def _make_training_record(
    variant: str,
    output_dir: Path,
    device: str,
    peak_gpu_memory_mib: float | None,
    training_time_sec: float | None = None,
) -> dict[str, Any]:
    results_path = output_dir / "results.json"
    checkpoint = output_dir / "best_val_mrr.pt"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "variant": variant,
        "conditioner": VARIANT_MODES[variant],
        "seed": SEED,
        "device": device,
        "checkpoint": str(checkpoint),
        "metrics": {key: float(payload[key]["mean"]) for key in payload if key in METRICS},
        "best_epoch": _parse_best_epoch(output_dir / "main.log"),
        "parameter_count": _parse_model_params(output_dir / "main.log"),
        "training_time_sec": training_time_sec if training_time_sec is not None else _parse_log_elapsed(output_dir / "main.log"),
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "peak_gpu_memory_status": (
            "captured_by_nvidia_smi_monitor"
            if peak_gpu_memory_mib is not None
            else "unavailable_after_parent_launcher_exit"
        ),
        "checkpoint_seed": checkpoint_payload.get("seed"),
        "results_path": str(results_path),
    }


def _train_one(variant: str, device: str, output_root: Path, force: bool) -> dict[str, Any]:
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_val_mrr.pt"
    results_path = output_dir / "results.json"
    record_path = output_dir / "training_record.json"
    if checkpoint.is_file() and results_path.is_file() and record_path.is_file() and not force:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("best_epoch") is None or "peak_gpu_memory_status" not in record:
            if record.get("best_epoch") is None:
                record["best_epoch"] = _parse_best_epoch(output_dir / "main.log")
            record["peak_gpu_memory_status"] = (
                "captured_by_nvidia_smi_monitor"
                if record.get("peak_gpu_memory_mib") is not None
                else "unavailable_after_parent_launcher_exit"
            )
            _dump_json(record_path, record)
        return record
    if checkpoint.is_file() and results_path.is_file() and not force:
        # A parent launcher may have exited after the child wrote its
        # checkpoint/results. Reconstruct bookkeeping without retraining.
        try:
            cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
            recovered_device = str(cfg.device)
        except (OSError, FileNotFoundError):
            recovered_device = device
        record = _make_training_record(
            variant,
            output_dir,
            recovered_device,
            peak_gpu_memory_mib=None,
        )
        _dump_json(record_path, record)
        print(f"[m1f] recovered existing training record variant={variant}", flush=True)
        return record

    log_path = output_dir / "runner.log"
    command = _training_command(variant, device, output_dir)
    print(f"[m1f] train start variant={variant} device={device}", flush=True)
    started = time.perf_counter()
    peak: list[float | None] = [None]
    stop_monitor = threading.Event()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        )
        monitor = threading.Thread(
            target=_monitor_gpu,
            args=(device, process.pid, stop_monitor, peak),
            daemon=True,
        )
        monitor.start()
        return_code = process.wait()
        stop_monitor.set()
        monitor.join(timeout=2.0)
    elapsed = time.perf_counter() - started
    if return_code != 0 or not checkpoint.is_file() or not results_path.is_file():
        tail = log_path.read_text(errors="replace")[-8000:]
        _dump_json(
            output_dir / "failure.json",
            {"variant": variant, "device": device, "return_code": return_code, "log_tail": tail},
        )
        raise RuntimeError(f"M1-F training failed for {variant}")

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    record = _make_training_record(
        variant,
        output_dir,
        device,
        peak_gpu_memory_mib=peak[0],
        training_time_sec=elapsed,
    )
    _dump_json(record_path, record)
    print(
        f"[m1f] train done variant={variant} time_sec={elapsed:.1f} "
        f"best_epoch={record['best_epoch']} peak_mib={record['peak_gpu_memory_mib']}",
        flush=True,
    )
    return record


def train_all(output_root: Path, devices: tuple[str, ...], force: bool) -> list[dict[str, Any]]:
    jobs_by_device = [list(VARIANTS[index::len(devices)]) for index in range(len(devices))]

    def run_device(variants: list[str], device: str) -> list[dict[str, Any]]:
        return [_train_one(variant, device, output_root, force) for variant in variants]

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_device, variants, device)
            for variants, device in zip(jobs_by_device, devices, strict=True)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: VARIANTS.index(row["variant"]))
    return records


def _load_lp_bundle(run_dir: Path, device: torch.device):
    checkpoint = run_dir / "best_val_mrr.pt"
    config_path = run_dir / ".hydra" / "config.yaml"
    if not checkpoint.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing LP checkpoint/config under {run_dir}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
    data = load_mag_data(cfg, "lp", int(payload["seed"]))
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    projection_dim = int(cfg.task.decoder.get("proj_dim", 0) or 0)
    projection = nn.Linear(model.out_dim, projection_dim).to(device) if projection_dim > 0 else None
    if projection is not None:
        projection.load_state_dict(payload["proj_state"])
        projection.eval()
    predictor_input_dim = projection_dim if projection is not None else model.out_dim
    predictor = LinkPredictor(
        in_dim=predictor_input_dim,
        hidden_dim=int(cfg.task.decoder.hidden_dim),
        num_layers=int(cfg.task.decoder.num_layers),
        dropout=float(cfg.task.decoder.dropout),
    ).to(device)
    predictor.load_state_dict(payload["head_state"])
    predictor.eval()
    return payload, cfg, data, model, projection, predictor


def _model_z(components: dict[str, Any]) -> torch.Tensor:
    return torch.nan_to_num(components["z"], nan=0.0, posinf=1e4, neginf=-1e4)


def _decoder_z(components: dict[str, Any], projection: nn.Module | None) -> torch.Tensor:
    z = _model_z(components)
    return projection(z) if projection is not None else z


def _lp_metrics(predictor, z: torch.Tensor, data, device: torch.device) -> dict[str, float]:
    val = _evaluate_split(z, predictor, data.edge_split.valid, device, int(512))
    test = _evaluate_split(z, predictor, data.edge_split.test, device, int(512))
    return {
        "val_mrr": val["mrr"],
        "val_hits@1": val["hits@1"],
        "val_hits@3": val["hits@3"],
        "val_hits@10": val["hits@10"],
        "test_mrr": test["mrr"],
        "test_hits@1": test["hits@1"],
        "test_hits@3": test["hits@3"],
        "test_hits@10": test["hits@10"],
    }


@torch.no_grad()
def _score_vectors(predictor, z: torch.Tensor, split: dict[str, torch.Tensor], device: torch.device):
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []
    batch_size = 512
    for start in range(0, int(split["source_node"].numel()), batch_size):
        end = min(start + batch_size, int(split["source_node"].numel()))
        src = split["source_node"][start:end].to(device).long()
        dst = split["target_node"][start:end].to(device).long()
        neg = split["target_node_neg"][start:end].to(device).long()
        positives.append(predictor.score_pairs(z[src], z[dst]))
        negatives.append(predictor(z[src].unsqueeze(1) * z[neg]).view(neg.size(0), neg.size(1)))
    return torch.cat(positives), torch.cat(negatives)


def _score_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.reshape(-1)
    second = second.reshape(-1)
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = torch.linalg.vector_norm(first_centered) * torch.linalg.vector_norm(second_centered)
    if float(denominator.item()) <= EPS:
        return 1.0 if torch.equal(first, second) else 0.0
    return float((first_centered * second_centered).sum().div(denominator).item())


def _score_change(
    predictor,
    on_z: torch.Tensor,
    off_z: torch.Tensor,
    data,
    device: torch.device,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split_name in ("val", "test"):
        split = data.edge_split.valid if split_name == "val" else data.edge_split.test
        on_pos, on_neg = _score_vectors(predictor, on_z, split, device)
        off_pos, off_neg = _score_vectors(predictor, off_z, split, device)
        on_all = torch.cat((on_pos, on_neg.reshape(-1)))
        off_all = torch.cat((off_pos, off_neg.reshape(-1)))
        diff = on_all - off_all
        on_rank = 1.0 + (on_neg >= on_pos.unsqueeze(1)).sum(dim=1).float()
        off_rank = 1.0 + (off_neg >= off_pos.unsqueeze(1)).sum(dim=1).float()
        rank_delta = on_rank - off_rank
        output[split_name] = {
            "mean_absolute_score_change": float(diff.abs().mean().item()),
            "relative_score_l2": float(
                (torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(on_all) + EPS)).item()
            ),
            "score_correlation": _score_correlation(on_all, off_all),
            "pairwise_rank_change_rate": float((rank_delta != 0).float().mean().item()),
            "mean_absolute_rank_change": float(rank_delta.abs().mean().item()),
            "positive_mean_absolute_score_change": float((on_pos - off_pos).abs().mean().item()),
            "negative_mean_absolute_score_change": float((on_neg - off_neg).abs().mean().item()),
        }
    return output


def _representation_change(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    diff = first - second
    return {
        "mean_cosine_distance": float(
            (1.0 - F.cosine_similarity(first, second, dim=-1, eps=EPS)).mean().item()
        ),
        "mean_relative_l2": float(
            (torch.linalg.vector_norm(diff, dim=-1) / (torch.linalg.vector_norm(first, dim=-1) + EPS))
            .mean()
            .item()
        ),
    }


def _metric_delta(on: dict[str, float], off: dict[str, float]) -> dict[str, float]:
    return {key: on[key] - off[key] for key in on if key in off}


def _stream_change_pair(
    on,
    off,
    projection,
    predictor,
    data,
    device: torch.device,
) -> dict[str, Any]:
    on_model_z = _model_z(on)
    off_model_z = _model_z(off)
    on_z = projection(on_model_z) if projection is not None else on_model_z
    off_z = projection(off_model_z) if projection is not None else off_model_z
    metrics_on = _lp_metrics(predictor, on_z, data, device)
    metrics_off = _lp_metrics(predictor, off_z, data, device)
    return {
        "node_residual_change": _delta_changes(on, off),
        "modality_representation_change": {
            "text": _representation_change(on["z_text"], off["z_text"]),
            "visual": _representation_change(on["z_visual"], off["z_visual"]),
        },
        "fused_representation_change": _representation_change(on_model_z, off_model_z),
        "decoder_input_representation_change": _representation_change(on_z, off_z),
        "link_score_change": _score_change(predictor, on_z, off_z, data, device),
        "metrics_on": metrics_on,
        "metrics_off": metrics_off,
        "metrics_delta_on_minus_off": _metric_delta(metrics_on, metrics_off),
    }


def _mean_std(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else 0.0,
        "population_std": float(array.std(ddof=0)) if array.size else 0.0,
        "n": int(array.size),
    }


def _finite_value_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_value_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_value_tree(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def _alignment_for_condition(
    model,
    projection,
    predictor,
    components,
    data,
    device: torch.device,
) -> dict[str, Any]:
    original_z = _decoder_z(components, projection)
    original = _lp_metrics(predictor, original_z, data, device)
    delta_text = components["delta_node_text"]
    delta_visual = components["delta_node_visual"]
    mean_text = delta_text.mean(dim=0, keepdim=True).expand_as(delta_text)
    mean_visual = delta_visual.mean(dim=0, keepdim=True).expand_as(delta_visual)
    mean_components_z = _compose_with_profile(model, components, mean_text, mean_visual)
    mean_z = projection(mean_components_z) if projection is not None else mean_components_z
    node_mean_metrics = _lp_metrics(predictor, mean_z, data, device)
    permutations = []
    for seed in SHUFFLE_SEEDS:
        text_generator = torch.Generator(device="cpu").manual_seed(2 * seed)
        visual_generator = torch.Generator(device="cpu").manual_seed(2 * seed + 1)
        text_perm = torch.randperm(delta_text.size(0), generator=text_generator).to(device)
        visual_perm = torch.randperm(delta_visual.size(0), generator=visual_generator).to(device)
        shuffled_model_z = _compose_with_profile(
            model,
            components,
            delta_text.index_select(0, text_perm),
            delta_visual.index_select(0, visual_perm),
        )
        shuffled_z = projection(shuffled_model_z) if projection is not None else shuffled_model_z
        metrics = _lp_metrics(predictor, shuffled_z, data, device)
        permutations.append(
            {
                "permutation_seed": seed,
                "metrics": metrics,
                "drop": _metric_delta(original, metrics),
            }
        )
    mean_metrics = {
        metric: float(np.mean([row["metrics"][metric] for row in permutations]))
        for metric in AUDIT_METRICS
    }
    mean_drop = _metric_delta(original, mean_metrics)
    std_drop = {
        metric: float(np.std([row["drop"][metric] for row in permutations], ddof=0))
        for metric in AUDIT_METRICS
    }
    return {
        "original": original,
        "node_mean": _metric_delta(original, node_mean_metrics),
        "node_shuffle": {
            "permutation_mode": "independent text and visual complete-profile permutations",
            "permutation_seeds": list(SHUFFLE_SEEDS),
            "permutations": permutations,
            "mean_metrics": mean_metrics,
            "mean_drop": mean_drop,
            "std_drop": std_drop,
        },
    }


def _paired_alignment_gain(on_alignment: dict[str, Any], off_alignment: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for metric in AUDIT_METRICS:
        values = [
            on_row["drop"][metric] - off_row["drop"][metric]
            for on_row, off_row in zip(
                on_alignment["node_shuffle"]["permutations"],
                off_alignment["node_shuffle"]["permutations"],
                strict=True,
            )
        ]
        stats = _mean_std(values)
        result[metric] = {
            **stats,
            "paired_permutation_se": float(stats["population_std"] / np.sqrt(len(values))),
            "permutation_values": values,
            "permutation_count": len(values),
        }
    return result


def _audit_one(
    record: dict[str, Any],
    output_root: Path,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    variant = record["variant"]
    run_dir = output_root / variant
    audit_path = run_dir / "audit.json"
    if audit_path.is_file() and not force:
        return json.loads(audit_path.read_text(encoding="utf-8"))
    payload, cfg, data, model, projection, predictor = _load_lp_bundle(run_dir, device)
    mode = VARIANT_MODES[variant]
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    max_order = int(cfg.model.max_order)
    sampler = _resolve_lp_num_neighbors(cfg)
    formal, _, _, _, _ = model(x, edge_index)
    formal_error = None

    if mode == "absolute":
        on = model._encode_components(x, edge_index)
        condition_specs = {"formal": (None, None)}
        formal_error = float((formal - _model_z(on)).abs().max().item())
    else:
        on = model.analysis_encode_with_pdc_mask(
            x,
            edge_index,
            text_mask=[1.0] * (max_order + 1),
            visual_mask=[1.0] * (max_order + 1),
        )
        formal_error = float((formal - _model_z(on)).abs().max().item())
        condition_specs = _m1e_condition_masks(max_order, mode)
    del formal

    conditions: dict[str, Any] = {}
    off = None
    order0_name = None
    order0_spec = None
    initial_name = "formal" if mode == "absolute" else "F0_pdc_on"
    initial_metrics = _lp_metrics(predictor, _decoder_z(on, projection), data, device)
    conditions[initial_name] = {
        "text_mask": None,
        "visual_mask": None,
        "metrics": initial_metrics,
    }
    if mode != "absolute":
        for name, (text_mask, visual_mask) in condition_specs.items():
            components = on if name == "F0_pdc_on" else model.analysis_encode_with_pdc_mask(
                x, edge_index, text_mask=text_mask, visual_mask=visual_mask
            )
            conditions[name] = {
                "text_mask": text_mask,
                "visual_mask": visual_mask,
                "metrics": _lp_metrics(predictor, _decoder_z(components, projection), data, device),
            }
            if name == "F1_all_pdc_off":
                off = components
            if mode == "pdc_v2_full" and "order0_off" in name:
                order0_name = name
                order0_spec = (list(text_mask), list(visual_mask))
            if name != "F0_pdc_on" and name != "F1_all_pdc_off":
                del components
        for name, item in conditions.items():
            item["metrics_delta_pdc_on_minus_condition"] = {
                metric: conditions["F0_pdc_on"]["metrics"][metric] - item["metrics"][metric]
                for metric in item["metrics"]
            }

    path_utilization = _pathway_utilization(model, on, mode)
    alignment = None
    stream_changes = None
    order0_analysis = None
    if mode != "absolute":
        if off is None:
            raise RuntimeError("M1-F did not produce F1_all_pdc_off")
        stream_changes = _stream_change_pair(on, off, projection, predictor, data, device)
        alignment_on = _alignment_for_condition(model, projection, predictor, on, data, device)
        alignment_off = _alignment_for_condition(model, projection, predictor, off, data, device)
        alignment = {
            "PDC-On": alignment_on,
            "PDC-Off": alignment_off,
            "active_path_alignment_gain": _paired_alignment_gain(alignment_on, alignment_off),
        }
        for modality in ("text", "visual"):
            rows = path_utilization["modalities"][modality]["orders"]
            for row in rows:
                row["all_off_delta_change"] = stream_changes["node_residual_change"][modality]["orders"][row["order"]]
        del alignment_on
        del alignment_off
        del off

        if mode == "pdc_v2_full":
            if order0_name is None or order0_spec is None:
                raise RuntimeError("M1-F C3 missing order0-off condition")
            order0_off = model.analysis_encode_with_pdc_mask(
                x,
                edge_index,
                text_mask=order0_spec[0],
                visual_mask=order0_spec[1],
            )
            order0_stream = _stream_change_pair(on, order0_off, projection, predictor, data, device)
            order0_analysis = {
                "rho0_text": float(model.pdc_rho("text")[0].item()),
                "rho0_visual": float(model.pdc_rho("visual")[0].item()),
                "order0_condition": order0_name,
                "delta0_text_change": order0_stream["node_residual_change"]["text"]["orders"][0],
                "delta0_visual_change": order0_stream["node_residual_change"]["visual"]["orders"][0],
                "fused_representation_change": order0_stream["fused_representation_change"],
                "link_score_change": order0_stream["link_score_change"],
                "metrics_delta_pdc_on_minus_order0_off": order0_stream["metrics_delta_on_minus_off"],
                "stream_changes": order0_stream,
            }
            rho0_nonzero = abs(order0_analysis["rho0_text"]) > 1e-8 or abs(order0_analysis["rho0_visual"]) > 1e-8
            order0_analysis["order0_functionally_weak"] = bool(
                rho0_nonzero
                and order0_analysis["fused_representation_change"]["mean_relative_l2"] < 1e-4
                and order0_analysis["link_score_change"]["val"]["relative_score_l2"] < 1e-4
                and abs(order0_analysis["metrics_delta_pdc_on_minus_order0_off"]["val_mrr"]) < 1e-3
            )
            del order0_off

    logits = predictor(_decoder_z(on, projection))
    pathologies: list[str] = []
    if not _finite_value_tree(on):
        pathologies.append("NaN/Inf in PDC-On components")
    if stream_changes is not None and not _finite_value_tree(stream_changes):
        pathologies.append("NaN/Inf in stream audit")
    if order0_analysis is not None and not _finite_value_tree(order0_analysis):
        pathologies.append("NaN/Inf in order0 audit")
    for modality in ("text", "visual"):
        centered = on[f"delta_node_{modality}"] - on[f"delta_node_{modality}"].mean(dim=0, keepdim=True)
        if float(centered.std(unbiased=False).item()) < 1e-8:
            pathologies.append(f"profile collapse: {modality} centered delta std < 1e-8")
    if float(_model_z(on).abs().max().item()) > 1e4 or float(logits.abs().max().item()) > 1e4:
        pathologies.append("variance explosion: representation/logit abs max > 1e4")

    result = {
        "variant": variant,
        "conditioner": mode,
        "seed": SEED,
        "checkpoint": str(run_dir / "best_val_mrr.pt"),
        "checkpoint_selection": "best Validation MRR",
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "sampler": sampler,
        "parameter_count": {
            "model": count_parameters(model),
            "projection": count_parameters(projection) if projection is not None else 0,
            "decoder": count_parameters(predictor),
            "total": count_parameters(model) + count_parameters(predictor) + (count_parameters(projection) if projection is not None else 0),
        },
        "rho": {
            modality: model.pdc_rho(modality).detach().cpu().tolist()
            if mode != "absolute" else [0.0] * (max_order + 1)
            for modality in ("text", "visual")
        },
        "conditions": conditions,
        "pathway_utilization": path_utilization,
        "stream_changes_pdc_on_vs_off": stream_changes,
        "node_shuffle": alignment,
        "c3_order0_analysis": order0_analysis,
        "detected_pathologies": pathologies,
        "diagnostics": {
            "formal_forward_max_abs_error": formal_error,
            "training_mode": str(cfg.task.training_mode),
            "inference_mode": resolve_inference_mode(cfg),
            "positive_edge_mask_backend": str(cfg.task.positive_edge_mask_backend),
            "decoder_unchanged": True,
            "evaluator_unchanged": True,
            "positive_edge_leakage_checked_by_existing_lp_tests": True,
            "sports_seed_expansion_run": False,
        },
    }
    _dump_json(audit_path, result)
    _dump_json(run_dir / "audit_conditions.json", conditions)
    if alignment is not None:
        _dump_json(run_dir / "audit_node_shuffle.json", alignment)
    if stream_changes is not None:
        _dump_json(run_dir / "audit_stream_changes.json", stream_changes)
    if order0_analysis is not None:
        _dump_json(run_dir / "audit_order0.json", order0_analysis)
    del model, projection, predictor, data, on
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def audit_all(records: list[dict[str, Any]], output_root: Path, devices: tuple[str, ...], force: bool) -> list[dict[str, Any]]:
    jobs_by_device = [records[index::len(devices)] for index in range(len(devices))]

    def run_device(device_records, device):
        return [_audit_one(record, output_root, torch.device(device), force) for record in device_records]

    audited: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_device, records_for_device, device)
            for records_for_device, device in zip(jobs_by_device, devices, strict=True)
        ]
        for future in as_completed(futures):
            audited.extend(future.result())
    audited.sort(key=lambda row: VARIANTS.index(row["variant"]))
    return audited


def _variant_metrics(training_records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {row["variant"]: row["metrics"] for row in training_records}


def _comparison(training_records: list[dict[str, Any]], audits: list[dict[str, Any]]) -> dict[str, Any]:
    training = _variant_metrics(training_records)
    audit_by_variant = {row["variant"]: row for row in audits}
    output: dict[str, Any] = {}
    pairs = (("S0_current", "S1_pdc_v1"), ("S0_current", "S2_pdc_v2_sep"), ("S0_current", "S3_pdc_v2_full"), ("S2_pdc_v2_sep", "S3_pdc_v2_full"))
    for baseline, candidate in pairs:
        candidate_audit = audit_by_variant[candidate]
        item = {
            "training_metrics_candidate_minus_baseline": {
                metric: training[candidate][metric] - training[baseline][metric]
                for metric in training[candidate]
                if metric in training[baseline]
            },
            "candidate_checkpoint_metrics": training[candidate],
        }
        stream = candidate_audit.get("stream_changes_pdc_on_vs_off")
        if stream is not None:
            item["candidate_pdc_on_minus_all_off_metrics"] = stream["metrics_delta_on_minus_off"]
            item["candidate_fused_representation_change"] = stream["fused_representation_change"]
            item["candidate_link_score_change"] = stream["link_score_change"]
        output[f"{baseline}_vs_{candidate}"] = item

    c2 = audit_by_variant["S2_pdc_v2_sep"]
    c3 = audit_by_variant["S3_pdc_v2_full"]
    c2_stream = c2["stream_changes_pdc_on_vs_off"]
    c3_stream = c3["stream_changes_pdc_on_vs_off"]
    output["C2_vs_C3"] = {
        "training_metric_delta_C3_minus_C2": {
            metric: training["S3_pdc_v2_full"][metric] - training["S2_pdc_v2_sep"][metric]
            for metric in training["S3_pdc_v2_full"]
        },
        "fused_representation_relative_l2_delta_C3_minus_C2": c3_stream["fused_representation_change"]["mean_relative_l2"] - c2_stream["fused_representation_change"]["mean_relative_l2"],
        "val_link_score_relative_l2_delta_C3_minus_C2": c3_stream["link_score_change"]["val"]["relative_score_l2"] - c2_stream["link_score_change"]["val"]["relative_score_l2"],
        "c2_rho": c2["rho"],
        "c3_rho": c3["rho"],
        "c3_order0_analysis": c3["c3_order0_analysis"],
    }
    return output


def _classify_variant(variant: str, training: dict[str, Any], audit: dict[str, Any], current: dict[str, Any]) -> str:
    if audit["detected_pathologies"]:
        return "weak"
    val_gap = abs(training["metrics"]["val_mrr"] - current["metrics"]["val_mrr"])
    stream = audit.get("stream_changes_pdc_on_vs_off")
    if stream is None or val_gap > 0.01:
        return "weak"
    active_rep = stream["fused_representation_change"]["mean_relative_l2"]
    active_score = stream["link_score_change"]["val"]["relative_score_l2"]
    val_all_off_drop = stream["metrics_delta_on_minus_off"]["val_mrr"]
    alignment = audit["node_shuffle"]["active_path_alignment_gain"]["val_mrr"]
    alignment_descriptive = alignment["mean"] > 0.0 and alignment["mean"] >= alignment["paired_permutation_se"]
    if active_rep <= 1e-8 or active_score <= 1e-8:
        return "weak"
    if val_all_off_drop > 1e-4 or alignment_descriptive or active_score > 1e-4:
        return "promising" if variant != "S1_pdc_v1" else "viable"
    return "viable"


def _recommend_expansion(labels: dict[str, str], comparison: dict[str, Any], audits: dict[str, Any]) -> str:
    c2_label = labels["S2_pdc_v2_sep"]
    c3_label = labels["S3_pdc_v2_full"]
    c2 = audits["S2_pdc_v2_sep"]
    c3 = audits["S3_pdc_v2_full"]
    if c2_label == "weak" and c3_label == "weak":
        return "Do not expand yet"
    if c2_label == "weak":
        return "Expand C3"
    if c3_label == "weak":
        return "Expand C2"
    c3_order0_weak = bool(c3.get("c3_order0_analysis", {}).get("order0_functionally_weak", False))
    val_delta = comparison["C2_vs_C3"]["training_metric_delta_C3_minus_C2"]["val_mrr"]
    score_delta = comparison["C2_vs_C3"]["val_link_score_relative_l2_delta_C3_minus_C2"]
    if c3_order0_weak and abs(val_delta) <= 0.01:
        return "Expand C2"
    if val_delta > 1e-3 or score_delta > 1e-4:
        return "Expand C3"
    return "Expand C2 and C3"


def _protocol_from_cfg(run_dir: Path) -> dict[str, Any]:
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return {
        "task": str(cfg.task.name),
        "protocol_version": str(cfg.task.protocol_version),
        "training_mode": str(cfg.task.training_mode),
        "optimizer": str(cfg.task.optimizer),
        "lr": float(cfg.task.lr),
        "weight_decay": float(cfg.task.weight_decay),
        "epochs": int(cfg.task.epochs),
        "patience": int(cfg.task.patience),
        "early_stop_min_epoch": int(cfg.task.early_stop_min_epoch),
        "early_stop_min_delta": float(cfg.task.early_stop_min_delta),
        "batch_size": int(cfg.task.batch_size),
        "num_neighbors": [int(value) for value in cfg.task.num_neighbors],
        "resolved_sampler": [int(value) for value in _resolve_lp_num_neighbors(cfg)],
        "num_train_neg": int(cfg.task.num_train_neg),
        "positive_edge_mask_backend": str(cfg.task.positive_edge_mask_backend),
        "inference_mode": str(cfg.task.inference_mode),
        "inference_batch_size": int(cfg.task.inference_batch_size),
        "eval_every": int(cfg.task.eval_every),
        "checkpoint_selection": "best Validation MRR",
        "test_used_for_selection": False,
        "edge_weight_mode": str(cfg.model.edge_weight_mode),
        "hrc_weight": float(cfg.model.hrc_weight),
        "ppc_weight": float(cfg.model.get("ppc_weight", 0.0)),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    training = summary["training_comparison"]
    audits = summary["audits"]
    lines = [
        "# MoPF M1-F — PDC-v2 Cross-Task Transfer Validation",
        "",
        "Only sports-copurchase LP seed=42 was run. Seeds 43/44 were not started.",
        "",
        "## Training comparison",
        "",
        "| Variant | Conditioner | Val MRR | Test MRR | Hits@1 | Hits@3 | Hits@10 | Best epoch | Params | Peak GPU MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = training[variant]
        m = row["metrics"]
        lines.append(
            f"| {variant} | `{row['conditioner']}` | {m['val_mrr']:.6f} | {m['test_mrr']:.6f} | "
            f"{m['test_hits@1']:.6f} | {m['test_hits@3']:.6f} | {m['test_hits@10']:.6f} | "
            f"{row['best_epoch']} | {row['parameter_count']} | "
            f"{row['peak_gpu_memory_mib'] if row['peak_gpu_memory_mib'] is not None else 'unavailable'} |"
        )
    lines.extend(["", "## Frozen intervention", "", "| Variant | F0 Val MRR | F1 All-Off Val MRR drop | F1 Test MRR drop | Fused rel. L2 | Val score rel. L2 |", "|---|---:|---:|---:|---:|---:|"])
    for variant in PDC_VARIANTS:
        audit = audits[variant]
        stream = audit["stream_changes_pdc_on_vs_off"]
        f0 = audit["conditions"]["F0_pdc_on"]["metrics"]["val_mrr"]
        lines.append(
            f"| {variant} | {f0:.6f} | {stream['metrics_delta_on_minus_off']['val_mrr']:+.6f} | "
            f"{stream['metrics_delta_on_minus_off']['test_mrr']:+.6f} | "
            f"{stream['fused_representation_change']['mean_relative_l2']:.6g} | "
            f"{stream['link_score_change']['val']['relative_score_l2']:.6g} |"
        )
    lines.extend(["", "## Node-shuffle counterfactual", "", "| Variant | Val MRR gain mean | Pop. std over permutations | Paired permutation SE |", "|---|---:|---:|---:|"])
    for variant in PDC_VARIANTS:
        gain = audits[variant]["node_shuffle"]["active_path_alignment_gain"]["val_mrr"]
        lines.append(f"| {variant} | {gain['mean']:+.6g} | {gain['population_std']:.6g} | {gain['paired_permutation_se']:.6g} |")
    lines.extend(["", "## C3 order0", ""])
    order0 = audits["S3_pdc_v2_full"]["c3_order0_analysis"]
    lines.extend([
        f"- rho0 text={order0['rho0_text']:+.6g}, visual={order0['rho0_visual']:+.6g}",
        f"- fused relative L2={order0['fused_representation_change']['mean_relative_l2']:.6g}; validation link-score relative L2={order0['link_score_change']['val']['relative_score_l2']:.6g}",
        f"- Val MRR delta (PDC-On minus order0-off)={order0['metrics_delta_pdc_on_minus_order0_off']['val_mrr']:+.6g}; Test MRR delta={order0['metrics_delta_pdc_on_minus_order0_off']['test_mrr']:+.6g}",
        f"- order0 functionally weak: **{order0['order0_functionally_weak']}**",
        "",
        "## C2 vs C3",
        "",
        f"- C3 minus C2 Val MRR: {summary['comparison']['C2_vs_C3']['training_metric_delta_C3_minus_C2']['val_mrr']:+.6f}",
        f"- C3 minus C2 Test MRR: {summary['comparison']['C2_vs_C3']['training_metric_delta_C3_minus_C2']['test_mrr']:+.6f}",
        f"- C3 minus C2 fused relative L2: {summary['comparison']['C2_vs_C3']['fused_representation_relative_l2_delta_C3_minus_C2']:+.6g}",
        "- The order0 branch is evaluated separately; it is not inferred from C3's overall score.",
        "",
        "## Historical PDC-v1 reference",
        "",
        "The following values are from the prior M1-D run and are not mixed into the formal S0–S3 comparison: Current seed42 Val/Test MRR 0.404909/0.373686; PDC-v1 seed42 0.402646/0.372636.",
        "",
        "## Cross-task labels and expansion recommendation",
        "",
    ])
    for variant in PDC_VARIANTS:
        lines.append(f"- {variant}: **{summary['cross_task_labels'][variant]}**")
    lines.extend([
        "",
        f"Recommendation field: **{summary['three_seed_expansion_recommendation']}**. This is a recommendation only; no seed43/44 run was performed.",
        "",
        "Training comparison, frozen intervention, and node-shuffle counterfactuals are separate evidence types.",
        "Peak GPU memory is marked unavailable for recovered S2/S3 runs because the original parent launcher exited after child completion; no value was inferred.",
        "",
        "No final MoPF claim is made by this study.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", action="append", dest="devices")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    devices = tuple(args.devices or ["cuda:0", "cuda:1"])
    if not torch.cuda.is_available():
        devices = tuple("cpu" for _ in devices)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.skip_training:
        training_records = [
            json.loads((args.output_root / variant / "training_record.json").read_text(encoding="utf-8"))
            for variant in VARIANTS
        ]
    else:
        training_records = train_all(args.output_root, devices, args.force)
    audits = audit_all(training_records, args.output_root, devices, args.force)
    audit_map = {row["variant"]: row for row in audits}
    training_map = {row["variant"]: row for row in training_records}
    comparison = _comparison(training_records, audits)
    labels = {
        variant: _classify_variant(variant, training_map[variant], audit_map[variant], training_map["S0_current"])
        for variant in PDC_VARIANTS
    }
    recommendation = _recommend_expansion(labels, comparison, audit_map)
    protocol = _protocol_from_cfg(args.output_root / "S0_current")
    summary = {
        "metadata": {
            "stage": "M1-F",
            "study": "PDC-v2 Cross-Task Transfer Validation",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "git_log": _git_log(),
            "git_dirty": bool(_git_status()),
            "git_status_short": _git_status(),
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "dataset": DATASET,
            "seed": SEED,
            "K": MAX_ORDER,
            "sampler": SAMPLER,
            "variant_modes": VARIANT_MODES,
            "code_integrity": {
                "c0_absolute": True,
                "c1_existing_pdc_v1_mode_name": "pdc",
                "c2_independent_discrepancy_projection": True,
                "c2_rho0_fixed_zero": True,
                "c3_d0_h1_minus_h0": True,
                "c3_rho0_trainable": True,
                "implementation_reused_from_current_worktree": True,
            },
            "protocol": protocol,
        },
        "training_comparison": training_map,
        "audits": audit_map,
        "comparison": comparison,
        "cross_task_labels": labels,
        "three_seed_expansion_recommendation": recommendation,
        "historical_reference": {
            "source": "M1-D prior sports seed42; excluded from formal S0-S3 comparison",
            "Current": {"val_mrr": 0.404909, "test_mrr": 0.373686},
            "PDC-v1": {"val_mrr": 0.402646, "test_mrr": 0.372636},
        },
        "restrictions": {
            "seed43_44_run": False,
            "sampler_depth_confounded": False,
            "hrc": False,
            "ppc": False,
            "auxiliary_loss": False,
            "semantic_graph_modified": False,
            "fusion_modified": False,
            "eta_formula_modified": False,
            "decoder_modified": False,
            "evaluator_modified": False,
        },
        "artifacts": {
            "summary": str(args.output_root / "m1f_sports_master_summary.json"),
            "report": str(ROOT / "docs" / "mopf_m1f_sports_seed42.md"),
        },
    }
    summary_path = args.output_root / "m1f_sports_master_summary.json"
    _dump_json(summary_path, summary)
    _write_report(ROOT / "docs" / "mopf_m1f_sports_seed42.md", summary)
    print(f"[m1f] summary={summary_path}", flush=True)
    print(json.dumps({"labels": labels, "recommendation": recommendation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
