"""Run M1-G: diagnose PDC cross-task over-calibration on sports LP.

This stage trains only Current and C3 for seeds 43/44.  The seed42
checkpoints from M1-F are reused after the M1-G freeze commit.  All other
diagnostics are frozen, analysis-only computations; no model formula is
changed and no diagnostic alpha is written back to a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from src.data.graph_utils import edge_dict_to_index  # noqa: E402
from src.models import LinkPredictor, build_model  # noqa: E402
from src.tasks.inference import resolve_inference_mode  # noqa: E402
from src.tasks.lp import (  # noqa: E402
    _build_epoch_train_labels,
    _build_forbidden_edge_keys,
    _build_link_loader,
    _build_message_edge_lookup,
    _evaluate_split,
    _exclude_positive_label_edges_by_global_eid,
    _resolve_lp_num_neighbors,
)
from src.utils.summary import count_parameters  # noqa: E402
from scripts.run_mopf_m1e_pdc_v2 import _m1e_condition_masks  # noqa: E402
from scripts.run_mopf_m1f_sports_seed42 import (  # noqa: E402
    EPS,
    SHUFFLE_SEEDS,
    _alignment_for_condition,
    _decoder_z,
    _delta_changes,
    _dump_json,
    _finite_value_tree,
    _lp_metrics,
    _model_z,
    _paired_alignment_gain,
    _pathway_utilization,
    _representation_change,
    _score_change,
    _stream_change_pair,
)


DATASET = "sports-copurchase"
ALL_SEEDS = (42, 43, 44)
NEW_SEEDS = (43, 44)
VARIANTS = ("S0_current", "S3_pdc_v2_full")
VARIANT_MODES = {"S0_current": "absolute", "S3_pdc_v2_full": "pdc_v2_full"}
PDC_VARIANT = "S3_pdc_v2_full"
MAX_ORDER = 3
SAMPLER = [5, 5, 5]
GLOBAL_ALPHAS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
TEXT_ALPHAS = (0.0, 0.5, 1.0)
VISUAL_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ORDER_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ORDER_PAIR_ALPHAS = (0.0, 0.5, 1.0)
SHUFFLE_COUNT = 10
SAMPLED_AUDIT_BATCHES = 100
OUTPUT_ROOT = ROOT / "outputs" / "m1g_sports_overcalibration"
M1F_ROOT = ROOT / "outputs" / "m1f_sports_seed42"
NC_SUMMARY = ROOT / "outputs" / "m1e_pdc_v2" / "m1e_master_summary.json"
# The code was clean at this commit before seed43/44 training started.  Later
# report-only commits are recorded separately in metadata.git_commit.
TRAINING_FREEZE_COMMIT = "4ed47eec03121a9f0ad87f82c8a83e39bd370193"


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


def _parse_model_params(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    matches = re.findall(r"model\+decoder params=(\d+)", log_path.read_text(errors="replace"))
    return int(matches[-1]) if matches else None


def _parse_best_epoch(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    rows: list[tuple[int, float]] = []
    current_epoch: int | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        epoch_match = re.search(r"Epoch\s+(\d+)\s+\|", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        val_match = re.search(r"Val MRR\s+([0-9]+(?:\.[0-9]+)?)", line)
        if val_match and current_epoch is not None:
            rows.append((current_epoch, float(val_match.group(1))))
    return int(max(rows, key=lambda row: row[1])[0]) if rows else None


def _parse_log_elapsed(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    stamps = re.findall(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]",
        log_path.read_text(errors="replace"),
    )
    if len(stamps) < 2:
        return None
    try:
        first = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M:%S")
        last = datetime.strptime(stamps[-1], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return max((last - first).total_seconds(), 0.0)


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


def _training_command(variant: str, seed: int, device: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "src/main.py",
        f"dataset={DATASET}",
        "task=lp",
        "model=mopf",
        f"seed={seed}",
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


def _new_run_dir(output_root: Path, variant: str, seed: int) -> Path:
    return output_root / "runs" / variant / f"seed{seed}"


def _record_from_run(
    variant: str,
    seed: int,
    device: str,
    run_dir: Path,
    peak_gpu_memory_mib: float | None,
    elapsed: float | None,
) -> dict[str, Any]:
    results_path = run_dir / "results.json"
    checkpoint = run_dir / "best_val_mrr.pt"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {
        "variant": variant,
        "conditioner": VARIANT_MODES[variant],
        "seed": seed,
        "device": device,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "metrics": {
            key: float(payload[key]["mean"])
            for key in payload
            if key.startswith("test_") or key == "val_mrr"
        },
        "best_epoch": _parse_best_epoch(run_dir / "main.log"),
        "parameter_count": _parse_model_params(run_dir / "main.log"),
        "training_time_sec": elapsed if elapsed is not None else _parse_log_elapsed(run_dir / "main.log"),
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "peak_gpu_memory_status": (
            "captured_by_nvidia_smi_monitor"
            if peak_gpu_memory_mib is not None
            else "reused_or_unavailable"
        ),
        "checkpoint_seed": checkpoint_payload.get("seed"),
        "results_path": str(results_path),
        "training_protocol": "unified_sampled_lp_v1",
    }


def _seed42_record(variant: str, output_root: Path) -> dict[str, Any]:
    run_dir = M1F_ROOT / variant
    source = json.loads((run_dir / "training_record.json").read_text(encoding="utf-8"))
    record = dict(source)
    record.update(
        {
            "variant": variant,
            "conditioner": VARIANT_MODES[variant],
            "seed": 42,
            "run_dir": str(run_dir),
            "training_source": "M1-F seed42 checkpoint reused after code freeze",
        }
    )
    return record


def _train_one(
    variant: str,
    seed: int,
    device: str,
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    if seed == 42:
        return _seed42_record(variant, output_root)
    run_dir = _new_run_dir(output_root, variant, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_val_mrr.pt"
    results_path = run_dir / "results.json"
    record_path = run_dir / "training_record.json"
    if checkpoint.is_file() and results_path.is_file() and record_path.is_file() and not force:
        return json.loads(record_path.read_text(encoding="utf-8"))
    if checkpoint.is_file() and results_path.is_file() and not force:
        record = _record_from_run(variant, seed, device, run_dir, None, None)
        _dump_json(record_path, record)
        print(f"[m1g] recovered training record variant={variant} seed={seed}", flush=True)
        return record

    log_path = run_dir / "runner.log"
    started = time.perf_counter()
    peak: list[float | None] = [None]
    stop_monitor = threading.Event()
    command = _training_command(variant, seed, device, run_dir)
    print(f"[m1g] train start variant={variant} seed={seed} device={device}", flush=True)
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
        tail = log_path.read_text(errors="replace")[-10000:]
        _dump_json(
            run_dir / "failure.json",
            {"variant": variant, "seed": seed, "device": device, "return_code": return_code, "log_tail": tail},
        )
        raise RuntimeError(f"M1-G training failed for {variant} seed{seed}")
    record = _record_from_run(variant, seed, device, run_dir, peak[0], elapsed)
    _dump_json(record_path, record)
    print(
        f"[m1g] train done variant={variant} seed={seed} time_sec={elapsed:.1f} "
        f"best_epoch={record['best_epoch']} peak_mib={record['peak_gpu_memory_mib']}",
        flush=True,
    )
    return record


def train_all(output_root: Path, devices: tuple[str, ...], force: bool) -> list[dict[str, Any]]:
    jobs = list(itertools.product(VARIANTS, NEW_SEEDS))
    jobs_by_device = [jobs[index::len(devices)] for index in range(len(devices))]

    def run_device(device_jobs, device):
        return [_train_one(variant, seed, device, output_root, force) for variant, seed in device_jobs]

    records = [_seed42_record(variant, output_root) for variant in VARIANTS]
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_device, device_jobs, device)
            for device_jobs, device in zip(jobs_by_device, devices, strict=True)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: (VARIANTS.index(row["variant"]), ALL_SEEDS.index(int(row["seed"]))))
    return records


def _load_bundle(record: dict[str, Any], device: torch.device):
    run_dir = Path(record["run_dir"])
    checkpoint = run_dir / "best_val_mrr.pt"
    cfg_path = run_dir / ".hydra" / "config.yaml"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = OmegaConf.load(cfg_path)
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


def _state_dict_digest(model: nn.Module, projection: nn.Module | None, predictor: nn.Module) -> str:
    digest = hashlib.sha256()
    for label, module in (("model", model), ("projection", projection), ("predictor", predictor)):
        if module is None:
            continue
        for name, value in sorted(module.state_dict().items()):
            digest.update(label.encode())
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _reencode_from_rho(
    model,
    base: dict[str, Any],
    rho_text: torch.Tensor,
    rho_visual: torch.Tensor,
) -> dict[str, Any]:
    delta_text, conditioned_text, aux_text = model._node_residuals(
        base["bases_text"], model.node_proj_text, model.node_vector_text, "text", rho_text
    )
    delta_visual, conditioned_visual, aux_visual = model._node_residuals(
        base["bases_visual"], model.node_proj_visual, model.node_vector_visual, "visual", rho_visual
    )
    eta_text = model._effective_coefficients("text", delta_text)
    eta_visual = model._effective_coefficients("visual", delta_visual)
    z_text = model._filter_bases(base["bases_text"], eta_text)
    z_visual = model._filter_bases(base["bases_visual"], eta_visual)
    z_text_refined = model.text_refine_norm(z_text + model.text_refine_mlp(z_text))
    z_visual_refined = model.visual_refine_norm(z_visual + model.visual_refine_mlp(z_visual))
    fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
    z = model.output_norm(model.fusion_skip(fused_input) + model.fusion_mlp(fused_input))
    output = dict(base)
    output.update(
        {
            "conditioned_bases_text": conditioned_text,
            "conditioned_bases_visual": conditioned_visual,
            "pdc_aux_text": aux_text,
            "pdc_aux_visual": aux_visual,
            "delta_node_text": delta_text,
            "delta_node_visual": delta_visual,
            "eta_text": eta_text,
            "eta_visual": eta_visual,
            "z_text": z_text,
            "z_visual": z_visual,
            "z_text_refined": z_text_refined,
            "z_visual_refined": z_visual_refined,
            "z": z,
        }
    )
    return output


def _rho_scale(model, text_scales: list[float] | tuple[float, ...], visual_scales: list[float] | tuple[float, ...]):
    rho_text = model.pdc_rho("text").detach() * torch.as_tensor(text_scales, device=next(model.parameters()).device)
    rho_visual = model.pdc_rho("visual").detach() * torch.as_tensor(visual_scales, device=next(model.parameters()).device)
    return rho_text, rho_visual


def _metric_row(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in (
        "val_mrr", "val_hits@1", "val_hits@3", "val_hits@10",
        "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10",
    )}


def _stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else 0.0,
        "population_std": float(array.std(ddof=0)) if array.size else 0.0,
        "n": int(array.size),
    }


def _aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    return {key: _stats([float(row[key]) for row in rows]) for key in rows[0]}


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return 0.0
    first_rank = np.argsort(np.argsort(first, kind="stable"), kind="stable").astype(np.float64)
    second_rank = np.argsort(np.argsort(second, kind="stable"), kind="stable").astype(np.float64)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denom = np.linalg.norm(first_rank) * np.linalg.norm(second_rank)
    return float(np.dot(first_rank, second_rank) / denom) if denom > 0 else 0.0


def _target_global_ids(n_id: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    """Map sampled edge-label endpoints to unique global target/root IDs."""
    local_ids = torch.unique(edge_label_index.reshape(-1)).long()
    return n_id[local_ids].long()


def _sampled_full_audit(
    model,
    cfg,
    data,
    full_components: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Compare target-node sampled-context branches to matched full nodes."""
    model.eval()
    uses_graph_data = torch_geometric_data = None
    del uses_graph_data, torch_geometric_data
    from torch_geometric.data import Data

    pyg_data = Data(x=data.x, edge_index=data.edge_index)
    forbidden = _build_forbidden_edge_keys(data.edge_split, data.num_nodes, undirected=True)
    pos_train = edge_dict_to_index(data.edge_split.train).cpu()
    message_lookup = _build_message_edge_lookup(data.edge_index, data.num_nodes)
    neg_generator = torch.Generator().manual_seed(seed + 10000 + 1)
    shuffle_generator = torch.Generator().manual_seed(seed + 20000 + 1)
    neighbor_seed = seed + 30000 + 1
    edge_label_index, edge_label = _build_epoch_train_labels(
        pos_train,
        data.num_nodes,
        int(cfg.task.num_train_neg),
        forbidden,
        neg_generator,
    )
    loader = _build_link_loader(cfg, pyg_data, edge_label_index, edge_label, shuffle_generator, neighbor_seed)

    full_rows: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for modality in ("text", "visual"):
        aux = full_components[f"pdc_aux_{modality}"]
        delta = full_components[f"delta_node_{modality}"]
        for order in range(MAX_ORDER + 1):
            full_rows[(modality, order)] = {
                "state": aux["state_projection"][order],
                "branch": aux["branch"][order],
                "delta": delta[:, order],
            }

    buckets: dict[tuple[str, int], dict[str, list[np.ndarray] | list[float] | int]] = {}
    for key in full_rows:
        buckets[key] = {
            "ratio_sample": [],
            "ratio_full": [],
            "ratio_quotient": [],
            "log_ratio_abs_error": [],
            "branch_cosine_sum": 0.0,
            "state_cosine_sum": 0.0,
            "delta_cosine_sample": [],
            "delta_rmse_sample": [],
            "count": 0,
        }

    batches_seen = 0
    observations = 0
    iterator = iter(loader)
    while batches_seen < SAMPLED_AUDIT_BATCHES:
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Sampled audit produced only {batches_seen} batches, expected {SAMPLED_AUDIT_BATCHES}"
            ) from exc
        removed = _exclude_positive_label_edges_by_global_eid(batch, message_lookup)
        del removed
        root_local = torch.unique(batch.edge_label_index.reshape(-1)).long()
        root_global = _target_global_ids(batch.n_id, batch.edge_label_index)
        batch = batch.to(device)
        root_local = root_local.to(device)
        with torch.no_grad():
            sampled = model._encode_components(batch.x, batch.edge_index)
            for modality in ("text", "visual"):
                aux = sampled[f"pdc_aux_{modality}"]
                sampled_delta = sampled[f"delta_node_{modality}"]
                for order in range(MAX_ORDER + 1):
                    key = (modality, order)
                    full = full_rows[key]
                    sample_state = aux["state_projection"][order].index_select(0, root_local)
                    sample_branch = aux["branch"][order].index_select(0, root_local)
                    sample_delta = sampled_delta.index_select(0, root_local)[:, order]
                    global_ids = root_global.to(device)
                    full_state = full["state"].index_select(0, global_ids)
                    full_branch = full["branch"].index_select(0, global_ids)
                    full_delta = full["delta"].index_select(0, global_ids)
                    sample_ratio = torch.linalg.vector_norm(sample_branch, dim=-1) / (torch.linalg.vector_norm(sample_state, dim=-1) + EPS)
                    full_ratio = torch.linalg.vector_norm(full_branch, dim=-1) / (torch.linalg.vector_norm(full_state, dim=-1) + EPS)
                    bucket = buckets[key]
                    bucket["ratio_sample"].append(sample_ratio.cpu().numpy())
                    bucket["ratio_full"].append(full_ratio.cpu().numpy())
                    bucket["ratio_quotient"].append((sample_ratio / (full_ratio + EPS)).cpu().numpy())
                    bucket["log_ratio_abs_error"].append((torch.log(sample_ratio + EPS) - torch.log(full_ratio + EPS)).abs().cpu().numpy())
                    bucket["branch_cosine_sum"] += float(F.cosine_similarity(sample_branch, full_branch, dim=-1, eps=EPS).sum().item())
                    bucket["state_cosine_sum"] += float(F.cosine_similarity(sample_state, full_state, dim=-1, eps=EPS).sum().item())
                    bucket["delta_cosine_sample"].append(torch.stack((sample_delta, full_delta)).cpu().numpy())
                    bucket["delta_rmse_sample"].append(((sample_delta - full_delta).square().mean().sqrt()).item())
                    bucket["count"] += int(root_local.numel())
        observations += int(root_local.numel())
        batches_seen += 1

    result: dict[str, Any] = {
        "sampler": _resolve_lp_num_neighbors(cfg),
        "batches_seen": batches_seen,
        "target_node_observations": observations,
        "root_definition": "unique global node IDs appearing in sampled batch edge_label_index endpoints; support-only nodes excluded",
        "positive_edge_mask_backend": str(cfg.task.positive_edge_mask_backend),
        "optimizer_step_count": 0,
        "modalities": {},
    }
    for modality in ("text", "visual"):
        result["modalities"][modality] = {"orders": []}
        for order in range(MAX_ORDER + 1):
            bucket = buckets[(modality, order)]
            ratio_sample = np.concatenate(bucket["ratio_sample"])
            ratio_full = np.concatenate(bucket["ratio_full"])
            ratio_quotient = np.concatenate(bucket["ratio_quotient"])
            log_error = np.concatenate(bucket["log_ratio_abs_error"])
            delta_pairs = np.concatenate(bucket["delta_cosine_sample"], axis=1)
            delta_a, delta_b = delta_pairs[0], delta_pairs[1]
            delta_denom = np.linalg.norm(delta_a) * np.linalg.norm(delta_b)
            result["modalities"][modality]["orders"].append(
                {
                    "order": order,
                    "observation_count": int(bucket["count"]),
                    "median_ratio_sampled_over_full": float(np.median(ratio_quotient)),
                    "log_ratio_absolute_error_mean": float(log_error.mean()),
                    "log_ratio_absolute_error_median": float(np.median(log_error)),
                    "spearman_ratio_sampled_vs_full": _spearman(ratio_sample, ratio_full),
                    "projected_discrepancy_branch_cosine": float(bucket["branch_cosine_sum"] / bucket["count"]),
                    "state_projection_cosine": float(bucket["state_cosine_sum"] / bucket["count"]),
                    "node_residual_delta_cosine": float(np.dot(delta_a, delta_b) / delta_denom) if delta_denom > 0 else 0.0,
                    "node_residual_delta_rmse": float(np.sqrt(np.mean((delta_a - delta_b) ** 2))),
                    "node_residual_delta_rmse_batch_mean": float(np.mean(bucket["delta_rmse_sample"])),
                }
            )
    return result


def _condition_metrics(model, projection, predictor, data, device, base, mode: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    conditions: dict[str, Any] = {}
    specs = _m1e_condition_masks(MAX_ORDER, mode)
    on = base
    for name, (text_mask, visual_mask) in specs.items():
        components = on if name == "F0_pdc_on" else model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=text_mask, visual_mask=visual_mask
        )
        conditions[name] = {
            "text_mask": text_mask,
            "visual_mask": visual_mask,
            "metrics": _metric_row(_lp_metrics(predictor, _decoder_z(components, projection), data, device)),
        }
        if name != "F0_pdc_on":
            del components
    for name, item in conditions.items():
        item["metrics_delta_pdc_on_minus_condition"] = {
            key: conditions["F0_pdc_on"]["metrics"][key] - item["metrics"][key]
            for key in item["metrics"]
        }
        item["gain_condition_minus_pdc_on"] = {
            key: -value for key, value in item["metrics_delta_pdc_on_minus_condition"].items()
        }
    off = model.analysis_encode_with_pdc_mask(
        x, edge_index, text_mask=[0.0] * (MAX_ORDER + 1), visual_mask=[0.0] * (MAX_ORDER + 1)
    )
    return conditions, on, off


def _scale_sweeps(model, projection, predictor, data, device, base) -> dict[str, Any]:
    rho_text = model.pdc_rho("text").detach()
    rho_visual = model.pdc_rho("visual").detach()
    ones = [1.0] * (MAX_ORDER + 1)

    def evaluate(text_scales, visual_scales):
        rt = rho_text * torch.as_tensor(text_scales, device=device)
        rv = rho_visual * torch.as_tensor(visual_scales, device=device)
        comp = _reencode_from_rho(model, base, rt, rv)
        return _metric_row(_lp_metrics(predictor, _decoder_z(comp, projection), data, device))

    global_rows = []
    for alpha in GLOBAL_ALPHAS:
        global_rows.append({"alpha": alpha, "metrics": evaluate([alpha] * 4, [alpha] * 4)})

    modality_rows = []
    for alpha_t, alpha_v in itertools.product(TEXT_ALPHAS, VISUAL_ALPHAS):
        modality_rows.append({"alpha_text": alpha_t, "alpha_visual": alpha_v, "metrics": evaluate([alpha_t] * 4, [alpha_v] * 4)})

    order2_rows = []
    order3_rows = []
    for alpha in ORDER_ALPHAS:
        scale2 = list(ones); scale2[2] = alpha
        scale3 = list(ones); scale3[3] = alpha
        order2_rows.append({"alpha_order2": alpha, "metrics": evaluate(ones, scale2)})
        order3_rows.append({"alpha_order3": alpha, "metrics": evaluate(ones, scale3)})

    pair_rows = []
    for alpha2, alpha3 in itertools.product(ORDER_PAIR_ALPHAS, ORDER_PAIR_ALPHAS):
        scales = list(ones); scales[2] = alpha2; scales[3] = alpha3
        pair_rows.append({"alpha_order2": alpha2, "alpha_order3": alpha3, "metrics": evaluate(scales, scales)})

    return {
        "global_strength": {"alphas": list(GLOBAL_ALPHAS), "rows": global_rows},
        "modality_grid": {"text_alphas": list(TEXT_ALPHAS), "visual_alphas": list(VISUAL_ALPHAS), "rows": modality_rows},
        "order2_scale": {"alphas": list(ORDER_ALPHAS), "rows": order2_rows},
        "order3_scale": {"alphas": list(ORDER_ALPHAS), "rows": order3_rows},
        "order2_order3_grid": {"alphas": list(ORDER_PAIR_ALPHAS), "rows": pair_rows},
    }


@torch.no_grad()
def _audit_one(record: dict[str, Any], output_root: Path, device: torch.device, force: bool) -> dict[str, Any]:
    variant = record["variant"]
    seed = int(record["seed"])
    audit_path = output_root / "audits" / variant / f"seed{seed}.json"
    if audit_path.is_file() and not force:
        return json.loads(audit_path.read_text(encoding="utf-8"))
    payload, cfg, data, model, projection, predictor = _load_bundle(record, device)
    state_before = _state_dict_digest(model, projection, predictor)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    mode = VARIANT_MODES[variant]
    formal, _, _, _, _ = model(x, edge_index)
    if mode == "absolute":
        base = model._encode_components(x, edge_index)
        formal_error = float((formal - _model_z(base)).abs().max().item())
        result = {
            "variant": variant,
            "seed": seed,
            "conditioner": mode,
            "checkpoint": record["checkpoint"],
            "conditions": {"formal": {"metrics": _metric_row(_lp_metrics(predictor, _decoder_z(base, projection), data, device))}},
            "training_metrics_from_best_checkpoint": _metric_row(_lp_metrics(predictor, _decoder_z(base, projection), data, device)),
            "pathway_utilization": _pathway_utilization(model, base, mode),
            "diagnostics": {"formal_forward_max_abs_error": formal_error, "inference_mode": resolve_inference_mode(cfg)},
        }
        del formal, base
    else:
        base = model._encode_components(x, edge_index)
        formal_error = float((formal - _model_z(base)).abs().max().item())
        conditions, on, off = _condition_metrics(model, projection, predictor, data, device, base, mode)
        stream = _stream_change_pair(on, off, projection, predictor, data, device)
        alignment_on = _alignment_for_condition(model, projection, predictor, on, data, device)
        alignment_off = _alignment_for_condition(model, projection, predictor, off, data, device)
        alignment = {
            "PDC-On": alignment_on,
            "PDC-Off": alignment_off,
            "active_path_alignment_gain": _paired_alignment_gain(alignment_on, alignment_off),
        }
        order0_off = model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=[0.0, 1.0, 1.0, 1.0], visual_mask=[0.0, 1.0, 1.0, 1.0]
        )
        order0_stream = _stream_change_pair(on, order0_off, projection, predictor, data, device)
        order0 = {
            "rho0_text": float(model.pdc_rho("text")[0].item()),
            "rho0_visual": float(model.pdc_rho("visual")[0].item()),
            "metrics_delta_pdc_on_minus_order0_off": order0_stream["metrics_delta_on_minus_off"],
            "delta0_text_change": order0_stream["node_residual_change"]["text"]["orders"][0],
            "delta0_visual_change": order0_stream["node_residual_change"]["visual"]["orders"][0],
            "fused_representation_change": order0_stream["fused_representation_change"],
            "link_score_change": order0_stream["link_score_change"],
            "stream_changes": order0_stream,
        }
        order0["order0_functionally_weak"] = bool(
            order0["fused_representation_change"]["mean_relative_l2"] < 1e-4
            and order0["link_score_change"]["val"]["relative_score_l2"] < 1e-4
            and abs(order0["metrics_delta_pdc_on_minus_order0_off"]["val_mrr"]) < 1e-3
        )
        scales = _scale_sweeps(model, projection, predictor, data, device, base)
        sampled_full = _sampled_full_audit(model, cfg, data, on, seed, device)
        state_after = _state_dict_digest(model, projection, predictor)
        result = {
            "variant": variant,
            "seed": seed,
            "conditioner": mode,
            "checkpoint": record["checkpoint"],
            "checkpoint_selection": "best Validation MRR",
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "sampler": _resolve_lp_num_neighbors(cfg),
            "parameter_count": {
                "model": count_parameters(model),
                "projection": count_parameters(projection) if projection is not None else 0,
                "decoder": count_parameters(predictor),
                "total": count_parameters(model) + count_parameters(predictor) + (count_parameters(projection) if projection is not None else 0),
            },
            "rho": {modality: model.pdc_rho(modality).detach().cpu().tolist() for modality in ("text", "visual")},
            "conditions": conditions,
            "training_metrics_from_best_checkpoint": conditions["F0_pdc_on"]["metrics"],
            "pathway_utilization": _pathway_utilization(model, on, mode),
            "stream_changes_pdc_on_vs_all_off": stream,
            "node_shuffle": alignment,
            "c3_order0_analysis": order0,
            "frozen_scale_sweeps": scales,
            "sampled_vs_full_audit": sampled_full,
            "detected_pathologies": [] if _finite_value_tree(on) and _finite_value_tree(scales) and _finite_value_tree(sampled_full) else ["non-finite diagnostic output"],
            "diagnostics": {
                "formal_forward_max_abs_error": formal_error,
                "training_mode": str(cfg.task.training_mode),
                "inference_mode": resolve_inference_mode(cfg),
                "positive_edge_mask_backend": str(cfg.task.positive_edge_mask_backend),
                "optimizer_step_count": 0,
                "checkpoint_state_digest_before": state_before,
                "checkpoint_state_digest_after": state_after,
                "checkpoint_state_unchanged": state_before == state_after,
                "analysis_only": True,
            },
        }
        del formal, base, on, off, order0_off, alignment_on, alignment_off
    if mode == "absolute":
        result["diagnostics"].update(
            {
                "optimizer_step_count": 0,
                "checkpoint_state_digest_before": state_before,
                "checkpoint_state_digest_after": _state_dict_digest(model, projection, predictor),
                "checkpoint_state_unchanged": state_before == _state_dict_digest(model, projection, predictor),
                "analysis_only": True,
            }
        )
    _dump_json(audit_path, result)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def audit_all(records: list[dict[str, Any]], output_root: Path, devices: tuple[str, ...], force: bool) -> list[dict[str, Any]]:
    jobs_by_device = [records[index::len(devices)] for index in range(len(devices))]

    def run_device(device_records, device):
        return [_audit_one(record, output_root, torch.device(device), force) for record in device_records]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_device, records_for_device, device)
            for records_for_device, device in zip(jobs_by_device, devices, strict=True)
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda row: (VARIANTS.index(row["variant"]), ALL_SEEDS.index(int(row["seed"]))))
    return results


def _aggregate_training(records: list[dict[str, Any]], audits: list[dict[str, Any]]) -> dict[str, Any]:
    audit_map = {(row["variant"], int(row["seed"])): row for row in audits}
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        per_seed = []
        for seed in ALL_SEEDS:
            record = next(row for row in records if row["variant"] == variant and int(row["seed"]) == seed)
            metrics = audit_map[(variant, seed)]["training_metrics_from_best_checkpoint"]
            per_seed.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "best_epoch": record.get("best_epoch"),
                    "training_time_sec": record.get("training_time_sec"),
                    "parameter_count": record.get("parameter_count"),
                }
            )
        metric_names = list(per_seed[0]["metrics"])
        result[variant] = {
            "per_seed": per_seed,
            "aggregate": {
                "metrics": {name: _stats([row["metrics"][name] for row in per_seed]) for name in metric_names},
                "best_epoch": _stats([row["best_epoch"] for row in per_seed]),
                "training_time_sec": _stats([row["training_time_sec"] for row in per_seed]),
                "parameter_count": _stats([row["parameter_count"] for row in per_seed]),
            },
        }
    deltas = []
    for seed in ALL_SEEDS:
        current = next(row for row in result["S0_current"]["per_seed"] if row["seed"] == seed)
        c3 = next(row for row in result[PDC_VARIANT]["per_seed"] if row["seed"] == seed)
        deltas.append({"seed": seed, "metrics": {key: c3["metrics"][key] - current["metrics"][key] for key in current["metrics"]}})
    result["C3_minus_Current"] = {
        "per_seed": deltas,
        "mean": {key: _stats([row["metrics"][key] for row in deltas]) for key in deltas[0]["metrics"]},
    }
    return result


def _aggregate_interventions(audits: list[dict[str, Any]]) -> dict[str, Any]:
    c3 = [row for row in audits if row["variant"] == PDC_VARIANT]
    names = list(c3[0]["conditions"])
    result: dict[str, Any] = {}
    for name in names:
        metric_names = list(c3[0]["conditions"][name]["metrics"])
        per_seed = []
        for row in c3:
            condition = row["conditions"][name]
            per_seed.append(
                {
                    "seed": row["seed"],
                    "metrics": condition["metrics"],
                    "pdc_on_minus_condition": condition["metrics_delta_pdc_on_minus_condition"],
                    "condition_minus_pdc_on_gain": condition["gain_condition_minus_pdc_on"],
                }
            )
        result[name] = {
            "per_seed": per_seed,
            "metrics": {key: _stats([row["metrics"][key] for row in per_seed]) for key in metric_names},
            "pdc_on_minus_condition": {
                key: {**_stats([row["pdc_on_minus_condition"][key] for row in per_seed]), "positive_seed_count": int(sum(row["pdc_on_minus_condition"][key] > 0 for row in per_seed)), "negative_seed_count": int(sum(row["pdc_on_minus_condition"][key] < 0 for row in per_seed))}
                for key in metric_names
            },
            "condition_minus_pdc_on_gain": {
                key: {**_stats([row["condition_minus_pdc_on_gain"][key] for row in per_seed]), "positive_seed_count": int(sum(row["condition_minus_pdc_on_gain"][key] > 0 for row in per_seed)), "negative_seed_count": int(sum(row["condition_minus_pdc_on_gain"][key] < 0 for row in per_seed))}
                for key in metric_names
            },
        }
    return result


def _aggregate_sweep(audits: list[dict[str, Any]], section: str, key_names: tuple[str, ...]) -> dict[str, Any]:
    c3 = [row for row in audits if row["variant"] == PDC_VARIANT]
    first_rows = c3[0]["frozen_scale_sweeps"][section]["rows"]
    output_rows = []
    for index, first in enumerate(first_rows):
        per_seed = []
        for audit in c3:
            row = audit["frozen_scale_sweeps"][section]["rows"][index]
            per_seed.append({"seed": audit["seed"], "parameters": {key: row[key] for key in key_names}, "metrics": row["metrics"]})
        output_rows.append(
            {
                "parameters": {key: first[key] for key in key_names},
                "per_seed": per_seed,
                "metrics": _aggregate_metric_rows([row["metrics"] for row in per_seed]),
            }
        )
    best = max(output_rows, key=lambda row: row["metrics"]["val_mrr"]["mean"])
    return {"rows": output_rows, "best_by_mean_validation_mrr": best}


def _aggregate_pathway(audits: list[dict[str, Any]]) -> dict[str, Any]:
    c3 = [row for row in audits if row["variant"] == PDC_VARIANT]
    output: dict[str, Any] = {"modalities": {}}
    for modality in ("text", "visual"):
        output["modalities"][modality] = {"orders": []}
        for order in range(MAX_ORDER + 1):
            values = [row["pathway_utilization"]["modalities"][modality]["orders"][order]["branch_to_state_ratio"]["mean"] for row in c3]
            output["modalities"][modality]["orders"].append({"order": order, "mean_branch_to_state_ratio": _stats(values), "seed_values": values})
    return output


def _aggregate_sampled_full(audits: list[dict[str, Any]]) -> dict[str, Any]:
    c3 = [row for row in audits if row["variant"] == PDC_VARIANT]
    output: dict[str, Any] = {"modalities": {}}
    metric_names = (
        "median_ratio_sampled_over_full", "log_ratio_absolute_error_mean", "log_ratio_absolute_error_median",
        "spearman_ratio_sampled_vs_full", "projected_discrepancy_branch_cosine", "state_projection_cosine",
        "node_residual_delta_cosine", "node_residual_delta_rmse", "node_residual_delta_rmse_batch_mean",
    )
    for modality in ("text", "visual"):
        output["modalities"][modality] = {"orders": []}
        for order in range(MAX_ORDER + 1):
            rows = [row["sampled_vs_full_audit"]["modalities"][modality]["orders"][order] for row in c3]
            output["modalities"][modality]["orders"].append({"order": order, "metrics": {key: _stats([float(row[key]) for row in rows]) for key in metric_names}, "per_seed": [{"seed": audit["seed"], **rows[index]} for index, audit in enumerate(c3)]})
    output["seeds"] = [{"seed": row["seed"], "batches_seen": row["sampled_vs_full_audit"]["batches_seen"], "target_node_observations": row["sampled_vs_full_audit"]["target_node_observations"]} for row in c3]
    return output


def _aggregate_node_shuffle(audits: list[dict[str, Any]]) -> dict[str, Any]:
    c3 = [row for row in audits if row["variant"] == PDC_VARIANT]
    result: dict[str, Any] = {}
    for key in ("PDC-On", "PDC-Off"):
        result[key] = {
            "val_mrr_shuffle_drop": _stats([row["node_shuffle"][key]["node_shuffle"]["mean_drop"]["val_mrr"] for row in c3]),
            "test_mrr_shuffle_drop": _stats([row["node_shuffle"][key]["node_shuffle"]["mean_drop"]["test_mrr"] for row in c3]),
            "per_seed": [{"seed": row["seed"], "val_mrr": row["node_shuffle"][key]["node_shuffle"]["mean_drop"]["val_mrr"], "test_mrr": row["node_shuffle"][key]["node_shuffle"]["mean_drop"]["test_mrr"]} for row in c3],
        }
    for metric in ("val_mrr", "test_mrr"):
        values = [row["node_shuffle"]["active_path_alignment_gain"][metric]["mean"] for row in c3]
        result.setdefault("active_path_alignment_gain", {})[metric] = {
            **_stats(values),
            "positive_seed_count": int(sum(value > 0 for value in values)),
            "negative_seed_count": int(sum(value < 0 for value in values)),
            "per_seed": [{"seed": row["seed"], "gain": row["node_shuffle"]["active_path_alignment_gain"][metric]["mean"], "paired_permutation_se": row["node_shuffle"]["active_path_alignment_gain"][metric]["paired_permutation_se"]} for row in c3],
        }
    return result


def _nc_scale_comparison(sports_pathway: dict[str, Any]) -> list[dict[str, Any]]:
    nc = json.loads(NC_SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for key, aggregate in sorted(nc["dataset_variant_aggregates"].items()):
        dataset, variant = key.split("/", 1)
        if variant != "C3_pdc_v2_full":
            continue
        for modality in ("text", "visual"):
            for order in range(len(aggregate["pathway_utilization"]["modalities"][modality]["orders"])):
                rows.append({
                    "dataset": dataset,
                    "task": "NC",
                    "modality": modality,
                    "order": order,
                    "mean_branch_to_state_ratio": aggregate["pathway_utilization"]["modalities"][modality]["orders"][order]["branch_to_state_ratio_mean"]["mean"],
                })
    for modality in ("text", "visual"):
        for order in range(MAX_ORDER + 1):
            rows.append({
                "dataset": DATASET,
                "task": "LP",
                "modality": modality,
                "order": order,
                "mean_branch_to_state_ratio": sports_pathway["modalities"][modality]["orders"][order]["mean_branch_to_state_ratio"]["mean"],
            })
    return rows


def _diagnosis(training: dict[str, Any], global_sweep: dict[str, Any], interventions: dict[str, Any], audits: list[dict[str, Any]]) -> dict[str, Any]:
    best = global_sweep["best_by_mean_validation_mrr"]
    alpha_star = float(best["parameters"]["alpha"])
    alpha1 = next(row for row in global_sweep["rows"] if row["parameters"]["alpha"] == 1.0)
    alpha0 = next(row for row in global_sweep["rows"] if row["parameters"]["alpha"] == 0.0)
    interior_gain = best["metrics"]["val_mrr"]["mean"] - alpha1["metrics"]["val_mrr"]["mean"]
    c3_current = training["C3_minus_Current"]["per_seed"]
    all_off_per_seed = interventions["F1_all_pdc_off"]["per_seed"]
    current_by_seed = {row["seed"]: row for row in training["S0_current"]["per_seed"]}
    c3_off_beats_current = [
        row["seed"]
        for row in all_off_per_seed
        if row["metrics"]["val_mrr"] > current_by_seed[row["seed"]]["metrics"]["val_mrr"]
    ]
    all_off_best_seeds = []
    for seed in (42, 43, 44):
        seed_rows = []
        for row in global_sweep["rows"]:
            seed_row = next(item for item in row["per_seed"] if item["seed"] == seed)
            seed_rows.append({"parameters": row["parameters"], "metrics": seed_row["metrics"]})
        alpha0_seed = next(row for row in seed_rows if row["parameters"]["alpha"] == 0.0)
        if alpha0_seed["metrics"]["val_mrr"] >= max(row["metrics"]["val_mrr"] for row in seed_rows):
            all_off_best_seeds.append(seed)
    alignment = _aggregate_node_shuffle(audits)["active_path_alignment_gain"]["val_mrr"]
    functional_evidence = bool(alignment["positive_seed_count"] >= 2 and any(
        row["pathway_utilization"]["modalities"]["visual"]["orders"][2]["branch_to_state_ratio"]["mean"] > 0
        for row in audits if row["variant"] == PDC_VARIANT
    ))
    if 0.0 < alpha_star < 1.0 and interior_gain > 1e-3 and functional_evidence:
        category = "B"
    elif alpha_star == 0.0 and len(c3_off_beats_current) == 3:
        category = "C"
    elif alpha_star == 1.0 and abs(training["C3_minus_Current"]["mean"]["val_mrr"]["mean"]) <= 0.01 and interventions["F1_all_pdc_off"]["condition_minus_pdc_on_gain"]["val_mrr"]["positive_seed_count"] <= 1:
        category = "A"
    else:
        category = "D"
    rationales = {
        "A": "alpha=1 is the validation optimum, C3 remains in the Current validation band, and PDC-Off does not show systematic improvement.",
        "B": "an interior global alpha is materially better than alpha=1 on mean validation MRR while functional/alignment evidence remains nonzero.",
        "C": "alpha=0/All-Off is the validation optimum and C3-trained checkpoints with PDC-Off beat independently trained Current on all three seeds.",
        "D": "the four diagnostic conditions for A/B/C are not jointly satisfied, or a stable pathology/value pattern prevents a positive active-PDC diagnosis.",
    }
    return {
        "category": category,
        "label": {"A": "C3 viable as-is", "B": "PDC requires scale balancing", "C": "PDC is mainly a training-time regularizer", "D": "Reject active PDC for LP"}[category],
        "rationale": rationales[category],
        "alpha_star_global": alpha_star,
        "alpha_star_validation_gain_over_alpha1": interior_gain,
        "alpha0_validation_mean": alpha0["metrics"]["val_mrr"],
        "alpha1_validation_mean": alpha1["metrics"]["val_mrr"],
        "c3_pdc_off_beats_independently_trained_current_seeds": c3_off_beats_current,
        "all_off_validation_best_seed_count": len(all_off_best_seeds),
        "functional_alignment_evidence": functional_evidence,
        "single_seed_claims_forbidden": True,
    }


def _write_master_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for item in summary["training_comparison"][variant]["per_seed"]:
            for metric, value in item["metrics"].items():
                rows.append({"section": "training_per_seed", "variant": variant, "seed": item["seed"], "metric": metric, "value": value})
    for variant in VARIANTS:
        for metric, item in summary["training_comparison"][variant]["aggregate"]["metrics"].items():
            rows.append({"section": "training_aggregate", "variant": variant, "seed": "mean", "metric": metric, "value": item["mean"], "population_std": item["population_std"]})
    for name, item in summary["frozen_interventions"].items():
        for metric, stats in item["condition_minus_pdc_on_gain"].items():
            rows.append({"section": "frozen_intervention_aggregate", "condition": name, "metric": metric, "value": stats["mean"], "population_std": stats["population_std"], "positive_seed_count": stats["positive_seed_count"], "negative_seed_count": stats["negative_seed_count"]})
    for section in ("global_strength", "order2_scale", "order3_scale"):
        for item in summary["frozen_scale_sweeps"][section]["rows"]:
            rows.append({"section": section, **item["parameters"], "metric": "val_mrr", "value": item["metrics"]["val_mrr"]["mean"], "population_std": item["metrics"]["val_mrr"]["population_std"]})
    for item in summary["frozen_scale_sweeps"]["modality_grid"]["rows"]:
        rows.append({"section": "modality_grid", **item["parameters"], "metric": "val_mrr", "value": item["metrics"]["val_mrr"]["mean"], "population_std": item["metrics"]["val_mrr"]["population_std"]})
    for modality in ("text", "visual"):
        for item in summary["sampled_vs_full_aggregate"]["modalities"][modality]["orders"]:
            for metric, stats in item["metrics"].items():
                rows.append({"section": "sampled_vs_full", "modality": modality, "order": item["order"], "metric": metric, "value": stats["mean"], "population_std": stats["population_std"]})
    for row in summary["nc_vs_lp_scale_comparison"]:
        rows.append({"section": "nc_vs_lp_scale", **row, "value": row["mean_branch_to_state_ratio"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    training = summary["training_comparison"]
    diagnosis = summary["final_diagnosis"]
    lines = [
        "# MoPF M1-G — PDC Cross-Task Over-Calibration Diagnosis",
        "",
        f"Frozen commit: `{summary['metadata']['git_commit']}`. Only Current/C3 seeds 43/44 were newly trained; seed42 was reused from M1-F.",
        "",
        "## Three-seed training",
        "",
        "| Variant | Val MRR | Test MRR | Test H@1 | Test H@3 | Test H@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        metrics = training[variant]["aggregate"]["metrics"]
        lines.append(
            f"| {variant} | {metrics['val_mrr']['mean']:.6f} ± {metrics['val_mrr']['population_std']:.6f} | "
            f"{metrics['test_mrr']['mean']:.6f} ± {metrics['test_mrr']['population_std']:.6f} | "
            f"{metrics['test_hits@1']['mean']:.6f} ± {metrics['test_hits@1']['population_std']:.6f} | "
            f"{metrics['test_hits@3']['mean']:.6f} ± {metrics['test_hits@3']['population_std']:.6f} | "
            f"{metrics['test_hits@10']['mean']:.6f} ± {metrics['test_hits@10']['population_std']:.6f} |"
        )
    lines.extend(["", "C3 − Current per-seed Val/Test MRR:"])
    for row in training["C3_minus_Current"]["per_seed"]:
        lines.append(f"- seed{row['seed']}: {row['metrics']['val_mrr']:+.6f} / {row['metrics']['test_mrr']:+.6f}")
    lines.append(f"- mean: {training['C3_minus_Current']['mean']['val_mrr']['mean']:+.6f} / {training['C3_minus_Current']['mean']['test_mrr']['mean']:+.6f}")
    lines.extend(["", "## Frozen C3 interventions", "", "| Condition | Val MRR gain over PDC-On | Test MRR gain over PDC-On | Positive Val seeds |", "|---|---:|---:|---:|"])
    for name in summary["frozen_interventions"]:
        item = summary["frozen_interventions"][name]["condition_minus_pdc_on_gain"]
        lines.append(f"| {name} | {item['val_mrr']['mean']:+.6f} ± {item['val_mrr']['population_std']:.6f} | {item['test_mrr']['mean']:+.6f} ± {item['test_mrr']['population_std']:.6f} | {item['val_mrr']['positive_seed_count']}/3 |")
    global_best = summary["frozen_scale_sweeps"]["global_strength"]["best_by_mean_validation_mrr"]
    lines.extend(["", "## Frozen global-strength sweep", "", f"alpha_star_global = **{global_best['parameters']['alpha']}**, selected only by mean Validation MRR.", "", "| alpha | Mean Val MRR | Val std | Mean Test MRR |", "|---:|---:|---:|---:|"])
    for row in summary["frozen_scale_sweeps"]["global_strength"]["rows"]:
        lines.append(f"| {row['parameters']['alpha']:.3f} | {row['metrics']['val_mrr']['mean']:.6f} | {row['metrics']['val_mrr']['population_std']:.6f} | {row['metrics']['test_mrr']['mean']:.6f} |")
    grid_best = summary["frozen_scale_sweeps"]["modality_grid"]["best_by_mean_validation_mrr"]
    lines.extend(["", "## Modality scale grid", "", f"alpha_t_star = **{grid_best['parameters']['alpha_text']}**, alpha_v_star = **{grid_best['parameters']['alpha_visual']}** (validation-only selection).", "", "The heatmap source table is in `m1g_master_table.csv`; visual attenuation is assessed by comparing the validation surface across alpha_t/alpha_v, not by Test MRR."])
    lines.extend(["", "## Sampled training-context vs full inference", "", "Ratios are matched by global node ID using only sampled edge-label endpoints as target/root nodes; support-only nodes are excluded. Each C3 checkpoint uses 100 sampled batches and no optimizer step.", "", "| Modality | Order | Median sampled/full ratio | Log-ratio abs. error | Spearman | Branch cosine | State cosine | Delta RMSE |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for modality in ("text", "visual"):
        for item in summary["sampled_vs_full_aggregate"]["modalities"][modality]["orders"]:
            m = item["metrics"]
            lines.append(f"| {modality} | {item['order']} | {m['median_ratio_sampled_over_full']['mean']:.4f} | {m['log_ratio_absolute_error_mean']['mean']:.4f} | {m['spearman_ratio_sampled_vs_full']['mean']:.4f} | {m['projected_discrepancy_branch_cosine']['mean']:.4f} | {m['state_projection_cosine']['mean']:.4f} | {m['node_residual_delta_rmse']['mean']:.4f} |")
    lines.extend(["", "## NC vs LP scale comparison", "", "The table is an analysis-only comparison to the existing M1-E C3 aggregates; cross-task differences are not treated as causal evidence.", "", "## Node-shuffle and order0", "", f"Active-path alignment gain (Val MRR) mean = {summary['node_shuffle']['active_path_alignment_gain']['val_mrr']['mean']:+.6f}, positive seeds = {summary['node_shuffle']['active_path_alignment_gain']['val_mrr']['positive_seed_count']}/3. Strong alignment indicates functional dependence, not better model quality.", f"Order0-on minus order0-off Val MRR mean = {summary['c3_order0_aggregate']['val_mrr_delta_pdc_on_minus_order0_off']['mean']:+.6f}; positive seeds = {summary['c3_order0_aggregate']['positive_val_seed_count']}/3.", "", "## Final M1-G diagnosis", "", f"**Category {diagnosis['category']}: {diagnosis['label']}**", "", diagnosis["rationale"], "", "This is a diagnostic category, not a final model declaration. No PDC-v3 or scale-balanced implementation was added in M1-G; no statistical significance claim is made from three training seeds."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_summary(records: list[dict[str, Any]], audits: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    training = _aggregate_training(records, audits)
    interventions = _aggregate_interventions(audits)
    scales = {
        "global_strength": _aggregate_sweep(audits, "global_strength", ("alpha",)),
        "modality_grid": _aggregate_sweep(audits, "modality_grid", ("alpha_text", "alpha_visual")),
        "order2_scale": _aggregate_sweep(audits, "order2_scale", ("alpha_order2",)),
        "order3_scale": _aggregate_sweep(audits, "order3_scale", ("alpha_order3",)),
        "order2_order3_grid": _aggregate_sweep(audits, "order2_order3_grid", ("alpha_order2", "alpha_order3")),
    }
    pathway = _aggregate_pathway(audits)
    sampled = _aggregate_sampled_full(audits)
    node_shuffle = _aggregate_node_shuffle(audits)
    c3_audits = [row for row in audits if row["variant"] == PDC_VARIANT]
    c3_order0 = {
        "per_seed": [{"seed": row["seed"], "rho0_text": row["c3_order0_analysis"]["rho0_text"], "rho0_visual": row["c3_order0_analysis"]["rho0_visual"], "val_mrr_delta_pdc_on_minus_order0_off": row["c3_order0_analysis"]["metrics_delta_pdc_on_minus_order0_off"]["val_mrr"], "test_mrr_delta_pdc_on_minus_order0_off": row["c3_order0_analysis"]["metrics_delta_pdc_on_minus_order0_off"]["test_mrr"], "fused_relative_l2": row["c3_order0_analysis"]["fused_representation_change"]["mean_relative_l2"], "val_score_relative_l2": row["c3_order0_analysis"]["link_score_change"]["val"]["relative_score_l2"]} for row in c3_audits],
        "rho0_text": _stats([row["c3_order0_analysis"]["rho0_text"] for row in c3_audits]),
        "rho0_visual": _stats([row["c3_order0_analysis"]["rho0_visual"] for row in c3_audits]),
        "val_mrr_delta_pdc_on_minus_order0_off": _stats([row["c3_order0_analysis"]["metrics_delta_pdc_on_minus_order0_off"]["val_mrr"] for row in c3_audits]),
        "test_mrr_delta_pdc_on_minus_order0_off": _stats([row["c3_order0_analysis"]["metrics_delta_pdc_on_minus_order0_off"]["test_mrr"] for row in c3_audits]),
        "positive_val_seed_count": int(sum(row["c3_order0_analysis"]["metrics_delta_pdc_on_minus_order0_off"]["val_mrr"] > 0 for row in c3_audits)),
        "positive_test_seed_count": int(sum(row["c3_order0_analysis"]["metrics_delta_pdc_on_minus_order0_off"]["test_mrr"] > 0 for row in c3_audits)),
    }
    summary: dict[str, Any] = {
        "metadata": {
            "stage": "M1-G",
            "study": "PDC Cross-Task Over-Calibration Diagnosis",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "git_log": _git_log(),
            "git_dirty": bool(_git_status()),
            "git_status_short": _git_status(),
            "training_freeze_commit": TRAINING_FREEZE_COMMIT,
            "environment": {"python": sys.version, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
            "dataset": DATASET,
            "seeds": list(ALL_SEEDS),
            "new_training_seeds": list(NEW_SEEDS),
            "protocol": {"K": MAX_ORDER, "num_neighbors": SAMPLER, "training_mode": "sampled", "inference_mode": "full", "optimizer": "adam", "lr": 1e-3, "weight_decay": 1e-5, "epochs": 150, "patience": 10, "batch_size": 2048, "checkpoint_selection": "best Validation MRR", "test_used_for_selection": False, "edge_weight_mode": "separate_cos", "hrc_weight": 0.0, "ppc_weight": 0.0, "protocol_version": "unified_sampled_lp_v1"},
            "executable_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {"src/models/mopf.py": ROOT / "src/models/mopf.py", "src/tasks/lp.py": ROOT / "src/tasks/lp.py", "configs/dataset/sports-copurchase.yaml": ROOT / "configs/dataset/sports-copurchase.yaml", "scripts/run_mopf_m1f_sports_seed42.py": ROOT / "scripts/run_mopf_m1f_sports_seed42.py"}.items()},
        },
        "training_comparison": training,
        "frozen_interventions": interventions,
        "frozen_scale_sweeps": scales,
        "pathway_utilization_c3_full_inference": pathway,
        "sampled_vs_full_aggregate": sampled,
        "nc_vs_lp_scale_comparison": _nc_scale_comparison(pathway),
        "node_shuffle": node_shuffle,
        "c3_order0_aggregate": c3_order0,
        "final_diagnosis": {},
        "restrictions": {"no_pdc_v3": True, "no_formula_change": True, "no_test_based_alpha_selection": True, "no_statistical_significance_claim": True, "current_c3_only_for_new_seeds": True, "seed42_reused": True},
        "artifacts": {"summary": str(output_root / "m1g_master_summary.json"), "csv": str(output_root / "m1g_master_table.csv"), "report": str(ROOT / "docs" / "mopf_m1g_sports_overcalibration.md")},
    }
    summary["final_diagnosis"] = _diagnosis(training, scales["global_strength"], interventions, audits)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", action="append", dest="devices")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()
    devices = tuple(args.devices or ["cuda:0", "cuda:1"])
    if not torch.cuda.is_available():
        devices = tuple("cpu" for _ in devices)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.skip_training:
        records = [_seed42_record(variant, args.output_root) for variant in VARIANTS]
        for variant, seed in itertools.product(VARIANTS, NEW_SEEDS):
            path = _new_run_dir(args.output_root, variant, seed) / "training_record.json"
            records.append(json.loads(path.read_text(encoding="utf-8")))
    else:
        records = train_all(args.output_root, devices, args.force)
    if args.skip_audit:
        audits = []
        for variant, seed in itertools.product(VARIANTS, ALL_SEEDS):
            audits.append(json.loads((args.output_root / "audits" / variant / f"seed{seed}.json").read_text(encoding="utf-8")))
    else:
        audits = audit_all(records, args.output_root, devices, args.force)
    summary = _build_summary(records, audits, args.output_root)
    summary_path = args.output_root / "m1g_master_summary.json"
    csv_path = args.output_root / "m1g_master_table.csv"
    report_path = ROOT / "docs" / "mopf_m1g_sports_overcalibration.md"
    _dump_json(summary_path, summary)
    _write_master_csv(csv_path, summary)
    _write_report(report_path, summary)
    print(f"[m1g] summary={summary_path}", flush=True)
    print(json.dumps(summary["final_diagnosis"], indent=2), flush=True)


if __name__ == "__main__":
    main()
