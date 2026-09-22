#!/usr/bin/env python3
"""Run the unified three-seed LP benchmark for the requested baselines.

The launcher runs one ``src.main`` process per model.  Each process loads the
dataset split once with base seed 42 and performs three training runs with
seeds 42, 43, and 44.  The LP neighbor schedule is pinned explicitly to
``[5, 5, 5]`` for every model, including models whose internal depth is not
three.

Example, from the project root::

    python scripts/run_lp_baseline_benchmark.py \
        --dataset sports-copurchase --devices cuda:0 cuda:1
    python scripts/run_lp_baseline_benchmark.py \
        --dataset cloth-copurchase --devices cuda:0 cuda:1

Use ``--resume`` to skip only output directories that pass the completion
audit, and ``--dry-run`` to inspect the generated Hydra commands.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
import warnings
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "lp_baseline_benchmark"
TASK_CONFIG_PATH = ROOT / "configs" / "task" / "lp.yaml"
DATASETS = ("sports-copurchase", "cloth-copurchase")
BASELINE_MODELS = (
    "gcn",
    "sage",
    "mmgcn",
    "mgat",
    "dgf",
    "dmgc",
    "lgmrec",
    "dip",
)
BASE_SEED = 42
RUN_SEEDS = [42, 43, 44]
NUM_RUNS = 3
NUM_NEIGHBORS = [5, 5, 5]
TASK_PROTOCOL = "unified_sampled_lp_v1"
REQUIRED_OUTPUTS = (
    "main.log",
    "train.log",
    "results.json",
    "metrics.json",
    "resolved_config.yaml",
    "resolved_config.json",
    "ablation_manifest.json",
    "best.pt",
    "complete.marker",
)
NONFINITE_TRAIN_OR_VAL_RE = re.compile(
    r"\b(?:Train Loss|Val (?:MRR|H@1|H@3|H@10))\s+"
    r"(?:nan|[+-]?inf(?:inity)?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Job:
    index: int
    model: str
    dataset: str
    device: str
    output_dir: Path
    command: tuple[str, ...]
    resume_skip: bool
    audit_reasons: tuple[str, ...]


@dataclass(frozen=True)
class JobResult:
    index: int
    model: str
    dataset: str
    device: str
    status: str
    returncode: int | None
    seconds: float
    output_dir: str
    log_path: str
    message: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_nonfinite(item) for item in value)
    return False


def validate_configs(dataset: str, models: Iterable[str]) -> dict[str, Any]:
    task_cfg = load_yaml(TASK_CONFIG_PATH)
    if task_cfg.get("name") != "lp":
        raise ValueError(f"Expected LP task config, got {task_cfg.get('name')!r}")
    if task_cfg.get("protocol_version") != TASK_PROTOCOL:
        raise ValueError(
            f"Expected LP protocol {TASK_PROTOCOL!r}, "
            f"got {task_cfg.get('protocol_version')!r}"
        )
    if task_cfg.get("num_neighbors") != NUM_NEIGHBORS:
        raise ValueError(
            "configs/task/lp.yaml must use task.num_neighbors=[5, 5, 5]; "
            f"got {task_cfg.get('num_neighbors')!r}"
        )

    dataset_path = ROOT / "configs" / "dataset" / f"{dataset}.yaml"
    dataset_cfg = load_yaml(dataset_path)
    if dataset_cfg.get("name") != dataset:
        raise ValueError(
            f"Dataset config name mismatch: expected {dataset!r}, "
            f"got {dataset_cfg.get('name')!r}"
        )

    model_hashes: dict[str, str] = {}
    for model in models:
        model_path = ROOT / "configs" / "model" / f"{model}.yaml"
        model_cfg = load_yaml(model_path)
        if model_cfg.get("name") != model:
            raise ValueError(
                f"Model config name mismatch in {model_path}: "
                f"expected {model!r}, got {model_cfg.get('name')!r}"
            )
        model_hashes[model] = sha256_file(model_path)

    return {
        "lp_task": sha256_file(TASK_CONFIG_PATH),
        "dataset": sha256_file(dataset_path),
        "models": model_hashes,
    }


def unit_dir(dataset: str, model: str) -> Path:
    return OUTPUT_ROOT / dataset / model / "runs_42_43_44"


def make_command(dataset: str, model: str, device: str, output_dir: Path) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=lp",
        f"model={model}",
        f"seed={BASE_SEED}",
        f"num_runs={NUM_RUNS}",
        f"device={device}",
        "ablation=full",
        "task.num_neighbors=[5,5,5]",
        f"task.save_ckpt_path={output_dir / 'best.pt'}",
        f"hydra.run.dir={output_dir}",
    )


def checkpoint_is_valid(path: Path, expected_seed: int) -> tuple[bool, str]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # A failed load means the checkpoint is not resumable.
        return False, f"checkpoint cannot be loaded: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        return False, "checkpoint is missing a model_state mapping"
    if payload.get("task") != "lp" or payload.get("seed") != expected_seed:
        return False, "checkpoint task/seed metadata does not match this run"
    if not payload["model_state"]:
        return False, "checkpoint model_state is empty"
    return True, "checkpoint is readable"


def resolved_config_matches(
    path: Path, dataset: str, model: str, device: str
) -> tuple[bool, str]:
    try:
        config = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"resolved config is unreadable: {exc}"
    if not isinstance(config, dict):
        return False, "resolved config is not a mapping"

    task_cfg = config.get("task", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    checks = {
        "model.name": (model_cfg.get("name"), model),
        "task.name": (task_cfg.get("name"), "lp"),
        "task.protocol_version": (task_cfg.get("protocol_version"), TASK_PROTOCOL),
        "task.num_neighbors": (list(task_cfg.get("num_neighbors", [])), NUM_NEIGHBORS),
        "dataset.name": (dataset_cfg.get("name"), dataset),
        "seed (split/base seed)": (config.get("seed"), BASE_SEED),
        "num_runs": (config.get("num_runs"), NUM_RUNS),
        "device": (config.get("device"), device),
        "ablation": (config.get("ablation"), "full"),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            return False, f"resolved config mismatch for {key}: expected {expected!r}, got {actual!r}"
    return True, "resolved config matches baseline settings"


def completion_status(
    output_dir: Path,
    dataset: str,
    model: str,
    device: str,
    config_hashes: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    run_checkpoints = [output_dir / f"best_run{run_id}.pt" for run_id in range(1, NUM_RUNS + 1)]
    required = [output_dir / name for name in REQUIRED_OUTPUTS] + run_checkpoints
    for path in required:
        if not path.is_file():
            reasons.append(f"missing {path.name}")

    for log_name in ("main.log", "train.log"):
        log_path = output_dir / log_name
        if not log_path.is_file():
            continue
        try:
            for line_number, line in enumerate(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if NONFINITE_TRAIN_OR_VAL_RE.search(line):
                    reasons.append(f"{log_name}:{line_number} contains a non-finite train/validation value")
                    break
        except OSError as exc:
            reasons.append(f"{log_name} is unreadable: {exc}")

    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = read_json(metrics_path)
            if not isinstance(metrics, dict):
                reasons.append("metrics.json is not a mapping")
            else:
                for key, expected in {
                    "model": model,
                    "task": "lp",
                    "dataset": dataset,
                    "seed": BASE_SEED,
                }.items():
                    if metrics.get(key) != expected:
                        reasons.append(
                            f"metrics.json {key} mismatch: expected {expected!r}, "
                            f"got {metrics.get(key)!r}"
                        )
                if contains_nonfinite(metrics):
                    reasons.append("metrics.json contains NaN/Inf")
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"metrics.json is unreadable: {exc}")

    for result_name in ("results.json",):
        result_path = output_dir / result_name
        if result_path.is_file():
            try:
                if contains_nonfinite(read_json(result_path)):
                    reasons.append(f"{result_name} contains NaN/Inf")
            except (OSError, json.JSONDecodeError) as exc:
                reasons.append(f"{result_name} is unreadable: {exc}")

    resolved_path = output_dir / "resolved_config.json"
    if resolved_path.is_file():
        matches, reason = resolved_config_matches(resolved_path, dataset, model, device)
        if not matches:
            reasons.append(reason)

    manifest_path = output_dir / "benchmark_manifest.json"
    if manifest_path.is_file():
        try:
            saved = read_json(manifest_path)
            saved_hashes = saved.get("config_sha256") if isinstance(saved, dict) else None
            if saved_hashes is not None and saved_hashes != config_hashes:
                reasons.append("source config SHA256 differs from the unit manifest")
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"benchmark_manifest.json is unreadable: {exc}")

    marker_path = output_dir / "complete.marker"
    if marker_path.is_file():
        try:
            marker = read_json(marker_path)
            expected_marker = {
                "status": "complete",
                "task": "lp",
                "dataset": dataset,
                "ablation": "full",
                "seed": BASE_SEED,
            }
            if not isinstance(marker, dict):
                reasons.append("complete.marker is not a mapping")
            else:
                for key, expected in expected_marker.items():
                    if marker.get(key) != expected:
                        reasons.append(
                            f"complete.marker {key} mismatch: expected {expected!r}, "
                            f"got {marker.get(key)!r}"
                        )
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"complete.marker is unreadable: {exc}")

    for run_id, checkpoint_path in enumerate(run_checkpoints, start=1):
        if checkpoint_path.is_file():
            valid, reason = checkpoint_is_valid(checkpoint_path, BASE_SEED + run_id - 1)
            if not valid:
                reasons.append(f"{checkpoint_path.name}: {reason}")

    pointer = output_dir / "best.pt"
    latest_checkpoint = output_dir / f"best_run{NUM_RUNS}.pt"
    if pointer.is_symlink():
        if pointer.resolve() != latest_checkpoint.resolve():
            reasons.append("best.pt does not point to best_run3.pt")
    elif pointer.exists():
        reasons.append("best.pt exists but is not the expected symlink")
    elif any(path.is_file() for path in run_checkpoints):
        reasons.append("best.pt symlink is missing")

    return not reasons, reasons


def prepare_checkpoint_pointer(output_dir: Path) -> tuple[bool, str]:
    pointer = output_dir / "best.pt"
    target = Path(f"best_run{NUM_RUNS}.pt")
    if pointer.is_symlink():
        if pointer.resolve() == (output_dir / target).resolve():
            return True, "checkpoint pointer is ready"
        return False, f"preserving unexpected existing symlink: {pointer} -> {pointer.readlink()}"
    if pointer.exists():
        return False, f"preserving existing non-symlink checkpoint path: {pointer}"
    pointer.symlink_to(target)
    return True, "created best.pt pointer to best_run3.pt"


def build_manifest(dataset: str, models: list[str], devices: list[str], config_hashes: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit_sha": git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(git_value("status", "--porcelain")),
        "dataset": dataset,
        "task": "lp",
        "models": models,
        "devices": devices,
        "base_seed": BASE_SEED,
        "split_seed": BASE_SEED,
        "run_seeds": RUN_SEEDS,
        "num_runs": NUM_RUNS,
        "num_neighbors": NUM_NEIGHBORS,
        "protocol_version": TASK_PROTOCOL,
        "config_sha256": config_hashes,
    }


def format_command(command: Iterable[str]) -> str:
    return shlex.join(list(command))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_MODELS,
        default=list(BASELINE_MODELS),
        help="Subset of the eight requested baselines.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="One independent device lane per concurrently running model.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root for baseline outputs and manifests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only units that pass the full completion audit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write the manifest without launching training.",
    )
    args = parser.parse_args()
    if not args.devices:
        parser.error("at least one --devices value is required")
    if len(set(args.devices)) != len(args.devices):
        parser.error("duplicate devices would create unsafe same-device concurrency")
    args.models = list(dict.fromkeys(args.models))
    args.output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    return args


def _run_one_job(
    job: Job,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    config_hashes: dict[str, Any],
    print_lock: threading.Lock,
) -> JobResult:
    start = time.monotonic()
    output_dir = job.output_dir
    log_path = output_dir / "launcher.log"

    if job.resume_skip:
        with print_lock:
            print(f"[RESUME SKIP] {job.model} ({job.dataset})", flush=True)
        return JobResult(
            index=job.index,
            model=job.model,
            dataset=job.dataset,
            device=job.device,
            status="skipped",
            returncode=0,
            seconds=0.0,
            output_dir=str(output_dir),
            log_path=str(log_path),
            message="validated complete output",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    unit_manifest = {
        **manifest,
        "model": job.model,
        "device": job.device,
        "output_dir": str(output_dir),
        "command": list(job.command),
        "config_sha256": config_hashes,
    }
    write_json(output_dir / "benchmark_manifest.json", unit_manifest)

    if args.dry_run:
        with print_lock:
            print(f"[DRY-RUN] {job.model} on {job.device}", flush=True)
            print(f"           {format_command(job.command)}", flush=True)
        return JobResult(
            index=job.index,
            model=job.model,
            dataset=job.dataset,
            device=job.device,
            status="dry_run",
            returncode=None,
            seconds=0.0,
            output_dir=str(output_dir),
            log_path=str(log_path),
        )

    pointer_ready, pointer_message = prepare_checkpoint_pointer(output_dir)
    if not pointer_ready:
        with print_lock:
            print(f"[FAIL] {job.model}: {pointer_message}", flush=True)
        return JobResult(
            index=job.index,
            model=job.model,
            dataset=job.dataset,
            device=job.device,
            status="failed",
            returncode=None,
            seconds=time.monotonic() - start,
            output_dir=str(output_dir),
            log_path=str(log_path),
            message=pointer_message,
        )

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    with print_lock:
        print(f"[RUN] {job.model} on {job.device} | {job.dataset}", flush=True)
        print(f"      {format_command(job.command)}", flush=True)

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                list(job.command),
                cwd=ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = int(completed.returncode)
    except OSError as exc:
        returncode = None
        message = f"could not launch child process: {exc}"
    else:
        message = "" if returncode == 0 else "child process failed; inspect launcher.log and main.log"

    seconds = time.monotonic() - start
    if returncode != 0:
        status = "failed"
    else:
        complete, reasons = completion_status(
            output_dir, job.dataset, job.model, job.device, config_hashes
        )
        status = "completed" if complete else "failed"
        if not complete:
            message = "output audit failed: " + "; ".join(reasons)

    with print_lock:
        marker = "COMPLETE" if status == "completed" else "FAIL"
        print(
            f"[{marker}] {job.model} on {job.device} ({seconds / 60.0:.1f} min)",
            flush=True,
        )
        if message:
            print(f"        {message}", flush=True)
    return JobResult(
        index=job.index,
        model=job.model,
        dataset=job.dataset,
        device=job.device,
        status=status,
        returncode=returncode,
        seconds=seconds,
        output_dir=str(output_dir),
        log_path=str(log_path),
        message=message,
    )


def main() -> int:
    args = parse_args()
    config_hashes = validate_configs(args.dataset, args.models)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.dataset, args.models, args.devices, config_hashes)
    manifest_path = output_root / "manifests" / f"{args.dataset}.json"
    write_json(manifest_path, manifest)

    jobs: list[Job] = []
    for index, model in enumerate(args.models, start=1):
        output_dir = output_root / args.dataset / model / "runs_42_43_44"
        command = make_command(args.dataset, model, args.devices[(index - 1) % len(args.devices)], output_dir)
        complete, reasons = completion_status(
            output_dir,
            args.dataset,
            model,
            args.devices[(index - 1) % len(args.devices)],
            config_hashes,
        )
        if output_dir.exists() and not complete and any(output_dir.iterdir()):
            warnings.warn(
                f"Existing incomplete output will be rerun in place: {output_dir}; "
                f"audit: {'; '.join(reasons) or 'not marked complete'}",
                RuntimeWarning,
            )
        jobs.append(
            Job(
                index=index,
                model=model,
                dataset=args.dataset,
                device=args.devices[(index - 1) % len(args.devices)],
                output_dir=output_dir,
                command=command,
                resume_skip=bool(args.resume and complete),
                audit_reasons=tuple(reasons),
            )
        )

    print(f"Baseline LP plan: {len(jobs)} models | dataset={args.dataset}", flush=True)
    print(f"Seeds: {RUN_SEEDS} | task.num_neighbors={NUM_NEIGHBORS}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    print_lock = threading.Lock()
    if args.dry_run:
        results = [
            _run_one_job(job, args, manifest, config_hashes, print_lock)
            for job in jobs
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
            futures = [
                executor.submit(_run_one_job, job, args, manifest, config_hashes, print_lock)
                for job in jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item.index)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "models": args.models,
        "run_seeds": RUN_SEEDS,
        "num_neighbors": NUM_NEIGHBORS,
        "results": [asdict(result) for result in results],
        "completed": sum(result.status == "completed" for result in results),
        "skipped": sum(result.status == "skipped" for result in results),
        "failed": sum(result.status == "failed" for result in results),
        "dry_run": args.dry_run,
    }
    summary_path = output_root / "summaries" / f"{args.dataset}.json"
    write_json(summary_path, summary)
    print(f"Summary: {summary_path}", flush=True)
    if args.dry_run:
        print(f"Dry-run commands: {len(jobs)}", flush=True)
        return 0
    print(
        f"Finished: completed={summary['completed']} skipped={summary['skipped']} "
        f"failed={summary['failed']}",
        flush=True,
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
