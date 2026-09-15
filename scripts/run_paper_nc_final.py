#!/usr/bin/env python3
"""Run and summarize the frozen paper Node Classification benchmark.

This runner is deliberately NC-only.  It launches one ``src.main`` process
per dataset/model/seed, keeps each seed in its own directory, fixes the NC
split to the repository's seed-42 split for every training seed, and performs
post-run provenance and health checks.  It never imports or calls the LP
runner.

Formal execution (from the repository root)::

    PYTHONPATH=src conda run --no-capture-output -n yhf_env \
      python scripts/run_paper_nc_final.py --devices cuda:0 cuda:1

Use ``--dry-run`` first and ``--smoke`` before the full run.  The final
summary and paper table are generated automatically after all 150 jobs pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402
from src.utils.summary import count_parameters  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODELS = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec", "mopf")
DISPLAY_NAMES = {
    "mlp": "MLP",
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "mmgcn": "MMGCN",
    "mgat": "MGAT",
    "dip": "DiP",
    "dgf": "DGF",
    "dmgc": "DMGC",
    "lgmrec": "LGMRec",
    "mopf": "MoPF",
}
K_BY_DATASET = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
}
MAGB_DATASETS = {"Movies", "Toys", "Grocery", "Reddit-S"}
PROTOCOL = "unified_full_graph_nc_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "paper_nc_final_fixed_v1"


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    model: str
    seed: int
    device: str
    run_dir: Path


@dataclass(frozen=True)
class JobResult:
    index: int
    dataset: str
    model: str
    seed: int
    device: str
    status: str
    returncode: int | None
    seconds: float
    run_dir: str
    message: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_path(dataset: str) -> Path:
    if dataset in MAGB_DATASETS:
        return Path("/hdd1/DataInHere/YHF/data/MAGB_split") / f"{dataset}_nc_seed42_train0.6_val0.2.pt"
    return Path("/hdd1/DataInHere/YHF/data/ele-fashion/split.pt")


def _model_overrides(dataset: str, model: str) -> list[str]:
    if model != "mopf":
        return []
    # Formal K was preregistered by dataset; this is not result-driven tuning.
    k = K_BY_DATASET[dataset]
    return [f"model.max_order={k}", f"model.num_layers={k}"]


def _base_overrides(dataset: str, model: str, seed: int, device: str, run_dir: Path) -> list[str]:
    checkpoint = run_dir / "best.pt"
    split_override_key = "dataset.nc_split_path" if dataset in MAGB_DATASETS else "dataset.node_split_path"
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        f"model={model}",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        f"hydra.run.dir={run_dir}",
        f"{split_override_key}={_split_path(dataset)}",
        "task.protocol_version=unified_full_graph_nc_v1",
        "task.training_mode=full_graph",
        "task.optimizer=adamw",
        "task.epochs=300",
        "task.lr=1e-3",
        "task.weight_decay=1e-4",
        "task.eval_every=1",
        "task.patience=30",
        "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4",
        "task.grad_clip=1.0",
        "task.max_train_batches=null",
        "task.inference_mode=full",
        "task.evaluate_test=true",
        f"task.save_ckpt_path={checkpoint}",
    ]
    # These keys exist only on the MoPF config.  Do not append structured
    # overrides to external baseline configs that intentionally omit them.
    if model == "mopf":
        overrides.extend([
            "model.export_aux_stats=false",
            "model.export_node_aux=false",
            "model.export_edge_aux=false",
        ])
    overrides.extend(_model_overrides(dataset, model))
    return overrides


def _compose(dataset: str, model: str, seed: int, device: str, run_dir: Path):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=_base_overrides(dataset, model, seed, device, run_dir))


def _resolve_devices(requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(min(torch.cuda.device_count(), 2))]
    return ["cpu"]


def _build_jobs(output_root: Path, devices: list[str]) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                index += 1
                jobs.append(
                    Job(
                        index=index,
                        dataset=dataset,
                        model=model,
                        seed=seed,
                        device=devices[(index - 1) % len(devices)],
                        run_dir=output_root / dataset / model / f"seed{seed}",
                    )
                )
    return jobs


def _preflight(output_root: Path, devices: list[str]) -> dict[str, Any]:
    """Resolve every formal config and load each fixed split once."""
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        split = _split_path(dataset)
        if not split.is_file():
            raise FileNotFoundError(f"Missing frozen NC split: {split}")
        cfg = _compose(dataset, "mlp", 42, "cpu", output_root / "preflight" / dataset)
        data = load_mag_data(cfg, "nc", 42)
        labels = _resolve_nc_eval_labels(data)
        datasets[dataset] = {
            "split_path": str(split),
            "split_sha256": _sha256(split),
            "num_nodes": int(data.num_nodes),
            "num_classes": int(data.num_classes),
            "input_dim": int(data.input_dim),
            "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
            "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
            "eval_labels": labels,
            "evaluated_class_count": len(labels),
        }
        del data

    configs: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                run_dir = output_root / dataset / model / f"seed{seed}"
                cfg = _compose(dataset, model, seed, devices[(len(configs)) % len(devices)], run_dir)
                resolved = OmegaConf.to_container(cfg, resolve=True)
                task = resolved["task"]
                dataset_cfg = resolved["dataset"]
                config_checks = {
                    "task_is_nc": task["name"] == "nc",
                    "protocol": task["protocol_version"] == PROTOCOL,
                    "full_graph": task["training_mode"] == "full_graph",
                    "max_train_batches_unlimited": task.get("max_train_batches") is None,
                    "eval_every": int(task["eval_every"]) == 1,
                    "checkpoint_selection": "validation_accuracy",
                    "seed_exact": int(resolved["seed"]) == seed and int(resolved["num_runs"]) == 1,
                    "split_exact": str(dataset_cfg.get("nc_split_path", dataset_cfg.get("node_split_path"))) == str(_split_path(dataset)),
                    "model_exact": str(resolved["model"]["name"]) == model,
                }
                if model == "mopf":
                    config_checks["formal_K"] = int(resolved["model"]["max_order"]) == K_BY_DATASET[dataset]
                if not all(value is True or value == "validation_accuracy" for value in config_checks.values()):
                    raise ValueError(f"Preflight config failed for {dataset}/{model}/seed{seed}: {config_checks}")
                configs.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "device": devices[len(configs) % len(devices)],
                        "resolved_config": resolved,
                        "checks": config_checks,
                    }
                )

    payload = {
        "status": "PASS",
        "protocol": PROTOCOL,
        "datasets": datasets,
        "models": list(MODELS),
        "display_names": DISPLAY_NAMES,
        "seeds": list(SEEDS),
        "devices": devices,
        "jobs": len(configs),
        "configs": configs,
        "lp_started": False,
        "note": "Preflight loads only NC data and never calls the LP runner.",
    }
    _dump(output_root / "preflight.json", payload)
    return payload


def _peak_gpu_memory_mib(pid: int) -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
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
        try:
            current = float(fields[1].split()[0])
        except (IndexError, ValueError):
            continue
        peak = current if peak is None else max(peak, current)
    return peak


def _command(job: Job) -> list[str]:
    return [sys.executable, "-m", "src.main", *_base_overrides(job.dataset, job.model, job.seed, job.device, job.run_dir)]


def _format_command(command: Iterable[str]) -> str:
    return " ".join(repr(part) if any(char in part for char in " \t\n") else part for part in command)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _parse_labels(log_text: str) -> list[int] | None:
    match = re.search(r"NC valid evaluation labels:\s*\[([^]]*)\]", log_text)
    if match is None:
        return None
    try:
        return [int(item.strip()) for item in match.group(1).split(",") if item.strip()]
    except ValueError:
        return None


def _parse_parameter_count(log_text: str) -> int | None:
    match = re.search(r"model params=(\d+)", log_text)
    return int(match.group(1)) if match else None


def _collect_record(job: Job, output_root: Path, elapsed: float, peak: float | None, returncode: int) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = job.run_dir
    log_path = run_dir / "main.log"
    result_path = run_dir / "results.json"
    checkpoint_path = run_dir / "best.pt"
    config_path = run_dir / ".hydra" / "config.yaml"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    result = _json_load(result_path) if result_path.is_file() else {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path.is_file() else {}
    preflight = _json_load(output_root / "preflight.json") if (output_root / "preflight.json").is_file() else {}
    expected = preflight.get("datasets", {}).get(job.dataset, {})
    expected_labels = expected.get("eval_labels", [])
    metric_values = {
        key: result.get(key, {}).get("mean")
        for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }
    parsed_labels = _parse_labels(log_text)
    checks = {
        "returncode_zero": returncode == 0,
        "results_present": result_path.is_file(),
        "checkpoint_present": checkpoint_path.is_file(),
        "resolved_config_present": config_path.is_file(),
        "split_present": bool(expected.get("split_path")) and Path(str(expected["split_path"])).is_file(),
        "metrics_finite": all(_finite(value) for value in metric_values.values()),
        "no_nan_inf_metrics": not any(re.search(r"\b(?:nan|inf)\b", str(value), flags=re.IGNORECASE) for value in metric_values.values()),
        "no_traceback": "Traceback (most recent call last)" not in log_text,
        "no_oom": not re.search(r"out of memory|cuda oom|CUBLAS_STATUS_ALLOC_FAILED", log_text, flags=re.IGNORECASE),
        "nonempty_prediction": all(_finite(metric_values[key]) for key in ("test_acc", "test_macro_f1")),
        "eval_labels_present": parsed_labels is not None,
        "eval_labels_match": parsed_labels == expected_labels,
        "evaluated_class_count_normal": parsed_labels is not None and len(parsed_labels) == int(expected.get("evaluated_class_count", -1)),
        "checkpoint_selection_best_val_accuracy": checkpoint.get("selection") == "best_val_accuracy",
        "checkpoint_restore_payload": bool(checkpoint.get("model_state")) and bool(checkpoint.get("head_state")),
        "test_evaluated_once_after_validation": log_text.count("Final Test Acc") == 1 and log_text.rfind("Final Test Acc") > log_text.rfind("Val Acc"),
        "protocol_match": f"Protocol: {PROTOCOL}" in log_text,
        "nc_only": "Task: lp" not in log_text and "LP split:" not in log_text,
    }
    if config_path.is_file():
        config_text = config_path.read_text(encoding="utf-8", errors="replace")
        checks["resolved_config_protocol"] = f"protocol_version: {PROTOCOL}" in config_text
        checks["resolved_config_full_graph"] = "training_mode: full_graph" in config_text
        checks["resolved_config_fixed_split"] = str(expected.get("split_path", "")) in config_text
        resolved_config_sha256 = _sha256(config_path)
    else:
        checks["resolved_config_protocol"] = False
        checks["resolved_config_full_graph"] = False
        checks["resolved_config_fixed_split"] = False
        resolved_config_sha256 = None

    record = {
        "dataset": job.dataset,
        "model": job.model,
        "model_display_name": DISPLAY_NAMES[job.model],
        "seed": job.seed,
        "protocol": PROTOCOL,
        "task": "nc",
        "device": job.device,
        "split_path": expected.get("split_path"),
        "split_sha256": expected.get("split_sha256"),
        "resolved_config_path": str(config_path),
        "resolved_config_sha256": resolved_config_sha256,
        "best_validation_accuracy": metric_values.get("val_acc"),
        "best_epoch": checkpoint.get("epoch"),
        "test_accuracy": metric_values.get("test_acc"),
        "test_macro_f1": metric_values.get("test_macro_f1"),
        "validation_macro_f1": metric_values.get("val_macro_f1"),
        "runtime_seconds": elapsed,
        "peak_gpu_memory_mib": peak,
        "peak_gpu_memory_status": "available" if peak is not None else "unavailable",
        "parameter_count": _parse_parameter_count(log_text),
        "eval_label_set": parsed_labels,
        "evaluated_class_count": len(parsed_labels) if parsed_labels is not None else None,
        "num_classifier_classes": expected.get("num_classes"),
        "checkpoint_path": str(checkpoint_path),
        "results_path": str(result_path),
        "health_checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    validation = {
        "dataset": job.dataset,
        "model": job.model,
        "seed": job.seed,
        "status": record["status"],
        "checks": checks,
        "metric_values": metric_values,
        "log_path": str(log_path),
        "checkpoint_path": str(checkpoint_path),
    }
    _dump(run_dir / "run_record.json", record)
    _dump(run_dir / "validation.json", validation)
    return record, validation


def _run_one(job: Job, output_root: Path, print_lock: threading.Lock, resume: bool) -> JobResult:
    started = time.monotonic()
    record_path = job.run_dir / "run_record.json"
    if resume and record_path.is_file():
        try:
            record = _json_load(record_path)
            if record.get("status") == "PASS":
                with print_lock:
                    print(f"[skip] [{job.index}/150] {job.dataset}/{job.model}/seed{job.seed}: valid run_record exists", flush=True)
                return JobResult(job.index, job.dataset, job.model, job.seed, job.device, "skipped", 0, 0.0, str(job.run_dir), "valid run_record already exists")
        except (OSError, json.JSONDecodeError):
            pass
    job.run_dir.mkdir(parents=True, exist_ok=True)
    # Hydra's logger appends to main.log.  Remove only this runner's known
    # attempt artifacts before retrying a failed run, so validation counts are
    # not contaminated by a previous traceback or test line.
    if record_path.exists():
        for stale in (
            job.run_dir / "main.log",
            job.run_dir / "results.json",
            job.run_dir / "best.pt",
            job.run_dir / "validation.json",
            job.run_dir / ".hydra" / "config.yaml",
            job.run_dir / ".hydra" / "overrides.yaml",
            job.run_dir / ".hydra" / "hydra.yaml",
        ):
            if stale.is_file():
                stale.unlink()
    command = _command(job)
    with print_lock:
        print(f"[start] [{job.index}/150] {job.dataset}/{job.model}/seed{job.seed} on {job.device}", flush=True)
        print(f"        {_format_command(command)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    log_path = job.run_dir / "launcher.log"
    returncode = -1
    peak = None
    try:
        with log_path.open("w", encoding="utf-8") as launcher_log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=launcher_log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                current = _peak_gpu_memory_mib(process.pid)
                if current is not None:
                    peak = current if peak is None else max(peak, current)
                time.sleep(2.0)
            returncode = int(process.returncode)
        elapsed = time.monotonic() - started
        record, _ = _collect_record(job, output_root, elapsed, peak, returncode)
        status = "completed" if record["status"] == "PASS" else "failed_validation"
        message = "" if status == "completed" else "post-run health/provenance checks failed"
    except Exception as exc:  # the launcher records the failure and continues other jobs
        elapsed = time.monotonic() - started
        _dump(job.run_dir / "validation.json", {"status": "FAIL", "error": repr(exc), "returncode": returncode})
        status = "failed"
        message = repr(exc)
    with print_lock:
        marker = "done" if status == "completed" else "FAIL"
        print(f"[{marker}] [{job.index}/150] {job.dataset}/{job.model}/seed{job.seed} ({elapsed / 60.0:.1f} min)", flush=True)
        if message:
            print(f"       {message}; inspect {job.run_dir}", flush=True)
    return JobResult(job.index, job.dataset, job.model, job.seed, job.device, status, returncode, elapsed, str(job.run_dir), message)


def _run_parallel(jobs: list[Job], output_root: Path, devices: list[str], resume: bool) -> list[JobResult]:
    pending: queue.Queue[Job | None] = queue.Queue()
    for job in jobs:
        pending.put(job)
    for _ in devices:
        pending.put(None)
    print_lock = threading.Lock()
    results: list[JobResult] = []
    results_lock = threading.Lock()
    attempts_path = output_root / "attempts.jsonl"

    def worker(device: str) -> None:
        while True:
            job = pending.get()
            if job is None:
                return
            # Job device is assigned in the manifest; replace it with the lane
            # device so a resumed/filtered schedule is still safe.
            lane_job = Job(job.index, job.dataset, job.model, job.seed, device, job.run_dir)
            result = _run_one(lane_job, output_root, print_lock, resume)
            with results_lock:
                results.append(result)
                with attempts_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    threads = [threading.Thread(target=worker, args=(device,), name=f"nc-worker-{device}") for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(results, key=lambda item: item.index)


def _write_manifest(output_root: Path, jobs: list[Job], devices: list[str], mode: str) -> None:
    _dump(
        output_root / "benchmark_manifest.json",
        {
            "runner": Path(__file__).name,
            "mode": mode,
            "protocol": PROTOCOL,
            "datasets": list(DATASETS),
            "models": list(MODELS),
            "display_names": DISPLAY_NAMES,
            "seeds": list(SEEDS),
            "jobs": len(jobs),
            "devices": devices,
            "lp_started": False,
            "jobs_detail": [
                {"index": job.index, "dataset": job.dataset, "model": job.model, "seed": job.seed, "run_dir": str(job.run_dir)}
                for job in jobs
            ],
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NC-only final paper benchmark runner")
    parser.add_argument("--devices", nargs="+", default=None, help="GPU lanes, e.g. cuda:0 cuda:1; defaults to visible GPUs or cpu")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the 150 commands without training")
    parser.add_argument("--smoke", action="store_true", help="Run one 3-epoch Movies/MLP NC smoke job only")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _smoke(args: argparse.Namespace, output_root: Path, device: str) -> int:
    smoke_dir = output_root / "smoke" / "Movies" / "mlp" / "seed42"
    job = Job(0, "Movies", "mlp", 42, device, smoke_dir)
    command = _command(job) + ["task.epochs=3", "task.patience=3", "task.early_stop_min_epoch=1"]
    # The smoke path is intentionally separate and visibly non-formal.
    smoke_dir.mkdir(parents=True, exist_ok=True)
    print(_format_command(command), flush=True)
    with (smoke_dir / "launcher.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"}, stdout=log, stderr=subprocess.STDOUT, check=False)
    if process.returncode != 0:
        print(f"Smoke failed; inspect {smoke_dir / 'launcher.log'}", flush=True)
        return 1
    print(f"Smoke passed: {smoke_dir / 'results.json'}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    devices = _resolve_devices(args.devices)
    if not devices:
        raise SystemExit("No execution device was selected")

    if args.smoke:
        return _smoke(args, output_root, devices[0])

    jobs = _build_jobs(output_root, devices)
    _write_manifest(output_root, jobs, devices, "dry_run" if args.dry_run else "formal")
    preflight = _preflight(output_root, devices)
    print(f"Preflight PASS: {preflight['jobs']} NC jobs, fixed labels recorded for {len(DATASETS)} datasets", flush=True)
    if args.dry_run:
        for job in jobs:
            lane_job = Job(job.index, job.dataset, job.model, job.seed, devices[(job.index - 1) % len(devices)], job.run_dir)
            print(f"[{job.index}/150] {_format_command(_command(lane_job))}", flush=True)
        print(f"Manifest: {output_root / 'benchmark_manifest.json'}", flush=True)
        return 0

    results = _run_parallel(jobs, output_root, devices, resume=not args.no_resume)
    summary = {
        "protocol": PROTOCOL,
        "total": len(jobs),
        "completed": sum(item.status == "completed" for item in results),
        "skipped": sum(item.status == "skipped" for item in results),
        "failed": sum(item.status in {"failed", "failed_validation"} for item in results),
        "lp_started": False,
        "results": [asdict(item) for item in results],
    }
    _dump(output_root / "benchmark_summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("total", "completed", "skipped", "failed", "lp_started")}, ensure_ascii=False), flush=True)
    if summary["failed"]:
        print("Formal NC is incomplete; rerun the failed jobs after inspecting validation.json.", flush=True)
        return 1

    summarizer = ROOT / "scripts" / "summarize_paper_nc_final.py"
    completed = subprocess.run([sys.executable, str(summarizer), "--output-root", str(output_root)], cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
