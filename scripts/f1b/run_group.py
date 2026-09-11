#!/usr/bin/env python3
"""Prepare or execute one frozen F1-B benchmark job group.

The launcher deliberately runs one seed per Hydra process.  This gives every
seed an independent output directory and provenance record, while preserving
the repository's existing ``src.main`` training and evaluation semantics.
No free-form Hydra overrides are accepted: the formal command is assembled
from the frozen F1 protocol below.

Examples (from the project root)::

    python scripts/f1b/run_group.py --group all --dry-run
    python scripts/f1b/run_group.py --group mopf-nc --gpu-id 0

The shell files in this directory are the intended user-facing entry points.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "f1_final_execution"
SMOKE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "f1b_smoke"
CONFIG_MODEL_DIR = PROJECT_ROOT / "configs" / "model"
FORMAL_CONFIG_PATH = CONFIG_MODEL_DIR / "mopf.yaml"

METHOD_FREEZE_SHA = "4ddbd6918ceebadc25eed2694e1b463f9aac87f4"
FORMAL_CONFIG_SHA256 = "1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a"

SEEDS = (42, 43, 44)
NC_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
NC_K = {"Movies": 3, "Toys": 3, "Grocery": 2, "ele-fashion": 3, "Reddit-S": 3}
EXTERNAL_BASELINES = (
    "mlp",
    "gcn",
    "sage",
    "mmgcn",
    "mgat",
    "dip",
    "dgf",
    "dmgc",
    "lgmrec",
)
BASELINE_SHARDS = {
    "shard0": EXTERNAL_BASELINES[:5],
    "shard1": EXTERNAL_BASELINES[5:],
    "all": EXTERNAL_BASELINES,
}


@dataclass(frozen=True)
class Job:
    task: str
    dataset: str
    model: str
    seed: int
    evaluation_scope: str
    role: str
    output_dir: Path
    source_output_dir: Path | None = None
    formal_k: int | None = None
    execution_mode: str = "fresh"

    @property
    def run_key(self) -> str:
        return "|".join(
            (self.task, self.dataset, self.model, str(self.seed), self.evaluation_scope)
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _normalize_device(value: str) -> str:
    value = str(value).strip()
    if value.isdigit():
        return f"cuda:{value}"
    if value == "cpu" or value.startswith("cuda:"):
        return value
    raise ValueError("GPU_ID/device must be an integer, cuda:<index>, or cpu")


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _job(
    *,
    root: Path,
    task: str,
    dataset: str,
    model: str,
    seed: int,
    scope: str,
    role: str,
    formal_k: int | None = None,
    source_output_dir: Path | None = None,
    execution_mode: str = "fresh",
) -> Job:
    if task == "nc":
        output_dir = root / "nc" / dataset / model / f"seed{seed}"
    elif scope == "formal":
        output_dir = root / "lp_formal" / dataset / model / f"seed{seed}"
    else:
        output_dir = root / "lp_quasi_held_out" / dataset / model / f"seed{seed}"
    return Job(
        task=task,
        dataset=dataset,
        model=model,
        seed=seed,
        evaluation_scope=scope,
        role=role,
        output_dir=output_dir,
        source_output_dir=source_output_dir,
        formal_k=formal_k,
        execution_mode=execution_mode,
    )


def build_jobs(
    group: str,
    root: Path,
    *,
    nc_policy: str,
    shard: str,
    smoke: bool,
) -> list[Job]:
    jobs: list[Job] = []
    nc_datasets = ("Movies",) if smoke else NC_DATASETS
    seeds = (42,) if smoke else SEEDS

    if group in {"all", "mopf-nc"}:
        for dataset in nc_datasets:
            for seed in seeds:
                source = (
                    PROJECT_ROOT
                    / "outputs"
                    / "u3b_transport_conditioned_composition"
                    / "runs"
                    / dataset
                    / "B1_TCPR"
                    / f"seed{seed}"
                )
                jobs.append(
                    _job(
                        root=root,
                        task="nc",
                        dataset=dataset,
                        model="mopf",
                        seed=seed,
                        scope="formal",
                        role="final_MoPF",
                        formal_k=NC_K[dataset],
                        source_output_dir=source,
                        execution_mode="reuse" if nc_policy == "reuse" else "fresh",
                    )
                )

    if group in {"all", "mopf-lp"}:
        lp_seeds = (42,) if smoke else SEEDS
        jobs.extend(
            _job(
                root=root,
                task="lp",
                dataset="sports-copurchase",
                model="mopf",
                seed=seed,
                scope="formal",
                role="final_MoPF",
            )
            for seed in lp_seeds
        )
        jobs.extend(
            _job(
                root=root,
                task="lp",
                dataset="cloth-copurchase",
                model="mopf",
                seed=seed,
                scope="quasi-held-out",
                role="final_MoPF",
            )
            for seed in lp_seeds
        )

    if group in {"all", "cloth-baselines-lp"}:
        models = ("mlp",) if smoke else BASELINE_SHARDS[shard]
        lp_seeds = (42,) if smoke else SEEDS
        jobs.extend(
            _job(
                root=root,
                task="lp",
                dataset="cloth-copurchase",
                model=model,
                seed=seed,
                scope="quasi-held-out",
                role="formal_external_baseline",
            )
            for model in models
            for seed in lp_seeds
        )

    if group == "mopf-nc" and nc_policy == "reuse":
        return jobs
    return jobs


def _validate_jobs(jobs: list[Job], *, smoke: bool, nc_policy: str, group: str, shard: str) -> None:
    if not jobs:
        raise ValueError("no jobs selected")
    keys = [job.run_key for job in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError("job plan contains duplicate run keys")
    for job in jobs:
        if not (CONFIG_MODEL_DIR / f"{job.model}.yaml").is_file():
            raise FileNotFoundError(f"missing model config: {job.model}")
    if smoke:
        expected = {"mopf-nc": 1, "mopf-lp": 2, "cloth-baselines-lp": 1, "all": 4}[group]
        if len(jobs) != expected:
            raise AssertionError(f"smoke plan expected {expected} jobs, got {len(jobs)}")
    elif group == "all":
        # The plan still contains all 15 NC rows when they are marked reuse;
        # only the number of new full launches drops from 48 to 33.
        expected = 48
        if len(jobs) != expected:
            raise AssertionError(f"full F1-B plan expected {expected} entries, got {len(jobs)}")
    elif group == "mopf-nc" and len(jobs) != 15:
        raise AssertionError(f"MoPF NC plan expected 15 entries, got {len(jobs)}")
    elif group == "mopf-lp" and len(jobs) != 6:
        raise AssertionError(f"MoPF LP plan expected 6 entries, got {len(jobs)}")
    elif group == "cloth-baselines-lp":
        expected = {"all": 27, "shard0": 15, "shard1": 12}[shard]
        if len(jobs) != expected:
            raise AssertionError(f"cloth baseline plan expected {expected} entries, got {len(jobs)}")


def _job_row(job: Job, command: str = "") -> dict[str, str]:
    return {
        "run_key": job.run_key,
        "task": job.task,
        "dataset": job.dataset,
        "model": job.model,
        "seed": str(job.seed),
        "formal_K": "" if job.formal_k is None else str(job.formal_k),
        "evaluation_scope": job.evaluation_scope,
        "role": job.role,
        "execution_mode": job.execution_mode,
        "status": "PLANNED",
        "output_dir": str(job.output_dir),
        "source_output_dir": "" if job.source_output_dir is None else str(job.source_output_dir),
        "command": command,
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


MANIFEST_FIELDS = [
    "run_key",
    "task",
    "dataset",
    "model",
    "seed",
    "formal_K",
    "evaluation_scope",
    "role",
    "execution_mode",
    "status",
    "output_dir",
    "source_output_dir",
    "command",
    "method_freeze_sha",
    "formal_config_sha256",
]
STATUS_FIELDS = [
    "task",
    "dataset",
    "model",
    "seed",
    "status",
    "exit_code",
    "output_dir",
    "start_time",
    "end_time",
    "run_key",
    "evaluation_scope",
    "role",
    "message",
]


def _formal_overrides(job: Job) -> list[str]:
    if job.task == "nc" and job.model == "mopf":
        assert job.formal_k is not None
        return [f"model.max_order={job.formal_k}", f"model.num_layers={job.formal_k}"]
    return []


def build_command(job: Job, device: str, *, smoke: bool) -> list[str]:
    checkpoint = job.output_dir / "best.pt"
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={job.dataset}",
        f"task={job.task}",
        f"model={job.model}",
        f"seed={job.seed}",
        "num_runs=1",
        f"device={device}",
        f"hydra.run.dir={job.output_dir}",
        f"task.save_ckpt_path={checkpoint}",
    ]
    command.extend(_formal_overrides(job))
    if smoke:
        smoke_overrides = [
            "task.epochs=1",
            "task.max_train_batches=1",
            "task.patience=1",
            "task.early_stop_min_epoch=1",
            "task.eval_every=1",
            "task.evaluate_test=false",
        ]
        if job.task == "lp":
            smoke_overrides.extend(["task.loader_num_workers=0", "task.torch_threads=2"])
        else:
            smoke_overrides.append("+task.torch_threads=2")
        command.extend(
            smoke_overrides
        )
    return command


def _format_command(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def _is_valid_metrics(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        return False
    return bool(payload["metrics"])


def is_complete(job: Job) -> bool:
    return (
        job.execution_mode == "fresh"
        and (job.output_dir / "complete.marker").is_file()
        and _is_valid_metrics(job.output_dir / "metrics.json")
        and (job.output_dir / "best.pt").is_file()
        and (job.output_dir / "best.pt").stat().st_size > 0
    )


def _existing_state(job: Job) -> str:
    if is_complete(job):
        return "SKIP_COMPLETE"
    if job.output_dir.exists() and any(job.output_dir.iterdir()):
        return "INCOMPLETE"
    return "MISSING"


def _write_run_metadata(job: Job, command: list[str], device: str, execution_head: str) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    (job.output_dir / "command.txt").write_text(
        _format_command(command) + "\n", encoding="utf-8"
    )
    (job.output_dir / "git_sha.txt").write_text(execution_head + "\n", encoding="utf-8")
    (job.output_dir / "seed.txt").write_text(str(job.seed) + "\n", encoding="utf-8")
    (job.output_dir / "device.txt").write_text(device + "\n", encoding="utf-8")
    (job.output_dir / "formal_config_sha256.txt").write_text(
        FORMAL_CONFIG_SHA256 + "\n", encoding="utf-8"
    )
    payload = {
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "execution_head": execution_head,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
        "task": job.task,
        "dataset": job.dataset,
        "model": job.model,
        "seed": job.seed,
        "device": device,
        "evaluation_scope": job.evaluation_scope,
        "formal_or_quasi_held_out": job.evaluation_scope,
        "formal_K": job.formal_k,
        "role": job.role,
        "selection_metric": "val_acc" if job.task == "nc" else "val_mrr",
        "test_role": "descriptive_only",
        "command": _format_command(command),
        "recorded_at": _utc_now(),
    }
    (job.output_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _finalize_run(job: Job, execution_head: str) -> bool:
    hydra_config = job.output_dir / ".hydra" / "config.yaml"
    if hydra_config.is_file():
        shutil.copyfile(hydra_config, job.output_dir / "resolved_config.yaml")
    main_log = job.output_dir / "main.log"
    if main_log.is_file():
        shutil.copyfile(main_log, job.output_dir / "training.log")
    results_path = job.output_dir / "results.json"
    if not results_path.is_file() or not (job.output_dir / "best.pt").is_file():
        return False
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metrics = {
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "execution_head": execution_head,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
        "task": job.task,
        "dataset": job.dataset,
        "model": job.model,
        "seed": job.seed,
        "evaluation_scope": job.evaluation_scope,
        "formal_or_quasi_held_out": job.evaluation_scope,
        "selection_metric": "val_acc" if job.task == "nc" else "val_mrr",
        "test_role": "descriptive_only",
        "metrics": results,
        "source_results": "results.json",
    }
    (job.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not _is_valid_metrics(job.output_dir / "metrics.json"):
        return False
    (job.output_dir / "complete.marker").write_text(
        json.dumps({"completed_at": _utc_now(), "status": "complete"}) + "\n",
        encoding="utf-8",
    )
    return True


def _read_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["run_key"]: row for row in csv.DictReader(handle) if row.get("run_key")}


def _write_status(path: Path, rows: dict[str, dict[str, str]]) -> None:
    ordered = [rows[key] for key in sorted(rows)]
    _write_csv(path, ordered, STATUS_FIELDS)


def _status_row(job: Job, status: str, exit_code: int | None, start: str, end: str, message: str) -> dict[str, str]:
    return {
        "task": job.task,
        "dataset": job.dataset,
        "model": job.model,
        "seed": str(job.seed),
        "status": status,
        "exit_code": "" if exit_code is None else str(exit_code),
        "output_dir": str(job.output_dir),
        "start_time": start,
        "end_time": end,
        "run_key": job.run_key,
        "evaluation_scope": job.evaluation_scope,
        "role": job.role,
        "message": message,
    }


def execute_jobs(jobs: list[Job], device: str, *, root: Path, dry_run: bool, smoke: bool, resume: bool, overwrite: bool) -> int:
    status_path = root / "run_status.csv"
    status_rows = _read_status(status_path)
    execution_head = _git_head()
    failures = 0

    for job in jobs:
        command = build_command(job, device, smoke=smoke)
        if dry_run:
            if job.execution_mode == "reuse":
                print(f"[DRY-RUN/REUSE] {job.run_key}")
                print(f"  no training; source={job.source_output_dir}")
            else:
                print(f"[DRY-RUN] {job.run_key}")
                print(f"  {_format_command(command)}")
            status_rows[job.run_key] = _status_row(
                job, "PLANNED", None, _utc_now(), _utc_now(), "dry-run; no training launched"
            )
            continue

        if job.execution_mode == "reuse":
            source = job.source_output_dir
            valid = bool(
                source
                and (source / "best.pt").is_file()
                and (source / "results.json").is_file()
                and (source / ".hydra" / "config.yaml").is_file()
            )
            status = "REUSED" if valid else "FAILED"
            message = "F1-A REUSE_EXACT U3-B1 source" if valid else "missing U3-B1 reuse source"
            if not valid:
                failures += 1
            print(f"[{status}] {job.run_key}: {source}")
            status_rows[job.run_key] = _status_row(
                job, status, 0 if valid else 1, _utc_now(), _utc_now(), message
            )
            _write_status(status_path, status_rows)
            continue

        state = _existing_state(job)
        if state == "SKIP_COMPLETE" and not overwrite:
            print(f"[SKIP_COMPLETE] {job.run_key}: {job.output_dir}")
            status_rows[job.run_key] = _status_row(job, state, 0, _utc_now(), _utc_now(), "valid marker, metrics, and checkpoint")
            _write_status(status_path, status_rows)
            continue
        if state == "INCOMPLETE" and not (resume or overwrite):
            print(f"[INCOMPLETE] {job.run_key}: use --resume or --overwrite: {job.output_dir}")
            failures += 1
            status_rows[job.run_key] = _status_row(job, "INCOMPLETE", 2, _utc_now(), _utc_now(), "explicit --resume or --overwrite required")
            _write_status(status_path, status_rows)
            continue

        start = _utc_now()
        _write_run_metadata(job, command, device, execution_head)
        log_path = job.output_dir / "launcher.log"
        print(f"[START] {job.run_key} on {device}")
        print(f"  {_format_command(command)}")
        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"[{start}] {_format_command(command)}\n")
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            exit_code = int(completed.returncode)
        except OSError as exc:
            exit_code = 127
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"launcher error: {exc}\n")
        end = _utc_now()
        if exit_code == 0 and _finalize_run(job, execution_head):
            status = "COMPLETED"
            message = "complete marker, metrics, and checkpoint written"
        else:
            status = "FAILED"
            message = "child failed or required artifacts were not produced"
            failures += 1
        print(f"[{status}] {job.run_key}: {job.output_dir}")
        status_rows[job.run_key] = _status_row(job, status, exit_code, start, end, message)
        _write_status(status_path, status_rows)

    _write_status(status_path, status_rows)
    return min(failures, 125)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or execute the frozen MoPF-vNext F1-B job groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--group", choices=("all", "mopf-nc", "mopf-lp", "cloth-baselines-lp"), required=True)
    parser.add_argument("--gpu-id", "--device", dest="gpu_id", default="0", help="Explicit device; integer 0/1 becomes cuda:0/cuda:1.")
    parser.add_argument("--shard", choices=tuple(BASELINE_SHARDS), default="all", help="Deterministic cloth-baseline shard.")
    parser.add_argument("--output-root", type=Path, default=FORMAL_OUTPUT_ROOT)
    parser.add_argument("--nc-policy", choices=("fresh", "reuse"), default="fresh", help="MoPF NC fresh reevaluation or F1-A exact U3-B1 reuse.")
    parser.add_argument("--dry-run", action="store_true", help="Print every command and do not launch training.")
    parser.add_argument("--resume", action="store_true", help="Explicitly rerun incomplete output in place; the current framework has no optimizer-resume state.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerun output in place without deleting existing files.")
    parser.add_argument("--smoke", action="store_true", help="Run only the short smoke subset into outputs/f1b_smoke/.")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if os.environ.get("DRY_RUN", "").strip().lower() in {"1", "true", "yes", "on"}:
        args.dry_run = True
    if args.group != "cloth-baselines-lp" and args.shard != "all":
        parser.error("--shard is only valid for cloth-baselines-lp")
    if args.nc_policy == "reuse" and args.group not in {"all", "mopf-nc"}:
        parser.error("--nc-policy reuse is only valid when the plan includes MoPF NC")
    return args


def main() -> int:
    args = _parse_args()
    actual_sha = _sha256(FORMAL_CONFIG_PATH)
    if actual_sha != FORMAL_CONFIG_SHA256:
        raise RuntimeError(
            "formal MoPF config SHA changed; refusing to prepare/execute F1-B: "
            f"expected {FORMAL_CONFIG_SHA256}, got {actual_sha}"
        )
    device = _normalize_device(args.gpu_id)
    root = SMOKE_OUTPUT_ROOT if args.smoke else _absolute(args.output_root)
    jobs = build_jobs(
        args.group,
        root,
        nc_policy=args.nc_policy,
        shard=args.shard,
        smoke=args.smoke,
    )
    _validate_jobs(jobs, smoke=args.smoke, nc_policy=args.nc_policy, group=args.group, shard=args.shard)

    rows = [_job_row(job, "" if job.execution_mode == "reuse" else _format_command(build_command(job, device, smoke=args.smoke))) for job in jobs]
    if args.group == "all" and not args.smoke:
        manifest_path = root / "f1b_plan.csv"
    else:
        manifest_path = root / f"f1b_plan_{args.group}.csv"
    _write_csv(manifest_path, rows, MANIFEST_FIELDS)
    print(f"Plan: {manifest_path}")
    print(f"Planned entries: {len(jobs)}")
    if args.group == "all" and not args.smoke:
        full_runs = sum(job.execution_mode == "fresh" for job in jobs)
        reused = sum(job.execution_mode == "reuse" for job in jobs)
        print(f"New full runs: {full_runs}; exact reused NC entries: {reused}")
    if args.smoke:
        print("SMOKE ONLY: task budget is one epoch/one batch; formal output root is not used")
    return execute_jobs(
        jobs,
        device,
        root=root,
        dry_run=args.dry_run,
        smoke=args.smoke,
        resume=args.resume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
