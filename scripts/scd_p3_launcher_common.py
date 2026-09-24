#!/usr/bin/env python3
"""Shared planning and resume-safe execution helpers for the P3 experiment matrix.

Dry-runs write only provenance.plan.json and never create a formal lock or
invoke src.main. Formal execution is deliberately separate and requires a
clean, synchronized V3 worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
FROZEN_S_SHA256 = "5b00505db5c504e9e6c57006673bd84c18e71504fbe13f97e0fe12df41c49132"
REQUIRED_ARTIFACTS = (
    "best.pt",
    "metrics.json",
    "results.json",
    "resolved_config.json",
    "complete.marker",
)
NC_METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
LP_METRICS = ("val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10")


@dataclass(frozen=True)
class Job:
    index: int
    label: str
    model: str
    ablation: str
    task: str
    dataset: str
    seed: int
    device: str
    run_dir: str


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    dataset_choices: Iterable[str],
    default_datasets: list[str],
    default_output: str,
) -> None:
    choices = list(dataset_choices)
    parser.add_argument("--datasets", nargs="+", choices=choices, default=default_datasets)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--output-root", default=default_output)
    parser.add_argument("--dry-run", action="store_true")


def resolve_devices(args: argparse.Namespace) -> list[str]:
    if args.devices is not None and args.device != "cuda:0":
        raise ValueError("pass either --device or --devices, not both")
    values = list(args.devices) if args.devices is not None else [str(args.device)]
    if not values:
        raise ValueError("at least one device is required")
    return values


def root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def make_jobs(
    output: Path,
    *,
    task: str,
    datasets: list[str],
    seeds: list[int],
    devices: list[str],
    variants: list[tuple[str, str, str]],
) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for label, model, ablation in variants:
        for dataset in datasets:
            for seed in seeds:
                index += 1
                run_dir = output / label / dataset / f"seed{seed}"
                jobs.append(
                    Job(
                        index=index,
                        label=label,
                        model=model,
                        ablation=ablation,
                        task=task,
                        dataset=dataset,
                        seed=int(seed),
                        device=devices[(index - 1) % len(devices)],
                        run_dir=str(run_dir),
                    )
                )
    return jobs


def command(job: Job, protocol: str) -> list[str]:
    run = Path(job.run_dir)
    training_mode = "full_graph" if job.task == "nc" else "sampled"
    return [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={job.dataset}",
        f"task={job.task}",
        f"model={job.model}",
        f"seed={job.seed}",
        "num_runs=1",
        f"device={job.device}",
        f"ablation={job.ablation}",
        f"task.training_mode={training_mode}",
        "task.evaluate_test=true",
        f"task.protocol_version={protocol}",
        f"task.save_ckpt_path={run / 'best.pt'}",
        f"hydra.run.dir={run}",
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config_path(kind: str, name: str) -> Path:
    if kind == "model":
        return ROOT / "configs" / "model" / f"{name}.yaml"
    if kind == "dataset":
        return ROOT / "configs" / "dataset" / f"{name}.yaml"
    if kind == "task":
        return ROOT / "configs" / "task" / f"{name}.yaml"
    raise ValueError(kind)


def _source_path(model: str) -> Path:
    return ROOT / "src" / "models" / f"{model}.py"


def _source_dependencies(model: str) -> list[Path]:
    dependencies = [ROOT / "src" / "models" / f"{model}.py"]
    if model == "ssi_mag_scd_ablation":
        dependencies.append(ROOT / "src" / "models" / "ssi_mag_scd.py")
    if model in {"ssi_mag_generic_ppr", "ssi_mag_generic_gpr"}:
        dependencies.append(ROOT / "src" / "models" / "ssi_mag_generic_common.py")
    return dependencies


def provenance(
    *,
    suite: str,
    task: str,
    protocol: str,
    jobs: list[Job],
    output: Path,
    selection_metric: str,
    checkpoint_selection: str,
    dry_run: bool,
) -> dict[str, Any]:
    models = sorted({job.model for job in jobs})
    datasets = sorted({job.dataset for job in jobs})
    seeds = sorted({job.seed for job in jobs})
    model_sources = {}
    model_configs = {}
    for model in models:
        source = _source_path(model)
        config = _config_path("model", model)
        if not source.is_file() or not config.is_file():
            raise FileNotFoundError(f"missing model source/config for {model}")
        for dependency in _source_dependencies(model):
            if not dependency.is_file():
                raise FileNotFoundError(f"missing model dependency: {dependency}")
            dependency_hash = sha256(dependency)
            if dependency.name == "ssi_mag_scd.py" and dependency_hash != FROZEN_S_SHA256:
                raise RuntimeError(
                    "frozen S source hash changed: "
                    f"{dependency_hash} != {FROZEN_S_SHA256}"
                )
            model_sources[str(dependency.relative_to(ROOT))] = dependency_hash
        model_configs[str(config.relative_to(ROOT))] = sha256(config)
    task_config = _config_path("task", task)
    data_configs = {}
    for dataset in datasets:
        config = _config_path("dataset", dataset)
        data_configs[str(config.relative_to(ROOT))] = sha256(config)
    return {
        "schema": "mopf_p3_scd_controls_provenance_v1",
        "suite": suite,
        "git_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "model_sources": model_sources,
        "model_configs": model_configs,
        "task_config": {
            str(task_config.relative_to(ROOT)): sha256(task_config)
        },
        "data_configs": data_configs,
        "task": task,
        "protocol": protocol,
        "datasets": datasets,
        "seeds": seeds,
        "variants": sorted({job.label for job in jobs}),
        "modes": sorted({job.model for job in jobs}),
        "planned_jobs": len(jobs),
        "output_root": str(output),
        "lp_num_neighbors": [5, 5, 5] if task == "lp" else None,
        "selection_metric": selection_metric,
        "checkpoint_selection": checkpoint_selection,
        "test_used_for_selection": False,
        "dry_run": bool(dry_run),
        "training_started": False,
        "formal_training_started": False,
    }


def plan_payload(
    *,
    suite: str,
    protocol: str,
    jobs: list[Job],
    provenance_payload: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    return {
        **provenance_payload,
        "launcher_suite": suite,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_plan": str(output / "provenance.plan.json"),
        "jobs": [
            {
                **asdict(job),
                "command": shlex.join(command(job, protocol)),
                "status": "planned",
            }
            for job in jobs
        ],
    }


def write_dry_run_plan(
    *,
    output: Path,
    suite: str,
    protocol: str,
    jobs: list[Job],
    provenance_payload: dict[str, Any],
) -> Path:
    # This is intentionally the only filesystem write made by a dry-run.
    plan = output / "provenance.plan.json"
    dump_json(
        plan,
        plan_payload(
            suite=suite,
            protocol=protocol,
            jobs=jobs,
            provenance_payload=provenance_payload,
            output=output,
        ),
    )
    return plan


def assert_clean_synced() -> None:
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError("formal training requires a clean working tree")
    if git("branch", "--show-current") != "V3":
        raise RuntimeError("formal training requires branch V3")
    head, remote = git("rev-parse", "HEAD"), git("rev-parse", "origin/V3")
    if head == "unknown" or remote == "unknown" or head != remote:
        raise RuntimeError(f"formal training requires HEAD == origin/V3, got {head} vs {remote}")


def prepare_formal_lock(
    *,
    output: Path,
    suite: str,
    protocol: str,
    jobs: list[Job],
    provenance_payload: dict[str, Any],
) -> Path:
    del suite, protocol
    assert_clean_synced()
    lock = output / "provenance.lock.json"
    if lock.is_file():
        stored = json.loads(lock.read_text(encoding="utf-8"))
        if stored != provenance_payload:
            raise RuntimeError("provenance lock mismatch; refusing resume")
    else:
        if any(
            _is_complete(job, checkpoint_selection=provenance_payload["checkpoint_selection"])[0]
            for job in jobs
        ):
            raise RuntimeError(
                "completed artifacts exist without immutable provenance.lock.json"
            )
        dump_json(lock, provenance_payload)
    return lock


def _is_complete(job: Job, *, checkpoint_selection: str) -> tuple[bool, str]:
    run = Path(job.run_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (run / name).is_file()]
    if missing:
        return False, "missing " + ", ".join(missing)
    try:
        marker = json.loads((run / "complete.marker").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        results = json.loads((run / "results.json").read_text(encoding="utf-8"))
        resolved = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid metadata: {exc}"
    if (
        marker.get("status") != "complete"
        or marker.get("task") != job.task
        or marker.get("dataset") != job.dataset
        or int(marker.get("seed", -1)) != job.seed
        or marker.get("ablation") != job.ablation
    ):
        return False, "completion identity/status mismatch"
    if (
        metrics.get("task") != job.task
        or metrics.get("model") != job.model
        or metrics.get("checkpoint_selection") != checkpoint_selection
    ):
        return False, "metrics identity/selection mismatch"
    if (
        resolved.get("model", {}).get("name") != job.model
        or resolved.get("task", {}).get("name") != job.task
        or resolved.get("ablation") != job.ablation
    ):
        return False, "resolved model/task/ablation mismatch"
    required_metrics = NC_METRICS if job.task == "nc" else LP_METRICS
    try:
        for key in required_metrics:
            value = results[key]["mean"]
            if not math.isfinite(float(value)):
                return False, f"invalid {key}"
    except (KeyError, TypeError, ValueError):
        return False, "invalid required metric payload"
    return True, "complete"


def run_formal(
    *,
    output: Path,
    suite: str,
    protocol: str,
    jobs: list[Job],
    provenance_payload: dict[str, Any],
) -> int:
    lock = prepare_formal_lock(
        output=output,
        suite=suite,
        protocol=protocol,
        jobs=jobs,
        provenance_payload=provenance_payload,
    )
    failures: list[str] = []
    for job in jobs:
        complete, reason = _is_complete(
            job, checkpoint_selection=provenance_payload["checkpoint_selection"]
        )
        if complete:
            print(f"[skip] {job.label}/{job.dataset}/seed{job.seed}: {reason}", flush=True)
            continue
        print(
            f"[start] [{job.index}/{len(jobs)}] "
            f"{shlex.join(command(job, protocol))}",
            flush=True,
        )
        completed = subprocess.run(command(job, protocol), cwd=ROOT, check=False)
        if completed.returncode:
            failures.append(
                f"{job.label}/{job.dataset}/seed{job.seed}: returncode={completed.returncode}"
            )
            continue
        ok, why = _is_complete(
            job, checkpoint_selection=provenance_payload["checkpoint_selection"]
        )
        if not ok:
            failures.append(f"{job.label}/{job.dataset}/seed{job.seed}: {why}")
    manifest = plan_payload(
        suite=suite,
        protocol=protocol,
        jobs=jobs,
        provenance_payload=provenance_payload,
        output=output,
    )
    manifest.update(
        {
            "provenance_lock": str(lock),
            "formal_training_started": True,
            "completed_jobs": sum(
                _is_complete(
                    job, checkpoint_selection=provenance_payload["checkpoint_selection"]
                )[0]
                for job in jobs
            ),
            "failed_jobs": len(failures),
            "failures": failures,
        }
    )
    manifest["jobs"] = [
        {
            **asdict(job),
            "command": shlex.join(command(job, protocol)),
            "status": (
                "complete"
                if _is_complete(
                    job, checkpoint_selection=provenance_payload["checkpoint_selection"]
                )[0]
                else "failed"
            ),
        }
        for job in jobs
    ]
    dump_json(output / "manifest.json", manifest)
    return int(bool(failures))
''