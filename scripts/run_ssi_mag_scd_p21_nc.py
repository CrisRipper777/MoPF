#!/usr/bin/env python3
"""Resume-safe formal P2.1 clean SCD NC launcher.

The launcher owns exactly 15 NC jobs (five datasets x three seeds), uses the
unified full-graph NC protocol, and never launches LP jobs.  It requires a
clean V3 worktree synchronized with origin/V3 before creating the immutable
formal lock.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
MODEL = "ssi_mag_scd"
CONFIG = ROOT / "configs/model/ssi_mag_scd.yaml"
MODEL_SOURCE = ROOT / "src/models/ssi_mag_scd.py"
PROTOCOL = "unified_full_graph_nc_v1"
OUTPUT_DEFAULT = "outputs/ssi_mag_scd_p21_nc"


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    seed: int
    device: str
    run_dir: str


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=OUTPUT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _jobs(output: Path, datasets: list[str], seeds: list[int], device: str) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for dataset in datasets:
        for seed in seeds:
            index += 1
            jobs.append(Job(index, dataset, seed, device, str(output / dataset / f"seed{seed}")))
    return jobs


def _command(job: Job) -> list[str]:
    run = Path(job.run_dir)
    return [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={job.dataset}",
        "task=nc",
        f"model={MODEL}",
        f"seed={job.seed}",
        "num_runs=1",
        f"device={job.device}",
        "ablation=full",
        "task.training_mode=full_graph",
        "task.evaluate_test=true",
        f"task.protocol_version={PROTOCOL}",
        f"task.save_ckpt_path={run / 'best.pt'}",
        f"hydra.run.dir={run}",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _provenance(commit: str, datasets: list[str], seeds: list[int], *, dry_run: bool) -> dict[str, Any]:
    return {
        "schema": "ssi_mag_scd_p21_nc_provenance_v1",
        "git_commit": commit,
        "branch": _git("branch", "--show-current"),
        "model": MODEL,
        "model_source": str(MODEL_SOURCE.relative_to(ROOT)),
        "model_sha256": _sha256(MODEL_SOURCE),
        "config_source": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "task": "nc",
        "protocol": PROTOCOL,
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "planned_jobs": len(datasets) * len(seeds),
        "lp_jobs": 0,
        "selection_metric": "val_acc",
        "checkpoint_selection": "best_val_accuracy",
        "test_used_for_selection": False,
        "dry_run": bool(dry_run),
        "training_started": False,
    }


def _complete(job: Job) -> tuple[bool, str]:
    run = Path(job.run_dir)
    required = [run / name for name in ("best.pt", "metrics.json", "results.json", "resolved_config.json", "complete.marker")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return False, "missing " + ", ".join(missing)
    try:
        marker = json.loads((run / "complete.marker").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        results = json.loads((run / "results.json").read_text())
        resolved = json.loads((run / "resolved_config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid metadata: {exc}"
    if marker.get("status") != "complete" or marker.get("task") != "nc" or marker.get("dataset") != job.dataset or int(marker.get("seed", -1)) != job.seed:
        return False, "completion identity/status mismatch"
    if metrics.get("task") != "nc" or metrics.get("model") != MODEL or metrics.get("checkpoint_selection") != "best_val_accuracy":
        return False, "metrics identity/selection mismatch"
    if resolved.get("model", {}).get("name") != MODEL or resolved.get("task", {}).get("name") != "nc":
        return False, "resolved model/task mismatch"
    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        if key not in results or not math.isfinite(float(results[key]["mean"])):
            return False, f"invalid {key}"
    return True, "complete"


def _assert_clean_synced() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("formal training requires a clean working tree")
    if _git("branch", "--show-current") != "V3":
        raise RuntimeError("formal training requires branch V3")
    head, remote = _git("rev-parse", "HEAD"), _git("rev-parse", "origin/V3")
    if head == "unknown" or remote == "unknown" or head != remote:
        raise RuntimeError(f"formal training requires HEAD == origin/V3, got {head} vs {remote}")


def _manifest(args: argparse.Namespace, jobs: list[Job], provenance: dict[str, Any], plan: Path, lock: Path | None) -> dict[str, Any]:
    return {
        "launcher": "run_ssi_mag_scd_p21_nc.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": provenance["git_commit"],
        "branch": provenance["branch"],
        "task": "nc",
        "model": MODEL,
        "protocol": PROTOCOL,
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "device": args.device,
        "dry_run": bool(args.dry_run),
        "lp_jobs": 0,
        "selection_metric": "val_acc",
        "checkpoint_selection": "best_val_accuracy",
        "test_used_for_selection": False,
        "provenance_plan": str(plan),
        "provenance_lock": str(lock) if lock else None,
        "formal_provenance": provenance if not args.dry_run else None,
        "jobs": [{**asdict(job), "command": shlex.join(_command(job)), "status": "planned" if args.dry_run else "pending"} for job in jobs],
    }


def _prepare_dry_run(output: Path, args: argparse.Namespace, jobs: list[Job], provenance: dict[str, Any]) -> Path:
    lock = output / "provenance.lock.json"
    if lock.is_file():
        stored = json.loads(lock.read_text())
        if not stored.get("dry_run", False):
            raise RuntimeError("formal root cannot be replaced by a dry-run plan")
        lock.rename(output / "provenance.lock.legacy-dry-run.json")
    plan = output / "provenance.plan.json"
    _dump(plan, provenance)
    _dump(output / "manifest.json", _manifest(args, jobs, provenance, plan, None))
    return plan


def _prepare_formal_lock(output: Path, args: argparse.Namespace, jobs: list[Job], provenance: dict[str, Any]) -> Path:
    _assert_clean_synced()
    lock = output / "provenance.lock.json"
    if lock.is_file():
        stored = json.loads(lock.read_text())
        if stored != provenance:
            raise RuntimeError("provenance lock mismatch; refusing resume")
    else:
        if any(_complete(job)[0] for job in jobs):
            raise RuntimeError("completed artifacts exist without immutable provenance.lock.json")
        _dump(lock, provenance)
    _dump(output / "manifest.json", _manifest(args, jobs, provenance, output / "provenance.plan.json", lock))
    return lock


def main() -> int:
    args = _args()
    output = _root(args.output_root)
    jobs = _jobs(output, list(args.datasets), list(args.seeds), args.device)
    provenance = _provenance(_git("rev-parse", "HEAD"), list(args.datasets), list(args.seeds), dry_run=bool(args.dry_run))
    output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        _prepare_dry_run(output, args, jobs, provenance)
        print(f"Dry-run planned {len(jobs)} NC jobs; LP jobs: 0; training started: false")
        return 0
    lock = _prepare_formal_lock(output, args, jobs, provenance)
    failures: list[str] = []
    for job in jobs:
        complete, reason = _complete(job)
        if complete:
            print(f"[skip] {job.dataset}/seed{job.seed}: {reason}", flush=True)
            continue
        print(f"[start] [{job.index}/{len(jobs)}] {shlex.join(_command(job))}", flush=True)
        completed = subprocess.run(_command(job), cwd=ROOT, check=False)
        if completed.returncode:
            failures.append(f"{job.dataset}/seed{job.seed}: returncode={completed.returncode}")
            continue
        ok, why = _complete(job)
        if not ok:
            failures.append(f"{job.dataset}/seed{job.seed}: {why}")
    manifest = _manifest(args, jobs, provenance, output / "provenance.plan.json", lock)
    manifest["training_started"] = True
    manifest["jobs"] = [{**asdict(job), "command": shlex.join(_command(job)), "status": "complete" if _complete(job)[0] else "failed"} for job in jobs]
    manifest["completed_jobs"] = sum(item["status"] == "complete" for item in manifest["jobs"])
    manifest["failed_jobs"] = len(failures)
    manifest["failures"] = failures
    _dump(output / "manifest.json", manifest)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
