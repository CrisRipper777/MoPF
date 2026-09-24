#!/usr/bin/env python3
"""Resume-safe planner/launcher for the frozen final NC ablation matrix.

P1.9 only executes this launcher in ``--dry-run`` mode. The implementation is
kept resume-safe for the pre-registered future matrix: 7 ablations x 5 NC
datasets x 3 seeds = 105 jobs, with no LP branch.
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
ABLATIONS = [
    "no_relation_modulation",
    "fixed_semantic_reference",
    "last_context_only",
    "no_context_change",
    "no_cross_hop_interaction",
    "global_filter_only",
    "no_formation_conditioning",
]
MODEL = "ssi_mag_final_ablation"
CONFIG = ROOT / "configs/model/ssi_mag_final_ablation.yaml"
MODEL_SOURCE = ROOT / "src/models/ssi_mag_final_ablation.py"
PROTOCOL = "unified_full_graph_nc_v1"


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    seed: int
    ablation: str
    device: str
    run_dir: str


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--ablations", nargs="+", choices=ABLATIONS, default=ABLATIONS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/ssi_mag_final_nc_ablation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _jobs(output: Path, datasets: list[str], seeds: list[int], ablations: list[str], device: str) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for ablation in ablations:
        for dataset in datasets:
            for seed in seeds:
                index += 1
                run = output / dataset / ablation / f"seed{seed}"
                jobs.append(Job(index, dataset, seed, ablation, device, str(run)))
    return jobs


def _command(job: Job) -> list[str]:
    run = Path(job.run_dir)
    return [
        sys.executable, "-m", "src.main",
        f"dataset={job.dataset}", "task=nc", f"model={MODEL}",
        f"seed={job.seed}", "num_runs=1", f"device={job.device}",
        f"ablation={job.ablation}", "task.training_mode=full_graph", "task.evaluate_test=true",
        f"task.protocol_version={PROTOCOL}", f"task.save_ckpt_path={run / 'best.pt'}",
        f"hydra.run.dir={run}",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _provenance(commit: str, datasets: list[str], seeds: list[int], ablations: list[str], dry_run: bool) -> dict[str, Any]:
    return {
        "schema": "ssi_mag_final_nc_ablation_provenance_v1",
        "git_commit": commit,
        "model": MODEL,
        "model_source": str(MODEL_SOURCE.relative_to(ROOT)),
        "model_sha256": _sha256(MODEL_SOURCE),
        "config_source": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "task": "nc",
        "protocol": PROTOCOL,
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "ablations": list(ablations),
        "planned_jobs": len(datasets) * len(seeds) * len(ablations),
        "lp_jobs": 0,
        "selection_metric": "val_acc",
        "test_used_for_selection": False,
        "dry_run": bool(dry_run),
        "training_started": False if dry_run else None,
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
    if marker.get("status") != "complete" or marker.get("task") != "nc" or marker.get("dataset") != job.dataset or marker.get("ablation") != job.ablation or int(marker.get("seed", -1)) != job.seed:
        return False, "completion identity/status mismatch"
    if metrics.get("task") != "nc" or metrics.get("model") != MODEL or metrics.get("checkpoint_selection") != "best_val_accuracy":
        return False, "metrics identity/selection mismatch"
    if resolved.get("model", {}).get("name") != MODEL or resolved.get("ablation") != job.ablation:
        return False, "resolved model/ablation mismatch"
    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        if key not in results or not math.isfinite(float(results[key]["mean"])):
            return False, f"invalid {key}"
    return True, "complete"


def main() -> int:
    args = _args()
    output = _root(args.output_root)
    jobs = _jobs(output, list(args.datasets), list(args.seeds), list(args.ablations), args.device)
    commit = _commit()
    provenance = _provenance(commit, list(args.datasets), list(args.seeds), list(args.ablations), bool(args.dry_run))
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "provenance.lock.json"
    if lock.is_file():
        stored = json.loads(lock.read_text())
        if stored != provenance:
            # A previous P1.9 dry-run may have used the canonical Full config
            # before the dedicated ablation config was added. It is safe to
            # refresh that lock only when it is still a dry-run and no run has
            # completed; formal/resume locks remain immutable.
            completed = any(_complete(job)[0] for job in jobs)
            if not (args.dry_run and stored.get("dry_run") and not completed):
                raise RuntimeError("provenance lock mismatch; refusing resume")
            _dump(lock, provenance)
            lock_status = "refreshed_dry_run"
        else:
            lock_status = "validated_existing"
    else:
        _dump(lock, provenance)
        lock_status = "created"
    manifest = {
        "launcher": "run_ssi_mag_final_nc_ablation.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "task": "nc", "model": MODEL, "protocol": PROTOCOL,
        "datasets": list(args.datasets), "seeds": list(args.seeds),
        "ablations": list(args.ablations), "device": args.device,
        "dry_run": bool(args.dry_run), "lp_jobs": 0,
        "selection_metric": "val_acc", "checkpoint_selection": "best_val_accuracy",
        "test_used_for_selection": False, "provenance_lock": str(lock),
        "provenance_lock_status": lock_status, "formal_provenance": provenance,
        "jobs": [{**asdict(job), "command": shlex.join(_command(job)), "status": "planned" if args.dry_run else "pending"} for job in jobs],
    }
    _dump(output / "manifest.json", manifest)
    if args.dry_run:
        print(f"Dry-run planned {len(jobs)} NC jobs; LP jobs: 0; training started: false")
        return 0
    failures: list[str] = []
    for job in jobs:
        complete, reason = _complete(job)
        if complete:
            print(f"[skip] {job.dataset}/{job.ablation}/seed{job.seed}: {reason}", flush=True)
            continue
        print(f"[start] [{job.index}/{len(jobs)}] {shlex.join(_command(job))}", flush=True)
        completed = subprocess.run(_command(job), cwd=ROOT, check=False)
        if completed.returncode:
            failures.append(f"{job.dataset}/{job.ablation}/seed{job.seed}: returncode={completed.returncode}")
            continue
        ok, why = _complete(job)
        if not ok:
            failures.append(f"{job.dataset}/{job.ablation}/seed{job.seed}: {why}")
    manifest["jobs"] = [{**asdict(job), "command": shlex.join(_command(job)), "status": "complete" if _complete(job)[0] else "failed"} for job in jobs]
    manifest["completed_jobs"] = sum(item["status"] == "complete" for item in manifest["jobs"])
    manifest["failed_jobs"] = len(failures)
    manifest["failures"] = failures
    _dump(output / "manifest.json", manifest)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
