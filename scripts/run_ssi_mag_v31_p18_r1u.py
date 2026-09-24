#!/usr/bin/env python3
"""Resume-safe P1.8 Direct Semantic Relation Utilization NC pilot launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
MODEL = "ssi_mag_v31_r1u"
VARIANT = "U"
CONFIG = ROOT / "configs/model/ssi_mag_v31_r1u.yaml"
MODEL_SOURCE = ROOT / "src/models/ssi_mag_v31_r1u.py"
PROTOCOL = "unified_full_graph_nc_v1"


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    seed: int
    variant: str
    device: str
    run_dir: str


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/ssi_mag_v31_p18_r1u")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _jobs(output: Path, datasets: list[str], seeds: list[int], device: str) -> list[Job]:
    jobs = []
    index = 0
    for dataset in datasets:
        for seed in seeds:
            index += 1
            run = output / dataset / VARIANT / f"seed{seed}"
            jobs.append(Job(index, dataset, seed, VARIANT, device, str(run)))
    return jobs


def _command(job: Job) -> list[str]:
    run = Path(job.run_dir)
    return [
        sys.executable, "-m", "src.main",
        f"dataset={job.dataset}", "task=nc", f"model={MODEL}",
        f"seed={job.seed}", "num_runs=1", f"device={job.device}",
        "ablation=full", "task.training_mode=full_graph", "task.evaluate_test=true",
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


def _resolved_config() -> dict[str, Any]:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(CONFIG)
    wrapped = OmegaConf.create({"model": cfg})
    return OmegaConf.to_container(wrapped.model, resolve=True)


def _provenance(commit: str, datasets: list[str], seeds: list[int]) -> dict[str, Any]:
    return {
        "schema": "ssi_mag_v31_p18_r1u_provenance_v1",
        "git_commit": commit,
        "model": MODEL,
        "variant": VARIANT,
        "model_source": str(MODEL_SOURCE.relative_to(ROOT)),
        "model_sha256": _sha256(MODEL_SOURCE),
        "config_source": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "task": "nc",
        "protocol": PROTOCOL,
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "lp_jobs": 0,
        "selection_metric": "val_acc",
        "test_used_for_selection": False,
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
    if metrics.get("task") != "nc" or metrics.get("model") != MODEL or metrics.get("dataset") != job.dataset or int(metrics.get("seed", -1)) != job.seed or metrics.get("checkpoint_selection") != "best_val_accuracy":
        return False, "metrics identity/selection mismatch"
    if resolved.get("model", {}).get("name") != MODEL or resolved.get("ablation") != "full":
        return False, "resolved model/ablation mismatch"
    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        if key not in results or not math.isfinite(float(results[key]["mean"])):
            return False, f"invalid {key}"
    return True, "complete"


def _lock(output: Path, jobs: list[Job], provenance: dict[str, Any], dry_run: bool) -> tuple[Path, str]:
    path = output / "provenance.lock.json"
    if dry_run:
        return path, "not_written_dry_run"
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored != provenance:
            raise RuntimeError("provenance lock mismatch; refusing resume")
        return path, "validated_existing"
    completed = [f"{job.dataset}/{job.variant}/seed{job.seed}" for job in jobs if _complete(job)[0]]
    if completed:
        raise RuntimeError("completed artifacts exist without provenance lock: " + ", ".join(completed))
    _dump(path, provenance)
    return path, "created"


def _manifest(args: argparse.Namespace, jobs: list[Job], commit: str, lock: Path, status: str, provenance: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "launcher": "run_ssi_mag_v31_p18_r1u.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "task": "nc", "model": MODEL, "variant": VARIANT, "protocol": PROTOCOL,
        "datasets": list(args.datasets), "seeds": list(args.seeds), "device": args.device,
        "dry_run": bool(args.dry_run), "lp_jobs": 0,
        "selection_metric": "val_acc", "checkpoint_selection": "best_val_accuracy", "test_used_for_selection": False,
        "resolved_model_config": _resolved_config(),
        "provenance_lock": str(lock), "provenance_lock_status": status, "formal_provenance": provenance,
        "jobs": [{**asdict(job), "command": shlex.join(_command(job)), "status": "planned" if args.dry_run else "pending"} for job in jobs],
    }


def _summary(output: Path, jobs: list[Job]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for job in jobs:
        run = Path(job.run_dir)
        if not (run / "results.json").is_file():
            continue
        results = json.loads((run / "results.json").read_text())
        values = grouped.setdefault(job.dataset, {key: [] for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")})
        for key in values:
            values[key].append(float(results[key]["mean"]))
    rows = []
    for dataset, values in grouped.items():
        row = {"dataset": dataset, "variant": VARIANT, "seed_count": len(values["val_acc"])}
        for key, items in values.items():
            row[f"{key}_mean"] = statistics.mean(items)
            row[f"{key}_population_std"] = statistics.pstdev(items)
        rows.append(row)
    payload = {"model": MODEL, "variant": VARIANT, "task": "nc", "protocol": PROTOCOL, "lp_jobs": 0, "rows": rows}
    _dump(output / "summary.json", payload)
    return payload


def main() -> int:
    args = _args()
    output = _root(args.output_root)
    jobs = _jobs(output, list(args.datasets), list(args.seeds), args.device)
    commit = _commit()
    provenance = _provenance(commit, list(args.datasets), list(args.seeds))
    lock, lock_status = _lock(output, jobs, provenance, bool(args.dry_run))
    manifest = _manifest(args, jobs, commit, lock, lock_status, None if args.dry_run else provenance)
    output.mkdir(parents=True, exist_ok=True)
    _dump(output / "manifest.json", manifest)
    if args.dry_run:
        print(f"Dry-run planned {len(jobs)} NC jobs; LP jobs: 0; training started: false")
        return 0
    failures = []
    for job in jobs:
        complete, reason = _complete(job)
        if complete:
            print(f"[skip] {job.dataset}/{VARIANT}/seed{job.seed}: {reason}", flush=True)
            continue
        print(f"[start] [{job.index}/{len(jobs)}] {shlex.join(_command(job))}", flush=True)
        completed = subprocess.run(_command(job), cwd=ROOT, check=False)
        if completed.returncode:
            failures.append(f"{job.dataset}/{VARIANT}/seed{job.seed}: returncode={completed.returncode}")
            continue
        ok, why = _complete(job)
        if not ok:
            failures.append(f"{job.dataset}/{VARIANT}/seed{job.seed}: {why}")
        else:
            print(f"[complete] {job.dataset}/{VARIANT}/seed{job.seed}", flush=True)
    manifest["jobs"] = [{**asdict(job), "command": shlex.join(_command(job)), "status": "complete" if _complete(job)[0] else "failed"} for job in jobs]
    manifest["completed_jobs"] = sum(item["status"] == "complete" for item in manifest["jobs"])
    manifest["failed_jobs"] = len(failures)
    manifest["failures"] = failures
    _dump(output / "manifest.json", manifest)
    summary = _summary(output, jobs)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
