#!/usr/bin/env python3
"""Resume-safe P1.7c NC launcher for controls B and AB.

This launcher only creates full-graph NC jobs.  ``r2_only`` is B and
``r1_r2`` is AB.  It has no LP or paper-ablation branch.
"""

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
CONTROLS = ["r2_only", "r1_r2"]
VARIANT_NAMES = {"r2_only": "B", "r1_r2": "AB"}
MODEL = "ssi_mag_v31_controls"
CONFIG = ROOT / "configs/model/ssi_mag_v31_controls.yaml"
MODEL_SOURCE = ROOT / "src/models/ssi_mag_v31_controls.py"


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    seed: int
    control: str
    variant: str
    device: str
    run_dir: str


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--controls", nargs="+", choices=CONTROLS, default=CONTROLS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-root", default="outputs/ssi_mag_v31_p17c_controls")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _jobs(output: Path, datasets: list[str], seeds: list[int], controls: list[str], device: str) -> list[Job]:
    result = []
    index = 0
    for dataset in datasets:
        for control in controls:
            for seed in seeds:
                index += 1
                result.append(Job(index, dataset, seed, control, VARIANT_NAMES[control], device, str(output / dataset / VARIANT_NAMES[control] / f"seed{seed}")))
    return result


def _command(job: Job) -> list[str]:
    run = Path(job.run_dir)
    return [
        sys.executable, "-m", "src.main",
        f"dataset={job.dataset}", "task=nc", f"model={MODEL}",
        f"model.control={job.control}", f"seed={job.seed}", "num_runs=1",
        f"device={job.device}", "ablation=full", "task.training_mode=full_graph",
        "task.evaluate_test=true", "task.protocol_version=unified_full_graph_nc_v1",
        f"task.save_ckpt_path={run / 'best.pt'}", f"hydra.run.dir={run}",
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


def _config_payload() -> dict[str, Any]:
    raw = CONFIG.read_text(encoding="utf-8")
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(CONFIG)
        wrapped = OmegaConf.create({"model": cfg})
        resolved = OmegaConf.to_container(wrapped.model, resolve=True)
    except Exception:
        resolved = None
    payload = {"source": str(CONFIG.relative_to(ROOT)), "sha256": _sha256(CONFIG), "resolved": resolved}
    if resolved is None:
        payload["raw_yaml"] = raw
    return payload


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _complete(job: Job) -> tuple[bool, str]:
    run = Path(job.run_dir)
    required = [run / name for name in ("best.pt", "metrics.json", "results.json", "resolved_config.json", "complete.marker")]
    missing = [p.name for p in required if not p.is_file()]
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
    if metrics.get("task") != "nc" or metrics.get("model") != MODEL or metrics.get("dataset") != job.dataset or int(metrics.get("seed", -1)) != job.seed or metrics.get("ablation") != "full" or metrics.get("checkpoint_selection") != "best_val_accuracy":
        return False, "metrics identity/selection mismatch"
    if resolved.get("model", {}).get("control") != job.control:
        return False, "control mismatch"
    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        if key not in results or not math.isfinite(float(results[key]["mean"])):
            return False, f"invalid {key}"
    return True, "complete"


def _provenance(commit: str, datasets: list[str], seeds: list[int], controls: list[str]) -> dict[str, Any]:
    return {
        "schema": "ssi_mag_v31_p17c_controls_provenance_v1",
        "git_commit": commit,
        "model": MODEL,
        "model_source": str(MODEL_SOURCE.relative_to(ROOT)),
        "model_sha256": _sha256(MODEL_SOURCE),
        "config_source": str(CONFIG.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG),
        "task": "nc",
        "protocol": "unified_full_graph_nc_v1",
        "controls": list(controls),
        "variants": {control: VARIANT_NAMES[control] for control in controls},
        "datasets": list(datasets),
        "seeds": [int(seed) for seed in seeds],
        "lp_jobs": 0,
    }


def _lock(output: Path, jobs: list[Job], provenance: dict[str, Any], dry_run: bool) -> tuple[Path, str]:
    path = output / "provenance.lock.json"
    if dry_run:
        return path, "not_written_dry_run"
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored != provenance:
            raise RuntimeError("provenance lock mismatch; refusing resume")
        return path, "validated_existing"
    completed = [f"{j.dataset}/{j.variant}/seed{j.seed}" for j in jobs if _complete(j)[0]]
    if completed:
        raise RuntimeError("completed artifacts exist without provenance lock: " + ", ".join(completed))
    _dump(path, provenance)
    return path, "created"


def _manifest(args: argparse.Namespace, jobs: list[Job], commit: str, config: dict[str, Any], lock: Path, status: str, provenance: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "launcher": "run_ssi_mag_v31_p17c_controls.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "task": "nc", "model": MODEL, "protocol": "unified_full_graph_nc_v1",
        "datasets": list(args.datasets), "seeds": list(args.seeds), "controls": list(args.controls),
        "variants": {control: VARIANT_NAMES[control] for control in args.controls},
        "device": args.device, "dry_run": bool(args.dry_run), "lp_jobs": 0,
        "selection_metric": "val_acc", "checkpoint_selection": "best_val_accuracy", "test_used_for_selection": False,
        "resolved_model_config": config, "provenance_lock": str(lock), "provenance_lock_status": status,
        "formal_provenance": provenance,
        "jobs": [{**asdict(job), "command": shlex.join(_command(job)), "status": "planned" if args.dry_run else "pending"} for job in jobs],
    }


def _summary(jobs: list[Job]) -> dict[str, Any]:
    keys = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for job in jobs:
        if not _complete(job)[0]:
            continue
        results = json.loads((Path(job.run_dir) / "results.json").read_text())
        metrics = json.loads((Path(job.run_dir) / "metrics.json").read_text())
        group = grouped.setdefault((job.dataset, job.variant), {key: [] for key in keys} | {"best_epoch": []})
        for key in keys:
            group[key].append(float(results[key]["mean"]))
        if metrics.get("best_epoch") is not None:
            group["best_epoch"].append(float(metrics["best_epoch"]))
    output = {}
    for (dataset, variant), values in grouped.items():
        entry = {"n": len(values["val_acc"])}
        for key, observations in values.items():
            if observations:
                entry[key] = {"mean": statistics.fmean(observations), "population_std": statistics.pstdev(observations), "values": observations}
        output[f"{dataset}/{variant}"] = entry
    return output


def main() -> int:
    args = _args()
    output = _root(args.output_root)
    jobs = _jobs(output, list(args.datasets), list(args.seeds), list(args.controls), args.device)
    output.mkdir(parents=True, exist_ok=True)
    commit = _commit()
    config = _config_payload()
    provenance = _provenance(commit, list(args.datasets), list(args.seeds), list(args.controls))
    lock, lock_status = _lock(output, jobs, provenance, bool(args.dry_run))
    manifest = _manifest(args, jobs, commit, config, lock, lock_status, None if args.dry_run else provenance)
    manifest_path = output / "manifest.json"
    _dump(manifest_path, manifest)
    for position, job in enumerate(jobs):
        run = Path(job.run_dir)
        run.mkdir(parents=True, exist_ok=True)
        complete, reason = _complete(job)
        if complete:
            status = "skipped_complete"
            print(f"[skip] [{job.index}/{len(jobs)}] {job.dataset}/{job.variant}/seed{job.seed}", flush=True)
        elif args.dry_run:
            status = "planned"
            print(f"[dry-run] [{job.index}/{len(jobs)}] {shlex.join(_command(job))}", flush=True)
        else:
            manifest["jobs"][position]["status"] = "running"
            _dump(manifest_path, manifest)
            print(f"[start] [{job.index}/{len(jobs)}] {shlex.join(_command(job))}", flush=True)
            completed = subprocess.run(_command(job), cwd=ROOT, check=False)
            if completed.returncode:
                status = f"failed:{completed.returncode}"
                print(f"[failed] {job.dataset}/{job.variant}/seed{job.seed} ({reason})", flush=True)
            else:
                ok, why = _complete(job)
                status = "complete" if ok else f"incomplete:{why}"
                print(f"[{status}] {job.dataset}/{job.variant}/seed{job.seed}", flush=True)
        manifest["jobs"][position]["status"] = status
        _dump(manifest_path, manifest)
    summary = _summary(jobs)
    _dump(output / "summary.json", summary)
    if args.dry_run:
        print(f"Dry-run planned {len(jobs)} NC jobs; LP jobs: 0; training started: false", flush=True)
    else:
        for group, values in summary.items():
            parts = [f"{key} {values[key]['mean']:.6f} ± {values[key]['population_std']:.6f}" for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1") if key in values]
            print(f"[summary] {group}: " + " | ".join(parts), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    return int(not args.dry_run and any(str(item.get("status", "")).startswith(("failed:", "incomplete:")) for item in manifest["jobs"]))


if __name__ == "__main__":
    raise SystemExit(main())
