#!/usr/bin/env python3
"""Resume-safe NC launcher for the future SSI-MAG-V3.1 Full benchmark.

The launcher supports execution for the next stage, but P1.7a.1 invokes it
with --dry-run only. It has no LP branch and only accepts the frozen full
variant.
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
DEFAULT_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
DEFAULT_SEEDS = [42, 43, 44]
VALID_ABLATIONS = ["full"]
MODEL_CONFIG_PATH = ROOT / "configs/model/ssi_mag_v31.yaml"


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    seed: int
    ablation: str
    device: str
    run_dir: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ablations", nargs="+", choices=VALID_ABLATIONS, default=VALID_ABLATIONS)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "outputs" / "ssi_mag_v31_p17b_full"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _output_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _build_jobs(
    output_root: Path,
    datasets: list[str],
    seeds: list[int],
    ablations: list[str],
    device: str,
) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for dataset in datasets:
        for ablation in ablations:
            for seed in seeds:
                index += 1
                jobs.append(
                    Job(
                        index,
                        dataset,
                        int(seed),
                        ablation,
                        device,
                        str(output_root / dataset / ablation / f"seed{seed}"),
                    )
                )
    return jobs


def _command(job: Job) -> list[str]:
    run_dir = Path(job.run_dir)
    return [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={job.dataset}",
        "task=nc",
        "model=ssi_mag_v31",
        f"seed={job.seed}",
        "num_runs=1",
        f"device={job.device}",
        "ablation=full",
        "task.training_mode=full_graph",
        "task.evaluate_test=true",
        "task.protocol_version=unified_full_graph_nc_v1",
        f"task.save_ckpt_path={run_dir / 'best.pt'}",
        f"hydra.run.dir={run_dir}",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolved_model_config() -> dict[str, Any]:
    raw = MODEL_CONFIG_PATH.read_text(encoding="utf-8")
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.load(MODEL_CONFIG_PATH)
        # The standalone model file contains ${model.max_order}; resolve it
        # under the same wrapper used by Hydra's root config.
        wrapped = OmegaConf.create({"model": config})
        return {
            "source": str(MODEL_CONFIG_PATH.relative_to(ROOT)),
            "sha256": _sha256(MODEL_CONFIG_PATH),
            "resolved": OmegaConf.to_container(wrapped.model, resolve=True),
        }
    except Exception:
        # The launcher remains usable from a system Python for dry-run. The
        # fallback records the exact file and its hash instead of fabricating a
        # partially resolved configuration.
        return {
            "source": str(MODEL_CONFIG_PATH.relative_to(ROOT)),
            "sha256": _sha256(MODEL_CONFIG_PATH),
            "resolved": None,
            "raw_yaml": raw,
            "resolution_note": "Install/use the project environment to materialize OmegaConf interpolation.",
        }


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _required_outputs(job: Job) -> tuple[Path, ...]:
    run_dir = Path(job.run_dir)
    return tuple(
        run_dir / name
        for name in ("metrics.json", "results.json", "resolved_config.json", "complete.marker", "best.pt")
    )


def _is_complete(job: Job) -> tuple[bool, str]:
    missing = [str(path.name) for path in _required_outputs(job) if not path.is_file()]
    if missing:
        return False, "missing " + ", ".join(missing)
    run_dir = Path(job.run_dir)
    try:
        marker = json.loads((run_dir / "complete.marker").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid completion metadata: {exc}"
    if (
        marker.get("status") != "complete"
        or marker.get("task") != "nc"
        or marker.get("dataset") != job.dataset
        or int(marker.get("seed", -1)) != job.seed
        or marker.get("ablation") != "full"
    ):
        return False, "completion marker identity/status mismatch"
    if (
        metrics.get("task") != "nc"
        or metrics.get("dataset") != job.dataset
        or metrics.get("model") != "ssi_mag_v31"
        or int(metrics.get("seed", -1)) != job.seed
        or metrics.get("ablation") != "full"
        or metrics.get("checkpoint_selection") != "best_val_accuracy"
    ):
        return False, "metrics identity/selection mismatch"
    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        if key not in results or not isinstance(results[key], dict):
            return False, f"results missing {key}"
        if not math.isfinite(float(results[key]["mean"])):
            return False, f"non-finite {key}"
    return True, "complete"


def _manifest_payload(
    args: argparse.Namespace,
    jobs: list[Job],
    commit: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "launcher": "run_ssi_mag_v31_nc.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "task": "nc",
        "model": "ssi_mag_v31",
        "datasets": list(args.datasets),
        "seeds": [int(seed) for seed in args.seeds],
        "ablations": list(args.ablations),
        "device": args.device,
        "dry_run": bool(args.dry_run),
        "formal_protocol": "unified_full_graph_nc_v1",
        "selection_metric": "val_acc",
        "checkpoint_selection": "best_val_accuracy",
        "test_used_for_selection": False,
        "lp_jobs": 0,
        "resolved_model_config": model_config,
        "jobs": [
            {
                **asdict(job),
                "command": shlex.join(_command(job)),
                "status": "planned" if args.dry_run else "pending",
            }
            for job in jobs
        ],
    }


def _result_value(job: Job, key: str) -> float | None:
    try:
        results = json.loads((Path(job.run_dir) / "results.json").read_text(encoding="utf-8"))
        value = float(results[key]["mean"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    return value if math.isfinite(value) else None


def _best_epoch(job: Job) -> int | None:
    try:
        metrics = json.loads((Path(job.run_dir) / "metrics.json").read_text(encoding="utf-8"))
        value = metrics.get("best_epoch")
        return int(value) if value is not None else None
    except (TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _summarize(jobs: list[Job]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    keys = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    for job in jobs:
        complete, _ = _is_complete(job)
        if not complete:
            continue
        group = grouped.setdefault((job.dataset, job.ablation), {key: [] for key in keys} | {"best_epoch": []})
        for key in keys:
            value = _result_value(job, key)
            if value is not None:
                group[key].append(value)
        epoch = _best_epoch(job)
        if epoch is not None:
            group["best_epoch"].append(float(epoch))
    summary: dict[str, Any] = {}
    for (dataset, ablation), values in grouped.items():
        entry: dict[str, Any] = {"n": len(values["val_acc"])}
        for key, observations in values.items():
            if not observations:
                continue
            entry[key] = {
                "mean": statistics.fmean(observations),
                "population_std": statistics.pstdev(observations),
                "values": observations,
            }
        summary[f"{dataset}/{ablation}"] = entry
    return summary


def main() -> int:
    args = _parse_args()
    unknown = sorted(set(args.datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise SystemExit(f"Unsupported formal NC dataset(s): {', '.join(unknown)}")
    if not args.seeds or not args.ablations:
        raise SystemExit("--seeds and --ablations must not be empty")
    if any(ablation != "full" for ablation in args.ablations):
        raise SystemExit("V3.1 launcher only supports the frozen full variant")

    output_root = _output_root(args.output_root)
    jobs = _build_jobs(output_root, args.datasets, args.seeds, args.ablations, args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    model_config = _resolved_model_config()
    manifest_path = output_root / "manifest.json"
    manifest = _manifest_payload(args, jobs, commit, model_config)
    _dump(manifest_path, manifest)
    training_started = False

    for position, job in enumerate(jobs):
        command = _command(job)
        run_dir = Path(job.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        complete, reason = _is_complete(job)
        if complete:
            status = "skipped_complete"
            print(f"[skip] [{job.index}/{len(jobs)}] {job.dataset}/full/seed{job.seed}", flush=True)
        elif args.dry_run:
            status = "planned"
            print(f"[dry-run] [{job.index}/{len(jobs)}] {shlex.join(command)}", flush=True)
        else:
            training_started = True
            status = "running"
            manifest["jobs"][position]["status"] = status
            _dump(manifest_path, manifest)
            print(f"[start] [{job.index}/{len(jobs)}] {shlex.join(command)}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                status = f"failed:{completed.returncode}"
                print(f"[failed] {job.dataset}/full/seed{job.seed} ({reason})", flush=True)
            else:
                now_complete, now_reason = _is_complete(job)
                status = "complete" if now_complete else f"incomplete:{now_reason}"
                print(f"[{status}] {job.dataset}/full/seed{job.seed}", flush=True)
        manifest["jobs"][position]["status"] = status
        _dump(manifest_path, manifest)

    summary = _summarize(jobs)
    summary_path = output_root / "summary.json"
    _dump(summary_path, summary)
    if args.dry_run:
        expected = len(args.datasets) * len(args.seeds) * len(args.ablations)
        print(f"Dry-run planned {expected} NC jobs; LP jobs: 0; training started: false", flush=True)
    else:
        for group, values in summary.items():
            parts = []
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                if key in values:
                    parts.append(f"{key} {values[key]['mean']:.6f} ± {values[key]['population_std']:.6f}")
            if "best_epoch" in values:
                parts.append(f"best_epoch {values['best_epoch']['mean']:.3f} ± {values['best_epoch']['population_std']:.3f}")
            print(f"[summary] {group}: " + " | ".join(parts), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    failed_or_incomplete = any(
        str(job_entry.get("status", "")).startswith(("failed:", "incomplete:"))
        for job_entry in manifest["jobs"]
    )
    return 1 if (not args.dry_run and failed_or_incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
