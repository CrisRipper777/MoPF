#!/usr/bin/env python3
"""NC-only launcher for SSI-MAG-V3.

The launcher creates one deterministic directory per dataset/seed/ablation,
skips a unit only when its successful completion marker and result files are
present, stores a manifest, and summarizes validation/test Accuracy and
Macro-F1 with population standard deviation.  The default plan is exactly
the five formal NC datasets and seeds 42, 43, 44.

Use ``--dry-run`` to inspect the complete plan without invoking training.
"""

from __future__ import annotations

import argparse
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
DEFAULT_ABLATIONS = ["full"]
VALID_ABLATIONS = [
    "full",
    "raw_relation",
    "fixed_reference",
    "terminal_context",
    "no_context_change",
    "no_cross_hop_interaction",
    "uniform_context",
]


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
    parser.add_argument("--ablations", nargs="+", choices=VALID_ABLATIONS, default=DEFAULT_ABLATIONS)
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "ssi_mag_v3_nc"))
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
                run_dir = output_root / dataset / ablation / f"seed{seed}"
                jobs.append(
                    Job(index, dataset, int(seed), ablation, device, str(run_dir))
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
        "model=ssi_mag_v3",
        f"seed={job.seed}",
        "num_runs=1",
        f"device={job.device}",
        f"ablation={job.ablation}",
        "task.training_mode=full_graph",
        "task.evaluate_test=true",
        "task.protocol_version=unified_full_graph_nc_v1",
        f"task.save_ckpt_path={run_dir / 'best.pt'}",
        f"hydra.run.dir={run_dir}",
    ]


def _required_outputs(job: Job) -> tuple[Path, ...]:
    run_dir = Path(job.run_dir)
    return tuple(
        run_dir / name
        for name in (
            "metrics.json",
            "results.json",
            "resolved_config.json",
            "complete.marker",
            "best.pt",
        )
    )


def _is_complete(job: Job) -> tuple[bool, str]:
    missing = [str(path.name) for path in _required_outputs(job) if not path.is_file()]
    if missing:
        return False, "missing " + ", ".join(missing)
    try:
        marker = json.loads((Path(job.run_dir) / "complete.marker").read_text(encoding="utf-8"))
        metrics = json.loads((Path(job.run_dir) / "metrics.json").read_text(encoding="utf-8"))
        results = json.loads((Path(job.run_dir) / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid completion metadata: {exc}"
    if marker.get("status") != "complete":
        return False, "completion marker is not complete"
    if marker.get("task") != "nc" or marker.get("seed") != job.seed:
        return False, "completion identity mismatch"
    if metrics.get("task") != "nc" or metrics.get("dataset") != job.dataset:
        return False, "metrics identity mismatch"
    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        if key not in results or not isinstance(results[key], dict):
            return False, f"results missing {key}"
        if not math.isfinite(float(results[key]["mean"])):
            return False, f"non-finite {key}"
    return True, "complete"


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _manifest_payload(args: argparse.Namespace, jobs: list[Job]) -> dict[str, Any]:
    return {
        "launcher": "run_ssi_mag_v3_nc.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "nc",
        "model": "ssi_mag_v3",
        "datasets": list(args.datasets),
        "seeds": [int(seed) for seed in args.seeds],
        "ablations": list(args.ablations),
        "device": args.device,
        "dry_run": bool(args.dry_run),
        "formal_protocol": "unified_full_graph_nc_v1",
        "selection_metric": "val_acc",
        "checkpoint_selection": "best_val_accuracy",
        "lp_jobs": 0,
        "jobs": [
            {
                **asdict(job),
                "command": shlex.join(_command(job)),
                "status": "planned",
            }
            for job in jobs
        ],
    }


def _result_value(job: Job, key: str) -> float | None:
    try:
        results = json.loads((Path(job.run_dir) / "results.json").read_text(encoding="utf-8"))
        value = results[key]["mean"]
        value = float(value)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    return value if math.isfinite(value) else None


def _summarize(jobs: list[Job]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    keys = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    for job in jobs:
        complete, _ = _is_complete(job)
        if not complete:
            continue
        group = grouped.setdefault((job.dataset, job.ablation), {key: [] for key in keys})
        for key in keys:
            value = _result_value(job, key)
            if value is not None:
                group[key].append(value)
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
    unknown_datasets = sorted(set(args.datasets) - set(DEFAULT_DATASETS))
    if unknown_datasets:
        raise SystemExit(f"Unsupported formal NC dataset(s): {', '.join(unknown_datasets)}")
    if not args.seeds or not args.ablations:
        raise SystemExit("--seeds and --ablations must not be empty")
    output_root = _output_root(args.output_root)
    jobs = _build_jobs(output_root, args.datasets, args.seeds, args.ablations, args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = _manifest_payload(args, jobs)
    _json_dump(manifest_path, manifest)

    for position, job in enumerate(jobs):
        command = _command(job)
        run_dir = Path(job.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        complete, reason = _is_complete(job)
        if complete:
            status = "skipped_complete"
            print(f"[skip] [{job.index}/{len(jobs)}] {job.dataset}/{job.ablation}/seed{job.seed}", flush=True)
        elif args.dry_run:
            status = "planned"
            print(f"[dry-run] [{job.index}/{len(jobs)}] {shlex.join(command)}", flush=True)
        else:
            print(f"[start] [{job.index}/{len(jobs)}] {shlex.join(command)}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                status = f"failed:{completed.returncode}"
                print(f"[failed] {job.dataset}/{job.ablation}/seed{job.seed} ({reason})", flush=True)
            else:
                now_complete, now_reason = _is_complete(job)
                status = "complete" if now_complete else f"incomplete:{now_reason}"
                print(f"[{status}] {job.dataset}/{job.ablation}/seed{job.seed}", flush=True)
        manifest["jobs"][position]["status"] = status
        _json_dump(manifest_path, manifest)

    summary = _summarize(jobs)
    summary_path = output_root / "summary.json"
    _json_dump(summary_path, summary)
    if args.dry_run:
        expected = len(args.datasets) * len(args.seeds) * len(args.ablations)
        print(f"Dry-run planned {expected} NC jobs; LP jobs: 0", flush=True)
    else:
        for group, values in summary.items():
            parts = []
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                if key in values:
                    parts.append(
                        f"{key} {values[key]['mean']:.6f} ± {values[key]['population_std']:.6f}"
                    )
            print(f"[summary] {group}: " + " | ".join(parts), flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

