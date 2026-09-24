#!/usr/bin/env python3
"""Dry-run-only future NC plan for SSI-MAG-V3.1.

P1.7a deliberately refuses execution. It materializes the future formal plan
(5 datasets x 3 seeds x full) and records zero LP jobs without starting a
training process.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
DEFAULT_SEEDS = [42, 43, 44]
VALID_ABLATIONS = ["full"]


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
        default=str(ROOT / "outputs" / "ssi_mag_v31_p17a_nc_plan"),
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


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if not args.dry_run:
        raise SystemExit("P1.7a launcher is dry-run-only; pass --dry-run and no training will be started")
    unknown = sorted(set(args.datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise SystemExit(f"Unsupported formal NC dataset(s): {', '.join(unknown)}")
    if not args.seeds or not args.ablations:
        raise SystemExit("--seeds and --ablations must not be empty")
    if any(ablation != "full" for ablation in args.ablations):
        raise SystemExit("P1.7a only plans the frozen V3.1 full variant")

    output_root = _output_root(args.output_root)
    jobs = _build_jobs(output_root, args.datasets, args.seeds, args.ablations, args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "launcher": "run_ssi_mag_v31_nc.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "nc",
        "model": "ssi_mag_v31",
        "datasets": list(args.datasets),
        "seeds": [int(seed) for seed in args.seeds],
        "ablations": list(args.ablations),
        "device": args.device,
        "dry_run": True,
        "execution_allowed": False,
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
    for job in jobs:
        Path(job.run_dir).mkdir(parents=True, exist_ok=True)
        print(f"[dry-run] [{job.index}/{len(jobs)}] {shlex.join(_command(job))}", flush=True)
    _dump(output_root / "manifest.json", manifest)
    summary = {
        "model": "ssi_mag_v31",
        "task": "nc",
        "variant": "full",
        "planned_jobs": len(jobs),
        "expected_jobs": len(args.datasets) * len(args.seeds) * len(args.ablations),
        "lp_jobs": 0,
        "training_started": False,
        "datasets": list(args.datasets),
        "seeds": [int(seed) for seed in args.seeds],
        "ablations": list(args.ablations),
    }
    _dump(output_root / "summary.json", summary)
    print(f"Dry-run planned {len(jobs)} NC jobs; LP jobs: 0; training started: false", flush=True)
    print(f"Manifest: {output_root / 'manifest.json'}", flush=True)
    print(f"Summary: {output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
