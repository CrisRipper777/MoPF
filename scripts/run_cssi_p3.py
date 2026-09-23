#!/usr/bin/env python3
"""Run the unified P3 conditional response adaptation family."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = (
    "mrc_plain",
    "self_lr",
    "same_scalar",
    "same_lr",
    "all_lr",
    "same_lr_mrc",
)

VARIANT_OVERRIDES = {
    "mrc_plain": {
        "model.use_mrc=true",
        "model.condition_mode=none",
        "model.adapter_type=none",
        "model.adapter_enabled=false",
    },
    "self_lr": {
        "model.use_mrc=false",
        "model.condition_mode=self",
        "model.adapter_type=lowrank",
        "model.adapter_enabled=true",
    },
    "same_scalar": {
        "model.use_mrc=false",
        "model.condition_mode=same",
        "model.adapter_type=scalar",
        "model.adapter_enabled=true",
    },
    "same_lr": {
        "model.use_mrc=false",
        "model.condition_mode=same",
        "model.adapter_type=lowrank",
        "model.adapter_enabled=true",
    },
    "all_lr": {
        "model.use_mrc=false",
        "model.condition_mode=all",
        "model.adapter_type=lowrank",
        "model.adapter_enabled=true",
    },
    "same_lr_mrc": {
        "model.use_mrc=true",
        "model.condition_mode=same",
        "model.adapter_type=lowrank",
        "model.adapter_enabled=true",
    },
}


def _job_command(
    project_root: Path,
    output_root: Path,
    variant: str,
    dataset: str,
    seed: int,
    gpu: str,
) -> tuple[list[str], Path, Path]:
    run_dir = output_root / "runs" / variant / dataset / f"seed_{seed}"
    checkpoint = output_root / "checkpoints" / variant / dataset / f"seed_{seed}.pt"
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=cssi_v0",
        *sorted(VARIANT_OVERRIDES[variant]),
        f"seed={seed}",
        "num_runs=1",
        f"device=cuda:{gpu}",
        "task.evaluate_test=false",
        "task.loss.aux_weight=0.0",
        f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={run_dir}",
    ]
    return [sys.executable, "-m", "src.main", *overrides], run_dir, checkpoint


def _run_job(job: tuple[list[str], Path, Path, str, str, int, Path, bool]) -> str:
    command, run_dir, checkpoint, variant, dataset, seed, project_root, force = job
    if not force and (run_dir / "complete.marker").is_file() and checkpoint.is_file():
        return f"SKIP {variant} {dataset} seed={seed} (complete)"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, env=env, check=True)
    if not checkpoint.is_file():
        raise RuntimeError(f"run completed without checkpoint: {checkpoint}")
    return f"DONE {variant} {dataset} seed={seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p3"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun jobs even when a prior completion marker/checkpoint exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    jobs = []
    index = 0
    for variant in args.variants:
        for dataset in args.datasets:
            for seed in args.seeds:
                command, run_dir, checkpoint = _job_command(
                    project_root,
                    output_root,
                    variant,
                    dataset,
                    seed,
                    str(args.gpus[index % len(args.gpus)]),
                )
                jobs.append(
                    (
                        command,
                        run_dir,
                        checkpoint,
                        variant,
                        dataset,
                        seed,
                        project_root,
                        bool(args.force),
                    )
                )
                index += 1
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.max_workers))
    ) as executor:
        futures = [executor.submit(_run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
