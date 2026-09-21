#!/usr/bin/env python3
"""Launch isolated IAMOC v1 runs under the frozen MoPF task protocols."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "iamoc_v1"
VARIANTS = {
    "V0": {"model": "mopf"},
    "V1": {
        "model": "mopf_iamoc",
        "model.hop_interaction_layers": "1",
        "model.relation_conditioning": "output",
    },
    "V2": {
        "model": "mopf_iamoc",
        "model.hop_interaction_layers": "1",
        "model.relation_conditioning": "none",
    },
    "V3": {
        "model": "mopf_iamoc",
        "model.hop_interaction_layers": "1",
        "model.relation_conditioning": "attention",
    },
    "V4": {
        "model": "mopf_iamoc",
        "model.hop_interaction_layers": "2",
        "model.relation_conditioning": "attention",
    },
}
DATASET_TASK = {
    "Movies": "nc",
    "Grocery": "nc",
    "sports-copurchase": "lp",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _make_run(dataset: str, variant: str, seed: int, device: str, smoke: bool) -> tuple[Path, list[str], dict[str, Any]]:
    task = DATASET_TASK[dataset]
    run_root = OUTPUT_ROOT / ("smoke" if smoke else "")
    run_dir = run_root / dataset / variant / f"seed_{seed}"
    checkpoint_path = run_dir / "best.pt"
    variant_cfg = VARIANTS[variant]
    model_name = str(variant_cfg["model"])
    overrides = [
        f"dataset={dataset}",
        f"task={task}",
        f"model={model_name}",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        "paths.output_root=outputs/iamoc_v1",
        f"task.save_ckpt_path={checkpoint_path}",
        f"hydra.run.dir={run_dir}",
    ]
    for key, value in variant_cfg.items():
        if key != "model":
            overrides.append(f"{key}={value}")
    if task == "lp" and model_name == "mopf_iamoc":
        # The formal LP runner automatically repeats [5, 5] to three hops
        # only when model.name == "mopf". Keep IAMOC's effective sampler
        # identical to V0 without touching the frozen task implementation.
        overrides.append("task.num_neighbors=[5,5,5]")
    if smoke:
        overrides.extend(
            [
                "task.epochs=1",
                "task.eval_every=1",
                "task.patience=1",
                "task.early_stop_min_epoch=1",
            ]
        )
    command = [sys.executable, "-m", "src.main", *overrides]
    config_record = {
        "dataset": dataset,
        "task": task,
        "variant": variant,
        "seed": seed,
        "device": device,
        "model": model_name,
        "overrides": overrides,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(run_dir),
        "smoke": smoke,
    }
    return run_dir, command, config_record


def _is_complete(run_dir: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            run_dir / "complete.marker",
            run_dir / "metrics.json",
            run_dir / "results.json",
            run_dir / "best.pt",
            run_dir / "resolved_config.yaml",
        )
    )


def _run_one(run_dir: Path, command: list[str], config: dict[str, Any], resume: bool, dry_run: bool) -> int:
    if dry_run:
        print(shlex.join(command))
        return 0
    status_path = run_dir / "run_record.json"
    if resume and _is_complete(run_dir):
        print(f"SKIP complete: {config['dataset']} {config['variant']} seed={config['seed']}")
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        **config,
        "command": command,
        "command_display": shlex.join(command),
        "status": "running",
        "started_at_unix": time.time(),
    }
    _write_json(status_path, record)
    log_path = run_dir / "runner.log"
    print(f"START {config['dataset']} {config['variant']} seed={config['seed']} -> {run_dir}", flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    record["duration_seconds"] = time.perf_counter() - started
    record["finished_at_unix"] = time.time()
    record["return_code"] = int(process.returncode)
    if process.returncode == 0 and _is_complete(run_dir):
        record["status"] = "complete"
        print(f"DONE  {config['dataset']} {config['variant']} seed={config['seed']}", flush=True)
    else:
        record["status"] = "failed"
        record["failure"] = (
            "training command returned non-zero"
            if process.returncode
            else "training command exited without all completion artifacts"
        )
        print(f"FAIL  {config['dataset']} {config['variant']} seed={config['seed']} (see {log_path})", flush=True)
    _write_json(status_path, record)
    return int(process.returncode or (record["status"] != "complete"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_TASK), default=list(DATASET_TASK))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run the five one-epoch Movies-NC seed-42 checks")
    args = parser.parse_args()
    if args.smoke:
        args.datasets, args.variants, args.seeds = ["Movies"], list(VARIANTS), [42]
    return args


def main() -> int:
    args = parse_args()
    failed = 0
    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                run_dir, command, config = _make_run(dataset, variant, seed, args.device, args.smoke)
                failed += _run_one(run_dir, command, config, args.resume, args.dry_run)
    print(f"Run summary: failed={failed} | root={OUTPUT_ROOT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
