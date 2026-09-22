#!/usr/bin/env python3
"""Run five-NC CoSI backbone probes and MMGCN ID-residual diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "cosi_mag_backbone_diagnostics"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
BACKBONE_MODES = ("no_graph", "last_hop", "simple_fusion", "early_fusion")
MMGCN_MODES = ("reference_fixed", "zero_fixed", "trainable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("backbone", "mmgcn"), required=True)
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_dir(family: str, mode: str, dataset: str) -> Path:
    return OUTPUT_ROOT / family / mode / dataset / "runs_42_43_44"


def is_complete(path: Path) -> bool:
    required = [path / "complete.marker", path / "metrics.json", path / "resolved_config.json"]
    required.extend(path / f"best_run{i}.pt" for i in range(1, 4))
    return all(item.is_file() for item in required)


def command(family: str, mode: str, dataset: str, device: str, path: Path) -> list[str]:
    if family == "backbone":
        model_args = [
            "model=cosi_mag_backbone_probe",
            f"model.probe_mode={mode}",
        ]
    else:
        model_args = ["model=mmgcn_probe", f"model.id_mode={mode}"]
    return [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        *model_args,
        "seed=42",
        "num_runs=3",
        f"device={device}",
        f"task.save_ckpt_path={path / 'best.pt'}",
        f"hydra.run.dir={path}",
    ]


def main() -> int:
    args = parse_args()
    allowed = BACKBONE_MODES if args.family == "backbone" else MMGCN_MODES
    modes = tuple(args.modes) if args.modes else allowed
    invalid = sorted(set(modes) - set(allowed))
    if invalid:
        raise ValueError(f"invalid {args.family} modes: {invalid}; allowed={allowed}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "modes": list(modes),
        "datasets": list(args.datasets),
        "split_seed": 42,
        "run_seeds": [42, 43, 44],
        "device": args.device,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / f"manifest_{args.family}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    failures = 0
    for mode in modes:
        for dataset in args.datasets:
            path = run_dir(args.family, mode, dataset)
            if args.resume and is_complete(path):
                print(f"[RESUME SKIP] {args.family}/{mode}/{dataset}", flush=True)
                continue
            path.mkdir(parents=True, exist_ok=True)
            pointer = path / "best.pt"
            if not pointer.exists() and not pointer.is_symlink():
                pointer.symlink_to("best_run3.pt")
            print(f"[RUN] {args.family}/{mode}/{dataset}", flush=True)
            result = subprocess.run(command(args.family, mode, dataset, args.device, path), cwd=ROOT)
            if result.returncode != 0 or not is_complete(path):
                failures += 1
                print(f"[FAIL] {args.family}/{mode}/{dataset}", flush=True)
            else:
                print(f"[COMPLETE] {args.family}/{mode}/{dataset}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
