#!/usr/bin/env python3
"""Run the five-dataset NC joint plain-control for CoSI-MAG.

The control replaces all three framework stages at once:

* MRC: unit physical-edge weights;
* semantic anchor: ordinary propagation with alpha=0;
* RCMI: uniform mean over orders 0..K.

The inherited modality projections, residual refiners, fusion block, and NC
classifier remain unchanged.  This is a diagnostic control and is kept under
its own output root so it cannot be confused with the three official
single-stage ablations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "cosi_mag_final_joint_ablation" / "all_plain" / "nc"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
RUN_SEEDS = (42, 43, 44)


def unit_dir(dataset: str) -> Path:
    return OUTPUT_ROOT / dataset / "runs_42_43_44"


def command(dataset: str, device: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        "model=cosi_mag_ablation",
        "model.ablation_mode=all_plain",
        "model.multihop_anchor_alpha=0.0",
        "seed=42",
        "num_runs=3",
        f"device={device}",
        "ablation=full",
        f"task.save_ckpt_path={output_dir / 'best.pt'}",
        f"hydra.run.dir={output_dir}",
    ]


def is_complete(output_dir: Path) -> bool:
    required = [
        output_dir / "complete.marker",
        output_dir / "metrics.json",
        output_dir / "resolved_config.json",
        *(output_dir / f"best_run{index}.pt" for index in range(1, 4)),
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        marker = json.loads((output_dir / "complete.marker").read_text(encoding="utf-8"))
        config = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        marker.get("status") == "complete"
        and config.get("model", {}).get("name") == "cosi_mag_ablation"
        and config.get("model", {}).get("ablation_mode") == "all_plain"
        and config.get("model", {}).get("multihop_anchor_alpha") == 0.0
        and config.get("seed") == 42
        and config.get("num_runs") == 3
    )


def prepare_pointer(output_dir: Path) -> None:
    pointer = output_dir / "best.pt"
    if pointer.is_symlink():
        if pointer.resolve() != (output_dir / "best_run3.pt").resolve():
            raise RuntimeError(f"unexpected checkpoint pointer: {pointer} -> {pointer.readlink()}")
        return
    if pointer.exists():
        raise RuntimeError(f"refusing to replace existing non-symlink checkpoint: {pointer}")
    pointer.symlink_to("best_run3.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = []
    for dataset in args.datasets:
        output_dir = unit_dir(dataset)
        plan.append(
            {
                "dataset": dataset,
                "device": args.device,
                "output_dir": str(output_dir),
                "command": command(dataset, args.device, output_dir),
                "resume_skip": bool(args.resume and is_complete(output_dir)),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "cosi_mag_final_all_plain_nc_v1",
        "definition": {
            "relation_weights": "unit",
            "multihop_anchor_alpha": 0.0,
            "order_composition": "uniform_mean",
            "retained": [
                "modality_projection",
                "ordinary_normalized_multihop_propagation",
                "modality_residual_refinement",
                "concat_residual_fusion",
                "linear_nc_head",
            ],
        },
        "split_seed": 42,
        "run_seeds": list(RUN_SEEDS),
        "datasets": list(args.datasets),
        "device": args.device,
        "plan": plan,
    }
    manifest_path = OUTPUT_ROOT.parent / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    failures = 0
    for item in plan:
        output_dir = Path(item["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        unit_manifest = {
            "protocol": "cosi_mag_final_all_plain_nc_v1",
            "dataset": item["dataset"],
            "task": "nc",
            "model": "cosi_mag_ablation",
            "ablation_mode": "all_plain",
            "split_seed": 42,
            "run_seeds": list(RUN_SEEDS),
            "relation_weights": "unit",
            "multihop_anchor_alpha": 0.0,
            "order_composition": "uniform_mean",
            "official_single_stage_ablation": False,
        }
        (output_dir / "joint_ablation_manifest.json").write_text(
            json.dumps(unit_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if item["resume_skip"]:
            print(f"[RESUME SKIP] {item['dataset']}", flush=True)
            continue
        print(f"[RUN] all_plain {item['dataset']} seeds={RUN_SEEDS}", flush=True)
        if args.dry_run:
            print(" ".join(item["command"]), flush=True)
            continue
        prepare_pointer(output_dir)
        result = subprocess.run(item["command"], cwd=ROOT, check=False)
        if result.returncode != 0 or not is_complete(output_dir):
            failures += 1
            print(f"[FAIL] {item['dataset']} status={result.returncode}", flush=True)
        else:
            print(f"[COMPLETE] {item['dataset']}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
