#!/usr/bin/env python3
"""Train the nine additional frozen RC-IAMOC R1 NC runs."""

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
OUTPUT_ROOT = ROOT / "outputs" / "rc_iamoc_final_candidate"
DATASETS = ("Toys", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def complete(run_dir: Path) -> bool:
    return all(
        (run_dir / name).is_file()
        for name in ("complete.marker", "metrics.json", "results.json", "best.pt", "resolved_config.json")
    )


def build_run(dataset: str, seed: int, device: str) -> tuple[Path, list[str], dict[str, Any]]:
    run_dir = OUTPUT_ROOT / dataset / "R1" / f"seed_{seed}"
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mopf_iamoc_v2",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        f"paths.output_root={OUTPUT_ROOT}",
        f"task.save_ckpt_path={run_dir / 'best.pt'}",
        f"hydra.run.dir={run_dir}",
        "model.hop_interaction_layers=1",
        "model.hop_interaction_heads=1",
        "model.hop_interaction_order_embedding=true",
        "model.relation_state_mode=rank1",
        "model.relation_conditioning=none",
        "model.relation_bias_init=0.10",
        "model.ppc_weight=0.0",
        "model.use_transport_residual=true",
    ]
    record = {
        "dataset": dataset,
        "task": "nc",
        "variant": "R1",
        "seed": seed,
        "device": device,
        "frozen_model": {
            "name": "mopf_iamoc_v2",
            "hop_interaction_layers": 1,
            "hop_interaction_heads": 1,
            "hop_interaction_order_embedding": True,
            "relation_state_mode": "rank1",
            "relation_conditioning": "none",
            "relation_bias_init": 0.10,
            "ppc_weight": 0.0,
            "use_transport_residual": True,
        },
        "output_dir": str(run_dir),
        "checkpoint": str(run_dir / "best.pt"),
        "overrides": overrides,
    }
    return run_dir, [sys.executable, "-m", "src.main", *overrides], record


def validate_config(run_dir: Path, expected: dict[str, Any]) -> None:
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    model = config["model"]
    task = config["task"]
    assert config["task"]["name"] == "nc"
    assert model["name"] == expected["name"] == "mopf_iamoc_v2"
    for key in (
        "hop_interaction_layers",
        "hop_interaction_heads",
        "hop_interaction_order_embedding",
        "relation_state_mode",
        "relation_conditioning",
        "relation_bias_init",
        "ppc_weight",
        "use_transport_residual",
    ):
        assert model[key] == expected[key], f"Frozen setting mismatch {key}: {model[key]} != {expected[key]}"
    assert task["protocol_version"] == "unified_full_graph_nc_v1"
    assert task["training_mode"] == "full_graph"
    assert task["optimizer"] == "adamw"
    assert task["epochs"] == 300 and task["patience"] == 30
    assert task["evaluate_test"] is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    results: list[dict[str, Any]] = []

    for dataset in DATASETS:
        for seed in SEEDS:
            run_dir, command, record = build_run(dataset, seed, args.device)
            if args.dry_run:
                print(shlex.join(command))
                continue
            if args.resume and complete(run_dir):
                validate_config(run_dir, record["frozen_model"])
                saved = json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
                print(f"SKIP complete {dataset} R1 seed={seed}", flush=True)
                results.append({**saved, "status": "complete", "resumed": True})
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            record.update(
                status="running",
                command=command,
                command_display=shlex.join(command),
                started_at_unix=time.time(),
            )
            write_json(run_dir / "run_record.json", record)
            print(f"START {dataset} R1 seed={seed} -> {run_dir}", flush=True)
            started = time.perf_counter()
            with (run_dir / "runner.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            record["duration_seconds"] = time.perf_counter() - started
            record["finished_at_unix"] = time.time()
            record["return_code"] = int(process.returncode)
            if process.returncode == 0 and complete(run_dir):
                validate_config(run_dir, record["frozen_model"])
                record["status"] = "complete"
                record["metrics"] = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")).get("metrics", {})
                print(f"DONE  {dataset} R1 seed={seed}", flush=True)
            else:
                record["status"] = "failed"
                record["failure"] = "training failed or required artifacts are missing"
                write_json(run_dir / "run_record.json", record)
                print(f"FAIL  {dataset} R1 seed={seed} (see {run_dir / 'runner.log'})", flush=True)
                return int(process.returncode or 1)
            write_json(run_dir / "run_record.json", record)
            results.append(record)

    if not args.dry_run:
        write_json(
            OUTPUT_ROOT / "training_grid_summary.json",
            {
                "datasets_added": list(DATASETS),
                "seeds": list(SEEDS),
                "variant": "R1",
                "device": args.device,
                "planned_slots": len(DATASETS) * len(SEEDS),
                "completed_slots": sum(row.get("status") == "complete" for row in results),
                "failures": sum(row.get("status") != "complete" for row in results),
                "protocol": "unified_full_graph_nc_v1",
                "runs": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
