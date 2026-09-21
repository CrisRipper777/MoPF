#!/usr/bin/env python3
"""Run the validation-only PPC sweep and frozen IAMOC-v2 NC experiments."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "iamoc_v2"
SWEEP_WEIGHTS = (0.001, 0.01, 0.05)
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
VARIANTS = {
    "R0": ("none", 0.0),
    "R1": ("rank1", 0.0),
    "R2": ("profile", 0.0),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_dir(dataset: str, variant: str, seed: int, ppc_weight: float | None) -> Path:
    if variant == "R3" and ppc_weight is not None:
        return OUTPUT_ROOT / "sweep" / f"ppc_{ppc_weight:g}" / dataset / f"seed_{seed}"
    return OUTPUT_ROOT / dataset / variant / f"seed_{seed}"


def _make_run(
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    *,
    ppc_weight: float = 0.0,
    validation_only: bool = False,
) -> tuple[Path, list[str], dict[str, Any]]:
    model_mode, default_ppc = VARIANTS.get(variant, ("profile", ppc_weight))
    if variant == "R3":
        model_mode, model_ppc = "profile", float(ppc_weight)
    else:
        model_ppc = default_ppc
    run_dir = _run_dir(dataset, variant, seed, ppc_weight if validation_only else None)
    checkpoint_path = run_dir / "best.pt"
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mopf_iamoc_v2",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        "paths.output_root=outputs/iamoc_v2",
        f"task.save_ckpt_path={checkpoint_path}",
        f"hydra.run.dir={run_dir}",
        f"model.relation_state_mode={model_mode}",
        "model.relation_conditioning=none",
        f"model.ppc_weight={model_ppc}",
    ]
    if validation_only:
        overrides.append("task.evaluate_test=false")
    config = {
        "dataset": dataset,
        "task": "nc",
        "variant": variant,
        "seed": int(seed),
        "device": device,
        "relation_state_mode": model_mode,
        "relation_conditioning": "none",
        "model_ppc_weight": model_ppc,
        "task_aux_weight": 1.0,
        "effective_aux_coefficient": model_ppc,
        "validation_only": validation_only,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(run_dir),
        "overrides": overrides,
    }
    return run_dir, [sys.executable, "-m", "src.main", *overrides], config


def _is_complete(run_dir: Path) -> bool:
    return all(
        (run_dir / name).is_file()
        for name in ("complete.marker", "metrics.json", "results.json", "best.pt", "resolved_config.yaml")
    )


def _logged_ppc_summary(run_dir: Path, weight: float) -> dict[str, Any]:
    log_path = run_dir / "train.log"
    if weight <= 0.0 or not log_path.is_file():
        return {"ppc_raw": None, "effective_aux_contribution": 0.0}
    values = [
        float(value)
        for value in re.findall(r"\bppc_raw\s+([-+0-9.eE]+)", log_path.read_text(encoding="utf-8"))
    ]
    if not values:
        return {"ppc_raw": None, "effective_aux_contribution": None}
    return {
        "ppc_raw": {
            "mean_logged": sum(values) / len(values),
            "min_logged": min(values),
            "max_logged": max(values),
            "epochs_logged": len(values),
            "log_precision": "rounded to 4 decimal places by the unchanged NC logger",
        },
        "effective_aux_contribution": weight * sum(values) / len(values),
    }


def _run_one(
    run_dir: Path,
    command: list[str],
    config: dict[str, Any],
    *,
    resume: bool,
    dry_run: bool,
) -> int:
    if dry_run:
        print(shlex.join(command))
        return 0
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
    _write_json(run_dir / "run_record.json", record)
    log_path = run_dir / "runner.log"
    print(
        f"START {config['dataset']} {config['variant']} seed={config['seed']} "
        f"ppc={config['model_ppc_weight']:g} -> {run_dir}",
        flush=True,
    )
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
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        record["best_epoch"] = metrics.get("best_epoch")
        record["metrics"] = metrics.get("metrics", {})
        record.update(_logged_ppc_summary(run_dir, config["model_ppc_weight"]))
        print(f"DONE  {config['dataset']} {config['variant']} seed={config['seed']}", flush=True)
    else:
        record["status"] = "failed"
        record["failure"] = (
            "training command returned non-zero"
            if process.returncode
            else "training command exited without all completion artifacts"
        )
        print(f"FAIL  {config['dataset']} {config['variant']} seed={config['seed']} (see {log_path})", flush=True)
    _write_json(run_dir / "run_record.json", record)
    return int(process.returncode or (record["status"] != "complete"))


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, dict) or "mean" not in value:
        raise ValueError(f"Missing validation metric {key!r}: {metrics}")
    result = float(value["mean"])
    if not math.isfinite(result):
        raise ValueError(f"Non-finite validation metric {key!r}: {result}")
    return result


def _select_ppc_weight() -> dict[str, Any]:
    candidates = []
    for weight in SWEEP_WEIGHTS:
        rows = []
        for dataset in DATASETS:
            run_dir = _run_dir(dataset, "R3", 42, weight)
            if not _is_complete(run_dir):
                raise RuntimeError(f"PPC sweep run incomplete: {run_dir}")
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            for split_key in metrics.get("metrics", {}):
                if split_key.startswith("test_"):
                    raise RuntimeError("Validation-only PPC sweep unexpectedly contains test metrics")
            log_text = (run_dir / "train.log").read_text(encoding="utf-8")
            if re.search(r"\b(?:nan|inf(?:inity)?)\b|traceback", log_text, flags=re.IGNORECASE):
                raise RuntimeError(f"Non-finite or failed training log found: {run_dir / 'train.log'}")
            ppc_log = _logged_ppc_summary(run_dir, weight)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": 42,
                    "best_epoch": metrics.get("best_epoch"),
                    "val_acc": _metric(metrics["metrics"], "val_acc"),
                    "val_macro_f1": _metric(metrics["metrics"], "val_macro_f1"),
                    "model_ppc_weight": weight,
                    "task_aux_weight": 1.0,
                    "effective_aux_coefficient": weight,
                    **ppc_log,
                }
            )
        candidates.append(
            {
                "ppc_weight": weight,
                "mean_val_acc": sum(row["val_acc"] for row in rows) / len(rows),
                "mean_val_macro_f1": sum(row["val_macro_f1"] for row in rows) / len(rows),
                "runs": rows,
            }
        )
    selected = max(candidates, key=lambda row: (row["mean_val_acc"], row["mean_val_macro_f1"]))
    payload = {
        "selection_metric_primary": "mean validation Accuracy across Movies/Grocery seed 42",
        "selection_metric_secondary": "mean validation Macro-F1",
        "test_metrics_used": False,
        "selected_ppc_weight": selected["ppc_weight"],
        "selected_effective_aux_coefficient": selected["ppc_weight"],
        "task_aux_weight": 1.0,
        "candidates": candidates,
    }
    _write_json(OUTPUT_ROOT / "ppc_sweep_selection.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("sweep", "formal"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device is None:
        import torch

        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    failed = 0
    if args.phase == "sweep":
        for weight in SWEEP_WEIGHTS:
            for dataset in DATASETS:
                run_dir, command, config = _make_run(
                    dataset,
                    "R3",
                    42,
                    args.device,
                    ppc_weight=weight,
                    validation_only=True,
                )
                failed += _run_one(run_dir, command, config, resume=args.resume, dry_run=args.dry_run)
        if failed == 0 and not args.dry_run:
            chosen = _select_ppc_weight()
            print(
                "PPC selection: "
                f"{chosen['selected_ppc_weight']:g} by mean validation Accuracy "
                "(Macro-F1 tie-break)",
                flush=True,
            )
    else:
        selection_path = OUTPUT_ROOT / "ppc_sweep_selection.json"
        if not selection_path.is_file():
            raise RuntimeError("Run --phase sweep first; no frozen PPC selection was found")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_weight = float(selection["selected_ppc_weight"])
        if selected_weight not in SWEEP_WEIGHTS:
            raise RuntimeError(f"Invalid frozen PPC weight: {selected_weight}")
        for dataset in args.datasets:
            for variant in ("R1", "R2", "R3"):
                for seed in args.seeds:
                    run_dir, command, config = _make_run(
                        dataset,
                        variant,
                        seed,
                        args.device,
                        ppc_weight=selected_weight if variant == "R3" else 0.0,
                    )
                    failed += _run_one(
                        run_dir,
                        command,
                        config,
                        resume=args.resume,
                        dry_run=args.dry_run,
                    )
    print(f"Run summary: failed={failed} | root={OUTPUT_ROOT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
