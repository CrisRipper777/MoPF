"""Run only the six short Core Story smoke cases (NC and LP, seed 42)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation import ABLATION_SPECS, CORE_STORY_ABLATIONS
from src.formal_protocol import FORMAL_K


CASES = (
    ("nc", "Movies"),
    ("lp", "sports-copurchase"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-epoch/one-batch Core Story smoke tests only."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/core_story_ablation_smoke")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_for(task: str, dataset: str, variant: str, device: str, output_dir: Path) -> list[str]:
    k = FORMAL_K[dataset]
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        f"task={task}",
        "model=mopf",
        "seed=42",
        "num_runs=1",
        f"device={device}",
        f"ablation={variant}",
        f"model.max_order={k}",
        f"model.num_layers={k}",
        f"hydra.run.dir={output_dir}",
        f"task.save_ckpt_path={output_dir / 'best.pt'}",
        "task.epochs=1",
        "task.max_train_batches=1",
        "task.patience=1",
        "task.early_stop_min_epoch=1",
        "task.eval_every=1",
        "task.evaluate_test=false",
    ]
    if task == "lp":
        command.extend(["task.loader_num_workers=0", "task.torch_threads=2"])
    else:
        command.append("+task.torch_threads=2")
    return command


def complete(output_dir: Path) -> bool:
    return all(
        (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
        for name in (
            "complete.marker",
            "metrics.json",
            "ablation_manifest.json",
            "resolved_config.yaml",
            "resolved_config.json",
            "train.log",
            "best.pt",
        )
    )


def finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return True


def inspect_output(task: str, dataset: str, variant: str, output_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task": task,
        "dataset": dataset,
        "variant": variant,
        "seed": 42,
        "output_dir": str(output_dir),
        "status": "PASS",
        "issues": [],
    }
    for required in (
        "complete.marker",
        "metrics.json",
        "ablation_manifest.json",
        "resolved_config.yaml",
        "resolved_config.json",
        "train.log",
        "best.pt",
    ):
        if not (output_dir / required).is_file():
            record["issues"].append(f"missing {required}")
    try:
        manifest = json.loads((output_dir / "ablation_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        manifest = None
        record["issues"].append(f"invalid manifest: {exc}")
    try:
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        metrics = None
        record["issues"].append(f"invalid metrics: {exc}")
    if isinstance(manifest, dict):
        expected = ABLATION_SPECS[variant]
        for field in (
            "ablation_name",
            "task",
            "dataset",
            "seed",
            "composition_mode",
            "edge_weight_mode",
            "multihop_state_mode",
            "multihop_response_mode",
        ):
            expected_value = {
                "ablation_name": variant,
                "task": task,
                "dataset": dataset,
                "seed": 42,
                "composition_mode": "uniform" if variant == "wo_adaptive_composition" else "adaptive",
                "edge_weight_mode": "raw_uniform" if variant == "wo_relation_calibration" else "learned_diag_cos",
                "multihop_state_mode": "ordinary" if variant == "wo_semantic_anchor" else "anchored",
                "multihop_response_mode": "cumulative",
            }[field]
            if manifest.get(field) != expected_value:
                record["issues"].append(
                    f"manifest mismatch {field}: expected {expected_value!r}, got {manifest.get(field)!r}"
                )
        if manifest.get("relation_calibration") is not (variant != "wo_relation_calibration"):
            record["issues"].append("manifest relation_calibration mismatch")
        if manifest.get("semantic_anchor") is not expected.semantic_anchor:
            record["issues"].append("manifest semantic_anchor mismatch")
    if isinstance(metrics, dict):
        record["metrics_finite"] = finite_json(metrics)
        record["metric_keys"] = sorted((metrics.get("metrics") or {}).keys())
        record["best_epoch"] = metrics.get("best_epoch")
        record["runtime_seconds"] = metrics.get("runtime_seconds")
        if not record["metrics_finite"]:
            record["issues"].append("non-finite metrics payload")
        if metrics.get("task") != task or metrics.get("dataset") != dataset:
            record["issues"].append("metrics identity mismatch")
        if metrics.get("ablation") != variant or metrics.get("seed") != 42:
            record["issues"].append("metrics ablation/seed mismatch")
    if record["issues"]:
        record["status"] = "FAIL"
    return record


def main() -> int:
    args = parse_args()
    root = Path(args.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for variant in CORE_STORY_ABLATIONS:
        for task, dataset in CASES:
            output_dir = root / task / dataset / variant / "seed42"
            if complete(output_dir) and args.skip_existing:
                record = inspect_output(task, dataset, variant, output_dir)
                record["execution"] = "SKIP_COMPLETE"
            elif output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
                record = {
                    "task": task,
                    "dataset": dataset,
                    "variant": variant,
                    "seed": 42,
                    "output_dir": str(output_dir),
                    "status": "FAIL",
                    "execution": "NOT_RUN",
                    "issues": ["existing output requires --overwrite or --skip-existing"],
                }
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                command = command_for(task, dataset, variant, args.device, output_dir)
                print("[SMOKE] " + "|".join((task, dataset, variant, "42")), flush=True)
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    (output_dir / "smoke_child.log").write_text(
                        completed.stdout or "", encoding="utf-8"
                    )
                    record = {
                        "task": task,
                        "dataset": dataset,
                        "variant": variant,
                        "seed": 42,
                        "output_dir": str(output_dir),
                        "status": "FAIL",
                        "execution": "LAUNCHED",
                        "returncode": completed.returncode,
                        "issues": ["child process failed"],
                    }
                else:
                    record = inspect_output(task, dataset, variant, output_dir)
                    record["execution"] = "LAUNCHED"
                    record["returncode"] = 0
            records.append(record)

    failures = sum(record["status"] != "PASS" for record in records)
    summary = {
        "status": "PASS" if failures == 0 else "FAIL",
        "smoke_only": True,
        "formal_training_launched": False,
        "formal_run_count": 0,
        "tasks": ["nc", "lp"],
        "datasets": ["Movies", "sports-copurchase"],
        "variants": list(CORE_STORY_ABLATIONS),
        "seed": 42,
        "epochs": 1,
        "max_train_batches": 1,
        "total_cases": len(records),
        "passed_cases": len(records) - failures,
        "failed_cases": failures,
        "records": records,
    }
    (root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (root / "smoke_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["task", "dataset", "variant", "seed", "status", "execution", "returncode", "output_dir", "issues"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                field: "; ".join(record.get("issues", [])) if field == "issues" else record.get(field, "")
                for field in fieldnames
            })
    print(json.dumps({key: summary[key] for key in ("status", "total_cases", "passed_cases", "failed_cases")}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
