"""Run the short F2 smoke matrix; never runs the formal seed matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation import ALL_ABLATIONS, ABLATION_SPECS


NC_DATASETS = ("Movies",)
LP_DATASETS = ("sports-copurchase",)
VARIANTS = ("full", *ALL_ABLATIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-epoch/one-batch F2 smoke tests only."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/f2_ablation_smoke")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def formal_k(dataset: str) -> int:
    return 2 if dataset == "Grocery" else 3


def command_for(
    task: str,
    dataset: str,
    variant: str,
    device: str,
    output_dir: Path,
) -> list[str]:
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
        f"hydra.run.dir={output_dir}",
        f"task.save_ckpt_path={output_dir / 'best.pt'}",
        "task.epochs=1",
        "task.max_train_batches=1",
        "task.patience=1",
        "task.early_stop_min_epoch=1",
        "task.eval_every=1",
        "task.evaluate_test=false",
    ]
    if task == "nc":
        k = formal_k(dataset)
        command.extend([f"model.max_order={k}", f"model.num_layers={k}", "+task.torch_threads=2"])
    else:
        command.extend(["task.loader_num_workers=0", "task.torch_threads=2"])
    return command


def complete(output_dir: Path) -> bool:
    return all(
        (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
        for name in ("ablation_manifest.json", "metrics.json", "best.pt", "complete.marker")
    )


def finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return value == value and abs(float(value)) != float("inf")
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return True


def inspect_output(task: str, dataset: str, variant: str, output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task": task,
        "dataset": dataset,
        "variant": variant,
        "seed": 42,
        "output_dir": str(output_dir),
        "status": "PASS",
        "issues": [],
    }
    manifest_path = output_dir / "ablation_manifest.json"
    metrics_path = output_dir / "metrics.json"
    for required in ("main.log", "train.log", "resolved_config.yaml", "best.pt", "complete.marker"):
        if not (output_dir / required).is_file():
            result["issues"].append(f"missing {required}")
    manifest = None
    metrics = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["issues"].append(f"invalid manifest: {exc}")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["issues"].append(f"invalid metrics: {exc}")
    if isinstance(manifest, dict):
        if (
            manifest.get("task") != task
            or manifest.get("dataset") != dataset
            or manifest.get("ablation_name") != variant
            or int(manifest.get("seed", -1)) != 42
        ):
            result["issues"].append("manifest identity mismatch")
        expected = ABLATION_SPECS[variant]
        for field in (
            "learned_relation_calibration",
            "semantic_anchor",
            "global_preference",
            "modality_residual",
            "node_residual",
            "tcpr",
        ):
            if manifest.get(field) is not getattr(expected, field):
                result["issues"].append(f"manifest flag mismatch: {field}")
    if isinstance(metrics, dict):
        if not finite_json(metrics):
            result["issues"].append("non-finite metrics payload")
        if metrics.get("ablation") != variant:
            result["issues"].append("metrics ablation mismatch")
        result["metric_keys"] = sorted((metrics.get("metrics") or {}).keys())
        result["best_epoch"] = metrics.get("best_epoch")
        result["runtime_seconds"] = metrics.get("runtime_seconds")
    if result["issues"]:
        result["status"] = "FAIL"
    return result


def main() -> int:
    args = parse_args()
    root = Path(args.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures = 0
    for variant in VARIANTS:
        for task, dataset in (("nc", "Movies"), ("lp", "sports-copurchase")):
            output_dir = root / task / dataset / variant / "seed42"
            if complete(output_dir) and args.skip_existing:
                record = inspect_output(task, dataset, variant, output_dir)
                record["execution"] = "SKIP_COMPLETE"
                records.append(record)
                failures += record["status"] != "PASS"
                continue
            if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
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
                records.append(record)
                failures += 1
                continue
            command = command_for(task, dataset, variant, args.device, output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print("[SMOKE] " + "|".join((task, dataset, variant, "42")), flush=True)
            completed_process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if completed_process.returncode != 0:
                (output_dir / "smoke_child.log").write_text(
                    completed_process.stdout or "", encoding="utf-8"
                )
                record = {
                    "task": task,
                    "dataset": dataset,
                    "variant": variant,
                    "seed": 42,
                    "output_dir": str(output_dir),
                    "status": "FAIL",
                    "execution": "LAUNCHED",
                    "returncode": completed_process.returncode,
                    "issues": ["child process failed"],
                }
            else:
                record = inspect_output(task, dataset, variant, output_dir)
                record["execution"] = "LAUNCHED"
                record["returncode"] = 0
            records.append(record)
            failures += record["status"] != "PASS"

    summary = {
        "status": "PASS" if failures == 0 else "FAIL",
        "smoke_only": True,
        "formal_training_launched": False,
        "interaction_training_launched": False,
        "tasks": ["nc", "lp"],
        "datasets": ["Movies", "sports-copurchase"],
        "variants": list(VARIANTS),
        "seed": 42,
        "epochs": 1,
        "max_train_batches": 1,
        "total_cases": len(records),
        "passed_cases": sum(record["status"] == "PASS" for record in records),
        "failed_cases": failures,
        "records": records,
    }
    (root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "task", "dataset", "variant", "seed", "status", "execution",
        "returncode", "output_dir", "best_epoch", "runtime_seconds",
        "metric_keys", "issues",
    ]
    with (root / "smoke_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fieldnames}
            row["metric_keys"] = ",".join(record.get("metric_keys", []))
            row["issues"] = "; ".join(record.get("issues", []))
            writer.writerow(row)
    print(json.dumps({key: summary[key] for key in ("status", "total_cases", "passed_cases", "failed_cases")}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
