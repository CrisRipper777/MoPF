"""Train the controlled legacy-anchored versus direct prior initialization."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("Movies", "Grocery", "ele-fashion")
MODES = ("legacy_anchored", "direct")


def _metric(metrics: dict, key: str) -> float:
    value = metrics[key]
    return float(value["mean"] if isinstance(value, dict) else value)


def _run(dataset: str, mode: str, args: argparse.Namespace) -> dict:
    out = args.output_dir.resolve() / f"{dataset}_seed42" / mode
    checkpoint = out / "best.pt"
    metrics_path = out / "metrics.json"
    out.mkdir(parents=True, exist_ok=True)
    if not (args.skip_existing and metrics_path.is_file() and checkpoint.is_file()):
        overrides = [
            f"dataset={dataset}", "task=nc", "model=mgsc_prior_init_control",
            f"model.prior_init_mode={mode}", "model.adaptive_context_gate=true",
            "model.direct_interacted_integration=true", "model.use_legacy_relation_order_bias=true",
            "seed=42", "num_runs=1", f"device={args.device}",
            f"task.save_ckpt_path={checkpoint}", f"hydra.run.dir={out}",
        ]
        subprocess.run([sys.executable, "-m", "src.main", *overrides], cwd=REPO_ROOT, check=True)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    log = (out / "main.log").read_text(encoding="utf-8") if (out / "main.log").is_file() else ""
    match = re.search(r"params=(\d+)", log)
    metrics = payload["metrics"]
    return {
        "dataset": dataset, "mode": mode, "seed": 42,
        "val_accuracy": _metric(metrics, "val_acc"), "val_macro_f1": _metric(metrics, "val_macro_f1"),
        "test_accuracy": _metric(metrics, "test_acc"), "test_macro_f1": _metric(metrics, "test_macro_f1"),
        "best_epoch": payload.get("checkpoint_metadata", {}).get("best_epoch"),
        "runtime_seconds": payload.get("runtime_seconds"), "params": int(match.group(1)) if match else "",
        "run_dir": str(out.relative_to(REPO_ROOT)),
    }


def run(args: argparse.Namespace) -> None:
    rows = [_run(dataset, mode, args) for dataset in DATASETS for mode in MODES]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    by_key = {(row["dataset"], row["mode"]): row for row in rows}
    deltas = []
    for dataset in DATASETS:
        legacy, direct = by_key[(dataset, "legacy_anchored")], by_key[(dataset, "direct")]
        deltas.append({"dataset": dataset, "direct_minus_legacy_accuracy": direct["test_accuracy"] - legacy["test_accuracy"], "direct_minus_legacy_macro_f1": direct["test_macro_f1"] - legacy["test_macro_f1"]})
    with (out / "deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0])); writer.writeheader(); writer.writerows(deltas)
    print(json.dumps({"rows": len(rows), "deltas": deltas}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/prior_init_control"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

