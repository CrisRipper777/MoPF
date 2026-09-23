"""Run the frozen P0 versus adaptive-gate P1 pilot without changing protocols."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.analyze_mgsc_gates import export_gate_summary  # noqa: E402


TASKS = (
    ("Movies", "nc"),
    ("Toys", "nc"),
    ("Grocery", "nc"),
    ("ele-fashion", "nc"),
    ("Reddit-S", "nc"),
)
MODELS = ("cosi_mag_final", "mgsc_mag")


def _device(raw: str) -> str:
    if raw == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {raw}, but CUDA is unavailable")
    return raw


def _run_dir(root: Path, dataset: str, task: str, model: str) -> Path:
    return root / f"{dataset}_{task}" / model


def _run_model(
    dataset: str,
    task: str,
    model: str,
    seed: int,
    device: str,
    root: Path,
    skip_existing: bool = False,
) -> dict:
    output_dir = _run_dir(root, dataset, task, model)
    checkpoint = output_dir / "best.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if not (skip_existing and metrics_path.is_file() and checkpoint.is_file()):
        overrides = [
            "dataset=" + dataset,
            "task=" + task,
            "model=" + model,
            f"seed={seed}",
            "num_runs=1",
            f"device={device}",
            f"task.save_ckpt_path={checkpoint}",
            f"hydra.run.dir={output_dir}",
        ]
        command = [sys.executable, "-m", "src.main", *overrides]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"training completed without {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    log_path = output_dir / "main.log"
    log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    parameter_match = re.search(r"params=(\d+)", log_text)
    metrics = payload.get("metrics", {})
    primary = "test_acc" if task == "nc" else "test_mrr"
    primary_summary = metrics.get(primary, {})
    if isinstance(primary_summary, dict):
        primary_value = primary_summary.get("mean")
    else:
        primary_value = primary_summary
    secondary = {
        key: value
        for key, value in metrics.items()
        if key not in {primary, "val_acc", "val_mrr"}
    }
    return {
        "dataset": dataset,
        "task": task,
        "model": "P0" if model == "cosi_mag_final" else "P1",
        "model_config": model,
        "seed": seed,
        "primary_metric": primary,
        "primary_value": primary_value,
        "secondary_metrics": json.dumps(metrics, sort_keys=True),
        "best_epoch": payload.get("checkpoint_metadata", {}).get("best_epoch"),
        "train_time": payload.get("runtime_seconds"),
        "params": int(parameter_match.group(1)) if parameter_match else "",
        "run_dir": str(output_dir.relative_to(REPO_ROOT)),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "crashed": False,
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_p1_gate(rows: list[dict], gate_dataset_rows: list[dict]) -> dict:
    p0 = {(row["dataset"], row["task"]): row for row in rows if row["model"] == "P0"}
    p1 = {(row["dataset"], row["task"]): row for row in rows if row["model"] == "P1"}
    drops = {}
    no_crash = len(rows) == len(TASKS) * len(MODELS) and all(not row["crashed"] for row in rows)
    for key in p0:
        p0_value = float(p0[key]["primary_value"])
        p1_value = float(p1[key]["primary_value"])
        drops[f"{key[0]}-{key[1]}"] = p1_value - p0_value
    no_large_drop = bool(drops) and all(value >= -0.01 for value in drops.values())
    relevant = [row for row in gate_dataset_rows if row["dataset"] in {"Movies", "Grocery"}]
    nontrivial = any(
        max(float(row["mean_nodewise_gate_std_text"]), float(row["mean_nodewise_gate_std_visual"])) > 0.0
        and float(row["max_order_distribution_mean_difference"]) > 1e-4
        for row in relevant
    )
    modality_difference = any(
        float(row["text_visual_mean_gate_difference"]) > 1e-4 for row in gate_dataset_rows
    )
    saturation_ok = all(float(row["max_gate_saturation_ratio"]) < 0.99 for row in gate_dataset_rows)
    passed = no_crash and no_large_drop and nontrivial and modality_difference and saturation_ok
    return {
        "p1_gate": "PASS" if passed else "FAIL",
        "no_training_crash": no_crash,
        "no_primary_metric_drop_over_1pp": no_large_drop,
        "primary_deltas_p1_minus_p0": drops,
        "gate_nontrivial": nontrivial,
        "text_visual_gate_difference": modality_difference,
        "gate_saturation_ok": saturation_ok,
    }


def run(args: argparse.Namespace) -> dict:
    device = _device(args.device)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    gate_dataset_rows: list[dict] = []
    selected_tasks = tuple((dataset, "nc") for dataset in args.datasets)
    for dataset, task in selected_tasks:
        for model in MODELS:
            row = _run_model(dataset, task, model, args.seed, device, root, args.skip_existing)
            rows.append(row)
            if model == "mgsc_mag":
                gate_dir = _run_dir(root, dataset, task, model) / "gate_analysis"
                _, _, dataset_summary = export_gate_summary(
                    Path(REPO_ROOT / row["checkpoint"]),
                    dataset,
                    task,
                    args.seed,
                    gate_dir,
                    torch.device(device),
                )
                with dataset_summary.open("r", encoding="utf-8", newline="") as handle:
                    gate_dataset_rows.extend(list(csv.DictReader(handle)))
    summary_path = root / "summary.csv"
    _write_csv(
        summary_path,
        rows,
        [
            "dataset", "task", "model", "model_config", "seed", "primary_metric", "primary_value",
            "secondary_metrics", "best_epoch", "train_time", "params", "run_dir", "checkpoint", "crashed",
        ],
    )
    gate_summary_path = root / "gate_summary.csv"
    # The per-dataset file is the compact pilot-level summary; detailed order
    # summaries stay beside each checkpoint under gate_analysis/.
    _write_csv(
        gate_summary_path,
        gate_dataset_rows,
        [
            "dataset", "task", "seed", "mean_text_gate", "mean_visual_gate", "text_visual_mean_gate_difference",
            "max_order_distribution_mean_difference", "max_gate_saturation_ratio", "mean_nodewise_gate_std_text",
            "mean_nodewise_gate_std_visual",
        ],
    )
    gate = _evaluate_p1_gate(rows, gate_dataset_rows)
    (root / "p1_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    report = REPO_ROOT / "docs" / "final" / "mgsc_p1_report.md"
    report.write_text(
        "\n".join(
            [
                "# MGSC-MAG P1 Pilot Report",
                "",
                "P1 compares the frozen CoSI-MAG reference (P0) with the adaptive context-state formation candidate on the five NC datasets under the same task runner, data splits, seed, optimizer, early stopping, classifier, and graph protocol. Sports-LP is outside this scope. The old relation-context bias and relation-order bias remain in place.",
                "",
                f"**P1 decision: {gate['p1_gate']}**",
                "",
                f"- No training crash: `{gate['no_training_crash']}`",
                f"- No primary metric drop greater than 1 percentage point: `{gate['no_primary_metric_drop_over_1pp']}`",
                f"- Gate behavior non-trivial: `{gate['gate_nontrivial']}`",
                f"- Text/Visual gate difference: `{gate['text_visual_gate_difference']}`",
                f"- Gate saturation check: `{gate['gate_saturation_ok']}`",
                "",
                "Primary metric deltas (P1 − P0):",
                "",
                *[f"- {key}: {value:.6f}" for key, value in gate["primary_deltas_p1_minus_p0"].items()],
                "",
                "Detailed performance is in `outputs/final/mgsc_p1_pilot/summary.csv`; detailed gate order distributions are stored in each P1 run's `gate_analysis/` directory and summarized in `outputs/final/mgsc_p1_pilot/gate_summary.csv`.",
                "",
                "P1 does not by itself establish that the mechanism is beneficial beyond this pilot. P2 remains conditional on the predeclared P1 gate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2))
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", choices=tuple(dataset for dataset, _ in TASKS), default=[dataset for dataset, _ in TASKS])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/mgsc_p1_pilot"))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse complete metrics/checkpoints already present.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
