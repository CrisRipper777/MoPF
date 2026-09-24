#!/usr/bin/env python3
"""Read-only P2 analysis for the formal Full + seven NC ablation matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = (
    "full", "no_relation_modulation", "fixed_semantic_reference", "last_context_only",
    "no_context_change", "no_cross_hop_interaction", "global_filter_only",
    "no_formation_conditioning",
)
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
PROTOCOL = "unified_full_graph_nc_v1"
EPS = 1.0e-12
ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["row_type"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _run_dir(root: Path, dataset: str, variant: str, seed: int) -> Path:
    return root / dataset / variant / f"seed{seed}"


def _validate_lock(root: Path, datasets: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    path = root / "provenance.lock.json"
    if not path.is_file():
        raise RuntimeError(f"missing formal provenance lock: {path}")
    lock = _load_json(path)
    expected = {
        "task": "nc", "protocol": PROTOCOL, "model": "ssi_mag_final_ablation",
        "datasets": list(datasets), "seeds": list(seeds), "variants": list(VARIANTS),
        "lp_jobs": 0, "test_used_for_selection": False, "dry_run": False,
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise RuntimeError(f"formal provenance mismatch at {key}: {lock.get(key)!r} != {value!r}")
    return lock


def _read_runs(root: Path, datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for dataset in datasets:
            for seed in seeds:
                run = _run_dir(root, dataset, variant, seed)
                required = [run / name for name in ("best.pt", "metrics.json", "results.json", "resolved_config.json", "complete.marker")]
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise RuntimeError("missing formal artifact(s): " + ", ".join(missing))
                marker = _load_json(run / "complete.marker")
                metrics = _load_json(run / "metrics.json")
                results = _load_json(run / "results.json")
                resolved = _load_json(run / "resolved_config.json")
                if marker.get("status") != "complete" or marker.get("task") != "nc" or marker.get("dataset") != dataset or marker.get("ablation") != variant or int(marker.get("seed", -1)) != seed:
                    raise RuntimeError(f"completion identity mismatch: {run}")
                if metrics.get("model") != "ssi_mag_final_ablation" or metrics.get("checkpoint_selection") != "best_val_accuracy":
                    raise RuntimeError(f"selection/model mismatch: {run}")
                if resolved.get("model", {}).get("name") != "ssi_mag_final_ablation" or resolved.get("ablation") != variant or resolved.get("task", {}).get("name") != "nc":
                    raise RuntimeError(f"resolved protocol mismatch: {run}")
                row: dict[str, Any] = {
                    "row_type": "run", "variant": variant, "dataset": dataset, "seed": seed,
                    "best_epoch": metrics.get("best_epoch"),
                    "runtime_seconds": metrics.get("runtime_seconds"),
                    "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
                    "finite": 1,
                }
                for key in METRICS:
                    value = results.get(key, {}).get("mean")
                    if not _finite_number(value):
                        row["finite"] = 0
                    row[key] = float(value) if _finite_number(value) else math.nan
                for key in ("best_epoch", "runtime_seconds", "peak_gpu_memory_mib"):
                    if row[key] is not None and not _finite_number(row[key]):
                        row["finite"] = 0
                rows.append(row)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            subset = [row for row in rows if row["variant"] == variant and row["dataset"] == dataset]
            if len(subset) != 3:
                raise RuntimeError(f"expected 3 seeds for {variant}/{dataset}, found {len(subset)}")
            row: dict[str, Any] = {"row_type": "dataset_summary", "variant": variant, "dataset": dataset, "seed_count": len(subset), "finite_count": sum(int(item["finite"]) for item in subset)}
            for key in METRICS + ("best_epoch", "runtime_seconds", "peak_gpu_memory_mib"):
                values = [float(item[key]) for item in subset if _finite_number(item.get(key))]
                if values:
                    row[f"{key}_mean"] = statistics.mean(values)
                    row[f"{key}_population_std"] = statistics.pstdev(values)
            dataset_rows.append(row)
    all_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        subset = [row for row in dataset_rows if row["variant"] == variant]
        row = {"row_type": "all_dataset_summary", "variant": variant, "dataset_count": len(subset), "run_count": len(rows) // len(VARIANTS), "aggregation": "unweighted_mean_of_dataset_means"}
        for key in METRICS + ("best_epoch", "runtime_seconds", "peak_gpu_memory_mib"):
            values = [float(item[f"{key}_mean"]) for item in subset if f"{key}_mean" in item]
            if values:
                row[f"{key}_mean"] = statistics.mean(values)
                row[f"{key}_population_std"] = statistics.pstdev(values)
        all_rows.append(row)
    return dataset_rows, all_rows


def _paired(rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["variant"], row["dataset"], int(row["seed"])): row for row in rows}
    out: list[dict[str, Any]] = []
    for variant in VARIANTS[1:]:
        for dataset in DATASETS:
            for seed in SEEDS:
                full, ablation = lookup[("full", dataset, seed)], lookup[(variant, dataset, seed)]
                row = {"row_type": "paired_run", "variant": variant, "comparison": f"{variant}-Full", "dataset": dataset, "seed": seed}
                for key in METRICS:
                    row[f"delta_{key}"] = ablation[key] - full[key]
                out.append(row)
            subset = [row for row in out if row["row_type"] == "paired_run" and row["variant"] == variant and row["dataset"] == dataset]
            row = {"row_type": "paired_dataset_summary", "variant": variant, "comparison": f"{variant}-Full", "dataset": dataset, "seed_count": len(subset)}
            for key in METRICS:
                values = [float(item[f"delta_{key}"]) for item in subset]
                row[f"delta_{key}_mean"] = statistics.mean(values)
                row[f"delta_{key}_population_std"] = statistics.pstdev(values)
            out.append(row)
        subset = [row for row in out if row["row_type"] == "paired_run" and row["variant"] == variant]
        row = {"row_type": "paired_all_dataset_summary", "variant": variant, "comparison": f"{variant}-Full", "dataset_count": len(DATASETS), "run_count": len(subset)}
        for key in METRICS:
            dataset_means = [item[f"delta_{key}_mean"] for item in out if item["row_type"] == "paired_dataset_summary" and item["variant"] == variant]
            values = [float(item[f"delta_{key}"]) for item in subset]
            row[f"delta_{key}_mean"] = statistics.mean(dataset_means)
            row[f"delta_{key}_population_std"] = statistics.pstdev(dataset_means)
            row[f"better_count_{key}"] = sum(value > EPS for value in values)
            row[f"worse_count_{key}"] = sum(value < -EPS for value in values)
            row[f"tie_count_{key}"] = len(values) - row[f"better_count_{key}"] - row[f"worse_count_{key}"]
        out.append(row)
    return out


def _tables(dataset_rows: list[dict[str, Any]], metric_prefix: str) -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            source = next(item for item in dataset_rows if item["variant"] == variant and item["dataset"] == dataset)
            acc_mean, acc_std = source[f"{metric_prefix}_acc_mean"], source[f"{metric_prefix}_acc_population_std"]
            f1_mean, f1_std = source[f"{metric_prefix}_macro_f1_mean"], source[f"{metric_prefix}_macro_f1_population_std"]
            rows.append({"variant": variant, "dataset": dataset, "accuracy_mean": acc_mean, "accuracy_population_std": acc_std, "accuracy_mean_pm_std": f"{100*acc_mean:.2f} ± {100*acc_std:.2f}", "macro_f1_mean": f1_mean, "macro_f1_population_std": f1_std, "macro_f1_mean_pm_std": f"{100*f1_mean:.2f} ± {100*f1_std:.2f}"})
    return rows


def _full_vs_u(root: Path, p18_path: Path) -> list[dict[str, Any]]:
    if not p18_path.is_file():
        return [{"status": "unavailable", "reason": f"missing {p18_path}"}]
    p18_rows = list(csv.DictReader(p18_path.open(encoding="utf-8")))
    p18 = {(row["dataset"], int(row["seed"])): row for row in p18_rows if row.get("row_type") == "run" and row.get("variant") == "U"}
    formal = {(row["dataset"], int(row["seed"])): row for row in _read_runs(root, DATASETS, SEEDS) if row["variant"] == "full"}
    out = []
    for key in sorted(formal):
        if key not in p18:
            out.append({"dataset": key[0], "seed": key[1], "status": "unavailable"})
            continue
        row = {"dataset": key[0], "seed": key[1], "status": "ok"}
        for metric in METRICS:
            row[f"full_minus_u_{metric}"] = formal[key][metric] - float(p18[key][metric])
        out.append(row)
    return out


def _report(output: Path, summary: dict[str, Any], all_rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# P2 Final NC Ablation Report", "",
        "The canonical architecture remains frozen as P1.8 U. This report describes the pre-registered Full plus seven NC variants; it does not reopen architecture search.", "",
        "## Integrity", "",
        f"- Formal jobs: {summary['completed_jobs']}/{summary['expected_jobs']}; finite runs: {summary['finite_runs']}/{summary['expected_jobs']}; errors: {summary['errors']}",
        f"- Protocol: `{summary['protocol']}`; LP jobs: {summary['lp_jobs']}; test used for selection: `{summary['test_used_for_selection']}`.",
        f"- Provenance commit: `{summary['git_commit']}`.", "",
        "## Performance", "",
        "`p2_table_validation.csv` is the development-facing validation table. `p2_table_test.csv` is descriptive paper output evaluated only at validation-selected checkpoints.",
        "Means and population standard deviations are across three seeds per dataset. The all-dataset summaries are unweighted means of five dataset means; no pooled node/dataset uncertainty is used.", "",
        "## Paired interpretation", "",
        "`p2_paired_deltas.csv` reports same-seed ablation minus Full deltas, including 15-run better/worse/tie counts. No p-values or significance claims are made.",
        "Each variant is interpreted only against its pre-registered question. A positive ablation delta is reported descriptively and does not trigger redesign or tuning.", "",
        "## Full versus historical P1.8 U", "",
        "This is an equivalence/development sanity comparison only. Historical U is not a P2 selection input.", "",
        "## Boundary", "",
        "No LP, hyperparameter search, auxiliary loss, test-based tuning, seed deletion, post-hoc ablation, or architecture redesign was performed.", "",
    ]
    (output / "p2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/ssi_mag_final_nc_ablation"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssi_mag_final_nc_ablation_analysis"))
    parser.add_argument("--p18-performance", type=Path, default=Path("outputs/ssi_mag_v31_p18_analysis/p18_performance.csv"))
    args = parser.parse_args()
    root = (ROOT / args.input_root).resolve() if not args.input_root.is_absolute() else args.input_root.resolve()
    output = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    p18 = (ROOT / args.p18_performance).resolve() if not args.p18_performance.is_absolute() else args.p18_performance.resolve()
    lock = _validate_lock(root, DATASETS, SEEDS)
    rows = _read_runs(root, DATASETS, SEEDS)
    dataset_rows, all_rows = _aggregate(rows)
    paired_rows = _paired(rows, dataset_rows, all_rows)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "p2_runs.csv", rows)
    _write_csv(output / "p2_dataset_summary.csv", dataset_rows)
    _write_csv(output / "p2_paired_deltas.csv", paired_rows)
    _write_csv(output / "p2_all_dataset_summary.csv", all_rows)
    _write_csv(output / "p2_table_test.csv", _tables(dataset_rows, "test"))
    _write_csv(output / "p2_table_validation.csv", _tables(dataset_rows, "val"))
    full_vs_u = _full_vs_u(root, p18)
    summary = {
        "schema": "ssi_mag_final_p2_analysis_v1",
        "git_commit": lock["git_commit"], "provenance_lock": str(root / "provenance.lock.json"),
        "protocol": PROTOCOL, "datasets": list(DATASETS), "seeds": list(SEEDS), "variants": list(VARIANTS),
        "expected_jobs": len(DATASETS) * len(SEEDS) * len(VARIANTS), "completed_jobs": len(rows),
        "finite_runs": sum(int(row["finite"]) for row in rows), "errors": [], "lp_jobs": 0,
        "test_used_for_selection": False, "training_invoked": False, "ablation_invoked": False,
        "all_dataset_aggregation": "unweighted_mean_of_dataset_means",
        "full_vs_p18_u": full_vs_u,
    }
    (output / "p2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _report(output, summary, all_rows, paired_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
