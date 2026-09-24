#!/usr/bin/env python3
"""Summarize completed P3 outputs without starting any job.

This interface is intentionally dormant during the preparation turn. It reads
only completed run artifacts and leaves the P2.1/P2.1a decision logic as the
source of the final FREEZE_S/P22_RESPONSE_PILOT_TRIGGERED decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(root: Path, suite: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for marker_path in sorted(root.rglob("complete.marker")):
        run = marker_path.parent
        try:
            marker = _json(marker_path)
            metrics = _json(run / "metrics.json")
            results = _json(run / "results.json")
        except (OSError, json.JSONDecodeError, FileNotFoundError):
            continue
        row: dict[str, Any] = {
            "suite": suite,
            "run_dir": str(run),
            "dataset": marker.get("dataset"),
            "seed": marker.get("seed"),
            "ablation": marker.get("ablation"),
            "model": metrics.get("model"),
            "task": marker.get("task"),
            "selection_metric": metrics.get("selection_metric"),
            "checkpoint_selection": metrics.get("checkpoint_selection"),
        }
        for key, value in results.items():
            if isinstance(value, dict) and "mean" in value:
                row[key] = value["mean"]
                row[f"{key}_std"] = value.get("std")
        rows.append(row)
    return rows


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None
    return statistics.mean(finite), (statistics.stdev(finite) if len(finite) > 1 else 0.0)


def _aggregate(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group, members in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = {key: value for key, value in zip(group_keys, group)}
        metric_keys = sorted(
            key for key in members[0]
            if key not in set(group_keys)
            and not key.endswith("_std")
            and isinstance(members[0].get(key), (int, float))
        )
        for key in metric_keys:
            mean, std = _mean_std(
                [float(member[key]) for member in members if key in member]
            )
            record[f"{key}_mean"] = mean
            record[f"{key}_std_across_runs"] = std
        record["n_runs"] = len(members)
        out.append(record)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    full = {
        (row.get("dataset"), row.get("seed")): row
        for row in rows
        if row.get("suite") == "p21_s_reference" and row.get("task") == "nc"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("suite") != "p3_nc_s_ablation":
            continue
        baseline = full.get((row.get("dataset"), row.get("seed")))
        if baseline is None:
            continue
        record = {
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "ablation": row.get("ablation"),
            "baseline_suite": "p21_s_reference",
        }
        for metric in metrics:
            if metric in row and metric in baseline:
                record[f"{metric}_delta_vs_full"] = float(row[metric]) - float(baseline[metric])
        output.append(record)
    return output


def _generic_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    full = {
        (row.get("dataset"), row.get("seed")): row
        for row in rows
        if row.get("suite") == "p21_s_reference" and row.get("task") == "nc"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        if row.get("suite") != "p3_nc_generic":
            continue
        baseline = full.get((row.get("dataset"), row.get("seed")))
        record = {
            "dataset": row.get("dataset"),
            "seed": row.get("seed"),
            "method": row.get("model"),
        }
        for metric in metrics:
            if metric in row:
                record[metric] = row[metric]
            if baseline is not None and metric in row and metric in baseline:
                record[f"{metric}_delta_vs_full"] = float(row[metric]) - float(baseline[metric])
        output.append(record)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc-s-root", default="outputs/ssi_mag_scd_p21_nc")
    parser.add_argument("--nc-ablation-root", default="outputs/scd_p3_nc_ablation")
    parser.add_argument("--nc-generic-root", default="outputs/scd_p3_nc_generic")
    parser.add_argument("--lp-main-root", default="outputs/scd_p3_lp_main")
    parser.add_argument("--lp-ablation-root", default="outputs/scd_p3_lp_ablation")
    parser.add_argument("--output-root", default="outputs/scd_p3_summary")
    args = parser.parse_args()

    roots = [
        ("p21_s_reference", Path(args.nc_s_root)),
        ("p3_nc_s_ablation", Path(args.nc_ablation_root)),
        ("p3_nc_generic", Path(args.nc_generic_root)),
        ("p3_lp_main", Path(args.lp_main_root)),
        ("p3_lp_s_ablation", Path(args.lp_ablation_root)),
    ]
    rows = [row for suite, root in roots for row in _rows(root, suite)]
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "run_rows.csv", rows)
    aggregate = _aggregate(rows, ("suite", "model", "ablation", "dataset"))
    nc_table = [row for row in aggregate if row.get("suite", "").startswith("p2") or row.get("suite", "").startswith("p3_nc")]
    lp_table = [row for row in aggregate if row.get("suite", "").startswith("p3_lp")]
    paired = _paired_deltas(rows)
    generic = _generic_comparison(rows)
    _write_csv(output / "aggregate.csv", aggregate)
    _write_csv(output / "nc_table.csv", nc_table)
    _write_csv(output / "lp_table.csv", lp_table)
    _write_csv(output / "paired_ablation_deltas.csv", paired)
    _write_csv(output / "generic_control_comparison.csv", generic)
    payload = {
        "schema": "mopf_p3_summary_v1",
        "formal_training_started": bool(rows),
        "run_count": len(rows),
        "decision": "deferred_to_p2_1a_evidence_audit",
        "lp_num_neighbors": [5, 5, 5],
        "rows_csv": str(output / "run_rows.csv"),
        "aggregate_csv": str(output / "aggregate.csv"),
        "nc_table_csv": str(output / "nc_table.csv"),
        "lp_table_csv": str(output / "lp_table.csv"),
        "paired_ablation_deltas_csv": str(output / "paired_ablation_deltas.csv"),
        "generic_control_comparison_csv": str(output / "generic_control_comparison.csv"),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
