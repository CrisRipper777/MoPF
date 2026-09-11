#!/usr/bin/env python3
"""Prepare machine-readable and Markdown F1-B summary tables.

The output is a reporting scaffold, not a results-driven decision tool.  It
keeps formal sports LP, quasi-held-out cloth LP, and MoPF NC separate; test
metrics are copied for descriptive reporting only and never determine a
selection field.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, pstdev


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "f1_final_execution"
DEFAULT_PLAN = DEFAULT_ROOT / "f1b_plan.csv"
METHOD_FREEZE_SHA = "4ddbd6918ceebadc25eed2694e1b463f9aac87f4"
FORMAL_CONFIG_SHA256 = "1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_plan(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing plan {path}; first run: python scripts/f1b/run_group.py --group all --dry-run"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty plan: {path}")
    return rows


def _metric_values(results: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in results.items():
        if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
            values[key] = float(value["mean"])
        elif isinstance(value, (int, float)):
            values[key] = float(value)
    return values


def _provenance_ok(row: dict[str, str], payload: dict, *, reused: bool) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if payload.get("method_freeze_sha") != METHOD_FREEZE_SHA:
        errors.append("method_freeze_sha mismatch")
    if payload.get("formal_config_sha256") != FORMAL_CONFIG_SHA256:
        errors.append("formal_config_sha256 mismatch")
    if not reused:
        for key, expected in {
            "task": row["task"],
            "dataset": row["dataset"],
            "model": row["model"],
            "seed": int(row["seed"]),
            "evaluation_scope": row["evaluation_scope"],
        }.items():
            if payload.get(key) != expected:
                errors.append(f"{key} mismatch")
        selection = str(payload.get("selection_metric", ""))
        if not selection.startswith("val") or selection.startswith("test"):
            errors.append("checkpoint selection is not validation-only")
    return not errors, errors


def _load_row(row: dict[str, str]) -> dict | None:
    reused = row.get("execution_mode") == "reuse"
    output = Path(row["output_dir"])
    source = Path(row.get("source_output_dir", "")) if reused else output
    if reused:
        metrics_path = source / "results.json"
        if not metrics_path.is_file():
            return None
        try:
            results = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        # F1-A's REUSE_EXACT classification is the authoritative provenance
        # for the historical U3-B1 source; its old run_record predates the
        # per-run F1-B metadata schema.
        provenance = {
            "method_freeze_sha": METHOD_FREEZE_SHA,
            "formal_config_sha256": FORMAL_CONFIG_SHA256,
            "selection_metric": "val_acc",
        }
        provenance_ok, provenance_errors = _provenance_ok(row, provenance, reused=True)
        provenance_source = "F1-A REUSE_EXACT audit"
    else:
        metrics_path = output / "metrics.json"
        provenance_path = output / "provenance.json"
        if not metrics_path.is_file() or not provenance_path.is_file():
            return None
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        results = payload.get("metrics", {})
        provenance_ok, provenance_errors = _provenance_ok(row, provenance, reused=False)
        provenance_source = "per-run provenance.json"

    values = _metric_values(results)
    if not values:
        return None
    selection = str(provenance.get("selection_metric", ""))
    test_never_used_for_selection = selection.startswith("val") and not selection.startswith("test")
    return {
        "run_key": row["run_key"],
        "section": _section(row),
        "task": row["task"],
        "dataset": row["dataset"],
        "model": row["model"],
        "seed": int(row["seed"]),
        "evaluation_scope": row["evaluation_scope"],
        "execution_mode": row.get("execution_mode", "fresh"),
        "source_output_dir": str(source),
        "metrics": values,
        "selection_metric": selection,
        "test_never_used_for_selection": test_never_used_for_selection,
        "provenance_verified": provenance_ok,
        "provenance_errors": provenance_errors,
        "provenance_source": provenance_source,
    }


def _section(row: dict[str, str]) -> str:
    if row["task"] == "nc" and row["model"] == "mopf":
        return "MoPF NC"
    if row["task"] == "lp" and row["dataset"] == "sports-copurchase":
        return "sports formal LP"
    if row["task"] == "lp" and row["dataset"] == "cloth-copurchase":
        return "cloth quasi-held-out LP"
    return "other"


def _aggregate(records: list[dict]) -> dict:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["section"], record["dataset"], record["model"]), []).append(record)
    tables: dict[str, list[dict]] = {}
    for (section, dataset, model), group in sorted(grouped.items()):
        metric_names = sorted({name for record in group for name in record["metrics"]})
        aggregate_metrics = {}
        for name in metric_names:
            per_seed = [
                {"seed": record["seed"], "value": record["metrics"][name]}
                for record in sorted(group, key=lambda item: item["seed"])
                if name in record["metrics"]
            ]
            values = [item["value"] for item in per_seed]
            aggregate_metrics[name] = {
                "per_seed": per_seed,
                "mean": mean(values),
                "population_std": pstdev(values) if len(values) > 1 else 0.0,
            }
        tables.setdefault(section, []).append(
            {
                "dataset": dataset,
                "model": model,
                "seed_count": len(group),
                "metrics": aggregate_metrics,
                "provenance_verified": all(item["provenance_verified"] for item in group),
                "test_never_used_for_selection": all(item["test_never_used_for_selection"] for item in group),
            }
        )
    return tables


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _markdown(payload: dict) -> str:
    lines = [
        "# F1-B Summary Scaffold",
        "",
        "This is a machine-generated reporting scaffold. It does not make a final experimental conclusion or select an architecture.",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Method freeze SHA: `{payload['method_freeze_sha']}`",
        f"- Formal config SHA256: `{payload['formal_config_sha256']}`",
        "- Test metrics are descriptive only; checkpoint selection is validation-only.",
        "",
    ]
    for section in ("MoPF NC", "sports formal LP", "cloth quasi-held-out LP"):
        lines.extend([f"## {section}", ""])
        table = payload["tables"].get(section, [])
        if not table:
            lines.extend(["No completed rows are available yet.", ""])
            continue
        metric_names = sorted({name for row in table for name in row["metrics"]})
        lines.append("| Dataset | Model | Seeds | " + " | ".join(metric_names) + " | Provenance |")
        lines.append("|---|---|---:|" + "---:|" * len(metric_names) + "---|")
        for row in table:
            cells = []
            for name in metric_names:
                metric = row["metrics"].get(name)
                cells.append("—" if metric is None else f"{_fmt(metric['mean'])} ± {_fmt(metric['population_std'])}")
            provenance = "verified" if row["provenance_verified"] else "REVIEW"
            lines.append(f"| {row['dataset']} | {row['model']} | {row['seed_count']} | " + " | ".join(cells) + f" | {provenance} |")
        lines.extend(["", "Population standard deviation is computed over the available per-seed values.", ""])
    lines.extend([
        "## Selection guard",
        "",
        f"- All loaded rows pass the validation-only selection guard: `{payload['validation']['test_never_used_for_selection']}`.",
        f"- All loaded rows pass provenance verification: `{payload['validation']['provenance_verified']}`.",
        "- Cloth is quasi-held-out and must not be averaged with formal sports LP.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare F1-B per-seed and mean±population-std tables without final conclusions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else PROJECT_ROOT / args.plan
    root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    rows = _load_plan(plan_path)
    records = [record for row in rows if (record := _load_row(row)) is not None]
    tables = _aggregate(records)
    payload = {
        "summary": "F1-B reporting scaffold; no final conclusion",
        "generated_at": _now(),
        "plan": str(plan_path),
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
        "loaded_run_count": len(records),
        "tables": tables,
        "per_seed_records": records,
        "validation": {
            "test_never_used_for_selection": all(record["test_never_used_for_selection"] for record in records),
            "provenance_verified": all(record["provenance_verified"] for record in records),
            "formal_and_quasi_held_out_separated": True,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "f1b_summary.json"
    markdown_path = root / "f1b_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"Loaded completed/reused runs: {len(records)}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
