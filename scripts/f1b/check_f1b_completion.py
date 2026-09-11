#!/usr/bin/env python3
"""Check completion of the pre-registered F1-B execution plan.

This checker is intentionally conservative: a ``results.json`` or a partial
log is not enough.  A run is complete only with a valid ``metrics.json``, a
non-empty ``best.pt``, and ``complete.marker`` written by ``run_group.py``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


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
            f"missing formal plan {path}; first run: "
            "python scripts/f1b/run_group.py --group all --dry-run"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty F1-B plan: {path}")
    required = {"run_key", "task", "dataset", "model", "seed", "output_dir", "execution_mode", "evaluation_scope"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"plan missing columns: {sorted(missing)}")
    return rows


def _valid_metrics(path: Path, row: dict[str, str]) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing/empty metrics.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid metrics.json: {exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict) or not payload["metrics"]:
        return False, "metrics.json has no machine-readable metrics mapping"
    for key, expected in {
        "task": row["task"],
        "dataset": row["dataset"],
        "model": row["model"],
        "seed": int(row["seed"]),
        "evaluation_scope": row["evaluation_scope"],
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
    }.items():
        if payload.get(key) != expected:
            return False, f"metrics provenance mismatch for {key}: {payload.get(key)!r}"
    selection = str(payload.get("selection_metric", ""))
    if selection.startswith("test") or not selection.startswith("val"):
        return False, f"invalid selection metric {selection!r}; test cannot select checkpoints"
    return True, "valid metrics and provenance"


def _classify(row: dict[str, str]) -> tuple[str, str]:
    if row.get("execution_mode") == "reuse":
        source = Path(row.get("source_output_dir", ""))
        required = [source / "best.pt", source / "results.json", source / ".hydra" / "config.yaml"]
        if all(path.is_file() and (path.stat().st_size > 0 if path.name == "best.pt" else True) for path in required):
            return "REUSED", "F1-A REUSE_EXACT U3-B1 source"
        return "FAILED", "missing exact U3-B1 reuse source"

    output = Path(row["output_dir"])
    marker = output / "complete.marker"
    checkpoint = output / "best.pt"
    valid, reason = _valid_metrics(output / "metrics.json", row)
    if marker.is_file() and checkpoint.is_file() and checkpoint.stat().st_size > 0 and valid:
        return "COMPLETED", reason
    if not output.exists():
        return "MISSING", reason
    return "INCOMPLETE", reason


def _section(row: dict[str, str]) -> str:
    if row["task"] == "nc" and row["model"] == "mopf":
        return "MoPF NC"
    if row["task"] == "lp" and row["dataset"] == "sports-copurchase":
        return "sports formal LP"
    if row["task"] == "lp" and row["dataset"] == "cloth-copurchase":
        return "cloth quasi-held-out LP"
    return "other"


def _write_csv(path: Path, records: list[dict[str, str]]) -> None:
    fields = ["run_key", "section", "task", "dataset", "model", "seed", "execution_mode", "status", "output_dir", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conservatively check the F1-B formal execution plan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else PROJECT_ROOT / args.plan
    root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    rows = _load_plan(plan_path)

    by_key: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_key.setdefault(row["run_key"], []).append(row)
    duplicate_keys = sorted(key for key, grouped in by_key.items() if len(grouped) > 1)

    records: list[dict[str, str]] = []
    for row in rows:
        status, reason = _classify(row)
        records.append(
            {
                "run_key": row["run_key"],
                "section": _section(row),
                "task": row["task"],
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": row["seed"],
                "execution_mode": row["execution_mode"],
                "status": status,
                "output_dir": row["output_dir"],
                "reason": reason,
            }
        )

    counts = {
        "expected_runs": len(records),
        "completed_runs": sum(record["status"] in {"COMPLETED", "REUSED"} for record in records),
        "failed_runs": sum(record["status"] == "FAILED" for record in records),
        "missing_runs": sum(record["status"] == "MISSING" for record in records),
        "incomplete_runs": sum(record["status"] == "INCOMPLETE" for record in records),
        "duplicate_runs": len(duplicate_keys),
    }
    sections: dict[str, dict[str, int]] = {}
    for section in ("MoPF NC", "sports formal LP", "cloth quasi-held-out LP"):
        subset = [record for record in records if record["section"] == section]
        sections[section] = {
            "expected": len(subset),
            "completed_or_reused": sum(record["status"] in {"COMPLETED", "REUSED"} for record in subset),
            "failed": sum(record["status"] == "FAILED" for record in subset),
            "missing": sum(record["status"] == "MISSING" for record in subset),
            "incomplete": sum(record["status"] == "INCOMPLETE" for record in subset),
        }

    payload = {
        "checker": "check_f1b_completion.py",
        "checked_at": _now(),
        "plan": str(plan_path),
        "output_root": str(root),
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
        "counts": counts,
        "sections": sections,
        "duplicate_run_keys": duplicate_keys,
        "records": records,
        "complete": counts["failed_runs"] == 0 and counts["missing_runs"] == 0 and counts["incomplete_runs"] == 0 and counts["duplicate_runs"] == 0,
    }
    json_path = root / "f1b_completion.json"
    csv_path = root / "f1b_completion.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(csv_path, records)

    print(json.dumps({"counts": counts, "sections": sections, "complete": payload["complete"]}, indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
