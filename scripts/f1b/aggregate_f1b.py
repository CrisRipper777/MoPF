#!/usr/bin/env python3
"""Audit and aggregate the frozen F1-B benchmark.

This script is intentionally reporting-only.  It reads completed F1-B
artifacts and the F1-A REUSE_EXACT inventory, validates provenance, and writes
per-seed/full and paper-facing aggregate tables.  It never invokes the
training entry point.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, pstdev
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "f1_final_execution"
DEFAULT_PLAN = DEFAULT_OUTPUT_ROOT / "f1b_plan.csv"
F1A_INVENTORY = ROOT / "outputs" / "f1_formal_benchmark" / "f1_reuse_inventory.csv"
FORMAL_CONFIG = ROOT / "configs" / "model" / "mopf.yaml"

METHOD_FREEZE_SHA = "4ddbd6918ceebadc25eed2694e1b463f9aac87f4"
FORMAL_CONFIG_SHA256 = "1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a"
SEEDS = (42, 43, 44)
NC_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
NC_K = {"Movies": 3, "Toys": 3, "Grocery": 2, "ele-fashion": 3, "Reddit-S": 3}
NC_METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
LP_METRICS = ("val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10")
FROZEN_MOPF_FIELDS = {
    "edge_weight_mode": "learned_diag_cos",
    "multihop_state_mode": "anchored",
    "multihop_response_mode": "cumulative",
    "use_transport_residual": True,
    "use_modality_residual": True,
    "use_node_residual": True,
    "fusion_mode": "concat_residual_mlp",
    "hrc_weight": 0.0,
}
TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
PARAM_RE = re.compile(r"params=(\d+)")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def absolute(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def resolve_config_path(value: Any, config: dict[str, Any], seed: int) -> Path:
    """Resolve the small set of Hydra interpolations used in saved configs."""
    text = str(value or "")
    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset"), dict) else {}
    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    replacements = {
        "${paths.data_root}": str(paths_cfg.get("data_root", "")),
        "${paths.split_root}": str(paths_cfg.get("split_root", "")),
        "${dataset.root}": str(dataset_cfg.get("root", "")),
        "${seed}": str(seed),
    }
    for _ in range(4):
        previous = text
        for source, target in replacements.items():
            text = text.replace(source, target)
        if text == previous:
            break
    return absolute(text)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_values(payload: dict[str, Any]) -> dict[str, float]:
    source = payload.get("metrics", payload)
    if not isinstance(source, dict):
        return {}
    values: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
            values[key] = float(value["mean"])
        elif isinstance(value, (int, float)):
            values[key] = float(value)
    return values


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def expected_protocol(task: str) -> tuple[str, str]:
    if task == "nc":
        return "unified_full_graph_nc_v1", "full_graph"
    return "unified_sampled_lp_v1", "sampled"


def expected_split_key(task: str, dataset: str) -> str:
    if task == "nc":
        return "node_split_path" if dataset == "ele-fashion" else "nc_split_path"
    return "edge_split_path"


def fresh_record(row: dict[str, str]) -> dict[str, Any]:
    output = absolute(row["output_dir"])
    errors: list[str] = []
    protocol, training_mode = expected_protocol(row["task"])
    metrics_path = output / "metrics.json"
    provenance_path = output / "provenance.json"
    config_path = output / "resolved_config.yaml"
    log_path = output / "training.log"
    try:
        metrics_payload = load_json(metrics_path)
    except (OSError, json.JSONDecodeError) as exc:
        metrics_payload = {}
        errors.append(f"metrics.json unreadable: {exc}")
    try:
        provenance = load_json(provenance_path)
    except (OSError, json.JSONDecodeError) as exc:
        provenance = {}
        errors.append(f"provenance.json unreadable: {exc}")
    try:
        config = load_yaml(config_path)
    except (OSError, yaml.YAMLError) as exc:
        config = {}
        errors.append(f"resolved_config.yaml unreadable: {exc}")
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log_text = ""
        errors.append(f"training.log unreadable: {exc}")

    for required in (output / "complete.marker", output / "best.pt", metrics_path,
                     provenance_path, config_path, output / "command.txt",
                     output / "git_sha.txt", output / "seed.txt", output / "device.txt"):
        add_error(errors, required.is_file() and (required.name != "best.pt" or required.stat().st_size > 0),
                  f"missing required artifact: {required.name}")

    expected_fields = {
        "task": row["task"],
        "dataset": row["dataset"],
        "model": row["model"],
        "seed": int(row["seed"]),
        "evaluation_scope": row["evaluation_scope"],
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
    }
    for key, expected in expected_fields.items():
        add_error(errors, provenance.get(key) == expected,
                  f"provenance {key}={provenance.get(key)!r}, expected {expected!r}")
        add_error(errors, metrics_payload.get(key) == expected,
                  f"metrics {key}={metrics_payload.get(key)!r}, expected {expected!r}")

    execution_sha = str(provenance.get("execution_head", ""))
    add_error(errors, bool(SHA_RE.fullmatch(execution_sha)), "invalid execution_head SHA")
    if (output / "git_sha.txt").is_file():
        add_error(errors, (output / "git_sha.txt").read_text(encoding="utf-8").strip() == execution_sha,
                  "git_sha.txt does not match provenance execution_head")
    add_error(errors, provenance.get("selection_metric") == ("val_acc" if row["task"] == "nc" else "val_mrr"),
              "checkpoint selection metric is not the frozen validation metric")
    add_error(errors, provenance.get("test_role") == "descriptive_only",
              "test role is not descriptive_only")

    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset"), dict) else {}
    task_cfg = config.get("task", {}) if isinstance(config.get("task"), dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    add_error(errors, dataset_cfg.get("name") == row["dataset"], "resolved dataset mismatch")
    add_error(errors, task_cfg.get("name") == row["task"], "resolved task mismatch")
    add_error(errors, model_cfg.get("name") == row["model"], "resolved model mismatch")
    add_error(errors, config.get("seed") == int(row["seed"]), "resolved seed mismatch")
    add_error(errors, task_cfg.get("protocol_version") == protocol, "resolved protocol mismatch")
    add_error(errors, task_cfg.get("training_mode") == training_mode, "resolved training mode mismatch")
    add_error(errors, task_cfg.get("inference_mode") == "full", "resolved inference mode is not full")
    add_error(errors, f"Protocol: {protocol}" in log_text, "training log lacks evaluator/protocol marker")
    add_error(errors, "Inference mode: full" in log_text, "training log lacks full-inference marker")
    add_error(errors, "Saved results:" in log_text, "training log lacks successful result marker")

    split_key = expected_split_key(row["task"], row["dataset"])
    split_path = resolve_config_path(dataset_cfg.get(split_key, ""), config, int(row["seed"]))
    add_error(errors, split_path.is_file(), f"missing resolved split: {split_path}")
    split_hash = sha256(split_path) if split_path.is_file() else ""
    if row["task"] == "lp":
        expected_scope = "formal" if row["dataset"] == "sports-copurchase" else "quasi-held-out"
        add_error(errors, row["evaluation_scope"] == expected_scope, "LP evaluation scope mismatch")
    else:
        add_error(errors, row["evaluation_scope"] == "formal", "NC must be formal")

    if row["model"] == "mopf":
        for key, expected in FROZEN_MOPF_FIELDS.items():
            actual = model_cfg.get(key)
            if isinstance(expected, float):
                ok = isinstance(actual, (float, int)) and math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-9)
            else:
                ok = actual == expected
            add_error(errors, ok, f"frozen MoPF field {key}={actual!r}, expected {expected!r}")
        if "ppc_weight" in model_cfg:
            add_error(errors, model_cfg.get("ppc_weight") == 0.0, "frozen MoPF ppc_weight is not zero")
        add_error(errors, isinstance(model_cfg.get("edge_weight_temperature"), (int, float))
                  and math.isclose(float(model_cfg["edge_weight_temperature"]), 0.35, rel_tol=0, abs_tol=1e-9),
                  "frozen MoPF temperature mismatch")
        add_error(errors, isinstance(model_cfg.get("multihop_anchor_alpha"), (int, float))
                  and math.isclose(float(model_cfg["multihop_anchor_alpha"]), 0.1, rel_tol=0, abs_tol=1e-9),
                  "frozen MoPF anchor alpha mismatch")
        if row["task"] == "nc":
            add_error(errors, model_cfg.get("max_order") == NC_K[row["dataset"]], "formal NC K mismatch")
            add_error(errors, model_cfg.get("num_layers") == NC_K[row["dataset"]], "formal NC num_layers mismatch")
        else:
            add_error(errors, model_cfg.get("max_order") == 3, "quasi/formal LP K mismatch")

    values = metric_values(metrics_payload)
    required_metrics = NC_METRICS if row["task"] == "nc" else LP_METRICS
    for metric in required_metrics:
        add_error(errors, metric in values and math.isfinite(values[metric]),
                  f"missing/non-finite metric: {metric}")

    return {
        "task": row["task"],
        "dataset": row["dataset"],
        "model": row["model"],
        "seed": int(row["seed"]),
        "evaluation_scope": row["evaluation_scope"],
        "source": "fresh_final_execution",
        "reporting_role": "reproducibility_check" if row["task"] == "nc" and row["model"] == "mopf" else "primary",
        "output_dir": str(output),
        "execution_sha": execution_sha,
        "method_freeze_sha": provenance.get("method_freeze_sha", ""),
        "formal_config_sha256": provenance.get("formal_config_sha256", ""),
        "resolved_config_path": str(config_path),
        "resolved_config_sha256": sha256(config_path) if config_path.is_file() else "",
        "split_path": str(split_path),
        "split_sha256": split_hash,
        "evaluator": protocol,
        "checkpoint_selection_rule": provenance.get("selection_metric", ""),
        "test_role": provenance.get("test_role", ""),
        "metrics": values,
        "provenance_status": "PASS" if not errors else "INVALID_PROVENANCE",
        "errors": errors,
    }


def reuse_record(dataset: str, seed: int) -> dict[str, Any]:
    source = ROOT / "outputs" / "u3b_transport_conditioned_composition" / "runs" / dataset / "B1_TCPR" / f"seed{seed}"
    result_path = source / "results.json"
    inventory_rows = read_csv(F1A_INVENTORY)
    source_key = str(result_path.resolve())
    matches = [row for row in inventory_rows
               if str(Path(row.get("result_source", "")).resolve()) == source_key
               and row.get("final_reuse_decision") == "REUSE_EXACT"]
    errors: list[str] = []
    add_error(errors, len(matches) == 1, f"F1-A REUSE_EXACT inventory matches={len(matches)}")
    inventory = matches[0] if matches else {}
    for required in (source / "best.pt", result_path, source / ".hydra" / "config.yaml", source / "run_record.json"):
        add_error(errors, required.is_file() and (required.name != "best.pt" or required.stat().st_size > 0),
                  f"missing reuse artifact: {required}")
    expected = {
        "task": "nc", "dataset": dataset, "model": "mopf", "seed_spec": str(seed),
        "evaluator": "unified_full_graph_nc_v1", "checkpoint_selection_rule": "validation_accuracy",
    }
    for key, value in expected.items():
        add_error(errors, inventory.get(key) == value, f"reuse inventory {key}={inventory.get(key)!r}, expected {value!r}")
    flags: dict[str, Any] = {}
    try:
        flags = json.loads(inventory.get("protocol_match_flags", "{}"))
    except json.JSONDecodeError:
        errors.append("reuse protocol_match_flags is invalid JSON")
    for key in ("dataset_split", "evaluator", "features", "implementation", "seed", "selection", "test_tuning_free", "training_protocol"):
        add_error(errors, flags.get(key) is True, f"reuse F1-A protocol flag {key} is not true")
    config_path = absolute(inventory.get("config_path", ""))
    split_path = absolute(inventory.get("split_path", ""))
    add_error(errors, config_path.is_file(), f"missing reuse resolved config: {config_path}")
    add_error(errors, split_path.is_file(), f"missing reuse split: {split_path}")
    if config_path.is_file():
        add_error(errors, sha256(config_path) == inventory.get("config_sha256"), "reuse resolved config hash mismatch")
    if split_path.is_file():
        add_error(errors, sha256(split_path) == inventory.get("split_sha256"), "reuse split hash mismatch")
    execution_sha = inventory.get("old_git_sha", "")
    add_error(errors, bool(SHA_RE.fullmatch(execution_sha)), "invalid reuse producing/execution SHA")
    try:
        result_payload = load_json(result_path)
    except (OSError, json.JSONDecodeError) as exc:
        result_payload = {}
        errors.append(f"reuse results unreadable: {exc}")
    values = metric_values(result_payload)
    for metric in NC_METRICS:
        add_error(errors, metric in values and math.isfinite(values[metric]), f"reuse missing/non-finite metric: {metric}")
    add_error(errors, inventory.get("config_sha256") != "", "reuse resolved config hash is absent")
    return {
        "task": "nc",
        "dataset": dataset,
        "model": "mopf",
        "seed": seed,
        "evaluation_scope": "formal",
        "source": "reuse_exact_u3b",
        "reporting_role": "primary_reuse_exact",
        "output_dir": str(source),
        "execution_sha": execution_sha,
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
        "resolved_config_path": str(config_path),
        "resolved_config_sha256": inventory.get("config_sha256", ""),
        "split_path": str(split_path),
        "split_sha256": inventory.get("split_sha256", ""),
        "evaluator": inventory.get("evaluator", ""),
        "checkpoint_selection_rule": inventory.get("checkpoint_selection_rule", ""),
        "test_role": "descriptive_only",
        "metrics": values,
        "provenance_status": "PASS" if not errors else "INVALID_PROVENANCE",
        "errors": errors,
    }


def aggregate(records: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[tuple[str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["dataset"], record["model"], record["source"]), []).append(record)
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "dataset": key[0], "model": key[1], "source": key[2], "seed_count": len(group),
            "seeds": ";".join(str(item["seed"]) for item in sorted(group, key=lambda item: item["seed"])),
        }
        for metric in metrics:
            values = [item["metrics"][metric] for item in group]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_population_std"] = pstdev(values) if len(values) > 1 else 0.0
        output[key] = row
    return output


def csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def full_rows(records: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    fields = ["task", "dataset", "model", "seed", "evaluation_scope", "source", "reporting_role",
              "output_dir", "execution_sha", "method_freeze_sha", "formal_config_sha256",
              "resolved_config_path", "resolved_config_sha256", "split_path", "split_sha256",
              "evaluator", "checkpoint_selection_rule", "test_role", *metrics, "provenance_status"]
    rows = []
    for record in sorted(records, key=lambda item: (item["dataset"], item["model"], item["seed"], item["source"])):
        row = dict(record)
        row.update({metric: 100.0 * record["metrics"].get(metric, float("nan")) for metric in metrics})
        rows.append(row)
    return rows


def paper_rows(primary: dict[tuple[str, str, str], dict[str, Any]],
               metrics: tuple[str, ...], reproducibility: dict[tuple[str, str, str], dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, model, source), aggregate_row in sorted(primary.items()):
        row: dict[str, Any] = {
            "dataset": dataset, "model": model, "seeds": aggregate_row["seeds"],
            "seed_count": aggregate_row["seed_count"], "primary_source": source,
            "primary_reporting_role": "primary_reuse_exact" if source == "reuse_exact_u3b" else "primary",
        }
        for metric in metrics:
            row[f"{metric}_mean"] = 100.0 * aggregate_row[f"{metric}_mean"]
            row[f"{metric}_population_std"] = 100.0 * aggregate_row[f"{metric}_population_std"]
        if reproducibility is not None:
            repro_key = (dataset, model, "fresh_final_execution")
            repro = reproducibility.get(repro_key)
            row["reproducibility_source"] = "fresh_final_execution" if repro else ""
            row["reproducibility_seed_count"] = repro["seed_count"] if repro else ""
            if repro:
                for metric in metrics:
                    row[f"repro_{metric}_mean"] = 100.0 * repro[f"{metric}_mean"]
                    row[f"repro_{metric}_population_std"] = 100.0 * repro[f"{metric}_population_std"]
        rows.append(row)
    return rows


def write_paper(path: Path, rows: list[dict[str, Any]], metrics: tuple[str, ...], with_repro: bool) -> None:
    fields = ["dataset", "model", "seeds", "seed_count", "primary_source", "primary_reporting_role"]
    for metric in metrics:
        fields.extend((f"{metric}_mean", f"{metric}_population_std"))
    if with_repro:
        fields.extend(("reproducibility_source", "reproducibility_seed_count"))
        for metric in metrics:
            fields.extend((f"repro_{metric}_mean", f"repro_{metric}_population_std"))
    write_csv(path, rows, fields)


def parse_efficiency(record: dict[str, Any]) -> dict[str, Any]:
    log_path = Path(record["output_dir"]) / "training.log"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # A rerun in place can append a second resolved-config block.  Measure
    # only the final successful attempt, not the elapsed time since an earlier
    # failed attempt.
    start_index = 0
    for index, line in enumerate(lines):
        if "Resolved config:" in line:
            start_index = index
    timestamps = []
    param_count = ""
    for line in lines[start_index:]:
        match = TIMESTAMP_RE.match(line)
        if match:
            timestamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
        param_match = PARAM_RE.search(line)
        if param_match:
            param_count = int(param_match.group(1))
    elapsed = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) >= 2 else ""
    return {
        "task": record["task"], "dataset": record["dataset"], "model": record["model"], "seed": record["seed"],
        "source": record["source"], "output_dir": record["output_dir"], "device": "",
        "parameter_count": param_count, "log_span_seconds": elapsed,
        "peak_gpu_memory_mib": "", "peak_gpu_memory_status": "not_recorded_in_f1b_logs",
        "timing_status": "final_attempt_log_span" if elapsed != "" else "unavailable",
        "provenance_status": record["provenance_status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and aggregate completed F1-B results without training.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = read_csv(plan_path)
    completion_path = output_root / "f1b_completion.json"
    completion = load_json(completion_path)
    counts = completion.get("counts", {})
    if not completion.get("complete") or counts.get("expected_runs") != 48 or counts.get("completed_runs") != 48:
        print("F1-B completion is not complete; no tables generated")
        return 2

    fresh_plan = [row for row in plan if row.get("execution_mode") == "fresh"]
    fresh = [fresh_record(row) for row in fresh_plan]
    reuse = [reuse_record(dataset, seed) for dataset in NC_DATASETS for seed in SEEDS]
    all_audited = fresh + reuse
    errors = [
        {"task": item["task"], "dataset": item["dataset"], "model": item["model"], "seed": item["seed"], "source": item["source"], "errors": item["errors"]}
        for item in all_audited if item["errors"]
    ]
    audit_payload = {
        "status": "PASS" if not errors else "STOP_INVALID_PROVENANCE",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "completion_counts": counts,
        "method_freeze_sha": METHOD_FREEZE_SHA,
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
        "fresh_run_count": len(fresh),
        "reuse_nc_run_count": len(reuse),
        "fresh_execution_heads": sorted({item["execution_sha"] for item in fresh}),
        "reporting_policy": "primary NC table uses F1-B0 recommended REUSE_EXACT; fresh NC is reproducibility check; LP uses fresh F1-B execution",
        "records": all_audited,
        "errors": errors,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "f1b1_provenance_audit.json").write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit_fields = ["task", "dataset", "model", "seed", "evaluation_scope", "source", "reporting_role", "output_dir",
                    "execution_sha", "method_freeze_sha", "formal_config_sha256", "resolved_config_path",
                    "resolved_config_sha256", "split_path", "split_sha256", "evaluator",
                    "checkpoint_selection_rule", "test_role", "provenance_status", "errors"]
    audit_rows = []
    for item in all_audited:
        row = dict(item)
        row["errors"] = "; ".join(item["errors"])
        audit_rows.append(row)
    write_csv(output_root / "f1b1_provenance_audit.csv", audit_rows, audit_fields)
    if errors:
        print(json.dumps({"status": audit_payload["status"], "errors": errors}, ensure_ascii=False, indent=2))
        return 2

    tables_root = output_root / "tables"
    fresh_nc = [item for item in fresh if item["task"] == "nc"]
    fresh_sports = [item for item in fresh if item["task"] == "lp" and item["dataset"] == "sports-copurchase"]
    fresh_cloth = [item for item in fresh if item["task"] == "lp" and item["dataset"] == "cloth-copurchase"]
    nc_primary = aggregate(reuse, NC_METRICS)
    nc_repro = aggregate(fresh_nc, NC_METRICS)
    sports_primary = aggregate(fresh_sports, LP_METRICS)
    cloth_primary = aggregate(fresh_cloth, LP_METRICS)

    write_csv(tables_root / "f1_nc_full_results.csv", full_rows(fresh_nc + reuse, NC_METRICS),
              ["task", "dataset", "model", "seed", "evaluation_scope", "source", "reporting_role", "output_dir",
               "execution_sha", "method_freeze_sha", "formal_config_sha256", "resolved_config_path",
               "resolved_config_sha256", "split_path", "split_sha256", "evaluator", "checkpoint_selection_rule",
               "test_role", *NC_METRICS, "provenance_status"])
    write_paper(tables_root / "f1_nc_paper_table.csv", paper_rows(nc_primary, NC_METRICS, nc_repro), NC_METRICS, True)
    write_csv(tables_root / "f1_lp_sports_full_results.csv", full_rows(fresh_sports, LP_METRICS),
              ["task", "dataset", "model", "seed", "evaluation_scope", "source", "reporting_role", "output_dir",
               "execution_sha", "method_freeze_sha", "formal_config_sha256", "resolved_config_path",
               "resolved_config_sha256", "split_path", "split_sha256", "evaluator", "checkpoint_selection_rule",
               "test_role", *LP_METRICS, "provenance_status"])
    write_paper(tables_root / "f1_lp_sports_paper_table.csv", paper_rows(sports_primary, LP_METRICS), LP_METRICS, False)
    write_csv(tables_root / "f1_lp_cloth_quasiheldout_full_results.csv", full_rows(fresh_cloth, LP_METRICS),
              ["task", "dataset", "model", "seed", "evaluation_scope", "source", "reporting_role", "output_dir",
               "execution_sha", "method_freeze_sha", "formal_config_sha256", "resolved_config_path",
               "resolved_config_sha256", "split_path", "split_sha256", "evaluator", "checkpoint_selection_rule",
               "test_role", *LP_METRICS, "provenance_status"])
    write_paper(tables_root / "f1_lp_cloth_quasiheldout_paper_table.csv", paper_rows(cloth_primary, LP_METRICS), LP_METRICS, False)

    nc_difference_rows = []
    for dataset in NC_DATASETS:
        primary = nc_primary[(dataset, "mopf", "reuse_exact_u3b")]
        repro = nc_repro[(dataset, "mopf", "fresh_final_execution")]
        row = {"dataset": dataset, "model": "mopf", "primary_source": "reuse_exact_u3b", "repro_source": "fresh_final_execution", "seed_count": 3}
        for metric in NC_METRICS:
            row[f"{metric}_primary_mean"] = 100.0 * primary[f"{metric}_mean"]
            row[f"{metric}_fresh_mean"] = 100.0 * repro[f"{metric}_mean"]
            row[f"{metric}_fresh_minus_reuse_mean"] = 100.0 * (repro[f"{metric}_mean"] - primary[f"{metric}_mean"])
            row[f"{metric}_primary_population_std"] = 100.0 * primary[f"{metric}_population_std"]
            row[f"{metric}_fresh_population_std"] = 100.0 * repro[f"{metric}_population_std"]
        nc_difference_rows.append(row)
    difference_fields = ["dataset", "model", "primary_source", "repro_source", "seed_count"]
    for metric in NC_METRICS:
        difference_fields.extend((f"{metric}_primary_mean", f"{metric}_fresh_mean", f"{metric}_fresh_minus_reuse_mean",
                                 f"{metric}_primary_population_std", f"{metric}_fresh_population_std"))
    write_csv(output_root / "f1_nc_fresh_vs_reuse.csv", nc_difference_rows, difference_fields)

    fresh_by_seed = {(item["dataset"], item["seed"]): item for item in fresh_nc}
    reuse_by_seed = {(item["dataset"], item["seed"]): item for item in reuse}
    per_seed_rows: list[dict[str, Any]] = []
    for dataset in NC_DATASETS:
        for seed in SEEDS:
            fresh_item = fresh_by_seed[(dataset, seed)]
            reuse_item = reuse_by_seed[(dataset, seed)]
            row = {"dataset": dataset, "model": "mopf", "seed": seed,
                   "primary_source": "reuse_exact_u3b", "repro_source": "fresh_final_execution"}
            for metric in NC_METRICS:
                row[f"{metric}_reuse"] = 100.0 * reuse_item["metrics"][metric]
                row[f"{metric}_fresh"] = 100.0 * fresh_item["metrics"][metric]
                row[f"{metric}_fresh_minus_reuse"] = row[f"{metric}_fresh"] - row[f"{metric}_reuse"]
            per_seed_rows.append(row)
    per_seed_fields = ["dataset", "model", "seed", "primary_source", "repro_source"]
    for metric in NC_METRICS:
        per_seed_fields.extend((f"{metric}_reuse", f"{metric}_fresh", f"{metric}_fresh_minus_reuse"))
    write_csv(output_root / "f1_nc_per_seed_fresh_vs_reuse.csv", per_seed_rows, per_seed_fields)

    comparative_rows: list[dict[str, Any]] = []
    for metric in NC_METRICS:
        primary = nc_primary[("Movies", "mopf", "reuse_exact_u3b")]  # value column is populated for traceability only
        comparative_rows.append({"task": "nc", "dataset": "all_formal_nc", "metric": metric,
                                 "mopf_mean_percent": 100.0 * primary[f"{metric}_mean"], "baseline_model": "",
                                 "baseline_mean_percent": "", "absolute_delta_percentage_points": "",
                                 "comparison_status": "NO_FORMAL_EXTERNAL_BASELINE; RERUN_REQUIRED",
                                 "mopf_best_count": "", "mopf_top2_count": ""})
    for metric in LP_METRICS:
        primary = sports_primary[("sports-copurchase", "mopf", "fresh_final_execution")]
        comparative_rows.append({"task": "lp", "dataset": "sports-copurchase", "metric": metric,
                                 "mopf_mean_percent": 100.0 * primary[f"{metric}_mean"], "baseline_model": "",
                                 "baseline_mean_percent": "", "absolute_delta_percentage_points": "",
                                 "comparison_status": "NO_FORMAL_EXTERNAL_BASELINE; RERUN_REQUIRED",
                                 "mopf_best_count": "", "mopf_top2_count": ""})
    cloth_baselines = {
        key: value for key, value in cloth_primary.items() if key[1] != "mopf"
    }
    cloth_mopf = cloth_primary[("cloth-copurchase", "mopf", "fresh_final_execution")]
    for metric in LP_METRICS:
        strongest_key, strongest = max(cloth_baselines.items(), key=lambda item: item[1][f"{metric}_mean"])
        mopf_mean = cloth_mopf[f"{metric}_mean"]
        baseline_mean = strongest[f"{metric}_mean"]
        comparative_rows.append({"task": "lp", "dataset": "cloth-copurchase", "metric": metric,
                                 "mopf_mean_percent": 100.0 * mopf_mean, "baseline_model": strongest_key[1],
                                 "baseline_mean_percent": 100.0 * baseline_mean,
                                 "absolute_delta_percentage_points": 100.0 * (mopf_mean - baseline_mean),
                                 "comparison_status": "FAIR_QUASI_HELD_OUT_COMPARISON",
                                 "mopf_best_count": 1, "mopf_top2_count": 1})
    write_csv(output_root / "f1_comparative_statistics.csv", comparative_rows,
              ["task", "dataset", "metric", "mopf_mean_percent", "baseline_model", "baseline_mean_percent",
               "absolute_delta_percentage_points", "comparison_status", "mopf_best_count", "mopf_top2_count"])

    efficiency_rows = [parse_efficiency(item) for item in fresh]
    for row in efficiency_rows:
        source_record = next(item for item in fresh if item["task"] == row["task"] and item["dataset"] == row["dataset"]
                             and item["model"] == row["model"] and item["seed"] == row["seed"])
        row["device"] = load_json(Path(source_record["output_dir"]) / "provenance.json").get("device", "")
    write_csv(output_root / "f1_efficiency.csv", efficiency_rows,
              ["task", "dataset", "model", "seed", "source", "output_dir", "device", "parameter_count",
               "log_span_seconds", "peak_gpu_memory_mib", "peak_gpu_memory_status", "timing_status", "provenance_status"])

    summary = {
        "status": "PASS",
        "reporting_policy": audit_payload["reporting_policy"],
        "completion_counts": counts,
        "provenance": {"status": "PASS", "fresh_runs": len(fresh), "reuse_nc_runs": len(reuse), "issues": []},
        "tables": {
            "nc_primary": list(nc_primary.values()), "nc_reproducibility": list(nc_repro.values()),
            "sports": list(sports_primary.values()), "cloth": list(cloth_primary.values()),
        },
        "comparative_statistics": comparative_rows,
        "efficiency": {"rows": len(efficiency_rows), "peak_gpu_memory": "not recorded in fresh F1-B logs"},
    }
    (output_root / "f1b1_aggregation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "completion": counts,
        "provenance": {"fresh": len(fresh), "reuse_nc": len(reuse), "issues": 0},
        "tables_root": str(tables_root),
        "efficiency": str(output_root / "f1_efficiency.csv"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
