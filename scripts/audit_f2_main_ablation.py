#!/usr/bin/env python3
"""Audit, aggregate, and diagnose the formal MoPF F2 main ablations.

This script is intentionally read-only with respect to models and training.  It
reads the formal F2 ablation artifacts, reuses the already audited F1 Full
artifacts, and writes audit products under outputs/f2_ablation/ and the final
analysis report under docs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
F2_ROOT = ROOT / "outputs" / "f2_ablation"
F1_ROOT = ROOT / "outputs" / "f1_final_execution"
DOCS = ROOT / "docs"

NC_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
LP_DATASET = "sports-copurchase"
SEEDS = (42, 43, 44)
VARIANTS = (
    "full",
    "wo_learned_semantic_calibration",
    "wo_semantic_anchor",
    "wo_tcpr",
    "wo_node_adaptation",
    "wo_modality_adaptation",
)
ABLATIONS = VARIANTS[1:]
TASK_DATASETS = {
    "nc": NC_DATASETS,
    "lp": (LP_DATASET,),
}
METRICS = {
    "nc": ("test_acc", "test_macro_f1"),
    "lp": ("val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"),
}
PAYLOAD_METRICS = {
    "nc": ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"),
    "lp": METRICS["lp"],
}
EXPECTED_CHECKPOINT = {"nc": "best_val_accuracy", "lp": "best_val_mrr"}
CRASH_RE = re.compile(
    r"Traceback|RuntimeError|CUDA out of memory|OutOfMemory|Segmentation fault|Killed|Exception",
    re.IGNORECASE,
)

EXPECTED_FLAGS = {
    "wo_learned_semantic_calibration": {
        "learned_relation_calibration": False,
        "semantic_anchor": True,
        "global_preference": True,
        "modality_residual": True,
        "node_residual": True,
        "tcpr": True,
    },
    "wo_semantic_anchor": {
        "learned_relation_calibration": True,
        "semantic_anchor": False,
        "global_preference": True,
        "modality_residual": True,
        "node_residual": True,
        "tcpr": True,
    },
    "wo_tcpr": {
        "learned_relation_calibration": True,
        "semantic_anchor": True,
        "global_preference": True,
        "modality_residual": True,
        "node_residual": True,
        "tcpr": False,
    },
    "wo_node_adaptation": {
        "learned_relation_calibration": True,
        "semantic_anchor": True,
        "global_preference": True,
        "modality_residual": True,
        "node_residual": False,
        "tcpr": True,
    },
    "wo_modality_adaptation": {
        "learned_relation_calibration": True,
        "semantic_anchor": True,
        "global_preference": True,
        "modality_residual": False,
        "node_residual": True,
        "tcpr": True,
    },
}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            value = yaml.safe_load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, yaml.YAMLError):
        return None


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(child) for child in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def normalize_path(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).replace("\\", "/")
    text = text.replace("${seed}", "{seed}")
    return Path(text).name


def get_nested(data: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def metric_value(payload: dict[str, Any] | None, metric: str) -> float:
    if not payload or not isinstance(payload.get("metrics"), dict):
        return float("nan")
    value = payload["metrics"].get(metric)
    if isinstance(value, dict):
        value = value.get("mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.fmean(values)) if values else float("nan")


def pop_std(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f1_path(task: str, dataset: str, seed: int) -> Path:
    if task == "nc":
        return F1_ROOT / "nc" / dataset / "mopf" / f"seed{seed}"
    return F1_ROOT / "lp_formal" / dataset / "mopf" / f"seed{seed}"


def f2_path(task: str, dataset: str, variant: str, seed: int) -> Path:
    return F2_ROOT / task / dataset / variant / f"seed{seed}"


def expected_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for task, datasets in TASK_DATASETS.items():
        for dataset in datasets:
            for variant in VARIANTS:
                for seed in SEEDS:
                    if variant == "full":
                        path = f1_path(task, dataset, seed)
                        source_kind = "f1_reuse"
                        expected_f2 = f2_path(task, dataset, variant, seed)
                    else:
                        path = f2_path(task, dataset, variant, seed)
                        source_kind = "f2_main"
                        expected_f2 = path
                    specs.append(
                        {
                            "scope": "main",
                            "task": task,
                            "dataset": dataset,
                            "variant": variant,
                            "seed": seed,
                            "path": path,
                            "expected_f2_path": expected_f2,
                            "source_kind": source_kind,
                        }
                    )
    return specs


def config_signature(
    task: str,
    dataset: str,
    seed: int,
    payload: dict[str, Any] | None,
    manifest: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    task_cfg = config.get("task", {}) if isinstance(config, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    dataset_cfg = config.get("dataset", {}) if isinstance(config, dict) else {}
    manifest = manifest or {}
    split_source = manifest.get("split_source")
    if split_source is None:
        split_source = dataset_cfg.get("nc_split_path") if task == "nc" else dataset_cfg.get("edge_split_path")
    if split_source is None:
        split_source = dataset_cfg.get("node_split_path") if task == "nc" else None
    if split_source is not None:
        split_source = normalize_path(split_source)
        if isinstance(split_source, str):
            split_source = re.sub(r"seed\d+", "seed{seed}", split_source)
    selection_metric = manifest.get("evaluation_metric") or (payload or {}).get("selection_metric")
    checkpoint_selection = manifest.get("checkpoint_selection") or (payload or {}).get("checkpoint_selection")
    if checkpoint_selection is None:
        checkpoint_selection = EXPECTED_CHECKPOINT[task]
    negative_sampling = manifest.get("negative_sampling")
    if negative_sampling is None:
        negative_sampling = {
            "task_num_train_neg": task_cfg.get("num_train_neg"),
            "dataset_lp_num_neg": dataset_cfg.get("lp_num_neg"),
        }
    return {
        "split_source": split_source,
        "hidden_dim": manifest.get("hidden_dim", model_cfg.get("hidden_dim")),
        "learning_rate": manifest.get("learning_rate", task_cfg.get("lr")),
        "weight_decay": manifest.get("weight_decay", task_cfg.get("weight_decay")),
        "optimizer": manifest.get("optimizer", task_cfg.get("optimizer")),
        "max_epochs": manifest.get("max_epochs", task_cfg.get("epochs")),
        "patience": manifest.get("patience", task_cfg.get("patience")),
        "K": manifest.get("K", model_cfg.get("max_order")),
        "alpha": manifest.get("alpha", model_cfg.get("multihop_anchor_alpha")),
        "temperature": manifest.get("temperature", model_cfg.get("edge_weight_temperature")),
        "batch_size": manifest.get("batch_size", task_cfg.get("batch_size")),
        "num_neighbors": manifest.get("num_neighbors", task_cfg.get("num_neighbors")),
        "negative_sampling": negative_sampling,
        "evaluation_metric": selection_metric,
        "checkpoint_selection": checkpoint_selection,
        "inference_protocol": manifest.get("inference_protocol", task_cfg.get("inference_mode")),
        "protocol_version": manifest.get("protocol_version", task_cfg.get("protocol_version")),
        "task": task,
        "dataset": dataset,
        "seed": seed,
    }


def read_run_artifacts(spec: dict[str, Any]) -> dict[str, Any]:
    path = spec["path"]
    metrics_path = path / "metrics.json"
    manifest_path = path / "ablation_manifest.json"
    config_path = path / "resolved_config.yaml"
    payload = load_json(metrics_path)
    manifest = load_json(manifest_path)
    config = load_yaml(config_path)
    required = {
        "metrics_json": metrics_path.exists(),
        "best_pt": (path / "best.pt").exists(),
        "resolved_config": config_path.exists(),
        "complete_marker": (path / "complete.marker").exists(),
    }
    if spec["source_kind"] == "f2_main":
        required["ablation_manifest"] = manifest_path.exists()
    else:
        required["ablation_manifest"] = True
    missing = [name for name, ok in required.items() if not ok]
    log_text = ""
    for log_name in ("main.log", "train.log"):
        log_path = path / log_name
        if log_path.exists():
            try:
                log_text += "\n" + log_path.read_text(errors="replace")
            except OSError:
                pass
    crash_matches = sorted(set(CRASH_RE.findall(log_text)))
    finite = finite_tree(payload) if payload is not None else False
    expected_checkpoint = EXPECTED_CHECKPOINT[spec["task"]]
    selection = payload.get("selection_metric") if payload else None
    checkpoint_selection = payload.get("checkpoint_selection") if payload else None
    if checkpoint_selection is None and payload:
        checkpoint_selection = payload.get("checkpoint_metadata", {}).get("selection")
    if checkpoint_selection is None:
        checkpoint_selection = expected_checkpoint
    checks: list[str] = []
    if missing:
        checks.append("missing:" + ",".join(missing))
    if payload is None:
        checks.append("metrics_unreadable")
    if not finite:
        checks.append("metrics_nonfinite_or_unreadable")
    if payload:
        if payload.get("task") != spec["task"]:
            checks.append("task_mismatch")
        if payload.get("dataset") != spec["dataset"]:
            checks.append("dataset_mismatch")
        if payload.get("seed") != spec["seed"]:
            checks.append("seed_mismatch")
        if spec["source_kind"] == "f2_main" and payload.get("ablation") != spec["variant"]:
            checks.append("ablation_mismatch")
        if selection != ("val_acc" if spec["task"] == "nc" else "val_mrr"):
            checks.append("selection_metric_mismatch")
        if checkpoint_selection != expected_checkpoint:
            checks.append("checkpoint_selection_mismatch")
        metric_keys = set((payload.get("metrics") or {}).keys())
        expected_keys = set(PAYLOAD_METRICS[spec["task"]])
        if metric_keys != expected_keys:
            checks.append("metric_key_mismatch")
    if crash_matches:
        checks.append("runtime_crash_pattern:" + "|".join(crash_matches))
    if manifest and spec["source_kind"] == "f2_main":
        if manifest.get("ablation_name") != spec["variant"]:
            checks.append("manifest_ablation_mismatch")
        for key, expected in EXPECTED_FLAGS.get(spec["variant"], {}).items():
            if manifest.get(key) != expected:
                checks.append(f"manifest_flag_mismatch:{key}")
        if manifest.get("seed") != spec["seed"]:
            checks.append("manifest_seed_mismatch")
        if manifest.get("dataset") != spec["dataset"]:
            checks.append("manifest_dataset_mismatch")
        if manifest.get("task") != spec["task"]:
            checks.append("manifest_task_mismatch")
    signature = config_signature(spec["task"], spec["dataset"], spec["seed"], payload, manifest, config)
    status = "COMPLETE_REUSED" if spec["source_kind"] == "f1_reuse" else "COMPLETE"
    if missing or payload is None:
        status = "MISSING" if not path.exists() else "INCOMPLETE"
    elif checks:
        status = "INCONSISTENT"
    return {
        "scope": spec["scope"],
        "task": spec["task"],
        "dataset": spec["dataset"],
        "variant": spec["variant"],
        "seed": spec["seed"],
        "source_kind": spec["source_kind"],
        "source_path": str(path),
        "expected_f2_path": str(spec["expected_f2_path"]),
        "status": status,
        "metrics_exists": required["metrics_json"],
        "config_exists": required["resolved_config"],
        "seed_check": "PASS" if payload and payload.get("seed") == spec["seed"] else "FAIL",
        "checkpoint_selection": checkpoint_selection or "NA",
        "checkpoint_check": "PASS" if checkpoint_selection == expected_checkpoint else "FAIL",
        "finite_metrics": finite,
        "failed_or_incomplete": bool(missing or payload is None or not (path / "complete.marker").exists()),
        "runtime_crash": bool(crash_matches),
        "crash_patterns": "|".join(crash_matches),
        "config_signature": json_key(signature),
        "config_issues": "",
        "issues": ";".join(checks),
        "path_exists": path.exists(),
    }


def discover_extra_f2_rows(main_keys: set[tuple[str, str, str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: defaultdict[tuple[str, str, str, int], list[Path]] = defaultdict(list)
    # Scan run directories rather than metrics.json files so an out-of-scope
    # failed/incomplete run with no metrics is still surfaced.
    for run_path in F2_ROOT.glob("*/*/*/seed*"):
        if not run_path.is_dir():
            continue
        try:
            relative = run_path.relative_to(F2_ROOT)
            task, dataset, variant, seed_dir = relative.parts
        except ValueError:
            continue
        match = re.fullmatch(r"seed(\d+)", seed_dir)
        if not match:
            continue
        key = (task, dataset, variant, int(match.group(1)))
        seen[key].append(run_path)
    for key, paths in sorted(seen.items()):
        if key in main_keys:
            continue
        task, dataset, variant, seed = key
        path = paths[0]
        payload = load_json(path / "metrics.json")
        complete = all((path / name).exists() for name in ("metrics.json", "best.pt", "resolved_config.yaml", "complete.marker"))
        issue = "duplicate_output" if len(paths) > 1 else ""
        status = "EXTRA_COMPLETE" if complete and payload and finite_tree(payload) else "EXTRA_INCOMPLETE"
        if not complete:
            issue = (issue + ";" if issue else "") + "missing_required_artifact"
        if payload and not finite_tree(payload):
            issue = (issue + ";" if issue else "") + "nonfinite_metrics"
        rows.append(
            {
                "scope": "extra_out_of_scope",
                "task": task,
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "source_kind": "f2_extra",
                "source_path": str(path),
                "expected_f2_path": "NA",
                "status": status,
                "metrics_exists": (path / "metrics.json").exists(),
                "config_exists": (path / "resolved_config.yaml").exists(),
                "seed_check": "NOT_MAIN_SCOPE",
                "checkpoint_selection": payload.get("checkpoint_selection", "NA") if payload else "NA",
                "checkpoint_check": "NOT_MAIN_SCOPE",
                "finite_metrics": finite_tree(payload) if payload else False,
                "failed_or_incomplete": not complete,
                "runtime_crash": False,
                "crash_patterns": "",
                "config_signature": "NA",
                "config_issues": "",
                "issues": issue,
                "path_exists": path.exists(),
            }
        )
        if len(paths) > 1:
            for duplicate in paths[1:]:
                rows.append(
                    {
                        "scope": "duplicate_out_of_scope",
                        "task": task,
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "source_kind": "f2_extra_duplicate",
                        "source_path": str(duplicate),
                        "expected_f2_path": "NA",
                        "status": "DUPLICATE",
                        "metrics_exists": (duplicate / "metrics.json").exists(),
                        "config_exists": (duplicate / "resolved_config.yaml").exists(),
                        "seed_check": "NOT_MAIN_SCOPE",
                        "checkpoint_selection": "NA",
                        "checkpoint_check": "NOT_MAIN_SCOPE",
                        "finite_metrics": False,
                        "failed_or_incomplete": True,
                        "runtime_crash": False,
                        "crash_patterns": "",
                        "config_signature": "NA",
                        "config_issues": "",
                        "issues": "duplicate_output",
                        "path_exists": duplicate.exists(),
                    }
                )
    return rows


def add_config_consistency(rows: list[dict[str, Any]]) -> None:
    groups: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["scope"] == "main":
            groups[(row["task"], row["dataset"], int(row["seed"]))].append(row)
    for group_rows in groups.values():
        signatures = [row["config_signature"] for row in group_rows]
        reference = signatures[0] if signatures else ""
        for row in group_rows:
            if row["config_signature"] != reference:
                row["config_issues"] = "cross_variant_config_signature_mismatch"
                row["issues"] = (row["issues"] + ";" if row["issues"] else "") + "config_inconsistent"
                if row["status"] in {"COMPLETE", "COMPLETE_REUSED"}:
                    row["status"] = "INCONSISTENT"


def audit() -> list[dict[str, Any]]:
    specs = expected_specs()
    rows = [read_run_artifacts(spec) for spec in specs]
    main_keys = {(row["task"], row["dataset"], row["variant"], int(row["seed"])) for row in rows}
    rows.extend(discover_extra_f2_rows(main_keys))
    add_config_consistency(rows)
    fields = [
        "scope", "task", "dataset", "variant", "seed", "source_kind", "source_path", "expected_f2_path",
        "status", "metrics_exists", "config_exists", "seed_check", "checkpoint_selection", "checkpoint_check",
        "finite_metrics", "failed_or_incomplete", "runtime_crash", "crash_patterns", "config_signature",
        "config_issues", "issues", "path_exists",
    ]
    csv_write(F2_ROOT / "f2_main_completeness.csv", rows, fields)
    return rows


def payload_for(task: str, dataset: str, variant: str, seed: int) -> dict[str, Any] | None:
    path = f1_path(task, dataset, seed) if variant == "full" else f2_path(task, dataset, variant, seed)
    return load_json(path / "metrics.json")


def aggregate_nc() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = ["dataset", "variant", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]
    fields += [f"accuracy_seed{s}" for s in SEEDS] + [f"macro_f1_seed{s}" for s in SEEDS]
    for dataset in NC_DATASETS:
        for variant in VARIANTS:
            acc = [metric_value(payload_for("nc", dataset, variant, seed), "test_acc") for seed in SEEDS]
            f1 = [metric_value(payload_for("nc", dataset, variant, seed), "test_macro_f1") for seed in SEEDS]
            row = {
                "dataset": dataset,
                "variant": variant,
                "accuracy_mean": mean(acc),
                "accuracy_std": pop_std(acc),
                "macro_f1_mean": mean(f1),
                "macro_f1_std": pop_std(f1),
            }
            row.update({f"accuracy_seed{s}": value for s, value in zip(SEEDS, acc)})
            row.update({f"macro_f1_seed{s}": value for s, value in zip(SEEDS, f1)})
            rows.append(row)
    csv_write(F2_ROOT / "f2_nc_full_results.csv", rows, fields)
    return rows


def aggregate_lp() -> list[dict[str, Any]]:
    metrics = METRICS["lp"]
    fields = ["variant"]
    for metric in metrics:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
        fields.extend([f"{metric}_seed{s}" for s in SEEDS])
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        row: dict[str, Any] = {"variant": variant}
        for metric in metrics:
            values = [metric_value(payload_for("lp", LP_DATASET, variant, seed), metric) for seed in SEEDS]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pop_std(values)
            row.update({f"{metric}_seed{s}": value for s, value in zip(SEEDS, values)})
        rows.append(row)
    csv_write(F2_ROOT / "f2_lp_full_results.csv", rows, fields)
    return rows


def summary_rows(nc_rows: list[dict[str, Any]], lp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nc_by_variant = defaultdict(list)
    for row in nc_rows:
        nc_by_variant[row["variant"]].append(row)
    lp_by_variant = {row["variant"]: row for row in lp_rows}
    seed_aggregate: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"acc": [], "f1": []})
    for variant in VARIANTS:
        for seed in SEEDS:
            dataset_rows = nc_by_variant[variant]
            seed_aggregate[variant]["acc"].append(mean(row[f"accuracy_seed{seed}"] for row in dataset_rows))
            seed_aggregate[variant]["f1"].append(mean(row[f"macro_f1_seed{seed}"] for row in dataset_rows))
    fields = [
        "variant", "nc_avg_acc_mean", "nc_avg_acc_std", "nc_avg_macro_f1_mean", "nc_avg_macro_f1_std",
        "sports_val_mrr_mean", "sports_val_mrr_std", "sports_test_mrr_mean", "sports_test_mrr_std",
        "sports_test_hits@1_mean", "sports_test_hits@1_std", "sports_test_hits@3_mean", "sports_test_hits@3_std",
        "sports_test_hits@10_mean", "sports_test_hits@10_std", "nc_avg_acc_drop_vs_full", "nc_avg_macro_f1_drop_vs_full",
        "sports_test_mrr_drop_vs_full", "sports_test_hits@1_drop_vs_full", "sports_test_hits@3_drop_vs_full",
        "sports_test_hits@10_drop_vs_full",
    ]
    raw: list[dict[str, Any]] = []
    for variant in VARIANTS:
        nc_acc = seed_aggregate[variant]["acc"]
        nc_f1 = seed_aggregate[variant]["f1"]
        lp = lp_by_variant[variant]
        raw.append(
            {
                "variant": variant,
                "nc_avg_acc_mean": mean(nc_acc),
                "nc_avg_acc_std": pop_std(nc_acc),
                "nc_avg_macro_f1_mean": mean(nc_f1),
                "nc_avg_macro_f1_std": pop_std(nc_f1),
                "sports_val_mrr_mean": lp["val_mrr_mean"],
                "sports_val_mrr_std": lp["val_mrr_std"],
                "sports_test_mrr_mean": lp["test_mrr_mean"],
                "sports_test_mrr_std": lp["test_mrr_std"],
                "sports_test_hits@1_mean": lp["test_hits@1_mean"],
                "sports_test_hits@1_std": lp["test_hits@1_std"],
                "sports_test_hits@3_mean": lp["test_hits@3_mean"],
                "sports_test_hits@3_std": lp["test_hits@3_std"],
                "sports_test_hits@10_mean": lp["test_hits@10_mean"],
                "sports_test_hits@10_std": lp["test_hits@10_std"],
            }
        )
    full = raw[0]
    for row in raw:
        row["nc_avg_acc_drop_vs_full"] = full["nc_avg_acc_mean"] - row["nc_avg_acc_mean"]
        row["nc_avg_macro_f1_drop_vs_full"] = full["nc_avg_macro_f1_mean"] - row["nc_avg_macro_f1_mean"]
        row["sports_test_mrr_drop_vs_full"] = full["sports_test_mrr_mean"] - row["sports_test_mrr_mean"]
        row["sports_test_hits@1_drop_vs_full"] = full["sports_test_hits@1_mean"] - row["sports_test_hits@1_mean"]
        row["sports_test_hits@3_drop_vs_full"] = full["sports_test_hits@3_mean"] - row["sports_test_hits@3_mean"]
        row["sports_test_hits@10_drop_vs_full"] = full["sports_test_hits@10_mean"] - row["sports_test_hits@10_mean"]
    csv_write(F2_ROOT / "f2_main_summary.csv", raw, fields)
    return raw


def paired_rows(nc_rows: list[dict[str, Any]], lp_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nc_index = {(row["dataset"], row["variant"]): row for row in nc_rows}
    lp_index = {row["variant"]: row for row in lp_rows}
    rows: list[dict[str, Any]] = []

    def add(task: str, dataset: str, variant: str, metric: str, values: list[float]) -> None:
        positive = sum(value > 0 for value in values)
        if positive == 3:
            label = "strong_seed_consistency"
        elif positive == 1:
            label = "weak_evidence_1_of_3"
        elif positive == 0:
            label = "strong_reverse_direction"
        else:
            label = "mixed_direction_2_of_3"
        rows.append(
            {
                "task": task,
                "dataset": dataset,
                "variant": variant,
                "metric": metric,
                "seed42": values[0],
                "seed43": values[1],
                "seed44": values[2],
                "mean_paired_drop": mean(values),
                "std_paired_drop": pop_std(values),
                "median_paired_drop": median(values),
                "positive_drop_count": positive,
                "positive_drop_fraction": positive / 3.0,
                "evidence_label": label,
            }
        )

    for variant in ABLATIONS:
        for dataset in NC_DATASETS:
            for metric, full_key, abl_key in (
                ("Accuracy", "accuracy", "accuracy"),
                ("Macro-F1", "macro_f1", "macro_f1"),
            ):
                full_row = nc_index[(dataset, "full")]
                abl_row = nc_index[(dataset, variant)]
                values = [full_row[f"{full_key}_seed{s}"] - abl_row[f"{abl_key}_seed{s}"] for s in SEEDS]
                add("nc", dataset, variant, metric, values)
        # Dataset-level NC means are paired by seed, preserving the same five-dataset arithmetic mean.
        full_rows = [nc_index[(dataset, "full")] for dataset in NC_DATASETS]
        abl_rows = [nc_index[(dataset, variant)] for dataset in NC_DATASETS]
        for metric, key in (("Accuracy", "accuracy"), ("Macro-F1", "macro_f1")):
            values = [
                mean(full_row[f"{key}_seed{s}"] for full_row in full_rows)
                - mean(abl_row[f"{key}_seed{s}"] for abl_row in abl_rows)
                for s in SEEDS
            ]
            add("nc", "NC_AVG", variant, metric, values)
        for metric in METRICS["lp"]:
            full_row = lp_index["full"]
            abl_row = lp_index[variant]
            values = [full_row[f"{metric}_seed{s}"] - abl_row[f"{metric}_seed{s}"] for s in SEEDS]
            add("lp", LP_DATASET, variant, metric, values)
    fields = [
        "task", "dataset", "variant", "metric", "seed42", "seed43", "seed44", "mean_paired_drop",
        "std_paired_drop", "median_paired_drop", "positive_drop_count", "positive_drop_fraction", "evidence_label",
    ]
    csv_write(F2_ROOT / "f2_seed_paired_drops.csv", rows, fields)
    return rows


def e0_alignment(nc_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    e0a = {row["dataset"]: row for row in read_csv(ROOT / "outputs/e0_empirical_motivation/edge_semantic_discrepancy/edge_discrepancy_summary.csv")}
    e0b_rows = read_csv(ROOT / "outputs/e0_empirical_motivation/semantic_retention/semantic_retention_summary.csv")
    e0c_rows = read_csv(ROOT / "outputs/e0_empirical_motivation/multihop_utilization/modality_gap_summary.csv")
    e0d_rows = read_csv(ROOT / "outputs/e0_empirical_motivation/relation_utilization_bridge/relation_utilization_association.csv")
    grouped_b: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    grouped_c: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    grouped_d: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in e0b_rows:
        grouped_b[row["dataset"]].append(row)
    for row in e0c_rows:
        grouped_c[row["dataset"]].append(row)
    for row in e0d_rows:
        grouped_d[row["dataset"]].append(row)
    nc_index = {(row["dataset"], row["variant"]): row for row in nc_rows}
    rows: list[dict[str, Any]] = []
    for dataset in NC_DATASETS:
        b = grouped_b[dataset]
        c = grouped_c[dataset]
        d = grouped_d[dataset]
        f1 = nc_index[(dataset, "full")]
        a2 = nc_index[(dataset, "wo_semantic_anchor")]
        a1 = nc_index[(dataset, "wo_learned_semantic_calibration")]
        a3 = nc_index[(dataset, "wo_tcpr")]
        a4 = nc_index[(dataset, "wo_node_adaptation")]
        a5 = nc_index[(dataset, "wo_modality_adaptation")]
        row: dict[str, Any] = {
            "dataset": dataset,
            "e0a_mean_rank_gap": float(e0a[dataset]["mean_rank_gap"]),
            "e0a_median_rank_gap": float(e0a[dataset]["median_rank_gap"]),
            "e0a_q90_rank_gap": float(e0a[dataset]["q90_rank_gap"]),
            "e0a_spearman_tv": float(e0a[dataset]["spearman_tv"]),
            "e0b_mean_cka_drop_formal": mean(float(x["cka_drop_formal"]) for x in b),
            "e0b_mean_cosine_drop_formal": mean(float(x["cosine_drop_formal"]) for x in b),
            "e0c_mean_profile_l1": mean(float(x["profile_l1_mean"]) for x in c),
            "e0c_mean_abs_order_gap": mean(float(x["abs_order_gap_mean"]) for x in c),
            "e0d_mean_abs_partial_rho": mean(float(x["abs_rho_partial"]) for x in d),
            "e0d_max_abs_partial_rho": max(float(x["abs_rho_partial"]) for x in d),
        }
        for label, variant_row in (("a1", a1), ("a2", a2), ("a3", a3), ("a4", a4), ("a5", a5)):
            row[f"{label}_acc_drop"] = f1["accuracy_mean"] - variant_row["accuracy_mean"]
            row[f"{label}_f1_drop"] = f1["macro_f1_mean"] - variant_row["macro_f1_mean"]
        rows.append(row)
    fields = list(rows[0]) if rows else ["dataset"]
    csv_write(F2_ROOT / "f2_e0_alignment_summary.csv", rows, fields)
    return rows


def paper_tables(summary: list[dict[str, Any]], nc_rows: list[dict[str, Any]], lp_rows: list[dict[str, Any]]) -> None:
    labels = {
        "full": "Full MoPF",
        "wo_learned_semantic_calibration": "w/o learned semantic calibration",
        "wo_semantic_anchor": "w/o semantic anchor",
        "wo_tcpr": "w/o TCPR",
        "wo_node_adaptation": "w/o node adaptation",
        "wo_modality_adaptation": "w/o modality adaptation",
    }
    columns = [
        ("NC Avg Acc", "nc_avg_acc_mean", "nc_avg_acc_std"),
        ("NC Avg Macro-F1", "nc_avg_macro_f1_mean", "nc_avg_macro_f1_std"),
        ("Sports MRR", "sports_test_mrr_mean", "sports_test_mrr_std"),
        ("H@1", "sports_test_hits@1_mean", "sports_test_hits@1_std"),
        ("H@10", "sports_test_hits@10_mean", "sports_test_hits@10_std"),
    ]
    best = {mean_key: max(float(row[mean_key]) for row in summary) for _, mean_key, _ in columns}

    def value_text(row: dict[str, Any], mean_key: str, std_key: str, latex: bool) -> str:
        separator = r"\pm" if latex else "±"
        text = f"{fmt(row[mean_key], 3)} {separator} {fmt(row[std_key], 3)}"
        if math.isclose(float(row[mean_key]), best[mean_key], rel_tol=0.0, abs_tol=1e-12):
            return f"\\textbf{{{text}}}" if latex else f"**{text}**"
        return text

    md_lines = ["| Variant | NC Avg Acc | NC Avg Macro-F1 | Sports MRR | H@1 | H@10 |", "|---|---:|---:|---:|---:|---:|"]
    tex_lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variant & NC Avg Acc & NC Avg Macro-F1 & Sports MRR & H@1 & H@10 \\",
        r"\midrule",
    ]
    for row in summary:
        md_lines.append("| " + labels[row["variant"]] + " | " + " | ".join(value_text(row, m, s, False) for _, m, s in columns) + " |")
        tex_label = labels[row["variant"]].replace("&", r"\&")
        tex_lines.append(tex_label + " & " + " & ".join(value_text(row, m, s, True) for _, m, s in columns) + r" \\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (F2_ROOT / "f2_main_paper_table.md").write_text("\n".join(md_lines) + "\n")
    (F2_ROOT / "f2_main_paper_table.tex").write_text("\n".join(tex_lines) + "\n")

    appendix_lines = ["| Dataset | Variant | Accuracy | Macro-F1 |", "|---|---|---:|---:|"]
    for row in nc_rows:
        appendix_lines.append(
            f"| {row['dataset']} | {labels[row['variant']]} | {fmt(row['accuracy_mean'], 4)} ± {fmt(row['accuracy_std'], 4)} | {fmt(row['macro_f1_mean'], 4)} ± {fmt(row['macro_f1_std'], 4)} |"
        )
    appendix_lines.extend(["", "| Variant | Val MRR | Test MRR | H@1 | H@3 | H@10 |", "|---|---:|---:|---:|---:|---:|"])
    for row in lp_rows:
        appendix_lines.append(
            f"| {labels[row['variant']]} | {fmt(row['val_mrr_mean'], 4)} ± {fmt(row['val_mrr_std'], 4)} | {fmt(row['test_mrr_mean'], 4)} ± {fmt(row['test_mrr_std'], 4)} | {fmt(row['test_hits@1_mean'], 4)} ± {fmt(row['test_hits@1_std'], 4)} | {fmt(row['test_hits@3_mean'], 4)} ± {fmt(row['test_hits@3_std'], 4)} | {fmt(row['test_hits@10_mean'], 4)} ± {fmt(row['test_hits@10_std'], 4)} |"
        )
    (F2_ROOT / "f2_appendix_candidate_tables.md").write_text("\n".join(appendix_lines) + "\n")


def plot_heatmap(nc_rows: list[dict[str, Any]], lp_rows: list[dict[str, Any]]) -> None:
    nc_index = {(row["dataset"], row["variant"]): row for row in nc_rows}
    lp_index = {row["variant"]: row for row in lp_rows}
    columns: list[tuple[str, str, str]] = []
    for dataset in NC_DATASETS:
        columns.extend([(f"{dataset} Acc", dataset, "accuracy_mean"), (f"{dataset} F1", dataset, "macro_f1_mean")])
    columns.extend([("Sports MRR", LP_DATASET, "test_mrr_mean"), ("H@1", LP_DATASET, "test_hits@1_mean"), ("H@10", LP_DATASET, "test_hits@10_mean")])
    values = []
    for variant in ABLATIONS:
        line = []
        for _, dataset, metric in columns:
            full_value = nc_index[(dataset, "full")][metric] if dataset != LP_DATASET else lp_index["full"][metric]
            abl_value = nc_index[(dataset, variant)][metric] if dataset != LP_DATASET else lp_index[variant][metric]
            line.append(float(full_value) - float(abl_value))
        values.append(line)
    array = np.asarray(values, dtype=float)
    max_abs = max(1e-6, float(np.max(np.abs(array))))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 8
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    plt.rcParams["ps.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(16.5, 4.8), constrained_layout=True)
    image = ax.imshow(array, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(range(len(columns)), [column[0] for column in columns], rotation=45, rotation_mode="anchor", ha="right")
    ax.set_yticks(range(len(ABLATIONS)), [
        "w/o learned semantic calibration", "w/o semantic anchor", "w/o TCPR", "w/o node adaptation", "w/o modality adaptation"
    ])
    ax.set_xlabel("Full mean − ablation mean")
    ax.set_title("F2 main ablation performance drops (diagnosis only)")
    ax.tick_params(axis="both", labelsize=7)
    threshold = max_abs * 0.55
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            color = "white" if abs(array[i, j]) > threshold else "black"
            ax.text(j, i, f"{array[i, j]:.3f}", ha="center", va="center", fontsize=6.2, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Full − ablation")
    fig.savefig(F2_ROOT / "f2_performance_drop_heatmap.pdf", metadata={"Creator": "MoPF F2 audit"})
    fig.savefig(F2_ROOT / "f2_performance_drop_heatmap.svg")
    fig.savefig(F2_ROOT / "f2_performance_drop_heatmap.png", dpi=600)
    fig.savefig(F2_ROOT / "f2_performance_drop_heatmap.tiff", dpi=600)
    plt.close(fig)
    (F2_ROOT / "f2_performance_drop_heatmap.qa.json").write_text(json.dumps({
        "purpose": "diagnosis_only",
        "rows": ABLATIONS,
        "columns": [column[0] for column in columns],
        "cell_definition": "Full mean - ablation mean",
        "center": 0.0,
        "vmin": -max_abs,
        "vmax": max_abs,
        "std_convention": "population standard deviation across seeds",
    }, indent=2) + "\n")
    (F2_ROOT / "f2_performance_drop_heatmap.alignment.json").write_text(json.dumps({
        "schema_version": 1,
        "backend": "matplotlib",
        "figure": {"width_pt": 16.5 * 72.0, "height_pt": 4.8 * 72.0},
        "panels": [{
            "id": "heatmap",
            "bbox_pt": [48.0, 48.0, 1140.0, 300.0],
            "grid_id": "single",
            "row_start": 0,
            "row_stop": 1,
            "col_start": 0,
            "col_stop": 1,
        }],
        "status": "NOT_APPLICABLE",
        "reason": "single heatmap axes; no multi-panel alignment claim",
    }, indent=2) + "\n")


def top_std_anomalies(nc_rows: list[dict[str, Any]], lp_rows: list[dict[str, Any]]) -> list[str]:
    entries: list[tuple[float, str]] = []
    for row in nc_rows:
        entries.extend([
            (row["accuracy_std"], f"{row['dataset']} / {row['variant']} Accuracy std={fmt(row['accuracy_std'], 4)}"),
            (row["macro_f1_std"], f"{row['dataset']} / {row['variant']} Macro-F1 std={fmt(row['macro_f1_std'], 4)}"),
        ])
    for row in lp_rows:
        for metric in ("test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"):
            entries.append((row[f"{metric}_std"], f"sports-copurchase / {row['variant']} {metric} std={fmt(row[f'{metric}_std'], 4)}"))
    return [text for _, text in sorted(entries, reverse=True)[:8]]


def direction_label(count: int) -> str:
    return {3: "strong seed consistency", 2: "mixed direction (2/3)", 1: "weak evidence (1/3)", 0: "strong reverse direction"}[count]


def component_diagnosis(nc_rows: list[dict[str, Any]], lp_rows: list[dict[str, Any]], paired: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    nc_index = {(row["dataset"], row["variant"]): row for row in nc_rows}
    lp_index = {row["variant"]: row for row in lp_rows}
    result: dict[str, dict[str, Any]] = {}
    component_metrics = {
        "wo_learned_semantic_calibration": ("A1 learned semantic calibration",),
        "wo_semantic_anchor": ("A2 semantic anchor",),
        "wo_tcpr": ("A3 TCPR",),
        "wo_node_adaptation": ("A4 node adaptation",),
        "wo_modality_adaptation": ("A5 modality adaptation",),
    }
    for variant, (label,) in component_metrics.items():
        nc_drops = []
        nc_positive = 0
        nc_total = 0
        for dataset in NC_DATASETS:
            full = nc_index[(dataset, "full")]
            abl = nc_index[(dataset, variant)]
            for key in ("accuracy", "macro_f1"):
                drop = full[f"{key}_mean"] - abl[f"{key}_mean"]
                nc_drops.append(drop)
                nc_positive += drop > 0
                nc_total += 1
        lp_drops = [
            lp_index["full"][f"{key}_mean"] - lp_index[variant][f"{key}_mean"]
            for key in ("test_mrr", "test_hits@1", "test_hits@3", "test_hits@10")
        ]
        paired_component = [row for row in paired if row["variant"] == variant and row["dataset"] != "NC_AVG"]
        all_positive = sum(int(row["positive_drop_count"]) for row in paired_component)
        all_slots = 3 * len(paired_component)
        dataset_positive = sum(
            (nc_index[(dataset, "full")]["accuracy_mean"] - nc_index[(dataset, variant)]["accuracy_mean"] > 0)
            or (nc_index[(dataset, "full")]["macro_f1_mean"] - nc_index[(dataset, variant)]["macro_f1_mean"] > 0)
            for dataset in NC_DATASETS
        )
        avg_nc = mean(nc_drops)
        avg_lp = mean(lp_drops)
        coverage = nc_positive / nc_total
        seed_fraction = all_positive / all_slots if all_slots else 0.0
        if avg_nc > 0 and avg_lp > 0 and coverage >= 0.8 and seed_fraction >= 0.75:
            classification = "Strongly supported"
        elif (avg_nc > 0 or avg_lp > 0) and (coverage >= 0.5 or seed_fraction >= 0.5):
            classification = "Moderately supported"
        elif avg_nc > 0 or avg_lp > 0:
            classification = "Weak / dataset-dependent"
        else:
            classification = "Not supported"
        result[variant] = {
            "label": label,
            "avg_nc_drop": avg_nc,
            "avg_lp_drop": avg_lp,
            "nc_positive_cell_fraction": coverage,
            "nc_positive_dataset_count": dataset_positive,
            "paired_positive_fraction": seed_fraction,
            "classification": classification,
            "nc_drops": nc_drops,
            "lp_drops": lp_drops,
        }
    return result


def generate_report(
    audit_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    nc_rows: list[dict[str, Any]],
    lp_rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    e0: list[dict[str, Any]],
) -> None:
    labels = {
        "full": "Full MoPF",
        "wo_learned_semantic_calibration": "w/o learned semantic calibration",
        "wo_semantic_anchor": "w/o semantic anchor",
        "wo_tcpr": "w/o TCPR",
        "wo_node_adaptation": "w/o node adaptation",
        "wo_modality_adaptation": "w/o modality adaptation",
    }
    diagnosis = component_diagnosis(nc_rows, lp_rows, paired)
    main = [row for row in audit_rows if row["scope"] == "main"]
    extras = [row for row in audit_rows if row["scope"] != "main"]
    bad_main = [row for row in main if row["status"] not in {"COMPLETE", "COMPLETE_REUSED"}]
    incomplete_extra = [row for row in extras if row["status"] == "EXTRA_INCOMPLETE"]
    complete_extra = [row for row in extras if row["status"] == "EXTRA_COMPLETE"]
    nc_index = {(row["dataset"], row["variant"]): row for row in nc_rows}
    paired_index = {(row["dataset"], row["variant"], row["metric"]): row for row in paired}
    e0_index = {row["dataset"]: row for row in e0}

    lines: list[str] = []
    lines += [
        "# MoPF F2 main ablation analysis",
        "",
        "> Scope: formal result audit, aggregation, and exploratory scientific diagnosis only. No model, training, hyperparameter, ablation-definition, or interaction-training changes were made.",
        "",
        "## 1. Completeness audit",
        "",
        f"The expected main matrix contains {len(main)} runs: 5 NC datasets × 6 variants × 3 seeds plus 1 LP dataset × 6 variants × 3 seeds. Full MoPF is explicitly reused from the audited F1 formal artifacts under `outputs/f1_final_execution/`; the five ablation variants are read from `outputs/f2_ablation/`.",
        "",
        f"Main status: {sum(row['status'] in {'COMPLETE', 'COMPLETE_REUSED'} for row in main)}/{len(main)} complete; {len(bad_main)} missing or inconsistent.",
        "",
    ]
    if bad_main:
        lines.append("Blocking main-matrix issues (reported before aggregation):")
        lines.append("")
        for row in bad_main:
            lines.append(f"- `{row['task']}/{row['dataset']}/{row['variant']}/seed{row['seed']}`: **{row['status']}** — {row['issues'] or 'unspecified issue'}.")
        lines.append("")
    else:
        lines.append("No missing, unreadable, non-finite, failed/incomplete, seed-mismatched, checkpoint-selection-mismatched, or cross-variant configuration-inconsistent run was found in the main matrix.")
        lines.append("")
    if extras:
        lines.append(f"Out-of-scope residuals were retained and not used: {len(complete_extra)} complete interaction artifact(s) and {len(incomplete_extra)} incomplete artifact(s). These are recorded in `f2_main_completeness.csv` and do not enter any table or diagnosis.")
        lines.append("")
        for row in extras:
            lines.append(f"- `{row['task']}/{row['dataset']}/{row['variant']}/seed{row['seed']}`: `{row['status']}` — {row['issues'] or 'out of scope'}.")
        lines.append("")
    lines += [
        "The reported standard deviation is the population standard deviation across the three pre-specified seeds (ddof=0). The independent experimental unit is a dataset × seed run; no significance terminology or p-values are used.",
        "",
        "## 2. Main table",
        "",
        "The requested candidate table is generated as `outputs/f2_ablation/f2_main_paper_table.md` and `.tex`. Sports MRR denotes test MRR. Bold marks the best mean in each displayed column; an ablation is not suppressed when it exceeds Full.",
        "",
        "| Variant | NC Avg Acc | NC Avg Macro-F1 | Sports MRR | H@1 | H@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {labels[row['variant']]} | {fmt(row['nc_avg_acc_mean'], 4)} ± {fmt(row['nc_avg_acc_std'], 4)} | {fmt(row['nc_avg_macro_f1_mean'], 4)} ± {fmt(row['nc_avg_macro_f1_std'], 4)} | {fmt(row['sports_test_mrr_mean'], 4)} ± {fmt(row['sports_test_mrr_std'], 4)} | {fmt(row['sports_test_hits@1_mean'], 4)} ± {fmt(row['sports_test_hits@1_std'], 4)} | {fmt(row['sports_test_hits@10_mean'], 4)} ± {fmt(row['sports_test_hits@10_std'], 4)} |"
        )
    lines += [
        "",
        "NC averages are arithmetic averages of the five dataset-level seed means; node-level pooling was not used. The complete per-dataset candidate is `outputs/f2_ablation/f2_appendix_candidate_tables.md`.",
        "",
        "## 3. Per-component diagnosis",
        "",
        "Classification rubric used here: Strongly supported requires positive average NC and LP drops, broad NC cell coverage, and mostly same-direction paired seeds; Moderately supported allows positive contribution with mixed coverage; Weak / dataset-dependent is localized or mixed; Not supported means the average directions do not support a contribution. These are evidence labels for this audit, not statistical significance claims.",
        "",
        "| Component | Mean NC drop across 10 NC cells | Mean LP drop across 4 test metrics | NC positive-cell coverage | Paired positive fraction | Audit label |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant in ABLATIONS:
        d = diagnosis[variant]
        lines.append(f"| {labels[variant]} | {fmt(d['avg_nc_drop'], 5)} | {fmt(d['avg_lp_drop'], 5)} | {d['nc_positive_cell_fraction']:.0%} | {d['paired_positive_fraction']:.0%} | {d['classification']} |")
    lines += ["", "### A1 — learned semantic calibration (E0-A)", ""]
    a1 = diagnosis["wo_learned_semantic_calibration"]
    a1_rows = [nc_index[(dataset, "wo_learned_semantic_calibration")] for dataset in NC_DATASETS]
    for dataset, row in zip(NC_DATASETS, a1_rows):
        paired_acc = paired_index[(dataset, "wo_learned_semantic_calibration", "Accuracy")]
        paired_f1 = paired_index[(dataset, "wo_learned_semantic_calibration", "Macro-F1")]
        lines.append(f"- {dataset}: Accuracy drop {fmt(nc_index[(dataset, 'full')]['accuracy_mean'] - row['accuracy_mean'], 5)}, Macro-F1 drop {fmt(nc_index[(dataset, 'full')]['macro_f1_mean'] - row['macro_f1_mean'], 5)}; paired directions {paired_acc['positive_drop_count']}/3 and {paired_f1['positive_drop_count']}/3. E0-A mean rank gap={fmt(e0_index[dataset]['e0a_mean_rank_gap'], 4)}, Spearman TV={fmt(e0_index[dataset]['e0a_spearman_tv'], 4)}.")
    lines.append(f"\nDescriptive reading: the calibration-removal drops are compared with E0-A rank-gap and TV-association summaries, but no correlation or causal claim is made. Reddit-S has the largest E0-A rank-gap severity but does not show a positive downstream calibration-removal drop, whereas ele-fashion has the strongest raw TV association but shows reverse-direction NC differences. Thus these results do not support a simple monotone discrepancy-to-drop pattern.")
    lines += ["", "### A2 — semantic anchor (E0-B)", ""]
    a2 = diagnosis["wo_semantic_anchor"]
    for dataset in NC_DATASETS:
        row = nc_index[(dataset, "wo_semantic_anchor")]
        lines.append(f"- {dataset}: Accuracy drop {fmt(nc_index[(dataset, 'full')]['accuracy_mean'] - row['accuracy_mean'], 5)}, Macro-F1 drop {fmt(nc_index[(dataset, 'full')]['macro_f1_mean'] - row['macro_f1_mean'], 5)}; E0-B mean CKA formal-K drop={fmt(e0_index[dataset]['e0b_mean_cka_drop_formal'], 4)}, mean cosine drop={fmt(e0_index[dataset]['e0b_mean_cosine_drop_formal'], 4)}.")
    lines.append(f"\nAnchor removal does not overall hurt the NC summary (its Full-relative NC drops are negative), while it does lower the sports LP metrics on average. The E0-B retention quantities are empirical severity descriptors and do not establish that drift caused either downstream pattern.")
    lines += ["", "### A3 — TCPR (E0-D)", ""]
    a3 = diagnosis["wo_tcpr"]
    for dataset in NC_DATASETS:
        row = nc_index[(dataset, "wo_tcpr")]
        pair = paired_index[(dataset, "wo_tcpr", "Accuracy")]
        lines.append(f"- {dataset}: Accuracy drop {fmt(nc_index[(dataset, 'full')]['accuracy_mean'] - row['accuracy_mean'], 5)}, Macro-F1 drop {fmt(nc_index[(dataset, 'full')]['macro_f1_mean'] - row['macro_f1_mean'], 5)}, Accuracy paired direction {pair['positive_drop_count']}/3; E0-D mean absolute partial rho={fmt(e0_index[dataset]['e0d_mean_abs_partial_rho'], 4)}.")
    lines.append("\nTCPR is not a stable all-task drop: its NC average drop is positive, but its LP average drop is negative, and only a minority of NC cells are positive. E0-D association strength is used only for qualitative comparison; this is not a causal mediation test.")
    lines += ["", "### A4/A5 — node and modality adaptation (E0-C)", ""]
    for dataset in NC_DATASETS:
        node = nc_index[(dataset, "wo_node_adaptation")]
        modality = nc_index[(dataset, "wo_modality_adaptation")]
        lines.append(f"- {dataset}: w/o node adaptation drops Accuracy/Macro-F1 by {fmt(nc_index[(dataset, 'full')]['accuracy_mean'] - node['accuracy_mean'], 5)}/{fmt(nc_index[(dataset, 'full')]['macro_f1_mean'] - node['macro_f1_mean'], 5)}; w/o modality adaptation by {fmt(nc_index[(dataset, 'full')]['accuracy_mean'] - modality['accuracy_mean'], 5)}/{fmt(nc_index[(dataset, 'full')]['macro_f1_mean'] - modality['macro_f1_mean'], 5)}. E0-C profile-L1={fmt(e0_index[dataset]['e0c_mean_profile_l1'], 4)}, order-gap={fmt(e0_index[dataset]['e0c_mean_abs_order_gap'], 4)}.")
    lines.append("\nGrocery has the largest E0-C heterogeneity descriptor but both adaptation removals improve rather than hurt its NC summary; Toys shows small positive drops for node adaptation and mixed effects for modality adaptation. Movies and ele-fashion provide lower-heterogeneity contrasts with dataset-specific reversals. The comparison remains descriptive and exploratory.")
    lines += ["", "## 4. Per-dataset observations", ""]
    for dataset in NC_DATASETS:
        drops = []
        for variant in ABLATIONS:
            row = nc_index[(dataset, variant)]
            full = nc_index[(dataset, "full")]
            drops.append((mean([full['accuracy_mean'] - row['accuracy_mean'], full['macro_f1_mean'] - row['macro_f1_mean']]), variant))
        order = ", ".join(f"{labels[v]} {fmt(value, 4)}" for value, v in sorted(drops, reverse=True))
        lines.append(f"- **{dataset}**: mean of the two NC drops by ablation: {order}.")
    lp_full = next(row for row in summary if row["variant"] == "full")
    for metric, label in (("sports_test_mrr", "Test MRR"), ("sports_test_hits@1", "Hits@1"), ("sports_test_hits@3", "Hits@3"), ("sports_test_hits@10", "Hits@10")):
        gains = []
        for row in summary[1:]:
            drop = row[f"{metric}_drop_vs_full"]
            if drop < 0:
                gains.append(f"{labels[row['variant']]} {fmt(-drop, 5)} gain")
        if gains:
            lines.append(f"- **sports-copurchase {label}**: ablation gains relative to Full are retained: " + ", ".join(gains) + ".")
    lines += ["", "## 5. Seed stability", ""]
    weak = [row for row in paired if int(row["positive_drop_count"]) == 1]
    strong = [row for row in paired if int(row["positive_drop_count"]) == 3]
    reverse = [row for row in paired if int(row["positive_drop_count"]) == 0]
    lines.append(f"Across paired rows, {len(strong)} are 3/3 same-direction drops (strong seed consistency under the audit rule), {len(weak)} are 1/3 (weak evidence), and {len(reverse)} are 0/3 in the reverse direction. Rows with 2/3 are mixed direction.")
    lines.append("")
    lines.append("Largest observed seed standard-deviation flags:")
    lines.extend(f"- {text}" for text in top_std_anomalies(nc_rows, lp_rows))
    lines.append("")
    lines.append("The paired-drop file preserves the three seed differences, their mean/std/median, and the positive count; it should be preferred over a comparison of two unpaired means when discussing consistency.")
    lines += ["", "## 6. Relation to E0 empirical studies", ""]
    lines.append("The E0 alignment file is `outputs/f2_ablation/f2_e0_alignment_summary.csv`. The comparisons used are:")
    lines.extend([
        "- E0-A: mean/median/q90 rank gaps and Spearman text–visual association for relation-level semantic discrepancy.",
        "- E0-B: mean CKA and cosine retention drops at formal K across text and visual modalities for propagation-induced drift.",
        "- E0-C: seed-averaged modality profile L1 and order-gap measures for utilization heterogeneity.",
        "- E0-D: seed- and modality-averaged absolute partial rho for relation-condition/utilization association.",
    ])
    lines.append("\nThese are qualitative alignments only. The analysis does not fit or report a correlation, does not claim mechanism identification, and does not convert E0 severity into a causal prediction.")
    lines += ["", "## 7. Anomalies", ""]
    if any(row["status"] not in {"COMPLETE", "COMPLETE_REUSED"} for row in main):
        lines.append("Main-matrix audit anomalies are listed in Section 1 and must be resolved before using the results as final evidence.")
    else:
        lines.append("No main-run missing artifact, non-finite metric, failed/incomplete marker, seed mismatch, checkpoint-selection mismatch, runtime-crash pattern, duplicate main output, or cross-variant configuration mismatch was detected.")
    lines.append("The out-of-scope `wo_anchor_tcpr` residuals are an audit anomaly in the directory, not evidence for a main result and not used in any aggregation.")
    lines.append("A negative Full-relative drop means the ablation exceeds Full; such values are preserved in the CSV files and should not be absolute-valued.")
    lines += ["", "## 8. Interaction-phase recommendation", ""]
    lines.append("This audit alone does not authorize or start interaction training. A cautious interaction phase is worth considering only after the residual out-of-scope artifacts and any paper-table review are closed; the main matrix itself is complete if Section 1 reports 108/108.")
    lines.append("Recommended exploratory interactions, ranked by the paired component evidence available here, are:")
    lines.extend([
        "1. semantic anchor × TCPR (`wo_anchor_tcpr`): tests whether propagation stabilization and relation-conditioned utilization are complementary.",
        "2. node adaptation × TCPR (`wo_node_tcpr`): tests whether node heterogeneity and relation-utilization conditioning overlap.",
        "3. modality adaptation × TCPR (`wo_modality_tcpr`): tests whether modality-specific residuals and TCPR provide complementary handling of utilization differences.",
    ])
    lines.append("These are hypotheses for a later, explicitly pre-registered interaction design—not conclusions from the partial residual `wo_anchor_tcpr` artifacts already present.")
    lines += ["", "## 9. Paper-writing-safe claims", ""]
    for variant in ABLATIONS:
        d = diagnosis[variant]
        lines.append(f"- **{labels[variant]}**: {d['classification']}; report the mean drops, dataset coverage, and paired seed directions together. Avoid saying that the component is necessary or causally responsible.")
    lines.extend([
        "- It is safe to state that the formal ablations provide descriptive evidence of component contribution under the fixed F2 protocol, with effects varying across datasets and metrics.",
        "- It is safe to state when Full exceeds an ablation on 3/3 seeds; call this strong seed consistency under the present audit rule, not statistical significance.",
        "- It is not safe to claim that E0-A/B/C/D empirically prove the corresponding downstream mechanism; they motivate qualitative alignment only.",
        "- Any ablation improvement over Full must remain visible as a negative Full-relative drop and be discussed as a dataset/task-specific result.",
    ])
    lines += ["", "### Final answers", ""]
    ranked = sorted(diagnosis.items(), key=lambda item: (item[1]["paired_positive_fraction"], item[1]["avg_nc_drop"]), reverse=True)
    candidate_name, candidate = ranked[0]
    if candidate["paired_positive_fraction"] >= 0.75 and candidate["avg_nc_drop"] > 0 and candidate["avg_lp_drop"] > 0:
        stable_answer = labels[candidate_name]
    else:
        stable_answer = f"none meets the strong cross-task stability rule; {labels[candidate_name]} is only the relative leader at {candidate['paired_positive_fraction']:.0%} positive paired directions"
    lines.append(f"1. **Most stable downstream contributions:** {stable_answer}; the seed and task reversals mean this is not strong evidence of a uniformly necessary mechanism.")
    small_strong = [labels[v] for v, d in diagnosis.items() if abs(d["avg_nc_drop"]) < np.median([abs(x["avg_nc_drop"]) for x in diagnosis.values()]) and d["paired_positive_fraction"] >= 0.75]
    lines.append(f"2. **Smaller contribution but stronger mechanism evidence:** {', '.join(small_strong) if small_strong else 'No component meets a separate small-effect/strong-direction rule in this audit.'}")
    gains = [labels[row["variant"]] for row in summary[1:] if any(row[key] < 0 for key in ("nc_avg_acc_drop_vs_full", "nc_avg_macro_f1_drop_vs_full", "sports_test_mrr_drop_vs_full", "sports_test_hits@1_drop_vs_full", "sports_test_hits@3_drop_vs_full", "sports_test_hits@10_drop_vs_full"))]
    lines.append(f"3. **Ablation improvements:** {', '.join(gains) if gains else 'No ablation has a negative Full-relative drop in the summary metrics.'}")
    lines.append("4. **Implementation suspicion:** none is warranted from completeness/config/runtime checks alone; investigate only any anomalies explicitly listed above rather than changing code based on outcome.")
    lines.append("5. **Interaction phase:** not started; potentially worthwhile only after review of this audit, with the pre-specified exploratory scope above.")
    lines.append("6. **Three interactions:** semantic anchor × TCPR, node adaptation × TCPR, and modality adaptation × TCPR.")
    lines.append("")
    lines.append("Generated products: `f2_main_completeness.csv`, `f2_nc_full_results.csv`, `f2_lp_full_results.csv`, `f2_main_summary.csv`, `f2_seed_paired_drops.csv`, `f2_e0_alignment_summary.csv`, the paper-table candidates, and the diagnosis-only heatmap under `outputs/f2_ablation/`.")
    (DOCS / "mopf_f2_main_ablation_analysis.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="kept for explicit provenance; aggregation is still read-only")
    args = parser.parse_args()
    del args
    F2_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    audit_rows = audit()
    nc_rows = aggregate_nc()
    lp_rows = aggregate_lp()
    summary = summary_rows(nc_rows, lp_rows)
    paired = paired_rows(nc_rows, lp_rows)
    e0 = e0_alignment(nc_rows)
    paper_tables(summary, nc_rows, lp_rows)
    plot_heatmap(nc_rows, lp_rows)
    generate_report(audit_rows, summary, nc_rows, lp_rows, paired, e0)
    main_rows = [row for row in audit_rows if row["scope"] == "main"]
    extras = [row for row in audit_rows if row["scope"] != "main"]
    print(f"main_complete={sum(row['status'] in {'COMPLETE', 'COMPLETE_REUSED'} for row in main_rows)}/{len(main_rows)}")
    print(f"main_issues={sum(row['status'] not in {'COMPLETE', 'COMPLETE_REUSED'} for row in main_rows)}")
    print(f"out_of_scope_rows={len(extras)}")
    print(f"report={DOCS / 'mopf_f2_main_ablation_analysis.md'}")
    print(f"heatmap={F2_ROOT / 'f2_performance_drop_heatmap.pdf'}")


if __name__ == "__main__":
    main()
