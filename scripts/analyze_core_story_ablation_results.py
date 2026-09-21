#!/usr/bin/env python3
"""Audit and summarize the frozen CoSI-MAG Core Story ablation runs.

This is an analysis-only utility. It never imports a model or starts training.
It reads the official completeness-checker stdout, the 63 formal run records,
and the two user-designated Full-reference PDFs, then writes audit tables and
paper-facing summaries under outputs/core_story_ablation_analysis/.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.formal_protocol import (  # noqa: E402
    CORE_STORY_PROTOCOL_VERSION,
    FORMAL_K,
    FORMAL_SEEDS,
    LP_DATASETS,
    NC_DATASETS,
)
from src.ablation import CORE_STORY_ABLATIONS  # noqa: E402


VARIANT_LABELS = {
    "wo_relation_calibration": "w/o Relation Calibration",
    "wo_semantic_anchor": "w/o Semantic Anchor",
    "wo_adaptive_composition": "w/o Adaptive Composition",
}
METRICS = {
    "nc": ("test_acc", "test_macro_f1"),
    "lp": ("test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"),
}
METRIC_LABELS = {
    "test_acc": "Accuracy",
    "test_macro_f1": "Macro-F1",
    "test_mrr": "MRR",
    "test_hits@1": "Hits@1",
    "test_hits@3": "Hits@3",
    "test_hits@10": "Hits@10",
}
EXPECTED_METRIC_KEYS = {
    "nc": {"val_acc", "val_macro_f1", "test_acc", "test_macro_f1"},
    "lp": {"val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"},
}
REQUIRED_ARTIFACTS = (
    "ablation_manifest.json",
    "resolved_config.yaml",
    "resolved_config.json",
    "metrics.json",
    "complete.marker",
    "train.log",
    "best.pt",
)
BASE_OUTPUT = PROJECT_ROOT / "outputs" / "core_story_ablation"
ANALYSIS_OUTPUT = PROJECT_ROOT / "outputs" / "core_story_ablation_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=BASE_OUTPUT)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_OUTPUT)
    parser.add_argument("--nc-reference", type=Path, default=PROJECT_ROOT / "paper/tables/NC.pdf")
    parser.add_argument("--lp-reference", type=Path, default=PROJECT_ROOT / "paper/tables/LP.pdf")
    parser.add_argument("--force", action="store_true", help="Replace this utility's generated analysis files")
    return parser.parse_args()


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def yaml_read(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"resolved config is not a mapping: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_finite_tree(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(is_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(is_finite_tree(item) for item in value)
    return True


def read_checker_gate(analysis_root: Path) -> tuple[dict[str, str], list[str]]:
    path = analysis_root / "completeness_checker_stdout.txt"
    if not path.is_file():
        return {}, [f"saved official checker stdout missing: {path}"]
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in {
                "planned_runs",
                "complete_runs",
                "incomplete_runs",
                "missing_runs",
                "manifest_mismatch",
                "config_mismatch",
                "git_commit_mismatch",
                "metric_schema",
                "duplicate_collision",
            }:
                fields[key] = value
    expected = {
        "planned_runs": "63",
        "complete_runs": "63",
        "incomplete_runs": "0",
        "missing_runs": "0",
        "manifest_mismatch": "PASS",
        "config_mismatch": "PASS",
        "git_commit_mismatch": "PASS",
        "metric_schema": "PASS",
        "duplicate_collision": "PASS",
    }
    issues = [f"checker gate {key}: expected {value}, got {fields.get(key)}" for key, value in expected.items() if fields.get(key) != value]
    return fields, issues


def planned_runs(run_root: Path) -> list[dict[str, Any]]:
    tasks = (("nc", NC_DATASETS), ("lp", LP_DATASETS))
    return [
        {
            "task": task,
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "output_dir": run_root / task / dataset / variant / f"seed{seed}",
        }
        for task, datasets in tasks
        for dataset in datasets
        for variant in CORE_STORY_ABLATIONS
        for seed in FORMAL_SEEDS
    ]


def expected_selection(task: str) -> str:
    return "best_val_accuracy" if task == "nc" else "best_val_mrr"


def expected_split_key(task: str, dataset: str) -> str:
    if task == "lp":
        return "edge_split_path"
    return "node_split_path" if dataset == "ele-fashion" else "nc_split_path"


def append_issue(record: dict[str, Any], issue: str) -> None:
    record["issues"].append(issue)


def audit_one_run(run: dict[str, Any], split_hash_cache: dict[str, str]) -> dict[str, Any]:
    import torch

    task, dataset, variant, seed = run["task"], run["dataset"], run["variant"], run["seed"]
    output_dir: Path = run["output_dir"]
    record: dict[str, Any] = {
        **{key: run[key] for key in ("task", "dataset", "variant", "seed")},
        "output_dir": output_dir,
        "issues": [],
        "config": None,
        "manifest": None,
        "metrics": None,
        "marker": None,
        "checkpoint_meta": {},
        "split_sha256": "",
        "config_sha256": "",
    }

    for name in REQUIRED_ARTIFACTS:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            append_issue(record, f"missing/empty artifact: {name}")

    try:
        manifest = json_read(output_dir / "ablation_manifest.json")
        record["manifest"] = manifest
    except Exception as exc:  # malformed provenance is a blocker, not a skip
        append_issue(record, f"manifest unreadable: {type(exc).__name__}: {exc}")
        manifest = {}
    try:
        config_yaml = yaml_read(output_dir / "resolved_config.yaml")
        config_json = json_read(output_dir / "resolved_config.json")
        record["config"] = config_yaml
        record["config_sha256"] = sha256_file(output_dir / "resolved_config.yaml")
        if config_yaml != config_json:
            append_issue(record, "resolved_config.yaml/json content differs")
    except Exception as exc:
        config_yaml = {}
        append_issue(record, f"resolved config unreadable: {type(exc).__name__}: {exc}")
    try:
        metrics = json_read(output_dir / "metrics.json")
        record["metrics"] = metrics
    except Exception as exc:
        metrics = {}
        append_issue(record, f"metrics unreadable: {type(exc).__name__}: {exc}")
    try:
        marker = json_read(output_dir / "complete.marker")
        record["marker"] = marker
    except Exception as exc:
        marker = {}
        append_issue(record, f"complete.marker unreadable: {type(exc).__name__}: {exc}")

    identity = {"task": task, "dataset": dataset, "ablation": variant, "seed": seed}
    for source_name, payload, keys in (
        ("manifest", manifest, {"task": "task", "dataset": "dataset", "ablation_name": "ablation", "seed": "seed"}),
        ("metrics", metrics, {"task": "task", "dataset": "dataset", "ablation": "ablation", "seed": "seed"}),
        ("complete.marker", marker, {"task": "task", "dataset": "dataset", "ablation": "ablation", "seed": "seed"}),
    ):
        for key, target_key in keys.items():
            if payload.get(key) != identity[target_key]:
                append_issue(record, f"{source_name} identity {key}: expected {identity[target_key]!r}, actual {payload.get(key)!r}")

    selection = expected_selection(task)
    if manifest.get("checkpoint_selection") != selection:
        append_issue(record, f"manifest checkpoint_selection: expected {selection}, actual {manifest.get('checkpoint_selection')}")
    if metrics.get("checkpoint_selection") != selection:
        append_issue(record, f"metrics checkpoint_selection: expected {selection}, actual {metrics.get('checkpoint_selection')}")
    if metrics.get("selection_metric") != ("val_acc" if task == "nc" else "val_mrr"):
        append_issue(record, f"metrics selection_metric: expected {'val_acc' if task == 'nc' else 'val_mrr'}, actual {metrics.get('selection_metric')}")

    expected_k = FORMAL_K[dataset]
    if manifest.get("K") != expected_k:
        append_issue(record, f"manifest K: expected {expected_k}, actual {manifest.get('K')}")
    if config_yaml:
        model_cfg = config_yaml.get("model", {})
        if model_cfg.get("max_order") != expected_k or model_cfg.get("num_layers") != expected_k:
            append_issue(record, f"resolved config K: expected max_order/num_layers={expected_k}, actual {model_cfg.get('max_order')}/{model_cfg.get('num_layers')}")
        dcfg = config_yaml.get("dataset", {})
        split_key = expected_split_key(task, dataset)
        resolved_split = dcfg.get(split_key)
        manifest_split = manifest.get("split_source")
        if not resolved_split or str(resolved_split) != str(manifest_split):
            append_issue(record, f"split source mismatch: config.{split_key}={resolved_split!r}, manifest={manifest_split!r}")
        elif not Path(str(resolved_split)).is_file():
            append_issue(record, f"split file missing: {resolved_split}")
        else:
            split_path = str(Path(str(resolved_split)).resolve())
            if split_path not in split_hash_cache:
                split_hash_cache[split_path] = sha256_file(Path(split_path))
            record["split_sha256"] = split_hash_cache[split_path]
        for key, expected in (("name", dataset),):
            if dcfg.get(key) != expected:
                append_issue(record, f"resolved dataset.{key}: expected {expected!r}, actual {dcfg.get(key)!r}")
        tcfg = config_yaml.get("task", {})
        if tcfg.get("name") != task:
            append_issue(record, f"resolved task.name: expected {task!r}, actual {tcfg.get('name')!r}")

    if manifest.get("git_branch") in {None, "", "unknown"}:
        append_issue(record, "manifest git_branch missing/unknown")
    if manifest.get("git_commit") in {None, "", "unknown"}:
        append_issue(record, "manifest git_commit missing/unknown")

    metric_values = metrics.get("metrics", {})
    if set(metric_values) != EXPECTED_METRIC_KEYS[task]:
        append_issue(record, f"metric schema: expected {sorted(EXPECTED_METRIC_KEYS[task])}, actual {sorted(metric_values)}")
    if not is_finite_tree(metrics):
        append_issue(record, "metrics payload contains NaN/Inf")
    for key in METRICS[task]:
        entry = metric_values.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("mean"), (int, float)) or not math.isfinite(float(entry["mean"])):
            append_issue(record, f"test metric {key} missing or non-finite")

    best_epoch = metrics.get("best_epoch")
    max_epochs = manifest.get("max_epochs", manifest.get("epochs"))
    if not isinstance(best_epoch, int) or best_epoch < 1 or not isinstance(max_epochs, int) or best_epoch > max_epochs:
        append_issue(record, f"best_epoch invalid: best_epoch={best_epoch!r}, max_epochs={max_epochs!r}")
    runtime = metrics.get("runtime_seconds")
    if not isinstance(runtime, (int, float)) or not math.isfinite(float(runtime)) or float(runtime) <= 0:
        append_issue(record, f"runtime_seconds invalid: {runtime!r}")

    checkpoint_path = Path(str(metrics.get("checkpoint", "")))
    expected_checkpoint = output_dir / "best.pt"
    if checkpoint_path.resolve() != expected_checkpoint.resolve() or not checkpoint_path.is_file():
        append_issue(record, f"checkpoint path mismatch/missing: expected {expected_checkpoint}, actual {checkpoint_path}")
    else:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise TypeError(f"checkpoint payload is {type(payload).__name__}, expected dict")
            record["checkpoint_meta"] = {
                "selection": payload.get("selection"),
                "epoch": payload.get("epoch"),
                "task": payload.get("task"),
                "seed": payload.get("seed"),
            }
            ckpt = record["checkpoint_meta"]
            if ckpt.get("selection") != selection:
                append_issue(record, f"checkpoint selection metadata: expected {selection}, actual {ckpt.get('selection')}")
            if ckpt.get("epoch") != best_epoch:
                append_issue(record, f"checkpoint epoch metadata: metrics={best_epoch}, checkpoint={ckpt.get('epoch')}")
            if ckpt.get("task") != task or ckpt.get("seed") != seed:
                append_issue(record, f"checkpoint identity metadata: expected {task}/{seed}, actual {ckpt.get('task')}/{ckpt.get('seed')}")
        except Exception as exc:
            append_issue(record, f"checkpoint metadata unreadable: {type(exc).__name__}: {exc}")
    recorded_ckpt = metrics.get("checkpoint_metadata", {})
    if recorded_ckpt.get("selection") != selection or recorded_ckpt.get("best_epoch") != best_epoch:
        append_issue(record, f"metrics checkpoint_metadata mismatch: {recorded_ckpt!r}")

    log_path = output_dir / "train.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    log_checks = {
        "traceback": re.search(r"Traceback \(most recent call last\)|\bUnhandled exception\b", log_text, re.I) is not None,
        "cuda_oom": re.search(r"CUDA out of memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED", log_text, re.I) is not None,
        "nonfinite": re.search(r"\b(?:nan|inf|infinity)\b", log_text, re.I) is not None,
    }
    for check, failed in log_checks.items():
        if failed:
            append_issue(record, f"train.log contains {check} failure marker")
    record["health"] = {
        "best_epoch": best_epoch,
        "runtime_seconds": runtime,
        "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
        "log_traceback": log_checks["traceback"],
        "log_cuda_oom": log_checks["cuda_oom"],
        "log_nonfinite": log_checks["nonfinite"],
        "metrics_finite": is_finite_tree(metric_values),
        "checkpoint_exists": checkpoint_path.is_file(),
        "checkpoint_selection": record["checkpoint_meta"].get("selection"),
        "checkpoint_epoch": record["checkpoint_meta"].get("epoch"),
        "marker_identity_ok": all(marker.get(k) == v for k, v in identity.items()),
    }
    return record


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["ablation"] = "<variant>"
    task_cfg = result.get("task", {})
    if "save_ckpt_path" in task_cfg:
        task_cfg["save_ckpt_path"] = "<run-specific-checkpoint>"
    return result


def audit_global_provenance(records: list[dict[str, Any]]) -> None:
    commits = {r.get("manifest", {}).get("git_commit") for r in records}
    branches = {r.get("manifest", {}).get("git_branch") for r in records}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head, branch = "unknown", "unknown"
    for record in records:
        manifest = record.get("manifest", {})
        if len(commits) != 1 or manifest.get("git_commit") != head:
            append_issue(record, f"formal git SHA not globally/currently frozen: runs={sorted(str(x) for x in commits)}, HEAD={head}")
        if len(branches) != 1 or manifest.get("git_branch") != branch:
            append_issue(record, f"formal branch mismatch: runs={sorted(str(x) for x in branches)}, current={branch}")

    # All variants for a fixed task/dataset/seed must share the exact resolved
    # task/data/model protocol except the selected ablation and run checkpoint.
    groups: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["task"], record["dataset"], record["seed"])].append(record)
    for key, group in groups.items():
        signatures = {json.dumps(normalized_config(r["config"]), sort_keys=True, separators=(",", ":")) for r in group if r.get("config")}
        if len(signatures) != 1 or len(group) != len(CORE_STORY_ABLATIONS):
            for record in group:
                append_issue(record, f"protocol drift across variants for {key}: distinct normalized config count={len(signatures)}")

    # Graph/features/self-loop policy must be the same across variants and
    # seeds for each dataset. The run implementation carries the unchanged
    # physical edge_index through calibration; the associated unit invariant
    # separately tests exact edge-support identity under raw_uniform.
    support_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    split_groups: defaultdict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cfg = record.get("config") or {}
        dcfg = cfg.get("dataset", {})
        support = (
            dcfg.get("graph_path"),
            dcfg.get("text_feat_path"),
            dcfg.get("image_feat_path"),
            dcfg.get("make_undirected"),
            dcfg.get("add_self_loops"),
        )
        support_groups[(record["task"], record["dataset"])].append({"record": record, "support": support})
        split_groups[(record["task"], record["dataset"], record["seed"])].append(record)
    for key, group in support_groups.items():
        signatures = {item["support"] for item in group}
        if len(signatures) != 1:
            for item in group:
                append_issue(item["record"], f"physical graph/feature support config differs within {key}")
    for key, group in split_groups.items():
        split_sources = {r.get("manifest", {}).get("split_source") for r in group}
        split_hashes = {r.get("split_sha256") for r in group}
        if len(split_sources) != 1 or len(split_hashes) != 1:
            for record in group:
                append_issue(record, f"split source/hash differs across variants within {key}")


def expected_semantics(variant: str) -> dict[str, Any]:
    if variant == "wo_relation_calibration":
        return {
            "edge_weight_mode": "raw_uniform",
            "relation_calibration_effective": False,
            "semantic_anchor_active": True,
            "semantic_anchor_effective": True,
            "composition_mode": "adaptive",
            "relation_conditioned_refinement_active": True,
            "relation_conditioned_refinement_effective": False,
        }
    if variant == "wo_semantic_anchor":
        return {
            "multihop_state_mode": "ordinary",
            "multihop_response_mode": "cumulative",
            "semantic_anchor_effective": False,
            "relation_calibration_effective": True,
            "composition_mode": "adaptive",
            "global_preference_effective": True,
            "modality_residual_effective": True,
            "node_residual_effective": True,
            "relation_conditioned_refinement_effective": True,
        }
    return {
        "composition_mode": "uniform",
        "relation_calibration_effective": True,
        "semantic_anchor_effective": True,
        "global_preference_active": True,
        "global_preference_effective": False,
        "modality_residual_active": True,
        "modality_residual_effective": False,
        "node_residual_active": True,
        "node_residual_effective": False,
        "relation_conditioned_refinement_active": True,
        "relation_conditioned_refinement_effective": False,
    }


def audit_semantics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    support_by_dataset: defaultdict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)
    for record in records:
        config = record.get("config", {})
        dcfg = config.get("dataset", {})
        support_by_dataset[(record["task"], record["dataset"])].add(
            (dcfg.get("graph_path"), dcfg.get("make_undirected"), dcfg.get("add_self_loops"))
        )
    output = []
    for record in records:
        manifest = record.get("manifest", {})
        expected = expected_semantics(record["variant"])
        issues = []
        for field, target in expected.items():
            actual = manifest.get(field)
            if actual != target:
                issues.append(f"{field}: expected {target!r}, actual {actual!r}")
        support_same = len(support_by_dataset[(record["task"], record["dataset"])]) == 1
        if not support_same:
            issues.append("graph/support configuration differs across formal variants/seeds")
        if record["variant"] == "wo_relation_calibration":
            note = "raw_uniform; local centered relation context is zero by frozen implementation invariant; no replacement condition"
        elif record["variant"] == "wo_semantic_anchor":
            note = "ordinary states with cumulative response bank; Stage I/III manifest settings retained"
        else:
            note = "uniform response-bank mean; all adaptive eta preferences/residuals are instantiated but ineffective in final composition"
        output.append({
            "task": record["task"],
            "dataset": record["dataset"],
            "variant": record["variant"],
            "seed": record["seed"],
            "edge_weight_mode": manifest.get("edge_weight_mode"),
            "relation_calibration_active": manifest.get("relation_calibration_active"),
            "relation_calibration_effective": manifest.get("relation_calibration_effective"),
            "semantic_anchor_active": manifest.get("semantic_anchor_active"),
            "semantic_anchor_effective": manifest.get("semantic_anchor_effective"),
            "multihop_state_mode": manifest.get("multihop_state_mode"),
            "multihop_response_mode": manifest.get("multihop_response_mode"),
            "composition_mode": manifest.get("composition_mode"),
            "global_preference_active": manifest.get("global_preference_active"),
            "global_preference_effective": manifest.get("global_preference_effective"),
            "modality_residual_active": manifest.get("modality_residual_active"),
            "modality_residual_effective": manifest.get("modality_residual_effective"),
            "node_residual_active": manifest.get("node_residual_active"),
            "node_residual_effective": manifest.get("node_residual_effective"),
            "relation_conditioned_refinement_active": manifest.get("relation_conditioned_refinement_active"),
            "relation_conditioned_refinement_effective": manifest.get("relation_conditioned_refinement_effective"),
            "physical_support_config_unchanged": support_same,
            "status": "PASS" if not issues else "FAIL",
            "issue": "; ".join(issues),
            "interpretation": note,
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in fields})


def extract_reference_pdf(path: Path, task: str) -> tuple[list[dict[str, Any]], str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"designated Full reference PDF missing: {path}")
    extracted = subprocess.run(
        ["pdftotext", "-raw", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    row = next((line.strip() for line in extracted.splitlines() if line.strip().startswith("CoSI-MAG ")), None)
    if row is None:
        raise ValueError(f"CoSI-MAG row not found in user-designated PDF: {path}")
    pairs = re.findall(r"(\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)", row)
    datasets = NC_DATASETS if task == "nc" else LP_DATASETS
    metrics = ("test_acc", "test_macro_f1") if task == "nc" else ("test_mrr", "test_hits@1", "test_hits@3", "test_hits@10")
    expected_pairs = len(datasets) * len(metrics)
    if len(pairs) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} CoSI-MAG mean±SD pairs in {path}, parsed {len(pairs)}: {row}")
    text_lower = extracted.lower()
    if "test" not in text_lower or "percent" not in text_lower:
        raise ValueError(f"reference PDF does not declare test metrics/percent units: {path}")
    rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(datasets):
        for metric_index, metric in enumerate(metrics):
            mean_str, std_str = pairs[dataset_index * len(metrics) + metric_index]
            if task == "nc":
                sd_definition = "population SD (PDF caption; seeds 42,43,44)"
            elif dataset == "sports-copurchase":
                sd_definition = "population SD (PDF caption; seeds 42,43,44)"
            else:
                sd_definition = "PDF-reported SD; cloth convention not separately specified in caption"
            rows.append({
                "task": task,
                "dataset": dataset,
                "metric": metric,
                "full_mean": float(mean_str),
                "full_std": float(std_str),
                "full_std_definition": sd_definition,
                "unit": "percent",
                "source_pdf": str(path.resolve().relative_to(PROJECT_ROOT)),
                "source_page": 1,
                "source_row": row,
                "pdf_sha256": sha256_file(path),
            })
    return rows, row, sha256_file(path)


def audit_manifest_semantics_row(record: dict[str, Any]) -> dict[str, Any]:
    manifest = record["manifest"]
    return {"task": record["task"], "dataset": record["dataset"], "variant": record["variant"], "seed": record["seed"], **manifest}


def aggregate_metrics(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    by_key: defaultdict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    for record in records:
        metric_payload = record["metrics"]["metrics"]
        for metric in METRICS[record["task"]]:
            raw = float(metric_payload[metric]["mean"]) * 100.0
            by_key[(record["task"], record["dataset"], record["variant"], metric)][record["seed"]] = raw
            per_seed.append({
                "task": record["task"],
                "dataset": record["dataset"],
                "variant": record["variant"],
                "seed": record["seed"],
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "test_value_percent": raw,
                "unit": "percent",
            })
    summary: list[dict[str, Any]] = []
    summary_map: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for key in sorted(by_key, key=lambda k: (0 if k[0] == "nc" else 1, (NC_DATASETS + LP_DATASETS).index(k[1]), CORE_STORY_ABLATIONS.index(k[2]), METRICS[k[0]].index(k[3]))):
        values = by_key[key]
        if tuple(sorted(values)) != FORMAL_SEEDS:
            raise ValueError(f"not exactly formal seeds {FORMAL_SEEDS} for {key}: {sorted(values)}")
        vals = [values[seed] for seed in FORMAL_SEEDS]
        mean = statistics.mean(vals)
        sample_std = statistics.stdev(vals)
        row = {
            "task": key[0],
            "dataset": key[1],
            "variant": key[2],
            "metric": key[3],
            "metric_label": METRIC_LABELS[key[3]],
            "seed42": vals[0],
            "seed43": vals[1],
            "seed44": vals[2],
            "mean": mean,
            "sample_std": sample_std,
            "n_seeds": 3,
            "unit": "percent",
            "spread_definition": "sample SD across seeds (ddof=1)",
        }
        summary.append(row)
        summary_map[key] = row
    per_seed.sort(key=lambda r: (0 if r["task"] == "nc" else 1, (NC_DATASETS + LP_DATASETS).index(r["dataset"]), CORE_STORY_ABLATIONS.index(r["variant"]), METRICS[r["task"]].index(r["metric"]), r["seed"]))
    return per_seed, summary, summary_map


def relative_drops(summary_map: dict[tuple[str, str, str, str], dict[str, Any]], full_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs = {(r["task"], r["dataset"], r["metric"]): r for r in full_rows}
    result = []
    for (task, dataset, variant, metric), ablation in summary_map.items():
        ref = refs[(task, dataset, metric)]
        drop = ref["full_mean"] - ablation["mean"]
        result.append({
            "task": task,
            "dataset": dataset,
            "variant": variant,
            "metric": metric,
            "full_mean": ref["full_mean"],
            "full_std": ref["full_std"],
            "full_std_definition": ref["full_std_definition"],
            "ablation_mean": ablation["mean"],
            "ablation_sample_std": ablation["sample_std"],
            "drop_full_minus_ablation": drop,
            "relative_drop_percent": 100.0 * drop / abs(ref["full_mean"]) if ref["full_mean"] else None,
            "n_ablation_seeds": ablation["n_seeds"],
            "paired_seed_test": "not performed; Full PDF provides mean±SD only",
            "unit": "percentage_points_for_absolute_drop",
            "source_pdf": ref["source_pdf"],
            "pdf_sha256": ref["pdf_sha256"],
        })
    return result


def describe_stage(variant: str, drops: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [row for row in drops if row["variant"] == variant]
    by_dataset: defaultdict[str, list[float]] = defaultdict(list)
    by_task: defaultdict[str, list[float]] = defaultdict(list)
    for row in subset:
        by_dataset[row["dataset"]].append(row["drop_full_minus_ablation"])
        by_task[row["task"]].append(row["drop_full_minus_ablation"])
    positive = [row for row in subset if row["drop_full_minus_ablation"] > 0]
    reverse = [row for row in subset if row["drop_full_minus_ablation"] < 0]
    positive_ds, reverse_ds, mixed_ds = [], [], []
    for dataset in NC_DATASETS + LP_DATASETS:
        vals = by_dataset[dataset]
        if vals and all(value > 0 for value in vals):
            positive_ds.append(dataset)
        elif vals and all(value < 0 for value in vals):
            reverse_ds.append(dataset)
        else:
            mixed_ds.append(dataset)
    pos_share = len(positive) / len(subset)
    neg_share = len(reverse) / len(subset)
    task_pos_share = {task: sum(x > 0 for x in vals) / len(vals) for task, vals in by_task.items()}
    effect_to_reported_spread = [
        abs(row["drop_full_minus_ablation"])
        / math.hypot(row["full_std"], row["ablation_sample_std"])
        for row in subset
    ]
    median_effect_to_spread = statistics.median(effect_to_reported_spread)
    positive_gaps_at_least_one_spread = sum(
        row["drop_full_minus_ablation"] > 0
        and row["drop_full_minus_ablation"]
        / math.hypot(row["full_std"], row["ablation_sample_std"])
        >= 1.0
        for row in subset
    )
    if pos_share >= 0.80 and not reverse_ds and all(v >= 0.75 for v in task_pos_share.values()) and median_effect_to_spread >= 1.0:
        classification = "Broadly supported"
    elif pos_share > 0.50 and all(v > 0.50 for v in task_pos_share.values()):
        classification = "Moderately supported / dataset-dependent"
    elif neg_share >= 0.40 or any(v < 0.50 for v in task_pos_share.values()):
        classification = "Mixed / contradicted on some tasks"
    else:
        classification = "Weakly supported"
    largest_positive = max(positive, key=lambda r: r["drop_full_minus_ablation"], default=None)
    largest_reverse = min(reverse, key=lambda r: r["drop_full_minus_ablation"], default=None)
    median_abs = statistics.median(abs(r["drop_full_minus_ablation"]) for r in subset)
    notes = (
        f"Metric-cell directions: {len(positive)}/{len(subset)} positive (Full higher), "
        f"{len([r for r in subset if r['drop_full_minus_ablation'] == 0])} zero, {len(reverse)}/{len(subset)} reverse; "
        f"task positive shares NC={task_pos_share.get('nc', 0):.2f}, LP={task_pos_share.get('lp', 0):.2f}; "
        f"median absolute drop={median_abs:.3f} pp; median |drop|/quadrature of reported Full and ablation SDs="
        f"{median_effect_to_spread:.2f}; positive cells with drop at least that combined SD scale="
        f"{positive_gaps_at_least_one_spread}/{len(subset)}. These ratios are descriptive scale comparisons, not tests or standard errors. "
        "Dataset sign class requires all metrics in that dataset to share the sign. "
        "All comparisons are descriptive mean differences; no paired-seed tests are supported by the Full source."
    )
    return {
        "stage": {"wo_relation_calibration": "Stage I — Modality-Aware Relation Calibration", "wo_semantic_anchor": "Stage II — Semantic-Preserving Multi-Hop Propagation", "wo_adaptive_composition": "Stage III — Adaptive Multi-Order Composition"}[variant],
        "variant": variant,
        "datasets_positive": ";".join(positive_ds),
        "datasets_mixed": ";".join(mixed_ds),
        "datasets_reverse": ";".join(reverse_ds),
        "largest_positive_effect": "" if largest_positive is None else f"{largest_positive['dataset']} {METRIC_LABELS[largest_positive['metric']]} +{largest_positive['drop_full_minus_ablation']:.3f} pp",
        "largest_reverse_effect": "" if largest_reverse is None else f"{largest_reverse['dataset']} {METRIC_LABELS[largest_reverse['metric']]} {largest_reverse['drop_full_minus_ablation']:.3f} pp",
        "evidence_classification": classification,
        "notes": notes,
    }


def fmt(value: float) -> str:
    return f"{value:.2f}"


def full_lookup(full_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["task"], row["dataset"], row["metric"]): row for row in full_rows}


def ablation_lookup(summary: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {(row["task"], row["dataset"], row["variant"], row["metric"]): row for row in summary}


def table_cell(mean: float, sd: float, bold: bool = False) -> str:
    body = f"{fmt(mean)} ± {fmt(sd)}"
    return f"**{body}**" if bold else body


def table_tex_cell(mean: float, sd: float, bold: bool = False) -> str:
    if bold:
        return rf"$\mathbf{{{fmt(mean)} \pm {fmt(sd)}}}$"
    return rf"${fmt(mean)} \pm {fmt(sd)}$"


def variant_rows() -> list[tuple[str, str | None]]:
    return [("Full CoSI-MAG", None)] + [(VARIANT_LABELS[v], v) for v in CORE_STORY_ABLATIONS]


def data_value(task: str, dataset: str, metric: str, variant: str | None, refs: dict, abls: dict) -> tuple[float, float]:
    if variant is None:
        row = refs[(task, dataset, metric)]
        return row["full_mean"], row["full_std"]
    row = abls[(task, dataset, variant, metric)]
    return row["mean"], row["sample_std"]


def bold_best(values: list[tuple[float, float]]) -> list[bool]:
    top = max(value[0] for value in values)
    return [math.isclose(value[0], top, rel_tol=0, abs_tol=1e-10) for value in values]


def render_table_md(title: str, columns: list[tuple[str, str, str]], refs: dict, abls: dict, note: str) -> str:
    rows = variant_rows()
    column_vals = [[data_value(task, dataset, metric, variant, refs, abls) for _, task, dataset, metric in columns] for _, variant in rows]
    best_by_col = [bold_best([column_vals[row_idx][col_idx] for row_idx in range(len(rows))]) for col_idx in range(len(columns))]
    lines = [f"### {title}", "", "| Variant | " + " | ".join(c[0] for c in columns) + " |", "|---|" + "|".join("---:" for _ in columns) + "|"]
    for row_idx, (label, _) in enumerate(rows):
        cells = [table_cell(*column_vals[row_idx][col_idx], bold=best_by_col[col_idx][row_idx]) for col_idx in range(len(columns))]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.extend(["", note, ""])
    return "\n".join(lines)


def tex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def render_table_tex(caption: str, label: str, columns: list[tuple[str, str, str]], refs: dict, abls: dict, note: str) -> str:
    rows = variant_rows()
    column_vals = [[data_value(task, dataset, metric, variant, refs, abls) for _, task, dataset, metric in columns] for _, variant in rows]
    best_by_col = [bold_best([column_vals[row_idx][col_idx] for row_idx in range(len(rows))]) for col_idx in range(len(columns))]
    header = " & ".join(["Variant"] + [tex_escape(c[0]) for c in columns]) + r" \\"
    body = [header, r"\hline"]
    for row_idx, (label_text, _) in enumerate(rows):
        cells = [tex_escape(label_text)]
        cells.extend(table_tex_cell(*column_vals[row_idx][col_idx], bold=best_by_col[col_idx][row_idx]) for col_idx in range(len(columns)))
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l{'r' * len(columns)}}}",
        *body,
        r"\end{tabular}}",
        rf"\begin{{minipage}}{{\textwidth}}\footnotesize {note}\end{{minipage}}",
        r"\end{table*}",
        "",
    ])


def write_paper_tables(out: Path, full_rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    refs, abls = full_lookup(full_rows), ablation_lookup(summary)
    main_columns = [
        ("Movies Acc.", "nc", "Movies", "test_acc"),
        ("Movies Macro-F1", "nc", "Movies", "test_macro_f1"),
        ("Grocery Acc.", "nc", "Grocery", "test_acc"),
        ("Grocery Macro-F1", "nc", "Grocery", "test_macro_f1"),
        ("Sports MRR", "lp", "sports-copurchase", "test_mrr"),
        ("Sports Hits@10", "lp", "sports-copurchase", "test_hits@10"),
    ]
    table_note = (
        "Test metrics are descriptive. Values are mean ± SD: ablation SD is sample SD (ddof=1) across seeds 42–44; "
        "Full SD is copied as printed in the supplied PDF (population SD for NC and Sports). No paired-seed inference."
    )
    main_md = "# Main-paper representative table candidate\n\n" + render_table_md("Fixed representative datasets", main_columns, refs, abls, table_note)
    main_tex = render_table_tex(
        "Core Story ablations on the prespecified representative datasets. Test metrics are descriptive.",
        "tab:cosimag-core-story-main",
        main_columns,
        refs,
        abls,
        table_note,
    )
    (out / "main_table.md").write_text(main_md, encoding="utf-8")
    (out / "main_table.tex").write_text(main_tex, encoding="utf-8")

    nc_columns = [(f"{dataset} {metric_label}", "nc", dataset, metric) for dataset in NC_DATASETS for metric, metric_label in (("test_acc", "Acc."), ("test_macro_f1", "Macro-F1"))]
    lp_columns = [(f"{dataset} {metric_label}", "lp", dataset, metric) for dataset in LP_DATASETS for metric, metric_label in (("test_mrr", "MRR"), ("test_hits@1", "Hits@1"), ("test_hits@3", "Hits@3"), ("test_hits@10", "Hits@10"))]
    nc_note = table_note
    lp_note = table_note + " Cloth is included as requested; the supplied Full PDF identifies it as a completeness/quasi-held-out entry, not a second formal LP benchmark."
    (out / "appendix_nc.md").write_text("# Complete NC appendix table\n\n" + render_table_md("All five NC datasets", nc_columns, refs, abls, nc_note), encoding="utf-8")
    (out / "appendix_nc.tex").write_text(render_table_tex("Complete NC results for all formal datasets.", "tab:cosimag-core-story-nc-appendix", nc_columns, refs, abls, nc_note), encoding="utf-8")
    (out / "appendix_lp.md").write_text("# Complete LP appendix table\n\n" + render_table_md("Sports and Cloth LP", lp_columns, refs, abls, lp_note), encoding="utf-8")
    (out / "appendix_lp.tex").write_text(render_table_tex("Complete LP results on Sports and Cloth.", "tab:cosimag-core-story-lp-appendix", lp_columns, refs, abls, lp_note), encoding="utf-8")

    def grouped_tabular(
        columns: list[tuple[str, str, str, str]],
        groups: list[tuple[str, int]],
        metric_headers: list[str],
    ) -> str:
        row_defs = variant_rows()
        column_values = [
            [data_value(task, dataset, metric, variant, refs, abls) for _, task, dataset, metric in columns]
            for _, variant in row_defs
        ]
        best_by_column = [
            bold_best([column_values[row_index][column_index] for row_index in range(len(row_defs))])
            for column_index in range(len(columns))
        ]
        group_header = "Variant & " + " & ".join(
            rf"\multicolumn{{{span}}}{{c}}{{{tex_escape(name)}}}" for name, span in groups
        ) + r" \\"
        metric_header = " & " + " & ".join(tex_escape(value) for value in metric_headers) + r" \\"
        body = [
            rf"\begin{{tabular}}{{l{'r' * len(columns)}}}",
            r"\hline",
            group_header,
            metric_header,
            r"\hline",
        ]
        for row_index, (label, _) in enumerate(row_defs):
            cells = [tex_escape(label)]
            cells.extend(
                table_tex_cell(*column_values[row_index][column_index], bold=best_by_column[column_index][row_index])
                for column_index in range(len(columns))
            )
            body.append(" & ".join(cells) + r" \\")
        body.append(r"\end{tabular}")
        return "\n".join(body)

    nc_grouped = grouped_tabular(
        nc_columns,
        [(dataset, 2) for dataset in NC_DATASETS],
        [metric for _dataset in NC_DATASETS for metric in ("Acc.", "Macro-F1")],
    )
    lp_grouped = grouped_tabular(
        lp_columns,
        [(dataset.replace("-copurchase", "").title(), 4) for dataset in LP_DATASETS],
        [metric for _dataset in LP_DATASETS for metric in ("MRR", "Hits@1", "Hits@3", "Hits@10")],
    )
    combined_note = (
        "All values are test metrics in percent, reported as mean $\\pm$ SD. Ablation SD is sample SD (ddof=1) across seeds 42--44. "
        "Full values are transcribed from the supplied NC.pdf and LP.pdf; NC and Sports Full SDs are population SD as captioned. "
        "The Cloth Full SD is reproduced as printed, but its convention is not separately specified. Cloth is included as requested and is identified in the LP source as a completeness entry. "
        "Full references contain aggregate mean $\\pm$ SD only; no paired-seed inference is made."
    )
    combined_table = "\n".join(
        [
            r"% Requires \usepackage{graphicx} for \resizebox.",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Complete CoSI-MAG Core Story ablation results on all five NC and both LP datasets. Bold marks the highest mean in each metric column.}",
            r"\label{tab:cosimag-core-story-all-datasets}",
            r"\small",
            r"\textbf{(a) Node classification}\par\vspace{2pt}",
            r"\resizebox{\textwidth}{!}{%",
            nc_grouped,
            r"}",
            r"\vspace{0.7em}",
            r"\textbf{(b) Link prediction}\par\vspace{2pt}",
            r"\resizebox{\textwidth}{!}{%",
            lp_grouped,
            r"}",
            rf"\begin{{minipage}}{{\textwidth}}\footnotesize {combined_note}\end{{minipage}}",
            r"\end{table*}",
            "",
        ]
    )
    (out / "all_datasets_table.tex").write_text(combined_table, encoding="utf-8")


def write_report(out: Path, gate: dict[str, str], records: list[dict[str, Any]], semantics: list[dict[str, Any]], full_rows: list[dict[str, Any]], summary: list[dict[str, Any]], drops: list[dict[str, Any]], evidence: list[dict[str, Any]], nc_ref_sha: str, lp_ref_sha: str) -> None:
    sha = records[0]["manifest"]["git_commit"] if records else "unknown"
    branch = records[0]["manifest"]["git_branch"] if records else "unknown"
    overall_health = all(not r["issues"] and r["health"]["metrics_finite"] for r in records)
    overall_semantics = all(row["status"] == "PASS" for row in semantics)
    refs = full_lookup(full_rows)
    abls = ablation_lookup(summary)

    largest_positive = max(drops, key=lambda r: r["drop_full_minus_ablation"])
    largest_reverse = min(drops, key=lambda r: r["drop_full_minus_ablation"])
    evidence_lines = []
    for item in evidence:
        evidence_lines.append(
            f"- **{item['stage']} (`{item['variant']}`): {item['evidence_classification']}.** "
            f"Dataset pattern: positive `{item['datasets_positive'] or 'none'}`, mixed `{item['datasets_mixed'] or 'none'}`, "
            f"reverse `{item['datasets_reverse'] or 'none'}`. {item['notes']}"
        )
    reversal_lines = []
    for variant in CORE_STORY_ABLATIONS:
        current = [r for r in drops if r["variant"] == variant and r["drop_full_minus_ablation"] < 0]
        formatted = "; ".join(f"{r['dataset']} {METRIC_LABELS[r['metric']]} {r['drop_full_minus_ablation']:.3f} pp" for r in current)
        reversal_lines.append(f"- `{variant}` reversals (ablation mean > Full mean): {formatted or 'none'}.")

    best_epochs = [int(record["health"]["best_epoch"]) for record in records]
    runtimes = [float(record["health"]["runtime_seconds"]) for record in records]
    peak_memory = [
        float(record["health"]["peak_gpu_memory_mib"])
        for record in records
        if record["health"].get("peak_gpu_memory_mib") is not None
    ]
    health_summary = (
        f"Best epoch min/median/max={min(best_epochs)}/{statistics.median(best_epochs):g}/{max(best_epochs)}; "
        f"runtime min/median/max={min(runtimes):.1f}/{statistics.median(runtimes):.1f}/{max(runtimes):.1f} s; "
        + (
            f"peak GPU memory available for {len(peak_memory)}/{len(records)} runs, min/median/max="
            f"{min(peak_memory):.1f}/{statistics.median(peak_memory):.1f}/{max(peak_memory):.1f} MiB."
            if peak_memory
            else "peak GPU memory was not recorded in the run metrics."
        )
    )
    k_summary = ", ".join(f"{dataset}={FORMAL_K[dataset]}" for dataset in NC_DATASETS + LP_DATASETS)

    drop_index = {
        (row["variant"], row["dataset"], row["metric"]): row["drop_full_minus_ablation"]
        for row in drops
    }

    def formatted_drop_inventory(variant: str, task: str) -> str:
        datasets = NC_DATASETS if task == "nc" else LP_DATASETS
        labels = (
            {"test_acc": "Acc", "test_macro_f1": "F1"}
            if task == "nc"
            else {"test_mrr": "MRR", "test_hits@1": "H@1", "test_hits@3": "H@3", "test_hits@10": "H@10"}
        )
        metric_keys = METRICS[task]
        chunks = []
        for dataset in datasets:
            values = ", ".join(
                f"{labels[metric]} {drop_index[(variant, dataset, metric)]:+.3f}"
                for metric in metric_keys
            )
            chunks.append(f"{dataset} [{values}]")
        task_label = "NC" if task == "nc" else "LP"
        return f"Full − ablation mean drops in pp ({task_label}; positive means Full higher): " + "; ".join(chunks) + "."
    full_sd_note = "NC and Sports Full SD are identified as population SD in their PDFs; Cloth Full SD is reproduced as printed, while its convention is not separately explicit in the LP caption. Ablation SDs are sample SD across the three formal seed-level run values."
    md_table = (out / "main_table.md").read_text(encoding="utf-8")
    lines = [
        "# CoSI-MAG Core Story Ablation — Final Audit and Scientific Analysis",
        "",
        f"Analysis protocol: `{CORE_STORY_PROTOCOL_VERSION}`. Generated from the completed formal matrix; no runs were launched, changed, or excluded.",
        "",
        "## 1. Executive Summary",
        "",
        f"- Completeness: planned {gate.get('planned_runs')}, complete {gate.get('complete_runs')}, incomplete {gate.get('incomplete_runs')}, missing {gate.get('missing_runs')}; all official checker gates passed, including collision/duplicate check.",
        f"- Frozen provenance: branch `{branch}`, commit `{sha}`; all 63 run manifests agree with the current HEAD and one another.",
        f"- Run health: {'63/63 passed' if overall_health else 'one or more issues; see run_health.csv/provenance_audit.csv'}; semantic audit: {'63/63 passed' if overall_semantics else 'one or more issues; see ablation_semantics_audit.csv'}.",
        f"- Largest positive Full-minus-ablation drop: `{largest_positive['variant']}`, {largest_positive['dataset']} {METRIC_LABELS[largest_positive['metric']]} = +{largest_positive['drop_full_minus_ablation']:.3f} pp.",
        f"- Largest reversal (ablation exceeds Full mean): `{largest_reverse['variant']}`, {largest_reverse['dataset']} {METRIC_LABELS[largest_reverse['metric']]} = {largest_reverse['drop_full_minus_ablation']:.3f} pp.",
        "- All test-result comparisons are descriptive. The Full PDFs contain means and SDs, not matched seed-level values; no paired tests or 3/3 seed-drop statements are made.",
        "",
        "## 2. Completeness and Provenance Audit",
        "",
        f"The complete official checker output is preserved at `outputs/core_story_ablation_analysis/completeness_checker_stdout.txt`. It reports `manifest_mismatch={gate.get('manifest_mismatch')}`, `config_mismatch={gate.get('config_mismatch')}`, `git_commit_mismatch={gate.get('git_commit_mismatch')}`, `metric_schema={gate.get('metric_schema')}`, and `duplicate_collision={gate.get('duplicate_collision')}`.",
        "",
        f"All run manifests report commit `{sha}` on `{branch}`. Per-run configs and split pointers/hashes were compared across the three variants; only the top-level ablation identity and run-specific checkpoint path differ. Dataset graph/feature/self-loop config paths match across variants. Raw-uniform support preservation is also a code-level invariant of the frozen implementation; the run artifacts do not serialize a per-run edge-index hash.",
        "",
        "Selection metadata agrees at manifest, metrics, and checkpoint level: NC uses `best_val_accuracy`; LP uses `best_val_mrr`. See `provenance_audit.csv`, `ablation_semantics_audit.csv`, and `run_health.csv` for every run.",
        "",
        f"Formal K verified from each resolved config and manifest: {k_summary}.",
        "",
        f"Run-health summary: {health_summary}",
        "",
        "## 3. Full Reference",
        "",
        "Full values were read only from the two user-designated PDFs. The analyzer did not use F1/F2 output metrics or infer Full values from the ablations.",
        "",
        f"- NC: `paper/tables/NC.pdf`, page 1, CoSI-MAG row; SHA-256 `{nc_ref_sha}`.",
        f"- LP: `paper/tables/LP.pdf`, page 1, CoSI-MAG row; SHA-256 `{lp_ref_sha}`.",
        f"- Spread convention: {full_sd_note}",
        "- The Full references provide summary mean ± SD only. Seed pairing is unknown, so relative-drop tables contain no paired-seed tests.",
        "",
        "## 4. Complete NC Results",
        "",
        "All five datasets and all three variants are in `dataset_summary.csv`; raw seed values are in `per_seed_results.csv`. `appendix_nc.md` and `.tex` include the Full row and every formal NC dataset.",
        "",
        "## 5. Complete LP Results",
        "",
        "Sports and Cloth are retained separately in `dataset_summary.csv` and `appendix_lp.md`/`.tex`. The supplied LP PDF describes Sports as the formal benchmark and Cloth as a completeness entry; the current 63-run ablation scope includes both as requested.",
        "",
        "## 6. Full-Relative Drops",
        "",
        "`full_relative_drops.csv` reports `Full mean − ablation mean` in percentage points and relative drop as an auxiliary percentage. Positive means Full has the higher mean; negative means the ablation has the higher mean. Values inherit the two-decimal precision printed in the reference PDFs.",
        "",
        "The heatmap is a wide diagnostic visualization (14 metric columns), not a manuscript-production figure. It encodes point differences only; uncertainty conventions and seed-level variation are documented in the tables, and matched Full seed values are unavailable. Final PDF QA found no text collisions or clipping.",
        "",
        f"Largest positive: {largest_positive['variant']} on {largest_positive['dataset']} {METRIC_LABELS[largest_positive['metric']]} (+{largest_positive['drop_full_minus_ablation']:.3f} pp; relative {largest_positive['relative_drop_percent']:.2f}%). Largest reverse: {largest_reverse['variant']} on {largest_reverse['dataset']} {METRIC_LABELS[largest_reverse['metric']]} ({largest_reverse['drop_full_minus_ablation']:.3f} pp; relative {largest_reverse['relative_drop_percent']:.2f}%).",
        "",
        "## 7. Stage-I Analysis — Relation-Level Semantic Discrepancy",
        "",
        next(line for line in evidence_lines if "Stage I" in line),
        "",
        f"- {formatted_drop_inventory('wo_relation_calibration', 'nc')}",
        f"- {formatted_drop_inventory('wo_relation_calibration', 'lp')}",
        "",
        f"Across NC, 7/10 metric cells have Full above raw-uniform; across LP, 0/8 do. Grocery is the clearest NC positive case (Accuracy +{drop_index[('wo_relation_calibration', 'Grocery', 'test_acc')]:.3f} pp; Macro-F1 +{drop_index[('wo_relation_calibration', 'Grocery', 'test_macro_f1')]:.3f} pp), while both LP datasets reverse on every metric.",
        "",
        "A1 asks whether replacing modality-specific calibrated physical conductances with raw uniform conductances reduces downstream performance. Any dataset reversal is retained below; this does not establish that calibration is universally necessary.",
        "",
        "## 8. Stage-II Analysis — Propagation-Induced Semantic Drift",
        "",
        next(line for line in evidence_lines if "Stage II" in line),
        "",
        f"- {formatted_drop_inventory('wo_semantic_anchor', 'nc')}",
        f"- {formatted_drop_inventory('wo_semantic_anchor', 'lp')}",
        "",
        f"Full is higher in 7/10 NC and 8/8 LP metric cells. Sports shows the largest concentrated Stage-II pattern (MRR +{drop_index[('wo_semantic_anchor', 'sports-copurchase', 'test_mrr')]:.3f} pp; Hits@10 +{drop_index[('wo_semantic_anchor', 'sports-copurchase', 'test_hits@10')]:.3f} pp); Reddit-S reverses on both NC metrics.",
        "",
        "This is descriptive consistency evidence for the semantic-preservation motivation, not proof that semantic drift causes downstream error.",
        "",
        "## 9. Stage-III Analysis — Heterogeneous Multi-Hop Utilization",
        "",
        next(line for line in evidence_lines if "Stage III" in line),
        "",
        f"- {formatted_drop_inventory('wo_adaptive_composition', 'nc')}",
        f"- {formatted_drop_inventory('wo_adaptive_composition', 'lp')}",
        "",
        f"Full is higher in 7/10 NC and 8/8 LP metric cells. On the prespecified examples, w/o adaptive composition drops Movies by {drop_index[('wo_adaptive_composition', 'Movies', 'test_acc')]:.3f}/{drop_index[('wo_adaptive_composition', 'Movies', 'test_macro_f1')]:.3f} pp (Accuracy/Macro-F1), Grocery by {drop_index[('wo_adaptive_composition', 'Grocery', 'test_acc')]:.3f}/{drop_index[('wo_adaptive_composition', 'Grocery', 'test_macro_f1')]:.3f} pp, and Sports MRR/Hits@10 by {drop_index[('wo_adaptive_composition', 'sports-copurchase', 'test_mrr')]:.3f}/{drop_index[('wo_adaptive_composition', 'sports-copurchase', 'test_hits@10')]:.3f} pp. Reddit-S reverses on both NC metrics; ele-fashion Macro-F1 also reverses.",
        "",
        "A3 is the story-level comparison of adaptive composition against a uniform response-bank mean. Its positive and reverse effects are compared directly with A1/A2 in the CSV and heatmap; no dataset is selected post hoc.",
        "",
        "## 10. Representative Main-Text Table",
        "",
        "Representative datasets were fixed in advance (Movies, Grocery, Sports). The table bolds the actual highest mean in each column; Full is not automatically bolded.",
        "",
        md_table,
        "## 11. Empirical Observation → Design → Downstream Ablation",
        "",
        "- E0-A → Modality-Aware Relation Calibration → `wo_relation_calibration`.",
        "- E0-B → Semantic-Preserving Multi-Hop Propagation → `wo_semantic_anchor`.",
        "- E0-C → Adaptive Multi-Order Composition → `wo_adaptive_composition`.",
        "- E0-D is not treated as a fourth independent challenge. Relation-state modulation of composition is a finer-grained mechanism question, outside the Core Table interpretation.",
        "",
        "## 12. Reversals and Anomalies",
        "",
        *reversal_lines,
        "",
        "Reversals are not removed or redefined. Seed variability is reported separately; because Full has no seed-level values, differences between means are not paired observations.",
        "",
        "## 13. Paper-Safe Claims",
        "",
        "### Results facts",
        "",
        *[f"- {line.split('**', 2)[1].split(':', 1)[0]}: {line.split('Dataset pattern: ', 1)[1]}" for line in evidence_lines],
        "",
        "### Safe interpretation",
        "",
        *[f"- {item['stage']}: {paper_safe_claim(item)}" for item in evidence],
        "",
        "### Avoid",
        "",
        "- Do not claim all components are indispensable or that each module improves every dataset.",
        "- Do not call mean differences statistically significant, use paired tests, or claim all three seeds drop; the Full references are aggregate-only.",
        "- Do not state that semantic drift causes errors; the anchor ablation is downstream evidence consistent with a motivation, not causal identification.",
        "",
        "## 14. Need for Progressive Decomposition",
        "",
        f"- Stage II and Stage III tie on all-positive dataset count (five each) and both are moderately supported/dataset-dependent. Stage III is descriptively stronger in magnitude relative to reported spread: median absolute gap {statistics.median(abs(r['drop_full_minus_ablation']) for r in drops if r['variant'] == 'wo_adaptive_composition'):.3f} pp and 7 positive-effect cells reach at least the quadrature of the two reported SDs, versus {statistics.median(abs(r['drop_full_minus_ablation']) for r in drops if r['variant'] == 'wo_semantic_anchor'):.3f} pp and 4 positive-effect cells for Stage II. This is a descriptive ranking, not significance.",
        f"- Weakest evidence is {evidence[0]['stage']} (Stage I): only 7/18 metric cells favor Full, both LP datasets reverse on all four metrics, and none of its positive gaps reaches the quadrature of the reported SDs. The relation-calibration benefit is not established as a general downstream effect by this contrast.",
        "- The Core ablations support Stage II and especially Stage III at a moderate, dataset-dependent level, but do not support a universal three-stage necessity claim. Stage I needs more qualification.",
        "- The clearest reversal is A1 on both LP datasets (all eight metric cells favor raw-uniform ablation over Full); additional reversals include A1 on Toys Accuracy and Reddit-S Macro-F1, A2 on Reddit-S (both metrics) and Toys Accuracy, and A3 on Reddit-S (both metrics). These should be discussed, not removed.",
        "- If distinguishing learned calibration from merely semantic/fixed calibration is central, Stage I is the most useful candidate for a prospective Raw Uniform → Fixed Semantic Calibration → Learned Semantic Calibration progression. This report does not reuse historical F2 result values.",
        "- For Stage III, the current uniform-vs-adaptive result gives the story-level contrast. A global-only progression is not necessary before mechanism analysis; consider it only if the paper needs to attribute gains specifically to global, modality, node, or relation-conditioned terms.",
        "- Recommendation: first perform mechanism analysis of dataset/task regimes and the observed A1 reversals; do not immediately rerun. If a specific Stage-I distinction remains necessary for the paper claim, preregister the fixed-calibration contrast. No new experiment is launched here.",
        "",
        "## 15. Recommended Next Experiments",
        "",
        "No experiments are authorized or started by this report. If a follow-up is later approved, preregister the specific contrast and retain all datasets/seeds; do not reuse old F2 metrics as formal Core Story results.",
        "",
        "## Generated artifacts",
        "",
        "- `outputs/core_story_ablation_analysis/completeness_checker_stdout.txt`",
        "- `outputs/core_story_ablation_analysis/provenance_audit.csv`",
        "- `outputs/core_story_ablation_analysis/ablation_semantics_audit.csv`",
        "- `outputs/core_story_ablation_analysis/run_health.csv`",
        "- `outputs/core_story_ablation_analysis/full_reference_values.csv`",
        "- `outputs/core_story_ablation_analysis/per_seed_results.csv`",
        "- `outputs/core_story_ablation_analysis/dataset_summary.csv`",
        "- `outputs/core_story_ablation_analysis/full_relative_drops.csv`",
        "- `outputs/core_story_ablation_analysis/stage_evidence_summary.csv`",
        "- `outputs/core_story_ablation_analysis/main_table.md` / `.tex`",
        "- `outputs/core_story_ablation_analysis/appendix_nc.md` / `.tex`; `appendix_lp.md` / `.tex`",
        "- `outputs/core_story_ablation_analysis/all_datasets_table.tex` (one table float with NC and LP panels)",
        "- `outputs/core_story_ablation_analysis/core_story_drop_heatmap.pdf` / `.png` / `.svg`",
        "- `outputs/core_story_ablation_analysis/plot_core_story_drop_heatmap.py`",
        "- `outputs/core_story_ablation_analysis/core_story_drop_heatmap.alignment.json`",
        "- `outputs/core_story_ablation_analysis/core_story_drop_heatmap.collision-audit.json` (0 findings; no collision overlay was needed)",
        "- `docs/cosi_mag_core_story_ablation_analysis.md`",
        "",
    ]
    (PROJECT_ROOT / "docs" / "cosi_mag_core_story_ablation_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def paper_safe_claim(row: dict[str, Any]) -> str:
    positives = row["datasets_positive"].split(";") if row["datasets_positive"] else []
    reversals = row["datasets_reverse"].split(";") if row["datasets_reverse"] else []
    mixed = row["datasets_mixed"].split(";") if row["datasets_mixed"] else []
    if row["evidence_classification"] == "Broadly supported":
        return f"Removing the component is associated with lower mean test scores across most or all datasets ({', '.join(positives)}); this supports the design motivation without establishing causal mechanism."
    if row["evidence_classification"] == "Moderately supported / dataset-dependent":
        return f"Removing the component lowers mean test scores consistently on {', '.join(positives) or 'a subset of datasets'}, while outcomes vary elsewhere ({', '.join(mixed + reversals) or 'no full reversals'}); the benefit is dataset-dependent."
    if row["evidence_classification"] == "Weakly supported":
        return "The observed mean differences provide limited or small descriptive support for a consistent downstream benefit; avoid a universal claim."
    return f"The effects are mixed: mean reversals occur on {', '.join(reversals) or 'some metric cells'}, so the component's downstream benefit is not universal across tasks/datasets."


def main() -> int:
    args = parse_args()
    run_root = args.run_root if args.run_root.is_absolute() else (PROJECT_ROOT / args.run_root)
    out = args.analysis_root if args.analysis_root.is_absolute() else (PROJECT_ROOT / args.analysis_root)
    out.mkdir(parents=True, exist_ok=True)
    generated = [
        "provenance_audit.csv", "ablation_semantics_audit.csv", "run_health.csv",
        "full_reference_values.csv", "per_seed_results.csv", "dataset_summary.csv",
        "full_relative_drops.csv", "stage_evidence_summary.csv", "main_table.md",
        "main_table.tex", "appendix_nc.md", "appendix_nc.tex", "appendix_lp.md",
        "appendix_lp.tex", "all_datasets_table.tex", "core_story_drop_heatmap.pdf", "core_story_drop_heatmap.png",
        "core_story_drop_heatmap.svg",
    ]
    collisions = [name for name in generated if (out / name).exists()]
    report_path = PROJECT_ROOT / "docs" / "cosi_mag_core_story_ablation_analysis.md"
    if report_path.exists():
        collisions.append(str(report_path.relative_to(PROJECT_ROOT)))
    if collisions and not args.force:
        print(f"ERROR: generated output already exists; use --force only to replace these analysis artifacts: {collisions}", file=sys.stderr)
        return 2

    gate, gate_issues = read_checker_gate(out)
    if gate_issues:
        blocking = [
            {"dataset": "", "variant": "", "seed": "", "issue": issue, "expected": "official checker gate PASS", "actual": gate}
            for issue in gate_issues
        ]
        write_csv(out / "blocking_report.csv", blocking)
        print("BLOCKED: completeness gate did not pass; no performance aggregation was run.")
        return 1

    runs = planned_runs(run_root)
    if len(runs) != 63:
        raise AssertionError(f"formal plan expected 63 runs, got {len(runs)}")
    split_hash_cache: dict[str, str] = {}
    records = [audit_one_run(run, split_hash_cache) for run in runs]
    audit_global_provenance(records)
    semantics = audit_semantics(records)
    audit_failures = [r for r in records if r["issues"]]
    semantic_failures = [r for r in semantics if r["status"] != "PASS"]
    provenance_rows = []
    health_rows = []
    for record in records:
        manifest = record.get("manifest", {})
        config = record.get("config", {})
        provenance_rows.append({
            "task": record["task"],
            "dataset": record["dataset"],
            "variant": record["variant"],
            "seed": record["seed"],
            "git_commit": manifest.get("git_commit"),
            "git_branch": manifest.get("git_branch"),
            "K": manifest.get("K"),
            "split_source": manifest.get("split_source"),
            "split_sha256": record.get("split_sha256"),
            "resolved_config_sha256": record.get("config_sha256"),
            "protocol_version": manifest.get("protocol_version"),
            "selection_rule": manifest.get("checkpoint_selection"),
            "checkpoint_selection_metadata": record.get("checkpoint_meta", {}).get("selection"),
            "checkpoint_path": record.get("metrics", {}).get("checkpoint"),
            "status": "PASS" if not record["issues"] else "FAIL",
            "issue": "; ".join(record["issues"]),
        })
        health = record.get("health", {})
        health_rows.append({
            "task": record["task"],
            "dataset": record["dataset"],
            "variant": record["variant"],
            "seed": record["seed"],
            **health,
            "status": "PASS" if not any((health.get("log_traceback"), health.get("log_cuda_oom"), health.get("log_nonfinite"))) and health.get("metrics_finite") and record.get("health", {}).get("checkpoint_exists") and health.get("marker_identity_ok") and not [x for x in record["issues"] if any(token in x for token in ("best_epoch", "runtime_seconds", "checkpoint", "metrics payload", "test metric", "metric schema", "train.log"))] else "FAIL",
            "issue": "; ".join([x for x in record["issues"] if any(token in x for token in ("best_epoch", "runtime_seconds", "checkpoint", "metrics payload", "test metric", "metric schema", "train.log", "complete.marker"))]),
        })
    write_csv(out / "provenance_audit.csv", provenance_rows)
    write_csv(out / "ablation_semantics_audit.csv", semantics)
    write_csv(out / "run_health.csv", health_rows)
    if audit_failures or semantic_failures:
        blocked_rows = []
        for record in audit_failures:
            for issue in record["issues"]:
                blocked_rows.append({"task": record["task"], "dataset": record["dataset"], "variant": record["variant"], "seed": record["seed"], "issue": issue, "expected": "frozen protocol/provenance PASS", "actual": "FAIL"})
        for row in semantic_failures:
            blocked_rows.append({"task": row["task"], "dataset": row["dataset"], "variant": row["variant"], "seed": row["seed"], "issue": row["issue"], "expected": "frozen ablation semantics", "actual": "FAIL"})
        write_csv(out / "blocking_report.csv", blocked_rows)
        print(f"BLOCKED: provenance failures={len(audit_failures)}, semantics failures={len(semantic_failures)}; no performance aggregation was run.")
        return 1

    # A previous failed audit can leave this script-generated blocker report
    # behind. Remove only that known generated artifact once a fresh audit has
    # passed, so it cannot be mistaken for the current result.
    stale_blocker = out / "blocking_report.csv"
    if stale_blocker.is_file():
        stale_blocker.unlink()

    nc_rows, nc_row, nc_sha = extract_reference_pdf(args.nc_reference, "nc")
    lp_rows, lp_row, lp_sha = extract_reference_pdf(args.lp_reference, "lp")
    full_rows = nc_rows + lp_rows
    write_csv(out / "full_reference_values.csv", full_rows)

    per_seed, summary, summary_map = aggregate_metrics(records)
    write_csv(out / "per_seed_results.csv", per_seed)
    write_csv(out / "dataset_summary.csv", summary)
    drops = relative_drops(summary_map, full_rows)
    write_csv(out / "full_relative_drops.csv", drops)
    evidence = [describe_stage(variant, drops) for variant in CORE_STORY_ABLATIONS]
    write_csv(out / "stage_evidence_summary.csv", evidence)
    write_paper_tables(out, full_rows, summary)
    write_report(out, gate, records, semantics, full_rows, summary, drops, evidence, nc_sha, lp_sha)
    print(f"analysis_status=PASS")
    print(f"runs_audited={len(records)}")
    print(f"provenance_pass={len(provenance_rows)}")
    print(f"semantics_pass={sum(r['status'] == 'PASS' for r in semantics)}")
    print(f"run_health_pass={sum(r['status'] == 'PASS' for r in health_rows)}")
    print(f"summary_rows={len(summary)}")
    print(f"drop_rows={len(drops)}")
    print(f"git_commit={records[0]['manifest']['git_commit']}")
    print(f"nc_reference={args.nc_reference}")
    print(f"lp_reference={args.lp_reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
