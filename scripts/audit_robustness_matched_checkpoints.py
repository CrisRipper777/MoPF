#!/usr/bin/env python3
"""Build the matched Figure-5 checkpoint map and run rho=0 clean gates only.

Preparation remaps the frozen perturbation plans to the matched checkpoint
family without evaluating any perturbed graph.  Clean mode performs ordinary
full-graph inference on the unchanged graph for each audited checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS_ROOT = ROOT / "outputs/robustness_analysis"
CHECKPOINT_ROOT = ROOT / "outputs/robustness_matched_checkpoints"
SOURCE_AUDIT = ROBUSTNESS_ROOT / "checkpoint_audit.csv"
MATCHED_AUDIT = ROBUSTNESS_ROOT / "checkpoint_audit_matched.csv"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
ABLATIONS = ("wo_relation_calibration", "wo_semantic_anchor")
K_BY_DATASET = {"Movies": 3, "Grocery": 2}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_robustness import (  # noqa: E402
    CLEAN_ATOL,
    _load_checkpoint_context,
    _metric_matches,
    _run_metrics,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_payload(path: Path, seed: int) -> tuple[bool, bool, str, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_state = payload.get("model_state")
    head_state = payload.get("head_state")
    if not isinstance(model_state, dict) or not isinstance(head_state, dict):
        raise ValueError(f"checkpoint lacks model/head state: {path}")
    for name, state in (("model_state", model_state), ("head_state", head_state)):
        for key, tensor in state.items():
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                raise ValueError(f"nonfinite {name}.{key} in {path}")
    payload_seed = payload.get("seed", seed)
    if int(payload_seed) != int(seed):
        raise ValueError(f"checkpoint payload seed mismatch: {path}: {payload_seed}")
    selection = str(payload.get("selection", ""))
    if selection != "best_val_accuracy":
        raise ValueError(f"checkpoint was not selected by best validation Accuracy: {path}: {selection}")
    epoch = int(payload.get("epoch", -1))
    if epoch < 0:
        raise ValueError(f"checkpoint has invalid best epoch: {path}")
    return True, True, selection, epoch


def prepare_matched_audit() -> list[dict[str, str]]:
    if not SOURCE_AUDIT.is_file():
        raise FileNotFoundError(SOURCE_AUDIT)
    old_rows = read_csv(SOURCE_AUDIT)
    old_map: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in old_rows:
        key = (row["dataset"], row["model"], int(row["model_seed"]))
        old_map[key] = row

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for model in ("mopf", "dip", *ABLATIONS):
            for seed in SEEDS:
                if model in ABLATIONS and seed != 42:
                    run_dir = CHECKPOINT_ROOT / dataset / model / f"seed{seed}"
                    sidecar_path = run_dir / "robustness_manifest.json"
                    if not sidecar_path.is_file():
                        raise FileNotFoundError(f"missing matched-run manifest: {sidecar_path}")
                    sidecar = read_json(sidecar_path)
                    if sidecar.get("purpose") != "figure5_robustness_matched_split":
                        raise ValueError(f"wrong run purpose in {sidecar_path}")
                    source_key = (dataset, model, 42)
                    if source_key not in old_map:
                        raise ValueError(f"missing seed-42 source ablation row: {source_key}")
                    row = dict(old_map[source_key])
                    checkpoint_path = Path(sidecar["checkpoint_path"]).resolve()
                    config_path = run_dir / ".hydra/config.yaml"
                    manifest_path = sidecar_path
                    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
                    if str(config["dataset"]["name"]) != dataset or int(config["seed"]) != seed:
                        raise ValueError(f"resolved config identity mismatch: {config_path}")
                    if str(config["ablation"]) != model or str(config["task"]["name"]) != "nc":
                        raise ValueError(f"resolved config model/ablation/task mismatch: {config_path}")
                    split_path = Path(sidecar["split_path"]).resolve()
                    split_sha = sha256_file(split_path)
                    if split_sha != sidecar["split_sha256"]:
                        raise ValueError(f"new-run split SHA mismatch: {sidecar_path}")
                    if Path(config["dataset"]["nc_split_path"]).resolve() != split_path:
                        raise ValueError(f"resolved NC split mismatch: {config_path}")
                    if int(sidecar["K"]) != K_BY_DATASET[dataset]:
                        raise ValueError(f"wrong K in robustness manifest: {sidecar_path}")
                    row.update({
                        "checkpoint_family": "robustness_matched_checkpoints",
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_candidates": json.dumps([str(checkpoint_path)]),
                        "checkpoint_candidate_count": "1",
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "checkpoint_selection": "best_val_accuracy",
                        "payload_selection": "best_val_accuracy",
                        "payload_seed": str(seed),
                        "config_path": str(config_path.resolve()),
                        "config_sha256": sha256_file(config_path),
                        "config_dataset": dataset,
                        "config_model": "mopf",
                        "config_task": "nc",
                        "protocol": "unified_full_graph_nc_v1",
                        "training_mode": "full_graph",
                        "inference_mode": "full",
                        "configured_K": str(K_BY_DATASET[dataset]),
                        "formal_K": str(K_BY_DATASET[dataset]),
                        "split_path": str(split_path),
                        "split_sha256": split_sha,
                        "split_design": "fixed_seed42_robustness_matched_split",
                        "git_branch": "vnext",
                        "git_commit": sidecar["git_commit"],
                        "clean_test_accuracy": str(sidecar["test_accuracy"]),
                        "clean_test_macro_f1": str(sidecar["test_macro_f1"]),
                        "run_record_or_manifest_path": str(manifest_path.resolve()),
                        "run_record_or_manifest_sha256": sha256_file(manifest_path),
                        "audit_status": "PASS",
                        "purpose": sidecar["purpose"],
                        "training_seed": str(seed),
                        "split_seed": "42",
                        "best_epoch": str(sidecar["best_epoch"]),
                        "best_validation_accuracy": str(sidecar["validation_accuracy"]),
                    })
                else:
                    source_key = (dataset, model, seed)
                    if source_key not in old_map:
                        raise ValueError(f"missing source checkpoint audit row: {source_key}")
                    row = dict(old_map[source_key])
                    checkpoint_path = Path(row["checkpoint_path"])
                    config_path = Path(row["config_path"])
                    if row.get("audit_status") != "PASS":
                        raise ValueError(f"source checkpoint row is not PASS: {source_key}")
                    if not checkpoint_path.is_file() or not config_path.is_file():
                        raise FileNotFoundError(f"missing source checkpoint/config: {source_key}")
                    if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
                        raise ValueError(f"source checkpoint SHA changed: {source_key}")
                    if sha256_file(config_path) != row["config_sha256"]:
                        raise ValueError(f"source config SHA changed: {source_key}")
                    row["purpose"] = "paper_nc_final_fixed_v1" if model in {"mopf", "dip"} else "core_story_ablation_seed42_reused"
                    row["training_seed"] = str(seed)
                    row["split_seed"] = "42"
                    row["best_epoch"] = row.get("payload_best_epoch", "")
                cp_path = Path(row["checkpoint_path"]).resolve()
                if not cp_path.is_file():
                    raise FileNotFoundError(cp_path)
                row["dataset"] = dataset
                row["model"] = model
                row["model_seed"] = str(seed)
                row["formal_K"] = str(K_BY_DATASET[dataset])
                if sha256_file(cp_path) != row["checkpoint_sha256"]:
                    raise ValueError(f"checkpoint SHA mismatch: {cp_path}")
                has_model, has_head, selection, epoch = _validate_payload(
                    cp_path, int(row["model_seed"])
                )
                if selection != row.get("checkpoint_selection"):
                    raise ValueError(f"checkpoint selection disagrees with audit row: {cp_path}")
                row["model_state_present"] = str(has_model)
                row["head_state_present"] = str(has_head)
                row["payload_selection"] = selection
                row["payload_best_epoch"] = str(epoch)
                if row.get("best_epoch") not in (None, "", str(epoch)):
                    raise ValueError(f"best epoch mismatch between run record and checkpoint: {cp_path}")
                row["best_epoch"] = str(epoch)
                row["checkpoint_path"] = str(cp_path)
                row["random_noise_included"] = "true"
                row["semantic_conflict_included"] = "true" if model in {"mopf", "dip", "wo_relation_calibration"} else "false"
                rows.append(row)

    if len(rows) != 24:
        raise ValueError(f"expected 24 matched audit rows, got {len(rows)}")
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    for dataset, dataset_rows in by_dataset.items():
        for condition, included in (
            ("random_noise", lambda row: row["random_noise_included"] == "true"),
            ("semantic_conflict", lambda row: row["semantic_conflict_included"] == "true"),
        ):
            selected = [row for row in dataset_rows if included(row)]
            expected_count = 12 if condition == "random_noise" else 9
            shas = {row["split_sha256"] for row in selected}
            if len(selected) != expected_count or len(shas) != 1:
                raise ValueError(
                    f"{dataset}/{condition}: expected {expected_count} rows on one split, "
                    f"got {len(selected)} rows and {len(shas)} split hashes"
                )

    write_csv(MATCHED_AUDIT, rows)
    _rebuild_matched_plans(rows)
    summary = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_audit": str(SOURCE_AUDIT),
        "matched_audit": str(MATCHED_AUDIT),
        "checkpoint_rows": len(rows),
        "random_noise_checkpoints_per_dataset": 12,
        "semantic_conflict_checkpoints_per_dataset": 9,
        "all_dataset_condition_split_hashes_unique": True,
        "dataset_split_sha256": {
            dataset: sorted({row["split_sha256"] for row in data_rows})[0]
            for dataset, data_rows in by_dataset.items()
        },
    }
    (ROBUSTNESS_ROOT / "matched_checkpoint_audit_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Matched checkpoint audit PASS: {MATCHED_AUDIT}", flush=True)
    print("Random Noise: 12 checkpoints/dataset; Semantic Conflict: 9 checkpoints/dataset", flush=True)
    return rows


def _rebuild_matched_plans(audit_rows: list[dict[str, Any]]) -> None:
    lookup = {
        (str(row["dataset"]), str(row["model"]), int(row["model_seed"])): row
        for row in audit_rows
    }
    for group in ("random_noise", "semantic_conflict"):
        original = ROBUSTNESS_ROOT / group / "dry_run_plan.csv"
        if not original.is_file():
            raise FileNotFoundError(original)
        plans = read_csv(original)
        matched: list[dict[str, Any]] = []
        for plan in plans:
            key = (plan["dataset"], plan["model"], int(plan["model_seed"]))
            if key not in lookup:
                raise ValueError(f"plan row lacks matched checkpoint audit entry: {key}")
            audit = lookup[key]
            if group == "semantic_conflict" and audit["semantic_conflict_included"] != "true":
                raise ValueError(f"excluded model appears in conflict plan: {key}")
            if plan["model"] != "mopf" and audit["random_noise_included"] != "true":
                raise ValueError(f"excluded model appears in random-noise plan: {key}")
            updated = dict(plan)
            updated.update({
                "checkpoint_path": audit["checkpoint_path"],
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "config_path": audit["config_path"],
                "split_sha256": audit["split_sha256"],
                "expected_clean_accuracy": audit["clean_test_accuracy"],
                "expected_clean_macro_f1": audit["clean_test_macro_f1"],
            })
            matched.append(updated)
        expected = 432 if group == "random_noise" else 270
        if len(matched) != expected:
            raise ValueError(f"{group} matched plan expected {expected} rows, got {len(matched)}")
        write_csv(ROBUSTNESS_ROOT / group / "dry_run_plan_matched.csv", matched)


def run_clean_gate(rows: list[dict[str, str]], device_name: str) -> list[dict[str, Any]]:
    device = torch.device(device_name)
    results: list[dict[str, Any]] = []
    test_indices: dict[str, torch.Tensor] = {}
    for index, row in enumerate(rows, start=1):
        checkpoint = Path(row["checkpoint_path"])
        config = Path(row["config_path"])
        expected_acc = float(row["clean_test_accuracy"])
        expected_f1 = float(row["clean_test_macro_f1"])
        gate: dict[str, Any] = {
            "dataset": row["dataset"],
            "model": row["model"],
            "model_seed": row["model_seed"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "split_sha256": row["split_sha256"],
            "expected_accuracy": expected_acc,
            "expected_macro_f1": expected_f1,
            "purpose": row["purpose"],
        }
        try:
            data, model, classifier, eval_labels, state = _load_checkpoint_context(
                checkpoint, config, device
            )
            actual_nodes = torch.sort(data.test_idx.detach().cpu().long()).values
            previous = test_indices.get(row["dataset"])
            nodes_identical = previous is None or torch.equal(previous, actual_nodes)
            if previous is None:
                test_indices[row["dataset"]] = actual_nodes.clone()
            measured = _run_metrics(
                data, model, classifier, eval_labels, state, device, data.edge_index
            )
            acc_ok = _metric_matches(measured["accuracy"], expected_acc)
            f1_ok = _metric_matches(measured["macro_f1"], expected_f1)
            finite_metrics = math.isfinite(measured["accuracy"]) and math.isfinite(measured["macro_f1"])
            status = "PASS" if acc_ok and f1_ok and finite_metrics and nodes_identical else "FAIL"
            gate.update({
                "measured_accuracy": measured["accuracy"],
                "measured_macro_f1": measured["macro_f1"],
                "accuracy_abs_error": abs(measured["accuracy"] - expected_acc),
                "macro_f1_abs_error": abs(measured["macro_f1"] - expected_f1),
                "test_node_count": int(actual_nodes.numel()),
                "test_nodes_sha256": hashlib.sha256(actual_nodes.numpy().tobytes()).hexdigest(),
                "test_nodes_identical_within_dataset": nodes_identical,
                "finite_metrics": finite_metrics,
                "accuracy_matches_checkpoint": acc_ok,
                "macro_f1_matches_checkpoint": f1_ok,
                "status": status,
            })
            del data, model, classifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[{index}/{len(rows)}] {status} {row['dataset']}/{row['model']}/seed{row['model_seed']} "
                f"Acc={measured['accuracy']:.6f} F1={measured['macro_f1']:.6f} "
                f"test_nodes={actual_nodes.numel()}",
                flush=True,
            )
        except Exception as exc:  # retain a row for every requested checkpoint
            gate.update({
                "measured_accuracy": "",
                "measured_macro_f1": "",
                "accuracy_abs_error": "",
                "macro_f1_abs_error": "",
                "test_node_count": "",
                "test_nodes_sha256": "",
                "test_nodes_identical_within_dataset": False,
                "finite_metrics": False,
                "accuracy_matches_checkpoint": False,
                "macro_f1_matches_checkpoint": False,
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{index}/{len(rows)}] FAIL {checkpoint}: {exc}", flush=True)
        results.append(gate)

    # Recheck identity over all rows after the complete set has been loaded.
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if result.get("test_nodes_sha256"):
            by_dataset[str(result["dataset"])].append(str(result["test_nodes_sha256"]))
    for dataset in DATASETS:
        group_hashes = by_dataset[dataset]
        same = len(group_hashes) == 12 and len(set(group_hashes)) == 1
        for result in results:
            if result["dataset"] == dataset:
                result["test_nodes_identical_within_dataset"] = same
                if not same:
                    result["status"] = "FAIL"

    gate_path = ROBUSTNESS_ROOT / "clean_gate_matched.csv"
    write_csv(gate_path, results)
    passed = len(results) == 24 and all(row["status"] == "PASS" for row in results)
    summary = {
        "status": "PASS" if passed else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_name,
        "checkpoints": len(results),
        "passed": sum(row["status"] == "PASS" for row in results),
        "test_nodes_identical": {
            dataset: len(by_dataset[dataset]) == 12 and len(set(by_dataset[dataset])) == 1
            for dataset in DATASETS
        },
        "clean_gate_csv": str(gate_path),
        "perturbation_evaluation_started": False,
    }
    (ROBUSTNESS_ROOT / "clean_gate_matched.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(f"Matched clean gate failed; see {gate_path}")
    print(f"Matched clean gate PASS: 24/24 checkpoints; CSV={gate_path}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="prepare audit/plans without inference")
    parser.add_argument("--clean-only", action="store_true", help="run rho=0 gate using existing matched audit")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.prepare_only and args.clean_only:
        raise SystemExit("choose either --prepare-only or --clean-only")
    if args.clean_only:
        if not MATCHED_AUDIT.is_file():
            raise FileNotFoundError(MATCHED_AUDIT)
        rows = read_csv(MATCHED_AUDIT)
    else:
        rows = prepare_matched_audit()
    if args.prepare_only:
        return
    run_clean_gate(rows, args.device)


if __name__ == "__main__":
    main()
