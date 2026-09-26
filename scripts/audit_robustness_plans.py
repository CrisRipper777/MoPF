#!/usr/bin/env python3
"""Validate authoritative matched robustness plans before any inference."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_robustness import _load_injected_edges
from scripts.prepare_robustness_perturbations import (
    RANDOM_RHOS,
    CONFLICT_RHOS,
    PERTURBATION_SEEDS,
    conflict_category_masks,
    sha256_edges,
    sha256_file,
    budget_for_rho,
)
from src.data import load_mag_data
from src.data.graph_utils import canonicalize_edges

OUT = ROOT / "outputs/robustness_analysis"
AUDIT_PATH = OUT / "checkpoint_audit_matched.csv"
CLEAN_GATE_PATH = OUT / "clean_gate_matched.csv"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
RANDOM_MODELS = ("mopf", "wo_relation_calibration", "wo_semantic_anchor", "dip")
CONFLICT_MODELS = ("mopf", "wo_relation_calibration", "dip")
EXPECTED_SPLITS = {
    "Movies": "9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211",
    "Grocery": "3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e",
}
FORMAL_K = {"Movies": 3, "Grocery": 2}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def edge_artifact(output_root: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray | None]:
    path = (output_root / row["edge_artifact"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != row["edge_artifact_sha256"]:
        raise ValueError(f"edge-artifact file SHA mismatch: {path}")
    if path.suffix == ".npy":
        edges = np.load(path, allow_pickle=False)
        categories = None
    else:
        with np.load(path, allow_pickle=False) as archive:
            edges = np.asarray(archive["edges"])
            categories = np.asarray(archive["category"])
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if sha256_edges(edges) != row["edge_sha256"]:
        raise ValueError(f"canonical edge-list SHA mismatch: {path}")
    if len(edges) != int(row["injected_edge_count"]):
        raise ValueError(f"edge count mismatch: {path}")
    if categories is not None and len(categories) != len(edges):
        raise ValueError(f"category count mismatch: {path}")
    return edges, categories


def _check_conditions(plan: list[dict[str, str]], perturbation: str) -> None:
    models = RANDOM_MODELS if perturbation == "random_noise" else CONFLICT_MODELS
    rhos = tuple(RANDOM_RHOS if perturbation == "random_noise" else CONFLICT_RHOS)
    expected = {
        (dataset, model, model_seed, perturb_seed, float(rho))
        for dataset in DATASETS
        for model in models
        for model_seed in SEEDS
        for perturb_seed in PERTURBATION_SEEDS
        for rho in rhos
    }
    observed = {
        (row["dataset"], row["model"], int(row["model_seed"]), int(row["perturbation_seed"]), float(row["rho"]))
        for row in plan
    }
    if len(plan) != len(expected) or observed != expected:
        raise ValueError(
            f"{perturbation} condition matrix mismatch: expected={len(expected)}, rows={len(plan)}, "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )


def _load_clean_edges(audit_rows: list[dict[str, str]], dataset: str) -> tuple[np.ndarray, int]:
    row = next(row for row in audit_rows if row["dataset"] == dataset and row["model"] == "mopf" and row["model_seed"] == "42")
    config = OmegaConf.load(row["config_path"])
    OmegaConf.resolve(config)
    data = load_mag_data(config, "nc", int(config.seed))
    canonical = canonicalize_edges(data.edge_index).detach().cpu().numpy().astype(np.int64, copy=False)
    return canonical.reshape(-1, 2), int(data.num_nodes)


def main() -> None:
    audit_rows = read_csv(AUDIT_PATH)
    clean_rows = read_csv(CLEAN_GATE_PATH)
    random_plan = read_csv(OUT / "random_noise/dry_run_plan_matched.csv")
    conflict_plan = read_csv(OUT / "semantic_conflict/dry_run_plan_matched.csv")
    _check_conditions(random_plan, "random_noise")
    _check_conditions(conflict_plan, "semantic_conflict")

    audit_by_key = {
        (row["dataset"], row["model"], int(row["model_seed"])): row for row in audit_rows
    }
    if len(audit_rows) != 24 or len(audit_by_key) != 24:
        raise ValueError("matched checkpoint audit must contain exactly 24 unique model-seed rows")
    clean_by_path = {str(Path(row["checkpoint_path"]).resolve()): row for row in clean_rows}
    if len(clean_by_path) != 24 or any(row.get("status") != "PASS" for row in clean_rows):
        raise ValueError("clean gate must contain exactly 24 passing checkpoints")
    for row in audit_rows:
        cp = Path(row["checkpoint_path"])
        if not cp.is_file() or sha256_file(cp) != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed since matched audit: {cp}")
        key = (row["dataset"], row["model"], int(row["model_seed"]))
        if key not in audit_by_key:
            raise ValueError(f"unregistered matched checkpoint: {key}")
        if row["split_sha256"] != EXPECTED_SPLITS[row["dataset"]]:
            raise ValueError(f"split SHA mismatch for {key}")
        gate = clean_by_path.get(str(cp.resolve()))
        if gate is None or gate["checkpoint_sha256"] != row["checkpoint_sha256"]:
            raise ValueError(f"clean gate does not match checkpoint: {cp}")
        if not math.isclose(float(gate["measured_accuracy"]), float(row["clean_test_accuracy"]), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"clean Accuracy mismatch: {cp}")
        if not math.isclose(float(gate["measured_macro_f1"]), float(row["clean_test_macro_f1"]), rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"clean Macro-F1 mismatch: {cp}")

    for plan in (random_plan, conflict_plan):
        for row in plan:
            key = (row["dataset"], row["model"], int(row["model_seed"]))
            audit = audit_by_key[key]
            if str(Path(row["checkpoint_path"]).resolve()) != str(Path(audit["checkpoint_path"]).resolve()):
                raise ValueError(f"plan checkpoint path differs from audit: {key}")
            if row["checkpoint_sha256"] != audit["checkpoint_sha256"]:
                raise ValueError(f"plan checkpoint SHA differs from audit: {key}")
            if row["split_sha256"] != audit["split_sha256"]:
                raise ValueError(f"plan split SHA differs from audit: {key}")
            num_edges = int(row["clean_canonical_edges"])
            budget = budget_for_rho(float(row["rho"]), num_edges)
            if int(row["injected_edge_count"]) != budget:
                raise ValueError(f"integer rounding budget mismatch for {key}, rho={row['rho']}")
            if int(row["perturbed_canonical_edges"]) != num_edges + budget:
                raise ValueError(f"perturbed edge total mismatch for {key}, rho={row['rho']}")

    # Every model/model seed receives the same stored edge artifact for a
    # dataset × perturbation seed × rho condition.
    edge_cache: dict[tuple[str, str, int, float], tuple[np.ndarray, np.ndarray | None]] = {}
    checks: dict[str, Any] = {}
    clean_support: dict[str, set[tuple[int, int]]] = {}
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for dataset in DATASETS:
        clean_edges, num_nodes = _load_clean_edges(audit_rows, dataset)
        node_counts[dataset] = num_nodes
        edge_counts[dataset] = len(clean_edges)
        if len(clean_edges) != int(next(r for r in random_plan if r["dataset"] == dataset)["clean_canonical_edges"]):
            raise ValueError(f"clean graph edge count differs from plan: {dataset}")
        clean_support[dataset] = set(map(tuple, clean_edges.tolist()))

    for name, plan in (("random_noise", random_plan), ("semantic_conflict", conflict_plan)):
        sharing: dict[tuple[str, int, float], set[str]] = defaultdict(set)
        for row in plan:
            condition_key = (row["dataset"], int(row["perturbation_seed"]), float(row["rho"]))
            sharing[condition_key].add(row["edge_sha256"])
            cache_key = (row["dataset"], name, int(row["perturbation_seed"]), float(row["rho"]))
            if cache_key not in edge_cache:
                edges, categories = edge_artifact(OUT, row)
                if len(edges) and (
                    np.any(edges[:, 0] < 0)
                    or np.any(edges[:, 0] >= edges[:, 1])
                    or np.any(edges[:, 1] >= node_counts[row["dataset"]])
                ):
                    raise ValueError(f"noncanonical/in-range injection in {row['edge_artifact']}")
                if len(edges) and any(tuple(edge) in clean_support[row["dataset"]] for edge in edges.tolist()):
                    raise ValueError(f"injected edge overlaps clean support: {row['edge_artifact']}")
                edge_cache[cache_key] = (edges, categories)
            edges, _ = edge_cache[cache_key]
            if int(row["injected_edge_count"]) != len(edges):
                raise ValueError(f"injected edge count differs: {row['edge_artifact']}")
        if any(len(hashes) != 1 for hashes in sharing.values()):
            raise ValueError(f"not all models share the same {name} perturbation edge SHA")
        checks[f"{name}_shared_graph_sha"] = True

    # Verify sampled random-noise files are exact prefixes of the frozen nested pools.
    capacity_rows = read_csv(OUT / "random_noise/perturbation_capacity.csv")
    random_rhos = tuple(RANDOM_RHOS)
    for capacity in capacity_rows:
        dataset = capacity["dataset"]
        pseed = int(capacity["perturbation_seed"])
        pool_path = (OUT / capacity["edge_list_path"]).resolve()
        if sha256_file(pool_path) != capacity["edge_artifact_sha256"]:
            raise ValueError(f"random ordered pool file SHA mismatch: {pool_path}")
        pool = np.asarray(np.load(pool_path, allow_pickle=False), dtype=np.int64).reshape(-1, 2)
        if sha256_edges(pool) != capacity["edge_list_sha256"] or len(pool) != int(capacity["maximum_requested_edges"]):
            raise ValueError(f"random ordered pool content mismatch: {pool_path}")
        previous = np.empty((0, 2), dtype=np.int64)
        clean_edges = int(capacity["clean_canonical_edges"])
        for rho in random_rhos:
            planrow = next(row for row in random_plan if row["dataset"] == dataset and int(row["perturbation_seed"]) == pseed and float(row["rho"]) == rho)
            edges, _ = edge_cache[(dataset, "random_noise", pseed, rho)]
            budget = budget_for_rho(rho, clean_edges)
            if len(edges) != budget or not np.array_equal(edges, pool[:budget]):
                raise ValueError(f"random-noise edge list is not the frozen nested pool prefix: {planrow['edge_artifact']}")
            if not np.array_equal(previous, edges[:len(previous)]):
                raise ValueError(f"random-noise budgets are not nested: {dataset}/pseed{pseed}/rho{rho}")
            previous = edges
    checks["random_noise_budgets_nested"] = True

    # Validate the saved raw-feature ranks and every balanced conflict prefix.
    conflict_capacities = read_csv(OUT / "semantic_conflict/candidate_capacity.csv")
    conflict_capacity_by_key = {
        (row["dataset"], int(row["perturbation_seed"])): row for row in conflict_capacities
    }
    for key, capacity in conflict_capacity_by_key.items():
        dataset, pseed = key
        pool_path = (OUT / capacity["candidate_pool_path"]).resolve()
        if sha256_file(pool_path) != capacity["candidate_pool_sha256"]:
            raise ValueError(f"semantic candidate-pool SHA mismatch: {pool_path}")
        with np.load(pool_path, allow_pickle=False) as archive:
            candidate_edges = np.asarray(archive["candidate_edges"], dtype=np.int64)
            rank_text = np.asarray(archive["rank_text"], dtype=np.float64)
            rank_visual = np.asarray(archive["rank_visual"], dtype=np.float64)
            saved_category = np.asarray(archive["category"], dtype=np.uint8)
        text_mask, visual_mask = conflict_category_masks(rank_text, rank_visual, 0.75, 0.25)
        recomputed_category = np.where(text_mask, 1, np.where(visual_mask, 2, 0)).astype(np.uint8)
        if not np.array_equal(saved_category, recomputed_category):
            raise ValueError(f"frozen semantic category labels disagree with saved raw rank rules: {pool_path}")
        if len(candidate_edges) != 1_000_000 or len(np.unique(candidate_edges, axis=0)) != len(candidate_edges):
            raise ValueError(f"semantic candidate pool size/uniqueness mismatch: {pool_path}")
        candidate_keys = candidate_edges[:, 0] * np.int64(node_counts[dataset]) + candidate_edges[:, 1]
        order = np.argsort(candidate_keys)
        sorted_candidate_keys = candidate_keys[order]
        if np.any(sorted_candidate_keys[1:] == sorted_candidate_keys[:-1]):
            raise ValueError(f"duplicate canonical candidate edge: {pool_path}")
        sorted_candidate_categories = saved_category[order]
        previous_edges = np.empty((0, 2), dtype=np.int64)
        previous_categories = np.empty(0, dtype=np.uint8)
        for rho in tuple(CONFLICT_RHOS):
            planrow = next(row for row in conflict_plan if row["dataset"] == dataset and int(row["perturbation_seed"]) == pseed and float(row["rho"]) == rho)
            edges, categories = edge_cache[(dataset, "semantic_conflict", pseed, rho)]
            if categories is None:
                raise ValueError(f"semantic conflict artifact lacks category labels: {planrow['edge_artifact']}")
            edge_keys = edges[:, 0] * np.int64(node_counts[dataset]) + edges[:, 1]
            positions = np.searchsorted(sorted_candidate_keys, edge_keys)
            found = positions < len(sorted_candidate_keys)
            found[found] &= sorted_candidate_keys[positions[found]] == edge_keys[found]
            expected_categories = np.zeros(len(edges), dtype=np.uint8)
            expected_categories[found] = sorted_candidate_categories[positions[found]]
            if not np.array_equal(categories, expected_categories) or np.any(categories == 0):
                raise ValueError(f"semantic injected edges fail the frozen C_T/C_V rank categories: {planrow['edge_artifact']}")
            text_count = int(np.sum(categories == 1))
            visual_count = int(np.sum(categories == 2))
            if abs(text_count - visual_count) > 1 or text_count + visual_count != len(edges):
                raise ValueError(f"semantic injection is not approximately 50/50: {planrow['edge_artifact']}")
            if not np.array_equal(previous_edges, edges[:len(previous_edges)]) or not np.array_equal(previous_categories, categories[:len(previous_categories)]):
                raise ValueError(f"semantic conflict budgets are not nested: {dataset}/pseed{pseed}/rho{rho}")
            previous_edges, previous_categories = edges, categories
    checks["semantic_categories_frozen_raw_rank"] = True
    checks["semantic_conflict_balanced_and_nested"] = True

    summary = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_audit_rows": len(audit_rows),
        "clean_gate_rows": len(clean_rows),
        "random_noise_plan_rows": len(random_plan),
        "semantic_conflict_plan_rows": len(conflict_plan),
        "dataset_split_sha256": EXPECTED_SPLITS,
        "dataset_clean_canonical_edges": edge_counts,
        "dataset_test_node_count": {
            dataset: int(next(row for row in clean_rows if row["dataset"] == dataset)["test_node_count"])
            for dataset in DATASETS
        },
        "checks": {
            "condition_matrix_complete": True,
            "no_nonfinite_clean_metrics": all(
                math.isfinite(float(row["measured_accuracy"])) and math.isfinite(float(row["measured_macro_f1"]))
                for row in clean_rows
            ),
            "rho0_clean_gate_pass": all(row["status"] == "PASS" for row in clean_rows),
            "checkpoint_hashes_unchanged": True,
            "all_models_share_matched_split": True,
            **checks,
        },
        "edge_files_regenerated": False,
        "candidate_pools_regenerated": False,
        "perturbation_inference_started": False,
    }
    path = OUT / "robustness_pre_evaluation_audit.json"
    write_json(path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
