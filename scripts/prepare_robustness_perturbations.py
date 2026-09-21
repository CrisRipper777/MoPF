#!/usr/bin/env python3
"""Audit robustness checkpoints and prepare fixed graph perturbations.

This script only reads checkpoints/configuration and frozen graph/features. It
does not instantiate a trained model, access labels for edge generation, or
perform inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mopf_e0a_edge_semantic_discrepancy import (  # noqa: E402
    _average_rank_cdf,
    _edge_cosines,
)
from src.data import load_mag_data  # noqa: E402
from src.data.graph_utils import canonicalize_edges  # noqa: E402
from src.data.loaders import resolve_path  # noqa: E402


DATASETS = ("Movies", "Grocery")
MODEL_SEEDS = (42, 43, 44)
PERTURBATION_SEEDS = (1001, 1002, 1003)
FORMAL_K = {"Movies": 3, "Grocery": 2}
RANDOM_RHOS = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40)
CONFLICT_RHOS = (0.00, 0.05, 0.10, 0.20, 0.30)
RANK_HIGH = 0.75
RANK_LOW = 0.25
# Frozen before candidate-capacity or model-performance inspection.
CONFLICT_CANDIDATE_POOL_SIZE = 1_000_000
EDGE_COSINE_CHUNK_SIZE = 65_536
RANDOM_MODELS = (
    ("mopf", "Full CoSI-MAG"),
    ("wo_relation_calibration", "wo_relation_calibration"),
    ("wo_semantic_anchor", "wo_semantic_anchor"),
    ("dip", "DiP"),
)
CONFLICT_MODELS = (
    ("mopf", "Full CoSI-MAG"),
    ("wo_relation_calibration", "wo_relation_calibration"),
    ("dip", "DiP"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_edges(edges: np.ndarray) -> str:
    values = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def budget_for_rho(rho: float, num_edges: int) -> int:
    """Round requested edge budgets half-up; this rule is fixed in the protocol."""
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if num_edges < 0:
        raise ValueError("num_edges must be nonnegative")
    return int(math.floor(float(rho) * int(num_edges) + 0.5))


def _edge_keys(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return edges[:, 0] * np.int64(num_nodes) + edges[:, 1]


def sample_true_nonedges(
    num_nodes: int,
    clean_edges: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample unique canonical nonedges by bounded-memory vectorized rejection."""
    n = int(num_nodes)
    requested = int(count)
    if n < 2 or requested < 0:
        raise ValueError("num_nodes must be >=2 and count must be nonnegative")
    clean = np.asarray(clean_edges, dtype=np.int64).reshape(-1, 2)
    if clean.size and (
        np.any(clean[:, 0] >= clean[:, 1])
        or np.any(clean < 0)
        or np.any(clean >= n)
    ):
        raise ValueError("clean_edges must be in-range canonical non-self pairs")
    if requested == 0:
        return np.empty((0, 2), dtype=np.int64)

    clean_keys = np.unique(_edge_keys(clean, n))
    available = n * (n - 1) // 2 - clean_keys.size
    if requested > available:
        raise ValueError(f"requested {requested} nonedges but only {available} exist")

    accepted_chunks: list[np.ndarray] = []
    accepted_keys = np.empty(0, dtype=np.int64)
    have = 0
    attempts = 0
    while have < requested:
        remaining = requested - have
        batch = min(max(65_536, int(remaining * 1.35)), 1_000_000)
        raw = rng.integers(0, n, size=(batch, 2), dtype=np.int64)
        raw.sort(axis=1)
        valid = raw[:, 0] != raw[:, 1]
        proposals = raw[valid]
        if proposals.size:
            keys = _edge_keys(proposals, n)
            nonclean = ~np.isin(keys, clean_keys, assume_unique=False)
            proposals = proposals[nonclean]
            keys = keys[nonclean]
            if keys.size:
                _, first = np.unique(keys, return_index=True)
                first.sort()
                proposals = proposals[first]
                keys = keys[first]
                fresh = ~np.isin(keys, accepted_keys, assume_unique=False)
                proposals = proposals[fresh]
                keys = keys[fresh]
                take = min(remaining, len(proposals))
                if take:
                    accepted_chunks.append(proposals[:take].copy())
                    accepted_keys = np.concatenate([accepted_keys, keys[:take]])
                    have += take
        attempts += 1
        if attempts > 100 and have < requested:
            raise RuntimeError(
                f"nonedge sampler stalled at {have}/{requested}; graph may be too dense"
            )
    return np.concatenate(accepted_chunks, axis=0).astype(np.int64, copy=False)


def conflict_category_masks(
    rank_text: np.ndarray,
    rank_visual: np.ndarray,
    high: float = RANK_HIGH,
    low: float = RANK_LOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen E0-A average-rank category thresholds."""
    text = np.asarray(rank_text, dtype=np.float64).reshape(-1)
    visual = np.asarray(rank_visual, dtype=np.float64).reshape(-1)
    if text.size != visual.size:
        raise ValueError("text and visual rank arrays must have equal length")
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("rank thresholds must satisfy 0 <= low < high <= 1")
    c_text = (text >= high) & (visual <= low)
    c_visual = (visual >= high) & (text <= low)
    return c_text, c_visual


def maximum_balanced_capacity(num_text: int, num_visual: int) -> int:
    """Largest injectible count whose two modality categories differ by <=1."""
    t, v = int(num_text), int(num_visual)
    if t < 0 or v < 0:
        raise ValueError("category capacities must be nonnegative")
    return 2 * min(t, v) + int(t != v)


def balanced_category_counts(total: int, num_text: int, num_visual: int) -> tuple[int, int]:
    """Allocate a requested prefix with at most one edge of class imbalance."""
    total, cap_t, cap_v = int(total), int(num_text), int(num_visual)
    if total < 0 or total > maximum_balanced_capacity(cap_t, cap_v):
        raise ValueError("requested count exceeds balanced candidate capacity")
    half = total // 2
    if total % 2 == 0:
        return half, half
    if cap_t > cap_v:
        return half + 1, half
    if cap_v > cap_t:
        return half, half + 1
    return half + 1, half


def build_balanced_nested_order(
    text_edges: np.ndarray,
    visual_edges: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic shuffled order with every prefix balanced to 1."""
    text_edges = np.asarray(text_edges, dtype=np.int64).reshape(-1, 2)
    visual_edges = np.asarray(visual_edges, dtype=np.int64).reshape(-1, 2)
    rng = np.random.default_rng(int(seed))
    text_order = text_edges[rng.permutation(len(text_edges))] if len(text_edges) else text_edges
    visual_order = (
        visual_edges[rng.permutation(len(visual_edges))]
        if len(visual_edges)
        else visual_edges
    )
    capacity = maximum_balanced_capacity(len(text_order), len(visual_order))
    if capacity == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.uint8)

    # A seed-stable tie-break while the class counts are equal.
    first_class = int(rng.integers(0, 2))
    t_pos = v_pos = 0
    edges: list[np.ndarray] = []
    classes: list[int] = []
    while len(edges) < capacity:
        if t_pos < v_pos:
            chosen = 1
        elif v_pos < t_pos:
            chosen = 2
        else:
            if t_pos >= len(text_order):
                chosen = 2
            elif v_pos >= len(visual_order):
                chosen = 1
            else:
                chosen = 1 if first_class == 0 else 2
                first_class = 1 - first_class
        if chosen == 1:
            if t_pos >= len(text_order) or t_pos - v_pos >= 1:
                break
            edges.append(text_order[t_pos])
            classes.append(1)
            t_pos += 1
        else:
            if v_pos >= len(visual_order) or v_pos - t_pos >= 1:
                break
            edges.append(visual_order[v_pos])
            classes.append(2)
            v_pos += 1
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2), np.asarray(classes, dtype=np.uint8)


def edge_index_from_canonical(
    clean_edge_index: torch.Tensor,
    injected_edges: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Append both orientations of injected pairs, preserving the clean prefix."""
    clean = torch.as_tensor(clean_edge_index, dtype=torch.long).contiguous()
    if clean.ndim != 2 or clean.shape[0] != 2:
        raise ValueError("clean_edge_index must have shape [2, num_edges]")
    injected = torch.as_tensor(injected_edges, dtype=torch.long).reshape(-1, 2)
    if injected.numel() == 0:
        return clean.clone()
    if torch.any(injected[:, 0] >= injected[:, 1]):
        raise ValueError("injected pairs must be canonical non-self edges")
    if torch.unique(injected, dim=0).size(0) != injected.size(0):
        raise ValueError("injected pairs contain duplicate/reverse-duplicate edges")
    clean_physical = canonicalize_edges(clean)
    # Direct pair membership remains exact for isolated highest-index nodes.
    clean_set = {tuple(pair) for pair in clean_physical.detach().cpu().tolist()}
    if any(tuple(pair) in clean_set for pair in injected.detach().cpu().tolist()):
        raise ValueError("injected pairs overlap clean physical support")
    added = torch.cat([injected.t().contiguous(), injected[:, [1, 0]].t().contiguous()], dim=1)
    return torch.cat([clean, added.to(device=clean.device)], dim=1).contiguous()


def audit_clean_graph(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """Return canonical support after enforcing the loader's bidirected convention."""
    edges = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu().contiguous()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("loader edge_index must have shape [2, E]")
    if edges.numel() and (edges.min().item() < 0 or edges.max().item() >= num_nodes):
        raise ValueError("clean edge_index endpoint outside node range")
    if edges.numel() and torch.any(edges[0] == edges[1]):
        raise ValueError("clean physical edge_index unexpectedly contains self loops")
    oriented = edges[0].numpy() * np.int64(num_nodes) + edges[1].numpy()
    if np.unique(oriented).size != oriented.size:
        raise ValueError("clean edge_index contains duplicate oriented edges")
    physical = canonicalize_edges(edges).numpy().astype(np.int64, copy=False)
    if edges.size(1) != 2 * len(physical):
        raise ValueError("clean loader graph is not exactly bidirected per physical edge")
    if len(physical):
        forward = set(map(tuple, edges.t().numpy().tolist()))
        if any((int(v), int(u)) not in forward for u, v in physical.tolist()):
            raise ValueError("clean graph lacks a reverse orientation for a physical pair")
    return physical


def validate_shared_plan_artifacts(rows: list[dict[str, Any]]) -> None:
    """Require one exact perturbation artifact per dataset/seed/rho group."""
    grouped: dict[tuple[str, str, int, float], set[tuple[str, str]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["perturbation_type"]),
            int(row["perturbation_seed"]),
            float(row["rho"]),
        )
        grouped.setdefault(key, set()).add(
            (str(row["edge_artifact"]), str(row["edge_sha256"]))
        )
    inconsistent = {key: values for key, values in grouped.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"models do not share the same perturbation artifact: {inconsistent}")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_config(path: Path) -> tuple[Any, dict[str, Any]]:
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg, OmegaConf.to_container(cfg, resolve=True)


def _checkpoint_candidates(root: Path, dataset: str, model: str, seed: int) -> list[Path]:
    if model in {"mopf", "dip"}:
        family = "paper_nc_final_fixed_v1"
        return [root / "outputs" / family / dataset / model / f"seed{seed}" / "best.pt"]
    return [
        root
        / "outputs/core_story_ablation/nc"
        / dataset
        / model
        / f"seed{seed}/best.pt"
    ]


def discover_dip_nc_candidates(root: Path) -> dict[tuple[str, int], list[Path]]:
    """Find all matching DiP checkpoints and retain only formal target NC runs."""
    candidates: dict[tuple[str, int], list[Path]] = {
        (dataset, seed): [] for dataset in DATASETS for seed in MODEL_SEEDS
    }
    for path in (root / "outputs").rglob("best.pt"):
        if path.parent.parent.name != "dip":
            continue
        if not path.parent.name.startswith("seed"):
            continue
        record_path = path.parent / "run_record.json"
        metrics_path = path.parent / "metrics.json"
        record = _json(record_path) if record_path.is_file() else {}
        if record.get("task") != "nc" or record.get("protocol") != "unified_full_graph_nc_v1":
            continue
        dataset = str(record.get("dataset", ""))
        seed = int(record.get("seed", -1))
        if (dataset, seed) in candidates:
            candidates[(dataset, seed)].append(path.resolve())
        elif metrics_path.is_file():
            # Other formal NC dataset families are not in this figure's fixed scope.
            continue
    return candidates


def _audit_one_checkpoint(
    root: Path,
    dataset: str,
    model: str,
    model_label: str,
    seed: int,
    dip_candidates: dict[tuple[str, int], list[Path]],
) -> dict[str, Any]:
    path_candidates = _checkpoint_candidates(root, dataset, model, seed)
    checkpoint = path_candidates[0]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required authoritative checkpoint missing: {checkpoint}")
    run_dir = checkpoint.parent
    config_path = run_dir / ".hydra/config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    cfg, resolved = _resolved_config(config_path)
    dataset_cfg = resolved["dataset"]
    task_cfg = resolved["task"]
    model_cfg = resolved["model"]

    if model in {"mopf", "dip"}:
        record_path = run_dir / "run_record.json"
        record = _json(record_path)
        protocol = record.get("protocol", "")
        metrics_path = run_dir / "results.json"
        metrics_json = _json(metrics_path)
        clean_acc = float(record["test_accuracy"])
        clean_f1 = float(record["test_macro_f1"])
        selection = "best_val_accuracy"
        health = record.get("health_checks", {})
        health_pass = record.get("status") == "PASS" and all(health.values())
        family = "paper_nc_final_fixed_v1"
        git_branch = record.get("git_branch", "not_recorded")
        git_commit = record.get("git_commit", "not_recorded")
        candidate_count = 1
        candidate_paths = [str(checkpoint.resolve())]
        if model == "dip":
            observed = dip_candidates[(dataset, seed)]
            candidate_count = len(observed)
            candidate_paths = [str(item) for item in observed]
            if len(observed) != 1 or observed[0] != checkpoint.resolve():
                raise RuntimeError(
                    "DiP formal NC provenance is not unique for "
                    f"{dataset}/seed{seed}; candidates={candidate_paths}"
                )
    else:
        record_path = run_dir / "ablation_manifest.json"
        record = _json(record_path)
        metrics_path = run_dir / "metrics.json"
        metrics_json = _json(metrics_path)
        result_json = _json(run_dir / "results.json")
        metric_values = metrics_json["metrics"]
        clean_acc = float(metric_values["test_acc"]["mean"])
        clean_f1 = float(metric_values["test_macro_f1"]["mean"])
        selection = str(record.get("checkpoint_selection", ""))
        protocol = str(record.get("protocol_version", ""))
        marker = run_dir / "complete.marker"
        health_pass = marker.is_file() and bool(result_json) and bool(metrics_json)
        family = "core_story_ablation"
        git_branch = record.get("git_branch", "not_recorded")
        git_commit = record.get("git_commit", "not_recorded")
        candidate_count = 1
        candidate_paths = [str(checkpoint.resolve())]

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload_has_model = isinstance(payload, dict) and isinstance(payload.get("model_state"), dict)
    payload_has_head = isinstance(payload, dict) and isinstance(payload.get("head_state"), dict)
    payload_selection = payload.get("selection") if isinstance(payload, dict) else None
    payload_seed = payload.get("seed") if isinstance(payload, dict) else None
    del payload

    cfg_matches = (
        str(dataset_cfg.get("name")) == dataset
        and str(model_cfg.get("name")) == ("mopf" if model.startswith("wo_") else model)
        and str(task_cfg.get("name")) == "nc"
        and int(resolved.get("seed", -1)) == seed
        and str(task_cfg.get("protocol_version", "")) == "unified_full_graph_nc_v1"
        and str(task_cfg.get("inference_mode", "")) == "full"
    )
    expected_ablation = model if model.startswith("wo_") else None
    ablation_matches = expected_ablation is None or record.get("ablation_name") == expected_ablation
    if model == "wo_relation_calibration":
        ablation_matches = ablation_matches and record.get("relation_calibration_effective") is False
    if model == "wo_semantic_anchor":
        ablation_matches = ablation_matches and record.get("semantic_anchor_effective") is False
    if model in {"mopf", "wo_relation_calibration", "wo_semantic_anchor"}:
        cfg_matches = cfg_matches and int(model_cfg.get("max_order", -1)) == FORMAL_K[dataset]
    if not cfg_matches or not ablation_matches or not health_pass:
        raise RuntimeError(
            f"Checkpoint audit failed for {dataset}/{model}/seed{seed}: "
            f"config={cfg_matches}, ablation={ablation_matches}, health={health_pass}"
        )
    if not payload_has_model or not payload_has_head:
        raise RuntimeError(f"Checkpoint restore payload incomplete: {checkpoint}")
    if payload_selection not in (None, selection, "best_val_accuracy"):
        raise RuntimeError(f"Checkpoint selection mismatch at {checkpoint}: {payload_selection}")
    if payload_seed is not None and int(payload_seed) != seed:
        raise RuntimeError(f"Checkpoint seed mismatch at {checkpoint}: {payload_seed}")
    if protocol != "unified_full_graph_nc_v1" or selection != "best_val_accuracy":
        raise RuntimeError(f"Unexpected protocol/selection at {checkpoint}")

    split_path = Path(str(dataset_cfg["nc_split_path"]))
    if not split_path.is_absolute():
        split_path = resolve_path(split_path)
    split_path = split_path.resolve()
    split_sha = sha256_file(split_path)
    expected_split_seed = 42 if family == "paper_nc_final_fixed_v1" else seed
    if f"nc_seed{expected_split_seed}_" not in split_path.name:
        raise RuntimeError(
            f"unexpected clean NC split for {dataset}/{model}/seed{seed}: {split_path}"
        )
    graph_path = resolve_path(dataset_cfg["graph_path"]).resolve()
    text_feature_path = resolve_path(dataset_cfg["text_feat_path"]).resolve()
    visual_feature_path = resolve_path(dataset_cfg["image_feat_path"]).resolve()
    return {
        "dataset": dataset,
        "model": model,
        "model_label": model_label,
        "model_seed": seed,
        "formal_K": FORMAL_K[dataset],
        "checkpoint_family": family,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_candidate_count": candidate_count,
        "checkpoint_candidates": json.dumps(candidate_paths),
        "checkpoint_selection": selection,
        "payload_selection": payload_selection or "not_recorded",
        "payload_seed": payload_seed if payload_seed is not None else "not_recorded",
        "model_state_present": payload_has_model,
        "head_state_present": payload_has_head,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "config_dataset": dataset_cfg.get("name"),
        "config_model": model_cfg.get("name"),
        "config_task": task_cfg.get("name"),
        "protocol": protocol,
        "training_mode": task_cfg.get("training_mode", "not_recorded"),
        "inference_mode": task_cfg.get("inference_mode", "not_recorded"),
        "configured_K": model_cfg.get("max_order", "not_applicable"),
        "split_path": str(split_path),
        "split_sha256": split_sha,
        "split_design": "fixed_seed42_across_model_seeds" if family == "paper_nc_final_fixed_v1" else "seed_specific_formal_ablation_split",
        "graph_path": str(graph_path),
        "text_feature_path": str(text_feature_path),
        "visual_feature_path": str(visual_feature_path),
        "feature_mode": str(dataset_cfg.get("feature_mode", "both")),
        "make_undirected": bool(dataset_cfg.get("make_undirected", True)),
        "add_self_loops": bool(dataset_cfg.get("add_self_loops", True)),
        "git_branch": git_branch,
        "git_commit": git_commit,
        "clean_test_accuracy": clean_acc,
        "clean_test_macro_f1": clean_f1,
        "run_record_or_manifest_path": str(record_path.resolve()),
        "run_record_or_manifest_sha256": sha256_file(record_path),
        "audit_status": "PASS",
    }


def build_checkpoint_audit(root: Path) -> list[dict[str, Any]]:
    dip_candidates = discover_dip_nc_candidates(root)
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        model_defs = (
            ("mopf", "Full CoSI-MAG"),
            ("wo_relation_calibration", "wo_relation_calibration"),
            ("wo_semantic_anchor", "wo_semantic_anchor"),
            ("dip", "DiP"),
        )
        for model, label in model_defs:
            for seed in MODEL_SEEDS:
                rows.append(_audit_one_checkpoint(root, dataset, model, label, seed, dip_candidates))
    source_fields = (
        "graph_path",
        "text_feature_path",
        "visual_feature_path",
        "feature_mode",
        "make_undirected",
        "add_self_loops",
    )
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        signatures = {tuple(str(row[field]) for field in source_fields) for row in dataset_rows}
        if len(signatures) != 1:
            raise RuntimeError(f"checkpoint configs disagree on clean data/graph sources for {dataset}")
    return rows


def _load_clean_dataset(root: Path, dataset: str) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    config_path = (
        root / "outputs/paper_nc_final_fixed_v1" / dataset / "mopf/seed42/.hydra/config.yaml"
    )
    cfg, _ = _resolved_config(config_path)
    # Official loader builds the exact clean NC graph. Split labels are loaded by
    # that API, but none are passed to either edge sampler or rank function.
    data = load_mag_data(cfg, "nc", 42)
    physical = audit_clean_graph(data.edge_index, int(data.num_nodes))
    data_cfg = cfg.dataset
    raw_sources = {
        "text_feature_path": str(resolve_path(data_cfg.text_feat_path)),
        "text_feature_sha256": sha256_file(resolve_path(data_cfg.text_feat_path)),
        "visual_feature_path": str(resolve_path(data_cfg.image_feat_path)),
        "visual_feature_sha256": sha256_file(resolve_path(data_cfg.image_feat_path)),
        "raw_text_feature_shape": list(data.x_t.shape),
        "raw_visual_feature_shape": list(data.x_i.shape),
        "graph_path": str(resolve_path(data_cfg.graph_path)),
        "graph_sha256": sha256_file(resolve_path(data_cfg.graph_path)),
        "split_path": str(resolve_path(data_cfg.nc_split_path)),
    }
    if data.x_t is None or data.x_i is None:
        raise ValueError(f"{dataset} loader did not expose both raw modality features")
    return cfg, data, physical, raw_sources


def _dataset_code(dataset: str) -> int:
    return {"Movies": 11, "Grocery": 23}[dataset]


def _common_conflict_schedule(capacities: list[int], num_edges: list[int]) -> tuple[float, ...]:
    feasible: list[float] = []
    for rho in CONFLICT_RHOS:
        ok = all(budget_for_rho(rho, edges) <= cap for cap, edges in zip(capacities, num_edges))
        if ok:
            feasible.append(rho)
    if not feasible or feasible[0] != 0.0:
        raise RuntimeError("rho=0 must be feasible for every semantic-conflict condition")
    max_common = max(feasible)
    return tuple(rho for rho in CONFLICT_RHOS if rho <= max_common and rho in feasible)


def _write_edge_artifact(
    path: Path,
    edges: np.ndarray,
    *,
    classes: np.ndarray | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if classes is None:
        np.save(path, np.ascontiguousarray(edges, dtype=np.int64))
    else:
        np.savez_compressed(
            path,
            edges=np.ascontiguousarray(edges, dtype=np.int64),
            category=np.ascontiguousarray(classes, dtype=np.uint8),
        )
    return sha256_edges(edges)


def _make_plans_and_perturbations(
    root: Path,
    output_root: Path,
    device: torch.device,
    candidate_pool_size: int = CONFLICT_CANDIDATE_POOL_SIZE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if candidate_pool_size != CONFLICT_CANDIDATE_POOL_SIZE:
        raise ValueError("Candidate pool size is frozen at 1,000,000 per dataset/perturbation seed")

    checkpoint_rows = build_checkpoint_audit(root)
    write_csv(output_root / "checkpoint_audit.csv", checkpoint_rows)

    random_capacity_rows: list[dict[str, Any]] = []
    random_manifest: list[dict[str, Any]] = []
    random_plan: list[dict[str, Any]] = []
    conflict_capacity_rows: list[dict[str, Any]] = []
    conflict_manifest: list[dict[str, Any]] = []
    conflict_plan: list[dict[str, Any]] = []
    graph_summary: dict[str, Any] = {}
    common_caps: list[int] = []
    common_edges: list[int] = []

    # First pass: load each dataset once and create nested random-noise artifacts.
    loaded: dict[str, tuple[Any, Any, np.ndarray, dict[str, Any]]] = {}
    for dataset in DATASETS:
        cfg, data, clean_edges, raw_sources = _load_clean_dataset(root, dataset)
        loaded[dataset] = cfg, data, clean_edges, raw_sources
        num_nodes = int(data.num_nodes)
        num_edges = int(len(clean_edges))
        clean_columns = int(data.edge_index.size(1))
        possible = num_nodes * (num_nodes - 1) // 2
        graph_summary[dataset] = {
            "num_nodes": num_nodes,
            "clean_canonical_edges": num_edges,
            "clean_edge_index_columns": clean_columns,
            "directed_columns_per_canonical_edge": 2,
            "possible_undirected_pairs": possible,
            "available_clean_nonedges": possible - num_edges,
            "formal_K": FORMAL_K[dataset],
            "graph_path": raw_sources["graph_path"],
            "graph_sha256": raw_sources["graph_sha256"],
            **raw_sources,
        }

        for perturb_seed in PERTURBATION_SEEDS:
            max_budget = budget_for_rho(max(RANDOM_RHOS), num_edges)
            rng = np.random.default_rng(np.random.SeedSequence([perturb_seed, _dataset_code(dataset), 100]))
            ordered = sample_true_nonedges(num_nodes, clean_edges, max_budget, rng)
            clean_keys = set(map(tuple, clean_edges.tolist()))
            if any(tuple(edge) in clean_keys for edge in ordered.tolist()):
                raise AssertionError("random perturbation sampler returned a clean edge")
            if len(np.unique(ordered, axis=0)) != len(ordered):
                raise AssertionError("random perturbation sampler returned duplicate pairs")
            pool_path = output_root / "random_noise/perturbation_edges" / f"{dataset}_pseed{perturb_seed}_maxrho040.npy"
            pool_sha = _write_edge_artifact(pool_path, ordered)
            pool_artifact_sha = sha256_file(pool_path)
            random_capacity_rows.append(
                {
                    "dataset": dataset,
                    "perturbation_seed": perturb_seed,
                    "num_nodes": num_nodes,
                    "clean_canonical_edges": num_edges,
                    "clean_edge_index_columns": clean_columns,
                    "available_clean_nonedges": possible - num_edges,
                    "maximum_requested_rho": max(RANDOM_RHOS),
                    "maximum_requested_edges": max_budget,
                    "perturbed_canonical_edges_at_maximum": num_edges + max_budget,
                    "perturbed_edge_index_columns_at_maximum": clean_columns + 2 * max_budget,
                    "sampled_ordered_edges": len(ordered),
                    "capacity_sufficient": len(ordered) == max_budget,
                    "edge_list_path": str(pool_path.relative_to(output_root)),
                    "edge_list_sha256": pool_sha,
                    "edge_artifact_sha256": pool_artifact_sha,
                }
            )
            for rho in RANDOM_RHOS:
                budget = budget_for_rho(rho, num_edges)
                prefix = ordered[:budget]
                rho_path = output_root / "random_noise/perturbation_edges" / f"{dataset}_pseed{perturb_seed}_rho{rho:.2f}.npy"
                rho_sha = _write_edge_artifact(rho_path, prefix)
                record = {
                    "dataset": dataset,
                    "perturbation_type": "random_structural_noise",
                    "perturbation_seed": perturb_seed,
                    "rho": rho,
                    "clean_canonical_edges": num_edges,
                    "clean_edge_index_columns": clean_columns,
                    "injected_edge_count": budget,
                    "perturbed_canonical_edges": num_edges + budget,
                    "perturbed_edge_index_columns": clean_columns + 2 * budget,
                    "edge_list_path": str(rho_path.relative_to(output_root)),
                    "edge_list_sha256": rho_sha,
                    "edge_artifact_sha256": sha256_file(rho_path),
                    "ordered_maxrho_pool_path": str(pool_path.relative_to(output_root)),
                    "ordered_maxrho_pool_sha256": pool_sha,
                    "raw_feature_source": None,
                    "category_threshold": None,
                    "candidate_pool_size": 0,
                    "git_commit": _git_commit(root),
                }
                random_manifest.append(record)
                for model, label in RANDOM_MODELS:
                    for model_seed in MODEL_SEEDS:
                        random_plan.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "model_label": label,
                                "model_seed": model_seed,
                                "perturbation_type": "random_structural_noise",
                                "perturbation_seed": perturb_seed,
                                "rho": rho,
                                "formal_K": FORMAL_K[dataset],
                                "clean_canonical_edges": num_edges,
                                "clean_edge_index_columns": clean_columns,
                                "injected_edge_count": budget,
                                "perturbed_canonical_edges": num_edges + budget,
                                "perturbed_edge_index_columns": clean_columns + 2 * budget,
                                "checkpoint_path": _checkpoint_path(root, dataset, model, model_seed),
                                "edge_artifact": str(rho_path.relative_to(output_root)),
                                "edge_sha256": rho_sha,
                                "edge_artifact_sha256": sha256_file(rho_path),
                                "expected_clean_accuracy": _audit_metric(checkpoint_rows, dataset, model, model_seed, "clean_test_accuracy"),
                                "expected_clean_macro_f1": _audit_metric(checkpoint_rows, dataset, model, model_seed, "clean_test_macro_f1"),
                            }
                        )

    # Second pass: independently sample raw-feature nonedges and audit category capacity.
    candidate_records: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in DATASETS:
        _, data, clean_edges, raw_sources = loaded[dataset]
        num_nodes = int(data.num_nodes)
        num_edges = int(len(clean_edges))
        for perturb_seed in PERTURBATION_SEEDS:
            rng = np.random.default_rng(np.random.SeedSequence([perturb_seed, _dataset_code(dataset), 200]))
            candidates = sample_true_nonedges(num_nodes, clean_edges, candidate_pool_size, rng)
            edge_tensor = torch.as_tensor(candidates.T.copy(), dtype=torch.long)
            text_cos = _edge_cosines(data.x_t, edge_tensor, device, EDGE_COSINE_CHUNK_SIZE)
            visual_cos = _edge_cosines(data.x_i, edge_tensor, device, EDGE_COSINE_CHUNK_SIZE)
            rank_text = _average_rank_cdf(text_cos)
            rank_visual = _average_rank_cdf(visual_cos)
            c_text, c_visual = conflict_category_masks(rank_text, rank_visual)
            text_edges = candidates[c_text]
            visual_edges = candidates[c_visual]
            capacity = maximum_balanced_capacity(len(text_edges), len(visual_edges))
            order_seed = int(np.random.SeedSequence([perturb_seed, _dataset_code(dataset), 300]).generate_state(1)[0])
            ordered, categories = build_balanced_nested_order(text_edges, visual_edges, order_seed)
            if len(ordered) != capacity:
                raise AssertionError("balanced conflict order does not attain the computed capacity")
            if np.any(ordered[:, 0] >= ordered[:, 1]) or len(np.unique(ordered, axis=0)) != len(ordered):
                raise AssertionError("conflict edges are not unique canonical non-self pairs")
            candidate_path = output_root / "semantic_conflict/candidate_pools" / f"{dataset}_pseed{perturb_seed}_pool{candidate_pool_size}.npz"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                candidate_path,
                candidate_edges=candidates,
                cosine_text=np.asarray(text_cos, dtype=np.float32),
                cosine_visual=np.asarray(visual_cos, dtype=np.float32),
                rank_text=np.asarray(rank_text, dtype=np.float32),
                rank_visual=np.asarray(rank_visual, dtype=np.float32),
                category=np.where(c_text, 1, np.where(c_visual, 2, 0)).astype(np.uint8),
            )
            candidate_hash = sha256_file(candidate_path)
            row = {
                "dataset": dataset,
                "perturbation_seed": perturb_seed,
                "clean_canonical_edges": num_edges,
                "clean_edge_index_columns": int(data.edge_index.size(1)),
                "num_nodes": num_nodes,
                "requested_rhos": ";".join(f"{rho:.2f}" for rho in CONFLICT_RHOS),
                "requested_edge_counts": ";".join(str(budget_for_rho(rho, num_edges)) for rho in CONFLICT_RHOS),
                "candidate_pool_size": candidate_pool_size,
                "candidate_pool_path": str(candidate_path.relative_to(output_root)),
                "candidate_pool_sha256": candidate_hash,
                "count_C_T": int(c_text.sum()),
                "count_C_V": int(c_visual.sum()),
                "max_balanced_injectable_edges": capacity,
                "max_individually_feasible_rho": max(
                    (rho for rho in CONFLICT_RHOS if budget_for_rho(rho, num_edges) <= capacity),
                    default=0.0,
                ),
                "text_cosine_min": float(np.min(text_cos)),
                "text_cosine_median": float(np.median(text_cos)),
                "text_cosine_max": float(np.max(text_cos)),
                "visual_cosine_min": float(np.min(visual_cos)),
                "visual_cosine_median": float(np.median(visual_cos)),
                "visual_cosine_max": float(np.max(visual_cos)),
                "text_feature_path": raw_sources["text_feature_path"],
                "text_feature_sha256": raw_sources["text_feature_sha256"],
                "visual_feature_path": raw_sources["visual_feature_path"],
                "visual_feature_sha256": raw_sources["visual_feature_sha256"],
            }
            conflict_capacity_rows.append(row)
            candidate_records[(dataset, perturb_seed)] = {
                "row": row,
                "edges": ordered,
                "classes": categories,
                "raw_sources": raw_sources,
                "candidate_path": candidate_path,
            }
            common_caps.append(capacity)
            common_edges.append(num_edges)

    common_schedule = _common_conflict_schedule(common_caps, common_edges)
    # Save the selected common schedule in every capacity row after global audit.
    for row in conflict_capacity_rows:
        row["final_common_rho_schedule"] = ";".join(f"{rho:.2f}" for rho in common_schedule)
        row["final_common_max_rho"] = max(common_schedule)

    for (dataset, perturb_seed), payload in candidate_records.items():
        _, data, clean_edges, raw_sources = loaded[dataset]
        num_edges = len(clean_edges)
        ordered = payload["edges"]
        categories = payload["classes"]
        text_count_available = int(np.sum(categories == 1))
        visual_count_available = int(np.sum(categories == 2))
        for rho in common_schedule:
            budget = budget_for_rho(rho, num_edges)
            prefix = ordered[:budget]
            prefix_classes = categories[:budget]
            n_text = int(np.sum(prefix_classes == 1))
            n_visual = int(np.sum(prefix_classes == 2))
            if abs(n_text - n_visual) > 1:
                raise AssertionError("conflict injection is not balanced")
            if (
                n_text + n_visual != budget
                or n_text > text_count_available
                or n_visual > visual_count_available
            ):
                raise AssertionError("conflict prefix exceeds its category capacities")
            edge_path = output_root / "semantic_conflict/perturbation_edges" / f"{dataset}_pseed{perturb_seed}_rho{rho:.2f}.npz"
            edge_sha = _write_edge_artifact(edge_path, prefix, classes=prefix_classes)
            candidate_path = payload["candidate_path"]
            record = {
                "dataset": dataset,
                "perturbation_type": "semantic_conflict_edge_injection",
                "perturbation_seed": perturb_seed,
                "rho": rho,
                "clean_canonical_edges": num_edges,
                "clean_edge_index_columns": int(data.edge_index.size(1)),
                "injected_edge_count": budget,
                "perturbed_canonical_edges": num_edges + budget,
                "perturbed_edge_index_columns": int(data.edge_index.size(1)) + 2 * budget,
                "injected_C_T_count": n_text,
                "injected_C_V_count": n_visual,
                "edge_list_path": str(edge_path.relative_to(output_root)),
                "edge_list_sha256": edge_sha,
                "edge_artifact_sha256": sha256_file(edge_path),
                "candidate_pool_path": str(candidate_path.relative_to(output_root)),
                "candidate_pool_size": candidate_pool_size,
                "candidate_pool_sha256": sha256_file(candidate_path),
                "raw_feature_source": json.dumps(
                    {
                        "text_path": raw_sources["text_feature_path"],
                        "text_sha256": raw_sources["text_feature_sha256"],
                        "visual_path": raw_sources["visual_feature_path"],
                        "visual_sha256": raw_sources["visual_feature_sha256"],
                    },
                    sort_keys=True,
                ),
                "category_threshold": "C_T: rank_text>=0.75 and rank_visual<=0.25; C_V: rank_visual>=0.75 and rank_text<=0.25; E0-A average-tie empirical ranks",
                "git_commit": _git_commit(root),
            }
            conflict_manifest.append(record)
            for model, label in CONFLICT_MODELS:
                for model_seed in MODEL_SEEDS:
                    conflict_plan.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "model_label": label,
                            "model_seed": model_seed,
                            "perturbation_type": "semantic_conflict_edge_injection",
                            "perturbation_seed": perturb_seed,
                            "rho": rho,
                            "formal_K": FORMAL_K[dataset],
                            "clean_canonical_edges": num_edges,
                            "clean_edge_index_columns": int(data.edge_index.size(1)),
                            "injected_edge_count": budget,
                            "perturbed_canonical_edges": num_edges + budget,
                            "perturbed_edge_index_columns": int(data.edge_index.size(1)) + 2 * budget,
                            "checkpoint_path": _checkpoint_path(root, dataset, model, model_seed),
                            "edge_artifact": str(edge_path.relative_to(output_root)),
                            "edge_sha256": edge_sha,
                            "edge_artifact_sha256": sha256_file(edge_path),
                            "expected_clean_accuracy": _audit_metric(checkpoint_rows, dataset, model, model_seed, "clean_test_accuracy"),
                            "expected_clean_macro_f1": _audit_metric(checkpoint_rows, dataset, model, model_seed, "clean_test_macro_f1"),
                        }
                    )

    validate_shared_plan_artifacts(random_plan)
    validate_shared_plan_artifacts(conflict_plan)
    write_csv(output_root / "random_noise/perturbation_capacity.csv", random_capacity_rows)
    write_csv(output_root / "random_noise/dry_run_plan.csv", random_plan)
    write_csv(output_root / "semantic_conflict/candidate_capacity.csv", conflict_capacity_rows)
    write_csv(output_root / "semantic_conflict/dry_run_plan.csv", conflict_plan)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(root),
        "scope": {"datasets": list(DATASETS), "formal_K": FORMAL_K, "model_seeds": list(MODEL_SEEDS)},
        "perturbation_seeds": list(PERTURBATION_SEEDS),
        "random_noise_rhos": list(RANDOM_RHOS),
        "semantic_conflict_preregistered_rhos": list(CONFLICT_RHOS),
        "semantic_conflict_final_common_rhos": list(common_schedule),
        "candidate_pool_size_per_dataset_seed": candidate_pool_size,
        "category_thresholds": {"high": RANK_HIGH, "low": RANK_LOW, "rank_method": "scripts/run_mopf_e0a_edge_semantic_discrepancy.py::_average_rank_cdf"},
        "cosine_method": "scripts/run_mopf_e0a_edge_semantic_discrepancy.py::_edge_cosines on original frozen raw features; chunked",
        "budget_rounding": "floor(rho*|E| + 0.5)",
        "edge_generation_invariants": {
            "canonical_pair": "0 <= u < v < num_nodes",
            "exclude_clean_physical_support": True,
            "conflict_class_counts_differ_by_at_most": 1,
            "model_independent_shared_graph": True,
            "directed_model_edge_representation": "append each injected pair in both orientations; preserve all clean edge_index columns",
        },
        "graph_summary": graph_summary,
        "dip_provenance_review": {
            "selected_family": "outputs/paper_nc_final_fixed_v1/{Movies,Grocery}/dip/seed{42,43,44}/best.pt",
            "selection_rule": "unique target-dataset NC run_record with protocol unified_full_graph_nc_v1; best_val_accuracy; fixed split and passing health checks",
            "git_provenance": "paper_nc_final_fixed_v1 run records do not record a Git commit; checkpoint and resolved-config SHA256 values are recorded in checkpoint_audit.csv",
            "rejected_dip_candidates": _other_dip_candidates(root),
            "unambiguous": True,
        },
        "random_noise_realizations": random_manifest,
        "semantic_conflict_realizations": conflict_manifest,
        "planned_evaluation_rows": {"random_noise": len(random_plan), "semantic_conflict": len(conflict_plan), "total": len(random_plan) + len(conflict_plan)},
        "execution_status": "NOT_RUN; perturbations prepared and capacity audited only",
    }
    (output_root / "perturbation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return checkpoint_rows, random_capacity_rows, random_plan, conflict_capacity_rows, {
        "conflict_plan": conflict_plan,
        "manifest": manifest,
    }


def _checkpoint_path(root: Path, dataset: str, model: str, seed: int) -> str:
    if model in {"mopf", "dip"}:
        return str((root / "outputs/paper_nc_final_fixed_v1" / dataset / model / f"seed{seed}/best.pt").resolve())
    return str((root / "outputs/core_story_ablation/nc" / dataset / model / f"seed{seed}/best.pt").resolve())


def _audit_metric(rows: list[dict[str, Any]], dataset: str, model: str, seed: int, key: str) -> float:
    for row in rows:
        if row["dataset"] == dataset and row["model"] == model and int(row["model_seed"]) == seed:
            return float(row[key])
    raise KeyError(f"missing checkpoint audit row {dataset}/{model}/seed{seed}")


def _git_commit(root: Path) -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not_available"


def _other_dip_candidates(root: Path) -> list[dict[str, str]]:
    rows = []
    for path in (root / "outputs").rglob("best.pt"):
        if path.parent.parent.name != "dip" or "paper_nc_final_fixed_v1" in path.parts:
            continue
        record_path = path.parent / "run_record.json"
        record = _json(record_path) if record_path.is_file() else {}
        in_lp_family = any("lp" in part.lower() for part in path.parts)
        task = record.get("task", "lp" if in_lp_family else "not_recorded")
        dataset = record.get("dataset", path.parents[2].name if len(path.parents) > 2 else "not_recorded")
        protocol = record.get("protocol", "not_recorded")
        rows.append(
            {
                "path": str(path.resolve()),
                "task": str(task),
                "protocol": str(protocol),
                "dataset": str(dataset),
                "rejection_reason": "unrelated cloth-copurchase LP checkpoint" if (
                    task == "lp" or dataset == "cloth-copurchase"
                ) else "not a target Movies/Grocery unified_full_graph_nc_v1 run" if (
                    task != "nc"
                    or protocol != "unified_full_graph_nc_v1"
                    or dataset not in DATASETS
                ) else "outside selected authoritative family; review required",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/robustness_analysis")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    device = torch.device(args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_rows, random_capacity, random_plan, conflict_capacity, extra = _make_plans_and_perturbations(
        args.root.resolve(), output_root, device
    )
    print(f"Checkpoint audit rows: {len(checkpoint_rows)}; all PASS")
    print(f"Random-noise capacity rows: {len(random_capacity)}; planned evaluations: {len(random_plan)}")
    print(f"Semantic capacity rows: {len(conflict_capacity)}")
    print(f"Common conflict schedule: {extra['manifest']['semantic_conflict_final_common_rhos']}")
    print(f"Semantic planned evaluations: {len(extra['conflict_plan'])}")
    print(f"Total formal evaluation rows (not executed): {extra['manifest']['planned_evaluation_rows']['total']}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
