#!/usr/bin/env python3
"""E0-D: local relation condition versus frozen multi-hop utilization.

This script is an analysis-only bridge diagnostic over the exact frozen U2-C
C1 checkpoints used by E0-C/U3-A.  It reuses E0-C's contribution-profile
artifacts and performs one no-gradient predecessor forward per checkpoint
only because E0-C did not export the Stage-I edge weights.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.u3a import contribution_profile  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from scripts.run_mopf_e0c_semantic_retention_heterogeneity import (  # noqa: E402
    DATASET_ORDER,
    FORMAL_K,
    REQUIRED_SEEDS,
    U3A_MASTER,
    _compose_c1_cfg,
    _model_state_digest,
    _sha256,
    _validate_model_config,
    _validate_source_records,
)
from src.models import build_model  # noqa: E402


E0C_ROOT = REPO_ROOT / "outputs/e0_empirical_motivation/multihop_utilization"
OUTPUT_ROOT = REPO_ROOT / "outputs/e0_empirical_motivation/relation_utilization_bridge"
REVERSE_TOLERANCE = 1e-6
E0C_REPRODUCTION_TOLERANCE = 5e-6
SIGN_TOLERANCE = 1e-12


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("Association statistics require a non-empty finite array")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        raise RuntimeError(f"Invalid association sample size: {x.size} and {y.size}")
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.all():
        raise RuntimeError("Non-finite values reached association calculation")
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    degenerate = bool(np.std(rx) <= 0.0 or np.std(ry) <= 0.0)
    return (0.0 if degenerate else _pearson(rx, ry), degenerate)


def _partial_spearman_degree(
    condition: np.ndarray,
    response_order: np.ndarray,
    log_degree: np.ndarray,
) -> tuple[float, bool]:
    """Fixed rank-then-OLS residual Pearson definition from the protocol."""

    x = rankdata(np.asarray(condition, dtype=np.float64), method="average")
    y = rankdata(np.asarray(response_order, dtype=np.float64), method="average")
    z = rankdata(np.asarray(log_degree, dtype=np.float64), method="average")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise RuntimeError("Non-finite rank values in partial Spearman")
    design = np.column_stack([np.ones(z.size, dtype=np.float64), z])
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    degenerate = bool(np.std(residual_x) <= 0.0 or np.std(residual_y) <= 0.0)
    return (0.0 if degenerate else _pearson(residual_x, residual_y), degenerate)


def _sign(value: float) -> str:
    if value > SIGN_TOLERANCE:
        return "positive"
    if value < -SIGN_TOLERANCE:
        return "negative"
    return "zero"


def _load_e0c_artifact(dataset: str, seed: int) -> dict[str, np.ndarray]:
    path = E0C_ROOT / dataset / f"seed{seed}" / "node_multihop_utilization.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen E0-C artifact: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        required = {
            "node_id",
            "p_text",
            "p_visual",
            "normalized_order_text",
            "normalized_order_visual",
        }
        missing = required.difference(loaded.files)
        if missing:
            raise RuntimeError(f"E0-C artifact {path} is missing {sorted(missing)}")
        return {name: loaded[name].copy() for name in loaded.files}


def _unique_physical_relations(
    edge_index: torch.Tensor,
    weight_text: torch.Tensor,
    weight_visual: torch.Tensor,
    num_nodes: int,
) -> dict[str, Any]:
    """Canonicalize undirected physical relations and audit reverse weights."""

    edge_array = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    text_array = weight_text.detach().cpu().numpy().astype(np.float64, copy=False)
    visual_array = weight_visual.detach().cpu().numpy().astype(np.float64, copy=False)
    if edge_array.shape[0] != 2 or edge_array.shape[1] != text_array.size or text_array.size != visual_array.size:
        raise RuntimeError("Edge index and learned relation-weight shapes do not match")
    if not (np.isfinite(text_array).all() and np.isfinite(visual_array).all()):
        raise RuntimeError("Non-finite learned relation weights")

    oriented: dict[tuple[int, int], list[tuple[float, float]]] = {}
    self_loop_count = 0
    for src, dst, w_text, w_visual in zip(
        edge_array[0], edge_array[1], text_array, visual_array
    ):
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            self_loop_count += 1
            continue
        if not (0 <= src_i < num_nodes and 0 <= dst_i < num_nodes):
            raise RuntimeError(f"Out-of-range physical edge ({src_i}, {dst_i})")
        oriented.setdefault((src_i, dst_i), []).append((float(w_text), float(w_visual)))

    canonical_keys = sorted({(min(src, dst), max(src, dst)) for src, dst in oriented})
    relation_src: list[int] = []
    relation_dst: list[int] = []
    relation_text: list[float] = []
    relation_visual: list[float] = []
    reverse_diffs_text: list[float] = []
    reverse_diffs_visual: list[float] = []
    missing_reverse = 0
    duplicate_oriented_records = 0
    for left, right in canonical_keys:
        forward = oriented.get((left, right), [])
        reverse = oriented.get((right, left), [])
        duplicate_oriented_records += max(len(forward) - 1, 0) + max(len(reverse) - 1, 0)
        if forward and reverse:
            forward_text = float(np.mean([item[0] for item in forward]))
            reverse_text = float(np.mean([item[0] for item in reverse]))
            forward_visual = float(np.mean([item[1] for item in forward]))
            reverse_visual = float(np.mean([item[1] for item in reverse]))
            reverse_diffs_text.append(abs(forward_text - reverse_text))
            reverse_diffs_visual.append(abs(forward_visual - reverse_visual))
            physical_text = 0.5 * (forward_text + reverse_text)
            physical_visual = 0.5 * (forward_visual + reverse_visual)
        elif forward:
            missing_reverse += 1
            physical_text = float(np.mean([item[0] for item in forward]))
            physical_visual = float(np.mean([item[1] for item in forward]))
        elif reverse:
            missing_reverse += 1
            physical_text = float(np.mean([item[0] for item in reverse]))
            physical_visual = float(np.mean([item[1] for item in reverse]))
        else:  # pragma: no cover - canonical_keys makes this unreachable.
            raise RuntimeError("Empty canonical relation group")
        relation_src.append(left)
        relation_dst.append(right)
        relation_text.append(physical_text)
        relation_visual.append(physical_visual)

    reverse_diffs_text_array = np.asarray(reverse_diffs_text, dtype=np.float64)
    reverse_diffs_visual_array = np.asarray(reverse_diffs_visual, dtype=np.float64)
    return {
        "src": np.asarray(relation_src, dtype=np.int64),
        "dst": np.asarray(relation_dst, dtype=np.int64),
        "weight_text": np.asarray(relation_text, dtype=np.float64),
        "weight_visual": np.asarray(relation_visual, dtype=np.float64),
        "audit": {
            "input_edge_records": int(edge_array.shape[1]),
            "self_loop_records_excluded": int(self_loop_count),
            "unique_physical_edges": len(canonical_keys),
            "unique_oriented_pairs": len(oriented),
            "duplicate_oriented_records_collapsed": int(duplicate_oriented_records),
            "reverse_pairs_compared": int(reverse_diffs_text_array.size),
            "missing_reverse_relation_count": int(missing_reverse),
            "reverse_weight_tolerance": REVERSE_TOLERANCE,
            "reverse_weight_max_abs_diff_text": float(reverse_diffs_text_array.max()) if reverse_diffs_text_array.size else 0.0,
            "reverse_weight_mean_abs_diff_text": float(reverse_diffs_text_array.mean()) if reverse_diffs_text_array.size else 0.0,
            "reverse_weight_max_abs_diff_visual": float(reverse_diffs_visual_array.max()) if reverse_diffs_visual_array.size else 0.0,
            "reverse_weight_mean_abs_diff_visual": float(reverse_diffs_visual_array.mean()) if reverse_diffs_visual_array.size else 0.0,
            "reverse_weight_symmetry_within_tolerance_text": bool(
                not reverse_diffs_text_array.size or reverse_diffs_text_array.max() <= REVERSE_TOLERANCE
            ),
            "reverse_weight_symmetry_within_tolerance_visual": bool(
                not reverse_diffs_visual_array.size or reverse_diffs_visual_array.max() <= REVERSE_TOLERANCE
            ),
            "one_physical_relation_counted_once": True,
            "self_loops_included": False,
        },
    }


def _incident_condition(
    relation_src: np.ndarray,
    relation_dst: np.ndarray,
    relation_weight: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    degree = np.zeros(num_nodes, dtype=np.int64)
    incident_sum = np.zeros(num_nodes, dtype=np.float64)
    for src, dst, weight in zip(relation_src, relation_dst, relation_weight):
        src_i = int(src)
        dst_i = int(dst)
        degree[src_i] += 1
        degree[dst_i] += 1
        incident_sum[src_i] += float(weight)
        incident_sum[dst_i] += float(weight)
    condition = np.zeros(num_nodes, dtype=np.float64)
    connected = degree > 0
    condition[connected] = incident_sum[connected] / degree[connected]
    return condition, degree


def _quartile_ids(condition: np.ndarray, connected: np.ndarray) -> np.ndarray:
    """Assign average-rank condition values to fixed Q1--Q4 bins."""

    ranks = rankdata(condition[connected], method="average")
    n = ranks.size
    rank_fraction = ranks / float(n)
    quartiles = np.minimum(np.ceil(4.0 * rank_fraction).astype(np.int64), 4)
    return quartiles


def _association_payload(
    condition: np.ndarray,
    centered_condition: np.ndarray,
    response_order: np.ndarray,
    profile: np.ndarray,
    degree: np.ndarray,
    connected: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = connected & np.isfinite(condition) & np.isfinite(centered_condition) & np.isfinite(response_order)
    if not valid[connected].all() or not np.isfinite(profile).all():
        raise RuntimeError("Non-finite node statistic in connected population")
    if int(valid.sum()) < 3:
        raise RuntimeError("Too few connected nodes for association")

    raw_rho, raw_degenerate = _spearman(condition[valid], response_order[valid])
    centered_rho, centered_degenerate = _spearman(centered_condition[valid], response_order[valid])
    partial_rho, partial_degenerate = _partial_spearman_degree(
        condition[valid], response_order[valid], np.log1p(degree[valid])
    )
    degree_order, degree_order_degenerate = _spearman(np.log1p(degree[valid]), response_order[valid])
    degree_condition, degree_condition_degenerate = _spearman(np.log1p(degree[valid]), condition[valid])
    condition_ranks = rankdata(condition[valid], method="average")
    quartiles = _quartile_ids(condition, valid)
    order_valid = response_order[valid]
    profile_valid = profile[valid]
    quartile_rows: list[dict[str, Any]] = []
    for quartile in range(1, 5):
        mask = quartiles == quartile
        if not mask.any():
            raise RuntimeError(f"Empty fixed condition quartile Q{quartile}")
        q_order = order_valid[mask]
        q_profile = profile_valid[mask]
        row = {
            "quartile": f"Q{quartile}",
            "count": int(mask.sum()),
            "condition_mean": float(np.mean(condition[valid][mask])),
            "order_mean": float(np.mean(q_order)),
            "order_median": float(np.median(q_order)),
            "order_q25": float(np.quantile(q_order, 0.25)),
            "order_q75": float(np.quantile(q_order, 0.75)),
        }
        row.update({f"mean_profile_k{k}": float(q_profile[:, k].mean()) for k in range(profile.shape[1])})
        quartile_rows.append(row)

    association = {
        "sample_count": int(valid.sum()),
        "finite_connected_count": int(valid.sum()),
        "rho_raw": raw_rho,
        "rho_raw_centered_audit": centered_rho,
        "raw_centered_rank_max_abs_diff": float(
            np.max(np.abs(condition_ranks - rankdata(centered_condition[valid], method="average")))
        ),
        "rho_partial_degree": partial_rho,
        "rho_degree_order": degree_order,
        "rho_degree_condition": degree_condition,
        "abs_rho_partial": abs(partial_rho),
        "sign_partial": _sign(partial_rho),
        "degenerate_raw": raw_degenerate,
        "degenerate_centered": centered_degenerate,
        "degenerate_partial": partial_degenerate,
        "degenerate_degree_order": degree_order_degenerate,
        "degenerate_degree_condition": degree_condition_degenerate,
    }
    return association, quartile_rows


def _analyze_checkpoint(
    record: dict[str, Any], device: torch.device, output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    dataset = str(record["dataset"])
    seed = int(record["seed"])
    formal_k = FORMAL_K[dataset]
    e0c = _load_e0c_artifact(dataset, seed)
    checkpoint = Path(record["checkpoint"])
    checkpoint_sha_before = _sha256(checkpoint)
    if checkpoint_sha_before != record["checkpoint_sha256_before"]:
        raise RuntimeError(f"Source checkpoint changed before E0-D load: {checkpoint}")

    cfg = _compose_c1_cfg(dataset, seed, str(device))
    data = load_mag_data(cfg, "nc", seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg, payload["data_info"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    model_config = _validate_model_config(model, dataset, cfg)
    model_state_digest_before = _model_state_digest(model)
    with torch.inference_mode():
        components = model._encode_components(data.x.to(device), data.edge_index.to(device))
        profile_text = contribution_profile(components["eta_text"], components["states_text"])
        profile_visual = contribution_profile(components["eta_visual"], components["states_visual"])

    p_text = profile_text["probabilities"].detach().cpu().numpy().astype(np.float32, copy=False)
    p_visual = profile_visual["probabilities"].detach().cpu().numpy().astype(np.float32, copy=False)
    r_text = profile_text["normalized_response_order"].detach().cpu().numpy().astype(np.float32, copy=False)
    r_visual = profile_visual["normalized_response_order"].detach().cpu().numpy().astype(np.float32, copy=False)
    e0c_node_id = np.asarray(e0c["node_id"], dtype=np.int64)
    node_id = np.arange(int(data.num_nodes), dtype=np.int64)
    if not np.array_equal(e0c_node_id, node_id):
        raise RuntimeError(f"E0-C node IDs do not match loader node IDs for {dataset}/{seed}")
    e0c_p_diff_text = float(np.max(np.abs(p_text - e0c["p_text"])))
    e0c_p_diff_visual = float(np.max(np.abs(p_visual - e0c["p_visual"])))
    e0c_r_diff_text = float(np.max(np.abs(r_text - e0c["normalized_order_text"])))
    e0c_r_diff_visual = float(np.max(np.abs(r_visual - e0c["normalized_order_visual"])))
    if max(e0c_p_diff_text, e0c_p_diff_visual, e0c_r_diff_text, e0c_r_diff_visual) > E0C_REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"E0-C response/profile reproduction exceeded tolerance for {dataset}/{seed}: "
            f"p=({e0c_p_diff_text:.3e},{e0c_p_diff_visual:.3e}), "
            f"r=({e0c_r_diff_text:.3e},{e0c_r_diff_visual:.3e})"
        )

    edges = components["edges"]
    relations = _unique_physical_relations(
        data.edge_index,
        edges["w_t"],
        edges["w_v"],
        int(data.num_nodes),
    )
    condition_text, degree = _incident_condition(
        relations["src"], relations["dst"], relations["weight_text"], int(data.num_nodes)
    )
    condition_visual, visual_degree = _incident_condition(
        relations["src"], relations["dst"], relations["weight_visual"], int(data.num_nodes)
    )
    if not np.array_equal(degree, visual_degree):
        raise RuntimeError(f"Modality physical degree mismatch for {dataset}/{seed}")
    connected = degree > 0
    if not connected.any():
        raise RuntimeError(f"No connected nodes for {dataset}/{seed}")
    centered_text = condition_text - float(condition_text[connected].mean())
    centered_visual = condition_visual - float(condition_visual[connected].mean())

    assoc_text, quartiles_text = _association_payload(
        condition_text, centered_text, r_text, p_text, degree, connected
    )
    assoc_visual, quartiles_visual = _association_payload(
        condition_visual, centered_visual, r_visual, p_visual, degree, connected
    )
    for modality, association in (("Text", assoc_text), ("Visual", assoc_visual)):
        association.update(
            {
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "num_nodes": int(data.num_nodes),
                "physical_edges": int(relations["audit"]["unique_physical_edges"]),
                "connected_nodes": int(connected.sum()),
                "isolated_nodes": int((~connected).sum()),
                "formal_K": formal_k,
            }
        )

    physical_artifact = {
        "node_id": node_id,
        "physical_degree": degree,
        "log_degree": np.log1p(degree).astype(np.float32),
        "c_text": condition_text.astype(np.float32),
        "c_visual": condition_visual.astype(np.float32),
        "centered_c_text": centered_text.astype(np.float32),
        "centered_c_visual": centered_visual.astype(np.float32),
        "response_order_text": r_text.astype(np.float32),
        "response_order_visual": r_visual.astype(np.float32),
        "contribution_profile_text": p_text.astype(np.float32),
        "contribution_profile_visual": p_visual.astype(np.float32),
    }
    seed_dir = output_root / dataset / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(seed_dir / "node_relation_utilization_bridge.npz", **physical_artifact)

    model_state_digest_after = _model_state_digest(model)
    checkpoint_sha_after = _sha256(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError(f"Source checkpoint changed during E0-D forward: {checkpoint}")
    if model_state_digest_after != model_state_digest_before:
        raise RuntimeError(f"Model state changed during E0-D forward: {dataset}/{seed}")

    seed_summary = {
        "dataset": dataset,
        "seed": seed,
        "formal_K": formal_k,
        "num_nodes": int(data.num_nodes),
        "physical_edges_unique": int(relations["audit"]["unique_physical_edges"]),
        "connected_nodes": int(connected.sum()),
        "isolated_nodes": int((~connected).sum()),
        "connected_fraction": float(connected.mean()),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256_before": checkpoint_sha_before,
        "source_checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_selection": "frozen U2-C C1 best-validation checkpoint from U3-A source manifest",
        "model_config": model_config,
        "e0c_artifact_source": str(E0C_ROOT / dataset / f"seed{seed}" / "node_multihop_utilization.npz"),
        "e0c_reproduction": {
            "max_abs_probability_diff_text": e0c_p_diff_text,
            "max_abs_probability_diff_visual": e0c_p_diff_visual,
            "max_abs_response_order_diff_text": e0c_r_diff_text,
            "max_abs_response_order_diff_visual": e0c_r_diff_visual,
            "tolerance": E0C_REPRODUCTION_TOLERANCE,
            "within_tolerance": True,
        },
        "physical_graph_definition": {
            "source": "data.edge_index from project load_mag_data",
            "canonicalization": "(min(i,j), max(i,j)); self-loops excluded; duplicate oriented records averaged within orientation",
            "unique_physical_relation_weight": "mean of forward/reverse learned weights when both are present; single available orientation otherwise",
            "physical_degree": "unique physical neighbor count",
            "centered_condition_mean_population": "connected nodes only",
            "self_loops_included": False,
        },
        "reverse_weight_audit": relations["audit"],
        "node_artifact": {
            "path": str(seed_dir / "node_relation_utilization_bridge.npz"),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in physical_artifact.items()
            },
        },
        "finite_status": bool(all(np.isfinite(value).all() for value in physical_artifact.values())),
        "association": {"Text": assoc_text, "Visual": assoc_visual},
        "quartiles": {"Text": quartiles_text, "Visual": quartiles_visual},
        "model_state_digest_before_forward": model_state_digest_before,
        "model_state_digest_after_forward": model_state_digest_after,
        "labels_accessed": False,
        "validation_test_metrics_accessed": False,
        "optimizer_step": False,
        "backward": False,
        "checkpoint_written": False,
    }
    _write_json(seed_dir / "summary.json", seed_summary)

    del components, profile_text, profile_visual, model, payload, data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return assoc_text, quartiles_text, {
        "association": assoc_visual,
        "quartiles": quartiles_visual,
        "seed_summary": seed_summary,
    }, physical_artifact


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_group(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    signs = [str(row["sign_partial"]) for row in rows]
    counts = {sign: signs.count(sign) for sign in ("positive", "negative", "zero")}
    if counts["positive"] == counts["negative"] == 0:
        dominant_sign = "zero"
        same_sign_fraction = 1.0
    else:
        dominant_sign = "positive" if counts["positive"] >= counts["negative"] else "negative"
        same_sign_fraction = counts[dominant_sign] / float(len(signs))
    result = {
        "dataset": rows[0]["dataset"],
        "modality": rows[0]["modality"],
        "seed_count": len(rows),
        "rho_partial_degree_mean": float(values.mean()),
        "rho_partial_degree_std": float(values.std()),
        "rho_partial_degree_median": float(np.median(values)),
        "rho_partial_degree_min": float(values.min()),
        "rho_partial_degree_max": float(values.max()),
        "same_sign_fraction_across_3_seeds": float(same_sign_fraction),
        "dominant_sign": dominant_sign,
        "positive_seed_count": counts["positive"],
        "negative_seed_count": counts["negative"],
        "zero_seed_count": counts["zero"],
    }
    return result


def _plot_quartiles(
    quartile_rows: list[dict[str, Any]], dataset_order: list[str], output_root: Path
) -> None:
    fig, axes = plt.subplots(1, len(dataset_order), figsize=(17, 3.7), sharey=True)
    if len(dataset_order) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, dataset_order):
        for modality, color in (("Text", "#2f6fbd"), ("Visual", "#3b9b62")):
            rows = [row for row in quartile_rows if row["dataset"] == dataset and row["modality"] == modality]
            rows.sort(key=lambda row: (int(row["seed"]), int(row["quartile"][1])))
            per_quartile = {
                q: np.asarray([float(row["order_mean"]) for row in rows if row["quartile"] == f"Q{q}"], dtype=float)
                for q in range(1, 5)
            }
            x = np.arange(1, 5, dtype=float)
            means = np.asarray([per_quartile[q].mean() for q in range(1, 5)])
            lows = np.asarray([per_quartile[q].min() for q in range(1, 5)])
            highs = np.asarray([per_quartile[q].max() for q in range(1, 5)])
            ax.plot(x, means, marker="o", linewidth=1.8, color=color, label=modality)
            ax.fill_between(x, lows, highs, color=color, alpha=0.14)
        ax.set_title(dataset)
        ax.set_xticks(np.arange(1, 5), ["Q1", "Q2", "Q3", "Q4"])
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean normalized contribution-weighted response order")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("E0-D: local relation-condition quartiles vs. multi-hop utilization")
    fig.tight_layout()
    fig.savefig(output_root / "relation_condition_response_order_quartiles.png", dpi=180)
    fig.savefig(output_root / "relation_condition_response_order_quartiles.pdf")
    plt.close(fig)


def _plot_partial(
    association_rows: list[dict[str, Any]], dataset_order: list[str], output_root: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    positions = np.arange(1, len(dataset_order) + 1, dtype=float)
    for modality, color, offset in (("Text", "#2f6fbd", -0.16), ("Visual", "#3b9b62", 0.16)):
        for index, dataset in enumerate(dataset_order, start=1):
            rows = [row for row in association_rows if row["dataset"] == dataset and row["modality"] == modality]
            values = np.asarray([float(row["rho_partial_degree"]) for row in rows], dtype=float)
            ax.scatter(np.full(values.size, index + offset), values, color=color, s=28, zorder=3)
            ax.errorbar(
                index + offset,
                float(values.mean()),
                yerr=float(values.std()),
                fmt="o",
                color=color,
                markerfacecolor="white",
                markeredgewidth=1.0,
                capsize=4,
                zorder=4,
            )
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions, dataset_order)
    ax.set_ylabel("Partial Spearman rho (degree-controlled)")
    ax.set_title("E0-D: local relation condition vs. normalized response order")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(handles=[
        plt.Line2D([], [], color="#2f6fbd", marker="o", linewidth=1.8, label="Text"),
        plt.Line2D([], [], color="#3b9b62", marker="o", linewidth=1.8, label="Visual"),
        plt.Line2D([], [], marker="o", color="#111111", markerfacecolor="white", linestyle="None", label="Seed mean ± std"),
    ], frameon=False, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(output_root / "partial_correlation_summary.png", dpi=180)
    fig.savefig(output_root / "partial_correlation_summary.pdf")
    plt.close(fig)


def _run(args: argparse.Namespace) -> None:
    started = time.time()
    records, records_by_key = _validate_source_records()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")

    association_rows: list[dict[str, Any]] = []
    quartile_rows: list[dict[str, Any]] = []
    seed_summaries: dict[tuple[str, int], dict[str, Any]] = {}
    artifact_cache: dict[str, dict[str, np.ndarray]] = {}
    for dataset in DATASET_ORDER:
        for seed in REQUIRED_SEEDS:
            assoc_text, quart_text, visual_payload, artifact = _analyze_checkpoint(
                records_by_key[(dataset, seed)], device, output_root
            )
            assoc_visual = visual_payload["association"]
            quart_visual = visual_payload["quartiles"]
            for row in (assoc_text, assoc_visual):
                association_rows.append(row)
            for modality, rows in (("Text", quart_text), ("Visual", quart_visual)):
                for row in rows:
                    quartile_rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "modality": modality,
                        "formal_K": FORMAL_K[dataset],
                        **row,
                    })
            seed_summaries[(dataset, seed)] = visual_payload["seed_summary"]
            artifact_cache.setdefault(dataset, {})[str(seed)] = artifact
            print(f"[E0-D] completed {dataset} seed={seed}", flush=True)

    association_fields = [
        "dataset", "seed", "modality", "formal_K", "num_nodes", "physical_edges", "connected_nodes",
        "rho_raw", "rho_raw_centered_audit", "rho_partial_degree", "rho_degree_order", "rho_degree_condition",
        "abs_rho_partial", "sign_partial", "sample_count", "raw_centered_rank_max_abs_diff",
        "degenerate_raw", "degenerate_partial", "degenerate_degree_order", "degenerate_degree_condition",
    ]
    quartile_fields = [
        "dataset", "seed", "modality", "formal_K", "quartile", "count", "condition_mean",
        "order_mean", "order_median", "order_q25", "order_q75",
    ]
    max_k = max(FORMAL_K.values())
    quartile_fields.extend([f"mean_profile_k{k}" for k in range(max_k + 1)])
    _write_csv(output_root / "relation_utilization_association.csv", association_rows, association_fields)
    _write_csv(output_root / "relation_condition_quartiles.csv", quartile_rows, quartile_fields)

    dataset_modality_aggregation = []
    for dataset in DATASET_ORDER:
        for modality in ("Text", "Visual"):
            cells = [row for row in association_rows if row["dataset"] == dataset and row["modality"] == modality]
            dataset_modality_aggregation.append(_aggregate_group(cells, "rho_partial_degree"))
    partial_values = np.asarray([float(row["rho_partial_degree"]) for row in association_rows], dtype=np.float64)
    signs = [str(row["sign_partial"]) for row in association_rows]
    nonzero_signs = [sign for sign in signs if sign != "zero"]
    positive_fraction = signs.count("positive") / float(len(signs))
    negative_fraction = signs.count("negative") / float(len(signs))
    dominant_sign = "positive" if positive_fraction >= negative_fraction else "negative"
    master = {
        "experiment": "E0-D Local Relation Condition vs. Multi-Hop Utilization",
        "status": "complete",
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "datasets": DATASET_ORDER,
        "seeds": REQUIRED_SEEDS,
        "formal_K": FORMAL_K,
        "device": str(device),
        "source": {
            "u3a_authoritative_manifest": str(U3A_MASTER),
            "e0c_artifact_root": str(E0C_ROOT),
            "source_checkpoint_count": len(records),
            "source_checkpoint_selection": "same U2-C C1 best-validation checkpoints resolved and SHA-checked by E0-C/U3-A",
            "checkpoint_sha_verified_before_and_after_each_forward": True,
            "source_checkpoints": [
                {
                    "dataset": record["dataset"],
                    "seed": record["seed"],
                    "checkpoint": record["checkpoint"],
                    "checkpoint_sha256_before": record["checkpoint_sha256_before"],
                    "variant": "C1",
                }
                for record in [records_by_key[(dataset, seed)] for dataset in DATASET_ORDER for seed in REQUIRED_SEEDS]
            ],
        },
        "fixed_model_configuration": {
            "edge_weight_mode": "learned_diag_cos",
            "edge_weight_temperature": 0.35,
            "multihop_state_mode": "anchored",
            "multihop_response_mode": "cumulative",
            "multihop_anchor_alpha": 0.1,
            "use_transport_residual": False,
            "tcpr_enabled": False,
        },
        "response_order_definition": {
            "source": "reused E0-C node_multihop_utilization.npz and independently reproduced from src.analysis.u3a.contribution_profile",
            "definition": "r_i^m=sum_k(k/formal_K)*p_i,k^m",
            "p_definition": "p_i,k^m=||eta_i,k^m*S_i,k^m||_2 / sum_r ||eta_i,r^m*S_i,r^m||_2",
            "reproduction_tolerance": E0C_REPRODUCTION_TOLERANCE,
        },
        "local_relation_condition_definition": {
            "weight_source": "frozen predecessor components['edges']['w_t'/'w_v']",
            "condition": "mean learned relation weight over unique physical neighbors",
            "physical_edge_unit": "canonical (min(i,j),max(i,j)) relation",
            "self_loops_included": False,
            "degree_control": "rank(log1p(unique_physical_degree))",
            "partial_spearman": "Pearson residual correlation after OLS of average-rank condition and average-rank response on average-rank log1p degree",
            "raw_centered_rank_audit_max_abs": float(max(row["raw_centered_rank_max_abs_diff"] for row in association_rows)),
        },
        "aggregation": {
            "primary_cell_count": len(association_rows),
            "dataset_modality_cell_count": len(dataset_modality_aggregation),
            "per_seed_first": True,
            "pooled_3xN_inference": False,
            "dataset_modality": dataset_modality_aggregation,
            "overall": {
                "median_absolute_partial_rho_30_cells": float(np.median(np.abs(partial_values))),
                "mean_absolute_partial_rho_30_cells": float(np.mean(np.abs(partial_values))),
                "positive_sign_fraction": positive_fraction,
                "negative_sign_fraction": negative_fraction,
                "zero_sign_fraction": signs.count("zero") / float(len(signs)),
                "dominant_sign": dominant_sign if nonzero_signs else "zero",
            },
        },
        "audit": {
            "tcpr_disabled": True,
            "tcpr_enabled": False,
            "optimizer_step": False,
            "backward": False,
            "training_started": False,
            "labels_used": False,
            "validation_test_metrics_used": False,
            "transport_interventions_run": False,
            "checkpoint_written": False,
            "final_model_configuration_modified": False,
            "node_statistics_finite": True,
            "reverse_weight_symmetry_tolerance": REVERSE_TOLERANCE,
        },
        "reproducibility": {
            "exact_command": f"MPLCONFIGDIR=/tmp/mopf_e0d_mpl PYTHONPATH=src conda run --no-capture-output -n yhf_env python scripts/run_mopf_e0d_relation_utilization_bridge.py --device {device}",
            "runtime_seconds": float(time.time() - started),
            "output_root": str(output_root),
        },
        "outputs": {
            "association_csv": str(output_root / "relation_utilization_association.csv"),
            "quartile_csv": str(output_root / "relation_condition_quartiles.csv"),
            "quartile_figure_png": str(output_root / "relation_condition_response_order_quartiles.png"),
            "partial_figure_png": str(output_root / "partial_correlation_summary.png"),
        },
    }
    _write_json(output_root / "relation_utilization_master_summary.json", master)
    _plot_quartiles(quartile_rows, DATASET_ORDER, output_root)
    _plot_partial(association_rows, DATASET_ORDER, output_root)
    # Re-write after figures so runtime/output metadata is final and auditable.
    master["reproducibility"]["runtime_seconds"] = float(time.time() - started)
    _write_json(output_root / "relation_utilization_master_summary.json", master)
    print(json.dumps({"output_root": str(output_root), "runtime_seconds": master["reproducibility"]["runtime_seconds"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
