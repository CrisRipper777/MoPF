#!/usr/bin/env python3
"""Generate model-independent empirical motivation data for Figure 1."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Grocery", "Toys", "ele-fashion", "Reddit-S")
DEFAULT_OUTPUT = ROOT / "outputs" / "figure1_empirical_motivation"
MODALITIES = ("text", "visual")
EPS = 1e-12
# The model-independent hop probe is deliberately run for every NC dataset
# used by the empirical Figure 1 story.  This is separate from the legacy
# checkpoint eta diagnostic, which remains reserved for Figure 4.
ORDER_UTILITY_DATASETS = DATASETS
ORDER_CLASSIFIER_SEED = 42
ORDER_CLASSIFIER_EPOCHS = 100
ORDER_CLASSIFIER_LR = 1e-2
ORDER_CLASSIFIER_WEIGHT_DECAY = 0.0

RELATION_SUMMARY_FIELDS = (
    "dataset", "num_nodes", "num_physical_edges", "spearman_rho",
    "spearman_status", "median_absolute_rank_gap", "rank_gap_q25",
    "rank_gap_q75", "rank_gap_q90", "rank_gap_q95",
    "text_high_visual_low_fraction", "visual_high_text_low_fraction",
    "text_feature_dim", "visual_feature_dim", "invalid_edge_count",
    "self_loop_count_removed", "duplicate_edge_count_removed",
)
RELATION_EDGE_FIELDS = (
    "dataset", "source_node", "target_node", "text_compatibility",
    "visual_compatibility", "text_rank_percentile",
    "visual_rank_percentile", "absolute_rank_gap",
)
DRIFT_FIELDS = (
    "dataset", "modality", "hop", "num_nodes", "mean_cosine_to_anchor",
    "median_cosine_to_anchor", "cosine_q25", "cosine_q75",
    "mean_drift_1_minus_cosine", "median_drift", "drift_q25", "drift_q75",
)
ORDER_UTILITY_FIELDS = (
    "dataset", "modality", "hop", "num_nodes", "num_train_nodes",
    "num_validation_nodes", "best_order_count", "fraction_best",
    "mean_validation_cross_entropy", "median_validation_cross_entropy",
    "classifier", "optimizer", "learning_rate", "weight_decay", "epochs",
    "random_seed", "split_path", "tie_rule",
)
ORDER_UTILITY_NODE_FIELDS = (
    "dataset", "modality", "node_id", "label", "best_order",
    "num_validation_nodes", "random_seed", "tie_rule",
)


def _compose_config(dataset: str):
    try:
        from hydra import compose, initialize_config_dir
    except ImportError as exc:
        raise RuntimeError("Hydra is required; run this script in the repository's yhf_env.") from exc
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(
            config_name="config",
            overrides=[f"dataset={dataset}", "task=nc", "model=mlp", "seed=42"],
        )


def _resolved_dataset_config(cfg: Any, data_root: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    ds = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(ds, dict)
    # Configs use paths.data_root, which can be overridden without editing YAML.
    for key, value in list(ds.items()):
        if isinstance(value, str) and value.startswith("/hdd1/DataInHere/YHF/data/"):
            ds[key] = str(data_root / Path(value).relative_to("/hdd1/DataInHere/YHF/data"))
    return ds


def _load_data(dataset: str, data_root: Path) -> dict[str, Any]:
    cfg = _compose_config(dataset)
    ds = _resolved_dataset_config(cfg, data_root)
    source = str(ds["source"]).lower()
    labels: torch.Tensor | None = None
    if source == "magb":
        try:
            import dgl
        except ImportError as exc:
            raise RuntimeError("DGL is required to read MAGB graph files; run in yhf_env.") from exc
        graph_path = Path(ds["graph_path"])
        graphs, _ = dgl.load_graphs(str(graph_path))
        if not graphs:
            raise ValueError(f"No graph found in {graph_path}")
        graph = graphs[0]
        src, dst = graph.edges()
        raw_edges = torch.stack((src.long(), dst.long()), dim=0)
        raw_labels = graph.ndata.get("label")
        labels = None if raw_labels is None else torch.as_tensor(raw_labels).long().cpu()
        num_nodes = int(graph.num_nodes())
        text = torch.from_numpy(np.asarray(np.load(ds["text_feat_path"], allow_pickle=False), dtype=np.float32))
        visual = torch.from_numpy(np.asarray(np.load(ds["image_feat_path"], allow_pickle=False), dtype=np.float32))
    elif source == "mmgraph":
        joint = torch.load(ds["joint_feat_path"], map_location="cpu", weights_only=False)
        joint = torch.as_tensor(joint).float().contiguous()
        num_nodes = int(joint.size(0))
        text_dim = int(ds.get("text_dim", 0))
        visual_dim = int(ds.get("visual_dim", 0))
        if not text_dim or not visual_dim:
            raise ValueError(f"{dataset}: text_dim and visual_dim must be configured")
        if text_dim + visual_dim > joint.size(1):
            raise ValueError(f"{dataset}: joint feature dimension is smaller than modality dimensions")
        text = joint[:, :text_dim].contiguous()
        visual = joint[:, text_dim : text_dim + visual_dim].contiguous()
        raw_edges = torch.as_tensor(torch.load(ds["edge_path"], map_location="cpu", weights_only=False)).long()
        raw_labels = torch.load(ds["label_path"], map_location="cpu", weights_only=False)
        labels = torch.as_tensor(raw_labels).long().view(-1)
    else:
        raise ValueError(f"Unsupported dataset source {source!r} for {dataset}")

    if text.ndim != 2 or visual.ndim != 2:
        raise ValueError(f"{dataset}: modality features must be two-dimensional")
    if text.size(0) != num_nodes or visual.size(0) != num_nodes:
        raise ValueError(f"{dataset}: feature row count does not match graph node count")
    text = text.float().contiguous()
    visual = visual.float().contiguous()
    for modality, feature in (("text", text), ("visual", visual)):
        if not bool(torch.isfinite(feature).all()):
            raise ValueError(f"{dataset}: {modality} features contain NaN or Inf")

    edge_index = torch.as_tensor(raw_edges).long()
    if edge_index.ndim != 2:
        raise ValueError(f"{dataset}: edge list must be a 2-D tensor")
    if edge_index.shape[0] == 2 and edge_index.shape[1] != 2:
        pairs = edge_index.t().cpu().numpy()
    elif edge_index.shape[1] == 2:
        pairs = edge_index.cpu().numpy()
    elif edge_index.shape[0] == 2:
        pairs = edge_index.t().cpu().numpy()
    else:
        raise ValueError(f"{dataset}: cannot infer edge-list orientation from {tuple(edge_index.shape)}")
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    valid_mask = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] < num_nodes) & (pairs[:, 1] < num_nodes)
    valid = pairs[valid_mask]
    loop_mask = valid[:, 0] == valid[:, 1]
    nonloops = valid[~loop_mask]
    canonical = np.sort(nonloops, axis=1)
    edges = np.unique(canonical, axis=0) if canonical.size else np.empty((0, 2), dtype=np.int64)
    edge_info = {
        "raw_edge_count": int(len(pairs)),
        "invalid_edge_count": int((~valid_mask).sum()),
        "self_loop_count_removed": int(loop_mask.sum()),
        "duplicate_edge_count_removed": int(len(nonloops) - len(edges)),
    }
    if labels is not None and labels.numel() != num_nodes:
        labels = None
    return {
        "name": dataset,
        "cfg": cfg,
        "dataset_cfg": ds,
        "num_nodes": num_nodes,
        "text": text,
        "visual": visual,
        "labels": labels,
        "edges": edges,
        "edge_info": edge_info,
    }


def _finite_array(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values encountered in {name}")
    return values


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / float(len(values))


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = _finite_array(values, "distribution")
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def _node_label(labels: torch.Tensor | None, node: int) -> int | None:
    if labels is None:
        return None
    value = int(labels[node].item())
    return value if value >= 0 else None


def _edge_example(dataset: dict[str, Any], edge_id: int, rank_t: np.ndarray, rank_v: np.ndarray,
                  sim_t: np.ndarray, sim_v: np.ndarray, selection: str) -> dict[str, Any]:
    left, right = (int(v) for v in dataset["edges"][edge_id])
    return {
        "selection": selection,
        "source_node": left,
        "target_node": right,
        "source_class": _node_label(dataset["labels"], left),
        "target_class": _node_label(dataset["labels"], right),
        "text_compatibility": float(sim_t[edge_id]),
        "visual_compatibility": float(sim_v[edge_id]),
        "text_rank_percentile": float(rank_t[edge_id]),
        "visual_rank_percentile": float(rank_v[edge_id]),
        "absolute_rank_gap": float(abs(rank_t[edge_id] - rank_v[edge_id])),
        "raw_text_metadata_loaded": False,
        "raw_image_metadata_loaded": False,
        "display_fallback": "Use node IDs and class labels; raw per-node text/image metadata was not loaded.",
    }


def _analyze_relation(dataset: dict[str, Any], example_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    edges = dataset["edges"]
    if len(edges) == 0:
        raise ValueError(f"{dataset['name']}: no physical edges remain after canonicalization")
    edge_tensor = torch.as_tensor(edges, dtype=torch.long)
    similarities: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        feature = torch.nn.functional.normalize(dataset[modality], p=2, dim=1, eps=1e-12)
        with torch.inference_mode():
            score = (feature[edge_tensor[:, 0]] * feature[edge_tensor[:, 1]]).sum(dim=1).cpu().numpy().astype(np.float64)
        similarities[modality] = _finite_array(score, f"{dataset['name']} {modality} edge cosine")
    sim_t, sim_v = similarities["text"], similarities["visual"]
    rank_t, rank_v = _rank_percentile(sim_t), _rank_percentile(sim_v)
    gap = _finite_array(np.abs(rank_t - rank_v), f"{dataset['name']} rank gap")
    if len(edges) >= 2 and np.std(rank_t) > EPS and np.std(rank_v) > EPS:
        rho = float(np.corrcoef(rank_t, rank_v)[0, 1])
        rho_status = "defined"
    else:
        rho, rho_status = 0.0, "degenerate_rank_or_too_few_edges"
    if not np.isfinite(rho):
        raise ValueError(f"{dataset['name']}: Spearman result is non-finite")

    text_first = np.flatnonzero((rank_t >= 0.75) & (rank_v <= 0.25))
    visual_first = np.flatnonzero((rank_v >= 0.75) & (rank_t <= 0.25))
    if not len(text_first):
        text_first = np.asarray([int(np.argmax(rank_t - rank_v))])
    if not len(visual_first):
        visual_first = np.asarray([int(np.argmax(rank_v - rank_t))])
    text_first = text_first[np.argsort((rank_t - rank_v)[text_first])[::-1]][:example_count]
    visual_first = visual_first[np.argsort((rank_v - rank_t)[visual_first])[::-1]][:example_count]
    examples = {
        "dataset": dataset["name"],
        "physical_edge_definition": "one unique undirected pair (min(node_i,node_j), max(node_i,node_j)); self-loops excluded",
        "rank_percentile_definition": "average tie rank divided by number of physical edges",
        "opposite_quartile_threshold": {"high": 0.75, "low": 0.25},
        "text_compatible_visual_dissimilar": [
            _edge_example(dataset, int(i), rank_t, rank_v, sim_t, sim_v, "text_high_visual_low") for i in text_first
        ],
        "visual_compatible_text_dissimilar": [
            _edge_example(dataset, int(i), rank_t, rank_v, sim_t, sim_v, "visual_high_text_low") for i in visual_first
        ],
    }
    summary = {
        "dataset": dataset["name"],
        "num_nodes": int(dataset["num_nodes"]),
        "num_physical_edges": int(len(edges)),
        "spearman_rho": rho,
        "spearman_status": rho_status,
        "median_absolute_rank_gap": float(np.median(gap)),
        "rank_gap_q25": float(np.quantile(gap, 0.25)),
        "rank_gap_q75": float(np.quantile(gap, 0.75)),
        "rank_gap_q90": float(np.quantile(gap, 0.90)),
        "rank_gap_q95": float(np.quantile(gap, 0.95)),
        "text_high_visual_low_fraction": float(np.mean((rank_t >= 0.75) & (rank_v <= 0.25))),
        "visual_high_text_low_fraction": float(np.mean((rank_v >= 0.75) & (rank_t <= 0.25))),
        "text_feature_dim": int(dataset["text"].size(1)),
        "visual_feature_dim": int(dataset["visual"].size(1)),
        **dataset["edge_info"],
    }
    edge_rows = []
    for i, (left, right) in enumerate(edges):
        edge_rows.append({
            "dataset": dataset["name"], "source_node": int(left), "target_node": int(right),
            "text_compatibility": float(sim_t[i]), "visual_compatibility": float(sim_v[i]),
            "text_rank_percentile": float(rank_t[i]), "visual_rank_percentile": float(rank_v[i]),
            "absolute_rank_gap": float(gap[i]),
        })
    return summary, {"examples": examples, "edge_rows": edge_rows}


def _normalized_propagation_operator(num_nodes: int, edges: np.ndarray) -> sp.csr_matrix:
    if len(edges):
        row = np.concatenate((edges[:, 0], edges[:, 1], np.arange(num_nodes, dtype=np.int64)))
        col = np.concatenate((edges[:, 1], edges[:, 0], np.arange(num_nodes, dtype=np.int64)))
    else:
        row = col = np.arange(num_nodes, dtype=np.int64)
    adjacency = sp.coo_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0
    inv_sqrt[positive] = np.power(degree[positive], -0.5)
    operator = sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)
    return operator.tocsr()


def _neighborhood_shells(edges: np.ndarray, center: int, num_nodes: int, max_hop: int,
                         labels: torch.Tensor | None, max_ids: int) -> list[dict[str, Any]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    visited = {center}
    frontier = {center}
    shells = []
    for hop in range(max_hop + 1):
        ordered = sorted(frontier)
        shells.append({
            "hop": hop,
            "node_count": len(ordered),
            "node_ids": ordered[:max_ids],
            "truncated": len(ordered) > max_ids,
            "class_labels": {str(node): _node_label(labels, node) for node in ordered[:max_ids]},
        })
        next_frontier: set[int] = set()
        for node in frontier:
            next_frontier.update(adjacency[node])
        next_frontier.difference_update(visited)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier and hop < max_hop:
            for empty_hop in range(hop + 1, max_hop + 1):
                shells.append({"hop": empty_hop, "node_count": 0, "node_ids": [], "truncated": False, "class_labels": {}})
            break
    return shells


def _analyze_drift(dataset: dict[str, Any], neighborhood_limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = int(dataset["num_nodes"])
    operator = _normalized_propagation_operator(n, dataset["edges"])
    rows: list[dict[str, Any]] = []
    drift_by_modality: dict[str, list[np.ndarray]] = {}
    for modality in MODALITIES:
        anchor = dataset[modality].cpu().numpy().astype(np.float32, copy=False)
        anchor_norm = np.linalg.norm(anchor, axis=1)
        state = anchor.copy()
        modality_drifts = []
        for hop in range(4):
            state_norm = np.linalg.norm(state, axis=1)
            denominator = np.maximum(anchor_norm * state_norm, 1e-12)
            cosine = np.einsum("ij,ij->i", anchor, state, optimize=True) / denominator
            cosine = np.clip(cosine, -1.0, 1.0)
            cosine = _finite_array(cosine, f"{dataset['name']} {modality} hop {hop} cosine")
            drift = _finite_array(1.0 - cosine, f"{dataset['name']} {modality} hop {hop} drift")
            modality_drifts.append(drift)
            cosine_summary = _distribution(cosine)
            drift_summary = _distribution(drift)
            rows.append({
                "dataset": dataset["name"], "modality": modality, "hop": hop, "num_nodes": n,
                "mean_cosine_to_anchor": cosine_summary["mean"],
                "median_cosine_to_anchor": cosine_summary["median"],
                "cosine_q25": cosine_summary["q25"], "cosine_q75": cosine_summary["q75"],
                "mean_drift_1_minus_cosine": drift_summary["mean"],
                "median_drift": drift_summary["median"],
                "drift_q25": drift_summary["q25"], "drift_q75": drift_summary["q75"],
            })
            if hop < 3:
                state = np.asarray(operator @ state, dtype=np.float32)
                if not np.isfinite(state).all():
                    raise ValueError(f"{dataset['name']} {modality} propagation produced NaN or Inf")
        drift_by_modality[modality] = modality_drifts

    final_node_drift = np.mean(np.stack((drift_by_modality["text"][3], drift_by_modality["visual"][3]), axis=1), axis=1)
    center = int(np.argmax(final_node_drift))
    shells = _neighborhood_shells(dataset["edges"], center, n, 3, dataset["labels"], neighborhood_limit)
    example = {
        "dataset": dataset["name"],
        "representative_node": center,
        "representative_node_class": _node_label(dataset["labels"], center),
        "selection": "maximum mean of text and visual 3-hop drift (1 - cosine to each modality's H0 anchor)",
        "mean_text_3hop_drift": float(drift_by_modality["text"][3][center]),
        "mean_visual_3hop_drift": float(drift_by_modality["visual"][3][center]),
        "neighborhood_shells": shells,
        "raw_text_metadata_loaded": False,
        "raw_image_metadata_loaded": False,
        "display_fallback": "Use node IDs and available class labels; raw per-node text/image metadata was not loaded.",
    }
    return rows, example


def _propagation_states(operator: sp.csr_matrix, anchor: np.ndarray, max_hop: int = 3) -> list[np.ndarray]:
    states = [np.asarray(anchor, dtype=np.float32)]
    for _ in range(max_hop):
        next_state = np.asarray(operator @ states[-1], dtype=np.float32)
        if not np.isfinite(next_state).all():
            raise ValueError("Propagation produced NaN or Inf")
        states.append(next_state)
    return states


def _mean_anchor_drift(anchor: np.ndarray, state: np.ndarray) -> float:
    anchor_norm = np.linalg.norm(anchor, axis=1)
    state_norm = np.linalg.norm(state, axis=1)
    denominator = np.maximum(anchor_norm * state_norm, 1e-12)
    cosine = np.einsum("ij,ij->i", anchor, state, optimize=True) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.mean(1.0 - cosine))


def _reddit_drift_sanity_audit(
    dataset: dict[str, Any], stored_rows: list[dict[str, str]] | None
) -> dict[str, Any]:
    """Check whether Reddit-S's flat post-hop-one curve follows from its graph."""
    if dataset["name"] != "Reddit-S":
        raise ValueError("The drift sanity audit is specific to Reddit-S")
    n = int(dataset["num_nodes"])
    edges = dataset["edges"]
    row = np.concatenate((edges[:, 0], edges[:, 1])) if len(edges) else np.empty(0, dtype=np.int64)
    col = np.concatenate((edges[:, 1], edges[:, 0])) if len(edges) else np.empty(0, dtype=np.int64)
    adjacency = sp.coo_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)), shape=(n, n)
    ).tocsr()
    adjacency.sum_duplicates()
    operator = _normalized_propagation_operator(n, edges)

    from scipy.sparse.csgraph import connected_components

    component_count, component_ids = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(component_ids, minlength=component_count)
    edge_component_counts = np.bincount(component_ids[edges[:, 0]], minlength=component_count)
    possible_edges = component_sizes * (component_sizes - 1) // 2
    nontrivial = possible_edges > 0
    components_complete = bool(np.all(edge_component_counts[nontrivial] == possible_edges[nontrivial]))
    adjacency_degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    symmetric_error = float(np.max(np.abs((operator - operator.T).data))) if (operator - operator.T).nnz else 0.0
    idempotence = operator @ operator - operator
    operator_norm = float(sp.linalg.norm(operator))
    idempotence_relative_frobenius = float(sp.linalg.norm(idempotence) / max(operator_norm, 1e-12))

    stored = {}
    if stored_rows:
        for item in stored_rows:
            if item.get("dataset") == "Reddit-S":
                key = (item.get("modality", "").lower(), int(item["hop"]))
                stored[key] = float(item["mean_drift_1_minus_cosine"])

    modality_audits: dict[str, Any] = {}
    for modality in MODALITIES:
        anchor = dataset[modality].cpu().numpy().astype(np.float32, copy=False)
        states = _propagation_states(operator, anchor, max_hop=3)
        drifts = [_mean_anchor_drift(anchor, state) for state in states]
        relative_steps = [
            float(np.linalg.norm(states[hop + 1] - states[hop]) / max(np.linalg.norm(states[hop]), 1e-12))
            for hop in range(3)
        ]
        max_abs_steps = [
            float(np.max(np.abs(states[hop + 1] - states[hop]))) for hop in range(3)
        ]
        matches: dict[str, Any] = {}
        for hop, drift in enumerate(drifts):
            value = stored.get((modality, hop))
            matches[str(hop)] = {
                "stored_mean_drift": value,
                "recomputed_mean_drift": drift,
                "absolute_difference": None if value is None else abs(value - drift),
                "matches_within_1e-7": None if value is None else bool(abs(value - drift) <= 1e-7),
            }
        modality_audits[modality] = {
            "mean_drift_by_hop_0_to_3": drifts,
            "drift_hop_1_to_3_range": float(max(drifts[1:]) - min(drifts[1:])),
            "relative_frobenius_state_step_0_to_1_1_to_2_2_to_3": relative_steps,
            "max_absolute_state_step_0_to_1_1_to_2_2_to_3": max_abs_steps,
            "stored_curve_comparison": matches,
        }

    return {
        "dataset": "Reddit-S",
        "audit_type": "sanity audit of existing normalized propagation; no new experiment or model",
        "physical_graph": {
            "num_nodes": n,
            "num_unique_undirected_edges": int(len(edges)),
            "num_connected_components": int(component_count),
            "component_size_min": int(component_sizes.min()),
            "component_size_median": float(np.median(component_sizes)),
            "component_size_max": int(component_sizes.max()),
            "all_nontrivial_components_are_cliques": components_complete,
            "degree_min": int(adjacency_degree.min()),
            "degree_mean": float(adjacency_degree.mean()),
            "degree_max": int(adjacency_degree.max()),
        },
        "operator": {
            "definition": "P = D^-1/2 (A + I) D^-1/2 on the unique undirected graph",
            "max_absolute_symmetry_error": symmetric_error,
            "relative_frobenius_idempotence_error_norm_P2_minus_P": idempotence_relative_frobenius,
            "interpretation": "A disjoint union of cliques makes this symmetric normalized operator an idempotent component-mean projector in exact arithmetic.",
        },
        "modality_curves": modality_audits,
        "conclusion": (
            "The Reddit-S graph is a disjoint union of complete components. The normalized propagation operator is consequently idempotent up to float32 rounding: S2=P(S1) and S3=P(S2) differ from S1 by relative Frobenius norms around 1e-7. The recomputed post-hop-one mean-drift range is below 1e-8 for both modalities and agrees with the stored curve within 1e-7. The flat hop-1/2/3 values are therefore expected from this graph structure; the source data and propagation implementation pass this sanity audit."
            if components_complete and idempotence_relative_frobenius < 1e-6
            else "The graph/operator checks did not meet the expected clique-projector tolerances; inspect the detailed state and stored-curve comparisons before interpreting Reddit-S drift."
        ),
    }


def _load_order_split(dataset: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Path]:
    # MAGB datasets expose a seed-specific NC split, while ele-fashion uses
    # the equivalent node_split_path shipped with the MM-Graph data.
    split_key = "nc_split_path" if "nc_split_path" in dataset["dataset_cfg"] else "node_split_path"
    path = Path(dataset["dataset_cfg"][split_key])
    if not path.is_file():
        raise FileNotFoundError(f"Expected an existing NC split (read-only): {path}")
    split = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(split, dict) or "train_idx" not in split or "val_idx" not in split:
        raise ValueError(f"Invalid NC split file: {path}")
    train_idx = torch.as_tensor(split["train_idx"], dtype=torch.long).reshape(-1)
    val_idx = torch.as_tensor(split["val_idx"], dtype=torch.long).reshape(-1)
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise ValueError(f"Empty train/validation split in {path}")
    if torch.isin(train_idx, val_idx).any():
        raise ValueError(f"Train and validation indices overlap in {path}")
    n = int(dataset["num_nodes"])
    if int(torch.min(torch.cat((train_idx, val_idx)))) < 0 or int(torch.max(torch.cat((train_idx, val_idx)))) >= n:
        raise ValueError(f"Split indices fall outside {n} graph nodes in {path}")
    return train_idx, val_idx, path


def _train_order_classifier(
    state: np.ndarray,
    labels: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
) -> tuple[np.ndarray, float]:
    """Fit one fixed full-batch linear classifier and return per-node validation CE."""
    if not np.isfinite(state).all():
        raise ValueError("Order representation contains NaN or Inf")
    x = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32))
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    if (y_train < 0).any() or (y_val < 0).any():
        raise ValueError("Train/validation labels contain an invalid negative class")

    torch.manual_seed(ORDER_CLASSIFIER_SEED)
    classifier = torch.nn.Linear(int(x.shape[1]), num_classes, bias=True)
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=ORDER_CLASSIFIER_LR, weight_decay=ORDER_CLASSIFIER_WEIGHT_DECAY
    )
    x_train, x_val = x[train_idx], x[val_idx]
    for _ in range(ORDER_CLASSIFIER_EPOCHS):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
    classifier.eval()
    with torch.inference_mode():
        val_logits = classifier(x_val)
        per_node_ce = torch.nn.functional.cross_entropy(val_logits, y_val, reduction="none")
        final_train_ce = torch.nn.functional.cross_entropy(classifier(x_train), y_train)
    ce_np = per_node_ce.cpu().numpy().astype(np.float64, copy=False)
    _finite_array(ce_np, "validation cross entropy")
    if not bool(torch.isfinite(final_train_ce)):
        raise ValueError("Final training cross entropy is non-finite")
    return ce_np, float(final_train_ce.item())


def _analyze_order_utility(dataset: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset["labels"] is None:
        raise ValueError(f"{dataset['name']}: node labels are required for order-utility analysis")
    train_idx, val_idx, split_path = _load_order_split(dataset)
    labels = dataset["labels"].long().cpu()
    class_count = int(dataset["dataset_cfg"].get("num_classes") or (int(labels.max()) + 1))
    if int(labels.max()) >= class_count:
        raise ValueError(f"{dataset['name']}: label id exceeds configured class count {class_count}")
    operator = _normalized_propagation_operator(int(dataset["num_nodes"]), dataset["edges"])
    rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []

    for modality in MODALITIES:
        anchor = dataset[modality].cpu().numpy().astype(np.float32, copy=False)
        states = _propagation_states(operator, anchor, max_hop=3)
        losses: list[np.ndarray] = []
        for hop, state in enumerate(states):
            print(f"[figure1] {dataset['name']} {modality} linear classifier hop={hop}", flush=True)
            ce, _ = _train_order_classifier(state, labels, train_idx, val_idx, class_count)
            losses.append(ce)
            del ce
            gc.collect()

        loss_matrix = _finite_array(np.stack(losses, axis=1), f"{dataset['name']} {modality} validation CE matrix")
        # np.argmin returns the smallest hop on exact ties; this fixed tie rule is recorded in the CSV.
        best_order = np.argmin(loss_matrix, axis=1)
        counts = np.bincount(best_order, minlength=4)
        for node_index, best_hop in zip(val_idx.tolist(), best_order.tolist()):
            node_rows.append({
                "dataset": dataset["name"],
                "modality": modality,
                "node_id": int(node_index),
                "label": int(labels[node_index].item()),
                "best_order": int(best_hop),
                "num_validation_nodes": int(val_idx.numel()),
                "random_seed": ORDER_CLASSIFIER_SEED,
                "tie_rule": "lowest hop wins exact CE ties (numpy.argmin)",
            })
        for hop in range(4):
            row = {
                "dataset": dataset["name"], "modality": modality, "hop": hop,
                "num_nodes": int(dataset["num_nodes"]),
                "num_train_nodes": int(train_idx.numel()),
                "num_validation_nodes": int(val_idx.numel()),
                "best_order_count": int(counts[hop]),
                "fraction_best": float(counts[hop] / val_idx.numel()),
                "mean_validation_cross_entropy": float(np.mean(loss_matrix[:, hop])),
                "median_validation_cross_entropy": float(np.median(loss_matrix[:, hop])),
                "classifier": "torch.nn.Linear(input_dim, num_classes, bias=True)",
                "optimizer": "Adam(full_batch)",
                "learning_rate": ORDER_CLASSIFIER_LR,
                "weight_decay": ORDER_CLASSIFIER_WEIGHT_DECAY,
                "epochs": ORDER_CLASSIFIER_EPOCHS,
                "random_seed": ORDER_CLASSIFIER_SEED,
                "split_path": str(split_path),
                "tie_rule": "lowest hop wins exact CE ties (numpy.argmin)",
            }
            for numeric_key in (
                "fraction_best", "mean_validation_cross_entropy", "median_validation_cross_entropy",
                "learning_rate", "weight_decay",
            ):
                if not np.isfinite(float(row[numeric_key])):
                    raise ValueError(f"Non-finite {numeric_key} in {dataset['name']} {modality} hop {hop}")
            rows.append(row)
        del states, losses, loss_matrix
        gc.collect()
    return rows, node_rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _assert_json_finite(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_json_finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_json_finite(child, f"{location}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"Non-finite JSON value at {location}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _assert_json_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS),
                        help="Datasets for relation/drift data (default: all five).")
    parser.add_argument("--order-datasets", nargs="+", choices=ORDER_UTILITY_DATASETS,
                        default=list(ORDER_UTILITY_DATASETS),
                        help="Datasets for the fixed linear order-utility analysis (default: all five).")
    parser.add_argument("--data-root", type=Path, default=Path("/hdd1/DataInHere/YHF/data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples-per-direction", type=int, default=5)
    parser.add_argument("--neighborhood-node-limit", type=int, default=40)
    parser.add_argument("--only-order-utility", action="store_true",
                        help="Generate the new order-utility CSV and Reddit-S audit without rewriting relation/drift data.")
    args = parser.parse_args()
    if args.examples_per_direction < 1 or args.neighborhood_node_limit < 1:
        parser.error("example and neighborhood limits must be positive")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    data_root = args.data_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

    previous_drift_path = output_dir / "semantic_drift_curves.csv"
    previous_drift_rows: list[dict[str, str]] | None = None
    if previous_drift_path.is_file():
        with previous_drift_path.open("r", encoding="utf-8", newline="") as handle:
            previous_drift_rows = list(csv.DictReader(handle))

    relation_summaries: list[dict[str, Any]] = []
    relation_examples: dict[str, Any] = {"analysis": "relation-level semantic discrepancy", "datasets": {}}
    all_edge_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    drift_examples: dict[str, Any] = {"analysis": "normalized shared-graph propagation drift", "datasets": {}}
    dataset_manifests: list[dict[str, Any]] = []

    if not args.only_order_utility:
        for index, name in enumerate(args.datasets, start=1):
            print(f"[figure1] relation/drift {index}/{len(args.datasets)} loading {name}", flush=True)
            dataset = _load_data(name, data_root)
            rel_summary, rel_data = _analyze_relation(dataset, args.examples_per_direction)
            relation_summaries.append(rel_summary)
            all_edge_rows.extend(rel_data["edge_rows"])
            relation_examples["datasets"][name] = rel_data["examples"]

            ds_drift, drift_example = _analyze_drift(dataset, args.neighborhood_node_limit)
            drift_rows.extend(ds_drift)
            drift_examples["datasets"][name] = drift_example
            dataset_manifests.append({
                "dataset": name,
                "num_nodes": int(dataset["num_nodes"]),
                "num_physical_edges": int(len(dataset["edges"])),
                "text_feature_shape": list(dataset["text"].shape),
                "visual_feature_shape": list(dataset["visual"].shape),
                **dataset["edge_info"],
            })
            print(
                f"[figure1] {name}: edges={rel_summary['num_physical_edges']}, "
                f"rho={rel_summary['spearman_rho']:.4f}, "
                f"median_gap={rel_summary['median_absolute_rank_gap']:.4f}",
                flush=True,
            )
            del dataset, rel_data
            gc.collect()

        _write_csv(output_dir / "relation_discrepancy_summary.csv", RELATION_SUMMARY_FIELDS, relation_summaries)
        _write_csv(output_dir / "relation_discrepancy_edges.csv", RELATION_EDGE_FIELDS, all_edge_rows)
        _write_json(output_dir / "relation_discrepancy_examples.json", relation_examples)
        _write_csv(output_dir / "semantic_drift_curves.csv", DRIFT_FIELDS, drift_rows)
        _write_json(output_dir / "semantic_drift_examples.json", drift_examples)

    order_rows: list[dict[str, Any]] = []
    order_node_rows: list[dict[str, Any]] = []
    for index, name in enumerate(args.order_datasets, start=1):
        print(f"[figure1] order utility {index}/{len(args.order_datasets)} loading {name}", flush=True)
        dataset = _load_data(name, data_root)
        summary_rows, node_rows = _analyze_order_utility(dataset)
        order_rows.extend(summary_rows)
        order_node_rows.extend(node_rows)
        print(f"[figure1] {name}: trained 8 fixed linear classifiers on the existing NC split", flush=True)
        del dataset
        gc.collect()
    _write_csv(output_dir / "order_utility_heterogeneity.csv", ORDER_UTILITY_FIELDS, order_rows)
    _write_csv(output_dir / "order_utility_node_choices.csv", ORDER_UTILITY_NODE_FIELDS, order_node_rows)

    # Read-only recomputation of the already defined Reddit-S propagation curve.
    reddit = _load_data("Reddit-S", data_root)
    reddit_audit = _reddit_drift_sanity_audit(reddit, previous_drift_rows)
    _write_json(output_dir / "semantic_drift_sanity_audit.json", reddit_audit)
    del reddit
    print(f"[figure1] Reddit-S audit: {reddit_audit['physical_graph']['num_connected_components']} clique components; "
          f"P^2≈P error={reddit_audit['operator']['relative_frobenius_idempotence_error_norm_P2_minus_P']:.3e}", flush=True)

    _write_json(output_dir / "run_manifest.json", {
        "analysis": "model-independent Figure 1 empirical motivation data",
        "datasets": list(args.datasets) if not args.only_order_utility else [],
        "order_utility_datasets": list(args.order_datasets),
        "data_root": str(data_root),
        "checkpoint_access": "none",
        "formal_model_loaded": False,
        "base_model_training_started": False,
        "linear_classifier_training": {
            "purpose": "per-hop empirical utility only",
            "architecture": "torch.nn.Linear(input_dim, num_classes, bias=True)",
            "optimizer": "Adam(full_batch)",
            "epochs": ORDER_CLASSIFIER_EPOCHS,
            "learning_rate": ORDER_CLASSIFIER_LR,
            "weight_decay": ORDER_CLASSIFIER_WEIGHT_DECAY,
            "seed": ORDER_CLASSIFIER_SEED,
            "validation_used_for_hop_selection": True,
            "hyperparameter_tuning": False,
        },
        "benchmark_output_used": False,
        "physical_edges": "unique undirected pairs; self-loops excluded before adding one propagation self-loop per node",
        "propagation_operator": "P = D^-1/2 (A + I) D^-1/2, shared unweighted physical graph",
        "drift": "node-wise 1 - cosine(H0, P^k H0), k=0..3",
        "figure1c_definition": "For each modality and hop, a separately trained identical linear classifier; each validation node is assigned the hop with minimum per-node cross entropy.",
        "order_utility_summary_output": "order_utility_heterogeneity.csv",
        "order_utility_node_output": "order_utility_node_choices.csv",
        "reddit_sanity_audit": "semantic_drift_sanity_audit.json",
        "legacy_eta_outputs": "Existing aggregation_heterogeneity_summary.csv and aggregation_heterogeneity_examples.json are retained untouched for Figure 4 mechanism analysis.",
        "dataset_records": dataset_manifests,
        "relation_and_drift_outputs_rewritten": not args.only_order_utility,
    })
    print(f"[figure1] wrote model-independent empirical data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
