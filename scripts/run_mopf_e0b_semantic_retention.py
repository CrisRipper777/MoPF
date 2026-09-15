"""E0-B: semantic retention under ordinary physical-graph propagation.

This is a method-independent, analysis-only study.  It reuses the E0-A NC
loader path and the validated U2-A linear CKA/sampling helpers, but it never
constructs a model or reads a checkpoint.  For each raw modality feature
matrix X, it evaluates Q_0=X and Q_k=P Q_{k-1} for k=1,...,6, where
P=D_tilde^{-1/2}(A+I)D_tilde^{-1/2} and every physical edge has weight one.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mopf_e0a_edge_semantic_discrepancy import (  # noqa: E402
    _compose_cfg,
    _feature_sources,
)
from src.analysis.u2a import (  # noqa: E402
    DEFAULT_NODE_SAMPLE_SEED,
    MAX_CKA_NODES,
    _linear_cka,
    deterministic_node_subset,
)
from src.data import load_mag_data  # noqa: E402
from src.data.graph_utils import canonicalize_edges  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
FORMAL_K = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
}
K_ANALYSIS = 6
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "e0_empirical_motivation" / "semantic_retention"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_EDGE_CHUNK_SIZE = 131_072
EPS = 1e-12


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _safe_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    hops = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(hops, np.asarray(values, dtype=np.float64), 1)[0])


def _retention_ratio(values: list[float]) -> list[float]:
    denominator = float(values[0]) if values else 0.0
    if abs(denominator) <= EPS:
        return [0.0 for _ in values]
    return [float(value / denominator) for value in values]


def _decrease_steps(values: list[float]) -> int:
    if len(values) < 2:
        return 0
    return int(np.count_nonzero(np.diff(np.asarray(values, dtype=np.float64)) < 0.0))


def _cosine_stats(values: np.ndarray, connected_mask: np.ndarray, special_mask: np.ndarray) -> dict[str, Any]:
    connected_values = values[connected_mask]
    if connected_values.size == 0:
        raise ValueError("Cannot summarize cosine values with no connected nodes")
    return {
        "mean": float(np.mean(connected_values)),
        "std": float(np.std(connected_values, ddof=0)),
        "median": float(np.median(connected_values)),
        "q25": float(np.quantile(connected_values, 0.25)),
        "q75": float(np.quantile(connected_values, 0.75)),
        "q10": float(np.quantile(connected_values, 0.10)),
        "q90": float(np.quantile(connected_values, 0.90)),
        "all_node_mean": float(np.mean(values)),
        "special_connected_nodes": int(np.count_nonzero(special_mask & connected_mask)),
        "special_all_nodes": int(np.count_nonzero(special_mask)),
        "finite": bool(np.isfinite(values).all()),
    }


def _nodewise_cosine(
    state: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite node-wise cosine values and explicit zero-norm handling."""
    state_double = state.to(dtype=torch.float64)
    reference_double = reference.to(dtype=torch.float64)
    state_norm = torch.linalg.vector_norm(state_double, dim=1)
    reference_norm = torch.linalg.vector_norm(reference_double, dim=1)
    special = (state_norm <= 0.0) | (reference_norm <= 0.0)
    valid = ~special
    values = torch.zeros(state.size(0), dtype=torch.float64, device=state.device)
    if bool(valid.any()):
        dot = (state_double[valid] * reference_double[valid]).sum(dim=1)
        values[valid] = dot / (state_norm[valid] * reference_norm[valid])
    values = torch.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    return (
        values.detach().cpu().numpy().astype(np.float32, copy=False),
        special.detach().cpu().numpy().astype(bool, copy=False),
    )


def _build_fixed_operator(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build P=D_tilde^-1/2(A+I)D_tilde^-1/2 from the loader edge index."""
    edge_index = edge_index.detach().cpu().to(dtype=torch.long).contiguous()
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"Expected edge_index [2,E], got {tuple(edge_index.shape)}")
    self_loop_mask = edge_index[0] == edge_index[1]
    if bool(self_loop_mask.any()):
        raise ValueError("The loader edge_index unexpectedly contains self-loops before analysis")

    keys = edge_index[0].numpy().astype(np.int64) * np.int64(num_nodes) + edge_index[1].numpy().astype(np.int64)
    if np.unique(keys).size != keys.size:
        raise ValueError("The loader edge_index contains duplicate oriented physical edges")
    physical_edges = canonicalize_edges(edge_index)
    if physical_edges.size(0) * 2 != edge_index.size(1):
        raise ValueError(
            "Expected the configured undirected loader graph to contain exactly two "
            "oriented entries per physical edge"
        )

    index = edge_index.to(device=device)
    loops = torch.arange(num_nodes, dtype=torch.long, device=device)
    loop_index = torch.stack([loops, loops], dim=0)
    operator_index = torch.cat([index, loop_index], dim=1)
    edge_weight = torch.ones(operator_index.size(1), dtype=torch.float32, device=device)
    degree_tilde = torch.bincount(operator_index[0], weights=edge_weight, minlength=num_nodes)
    inverse_sqrt_degree = degree_tilde.clamp_min(EPS).rsqrt()
    normalized_weight = edge_weight * inverse_sqrt_degree[operator_index[0]] * inverse_sqrt_degree[operator_index[1]]
    operator = torch.sparse_coo_tensor(
        operator_index,
        normalized_weight,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
        device=device,
    ).coalesce()
    coalesced_index = operator.indices()
    if coalesced_index.size(1) != num_nodes + edge_index.size(1):
        raise ValueError("A+I unexpectedly coalesced entries; each node must have one added self-loop")
    metadata = {
        "operator": "P = D_tilde^(-1/2) (A + I) D_tilde^(-1/2)",
        "edge_weight": 1.0,
        "input_edge_index_source": "src.data.load_mag_data task=nc",
        "loader_edge_count_without_analysis_self_loops": int(edge_index.size(1)),
        "num_physical_edges": int(physical_edges.size(0)),
        "canonical_physical_edges_for_audit": int(physical_edges.size(0)),
        "reverse_edges_added_by_analysis": False,
        "added_self_loops": int(num_nodes),
        "operator_nnz_after_coalesce": int(operator.indices().size(1)),
        "each_node_exactly_one_self_loop": bool(
            int(operator.indices().size(1)) == int(edge_index.size(1) + num_nodes)
        ),
        "normalization": "symmetric degree normalization with degree(A+I)",
        "degree_tilde_min": float(degree_tilde.min().item()),
        "degree_tilde_max": float(degree_tilde.max().item()),
    }
    return operator, metadata


def _propagate_states(
    feature: torch.Tensor,
    operator: torch.Tensor,
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    reference = feature.to(device=device, dtype=torch.float32).contiguous()
    states = [reference]
    with torch.inference_mode():
        for _ in range(K_ANALYSIS):
            states.append(torch.sparse.mm(operator, states[-1]))
    return states


def _analyze_modality(
    feature: torch.Tensor,
    modality: str,
    connected_mask_cpu: np.ndarray,
    cka_node_ids: torch.Tensor,
    operator: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    if feature.ndim != 2:
        raise ValueError(f"{modality}: expected a 2-D feature matrix, got {tuple(feature.shape)}")
    feature_finite = bool(torch.isfinite(feature).all().item())
    if not feature_finite:
        raise ValueError(f"{modality}: raw frozen feature contains non-finite values")
    raw_zero_mask = torch.linalg.vector_norm(feature.to(dtype=torch.float64), dim=1) <= 0.0
    raw_zero_count = int(raw_zero_mask.sum().item())
    raw_zero_connected_count = int((raw_zero_mask & torch.from_numpy(connected_mask_cpu)).sum().item())

    states = _propagate_states(feature, operator, device=device)
    reference = states[0]
    sampled_ids = cka_node_ids.to(device=device)
    cka_values: list[float] = []
    cosine_values: list[float] = []
    cosine_stats: list[dict[str, Any]] = []
    cosine_by_hop: list[np.ndarray] = []
    special_counts: list[dict[str, int]] = []
    finite_by_hop: list[bool] = []
    with torch.inference_mode():
        for state in states:
            finite = bool(torch.isfinite(state).all().item())
            finite_by_hop.append(finite)
            if not finite:
                raise ValueError(f"{modality}: propagated state is non-finite")
            cka = float(_linear_cka(state.index_select(0, sampled_ids), reference.index_select(0, sampled_ids)))
            cka = float(np.clip(cka, 0.0, 1.0))
            cosine, special = _nodewise_cosine(state, reference)
            if not np.isfinite(cosine).all():
                raise ValueError(f"{modality}: node-wise cosine contains non-finite values")
            cka_values.append(cka)
            cosine_by_hop.append(cosine)
            cosine_values.append(float(np.mean(cosine[connected_mask_cpu])))
            cosine_stats.append(_cosine_stats(cosine, connected_mask_cpu, special))
            special_counts.append({
                "connected_nodes": int(np.count_nonzero(special & connected_mask_cpu)),
                "all_nodes": int(np.count_nonzero(special)),
            })

    cka_retention = _retention_ratio(cka_values)
    cosine_retention = _retention_ratio(cosine_values)
    return {
        "modality": modality,
        "feature_finite": feature_finite,
        "feature_shape": [int(v) for v in feature.shape],
        "feature_dtype": str(feature.dtype),
        "raw_zero_norm_nodes": raw_zero_count,
        "raw_zero_norm_connected_nodes": raw_zero_connected_count,
        "states_finite_by_hop": finite_by_hop,
        "cka": cka_values,
        "cka_retention_ratio_to_hop0": cka_retention,
        "cka_slope_0_6": _safe_slope(cka_values),
        "cka_monotonic_decrease_steps": _decrease_steps(cka_values),
        "cosine_mean": cosine_values,
        "cosine_retention_ratio_to_hop0": cosine_retention,
        "cosine_slope_0_6": _safe_slope(cosine_values),
        "cosine_monotonic_decrease_steps": _decrease_steps(cosine_values),
        "cosine_stats_by_hop": cosine_stats,
        "cosine_special_counts_by_hop": special_counts,
        "cosine_by_hop": cosine_by_hop,
    }


def _summary_fields() -> tuple[str, ...]:
    fields = [
        "dataset", "modality", "formal_K", "num_nodes", "connected_nodes", "isolated_nodes", "connected_fraction",
    ]
    fields += [f"cka_k{hop}" for hop in range(K_ANALYSIS + 1)]
    fields += ["cka_formal_K", "cka_drop_formal", "cka_k6", "cka_drop_k6", "cka_slope_0_6"]
    fields += [f"cka_retention_k{hop}" for hop in range(K_ANALYSIS + 1)]
    fields += ["monotonic_decrease_steps_cka"]
    fields += [f"cosine_mean_k{hop}" for hop in range(K_ANALYSIS + 1)]
    fields += ["cosine_formal_K", "cosine_drop_formal", "cosine_k6", "cosine_drop_k6", "cosine_slope_0_6"]
    fields += [f"cosine_retention_k{hop}" for hop in range(K_ANALYSIS + 1)]
    for statistic in ("std", "median", "q25", "q75", "q10", "q90"):
        fields += [f"cosine_{statistic}_k{hop}" for hop in range(K_ANALYSIS + 1)]
    fields += [f"cosine_all_node_mean_k{hop}" for hop in range(K_ANALYSIS + 1)]
    fields += [f"cosine_special_connected_nodes_k{hop}" for hop in range(K_ANALYSIS + 1)]
    fields += [f"cosine_special_all_nodes_k{hop}" for hop in range(K_ANALYSIS + 1)]
    fields += ["monotonic_decrease_steps_cosine"]
    return tuple(fields)


def _make_summary_row(
    dataset: str,
    modality_result: dict[str, Any],
    *,
    num_nodes: int,
    connected_nodes: int,
    isolated_nodes: int,
) -> dict[str, Any]:
    formal_k = FORMAL_K[dataset]
    cka = modality_result["cka"]
    cka_retention = modality_result["cka_retention_ratio_to_hop0"]
    cosine = modality_result["cosine_mean"]
    cosine_retention = modality_result["cosine_retention_ratio_to_hop0"]
    row: dict[str, Any] = {
        "dataset": dataset,
        "modality": modality_result["modality"],
        "formal_K": formal_k,
        "num_nodes": num_nodes,
        "connected_nodes": connected_nodes,
        "isolated_nodes": isolated_nodes,
        "connected_fraction": float(connected_nodes / max(num_nodes, 1)),
    }
    row.update({f"cka_k{hop}": cka[hop] for hop in range(K_ANALYSIS + 1)})
    row.update({
        "cka_formal_K": cka[formal_k],
        "cka_drop_formal": float(cka[0] - cka[formal_k]),
        "cka_k6": cka[K_ANALYSIS],
        "cka_drop_k6": float(cka[0] - cka[K_ANALYSIS]),
        "cka_slope_0_6": modality_result["cka_slope_0_6"],
    })
    row.update({f"cka_retention_k{hop}": cka_retention[hop] for hop in range(K_ANALYSIS + 1)})
    row["monotonic_decrease_steps_cka"] = modality_result["cka_monotonic_decrease_steps"]
    row.update({f"cosine_mean_k{hop}": cosine[hop] for hop in range(K_ANALYSIS + 1)})
    row.update({
        "cosine_formal_K": cosine[formal_k],
        "cosine_drop_formal": float(cosine[0] - cosine[formal_k]),
        "cosine_k6": cosine[K_ANALYSIS],
        "cosine_drop_k6": float(cosine[0] - cosine[K_ANALYSIS]),
        "cosine_slope_0_6": modality_result["cosine_slope_0_6"],
    })
    row.update({f"cosine_retention_k{hop}": cosine_retention[hop] for hop in range(K_ANALYSIS + 1)})
    for statistic in ("std", "median", "q25", "q75", "q10", "q90"):
        row.update({
            f"cosine_{statistic}_k{hop}": modality_result["cosine_stats_by_hop"][hop][statistic]
            for hop in range(K_ANALYSIS + 1)
        })
    row.update({
        f"cosine_all_node_mean_k{hop}": modality_result["cosine_stats_by_hop"][hop]["all_node_mean"]
        for hop in range(K_ANALYSIS + 1)
    })
    row.update({
        f"cosine_special_connected_nodes_k{hop}": modality_result["cosine_stats_by_hop"][hop]["special_connected_nodes"]
        for hop in range(K_ANALYSIS + 1)
    })
    row.update({
        f"cosine_special_all_nodes_k{hop}": modality_result["cosine_stats_by_hop"][hop]["special_all_nodes"]
        for hop in range(K_ANALYSIS + 1)
    })
    row["monotonic_decrease_steps_cosine"] = modality_result["cosine_monotonic_decrease_steps"]
    return row


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = _summary_fields()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _plot_trajectories(
    results: list[dict[str, Any]],
    *,
    output_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"text": "#1f77b4", "visual": "#2ca02c"}
    labels = {"text": "Text", "visual": "Visual"}
    hops = np.arange(K_ANALYSIS + 1)

    def make_figure(metric: str, filename: str, ylabel: str, with_band: bool) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), squeeze=False, constrained_layout=True)
        axes_flat = list(axes.flat)
        for axis, result in zip(axes_flat, results):
            dataset = result["dataset"]
            formal_k = FORMAL_K[dataset]
            for modality in ("text", "visual"):
                detail = result["modalities"][modality]
                if metric == "cka":
                    values = np.asarray(detail["cka"], dtype=np.float64)
                else:
                    values = np.asarray(detail["cosine_mean"], dtype=np.float64)
                axis.plot(hops, values, marker="o", linewidth=2.0, markersize=4.5,
                          color=colors[modality], label=labels[modality])
                if with_band:
                    stats = detail["cosine_stats_by_hop"]
                    q25 = np.asarray([item["q25"] for item in stats], dtype=np.float64)
                    q75 = np.asarray([item["q75"] for item in stats], dtype=np.float64)
                    axis.fill_between(hops, q25, q75, color=colors[modality], alpha=0.12, linewidth=0)
            axis.axvline(formal_k, color="#777777", linestyle="--", linewidth=1.2,
                         label=f"formal K={formal_k}")
            axis.set_xticks(hops)
            axis.set_xlabel("Propagation hop k")
            axis.set_ylabel(ylabel)
            axis.set_title(dataset)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="best")
        for axis in axes_flat[len(results):]:
            axis.axis("off")
        fig.suptitle(f"E0-B Semantic retention — {ylabel}", fontsize=14)
        fig.savefig(output_root / f"{filename}.png", dpi=220)
        fig.savefig(output_root / f"{filename}.pdf")
        plt.close(fig)

    make_figure("cka", "semantic_retention_cka", "Linear CKA to raw modality feature", False)
    make_figure("cosine", "semantic_retention_node_cosine", "Mean node-wise cosine to raw modality feature", True)


def _analyze_dataset(
    dataset: str,
    *,
    seed: int,
    device: torch.device,
    output_root: Path,
    sample_seed: int,
    max_cka_nodes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = _compose_cfg(dataset, seed)
    data = load_mag_data(cfg, "nc", seed)
    if data.x_t is None or data.x_i is None:
        raise ValueError(f"{dataset}: NC loader did not expose both raw modalities")
    if data.x_t.size(0) != data.num_nodes or data.x_i.size(0) != data.num_nodes:
        raise ValueError(f"{dataset}: feature/node count mismatch")

    feature_sources = _feature_sources(cfg, data)
    loader_edge_index = data.edge_index.detach().cpu().contiguous()
    if loader_edge_index.numel() == 0:
        raise ValueError(f"{dataset}: loader returned an empty physical graph")
    if bool((loader_edge_index[0] == loader_edge_index[1]).any()):
        raise ValueError(f"{dataset}: self-loop found in physical graph before analysis")
    physical_edges = canonicalize_edges(loader_edge_index)
    degree_a = torch.bincount(loader_edge_index[0], minlength=data.num_nodes)
    connected_mask = degree_a > 0
    connected_mask_np = connected_mask.numpy().astype(bool, copy=False)
    connected_nodes = int(connected_mask.sum().item())
    isolated_nodes = int(data.num_nodes - connected_nodes)
    cka_subset_relative = deterministic_node_subset(
        connected_nodes,
        max_nodes=max_cka_nodes,
        seed=sample_seed,
    )
    connected_ids = torch.nonzero(connected_mask, as_tuple=False).flatten()
    cka_node_ids = connected_ids.index_select(0, cka_subset_relative)
    if not bool(connected_mask[cka_node_ids].all()):
        raise RuntimeError(f"{dataset}: CKA sample contains an isolated node")

    operator, operator_metadata = _build_fixed_operator(loader_edge_index, data.num_nodes, device)
    modality_results: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        modality_results["text"] = _analyze_modality(
            data.x_t, "text", connected_mask_np, cka_node_ids, operator, device
        )
        modality_results["visual"] = _analyze_modality(
            data.x_i, "visual", connected_mask_np, cka_node_ids, operator, device
        )

    for modality, result in modality_results.items():
        if abs(result["cka"][0] - 1.0) > 1e-5:
            raise RuntimeError(f"{dataset}/{modality}: CKA(Q0, X) failed sanity check: {result['cka'][0]}")
        raw_cosine = result["cosine_by_hop"][0]
        # At hop 0, special nodes are exactly the zero-norm reference nodes.
        raw_feature = data.x_t if modality == "text" else data.x_i
        raw_special = (torch.linalg.vector_norm(raw_feature.to(dtype=torch.float64), dim=1) <= 0.0).numpy()
        valid_connected = connected_mask_np & ~raw_special
        if valid_connected.any() and not np.allclose(raw_cosine[valid_connected], 1.0, atol=1e-5, rtol=1e-5):
            raise RuntimeError(f"{dataset}/{modality}: cosine(Q0_i, X_i) failed sanity check")

    summary_rows = [
        _make_summary_row(
            dataset,
            modality_results[modality],
            num_nodes=int(data.num_nodes),
            connected_nodes=connected_nodes,
            isolated_nodes=isolated_nodes,
        )
        for modality in ("text", "visual")
    ]
    dataset_dir = output_root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cosine_arrays = {
        f"text_cos_k{hop}": modality_results["text"]["cosine_by_hop"][hop]
        for hop in range(K_ANALYSIS + 1)
    }
    cosine_arrays.update({
        f"visual_cos_k{hop}": modality_results["visual"]["cosine_by_hop"][hop]
        for hop in range(K_ANALYSIS + 1)
    })
    np.savez_compressed(
        dataset_dir / "node_semantic_retention.npz",
        node_id=np.arange(data.num_nodes, dtype=np.int64),
        physical_degree=degree_a.numpy().astype(np.int64, copy=False),
        connected_mask=connected_mask_np,
        **cosine_arrays,
    )
    np.save(dataset_dir / "cka_node_ids.npy", cka_node_ids.numpy().astype(np.int64, copy=False))

    dataset_summary = {
        "analysis": "E0-B semantic retention under ordinary physical-graph propagation",
        "git_commit": _git("rev-parse", "HEAD"),
        "dataset": dataset,
        "formal_K": FORMAL_K[dataset],
        "K_analysis": K_ANALYSIS,
        "loader": {
            "loader": "src.data.load_mag_data",
            "task_name": "nc",
            "seed_for_loader": int(seed),
            "source": str(data.source),
            "feature_mode": "both",
            "make_undirected": bool(cfg.dataset.get("make_undirected", True)),
            "add_self_loops_config": bool(cfg.dataset.get("add_self_loops", False)),
            "loader_edge_count": int(loader_edge_index.size(1)),
            "num_physical_edges": int(physical_edges.size(0)),
            "loader_self_loop_count": 0,
            "duplicated_undirected_edge_removed_for_physical_audit": bool(
                int(loader_edge_index.size(1)) > int(physical_edges.size(0))
            ),
        },
        "features": {
            **feature_sources,
            "text_shape": [int(v) for v in data.x_t.shape],
            "visual_shape": [int(v) for v in data.x_i.shape],
            "text_dtype": str(data.x_t.dtype),
            "visual_dtype": str(data.x_i.dtype),
            "text_finite": modality_results["text"]["feature_finite"],
            "visual_finite": modality_results["visual"]["feature_finite"],
            "text_zero_norm_nodes": modality_results["text"]["raw_zero_norm_nodes"],
            "visual_zero_norm_nodes": modality_results["visual"]["raw_zero_norm_nodes"],
            "text_zero_norm_connected_nodes": modality_results["text"]["raw_zero_norm_connected_nodes"],
            "visual_zero_norm_connected_nodes": modality_results["visual"]["raw_zero_norm_connected_nodes"],
            "frozen_raw_input_features": True,
        },
        "population": {
            "num_nodes": int(data.num_nodes),
            "connected_nodes": connected_nodes,
            "isolated_nodes": isolated_nodes,
            "connected_fraction": float(connected_nodes / max(data.num_nodes, 1)),
            "physical_degree_definition": "degree of loader physical A before analysis self-loops",
        },
        "propagation": operator_metadata,
        "cka_sampling": {
            "seed": int(sample_seed),
            "max_nodes": int(max_cka_nodes),
            "candidate_population": "connected nodes only",
            "sample_count": int(cka_node_ids.numel()),
            "saved_node_ids": str(dataset_dir / "cka_node_ids.npy"),
        },
        "modalities": {
            modality: {
                key: value
                for key, value in result.items()
                if key not in {"cosine_by_hop"}
            }
            for modality, result in modality_results.items()
        },
        "summary_rows": summary_rows,
        "artifacts": {
            "node_semantic_retention": str(dataset_dir / "node_semantic_retention.npz"),
            "cka_node_ids": str(dataset_dir / "cka_node_ids.npy"),
        },
        "audit": {
            "q0_is_raw_feature_after_dtype_device_only": True,
            "cka_q0_sanity_passed": True,
            "cosine_q0_valid_node_sanity_passed": True,
            "all_propagated_states_finite": all(
                all(result["states_finite_by_hop"]) for result in modality_results.values()
            ),
            "zero_norm_policy": "assign cosine 0.0 for zero-norm state/reference and record counts; do not silently drop nodes",
            "training_started": False,
            "optimizer_step": False,
            "model_instantiated": False,
            "checkpoint_loaded": False,
            "checkpoint_written": False,
            "labels_used": False,
            "learned_operator_used": False,
            "final_configuration_modified": False,
            "propagation_definition_changed_after_results": False,
        },
    }
    _dump_json(dataset_dir / "summary.json", dataset_summary)

    del data, cfg, operator, modality_results, physical_edges, loader_edge_index
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "dataset": dataset,
        "modalities": {
            "text": dataset_summary["modalities"]["text"],
            "visual": dataset_summary["modalities"]["visual"],
        },
        "summary_rows": summary_rows,
    }, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analysis-only E0-B semantic retention study")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42, help="Existing NC loader seed; no labels are used")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_NODE_SAMPLE_SEED)
    parser.add_argument("--max-cka-nodes", type=int, default=MAX_CKA_NODES)
    args = parser.parse_args()
    if args.max_cka_nodes <= 0:
        raise ValueError("--max-cka-nodes must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    manifest = {
        "analysis": "E0-B semantic retention under ordinary physical-graph propagation",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_porcelain_at_start": _git("status", "--porcelain"),
        "datasets": list(args.datasets),
        "formal_K": {dataset: FORMAL_K[dataset] for dataset in args.datasets},
        "K_analysis": K_ANALYSIS,
        "device": str(device),
        "propagation_definition": "P = D_tilde^(-1/2) (A + I) D_tilde^(-1/2), A physical uniform-weight graph",
        "physical_edge_weight": 1.0,
        "reverse_edges_added_by_analysis": False,
        "one_self_loop_per_node": True,
        "cka_sampling_seed": int(args.sampling_seed),
        "cka_max_nodes": int(args.max_cka_nodes),
        "command": " ".join(sys.argv),
        "training_started": False,
        "optimizer_step": False,
        "model_instantiated": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "labels_used": False,
        "learned_operator_used": False,
        "final_configuration_modified": False,
        "propagation_definition_changed_after_results": False,
    }
    _dump_json(output_root / "run_manifest.json", manifest)

    figure_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_dataset_runtime: dict[str, float] = {}
    for index, dataset in enumerate(args.datasets, start=1):
        dataset_start = time.monotonic()
        print(f"[E0-B] {index}/{len(args.datasets)} loading/analyzing {dataset} on {device}", flush=True)
        figure_result, rows = _analyze_dataset(
            dataset,
            seed=args.seed,
            device=device,
            output_root=output_root,
            sample_seed=args.sampling_seed,
            max_cka_nodes=args.max_cka_nodes,
        )
        figure_results.append(figure_result)
        summary_rows.extend(rows)
        per_dataset_runtime[dataset] = float(time.monotonic() - dataset_start)
        for row in rows:
            print(
                f"[E0-B] {dataset}/{row['modality']}: "
                f"CKA K={row['cka_formal_K']:.6f}, K6={row['cka_k6']:.6f}; "
                f"cos K={row['cosine_formal_K']:.6f}, K6={row['cosine_k6']:.6f}",
                flush=True,
            )

    _write_summary_csv(output_root / "semantic_retention_summary.csv", summary_rows)
    _plot_trajectories(figure_results, output_root=output_root)
    manifest["per_dataset_runtime_seconds"] = per_dataset_runtime
    manifest["runtime_seconds"] = float(time.monotonic() - start_time)
    manifest["completed"] = True
    _dump_json(output_root / "run_manifest.json", manifest)
    print(f"[E0-B] wrote results to {output_root}", flush=True)


if __name__ == "__main__":
    main()
