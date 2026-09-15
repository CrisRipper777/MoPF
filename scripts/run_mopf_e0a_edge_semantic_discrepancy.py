"""E0-A: physical-edge cross-modal semantic discrepancy.

This is an inference-only, method-independent analysis.  It reuses the
project NC data loader to obtain the frozen text/visual features and the graph
support, then computes raw modality-specific cosine similarities on each
unique undirected physical edge.  No model, checkpoint, label, optimizer, or
training code is involved.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.data.graph_utils import canonicalize_edges  # noqa: E402
from src.data.loaders import resolve_path  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "e0_empirical_motivation" / "edge_semantic_discrepancy"
SUMMARY_FIELDS = (
    "dataset",
    "num_nodes",
    "num_physical_edges",
    "spearman_tv",
    "pearson_tv",
    "mean_rank_gap",
    "median_rank_gap",
    "q25_rank_gap",
    "q75_rank_gap",
    "q90_rank_gap",
    "text_high_visual_low_frac",
    "visual_high_text_low_frac",
)
DEFAULT_PLOT_SAMPLE_SEED = 20260914
DEFAULT_MAX_PLOT_EDGES = 50_000
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


def _compose_cfg(dataset: str, seed: int):
    # The model group is irrelevant to the loader, but keeping the standard
    # NC composition makes interpolation and dataset paths identical to the
    # existing NC pipeline.
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mlp",
        f"seed={seed}",
        "+dataset.feature_mode=both",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _average_rank_cdf(values: np.ndarray) -> np.ndarray:
    """Return average-tie empirical ranks scaled to [0, 1]."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    if not np.isfinite(values).all():
        raise ValueError("Cannot rank-normalize non-finite edge similarities")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    boundaries = np.r_[
        0,
        np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1,
        values.size,
    ]
    group_sizes = np.diff(boundaries)
    average_ranks = (boundaries[:-1] + 1.0 + boundaries[1:]) / 2.0
    sorted_ranks = np.repeat(average_ranks, group_sizes)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks / float(values.size)


def _safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size != second.size or first.size < 2:
        return float("nan")
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator <= EPS:
        return float("nan")
    return float(np.dot(first_centered, second_centered) / denominator)


def _edge_cosines(
    features: torch.Tensor,
    edges: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    """Compute edge cosine similarities in chunks without gradients."""
    features = torch.as_tensor(features).contiguous()
    edges = edges.to(dtype=torch.long, device="cpu").contiguous()
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got {tuple(features.shape)}")
    if features.size(0) == 0 or edges.size(1) == 0:
        return np.empty(0, dtype=np.float64)

    feature_device = features.to(device=device, dtype=torch.float32)
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, edges.size(1), chunk_size):
            stop = min(start + chunk_size, edges.size(1))
            src = edges[0, start:stop].to(device=device)
            dst = edges[1, start:stop].to(device=device)
            similarity = F.cosine_similarity(
                feature_device.index_select(0, src),
                feature_device.index_select(0, dst),
                dim=1,
                eps=1e-8,
            )
            output.append(similarity.detach().cpu().numpy().astype(np.float64, copy=False))
    del feature_device
    if not output:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(output)


def _feature_sources(cfg: Any, data: Any) -> dict[str, Any]:
    dataset_cfg = cfg.dataset
    source = str(dataset_cfg.source).lower()
    if source == "magb":
        return {
            "source_type": "magb",
            "text_path": str(resolve_path(dataset_cfg.text_feat_path)),
            "visual_path": str(resolve_path(dataset_cfg.image_feat_path)),
            "feature_dtype_config": str(dataset_cfg.get("feature_dtype", "float32")),
            "joint_feature_path": None,
        }
    if source == "mmgraph":
        return {
            "source_type": "mmgraph",
            "text_path": str(resolve_path(dataset_cfg.joint_feat_path)),
            "visual_path": str(resolve_path(dataset_cfg.joint_feat_path)),
            "joint_feature_path": str(resolve_path(dataset_cfg.joint_feat_path)),
            "joint_feature_layout": "[text, visual]",
            "text_slice": [0, int(dataset_cfg.text_dim)],
            "visual_slice": [int(dataset_cfg.text_dim), int(dataset_cfg.text_dim) + int(dataset_cfg.visual_dim)],
            "feature_dtype_config": str(dataset_cfg.get("feature_dtype", "float32")),
        }
    raise ValueError(f"Unsupported dataset source {source!r}")


def _sample_indices(num_edges: int, max_edges: int, seed: int) -> np.ndarray:
    if num_edges <= max_edges:
        return np.arange(num_edges, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_edges, size=max_edges, replace=False).astype(np.int64))


def _write_sample_csv(path: Path, sample: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "source_node",
        "target_node",
        "sim_text",
        "sim_visual",
        "rank_text",
        "rank_visual",
        "rank_gap",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in zip(*(sample[key] for key in fieldnames)):
            writer.writerow(dict(zip(fieldnames, row)))


def _analyze_dataset(
    dataset: str,
    *,
    seed: int,
    device: torch.device,
    output_root: Path,
    plot_seed: int,
    max_plot_edges: int,
    edge_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cfg = _compose_cfg(dataset, seed)
    data = load_mag_data(cfg, "nc", seed)
    if data.x_t is None or data.x_i is None:
        raise ValueError(f"{dataset}: the NC loader did not expose both text and visual features")

    feature_sources = _feature_sources(cfg, data)
    if data.x_t.size(0) != data.num_nodes or data.x_i.size(0) != data.num_nodes:
        raise ValueError(
            f"{dataset}: feature/node mismatch, text={tuple(data.x_t.shape)}, "
            f"visual={tuple(data.x_i.shape)}, nodes={data.num_nodes}"
        )

    loader_edge_index = data.edge_index.detach().cpu().contiguous()
    if loader_edge_index.ndim != 2 or loader_edge_index.size(0) != 2:
        raise ValueError(f"{dataset}: loader returned invalid edge_index {tuple(loader_edge_index.shape)}")
    loader_self_loops = int((loader_edge_index[0] == loader_edge_index[1]).sum().item())
    edge_index_without_self_loops = loader_edge_index[:, loader_edge_index[0] != loader_edge_index[1]]
    edges = canonicalize_edges(edge_index_without_self_loops)
    duplicate_undirected_edges_removed = int(edge_index_without_self_loops.size(1)) > int(edges.size(0))
    if edges.numel() and int(edges.max().item()) >= data.num_nodes:
        raise ValueError(f"{dataset}: edge index exceeds num_nodes={data.num_nodes}")
    if edges.size(0) == 0:
        raise ValueError(f"{dataset}: no non-self physical edges available")

    sim_text = _edge_cosines(data.x_t, edges.t(), device, edge_chunk_size)
    sim_visual = _edge_cosines(data.x_i, edges.t(), device, edge_chunk_size)
    if sim_text.size != edges.size(0) or sim_visual.size != edges.size(0):
        raise RuntimeError(f"{dataset}: similarity/edge count mismatch")
    if not np.isfinite(sim_text).all() or not np.isfinite(sim_visual).all():
        raise ValueError(f"{dataset}: non-finite raw cosine similarity")

    rank_text = _average_rank_cdf(sim_text)
    rank_visual = _average_rank_cdf(sim_visual)
    rank_gap = np.abs(rank_text - rank_visual)
    summary_row: dict[str, Any] = {
        "dataset": dataset,
        "num_nodes": int(data.num_nodes),
        "num_physical_edges": int(edges.size(0)),
        "spearman_tv": _safe_pearson(rank_text, rank_visual),
        "pearson_tv": _safe_pearson(sim_text, sim_visual),
        "mean_rank_gap": float(np.mean(rank_gap)),
        "median_rank_gap": float(np.median(rank_gap)),
        "q25_rank_gap": float(np.quantile(rank_gap, 0.25)),
        "q75_rank_gap": float(np.quantile(rank_gap, 0.75)),
        "q90_rank_gap": float(np.quantile(rank_gap, 0.90)),
        "text_high_visual_low_frac": float(np.mean((rank_text > 0.75) & (rank_visual < 0.25))),
        "visual_high_text_low_frac": float(np.mean((rank_visual > 0.75) & (rank_text < 0.25))),
    }

    sample_indices = _sample_indices(edges.size(0), max_plot_edges, plot_seed)
    sample = {
        "source_node": edges[sample_indices, 0].numpy().astype(np.int64, copy=False),
        "target_node": edges[sample_indices, 1].numpy().astype(np.int64, copy=False),
        "sim_text": sim_text[sample_indices],
        "sim_visual": sim_visual[sample_indices],
        "rank_text": rank_text[sample_indices],
        "rank_visual": rank_visual[sample_indices],
        "rank_gap": rank_gap[sample_indices],
    }

    dataset_dir = output_root / dataset
    sample_path = dataset_dir / "edge_rank_plot_sample.csv"
    _write_sample_csv(sample_path, sample)
    dataset_summary = {
        "analysis": "E0-A physical-edge cross-modal semantic discrepancy",
        "git_commit": _git("rev-parse", "HEAD"),
        "dataset": dataset,
        "loader": {
            "loader": "src.data.load_mag_data",
            "task_name": "nc",
            "seed_for_loader": int(seed),
            "source": str(data.source),
            "make_undirected": bool(cfg.dataset.get("make_undirected", True)),
            "add_self_loops_config": bool(cfg.dataset.get("add_self_loops", False)),
            "loader_edge_count_after_preprocessing": int(loader_edge_index.size(1)),
            "loader_self_loop_count_after_preprocessing": loader_self_loops,
            "edge_count_after_self_loop_exclusion": int(edge_index_without_self_loops.size(1)),
            "num_physical_edges": int(edges.size(0)),
            "duplicated_undirected_edge_removed": bool(duplicate_undirected_edges_removed),
            "self_loops_included_in_analysis": False,
            "physical_edge_definition": "unique canonical (min(i,j), max(i,j)) pairs from loader edge_index",
        },
        "features": {
            **feature_sources,
            "text_shape": [int(v) for v in data.x_t.shape],
            "visual_shape": [int(v) for v in data.x_i.shape],
            "text_dtype": str(data.x_t.dtype),
            "visual_dtype": str(data.x_i.dtype),
            "frozen_input_features": True,
        },
        "statistics": summary_row,
        "rank_normalization": {
            "method": "average rank / num_physical_edges",
            "tie_method": "average",
            "range": [0.0, 1.0],
            "opposite_quartile_thresholds": {
                "low_strict_lt": 0.25,
                "high_strict_gt": 0.75,
            },
        },
        "plot": {
            "sample_seed": int(plot_seed),
            "max_sample_size": int(max_plot_edges),
            "sample_size": int(sample_indices.size),
            "sample_csv": str(sample_path),
        },
        "audit": {
            "training_started": False,
            "optimizer_step": False,
            "checkpoint_loaded": False,
            "checkpoint_written": False,
            "model_parameters_used": False,
            "test_labels_used": False,
            "threshold_selection_from_results": False,
            "final_config_modified": False,
        },
    }
    _dump_json(dataset_dir / "summary.json", dataset_summary)

    del data, cfg, sim_text, sim_visual, rank_text, rank_visual, rank_gap, edges
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary_row, sample


def _plot_results(
    results: list[tuple[dict[str, Any], dict[str, np.ndarray]]],
    *,
    output_root: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    axes_flat = list(axes.flat)
    last_hexbin = None
    for axis, (row, sample) in zip(axes_flat, results):
        last_hexbin = axis.hexbin(
            sample["rank_text"],
            sample["rank_visual"],
            gridsize=45,
            mincnt=1,
            bins="log",
            cmap="viridis",
        )
        axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.0, alpha=0.75)
        axis.add_patch(Rectangle((0.0, 0.75), 0.25, 0.25, fill=False, linewidth=1.4, edgecolor="#d62728"))
        axis.add_patch(Rectangle((0.75, 0.0), 0.25, 0.25, fill=False, linewidth=1.4, edgecolor="#d62728"))
        axis.text(0.02, 0.97, "V high / T low", transform=axis.transAxes, va="top", ha="left", fontsize=8)
        axis.text(0.98, 0.03, "T high / V low", transform=axis.transAxes, va="bottom", ha="right", fontsize=8)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Text edge-similarity rank")
        axis.set_ylabel("Visual edge-similarity rank")
        axis.set_title(f"{row['dataset']}\nSpearman $\\rho$ = {row['spearman_tv']:.4f}")
        axis.grid(alpha=0.15)
    for axis in axes_flat[len(results):]:
        axis.axis("off")
    if last_hexbin is not None:
        fig.colorbar(last_hexbin, ax=axes_flat[: len(results)], label="Edge count (log scale)", shrink=0.88)
    fig.suptitle("E0-A Physical-edge cross-modal semantic discrepancy", fontsize=14)
    fig.savefig(output_root / "edge_semantic_discrepancy.png", dpi=220)
    fig.savefig(output_root / "edge_semantic_discrepancy.pdf")
    plt.close(fig)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in SUMMARY_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference-only E0-A edge semantic discrepancy analysis")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42, help="Loader seed; labels/splits are not used by this analysis")
    parser.add_argument("--device", default="cuda:0", help="Device for chunked cosine computation")
    parser.add_argument("--plot-seed", type=int, default=DEFAULT_PLOT_SAMPLE_SEED)
    parser.add_argument("--max-plot-edges", type=int, default=DEFAULT_MAX_PLOT_EDGES)
    parser.add_argument("--edge-chunk-size", type=int, default=DEFAULT_EDGE_CHUNK_SIZE)
    args = parser.parse_args()
    if args.max_plot_edges <= 0:
        raise ValueError("--max-plot-edges must be positive")
    if args.edge_chunk_size <= 0:
        raise ValueError("--edge-chunk-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "analysis": "E0-A physical-edge cross-modal semantic discrepancy",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "datasets": list(args.datasets),
        "loader_seed": int(args.seed),
        "device": str(device),
        "plot_sampling_seed": int(args.plot_seed),
        "max_plot_edges": int(args.max_plot_edges),
        "edge_chunk_size": int(args.edge_chunk_size),
        "command": " ".join(sys.argv),
        "training_started": False,
        "optimizer_step": False,
        "checkpoint_loaded": False,
        "checkpoint_written": False,
        "final_config_modified": False,
        "test_labels_used": False,
        "thresholds": {"low": 0.25, "high": 0.75, "strict_inequalities": True},
    }
    _dump_json(output_root / "run_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    plot_results: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    for index, dataset in enumerate(args.datasets, start=1):
        print(f"[E0-A] {index}/{len(args.datasets)} loading/analyzing {dataset} on {device}", flush=True)
        row, sample = _analyze_dataset(
            dataset,
            seed=args.seed,
            device=device,
            output_root=output_root,
            plot_seed=args.plot_seed,
            max_plot_edges=args.max_plot_edges,
            edge_chunk_size=args.edge_chunk_size,
        )
        rows.append(row)
        plot_results.append((row, sample))
        print(
            f"[E0-A] {dataset}: edges={row['num_physical_edges']} "
            f"spearman={row['spearman_tv']:.6f} median_gap={row['median_rank_gap']:.6f}",
            flush=True,
        )

    _write_summary_csv(output_root / "edge_discrepancy_summary.csv", rows)
    _plot_results(plot_results, output_root=output_root)
    print(f"[E0-A] wrote results to {output_root}", flush=True)


if __name__ == "__main__":
    main()
