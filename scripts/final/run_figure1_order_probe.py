"""Build model-independent evidence for Figure 1(c).

This is a lightweight diagnostic, not MGSC training.  For each raw modality
feature matrix it constructs fixed symmetric-normalized physical-graph
propagation states at orders 0..K and fits the same linear probe used by M1.
Only the official train split (with a deterministic internal train/dev split)
is used for probe fitting and selection; official validation is read once for
the reported profile and test labels are never indexed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import scatter
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.final.run_context_demand import (  # noqa: E402
    _load_cfg,
    deterministic_stratified_split,
    fit_linear_probe,
)
from src.data import load_mag_data  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
MODALITIES = ("text", "visual")
SEEDS = (42, 43, 44)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _unique_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index, _ = remove_self_loops(edge_index.long())
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0).contiguous()
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise ValueError("physical edge index is outside the node range")
    return torch.unique(edge_index.t().contiguous(), dim=0).t().contiguous()


def _fixed_order_states(feature: torch.Tensor, edge_index: torch.Tensor, max_order: int) -> list[torch.Tensor]:
    """Return raw L2-normalized states under a fixed physical graph operator."""
    h0 = F.normalize(feature.float().cpu().contiguous(), p=2.0, dim=1)
    num_nodes = int(h0.size(0))
    weights = torch.ones(edge_index.size(1), dtype=h0.dtype)
    norm_index, norm_weight = gcn_norm(
        edge_index,
        edge_weight=weights,
        num_nodes=num_nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=h0.dtype,
    )
    states = [h0]
    current = h0
    for _ in range(int(max_order)):
        if norm_index.numel() == 0:
            current = torch.zeros_like(h0)
        else:
            src, dst = norm_index
            current = scatter(
                current[src] * norm_weight.unsqueeze(-1),
                dst,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
        current = F.normalize(current, p=2.0, dim=1)
        states.append(current.contiguous())
    return states


def _raw_relation_discrepancy(
    x_text: torch.Tensor, x_visual: torch.Tensor, edge_index: torch.Tensor
) -> dict[str, float | int]:
    """Compute relation semantic discrepancy without a learned model."""
    text = F.normalize(x_text.float(), p=2.0, dim=1)
    visual = F.normalize(x_visual.float(), p=2.0, dim=1)
    if edge_index.numel() == 0:
        return {"physical_edges": 0, "mean_abs_cosine_difference": 0.0, "std_abs_cosine_difference": 0.0}
    src, dst = edge_index
    text_cos = (text[src] * text[dst]).sum(dim=-1)
    visual_cos = (visual[src] * visual[dst]).sum(dim=-1)
    discrepancy = (text_cos - visual_cos).abs()
    return {
        "physical_edges": int(edge_index.size(1)),
        "mean_abs_cosine_difference": float(discrepancy.mean()),
        "std_abs_cosine_difference": float(discrepancy.std(unbiased=False)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {result}, but CUDA is unavailable")
    return result


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    relation_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    split_fallbacks: dict[str, Any] = {}
    profile_rows: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        cfg = _load_cfg(dataset_name, SEEDS[0], args.data_root, args.split_root)
        data = load_mag_data(cfg, "nc", SEEDS[0])
        if data.x_t is None or data.x_i is None:
            raise ValueError(f"{dataset_name} has no independent text/visual features")
        edge_index = _unique_edges(data.edge_index.detach().cpu(), data.num_nodes)
        discrepancy = _raw_relation_discrepancy(data.x_t, data.x_i, edge_index)
        relation_rows.append({"dataset": dataset_name, **discrepancy, "source": "raw_features_physical_edges"})

        # Build all fixed states once per modality.  The states are detached
        # raw representations; only the linear probe is fitted below.
        state_banks = {
            "text": _fixed_order_states(data.x_t, edge_index, args.max_order),
            "visual": _fixed_order_states(data.x_i, edge_index, args.max_order),
        }
        labels = data.y.detach().cpu().long()
        test_idx = data.test_idx.detach().cpu().long()
        labels_masked = labels.clone()
        labels_masked[test_idx] = -1
        for seed in SEEDS:
            train_idx = data.train_idx.detach().cpu().long()
            val_idx = data.val_idx.detach().cpu().long()
            probe_train, probe_dev, fallback = deterministic_stratified_split(train_idx, labels_masked, seed)
            split_fallbacks[f"{dataset_name}:{seed}"] = fallback
            for modality in MODALITIES:
                for order, features in enumerate(state_banks[modality]):
                    result = fit_linear_probe(
                        features,
                        labels_masked,
                        probe_train,
                        probe_dev,
                        val_idx,
                        int(data.num_classes),
                        seed,
                        device,
                        int(args.epochs),
                        int(args.patience),
                        float(args.learning_rate),
                        float(args.weight_decay),
                        float(args.min_delta),
                        int(args.eval_batch_size),
                    )
                    metric_rows.append({
                        "dataset": dataset_name,
                        "modality": modality,
                        "order": order,
                        "seed": seed,
                        "val_loss": result["val_loss"],
                        "val_accuracy": result["val_accuracy"],
                        "val_macro_f1": result["val_macro_f1"],
                        "best_epoch": result["best_epoch"],
                        "probe_train_nodes": int(probe_train.numel()),
                        "probe_dev_nodes": int(probe_dev.numel()),
                        "official_val_nodes": int(val_idx.numel()),
                        "test_labels_used": False,
                    })

    for dataset in args.datasets:
        for modality in MODALITIES:
            for order in range(args.max_order + 1):
                subset = [
                    row for row in metric_rows
                    if row["dataset"] == dataset and row["modality"] == modality and row["order"] == order
                ]
                losses = np.asarray([row["val_loss"] for row in subset], dtype=float)
                acc = np.asarray([row["val_accuracy"] for row in subset], dtype=float)
                f1 = np.asarray([row["val_macro_f1"] for row in subset], dtype=float)
                profile_rows.append({
                    "dataset": dataset,
                    "modality": modality,
                    "order": order,
                    "val_loss_mean": float(losses.mean()),
                    "val_loss_population_std": float(losses.std(ddof=0)),
                    "val_accuracy_mean": float(acc.mean()),
                    "val_accuracy_population_std": float(acc.std(ddof=0)),
                    "val_macro_f1_mean": float(f1.mean()),
                    "val_macro_f1_population_std": float(f1.std(ddof=0)),
                    "n_seeds": len(subset),
                    "evidence_scope": "raw_fixed_physical_graph_linear_probe",
                })

    _write_csv(output / "relation_discrepancy.csv", relation_rows, list(relation_rows[0]))
    _write_csv(output / "order_probe_metrics.csv", metric_rows, list(metric_rows[0]))
    _write_csv(output / "order_probe_summary.csv", profile_rows, list(profile_rows[0]))
    (output / "order_probe_split_fallbacks.json").write_text(json.dumps(split_fallbacks, indent=2), encoding="utf-8")
    manifest = {
        "scope": "Figure 1(c) model-independent diagnostic",
        "datasets": list(args.datasets),
        "seeds": list(SEEDS),
        "max_order": int(args.max_order),
        "device": str(device),
        "model_training": False,
        "probe": {
            "classifier": "Linear(feature_dim, num_classes)",
            "optimizer": "AdamW",
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "selection": "probe-dev CE",
        },
        "test_labels_used": False,
        "split_sources": {
            dataset: str(load_mag_data(_load_cfg(dataset, 42, args.data_root, args.split_root), "nc", 42).info.get("nc_split_path", ""))
            for dataset in args.datasets
        },
    }
    (output / "figure1c_order_probe_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", default="cuda:0", help="Torch device; default is GPU cuda:0.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/paper_figures/figure1_data"))
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
