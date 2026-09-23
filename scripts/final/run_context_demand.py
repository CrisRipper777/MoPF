"""Run the M1 Contextualization-Demand empirical study.

The study is deliberately independent of CoSI/MGSC checkpoints.  It loads the
official NC data splits, normalizes each frozen modality feature row, builds a
physical-neighborhood mean without self-loops, and trains only a linear probe
for each modality/lambda/seed combination.  Test labels are never indexed or
used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch_geometric.utils import remove_self_loops

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_mag_data  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


DATASETS = ("Movies", "Grocery", "ele-fashion")
MODALITIES = ("text", "visual")
SEEDS = (42, 43, 44)
LAMBDAS = (0.00, 0.25, 0.50, 0.75, 1.00)
DEGREE_BINS = (
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    (">10", 11, math.inf),
)

SUPPORTED_NON_GLOBAL = 0.25
SUPPORTED_ENTROPY = 0.35
SUPPORTED_ORACLE_GAP = 0.01
COLLAPSE_PROPORTION = 0.90


def _configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("context_demand")


def _parse_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    return device


def _load_cfg(dataset_name: str, seed: int, data_root: Path | None, split_root: Path | None):
    dataset_cfg = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / f"{dataset_name}.yaml")
    task_cfg = OmegaConf.load(REPO_ROOT / "configs" / "task" / "nc.yaml")
    base_cfg = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    paths = OmegaConf.to_container(base_cfg.paths, resolve=False)
    if data_root is not None:
        paths["data_root"] = str(data_root.resolve())
    if split_root is not None:
        paths["split_root"] = str(split_root.resolve())
    cfg = OmegaConf.create(
        {
            "seed": int(seed),
            "paths": paths,
            "dataset": dataset_cfg,
            "task": task_cfg,
        }
    )
    OmegaConf.resolve(cfg)
    return cfg


def _unique_physical_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index, _ = remove_self_loops(edge_index.long())
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0).contiguous()
    if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
        raise ValueError("physical edge index contains a node outside the loader graph")
    return torch.unique(edge_index.t().contiguous(), dim=0).t().contiguous()


def build_context_states(
    feature: torch.Tensor, edge_index: torch.Tensor
) -> tuple[dict[float, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalized local/context mixtures from one modality only.

    Returns lambda-indexed features, normalized local features, normalized
    neighborhood features, and the physical degree vector.  No learned model,
    labels, self-loop, or semantic-neighbor operation is used here.
    """
    feature = feature.float().cpu().contiguous()
    h_local = F.normalize(feature, p=2.0, dim=1)
    num_nodes = int(h_local.size(0))
    neighbor_sum = torch.zeros_like(h_local)
    if edge_index.numel():
        src, dst = edge_index
        neighbor_sum.index_add_(0, dst, h_local[src])
        degree = torch.bincount(dst, minlength=num_nodes).to(dtype=h_local.dtype)
    else:
        degree = torch.zeros(num_nodes, dtype=h_local.dtype)
    neighborhood = neighbor_sum / degree.clamp_min(1.0).unsqueeze(1)
    neighborhood = F.normalize(neighborhood, p=2.0, dim=1)
    states = {
        float(lambda_value): F.normalize(
            (1.0 - float(lambda_value)) * h_local + float(lambda_value) * neighborhood,
            p=2.0,
            dim=1,
        )
        for lambda_value in LAMBDAS
    }
    isolated = degree == 0
    return states, h_local, neighborhood, degree.to(dtype=torch.long)


def deterministic_stratified_split(
    train_idx: torch.Tensor, labels: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int | str]]]:
    """Split official train nodes into deterministic 90/10 train/dev subsets.

    Classes with one training example cannot support a class-preserving dev
    example; those classes remain in probe_train and are recorded as a fallback.
    """
    train_idx = train_idx.detach().cpu().long()
    labels = labels.detach().cpu().long()
    rng = np.random.default_rng(int(seed))
    probe_train: list[int] = []
    probe_dev: list[int] = []
    fallback: list[dict[str, int | str]] = []
    train_labels = labels[train_idx]
    for class_value in sorted({int(value) for value in train_labels.tolist()}):
        class_nodes = train_idx[train_labels == class_value].tolist()
        class_nodes = [int(node) for node in class_nodes]
        permutation = rng.permutation(len(class_nodes))
        shuffled = [class_nodes[int(position)] for position in permutation]
        if len(shuffled) < 2:
            probe_train.extend(shuffled)
            fallback.append({"class": class_value, "count": len(shuffled), "reason": "class_count_lt_2"})
            continue
        dev_count = max(1, int(round(0.10 * len(shuffled))))
        dev_count = min(dev_count, len(shuffled) - 1)
        probe_dev.extend(shuffled[:dev_count])
        probe_train.extend(shuffled[dev_count:])
    if not probe_train:
        raise ValueError("deterministic probe split produced no training nodes")
    probe_train_tensor = torch.tensor(sorted(probe_train), dtype=torch.long)
    probe_dev_tensor = torch.tensor(sorted(probe_dev), dtype=torch.long)
    return probe_train_tensor, probe_dev_tensor, fallback


def _eval_linear(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    num_classes: int,
    batch_size: int,
) -> tuple[float, float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    loss_values: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(indices.numel()), batch_size):
            batch_indices = indices[start : start + batch_size]
            logits = model(features[batch_indices].to(device))
            batch_labels = labels[batch_indices].to(device)
            loss_values.append(F.cross_entropy(logits, batch_labels, reduction="none").cpu())
            predictions.append(logits.argmax(dim=-1).cpu())
            targets.append(batch_labels.cpu())
    if not loss_values:
        raise ValueError("cannot evaluate a linear probe on an empty split")
    node_loss = torch.cat(loss_values)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    accuracy = float((prediction == target).float().mean().item())
    macro_f1 = float(
        f1_score(
            target.numpy(),
            prediction.numpy(),
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        )
    )
    return float(node_loss.mean().item()), accuracy, macro_f1, node_loss, prediction == target


def fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    probe_train_idx: torch.Tensor,
    probe_dev_idx: torch.Tensor,
    official_val_idx: torch.Tensor,
    num_classes: int,
    seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    min_delta: float,
    eval_batch_size: int,
) -> dict[str, Any]:
    """Train one linear probe and select it using only probe-dev CE."""
    set_seed(int(seed))
    model = nn.Linear(int(features.size(1)), int(num_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    labels = labels.long().cpu()
    x_train = features[probe_train_idx].to(device)
    y_train = labels[probe_train_idx].to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_loss = math.inf
    best_epoch = 0
    remaining = int(patience)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = F.cross_entropy(model(x_train), y_train)
        train_loss.backward()
        optimizer.step()
        model.eval()
        if probe_dev_idx.numel():
            with torch.no_grad():
                dev_loss = float(
                    F.cross_entropy(
                        model(features[probe_dev_idx].to(device)),
                        labels[probe_dev_idx].to(device),
                    ).item()
                )
        else:
            dev_loss = float(train_loss.detach().item())
        if dev_loss < best_dev_loss - float(min_delta):
            best_dev_loss = dev_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            remaining = int(patience)
        else:
            remaining -= 1
            if remaining <= 0:
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = int(epoch)
    model.load_state_dict(best_state)
    val_loss, val_accuracy, val_macro_f1, node_loss, correct = _eval_linear(
        model,
        features,
        labels,
        official_val_idx,
        device,
        int(num_classes),
        int(eval_batch_size),
    )
    return {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "node_loss": node_loss,
        "correct": correct,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _entropy(proportions: list[float]) -> float:
    values = np.asarray(proportions, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-(values * np.log(values)).sum() / math.log(len(LAMBDAS)))


def _distribution(preferred: dict[int, float], scope: str, seed: int | str, dataset: str, modality: str) -> list[dict[str, Any]]:
    eligible = list(preferred.values())
    counts = {float(lambda_value): 0 for lambda_value in LAMBDAS}
    for value in eligible:
        counts[float(value)] += 1
    denominator = max(len(eligible), 1)
    return [
        {
            "dataset": dataset,
            "modality": modality,
            "scope": scope,
            "seed": seed,
            "lambda": f"{lambda_value:.2f}",
            "count": counts[float(lambda_value)],
            "proportion": counts[float(lambda_value)] / denominator,
            "eligible_nodes": len(eligible),
        }
        for lambda_value in LAMBDAS
    ]


def _degree_bin(degree: int) -> str:
    for name, lower, upper in DEGREE_BINS:
        if lower <= degree <= upper:
            return name
    raise ValueError(f"degree {degree} did not match a configured bin")


def _make_report(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    disagreement_rows: list[dict[str, Any]],
    gate: dict[str, Any],
    fallback_records: dict[str, Any],
) -> Path:
    report_path = REPO_ROOT / "docs" / "final" / "context_demand_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M1 Contextualization-Demand Report",
        "",
        "## Experimental definition",
        "",
        "M1 evaluates whether node-local modality semantics and physical-neighborhood context have heterogeneous utility in the official NC splits of Movies, Grocery, and ele-fashion. For each modality, frozen raw features are row-wise L2-normalized, a simple mean over actual non-self-loop physical neighbors is formed, and five normalized mixtures are evaluated: `C(lambda) = normalize((1-lambda)H + lambda N)` for lambda in `{0.00, 0.25, 0.50, 0.75, 1.00}`.",
        "",
        "The only learned component is `Linear(feature_dim, num_classes)`. Every modality/lambda/seed uses the same AdamW hyperparameters, CE objective, deterministic 90/10 stratified split inside official train, and probe-dev CE early stopping. Official validation is evaluated once after probe checkpoint selection. Official test labels are not indexed, used for training, used for model selection, or used in any M1 statistic.",
        "",
        "Physical degree zero nodes are retained in the node-level loss export but are explicitly excluded from preferred-lambda primary statistics. The node-wise oracle is descriptive only: it is not a realizable model and is not test performance.",
        "",
        "## Core statistics",
        "",
        "| Dataset | Modality | Global-best lambda | Preference entropy | Non-global ratio | T/V disagreement | Mean abs T/V lambda difference | Oracle relative CE gap | Eligible val nodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    disagreement_by_dataset = {row["dataset"]: row for row in disagreement_rows if row["scope"] == "aggregated"}
    for row in summary_rows:
        tv = disagreement_by_dataset.get(row["dataset"], {})
        lines.append(
            f"| {row['dataset']} | {row['modality']} | {float(row['global_best_lambda']):.2f} | {float(row['normalized_preference_entropy']):.4f} | {float(row['non_global_preference_ratio']):.4f} | {float(tv.get('disagreement_ratio', 0.0)):.4f} | {float(tv.get('mean_absolute_lambda_difference', 0.0)):.4f} | {float(row['oracle_relative_gap']):.4f} | {row['eligible_nodes']} |"
        )
    lines.extend(
        [
            "",
            "Global-best lambda minimizes mean official-validation CE across seeds for the indicated modality. Preference entropy and non-global ratio are computed from aggregated node-wise CE across seeds. The T/V disagreement is computed on simultaneously valid, non-isolated validation nodes.",
            "",
            "## Dataset decisions and Gate A",
            "",
            "The predeclared engineering support rule is: a dataset is supported when at least one modality has non-global preference ratio ≥ 0.25, normalized preference entropy ≥ 0.35, and node-wise oracle relative CE gap ≥ 0.01. Gate A is PASS when at least two of three datasets are supported and no dataset has both modalities collapsing to the same lambda with >90% of eligible nodes.",
            "",
            f"**Gate A: {gate['gate_a']}**",
            "",
            f"Supported datasets: {', '.join(gate['supported_datasets']) if gate['supported_datasets'] else 'none'}. Collapse violation datasets: {', '.join(gate['collapse_datasets']) if gate['collapse_datasets'] else 'none'}.",
            "",
            f"The secondary descriptive weak-evidence count is {gate['weak_evidence_dataset_count']} dataset(s); this count does not relax the PASS thresholds. The implementation decision is therefore **{'enter P1' if gate['enter_p1'] else 'stop after M1'}**.",
            "",
            "If Gate A is not sufficient to justify P1, the recommended fallback is relation-calibrated multi-order state-bank modeling, without an adaptive gate search in this round.",
            "",
            "## Stability and probe fallbacks",
            "",
            "Seed-wise distributions, entropy, and modality disagreement are available in `stability.csv`; node-level losses are in `node_lambda_losses.csv`; global probe metrics are in `global_probe_metrics.csv`; and the degree-stratified descriptive analysis is in `preferred_lambda_vs_degree.csv`.",
            "",
            "Deterministic probe split fallback records are stored in `outputs/final/context_demand/probe_split_fallbacks.json`. Fallbacks only occur when a class has fewer than two official training examples and therefore cannot contribute one example to both probe-train and probe-dev.",
            "",
            "## Interpretation boundary",
            "",
            "These results test contextualization demand using frozen modality features and a linear probe. They do not establish that an adaptive neural gate improves NC/LP performance, and they do not justify interpreting semantic similarity as relation reliability. Any node-wise oracle gap is a descriptive motivation upper bound, not a realizable model or test result.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    logger = _configure_logging()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _parse_device(args.device)
    logger.info("M1 device=%s datasets=%s seeds=%s", device, args.datasets, SEEDS)

    global_metric_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    fallback_records: dict[str, Any] = {}
    loss_store: dict[tuple[str, str, int, float], dict[int, float]] = {}
    correct_store: dict[tuple[str, str, int, float], dict[int, bool]] = {}
    node_metadata: dict[str, dict[int, dict[str, Any]]] = {}
    distributions: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    degree_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    dataset_payload: dict[str, Any] = {}

    for dataset_name in args.datasets:
        cfg = _load_cfg(dataset_name, SEEDS[0], args.data_root, args.split_root)
        data = load_mag_data(cfg, "nc", SEEDS[0])
        if data.x_t is None or data.x_i is None:
            raise ValueError(f"{dataset_name} did not expose both text and visual features")
        train_idx = data.train_idx.detach().cpu().long()
        val_idx = data.val_idx.detach().cpu().long()
        test_idx = data.test_idx.detach().cpu().long()
        if torch.isin(train_idx, test_idx).any() or torch.isin(val_idx, test_idx).any():
            raise AssertionError(f"{dataset_name} official NC split overlaps test indices")
        # Only these label slices are permitted below.  Mask the official test
        # slice immediately so later code cannot accidentally consume a test
        # label, even if a future refactor indexes the label tensor broadly.
        allowed_labels = data.y.detach().cpu().long().clone()
        allowed_labels[test_idx] = -1
        train_labels = allowed_labels[train_idx]
        val_labels = allowed_labels[val_idx]
        if (train_labels < 0).any() or (train_labels >= int(data.num_classes)).any():
            raise ValueError(f"{dataset_name} official train split contains invalid labels")
        if (val_labels < 0).any() or (val_labels >= int(data.num_classes)).any():
            raise ValueError(f"{dataset_name} official validation split contains invalid labels")
        physical_edges = _unique_physical_edges(data.edge_index.detach().cpu(), data.num_nodes)
        degree = None
        node_metadata[dataset_name] = {}
        for node in val_idx.tolist():
            node_metadata[dataset_name][int(node)] = {"target": int(allowed_labels[int(node)].item())}

        logger.info(
            "%s: nodes=%d physical_edges=%d train=%d val=%d test=%d",
            dataset_name,
            data.num_nodes,
            physical_edges.size(1),
            train_idx.numel(),
            val_idx.numel(),
            test_idx.numel(),
        )

        modality_features = {"text": data.x_t.detach().cpu(), "visual": data.x_i.detach().cpu()}
        for modality in MODALITIES:
            states, _, _, degree = build_context_states(modality_features[modality], physical_edges)
            for node in val_idx.tolist():
                node_metadata[dataset_name][int(node)]["degree"] = int(degree[int(node)].item())
                node_metadata[dataset_name][int(node)]["is_isolated"] = bool(degree[int(node)].item() == 0)
            for seed in SEEDS:
                probe_train_idx, probe_dev_idx, fallbacks = deterministic_stratified_split(train_idx, allowed_labels, seed)
                fallback_records[f"{dataset_name}/seed{seed}"] = {
                    "probe_train_nodes": int(probe_train_idx.numel()),
                    "probe_dev_nodes": int(probe_dev_idx.numel()),
                    "fallback_classes": fallbacks,
                }
                for lambda_value in LAMBDAS:
                    result = fit_linear_probe(
                        states[float(lambda_value)],
                        allowed_labels,
                        probe_train_idx,
                        probe_dev_idx,
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
                    key = (dataset_name, modality, int(seed), float(lambda_value))
                    node_loss_map = {
                        int(node): float(loss)
                        for node, loss in zip(val_idx.tolist(), result["node_loss"].tolist())
                    }
                    correct_map = {
                        int(node): bool(correct)
                        for node, correct in zip(val_idx.tolist(), result["correct"].tolist())
                    }
                    loss_store[key] = node_loss_map
                    correct_store[key] = correct_map
                    for node in val_idx.tolist():
                        metadata = node_metadata[dataset_name][int(node)]
                        node_rows.append(
                            {
                                "dataset": dataset_name,
                                "node_id": int(node),
                                "modality": modality,
                                "lambda": f"{lambda_value:.2f}",
                                "seed": int(seed),
                                "target": metadata["target"],
                                "ce_loss": node_loss_map[int(node)],
                                "correct": int(correct_map[int(node)]),
                                "is_isolated": int(metadata["is_isolated"]),
                                "degree": metadata["degree"],
                            }
                        )
                    global_metric_rows.append(
                        {
                            "dataset": dataset_name,
                            "modality": modality,
                            "lambda": f"{lambda_value:.2f}",
                            "seed": int(seed),
                            "val_loss": result["val_loss"],
                            "val_accuracy": result["val_accuracy"],
                            "val_macro_f1": result["val_macro_f1"],
                            "best_epoch": result["best_epoch"],
                            "best_probe_dev_loss": result["best_dev_loss"],
                            "probe_train_nodes": int(probe_train_idx.numel()),
                            "probe_dev_nodes": int(probe_dev_idx.numel()),
                            "fallback_class_count": len(fallbacks),
                        }
                    )
                    logger.info(
                        "%s %s lambda=%.2f seed=%d val_loss=%.4f val_acc=%.4f best_epoch=%d",
                        dataset_name,
                        modality,
                        lambda_value,
                        seed,
                        result["val_loss"],
                        result["val_accuracy"],
                        result["best_epoch"],
                    )

            modality_summary_rows = [
                row for row in global_metric_rows if row["dataset"] == dataset_name and row["modality"] == modality
            ]
            mean_by_lambda = {
                float(lambda_value): float(
                    np.mean([
                        float(row["val_loss"])
                        for row in modality_summary_rows
                        if abs(float(row["lambda"]) - float(lambda_value)) < 1e-8
                    ])
                )
                for lambda_value in LAMBDAS
            }
            global_best = min(LAMBDAS, key=lambda value: (mean_by_lambda[float(value)], float(value)))
            aggregate_losses: dict[int, dict[float, float]] = {}
            for node in val_idx.tolist():
                aggregate_losses[int(node)] = {
                    float(lambda_value): float(
                        np.mean([
                            loss_store[(dataset_name, modality, seed, float(lambda_value))][int(node)]
                            for seed in SEEDS
                        ])
                    )
                    for lambda_value in LAMBDAS
                }
            valid_nodes = [node for node in val_idx.tolist() if not node_metadata[dataset_name][int(node)]["is_isolated"]]
            aggregate_preferred = {
                int(node): min(LAMBDAS, key=lambda value: (aggregate_losses[int(node)][float(value)], float(value)))
                for node in valid_nodes
            }
            distributions.extend(_distribution(aggregate_preferred, "aggregated", "", dataset_name, modality))
            seed_preferred: dict[int, dict[int, float]] = {}
            for seed in SEEDS:
                preferred = {
                    int(node): min(
                        LAMBDAS,
                        key=lambda value: (loss_store[(dataset_name, modality, seed, float(value))][int(node)], float(value)),
                    )
                    for node in valid_nodes
                }
                seed_preferred[int(seed)] = preferred
                distributions.extend(_distribution(preferred, "seed", seed, dataset_name, modality))
                proportions = [
                    row["proportion"]
                    for row in distributions[-len(LAMBDAS) :]
                ]
                stability_rows.append(
                    {
                        "dataset": dataset_name,
                        "modality": modality,
                        "seed": seed,
                        "normalized_preference_entropy": _entropy([float(value) for value in proportions]),
                        "non_global_preference_ratio": float(
                            np.mean([value != global_best for value in preferred.values()])
                        )
                        if preferred
                        else 0.0,
                        "eligible_nodes": len(preferred),
                    }
                )
            aggregate_distribution = [
                row["proportion"]
                for row in distributions
                if row["dataset"] == dataset_name
                and row["modality"] == modality
                and row["scope"] == "aggregated"
            ]
            global_loss = float(np.mean([aggregate_losses[node][float(global_best)] for node in valid_nodes])) if valid_nodes else float("nan")
            oracle_loss = float(np.mean([min(aggregate_losses[node].values()) for node in valid_nodes])) if valid_nodes else float("nan")
            absolute_gap = global_loss - oracle_loss
            relative_gap = absolute_gap / global_loss if global_loss > 0 else 0.0
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "modality": modality,
                    "global_best_lambda": f"{global_best:.2f}",
                    "global_best_mean_val_ce": mean_by_lambda[float(global_best)],
                    "normalized_preference_entropy": _entropy([float(value) for value in aggregate_distribution]),
                    "non_global_preference_ratio": float(
                        np.mean([value != global_best for value in aggregate_preferred.values()])
                    )
                    if aggregate_preferred
                    else 0.0,
                    "global_ce_on_valid_val": global_loss,
                    "oracle_ce_on_valid_val": oracle_loss,
                    "oracle_absolute_gap": absolute_gap,
                    "oracle_relative_gap": relative_gap,
                    "eligible_nodes": len(valid_nodes),
                    "isolated_val_nodes": int(len(val_idx) - len(valid_nodes)),
                    "mean_val_ce_lambda_0.00": mean_by_lambda[0.00],
                    "mean_val_ce_lambda_0.25": mean_by_lambda[0.25],
                    "mean_val_ce_lambda_0.50": mean_by_lambda[0.50],
                    "mean_val_ce_lambda_0.75": mean_by_lambda[0.75],
                    "mean_val_ce_lambda_1.00": mean_by_lambda[1.00],
                }
            )
            for degree_bin, lower, upper in DEGREE_BINS[1:]:
                bin_nodes = [
                    node
                    for node in valid_nodes
                    if lower <= node_metadata[dataset_name][int(node)]["degree"] <= upper
                ]
                if not bin_nodes:
                    continue
                preferred_bin = {node: aggregate_preferred[node] for node in bin_nodes}
                distributions_bin = _distribution(preferred_bin, "aggregated", "", dataset_name, modality)
                for row in distributions_bin:
                    row["degree_bin"] = degree_bin
                    row["eligible_nodes_in_degree_bin"] = len(bin_nodes)
                    degree_rows.append(row)
            dataset_payload.setdefault(dataset_name, {})[modality] = {
                "global_best_lambda": global_best,
                "mean_val_ce_by_lambda": mean_by_lambda,
                "aggregate_preferred": aggregate_preferred,
                "seed_preferred": seed_preferred,
                "valid_nodes": valid_nodes,
            }

        text_pref = dataset_payload[dataset_name]["text"]["aggregate_preferred"]
        visual_pref = dataset_payload[dataset_name]["visual"]["aggregate_preferred"]
        common_nodes = sorted(set(text_pref).intersection(visual_pref))
        mismatch = [node for node in common_nodes if text_pref[node] != visual_pref[node]]
        disagreement_rows.append(
            {
                "dataset": dataset_name,
                "scope": "aggregated",
                "seed": "",
                "disagreement_ratio": len(mismatch) / max(len(common_nodes), 1),
                "mean_absolute_lambda_difference": float(
                    np.mean([abs(text_pref[node] - visual_pref[node]) for node in common_nodes])
                )
                if common_nodes
                else 0.0,
                "common_valid_nodes": len(common_nodes),
            }
        )
        for seed in SEEDS:
            text_seed = dataset_payload[dataset_name]["text"]["seed_preferred"][seed]
            visual_seed = dataset_payload[dataset_name]["visual"]["seed_preferred"][seed]
            common_seed = sorted(set(text_seed).intersection(visual_seed))
            mismatch_seed = [node for node in common_seed if text_seed[node] != visual_seed[node]]
            disagreement_rows.append(
                {
                    "dataset": dataset_name,
                    "scope": "seed",
                    "seed": seed,
                    "disagreement_ratio": len(mismatch_seed) / max(len(common_seed), 1),
                    "mean_absolute_lambda_difference": float(
                        np.mean([abs(text_seed[node] - visual_seed[node]) for node in common_seed])
                    )
                    if common_seed
                    else 0.0,
                    "common_valid_nodes": len(common_seed),
                }
            )

    summary_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        summary_by_dataset[row["dataset"]].append(row)
    supported_datasets: list[str] = []
    weak_evidence_datasets: list[str] = []
    collapse_datasets: list[str] = []
    dataset_gate_rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        rows = summary_by_dataset[dataset_name]
        supported_modalities = [
            row["modality"]
            for row in rows
            if float(row["non_global_preference_ratio"]) >= SUPPORTED_NON_GLOBAL
            and float(row["normalized_preference_entropy"]) >= SUPPORTED_ENTROPY
            and float(row["oracle_relative_gap"]) >= SUPPORTED_ORACLE_GAP
        ]
        weak_modalities = [
            row["modality"]
            for row in rows
            if float(row["non_global_preference_ratio"]) >= SUPPORTED_NON_GLOBAL
            and float(row["normalized_preference_entropy"]) >= SUPPORTED_ENTROPY
        ]
        dominant = {
            row["modality"]: max(
                [
                    row_distribution
                    for row_distribution in distributions
                    if row_distribution["dataset"] == dataset_name
                    and row_distribution["modality"] == row["modality"]
                    and row_distribution["scope"] == "aggregated"
                ],
                key=lambda item: float(item["proportion"]),
            )
            for row in rows
        }
        collapse = (
            all(float(dominant[modality]["proportion"]) > COLLAPSE_PROPORTION for modality in MODALITIES)
            and dominant["text"]["lambda"] == dominant["visual"]["lambda"]
        )
        if supported_modalities:
            supported_datasets.append(dataset_name)
        if weak_modalities:
            weak_evidence_datasets.append(dataset_name)
        if collapse:
            collapse_datasets.append(dataset_name)
        dataset_gate_rows.append(
            {
                "dataset": dataset_name,
                "supported": bool(supported_modalities),
                "supported_modalities": ",".join(supported_modalities),
                "weak_evidence": bool(weak_modalities),
                "weak_modalities": ",".join(weak_modalities),
                "collapse_violation": bool(collapse),
                "text_dominant_lambda": dominant["text"]["lambda"],
                "text_dominant_proportion": dominant["text"]["proportion"],
                "visual_dominant_lambda": dominant["visual"]["lambda"],
                "visual_dominant_proportion": dominant["visual"]["proportion"],
            }
        )
    if len(supported_datasets) >= 2 and not collapse_datasets:
        gate_a = "PASS"
    elif len(weak_evidence_datasets) >= 2 and not collapse_datasets:
        gate_a = "WEAK"
    else:
        gate_a = "FAIL"
    gate = {
        "gate_a": gate_a,
        "supported_datasets": supported_datasets,
        "weak_evidence_datasets": weak_evidence_datasets,
        "weak_evidence_dataset_count": len(weak_evidence_datasets),
        "collapse_datasets": collapse_datasets,
        "enter_p1": gate_a in {"PASS", "WEAK"} and len(weak_evidence_datasets) >= 2,
        "thresholds": {
            "non_global_ratio": SUPPORTED_NON_GLOBAL,
            "entropy": SUPPORTED_ENTROPY,
            "oracle_relative_gap": SUPPORTED_ORACLE_GAP,
            "collapse_proportion": COLLAPSE_PROPORTION,
        },
        "dataset_rows": dataset_gate_rows,
    }

    _write_csv(
        output_dir / "global_probe_metrics.csv",
        global_metric_rows,
        [
            "dataset", "modality", "lambda", "seed", "val_loss", "val_accuracy", "val_macro_f1",
            "best_epoch", "best_probe_dev_loss", "probe_train_nodes", "probe_dev_nodes", "fallback_class_count",
        ],
    )
    _write_csv(
        output_dir / "node_lambda_losses.csv",
        node_rows,
        ["dataset", "node_id", "modality", "lambda", "seed", "target", "ce_loss", "correct", "is_isolated", "degree"],
    )
    _write_csv(
        output_dir / "preferred_lambda_distribution.csv",
        distributions,
        ["dataset", "modality", "scope", "seed", "lambda", "count", "proportion", "eligible_nodes", "degree_bin", "eligible_nodes_in_degree_bin"],
    )
    _write_csv(
        output_dir / "context_demand_summary.csv",
        summary_rows,
        [
            "dataset", "modality", "global_best_lambda", "global_best_mean_val_ce", "normalized_preference_entropy",
            "non_global_preference_ratio", "global_ce_on_valid_val", "oracle_ce_on_valid_val", "oracle_absolute_gap",
            "oracle_relative_gap", "eligible_nodes", "isolated_val_nodes", "mean_val_ce_lambda_0.00", "mean_val_ce_lambda_0.25",
            "mean_val_ce_lambda_0.50", "mean_val_ce_lambda_0.75", "mean_val_ce_lambda_1.00",
        ],
    )
    _write_csv(
        output_dir / "modality_disagreement.csv",
        disagreement_rows,
        ["dataset", "scope", "seed", "disagreement_ratio", "mean_absolute_lambda_difference", "common_valid_nodes"],
    )
    _write_csv(
        output_dir / "stability.csv",
        stability_rows,
        ["dataset", "modality", "seed", "normalized_preference_entropy", "non_global_preference_ratio", "eligible_nodes"],
    )
    _write_csv(
        output_dir / "preferred_lambda_vs_degree.csv",
        degree_rows,
        ["dataset", "modality", "scope", "seed", "lambda", "count", "proportion", "eligible_nodes", "degree_bin", "eligible_nodes_in_degree_bin"],
    )
    (output_dir / "probe_split_fallbacks.json").write_text(json.dumps(fallback_records, indent=2), encoding="utf-8")
    (output_dir / "gate_a.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (output_dir / "m1_run_metadata.json").write_text(
        json.dumps(
            {
                "datasets": list(args.datasets),
                "seeds": list(SEEDS),
                "lambdas": list(LAMBDAS),
                "device": str(device),
                "probe": {
                    "epochs": int(args.epochs),
                    "patience": int(args.patience),
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay),
                    "min_delta": float(args.min_delta),
                },
                "test_labels_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = _make_report(output_dir, summary_rows, disagreement_rows, gate, fallback_records)

    try:
        from scripts.final.plot_context_demand import render

        render(output_dir)
    except Exception:
        logger.exception("Figure rendering failed; CSV/JSON/report outputs remain available")
        raise
    logger.info("Gate A=%s | supported=%s | enter_p1=%s", gate_a, supported_datasets, gate["enter_p1"])
    logger.info("Saved report=%s output_dir=%s", report_path, output_dir)
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", default="auto", help="Torch device, e.g. auto, cpu, cuda:0.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional override for paths.data_root.")
    parser.add_argument("--split-root", type=Path, default=None, help="Optional override for paths.split_root.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/context_demand"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
