"""Export analysis-only evidence for the current MoPF NC checkpoints.

This script never trains or changes a model.  It loads a best-validation-
accuracy checkpoint, calls MoPF's existing ``analysis_stats`` API, and writes
raw coefficients plus descriptive structural summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _resolve_nc_eval_labels


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _summary(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    values = _as_numpy(value).astype(np.float64, copy=False).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "population_std": None,
            "min": None,
            "max": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "mean_abs": None,
        }
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "population_std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "mean_abs": float(np.abs(values).mean()),
    }


def _summary_short(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    result = _summary(value)
    return {
        key: result[key]
        for key in ("n", "mean", "population_std", "q25", "median", "q75")
    }


def _correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    mask = mask & np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return None
    left_valid = left[mask].astype(np.float64, copy=False)
    right_valid = right[mask].astype(np.float64, copy=False)
    if np.std(left_valid) == 0.0 or np.std(right_valid) == 0.0:
        return None
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _split_names(data) -> np.ndarray:
    names = np.full(int(data.num_nodes), "none", dtype=object)
    for index, name in (
        (data.train_idx, "train"),
        (data.val_idx, "val"),
        (data.test_idx, "test"),
    ):
        if index is not None:
            names[_as_numpy(index).astype(np.int64)] = name
    return names


def _degree_and_homophily(data) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu().long()
    if edge_index.numel() == 0:
        return np.zeros(num_nodes, dtype=np.int64), np.full(num_nodes, np.nan)

    source, target = edge_index
    degree = torch.bincount(
        torch.cat([source, target]), minlength=num_nodes
    ).cpu().numpy().astype(np.int64, copy=False)

    labels = data.y.detach().cpu().long()
    valid_label = (labels >= 0) & (labels < int(data.num_classes))
    valid_edges = valid_label[source] & valid_label[target]
    source_valid = source[valid_edges]
    same_label = (labels[source_valid] == labels[target[valid_edges]]).long()
    neighbor_count = torch.bincount(source_valid, minlength=num_nodes)
    same_count = torch.bincount(
        source_valid, weights=same_label.float(), minlength=num_nodes
    )
    homophily = np.full(num_nodes, np.nan, dtype=np.float64)
    has_neighbors = neighbor_count > 0
    homophily[has_neighbors.numpy()] = (
        same_count[has_neighbors].numpy() / neighbor_count[has_neighbors].numpy()
    )
    return degree, homophily


def _rank_quartiles(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    if valid is None:
        valid = np.ones(values.shape[0], dtype=bool)
    groups = np.full(values.shape[0], -1, dtype=np.int8)
    valid_ids = np.flatnonzero(valid & np.isfinite(values))
    if valid_ids.size == 0:
        return groups
    order = valid_ids[np.argsort(values[valid_ids], kind="stable")]
    groups[order] = np.minimum(
        (np.arange(order.size, dtype=np.int64) * 4) // order.size, 3
    ).astype(np.int8)
    return groups


def _write_node_profiles(
    path: Path,
    data,
    stats: dict[str, torch.Tensor],
    degree: np.ndarray,
    homophily: np.ndarray,
) -> None:
    split = _split_names(data)
    labels = _as_numpy(data.y).astype(np.int64, copy=False)
    eta_text = _as_numpy(stats["eta_text"])
    eta_visual = _as_numpy(stats["eta_visual"])
    delta_text = _as_numpy(stats["delta_node_text"])
    delta_visual = _as_numpy(stats["delta_node_visual"])
    radius_text = _as_numpy(stats["effective_radius_text"])
    radius_visual = _as_numpy(stats["effective_radius_visual"])
    max_order = eta_text.shape[1] - 1

    fieldnames = [
        "node_id",
        "split",
        "label",
        "degree",
        "local_label_homophily",
    ]
    for name in (
        "delta_node_text",
        "delta_node_visual",
        "eta_text",
        "eta_visual",
    ):
        fieldnames.extend(f"{name}_{order}" for order in range(max_order + 1))
    fieldnames.extend(["effective_radius_text", "effective_radius_visual"])

    def csv_value(value: Any) -> Any:
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return ""
        if isinstance(value, np.generic):
            return value.item()
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for node_id in range(int(data.num_nodes)):
            row: list[Any] = [
                node_id,
                split[node_id],
                int(labels[node_id]),
                int(degree[node_id]),
                csv_value(float(homophily[node_id])),
            ]
            for values in (delta_text, delta_visual, eta_text, eta_visual):
                row.extend(float(value) for value in values[node_id])
            row.extend([float(radius_text[node_id]), float(radius_visual[node_id])])
            writer.writerow(row)


def _write_distribution_summary(
    path: Path,
    dataset: str,
    seed: int,
    stats: dict[str, torch.Tensor],
) -> None:
    rows: list[dict[str, Any]] = []

    def add_vector(variable: str, modality: str, values: torch.Tensor) -> None:
        for order in range(int(values.numel())):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variable": variable,
                    "modality": modality,
                    "order": order,
                    **_summary(values[order]),
                }
            )

    def add_profile(variable: str, modality: str, values: torch.Tensor) -> None:
        for order in range(int(values.size(1))):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "variable": variable,
                    "modality": modality,
                    "order": order,
                    **_summary(values[:, order]),
                }
            )

    add_vector("gamma_global", "global", stats["gamma_global"])
    add_vector("delta_gamma", "text", stats["delta_gamma_text"])
    add_vector("delta_gamma", "visual", stats["delta_gamma_visual"])
    add_vector("gamma_modality", "text", stats["gamma_text"])
    add_vector("gamma_modality", "visual", stats["gamma_visual"])
    add_profile("delta_node", "text", stats["delta_node_text"])
    add_profile("delta_node", "visual", stats["delta_node_visual"])
    add_profile("eta", "text", stats["eta_text"])
    add_profile("eta", "visual", stats["eta_visual"])
    for modality in ("text", "visual"):
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "variable": "effective_radius",
                "modality": modality,
                "order": "all",
                **_summary(stats[f"effective_radius_{modality}"]),
            }
        )

    fieldnames = [
        "dataset",
        "seed",
        "variable",
        "modality",
        "order",
        "n",
        "mean",
        "population_std",
        "min",
        "max",
        "q10",
        "q25",
        "median",
        "q75",
        "q90",
        "mean_abs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_modality_summary(
    path: Path,
    dataset: str,
    seed: int,
    stats: dict[str, torch.Tensor],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    max_order = int(stats["eta_text"].size(1) - 1)
    for modality in ("text", "visual"):
        delta_gamma = stats[f"delta_gamma_{modality}"]
        delta_node = stats[f"delta_node_{modality}"]
        eta = stats[f"eta_{modality}"]
        for order in range(max_order + 1):
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "row_type": "coefficient",
                    "modality": modality,
                    "order": order,
                    "gamma_global": float(stats["gamma_global"][order]),
                    "delta_gamma": float(delta_gamma[order]),
                    "gamma_modality": float(stats[f"gamma_{modality}"][order]),
                    "delta_node_mean": float(delta_node[:, order].mean()),
                    "delta_node_std": float(delta_node[:, order].std(unbiased=False)),
                    "eta_mean": float(eta[:, order].mean()),
                    "eta_std": float(eta[:, order].std(unbiased=False)),
                }
            )

    profile_distance = torch.linalg.vector_norm(
        stats["eta_text"] - stats["eta_visual"], dim=1
    )
    distance_summary = _summary_short(profile_distance)
    rows.append(
        {
            "dataset": dataset,
            "seed": seed,
            "row_type": "profile_distance_l2",
            "modality": "text_visual",
            "order": "all",
            **{key: "" for key in (
                "gamma_global",
                "delta_gamma",
                "gamma_modality",
                "delta_node_mean",
                "delta_node_std",
                "eta_mean",
                "eta_std",
            )},
            **distance_summary,
        }
    )
    radius_summaries: dict[str, Any] = {}
    for modality in ("text", "visual"):
        short = _summary_short(stats[f"effective_radius_{modality}"])
        radius_summaries[modality] = short
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "row_type": "effective_radius",
                "modality": modality,
                "order": "all",
                **{key: "" for key in (
                    "gamma_global",
                    "delta_gamma",
                    "gamma_modality",
                    "delta_node_mean",
                    "delta_node_std",
                    "eta_mean",
                    "eta_std",
                )},
                **short,
            }
        )

    fieldnames = [
        "dataset",
        "seed",
        "row_type",
        "modality",
        "order",
        "gamma_global",
        "delta_gamma",
        "gamma_modality",
        "delta_node_mean",
        "delta_node_std",
        "eta_mean",
        "eta_std",
        "n",
        "mean",
        "population_std",
        "q25",
        "median",
        "q75",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "profile_distance_l2": distance_summary,
        "effective_radius": radius_summaries,
    }


def _write_structure_summary(
    path: Path,
    dataset: str,
    seed: int,
    stats: dict[str, torch.Tensor],
    degree: np.ndarray,
    homophily: np.ndarray,
) -> dict[str, Any]:
    degree_groups = _rank_quartiles(degree.astype(np.float64))
    homophily_groups = _rank_quartiles(homophily)
    rows: list[dict[str, Any]] = []
    max_order = int(stats["eta_text"].size(1) - 1)

    for group_type, groups, group_names in (
        ("degree_quartile", degree_groups, ("q1", "q2", "q3", "q4")),
        ("homophily_quartile", homophily_groups, ("q1", "q2", "q3", "q4")),
    ):
        for group_id, group_name in enumerate(group_names):
            mask = groups == group_id
            for modality in ("text", "visual"):
                eta = _as_numpy(stats[f"eta_{modality}"])
                delta_node = _as_numpy(stats[f"delta_node_{modality}"])
                for order in range(max_order + 1):
                    eta_values = eta[mask, order]
                    delta_values = delta_node[mask, order]
                    if group_type == "degree_quartile":
                        degree_mean = float(degree[mask].mean()) if mask.any() else None
                        homophily_mean = (
                            float(np.nanmean(homophily[mask])) if mask.any() else None
                        )
                    else:
                        degree_mean = float(degree[mask].mean()) if mask.any() else None
                        homophily_mean = (
                            float(np.nanmean(homophily[mask])) if mask.any() else None
                        )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "group_type": group_type,
                            "group": group_name,
                            "n": int(mask.sum()),
                            "degree_mean": degree_mean,
                            "homophily_mean": homophily_mean,
                            "modality": modality,
                            "order": order,
                            "eta_mean": float(eta_values.mean()) if eta_values.size else None,
                            "eta_std": float(eta_values.std(ddof=0)) if eta_values.size else None,
                            "delta_node_mean": float(delta_values.mean()) if delta_values.size else None,
                            "delta_node_std": float(delta_values.std(ddof=0)) if delta_values.size else None,
                        }
                    )

    fieldnames = [
        "dataset",
        "seed",
        "group_type",
        "group",
        "n",
        "degree_mean",
        "homophily_mean",
        "modality",
        "order",
        "eta_mean",
        "eta_std",
        "delta_node_mean",
        "delta_node_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    correlations: dict[str, Any] = {}
    for modality in ("text", "visual"):
        eta = _as_numpy(stats[f"eta_{modality}"])
        correlations[modality] = {
            "degree_vs_eta": [
                _correlation(eta[:, order], degree.astype(np.float64), np.isfinite(degree))
                for order in range(max_order + 1)
            ],
            "homophily_vs_eta": [
                _correlation(eta[:, order], homophily, np.isfinite(homophily))
                for order in range(max_order + 1)
            ],
        }
    return {
        "degree_quartile_counts": [int((degree_groups == group).sum()) for group in range(4)],
        "homophily_quartile_counts": [int((homophily_groups == group).sum()) for group in range(4)],
        "correlations": correlations,
    }


def _load_checkpoint_bundle(output_dir: Path, device: torch.device):
    checkpoint = output_dir / "best_val_accuracy.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing best-validation checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
    data = load_mag_data(cfg, "nc", int(payload["seed"]))
    model = build_model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return payload, cfg, data, model


def export_audit(dataset: str, output_dir: Path, device_name: str) -> None:
    device = torch.device(device_name)
    payload, cfg, data, model = _load_checkpoint_bundle(output_dir, device)
    seed = int(payload["seed"])
    eval_labels = _resolve_nc_eval_labels(data)
    with torch.no_grad():
        stats = model.analysis_stats(data.x.to(device), data.edge_index.to(device))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stats["gamma_text"] = stats["gamma_global"] + stats["delta_gamma_text"]
    stats["gamma_visual"] = stats["gamma_global"] + stats["delta_gamma_visual"]
    coefficients = {key: value.detach().cpu() for key, value in stats.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(coefficients, output_dir / "coefficients.pt")

    degree, homophily = _degree_and_homophily(data)
    _write_distribution_summary(output_dir / "distribution_summary.csv", dataset, seed, coefficients)
    _write_node_profiles(output_dir / "node_profiles.csv", data, coefficients, degree, homophily)
    modality_summary = _write_modality_summary(
        output_dir / "modality_summary.csv", dataset, seed, coefficients
    )
    structure_summary = _write_structure_summary(
        output_dir / "structure_profile_summary.csv",
        dataset,
        seed,
        coefficients,
        degree,
        homophily,
    )

    max_order = int(coefficients["eta_text"].size(1) - 1)
    high_order = {
        "order": max_order,
        "gamma_global": float(coefficients["gamma_global"][max_order]),
        "delta_gamma_text": float(coefficients["delta_gamma_text"][max_order]),
        "delta_gamma_visual": float(coefficients["delta_gamma_visual"][max_order]),
        "delta_node_text": _summary(coefficients["delta_node_text"][:, max_order]),
        "delta_node_visual": _summary(coefficients["delta_node_visual"][:, max_order]),
        "eta_text": _summary(coefficients["eta_text"][:, max_order]),
        "eta_visual": _summary(coefficients["eta_visual"][:, max_order]),
    }
    summary = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint": str(output_dir / "best_val_accuracy.pt"),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metrics": payload.get("metrics", {}),
        "max_order": max_order,
        "num_layers": int(cfg.model.num_layers),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_classes": int(data.num_classes),
        "valid_eval_labels": eval_labels,
        "gamma_global": coefficients["gamma_global"].tolist(),
        "delta_gamma_text": coefficients["delta_gamma_text"].tolist(),
        "delta_gamma_visual": coefficients["delta_gamma_visual"].tolist(),
        "gamma_text": coefficients["gamma_text"].tolist(),
        "gamma_visual": coefficients["gamma_visual"].tolist(),
        "node_residual": {
            "text": [_summary(coefficients["delta_node_text"][:, order]) for order in range(max_order + 1)],
            "visual": [_summary(coefficients["delta_node_visual"][:, order]) for order in range(max_order + 1)],
        },
        "eta": {
            "text": [_summary(coefficients["eta_text"][:, order]) for order in range(max_order + 1)],
            "visual": [_summary(coefficients["eta_visual"][:, order]) for order in range(max_order + 1)],
        },
        "modality_summary": modality_summary,
        "structure_summary": structure_summary,
        "high_order": high_order,
        "effective_radius": {
            modality: _summary_short(coefficients[f"effective_radius_{modality}"])
            for modality in ("text", "visual")
        },
        "analysis_notes": {
            "effective_radius": "sum_k k*abs(eta_i,k) / sum_k abs(eta_i,k)",
            "grouping": "degree and finite local-label-homophily rank quartiles",
            "standard_deviation": "population standard deviation (ddof=0)",
            "training_unchanged": True,
        },
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[mechanism-audit] {dataset} seed={seed} K={max_order} "
        f"labels={eval_labels} output={output_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "mechanism_audit",
    )
    args = parser.parse_args()
    output_dir = args.output_root / args.dataset / "seed42"
    export_audit(args.dataset, output_dir, args.device)


if __name__ == "__main__":
    main()
