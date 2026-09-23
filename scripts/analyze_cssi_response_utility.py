#!/usr/bin/env python3
"""Frozen-forward Structural Response analysis for CSSI P1.

The script consumes best-validation NC checkpoints produced by
``scripts/run_cssi_p1.py``.  It never fits a response-specific classifier and
never uses test labels for a decision.  Node-level exports are compressed CSV
files; the compact ``summary.csv`` is the machine-readable analysis product.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.data import load_mag_data
from src.models.cssi_response_probe import CSSIResponseProbe, RESPONSE_COEFFICIENTS


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
ORDERS = (1, 2, 3)
DEFAULT_EPSILON = 1.0e-6
# GPU scatter-reduce order is non-associative in float32; the algebraic
# identities remain exact up to this small runtime tolerance.
DEFAULT_TOLERANCE = 2.0e-5


def _as_float(value: Any) -> float:
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return float("nan")
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson(left_rank, right_rank)


def _margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_score = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels.unsqueeze(1), float("-inf"))
    return true_score - other.max(dim=1).values


def _degree_groups(degrees: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """Assign deterministic low/medium/high groups by within-split rank."""
    count = int(node_ids.size)
    if count == 0:
        return np.asarray([], dtype=object)
    order = np.argsort(degrees[node_ids], kind="stable")
    groups = np.empty(count, dtype=object)
    cut_one = count // 3
    cut_two = (2 * count) // 3
    groups[order[:cut_one]] = "low"
    groups[order[cut_one:cut_two]] = "medium"
    groups[order[cut_two:]] = "high"
    return groups


def _distribution(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {f"{prefix}_N": 0}
    return {
        f"{prefix}_N": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q90": float(np.quantile(values, 0.90)),
    }


def _summary_for_group(
    frame: pd.DataFrame,
    *,
    dataset: str,
    seed: int | None,
    split: str,
    modality: str,
    order: int,
    degree_group: str,
    aggregation: str,
    epsilon: float,
) -> dict[str, Any]:
    subset = frame[
        (frame["dataset"] == dataset)
        & (frame["split"] == split)
        & (frame["modality"] == modality)
        & (frame["order"] == order)
        & (frame["degree_group"] == degree_group)
    ]
    if seed is not None:
        subset = subset[subset["seed"] == seed]
    ce = subset["utility_ce"].to_numpy(dtype=np.float64)
    margin = subset["utility_margin"].to_numpy(dtype=np.float64)
    response_norm = subset["response_norm"].to_numpy(dtype=np.float64)
    normalized = subset["normalized_response_norm"].to_numpy(dtype=np.float64)
    row: dict[str, Any] = {
        "aggregation": aggregation,
        "dataset": dataset,
        "seed": seed if seed is not None else "aggregate",
        "split": split,
        "modality": modality,
        "order": order,
        "degree_group": degree_group,
        "epsilon": epsilon,
    }
    row.update(_distribution(ce, "utility_ce"))
    row.update(_distribution(margin, "utility_margin"))
    row.update(_distribution(response_norm, "response_norm"))
    row.update(_distribution(normalized, "normalized_response_norm"))
    if ce.size:
        row.update(
            {
                "p_utility_ce_gt0": float(np.mean(ce > 0.0)),
                "p_utility_ce_lt0": float(np.mean(ce < 0.0)),
                "p_utility_ce_abs_le_eps": float(np.mean(np.abs(ce) <= epsilon)),
                "p_utility_margin_gt0": float(np.mean(margin > 0.0)),
                "p_utility_margin_lt0": float(np.mean(margin < 0.0)),
                "p_utility_margin_abs_le_eps": float(
                    np.mean(np.abs(margin) <= epsilon)
                ),
                "pearson_response_norm_utility_ce": _pearson(
                    response_norm, ce
                ),
                "spearman_response_norm_utility_ce": _spearman(
                    response_norm, ce
                ),
                "pearson_response_norm_utility_margin": _pearson(
                    response_norm, margin
                ),
                "spearman_response_norm_utility_margin": _spearman(
                    response_norm, margin
                ),
                "ce_margin_sign_agreement": float(
                    np.mean(np.sign(ce) == np.sign(margin))
                ),
            }
        )
    else:
        for key in (
            "p_utility_ce_gt0",
            "p_utility_ce_lt0",
            "p_utility_ce_abs_le_eps",
            "p_utility_margin_gt0",
            "p_utility_margin_lt0",
            "p_utility_margin_abs_le_eps",
            "pearson_response_norm_utility_ce",
            "spearman_response_norm_utility_ce",
            "pearson_response_norm_utility_margin",
            "spearman_response_norm_utility_margin",
            "ce_margin_sign_agreement",
        ):
            row[key] = float("nan")
    return row


def _build_model_and_data(
    project_root: Path,
    run_dir: Path,
    checkpoint_path: Path,
    dataset: str,
    seed: int,
    device: torch.device,
) -> tuple[CSSIResponseProbe, torch.nn.Module, Any, Any]:
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing resolved config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    cfg.seed = seed
    cfg.dataset = OmegaConf.load(project_root / "configs" / "dataset" / f"{dataset}.yaml")
    cfg.model.name = "cssi_response_probe"
    data = load_mag_data(cfg, "nc", seed)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data_info = dict(payload["data_info"])
    model = CSSIResponseProbe(cfg, data_info).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()
    return model, classifier, data, payload


def _run_one(
    project_root: Path,
    experiment_root: Path,
    output_root: Path,
    dataset: str,
    seed: int,
    device: torch.device,
    splits: Iterable[str],
    epsilon: float,
    tolerance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = experiment_root / "runs" / "plain" / dataset / f"seed_{seed}"
    checkpoint_path = experiment_root / "checkpoints" / "plain" / dataset / f"seed_{seed}.pt"
    model, classifier, data, payload = _build_model_and_data(
        project_root, run_dir, checkpoint_path, dataset, seed, device
    )
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    with torch.no_grad():
        components = model.response_components(
            x, edge_index, check_equivalences=True, tolerance=tolerance
        )
        base_logits = model.leave_one_response_out_logits(classifier, components)
        ordinary_logits = classifier(components["z"])
    base_case_error = float((base_logits - ordinary_logits).abs().max().item())
    if base_case_error > tolerance:
        raise AssertionError(
            f"leave-one-response-out base case failed for {dataset}/{seed}: "
            f"{base_case_error}"
        )

    labels = data.y.to(device)
    degrees = torch.bincount(
        data.edge_index.reshape(-1), minlength=data.num_nodes
    ).cpu().numpy()
    checks = components["equivalence_checks"]
    check_row = {
        "dataset": dataset,
        "seed": seed,
        "operator_index_equal": checks["operator_index_equal"],
        "operator_weight_max_abs_error": checks["operator_weight_max_abs_error"],
        "text_recurrence_max_abs_error": checks["text"]["recurrence_max_abs_error"],
        "visual_recurrence_max_abs_error": checks["visual"]["recurrence_max_abs_error"],
        "text_response_reconstruction_max_abs_error": checks["text"][
            "response_reconstruction_max_abs_error"
        ],
        "visual_response_reconstruction_max_abs_error": checks["visual"][
            "response_reconstruction_max_abs_error"
        ],
        "text_operator_response_max_abs_error": checks["text"][
            "operator_response_max_abs_error"
        ],
        "visual_operator_response_max_abs_error": checks["visual"][
            "operator_response_max_abs_error"
        ],
        "text_plain_representation_max_abs_error": checks["text"][
            "plain_representation_max_abs_error"
        ],
        "visual_plain_representation_max_abs_error": checks["visual"][
            "plain_representation_max_abs_error"
        ],
        "leave_one_response_out_base_max_abs_error": base_case_error,
        "tolerance": tolerance,
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_selection": payload.get("selection"),
    }

    frames: list[pd.DataFrame] = []
    split_indices = {
        "train": data.train_idx,
        "validation": data.val_idx,
        "test": data.test_idx,
    }
    node_output_dir = output_root / "node_level" / dataset
    node_output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if split not in split_indices or split_indices[split] is None:
            raise ValueError(f"dataset {dataset} has no split {split}")
        node_ids = split_indices[split].cpu().numpy().astype(np.int64, copy=False)
        split_groups = _degree_groups(degrees, node_ids)
        split_rows: list[dict[str, Any]] = []
        valid_mask = (labels[node_ids] >= 0) & (labels[node_ids] < int(data.num_classes))
        valid_node_ids = node_ids[valid_mask.cpu().numpy()]
        if valid_node_ids.size == 0:
            continue
        valid_labels = labels[valid_node_ids]
        with torch.no_grad():
            base_split_logits = base_logits[valid_node_ids]
            base_loss = F.cross_entropy(
                base_split_logits, valid_labels, reduction="none"
            )
            base_margin = _margin(base_split_logits, valid_labels)

        node_position = {int(node_id): pos for pos, node_id in enumerate(node_ids)}
        for modality in MODALITIES:
            states = components[f"states_{modality}"]
            responses = components[f"responses_{modality}"]
            h0 = components[f"h0_{modality}"]
            for order, (coefficient, response) in enumerate(
                zip(RESPONSE_COEFFICIENTS, responses), start=1
            ):
                with torch.no_grad():
                    counterfactual_logits = model.leave_one_response_out_logits(
                        classifier,
                        components,
                        modality=modality,
                        order=order,
                    )
                    cf_split_logits = counterfactual_logits[valid_node_ids]
                    cf_loss = F.cross_entropy(
                        cf_split_logits, valid_labels, reduction="none"
                    )
                    cf_margin = _margin(cf_split_logits, valid_labels)
                    # Positive means that removing the response increases CE
                    # loss, i.e. the response was helpful.
                    utility_ce = cf_loss - base_loss
                    utility_margin = base_margin - cf_margin
                    previous = states[order - 1][valid_node_ids]
                    response_selected = response[valid_node_ids]
                    h0_selected = h0[valid_node_ids]
                    response_norm = response_selected.norm(dim=-1)
                    previous_norm = previous.norm(dim=-1)
                    normalized_norm = response_norm / (previous_norm + epsilon)
                    cosine_h0 = F.cosine_similarity(
                        response_selected, h0_selected, dim=-1, eps=epsilon
                    )
                    cosine_previous = F.cosine_similarity(
                        response_selected, previous, dim=-1, eps=epsilon
                    )
                    cosine_previous_h0 = F.cosine_similarity(
                        previous, h0_selected, dim=-1, eps=epsilon
                    )
                for row_index, node_id in enumerate(valid_node_ids.tolist()):
                    position = node_position[int(node_id)]
                    split_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "split": split,
                            "node_id": int(node_id),
                            "label": int(valid_labels[row_index].item()),
                            "modality": modality,
                            "order": order,
                            "response_norm": _as_float(response_norm[row_index]),
                            "previous_state_norm": _as_float(previous_norm[row_index]),
                            "normalized_response_norm": _as_float(
                                normalized_norm[row_index]
                            ),
                            "cosine_response_vs_h0": _as_float(cosine_h0[row_index]),
                            "cosine_response_vs_previous": _as_float(
                                cosine_previous[row_index]
                            ),
                            "cosine_previous_vs_h0": _as_float(
                                cosine_previous_h0[row_index]
                            ),
                            "utility_ce": _as_float(utility_ce[row_index]),
                            "utility_margin": _as_float(utility_margin[row_index]),
                            "node_degree": int(degrees[node_id]),
                            "degree_group": str(split_groups[position]),
                        }
                    )
        frame = pd.DataFrame(split_rows)
        output_path = node_output_dir / f"seed_{seed}_{split}.csv.gz"
        frame.to_csv(output_path, index=False, compression="gzip", float_format="%.9g")
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"no valid analysis nodes for {dataset}/{seed}")
    return pd.concat(frames, ignore_index=True), check_row


def _load_backbone_sanity(experiment_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in ("no_graph", "last_hop", "plain"):
        for dataset in DATASETS:
            for seed in SEEDS:
                metrics_path = (
                    experiment_root / "runs" / variant / dataset / f"seed_{seed}" / "metrics.json"
                )
                if not metrics_path.is_file():
                    continue
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics = payload.get("metrics", {})
                row = {
                    "variant": variant,
                    "dataset": dataset,
                    "seed": seed,
                    "best_epoch": payload.get("best_epoch"),
                    "selection": payload.get("checkpoint_selection"),
                    "val_acc": metrics.get("val_acc", {}).get("mean"),
                    "val_macro_f1": metrics.get("val_macro_f1", {}).get("mean"),
                    "test_evaluated": "test_acc" in metrics,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _p1_diagnostics(
    all_nodes: pd.DataFrame, summary: pd.DataFrame, epsilon: float
) -> dict[str, pd.DataFrame]:
    """Build corrected P1 diagnostics without pooling across unrelated cells."""
    validation = summary[
        (summary["split"] == "validation")
        & (summary["degree_group"] == "all")
        & (summary["aggregation"] == "seed_aggregate")
    ].copy()
    mean_dispersion = validation[
        [
            "dataset",
            "modality",
            "order",
            "utility_ce_mean",
            "utility_ce_std",
        ]
    ].copy()
    mean_dispersion["mean_abs_over_std"] = mean_dispersion["utility_ce_mean"].abs() / (
        mean_dispersion["utility_ce_std"] + epsilon
    )

    seed_rows = summary[
        (summary["split"] == "validation")
        & (summary["degree_group"] == "all")
        & (summary["aggregation"] == "seed")
    ].copy()
    stability_rows: list[dict[str, Any]] = []
    for (dataset, modality, order), group in seed_rows.groupby(
        ["dataset", "modality", "order"]
    ):
        aggregate = validation[
            (validation["dataset"] == dataset)
            & (validation["modality"] == modality)
            & (validation["order"] == order)
        ].iloc[0]
        seed_means = group["utility_ce_mean"].to_numpy(dtype=np.float64)
        stability_rows.append(
            {
                "dataset": dataset,
                "modality": modality,
                "order": order,
                "n_seeds": len(seed_means),
                "aggregate_utility_mean": float(aggregate["utility_ce_mean"]),
                "aggregate_utility_std": float(aggregate["utility_ce_std"]),
                "effect_size_abs_mean_over_std": float(
                    abs(aggregate["utility_ce_mean"])
                    / (float(aggregate["utility_ce_std"]) + epsilon)
                ),
                "same_sign_across_seeds": bool(
                    np.all(seed_means > 0.0) or np.all(seed_means < 0.0)
                ),
                "seed_mean_min": float(seed_means.min()),
                "seed_mean_max": float(seed_means.max()),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability_summary_rows: list[dict[str, Any]] = []
    for threshold in (0.0, 0.02, 0.05, 0.10):
        selected = stability[stability["effect_size_abs_mean_over_std"] >= threshold]
        stability_summary_rows.append(
            {
                "effect_threshold": threshold,
                "n_cells": len(selected),
                "n_same_sign": int(selected["same_sign_across_seeds"].sum()),
                "same_sign_fraction": float(
                    selected["same_sign_across_seeds"].mean()
                )
                if len(selected)
                else float("nan"),
            }
        )
    stability_summary = pd.DataFrame(stability_summary_rows)

    response_association = validation[
        [
            "dataset",
            "modality",
            "order",
            "response_norm_mean",
            "utility_ce_mean",
            "spearman_response_norm_utility_ce",
        ]
    ].copy()
    response_association["association_type"] = "within_cell_node_level"
    cross_cell = pd.DataFrame(
        [
            {
                "association_type": "across_30_aggregate_cells",
                "dataset": "all",
                "modality": "all",
                "order": "all",
                "response_norm_mean": float(validation["response_norm_mean"].mean()),
                "utility_ce_mean": float(validation["utility_ce_mean"].abs().mean()),
                "spearman_response_norm_utility_ce": _spearman(
                    validation["response_norm_mean"].to_numpy(dtype=np.float64),
                    validation["utility_ce_mean"].abs().to_numpy(dtype=np.float64),
                ),
            }
        ]
    )
    response_association = pd.concat(
        [response_association, cross_cell], ignore_index=True
    )

    modality_pivot = validation.pivot_table(
        index=["dataset", "order"], columns="modality", values="utility_ce_mean"
    ).reset_index()
    modality_difference = modality_pivot.rename(
        columns={"text": "text_utility_mean", "visual": "visual_utility_mean"}
    )
    modality_difference["absolute_difference"] = (
        modality_difference["text_utility_mean"]
        - modality_difference["visual_utility_mean"]
    ).abs()
    modality_order_summary = (
        modality_difference.groupby("order", as_index=False)["absolute_difference"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(
            columns={
                "mean": "absolute_difference_mean",
                "median": "absolute_difference_median",
            }
        )
    )
    modality_order_summary["dataset"] = "order_summary"
    modality_order_summary["text_utility_mean"] = np.nan
    modality_order_summary["visual_utility_mean"] = np.nan
    modality_order_summary["absolute_difference"] = np.nan
    modality_order_summary = modality_order_summary[
        [
            "dataset",
            "order",
            "text_utility_mean",
            "visual_utility_mean",
            "absolute_difference",
            "absolute_difference_mean",
            "absolute_difference_median",
        ]
    ]
    modality_difference["absolute_difference_mean"] = np.nan
    modality_difference["absolute_difference_median"] = np.nan
    modality_difference = pd.concat(
        [modality_difference, modality_order_summary], ignore_index=True
    )

    saturation = validation[
        [
            "dataset",
            "modality",
            "order",
            "response_norm_mean",
            "normalized_response_norm_mean",
            "normalized_response_norm_median",
            "utility_ce_mean",
            "utility_ce_std",
        ]
    ].copy()
    saturation["near_degenerate_threshold"] = 1.0e-3
    saturation["near_degenerate"] = (
        saturation["normalized_response_norm_median"] <= 1.0e-3
    )

    return {
        "mean_dispersion": mean_dispersion,
        "seed_stability": stability,
        "seed_stability_summary": stability_summary,
        "response_association": response_association,
        "modality_difference": modality_difference,
        "saturation": saturation,
    }


def _write_report(
    report_path: Path,
    summary: pd.DataFrame,
    sanity: pd.DataFrame,
    checks: pd.DataFrame,
    diagnostics: dict[str, pd.DataFrame],
    output_root: Path,
    epsilon: float,
    splits: tuple[str, ...],
) -> None:
    validation = summary[
        (summary["split"] == "validation")
        & (summary["degree_group"] == "all")
        & (summary["aggregation"] == "seed_aggregate")
    ]
    if validation.empty:
        raise RuntimeError("validation aggregate summary is empty")
    nondegenerate = validation[
        (validation["p_utility_ce_gt0"] > 0.0)
        & (validation["p_utility_ce_lt0"] > 0.0)
    ]
    conditional = validation[validation["utility_ce_std"] > 0.0]
    magnitude = validation["spearman_response_norm_utility_ce"].dropna()
    agreement = validation["ce_margin_sign_agreement"].dropna()
    modality_piv = validation.pivot_table(
        index=["dataset", "order"], columns="modality", values="utility_ce_mean"
    )
    modality_diffs = (
        (modality_piv["text"] - modality_piv["visual"]).abs()
        if {"text", "visual"}.issubset(modality_piv.columns)
        else pd.Series(dtype=float)
    )
    order_piv = validation.pivot_table(
        index=["dataset", "modality"], columns="order", values="utility_ce_mean"
    )
    order_spread = (
        order_piv.max(axis=1) - order_piv.min(axis=1)
        if not order_piv.empty
        else pd.Series(dtype=float)
    )
    seed_rows = summary[
        (summary["split"] == "validation")
        & (summary["degree_group"] == "all")
        & (summary["aggregation"] == "seed")
    ]
    stable_sign_cells = 0
    total_seed_cells = 0
    for _, group in seed_rows.groupby(["dataset", "modality", "order"]):
        if len(group) != len(SEEDS):
            continue
        total_seed_cells += 1
        if (group["utility_ce_mean"] > 0).all() or (group["utility_ce_mean"] < 0).all():
            stable_sign_cells += 1
    check_cols = [
        "operator_weight_max_abs_error",
        "text_recurrence_max_abs_error",
        "visual_recurrence_max_abs_error",
        "text_response_reconstruction_max_abs_error",
        "visual_response_reconstruction_max_abs_error",
        "text_operator_response_max_abs_error",
        "visual_operator_response_max_abs_error",
        "text_plain_representation_max_abs_error",
        "visual_plain_representation_max_abs_error",
        "leave_one_response_out_base_max_abs_error",
    ]
    max_errors = {
        col: float(checks[col].max()) for col in check_cols if col in checks
    }
    mean_dispersion = diagnostics["mean_dispersion"]
    stability_summary = diagnostics["seed_stability_summary"]
    response_association = diagnostics["response_association"]
    within_association = response_association[
        response_association["association_type"] == "within_cell_node_level"
    ]["spearman_response_norm_utility_ce"].dropna()
    cross_association = response_association[
        response_association["association_type"] == "across_30_aggregate_cells"
    ]["spearman_response_norm_utility_ce"].iloc[0]
    modality_difference = diagnostics["modality_difference"]
    modality_order_summary = modality_difference[
        modality_difference["dataset"] == "order_summary"
    ]
    saturation = diagnostics["saturation"]
    saturated_cells = saturation[saturation["near_degenerate"]]

    lines = [
        "# CSSI P1 Structural Response Report",
        "",
        "## 1. Code / protocol audit",
        "",
        "This report uses the canonical `all_plain` backbone and the frozen NC "
        "protocol `unified_full_graph_nc_v1`. Only validation nodes are used for "
        "architecture and hypothesis decisions; test evaluation is disabled in "
        "the P1 launcher.",
        "",
        f"The near-zero utility band is `|u| <= {epsilon:.1e}`. Standard deviations "
        "are population standard deviations (`ddof=0`). Degree groups are "
        "within-validation rank terciles.",
        "",
        "## 2. Exact definition of `all_plain`",
        "",
        "The implementation and call-chain audit is recorded in "
        "`docs/cssi_all_plain_audit.md`. The response probe subclasses "
        "`CoSIMAGAblation` without adding parameters or changing its forward path.",
        "",
        "## 3. Mathematical equivalence checks",
        "",
        f"All `{len(checks)}` dataset/seed runs passed the float32 tolerance "
        f"`{DEFAULT_TOLERANCE:.1e}`. Maximum observed errors: "
        + "; ".join(f"`{key}`={value:.3e}" for key, value in max_errors.items())
        + ".",
        "",
        "The checks cover text/visual normalized-operator equality, "
        "the direct `R_k=(P-I)S_{k-1}` residual, response reconstruction, plain-representation "
        "equivalence, and the leave-one-response-out base case.",
        "",
        *(
            [
                "The largest recurrence residual is slightly above `1e-5` only "
                "because GPU scatter accumulation is non-associative in float32; "
                "the maximum remains below the predeclared `2e-5` runtime tolerance."
            ]
            if max(
                [
                    max_errors.get("text_recurrence_max_abs_error", 0.0),
                    max_errors.get("visual_recurrence_max_abs_error", 0.0),
                ]
            )
            > 1.0e-5
            else []
        ),
        "",
        "## 4. Backbone sanity results",
        "",
        "| Variant | Dataset | validation accuracy mean | validation Macro-F1 mean |",
        "|---|---|---:|---:|",
    ]
    if not sanity.empty:
        sanity_agg = (
            sanity.groupby(["variant", "dataset"], as_index=False)[
                ["val_acc", "val_macro_f1"]
            ]
            .mean(numeric_only=True)
            .sort_values(["variant", "dataset"])
        )
        for _, row in sanity_agg.iterrows():
            lines.append(
                f"| {row['variant']} | {row['dataset']} | "
                f"{row['val_acc']:.4f} | {row['val_macro_f1']:.4f} |"
            )
    lines += [
        "",
        "No test result is used in the following interpretation.",
        "",
        "## 5. Structural Response and utility definition",
        "",
        "For each frozen best-validation checkpoint, `R_k=S_k-S_{k-1}`. "
        "The plain representation is `H0 + 0.75 R1 + 0.50 R2 + 0.25 R3`. "
        "Each counterfactual removes one response from one modality, keeps the "
        "other modality unchanged, then reruns modality refinement, late fusion, "
        "and the original classifier. The reported quantity is frozen-forward "
        "functional utility: `u_ce = loss_counterfactual - loss_base` and "
        "`u_margin = margin_base - margin_counterfactual`; positive means helpful.",
        "",
        "## 6. Per-dataset / modality / order statistics",
        "",
        "Detailed machine-readable values are in the selected output directory's "
        "`summary.csv`; "
        f"node-level exports are compressed under `{output_root}/node_level/`. "
        "The principal aggregate columns are "
        "`utility_ce_mean`, `utility_ce_std`, `p_utility_ce_gt0`, "
        "`p_utility_ce_lt0`, `response_norm_mean`, and the Pearson/Spearman "
        "response-norm correlations.",
        "",
        "| Dataset | Modality | Order | N | CE mean | CE std | P(CE>0) | P(CE<0) | Spearman(norm, CE) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in validation.sort_values(["dataset", "modality", "order"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['modality']} | {int(row['order'])} | "
            f"{int(row['utility_ce_N'])} | {row['utility_ce_mean']:.5g} | "
            f"{row['utility_ce_std']:.5g} | {row['p_utility_ce_gt0']:.3f} | "
            f"{row['p_utility_ce_lt0']:.3f} | "
            f"{row['spearman_response_norm_utility_ce']:.4f} |"
        )
    lines += [
        "",
        "## 7. Cross-seed stability",
        "",
        f"There are {total_seed_cells} dataset × modality × order cells with all "
        f"three seeds. Their seed-level utility means have the same sign in "
        f"{stable_sign_cells} cells. This is a descriptive stability count, not "
        "a success threshold.",
        "",
        "## 8. Degree-group analysis",
        "",
        "The summary contains low/medium/high within-validation degree-tercile "
        "rows for every dataset, seed, modality, and order. Differences should "
        "be read from the corresponding `degree_group` rows rather than inferred "
        "from the all-node aggregate.",
        "",
        "## 9. Response magnitude vs utility",
        "",
        f"Across the {len(magnitude)} aggregate cells with defined Spearman "
        f"correlation, the mean correlation is "
        f"`{magnitude.mean():.4f}` and the median is `{magnitude.median():.4f}`. "
        "This reports association only; it does not fit an adaptation rule.",
        "",
        "## 10. CE vs margin consistency",
        "",
        f"CE/margin sign agreement across aggregate cells has mean "
        f"`{agreement.mean():.4f}` and median `{agreement.median():.4f}`. "
        "Disagreements are retained in the raw and summary outputs.",
        "",
        "## 11. Mean versus dispersion",
        "",
        f"Across the 30 cells, the mean and median of `|mean(u)|/(std(u)+epsilon)` "
        f"are `{mean_dispersion['mean_abs_over_std'].mean():.4f}` and "
        f"`{mean_dispersion['mean_abs_over_std'].median():.4f}`. The full cell-level "
        "values are in `mean_dispersion.csv`; this separates population-average "
        "utility from node-level heterogeneity.",
        "",
        "## 12. Effect-size-aware seed stability",
        "",
        "| minimum |mu|/sigma | cells | same-sign cells | fraction |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in stability_summary.iterrows():
        lines.append(
            f"| {row['effect_threshold']:.2f} | {int(row['n_cells'])} | "
            f"{int(row['n_same_sign'])} | {row['same_sign_fraction']:.3f} |"
        )
    lines += [
        "",
        "The denominator for each row is restricted to cells whose aggregate "
        "effect magnitude meets that threshold; near-zero cells are not treated "
        "as equally strong evidence.",
        "",
        "## 13. Response magnitude association and modality differences",
        "",
        f"Within-cell node-level Spearman correlations have mean "
        f"`{within_association.mean():.4f}` and median `{within_association.median():.4f}`. "
        f"Across the 30 aggregate cells, Spearman(mean response norm, "
        f"|mean utility|) is `{cross_association:.4f}`. These answer different "
        "questions: node-level prediction within a condition versus utility scale "
        "differences across conditions.",
        "",
        "| order | modality mean absolute difference | modality median absolute difference |",
        "|---:|---:|---:|",
    ]
    for _, row in modality_order_summary.sort_values("order").iterrows():
        lines.append(
            f"| {int(row['order'])} | {row['absolute_difference_mean']:.5g} | "
            f"{row['absolute_difference_median']:.5g} |"
        )
    lines += [
        "",
        "Per-dataset/order modality differences are in `modality_difference.csv`.",
        "",
        "## 14. Propagation saturation",
        "",
        "A validation cell is flagged near-degenerate when its aggregate median "
        "`||R_k||/(||S_{k-1}||+epsilon)` is at most `1e-3`. This is a descriptive "
        "flag, not a utility sign threshold.",
    ]
    if saturated_cells.empty:
        lines.append("No aggregate cell crossed the near-degenerate threshold.")
    else:
        lines.append(
            "Flagged cells: "
            + ", ".join(
                f"{row['dataset']}/{row['modality']}/order{int(row['order'])}"
                for _, row in saturated_cells.iterrows()
            )
            + "."
        )
    lines += [
        "The full saturation table is `saturation.csv`; utility signs in flagged "
        "cells should not be interpreted as strong evidence of useful response "
        "heterogeneity.",
        "",
        "## 15. Findings for H1.1–H1.6",
        "",
        f"- **H1.1 Non-degeneracy:** {len(nondegenerate)}/{len(validation)} "
        "aggregate cells contain both positive and negative CE utility; this "
        "directly describes whether response utility is mixed rather than imposing "
        "a threshold.",
        f"- **H1.2 Node conditionality:** {len(conditional)}/{len(validation)} "
        "cells have nonzero node-level CE dispersion; inspect the exported std "
        "and quantiles for its magnitude.",
        f"- **H1.3 Modality conditionality:** the mean absolute text/visual "
        f"utility-mean difference over {len(modality_diffs)} paired cells is "
        f"`{modality_diffs.mean():.5g}`.",
        f"- **H1.4 Order conditionality:** the mean within-cell spread across "
        f"orders over {len(order_spread)} dataset × modality pairs is "
        f"`{order_spread.mean():.5g}`.",
        f"- **H1.5 Magnitude insufficiency:** within-cell response-norm/CE "
        f"Spearman has mean `{within_association.mean():.4f}` and median "
        f"`{within_association.median():.4f}`, so magnitude is weak for node-level "
        "utility ranking. Across aggregate cells, however, the correlation with "
        f"`|mean utility|` is `{cross_association:.4f}`; response attenuation can "
        "explain utility scale across orders while remaining insufficient to decide "
        "node-level helpful versus harmful sign.",
        f"- **H1.6 Stability:** same-sign seed means occur in "
        f"{stable_sign_cells}/{total_seed_cells} complete cells. Cross-dataset "
        "repetition is therefore reported explicitly rather than declared from "
        "one pooled number.",
        "",
        "## 16. Implementation caveats / limitations",
        "",
        "This is a frozen-forward functional leave-one-response-out intervention, "
        "not a strict causal effect. It uses validation nodes for the decision, "
        "does not retrain a response-specific head, and keeps the original "
        "classifier and downstream modules fixed. The inherited inactive RCMI/MRC "
        "parameters remain in the checkpoint but are bypassed by `all_plain`.",
        "",
        "## 17. Recommendation for P2",
        "",
        "The follow-up P2 cross-modal evidence probe is reported separately in "
        "`docs/cssi_p2_crossmodal_evidence_report.md`; this P1 implementation "
        "does not implement or select a P2 mechanism.",
        "",
        "## Reproducibility",
        "",
        "- Analysis splits: " + ", ".join(splits),
        "- Seeds: 42, 43, 44",
        "- Checkpoint selection: best validation accuracy",
        "- Test evaluation: disabled",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/cssi_p1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p1"))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--splits", nargs="+", default=["validation"], choices=["train", "validation", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/cssi_p1_corrected_report.md"),
    )
    parser.add_argument(
        "--reuse-node-exports",
        action="store_true",
        help="reuse already generated node-level exports and only rebuild summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    experiment_root = args.experiment_root
    output_root = args.output_root
    if not experiment_root.is_absolute():
        experiment_root = (project_root / experiment_root).resolve()
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    report_path = args.report_path
    if not report_path.is_absolute():
        report_path = (project_root / report_path).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for response analysis but CUDA is unavailable")

    frames: list[pd.DataFrame] = []
    check_rows: list[dict[str, Any]] = []
    if args.reuse_node_exports:
        for dataset in args.datasets:
            for seed in args.seeds:
                split_frames = []
                for split in args.splits:
                    node_path = output_root / "node_level" / dataset / f"seed_{seed}_{split}.csv.gz"
                    if not node_path.is_file():
                        raise FileNotFoundError(f"missing node export: {node_path}")
                    split_frames.append(pd.read_csv(node_path))
                frame = pd.concat(split_frames, ignore_index=True)
                frames.append(frame)
                print(f"reused {dataset} seed={seed} rows={len(frame)}", flush=True)
        existing_checks = experiment_root / "equivalence_checks.csv"
        if existing_checks.is_file():
            check_rows = pd.read_csv(existing_checks).to_dict("records")
        else:
            raise FileNotFoundError(f"missing equivalence checks: {existing_checks}")
    else:
        for dataset in args.datasets:
            for seed in args.seeds:
                frame, check_row = _run_one(
                    project_root,
                    experiment_root,
                    output_root,
                    dataset,
                    seed,
                    device,
                    tuple(args.splits),
                    args.epsilon,
                    args.tolerance,
                )
                frames.append(frame)
                check_rows.append(check_row)
                print(f"analyzed {dataset} seed={seed} rows={len(frame)}", flush=True)

    all_nodes = pd.concat(frames, ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    seed_groups = {
        key: group
        for key, group in all_nodes.groupby(
            ["dataset", "seed", "split", "modality", "order"],
            observed=True,
            sort=False,
        )
    }
    cell_groups = {
        key: group
        for key, group in all_nodes.groupby(
            ["dataset", "split", "modality", "order"],
            observed=True,
            sort=False,
        )
    }
    for dataset in args.datasets:
        for seed in args.seeds:
            for split in args.splits:
                for modality in MODALITIES:
                    for order in ORDERS:
                        for degree_group in ("all", "low", "medium", "high"):
                            frame = seed_groups[
                                (dataset, seed, split, modality, order)
                            ].copy()
                            if degree_group == "all":
                                frame["degree_group"] = "all"
                            else:
                                frame = frame[frame["degree_group"] == degree_group]
                            summary_rows.append(
                                _summary_for_group(
                                    frame,
                                    dataset=dataset,
                                    seed=seed,
                                    split=split,
                                    modality=modality,
                                    order=order,
                                    degree_group=degree_group,
                                    aggregation="seed",
                                    epsilon=args.epsilon,
                                )
                            )
    for dataset in args.datasets:
        for split in args.splits:
            for modality in MODALITIES:
                for order in ORDERS:
                    for degree_group in ("all", "low", "medium", "high"):
                        frame = cell_groups[
                            (dataset, split, modality, order)
                        ].copy()
                        if degree_group == "all":
                            frame["degree_group"] = "all"
                        else:
                            frame = frame[frame["degree_group"] == degree_group]
                        summary_rows.append(
                            _summary_for_group(
                                frame,
                                dataset=dataset,
                                seed=None,
                                split=split,
                                modality=modality,
                                order=order,
                                degree_group=degree_group,
                                aggregation="seed_aggregate",
                                epsilon=args.epsilon,
                            )
                        )
    summary = pd.DataFrame(summary_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "summary.csv", index=False)
    diagnostics = _p1_diagnostics(all_nodes, summary, args.epsilon)
    for name, frame in diagnostics.items():
        frame.to_csv(output_root / f"{name}.csv", index=False)
    checks = pd.DataFrame(check_rows)
    checks.to_csv(output_root / "equivalence_checks.csv", index=False)
    sanity = _load_backbone_sanity(experiment_root)
    sanity.to_csv(output_root / "backbone_sanity.csv", index=False)
    _write_report(
        report_path,
        summary,
        sanity,
        checks,
        diagnostics,
        output_root,
        args.epsilon,
        tuple(args.splits),
    )
    print(f"wrote {output_root / 'summary.csv'}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
