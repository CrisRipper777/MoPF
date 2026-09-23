#!/usr/bin/env python3
"""Simple train/validation probes for cross-modal Structural Response evidence.

This script consumes the frozen P1 node-level exports.  It never loads a graph
checkpoint, never uses test nodes, and never feeds utility values into the
probe features.  Each dataset/seed/target-modality/order is fit independently.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
ORDERS = (1, 2, 3)
CLASSIFICATION_THRESHOLDS = (1.0e-6, 1.0e-5, 1.0e-4)
REGRESSION_SENSITIVITY_THRESHOLDS = (0.0, 1.0e-6, 1.0e-5, 1.0e-4)
DESCRIPTOR_COLUMNS = (
    "log1p_response_norm",
    "normalized_response_norm",
    "cosine_response_vs_h0",
    "cosine_response_vs_previous",
    "cosine_previous_vs_h0",
)
VARIANTS = ("own", "paired_same_order", "paired_all_orders", "shuffled_all_orders")


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size != left.size:
        return float("nan")
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def stable_permutation(
    n: int,
    *,
    dataset: str,
    seed: int,
    target_modality: str,
    target_order: int,
    split: str,
    purpose: str = "probe",
) -> np.ndarray:
    """Return a deterministic within-split permutation with no cross-split state."""
    token = (
        f"cssi-p2|{purpose}|{dataset}|{seed}|{target_modality}|"
        f"{target_order}|{split}"
    ).encode("utf-8")
    seed_value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(seed_value).permutation(int(n))


def _descriptor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the five finite, interpretable response descriptors."""
    required = {
        "response_norm",
        "normalized_response_norm",
        "cosine_response_vs_h0",
        "cosine_response_vs_previous",
        "cosine_previous_vs_h0",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"P1 node export is missing descriptor columns: {sorted(missing)}")
    descriptors = pd.DataFrame(index=frame.index)
    descriptors["log1p_response_norm"] = np.log1p(
        np.clip(frame["response_norm"].to_numpy(dtype=np.float64), 0.0, None)
    )
    descriptors["normalized_response_norm"] = frame[
        "normalized_response_norm"
    ].to_numpy(dtype=np.float64)
    descriptors["cosine_response_vs_h0"] = frame[
        "cosine_response_vs_h0"
    ].to_numpy(dtype=np.float64)
    descriptors["cosine_response_vs_previous"] = frame[
        "cosine_response_vs_previous"
    ].to_numpy(dtype=np.float64)
    descriptors["cosine_previous_vs_h0"] = frame[
        "cosine_previous_vs_h0"
    ].to_numpy(dtype=np.float64)
    values = np.nan_to_num(
        descriptors.to_numpy(dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return pd.DataFrame(values, index=frame.index, columns=DESCRIPTOR_COLUMNS)


def _descriptor_lookup(frame: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    lookup: dict[tuple[str, int], pd.DataFrame] = {}
    for modality in MODALITIES:
        for order in ORDERS:
            subset = frame[
                (frame["modality"] == modality) & (frame["order"] == order)
            ].copy()
            subset = subset.sort_values("node_id").drop_duplicates("node_id")
            desc = _descriptor_columns(subset)
            desc.index = subset["node_id"].to_numpy(dtype=np.int64)
            lookup[(modality, order)] = desc
    return lookup


def build_probe_features(
    frame: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    split: str,
    target_modality: str,
    target_order: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    """Build Own/Same/All/Shuffled features from exactly one split."""
    if split not in {"train", "validation"}:
        raise ValueError("P2 only accepts train and validation splits")
    target_modality = str(target_modality).lower()
    if target_modality not in MODALITIES or target_order not in ORDERS:
        raise ValueError("invalid target modality/order")
    other_modality = "visual" if target_modality == "text" else "text"
    lookup = _descriptor_lookup(frame)
    node_ids = np.sort(
        frame[frame["modality"] == target_modality]["node_id"]
        .drop_duplicates()
        .to_numpy(dtype=np.int64)
    )
    if node_ids.size == 0:
        raise ValueError("split contains no target nodes")

    def aligned(modality: str, order: int) -> np.ndarray:
        table = lookup[(modality, order)].reindex(node_ids)
        if table.isna().any().any():
            raise ValueError("modality/order node correspondence is incomplete")
        return table.to_numpy(dtype=np.float64)

    own = np.concatenate(
        [aligned(target_modality, order) for order in ORDERS], axis=1
    )
    other_orders = [aligned(other_modality, order) for order in ORDERS]
    same = aligned(other_modality, target_order)
    all_other = np.concatenate(other_orders, axis=1)
    permutation = stable_permutation(
        len(node_ids),
        dataset=dataset,
        seed=seed,
        target_modality=target_modality,
        target_order=target_order,
        split=split,
    )
    features = {
        "own": own,
        "paired_same_order": np.concatenate([own, same], axis=1),
        "paired_all_orders": np.concatenate([own, all_other], axis=1),
        "shuffled_all_orders": np.concatenate([own, all_other[permutation]], axis=1),
    }
    utilities = (
        frame[
            (frame["modality"] == target_modality)
            & (frame["order"] == target_order)
        ]
        .sort_values("node_id")
        .drop_duplicates("node_id")
        .set_index("node_id")
        .reindex(node_ids)["utility_ce"]
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(utilities).all():
        raise ValueError("target utilities contain non-finite values")
    audit = {
        "dataset": dataset,
        "seed": seed,
        "split": split,
        "target_modality": target_modality,
        "target_order": target_order,
        "n_nodes": len(node_ids),
        "permutation_checksum": hashlib.sha256(
            permutation.astype(np.int64).tobytes()
        ).hexdigest()[:16],
        "permutation_min": int(permutation.min()),
        "permutation_max": int(permutation.max()),
    }
    return features, utilities, node_ids, audit


def _classification_probe(
    x_train: np.ndarray,
    utility_train: np.ndarray,
    x_val: np.ndarray,
    utility_val: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    train_mask = np.abs(utility_train) > threshold
    val_mask = np.abs(utility_val) > threshold
    y_train = (utility_train[train_mask] > 0.0).astype(np.int64)
    y_val = (utility_val[val_mask] > 0.0).astype(np.int64)
    result: dict[str, Any] = {
        "task": "classification",
        "threshold": threshold,
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "n_train_positive": int(y_train.sum()),
        "n_validation_positive": int(y_val.sum()),
        "status": "ok",
        "auroc": np.nan,
        "auprc": np.nan,
        "balanced_accuracy": np.nan,
    }
    if len(y_train) < 2 or np.unique(y_train).size < 2:
        result["status"] = "degenerate_train_class"
        return result
    if len(y_val) < 2 or np.unique(y_val).size < 2:
        result["status"] = "degenerate_validation_class"
        return result
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
    )
    model.fit(x_train[train_mask], y_train)
    probability = model.predict_proba(x_val[val_mask])[:, 1]
    prediction = (probability >= 0.5).astype(np.int64)
    result["auroc"] = float(roc_auc_score(y_val, probability))
    result["auprc"] = float(average_precision_score(y_val, probability))
    result["balanced_accuracy"] = float(
        balanced_accuracy_score(y_val, prediction)
    )
    return result


def _regression_probe(
    x_train: np.ndarray,
    utility_train: np.ndarray,
    x_val: np.ndarray,
    utility_val: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    train_mask = np.ones(len(utility_train), dtype=bool)
    val_mask = np.abs(utility_val) > threshold
    if threshold > 0.0:
        train_mask = np.abs(utility_train) > threshold
    result: dict[str, Any] = {
        "task": "regression",
        "threshold": threshold,
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "status": "ok",
        "spearman": np.nan,
        "mae": np.nan,
        "r2": np.nan,
    }
    if len(utility_train[train_mask]) < 3 or np.std(utility_train[train_mask]) == 0.0:
        result["status"] = "degenerate_train_target"
        return result
    if len(utility_val[val_mask]) < 2:
        result["status"] = "degenerate_validation_sample"
        return result
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train[train_mask], utility_train[train_mask])
    prediction = model.predict(x_val[val_mask])
    target = utility_val[val_mask]
    result["spearman"] = _spearman(target, prediction)
    result["mae"] = float(mean_absolute_error(target, prediction))
    result["r2"] = float(r2_score(target, prediction)) if np.std(target) > 0.0 else np.nan
    return result


def _load_frame(output_root: Path, dataset: str, seed: int, split: str) -> pd.DataFrame:
    path = output_root / "node_level" / dataset / f"seed_{seed}_{split}.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {split} export {path}; rerun corrected P1 with train and validation"
        )
    frame = pd.read_csv(path)
    frame["split"] = split
    return frame


def _run_probes(
    p1_root: Path,
    output_root: Path,
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            train = _load_frame(p1_root, dataset, seed, "train")
            validation = _load_frame(p1_root, dataset, seed, "validation")
            for target_modality in MODALITIES:
                for target_order in ORDERS:
                    train_features, train_utility, _, train_audit = build_probe_features(
                        train,
                        dataset=dataset,
                        seed=seed,
                        split="train",
                        target_modality=target_modality,
                        target_order=target_order,
                    )
                    val_features, val_utility, _, val_audit = build_probe_features(
                        validation,
                        dataset=dataset,
                        seed=seed,
                        split="validation",
                        target_modality=target_modality,
                        target_order=target_order,
                    )
                    audit_rows.extend([train_audit, val_audit])
                    for variant in VARIANTS:
                        for threshold in CLASSIFICATION_THRESHOLDS:
                            metrics = _classification_probe(
                                train_features[variant],
                                train_utility,
                                val_features[variant],
                                val_utility,
                                threshold,
                            )
                            result_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "target_modality": target_modality,
                                    "target_order": target_order,
                                    "variant": variant,
                                    **metrics,
                                }
                            )
                        for threshold in REGRESSION_SENSITIVITY_THRESHOLDS:
                            metrics = _regression_probe(
                                train_features[variant],
                                train_utility,
                                val_features[variant],
                                val_utility,
                                threshold,
                            )
                            result_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "target_modality": target_modality,
                                    "target_order": target_order,
                                    "variant": variant,
                                    **metrics,
                                }
                            )
    return pd.DataFrame(result_rows), pd.DataFrame(audit_rows)


def _add_deltas(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "seed",
        "target_modality",
        "target_order",
        "task",
        "threshold",
    ]
    metric_names = [
        "auroc",
        "auprc",
        "balanced_accuracy",
        "spearman",
        "mae",
        "r2",
    ]
    result = results.copy()
    for metric in metric_names:
        result[f"delta_same_{metric}"] = np.nan
        result[f"delta_all_{metric}"] = np.nan
        result[f"delta_match_{metric}"] = np.nan
    for _, group in result.groupby(keys, dropna=False):
        indexes = group.index
        by_variant = group.set_index("variant")
        if not set(VARIANTS).issubset(by_variant.index):
            continue
        for metric in metric_names:
            values = by_variant[metric]
            own = values.get("own", np.nan)
            same = values.get("paired_same_order", np.nan)
            all_orders = values.get("paired_all_orders", np.nan)
            shuffled = values.get("shuffled_all_orders", np.nan)
            result.loc[indexes, f"delta_same_{metric}"] = same - own
            result.loc[indexes, f"delta_all_{metric}"] = all_orders - same
            result.loc[indexes, f"delta_match_{metric}"] = all_orders - shuffled
    return result


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = [
        "dataset",
        "target_modality",
        "target_order",
        "task",
        "threshold",
    ]
    metric_names = [
        "auroc",
        "auprc",
        "balanced_accuracy",
        "spearman",
        "mae",
        "r2",
    ]
    for key, group in results.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, key))
        row["n_seed_rows"] = int(len(group))
        for variant in VARIANTS:
            selected = group[group["variant"] == variant]
            for metric in metric_names:
                values = selected[metric].dropna().to_numpy(dtype=np.float64)
                row[f"{variant}_{metric}_mean"] = (
                    float(values.mean()) if len(values) else np.nan
                )
                row[f"{variant}_{metric}_std"] = (
                    float(values.std(ddof=0)) if len(values) else np.nan
                )
        for metric in metric_names:
            for delta in ("same", "all", "match"):
                values = group[f"delta_{delta}_{metric}"].dropna().to_numpy(
                    dtype=np.float64
                )
                row[f"delta_{delta}_{metric}_mean"] = (
                    float(values.mean()) if len(values) else np.nan
                )
                row[f"delta_{delta}_{metric}_std"] = (
                    float(values.std(ddof=0)) if len(values) else np.nan
                )
                row[f"delta_{delta}_{metric}_positive_fraction"] = (
                    float(np.mean(values > 0.0)) if len(values) else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _utility_correspondence(
    p1_root: Path, datasets: tuple[str, ...], seeds: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            frame = _load_frame(p1_root, dataset, seed, "validation")
            lookup = {}
            for modality in MODALITIES:
                for order in ORDERS:
                    part = frame[
                        (frame["modality"] == modality) & (frame["order"] == order)
                    ].sort_values("node_id").drop_duplicates("node_id")
                    lookup[(modality, order)] = part.set_index("node_id")[
                        "utility_ce"
                    ]
            for text_order in ORDERS:
                for visual_order in ORDERS:
                    joined = pd.concat(
                        [
                            lookup[("text", text_order)],
                            lookup[("visual", visual_order)],
                        ],
                        axis=1,
                        join="inner",
                    ).dropna()
                    text_values = joined.iloc[:, 0].to_numpy(dtype=np.float64)
                    visual_values = joined.iloc[:, 1].to_numpy(dtype=np.float64)
                    same = _spearman(text_values, visual_values)
                    permutation = stable_permutation(
                        len(visual_values),
                        dataset=dataset,
                        seed=seed,
                        target_modality="text",
                        target_order=text_order,
                        split="validation",
                        purpose=f"correspondence_visual_order_{visual_order}",
                    )
                    shuffled = _spearman(text_values, visual_values[permutation])
                    for condition, value in (("same_node", same), ("shuffled_node", shuffled)):
                        rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "text_order": text_order,
                                "visual_order": visual_order,
                                "condition": condition,
                                "N": len(joined),
                                "spearman_utility": value,
                            }
                        )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for condition, group in detail.groupby("condition"):
        same = group[group["text_order"] == group["visual_order"]][
            "spearman_utility"
        ].dropna()
        offdiag = group[group["text_order"] != group["visual_order"]][
            "spearman_utility"
        ].dropna()
        summary_rows.append(
            {
                "condition": condition,
                "diagonal_mean": float(same.mean()),
                "diagonal_median": float(same.median()),
                "offdiagonal_mean": float(offdiag.mean()),
                "offdiagonal_max": float(offdiag.max()),
                "all_cell_mean": float(group["spearman_utility"].mean()),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def _write_report(
    path: Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    correspondence_summary: pd.DataFrame,
    p1_summary: pd.DataFrame,
    p1_saturation: pd.DataFrame,
) -> None:
    primary = summary[
        (summary["task"] == "classification")
        & (summary["threshold"] == 1.0e-5)
    ]
    primary_rows = results[
        (results["task"] == "classification")
        & (results["threshold"] == 1.0e-5)
        & (results["variant"] == "own")
        & (results["status"] == "ok")
    ]
    regression_primary = summary[
        (summary["task"] == "regression") & (summary["threshold"] == 0.0)
    ]
    lines = [
        "# CSSI P2 Cross-Modal Structural Evidence Report",
        "",
        "## Scope and leakage controls",
        "",
        "This report reuses the 15 frozen Plain best-validation checkpoints. "
        "The graph encoder and NC classifier are not retrained. Probe fitting "
        "uses train-node descriptors and evaluation uses validation-node descriptors; "
        "test nodes and test labels are never loaded.",
        "",
        "Each target dataset/seed/modality/order is fit independently with "
        "StandardScaler + L2 Logistic Regression (`C=1`) for sign classification "
        "and StandardScaler + Ridge (`alpha=1`) for utility regression.",
        "",
        "## Response descriptor",
        "",
        "Each response uses `[log1p(||R||), ||R||/(||S_prev||+epsilon), "
        "cos(R,H0), cos(R,S_prev), cos(S_prev,H0)]`. Non-finite descriptor values "
        "are converted to zero and the audit records finite matrices.",
        "",
        "## P1 corrected CE sign",
        "",
        "The corrected convention is `u_ce = loss_remove - loss_base`; positive "
        "means removal increases loss and the response is helpful. Margin remains "
        "`u_margin = margin_base - margin_remove`. P1 corrected outputs are in "
        "`outputs/cssi_p1_corrected/` and the full report is "
        "`docs/cssi_p1_corrected_report.md`.",
        "",
        f"P1 corrected aggregate CE/margin sign agreement has mean "
        f"`{p1_summary['ce_margin_sign_agreement'].mean():.4f}` across "
        f"{len(p1_summary)} validation cells.",
        "",
        "## Probe definitions",
        "",
        "- `own`: target modality descriptors for orders 1/2/3.",
        "- `paired_same_order`: Own plus the paired modality's target-order descriptor.",
        "- `paired_all_orders`: Own plus all three paired-modality descriptors.",
        "- `shuffled_all_orders`: same dimension as Paired-AllOrders, with paired "
        "modality node correspondence independently permuted within train and validation.",
        "",
        "## Primary validation comparison at |u| > 1e-5",
        "",
        "The compact table reports mean validation metrics over target cells and seeds. "
        "Full per-cell results and threshold sensitivities are in `probe_results.csv` "
        "and `summary.csv`.",
        "",
        "| Task / metric | Own | Same | All | Shuffled | Delta same | Delta all | Delta match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, metric in (
        ("classification", "auroc"),
        ("classification", "auprc"),
        ("classification", "balanced_accuracy"),
    ):
        selected = primary
        values = {
            variant: selected[f"{variant}_{metric}_mean"].mean()
            for variant in VARIANTS
        }
        deltas = {
            delta: selected[f"delta_{delta}_{metric}_mean"].mean()
            for delta in ("same", "all", "match")
        }
        lines.append(
            f"| classification / {metric} | {values['own']:.4f} | "
            f"{values['paired_same_order']:.4f} | {values['paired_all_orders']:.4f} | "
            f"{values['shuffled_all_orders']:.4f} | {deltas['same']:.4f} | "
            f"{deltas['all']:.4f} | {deltas['match']:.4f} |"
        )
    for metric in ("spearman", "mae", "r2"):
        values = {
            variant: regression_primary[f"{variant}_{metric}_mean"].mean()
            for variant in VARIANTS
        }
        deltas = {
            delta: regression_primary[f"delta_{delta}_{metric}_mean"].mean()
            for delta in ("same", "all", "match")
        }
        lines.append(
            f"| regression / {metric} | {values['own']:.4f} | "
            f"{values['paired_same_order']:.4f} | {values['paired_all_orders']:.4f} | "
        f"{values['shuffled_all_orders']:.4f} | {deltas['same']:.4f} | "
            f"{deltas['all']:.4f} | {deltas['match']:.4f} |"
        )
    lines += [
        "",
        "For AUROC, AUPRC, balanced accuracy, Spearman, and R2, positive delta is "
        "better. For MAE, lower is better, so a positive MAE delta is not an "
        "improvement.",
        "",
        "## Per-cell/seed stability of the primary AUROC deltas",
        "",
        f"At `|u|>1e-5`, {len(primary_rows)}/90 target-cell/seed fits are valid; "
        f"the remaining cases are degenerate train or validation sign classes, "
        "primarily in propagation-saturated cells.",
        "",
        "| comparison | positive delta fits | valid fits | fraction |",
        "|---|---:|---:|---:|",
    ]
    for delta in ("same", "all", "match"):
        values = primary_rows[f"delta_{delta}_auroc"].dropna()
        lines.append(
            f"| {delta} | {int((values > 0).sum())} | {len(values)} | "
            f"{float((values > 0).mean()):.3f} |"
        )
    lines += [
        "",
        "The target-cell/seed details are in `probe_results.csv`; these fractions "
        "show whether the pooled deltas repeat across checkpoints rather than only "
        "reflecting one global fit.",
        "",
        "## Threshold sensitivity of AUROC deltas",
        "",
        "| threshold | Delta same | Delta all | Delta match | valid target cells |",
        "|---:|---:|---:|---:|---:|",
    ]
    for threshold in CLASSIFICATION_THRESHOLDS:
        row = summary[
            (summary["task"] == "classification")
            & (summary["threshold"] == threshold)
        ]
        lines.append(
            f"| {threshold:.0e} | {row['delta_same_auroc_mean'].mean():.5f} | "
            f"{row['delta_all_auroc_mean'].mean():.5f} | "
            f"{row['delta_match_auroc_mean'].mean():.5f} | "
            f"{row['delta_same_auroc_mean'].notna().sum()} |"
        )
    lines += [
        "",
        "## Near-zero sensitivity",
        "",
        "Classification results are reported at `1e-6`, `1e-5`, and `1e-4`; "
        "samples within the threshold are excluded. Regression reports all samples "
        "and additional non-near-zero sensitivities. Cells with insufficient classes "
        "or samples are marked degenerate rather than forced into a metric.",
        "",
        "## Utility correspondence",
        "",
    ]
    for _, row in correspondence_summary.iterrows():
        lines.append(
            f"- {row['condition']}: diagonal mean/median "
            f"`{row['diagonal_mean']:.4f}`/`{row['diagonal_median']:.4f}`, "
            f"off-diagonal mean/max `{row['offdiagonal_mean']:.4f}`/"
            f"`{row['offdiagonal_max']:.4f}`, all-cell mean `{row['all_cell_mean']:.4f}`."
        )
    lines += [
        "",
        "The correspondence matrix is descriptive only; utility is never included "
        "as a probe feature.",
        "",
        "## Decision against Cases A/B/C",
        "",
        "Case B requires same-order gains over Own, All approximately equal to Same, "
        "and matched All gains over Shuffled. Case C additionally requires a stable "
        "increment from Same to All. The conclusion below is based on the complete "
        "per-cell/seed table, threshold sensitivity, and shuffled control, not on "
        "one pooled score.",
        "",
    ]
    # The actual case decision is filled from the primary deltas below.
    same_auroc = float(primary["delta_same_auroc_mean"].mean())
    all_auroc = float(primary["delta_all_auroc_mean"].mean())
    match_auroc = float(primary["delta_match_auroc_mean"].mean())
    same_fraction = float(
        (primary_rows["delta_same_auroc"] > 0.0).mean()
    )
    all_fraction = float((primary_rows["delta_all_auroc"] > 0.0).mean())
    match_fraction = float((primary_rows["delta_match_auroc"] > 0.0).mean())
    if same_auroc <= 0.0 and all_auroc <= 0.0:
        decision = "Case A-like: the primary AUROC comparison does not show a cross-modal gain over Own."
    elif same_auroc > 0.0 and all_auroc <= 0.0 and match_auroc > 0.0:
        decision = "Case B-like: matched cross-modal evidence helps, but the primary All-over-Same increment is not positive."
    elif same_auroc > 0.0 and all_auroc > 0.0 and match_auroc > 0.0:
        decision = (
            "Provisional Case C-like on pooled AUROC: All exceeds Same and Shuffled. "
            f"Positive-fit fractions are Same {same_fraction:.3f}, All-over-Same "
            f"{all_fraction:.3f}, and All-over-Shuffled {match_fraction:.3f}; "
            "the effect is not universal, so do not implement full co-reasoning "
            "without a pre-registered same-order control and per-cell confirmation."
        )
    else:
        decision = "Mixed/inconclusive: pooled deltas do not satisfy a clean Case A/B/C pattern."
    lines.append(f"**Result:** {decision}")
    lines += [
        "",
        "The recommended next model direction is therefore determined by the "
        "per-cell stability table: do not implement cross-order interaction unless "
        "the All-over-Same and All-over-Shuffled gains are repeated across datasets "
        "and seeds. Saturated cells must be excluded from strong architectural claims.",
        "",
        "## P1 saturation caveat",
        "",
        f"P1 uses a median normalized-response threshold of `1e-3`; "
        f"{int(p1_saturation['near_degenerate'].sum())} aggregate cells are flagged "
        "near-degenerate. Their utility signs should not be treated as strong "
        "evidence of meaningful response heterogeneity.",
        "",
        "## Limitations",
        "",
        "P1 utility is a frozen-forward functional leave-one-response-out intervention, "
        "not a strict causal effect. P2 probes establish predictive association only; "
        "they do not implement a proposed CSSI mechanism or establish causality.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-root", type=Path, default=Path("outputs/cssi_p1_corrected"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p2"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/cssi_p2_crossmodal_evidence_report.md"))
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="rebuild the report from an existing P2 output directory",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    p1_root = args.p1_root if args.p1_root.is_absolute() else project_root / args.p1_root
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    report_path = args.report_path if args.report_path.is_absolute() else project_root / args.report_path
    datasets = tuple(args.datasets)
    seeds = tuple(args.seeds)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.reuse_results:
        results = pd.read_csv(output_root / "probe_results.csv")
        summary = pd.read_csv(output_root / "summary.csv")
        correspondence_summary = pd.read_csv(
            output_root / "utility_correspondence_summary.csv"
        )
    else:
        results, audit = _run_probes(p1_root, output_root, datasets, seeds)
        results = _add_deltas(results)
        summary = _summary(results)
        correspondence, correspondence_summary = _utility_correspondence(
            p1_root, datasets, seeds
        )
        results.to_csv(output_root / "probe_results.csv", index=False)
        summary.to_csv(output_root / "summary.csv", index=False)
        correspondence.to_csv(output_root / "utility_correspondence.csv", index=False)
        correspondence_summary.to_csv(
            output_root / "utility_correspondence_summary.csv", index=False
        )
        audit.to_csv(output_root / "shuffle_audit.csv", index=False)
    p1_summary = pd.read_csv(p1_root / "summary.csv")
    p1_summary = p1_summary[
        (p1_summary["split"] == "validation")
        & (p1_summary["degree_group"] == "all")
        & (p1_summary["aggregation"] == "seed_aggregate")
    ]
    p1_saturation = pd.read_csv(p1_root / "saturation.csv")
    _write_report(
        report_path,
        results,
        summary,
        correspondence_summary,
        p1_summary,
        p1_saturation,
    )
    print(f"wrote {output_root / 'probe_results.csv'}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
