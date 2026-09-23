from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_cssi_p2_crossmodal_evidence import (
    DESCRIPTOR_COLUMNS,
    _descriptor_columns,
    build_probe_features,
    stable_permutation,
)


def _frame(split: str, node_start: int, count: int = 6) -> pd.DataFrame:
    rows = []
    for node in range(node_start, node_start + count):
        for modality in ("text", "visual"):
            for order in (1, 2, 3):
                base = float(node + 10 * (modality == "visual") + order)
                rows.append(
                    {
                        "dataset": "Movies",
                        "seed": 42,
                        "split": split,
                        "node_id": node,
                        "label": node % 2,
                        "modality": modality,
                        "order": order,
                        "response_norm": base,
                        "normalized_response_norm": base / 10.0,
                        "cosine_response_vs_h0": 0.1,
                        "cosine_response_vs_previous": 0.2,
                        "cosine_previous_vs_h0": 0.3,
                        "utility_ce": 1.0 if node % 2 else -1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_descriptor_columns_are_finite_and_have_five_fields():
    frame = _frame("train", 0, 2).iloc[:6].copy()
    frame.loc[frame.index[0], "response_norm"] = np.inf
    frame.loc[frame.index[1], "normalized_response_norm"] = np.nan
    values = _descriptor_columns(frame)
    assert list(values.columns) == list(DESCRIPTOR_COLUMNS)
    assert np.isfinite(values.to_numpy()).all()


def test_probe_dimensions_and_shuffled_marginal_are_equal():
    frame = _frame("train", 0)
    features, utility, node_ids, _ = build_probe_features(
        frame,
        dataset="Movies",
        seed=42,
        split="train",
        target_modality="text",
        target_order=2,
    )
    assert utility.shape == (6,)
    assert node_ids.tolist() == list(range(6))
    assert features["own"].shape == (6, 15)
    assert features["paired_same_order"].shape == (6, 20)
    assert features["paired_all_orders"].shape == (6, 30)
    assert features["shuffled_all_orders"].shape == (6, 30)
    assert np.allclose(
        np.sort(features["paired_all_orders"][:, 15:], axis=0),
        np.sort(features["shuffled_all_orders"][:, 15:], axis=0),
    )


def test_split_isolation_and_shuffle_are_deterministic_and_split_specific():
    train = _frame("train", 0)
    validation = _frame("validation", 100)
    _, _, train_nodes, train_audit = build_probe_features(
        train,
        dataset="Movies",
        seed=42,
        split="train",
        target_modality="visual",
        target_order=1,
    )
    _, _, val_nodes, val_audit = build_probe_features(
        validation,
        dataset="Movies",
        seed=42,
        split="validation",
        target_modality="visual",
        target_order=1,
    )
    assert set(train_nodes).isdisjoint(set(val_nodes))
    assert train_audit["split"] == "train"
    assert val_audit["split"] == "validation"
    train_perm_a = stable_permutation(
        10,
        dataset="Movies",
        seed=42,
        target_modality="visual",
        target_order=1,
        split="train",
    )
    train_perm_b = stable_permutation(
        10,
        dataset="Movies",
        seed=42,
        target_modality="visual",
        target_order=1,
        split="train",
    )
    validation_perm = stable_permutation(
        10,
        dataset="Movies",
        seed=42,
        target_modality="visual",
        target_order=1,
        split="validation",
    )
    assert np.array_equal(train_perm_a, train_perm_b)
    assert not np.array_equal(train_perm_a, validation_perm)
