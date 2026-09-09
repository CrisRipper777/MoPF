from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.data.types import MAGData
from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels


def _data(
    labels: list[int],
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    num_classes: int,
) -> MAGData:
    num_nodes = len(labels)
    return MAGData(
        name="metric-toy",
        source="test",
        task="nc",
        x=torch.zeros((num_nodes, 1)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=num_nodes,
        y=torch.tensor(labels, dtype=torch.long),
        train_idx=torch.tensor(train_idx, dtype=torch.long),
        val_idx=torch.tensor(val_idx, dtype=torch.long),
        test_idx=torch.tensor(test_idx, dtype=torch.long),
        num_classes=num_classes,
    )


def _evaluate(logit_rows: list[list[float]], labels: list[int], eval_labels: list[int]) -> dict[str, float]:
    logits = torch.tensor(logit_rows, dtype=torch.float32)
    return _evaluate_split(
        nn.Identity(),
        logits,
        torch.tensor(labels, dtype=torch.long),
        torch.arange(len(labels)),
        torch.device("cpu"),
        batch_size=32,
        eval_labels=eval_labels,
    )


def test_phantom_prediction_is_penalized_without_entering_macro_f1_denominator() -> None:
    metrics = _evaluate(
        [[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]],
        [0, 1],
        [0, 1],
    )

    assert metrics["acc"] == 0.5
    assert metrics["macro_f1"] == 0.5


def test_num_classes_larger_than_observed_labels_does_not_add_phantom_classes() -> None:
    data = _data(
        labels=[0, 2, 0, 2, -1],
        train_idx=[0, 1],
        val_idx=[2, 3],
        test_idx=[4],
        num_classes=6,
    )

    assert _resolve_nc_eval_labels(data) == [0, 2]


def test_union_label_set_keeps_class_missing_from_test_split() -> None:
    data = _data(
        labels=[0, 1, 0, 0, 0],
        train_idx=[0, 1],
        val_idx=[2, 3],
        test_idx=[2, 3, 4],
        num_classes=3,
    )

    eval_labels = _resolve_nc_eval_labels(data)
    assert eval_labels == [0, 1]
    test_metrics = _evaluate(
        [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        [0, 0, 0],
        eval_labels,
    )
    assert test_metrics["macro_f1"] == 0.5


def test_contiguous_labels_match_legacy_macro_f1() -> None:
    labels = [0, 1, 2, 1, 0, 2]
    logit_rows = [
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 3.0],
    ]
    expected = f1_score(labels, [0, 1, 2, 0, 1, 2], average="macro", zero_division=0)

    metrics = _evaluate(logit_rows, labels, [0, 1, 2])
    assert metrics["macro_f1"] == expected


def test_accuracy_is_unchanged_by_fixed_label_set() -> None:
    metrics = _evaluate(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        [0, 1, 1],
        [0, 1],
    )

    assert abs(metrics["acc"] - 2 / 3) < 1e-6
