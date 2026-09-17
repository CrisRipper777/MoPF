"""Frozen Core Story formal matrix and dataset-level protocol constants."""

from __future__ import annotations


CORE_STORY_PROTOCOL_VERSION = "cosi_mag_core_story_ablation_v1"
CORE_STORY_VARIANTS = (
    "wo_relation_calibration",
    "wo_semantic_anchor",
    "wo_adaptive_composition",
)
NC_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
LP_DATASETS = ("sports-copurchase", "cloth-copurchase")
FORMAL_SEEDS = (42, 43, 44)

# Confirmed against the final MoPF resolved configurations/manifests in the
# repository.  LP uses K=3 for both established copurchase protocols.
FORMAL_K = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
    "sports-copurchase": 3,
    "cloth-copurchase": 3,
}


def formal_k(dataset: str) -> int:
    try:
        return FORMAL_K[dataset]
    except KeyError as exc:
        valid = ", ".join(FORMAL_K)
        raise ValueError(f"no frozen formal K for {dataset!r}; expected one of: {valid}") from exc


def datasets_for_task(task: str) -> tuple[str, ...]:
    if task == "nc":
        return NC_DATASETS
    if task == "lp":
        return LP_DATASETS
    raise ValueError(f"task must be nc or lp, got {task!r}")


def expected_metric_keys(task: str) -> tuple[str, ...]:
    if task == "nc":
        return ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    if task == "lp":
        return ("val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10")
    raise ValueError(f"task must be nc or lp, got {task!r}")
