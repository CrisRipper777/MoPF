"""Frozen MoPF F2 ablation definitions and provenance helpers.

The resolver is independent of the model implementation so NC and LP share
exactly the same ablation vocabulary and manifest semantics.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AblationSpec:
    name: str
    learned_relation_calibration: bool
    semantic_anchor: bool
    global_preference: bool
    modality_residual: bool
    node_residual: bool
    tcpr: bool


_FULL = AblationSpec("full", True, True, True, True, True, True)

ABLATION_SPECS: dict[str, AblationSpec] = {
    "full": _FULL,
    "wo_learned_semantic_calibration": AblationSpec(
        "wo_learned_semantic_calibration", False, True, True, True, True, True
    ),
    "wo_semantic_anchor": AblationSpec(
        "wo_semantic_anchor", True, False, True, True, True, True
    ),
    "wo_tcpr": AblationSpec("wo_tcpr", True, True, True, True, True, False),
    "wo_node_adaptation": AblationSpec(
        "wo_node_adaptation", True, True, True, True, False, True
    ),
    "wo_modality_adaptation": AblationSpec(
        "wo_modality_adaptation", True, True, True, False, True, True
    ),
    "wo_anchor_tcpr": AblationSpec(
        "wo_anchor_tcpr", True, False, True, True, True, False
    ),
    "wo_node_tcpr": AblationSpec(
        "wo_node_tcpr", True, True, True, True, False, False
    ),
    "wo_modality_tcpr": AblationSpec(
        "wo_modality_tcpr", True, True, True, False, True, False
    ),
}

MAIN_ABLATIONS = (
    "wo_learned_semantic_calibration",
    "wo_semantic_anchor",
    "wo_tcpr",
    "wo_node_adaptation",
    "wo_modality_adaptation",
)
INTERACTION_ABLATIONS = (
    "wo_anchor_tcpr",
    "wo_node_tcpr",
    "wo_modality_tcpr",
)
ALL_ABLATIONS = MAIN_ABLATIONS + INTERACTION_ABLATIONS


def resolve_ablation(name: str | None) -> AblationSpec:
    key = "full" if name is None else str(name).strip().lower()
    try:
        return ABLATION_SPECS[key]
    except KeyError as exc:
        valid = ", ".join(ABLATION_SPECS)
        raise ValueError(f"unknown ablation {key!r}; expected one of: {valid}") from exc


def cfg_ablation(cfg: Any) -> AblationSpec:
    """Resolve top-level ``ablation`` while supporting legacy test configs."""
    if hasattr(cfg, "get"):
        value = cfg.get("ablation", "full")
    else:
        value = getattr(cfg, "ablation", "full")
    return resolve_ablation(value)


def effective_edge_weight_mode(model_cfg: Any, spec: AblationSpec) -> str:
    configured = str(model_cfg.get("edge_weight_mode", "separate_cos")).strip().lower()
    if not spec.learned_relation_calibration and configured == "learned_diag_cos":
        return "separate_cos"
    return configured


def effective_multihop_modes(model_cfg: Any, spec: AblationSpec) -> tuple[str, str]:
    """Resolve the frozen state/response factors with A2 changing state only."""
    legacy = model_cfg.get("multihop_mode", None)
    state_mode = model_cfg.get("multihop_state_mode", None)
    response_mode = model_cfg.get("multihop_response_mode", None)
    if legacy is not None:
        legacy_key = str(legacy).strip().lower()
        legacy_mapping = {
            "cumulative": ("ordinary", "cumulative"),
            "anchored_differential": ("anchored", "differential"),
        }
        if legacy_key not in legacy_mapping:
            raise ValueError(
                "model.multihop_mode must be cumulative|anchored_differential, "
                f"got {legacy_key!r}"
            )
        state_mode, response_mode = legacy_mapping[legacy_key]
    elif state_mode is None and response_mode is None:
        state_mode, response_mode = "ordinary", "cumulative"
    else:
        state_mode = "ordinary" if state_mode is None else str(state_mode)
        response_mode = "cumulative" if response_mode is None else str(response_mode)
    if not spec.semantic_anchor:
        state_mode = "ordinary"
    return str(state_mode).strip().lower(), str(response_mode).strip().lower()


def build_ablation_manifest(
    cfg: Any,
    *,
    dataset: str,
    task: str,
    seed: int,
    split_source: str | None,
    project_root: Path,
) -> dict[str, Any]:
    """Build the human- and machine-readable per-run F2 manifest."""
    spec = cfg_ablation(cfg)
    model_cfg = cfg.model
    task_cfg = cfg.task
    state_mode, response_mode = effective_multihop_modes(model_cfg, spec)
    raw_neighbors = task_cfg.get("num_neighbors", None)
    if raw_neighbors is not None and not isinstance(raw_neighbors, (str, bytes)):
        try:
            raw_neighbors = list(raw_neighbors)
        except TypeError:
            pass
    manifest = asdict(spec)
    manifest.update(
        {
            "ablation_name": spec.name,
            "edge_weight_mode": effective_edge_weight_mode(model_cfg, spec),
            "multihop_state_mode": state_mode,
            "multihop_response_mode": response_mode,
            "alpha": float(model_cfg.get("multihop_anchor_alpha", 0.1)),
            "K": int(model_cfg.get("max_order", 3)),
            "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
            "learning_rate": float(model_cfg.get("lr", task_cfg.get("lr"))),
            "weight_decay": float(model_cfg.get("weight_decay", task_cfg.get("weight_decay"))),
            "optimizer": str(task_cfg.get("optimizer", "adamw")),
            "max_epochs": int(task_cfg.get("epochs")),
            "patience": int(task_cfg.get("patience")),
            "temperature": float(model_cfg.get("edge_weight_temperature", 2.0)),
            "batch_size": int(task_cfg.get("batch_size", 0)),
            "num_neighbors": raw_neighbors,
            "negative_sampling": {
                "task_num_train_neg": task_cfg.get("num_train_neg", None),
                "dataset_lp_num_neg": cfg.dataset.get("lp_num_neg", None),
            },
            "evaluation_metric": "val_acc" if task == "nc" else "val_mrr",
            "checkpoint_selection": (
                "best_val_accuracy" if task == "nc" else "best_val_mrr"
            ),
            "inference_protocol": str(task_cfg.get("inference_mode", "full")),
            "protocol_version": str(task_cfg.get("protocol_version", "unknown")),
            "dataset": dataset,
            "task": task,
            "seed": int(seed),
            "split_source": split_source or "unknown",
            "git_branch": _git_value(project_root, "branch", "--show-current"),
            "git_commit": _git_value(project_root, "rev-parse", "HEAD"),
        }
    )
    return manifest


def _git_value(project_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
