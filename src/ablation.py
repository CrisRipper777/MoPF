"""Frozen MoPF ablation definitions and provenance helpers.

The resolver is independent of the model implementation so NC and LP share
exactly the same ablation vocabulary and manifest semantics.  The historical
F2 names remain frozen; the paper-facing Core Story variants use explicit
stage-level overrides and are kept in a separate catalog.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AblationSpec:
    name: str
    # Historical F2 compatibility field.  Do not reinterpret this field:
    # ``wo_learned_semantic_calibration`` means learned_diag_cos -> separate_cos.
    learned_relation_calibration: bool
    semantic_anchor: bool
    global_preference: bool
    modality_residual: bool
    node_residual: bool
    tcpr: bool
    # New Core Story controls.  ``None`` means use the configured Full value;
    # a non-None value is an explicit paper-facing override.
    relation_calibration: bool = True
    edge_weight_override: str | None = None
    composition_mode: str | None = None


_FULL = AblationSpec(
    "full", True, True, True, True, True, True, composition_mode="adaptive"
)

ABLATION_SPECS: dict[str, AblationSpec] = {
    "full": _FULL,
    # SSI-MAG-V3 development switches.  The model consumes the exact name
    # for stage-local interventions; these entries keep the existing Hydra
    # manifest resolver and checkpoint metadata valid without changing any
    # historical MoPF/CoSI-MAG ablation semantics.
    "raw_relation": AblationSpec(
        "raw_relation", True, True, True, True, True, True,
        relation_calibration=False, edge_weight_override="raw_uniform",
    ),
    "fixed_reference": AblationSpec(
        "fixed_reference", True, True, True, True, True, True,
        composition_mode="adaptive",
    ),
    "terminal_context": AblationSpec(
        "terminal_context", True, True, True, True, True, True,
        composition_mode="adaptive",
    ),
    "no_context_change": AblationSpec(
        "no_context_change", True, True, True, True, True, True,
        composition_mode="adaptive",
    ),
    "no_cross_hop_interaction": AblationSpec(
        "no_cross_hop_interaction", True, True, True, True, True, True,
        composition_mode="adaptive",
    ),
    "uniform_context": AblationSpec(
        "uniform_context", True, True, True, True, True, True,
        composition_mode="uniform",
    ),
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
    # Core Story A1 is intentionally not represented by the historical
    # learned_relation_calibration switch.  Its exact degeneration is the
    # raw physical-support path, while the old variant remains separate_cos.
    "wo_relation_calibration": AblationSpec(
        "wo_relation_calibration",
        True,
        True,
        True,
        True,
        True,
        True,
        relation_calibration=False,
        edge_weight_override="raw_uniform",
    ),
    "wo_adaptive_composition": AblationSpec(
        "wo_adaptive_composition",
        True,
        True,
        True,
        True,
        True,
        True,
        relation_calibration=True,
        composition_mode="uniform",
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

CORE_STORY_ABLATIONS = (
    "wo_relation_calibration",
    "wo_semantic_anchor",
    "wo_adaptive_composition",
)


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
    if spec.edge_weight_override is not None:
        return str(spec.edge_weight_override).strip().lower()
    if not spec.learned_relation_calibration and configured == "learned_diag_cos":
        return "separate_cos"
    return configured


def effective_composition_mode(model_cfg: Any, spec: AblationSpec) -> str:
    """Resolve adaptive/uniform final response-bank composition."""
    configured = str(model_cfg.get("composition_mode", "adaptive")).strip().lower()
    mode = spec.composition_mode or configured
    if mode not in {"adaptive", "uniform"}:
        raise ValueError(
            "model.composition_mode must be adaptive|uniform, "
            f"got {mode!r}"
        )
    return mode


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
    """Build the human- and machine-readable per-run manifest."""
    spec = cfg_ablation(cfg)
    model_cfg = cfg.model
    task_cfg = cfg.task
    state_mode, response_mode = effective_multihop_modes(model_cfg, spec)
    edge_weight_mode = effective_edge_weight_mode(model_cfg, spec)
    composition_mode = effective_composition_mode(model_cfg, spec)
    configured_relation_calibration = str(
        model_cfg.get("edge_weight_mode", "separate_cos")
    ).strip().lower() != "raw_uniform"
    configured_global_preference = bool(
        model_cfg.get("global_filter_trainable", True)
    )
    configured_modality_residual = bool(
        model_cfg.get("use_modality_residual", True)
    )
    configured_node_residual = bool(model_cfg.get("use_node_residual", True))
    configured_relation_conditioned = bool(
        model_cfg.get("use_transport_residual", False)
    )
    uses_adaptive_composition = composition_mode == "adaptive"
    uses_raw_uniform_edges = edge_weight_mode == "raw_uniform"
    effective_relation_calibration = bool(
        spec.relation_calibration
        and configured_relation_calibration
        and not uses_raw_uniform_edges
    )
    effective_global_preference = bool(
        spec.global_preference
        and configured_global_preference
        and uses_adaptive_composition
    )
    effective_modality_residual = bool(
        spec.modality_residual
        and configured_modality_residual
        and uses_adaptive_composition
    )
    effective_node_residual = bool(
        spec.node_residual
        and configured_node_residual
        and uses_adaptive_composition
    )
    effective_relation_conditioned = bool(
        spec.tcpr
        and configured_relation_conditioned
        and not uses_raw_uniform_edges
        and uses_adaptive_composition
    )
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
            "edge_weight_mode": edge_weight_mode,
            "configured_edge_weight_mode": str(
                model_cfg.get("edge_weight_mode", "separate_cos")
            ).strip().lower(),
            "edge_weight_override": spec.edge_weight_override,
            "composition_mode": composition_mode,
            "configured_composition_mode": str(
                model_cfg.get("composition_mode", "adaptive")
            ).strip().lower(),
            "multihop_state_mode": state_mode,
            "multihop_response_mode": response_mode,
            "semantic_anchor_active": bool(spec.semantic_anchor),
            "semantic_anchor_effective": state_mode == "anchored",
            "relation_calibration_active": bool(spec.relation_calibration),
            "relation_calibration_effective": effective_relation_calibration,
            "global_preference_active": bool(spec.global_preference),
            "global_preference_effective": effective_global_preference,
            "modality_residual_active": bool(spec.modality_residual),
            "modality_residual_effective": effective_modality_residual,
            "node_residual_active": bool(spec.node_residual),
            "node_residual_effective": effective_node_residual,
            "relation_conditioned_refinement_active": bool(spec.tcpr),
            "relation_conditioned_refinement_effective": effective_relation_conditioned,
            "alpha": float(model_cfg.get("multihop_anchor_alpha", 0.1)),
            "K": int(model_cfg.get("max_order", 3)),
            "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
            "dropout": float(model_cfg.get("dropout", 0.2)),
            "learning_rate": float(model_cfg.get("lr", task_cfg.get("lr"))),
            "lr": float(model_cfg.get("lr", task_cfg.get("lr"))),
            "weight_decay": float(model_cfg.get("weight_decay", task_cfg.get("weight_decay"))),
            "optimizer": str(task_cfg.get("optimizer", "adamw")),
            "max_epochs": int(task_cfg.get("epochs")),
            "epochs": int(task_cfg.get("epochs")),
            "patience": int(task_cfg.get("patience")),
            "temperature": float(model_cfg.get("edge_weight_temperature", 2.0)),
            "relation_temperature": float(
                model_cfg.get("edge_weight_temperature", 2.0)
            ),
            "filter_rank": int(model_cfg.get("filter_rank", 4)),
            "batch_size": int(task_cfg.get("batch_size", 0)),
            "num_neighbors": raw_neighbors,
            "negative_sampling": {
                "task_num_train_neg": task_cfg.get("num_train_neg", None),
                "dataset_lp_num_neg": cfg.dataset.get("lp_num_neg", None),
            },
            "sampling_config": {
                "training_mode": str(task_cfg.get("training_mode", "unknown")),
                "batch_size": int(task_cfg.get("batch_size", 0)),
                "num_neighbors": raw_neighbors,
                "subgraph_type": task_cfg.get("subgraph_type", None),
                "num_train_neg": task_cfg.get("num_train_neg", None),
                "train_pos_per_epoch": task_cfg.get("train_pos_per_epoch", None),
                "positive_edge_mask_backend": task_cfg.get(
                    "positive_edge_mask_backend", None
                ),
                "loader_num_workers": task_cfg.get("loader_num_workers", None),
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
