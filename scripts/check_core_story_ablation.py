"""Audit the paper-facing Core Story matrix without launching training."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation import ABLATION_SPECS, CORE_STORY_ABLATIONS
from src.formal_protocol import (
    CORE_STORY_PROTOCOL_VERSION,
    FORMAL_K,
    FORMAL_SEEDS,
    LP_DATASETS,
    NC_DATASETS,
    expected_metric_keys,
    formal_k,
)

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - Hydra already supplies this in normal use.
    OmegaConf = None


REQUIRED_ARTIFACTS = (
    "complete.marker",
    "metrics.json",
    "ablation_manifest.json",
    "resolved_config.yaml",
    "resolved_config.json",
    "train.log",
    "best.pt",
)
REQUIRED_MANIFEST_FIELDS = {
    "ablation_name",
    "edge_weight_mode",
    "composition_mode",
    "semantic_anchor",
    "relation_calibration",
    "global_preference",
    "modality_residual",
    "node_residual",
    "tcpr",
    "semantic_anchor_active",
    "semantic_anchor_effective",
    "relation_calibration_active",
    "relation_calibration_effective",
    "global_preference_active",
    "global_preference_effective",
    "modality_residual_active",
    "modality_residual_effective",
    "node_residual_active",
    "node_residual_effective",
    "relation_conditioned_refinement_active",
    "relation_conditioned_refinement_effective",
    "multihop_state_mode",
    "multihop_response_mode",
    "alpha",
    "K",
    "hidden_dim",
    "dropout",
    "optimizer",
    "learning_rate",
    "lr",
    "weight_decay",
    "epochs",
    "max_epochs",
    "patience",
    "batch_size",
    "num_neighbors",
    "sampling_config",
    "negative_sampling",
    "split_source",
    "checkpoint_selection",
    "protocol_version",
    "git_branch",
    "git_commit",
    "dataset",
    "task",
    "seed",
}


@dataclass(frozen=True)
class PlannedRun:
    task: str
    dataset: str
    variant: str
    seed: int
    output_dir: Path

    @property
    def identity(self) -> tuple[str, str, str, int]:
        return self.task, self.dataset, self.variant, self.seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the Core Story matrix; never launches training."
    )
    parser.add_argument("--task", choices=("nc", "lp", "all"), default="all")
    parser.add_argument("--datasets", default="", help="Comma-separated formal dataset filter")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    parser.add_argument(
        "--variants",
        default=",".join(CORE_STORY_ABLATIONS),
        help="Comma-separated Core Story variants; Full is rejected",
    )
    parser.add_argument("--output-root", default="outputs/core_story_ablation")
    parser.add_argument(
        "--expected-git-commit",
        default="",
        help="Optional required SHA; otherwise all runs must match current HEAD",
    )
    return parser.parse_args()


def parse_csv(raw: str, cast=str) -> tuple[Any, ...]:
    values = tuple(cast(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("comma-separated option cannot be empty")
    return values


def _selected(raw: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    requested = set(parse_csv(raw)) if raw else set(allowed)
    unknown = sorted(requested.difference(allowed))
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return tuple(item for item in allowed if item in requested)


def build_plan(args: argparse.Namespace, root: Path) -> list[PlannedRun]:
    variants = _selected(args.variants, CORE_STORY_ABLATIONS, "Core Story variant")
    if "full" in variants:
        raise ValueError("Full is not part of the Core Story checker plan")
    seeds = parse_csv(args.seeds, cast=int)
    unknown_seed = [seed for seed in seeds if seed < 0]
    if unknown_seed:
        raise ValueError(f"invalid seeds: {unknown_seed}")
    all_datasets = NC_DATASETS + LP_DATASETS
    requested = set(parse_csv(args.datasets)) if args.datasets else set(all_datasets)
    unknown_datasets = sorted(requested.difference(all_datasets))
    if unknown_datasets:
        raise ValueError(f"unknown formal dataset: {unknown_datasets}")
    selected_nc = tuple(item for item in NC_DATASETS if item in requested)
    selected_lp = tuple(item for item in LP_DATASETS if item in requested)
    if args.task == "nc" and not selected_nc:
        raise ValueError("dataset filter selected no formal NC dataset")
    if args.task == "lp" and not selected_lp:
        raise ValueError("dataset filter selected no formal LP dataset")
    plan: list[PlannedRun] = []
    if args.task in {"nc", "all"}:
        for dataset in selected_nc:
            for variant in variants:
                for seed in seeds:
                    plan.append(
                        PlannedRun(
                            "nc",
                            dataset,
                            variant,
                            seed,
                            root / "nc" / dataset / variant / f"seed{seed}",
                        )
                    )
    if args.task in {"lp", "all"}:
        for dataset in selected_lp:
            for variant in variants:
                for seed in seeds:
                    plan.append(
                        PlannedRun(
                            "lp",
                            dataset,
                            variant,
                            seed,
                            root / "lp" / dataset / variant / f"seed{seed}",
                        )
                    )
    if not plan:
        raise ValueError("dataset/task filter selected no formal Core Story runs")
    return plan


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return True


def run_status(run: PlannedRun) -> str:
    path = run.output_dir
    if all((path / name).is_file() and (path / name).stat().st_size > 0 for name in REQUIRED_ARTIFACTS):
        return "COMPLETE"
    if path.exists() and any(path.iterdir()):
        return "INCOMPLETE"
    return "MISSING"


def _expected_manifest(run: PlannedRun) -> dict[str, Any]:
    spec = ABLATION_SPECS[run.variant]
    anchored = run.variant != "wo_semantic_anchor"
    uniform = run.variant == "wo_adaptive_composition"
    raw_uniform = run.variant == "wo_relation_calibration"
    return {
        "ablation_name": run.variant,
        "task": run.task,
        "dataset": run.dataset,
        "seed": run.seed,
        "edge_weight_mode": "raw_uniform" if raw_uniform else "learned_diag_cos",
        "composition_mode": "uniform" if uniform else "adaptive",
        "semantic_anchor": spec.semantic_anchor,
        "relation_calibration": not raw_uniform,
        "global_preference": True,
        "modality_residual": True,
        "node_residual": True,
        "tcpr": True,
        "semantic_anchor_active": anchored,
        "semantic_anchor_effective": anchored,
        "relation_calibration_active": not raw_uniform,
        "relation_calibration_effective": not raw_uniform,
        "global_preference_active": True,
        "global_preference_effective": not uniform,
        "modality_residual_active": True,
        "modality_residual_effective": not uniform,
        "node_residual_active": True,
        "node_residual_effective": not uniform,
        "relation_conditioned_refinement_active": True,
        "relation_conditioned_refinement_effective": not raw_uniform and not uniform,
        "multihop_state_mode": "anchored" if anchored else "ordinary",
        "multihop_response_mode": "cumulative",
        "alpha": 0.1,
        "K": formal_k(run.dataset),
        "hidden_dim": 256,
        "dropout": 0.2,
        "optimizer": "adamw" if run.task == "nc" else "adam",
        "learning_rate": 1e-3,
        "lr": 1e-3,
        "weight_decay": 1e-4 if run.task == "nc" else 1e-5,
        "epochs": 300 if run.task == "nc" else 150,
        "max_epochs": 300 if run.task == "nc" else 150,
        "patience": 30 if run.task == "nc" else 10,
        "batch_size": 1024 if run.task == "nc" else 2048,
        "protocol_version": (
            "unified_full_graph_nc_v1" if run.task == "nc" else "unified_sampled_lp_v1"
        ),
        "checkpoint_selection": (
            "best_val_accuracy" if run.task == "nc" else "best_val_mrr"
        ),
    }


def audit_manifest(run: PlannedRun, payload: dict[str, Any] | None) -> list[str]:
    issues: list[str] = []
    if payload is None:
        return [f"invalid or missing manifest: {run.output_dir / 'ablation_manifest.json'}"]
    missing = sorted(REQUIRED_MANIFEST_FIELDS.difference(payload))
    if missing:
        issues.append(f"manifest missing fields {missing}")
    expected = _expected_manifest(run)
    for field, value in expected.items():
        actual = payload.get(field)
        if field in {"alpha", "learning_rate", "lr", "weight_decay", "dropout"}:
            if actual is None or not math.isclose(float(actual), float(value), rel_tol=0.0, abs_tol=1e-9):
                issues.append(f"manifest mismatch {field}: expected {value!r}, got {actual!r}")
        elif actual != value:
            issues.append(f"manifest mismatch {field}: expected {value!r}, got {actual!r}")
    if payload.get("split_source") in {None, "", "unknown"}:
        issues.append("manifest split_source is missing/unknown")
    if payload.get("git_branch") in {None, "", "unknown"}:
        issues.append("manifest git_branch is missing/unknown")
    if payload.get("git_commit") in {None, "", "unknown"}:
        issues.append("manifest git_commit is missing/unknown")
    return issues


def _resolved_config(path: Path) -> dict[str, Any] | None:
    try:
        if OmegaConf is not None:
            payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        else:
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def audit_config(run: PlannedRun, payload: dict[str, Any] | None) -> list[str]:
    issues: list[str] = []
    if payload is None:
        return [f"invalid or missing resolved config: {run.output_dir / 'resolved_config.yaml'}"]
    dataset_cfg = payload.get("dataset", {})
    task_cfg = payload.get("task", {})
    model_cfg = payload.get("model", {})
    expected = _expected_manifest(run)
    checks = {
        "dataset.name": (dataset_cfg.get("name"), run.dataset),
        "task.name": (task_cfg.get("name"), run.task),
        "model.name": (model_cfg.get("name"), "mopf"),
        "model.max_order": (model_cfg.get("max_order"), expected["K"]),
        "model.num_layers": (model_cfg.get("num_layers"), expected["K"]),
        "model.hidden_dim": (model_cfg.get("hidden_dim"), 256),
        "model.dropout": (model_cfg.get("dropout"), 0.2),
        "model.edge_weight_mode": (model_cfg.get("edge_weight_mode"), "learned_diag_cos"),
        "model.composition_mode": (model_cfg.get("composition_mode"), "adaptive"),
        "model.multihop_state_mode": (model_cfg.get("multihop_state_mode"), "anchored"),
        "model.multihop_response_mode": (model_cfg.get("multihop_response_mode"), "cumulative"),
        "model.multihop_anchor_alpha": (model_cfg.get("multihop_anchor_alpha"), 0.1),
        "model.edge_weight_temperature": (model_cfg.get("edge_weight_temperature"), 0.35),
        "model.edge_weight_min": (model_cfg.get("edge_weight_min"), 0.1),
        "model.num_metric_perspectives": (model_cfg.get("num_metric_perspectives"), 4),
        "model.filter_rank": (model_cfg.get("filter_rank"), 4),
        "model.global_filter_trainable": (model_cfg.get("global_filter_trainable"), True),
        "model.use_transport_residual": (model_cfg.get("use_transport_residual"), True),
        "model.use_modality_residual": (model_cfg.get("use_modality_residual"), True),
        "model.use_node_residual": (model_cfg.get("use_node_residual"), True),
        "model.hrc_weight": (model_cfg.get("hrc_weight"), 0.0),
        # MoPF runtime semantics use cfg.model.get("ppc_weight", 0.0).
        # A missing key is therefore equivalent to an explicit zero default.
        "model.ppc_weight": (model_cfg.get("ppc_weight", 0.0), 0.0),
        "model.fusion_mode": (model_cfg.get("fusion_mode"), "concat_residual_mlp"),
        "task.protocol_version": (task_cfg.get("protocol_version"), expected["protocol_version"]),
        "task.optimizer": (task_cfg.get("optimizer"), expected["optimizer"]),
        "task.epochs": (task_cfg.get("epochs"), expected["epochs"]),
        "task.patience": (task_cfg.get("patience"), expected["patience"]),
        "task.batch_size": (task_cfg.get("batch_size"), expected["batch_size"]),
        "task.inference_mode": (task_cfg.get("inference_mode"), "full"),
        "task.inference_batch_size": (task_cfg.get("inference_batch_size"), 4096),
        "task.grad_clip": (task_cfg.get("grad_clip"), 1.0),
        "task.early_stop_min_epoch": (
            task_cfg.get("early_stop_min_epoch"),
            30 if run.task == "nc" else 20,
        ),
        "task.early_stop_min_delta": (task_cfg.get("early_stop_min_delta"), 1e-4),
        "task.eval_every": (task_cfg.get("eval_every"), 1 if run.task == "nc" else 2),
        "task.scheduler": (task_cfg.get("scheduler"), None),
    }
    for field, (actual, value) in checks.items():
        if isinstance(value, float):
            if actual is None or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-9):
                issues.append(f"config mismatch {field}: expected {value!r}, got {actual!r}")
        elif actual != value:
            issues.append(f"config mismatch {field}: expected {value!r}, got {actual!r}")
    split_key = "edge_split_path" if run.task == "lp" else (
        "node_split_path" if run.dataset == "ele-fashion" else "nc_split_path"
    )
    if not dataset_cfg.get(split_key):
        issues.append(f"config mismatch dataset.{split_key}: split source is absent")
    if run.task == "nc":
        if task_cfg.get("training_mode") != "full_graph":
            issues.append("config mismatch task.training_mode: expected 'full_graph'")
        if task_cfg.get("num_neighbors") != 15:
            issues.append(
                f"config mismatch task.num_neighbors: expected 15, got {task_cfg.get('num_neighbors')!r}"
            )
        if task_cfg.get("loss", {}).get("aux_weight") != 1.0:
            issues.append("config mismatch task.loss.aux_weight: expected 1.0")
    else:
        if task_cfg.get("training_mode") != "sampled":
            issues.append("config mismatch task.training_mode: expected 'sampled'")
        if task_cfg.get("num_neighbors") != [5, 5]:
            issues.append(
                f"config mismatch task.num_neighbors: expected [5, 5], got {task_cfg.get('num_neighbors')!r}"
            )
        for field, value in {
            "num_train_neg": 1,
            "subgraph_type": "bidirectional",
            "positive_edge_mask_backend": "global_eid",
            "loader_num_workers": 4,
            "eval_preload_node_emb": True,
            "eval_edge_batch_size": 512,
        }.items():
            if task_cfg.get(field) != value:
                issues.append(
                    f"config mismatch task.{field}: expected {value!r}, got {task_cfg.get(field)!r}"
                )
        if task_cfg.get("loss", {}).get("aux_weight") != 1.0:
            issues.append("config mismatch task.loss.aux_weight: expected 1.0")
        decoder = task_cfg.get("decoder", {})
        for field, value in {
            "hidden_dim": 256,
            "num_layers": 3,
            "dropout": 0.02,
            "proj_dim": 128,
        }.items():
            actual = decoder.get(field)
            if isinstance(value, float):
                matches = actual is not None and math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-9)
            else:
                matches = actual == value
            if not matches:
                issues.append(
                    f"config mismatch task.decoder.{field}: expected {value!r}, got {actual!r}"
                )
    return issues


def audit_metrics(run: PlannedRun, payload: dict[str, Any] | None) -> list[str]:
    issues: list[str] = []
    if payload is None:
        return [f"invalid or missing metrics: {run.output_dir / 'metrics.json'}"]
    expected_keys = set(expected_metric_keys(run.task))
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return ["metric schema missing metrics object"]
    if set(metrics) != expected_keys:
        issues.append(
            f"metric schema mismatch: expected {sorted(expected_keys)}, got {sorted(metrics)}"
        )
    if not finite_json(payload):
        issues.append("metric payload contains a non-finite value")
    for key in expected_keys.intersection(metrics):
        value = metrics[key]
        if not isinstance(value, dict) or set(value) != {"mean", "std"}:
            issues.append(f"metric schema invalid for {key}: expected mean/std object")
        elif not all(
            isinstance(value[field], (int, float)) and math.isfinite(float(value[field]))
            for field in ("mean", "std")
        ):
            issues.append(f"metric schema non-finite/non-numeric for {key}")
    for field, value in {
        "task": run.task,
        "dataset": run.dataset,
        "model": "mopf",
        "ablation": run.variant,
        "seed": run.seed,
        "selection_metric": "val_acc" if run.task == "nc" else "val_mrr",
        "checkpoint_selection": "best_val_accuracy" if run.task == "nc" else "best_val_mrr",
    }.items():
        if payload.get(field) != value:
            issues.append(f"metric identity mismatch {field}: expected {value!r}, got {payload.get(field)!r}")
    if not isinstance(payload.get("runtime_seconds"), (int, float)) or float(payload["runtime_seconds"]) < 0:
        issues.append("metric runtime_seconds is invalid")
    checkpoint = payload.get("checkpoint")
    if not checkpoint or Path(str(checkpoint)).name != "best.pt":
        issues.append("metric checkpoint reference does not point to best.pt")
    elif not Path(str(checkpoint)).is_file():
        issues.append(f"metric checkpoint path does not exist: {checkpoint}")
    return issues


def audit_duplicates(plan: list[PlannedRun], root: Path) -> list[str]:
    issues: list[str] = []
    paths = [run.output_dir.resolve() for run in plan]
    if len(paths) != len(set(paths)):
        issues.append("planned output paths collide")
    identities: defaultdict[tuple[str, str, str, int], list[Path]] = defaultdict(list)
    if root.exists():
        for manifest_path in root.rglob("ablation_manifest.json"):
            payload = load_json(manifest_path)
            if payload is None:
                continue
            try:
                identity = (
                    str(payload.get("task")),
                    str(payload.get("dataset")),
                    str(payload.get("ablation_name")),
                    int(payload.get("seed", -1)),
                )
            except (TypeError, ValueError):
                issues.append(f"invalid manifest identity: {manifest_path}")
                continue
            identities[identity].append(manifest_path.parent.resolve())
    planned_identities = {run.identity for run in plan}
    for identity, paths_for_identity in sorted(identities.items()):
        if len(paths_for_identity) > 1:
            issues.append(f"duplicate manifest identity {identity}: {paths_for_identity}")
        if identity not in planned_identities:
            issues.append(f"unplanned manifest identity found: {identity}")
    return issues


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def audit_commits(plan: list[PlannedRun], manifests: dict[Path, dict[str, Any] | None], expected: str) -> list[str]:
    issues: list[str] = []
    commits = {
        str(payload.get("git_commit"))
        for run in plan
        for payload in [manifests.get(run.output_dir)]
        if payload is not None and payload.get("git_commit") not in {None, "", "unknown"}
    }
    if len(commits) > 1:
        issues.append(f"git-commit mismatch: multiple run SHAs {sorted(commits)}")
    target = expected or current_commit()
    if target != "unknown":
        for run in plan:
            payload = manifests.get(run.output_dir)
            if payload is not None and payload.get("git_commit") not in {None, "", "unknown", target}:
                issues.append(
                    f"git-commit mismatch {run.output_dir}: expected {target}, got {payload.get('git_commit')}"
                )
    return issues


def main() -> int:
    args = parse_args()
    root = Path(args.output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    try:
        plan = build_plan(args, root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    statuses = Counter(run_status(run) for run in plan)
    manifests: dict[Path, dict[str, Any] | None] = {}
    all_issues: list[tuple[str, Path, str]] = []
    for run in plan:
        if run_status(run) == "MISSING":
            continue
        manifest = load_json(run.output_dir / "ablation_manifest.json")
        manifests[run.output_dir] = manifest
        for issue in audit_manifest(run, manifest):
            all_issues.append(("MANIFEST", run.output_dir, issue))
        config = _resolved_config(run.output_dir / "resolved_config.yaml")
        for issue in audit_config(run, config):
            all_issues.append(("CONFIG", run.output_dir, issue))
        metrics = load_json(run.output_dir / "metrics.json")
        for issue in audit_metrics(run, metrics):
            all_issues.append(("METRIC", run.output_dir, issue))

    duplicate_issues = audit_duplicates(plan, root)
    for issue in duplicate_issues:
        all_issues.append(("COLLISION", root, issue))
    for issue in audit_commits(plan, manifests, args.expected_git_commit):
        all_issues.append(("GIT", root, issue))

    print(f"protocol={CORE_STORY_PROTOCOL_VERSION}")
    print(f"output_root={root}")
    print(f"planned_runs={len(plan)}")
    print(f"complete_runs={statuses['COMPLETE']}")
    print(f"incomplete_runs={statuses['INCOMPLETE']}")
    print(f"missing_runs={statuses['MISSING']}")
    print(f"expected_nc_runs={len(NC_DATASETS) * len(CORE_STORY_ABLATIONS) * len(FORMAL_SEEDS)}")
    print(f"expected_lp_runs={len(LP_DATASETS) * len(CORE_STORY_ABLATIONS) * len(FORMAL_SEEDS)}")
    print(f"expected_total_runs={len((NC_DATASETS + LP_DATASETS)) * len(CORE_STORY_ABLATIONS) * len(FORMAL_SEEDS)}")
    print(f"manifest_mismatch={'FAIL' if any(kind == 'MANIFEST' for kind, _, _ in all_issues) else 'PASS'}")
    print(f"config_mismatch={'FAIL' if any(kind == 'CONFIG' for kind, _, _ in all_issues) else 'PASS'}")
    print(f"git_commit_mismatch={'FAIL' if any(kind == 'GIT' for kind, _, _ in all_issues) else 'PASS'}")
    print(f"metric_schema={'FAIL' if any(kind == 'METRIC' for kind, _, _ in all_issues) else 'PASS'}")
    print(f"duplicate_collision={'FAIL' if duplicate_issues else 'PASS'}")
    for run in plan:
        print(f"{run.task}|{run.dataset}|{run.variant}|{run.seed}|{run_status(run)}|{run.output_dir}")
    for kind, path, issue in all_issues:
        print(f"{kind}_ISSUE {path}: {issue}")
    return 1 if statuses["INCOMPLETE"] or statuses["MISSING"] or all_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
