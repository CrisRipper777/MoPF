"""Validate the frozen MoPF F2 matrix without launching any training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation import (
    ALL_ABLATIONS,
    ABLATION_SPECS,
    INTERACTION_ABLATIONS,
    MAIN_ABLATIONS,
)


NC_DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
LP_DATASETS = ("sports-copurchase",)
SEEDS = (42, 43, 44)
REQUIRED_MANIFEST_FIELDS = {
    "ablation_name",
    "learned_relation_calibration",
    "semantic_anchor",
    "global_preference",
    "modality_residual",
    "node_residual",
    "tcpr",
    "edge_weight_mode",
    "multihop_state_mode",
    "multihop_response_mode",
    "alpha",
    "K",
    "hidden_dim",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "max_epochs",
    "patience",
    "temperature",
    "batch_size",
    "num_neighbors",
    "negative_sampling",
    "evaluation_metric",
    "checkpoint_selection",
    "inference_protocol",
    "protocol_version",
    "dataset",
    "task",
    "seed",
    "split_source",
    "git_branch",
    "git_commit",
}


@dataclass(frozen=True)
class PlannedRun:
    task: str
    dataset: str
    variant: str
    seed: int
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the frozen MoPF F2 plan; never launches training."
    )
    parser.add_argument("--phase", choices=("main", "interaction", "all"), default="main")
    parser.add_argument("--task", choices=("nc", "lp", "all"), default="all")
    parser.add_argument("--datasets", default="", help="Comma-separated formal dataset filter")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    parser.add_argument("--output-root", default="outputs/f2_ablation")
    return parser.parse_args()


def selected_variants(phase: str) -> tuple[str, ...]:
    if phase == "main":
        return MAIN_ABLATIONS
    if phase == "interaction":
        return INTERACTION_ABLATIONS
    return ALL_ABLATIONS


def print_ablation_audit_table(phase: str) -> None:
    """Print the frozen mechanism table before the run matrix."""
    variants = ("full", *selected_variants(phase))
    fields = (
        "learned_relation_calibration",
        "semantic_anchor",
        "global_preference",
        "modality_residual",
        "node_residual",
        "tcpr",
    )
    print("ablation_audit_table")
    print("variant|" + "|".join(fields))
    for variant in variants:
        spec = ABLATION_SPECS[variant]
        values = ["ON" if getattr(spec, field) else "OFF" for field in fields]
        print(f"{variant}|" + "|".join(values))


def parse_filter(raw: str, *, cast: Any = str) -> tuple[Any, ...]:
    return tuple(cast(value.strip()) for value in raw.split(",") if value.strip())


def build_plan(args: argparse.Namespace, root: Path) -> list[PlannedRun]:
    requested = set(parse_filter(args.datasets)) if args.datasets else None
    seeds = parse_filter(args.seeds, cast=int)
    variants = selected_variants(args.phase)
    plan: list[PlannedRun] = []
    if args.task in {"all", "nc"}:
        datasets = tuple(dataset for dataset in NC_DATASETS if requested is None or dataset in requested)
        for variant in variants:
            for dataset in datasets:
                for seed in seeds:
                    plan.append(
                        PlannedRun("nc", dataset, variant, seed, root / "nc" / dataset / variant / f"seed{seed}")
                    )
    if args.task in {"all", "lp"}:
        datasets = tuple(dataset for dataset in LP_DATASETS if requested is None or dataset in requested)
        for variant in variants:
            for dataset in datasets:
                for seed in seeds:
                    plan.append(
                        PlannedRun("lp", dataset, variant, seed, root / "lp" / dataset / variant / f"seed{seed}")
                    )
    if not plan:
        raise SystemExit("dataset/task filter selected no formal F2 runs")
    return plan


def run_status(run: PlannedRun) -> str:
    path = run.output_dir
    complete = all(
        (path / filename).is_file() and (path / filename).stat().st_size > 0
        for filename in ("complete.marker", "metrics.json", "ablation_manifest.json", "best.pt")
    )
    if complete:
        return "COMPLETE"
    if path.exists() and any(path.iterdir()):
        return "INCOMPLETE"
    return "MISSING"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def config_audit(plan: list[PlannedRun]) -> list[str]:
    issues: list[str] = []
    by_context: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in plan:
        manifest_path = run.output_dir / "ablation_manifest.json"
        if not manifest_path.is_file():
            continue
        payload = load_json(manifest_path)
        if payload is None:
            issues.append(f"invalid manifest: {manifest_path}")
            continue
        missing = REQUIRED_MANIFEST_FIELDS - payload.keys()
        if missing:
            issues.append(f"manifest missing {sorted(missing)}: {manifest_path}")
        identity = (
            payload.get("task"),
            payload.get("dataset"),
            payload.get("ablation_name"),
            payload.get("seed"),
        )
        expected = (run.task, run.dataset, run.variant, run.seed)
        if identity != expected:
            issues.append(
                f"manifest identity mismatch expected={expected} actual={identity}: {manifest_path}"
            )
        by_context[(run.task, run.dataset, run.seed)].append(payload)

    common_fields = (
        "hidden_dim",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "max_epochs",
        "patience",
        "alpha",
        "temperature",
        "batch_size",
        "num_neighbors",
        "negative_sampling",
        "evaluation_metric",
        "checkpoint_selection",
        "inference_protocol",
        "protocol_version",
        "split_source",
        "K",
    )
    for context, payloads in sorted(by_context.items()):
        if len(payloads) < 2:
            continue
        for field in common_fields:
            values = {
                json.dumps(item.get(field), sort_keys=True, default=str)
                for item in payloads
            }
            if len(values) > 1:
                issues.append(
                    f"config inconsistency {context} field={field}: {sorted(values)}"
                )
    return issues


def duplicate_audit(plan: list[PlannedRun], root: Path) -> list[str]:
    issues: list[str] = []
    paths = [run.output_dir.resolve() for run in plan]
    if len(paths) != len(set(paths)):
        issues.append("planned output paths collide")
    identities: Counter[tuple[Any, ...]] = Counter()
    for manifest_path in root.rglob("ablation_manifest.json") if root.exists() else ():
        payload = load_json(manifest_path)
        if payload is None:
            continue
        identity = (
            payload.get("task"),
            payload.get("dataset"),
            payload.get("ablation_name"),
            payload.get("seed"),
        )
        identities[identity] += 1
    for identity, count in identities.items():
        if count > 1:
            issues.append(f"duplicate manifest identity {identity}: {count} copies")
    return issues


def full_reuse_audit(project_root: Path) -> dict[str, Any]:
    paths = []
    for dataset in NC_DATASETS:
        for seed in SEEDS:
            paths.append(
                project_root
                / "outputs"
                / "f1_final_execution"
                / "nc"
                / dataset
                / "mopf"
                / f"seed{seed}"
            )
    for seed in SEEDS:
        paths.append(
            project_root
            / "outputs"
            / "f1_final_execution"
            / "lp_formal"
            / "sports-copurchase"
            / "mopf"
            / f"seed{seed}"
        )
    required = ("metrics.json", "best.pt", "resolved_config.yaml", "complete.marker")
    complete = [
        path for path in paths if all((path / name).is_file() for name in required)
    ]
    summary = load_json(project_root / "outputs" / "f1_final_execution" / "f1b_completion.json")
    config_markers = (
        "edge_weight_mode: learned_diag_cos",
        "multihop_state_mode: anchored",
        "multihop_response_mode: cumulative",
        "multihop_anchor_alpha: 0.1",
        "global_filter_trainable: true",
        "use_modality_residual: true",
        "use_node_residual: true",
        "use_transport_residual: true",
        "hidden_dim: 256",
        "edge_weight_temperature: 0.35",
    )
    config_issues: list[str] = []
    selection_issues: list[str] = []
    split_issues: list[str] = []
    formal_hashes: set[str] = set()
    method_freeze_hashes: set[str] = set()
    for path in complete:
        config_text = (path / "resolved_config.yaml").read_text(encoding="utf-8")
        missing_markers = [marker for marker in config_markers if marker not in config_text]
        if missing_markers:
            config_issues.append(f"{path}: missing {missing_markers}")
        task = "lp" if "lp_formal" in path.parts else "nc"
        if task == "lp":
            split_marker = "edge_split_path:"
        elif "ele-fashion" in path.parts:
            split_marker = "node_split_path:"
        else:
            split_marker = "nc_split_path:"
        if split_marker not in config_text:
            split_issues.append(f"{path}: missing {split_marker}")
        metrics = load_json(path / "metrics.json")
        if metrics is None:
            selection_issues.append(f"{path}: invalid metrics")
            continue
        expected_selection = "val_mrr" if task == "lp" else "val_acc"
        if metrics.get("selection_metric") != expected_selection:
            selection_issues.append(
                f"{path}: selection_metric={metrics.get('selection_metric')!r}, expected={expected_selection!r}"
            )
        if metrics.get("task") != task:
            selection_issues.append(f"{path}: task metadata mismatch")
        for key, target in (
            ("formal_config_sha256", formal_hashes),
            ("method_freeze_sha", method_freeze_hashes),
        ):
            value = metrics.get(key)
            if value:
                target.add(str(value))
    summary_counts = None if summary is None else summary.get("counts")
    summary_complete = bool(
        summary_counts
        and summary_counts.get("completed_runs") == summary_counts.get("expected_runs")
        and all(summary_counts.get(field) == 0 for field in ("failed_runs", "missing_runs", "incomplete_runs", "duplicate_runs"))
    )
    audit_issues = config_issues + selection_issues + split_issues
    return {
        "expected": len(paths),
        "complete_artifact_dirs": len(complete),
        "completion_summary_status": summary_counts,
        "resolved_config_markers": "PASS" if not config_issues else "FAIL",
        "split_config_markers": "PASS" if not split_issues else "FAIL",
        "checkpoint_selection_metadata": "PASS" if not selection_issues else "FAIL",
        "formal_config_sha256_count": len(formal_hashes),
        "method_freeze_sha_count": len(method_freeze_hashes),
        "f1_completion_summary": "PASS" if summary_complete else "FAIL",
        "reuse_candidate": (
            len(complete) == len(paths)
            and summary_complete
            and not audit_issues
            and len(formal_hashes) == 1
            and len(method_freeze_hashes) == 1
        ),
        "issues": audit_issues,
    }


def main() -> int:
    args = parse_args()
    project_root = PROJECT_ROOT
    root = Path(args.output_root)
    if not root.is_absolute():
        root = project_root / root
    plan = build_plan(args, root)
    statuses = Counter(run_status(run) for run in plan)
    config_issues = config_audit(plan)
    duplicate_issues = duplicate_audit(plan, root)
    disk = shutil.disk_usage(root if root.exists() else project_root)

    print(f"phase={args.phase}")
    print(f"task={args.task}")
    print(f"total_planned_runs={len(plan)}")
    print(f"missing_runs={statuses['MISSING']}")
    print(f"existing_complete_runs={statuses['COMPLETE']}")
    print(f"existing_incomplete_runs={statuses['INCOMPLETE']}")
    print(f"output_root={root}")
    print(f"disk_free_bytes={disk.free}")
    print("full_reuse_audit=" + json.dumps(full_reuse_audit(project_root), sort_keys=True))
    print_ablation_audit_table(args.phase)
    print(f"config_consistency={'PASS' if not config_issues else 'FAIL'}")
    for issue in config_issues:
        print(f"CONFIG_ISSUE {issue}")
    print(f"output_collision={'PASS' if not duplicate_issues else 'FAIL'}")
    for issue in duplicate_issues:
        print(f"COLLISION_ISSUE {issue}")
    print("dataset|variant|seed|status|output_dir")
    for run in plan:
        print(
            f"{run.task}|{run.dataset}|{run.variant}|{run.seed}|"
            f"{run_status(run)}|{run.output_dir}"
        )
    return 1 if config_issues or duplicate_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
