#!/usr/bin/env python3
"""Audit and assemble Table 2 from CoSI-MAG and its three framework ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "no_mrc": "w/o MRC (Uniform Relations)",
    "no_semantic_anchor": "w/o Semantic Anchor",
    "no_rcmi": "w/o RCMI (Uniform Composition)",
}
DATASET_TASK = {
    "Movies": "nc",
    "Toys": "nc",
    "Grocery": "nc",
    "ele-fashion": "nc",
    "Reddit-S": "nc",
    "sports-copurchase": "lp",
}
TASK_PROTOCOLS = {
    "nc": "unified_full_graph_nc_v1",
    "lp": "unified_sampled_lp_v1",
}
METRICS = {
    "Movies": [("Acc", "test_acc"), ("Macro-F1", "test_macro_f1")],
    "Toys": [("Acc", "test_acc"), ("Macro-F1", "test_macro_f1")],
    "Grocery": [("Acc", "test_acc"), ("Macro-F1", "test_macro_f1")],
    "ele-fashion": [("Acc", "test_acc"), ("Macro-F1", "test_macro_f1")],
    "Reddit-S": [("Acc", "test_acc"), ("Macro-F1", "test_macro_f1")],
    "sports-copurchase": [("MRR", "test_mrr"), ("H@10", "test_hits@10")],
}
RUN_SEEDS = (42, 43, 44)
NUM_RUNS = 3
BASE_SEED = 42
NONFINITE_RE = re.compile(
    r"\b(?:Train Loss|Val [A-Za-z@0-9_-]+)\s+(?:nan|[+-]?inf(?:inity)?)\b",
    re.IGNORECASE,
)


class AuditError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-9)


def expected_alpha(variant: str | None) -> float:
    return 0.0 if variant == "no_semantic_anchor" else 0.1


def unit_path(root: Path, dataset: str, variant: str | None = None) -> Path:
    task = DATASET_TASK[dataset]
    parts = [root]
    if variant is not None:
        parts.append(Path(variant))
    parts.extend((Path(task), Path(dataset), Path("runs_42_43_44")))
    return Path(*parts)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def check_top_manifest(path: Path, task: str, model: str, variant: str | None) -> None:
    manifest = read_json(path)
    require(manifest.get("model") == model, f"{path}: wrong model")
    require(manifest.get("task") == task, f"{path}: wrong task")
    require(manifest.get("base_seed") == BASE_SEED, f"{path}: base_seed must be 42")
    require(manifest.get("split_seed") == BASE_SEED, f"{path}: split_seed must be 42")
    require(manifest.get("num_runs") == NUM_RUNS, f"{path}: num_runs must be 3")
    require(manifest.get("run_seeds") == list(RUN_SEEDS), f"{path}: run seeds mismatch")
    require(
        manifest.get("frozen_task_protocols", manifest.get("task_protocols", {})).get(task)
        == TASK_PROTOCOLS[task],
        f"{path}: task protocol mismatch",
    )
    if variant is None:
        require(manifest.get("ablation") == "full", f"{path}: clean benchmark is not full")
    else:
        require(variant in manifest.get("variants", []), f"{path}: missing variant {variant}")


def check_ablation_root_manifest(path: Path) -> None:
    manifest = read_json(path)
    require(manifest.get("protocol") == "cosi_mag_final_framework_ablation_v1", f"{path}: protocol mismatch")
    require(manifest.get("model") == "cosi_mag_ablation", f"{path}: model mismatch")
    require(set(manifest.get("variants", [])) == set(VARIANTS), f"{path}: variants mismatch")
    require(set(manifest.get("datasets", [])) == set(DATASET_TASK), f"{path}: datasets mismatch")
    require(manifest.get("base_seed") == BASE_SEED, f"{path}: base seed mismatch")
    require(manifest.get("split_seed") == BASE_SEED, f"{path}: split seed mismatch")
    require(manifest.get("num_runs") == NUM_RUNS, f"{path}: num_runs mismatch")
    require(manifest.get("run_seeds") == list(RUN_SEEDS), f"{path}: run seeds mismatch")
    require(manifest.get("task_protocols") == TASK_PROTOCOLS, f"{path}: task protocols mismatch")
    expected_units = {
        (variant, dataset)
        for variant in VARIANTS
        for dataset in DATASET_TASK
    }
    found_units = {
        (item.get("variant"), item.get("dataset"))
        for item in manifest.get("units", [])
    }
    require(found_units == expected_units, f"{path}: expected exactly nine dataset-variant units")


def check_unit_manifest(
    path: Path,
    *,
    model: str,
    dataset: str,
    variant: str | None,
) -> dict[str, Any]:
    manifest = read_json(path)
    task = DATASET_TASK[dataset]
    require(manifest.get("model") == model, f"{path}: model mismatch")
    require(manifest.get("dataset") == dataset, f"{path}: dataset mismatch")
    require(manifest.get("task") == task, f"{path}: task mismatch")
    require(manifest.get("base_seed") == BASE_SEED, f"{path}: base seed mismatch")
    require(manifest.get("split_seed") == BASE_SEED, f"{path}: split seed mismatch")
    require(manifest.get("num_runs") == NUM_RUNS, f"{path}: num_runs mismatch")
    require(manifest.get("run_seeds") == list(RUN_SEEDS), f"{path}: run seeds mismatch")
    if variant is not None:
        require(manifest.get("ablation_mode") == variant, f"{path}: ablation mode mismatch")
        require(
            close(manifest.get("multihop_anchor_alpha"), expected_alpha(variant)),
            f"{path}: alpha mismatch for {variant}",
        )
    else:
        require(manifest.get("ablation") == "full", f"{path}: expected full model")
    return manifest


def load_pair_configs(
    full_dir: Path, variant_dir: Path, dataset: str, variant: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    full_cfg = read_json(full_dir / "resolved_config.json")
    variant_cfg = read_json(variant_dir / "resolved_config.json")
    task = DATASET_TASK[dataset]
    for name, cfg, expected_model, alpha in (
        ("full", full_cfg, "cosi_mag_final", 0.1),
        (variant, variant_cfg, "cosi_mag_ablation", expected_alpha(variant)),
    ):
        require(cfg.get("model", {}).get("name") == expected_model, f"{name}: model name mismatch")
        require(cfg.get("task", {}).get("name") == task, f"{name}: task mismatch")
        require(
            cfg.get("task", {}).get("protocol_version") == TASK_PROTOCOLS[task],
            f"{name}: task protocol mismatch",
        )
        require(cfg.get("dataset", {}).get("name") == dataset, f"{name}: dataset mismatch")
        require(cfg.get("seed") == BASE_SEED, f"{name}: base/split seed must be 42")
        require(cfg.get("num_runs") == NUM_RUNS, f"{name}: num_runs must be 3")
        require(cfg.get("ablation") == "full", f"{name}: top-level training protocol must be full")
        require(close(cfg.get("model", {}).get("multihop_anchor_alpha"), alpha), f"{name}: model alpha mismatch")
        require(cfg.get("model", {}).get("max_order") == 3, f"{name}: K must be 3")
        require(cfg.get("model", {}).get("num_layers") == 3, f"{name}: encoder depth must be 3")
    require(variant_cfg["model"].get("ablation_mode") == variant, "ablation mode mismatch in resolved config")

    full_task_cfg = dict(full_cfg.get("task", {}))
    variant_task_cfg = dict(variant_cfg.get("task", {}))
    full_task_cfg.pop("save_ckpt_path", None)
    variant_task_cfg.pop("save_ckpt_path", None)
    require(full_task_cfg == variant_task_cfg, "task protocol config differs from the clean benchmark")
    require(
        full_cfg.get("dataset") == variant_cfg.get("dataset"),
        "dataset config or split paths differ from the clean benchmark",
    )

    variable_model_fields = {"name", "version", "ablation_mode", "multihop_anchor_alpha"}
    full_model = {
        key: value
        for key, value in full_cfg.get("model", {}).items()
        if key not in variable_model_fields
    }
    variant_model = {
        key: value
        for key, value in variant_cfg.get("model", {}).items()
        if key not in variable_model_fields
    }
    require(full_model == variant_model, "frozen model hyperparameters differ")

    if task == "nc":
        split_key = (
            "nc_split_path"
            if "nc_split_path" in full_cfg.get("dataset", {})
            else "node_split_path"
        )
    else:
        split_key = "edge_split_path"
    split_path_text = full_cfg.get("dataset", {}).get(split_key)
    require(bool(split_path_text), f"resolved dataset config is missing {split_key}")
    split_path = Path(str(split_path_text))
    require(split_path.is_file(), f"split file does not exist: {split_path}")
    split_hash = sha256_file(split_path)

    full_ablation = read_json(full_dir / "ablation_manifest.json")
    variant_ablation = read_json(variant_dir / "ablation_manifest.json")
    full_source = full_ablation.get("split_source")
    variant_source = variant_ablation.get("split_source")
    require(full_source == variant_source, "ablation manifests name different split sources")
    require(Path(str(full_source)) == split_path, "manifest split source differs from resolved config")
    return full_cfg, variant_cfg, {
        "split_key": split_key,
        "split_path": str(split_path),
        "split_sha256": split_hash,
        "task_protocol": TASK_PROTOCOLS[task],
        "base_seed": BASE_SEED,
        "num_runs": NUM_RUNS,
        "full_alpha": 0.1,
        "ablation_alpha": expected_alpha(variant),
    }


def load_seed_metrics(
    output_dir: Path,
    *,
    dataset: str,
    model: str,
    variant: str | None,
) -> tuple[dict[int, dict[str, float]], dict[str, tuple[float, float]]]:
    import torch

    task = DATASET_TASK[dataset]
    by_seed: dict[int, dict[str, float]] = {}
    for index, seed in enumerate(RUN_SEEDS, 1):
        checkpoint_path = output_dir / f"best_run{index}.pt"
        require(checkpoint_path.is_file(), f"missing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(checkpoint.get("task") == task, f"{checkpoint_path}: task mismatch")
        require(checkpoint.get("seed") == seed, f"{checkpoint_path}: seed mismatch")
        require(bool(checkpoint.get("model_state")), f"{checkpoint_path}: missing model state")
        metrics = checkpoint.get("metrics")
        require(isinstance(metrics, dict), f"{checkpoint_path}: missing per-run metrics")
        wanted = dict(METRICS[dataset])
        values: dict[str, float] = {}
        for metric_name in wanted.values():
            require(metric_name in metrics, f"{checkpoint_path}: missing {metric_name}")
            value = float(metrics[metric_name])
            require(math.isfinite(value), f"{checkpoint_path}: non-finite {metric_name}")
            values[metric_name] = value
        by_seed[seed] = values

    mean_std = {
        metric_name: (
            statistics.fmean(values[metric_name] for values in by_seed.values()),
            statistics.pstdev(values[metric_name] for values in by_seed.values()),
        )
        for metric_name in dict(METRICS[dataset]).values()
    }

    metrics_path = output_dir / "metrics.json"
    aggregate = read_json(metrics_path)
    require(aggregate.get("model") == model, f"{metrics_path}: model mismatch")
    require(aggregate.get("task") == task, f"{metrics_path}: task mismatch")
    require(aggregate.get("dataset") == dataset, f"{metrics_path}: dataset mismatch")
    require(aggregate.get("seed") == BASE_SEED, f"{metrics_path}: split/base seed mismatch")
    require(aggregate.get("ablation") == "full", f"{metrics_path}: top-level protocol mismatch")
    for metric_name, (mean, std) in mean_std.items():
        saved = aggregate.get("metrics", {}).get(metric_name)
        require(isinstance(saved, dict), f"{metrics_path}: missing aggregate {metric_name}")
        require(close(saved.get("mean"), mean), f"{metrics_path}: mean differs from per-seed values for {metric_name}")
        require(close(saved.get("std"), std), f"{metrics_path}: population std differs for {metric_name}")
    return by_seed, mean_std


def check_complete_marker(output_dir: Path, dataset: str) -> None:
    marker = read_json(output_dir / "complete.marker")
    require(marker.get("status") == "complete", f"{output_dir}: incomplete marker")
    require(marker.get("dataset") == dataset, f"{output_dir}: marker dataset mismatch")
    require(marker.get("task") == DATASET_TASK[dataset], f"{output_dir}: marker task mismatch")
    require(marker.get("seed") == BASE_SEED, f"{output_dir}: marker seed mismatch")


def check_unit_artifacts(output_dir: Path, dataset: str) -> None:
    required = (
        "main.log",
        "train.log",
        "results.json",
        "metrics.json",
        "resolved_config.json",
        "ablation_manifest.json",
        "benchmark_manifest.json",
        "complete.marker",
        "best.pt",
        "best_run1.pt",
        "best_run2.pt",
        "best_run3.pt",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    require(not missing, f"{output_dir}: incomplete outputs, missing {', '.join(missing)}")
    pointer = output_dir / "best.pt"
    require(pointer.is_symlink(), f"{output_dir}: best.pt is not the expected symlink")
    require(
        pointer.resolve() == (output_dir / "best_run3.pt").resolve(),
        f"{output_dir}: best.pt does not point to best_run3.pt",
    )
    for log_name in ("main.log", "train.log"):
        log = (output_dir / log_name).read_text(encoding="utf-8", errors="replace")
        require(NONFINITE_RE.search(log) is None, f"{output_dir}: non-finite token in {log_name}")
    results = read_json(output_dir / "results.json")
    require(isinstance(results, dict) and bool(results), f"{output_dir}: results.json is empty")
    check_complete_marker(output_dir, dataset)


def format_mean_std(value: tuple[float, float]) -> str:
    mean, std = value
    return f"{mean:.6f} ± {std:.6f}"


def run_summarizer(
    benchmark_root: Path,
    ablation_root: Path,
    datasets: tuple[str, ...] = tuple(DATASET_TASK),
    output_prefix: str = "table2",
) -> int:
    audit_suffix = output_prefix[len("table2"):] if output_prefix.startswith("table2") else f"_{output_prefix}"
    audit_name = "resolved_protocol_audit.json" if not audit_suffix else f"resolved_protocol_audit{audit_suffix}.json"
    audit: dict[str, Any] = {
        "protocol": "cosi_mag_final_framework_ablation_v1",
        "benchmark_root": str(benchmark_root.resolve()),
        "ablation_root": str(ablation_root.resolve()),
        "base_seed": BASE_SEED,
        "run_seeds": list(RUN_SEEDS),
        "num_runs": NUM_RUNS,
        "task_protocols": TASK_PROTOCOLS,
        "datasets": list(datasets),
        "units": [],
        "errors": [],
        "passed": False,
    }
    table_values: dict[tuple[str, str, str], tuple[float, float]] = {}
    seed_values: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    try:
        # The clean baseline manifests are checked once; every unit below also
        # validates its own resolved config and the exact split-file digest.
        check_ablation_root_manifest(
            ablation_root / "manifests" / "framework_ablation.json"
        )
        for task, manifest_name in (("nc", "nc.json"), ("lp", "sports-lp.json")):
            path = benchmark_root / "manifests" / manifest_name
            check_top_manifest(path, task, "cosi_mag_final", None)

        for dataset in datasets:
            task = DATASET_TASK[dataset]
            full_dir = unit_path(benchmark_root, dataset)
            check_unit_manifest(
                full_dir / "benchmark_manifest.json",
                model="cosi_mag_final",
                dataset=dataset,
                variant=None,
            )
            check_unit_artifacts(full_dir, dataset)
            clean_metrics, _ = load_seed_metrics(
                full_dir,
                dataset=dataset,
                model="cosi_mag_final",
                variant=None,
            )
            seed_values[("full", dataset)] = clean_metrics
            audit["units"].append(
                {
                    "variant": "full",
                    "dataset": dataset,
                    "task": task,
                    "model": "cosi_mag_final",
                    "base_seed": BASE_SEED,
                    "num_runs": NUM_RUNS,
                    "run_seeds": list(RUN_SEEDS),
                    "benchmark_manifest": str((full_dir / "benchmark_manifest.json").resolve()),
                    "passed": True,
                }
            )

            for variant in VARIANTS:
                variant_dir = unit_path(ablation_root, dataset, variant)
                check_unit_manifest(
                    variant_dir / "benchmark_manifest.json",
                    model="cosi_mag_ablation",
                    dataset=dataset,
                    variant=variant,
                )
                check_unit_artifacts(variant_dir, dataset)
                _, _, protocol_audit = load_pair_configs(
                    full_dir, variant_dir, dataset, variant
                )
                variant_metrics, variant_mean_std = load_seed_metrics(
                    variant_dir,
                    dataset=dataset,
                    model="cosi_mag_ablation",
                    variant=variant,
                )
                for metric_name, summary in variant_mean_std.items():
                    table_values[(variant, dataset, metric_name)] = summary
                seed_values[(variant, dataset)] = variant_metrics
                audit["units"].append(
                    {
                        "variant": variant,
                        "dataset": dataset,
                        "task": task,
                        "model": "cosi_mag_ablation",
                        "run_seeds": list(RUN_SEEDS),
                        "benchmark_manifest": str((variant_dir / "benchmark_manifest.json").resolve()),
                        "checks": protocol_audit,
                        "passed": True,
                    }
                )

        # Full values have already been read from the clean benchmark. Keep
        # those values alongside the ablation rows in the final table.
        baseline_summaries: dict[tuple[str, str], tuple[float, float]] = {}
        for dataset in datasets:
            by_seed = seed_values[("full", dataset)]
            for _, metric_name in METRICS[dataset]:
                values = [by_seed[seed][metric_name] for seed in RUN_SEEDS]
                baseline_summaries[(dataset, metric_name)] = (
                    statistics.fmean(values), statistics.pstdev(values)
                )
                table_values[("full", dataset, metric_name)] = baseline_summaries[(dataset, metric_name)]

        audit["passed"] = True
    except (AuditError, OSError, KeyError, TypeError, ValueError, ImportError) as exc:
        audit["errors"].append(str(exc))
        write_json(ablation_root / audit_name, audit)
        print(f"Protocol audit failed: {exc}", file=sys.stderr)
        print(f"Saved audit: {ablation_root / audit_name}", file=sys.stderr)
        return 1

    write_table(
        benchmark_root,
        ablation_root,
        table_values,
        seed_values,
        datasets=datasets,
        output_prefix=output_prefix,
    )
    write_json(ablation_root / audit_name, audit)
    suffix = output_prefix[len("table2"):] if output_prefix.startswith("table2") else f"_{output_prefix}"
    delta_name = "per_seed_deltas" if not suffix else f"per_seed_deltas{suffix}"
    print(f"Saved {ablation_root / f'{output_prefix}.csv'}")
    print(f"Saved {ablation_root / f'{delta_name}.csv'}")
    print(f"Saved {ablation_root / audit_name}")
    return 0


def write_table(
    benchmark_root: Path,
    ablation_root: Path,
    table_values: dict[tuple[str, str, str], tuple[float, float]],
    seed_values: dict[tuple[str, str], dict[int, dict[str, float]]],
    *,
    datasets: tuple[str, ...] = tuple(DATASET_TASK),
    output_prefix: str = "table2",
) -> None:
    del benchmark_root  # Paths were validated and recorded in the audit payload.
    table_path = ablation_root / f"{output_prefix}.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["Model"]
    for dataset in datasets:
        prefix = "Sports" if dataset == "sports-copurchase" else dataset
        columns.extend(f"{prefix} {display}" for display, _ in METRICS[dataset])
    rows = []
    row_order = [
        ("full", "CoSI-MAG"),
        ("no_mrc", VARIANTS["no_mrc"]),
        ("no_semantic_anchor", VARIANTS["no_semantic_anchor"]),
        ("no_rcmi", VARIANTS["no_rcmi"]),
    ]
    for variant, label in row_order:
        row: dict[str, str] = {"Model": label}
        for dataset in datasets:
            prefix = "Sports" if dataset == "sports-copurchase" else dataset
            for display, metric_name in METRICS[dataset]:
                row[f"{prefix} {display}"] = format_mean_std(
                    table_values[(variant, dataset, metric_name)]
                )
        rows.append(row)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    suffix = output_prefix[len("table2"):] if output_prefix.startswith("table2") else f"_{output_prefix}"
    delta_name = "per_seed_deltas" if not suffix else f"per_seed_deltas{suffix}"
    delta_path = ablation_root / f"{delta_name}.csv"
    with delta_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "task",
            "variant",
            "seed",
            "metric",
            "full_value",
            "ablation_value",
            "delta_ablation_minus_full",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            full_by_seed = seed_values[("full", dataset)]
            for variant, label in VARIANTS.items():
                ablation_by_seed = seed_values[(variant, dataset)]
                for seed in RUN_SEEDS:
                    for _, metric_name in METRICS[dataset]:
                        full_value = full_by_seed[seed][metric_name]
                        ablation_value = ablation_by_seed[seed][metric_name]
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "task": DATASET_TASK[dataset],
                                "variant": label,
                                "seed": seed,
                                "metric": metric_name,
                                "full_value": f"{full_value:.10f}",
                                "ablation_value": f"{ablation_value:.10f}",
                                "delta_ablation_minus_full": f"{ablation_value - full_value:.10f}",
                            }
                        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=ROOT / "outputs" / "cosi_mag_final_ablation",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_TASK),
        default=list(DATASET_TASK),
        help="dataset subset to audit and summarize",
    )
    parser.add_argument(
        "--output-prefix",
        default="table2",
        help="prefix for the CSV outputs (e.g. table2_nc)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_summarizer(
        args.benchmark_root,
        args.ablation_root,
        datasets=tuple(args.datasets),
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    raise SystemExit(main())
