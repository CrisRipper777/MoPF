#!/usr/bin/env python3
"""Run the frozen IAMOC NC 2x2 factorial grid in an isolated output tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "iamoc_nc_generalization"
LEGACY_ROOT = ROOT / "outputs" / "iamoc_v1"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS: dict[str, dict[str, Any]] = {
    "V00": {"model": "mopf", "model.use_transport_residual": "false"},
    "V0": {"model": "mopf", "model.use_transport_residual": "true"},
    "V1": {
        "model": "mopf_iamoc",
        "model.hop_interaction_layers": "1",
        "model.relation_conditioning": "output",
    },
    "V2": {
        "model": "mopf_iamoc",
        "model.hop_interaction_layers": "1",
        "model.relation_conditioning": "none",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _complete(run_dir: Path) -> bool:
    return all(
        (run_dir / filename).is_file()
        for filename in (
            "complete.marker",
            "metrics.json",
            "results.json",
            "best.pt",
            "resolved_config.yaml",
            "resolved_config.json",
            "run_record.json",
            "ablation_manifest.json",
        )
    )


def _config_mapping(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)  # type: ignore[return-value]


def _tree_equal(actual: Any, expected: Any, *, skip: frozenset[str] = frozenset()) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _tree_equal(actual[key], value, skip=skip)
            for key, value in expected.items()
            if key not in skip
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _tree_equal(a, e, skip=skip) for a, e in zip(actual, expected)
        )
    return actual == expected


def _legacy_reuse_audit(
    dataset: str, variant: str, seed: int, source_root: Path | None = None
) -> dict[str, Any] | None:
    """Return a reuse manifest only when the saved protocol matches this grid."""
    if dataset not in {"Movies", "Grocery"} or variant not in {"V0", "V1", "V2"}:
        return None
    source = (source_root or LEGACY_ROOT) / dataset / variant / f"seed_{seed}"
    if not _complete(source):
        return None

    resolved = _load_json(source / "resolved_config.json")
    record = _load_json(source / "run_record.json")
    metrics = _load_json(source / "metrics.json")
    manifest = _load_json(source / "ablation_manifest.json")
    expected_model_name = "mopf" if variant == "V0" else "mopf_iamoc"
    config_root = ROOT / "configs"
    formal_model = _config_mapping(config_root / "model" / "mopf.yaml")
    expected_task = _config_mapping(config_root / "task" / "nc.yaml")

    reasons: list[str] = []
    if record.get("status") != "complete" or record.get("return_code") != 0:
        reasons.append("run record is not a successful completed run")
    if resolved.get("dataset", {}).get("name") != dataset:
        reasons.append("dataset differs")
    if resolved.get("task", {}).get("name") != "nc":
        reasons.append("task is not NC")
    if int(resolved.get("seed", -1)) != seed:
        reasons.append("seed differs")
    if resolved.get("model", {}).get("name") != expected_model_name:
        reasons.append("model name differs")

    # Every shared model field must still match the frozen formal MoPF YAML.
    actual_model = resolved.get("model", {})
    for key, expected_value in formal_model.items():
        if key in {"name", "version"}:
            continue
        if isinstance(expected_value, str) and expected_value.startswith("${model."):
            expected_value = actual_model.get(expected_value[8:-1])
        if actual_model.get(key) != expected_value:
            reasons.append(f"formal model field differs: {key}")
            break
    if variant in {"V1", "V2"}:
        iamoc_model = _config_mapping(config_root / "model" / "mopf_iamoc.yaml")
        if actual_model.get("name") != iamoc_model.get("name") or actual_model.get("version") != iamoc_model.get("version"):
            reasons.append("IAMOC model identity/version differs")
        else:
            expected_relation = "output" if variant == "V1" else "none"
            for key, expected_value in iamoc_model.items():
                if key in {"name", "version"}:
                    continue
                if isinstance(expected_value, str) and expected_value.startswith("${model."):
                    expected_value = actual_model.get(expected_value[8:-1])
                if key == "hop_interaction_layers":
                    expected_value = 1
                elif key == "relation_conditioning":
                    expected_value = expected_relation
                if actual_model.get(key) != expected_value:
                    reasons.append(f"IAMOC model field differs: {key}")
                    break
    if variant == "V0" and actual_model.get("use_transport_residual") is not True:
        reasons.append("V0 does not have TCPR enabled")
    if variant == "V1" and (
        actual_model.get("hop_interaction_layers") != 1
        or actual_model.get("relation_conditioning") != "output"
    ):
        reasons.append("V1 interaction/TCPR settings differ")
    if variant == "V2" and (
        actual_model.get("hop_interaction_layers") != 1
        or actual_model.get("relation_conditioning") != "none"
    ):
        reasons.append("V2 interaction/TCPR settings differ")

    actual_task = resolved.get("task", {})
    if not _tree_equal(actual_task, expected_task, skip=frozenset({"save_ckpt_path"})):
        reasons.append("NC training/evaluation protocol differs from configs/task/nc.yaml")
    if actual_task.get("protocol_version") != "unified_full_graph_nc_v1":
        reasons.append("NC protocol version differs")
    if metrics.get("checkpoint_selection") != "best_val_accuracy":
        reasons.append("checkpoint selection is not best validation accuracy")
    if manifest.get("checkpoint_selection") != "best_val_accuracy":
        reasons.append("manifest checkpoint selection differs")
    if metrics.get("selection_metric") != "val_acc":
        reasons.append("selection metric is not val_acc")

    split_value = resolved.get("dataset", {}).get("nc_split_path")
    split_path = Path(split_value) if split_value else None
    if split_path is None or not split_path.is_file():
        reasons.append("configured NC split is missing")
    else:
        expected_split = Path("/hdd1/DataInHere/YHF/data/MAGB_split") / f"{dataset}_nc_seed{seed}_train0.6_val0.2.pt"
        if split_path.resolve() != expected_split.resolve():
            reasons.append("configured split path differs")
        started_at = float(record.get("started_at_unix", float("inf")))
        if split_path.stat().st_mtime > started_at:
            reasons.append("split file is newer than the original run record")

    # Resolve the current dataset YAML against the original recorded roots and
    # compare every dataset field, including graph/features/splits and ratios.
    try:
        expected_dataset_cfg = OmegaConf.create(
            {
                "paths": resolved["paths"],
                "dataset": OmegaConf.load(config_root / "dataset" / f"{dataset}.yaml"),
                "seed": seed,
            }
        )
        expected_dataset = OmegaConf.to_container(expected_dataset_cfg, resolve=True)["dataset"]
        if not _tree_equal(resolved.get("dataset", {}), expected_dataset):
            reasons.append("resolved dataset config differs from the current dataset YAML")
    except Exception as error:
        reasons.append(f"could not resolve dataset config for reuse audit: {error}")

    if reasons:
        print(f"LEGACY REJECT {dataset}/{variant}/seed_{seed}: " + "; ".join(reasons), flush=True)
        return None

    return {
        "reused": True,
        "source_run_dir": str(source),
        "source_checkpoint": str(source / "best.pt"),
        "source_checkpoint_sha256": _sha256(source / "best.pt"),
        "source_resolved_config_sha256": _sha256(source / "resolved_config.json"),
        "split_path": str(split_path),
        "split_sha256_at_reuse": _sha256(split_path),
        "split_mtime_unix": split_path.stat().st_mtime,
        "original_run_started_at_unix": record.get("started_at_unix"),
        "checks": {
            "dataset_task_seed": True,
            "frozen_formal_model_fields": True,
            "all_iamoc_model_fields": True,
            "iamoc_specific_fields_checked": True if variant in {"V1", "V2"} else "not_applicable",
            "variant_definition": True,
            "complete_dataset_yaml_match": True,
            "nc_training_and_evaluation_protocol": True,
            "split_path_and_pre_run_timestamp": True,
            "best_validation_accuracy_checkpoint_selection": True,
            "complete_checkpoint_and_metrics_artifacts": True,
        },
    }


def _build_run(dataset: str, variant: str, seed: int, device: str) -> tuple[Path, list[str], dict[str, Any]]:
    run_dir = OUTPUT_ROOT / dataset / variant / f"seed_{seed}"
    variant_config = VARIANTS[variant]
    model_name = str(variant_config["model"])
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        f"model={model_name}",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        f"paths.output_root={OUTPUT_ROOT}",
        f"task.save_ckpt_path={run_dir / 'best.pt'}",
        f"hydra.run.dir={run_dir}",
    ]
    overrides.extend(f"{key}={value}" for key, value in variant_config.items() if key != "model")
    command = [sys.executable, "-m", "src.main", *overrides]
    config_record = {
        "dataset": dataset,
        "task": "nc",
        "variant": variant,
        "seed": seed,
        "device": device,
        "model": model_name,
        "variant_definition": variant_config,
        "command": command,
        "command_display": shlex.join(command),
        "output_dir": str(run_dir),
        "status": "planned",
        "reused_legacy_result": False,
    }
    return run_dir, command, config_record


def _run_one(dataset: str, variant: str, seed: int, device: str, *, dry_run: bool, resume: bool) -> int:
    run_dir, command, record = _build_run(dataset, variant, seed, device)
    reuse = _legacy_reuse_audit(dataset, variant, seed)
    if reuse is not None:
        record.update({"status": "reused", "reused_legacy_result": True, "reuse_audit": reuse})
        manifest_path = run_dir / "reuse_manifest.json"
        if not dry_run:
            _write_json(manifest_path, {**record, "reuse_audit": reuse})
        print(f"REUSE {dataset} {variant} seed={seed} <- {reuse['source_run_dir']}", flush=True)
        return 0

    if dry_run:
        print(shlex.join(command))
        return 0
    if resume and _complete(run_dir):
        print(f"SKIP complete: {dataset} {variant} seed={seed}", flush=True)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    record.update({"status": "running", "started_at_unix": time.time()})
    _write_json(run_dir / "run_record.json", record)
    started = time.perf_counter()
    with (run_dir / "runner.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    record["duration_seconds"] = time.perf_counter() - started
    record["finished_at_unix"] = time.time()
    record["return_code"] = int(process.returncode)
    if process.returncode == 0 and _complete(run_dir):
        record["status"] = "complete"
        print(f"DONE {dataset} {variant} seed={seed}", flush=True)
    else:
        record["status"] = "failed"
        record["failure"] = "training command failed or did not produce all completion artifacts"
        print(f"FAIL {dataset} {variant} seed={seed}: {run_dir / 'runner.log'}", flush=True)
    _write_json(run_dir / "run_record.json", record)
    return 0 if record["status"] == "complete" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.device).startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    failures = 0
    total = len(args.datasets) * len(args.variants) * len(args.seeds)
    print(f"Grid: {total} slots; output={OUTPUT_ROOT}; device={args.device}", flush=True)
    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                failures += _run_one(dataset, variant, seed, args.device, dry_run=args.dry_run, resume=args.resume)
    if not args.dry_run:
        _write_json(
            OUTPUT_ROOT / "run_grid_summary.json",
            {
                "datasets": args.datasets,
                "variants": args.variants,
                "seeds": args.seeds,
                "device": args.device,
                "planned_slots": total,
                "failures": failures,
                "legacy_root": str(LEGACY_ROOT),
                "output_root": str(OUTPUT_ROOT),
            },
        )
    print(f"Run summary: failures={failures}/{total}; root={OUTPUT_ROOT}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
