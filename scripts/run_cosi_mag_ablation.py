#!/usr/bin/env python3
"""Launch the three frozen CoSI-MAG framework ablations.

Each launcher unit loads its split once with base seed 42 and runs seeds
42/43/44 internally. ``--dry-run`` prints the nine default commands without
invoking ``src.main``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "cosi_mag_final_ablation"
FINAL_MODEL_CONFIG_PATH = ROOT / "configs" / "model" / "cosi_mag_final.yaml"
ABLATION_MODEL_CONFIG_PATH = ROOT / "configs" / "model" / "cosi_mag_ablation.yaml"
TASK_CONFIG_PATHS = {
    "nc": ROOT / "configs" / "task" / "nc.yaml",
    "lp": ROOT / "configs" / "task" / "lp.yaml",
}
DATASET_TASK = {
    "Movies": "nc",
    "Grocery": "nc",
    "sports-copurchase": "lp",
}
VARIANTS = ("no_mrc", "no_semantic_anchor", "no_rcmi")
ALL_DATASETS = ("Movies", "Grocery", "sports-copurchase")
BASE_SEED = 42
RUN_SEEDS = (42, 43, 44)
NUM_RUNS = 3
EXPECTED_PROTOCOLS = {
    "nc": "unified_full_graph_nc_v1",
    "lp": "unified_sampled_lp_v1",
}
FINAL_MODEL_AUDIT_KEYS = (
    "hidden_dim",
    "max_order",
    "multihop_anchor_alpha",
    "edge_weight_min",
    "edge_weight_temperature",
    "filter_rank",
    "hop_interaction_layers",
    "hop_interaction_heads",
    "relation_bias_init",
)
NONFINITE_RE = re.compile(
    r"\b(?:Train Loss|Val (?:MRR|H@1|H@3|H@10))\s+(?:nan|[+-]?inf(?:inity)?)\b",
    re.IGNORECASE,
)
REQUIRED_FILES = (
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_frozen_configs(datasets: list[str]) -> dict[str, str]:
    final_cfg = read_yaml(FINAL_MODEL_CONFIG_PATH)
    ablation_cfg = read_yaml(ABLATION_MODEL_CONFIG_PATH)
    if final_cfg.get("name") != "cosi_mag_final":
        raise ValueError("frozen final model config name changed")
    if ablation_cfg.get("name") != "cosi_mag_ablation":
        raise ValueError("ablation model config name must be cosi_mag_ablation")
    expected_ablation_cfg = {
        key: value for key, value in final_cfg.items() if key not in {"name", "version"}
    }
    expected_ablation_cfg.update(
        {"name": "cosi_mag_ablation", "version": "cosi_mag_ablation", "ablation_mode": "full"}
    )
    if ablation_cfg != expected_ablation_cfg:
        raise ValueError(
            "cosi_mag_ablation.yaml must match cosi_mag_final.yaml except for "
            "name/version and ablation_mode"
        )

    for key in FINAL_MODEL_AUDIT_KEYS:
        if key not in final_cfg:
            raise ValueError(f"final model config is missing frozen field {key}")

    for task, path in TASK_CONFIG_PATHS.items():
        task_cfg = read_yaml(path)
        if task_cfg.get("protocol_version") != EXPECTED_PROTOCOLS[task]:
            raise ValueError(
                f"frozen {task} task protocol mismatch: {task_cfg.get('protocol_version')!r}"
            )
    for dataset in datasets:
        path = ROOT / "configs" / "dataset" / f"{dataset}.yaml"
        dataset_cfg = read_yaml(path)
        if dataset_cfg.get("name") != dataset:
            raise ValueError(f"dataset config name mismatch in {path}")
        if DATASET_TASK[dataset] not in dataset_cfg.get("tasks", []):
            raise ValueError(f"dataset {dataset} does not declare task {DATASET_TASK[dataset]}")

    paths = {
        "model": ABLATION_MODEL_CONFIG_PATH,
        "final_model": FINAL_MODEL_CONFIG_PATH,
        "nc_task": TASK_CONFIG_PATHS["nc"],
        "lp_task": TASK_CONFIG_PATHS["lp"],
    }
    paths.update(
        {
            f"dataset_{dataset}": ROOT / "configs" / "dataset" / f"{dataset}.yaml"
            for dataset in datasets
        }
    )
    return {name: sha256_file(path) for name, path in paths.items()}


def unit_dir(variant: str, dataset: str) -> Path:
    task = DATASET_TASK[dataset]
    return OUTPUT_ROOT / variant / task / dataset / "runs_42_43_44"


def make_command(variant: str, dataset: str, device: str, output_dir: Path) -> list[str]:
    task = DATASET_TASK[dataset]
    alpha = "0.0" if variant == "no_semantic_anchor" else "0.1"
    return [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        f"task={task}",
        "model=cosi_mag_ablation",
        f"model.ablation_mode={variant}",
        f"model.multihop_anchor_alpha={alpha}",
        f"seed={BASE_SEED}",
        f"num_runs={NUM_RUNS}",
        f"device={device}",
        "ablation=full",
        f"task.save_ckpt_path={output_dir / 'best.pt'}",
        f"hydra.run.dir={output_dir}",
    ]


def expected_alpha(variant: str) -> float:
    return 0.0 if variant == "no_semantic_anchor" else 0.1


def validate_resolved_config(
    path: Path, variant: str, dataset: str, device: str
) -> tuple[bool, str]:
    try:
        config = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"resolved config unreadable: {exc}"
    if not isinstance(config, dict):
        return False, "resolved config is not a mapping"
    task = DATASET_TASK[dataset]
    checks = {
        "model.name": (config.get("model", {}).get("name"), "cosi_mag_ablation"),
        "model.ablation_mode": (config.get("model", {}).get("ablation_mode"), variant),
        "task.name": (config.get("task", {}).get("name"), task),
        "task.protocol_version": (
            config.get("task", {}).get("protocol_version"),
            EXPECTED_PROTOCOLS[task],
        ),
        "dataset.name": (config.get("dataset", {}).get("name"), dataset),
        "seed": (config.get("seed"), BASE_SEED),
        "num_runs": (config.get("num_runs"), NUM_RUNS),
        "device": (config.get("device"), device),
        "model.multihop_anchor_alpha": (
            config.get("model", {}).get("multihop_anchor_alpha"),
            expected_alpha(variant),
        ),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            return False, f"{name}: expected {expected!r}, got {actual!r}"
    return True, "resolved configuration matches the ablation protocol"


def completion_status(
    output_dir: Path,
    variant: str,
    dataset: str,
    device: str,
    config_hashes: dict[str, str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for name in REQUIRED_FILES:
        path = output_dir / name
        if not path.is_file():
            reasons.append(f"missing {name}")

    for name in ("main.log", "train.log"):
        path = output_dir / name
        if path.is_file():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if NONFINITE_RE.search(line):
                    reasons.append(f"{name}:{line_number} contains non-finite training/validation value")
                    break

    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = read_json(metrics_path)
            expected = {
                "model": "cosi_mag_ablation",
                "task": DATASET_TASK[dataset],
                "dataset": dataset,
                "seed": BASE_SEED,
                "ablation": "full",
            }
            for key, value in expected.items():
                if metrics.get(key) != value:
                    reasons.append(f"metrics.json {key} mismatch")
            if contains_nonfinite(metrics):
                reasons.append("metrics.json contains NaN/Inf")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            reasons.append(f"metrics.json unreadable: {exc}")

    results_path = output_dir / "results.json"
    if results_path.is_file():
        try:
            if contains_nonfinite(read_json(results_path)):
                reasons.append("results.json contains NaN/Inf")
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"results.json unreadable: {exc}")

    resolved_path = output_dir / "resolved_config.json"
    if resolved_path.is_file():
        valid, reason = validate_resolved_config(
            resolved_path, variant, dataset, device
        )
        if not valid:
            reasons.append(f"resolved config mismatch: {reason}")

    unit_manifest_path = output_dir / "benchmark_manifest.json"
    if unit_manifest_path.is_file():
        try:
            manifest = read_json(unit_manifest_path)
            if manifest.get("config_sha256") != config_hashes:
                reasons.append("benchmark manifest config hashes mismatch")
            for key, value in {
                "model": "cosi_mag_ablation",
                "ablation_mode": variant,
                "dataset": dataset,
                "task": DATASET_TASK[dataset],
                "base_seed": BASE_SEED,
                "run_seeds": list(RUN_SEEDS),
                "num_runs": NUM_RUNS,
            }.items():
                if manifest.get(key) != value:
                    reasons.append(f"benchmark manifest {key} mismatch")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            reasons.append(f"benchmark manifest unreadable: {exc}")

    marker_path = output_dir / "complete.marker"
    if marker_path.is_file():
        try:
            marker = read_json(marker_path)
            for key, value in {
                "status": "complete",
                "task": DATASET_TASK[dataset],
                "dataset": dataset,
                "seed": BASE_SEED,
            }.items():
                if marker.get(key) != value:
                    reasons.append(f"complete marker {key} mismatch")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            reasons.append(f"complete marker unreadable: {exc}")

    try:
        import torch

        for index, seed in enumerate(RUN_SEEDS, 1):
            path = output_dir / f"best_run{index}.pt"
            if not path.is_file():
                continue
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                reasons.append(f"{path.name} unreadable: {exc}")
                continue
            if checkpoint.get("task") != DATASET_TASK[dataset] or checkpoint.get("seed") != seed:
                reasons.append(f"{path.name} task/seed mismatch")
            if not isinstance(checkpoint.get("model_state"), dict) or not checkpoint["model_state"]:
                reasons.append(f"{path.name} missing model_state")
    except ImportError:
        reasons.append("PyTorch is required for checkpoint audit")

    pointer = output_dir / "best.pt"
    latest = output_dir / f"best_run{NUM_RUNS}.pt"
    if pointer.is_symlink() and pointer.resolve() != latest.resolve():
        reasons.append("best.pt symlink does not point to best_run3.pt")
    elif pointer.exists() and not pointer.is_symlink():
        reasons.append("best.pt exists but is not a checkpoint symlink")
    return not reasons, reasons


def contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_nonfinite(item) for item in value)
    return False


def build_manifest(
    requested_variants: list[str],
    requested_datasets: list[str],
    device: str,
    config_hashes: dict[str, str],
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    requested = {
        (item["variant"], item["dataset"]): item
        for item in plan
    }
    units: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for dataset in ALL_DATASETS:
            selected = requested.get((variant, dataset))
            if selected is None:
                output_dir = unit_dir(variant, dataset)
                selected = {
                    "variant": variant,
                    "dataset": dataset,
                    "task": DATASET_TASK[dataset],
                    "base_seed": BASE_SEED,
                    "split_seed": BASE_SEED,
                    "run_seeds": list(RUN_SEEDS),
                    "num_runs": NUM_RUNS,
                    "device": device,
                    "output_dir": str(output_dir),
                    "checkpoint": str(output_dir / "best.pt"),
                    "command": make_command(variant, dataset, device, output_dir),
                    "requested": False,
                }
            else:
                selected = {**selected, "requested": True}
            units.append(selected)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "cosi_mag_final_framework_ablation_v1",
        "model": "cosi_mag_ablation",
        "config_sha256": config_hashes,
        "task_protocols": EXPECTED_PROTOCOLS,
        "variants": list(VARIANTS),
        "datasets": list(ALL_DATASETS),
        "requested_variants": requested_variants,
        "requested_datasets": requested_datasets,
        "base_seed": BASE_SEED,
        "split_seed": BASE_SEED,
        "run_seeds": list(RUN_SEEDS),
        "num_runs": NUM_RUNS,
        "device": device,
        "units": units,
    }


def prepare_checkpoint_pointer(output_dir: Path) -> tuple[bool, str]:
    pointer = output_dir / "best.pt"
    target = Path(f"best_run{NUM_RUNS}.pt")
    if pointer.is_symlink():
        if pointer.resolve() == (output_dir / target).resolve():
            return True, "checkpoint pointer is ready"
        return False, f"preserving unexpected best.pt symlink: {pointer.readlink()}"
    if pointer.exists():
        return False, "preserving existing non-symlink best.pt"
    pointer.symlink_to(target)
    return True, "created best.pt pointer to best_run3.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=list(ALL_DATASETS))
    parser.add_argument("--device", default="cuda:0", help="Hydra device value")
    parser.add_argument("--resume", action="store_true", help="skip only fully audited completed units")
    parser.add_argument("--dry-run", action="store_true", help="print commands without invoking src.main")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.device.strip():
        raise SystemExit("--device must not be empty")
    if len(set(args.variants)) != len(args.variants):
        raise SystemExit("--variants must not contain duplicates")
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets must not contain duplicates")

    # Keep launcher manifests stable even when a user launches the matrix in
    # dataset/variant subsets over several sessions.
    config_hashes = validate_frozen_configs(list(ALL_DATASETS))
    plan: list[dict[str, Any]] = []
    for variant in args.variants:
        for dataset in args.datasets:
            output_dir = unit_dir(variant, dataset)
            valid, reasons = completion_status(
                output_dir, variant, dataset, args.device, config_hashes
            )
            skip = bool(args.resume and valid)
            if output_dir.exists() and any(output_dir.iterdir()) and not valid:
                warnings.warn(
                    f"Existing ablation unit is incomplete and will be preserved while launcher-owned "
                    f"files are regenerated: {output_dir}. Audit: {'; '.join(reasons)}",
                    RuntimeWarning,
                )
            plan.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "task": DATASET_TASK[dataset],
                    "base_seed": BASE_SEED,
                    "split_seed": BASE_SEED,
                    "run_seeds": list(RUN_SEEDS),
                    "num_runs": NUM_RUNS,
                    "device": args.device,
                    "output_dir": str(output_dir),
                    "checkpoint": str(output_dir / "best.pt"),
                    "resume_skip": skip,
                    "resume_audit": reasons,
                    "command": make_command(variant, dataset, args.device, output_dir),
                }
            )

    manifest = build_manifest(
        args.variants, args.datasets, args.device, config_hashes, plan
    )
    manifest_path = OUTPUT_ROOT / "manifests" / "framework_ablation.json"
    write_json(manifest_path, manifest)
    print("CoSI-MAG framework ablation manifest:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved manifest: {manifest_path}")

    invocations = 0
    failures = 0
    for item in plan:
        output_dir = Path(item["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if item["resume_skip"]:
            print(f"[RESUME SKIP] {item['variant']} {item['dataset']} seeds={RUN_SEEDS}")
            continue

        unit_manifest = {
            "protocol": manifest["protocol"],
            "model": "cosi_mag_ablation",
            "ablation_mode": item["variant"],
            "task": item["task"],
            "dataset": item["dataset"],
            "base_seed": BASE_SEED,
            "split_seed": BASE_SEED,
            "run_seeds": list(RUN_SEEDS),
            "num_runs": NUM_RUNS,
            "device": args.device,
            "multihop_anchor_alpha": expected_alpha(item["variant"]),
            "task_protocol": EXPECTED_PROTOCOLS[item["task"]],
            "config_sha256": config_hashes,
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "benchmark_manifest.json", unit_manifest)

        command = item["command"]
        if args.dry_run:
            invocations += 1
            print(f"[DRY-RUN] {item['variant']} {item['dataset']} split_seed={BASE_SEED} run_seeds={RUN_SEEDS}")
            print(shlex.join(command))
            continue

        pointer_ready, message = prepare_checkpoint_pointer(output_dir)
        if not pointer_ready:
            failures += 1
            print(f"[FAIL] {message}")
            continue
        print(f"[CHECKPOINT] {message}")
        invocations += 1
        print(
            f"[RUN] {item['variant']} {item['dataset']} split_seed={BASE_SEED} "
            f"run_seeds={RUN_SEEDS}",
            flush=True,
        )
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            failures += 1
            print(f"[FAIL] process exited with status {result.returncode}: {output_dir}")
            continue
        valid, reasons = completion_status(
            output_dir,
            item["variant"],
            item["dataset"],
            args.device,
            config_hashes,
        )
        if valid:
            print(f"[COMPLETE] {item['variant']} {item['dataset']}")
        else:
            failures += 1
            print(f"[FAIL] output audit failed for {output_dir}: {'; '.join(reasons)}")

    if args.dry_run:
        print(f"Dry-run launcher units: {invocations}; src.main invocations: 0")
        return 0
    print(f"Launched units: {invocations}; failed audits/processes: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
