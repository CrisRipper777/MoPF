#!/usr/bin/env python3
"""Launch the frozen CoSI-MAG final NC/LP benchmark, one dataset per process.

Successful units are never repeated because a test score is low. With
``--resume``, a unit is skipped only when its completion marker, metrics,
checkpoint, resolved configuration, and run identity all validate. Reruns are
for crashes, NaN/Inf, missing outputs, corrupt checkpoints, or config/protocol
mismatches; this launcher never retries a completed unit based on its score.

Use ``--dry-run`` to print commands and create the deterministic output
directories without invoking ``src.main``.
Each process loads the dataset split once with base seed 42, then performs
three training runs whose run seeds are 42, 43, and 44.
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
OUTPUT_ROOT = ROOT / "outputs" / "cosi_mag_final_benchmark"
MODEL_CONFIG_PATH = ROOT / "configs" / "model" / "cosi_mag_final.yaml"
TASK_CONFIG_PATHS = {
    "nc": ROOT / "configs" / "task" / "nc.yaml",
    "lp": ROOT / "configs" / "task" / "lp.yaml",
}

SUITES = {
    "nc": {"task": "nc", "datasets": ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]},
    "sports-lp": {"task": "lp", "datasets": ["sports-copurchase"]},
    "cloth-lp": {"task": "lp", "datasets": ["cloth-copurchase"]},
}
DEFAULT_SEEDS = [42, 43, 44]
BASE_SEED = 42
NUM_RUNS = 3
NONFINITE_TRAIN_OR_VAL_RE = re.compile(
    r"\b(?:Train Loss|Val (?:MRR|H@1|H@3|H@10))\s+(?:nan|[+-]?inf(?:inity)?)\b",
    re.IGNORECASE,
)
EXPECTED_PROTOCOLS = {"nc": "unified_full_graph_nc_v1", "lp": "unified_sampled_lp_v1"}
FINAL_MODEL_CONFIG = {
    "hidden_dim": 256,
    "max_order": 3,
    "multihop_anchor_alpha": 0.1,
    "edge_weight_min": 0.1,
    "edge_weight_temperature": 0.35,
    "filter_rank": 4,
    "hop_interaction_layers": 1,
    "hop_interaction_heads": 1,
    "relation_bias_init": 0.10,
}
REQUIRED_OUTPUTS = [
    "main.log",
    "train.log",
    "results.json",
    "metrics.json",
    "resolved_config.yaml",
    "resolved_config.json",
    "ablation_manifest.json",
    "best.pt",
    "complete.marker",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def validate_frozen_configs(datasets: list[str]) -> dict[str, str]:
    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    if model_cfg.get("name") != "cosi_mag_final":
        raise ValueError(
            f"Expected model config name 'cosi_mag_final', got {model_cfg.get('name')!r}"
        )
    for key, expected in FINAL_MODEL_CONFIG.items():
        actual = model_cfg.get(key)
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"Frozen model config mismatch for {key}: expected {expected!r}, got {actual!r}"
            )

    for task, path in TASK_CONFIG_PATHS.items():
        task_cfg = load_yaml(path)
        expected_protocol = EXPECTED_PROTOCOLS[task]
        if task_cfg.get("protocol_version") != expected_protocol:
            raise ValueError(
                f"Frozen {task.upper()} protocol mismatch: expected {expected_protocol!r}, "
                f"got {task_cfg.get('protocol_version')!r}"
            )

    for dataset in datasets:
        dataset_path = ROOT / "configs" / "dataset" / f"{dataset}.yaml"
        dataset_cfg = load_yaml(dataset_path)
        if dataset_cfg.get("name") != dataset:
            raise ValueError(
                f"Dataset config name mismatch in {dataset_path}: expected {dataset!r}, "
                f"got {dataset_cfg.get('name')!r}"
            )

    return {
        "model": sha256_file(MODEL_CONFIG_PATH),
        "nc_task": sha256_file(TASK_CONFIG_PATHS["nc"]),
        "lp_task": sha256_file(TASK_CONFIG_PATHS["lp"]),
    }


def unit_dir(task: str, dataset: str) -> Path:
    family = "nc" if task == "nc" else "lp"
    return OUTPUT_ROOT / family / dataset / "runs_42_43_44"


def make_command(dataset: str, task: str, device: str, output_dir: Path) -> list[str]:
    checkpoint = output_dir / "best.pt"
    return [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        f"task={task}",
        "model=cosi_mag_final",
        f"seed={BASE_SEED}",
        f"num_runs={NUM_RUNS}",
        f"device={device}",
        "ablation=full",
        f"task.save_ckpt_path={checkpoint}",
        f"hydra.run.dir={output_dir}",
    ]


def contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_nonfinite(item) for item in value)
    return False


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_is_valid(path: Path, task: str, seed: int) -> tuple[bool, str]:
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # A failed load means the checkpoint is not resumable.
        return False, f"checkpoint cannot be loaded: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        return False, "checkpoint is missing a model_state mapping"
    if payload.get("task") != task or payload.get("seed") != seed:
        return False, "checkpoint task/seed metadata does not match this unit"
    if not payload["model_state"]:
        return False, "checkpoint model_state is empty"
    return True, "checkpoint is readable"


def resolved_config_matches(
    path: Path, dataset: str, task: str, device: str
) -> tuple[bool, str]:
    try:
        config = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"resolved config is unreadable: {exc}"
    if not isinstance(config, dict):
        return False, "resolved config is not a mapping"

    checks = {
        "model.name": (config.get("model", {}).get("name"), "cosi_mag_final"),
        "task.name": (config.get("task", {}).get("name"), task),
        "task.protocol_version": (
            config.get("task", {}).get("protocol_version"),
            EXPECTED_PROTOCOLS[task],
        ),
        "dataset.name": (config.get("dataset", {}).get("name"), dataset),
        "seed (split/base seed)": (config.get("seed"), BASE_SEED),
        "num_runs": (config.get("num_runs"), NUM_RUNS),
        "device": (config.get("device"), device),
        "ablation": (config.get("ablation"), "full"),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            return False, f"resolved config mismatch for {key}: expected {expected!r}, got {actual!r}"

    model_cfg = config.get("model", {})
    for key, expected in FINAL_MODEL_CONFIG.items():
        actual = model_cfg.get(key)
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            return False, f"resolved model config mismatch for {key}: expected {expected!r}, got {actual!r}"
    return True, "resolved config matches frozen benchmark settings"


def completion_status(
    output_dir: Path,
    dataset: str,
    task: str,
    device: str,
    config_hashes: dict[str, str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    run_checkpoints = [output_dir / f"best_run{run_id + 1}.pt" for run_id in range(NUM_RUNS)]
    required = list(REQUIRED_OUTPUTS)
    required.extend(path.name for path in run_checkpoints)
    for name in required:
        if not (output_dir / name).is_file():
            reasons.append(f"missing {name}")

    for log_name in ("main.log", "train.log"):
        log_path = output_dir / log_name
        if log_path.is_file():
            try:
                for line_number, line in enumerate(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    if NONFINITE_TRAIN_OR_VAL_RE.search(line):
                        reasons.append(
                            f"{log_name}:{line_number} contains a non-finite train loss or validation metric"
                        )
                        break
            except OSError as exc:
                reasons.append(f"{log_name} is unreadable: {exc}")

    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = read_json(metrics_path)
            if not isinstance(metrics, dict):
                reasons.append("metrics.json is not a mapping")
            else:
                expected_metrics = {
                    "model": "cosi_mag_final",
                    "task": task,
                    "dataset": dataset,
                    "seed": BASE_SEED,
                }
                for key, expected in expected_metrics.items():
                    if metrics.get(key) != expected:
                        reasons.append(
                            f"metrics.json {key} mismatch: expected {expected!r}, got {metrics.get(key)!r}"
                        )
                if contains_nonfinite(metrics):
                    reasons.append("metrics.json contains NaN/Inf")
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"metrics.json is unreadable: {exc}")

    results_path = output_dir / "results.json"
    if results_path.is_file():
        try:
            if contains_nonfinite(read_json(results_path)):
                reasons.append("results.json contains NaN/Inf")
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"results.json is unreadable: {exc}")

    resolved_path = output_dir / "resolved_config.json"
    if resolved_path.is_file():
        matches, reason = resolved_config_matches(resolved_path, dataset, task, device)
        if not matches:
            reasons.append(reason)

    # The per-unit manifest binds the output to the source config hashes used
    # at launch. If absent, the resolved values above still get audited.
    unit_manifest_path = output_dir / "benchmark_manifest.json"
    if unit_manifest_path.is_file():
        try:
            saved_manifest = read_json(unit_manifest_path)
            saved_hashes = saved_manifest.get("config_sha256") if isinstance(saved_manifest, dict) else None
            if saved_hashes is not None and saved_hashes != config_hashes:
                reasons.append("source config SHA256 differs from the unit manifest")
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"benchmark_manifest.json is unreadable: {exc}")

    marker_path = output_dir / "complete.marker"
    if marker_path.is_file():
        try:
            marker = read_json(marker_path)
            expected_marker = {
                "status": "complete",
                "task": task,
                "dataset": dataset,
                "seed": BASE_SEED,
            }
            if not isinstance(marker, dict):
                reasons.append("complete.marker is not a mapping")
            else:
                for key, expected in expected_marker.items():
                    if marker.get(key) != expected:
                        reasons.append(
                            f"complete.marker {key} mismatch: expected {expected!r}, "
                            f"got {marker.get(key)!r}"
                        )
        except (OSError, json.JSONDecodeError) as exc:
            reasons.append(f"complete.marker is unreadable: {exc}")

    for run_id, checkpoint_path in enumerate(run_checkpoints):
        run_seed = BASE_SEED + run_id
        if checkpoint_path.is_file():
            valid, reason = checkpoint_is_valid(checkpoint_path, task, run_seed)
            if not valid:
                reasons.append(f"{checkpoint_path.name}: {reason}")

    checkpoint_path = output_dir / "best.pt"
    latest_checkpoint = output_dir / f"best_run{NUM_RUNS}.pt"
    if checkpoint_path.is_symlink():
        if checkpoint_path.resolve() != latest_checkpoint.resolve():
            reasons.append("best.pt checkpoint pointer does not target the final run checkpoint")
    elif checkpoint_path.exists():
        reasons.append("best.pt exists but is not the expected checkpoint pointer")
    elif any(path.is_file() for path in run_checkpoints):
        reasons.append("best.pt checkpoint pointer is missing")
    return not reasons, reasons


def build_manifest(
    suite: str,
    task: str,
    datasets: list[str],
    seeds: list[int],
    device: str,
    config_hashes: dict[str, str],
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    dirty = bool(git_value("status", "--porcelain"))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": suite,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit_sha": git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": dirty,
        "config_sha256": config_hashes,
        "model": "cosi_mag_final",
        "frozen_task_protocols": EXPECTED_PROTOCOLS,
        "final_model_config": FINAL_MODEL_CONFIG,
        "task": task,
        "datasets": datasets,
        "base_seed": BASE_SEED,
        "split_seed": BASE_SEED,
        "run_seeds": seeds,
        "device": device,
        "num_runs": NUM_RUNS,
        "ablation": "full",
        "units": units,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_checkpoint_pointer(output_dir: Path) -> tuple[bool, str]:
    """Point main.py's completion check at the last of the three run checkpoints."""
    pointer = output_dir / "best.pt"
    target = Path(f"best_run{NUM_RUNS}.pt")
    if pointer.is_symlink():
        if pointer.resolve() == (output_dir / target).resolve():
            return True, "checkpoint pointer is ready"
        return False, f"preserving unexpected existing symlink: {pointer} -> {pointer.readlink()}"
    if pointer.exists():
        return False, f"preserving existing non-symlink checkpoint path: {pointer}"
    pointer.symlink_to(target)
    return True, "created best.pt pointer to best_run3.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=tuple(SUITES))
    parser.add_argument("--device", default="cuda:0", help="Hydra device value (default: cuda:0)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--resume", action="store_true", help="Skip only validated completed units")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without invoking src.main")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seeds != DEFAULT_SEEDS:
        raise SystemExit("The frozen benchmark uses exactly --seeds 42 43 44")
    if not args.device.strip():
        raise SystemExit("--device must not be empty")

    suite_cfg = SUITES[args.suite]
    task = suite_cfg["task"]
    datasets = suite_cfg["datasets"]
    config_hashes = validate_frozen_configs(datasets)
    plan: list[dict[str, Any]] = []
    for dataset in datasets:
        output_dir = unit_dir(task, dataset)
        command = make_command(dataset, task, args.device, output_dir)
        complete, reasons = completion_status(
            output_dir, dataset, task, args.device, config_hashes
        )
        skip = bool(args.resume and complete)
        if output_dir.exists() and not complete and any(output_dir.iterdir()):
            warnings.warn(
                f"Existing three-run unit is incomplete and will be preserved while launcher-owned files "
                f"are regenerated: {output_dir}. Audit: {'; '.join(reasons) or 'not marked complete'}",
                RuntimeWarning,
            )
        plan.append(
            {
                "dataset": dataset,
                "task": task,
                "base_seed": BASE_SEED,
                "run_seeds": args.seeds,
                "device": args.device,
                "output_dir": str(output_dir),
                "checkpoint": str(output_dir / "best.pt"),
                "resume_skip": skip,
                "resume_audit": reasons,
                "command": command,
            }
        )

    manifest = build_manifest(
        args.suite,
        task,
        datasets,
        args.seeds,
        args.device,
        config_hashes,
        plan,
    )
    manifest_path = OUTPUT_ROOT / "manifests" / f"{args.suite}.json"
    write_json(manifest_path, manifest)
    print("Frozen benchmark manifest:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved manifest: {manifest_path}")

    command_count = 0
    failure_count = 0
    for item in plan:
        output_dir = Path(item["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if item["resume_skip"]:
            print(
                f"[RESUME SKIP] {item['task']} {item['dataset']} "
                f"run_seeds={item['run_seeds']}"
            )
            continue

        per_unit_manifest = {
            **{key: value for key, value in manifest.items() if key != "units"},
            "dataset": item["dataset"],
            "task": item["task"],
            "base_seed": item["base_seed"],
            "split_seed": BASE_SEED,
            "run_seeds": item["run_seeds"],
            "num_runs": NUM_RUNS,
            "device": item["device"],
            "output_dir": item["output_dir"],
            "config_sha256": config_hashes,
        }
        write_json(output_dir / "benchmark_manifest.json", per_unit_manifest)

        command = item["command"]
        if args.dry_run:
            command_count += 1
            print(f"[DRY-RUN] {item['dataset']} run_seeds={item['run_seeds']}")
            print(shlex.join(command))
            continue

        pointer_ready, pointer_message = prepare_checkpoint_pointer(output_dir)
        if not pointer_ready:
            failure_count += 1
            print(f"[FAIL] {pointer_message}")
            continue
        print(f"[CHECKPOINT] {pointer_message}")
        command_count += 1
        print(
            f"[RUN] {item['dataset']} split_seed={BASE_SEED} run_seeds={item['run_seeds']}",
            flush=True,
        )
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            failure_count += 1
            print(f"[FAIL] process exited with status {result.returncode}: {output_dir}")
            continue
        complete, reasons = completion_status(
            output_dir, item["dataset"], task, args.device, config_hashes
        )
        if not complete:
            failure_count += 1
            print(f"[FAIL] run outputs failed audit for {output_dir}: {'; '.join(reasons)}")
        else:
            print(f"[COMPLETE] {item['dataset']} run_seeds={item['run_seeds']}")

    if args.dry_run:
        print(f"Dry-run commands: {command_count}; src.main invocations: 0")
        return 0
    print(f"Launched units: {command_count}; failed audits/processes: {failure_count}")
    return 1 if failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
