from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
VARIANTS = ["plain", "scalar_weight", "global_expert", "relation_expert"]
OUTPUT = ROOT / "outputs/relation_operator_audit_v1"
FORMAL_FIELDS = [
    "run_key", "mode", "variant", "dataset", "seed", "device", "status",
    "return_code", "command", "git_commit", "run_dir", "checkpoint",
    "resolved_config", "runtime_seconds", "peak_gpu_memory_mib",
    "test_evaluated", "error_reason", "updated_at",
]


@dataclass(frozen=True)
class Context:
    mode: str
    variant: str
    dataset: str
    seed: int
    epochs: int
    device: str
    edge_chunk_size: int

    @property
    def run_key(self) -> str:
        return f"{self.mode}/{self.variant}/{self.dataset}/seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        if self.mode == "formal":
            return OUTPUT / "runs" / self.variant / self.dataset / f"seed{self.seed}"
        return OUTPUT / "smoke" / self.variant / self.dataset / f"seed{self.seed}"

    @property
    def checkpoint(self) -> Path:
        if self.mode == "formal":
            root = OUTPUT / "checkpoints"
        else:
            root = OUTPUT / "smoke" / "checkpoints"
        return root / self.variant / self.dataset / f"seed{self.seed}.pt"

    @property
    def manifest(self) -> Path:
        return OUTPUT / ("formal_run_manifest.csv" if self.mode == "formal" else "smoke_run_manifest.csv")


def formal_contexts(device: str, edge_chunk_size: int = 65536) -> list[Context]:
    return [
        Context("formal", variant, dataset, seed, 300, device, edge_chunk_size)
        for variant in VARIANTS
        for dataset in DATASETS
        for seed in SEEDS
    ]


def smoke_contexts(
    dataset: str,
    variant: str,
    seed: int,
    epochs: int,
    device: str,
    edge_chunk_size: int,
) -> list[Context]:
    if dataset not in DATASETS or variant not in VARIANTS:
        raise ValueError("smoke dataset or variant is outside the registered P0 matrix")
    if epochs < 1:
        raise ValueError("smoke epochs must be positive")
    return [Context("smoke", variant, dataset, seed, epochs, device, edge_chunk_size)]


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def build_command(context: Context) -> list[str]:
    return [
        sys.executable, "-m", "src.main",
        f"dataset={context.dataset}",
        "task=nc",
        "model=relation_operator_audit",
        f"seed={context.seed}",
        "num_runs=1",
        f"device={context.device}",
        f"model.operator_variant={context.variant}",
        "model.hidden_dim=256",
        "model.max_order=3",
        f"model.edge_chunk_size={context.edge_chunk_size}",
        "task.protocol_version=unified_full_graph_nc_v1",
        "task.training_mode=full_graph",
        "task.optimizer=adamw",
        "task.lr=1e-3",
        "task.weight_decay=1e-4",
        f"task.epochs={context.epochs}",
        "task.patience=30",
        "task.early_stop_min_epoch=30" if context.mode == "formal" else "task.early_stop_min_epoch=1",
        "task.early_stop_min_delta=1e-4",
        "task.grad_clip=1.0",
        "task.eval_every=1",
        "task.inference_mode=full",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={context.checkpoint}",
        f"hydra.run.dir={context.run_dir}",
    ]


def _read_metrics(context: Context) -> dict | None:
    path = context.run_dir / "metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _is_complete(context: Context) -> bool:
    marker = context.run_dir / "complete.marker"
    resolved = context.run_dir / "resolved_config.json"
    payload = _read_metrics(context)
    if not (context.checkpoint.is_file() and marker.is_file() and resolved.is_file() and payload):
        return False
    try:
        config = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    metrics = payload.get("metrics", {})
    try:
        checkpoint = torch.load(context.checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError):
        return False
    return (
        config.get("dataset", {}).get("name") == context.dataset
        and config.get("model", {}).get("name") == "relation_operator_audit"
        and config.get("model", {}).get("operator_variant") == context.variant
        and int(config.get("seed", -1)) == context.seed
        and int(config.get("num_runs", -1)) == 1
        and int(config.get("model", {}).get("hidden_dim", -1)) == 256
        and int(config.get("model", {}).get("max_order", -1)) == 3
        and int(config.get("model", {}).get("edge_chunk_size", -1)) == context.edge_chunk_size
        and int(config.get("task", {}).get("epochs", -1)) == context.epochs
        and config.get("task", {}).get("evaluate_test") is False
        and not any("test" in key.lower() for key in metrics)
        and not any("test" in key.lower() for key in checkpoint.get("metrics", {}))
        and checkpoint.get("seed") == context.seed
        and checkpoint.get("selection") == "best_val_accuracy"
        and payload.get("best_epoch") is not None
    )


def _append_manifest(context: Context, row: dict) -> None:
    context.manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = context.manifest.is_file()
    with context.manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORMAL_FIELDS, extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FORMAL_FIELDS})


def _tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def run_context(context: Context, git_commit: str, *, dry_run: bool = False) -> dict:
    command = build_command(context)
    command_text = shlex.join(command)
    row = {
        "run_key": context.run_key,
        "mode": context.mode,
        "variant": context.variant,
        "dataset": context.dataset,
        "seed": context.seed,
        "device": context.device,
        "command": command_text,
        "git_commit": git_commit,
        "run_dir": str(context.run_dir),
        "checkpoint": str(context.checkpoint),
        "resolved_config": str(context.run_dir / "resolved_config.json"),
        "test_evaluated": "false",
    }
    if dry_run:
        print(command_text)
        return {**row, "status": "dry_run"}

    if _is_complete(context):
        payload = _read_metrics(context) or {}
        row.update(
            status="already_complete",
            return_code=0,
            runtime_seconds=payload.get("runtime_seconds", ""),
            peak_gpu_memory_mib=payload.get("peak_gpu_memory_mib", ""),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        _append_manifest(context, row)
        print(f"[skip complete] {context.run_key}", flush=True)
        return row

    OUTPUT.mkdir(parents=True, exist_ok=True)
    context.run_dir.mkdir(parents=True, exist_ok=True)
    context.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Remove completion artifacts from an incomplete prior attempt so a failed
    # rerun cannot be mistaken for a successful context.
    for stale in (context.run_dir / "complete.marker", context.run_dir / "metrics.json", context.checkpoint):
        stale.unlink(missing_ok=True)
    log_path = context.run_dir / "launcher.log"
    row.update(status="running", updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    _append_manifest(context, row)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    wall_seconds = time.perf_counter() - started
    payload = _read_metrics(context)
    if proc.returncode == 0 and _is_complete(context):
        row.update(
            status="complete",
            return_code=0,
            runtime_seconds=(payload or {}).get("runtime_seconds", wall_seconds),
            peak_gpu_memory_mib=(payload or {}).get("peak_gpu_memory_mib", ""),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
    else:
        log_tail = _tail(log_path)
        failure_reason = (
            f"training command failed (exit {proc.returncode})"
            if proc.returncode
            else "required completion artifact missing or test isolation check failed"
        )
        if log_tail:
            failure_reason = f"{failure_reason}; final log lines: {log_tail.splitlines()[-1]}"
        failure = {
            "run_key": context.run_key,
            "return_code": int(proc.returncode),
            "reason": failure_reason,
            "log_tail": log_tail,
            "command": command,
            "git_commit": git_commit,
            "runtime_seconds": wall_seconds,
        }
        failure_path = context.run_dir / "failure.json"
        failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        row.update(
            status="failed",
            return_code=int(proc.returncode or 97),
            runtime_seconds=wall_seconds,
            error_reason=failure_reason,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
    _append_manifest(context, row)
    print(f"[{row['status']}] {context.run_key}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated P0 relation operator audit.")
    parser.add_argument("--mode", choices=["formal", "smoke"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=DATASETS, default="Movies", help="smoke mode only")
    parser.add_argument("--variant", choices=VARIANTS, default="plain", help="smoke mode only")
    parser.add_argument("--seed", type=int, default=42, help="smoke mode only")
    parser.add_argument("--epochs", type=int, default=1, help="smoke mode only")
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
        device_index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index does not exist: {args.device}")
    if args.mode == "formal":
        contexts = formal_contexts(args.device, args.edge_chunk_size)
        if len(contexts) != 60:
            raise RuntimeError(f"formal P0 matrix must have 60 contexts, got {len(contexts)}")
    else:
        contexts = smoke_contexts(
            args.dataset, args.variant, args.seed, args.epochs,
            args.device, args.edge_chunk_size,
        )

    git_commit = _git_commit()
    if args.dry_run:
        print(f"mode={args.mode} contexts={len(contexts)} git_commit={git_commit}")
        for context in contexts:
            run_context(context, git_commit, dry_run=True)
        return

    failures = []
    for context in contexts:
        row = run_context(context, git_commit)
        if row.get("status") == "failed":
            failures.append((context.run_key, row.get("return_code"), row.get("error_reason")))
    if failures:
        raise RuntimeError("P0 run failures: " + "; ".join(map(str, failures)))
    print(f"Finished {len(contexts)} {args.mode} context(s).", flush=True)


if __name__ == "__main__":
    main()
