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

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
VARIANTS = ["shared_prior", "global_residual", "attribute_residual", "context_residual"]
OUTPUT = ROOT / "outputs/relation_residual_audit_v1"
FORMAL_FIELDS = [
    "run_key", "mode", "variant", "dataset", "seed", "device", "status",
    "return_code", "command", "git_branch", "git_commit", "run_dir", "checkpoint",
    "resolved_config", "git_worktree_clean", "runtime_seconds", "peak_gpu_memory_mib",
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
        root = OUTPUT / ("checkpoints" if self.mode == "formal" else "smoke/checkpoints")
        return root / self.variant / self.dataset / f"seed{self.seed}.pt"

    @property
    def manifest(self) -> Path:
        name = "formal_run_manifest.csv" if self.mode == "formal" else "smoke_run_manifest.csv"
        return OUTPUT / name


def formal_contexts(device: str, edge_chunk_size: int = 65536) -> list[Context]:
    return [
        Context("formal", variant, dataset, seed, 300, device, edge_chunk_size)
        for variant in VARIANTS
        for dataset in DATASETS
        for seed in SEEDS
    ]


def smoke_contexts(
    dataset: str, variants: list[str], seed: int, epochs: int, device: str,
    edge_chunk_size: int = 65536,
) -> list[Context]:
    if dataset not in DATASETS or not variants or set(variants) - set(VARIANTS):
        raise ValueError("smoke dataset/variants must be inside the registered P0-R matrix")
    if epochs < 1:
        raise ValueError("smoke epochs must be positive")
    if len(set(variants)) != len(variants):
        raise ValueError("smoke variants must be unique")
    return [Context("smoke", variant, dataset, seed, epochs, device, edge_chunk_size)
            for variant in variants]


def git_info() -> dict[str, str]:
    def call(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {
        "branch": call("branch", "--show-current"),
        "commit": call("rev-parse", "HEAD"),
        "status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
    }


def require_formal_git_state(info: dict[str, str]) -> None:
    if info["branch"] != "relation_residual_audit":
        raise RuntimeError(f"formal mode requires branch relation_residual_audit, got {info['branch']!r}")
    if info["status"]:
        raise RuntimeError(
            "formal mode requires a clean Git worktree; git status --porcelain:\n"
            + info["status"].rstrip()
        )


def build_command(context: Context) -> list[str]:
    return [
        sys.executable, "-m", "src.main",
        f"dataset={context.dataset}", "task=nc", "model=relation_residual_audit",
        f"seed={context.seed}", "num_runs=1", f"device={context.device}",
        f"model.residual_variant={context.variant}", "model.hidden_dim=256",
        "model.max_order=3", "model.dropout=0.2", "model.relation_dim=64",
        "model.relation_heads=4", "model.relation_ffn_dim=128",
        "model.shared_bottleneck_dim=256", "model.adapter_bottleneck_dim=32",
        "model.num_residual_adapters=3", f"model.edge_chunk_size={context.edge_chunk_size}",
        "task.protocol_version=unified_full_graph_nc_v1", "task.training_mode=full_graph",
        "task.optimizer=adamw", "task.lr=1e-3", "task.weight_decay=1e-4",
        f"task.epochs={context.epochs}", "task.patience=30",
        "task.early_stop_min_epoch=30" if context.mode == "formal" else "task.early_stop_min_epoch=1",
        "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0", "task.eval_every=1",
        "task.inference_mode=full", "task.evaluate_test=false",
        f"task.save_ckpt_path={context.checkpoint}", f"hydra.run.dir={context.run_dir}",
    ]


def read_metrics(context: Context) -> dict | None:
    try:
        return json.loads((context.run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_complete(context: Context) -> bool:
    marker = context.run_dir / "complete.marker"
    resolved = context.run_dir / "resolved_config.json"
    payload = read_metrics(context)
    if not (context.checkpoint.is_file() and marker.is_file() and resolved.is_file() and payload):
        return False
    try:
        config = json.loads(resolved.read_text(encoding="utf-8"))
        checkpoint = __import__("torch").load(
            context.checkpoint, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, EOFError, json.JSONDecodeError):
        return False

    if not isinstance(config, dict) or not isinstance(checkpoint, dict):
        return False
    if any(not isinstance(config.get(section), dict)
           for section in ("model", "task", "dataset")):
        return False
    model_cfg, task_cfg = config["model"], config["task"]
    expected_model = {
        "name": "relation_residual_audit", "residual_variant": context.variant,
        "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
        "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
        "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
        "num_residual_adapters": 3, "edge_chunk_size": context.edge_chunk_size,
    }
    if any(model_cfg.get(key) != value for key, value in expected_model.items()):
        return False
    expected_task = {
        "name": "nc", "protocol_version": "unified_full_graph_nc_v1",
        "training_mode": "full_graph", "optimizer": "adamw",
        "epochs": context.epochs, "patience": 30,
        "early_stop_min_epoch": 30 if context.mode == "formal" else 1,
        "eval_every": 1, "inference_mode": "full", "evaluate_test": False,
        "scheduler": None,
    }
    if any(task_cfg.get(key) != value for key, value in expected_task.items()):
        return False
    try:
        numeric_expected = {"lr": 1e-3, "weight_decay": 1e-4,
                           "early_stop_min_delta": 1e-4, "grad_clip": 1.0}
        if any(abs(float(task_cfg.get(key)) - value) > 1e-12
               for key, value in numeric_expected.items()):
            return False
        valid_epoch = 1 <= int(checkpoint.get("epoch", 0)) <= context.epochs
    except (TypeError, ValueError):
        return False

    def contains_test_key(value) -> bool:
        if isinstance(value, dict):
            return any("test" in str(key).lower() or contains_test_key(child)
                       for key, child in value.items())
        if isinstance(value, (list, tuple)):
            return any(contains_test_key(child) for child in value)
        return False

    metrics = payload.get("metrics", {})
    checkpoint_metrics = checkpoint.get("metrics", {})
    return (
        config.get("dataset", {}).get("name") == context.dataset
        and int(config.get("seed", -1)) == context.seed
        and int(config.get("num_runs", -1)) == 1
        and checkpoint.get("task") == "nc"
        and checkpoint.get("seed") == context.seed
        and checkpoint.get("selection") == "best_val_accuracy"
        and valid_epoch
        and payload.get("best_epoch") == checkpoint.get("epoch")
        and payload.get("checkpoint_selection") == "best_val_accuracy"
        and {"val_acc", "val_macro_f1"}.issubset(checkpoint_metrics)
        and {"val_acc", "val_macro_f1"}.issubset(metrics)
        and isinstance(checkpoint.get("model_state"), dict) and bool(checkpoint["model_state"])
        and isinstance(checkpoint.get("head_state"), dict) and bool(checkpoint["head_state"])
        and isinstance(checkpoint.get("data_info"), dict)
        and not contains_test_key(metrics)
        and not contains_test_key(checkpoint_metrics)
    )


def upsert_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [old for old in rows if old.get("run_key") != row["run_key"]]
    rows.append({key: row.get(key, "") for key in FORMAL_FIELDS})
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORMAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def run_context(context: Context, info: dict[str, str], *, dry_run: bool = False) -> dict:
    command = build_command(context)
    command_text = shlex.join(command)
    row = {
        "run_key": context.run_key, "mode": context.mode, "variant": context.variant,
        "dataset": context.dataset, "seed": context.seed, "device": context.device,
        "command": command_text, "git_branch": info["branch"], "git_commit": info["commit"],
        "run_dir": str(context.run_dir), "checkpoint": str(context.checkpoint),
        "resolved_config": str(context.run_dir / "resolved_config.json"),
        "git_worktree_clean": str(not bool(info["status"])).lower(),
        "test_evaluated": "false",
    }
    if dry_run:
        print(command_text)
        return {**row, "status": "dry_run"}
    if is_complete(context):
        payload = read_metrics(context) or {}
        row.update(
            status="already_complete", return_code=0,
            runtime_seconds=payload.get("runtime_seconds", ""),
            peak_gpu_memory_mib=payload.get("peak_gpu_memory_mib", ""),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        upsert_manifest(context.manifest, row)
        print(f"[skip complete] {context.run_key}", flush=True)
        return row

    context.run_dir.mkdir(parents=True, exist_ok=True)
    context.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for stale in (context.run_dir / "complete.marker", context.run_dir / "metrics.json", context.checkpoint):
        stale.unlink(missing_ok=True)
    log_path = context.run_dir / "launcher.log"
    row.update(status="running", updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    upsert_manifest(context.manifest, row)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    elapsed = time.perf_counter() - started
    payload = read_metrics(context)
    if proc.returncode == 0 and is_complete(context):
        row.update(
            status="complete", return_code=0,
            runtime_seconds=(payload or {}).get("runtime_seconds", elapsed),
            peak_gpu_memory_mib=(payload or {}).get("peak_gpu_memory_mib", ""),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
    else:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        reason = f"training failed (exit {proc.returncode}) or completion/isolation check failed"
        row.update(
            status="failed", return_code=int(proc.returncode), runtime_seconds=elapsed,
            error_reason=(reason + (f"; final log line: {tail.splitlines()[-1]}" if tail else "")),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
    upsert_manifest(context.manifest, row)
    print(f"[{row['status']}] {context.run_key}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="P0-R shared-prior residual NC launcher")
    parser.add_argument("--mode", choices=("formal", "smoke"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1, help="Smoke epochs; formal is fixed at 300")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.edge_chunk_size < 1:
        raise ValueError("edge chunk size must be positive")
    info = git_info()
    if args.mode == "formal":
        if args.dataset is not None or args.variants is not None:
            raise ValueError("formal mode always uses the complete 4x5x3 matrix")
        contexts = formal_contexts(args.device, args.edge_chunk_size)
        if len(contexts) != 60 or len({context.run_key for context in contexts}) != 60:
            raise RuntimeError("formal matrix must contain 60 unique contexts")
        if not args.dry_run:
            require_formal_git_state(info)
    else:
        if args.dataset is None or args.variants is None:
            raise ValueError("smoke mode requires --dataset and one or more --variants")
        contexts = smoke_contexts(
            args.dataset, args.variants, args.seed, args.epochs, args.device, args.edge_chunk_size
        )
    results = [run_context(context, info, dry_run=args.dry_run) for context in contexts]
    if args.mode == "formal" and args.dry_run:
        print(f"FORMAL_MATRIX_CONTEXTS={len(results)}")
    if not args.dry_run and any(row["status"] == "failed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
