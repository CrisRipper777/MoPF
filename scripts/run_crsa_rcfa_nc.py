from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
SMOKE_VARIANTS = ("crsa_ref", "full", "wo_relation_condition")
OUTPUT = ROOT / "outputs/crsa_rcfa_v2"
MANIFEST_FIELDS = [
    "run_key", "mode", "variant", "dataset", "seed", "device", "status",
    "return_code", "command", "git_branch", "git_commit", "git_worktree_clean",
    "run_dir", "checkpoint", "resolved_config", "runtime_seconds",
    "peak_gpu_memory_mib", "test_evaluated", "error_reason", "updated_at",
]


@dataclass(frozen=True)
class RunContext:
    mode: str
    variant: str
    dataset: str
    seed: int
    epochs: int
    device: str
    edge_chunk_size: int = 65536
    rcfa_node_chunk_size: int = 4096

    @property
    def run_key(self) -> str:
        return f"{self.mode}/{self.variant}/{self.dataset}/seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        prefix = "runs" if self.mode == "formal" else "smoke/runs"
        return OUTPUT / prefix / self.variant / self.dataset / f"seed{self.seed}"

    @property
    def checkpoint(self) -> Path:
        prefix = "checkpoints" if self.mode == "formal" else "smoke/checkpoints"
        return OUTPUT / prefix / self.variant / self.dataset / f"seed{self.seed}.pt"

    @property
    def manifest(self) -> Path:
        name = "formal_run_manifest.csv" if self.mode == "formal" else "smoke_run_manifest.csv"
        return OUTPUT / name

    @property
    def use_rcfa(self) -> bool:
        return self.variant != "crsa_ref"

    @property
    def use_relation_condition(self) -> bool:
        return self.variant != "wo_relation_condition"


def formal_contexts(device: str, edge_chunk_size: int = 65536,
                    rcfa_node_chunk_size: int = 4096) -> list[RunContext]:
    return [
        RunContext("formal", "full", dataset, seed, 300, device,
                   edge_chunk_size, rcfa_node_chunk_size)
        for dataset in DATASETS for seed in SEEDS
    ]


def smoke_contexts(dataset: str, variants: list[str], seed: int, epochs: int,
                   device: str, edge_chunk_size: int = 65536,
                   rcfa_node_chunk_size: int = 4096) -> list[RunContext]:
    if dataset not in DATASETS or not variants or set(variants) - set(SMOKE_VARIANTS):
        raise ValueError("smoke dataset and variants must be inside the prescribed smoke set")
    if len(set(variants)) != len(variants):
        raise ValueError("smoke variants must be unique")
    if epochs < 1 or epochs > 3:
        raise ValueError("smoke epochs must be between one and three")
    return [
        RunContext("smoke", variant, dataset, seed, epochs, device,
                   edge_chunk_size, rcfa_node_chunk_size)
        for variant in variants
    ]


def smoke_suite(device: str, edge_chunk_size: int = 65536,
                rcfa_node_chunk_size: int = 4096) -> list[RunContext]:
    contexts = smoke_contexts(
        "Movies", ["crsa_ref", "full", "wo_relation_condition"], 42, 2,
        device, edge_chunk_size, rcfa_node_chunk_size,
    )
    contexts.extend(smoke_contexts(
        "ele-fashion", ["full"], 42, 1, device, edge_chunk_size, rcfa_node_chunk_size
    ))
    contexts.extend(smoke_contexts(
        "Reddit-S", ["full"], 42, 1, device, edge_chunk_size, rcfa_node_chunk_size
    ))
    return contexts


def git_info() -> dict[str, str]:
    def call(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {
        "branch": call("branch", "--show-current"),
        "commit": call("rev-parse", "HEAD"),
        "status": status,
        "clean": "true" if not status else "false",
    }


def require_formal_git_state(info: dict[str, str]) -> None:
    if info["branch"] != "crsa_rcfa_v2":
        raise RuntimeError(f"formal mode requires branch crsa_rcfa_v2, got {info['branch']!r}")
    if info["status"]:
        raise RuntimeError(
            "formal mode requires a clean Git worktree; git status --porcelain:\n"
            + info["status"].rstrip()
        )


def build_command(context: RunContext) -> list[str]:
    return [
        sys.executable, "-m", "src.main",
        f"dataset={context.dataset}", "task=nc", "model=crsa_rcfa",
        f"seed={context.seed}", "num_runs=1", f"device={context.device}",
        "model.hidden_dim=256", "model.max_order=3", "model.dropout=0.2",
        "model.relation_dim=64", "model.relation_heads=4",
        "model.relation_ffn_dim=128", "model.shared_bottleneck_dim=256",
        "model.adapter_bottleneck_dim=32", "model.num_residual_adapters=3",
        f"model.edge_chunk_size={context.edge_chunk_size}",
        f"model.rcfa_node_chunk_size={context.rcfa_node_chunk_size}",
        "model.rcfa_hidden_dim=256",
        f"model.use_rcfa={str(context.use_rcfa).lower()}",
        f"model.use_relation_condition={str(context.use_relation_condition).lower()}",
        "task.protocol_version=unified_full_graph_nc_v1",
        "task.training_mode=full_graph", "task.optimizer=adamw",
        "task.lr=1e-3", "task.weight_decay=1e-4",
        f"task.epochs={context.epochs}", "task.patience=30",
        "task.early_stop_min_epoch=30" if context.mode == "formal"
        else "task.early_stop_min_epoch=1",
        "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0",
        "task.eval_every=1", "task.inference_mode=full",
        "task.scheduler=null", "task.evaluate_test=false",
        "task.eval_modality_masks.enabled=false", "task.loss.aux_weight=0.0",
        f"task.save_ckpt_path={context.checkpoint}",
        f"hydra.run.dir={context.run_dir}",
    ]


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def contains_test_key(value) -> bool:
    if isinstance(value, dict):
        return any("test" in str(key).lower() or contains_test_key(child)
                   for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_test_key(child) for child in value)
    return False


def checkpoint_is_finite(payload: dict) -> bool:
    import torch

    for name in ("model_state", "head_state"):
        state = payload.get(name)
        if not isinstance(state, dict) or not state:
            return False
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor) and (tensor.is_floating_point() or tensor.is_complex()):
                if not bool(torch.isfinite(tensor).all()):
                    return False
    return True


def finite_training_log(run_dir: Path) -> bool:
    log = run_dir / "train.log"
    if not log.is_file():
        log = run_dir / "main.log"
    if not log.is_file():
        return False
    losses = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Train Loss" not in line:
            continue
        try:
            value = float(line.split("Train Loss", 1)[1].strip().split()[0])
        except (IndexError, ValueError):
            return False
        losses.append(value)
    return bool(losses) and all(math.isfinite(value) for value in losses)


def audit_context(context: RunContext) -> tuple[bool, str, dict | None]:
    import torch
    from omegaconf import OmegaConf
    from src.models.crsa_rcfa import Model

    config_path = context.run_dir / "resolved_config.json"
    metrics_path = context.run_dir / "metrics.json"
    marker_path = context.run_dir / "complete.marker"
    if not all(path.is_file() for path in (config_path, metrics_path, marker_path, context.checkpoint)):
        return False, "missing resolved config, metrics, marker, or checkpoint", None
    config, metrics_payload, marker = (
        read_json(config_path), read_json(metrics_path), read_json(marker_path)
    )
    if not all(isinstance(item, dict) for item in (config, metrics_payload, marker)):
        return False, "malformed run metadata", None
    task = config.get("task", {})
    model_cfg = config.get("model", {})
    metrics = metrics_payload.get("metrics", {})
    try:
        checkpoint = torch.load(context.checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        return False, f"checkpoint load failed: {exc}", None
    if not isinstance(checkpoint, dict):
        return False, "checkpoint payload is not a mapping", None

    checks = [
        (task.get("evaluate_test") is False, "resolved config enabled Test evaluation"),
        (task.get("eval_modality_masks", {}).get("enabled") is False,
         "modality-mask evaluation was enabled"),
        (not contains_test_key(metrics), "test metric key found in metrics"),
        (not contains_test_key(checkpoint.get("metrics", {})), "test metric key found in checkpoint"),
        (model_cfg.get("name") == "crsa_rcfa", "wrong model in resolved config"),
        (model_cfg.get("use_rcfa") is context.use_rcfa, "resolved RCFA flag does not match context"),
        (model_cfg.get("use_relation_condition") is context.use_relation_condition,
         "resolved relation-condition flag does not match context"),
        (model_cfg.get("max_order") == 3, "formal/smoke run did not use K=3"),
        (config.get("dataset", {}).get("name") == context.dataset, "dataset mismatch"),
        (int(config.get("seed", -1)) == context.seed, "seed mismatch"),
        (int(config.get("num_runs", -1)) == 1, "num_runs must equal one"),
        (metrics_payload.get("task") == "nc", "metrics task is not NC"),
        (metrics_payload.get("dataset") == context.dataset, "metrics dataset mismatch"),
        (metrics_payload.get("seed") == context.seed, "metrics seed mismatch"),
        (metrics_payload.get("model") == "crsa_rcfa", "metrics model mismatch"),
        (marker.get("status") == "complete", "completion marker missing"),
        (checkpoint.get("task") == "nc", "checkpoint task is not NC"),
        (checkpoint.get("seed") == context.seed, "checkpoint seed mismatch"),
        (checkpoint.get("selection") == "best_val_accuracy", "wrong checkpoint selection"),
        (checkpoint.get("epoch") == metrics_payload.get("best_epoch"), "checkpoint epoch mismatch"),
        (checkpoint_is_finite(checkpoint), "checkpoint contains non-finite tensors"),
        (finite_training_log(context.run_dir), "training log has no finite loss records"),
    ]
    for passed, reason in checks:
        if not passed:
            return False, reason, metrics_payload

    expected_task = {
        "name": "nc", "protocol_version": "unified_full_graph_nc_v1",
        "training_mode": "full_graph", "optimizer": "adamw",
        "lr": 1e-3, "weight_decay": 1e-4, "patience": 30,
        "early_stop_min_epoch": 30 if context.mode == "formal" else 1,
        "early_stop_min_delta": 1e-4, "grad_clip": 1.0, "eval_every": 1,
        "inference_mode": "full", "scheduler": None,
    }
    if any(task.get(key) != value for key, value in expected_task.items()):
        return False, "resolved task protocol mismatch", metrics_payload
    if context.mode == "formal" and task.get("epochs") != 300:
        return False, "formal run did not use 300 epochs", metrics_payload
    if any(model_cfg.get(key) != value for key, value in {
        "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
        "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
        "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
        "num_residual_adapters": 3, "rcfa_hidden_dim": 256,
    }.items()):
        return False, "resolved model dimensions mismatch", metrics_payload
    if int(model_cfg.get("edge_chunk_size", 0)) < 1 or int(model_cfg.get("rcfa_node_chunk_size", 0)) < 1:
        return False, "invalid exact chunk sizes", metrics_payload
    for key in ("val_acc", "val_macro_f1"):
        try:
            value = float(metrics[key]["mean"])
        except (KeyError, TypeError, ValueError):
            return False, f"missing validation metric {key}", metrics_payload
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return False, f"invalid validation metric {key}", metrics_payload
    data_info = checkpoint.get("data_info")
    if not isinstance(data_info, dict):
        return False, "checkpoint missing data_info", metrics_payload
    try:
        model = Model(OmegaConf.create(config), data_info)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        head = torch.nn.Linear(model.out_dim, int(data_info["num_classes"]))
        head.load_state_dict(checkpoint["head_state"], strict=True)
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        return False, f"strict checkpoint load failed: {exc}", metrics_payload
    return True, "", metrics_payload


def is_complete(context: RunContext) -> bool:
    return audit_context(context)[0]


def upsert_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [item for item in rows if item.get("run_key") != row.get("run_key")]
    rows.append({**{field: "" for field in MANIFEST_FIELDS}, **row})
    rows.sort(key=lambda item: item["run_key"])
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: item.get(key, "") for key in MANIFEST_FIELDS} for item in rows)
    temp_path.replace(path)


def run_context(context: RunContext, git: dict[str, str], *, dry_run: bool, force: bool = False) -> dict:
    command = build_command(context)
    row = {
        "run_key": context.run_key, "mode": context.mode, "variant": context.variant,
        "dataset": context.dataset, "seed": context.seed, "device": context.device,
        "git_branch": git["branch"], "git_commit": git["commit"],
        "git_worktree_clean": git["clean"], "run_dir": str(context.run_dir),
        "checkpoint": str(context.checkpoint),
        "resolved_config": str(context.run_dir / "resolved_config.json"),
        "test_evaluated": "false", "command": shlex.join(command),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if dry_run:
        print(f"{context.run_key}: {shlex.join(command)}", flush=True)
        return {**row, "status": "dry_run"}

    context.run_dir.mkdir(parents=True, exist_ok=True)
    context.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not force and is_complete(context):
        row.update(status="complete", return_code=0, error_reason="skipped: complete artifacts already audited")
        print(f"[skip complete] {context.run_key}", flush=True)
        upsert_manifest(context.manifest, row)
        return row

    row.update(status="running", return_code="", updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    upsert_manifest(context.manifest, row)
    output_log = context.run_dir / "launcher_output.log"
    started = time.perf_counter()
    with output_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - started
    valid, reason, metrics_payload = audit_context(context)
    if proc.returncode == 0 and valid:
        row.update(
            status="complete", return_code=0,
            runtime_seconds=(metrics_payload or {}).get("runtime_seconds", elapsed),
            peak_gpu_memory_mib=(metrics_payload or {}).get("peak_gpu_memory_mib", ""),
            error_reason="", updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
    else:
        tail = output_log.read_text(encoding="utf-8", errors="replace").splitlines()
        detail = f"training exit={proc.returncode}; artifact audit: {reason}"
        if tail:
            detail += f"; last output: {tail[-1]}"
        row.update(
            status="failed", return_code=proc.returncode, runtime_seconds=elapsed,
            error_reason=detail, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
    upsert_manifest(context.manifest, row)
    print(f"[{row['status']}] {context.run_key}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="CRSA+RSE+RCFA v2 NC launcher")
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--edge-chunk-size", type=int, default=65536)
    parser.add_argument("--rcfa-node-chunk-size", type=int, default=4096)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variants", nargs="+", choices=SMOKE_VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--smoke-suite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun otherwise complete smoke contexts")
    args = parser.parse_args()
    if args.edge_chunk_size < 1 or args.rcfa_node_chunk_size < 1:
        raise ValueError("chunk sizes must be positive")
    git = git_info()
    if args.force and args.mode != "smoke":
        raise ValueError("--force is only valid for smoke contexts")

    if args.mode == "formal":
        if args.dataset is not None or args.variants is not None or args.smoke_suite:
            raise ValueError("formal mode always uses the fixed 5x3 Full-v2 matrix")
        contexts = formal_contexts(args.device, args.edge_chunk_size, args.rcfa_node_chunk_size)
        if len(contexts) != 15 or len({context.run_key for context in contexts}) != 15:
            raise RuntimeError("formal matrix must contain exactly 15 unique contexts")
        if not args.dry_run:
            require_formal_git_state(git)
    elif args.smoke_suite:
        if args.dataset is not None or args.variants is not None:
            raise ValueError("--smoke-suite owns its fixed context list")
        contexts = smoke_suite(args.device, args.edge_chunk_size, args.rcfa_node_chunk_size)
    else:
        if args.dataset is None or args.variants is None:
            raise ValueError("smoke mode requires --dataset and --variants, or use --smoke-suite")
        contexts = smoke_contexts(
            args.dataset, args.variants, args.seed, args.epochs, args.device,
            args.edge_chunk_size, args.rcfa_node_chunk_size,
        )

    print(f"branch={git['branch']} source_commit={git['commit']} worktree_clean={git['clean']}")
    print(f"matrix_contexts={len(contexts)} mode={args.mode}")
    rows = [run_context(context, git, dry_run=args.dry_run, force=args.force) for context in contexts]
    if args.mode == "formal" and args.dry_run:
        print("FORMAL_MATRIX_CONTEXTS=15")
    if not args.dry_run and any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
