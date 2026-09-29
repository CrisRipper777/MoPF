from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
BRANCH = "crsa_icsr_v3"
OUTPUT = ROOT / "outputs/crsa_icsr_v3"
MANIFEST_FIELDS = [
    "run_key", "mode", "variant", "dataset", "seed", "epochs", "device", "status",
    "return_code", "command", "git_branch", "git_commit", "git_worktree_clean",
    "run_dir", "checkpoint", "resolved_config", "runtime_seconds", "peak_gpu_memory_mib",
    "test_evaluated", "error_reason", "updated_at",
]


@dataclass(frozen=True)
class RunContext:
    dataset: str
    seed: int
    mode: str = "formal"
    epochs: int = 300
    device: str = "cuda:1"
    edge_chunk_size: int = 65536
    node_chunk_size: int = 4096

    @property
    def run_key(self) -> str:
        return f"{self.mode}/full/{self.dataset}/seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        if self.mode == "formal":
            return OUTPUT / "runs/full" / self.dataset / f"seed{self.seed}"
        return OUTPUT / "runs/smoke/full" / self.dataset / f"seed{self.seed}"

    @property
    def checkpoint(self) -> Path:
        if self.mode == "formal":
            return OUTPUT / "checkpoints/full" / self.dataset / f"seed{self.seed}.pt"
        return OUTPUT / "checkpoints/smoke/full" / self.dataset / f"seed{self.seed}.pt"


def formal_contexts(device: str) -> list[RunContext]:
    return [RunContext(dataset, seed, device=device) for dataset in DATASETS for seed in SEEDS]


def smoke_contexts(device: str) -> list[RunContext]:
    return [
        RunContext("Movies", 42, "smoke", 2, device),
        RunContext("ele-fashion", 42, "smoke", 1, device),
        RunContext("Reddit-S", 42, "smoke", 1, device),
    ]


def git_info() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    porcelain = git("status", "--porcelain")
    return {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
            "status": porcelain, "clean": "true" if not porcelain else "false"}


def build_command(context: RunContext) -> list[str]:
    min_epoch = 30 if context.mode == "formal" else 1
    return [
        sys.executable, "-m", "src.main", f"dataset={context.dataset}", "task=nc",
        "model=crsa_icsr", f"seed={context.seed}", "num_runs=1", f"device={context.device}",
        "model.hidden_dim=256", "model.max_order=3", "model.dropout=0.2",
        "model.relation_dim=64", "model.relation_heads=4", "model.relation_ffn_dim=128",
        "model.shared_bottleneck_dim=256", "model.adapter_bottleneck_dim=32",
        "model.num_residual_adapters=3", f"model.edge_chunk_size={context.edge_chunk_size}",
        "model.use_icsr=true", "model.icsr_bottleneck_dim=32",
        f"model.icsr_node_chunk_size={context.node_chunk_size}",
        "task.protocol_version=unified_full_graph_nc_v1", "task.training_mode=full_graph",
        "task.optimizer=adamw", "task.lr=1e-3", "task.weight_decay=1e-4",
        f"task.epochs={context.epochs}", "task.patience=30",
        f"task.early_stop_min_epoch={min_epoch}", "task.early_stop_min_delta=1e-4",
        "task.grad_clip=1.0", "task.eval_every=1", "task.inference_mode=full",
        "task.scheduler=null", "task.evaluate_test=false",
        "task.eval_modality_masks.enabled=false", "task.loss.aux_weight=0.0",
        f"task.save_ckpt_path={context.checkpoint}", f"hydra.run.dir={context.run_dir}",
    ]


def read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except (OSError, json.JSONDecodeError):
        return None


def contains_test_key(value) -> bool:
    if isinstance(value, dict):
        return any("test" in str(k).lower() or contains_test_key(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_test_key(v) for v in value)
    return False


def finite_state(checkpoint: dict) -> bool:
    import torch
    for field in ("model_state", "head_state"):
        state = checkpoint.get(field)
        if not isinstance(state, dict) or not state:
            return False
        for value in state.values():
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                if not torch.isfinite(value).all().item():
                    return False
    return True


def finite_losses(run_dir: Path) -> bool:
    logs = [run_dir / "main.log", run_dir / "train.log"]
    path = next((p for p in logs if p.is_file()), None)
    if path is None:
        return False
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Train Loss" not in line:
            continue
        match = re.search(r"Train Loss\s+([^\s]+)", line)
        if not match:
            return False
        try:
            values.append(float(match.group(1)))
        except ValueError:
            return False
    return bool(values) and all(math.isfinite(x) for x in values)


def audit_context(context: RunContext) -> tuple[bool, str, dict | None]:
    import torch
    from omegaconf import OmegaConf
    from src.models.crsa_icsr import Model

    config_path = context.run_dir / "resolved_config.json"
    metrics_path = context.run_dir / "metrics.json"
    marker_path = context.run_dir / "complete.marker"
    if not all(p.is_file() for p in (config_path, metrics_path, marker_path, context.checkpoint)):
        return False, "missing config, metrics, completion marker, or checkpoint", None
    config, payload, marker = (read_json(config_path), read_json(metrics_path), read_json(marker_path))
    if not all(isinstance(x, dict) for x in (config, payload, marker)):
        return False, "malformed run metadata", None
    try:
        checkpoint = torch.load(context.checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        return False, f"checkpoint load failed: {exc}", payload
    if not isinstance(checkpoint, dict):
        return False, "checkpoint is not a mapping", payload

    task, mc, dataset_cfg = config.get("task", {}), config.get("model", {}), config.get("dataset", {})
    metrics = payload.get("metrics", {})
    expected_task = {
        "name": "nc", "protocol_version": "unified_full_graph_nc_v1", "training_mode": "full_graph",
        "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4, "epochs": context.epochs,
        "patience": 30, "early_stop_min_epoch": 30 if context.mode == "formal" else 1,
        "early_stop_min_delta": 1e-4, "grad_clip": 1.0, "eval_every": 1,
        "inference_mode": "full", "scheduler": None, "evaluate_test": False,
    }
    expected_model = {
        "name": "crsa_icsr", "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
        "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
        "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
        "num_residual_adapters": 3, "edge_chunk_size": context.edge_chunk_size,
        "use_icsr": True, "icsr_bottleneck_dim": 32,
        "icsr_node_chunk_size": context.node_chunk_size,
    }
    checks = [
        (isinstance(task, dict) and all(task.get(k) == v for k, v in expected_task.items()), "NC protocol mismatch"),
        (task.get("eval_modality_masks", {}).get("enabled") is False, "modality mask evaluation enabled"),
        (isinstance(mc, dict) and all(mc.get(k) == v for k, v in expected_model.items()), "model config mismatch"),
        (dataset_cfg.get("name") == context.dataset, "dataset mismatch"),
        (int(config.get("seed", -1)) == context.seed, "seed mismatch"),
        (int(config.get("num_runs", -1)) == 1, "num_runs mismatch"),
        (payload.get("task") == "nc" and payload.get("dataset") == context.dataset, "metrics task/dataset mismatch"),
        (payload.get("model") == "crsa_icsr" and payload.get("seed") == context.seed, "metrics model/seed mismatch"),
        (marker.get("status") == "complete", "completion marker incomplete"),
        (checkpoint.get("task") == "nc" and checkpoint.get("seed") == context.seed, "checkpoint task/seed mismatch"),
        (checkpoint.get("selection") == "best_val_accuracy", "checkpoint not selected by validation accuracy"),
        (checkpoint.get("epoch") == payload.get("best_epoch"), "checkpoint epoch mismatch"),
        (not contains_test_key(metrics), "Test key found in metrics"),
        (not contains_test_key(checkpoint.get("metrics", {})), "Test key found in checkpoint metrics"),
        (finite_state(checkpoint), "non-finite model/head checkpoint tensor"),
        (finite_losses(context.run_dir), "training log has no finite losses"),
    ]
    for ok, why in checks:
        if not ok:
            return False, why, payload
    try:
        for key in ("val_acc", "val_macro_f1"):
            value = float(metrics[key]["mean"])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                return False, f"invalid validation metric {key}", payload
        model = Model(OmegaConf.create(config), checkpoint["data_info"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        head = torch.nn.Linear(model.out_dim, int(checkpoint["data_info"]["num_classes"]))
        head.load_state_dict(checkpoint["head_state"], strict=True)
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        return False, f"strict checkpoint reload failed: {exc}", payload
    return True, "", payload


def upsert_manifest(row: dict) -> None:
    path = OUTPUT / "formal_run_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    rows = [old for old in rows if old.get("run_key") != row.get("run_key")]
    rows.append({**{key: "" for key in MANIFEST_FIELDS}, **row})
    rows.sort(key=lambda item: item["run_key"])
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: item.get(k, "") for k in MANIFEST_FIELDS} for item in rows)
    tmp.replace(path)


def run_context(context: RunContext, git: dict[str, str], dry_run: bool) -> dict:
    command = build_command(context)
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z")
    row = {
        "run_key": context.run_key, "mode": context.mode, "variant": "full", "dataset": context.dataset,
        "seed": context.seed, "epochs": context.epochs, "device": context.device,
        "git_branch": git["branch"], "git_commit": git["commit"], "git_worktree_clean": git["clean"],
        "run_dir": str(context.run_dir), "checkpoint": str(context.checkpoint),
        "resolved_config": str(context.run_dir / "resolved_config.json"), "test_evaluated": "false",
        "command": shlex.join(command), "updated_at": now(),
    }
    if dry_run:
        print(f"{context.run_key}: {shlex.join(command)}", flush=True)
        return {**row, "status": "dry_run"}
    context.run_dir.mkdir(parents=True, exist_ok=True)
    context.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    valid, _, _ = audit_context(context)
    if valid:
        row.update(status="complete", return_code=0, error_reason="skipped: complete artifacts audited")
        upsert_manifest(row)
        print(f"[skip complete] {context.run_key}", flush=True)
        return row
    row.update(status="running", updated_at=now())
    upsert_manifest(row)
    log_path = context.run_dir / "launcher_output.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - started
    valid, reason, _ = audit_context(context)
    if proc.returncode == 0 and valid:
        payload = read_json(context.run_dir / "metrics.json") or {}
        row.update(status="complete", return_code=0, runtime_seconds=payload.get("runtime_seconds", elapsed),
                   peak_gpu_memory_mib=payload.get("peak_gpu_memory_mib", ""), error_reason="", updated_at=now())
    else:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        detail = f"training exit={proc.returncode}; artifact audit: {reason}"
        if tail:
            detail += f"; last output: {tail[-1]}"
        row.update(status="failed", return_code=proc.returncode, runtime_seconds=elapsed,
                   error_reason=detail, updated_at=now())
    upsert_manifest(row)
    print(f"[{row['status']}] {context.run_key}", flush=True)
    return row


def require_formal_prerequisites(git: dict[str, str]) -> None:
    if git["branch"] != BRANCH:
        raise RuntimeError(f"formal run requires branch {BRANCH}, got {git['branch']!r}")
    if git["status"]:
        raise RuntimeError("formal run requires a clean worktree:\n" + git["status"])
    preflight = (ROOT / "results/crsa_icsr_v3/preflight_audit.json")
    report = (ROOT / "results/crsa_icsr_v3/smoke_audit.json")
    pf, sm = read_json(preflight), read_json(report)
    if not isinstance(pf, dict) or not pf.get("checkpoint_reload_equal"):
        raise RuntimeError("passing preflight audit must exist before formal runs")
    if not isinstance(sm, dict) or sm.get("complete_contexts") != 3:
        raise RuntimeError("all three fixed smoke contexts must pass before formal runs")
    if sm.get("invalid_contexts") != 0:
        raise RuntimeError("smoke audit contains an invalid context")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed Full-v3 NC smoke or formal matrix")
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    git = git_info()
    if args.mode == "formal" and not args.dry_run:
        require_formal_prerequisites(git)
    contexts = formal_contexts(args.device) if args.mode == "formal" else smoke_contexts(args.device)
    expected = 15 if args.mode == "formal" else 3
    if len(contexts) != expected or len({c.run_key for c in contexts}) != expected:
        raise RuntimeError(f"{args.mode} context matrix size is invalid")
    print(f"branch={git['branch']} source_commit={git['commit']} clean={git['clean']}")
    print(f"mode={args.mode} contexts={len(contexts)} task=nc K=3 evaluate_test=false")
    rows = [run_context(context, git, args.dry_run) for context in contexts]
    if not args.dry_run and any(row.get("status") != "complete" for row in rows):
        raise SystemExit(f"one or more {args.mode} contexts failed; inspect {OUTPUT / 'formal_run_manifest.csv'}")


if __name__ == "__main__":
    main()
