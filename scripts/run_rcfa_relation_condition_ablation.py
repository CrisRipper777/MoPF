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
VARIANT = "wo_relation_condition"
OUTPUT = ROOT / "outputs/rcfa_relation_condition_ablation"
BRANCH = "exp/rcfa_relation_condition_ablation"
MANIFEST_FIELDS = [
    "run_key", "mode", "variant", "dataset", "seed", "device", "status",
    "return_code", "command", "git_branch", "git_commit", "git_worktree_clean",
    "run_dir", "checkpoint", "resolved_config", "runtime_seconds",
    "peak_gpu_memory_mib", "test_evaluated", "error_reason", "updated_at",
]


@dataclass(frozen=True)
class RunContext:
    dataset: str
    seed: int
    device: str = "cuda:1"
    edge_chunk_size: int = 65536
    rcfa_node_chunk_size: int = 4096

    @property
    def run_key(self) -> str:
        return f"formal/{VARIANT}/{self.dataset}/seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        return OUTPUT / "runs" / VARIANT / self.dataset / f"seed{self.seed}"

    @property
    def checkpoint(self) -> Path:
        return OUTPUT / "checkpoints" / VARIANT / self.dataset / f"seed{self.seed}.pt"


def formal_contexts(device: str = "cuda:1") -> list[RunContext]:
    return [RunContext(dataset, seed, device) for dataset in DATASETS for seed in SEEDS]


def git_info() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    status = git("status", "--porcelain")
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "status": status,
        "clean": "true" if not status else "false",
    }


def require_formal_git_state(info: dict[str, str]) -> None:
    if info["branch"] != BRANCH:
        raise RuntimeError(f"formal mode requires branch {BRANCH}, got {info['branch']!r}")
    if info["status"]:
        raise RuntimeError("formal mode requires a clean Git worktree:\n" + info["status"])


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
        "model.rcfa_hidden_dim=256", "model.use_rcfa=true",
        "model.use_relation_condition=false",
        "task.protocol_version=unified_full_graph_nc_v1", "task.training_mode=full_graph",
        "task.optimizer=adamw", "task.lr=1e-3", "task.weight_decay=1e-4",
        "task.epochs=300", "task.patience=30", "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4", "task.grad_clip=1.0", "task.eval_every=1",
        "task.inference_mode=full", "task.scheduler=null", "task.evaluate_test=false",
        "task.eval_modality_masks.enabled=false", "task.loss.aux_weight=0.0",
        f"task.save_ckpt_path={context.checkpoint}", f"hydra.run.dir={context.run_dir}",
    ]


def _read_json(path: Path) -> dict | None:
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


def _finite_checkpoint(checkpoint: dict) -> bool:
    import torch
    for key in ("model_state", "head_state"):
        state = checkpoint.get(key)
        if not isinstance(state, dict) or not state:
            return False
        for tensor in state.values():
            if isinstance(tensor, torch.Tensor) and (tensor.is_floating_point() or tensor.is_complex()):
                if not bool(torch.isfinite(tensor).all()):
                    return False
    return True


def _finite_losses(run_dir: Path) -> bool:
    import re
    log = next((p for p in (run_dir / "train.log", run_dir / "main.log") if p.is_file()), None)
    if log is None:
        return False
    values = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Train Loss" not in line:
            continue
        match = re.search(r"Train Loss\s+([^\s]+)", line)
        if match is None:
            return False
        try:
            values.append(float(match.group(1)))
        except ValueError:
            return False
    return bool(values) and all(math.isfinite(value) for value in values)


def audit_context(context: RunContext) -> tuple[bool, str, dict | None]:
    import torch
    from omegaconf import OmegaConf
    from src.models.crsa_rcfa import Model

    config_path = context.run_dir / "resolved_config.json"
    metrics_path = context.run_dir / "metrics.json"
    marker_path = context.run_dir / "complete.marker"
    required = (config_path, metrics_path, marker_path, context.checkpoint)
    if not all(path.is_file() for path in required):
        return False, "missing resolved config, metrics, completion marker, or checkpoint", None
    config, payload, marker = map(_read_json, (config_path, metrics_path, marker_path))
    if not all(isinstance(value, dict) for value in (config, payload, marker)):
        return False, "malformed run metadata", None
    try:
        checkpoint = torch.load(context.checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        return False, f"checkpoint load failed: {exc}", None
    if not isinstance(checkpoint, dict):
        return False, "checkpoint is not a mapping", None

    task, model_cfg = config.get("task", {}), config.get("model", {})
    metrics = payload.get("metrics", {})
    checks = [
        (task.get("evaluate_test") is False, "Test evaluation is enabled"),
        (task.get("eval_modality_masks", {}).get("enabled") is False, "modality masks are enabled"),
        (not contains_test_key(metrics), "Test metric key found in metrics"),
        (not contains_test_key(checkpoint.get("metrics", {})), "Test metric key found in checkpoint"),
        (model_cfg.get("name") == "crsa_rcfa", "wrong model"),
        (model_cfg.get("use_rcfa") is True, "RCFA is disabled"),
        (model_cfg.get("use_relation_condition") is False, "relation conditioning is enabled"),
        (model_cfg.get("max_order") == 3, "K is not 3"),
        (model_cfg.get("edge_chunk_size") == context.edge_chunk_size, "edge chunk size mismatch"),
        (model_cfg.get("rcfa_node_chunk_size") == context.rcfa_node_chunk_size, "RCFA chunk size mismatch"),
        (config.get("dataset", {}).get("name") == context.dataset, "dataset mismatch"),
        (int(config.get("seed", -1)) == context.seed, "seed mismatch"),
        (int(config.get("num_runs", -1)) == 1, "num_runs is not 1"),
        (payload.get("task") == "nc", "metrics task is not NC"),
        (payload.get("dataset") == context.dataset, "metrics dataset mismatch"),
        (payload.get("seed") == context.seed, "metrics seed mismatch"),
        (payload.get("model") == "crsa_rcfa", "metrics model mismatch"),
        (marker.get("status") == "complete", "completion marker is not complete"),
        (checkpoint.get("task") == "nc", "checkpoint task mismatch"),
        (checkpoint.get("seed") == context.seed, "checkpoint seed mismatch"),
        (checkpoint.get("selection") == "best_val_accuracy", "checkpoint was not selected by validation accuracy"),
        (checkpoint.get("epoch") == payload.get("best_epoch"), "checkpoint epoch differs from metrics"),
        (_finite_checkpoint(checkpoint), "checkpoint has non-finite tensors"),
        (_finite_losses(context.run_dir), "training log has no finite loss records"),
    ]
    for ok, reason in checks:
        if not ok:
            return False, reason, payload

    expected_task = {
        "name": "nc", "protocol_version": "unified_full_graph_nc_v1",
        "training_mode": "full_graph", "optimizer": "adamw", "lr": 1e-3,
        "weight_decay": 1e-4, "epochs": 300, "patience": 30,
        "early_stop_min_epoch": 30, "early_stop_min_delta": 1e-4,
        "grad_clip": 1.0, "eval_every": 1, "inference_mode": "full", "scheduler": None,
    }
    if any(task.get(key) != value for key, value in expected_task.items()):
        return False, "resolved task protocol mismatch", payload
    dims = {
        "hidden_dim": 256, "max_order": 3, "dropout": 0.2,
        "relation_dim": 64, "relation_heads": 4, "relation_ffn_dim": 128,
        "shared_bottleneck_dim": 256, "adapter_bottleneck_dim": 32,
        "num_residual_adapters": 3, "rcfa_hidden_dim": 256,
    }
    if any(model_cfg.get(key) != value for key, value in dims.items()):
        return False, "resolved model dimensions mismatch", payload
    for key in ("val_acc", "val_macro_f1"):
        try:
            value = float(metrics[key]["mean"])
        except (KeyError, TypeError, ValueError):
            return False, f"missing validation metric {key}", payload
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return False, f"invalid validation metric {key}", payload

    data_info = checkpoint.get("data_info")
    if not isinstance(data_info, dict):
        return False, "checkpoint has no data_info", payload
    try:
        model = Model(OmegaConf.create(config), data_info)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        head = torch.nn.Linear(model.out_dim, int(data_info["num_classes"]))
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
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: item.get(key, "") for key in MANIFEST_FIELDS} for item in rows)
    temp.replace(path)


def run_context(context: RunContext, git: dict[str, str], dry_run: bool) -> dict:
    command = build_command(context)
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z")
    row = {
        "run_key": context.run_key, "mode": "formal", "variant": VARIANT,
        "dataset": context.dataset, "seed": context.seed, "device": context.device,
        "git_branch": git["branch"], "git_commit": git["commit"],
        "git_worktree_clean": git["clean"], "run_dir": str(context.run_dir),
        "checkpoint": str(context.checkpoint),
        "resolved_config": str(context.run_dir / "resolved_config.json"),
        "test_evaluated": "false", "command": shlex.join(command), "updated_at": now(),
    }
    if dry_run:
        print(f"{context.run_key}: {shlex.join(command)}", flush=True)
        return {**row, "status": "dry_run"}
    context.run_dir.mkdir(parents=True, exist_ok=True)
    context.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if audit_context(context)[0]:
        row.update(status="complete", return_code=0, error_reason="skipped: complete artifacts audited")
        upsert_manifest(row)
        print(f"[skip complete] {context.run_key}", flush=True)
        return row
    row.update(status="running", updated_at=now())
    upsert_manifest(row)
    output_log = context.run_dir / "launcher_output.log"
    started = time.perf_counter()
    with output_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - started
    valid, reason, payload = audit_context(context)
    if proc.returncode == 0 and valid:
        metrics = (payload or {}).get("metrics", {})
        # Training writes runtime and peak allocator statistics to metrics.json.
        metadata = _read_json(context.run_dir / "metrics.json") or {}
        row.update(status="complete", return_code=0,
                   runtime_seconds=metadata.get("runtime_seconds", elapsed),
                   peak_gpu_memory_mib=metadata.get("peak_gpu_memory_mib", ""),
                   error_reason="", updated_at=now())
    else:
        tail = output_log.read_text(encoding="utf-8", errors="replace").splitlines()
        detail = f"training exit={proc.returncode}; artifact audit: {reason}"
        if tail:
            detail += f"; last output: {tail[-1]}"
        row.update(status="failed", return_code=proc.returncode, runtime_seconds=elapsed,
                   error_reason=detail, updated_at=now())
    upsert_manifest(row)
    print(f"[{row['status']}] {context.run_key}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal NC ablation: RCFA without explicit RSE conditioning")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    git = git_info()
    if not args.dry_run:
        require_formal_git_state(git)
    contexts = formal_contexts(args.device)
    if len(contexts) != 15 or len({context.run_key for context in contexts}) != 15:
        raise RuntimeError("formal matrix must have exactly 15 unique contexts")
    print(f"branch={git['branch']} source_commit={git['commit']} worktree_clean={git['clean']}")
    print(f"variant={VARIANT} contexts={len(contexts)} task=nc K=3 test=false")
    rows = [run_context(context, git, args.dry_run) for context in contexts]
    if not args.dry_run and any(row.get("status") != "complete" for row in rows):
        raise SystemExit("one or more formal contexts failed; inspect formal_run_manifest.csv")


if __name__ == "__main__":
    main()
