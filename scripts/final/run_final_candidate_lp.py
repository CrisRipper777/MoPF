"""Run the formal Sports-Copurchase LP qualification matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET = "sports-copurchase"
MODELS = ("P0", "P1", "P2")
MODEL_CONFIG = {"P0": "cosi_mag_final", "P1": "mgsc_mag", "P2": "mgsc_mag"}


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _code_sha(model: str) -> str:
    files = [REPO_ROOT / "src" / "main.py", REPO_ROOT / "src" / "tasks" / "lp.py"]
    files.append(REPO_ROOT / "src" / "models" / ("cosi_mag_final.py" if model == "P0" else "mgsc_mag.py"))
    return _sha256_files(files)


def _config_sha(model: str) -> str:
    return _sha256_files(
        [
            REPO_ROOT / "configs" / "config.yaml",
            REPO_ROOT / "configs" / "task" / "lp.yaml",
            REPO_ROOT / "configs" / "dataset" / "sports-copurchase.yaml",
            REPO_ROOT / "configs" / "model" / f"{MODEL_CONFIG[model]}.yaml",
        ]
    )


def _mean(metrics: dict, key: str) -> float:
    value = metrics[key]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_one(seed: int, model: str, device: str, root: Path, skip_existing: bool) -> dict:
    output_dir = root / f"{DATASET}_seed{seed}" / model
    checkpoint = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "qualification_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    code_sha = _code_sha(model)
    config_sha = _config_sha(model)
    reused = False
    if metadata_path.is_file() and metrics_path.is_file() and checkpoint.is_file():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        reused = bool(
            skip_existing
            and old.get("training_code_sha256") == code_sha
            and old.get("config_sha256") == config_sha
            and old.get("model") == model
            and int(old.get("seed", -1)) == seed
        )
    if not reused:
        overrides = [
            f"dataset={DATASET}",
            "task=lp",
            f"model={MODEL_CONFIG[model]}",
            f"seed={seed}",
            "num_runs=1",
            f"device={device}",
            f"task.save_ckpt_path={checkpoint}",
            f"hydra.run.dir={output_dir}",
        ]
        if model == "P1":
            overrides.extend([
                "model.adaptive_context_gate=true",
                "model.direct_interacted_integration=false",
                "model.use_legacy_relation_order_bias=true",
            ])
        elif model == "P2":
            overrides.extend([
                "model.adaptive_context_gate=true",
                "model.direct_interacted_integration=true",
                "model.use_legacy_relation_order_bias=true",
            ])
        subprocess.run([sys.executable, "-m", "src.main", *overrides], cwd=REPO_ROOT, check=True)
    if not metrics_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"missing formal LP artifact: {output_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    values = {
        "val_mrr": _mean(metrics, "val_mrr"),
        "test_mrr": _mean(metrics, "test_mrr"),
        "test_hits@1": _mean(metrics, "test_hits@1"),
        "test_hits@3": _mean(metrics, "test_hits@3"),
        "test_hits@10": _mean(metrics, "test_hits@10"),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError(f"NaN/Inf in LP metrics for {seed}/{model}: {values}")
    manifest = json.loads((output_dir / "ablation_manifest.json").read_text(encoding="utf-8"))
    split_path = Path(manifest["split_source"])
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    max_order = int(resolved["model"]["max_order"])
    sampler_depth = len(resolved["task"]["num_neighbors"])
    if max_order != sampler_depth:
        raise ValueError(f"P2 sampler depth mismatch: max_order={max_order}, sampler_depth={sampler_depth}")
    metadata = {
        "dataset": DATASET,
        "seed": seed,
        "model": model,
        "training_code_sha256": code_sha,
        "config_sha256": config_sha,
        "split_source": str(split_path),
        "split_sha256": split_sha,
        "max_order": max_order,
        "sampler_depth": sampler_depth,
        "reused_existing": reused,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "dataset": DATASET,
        "seed": seed,
        "model": model,
        "model_config": MODEL_CONFIG[model],
        **values,
        "best_epoch": payload.get("checkpoint_metadata", {}).get("best_epoch"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "training_code_sha256": code_sha,
        "config_sha256": config_sha,
        "split_sha256": split_sha,
        "max_order": max_order,
        "sampler_depth": sampler_depth,
        "reused_existing": reused,
    }


def run(args: argparse.Namespace) -> None:
    root = args.output_dir.resolve()
    rows = []
    for seed in args.seeds:
        for model in MODELS:
            rows.append(_run_one(seed, model, args.device, root, args.skip_existing))
    _write_csv(
        root / "per_seed_results.csv",
        rows,
        ["dataset", "seed", "model", "model_config", "val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10", "best_epoch", "runtime_seconds", "checkpoint", "training_code_sha256", "config_sha256", "split_sha256", "max_order", "sampler_depth", "reused_existing"],
    )
    _write_csv(root / "summary.csv", rows, ["dataset", "seed", "model", "model_config", "val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10", "best_epoch", "runtime_seconds", "checkpoint", "training_code_sha256", "config_sha256", "split_sha256", "max_order", "sampler_depth", "reused_existing"])
    (root / "qualification_metadata.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "seeds": list(args.seeds),
                "models": list(MODELS),
                "protocol": "unified_sampled_lp_v1",
                "required_sampler_depth": 3,
                "note": "Metrics are exported only after finite-value and max_order/sampler_depth checks.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/final_candidate_lp"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
