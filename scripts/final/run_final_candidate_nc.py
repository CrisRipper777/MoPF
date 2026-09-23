"""Run the formal five-NC, three-seed P0/P1/P2 qualification matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
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
    files = [REPO_ROOT / "src" / "main.py", REPO_ROOT / "src" / "tasks" / "nc.py"]
    if model == "P0":
        files.append(REPO_ROOT / "src" / "models" / "cosi_mag_final.py")
    else:
        files.append(REPO_ROOT / "src" / "models" / "mgsc_mag.py")
    return _sha256_files(files)


def _config_sha(dataset: str, model: str) -> str:
    files = [
        REPO_ROOT / "configs" / "config.yaml",
        REPO_ROOT / "configs" / "task" / "nc.yaml",
        REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml",
        REPO_ROOT / "configs" / "model" / f"{MODEL_CONFIG[model]}.yaml",
    ]
    return _sha256_files(files)


def _metric_mean(metrics: dict, key: str) -> float:
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


def _run_one(
    dataset: str,
    seed: int,
    model: str,
    device: str,
    root: Path,
    skip_existing: bool,
) -> dict:
    output_dir = root / f"{dataset}_seed{seed}" / model
    checkpoint = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "qualification_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    current_code_sha = _code_sha(model)
    current_config_sha = _config_sha(dataset, model)
    can_reuse = False
    metadata = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        can_reuse = bool(
            skip_existing
            and metrics_path.is_file()
            and checkpoint.is_file()
            and metadata.get("training_code_sha256") == current_code_sha
            and metadata.get("config_sha256") == current_config_sha
            and metadata.get("model") == model
            and int(metadata.get("seed", -1)) == int(seed)
        )
    if not can_reuse:
        overrides = [
            f"dataset={dataset}",
            "task=nc",
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
        raise FileNotFoundError(f"missing formal NC artifact: {output_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_path = output_dir / "ablation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_path = Path(manifest["split_source"])
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    metadata = {
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "training_code_sha256": current_code_sha,
        "config_sha256": current_config_sha,
        "split_source": str(split_path),
        "split_sha256": split_sha,
        "reused_existing": can_reuse,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics = payload["metrics"]
    return {
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "model_config": MODEL_CONFIG[model],
        "val_accuracy": _metric_mean(metrics, "val_acc"),
        "val_macro_f1": _metric_mean(metrics, "val_macro_f1"),
        "test_accuracy": _metric_mean(metrics, "test_acc"),
        "test_macro_f1": _metric_mean(metrics, "test_macro_f1"),
        "best_epoch": payload.get("checkpoint_metadata", {}).get("best_epoch"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "training_code_sha256": current_code_sha,
        "config_sha256": current_config_sha,
        "split_sha256": split_sha,
        "reused_existing": can_reuse,
    }


def _summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"])].append(row)
    output = []
    for (dataset, model), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "n_seeds": len(values),
                "val_accuracy_mean": float(np.mean([row["val_accuracy"] for row in values])),
                "val_accuracy_population_std": float(np.std([row["val_accuracy"] for row in values], ddof=0)),
                "val_macro_f1_mean": float(np.mean([row["val_macro_f1"] for row in values])),
                "val_macro_f1_population_std": float(np.std([row["val_macro_f1"] for row in values], ddof=0)),
                "test_accuracy_mean": float(np.mean([row["test_accuracy"] for row in values])),
                "test_accuracy_population_std": float(np.std([row["test_accuracy"] for row in values], ddof=0)),
                "test_macro_f1_mean": float(np.mean([row["test_macro_f1"] for row in values])),
                "test_macro_f1_population_std": float(np.std([row["test_macro_f1"] for row in values], ddof=0)),
            }
        )
    return output


def _paired_deltas(rows: list[dict], datasets: list[str], seeds: list[int]) -> list[dict]:
    keyed = {(row["dataset"], row["seed"], row["model"]): row for row in rows}
    output = []
    for dataset in datasets:
        for seed in seeds:
            p0 = keyed[(dataset, seed, "P0")]
            for left, right in (("P1", "P0"), ("P2", "P0"), ("P2", "P1")):
                a = keyed[(dataset, seed, left)]
                b = keyed[(dataset, seed, right)]
                output.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "comparison": f"{left}-{right}",
                        "accuracy_delta": a["test_accuracy"] - b["test_accuracy"],
                        "macro_f1_delta": a["test_macro_f1"] - b["test_macro_f1"],
                        "left_accuracy": a["test_accuracy"],
                        "right_accuracy": b["test_accuracy"],
                        "left_macro_f1": a["test_macro_f1"],
                        "right_macro_f1": b["test_macro_f1"],
                    }
                )
    return output


def run(args: argparse.Namespace) -> None:
    root = args.output_dir.resolve()
    datasets = list(args.datasets)
    seeds = list(args.seeds)
    rows = []
    for dataset in datasets:
        for seed in seeds:
            for model in MODELS:
                rows.append(_run_one(dataset, seed, model, args.device, root, args.skip_existing))
    per_seed_fields = [
        "dataset", "seed", "model", "model_config", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1",
        "best_epoch", "runtime_seconds", "checkpoint", "training_code_sha256", "config_sha256", "split_sha256", "reused_existing",
    ]
    _write_csv(root / "per_seed_results.csv", rows, per_seed_fields)
    summary_rows = _summary(rows)
    _write_csv(
        root / "summary.csv",
        summary_rows,
        [
            "dataset", "model", "n_seeds", "val_accuracy_mean", "val_accuracy_population_std", "val_macro_f1_mean", "val_macro_f1_population_std",
            "test_accuracy_mean", "test_accuracy_population_std", "test_macro_f1_mean", "test_macro_f1_population_std",
        ],
    )
    paired = _paired_deltas(rows, datasets, seeds)
    _write_csv(
        root / "paired_deltas.csv",
        paired,
        ["dataset", "seed", "comparison", "accuracy_delta", "macro_f1_delta", "left_accuracy", "right_accuracy", "left_macro_f1", "right_macro_f1"],
    )
    metadata = {
        "datasets": datasets,
        "seeds": seeds,
        "models": list(MODELS),
        "protocol": "unified_full_graph_nc_v1",
        "old_seed42_reuse_policy": "reused only with matching saved code/config fingerprint; otherwise rerun",
    }
    (root / "qualification_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary_rows), "paired_rows": len(paired)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/final_candidate_nc"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
