"""Run the final four-way corrected-control NC matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
CONTROLS = ("full", "raw_terminal_only", "raw_bank_mean", "plain_multi_order_backbone")


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def control_code_sha() -> str:
    return sha256_files([
        REPO_ROOT / "src" / "main.py",
        REPO_ROOT / "src" / "tasks" / "nc.py",
        REPO_ROOT / "src" / "models" / "mgsc_mag.py",
        REPO_ROOT / "src" / "models" / "mgsc_mag_corrected_controls.py",
    ])


def canonical_p2_code_sha() -> str:
    return sha256_files([
        REPO_ROOT / "src" / "main.py",
        REPO_ROOT / "src" / "tasks" / "nc.py",
        REPO_ROOT / "src" / "models" / "mgsc_mag.py",
    ])


def config_sha(dataset: str) -> str:
    return sha256_files([
        REPO_ROOT / "configs" / "config.yaml",
        REPO_ROOT / "configs" / "task" / "nc.yaml",
        REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml",
        REPO_ROOT / "configs" / "model" / "mgsc_mag_corrected_controls.yaml",
    ])


def metric_mean(payload: dict, key: str) -> float:
    value = payload["metrics"][key]
    return float(value["mean"] if isinstance(value, dict) else value)


def split_sha(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = Path(manifest.get("split_source", ""))
    if not source.is_file():
        raise FileNotFoundError(f"split source missing from {manifest_path}: {source}")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def strip_model_metadata(model_cfg: dict) -> dict:
    return {key: value for key, value in model_cfg.items() if key not in {"name", "version", "control"}}


def old_full_compatible(source: Path, dataset: str, seed: int) -> bool:
    metadata_path = source / "qualification_metadata.json"
    resolved_path = source / "resolved_config.json"
    manifest_path = source / "ablation_manifest.json"
    if not all(path.is_file() for path in (source / "best.pt", source / "metrics.json", metadata_path, resolved_path, manifest_path)):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("seed", -1)) != seed:
        return False
    if metadata.get("training_code_sha256") != canonical_p2_code_sha():
        return False
    if metadata.get("split_sha256") != split_sha(manifest_path):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != "unified_full_graph_nc_v1":
        return False
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved_model = resolved.get("model", resolved)
    canonical = OmegaConf.to_container(
        OmegaConf.create({"model": OmegaConf.load(REPO_ROOT / "configs" / "model" / "mgsc_mag_p2.yaml")}).model,
        resolve=True,
    )
    return strip_model_metadata(dict(canonical)) == strip_model_metadata(dict(resolved_model))


def copy_full(dataset: str, seed: int, output_dir: Path) -> str:
    source = REPO_ROOT / "outputs/final/final_candidate_nc" / f"{dataset}_seed{seed}" / "P2"
    if not old_full_compatible(source, dataset, seed):
        raise RuntimeError(f"canonical P2 reuse fingerprint mismatch: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "metrics.json", "resolved_config.json", "resolved_config.yaml", "ablation_manifest.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, output_dir / name)
    manifest = json.loads((output_dir / "ablation_manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "control": "full",
        "canonical_model": "mgsc_mag_p2",
        "corrected_control_equivalence": "tests/test_mgsc_corrected_controls.py",
    })
    (output_dir / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(source.relative_to(REPO_ROOT))


def data_info_from_checkpoint(checkpoint: Path) -> dict:
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return dict(payload["data_info"])


def parameter_row(dataset: str, control: str, checkpoint: Path) -> dict:
    from src.models import build_model
    import torch.nn as nn

    def count(module) -> int:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    data_info = data_info_from_checkpoint(checkpoint)
    model_cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / "mgsc_mag_corrected_controls.yaml")
    model_cfg.control = control
    cfg = OmegaConf.create({"model": model_cfg})
    model = build_model(cfg, data_info)
    encoder = count(model)
    classifier = nn.Linear(model.out_dim, int(data_info["num_classes"]))
    head = count(classifier)
    return {"dataset": dataset, "variant": control, "encoder_params": encoder, "classifier_params": head, "total_task_params": encoder + head}


def run_one(dataset: str, seed: int, control: str, args: argparse.Namespace) -> dict:
    output_dir = args.output_dir.resolve() / f"{dataset}_seed{seed}" / control
    checkpoint = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "corrected_control_metadata.json"
    code_sha = control_code_sha()
    cfg_sha = config_sha(dataset)
    source = "trained"
    reusable = False
    if control == "full" and not metrics_path.is_file():
        try:
            source = f"reused:{copy_full(dataset, seed, output_dir)}"
            reusable = True
        except RuntimeError:
            # A stale formal checkpoint must not bypass the declared
            # fingerprint gate.  Train corrected-control Full below.
            source = "trained_fingerprint_mismatch"
    elif args.skip_existing and metrics_path.is_file() and checkpoint.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reusable = (
            metadata.get("code_sha256") == code_sha
            and metadata.get("config_sha256") == cfg_sha
            and metadata.get("control") == control
            and int(metadata.get("seed", -1)) == seed
        )
        if reusable:
            source = "reused_local"
    if not reusable:
        if control == "full":
            try:
                source = f"reused:{copy_full(dataset, seed, output_dir)}"
                reusable = True
            except RuntimeError:
                source = "trained_fingerprint_mismatch"
        if not reusable:
            output_dir.mkdir(parents=True, exist_ok=True)
            overrides = [
                f"dataset={dataset}",
                "task=nc",
                "model=mgsc_mag_corrected_controls",
                f"model.control={control}",
                "ablation=full",
                f"seed={seed}",
                "num_runs=1",
                f"device={args.device}",
                f"task.save_ckpt_path={checkpoint}",
                f"hydra.run.dir={output_dir}",
            ]
            subprocess.run([sys.executable, "-m", "src.main", *overrides], cwd=REPO_ROOT, check=True)
    if not metrics_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"incomplete corrected-control run: {output_dir}")
    manifest_path = output_dir / "ablation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    split_hash = split_sha(manifest_path)
    metadata_path.write_text(json.dumps({
        "dataset": dataset,
        "seed": seed,
        "control": control,
        "code_sha256": code_sha,
        "config_sha256": cfg_sha,
        "split_sha256": split_hash,
        "protocol": "unified_full_graph_nc_v1",
        "source": source,
    }, indent=2), encoding="utf-8")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    params = parameter_row(dataset, control, checkpoint)
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": control,
        "accuracy": metric_mean(payload, "test_acc"),
        "macro_f1": metric_mean(payload, "test_macro_f1"),
        "val_accuracy": metric_mean(payload, "val_acc"),
        "val_macro_f1": metric_mean(payload, "val_macro_f1"),
        "best_epoch": payload.get("checkpoint_metadata", {}).get("best_epoch"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "source": source,
        **params,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict],
    datasets: tuple[str, ...],
    seeds: tuple[int, ...],
    controls: tuple[str, ...],
) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"])].append(row)
    summary = []
    for (dataset, variant), values in sorted(grouped.items()):
        acc = [float(row["accuracy"]) for row in values]
        f1 = [float(row["macro_f1"]) for row in values]
        summary.append({
            "dataset": dataset,
            "variant": variant,
            "n_seeds": len(values),
            "accuracy_mean": sum(acc) / len(acc),
            "accuracy_population_std": (sum((value - sum(acc) / len(acc)) ** 2 for value in acc) / len(acc)) ** 0.5,
            "macro_f1_mean": sum(f1) / len(f1),
            "macro_f1_population_std": (sum((value - sum(f1) / len(f1)) ** 2 for value in f1) / len(f1)) ** 0.5,
            "encoder_params": values[0]["encoder_params"],
            "classifier_params": values[0]["classifier_params"],
            "total_task_params": values[0]["total_task_params"],
        })
    by_key = {(row["dataset"], row["seed"], row["variant"]): row for row in rows}
    paired = []
    for dataset in datasets:
        for seed in seeds:
            full = by_key[(dataset, seed, "full")]
            for control in controls:
                if control == "full":
                    continue
                other = by_key[(dataset, seed, control)]
                paired.append({
                    "dataset": dataset,
                    "seed": seed,
                    "control": control,
                    "full_minus_control_accuracy": full["accuracy"] - other["accuracy"],
                    "full_minus_control_macro_f1": full["macro_f1"] - other["macro_f1"],
                    "full_accuracy": full["accuracy"],
                    "control_accuracy": other["accuracy"],
                    "full_macro_f1": full["macro_f1"],
                    "control_macro_f1": other["macro_f1"],
                })
    extra = []
    for control in controls:
        if control == "full":
            continue
        values = [row for row in paired if row["control"] == control]
        acc = [float(row["full_minus_control_accuracy"]) for row in values]
        f1 = [float(row["full_minus_control_macro_f1"]) for row in values]
        sign_rows = []
        for dataset in datasets:
            ds = [row for row in values if row["dataset"] == dataset]
            ds_acc = [float(row["full_minus_control_accuracy"]) for row in ds]
            ds_f1 = [float(row["full_minus_control_macro_f1"]) for row in ds]
            sign_rows.append({
                "dataset": dataset,
                "accuracy_nonnegative_seed_count": sum(value >= 0 for value in ds_acc),
                "macro_f1_nonnegative_seed_count": sum(value >= 0 for value in ds_f1),
            })
        extra.append({
            "control": control,
            "five_dataset_mean_paired_accuracy": sum(acc) / len(acc),
            "five_dataset_mean_paired_macro_f1": sum(f1) / len(f1),
            "median_paired_accuracy": median(acc),
            "median_paired_macro_f1": median(f1),
            "full_better_accuracy_seed_count_of_15": sum(value > 0 for value in acc),
            "full_better_macro_f1_seed_count_of_15": sum(value > 0 for value in f1),
            "datasetwise_accuracy_sign_consistency": json.dumps(sign_rows),
        })
    return summary, paired, extra


def run(args: argparse.Namespace) -> None:
    rows = [run_one(dataset, seed, control, args) for dataset in args.datasets for seed in args.seeds for control in args.controls]
    datasets = tuple(args.datasets)
    seeds = tuple(args.seeds)
    controls = tuple(args.controls)
    summary, paired, extra = aggregate(rows, datasets, seeds, controls)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "per_seed_results.csv", rows, list(rows[0]))
    write_csv(root / "summary.csv", summary, list(summary[0]))
    write_csv(root / "paired_deltas.csv", paired, list(paired[0]))
    write_csv(root / "paired_delta_summary.csv", extra, list(extra[0]))
    parameter_rows = []
    for row in summary:
        full = next(item for item in summary if item["dataset"] == row["dataset"] and item["variant"] == "full")
        parameter_rows.append({
            "dataset": row["dataset"],
            "variant": row["variant"],
            "full_encoder_params": full["encoder_params"],
            "encoder_params": row["encoder_params"],
            "classifier_params": row["classifier_params"],
            "total_task_params": row["total_task_params"],
            "encoder_difference_vs_full_pct": 100.0 * (row["encoder_params"] - full["encoder_params"]) / full["encoder_params"],
            "total_difference_vs_full_pct": 100.0 * (row["total_task_params"] - full["total_task_params"]) / full["total_task_params"],
        })
    write_csv(root / "parameter_counts.csv", parameter_rows, list(parameter_rows[0]))
    (root / "manifest.json").write_text(json.dumps({
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "controls": list(args.controls),
        "protocol": "unified_full_graph_nc_v1",
        "pairing": "same dataset and seed; full minus control",
        "std_definition": "population_std",
        "full_reuse": "only after canonical P2 config, split, protocol, and code fingerprint checks plus C0 equivalence test",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired), "parameter_rows": len(parameter_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=SEEDS)
    parser.add_argument("--controls", nargs="+", default=list(CONTROLS), choices=CONTROLS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/mgsc_corrected_controls"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
