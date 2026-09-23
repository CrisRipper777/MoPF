"""Run the formal five-NC functional ablation matrix for canonical MGSC P2."""

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

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
FORMAL_VARIANTS = (
    "full",
    "uniform_relations",
    "global_context_gate",
    "terminal_state_only",
    "uniform_integration",
    "no_cross_order_interaction",
    "attribute_only",
)
OPTIONAL_VARIANTS = FORMAL_VARIANTS + ("fixed_gate_09",)


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _training_code_sha(variant: str) -> str:
    files = [
        REPO_ROOT / "src" / "main.py",
        REPO_ROOT / "src" / "tasks" / "nc.py",
        REPO_ROOT / "src" / "models" / "mgsc_mag.py",
    ]
    if variant != "full":
        files.append(REPO_ROOT / "src" / "models" / "mgsc_mag_ablation.py")
    return _sha256_files(files)


def _base_config_sha(dataset: str, model_file: str) -> str:
    return _sha256_files(
        [
            REPO_ROOT / "configs" / "config.yaml",
            REPO_ROOT / "configs" / "task" / "nc.yaml",
            REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml",
            REPO_ROOT / "configs" / "model" / f"{model_file}.yaml",
        ]
    )


def _split_sha_from_manifest(path: Path) -> str:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split_source = Path(manifest["split_source"])
    return hashlib.sha256(split_source.read_bytes()).hexdigest()


def _metric_mean(metrics: dict, key: str) -> float:
    value = metrics[key]
    return float(value["mean"] if isinstance(value, dict) else value)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _strip_config_metadata(model_cfg: dict) -> dict:
    return {
        key: value
        for key, value in model_cfg.items()
        if key not in {"name", "version", "ablation"}
    }


def _canonical_full_compatible(source_dir: Path, dataset: str, seed: int) -> bool:
    """Check prior P2 by behavior config, code hash, and split hash.

    Canonicalization changes only the config label/version.  The comparison
    therefore strips those metadata keys while requiring every effective
    model parameter to match the new P2 config.
    """
    metadata_path = source_dir / "qualification_metadata.json"
    metrics_path = source_dir / "metrics.json"
    checkpoint = source_dir / "best.pt"
    resolved_path = source_dir / "resolved_config.json"
    manifest_path = source_dir / "ablation_manifest.json"
    if not all(path.is_file() for path in (metadata_path, metrics_path, checkpoint, resolved_path, manifest_path)):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model") not in {"P1", "P2", "mgsc_mag"}:
        return False
    if int(metadata.get("seed", -1)) != int(seed):
        return False
    if metadata.get("training_code_sha256") != _training_code_sha("full"):
        return False
    if metadata.get("split_sha256") != _split_sha_from_manifest(manifest_path):
        return False
    canonical_cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / "mgsc_mag_p2.yaml")
    canonical_cfg = OmegaConf.create({"model": canonical_cfg})
    canonical = OmegaConf.to_container(canonical_cfg.model, resolve=True)
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved_model = resolved.get("model", resolved)
    return _strip_config_metadata(dict(canonical)) == _strip_config_metadata(dict(resolved_model))


def _copy_full_from_formal(dataset: str, seed: int, output_dir: Path) -> str:
    source_dir = REPO_ROOT / "outputs/final/final_candidate_nc" / f"{dataset}_seed{seed}" / "P2"
    if not _canonical_full_compatible(source_dir, dataset, seed):
        raise RuntimeError(
            f"formal P2 checkpoint cannot be reused for {dataset} seed{seed}; "
            "code/config/split fingerprint mismatch"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "best.pt",
        "metrics.json",
        "resolved_config.json",
        "resolved_config.yaml",
        "ablation_manifest.json",
    ):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    manifest_path = output_dir / "ablation_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "ablation_name": "full",
                "multihop_state_mode": "anchored",
                "multihop_response_mode": "cumulative",
                "semantic_anchor_active": True,
                "semantic_anchor_effective": True,
                "canonical_model": "mgsc_mag_p2",
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    (output_dir / "reuse_metadata.json").write_text(
        json.dumps(
            {
                "source": str(source_dir.relative_to(REPO_ROOT)),
                "reason": "canonicalization-only model config metadata change; effective P2 parameters matched",
                "dataset": dataset,
                "seed": seed,
                "model": "mgsc_mag_p2",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(source_dir.relative_to(REPO_ROOT))


def _run_variant(
    dataset: str,
    seed: int,
    variant: str,
    device: str,
    root: Path,
    skip_existing: bool,
) -> dict:
    output_dir = root / f"{dataset}_seed{seed}" / variant
    checkpoint = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    metadata_path = output_dir / "qualification_metadata.json"
    current_code_sha = _training_code_sha(variant)
    model_file = "mgsc_mag_p2" if variant == "full" else "mgsc_mag_ablation"
    current_config_sha = _base_config_sha(dataset, model_file)
    if variant == "full" and not metrics_path.is_file():
        source = _copy_full_from_formal(dataset, seed, output_dir)
        source_type = f"reused:{source}"
        can_reuse = True
    else:
        can_reuse = False
        source_type = "trained"
        existing = {}
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        can_reuse = bool(
            skip_existing
            and metrics_path.is_file()
            and checkpoint.is_file()
            and existing.get("training_code_sha256") == current_code_sha
            and existing.get("config_sha256") == current_config_sha
            and existing.get("variant") == variant
            and int(existing.get("seed", -1)) == int(seed)
        )
        if not can_reuse:
            output_dir.mkdir(parents=True, exist_ok=True)
            overrides = [
                f"dataset={dataset}",
                "task=nc",
                f"model={model_file}",
                f"model.ablation={variant}",
                f"ablation={variant}",
                f"seed={seed}",
                "num_runs=1",
                f"device={device}",
                f"task.save_ckpt_path={checkpoint}",
                f"hydra.run.dir={output_dir}",
            ]
            subprocess.run([sys.executable, "-m", "src.main", *overrides], cwd=REPO_ROOT, check=True)
        source_type = "reused_local" if can_reuse else "trained"
    if not metrics_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"missing ablation artifact: {output_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_path = output_dir / "ablation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing ablation manifest: {manifest_path}")
    split_sha = _split_sha_from_manifest(manifest_path)
    metadata = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "model_config": model_file,
        "training_code_sha256": current_code_sha,
        "config_sha256": current_config_sha,
        "split_sha256": split_sha,
        "source": source_type,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics = payload["metrics"]
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "model_config": model_file,
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
        "source": source_type,
    }


def _summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"])].append(row)
    output = []
    for (dataset, variant), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_seeds": len(values),
                "test_accuracy_mean": float(np.mean([v["test_accuracy"] for v in values])),
                "test_accuracy_population_std": float(np.std([v["test_accuracy"] for v in values], ddof=0)),
                "test_macro_f1_mean": float(np.mean([v["test_macro_f1"] for v in values])),
                "test_macro_f1_population_std": float(np.std([v["test_macro_f1"] for v in values], ddof=0)),
                "val_accuracy_mean": float(np.mean([v["val_accuracy"] for v in values])),
                "val_macro_f1_mean": float(np.mean([v["val_macro_f1"] for v in values])),
            }
        )
    return output


def _paired_deltas(rows: list[dict], datasets: list[str], seeds: list[int], variants: list[str]) -> list[dict]:
    keyed = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in rows}
    output = []
    for dataset in datasets:
        for seed in seeds:
            full = keyed[(dataset, seed, "full")]
            for variant in variants:
                if variant == "full":
                    continue
                ablation = keyed[(dataset, seed, variant)]
                output.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "ablation": variant,
                        "full_minus_ablation_accuracy": full["test_accuracy"] - ablation["test_accuracy"],
                        "full_minus_ablation_macro_f1": full["test_macro_f1"] - ablation["test_macro_f1"],
                        "full_accuracy": full["test_accuracy"],
                        "ablation_accuracy": ablation["test_accuracy"],
                        "full_macro_f1": full["test_macro_f1"],
                        "ablation_macro_f1": ablation["test_macro_f1"],
                    }
                )
    return output


def run(args: argparse.Namespace) -> None:
    datasets = list(args.datasets)
    seeds = list(args.seeds)
    variants = list(args.variants)
    if "full" not in variants:
        variants.insert(0, "full")
    root = args.output_dir.resolve()
    rows = []
    for dataset in datasets:
        for seed in seeds:
            for variant in variants:
                rows.append(_run_variant(dataset, seed, variant, args.device, root, args.skip_existing))
    per_seed_fields = [
        "dataset", "seed", "variant", "model_config", "val_accuracy", "val_macro_f1",
        "test_accuracy", "test_macro_f1", "best_epoch", "runtime_seconds", "checkpoint",
        "training_code_sha256", "config_sha256", "split_sha256", "source",
    ]
    _write_csv(root / "per_seed_results.csv", rows, per_seed_fields)
    _write_csv(
        root / "summary.csv",
        _summary(rows),
        [
            "dataset", "variant", "n_seeds", "test_accuracy_mean",
            "test_accuracy_population_std", "test_macro_f1_mean",
            "test_macro_f1_population_std", "val_accuracy_mean", "val_macro_f1_mean",
        ],
    )
    paired = _paired_deltas(rows, datasets, seeds, variants)
    _write_csv(
        root / "paired_deltas.csv",
        paired,
        [
            "dataset", "seed", "ablation", "full_minus_ablation_accuracy",
            "full_minus_ablation_macro_f1", "full_accuracy", "ablation_accuracy",
            "full_macro_f1", "ablation_macro_f1",
        ],
    )
    manifest = {
        "datasets": datasets,
        "seeds": seeds,
        "variants": variants,
        "metrics": ["accuracy", "macro_f1"],
        "std_definition": "population_std",
        "pairing": "same dataset and seed; full minus ablation",
        "protocol": "unified_full_graph_nc_v1",
        "full_model": "mgsc_mag_p2",
        "ablation_model": "mgsc_mag_ablation",
        "fixed_gate_09": "optional appendix diagnostic; not in default formal matrix",
    }
    (root / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summary_rows": len(_summary(rows)), "paired_rows": len(paired)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", choices=OPTIONAL_VARIANTS, default=list(FORMAL_VARIANTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/mgsc_functional_ablation"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
