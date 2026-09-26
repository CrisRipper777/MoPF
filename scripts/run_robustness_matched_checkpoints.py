#!/usr/bin/env python3
"""Train only the eight Figure-5 matched-split robustness ablations.

The script first audits the already-frozen seed-42 split against the existing
Full, DiP, and seed-42 Core Story checkpoints.  It stops before training if
any source differs.  The only trained runs are the two requested ablations
for Movies/Grocery and RNG seeds 43/44.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/robustness_matched_checkpoints"
AUDIT = ROOT / "outputs/robustness_analysis/checkpoint_audit.csv"
DATASETS = ("Movies", "Grocery")
VARIANTS = ("wo_relation_calibration", "wo_semantic_anchor")
SEEDS = (43, 44)
K_BY_DATASET = {"Movies": 3, "Grocery": 2}
SPLIT_DIR = Path("/hdd1/DataInHere/YHF/data/MAGB_split")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return True


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_split_path(dataset: str) -> Path:
    return SPLIT_DIR / f"{dataset}_nc_seed42_train0.6_val0.2.pt"


def audit_before_training() -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not AUDIT.is_file():
        raise FileNotFoundError(f"Existing robustness checkpoint audit missing: {AUDIT}")
    with AUDIT.open("r", newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))

    evidence: list[dict[str, Any]] = []
    dataset_sha: dict[str, str] = {}
    for dataset in DATASETS:
        split_path = fixed_split_path(dataset)
        if not split_path.is_file():
            raise FileNotFoundError(f"Required frozen seed-42 split missing: {split_path}")
        expected_sha = sha256_file(split_path)
        relevant = [row for row in all_rows if row.get("dataset") == dataset]
        source_specs = [
            ("Full CoSI-MAG", "mopf", seed)
            for seed in (42, 43, 44)
        ] + [
            ("DiP", "dip", seed)
            for seed in (42, 43, 44)
        ] + [
            (variant, variant, 42)
            for variant in VARIANTS
        ]
        observed_shas: set[str] = {expected_sha}
        for label, model, seed in source_specs:
            matches = [
                row for row in relevant
                if row.get("model") == model and int(row.get("model_seed", -1)) == seed
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Pre-run audit expected one {dataset}/{label}/seed{seed} row; got {len(matches)}"
                )
            row = matches[0]
            if row.get("audit_status") != "PASS":
                raise RuntimeError(f"Pre-run source audit is not PASS: {dataset}/{label}/seed{seed}")
            if Path(row["split_path"]).resolve() != split_path.resolve():
                raise RuntimeError(
                    f"Split path differs for {dataset}/{label}/seed{seed}: {row['split_path']}"
                )
            actual_sha = sha256_file(Path(row["split_path"]))
            if actual_sha != row.get("split_sha256"):
                raise RuntimeError(f"Recorded split SHA mismatch for {dataset}/{label}/seed{seed}")
            observed_shas.add(actual_sha)
            evidence.append({
                "dataset": dataset,
                "source": label,
                "model": model,
                "training_seed": seed,
                "split_seed": 42,
                "split_path": str(split_path),
                "split_sha256": actual_sha,
                "checkpoint_path": row["checkpoint_path"],
                "audit_status": row["audit_status"],
            })
        for variant in VARIANTS:
            for seed in SEEDS:
                evidence.append({
                    "dataset": dataset,
                    "source": "planned_robustness_matched_run",
                    "model": variant,
                    "training_seed": seed,
                    "split_seed": 42,
                    "split_path": str(split_path),
                    "split_sha256": expected_sha,
                    "checkpoint_path": str(
                        OUT / dataset / variant / f"seed{seed}" / "best.pt"
                    ),
                    "audit_status": "PLANNED",
                })
        if len(observed_shas) != 1:
            raise RuntimeError(
                f"STOP: {dataset} source checkpoints do not share exactly one split SHA256: "
                f"{sorted(observed_shas)}"
            )
        dataset_sha[dataset] = expected_sha

    write_csv(OUT / "pre_run_split_audit.csv", evidence)
    payload = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_audit_path": str(AUDIT),
        "dataset_split_sha256": dataset_sha,
        "evidence_rows": len(evidence),
        "source_rows_per_dataset": 8,
        "planned_training_runs": 8,
    }
    (OUT / "pre_run_split_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Pre-run split audit PASS", flush=True)
    for dataset, split_sha in dataset_sha.items():
        print(f"  {dataset}: {split_sha}", flush=True)
    return evidence, dataset_sha


def run_one(dataset: str, variant: str, seed: int, device: str, expected_sha: str) -> dict[str, Any]:
    output_dir = OUT / dataset / variant / f"seed{seed}"
    required = (
        "best.pt", "metrics.json", "ablation_manifest.json", "resolved_config.yaml",
        "resolved_config.json", "complete.marker", "train.log",
    )
    is_complete = output_dir.is_dir() and all((output_dir / name).is_file() for name in required)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not is_complete:
            prior_log = output_dir / "launcher.log"
            has_training_artifact = any(
                (output_dir / name).exists()
                for name in ("best.pt", "metrics.json", "ablation_manifest.json", "complete.marker")
            )
            prior_text = prior_log.read_text(encoding="utf-8", errors="replace").lower() if prior_log.is_file() else ""
            if has_training_artifact or "cuda is not available in this environment" not in prior_text:
                raise RuntimeError(f"Refusing to overwrite nonempty output directory: {output_dir}")
            attempts_dir = output_dir / "attempt_logs"
            attempts_dir.mkdir(exist_ok=True)
            archived = attempts_dir / "pre_training_cuda_unavailable.log"
            if not archived.exists():
                shutil.copy2(prior_log, archived)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = fixed_split_path(dataset)
    k = K_BY_DATASET[dataset]
    command = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=mopf",
        f"seed={seed}", "num_runs=1", f"device={device}",
        f"ablation={variant}", f"model.max_order={k}", f"model.num_layers={k}",
        f"dataset.nc_split_path={split_path}",
        f"hydra.run.dir={output_dir}", f"task.save_ckpt_path={output_dir / 'best.pt'}",
    ]
    print(
        f"[START] {dataset}/{variant}/seed{seed} (training RNG seed={seed}, split seed=42)",
        flush=True,
    )
    if is_complete:
        print(f"[REUSE_COMPLETE] {dataset}/{variant}/seed{seed}; validating existing successful run", flush=True)
    else:
        log_path = output_dir / "launcher.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Training failed with exit code {return_code}: {dataset}/{variant}/seed{seed}")

    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete run {dataset}/{variant}/seed{seed}; missing {missing}")
    full_log = "\n".join(
        (output_dir / name).read_text(encoding="utf-8", errors="replace")
        for name in ("launcher.log", "train.log")
    ).lower()
    if any(token in full_log for token in ("cuda out of memory", "out of memory", "traceback (most recent call last)")):
        raise RuntimeError(f"OOM/traceback marker found in {dataset}/{variant}/seed{seed} logs")

    config = OmegaConf.to_container(OmegaConf.load(output_dir / "resolved_config.yaml"), resolve=True)
    metrics = read_json(output_dir / "metrics.json")
    manifest = read_json(output_dir / "ablation_manifest.json")
    if int(config["seed"]) != seed or config["dataset"]["name"] != dataset:
        raise RuntimeError(f"Resolved dataset/seed mismatch in {output_dir}")
    if config["ablation"] != variant:
        raise RuntimeError(f"Resolved ablation mismatch in {output_dir}: {config['ablation']}")
    if Path(config["dataset"]["nc_split_path"]).resolve() != split_path.resolve():
        raise RuntimeError(f"Resolved split path mismatch in {output_dir}")
    if sha256_file(split_path) != expected_sha:
        raise RuntimeError(f"Frozen seed-42 split changed during training: {split_path}")
    if manifest.get("protocol_version") != "unified_full_graph_nc_v1":
        raise RuntimeError(f"Protocol mismatch in {output_dir}")
    if manifest.get("K") != k or manifest.get("seed") != seed or manifest.get("dataset") != dataset:
        raise RuntimeError(f"Ablation manifest identity/K mismatch in {output_dir}")
    if manifest.get("checkpoint_selection") != "best_val_accuracy":
        raise RuntimeError(f"Checkpoint selection mismatch in {output_dir}")
    expected_modes = {
        "wo_relation_calibration": {"edge_weight_mode": "raw_uniform"},
        "wo_semantic_anchor": {
            "multihop_state_mode": "ordinary",
            "multihop_response_mode": "cumulative",
        },
    }[variant]
    for key, value in expected_modes.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Ablation semantic mismatch {key}={manifest.get(key)!r} in {output_dir}")
    for key, expected in (("hidden_dim", 256), ("dropout", 0.2), ("optimizer", "adamw"),
                          ("learning_rate", 0.001), ("weight_decay", 0.0001),
                          ("max_epochs", 300), ("patience", 30),
                          ("alpha", 0.1), ("relation_temperature", 0.35)):
        got = manifest.get(key)
        if isinstance(expected, float):
            if got is None or not math.isclose(float(got), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"Formal protocol mismatch {key}={got!r} in {output_dir}")
        elif got != expected:
            raise RuntimeError(f"Formal protocol mismatch {key}={got!r} in {output_dir}")
    if metrics.get("checkpoint_selection") != "best_val_accuracy":
        raise RuntimeError(f"Metrics do not record best validation Accuracy selection in {output_dir}")
    required_metric_values = [
        metrics["metrics"]["val_acc"]["mean"], metrics["metrics"]["test_acc"]["mean"],
        metrics["metrics"]["test_macro_f1"]["mean"],
    ]
    if not all(math.isfinite(float(value)) for value in required_metric_values):
        raise RuntimeError(f"Non-finite metric in {output_dir}")

    # Compare the resolved scientific protocol to its frozen seed-42 run. The
    # only expected config differences are RNG seed, ablation identity, and
    # the run-specific checkpoint destination.
    reference_path = ROOT / "outputs/core_story_ablation/nc" / dataset / variant / "seed42/resolved_config.yaml"
    reference = OmegaConf.to_container(OmegaConf.load(reference_path), resolve=True)
    actual_compare = json.loads(json.dumps(config))
    reference_compare = json.loads(json.dumps(reference))
    for item in (actual_compare, reference_compare):
        item.pop("seed", None)
        item.pop("ablation", None)
        item.get("task", {}).pop("save_ckpt_path", None)
        # The seed-derived LP split is present in the shared dataset config but
        # is unused by task=nc. The NC split itself is explicitly fixed above.
        item.get("dataset", {}).pop("lp_split_path", None)
    if actual_compare != reference_compare:
        raise RuntimeError(f"Resolved scientific config differs from seed42 formal run: {output_dir}")

    sidecar = {
        "purpose": "figure5_robustness_matched_split",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_seed": seed,
        "split_seed": 42,
        "split_path": str(split_path),
        "split_sha256": expected_sha,
        "dataset": dataset,
        "ablation": variant,
        "K": k,
        "resolved_config": config,
        "best_epoch": metrics["best_epoch"],
        "validation_accuracy": metrics["metrics"]["val_acc"]["mean"],
        "test_accuracy": metrics["metrics"]["test_acc"]["mean"],
        "test_macro_f1": metrics["metrics"]["test_macro_f1"]["mean"],
        "checkpoint_path": str((output_dir / "best.pt").resolve()),
        "checkpoint_sha256": sha256_file(output_dir / "best.pt"),
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "protocol_name": "unified_full_graph_nc_v1",
        "checkpoint_selection": metrics["checkpoint_selection"],
        "ablation_semantics": expected_modes,
        "status": "PASS",
    }
    sidecar_path = output_dir / "robustness_manifest.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[PASS] {dataset}/{variant}/seed{seed} best_epoch={sidecar['best_epoch']} "
        f"val_acc={sidecar['validation_accuracy']:.6f} test_acc={sidecar['test_accuracy']:.6f} "
        f"test_f1={sidecar['test_macro_f1']:.6f}",
        flush=True,
    )
    return sidecar


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true", help="run the split SHA audit without training")
    args = parser.parse_args()
    _, split_shas = audit_before_training()
    if args.preflight_only:
        return
    records: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                records.append(run_one(dataset, variant, seed, args.device, split_shas[dataset]))
    expected_identities = {
        (dataset, variant, seed)
        for dataset in DATASETS for variant in VARIANTS for seed in SEEDS
    }
    observed_identities = {
        (record["dataset"], record["ablation"], int(record["training_seed"]))
        for record in records
    }
    if len(records) != 8 or observed_identities != expected_identities:
        raise RuntimeError(f"Expected exactly 8 complete runs, observed {len(records)}")
    (OUT / "run_manifest.json").write_text(
        json.dumps({
            "status": "PASS",
            "purpose": "figure5_robustness_matched_split",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": records[0]["git_commit"],
            "planned_runs": 8,
            "completed_runs": len(records),
            "dataset_split_sha256": split_shas,
            "runs": records,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(OUT / "run_audit.csv", records)
    print("Matched-split checkpoint training complete: 8/8 PASS", flush=True)


if __name__ == "__main__":
    main()
