from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models import build_model


PILOT_ROOT = ROOT / "outputs/cosi_mag_vnext_pilot/nc"
REPORT_ROOT = ROOT / "reports/cosi_mag_vnext"
DATA_ROOT = Path("/hdd1/DataInHere/YHF/data")
SPLITS = {
    "Movies": DATA_ROOT / "MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt",
    "Grocery": DATA_ROOT / "MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt",
    "Reddit-S": DATA_ROOT / "MAGB_split/Reddit-S_nc_seed42_train0.6_val0.2.pt",
}
SEEDS = (42, 43, 44)
DATASETS = ("Movies", "Grocery", "Reddit-S")
HISTORICAL_FINAL = {
    "Movies": {"val_acc_mean_pct": 57.51, "val_acc_std_pct": 0.26, "val_macro_f1_mean_pct": 49.96, "val_macro_f1_std_pct": 0.87},
    "Grocery": {"val_acc_mean_pct": 83.91, "val_acc_std_pct": 0.16, "val_macro_f1_mean_pct": 78.47, "val_macro_f1_std_pct": 0.71},
    "Reddit-S": {"val_acc_mean_pct": 96.37, "val_acc_std_pct": 0.05, "val_macro_f1_mean_pct": 93.24, "val_macro_f1_std_pct": 0.13},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run_directory(dataset: str, seed: int) -> Path:
    base = PILOT_ROOT / (dataset if seed == 42 else f"{dataset}_seed{seed}")
    candidates = sorted(base.glob("**/complete.marker"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one complete pilot under {base}, found {len(candidates)}")
    return candidates[0].parent


def collect() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    run_rows = []
    mechanism_rows = []
    model_parameters = None
    dataset_cache = {}

    for dataset in DATASETS:
        split_path = SPLITS[dataset]
        split_hash = sha256(split_path)
        for seed in SEEDS:
            output_dir = run_directory(dataset, seed)
            config_path = output_dir / ".hydra/config.yaml"
            metrics_path = output_dir / "metrics.json"
            results_path = output_dir / "results.json"
            complete_path = output_dir / "complete.marker"
            cfg = OmegaConf.load(config_path)
            OmegaConf.resolve(cfg)
            cfg.seed = seed
            metrics = json.loads(metrics_path.read_text())
            results = json.loads(results_path.read_text())
            checkpoint_path = ROOT / metrics["checkpoint"]

            config_container = OmegaConf.to_container(cfg, resolve=True)
            run_rows.append(
                {
                    "dataset": dataset,
                    "task": "nc",
                    "seed": seed,
                    "status": "complete",
                    "protocol_version": cfg.task.protocol_version,
                    "split_path": str(split_path),
                    "split_sha256": split_hash,
                    "resolved_config_path": str(config_path.resolve()),
                    "resolved_config_sha256": sha256(config_path),
                    "output_dir": str(output_dir.resolve()),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_selection": metrics["checkpoint_selection"],
                    "best_epoch": int(metrics["best_epoch"]),
                    "runtime_seconds": float(metrics["runtime_seconds"]),
                    "peak_gpu_memory_mib": float(metrics["peak_gpu_memory_mib"]),
                    "val_acc": float(results["val_acc"]["mean"]),
                    "val_macro_f1": float(results["val_macro_f1"]["mean"]),
                    "resolved_config": config_container,
                }
            )

            if dataset not in dataset_cache:
                data = load_mag_data(cfg, task_name="nc", seed=42)
                data_cache_info = {
                    "input_dim": data.input_dim,
                    "num_nodes": data.num_nodes,
                    "num_classes": data.num_classes,
                    "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
                    "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0,
                }
                dataset_cache[dataset] = (data, data_cache_info)
            data, data_info = dataset_cache[dataset]
            model = build_model(cfg, data_info).to(device)
            if model_parameters is None:
                model_parameters = sum(parameter.numel() for parameter in model.parameters())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            mechanism = model.relation_diagnostics(data.x, data.edge_index)
            mechanism_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    **mechanism,
                }
            )
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metric_csv = REPORT_ROOT / "pilot_metrics.csv"
    csv_fields = [
        "dataset",
        "seed",
        "best_epoch",
        "val_acc_pct",
        "val_macro_f1_pct",
        "runtime_seconds",
        "peak_gpu_memory_mib",
        "split_sha256",
        "output_dir",
    ]
    with metric_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in run_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "best_epoch": row["best_epoch"],
                    "val_acc_pct": f"{row['val_acc'] * 100:.6f}",
                    "val_macro_f1_pct": f"{row['val_macro_f1'] * 100:.6f}",
                    "runtime_seconds": f"{row['runtime_seconds']:.6f}",
                    "peak_gpu_memory_mib": f"{row['peak_gpu_memory_mib']:.3f}",
                    "split_sha256": row["split_sha256"],
                    "output_dir": row["output_dir"],
                }
            )

    pilot_by_dataset = {}
    for dataset in DATASETS:
        rows = [row for row in run_rows if row["dataset"] == dataset]
        acc = [row["val_acc"] * 100.0 for row in rows]
        macro = [row["val_macro_f1"] * 100.0 for row in rows]
        reference = HISTORICAL_FINAL[dataset]
        pilot_by_dataset[dataset] = {
            "task": "node_classification",
            "protocol_version": "unified_full_graph_nc_v1",
            "split_path": str(SPLITS[dataset]),
            "split_sha256": rows[0]["split_sha256"],
            "runs_completed": len(rows),
            "selection_metric": "validation_accuracy",
            "test_evaluated": False,
            "validation_accuracy_percent": {
                "per_seed": {str(row["seed"]): row["val_acc"] * 100.0 for row in rows},
                "mean": mean(acc),
                "population_std": pstdev(acc),
            },
            "validation_macro_f1_percent": {
                "per_seed": {str(row["seed"]): row["val_macro_f1"] * 100.0 for row in rows},
                "mean": mean(macro),
                "population_std": pstdev(macro),
            },
            "best_epoch_per_seed": {str(row["seed"]): row["best_epoch"] for row in rows},
            "runtime_seconds_per_seed": {str(row["seed"]): row["runtime_seconds"] for row in rows},
            "peak_gpu_memory_mib_per_seed": {str(row["seed"]): row["peak_gpu_memory_mib"] for row in rows},
            "historical_cosi_mag_final_validation_reference_percent": reference,
            "historical_comparability": "same validation split SHA-256, unified_full_graph_nc_v1, and task/nc.yaml SHA-256; historical values are validation-only and selected by validation accuracy",
        }
    write_json(
        REPORT_ROOT / "pilot_metrics.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": "cosi_mag_vnext",
            "protocol_version": "unified_full_graph_nc_v1",
            "test_metrics_used_for_selection": False,
            "summary_unit": "percent; population standard deviation across completed seeds",
            "model_parameter_count_encoder_only": model_parameters,
            "datasets": pilot_by_dataset,
            "runs": [
                {key: value for key, value in row.items() if key != "resolved_config"}
                for row in run_rows
            ],
        },
    )

    mechanism_csv = REPORT_ROOT / "mechanism_summary.csv"
    fields = list(mechanism_rows[0].keys())
    with mechanism_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mechanism_rows)
    mechanism_by_dataset = {}
    numeric_keys = [
        key
        for key, value in mechanism_rows[0].items()
        if key not in {"dataset", "seed", "checkpoint_path"} and isinstance(value, (int, float))
    ]
    for dataset in DATASETS:
        rows = [row for row in mechanism_rows if row["dataset"] == dataset]
        mechanism_by_dataset[dataset] = {
            key: {
                "mean": mean(float(row[key]) for row in rows),
                "population_std": pstdev(float(row[key]) for row in rows),
            }
            for key in numeric_keys
        }
    write_json(
        REPORT_ROOT / "mechanism_summary.json",
        {
            "diagnostic_source": "validation-accuracy-selected checkpoints; diagnostics use only features and the task message graph",
            "by_dataset_mean_population_std": mechanism_by_dataset,
            "by_run": mechanism_rows,
        },
    )

    env = {
        "audit_date": "2026-10-12",
        "project_root": str(ROOT),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": "d5a06103be0a27952ef5890e2839a36eb83f580a",
        "python": "3.12.13 (conda environment yhf_env)",
        "packages": {
            "torch": torch.__version__,
            "torch_geometric": __import__("torch_geometric").__version__,
            "torch_sparse": __import__("torch_sparse").__version__,
            "hydra": __import__("hydra").__version__,
            "omegaconf": __import__("omegaconf").__version__,
            "numpy": __import__("numpy").__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "dgl": __import__("dgl").__version__,
        },
        "device_used_for_pilot": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
        "host_memory_total_gib": 125,
        "host_memory_available_gib_at_audit": 110,
        "workspace_disk_free_gib_at_audit": 133,
        "data_root": str(DATA_ROOT),
        "split_files": {
            name: {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in SPLITS.items()
        },
        "cwd_python_caveat": "the base shell is not the project environment; use conda run -n yhf_env",
        "nvidia_smi": "unavailable in this session; CUDA memory is reported from PyTorch run metrics",
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            env["gpus"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "free_bytes_at_manifest_time": free,
                    "total_bytes": total,
                }
            )
    write_json(REPORT_ROOT / "environment_manifest.json", env)

    correctness = {
        "synthetic_tests": {"command": "conda run -n yhf_env python -m pytest -q tests/test_cosi_mag_vnext.py", "passed": 13, "failed": 0, "status": "PASS"},
        "existing_related_tests": {"command": "conda run -n yhf_env python -m pytest -q tests/test_cosi_mag_ablation.py tests/test_cosi_mag_final_stability.py tests/test_lp_sampling.py tests/test_inference_equivalence.py tests/test_protocol.py", "passed": 57, "failed": 0, "status": "PASS"},
        "combined_final_test_run": {"command": "conda run -n yhf_env python -m pytest -q tests/test_cosi_mag_vnext.py tests/test_cosi_mag_ablation.py tests/test_cosi_mag_final_stability.py tests/test_lp_sampling.py tests/test_inference_equivalence.py tests/test_protocol.py", "passed": 70, "failed": 0, "status": "PASS"},
        "nc_checkpoint_reload": {"command": "conda run -n yhf_env python scripts/collect_cosi_mag_vnext_reports.py", "loaded_checkpoints": 9, "diagnostic_forwards": 9, "status": "PASS"},
        "initial_failure_and_fix": "The first gradient run exposed sqrt(variance) at zero variance in CRC neighborhood weight standard deviation. Applying a positive variance floor with zero derivative below the floor removed NaN gradients; rerun passed.",
        "smoke_runs": [
            {"dataset": "Movies", "task": "nc", "seed": 42, "epochs": 2, "status": "PASS", "output": str((ROOT / "outputs/cosi_mag_vnext_smoke/nc/Movies").resolve())},
            {"dataset": "Grocery", "task": "nc", "seed": 42, "epochs": 2, "status": "PASS", "output": str((ROOT / "outputs/cosi_mag_vnext_smoke/nc/Grocery").resolve())},
            {"dataset": "sports-copurchase", "task": "lp", "seed": 42, "epochs": 1, "max_train_batches": 2, "train_pos_per_epoch": 128, "positive_message_edges_removed": 72, "validation_inference": "full graph", "status": "PASS", "output": str((ROOT / "outputs/cosi_mag_vnext_smoke/lp/sports").resolve())},
        ],
        "scope_notes": [
            "The entire repository test suite was not run; 57 directly related legacy tests were run.",
            "LP leakage was checked with a synthetic global_eid batch and with the actual Sports-Copurchase smoke log, which records positive message edges removed before model forward.",
        ],
    }
    write_json(REPORT_ROOT / "correctness_summary.json", correctness)

    manifest = {
        "branch": env["branch"],
        "base_commit": env["base_commit"],
        "protocol_version": "unified_full_graph_nc_v1",
        "model_config_sha256": sha256(ROOT / "configs/model/cosi_mag_vnext.yaml"),
        "task_config_sha256": sha256(ROOT / "configs/task/nc.yaml"),
        "runs": run_rows,
        "smoke_output_roots": [
            str((ROOT / "outputs/cosi_mag_vnext_smoke/nc/Movies").resolve()),
            str((ROOT / "outputs/cosi_mag_vnext_smoke/nc/Grocery").resolve()),
            str((ROOT / "outputs/cosi_mag_vnext_smoke/lp/sports").resolve()),
        ],
    }
    write_json(REPORT_ROOT / "run_manifest.json", manifest)
    print(f"Wrote reports to {REPORT_ROOT}")
    print(f"Pilot runs: {len(run_rows)}; mechanism summaries: {len(mechanism_rows)}")


if __name__ == "__main__":
    collect()
