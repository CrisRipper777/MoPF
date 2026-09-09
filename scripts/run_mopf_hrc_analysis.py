"""Export M0 coefficient audits and HRC comparison metrics for M1-A runs.

The script is analysis-only: it never retrains a checkpoint.  Each run
directory must contain the seed-42 best-validation-accuracy checkpoint and
its resolved Hydra config.  The existing M0 exporter is reused so every M1
checkpoint has the same coefficient/effective-radius evidence as M0.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_mopf_mechanism_audit import (  # noqa: E402
    DATASETS,
    _load_checkpoint_bundle,
    export_audit,
)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _metric_row(
    dataset: str,
    hrc_weight: float,
    modality: str,
    order: int,
    delta_node: torch.Tensor,
    eta: torch.Tensor,
    delta_gamma: torch.Tensor,
) -> dict[str, Any]:
    mean = _scalar(delta_node.mean())
    std = _scalar(delta_node.std(unbiased=False))
    return {
        "dataset": dataset,
        "hrc_weight": hrc_weight,
        "modality": modality,
        "order": order,
        "delta_node_mean": mean,
        "abs_delta_node_mean": abs(mean),
        "delta_node_std": std,
        "abs_mean_over_std_eps": abs(mean) / (std + 1e-8),
        "eta_mean": _scalar(eta.mean()),
        "delta_gamma": _scalar(delta_gamma[order]),
        "num_training_nodes": int(delta_node.numel()),
    }


def summarize_run(dataset: str, output_dir: Path, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    checkpoint = output_dir / "best_val_accuracy.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing best-validation checkpoint: {checkpoint}")

    # Re-export all M0 artifacts from this exact best-validation checkpoint.
    export_audit(dataset, output_dir, device_name)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
    _, _, data, _ = _load_checkpoint_bundle(output_dir, torch.device("cpu"))
    coefficients = torch.load(output_dir / "coefficients.pt", map_location="cpu", weights_only=False)
    train_idx = data.train_idx.cpu()
    hrc_weight = float(cfg.model.get("hrc_weight", 0.0))

    rows: list[dict[str, Any]] = []
    for modality in ("text", "visual"):
        delta_node = coefficients[f"delta_node_{modality}"][train_idx]
        eta = coefficients[f"eta_{modality}"][train_idx]
        delta_gamma = coefficients[f"delta_gamma_{modality}"]
        for order in range(int(delta_node.size(1))):
            rows.append(
                _metric_row(
                    dataset,
                    hrc_weight,
                    modality,
                    order,
                    delta_node[:, order],
                    eta[:, order],
                    delta_gamma,
                )
            )

    fieldnames = list(rows[0])
    with (output_dir / "hrc_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    hrc_raw_from_export = float(
        np.mean(
            [
                row["delta_node_mean"] ** 2
                for row in rows
            ]
        )
    )
    summary = {
        "dataset": dataset,
        "seed": int(payload["seed"]),
        "hrc_weight": hrc_weight,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metrics": payload.get("metrics", {}),
        "num_training_nodes": int(train_idx.numel()),
        "hrc_raw_loss_from_training_node_means": hrc_raw_from_export,
        "metric_definitions": {
            "delta_node_mean": "mean over current NC training nodes",
            "delta_node_std": "population std over current NC training nodes",
            "abs_mean_over_std_eps": "abs(mean)/(population_std+1e-8), descriptive only",
            "eta_mean": "mean effective coefficient over current NC training nodes",
            "delta_gamma": "learned modality residual coefficient",
            "hrc_raw_loss": "mean over two modalities and all orders of squared training-node means",
        },
        "rows": rows,
        "m0_artifacts": [
            "coefficients.pt",
            "distribution_summary.csv",
            "node_profiles.csv",
            "modality_summary.csv",
            "structure_profile_summary.csv",
            "audit_summary.json",
        ],
    }
    (output_dir / "hrc_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[m1-hrc-analysis] {dataset} lambda={hrc_weight:g} "
        f"output={output_dir}",
        flush=True,
    )
    return summary


def write_comparison(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [row for summary in summaries for row in summary["rows"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "hrc_compare.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison = {
        "datasets": [summary["dataset"] for summary in summaries],
        "runs": [
            {
                "dataset": summary["dataset"],
                "hrc_weight": summary["hrc_weight"],
                "checkpoint_epoch": summary["checkpoint_epoch"],
                "checkpoint_metrics": summary["checkpoint_metrics"],
                "hrc_raw_loss_from_training_node_means": summary[
                    "hrc_raw_loss_from_training_node_means"
                ],
            }
            for summary in summaries
        ],
        "interpretation": {
            "priority": [
                "identifiability via lower abs_delta_node_mean",
                "preservation of delta_node_std",
                "stable eta_mean",
                "validation metrics in the same band",
            ],
            "hierarchical_transfer": "Compare delta_gamma and eta_mean as delta_node_mean changes; descriptive only, not an exact conservation claim.",
        },
    }
    (output_dir / "hrc_compare.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "m1_hrc",
    )
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="all",
    )
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_dir = args.output_root / dataset
        for lambda_dir in sorted(dataset_dir.glob("lambda*")):
            if not (lambda_dir / "best_val_accuracy.pt").is_file():
                continue
            summaries.append(summarize_run(dataset, lambda_dir, args.device))
    if not summaries:
        raise FileNotFoundError(f"No M1-A run directories found under {args.output_root}")
    write_comparison(summaries, args.output_root)


if __name__ == "__main__":
    main()
