#!/usr/bin/env python3
"""Audit learned CoSI-MAG mechanisms on the five final NC checkpoints.

This script is inference-only.  It measures how far learned MRC parameters
move from uniform weighting, how strongly the RCMI control path is gated, and
the frozen-checkpoint effect of disabling relation context or cross-order
interaction.  It never constructs an optimizer or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
ACTIVE_ALL_PLAIN_PREFIXES = (
    "text_proj.",
    "visual_proj.",
    "text_refine_mlp.",
    "visual_refine_mlp.",
    "text_refine_norm.",
    "visual_refine_norm.",
    "fusion_skip.",
    "fusion_mlp.",
    "output_norm.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "outputs" / "cosi_mag_final_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "cosi_mag_final_joint_ablation" / "audit",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def checkpoint_path(root: Path, dataset: str, seed: int) -> Path:
    return root / "nc" / dataset / "runs_42_43_44" / f"best_run{SEEDS.index(seed) + 1}.pt"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def eval_labels(data: Any) -> list[int]:
    idx = torch.cat([data.train_idx, data.val_idx, data.test_idx])
    labels = data.y[idx]
    return sorted({int(value) for value in labels.tolist() if 0 <= int(value) < int(data.num_classes)})


@torch.no_grad()
def evaluate(classifier: nn.Module, z: torch.Tensor, data: Any, labels: list[int]) -> dict[str, Any]:
    idx = data.test_idx.to(z.device)
    logits = classifier(z[idx])
    pred = logits.argmax(dim=-1).cpu()
    target = data.y[data.test_idx].cpu()
    return {
        "acc": float((pred == target).float().mean().item()),
        "macro_f1": float(
            f1_score(target.numpy(), pred.numpy(), labels=labels, average="macro", zero_division=0)
        ),
        "pred": pred,
    }


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    x = value.detach().float()
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75], device=x.device))
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "q25": float(q[0].item()),
        "median": float(q[1].item()),
        "q75": float(q[2].item()),
        "max": float(x.max().item()),
    }


def finite_row(row: dict[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite {key}: {value}")


def main() -> int:
    args = parse_args()
    from src.data import load_mag_data
    from src.models import build_model

    device = torch.device(args.device)
    parameter_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        first = checkpoint_path(args.benchmark_root, dataset, SEEDS[0])
        cfg = OmegaConf.load(first.parent / "resolved_config.yaml")
        data = load_mag_data(cfg, "nc", int(cfg.seed))
        labels = eval_labels(data)
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        degree = torch.bincount(edge_index[0], minlength=data.num_nodes)
        isolated = degree == 0

        for seed in SEEDS:
            path = checkpoint_path(args.benchmark_root, dataset, seed)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(cfg, payload["data_info"]).to(device).eval()
            model.load_state_dict(payload["model_state"], strict=True)
            classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device).eval()
            classifier.load_state_dict(payload["head_state"], strict=True)

            model_parameters = sum(value.numel() for value in payload["model_state"].values())
            active_plain_parameters = sum(
                value.numel()
                for name, value in payload["model_state"].items()
                if name.startswith(ACTIVE_ALL_PLAIN_PREFIXES)
            )
            head_parameters = sum(value.numel() for value in payload["head_state"].values())
            parameter_row: dict[str, Any] = {
                "dataset": dataset,
                "seed": seed,
                "epoch": int(payload["epoch"]),
                "model_parameters": model_parameters,
                "all_plain_active_encoder_parameters": active_plain_parameters,
                "nc_head_parameters": head_parameters,
                "all_plain_active_fraction": active_plain_parameters / model_parameters,
            }
            for modality in ("text", "visual"):
                metric = model._normalized_metric_weights(
                    getattr(model, f"metric_theta_{modality}")
                ).detach().float()
                parameter_row[f"metric_cv_{modality}"] = float(
                    metric.std(unbiased=False).item() / metric.mean().item()
                )
                parameter_row[f"hop_gate_{modality}"] = float(
                    torch.tanh(getattr(model, f"theta_hop_gate_{modality}")).item()
                )
                parameter_row[f"relation_scale_{modality}"] = float(
                    torch.sigmoid(getattr(model, f"theta_relation_scale_{modality}")).item()
                )
                parameter_row[f"node_vector_rms_{modality}"] = float(
                    getattr(model, f"node_vector_{modality}").square().mean().sqrt().item()
                )
            finite_row(parameter_row)
            parameter_rows.append(parameter_row)

            normal = model.analysis_intervention(x, edge_index)
            normal_eval = evaluate(classifier, normal["z"], data, labels)
            unit_index, unit_weight = model._normalized_operator(
                edge_index,
                torch.ones(edge_index.size(1), device=device, dtype=x.dtype),
                data.num_nodes,
                x.dtype,
            )

            for modality in ("text", "visual"):
                weights = normal[f"relation_weight_{modality}"]
                stats = tensor_stats(weights)
                norm_index = normal[f"normalized_edge_index_{modality}"]
                norm_weight = normal[f"normalized_edge_weight_{modality}"]
                if not torch.equal(norm_index, unit_index):
                    raise ValueError(f"normalized edge ordering changed for {dataset}/{seed}/{modality}")
                context = normal[f"local_relation_context_{modality}"]
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "modality": modality,
                    "num_nodes": data.num_nodes,
                    "num_physical_edges": int(edge_index.size(1)),
                    "isolated_nodes": int(isolated.sum().item()),
                    "relation_weight_mean": stats["mean"],
                    "relation_weight_std": stats["std"],
                    "relation_weight_cv": stats["std"] / stats["mean"],
                    "relation_weight_min": stats["min"],
                    "relation_weight_q25": stats["q25"],
                    "relation_weight_median": stats["median"],
                    "relation_weight_q75": stats["q75"],
                    "relation_weight_max": stats["max"],
                    "normalized_operator_weight_mae_vs_unit": float(
                        (norm_weight - unit_weight).abs().mean().item()
                    ),
                    "local_context_mean_abs": float(context.abs().mean().item()),
                    "isolated_context_mean_abs": (
                        float(context[isolated].abs().mean().item()) if bool(isolated.any()) else 0.0
                    ),
                    "node_delta_mean_abs": float(normal[f"delta_{modality}"].abs().mean().item()),
                    "eta_mean_abs": float(normal[f"eta_{modality}"].abs().mean().item()),
                    "effective_order_mean": float(normal[f"effective_order_{modality}"].mean().item()),
                }
                finite_row(row)
                mechanism_rows.append(row)

            for condition, kwargs in (
                ("relation_context_off", {"relation": "off"}),
                ("cross_order_interaction_off", {"interaction": "off"}),
            ):
                changed = model.analysis_intervention(x, edge_index, **kwargs)
                changed_eval = evaluate(classifier, changed["z"], data, labels)
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "condition": condition,
                    "z_mae": float((changed["z"] - normal["z"]).abs().mean().item()),
                    "prediction_flip_rate": float(
                        (changed_eval["pred"] != normal_eval["pred"]).float().mean().item()
                    ),
                    "test_acc_normal": normal_eval["acc"],
                    "test_acc_changed": changed_eval["acc"],
                    "test_acc_delta_changed_minus_normal": changed_eval["acc"] - normal_eval["acc"],
                    "test_macro_f1_normal": normal_eval["macro_f1"],
                    "test_macro_f1_changed": changed_eval["macro_f1"],
                    "test_macro_f1_delta_changed_minus_normal": (
                        changed_eval["macro_f1"] - normal_eval["macro_f1"]
                    ),
                }
                finite_row(row)
                intervention_rows.append(row)
                del changed

            del normal, model, classifier, unit_index, unit_weight
            torch.cuda.empty_cache()
            print(f"[AUDITED] {dataset} seed={seed}", flush=True)

        del x, edge_index
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "checkpoint_parameter_stats.csv", parameter_rows)
    write_csv(args.output_dir / "mechanism_activation_stats.csv", mechanism_rows)
    write_csv(args.output_dir / "frozen_intervention_metrics.csv", intervention_rows)

    summary = {
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "device": str(device),
        "training_performed": False,
        "parameter_rows": len(parameter_rows),
        "mechanism_rows": len(mechanism_rows),
        "intervention_rows": len(intervention_rows),
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
