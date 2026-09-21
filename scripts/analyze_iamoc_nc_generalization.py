#!/usr/bin/env python3
"""Summarize the IAMOC NC factorial grid and audit V2 mechanisms/interventions."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_iamoc_v1 import _downstream_metrics  # noqa: E402
from scripts.diagnose_iamoc_v1_mechanism import (  # noqa: E402
    _distribution,
    _pair_node_metrics,
    _profile,
    _query_diversity,
    _run_order_embedding_off,
    _stage12_diff,
)
from scripts.run_iamoc_nc_generalization import _legacy_reuse_audit  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
VARIANTS = ("V00", "V0", "V1", "V2")
SEEDS = (42, 43, 44)
METRICS = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
COMPARISONS = {
    "A_hop_without_tcpR": ("V2", "V00"),
    "B_hop_with_tcpR": ("V1", "V0"),
    "C_tcpR_without_hop": ("V0", "V00"),
    "D_tcpR_with_hop": ("V1", "V2"),
}
FACTORIAL_EFFECTS = ("hop_main_effect", "tcpr_main_effect", "interaction_effect")
STAGE12_ABS_TOLERANCE = 1e-4


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: math.nan for key in ("mean", "std_population", "median", "min", "max")}
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _checkpoint_dir(
    dataset: str,
    variant: str,
    seed: int,
    output_root: Path,
    legacy_root: Path,
) -> Path:
    current = output_root / dataset / variant / f"seed_{seed}"
    if (current / "best.pt").is_file() and (current / "resolved_config.yaml").is_file():
        return current
    reuse_path = current / "reuse_manifest.json"
    if reuse_path.is_file():
        reuse = _json(reuse_path).get("reuse_audit", {})
        source = Path(reuse.get("source_run_dir", ""))
        checkpoint = Path(reuse.get("source_checkpoint", ""))
        if source.is_dir() and checkpoint.is_file():
            expected_hash = reuse.get("source_checkpoint_sha256")
            if expected_hash and _sha256(checkpoint) == expected_hash:
                return source
            raise RuntimeError(f"Reused source checkpoint hash changed: {checkpoint}")
        raise FileNotFoundError(f"Reuse manifest points to a missing source checkpoint: {reuse_path}")
    if dataset in {"Movies", "Grocery"} and variant in {"V0", "V1", "V2"}:
        legacy = legacy_root / dataset / variant / f"seed_{seed}"
        if (legacy / "best.pt").is_file() and (legacy / "resolved_config.yaml").is_file():
            if _legacy_reuse_audit(dataset, variant, seed, legacy_root) is not None:
                return legacy
    raise FileNotFoundError(f"No complete checkpoint for {dataset}/{variant}/seed_{seed}")


def _validate_run(run_dir: Path, dataset: str, variant: str, seed: int) -> dict[str, Any]:
    required = ("complete.marker", "metrics.json", "results.json", "best.pt", "resolved_config.yaml", "resolved_config.json", "run_record.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete run {run_dir}: {missing}")
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    expected_model = "mopf" if variant in {"V00", "V0"} else "mopf_iamoc"
    if str(cfg.dataset.name) != dataset or str(cfg.task.name) != "nc" or int(cfg.seed) != seed:
        raise ValueError(f"Dataset/task/seed mismatch in {run_dir}")
    if str(cfg.model.name) != expected_model:
        raise ValueError(f"Model mismatch in {run_dir}: expected {expected_model}")
    if str(cfg.task.protocol_version) != "unified_full_graph_nc_v1":
        raise ValueError(f"Unexpected NC protocol in {run_dir}")
    if str(cfg.task.training_mode) != "full_graph" or int(cfg.task.epochs) != 300:
        raise ValueError(f"Unexpected NC training configuration in {run_dir}")
    run_record = _json(run_dir / "run_record.json")
    if run_record.get("status") != "complete" or run_record.get("return_code") != 0:
        raise ValueError(f"Run record is not complete in {run_dir}")
    if variant == "V00" and bool(cfg.model.use_transport_residual):
        raise ValueError(f"V00 has TCPR enabled in {run_dir}")
    if variant == "V0" and not bool(cfg.model.use_transport_residual):
        raise ValueError(f"V0 has TCPR disabled in {run_dir}")
    if variant == "V1" and (
        int(cfg.model.hop_interaction_layers) != 1
        or str(cfg.model.relation_conditioning) != "output"
    ):
        raise ValueError(f"V1 settings mismatch in {run_dir}")
    if variant == "V2" and (
        int(cfg.model.hop_interaction_layers) != 1
        or str(cfg.model.relation_conditioning) != "none"
    ):
        raise ValueError(f"V2 settings mismatch in {run_dir}")
    metrics = _json(run_dir / "metrics.json")
    if metrics.get("checkpoint_selection") != "best_val_accuracy":
        raise ValueError(f"Checkpoint selection mismatch in {run_dir}")
    for metric in METRICS:
        if metric not in metrics.get("metrics", {}):
            raise ValueError(f"Missing {metric} in {run_dir}/metrics.json")
    return metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _main_grid(output_root: Path, legacy_root: Path) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                run_dir = _checkpoint_dir(dataset, variant, seed, output_root, legacy_root)
                metrics_doc = _validate_run(run_dir, dataset, variant, seed)
                resolved = _json(run_dir / "resolved_config.json")
                dataset_config = resolved.get("dataset", {})
                split_value = dataset_config.get("nc_split_path") or dataset_config.get("node_split_path")
                if not split_value:
                    raise ValueError(f"No resolved NC split path in {run_dir}")
                split_path = Path(split_value)
                if not split_path.is_file():
                    raise FileNotFoundError(f"Resolved NC split is missing: {split_path}")
                row = {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "best_epoch": int(metrics_doc["best_epoch"]),
                    "run_dir": str(run_dir),
                    "checkpoint": str(run_dir / "best.pt"),
                    "checkpoint_sha256": _sha256(run_dir / "best.pt"),
                    "split_path": str(split_path),
                    "split_sha256_current": _sha256(split_path),
                    **{
                        metric: float(metrics_doc["metrics"][metric]["mean"])
                        for metric in METRICS
                    },
                }
                runs[(dataset, variant, seed)] = row
                run_rows.append(row)
            dataset_runs = [runs[(dataset, variant, seed)] for seed in SEEDS]
            summary: dict[str, Any] = {
                "dataset": dataset,
                "variant": variant,
                "best_epoch_seeds": ",".join(str(row["best_epoch"]) for row in dataset_runs),
                "best_epoch_mean": float(np.mean([row["best_epoch"] for row in dataset_runs])),
                "best_epoch_std_population": float(np.std([row["best_epoch"] for row in dataset_runs], ddof=0)),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in dataset_runs]
                stat = _stats(values)
                summary[f"{metric}_mean"] = stat["mean"]
                summary[f"{metric}_std_population"] = stat["std_population"]
            summary_rows.append(summary)
    return runs, run_rows, summary_rows


def _paired_and_factorial(
    runs: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paired_seed_rows: list[dict[str, Any]] = []
    paired_summary_rows: list[dict[str, Any]] = []
    factorial_seed_rows: list[dict[str, Any]] = []
    factorial_summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for comparison, (positive_variant, reference_variant) in COMPARISONS.items():
            for metric in METRICS:
                deltas = []
                for seed in SEEDS:
                    delta = float(runs[(dataset, positive_variant, seed)][metric] - runs[(dataset, reference_variant, seed)][metric])
                    deltas.append(delta)
                    paired_seed_rows.append(
                        {
                            "dataset": dataset,
                            "comparison": comparison,
                            "variant_delta": f"{positive_variant}-{reference_variant}",
                            "metric": metric,
                            "seed": seed,
                            "delta": delta,
                        }
                    )
                stat = _stats(deltas)
                paired_summary_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": comparison,
                        "variant_delta": f"{positive_variant}-{reference_variant}",
                        "metric": metric,
                        **stat,
                        "positive_seed_count": int(sum(delta > 0 for delta in deltas)),
                        "paired_seed_count": len(deltas),
                    }
                )

        for metric in METRICS:
            effect_values: dict[str, list[float]] = {name: [] for name in FACTORIAL_EFFECTS}
            for seed in SEEDS:
                v00 = float(runs[(dataset, "V00", seed)][metric])
                v0 = float(runs[(dataset, "V0", seed)][metric])
                v1 = float(runs[(dataset, "V1", seed)][metric])
                v2 = float(runs[(dataset, "V2", seed)][metric])
                values = {
                    "hop_main_effect": 0.5 * ((v2 - v00) + (v1 - v0)),
                    "tcpr_main_effect": 0.5 * ((v0 - v00) + (v1 - v2)),
                    "interaction_effect": (v1 - v0) - (v2 - v00),
                }
                for effect, value in values.items():
                    effect_values[effect].append(float(value))
                    factorial_seed_rows.append(
                        {"dataset": dataset, "metric": metric, "effect": effect, "seed": seed, "delta": float(value)}
                    )
            for effect, deltas in effect_values.items():
                factorial_summary_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "effect": effect,
                        **_stats(deltas),
                        "positive_seed_count": int(sum(value > 0 for value in deltas)),
                        "paired_seed_count": len(deltas),
                    }
                )

    across_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for effect in FACTORIAL_EFFECTS:
            values = [
                row["delta"]
                for row in factorial_seed_rows
                if row["metric"] == metric and row["effect"] == effect
            ]
            dataset_means = [
                row["mean"]
                for row in factorial_summary_rows
                if row["metric"] == metric and row["effect"] == effect
            ]
            across_rows.append(
                {
                    "metric": metric,
                    "effect": effect,
                    **_stats(values),
                    "positive_dataset_count": int(sum(value > 0 for value in dataset_means)),
                    "dataset_count": len(dataset_means),
                    "paired_positive_seed_count": int(sum(value > 0 for value in values)),
                    "paired_seed_count": len(values),
                }
            )
    return paired_seed_rows, paired_summary_rows, factorial_seed_rows, factorial_summary_rows, across_rows


def _quantile_payload(tensor: torch.Tensor) -> dict[str, float]:
    return _distribution(tensor.detach().float().cpu().numpy())


def _node_intervention_arrays(
    normal: dict[str, Any], changed: dict[str, Any], normal_logits: torch.Tensor, changed_logits: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, float]]]]:
    return _pair_node_metrics(normal, changed, normal_logits, changed_logits)


def _write_mechanism_nodes(
    writer: csv.DictWriter,
    dataset: str,
    seed: int,
    normal: dict[str, Any],
    query: dict[str, tuple[torch.Tensor, torch.Tensor]],
    intervention: dict[str, np.ndarray],
) -> None:
    profiles = {modality: _profile(normal[f"hop_attention_{modality}"]) for modality in ("text", "visual")}
    count = int(profiles["text"]["r_attn"].numel())
    arrays: dict[str, np.ndarray] = {}
    for modality in ("text", "visual"):
        profile = profiles[modality]
        arrays[f"{modality}_r_attn"] = profile["r_attn"].detach().cpu().numpy()
        arrays[f"{modality}_kl_uniform"] = profile["kl_uniform"].detach().cpu().numpy()
        arrays[f"{modality}_normalized_entropy"] = profile["normalized_entropy"].detach().cpu().numpy()
        arrays[f"{modality}_max_key_mass"] = profile["max_key_mass"].detach().cpu().numpy()
        arrays[f"{modality}_query_row_mae"] = query[modality][0].detach().cpu().numpy()
        arrays[f"{modality}_pairwise_query_js"] = query[modality][1].detach().cpu().numpy()
    intervention_columns = {
        "text_attention_mae_order_embedding_off": "attention_mae_text",
        "visual_attention_mae_order_embedding_off": "attention_mae_visual",
        "text_eta_mae_order_embedding_off": "eta_mae_text",
        "visual_eta_mae_order_embedding_off": "eta_mae_visual",
        "z_mae_order_embedding_off": "z_mae",
        "logit_mae_order_embedding_off": "logit_mae",
        "probability_kl_normal_to_off": "probability_kl_normal_to_intervention",
        "prediction_flip_normal_to_off": "prediction_flip",
    }
    for destination, source in intervention_columns.items():
        arrays[destination] = intervention[source]
    for node in range(count):
        row: dict[str, Any] = {"dataset": dataset, "seed": seed, "node_id": node}
        for name, values in arrays.items():
            row[name] = float(values[node])
        writer.writerow(row)


def _mechanism_analysis(
    output_root: Path,
    legacy_root: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    run_payloads: list[dict[str, Any]] = []
    node_path = analysis_root / "mechanism_node_metrics.csv.gz"
    mean_attention_csv = analysis_root / "mean_attention_matrices.csv"
    if mean_attention_csv.exists():
        mean_attention_csv.unlink()
    node_header = ["dataset", "seed", "node_id"]
    for modality in ("text", "visual"):
        node_header.extend(
            [
                f"{modality}_r_attn",
                f"{modality}_kl_uniform",
                f"{modality}_normalized_entropy",
                f"{modality}_max_key_mass",
                f"{modality}_query_row_mae",
                f"{modality}_pairwise_query_js",
            ]
        )
    node_header.extend(
        [
            "text_attention_mae_order_embedding_off",
            "visual_attention_mae_order_embedding_off",
            "text_eta_mae_order_embedding_off",
            "visual_eta_mae_order_embedding_off",
            "z_mae_order_embedding_off",
            "logit_mae_order_embedding_off",
            "probability_kl_normal_to_off",
            "prediction_flip_normal_to_off",
        ]
    )

    with gzip.open(node_path, "wt", newline="", encoding="utf-8", compresslevel=6) as handle:
        node_writer = csv.DictWriter(handle, fieldnames=node_header, extrasaction="ignore")
        node_writer.writeheader()
        for dataset in DATASETS:
            for seed in SEEDS:
                run_dir = _checkpoint_dir(dataset, "V2", seed, output_root, legacy_root)
                _validate_run(run_dir, dataset, "V2", seed)
                cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
                checkpoint_path = run_dir / "best.pt"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                metrics_doc = _validate_run(run_dir, dataset, "V2", seed)
                data = load_mag_data(cfg, "nc", seed)
                model = build_model(cfg, checkpoint["data_info"]).to(device)
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.eval()
                if not hasattr(model, "analysis_hop_attention"):
                    raise TypeError(f"V2 checkpoint lacks attention analysis API: {checkpoint_path}")
                before_hash = _parameter_hash(model)
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                with torch.inference_mode():
                    normal = model.analysis_hop_attention(x, edge_index)
                    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
                    head.load_state_dict(checkpoint["head_state"], strict=True)
                    head.eval()
                    normal_logits = head(normal["z"])
                    normal_metrics = _downstream_metrics(cfg, checkpoint, model, data, normal, device)
                    for metric in ("test_acc", "test_macro_f1"):
                        saved = float(metrics_doc["metrics"][metric]["mean"])
                        if abs(float(normal_metrics[metric]) - saved) > 1e-4:
                            raise AssertionError(
                                f"Recomputed normal {metric} differs from saved metric in {checkpoint_path}: "
                                f"{normal_metrics[metric]} vs {saved}"
                            )

                    profiles = {modality: _profile(normal[f"hop_attention_{modality}"]) for modality in ("text", "visual")}
                    query: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                    query_payload: dict[str, dict[str, float]] = {}
                    for modality in ("text", "visual"):
                        layers = normal[f"hop_attention_layers_{modality}"]
                        if len(layers) != 1:
                            raise ValueError(f"Expected 1 hop interaction layer, found {len(layers)} in {checkpoint_path}")
                        row_mae, pair_js = _query_diversity(layers[0])
                        query[modality] = (row_mae, pair_js)
                        query_payload[modality] = {
                            "query_row_mae": _quantile_payload(row_mae),
                            "pairwise_query_js": _quantile_payload(pair_js),
                        }

                    mean_attention = {
                        modality: normal[f"hop_attention_{modality}"].detach().float().mean(dim=0)
                        for modality in ("text", "visual")
                    }
                    tv_attention_mae = float((mean_attention["text"] - mean_attention["visual"]).abs().mean().item())
                    tv_r_attn_delta = float(profiles["text"]["r_attn"].mean().item() - profiles["visual"]["r_attn"].mean().item())
                    # This file is small (four propagation orders); append seed/modality-specific cells.
                    order_count = int(mean_attention["text"].size(0))
                    matrix_rows = []
                    for modality in ("text", "visual"):
                        matrix = mean_attention[modality].detach().cpu().numpy()
                        for query_order in range(order_count):
                            matrix_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "modality": modality,
                                    "query_order": query_order,
                                    **{f"key_order_{key}": float(matrix[query_order, key]) for key in range(order_count)},
                                }
                            )
                    matrix_file_exists = mean_attention_csv.exists()
                    if matrix_file_exists:
                        with mean_attention_csv.open("a", newline="", encoding="utf-8") as matrix_handle:
                            writer = csv.DictWriter(matrix_handle, fieldnames=list(matrix_rows[0]))
                            writer.writerows(matrix_rows)
                    else:
                        _write_csv(mean_attention_csv, matrix_rows)

                    # Attention and interaction state diagnostics from the frozen best checkpoint.
                    order_payload = {}
                    for modality in ("text", "visual"):
                        embedding = getattr(model, f"hop_order_embedding_{modality}").detach().float()
                        embedding_norm = embedding.norm(dim=-1)
                        state_norm = torch.stack(
                            [state.detach().float().norm(dim=-1).mean() for state in normal[f"states_{modality}"]]
                        )
                        ratio = embedding_norm / state_norm.clamp_min(1e-12)
                        cosine = F.normalize(embedding, dim=-1) @ F.normalize(embedding, dim=-1).T
                        order_payload[modality] = {
                            "embedding_norm_per_order": embedding_norm.cpu().tolist(),
                            "mean_state_norm_per_order": state_norm.cpu().tolist(),
                            "embedding_to_state_norm_ratio_per_order": ratio.cpu().tolist(),
                            "pairwise_embedding_cosine": cosine.cpu().tolist(),
                        }
                        for order in range(embedding.size(0)):
                            order_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "modality": modality,
                                    "order": order,
                                    "embedding_norm": float(embedding_norm[order].item()),
                                    "mean_state_norm": float(state_norm[order].item()),
                                    "embedding_to_state_norm_ratio": float(ratio[order].item()),
                                }
                            )

                    gate_text = float(torch.tanh(model.theta_hop_gate_text).item())
                    gate_visual = float(torch.tanh(model.theta_hop_gate_visual).item())

                    changed = _run_order_embedding_off(model, x, edge_index)
                    stage12_diff = _stage12_diff(normal, changed)
                    if max(stage12_diff.values(), default=0.0) > STAGE12_ABS_TOLERANCE:
                        raise AssertionError(f"OrderEmbedding-Off changed Stage-I/II tensors: {stage12_diff}")
                    changed_logits = head(changed["z"])
                    node_intervention, intervention_dist = _node_intervention_arrays(
                        normal, changed, normal_logits, changed_logits
                    )
                    changed_metrics = _downstream_metrics(cfg, checkpoint, model, data, changed, device)
                    test_idx = torch.as_tensor(data.test_idx, dtype=torch.long).cpu().numpy()
                    test_flip = node_intervention["prediction_flip"][test_idx]
                    intervention = {
                        "dataset": dataset,
                        "seed": seed,
                        "intervention": "order_embedding_off",
                        "text_attention_mae": intervention_dist["text"]["attention_mae"]["mean"],
                        "visual_attention_mae": intervention_dist["visual"]["attention_mae"]["mean"],
                        "text_eta_mae": intervention_dist["text"]["eta_mae"]["mean"],
                        "visual_eta_mae": intervention_dist["visual"]["eta_mae"]["mean"],
                        "z_mae": intervention_dist["fused"]["z_mae"]["mean"],
                        "logit_mae": intervention_dist["fused"]["logit_mae"]["mean"],
                        "probability_kl_normal_to_off": intervention_dist["fused"]["probability_kl_normal_to_intervention"]["mean"],
                        "test_prediction_flip_rate": float(test_flip.mean()) if test_flip.size else math.nan,
                        "test_acc_delta_off_minus_normal": float(changed_metrics["test_acc"] - normal_metrics["test_acc"]),
                        "test_macro_f1_delta_off_minus_normal": float(changed_metrics["test_macro_f1"] - normal_metrics["test_macro_f1"]),
                        "stage12_max_abs_diff": max(stage12_diff.values(), default=0.0),
                    }
                    intervention_rows.append(intervention)

                    # Per-node CSV retains distribution shape for later analysis without storing activations.
                    _write_mechanism_nodes(node_writer, dataset, seed, normal, query, node_intervention)

                    modality_records = {}
                    for modality in ("text", "visual"):
                        prof = profiles[modality]
                        modality_record = {
                            "kl_p_uniform": _quantile_payload(prof["kl_uniform"]),
                            "normalized_attention_entropy": _quantile_payload(prof["normalized_entropy"]),
                            "max_key_mass": _quantile_payload(prof["max_key_mass"]),
                            "r_attn": _quantile_payload(prof["r_attn"]),
                            "mean_key_mass_by_order": prof["key_mass"].mean(dim=0).detach().cpu().tolist(),
                            "mean_attention_matrix": mean_attention[modality].detach().cpu().tolist(),
                            **query_payload[modality],
                        }
                        modality_records[modality] = modality_record
                        summary_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "modality": modality,
                                "kl_p_uniform_mean": modality_record["kl_p_uniform"]["mean"],
                                "kl_p_uniform_std": modality_record["kl_p_uniform"]["std"],
                                "normalized_entropy_mean": modality_record["normalized_attention_entropy"]["mean"],
                                "normalized_entropy_std": modality_record["normalized_attention_entropy"]["std"],
                                "max_key_mass_mean": modality_record["max_key_mass"]["mean"],
                                "max_key_mass_std": modality_record["max_key_mass"]["std"],
                                "query_row_mae_mean": modality_record["query_row_mae"]["mean"],
                                "query_row_mae_std": modality_record["query_row_mae"]["std"],
                                "pairwise_query_js_mean": modality_record["pairwise_query_js"]["mean"],
                                "pairwise_query_js_std": modality_record["pairwise_query_js"]["std"],
                                "r_attn_mean": modality_record["r_attn"]["mean"],
                                "r_attn_std": modality_record["r_attn"]["std"],
                                "r_attn_q25": modality_record["r_attn"]["p25"],
                                "r_attn_median": modality_record["r_attn"]["median"],
                                "r_attn_q75": modality_record["r_attn"]["p75"],
                                "text_visual_attention_matrix_mae": tv_attention_mae,
                                "text_minus_visual_r_attn_mean": tv_r_attn_delta,
                                "hop_gate_text": gate_text,
                                "hop_gate_visual": gate_visual,
                            }
                        )
                    modality_rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "text_visual_mean_attention_matrix_mae": tv_attention_mae,
                            "text_minus_visual_mean_r_attn": tv_r_attn_delta,
                            "text_mean_r_attn": modality_records["text"]["r_attn"]["mean"],
                            "visual_mean_r_attn": modality_records["visual"]["r_attn"]["mean"],
                            "text_minus_visual_mean_normalized_entropy": modality_records["text"]["normalized_attention_entropy"]["mean"] - modality_records["visual"]["normalized_attention_entropy"]["mean"],
                            "text_hop_gate": gate_text,
                            "visual_hop_gate": gate_visual,
                        }
                    )
                    run_payload = {
                        "dataset": dataset,
                        "variant": "V2",
                        "seed": seed,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": _sha256(checkpoint_path),
                        "num_nodes": int(x.size(0)),
                        "normal_test_metrics": normal_metrics,
                        "order_embedding_off_test_metrics": changed_metrics,
                        "modalities": modality_records,
                        "text_visual_mean_attention_matrix_mae": tv_attention_mae,
                        "text_minus_visual_mean_r_attn": tv_r_attn_delta,
                        "hop_gate_text": gate_text,
                        "hop_gate_visual": gate_visual,
                        "order_embedding_diagnostics": order_payload,
                        "order_embedding_off": intervention,
                        "stage12_max_abs_diff": stage12_diff,
                        "checkpoint_parameter_hash_unchanged": None,
                    }
                    run_payloads.append(run_payload)

                after_hash = _parameter_hash(model)
                if before_hash != after_hash:
                    raise AssertionError(f"V2 checkpoint parameters changed during analysis: {checkpoint_path}")
                run_payloads[-1]["checkpoint_parameter_hash_unchanged"] = True
                del (
                    model,
                    data,
                    checkpoint,
                    normal,
                    changed,
                    normal_logits,
                    changed_logits,
                    head,
                    x,
                    edge_index,
                    profiles,
                    query,
                    mean_attention,
                    embedding,
                    embedding_norm,
                    state_norm,
                    ratio,
                    cosine,
                )
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"MECHANISM {dataset} V2 seed={seed}", flush=True)

    return summary_rows, order_rows, modality_rows, intervention_rows, run_payloads


def _parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _summarize_interventions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    metrics = (
        "text_attention_mae",
        "visual_attention_mae",
        "text_eta_mae",
        "visual_eta_mae",
        "z_mae",
        "logit_mae",
        "probability_kl_normal_to_off",
        "test_prediction_flip_rate",
        "test_acc_delta_off_minus_normal",
        "test_macro_f1_delta_off_minus_normal",
    )
    for dataset in DATASETS:
        subset = [row for row in rows if row["dataset"] == dataset]
        record: dict[str, Any] = {"dataset": dataset, "seed_count": len(subset)}
        for metric in metrics:
            stat = _stats([float(row[metric]) for row in subset])
            record[f"{metric}_mean"] = stat["mean"]
            record[f"{metric}_std_population"] = stat["std_population"]
        summaries.append(record)
    return summaries


def _fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _report(
    path: Path,
    main_rows: list[dict[str, Any]],
    paired_summaries: list[dict[str, Any]],
    factorial_summaries: list[dict[str, Any]],
    factorial_across: list[dict[str, Any]],
    mechanism_rows: list[dict[str, Any]],
    order_rows: list[dict[str, Any]],
    modality_rows: list[dict[str, Any]],
    intervention_summary: list[dict[str, Any]],
    mechanism_runs: list[dict[str, Any]],
) -> None:
    by_main = {(row["dataset"], row["variant"]): row for row in main_rows}
    lines = [
        "# IAMOC Core Generalization Across Five NC Datasets",
        "",
        "## 1. Protocol",
        "",
        "Node classification was evaluated on Movies, Toys, Grocery, ele-fashion, and Reddit-S with seeds 42/43/44. Runs use the frozen full-graph NC protocol (`unified_full_graph_nc_v1`), 300 maximum epochs, AdamW, validation-accuracy checkpoint selection, and the existing data splits. Reported seed summaries use population standard deviation. No tuning or test-driven model changes were used.",
        "",
        "Existing Movies/Grocery V0/V1/V2 runs were reused only after checking the saved resolved configuration against the frozen formal MoPF and NC task configuration, the variant settings, seed, split path and pre-run split-file timestamp, protocol, checkpoint-selection rule, metrics, and checkpoint. Their source files remain in `outputs/iamoc_v1/`. New runs and all new artifacts are under `outputs/iamoc_nc_generalization/`; resolved split paths and current SHA-256 checksums are in `analysis/split_manifest.csv`.",
        "",
        "## 2. 2×2 Factorial Design",
        "",
        "| Variant | TCPR | Hop interaction | Definition |",
        "|---|---|---|---|",
        "| V00 | Off | Off | MoPF, `use_transport_residual=false` |",
        "| V0 | On | Off | Formal MoPF |",
        "| V1 | On | On | IAMOC, one interaction layer, `relation_conditioning=output` |",
        "| V2 | Off | On | IAMOC, one interaction layer, `relation_conditioning=none` |",
        "",
        "## 3. Five-Dataset Main Results",
        "",
        "Values are mean ± population SD across the three seeds; best epochs are listed in seed order 42/43/44.",
        "",
        "| Dataset | Variant | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Best epoch (42/43/44) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = by_main[(dataset, variant)]
            lines.append(
                f"| {dataset} | {variant} | {_fmt(row['val_acc_mean'], row['val_acc_std_population'])} | "
                f"{_fmt(row['val_macro_f1_mean'], row['val_macro_f1_std_population'])} | "
                f"{_fmt(row['test_acc_mean'], row['test_acc_std_population'])} | "
                f"{_fmt(row['test_macro_f1_mean'], row['test_macro_f1_std_population'])} | {row['best_epoch_seeds']} |"
            )
    lines.extend(
        [
            "",
            "## 4. Paired Seed Deltas",
            "",
            "Each paired value is the first variant minus the second at the same seed. The table shows test metrics; all four metrics and all seed-level values are in `outputs/iamoc_nc_generalization/analysis/paired_seed_deltas.csv`.",
            "",
            "| Dataset | Comparison | Test Acc Δ mean ± SD; positive seeds | Test Macro-F1 Δ mean ± SD; positive seeds |",
            "|---|---|---:|---:|",
        ]
    )
    paired_lookup = {(row["dataset"], row["comparison"], row["metric"]): row for row in paired_summaries}
    for dataset in DATASETS:
        for comparison in COMPARISONS:
            acc = paired_lookup[(dataset, comparison, "test_acc")]
            f1 = paired_lookup[(dataset, comparison, "test_macro_f1")]
            lines.append(
                f"| {dataset} | {comparison.replace('tcpR', 'TCPR')} ({acc['variant_delta']}) | "
                f"{_fmt(acc['mean'], acc['std_population'])}; {acc['positive_seed_count']}/3 | "
                f"{_fmt(f1['mean'], f1['std_population'])}; {f1['positive_seed_count']}/3 |"
            )
    lines.extend(
        [
            "",
            "The factorial effects below are calculated for each seed and metric: Hop main effect = ½[(V2−V00)+(V1−V0)], TCPR main effect = ½[(V0−V00)+(V1−V2)], and interaction = (V1−V0)−(V2−V00). These are descriptive decompositions; they are not statistical causal estimates.",
            "",
            "Each summary reports the mean and median over the 15 dataset-seed effects, the number of positive dataset means, and the number of positive paired seed effects. CSV files retain all four metrics.",
            "",
            "## 5. Hop Main Effect",
            "",
            "| Metric | Effect | Mean Δ | Median Δ | + datasets / 5 | + seeds / 15 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for effect in FACTORIAL_EFFECTS:
        if effect != FACTORIAL_EFFECTS[0]:
            section_number = 6 if effect == "tcpr_main_effect" else 7
            heading = "TCPR Main Effect" if section_number == 6 else "Interaction Effect"
            lines.extend(
                [
                    "",
                    f"## {section_number}. {heading}",
                    "",
                    "| Metric | Mean Δ | Median Δ | + datasets / 5 | + seeds / 15 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
        for metric in ("test_acc", "test_macro_f1"):
            row = next(item for item in factorial_across if item["metric"] == metric and item["effect"] == effect)
            lines.append(
                f"| {metric} | "
                + (f"{effect} | " if effect == FACTORIAL_EFFECTS[0] else "")
                + f"{row['mean']:.5f} | {row['median']:.5f} | "
                f"{row['positive_dataset_count']}/5 | {row['paired_positive_seed_count']}/15 |"
            )
    lines.extend(
        [
            "",
            "Per-dataset and per-seed decompositions for all four reported metrics are in `factorial_seed_effects.csv` and `factorial_summary.csv`; cross-dataset counts are in `factorial_across_datasets.csv`.",
            "",
            "## 8. Mechanism Generalization",
            "",
            "All 15 V2 best checkpoints were analyzed by inference only. KL is KL(attended order mass || uniform); normalized entropy divides entropy by log(number of orders). Query-row MAE and pairwise query JS compare query-order distributions within each node. Entries below average per-run node means, node standard deviations, and node quartiles across the three seeds.",
            "",
            "| Dataset | Modality | KL(p‖uniform) | Normalized entropy | Max-key mass | Query-row MAE | Pairwise query JS | r_attn mean ± node SD | r_attn Q25 / median / Q75 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            rows = [row for row in mechanism_rows if row["dataset"] == dataset and row["modality"] == modality]
            mean = lambda key: float(np.mean([float(row[key]) for row in rows]))
            lines.append(
                f"| {dataset} | {modality} | {mean('kl_p_uniform_mean'):.4f} | {mean('normalized_entropy_mean'):.4f} | "
                f"{mean('max_key_mass_mean'):.4f} | {mean('query_row_mae_mean'):.4f} | {mean('pairwise_query_js_mean'):.4f} | "
                f"{mean('r_attn_mean'):.4f} ± {mean('r_attn_std'):.4f} | "
                f"{mean('r_attn_q25'):.4f} / {mean('r_attn_median'):.4f} / {mean('r_attn_q75'):.4f} |"
            )
    lines.extend(
        [
            "",
            "Order-embedding norm divided by mean propagation-state norm is averaged across seeds for each order:",
            "",
            "| Dataset | Modality | Order 0 | Order 1 | Order 2 | Order 3 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            ratios = []
            for order in range(4):
                rows = [row for row in order_rows if row["dataset"] == dataset and row["modality"] == modality and row["order"] == order]
                ratios.append(float(np.mean([row["embedding_to_state_norm_ratio"] for row in rows])))
            lines.append(f"| {dataset} | {modality} | " + " | ".join(f"{ratio:.4f}" for ratio in ratios) + " |")
    lines.extend(
        [
            "",
            "## 9. Modality Heterogeneity",
            "",
            "| Dataset | Mean-attention matrix MAE | Text − Visual mean r_attn | Text hop gate | Visual hop gate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        rows = [row for row in modality_rows if row["dataset"] == dataset]
        lines.append(
            f"| {dataset} | {np.mean([row['text_visual_mean_attention_matrix_mae'] for row in rows]):.5f} | "
            f"{np.mean([row['text_minus_visual_mean_r_attn'] for row in rows]):+.5f} | "
            f"{np.mean([row['text_hop_gate'] for row in rows]):.4f} | {np.mean([row['visual_hop_gate'] for row in rows]):.4f} |"
        )
    lines.extend(
        [
            "",
            "Order-embedding norms, mean state norms, per-order ratios, and pairwise embedding cosine matrices are in `mechanism_order_embeddings.csv` and `results.json`. Per-node values are retained in compressed `mechanism_node_metrics.csv.gz`; average attention matrices are in `mean_attention_matrices.csv`.",
            "",
            "## 10. OrderEmbedding-Off",
            "",
            f"This inference-only intervention removes the learned order embedding from V2 attention inputs by a temporary forward hook. Metrics are off minus normal for accuracy/F1; probability KL is KL(p_normal‖p_off). Repeated Stage-I/II outputs were checked at max absolute tolerance {STAGE12_ABS_TOLERANCE:.0e}; the largest observed difference was {max(max(run['stage12_max_abs_diff'].values()) for run in mechanism_runs):.3e}, consistent with float32 sparse propagation roundoff. No checkpoint state is written or changed.",
            "",
            "| Dataset | Attention MAE text / visual | Eta MAE text / visual | Z MAE | Logit MAE | Probability KL | Test flip rate | Test Acc Δ | Test Macro-F1 Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in intervention_summary:
        lines.append(
            f"| {row['dataset']} | {row['text_attention_mae_mean']:.3e} / {row['visual_attention_mae_mean']:.3e} | "
            f"{row['text_eta_mae_mean']:.3e} / {row['visual_eta_mae_mean']:.3e} | {row['z_mae_mean']:.3e} | "
            f"{row['logit_mae_mean']:.3e} | {row['probability_kl_normal_to_off_mean']:.3e} | "
            f"{row['test_prediction_flip_rate_mean']:.4f} | {row['test_acc_delta_off_minus_normal_mean']:+.5f} | "
            f"{row['test_macro_f1_delta_off_minus_normal_mean']:+.5f} |"
        )

    # Compact evidence-based interpretation, based on paired test deltas.
    paired = {(row["dataset"], row["comparison"], row["metric"]): row for row in paired_summaries}
    effects = {(row["metric"], row["effect"]): row for row in factorial_across}
    hop_no = [paired[(dataset, "A_hop_without_tcpR", "test_acc")] for dataset in DATASETS]
    hop_yes = [paired[(dataset, "B_hop_with_tcpR", "test_acc")] for dataset in DATASETS]
    tcp_no = [paired[(dataset, "C_tcpR_without_hop", "test_acc")] for dataset in DATASETS]
    tcp_yes = [paired[(dataset, "D_tcpR_with_hop", "test_acc")] for dataset in DATASETS]
    nonuniform_count = sum(
        float(row["kl_p_uniform_mean"]) > 0 and float(row["normalized_entropy_mean"]) < 1
        for row in mechanism_rows
        if row["modality"] in {"text", "visual"}
    )
    query_positive_count = sum(float(row["pairwise_query_js_mean"]) > 0 for row in mechanism_rows)
    heterogeneity_count = sum(float(row["text_visual_mean_attention_matrix_mae"]) > 0 for row in modality_rows)
    intervention_changes = sum(
        float(row["z_mae_mean"]) > 1e-8 or float(row["logit_mae_mean"]) > 1e-8
        for row in intervention_summary
    )
    lines.extend(
        [
            "",
            "## 11. Interim Scientific Interpretation",
            "",
            f"**A. IAMOC core across datasets.** Test-accuracy mean deltas for Hop Interaction were positive in {sum(row['mean'] > 0 for row in hop_no)}/5 datasets without TCPR and {sum(row['mean'] > 0 for row in hop_yes)}/5 with TCPR. The corresponding test Macro-F1 positive-dataset counts are {sum(paired[(d, 'A_hop_without_tcpR', 'test_macro_f1')]['mean'] > 0 for d in DATASETS)}/5 and {sum(paired[(d, 'B_hop_with_tcpR', 'test_macro_f1')]['mean'] > 0 for d in DATASETS)}/5. This is a broad but non-universal mean gain, with Toys negative on both accuracy comparisons; seed-level variation is shown above.",
            "",
            f"**B. TCPR after Hop Interaction.** For V1−V2, TCPR had positive mean test-accuracy deltas in {sum(row['mean'] > 0 for row in tcp_yes)}/5 datasets and positive Macro-F1 deltas in {sum(paired[(d, 'D_tcpR_with_hop', 'test_macro_f1')]['mean'] > 0 for d in DATASETS)}/5; the pooled 15-pair means were {np.mean([row['mean'] for row in tcp_yes]):+.5f} for accuracy and {np.mean([paired[(d, 'D_tcpR_with_hop', 'test_macro_f1')]['mean'] for d in DATASETS]):+.5f} for Macro-F1. Without Hop Interaction (V0−V00), the corresponding pooled means were {np.mean([row['mean'] for row in tcp_no]):+.5f} and {np.mean([paired[(d, 'C_tcpR_without_hop', 'test_macro_f1')]['mean'] for d in DATASETS]):+.5f}. Thus these results do not show a consistent added TCPR benefit once Hop Interaction is present. The factorial TCPR test-accuracy effect is {effects[('test_acc', 'tcpr_main_effect')]['mean']:+.5f}; interaction effect is {effects[('test_acc', 'interaction_effect')]['mean']:+.5f}.",
            "",
            f"**C. Mechanism transfer.** Non-uniform attended order mass was present by the descriptive KL/entropy criteria in {nonuniform_count}/{len(mechanism_rows)} dataset-seed-modality profiles; query-row MAE and pairwise query JS were nonzero in {query_positive_count}/{len(mechanism_rows)} profiles. Text and Visual mean attention matrices differed in {heterogeneity_count}/{len(modality_rows)} dataset-seed checkpoints, though the size of that gap varied by dataset. OrderEmbedding-Off changed Z or logits above 1e-8 in {intervention_changes}/5 dataset-level summaries; the detailed effect sizes and test metrics are in Section 10.",
            "",
            "The design supports comparisons within the chosen datasets and seeds. The factorial terms are descriptive decompositions, and no statistical causal claim is made. Dataset-level signs should be read together with paired seed deltas and population SDs rather than as universal model guarantees.",
            "",
            "## Machine-readable outputs",
            "",
            "`analysis/results.json` contains the complete grid, paired deltas, factorial effects, mechanism summaries, and OrderEmbedding-Off results. CSV files in the same directory provide main results, seed-level deltas, decompositions, order-embedding diagnostics, modality heterogeneity, and the compressed node table.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "iamoc_nc_generalization")
    parser.add_argument("--legacy-root", type=Path, default=ROOT / "outputs" / "iamoc_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "iamoc_nc_generalization.md")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    output_root = args.output_root.resolve()
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    runs, run_rows, main_rows = _main_grid(output_root, args.legacy_root.resolve())
    split_by_dataset_seed: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for row in run_rows:
        split_by_dataset_seed.setdefault((row["dataset"], row["seed"]), set()).add(
            (row["split_path"], row["split_sha256_current"])
        )
    for key, values in split_by_dataset_seed.items():
        if len(values) != 1:
            raise AssertionError(f"Model variants do not share one split for {key}: {values}")
    split_manifest = [
        {
            "dataset": dataset,
            "seed": seed,
            "split_path": next(iter(values))[0],
            "split_sha256_current": next(iter(values))[1],
        }
        for (dataset, seed), values in sorted(split_by_dataset_seed.items())
    ]
    paired_seed, paired_summary, factorial_seed, factorial_summary, factorial_across = _paired_and_factorial(runs)
    mechanism_rows, order_rows, modality_rows, intervention_rows, mechanism_runs = _mechanism_analysis(
        output_root, args.legacy_root.resolve(), device
    )
    intervention_summary = _summarize_interventions(intervention_rows)

    _write_csv(analysis_root / "run_metrics.csv", run_rows)
    _write_csv(analysis_root / "split_manifest.csv", split_manifest)
    _write_csv(analysis_root / "main_metrics_summary.csv", main_rows)
    _write_csv(analysis_root / "paired_seed_deltas.csv", paired_seed)
    _write_csv(analysis_root / "paired_summary.csv", paired_summary)
    _write_csv(analysis_root / "factorial_seed_effects.csv", factorial_seed)
    _write_csv(analysis_root / "factorial_summary.csv", factorial_summary)
    _write_csv(analysis_root / "factorial_across_datasets.csv", factorial_across)
    _write_csv(analysis_root / "mechanism_summary.csv", mechanism_rows)
    _write_csv(analysis_root / "mechanism_order_embeddings.csv", order_rows)
    _write_csv(analysis_root / "modality_heterogeneity.csv", modality_rows)
    _write_csv(analysis_root / "order_embedding_off.csv", intervention_rows)
    _write_csv(analysis_root / "order_embedding_off_summary.csv", intervention_summary)
    payload = {
        "protocol": {
            "datasets": list(DATASETS),
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "metrics": list(METRICS),
            "checkpoint_selection": "best_val_accuracy",
            "factorial_interpretation": "descriptive decomposition only",
            "order_embedding_off": "inference-only temporary hook; checkpoint unchanged",
            "stage12_repeat_forward_abs_tolerance": STAGE12_ABS_TOLERANCE,
        },
        "run_metrics": run_rows,
        "split_manifest": split_manifest,
        "main_metrics_summary": main_rows,
        "paired_seed_deltas": paired_seed,
        "paired_summary": paired_summary,
        "factorial_seed_effects": factorial_seed,
        "factorial_summary": factorial_summary,
        "factorial_across_datasets": factorial_across,
        "mechanism_summary": mechanism_rows,
        "mechanism_runs": mechanism_runs,
        "mechanism_order_embeddings": order_rows,
        "modality_heterogeneity": modality_rows,
        "order_embedding_off": intervention_rows,
        "order_embedding_off_summary": intervention_summary,
    }
    _write_json(analysis_root / "results.json", payload)
    _report(
        args.report,
        main_rows,
        paired_summary,
        factorial_summary,
        factorial_across,
        mechanism_rows,
        order_rows,
        modality_rows,
        intervention_summary,
        mechanism_runs,
    )
    print(f"Wrote report: {args.report}", flush=True)
    print(f"Wrote analysis artifacts: {analysis_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
