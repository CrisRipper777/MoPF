"""Run MoPF-vNext U1-R: identifiable semantic metrics and frozen audits.

U1-R deliberately leaves the NC runner and all LP code untouched. R0 reuses
the completed U1 S0 checkpoints after an equivalence audit; R1 and R2 are
trained as new full-graph NC jobs. All frozen interventions are analysis-only
and never write back to a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.mopf import MoPF  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402
from src.utils.summary import count_parameters, mean_std  # noqa: E402
from scripts.run_mopf_u1_semantic_conductance import (  # noqa: E402
    DATASETS,
    K_BY_DATASET,
    MAGB_DATASETS,
    SEEDS,
    _coalesced_sparse,
    _correlation,
    _distribution,
    _dump_json,
    _fixed_split_override,
    _peak_gpu_memory_mib,
)


VARIANTS = ("R0", "R1", "R2")
NEW_VARIANTS = ("R1", "R2")
MODE_BY_VARIANT = {
    "R0": "separate_cos",
    "R1": "learned_diag_cos",
    "R2": "multi_perspective_cos_broken",
}
U1_OUTPUT_ROOT = ROOT / "outputs" / "u1_semantic_conductance"


def _compose_cfg(dataset: str, seed: int, variant: str, device: str):
    k = K_BY_DATASET[dataset]
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        f"seed={seed}",
        f"device={device}",
        f"model.edge_weight_mode={MODE_BY_VARIANT[variant]}",
        f"model.max_order={k}",
        f"model.num_layers={k}",
        "model.num_metric_perspectives=4",
        "model.metric_init_seed=20260910",
        "model.metric_init_noise_std=0.01",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _run_formal_job(
    *,
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    k = K_BY_DATASET[dataset]
    run_dir = output_root / "runs" / dataset / variant / f"seed{seed}"
    checkpoint_path = run_dir / "best.pt"
    record_path = run_dir / "run_record.json"
    if resume and record_path.is_file() and checkpoint_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        "num_runs=1",
        f"seed={seed}",
        f"device={device}",
        f"model.edge_weight_mode={MODE_BY_VARIANT[variant]}",
        "model.num_metric_perspectives=4",
        "model.metric_init_seed=20260910",
        "model.metric_init_noise_std=0.01",
        f"model.max_order={k}",
        f"model.num_layers={k}",
        "task.training_mode=full_graph",
        "task.optimizer=adamw",
        "task.epochs=300",
        "task.lr=1e-3",
        "task.weight_decay=1e-4",
        "task.patience=30",
        "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4",
        "task.eval_every=1",
        "task.grad_clip=1.0",
        "task.evaluate_test=true",
        "model.export_aux_stats=false",
        "model.export_node_aux=false",
        "model.export_edge_aux=false",
        f"task.save_ckpt_path={checkpoint_path}",
        f"hydra.run.dir={run_dir}",
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        command.append(f"dataset.nc_split_path={split_override}")

    process_log = (run_dir / "process.log").open("w", encoding="utf-8")
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=process_log,
            stderr=subprocess.STDOUT,
        )
        peak_memory = None
        while process.poll() is None:
            current = _peak_gpu_memory_mib(process.pid, device)
            if current is not None:
                peak_memory = current if peak_memory is None else max(peak_memory, current)
            time.sleep(2.0)
        current = _peak_gpu_memory_mib(process.pid, device)
        if current is not None:
            peak_memory = current if peak_memory is None else max(peak_memory, current)
    finally:
        process_log.close()
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"U1-R job failed ({dataset}, {variant}, seed={seed}) exit={process.returncode}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = dict(checkpoint.get("metrics", {}))
    cfg = _compose_cfg(dataset, seed, variant, device)
    model = build_model(cfg, checkpoint["data_info"])
    parameter_count = count_parameters(model) + int(model.out_dim + 1) * int(
        checkpoint["data_info"]["num_classes"]
    )
    record = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "k": k,
        "device": device,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "downstream": {
            key: metrics.get(key)
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
        },
        "best_epoch": checkpoint.get("epoch"),
        "parameter_count": parameter_count,
        "training_time_seconds": elapsed,
        "peak_gpu_memory_mib": peak_memory,
        "peak_gpu_memory_status": "available" if peak_memory is not None else "unavailable",
    }
    _dump_json(record_path, record)
    return record


def _r0_source_records() -> list[dict[str, Any]]:
    summary_path = U1_OUTPUT_ROOT / "u1_master_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Required U1 summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for row in summary.get("per_run", []):
        if row.get("variant") != "S0":
            continue
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required U1 S0 checkpoint is missing: {checkpoint}")
        rows.append(
            {
                **row,
                "variant": "R0",
                "source_variant": "U1-S0",
                "source_summary": str(summary_path),
            }
        )
    if len(rows) != len(DATASETS) * len(SEEDS):
        raise RuntimeError(f"Expected 15 reusable U1 S0 records, got {len(rows)}")
    return rows


def _r0_code_equivalence_audit() -> dict[str, Any]:
    """Compare current R0 output with the exact frozen-tag implementation."""
    data_info = {
        "input_dim": 10,
        "num_nodes": 6,
        "num_classes": 3,
        "text_dim": 4,
        "visual_dim": 6,
    }
    model_cfg = {
        "name": "mopf",
        "version": "mopf",
        "hidden_dim": 8,
        "dropout": 0.0,
        "norm": "layernorm",
        "max_order": 3,
        "num_layers": 3,
        "map_prior_restart": 0.15,
        "map_prior_order": 2,
        "diffusion_add_self_loops": True,
        "edge_weight_mode": "separate_cos",
        "edge_weight_min": 0.1,
        "edge_weight_temperature": 2.0,
        "filter_rank": 4,
        "global_filter_trainable": True,
        "use_modality_residual": True,
        "use_node_residual": True,
        "hrc_weight": 0.0,
        "ppc_weight": 0.0,
        "fusion_mode": "concat_residual_mlp",
        "node_conditioner_mode": "absolute",
    }
    cfg = OmegaConf.create({"model": model_cfg})
    set_seed(20260910)
    current = MoPF(cfg, data_info).eval()
    x = torch.arange(60, dtype=torch.float32).view(6, 10) / 10.0
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0]],
        dtype=torch.long,
    )
    with torch.no_grad():
        current_z = current(x, edge_index)[0]
    payload = {
        "config": {"model": model_cfg},
        "data_info": data_info,
        "state": {key: value.detach().cpu() for key, value in current.state_dict().items()},
        "x": x,
        "edge_index": edge_index,
    }
    frozen_code = (
        "import sys, torch\n"
        "from omegaconf import OmegaConf\n"
        "from src.models.mopf import MoPF\n"
        "payload=torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
        "model=MoPF(OmegaConf.create(payload['config']), payload['data_info']).eval()\n"
        "model.load_state_dict(payload['state'])\n"
        "with torch.no_grad(): z=model(payload['x'], payload['edge_index'])[0]\n"
        "torch.save(z, sys.argv[2])\n"
    )
    with tempfile.TemporaryDirectory(prefix="mopf_u1r_r0_") as temp_dir:
        temp = Path(temp_dir)
        archive = subprocess.run(
            ["git", "archive", "mopf-v0-frozen"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        with tarfile.open(fileobj=__import__("io").BytesIO(archive), mode="r:") as tar:
            tar.extractall(temp)
        payload_path = temp / "audit_payload.pt"
        output_path = temp / "frozen_output.pt"
        torch.save(payload, payload_path)
        subprocess.run(
            [sys.executable, "-c", frozen_code, str(payload_path), str(output_path)],
            cwd=temp,
            env={**os.environ, "PYTHONPATH": str(temp)},
            check=True,
        )
        frozen_z = torch.load(output_path, map_location="cpu", weights_only=False)
    diff = (current_z.cpu() - frozen_z).abs()
    current_state_keys = set(current.state_dict())
    return {
        "passed": bool(torch.equal(current_z.cpu(), frozen_z)),
        "max_abs_difference": float(diff.max().item()),
        "current_state_has_no_metric_parameters": not any(
            "metric_theta" in key for key in current_state_keys
        ),
        "frozen_reference": "mopf-v0-frozen",
        "audit_input": "deterministic six-node graph with 4+6 modality dimensions",
    }


def _rank_summary(values: np.ndarray) -> dict[str, float]:
    return _distribution(values)


def _metric_shape(weights: torch.Tensor) -> dict[str, Any]:
    values = weights.detach().cpu().numpy().astype(np.float64)
    if values.ndim == 1:
        values = values[None, :]
    identity = np.ones(values.shape[1], dtype=np.float64)
    mean_by_dim = values.mean(axis=0)
    std_by_dim = values.std(axis=0, ddof=0)
    cv_by_dim = std_by_dim / np.maximum(np.abs(mean_by_dim), 1e-12)
    anisotropy = np.sqrt(np.mean(np.square(values - 1.0), axis=1))
    max_min = values.max(axis=1) / np.maximum(values.min(axis=1), 1e-12)
    cosine_identity = np.asarray(
        [np.dot(row, identity) / (np.linalg.norm(row) * np.linalg.norm(identity)) for row in values]
    )
    return {
        "perspective_count": int(values.shape[0]),
        "anisotropy_rms": anisotropy.tolist(),
        "anisotropy_rms_mean": float(anisotropy.mean()),
        "std_across_dimensions": std_by_dim.tolist(),
        "std_across_dimensions_mean": float(std_by_dim.mean()),
        "coefficient_of_variation_across_dimensions": cv_by_dim.tolist(),
        "coefficient_of_variation_mean": float(cv_by_dim.mean()),
        "max_to_min_ratio": max_min.tolist(),
        "cosine_to_identity_ones": cosine_identity.tolist(),
        "q10": np.quantile(values, 0.10, axis=1).tolist(),
        "q25": np.quantile(values, 0.25, axis=1).tolist(),
        "q50": np.quantile(values, 0.50, axis=1).tolist(),
        "q75": np.quantile(values, 0.75, axis=1).tolist(),
        "q90": np.quantile(values, 0.90, axis=1).tolist(),
        "normalized_weight_mean_by_dimension": mean_by_dim.tolist(),
    }


def _pairwise_matrix(values: np.ndarray, kind: str) -> np.ndarray:
    count = values.shape[0]
    matrix = np.ones((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(count):
            if kind == "cosine":
                matrix[left, right] = float(
                    np.dot(values[left], values[right])
                    / (np.linalg.norm(values[left]) * np.linalg.norm(values[right]) + 1e-12)
                )
            elif kind == "pearson":
                matrix[left, right] = _correlation(values[left], values[right])
            elif kind == "rms":
                matrix[left, right] = float(np.sqrt(np.mean(np.square(values[left] - values[right]))))
    return matrix


def _specialization(model: MoPF, components: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality, score_key in (("text", "perspective_cos_t"), ("visual", "perspective_cos_v")):
        weights = model.semantic_metric_weights(modality).cpu().numpy().astype(np.float64)
        scores = components["edges"][score_key].detach().cpu().numpy().astype(np.float64)
        aggregate = scores.mean(axis=0)
        score_corr = np.corrcoef(scores) if scores.shape[0] > 1 else np.ones((1, 1))
        score_corr = np.nan_to_num(score_corr, nan=0.0)
        mad = np.abs(scores - aggregate).mean(axis=1)
        result[modality] = {
            "weights": _metric_shape(torch.from_numpy(weights)),
            "weight_pairwise_cosine_matrix": _pairwise_matrix(weights, "cosine").tolist(),
            "weight_pairwise_pearson_matrix": _pairwise_matrix(weights, "pearson").tolist(),
            "weight_pairwise_normalized_rms_distance": _pairwise_matrix(weights, "rms").tolist(),
            "score_pairwise_correlation_matrix": score_corr.tolist(),
            "score_mean_abs_deviation_per_perspective": mad.tolist(),
            "score_specialization_mad": float(mad.mean()),
            "weight_shape_rms_spread": float(_pairwise_matrix(weights, "rms")[np.triu_indices(weights.shape[0], 1)].mean())
            if weights.shape[0] > 1
            else 0.0,
        }
    return result


def _specialization_evolution(initial: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for modality in ("text", "visual"):
        initial_weight = initial[modality]["weight_shape_rms_spread"]
        final_weight = final[modality]["weight_shape_rms_spread"]
        initial_score = initial[modality]["score_specialization_mad"]
        final_score = final[modality]["score_specialization_mad"]
        output[modality] = {
            "initial_weight_shape_rms_spread": initial_weight,
            "final_weight_shape_rms_spread": final_weight,
            "final_over_initial_weight_shape_rms_spread": final_weight / max(initial_weight, 1e-12),
            "initial_score_specialization_mad": initial_score,
            "final_score_specialization_mad": final_score,
            "final_over_initial_score_specialization_mad": final_score / max(initial_score, 1e-12),
        }
    return output


def _metric_shape_comparison(model: MoPF) -> dict[str, Any]:
    text = model.semantic_metric_weights("text").cpu().numpy().astype(np.float64).mean(axis=0)
    visual = model.semantic_metric_weights("visual").cpu().numpy().astype(np.float64).mean(axis=0)
    return {
        "text_visual_metric_shape_cosine": float(np.dot(text, visual) / (np.linalg.norm(text) * np.linalg.norm(visual) + 1e-12)),
        "text_visual_metric_shape_rms_difference": float(np.sqrt(np.mean(np.square(text - visual)))),
    }


def _operator_relative_change(
    on_index: torch.Tensor,
    on_weight: torch.Tensor,
    alt_index: torch.Tensor,
    alt_weight: torch.Tensor,
    num_nodes: int,
) -> float:
    on_keys, on_values = _coalesced_sparse(on_index, on_weight, num_nodes)
    alt_keys, alt_values = _coalesced_sparse(alt_index, alt_weight, num_nodes)
    keys = np.union1d(on_keys, alt_keys)
    on_pos = np.searchsorted(on_keys, keys)
    alt_pos = np.searchsorted(alt_keys, keys)
    on_present = (on_pos < on_keys.size) & (on_keys[np.minimum(on_pos, max(on_keys.size - 1, 0))] == keys) if on_keys.size else np.zeros(keys.size, dtype=bool)
    alt_present = (alt_pos < alt_keys.size) & (alt_keys[np.minimum(alt_pos, max(alt_keys.size - 1, 0))] == keys) if alt_keys.size else np.zeros(keys.size, dtype=bool)
    on_full = np.zeros(keys.size, dtype=np.float64)
    alt_full = np.zeros(keys.size, dtype=np.float64)
    on_full[on_present] = on_values[on_pos[on_present]]
    alt_full[alt_present] = alt_values[alt_pos[alt_present]]
    return float(np.linalg.norm(on_full - alt_full) / (np.linalg.norm(on_full) + 1e-12))


def _relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).norm().item() / (first.norm().item() + 1e-12))


def _mean_cosine_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    first_flat = first.reshape(first.size(0), -1)
    second_flat = second.reshape(second.size(0), -1)
    return float((1.0 - F.cosine_similarity(first_flat, second_flat, dim=-1, eps=1e-8)).mean().item())


def _edge_score_change(on: torch.Tensor, alt: torch.Tensor) -> dict[str, Any]:
    on_np = on.detach().cpu().numpy().astype(np.float64)
    alt_np = alt.detach().cpu().numpy().astype(np.float64)
    delta = alt_np - on_np
    return {
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "pearson": _correlation(on_np, alt_np),
        "spearman": _correlation(on_np, alt_np, spearman=True),
    }


def _conductance_change(on: torch.Tensor, alt: torch.Tensor) -> dict[str, Any]:
    on_np = on.detach().cpu().numpy().astype(np.float64)
    alt_np = alt.detach().cpu().numpy().astype(np.float64)
    delta = alt_np - on_np
    on_q = _distribution(on_np)
    alt_q = _distribution(alt_np)
    return {
        "mae": float(np.abs(delta).mean()),
        "relative_l2": float(np.linalg.norm(delta) / (np.linalg.norm(on_np) + 1e-12)),
        "pearson": _correlation(on_np, alt_np),
        "spearman": _correlation(on_np, alt_np, spearman=True),
        "on_distribution": on_q,
        "alternative_distribution": alt_q,
        "quantile_delta": {
            key: alt_q[key] - on_q[key]
            for key in ("q10", "q25", "q50", "q75", "q90")
        },
    }


def _split_metrics(logits: torch.Tensor, data, eval_labels: list[int]) -> dict[str, dict[str, float]]:
    output = {}
    prediction = logits.argmax(dim=-1).cpu()
    labels = data.y.cpu()
    for name, index in (("val", data.val_idx), ("test", data.test_idx)):
        target = labels[index.cpu()]
        pred = prediction[index.cpu()]
        output[name] = {
            "acc": float((pred == target).float().mean().item()),
            "macro_f1": float(
                f1_score(
                    target.numpy(),
                    pred.numpy(),
                    labels=list(eval_labels),
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    return output


def _prediction_flip_rate(on_pred: torch.Tensor, alt_pred: torch.Tensor, index: torch.Tensor) -> float:
    return float((on_pred[index.cpu()] != alt_pred[index.cpu()]).float().mean().item())


def _causal_intervention(
    *,
    model: MoPF,
    classifier: torch.nn.Module,
    data,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    on: dict[str, Any],
    alternative: dict[str, Any],
    eval_labels: list[int],
) -> dict[str, Any]:
    on_z = on["z"]
    alt_z = alternative["z"]
    on_logits = classifier(on_z)
    alt_logits = classifier(alt_z)
    on_prob = torch.softmax(on_logits, dim=-1)
    alt_prob = torch.softmax(alt_logits, dim=-1)
    on_pred = on_logits.argmax(dim=-1)
    alt_pred = alt_logits.argmax(dim=-1)
    metrics_on = _split_metrics(on_logits, data, eval_labels)
    metrics_alt = _split_metrics(alt_logits, data, eval_labels)
    probability_mean = 0.5 * (on_prob + alt_prob)
    js = 0.5 * (
        (on_prob * (on_prob.clamp_min(1e-12).log() - probability_mean.clamp_min(1e-12).log())).sum(dim=-1)
        + (alt_prob * (alt_prob.clamp_min(1e-12).log() - probability_mean.clamp_min(1e-12).log())).sum(dim=-1)
    ).mean()
    modality_changes = {}
    for modality, score_key, weight_key, index_key, weight_key_norm, bases_key, z_key in (
        ("text", "cos_t", "w_t", "norm_t_index", "norm_t_weight", "bases_text", "z_text"),
        ("visual", "cos_v", "w_v", "norm_v_index", "norm_v_weight", "bases_visual", "z_visual"),
    ):
        on_edges = on["edges"]
        alt_edges = alternative["edges"]
        on_bases = on[bases_key]
        alt_bases = alternative[bases_key]
        modality_changes[modality] = {
            "raw_edge_semantic_score": _edge_score_change(
                on_edges[score_key], alt_edges[score_key]
            ),
            "conductance": _conductance_change(
                on_edges[weight_key], alt_edges[weight_key]
            ),
            "normalized_operator_relative_l2_on_denominator": _operator_relative_change(
                on[index_key],
                on[weight_key_norm],
                alternative[index_key],
                alternative[weight_key_norm],
                int(data.num_nodes),
            ),
            "propagation_response_bank": [
                {
                    "order": order,
                    "relative_l2": _relative_l2(on_base, alt_base),
                    "mean_cosine_distance": _mean_cosine_distance(on_base, alt_base),
                }
                for order, (on_base, alt_base) in enumerate(zip(on_bases, alt_bases, strict=True))
            ],
            "filtered_modality_representation": {
                "relative_l2": _relative_l2(on[z_key], alternative[z_key]),
                "mean_cosine_distance": _mean_cosine_distance(on[z_key], alternative[z_key]),
            },
        }
    prob_delta = torch.abs(on_prob - alt_prob)
    return {
        "modality": modality_changes,
        "fused_representation": {
            "relative_l2": _relative_l2(on_z, alt_z),
            "mean_cosine_distance": _mean_cosine_distance(on_z, alt_z),
        },
        "logits": {
            "relative_l2": _relative_l2(on_logits, alt_logits),
            "mean_abs_change": float(torch.abs(on_logits - alt_logits).mean().item()),
        },
        "probabilities": {
            "mean_abs_change": float(prob_delta.mean().item()),
            "mean_js_divergence": float(js.item()),
        },
        "predictions": {
            "all_flip_rate": float((on_pred != alt_pred).float().mean().item()),
            "val_flip_rate": _prediction_flip_rate(on_pred, alt_pred, data.val_idx),
            "test_flip_rate": _prediction_flip_rate(on_pred, alt_pred, data.test_idx),
        },
        "downstream": {
            "metric_on": metrics_on,
            "metric_alternative": metrics_alt,
            "alternative_minus_on": {
                f"{split}_{metric}": metrics_alt[split][metric] - metrics_on[split][metric]
                for split in ("val", "test")
                for metric in ("acc", "macro_f1")
            },
        },
    }


def _entropy_summaries(src: np.ndarray, conductance: np.ndarray, num_nodes: int) -> dict[str, Any]:
    degree = np.bincount(src, weights=conductance, minlength=num_nodes)
    probability = conductance / np.maximum(degree[src], np.finfo(np.float64).tiny)
    entropy = np.bincount(
        src,
        weights=-probability * np.log(np.maximum(probability, np.finfo(np.float64).tiny)),
        minlength=num_nodes,
    )
    nonisolated = degree > 0.0
    select = lambda values: {key: values[key] for key in ("mean", "std", "q25", "q50", "q75")}
    return {
        "entropy_all_nodes": select(_distribution(entropy)),
        "entropy_nonisolated_nodes": select(_distribution(entropy[nonisolated])),
    }


def _conductance_diagnostics(components: dict[str, Any], data) -> dict[str, Any]:
    edges = components["edges"]
    src = edges["src"].detach().cpu().numpy().astype(np.int64) if "src" in edges else components["norm_t_index"][0].detach().cpu().numpy()
    if "src" not in edges:
        src = data.edge_index[0].cpu().numpy().astype(np.int64)
    output = {}
    for modality, weight_key in (("text", "w_t"), ("visual", "w_v")):
        values = edges[weight_key].detach().cpu().numpy().astype(np.float64)
        distribution = _distribution(values)
        distribution["cv"] = float(values.std(ddof=0) / max(abs(values.mean()), 1e-12))
        distribution["effective_dynamic_range_q90_minus_q10"] = distribution["q90"] - distribution["q10"]
        distribution["effective_dynamic_range_q90_over_q10"] = distribution["q90"] / max(distribution["q10"], 1e-12)
        output[modality] = {
            "conductance_distribution": distribution,
            "neighborhood_entropy": _entropy_summaries(src, values, int(data.num_nodes)),
        }
    text = edges["w_t"].detach().cpu().numpy().astype(np.float64)
    visual = edges["w_v"].detach().cpu().numpy().astype(np.float64)
    output["modality_conductance_gap"] = {
        "mean_abs": float(np.abs(text - visual).mean()),
        "median_abs": float(np.median(np.abs(text - visual))),
        "pearson": _correlation(text, visual),
        "spearman": _correlation(text, visual, spearman=True),
    }
    output["normalized_operator_difference"] = float(
        _operator_relative_change(
            edges["norm_t_index"],
            edges["norm_t_weight"],
            edges["norm_v_index"],
            edges["norm_v_weight"],
            int(data.num_nodes),
        )
    )
    output["pathology"] = {
        "all_finite": all(
            bool(torch.isfinite(value).all())
            for value in (edges["cos_t"], edges["cos_v"], edges["w_t"], edges["w_v"])
        ),
        "within_configured_bounds": bool(
            float(edges["w_t"].min().item()) >= float(model_edge_weight_min(components)) - 1e-6
            and float(edges["w_v"].min().item()) >= float(model_edge_weight_min(components)) - 1e-6
            and float(edges["w_t"].max().item()) <= 1.0 + 1e-6
            and float(edges["w_v"].max().item()) <= 1.0 + 1e-6
        ),
    }
    return output


def model_edge_weight_min(components: dict[str, Any]) -> float:
    # This value is attached by the checkpoint analysis caller; the helper is
    # kept separate so diagnostics never hard-code the formal floor.
    return float(components["_edge_weight_min"])


def _checkpoint_analysis(
    *,
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    cfg = _compose_cfg(dataset, seed, variant, device)
    data = load_mag_data(cfg, "nc", seed)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(checkpoint["head_state"])
    classifier.eval()
    eval_labels = _resolve_nc_eval_labels(data)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Recreate the exact model initialization used by the task runner for the
    # initial-vs-final R2 audit. The metric seed itself remains separate and
    # deterministic inside MoPF.
    set_seed(seed)
    initial_model = build_model(cfg, checkpoint["data_info"]).to(device).eval()
    with torch.no_grad():
        initial_components = initial_model._encode_components(x, edge_index)
        on = model._encode_components(x, edge_index)
    initial_specialization = _specialization(initial_model, initial_components)
    final_specialization = _specialization(model, on)
    mechanism = {
        "metric_shape": {
            "text": _metric_shape(model.semantic_metric_weights("text")),
            "visual": _metric_shape(model.semantic_metric_weights("visual")),
            **_metric_shape_comparison(model),
        },
        "specialization": {
            "initial": initial_specialization,
            "final": final_specialization,
            "evolution": _specialization_evolution(initial_specialization, final_specialization),
        },
    }
    if variant == "R2":
        learned_text = model.semantic_metric_weights("text")
        learned_visual = model.semantic_metric_weights("visual")
        mean_text = learned_text.mean(dim=0, keepdim=True).repeat(4, 1)
        mean_visual = learned_visual.mean(dim=0, keepdim=True).repeat(4, 1)
        interventions = {
            "F1_identity_metric": (torch.ones_like(learned_text), torch.ones_like(learned_visual)),
            "F2_text_identity": (torch.ones_like(learned_text), None),
            "F3_visual_identity": (None, torch.ones_like(learned_visual)),
            "F4_collapse_to_mean_metric": (mean_text, mean_visual),
        }
    else:
        learned_text = model.semantic_metric_weights("text")
        learned_visual = model.semantic_metric_weights("visual")
        interventions = {
            "F1_identity_metric": (torch.ones_like(learned_text), torch.ones_like(learned_visual)),
            "F2_text_identity": (torch.ones_like(learned_text), None),
            "F3_visual_identity": (None, torch.ones_like(learned_visual)),
        }
    intervention_results = {}
    for name, (text_weights, visual_weights) in interventions.items():
        with torch.no_grad():
            alternative = model.analysis_encode_with_metric_override(
                x,
                edge_index,
                text_weights=text_weights,
                visual_weights=visual_weights,
            )
        intervention_results[name] = _causal_intervention(
            model=model,
            classifier=classifier,
            data=data,
            x=x,
            edge_index=edge_index,
            on=on,
            alternative=alternative,
            eval_labels=eval_labels,
        )
    mechanism["frozen_interventions"] = intervention_results
    on_with_floor = {
        **on,
        "_edge_weight_min": float(model.edge_weight_min),
    }
    mechanism["conductance"] = _conductance_diagnostics(on_with_floor, data)
    # Store the metric-on downstream values from the same best checkpoint.
    with torch.no_grad():
        on_logits = classifier(on["z"])
    mechanism["metric_on_downstream"] = _split_metrics(on_logits, data, eval_labels)
    del initial_model, model, classifier, data, x, edge_index, on
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return mechanism


def _numeric(values: list[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "population_std": None}
    mean, std = mean_std(clean)
    return {"mean": mean, "population_std": std}


def _aggregate_downstream(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    keys = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    for dataset in DATASETS:
        output[dataset] = {}
        for variant in VARIANTS:
            selected = [row for row in records if row["dataset"] == dataset and row["variant"] == variant]
            output[dataset][variant] = {
                "seed_count": len(selected),
                "seeds": [row["seed"] for row in selected],
                "downstream": {key: _numeric([row["downstream"].get(key) for row in selected]) for key in keys},
                "best_epoch": _numeric([row.get("best_epoch") for row in selected]),
                "parameter_count": _numeric([row.get("parameter_count") for row in selected]),
                "training_time_seconds": _numeric([row.get("training_time_seconds") for row in selected]),
                "peak_gpu_memory_mib": _numeric([row.get("peak_gpu_memory_mib") for row in selected]),
            }
    return output


def _comparisons(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for left, right in (("R1", "R0"), ("R2", "R0"), ("R2", "R1")):
            rows = []
            for seed in SEEDS:
                first = next(row for row in records if row["dataset"] == dataset and row["variant"] == left and row["seed"] == seed)
                second = next(row for row in records if row["dataset"] == dataset and row["variant"] == right and row["seed"] == seed)
                rows.append(
                    {
                        "seed": seed,
                        **{
                            key: float(first["downstream"][key] - second["downstream"][key])
                            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
                        },
                    }
                )
            output[dataset][f"{left}_minus_{right}"] = {
                "per_seed": rows,
                "aggregate": {key: _numeric([row[key] for row in rows]) for key in rows[0] if key != "seed"},
            }
    return output


def _candidate_decision(summary: dict[str, Any]) -> dict[str, Any]:
    aggregates = summary["aggregates"]
    records = summary["per_run"]
    r1_same_band = {}
    r2_same_band = {}
    r1_nonidentity = {}
    r1_active = {}
    r2_specialized = {}
    r2_collapse_active = {}
    pathology_ok = True
    for dataset in DATASETS:
        r0_val = aggregates[dataset]["R0"]["downstream"]["val_acc"]["mean"]
        r1_val = aggregates[dataset]["R1"]["downstream"]["val_acc"]["mean"]
        r2_val = aggregates[dataset]["R2"]["downstream"]["val_acc"]["mean"]
        r1_same_band[dataset] = bool(r1_val >= r0_val - 0.01)
        r2_same_band[dataset] = bool(r2_val >= r0_val - 0.01)
        r1_rows = [row for row in records if row["dataset"] == dataset and row["variant"] == "R1"]
        r2_rows = [row for row in records if row["dataset"] == dataset and row["variant"] == "R2"]
        r1_nonidentity[dataset] = bool(
            all(
                max(
                    row["mechanism"]["metric_shape"]["text"]["anisotropy_rms_mean"],
                    row["mechanism"]["metric_shape"]["visual"]["anisotropy_rms_mean"],
                )
                > 1e-4
                for row in r1_rows
            )
        )
        r1_active[dataset] = bool(
            sum(
                row["mechanism"]["frozen_interventions"]["F1_identity_metric"]["fused_representation"]["relative_l2"]
                > 1e-5
                or row["mechanism"]["frozen_interventions"]["F1_identity_metric"]["logits"]["relative_l2"]
                > 1e-5
                for row in r1_rows
            )
            >= 2
        )
        r2_specialized[dataset] = bool(
            sum(
                row["mechanism"]["specialization"]["final"]["text"]["score_specialization_mad"] > 1e-5
                or row["mechanism"]["specialization"]["final"]["visual"]["score_specialization_mad"] > 1e-5
                for row in r2_rows
            )
            >= 2
        )
        r2_collapse_active[dataset] = bool(
            sum(
                row["mechanism"]["frozen_interventions"]["F4_collapse_to_mean_metric"]["fused_representation"]["relative_l2"] > 1e-5
                or row["mechanism"]["frozen_interventions"]["F4_collapse_to_mean_metric"]["logits"]["relative_l2"] > 1e-5
                for row in r2_rows
            )
            >= 2
        )
        for row in r1_rows + r2_rows:
            checks = row["mechanism"]["conductance"]["pathology"]
            pathology_ok = pathology_ok and checks["all_finite"] and checks["within_configured_bounds"]
    r1_keep = all(r1_same_band.values()) and all(r1_nonidentity.values()) and sum(r1_active.values()) >= 1 and pathology_ok
    r2_keep = (
        r1_keep
        and all(r2_same_band.values())
        and sum(r2_specialized.values()) >= 3
        and sum(r2_collapse_active.values()) >= 1
        and pathology_ok
    )
    if r2_keep:
        recommendation = "Select R2 — Multi-Perspective Semantic Conductance"
    elif r1_keep:
        recommendation = "Select R1 — Modality-Adaptive Semantic Conductance"
    elif pathology_ok and all(r1_same_band.values()) and not any(r1_active.values()):
        recommendation = "Select R0 — Separate Cosine"
    else:
        recommendation = "U1 relation-level design still unresolved"
    return {
        "recommendation": recommendation,
        "r1_keep": r1_keep,
        "r2_keep": r2_keep,
        "criteria": {
            "validation_same_band_tolerance": 0.01,
            "r1_same_band_by_dataset": r1_same_band,
            "r2_same_band_by_dataset": r2_same_band,
            "r1_normalized_metric_nonidentity_by_dataset": r1_nonidentity,
            "r1_frozen_identity_functional_change_by_dataset": r1_active,
            "r2_repeated_specialization_by_dataset": r2_specialized,
            "r2_collapse_mean_functional_change_by_dataset": r2_collapse_active,
            "pathology_ok": pathology_ok,
        },
        "rationale": (
            "This is a frozen functional screening decision over validation-side "
            "evidence and same-checkpoint interventions. It does not claim statistical "
            "significance. Between-model scalar differences are descriptive; causal "
            "mechanism claims use identity and collapse-to-mean interventions."
        ),
    }


def _write_master_table(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for row in summary["per_run"]:
        item = {
            "dataset": row["dataset"],
            "variant": row["variant"],
            "seed": row["seed"],
            "k": row["k"],
            **row["downstream"],
            "best_epoch": row.get("best_epoch"),
            "parameter_count": row.get("parameter_count"),
            "training_time_seconds": row.get("training_time_seconds"),
            "peak_gpu_memory_mib": row.get("peak_gpu_memory_mib"),
        }
        if "mechanism" in row:
            item.update(
                {
                    "text_anisotropy_rms": row["mechanism"].get("metric_shape", {}).get("text", {}).get("anisotropy_rms_mean"),
                    "visual_anisotropy_rms": row["mechanism"].get("metric_shape", {}).get("visual", {}).get("anisotropy_rms_mean"),
                    "text_score_specialization_mad": row["mechanism"].get("specialization", {}).get("final", {}).get("text", {}).get("score_specialization_mad"),
                    "visual_score_specialization_mad": row["mechanism"].get("specialization", {}).get("final", {}).get("visual", {}).get("score_specialization_mad"),
                    "conductance_gap_mean_abs": row["mechanism"].get("conductance", {}).get("modality_conductance_gap", {}).get("mean_abs"),
                    "operator_difference": row["mechanism"].get("conductance", {}).get("normalized_operator_difference"),
                }
            )
        rows.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["final_relation_level_recommendation"]
    lines = [
        "# MoPF-vNext U1-R — Semantic Metric Resolution and Frozen Functional Audit",
        "",
        "NC-only controlled upgrade. LP, sports-copurchase, decoder, sampler, and LP protocol were not run or modified.",
        "",
        f"- Frozen reference: `{summary['frozen_reference_commit']}`",
        f"- Analysis commit: `{summary['vnext_commit']}`",
        f"- R0 source: `{summary['r0_source_reference']['summary']}`",
        f"- Final relation-level recommendation: **{decision['recommendation']}**",
        "",
        "R1 and R2 use mean-normalized positive metric vectors, which remove the weighted-cosine global-scale gauge.",
        "R2 starts with deterministic, zero-mean per-dimension perturbations at RMS 0.01; no diversity or auxiliary loss is used.",
        "",
        "The JSON is the authoritative artifact. It separates descriptive R0/R1/R2 comparisons from same-checkpoint frozen interventions.",
        "",
        "## Decision rationale",
        "",
        decision["rationale"],
        "",
        "No final MoPF claim is made here; this freezes only the relation-level structuralization decision.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/u1r_semantic_metric_resolution")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run R0 equivalence and all-dataset initialization audits without training.",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    devices = [str(device) for device in args.devices]

    r0_records = _r0_source_records()
    r0_audit = _r0_code_equivalence_audit()
    if not r0_audit["passed"] or not r0_audit["current_state_has_no_metric_parameters"]:
        raise RuntimeError(f"R0 equivalence audit failed: {r0_audit}")
    _dump_json(args.output_root / "r0_equivalence_audit.json", r0_audit)
    print(f"[U1-R] R0 equivalence audit passed: max_abs={r0_audit['max_abs_difference']}", flush=True)

    initialization_audit: dict[str, Any] = {}
    for index, dataset in enumerate(DATASETS):
        device = devices[index % len(devices)]
        cfg_r0 = _compose_cfg(dataset, 42, "R0", device)
        cfg_r1 = _compose_cfg(dataset, 42, "R1", device)
        cfg_r2 = _compose_cfg(dataset, 42, "R2", device)
        set_seed(42)
        data = load_mag_data(cfg_r0, "nc", 42)
        info = _data_info(data)
        r0 = build_model(cfg_r0, info).to(device).eval()
        set_seed(42)
        r1 = build_model(cfg_r1, info).to(device).eval()
        r1.load_state_dict(r0.state_dict(), strict=False)
        set_seed(42)
        r2 = build_model(cfg_r2, info).to(device).eval()
        r2.load_state_dict(r0.state_dict(), strict=False)
        with torch.no_grad():
            r0_edges = r0._encode_components(data.x.to(device), data.edge_index.to(device))["edges"]
            r1_edges = r1._encode_components(data.x.to(device), data.edge_index.to(device))["edges"]
            r2_edges = r2._encode_components(data.x.to(device), data.edge_index.to(device))["edges"]
        item: dict[str, Any] = {"dataset": dataset, "seed": 42, "modalities": {}}
        passed = True
        for modality in ("t", "v"):
            ordinary = r0_edges[f"cos_{modality}"].detach().cpu().numpy().astype(np.float64)
            r1_score = r1_edges[f"perspective_cos_{modality}"].mean(dim=0).detach().cpu().numpy().astype(np.float64)
            r2_score = r2_edges[f"perspective_cos_{modality}"].mean(dim=0).detach().cpu().numpy().astype(np.float64)
            r1_item = {
                "pearson": _correlation(ordinary, r1_score),
                "spearman": _correlation(ordinary, r1_score, spearman=True),
                "mean_absolute_difference": float(np.abs(ordinary - r1_score).mean()),
                "passed": bool(
                    _correlation(ordinary, r1_score) > 0.9999
                    and _correlation(ordinary, r1_score, spearman=True) > 0.9999
                    and np.abs(ordinary - r1_score).mean() < 1e-5
                ),
            }
            r2_weights = r2.semantic_metric_weights("text" if modality == "t" else "visual")
            r2_weights_np = r2_weights.detach().cpu().numpy()
            r2_item = {
                "pearson": _correlation(ordinary, r2_score),
                "spearman": _correlation(ordinary, r2_score, spearman=True),
                "mean_absolute_difference": float(np.abs(ordinary - r2_score).mean()),
                "weight_pairwise_l2_min": float(np.linalg.norm(r2_weights_np[0] - r2_weights_np[1])),
                "weight_pairwise_l2_max": float(max(np.linalg.norm(r2_weights_np[a] - r2_weights_np[b]) for a in range(4) for b in range(a + 1, 4))),
                "score_pairwise_exactly_identical": bool(torch.equal(r2_edges[f"perspective_cos_{modality}"][0], r2_edges[f"perspective_cos_{modality}"][1])),
                "passed": bool(
                    _correlation(ordinary, r2_score) > 0.999
                    and _correlation(ordinary, r2_score, spearman=True) > 0.999
                    and np.linalg.norm(r2_weights_np[0] - r2_weights_np[1]) > 0.0
                    and not torch.equal(r2_edges[f"perspective_cos_{modality}"][0], r2_edges[f"perspective_cos_{modality}"][1])
                ),
            }
            item["modalities"]["text" if modality == "t" else "visual"] = {"R1": r1_item, "R2": r2_item}
            passed = passed and r1_item["passed"] and r2_item["passed"]
        item["passed"] = passed
        initialization_audit[dataset] = item
        print(f"[U1-R] initialization {dataset}: passed={passed}", flush=True)
        del r0, r1, r2, data
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    _dump_json(args.output_root / "initialization_audit.json", initialization_audit)
    if not all(item["passed"] for item in initialization_audit.values()):
        raise RuntimeError("U1-R initialization audit failed; formal training was not started")
    if args.audit_only:
        print("[U1-R] initialization audits passed; audit-only requested, formal training not started", flush=True)
        return

    records = list(r0_records)
    jobs = [(dataset, variant, seed) for variant in NEW_VARIANTS for dataset in DATASETS for seed in SEEDS]
    for job_index, (dataset, variant, seed) in enumerate(jobs, start=1):
        device = devices[(job_index - 1) % len(devices)]
        print(f"[U1-R] training {job_index}/{len(jobs)} {dataset} {variant} seed={seed} device={device}", flush=True)
        row = _run_formal_job(
            dataset=dataset,
            variant=variant,
            seed=seed,
            device=device,
            output_root=args.output_root,
            resume=not args.no_resume,
        )
        print(f"[U1-R] frozen audit {dataset} {variant} seed={seed}", flush=True)
        row["mechanism"] = _checkpoint_analysis(
            dataset=dataset,
            variant=variant,
            seed=seed,
            device=device,
            checkpoint_path=Path(row["checkpoint"]),
        )
        _dump_json(Path(row["run_dir"]) / "mechanism_diagnostics.json", row["mechanism"])
        records.append(row)

    records.sort(key=lambda row: (row["dataset"], VARIANTS.index(row["variant"]), row["seed"]))
    summary: dict[str, Any] = {
        "metadata": {
            "stage": "MoPF-vNext U1-R Semantic Metric Resolution and Frozen Functional Audit",
            "scope": "NC only",
            "no_lp_experiments": True,
            "no_sports_copurchase": True,
            "fetch_origin_status": "failed: temporary DNS resolution failure for github.com; local vnext used",
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_clean_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip() or "clean",
        "frozen_reference_commit": subprocess.check_output(["git", "rev-parse", "mopf-v0-frozen"], cwd=ROOT, text=True).strip(),
        "vnext_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "r0_source_reference": {
            "summary": str(U1_OUTPUT_ROOT / "u1_master_summary.json"),
            "variant": "U1-S0 reused as R0",
            "checkpoint_count": len(r0_records),
            "equivalence_audit": r0_audit,
        },
        "variants": {
            "R0": {"edge_weight_mode": "separate_cos", "source": "Frozen MoPF-v0 / U1-S0"},
            "R1": {"edge_weight_mode": "learned_diag_cos", "metric_parameterization": "softplus(theta)/mean(softplus(theta))"},
            "R2": {"edge_weight_mode": "multi_perspective_cos_broken", "num_metric_perspectives": 4, "metric_init_seed": 20260910, "metric_init_noise_std": 0.01, "metric_parameterization": "softplus(theta)/mean(softplus(theta))"},
        },
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "K": dict(K_BY_DATASET),
        "protocol": {
            "name": "unified_full_graph_nc_v1",
            "training_mode": "full_graph",
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 300,
            "patience": 30,
            "hidden_dim": 256,
            "dropout": 0.2,
            "checkpoint_selection": "best_validation_accuracy",
            "macro_f1": "fixed supervised train/val/test union labels, zero_division=0",
            "test_not_used_for_selection": True,
        },
        "initialization_audit": initialization_audit,
        "per_run": records,
    }
    summary["aggregates"] = _aggregate_downstream(records)
    summary["downstream_comparisons"] = _comparisons(records)
    summary["final_relation_level_recommendation"] = _candidate_decision(summary)
    summary["efficiency"] = {
        variant: {
            dataset: summary["aggregates"][dataset][variant]
            for dataset in DATASETS
        }
        for variant in VARIANTS
    }
    summary["artifacts"] = {
        "summary": str(args.output_root / "u1r_master_summary.json"),
        "master_table": str(args.output_root / "u1r_master_table.csv"),
        "report": str(ROOT / "docs/mopf_u1r_semantic_metric_resolution.md"),
        "initialization_audit": str(args.output_root / "initialization_audit.json"),
        "r0_equivalence_audit": str(args.output_root / "r0_equivalence_audit.json"),
    }
    _dump_json(args.output_root / "u1r_master_summary.json", summary)
    _write_master_table(args.output_root / "u1r_master_table.csv", summary)
    _write_report(ROOT / "docs/mopf_u1r_semantic_metric_resolution.md", summary)
    print(f"[U1-R] summary={args.output_root / 'u1r_master_summary.json'}", flush=True)
    print(f"[U1-R] recommendation={summary['final_relation_level_recommendation']['recommendation']}", flush=True)


if __name__ == "__main__":
    main()
