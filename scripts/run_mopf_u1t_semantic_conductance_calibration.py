"""Run MoPF-vNext U1-T semantic-conductance calibration.

Phase A is a frozen, inference-only temperature intervention over the 15 U1-R
R1 checkpoints.  Phase B is entered only when the validation/mechanism gate
accepts at least one temperature.  The temperature override is passed through
the model's analysis path and never assigned to model state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.mopf import MoPF  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402
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
from scripts.run_mopf_u1r_semantic_metric_resolution import (  # noqa: E402
    _causal_intervention,
    _split_metrics,
)


TEMPERATURE_GRID = (2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25)
TAU0 = 2.0
EPS = 1e-12
SCORE_INVARIANCE_TOLERANCE = 1e-7
LOW_SATURATION_THRESHOLD = 0.05
VAL_MEAN_DROP_TOLERANCE = 0.003
VAL_DATASET_DROP_TOLERANCE = 0.01
FINAL_VAL_MEAN_DROP_TOLERANCE = 0.002
FINAL_VAL_DATASET_DROP_TOLERANCE = 0.005
FUNCTIONAL_NUMERICAL_THRESHOLD = 1e-10
FUNCTIONAL_AMPLIFIED_THRESHOLD = 1e-4

U1R_ROOT = ROOT / "outputs" / "u1r_semantic_metric_resolution"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "u1t_semantic_conductance_calibration"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _compose_r1_cfg(dataset: str, seed: int, device: str, temperature: float = TAU0):
    k = K_BY_DATASET[dataset]
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        f"seed={seed}",
        f"device={device}",
        "model.edge_weight_mode=learned_diag_cos",
        f"model.edge_weight_temperature={temperature}",
        f"model.max_order={k}",
        f"model.num_layers={k}",
        "model.num_metric_perspectives=4",
        "model.metric_init_seed=20260910",
        "model.metric_init_noise_std=0.01",
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        overrides.append(f"dataset.nc_split_path={split_override}")
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


def _numeric(values: list[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"mean": None, "population_std": None, "median": None, "count": 0}
    mean, std = mean_std(clean)
    return {
        "mean": float(mean),
        "population_std": float(std),
        "median": float(np.median(clean)),
        "count": len(clean),
    }


def _ratio(value: float, denominator: float) -> float:
    if not math.isfinite(float(value)):
        return float("nan")
    if abs(float(denominator)) <= EPS:
        return 0.0 if abs(float(value)) <= EPS else float(abs(value) / EPS)
    return float(value / max(abs(float(denominator)), EPS))


def _full_distribution(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    array = values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    output = _distribution(array.astype(np.float64, copy=False))
    mean_value = max(abs(output["mean"]), EPS)
    output["cv"] = float(output["std"] / mean_value)
    output["dynamic_range_q90_minus_q10"] = float(output["q90"] - output["q10"])
    output["dynamic_range_q90_over_q10"] = float(output["q90"] / max(output["q10"], EPS))
    return output


def _state_metric_shape(weights: torch.Tensor) -> dict[str, Any]:
    values = weights.detach().cpu().numpy().astype(np.float64)
    if values.ndim == 1:
        values = values[None, :]
    mean_by_dimension = values.mean(axis=0)
    std_by_dimension = values.std(axis=0, ddof=0)
    cv_by_dimension = std_by_dimension / np.maximum(np.abs(mean_by_dimension), EPS)
    within_std = values.std(axis=1, ddof=0)
    return {
        "perspective_count": int(values.shape[0]),
        "anisotropy_rms": np.sqrt(np.mean(np.square(values - 1.0), axis=1)).tolist(),
        "anisotropy_rms_mean": float(np.sqrt(np.mean(np.square(values - 1.0), axis=1)).mean()),
        "std_across_perspectives_by_dimension": std_by_dimension.tolist(),
        "std_across_perspectives_by_dimension_mean": float(std_by_dimension.mean()),
        "cv_across_perspectives_by_dimension": cv_by_dimension.tolist(),
        "cv_across_perspectives_by_dimension_mean": float(cv_by_dimension.mean()),
        "within_metric_std_across_dimensions": within_std.tolist(),
        "within_metric_std_across_dimensions_mean": float(within_std.mean()),
        "max_to_min_ratio": (values.max(axis=1) / np.maximum(values.min(axis=1), EPS)).tolist(),
        "q10": np.quantile(values, 0.10, axis=1).tolist(),
        "q25": np.quantile(values, 0.25, axis=1).tolist(),
        "q50": np.quantile(values, 0.50, axis=1).tolist(),
        "q75": np.quantile(values, 0.75, axis=1).tolist(),
        "q90": np.quantile(values, 0.90, axis=1).tolist(),
    }


def _metric_level(fused_effect: float, logits_effect: float, amplification: float | None = None) -> str:
    max_effect = max(float(fused_effect), float(logits_effect))
    if max_effect >= FUNCTIONAL_AMPLIFIED_THRESHOLD and (amplification is None or amplification >= 2.0):
        return "Functionally Material"
    if amplification is not None and amplification >= 2.0:
        return "Functionally Amplified"
    if max_effect > FUNCTIONAL_NUMERICAL_THRESHOLD:
        return "Numerically Active"
    return "Numerically Inactive"


def _metric_activity(effect: dict[str, Any], ratios: dict[str, float]) -> dict[str, Any]:
    fused = float(effect["fused_representation"]["relative_l2"])
    logits = float(effect["logits"]["relative_l2"])
    amplification = max(float(ratios["fused"]), float(ratios["logits"]))
    return {
        "level": _metric_level(fused, logits, amplification),
        "fused_relative_l2": fused,
        "logit_relative_l2": logits,
        "max_effect_size": max(fused, logits),
        "amplification": amplification,
        "thresholds": {
            "numerically_active_gt": FUNCTIONAL_NUMERICAL_THRESHOLD,
            "functionally_amplified_ratio_gte": 2.0,
            "functionally_material_effect_gte": FUNCTIONAL_AMPLIFIED_THRESHOLD,
        },
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
    if on_keys.size:
        on_present = (on_pos < on_keys.size) & (on_keys[np.minimum(on_pos, on_keys.size - 1)] == keys)
    else:
        on_present = np.zeros(keys.size, dtype=bool)
    if alt_keys.size:
        alt_present = (alt_pos < alt_keys.size) & (alt_keys[np.minimum(alt_pos, alt_keys.size - 1)] == keys)
    else:
        alt_present = np.zeros(keys.size, dtype=bool)
    on_full = np.zeros(keys.size, dtype=np.float64)
    alt_full = np.zeros(keys.size, dtype=np.float64)
    on_full[on_present] = on_values[on_pos[on_present]]
    alt_full[alt_present] = alt_values[alt_pos[alt_present]]
    return float(np.linalg.norm(on_full - alt_full) / (np.linalg.norm(on_full) + EPS))


def _relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).norm().item() / (first.norm().item() + EPS))


def _mean_cosine_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    first_flat = first.reshape(first.size(0), -1)
    second_flat = second.reshape(second.size(0), -1)
    return float((1.0 - torch.nn.functional.cosine_similarity(first_flat, second_flat, dim=-1, eps=1e-8)).mean().item())


def _prediction_collapse(logits: torch.Tensor, data) -> dict[str, Any]:
    prediction = logits.argmax(dim=-1).detach().cpu()
    unique_all = int(torch.unique(prediction).numel())
    unique_val = int(torch.unique(prediction[data.val_idx.cpu()]).numel())
    unique_test = int(torch.unique(prediction[data.test_idx.cpu()]).numel())
    return {
        "all_unique_classes": unique_all,
        "val_unique_classes": unique_val,
        "test_unique_classes": unique_test,
        "prediction_collapse": bool(unique_all <= 1 or unique_val <= 1 or unique_test <= 1),
    }


def _saturation(values: torch.Tensor, c_min: float) -> dict[str, Any]:
    array = values.detach().cpu().numpy().astype(np.float64)
    low_cut = c_min + 0.01 * (1.0 - c_min)
    high_cut = 1.0 - 0.01 * (1.0 - c_min)
    distribution = _full_distribution(values)
    low_fraction = float(np.mean(array <= low_cut))
    high_fraction = float(np.mean(array >= high_cut))
    return {
        "c_min": float(c_min),
        "low_cut": float(low_cut),
        "high_cut": float(high_cut),
        "low_saturation_fraction": low_fraction,
        "high_saturation_fraction": high_fraction,
        "distribution": distribution,
        "all_finite": bool(np.isfinite(array).all()),
        "conductance_collapse": bool(distribution["std"] <= 1e-12 or distribution["dynamic_range_q90_minus_q10"] <= 1e-12),
    }


def _snapshot(model: MoPF, classifier: torch.nn.Module, components: dict[str, Any], data, eval_labels: list[int]) -> dict[str, Any]:
    edges = components["edges"]
    with torch.no_grad():
        logits = classifier(components["z"])
    output: dict[str, Any] = {
        "raw_semantic_score": {
            "text": _full_distribution(edges["cos_t"]),
            "visual": _full_distribution(edges["cos_v"]),
        },
        "conductance": {
            "text": _full_distribution(edges["w_t"]),
            "visual": _full_distribution(edges["w_v"]),
        },
        "normalized_operator": {
            "text": _full_distribution(components["norm_t_weight"]),
            "visual": _full_distribution(components["norm_v_weight"]),
        },
        "downstream": _split_metrics(logits, data, eval_labels),
        "prediction": _prediction_collapse(logits, data),
        "finite": bool(
            all(
                bool(torch.isfinite(value).all())
                for value in (
                    edges["cos_t"], edges["cos_v"], edges["w_t"], edges["w_v"],
                    components["norm_t_weight"], components["norm_v_weight"],
                    components["z"], logits,
                )
            )
        ),
    }
    return output


def _core_effects(effect: dict[str, Any]) -> dict[str, float]:
    operator = np.mean([
        effect["modality"]["text"]["normalized_operator_relative_l2_on_denominator"],
        effect["modality"]["visual"]["normalized_operator_relative_l2_on_denominator"],
    ])
    bank_values = [
        item["relative_l2"]
        for modality in ("text", "visual")
        for item in effect["modality"][modality]["propagation_response_bank"]
    ]
    return {
        "operator": float(operator),
        "bank": float(np.mean(bank_values)),
        "fused": float(effect["fused_representation"]["relative_l2"]),
        "logits": float(effect["logits"]["relative_l2"]),
        "probability": float(effect["probabilities"]["mean_abs_change"]),
    }


def _compare(
    model: MoPF,
    classifier: torch.nn.Module,
    data,
    on: dict[str, Any],
    alternative: dict[str, Any],
    eval_labels: list[int],
) -> dict[str, Any]:
    result = _causal_intervention(
        model=model,
        classifier=classifier,
        data=data,
        x=data.x,
        edge_index=data.edge_index,
        on=on,
        alternative=alternative,
        eval_labels=eval_labels,
    )
    result["core_effects"] = _core_effects(result)
    result["on_minus_alternative_downstream"] = {
        key: -float(value)
        for key, value in result["downstream"]["alternative_minus_on"].items()
    }
    return result


def _temperature_diagnostics(
    model: MoPF,
    classifier: torch.nn.Module,
    data,
    eval_labels: list[int],
    learned_text: torch.Tensor,
    learned_visual: torch.Tensor,
    temperatures: tuple[float, ...],
) -> dict[str, Any]:
    x = data.x
    edge_index = data.edge_index
    identity_text = torch.ones_like(learned_text)
    identity_visual = torch.ones_like(learned_visual)

    def encode(text_weights: torch.Tensor | None, visual_weights: torch.Tensor | None, temperature: float):
        with torch.no_grad():
            return model.analysis_encode_with_metric_override(
                x,
                edge_index,
                text_weights=text_weights,
                visual_weights=visual_weights,
                temperature_override=float(temperature),
            )

    learned_2 = encode(None, None, TAU0)
    identity_2 = encode(identity_text, identity_visual, TAU0)
    control_snapshot = {
        "L_2": _snapshot(model, classifier, learned_2, data, eval_labels),
        "I_2": _snapshot(model, classifier, identity_2, data, eval_labels),
    }
    baseline_scores = {
        "learned": {
            "text": learned_2["edges"]["cos_t"].detach().clone(),
            "visual": learned_2["edges"]["cos_v"].detach().clone(),
        },
        "identity": {
            "text": identity_2["edges"]["cos_t"].detach().clone(),
            "visual": identity_2["edges"]["cos_v"].detach().clone(),
        },
    }
    base_effect = _compare(model, classifier, data, learned_2, identity_2, eval_labels)
    base_core = base_effect["core_effects"]
    base_ratios = {name: 1.0 for name in base_core}
    tau_results: dict[str, Any] = {}

    for temperature in temperatures:
        if float(temperature) == TAU0:
            learned = learned_2
            identity = identity_2
        else:
            learned = encode(None, None, temperature)
            identity = encode(identity_text, identity_visual, temperature)
        metric_effect = base_effect if float(temperature) == TAU0 else _compare(
            model, classifier, data, learned, identity, eval_labels
        )
        temp_learned = _compare(model, classifier, data, learned, learned_2, eval_labels)
        temp_identity = _compare(model, classifier, data, identity, identity_2, eval_labels)
        learned_core = metric_effect["core_effects"]
        ratios = {
            name: _ratio(value, base_core[name])
            for name, value in learned_core.items()
        }
        raw_invariance = {}
        for state_name, components, state_scores in (
            ("learned", learned, baseline_scores["learned"]),
            ("identity", identity, baseline_scores["identity"]),
        ):
            raw_invariance[state_name] = {
                "text_max_abs_difference_from_tau2": float(
                    (components["edges"]["cos_t"] - state_scores["text"]).abs().max().item()
                ),
                "visual_max_abs_difference_from_tau2": float(
                    (components["edges"]["cos_v"] - state_scores["visual"]).abs().max().item()
                ),
            }
        saturation = {
            "learned": {
                "text": _saturation(learned["edges"]["w_t"], model.edge_weight_min),
                "visual": _saturation(learned["edges"]["w_v"], model.edge_weight_min),
            },
            "identity": {
                "text": _saturation(identity["edges"]["w_t"], model.edge_weight_min),
                "visual": _saturation(identity["edges"]["w_v"], model.edge_weight_min),
            },
        }
        support_equal = all(
            torch.equal(learned[key], learned_2[key])
            and torch.equal(identity[key], identity_2[key])
            for key in ("norm_t_index", "norm_v_index")
        )
        finite = bool(
            metric_effect["core_effects"]
            and metric_effect["downstream"]
            and _snapshot(model, classifier, learned, data, eval_labels)["finite"]
            and _snapshot(model, classifier, identity, data, eval_labels)["finite"]
        )
        prediction_collapse = bool(
            control_snapshot["L_2"]["prediction"]["prediction_collapse"]
            or control_snapshot["I_2"]["prediction"]["prediction_collapse"]
            or _snapshot(model, classifier, learned, data, eval_labels)["prediction"]["prediction_collapse"]
            or _snapshot(model, classifier, identity, data, eval_labels)["prediction"]["prediction_collapse"]
        )
        conductance_collapse = any(
            saturation[state][modality]["conductance_collapse"]
            for state in ("learned", "identity")
            for modality in ("text", "visual")
        )
        pathology = {
            "all_finite": finite,
            "normalized_operator_finite": all(
                bool(torch.isfinite(components[key]).all())
                for components in (learned, identity)
                for key in ("norm_t_weight", "norm_v_weight")
            ),
            "no_nan_inf": finite,
            "no_conductance_collapse": not conductance_collapse,
            "no_prediction_collapse": not prediction_collapse,
            "passed": bool(finite and not conductance_collapse and not prediction_collapse),
        }
        tau_results[str(float(temperature))] = {
            "temperature": float(temperature),
            "L_tau": _snapshot(model, classifier, learned, data, eval_labels),
            "I_tau": _snapshot(model, classifier, identity, data, eval_labels),
            "metric_effect": metric_effect,
            "temperature_effect_learned": temp_learned,
            "temperature_effect_identity": temp_identity,
            "amplification_ratios": ratios,
            "functional_activity": _metric_activity(metric_effect, ratios),
            "saturation": saturation,
            "pathology": pathology,
            "raw_score_invariance": raw_invariance,
            "edge_support_unchanged": True,
            "gcn_norm_support_unchanged": bool(support_equal),
        }

        if float(temperature) != TAU0:
            del learned, identity
            if str(model.edge_weight_temperature).startswith("2") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    max_score_difference = max(
        item["raw_score_invariance"][state][field]
        for item in tau_results.values()
        for state in ("learned", "identity")
        for field in ("text_max_abs_difference_from_tau2", "visual_max_abs_difference_from_tau2")
    )
    return {
        "control_streams": control_snapshot,
        "tau_results": tau_results,
        "score_invariance": {
            "max_abs_difference": float(max_score_difference),
            "tolerance": SCORE_INVARIANCE_TOLERANCE,
            "passed": bool(max_score_difference < SCORE_INVARIANCE_TOLERANCE),
        },
        "metric_shape": {
            "text": _state_metric_shape(learned_text),
            "visual": _state_metric_shape(learned_visual),
        },
    }


def _load_r1_records() -> list[dict[str, Any]]:
    summary_path = U1R_ROOT / "u1r_master_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Required U1-R summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recommendation = summary.get("final_relation_level_recommendation", {}).get("recommendation")
    if recommendation != "Select R1 — Modality-Adaptive Semantic Conductance":
        raise RuntimeError(f"U1-R recommendation is not R1: {recommendation!r}")
    rows = [row for row in summary.get("per_run", []) if row.get("variant") == "R1"]
    if len(rows) != len(DATASETS) * len(SEEDS):
        raise RuntimeError(f"Expected 15 U1-R R1 records, got {len(rows)}")
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required U1-R R1 checkpoint is missing: {checkpoint}")
    return rows


def _audit_checkpoint(
    row: dict[str, Any],
    device: str,
    temperatures: tuple[float, ...] = TEMPERATURE_GRID,
) -> dict[str, Any]:
    dataset = str(row["dataset"])
    seed = int(row["seed"])
    checkpoint_path = Path(row["checkpoint"])
    checkpoint_before = _checkpoint_digest(checkpoint_path)
    cfg = _compose_r1_cfg(dataset, seed, device, TAU0)
    data = load_mag_data(cfg, "nc", seed)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(checkpoint["head_state"])
    classifier.eval()
    eval_labels = _resolve_nc_eval_labels(data)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    learned_text = model.semantic_metric_weights("text")
    learned_visual = model.semantic_metric_weights("visual")
    model_state_before = _model_state_digest(model)
    diagnostics = _temperature_diagnostics(
        model,
        classifier,
        data,
        eval_labels,
        learned_text,
        learned_visual,
        tuple(float(value) for value in temperatures),
    )
    model_state_after = _model_state_digest(model)
    checkpoint_after = _checkpoint_digest(checkpoint_path)
    diagnostics["state_safety"] = {
        "model_state_unchanged": bool(model_state_before == model_state_after),
        "checkpoint_unchanged": bool(checkpoint_before == checkpoint_after),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
    }
    diagnostics["source"] = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "training_temperature": TAU0,
        "device": device,
        "edge_weight_mode": model.edge_weight_mode,
        "edge_weight_min": float(model.edge_weight_min),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
    }
    if not diagnostics["score_invariance"]["passed"]:
        raise RuntimeError(f"Raw semantic score invariance failed for {dataset} seed={seed}")
    if not diagnostics["state_safety"]["model_state_unchanged"] or not diagnostics["state_safety"]["checkpoint_unchanged"]:
        raise RuntimeError(f"State-safe temperature override failed for {dataset} seed={seed}")
    del model, classifier, data, checkpoint
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return diagnostics


def _phase_a_row(row: dict[str, Any], device: str, output_root: Path, resume: bool) -> dict[str, Any]:
    run_dir = output_root / "phase_a" / "runs" / str(row["dataset"]) / f"seed{row['seed']}"
    result_path = run_dir / "diagnostics.json"
    if resume and result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    diagnostics = _audit_checkpoint(row, device)
    diagnostics["dataset"] = row["dataset"]
    diagnostics["seed"] = int(row["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(result_path, diagnostics)
    return diagnostics


def _phase_a_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for temperature in TEMPERATURE_GRID:
        selected = [row for row in rows if float(row["tau_results"][str(float(temperature))]["temperature"]) == float(temperature)]
        # The selected rows above are one per checkpoint; retain the explicit
        # dataset/seed filter and use the temperature payload for values.
        selected = [row for row in rows]
        payloads = [row["tau_results"][str(float(temperature))] for row in selected]
        dataset_summary: dict[str, Any] = {}
        for dataset in DATASETS:
            group = [row for row in rows if row["dataset"] == dataset]
            values = [row["tau_results"][str(float(temperature))] for row in group]
            val_deltas = [item["temperature_effect_learned"]["on_minus_alternative_downstream"]["val_acc"] for item in values]
            f1_deltas = [item["temperature_effect_learned"]["on_minus_alternative_downstream"]["val_macro_f1"] for item in values]
            ratios = {name: [item["amplification_ratios"][name] for item in values] for name in ("operator", "bank", "fused", "logits", "probability")}
            saturation_values = [
                max(
                    item["saturation"][state][modality][field]
                    for state in ("learned", "identity")
                    for modality in ("text", "visual")
                    for field in ("low_saturation_fraction", "high_saturation_fraction")
                )
                for item in values
            ]
            dataset_summary[dataset] = {
                "seed_count": len(group),
                "seeds": [int(row["seed"]) for row in group],
                "validation": {
                    "val_acc_drift_from_tau2": _numeric(val_deltas),
                    "val_macro_f1_drift_from_tau2": _numeric(f1_deltas),
                    "direction_consistency_nonnegative_val_acc": float(np.mean(np.asarray(val_deltas) >= 0.0)),
                },
                "amplification": {name: _numeric(values_) for name, values_ in ratios.items()},
                "saturation_fraction_max": _numeric(saturation_values),
                "prediction_flip_rate": _numeric([
                    item["metric_effect"]["predictions"]["all_flip_rate"] for item in values
                ]),
                "pathology_passed_all_seeds": bool(all(item["pathology"]["passed"] for item in values)),
                "metric_effect": {
                    "fused": _numeric([item["metric_effect"]["core_effects"]["fused"] for item in values]),
                    "logits": _numeric([item["metric_effect"]["core_effects"]["logits"] for item in values]),
                },
            }
        overall = {
            "validation_val_acc_drift_from_tau2": _numeric([
                item["temperature_effect_learned"]["on_minus_alternative_downstream"]["val_acc"] for item in payloads
            ]),
            "validation_val_macro_f1_drift_from_tau2": _numeric([
                item["temperature_effect_learned"]["on_minus_alternative_downstream"]["val_macro_f1"] for item in payloads
            ]),
            "amplification": {
                name: _numeric([item["amplification_ratios"][name] for item in payloads])
                for name in ("operator", "bank", "fused", "logits", "probability")
            },
            "prediction_flip_rate": _numeric([
                item["metric_effect"]["predictions"]["all_flip_rate"] for item in payloads
            ]),
        }
        output[str(float(temperature))] = {
            "temperature": float(temperature),
            "overall_15_checkpoints": overall,
            "per_dataset_3_seed": dataset_summary,
        }
    return output


def _phase_a_candidate_selection(rows: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    details: dict[str, Any] = {}
    for temperature in TEMPERATURE_GRID:
        if temperature == TAU0:
            continue
        tau_key = str(float(temperature))
        tau_aggregate = aggregate[tau_key]
        groups = tau_aggregate["per_dataset_3_seed"]
        selected_rows = [row for row in rows]
        no_pathology = all(
            row["tau_results"][tau_key]["pathology"]["passed"]
            for row in selected_rows
        )
        no_excessive_saturation = all(
            groups[dataset]["saturation_fraction_max"]["mean"] <= LOW_SATURATION_THRESHOLD
            for dataset in DATASETS
        )
        amplified_datasets = []
        for dataset in DATASETS:
            group = [row for row in rows if row["dataset"] == dataset]
            values = [row["tau_results"][tau_key] for row in group]
            fused_ratios = [value["amplification_ratios"]["fused"] for value in values]
            logit_ratios = [value["amplification_ratios"]["logits"] for value in values]
            direction = [max(fused, logits) >= 1.0 for fused, logits in zip(fused_ratios, logit_ratios, strict=True)]
            dataset_amplified = bool(
                (
                    groups[dataset]["amplification"]["fused"]["mean"] >= 2.0
                    or groups[dataset]["amplification"]["logits"]["mean"] >= 2.0
                )
                and sum(direction) >= 2
            )
            if dataset_amplified:
                amplified_datasets.append(dataset)
        mean_val_drift = float(tau_aggregate["overall_15_checkpoints"]["validation_val_acc_drift_from_tau2"]["mean"])
        dataset_val_drifts = {
            dataset: float(groups[dataset]["validation"]["val_acc_drift_from_tau2"]["mean"])
            for dataset in DATASETS
        }
        validation_safe = bool(
            mean_val_drift >= -VAL_MEAN_DROP_TOLERANCE
            and all(value >= -VAL_DATASET_DROP_TOLERANCE for value in dataset_val_drifts.values())
        )
        passed = bool(
            no_pathology
            and no_excessive_saturation
            and len(amplified_datasets) >= 3
            and validation_safe
        )
        details[tau_key] = {
            "temperature": float(temperature),
            "passed": passed,
            "no_pathology": no_pathology,
            "no_excessive_saturation": no_excessive_saturation,
            "amplified_datasets": amplified_datasets,
            "amplified_dataset_count": len(amplified_datasets),
            "validation_safe": validation_safe,
            "mean_val_acc_drift": mean_val_drift,
            "per_dataset_val_acc_drift": dataset_val_drifts,
            "selection_uses_test": False,
        }
        if passed:
            candidates.append(float(temperature))
    candidates.sort(reverse=True)
    selected: list[float] = []
    if candidates:
        selected.append(max(candidates))
        if len(candidates) > 1:
            strongest = min(
                candidates,
                key=lambda tau: -max(
                    float(aggregate[str(float(tau))]["overall_15_checkpoints"]["amplification"]["fused"]["mean"]),
                    float(aggregate[str(float(tau))]["overall_15_checkpoints"]["amplification"]["logits"]["mean"]),
                ),
            )
            if strongest not in selected:
                selected.append(strongest)
    return {
        "control_temperature": TAU0,
        "candidate_gate": {
            "no_pathology": "all 15 frozen runs finite, no prediction collapse, no conductance collapse",
            "saturation_threshold": LOW_SATURATION_THRESHOLD,
            "amplification_threshold": 2.0,
            "validation_mean_drop_tolerance": VAL_MEAN_DROP_TOLERANCE,
            "validation_dataset_drop_tolerance": VAL_DATASET_DROP_TOLERANCE,
        },
        "per_temperature": details,
        "passing_temperatures": candidates,
        "selected_retraining_temperatures": selected[:2],
    }


def _write_temperature_table(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        for tau_key, item in row["tau_results"].items():
            flat_rows.append({
                "dataset": row["dataset"],
                "seed": row["seed"],
                "temperature": item["temperature"],
                "val_acc_drift_from_tau2": item["temperature_effect_learned"]["on_minus_alternative_downstream"]["val_acc"],
                "val_macro_f1_drift_from_tau2": item["temperature_effect_learned"]["on_minus_alternative_downstream"]["val_macro_f1"],
                "operator_amplification": item["amplification_ratios"]["operator"],
                "bank_amplification": item["amplification_ratios"]["bank"],
                "fused_amplification": item["amplification_ratios"]["fused"],
                "logit_amplification": item["amplification_ratios"]["logits"],
                "probability_amplification": item["amplification_ratios"]["probability"],
                "metric_fused_relative_l2": item["metric_effect"]["core_effects"]["fused"],
                "metric_logit_relative_l2": item["metric_effect"]["core_effects"]["logits"],
                "prediction_flip_rate": item["metric_effect"]["predictions"]["all_flip_rate"],
                "learned_low_saturation_max": max(item["saturation"]["learned"][modality]["low_saturation_fraction"] for modality in ("text", "visual")),
                "learned_high_saturation_max": max(item["saturation"]["learned"][modality]["high_saturation_fraction"] for modality in ("text", "visual")),
                "identity_low_saturation_max": max(item["saturation"]["identity"][modality]["low_saturation_fraction"] for modality in ("text", "visual")),
                "identity_high_saturation_max": max(item["saturation"]["identity"][modality]["high_saturation_fraction"] for modality in ("text", "visual")),
                "pathology_passed": item["pathology"]["passed"],
                "score_invariance_passed": row["score_invariance"]["passed"],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def _run_retraining_job(
    dataset: str,
    seed: int,
    temperature: float,
    label: str,
    device: str,
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    k = K_BY_DATASET[dataset]
    run_dir = output_root / "phase_b" / "runs" / label / dataset / f"seed{seed}"
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
        "model.edge_weight_mode=learned_diag_cos",
        f"model.edge_weight_temperature={temperature}",
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
    log_path = run_dir / "process.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        peak = None
        while process.poll() is None:
            current = _peak_gpu_memory_mib(process.pid, device)
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(2.0)
        current = _peak_gpu_memory_mib(process.pid, device)
        if current is not None:
            peak = current if peak is None else max(peak, current)
    if process.returncode != 0:
        raise RuntimeError(f"U1-T retraining failed: {dataset} seed={seed} tau={temperature}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metrics = dict(checkpoint.get("metrics", {}))
    cfg = _compose_r1_cfg(dataset, seed, device, temperature)
    model = build_model(cfg, checkpoint["data_info"])
    record = {
        "dataset": dataset,
        "seed": seed,
        "temperature": float(temperature),
        "label": label,
        "device": device,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "downstream": {key: metrics.get(key) for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")},
        "best_epoch": checkpoint.get("epoch"),
        "parameter_count": count_parameters(model) + int(model.out_dim + 1) * int(checkpoint["data_info"]["num_classes"]),
        "training_time_seconds": float(time.monotonic() - started),
        "peak_gpu_memory_mib": peak,
        "peak_gpu_memory_status": "available" if peak is not None else "unavailable",
    }
    _dump_json(record_path, record)
    return record


def _candidate_vs_t0(retrained: list[dict[str, Any]], t0_rows: list[dict[str, Any]]) -> dict[str, Any]:
    t0 = {(row["dataset"], int(row["seed"])): row["downstream"] for row in t0_rows}
    output: dict[str, Any] = {}
    for label in sorted({row["label"] for row in retrained}):
        rows = [row for row in retrained if row["label"] == label]
        per_seed = []
        for row in rows:
            control = t0[(row["dataset"], int(row["seed"]))]
            per_seed.append({
                "dataset": row["dataset"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                **{
                    f"{metric}_delta_candidate_minus_T0": float(row["downstream"][metric] - control[metric])
                    for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
                },
            })
        per_dataset = {}
        for dataset in DATASETS:
            group = [item for item in per_seed if item["dataset"] == dataset]
            per_dataset[dataset] = {
                metric: _numeric([item[f"{metric}_delta_candidate_minus_T0"] for item in group])
                for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
            }
        output[label] = {
            "temperature": rows[0]["temperature"],
            "per_seed": per_seed,
            "per_dataset_3_seed": per_dataset,
            "unweighted_dataset_mean_delta": {
                metric: _numeric([per_dataset[dataset][metric]["mean"] for dataset in DATASETS])
                for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
            },
        }
    return output


def _post_training_decision(retrained: list[dict[str, Any]], t0_rows: list[dict[str, Any]], comparisons: dict[str, Any]) -> dict[str, Any]:
    candidate_details = {}
    pass_labels = []
    for label, comparison in comparisons.items():
        rows = [row for row in retrained if row["label"] == label]
        dataset_amplified = []
        dataset_material = []
        temp_off_active = []
        for dataset in DATASETS:
            group = [row for row in rows if row["dataset"] == dataset]
            activities = [row["frozen_identity_audit"]["functional_activity"] for row in group]
            amps = [activity["amplification"] for activity in activities]
            effects = [activity["max_effect_size"] for activity in activities]
            amp_direction = sum(amp >= 2.0 for amp in amps)
            material_direction = sum(effect >= FUNCTIONAL_AMPLIFIED_THRESHOLD and amp >= 2.0 for effect, amp in zip(effects, amps, strict=True))
            if float(np.mean(amps)) >= 2.0 and amp_direction >= 2:
                dataset_amplified.append(dataset)
            if np.median(effects) >= FUNCTIONAL_AMPLIFIED_THRESHOLD and np.median(amps) >= 2.0:
                dataset_material.append(dataset)
            off_effects = [
                max(
                    row["frozen_tau2_audit"]["core_effects"]["fused"],
                    row["frozen_tau2_audit"]["core_effects"]["logits"],
                )
                for row in group
            ]
            if float(np.mean(off_effects)) > FUNCTIONAL_NUMERICAL_THRESHOLD:
                temp_off_active.append(dataset)
        mean_val_delta = float(comparison["unweighted_dataset_mean_delta"]["val_acc"]["mean"])
        dataset_val_delta = {
            dataset: float(comparison["per_dataset_3_seed"][dataset]["val_acc"]["mean"])
            for dataset in DATASETS
        }
        validation_same_band = bool(
            mean_val_delta >= -FINAL_VAL_MEAN_DROP_TOLERANCE
            and all(value >= -FINAL_VAL_DATASET_DROP_TOLERANCE for value in dataset_val_delta.values())
        )
        no_pathology = all(row["frozen_identity_audit"]["pathology"]["passed"] for row in rows)
        functionally_amplified = len(dataset_amplified) >= 3
        functionally_material = len(dataset_material) >= 3
        inference_path_active = len(temp_off_active) >= 3
        passed = bool(no_pathology and validation_same_band and functionally_amplified and functionally_material and inference_path_active)
        candidate_details[label] = {
            "temperature": rows[0]["temperature"],
            "no_pathology": no_pathology,
            "validation_same_band": validation_same_band,
            "mean_val_acc_delta": mean_val_delta,
            "per_dataset_val_acc_delta": dataset_val_delta,
            "functionally_amplified_datasets": dataset_amplified,
            "functionally_material_datasets": dataset_material,
            "temperature_off_inference_active_datasets": temp_off_active,
            "passed_final_selection": passed,
        }
        if passed:
            pass_labels.append(label)
    selected_label = None
    if pass_labels:
        selected_label = max(
            pass_labels,
            key=lambda label: (
                float(comparisons[label]["unweighted_dataset_mean_delta"]["val_acc"]["mean"]),
                len(candidate_details[label]["functionally_material_datasets"]),
                -float(candidate_details[label]["temperature"]),
            ),
        )
    return {
        "candidate_details": candidate_details,
        "selected_label": selected_label,
        "selected_tau": None if selected_label is None else candidate_details[selected_label]["temperature"],
        "decision": (
            "Select Calibrated R1 with tau = " + str(candidate_details[selected_label]["temperature"])
            if selected_label is not None
            else "Keep R1 with tau = 2.0 (calibration rejected)"
        ),
        "selection_uses_test": False,
    }


def _write_retraining_table(path: Path, rows: list[dict[str, Any]], comparisons: dict[str, Any]) -> None:
    flat = []
    for row in rows:
        comparison = next(
            item for item in comparisons[row["label"]]["per_seed"]
            if item["dataset"] == row["dataset"] and int(item["seed"]) == int(row["seed"])
        )
        flat.append({
            "label": row["label"],
            "temperature": row["temperature"],
            "dataset": row["dataset"],
            "seed": row["seed"],
            **row["downstream"],
            **{key: value for key, value in comparison.items() if key.endswith("_delta_candidate_minus_T0")},
            "frozen_metric_level": row["frozen_identity_audit"]["functional_activity"]["level"],
            "frozen_metric_effect_max": row["frozen_identity_audit"]["functional_activity"]["max_effect_size"],
            "frozen_metric_amplification": row["frozen_identity_audit"]["functional_activity"]["amplification"],
            "temperature_off_fused_effect": row["frozen_tau2_audit"]["core_effects"]["fused"],
            "temperature_off_logit_effect": row["frozen_tau2_audit"]["core_effects"]["logits"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]) if flat else ["label"])
        writer.writeheader()
        writer.writerows(flat)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["final_relation_temperature_decision"]
    phase_a = summary["phase_a_candidate_selection"]
    lines = [
        "# MoPF-vNext U1-T — Semantic Conductance Calibration",
        "",
        "NC-only controlled temperature calibration. The R1 metric form is frozen; only the global fixed conductance temperature was varied.",
        "",
        f"- Experiment commit: `{summary['experiment_runtime_commit']}`",
        f"- Public reproducible commit: `{summary['public_reproducible_commit']}`",
        f"- U1-R source: `{summary['u1r_source']['summary']}`",
        f"- Phase A passing temperatures: `{phase_a['passing_temperatures']}`",
        f"- Final decision: **{decision['decision']}**",
        "",
        "## Phase A diagnosis",
        "",
        "Every R1 checkpoint was evaluated with learned and identity metric streams at each temperature. Raw semantic scores were required to remain invariant to temperature, and all state/checkpoint hashes were required to remain unchanged.",
        "",
        "Candidate ranking uses validation drift and continuous mechanism effects only; Test metrics are descriptive and are not used for selection.",
        "",
        "## Functional activity levels",
        "",
        "- Numerically Active: effect above numerical-noise screening.",
        "- Functionally Amplified: at least 2x the tau=2 metric effect.",
        "- Functionally Material: amplified and max fused/logit effect at least 1e-4.",
        "",
        "The authoritative detailed evidence is in `u1t_master_summary.json`.",
        "",
        "## Phase B",
        "",
        "Phase B was run only for temperatures that passed the Phase-A gate. The retrained frozen identity and temperature-off audits test whether sharpening remains in the inference path after co-adaptation.",
        "",
        "No U2, LP, propagation-bank, personalized-filter, fusion, or auxiliary-loss changes were started.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase-a-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    devices = [str(device) for device in args.devices]
    r1_rows = _load_r1_records()
    phase_a_rows = []
    for index, row in enumerate(r1_rows):
        device = devices[index % len(devices)]
        print(f"[U1-T] Phase A {index + 1}/{len(r1_rows)} {row['dataset']} seed={row['seed']} {device}", flush=True)
        phase_a_rows.append(_phase_a_row(row, device, output_root, resume=not args.no_resume))
    phase_a_rows.sort(key=lambda item: (DATASETS.index(item["dataset"]), SEEDS.index(int(item["seed"]))))
    phase_a_aggregate = _phase_a_aggregate(phase_a_rows)
    candidate_selection = _phase_a_candidate_selection(phase_a_rows, phase_a_aggregate)
    phase_a_payload = {
        "metadata": {
            "stage": "MoPF-vNext U1-T Semantic Conductance Calibration",
            "phase": "A Frozen Temperature Diagnosis",
            "scope": "NC only",
            "temperature_grid": list(TEMPERATURE_GRID),
            "control_temperature": TAU0,
            "candidate_selection_uses_test": False,
        },
        "per_checkpoint": phase_a_rows,
        "aggregate": phase_a_aggregate,
        "candidate_selection": candidate_selection,
    }
    _dump_json(output_root / "u1t_frozen_temperature_sweep.json", phase_a_payload)
    _write_temperature_table(output_root / "u1t_frozen_temperature_table.csv", phase_a_rows)
    if candidate_selection["selected_retraining_temperatures"] and not args.phase_a_only:
        print(f"[U1-T] Phase A passed candidates: {candidate_selection['selected_retraining_temperatures']}", flush=True)
    else:
        print("[U1-T] Phase A selected no retraining candidate or phase-a-only was requested", flush=True)

    retrained: list[dict[str, Any]] = []
    if candidate_selection["selected_retraining_temperatures"] and not args.phase_a_only:
        for candidate_index, temperature in enumerate(candidate_selection["selected_retraining_temperatures"], start=1):
            label = f"T{candidate_index}"
            for job_index, (dataset, seed) in enumerate(((dataset, seed) for dataset in DATASETS for seed in SEEDS), start=1):
                device = devices[(candidate_index * len(DATASETS) * len(SEEDS) + job_index - 1) % len(devices)]
                print(f"[U1-T] Phase B {label} {job_index}/15 {dataset} seed={seed} tau={temperature} {device}", flush=True)
                record = _run_retraining_job(dataset, seed, temperature, label, device, output_root, resume=not args.no_resume)
                source_row = {"dataset": dataset, "seed": seed, "checkpoint": record["checkpoint"]}
                audit = _audit_checkpoint(source_row, device, temperatures=tuple(sorted({TAU0, float(temperature)})))
                selected_key = str(float(temperature))
                control_key = str(float(TAU0))
                record["frozen_identity_audit"] = audit["tau_results"][selected_key]["metric_effect"]
                record["frozen_identity_audit"]["functional_activity"] = audit["tau_results"][selected_key]["functional_activity"]
                record["frozen_identity_audit"]["pathology"] = audit["tau_results"][selected_key]["pathology"]
                record["frozen_tau2_audit"] = audit["tau_results"][selected_key]["temperature_effect_learned"]
                record["frozen_audit_score_invariance"] = audit["score_invariance"]
                record["frozen_audit_state_safety"] = audit["state_safety"]
                retrained.append(record)
                _dump_json(Path(record["run_dir"]) / "frozen_post_training_audit.json", audit)
    t0_rows = r1_rows
    comparisons = _candidate_vs_t0(retrained, t0_rows) if retrained else {}
    final_decision = _post_training_decision(retrained, t0_rows, comparisons) if retrained else {
        "candidate_details": {},
        "selected_label": None,
        "selected_tau": None,
        "decision": "Keep R1 with tau = 2.0 (calibration rejected)",
        "selection_uses_test": False,
    }
    _write_retraining_table(output_root / "u1t_retraining_table.csv", retrained, comparisons) if retrained else (output_root / "u1t_retraining_table.csv").write_text("label\n", encoding="utf-8")

    phase_b_summary = {
        "trained_candidates": sorted({row["label"] for row in retrained}),
        "per_run": retrained,
        "candidate_vs_T0": comparisons,
        "final_candidate_gate": final_decision,
    }
    summary = {
        "metadata": {
            "stage": "MoPF-vNext U1-T Semantic Conductance Calibration",
            "scope": "NC only",
            "no_lp_experiments": True,
            "no_u2_started": True,
            "edge_weight_mode": "learned_diag_cos",
            "metric_frozen": "s_ij=cos(w^m* h_i^m, w^m* h_j^m), w=softplus(theta)/mean(softplus(theta))",
            "temperature_override_path": "analysis_encode_with_metric_override(..., temperature_override=tau)",
            "temperature_global_fixed_shared": True,
            "mopf_v0_frozen_untouched": True,
            "candidate_selection_uses_test": False,
        },
        "experiment_runtime_commit": _git("rev-parse", "HEAD"),
        "public_reproducible_commit": _git("rev-parse", "HEAD"),
        "git_clean_status": _git("status", "--short") or "clean",
        "frozen_reference_commit": _git("rev-parse", "mopf-v0-frozen"),
        "u1r_source": {
            "summary": str(U1R_ROOT / "u1r_master_summary.json"),
            "recommendation": "Select R1 — Modality-Adaptive Semantic Conductance",
            "checkpoint_count": len(r1_rows),
            "checkpoints": [row["checkpoint"] for row in r1_rows],
        },
        "temperature_grid": list(TEMPERATURE_GRID),
        "phase_a": phase_a_payload,
        "phase_a_candidate_selection": candidate_selection,
        "phase_b": phase_b_summary,
        "functional_activity_levels": {
            "numerically_active_gt": FUNCTIONAL_NUMERICAL_THRESHOLD,
            "functionally_amplified_ratio_gte": 2.0,
            "functionally_material_effect_gte": FUNCTIONAL_AMPLIFIED_THRESHOLD,
        },
        "final_relation_temperature_decision": final_decision,
        "artifacts": {
            "frozen_temperature_sweep": str(output_root / "u1t_frozen_temperature_sweep.json"),
            "frozen_temperature_table": str(output_root / "u1t_frozen_temperature_table.csv"),
            "retraining_table": str(output_root / "u1t_retraining_table.csv"),
            "master_summary": str(output_root / "u1t_master_summary.json"),
            "report": str(ROOT / "docs/mopf_u1t_semantic_conductance_calibration.md"),
        },
    }
    _dump_json(output_root / "u1t_master_summary.json", summary)
    _write_report(ROOT / "docs/mopf_u1t_semantic_conductance_calibration.md", summary)
    print(f"[U1-T] master summary={output_root / 'u1t_master_summary.json'}", flush=True)
    print(f"[U1-T] final decision={final_decision['decision']}", flush=True)


if __name__ == "__main__":
    main()
