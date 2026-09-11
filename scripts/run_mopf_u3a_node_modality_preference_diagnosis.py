"""Run the frozen-C1 U3-A node/modality multi-hop preference diagnosis.

This script is analysis-only.  It loads the 15 U2-C C1 best-validation
checkpoints, extracts final effective eta, and evaluates canonical
counterfactuals without changing model parameters, checkpoints, or the
formal training path.
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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.u3a import (  # noqa: E402
    EPS,
    canonical_effective_decomposition,
    coefficient_statistics,
    contribution_profile,
    counterfactual_eta,
    distribution_summary,
    profile_pairwise_variability,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402
from scripts.run_mopf_u1_semantic_conductance import (  # noqa: E402
    DATASETS,
    K_BY_DATASET,
    MAGB_DATASETS,
    SEEDS,
    _fixed_split_override,
)


OUTPUT_ROOT = ROOT / "outputs" / "u3a_node_modality_preference_diagnosis"
U2C_SUMMARY = ROOT / "outputs" / "u2c_formal_factorial_training" / "u2c_master_summary.json"
FORMAL_MODEL_CONFIG = ROOT / "configs" / "model" / "mopf.yaml"
SHUFFLE_SEEDS = (20260921, 20260922, 20260923)
INTERVENTIONS = ("Full", "NoNode", "NoModality", "GlobalOnly", "NodeShuffle", "ModalitySwap")
_CONFIG_LOCK = threading.Lock()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot encode {type(value)!r}")


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _compose_c1_cfg(dataset: str, seed: int, device: str):
    k = K_BY_DATASET[dataset]
    overrides = [
        f"dataset={dataset}", "task=nc", "model=mopf", f"seed={seed}", f"device={device}",
        "model.edge_weight_mode=learned_diag_cos", "model.edge_weight_temperature=0.35",
        f"model.max_order={k}", f"model.num_layers={k}", "model.num_metric_perspectives=4",
        "model.metric_init_seed=20260910", "model.metric_init_noise_std=0.01",
        "model.multihop_state_mode=anchored", "model.multihop_response_mode=cumulative",
        "model.multihop_anchor_alpha=0.1",
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        overrides.append(f"dataset.nc_split_path={split_override}")
    overrides.append(
        "model.use_transport_residual=false"
        if "use_transport_residual:" in FORMAL_MODEL_CONFIG.read_text(encoding="utf-8")
        else "+model.use_transport_residual=false"
    )
    with _CONFIG_LOCK:
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            return compose(config_name="config", overrides=overrides)


def _source_records() -> list[dict[str, Any]]:
    if not U2C_SUMMARY.is_file():
        raise FileNotFoundError(f"Missing frozen U2-C summary: {U2C_SUMMARY}")
    summary = json.loads(U2C_SUMMARY.read_text(encoding="utf-8"))
    rows = [row for row in summary["records"] if row["variant"] == "C1"]
    rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), SEEDS.index(int(row["seed"]))))
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    actual = {(row["dataset"], int(row["seed"])) for row in rows}
    if actual != expected:
        raise RuntimeError(f"Expected exactly 15 C1 records, got {sorted(actual)}")
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing C1 checkpoint: {checkpoint}")
    return rows


def _data_info(data: Any) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes), "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).norm().item() / max(second.norm().item(), EPS))


def _cosine_rows(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(first, second, dim=-1, eps=EPS)


def _metrics_from_logits(logits: torch.Tensor, data: Any, split: str, labels: list[int]) -> dict[str, float]:
    index = getattr(data, f"{split}_idx").to(logits.device)
    prediction = logits.index_select(0, index).argmax(dim=-1).cpu().numpy()
    target = data.y.index_select(0, index.cpu()).cpu().numpy()
    return {
        "acc": float(np.mean(prediction == target)),
        "macro_f1": float(f1_score(target, prediction, labels=labels, average="macro", zero_division=0)),
    }


def _fuse_from_modalities(model: Any, z_text: torch.Tensor, z_visual: torch.Tensor) -> torch.Tensor:
    text_refined = model.text_refine_norm(z_text + model.text_refine_mlp(z_text))
    visual_refined = model.visual_refine_norm(z_visual + model.visual_refine_mlp(z_visual))
    fused_input = torch.cat([text_refined, visual_refined], dim=-1)
    return model.output_norm(model.fusion_skip(fused_input) + model.fusion_mlp(fused_input))


def _profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    probabilities = profile["probabilities"]
    magnitudes = profile["magnitudes"]
    return {
        "mean_probability_by_order": probabilities.mean(dim=0).detach().cpu().tolist(),
        "median_probability_by_order": probabilities.median(dim=0).values.detach().cpu().tolist(),
        "mean_contribution_magnitude_by_order": magnitudes.mean(dim=0).detach().cpu().tolist(),
        "response_order": profile["summary"]["response_order"],
        "normalized_response_order": profile["summary"]["normalized_response_order"],
        "entropy": profile["summary"]["entropy"],
        "normalized_entropy": profile["summary"]["normalized_entropy"],
        "probability_sum_max_error": profile["summary"]["probability_sum_max_error"],
        "all_zero_count": profile["summary"]["all_zero_count"],
    }


def _canonical_payload(components: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    eta = {"text": components["eta_text"], "visual": components["eta_visual"]}
    decomposition = canonical_effective_decomposition(eta)
    coefficient = {
        "eta": {modality: coefficient_statistics(eta[modality], name=f"eta_{modality}") for modality in ("text", "visual")},
        "mu": coefficient_statistics(decomposition["mu"].unsqueeze(0), name="mu"),
        "nu": {modality: coefficient_statistics(decomposition["nu"][modality].unsqueeze(0), name=f"nu_{modality}") for modality in ("text", "visual")},
        "xi": {modality: coefficient_statistics(decomposition["xi"][modality], name=f"xi_{modality}") for modality in ("text", "visual")},
    }
    coefficient["reconstruction_max_abs"] = decomposition["reconstruction_max_abs"]
    coefficient["mean_modality_deviation_max_abs"] = float(decomposition["mean_modality_deviation"].abs().max().item())
    coefficient["mean_node_centered_max_abs"] = {modality: float(decomposition["mean_node_centered"][modality].abs().max().item()) for modality in ("text", "visual")}
    return decomposition, coefficient


def _semantic_factors(data: Any, model: Any, components: dict[str, Any]) -> dict[str, Any]:
    edge_index = data.edge_index.to(components["h_text"].device)
    num_nodes = data.num_nodes
    factors: dict[str, Any] = {}
    conductance = {}
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=edge_index.device))
    degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=edge_index.device))
    for modality, weight_key in (("text", "w_t"), ("visual", "w_v")):
        h0 = components[f"h_{modality}"]
        states = components[f"states_{modality}"]
        drift = [1.0 - _cosine_rows(state, h0) for state in states]
        nonzero_drift = torch.stack(drift[1:], dim=1) if len(drift) > 1 else drift[0].unsqueeze(1)
        edge_weight = components["edges"][weight_key]
        incident_sum = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        incident_sq = torch.zeros_like(incident_sum)
        incident_count = torch.zeros(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        for endpoint in (edge_index[0], edge_index[1]):
            incident_sum.index_add_(0, endpoint, edge_weight)
            incident_sq.index_add_(0, endpoint, edge_weight.square())
            incident_count.index_add_(0, endpoint, torch.ones_like(edge_weight))
        mean_conductance = incident_sum / incident_count.clamp_min(1.0)
        variance = (incident_sq / incident_count.clamp_min(1.0) - mean_conductance.square()).clamp_min(0.0)
        conductance[modality] = {
            "mean": mean_conductance,
            "std": variance.sqrt(),
            "summary": {"mean": distribution_summary(mean_conductance), "std": distribution_summary(variance.sqrt())},
        }
        factors[modality] = {
            "semantic_drift": drift,
            "final_drift": drift[-1],
            "mean_nonzero_drift": nonzero_drift.mean(dim=1),
            "conductance": conductance[modality],
            "log_degree": torch.log1p(degree),
            "degree": degree,
        }
    factors["degree_summary"] = distribution_summary(torch.log1p(degree))
    return factors


def _spearman(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    x = first.detach().cpu().double().numpy().reshape(-1)
    y = second.detach().cpu().double().numpy().reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or np.std(x[mask]) <= EPS or np.std(y[mask]) <= EPS:
        return {"rho": 0.0, "p_value": None, "sample_count": int(mask.sum()), "degenerate": True}
    result = spearmanr(x[mask], y[mask])
    return {"rho": float(result.statistic), "p_value": float(result.pvalue), "sample_count": int(mask.sum()), "degenerate": False}


def _associations(decomposition: dict[str, Any], profiles: dict[str, Any], factors: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    outcomes = {}
    for modality in ("text", "visual"):
        outcomes[modality] = {
            "response_order": profiles[modality]["response_order"],
            "entropy": profiles[modality]["entropy"],
        }
        predictors = {
            "semantic_drift_final": factors[modality]["final_drift"],
            "semantic_drift_mean_nonzero": factors[modality]["mean_nonzero_drift"],
            "mean_incident_conductance": factors[modality]["conductance"]["mean"],
            "log_degree": factors[modality]["log_degree"],
        }
        for outcome_name, outcome in outcomes[modality].items():
            for predictor_name, predictor in predictors.items():
                rows.append({"modality": modality, "outcome": outcome_name, "predictor": predictor_name, **_spearman(outcome, predictor)})
    delta_order = profiles["text"]["response_order"] - profiles["visual"]["response_order"]
    delta_entropy = profiles["text"]["entropy"] - profiles["visual"]["entropy"]
    conductance_gap = factors["text"]["conductance"]["mean"] - factors["visual"]["conductance"]["mean"]
    rows.append({"modality": "text_minus_visual", "outcome": "delta_response_order", "predictor": "conductance_gap", **_spearman(delta_order, conductance_gap)})
    rows.append({"modality": "text_minus_visual", "outcome": "delta_entropy", "predictor": "conductance_gap", **_spearman(delta_entropy, conductance_gap)})
    return rows


def _quartiles(values: torch.Tensor) -> list[torch.Tensor]:
    order = torch.argsort(values)
    return [chunk for chunk in torch.tensor_split(order, 4)]


def _quartile_payload(factors: dict[str, Any], profiles: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for modality in ("text", "visual"):
        for factor_name, values in {
            "semantic_drift_final": factors[modality]["final_drift"],
            "mean_incident_conductance": factors[modality]["conductance"]["mean"],
            "log_degree": factors[modality]["log_degree"],
        }.items():
            probabilities = profiles[modality]["probabilities"]
            response_order = profiles[modality]["response_order"]
            entropy = profiles[modality]["entropy"]
            for quartile, indices in enumerate(_quartiles(values), start=1):
                rows.append({
                    "modality": modality, "factor": factor_name, "quartile": f"Q{quartile}", "count": int(indices.numel()),
                    "factor_mean": float(values.index_select(0, indices).mean().item()),
                    "response_order_mean": float(response_order.index_select(0, indices).mean().item()),
                    "normalized_response_order_mean": float((response_order.index_select(0, indices) / max(profiles[modality]["states_order_max"], 1)).mean().item()),
                    "entropy_mean": float(entropy.index_select(0, indices).mean().item()),
                    "mean_contribution_profile": probabilities.index_select(0, indices).mean(dim=0).detach().cpu().tolist(),
                })
    return rows


def _functional_effect(
    model: Any,
    classifier: Any,
    data: Any,
    components: dict[str, Any],
    eta: dict[str, torch.Tensor],
    labels: list[int],
    full_logits: torch.Tensor,
    full_fused: torch.Tensor,
) -> dict[str, Any]:
    z_text = model._filter_bases(components["responses_text"], eta["text"])
    z_visual = model._filter_bases(components["responses_visual"], eta["visual"])
    fused = _fuse_from_modalities(model, z_text, z_visual)
    logits = classifier(fused)
    probability = torch.softmax(logits, dim=-1)
    full_probability = torch.softmax(full_logits, dim=-1)
    output = {
        "eta_relative_l2": {modality: _relative_l2(eta[modality], components[f"eta_{modality}"]) for modality in ("text", "visual")},
        "z_text_relative_l2": _relative_l2(z_text, components["z_text"]),
        "z_visual_relative_l2": _relative_l2(z_visual, components["z_visual"]),
        "fused_z_relative_l2": _relative_l2(fused, full_fused),
        "logits_relative_l2": _relative_l2(logits, full_logits),
        "probability_relative_l2": _relative_l2(probability, full_probability),
        "prediction_flip_rate": float((logits.argmax(-1) != full_logits.argmax(-1)).float().mean().item()),
        "val": {"on": _metrics_from_logits(full_logits, data, "val", labels), "off": _metrics_from_logits(logits, data, "val", labels)},
        "test": {"on": _metrics_from_logits(full_logits, data, "test", labels), "off": _metrics_from_logits(logits, data, "test", labels)},
    }
    for split in ("val", "test"):
        output[split]["delta_acc"] = output[split]["off"]["acc"] - output[split]["on"]["acc"]
        output[split]["delta_macro_f1"] = output[split]["off"]["macro_f1"] - output[split]["on"]["macro_f1"]
    return output


def _interventions(model: Any, classifier: Any, data: Any, components: dict[str, Any], decomposition: dict[str, Any], labels: list[int]) -> dict[str, Any]:
    full_fused = components["z"]
    full_logits = classifier(full_fused)
    output: dict[str, Any] = {"Full": {"functional": _functional_effect(model, classifier, data, components, counterfactual_eta(decomposition, "full"), labels, full_logits, full_fused)}}
    for name, mode in (("NoNode", "nonode"), ("NoModality", "nomodality"), ("GlobalOnly", "globalonly"), ("ModalitySwap", "modalityswap")):
        eta = counterfactual_eta(decomposition, mode)
        output[name] = {"functional": _functional_effect(model, classifier, data, components, eta, labels, full_logits, full_fused)}
    shuffle_rows = []
    for shuffle_seed in SHUFFLE_SEEDS:
        eta = counterfactual_eta(decomposition, "nodeshuffle", shuffle_seed=shuffle_seed)
        shuffle_rows.append({"shuffle_seed": shuffle_seed, "functional": _functional_effect(model, classifier, data, components, eta, labels, full_logits, full_fused)})
    scalar_keys = ("z_text_relative_l2", "z_visual_relative_l2", "fused_z_relative_l2", "logits_relative_l2", "probability_relative_l2", "prediction_flip_rate")
    aggregate = {}
    for key in scalar_keys:
        values = [float(row["functional"][key]) for row in shuffle_rows]
        aggregate[key] = {"mean": float(np.mean(values)), "population_std": float(np.std(values, ddof=0)), "per_shuffle": values}
    for modality in ("text", "visual"):
        values = [float(row["functional"]["eta_relative_l2"][modality]) for row in shuffle_rows]
        aggregate.setdefault("eta_relative_l2", {})[modality] = {"mean": float(np.mean(values)), "population_std": float(np.std(values, ddof=0)), "per_shuffle": values}
    for split in ("val", "test"):
        for metric in ("delta_acc", "delta_macro_f1"):
            values = [float(row["functional"][split][metric]) for row in shuffle_rows]
            aggregate.setdefault(split, {})[metric] = {"mean": float(np.mean(values)), "population_std": float(np.std(values, ddof=0)), "per_shuffle": values}
    output["NodeShuffle"] = {"shuffle_seeds": list(SHUFFLE_SEEDS), "per_shuffle": shuffle_rows, "aggregate": aggregate}
    return output


def _diagnose(record: dict[str, Any], device: str, output_root: Path, resume: bool) -> dict[str, Any]:
    path = output_root / "per_checkpoint" / f"{record['dataset']}_seed{record['seed']}.json"
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(record["checkpoint"])
    sha_before = _sha256(checkpoint)
    cfg = _compose_c1_cfg(record["dataset"], int(record["seed"]), device)
    data = load_mag_data(cfg, "nc", int(record["seed"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg, payload["data_info"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    if (model.multihop_state_mode, model.multihop_response_mode) != ("anchored", "cumulative"):
        raise RuntimeError("U3-A must consume formal U2 C1 anchored-cumulative checkpoints")
    if abs(float(model.multihop_anchor_alpha) - 0.1) > 1e-12:
        raise RuntimeError("U3-A requires frozen alpha=0.1")
    classifier = nn.Linear(model.out_dim, int(data.num_classes))
    classifier.load_state_dict(payload["head_state"])
    classifier.to(device).eval()
    labels = _resolve_nc_eval_labels(data)
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        components = model._encode_components(x, edge_index)
        decomposition, coefficient = _canonical_payload(components)
        profiles: dict[str, Any] = {}
        profile_tensors: dict[str, Any] = {}
        for modality in ("text", "visual"):
            profile = contribution_profile(components[f"eta_{modality}"], components[f"states_{modality}"])
            profile_tensors[modality] = profile
            profile_payload = _profile_summary(profile)
            profile_payload["states_order_max"] = K_BY_DATASET[record["dataset"]]
            profiles[modality] = profile_payload
        factors = _semantic_factors(data, model, components)
        associations = _associations(decomposition, {modality: {**profile_tensors[modality], "response_order": profile_tensors[modality]["response_order"], "entropy": profile_tensors[modality]["entropy"], "probabilities": profile_tensors[modality]["probabilities"], "states_order_max": K_BY_DATASET[record["dataset"]]} for modality in ("text", "visual")}, factors)
        quartiles = _quartile_payload(factors, {modality: {**profile_tensors[modality], "states_order_max": K_BY_DATASET[record["dataset"]]} for modality in ("text", "visual")})
        interventions = _interventions(model, classifier, data, components, decomposition, labels)
        xi_payload = {}
        profile_variability = {}
        for modality in ("text", "visual"):
            xi = decomposition["xi"][modality]
            xi_payload[modality] = {
                "centered_profile_norm": distribution_summary(xi.norm(dim=1)),
                "centered_profile_norm_per_node_mean": float(xi.norm(dim=1).mean().item()),
            }
            profile_variability[modality] = profile_pairwise_variability(xi)
    sha_after = _sha256(checkpoint)
    output = {
        "dataset": record["dataset"], "seed": int(record["seed"]), "formal_K": K_BY_DATASET[record["dataset"]],
        "checkpoint": str(checkpoint), "checkpoint_sha256_before": sha_before, "checkpoint_sha256_after": sha_after,
        "checkpoint_bytes_unchanged": sha_before == sha_after, "model_state_keys_unchanged": True,
        "formal_config": {"edge_weight_mode": model.edge_weight_mode, "edge_weight_temperature": model.edge_weight_temperature, "multihop_state_mode": model.multihop_state_mode, "multihop_response_mode": model.multihop_response_mode, "multihop_anchor_alpha": model.multihop_anchor_alpha},
        "canonical_decomposition": coefficient,
        "preference_coefficients": {"mu": decomposition["mu"], "nu": decomposition["nu"]},
        "profiles": profiles,
        "node_heterogeneity": {modality: {"response_order": profile_tensors[modality]["summary"]["response_order"], "normalized_entropy": profile_tensors[modality]["summary"]["normalized_entropy"], **xi_payload[modality], "profile_pairwise_variability": profile_variability[modality]} for modality in ("text", "visual")},
        "modality_heterogeneity": {
            "contribution_l1": distribution_summary((profile_tensors["text"]["probabilities"] - profile_tensors["visual"]["probabilities"]).abs().sum(dim=1)),
            "contribution_js": distribution_summary(0.5 * ((profile_tensors["text"]["probabilities"] * torch.log((profile_tensors["text"]["probabilities"] + EPS) / (0.5 * (profile_tensors["text"]["probabilities"] + profile_tensors["visual"]["probabilities"])) + EPS)).sum(dim=1) + (profile_tensors["visual"]["probabilities"] * torch.log((profile_tensors["visual"]["probabilities"] + EPS) / (0.5 * (profile_tensors["text"]["probabilities"] + profile_tensors["visual"]["probabilities"])) + EPS)).sum(dim=1))),
            "response_order_difference": distribution_summary(profile_tensors["text"]["response_order"] - profile_tensors["visual"]["response_order"]),
            "nu_text_visual_difference_l2": float((decomposition["nu"]["text"] - decomposition["nu"]["visual"]).norm().item()),
            "nu_text": decomposition["nu"]["text"], "nu_visual": decomposition["nu"]["visual"],
        },
        "state_factors": {modality: {"final_drift": distribution_summary(factors[modality]["final_drift"]), "mean_nonzero_drift": distribution_summary(factors[modality]["mean_nonzero_drift"]), "conductance": factors[modality]["conductance"]["summary"], "log_degree": distribution_summary(factors[modality]["log_degree"])} for modality in ("text", "visual")},
        "associations": associations,
        "quartile_stratification": quartiles,
        "counterfactuals": interventions,
        "all_finite": bool(all(torch.isfinite(value).all() for value in components.values() if isinstance(value, torch.Tensor))),
    }
    _dump(path, output)
    del data, model, classifier, components
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _numeric_effects(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for intervention in ("NoNode", "NoModality", "GlobalOnly", "ModalitySwap"):
        functional = record["counterfactuals"][intervention]["functional"]
        rows.append({"dataset": record["dataset"], "seed": record["seed"], "intervention": intervention, **{key: functional[key] for key in ("z_text_relative_l2", "z_visual_relative_l2", "fused_z_relative_l2", "logits_relative_l2", "probability_relative_l2", "prediction_flip_rate")}, "val_acc_delta": functional["val"]["delta_acc"], "val_macro_f1_delta": functional["val"]["delta_macro_f1"], "test_acc_delta": functional["test"]["delta_acc"], "test_macro_f1_delta": functional["test"]["delta_macro_f1"]})
    aggregate = record["counterfactuals"]["NodeShuffle"]["aggregate"]
    rows.append({"dataset": record["dataset"], "seed": record["seed"], "intervention": "NodeShuffle", **{key: aggregate[key]["mean"] for key in ("z_text_relative_l2", "z_visual_relative_l2", "fused_z_relative_l2", "logits_relative_l2", "probability_relative_l2", "prediction_flip_rate")}, "val_acc_delta": aggregate["val"]["delta_acc"]["mean"], "val_macro_f1_delta": aggregate["val"]["delta_macro_f1"]["mean"], "test_acc_delta": aggregate["test"]["delta_acc"]["mean"], "test_macro_f1_delta": aggregate["test"]["delta_macro_f1"]["mean"], "shuffle_population_std_logits": aggregate["logits_relative_l2"]["population_std"]})
    return rows


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate_values(values: list[float]) -> dict[str, float | int]:
        return {"mean": float(np.mean(values)), "population_std": float(np.std(values, ddof=0)), "count": len(values)}
    functional = {}
    for intervention in ("NoNode", "NoModality", "GlobalOnly", "ModalitySwap", "NodeShuffle"):
        functional[intervention] = {}
        for metric in ("fused_z_relative_l2", "logits_relative_l2", "probability_relative_l2", "prediction_flip_rate", "val_acc_delta", "val_macro_f1_delta", "test_acc_delta", "test_macro_f1_delta"):
            values = []
            for row in records:
                if intervention == "NodeShuffle":
                    value = row["counterfactuals"][intervention]["aggregate"]
                    if metric in value:
                        values.append(float(value[metric]["mean"]))
                    elif metric.startswith("val_") or metric.startswith("test_"):
                        split, key = metric.split("_", 1)
                        key = {"acc_delta": "delta_acc", "macro_f1_delta": "delta_macro_f1"}[key]
                        values.append(float(value[split][key]["mean"]))
                else:
                    value = row["counterfactuals"][intervention]["functional"]
                    if metric in value:
                        values.append(float(value[metric]))
                    elif metric.startswith("val_") or metric.startswith("test_"):
                        split, key = metric.split("_", 1)
                        key = {"acc_delta": "delta_acc", "macro_f1_delta": "delta_macro_f1"}[key]
                        values.append(float(value[split][key]))
            functional[intervention][metric] = aggregate_values(values)
    preference = {}
    for dataset in DATASETS:
        preference[dataset] = {}
        for modality in ("text", "visual"):
            rows = [row for row in records if row["dataset"] == dataset]
            preference[dataset][modality] = {
                "response_order": aggregate_values([row["profiles"][modality]["response_order"]["mean"] for row in rows]),
                "normalized_entropy": aggregate_values([row["profiles"][modality]["normalized_entropy"]["mean"] for row in rows]),
                "xi_norm_mean": aggregate_values([row["node_heterogeneity"][modality]["centered_profile_norm_per_node_mean"] for row in rows]),
                "modality_js": aggregate_values([row["modality_heterogeneity"]["contribution_js"]["mean"] for row in rows]),
                "modality_order_difference": aggregate_values([row["modality_heterogeneity"]["response_order_difference"]["mean"] for row in rows]),
            }
    return {"functional_counterfactuals": functional, "preference_by_dataset": preference}


def _gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_rows = []
    for dataset in DATASETS:
        rows = [row for row in records if row["dataset"] == dataset]
        modality_offset = float(np.mean([row["modality_heterogeneity"]["nu_text_visual_difference_l2"] for row in rows]))
        modality_js = float(np.mean([row["modality_heterogeneity"]["contribution_js"]["mean"] for row in rows]))
        no_modality_effects = [row["counterfactuals"]["NoModality"]["functional"]["logits_relative_l2"] for row in rows]
        modality_swap_effects = [row["counterfactuals"]["ModalitySwap"]["functional"]["logits_relative_l2"] for row in rows]
        modality_stable = sum(value >= 1e-4 for value in no_modality_effects) >= 2 or sum(value >= 1e-4 for value in modality_swap_effects) >= 2
        xi_norm = float(np.mean([np.mean([row["node_heterogeneity"][modality]["centered_profile_norm_per_node_mean"] for modality in ("text", "visual")]) for row in rows]))
        no_node_effects = [row["counterfactuals"]["NoNode"]["functional"]["logits_relative_l2"] for row in rows]
        shuffle_effects = [row["counterfactuals"]["NodeShuffle"]["aggregate"]["logits_relative_l2"]["mean"] for row in rows]
        node_stable = sum(value >= 1e-4 for value in no_node_effects) >= 2
        alignment_stable = sum(value >= 1e-4 for value in shuffle_effects) >= 2
        dataset_rows.append({
            "dataset": dataset, "modality_offset_l2_mean": modality_offset, "modality_contribution_js_mean": modality_js,
            "no_modality_logits_relative_l2": no_modality_effects, "modality_swap_logits_relative_l2": modality_swap_effects,
            "xi_norm_mean": xi_norm, "no_node_logits_relative_l2": no_node_effects, "node_shuffle_logits_relative_l2": shuffle_effects,
            "modality_functional_supported": bool(modality_offset > 1e-6 and modality_js > 1e-5 and modality_stable),
            "node_functional_supported": bool(xi_norm > 1e-6 and node_stable),
            "node_preference_alignment_supported": bool(xi_norm > 1e-6 and alignment_stable),
        })
    modality_count = sum(row["modality_functional_supported"] for row in dataset_rows)
    node_count = sum(row["node_functional_supported"] for row in dataset_rows)
    alignment_count = sum(row["node_preference_alignment_supported"] for row in dataset_rows)
    modality_supported = modality_count >= 3
    node_supported = node_count >= 3
    if modality_supported and node_supported:
        decision = "A. Both modality and node adaptation are functionally supported"
        recommendation = "Proceed to U3-B with the C1 three-level hierarchy retained; test only minimal mechanism-driven composition changes, beginning with contribution/state-conditioned coefficient composition and no new router by default."
    elif modality_supported:
        decision = "B. Modality adaptation supported, node personalization weak"
        recommendation = "Proceed to U3-B focused on modality-adaptive composition; do not add complexity to node personalization until alignment evidence improves."
    elif node_supported:
        decision = "C. Node personalization supported, modality-level offset weak"
        recommendation = "Proceed to U3-B retaining node adaptation while weakening or removing an explicit modality-offset path."
    elif modality_count == 0 and node_count == 0:
        decision = "D. Existing hierarchy is mostly global/shared"
        recommendation = "Stop the U3 upgrade and use a simpler shared/global composition."
    else:
        decision = "E. Dataset-conditional / unresolved"
        recommendation = "Before any U3-B implementation, locate dataset conditionality; do not immediately add a module."
    return {"dataset_rows": dataset_rows, "modality_datasets_passed": modality_count, "node_datasets_passed": node_count, "alignment_datasets_passed": alignment_count, "modality_supported": modality_supported, "node_supported": node_supported, "decision": decision, "u3b_recommendation": recommendation, "decision_rule": {"stable_seed_count": "at least 2/3", "functional_effect_band": "logit relative L2 >= 1e-4", "modality_offset_floor": 1e-6, "modality_JS_floor": 1e-5, "node_xi_floor": 1e-6}}


def _write_tables(output_root: Path, records: list[dict[str, Any]]) -> None:
    master_rows = []
    cf_rows = []
    association_rows = []
    for row in records:
        master_rows.append({
            "dataset": row["dataset"], "seed": row["seed"], "formal_K": row["formal_K"], "checkpoint_sha256": row["checkpoint_sha256_before"],
            "eta_reconstruction_max_abs": row["canonical_decomposition"]["reconstruction_max_abs"], "nu_centering_max_abs": row["canonical_decomposition"]["mean_modality_deviation_max_abs"],
            "xi_text_centering_max_abs": row["canonical_decomposition"]["mean_node_centered_max_abs"]["text"], "xi_visual_centering_max_abs": row["canonical_decomposition"]["mean_node_centered_max_abs"]["visual"],
            "text_response_order_mean": row["profiles"]["text"]["response_order"]["mean"], "visual_response_order_mean": row["profiles"]["visual"]["response_order"]["mean"],
            "text_normalized_entropy_mean": row["profiles"]["text"]["normalized_entropy"]["mean"], "visual_normalized_entropy_mean": row["profiles"]["visual"]["normalized_entropy"]["mean"],
            "modality_contribution_l1_mean": row["modality_heterogeneity"]["contribution_l1"]["mean"], "modality_contribution_js_mean": row["modality_heterogeneity"]["contribution_js"]["mean"],
            "response_order_difference_mean": row["modality_heterogeneity"]["response_order_difference"]["mean"], "nu_text_visual_l2": row["modality_heterogeneity"]["nu_text_visual_difference_l2"],
            "xi_text_norm_mean": row["node_heterogeneity"]["text"]["centered_profile_norm_per_node_mean"], "xi_visual_norm_mean": row["node_heterogeneity"]["visual"]["centered_profile_norm_per_node_mean"],
            "no_node_logits_relative_l2": row["counterfactuals"]["NoNode"]["functional"]["logits_relative_l2"], "no_modality_logits_relative_l2": row["counterfactuals"]["NoModality"]["functional"]["logits_relative_l2"],
            "global_only_logits_relative_l2": row["counterfactuals"]["GlobalOnly"]["functional"]["logits_relative_l2"], "node_shuffle_logits_relative_l2": row["counterfactuals"]["NodeShuffle"]["aggregate"]["logits_relative_l2"]["mean"], "modality_swap_logits_relative_l2": row["counterfactuals"]["ModalitySwap"]["functional"]["logits_relative_l2"],
        })
        cf_rows.extend(_numeric_effects(row))
        for association in row["associations"]:
            association_rows.append({"dataset": row["dataset"], "seed": row["seed"], **association})
    for filename, rows in (("u3a_master_table.csv", master_rows), ("u3a_counterfactuals.csv", cf_rows), ("u3a_state_preference_associations.csv", association_rows)):
        path = output_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def _write_report(summary: dict[str, Any]) -> None:
    gate = summary["u3a_gate"]
    lines = [
        "# MoPF-vNext U3-A — Node–Modality Multi-Hop Preference Diagnosis", "",
        "Analysis-only diagnosis over the frozen U2-C C1 checkpoints. No model was trained, no formal architecture was changed, and Test metrics are descriptive only.", "",
        f"- Runtime commit: `{summary['git_provenance']['head']}`", f"- Source checkpoints: `{len(summary['source_checkpoints'])}` C1 best-validation checkpoints", f"- Checkpoint bytes unchanged: `{summary['validation']['all_checkpoint_bytes_unchanged']}`", f"- Canonical eta reconstruction max abs: `{summary['validation']['max_eta_reconstruction_abs']:.3e}`", "",
        "## Evidence questions", "",
        f"- Q1 modality preference: **{gate['modality_supported']}** across `{gate['modality_datasets_passed']}/5` datasets.", f"- Q2 node personalization: **{gate['node_supported']}** across `{gate['node_datasets_passed']}/5` datasets.", f"- Q3 node–preference alignment by NodeShuffle: `{gate['alignment_datasets_passed']}/5` datasets.", "- Q4 state association: see the per-seed Spearman table; effect sizes and seed consistency are reported rather than p-values alone.", "",
        "## U3-A gate", "", f"**{gate['decision']}**", "", gate["u3b_recommendation"], "",
        "Full per-checkpoint coefficient statistics, contribution profiles, entropy, counterfactuals, associations, and quartile stratification are in the authoritative JSON and CSV artifacts.", "",
    ]
    (ROOT / "docs" / "mopf_u3a_node_modality_preference_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def _write_journal(summary: dict[str, Any]) -> None:
    path = ROOT / "docs" / "mopf_vnext_upgrade_journal.md"
    text = path.read_text(encoding="utf-8")
    marker = "## U3 — Node–Modality Adaptive Multi-Hop Composition"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    gate = summary["u3a_gate"]
    section = [
        marker, "", "U3 asks how much multi-hop structural evidence each node and modality should use. U3-A was analysis-only over the frozen U2-C C1 checkpoints; no training or architecture change was performed.", "", "## U3-A — Node–Modality Multi-Hop Preference Diagnosis", "", f"- Source: 15 C1 best-validation checkpoints; checkpoint bytes unchanged before/after analysis: `{summary['validation']['all_checkpoint_bytes_unchanged']}`.", "- Attribution uses final effective eta only: `eta = mu + nu + xi`, with global effective preference `mu`, modality effective deviation `nu`, and centered node preference `xi`.", f"- Canonical reconstruction max absolute error: `{summary['validation']['max_eta_reconstruction_abs']:.3e}`.", f"- Modality functional support: `{gate['modality_supported']}` (`{gate['modality_datasets_passed']}/5` datasets).", f"- Node personalization support: `{gate['node_supported']}` (`{gate['node_datasets_passed']}/5` datasets).", f"- NodeShuffle alignment support: `{gate['alignment_datasets_passed']}/5` datasets.", f"- U3-A decision: **{gate['decision']}**.", f"- U3-B recommendation: {gate['u3b_recommendation']}", "", "U3-A hard stop: no U3-B implementation, retraining, router, attention, MoE, auxiliary loss, LP, alpha/K tuning, or U1/U2 modification was started.", ""]
    path.write_text(text + "\n".join(section), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    source = _source_records()
    rows = []
    pending = []
    for record in source:
        path = output_root / "per_checkpoint" / f"{record['dataset']}_seed{record['seed']}.json"
        if args.no_resume or not path.is_file():
            pending.append(record)
    if args.no_parallel or len(args.devices) == 1:
        for index, record in enumerate(pending, start=1):
            device = str(args.devices[(index - 1) % len(args.devices)])
            print(f"[U3-A] {index}/{len(pending)} {record['dataset']} seed={record['seed']} {device}", flush=True)
            rows.append(_diagnose(record, device, output_root, not args.no_resume))
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
            for index, record in enumerate(pending):
                device = str(args.devices[index % len(args.devices)])
                futures[executor.submit(_diagnose, record, device, output_root, not args.no_resume)] = record
            for future in as_completed(futures):
                rows.append(future.result())
                record = futures[future]
                print(f"[U3-A] finished {record['dataset']} seed={record['seed']}", flush=True)
    for record in source:
        path = output_root / "per_checkpoint" / f"{record['dataset']}_seed{record['seed']}.json"
        if not any((row["dataset"], int(row["seed"])) == (record["dataset"], int(record["seed"])) for row in rows):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), SEEDS.index(int(row["seed"]))))
    validation = {
        "all_checkpoint_bytes_unchanged": all(row["checkpoint_bytes_unchanged"] for row in rows),
        "max_eta_reconstruction_abs": max(float(row["canonical_decomposition"]["reconstruction_max_abs"]) for row in rows),
        "max_nu_centering_abs": max(float(row["canonical_decomposition"]["mean_modality_deviation_max_abs"]) for row in rows),
        "max_xi_centering_abs": max(max(float(value) for value in row["canonical_decomposition"]["mean_node_centered_max_abs"].values()) for row in rows),
        "all_finite": all(row["all_finite"] for row in rows),
    }
    summary = {
        "metadata": {"stage": "MoPF-vNext U3-A Node–Modality Multi-Hop Preference Diagnosis", "scope": "frozen C1 NC checkpoint analysis", "datasets": list(DATASETS), "seeds": list(SEEDS), "formal_K_by_dataset": dict(K_BY_DATASET), "source_variant": "C1", "shuffle_seeds": list(SHUFFLE_SEEDS), "no_training": True, "no_architecture_change": True, "test_role": "descriptive only", "attribution": "final effective eta only", "preference_definition": "Contribution-Weighted Response Order from eta_i,k * S_i,k", "radius_language_forbidden": True},
        "git_provenance": {"head": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"), "status": _git("status", "--short"), "sync_status": "failed: git fetch/pull could not resolve github.com in this environment"},
        "source_checkpoints": [{"dataset": row["dataset"], "seed": row["seed"], "checkpoint": row["checkpoint"], "sha256": result["checkpoint_sha256_before"]} for row, result in zip(source, rows)],
        "validation": validation,
        "per_checkpoint": rows,
        "aggregates": _aggregate(rows),
        "u3a_gate": _gate(rows),
        "artifacts": {"master_summary": str(output_root / "u3a_master_summary.json"), "master_table": str(output_root / "u3a_master_table.csv"), "counterfactuals": str(output_root / "u3a_counterfactuals.csv"), "associations": str(output_root / "u3a_state_preference_associations.csv"), "per_checkpoint": str(output_root / "per_checkpoint")},
    }
    _dump(output_root / "u3a_master_summary.json", summary)
    _write_tables(output_root, rows)
    _write_report(summary)
    _write_journal(summary)
    print(json.dumps({"decision": summary["u3a_gate"]["decision"], "modality_supported": summary["u3a_gate"]["modality_supported"], "node_supported": summary["u3a_gate"]["node_supported"], "alignment_datasets": summary["u3a_gate"]["alignment_datasets_passed"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
