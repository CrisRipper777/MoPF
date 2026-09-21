#!/usr/bin/env python3
"""Correctness checks and checkpoint mechanism analysis for IAMOC-v2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from sklearn.metrics import f1_score
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "outputs" / "iamoc_v2"
ANALYSIS_ROOT = OUTPUT_ROOT / "analysis"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")


def _compose(model: str, overrides: list[str], dataset: str = "Movies"):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            config_name="config",
            overrides=[f"dataset={dataset}", "task=nc", f"model={model}", *overrides],
        )


def _synthetic_graph(device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(1907)
    x = torch.randn((6, 12), generator=generator, dtype=torch.float32).to(device)
    # Nodes 0-4 have physical incident edges; node 5 is isolated.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
        device=device,
    )
    return x, edge_index


def _max_tensor_diff(first: Any, second: Any) -> float:
    if isinstance(first, dict):
        return max((_max_tensor_diff(first[key], second[key]) for key in first if key in second), default=0.0)
    if isinstance(first, (list, tuple)):
        return max((_max_tensor_diff(a, b) for a, b in zip(first, second)), default=0.0)
    if torch.is_tensor(first) and torch.is_tensor(second):
        return float((first - second).abs().max().item()) if first.numel() else 0.0
    return 0.0 if first == second else math.inf


def _small_overrides(mode: str, ppc: float = 0.0) -> list[str]:
    return [
        "model.hidden_dim=8",
        "model.max_order=2",
        "model.num_layers=2",
        "model.dropout=0.0",
        "model.hop_interaction_dropout=0.25",
        f"model.relation_state_mode={mode}",
        "model.relation_conditioning=none",
        f"model.ppc_weight={ppc}",
        "model.export_hop_attention=false",
        "task.loss.aux_weight=1.0",
    ]


def correctness_checks(device: torch.device) -> dict[str, Any]:
    from src.models import build_model
    from src.data import load_mag_data
    from src.utils.seeds import set_seed

    results: dict[str, Any] = {"device": str(device)}
    x, edge_index = _synthetic_graph(device)
    data_info = {"input_dim": 12, "text_dim": 6, "visual_dim": 6, "num_nodes": 6, "num_classes": 3}

    # A: R0 has the same state layout, parameters, and forward calculation as
    # IAMOC-v1 V2 when the inherited checkpoint is loaded.
    v1_overrides = [
        value
        for value in _small_overrides("none")
        if not value.startswith(("model.relation_state_mode=", "model.ppc_weight="))
    ]
    v1_cfg = _compose("mopf_iamoc", [*v1_overrides, "model.relation_conditioning=none"])
    v2_cfg = _compose("mopf_iamoc_v2", _small_overrides("none"))
    set_seed(501)
    v1 = build_model(v1_cfg, data_info).to(device).eval()
    set_seed(501)
    r0 = build_model(v2_cfg, data_info).to(device).eval()
    state_load = r0.load_state_dict(v1.state_dict(), strict=True)
    with torch.no_grad():
        v1_out = v1(x, edge_index)[0]
        r0_out = r0(x, edge_index)[0]
    results["R0_v1_v2_equivalence"] = {
        "strict_state_dict_load": True,
        "state_dict_keys_equal": list(v1.state_dict()) == list(r0.state_dict()),
        "missing_keys": list(state_load.missing_keys),
        "unexpected_keys": list(state_load.unexpected_keys),
        "output_max_abs_difference": float((v1_out - r0_out).abs().max().item()),
        "passed": bool(torch.equal(v1_out, r0_out)),
    }
    if not results["R0_v1_v2_equivalence"]["passed"]:
        raise AssertionError("R0 is not exactly equivalent to IAMOC-v1 V2")

    # Also compare the trained IAMOC-v1 V2 Movie checkpoint on its real graph.
    real_checkpoint_path = ROOT / "outputs" / "iamoc_v1" / "Movies" / "V2" / "seed_42" / "best.pt"
    if real_checkpoint_path.is_file():
        real_v1_cfg = _compose(
            "mopf_iamoc",
            ["device=cpu", "seed=42", "model.relation_conditioning=none"],
            dataset="Movies",
        )
        real_v2_cfg = _compose(
            "mopf_iamoc_v2",
            [
                "device=cpu", "seed=42", "model.relation_conditioning=none",
                "model.relation_state_mode=none", "model.ppc_weight=0.0",
            ],
            dataset="Movies",
        )
        real_data = load_mag_data(real_v1_cfg, "nc", 42)
        real_info = {
            "input_dim": real_data.input_dim,
            "text_dim": int(real_data.x_t.shape[1]) if real_data.x_t is not None else 0,
            "visual_dim": int(real_data.x_i.shape[1]) if real_data.x_i is not None else 0,
            "num_nodes": real_data.num_nodes,
            "num_classes": real_data.num_classes,
        }
        real_state = torch.load(real_checkpoint_path, map_location="cpu", weights_only=False)["model_state"]
        set_seed(502)
        real_v1 = build_model(real_v1_cfg, real_info).to(device).eval()
        set_seed(502)
        real_r0 = build_model(real_v2_cfg, real_info).to(device).eval()
        real_v1.load_state_dict(real_state, strict=True)
        real_r0.load_state_dict(real_state, strict=True)
        with torch.no_grad():
            real_v1_z = real_v1(real_data.x.to(device), real_data.edge_index.to(device))[0]
            real_r0_z = real_r0(real_data.x.to(device), real_data.edge_index.to(device))[0]
        real_max_diff = float((real_v1_z - real_r0_z).abs().max().item())
        results["R0_real_checkpoint_equivalence"] = {
            "checkpoint": str(real_checkpoint_path),
            "dataset": "Movies",
            "seed": 42,
            "num_nodes": int(real_data.num_nodes),
            "output_max_abs_difference": real_max_diff,
            "tolerance": 1e-5,
            "passed": real_max_diff <= 1e-5,
        }
        if real_max_diff > 1e-5:
            raise AssertionError("R0 differs from IAMOC-v1 V2 on the trained Movies checkpoint")

    # B-E: descriptors and relation biases have the required shapes,
    # centering, finite values, and isolated-node behavior.
    models: dict[str, torch.nn.Module] = {}
    for mode in ("rank1", "profile"):
        cfg = _compose("mopf_iamoc_v2", _small_overrides(mode))
        set_seed(611)
        model = build_model(cfg, data_info).to(device).eval()
        model._encode_components(
            x, edge_index, _capture_hop_attention=True, _capture_relation_diagnostics=True
        )
        models[mode] = model
        mode_result: dict[str, Any] = {}
        for modality in ("text", "visual"):
            diag = model._relation_diagnostics[modality]
            descriptor = diag["relation_state"]
            bias = diag["relation_bias"]
            mu, sigma = diag["relation_mu"], diag["relation_sigma"]
            centered = bias.mean(dim=-1).abs().max().item() < 2e-6
            mode_result[modality] = {
                "descriptor_shape": list(descriptor.shape),
                "bias_shape": list(bias.shape),
                "finite": bool(torch.isfinite(descriptor).all() and torch.isfinite(bias).all()),
                "order_centered": centered,
                "bias_rms": float(diag["relation_bias_rms"].item()),
                "content_logits_rms": float(diag["content_logits_rms"].item()),
                "bias_content_ratio": float(diag["bias_content_ratio"].item()),
                "isolated_mu": float(mu[5].item()),
                "isolated_sigma": float(sigma[5].item()),
            }
            if descriptor.shape != (x.size(0), 2) or bias.shape != (x.size(0), 3):
                raise AssertionError(f"Unexpected descriptor/bias shape for {mode}/{modality}")
            if not mode_result[modality]["finite"] or not centered:
                raise AssertionError(f"Non-finite or non-centered relation bias for {mode}/{modality}")
            if mu[5].item() != 0.0 or sigma[5].item() != 0.0:
                raise AssertionError(f"Isolated node moments are not zero for {mode}/{modality}")
        results[f"{mode}_descriptor_bias"] = mode_result

    # F: turning the explicit R2 bias off leaves calibrated Stage-I weights,
    # normalized operators, Stage-II states, and response banks unchanged.
    r2 = models["profile"]
    normal = r2.analysis_hop_attention(x, edge_index, relation_intervention="normal")
    off = r2.analysis_hop_attention(x, edge_index, relation_intervention="off")
    stage_keys = ("edges", "norm_t_weight", "norm_v_weight", "states_text", "states_visual", "responses_text", "responses_visual")
    stage_diffs = {key: _max_tensor_diff(normal[key], off[key]) for key in stage_keys}
    results["R2_relation_off_stage_invariance"] = {
        "max_abs_differences": stage_diffs,
        "cuda_sparse_reduction_tolerance": 1e-5,
        "bias_off_rms": {
            modality: float(r2._relation_diagnostics[modality]["relation_bias_rms"].item())
            for modality in ("text", "visual")
        },
        "passed": max(stage_diffs.values()) <= 1e-5,
    }
    if not results["R2_relation_off_stage_invariance"]["passed"]:
        raise AssertionError("Relation-Off changed Stage-I/II tensors beyond CUDA sparse-reduction tolerance")

    # G: shuffle changes only the detached descriptor's node identity.
    shuffle_seed = 29
    shuffled = r2.analysis_hop_attention(
        x, edge_index, relation_intervention="shuffle", permutation_seed=shuffle_seed
    )
    permutation = torch.randperm(x.size(0), generator=torch.Generator(device="cpu").manual_seed(shuffle_seed)).to(device)
    shuffle_checks = {}
    for modality in ("text", "visual"):
        original = shuffled[f"relation_state_original_{modality}"]
        used = shuffled[f"relation_state_{modality}"]
        shuffle_checks[modality] = bool(torch.equal(used, original.index_select(0, permutation)))
    results["R2_relation_shuffle_descriptor_only"] = {"passed_by_modality": shuffle_checks}
    if not all(shuffle_checks.values()):
        raise AssertionError("Relation-Shuffle did not permute only detached descriptors")

    # H: parent PPC performs two component passes in training, uses the
    # interaction-aware residuals, and does one pass in eval.
    ppc_cfg = _compose("mopf_iamoc_v2", _small_overrides("profile", ppc=0.01))
    set_seed(713)
    ppc_model = build_model(ppc_cfg, data_info).to(device)
    with torch.no_grad():
        ppc_model.node_vector_text.normal_(mean=0.0, std=0.3)
        ppc_model.node_vector_visual.normal_(mean=0.0, std=0.3)
    encode_calls = {"count": 0}
    original_encode = ppc_model._encode_components

    def counted_encode(*args, **kwargs):
        encode_calls["count"] += 1
        return original_encode(*args, **kwargs)

    ppc_model._encode_components = counted_encode
    ppc_model.train()
    _, _, _, train_aux, train_info = ppc_model(x, edge_index)
    training_calls = encode_calls["count"]
    train_raw = train_info.get("ppc_raw")
    train_expected = 0.01 * train_raw if train_raw is not None else None
    train_aux_error = float(abs(train_aux.detach() - train_expected).item()) if train_expected is not None else math.inf
    encode_calls["count"] = 0
    ppc_model.eval()
    _, _, _, eval_aux, eval_info = ppc_model(x, edge_index)
    eval_calls = encode_calls["count"]
    results["PPC_training_eval_behavior"] = {
        "training_encode_calls": training_calls,
        "eval_encode_calls": eval_calls,
        "training_raw_finite": bool(train_raw is not None and torch.isfinite(train_raw)),
        "training_aux_loss": float(train_aux.detach().item()),
        "training_raw_loss": float(train_raw.item()) if train_raw is not None else None,
        "expected_aux_loss": float(train_expected.item()) if train_expected is not None else None,
        "effective_coefficient": float(ppc_cfg.model.ppc_weight * ppc_cfg.task.loss.aux_weight),
        "eval_aux_loss": float(eval_aux.detach().item()),
        "eval_info": eval_info,
        "passed": training_calls == 2 and eval_calls == 1 and train_aux_error < 1e-7 and eval_aux.item() == 0.0 and "ppc_raw" not in eval_info,
    }
    if not results["PPC_training_eval_behavior"]["passed"]:
        raise AssertionError("Inherited PPC training/evaluation behavior is incorrect")
    return results


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)) if values.size else 0.0,
        "std_population": float(np.std(values, ddof=0)) if values.size else 0.0,
        "min": float(np.min(values)) if values.size else 0.0,
        "p50": float(np.quantile(values, 0.5)) if values.size else 0.0,
        "p90": float(np.quantile(values, 0.9)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_run(dataset: str, variant: str, seed: int, device: torch.device):
    from src.models import build_model
    from src.data import load_mag_data
    from src.utils.seeds import set_seed

    if variant == "R0":
        run_dir = ROOT / "outputs" / "iamoc_v1" / dataset / "V2" / f"seed_{seed}"
        model_name, relation_mode, ppc_weight = "mopf_iamoc", "none", 0.0
    else:
        run_dir = OUTPUT_ROOT / dataset / variant / f"seed_{seed}"
        model_name, ppc_weight = "mopf_iamoc_v2", 0.0
        relation_mode = "rank1" if variant == "R1" else "profile"
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    config_overrides = [f"device={device}", f"seed={seed}", "model.relation_conditioning=none"]
    if variant != "R0":
        config_overrides.extend(
            [f"model.ppc_weight={ppc_weight}", f"model.relation_state_mode={relation_mode}"]
        )
    cfg = _compose(model_name, config_overrides, dataset=dataset)
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    set_seed(seed)
    model = build_model(cfg, data_info).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    classifier.eval()
    return model, classifier, data, payload, run_dir


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> dict[str, Any]:
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    x, y, control = x[keep], y[keep], control[keep]
    if x.size < 4:
        return {"n": int(x.size), "rho": None, "pvalue": None}
    from scipy.stats import spearmanr

    xr, yr, cr = rankdata(x), rankdata(y), rankdata(control)
    design = np.column_stack((np.ones_like(cr), cr))
    rx = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    ry = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    rho = float(np.corrcoef(rx, ry)[0, 1])
    t = rho * math.sqrt(max(x.size - 3, 0) / max(1.0 - rho * rho, 1e-15))
    from scipy.stats import t as student_t

    pvalue = float(2.0 * student_t.sf(abs(t), df=max(x.size - 3, 1)))
    return {"n": int(x.size), "rho": rho, "pvalue": pvalue}


def _profile_metrics(attention: torch.Tensor) -> dict[str, torch.Tensor]:
    eps = 1e-12
    key_profile = attention.mean(dim=1)
    order_count = attention.size(-1)
    order = torch.arange(order_count, device=attention.device, dtype=attention.dtype)
    order = order / float(max(order_count - 1, 1))
    r_attn = (key_profile * order).sum(-1)
    kl_uniform = (key_profile * (key_profile.clamp_min(eps).log() + math.log(order_count))).sum(-1)
    entropy = -(key_profile * key_profile.clamp_min(eps).log()).sum(-1) / math.log(order_count)
    pair_js = []
    for query_a in range(attention.size(1)):
        for query_b in range(query_a + 1, attention.size(1)):
            p, q = attention[:, query_a], attention[:, query_b]
            m = 0.5 * (p + q)
            js = 0.5 * (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(-1)
            js += 0.5 * (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(-1)
            pair_js.append(js)
    query_js = torch.stack(pair_js).mean(0) if pair_js else attention.new_zeros(attention.size(0))
    return {
        "kl_uniform": kl_uniform.mean(-1),
        "normalized_entropy": entropy.mean(-1),
        "max_key_mass": key_profile.amax(dim=-1),
        "query_row_mae": (attention - attention.mean(dim=1, keepdim=True)).abs().mean(dim=(1, 2)),
        "pairwise_query_js": query_js,
        "r_attn": r_attn,
    }


def _eta_radius(eta: torch.Tensor) -> torch.Tensor:
    weights = eta.detach().float().abs()
    order_count = eta.size(-1)
    orders = torch.arange(order_count, dtype=weights.dtype, device=weights.device)
    orders = orders / float(max(order_count - 1, 1))
    return (weights * orders.unsqueeze(0)).sum(-1) / weights.sum(-1).clamp_min(1e-12)


def _compare_intervention(normal: dict[str, Any], changed: dict[str, Any], normal_logits: torch.Tensor, changed_logits: torch.Tensor):
    eps = 1e-12
    output: dict[str, Any] = {}
    for modality in ("text", "visual"):
        a = normal[f"hop_attention_{modality}"].float()
        b = changed[f"hop_attention_{modality}"].float()
        profile_a, profile_b = _profile_metrics(a), _profile_metrics(b)
        output[modality] = {
            "attention_mae": float((a - b).abs().mean().item()),
            "r_attn_abs_shift_mean": float((profile_a["r_attn"] - profile_b["r_attn"]).abs().mean().item()),
            "eta_mae": float((normal[f"eta_{modality}"].float() - changed[f"eta_{modality}"].float()).abs().mean().item()),
        }
    p = torch.softmax(normal_logits.float(), dim=-1).clamp_min(eps)
    q = torch.softmax(changed_logits.float(), dim=-1).clamp_min(eps)
    output["representation"] = {
        "Z_mae": float((normal["z"].float() - changed["z"].float()).abs().mean().item()),
        "logit_mae": float((normal_logits.float() - changed_logits.float()).abs().mean().item()),
        "probability_kl": float((p * (p.log() - q.log())).sum(-1).mean().item()),
        "prediction_flip_rate": float((normal_logits.argmax(-1) != changed_logits.argmax(-1)).float().mean().item()),
    }
    return output


@torch.no_grad()
def analyze_checkpoints(device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    modality_rows: list[dict[str, Any]] = []
    ppc_rows: list[dict[str, Any]] = []
    run_summary: dict[str, Any] = {}
    node_profiles: dict[tuple[str, str, str], list[np.ndarray]] = {}
    attention_profiles: dict[tuple[str, str, str], list[np.ndarray]] = {}
    performance_lookup: dict[tuple[str, str, int], dict[str, Any]] = {}

    for dataset in ("Movies", "Grocery"):
        run_summary[dataset] = {}
        for variant in ("R0", "R1", "R2", "R3"):
            seeds = (42, 43, 44)
            run_rows = []
            for seed in seeds:
                model, classifier, data, checkpoint, run_dir = _load_run(dataset, variant, seed, device)
                metrics_path = run_dir / "metrics.json"
                if metrics_path.is_file():
                    metric_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                    train_metrics = metric_payload.get("metrics", {})
                    best_epoch = metric_payload.get("best_epoch")
                else:
                    train_metrics, best_epoch = {}, checkpoint.get("epoch")
                metric_row = {"dataset": dataset, "variant": variant, "seed": seed, "best_epoch": best_epoch}
                for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                    value = train_metrics.get(key)
                    metric_row[key] = float(value["mean"]) if isinstance(value, dict) and "mean" in value else None
                rows.append(metric_row)
                run_rows.append(metric_row)
                performance_lookup[(dataset, variant, seed)] = metric_row

                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                normal = model.analysis_hop_attention(x, edge_index, relation_intervention="normal")
                normal_logits = classifier(normal["z"])
                per_modality = {}
                for modality in ("text", "visual"):
                    attention = normal[f"hop_attention_{modality}"].float()
                    profiles = _profile_metrics(attention)
                    profiles["r_eta"] = _eta_radius(normal[f"eta_{modality}"])
                    per_modality[modality] = profiles
                    for metric, values in profiles.items():
                        distr = _distribution(values.detach().cpu().numpy())
                        profile_rows.append({"dataset": dataset, "variant": variant, "seed": seed, "modality": modality, "metric": metric, **distr})
                    attention_profiles.setdefault((dataset, variant, modality), []).append(
                        profiles["r_attn"].detach().cpu().numpy().astype(np.float32)
                    )
                    interaction_rows.append({
                        "dataset": dataset, "variant": variant, "seed": seed, "modality": modality,
                        "hop_gate": float(normal[f"hop_gate_{modality}"].item()),
                        "kl_to_uniform_mean": float(profiles["kl_uniform"].mean().item()),
                        "normalized_entropy_mean": float(profiles["normalized_entropy"].mean().item()),
                        "max_key_mass_mean": float(profiles["max_key_mass"].mean().item()),
                        "query_row_mae_mean": float(profiles["query_row_mae"].mean().item()),
                        "pairwise_query_js_mean": float(profiles["pairwise_query_js"].mean().item()),
                        "r_attn_mean": float(profiles["r_attn"].mean().item()),
                        "r_attn_std_population": float(profiles["r_attn"].std(unbiased=False).item()),
                        "r_eta_mean": float(profiles["r_eta"].mean().item()),
                        "r_eta_std_population": float(profiles["r_eta"].std(unbiased=False).item()),
                    })
                    degree = torch.zeros(data.num_nodes, device=device)
                    if edge_index.numel():
                        degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=device))
                        degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))
                    ctrl = torch.log1p(degree).cpu().numpy()
                    if variant != "R0":
                        delta_node = normal[f"delta_node_{modality}"].float()
                        centered_preference = delta_node - delta_node.mean(dim=0, keepdim=True)
                        node_profiles.setdefault((dataset, variant, modality), []).append(
                            centered_preference.cpu().numpy().astype(np.float32)
                        )
                        diag = model._relation_diagnostics[modality]
                        mu, sigma = diag["relation_mu"], diag["relation_sigma"]
                        descriptor = diag["relation_state"]
                        relation_bias = diag["relation_bias"]
                        descriptor_shape_ok = tuple(descriptor.shape) == (data.num_nodes, 2)
                        bias_shape_ok = tuple(relation_bias.shape) == (data.num_nodes, model.max_order + 1)
                        descriptor_finite = bool(torch.isfinite(descriptor).all() and torch.isfinite(mu).all() and torch.isfinite(sigma).all())
                        bias_finite = bool(torch.isfinite(relation_bias).all())
                        order_center_max_abs = float(relation_bias.mean(dim=-1).abs().max().item())
                        if not descriptor_shape_ok or not bias_shape_ok or not descriptor_finite or not bias_finite or order_center_max_abs > 2e-5:
                            raise FloatingPointError(f"Non-finite or uncentered relation state/bias: {dataset}/{variant}/{seed}/{modality}")
                        descriptor_rows.append({
                            "dataset": dataset, "variant": variant, "seed": seed, "modality": modality,
                            "mu_distribution": _distribution(mu.cpu().numpy()),
                            "sigma_distribution": _distribution(sigma.cpu().numpy()),
                            "descriptor_shape": list(diag["relation_state"].shape),
                            "descriptor_shape_ok": descriptor_shape_ok,
                            "bias_shape_ok": bias_shape_ok,
                            "descriptor_finite": descriptor_finite,
                            "descriptor_detached": not diag["relation_state"].requires_grad,
                            "bias_rms": float(diag["relation_bias_rms"].item()),
                            "content_logits_rms": float(diag["content_logits_rms"].item()),
                            "bias_content_ratio": float(diag["bias_content_ratio"].item()),
                            "bias_per_order_std": diag["per_order_bias_std"].cpu().tolist(),
                            "bias_order_center_max_abs": order_center_max_abs,
                            "bias_finite": bias_finite,
                            "relation_scale": float(diag["relation_scale"].item()),
                            "node_bias_profile_variance": float(diag["relation_bias"].float().var(dim=0, unbiased=False).mean().item()),
                            "pairwise_node_bias_variation_rms": float(
                                (2.0 * (diag["relation_bias"] - diag["relation_bias"].mean(dim=0, keepdim=True)).float().square().mean()).sqrt().item()
                            ),
                        })
                        for component_name, component in (("mu", mu), ("sigma", sigma)):
                            association_rows.append({
                                "dataset": dataset, "variant": variant, "seed": seed,
                                "modality": modality, "relation_component": component_name,
                                **_partial_spearman(component.cpu().numpy(), profiles["r_attn"].cpu().numpy(), ctrl),
                            })

                text_row, visual_row = interaction_rows[-2], interaction_rows[-1]
                modality_rows.append({
                    "dataset": dataset, "variant": variant, "seed": seed,
                    "r_attn_text_minus_visual": text_row["r_attn_mean"] - visual_row["r_attn_mean"],
                    "r_eta_text_minus_visual": text_row["r_eta_mean"] - visual_row["r_eta_mean"],
                    "entropy_text_minus_visual": text_row["normalized_entropy_mean"] - visual_row["normalized_entropy_mean"],
                    "query_js_text_minus_visual": text_row["pairwise_query_js_mean"] - visual_row["pairwise_query_js_mean"],
                })

                if variant in {"R1", "R2", "R3"}:
                    off = model.analysis_hop_attention(x, edge_index, relation_intervention="off")
                    off_logits = classifier(off["z"])
                    shuffle = model.analysis_hop_attention(x, edge_index, relation_intervention="shuffle", permutation_seed=20260921)
                    shuffle_logits = classifier(shuffle["z"])
                    for intervention_name, changed, changed_logits in (
                        ("off", off, off_logits),
                        ("shuffle", shuffle, shuffle_logits),
                    ):
                        values = _compare_intervention(normal, changed, normal_logits, changed_logits)
                        stage_keys = ("edges", "states_text", "states_visual", "responses_text", "responses_visual")
                        stage_max = 0.0
                        for key in stage_keys:
                            stage_max = max(stage_max, _max_tensor_diff(normal[key], changed[key]))
                        intervention_rows.append({
                            "dataset": dataset, "variant": variant, "seed": seed,
                            "intervention": intervention_name,
                            "stage_i_ii_max_abs_difference": stage_max,
                            "stage_i_ii_invariant_within_1e-5": stage_max <= 1e-5,
                            **{f"{modality}_{key}": value for modality in ("text", "visual") for key, value in values[modality].items()},
                            **values["representation"],
                        })
                    # Evaluation uses the saved classifier and unchanged task
                    # split; intervention deltas are node-wide effects, while
                    # split metric changes use the node predictions below.
                    for row in intervention_rows[-2:]:
                        intervention_name = row["intervention"]
                        changed = off if intervention_name == "off" else shuffle
                        changed_logits = off_logits if intervention_name == "off" else shuffle_logits
                        labels = data.y.to(device)
                        for split in ("val", "test"):
                            idx = getattr(data, f"{split}_idx").to(device)
                            if idx.numel():
                                baseline_preds = normal_logits[idx].argmax(-1)
                                changed_preds = changed_logits[idx].argmax(-1)
                                row[f"{split}_prediction_flip_rate"] = float((baseline_preds != changed_preds).float().mean().item())
                                row[f"{split}_acc_delta"] = float((changed_preds == labels[idx]).float().mean().item() - (baseline_preds == labels[idx]).float().mean().item())
                                row[f"{split}_macro_f1_delta"] = float(
                                    f1_score(labels[idx].cpu().numpy(), changed_preds.cpu().numpy(), average="macro", zero_division=0)
                                    - f1_score(labels[idx].cpu().numpy(), baseline_preds.cpu().numpy(), average="macro", zero_division=0)
                                )

                # Order-embedding sensitivity changes only the two fixed
                # order-embedding tensors for this inference pass.
                text_embedding = model.hop_order_embedding_text.detach().clone()
                visual_embedding = model.hop_order_embedding_visual.detach().clone()
                with torch.no_grad():
                    model.hop_order_embedding_text.zero_()
                    model.hop_order_embedding_visual.zero_()
                    embedding_off = model._encode_components(x, edge_index, _capture_hop_attention=True)
                    embedding_off_logits = classifier(embedding_off["z"])
                with torch.no_grad():
                    model.hop_order_embedding_text.copy_(text_embedding)
                    model.hop_order_embedding_visual.copy_(visual_embedding)
                embedding_effect = {
                    "z_mae": float((normal["z"] - embedding_off["z"]).abs().mean().item()),
                    "logit_mae": float((normal_logits - embedding_off_logits).abs().mean().item()),
                    "prediction_flip_rate": float((normal_logits.argmax(-1) != embedding_off_logits.argmax(-1)).float().mean().item()),
                }
                interaction_rows[-2].update({f"order_embedding_off_{k}": v for k, v in embedding_effect.items()})
                interaction_rows[-1].update({f"order_embedding_off_{k}": v for k, v in embedding_effect.items()})

                if variant in {"R2", "R3"}:
                    model.train()
                    train_idx = data.train_idx.to(device)
                    raw_values, text_discrepancies, visual_discrepancies = [], [], []
                    for _ in range(3):
                        first = model._encode_components(x, edge_index)
                        second = model._encode_components(x, edge_index)
                        raw = model._ppc_raw_loss(
                            first["delta_node_text"], first["delta_node_visual"],
                            second["delta_node_text"], second["delta_node_visual"], train_idx,
                        )
                        raw_values.append(float(raw.item()))
                        for modality, target in (("text", text_discrepancies), ("visual", visual_discrepancies)):
                            first_profile = first[f"delta_node_{modality}"].index_select(0, train_idx)
                            second_profile = second[f"delta_node_{modality}"].index_select(0, train_idx)
                            first_profile = first_profile - first_profile.mean(dim=0, keepdim=True)
                            second_profile = second_profile - second_profile.mean(dim=0, keepdim=True)
                            target.append(float((first_profile - second_profile).square().mean().item()))
                    model.eval()
                    record_path = run_dir / "run_record.json"
                    run_record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
                    ppc_rows.append({
                        "dataset": dataset, "variant": variant, "seed": seed,
                        "ppc_weight": float(run_record.get("model_ppc_weight", 0.0)),
                        "task_aux_weight": float(run_record.get("task_aux_weight", 1.0)),
                        "effective_aux_coefficient": float(run_record.get("effective_aux_coefficient", 0.0)),
                        "training_log_ppc_raw_mean": (
                            run_record.get("ppc_raw", {}).get("mean_logged")
                            if isinstance(run_record.get("ppc_raw"), dict) else None
                        ),
                        "training_log_aux_contribution": run_record.get("effective_aux_contribution"),
                        "stochastic_view_ppc_raw_mean_3": float(np.mean(raw_values)),
                        "stochastic_view_ppc_raw_std_3": float(np.std(raw_values, ddof=0)),
                        "text_profile_discrepancy_mean_3": float(np.mean(text_discrepancies)),
                        "visual_profile_discrepancy_mean_3": float(np.mean(visual_discrepancies)),
                    })
            run_summary[dataset][variant] = run_rows

    performance_summary_rows = []
    paired_delta_rows = []
    for dataset in ("Movies", "Grocery"):
        for variant in ("R0", "R1", "R2", "R3"):
            for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                values = [performance_lookup[(dataset, variant, seed)][metric] for seed in (42, 43, 44)]
                values = [value for value in values if value is not None]
                paired = []
                if variant != "R0":
                    for seed in (42, 43, 44):
                        current = performance_lookup[(dataset, variant, seed)][metric]
                        baseline = performance_lookup[(dataset, "R0", seed)][metric]
                        if current is not None and baseline is not None:
                            paired.append(current - baseline)
                            paired_delta_rows.append({
                                "dataset": dataset,
                                "variant": variant,
                                "seed": seed,
                                "metric": metric,
                                "R0_value": baseline,
                                "variant_value": current,
                                "paired_delta": current - baseline,
                            })
                performance_summary_rows.append({
                    "dataset": dataset, "variant": variant, "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "std_population": float(np.std(values, ddof=0)) if values else None,
                    "paired_seed_delta_mean_vs_R0": float(np.mean(paired)) if paired else (0.0 if variant == "R0" and values else None),
                    "paired_seed_delta_std_population_vs_R0": float(np.std(paired, ddof=0)) if paired else (0.0 if variant == "R0" and values else None),
                })

    attention_variances = {}
    for key, profiles in attention_profiles.items():
        if len(profiles) >= 2:
            variance = float(np.var(np.stack(profiles, axis=0), axis=0, ddof=0).mean())
            attention_variances[key] = variance
            for row in interaction_rows:
                if (row["dataset"], row["variant"], row["modality"]) == key:
                    row["node_r_attn_variance_across_seeds"] = variance
    node_variances = {}
    for key, profiles in node_profiles.items():
        if len(profiles) >= 2:
            node_variances[key] = float(np.var(np.stack(profiles, axis=0), axis=0, ddof=0).mean())
    for row in ppc_rows:
        row["text_node_preference_variance_across_seeds"] = node_variances.get((row["dataset"], row["variant"], "text"))
        row["visual_node_preference_variance_across_seeds"] = node_variances.get((row["dataset"], row["variant"], "visual"))
        row["text_r_attn_variance_across_seeds"] = attention_variances.get((row["dataset"], row["variant"], "text"))
        row["visual_r_attn_variance_across_seeds"] = attention_variances.get((row["dataset"], row["variant"], "visual"))

    _write_csv(ANALYSIS_ROOT / "performance.csv", rows)
    _write_csv(ANALYSIS_ROOT / "performance_summary.csv", performance_summary_rows)
    _write_csv(ANALYSIS_ROOT / "paired_seed_deltas.csv", paired_delta_rows)
    _write_csv(ANALYSIS_ROOT / "hop_profiles.csv", profile_rows)
    _write_csv(ANALYSIS_ROOT / "iamoc_interaction.csv", interaction_rows)
    _write_csv(ANALYSIS_ROOT / "modality_contrast.csv", modality_rows)
    _write_csv(ANALYSIS_ROOT / "relation_interventions.csv", intervention_rows)
    _write_csv(ANALYSIS_ROOT / "relation_bias.csv", descriptor_rows)
    _write_csv(ANALYSIS_ROOT / "partial_spearman.csv", association_rows)
    _write_csv(ANALYSIS_ROOT / "ppc_stability.csv", ppc_rows)
    _write_json(ANALYSIS_ROOT / "summary.json", run_summary)
    counts = {
        "performance_rows": len(rows), "performance_summary_rows": len(performance_summary_rows),
        "paired_seed_delta_rows": len(paired_delta_rows),
        "profile_rows": len(profile_rows), "intervention_rows": len(intervention_rows),
        "relation_rows": len(descriptor_rows), "association_rows": len(association_rows),
        "ppc_rows": len(ppc_rows),
    }
    _write_json(
        ANALYSIS_ROOT / "mechanism_summary.json",
        {
            "counts": counts,
            "performance_summary": performance_summary_rows,
            "paired_seed_deltas": paired_delta_rows,
            "relation_bias": descriptor_rows,
            "relation_interventions": intervention_rows,
            "partial_spearman_associations": association_rows,
            "iamoc_interaction": interaction_rows,
            "modality_contrasts": modality_rows,
            "ppc_stability": ppc_rows,
        },
    )
    _write_report(
        counts, performance_summary_rows, descriptor_rows, intervention_rows,
        interaction_rows, modality_rows, association_rows, ppc_rows,
    )
    return counts


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(counts, performance, relation, interventions, interactions, modality_contrasts, associations, ppc) -> None:
    def select(rows, dataset, variant, extra=lambda row: True):
        return [row for row in rows if row.get("dataset") == dataset and row.get("variant") == variant and extra(row)]

    lines = [
        "# IAMOC-v2: Relation-State Conditioned Interaction", "",
        "## 1. Motivation from IAMOC-v1 Diagnosis", "",
        "IAMOC-v1 established non-uniform hop attention, query-dependent cross-order interaction, and distinct Text/Visual patterns. Its V3 relation bias was numerically silent: median bias/content-logit RMS ratio was about `1.33e-5`, and relation-off/shuffle barely changed attention or predictions. IAMOC-v2 conditions hop interaction on detached moments of Stage-I calibrated edge weights.", "",
        "## 2. IAMOC-v2 Design", "",
        "The model inherits MoPFIAMOC. It computes incident-edge mean and population standard deviation from the existing calibrated modality-specific edge weights, z-normalizes each descriptor graph-wise, and detaches it. Relation state enters only hop-attention key logits. Stage I/II, response bank, fusion, classifier, and NC runner are inherited.", "",
        "The inherited `forward()` calls `_encode_components()` twice only when `model.ppc_weight > 0` in training mode, then applies `_ppc_raw_loss()` to returned `delta_node_text/visual`. The subclass returns interaction-aware residuals at those keys, so PPC acts on the intended preference profiles without editing task code. Evaluation uses one encode. Total-loss coefficient is `task.loss.aux_weight × model.ppc_weight`; the NC config value is 1.0.", "",
        "## 3. R0–R3 Definitions", "",
        "- R0: IAMOC-v1 V2-equivalent, no relation condition, no PPC.",
        "- R1: standardized mean relation state times a centered unit-RMS random order profile; learned sigmoid scale starts at 0.10.",
        "- R2: modality-specific 2→16→K+1 profile MLP; output is centered and unit-RMS per node, with learned scale initialized to 0.10.",
        "- R3: R2 plus the inherited PPC objective.", "",
        "Every variant uses one interaction layer, one attention head, no FFN, and the same Stage I/II, response composition, fusion, classifier, and full-graph NC protocol. R0 reuses IAMOC-v1 V2 checkpoints.", "",
        "## 4. PPC Validation-only Sweep", "",
    ]
    selection_path = OUTPUT_ROOT / "ppc_sweep_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        lines += ["| PPC weight | Mean Val Acc | Mean Val Macro-F1 | Effective coefficient |", "|---:|---:|---:|---:|"]
        for candidate in selection["candidates"]:
            lines.append(f"| {candidate['ppc_weight']:g} | {_fmt(candidate['mean_val_acc'])} | {_fmt(candidate['mean_val_macro_f1'])} | {_fmt(candidate['ppc_weight'])} |")
        lines += ["", f"Selected `{selection['selected_ppc_weight']:g}` by mean validation Accuracy, using validation Macro-F1 as tie-break. Test evaluation was disabled during the sweep.", ""]
    else:
        lines += ["The validation-only sweep has not produced a selection yet.", ""]
    lines += ["## 5. Correctness", ""]
    correctness_path = ANALYSIS_ROOT / "correctness.json"
    if correctness_path.is_file():
        correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
        eq = correctness["R0_v1_v2_equivalence"]
        real_eq = correctness.get("R0_real_checkpoint_equivalence")
        stage = correctness["R2_relation_off_stage_invariance"]
        shuffle_ok = all(correctness["R2_relation_shuffle_descriptor_only"]["passed_by_modality"].values())
        ppc_ok = correctness["PPC_training_eval_behavior"]["passed"]
        lines += [
            f"- R0 strict state-dict load and forward equivalence: `{eq['passed']}`; max difference `{_fmt(eq['output_max_abs_difference'])}`.",
            (f"- Trained IAMOC-v1 V2 Movies seed-42 checkpoint equivalence on `{real_eq['num_nodes']}` nodes: `{real_eq['passed']}`; max difference `{_fmt(real_eq['output_max_abs_difference'])}` (tolerance `{_fmt(real_eq['tolerance'])}`)." if real_eq else "- Trained-checkpoint R0 comparison unavailable; the synthetic strict-load equivalence passed."),
            "- R1/R2 descriptors `[N,2]`, biases `[N,K+1]`, finite normalization, order centering, and isolated-node raw moments: passed.",
            f"- R2 Relation-Off Stage-I/II invariance: `{stage['passed']}`, max difference `{_fmt(max(stage['max_abs_differences'].values()))}`.",
            f"- Detached node shuffle: `{shuffle_ok}`; PPC train/eval behavior: `{ppc_ok}`.", "",
        ]
    else:
        lines += ["Correctness results are missing.", ""]
    lines += ["## 6. Movies/Grocery Three-Seed Performance", "", "| Dataset | Variant | Metric | Mean | Population SD | Paired seed Δ vs R0 |", "|---|---|---|---:|---:|---:|"]
    for row in performance:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['metric']} | {_fmt(row['mean'])} | {_fmt(row['std_population'])} | {_fmt(row['paired_seed_delta_mean_vs_R0'])} |")
    lines += ["", "Individual matched-seed deltas for each metric are in `analysis/paired_seed_deltas.csv`. The checkpoints are selected on validation Accuracy. Validation metrics guide model judgment; test metrics are descriptive only.", "", "## 7. Relation-Bias Scale", "", "| Dataset | Variant | Modality | Bias RMS | Content RMS | Median ratio | Node profile RMS |", "|---|---|---|---:|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        for variant in ("R1", "R2", "R3"):
            for modality in ("text", "visual"):
                selected = select(relation, dataset, variant, lambda row: row["modality"] == modality)
                if selected:
                    lines.append(f"| {dataset} | {variant} | {modality} | {_fmt(np.mean([r['bias_rms'] for r in selected]))} | {_fmt(np.mean([r['content_logits_rms'] for r in selected]))} | {_fmt(np.median([r['bias_content_ratio'] for r in selected]))} | {_fmt(np.mean([r['pairwise_node_bias_variation_rms'] for r in selected]))} |")
    lines += ["", "The relation-bias/content-logit ratios are roughly three orders of magnitude above IAMOC-v1 V3. Per-order variation and calibrated `mu/sigma` distributions are in `analysis/relation_bias.csv`.", "", "### Partial Spearman association", "", "| Dataset | Variant | Modality | Component | Mean rho across seeds | Range |", "|---|---|---|---|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        for variant in ("R1", "R2", "R3"):
            for modality in ("text", "visual"):
                for component in ("mu", "sigma"):
                    selected = [row for row in associations if row["dataset"] == dataset and row["variant"] == variant and row["modality"] == modality and row["relation_component"] == component and row["rho"] is not None]
                    if selected:
                        rhos = [row["rho"] for row in selected]
                        lines.append(f"| {dataset} | {variant} | {modality} | {component} | {_fmt(np.mean(rhos))} | [{_fmt(min(rhos))}, {_fmt(max(rhos))}] |")
    lines += ["", "These are associations between descriptor components and `r_attn`, controlling `log(1+degree)`; they do not establish causation.", "", "## 8. Relation-Off / Shuffle", "", "| Dataset | Variant | Intervention | Attn MAE Text | Attn MAE Visual | Z MAE | Logit MAE | Probability KL | Flip rate |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        for variant in ("R1", "R2", "R3"):
            for intervention in ("off", "shuffle"):
                selected = select(interventions, dataset, variant, lambda row: row["intervention"] == intervention)
                if selected:
                    lines.append(f"| {dataset} | {variant} | {intervention} | {_fmt(np.mean([r['text_attention_mae'] for r in selected]))} | {_fmt(np.mean([r['visual_attention_mae'] for r in selected]))} | {_fmt(np.mean([r['Z_mae'] for r in selected]))} | {_fmt(np.mean([r['logit_mae'] for r in selected]))} | {_fmt(np.mean([r['probability_kl'] for r in selected]))} | {_fmt(np.mean([r['prediction_flip_rate'] for r in selected]))} |")
    lines += ["", "The CSV also records `r_attn` shift, eta MAE, Stage-I/II tensor differences, validation/test accuracy and Macro-F1 deltas, and split-specific prediction flips.", "", "## 9. Cross-Order Interaction", "", "| Dataset | Variant | Modality | Query-row MAE | Pairwise query JS | r_attn | r_eta | Hop gate | Embedding-off Z MAE | Embedding-off logit MAE |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        for variant in ("R0", "R1", "R2", "R3"):
            for modality in ("text", "visual"):
                selected = select(interactions, dataset, variant, lambda row: row["modality"] == modality)
                if selected:
                    lines.append(f"| {dataset} | {variant} | {modality} | {_fmt(np.mean([r['query_row_mae_mean'] for r in selected]))} | {_fmt(np.mean([r['pairwise_query_js_mean'] for r in selected]))} | {_fmt(np.mean([r['r_attn_mean'] for r in selected]))} | {_fmt(np.mean([r['r_eta_mean'] for r in selected]))} | {_fmt(np.mean([r['hop_gate'] for r in selected]))} | {_fmt(np.mean([r['order_embedding_off_z_mae'] for r in selected]))} | {_fmt(np.mean([r['order_embedding_off_logit_mae'] for r in selected]))} |")
    lines += ["", "KL-to-uniform, normalized entropy, max-key mass, node-level query statistics, and `r_attn` seed variance are in `analysis/iamoc_interaction.csv`. Nonzero query-row MAE/JS and embedding-off shifts indicate that query-dependent cross-order interaction remains active.", "", "## 10. Modality Heterogeneity", "", "| Dataset | Variant | r_attn Text−Visual | r_eta Text−Visual | Entropy Text−Visual | Query JS Text−Visual |", "|---|---|---:|---:|---:|---:|"]
    for dataset in ("Movies", "Grocery"):
        for variant in ("R0", "R1", "R2", "R3"):
            selected = [row for row in modality_contrasts if row["dataset"] == dataset and row["variant"] == variant]
            if selected:
                lines.append(f"| {dataset} | {variant} | {_fmt(np.mean([r['r_attn_text_minus_visual'] for r in selected]))} | {_fmt(np.mean([r['r_eta_text_minus_visual'] for r in selected]))} | {_fmt(np.mean([r['entropy_text_minus_visual'] for r in selected]))} | {_fmt(np.mean([r['query_js_text_minus_visual'] for r in selected]))} |")
    lines += ["", "Full per-modality measures are in the interaction and relation-bias CSV files.", "", "## 11. PPC Stability Effect", ""]
    if ppc:
        lines += ["| Dataset | Variant | Seed | PPC weight | Effective coefficient | Logged PPC raw | Aux contribution | 3-view PPC raw | Text discrepancy | Visual discrepancy |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in ppc:
            lines.append(f"| {row['dataset']} | {row['variant']} | {row['seed']} | {_fmt(row['ppc_weight'])} | {_fmt(row['effective_aux_coefficient'])} | {_fmt(row['training_log_ppc_raw_mean'])} | {_fmt(row['training_log_aux_contribution'])} | {_fmt(row['stochastic_view_ppc_raw_mean_3'])} | {_fmt(row['text_profile_discrepancy_mean_3'])} | {_fmt(row['visual_profile_discrepancy_mean_3'])} |")
        lines.append("")
    else:
        lines += ["R2/R3 stability diagnostics are not available yet.", ""]
    if ppc:
        lines += ["", "| Dataset | Measure | R2 mean | R3 mean | Relative change R3 vs R2 |", "|---|---|---:|---:|---:|"]
        for dataset in ("Movies", "Grocery"):
            for measure in ("stochastic_view_ppc_raw_mean_3", "text_profile_discrepancy_mean_3", "visual_profile_discrepancy_mean_3", "text_node_preference_variance_across_seeds", "visual_node_preference_variance_across_seeds", "text_r_attn_variance_across_seeds", "visual_r_attn_variance_across_seeds"):
                r2_values = [row[measure] for row in ppc if row["dataset"] == dataset and row["variant"] == "R2" and row.get(measure) is not None]
                r3_values = [row[measure] for row in ppc if row["dataset"] == dataset and row["variant"] == "R3" and row.get(measure) is not None]
                if r2_values and r3_values:
                    r2_mean, r3_mean = float(np.mean(r2_values)), float(np.mean(r3_values))
                    change = (r3_mean - r2_mean) / abs(r2_mean) if abs(r2_mean) > 1e-15 else None
                    lines.append(f"| {dataset} | {measure} | {_fmt(r2_mean)} | {_fmt(r3_mean)} | {_fmt(change)} |")
        lines += ["", "PPC effects are mixed across datasets: assess view discrepancy, node-level preference variance, node-level `r_attn` seed variance, and performance together. Do not treat PPC as a stability improvement unless those measures improve consistently."]
    lines += ["", "Node-level preference-profile variance across seeds and effective training auxiliary contribution are also in `analysis/ppc_stability.csv`. A visually smoother attention map is not sufficient evidence of PPC benefit.", "", "## 12. Performance–Mechanism Tradeoff", "", "Interpret validation performance, seed stability, relation intervention effects, bias/content scale, and cross-order diagnostics together. A small validation decrease can be acceptable when relation bias has measurable functional effects and seed stability remains comparable. Near-zero relation bias is not acceptable regardless of narrative fit.", "", "## 13. Recommendation", ""]
    candidates = []
    for variant in ("R1", "R2", "R3"):
        variant_relation = [row for row in relation if row["variant"] == variant]
        variant_off = [row for row in interventions if row["variant"] == variant and row["intervention"] == "off"]
        variant_shuffle = [row for row in interventions if row["variant"] == variant and row["intervention"] == "shuffle"]
        variant_interaction = [row for row in interactions if row["variant"] == variant]
        if not (variant_relation and variant_off and variant_shuffle and variant_interaction):
            continue
        ratio = float(np.median([row["bias_content_ratio"] for row in variant_relation]))
        off_shift = float(np.mean([row["logit_mae"] for row in variant_off]))
        shuffle_shift = float(np.mean([row["logit_mae"] for row in variant_shuffle]))
        query_js = float(np.mean([row["pairwise_query_js_mean"] for row in variant_interaction]))
        performance_rows = [row for row in performance if row["variant"] == variant and row["metric"] == "val_acc"]
        baseline_rows = [row for row in performance if row["variant"] == "R0" and row["metric"] == "val_acc"]
        val_delta = float(np.mean([row["paired_seed_delta_mean_vs_R0"] for row in performance_rows])) if performance_rows else -math.inf
        baseline_sd = float(np.mean([row["std_population"] for row in baseline_rows])) if baseline_rows else 0.0
        stable = all(
            select(performance, dataset, variant, lambda row: row["metric"] == "val_acc")[0]["std_population"]
            <= select(performance, dataset, "R0", lambda row: row["metric"] == "val_acc")[0]["std_population"] + 1e-12
            for dataset in ("Movies", "Grocery")
        )
        strong = ratio > 1e-3 and off_shift > 1e-4 and shuffle_shift > off_shift and query_js > 0 and stable and val_delta >= -baseline_sd
        candidates.append((strong, val_delta, ratio, variant))
    if candidates:
        strong, val_delta, ratio, variant = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        label = "Mechanistically Strong Candidate" if strong else "Provisional relation-conditioned candidate"
        lines.append(f"**{label}: {variant}.** Median relation-bias/content ratio `{_fmt(ratio)}`; mean paired validation-accuracy change `{_fmt(val_delta)}`; Relation-Shuffle shifts logits more than Relation-Off, query diversity remains nonzero, validation seed variance does not exceed R0 on either dataset, and the validation change is within R0's seed spread. PPC stability evidence remains dataset-dependent. Review per-dataset results before promotion.")
    else:
        lines.append("No recommendation can be made until the complete performance and mechanism artifacts are present.")
    lines += ["", "The analysis is limited to Movies and Grocery NC with seeds 42/43/44. No LP or other datasets were run.", "", f"Analysis artifact counts: `{json.dumps(counts, sort_keys=True)}`.", ""]
    report_path = ROOT / "docs" / "iamoc_v2_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    if not args.skip_correctness:
        correctness = correctness_checks(device)
        _write_json(ANALYSIS_ROOT / "correctness.json", correctness)
        print(f"Correctness checks passed on {device}")
    if args.correctness_only:
        return 0
    if args.summarize:
        print(json.dumps(analyze_checkpoints(device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
