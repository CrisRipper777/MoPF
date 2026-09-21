#!/usr/bin/env python3
"""Audit RC-IAMOC R1 generalization and inference-only interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import types
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from scipy.stats import rankdata
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "outputs" / "rc_iamoc_final_candidate"
GENERALIZATION_ROOT = ROOT / "outputs" / "iamoc_nc_generalization"
LEGACY_ROOT = ROOT / "outputs" / "iamoc_v1"
V2_ROOT = ROOT / "outputs" / "iamoc_v2"
DATASETS = ("Movies", "Grocery", "Toys", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
DEVICE_DEFAULT = "cuda:1"
SHUFFLE_SEED = 73129


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=float) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose_cfg(model_name: str, dataset: str, seed: int, device: str):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                f"model={model_name}",
                f"seed={seed}",
                f"device={device}",
                "model.relation_conditioning=none",
                "model.relation_state_mode=rank1",
                "model.ppc_weight=0.0",
            ],
        )


def locate_source(dataset: str, variant: str, seed: int) -> Path:
    """Resolve a reused Movies/Grocery baseline through its audit manifest."""
    candidate = GENERALIZATION_ROOT / dataset / variant / f"seed_{seed}"
    if (candidate / "metrics.json").is_file():
        return candidate
    manifest_path = candidate / "reuse_manifest.json"
    if manifest_path.is_file():
        source = Path(json.loads(manifest_path.read_text(encoding="utf-8"))["reuse_audit"]["source_run_dir"])
        if (source / "metrics.json").is_file():
            return source
    return candidate


def load_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing per-seed metrics: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    values = payload.get("metrics", payload)
    output: dict[str, float] = {}
    if payload.get("best_epoch") is not None:
        output["best_epoch"] = int(payload["best_epoch"])
    for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        item = values.get(key)
        if isinstance(item, dict):
            output[key] = float(item["mean"])
        elif item is not None:
            output[key] = float(item)
    return output


def population_summary(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def performance_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    seed_rows: dict[tuple[str, str, int], dict[str, float]] = {}
    for dataset in DATASETS:
        for model_label in ("Formal V0", "IAMOC core V2/R0", "R1"):
            for seed in SEEDS:
                if model_label == "R1":
                    run_dir = (
                        V2_ROOT / dataset / "R1" / f"seed_{seed}"
                        if dataset in {"Movies", "Grocery"}
                        else OUTPUT_ROOT / dataset / "R1" / f"seed_{seed}"
                    )
                else:
                    variant = "V0" if model_label == "Formal V0" else "V2"
                    run_dir = locate_source(dataset, variant, seed)
                if not (run_dir / "best.pt").is_file():
                    raise FileNotFoundError(f"Missing comparison checkpoint for {dataset} {model_label} seed={seed}: {run_dir}")
                metrics = load_metrics(run_dir)
                row = {"dataset": dataset, "model": model_label, "seed": seed, "source_run_dir": str(run_dir), **metrics}
                raw.append(row)
                seed_rows[(dataset, model_label, seed)] = metrics
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                values = [seed_rows[(dataset, model_label, seed)][key] for seed in SEEDS]
                mean, std = population_summary(values)
                summaries.append(
                    {
                        "dataset": dataset,
                        "model": model_label,
                        "metric": key,
                        "mean": mean,
                        "std_population": std,
                        "formatted_mean_sd": f"{mean:.6f} ± {std:.6f}",
                        "n_seeds": len(values),
                    }
                )

    paired: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for reference in ("Formal V0", "IAMOC core V2/R0"):
        counts[reference] = {"positive": 0, "neutral": 0, "negative": 0, "cells": []}
        for dataset in DATASETS:
            for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                deltas = [
                    seed_rows[(dataset, "R1", seed)][metric]
                    - seed_rows[(dataset, reference, seed)][metric]
                    for seed in SEEDS
                ]
                mean, std = population_summary(deltas)
                baseline_sd = population_summary(
                    [seed_rows[(dataset, reference, seed)][metric] for seed in SEEDS]
                )[1]
                if mean > baseline_sd:
                    category = "positive"
                elif mean < -baseline_sd:
                    category = "negative"
                else:
                    category = "neutral"
                if metric.startswith("test_"):
                    counts[reference][category] += 1
                cell = {
                    "dataset": dataset,
                    "metric": metric,
                    "reference": reference,
                    "paired_delta_by_seed": {str(seed): deltas[index] for index, seed in enumerate(SEEDS)},
                    "mean_delta": mean,
                    "std_population_delta": std,
                    "baseline_std_population": baseline_sd,
                    "classification": category,
                }
                paired.append(cell)
                if metric.startswith("test_"):
                    counts[reference]["cells"].append({"dataset": dataset, "metric": metric, "classification": category})
    return raw, summaries, {"paired": paired, "counts": counts}


def normalized_profile(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = raw.float() - raw.float().mean()
    return centered / centered.square().mean().sqrt().clamp_min(eps)


def parameter_audit(model, dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"dataset": dataset, "seed": seed, "modalities": {}}
    for offset, modality in enumerate(MODALITIES):
        raw_initial = torch.randn(
            model.max_order + 1,
            generator=torch.Generator(device="cpu").manual_seed(model.relation_init_seed + offset),
            dtype=torch.float32,
        ) * model.relation_init_std
        raw_final = getattr(model, f"relation_beta_raw_{modality}").detach().cpu().float()
        init_profile = normalized_profile(raw_initial, model.relation_descriptor_eps)
        final_profile = normalized_profile(raw_final, model.relation_descriptor_eps)
        initial_norm = torch.linalg.vector_norm(init_profile).clamp_min(1e-12)
        final_norm = torch.linalg.vector_norm(final_profile).clamp_min(1e-12)
        profile_cosine = float(F.cosine_similarity(init_profile, final_profile, dim=0).item())
        effective_relative_change = float((torch.linalg.vector_norm(final_profile - init_profile) / initial_norm).item())
        raw_relative_change = float((torch.linalg.vector_norm(raw_final - raw_initial) / torch.linalg.vector_norm(raw_initial).clamp_min(1e-12)).item())
        scale_final = float(torch.sigmoid(getattr(model, f"theta_relation_scale_{modality}").detach()).cpu().item())
        summary["modalities"][modality] = {
            "initial_beta_raw": raw_initial.tolist(),
            "final_beta_raw": raw_final.tolist(),
            "initial_centered_normalized_profile": init_profile.tolist(),
            "final_centered_normalized_profile": final_profile.tolist(),
            "cosine_initial_final_profile": profile_cosine,
            "cosine_initial_final_raw": float(F.cosine_similarity(raw_initial, raw_final, dim=0).item()),
            "relative_l2_change_effective_profile": effective_relative_change,
            "relative_l2_change_raw": raw_relative_change,
            "relation_scale_initial": float(model.relation_bias_init),
            "relation_scale_final": scale_final,
            "theta_relation_scale_final": float(getattr(model, f"theta_relation_scale_{modality}").detach().cpu().item()),
        }
        for order, (initial, final, pi, pf) in enumerate(zip(raw_initial, raw_final, init_profile, final_profile)):
            profile_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "modality": modality,
                    "order": order,
                    "initial_beta_raw": float(initial.item()),
                    "final_beta_raw": float(final.item()),
                    "initial_centered_normalized_profile": float(pi.item()),
                    "final_centered_normalized_profile": float(pf.item()),
                }
            )
    return profile_rows, summary


def compact(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: output[key].detach()
        for key in (
            "z",
            "eta_text",
            "eta_visual",
            "hop_attention_text",
            "hop_attention_visual",
        )
    }


def run_hop_intervention(model, x: torch.Tensor, edge_index: torch.Tensor, intervention: str) -> dict[str, Any]:
    """Replace attention/gate math only for this eval forward, then restore method."""
    allowed = {"uniform_attention", "query_collapse", "interaction_off"}
    if intervention not in allowed:
        raise ValueError(intervention)

    def patched(
        self,
        states,
        modality,
        relation_context,
        beta_order,
        *,
        capture_attention,
        relation_bias_off,
        relation_context_override,
    ):
        del relation_context, beta_order, relation_context_override
        state_bank = torch.stack(states, dim=1)
        if modality == "text":
            embedding = self.hop_order_embedding_text
            layers = self.hop_layers_text
            gate = torch.tanh(self.theta_hop_gate_text)
        else:
            embedding = self.hop_order_embedding_visual
            layers = self.hop_layers_visual
            gate = torch.tanh(self.theta_hop_gate_visual)
        if intervention == "interaction_off":
            gate = torch.zeros_like(gate)

        descriptor = self._relation_state_used[modality].to(device=state_bank.device, dtype=state_bank.dtype).detach()
        bias = self._relation_bias(modality, descriptor).to(dtype=state_bank.dtype)
        bias_active = not (self._analysis_relation_bias_off or relation_bias_off)
        tokens = state_bank + embedding.unsqueeze(0) if self.hop_interaction_order_embedding else state_bank
        current = state_bank
        captured: list[torch.Tensor] = []
        for layer_index, layer in enumerate(layers):
            normalized = layer.norm(tokens)
            query = layer.query(normalized)
            key = layer.key(normalized)
            value = layer.value(normalized)
            logits = torch.matmul(query, key.transpose(-1, -2)) * layer.scale
            if bias_active:
                logits = logits + bias.unsqueeze(1)
            attention = torch.softmax(logits, dim=-1)
            if intervention == "uniform_attention":
                attention = torch.full_like(attention, 1.0 / float(attention.size(-1)))
            elif intervention == "query_collapse":
                key_mass = attention.mean(dim=1, keepdim=True)
                attention = key_mass.expand_as(attention)
            interaction = torch.matmul(layer.attention_dropout(attention), value)
            residual_base = state_bank if layer_index == 0 else current
            current = residual_base + gate * interaction
            if capture_attention:
                captured.append(attention)
            tokens = current
        last = captured[-1] if captured else state_bank.new_zeros((state_bank.size(0), state_bank.size(1), state_bank.size(1)))
        return [current[:, order, :] for order in range(current.size(1))], captured, last

    sentinel = object()
    previous = model.__dict__.get("_hop_interaction", sentinel)
    object.__setattr__(model, "_hop_interaction", types.MethodType(patched, model))
    try:
        return model._encode_components(x, edge_index, _capture_hop_attention=True)
    finally:
        if previous is sentinel:
            model.__dict__.pop("_hop_interaction", None)
        else:
            object.__setattr__(model, "_hop_interaction", previous)


def r_attn(attention: torch.Tensor) -> torch.Tensor:
    key_profile = attention.float().mean(dim=1)
    order = torch.arange(attention.size(-1), dtype=key_profile.dtype, device=key_profile.device)
    order = order / float(max(attention.size(-1) - 1, 1))
    return (key_profile * order).sum(dim=-1)


def query_metrics(attention: torch.Tensor) -> dict[str, float]:
    key_profile = attention.float().mean(dim=1)
    order_count = attention.size(-1)
    eps = 1e-12
    uniform_kl = (key_profile * (key_profile.clamp_min(eps).log() + math.log(order_count))).sum(-1)
    entropy = -(key_profile * key_profile.clamp_min(eps).log()).sum(-1) / math.log(order_count)
    pair_js: list[torch.Tensor] = []
    for first, second in combinations(range(attention.size(1)), 2):
        p, q = attention[:, first].float(), attention[:, second].float()
        mean = 0.5 * (p + q)
        js = 0.5 * (p * (p.clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(-1)
        js += 0.5 * (q * (q.clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(-1)
        pair_js.append(js)
    pair_js_mean = torch.stack(pair_js).mean(0) if pair_js else torch.zeros_like(uniform_kl)
    return {
        "kl_to_uniform_mean": float(uniform_kl.mean().item()),
        "normalized_entropy_mean": float(entropy.mean().item()),
        "max_key_mass_mean": float(key_profile.amax(dim=-1).mean().item()),
        "query_row_mae_mean": float((attention.float() - attention.float().mean(dim=1, keepdim=True)).abs().mean().item()),
        "pairwise_query_js_mean": float(pair_js_mean.mean().item()),
        "r_attn_mean": float(r_attn(attention).mean().item()),
        "r_attn_std_nodes": float(r_attn(attention).std(unbiased=False).item()),
        "attention_max_abs_deviation_from_uniform": float((key_profile - 1.0 / order_count).abs().max().item()),
    }


def resolve_r1_checkpoint(dataset: str, seed: int) -> Path:
    if dataset in {"Movies", "Grocery"}:
        return V2_ROOT / dataset / "R1" / f"seed_{seed}" / "best.pt"
    return OUTPUT_ROOT / dataset / "R1" / f"seed_{seed}" / "best.pt"


def load_r1(dataset: str, seed: int, device: torch.device):
    from src.data import load_mag_data
    from src.models import build_model
    from src.utils.seeds import set_seed

    checkpoint = resolve_r1_checkpoint(dataset, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"R1 checkpoint missing: {checkpoint}")
    cfg = compose_cfg("mopf_iamoc_v2", dataset, seed, str(device))
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
    }
    set_seed(seed)
    model = build_model(cfg, data_info).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    classifier.eval()
    return model, classifier, data, payload, checkpoint


def compare_outputs(
    reference: dict[str, torch.Tensor],
    changed: dict[str, torch.Tensor],
    classifier,
    data,
    eval_labels: list[int],
    *,
    dataset: str,
    seed: int,
    intervention: str,
    reference_name: str = "Normal",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eps = 1e-12
    metric_rows: list[dict[str, Any]] = []
    representation: dict[str, Any] = {}
    for modality in MODALITIES:
        a = reference[f"hop_attention_{modality}"].float()
        b = changed[f"hop_attention_{modality}"].float()
        profile_a = r_attn(a)
        profile_b = r_attn(b)
        item = {
            "dataset": dataset,
            "seed": seed,
            "reference": reference_name,
            "condition": intervention,
            "modality": modality,
            "metric": "attention_mae",
            "value": float((a - b).abs().mean().item()),
        }
        metric_rows.append(item)
        metric_rows.append(
            {
                **item,
                "metric": "r_attn_shift_signed",
                "value": float((profile_a - profile_b).mean().item()),
            }
        )
        metric_rows.append(
            {
                **item,
                "metric": "r_attn_shift_abs_mean",
                "value": float((profile_a - profile_b).abs().mean().item()),
            }
        )
        metric_rows.append(
            {
                **item,
                "metric": "eta_mae",
                "value": float((reference[f"eta_{modality}"].float() - changed[f"eta_{modality}"].float()).abs().mean().item()),
            }
        )
        representation[f"{modality}_attention_mae"] = item["value"]
        representation[f"{modality}_r_attn_shift_signed"] = float((profile_a - profile_b).mean().item())
        representation[f"{modality}_r_attn_shift_abs_mean"] = float((profile_a - profile_b).abs().mean().item())
        representation[f"{modality}_eta_mae"] = float(
            (reference[f"eta_{modality}"].float() - changed[f"eta_{modality}"].float()).abs().mean().item()
        )

    z_a, z_b = reference["z"].float(), changed["z"].float()
    logits_a, logits_b = classifier(z_a), classifier(z_b)
    prob_a = torch.softmax(logits_a, dim=-1).clamp_min(eps)
    prob_b = torch.softmax(logits_b, dim=-1).clamp_min(eps)
    all_metrics = {
        "Z_mae": float((z_a - z_b).abs().mean().item()),
        "logit_mae": float((logits_a - logits_b).abs().mean().item()),
        "probability_kl_reference_to_condition": float((prob_a * (prob_a.log() - prob_b.log())).sum(-1).mean().item()),
        "prediction_flip_rate_all_nodes": float((logits_a.argmax(-1) != logits_b.argmax(-1)).float().mean().item()),
    }
    test_idx = data.test_idx.to(z_a.device)
    y_test = data.y[test_idx.detach().cpu()].detach().cpu().numpy()
    pred_test = logits_b[test_idx].argmax(-1).detach().cpu().numpy()
    normal_pred_test = logits_a[test_idx].argmax(-1).detach().cpu().numpy()
    test_acc = float(np.mean(pred_test == y_test))
    test_f1 = float(f1_score(y_test, pred_test, labels=eval_labels, average="macro", zero_division=0))
    normal_acc = float(np.mean(normal_pred_test == y_test))
    normal_f1 = float(f1_score(y_test, normal_pred_test, labels=eval_labels, average="macro", zero_division=0))
    all_metrics.update(
        {
            "prediction_flip_rate_test": float(np.mean(pred_test != normal_pred_test)),
            "test_acc": test_acc,
            "test_macro_f1": test_f1,
            "test_acc_delta_vs_reference": test_acc - normal_acc,
            "test_macro_f1_delta_vs_reference": test_f1 - normal_f1,
            "reference_test_acc": normal_acc,
            "reference_test_macro_f1": normal_f1,
        }
    )
    representation.update(all_metrics)
    for key, value in all_metrics.items():
        metric_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "reference": reference_name,
                "condition": intervention,
                "modality": "fused",
                "metric": key,
                "value": value,
            }
        )
    return metric_rows, representation


@torch.no_grad()
def analyze_checkpoints(device: torch.device) -> dict[str, Any]:
    perf_raw, perf_summary, performance = performance_audit()
    write_csv(OUTPUT_ROOT / "performance" / "per_seed_metrics.csv", perf_raw)
    write_csv(OUTPUT_ROOT / "performance" / "mean_population_std.csv", perf_summary)
    write_csv(OUTPUT_ROOT / "performance" / "paired_seed_deltas.csv", performance["paired"])
    write_json(OUTPUT_ROOT / "performance" / "performance_audit.json", performance)

    beta_profile_rows: list[dict[str, Any]] = []
    beta_run_rows: list[dict[str, Any]] = []
    beta_vectors: dict[tuple[str, int, str], torch.Tensor] = {}
    heatmap_run_rows: list[dict[str, Any]] = []
    r_attn_rows: list[dict[str, Any]] = []
    relation_node_rows: list[dict[str, Any]] = []
    quartile_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    intervention_details: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    beta_json: list[dict[str, Any]] = []
    all_attention: dict[tuple[str, str], list[np.ndarray]] = {}
    n_orders: int | None = None

    from src.tasks.nc import _resolve_nc_eval_labels

    for dataset in DATASETS:
        for seed in SEEDS:
            model, classifier, data, payload, checkpoint = load_r1(dataset, seed, device)
            if payload.get("seed") != seed or payload.get("selection") != "best_val_accuracy":
                raise AssertionError(f"Unexpected best-checkpoint metadata: {checkpoint}")
            run_config_path = checkpoint.parent / "resolved_config.json"
            if dataset in {"Movies", "Grocery"}:
                run_config_path = checkpoint.parent / "resolved_config.json"
            if not run_config_path.is_file():
                raise FileNotFoundError(f"Missing frozen R1 resolved config: {run_config_path}")
            saved_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            model_config = saved_config["model"]
            frozen = {
                "hop_interaction_layers": 1,
                "hop_interaction_heads": 1,
                "hop_interaction_order_embedding": True,
                "relation_state_mode": "rank1",
                "relation_conditioning": "none",
                "relation_bias_init": 0.10,
                "ppc_weight": 0.0,
            }
            mismatches = {key: {"expected": value, "actual": model_config.get(key)} for key, value in frozen.items() if model_config.get(key) != value}
            if mismatches:
                raise AssertionError(f"R1 frozen configuration mismatch {dataset}/{seed}: {mismatches}")
            checkpoint_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "sha256": sha256(checkpoint),
                    "selection": payload.get("selection"),
                    "epoch": payload.get("epoch"),
                    "frozen_config_pass": True,
                    "num_nodes": int(data.num_nodes),
                    "num_edges": int(data.num_edges),
                    "test_nodes": int(data.test_idx.numel()),
                }
            )

            profiles, beta_record = parameter_audit(model, dataset, seed)
            beta_profile_rows.extend(profiles)
            beta_json.append(beta_record)
            for modality in MODALITIES:
                beta_vectors[(dataset, seed, modality)] = torch.tensor(
                    beta_record["modalities"][modality]["final_centered_normalized_profile"], dtype=torch.float32
                )
                beta_run_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "modality": modality,
                        **{
                            key: value
                            for key, value in beta_record["modalities"][modality].items()
                            if isinstance(value, (int, float))
                        },
                    }
                )

            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            labels = _resolve_nc_eval_labels(data)
            normal_full = model.analysis_hop_attention(x, edge_index, relation_intervention="normal")
            normal = compact(normal_full)
            del normal_full
            attention_by_modality = {
                modality: normal[f"hop_attention_{modality}"].float().detach()
                for modality in MODALITIES
            }
            diag = {
                modality: {key: value.detach().cpu().clone() for key, value in model._relation_diagnostics[modality].items()}
                for modality in MODALITIES
            }
            n_orders = int(attention_by_modality["text"].size(-1))
            for modality in MODALITIES:
                attention = attention_by_modality[modality]
                matrix = attention.mean(dim=0).detach().cpu().numpy()
                all_attention.setdefault((dataset, modality), []).append(matrix)
                for query_order in range(n_orders):
                    for key_order in range(n_orders):
                        heatmap_run_rows.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "modality": modality,
                                "query_order": query_order,
                                "key_order": key_order,
                                "mean_attention": float(matrix[query_order, key_order]),
                            }
                        )
                profile = attention.mean(dim=1).detach().cpu().numpy()
                radii = r_attn(attention).detach().cpu().numpy()
                d = diag[modality]
                raw_mu = d["relation_mu"].numpy()
                raw_sigma = d["relation_sigma"].numpy()
                zstate = d["relation_state"][:, 0].numpy()
                ranks = rankdata(zstate, method="average") / max(1, len(zstate))
                quartile = np.minimum((ranks * 4).astype(int) + 1, 4)
                for node in range(int(data.num_nodes)):
                    node_row = {
                        "dataset": dataset,
                        "seed": seed,
                        "node": node,
                        "modality": modality,
                        "r_attn": float(radii[node]),
                        "relation_mu": float(raw_mu[node]),
                        "relation_sigma": float(raw_sigma[node]),
                        "relation_state_mu_z": float(zstate[node]),
                        "relation_state_quartile": int(quartile[node]),
                    }
                    for order in range(n_orders):
                        node_row[f"attended_order_mass_{order}"] = float(profile[node, order])
                    relation_node_rows.append(node_row)
                    r_attn_rows.append({key: node_row[key] for key in ("dataset", "seed", "node", "modality", "r_attn")})
                for q in range(1, 5):
                    mask = quartile == q
                    row: dict[str, Any] = {
                        "dataset": dataset,
                        "seed": seed,
                        "modality": modality,
                        "relation_state_quartile": q,
                        "n_nodes": int(mask.sum()),
                        "relation_state_mu_z_mean": float(zstate[mask].mean()) if mask.any() else None,
                        "relation_mu_mean": float(raw_mu[mask].mean()) if mask.any() else None,
                        "relation_sigma_mean": float(raw_sigma[mask].mean()) if mask.any() else None,
                        "r_attn_mean": float(radii[mask].mean()) if mask.any() else None,
                    }
                    for order in range(n_orders):
                        row[f"attended_order_mass_{order}_mean"] = float(profile[mask, order].mean()) if mask.any() else None
                    quartile_rows.append(row)
                qstats = query_metrics(attention)
                mechanism_rows.append({"dataset": dataset, "seed": seed, "modality": modality, **qstats})

            for condition in ("Interaction-Off", "Uniform-Attention", "Query-Collapse"):
                if condition == "Interaction-Off":
                    changed_full = run_hop_intervention(model, x, edge_index, "interaction_off")
                elif condition == "Uniform-Attention":
                    changed_full = run_hop_intervention(model, x, edge_index, "uniform_attention")
                else:
                    changed_full = run_hop_intervention(model, x, edge_index, "query_collapse")
                changed = compact(changed_full)
                del changed_full
                rows, details = compare_outputs(
                    normal,
                    changed,
                    classifier,
                    data,
                    labels,
                    dataset=dataset,
                    seed=seed,
                    intervention=condition,
                )
                intervention_rows.extend(rows)
                intervention_details.append({"family": "hop", "dataset": dataset, "seed": seed, "condition": condition, **details})
                del changed

            # This pairwise contrast keeps each node's averaged key mass fixed in
            # Query-Collapse and replaces only that profile in Uniform-Attention.
            collapsed_full = run_hop_intervention(model, x, edge_index, "query_collapse")
            collapsed = compact(collapsed_full)
            del collapsed_full
            uniform_full = run_hop_intervention(model, x, edge_index, "uniform_attention")
            uniform = compact(uniform_full)
            del uniform_full
            rows, details = compare_outputs(
                collapsed,
                uniform,
                classifier,
                data,
                labels,
                dataset=dataset,
                seed=seed,
                intervention="Uniform-Attention",
                reference_name="Query-Collapse",
            )
            intervention_rows.extend(rows)
            intervention_details.append({"family": "hop_pairwise", "dataset": dataset, "seed": seed, "condition": "Uniform-Attention", **details})
            del collapsed, uniform

            for condition in ("Relation-Off", "Relation-Shuffle"):
                intervention_name = "off" if condition == "Relation-Off" else "shuffle"
                changed_full = model.analysis_hop_attention(
                    x,
                    edge_index,
                    relation_intervention=intervention_name,
                    permutation_seed=SHUFFLE_SEED,
                )
                changed = compact(changed_full)
                del changed_full
                rows, details = compare_outputs(
                    normal,
                    changed,
                    classifier,
                    data,
                    labels,
                    dataset=dataset,
                    seed=seed,
                    intervention=condition,
                )
                intervention_rows.extend(rows)
                intervention_details.append({"family": "relation", "dataset": dataset, "seed": seed, "condition": condition, **details})
                del changed

            del normal, model, classifier, data, payload, x, edge_index
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"ANALYZED {dataset} seed={seed}", flush=True)

    write_csv(OUTPUT_ROOT / "mechanism" / "checkpoint_audit.csv", checkpoint_rows)
    write_csv(OUTPUT_ROOT / "parameters" / "beta_profiles.csv", beta_profile_rows)
    write_csv(OUTPUT_ROOT / "parameters" / "beta_learning_per_run.csv", beta_run_rows)
    write_json(OUTPUT_ROOT / "parameters" / "beta_learning_audit.json", beta_json)
    write_csv(OUTPUT_ROOT / "mechanism" / "figure1_attention_by_seed.csv", heatmap_run_rows)
    write_csv(OUTPUT_ROOT / "mechanism" / "figure2_node_r_attn.csv", r_attn_rows)
    write_csv(OUTPUT_ROOT / "mechanism" / "figure3_relation_nodes.csv", relation_node_rows)
    write_csv(OUTPUT_ROOT / "mechanism" / "figure3_relation_quartiles.csv", quartile_rows)
    write_csv(OUTPUT_ROOT / "mechanism" / "normal_attention_mechanism.csv", mechanism_rows)
    write_csv(OUTPUT_ROOT / "interventions" / "intervention_metrics_per_run.csv", intervention_rows)
    write_json(OUTPUT_ROOT / "interventions" / "intervention_details.json", intervention_details)

    # Pairwise seed similarity of the final, centered and normalized beta profile.
    beta_seed_rows: list[dict[str, Any]] = []
    beta_similarity_summary: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for modality in MODALITIES:
            pair_values = []
            for first, second in combinations(SEEDS, 2):
                a = beta_vectors[(dataset, first, modality)]
                b = beta_vectors[(dataset, second, modality)]
                cosine = float(F.cosine_similarity(a, b, dim=0).item())
                pair_values.append(cosine)
                beta_seed_rows.append({"dataset": dataset, "modality": modality, "seed_a": first, "seed_b": second, "final_profile_cosine": cosine})
            mean, std = population_summary(pair_values)
            beta_similarity_summary.append({"dataset": dataset, "modality": modality, "mean_pairwise_final_profile_cosine": mean, "std_population": std, "min_pairwise_cosine": min(pair_values), "n_pairs": len(pair_values)})
    write_csv(OUTPUT_ROOT / "parameters" / "across_seed_beta_cosines.csv", beta_seed_rows)
    write_csv(OUTPUT_ROOT / "parameters" / "across_seed_beta_cosine_summary.csv", beta_similarity_summary)

    # Relation shuffle/off impact ratios use the three seed means, preserving
    # separate ratios for Text/Visual attention and fused Z/logits.
    value_lookup = {
        (row["family"], row["dataset"], row["seed"], row["condition"], "fused", key): float(value)
        for row in intervention_details
        for key, value in row.items()
        if key in {"text_attention_mae", "visual_attention_mae", "Z_mae", "logit_mae"}
    }
    ratio_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for metric, modality in (("attention", "text"), ("attention", "visual"), ("Z_mae", "fused"), ("logit_mae", "fused")):
            off_key = f"{modality}_attention_mae" if metric == "attention" else metric
            off_vals = [value_lookup[("relation", dataset, seed, "Relation-Off", "fused", off_key)] for seed in SEEDS]
            shuffle_vals = [value_lookup[("relation", dataset, seed, "Relation-Shuffle", "fused", off_key)] for seed in SEEDS]
            off_mean = float(np.mean(off_vals))
            shuffle_mean = float(np.mean(shuffle_vals))
            ratio_rows.append({
                "dataset": dataset,
                "quantity": metric,
                "modality": modality,
                "relation_off_impact_mean": off_mean,
                "relation_shuffle_impact_mean": shuffle_mean,
                "shuffle_over_off_ratio": shuffle_mean / off_mean if off_mean > 1e-15 else None,
                "shuffle_greater_than_off": bool(shuffle_mean > off_mean),
            })
    write_csv(OUTPUT_ROOT / "interventions" / "relation_shuffle_off_ratios.csv", ratio_rows)

    modality_data = write_figure_data(
        heatmap_run_rows,
        r_attn_rows,
        quartile_rows,
        intervention_rows,
        intervention_details,
        all_attention,
        ratio_rows,
        beta_run_rows,
    )
    return {
        "performance": performance,
        "performance_summary": perf_summary,
        "parameter_runs": beta_json,
        "beta_seed_similarity": beta_similarity_summary,
        "intervention_details": intervention_details,
        "relation_impact_ratios": ratio_rows,
        "normal_mechanism": mechanism_rows,
        "checkpoint_audit": checkpoint_rows,
        "text_visual_data_dependence": modality_data,
    }


def write_figure_data(
    heatmap_run_rows,
    r_attn_rows,
    quartile_rows,
    intervention_rows,
    intervention_details,
    all_attention,
    ratio_rows,
    beta_run_rows,
) -> list[dict[str, Any]]:
    aggregate_heat_rows: list[dict[str, Any]] = []
    for (dataset, modality), mats in all_attention.items():
        avg = np.mean(np.stack(mats, axis=0), axis=0)
        for query in range(avg.shape[0]):
            for key in range(avg.shape[1]):
                aggregate_heat_rows.append({"dataset": dataset, "modality": modality, "query_order": query, "key_order": key, "mean_attention": float(avg[query, key])})
    write_csv(OUTPUT_ROOT / "figures" / "figure1_mean_attention_heatmaps.csv", aggregate_heat_rows)

    modality_contrasts: list[dict[str, Any]] = []
    for dataset in DATASETS:
        text_matrix = np.mean(np.stack(all_attention[(dataset, "text")], axis=0), axis=0)
        visual_matrix = np.mean(np.stack(all_attention[(dataset, "visual")], axis=0), axis=0)
        modality_contrasts.append(
            {
                "contrast": "Text-vs-Visual within dataset",
                "dataset_a": dataset,
                "dataset_b": dataset,
                "modality": "both",
                "mean_attention_matrix_mae": float(np.mean(np.abs(text_matrix - visual_matrix))),
            }
        )
    for modality in MODALITIES:
        for first, second in combinations(DATASETS, 2):
            first_matrix = np.mean(np.stack(all_attention[(first, modality)], axis=0), axis=0)
            second_matrix = np.mean(np.stack(all_attention[(second, modality)], axis=0), axis=0)
            modality_contrasts.append(
                {
                    "contrast": "between dataset mean attention profiles",
                    "dataset_a": first,
                    "dataset_b": second,
                    "modality": modality,
                    "mean_attention_matrix_mae": float(np.mean(np.abs(first_matrix - second_matrix))),
                }
            )
    write_csv(OUTPUT_ROOT / "figures" / "figure1_text_visual_dataset_contrasts.csv", modality_contrasts)

    # Aggregate R1 mechanisms over seeds for a compact figure-data handoff.
    mechanism_rows = []
    for (dataset, modality), mats in all_attention.items():
        # The node-level CSV is the source for Figure 2; no node is subsampled.
        matching = [row for row in r_attn_rows if row["dataset"] == dataset and row["modality"] == modality]
        values = np.asarray([row["r_attn"] for row in matching], dtype=np.float64)
        mechanism_rows.append({
            "dataset": dataset,
            "modality": modality,
            "r_attn_mean_all_nodes_seeds": float(values.mean()),
            "r_attn_sd_all_nodes_seeds": float(values.std(ddof=0)),
            "r_attn_q25": float(np.quantile(values, 0.25)),
            "r_attn_median": float(np.quantile(values, 0.50)),
            "r_attn_q75": float(np.quantile(values, 0.75)),
        })
    write_csv(OUTPUT_ROOT / "figures" / "figure2_r_attn_distribution_summary.csv", mechanism_rows)

    # Quartile aggregation across seeds retains all node-level source rows above.
    quartile_summary: list[dict[str, Any]] = []
    group_keys = sorted({(r["dataset"], r["modality"], r["relation_state_quartile"]) for r in quartile_rows})
    for dataset, modality, quartile in group_keys:
        group = [r for r in quartile_rows if (r["dataset"], r["modality"], r["relation_state_quartile"]) == (dataset, modality, quartile)]
        n = sum(int(r["n_nodes"]) for r in group)
        row = {
            "dataset": dataset,
            "modality": modality,
            "relation_state_quartile": quartile,
            "n_nodes_total_across_seeds": n,
            "relation_state_mu_z_mean": float(np.average([r["relation_state_mu_z_mean"] for r in group], weights=[r["n_nodes"] for r in group])),
            "r_attn_mean": float(np.average([r["r_attn_mean"] for r in group], weights=[r["n_nodes"] for r in group])),
        }
        for order in range(4):
            key = f"attended_order_mass_{order}_mean"
            row[key] = float(np.average([r[key] for r in group], weights=[r["n_nodes"] for r in group]))
        quartile_summary.append(row)
    write_csv(OUTPUT_ROOT / "figures" / "figure3_relation_quartile_summary.csv", quartile_summary)

    # Figure 4/5 data are structured long-form rows already; also provide seed
    # aggregates for quick charting and report tables.
    metric_aggregates: list[dict[str, Any]] = []
    for family in ("relation", "hop", "hop_pairwise"):
        for dataset in DATASETS:
            conditions = sorted({r["condition"] for r in intervention_details if r["family"] == family and r["dataset"] == dataset})
            for condition in conditions:
                runs = [r for r in intervention_details if r["family"] == family and r["dataset"] == dataset and r["condition"] == condition]
                for metric in ("text_attention_mae", "visual_attention_mae", "Z_mae", "logit_mae", "probability_kl_reference_to_condition", "prediction_flip_rate_all_nodes", "prediction_flip_rate_test", "test_acc_delta_vs_reference", "test_macro_f1_delta_vs_reference"):
                    vals = [float(r[metric]) for r in runs if metric in r]
                    if not vals:
                        continue
                    mean, std = population_summary(vals)
                    metric_aggregates.append({"family": family, "dataset": dataset, "condition": condition, "metric": metric, "mean": mean, "std_population": std, "n_seeds": len(vals)})
    write_csv(OUTPUT_ROOT / "figures" / "figure4_5_intervention_summary.csv", metric_aggregates)

    # Five single-file diagnostic charts, using all five NC datasets.
    fig_root = OUTPUT_ROOT / "figures" / "diagnostics"
    fig_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})

    chosen = ("Movies", "Grocery", "Toys")
    fig, axes = plt.subplots(len(chosen), 2, figsize=(7.2, 8.0), constrained_layout=True)
    for row_idx, dataset in enumerate(chosen):
        for col_idx, modality in enumerate(MODALITIES):
            matrix = np.mean(np.stack(all_attention[(dataset, modality)], axis=0), axis=0)
            ax = axes[row_idx, col_idx]
            im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=max(0.4, float(matrix.max())))
            ax.set_title(f"{dataset} · {modality.title()}")
            ax.set_xlabel("Attended order k")
            ax.set_ylabel("Query order q")
            ax.set_xticks(range(matrix.shape[1]))
            ax.set_yticks(range(matrix.shape[0]))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.savefig(fig_root / "figure1_mean_hop_attention.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    labels = []
    values = []
    positions = []
    pos = 1
    for dataset in DATASETS:
        for modality in MODALITIES:
            vals = [r["r_attn"] for r in r_attn_rows if r["dataset"] == dataset and r["modality"] == modality]
            values.append(np.asarray(vals))
            positions.append(pos)
            labels.append(f"{dataset}\n{modality.title()}")
            pos += 1
        pos += 0.8
    box = ax.boxplot(values, positions=positions, widths=0.58, patch_artist=True, showfliers=False)
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor("#9BBBD2" if i % 2 == 0 else "#D6B589")
        patch.set_alpha(0.82)
    ax.set_xticks(positions, labels, rotation=0)
    ax.set_ylabel("Node-level r_attn")
    ax.set_title("R1 attended-order preference across nodes and seeds")
    ax.set_ylim(0.0, 1.0)
    fig.savefig(fig_root / "figure2_node_r_attn_distributions.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    dataset_colors = {"Movies": "#3E77A8", "Grocery": "#C37A2C", "Toys": "#5B8A5B"}
    modality_styles = {"text": ("-", "o"), "visual": ("--", "s")}
    for dataset in chosen:
        for modality in MODALITIES:
            linestyle, marker = modality_styles[modality]
            rows = sorted([r for r in quartile_summary if r["dataset"] == dataset and r["modality"] == modality], key=lambda r: r["relation_state_quartile"])
            axes[0].plot([r["relation_state_quartile"] for r in rows], [r["r_attn_mean"] for r in rows], marker=marker, linestyle=linestyle, color=dataset_colors[dataset], alpha=0.9, label=f"{dataset} {modality}")
    axes[0].set_xlabel("Relation-state μ quartile")
    axes[0].set_ylabel("Mean r_attn")
    axes[0].set_xticks((1, 2, 3, 4))
    axes[0].legend(fontsize=6, ncol=2)
    datasets_to_show = chosen
    xlocs = np.arange(4)
    for dataset in datasets_to_show:
        for modality in MODALITIES:
            linestyle, marker = modality_styles[modality]
            rows = sorted([r for r in quartile_summary if r["modality"] == modality and r["dataset"] == dataset], key=lambda r: r["relation_state_quartile"])
            r0 = [r["attended_order_mass_0_mean"] for r in rows]
            axes[1].plot(xlocs, r0, marker=marker, linestyle=linestyle, color=dataset_colors[dataset], alpha=0.9, label=f"{dataset} {modality}")
    axes[1].set_xlabel("Relation-state μ quartile")
    axes[1].set_ylabel("Mean attended order-0 mass")
    axes[1].set_xticks(xlocs, ("Q1", "Q2", "Q3", "Q4"))
    axes[1].legend(fontsize=6, ncol=2)
    fig.savefig(fig_root / "figure3_relation_state_quartiles.png", dpi=160)
    plt.close(fig)

    # Relation-only intervention shift heatmaps, one panel per output family.
    rel = [r for r in metric_aggregates if r["family"] == "relation"]
    rel_metrics = (("text_attention_mae", "Text attention"), ("visual_attention_mae", "Visual attention"), ("Z_mae", "Fused Z"), ("logit_mae", "Logits"))
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 4.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes, rel_metrics):
        matrix = np.full((len(DATASETS), 2), np.nan)
        for di, dataset in enumerate(DATASETS):
            for ci, condition in enumerate(("Relation-Off", "Relation-Shuffle")):
                row = next((r for r in rel if r["dataset"] == dataset and r["condition"] == condition and r["metric"] == metric), None)
                if row:
                    matrix[di, ci] = row["mean"]
        im = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xticks((0, 1), ("Off", "Shuffle"), rotation=35)
        ax.set_yticks(range(len(DATASETS)), DATASETS)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i,j]:.3g}", ha="center", va="center", fontsize=7, color="white" if matrix[i,j] > np.nanmedian(matrix) else "black")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.savefig(fig_root / "figure4_relation_intervention_shifts.png", dpi=160)
    plt.close(fig)

    hop = [r for r in metric_aggregates if r["family"] == "hop"]
    conditions = ("Query-Collapse", "Uniform-Attention", "Interaction-Off")
    metrics = ("test_acc_delta_vs_reference", "test_macro_f1_delta_vs_reference", "Z_mae", "logit_mae")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.1), constrained_layout=True)
    colors = {"Query-Collapse": "#6B93B5", "Uniform-Attention": "#C69A61", "Interaction-Off": "#898989"}
    for ax, metric in zip(axes.flat, metrics):
        width = 0.24
        x = np.arange(len(DATASETS))
        for ci, condition in enumerate(conditions):
            values = []
            for dataset in DATASETS:
                row = next((r for r in hop if r["dataset"] == dataset and r["condition"] == condition and r["metric"] == metric), None)
                values.append(row["mean"] if row else 0.0)
            ax.bar(x + (ci - 1) * width, values, width, color=colors[condition], label=condition)
        ax.axhline(0.0, color="#333333", lw=0.7)
        ax.set_xticks(x, DATASETS, rotation=20)
        ax.set_title(metric.replace("_", " "))
        ax.legend(fontsize=6)
    fig.savefig(fig_root / "figure5_hop_intervention_degradation.png", dpi=160)
    plt.close(fig)
    return modality_contrasts


def aggregate_interventions(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in ("hop", "hop_pairwise", "relation"):
        for dataset in DATASETS:
            conditions = sorted({row["condition"] for row in details if row["family"] == family and row["dataset"] == dataset})
            for condition in conditions:
                run_rows = [row for row in details if row["family"] == family and row["dataset"] == dataset and row["condition"] == condition]
                metrics = sorted({key for row in run_rows for key, value in row.items() if key not in {"family", "dataset", "seed", "condition"} and isinstance(value, (int, float))})
                for metric in metrics:
                    values = [float(row[metric]) for row in run_rows if metric in row]
                    mean, std = population_summary(values)
                    rows.append({"family": family, "dataset": dataset, "condition": condition, "metric": metric, "mean": mean, "std_population": std, "n_seeds": len(values)})
    return rows


def decide(audit: dict[str, Any]) -> dict[str, Any]:
    beta_runs = audit["parameter_runs"]
    beta_values = [m for run in beta_runs for m in run["modalities"].values()]
    rel_change = [m["relative_l2_change_effective_profile"] for m in beta_values]
    cosine = [m["cosine_initial_final_profile"] for m in beta_values]
    beta_learned = bool(np.median(rel_change) > 0.10 and np.median(cosine) < 0.99)
    mech_by_dataset_mod = {(r["dataset"], r["seed"], r["modality"]): r for r in audit["normal_mechanism"]}
    attention_nonuniform = all(
        mech_by_dataset_mod[(ds, seed, mod)]["attention_max_abs_deviation_from_uniform"] > 1e-6
        for ds in DATASETS for seed in SEEDS for mod in MODALITIES
    )
    hop_rows = [r for r in audit["intervention_details"] if r["family"] == "hop"]
    collapse_effect = sum(r["logit_mae"] > 1e-8 or r["Z_mae"] > 1e-8 for r in hop_rows if r["condition"] == "Query-Collapse")
    relation_rows = [r for r in audit["intervention_details"] if r["family"] == "relation"]
    relation_off_effect = sum(r["logit_mae"] > 1e-8 or r["Z_mae"] > 1e-8 for r in relation_rows if r["condition"] == "Relation-Off")
    ratios = audit["relation_impact_ratios"]
    relation_shuffle_usually_ge_off = sum(r["shuffle_greater_than_off"] for r in ratios) >= math.ceil(len(ratios) / 2)
    formal_negative = audit["performance"]["counts"]["Formal V0"]["negative"]
    core_negative = audit["performance"]["counts"]["IAMOC core V2/R0"]["negative"]
    # “Broad, clear regression” is conservatively operationalized as over half
    # of the 10 test metric cells outside the baseline-seed-SD neutral band.
    broad_formal_degradation = formal_negative > 5
    close_to_core = core_negative <= 5
    contrasts = audit["text_visual_data_dependence"]
    within_dataset = [r["mean_attention_matrix_mae"] for r in contrasts if r["contrast"] == "Text-vs-Visual within dataset"]
    between_dataset = [r["mean_attention_matrix_mae"] for r in contrasts if r["contrast"] == "between dataset mean attention profiles"]
    modality_heterogeneity = bool(within_dataset and between_dataset and np.median(within_dataset) > 1e-6 and np.median(between_dataset) > 1e-6)
    if beta_learned and attention_nonuniform and collapse_effect > 0 and relation_off_effect > 0 and relation_shuffle_usually_ge_off and modality_heterogeneity and close_to_core and not broad_formal_degradation:
        label = "FINAL ARCHITECTURE CANDIDATE"
    elif not beta_learned or not relation_shuffle_usually_ge_off:
        label = "RELATION MECHANISM NOT YET SUFFICIENT"
    else:
        label = "CANDIDATE EVIDENCE MIXED"
    return {
        "label": label,
        "criteria": {
            "beta_profile_learning_rule": "median effective centered-profile relative L2 change > 0.10 and median initial/final cosine < 0.99",
            "beta_profile_learned": beta_learned,
            "median_effective_profile_relative_l2_change": float(np.median(rel_change)),
            "median_initial_final_profile_cosine": float(np.median(cosine)),
            "hop_attention_nonuniform_all_runs_modalities": attention_nonuniform,
            "query_collapse_measurable_run_count_of_15": int(collapse_effect),
            "relation_off_measurable_run_count_of_15": int(relation_off_effect),
            "shuffle_impact_greater_than_off_count_of_20_ratios": int(sum(r["shuffle_greater_than_off"] for r in ratios)),
            "relation_shuffle_usually_ge_off": relation_shuffle_usually_ge_off,
            "text_visual_and_dataset_dependent_attention_patterns": modality_heterogeneity,
            "R1_minus_Formal_V0_negative_cells_out_of_10": formal_negative,
            "broad_formal_degradation_rule": "more than 5/10 test metric cells fall below negative baseline-seed-SD neutral band",
            "broad_formal_degradation": broad_formal_degradation,
            "R1_minus_IAMOC_core_V2_negative_cells_out_of_10": core_negative,
            "performance_close_to_core_rule": "no more than 5/10 test metric cells fall below negative core-seed-SD neutral band",
            "performance_close_to_core": close_to_core,
        },
    }


def write_report(audit: dict[str, Any]) -> None:
    summary = audit["performance_summary"]
    summary_lookup = {(r["dataset"], r["model"], r["metric"]): r for r in summary}
    lines = [
        "# RC-IAMOC Final Candidate Consolidation",
        "",
        "## Scope and protocol",
        "",
        "R1 denotes Relation-Conditioned Interaction-Aware Multi-Order Composition (RC-IAMOC). The frozen R1 configuration uses one single-head hop-interaction layer, order embeddings, rank-1 relation state, relation scale initialization 0.10, `relation_conditioning=none`, and `ppc_weight=0`. The model's inherited Stage I/II, fusion, and NC classifier settings match the completed R1 configuration. `relation_conditioning=none` disables the relation output path; it leaves the inherited transport residual used by this model intact.",
        "",
        "The added training runs cover Toys, ele-fashion, and Reddit-S with seeds 42/43/44. Movies and Grocery R1 checkpoints were reused. Formal V0 and IAMOC core V2/R0 baselines were reused from the existing five-dataset NC grid and its audited legacy Movies/Grocery runs. All seed summaries below use population standard deviation. No LP or hyperparameter search was run.",
        "",
        "## A. Five-dataset NC generalization",
        "",
        "Test Accuracy and Test Macro-F1 are reported as mean ± population SD across seeds. Validation metrics, epoch and source paths are retained in the result CSVs.",
        "",
        "| Dataset | Model | Test Accuracy | Test Macro-F1 |",
        "|---|---|---:|---:|",
    ]
    for dataset in DATASETS:
        for model in ("Formal V0", "IAMOC core V2/R0", "R1"):
            acc = summary_lookup[(dataset, model, "test_acc")]
            f1 = summary_lookup[(dataset, model, "test_macro_f1")]
            lines.append(f"| {dataset} | {model} | {acc['mean']:.4f} ± {acc['std_population']:.4f} | {f1['mean']:.4f} ± {f1['std_population']:.4f} |")
    lines += [
        "",
        "Validation Accuracy and Macro-F1 are included as a separate checkpoint-selection view:",
        "",
        "| Dataset | Model | Val Accuracy | Val Macro-F1 |",
        "|---|---|---:|---:|",
    ]
    for dataset in DATASETS:
        for model in ("Formal V0", "IAMOC core V2/R0", "R1"):
            acc = summary_lookup[(dataset, model, "val_acc")]
            f1 = summary_lookup[(dataset, model, "val_macro_f1")]
            lines.append(f"| {dataset} | {model} | {acc['mean']:.4f} ± {acc['std_population']:.4f} | {f1['mean']:.4f} ± {f1['std_population']:.4f} |")
    lines += [
        "",
        "Paired seed deltas are R1 minus the named baseline. Positive/neutral/negative cell counts classify each of the 10 dataset × test-metric mean deltas against a neutral band of ±1 baseline population SD; the per-seed deltas remain available in `performance/paired_seed_deltas.csv`.",
        "",
        "| Baseline | Positive | Neutral | Negative |",
        "|---|---:|---:|---:|",
    ]
    for baseline, counts in audit["performance"]["counts"].items():
        lines.append(f"| {baseline} | {counts['positive']} | {counts['neutral']} | {counts['negative']} |")
    lines += ["", "| Dataset | Baseline | Metric | Seed 42 Δ | Seed 43 Δ | Seed 44 Δ | Mean Δ ± population SD | Class |", "|---|---|---|---:|---:|---:|---:|---|"]
    for row in audit["performance"]["paired"]:
        deltas = row["paired_delta_by_seed"]
        lines.append(f"| {row['dataset']} | {row['reference']} | {row['metric']} | {deltas['42']:+.4f} | {deltas['43']:+.4f} | {deltas['44']:+.4f} | {row['mean_delta']:+.4f} ± {row['std_population_delta']:.4f} | {row['classification']} |")

    lines += [
        "",
        "## B. R1 parameter-learning audit",
        "",
        "The initialization vector is reconstructed from the recorded fixed seed and initialization standard deviation. The effective profile is `center(beta_raw) / RMS(center(beta_raw))`, matching the R1 forward calculation. Relative L2 change is computed on that effective profile; raw-parameter change and both raw/profile cosine values are also in the JSON. Relation scales are sigmoid-transformed parameters.",
        "",
        "| Dataset | Modality | Median cosine(initial, final) | Median effective-profile relative L2 change | Relation scale initial → final (mean over seeds) | Mean across-seed final-profile cosine |",
        "|---|---|---:|---:|---:|---:|",
    ]
    cos_lookup = {(r["dataset"], r["modality"]): r for r in audit["beta_seed_similarity"]}
    for dataset in DATASETS:
        for modality in MODALITIES:
            entries = [run["modalities"][modality] for run in audit["parameter_runs"] if run["dataset"] == dataset]
            lines.append(
                f"| {dataset} | {modality.title()} | {np.median([x['cosine_initial_final_profile'] for x in entries]):.4f} | "
                f"{np.median([x['relative_l2_change_effective_profile'] for x in entries]):.4f} | "
                f"{np.mean([x['relation_scale_initial'] for x in entries]):.4f} → {np.mean([x['relation_scale_final'] for x in entries]):.4f} | "
                f"{cos_lookup[(dataset, modality)]['mean_pairwise_final_profile_cosine']:.4f} |"
            )
    beta_rule = audit["decision"]["criteria"]
    lines += [
        "",
        f"Using the declared learning diagnostic (median effective-profile relative L2 change > 0.10 and median initial/final profile cosine < 0.99), beta profile learned: **{beta_rule['beta_profile_learned']}**. Initial/final raw vectors and normalized profiles are in `parameters/beta_learning_audit.json`; order-wise CSV rows are in `parameters/beta_profiles.csv`.",
        "",
        "## C. Hop-interaction functional interventions",
        "",
        "Every intervention is inference-only and reloads the unchanged best checkpoint. Interaction-Off sets only the local hop gate to zero; Uniform-Attention replaces each query's key distribution with 1/(K+1); Query-Collapse broadcasts each node's query-mean key-mass profile across queries. Thus Query-Collapse retains the node-specific key-mass profile while removing query-dependent cross-order interaction. Attention and representation shifts use all nodes; test metrics use the frozen test split.",
        "",
        "| Dataset | Intervention | Attention MAE (Text / Visual) | η MAE (Text / Visual) | Z MAE | Logit MAE | KL(Pnormal‖Pcondition) | Prediction flip (test) | Test Acc Δ | Test Macro-F1 Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    hop_details = [r for r in audit["intervention_details"] if r["family"] == "hop"]
    for dataset in DATASETS:
        for condition in ("Interaction-Off", "Uniform-Attention", "Query-Collapse"):
            runs = [r for r in hop_details if r["dataset"] == dataset and r["condition"] == condition]
            mean = lambda key: float(np.mean([r[key] for r in runs]))
            lines.append(
                f"| {dataset} | {condition} | {mean('text_attention_mae'):.4g} / {mean('visual_attention_mae'):.4g} | "
                f"{mean('text_eta_mae'):.4g} / {mean('visual_eta_mae'):.4g} | {mean('Z_mae'):.4g} | {mean('logit_mae'):.4g} | "
                f"{mean('probability_kl_reference_to_condition'):.4g} | {mean('prediction_flip_rate_test'):.4g} | "
                f"{mean('test_acc_delta_vs_reference'):+.4g} | {mean('test_macro_f1_delta_vs_reference'):+.4g} |"
            )
    lines += [
        "",
        "Normal attention structure averaged over seeds (node means are calculated before the seed summary):",
        "",
        "| Dataset | Modality | KL to uniform | Normalized entropy | Max key mass | Query-row MAE | Pairwise query JS | r_attn mean ± node SD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    mechanism_rows = audit["normal_mechanism"]
    for dataset in DATASETS:
        for modality in MODALITIES:
            runs = [r for r in mechanism_rows if r["dataset"] == dataset and r["modality"] == modality]
            mean = lambda key: float(np.mean([r[key] for r in runs]))
            lines.append(
                f"| {dataset} | {modality.title()} | {mean('kl_to_uniform_mean'):.4g} | {mean('normalized_entropy_mean'):.4g} | "
                f"{mean('max_key_mass_mean'):.4g} | {mean('query_row_mae_mean'):.4g} | {mean('pairwise_query_js_mean'):.4g} | "
                f"{mean('r_attn_mean'):.4f} ± {mean('r_attn_std_nodes'):.4f} |"
            )
    lines += [
        "",
        "Direct Query-Collapse vs Uniform-Attention contrast (Uniform minus Query-Collapse for test metrics):",
        "",
        "| Dataset | Attention MAE (Text / Visual) | η MAE (Text / Visual) | Z MAE | Logit MAE | KL(Pcollapse‖Puniform) | Test flip rate | Test Acc Δ | Test Macro-F1 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pair_details = [r for r in audit["intervention_details"] if r["family"] == "hop_pairwise"]
    for dataset in DATASETS:
        runs = [r for r in pair_details if r["dataset"] == dataset]
        mean = lambda key: float(np.mean([r[key] for r in runs]))
        lines.append(
            f"| {dataset} | {mean('text_attention_mae'):.4g} / {mean('visual_attention_mae'):.4g} | "
            f"{mean('text_eta_mae'):.4g} / {mean('visual_eta_mae'):.4g} | {mean('Z_mae'):.4g} | {mean('logit_mae'):.4g} | "
            f"{mean('probability_kl_reference_to_condition'):.4g} | {mean('prediction_flip_rate_test'):.4g} | "
            f"{mean('test_acc_delta_vs_reference'):+.4g} | {mean('test_macro_f1_delta_vs_reference'):+.4g} |"
        )
    lines += [
        "",
        "Normal attention non-uniformity and query dependence are summarized in `mechanism/normal_attention_mechanism.csv`. Normal vs Query-Collapse measures the value of query-dependent cross-order interaction. Query-Collapse vs Uniform-Attention is separately recorded with `reference=Query-Collapse`; it measures the value of node-specific hop preference, because collapse keeps each node's averaged key mass. Performance changes are reported alongside tensor shifts, so attention changes alone are not interpreted as performance causation.",
        "",
        "## D. Relation functional interventions",
        "",
        "Relation-Off removes the explicit rank-1 relation bias. Relation-Shuffle applies the same fixed node permutation to the detached text and visual relation states while preserving the learned parameters and other computations. MAEs and KL are Normal-relative; `r_attn shift` is signed Normal minus intervention, with absolute node-mean shift also stored. Accuracy/F1 deltas are intervention minus Normal on test nodes.",
        "",
        "| Dataset | Condition | Attention MAE (Text / Visual) | r_attn abs shift (Text / Visual) | η MAE (Text / Visual) | Z MAE | Logit MAE | Probability KL | Flip rate (test) | Test Acc Δ | Test Macro-F1 Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rel_details = [r for r in audit["intervention_details"] if r["family"] == "relation"]
    for dataset in DATASETS:
        for condition in ("Relation-Off", "Relation-Shuffle"):
            runs = [r for r in rel_details if r["dataset"] == dataset and r["condition"] == condition]
            mean = lambda key: float(np.mean([r[key] for r in runs]))
            lines.append(
                f"| {dataset} | {condition} | {mean('text_attention_mae'):.4g} / {mean('visual_attention_mae'):.4g} | "
                f"{mean('text_r_attn_shift_abs_mean'):.4g} / {mean('visual_r_attn_shift_abs_mean'):.4g} | "
                f"{mean('text_eta_mae'):.4g} / {mean('visual_eta_mae'):.4g} | {mean('Z_mae'):.4g} | {mean('logit_mae'):.4g} | "
                f"{mean('probability_kl_reference_to_condition'):.4g} | {mean('prediction_flip_rate_test'):.4g} | "
                f"{mean('test_acc_delta_vs_reference'):+.4g} | {mean('test_macro_f1_delta_vs_reference'):+.4g} |"
            )
    lines += [
        "",
        "| Dataset | Quantity | Modality | Off impact | Shuffle impact | Shuffle / Off |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in audit["relation_impact_ratios"]:
        ratio = "n/a" if row["shuffle_over_off_ratio"] is None else f"{row['shuffle_over_off_ratio']:.4g}"
        lines.append(f"| {row['dataset']} | {row['quantity']} | {row['modality']} | {row['relation_off_impact_mean']:.4g} | {row['relation_shuffle_impact_mean']:.4g} | {ratio} |")
    lines += [
        "",
        "A Shuffle impact above Off means the correct node-to-relation-state assignment carries functional information for that measured output. This remains an inference intervention result and does not establish training-time or downstream causal benefit by itself.",
        "",
        "## E. Figure-ready data and diagnostic plots",
        "",
        "Figure data use all nodes and all three seeds. Figure 1 reports mean Text/Visual query-by-key attention matrices for all datasets. Figure 2 has per-node `r_attn` values. Figure 3 contains relation-state quartile rows and attended-order profiles. Figure 4 contains Normal/Off/Shuffle attention, Z and logit shifts. Figure 5 contains Normal/Query-Collapse/Uniform/Interaction-Off degradation and performance deltas.",
        "",
        "Quick diagnostic plots:",
        "",
        "- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure1_mean_hop_attention.png`",
        "- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure2_node_r_attn_distributions.png`",
        "- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure3_relation_state_quartiles.png`",
        "- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure4_relation_intervention_shifts.png`",
        "- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure5_hop_intervention_degradation.png`",
        "",
        "Text-vs-Visual and between-dataset matrix distances are in `figures/figure1_text_visual_dataset_contrasts.csv`. All plotted source rows are CSV files alongside the figure data.",
        "",
        "## F. Candidate decision",
        "",
        f"**{audit['decision']['label']}**",
        "",
        f"- Learned beta profile: {beta_rule['beta_profile_learned']} (median effective-profile relative L2 change {beta_rule['median_effective_profile_relative_l2_change']:.4f}; median initial/final cosine {beta_rule['median_initial_final_profile_cosine']:.4f}).",
        f"- Non-uniform hop attention across all runs and modalities: {beta_rule['hop_attention_nonuniform_all_runs_modalities']}.",
        f"- Query-Collapse changed Z or logits beyond numerical tolerance in {beta_rule['query_collapse_measurable_run_count_of_15']}/15 runs.",
        f"- Relation-Off changed Z or logits beyond numerical tolerance in {beta_rule['relation_off_measurable_run_count_of_15']}/15 runs.",
        f"- Shuffle impact exceeded Off in {beta_rule['shuffle_impact_greater_than_off_count_of_20_ratios']}/20 dataset/quantity ratios; the 'usually' criterion is {beta_rule['relation_shuffle_usually_ge_off']}.",
        f"- Text/Visual and between-dataset attention profiles differ under the declared matrix-MAE diagnostic: {beta_rule['text_visual_and_dataset_dependent_attention_patterns']}.",
        f"- R1 negative cells beyond the baseline-SD band: {beta_rule['R1_minus_Formal_V0_negative_cells_out_of_10']}/10 vs Formal V0 and {beta_rule['R1_minus_IAMOC_core_V2_negative_cells_out_of_10']}/10 vs core V2/R0.",
        "",
        "Interpretation: the neutral band is descriptive and based on the baseline's three-seed population SD; it is not a hypothesis test. The intervention contrasts establish sensitivity of the frozen forward computation to the specified functional changes, and do not imply that a plotted attention pattern alone improves NC performance.",
        "",
        "## G. Reproducibility files",
        "",
        "All newly generated results are stored under `outputs/rc_iamoc_final_candidate/`; the nine additional run directories record their exact resolved configuration and checkpoint. `final_candidate_audit.json` contains the combined decision evidence. Existing IAMOC v1/v2 and generalization trees were read only.",
        "",
    ]
    report = ROOT / "docs" / "rc_iamoc_final_candidate_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE_DEFAULT)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    result = analyze_checkpoints(device)
    result["decision"] = decide(result)
    write_json(OUTPUT_ROOT / "final_candidate_audit.json", result)
    write_report(result)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
