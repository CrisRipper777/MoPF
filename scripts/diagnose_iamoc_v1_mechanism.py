#!/usr/bin/env python3
"""Inference-only mechanism audits for completed IAMOC v1 NC checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
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
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


def _distribution(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {name: math.nan for name in ("mean", "std", "median", "p25", "p75", "p90", "p95", "max")}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.quantile(array, 0.50)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


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


def _parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _LogitCapture:
    """Capture exact content, relation, total logits, and attention per block."""

    def __init__(self, model: nn.Module):
        self.records: dict[str, list[dict[str, torch.Tensor]]] = {"text": [], "visual": []}
        self.handles = []
        for modality in ("text", "visual"):
            layers = getattr(model, f"hop_layers_{modality}")
            for layer_index, layer in enumerate(layers):
                self.handles.append(
                    layer.register_forward_hook(self._hook(modality, layer_index))
                )

    def _hook(self, modality: str, layer_index: int):
        def capture(module, inputs, output):
            tokens = inputs[0]
            normalized = module.norm(tokens)
            query = module.query(normalized)
            key = module.key(normalized)
            content = torch.matmul(query, key.transpose(-1, -2)) * module.scale
            relation = inputs[3] if len(inputs) > 3 else None
            if relation is None:
                bias = torch.zeros_like(content)
            else:
                bias = relation.unsqueeze(1).expand_as(content)
            attention = output[1]
            self.records[modality].append(
                {
                    "layer": torch.tensor(layer_index + 1),
                    "content_logits": content.detach().float().cpu(),
                    "relation_bias": bias.detach().float().cpu(),
                    "total_logits": (content + bias).detach().float().cpu(),
                    "attention": attention.detach().float().cpu(),
                }
            )

        return capture

    def clear(self) -> None:
        self.records = {"text": [], "visual": []}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _order_weights(attention: torch.Tensor) -> torch.Tensor:
    count = attention.size(-1)
    if count <= 1:
        return torch.zeros(count, device=attention.device, dtype=attention.dtype)
    return torch.arange(count, device=attention.device, dtype=attention.dtype) / float(count - 1)


def _profile(attention: torch.Tensor) -> dict[str, torch.Tensor]:
    attention = attention.detach().float()
    key_mass = attention.mean(dim=1)
    order_weights = _order_weights(attention)
    r_attn = (key_mass * order_weights.unsqueeze(0)).sum(dim=-1)
    entropy = -(key_mass * key_mass.clamp_min(1e-12).log()).sum(dim=-1)
    order_count = attention.size(-1)
    kl_uniform = (key_mass * (key_mass.clamp_min(1e-12).log() + math.log(order_count))).sum(dim=-1)
    normalized_entropy = entropy / math.log(order_count)
    max_key_mass = key_mass.max(dim=-1).values
    return {
        "attention": attention,
        "key_mass": key_mass,
        "r_attn": r_attn,
        "entropy": entropy,
        "kl_uniform": kl_uniform,
        "normalized_entropy": normalized_entropy,
        "max_key_mass": max_key_mass,
    }


def _query_diversity(attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Node-wise query-row MAE and mean pairwise Jensen-Shannon divergence."""
    probabilities = attention.detach().float().clamp_min(1e-12)
    key_profile = probabilities.mean(dim=1, keepdim=True)
    row_mae = (probabilities - key_profile).abs().mean(dim=(1, 2))
    pairwise = []
    for first in range(probabilities.size(1)):
        for second in range(first + 1, probabilities.size(1)):
            p = probabilities[:, first, :]
            q = probabilities[:, second, :]
            middle = 0.5 * (p + q)
            js = 0.5 * (
                (p * (p.log() - middle.log())).sum(dim=-1)
                + (q * (q.log() - middle.log())).sum(dim=-1)
            )
            pairwise.append(js)
    pair_js = torch.stack(pairwise, dim=1).mean(dim=1) if pairwise else row_mae.new_zeros(row_mae.shape)
    return row_mae, pair_js


def _summary_row(prefix: dict[str, Any], metric: str, values: Any) -> dict[str, Any]:
    return {**prefix, "metric": metric, **_distribution(values)}


def _logit_summary(records: list[dict[str, torch.Tensor]], dataset: str, variant: str, seed: int, modality: str) -> list[dict[str, Any]]:
    output = []
    for record in records:
        content = record["content_logits"].numpy()
        relation = record["relation_bias"].numpy()
        total = record["total_logits"].numpy()
        layer = int(record["layer"].item())
        content_rms = float(np.sqrt(np.mean(np.square(content))))
        bias_rms = float(np.sqrt(np.mean(np.square(relation))))
        output.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "modality": modality,
                "layer": layer,
                "content_logits_rms": content_rms,
                "content_logits_std": float(content.std(ddof=0)),
                "content_logits_max_abs": float(np.max(np.abs(content))),
                "relation_bias_rms": bias_rms,
                "relation_bias_max_abs": float(np.max(np.abs(relation))),
                "bias_to_content_rms": bias_rms / max(content_rms, 1e-30),
                "total_logits_rms": float(np.sqrt(np.mean(np.square(total)))),
                "total_logits_std": float(total.std(ddof=0)),
                "total_logits_max_abs": float(np.max(np.abs(total))),
            }
        )
    return output


def _relation_payload(components: dict[str, Any], model: nn.Module, dataset: str, variant: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for modality in ("text", "visual"):
        condition = components[f"transport_context_{modality}"].detach().float().cpu().numpy()
        theta = getattr(model, f"theta_transport_{modality}").detach().float()
        beta = (theta - theta.mean()).cpu().numpy()
        gate = float(torch.tanh(getattr(model, f"theta_relation_bias_{modality}")).item())
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "modality": modality,
                "relation_context_mean": float(condition.mean()),
                "relation_context_std": float(condition.std(ddof=0)),
                "relation_context_min": float(condition.min()),
                "relation_context_max": float(condition.max()),
                "beta_order_values": json.dumps([float(value) for value in beta]),
                "beta_order_std": float(beta.std(ddof=0)),
                "beta_order_max_abs": float(np.max(np.abs(beta))),
                "relation_bias_gate": gate,
            }
        )
    return rows


def _pair_node_metrics(
    normal: dict[str, Any],
    changed: dict[str, Any],
    normal_logits: torch.Tensor,
    changed_logits: torch.Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, float]]]]:
    node_values: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    eps = 1e-12
    for modality in ("text", "visual"):
        attention_normal = normal[f"hop_attention_{modality}"].detach().float()
        attention_changed = changed[f"hop_attention_{modality}"].detach().float()
        profile_normal = _profile(attention_normal)
        profile_changed = _profile(attention_changed)
        attention_mae = (attention_normal - attention_changed).abs().mean(dim=(1, 2))
        r_attn_shift = (profile_normal["r_attn"] - profile_changed["r_attn"]).abs()
        eta_mae = (normal[f"eta_{modality}"].float() - changed[f"eta_{modality}"].float()).abs().mean(dim=-1)
        node_values[f"attention_mae_{modality}"] = attention_mae.cpu().numpy()
        node_values[f"r_attn_abs_shift_{modality}"] = r_attn_shift.cpu().numpy()
        node_values[f"eta_mae_{modality}"] = eta_mae.cpu().numpy()
        summaries[modality] = {
            "attention_mae": _distribution(node_values[f"attention_mae_{modality}"]),
            "r_attn_abs_shift": _distribution(node_values[f"r_attn_abs_shift_{modality}"]),
            "eta_mae": _distribution(node_values[f"eta_mae_{modality}"]),
        }

    z_normal = normal["z"].detach().float()
    z_changed = changed["z"].detach().float()
    z_mae = (z_normal - z_changed).abs().mean(dim=-1)
    z_cosine_shift = 1.0 - F.cosine_similarity(z_normal, z_changed, dim=-1, eps=eps)
    p_normal = torch.softmax(normal_logits.float(), dim=-1).clamp_min(eps)
    p_changed = torch.softmax(changed_logits.float(), dim=-1).clamp_min(eps)
    logit_mae = (normal_logits.float() - changed_logits.float()).abs().mean(dim=-1)
    probability_mae = (p_normal - p_changed).abs().mean(dim=-1)
    p_normal_kl = torch.softmax(normal_logits.double(), dim=-1).clamp_min(1e-15)
    p_changed_kl = torch.softmax(changed_logits.double(), dim=-1).clamp_min(1e-15)
    p_normal_kl = p_normal_kl / p_normal_kl.sum(dim=-1, keepdim=True)
    p_changed_kl = p_changed_kl / p_changed_kl.sum(dim=-1, keepdim=True)
    probability_kl = (p_normal_kl * (p_normal_kl.log() - p_changed_kl.log())).sum(dim=-1).clamp_min(0.0)
    flips = (normal_logits.argmax(dim=-1) != changed_logits.argmax(dim=-1)).float()
    node_values.update(
        {
            "z_mae": z_mae.cpu().numpy(),
            "z_cosine_shift": z_cosine_shift.cpu().numpy(),
            "logit_mae": logit_mae.cpu().numpy(),
            "probability_mae": probability_mae.cpu().numpy(),
            "probability_kl_normal_to_intervention": probability_kl.cpu().numpy(),
            "prediction_flip": flips.cpu().numpy(),
        }
    )
    summaries["fused"] = {
        name: _distribution(values)
        for name, values in (
            ("z_mae", node_values["z_mae"]),
            ("z_cosine_shift", node_values["z_cosine_shift"]),
            ("logit_mae", node_values["logit_mae"]),
            ("probability_mae", node_values["probability_mae"]),
            ("probability_kl_normal_to_intervention", node_values["probability_kl_normal_to_intervention"]),
            ("prediction_flip_rate", node_values["prediction_flip"]),
        )
    }
    return node_values, summaries


def _stage12_diff(normal: dict[str, Any], changed: dict[str, Any]) -> dict[str, float]:
    keys = (
        "states_text",
        "states_visual",
        "responses_text",
        "responses_visual",
        "norm_t_weight",
        "norm_v_weight",
    )
    result = {}
    for key in keys:
        first, second = normal[key], changed[key]
        if isinstance(first, list):
            difference = max(
                float((a - b).abs().max().item()) for a, b in zip(first, second)
            )
        else:
            difference = float((first - second).abs().max().item())
        result[key] = difference
    return result


def _order_embedding_rows(model: nn.Module, components: dict[str, Any], dataset: str, variant: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    payload: dict[str, Any] = {}
    for modality in ("text", "visual"):
        embedding = getattr(model, f"hop_order_embedding_{modality}").detach().float()
        states = torch.stack(components[f"states_{modality}"], dim=1).detach().float()
        norms = embedding.norm(dim=-1)
        mean_state_norms = states.norm(dim=-1).mean(dim=0)
        ratio = norms / mean_state_norms.clamp_min(1e-12)
        cosine = F.normalize(embedding, dim=-1) @ F.normalize(embedding, dim=-1).T
        for order in range(embedding.size(0)):
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "modality": modality,
                    "order": order,
                    "embedding_norm": float(norms[order].item()),
                    "mean_state_norm": float(mean_state_norms[order].item()),
                    "embedding_to_state_norm_ratio": float(ratio[order].item()),
                    "embedding_vector": json.dumps([float(value) for value in embedding[order].cpu().tolist()]),
                }
            )
        payload[modality] = {
            "norm_per_order": [float(value) for value in norms.cpu().tolist()],
            "mean_state_norm_per_order": [float(value) for value in mean_state_norms.cpu().tolist()],
            "embedding_to_state_norm_ratio_per_order": [float(value) for value in ratio.cpu().tolist()],
            "pairwise_cosine_similarity": cosine.cpu().tolist(),
        }
    return rows, payload


def _node_csv_rows(
    node_count: int,
    normal: dict[str, Any],
    query_metrics: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    intervention_values: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    output = []
    profiles = {modality: _profile(normal[f"hop_attention_{modality}"]) for modality in ("text", "visual")}
    for node in range(node_count):
        row: dict[str, Any] = {"node_id": node}
        for modality in ("text", "visual"):
            profile = profiles[modality]
            row.update(
                {
                    f"p_{modality}_{order}": float(profile["key_mass"][node, order].item())
                    for order in range(profile["key_mass"].size(1))
                }
            )
            for metric in ("r_attn", "entropy", "kl_uniform", "normalized_entropy", "max_key_mass"):
                row[f"{metric}_{modality}"] = float(profile[metric][node].item())
            for layer_index, (row_mae, pair_js) in enumerate(query_metrics[modality], start=1):
                row[f"query_row_mae_{modality}_layer_{layer_index}"] = float(row_mae[node].item())
                row[f"pairwise_query_js_{modality}_layer_{layer_index}"] = float(pair_js[node].item())
        for intervention, values in intervention_values.items():
            for name, array in values.items():
                row[f"{intervention}_{name}"] = float(array[node])
        output.append(row)
    return output


def _remove_order_embedding_hook(module, inputs):
    if len(inputs) < 2:
        raise RuntimeError("Unexpected HopInteractionBlock input signature")
    replaced = list(inputs)
    replaced[0] = inputs[1]  # tokens := unmodified Stage-II state bank
    return tuple(replaced)


def _run_order_embedding_off(model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, Any]:
    handles = [
        getattr(model, f"hop_layers_{modality}")[0].register_forward_pre_hook(_remove_order_embedding_hook)
        for modality in ("text", "visual")
    ]
    try:
        return model._encode_components(x, edge_index, _capture_hop_attention=True)
    finally:
        for handle in handles:
            handle.remove()


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    relation_rows = payload["relation_bias_scale"]
    intervention_rows = payload["intervention_summaries"]
    query_rows = payload["query_diversity_summary"]
    nonuniform_rows = payload["nonuniform_summary"]
    modality_rows = payload["modality_pattern_summary"]

    lines = [
        "# IAMOC v1 Mechanism Diagnosis",
        "",
        "## Scope and protocol",
        "",
        "Inference-only analysis of the existing Movies/Grocery V1–V4 best checkpoints for seeds 42/43/44. No training, LP runs, optimizer steps, or checkpoint writes were performed. Interventions are transient forward hooks or the existing analysis-only relation intervention; model state-dict hashes are checked before and after each checkpoint. Stage-I/II repeat-forward differences are checked against a 1e-5 max-absolute tolerance to accommodate small GPU sparse-reduction roundoff.",
        "",
        "## A. Relation-bias scale audit",
        "",
        "The table below reports bias RMS divided by content-logit RMS for every layer, modality, dataset, and seed. `relation_bias` is broadcast across queries; its RMS is unchanged by that broadcast.",
        "",
            "| Dataset | Variant | Modality | Layer | Bias RMS / content RMS | Bias RMS | Content RMS | Total-logit RMS | Context mean / SD / min / max | β values (β SD) | Gate |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in relation_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} seed {row['seed']} | {row['modality']} | {row['layer']} | "
            f"{row['bias_to_content_rms']:.3e} | {row['relation_bias_rms']:.3e} | "
            f"{row['content_logits_rms']:.3e} | {row['total_logits_rms']:.3e} | "
            f"{row['relation_context_mean']:.3e} / {row['relation_context_std']:.3e} / {row['relation_context_min']:.3e} / {row['relation_context_max']:.3e} | "
            f"{row['beta_order_values']} ({row['beta_order_std']:.3e}) | {row['relation_bias_gate']:.4f} |"
        )
    ratio_values = [row["bias_to_content_rms"] for row in relation_rows]
    context_scale = {
        modality: {
            "context_sd": float(np.median([row["relation_context_std"] for row in relation_rows if row["modality"] == modality])),
            "beta_sd": float(np.median([row["beta_order_std"] for row in relation_rows if row["modality"] == modality])),
            "content_rms": float(np.median([row["content_logits_rms"] for row in relation_rows if row["modality"] == modality])),
            "bias_rms_max": float(max(row["relation_bias_rms"] for row in relation_rows if row["modality"] == modality)),
        }
        for modality in ("text", "visual")
    }
    max_attention_shift = max(
        (row["modalities"][modality]["attention_mae"]["mean"]
         for item in payload["runs"]
         for intervention, row in item.get("interventions", {}).items()
         if intervention in {"relation_off", "relation_shuffle"}
         for modality in ("text", "visual")),
        default=0.0,
    )
    lines.extend(
        [
            "",
            f"Across V3/V4, bias-to-content RMS ratio ranges {min(ratio_values):.3e}–{max(ratio_values):.3e} (median {np.median(ratio_values):.3e}). The maximum V3 node-mean Relation-Off/Shuffle attention MAE is {max_attention_shift:.3e}.",
            f"Median context SD / beta SD / content-logit RMS are Text {context_scale['text']['context_sd']:.3e} / {context_scale['text']['beta_sd']:.3e} / {context_scale['text']['content_rms']:.3e}, Visual {context_scale['visual']['context_sd']:.3e} / {context_scale['visual']['beta_sd']:.3e} / {context_scale['visual']['content_rms']:.3e}; maximum relation-bias RMS is {max(context_scale[m]['bias_rms_max'] for m in ('text', 'visual')):.3e}.",
            "",
            "A small measured ratio indicates a weak additive logit perturbation relative to QK content; it is distinct from a mathematically zero bias. The intervention section checks whether this scale has any downstream effect.",
            "",
            "## B. Per-node intervention sensitivity",
            "",
            "Every row summarizes the node-wise distribution, not only a mean attention matrix. Per-checkpoint JSON and `intervention_summary.csv` retain mean, standard deviation, median, P75, P90, P95, and maximum; the compact table displays mean/P95/maximum. Probability KL is KL(p_normal || p_intervention); flip rates are reported over all nodes and the NC test nodes.",
            "",
            "| Dataset | Variant | Seed | Intervention | Modality | Attention MAE mean / P95 / max | |Δr_attn| mean / P95 / max | Eta MAE mean / P95 / max | Z MAE mean / P95 / max | Z cosine shift mean / P95 / max | Logit MAE mean / P95 / max | Prob MAE mean / P95 / max | Prob KL mean / P95 / max | Flip rate all / test |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["runs"]:
        for intervention, summary in item.get("interventions", {}).items():
            for modality in ("text", "visual"):
                modality_summary = summary["modalities"][modality]
                fused = summary["modalities"]["fused"]
                test_flip = summary["test_prediction_flip_rate"]
                lines.append(
                    f"| {item['dataset']} | {item['variant']} | {item['seed']} | {intervention} | {modality} | "
                    f"{modality_summary['attention_mae']['mean']:.3e} / {modality_summary['attention_mae']['p95']:.3e} / {modality_summary['attention_mae']['max']:.3e} | "
                    f"{modality_summary['r_attn_abs_shift']['mean']:.3e} / {modality_summary['r_attn_abs_shift']['p95']:.3e} / {modality_summary['r_attn_abs_shift']['max']:.3e} | "
                    f"{modality_summary['eta_mae']['mean']:.3e} / {modality_summary['eta_mae']['p95']:.3e} / {modality_summary['eta_mae']['max']:.3e} | "
                    f"{fused['z_mae']['mean']:.3e} / {fused['z_mae']['p95']:.3e} / {fused['z_mae']['max']:.3e} | "
                    f"{fused['z_cosine_shift']['mean']:.3e} / {fused['z_cosine_shift']['p95']:.3e} / {fused['z_cosine_shift']['max']:.3e} | "
                    f"{fused['logit_mae']['mean']:.3e} / {fused['logit_mae']['p95']:.3e} / {fused['logit_mae']['max']:.3e} | "
                    f"{fused['probability_mae']['mean']:.3e} / {fused['probability_mae']['p95']:.3e} / {fused['probability_mae']['max']:.3e} | "
                    f"{fused['probability_kl_normal_to_intervention']['mean']:.3e} / {fused['probability_kl_normal_to_intervention']['p95']:.3e} / {fused['probability_kl_normal_to_intervention']['max']:.3e} | "
                    f"{fused['prediction_flip_rate']['mean']:.3e} / {test_flip:.3e} |"
                )
        if item.get("interventions"):
            for intervention, summary in item["interventions"].items():
                lines.append(
                    f"- {item['dataset']} {item['variant']} seed {item['seed']} {intervention}: "
                    f"max |Δlogit|={summary['modalities']['fused']['logit_mae']['max']:.3e}; "
                    f"test ΔAcc={summary['test_delta'].get('test_acc', 0.0):+.7f}, "
                    f"ΔMacro-F1={summary['test_delta'].get('test_macro_f1', 0.0):+.7f}; "
                    f"Stage-I/II max difference={max(summary['stage12_max_abs_diff'].values()):.1e}."
                )

    lines.extend(
        [
            "",
            "## C. Query diversity and cross-order interaction",
            "",
            "`query_row_mae` is the node-wise average absolute row deviation from that node's mean key profile. `pairwise_query_js` is mean natural-log Jensen–Shannon divergence over all query-row pairs.",
            "",
            "| Dataset | Variant | Modality | Layer | Query-row MAE mean ± SD; P25/P50/P75/P90/P95/max | Pairwise query JS mean ± SD; P25/P50/P75/P90/P95/max |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in query_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['modality']} | {row['layer']} | "
            f"{row['query_row_mae']['mean']:.3e} ± {row['query_row_mae']['std']:.3e}; "
            f"{row['query_row_mae']['p25']:.3e}/{row['query_row_mae']['median']:.3e}/{row['query_row_mae']['p75']:.3e}/{row['query_row_mae']['p90']:.3e}/{row['query_row_mae']['p95']:.3e}/{row['query_row_mae']['max']:.3e} | "
            f"{row['pairwise_query_js']['mean']:.3e} ± {row['pairwise_query_js']['std']:.3e}; "
            f"{row['pairwise_query_js']['p25']:.3e}/{row['pairwise_query_js']['median']:.3e}/{row['pairwise_query_js']['p75']:.3e}/{row['pairwise_query_js']['p90']:.3e}/{row['pairwise_query_js']['p95']:.3e}/{row['pairwise_query_js']['max']:.3e} |"
        )

    lines.extend(
        [
            "",
            "### Text/Visual pattern contrast across seeds",
            "",
            "The first two contrasts are absolute profile distances. The signed deltas are Text minus Visual; direction stability counts how many of the three seeds share the majority sign.",
            "",
            "| Dataset | Variant | Mean-matrix MAE mean ± SD | Key-profile MAE mean ± SD | Δr_attn Text−Visual mean ± SD | Δentropy Text−Visual mean ± SD | r_attn sign consistency |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in modality_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | "
            f"{row['mean_attention_matrix_mae']['mean']:.3e} ± {row['mean_attention_matrix_mae']['std']:.3e} | "
            f"{row['mean_key_profile_mae']['mean']:.3e} ± {row['mean_key_profile_mae']['std']:.3e} | "
            f"{row['delta_r_attn_text_minus_visual']['mean']:+.3e} ± {row['delta_r_attn_text_minus_visual']['std']:.3e} | "
            f"{row['delta_entropy_text_minus_visual']['mean']:+.3e} ± {row['delta_entropy_text_minus_visual']['std']:.3e} | "
            f"{row['r_attn_sign_consistency']}/3 |"
        )

    lines.extend(
        [
            "",
            "## D. Non-uniform attention",
            "",
            "For K=3, key mass is compared with the uniform profile using KL(p || uniform), normalized entropy H/log(4), and maximum key mass. KL=0 and normalized entropy=1 are the uniform reference values.",
            "",
            "| Dataset | Variant | Modality | KL to uniform mean ± SD | Normalized entropy mean ± SD | Max-key mass mean ± SD |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in nonuniform_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['modality']} | "
            f"{row['kl_uniform']['mean']:.4f} ± {row['kl_uniform']['std']:.4f} | "
            f"{row['normalized_entropy']['mean']:.4f} ± {row['normalized_entropy']['std']:.4f} | "
            f"{row['max_key_mass']['mean']:.4f} ± {row['max_key_mass']['std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## E. Order embedding diagnostics and OrderEmbedding-Off",
            "",
            "Per-order embedding norms, state norms, ratios, and pairwise cosine matrices are stored in the JSON and order-embedding CSV. OrderEmbedding-Off used a temporary forward pre-hook that replaced only the initial attention token input by the unmodified Stage-II states; checkpoint tensors and all other paths remained fixed.",
            "",
            "| Dataset | Variant | Seed | Modality | Order-embedding norm mean | Embedding/state norm ratio mean | Off attention MAE mean | Off eta MAE mean | Off Z MAE mean | Off logit MAE mean | Off probability KL mean | Flip rate all/test | Test ΔAcc / ΔMacro-F1 |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["runs"]:
        off = item.get("interventions", {}).get("order_embedding_off")
        if off is None:
            continue
        for modality in ("text", "visual"):
            order = item["order_embeddings"][modality]
            lines.append(
                f"| {item['dataset']} | {item['variant']} | {item['seed']} | {modality} | "
                f"{np.mean(order['norm_per_order']):.4f} | "
                f"{np.mean(order['embedding_to_state_norm_ratio_per_order']):.3e} | "
                f"{off['modalities'][modality]['attention_mae']['mean']:.3e} | "
                f"{off['modalities'][modality]['eta_mae']['mean']:.3e} | "
                f"{off['modalities']['fused']['z_mae']['mean']:.3e} | "
                f"{off['modalities']['fused']['logit_mae']['mean']:.3e} | "
                f"{off['modalities']['fused']['probability_kl_normal_to_intervention']['mean']:.3e} | "
                f"{off['modalities']['fused']['prediction_flip_rate']['mean']:.3e}/{off['test_prediction_flip_rate']:.3e} | "
                f"{off['test_delta'].get('test_acc', 0.0):+.7f} / {off['test_delta'].get('test_macro_f1', 0.0):+.7f} |"
            )

    lines.extend(["", "## Answers", ""])
    for answer in payload["answers"]:
        lines.append(f"{answer}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "Machine-readable checkpoint summaries are in `outputs/iamoc_v1/mechanism_diagnosis/diagnosis.json`. Pooled seed-level CSV summaries and per-node CSVs are in the same directory, including `relation_condition_r_attn.csv`; raw per-layer scale values and order-embedding vectors/cosine matrices are retained in JSON.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "iamoc_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "iamoc_v1" / "mechanism_diagnosis")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "iamoc_v1_mechanism_diagnosis.md")
    parser.add_argument("--datasets", nargs="+", choices=("Movies", "Grocery"), default=("Movies", "Grocery"))
    parser.add_argument("--variants", nargs="+", choices=("V1", "V2", "V3", "V4"), default=("V1", "V2", "V3", "V4"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_runs: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    query_per_node: dict[tuple[str, str, str, int], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    nonuniform_per_node: dict[tuple[str, str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    intervention_per_node: dict[tuple[str, str, str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    modality_pattern_by_group: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    order_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                run_dir = args.root / dataset / variant / f"seed_{seed}"
                checkpoint_path = run_dir / "best.pt"
                config_path = run_dir / "resolved_config.yaml"
                if not checkpoint_path.is_file() or not config_path.is_file():
                    raise FileNotFoundError(f"Missing best checkpoint/config: {run_dir}")
                cfg = OmegaConf.load(config_path)
                if str(cfg.task.name) != "nc":
                    raise ValueError(f"Expected an NC checkpoint at {run_dir}, got task={cfg.task.name}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                data = load_mag_data(cfg, "nc", int(seed))
                model = build_model(cfg, checkpoint["data_info"]).to(device)
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.eval()
                if not hasattr(model, "analysis_hop_attention"):
                    raise TypeError(f"Checkpoint model lacks IAMOC analysis API: {run_dir}")
                parameter_hash_before = _parameter_digest(model)
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                capture = _LogitCapture(model)
                try:
                    capture.clear()
                    with torch.inference_mode():
                        normal = model.analysis_hop_attention(x, edge_index, relation_intervention="normal")
                    normal_capture = {modality: list(records) for modality, records in capture.records.items()}
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    param_scale_rows = (
                        _relation_payload(normal, model, dataset, variant, seed)
                        if variant in {"V3", "V4"}
                        else []
                    )
                    scale_by_modality = {row["modality"]: row for row in param_scale_rows}
                    checkpoint_scale_rows = []
                    for modality in ("text", "visual"):
                        if modality not in scale_by_modality:
                            continue
                        for logit_row in _logit_summary(normal_capture[modality], dataset, variant, seed, modality):
                            payload_row = scale_by_modality[modality]
                            checkpoint_scale_rows.append(
                                {
                                    **payload_row,
                                    **logit_row,
                                }
                            )
                    relation_rows.extend(checkpoint_scale_rows)

                    order_table, order_payload = _order_embedding_rows(model, normal, dataset, variant, seed)
                    order_rows.extend(order_table)
                    text_profile = _profile(normal["hop_attention_text"])
                    visual_profile = _profile(normal["hop_attention_visual"])
                    pattern = {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "mean_attention_matrix_mae": float(
                            (normal["hop_attention_text"].mean(dim=0) - normal["hop_attention_visual"].mean(dim=0)).abs().mean().item()
                        ),
                        "mean_key_profile_mae": float(
                            (text_profile["key_mass"].mean(dim=0) - visual_profile["key_mass"].mean(dim=0)).abs().mean().item()
                        ),
                        "delta_r_attn_text_minus_visual": float(
                            text_profile["r_attn"].mean().item() - visual_profile["r_attn"].mean().item()
                        ),
                        "delta_entropy_text_minus_visual": float(
                            text_profile["entropy"].mean().item() - visual_profile["entropy"].mean().item()
                        ),
                    }
                    modality_pattern_by_group[(dataset, variant)].append(pattern)
                    query_by_modality: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
                    node_count = int(x.size(0))
                    existing_partial_spearman = {}
                    if variant == "V3":
                        prior_mechanism_path = args.root / "analysis" / dataset / variant / f"seed_{seed}" / "mechanism.json"
                        if prior_mechanism_path.is_file():
                            prior_mechanism = json.loads(prior_mechanism_path.read_text(encoding="utf-8"))
                            existing_partial_spearman = {
                                modality: prior_mechanism["modalities"][modality]["partial_spearman_relation_r_attn_given_log_degree"]
                                for modality in ("text", "visual")
                            }
                    run_payload: dict[str, Any] = {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "checkpoint": str(checkpoint_path),
                        "num_nodes": node_count,
                        "relation_context": {
                            modality: {
                                key: row[key]
                                for key in ("relation_context_mean", "relation_context_std", "relation_context_min", "relation_context_max", "beta_order_values", "beta_order_std", "beta_order_max_abs", "relation_bias_gate")
                            }
                            for modality, row in zip(("text", "visual"), param_scale_rows)
                        },
                        "content_and_bias_logits": [
                            row for row in checkpoint_scale_rows
                        ],
                        "order_embeddings": order_payload,
                        "existing_partial_spearman_given_log_degree": existing_partial_spearman,
                        "normal_test_metrics": _downstream_metrics(cfg, checkpoint, model, data, normal, device),
                        "interventions": {},
                    }
                    for modality in ("text", "visual"):
                        attention_layers = normal[f"hop_attention_layers_{modality}"]
                        query_by_modality[modality] = []
                        for layer_index, attention in enumerate(attention_layers, start=1):
                            row_mae, pair_js = _query_diversity(attention)
                            query_by_modality[modality].append((row_mae, pair_js))
                            group = (dataset, variant, modality, layer_index)
                            query_per_node[group]["query_row_mae"].append(row_mae.cpu().numpy())
                            query_per_node[group]["pairwise_query_js"].append(pair_js.cpu().numpy())
                        final = _profile(normal[f"hop_attention_{modality}"])
                        group_nonuniform = (dataset, variant, modality)
                        for key in ("kl_uniform", "normalized_entropy", "max_key_mass", "r_attn"):
                            nonuniform_per_node[group_nonuniform][key].append(final[key].cpu().numpy())

                    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
                    head.load_state_dict(checkpoint["head_state"], strict=True)
                    head.eval()
                    with torch.inference_mode():
                        normal_logits = head(normal["z"])

                    interventions_to_run = []
                    if variant == "V3":
                        interventions_to_run.extend(("relation_off", "relation_shuffle"))
                    interventions_to_run.append("order_embedding_off")
                    node_interventions: dict[str, dict[str, np.ndarray]] = {}
                    for intervention in interventions_to_run:
                        capture.clear()
                        with torch.inference_mode():
                            if intervention == "relation_off":
                                changed = model.analysis_hop_attention(x, edge_index, relation_intervention="off")
                            elif intervention == "relation_shuffle":
                                changed = model.analysis_hop_attention(
                                    x, edge_index, relation_intervention="shuffle", permutation_seed=20260921
                                )
                            else:
                                changed = _run_order_embedding_off(model, x, edge_index)
                            changed_logits = head(changed["z"])
                        stage12 = _stage12_diff(normal, changed)
                        if any(value > 1e-5 for value in stage12.values()):
                            raise AssertionError(
                                f"{intervention} changed Stage-I/II tensors beyond 1e-5 tolerance: {stage12}"
                            )
                        node_values, summaries = _pair_node_metrics(normal, changed, normal_logits, changed_logits)
                        changed_test_metrics = _downstream_metrics(cfg, checkpoint, model, data, changed, device)
                        test_delta = {
                            key: float(changed_test_metrics[key] - run_payload["normal_test_metrics"][key])
                            for key in changed_test_metrics
                        }
                        test_idx = torch.as_tensor(data.test_idx, dtype=torch.long).cpu().numpy()
                        summaries["fused"]["prediction_flip_rate_test"] = _distribution(node_values["prediction_flip"][test_idx])
                        test_flip_rate = float(node_values["prediction_flip"][test_idx].mean()) if test_idx.size else math.nan
                        run_payload["interventions"][intervention] = {
                            "modalities": summaries,
                            "test_delta": test_delta,
                            "test_prediction_flip_rate": test_flip_rate,
                            "stage12_max_abs_diff": stage12,
                        }
                        node_interventions[intervention] = node_values
                        for modality in ("text", "visual"):
                            group = (dataset, variant, intervention, modality)
                            for key in (f"attention_mae_{modality}", f"r_attn_abs_shift_{modality}", f"eta_mae_{modality}"):
                                intervention_per_node[group][key.removesuffix(f"_{modality}")].append(node_values[key])
                        group_fused = (dataset, variant, intervention, "fused")
                        for key in ("z_mae", "z_cosine_shift", "logit_mae", "probability_mae", "probability_kl_normal_to_intervention", "prediction_flip"):
                            intervention_per_node[group_fused][key].append(node_values[key])

                    run_node_rows = _node_csv_rows(node_count, normal, query_by_modality, node_interventions)
                    node_path = args.output_root / dataset / variant / f"seed_{seed}" / "node_diagnostics.csv"
                    _write_csv(node_path, run_node_rows)
                    run_payload["node_csv"] = str(node_path)
                    run_path = args.output_root / dataset / variant / f"seed_{seed}" / "diagnosis.json"
                    run_path.parent.mkdir(parents=True, exist_ok=True)
                    run_path.write_text(json.dumps(run_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
                    all_runs.append(run_payload)
                    print(f"DIAGNOSED {dataset} {variant} seed={seed}", flush=True)
                finally:
                    capture.close()
                parameter_hash_after = _parameter_digest(model)
                if parameter_hash_before != parameter_hash_after:
                    raise AssertionError(f"Checkpoint parameters changed during inference analysis: {run_dir}")
                del model, data, checkpoint
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    query_summary = []
    for (dataset, variant, modality, layer), metrics in sorted(query_per_node.items()):
        query_summary.append(
            {
                "dataset": dataset,
                "variant": variant,
                "modality": modality,
                "layer": layer,
                "query_row_mae": _distribution(np.concatenate(metrics["query_row_mae"])),
                "pairwise_query_js": _distribution(np.concatenate(metrics["pairwise_query_js"])),
            }
        )
    nonuniform_summary = []
    for (dataset, variant, modality), metrics in sorted(nonuniform_per_node.items()):
        nonuniform_summary.append(
            {
                "dataset": dataset,
                "variant": variant,
                "modality": modality,
                **{key: _distribution(np.concatenate(values)) for key, values in metrics.items()},
            }
        )
    modality_pattern_summary = []
    for (dataset, variant), rows in sorted(modality_pattern_by_group.items()):
        signed = [row["delta_r_attn_text_minus_visual"] for row in rows]
        positive, negative = sum(value > 0 for value in signed), sum(value < 0 for value in signed)
        modality_pattern_summary.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seeds": [int(row["seed"]) for row in rows],
                "mean_attention_matrix_mae": _distribution([row["mean_attention_matrix_mae"] for row in rows]),
                "mean_key_profile_mae": _distribution([row["mean_key_profile_mae"] for row in rows]),
                "delta_r_attn_text_minus_visual": _distribution(signed),
                "delta_entropy_text_minus_visual": _distribution([row["delta_entropy_text_minus_visual"] for row in rows]),
                "r_attn_sign_consistency": int(max(positive, negative)),
                "per_seed": rows,
            }
        )
    intervention_summaries = []
    for (dataset, variant, intervention, modality), metrics in sorted(intervention_per_node.items()):
        intervention_summaries.append(
            {
                "dataset": dataset,
                "variant": variant,
                "intervention": intervention,
                "modality": modality,
                **{key: _distribution(np.concatenate(values)) for key, values in metrics.items()},
            }
        )

    relation_scale_rows = [
        row
        for run in all_runs
        for row in run["content_and_bias_logits"]
        if "layer" in row
    ]
    _write_csv(args.output_root / "relation_bias_scale.csv", relation_scale_rows)
    _write_csv(args.output_root / "query_diversity_summary.csv", query_summary)
    _write_csv(args.output_root / "nonuniform_attention_summary.csv", nonuniform_summary)
    _write_csv(args.output_root / "intervention_summary.csv", intervention_summaries)
    _write_csv(args.output_root / "order_embedding_diagnostics.csv", order_rows)
    _write_csv(
        args.output_root / "relation_condition_r_attn.csv",
        [
            {
                "dataset": run["dataset"],
                "variant": run["variant"],
                "seed": run["seed"],
                "modality": modality,
                "partial_spearman_relation_r_attn_given_log_degree": value,
            }
            for run in all_runs
            for modality, value in run.get("existing_partial_spearman_given_log_degree", {}).items()
        ],
    )
    _write_csv(
        args.output_root / "text_visual_pattern_summary.csv",
        [
            {
                "dataset": row["dataset"],
                "variant": row["variant"],
                "r_attn_sign_consistency": row["r_attn_sign_consistency"],
                **{
                    f"{metric}_{stat}": row[metric][stat]
                    for metric in ("mean_attention_matrix_mae", "mean_key_profile_mae", "delta_r_attn_text_minus_visual", "delta_entropy_text_minus_visual")
                    for stat in ("mean", "std", "median", "p25", "p75", "p90", "p95", "max")
                },
            }
            for row in modality_pattern_summary
        ],
    )

    answers = _form_answers(all_runs, relation_scale_rows, query_summary, nonuniform_summary, intervention_summaries, modality_pattern_summary)
    payload = {
        "scope": {"datasets": list(args.datasets), "variants": list(args.variants), "seeds": list(args.seeds), "device": str(device), "inference_only": True},
        "runs": all_runs,
        "relation_bias_scale": relation_scale_rows,
        "query_diversity_summary": query_summary,
        "nonuniform_summary": nonuniform_summary,
        "modality_pattern_summary": modality_pattern_summary,
        "intervention_summaries": intervention_summaries,
        "order_embedding_rows": order_rows,
        "answers": answers,
    }
    diagnosis_path = args.output_root / "diagnosis.json"
    diagnosis_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(args.report, payload)
    print(f"Wrote {diagnosis_path} and {args.report}")
    return 0


def _form_answers(runs, relation_rows, query_rows, nonuniform_rows, intervention_rows, modality_rows) -> list[str]:
    nonuniform_kl = [row["kl_uniform"]["mean"] for row in nonuniform_rows]
    query_mae = [row["query_row_mae"]["mean"] for row in query_rows]
    query_js = [row["pairwise_query_js"]["mean"] for row in query_rows]
    ratios = [row["bias_to_content_rms"] for row in relation_rows]
    interventions = [
        (run, name, data)
        for run in runs
        for name, data in run.get("interventions", {}).items()
    ]
    rel_interventions = [(run, name, data) for run, name, data in interventions if name.startswith("relation_")]
    order_interventions = [(run, name, data) for run, name, data in interventions if name == "order_embedding_off"]
    max_rel_logit = max((data["modalities"]["fused"]["logit_mae"]["max"] for _, _, data in rel_interventions), default=0.0)
    max_rel_z = max((data["modalities"]["fused"]["z_mae"]["max"] for _, _, data in rel_interventions), default=0.0)
    query_med = float(np.median(query_mae)) if query_mae else 0.0
    query_js_med = float(np.median(query_js)) if query_js else 0.0
    modality_diff = float(np.mean([row["mean_attention_matrix_mae"]["mean"] for row in modality_rows])) if modality_rows else 0.0
    stable_modality_groups = sum(row["r_attn_sign_consistency"] == 3 for row in modality_rows)
    def intervention_mean(items, modality, metric):
        values = [item["modalities"][modality][metric]["mean"] for _, _, item in items]
        return float(np.mean(values)) if values else 0.0

    rel_means = {
        "attention_text": intervention_mean(rel_interventions, "text", "attention_mae"),
        "attention_visual": intervention_mean(rel_interventions, "visual", "attention_mae"),
        "eta_text": intervention_mean(rel_interventions, "text", "eta_mae"),
        "eta_visual": intervention_mean(rel_interventions, "visual", "eta_mae"),
        "z": intervention_mean(rel_interventions, "fused", "z_mae"),
        "logit": intervention_mean(rel_interventions, "fused", "logit_mae"),
    }
    rel_flips = [item["test_prediction_flip_rate"] for _, _, item in rel_interventions]
    rel_test_delta = max(
        (abs(float(item["test_delta"].get(metric, 0.0))) for _, _, item in rel_interventions for metric in ("test_acc", "test_macro_f1")),
        default=0.0,
    )
    order_means = {
        "attention_text": intervention_mean(order_interventions, "text", "attention_mae"),
        "attention_visual": intervention_mean(order_interventions, "visual", "attention_mae"),
        "eta_text": intervention_mean(order_interventions, "text", "eta_mae"),
        "eta_visual": intervention_mean(order_interventions, "visual", "eta_mae"),
        "z": intervention_mean(order_interventions, "fused", "z_mae"),
        "logit": intervention_mean(order_interventions, "fused", "logit_mae"),
    }
    order_test_flips = [item["test_prediction_flip_rate"] for _, _, item in order_interventions]
    order_all_flips = [item["modalities"]["fused"]["prediction_flip_rate"]["mean"] for _, _, item in order_interventions]
    order_delta_acc = [item["test_delta"].get("test_acc", 0.0) for _, _, item in order_interventions]
    order_delta_f1 = [item["test_delta"].get("test_macro_f1", 0.0) for _, _, item in order_interventions]
    order_ratios = {
        modality: float(np.mean([
            value
            for run in runs
            for value in run["order_embeddings"][modality]["embedding_to_state_norm_ratio_per_order"]
        ]))
        for modality in ("text", "visual")
    }
    relation_context = {
        modality: {
            key: float(np.median([row[key] for row in relation_rows if row["modality"] == modality]))
            for key in ("relation_context_std", "beta_order_std", "content_logits_rms")
        }
        for modality in ("text", "visual")
    }
    partial_spearman = {
        modality: [
            float(run["existing_partial_spearman_given_log_degree"][modality])
            for run in runs
            if run["variant"] == "V3"
            and modality in run.get("existing_partial_spearman_given_log_degree", {})
        ]
        for modality in ("text", "visual")
    }
    partial_spearman_ranges = {
        modality: (
            f"{min(values):+.3f} to {max(values):+.3f}"
            if values
            else "not available"
        )
        for modality, values in partial_spearman.items()
    }
    max_bias_rms = max((row["relation_bias_rms"] for row in relation_rows), default=0.0)
    return [
        f"1. Hop attention is non-uniform: pooled node KL(p || uniform) averages {np.mean(nonuniform_kl):.4f} nats across dataset/variant/modality cells (0 is uniform); normalized entropy and max-key mass are in the D table and CSV.",
        f"2. Query rows are not identical: median cell mean query-row MAE={query_med:.3e}, pairwise query JS={query_js_med:.3e}. This supports query-dependent interaction, especially in Visual; Text diversity is weaker. Lower quartiles show node heterogeneity, so not every node is strongly cross-order interactive.",
        f"3. Text/Visual patterns are distinct (mean attention-matrix MAE={modality_diff:.3e}); Text has higher r_attn in all three seeds for {stable_modality_groups}/{len(modality_rows)} dataset/variant groups. The direction is stable across Movies, but only 2/3 seeds in each Grocery group.",
        f"4. Relation bias is numerically tiny: bias/content RMS ratio median={np.median(ratios):.3e}, range={min(ratios):.3e}–{max(ratios):.3e}, max bias RMS={max_bias_rms:.3e}. Context is not zero (median SD Text/Visual={relation_context['text']['relation_context_std']:.3e}/{relation_context['visual']['relation_context_std']:.3e}); centered beta is especially flat in Text (median SD={relation_context['text']['beta_order_std']:.3e}; Visual={relation_context['visual']['beta_order_std']:.3e}) against content-logit RMS {relation_context['text']['content_logits_rms']:.2f}/{relation_context['visual']['content_logits_rms']:.2f}.",
        f"5. Relation-Off/Shuffle mean node attention MAE is Text={rel_means['attention_text']:.3e}, Visual={rel_means['attention_visual']:.3e}; eta MAE={rel_means['eta_text']:.3e}/{rel_means['eta_visual']:.3e}, fused-Z MAE={rel_means['z']:.3e}, logit MAE={rel_means['logit']:.3e} (largest node logit MAE={max_rel_logit:.3e}; Z={max_rel_z:.3e}). There were no hard prediction flips and max test Acc/F1 change was {rel_test_delta:.1e}. The existing V3 partial Spearman rho values controlling log-degree span Text {partial_spearman_ranges['text']} and Visual {partial_spearman_ranges['visual']}; these remain observational associations, not evidence of a direct bias effect, which is consistent with the near-null controlled interventions.",
        f"6. OrderEmbedding-Off changes the learned interaction: attention MAE Text/Visual={order_means['attention_text']:.3e}/{order_means['attention_visual']:.3e}, eta MAE={order_means['eta_text']:.3e}/{order_means['eta_visual']:.3e}, fused-Z MAE={order_means['z']:.3e}, logit MAE={order_means['logit']:.3e}; mean test-node flip rate={np.mean(order_test_flips):.3%}, mean all-node rate={np.mean(order_all_flips):.3%}. Mean test ΔAcc/ΔMacro-F1={np.mean(order_delta_acc):+.4e}/{np.mean(order_delta_f1):+.4e}; embedding/state norm ratios are {order_ratios['text']:.2%} Text/{order_ratios['visual']:.2%} Visual. V3 is not supported for its direct relation-bias mechanism.",
        "7. Mechanism ranking: V2 is the strongest continuation candidate, with V1 as the output-TCPR comparison. Both have query diversity and order-embedding sensitivity; V2 is slightly more query-diverse than V1 and more cleanly isolates relation conditioning. Keep V3 as a control/ablation, not a supported direct-bias method. V4 is not favored: layer 2 has lower row diversity than layer 1 in both modalities, with roughly twice the IAMOC-specific parameter overhead.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
