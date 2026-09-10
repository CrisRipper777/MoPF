"""Frozen functional audit for the existing M1-C PDC NC checkpoints.

This script never trains, edits, or resaves an NC checkpoint.  It loads the
best-validation checkpoints from ``m1c_pdc_ppc`` and applies PDC masks through
the analysis-only MoPF helper.  The resulting artifacts are the M1-D frozen
functional audit and its cross-seed summary.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
K_BY_DATASET = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
}
SHUFFLE_SEEDS = tuple(range(10))
EPS = 1e-8
CONDITION_NAMES = ("F0_pdc_on", "F1_all_pdc_off", "F2_text_pdc_off", "F3_visual_pdc_off")


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _summary(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float | int | None]:
    array = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    array = array.astype(np.float64, copy=False).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "q10": float(np.quantile(finite, 0.10)),
        "q25": float(np.quantile(finite, 0.25)),
        "q50": float(np.quantile(finite, 0.50)),
        "q75": float(np.quantile(finite, 0.75)),
        "q90": float(np.quantile(finite, 0.90)),
        "max": float(finite.max()),
    }


def _mean_std(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if array.size else 0.0,
        "population_std": float(array.std(ddof=0)) if array.size else 0.0,
        "n": int(array.size),
    }


def _resolve_eval_labels(data) -> list[int]:
    indices = [idx for idx in (data.train_idx, data.val_idx, data.test_idx) if idx is not None]
    if data.y is None or data.num_classes is None or not indices:
        raise ValueError("NC data must provide labels, classes, and supervised splits")
    labels = data.y[torch.cat(indices).reshape(-1)].detach().cpu()
    valid = labels[(labels >= 0) & (labels < int(data.num_classes))]
    return sorted({int(value) for value in valid.tolist()})


def _split_metrics(
    classifier: nn.Module,
    z: torch.Tensor,
    data,
    index: torch.Tensor,
    device: torch.device,
    eval_labels: list[int],
) -> dict[str, float]:
    logits = classifier(z.index_select(0, index.to(device)))
    target = data.y.index_select(0, index.cpu()).to(device)
    pred = logits.argmax(dim=-1)
    return {
        "acc": float((pred == target).float().mean().item()),
        "macro_f1": float(
            f1_score(
                target.cpu().numpy(),
                pred.cpu().numpy(),
                labels=eval_labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _metrics(classifier, z, data, device, eval_labels) -> dict[str, float]:
    val = _split_metrics(classifier, z, data, data.val_idx, device, eval_labels)
    test = _split_metrics(classifier, z, data, data.test_idx, device, eval_labels)
    return {
        "val_acc": val["acc"],
        "val_macro_f1": val["macro_f1"],
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
    }


def _load_bundle(checkpoint_dir: Path, device: torch.device):
    checkpoint = checkpoint_dir / "best_val_accuracy.pt"
    config_path = checkpoint_dir / ".hydra" / "config.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=True))
    data = load_mag_data(cfg, "nc", int(payload["seed"]))
    model = build_model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"])
    classifier.eval()
    return payload, cfg, data, model, classifier


def _condition_masks(max_order: int) -> dict[str, tuple[list[float], list[float]]]:
    ones = [1.0] * (max_order + 1)
    zeros = [0.0] * (max_order + 1)
    masks: dict[str, tuple[list[float], list[float]]] = {
        "F0_pdc_on": (ones, ones),
        "F1_all_pdc_off": (zeros, zeros),
        "F2_text_pdc_off": (zeros, ones),
        "F3_visual_pdc_off": (ones, zeros),
    }
    for order in range(1, max_order + 1):
        text = list(ones)
        visual = list(ones)
        text[order] = 0.0
        visual[order] = 0.0
        masks[f"F{order + 3}_order{order}_off"] = (text, visual)
    return masks


def _pdc_path_utilization(model, components) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality in ("text", "visual"):
        bases = components[f"bases_{modality}"]
        rho = model.pdc_rho(modality).detach()
        orders = []
        for order, base in enumerate(bases):
            if order == 0:
                correction = torch.zeros_like(base)
            else:
                discrepancy = base - bases[order - 1]
                discrepancy_norm = F.layer_norm(
                    discrepancy,
                    (discrepancy.size(-1),),
                    weight=None,
                    bias=None,
                    eps=model.eps,
                )
                correction = rho[order] * discrepancy_norm
            ratio = torch.linalg.vector_norm(correction, dim=-1) / (
                torch.linalg.vector_norm(base, dim=-1) + EPS
            )
            orders.append(
                {
                    "order": order,
                    "learned_rho": float(rho[order].item()),
                    "correction_magnitude": _summary(torch.linalg.vector_norm(correction, dim=-1)),
                    "pdc_input_ratio": _summary(ratio),
                }
            )
        result[modality] = {"orders": orders}
    return result


def _delta_changes(on, off) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for modality in ("text", "visual"):
        on_delta = on[f"delta_node_{modality}"]
        off_delta = off[f"delta_node_{modality}"]
        rows = []
        for order in range(on_delta.size(1)):
            diff = on_delta[:, order] - off_delta[:, order]
            centered_on = on_delta[:, order] - on_delta[:, order].mean()
            rms = torch.sqrt(diff.square().mean())
            rows.append(
                {
                    "order": order,
                    "rms_delta_change": float(rms.item()),
                    "mean_abs_delta_change": float(diff.abs().mean().item()),
                    "delta_relative_change": float(
                        (rms / (torch.sqrt(centered_on.square().mean()) + EPS)).item()
                    ),
                }
            )
        result[modality] = {"orders": rows}
    return result


def _representation_change(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    diff = first - second
    return {
        "mean_cosine_distance": float(
            (1.0 - F.cosine_similarity(first, second, dim=-1, eps=EPS)).mean().item()
        ),
        "mean_relative_l2": float(
            (torch.linalg.vector_norm(diff, dim=-1)
             / (torch.linalg.vector_norm(first, dim=-1) + EPS)).mean().item()
        ),
    }


def _logit_probability_change(
    classifier: nn.Module,
    on_z: torch.Tensor,
    off_z: torch.Tensor,
    data,
) -> dict[str, Any]:
    on_logits = classifier(on_z)
    off_logits = classifier(off_z)
    on_prob = on_logits.softmax(dim=-1)
    off_prob = off_logits.softmax(dim=-1)
    midpoint = 0.5 * (on_prob + off_prob)
    jsd = 0.5 * (
        (on_prob * (on_prob.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())).sum(dim=-1)
        + (off_prob * (off_prob.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())).sum(dim=-1)
    )
    flip = on_logits.argmax(dim=-1) != off_logits.argmax(dim=-1)

    def summarize(index: torch.Tensor) -> dict[str, float]:
        idx = index.to(on_z.device)
        logit_a = on_logits.index_select(0, idx)
        logit_b = off_logits.index_select(0, idx)
        prob_a = on_prob.index_select(0, idx)
        prob_b = off_prob.index_select(0, idx)
        jsd_part = jsd.index_select(0, idx)
        flip_part = flip.index_select(0, idx)
        return {
            "mean_relative_logit_l2": float(
                (torch.linalg.vector_norm(logit_a - logit_b, dim=-1)
                 / (torch.linalg.vector_norm(logit_a, dim=-1) + EPS)).mean().item()
            ),
            "mean_absolute_probability_change": float((prob_a - prob_b).abs().mean().item()),
            "jensen_shannon_divergence": float(jsd_part.mean().item()),
            "prediction_flip_rate": float(flip_part.float().mean().item()),
        }

    return {
        "all": summarize(torch.arange(on_z.size(0), device=on_z.device)),
        "val": summarize(data.val_idx),
        "test": summarize(data.test_idx),
    }


def _compose_with_profile(model, components, delta_text, delta_visual) -> torch.Tensor:
    eta_text = components["eta_text"] - components["delta_node_text"] + delta_text
    eta_visual = components["eta_visual"] - components["delta_node_visual"] + delta_visual
    z_text = model._filter_bases(components["bases_text"], eta_text)
    z_visual = model._filter_bases(components["bases_visual"], eta_visual)
    z_text_refined = model.text_refine_norm(z_text + model.text_refine_mlp(z_text))
    z_visual_refined = model.visual_refine_norm(z_visual + model.visual_refine_mlp(z_visual))
    fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
    z = model.output_norm(model.fusion_skip(fused_input) + model.fusion_mlp(fused_input))
    return torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)


def _with_drop(metrics: dict[str, float], original: dict[str, float]) -> dict[str, float]:
    result = dict(metrics)
    for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        result[f"delta_{metric}"] = original[metric] - metrics[metric]
    return result


def _alignment_for_condition(model, classifier, components, data, device, eval_labels) -> dict[str, Any]:
    original = _metrics(classifier, components["z"], data, device, eval_labels)
    delta_text = components["delta_node_text"]
    delta_visual = components["delta_node_visual"]
    mean_text = delta_text.mean(dim=0, keepdim=True).expand_as(delta_text)
    mean_visual = delta_visual.mean(dim=0, keepdim=True).expand_as(delta_visual)
    node_mean = _with_drop(
        _metrics(
            classifier,
            _compose_with_profile(model, components, mean_text, mean_visual),
            data,
            device,
            eval_labels,
        ),
        original,
    )
    permutations = []
    for seed in SHUFFLE_SEEDS:
        text_generator = torch.Generator(device="cpu").manual_seed(2 * seed)
        visual_generator = torch.Generator(device="cpu").manual_seed(2 * seed + 1)
        text_perm = torch.randperm(delta_text.size(0), generator=text_generator).to(device)
        visual_perm = torch.randperm(delta_visual.size(0), generator=visual_generator).to(device)
        shuffled = _compose_with_profile(
            model,
            components,
            delta_text.index_select(0, text_perm),
            delta_visual.index_select(0, visual_perm),
        )
        permutations.append(
            {
                "permutation_seed": seed,
                "metrics": _with_drop(
                    _metrics(classifier, shuffled, data, device, eval_labels),
                    original,
                ),
            }
        )
    mean_metrics = {
        metric: float(np.mean([row["metrics"][metric] for row in permutations]))
        for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }
    mean_drop = _with_drop(mean_metrics, original)
    return {
        "original": original,
        "node_mean": node_mean,
        "node_shuffle": {
            "permutation_mode": "independent text and visual complete-profile permutations",
            "permutation_seeds": list(SHUFFLE_SEEDS),
            "permutations": permutations,
            "mean_metrics": mean_metrics,
            "mean_drop": mean_drop,
            "std_metrics": {
                metric: float(np.std([row["metrics"][metric] for row in permutations], ddof=0))
                for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
            },
        },
    }


@torch.no_grad()
def _audit_one(dataset: str, seed: int, checkpoint_root: Path, output_root: Path, device: torch.device, force: bool) -> dict[str, Any]:
    run_dir = output_root / dataset / f"seed{seed}"
    audit_path = run_dir / "audit.json"
    if audit_path.is_file() and not force:
        return json.loads(audit_path.read_text(encoding="utf-8"))

    checkpoint_dir = checkpoint_root / dataset / "v1_pdc" / f"seed{seed}"
    payload, cfg, data, model, classifier = _load_bundle(checkpoint_dir, device)
    if str(cfg.model.get("node_conditioner_mode", "")) != "pdc":
        raise ValueError(f"Expected PDC checkpoint, got {cfg.model.get('node_conditioner_mode')}")
    eval_labels = _resolve_eval_labels(data)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    max_order = int(cfg.model.max_order)
    masks = _condition_masks(max_order)

    formal, _, _, _, _ = model(x, edge_index)
    f0 = model.analysis_encode_with_pdc_mask(
        x, edge_index, text_mask=masks["F0_pdc_on"][0], visual_mask=masks["F0_pdc_on"][1]
    )
    formal_error = float((formal - f0["z"]).abs().max().item())
    conditions: dict[str, dict[str, Any]] = {}
    component_cache: dict[str, Any] = {"F0_pdc_on": f0}
    for name, (text_mask, visual_mask) in masks.items():
        components = f0 if name == "F0_pdc_on" else model.analysis_encode_with_pdc_mask(
            x, edge_index, text_mask=text_mask, visual_mask=visual_mask
        )
        if name != "F0_pdc_on":
            component_cache[name] = components
        conditions[name] = {
            "text_mask": text_mask,
            "visual_mask": visual_mask,
            "metrics": _metrics(classifier, components["z"], data, device, eval_labels),
        }

    # Keep F0 at zero and make every intervention delta have the requested
    # PDC-On minus intervention sign.
    for name in conditions:
        conditions[name]["delta_pdc_on_minus_condition"] = {
            metric: conditions["F0_pdc_on"]["metrics"][metric] - conditions[name]["metrics"][metric]
            for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
        }

    f1 = component_cache["F1_all_pdc_off"]
    delta_changes = _delta_changes(f0, f1)
    eta_change = f0["eta_text"] - f1["eta_text"]
    eta_change_visual = f0["eta_visual"] - f1["eta_visual"]
    eta_delta_error = max(
        float((eta_change - (f0["delta_node_text"] - f1["delta_node_text"])).abs().max().item()),
        float((eta_change_visual - (f0["delta_node_visual"] - f1["delta_node_visual"])).abs().max().item()),
    )
    stream_changes = {
        "node_residual_change": delta_changes,
        "eta_change": {
            "text": eta_change.abs().mean(dim=0).detach().cpu().tolist(),
            "visual": eta_change_visual.abs().mean(dim=0).detach().cpu().tolist(),
        },
        "eta_change_equals_delta_change_max_abs_error": eta_delta_error,
        "modality_representation_change": {
            "text": _representation_change(f0["z_text"], f1["z_text"]),
            "visual": _representation_change(f0["z_visual"], f1["z_visual"]),
        },
        "fused_representation_change": _representation_change(f0["z"], f1["z"]),
        "logit_probability_change": _logit_probability_change(
            classifier, f0["z"], f1["z"], data
        ),
    }
    alignment: dict[str, Any] = {}
    for name in ("F0_pdc_on", "F1_all_pdc_off"):
        alignment[name] = _alignment_for_condition(
            model, classifier, component_cache[name], data, device, eval_labels
        )

    common = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint": str(checkpoint_dir / "best_val_accuracy.pt"),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_selection": payload.get("selection", "best Validation Accuracy"),
        "max_order": max_order,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "eval_labels": eval_labels,
        "path_utilization": _pdc_path_utilization(model, f0),
        "formal_forward_max_abs_error": formal_error,
        "conditions": conditions,
        "stream_changes_f0_vs_f1": stream_changes,
        "alignment": alignment,
        "restrictions": {
            "training_performed": False,
            "checkpoint_modified": False,
            "formal_config_modified": False,
            "semantic_graph_modified": False,
            "polynomial_bank_modified": False,
            "eta_formula_modified": False,
            "fusion_modified": False,
            "classifier_modified": False,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(run_dir / "path_utilization.json", common["path_utilization"])
    _dump_json(run_dir / "stream_changes.json", stream_changes)
    _dump_json(run_dir / "alignment.json", alignment)
    _dump_json(audit_path, common)
    del model, classifier, data, f0, f1, component_cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return common


def _current_alignment_from_m1c(path: Path, dataset: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload["dataset_variant_aggregates"][f"{dataset}/v0_current"]
    shuffle = aggregate["frozen_counterfactual"]["node_shuffle"]
    return {
        metric: shuffle[f"delta_{metric}_across_training_seeds"]
        for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }


def _aggregate_dataset(records: list[dict[str, Any]], current_alignment: dict[str, Any]) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    all_names = list(records[0]["conditions"])
    for name in all_names:
        conditions[name] = {
            "downstream": {
                metric: {
                    **_mean_std([record["conditions"][name]["metrics"][metric] for record in records]),
                    "positive_sign_count": int(sum(
                        record["conditions"][name]["delta_pdc_on_minus_condition"][metric] > 0
                        for record in records
                    )),
                }
                for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
            },
            "intervention_delta_pdc_on_minus_condition": {
                metric: {
                    **_mean_std([
                        record["conditions"][name]["delta_pdc_on_minus_condition"][metric]
                        for record in records
                    ]),
                    "positive_sign_count": int(sum(
                        record["conditions"][name]["delta_pdc_on_minus_condition"][metric] > 0
                        for record in records
                    )),
                }
                for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
            },
        }

    alignment: dict[str, Any] = {}
    for condition in ("F0_pdc_on", "F1_all_pdc_off"):
        alignment[condition] = {}
        for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
            values = [
                record["alignment"][condition]["node_shuffle"]["mean_drop"][f"delta_{metric}"]
                for record in records
            ]
            alignment[condition][metric] = {
                **_mean_std(values),
                "positive_sign_count": int(sum(value > 0 for value in values)),
            }
    gain: dict[str, Any] = {}
    for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        values = [
            record["alignment"]["F0_pdc_on"]["node_shuffle"]["mean_drop"][f"delta_{metric}"]
            - record["alignment"]["F1_all_pdc_off"]["node_shuffle"]["mean_drop"][f"delta_{metric}"]
            for record in records
        ]
        gain[metric] = {
            **_mean_std(values),
            "positive_sign_count": int(sum(value > 0 for value in values)),
        }

    representation = {}
    for key in (
        "fused_representation_change",
        "modality_representation_change",
        "logit_probability_change",
    ):
        if key == "modality_representation_change":
            representation[key] = {
                modality: {
                    metric: _mean_std([
                        record["stream_changes_f0_vs_f1"][key][modality][metric]
                        for record in records
                    ])
                    for metric in ("mean_cosine_distance", "mean_relative_l2")
                }
                for modality in ("text", "visual")
            }
        elif key == "fused_representation_change":
            representation[key] = {
                metric: _mean_std([
                    record["stream_changes_f0_vs_f1"][key][metric] for record in records
                ])
                for metric in ("mean_cosine_distance", "mean_relative_l2")
            }
        else:
            representation[key] = {
                split: {
                    metric: _mean_std([
                        record["stream_changes_f0_vs_f1"][key][split][metric]
                        for record in records
                    ])
                    for metric in (
                        "mean_relative_logit_l2",
                        "mean_absolute_probability_change",
                        "jensen_shannon_divergence",
                        "prediction_flip_rate",
                    )
                }
                for split in ("all", "val", "test")
            }

    return {
        "n_training_seeds": len(records),
        "seeds": [record["seed"] for record in records],
        "conditions": conditions,
        "stream_changes_f0_vs_f1": representation,
        "alignment": {
            "node_shuffle_drop": alignment,
            "alignment_gain_due_to_pdc": gain,
            "current_m1c": current_alignment,
            "three_layer_comparison": {
                "Current": current_alignment,
                "PDC-On": alignment["F0_pdc_on"],
                "PDC-Off": alignment["F1_all_pdc_off"],
            },
        },
        "formal_forward_max_abs_error": _mean_std(
            [record["formal_forward_max_abs_error"] for record in records]
        ),
    }


def _continue_decision(dataset_summaries: dict[str, Any]) -> dict[str, Any]:
    positive_alignment_datasets = []
    functional_runs = 0
    total_runs = 0
    for dataset, summary in dataset_summaries.items():
        gain = summary["alignment"]["alignment_gain_due_to_pdc"]
        if gain["val_macro_f1"]["mean"] > 0.0 or gain["test_macro_f1"]["mean"] > 0.0:
            positive_alignment_datasets.append(dataset)
        # Every run with a nonzero fused and logit response is functionally
        # engaged; the threshold is numerical, not an accuracy-selection rule.
        fused = summary["stream_changes_f0_vs_f1"]["fused_representation_change"]["mean_relative_l2"]["mean"]
        logit = summary["stream_changes_f0_vs_f1"]["logit_probability_change"]["all"]["mean_relative_logit_l2"]["mean"]
        if fused > 1e-8 and logit > 1e-8:
            functional_runs += summary["n_training_seeds"]
        total_runs += summary["n_training_seeds"]
    supported = bool(functional_runs > 0 and positive_alignment_datasets)
    return {
        "criterion": "continue if frozen removal changes the task stream and at least one dataset has a positive PDC-vs-off node-shuffle alignment gain on Val Macro-F1 or Test Macro-F1",
        "functional_runs": functional_runs,
        "total_runs": total_runs,
        "positive_alignment_datasets": positive_alignment_datasets,
        "pdc_supported_for_followup": supported,
        "sports_lp_triggered_by_this_audit": False,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MoPF M1-D — PDC Functional Verification",
        "",
        "This audit reuses the M1-C `v1_pdc` best-Validation-Accuracy NC checkpoints. No NC training or checkpoint mutation was performed.",
        "",
        f"- Checkpoints audited: {len(summary['runs'])} (5 datasets × 3 seeds)",
        f"- Device: {summary['metadata']['device']}",
        f"- PDC follow-up decision: **{'continue' if summary['continue_decision']['pdc_supported_for_followup'] else 'do not continue'}**",
        f"- Formal-forward re-evaluation max absolute difference: `{max(record['formal_forward_max_abs_error'] for record in summary['runs']):.3g}` (CUDA sparse-reduction repeat-rounding)",
        f"- Eta-change vs delta-change max absolute error: `{max(record['stream_changes_f0_vs_f1']['eta_change_equals_delta_change_max_abs_error'] for record in summary['runs']):.3g}`",
        "",
        "## Frozen task-stream evidence",
        "",
        "| Dataset | fused relative L2 | all-node logit relative L2 | alignment gain Val Macro-F1 | alignment gain Test Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, item in summary["dataset_summaries"].items():
        fused = item["stream_changes_f0_vs_f1"]["fused_representation_change"]["mean_relative_l2"]["mean"]
        logit = item["stream_changes_f0_vs_f1"]["logit_probability_change"]["all"]["mean_relative_logit_l2"]["mean"]
        gain = item["alignment"]["alignment_gain_due_to_pdc"]
        lines.append(
            f"| {dataset} | {fused:.6g} | {logit:.6g} | {gain['val_macro_f1']['mean']:.6g} | {gain['test_macro_f1']['mean']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`PDC-On - intervention` is used for downstream deltas. Positive values mean that closing the PDC path lowers the metric. Node-shuffle alignment gain is `shuffle_drop_on - shuffle_drop_off`; positive values mean that the learned PDC path makes node/profile alignment more task-functional.",
            "",
            "The authoritative machine-readable artifact is `m1d_master_summary.json`; each run also contains path-utilization, stream-change, and alignment JSON files.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "outputs" / "m1c_pdc_ppc")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "m1d_pdc_functional")
    parser.add_argument("--m1c-summary", type=Path, default=ROOT / "outputs" / "m1c_pdc_ppc" / "m1c_master_summary.json")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for dataset, seed in itertools.product(DATASETS, SEEDS):
        print(f"[m1d] auditing {dataset}/seed{seed} on {device}", flush=True)
        records.append(_audit_one(dataset, seed, args.checkpoint_root, args.output_root, device, args.force))
    records.sort(key=lambda record: (DATASETS.index(record["dataset"]), SEEDS.index(record["seed"])))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["dataset"]].append(record)
    dataset_summaries = {
        dataset: _aggregate_dataset(
            grouped[dataset], _current_alignment_from_m1c(args.m1c_summary, dataset)
        )
        for dataset in DATASETS
    }
    summary = {
        "metadata": {
            "stage": "M1-D",
            "study": "PDC Functional Verification",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_source": str(args.checkpoint_root),
            "device": str(device),
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "K_by_dataset": K_BY_DATASET,
            "node_shuffle_seeds": list(SHUFFLE_SEEDS),
            "frozen_conditions": list(records[0]["conditions"]),
        },
        "runs": records,
        "dataset_summaries": dataset_summaries,
        "continue_decision": _continue_decision(dataset_summaries),
        "artifacts": {
            "report": str(ROOT / "docs" / "mopf_m1d_pdc_functional.md"),
            "m1c_summary": str(args.m1c_summary),
        },
    }
    _dump_json(args.output_root / "m1d_master_summary.json", summary)
    _write_report(ROOT / "docs" / "mopf_m1d_pdc_functional.md", summary)
    print(f"[m1d] summary={args.output_root / 'm1d_master_summary.json'}", flush=True)
    print(f"[m1d] decision={summary['continue_decision']}", flush=True)


if __name__ == "__main__":
    main()
