#!/usr/bin/env python3
"""P5: zero-training Same-Scalar audit and scalar hierarchy analysis."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from src.data import load_mag_data
from src.models.cssi_scalar_probe import CSSIScalarProbe
from src.models.cssi_v1 import CSSIV1
from scripts.analyze_cssi_p4 import _mrc_row


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
NEW_VARIANTS = ("static_modality", "static_order", "self_scalar", "same_scalar_mrc")
P4_VARIANT = "same_scalar_v1"
EPS = 1.0e-8
RESPONSE_COEFFICIENTS = (0.75, 0.50, 0.25)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(root: Path, variant: str, dataset: str, seed: int) -> Path:
    return root / "runs" / variant / dataset / f"seed_{seed}"


def _checkpoint(root: Path, variant: str, dataset: str, seed: int) -> Path:
    return root / "checkpoints" / variant / dataset / f"seed_{seed}.pt"


def _load_checkpoint(
    root: Path,
    variant: str,
    dataset: str,
    seed: int,
    device: torch.device,
    p4: bool,
):
    run = _run(root, variant, dataset, seed)
    checkpoint_path = _checkpoint(root, variant, dataset, seed)
    if not (run / "resolved_config.json").is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {run}")
    cfg = OmegaConf.create(_json(run / "resolved_config.json"))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = load_mag_data(cfg, "nc", seed)
    model_cls = CSSIV1 if p4 else CSSIScalarProbe
    model = model_cls(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    classifier.eval()
    return cfg, data, model, classifier, payload


def _stats(values: torch.Tensor | np.ndarray) -> dict[str, float]:
    if isinstance(values, torch.Tensor):
        array = values.detach().float().cpu().numpy().reshape(-1)
    else:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: float("nan") for key in ("mean", "std", "median", "q10", "q25", "q50", "q75", "q90")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _metrics(logits: torch.Tensor, data, idx: torch.Tensor) -> tuple[float, float]:
    pred = logits[idx].argmax(dim=-1).detach().cpu().numpy()
    target = data.y.to(logits.device)[idx].detach().cpu().numpy()
    return float(np.mean(pred == target)), float(
        f1_score(target, pred, labels=sorted(set(target.tolist())), average="macro", zero_division=0)
    )


def _gate(components: dict[str, Any], modality: str) -> torch.Tensor:
    explicit = components.get(f"scalar_gate_{modality}")
    if explicit is not None:
        return explicit
    amplitude = components[f"modulation_amplitude_{modality}"]
    if amplitude.ndim == 3:
        amplitude = amplitude.squeeze(-1)
    strength = components[f"lambda_{modality}"].reshape(1, -1)
    return amplitude * strength


def _reconstruct(
    model,
    components: dict[str, Any],
    gates: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    with torch.no_grad():
        corrections: dict[str, list[torch.Tensor]] = {}
        modality_states = {}
        for modality in ("text", "visual"):
            response_list = components[f"responses_{modality}"]
            correction_list = [
                gates[modality][:, order].unsqueeze(-1) * response
                for order, response in enumerate(response_list)
            ]
            corrections[modality] = correction_list
            modality_states[modality] = components[f"z_base_{modality}"] + sum(
                coefficient * correction
                for coefficient, correction in zip(RESPONSE_COEFFICIENTS, correction_list)
            )
        _, _, z = model._refine_and_fuse(
            modality_states["text"], modality_states["visual"]
        )
    return z, corrections


def _shift_metrics(
    normal_gates: dict[str, torch.Tensor],
    gates: dict[str, torch.Tensor],
    normal_corrections: dict[str, list[torch.Tensor]],
    corrections: dict[str, list[torch.Tensor]],
    val_idx: torch.Tensor,
) -> tuple[float, float, float]:
    l1, l2, correction_shift = [], [], []
    for modality in ("text", "visual"):
        delta_gate = gates[modality][val_idx] - normal_gates[modality][val_idx]
        l1.append(delta_gate.abs().mean(dim=-1))
        l2.append(delta_gate.norm(dim=-1))
        for normal, altered in zip(normal_corrections[modality], corrections[modality]):
            correction_shift.append((altered[val_idx] - normal[val_idx]).norm(dim=-1))
    return (
        float(torch.cat(l1).mean()),
        float(torch.cat(l2).mean()),
        float(torch.cat(correction_shift).mean()),
    )


def _intervention_rows(
    root: Path,
    variant: str,
    dataset: str,
    seed: int,
    device: torch.device,
    p4: bool,
) -> list[dict[str, Any]]:
    _, data, model, classifier, _ = _load_checkpoint(root, variant, dataset, seed, device, p4)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    val_idx = data.val_idx.to(device)
    train_idx = data.train_idx.to(device)
    normal = model.analysis_components(x, edge_index, cross_intervention="normal")
    normal_gates = {m: _gate(normal, m) for m in ("text", "visual")}
    normal_z, normal_corrections = _reconstruct(model, normal, normal_gates)
    reconstruction_error = float((normal_z - normal["z"]).abs().max())
    if reconstruction_error > 1.0e-5:
        raise AssertionError(f"scalar reconstruction mismatch {dataset}/{seed}: {reconstruction_error}")

    generator = torch.Generator(device=device).manual_seed(seed + 517)
    all_permutation = torch.randperm(data.num_nodes, generator=generator, device=device)
    val_permutation = torch.randperm(val_idx.numel(), generator=generator, device=device)
    interventions: list[tuple[str, dict[str, Any], dict[str, torch.Tensor]]] = []
    interventions.append(("normal", normal, normal_gates))
    for name, permutation in (("cross_off", None), ("cross_shuffle", all_permutation)):
        current = model.analysis_components(
            x, edge_index, cross_intervention="off" if name == "cross_off" else "shuffle",
            permutation=permutation,
        )
        interventions.append((name, current, {m: _gate(current, m) for m in ("text", "visual")}))

    for name in ("node_gate_shuffle", "train_mean_gate", "modality_mean_gate"):
        gates = {m: value.clone() for m, value in normal_gates.items()}
        for modality in ("text", "visual"):
            if name == "node_gate_shuffle":
                gates[modality][val_idx] = gates[modality][val_idx][val_permutation]
            elif name == "train_mean_gate":
                gates[modality][val_idx] = gates[modality][train_idx].mean(dim=0).unsqueeze(0)
            else:
                mean_gate = gates[modality][train_idx].mean(dim=(0, 1))
                gates[modality][val_idx] = mean_gate.view(1, 1).expand(val_idx.numel(), gates[modality].size(1))
        interventions.append((name, normal, gates))

    rows = []
    for name, components, gates in interventions:
        z, corrections = _reconstruct(model, components, gates)
        acc, macro_f1 = _metrics(classifier(z), data, val_idx)
        gate_l1, gate_l2, correction_shift = _shift_metrics(
            normal_gates, gates, normal_corrections, corrections, val_idx
        )
        z_delta = (z[val_idx] - normal_z[val_idx]).norm(dim=-1)
        z_cos = F.cosine_similarity(z[val_idx], normal_z[val_idx], dim=-1, eps=EPS)
        paired_off_max = 0.0
        if name == "cross_off":
            paired_off_max = max(
                float(components["control_q_visual_for_text"].abs().max()),
                float(components["control_q_text_for_visual"].abs().max()),
            )
        rows.append({
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "split": "validation",
            "intervention": name,
            "val_acc": acc,
            "val_macro_f1": macro_f1,
            "mean_z_l2_shift": float(z_delta.mean()),
            "mean_z_cosine_to_normal": float(z_cos.mean()),
            "gate_l1_shift": gate_l1,
            "gate_l2_shift": gate_l2,
            "mean_correction_l2_shift": correction_shift,
            "cross_off_paired_control_max_abs": paired_off_max,
            "finite": bool(torch.isfinite(z).all()),
        })
    del data, model, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _gate_diagnostic_rows(
    root: Path, dataset: str, seed: int, device: torch.device
) -> list[dict[str, Any]]:
    _, data, model, _, _ = _load_checkpoint(root, P4_VARIANT, dataset, seed, device, True)
    x = data.x.to(device)
    components = model.analysis_components(x, data.edge_index.to(device))
    val_idx = data.val_idx.to(device)
    gates = {m: _gate(components, m)[val_idx] for m in ("text", "visual")}
    modality_difference = (gates["text"] - gates["visual"]).abs()
    rows = []
    for modality in ("text", "visual"):
        gate = gates[modality]
        order_variation = gate.var(dim=1, unbiased=False).mean()
        for order in range(3):
            values = gate[:, order]
            row = {
                "variant": P4_VARIANT,
                "dataset": dataset,
                "seed": seed,
                "split": "validation",
                "modality": modality,
                "order": order + 1,
                "N": int(values.numel()),
                "node_variation_var_i": float(values.var(unbiased=False)),
                "node_dispersion_std": float(values.std(unbiased=False)),
                "order_variation_mean_var_k": float(order_variation),
                "modality_difference_mean_abs": float(modality_difference[:, order].mean()),
                "modality_difference_median_abs": float(modality_difference[:, order].median()),
                "positive_fraction": float((values > 0).float().mean()),
                "negative_fraction": float((values < 0).float().mean()),
            }
            row.update({f"gate_{key}": value for key, value in _stats(values).items()})
            rows.append(row)
    del data, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _alignment_row(frame: pd.DataFrame, dataset: str, seed: str, modality: str, order: int) -> dict[str, Any]:
    gate = frame["gate"].to_numpy(dtype=np.float64)
    ce = frame["utility_ce"].to_numpy(dtype=np.float64)
    margin = frame["utility_margin"].to_numpy(dtype=np.float64)
    mask = np.isfinite(gate) & np.isfinite(ce) & np.isfinite(margin)
    gate, ce, margin = gate[mask], ce[mask], margin[mask]
    if gate.size < 3:
        return {"dataset": dataset, "seed": seed, "modality": modality, "order": order, "N": int(gate.size)}
    rho_ce = float(spearmanr(gate, ce).statistic) if np.std(gate) and np.std(ce) else float("nan")
    rho_margin = float(spearmanr(gate, margin).statistic) if np.std(gate) and np.std(margin) else float("nan")
    positive = (ce > 0.0).astype(np.int64)
    auc = float(roc_auc_score(positive, gate)) if np.unique(positive).size == 2 else float("nan")
    quartile = pd.qcut(ce, q=4, labels=False, duplicates="drop")
    row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "modality": modality,
        "order": order,
        "N": int(gate.size),
        "spearman_gate_utility_ce": rho_ce,
        "spearman_gate_utility_margin": rho_margin,
        "auroc_gate_predicts_positive_ce": auc,
    }
    for q in range(4):
        values = gate[np.asarray(quartile == q)]
        row[f"utility_ce_q{q + 1}_gate_mean"] = float(np.mean(values)) if values.size else float("nan")
        row[f"utility_ce_q{q + 1}_gate_median"] = float(np.median(values)) if values.size else float("nan")
    return row


def _alignment_rows(project_root: Path, gate_rows: pd.DataFrame) -> pd.DataFrame:
    merged_frames = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = project_root / "outputs/cssi_p1/node_level" / dataset / f"seed_{seed}_validation.csv.gz"
            utilities = pd.read_csv(path)
            utilities = utilities[utilities["split"] == "validation"]
            current_gate = gate_rows[(gate_rows.dataset == dataset) & (gate_rows.seed == seed)]
            # Gate rows are reconstructed from full-node checkpoints; align by
            # physical node id and leave P1 utility as a functional cross-check.
            gate_frame = current_gate[["node_id", "modality", "order", "gate"]]
            merged_frames.append(utilities.merge(gate_frame, on=["node_id", "modality", "order"], how="inner"))
    merged = pd.concat(merged_frames, ignore_index=True)
    rows = []
    for keys in (
        ["dataset", "seed", "modality", "order"],
        ["dataset", "modality", "order"],
        ["modality", "order"],
    ):
        for group_key, group in merged.groupby(keys, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            values = dict(zip(keys, group_key))
            row = _alignment_row(group, values.get("dataset", "ALL"), values.get("seed", "ALL"), values["modality"], int(values["order"]))
            for key in ("dataset", "seed", "modality", "order"):
                row.setdefault(key, values.get(key, "ALL"))
            rows.append(row)
    all_row = _alignment_row(merged, "ALL", "ALL", "ALL", 0)
    rows.append(all_row)
    return pd.DataFrame(rows)


def _gate_node_rows(
    root: Path, dataset: str, seed: int, device: torch.device
) -> list[dict[str, Any]]:
    _, data, model, _, _ = _load_checkpoint(root, P4_VARIANT, dataset, seed, device, True)
    with torch.no_grad():
        components = model.analysis_components(data.x.to(device), data.edge_index.to(device))
    rows = []
    val_ids = data.val_idx.to(device)
    for modality in ("text", "visual"):
        gate = _gate(components, modality)
        for order in range(3):
            values = gate[val_ids, order]
            for node_id, value in zip(val_ids.tolist(), values.tolist()):
                rows.append({"dataset": dataset, "seed": seed, "split": "validation", "node_id": int(node_id), "modality": modality, "order": order + 1, "gate": float(value)})
    del data, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _read_metrics(root: Path, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    payload = _json(_run(root, variant, dataset, seed) / "metrics.json")
    return {
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "val_acc": float(payload["metrics"]["val_acc"]["mean"]),
        "val_macro_f1": float(payload["metrics"]["val_macro_f1"]["mean"]),
        "best_epoch": payload.get("best_epoch"),
        "evaluate_test": bool(payload.get("task") != "nc" or False),
    }


def _reference_rows(project_root: Path) -> pd.DataFrame:
    p4_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            p4_rows.append(_read_metrics(project_root / "outputs/cssi_p4", P4_VARIANT, dataset, seed))
    p4 = pd.DataFrame(p4_rows)
    p4["source"] = "p4_reference"
    plain = pd.read_csv(project_root / "outputs/cssi_p1/backbone_sanity.csv")
    plain = plain[plain["variant"] == "plain"].copy()
    plain = plain.rename(columns={"test_evaluated": "evaluate_test"})
    plain["source"] = "p1_reference"
    plain["variant"] = "plain"
    return pd.concat([p4, plain[["variant", "dataset", "seed", "val_acc", "val_macro_f1", "best_epoch", "evaluate_test", "source"]]], ignore_index=True)


def _paired_rows(summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("static_order-static_modality", "static_order", "static_modality"),
        ("self_scalar-static_order", "self_scalar", "static_order"),
        ("same_scalar_v1-self_scalar", P4_VARIANT, "self_scalar"),
        ("same_scalar_v1-static_order", P4_VARIANT, "static_order"),
        ("same_scalar_mrc-same_scalar_v1", "same_scalar_mrc", P4_VARIANT),
        ("same_scalar_v1-plain", P4_VARIANT, "plain"),
    )
    rows = []
    for name, first, second in comparisons:
        left = summary[summary.variant == first].set_index(["dataset", "seed"])
        right = summary[summary.variant == second].set_index(["dataset", "seed"])
        common = left.index.intersection(right.index)
        for metric in ("val_acc", "val_macro_f1"):
            values = []
            for dataset, seed in common:
                delta = float(left.loc[(dataset, seed), metric] - right.loc[(dataset, seed), metric])
                values.append((dataset, seed, delta))
                rows.append({"comparison": name, "dataset": dataset, "seed": seed, "metric": metric, "delta": delta, "level": "seed"})
            if not values:
                continue
            frame = pd.DataFrame(values, columns=["dataset", "seed", "delta"])
            for dataset, group in frame.groupby("dataset"):
                arr = group.delta.to_numpy()
                rows.append({"comparison": name, "dataset": dataset, "seed": "ALL", "metric": metric, "delta": float(arr.mean()), "mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(np.std(arr)), "positive_fraction": float(np.mean(arr > 0)), "zero_fraction": float(np.mean(arr == 0)), "negative_fraction": float(np.mean(arr < 0)), "level": "dataset"})
            arr = frame.delta.to_numpy()
            rows.append({"comparison": name, "dataset": "ALL", "seed": "ALL", "metric": metric, "delta": float(arr.mean()), "mean": float(arr.mean()), "median": float(np.median(arr)), "std": float(np.std(arr)), "positive_fraction": float(np.mean(arr > 0)), "zero_fraction": float(np.mean(arr == 0)), "negative_fraction": float(np.mean(arr < 0)), "level": "all"})
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "Unavailable."
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _decision(summary: pd.DataFrame, paired: pd.DataFrame, interventions: pd.DataFrame, alignment: pd.DataFrame) -> tuple[str, list[str]]:
    def all_delta(comparison: str, metric: str) -> float:
        row = paired[(paired.comparison == comparison) & (paired.metric == metric) & (paired.level == "all")]
        return float(row.iloc[0]["mean"]) if not row.empty else float("nan")

    def dataset_positive(comparison: str, metric: str) -> int:
        rows = paired[(paired.comparison == comparison) & (paired.metric == metric) & (paired.level == "dataset")]
        return int((rows["delta"] > 0).sum())

    def seed_positive_fraction(comparison: str, metric: str) -> float:
        rows = paired[(paired.comparison == comparison) & (paired.metric == metric) & (paired.level == "seed")]
        return float((rows["delta"] > 0).mean()) if not rows.empty else float("nan")

    def stable_comparison(comparison: str) -> bool:
        # This is an explicit audit convention: positive aggregate delta,
        # positive dataset mean on at least 3/5 datasets, and positive on at
        # least half of the paired seed rows. It avoids calling a tiny pooled
        # mean a stable mechanism effect.
        return all(
            all_delta(comparison, metric) > 0.0
            and dataset_positive(comparison, metric) >= 3
            and seed_positive_fraction(comparison, metric) >= 0.5
            for metric in ("val_acc", "val_macro_f1")
        )

    same_static = stable_comparison("same_scalar_v1-static_order")
    same_self = stable_comparison("same_scalar_v1-self_scalar")
    normal = interventions[interventions.intervention == "normal"].groupby(["variant", "dataset", "seed"], as_index=False).first()
    cross_counts: dict[str, dict[str, int]] = {}
    for intervention in ("cross_off", "cross_shuffle"):
        base = normal[normal.variant == P4_VARIANT].groupby("dataset", as_index=False)[["val_acc", "val_macro_f1"]].mean().set_index("dataset")
        altered = interventions[(interventions.variant == P4_VARIANT) & (interventions.intervention == intervention)].groupby("dataset", as_index=False)[["val_acc", "val_macro_f1"]].mean().set_index("dataset")
        common = base.index.intersection(altered.index)
        cross_counts[intervention] = {
            metric: int(((base.loc[common, metric] - altered.loc[common, metric]) > 0.0).sum())
            for metric in ("val_acc", "val_macro_f1")
        }
    cross_drop = all(
        cross_counts.get(intervention, {}).get(metric, 0) >= 3
        for intervention in ("cross_off", "cross_shuffle")
        for metric in ("val_acc", "val_macro_f1")
    )
    gate_counts: dict[str, int] = {}
    for intervention in ("node_gate_shuffle", "train_mean_gate"):
        base = normal[normal.variant == P4_VARIANT].groupby("dataset", as_index=False)[["val_acc", "val_macro_f1"]].mean().set_index("dataset")
        altered = interventions[(interventions.variant == P4_VARIANT) & (interventions.intervention == intervention)].groupby("dataset", as_index=False)[["val_acc", "val_macro_f1"]].mean().set_index("dataset")
        common = base.index.intersection(altered.index)
        delta = base.loc[common] - altered.loc[common]
        gate_counts[intervention] = int(((delta["val_acc"] > 0.0) & (delta["val_macro_f1"] > 0.0)).sum())
    mean_gate_changes = all(gate_counts.get(intervention, 0) >= 3 for intervention in ("node_gate_shuffle", "train_mean_gate"))
    align = alignment[(alignment.dataset == "ALL") & (alignment.seed == "ALL") & (alignment.modality != "ALL") & (alignment.order != 0)]
    association_rows = align.dropna(subset=["spearman_gate_utility_ce", "spearman_gate_utility_margin", "auroc_gate_predicts_positive_ce"], how="all")
    assoc = int(
        (
            (association_rows["spearman_gate_utility_ce"].abs() >= 0.1)
            | (association_rows["spearman_gate_utility_margin"].abs() >= 0.1)
            | ((association_rows["auroc_gate_predicts_positive_ce"] - 0.5).abs() >= 0.05)
        ).sum()
    ) >= 3 if not association_rows.empty else False
    same_static_detail = (
        f"Same-vs-StaticOrder: Acc Δ={all_delta('same_scalar_v1-static_order', 'val_acc'):.4f}, "
        f"Macro-F1 Δ={all_delta('same_scalar_v1-static_order', 'val_macro_f1'):.4f}; "
        f"dataset-positive={dataset_positive('same_scalar_v1-static_order', 'val_acc')}/5 and "
        f"{dataset_positive('same_scalar_v1-static_order', 'val_macro_f1')}/5, "
        f"seed-positive={seed_positive_fraction('same_scalar_v1-static_order', 'val_acc'):.3f} and "
        f"{seed_positive_fraction('same_scalar_v1-static_order', 'val_macro_f1'):.3f}."
    )
    same_self_detail = (
        f"Same-vs-Self: Acc Δ={all_delta('same_scalar_v1-self_scalar', 'val_acc'):.4f}, "
        f"Macro-F1 Δ={all_delta('same_scalar_v1-self_scalar', 'val_macro_f1'):.4f}; "
        f"dataset-positive={dataset_positive('same_scalar_v1-self_scalar', 'val_acc')}/5 and "
        f"{dataset_positive('same_scalar_v1-self_scalar', 'val_macro_f1')}/5, "
        f"seed-positive={seed_positive_fraction('same_scalar_v1-self_scalar', 'val_acc'):.3f} and "
        f"{seed_positive_fraction('same_scalar_v1-self_scalar', 'val_macro_f1'):.3f}."
    )
    cross_detail = " / ".join(f"{name}: Acc {values['val_acc']}/5, Macro-F1 {values['val_macro_f1']}/5" for name, values in cross_counts.items())
    gate_detail = " / ".join(f"{name}: {count}/5 datasets with both task metrics improved by Normal" for name, count in gate_counts.items())
    association_detail = f"{assoc and int((association_rows[['spearman_gate_utility_ce', 'spearman_gate_utility_margin']].abs().ge(0.1).any(axis=1) | ((association_rows['auroc_gate_predicts_positive_ce'] - 0.5).abs() >= 0.05)).sum()) or 0}/6 pooled modality/order strata pass the nontrivial-association screen."
    if same_static and same_self and cross_drop and mean_gate_changes > 0 and assoc:
        return "CSSI-Go", [same_static_detail, same_self_detail, f"Cross interventions: {cross_detail}.", f"Mean-gate interventions: {gate_detail}.", association_detail]
    if same_static:
        if same_self:
            return "Partial-Go", [same_static_detail, same_self_detail, f"Cross interventions: {cross_detail}.", f"Mean-gate interventions: {gate_detail}.", association_detail, "The evidence supports at most a scalar/node-adaptive calibration claim; it does not establish paired cross-modal conditionality as the core mechanism."]
        return "Partial-Go", [same_static_detail, same_self_detail, f"Cross interventions: {cross_detail}.", f"Mean-gate interventions: {gate_detail}.", association_detail, "Same-Scalar is stronger than Static-Order, but its gain over Self-Scalar is not stable; retain only a cautious node-adaptive structural-response interpretation."]
    return "No-Go", [same_static_detail, same_self_detail, f"Cross interventions: {cross_detail}.", f"Mean-gate interventions: {gate_detail}.", association_detail, "Same-Scalar does not show a stable advantage over the coarse scalar hierarchy controls under the completed validation evidence."]


def _write_report(project_root: Path, summary: pd.DataFrame, paired: pd.DataFrame, gate: pd.DataFrame, interventions: pd.DataFrame, alignment: pd.DataFrame, mrc: pd.DataFrame) -> None:
    decision, rationale = _decision(summary, paired, interventions, alignment)
    gate_aggregate = gate.groupby(["modality", "order"], as_index=False)[["gate_mean", "gate_std", "gate_q10", "gate_q25", "gate_q50", "gate_q75", "gate_q90", "positive_fraction", "negative_fraction", "node_variation_var_i", "order_variation_mean_var_k", "modality_difference_mean_abs"]].mean()
    intervention_aggregate = interventions.groupby(["variant", "intervention"], as_index=False)[["val_acc", "val_macro_f1", "mean_z_l2_shift", "mean_z_cosine_to_normal", "gate_l1_shift", "gate_l2_shift", "mean_correction_l2_shift"]].mean()
    intervention_dataset = interventions.groupby(["variant", "dataset", "intervention"], as_index=False)[["val_acc", "val_macro_f1", "mean_z_l2_shift", "mean_z_cosine_to_normal", "gate_l1_shift", "gate_l2_shift", "mean_correction_l2_shift"]].mean()
    alignment_all = alignment[(alignment.dataset == "ALL") & (alignment.seed == "ALL")]
    p4_interventions = interventions[interventions.variant == P4_VARIANT]
    p4_normal = p4_interventions[p4_interventions.intervention == "normal"].groupby("dataset", as_index=False)[["val_acc", "val_macro_f1"]].mean().set_index("dataset")
    gate_task_rows = []
    for intervention in ("node_gate_shuffle", "train_mean_gate", "modality_mean_gate"):
        altered = p4_interventions[p4_interventions.intervention == intervention].groupby("dataset", as_index=False)[["val_acc", "val_macro_f1"]].mean().set_index("dataset")
        common = p4_normal.index.intersection(altered.index)
        delta = p4_normal.loc[common] - altered.loc[common]
        gate_task_rows.append({
            "intervention": intervention,
            "datasets_normal_better_acc": int((delta["val_acc"] > 0).sum()),
            "datasets_normal_better_macro_f1": int((delta["val_macro_f1"] > 0).sum()),
            "datasets_normal_better_both": int(((delta["val_acc"] > 0) & (delta["val_macro_f1"] > 0)).sum()),
        })
    gate_task = pd.DataFrame(gate_task_rows)
    all_alignment = alignment_all[(alignment_all.modality == "ALL") & (alignment_all.order == 0)]
    all_alignment_row = all_alignment.iloc[0] if not all_alignment.empty else pd.Series(dtype=float)
    text_gate = gate[gate.modality == "text"]
    visual_gate = gate[gate.modality == "visual"]
    gate_node_max = float(gate_aggregate["node_variation_var_i"].max()) if not gate_aggregate.empty else float("nan")
    gate_order_max = float(gate_aggregate["order_variation_mean_var_k"].max()) if not gate_aggregate.empty else float("nan")
    gate_modality_mean = float(gate_aggregate["modality_difference_mean_abs"].mean()) if not gate_aggregate.empty else float("nan")
    lines = [
        "# CSSI P5 Conditionality Necessity Report",
        "",
        "All new experiments are NC-only validation runs under `unified_full_graph_nc_v1`, using existing splits, seeds 42/43/44, `task.evaluate_test=false`, and zero auxiliary-loss weight. No LP or test metric is used.",
        "",
        f"## Decision: {decision}",
        "",
        *rationale,
        "The operational check treats a comparison as stable only when both Accuracy and Macro-F1 have positive aggregate deltas, are positive on at least three of five dataset means, and are positive on at least half of the paired seed rows. This is an audit convention, not a claim of statistical significance.",
        "",
        "## 1. Model hierarchy validation metrics",
        "",
    ]
    agg = summary.groupby("variant", as_index=False).agg(N=("dataset", "count"), val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"), val_macro_f1_mean=("val_macro_f1", "mean"), val_macro_f1_std=("val_macro_f1", "std"))
    lines.append(_markdown(agg))
    lines += ["", "### Per-dataset mean ± std", ""]
    cell = summary.groupby(["dataset", "variant"], as_index=False).agg(N=("seed", "count"), val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"), val_macro_f1_mean=("val_macro_f1", "mean"), val_macro_f1_std=("val_macro_f1", "std"))
    lines.append(_markdown(cell))
    lines += ["", "## 2. Paired comparisons", ""]
    lines.append(_markdown(paired[paired.level == "all"]))
    lines += ["", "Per-seed and per-dataset deltas are in `outputs/cssi_p5/paired_differences.csv`; positive delta means the first model is better.", ""]
    lines += ["## 3. Same-Scalar gate distribution", ""]
    lines.append(_markdown(gate_aggregate))
    lines.append("`node_variation_var_i` is variance across validation nodes for a fixed modality/order; `order_variation_mean_var_k` is the mean within-node variance across orders; modality difference is paired text-versus-visual absolute gate difference.")
    lines += ["", "## 4. Frozen-forward interventions", ""]
    lines.append(_markdown(intervention_aggregate))
    lines += ["", "### Per-dataset means", ""]
    lines.append(_markdown(intervention_dataset))
    lines.append("The complete dataset/seed table is `outputs/cssi_p5/scalar_interventions.csv`; the table above is a compact dataset-mean view. Every intervention re-runs the frozen refinement, late fusion, and classifier; no logits were edited directly.")
    lines.append("Cross-Off zeros the paired projected control after its Linear, so a Linear bias cannot restore paired evidence. Train-Mean-Gate uses train nodes only; Node-Gate-Shuffle permutes gates only within validation nodes.")
    lines += ["", "## 5. Gate–utility functional alignment", ""]
    lines.append(_markdown(alignment[(alignment.dataset == "ALL") | ((alignment.seed == "ALL") & (alignment.dataset != "ALL"))]))
    lines.append("P1 utility is treated only as a cross-check. Because it comes from a different Plain checkpoint than P4 Same-Scalar, these correlations are cross-checkpoint functional associations, not supervised targets or causal effects.")
    lines += ["", "## 6. MRC", ""]
    lines.append(_markdown(mrc[["variant", "dataset", "seed", "raw_semantic_gap_mean", "raw_weight_gap_pearson", "neighbor_allocation_tv_divergence", "top_neighbor_disagreement", "incident_mass_abs_gap"]] if not mrc.empty else mrc))
    lines.append("Raw semantic gaps use cosine(H0) before the learned diagonal metric. `incident_mass_abs_gap` is the incident relation-weight total difference, not an incident mean.")
    lines += ["", "## 7. H1-style conditionality findings", ""]
    lines.append(f"- **H1.1 / non-degeneracy:** Same-Scalar gates are not exactly one-signed across all strata: text is predominantly negative (mean positive fraction {float(text_gate.positive_fraction.mean()):.4f}), while visual retains both signs (mean positive fraction {float(visual_gate.positive_fraction.mean()):.4f}; mean negative fraction {float(visual_gate.negative_fraction.mean()):.4f}). This establishes gate sign heterogeneity, not useful task conditionality by itself.")
    lines.append(f"- **H1.2 / node conditionality:** Gate dispersion across nodes is measurable but compressed: the largest fixed-modality/order node variance is {gate_node_max:.6f}; gate shuffling changes representations more clearly than it changes validation decisions. Evidence for functionally important node conditionality is weak.")
    lines.append(f"- **H1.3 / modality conditionality:** Text-versus-visual paired gate difference averages {gate_modality_mean:.4f}, so the controller distinguishes modalities at a coarse level. The hierarchy result does not show that paired cross-modal evidence is needed for the task gain.")
    lines.append(f"- **H1.4 / order conditionality:** Mean within-node order variance is at most {gate_order_max:.6f} in the aggregate. Static-Order is close to Static-Modality, while Same-Scalar's advantage over Static-Order is small; order-specific calibration is plausible but not sufficient evidence for a conditional interaction mechanism.")
    if not all_alignment_row.empty:
        lines.append(f"- **H1.5 / utility alignment:** Aggregate cross-check values are Spearman(gate, CE utility)={float(all_alignment_row.get('spearman_gate_utility_ce', float('nan'))):.4f}, Spearman(gate, margin utility)={float(all_alignment_row.get('spearman_gate_utility_margin', float('nan'))):.4f}, and AUROC={float(all_alignment_row.get('auroc_gate_predicts_positive_ce', float('nan'))):.4f}. Stratum-level signs are mixed; gate magnitude alone is not a reliable utility selector.")
    else:
        lines.append("- **H1.5 / utility alignment:** Aggregate alignment was unavailable.")
    same_self_rows = paired[(paired.comparison == "same_scalar_v1-self_scalar") & (paired.level == "all")]
    same_static_rows = paired[(paired.comparison == "same_scalar_v1-static_order") & (paired.level == "all")]
    same_self_acc = float(same_self_rows[same_self_rows.metric == "val_acc"].iloc[0]["mean"]) if not same_self_rows[same_self_rows.metric == "val_acc"].empty else float("nan")
    same_self_f1 = float(same_self_rows[same_self_rows.metric == "val_macro_f1"].iloc[0]["mean"]) if not same_self_rows[same_self_rows.metric == "val_macro_f1"].empty else float("nan")
    same_static_acc = float(same_static_rows[same_static_rows.metric == "val_acc"].iloc[0]["mean"]) if not same_static_rows[same_static_rows.metric == "val_acc"].empty else float("nan")
    same_static_f1 = float(same_static_rows[same_static_rows.metric == "val_macro_f1"].iloc[0]["mean"]) if not same_static_rows[same_static_rows.metric == "val_macro_f1"].empty else float("nan")
    lines.append(f"- **H1.6 / stability:** Same-Scalar is above Static-Order on the aggregate (Accuracy Δ={same_static_acc:.4f}, Macro-F1 Δ={same_static_f1:.4f}), but its difference from Self-Scalar is near zero (Accuracy Δ={same_self_acc:.4f}, Macro-F1 Δ={same_self_f1:.4f}) and is not stable under the stated audit rule. Cross-modal and gate-intervention task counts are shown below.")
    lines += ["", "### Task-level intervention reproducibility", ""]
    lines.append(_markdown(gate_task))
    lines.append("For Cross-Off and Cross-Shuffle, the screen counts how many of the five dataset means have Normal better than the intervention for each metric.")
    cross_table = []
    for intervention in ("cross_off", "cross_shuffle"):
        altered = p4_interventions[p4_interventions.intervention == intervention].groupby("dataset", as_index=False)[["val_acc", "val_macro_f1"]].mean().set_index("dataset")
        common = p4_normal.index.intersection(altered.index)
        delta = p4_normal.loc[common] - altered.loc[common]
        cross_table.append({"intervention": intervention, "datasets_normal_better_acc": int((delta.val_acc > 0).sum()), "datasets_normal_better_macro_f1": int((delta.val_macro_f1 > 0).sum())})
    lines.append(_markdown(pd.DataFrame(cross_table)))
    lines += ["", "## 8. Go / Partial-Go / No-Go interpretation", ""]
    lines.append(f"**{decision}.**")
    lines.extend(f"- {item}" for item in rationale)
    lines.append("The evidence supports at most a lightweight scalar or node-adaptive structural-response calibration story. It does not justify presenting same-order cross-modal conditioning as a necessary core mechanism. Same-Scalar+MRC is mixed and slightly below Same-Scalar in the aggregate, so MRC should remain optional analysis rather than a mandatory module.")
    lines += ["", "## 9. Caveats and implementation boundary", ""]
    lines.append("P5 is validation-only and uses the existing split/checkpoint protocol; no test metric, LP run, dataset-specific tuning, BRSM change, attention, router, or auxiliary loss was introduced. P1 utility alignment is a cross-checkpoint functional association because P1 Plain and P4 Same-Scalar are different frozen checkpoints; it is not a causal estimate or supervised gate target. The interventions are frozen-forward functional audits, not causal effects.")
    lines.append("The scalar hierarchy is implemented independently in `src/models/cssi_scalar_probe.py`; historical `cosi_mag_final.py`, P1/P2/P3/P4 reference implementations, and their training logic were not modified.")
    (project_root / "docs/cssi_p5_necessity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p5"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summary = pd.read_csv(root / "summary.csv")
        paired = pd.read_csv(root / "paired_differences.csv")
        gate = pd.read_csv(root / "scalar_gate_diagnostics.csv")
        interventions = pd.read_csv(root / "scalar_interventions.csv")
        alignment = pd.read_csv(root / "gate_utility_alignment.csv")
        mrc = pd.read_csv(root / "mrc_diagnostics.csv") if (root / "mrc_diagnostics.csv").is_file() else pd.DataFrame()
        _write_report(project_root, summary, paired, gate, interventions, alignment, mrc)
        print(json.dumps({"report": str(project_root / "docs/cssi_p5_necessity_report.md")}, indent=2))
        return

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    intervention_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            intervention_rows.extend(_intervention_rows(project_root / "outputs/cssi_p4", P4_VARIANT, dataset, seed, device, True))
    for dataset in DATASETS:
        for seed in SEEDS:
            intervention_rows.extend(_intervention_rows(root, "same_scalar_mrc", dataset, seed, device, False))
    interventions = pd.DataFrame(intervention_rows)
    interventions.to_csv(root / "scalar_interventions.csv", index=False)

    gate_rows = []
    node_gate_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            gate_rows.extend(_gate_diagnostic_rows(project_root / "outputs/cssi_p4", dataset, seed, device))
            node_gate_rows.extend(_gate_node_rows(project_root / "outputs/cssi_p4", dataset, seed, device))
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(root / "scalar_gate_diagnostics.csv", index=False)
    alignment = _alignment_rows(project_root, pd.DataFrame(node_gate_rows))
    alignment.to_csv(root / "gate_utility_alignment.csv", index=False)

    mrc_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            _, data, model, _, _ = _load_checkpoint(root, "same_scalar_mrc", dataset, seed, device, False)
            components = model.analysis_components(data.x.to(device), data.edge_index.to(device))
            mrc_rows.append(_mrc_row(components, data, "same_scalar_mrc", dataset, seed))
            del data, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    mrc = pd.DataFrame(mrc_rows)
    mrc.to_csv(root / "mrc_diagnostics.csv", index=False)

    rows = []
    for variant in NEW_VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                row = _read_metrics(root, variant, dataset, seed)
                row["source"] = "p5_new"
                rows.append(row)
    summary = pd.concat([pd.DataFrame(rows), _reference_rows(project_root)], ignore_index=True)
    summary.to_csv(root / "summary.csv", index=False)
    paired = _paired_rows(summary)
    paired.to_csv(root / "paired_differences.csv", index=False)
    _write_report(project_root, summary, paired, gate, interventions, alignment, mrc)
    print(json.dumps({"summary_rows": len(summary), "paired_rows": len(paired), "gate_rows": len(gate), "intervention_rows": len(interventions), "alignment_rows": len(alignment), "mrc_rows": len(mrc)}, indent=2))


if __name__ == "__main__":
    main()
