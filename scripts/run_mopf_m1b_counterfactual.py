"""Run frozen M1-B counterfactuals for the five MoPF NC checkpoints.

This is deliberately an analysis-only script.  It loads an existing
best-validation-accuracy checkpoint, computes the normal full-graph embedding,
and then reuses the checkpoint's already-computed propagation bases while
replacing only the requested polynomial coefficient components.  Training,
losses, and the default MoPF forward path are not modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
DEFAULT_SHUFFLE_SEEDS = tuple(range(10))


def _json_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_eval_labels(data) -> list[int]:
    """Match the fixed NC Macro-F1 label protocol without importing sklearn."""
    if data.y is None or data.num_classes is None:
        raise ValueError("NC data must provide y and num_classes")
    split_indices = [
        index for index in (data.train_idx, data.val_idx, data.test_idx)
        if index is not None
    ]
    if not split_indices:
        raise ValueError("NC data must contain train/val/test indices")
    all_indices = torch.cat([index.reshape(-1) for index in split_indices])
    labels = data.y[all_indices].detach().cpu()
    valid = labels[(labels >= 0) & (labels < int(data.num_classes))]
    resolved = sorted({int(value) for value in valid.tolist()})
    if not resolved:
        raise ValueError("NC splits contain no valid labels")
    return resolved


def _macro_f1_fixed(target: torch.Tensor, pred: torch.Tensor, labels: list[int]) -> float:
    """Compute sklearn-compatible fixed-label Macro-F1 with zero_division=0."""
    target = target.detach().cpu().long()
    pred = pred.detach().cpu().long()
    scores: list[float] = []
    for label in labels:
        true_positive = int(((target == label) & (pred == label)).sum())
        false_positive = int(((target != label) & (pred == label)).sum())
        false_negative = int(((target == label) & (pred != label)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores))


@torch.no_grad()
def _evaluate_split(
    classifier: nn.Module,
    z: torch.Tensor,
    labels: torch.Tensor,
    index: torch.Tensor,
    device: torch.device,
    eval_labels: list[int],
) -> dict[str, float]:
    classifier.eval()
    index_device = index.to(device=device, dtype=torch.long)
    logits = classifier(z.index_select(0, index_device))
    pred = logits.argmax(dim=-1).cpu()
    target = labels.index_select(0, index.cpu()).cpu()
    return {
        "acc": float((pred == target).float().mean().item()),
        "macro_f1": _macro_f1_fixed(target, pred, eval_labels),
    }


def _load_checkpoint_bundle(output_dir: Path, device: torch.device):
    checkpoint = output_dir / "best_val_accuracy.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing best-validation checkpoint: {checkpoint}")
    config_path = output_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")

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


def _analysis_stats(model, data, device: torch.device) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        stats = model.analysis_stats(data.x.to(device), data.edge_index.to(device))
    return {key: value.detach().clone() for key, value in stats.items()}


def _canonical_decomposition(
    stats: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Create the centered representation and prove the identity numerically."""
    gamma_global = stats["gamma_global"].detach().cpu()
    saved: dict[str, torch.Tensor] = {"gamma_global": gamma_global}
    errors: dict[str, float] = {}
    for modality in ("text", "visual"):
        delta_gamma = stats[f"delta_gamma_{modality}"].detach().cpu()
        delta_node = stats[f"delta_node_{modality}"].detach().cpu()
        # Use float64 for the validation arithmetic; stored tensors remain the
        # original checkpoint precision and the tolerance is 1e-6.
        gamma64 = gamma_global.double()
        delta_gamma64 = delta_gamma.double()
        delta_node64 = delta_node.double()
        mean64 = delta_node64.mean(dim=0)
        centered64 = delta_node64 - mean64
        effective64 = delta_gamma64 + mean64
        lhs = gamma64.unsqueeze(0) + delta_gamma64.unsqueeze(0) + delta_node64
        rhs = gamma64.unsqueeze(0) + effective64.unsqueeze(0) + centered64
        max_error = float((lhs - rhs).abs().max().item())
        if max_error >= 1e-6:
            raise AssertionError(
                f"Canonical decomposition failed for {modality}: {max_error:.3e}"
            )
        saved[f"delta_gamma_{modality}"] = delta_gamma
        saved[f"delta_node_{modality}"] = delta_node
        saved[f"delta_node_mean_{modality}"] = mean64.float()
        saved[f"delta_node_centered_{modality}"] = centered64.float()
        saved[f"delta_gamma_effective_{modality}"] = effective64.float()
        saved[f"original_total_{modality}"] = lhs.float()
        saved[f"canonical_total_{modality}"] = rhs.float()
        errors[modality] = max_error
    saved["max_abs_error"] = torch.tensor(max(errors.values()), dtype=torch.float64)
    return saved, {
        "max_abs_error": max(errors.values()),
        "errors_by_modality": errors,
        "tolerance": 1e-6,
        "identity": (
            "gamma_global + delta_gamma + delta_node = "
            "gamma_global + delta_gamma_effective + delta_node_centered"
        ),
    }


def _effective_delta_gamma(model, stats: dict[str, torch.Tensor], modality: str):
    delta = stats[f"delta_gamma_{modality}"].to(next(model.parameters()).device)
    if not bool(model.use_modality_residual):
        delta = torch.zeros_like(delta)
    return delta


def _eta_from_parts(
    model,
    stats: dict[str, torch.Tensor],
    modality: str,
    node_residual: torch.Tensor,
    delta_gamma: torch.Tensor | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    gamma_global = stats["gamma_global"].to(device)
    if delta_gamma is None:
        delta_gamma = _effective_delta_gamma(model, stats, modality)
    if not bool(model.use_node_residual):
        node_residual = torch.zeros_like(node_residual)
    return gamma_global.unsqueeze(0) + delta_gamma.unsqueeze(0) + node_residual


def _compose_from_bases(model, components, eta_text, eta_visual) -> torch.Tensor:
    z_text = model._filter_bases(components["bases_text"], eta_text)
    z_visual = model._filter_bases(components["bases_visual"], eta_visual)
    z_text_refined = model.text_refine_norm(z_text + model.text_refine_mlp(z_text))
    z_visual_refined = model.visual_refine_norm(
        z_visual + model.visual_refine_mlp(z_visual)
    )
    fused_input = torch.cat([z_text_refined, z_visual_refined], dim=-1)
    skip = model.fusion_skip(fused_input)
    fusion = model.fusion_mlp(fused_input)
    z = model.output_norm(skip + fusion)
    return torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)


@torch.no_grad()
def _compute_counterfactual_embeddings(model, components, stats, mode: str):
    device = next(model.parameters()).device
    node_text = stats["delta_node_text"].to(device)
    node_visual = stats["delta_node_visual"].to(device)
    dg_text = _effective_delta_gamma(model, stats, "text")
    dg_visual = _effective_delta_gamma(model, stats, "visual")

    if mode == "original":
        # The original is the exact output already produced by the normal
        # encoder, avoiding any analysis-side reimplementation drift.
        return components["z"].nan_to_num(nan=0.0, posinf=1e4, neginf=-1e4)
    if mode == "node_mean":
        node_text = node_text.mean(dim=0, keepdim=True).expand_as(node_text)
        node_visual = node_visual.mean(dim=0, keepdim=True).expand_as(node_visual)
    elif mode == "modality_mean":
        mean_delta = 0.5 * (dg_text + dg_visual)
        dg_text = mean_delta
        dg_visual = mean_delta
    elif mode == "modality_swap":
        dg_text, dg_visual = dg_visual, dg_text
    elif mode == "node_mean_modality_mean":
        node_text = node_text.mean(dim=0, keepdim=True).expand_as(node_text)
        node_visual = node_visual.mean(dim=0, keepdim=True).expand_as(node_visual)
        mean_delta = 0.5 * (dg_text + dg_visual)
        dg_text = mean_delta
        dg_visual = mean_delta
    elif mode.startswith("node_shuffle:"):
        seed = int(mode.split(":", 1)[1])
        # Shuffle each modality's complete K+1 profile independently.  The
        # two streams are deterministic functions of the same public seed,
        # but using separate generators ensures that a text profile is not
        # accidentally kept paired with the visual profile from that node.
        text_generator = torch.Generator(device="cpu")
        visual_generator = torch.Generator(device="cpu")
        text_generator.manual_seed(2 * seed)
        visual_generator.manual_seed(2 * seed + 1)
        text_permutation = torch.randperm(
            node_text.size(0), generator=text_generator
        )
        visual_permutation = torch.randperm(
            node_visual.size(0), generator=visual_generator
        )
        node_text = node_text.index_select(0, text_permutation.to(device))
        node_visual = node_visual.index_select(0, visual_permutation.to(device))
    else:
        raise ValueError(f"Unknown counterfactual mode: {mode}")

    eta_text = _eta_from_parts(model, stats, "text", node_text, dg_text)
    eta_visual = _eta_from_parts(model, stats, "visual", node_visual, dg_visual)
    return _compose_from_bases(model, components, eta_text, eta_visual)


def _metrics(
    classifier,
    z,
    data,
    device,
    eval_labels,
) -> dict[str, float]:
    val = _evaluate_split(
        classifier, z, data.y, data.val_idx, device, eval_labels
    )
    test = _evaluate_split(
        classifier, z, data.y, data.test_idx, device, eval_labels
    )
    return {
        "val_acc": val["acc"],
        "val_macro_f1": val["macro_f1"],
        "test_acc": test["acc"],
        "test_macro_f1": test["macro_f1"],
    }


def _structural_summary(stats: dict[str, torch.Tensor]) -> dict[str, Any]:
    centered_stds: dict[str, list[float]] = {}
    for modality in ("text", "visual"):
        centered = stats[f"delta_node_{modality}"] - stats[
            f"delta_node_{modality}"
        ].mean(dim=0, keepdim=True)
        centered_stds[modality] = centered.std(dim=0, unbiased=False).tolist()
    all_stds = np.asarray([value for values in centered_stds.values() for value in values])
    profile_distance = torch.linalg.vector_norm(
        stats["eta_text"] - stats["eta_visual"], dim=1
    )
    radius_text = stats["effective_radius_text"]
    radius_visual = stats["effective_radius_visual"]
    return {
        "node_centered_std_by_modality_order": centered_stds,
        "node_centered_std_max": float(all_stds.max()),
        "node_centered_std_mean": float(all_stds.mean()),
        "modality_profile_distance_mean": float(profile_distance.mean().item()),
        "modality_profile_distance_max": float(profile_distance.max().item()),
        "modality_delta_gamma_distance": float(
            torch.linalg.vector_norm(
                stats["delta_gamma_text"] - stats["delta_gamma_visual"]
            ).item()
        ),
        "effective_radius_text_mean": float(radius_text.mean().item()),
        "effective_radius_visual_mean": float(radius_visual.mean().item()),
        "effective_radius_mean": float(
            0.5 * (radius_text.mean() + radius_visual.mean()).item()
        ),
    }


def _with_drops(metrics: dict[str, float], original: dict[str, float]) -> dict[str, float]:
    result = dict(metrics)
    result["delta_test_acc"] = original["test_acc"] - metrics["test_acc"]
    result["delta_test_macro_f1"] = (
        original["test_macro_f1"] - metrics["test_macro_f1"]
    )
    result["delta_val_acc"] = original["val_acc"] - metrics["val_acc"]
    result["delta_val_macro_f1"] = (
        original["val_macro_f1"] - metrics["val_macro_f1"]
    )
    return result


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "seed",
        "condition",
        "permutation_seed",
        "val_acc",
        "val_macro_f1",
        "test_acc",
        "test_macro_f1",
        "delta_test_acc",
        "delta_test_macro_f1",
        "delta_val_acc",
        "delta_val_macro_f1",
        "metric_std_test_acc",
        "metric_std_test_macro_f1",
        "node_centered_std_max",
        "node_centered_std_mean",
        "modality_profile_distance_mean",
        "modality_profile_distance_max",
        "effective_radius_text_mean",
        "effective_radius_visual_mean",
        "effective_radius_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_dataset(
    dataset: str,
    checkpoint_root: Path,
    output_root: Path,
    device: torch.device,
    shuffle_seeds: tuple[int, ...],
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_root / dataset / "seed42"
    output_dir = output_root / dataset / "seed42"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, cfg, data, model, classifier = _load_checkpoint_bundle(
        checkpoint_dir, device
    )
    eval_labels = _resolve_eval_labels(data)
    stats = _analysis_stats(model, data, device)
    canonical, canonical_summary = _canonical_decomposition(stats)
    torch.save(canonical, output_dir / "canonical_decomposition.pt")

    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        components = model._encode_components(x, edge_index)
        original_z = _compute_counterfactual_embeddings(
            model, components, stats, "original"
        )
        original_metrics = _metrics(classifier, original_z, data, device, eval_labels)

    structural = _structural_summary(stats)
    common = {
        "dataset": dataset,
        "seed": int(payload["seed"]),
        "checkpoint": str(checkpoint_dir / "best_val_accuracy.pt"),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metrics": payload.get("metrics", {}),
        "config_max_order": int(cfg.model.max_order),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "eval_labels": eval_labels,
        "macro_f1_protocol": "fixed labels from union(train, val, test), zero_division=0",
        "canonical_decomposition": canonical_summary,
        "structural_statistics": structural,
        "training_unchanged": True,
        "formal_forward_unchanged": True,
    }

    original_record = dict(common)
    original_record["condition"] = "original"
    original_record["metrics"] = original_metrics
    _dump_json(output_dir / "original.json", original_record)

    condition_metrics: dict[str, dict[str, float]] = {"original": original_metrics}
    for condition in ("node_mean", "modality_mean", "modality_swap", "node_mean_modality_mean"):
        with torch.no_grad():
            z = _compute_counterfactual_embeddings(model, components, stats, condition)
            metrics = _metrics(classifier, z, data, device, eval_labels)
        record = dict(common)
        record["condition"] = condition
        record["metrics"] = _with_drops(metrics, original_metrics)
        condition_metrics[condition] = metrics
        filename = {
            "node_mean": "node_mean.json",
            "modality_mean": "modality_mean.json",
            "modality_swap": "modality_swap.json",
            "node_mean_modality_mean": "node_mean_modality_mean.json",
        }[condition]
        _dump_json(output_dir / filename, record)

    shuffle_rows: list[dict[str, Any]] = []
    for shuffle_seed in shuffle_seeds:
        condition = f"node_shuffle:{shuffle_seed}"
        with torch.no_grad():
            z = _compute_counterfactual_embeddings(model, components, stats, condition)
            metrics = _metrics(classifier, z, data, device, eval_labels)
        shuffle_rows.append(
            {
                "permutation_seed": int(shuffle_seed),
                "metrics": metrics,
                "delta": _with_drops(metrics, original_metrics),
            }
        )
    shuffle_test_acc = np.asarray(
        [row["metrics"]["test_acc"] for row in shuffle_rows], dtype=np.float64
    )
    shuffle_test_f1 = np.asarray(
        [row["metrics"]["test_macro_f1"] for row in shuffle_rows], dtype=np.float64
    )
    shuffle_mean_metrics = {
        key: float(np.mean([row["metrics"][key] for row in shuffle_rows]))
        for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }
    shuffle_mean_drop = _with_drops(shuffle_mean_metrics, original_metrics)
    shuffle_summary = dict(common)
    shuffle_summary.update(
        {
            "condition": "node_shuffle",
            "permutation_mode": "independent text and visual profile permutations",
            "permutation_seeds": list(shuffle_seeds),
            "num_permutations": len(shuffle_seeds),
            "permutations": shuffle_rows,
            "mean_metrics": shuffle_mean_metrics,
            "mean_drop": shuffle_mean_drop,
            "std_metrics": {
                "test_acc": float(shuffle_test_acc.std(ddof=0)),
                "test_macro_f1": float(shuffle_test_f1.std(ddof=0)),
                "val_acc": float(
                    np.std([row["metrics"]["val_acc"] for row in shuffle_rows], ddof=0)
                ),
                "val_macro_f1": float(
                    np.std(
                        [row["metrics"]["val_macro_f1"] for row in shuffle_rows],
                        ddof=0,
                    )
                ),
            },
            "std_drop": {
                "test_acc": float(
                    np.std(
                        [row["delta"]["delta_test_acc"] for row in shuffle_rows],
                        ddof=0,
                    )
                ),
                "test_macro_f1": float(
                    np.std(
                        [row["delta"]["delta_test_macro_f1"] for row in shuffle_rows],
                        ddof=0,
                    )
                ),
            },
        }
    )
    _dump_json(output_dir / "node_shuffle_summary.json", shuffle_summary)

    csv_rows: list[dict[str, Any]] = []
    base_csv = {
        "dataset": dataset,
        "seed": int(payload["seed"]),
        **structural,
    }
    csv_rows.append({**base_csv, "condition": "original", **original_metrics,
                     **{key: 0.0 for key in ("delta_test_acc", "delta_test_macro_f1", "delta_val_acc", "delta_val_macro_f1")},
                     "permutation_seed": ""})
    for condition in ("node_mean", "modality_mean", "modality_swap", "node_mean_modality_mean"):
        csv_rows.append(
            {
                **base_csv,
                "condition": condition,
                **_with_drops(condition_metrics[condition], original_metrics),
                "permutation_seed": "",
            }
        )
    csv_rows.append(
        {
            **base_csv,
            "condition": "node_shuffle",
            "permutation_seed": "0..9",
            **shuffle_mean_drop,
            "metric_std_test_acc": shuffle_summary["std_metrics"]["test_acc"],
            "metric_std_test_macro_f1": shuffle_summary["std_metrics"]["test_macro_f1"],
        }
    )
    _write_summary_csv(output_dir / "counterfactual_summary.csv", csv_rows)

    # Release GPU allocations before the next dataset is loaded.
    del components, model, classifier, data
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {
        **common,
        "output_dir": str(output_dir),
        "conditions": {
            condition: _with_drops(metrics, original_metrics)
            for condition, metrics in condition_metrics.items()
        },
        "node_shuffle": {
            "mean_drop": shuffle_mean_drop,
            "std_drop": shuffle_summary["std_drop"],
            "mean_metrics": shuffle_mean_metrics,
            "std_metrics": shuffle_summary["std_metrics"],
        },
    }
    print(
        f"[m1b] {dataset} original test_acc={original_metrics['test_acc']:.6f} "
        f"node_mean_drop={result['conditions']['node_mean']['delta_test_acc']:.6f} "
        f"shuffle_drop={shuffle_mean_drop['delta_test_acc']:.6f} "
        f"modality_mean_drop={result['conditions']['modality_mean']['delta_test_acc']:.6f}",
        flush=True,
    )
    return result


def _safe_corr(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or x[valid].std() == 0.0 or y[valid].std() == 0.0:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _write_report(results: list[dict[str, Any]], path: Path) -> None:
    def pct(value: float) -> str:
        return f"{100.0 * value:+.3f} pp"

    rows: list[str] = []
    for result in results:
        cond = result["conditions"]
        shuffle = result["node_shuffle"]
        rows.append(
            "| {dataset} | {node_var:.5f} | {profile:.5f} | {radius:.3f} | "
            "{node_mean} | {shuffle} ± {shuffle_std} | {mod_mean} | {mod_swap} |".format(
                dataset=result["dataset"],
                node_var=result["structural_statistics"]["node_centered_std_mean"],
                profile=result["structural_statistics"]["modality_profile_distance_mean"],
                radius=result["structural_statistics"]["effective_radius_mean"],
                node_mean=pct(cond["node_mean"]["delta_test_acc"]),
                shuffle=pct(shuffle["mean_drop"]["delta_test_acc"]),
                shuffle_std=pct(shuffle["std_drop"]["test_acc"]),
                mod_mean=pct(cond["modality_mean"]["delta_test_acc"]),
                mod_swap=pct(cond["modality_swap"]["delta_test_acc"]),
            )
        )

    node_var = [
        result["structural_statistics"]["node_centered_std_mean"] for result in results
    ]
    node_shuffle_drop = [
        result["node_shuffle"]["mean_drop"]["delta_test_acc"] for result in results
    ]
    node_mean_drop = [
        result["conditions"]["node_mean"]["delta_test_acc"] for result in results
    ]
    modality_distance = [
        result["structural_statistics"]["modality_profile_distance_mean"]
        for result in results
    ]
    modality_mean_drop = [
        result["conditions"]["modality_mean"]["delta_test_acc"] for result in results
    ]
    modality_swap_drop = [
        result["conditions"]["modality_swap"]["delta_test_acc"] for result in results
    ]

    highest_node = max(results, key=lambda item: item["node_shuffle"]["mean_drop"]["delta_test_acc"])
    highest_modality = max(results, key=lambda item: item["conditions"]["modality_mean"]["delta_test_acc"])
    node_supported = any(value > 0.001 for value in node_mean_drop)
    node_alignment_supported = any(value > 0.001 for value in node_shuffle_drop)
    modality_supported = any(value > 0.001 for value in modality_mean_drop + modality_swap_drop)

    report = f"""# MoPF M1-B — Counterfactual Functional Validation

## Scope and protocol

This report evaluates frozen counterfactuals on the seed-42 best-validation-accuracy checkpoints for Movies, Toys, Grocery, ele-fashion, and Reddit-S. Grocery uses K=2; the other datasets use K=3. All metrics use the fixed Macro-F1 protocol whose label set is the union of valid labels in train, validation, and test splits, with `zero_division=0`.

No model was retrained. The formal MoPF forward path, training objective, checkpoints, semantic graphs, bases, fusion layers, and classifier heads were kept fixed. Counterfactuals only replace the requested coefficient component in the analysis-side embedding reconstruction.

## Canonical centered decomposition

For every dataset and modality, the identity

`gamma_global + delta_gamma + delta_node = gamma_global + delta_gamma_effective + delta_node_centered`

was checked with maximum absolute error below `1e-6`. The exact errors are stored in each `canonical_decomposition.pt` and JSON metadata file.

## Frozen results

The table reports test-accuracy drops relative to Original. Positive values mean degradation under the counterfactual. Node-shuffle is the mean ± population standard deviation over the fixed permutation seeds 0–9.

| Dataset | mean centered node std | modality profile distance | effective radius | Node-mean drop | Node-shuffle drop | Modality-mean drop | Modality-swap drop |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The full Val Acc, Val Macro-F1, Test Acc, and Test Macro-F1 values are in `counterfactual_summary.csv`; per-permutation results are in `node_shuffle_summary.json`.

## Observational evidence

- Node residual variability is heterogeneous across datasets; the structural values are descriptive statistics, not evidence of function by themselves.
- Modality profile distance and effective radius summarize the learned checkpoint profiles. They are likewise observational and do not establish that the corresponding corrections are used functionally.

## Frozen counterfactual evidence

- Node-mean removes node-specific variation while retaining the shared node-generator contribution. A positive drop is evidence that node-specific variation contributes under a frozen intervention.
- Node-shuffle preserves each modality's complete K+1 profile distribution, including order-wise means, standard deviations, quantiles, and the overall coefficient multiset, while breaking node-to-profile alignment. A positive drop supports functional importance of alignment.
- Modality-mean removes the systematic text-versus-visual correction difference while retaining node residuals. Modality-swap preserves correction magnitudes but breaks modality assignment. A positive drop in either is counterfactual evidence for modality-level use; coefficient distance alone is insufficient.

## Dataset-level interpretation

- Largest node-profile-alignment drop: **{highest_node['dataset']}** (`{pct(highest_node['node_shuffle']['mean_drop']['delta_test_acc'])}` mean Test Acc drop).
- Largest modality-mean drop: **{highest_modality['dataset']}** (`{pct(highest_modality['conditions']['modality_mean']['delta_test_acc'])}` Test Acc drop).
- Across the five datasets, node-shuffle gives a positive drop above 0.1 percentage points in at least one dataset: **{str(node_alignment_supported).lower()}**. Node-mean gives such a drop in at least one dataset: **{str(node_supported).lower()}**.
- At least one modality counterfactual gives a positive drop above 0.1 percentage points: **{str(modality_supported).lower()}**.

These are descriptive frozen-checkpoint findings. They do not establish retraining ablation results, statistical significance, or causal claims beyond the stated interventions.

## Descriptive cross-dataset correlations

With only five datasets, these Pearson correlations are descriptive summaries only and must not be interpreted as significance tests or strong causal evidence.

| Structural quantity | Counterfactual degradation | Pearson r |
|---|---|---:|
| mean centered node std | Node-shuffle Test Acc drop | {_safe_corr(node_var, node_shuffle_drop)} |
| mean centered node std | Node-mean Test Acc drop | {_safe_corr(node_var, node_mean_drop)} |
| mean modality profile distance | Modality-mean Test Acc drop | {_safe_corr(modality_distance, modality_mean_drop)} |
| mean modality profile distance | Modality-swap Test Acc drop | {_safe_corr(modality_distance, modality_swap_drop)} |

## Required answers

1. **Is node-level personalization functionally used?** The answer should be read from Node-mean drops above: positive drops indicate frozen evidence for use on the corresponding datasets; near-zero drops do not support a universal claim.
2. **Is node-profile alignment important?** Node-shuffle is the direct test. Positive mean drops indicate that assignment of a complete profile to its original node matters while the profile distribution is held fixed.
3. **Is modality-level personalization functionally used?** Only datasets with positive Modality-mean or Modality-swap drops provide frozen evidence. Modality coefficient distance alone is not enough.
4. **Which datasets rely most on each level?** By this protocol, `{highest_node['dataset']}` has the largest mean Node-shuffle degradation and `{highest_modality['dataset']}` has the largest Modality-mean degradation. The complete ranking is in the CSV.
5. **Does the evidence justify keeping the node residual module?** The module is justified as a mechanism worth retaining when Node-mean and/or Node-shuffle produce meaningful positive degradation on the target datasets. The results support a dataset-conditional conclusion, not a universal claim that every dataset requires node personalization.

## Unsupported claims

This stage does not support claims that node residuals are universally necessary, that modality distance causes performance differences, that frozen counterfactual drops equal retraining ablation effects, or that any observed cross-dataset correlation is statistically significant.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--checkpoint-root", type=Path, default=ROOT / "outputs" / "mechanism_audit"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "m1b_counterfactual"
    )
    parser.add_argument(
        "--shuffle-seeds",
        type=int,
        nargs="*",
        default=list(DEFAULT_SHUFFLE_SEEDS),
        help="Fixed node-shuffle permutation seeds; default: 0..9",
    )
    args = parser.parse_args()
    if not args.shuffle_seeds:
        raise ValueError("At least one shuffle seed is required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    results = [
        _run_dataset(
            dataset,
            args.checkpoint_root,
            args.output_root,
            device,
            tuple(args.shuffle_seeds),
        )
        for dataset in datasets
    ]
    report_path = ROOT / "docs" / "mopf_m1b_counterfactual.md"
    _write_report(results, report_path)
    _dump_json(
        args.output_root / "m1b_summary.json",
        {
            "datasets": list(datasets),
            "shuffle_seeds": list(args.shuffle_seeds),
            "results": results,
            "report": str(report_path),
        },
    )
    print(f"[m1b] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
