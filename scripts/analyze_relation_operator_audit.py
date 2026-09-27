from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_relation_operator_audit import (  # noqa: E402
    DATASETS, FORMAL_FIELDS, OUTPUT, SEEDS, VARIANTS,
)
from src.data import load_mag_data  # noqa: E402
from src.models.relation_operator_audit import Model  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402

RESULTS = ROOT / "results/relation_operator_audit_v1"
COMPARISONS = [
    ("relation_expert_minus_global_expert", "relation_expert", "global_expert"),
    ("relation_expert_minus_scalar_weight", "relation_expert", "scalar_weight"),
    ("relation_expert_minus_plain", "relation_expert", "plain"),
    ("global_expert_minus_plain", "global_expert", "plain"),
]
INTERVENTIONS = ["shuffle", "global_mean", "zero_expert", "mean_expert"]
VALIDATION_FIELDS = [
    "variant", "dataset", "seed", "best_epoch", "val_acc", "val_macro_f1",
    "runtime_seconds", "peak_gpu_memory_mib", "model_trainable_params",
    "head_trainable_params", "total_trainable_params", "checkpoint", "run_dir",
    "test_evaluated", "status",
]
PARAMETER_FIELDS = [
    "variant", "dataset", "model_trainable_params", "head_trainable_params",
    "total_trainable_params",
]
INTERVENTION_FIELDS = [
    "variant", "dataset", "seed", "intervention", "val_acc", "val_macro_f1",
    "delta_val_acc", "delta_val_macro_f1", "prediction_flip_rate",
    "routing_change", "mean_message_change", "selected_edges", "test_evaluated",
]
SPECIALIZATION_FIELDS = [
    "variant", "dataset", "seed", "modality", "diagnostic", "expert", "value",
    "routing_entropy", "top1_fraction", "collapse_flag",
]
RELATION_FIELDS = [
    "variant", "dataset", "seed", "modality", "diagnostic", "similarity_quartile",
    "hop", "n_edges", "train_similarity_q25", "train_similarity_q50",
    "train_similarity_q75", "similarity_mean", "mean", "median", "p10", "p90",
    "correction_relative_mean", "correction_relative_p10",
    "correction_relative_median", "correction_relative_p90",
    "message_direction_cosine_mean", "message_direction_cosine_p10",
    "message_direction_cosine_median", "message_direction_cosine_p90",
    "mean_expert_usage_0", "mean_expert_usage_1", "mean_expert_usage_2",
    "mean_expert_usage_3",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_directory(variant: str, dataset: str, seed: int) -> Path:
    return OUTPUT / "runs" / variant / dataset / f"seed{seed}"


def checkpoint_path(variant: str, dataset: str, seed: int) -> Path:
    return OUTPUT / "checkpoints" / variant / dataset / f"seed{seed}.pt"


def _latest_manifest_rows() -> list[dict]:
    path = OUTPUT / "formal_run_manifest.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_formal_manifest() -> list[dict]:
    rows = _latest_manifest_rows()
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "formal_run_manifest.csv", rows, FORMAL_FIELDS)
    return rows


def _read_scalar(metrics: dict, key: str):
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def _load_completed_runs() -> tuple[list[dict], list[dict]]:
    runs, parameter_rows = [], []
    parameter_cache = {}
    for variant in VARIANTS:
        for dataset in DATASETS:
            for seed in SEEDS:
                run_dir = run_directory(variant, dataset, seed)
                ckpt_path = checkpoint_path(variant, dataset, seed)
                metrics_path = run_dir / "metrics.json"
                config_path = run_dir / "resolved_config.json"
                marker_path = run_dir / "complete.marker"
                if not all(path.is_file() for path in (ckpt_path, metrics_path, config_path, marker_path)):
                    continue
                payload = read_json(metrics_path)
                config = read_json(config_path)
                metrics = payload.get("metrics", {})
                if config.get("task", {}).get("evaluate_test") is not False:
                    raise RuntimeError(f"test evaluation was enabled in {config_path}")
                if any("test" in key.lower() for key in metrics):
                    raise RuntimeError(f"test metric present in {metrics_path}")
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if any("test" in key.lower() for key in ckpt.get("metrics", {})):
                    raise RuntimeError(f"test metric present in checkpoint {ckpt_path}")
                row = {
                    "variant": variant,
                    "dataset": dataset,
                    "seed": int(seed),
                    "best_epoch": payload.get("best_epoch", ckpt.get("epoch")),
                    "val_acc": _read_scalar(metrics, "val_acc"),
                    "val_macro_f1": _read_scalar(metrics, "val_macro_f1"),
                    "runtime_seconds": payload.get("runtime_seconds"),
                    "peak_gpu_memory_mib": payload.get("peak_gpu_memory_mib"),
                    "model_trainable_params": None,
                    "head_trainable_params": None,
                    "total_trainable_params": None,
                    "checkpoint": str(ckpt_path),
                    "run_dir": str(run_dir),
                    "test_evaluated": False,
                    "status": "complete",
                }
                cache_key = (variant, dataset)
                if cache_key not in parameter_cache:
                    cfg = OmegaConf.create(config)
                    model = Model(cfg, ckpt["data_info"])
                    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    head_params = int(ckpt["data_info"]["num_classes"]) * (model.out_dim + 1)
                    parameter_cache[cache_key] = (model_params, head_params)
                    parameter_rows.append({
                        "variant": variant,
                        "dataset": dataset,
                        "model_trainable_params": model_params,
                        "head_trainable_params": head_params,
                        "total_trainable_params": model_params + head_params,
                    })
                    del model
                row["model_trainable_params"], row["head_trainable_params"] = parameter_cache[cache_key]
                row["total_trainable_params"] = sum(parameter_cache[cache_key])
                runs.append(row)
                del ckpt

    known = {(row["variant"], row["dataset"]) for row in parameter_rows}
    missing = [
        (variant, dataset)
        for variant in VARIANTS
        for dataset in DATASETS
        if (variant, dataset) not in known
    ]
    if missing:
        data_info_by_dataset = {}
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            for variant, dataset in missing:
                if dataset not in data_info_by_dataset:
                    cfg = compose(
                        config_name="config",
                        overrides=[f"dataset={dataset}", "task=nc", "model=relation_operator_audit", "seed=42"],
                    )
                    data = load_mag_data(cfg, "nc", 42)
                    data_info_by_dataset[dataset] = {
                        "input_dim": data.input_dim,
                        "num_nodes": data.num_nodes,
                        "num_classes": data.num_classes,
                        "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
                        "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0,
                    }
                    del data
                cfg = compose(
                    config_name="config",
                    overrides=[
                        f"dataset={dataset}", "task=nc", "model=relation_operator_audit",
                        f"model.operator_variant={variant}", "seed=42",
                    ],
                )
                model = Model(cfg, data_info_by_dataset[dataset])
                model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                head_params = int(data_info_by_dataset[dataset]["num_classes"]) * (model.out_dim + 1)
                parameter_rows.append({
                    "variant": variant,
                    "dataset": dataset,
                    "model_trainable_params": model_params,
                    "head_trainable_params": head_params,
                    "total_trainable_params": model_params + head_params,
                })
                del model
    return runs, parameter_rows


def _metric_comparisons(runs: list[dict]) -> tuple[list[dict], dict]:
    lookup = {(r["variant"], r["dataset"], int(r["seed"])): r for r in runs}
    rows, summaries = [], {}
    for name, candidate, baseline in COMPARISONS:
        for metric in ("val_acc", "val_macro_f1"):
            pairs = []
            for dataset in DATASETS:
                for seed in SEEDS:
                    a = lookup.get((candidate, dataset, seed))
                    b = lookup.get((baseline, dataset, seed))
                    if a is not None and b is not None:
                        pairs.append((dataset, seed, float(a[metric]) - float(b[metric])))
            deltas = np.asarray([item[2] for item in pairs], dtype=float)
            by_dataset = {
                dataset: [delta for ds, _, delta in pairs if ds == dataset]
                for dataset in DATASETS
            }
            dataset_means = {
                dataset: (float(np.mean(values)) if values else None)
                for dataset, values in by_dataset.items()
            }
            row = {
                "comparison": name,
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "aggregate_mean_delta": float(deltas.mean()) if deltas.size else None,
                "population_sd": float(deltas.std(ddof=0)) if deltas.size else None,
                "positive_seed_pairs": int((deltas > 0).sum()) if deltas.size else 0,
                "complete_seed_pairs": int(deltas.size),
                "expected_seed_pairs": 15,
                "positive_dataset_means": sum(v is not None and v > 0 for v in dataset_means.values()),
                "nonnegative_dataset_means": sum(v is not None and v >= 0 for v in dataset_means.values()),
                "expected_datasets": 5,
                **{f"mean_delta_{dataset}": dataset_means[dataset] for dataset in DATASETS},
            }
            rows.append(row)
            summaries[(name, metric)] = row
    return rows, summaries


def _validation_metrics(head, embedding: torch.Tensor, data, eval_labels: list[int]):
    val_idx = data.val_idx.to(embedding.device)
    logits = head(embedding[val_idx])
    pred = logits.argmax(dim=-1)
    target = data.y[data.val_idx].to(embedding.device)
    acc = float((pred == target).float().mean().item())
    macro = float(f1_score(
        target.detach().cpu().numpy(),
        pred.detach().cpu().numpy(),
        labels=eval_labels,
        average="macro",
        zero_division=0,
    ))
    return acc, macro, pred.detach().cpu()


def _load_checkpoint_context(variant: str, dataset: str, seed: int, device: torch.device):
    run_dir = run_directory(variant, dataset, seed)
    config = read_json(run_dir / "resolved_config.json")
    payload = torch.load(checkpoint_path(variant, dataset, seed), map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(config)
    data = load_mag_data(cfg, "nc", seed)
    model = Model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    head = torch.nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"])).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    head.eval()
    eval_labels = _resolve_nc_eval_labels(data, include_test=False)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    return cfg, data, model, head, payload, eval_labels, x, edge_index


@torch.no_grad()
def _run_interventions(runs: list[dict], device: torch.device) -> tuple[list[dict], list[dict]]:
    rows, specialization = [], []
    by_key = {(r["variant"], r["dataset"], int(r["seed"])): r for r in runs}
    for dataset in DATASETS:
        for seed in SEEDS:
            a3 = by_key.get(("relation_expert", dataset, seed))
            if a3 is None:
                continue
            _, data, model, head, _, labels, x, edge_index = _load_checkpoint_context(
                "relation_expert", dataset, seed, device
            )
            with torch.no_grad():
                normal = model.analyze(x, edge_index)
                normal_acc, normal_f1, normal_pred = _validation_metrics(head, normal["fused_z"], data, labels)
            for modality in ("text", "visual"):
                stats = normal[f"routing_stats_{modality}"]
                for expert_id, load in enumerate(stats.get("expert_load", torch.empty(0)).detach().cpu().tolist()):
                    specialization.append({
                        "variant": "relation_expert", "dataset": dataset, "seed": seed,
                        "modality": modality, "diagnostic": "expert_load",
                        "expert": expert_id, "value": load,
                        "routing_entropy": float(stats["routing_entropy"].item()),
                        "top1_fraction": float(stats["top1_fraction"][expert_id].item()),
                        "collapse_flag": "EXPERT_COLLAPSE_OBSERVED" if load > 0.90 else "",
                    })
                h0 = normal[f"H0_{modality}"]
                train_idx = data.train_idx.to(device)
                generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + (0 if modality == "text" else 1))
                count = min(4096, int(train_idx.numel()))
                chosen = train_idx[torch.randperm(train_idx.numel(), generator=generator)[:count].to(device)]
                same_input = h0[chosen]
                expert_outputs = [expert(same_input) for expert in getattr(model, f"experts_{modality}")]
                for left in range(len(expert_outputs)):
                    for right in range(left + 1, len(expert_outputs)):
                        cosine = F.cosine_similarity(expert_outputs[left], expert_outputs[right], dim=-1).mean()
                        specialization.append({
                            "variant": "relation_expert", "dataset": dataset, "seed": seed,
                            "modality": modality, "diagnostic": "pairwise_expert_output_cosine",
                            "expert": f"{left}:{right}", "value": float(cosine.item()),
                        })
            rows.append({
                "variant": "relation_expert", "dataset": dataset, "seed": seed,
                "intervention": "normal", "val_acc": normal_acc, "val_macro_f1": normal_f1,
                "delta_val_acc": 0.0, "delta_val_macro_f1": 0.0,
                "prediction_flip_rate": 0.0, "routing_change": 0.0,
                "mean_message_change": 0.0, "test_evaluated": False,
            })
            for intervention in INTERVENTIONS:
                with torch.no_grad():
                    changed = model.analyze(
                        x, edge_index, intervention=intervention,
                        intervention_seed=seed * 10007,
                    )
                    acc, macro, pred = _validation_metrics(head, changed["fused_z"], data, labels)
                flip = float((pred != normal_pred).float().mean().item())
                ints = changed["intervention_stats"]
                routing_values = [ints[f"routing_change_{m}"] for m in ("text", "visual")]
                message_values = [ints[f"mean_message_change_{m}"] for m in ("text", "visual")]
                routing_change = float(torch.stack([v for v in routing_values if v is not None]).mean().item()) if any(v is not None for v in routing_values) else 0.0
                message_change = float(torch.stack([v for v in message_values if v is not None]).mean().item()) if any(v is not None for v in message_values) else 0.0
                rows.append({
                    "variant": "relation_expert", "dataset": dataset, "seed": seed,
                    "intervention": intervention, "val_acc": acc, "val_macro_f1": macro,
                    "delta_val_acc": acc - normal_acc, "delta_val_macro_f1": macro - normal_f1,
                    "prediction_flip_rate": flip, "routing_change": routing_change,
                    "mean_message_change": message_change, "test_evaluated": False,
                })
            del model, head, data, normal
    return rows, specialization


def _semantic_bins(model, analysis, data, target, source, modality: str):
    device = target.device
    h0 = analysis[f"H0_{modality}"]
    train_nodes = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_nodes = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_nodes[data.train_idx.to(device)] = True
    val_nodes[data.val_idx.to(device)] = True
    similarity_parts = []
    for start in range(0, target.numel(), model.edge_chunk_size):
        end = min(start + model.edge_chunk_size, target.numel())
        similarity_parts.append(
            F.cosine_similarity(h0[target[start:end]], h0[source[start:end]], dim=-1)
        )
    similarity = torch.cat(similarity_parts) if similarity_parts else h0.new_empty((0,))
    train_edge = train_nodes[target] & train_nodes[source]
    val_related = val_nodes[target] | val_nodes[source]
    if not train_edge.any():
        return None
    thresholds = torch.quantile(similarity[train_edge], torch.tensor([0.25, 0.5, 0.75], device=device))
    positions = torch.nonzero(val_related, as_tuple=False).flatten()
    bins = torch.bucketize(similarity[positions], thresholds)
    return thresholds, positions, bins, similarity


@torch.no_grad()
def _quartile_relation_rows(
    variant: str, dataset: str, seed: int, model, analysis, data,
    target: torch.Tensor, source: torch.Tensor, modality: str,
) -> tuple[list[dict], dict | None]:
    if variant not in {"scalar_weight", "relation_expert"}:
        return [], None
    semantic = _semantic_bins(model, analysis, data, target, source, modality)
    if semantic is None:
        return [], None
    thresholds, positions, bins, similarity = semantic
    target_val, source_val = target[positions], source[positions]
    h0 = analysis[f"H0_{modality}"]
    control, _ = model._edge_controls(modality, h0, target_val, source_val)
    rows = []
    if variant == "scalar_weight":
        for q in range(4):
            selected = bins == q
            vals = control[selected].detach()
            if not vals.numel():
                continue
            quantiles = torch.quantile(vals, torch.tensor([0.1, 0.9], device=vals.device))
            rows.append({
                "variant": variant, "dataset": dataset, "seed": seed,
                "modality": modality, "diagnostic": "scalar_weight",
                "similarity_quartile": q + 1, "hop": "all",
                "n_edges": int(vals.numel()), "train_similarity_q25": float(thresholds[0].item()),
                "train_similarity_q50": float(thresholds[1].item()),
                "train_similarity_q75": float(thresholds[2].item()),
                "similarity_mean": float(similarity[positions[selected]].mean().item()),
                "mean": float(vals.mean().item()), "median": float(vals.median().item()),
                "p10": float(quantiles[0].item()), "p90": float(quantiles[1].item()),
            })
        return rows, {"thresholds": thresholds, "positions": positions, "bins": bins}

    experts = getattr(model, f"experts_{modality}")
    for q in range(4):
        selected = bins == q
        local_positions = torch.nonzero(selected, as_tuple=False).flatten()
        if not local_positions.numel():
            continue
        route_q = control[local_positions]
        for hop in range(1, 4):
            magnitudes, directions = [], []
            for start in range(0, local_positions.numel(), model.edge_chunk_size):
                ids = local_positions[start : start + model.edge_chunk_size]
                src_ids = source_val[ids]
                src_state = analysis[f"C_{modality}"][hop - 1][src_ids]
                expert_values = torch.stack([expert(src_state) for expert in experts], dim=1)
                delta = (expert_values * control[ids].unsqueeze(-1)).sum(dim=1)
                magnitudes.append(delta.norm(dim=-1) / (src_state.norm(dim=-1) + 1e-12))
                directions.append(F.cosine_similarity(src_state, src_state + delta, dim=-1))
            magnitude = torch.cat(magnitudes).detach()
            direction = torch.cat(directions).detach()
            mag_q = torch.quantile(magnitude, torch.tensor([0.1, 0.5, 0.9], device=magnitude.device))
            dir_q = torch.quantile(direction, torch.tensor([0.1, 0.5, 0.9], device=direction.device))
            row = {
                "variant": variant, "dataset": dataset, "seed": seed,
                "modality": modality, "diagnostic": "relative_operator_correction",
                "similarity_quartile": q + 1, "hop": hop,
                "n_edges": int(magnitude.numel()),
                "train_similarity_q25": float(thresholds[0].item()),
                "train_similarity_q50": float(thresholds[1].item()),
                "train_similarity_q75": float(thresholds[2].item()),
                "similarity_mean": float(similarity[positions[selected]].mean().item()),
                "correction_relative_mean": float(magnitude.mean().item()),
                "correction_relative_p10": float(mag_q[0].item()),
                "correction_relative_median": float(mag_q[1].item()),
                "correction_relative_p90": float(mag_q[2].item()),
                "message_direction_cosine_mean": float(direction.mean().item()),
                "message_direction_cosine_p10": float(dir_q[0].item()),
                "message_direction_cosine_median": float(dir_q[1].item()),
                "message_direction_cosine_p90": float(dir_q[2].item()),
            }
            for expert_id in range(model.NUM_EXPERTS):
                row[f"mean_expert_usage_{expert_id}"] = float(route_q[:, expert_id].mean().item())
            rows.append(row)
    return rows, {"thresholds": thresholds, "positions": positions, "bins": bins}


@torch.no_grad()
def _targeted_edge_interventions(
    variant: str, dataset: str, seed: int, model, data, x, edge_index,
    labels: list[int], head, analysis, target, source,
) -> list[dict]:
    if variant != "relation_expert":
        return []
    rows = []
    edge_count = int(target.numel())
    for modality in ("text", "visual"):
        semantic = _semantic_bins(model, analysis, data, target, source, modality)
        if semantic is None:
            continue
        _, positions, bins, _ = semantic
        val_positions = positions
        for q in range(4):
            group = val_positions[bins == q]
            if not group.numel():
                continue
            candidates = val_positions
            generator = torch.Generator(device="cpu").manual_seed(seed * 100003 + q * 101 + (0 if modality == "text" else 1))
            chosen_random = candidates[torch.randperm(candidates.numel(), generator=generator)[: group.numel()].to(target.device)]
            for name, selected in ((f"remove_{modality}_expert_residual_val_similarity_q{q + 1}", group),
                                   (f"matched_random_{modality}_edge_group_q{q + 1}", chosen_random)):
                scale_t = torch.ones(edge_count, device=target.device)
                scale_v = torch.ones(edge_count, device=target.device)
                (scale_t if modality == "text" else scale_v)[selected] = 0.0
                with torch.no_grad():
                    result = model.analyze(
                        x, edge_index,
                        residual_scale_text=scale_t,
                        residual_scale_visual=scale_v,
                    )
                    acc, macro, pred = _validation_metrics(head, result["fused_z"], data, labels)
                normal_acc, normal_macro, normal_pred = _validation_metrics(
                    head, analysis["fused_z"], data, labels
                )
                stats = result["intervention_stats"]
                changed = stats[f"mean_message_change_{modality}"]
                rows.append({
                    "variant": variant, "dataset": dataset, "seed": seed,
                    "intervention": name, "val_acc": acc, "val_macro_f1": macro,
                    "delta_val_acc": acc - normal_acc,
                    "delta_val_macro_f1": macro - normal_macro,
                    "prediction_flip_rate": float((pred != normal_pred).float().mean().item()),
                    "routing_change": 0.0,
                    "mean_message_change": float(changed.item()) if changed is not None else 0.0,
                    "selected_edges": int(selected.numel()),
                    "test_evaluated": False,
                })
    return rows


@torch.no_grad()
def _mechanism_analysis(
    runs: list[dict], device: torch.device, *, targeted: bool
) -> tuple[list[dict], list[dict], list[dict]]:
    intervention_rows, specialization_rows, relation_rows = [], [], []
    run_map = {(r["variant"], r["dataset"], int(r["seed"])): r for r in runs}
    for variant in ("scalar_weight", "global_expert", "relation_expert"):
        for dataset in DATASETS:
            for seed in SEEDS:
                if (variant, dataset, seed) not in run_map:
                    continue
                _, data, model, head, _, labels, x, edge_index = _load_checkpoint_context(
                    variant, dataset, seed, device
                )
                with torch.no_grad():
                    analysis = model.analyze(x, edge_index)
                if variant == "relation_expert":
                    rows, expert_rows = _run_interventions(
                        [run_map[(variant, dataset, seed)]], device
                    )
                    intervention_rows.extend(rows)
                    specialization_rows.extend(expert_rows)
                operator = model._get_operator(edge_index, data.num_nodes, x.dtype)
                target, source, _, _, _, _ = model._operator_edges(operator)
                for modality in ("text", "visual"):
                    diag_rows, semantic = _quartile_relation_rows(
                        variant, dataset, seed, model, analysis, data,
                        target, source, modality,
                    )
                    relation_rows.extend(diag_rows)
                    if targeted and variant == "relation_expert" and semantic is not None:
                        # The targeted utility evaluates validation only; no test indices or labels are read.
                        intervention_rows.extend(_targeted_edge_interventions(
                            variant, dataset, seed, model, data, x, edge_index,
                            labels, head, analysis, target, source,
                        ))
                if variant == "global_expert":
                    for modality in ("text", "visual"):
                        stats = analysis[f"routing_stats_{modality}"]
                        for expert_id, load in enumerate(stats["expert_load"].detach().cpu().tolist()):
                            specialization_rows.append({
                                "variant": variant, "dataset": dataset, "seed": seed,
                                "modality": modality, "diagnostic": "expert_load",
                                "expert": expert_id, "value": load,
                                "routing_entropy": float(stats["routing_entropy"].item()),
                                "top1_fraction": float(stats["top1_fraction"][expert_id].item()),
                            })
                        outputs = [expert(analysis[f"H0_{modality}"][data.train_idx[:min(4096, len(data.train_idx))].to(device)])
                                   for expert in getattr(model, f"experts_{modality}")]
                        for left in range(len(outputs)):
                            for right in range(left + 1, len(outputs)):
                                specialization_rows.append({
                                    "variant": variant, "dataset": dataset, "seed": seed,
                                    "modality": modality, "diagnostic": "pairwise_expert_output_cosine",
                                    "expert": f"{left}:{right}",
                                    "value": float(F.cosine_similarity(outputs[left], outputs[right], dim=-1).mean().item()),
                                    "collapse_flag": (
                                        "EXPERT_COLLAPSE_OBSERVED"
                                        if F.cosine_similarity(outputs[left], outputs[right], dim=-1).mean().item() > 0.98
                                        else ""
                                    ),
                                })
                del model, head, data, analysis
    return intervention_rows, specialization_rows, relation_rows


def _decision_rows(
    runs: list[dict], comparisons: dict, interventions: list[dict], specialization_rows: list[dict]
) -> list[dict]:
    complete = len(runs)
    rows = []
    h1_acc = comparisons[("relation_expert_minus_global_expert", "val_acc")]
    h1_f1 = comparisons[("relation_expert_minus_global_expert", "val_macro_f1")]
    h1_ready = h1_acc["complete_seed_pairs"] == 15 and h1_f1["complete_seed_pairs"] == 15
    h1_pass = h1_ready and (
        h1_acc["positive_dataset_means"] >= 3
        and h1_acc["positive_seed_pairs"] >= 9
        and h1_acc["aggregate_mean_delta"] > 0
        and h1_f1["aggregate_mean_delta"] >= 0
    )
    rows.append({
        "hypothesis": "H1_relation_conditioning_beyond_capacity",
        "status": "RELATION_CONDITIONING_PROMISING" if h1_pass else ("PENDING" if not h1_ready else "GATE_NOT_MET"),
        "complete_formal_contexts": complete, "expected_formal_contexts": 60,
        "accuracy_dataset_means_positive": h1_acc["positive_dataset_means"],
        "accuracy_seed_pairs_positive": h1_acc["positive_seed_pairs"],
        "mean_delta_accuracy": h1_acc["aggregate_mean_delta"],
        "mean_delta_macro_f1": h1_f1["aggregate_mean_delta"],
        "criteria": "A3-A2: >=3/5 positive dataset means; >=9/15 positive seed pairs; mean Accuracy >0; mean Macro-F1 >=0",
    })
    h2_acc = comparisons[("relation_expert_minus_scalar_weight", "val_acc")]
    h2_f1 = comparisons[("relation_expert_minus_scalar_weight", "val_macro_f1")]
    h2_ready = h2_acc["complete_seed_pairs"] == 15 and h2_f1["complete_seed_pairs"] == 15
    h2_metrics_pass = h2_ready and (
        h2_acc["aggregate_mean_delta"] >= 0
        and h2_f1["aggregate_mean_delta"] >= 0
        and h2_acc["nonnegative_dataset_means"] >= 3
    )
    rows.append({
        "hypothesis": "H2_transformation_vs_scalar_weighting",
        "status": "METRIC_GATES_PASS_MECHANISM_REVIEW_REQUIRED" if h2_metrics_pass else ("PENDING" if not h2_ready else "GATE_NOT_MET"),
        "complete_formal_contexts": complete, "expected_formal_contexts": 60,
        "accuracy_dataset_means_nonnegative": h2_acc["nonnegative_dataset_means"],
        "mean_delta_accuracy": h2_acc["aggregate_mean_delta"],
        "mean_delta_macro_f1": h2_f1["aggregate_mean_delta"],
        "most_negative_dataset_mean_accuracy_delta": min(
            [h2_acc[f"mean_delta_{d}"] for d in DATASETS if h2_acc[f"mean_delta_{d}"] is not None],
            default=None,
        ),
        "criteria": "A3-A1 aggregate Accuracy/F1 >=0; >=3/5 dataset Accuracy means >=0; review catastrophic transfer and mechanism evidence descriptively",
    })
    h3_rows = [r for r in interventions if r.get("intervention") in {"shuffle", "global_mean", "zero_expert"}]
    h3_contexts = {(r["dataset"], int(r["seed"])) for r in h3_rows}
    all_no_effect = bool(h3_rows) and all(
        abs(float(r["delta_val_acc"])) <= 1e-6
        and abs(float(r["delta_val_macro_f1"])) <= 1e-6
        and float(r["prediction_flip_rate"]) <= 1e-6
        and float(r["routing_change"]) <= 1e-6
        and float(r["mean_message_change"]) <= 1e-6
        for r in h3_rows
    )
    h3_ready = len(h3_contexts) == 15 and all(
        sum(r["dataset"] == ds and int(r["seed"]) == seed for r in h3_rows) == 3
        for ds in DATASETS for seed in SEEDS
    )
    rows.append({
        "hypothesis": "H3_relation_conditioning_functionally_used",
        "status": "NO_FUNCTIONAL_EFFECT_OBSERVED" if h3_ready and all_no_effect else ("FUNCTIONAL_EFFECT_OBSERVED" if h3_ready else "PENDING"),
        "complete_formal_contexts": complete, "expected_formal_contexts": 60,
        "intervention_contexts": len(h3_contexts), "expected_intervention_contexts": 15,
        "all_interventions_no_effect": all_no_effect if h3_ready else None,
        "criteria": "Validation-only routing shuffle, modality-global route, and zero residual; compare task metrics, flips, message change, and routing change",
    })
    collapse_rows = [
        row for row in specialization_rows
        if row.get("collapse_flag") == "EXPERT_COLLAPSE_OBSERVED"
    ]
    mean_expert_rows = [
        row for row in interventions if row.get("intervention") == "mean_expert"
    ]
    if collapse_rows or (
        mean_expert_rows and all(
            abs(float(row["delta_val_acc"])) <= 1e-6
            and abs(float(row["delta_val_macro_f1"])) <= 1e-6
            and float(row["prediction_flip_rate"]) <= 1e-6
            for row in mean_expert_rows
        )
    ):
        collapse_status = "EXPERT_COLLAPSE_OBSERVED"
    else:
        collapse_status = "NO_COLLAPSE_TRIGGER" if specialization_rows else "PENDING"
    rows.append({
        "hypothesis": "expert_collapse_diagnostic",
        "status": collapse_status,
        "complete_formal_contexts": complete,
        "expected_formal_contexts": 60,
        "collapse_load_or_cosine_rows": len(collapse_rows),
        "mean_expert_intervention_rows": len(mean_expert_rows),
        "criteria": "Flag if any expert load >90%, pairwise expert-output cosine >0.98, or mean-expert intervention has negligible Validation effect across measured contexts",
    })
    return rows


def _write_report(runs, comparisons, decision_rows, intervention_rows, specialization_rows, parameter_rows, targeted: bool) -> None:
    gate = {row["hypothesis"]: row["status"] for row in decision_rows}
    collapse_status = gate.get("expert_collapse_diagnostic", "PENDING")
    lines = [
        "# P0 — Relation-Conditioned Semantic Operator Audit",
        "",
        f"Completed formal contexts: **{len(runs)}/60**. No formal runs are launched by this analyzer.",
        "",
        "## Registered question and controls",
        "",
        "On the same unit-weight, symmetrically normalized physical graph, does relation-conditioned semantic transformation improve NC over plain physical propagation or scalar semantic weighting? All variants use independent Text/Visual projectors, K=3, uniform four-state readout, and CMRF plain fusion.",
        "Only Validation Accuracy selects checkpoints. Macro-F1 is reported. `task.evaluate_test=false`; the analyzer checks checkpoint and metrics payloads for test fields and reads only Train/Validation labels. The data loader reads the dataset's full CPU label tensor as required by its format, while the NC training path moves only Train targets to the device.",
        "",
        "## Paired comparisons",
        "",
        "| Contrast | Metric | Mean Δ | Population SD | Positive pairs | Positive dataset means |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        mean = "—" if row["aggregate_mean_delta"] is None else f"{row['aggregate_mean_delta']:.6f}"
        sd = "—" if row["population_sd"] is None else f"{row['population_sd']:.6f}"
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {mean} | {sd} | {row['positive_seed_pairs']}/{row['expected_seed_pairs']} ({row['complete_seed_pairs']} complete) | {row['positive_dataset_means']}/{row['expected_datasets']} |"
        )
    lines += ["", "## Descriptive gates", ""]
    for row in decision_rows:
        lines.append(f"- **{row['hypothesis']}** — `{row['status']}`. {row['criteria']}")
    lines += [
        "",
        "H2's phrase “catastrophic negative transfer” has no numeric cutoff in the frozen request. The report exposes the most negative per-dataset Accuracy delta for review and does not silently add a threshold. A3-vs-A1 cannot be promoted from metrics alone; correction/routing diagnostics must also be examined.",
        "",
        "## Mechanism and expert diagnostics",
        "",
        f"Validation-only frozen interventions recorded: {len(intervention_rows)} rows. Targeted similarity-quartile edge interventions: {'included' if targeted else 'not run (optional flag not supplied)'}.",
        "`relation_diagnostics.csv` uses cosine similarity in projected H0 space. Quartile boundaries are computed from physical edges whose endpoints are both Train nodes; reported edges are physical edges incident to at least one Validation node. Test labels are not used.",
        f"expert_specialization.csv contains mean load, routing entropy, top-1 fractions, and pairwise output cosine on the same Train-node inputs. Collapse diagnostic: {collapse_status}. No balancing/diversity regularizer is added.",
        "",
        "## Parameter counts",
        "",
        f"Parameter-count rows: {len(parameter_rows)}. Counts distinguish the Text+Visual model, NC classifier head, and their sum. A2/A3 parity is guarded by a unit test.",
        "",
        "## Run integrity",
        "",
        "Every completed run is checked for a resolved config with test evaluation disabled and for absence of test metrics in the checkpoint and run metrics. The formal launcher writes the exact command, git SHA, deterministic run/checkpoint paths, runtime, peak GPU allocation, and failure reason to its manifest.",
    ]
    (RESULTS / "relation_operator_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and audit P0 relation-operator results.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--targeted-edge-interventions", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    copy_formal_manifest()
    runs, parameter_rows = _load_completed_runs()
    comparison_rows, comparison_summaries = _metric_comparisons(runs)
    intervention_rows, specialization_rows, relation_rows = _mechanism_analysis(
        runs, device, targeted=args.targeted_edge_interventions
    )
    decision_rows = _decision_rows(runs, comparison_summaries, intervention_rows, specialization_rows)
    write_csv(RESULTS / "validation_results.csv", runs, VALIDATION_FIELDS)
    write_csv(RESULTS / "parameter_counts.csv", parameter_rows, PARAMETER_FIELDS)
    write_csv(RESULTS / "paired_comparisons.csv", comparison_rows)
    write_csv(RESULTS / "intervention_results.csv", intervention_rows, INTERVENTION_FIELDS)
    write_csv(RESULTS / "expert_specialization.csv", specialization_rows, SPECIALIZATION_FIELDS)
    write_csv(RESULTS / "relation_diagnostics.csv", relation_rows, RELATION_FIELDS)
    write_csv(RESULTS / "decision_matrix.csv", decision_rows)
    _write_report(
        runs, comparison_rows, decision_rows, intervention_rows,
        specialization_rows, parameter_rows, args.targeted_edge_interventions,
    )
    print(f"Aggregated {len(runs)}/60 completed formal contexts into {RESULTS}")


if __name__ == "__main__":
    main()
