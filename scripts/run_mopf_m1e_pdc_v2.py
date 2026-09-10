"""Run the complete M1-E PDC-v2 structural refinement study.

The script retrains C0/C1/C2/C3 under one shared NC protocol, then performs
the mandatory frozen functional and node-profile alignment audits.  It never
runs the M1-D sports follow-up; sports is intentionally out of scope here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import build_model  # noqa: E402
from src.utils.summary import count_parameters  # noqa: E402
from scripts.run_mopf_m1d_pdc_functional_audit import (  # noqa: E402
    EPS,
    SHUFFLE_SEEDS,
    _alignment_for_condition,
    _compose_with_profile,
    _delta_changes,
    _dump_json,
    _load_bundle,
    _logit_probability_change,
    _mean_std,
    _metrics,
    _representation_change,
    _resolve_eval_labels,
    _summary,
    _with_drop,
)
from src.data import load_mag_data  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
K_BY_DATASET = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
}
VARIANTS = ("C0_current", "C1_pdc_v1", "C2_pdc_v2_sep", "C3_pdc_v2_full")
VARIANT_MODES = {
    "C0_current": "absolute",
    "C1_pdc_v1": "pdc",
    "C2_pdc_v2_sep": "pdc_v2_sep",
    "C3_pdc_v2_full": "pdc_v2_full",
}


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_model_params(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    matches = re.findall(r"model params=(\d+)", log_path.read_text(errors="replace"))
    return int(matches[-1]) if matches else None


def _training_command(
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "src/main.py",
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        f"seed={seed}",
        "num_runs=1",
        f"device={device}",
        f"model.max_order={K_BY_DATASET[dataset]}",
        f"model.num_layers={K_BY_DATASET[dataset]}",
        f"+model.node_conditioner_mode={VARIANT_MODES[variant]}",
        "model.hrc_weight=0.0",
        "+model.ppc_weight=0.0",
        "model.edge_weight_mode=separate_cos",
        "task.training_mode=full_graph",
        "task.optimizer=adamw",
        "task.epochs=300",
        "task.lr=1e-3",
        "task.weight_decay=1e-4",
        "task.patience=30",
        "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4",
        "task.evaluate_test=true",
        "task.save_ckpt_path=" + str(output_dir / "best_val_accuracy.pt"),
        "hydra.job.chdir=false",
        "hydra.run.dir=" + str(output_dir),
    ]


def _train_one(
    dataset: str,
    variant: str,
    seed: int,
    device: str,
    output_root: Path,
    force: bool,
) -> dict[str, Any]:
    output_dir = output_root / dataset / variant / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_val_accuracy.pt"
    record_path = output_dir / "training_record.json"
    if checkpoint.is_file() and record_path.is_file() and not force:
        return json.loads(record_path.read_text(encoding="utf-8"))

    log_path = output_dir / "runner.log"
    started = time.perf_counter()
    command = _training_command(dataset, variant, seed, device, output_dir)
    print(f"[m1e] train start dataset={dataset} variant={variant} seed={seed} device={device}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0 or not checkpoint.is_file():
        tail = log_path.read_text(errors="replace")[-5000:]
        failure = {
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "device": device,
            "return_code": return_code,
            "train_time_sec": elapsed,
            "log_tail": tail,
        }
        _dump_json(output_dir / "failure.json", failure)
        raise RuntimeError(f"M1-E training failed for {dataset}/{variant}/seed{seed}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    record = {
        "dataset": dataset,
        "variant": variant,
        "mode": VARIANT_MODES[variant],
        "seed": seed,
        "device": device,
        "checkpoint": str(checkpoint),
        "best_epoch": payload.get("epoch"),
        "selection": payload.get("selection", "best Validation Accuracy"),
        "metrics": payload.get("metrics", {}),
        "train_time_sec": elapsed,
        "parameter_count_from_log": _parse_model_params(output_dir / "main.log"),
        "data_info": payload.get("data_info", {}),
    }
    _dump_json(record_path, record)
    print(
        f"[m1e] train done dataset={dataset} variant={variant} seed={seed} "
        f"time_sec={elapsed:.1f} best_epoch={record['best_epoch']}",
        flush=True,
    )
    return record


def train_all(output_root: Path, devices: tuple[str, ...], force: bool) -> list[dict[str, Any]]:
    jobs = list(itertools.product(DATASETS, VARIANTS, SEEDS))
    jobs_by_device = [jobs[index::len(devices)] for index in range(len(devices))]

    def run_device(device_jobs, device):
        return [
            _train_one(dataset, variant, seed, device, output_root, force)
            for dataset, variant, seed in device_jobs
        ]

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_device, device_jobs, device)
            for device_jobs, device in zip(jobs_by_device, devices, strict=True)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: (DATASETS.index(row["dataset"]), VARIANTS.index(row["variant"]), row["seed"]))
    return records


def _finite_tensor_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensor_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensor_tree(item) for item in value)
    return True


def _pathway_utilization(model, components, mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {"mode": mode, "modalities": {}}
    for modality in ("text", "visual"):
        rho = (
            model.pdc_rho(modality).detach().cpu().tolist()
            if mode != "absolute"
            else [0.0] * (model.max_order + 1)
        )
        aux = components[f"pdc_aux_{modality}"]
        bases = components[f"bases_{modality}"]
        orders = []
        for order, (discrepancy_norm, branch, state_projection) in enumerate(
            zip(aux["discrepancy_norm"], aux["branch"], aux["state_projection"], strict=True)
        ):
            discrepancy_norm_size = torch.linalg.vector_norm(discrepancy_norm, dim=-1)
            branch_size = torch.linalg.vector_norm(branch, dim=-1)
            if mode == "pdc":
                denominator = torch.linalg.vector_norm(bases[order], dim=-1)
            else:
                denominator = torch.linalg.vector_norm(state_projection, dim=-1)
            orders.append(
                {
                    "order": order,
                    "learned_rho": rho[order],
                    "discrepancy_norm": _summary(discrepancy_norm_size),
                    "branch_magnitude": _summary(branch_size),
                    "branch_to_state_ratio": _summary(branch_size / (denominator + EPS)),
                }
            )
        result["modalities"][modality] = {"orders": orders}
    return result


def _stream_change_pair(on, off, classifier, data, device, eval_labels) -> dict[str, Any]:
    eta_text = on["eta_text"] - off["eta_text"]
    eta_visual = on["eta_visual"] - off["eta_visual"]
    delta_changes = _delta_changes(on, off)
    eta_delta_error = max(
        float((eta_text - (on["delta_node_text"] - off["delta_node_text"])).abs().max().item()),
        float((eta_visual - (on["delta_node_visual"] - off["delta_node_visual"])).abs().max().item()),
    )
    return {
        "node_residual_change": delta_changes,
        "eta_change": {
            "text_mean_abs_by_order": eta_text.abs().mean(dim=0).detach().cpu().tolist(),
            "visual_mean_abs_by_order": eta_visual.abs().mean(dim=0).detach().cpu().tolist(),
            "equals_delta_change_max_abs_error": eta_delta_error,
        },
        "modality_representation_change": {
            "text": _representation_change(on["z_text"], off["z_text"]),
            "visual": _representation_change(on["z_visual"], off["z_visual"]),
        },
        "fused_representation_change": _representation_change(on["z"], off["z"]),
        "logit_probability_change": _logit_probability_change(
            classifier, on["z"], off["z"], data
        ),
        "metrics_on": _metrics(classifier, on["z"], data, device, eval_labels),
        "metrics_off": _metrics(classifier, off["z"], data, device, eval_labels),
    }


def _m1e_condition_masks(max_order: int, mode: str) -> dict[str, tuple[list[float], list[float]]]:
    """Return frozen audit masks, including C3's required order-0 mask."""
    ones = [1.0] * (max_order + 1)
    zeros = [0.0] * (max_order + 1)
    masks: dict[str, tuple[list[float], list[float]]] = {
        "F0_pdc_on": (list(ones), list(ones)),
        "F1_all_pdc_off": (list(zeros), list(zeros)),
        "F2_text_pdc_off": (list(zeros), list(ones)),
        "F3_visual_pdc_off": (list(ones), list(zeros)),
    }
    next_index = 4
    if mode == "pdc_v2_full":
        order0 = list(ones)
        order0[0] = 0.0
        masks[f"F{next_index}_order0_off"] = (list(order0), list(order0))
        next_index += 1
    for order in range(1, max_order + 1):
        text = list(ones)
        visual = list(ones)
        text[order] = 0.0
        visual[order] = 0.0
        masks[f"F{next_index}_order{order}_off"] = (text, visual)
        next_index += 1
    return masks


def _paired_alignment_gain(on_alignment: dict[str, Any], off_alignment: dict[str, Any]) -> dict[str, Any]:
    gain: dict[str, Any] = {}
    for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        values = [
            on_row["metrics"][f"delta_{metric}"] - off_row["metrics"][f"delta_{metric}"]
            for on_row, off_row in zip(
                on_alignment["node_shuffle"]["permutations"],
                off_alignment["node_shuffle"]["permutations"],
                strict=True,
            )
        ]
        stats = _summary(values)
        gain[metric] = {
            "mean": stats["mean"],
            "population_std": stats["std"],
            "paired_permutation_se": float(stats["std"] / np.sqrt(len(values))),
            "permutation_count": len(values),
            "permutation_values": values,
        }
    return gain


@torch.no_grad()
def _audit_one(
    train_record: dict[str, Any],
    output_root: Path,
    device: torch.device,
    force: bool,
) -> dict[str, Any]:
    dataset = train_record["dataset"]
    variant = train_record["variant"]
    seed = int(train_record["seed"])
    run_dir = output_root / dataset / variant / f"seed{seed}"
    audit_path = run_dir / "audit.json"
    if audit_path.is_file() and not force:
        return json.loads(audit_path.read_text(encoding="utf-8"))

    payload, cfg, data, model, classifier = _load_bundle(run_dir, device)
    mode = VARIANT_MODES[variant]
    eval_labels = _resolve_eval_labels(data)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    max_order = int(cfg.model.max_order)

    formal, _, _, _, _ = model(x, edge_index)
    if mode == "absolute":
        on = model._encode_components(x, edge_index)
        condition_specs = {"formal": (None, None)}
    else:
        masks = _m1e_condition_masks(max_order, mode)
        on = model.analysis_encode_with_pdc_mask(
            x,
            edge_index,
            text_mask=[1.0] * (max_order + 1),
            visual_mask=[1.0] * (max_order + 1),
        )
        condition_specs = masks

    formal_error = float((formal - on["z"]).abs().max().item())
    # The formal output is only needed for this equivalence check.  Keeping it
    # alive alongside the full analysis component tree is costly on ele-fashion.
    del formal

    conditions: dict[str, Any] = {}
    off = None
    order0_name: str | None = None
    order0_spec: tuple[list[float], list[float]] | None = None
    conditions["F0_pdc_on" if mode != "absolute" else "formal"] = {
        "text_mask": None,
        "visual_mask": None,
        "metrics": _metrics(classifier, on["z"], data, device, eval_labels),
    }
    if mode != "absolute":
        for name, (text_mask, visual_mask) in condition_specs.items():
            components = on if name == "F0_pdc_on" else model.analysis_encode_with_pdc_mask(
                x, edge_index, text_mask=text_mask, visual_mask=visual_mask
            )
            conditions[name] = {
                "text_mask": text_mask,
                "visual_mask": visual_mask,
                "metrics": _metrics(classifier, components["z"], data, device, eval_labels),
            }
            if name == "F1_all_pdc_off":
                off = components
            elif mode == "pdc_v2_full" and "order0_off" in name:
                order0_name = name
                order0_spec = (list(text_mask), list(visual_mask))
            if name != "F0_pdc_on":
                # Retain only the all-off component for the required stream
                # and alignment comparison; do not accumulate K full trees.
                if name != "F1_all_pdc_off":
                    del components
        for name, item in conditions.items():
            item["delta_pdc_on_minus_condition"] = {
                metric: conditions["F0_pdc_on"]["metrics"][metric] - item["metrics"][metric]
                for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
            }

    path_utilization = _pathway_utilization(model, on, mode)
    alignment_on = _alignment_for_condition(
        model, classifier, on, data, device, eval_labels
    )
    alignment: dict[str, Any] = {"PDC-On" if mode != "absolute" else "formal": alignment_on}
    stream_changes: dict[str, Any] | None = None
    c3_order0_analysis: dict[str, Any] | None = None
    if mode != "absolute":
        if off is None:
            raise RuntimeError("M1-E condition audit did not produce F1_all_pdc_off")
        stream_changes = _stream_change_pair(
            on, off, classifier, data, device, eval_labels
        )
        alignment_off = _alignment_for_condition(
            model, classifier, off, data, device, eval_labels
        )
        alignment["PDC-Off"] = alignment_off
        alignment["active_path_alignment_gain"] = _paired_alignment_gain(
            alignment_on, alignment_off
        )
        del alignment_off
        del off

        if mode == "pdc_v2_full":
            if order0_name is None or order0_spec is None:
                raise RuntimeError("C3 audit did not produce order0-off specification")
            order0_off = model.analysis_encode_with_pdc_mask(
                x,
                edge_index,
                text_mask=order0_spec[0],
                visual_mask=order0_spec[1],
            )
            order0_stream = _stream_change_pair(
                on, order0_off, classifier, data, device, eval_labels
            )
            c3_order0_analysis = {
                "rho0_text": float(model.pdc_rho("text")[0].item()),
                "rho0_visual": float(model.pdc_rho("visual")[0].item()),
                "order0_condition": order0_name,
                "delta0_text_rms_change": order0_stream["node_residual_change"]["text"]["orders"][0]["rms_delta_change"],
                "delta0_visual_rms_change": order0_stream["node_residual_change"]["visual"]["orders"][0]["rms_delta_change"],
                "eta0_text_abs_change": order0_stream["eta_change"]["text_mean_abs_by_order"][0],
                "eta0_visual_abs_change": order0_stream["eta_change"]["visual_mean_abs_by_order"][0],
                "stream_changes": order0_stream,
                "metrics_delta_pdc_on_minus_order0_off": conditions[order0_name][
                    "delta_pdc_on_minus_condition"
                ],
            }
            del order0_off

    logits = classifier(on["z"])
    centered_profile_std = {}
    for modality in ("text", "visual"):
        centered = on[f"delta_node_{modality}"] - on[
            f"delta_node_{modality}"
        ].mean(dim=0, keepdim=True)
        centered_profile_std[modality] = float(centered.std(unbiased=False).item())
    pathologies: list[str] = []
    if not _finite_tensor_tree(on):
        pathologies.append("NaN/Inf in PDC-On components")
    if stream_changes is not None and not _finite_tensor_tree(stream_changes):
        pathologies.append("NaN/Inf in stream audit")
    if any(value < 1e-8 for value in centered_profile_std.values()):
        pathologies.append("profile collapse: centered delta std < 1e-8")
    if float(on["z"].abs().max().item()) > 1e4 or float(logits.abs().max().item()) > 1e4:
        pathologies.append("variance explosion: representation/logit abs max > 1e4")
    if int(torch.unique(logits.argmax(dim=-1)).numel()) <= 1:
        pathologies.append("prediction collapse: PDC-On predicts one class")

    record = {
        "dataset": dataset,
        "variant": variant,
        "mode": mode,
        "seed": seed,
        "checkpoint": str(run_dir / "best_val_accuracy.pt"),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_selection": payload.get("selection", "best Validation Accuracy"),
        "max_order": max_order,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "parameter_count": {
            "model": count_parameters(model),
            "classifier": count_parameters(classifier),
            "total": count_parameters(model) + count_parameters(classifier),
        },
        "pdc_rho": {
            modality: model.pdc_rho(modality).detach().cpu().tolist()
            if mode != "absolute"
            else [0.0] * (max_order + 1)
            for modality in ("text", "visual")
        },
        "pathway_utilization": path_utilization,
        "conditions": conditions,
        "stream_changes_pdc_on_vs_off": stream_changes,
        "alignment": alignment,
        "c3_order0_analysis": c3_order0_analysis,
        "detected_pathologies": pathologies,
        "diagnostics": {
            "formal_forward_max_abs_error": formal_error,
            "centered_delta_std": centered_profile_std,
            "training_performed_in_m1e": True,
            "sports_run": False,
            "semantic_graph_modified": False,
            "fusion_modified": False,
            "eta_formula_modified": False,
            "auxiliary_loss_added": False,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(run_dir / "audit.json", record)
    _dump_json(run_dir / "pathway_utilization.json", path_utilization)
    _dump_json(run_dir / "alignment.json", alignment)
    if stream_changes is not None:
        _dump_json(run_dir / "stream_changes.json", stream_changes)
    if c3_order0_analysis is not None:
        _dump_json(run_dir / "c3_order0_analysis.json", c3_order0_analysis)
    del model, classifier, data, on
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def audit_all(
    training_records: list[dict[str, Any]],
    output_root: Path,
    devices: tuple[str, ...],
    force: bool,
) -> list[dict[str, Any]]:
    jobs = list(training_records)
    jobs_by_device = [jobs[index::len(devices)] for index in range(len(devices))]

    def run_device(device_jobs, device):
        return [
            _audit_one(record, output_root, torch.device(device), force)
            for record in device_jobs
        ]

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(run_device, device_jobs, device)
            for device_jobs, device in zip(jobs_by_device, devices, strict=True)
        ]
        for future in as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: (DATASETS.index(row["dataset"]), VARIANTS.index(row["variant"]), row["seed"]))
    return records


def _aggregate_pathway(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"modalities": {}}
    for modality in ("text", "visual"):
        orders = []
        for order in range(int(records[0]["max_order"]) + 1):
            rows = [record["pathway_utilization"]["modalities"][modality]["orders"][order] for record in records]
            orders.append(
                {
                    "order": order,
                    "learned_rho": _mean_std([row["learned_rho"] for row in rows]),
                    "discrepancy_norm_mean": _mean_std([row["discrepancy_norm"]["mean"] for row in rows]),
                    "branch_magnitude_mean": _mean_std([row["branch_magnitude"]["mean"] for row in rows]),
                    "branch_to_state_ratio_mean": _mean_std([row["branch_to_state_ratio"]["mean"] for row in rows]),
                }
            )
        output["modalities"][modality] = {"orders": orders}
    return output


def _aggregate_conditions(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = list(records[0]["conditions"])
    result: dict[str, Any] = {}
    metrics = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    for name in names:
        result[name] = {
            "metrics": {
                metric: _mean_std([record["conditions"][name]["metrics"][metric] for record in records])
                for metric in metrics
            }
        }
        if "delta_pdc_on_minus_condition" in records[0]["conditions"][name]:
            result[name]["delta_pdc_on_minus_condition"] = {
                metric: {
                    **_mean_std([
                        record["conditions"][name]["delta_pdc_on_minus_condition"][metric]
                        for record in records
                    ]),
                    "positive_seed_count": int(sum(
                        record["conditions"][name]["delta_pdc_on_minus_condition"][metric] > 0
                        for record in records
                    )),
                }
                for metric in metrics
            }
    return result


def _aggregate_stream(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    streams = [record["stream_changes_pdc_on_vs_off"] for record in records]
    streams = [stream for stream in streams if stream is not None]
    if not streams:
        return None

    def seed_stats(values: list[float]) -> dict[str, Any]:
        return {**_mean_std(values), "seed_values": values}

    result: dict[str, Any] = {
        "fused_representation_change": {
            metric: seed_stats([stream["fused_representation_change"][metric] for stream in streams])
            for metric in ("mean_cosine_distance", "mean_relative_l2")
        },
        "modality_representation_change": {},
        "logit_probability_change": {},
        "eta_change_equals_delta_change_max_abs_error": seed_stats([
            stream["eta_change"]["equals_delta_change_max_abs_error"] for stream in streams
        ]),
    }
    for modality in ("text", "visual"):
        result["modality_representation_change"][modality] = {
            metric: seed_stats([
                stream["modality_representation_change"][modality][metric] for stream in streams
            ])
            for metric in ("mean_cosine_distance", "mean_relative_l2")
        }
    for split in ("all", "val", "test"):
        result["logit_probability_change"][split] = {
            metric: seed_stats([
                stream["logit_probability_change"][split][metric] for stream in streams
            ])
            for metric in (
                "mean_relative_logit_l2",
                "mean_absolute_probability_change",
                "jensen_shannon_divergence",
                "prediction_flip_rate",
            )
        }
    return result


def _aggregate_alignment(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if records[0]["mode"] == "absolute":
        alignment = [record["alignment"]["formal"] for record in records]
        output["node_shuffle_drop"] = {
            metric: _mean_std([
                item["node_shuffle"]["mean_drop"][f"delta_{metric}"] for item in alignment
            ])
            for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
        }
        return output
    on = [record["alignment"]["PDC-On"] for record in records]
    off = [record["alignment"]["PDC-Off"] for record in records]
    output["node_shuffle_drop_on"] = {
        metric: _mean_std([
            item["node_shuffle"]["mean_drop"][f"delta_{metric}"] for item in on
        ])
        for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }
    output["node_shuffle_drop_off"] = {
        metric: _mean_std([
            item["node_shuffle"]["mean_drop"][f"delta_{metric}"] for item in off
        ])
        for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    }
    gain = [record["alignment"]["active_path_alignment_gain"] for record in records]
    output["active_path_alignment_gain"] = {}
    for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
        values = [item[metric]["mean"] for item in gain]
        ses = [item[metric]["paired_permutation_se"] for item in gain]
        output["active_path_alignment_gain"][metric] = {
            **_mean_std(values),
            "mean_paired_permutation_se": float(np.mean(ses)),
            "positive_seed_count": int(sum(value > 0 for value in values)),
            "seed_values": values,
            "seed_paired_permutation_se": ses,
        }
    return output


def _aggregate_dataset_variant(records: list[dict[str, Any]]) -> dict[str, Any]:
    pathologies = [
        pathology
        for record in records
        for pathology in record["detected_pathologies"]
    ]
    return {
        "n_training_seeds": len(records),
        "seeds": [record["seed"] for record in records],
        "mode": records[0]["mode"],
        "parameter_counts": {
            key: _mean_std([record["parameter_count"][key] for record in records])
            for key in ("model", "classifier", "total")
        },
        "downstream_and_interventions": _aggregate_conditions(records),
        "pathway_utilization": _aggregate_pathway(records),
        "stream_changes_pdc_on_vs_off": _aggregate_stream(records),
        "node_shuffle": _aggregate_alignment(records),
        "pathologies": {
            "count": len(pathologies),
            "by_label": {label: pathologies.count(label) for label in sorted(set(pathologies))},
        },
        "c3_order0_analysis": [
            record["c3_order0_analysis"] for record in records
            if record["c3_order0_analysis"] is not None
        ],
    }


def _variant_dataset_lookup(aggregates: dict[str, Any], variant: str, dataset: str) -> dict[str, Any]:
    return aggregates[f"{dataset}/{variant}"]


def _candidate_recommendations(aggregates: dict[str, Any]) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    current_variant = "C0_current"
    c1_variant = "C1_pdc_v1"
    for variant in (c1_variant, "C2_pdc_v2_sep", "C3_pdc_v2_full"):
        pathway_active_datasets = []
        stronger_datasets = []
        stronger_seed_direction_datasets = []
        alignment_supported_datasets = []
        validation_preserved = True
        pathological_datasets = []
        for dataset in DATASETS:
            item = _variant_dataset_lookup(aggregates, variant, dataset)
            current = _variant_dataset_lookup(aggregates, current_variant, dataset)
            if item["pathologies"]["count"]:
                pathological_datasets.append(dataset)
            fused = item["stream_changes_pdc_on_vs_off"]
            if fused is not None:
                fused_rel = fused["fused_representation_change"]["mean_relative_l2"]["mean"]
                logit_rel = fused["logit_probability_change"]["all"]["mean_relative_logit_l2"]["mean"]
                if fused_rel is not None and logit_rel is not None and fused_rel > 1e-8 and logit_rel > 1e-8:
                    pathway_active_datasets.append(dataset)
                c1_item = _variant_dataset_lookup(aggregates, c1_variant, dataset)
                c1_stream = c1_item["stream_changes_pdc_on_vs_off"]
                if c1_stream is not None:
                    c1_fused = c1_stream["fused_representation_change"]["mean_relative_l2"]
                    c1_logit = c1_stream["logit_probability_change"]["all"]["mean_relative_logit_l2"]
                    candidate_fused = fused["fused_representation_change"]["mean_relative_l2"]
                    candidate_logit = fused["logit_probability_change"]["all"]["mean_relative_logit_l2"]
                    fused_seed_direction = sum(
                        candidate > baseline
                        for candidate, baseline in zip(
                            candidate_fused["seed_values"], c1_fused["seed_values"], strict=True
                        )
                    )
                    logit_seed_direction = sum(
                        candidate > baseline
                        for candidate, baseline in zip(
                            candidate_logit["seed_values"], c1_logit["seed_values"], strict=True
                        )
                    )
                    if fused_seed_direction >= 2 or logit_seed_direction >= 2:
                        stronger_seed_direction_datasets.append(dataset)
                    if (
                        fused_seed_direction >= 2
                        and candidate_fused["mean"] > 1.10 * c1_fused["mean"]
                    ) or (
                        logit_seed_direction >= 2
                        and candidate_logit["mean"] > 1.10 * c1_logit["mean"]
                    ):
                        stronger_datasets.append(dataset)
                gain = item["node_shuffle"]["active_path_alignment_gain"]
                if any(
                    gain[metric]["mean"] > 0
                    and gain[metric]["positive_seed_count"] >= 2
                    and gain[metric]["mean"] >= gain[metric]["mean_paired_permutation_se"]
                    for metric in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
                ):
                    alignment_supported_datasets.append(dataset)
            val = item["downstream_and_interventions"]["F0_pdc_on"]["metrics"]["val_acc"]["mean"]
            current_val = current["downstream_and_interventions"].get("formal", current["downstream_and_interventions"].get("F0_pdc_on"))["metrics"]["val_acc"]["mean"]
            if abs(val - current_val) > 0.01:
                validation_preserved = False

        no_pathology = not pathological_datasets
        if variant == c1_variant:
            recommendation = "Keep" if pathway_active_datasets and alignment_supported_datasets and no_pathology else "Reject"
        else:
            all_success = (
                bool(pathway_active_datasets)
                and bool(stronger_datasets)
                and bool(alignment_supported_datasets)
                and no_pathology
                and validation_preserved
            )
            partial = bool(pathway_active_datasets) and no_pathology and validation_preserved
            recommendation = "Strong candidate" if all_success else "Conditional" if partial else "Reject"
        recommendations[variant] = {
            "recommendation": recommendation,
            "pathway_genuinely_used_datasets": pathway_active_datasets,
            "stronger_than_c1_datasets_10pct_rule": stronger_datasets,
            "stronger_than_c1_2of3_seed_direction_datasets": stronger_seed_direction_datasets,
            "alignment_supported_datasets": alignment_supported_datasets,
            "pathological_datasets": pathological_datasets,
            "validation_preserved_within_0.01": validation_preserved,
            "rules": {
                "stronger_than_c1_relative_margin": 0.10,
                "alignment_requires_positive_seed_count": 2,
                "alignment_requires_mean_at_least_paired_se": True,
                "validation_band_absolute_accuracy": 0.01,
            },
        }
    return recommendations


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MoPF M1-E — PDC-v2 Structural Refinement",
        "",
        "This study retrains C0/C1/C2/C3 under one full-graph NC protocol. Sports is intentionally not run in M1-E.",
        "",
        f"- Runs: {len(summary['runs'])} (4 variants × 5 datasets × 3 seeds)",
        f"- Git commit: `{summary['metadata']['git_commit']}`",
        "",
        "## Candidate recommendations",
        "",
        "| Candidate | Recommendation |",
        "|---|---|",
    ]
    for variant in VARIANTS[1:]:
        lines.append(f"| {variant} | **{summary['candidate_recommendations'][variant]['recommendation']}** |")
    lines.extend(
        [
            "",
            "## NC validation and active-path evidence",
            "",
            "| Dataset | C0 Val Acc | C1 fused rel. L2 | C2 fused rel. L2 | C3 fused rel. L2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        current = summary["dataset_variant_aggregates"][f"{dataset}/C0_current"]["downstream_and_interventions"]["formal"]["metrics"]["val_acc"]["mean"]
        values = []
        for variant in VARIANTS[1:]:
            stream = summary["dataset_variant_aggregates"][f"{dataset}/{variant}"]["stream_changes_pdc_on_vs_off"]
            values.append(stream["fused_representation_change"]["mean_relative_l2"]["mean"] if stream else None)
        lines.append(
            f"| {dataset} | {current:.6f} | "
            + " | ".join("N/A" if value is None else f"{value:.6g}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "`active_path_alignment_gain` is computed per shared permutation seed as `shuffle_drop_on - shuffle_drop_off`; aggregates include population standard deviation, positive training-seed count, and paired 10-permutation standard error. The machine-readable `m1e_master_summary.json` is authoritative.",
            "For the PDC-v2 comparison gate, `stronger than C1` requires a 10% mean fused/logit relative-change margin and the same direction in at least 2/3 training seeds; the summary retains the per-seed values.",
            "",
            "## C3 order0 analysis",
            "",
            "| Dataset | mean rho0 text | mean rho0 visual | fused rel. L2 | logit rel. L2 | mean Val Acc delta | mean Test Acc delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        rows = summary["dataset_variant_aggregates"][f"{dataset}/C3_pdc_v2_full"]["c3_order0_analysis"]
        mean = lambda values: float(np.mean(values)) if values else 0.0
        lines.append(
            f"| {dataset} | {mean([row['rho0_text'] for row in rows]):.6g} | "
            f"{mean([row['rho0_visual'] for row in rows]):.6g} | "
            f"{mean([row['stream_changes']['fused_representation_change']['mean_relative_l2'] for row in rows]):.6g} | "
            f"{mean([row['stream_changes']['logit_probability_change']['all']['mean_relative_logit_l2'] for row in rows]):.6g} | "
            f"{mean([row['metrics_delta_pdc_on_minus_order0_off']['val_acc'] for row in rows]):.6g} | "
            f"{mean([row['metrics_delta_pdc_on_minus_order0_off']['test_acc'] for row in rows]):.6g} |"
        )
    lines.extend(
        [
            "",
            "The order0-off deltas are PDC-On minus order0-off; ele-fashion is included explicitly because nonzero learned rho0 does not by itself imply a large downstream effect.",
            "",
            "No final MoPF claim is made by this study.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "m1e_pdc_v2")
    parser.add_argument("--device", action="append", dest="devices")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    devices = tuple(args.devices or ["cuda:0", "cuda:1"])
    if not torch.cuda.is_available():
        devices = tuple("cpu" for _ in devices)
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.skip_training:
        training_records = []
        for dataset, variant, seed in itertools.product(DATASETS, VARIANTS, SEEDS):
            record_path = args.output_root / dataset / variant / f"seed{seed}" / "training_record.json"
            training_records.append(json.loads(record_path.read_text(encoding="utf-8")))
    else:
        training_records = train_all(args.output_root, devices, args.force)

    audit_records = audit_all(training_records, args.output_root, devices, args.force)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in audit_records:
        grouped[f"{record['dataset']}/{record['variant']}"].append(record)
    aggregates = {
        key: _aggregate_dataset_variant(records)
        for key, records in sorted(grouped.items())
    }
    summary = {
        "metadata": {
            "stage": "M1-E",
            "study": "PDC-v2 Structural Refinement",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "protocol": {
                "task": "NC",
                "training_mode": "full_graph",
                "optimizer": "AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "epochs": 300,
                "patience": 30,
                "checkpoint_selection": "best Validation Accuracy",
                "fixed_macro_f1": True,
                "edge_weight_mode": "separate_cos",
                "node_shuffle_seeds": list(SHUFFLE_SEEDS),
                "sports_run": False,
            },
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "K_by_dataset": K_BY_DATASET,
            "variants": list(VARIANTS),
        },
        "runs": audit_records,
        "training_records": training_records,
        "dataset_variant_aggregates": aggregates,
        "candidate_recommendations": _candidate_recommendations(aggregates),
        "restrictions": {
            "hrc": False,
            "ppc": False,
            "auxiliary_loss": False,
            "prior_anchoring": False,
            "router": False,
            "moe": False,
            "prototype": False,
            "semantic_graph_modified": False,
            "fusion_modified": False,
            "eta_formula_modified": False,
            "sports_run": False,
        },
        "artifacts": {
            "report": str(ROOT / "docs" / "mopf_m1e_pdc_v2.md"),
            "summary": str(args.output_root / "m1e_master_summary.json"),
        },
    }
    _dump_json(args.output_root / "m1e_master_summary.json", summary)
    _write_report(ROOT / "docs" / "mopf_m1e_pdc_v2.md", summary)
    print(f"[m1e] summary={args.output_root / 'm1e_master_summary.json'}", flush=True)
    print(json.dumps(summary["candidate_recommendations"], indent=2), flush=True)


if __name__ == "__main__":
    main()
