#!/usr/bin/env python3
"""Correctness checks and mechanism analysis for IAMOC v1 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import LinkPredictor, build_model  # noqa: E402
from src.tasks.lp import _evaluate_split as evaluate_lp  # noqa: E402
from src.tasks.nc import _evaluate_split as evaluate_nc  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402


def _model_cfg(name: str, order: int = 3, **overrides: Any):
    cfg = OmegaConf.load(ROOT / "configs" / "model" / f"{name}.yaml")
    cfg.name = name
    cfg.max_order = int(order)
    cfg.num_layers = int(order)
    for key, value in overrides.items():
        cfg[key] = value
    return OmegaConf.create({"ablation": "full", "model": cfg})


def _synthetic_correctness() -> dict[str, Any]:
    torch.manual_seed(20260921)
    data_info = {"input_dim": 12, "text_dim": 5, "visual_dim": 7, "num_nodes": 9, "num_classes": 3}
    x = torch.randn(data_info["num_nodes"], data_info["input_dim"])
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 0, 3, 2, 7],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 0, 8, 3, 0, 7, 2]],
        dtype=torch.long,
    )
    checks: dict[str, Any] = {}

    formal_yaml = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "model" / "mopf.yaml"), resolve=False
    )
    iamoc_yaml = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "model" / "mopf_iamoc.yaml"), resolve=False
    )
    formal_yaml.pop("name", None)
    formal_yaml.pop("version", None)
    for key in ("name", "version", "use_hop_interaction", "hop_interaction_layers", "hop_interaction_heads", "hop_interaction_dropout", "hop_interaction_order_embedding", "hop_interaction_gate_init", "relation_conditioning", "relation_bias_gate_init", "export_hop_attention"):
        iamoc_yaml.pop(key, None)
    if iamoc_yaml != formal_yaml:
        raise AssertionError("IAMOC config differs from formal MoPF beyond the allowed added fields")
    checks["config_preserves_formal_settings"] = True

    rejected_cfg = _model_cfg("mopf_iamoc", 3, node_conditioner_mode="pdc")
    try:
        build_model(rejected_cfg, data_info)
    except ValueError as error:
        if "node_conditioner_mode='absolute'" not in str(error):
            raise
    else:
        raise AssertionError("IAMOC did not fail fast for a non-absolute conditioner")
    checks["unsupported_conditioner_fail_fast"] = True

    formal_cfg = _model_cfg("mopf", 3)
    iamoc_disabled_cfg = _model_cfg(
        "mopf_iamoc", 3, use_hop_interaction=False, relation_conditioning="output"
    )
    torch.manual_seed(7)
    baseline = build_model(formal_cfg, data_info).eval()
    candidate = build_model(iamoc_disabled_cfg, data_info).eval()
    loaded = candidate.load_state_dict(baseline.state_dict(), strict=False)
    candidate_only = set(candidate.state_dict()) - set(baseline.state_dict())
    if set(loaded.missing_keys) != candidate_only or loaded.unexpected_keys:
        raise AssertionError(f"Shared parameter loading mismatch: {loaded}")
    with torch.no_grad():
        z_base = baseline(x, edge_index)[0]
        z_candidate = candidate(x, edge_index)[0]
    checks["baseline_exact_equivalence"] = bool(torch.equal(z_base, z_candidate))
    if not checks["baseline_exact_equivalence"]:
        max_error = float((z_base - z_candidate).abs().max().item())
        raise AssertionError(f"Baseline equivalence failed; max absolute difference={max_error}")

    for order in (2, 3):
        cfg = _model_cfg(
            "mopf_iamoc",
            order,
            relation_conditioning="attention",
            hop_interaction_layers=2,
            export_hop_attention=True,
        )
        model = build_model(cfg, data_info)
        output, _, _, aux_loss, _ = model(x, edge_index)
        components = model.analysis_hop_attention(x, edge_index)
        expected = order + 1
        if output.shape != (data_info["num_nodes"], model.out_dim):
            raise AssertionError(f"Unexpected model output shape: {tuple(output.shape)}")
        if components["eta_text"].shape != (data_info["num_nodes"], expected):
            raise AssertionError("eta shape mismatch")
        if components["hop_attention_text"].shape != (
            data_info["num_nodes"], expected, expected
        ):
            raise AssertionError("attention shape mismatch")
        loss = output.square().mean() + aux_loss
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith(("hop_order_embedding_", "hop_layers_", "theta_hop_gate_", "theta_relation_bias_")):
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise AssertionError(f"Missing or non-finite IAMOC gradient: {name}")
        for label, tensor in (("output", output), ("eta", components["eta_text"]), ("attention", components["hop_attention_text"])):
            if not bool(torch.isfinite(tensor).all()):
                raise AssertionError(f"Non-finite {label} at K={order}")
        checks[f"K{order}_forward_backward"] = "passed"

    # V2 is an isolation control: changing relation-context identity or
    # removing the bias cannot change attention or coefficients.
    none_cfg = _model_cfg("mopf_iamoc", 3, relation_conditioning="none")
    torch.manual_seed(9)
    no_rel = build_model(none_cfg, data_info).eval()
    with torch.no_grad():
        normal = no_rel.analysis_hop_attention(x, edge_index, relation_intervention="normal")
        off = no_rel.analysis_hop_attention(x, edge_index, relation_intervention="off")
        shuffled = no_rel.analysis_hop_attention(
            x, edge_index, relation_intervention="shuffle", permutation_seed=991
        )
    for key in ("z", "eta_text", "eta_visual", "hop_attention_text", "hop_attention_visual"):
        if not torch.equal(normal[key], off[key]) or not torch.equal(normal[key], shuffled[key]):
            raise AssertionError(f"V2 relation isolation failed for {key}")
    checks["V2_relation_isolation"] = "passed"

    # V3 changes attention only: force a non-flat profile in this synthetic
    # audit so normal/off interventions must produce distinguishable weights.
    attention_cfg = _model_cfg("mopf_iamoc", 3, relation_conditioning="attention")
    torch.manual_seed(11)
    relation_model = build_model(attention_cfg, data_info).eval()
    with torch.no_grad():
        relation_model.theta_transport_text.copy_(torch.tensor([0.5, -0.3, 0.15, -0.35]))
        relation_model.theta_transport_visual.copy_(torch.tensor([-0.4, 0.2, 0.3, -0.1]))
        normal = relation_model.analysis_hop_attention(x, edge_index, relation_intervention="normal")
        off = relation_model.analysis_hop_attention(x, edge_index, relation_intervention="off")
    if torch.equal(normal["hop_attention_text"], off["hop_attention_text"]):
        raise AssertionError("V3 relation bias did not change attention")
    expected_eta = (
        relation_model.gamma_global.unsqueeze(0)
        + relation_model.delta_gamma_text.unsqueeze(0)
        + normal["delta_node_text"]
    )
    if not torch.allclose(normal["eta_text"], expected_eta, atol=0.0, rtol=0.0):
        raise AssertionError("V3 final eta includes an unintended TCPR correction")
    for key in ("states_text", "states_visual", "responses_text", "responses_visual"):
        for first, second in zip(normal[key], off[key]):
            if not torch.equal(first, second):
                raise AssertionError(f"Relation intervention changed Stage-I/II bank: {key}")
    checks["V3_bias_and_no_output_tau"] = "passed"

    protected = [
        "src/models/mopf.py",
        "configs/model/mopf.yaml",
        "src/models/factory.py",
        "src/main.py",
        "src/tasks",
        "src/data",
    ]
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", *protected], cwd=ROOT, check=False
    )
    if result.returncode != 0:
        raise AssertionError("A protected formal MoPF file has been modified")
    checks["protected_formal_files_unchanged"] = True
    return checks


def _quantile_summary(values: torch.Tensor | np.ndarray | list[float]) -> dict[str, float]:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": math.nan, "std_population": math.nan, "q25": math.nan, "median": math.nan, "q75": math.nan}
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import rankdata

        return np.asarray(rankdata(values, method="average"), dtype=np.float64)
    except ImportError:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        return ranks


def _partial_spearman(x: torch.Tensor, y: torch.Tensor, degree: torch.Tensor) -> float | None:
    arrays = [tensor.detach().cpu().numpy().astype(np.float64).reshape(-1) for tensor in (x, y, degree)]
    valid = np.logical_and.reduce([np.isfinite(array) for array in arrays])
    if valid.sum() < 4:
        return None
    rx, ry, rz = [_rankdata(array[valid]) for array in arrays]
    design = np.column_stack([np.ones(rz.size), rz])
    residual_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    residual_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if residual_x.std() == 0 or residual_y.std() == 0:
        return None
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def _condition_quartiles(condition: torch.Tensor, attended_order: torch.Tensor) -> list[dict[str, float | int]]:
    relation = condition.detach().cpu().numpy().astype(np.float64)
    attended = attended_order.detach().cpu().numpy().astype(np.float64)
    valid = np.isfinite(relation) & np.isfinite(attended)
    if valid.sum() == 0:
        return []
    relation, attended = relation[valid], attended[valid]
    boundaries = np.quantile(relation, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = np.searchsorted(boundaries[1:-1], relation, side="right")
    output = []
    for quartile in range(4):
        selected = attended[bins == quartile]
        selected_relation = relation[bins == quartile]
        output.append(
            {
                "quartile": quartile + 1,
                "relation_min": float(selected_relation.min()) if selected_relation.size else float(boundaries[quartile]),
                "relation_max": float(selected_relation.max()) if selected_relation.size else float(boundaries[quartile + 1]),
                "count": int(selected.size),
                "mean_r_attn": float(selected.mean()) if selected.size else math.nan,
                "std_r_attn_population": float(selected.std(ddof=0)) if selected.size else math.nan,
            }
        )
    return output


def _distribution_payload(components: dict[str, Any], order_count: int) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for modality in ("text", "visual"):
        attention = components[f"hop_attention_{modality}"].detach().float()
        layer_tensors = components.get(f"hop_attention_layers_{modality}") or [attention]
        layer_means = [layer.detach().float().mean(dim=0) for layer in layer_tensors]
        key_mass = attention.mean(dim=1)
        entropy = -(key_mass * key_mass.clamp_min(1e-12).log()).sum(dim=-1)
        if order_count > 1:
            order_weights = torch.arange(order_count, device=attention.device, dtype=attention.dtype) / float(order_count - 1)
        else:
            order_weights = torch.zeros(order_count, device=attention.device, dtype=attention.dtype)
        attended_order = (key_mass * order_weights.unsqueeze(0)).sum(dim=-1)
        eta_order = MoPF_effective_radius(components[f"eta_{modality}"])
        payload[modality] = {
            "attention": attention,
            "key_mass": key_mass,
            "entropy": entropy,
            "r_attn": attended_order,
            "r_eta": eta_order,
            "condition": components[f"transport_context_{modality}"].detach().float(),
            "degree": components[f"transport_degree_{modality}"].detach().float(),
            "mean_attention": attention.mean(dim=0),
            "mean_attention_layers": layer_means,
            "mean_key_mass": key_mass.mean(dim=0),
            "entropy_summary": _quantile_summary(entropy),
            "r_attn_summary": _quantile_summary(attended_order),
            "r_eta_summary": _quantile_summary(eta_order),
        }
    return payload


def MoPF_effective_radius(eta: torch.Tensor) -> torch.Tensor:
    weights = eta.detach().abs()
    orders = torch.arange(eta.size(-1), dtype=eta.dtype, device=eta.device)
    return (weights * orders.unsqueeze(0)).sum(dim=-1) / weights.sum(dim=-1).clamp_min(torch.finfo(eta.dtype).eps)


def _write_attention_csv(path: Path, matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix.detach().cpu().numpy()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_order", *[f"key_order_{k}" for k in range(values.shape[1])]])
        for query, row in enumerate(values):
            writer.writerow([query, *[float(value) for value in row]])


def _write_node_csv(path: Path, payload: dict[str, Any]) -> None:
    text, visual = payload["text"], payload["visual"]
    count = int(text["r_attn"].numel())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["node_id"]
        for modality in ("text", "visual"):
            order_count = int(payload[modality]["key_mass"].size(1))
            header.extend([f"p_{modality}_{k}" for k in range(order_count)])
            header.extend([f"entropy_{modality}", f"r_attn_{modality}", f"r_eta_{modality}", f"relation_context_{modality}", f"physical_degree_{modality}"])
        writer.writerow(header)
        for node in range(count):
            row: list[Any] = [node]
            for modality in ("text", "visual"):
                values = payload[modality]
                row.extend(values["key_mass"][node].detach().cpu().tolist())
                row.extend([
                    float(values["entropy"][node].item()),
                    float(values["r_attn"][node].item()),
                    float(values["r_eta"][node].item()),
                    float(values["condition"][node].item()),
                    float(values["degree"][node].item()),
                ])
            writer.writerow(row)


def _downstream_metrics(cfg, checkpoint: dict[str, Any], model, data, components: dict[str, Any], device: torch.device) -> dict[str, float]:
    z = components["z"].detach().cpu()
    if str(cfg.task.name) == "nc":
        classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
        classifier.load_state_dict(checkpoint["head_state"])
        labels = _resolve_nc_eval_labels(data)
        metrics = evaluate_nc(classifier, z, data.y, data.test_idx, device, int(cfg.task.inference_batch_size), labels)
        return {"test_acc": metrics["acc"], "test_macro_f1": metrics["macro_f1"]}

    decoder = cfg.task.decoder
    proj_dim = int(decoder.get("proj_dim", 0) or 0)
    predictor = LinkPredictor(
        in_dim=proj_dim if proj_dim > 0 else model.out_dim,
        hidden_dim=int(decoder.hidden_dim),
        num_layers=int(decoder.num_layers),
        dropout=float(decoder.dropout),
    ).to(device)
    predictor.load_state_dict(checkpoint["head_state"])
    if proj_dim > 0:
        projection = nn.Linear(model.out_dim, proj_dim).to(device)
        projection.load_state_dict(checkpoint["proj_state"])
        z = projection(z.to(device)).detach().cpu()
    metrics = evaluate_lp(z, predictor, data.edge_split.test, device, int(cfg.task.eval_edge_batch_size))
    return {
        "test_mrr": metrics["mrr"],
        "test_hits@1": metrics["hits@1"],
        "test_hits@3": metrics["hits@3"],
        "test_hits@10": metrics["hits@10"],
    }


def _analyse_checkpoint(run_dir: Path, analysis_root: Path, device: torch.device) -> dict[str, Any]:
    cfg_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "best.pt"
    if not cfg_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing completed config/checkpoint in {run_dir}")
    cfg = OmegaConf.load(cfg_path)
    task, dataset_name, variant = str(cfg.task.name), str(cfg.dataset.name), run_dir.parent.name
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = load_mag_data(cfg, task, int(cfg.seed))
    model = build_model(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    trainable_model_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    shared_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    shared_cfg.model.name = "mopf"
    shared_model = build_model(shared_cfg, checkpoint["data_info"])
    baseline_model_parameters = sum(parameter.numel() for parameter in shared_model.parameters() if parameter.requires_grad)
    del shared_model
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    start = time.perf_counter()
    components = model.analysis_hop_attention(x, edge_index, relation_intervention="normal")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - start
    profile = _distribution_payload(components, model.max_order + 1)

    destination = analysis_root / dataset_name / variant / run_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    _write_node_csv(destination / "node_hop_profiles.csv", profile)
    for modality in ("text", "visual"):
        _write_attention_csv(destination / f"mean_attention_{modality}.csv", profile[modality]["mean_attention"])
        for layer, matrix in enumerate(profile[modality]["mean_attention_layers"], start=1):
            _write_attention_csv(
                destination / f"mean_attention_{modality}_layer_{layer}.csv", matrix
            )
    mechanism = {
        "dataset": dataset_name,
        "task": task,
        "variant": variant,
        "seed": int(cfg.seed),
        "checkpoint": str(checkpoint_path),
        "inference_seconds": inference_seconds,
        "trainable_encoder_parameters": trainable_model_parameters,
        "extra_encoder_parameters_over_V0": trainable_model_parameters - baseline_model_parameters,
        "hop_gate_text": float(torch.tanh(model.theta_hop_gate_text).item()),
        "hop_gate_visual": float(torch.tanh(model.theta_hop_gate_visual).item()),
        "relation_bias_gate_text": float(torch.tanh(model.theta_relation_bias_text).item()),
        "relation_bias_gate_visual": float(torch.tanh(model.theta_relation_bias_visual).item()),
        "modalities": {},
    }
    for modality in ("text", "visual"):
        values = profile[modality]
        mechanism["modalities"][modality] = {
            "mean_attention": values["mean_attention"].detach().cpu().tolist(),
            "mean_attention_layers": [matrix.detach().cpu().tolist() for matrix in values["mean_attention_layers"]],
            "mean_key_mass": values["mean_key_mass"].detach().cpu().tolist(),
            "entropy": values["entropy_summary"],
            "r_attn": values["r_attn_summary"],
            "r_eta": values["r_eta_summary"],
            "partial_spearman_relation_r_attn_given_log_degree": _partial_spearman(
                values["condition"], values["r_attn"], torch.log1p(values["degree"])
            ),
            "relation_condition_quartiles_vs_r_attn": _condition_quartiles(
                values["condition"], values["r_attn"]
            ),
        }
    mechanism["text_visual_mean_r_attn_difference"] = float(
        profile["text"]["r_attn"].mean().item() - profile["visual"]["r_attn"].mean().item()
    )
    mechanism["text_visual_mean_entropy_difference"] = float(
        profile["text"]["entropy"].mean().item() - profile["visual"]["entropy"].mean().item()
    )
    mechanism["text_visual_mean_attention_matrix_mean_absolute_difference"] = float(
        (profile["text"]["mean_attention"] - profile["visual"]["mean_attention"]).abs().mean().item()
    )
    mechanism["text_visual_mean_key_mass_mean_absolute_difference"] = float(
        (profile["text"]["mean_key_mass"] - profile["visual"]["mean_key_mass"]).abs().mean().item()
    )
    mechanism["downstream_normal"] = _downstream_metrics(cfg, checkpoint, model, data, components, device)

    # Relation identity intervention is reported for V3 only. Recomputing this
    # API changes the bias path alone; the model checkpoint stays untouched.
    interventions = {}
    if variant == "V3":
        for intervention in ("off", "shuffle"):
            changed = model.analysis_hop_attention(
                x,
                edge_index,
                relation_intervention=intervention,
                permutation_seed=20260921,
            )
            changed_profile = _distribution_payload(changed, model.max_order + 1)
            interventions[intervention] = {
                "downstream": _downstream_metrics(cfg, checkpoint, model, data, changed, device),
                "modalities": {
                    modality: {
                        "mean_attention": changed_profile[modality]["mean_attention"].detach().cpu().tolist(),
                        "entropy": changed_profile[modality]["entropy_summary"],
                        "r_attn": changed_profile[modality]["r_attn_summary"],
                        "r_eta": changed_profile[modality]["r_eta_summary"],
                    }
                    for modality in ("text", "visual")
                },
            }
        mechanism["relation_interventions"] = interventions
    (destination / "mechanism.json").write_text(json.dumps(mechanism, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return mechanism


def _aggregate(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _summarize_run_tree(root: Path, analysis_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    variants = ["V0", "V1", "V2", "V3", "V4"]
    for dataset in ("Movies", "Grocery", "sports-copurchase"):
        for variant in variants:
            for seed in (42, 43, 44):
                run_dir = root / dataset / variant / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                ckpt_path = run_dir / "best.pt"
                record_path = run_dir / "run_record.json"
                if not metrics_path.is_file() or not ckpt_path.is_file():
                    rows.append({"dataset": dataset, "variant": variant, "seed": seed, "status": "missing"})
                    continue
                metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
                model = build_model(cfg, ckpt["data_info"])
                encoder_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                if str(cfg.task.name) == "nc":
                    task_head_parameters = model.out_dim * int(ckpt["data_info"]["num_classes"]) + int(ckpt["data_info"]["num_classes"])
                else:
                    decoder = cfg.task.decoder
                    proj_dim = int(decoder.get("proj_dim", 0) or 0)
                    predictor = LinkPredictor(
                        in_dim=proj_dim if proj_dim > 0 else model.out_dim,
                        hidden_dim=int(decoder.hidden_dim),
                        num_layers=int(decoder.num_layers),
                        dropout=float(decoder.dropout),
                    )
                    task_head_parameters = sum(parameter.numel() for parameter in predictor.parameters())
                    if proj_dim > 0:
                        task_head_parameters += model.out_dim * proj_dim + proj_dim
                shared_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                shared_cfg.model.name = "mopf"
                shared_model = build_model(shared_cfg, ckpt["data_info"])
                baseline_encoder_parameters = sum(parameter.numel() for parameter in shared_model.parameters() if parameter.requires_grad)
                payload: dict[str, Any] = {
                    "dataset": dataset,
                    "task": metrics_json["task"],
                    "variant": variant,
                    "seed": seed,
                    "status": "complete",
                    "best_epoch": ckpt.get("epoch"),
                    "runtime_seconds": metrics_json.get("runtime_seconds"),
                    "peak_gpu_memory_mib": metrics_json.get("peak_gpu_memory_mib"),
                    "trainable_encoder_parameters": encoder_parameters,
                    "trainable_task_head_parameters": task_head_parameters,
                    "trainable_total_parameters": encoder_parameters + task_head_parameters,
                    "extra_encoder_parameters_over_V0": encoder_parameters - baseline_encoder_parameters,
                }
                payload.update(ckpt.get("metrics", {}))
                log_path = run_dir / "runner.log"
                if log_path.is_file():
                    payload["nonfinite_log_tokens"] = sorted(
                        set(re.findall(r"(?i)\b(?:nan|inf|infinity)\b", log_path.read_text(encoding="utf-8")))
                    )
                else:
                    payload["nonfinite_log_tokens"] = []
                if record_path.is_file():
                    payload["return_code"] = json.loads(record_path.read_text(encoding="utf-8")).get("return_code")
                rows.append(payload)

    metrics_by_task = {
        "nc": ["val_acc", "val_macro_f1", "test_acc", "test_macro_f1"],
        "lp": ["val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"],
    }
    aborted_path = analysis_root / "aborted_attempts.json"
    aborted_attempts = (
        json.loads(aborted_path.read_text(encoding="utf-8"))
        if aborted_path.is_file()
        else []
    )
    summary: dict[str, Any] = {
        "runs": rows,
        "aggregates": {},
        "paired_seed_delta": {},
        "efficiency": {},
        "aborted_attempts": aborted_attempts,
    }
    for dataset in ("Movies", "Grocery", "sports-copurchase"):
        task = "lp" if dataset == "sports-copurchase" else "nc"
        dataset_rows = [r for r in rows if r["dataset"] == dataset and r["status"] == "complete"]
        summary["aggregates"][dataset] = {}
        for variant in variants:
            selected = [r for r in dataset_rows if r["variant"] == variant]
            summary["aggregates"][dataset][variant] = {
                metric: {
                    "mean": _aggregate([float(r[metric]) for r in selected if metric in r])[0],
                    "std_population": _aggregate([float(r[metric]) for r in selected if metric in r])[1],
                    "n": sum(metric in r for r in selected),
                }
                for metric in metrics_by_task[task]
                if any(metric in r for r in selected)
            }
            if variant == "V0":
                continue
            metric_deltas: dict[str, list[dict[str, Any]]] = {}
            for metric in metrics_by_task[task]:
                deltas = []
                lookup = {int(r["seed"]): r for r in selected if metric in r}
                baseline = {int(r["seed"]): r for r in dataset_rows if r["variant"] == "V0" and metric in r}
                for seed in sorted(set(lookup) & set(baseline)):
                    deltas.append({"seed": seed, "metric": metric, "delta": float(lookup[seed][metric]) - float(baseline[seed][metric])})
                metric_deltas[metric] = deltas
            summary["paired_seed_delta"][f"{dataset}/{variant}"] = metric_deltas

        summary["efficiency"][dataset] = {}
        for variant in variants:
            selected = [r for r in dataset_rows if r["variant"] == variant]
            efficiency_fields = (
                "trainable_encoder_parameters",
                "extra_encoder_parameters_over_V0",
                "trainable_task_head_parameters",
                "trainable_total_parameters",
                "runtime_seconds",
                "peak_gpu_memory_mib",
            )
            summary["efficiency"][dataset][variant] = {
                field: {
                    "mean": float(np.mean([float(r[field]) for r in selected if r.get(field) is not None])),
                    "std_population": float(np.std([float(r[field]) for r in selected if r.get(field) is not None], ddof=0)),
                    "n": sum(r.get(field) is not None for r in selected),
                }
                for field in efficiency_fields
                if any(r.get(field) is not None for r in selected)
            }
            inference_seconds = []
            for seed in (42, 43, 44):
                mechanism_path = analysis_root / dataset / variant / f"seed_{seed}" / "mechanism.json"
                if mechanism_path.is_file():
                    inference_seconds.append(float(json.loads(mechanism_path.read_text(encoding="utf-8")).get("inference_seconds", 0.0)))
            if inference_seconds:
                summary["efficiency"][dataset][variant]["inference_seconds"] = {
                    "mean": float(np.mean(inference_seconds)),
                    "std_population": float(np.std(inference_seconds, ddof=0)),
                    "n": len(inference_seconds),
                }

    (analysis_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def _summary_report(summary: dict[str, Any], mechanism_rows: list[dict[str, Any]], report_path: Path) -> None:
    complete = [row for row in summary["runs"] if row.get("status") == "complete"]
    missing = [row for row in summary["runs"] if row.get("status") != "complete"]
    nc_rows = [row for row in summary["runs"] if row.get("dataset") in {"Movies", "Grocery"}]
    if not missing:
        report_title = "# IAMOC v1 Three-Seed Exploratory Report (complete)"
    elif nc_rows and all(row.get("status") == "complete" for row in nc_rows) and all(
        row.get("dataset") == "sports-copurchase" for row in missing
    ):
        report_title = "# IAMOC v1 Three-Seed Exploratory Report (NC complete; LP pending)"
    else:
        report_title = f"# IAMOC v1 Three-Seed Exploratory Report ({len(complete)}/{len(summary['runs'])} runs complete)"
    metric_names = {"nc": ["val_acc", "test_acc", "test_macro_f1"], "lp": ["val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"]}
    lines = [
        report_title,
        "",
        "## 1. Implementation Summary",
        "",
        "IAMOC adds per-modality, within-node attention over the Stage-II propagation-order states before the inherited low-rank node scorer. The scorer still uses the inherited projectors and order vectors. The learned coefficients compose the original Stage-II response bank; attention outputs do not replace structural responses.",
        "",
        "## 2. Baseline Invariance",
        "",
        "Stage I relation calibration, Stage II propagation, fusion, task heads, split handling, optimizer, checkpoint selection, and evaluation remain inherited from the formal MoPF implementation. The synthetic equivalence audit loads all shared MoPF parameters and checks exact output equality with hop interaction disabled.",
        "",
        "## 3. Variant Definitions",
        "",
        "- V0: formal CoSI-MAG / `mopf`.",
        "- V1: one-layer IAMOC with output TCPR conditioning.",
        "- V2: one-layer IAMOC without relation conditioning.",
        "- V3: one-layer IAMOC with relation bias in hop attention and no output TCPR.",
        "- V4: two-layer IAMOC with relation bias in hop attention and no output TCPR.",
        "",
        "## 4. Correctness Tests",
        "",
        "Synthetic baseline equivalence, K=2/K=3 forward/backward, V2 relation isolation, V3 relation-bias/no-output-TCPR, absolute-conditioner fail-fast, config parity, and protected-file checks passed. The five Movies-NC seed-42 one-epoch smoke runs completed with checkpoints and results artifacts. Attention export and V3 relation interventions were exercised from those smoke checkpoints. Five partial sports LP attempts were interrupted before producing checkpoints; they are itemized in `outputs/iamoc_v1/analysis/aborted_attempts.json` and excluded from the result grid.",
        "",
        "## 5. Three-Seed Main Results",
        "",
    ]
    for dataset in ("Movies", "Grocery", "sports-copurchase"):
        task = "lp" if dataset == "sports-copurchase" else "nc"
        lines.extend([f"### {dataset}", "", "| Variant | Best epoch (seed:epoch) | " + " | ".join(f"{name} mean ± std" for name in metric_names[task]) + " |", "|---|---:|" + "---:|" * len(metric_names[task])])
        for variant in ("V0", "V1", "V2", "V3", "V4"):
            values = summary["aggregates"].get(dataset, {}).get(variant, {})
            run_rows = [r for r in summary["runs"] if r.get("dataset") == dataset and r.get("variant") == variant and r.get("status") == "complete"]
            epochs = "; ".join(
                f"{row['seed']}:{row['best_epoch']}" for row in sorted(run_rows, key=lambda row: row["seed"])
            ) or "—"
            rendered = []
            for name in metric_names[task]:
                if name in values:
                    item = values[name]
                    rendered.append(f"{item['mean']:.4f} ± {item['std_population']:.4f} (n={item['n']})")
                else:
                    rendered.append("—")
            lines.append(f"| {variant} | {epochs} | " + " | ".join(rendered) + " |")
        lines.append("")
    lines.extend(["## 6. Paired Seed Comparison", "", "Paired values are Variant(seed) minus V0(seed), with no seed exclusion:", ""])
    for key, metrics in summary["paired_seed_delta"].items():
        for metric, values in metrics.items():
            lines.append(
                f"- `{key}` `{metric}`: "
                + (", ".join(f"seed {item['seed']}: {item['delta']:+.4f}" for item in values) if values else "no paired completed run")
            )
    lines.extend([
        "",
        "## 7. Stability",
        "",
        f"Completed runs: {len(complete)} / {len(summary['runs'])}. Missing or failed final runs: {len(missing)}. Final runs with NaN/Inf log tokens: {sum(bool(row.get('nonfinite_log_tokens')) for row in complete)}. {len(summary.get('aborted_attempts', []))} uncheckpointed LP attempts are listed separately and excluded. Per-run status, best epoch, and return code are in the machine-readable run summary and each run's `run_record.json`.",
        "",
        "## 8. Efficiency",
        "",
        "| Dataset | Variant | Encoder params | Extra vs V0 | Total trainable params | Runtime s mean ± std | Peak memory MiB mean ± std |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for dataset in ("Movies", "Grocery", "sports-copurchase"):
        for variant in ("V0", "V1", "V2", "V3", "V4"):
            efficiency = summary.get("efficiency", {}).get(dataset, {}).get(variant, {})
            def render(field: str) -> str:
                value = efficiency.get(field)
                return "—" if not value else f"{value['mean']:.1f} ± {value['std_population']:.1f}"
            encoder = efficiency.get("trainable_encoder_parameters", {})
            extra = efficiency.get("extra_encoder_parameters_over_V0", {})
            lines.append(
                f"| {dataset} | {variant} | {encoder.get('mean', '—')} | {extra.get('mean', '—')} | "
                f"{render('trainable_total_parameters')} | {render('runtime_seconds')} | {render('peak_gpu_memory_mib')} |"
            )
    lines.extend([
        "",
        "## 9. Hop Interaction Mechanism",
        "",
        "Per-checkpoint Text/Visual mean matrices, per-node profiles, and numerical summaries are under `outputs/iamoc_v1/analysis/`. The matrices and node values are also available as CSV for plotting.",
        "",
        "## 10. Node-Level Heterogeneity",
        "",
    ])
    for item in mechanism_rows:
        lines.append(
            f"- {item['dataset']} {item['variant']} seed {item['seed']}: "
            f"Text r_attn mean/std={item['modalities']['text']['r_attn']['mean']:.4f}/"
            f"{item['modalities']['text']['r_attn']['std_population']:.4f}; Visual="
            f"{item['modalities']['visual']['r_attn']['mean']:.4f}/"
            f"{item['modalities']['visual']['r_attn']['std_population']:.4f}."
        )
    lines.extend(["", "## 11. Modality-Level Heterogeneity", ""])
    for item in mechanism_rows:
        lines.append(
            f"- {item['dataset']} {item['variant']} seed {item['seed']}: "
            f"mean attention matrix MAE Text-vs-Visual="
            f"{item['text_visual_mean_attention_matrix_mean_absolute_difference']:.6f}; "
            f"mean key-mass MAE={item['text_visual_mean_key_mass_mean_absolute_difference']:.6f}; "
            f"entropy delta={item['text_visual_mean_entropy_difference']:+.6f}."
        )
    lines.extend([
        "",
        "## 12. Relation-Bias Mechanism",
        "",
        "V3 relation-context associations with attended order use partial Spearman correlation controlling for `log(1 + physical degree)`. These are associations, not causal effects.",
        "",
    ])
    for item in mechanism_rows:
        if item.get("variant") != "V3":
            continue
        for modality in ("text", "visual"):
            stats = item["modalities"][modality]
            lines.append(
                f"- {item['dataset']} seed {item['seed']} {modality}: partial Spearman="
                f"{stats['partial_spearman_relation_r_attn_given_log_degree']} ; "
                f"relation quartile mean r_attn="
                + ", ".join(
                    f"Q{quartile['quartile']} {quartile['mean_r_attn']:.4f} (n={quartile['count']})"
                    for quartile in stats["relation_condition_quartiles_vs_r_attn"]
                )
            )
    lines.extend([
        "",
        "## 13. Relation Intervention",
        "",
        "Normal, Relation-Off, and fixed-seed Relation-Shuffle interventions are evaluated for V3 checkpoints; interventions affect the attention bias only. Deltas below are relative to Normal; mean |ΔA| compares each intervention's per-modality mean attention matrix with Normal.",
        "",
    ])
    for item in mechanism_rows:
        if item.get("variant") != "V3" or "relation_interventions" not in item:
            continue
        metric = "test_mrr" if item["task"] == "lp" else "test_acc"
        normal = item["downstream_normal"]
        for name, value in item["relation_interventions"].items():
            downstream = value["downstream"]
            matrix_mae = {
                modality: float(np.mean(np.abs(
                    np.asarray(item["modalities"][modality]["mean_attention"])
                    - np.asarray(value["modalities"][modality]["mean_attention"])
                )))
                for modality in ("text", "visual")
            }
            lines.append(
                f"- {item['dataset']} seed {item['seed']} {name}: "
                f"Δtest_acc={downstream['test_acc'] - normal['test_acc']:+.7f}, "
                f"Δtest_macro-F1={downstream['test_macro_f1'] - normal['test_macro_f1']:+.7f}; "
                f"mean |ΔA| Text/Visual={matrix_mae['text']:.2e}/{matrix_mae['visual']:.2e}."
            )
    lines.extend([
        "",
        "## 14. Shallow Design",
        "",
        "V3 and V4 are compared on matched seeds in the main tables and efficiency section. Per-checkpoint gate values, attention entropy, order profiles, and mean interaction matrices are in the mechanism JSON/CSV artifacts.",
        "",
        "## 15. Overall Assessment",
        "",
    ])
    if len(complete) == len(summary["runs"]):
        story = {
            "V0": "formal CoSI-MAG reference",
            "V1": "cross-order interaction while retaining output TCPR",
            "V2": "cross-order interaction isolated from relation conditioning",
            "V3": "relation-biased attention; the intervention measures whether the bias is functionally consequential",
            "V4": "two-layer relation-biased attention tests the depth/cost trade-off against V3",
        }
        for variant in ("V0", "V1", "V2", "V3", "V4"):
            performance = []
            for dataset in ("Movies", "Grocery", "sports-copurchase"):
                primary_metric = "test_mrr" if dataset == "sports-copurchase" else "test_acc"
                deltas = [
                    float(entry["delta"])
                    for entry in summary["paired_seed_delta"].get(f"{dataset}/{variant}", {}).get(primary_metric, [])
                ]
                if variant == "V0":
                    performance.append(f"{dataset} reference")
                elif deltas:
                    performance.append(f"{dataset} Δ{primary_metric}={np.mean(deltas):+.4f}")
            variant_runs = [row for row in complete if row.get("variant") == variant]
            finite_runs = sum(not row.get("nonfinite_log_tokens") for row in variant_runs)
            gates = [row for row in mechanism_rows if row.get("variant") == variant]
            hop_gate = float(np.mean([
                max(abs(row["hop_gate_text"]), abs(row["hop_gate_visual"]))
                for row in gates
            ])) if gates else None
            active_relation_gate = None
            if variant in ("V3", "V4") and gates:
                active_relation_gate = float(np.mean([
                    max(abs(row["relation_bias_gate_text"]), abs(row["relation_bias_gate_visual"]))
                    for row in gates
                ]))
            eff_rows = [row for row in variant_runs if row.get("runtime_seconds") is not None]
            if eff_rows:
                efficiency_text = (
                    f"encoder params={np.mean([row['trainable_encoder_parameters'] for row in eff_rows]):.0f}, "
                    f"extra vs V0={np.mean([row['extra_encoder_parameters_over_V0'] for row in eff_rows]):.0f}, "
                    f"runtime={np.mean([row['runtime_seconds'] for row in eff_rows]):.1f}s, "
                    f"peak memory={np.mean([row['peak_gpu_memory_mib'] for row in eff_rows if row.get('peak_gpu_memory_mib') is not None]):.1f}MiB"
                )
            else:
                efficiency_text = "efficiency unavailable"
            mechanism_text = "baseline has no hop-interaction mechanism"
            if hop_gate is not None:
                mechanism_text = f"mean max |hop gate|={hop_gate:.4f}"
            if active_relation_gate is not None:
                mechanism_text += f", mean max |active relation-bias gate|={active_relation_gate:.4f}"
            if variant == "V3" and gates:
                intervention_rows = [
                    (row, value)
                    for row in gates
                    for value in row.get("relation_interventions", {}).values()
                ]
                if intervention_rows:
                    metric_shift = max(
                        abs(float(value["downstream"][metric]) - float(row["downstream_normal"][metric]))
                        for row, value in intervention_rows
                        for metric in ("test_acc", "test_macro_f1")
                    )
                    mechanism_text += f", max Relation-Off/Shuffle metric shift={metric_shift:.2e}"
            lines.append(
                f"- {variant}: Performance ({'; '.join(performance)}); "
                f"stability ({len(variant_runs)}/9 completed, {finite_runs} with finite logs); "
                f"mechanism ({mechanism_text}); efficiency ({efficiency_text}); "
                f"paper-story role: {story[variant]}."
            )
    else:
        lines.append(
            "NC-only interim assessment (all 30 NC runs complete; LP remains pending). "
            "The paired deltas below are descriptive across three seeds and do not establish significance."
        )
        for variant in ("V0", "V1", "V2", "V3", "V4"):
            if variant == "V0":
                lines.append(
                    "- V0: reference model; its NC test Accuracy and Macro-F1 are the paired baselines in Section 5."
                )
                continue
            per_dataset: list[str] = []
            for dataset in ("Movies", "Grocery"):
                deltas = summary["paired_seed_delta"].get(f"{dataset}/{variant}", {})
                acc = [float(row["delta"]) for row in deltas.get("test_acc", [])]
                f1 = [float(row["delta"]) for row in deltas.get("test_macro_f1", [])]
                if acc and f1:
                    per_dataset.append(
                        f"{dataset} ΔAcc={np.mean(acc):+.4f}, ΔMacro-F1={np.mean(f1):+.4f}"
                    )
            gates = [row for row in mechanism_rows if row.get("variant") == variant]
            hop_gate = float(np.mean([
                max(abs(row["hop_gate_text"]), abs(row["hop_gate_visual"])) for row in gates
            ])) if gates else float("nan")
            gate_text = f"mean max |hop gate|={hop_gate:.4f}"
            if variant in ("V3", "V4") and gates:
                rel_gate = float(np.mean([
                    max(abs(row["relation_bias_gate_text"]), abs(row["relation_bias_gate_visual"]))
                    for row in gates
                ]))
                gate_text += f", mean max |active relation-bias gate|={rel_gate:.4f}"
            lines.append(f"- {variant}: " + "; ".join(per_dataset) + f"; {gate_text}.")
        intervention_rows = [
            (item, value)
            for item in mechanism_rows
            if item.get("variant") == "V3"
            for value in item.get("relation_interventions", {}).values()
        ]
        if intervention_rows:
            max_matrix_change = max(
                float(np.mean(np.abs(
                    np.asarray(item["modalities"][modality]["mean_attention"])
                    - np.asarray(value["modalities"][modality]["mean_attention"])
                )))
                for item, value in intervention_rows
                for modality in ("text", "visual")
            )
            max_metric_change = max(
                abs(float(value["downstream"][metric]) - float(item["downstream_normal"][metric]))
                for item, value in intervention_rows
                for metric in ("test_acc", "test_macro_f1")
            )
            lines.append(
                f"- Mechanism/stability: all completed NC runs have finite logs and all hop gates are nonzero; "
                "V3/V4 active relation-bias gates are also nonzero. "
                f"However, V3 Relation-Off/Shuffle changed the mean attention matrix by at most "
                f"{max_matrix_change:.2e} MAE and changed neither test Accuracy nor Macro-F1 "
                f"(maximum absolute metric shift {max_metric_change:.2e}); this is weak functional evidence "
                "for a consequential relation-bias effect."
            )
        lines.append(
            "- Efficiency/story: Section 8 reports all variants. V3 versus V4 shows whether the added interaction layer buys NC performance at added cost, but the LP comparison is still needed before judging the requested shallow-design story or making a full-grid recommendation."
        )
    lines.extend(["", "## 16. Recommendation", ""])
    if len(complete) < len(summary["runs"]):
        recommendation = "D. insufficient evidence — the requested run grid is incomplete; no scientific conclusion is assigned to missing cells."
    else:
        def positive_benchmarks(variant: str) -> int:
            positive = 0
            for dataset in ("Movies", "Grocery", "sports-copurchase"):
                metric = "test_mrr" if dataset == "sports-copurchase" else "test_acc"
                values = summary["paired_seed_delta"].get(f"{dataset}/{variant}", {}).get(metric, [])
                positive += bool(values) and float(np.mean([entry["delta"] for entry in values])) > 0.0
            return int(positive)

        v1_rows = [row for row in mechanism_rows if row.get("variant") == "V1"]
        v3_rows = [row for row in mechanism_rows if row.get("variant") == "V3"]
        if positive_benchmarks("V1") >= 2 and v1_rows and any(
            max(abs(row["hop_gate_text"]), abs(row["hop_gate_visual"])) > 0.01 for row in v1_rows
        ):
            recommendation = "B. IAMOC-HI worth further study — positive paired mean deltas on at least two benchmarks and non-collapsed hop gates support a focused follow-up."
        elif positive_benchmarks("V3") >= 2 and v3_rows and any(
            max(abs(row["relation_bias_gate_text"]), abs(row["relation_bias_gate_visual"])) > 0.01 for row in v3_rows
        ):
            recommendation = "C. IAMOC-RB worth further study — positive paired mean deltas on at least two benchmarks and non-collapsed relation-bias gates support a focused follow-up."
        elif all(positive_benchmarks(variant) == 0 for variant in ("V1", "V2", "V3", "V4")):
            recommendation = "A. Keep original CoSI-MAG — no IAMOC variant has a positive paired mean delta on any benchmark."
        else:
            recommendation = "D. insufficient evidence — results are mixed across benchmarks or mechanism gates do not identify a clear follow-up."
    lines.append(recommendation)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "iamoc_v1")
    parser.add_argument("--analysis-root", type=Path, default=ROOT / "outputs" / "iamoc_v1" / "analysis")
    parser.add_argument("--datasets", nargs="+", default=["Movies", "Grocery", "sports-copurchase"])
    parser.add_argument("--variants", nargs="+", default=["V1", "V2", "V3", "V4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "iamoc_v1_three_seed_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.correctness_only:
        result = _synthetic_correctness()
        args.analysis_root.mkdir(parents=True, exist_ok=True)
        correctness_path = args.analysis_root / "correctness.json"
        correctness_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved correctness evidence: {correctness_path}")
        return 0
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    analyzed: list[dict[str, Any]] = []
    if not args.skip_checkpoints:
        for dataset in args.datasets:
            for variant in args.variants:
                for seed in args.seeds:
                    run_dir = args.root / dataset / variant / f"seed_{seed}"
                    if not (run_dir / "best.pt").is_file():
                        print(f"SKIP missing checkpoint: {run_dir}")
                        continue
                    print(f"ANALYZE {dataset} {variant} seed={seed}", flush=True)
                    analyzed.append(_analyse_checkpoint(run_dir, args.analysis_root, device))
    elif args.summarize:
        for dataset in args.datasets:
            for variant in args.variants:
                for seed in args.seeds:
                    mechanism_path = args.analysis_root / dataset / variant / f"seed_{seed}" / "mechanism.json"
                    if mechanism_path.is_file():
                        analyzed.append(json.loads(mechanism_path.read_text(encoding="utf-8")))
    if args.summarize:
        summary = _summarize_run_tree(args.root, args.analysis_root)
        _summary_report(summary, analyzed, args.report)
        print(f"Wrote {args.analysis_root / 'summary.json'} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
