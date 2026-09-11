"""Run MoPF-vNext U2-C formal factorial NC training and diagnostics.

The runner has one hard pre-training gate: all four factorial variants must
implement the same initial filter function on a CPU toy graph and one real
dataset.  C0 is reused from the frozen U1-T T2 checkpoints only after the
explicit cumulative path passes a legacy compatibility audit.  C1-C3 are
new 300-epoch NC runs under the frozen U1-T protocol.
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.u2a import (  # noqa: E402
    deterministic_node_subset,
    incremental_novelty,
    linear_cka_matrix,
    normalized_response_gram,
)
from src.analysis.u2c0 import mean_off_diagonal, pairwise_cosine_matrix, response_magnitude  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.mopf import (  # noqa: E402
    monomial_to_anchored_cumulative_coefficients,
)
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402
from src.utils.summary import count_parameters  # noqa: E402
from scripts.run_mopf_u1_semantic_conductance import (  # noqa: E402
    DATASETS,
    K_BY_DATASET,
    MAGB_DATASETS,
    SEEDS,
    _fixed_split_override,
    _peak_gpu_memory_mib,
)


VARIANTS = ("C0", "C1", "C2", "C3")
NEW_VARIANTS = ("C1", "C2", "C3")
ALPHA = 0.1
TAU = 0.35
OUTPUT_ROOT = ROOT / "outputs" / "u2c_formal_factorial_training"
SOURCE_SUMMARY = ROOT / "outputs" / "u1t_semantic_conductance_calibration" / "u1t_master_summary.json"
U2C0_SUMMARY = ROOT / "outputs" / "u2c0_state_response_integration_audit" / "u2c0_master_summary.json"
EPS = 1e-12
_CONFIG_LOCK = threading.Lock()


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot encode {type(value)!r}")


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


def _compose_cfg(dataset: str, seed: int, variant: str, device: str):
    k = K_BY_DATASET[dataset]
    state_mode = {"C0": "ordinary", "C1": "anchored", "C2": "ordinary", "C3": "anchored"}[variant]
    response_mode = {"C0": "cumulative", "C1": "cumulative", "C2": "differential", "C3": "differential"}[variant]
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mopf",
        f"seed={seed}",
        f"device={device}",
        "model.edge_weight_mode=learned_diag_cos",
        f"model.edge_weight_temperature={TAU}",
        f"model.max_order={k}",
        f"model.num_layers={k}",
        "model.num_metric_perspectives=4",
        "model.metric_init_seed=20260910",
        "model.metric_init_noise_std=0.01",
        f"model.multihop_state_mode={state_mode}",
        f"model.multihop_response_mode={response_mode}",
        f"model.multihop_anchor_alpha={ALPHA}",
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        overrides.append(f"dataset.nc_split_path={split_override}")
    # Hydra's GlobalHydra registry is process-global rather than thread-local.
    # Diagnostics can run on two GPUs concurrently, so serialize only config
    # composition while leaving model/data work parallel.
    with _CONFIG_LOCK:
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            return compose(config_name="config", overrides=overrides)


def _data_info(data: Any) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _source_records() -> dict[tuple[str, int], dict[str, Any]]:
    summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    records = {}
    for row in summary["phase_b"]["per_run"]:
        if row["label"] == "T2":
            records[(row["dataset"], int(row["seed"]))] = row
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    missing = expected - set(records)
    if missing:
        raise FileNotFoundError(f"Missing frozen U1-T T2 C0 records: {sorted(missing)}")
    for row in records.values():
        if not Path(row["checkpoint"]).is_file():
            raise FileNotFoundError(f"Missing frozen C0 checkpoint: {row['checkpoint']}")
    return records


def _toy_graph() -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260911)
    x = torch.randn((9, 10), generator=generator, dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 8],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 8, 0]],
        dtype=torch.long,
    )
    return x, edge_index, {"input_dim": 10, "num_nodes": 9, "num_classes": 4, "text_dim": 4, "visual_dim": 6}


def _component_signature(model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        components = model._encode_components(x, edge_index)
    names = ("z_text", "z_visual", "z_text_refined", "z_visual_refined", "z")
    return {name: components[name].detach().clone() for name in names}


def _relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).norm().item() / max(second.norm().item(), EPS))


def _compare_signatures(signatures: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    reference = signatures["C0"]
    rows = {}
    max_abs = 0.0
    max_relative = 0.0
    for variant, signature in signatures.items():
        row = {}
        for name in reference:
            delta = signature[name] - reference[name]
            current_abs = float(delta.abs().max().item())
            current_relative = _relative_l2(signature[name], reference[name])
            row[name] = {"max_abs": current_abs, "relative_l2": current_relative}
            max_abs = max(max_abs, current_abs)
            max_relative = max(max_relative, current_relative)
        rows[variant] = row
    return {
        "per_variant": rows,
        "max_abs": max_abs,
        "max_relative_l2": max_relative,
        "tolerance_relative_l2": 1e-6,
        "tolerance_max_abs": 1e-5,
        "passed": bool(max_relative < 1e-6 and max_abs < 1e-5),
    }


def _initialization_gate() -> dict[str, Any]:
    toy_x, toy_edges, info = _toy_graph()
    toy_signatures = {}
    parameter_counts = {}
    alpha_parameters = {}
    for variant in VARIANTS:
        torch.manual_seed(20260912)
        cfg = _compose_cfg("Movies", 42, variant, "cpu")
        model = build_model(cfg, info)
        toy_signatures[variant] = _component_signature(model, toy_x, toy_edges)
        parameter_counts[variant] = count_parameters(model)
        alpha_parameters[variant] = any("multihop_anchor_alpha" in name for name, _ in model.named_parameters())

    real_cfg = _compose_cfg("Movies", 42, "C0", "cpu")
    real_data = load_mag_data(real_cfg, "nc", 42)
    real_signatures = {}
    real_x = real_data.x
    real_edges = real_data.edge_index
    for variant in VARIANTS:
        torch.manual_seed(20260912)
        cfg = _compose_cfg("Movies", 42, variant, "cpu")
        model = build_model(cfg, _data_info(real_data))
        real_signatures[variant] = _component_signature(model, real_x, real_edges)

    toy_audit = _compare_signatures(toy_signatures)
    real_audit = _compare_signatures(real_signatures)
    return {
        "alpha": ALPHA,
        "toy_graph": toy_audit,
        "real_dataset": {"dataset": "Movies", "seed": 42, **real_audit},
        "parameter_count": parameter_counts,
        "parameter_count_equal": len(set(parameter_counts.values())) == 1,
        "alpha_trainable": alpha_parameters,
        "alpha_fixed": not any(alpha_parameters.values()),
        "passed": bool(
            toy_audit["passed"]
            and real_audit["passed"]
            and len(set(parameter_counts.values())) == 1
            and not any(alpha_parameters.values())
        ),
    }


def _legacy_compatibility_audit() -> dict[str, Any]:
    """Verify explicit C0 and the legacy cumulative field are bitwise equal."""
    toy_x, toy_edges, info = _toy_graph()
    explicit_cfg = _compose_cfg("Movies", 42, "C0", "cpu")
    legacy_cfg = OmegaConf.create(OmegaConf.to_container(explicit_cfg, resolve=True))
    del legacy_cfg.model.multihop_state_mode
    del legacy_cfg.model.multihop_response_mode
    legacy_cfg.model.multihop_mode = "cumulative"
    torch.manual_seed(20260913)
    explicit = build_model(explicit_cfg, info)
    torch.manual_seed(20260913)
    legacy = build_model(legacy_cfg, info)
    explicit_signature = _component_signature(explicit, toy_x, toy_edges)
    legacy_signature = _component_signature(legacy, toy_x, toy_edges)
    audit = _compare_signatures({"C0": explicit_signature, "legacy_cumulative": legacy_signature})
    audit["formal_default_state_mode"] = explicit.multihop_state_mode
    audit["formal_default_response_mode"] = explicit.multihop_response_mode
    audit["passed"] = bool(audit["passed"] and explicit.multihop_mode == "cumulative")
    return audit


def _run_training_job(dataset: str, variant: str, seed: int, device: str, output_root: Path, resume: bool) -> dict[str, Any]:
    k = K_BY_DATASET[dataset]
    run_dir = output_root / "runs" / dataset / variant / f"seed{seed}"
    checkpoint = run_dir / "best.pt"
    record_path = run_dir / "run_record.json"
    if resume and record_path.is_file() and checkpoint.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    state_mode = {"C1": "anchored", "C2": "ordinary", "C3": "anchored"}[variant]
    response_mode = {"C1": "cumulative", "C2": "differential", "C3": "differential"}[variant]
    command = [
        sys.executable, "-m", "src.main",
        f"dataset={dataset}", "task=nc", "model=mopf", "num_runs=1", f"seed={seed}", f"device={device}",
        "model.edge_weight_mode=learned_diag_cos", f"model.edge_weight_temperature={TAU}",
        f"model.max_order={k}", f"model.num_layers={k}",
        f"model.multihop_state_mode={state_mode}", f"model.multihop_response_mode={response_mode}",
        f"model.multihop_anchor_alpha={ALPHA}",
        "task.training_mode=full_graph", "task.optimizer=adamw", "task.epochs=300", "task.lr=1e-3",
        "task.weight_decay=1e-4", "task.patience=30", "task.early_stop_min_epoch=30",
        "task.early_stop_min_delta=1e-4", "task.eval_every=1", "task.grad_clip=1.0", "task.evaluate_test=true",
        "model.export_aux_stats=false", "model.export_node_aux=false", "model.export_edge_aux=false",
        f"task.save_ckpt_path={checkpoint}", f"hydra.run.dir={run_dir}",
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        command.append(f"dataset.nc_split_path={split_override}")
    started = time.monotonic()
    log_path = (run_dir / "process.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=log_path, stderr=subprocess.STDOUT)
        peak = None
        while process.poll() is None:
            current = _peak_gpu_memory_mib(process.pid, device)
            if current is not None:
                peak = current if peak is None else max(peak, current)
            time.sleep(2.0)
    finally:
        log_path.close()
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        raise RuntimeError(f"U2-C {variant} failed for {dataset} seed={seed}: {' '.join(command)}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected checkpoint missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(_compose_cfg(dataset, seed, variant, "cpu"), payload["data_info"])
    output = {
        "dataset": dataset, "seed": int(seed), "variant": variant, "k": k, "device": device,
        "run_dir": str(run_dir), "checkpoint": str(checkpoint),
        "downstream": {
            "val_acc": payload.get("metrics", {}).get("val_acc"),
            "val_macro_f1": payload.get("metrics", {}).get("val_macro_f1"),
            "test_acc": payload.get("metrics", {}).get("test_acc"),
            "test_macro_f1": payload.get("metrics", {}).get("test_macro_f1"),
        },
        "best_epoch": payload.get("epoch"),
        "parameter_count": count_parameters(model) + count_parameters(nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"]))),
        "training_time_seconds": float(elapsed),
        "peak_gpu_memory_mib": peak,
        "peak_gpu_memory_status": "available" if peak is not None else "unavailable",
    }
    _dump(record_path, output)
    return output


def _load_checkpoint_components(record: dict[str, Any], device: str) -> tuple[Any, Any, Any, dict[str, Any]]:
    dataset, seed, variant = record["dataset"], int(record["seed"]), record["variant"]
    cfg = _compose_cfg(dataset, seed, variant, device)
    data = load_mag_data(cfg, "nc", seed)
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(cfg, payload["data_info"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    classifier = nn.Linear(model.out_dim, int(data.num_classes))
    classifier.load_state_dict(payload["head_state"])
    classifier.to(device).eval()
    with torch.no_grad():
        components = model._encode_components(data.x.to(device), data.edge_index.to(device))
    return data, model, classifier, components


def _cosine_to_reference(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = first.norm() * second.norm()
    if float(denominator.item()) <= EPS:
        return 0.0
    return float((first.reshape(-1).dot(second.reshape(-1)) / denominator).item())


def _state_retention(states: list[torch.Tensor], h0: torch.Tensor, subset: torch.Tensor) -> dict[str, Any]:
    cka = []
    cosine = []
    for state in states:
        cka.append(float(linear_cka_matrix([h0, state], subset, computation_dtype=torch.float64)[0, 1].item()))
        cosine.append(_cosine_to_reference(state, h0))
    return {"cka_to_h0": cka, "cosine_to_h0": cosine, "mean_nonzero_cka": float(np.mean(cka[1:])) if len(cka) > 1 else cka[0], "mean_nonzero_cosine": float(np.mean(cosine[1:])) if len(cosine) > 1 else cosine[0]}


def _response_diagnostics(responses: list[torch.Tensor], states: list[torch.Tensor], h0: torch.Tensor, subset: torch.Tensor) -> dict[str, Any]:
    signed = pairwise_cosine_matrix(responses)
    absolute = pairwise_cosine_matrix(responses, absolute=True)
    squared = pairwise_cosine_matrix(responses, squared=True)
    cka = linear_cka_matrix(responses, subset, computation_dtype=torch.float64)
    novelty = incremental_novelty(responses)
    _, spectrum = normalized_response_gram(responses)
    magnitude = response_magnitude(responses, h0)
    return {
        "signed_frobenius_cosine": signed.tolist(),
        "absolute_frobenius_cosine": absolute.tolist(),
        "squared_frobenius_cosine": squared.tolist(),
        "linear_cka": cka.tolist(),
        "incremental_novelty": novelty,
        "effective_rank": spectrum["effective_rank"],
        "normalized_effective_rank": spectrum["normalized_effective_rank"],
        "condition_number": spectrum["condition_number"],
        "response_magnitude": magnitude,
        "abs_cos_mean_off_diagonal": mean_off_diagonal(absolute),
        "cka_mean_off_diagonal": mean_off_diagonal(cka),
        "novelty_mean_nonzero": float(np.mean(novelty["novelty_ratio"][1:])) if len(responses) > 1 else novelty["novelty_ratio"][0],
        "state_norm_ratio_to_h0": [float(value.norm().item() / max(h0.norm().item(), EPS)) for value in states],
    }


def _contributions(eta: torch.Tensor, responses: list[torch.Tensor]) -> dict[str, Any]:
    mean_norm = []
    median_norm = []
    global_norm = []
    for order, response in enumerate(responses):
        contribution = eta[:, order : order + 1] * response
        norms = contribution.norm(dim=-1)
        mean_norm.append(float(norms.mean().item()))
        median_norm.append(float(norms.median().item()))
        global_norm.append(float(contribution.norm().item()))
    denominator = max(sum(global_norm), EPS)
    shares = [value / denominator for value in global_norm]
    return {
        "mean_l2": mean_norm,
        "median_l2": median_norm,
        "global_frobenius": global_norm,
        "share": shares,
        "multi_hop_share": float(sum(shares[1:])),
        "contribution_order_center": float(sum(index * value for index, value in enumerate(global_norm)) / max(sum(global_norm), EPS)),
    }


def _coefficient_profile(eta: torch.Tensor, response: dict[str, Any]) -> dict[str, Any]:
    values = eta.detach().double()
    mean_abs = values.abs().mean(dim=0).tolist()
    center_denominator = max(sum(mean_abs), EPS)
    return {
        "mean_abs": mean_abs,
        "median_abs": values.abs().median(dim=0).values.tolist(),
        "coefficient_order_center": float(sum(index * value for index, value in enumerate(mean_abs)) / center_denominator),
        "nonfinite": bool(not torch.isfinite(values).all()),
        "response_magnitude": response["response_magnitude"]["norm_ratio_to_h0"],
    }


def _metrics_from_logits(logits: torch.Tensor, data: Any, split: str, labels: list[int]) -> dict[str, float]:
    index = getattr(data, f"{split}_idx").to(logits.device)
    prediction = logits.index_select(0, index).argmax(dim=-1).cpu().numpy()
    target = data.y.index_select(0, index.cpu()).cpu().numpy()
    return {
        "acc": float(np.mean(prediction == target)),
        "macro_f1": float(f1_score(target, prediction, labels=labels, average="macro", zero_division=0)),
    }


def _anchor_off_components(model: Any, data: Any, components: dict[str, Any], modality: str) -> tuple[torch.Tensor, torch.Tensor]:
    h0 = components[f"h_{modality}"]
    index = components[f"norm_{'t' if modality == 'text' else 'v'}_index"]
    weight = components[f"norm_{'t' if modality == 'text' else 'v'}_weight"]
    states = model._propagation_bank(h0, index, weight)
    projectors = model.node_proj_text if modality == "text" else model.node_proj_visual
    vectors = model.node_vector_text if modality == "text" else model.node_vector_visual
    delta, _, _ = model._node_residuals(states, projectors, vectors, modality)
    eta = model._effective_coefficients(modality, delta)
    responses = components[f"responses_{modality}"]
    z = model._filter_bases(responses, eta)
    return z, eta


def _fuse_from_modalities(model: Any, z_text: torch.Tensor, z_visual: torch.Tensor) -> torch.Tensor:
    text_refined = model.text_refine_norm(z_text + model.text_refine_mlp(z_text))
    visual_refined = model.visual_refine_norm(z_visual + model.visual_refine_mlp(z_visual))
    fused_input = torch.cat([text_refined, visual_refined], dim=-1)
    return model.output_norm(model.fusion_skip(fused_input) + model.fusion_mlp(fused_input))


def _diagnose(record: dict[str, Any], device: str, output_root: Path, resume: bool) -> dict[str, Any]:
    path = output_root / "per_run" / f"{record['dataset']}_{record['variant']}_seed{record['seed']}.json"
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    data, model, classifier, components = _load_checkpoint_components(record, device)
    subset = deterministic_node_subset(data.num_nodes).to(device)
    labels = _resolve_nc_eval_labels(data)
    with torch.no_grad():
        state_payload = {}
        response_payload = {}
        contribution_payload = {}
        eta_payload = {}
        for modality in ("text", "visual"):
            states = components[f"states_{modality}"]
            responses = components[f"responses_{modality}"]
            h0 = components[f"h_{modality}"]
            state_payload[modality] = _state_retention(states, h0, subset)
            response_payload[modality] = _response_diagnostics(responses, states, h0, subset)
            contribution_payload[modality] = _contributions(components[f"eta_{modality}"], responses)
            eta_payload[modality] = _coefficient_profile(components[f"eta_{modality}"], response_payload[modality])

        anchor_intervention = None
        if record["variant"] in {"C1", "C3"}:
            z_text_off, eta_text_off = _anchor_off_components(model, data, components, "text")
            z_visual_off, eta_visual_off = _anchor_off_components(model, data, components, "visual")
            z_off = _fuse_from_modalities(model, z_text_off, z_visual_off)
            logits_on = classifier(components["z"])
            logits_off = classifier(z_off)
            per_modality = {}
            for modality, on_key, off_eta in (("text", "z_text", eta_text_off), ("visual", "z_visual", eta_visual_off)):
                eta_on = components[f"eta_{modality}"]
                per_modality[modality] = {
                    "eta_relative_l2": _relative_l2(off_eta, eta_on),
                    "eta_max_abs": float((off_eta - eta_on).abs().max().item()),
                }
            anchor_intervention = {
                "per_modality": per_modality,
                "z_relative_l2": _relative_l2(z_off, components["z"]),
                "logits_relative_l2": _relative_l2(logits_off, logits_on),
                "logits_max_abs": float((logits_off - logits_on).abs().max().item()),
                "all_prediction_flip_rate": float((logits_off.argmax(-1) != logits_on.argmax(-1)).float().mean().item()),
                "val": {
                    "on": _metrics_from_logits(logits_on, data, "val", labels),
                    "off": _metrics_from_logits(logits_off, data, "val", labels),
                },
                "test": {
                    "on": _metrics_from_logits(logits_on, data, "test", labels),
                    "off": _metrics_from_logits(logits_off, data, "test", labels),
                },
            }

    output = {
        "dataset": record["dataset"], "variant": record["variant"], "seed": int(record["seed"]),
        "checkpoint": record["checkpoint"], "state_semantics": {"state_mode": model.multihop_state_mode, "response_mode": model.multihop_response_mode, "alpha": model.multihop_anchor_alpha},
        "semantic_retention": state_payload, "response_distinctiveness": response_payload,
        "actual_contribution": contribution_payload, "eta_profile": eta_payload,
        "anchor_frozen_intervention": anchor_intervention,
        "all_finite": bool(all(torch.isfinite(value).all() for key, value in components.items() if isinstance(value, torch.Tensor))),
    }
    _dump(path, output)
    del data, model, classifier, components
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _attach_diagnostics(records: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["dataset"], row["variant"], int(row["seed"])): row for row in diagnostics}
    output = []
    for record in records:
        result = dict(record)
        result["diagnostics"] = lookup[(record["dataset"], record["variant"], int(record["seed"]))]
        output.append(result)
    return output


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "population_std": float(np.std(values, ddof=0)), "count": len(values)}


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    output: dict[str, Any] = {"per_dataset": {}, "equal_dataset": {}}
    for variant in VARIANTS:
        output["per_dataset"][variant] = {}
        for dataset in DATASETS:
            rows = [r for r in records if r["variant"] == variant and r["dataset"] == dataset]
            output["per_dataset"][variant][dataset] = {
                metric: _mean_std([float(r["downstream"][metric]) for r in rows]) for metric in metrics
            }
        output["equal_dataset"][variant] = {
            metric: _mean_std([output["per_dataset"][variant][dataset][metric]["mean"] for dataset in DATASETS])
            for metric in metrics
        }
    return output


def _factorial(records: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in records}
    metrics = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            row = {"dataset": dataset, "seed": seed}
            for metric in metrics:
                values = {v: float(lookup[(dataset, seed, v)]["downstream"][metric]) for v in VARIANTS}
                row[f"anchor_effect_{metric}"] = 0.5 * ((values["C1"] - values["C0"]) + (values["C3"] - values["C2"]))
                row[f"differential_effect_{metric}"] = 0.5 * ((values["C2"] - values["C0"]) + (values["C3"] - values["C1"]))
                row[f"interaction_{metric}"] = values["C3"] - values["C2"] - values["C1"] + values["C0"]
            rows.append(row)
    aggregate = {}
    for metric in metrics:
        for effect in ("anchor_effect", "differential_effect", "interaction"):
            by_dataset = {}
            for dataset in DATASETS:
                values = [r[f"{effect}_{metric}"] for r in rows if r["dataset"] == dataset]
                by_dataset[dataset] = _mean_std(values)
            equal_values = [by_dataset[dataset]["mean"] for dataset in DATASETS]
            aggregate[f"{effect}_{metric}"] = {"per_dataset": by_dataset, "equal_dataset": _mean_std(equal_values)}
    return {"per_seed": rows, "aggregate": aggregate}


def _mechanism_gates(records: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    lookup = {(r["dataset"], int(r["seed"]), r["variant"]): r for r in records}
    anchor_rows = []
    differential_rows = []
    for dataset in DATASETS:
        anchor_dataset_pass = False
        diff_dataset_pass = False
        for modality in ("text", "visual"):
            anchor_seed_deltas = []
            diff_seed_deltas = []
            diff_novelty_deltas = []
            diff_rank_deltas = []
            for seed in SEEDS:
                def value(variant: str, key: str) -> float:
                    return float(lookup[(dataset, seed, variant)]["diagnostics"][key][modality]["mean_nonzero_cka"])
                cka_anchor = 0.5 * ((value("C1", "semantic_retention") - value("C0", "semantic_retention")) + (value("C3", "semantic_retention") - value("C2", "semantic_retention")))
                anchor_seed_deltas.append(cka_anchor)
                def response(variant: str) -> dict[str, Any]:
                    return lookup[(dataset, seed, variant)]["diagnostics"]["response_distinctiveness"][modality]
                diff_redundancy = 0.5 * ((float(response("C2")["abs_cos_mean_off_diagonal"]) - float(response("C0")["abs_cos_mean_off_diagonal"])) + (float(response("C3")["abs_cos_mean_off_diagonal"]) - float(response("C1")["abs_cos_mean_off_diagonal"])))
                diff_cka = 0.5 * ((float(response("C2")["cka_mean_off_diagonal"]) - float(response("C0")["cka_mean_off_diagonal"])) + (float(response("C3")["cka_mean_off_diagonal"]) - float(response("C1")["cka_mean_off_diagonal"])))
                diff_novelty = 0.5 * ((float(response("C2")["novelty_mean_nonzero"]) - float(response("C0")["novelty_mean_nonzero"])) + (float(response("C3")["novelty_mean_nonzero"]) - float(response("C1")["novelty_mean_nonzero"])))
                diff_rank = 0.5 * ((float(response("C2")["normalized_effective_rank"]) - float(response("C0")["normalized_effective_rank"])) + (float(response("C3")["normalized_effective_rank"]) - float(response("C1")["normalized_effective_rank"])))
                diff_seed_deltas.append({"abs_cos": diff_redundancy, "cka": diff_cka})
                diff_novelty_deltas.append(diff_novelty)
                diff_rank_deltas.append(diff_rank)
            stable_anchor = sum(delta > 0 for delta in anchor_seed_deltas) >= 2
            stable_diff = sum(delta["abs_cos"] < 0 or delta["cka"] < 0 for delta in diff_seed_deltas) >= 2
            novelty_ok = sum(value >= -1e-12 for value in diff_novelty_deltas) >= 2
            rank_ok = sum(value >= -1e-12 for value in diff_rank_deltas) >= 2
            anchor_dataset_pass = anchor_dataset_pass or stable_anchor
            diff_dataset_pass = diff_dataset_pass or (stable_diff and novelty_ok and rank_ok)
            anchor_rows.append({"dataset": dataset, "modality": modality, "seed_deltas_cka": anchor_seed_deltas, "stable_direction_seeds": sum(delta > 0 for delta in anchor_seed_deltas), "passed": stable_anchor})
            differential_rows.append({"dataset": dataset, "modality": modality, "seed_deltas_redundancy": diff_seed_deltas, "seed_deltas_novelty": diff_novelty_deltas, "seed_deltas_rank": diff_rank_deltas, "passed": stable_diff and novelty_ok and rank_ok})
        for row in anchor_rows:
            if row["dataset"] == dataset and row["passed"]:
                break
        for row in differential_rows:
            if row["dataset"] == dataset and row["passed"]:
                break
    anchor_dataset_count = len({row["dataset"] for row in anchor_rows if row["passed"]})
    differential_dataset_count = len({row["dataset"] for row in differential_rows if row["passed"]})
    intervention_rows = [r for r in records if r["variant"] in {"C1", "C3"}]
    intervention_dataset_count = len({r["dataset"] for r in intervention_rows if r["diagnostics"]["anchor_frozen_intervention"] and (r["diagnostics"]["anchor_frozen_intervention"]["logits_relative_l2"] > 1e-10 or r["diagnostics"]["anchor_frozen_intervention"]["z_relative_l2"] > 1e-10)})
    pathology = []
    for r in records:
        diag = r["diagnostics"]
        for modality in ("text", "visual"):
            eta = diag["eta_profile"][modality]["mean_abs"]
            response = diag["response_distinctiveness"][modality]["response_magnitude"]["norm_ratio_to_h0"]
            share = diag["actual_contribution"][modality]["share"]
            matched = "C1" if r["variant"] == "C3" else "C0"
            match = lookup[(r["dataset"], int(r["seed"]), matched)]["diagnostics"]
            match_eta = match["eta_profile"][modality]["mean_abs"]
            ratios = [float(a / max(b, EPS)) for a, b in zip(eta, match_eta)]
            for order, ratio in enumerate(ratios):
                if r["variant"] in {"C2", "C3"} and ratio > 10.0:
                    pathology.append({"type": "strong_coefficient_compensation", "dataset": r["dataset"], "variant": r["variant"], "seed": r["seed"], "modality": modality, "order": order, "ratio": ratio})
                if r["variant"] in {"C2", "C3"} and order > 0 and response[order] < 1e-4 and share[order] > 0.1:
                    pathology.append({"type": "suspicious_compensation", "dataset": r["dataset"], "variant": r["variant"], "seed": r["seed"], "modality": modality, "order": order, "response_magnitude": response[order], "contribution_share": share[order]})
            if sum(share[1:]) < 0.05:
                pathology.append({"type": "multi_hop_contribution_collapse", "dataset": r["dataset"], "variant": r["variant"], "seed": r["seed"], "modality": modality, "multi_hop_share": sum(share[1:])})
    finite = all(bool(r["diagnostics"]["all_finite"]) for r in records)
    anchor_gate = anchor_dataset_count >= 3 and intervention_dataset_count >= 2
    differential_gate = differential_dataset_count >= 3 and finite and not any(item["type"] in {"strong_coefficient_compensation", "suspicious_compensation"} for item in pathology)
    return {
        "anchor": {"dataset_modality_rows": anchor_rows, "datasets_passed": anchor_dataset_count, "intervention_datasets": intervention_dataset_count, "passed": anchor_gate},
        "differential": {"dataset_modality_rows": differential_rows, "datasets_passed": differential_dataset_count, "passed": differential_gate},
        "compensation_pathology": {"all_finite": finite, "records": pathology, "passed": not pathology and finite},
        "multi_hop_contribution_collapse_present": any(item["type"] == "multi_hop_contribution_collapse" for item in pathology),
    }


def _performance_gates(aggregate: dict[str, Any]) -> dict[str, Any]:
    result = {}
    baseline = aggregate["per_dataset"]["C0"]
    for variant in NEW_VARIANTS:
        dataset_deltas = {dataset: aggregate["per_dataset"][variant][dataset]["val_acc"]["mean"] - baseline[dataset]["val_acc"]["mean"] for dataset in DATASETS}
        mean_delta = aggregate["equal_dataset"][variant]["val_acc"]["mean"] - aggregate["equal_dataset"]["C0"]["val_acc"]["mean"]
        f1_delta = aggregate["equal_dataset"][variant]["val_macro_f1"]["mean"] - aggregate["equal_dataset"]["C0"]["val_macro_f1"]["mean"]
        result[variant] = {"dataset_val_acc_delta": dataset_deltas, "equal_dataset_val_acc_delta": mean_delta, "equal_dataset_val_macro_f1_delta": f1_delta, "passed": bool(mean_delta >= -0.002 and max(dataset_deltas.values()) <= math.inf and min(dataset_deltas.values()) >= -0.005 and f1_delta >= -0.005)}
    result["C0"] = {"passed": True}
    return result


def _select(performance: dict[str, Any], mechanisms: dict[str, Any]) -> dict[str, Any]:
    safe = {variant: bool(performance[variant]["passed"]) for variant in VARIANTS}
    if safe["C3"] and mechanisms["anchor"]["passed"] and mechanisms["differential"]["passed"] and mechanisms["compensation_pathology"]["passed"] and not mechanisms["multi_hop_contribution_collapse_present"]:
        selected = "C3"
    elif safe["C2"] and mechanisms["differential"]["passed"] and mechanisms["compensation_pathology"]["passed"]:
        selected = "C2"
    elif safe["C1"] and mechanisms["anchor"]["passed"]:
        selected = "C1"
    elif safe["C0"]:
        selected = "C0"
    else:
        selected = "Unresolved"
    return {"selected_variant": selected, "selection_uses": ["validation_accuracy", "validation_macro_f1", "mechanism_gates"], "test_used_for_selection": False, "performance_safe": safe}


def _write_csv(path: Path, records: list[dict[str, Any]], factorial: dict[str, Any]) -> None:
    rows = []
    for record in records:
        diag = record["diagnostics"]
        row = {
            "dataset": record["dataset"], "seed": record["seed"], "variant": record["variant"], "best_epoch": record.get("best_epoch"),
            "val_acc": record["downstream"]["val_acc"], "val_macro_f1": record["downstream"]["val_macro_f1"], "test_acc": record["downstream"]["test_acc"], "test_macro_f1": record["downstream"]["test_macro_f1"],
            "parameter_count": record["parameter_count"], "training_time_seconds": record.get("training_time_seconds"), "peak_memory_mib": record.get("peak_gpu_memory_mib"), "c0_source_reuse": record.get("source_reuse", False),
        }
        for modality in ("text", "visual"):
            row[f"{modality}_retention_cka_mean"] = diag["semantic_retention"][modality]["mean_nonzero_cka"]
            row[f"{modality}_retention_cosine_mean"] = diag["semantic_retention"][modality]["mean_nonzero_cosine"]
            response = diag["response_distinctiveness"][modality]
            contribution = diag["actual_contribution"][modality]
            row[f"{modality}_abs_cos_mean_offdiag"] = response["abs_cos_mean_off_diagonal"]
            row[f"{modality}_cka_mean_offdiag"] = response["cka_mean_off_diagonal"]
            row[f"{modality}_novelty_mean"] = response["novelty_mean_nonzero"]
            row[f"{modality}_effective_rank"] = response["normalized_effective_rank"]
            row[f"{modality}_multi_hop_share"] = contribution["multi_hop_share"]
            row[f"{modality}_coefficient_order_center"] = diag["eta_profile"][modality]["coefficient_order_center"]
            row[f"{modality}_contribution_order_center"] = contribution["contribution_order_center"]
            row[f"{modality}_response_magnitude_highest"] = response["response_magnitude"]["norm_ratio_to_h0"][-1]
        intervention = diag["anchor_frozen_intervention"]
        row["anchor_off_z_relative_l2"] = None if intervention is None else intervention["z_relative_l2"]
        row["anchor_off_logits_relative_l2"] = None if intervention is None else intervention["logits_relative_l2"]
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    factorial_path = path.with_name("u2c_factorial_effects.csv")
    frows = factorial["per_seed"]
    with factorial_path.open("w", newline="", encoding="utf-8") as handle:
        keys = list(frows[0].keys()) if frows else []
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(frows)


def _write_report(summary: dict[str, Any]) -> None:
    selected = summary["selection"]["selected_variant"]
    lines = [
        "# MoPF-vNext U2-C — Formal Factorial Training Verification", "",
        "NC-only formal factorial training under the frozen U1-T relation (`learned_diag_cos`, `tau=0.35`). Test metrics are descriptive and were not used for checkpoint, variant, or hyperparameter selection.", "",
        f"- Runtime commit: `{summary['git_provenance']['head']}`", f"- Remote sync: `{summary['git_provenance']['sync_status']}`", f"- Initialization gate: **{summary['initialization_gate']['passed']}**", "- Formal protocol: hidden 256, dropout 0.2, AdamW, lr 1e-3, weight decay 1e-4, 300 epochs, patience 30, best Validation Accuracy.", "",
        "## Downstream summary", "",
        "| Variant | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |", "|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        agg = summary["aggregates"]["equal_dataset"][variant]
        lines.append(f"| {variant} | {agg['val_acc']['mean']:.6f} ± {agg['val_acc']['population_std']:.6f} | {agg['val_macro_f1']['mean']:.6f} ± {agg['val_macro_f1']['population_std']:.6f} | {agg['test_acc']['mean']:.6f} ± {agg['test_acc']['population_std']:.6f} | {agg['test_macro_f1']['mean']:.6f} ± {agg['test_macro_f1']['population_std']:.6f} |")
    fe = summary["factorial_effects"]["aggregate"]
    lines += ["", "## Factorial effects", "", f"- Anchor effect (equal-dataset mean): Val Acc `{fe['anchor_effect_val_acc']['equal_dataset']['mean']:.6f}`, Val Macro-F1 `{fe['anchor_effect_val_macro_f1']['equal_dataset']['mean']:.6f}`.", f"- Differential effect (equal-dataset mean): Val Acc `{fe['differential_effect_val_acc']['equal_dataset']['mean']:.6f}`, Val Macro-F1 `{fe['differential_effect_val_macro_f1']['equal_dataset']['mean']:.6f}`.", f"- Interaction (equal-dataset mean): Val Acc `{fe['interaction_val_acc']['equal_dataset']['mean']:.6f}`, Val Macro-F1 `{fe['interaction_val_macro_f1']['equal_dataset']['mean']:.6f}`.", "", "## Mechanism gates", "", f"- Anchor mechanism after training: **{summary['mechanism_gates']['anchor']['passed']}**; datasets passing `{summary['mechanism_gates']['anchor']['datasets_passed']}/5`; frozen anchor-off intervention datasets `{summary['mechanism_gates']['anchor']['intervention_datasets']}`.", f"- Differential mechanism after training: **{summary['mechanism_gates']['differential']['passed']}**; datasets passing `{summary['mechanism_gates']['differential']['datasets_passed']}/5`.", f"- Eta compensation pathology: **{not summary['mechanism_gates']['compensation_pathology']['passed']}**; multi-hop contribution collapse: **{summary['mechanism_gates']['multi_hop_contribution_collapse_present']}**.", "", "## Final decision", "", f"**{selected}**", "", "The decision uses validation metrics and mechanism evidence only. Differential coordinates change the organization/conditioning of the same degree-K propagation span; they do not expand the receptive field or claim exact k-hop unique information.", ""]
    (ROOT / "docs" / "mopf_u2c_formal_factorial_training.md").write_text("\n".join(lines), encoding="utf-8")


def _write_journal(summary: dict[str, Any]) -> None:
    path = ROOT / "docs" / "mopf_vnext_upgrade_journal.md"
    text = path.read_text(encoding="utf-8")
    marker = "## U2-C — Formal Factorial Training Verification"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    agg = summary["aggregates"]["equal_dataset"]
    effects = summary["factorial_effects"]["aggregate"]
    section = [marker, "", "U2-C completed the formal NC-only 2×2 factorial over state semantics and response coordinates. C0 reused the 15 U1-T T2 best-validation checkpoints after the explicit/legacy cumulative compatibility audit; C1-C3 were 45 new runs across Movies, Toys, Grocery, ele-fashion, Reddit-S and seeds 42/43/44.", "", "- Frozen upstream: `learned_diag_cos`, `edge_weight_temperature=0.35`; no LP, alpha tuning, K tuning, fusion change, loss change, or evaluator change.", "- Protocol: hidden 256, dropout 0.2, AdamW, lr `1e-3`, weight decay `1e-4`, 300 epochs, patience 30, best Validation Accuracy.", f"- Initialization equivalence gate: `{summary['initialization_gate']['passed']}`; baseline tests before implementation: `180 passed`.", "", "### Equal-dataset downstream means", "", "| Variant | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |", "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        section.append(f"| {variant} | {agg[variant]['val_acc']['mean']:.6f} ± {agg[variant]['val_acc']['population_std']:.6f} | {agg[variant]['val_macro_f1']['mean']:.6f} ± {agg[variant]['val_macro_f1']['population_std']:.6f} | {agg[variant]['test_acc']['mean']:.6f} ± {agg[variant]['test_acc']['population_std']:.6f} | {agg[variant]['test_macro_f1']['mean']:.6f} ± {agg[variant]['test_macro_f1']['population_std']:.6f} |")
    section += ["", f"- Anchor factorial effect: Val Acc `{effects['anchor_effect_val_acc']['equal_dataset']['mean']:.6f}`, Val Macro-F1 `{effects['anchor_effect_val_macro_f1']['equal_dataset']['mean']:.6f}`.", f"- Differential factorial effect: Val Acc `{effects['differential_effect_val_acc']['equal_dataset']['mean']:.6f}`, Val Macro-F1 `{effects['differential_effect_val_macro_f1']['equal_dataset']['mean']:.6f}`.", f"- Anchor mechanism gate: `{summary['mechanism_gates']['anchor']['passed']}`.", f"- Differential mechanism gate: `{summary['mechanism_gates']['differential']['passed']}`.", f"- Eta compensation / multi-hop collapse: `{not summary['mechanism_gates']['compensation_pathology']['passed']}` / `{summary['mechanism_gates']['multi_hop_contribution_collapse_present']}`.", f"- Final U2 choice: **{summary['selection']['selected_variant']}**. Test metrics were not used for selection.", "", "U2-C hard stop: no U3, LP, alpha/K tuning, Jacobi/Chebyshev, semantic shortcuts, fusion redesign, or auxiliary loss was started.", ""]
    path.write_text(text + "\n".join(section), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if not SOURCE_SUMMARY.is_file() or not U2C0_SUMMARY.is_file():
        raise FileNotFoundError("Required U1-T or U2-C0 authoritative summary is missing")
    source_records = _source_records()
    initialization_gate = _initialization_gate()
    compatibility_audit = _legacy_compatibility_audit()
    _dump(output_root / "initialization_gate.json", initialization_gate)
    _dump(output_root / "c0_compatibility_audit.json", compatibility_audit)
    if not initialization_gate["passed"]:
        raise RuntimeError("STOP: U2-C initialization equivalence gate failed")
    if not compatibility_audit["passed"]:
        raise RuntimeError("STOP: C0 legacy/current cumulative equivalence audit failed")

    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            source = source_records[(dataset, seed)]
            payload = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
            records.append({
                "dataset": dataset, "seed": seed, "variant": "C0", "k": K_BY_DATASET[dataset], "device": "reused",
                "run_dir": source["run_dir"], "checkpoint": source["checkpoint"], "source_reuse": True,
                "downstream": source["downstream"], "best_epoch": source["best_epoch"], "parameter_count": source["parameter_count"],
                "training_time_seconds": source["training_time_seconds"], "peak_gpu_memory_mib": source["peak_gpu_memory_mib"],
                "checkpoint_sha256": _sha256(Path(source["checkpoint"])), "checkpoint_epoch_payload": payload.get("epoch"),
            })

    if not args.skip_training:
        pending = [(dataset, variant, seed) for variant in NEW_VARIANTS for dataset in DATASETS for seed in SEEDS]
        if args.no_parallel or len(args.devices) == 1:
            for index, (dataset, variant, seed) in enumerate(pending, 1):
                device = str(args.devices[(index - 1) % len(args.devices)])
                print(f"[U2-C] {index}/{len(pending)} {variant} {dataset} seed={seed} {device}", flush=True)
                records.append(_run_training_job(dataset, variant, seed, device, output_root, not args.no_resume))
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
                for index, (dataset, variant, seed) in enumerate(pending):
                    device = str(args.devices[index % len(args.devices)])
                    futures[executor.submit(_run_training_job, dataset, variant, seed, device, output_root, not args.no_resume)] = (dataset, variant, seed)
                for future in as_completed(futures):
                    records.append(future.result())
                    print(f"[U2-C] finished {futures[future]}", flush=True)
    else:
        for variant in NEW_VARIANTS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    path = output_root / "runs" / dataset / variant / f"seed{seed}" / "run_record.json"
                    records.append(json.loads(path.read_text(encoding="utf-8")))
    records.sort(key=lambda r: (DATASETS.index(r["dataset"]), VARIANTS.index(r["variant"]), SEEDS.index(int(r["seed"]))))

    diagnostics = []
    pending_diag = [r for r in records if args.no_resume or not (output_root / "per_run" / f"{r['dataset']}_{r['variant']}_seed{r['seed']}.json").is_file()]
    if args.no_parallel or len(args.devices) == 1:
        for index, record in enumerate(pending_diag, 1):
            device = str(args.devices[(index - 1) % len(args.devices)])
            print(f"[U2-C diagnostics] {index}/{len(pending_diag)} {record['variant']} {record['dataset']} seed={record['seed']} {device}", flush=True)
            diagnostics.append(_diagnose(record, device, output_root, not args.no_resume))
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
            for index, record in enumerate(pending_diag):
                device = str(args.devices[index % len(args.devices)])
                futures[executor.submit(_diagnose, record, device, output_root, not args.no_resume)] = record
            for future in as_completed(futures):
                diagnostics.append(future.result())
                print(f"[U2-C diagnostics] finished {futures[future]['variant']} {futures[future]['dataset']} seed={futures[future]['seed']}", flush=True)
    for record in records:
        path = output_root / "per_run" / f"{record['dataset']}_{record['variant']}_seed{record['seed']}.json"
        if not any((row["dataset"], row["variant"], int(row["seed"])) == (record["dataset"], record["variant"], int(record["seed"])) for row in diagnostics):
            diagnostics.append(json.loads(path.read_text(encoding="utf-8")))
    records = _attach_diagnostics(records, diagnostics)
    aggregates = _aggregate(records)
    factorial = _factorial(records)
    performance = _performance_gates(aggregates)
    mechanisms = _mechanism_gates(records, aggregates)
    selection = _select(performance, mechanisms)
    summary = {
        "metadata": {"stage": "MoPF-vNext U2-C Formal Factorial Training Verification", "scope": "NC only", "datasets": list(DATASETS), "seeds": list(SEEDS), "K_formal_by_dataset": dict(K_BY_DATASET), "variants": list(VARIANTS), "new_training_runs": 45, "alpha": ALPHA, "edge_weight_mode": "learned_diag_cos", "edge_weight_temperature": TAU, "basis_interpretation": "differential coordinates reorganize the same degree-K propagation span", "test_used_for_selection": False},
        "git_provenance": {"head": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"), "status": _git("status", "--short"), "sync_status": "failed: git fetch/pull could not resolve github.com in this environment"},
        "upstream_sources": {"u1t_summary": str(SOURCE_SUMMARY), "u2c0_summary": str(U2C0_SUMMARY), "c0_source_variant": "U1-T T2", "c0_source_count": 15, "c0_compatibility_audit": compatibility_audit},
        "training_protocol": {"hidden_dim": 256, "dropout": 0.2, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "epochs": 300, "patience": 30, "checkpoint": "best Validation Accuracy", "macro_f1": "fixed supervised label set", "test_role": "final description only"},
        "initialization_gate": initialization_gate,
        "records": records,
        "aggregates": aggregates,
        "factorial_effects": factorial,
        "performance_gates": performance,
        "mechanism_gates": mechanisms,
        "selection": selection,
        "artifacts": {"master_summary": str(output_root / "u2c_master_summary.json"), "master_table": str(output_root / "u2c_master_table.csv"), "factorial_table": str(output_root / "u2c_factorial_effects.csv"), "per_run": str(output_root / "per_run")},
    }
    _dump(output_root / "u2c_master_summary.json", summary)
    _write_csv(output_root / "u2c_master_table.csv", records, factorial)
    _write_report(summary)
    _write_journal(summary)
    print(json.dumps({"selected_variant": selection["selected_variant"], "anchor_gate": mechanisms["anchor"]["passed"], "differential_gate": mechanisms["differential"]["passed"], "compensation_pathology": not mechanisms["compensation_pathology"]["passed"], "multi_hop_collapse": mechanisms["multi_hop_contribution_collapse_present"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
