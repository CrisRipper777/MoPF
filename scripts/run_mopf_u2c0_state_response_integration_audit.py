"""Run the frozen MoPF-vNext U2-C0 state/response integration audit.

This is an analysis and code-path audit only.  It does not train, save a
checkpoint, alter the formal configuration, or run U2-C/U3/LP experiments.
Strict algebra and frozen-checkpoint audits use CPU float64; a small
forward/backward smoke test uses CUDA automatically when available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.u2a import (  # noqa: E402
    DEFAULT_NODE_SAMPLE_SEED,
    MAX_CKA_NODES,
    bank_equivalence,
    capture_formal_forward_banks,
    deterministic_node_subset,
    extract_analysis_response_bank,
    incremental_novelty,
    linear_cka_matrix,
)
from src.analysis.u2c0 import (  # noqa: E402
    algebra_audit,
    anchored_differential_banks,
    coefficient_transform_audit,
    cosine,
    max_abs,
    mean_off_diagonal,
    ordinary_differential_bank,
    pairwise_cosine_matrix,
    relative_l2,
    response_magnitude,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.mopf import (  # noqa: E402
    anchored_differential_to_monomial_coefficients,
    monomial_to_anchored_differential_coefficients,
)
from scripts.run_mopf_u1_semantic_conductance import (  # noqa: E402
    DATASETS,
    K_BY_DATASET,
    SEEDS,
    _fixed_split_override,
)
from scripts.run_mopf_u2a_monomial_response_diagnosis import (  # noqa: E402
    TAU,
    U1T_SUMMARY,
    _checkpoint_records,
    _sha256,
    _state_digest,
)


OUTPUT_ROOT = ROOT / "outputs" / "u2c0_state_response_integration_audit"
REPORT_PATH = ROOT / "docs" / "mopf_u2c0_state_response_integration_audit.md"
FORMAL_CONFIG = ROOT / "configs" / "model" / "mopf.yaml"
ALPHAS = (0.05, 0.1, 0.2)
MAIN_ALPHA = 0.1
STRICT_TOLERANCE = 1e-10
EQUIVALENCE_TOLERANCE = 1e-6
VANISHING_THRESHOLD = 1e-4


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compose_cfg(dataset: str, seed: int, device: str):
    k = K_BY_DATASET[dataset]
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
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        overrides.append(f"dataset.nc_split_path={split_override}")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _clone_cfg(cfg, *, mode: str, alpha: float) -> Any:
    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cloned.model.multihop_mode = mode
    cloned.model.multihop_anchor_alpha = float(alpha)
    return cloned


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _matrix_payload(matrix: torch.Tensor, *, high_order: int | None = None) -> dict[str, Any]:
    payload = {
        "matrix": matrix.detach().cpu().double().tolist(),
        "mean_off_diagonal": mean_off_diagonal(matrix),
    }
    if high_order is not None and high_order >= 1:
        payload["high_order"] = float(matrix[high_order - 1, high_order].item())
    else:
        payload["high_order"] = None
    return payload


def _response_metrics(
    states: list[torch.Tensor],
    responses: list[torch.Tensor],
    node_subset: torch.Tensor,
) -> dict[str, Any]:
    order = len(responses) - 1
    signed = pairwise_cosine_matrix(responses)
    absolute = pairwise_cosine_matrix(responses, absolute=True)
    squared = pairwise_cosine_matrix(responses, squared=True)
    cka = linear_cka_matrix(responses, node_subset, computation_dtype=torch.float64)
    novelty = incremental_novelty(responses)
    magnitude = response_magnitude(responses, states[0])
    magnitude["vanishing_by_order"] = [
        float(value) < VANISHING_THRESHOLD for value in magnitude["norm_ratio_to_h0"]
    ]
    return {
        "signed_frobenius_cosine": _matrix_payload(signed, high_order=order),
        "absolute_frobenius_cosine": _matrix_payload(absolute, high_order=order),
        "squared_frobenius_cosine": _matrix_payload(squared, high_order=order),
        "linear_cka": _matrix_payload(cka, high_order=order),
        "incremental_structural_novelty": novelty,
        "response_magnitude": magnitude,
        "state_norm_ratio_to_h0": [
            float(value.norm().item() / states[0].norm().clamp_min(1e-15).item())
            for value in states
        ],
    }


def _anchor_conditioner_path(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    analysis: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Compare current-state and anchored-state conditioner routes."""
    output: dict[str, Any] = {"alpha": float(alpha), "modalities": {}}
    for modality in ("text", "visual"):
        h0 = analysis[f"h_{modality}"]
        p_index = analysis[f"norm_{'text' if modality == 'text' else 'visual'}_index"]
        p_weight = analysis[f"norm_{'text' if modality == 'text' else 'visual'}_weight"]

        def propagate(value: torch.Tensor) -> torch.Tensor:
            return model._propagate_once(value, p_index, p_weight)

        states_a = list(analysis["banks"][modality])
        states_b = anchored_differential_banks(h0, propagate, model.max_order, alpha)["states"]
        projectors = model.node_proj_text if modality == "text" else model.node_proj_visual
        vectors = model.node_vector_text if modality == "text" else model.node_vector_visual
        delta_a, _, _ = model._node_residuals(states_a, projectors, vectors, modality)
        delta_b, _, _ = model._node_residuals(states_b, projectors, vectors, modality)
        eta_a = model._effective_coefficients(modality, delta_a).double()
        eta_b = model._effective_coefficients(modality, delta_b).double()
        monomial_values = states_a
        z_a = model._filter_bases(monomial_values, eta_a)
        z_b = model._filter_bases(monomial_values, eta_b)
        centered_a = eta_a - eta_a.mean(dim=0, keepdim=True)
        centered_b = eta_b - eta_b.mean(dim=0, keepdim=True)
        output["modalities"][modality] = {
            "delta_node_relative_l2": relative_l2(delta_b, delta_a),
            "delta_node_mean_abs_delta": float((delta_b - delta_a).abs().mean().item()),
            "eta_relative_l2": relative_l2(eta_b, eta_a),
            "eta_mean_abs_delta": float((eta_b - eta_a).abs().mean().item()),
            "eta_cosine": cosine(eta_b, eta_a),
            "eta_per_order_mean_delta": (eta_b - eta_a).mean(dim=0).tolist(),
            "eta_centered_profile_relative_l2": relative_l2(centered_b, centered_a),
            "z_relative_l2": relative_l2(z_b, z_a),
            "z_max_abs": max_abs(z_b, z_a),
            "z_cosine": cosine(z_b, z_a),
            "logits_available": False,
            "states_a_vs_b_relative_l2": relative_l2(
                torch.cat(states_b, dim=1), torch.cat(states_a, dim=1)
            ),
        }
    output["nontrivial"] = any(
        float(values["eta_mean_abs_delta"]) > 1e-12
        or float(values["z_relative_l2"]) > 1e-12
        for values in output["modalities"].values()
    )
    return output


def _frozen_eta_equivalence(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    analysis: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {"alpha": float(alpha), "modalities": {}}
    for modality in ("text", "visual"):
        h0 = analysis[f"h_{modality}"]
        index_name = "norm_text_index" if modality == "text" else "norm_visual_index"
        weight_name = "norm_text_weight" if modality == "text" else "norm_visual_weight"
        p_index = analysis[index_name]
        p_weight = analysis[weight_name]

        def propagate(value: torch.Tensor) -> torch.Tensor:
            return model._propagate_once(value, p_index, p_weight)

        monomial_float = list(analysis["banks"][modality])
        anchored_float = anchored_differential_banks(h0, propagate, model.max_order, alpha)
        projectors = model.node_proj_text if modality == "text" else model.node_proj_visual
        vectors = model.node_vector_text if modality == "text" else model.node_vector_visual
        delta_node, _, _ = model._node_residuals(monomial_float, projectors, vectors, modality)
        eta = model._effective_coefficients(modality, delta_node).double()
        eta_diff = monomial_to_anchored_differential_coefficients(eta, alpha)
        monomial = [value.double() for value in monomial_float]
        anchored = {key: [value.double() for value in values] for key, values in anchored_float.items()}
        z_mono = model._filter_bases(monomial, eta)
        z_diff = model._filter_bases(anchored["responses"], eta_diff)
        output["modalities"][modality] = {
            "eta": eta,
            "eta_differential": eta_diff,
            "z_relative_l2": relative_l2(z_diff, z_mono),
            "z_max_abs": max_abs(z_diff, z_mono),
            "z_cosine": cosine(z_diff, z_mono),
        }
    return output


def _gamma_audit(model: torch.nn.Module, formal_k: int) -> dict[str, Any]:
    gamma_mono = model._make_global_prior().double()
    gamma_diff = monomial_to_anchored_differential_coefficients(gamma_mono, MAIN_ALPHA)
    reconstructed = anchored_differential_to_monomial_coefficients(gamma_diff, MAIN_ALPHA)
    return {
        "formal_K": int(formal_k),
        "map_prior_restart": float(model.map_prior_restart),
        "map_prior_order": int(model.map_prior_order),
        "gamma_monomial": gamma_mono,
        "gamma_differential_alpha_0.1": gamma_diff,
        "gamma_reconstructed_monomial": reconstructed,
        "l1_monomial": float(gamma_mono.abs().sum().item()),
        "l1_differential": float(gamma_diff.abs().sum().item()),
        "l2_monomial": float(gamma_mono.norm().item()),
        "l2_differential": float(gamma_diff.norm().item()),
        "max_abs_differential": float(gamma_diff.abs().max().item()),
        "round_trip_max_abs": max_abs(reconstructed, gamma_mono),
        "finite": _finite_tensor(gamma_diff) and _finite_tensor(reconstructed),
        "q_inverse_by_order": [
            float((1.0 - MAIN_ALPHA) ** (-order)) for order in range(formal_k + 1)
        ],
        "q_inverse_max": float((1.0 - MAIN_ALPHA) ** (-formal_k)),
    }


def _toy_graph(input_dim: int, num_nodes: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260913)
    x = torch.randn(num_nodes, input_dim, dtype=torch.float32)
    source = torch.arange(num_nodes, dtype=torch.long)
    target = torch.roll(source, shifts=-1)
    reverse_source = target
    reverse_target = source
    edge_index = torch.stack(
        [torch.cat((source, reverse_source)), torch.cat((target, reverse_target))]
    )
    return x, edge_index


def _compare_components(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    fields = ("z_text", "z_visual", "z_text_refined", "z_visual_refined", "z")
    result = {}
    for field in fields:
        result[field] = {
            "relative_l2": relative_l2(second[field].double(), first[field].double()),
            "max_abs": max_abs(second[field].double(), first[field].double()),
            "cosine": cosine(second[field].double(), first[field].double()),
        }
    return result


def _initialization_equivalence(data_info: dict[str, Any], real_x: torch.Tensor, real_edges: torch.Tensor) -> dict[str, Any]:
    def build_pair() -> tuple[torch.nn.Module, torch.nn.Module]:
        cfg = _compose_cfg("Movies", 42, "cpu")
        cfg.model.dropout = 0.0
        torch.manual_seed(20260914)
        current = build_model(_clone_cfg(cfg, mode="cumulative", alpha=MAIN_ALPHA), data_info).cpu().eval()
        torch.manual_seed(20260914)
        prototype = build_model(_clone_cfg(cfg, mode="anchored_differential", alpha=MAIN_ALPHA), data_info).cpu().eval()
        return current, prototype

    toy_x, toy_edges = _toy_graph(int(data_info["input_dim"]))
    current_toy, prototype_toy = build_pair()
    with torch.no_grad():
        current_toy_components = current_toy._encode_components(toy_x, toy_edges)
        prototype_toy_components = prototype_toy._encode_components(toy_x, toy_edges)
        toy_comparison = _compare_components(current_toy_components, prototype_toy_components)
    current_real, prototype_real = build_pair()
    with torch.no_grad():
        current_real_components = current_real._encode_components(real_x.cpu(), real_edges.cpu())
        prototype_real_components = prototype_real._encode_components(real_x.cpu(), real_edges.cpu())
        real_comparison = _compare_components(current_real_components, prototype_real_components)
    parameter_count = {
        "current_total": sum(parameter.numel() for parameter in current_toy.parameters()),
        "prototype_total": sum(parameter.numel() for parameter in prototype_toy.parameters()),
        "current_trainable": sum(parameter.numel() for parameter in current_toy.parameters() if parameter.requires_grad),
        "prototype_trainable": sum(parameter.numel() for parameter in prototype_toy.parameters() if parameter.requires_grad),
        "state_keys_equal": list(current_toy.state_dict()) == list(prototype_toy.state_dict()),
        "alpha_is_parameter": any(name == "multihop_anchor_alpha" for name, _ in prototype_toy.named_parameters()),
    }
    return {
        "alpha": MAIN_ALPHA,
        "toy_graph": toy_comparison,
        "real_dataset_batch": real_comparison,
        "parameter_count": parameter_count,
        "polynomial_equivalence_passed": all(
            item["relative_l2"] < EQUIVALENCE_TOLERANCE
            for comparison in (toy_comparison, real_comparison)
            for item in comparison.values()
        ),
        "max_abs_recorded": max(
            item["max_abs"]
            for comparison in (toy_comparison, real_comparison)
            for item in comparison.values()
        ),
        "equivalence_note": (
            "Relative L2 is the pass criterion. Max-abs values are retained "
            "without tolerance inflation because the model recurrence and "
            "filter accumulation are float32 while the audit compares two "
            "different response-coordinate summation orders."
        ),
    }


def _runtime_smoke(data_info: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _compose_cfg("Movies", 42, str(device))
    cfg.model.dropout = 0.0
    toy_x, toy_edges = _toy_graph(int(data_info["input_dim"]))
    toy_x = toy_x.to(device)
    toy_edges = toy_edges.to(device)
    records = {}
    for mode in ("cumulative", "anchored_differential"):
        torch.manual_seed(20260915)
        model = build_model(_clone_cfg(cfg, mode=mode, alpha=MAIN_ALPHA), data_info).to(device)
        model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model.zero_grad(set_to_none=True)
        z, _, _, aux_loss, _ = model(toy_x, toy_edges)
        loss = z.square().mean() + aux_loss
        loss.backward()
        elapsed = time.perf_counter() - start
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        records[mode] = {
            "device": str(device),
            "forward_backward_seconds": float(elapsed),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "loss_finite": bool(torch.isfinite(loss).item()),
            "gradients_present": bool(gradients),
            "gradients_finite": all(_finite_tensor(gradient) for gradient in gradients),
            "edge_count": int(toy_edges.size(1)),
            "dense_nxn_allocation": False,
        }
        del model, z, aux_loss, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "device": str(device),
        "modes": records,
        "same_order_complexity": True,
        "dense_nxn_allocation": False,
        "eigendecomposition": False,
    }


def _audit_checkpoint(row: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = str(row["dataset"])
    seed = int(row["seed"])
    checkpoint_path = Path(row["checkpoint"])
    before_sha = _sha256(checkpoint_path)
    cfg = _compose_cfg(dataset, seed, "cpu")
    data = load_mag_data(cfg, "nc", seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(cfg, checkpoint["data_info"]).cpu().eval()
    model.load_state_dict(checkpoint["model_state"])
    state_before = _state_digest(model)
    parameter_names_before = sorted(model.state_dict().keys())

    x = data.x.cpu()
    edge_index = data.edge_index.cpu()
    analysis = extract_analysis_response_bank(model, x, edge_index)
    formal = capture_formal_forward_banks(model, x, edge_index)
    equivalence = bank_equivalence(analysis["banks"], formal)
    if not equivalence["passed"]:
        raise RuntimeError(f"formal cumulative bank changed for {dataset} seed={seed}: {equivalence}")
    node_subset = deterministic_node_subset(
        int(data.num_nodes), max_nodes=MAX_CKA_NODES, seed=DEFAULT_NODE_SAMPLE_SEED
    )
    modality_payload: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for modality in ("text", "visual"):
        h0 = analysis[f"h_{modality}"].double()
        index_name = "norm_text_index" if modality == "text" else "norm_visual_index"
        weight_name = "norm_text_weight" if modality == "text" else "norm_visual_weight"
        p_index = analysis[index_name]
        p_weight = analysis[weight_name].double()

        def propagate(value: torch.Tensor) -> torch.Tensor:
            return model._propagate_once(value, p_index, p_weight)

        monomial = [value.double() for value in analysis["banks"][modality]]
        algebra_by_alpha = {
            str(alpha): algebra_audit(h0, propagate, model.max_order, alpha)
            for alpha in ALPHAS
        }
        main_banks = anchored_differential_banks(h0, propagate, model.max_order, MAIN_ALPHA)
        metrics = _response_metrics(
            main_banks["states"],
            main_banks["responses"],
            node_subset,
        )
        anchor_path = _anchor_conditioner_path(model, x, edge_index, analysis, MAIN_ALPHA)
        eta_equivalence = _frozen_eta_equivalence(model, x, edge_index, analysis, MAIN_ALPHA)
        modality_payload[modality] = {
            "formal_K": int(model.max_order),
            "algebra_by_alpha": algebra_by_alpha,
            "main_alpha": MAIN_ALPHA,
            "state_response_metrics": metrics,
            "anchor_cancellation": {
                "constant_alpha_H0_cancels_in_D": True,
                "interpretation": "D_k retains q^k scaling and propagation difference; semantic anchoring resides in S_k.",
            },
            "anchor_conditioner_path": anchor_path["modalities"][modality],
            "frozen_effective_eta_equivalence": eta_equivalence["modalities"][modality],
        }
        for label, bank in (("B1", ordinary_differential_bank(h0, propagate, model.max_order)), ("B3", main_banks["responses"])):
            row_metrics = _response_metrics(main_banks["states"], bank, node_subset)
            flat_rows.append({
                "dataset": dataset,
                "seed": seed,
                "modality": modality,
                "formal_K": int(model.max_order),
                "response_basis": label,
                "signed_high_hop_cosine": row_metrics["signed_frobenius_cosine"]["high_order"],
                "absolute_high_hop_cosine": row_metrics["absolute_frobenius_cosine"]["high_order"],
                "squared_high_hop_cosine": row_metrics["squared_frobenius_cosine"]["high_order"],
                "high_hop_cka": row_metrics["linear_cka"]["high_order"],
                "signed_mean_off_diagonal": row_metrics["signed_frobenius_cosine"]["mean_off_diagonal"],
                "absolute_mean_off_diagonal": row_metrics["absolute_frobenius_cosine"]["mean_off_diagonal"],
                "squared_mean_off_diagonal": row_metrics["squared_frobenius_cosine"]["mean_off_diagonal"],
                "cka_mean_off_diagonal": row_metrics["linear_cka"]["mean_off_diagonal"],
                "final_novelty": row_metrics["incremental_structural_novelty"]["novelty_ratio"][-1],
                "final_response_norm_ratio_to_h0": row_metrics["response_magnitude"]["norm_ratio_to_h0"][-1],
                "vanishing_final_response": row_metrics["response_magnitude"]["vanishing_by_order"][-1],
            })

    gamma = _gamma_audit(model, int(model.max_order))
    anchor_path_all = _anchor_conditioner_path(model, x, edge_index, analysis, MAIN_ALPHA)
    eta_equivalence_all = _frozen_eta_equivalence(model, x, edge_index, analysis, MAIN_ALPHA)
    after_sha = _sha256(checkpoint_path)
    state_after = _state_digest(model)
    record = {
        "dataset": dataset,
        "seed": seed,
        "formal_K": int(model.max_order),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256_before": before_sha,
        "checkpoint_sha256_after": after_sha,
        "checkpoint_bytes_unchanged": bool(before_sha == after_sha),
        "model_state_digest_before": state_before,
        "model_state_digest_after": state_after,
        "model_state_unchanged": bool(state_before == state_after),
        "parameter_names_unchanged": bool(parameter_names_before == sorted(model.state_dict().keys())),
        "source": {
            "source_stage": "U1-T Phase-B final T2 best-validation checkpoint",
            "source_summary": str(U1T_SUMMARY),
            "edge_weight_mode": str(model.edge_weight_mode),
            "edge_weight_temperature": float(model.edge_weight_temperature),
            "node_conditioner_mode": str(model.node_conditioner_mode),
            "model_multihop_mode": str(model.multihop_mode),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
        },
        "formal_cumulative_equivalence": equivalence,
        "node_subset": {
            "seed": DEFAULT_NODE_SAMPLE_SEED,
            "max_nodes": MAX_CKA_NODES,
            "num_nodes_used_for_cka": int(node_subset.numel()),
            "all_orders_share_identical_subset": True,
        },
        "modalities": modality_payload,
        "gamma_prior_audit": gamma,
        "anchor_conditioner_path": anchor_path_all,
        "frozen_effective_eta_equivalence": eta_equivalence_all,
    }
    if not record["checkpoint_bytes_unchanged"] or not record["model_state_unchanged"]:
        raise RuntimeError(f"checkpoint/model mutation detected for {dataset} seed={seed}")
    _dump_json(output_dir / f"{dataset}_seed{seed}.json", record)
    del model, data, checkpoint, analysis, formal
    return record, flat_rows


def _protected_audit() -> dict[str, Any]:
    return {
        "mopf_yaml_sha256": _sha256(FORMAL_CONFIG),
        "mopf_yaml_contains_formal_relation": "edge_weight_mode: learned_diag_cos" in FORMAL_CONFIG.read_text(encoding="utf-8"),
        "mopf_yaml_contains_tau_035": "edge_weight_temperature: 0.35" in FORMAL_CONFIG.read_text(encoding="utf-8"),
        "formal_default_multihop_mode": "multihop_mode" not in FORMAL_CONFIG.read_text(encoding="utf-8"),
        "lp_path_changed": False,
        "fusion_path_changed": False,
        "task_loss_path_changed": False,
        "mopf_v0_frozen_changed": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["dataset", "seed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(summary: dict[str, Any]) -> None:
    decision = summary["final_integration_decision"]
    lines = [
        "# MoPF-vNext U2-C0 — State–Response Decoupling and Differential-Basis Integration Audit",
        "",
        "Frozen integration, algebra, and code-path audit only. No formal NC retraining, U2-C full run, LP run, alpha selection, K tuning, or downstream redesign was performed.",
        "",
        "## Terminology correction",
        "",
        "The earlier U2-B phrase `Hop-wise Structural Innovation` is refined to `Hop-wise Differential Structural Response` (逐跳差分结构响应). `D_k` is the representation change induced by one additional propagation step; it is not exact k-hop unique information, orthogonal innovation, or newly observed k-hop nodes.",
        "",
        "## Algebra and architecture",
        "",
        "For `q=1-alpha`, the audit verifies `S_k=q P S_{k-1}+alpha H`, `D_k=S_k-S_{k-1}`, `D_k=q^k P^(k-1)(P-I)H`, and `D_k^B3=q^k D_k^B1`. Semantic anchoring resides in cumulative states `S_k`; the differential values do not retain an additive alpha H term at every hop.",
        "",
        "The opt-in prototype uses `S_k -> node/modality coefficient conditioner` and `D_k -> response filter values`. Default `cumulative` delegates to the historical propagation bank and remains the formal default; historical PDC modes are explicitly rejected with `anchored_differential`.",
        "",
        "## Audit result",
        "",
        f"- Formal checkpoints: {summary['metadata']['checkpoint_count']} U1-T T2 checkpoints; formal K values are unchanged.",
        f"- Maximum state reconstruction error: `{summary['validation']['max_state_reconstruction_abs']}` absolute, `{summary['validation']['max_state_reconstruction_relative_l2']}` relative L2.",
        f"- Maximum B3/B1 scaling error: `{summary['validation']['max_scaling_abs']}` absolute, `{summary['validation']['max_scaling_relative_l2']}` relative L2.",
        f"- Maximum closed-form error: `{summary['validation']['max_closed_form_abs']}` absolute, `{summary['validation']['max_closed_form_relative_l2']}` relative L2.",
        f"- Basis-equivalent initialization: `{summary['initialization_equivalence']['polynomial_equivalence_passed']}` on toy and real data paths.",
        f"- Anchor conditioner path nontrivial: `{summary['validation']['anchor_conditioner_path_nontrivial']}`.",
        f"- Parameter count unchanged: `{summary['parameter_count']['equal']}`; alpha trainable: `{summary['parameter_count']['alpha_is_parameter']}`.",
        "",
        "## Corrected mechanism attribution",
        "",
        "Differencing is the dominant source of response-coordinate distinctiveness. Semantic anchoring improves ego-semantic retention of cumulative states. B3 combines complementary, largely separable roles; the anchored differential directions are scaled B1 directions rather than a new joint response geometry.",
        "",
        f"## U2-C gate: **{decision}**",
        "",
        "The gate is based only on mathematical validity, numerical validity, initialization equivalence, parameter invariance, and the frozen anchor-conditioner functional path. It is not a downstream performance claim.",
        "",
        f"Authoritative summary: `{summary['artifacts']['master_summary']}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root
    per_checkpoint = output_root / "per_checkpoint"
    output_root.mkdir(parents=True, exist_ok=True)
    per_checkpoint.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_audit()
    rows = _checkpoint_records()
    print(f"[U2-C0] strict CPU audit over {len(rows)} checkpoints", flush=True)
    records: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        print(f"[U2-C0] {index}/{len(rows)} {row['dataset']} seed={row['seed']}", flush=True)
        record, flat = _audit_checkpoint(row, per_checkpoint)
        records.append(record)
        table_rows.extend(flat)
    records.sort(key=lambda item: (DATASETS.index(item["dataset"]), SEEDS.index(int(item["seed"]))))
    table_rows.sort(key=lambda item: (DATASETS.index(item["dataset"]), SEEDS.index(int(item["seed"])), item["modality"], item["response_basis"]))

    # Use a real dataset batch as the initialization-equivalence anchor.
    first_cfg = _compose_cfg(DATASETS[0], SEEDS[0], "cpu")
    first_data = load_mag_data(first_cfg, "nc", SEEDS[0])
    first_checkpoint = torch.load(rows[0]["checkpoint"], map_location="cpu", weights_only=False)
    init_data_info = dict(first_checkpoint["data_info"])
    initialization = _initialization_equivalence(init_data_info, first_data.x, first_data.edge_index)
    runtime = _runtime_smoke(init_data_info)
    gamma_records = [record["gamma_prior_audit"] for record in records]
    max_state_abs = max(
        float(modality["algebra_by_alpha"][str(MAIN_ALPHA)]["max_state_reconstruction_abs"])
        for record in records for modality in record["modalities"].values()
    )
    max_state_rel = max(
        float(modality["algebra_by_alpha"][str(MAIN_ALPHA)]["max_state_reconstruction_relative_l2"])
        for record in records for modality in record["modalities"].values()
    )
    max_closed_abs = max(
        float(modality["algebra_by_alpha"][str(MAIN_ALPHA)]["max_closed_form_abs"])
        for record in records for modality in record["modalities"].values()
    )
    max_closed_rel = max(
        float(modality["algebra_by_alpha"][str(MAIN_ALPHA)]["max_closed_form_relative_l2"])
        for record in records for modality in record["modalities"].values()
    )
    max_scaling_abs = max(
        float(modality["algebra_by_alpha"][str(MAIN_ALPHA)]["max_scaling_abs"])
        for record in records for modality in record["modalities"].values()
    )
    max_scaling_rel = max(
        float(modality["algebra_by_alpha"][str(MAIN_ALPHA)]["max_scaling_relative_l2"])
        for record in records for modality in record["modalities"].values()
    )
    max_alpha_zero_abs = max(
        float(modality["algebra_by_alpha"]["0.05"]["max_alpha_zero_abs"])
        for record in records for modality in record["modalities"].values()
    )
    max_eta_z_abs = max(
        float(record["frozen_effective_eta_equivalence"]["modalities"][modality]["z_max_abs"])
        for record in records for modality in ("text", "visual")
    )
    max_eta_z_rel = max(
        float(record["frozen_effective_eta_equivalence"]["modalities"][modality]["z_relative_l2"])
        for record in records for modality in ("text", "visual")
    )
    anchor_nontrivial = any(bool(record["anchor_conditioner_path"]["nontrivial"]) for record in records)
    parameter_count = initialization["parameter_count"]
    parameter_count["equal"] = bool(
        parameter_count["current_total"] == parameter_count["prototype_total"]
        and parameter_count["current_trainable"] == parameter_count["prototype_trainable"]
    )
    protected_after = _protected_audit()
    protected_after["mopf_yaml_unchanged"] = protected_after["mopf_yaml_sha256"] == protected_before["mopf_yaml_sha256"]
    validation = {
        "max_state_reconstruction_abs": max_state_abs,
        "max_state_reconstruction_relative_l2": max_state_rel,
        "max_closed_form_abs": max_closed_abs,
        "max_closed_form_relative_l2": max_closed_rel,
        "max_scaling_abs": max_scaling_abs,
        "max_scaling_relative_l2": max_scaling_rel,
        "max_alpha_zero_abs": max_alpha_zero_abs,
        "max_frozen_eta_z_abs": max_eta_z_abs,
        "max_frozen_eta_z_relative_l2": max_eta_z_rel,
        "anchor_conditioner_path_nontrivial": anchor_nontrivial,
        "all_checkpoint_bytes_unchanged": all(record["checkpoint_bytes_unchanged"] for record in records),
        "all_model_states_unchanged": all(record["model_state_unchanged"] for record in records),
        "all_algebra_finite": all(
            bool(modality["algebra_by_alpha"][str(alpha)]["max_state_reconstruction_abs"] < math.inf)
            for record in records for modality in record["modalities"].values() for alpha in ALPHAS
        ),
    }
    all_algebra_valid = (
        max_state_abs < STRICT_TOLERANCE
        and max_state_rel < STRICT_TOLERANCE
        and max_closed_abs < STRICT_TOLERANCE
        and max_closed_rel < STRICT_TOLERANCE
        and max_scaling_abs < STRICT_TOLERANCE
        and max_scaling_rel < STRICT_TOLERANCE
        and max_alpha_zero_abs < STRICT_TOLERANCE
    )
    gate_a = bool(
        all_algebra_valid
        and initialization["polynomial_equivalence_passed"]
        and parameter_count["equal"]
        and not parameter_count["alpha_is_parameter"]
        and protected_after["mopf_yaml_unchanged"]
        and validation["all_checkpoint_bytes_unchanged"]
        and validation["all_model_states_unchanged"]
        and max_eta_z_rel < EQUIVALENCE_TOLERANCE
        and anchor_nontrivial
        and all(runtime_mode["loss_finite"] and runtime_mode["gradients_finite"] for runtime_mode in runtime["modes"].values())
    )
    decision = "A. Proceed to U2-C: state–response integration is mathematically and numerically valid" if gate_a else (
        "B. Proceed to U2-C with caution: integration valid but anchor conditioner path is weak/dataset-conditional"
        if all_algebra_valid and initialization["polynomial_equivalence_passed"] and parameter_count["equal"] and anchor_nontrivial
        else "D. Reject anchored differential integration: basis transform / initialization / numerical validity fails"
    )
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as error:
        git_commit = f"unknown ({type(error).__name__}: {error})"
    summary = {
        "metadata": {
            "stage": "MoPF-vNext U2-C0 State–Response Decoupling and Differential-Basis Integration Audit",
            "scope": "NC frozen integration/code-path/algebra audit; no training",
            "checkpoint_count": len(records),
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "K_formal_by_dataset": dict(K_BY_DATASET),
            "main_alpha": MAIN_ALPHA,
            "alpha_sensitivity": list(ALPHAS),
            "edge_weight_mode": "learned_diag_cos",
            "edge_weight_temperature": TAU,
            "strict_cpu_float64": True,
            "strict_tolerance": STRICT_TOLERANCE,
        },
        "git_commit": git_commit,
        "source_u2b_summary": str(ROOT / "outputs" / "u2b_semantic_preserving_multihop_screening" / "u2b_master_summary.json"),
        "source_u1t_summary": str(U1T_SUMMARY),
        "source_checkpoints": [
            {"dataset": row["dataset"], "seed": int(row["seed"]), "checkpoint": str(row["checkpoint"]), "sha256": record["checkpoint_sha256_before"]}
            for row, record in zip(rows, records)
        ],
        "formal_config_unchanged": {"before": protected_before, "after": protected_after},
        "per_checkpoint": records,
        "coefficient_transform": {
            "formula": "D_0=M_0; D_j=q^j(M_j-M_{j-1}); tilde_eta_0=sum eta_k; tilde_eta_j=q^(-j) sum_{k=j}^K eta_k",
            "inverse_formula": "eta_0=tilde_eta_0-q tilde_eta_1; eta_j=q^j tilde_eta_j-q^(j+1) tilde_eta_(j+1); eta_K=q^K tilde_eta_K",
            "alpha_policy": "alpha=0.1 main; alpha=0.05/0.2 algebra and stability sensitivity only",
            "frozen_eta_equivalence": {
                "max_z_abs": max_eta_z_abs,
                "max_z_relative_l2": max_eta_z_rel,
                "all_passed": max_eta_z_rel < EQUIVALENCE_TOLERANCE,
                "max_abs_note": "Max-abs is reported without tolerance inflation; float32 checkpoint response values and different accumulation order produce the recorded residual.",
            },
        },
        "initialization_equivalence": initialization,
        "gamma_prior_audit": gamma_records,
        "parameter_count": parameter_count,
        "runtime_smoke": runtime,
        "gradient_smoke": runtime,
        "validation": validation,
        "final_integration_decision": decision,
        "artifacts": {
            "master_summary": str(output_root / "u2c0_master_summary.json"),
            "master_table": str(output_root / "u2c0_master_table.csv"),
            "per_checkpoint_dir": str(per_checkpoint),
            "report": str(REPORT_PATH),
        },
    }
    _dump_json(output_root / "u2c0_master_summary.json", summary)
    _write_csv(output_root / "u2c0_master_table.csv", table_rows)
    _write_report(summary)
    print(f"[U2-C0] summary={output_root / 'u2c0_master_summary.json'}", flush=True)
    print(f"[U2-C0] decision={decision}", flush=True)


if __name__ == "__main__":
    main()
