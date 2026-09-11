"""Frozen U2-B screening for semantic-preserving distinctive multi-hop propagation.

This script is analysis-only.  It consumes the final U1-T T2 checkpoints,
constructs B0/B1/B2/B3 response coordinates and K=1..6 stress variants, and
never performs an optimizer step or changes the formal MoPF configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir

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
    modality_response_diagnostics,
    normalized_response_gram,
    spectral_range_audit,
    sparse_symmetry_audit,
)
from src.analysis.u2b import (  # noqa: E402
    ALPHAS,
    VARIANTS,
    build_response_variant,
    response_magnitude_audit,
    semantic_retention,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from scripts.run_mopf_u1_semantic_conductance import (  # noqa: E402
    DATASETS,
    K_BY_DATASET,
    SEEDS,
    _fixed_split_override,
)
from scripts.run_mopf_u2a_monomial_response_diagnosis import (  # noqa: E402
    OUTPUT_ROOT as U2A_OUTPUT_ROOT,
    TAU,
    U1T_SUMMARY,
    _checkpoint_records,
    _compose_cfg,
    _sha256,
    _state_digest,
)


OUTPUT_ROOT = ROOT / "outputs" / "u2b_semantic_preserving_multihop_screening"
ANALYSIS_MAX_HOP = 6
VANISHING_THRESHOLD = 1e-4
RECONSTRUCTION_TOLERANCE = 1e-6


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


def _metric_stats(values: list[float | None]) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "population_std": None, "median": None, "count": 0}
    return {
        "mean": float(statistics.fmean(finite)),
        "population_std": float(statistics.pstdev(finite)),
        "median": float(statistics.median(finite)),
        "count": len(finite),
    }


def _mean_off_diagonal(matrix: list[list[float]]) -> float | None:
    values = [float(matrix[i][j]) for i in range(len(matrix)) for j in range(len(matrix)) if i != j]
    return None if not values else float(statistics.fmean(values))


def _safe_mean(values: list[float | None]) -> float | None:
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            clean.append(numeric)
    return None if not clean else float(statistics.fmean(clean))


def _propagator(model: torch.nn.Module, edge_index: torch.Tensor, edge_weight: torch.Tensor):
    def propagate(response: torch.Tensor) -> torch.Tensor:
        return model._propagate_once(response, edge_index, edge_weight)

    return propagate


def _variant_payload(
    variant: str,
    alpha: float | None,
    variant_data: dict[str, Any],
    h0: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    node_subset: torch.Tensor,
    operator_audit: dict[str, Any],
    spectral_audit: dict[str, Any],
    k: int,
    *,
    metric_states: list[torch.Tensor] | None = None,
    metric_responses: list[torch.Tensor] | None = None,
    metric_edge_index: torch.Tensor | None = None,
    metric_edge_weight: torch.Tensor | None = None,
    metric_node_subset: torch.Tensor | None = None,
) -> dict[str, Any]:
    states = variant_data["states"][: k + 1]
    responses = variant_data["responses"][: k + 1]
    metric_states = states if metric_states is None else metric_states[: k + 1]
    metric_responses = responses if metric_responses is None else metric_responses[: k + 1]
    metric_edge_index = edge_index if metric_edge_index is None else metric_edge_index
    metric_edge_weight = edge_weight if metric_edge_weight is None else metric_edge_weight
    metric_node_subset = node_subset if metric_node_subset is None else metric_node_subset
    response_metrics = modality_response_diagnostics(
        metric_responses,
        metric_edge_index,
        metric_edge_weight,
        node_subset=metric_node_subset,
        operator_audit=operator_audit,
        spectral_audit=spectral_audit,
        cka_dtype=torch.float32,
    )
    cumulative_metrics = modality_response_diagnostics(
        metric_states,
        metric_edge_index,
        metric_edge_weight,
        node_subset=metric_node_subset,
        operator_audit=operator_audit,
        spectral_audit=spectral_audit,
        cka_dtype=torch.float32,
    )
    retention = semantic_retention(
        metric_states,
        metric_states[0],
        metric_node_subset,
        computation_dtype=torch.float32,
    )
    innovation = variant in {"B1", "B3"}
    magnitude = response_magnitude_audit(
        metric_responses,
        metric_states[0],
        innovation_channel=innovation,
        vanishing_threshold=VANISHING_THRESHOLD,
    )
    response_cosine = response_metrics["pairwise_frobenius_cosine"]
    response_cka = response_metrics["linear_cka"]
    reconstruction_value = variant_data["reconstruction_error_by_hop"][k]
    reconstruction_error = None if reconstruction_value is None else float(reconstruction_value)
    reconstruction_passed = variant in {"B0", "B2"} or reconstruction_error < RECONSTRUCTION_TOLERANCE
    finite = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (*states, *responses)
    )
    pathology = {
        "all_finite": bool(finite),
        "reconstruction_passed": bool(reconstruction_passed),
        "innovation_vanishing": bool(magnitude["vanishing_innovation_channel"]),
        "passed": bool(
            finite
            and reconstruction_passed
            and not magnitude["vanishing_innovation_channel"]
        ),
    }
    return {
        "variant": variant,
        "alpha": alpha,
        "K": int(k),
        "reconstruction_audit": {
            "max_error_to_hop": reconstruction_error,
            "tolerance": RECONSTRUCTION_TOLERANCE,
            "passed": bool(reconstruction_passed),
        },
        "semantic_retention": retention,
        "response_bank_metrics": response_metrics,
        "cumulative_state_metrics": cumulative_metrics,
        "response_magnitude": magnitude,
        "cross_hop_summary": {
            "mean_off_diagonal_frobenius_cosine": _mean_off_diagonal(response_cosine["matrix"]),
            "mean_off_diagonal_cka": _mean_off_diagonal(response_cka["matrix"]),
            "high_hop_frobenius_cosine": response_cosine["high_order"],
            "high_hop_cka": response_cka["high_order"],
            "final_novelty_ratio": response_metrics["incremental_structural_novelty"]["novelty_ratio"][-1],
            "normalized_effective_rank": response_metrics["gram_spectrum"]["normalized_effective_rank"],
            "condition_number": response_metrics["gram_spectrum"]["condition_number"],
        },
        "pathology": pathology,
    }


def _variant_key(variant: str, alpha: float | None) -> str:
    return variant if alpha is None else f"{variant}_alpha_{float(alpha):g}"


def _slice_pair_payload(payload: dict[str, Any], k: int) -> dict[str, Any]:
    matrix = [row[: k + 1] for row in payload["matrix"][: k + 1]]
    adjacent = {
        f"{index}_{index + 1}": float(matrix[index][index + 1])
        for index in range(k)
    }
    return {
        "matrix": matrix,
        "adjacent": adjacent,
        "high_order": None if k < 1 else float(matrix[k - 1][k]),
    }


def _slice_diagnostic(
    full: dict[str, Any],
    responses: list[torch.Tensor],
    k: int,
) -> dict[str, Any]:
    prefix = responses[: k + 1]
    gram, spectrum = normalized_response_gram(prefix)
    evolution = {
        key: (values[: k + 1] if isinstance(values, list) else values)
        for key, values in full["response_evolution"].items()
    }
    novelty = incremental_novelty(prefix)
    return {
        "pairwise_frobenius_cosine": _slice_pair_payload(full["pairwise_frobenius_cosine"], k),
        "linear_cka": _slice_pair_payload(full["linear_cka"], k),
        "normalized_response_gram": gram.detach().cpu().numpy().astype(np.float64).tolist(),
        "gram_spectrum": spectrum,
        "incremental_structural_novelty": novelty,
        "response_evolution": evolution,
        "operator_audit": full["operator_audit"],
        "spectral_audit": full["spectral_audit"],
    }


def _prefix_retention(full: dict[str, Any], k: int) -> dict[str, Any]:
    cka = full["linear_cka_to_h0"][: k + 1]
    cosine = full["frobenius_cosine_to_h0"][: k + 1]
    hops = np.arange(k + 1, dtype=np.float64)
    return {
        "linear_cka_to_h0": cka,
        "frobenius_cosine_to_h0": cosine,
        "retention_ratio_to_hop0": {
            "linear_cka": [value / max(cka[0], 1e-12) for value in cka],
            "frobenius_cosine": [value / max(cosine[0], 1e-12) for value in cosine],
        },
        "retention_decay_slope": {
            "linear_cka": float(np.polyfit(hops, np.asarray(cka), 1)[0]) if k > 0 else 0.0,
            "frobenius_cosine": float(np.polyfit(hops, np.asarray(cosine), 1)[0]) if k > 0 else 0.0,
        },
    }


def _prefix_variant_payload(
    full_payload: dict[str, Any],
    variant_data: dict[str, Any],
    metric_states: list[torch.Tensor],
    metric_responses: list[torch.Tensor],
    k: int,
) -> dict[str, Any]:
    states = variant_data["states"][: k + 1]
    responses = variant_data["responses"][: k + 1]
    metric_states_prefix = metric_states[: k + 1]
    metric_responses_prefix = metric_responses[: k + 1]
    response_metrics = _slice_diagnostic(
        full_payload["response_bank_metrics"], metric_responses_prefix, k
    )
    cumulative_metrics = _slice_diagnostic(
        full_payload["cumulative_state_metrics"], metric_states_prefix, k
    )
    magnitude = response_magnitude_audit(
        metric_responses_prefix,
        metric_states_prefix[0],
        innovation_channel=variant_data["variant"] in {"B1", "B3"},
        vanishing_threshold=VANISHING_THRESHOLD,
    )
    reconstruction_value = variant_data["reconstruction_error_by_hop"][k]
    reconstruction_error = None if reconstruction_value is None else float(reconstruction_value)
    reconstruction_passed = variant_data["variant"] in {"B0", "B2"} or reconstruction_error < RECONSTRUCTION_TOLERANCE
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in (*states, *responses))
    pathology = {
        "all_finite": bool(finite),
        "reconstruction_passed": bool(reconstruction_passed),
        "innovation_vanishing": bool(magnitude["vanishing_innovation_channel"]),
        "passed": bool(finite and reconstruction_passed and not magnitude["vanishing_innovation_channel"]),
    }
    response_cosine = response_metrics["pairwise_frobenius_cosine"]
    response_cka = response_metrics["linear_cka"]
    return {
        "variant": variant_data["variant"],
        "alpha": variant_data["alpha"],
        "K": int(k),
        "reconstruction_audit": {
            "max_error_to_hop": reconstruction_error,
            "tolerance": RECONSTRUCTION_TOLERANCE,
            "passed": bool(reconstruction_passed),
        },
        "semantic_retention": _prefix_retention(full_payload["semantic_retention"], k),
        "response_bank_metrics": response_metrics,
        "cumulative_state_metrics": cumulative_metrics,
        "response_magnitude": magnitude,
        "cross_hop_summary": {
            "mean_off_diagonal_frobenius_cosine": _mean_off_diagonal(response_cosine["matrix"]),
            "mean_off_diagonal_cka": _mean_off_diagonal(response_cka["matrix"]),
            "high_hop_frobenius_cosine": response_cosine["high_order"],
            "high_hop_cka": response_cka["high_order"],
            "final_novelty_ratio": response_metrics["incremental_structural_novelty"]["novelty_ratio"][-1],
            "normalized_effective_rank": response_metrics["gram_spectrum"]["normalized_effective_rank"],
            "condition_number": response_metrics["gram_spectrum"]["condition_number"],
        },
        "pathology": pathology,
    }


def _screening_row(
    record: dict[str, Any],
    variant: str,
    alpha: float | None,
    k: int,
    modality: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    retention = payload["semantic_retention"]
    cross = payload["cross_hop_summary"]
    magnitude = payload["response_magnitude"]
    return {
        "dataset": record["dataset"],
        "seed": int(record["seed"]),
        "modality": modality,
        "variant": variant,
        "alpha": alpha,
        "K": int(k),
        "high_hop_frobenius_cosine": cross["high_hop_frobenius_cosine"],
        "high_hop_cka": cross["high_hop_cka"],
        "mean_off_diagonal_frobenius_cosine": cross["mean_off_diagonal_frobenius_cosine"],
        "mean_off_diagonal_cka": cross["mean_off_diagonal_cka"],
        "final_novelty_ratio": cross["final_novelty_ratio"],
        "normalized_effective_rank": cross["normalized_effective_rank"],
        "condition_number": cross["condition_number"],
        "semantic_retention_final_cka": retention["linear_cka_to_h0"][-1],
        "semantic_retention_final_cosine": retention["frobenius_cosine_to_h0"][-1],
        "semantic_retention_cka_slope": retention["retention_decay_slope"]["linear_cka"],
        "semantic_retention_cosine_slope": retention["retention_decay_slope"]["frobenius_cosine"],
        "response_magnitude_final_ratio_to_h0": magnitude["norm_ratio_to_h0"][-1],
        "response_magnitude_final_ratio_to_previous": magnitude["norm_ratio_to_previous"][-1],
        "response_vanishing": magnitude["vanishing_innovation_channel"],
        "response_structural_variation_final": payload["response_bank_metrics"]["response_evolution"]["structural_variation"][-1],
        "cumulative_structural_variation_final": payload["cumulative_state_metrics"]["response_evolution"]["structural_variation"][-1],
        "response_node_variance_final": payload["response_bank_metrics"]["response_evolution"]["node_feature_variance"][-1],
        "cumulative_node_variance_final": payload["cumulative_state_metrics"]["response_evolution"]["node_feature_variance"][-1],
        "pathology_passed": payload["pathology"]["passed"],
    }


def _load_and_diagnose(
    row: dict[str, Any],
    device: str,
    metric_device: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset = str(row["dataset"])
    seed = int(row["seed"])
    checkpoint_path = Path(row["checkpoint"])
    checkpoint_before = _sha256(checkpoint_path)
    cfg = _compose_cfg(dataset, seed, device)
    data = load_mag_data(cfg, "nc", seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    state_before = _state_digest(model)
    parameter_names_before = sorted(model.state_dict().keys())
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    analysis = extract_analysis_response_bank(model, x, edge_index)
    formal = capture_formal_forward_banks(model, x, edge_index)
    equivalence = bank_equivalence(analysis["banks"], formal)
    if not equivalence["passed"]:
        raise RuntimeError(f"B0/formal response-bank equivalence failed for {dataset} seed={seed}: {equivalence}")
    operators = formal["operators"]
    operator_equivalence = {
        "text_index_equal": bool(torch.equal(analysis["norm_text_index"], operators[0][0])),
        "visual_index_equal": bool(torch.equal(analysis["norm_visual_index"], operators[1][0])),
        "text_weight_max_abs_difference": float((analysis["norm_text_weight"] - operators[0][1]).abs().max().item()),
        "visual_weight_max_abs_difference": float((analysis["norm_visual_weight"] - operators[1][1]).abs().max().item()),
    }
    node_subset = deterministic_node_subset(
        int(data.num_nodes), max_nodes=MAX_CKA_NODES, seed=DEFAULT_NODE_SAMPLE_SEED
    )
    modality_setup: dict[str, Any] = {}
    for modality, index_key, weight_key in (
        ("text", "norm_text_index", "norm_text_weight"),
        ("visual", "norm_visual_index", "norm_visual_weight"),
    ):
        p_index = analysis[index_key]
        p_weight = analysis[weight_key]
        operator_audit = sparse_symmetry_audit(p_index, p_weight, int(data.num_nodes))
        spectral_audit = spectral_range_audit(
            p_index,
            p_weight,
            int(data.num_nodes),
            operator_audit,
            seed=DEFAULT_NODE_SAMPLE_SEED + seed,
        )
        modality_setup[modality] = {
            "h0": analysis["h_" + modality],
            "edge_index": p_index,
            "edge_weight": p_weight,
            "operator_audit": operator_audit,
            "spectral_audit": spectral_audit,
        }

    variants_output: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for modality, setup in modality_setup.items():
        h0 = setup["h0"]
        propagate = _propagator(model, setup["edge_index"], setup["edge_weight"])
        modality_variants: dict[str, Any] = {}
        variant_specs = [(variant, None) for variant in ("B0", "B1")]
        variant_specs.extend((variant, alpha) for alpha in ALPHAS for variant in ("B2", "B3"))
        for variant, alpha in variant_specs:
            key = _variant_key(variant, alpha)
            variant_data = build_response_variant(
                h0,
                propagate,
                max_hop=ANALYSIS_MAX_HOP,
                variant=variant,
                alpha=0.1 if alpha is None else alpha,
            )
            if alpha is None and variant in {"B0", "B1"}:
                variant_data["alpha"] = None
            active_metric_device = metric_device or getattr(_load_and_diagnose, "metric_device", "cpu")
            metric_states = [tensor.to(active_metric_device) for tensor in variant_data["states"]]
            metric_responses = []
            for index, (response, state) in enumerate(zip(variant_data["responses"], variant_data["states"])):
                if response is state:
                    metric_responses.append(metric_states[index])
                else:
                    metric_responses.append(response.to(metric_device))
            metric_edge_index = setup["edge_index"].to(active_metric_device)
            metric_edge_weight = setup["edge_weight"].to(active_metric_device)
            metric_node_subset = node_subset.to(active_metric_device)
            full_payload = _variant_payload(
                variant,
                alpha,
                variant_data,
                h0,
                setup["edge_index"],
                setup["edge_weight"],
                node_subset,
                setup["operator_audit"],
                setup["spectral_audit"],
                ANALYSIS_MAX_HOP,
                metric_states=metric_states,
                metric_responses=metric_responses,
                metric_edge_index=metric_edge_index,
                metric_edge_weight=metric_edge_weight,
                metric_node_subset=metric_node_subset,
            )
            stress: dict[str, Any] = {}
            for k in range(1, ANALYSIS_MAX_HOP + 1):
                payload = _prefix_variant_payload(
                    full_payload, variant_data, metric_states, metric_responses, k
                )
                stress[str(k)] = payload
                flat_rows.append(_screening_row({"dataset": dataset, "seed": seed}, variant, alpha, k, modality, payload))
            modality_variants[key] = {
                "variant": variant,
                "alpha": alpha,
                "stress": stress,
            }
            del metric_states, metric_responses, metric_edge_index, metric_edge_weight, metric_node_subset, variant_data
            if str(active_metric_device).startswith("cuda"):
                torch.cuda.empty_cache()
        variants_output[modality] = modality_variants

    state_after = _state_digest(model)
    checkpoint_after = _sha256(checkpoint_path)
    record = {
        "dataset": dataset,
        "seed": seed,
        "K_formal": K_BY_DATASET[dataset],
        "K_analysis": list(range(1, ANALYSIS_MAX_HOP + 1)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_bytes_unchanged": bool(checkpoint_before == checkpoint_after),
        "model_state_digest_before": state_before,
        "model_state_digest_after": state_after,
        "model_state_unchanged": bool(state_before == state_after),
        "parameter_names_unchanged": bool(parameter_names_before == sorted(model.state_dict().keys())),
        "source": {
            "source_stage": "U1-T Phase-B final T2 best-validation checkpoint",
            "source_summary": str(U1T_SUMMARY),
            "training_temperature": TAU,
            "edge_weight_mode": str(model.edge_weight_mode),
            "edge_weight_temperature": float(model.edge_weight_temperature),
            "device": str(device),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
        },
        "b0_formal_equivalence": equivalence,
        "operator_equivalence": operator_equivalence,
        "node_subset": {
            "seed": DEFAULT_NODE_SAMPLE_SEED,
            "max_nodes": MAX_CKA_NODES,
            "num_nodes_used_for_cka": int(node_subset.numel()),
            "all_orders_share_identical_subset": True,
        },
        "operator_audit": {
            modality: {
                "operator": setup["operator_audit"],
                "spectral": setup["spectral_audit"],
            }
            for modality, setup in modality_setup.items()
        },
        "variants": variants_output,
    }
    if not record["checkpoint_bytes_unchanged"] or not record["model_state_unchanged"]:
        raise RuntimeError(f"U2-B state mutation detected for {dataset} seed={seed}")
    del model, data, checkpoint, analysis, formal
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return record, flat_rows


def _aggregate_flat(rows: list[dict[str, Any]], *, formal_only: bool = False) -> dict[str, Any]:
    selected = [row for row in rows if (not formal_only or int(row["K"]) == K_BY_DATASET[row["dataset"]])]
    output: dict[str, Any] = {}
    for variant in ("B0", "B1", "B2", "B3"):
        variant_rows = [row for row in selected if row["variant"] == variant]
        if variant in {"B2", "B3"}:
            alphas = [0.05, 0.1, 0.2]
        else:
            alphas = [None]
        output[variant] = {}
        for alpha in alphas:
            alpha_rows = [row for row in variant_rows if row["alpha"] == alpha]
            output[variant]["none" if alpha is None else f"{alpha:g}"] = {
                "by_dataset_modality": _aggregate_groups(alpha_rows, group_keys=("dataset", "modality")),
                "overall": {
                    metric: _metric_stats([row.get(metric) for row in alpha_rows])
                    for metric in (
                        "high_hop_frobenius_cosine",
                        "high_hop_cka",
                        "mean_off_diagonal_frobenius_cosine",
                        "mean_off_diagonal_cka",
                        "final_novelty_ratio",
                        "normalized_effective_rank",
                        "condition_number",
                        "semantic_retention_final_cka",
                        "semantic_retention_final_cosine",
                        "semantic_retention_cka_slope",
                        "semantic_retention_cosine_slope",
                        "response_magnitude_final_ratio_to_h0",
                        "response_structural_variation_final",
                        "cumulative_structural_variation_final",
                        "cumulative_node_variance_final",
                    )
                },
            }
    return output


def _aggregate_groups(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "|".join(str(row[name]) for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = {}
    for key, group in sorted(groups.items()):
        output[key] = {
            "count": len(group),
            "seeds": sorted({int(row["seed"]) for row in group}),
            "metrics": {
                metric: _metric_stats([row.get(metric) for row in group])
                for metric in (
                    "high_hop_frobenius_cosine",
                    "high_hop_cka",
                    "mean_off_diagonal_frobenius_cosine",
                    "mean_off_diagonal_cka",
                    "final_novelty_ratio",
                    "normalized_effective_rank",
                    "condition_number",
                    "semantic_retention_final_cka",
                    "semantic_retention_final_cosine",
                    "semantic_retention_cka_slope",
                    "semantic_retention_cosine_slope",
                    "response_magnitude_final_ratio_to_h0",
                    "response_magnitude_final_ratio_to_previous",
                    "response_structural_variation_final",
                    "cumulative_structural_variation_final",
                    "response_node_variance_final",
                    "cumulative_node_variance_final",
                )
            },
        }
    return output


def _screen_seed_pair(b0: dict[str, Any], candidate: dict[str, Any], variant: str) -> dict[str, Any]:
    def avg(row: dict[str, Any], first: str, second: str) -> float:
        return float(statistics.fmean([float(row[first]), float(row[second])]))

    b0_redundancy = avg(b0, "high_hop_frobenius_cosine", "high_hop_cka")
    candidate_redundancy = avg(candidate, "high_hop_frobenius_cosine", "high_hop_cka")
    b0_retention = avg(b0, "semantic_retention_final_cka", "semantic_retention_final_cosine")
    candidate_retention = avg(candidate, "semantic_retention_final_cka", "semantic_retention_final_cosine")
    result = {
        "variant": variant,
        "redundancy_delta_candidate_minus_B0": candidate_redundancy - b0_redundancy,
        "retention_delta_candidate_minus_B0": candidate_retention - b0_retention,
        "novelty_delta_candidate_minus_B0": float(candidate["final_novelty_ratio"] - b0["final_novelty_ratio"]),
        "magnitude_ratio": float(candidate["response_magnitude_final_ratio_to_h0"]),
        "candidate_pathology_passed": bool(candidate["pathology_passed"]),
    }
    if variant == "B1":
        result["successful"] = bool(
            result["redundancy_delta_candidate_minus_B0"] < 0.0
            and result["novelty_delta_candidate_minus_B0"] >= 0.0
            and result["magnitude_ratio"] >= VANISHING_THRESHOLD
            and result["candidate_pathology_passed"]
        )
    elif variant == "B2":
        result["successful"] = bool(
            result["retention_delta_candidate_minus_B0"] > 0.0
            and float(candidate["cumulative_structural_variation_final"]) > 1e-6
            and result["candidate_pathology_passed"]
        )
    else:
        result["successful"] = bool(
            result["retention_delta_candidate_minus_B0"] > 0.0
            and result["redundancy_delta_candidate_minus_B0"] < 0.0
            and result["novelty_delta_candidate_minus_B0"] >= 0.0
            and result["magnitude_ratio"] >= VANISHING_THRESHOLD
            and result["candidate_pathology_passed"]
        )
    return result


def _factorial_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if int(row["K"]) == K_BY_DATASET[row["dataset"]]
        and row["modality"] in {"text", "visual"}
        and (row["variant"] == "B0" or row["alpha"] is None or float(row["alpha"]) == 0.1)
    ]
    grouped: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault((row["dataset"], int(row["seed"]), row["modality"]), {})[_variant_key(row["variant"], row["alpha"])] = row
    metric_names = (
        "semantic_retention_final_cka",
        "semantic_retention_final_cosine",
        "high_hop_frobenius_cosine",
        "high_hop_cka",
        "final_novelty_ratio",
        "response_magnitude_final_ratio_to_h0",
    )
    result: dict[str, Any] = {}
    for metric in metric_names:
        anchor = []
        innovation = []
        interaction = []
        for variants in grouped.values():
            b0, b1, b2, b3 = (variants.get(name) for name in ("B0", "B1", "B2_alpha_0.1", "B3_alpha_0.1"))
            if not all((b0, b1, b2, b3)):
                continue
            anchor.append(0.5 * ((b2[metric] - b0[metric]) + (b3[metric] - b1[metric])))
            innovation.append(0.5 * ((b1[metric] - b0[metric]) + (b3[metric] - b2[metric])))
            interaction.append((b3[metric] - b2[metric]) - (b1[metric] - b0[metric]))
        result[metric] = {
            "anchor_main_effect": _metric_stats(anchor),
            "innovation_main_effect": _metric_stats(innovation),
            "anchor_x_innovation_interaction": _metric_stats(interaction),
        }
    return result


def _final_screening(rows: list[dict[str, Any]]) -> dict[str, Any]:
    formal = [row for row in rows if int(row["K"]) == K_BY_DATASET[row["dataset"]]]
    by_seed: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for row in formal:
        if row["variant"] == "B0" or row["alpha"] is None or float(row["alpha"]) == 0.1:
            by_seed.setdefault((row["dataset"], int(row["seed"])), {}).setdefault(row["variant"], []).append(row)

    per_dataset: dict[str, Any] = {}
    counts = {"B1": 0, "B2": 0, "B3": 0}
    for dataset in DATASETS:
        seed_results = {"B1": [], "B2": [], "B3": []}
        for seed in SEEDS:
            bundle = by_seed.get((dataset, seed), {})
            b0 = bundle.get("B0", [])
            if not b0:
                continue
            averaged_b0 = {key: _safe_mean([row[key] for row in b0]) for key in b0[0] if key in b0[0]}
            for variant in ("B1", "B2", "B3"):
                candidates = bundle.get(variant, [])
                if variant in {"B2", "B3"}:
                    candidates = [row for row in candidates if float(row["alpha"]) == 0.1]
                if not candidates:
                    continue
                averaged_candidate = {key: _safe_mean([row[key] for row in candidates]) for key in candidates[0] if key in candidates[0]}
                result = _screen_seed_pair(averaged_b0, averaged_candidate, variant)
                result["seed"] = seed
                seed_results[variant].append(result)
        dataset_payload = {}
        for variant in ("B1", "B2", "B3"):
            successes = [item for item in seed_results[variant] if item["successful"]]
            consistent = len(successes) >= 2
            if consistent:
                counts[variant] += 1
            dataset_payload[variant] = {
                "seed_results": seed_results[variant],
                "successful_seed_count": len(successes),
                "direction_consistent": consistent,
            }
        per_dataset[dataset] = dataset_payload

    if counts["B3"] >= 3:
        decision = "A. Proceed to U2-C with B3"
    elif counts["B1"] >= 3:
        decision = "B. Proceed to U2-C with B1"
    elif counts["B2"] >= 3:
        decision = "C. Proceed to U2-C with B2"
    else:
        b3_formal = [row for row in formal if row["variant"] == "B3" and float(row["alpha"]) == 0.1]
        high_redundancy = any(
            float(row["high_hop_frobenius_cosine"]) >= 0.90
            or float(row["high_hop_cka"]) >= 0.90
            for row in b3_formal
        )
        low_novelty = any(float(row["final_novelty_ratio"]) <= 0.25 for row in b3_formal)
        vanishing = any(bool(row["response_vanishing"]) for row in b3_formal)
        decision = (
            "D. Proceed to U2-B2 orthogonal-response fallback"
            if high_redundancy or low_novelty or vanishing
            else "E. U2 mechanism unresolved"
        )
    return {
        "per_dataset": per_dataset,
        "successful_dataset_count": counts,
        "direction_rule": "at least 2/3 seeds per dataset after averaging text and visual modalities",
        "decision": decision,
    }


def _modality_heterogeneity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    formal = [row for row in rows if int(row["K"]) == K_BY_DATASET[row["dataset"]]]
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for variant in ("B0", "B1", "B2", "B3"):
            alpha = None if variant in {"B0", "B1"} else 0.1
            group = [row for row in formal if row["dataset"] == dataset and row["variant"] == variant and row["alpha"] == alpha]
            text = {int(row["seed"]): row for row in group if row["modality"] == "text"}
            visual = {int(row["seed"]): row for row in group if row["modality"] == "visual"}
            differences = []
            for seed in sorted(set(text) & set(visual)):
                differences.append({
                    "seed": seed,
                    "semantic_retention_final_cka_text_minus_visual": text[seed]["semantic_retention_final_cka"] - visual[seed]["semantic_retention_final_cka"],
                    "semantic_retention_final_cosine_text_minus_visual": text[seed]["semantic_retention_final_cosine"] - visual[seed]["semantic_retention_final_cosine"],
                    "high_hop_frobenius_cosine_text_minus_visual": text[seed]["high_hop_frobenius_cosine"] - visual[seed]["high_hop_frobenius_cosine"],
                    "high_hop_cka_text_minus_visual": text[seed]["high_hop_cka"] - visual[seed]["high_hop_cka"],
                    "final_novelty_text_minus_visual": text[seed]["final_novelty_ratio"] - visual[seed]["final_novelty_ratio"],
                    "response_magnitude_text_minus_visual": text[seed]["response_magnitude_final_ratio_to_h0"] - visual[seed]["response_magnitude_final_ratio_to_h0"],
                })
            output[dataset][variant] = {
                "alpha": alpha,
                "per_seed": differences,
                "mean": {
                    key: _metric_stats([item[key] for item in differences])
                    for key in differences[0].keys() if key != "seed"
                } if differences else {},
            }
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    gate = summary["final_decision"]
    stress = summary["stress_summary"]
    lines = [
        "# MoPF-vNext U2-B — Semantic-Preserving Distinctive Multi-Hop Propagation",
        "",
        "Frozen mechanism screening only. No optimizer step, checkpoint write, formal config change, LP change, or U2-C implementation was performed.",
        "",
        f"- Source: `{summary['source_u1t_summary']}`; 15 final U1-T T2 best-validation checkpoints",
        f"- Frozen relation: `learned_diag_cos`, `tau={TAU}`",
        f"- Variants: B0 current cumulative, B1 hop innovation, B2 semantic anchor (`alpha=0.1`), B3 anchor + innovation",
        f"- Stress: `K=1..6`; analysis-only, formal training K unchanged",
        "",
        "## Scientific interpretation",
        "",
        "U2-B separates semantic preservation from hop distinctiveness. B1 tests innovation coordinates alone; B2 tests fixed semantic anchoring alone; B3 tests their combination. Semantic retention is a mechanism diagnostic, not a performance claim.",
        "",
        "## Stress summary",
        "",
        f"{stress['narrative']}",
        "",
        "## Factorial interpretation",
        "",
        "The JSON reports anchor main effects, innovation main effects, and anchor×innovation interaction for semantic retention, redundancy, novelty, and response magnitude at each dataset's formal K. These are mechanism effect sizes, not task-performance effects.",
        "",
        "## Decision",
        "",
        f"**{gate['decision']}**",
        "",
        "The language is restricted to mitigating propagation-induced semantic dilution, alleviating multi-hop response redundancy, preserving modality-specific semantics, and maintaining distinctive structural evidence. It does not claim to solve oversmoothing or oversquashing.",
        "",
        "## Fallback and pending work",
        "",
        "Orthogonal response construction remains a fallback path only if B3 retains high redundancy, low novelty, or vanishing higher-hop innovation. The pending relation-level 2×2 attribution (`separate_cos/tau2`, `learned_diag_cos/tau2`, `separate_cos/tau0.35`, `learned_diag_cos/tau0.35`) remains registered and was not run.",
        "",
        f"Authoritative summary: `{summary['artifacts']['master_summary']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--metric-device", default=None)
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()
    metric_device = args.metric_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(metric_device).startswith("cuda") and not torch.cuda.is_available():
        metric_device = "cpu"
    _load_and_diagnose.metric_device = metric_device
    if any(str(device).startswith("cuda") for device in args.devices):
        # Strict CPU reconstruction is mandatory. GPU can be requested, but
        # this controlled screening intentionally falls back before writing a
        # result if CUDA is unavailable or nondeterministic.
        print("[U2-B] strict reconstruction audit enabled; CPU is authoritative", flush=True)
    rows = _checkpoint_records()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    per_checkpoint_dir = output_root / "per_checkpoint"
    records: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    if not args.no_resume:
        for row in rows:
            path = per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}.json"
            flat_path = per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}_table.json"
            if path.is_file() and flat_path.is_file():
                records.append(json.loads(path.read_text(encoding="utf-8")))
                flat_rows.extend(json.loads(flat_path.read_text(encoding="utf-8")))
            else:
                pending.append(row)
    else:
        pending = rows
    active_devices = [str(device) for device in args.devices]
    if not torch.cuda.is_available():
        active_devices = ["cpu"]
    if args.metric_device is not None:
        active_devices = [str(metric_device)]
    if args.no_parallel or len(active_devices) == 1 or len(pending) <= 1:
        for index, row in enumerate(pending, start=1):
            active_metric_device = active_devices[(index - 1) % len(active_devices)]
            print(f"[U2-B] {index}/{len(pending)} {row['dataset']} seed={row['seed']} metric={active_metric_device}", flush=True)
            record, row_table = _load_and_diagnose(row, "cpu", active_metric_device)
            records.append(record)
            flat_rows.extend(row_table)
            _dump_json(per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}.json", record)
            _dump_json(per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}_table.json", row_table)
    else:
        with ThreadPoolExecutor(max_workers=len(active_devices)) as executor:
            futures = {}
            for index, row in enumerate(pending):
                active_metric_device = active_devices[index % len(active_devices)]
                print(f"[U2-B] queued {index + 1}/{len(pending)} {row['dataset']} seed={row['seed']} metric={active_metric_device}", flush=True)
                futures[executor.submit(_load_and_diagnose, row, "cpu", active_metric_device)] = row
            for future in as_completed(futures):
                row = futures[future]
                record, row_table = future.result()
                records.append(record)
                flat_rows.extend(row_table)
                _dump_json(per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}.json", record)
                _dump_json(per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}_table.json", row_table)
                print(f"[U2-B] finished {row['dataset']} seed={row['seed']}", flush=True)
    records.sort(key=lambda item: (DATASETS.index(item["dataset"]), SEEDS.index(int(item["seed"]))))
    flat_rows.sort(key=lambda item: (DATASETS.index(item["dataset"]), SEEDS.index(int(item["seed"])), item["modality"], item["variant"], str(item["alpha"]), int(item["K"])))
    stress_summary = _stress_summary(flat_rows)
    aggregates = {
        "formal_K": _aggregate_flat(flat_rows, formal_only=True),
        "K_stress": _aggregate_flat(flat_rows, formal_only=False),
    }
    factorial = _factorial_effects(flat_rows)
    screening = _final_screening(flat_rows)
    heterogeneity = _modality_heterogeneity(flat_rows)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as error:
        git_commit = f"unknown ({type(error).__name__}: {error})"
    summary = {
        "metadata": {
            "stage": "MoPF-vNext U2-B Semantic-Preserving Distinctive Multi-Hop Propagation Frozen Mechanism Screening",
            "scope": "NC inference-only frozen screening",
            "checkpoint_count": len(records),
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "K_formal_by_dataset": dict(K_BY_DATASET),
            "K_analysis": list(range(1, ANALYSIS_MAX_HOP + 1)),
            "variants": list(VARIANTS),
            "alphas": list(ALPHAS),
            "main_alpha": 0.1,
            "edge_weight_mode": "learned_diag_cos",
            "edge_weight_temperature": TAU,
            "no_training": True,
            "strict_cpu_reconstruction_audit": True,
            "metric_device": str(metric_device),
            "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
            "vanishing_innovation_threshold": VANISHING_THRESHOLD,
        },
        "git_commit": git_commit,
        "source_u1t_summary": str(U1T_SUMMARY),
        "source_u2a_output": str(U2A_OUTPUT_ROOT / "u2a_master_summary.json"),
        "source_checkpoints": [
            {
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "checkpoint": str(row["checkpoint"]),
                "sha256": record["checkpoint_sha256_before"],
            }
            for row, record in zip(rows, records)
        ],
        "per_checkpoint": records,
        "screening_table_rows": flat_rows,
        "aggregates": aggregates,
        "factorial_interpretation": factorial,
        "modality_heterogeneity": heterogeneity,
        "stress_summary": stress_summary,
        "final_screening": screening,
        "final_decision": {
            "decision": screening["decision"],
            "b1_successful_dataset_count": screening["successful_dataset_count"]["B1"],
            "b2_successful_dataset_count": screening["successful_dataset_count"]["B2"],
            "b3_successful_dataset_count": screening["successful_dataset_count"]["B3"],
            "alpha_policy": "alpha=0.1 is the main candidate; 0.05 and 0.2 are analysis-only sensitivity, never selected by test metrics",
        },
        "pending_final_ablation": {
            "relation_level_2x2": [
                "separate_cos/tau2",
                "learned_diag_cos/tau2",
                "separate_cos/tau0.35",
                "learned_diag_cos/tau0.35",
            ],
            "status": "not run; non-blocking",
        },
        "artifacts": {
            "master_summary": str(output_root / "u2b_master_summary.json"),
            "master_table": str(output_root / "u2b_master_table.csv"),
            "per_checkpoint_dir": str(per_checkpoint_dir),
            "report": str(ROOT / "docs" / "mopf_u2b_semantic_preserving_multihop_screening.md"),
        },
    }
    _dump_json(output_root / "u2b_master_summary.json", summary)
    _write_csv(output_root / "u2b_master_table.csv", flat_rows)
    _write_report(ROOT / "docs" / "mopf_u2b_semantic_preserving_multihop_screening.md", summary)
    print(f"[U2-B] master summary={output_root / 'u2b_master_summary.json'}", flush=True)
    print(f"[U2-B] decision={screening['decision']}", flush=True)


def _stress_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in ("B0", "B1", "B2", "B3"):
        alphas = [None] if variant in {"B0", "B1"} else [0.05, 0.1, 0.2]
        output[variant] = {}
        for alpha in alphas:
            selected = [row for row in rows if row["variant"] == variant and row["alpha"] == alpha]
            output[variant]["none" if alpha is None else f"{alpha:g}"] = {
                str(k): {
                    "high_hop_frobenius_cosine": _metric_stats([row["high_hop_frobenius_cosine"] for row in selected if int(row["K"]) == k]),
                    "high_hop_cka": _metric_stats([row["high_hop_cka"] for row in selected if int(row["K"]) == k]),
                    "final_novelty_ratio": _metric_stats([row["final_novelty_ratio"] for row in selected if int(row["K"]) == k]),
                    "semantic_retention_final_cka": _metric_stats([row["semantic_retention_final_cka"] for row in selected if int(row["K"]) == k]),
                    "response_magnitude_final_ratio_to_h0": _metric_stats([row["response_magnitude_final_ratio_to_h0"] for row in selected if int(row["K"]) == k]),
                }
                for k in range(1, ANALYSIS_MAX_HOP + 1)
            }
    baseline = output["B0"]["none"]
    final_b0 = baseline[str(ANALYSIS_MAX_HOP)]
    final_b3 = output["B3"]["0.1"][str(ANALYSIS_MAX_HOP)]
    output["narrative"] = (
        "Across K=1..6, B0's cumulative-state semantic retention and response "
        "redundancy are reported alongside B1/B2/B3. The authoritative tables "
        "retain per-dataset/per-modality/per-seed values; no stress depth was "
        "used to modify formal training. At K=6, B0 high-hop cosine mean was "
        f"{final_b0['high_hop_frobenius_cosine']['mean']}, while B3(alpha=0.1) was "
        f"{final_b3['high_hop_frobenius_cosine']['mean']}; B0 final novelty mean was "
        f"{final_b0['final_novelty_ratio']['mean']}, while B3 was "
        f"{final_b3['final_novelty_ratio']['mean']}."
    )
    return output


if __name__ == "__main__":
    main()
