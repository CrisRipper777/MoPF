"""Run the frozen MoPF-vNext U2-A monomial response-bank diagnosis.

The script is inference-only.  It consumes only the 15 final U1-T Phase-B
T2 best-validation checkpoints, reconstructs the frozen response banks, and
writes analysis artifacts under ``outputs/u2a_monomial_response_diagnosis``.
It never calls an optimizer, saves a checkpoint, or edits the formal model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
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
    modality_response_diagnostics,
    spectral_range_audit,
    sparse_symmetry_audit,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from scripts.run_mopf_u1_semantic_conductance import (  # noqa: E402
    DATASETS,
    K_BY_DATASET,
    MAGB_DATASETS,
    SEEDS,
    _fixed_split_override,
)


OUTPUT_ROOT = ROOT / "outputs" / "u2a_monomial_response_diagnosis"
U1T_SUMMARY = ROOT / "outputs" / "u1t_semantic_conductance_calibration" / "u1t_master_summary.json"
TAU = 0.35
STRONG_THRESHOLDS = {
    "high_order_frobenius_cosine_gte": 0.90,
    "high_order_cka_gte": 0.90,
    "final_novelty_ratio_lte": 0.25,
    "normalized_effective_rank_lte": 0.75,
    "gram_condition_number_gte": 100.0,
}


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


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
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


def _same_support(first: torch.Tensor, second: torch.Tensor) -> bool:
    first_cpu = first.detach().cpu()
    second_cpu = second.detach().cpu()
    if first_cpu.size(1) != second_cpu.size(1):
        return False
    first_keys = (first_cpu[0].long() * (int(first_cpu.max().item()) + 1) + first_cpu[1].long()).sort().values
    second_keys = (second_cpu[0].long() * (int(second_cpu.max().item()) + 1) + second_cpu[1].long()).sort().values
    return bool(torch.equal(first_keys, second_keys))


def _checkpoint_records() -> list[dict[str, Any]]:
    if not U1T_SUMMARY.is_file():
        raise FileNotFoundError(f"Required U1-T summary is missing: {U1T_SUMMARY}")
    summary = json.loads(U1T_SUMMARY.read_text(encoding="utf-8"))
    final = summary.get("final_relation_temperature_decision", {})
    if final.get("selected_label") != "T2" or float(final.get("selected_tau")) != TAU:
        raise RuntimeError(f"U1-T final selection is not T2/tau={TAU}: {final}")
    rows = [row for row in summary.get("phase_b", {}).get("per_run", []) if row.get("label") == "T2"]
    if len(rows) != len(DATASETS) * len(SEEDS):
        raise RuntimeError(f"Expected 15 final U1-T T2 checkpoints, found {len(rows)}")
    result = []
    for row in rows:
        path = Path(row["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing required U1-T T2 checkpoint: {path}")
        result.append({"dataset": str(row["dataset"]), "seed": int(row["seed"]), "checkpoint": path})
    return sorted(result, key=lambda item: (DATASETS.index(item["dataset"]), SEEDS.index(item["seed"])))


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


def _diff_stats(first: list[float | None], second: list[float | None]) -> dict[str, Any]:
    return _metric_stats([
        None if a is None or b is None else float(a) - float(b)
        for a, b in zip(first, second)
    ])


def _last_or_none(values: list[float]) -> float | None:
    return None if not values else float(values[-1])


def _text_visual_difference(record: dict[str, Any]) -> dict[str, Any]:
    text = record["modalities"]["text"]
    visual = record["modalities"]["visual"]

    def difference(first: float | None, second: float | None) -> float | None:
        if first is None or second is None:
            return None
        return float(first) - float(second)

    def difference_by_order(first: list[float], second: list[float]) -> list[float | None]:
        return [difference(a, b) for a, b in zip(first, second)]

    return {
        "pairwise_frobenius_cosine_matrix": [
            [difference(a, b) for a, b in zip(text["pairwise_frobenius_cosine"]["matrix"][i], visual["pairwise_frobenius_cosine"]["matrix"][i])]
            for i in range(len(text["pairwise_frobenius_cosine"]["matrix"]))
        ],
        "linear_cka_matrix": [
            [difference(a, b) for a, b in zip(text["linear_cka"]["matrix"][i], visual["linear_cka"]["matrix"][i])]
            for i in range(len(text["linear_cka"]["matrix"]))
        ],
        "high_order_frobenius_cosine": difference(text["pairwise_frobenius_cosine"]["high_order"], visual["pairwise_frobenius_cosine"]["high_order"]),
        "high_order_cka": difference(text["linear_cka"]["high_order"], visual["linear_cka"]["high_order"]),
        "gram_condition_number": difference(text["gram_spectrum"]["condition_number"], visual["gram_spectrum"]["condition_number"]),
        "normalized_effective_rank": difference(text["gram_spectrum"]["normalized_effective_rank"], visual["gram_spectrum"]["normalized_effective_rank"]),
        "novelty_ratio_by_order": difference_by_order(
            text["incremental_structural_novelty"]["novelty_ratio"],
            visual["incremental_structural_novelty"]["novelty_ratio"],
        ),
        "norm_ratio_by_order": difference_by_order(
            text["response_evolution"]["norm_ratio_to_order_0"],
            visual["response_evolution"]["norm_ratio_to_order_0"],
        ),
        "node_variance_ratio_by_order": difference_by_order(
            text["response_evolution"]["node_feature_variance_ratio_to_order_0"],
            visual["response_evolution"]["node_feature_variance_ratio_to_order_0"],
        ),
        "structural_variation_ratio_by_order": difference_by_order(
            text["response_evolution"]["structural_variation_ratio_to_order_0"],
            visual["response_evolution"]["structural_variation_ratio_to_order_0"],
        ),
        "operator_asymmetry_ratio": difference(text["operator_audit"]["asymmetry_ratio"], visual["operator_audit"]["asymmetry_ratio"]),
        "spectral_lambda_min": difference(text["spectral_audit"].get("lambda_min"), visual["spectral_audit"].get("lambda_min")),
        "spectral_lambda_max": difference(text["spectral_audit"].get("lambda_max"), visual["spectral_audit"].get("lambda_max")),
    }


def _classification_for_record(payload: dict[str, Any]) -> dict[str, Any]:
    k = len(payload["incremental_structural_novelty"]["novelty_ratio"]) - 1
    high_cos = payload["pairwise_frobenius_cosine"]["high_order"]
    high_cka = payload["linear_cka"]["high_order"]
    novelty = _last_or_none(payload["incremental_structural_novelty"]["novelty_ratio"])
    spectrum = payload["gram_spectrum"]
    normalized_rank = spectrum["normalized_effective_rank"]
    condition = spectrum["condition_number"]
    criteria = {
        "high_order_frobenius_cosine": bool(high_cos is not None and high_cos >= STRONG_THRESHOLDS["high_order_frobenius_cosine_gte"]),
        "high_order_cka": bool(high_cka is not None and high_cka >= STRONG_THRESHOLDS["high_order_cka_gte"]),
        "final_novelty_ratio": bool(novelty is not None and novelty <= STRONG_THRESHOLDS["final_novelty_ratio_lte"]),
        "normalized_effective_rank": bool(normalized_rank is not None and normalized_rank <= STRONG_THRESHOLDS["normalized_effective_rank_lte"]),
        "gram_condition_number": bool(condition is not None and condition >= STRONG_THRESHOLDS["gram_condition_number_gte"]),
    }
    count = sum(criteria.values())
    return {
        "K": k,
        "criteria": criteria,
        "criteria_count": int(count),
        "strong_seed_flag": bool(count >= 2),
        "screening_thresholds_are_not_theorems": True,
    }


def _aggregate_metric_payload(rows: list[dict[str, Any]], seeds: list[int] | None = None) -> dict[str, Any]:
    if not rows:
        return {}
    max_k = max(len(row["incremental_structural_novelty"]["novelty_ratio"]) - 1 for row in rows)
    adjacent_cosine: dict[str, Any] = {}
    adjacent_cka: dict[str, Any] = {}
    novelty: dict[str, Any] = {}
    norm_ratio: dict[str, Any] = {}
    variance_ratio: dict[str, Any] = {}
    structural_variation_ratio: dict[str, Any] = {}
    for order in range(max_k):
        key = f"{order}_{order + 1}"
        adjacent_cosine[key] = _metric_stats([
            row["pairwise_frobenius_cosine"]["adjacent"].get(key) for row in rows
        ])
        adjacent_cka[key] = _metric_stats([
            row["linear_cka"]["adjacent"].get(key) for row in rows
        ])
    for order in range(max_k + 1):
        novelty[str(order)] = _metric_stats([
            row["incremental_structural_novelty"]["novelty_ratio"][order] for row in rows
        ])
        norm_ratio[str(order)] = _metric_stats([
            row["response_evolution"]["norm_ratio_to_order_0"][order] for row in rows
        ])
        variance_ratio[str(order)] = _metric_stats([
            row["response_evolution"]["node_feature_variance_ratio_to_order_0"][order] for row in rows
        ])
        structural_variation_ratio[str(order)] = _metric_stats([
            row["response_evolution"]["structural_variation_ratio_to_order_0"][order] for row in rows
        ])
    return {
        "checkpoint_count": len(rows),
        "seeds": sorted(int(seed) for seed in (seeds or [])),
        "adjacent_frobenius_cosine": adjacent_cosine,
        "high_order_frobenius_cosine": _metric_stats([row["pairwise_frobenius_cosine"]["high_order"] for row in rows]),
        "adjacent_cka": adjacent_cka,
        "high_order_cka": _metric_stats([row["linear_cka"]["high_order"] for row in rows]),
        "gram_condition_number": _metric_stats([row["gram_spectrum"]["condition_number"] for row in rows]),
        "effective_rank": _metric_stats([row["gram_spectrum"]["effective_rank"] for row in rows]),
        "normalized_effective_rank": _metric_stats([row["gram_spectrum"]["normalized_effective_rank"] for row in rows]),
        "novelty_ratio_by_order": novelty,
        "response_norm_ratio_by_order": norm_ratio,
        "node_variance_ratio_by_order": variance_ratio,
        "structural_variation_ratio_by_order": structural_variation_ratio,
        "operator_asymmetry_ratio": _metric_stats([row["operator_audit"]["asymmetry_ratio"] for row in rows]),
        "spectral_lambda_min": _metric_stats([row["spectral_audit"].get("lambda_min") for row in rows]),
        "spectral_lambda_max": _metric_stats([row["spectral_audit"].get("lambda_max") for row in rows]),
        "spectral_max_abs_eigenvalue": _metric_stats([row["spectral_audit"].get("max_abs_eigenvalue") for row in rows]),
        "spectral_statuses": sorted({row["spectral_audit"].get("status") for row in rows}),
    }


def _aggregate_group(rows: list[dict[str, Any]], dataset: str, modality: str) -> dict[str, Any]:
    payloads = [row["modalities"][modality] for row in rows]
    seed_classifications = [
        {"seed": int(row["seed"]), **_classification_for_record(payload)}
        for row, payload in zip(rows, payloads)
    ]
    strong_count = sum(item["strong_seed_flag"] for item in seed_classifications)
    any_count = sum(item["criteria_count"] > 0 for item in seed_classifications)
    if strong_count >= 2:
        classification = "Strong Monomial Redundancy Evidence"
    elif any_count > 0:
        classification = "Moderate Evidence"
    else:
        classification = "Weak/No Evidence"
    return {
        "dataset": dataset,
        "modality": modality,
        "metrics": _aggregate_metric_payload(
            payloads, seeds=[int(row["seed"]) for row in rows]
        ),
        "seed_screening": seed_classifications,
        "strong_seed_count": int(strong_count),
        "strong_seed_consistency": f"{strong_count}/3",
        "classification": classification,
    }


def _combined_screening(dataset: str, rows: list[dict[str, Any]], per_modality: dict[str, Any]) -> dict[str, Any]:
    seed_rows = []
    for row in rows:
        text = row["modalities"]["text"]
        visual = row["modalities"]["visual"]
        combined = dict(text)
        combined["pairwise_frobenius_cosine"] = {
            "high_order": float(np.mean([text["pairwise_frobenius_cosine"]["high_order"], visual["pairwise_frobenius_cosine"]["high_order"]]))
        }
        combined["linear_cka"] = {
            "high_order": float(np.mean([text["linear_cka"]["high_order"], visual["linear_cka"]["high_order"]]))
        }
        combined["gram_spectrum"] = {
            "condition_number": float(np.mean([
                text["gram_spectrum"]["condition_number"], visual["gram_spectrum"]["condition_number"]
            ])),
            "normalized_effective_rank": float(np.mean([
                text["gram_spectrum"]["normalized_effective_rank"], visual["gram_spectrum"]["normalized_effective_rank"]
            ])),
        }
        novelty_text = text["incremental_structural_novelty"]["novelty_ratio"][-1]
        novelty_visual = visual["incremental_structural_novelty"]["novelty_ratio"][-1]
        combined["incremental_structural_novelty"] = {"novelty_ratio": [0.0, float(np.mean([novelty_text, novelty_visual]))]}
        seed_rows.append({"seed": int(row["seed"]), **_classification_for_record(combined)})
    strong_count = sum(item["strong_seed_flag"] for item in seed_rows)
    any_count = sum(item["criteria_count"] > 0 for item in seed_rows)
    classification = (
        "Strong Monomial Redundancy Evidence" if strong_count >= 2
        else "Moderate Evidence" if any_count > 0
        else "Weak/No Evidence"
    )
    return {
        "dataset": dataset,
        "modality": "text+visual aggregate",
        "seed_screening": seed_rows,
        "strong_seed_count": int(strong_count),
        "strong_seed_consistency": f"{strong_count}/3",
        "classification": classification,
    }


def _build_aggregates(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_dataset: dict[str, Any] = {}
    strong_by_modality = {"text": 0, "visual": 0}
    combined_moderate_or_strong = 0
    for dataset in DATASETS:
        dataset_rows = [row for row in records if row["dataset"] == dataset]
        modality_groups = {
            modality: _aggregate_group(dataset_rows, dataset, modality)
            for modality in ("text", "visual")
        }
        for modality, group in modality_groups.items():
            if group["classification"] == "Strong Monomial Redundancy Evidence":
                strong_by_modality[modality] += 1
        combined = _combined_screening(dataset, dataset_rows, modality_groups)
        if combined["classification"] in {"Strong Monomial Redundancy Evidence", "Moderate Evidence"}:
            combined_moderate_or_strong += 1
        text_rows = [row["modalities"]["text"] for row in dataset_rows]
        visual_rows = [row["modalities"]["visual"] for row in dataset_rows]
        text_minus_visual = {
            "high_order_frobenius_cosine": _diff_stats(
                [row["pairwise_frobenius_cosine"]["high_order"] for row in text_rows],
                [row["pairwise_frobenius_cosine"]["high_order"] for row in visual_rows],
            ),
            "high_order_cka": _diff_stats(
                [row["linear_cka"]["high_order"] for row in text_rows],
                [row["linear_cka"]["high_order"] for row in visual_rows],
            ),
            "normalized_effective_rank": _diff_stats(
                [row["gram_spectrum"]["normalized_effective_rank"] for row in text_rows],
                [row["gram_spectrum"]["normalized_effective_rank"] for row in visual_rows],
            ),
            "gram_condition_number": _diff_stats(
                [row["gram_spectrum"]["condition_number"] for row in text_rows],
                [row["gram_spectrum"]["condition_number"] for row in visual_rows],
            ),
            "operator_asymmetry_ratio": _diff_stats(
                [row["operator_audit"]["asymmetry_ratio"] for row in text_rows],
                [row["operator_audit"]["asymmetry_ratio"] for row in visual_rows],
            ),
        }
        per_dataset[dataset] = {
            "per_modality": modality_groups,
            "text_minus_visual": text_minus_visual,
            "text_visual_aggregate": combined,
        }
    per_modality = {}
    for modality in ("text", "visual"):
        per_modality[modality] = {
            "per_dataset": {
                dataset: per_dataset[dataset]["per_modality"][modality]
                for dataset in DATASETS
            },
            "strong_dataset_count": int(strong_by_modality[modality]),
        }
    return {
        "per_dataset": per_dataset,
        "per_modality": per_modality,
        "strong_dataset_count_by_modality": strong_by_modality,
        "combined_moderate_or_strong_dataset_count": int(combined_moderate_or_strong),
    }


def _spectral_prerequisite(records: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [row["modalities"][modality] for row in records for modality in ("text", "visual")]
    operator_audits = [row["operator_audit"] for row in audits]
    spectral_audits = [row["spectral_audit"] for row in audits]
    status_counts: dict[str, int] = {}
    for item in spectral_audits:
        status = str(item.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    materially_asymmetric = sum(item["classification"] == "materially_asymmetric" for item in operator_audits)
    unavailable = sum(item.get("lambda_max") is None for item in spectral_audits)
    out_of_range = [
        float(item["max_abs_eigenvalue"])
        for item in spectral_audits
        if item.get("max_abs_eigenvalue") is not None and float(item["max_abs_eigenvalue"]) > 1.0 + 1e-3
    ]
    resolved = materially_asymmetric == 0 and unavailable == 0 and not out_of_range
    return {
        "operator_audit_count": len(operator_audits),
        "operator_classification_counts": {
            label: sum(item["classification"] == label for item in operator_audits)
            for label in ("near_symmetric", "approximately_symmetric", "materially_asymmetric")
        },
        "spectral_status_counts": status_counts,
        "materially_asymmetric_count": int(materially_asymmetric),
        "spectral_unavailable_count": int(unavailable),
        "spectral_out_of_range_values": out_of_range,
        "resolved_for_clean_orthogonal_polynomial_interpretation": bool(resolved),
        "interpretation": (
            "supports_clean_symmetric spectral interpretation"
            if resolved else
            "operator asymmetry or spectral-range uncertainty requires caution"
        ),
    }


def _gate_decision(aggregates: dict[str, Any], prerequisite: dict[str, Any]) -> dict[str, Any]:
    strong_counts = aggregates["strong_dataset_count_by_modality"]
    if not prerequisite["resolved_for_clean_orthogonal_polynomial_interpretation"]:
        decision = "D. Spectral prerequisite unresolved: operator asymmetry/range prevents a clean orthogonal-polynomial interpretation"
    elif max(strong_counts.values()) >= 3:
        decision = "A. Proceed to U2-B: monomial response redundancy/conditioning is empirically supported"
    elif aggregates["combined_moderate_or_strong_dataset_count"] >= 3:
        decision = "B. Proceed to U2-B cautiously: evidence is dataset/modality conditional"
    else:
        decision = "C. Do not replace monomial bank: no meaningful redundancy/conditioning problem was established"
    return {
        "decision": decision,
        "strong_dataset_count_by_modality": strong_counts,
        "combined_moderate_or_strong_dataset_count": aggregates["combined_moderate_or_strong_dataset_count"],
        "screening_thresholds": STRONG_THRESHOLDS,
        "rationale": (
            "The gate uses continuous response diagnostics and the prescribed screening thresholds; "
            "it does not treat any one metric as a theorem. Monomial and complete degree-K polynomial "
            "bases span the same non-degenerate polynomial subspace, so any future U2-B study concerns "
            "coordinate quality, conditioning, separation, and optimisation suitability rather than larger span."
        ),
    }


def _checkpoint_diagnosis(
    row: dict[str, Any],
    device: str,
    *,
    allow_cpu_fallback: bool = True,
    execution_note: str | None = None,
) -> dict[str, Any]:
    dataset = row["dataset"]
    seed = int(row["seed"])
    checkpoint_path = Path(row["checkpoint"])
    checkpoint_before = _sha256(checkpoint_path)
    cfg = _compose_cfg(dataset, seed, device)
    data = load_mag_data(cfg, "nc", seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu_fallback:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return _checkpoint_diagnosis(
            row,
            "cpu",
            allow_cpu_fallback=False,
            execution_note="CUDA unavailable in runtime; strict-equivalence CPU fallback",
        )
    try:
        model = build_model(cfg, checkpoint["data_info"]).to(device)
    except RuntimeError as error:
        if not str(device).startswith("cuda") or not allow_cpu_fallback:
            raise
        return _checkpoint_diagnosis(
            row,
            "cpu",
            allow_cpu_fallback=False,
            execution_note=f"CUDA model initialization failed ({type(error).__name__}); strict-equivalence CPU fallback",
        )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    parameter_names_before = sorted(model.state_dict().keys())
    state_before = _state_digest(model)
    data_x = data.x.to(device)
    data_edge_index = data.edge_index.to(device)
    analysis = extract_analysis_response_bank(model, data_x, data_edge_index)
    formal = capture_formal_forward_banks(model, data_x, data_edge_index)
    equivalence = bank_equivalence(analysis["banks"], formal)
    formal_operators = formal["operators"]
    operator_equivalence = {
        "text_index_equal": bool(torch.equal(analysis["norm_text_index"], formal_operators[0][0])),
        "visual_index_equal": bool(torch.equal(analysis["norm_visual_index"], formal_operators[1][0])),
        "text_weight_max_abs_difference": float((analysis["norm_text_weight"] - formal_operators[0][1]).abs().max().item()),
        "visual_weight_max_abs_difference": float((analysis["norm_visual_weight"] - formal_operators[1][1]).abs().max().item()),
    }
    if not equivalence["passed"]:
        if str(device).startswith("cuda") and allow_cpu_fallback:
            del model, data, checkpoint, analysis, formal
            torch.cuda.empty_cache()
            return _checkpoint_diagnosis(
                row,
                "cpu",
                allow_cpu_fallback=False,
                execution_note=(
                    "GPU formal-bank repeat-call drift exceeded the 1e-6 gate; "
                    "strict-equivalence CPU fallback"
                ),
            )
        raise RuntimeError(f"Response-bank equivalence failed for {dataset} seed={seed}: {equivalence}")
    node_subset = deterministic_node_subset(
        int(data.num_nodes), max_nodes=MAX_CKA_NODES, seed=DEFAULT_NODE_SAMPLE_SEED
    )
    modality_payloads: dict[str, Any] = {}
    for modality, index_key, weight_key in (
        ("text", "norm_text_index", "norm_text_weight"),
        ("visual", "norm_visual_index", "norm_visual_weight"),
    ):
        norm_index = analysis[index_key]
        norm_weight = analysis[weight_key]
        operator_audit = sparse_symmetry_audit(norm_index, norm_weight, int(data.num_nodes))
        spectral_audit = spectral_range_audit(
            norm_index, norm_weight, int(data.num_nodes), operator_audit,
            seed=DEFAULT_NODE_SAMPLE_SEED + seed,
        )
        modality_payloads[modality] = modality_response_diagnostics(
            analysis["banks"][modality],
            norm_index,
            norm_weight,
            node_subset=node_subset,
            operator_audit=operator_audit,
            spectral_audit=spectral_audit,
        )
    edge_weights = analysis["edges"]
    support_equal = bool(
        torch.equal(analysis["norm_text_index"], analysis["norm_visual_index"])
    )
    edge_support_audit = {
        "original_edge_count": int(data_edge_index.size(1)),
        "text_conductance_edge_count": int(edge_weights["w_t"].numel()),
        "visual_conductance_edge_count": int(edge_weights["w_v"].numel()),
        "conductance_support_same_as_input": bool(
            edge_weights["w_t"].numel() == data_edge_index.size(1)
            and edge_weights["w_v"].numel() == data_edge_index.size(1)
        ),
        "normalized_operator_support_equal_text_visual": support_equal,
        "self_loops_added_only_by_frozen_gcn_norm": True,
    }
    state_after = _state_digest(model)
    checkpoint_after = _sha256(checkpoint_path)
    if state_before != state_after or checkpoint_before != checkpoint_after:
        raise RuntimeError(f"State mutation detected for {dataset} seed={seed}")
    record = {
        "dataset": dataset,
        "seed": seed,
        "K": K_BY_DATASET[dataset],
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
            "edge_weight_min": float(model.edge_weight_min),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "device": str(device),
        },
        "execution_note": execution_note,
        "response_bank_equivalence": equivalence,
        "operator_equivalence": operator_equivalence,
        "edge_support_audit": edge_support_audit,
        "node_subset": {
            "seed": DEFAULT_NODE_SAMPLE_SEED,
            "max_nodes": MAX_CKA_NODES,
            "num_nodes_used_for_cka": int(node_subset.numel()),
            "all_orders_share_identical_subset": True,
        },
        "modalities": modality_payloads,
    }
    record["text_minus_visual"] = _text_visual_difference(record)
    del model, data, checkpoint, analysis, formal
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return record


def _flat_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for modality in ("text", "visual"):
            payload = record["modalities"][modality]
            rows.append({
                "dataset": record["dataset"],
                "seed": record["seed"],
                "K": record["K"],
                "modality": modality,
                "checkpoint": record["checkpoint"],
                "high_order_frobenius_cosine": payload["pairwise_frobenius_cosine"]["high_order"],
                "high_order_cka": payload["linear_cka"]["high_order"],
                "gram_condition_number": payload["gram_spectrum"]["condition_number"],
                "effective_rank": payload["gram_spectrum"]["effective_rank"],
                "normalized_effective_rank": payload["gram_spectrum"]["normalized_effective_rank"],
                "final_novelty_ratio": payload["incremental_structural_novelty"]["novelty_ratio"][-1],
                "final_norm_ratio": payload["response_evolution"]["norm_ratio_to_order_0"][-1],
                "final_node_variance_ratio": payload["response_evolution"]["node_feature_variance_ratio_to_order_0"][-1],
                "final_structural_variation_ratio": payload["response_evolution"]["structural_variation_ratio_to_order_0"][-1],
                "operator_asymmetry_ratio": payload["operator_audit"]["asymmetry_ratio"],
                "operator_classification": payload["operator_audit"]["classification"],
                "spectral_status": payload["spectral_audit"]["status"],
                "spectral_lambda_min": payload["spectral_audit"].get("lambda_min"),
                "spectral_lambda_max": payload["spectral_audit"].get("lambda_max"),
                "spectral_max_abs_eigenvalue": payload["spectral_audit"].get("max_abs_eigenvalue"),
            })
    return rows


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = _flat_rows(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    gate = summary["u2b_gate_decision"]
    prerequisite = summary["spectral_prerequisite"]
    aggregates = summary["aggregates"]
    lines = [
        "# MoPF-vNext U2-A — Monomial Structural Response Bank Diagnosis",
        "",
        "This is an inference-only diagnosis of the frozen U1-T Calibrated R1 relation module. No model was trained or fine-tuned, no checkpoint was written, and no propagation/filter/fusion/classifier code was changed.",
        "",
        f"- Analysis commit: `{summary['git_commit']}`",
        f"- Source U1-T summary: `{summary['source_u1t_summary']}`",
        f"- Source checkpoints: `{summary['metadata']['checkpoint_count']}` final T2 checkpoints, datasets `{', '.join(DATASETS)}`, seeds `{SEEDS}`",
        f"- Frozen relation: `edge_weight_mode=learned_diag_cos`, `tau={TAU}`",
        f"- CKA subset: deterministic seed `{DEFAULT_NODE_SAMPLE_SEED}`, maximum `{MAX_CKA_NODES}` nodes, identical across all orders",
        "",
        "## Scientific scope",
        "",
        "For each modality, the analysis extracts the exact raw response bank `R_0=H_0`, `R_k=P^k H_0` from the frozen projection, semantic conductance, and GCN-normalized operator. Frobenius cosine, feature-space linear CKA, normalized response Gram conditioning, incremental structural response novelty, norm/variance evolution, operator-native structural variation, sparse symmetry, and the conditional Dirichlet-energy diagnostic are reported.",
        "",
        "Monomial `{1, x, ..., x^K}` and any complete degree-K polynomial basis `{phi_0(x), ..., phi_K(x)}` span the same polynomial subspace under non-degenerate conditions. Therefore a future orthogonal-polynomial study concerns response-coordinate quality, redundancy, conditioning, structural-response separation, and optimization suitability—not larger degree-K expressivity or a larger receptive field.",
        "",
        "## Dataset-level screening",
        "",
        f"- Strong evidence datasets by modality: `{aggregates['strong_dataset_count_by_modality']}`",
        f"- Text+visual aggregate Moderate-or-Strong datasets: `{aggregates['combined_moderate_or_strong_dataset_count']}/5`",
        "- Strong/Moderate/Weak labels are screening classifications from the prescribed thresholds, not theoretical claims.",
        "",
        "## Operator prerequisite",
        "",
        f"- Classification counts: `{prerequisite['operator_classification_counts']}`",
        f"- Spectral statuses: `{prerequisite['spectral_status_counts']}`",
        f"- Interpretation: {prerequisite['interpretation']}",
        "",
        "## U2-B gate",
        "",
        f"**{gate['decision']}**",
        "",
        f"Rationale: {gate['rationale']}",
        "",
        "## Pending attribution ablation",
        "",
        "The final-paper pending 2x2 relation attribution remains registered and was not run in U2-A: A `separate_cos,tau=2`; B `learned_diag_cos,tau=2`; C `separate_cos,tau=0.35`; D `learned_diag_cos,tau=0.35`. C remains non-blocking for U2 but must not be forgotten.",
        "",
        "## Artifacts",
        "",
        f"- Authoritative JSON: `{summary['artifacts']['master_summary']}`",
        f"- Flat CSV: `{summary['artifacts']['master_table']}`",
        f"- Per-stage records: `{summary['artifacts']['per_checkpoint_dir']}`",
        "",
        "U2-A hard stop: no Jacobi, Chebyshev, U2-B implementation, U3/U4 work, auxiliary loss, LP, retraining, or formal propagation-bank changes were performed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    devices = [str(device) for device in args.devices]
    all_rows = _checkpoint_records()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    per_checkpoint_dir = output_root / "per_checkpoint"
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    if not args.no_resume:
        for row in all_rows:
            cached = per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}.json"
            if cached.is_file():
                results.append(json.loads(cached.read_text(encoding="utf-8")))
            else:
                pending.append(row)
    else:
        pending = all_rows
    if not pending:
        print(f"[U2-A] reusing {len(results)} completed per-checkpoint diagnostics", flush=True)
    if args.no_parallel or len(devices) == 1:
        for index, row in enumerate(pending, start=1):
            device = devices[(index - 1) % len(devices)]
            print(f"[U2-A] {index}/{len(pending)} {row['dataset']} seed={row['seed']} {device}", flush=True)
            result = _checkpoint_diagnosis(row, device)
            results.append(result)
            _dump_json(per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}.json", result)
    else:
        future_to_row = {}
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            for index, row in enumerate(pending):
                device = devices[index % len(devices)]
                print(f"[U2-A] queued {index + 1}/{len(pending)} {row['dataset']} seed={row['seed']} {device}", flush=True)
                future_to_row[executor.submit(_checkpoint_diagnosis, row, device)] = row
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                result = future.result()
                results.append(result)
                _dump_json(per_checkpoint_dir / f"{row['dataset']}_seed{row['seed']}.json", result)
                print(f"[U2-A] finished {row['dataset']} seed={row['seed']}", flush=True)
    results.sort(key=lambda item: (DATASETS.index(item["dataset"]), SEEDS.index(int(item["seed"]))))
    for result in results:
        if "text_minus_visual" not in result:
            result["text_minus_visual"] = _text_visual_difference(result)
        _dump_json(per_checkpoint_dir / f"{result['dataset']}_seed{result['seed']}.json", result)
    aggregates = _build_aggregates(results)
    prerequisite = _spectral_prerequisite(results)
    gate = _gate_decision(aggregates, prerequisite)
    summary = {
        "metadata": {
            "stage": "MoPF-vNext U2-A Monomial Structural Response Bank Diagnosis",
            "scope": "NC inference-only diagnostics",
            "checkpoint_count": len(results),
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "K_by_dataset": dict(K_BY_DATASET),
            "edge_weight_mode": "learned_diag_cos",
            "edge_weight_temperature": TAU,
            "response_definition": "R_0=H_0; R_k=P^k H_0 using the frozen formal recurrence",
            "no_training": True,
            "no_checkpoint_mutation": True,
            "no_response_normalization_before_metrics": True,
            "cka_implementation": "feature-space linear CKA; no N x N kernel matrix",
        },
        "git_commit": "unknown",
        "git_sync_attempt": "git fetch origin && git switch vnext && git pull --ff-only origin vnext",
        "git_sync_status": "failed: ssh could not resolve github.com (Temporary failure in name resolution)",
        "source_u1t_summary": str(U1T_SUMMARY),
        "source_checkpoints": [
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "checkpoint": str(row["checkpoint"]),
                "sha256": result["checkpoint_sha256_before"],
            }
            for row, result in zip(all_rows, results)
        ],
        "per_checkpoint": results,
        "aggregates": aggregates,
        "spectral_prerequisite": prerequisite,
        "evidence_classification": {
            "screening_thresholds": STRONG_THRESHOLDS,
            "strong_definition": "at least two of five criteria on at least 2/3 seeds for a dataset x modality",
            "moderate_definition": "one criterion or insufficient seed consistency",
            "weak_definition": "no stable screening signal",
        },
        "u2b_gate_decision": gate,
        "scientific_rationale": gate["rationale"],
        "artifacts": {
            "master_summary": str(output_root / "u2a_master_summary.json"),
            "master_table": str(output_root / "u2a_master_table.csv"),
            "per_checkpoint_dir": str(per_checkpoint_dir),
            "report": str(ROOT / "docs" / "mopf_u2a_monomial_response_diagnosis.md"),
        },
    }
    try:
        import subprocess

        summary["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception as error:
        summary["git_commit_error"] = f"{type(error).__name__}: {error}"
    _dump_json(output_root / "u2a_master_summary.json", summary)
    _write_csv(output_root / "u2a_master_table.csv", results)
    _write_report(ROOT / "docs" / "mopf_u2a_monomial_response_diagnosis.md", summary)
    print(f"[U2-A] master summary={output_root / 'u2a_master_summary.json'}", flush=True)
    print(f"[U2-A] gate={gate['decision']}", flush=True)


if __name__ == "__main__":
    main()
