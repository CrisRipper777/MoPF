#!/usr/bin/env python3
"""E0-C: node--modality multi-hop utilization heterogeneity.

This is an analysis-only export over the frozen U2-C C1 checkpoints used by
U3-A.  It deliberately loads only the model state and the raw graph/features
needed to obtain the predecessor model's contribution profiles; classifier
heads, labels, optimizer state, and evaluation metrics are not used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.u3a import EPS, contribution_profile  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.data.loaders import resolve_path  # noqa: E402
from src.models import build_model  # noqa: E402
from scripts.run_mopf_u3a_node_modality_preference_diagnosis import (  # noqa: E402
    DATASETS,
    SEEDS,
    _compose_c1_cfg,
    _sha256,
    _source_records,
)


DATASET_ORDER = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
FORMAL_K = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
}
REQUIRED_SEEDS = [42, 43, 44]
OUTPUT_ROOT = REPO_ROOT / "outputs/e0_empirical_motivation/multihop_utilization"
U3A_ROOT = REPO_ROOT / "outputs/u3a_node_modality_preference_diagnosis"
U3A_MASTER = U3A_ROOT / "u3a_master_summary.json"
PLOT_SEED = 20260914


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _model_state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _as_float32_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _finite_or_raise(name: str, array: np.ndarray) -> bool:
    finite = bool(np.isfinite(array).all())
    if not finite:
        bad = int((~np.isfinite(array)).sum())
        raise RuntimeError(f"Non-finite {name}: {bad} values")
    return finite


def _stats(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    _finite_or_raise(prefix or "statistic", values)
    return {
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}std": float(np.std(values)),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}q10": float(np.quantile(values, 0.10)),
        f"{prefix}q25": float(np.quantile(values, 0.25)),
        f"{prefix}q75": float(np.quantile(values, 0.75)),
        f"{prefix}q90": float(np.quantile(values, 0.90)),
    }


def _stats_with_width(values: np.ndarray, prefix: str) -> dict[str, float]:
    result = _stats(values, prefix)
    result[f"{prefix}iqr"] = result[f"{prefix}q75"] - result[f"{prefix}q25"]
    result[f"{prefix}q90_q10"] = result[f"{prefix}q90"] - result[f"{prefix}q10"]
    return result


def _validate_source_records() -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    """Resolve the exact U3-A source and cross-check its authoritative manifest."""

    records = _source_records()
    expected_keys = {(dataset, seed) for dataset in DATASET_ORDER for seed in REQUIRED_SEEDS}
    actual_keys = {(str(record["dataset"]), int(record["seed"])) for record in records}
    if actual_keys != expected_keys or len(records) != len(expected_keys):
        raise RuntimeError(
            "U3-A source does not contain exactly the required 5 x 3 C1 checkpoints: "
            f"{sorted(actual_keys)}"
        )

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (str(record["dataset"]), int(record["seed"]))
        checkpoint = Path(record["checkpoint"]).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing frozen source checkpoint: {checkpoint}")
        current_sha = _sha256(checkpoint)
        normalized = dict(record)
        normalized["dataset"] = key[0]
        normalized["seed"] = key[1]
        normalized["checkpoint"] = str(checkpoint)
        normalized["checkpoint_sha256_before"] = current_sha
        by_key[key] = normalized

    if not U3A_MASTER.exists():
        raise FileNotFoundError(f"Missing authoritative U3-A manifest: {U3A_MASTER}")
    master = json.loads(U3A_MASTER.read_text(encoding="utf-8"))
    u3a_sources = {
        (str(item["dataset"]), int(item["seed"])): item
        for item in master.get("source_checkpoints", [])
    }
    if set(u3a_sources) != expected_keys:
        raise RuntimeError("U3-A authoritative artifact source set is incomplete or changed")
    for key, record in by_key.items():
        item = u3a_sources[key]
        if str(Path(item["checkpoint"]).resolve()) != record["checkpoint"]:
            raise RuntimeError(f"U3-A checkpoint path mismatch for {key}")
        authoritative_sha = item.get("sha256", item.get("checkpoint_sha256"))
        if authoritative_sha != record["checkpoint_sha256_before"]:
            raise RuntimeError(f"U3-A checkpoint SHA mismatch for {key}")
    return records, by_key


def _authoritative_node_profiles_available() -> bool:
    """Check whether U3-A already has the required per-node tensor exports."""

    if not U3A_ROOT.exists():
        return False
    return any(U3A_ROOT.rglob("node_multihop_utilization.npz"))


def _validate_model_config(model: torch.nn.Module, dataset: str, cfg: Any) -> dict[str, Any]:
    checks = {
        "edge_weight_mode": getattr(model, "edge_weight_mode", None),
        "edge_weight_temperature": float(getattr(model, "edge_weight_temperature", float("nan"))),
        "multihop_state_mode": getattr(model, "multihop_state_mode", None),
        "multihop_response_mode": getattr(model, "multihop_response_mode", None),
        "multihop_anchor_alpha": float(getattr(model, "multihop_anchor_alpha", float("nan"))),
        "formal_K": int(getattr(model, "max_order", -1)),
        "use_transport_residual": bool(getattr(model, "use_transport_residual", True)),
    }
    expected = {
        "edge_weight_mode": "learned_diag_cos",
        "edge_weight_temperature": 0.35,
        "multihop_state_mode": "anchored",
        "multihop_response_mode": "cumulative",
        "multihop_anchor_alpha": 0.1,
        "formal_K": FORMAL_K[dataset],
        "use_transport_residual": False,
    }
    if checks["edge_weight_mode"] != expected["edge_weight_mode"]:
        raise RuntimeError(f"Unexpected edge_weight_mode for {dataset}: {checks}")
    if not np.isclose(checks["edge_weight_temperature"], expected["edge_weight_temperature"]):
        raise RuntimeError(f"Unexpected edge_weight_temperature for {dataset}: {checks}")
    if checks["multihop_state_mode"] != expected["multihop_state_mode"]:
        raise RuntimeError(f"Unexpected multihop_state_mode for {dataset}: {checks}")
    if checks["multihop_response_mode"] != expected["multihop_response_mode"]:
        raise RuntimeError(f"Unexpected multihop_response_mode for {dataset}: {checks}")
    if not np.isclose(checks["multihop_anchor_alpha"], expected["multihop_anchor_alpha"]):
        raise RuntimeError(f"Unexpected multihop_anchor_alpha for {dataset}: {checks}")
    if checks["formal_K"] != expected["formal_K"]:
        raise RuntimeError(f"Unexpected formal K for {dataset}: {checks}")
    if checks["use_transport_residual"] is not False:
        raise RuntimeError(f"TCPR/transport residual is enabled for {dataset}: {checks}")
    # Also retain the composed configuration value in the audit payload.
    checks["config_use_transport_residual"] = bool(
        getattr(getattr(cfg, "model", None), "use_transport_residual", True)
    )
    if checks["config_use_transport_residual"]:
        raise RuntimeError(f"Composed C1 config did not disable TCPR for {dataset}")
    return checks


def _js_divergence_exact_u3a(p_text: np.ndarray, p_visual: np.ndarray) -> np.ndarray:
    """Match src.analysis.u3a's Jensen--Shannon implementation exactly."""

    p_t = torch.from_numpy(np.asarray(p_text, dtype=np.float32))
    p_v = torch.from_numpy(np.asarray(p_visual, dtype=np.float32))
    midpoint = 0.5 * (p_t + p_v)
    js = 0.5 * (
        (p_t * torch.log((p_t + EPS) / midpoint + EPS)).sum(dim=1)
        + (p_v * torch.log((p_v + EPS) / midpoint + EPS)).sum(dim=1)
    )
    return js.numpy().astype(np.float32, copy=False)


def _summarize_profiles(
    dataset: str,
    seed: int,
    formal_k: int,
    node_id: np.ndarray,
    p_text: np.ndarray,
    p_visual: np.ndarray,
    r_text: np.ndarray,
    r_visual: np.ndarray,
    raw_text: np.ndarray,
    raw_visual: np.ndarray,
    entropy_text: np.ndarray,
    entropy_visual: np.ndarray,
    profile_sum_error_text: float,
    profile_sum_error_visual: float,
    all_zero_text: int,
    all_zero_visual: int,
    finite_status: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    _finite_or_raise("p_text", p_text)
    _finite_or_raise("p_visual", p_visual)
    _finite_or_raise("normalized_order_text", r_text)
    _finite_or_raise("normalized_order_visual", r_visual)
    _finite_or_raise("raw_order_text", raw_text)
    _finite_or_raise("raw_order_visual", raw_visual)
    _finite_or_raise("entropy_text", entropy_text)
    _finite_or_raise("entropy_visual", entropy_visual)
    order_count = formal_k + 1
    if p_text.shape != p_visual.shape or p_text.shape[1] != order_count:
        raise RuntimeError(f"Unexpected contribution profile shape for {dataset}/{seed}")
    expected_r_text = (p_text * np.arange(order_count, dtype=np.float32)).sum(axis=1) / formal_k
    expected_r_visual = (p_visual * np.arange(order_count, dtype=np.float32)).sum(axis=1) / formal_k
    if not np.allclose(r_text, expected_r_text, rtol=2e-5, atol=2e-6):
        raise RuntimeError(f"Text normalized response order mismatch for {dataset}/{seed}")
    if not np.allclose(r_visual, expected_r_visual, rtol=2e-5, atol=2e-6):
        raise RuntimeError(f"Visual normalized response order mismatch for {dataset}/{seed}")

    signed_gap = r_text - r_visual
    abs_gap = np.abs(signed_gap)
    profile_l1 = np.abs(p_text - p_visual).sum(axis=1)
    profile_js = _js_divergence_exact_u3a(p_text, p_visual)
    _finite_or_raise("signed_order_gap", signed_gap)
    _finite_or_raise("abs_order_gap", abs_gap)
    _finite_or_raise("profile_l1", profile_l1)
    _finite_or_raise("profile_js", profile_js)

    rows: list[dict[str, Any]] = []
    for modality, order, raw_order, entropy, sum_error, zero_count in (
        ("Text", r_text, raw_text, entropy_text, profile_sum_error_text, all_zero_text),
        ("Visual", r_visual, raw_visual, entropy_visual, profile_sum_error_visual, all_zero_visual),
    ):
        order_stats = _stats_with_width(order, "order_")
        entropy_stats = _stats(entropy, "entropy_")
        raw_stats = _stats(raw_order, "raw_order_")
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "modality": modality,
            "formal_K": formal_k,
            "num_nodes": int(node_id.size),
            "finite_status": bool(finite_status),
            "probability_sum_max_error": float(sum_error),
            "all_zero_contribution_profile_count": int(zero_count),
            **order_stats,
            **entropy_stats,
            "raw_order_mean": raw_stats["raw_order_mean"],
            "raw_order_median": raw_stats["raw_order_median"],
        }
        rows.append(row)

    gap_row: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "formal_K": formal_k,
        "num_nodes": int(node_id.size),
        "finite_status": bool(finite_status),
        **{f"abs_order_gap_{k}": v for k, v in _stats(abs_gap).items()},
        **{f"profile_l1_{k}": v for k, v in _stats(profile_l1).items()},
        **{f"profile_js_{k}": v for k, v in _stats(profile_js).items()},
        "signed_order_gap_mean": float(np.mean(signed_gap)),
        "signed_order_gap_median": float(np.median(signed_gap)),
        "signed_order_gap_q25": float(np.quantile(signed_gap, 0.25)),
        "signed_order_gap_q75": float(np.quantile(signed_gap, 0.75)),
    }
    artifacts = {
        "node_id": node_id.astype(np.int64, copy=False),
        "p_text": p_text.astype(np.float32, copy=False),
        "p_visual": p_visual.astype(np.float32, copy=False),
        "normalized_order_text": r_text.astype(np.float32, copy=False),
        "normalized_order_visual": r_visual.astype(np.float32, copy=False),
        "entropy_text": entropy_text.astype(np.float32, copy=False),
        "entropy_visual": entropy_visual.astype(np.float32, copy=False),
        "signed_order_gap": signed_gap.astype(np.float32, copy=False),
        "abs_order_gap": abs_gap.astype(np.float32, copy=False),
        "profile_l1": profile_l1.astype(np.float32, copy=False),
        "profile_js": profile_js.astype(np.float32, copy=False),
    }
    return rows, gap_row, artifacts


def _load_and_export_checkpoint(
    record: dict[str, Any], device: torch.device, output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    dataset = str(record["dataset"])
    seed = int(record["seed"])
    formal_k = FORMAL_K[dataset]
    cfg = _compose_c1_cfg(dataset, seed, str(device))
    checkpoint = Path(record["checkpoint"])
    checkpoint_sha_before = _sha256(checkpoint)
    if checkpoint_sha_before != record["checkpoint_sha256_before"]:
        raise RuntimeError(f"Source checkpoint changed before load: {checkpoint}")

    # Load only model/data state needed for the frozen predecessor forward.
    data = load_mag_data(cfg, "nc", seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg, payload["data_info"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    model_config = _validate_model_config(model, dataset, cfg)
    model_state_digest_before = _model_state_digest(model)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    with torch.inference_mode():
        components = model._encode_components(x, edge_index)
        text_profile = contribution_profile(
            components["eta_text"], components["states_text"]
        )
        visual_profile = contribution_profile(
            components["eta_visual"], components["states_visual"]
        )

    p_text = _as_float32_numpy(text_profile["probabilities"])
    p_visual = _as_float32_numpy(visual_profile["probabilities"])
    r_text = _as_float32_numpy(text_profile["normalized_response_order"])
    r_visual = _as_float32_numpy(visual_profile["normalized_response_order"])
    raw_text = _as_float32_numpy(text_profile["response_order"])
    raw_visual = _as_float32_numpy(visual_profile["response_order"])
    entropy_text = _as_float32_numpy(text_profile["normalized_entropy"])
    entropy_visual = _as_float32_numpy(visual_profile["normalized_entropy"])
    all_zero_text = int(text_profile["all_zero"].sum().item())
    all_zero_visual = int(visual_profile["all_zero"].sum().item())
    sums_text = p_text.sum(axis=1)
    sums_visual = p_visual.sum(axis=1)
    nonzero_text = sums_text > 0.0
    nonzero_visual = sums_visual > 0.0
    sum_error_text = float(np.max(np.abs(sums_text[nonzero_text] - 1.0))) if nonzero_text.any() else 0.0
    sum_error_visual = float(np.max(np.abs(sums_visual[nonzero_visual] - 1.0))) if nonzero_visual.any() else 0.0
    finite_status = bool(
        np.isfinite(p_text).all()
        and np.isfinite(p_visual).all()
        and np.isfinite(r_text).all()
        and np.isfinite(r_visual).all()
        and np.isfinite(raw_text).all()
        and np.isfinite(raw_visual).all()
        and np.isfinite(entropy_text).all()
        and np.isfinite(entropy_visual).all()
    )
    if not finite_status:
        raise RuntimeError(f"Non-finite contribution profile for {dataset}/{seed}")
    model_state_digest_after = _model_state_digest(model)
    checkpoint_sha_after = _sha256(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError(f"Source checkpoint changed during analysis: {checkpoint}")
    if model_state_digest_after != model_state_digest_before:
        raise RuntimeError(f"Model state changed during analysis: {dataset}/{seed}")

    node_id = np.arange(int(data.num_nodes), dtype=np.int64)
    if data.x_t is None or data.x_i is None:
        raise RuntimeError(f"Both raw modalities are required for {dataset}/{seed}")
    text_input = data.x_t
    visual_input = data.x_i
    if "joint_feat_path" in cfg.dataset:
        joint_feature_path = str(resolve_path(cfg.dataset.joint_feat_path))
    else:
        joint_feature_path = (
            f"constructed by load_mag_data from {resolve_path(cfg.dataset.text_feat_path)} "
            f"and {resolve_path(cfg.dataset.image_feat_path)}"
        )
    text_feature_path = str(resolve_path(cfg.dataset.text_feat_path)) if "text_feat_path" in cfg.dataset else joint_feature_path
    visual_feature_path = str(resolve_path(cfg.dataset.image_feat_path)) if "image_feat_path" in cfg.dataset else joint_feature_path
    feature_input = {
        "joint": {
            "source": "data.x from project load_mag_data (raw text||visual concatenation)",
            "path": joint_feature_path,
            "shape": list(data.x.shape),
            "dtype": str(data.x.dtype),
            "finite": bool(torch.isfinite(data.x).all().item()),
            "zero_norm_node_count": int(torch.linalg.vector_norm(data.x, dim=1).eq(0).sum().item()),
        },
        "text": {
            "source": "data.x_t from project load_mag_data (raw frozen text feature)",
            "path": text_feature_path,
            "shape": list(text_input.shape),
            "dtype": str(text_input.dtype),
            "finite": bool(torch.isfinite(text_input).all().item()),
            "zero_norm_node_count": int(torch.linalg.vector_norm(text_input, dim=1).eq(0).sum().item()),
        },
        "visual": {
            "source": "data.x_i from project load_mag_data (raw frozen visual feature)",
            "path": visual_feature_path,
            "shape": list(visual_input.shape),
            "dtype": str(visual_input.dtype),
            "finite": bool(torch.isfinite(visual_input).all().item()),
            "zero_norm_node_count": int(torch.linalg.vector_norm(visual_input, dim=1).eq(0).sum().item()),
        },
    }
    rows, gap_row, artifacts = _summarize_profiles(
        dataset=dataset,
        seed=seed,
        formal_k=formal_k,
        node_id=node_id,
        p_text=p_text,
        p_visual=p_visual,
        r_text=r_text,
        r_visual=r_visual,
        raw_text=raw_text,
        raw_visual=raw_visual,
        entropy_text=entropy_text,
        entropy_visual=entropy_visual,
        profile_sum_error_text=sum_error_text,
        profile_sum_error_visual=sum_error_visual,
        all_zero_text=all_zero_text,
        all_zero_visual=all_zero_visual,
        finite_status=finite_status,
    )

    seed_dir = output_root / dataset / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(seed_dir / "node_multihop_utilization.npz", **artifacts)
    seed_summary = {
        "dataset": dataset,
        "seed": seed,
        "formal_K": formal_k,
        "num_nodes": int(data.num_nodes),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256_before": checkpoint_sha_before,
        "source_checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_selection": "frozen U2-C C1 best-validation checkpoint from U3-A source manifest",
        "model_config": model_config,
        "feature_input": feature_input,
        "contribution_profile_definition": "src.analysis.u3a.contribution_profile: G=eta*S; g=||G||_2; p=g/sum_k(g)",
        "probability_sum_max_error_text": sum_error_text,
        "probability_sum_max_error_visual": sum_error_visual,
        "all_zero_contribution_profile_count_text": all_zero_text,
        "all_zero_contribution_profile_count_visual": all_zero_visual,
        "finite_status": finite_status,
        "model_state_digest_before_forward": model_state_digest_before,
        "model_state_digest_after_forward": model_state_digest_after,
        "artifacts": {
            "path": str(seed_dir / "node_multihop_utilization.npz"),
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in artifacts.items()},
        },
        "node_summary": rows,
        "modality_gap_summary": gap_row,
        "labels_accessed": False,
        "optimizer_step": False,
        "checkpoint_written": False,
    }
    _write_json(seed_dir / "summary.json", seed_summary)

    del components, text_profile, visual_profile, model, payload, data, x, edge_index
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, gap_row, artifacts


def _save_seed_mean(dataset: str, records: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    if len(records) != len(REQUIRED_SEEDS):
        raise RuntimeError(f"Incomplete seed results for {dataset}")
    node_id = records[0]["node_id"]
    if not all(np.array_equal(node_id, item["node_id"]) for item in records[1:]):
        raise RuntimeError(f"Node IDs are not aligned across seeds for {dataset}")
    p_text = np.mean(np.stack([item["p_text"] for item in records], axis=0), axis=0).astype(np.float32)
    p_visual = np.mean(np.stack([item["p_visual"] for item in records], axis=0), axis=0).astype(np.float32)
    r_text = np.mean(np.stack([item["normalized_order_text"] for item in records], axis=0), axis=0).astype(np.float32)
    r_visual = np.mean(np.stack([item["normalized_order_visual"] for item in records], axis=0), axis=0).astype(np.float32)
    entropy_text = -np.sum(p_text * np.log(p_text + EPS), axis=1) / np.log(FORMAL_K[dataset] + 1)
    entropy_visual = -np.sum(p_visual * np.log(p_visual + EPS), axis=1) / np.log(FORMAL_K[dataset] + 1)
    signed_gap = r_text - r_visual
    abs_gap = np.abs(signed_gap)
    profile_l1 = np.abs(p_text - p_visual).sum(axis=1)
    profile_js = _js_divergence_exact_u3a(p_text, p_visual)
    arrays = {
        "node_id": node_id.astype(np.int64, copy=False),
        "p_text": p_text,
        "p_visual": p_visual,
        "normalized_order_text": r_text,
        "normalized_order_visual": r_visual,
        "entropy_text": entropy_text.astype(np.float32),
        "entropy_visual": entropy_visual.astype(np.float32),
        "signed_order_gap": signed_gap,
        "abs_order_gap": abs_gap,
        "profile_l1": profile_l1,
        "profile_js": profile_js,
    }
    path = output_root / dataset / "seed_mean_node_multihop_utilization.npz"
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "seed_count": len(records),
        "seed_ids": REQUIRED_SEEDS,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        "finite_status": bool(all(np.isfinite(value).all() for value in arrays.values())),
    }


def _csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_order_violin(
    dataset_order: list[str],
    seed_mean: dict[str, dict[str, np.ndarray]],
    seed_results: dict[str, list[dict[str, Any]]],
    output_root: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.4))
    positions = np.arange(1, len(dataset_order) + 1, dtype=float)
    text_data = [seed_mean[d]["normalized_order_text"] for d in dataset_order]
    visual_data = [seed_mean[d]["normalized_order_visual"] for d in dataset_order]
    text_parts = ax.violinplot(text_data, positions=positions - 0.17, widths=0.30, showmedians=False, showextrema=False)
    visual_parts = ax.violinplot(visual_data, positions=positions + 0.17, widths=0.30, showmedians=False, showextrema=False)
    for body in text_parts["bodies"]:
        body.set_facecolor("#2f6fbd")
        body.set_edgecolor("#1f4f8a")
        body.set_alpha(0.65)
    for body in visual_parts["bodies"]:
        body.set_facecolor("#3b9b62")
        body.set_edgecolor("#267344")
        body.set_alpha(0.65)
    for idx, dataset in enumerate(dataset_order, start=1):
        text_medians = [float(np.median(item["normalized_order_text"])) for item in seed_results[dataset]]
        visual_medians = [float(np.median(item["normalized_order_visual"])) for item in seed_results[dataset]]
        ax.scatter(np.full(3, idx - 0.17), text_medians, color="#173b68", s=22, zorder=4, label="seed median" if idx == 1 else None)
        ax.scatter(np.full(3, idx + 0.17), visual_medians, color="#185b35", s=22, zorder=4)
    ax.set_xticks(positions, dataset_order)
    ax.set_ylabel("Normalized contribution-weighted response order")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("E0-C: node-level multi-hop utilization (seed-mean profiles)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(handles=[
        plt.Line2D([], [], color="#2f6fbd", linewidth=8, alpha=0.65, label="Text"),
        plt.Line2D([], [], color="#3b9b62", linewidth=8, alpha=0.65, label="Visual"),
        plt.Line2D([], [], marker="o", color="#173b68", linestyle="None", markersize=5, label="Seed median"),
    ], frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_root / "normalized_response_order_violin.png", dpi=180)
    fig.savefig(output_root / "normalized_response_order_violin.pdf")
    plt.close(fig)


def _plot_modality_gap(
    dataset_order: list[str],
    gap_rows: dict[str, list[dict[str, Any]]],
    output_root: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    positions = np.arange(1, len(dataset_order) + 1, dtype=float)
    specs = [
        ("abs_order_gap_median", "Median |normalized response-order gap|"),
        ("profile_l1_median", "Median L1 distance between contribution profiles"),
    ]
    for ax, (field, ylabel) in zip(axes, specs):
        for idx, dataset in enumerate(dataset_order, start=1):
            values = np.asarray([float(row[field]) for row in gap_rows[dataset]], dtype=float)
            ax.scatter(np.full(values.size, idx), values, color="#555555", s=28, zorder=3)
            ax.errorbar(
                idx,
                float(values.mean()),
                yerr=float(values.std()),
                fmt="o",
                color="#111111",
                markerfacecolor="white",
                markeredgewidth=1.0,
                capsize=4,
                zorder=4,
            )
        ax.set_xticks(positions, dataset_order)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_title("Order gap")
    axes[1].set_title("Full profile gap")
    fig.suptitle("E0-C: text--visual multi-hop utilization discrepancy")
    fig.tight_layout()
    fig.savefig(output_root / "modality_profile_gap.png", dpi=180)
    fig.savefig(output_root / "modality_profile_gap.pdf")
    plt.close(fig)


def _run(args: argparse.Namespace) -> None:
    start_time = time.time()
    if list(DATASETS) != DATASET_ORDER:
        raise RuntimeError(f"Unexpected U3-A dataset order: {DATASETS}")
    if list(SEEDS) != REQUIRED_SEEDS:
        raise RuntimeError(f"Unexpected U3-A seeds: {SEEDS}")
    records, records_by_key = _validate_source_records()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")

    authoritative_profiles = _authoritative_node_profiles_available()
    source_manifest = [
        {
            "dataset": record["dataset"],
            "seed": record["seed"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256_before": record["checkpoint_sha256_before"],
            "variant": "C1",
        }
        for record in [records_by_key[(dataset, seed)] for dataset in DATASET_ORDER for seed in REQUIRED_SEEDS]
    ]
    all_summary_rows: list[dict[str, Any]] = []
    all_gap_rows: list[dict[str, Any]] = []
    seed_results: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in DATASET_ORDER}
    seed_mean_arrays: dict[str, dict[str, np.ndarray]] = {}
    dataset_summaries: dict[str, Any] = {}

    # Source artifact inspection is intentionally explicit: U3-A summaries do
    # not contain per-node tensors, so this path performs one inference-only
    # export from each already frozen checkpoint.
    export_mode = "reuse_existing_authoritative_node_profiles" if authoritative_profiles else "analysis_only_forward_export_from_frozen_checkpoints"
    for dataset in DATASET_ORDER:
        dataset_records: list[dict[str, Any]] = []
        for seed in REQUIRED_SEEDS:
            record = records_by_key[(dataset, seed)]
            rows, gap_row, artifacts = _load_and_export_checkpoint(record, device, output_root)
            all_summary_rows.extend(rows)
            all_gap_rows.append(gap_row)
            dataset_records.append({
                "seed": seed,
                "rows": rows,
                "gap_row": gap_row,
                **artifacts,
            })
            seed_results[dataset].append(artifacts)
            print(f"[E0-C] completed {dataset} seed={seed} nodes={artifacts['node_id'].size}", flush=True)
        seed_mean_meta = _save_seed_mean(dataset, dataset_records, output_root)
        seed_mean_path = Path(seed_mean_meta["path"])
        with np.load(seed_mean_path) as loaded:
            seed_mean_arrays[dataset] = {name: loaded[name].copy() for name in loaded.files}
        dataset_summaries[dataset] = {
            "dataset": dataset,
            "formal_K": FORMAL_K[dataset],
            "seeds": REQUIRED_SEEDS,
            "seed_summaries": [
                {
                    "seed": item["seed"],
                    "summary_json": str(output_root / dataset / f"seed{item['seed']}" / "summary.json"),
                    "node_artifact": str(output_root / dataset / f"seed{item['seed']}" / "node_multihop_utilization.npz"),
                    "modality_gap_summary": item["gap_row"],
                }
                for item in dataset_records
            ],
            "seed_mean_artifact": seed_mean_meta,
        }
        _write_json(output_root / dataset / "summary.json", dataset_summaries[dataset])

    summary_fields = [
        "dataset", "seed", "modality", "formal_K", "num_nodes", "finite_status",
        "probability_sum_max_error", "all_zero_contribution_profile_count",
        "order_mean", "order_std", "order_median", "order_q10", "order_q25", "order_q75", "order_q90",
        "order_iqr", "order_q90_q10", "entropy_mean", "entropy_std", "entropy_median",
        "entropy_q10", "entropy_q25", "entropy_q75", "entropy_q90", "raw_order_mean", "raw_order_median",
    ]
    gap_fields = [
        "dataset", "seed", "formal_K", "num_nodes", "finite_status",
        "abs_order_gap_mean", "abs_order_gap_std", "abs_order_gap_median", "abs_order_gap_q10", "abs_order_gap_q25", "abs_order_gap_q75", "abs_order_gap_q90",
        "profile_l1_mean", "profile_l1_std", "profile_l1_median", "profile_l1_q10", "profile_l1_q25", "profile_l1_q75", "profile_l1_q90",
        "profile_js_mean", "profile_js_std", "profile_js_median", "profile_js_q10", "profile_js_q25", "profile_js_q75", "profile_js_q90",
        "signed_order_gap_mean", "signed_order_gap_median", "signed_order_gap_q25", "signed_order_gap_q75",
    ]
    _csv_write(output_root / "multihop_utilization_summary.csv", all_summary_rows, summary_fields)
    _csv_write(output_root / "modality_gap_summary.csv", all_gap_rows, gap_fields)
    _plot_order_violin(DATASET_ORDER, seed_mean_arrays, seed_results, output_root)
    gap_rows_by_dataset = {
        dataset: [item["modality_gap_summary"] for item in dataset_summaries[dataset]["seed_summaries"]]
        for dataset in DATASET_ORDER
    }
    _plot_modality_gap(DATASET_ORDER, gap_rows_by_dataset, output_root)

    manifest = {
        "experiment": "E0-C Node-Modality Multi-Hop Utilization Heterogeneity",
        "status": "complete",
        "repository": str(REPO_ROOT),
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty_before_experiment": "not captured by script; existing unrelated E0-A/E0-B files were preserved",
        "datasets": DATASET_ORDER,
        "seeds": REQUIRED_SEEDS,
        "formal_K": FORMAL_K,
        "analysis_orders_by_dataset": {dataset: FORMAL_K[dataset] + 1 for dataset in DATASET_ORDER},
        "device": str(device),
        "source": {
            "authoritative_u3a_master": str(U3A_MASTER),
            "source_records_resolved_from": "scripts/run_mopf_u3a_node_modality_preference_diagnosis.py::_source_records",
            "source_checkpoints": source_manifest,
            "checkpoint_sha_verified_before_and_after_each_forward": True,
            "authoritative_node_profiles_available_before_run": authoritative_profiles,
            "export_mode": export_mode,
        },
        "fixed_model_configuration": {
            "edge_weight_mode": "learned_diag_cos",
            "edge_weight_temperature": 0.35,
            "multihop_state_mode": "anchored",
            "multihop_response_mode": "cumulative",
            "multihop_anchor_alpha": 0.1,
            "use_transport_residual": False,
            "tcpr_enabled": False,
        },
        "contribution_profile": {
            "implementation": "src.analysis.u3a.contribution_profile",
            "definition": "G_i,k^m=eta_i,k^m*S_i,k^m; g_i,k^m=||G_i,k^m||_2; p_i,k^m=g_i,k^m/sum_r g_i,r^m",
            "normalized_order": "sum_k (k / formal_K) * p_i,k^m",
            "raw_order_retained": True,
            "probability_sum_error_is_reported_on_nonzero_profiles": True,
        },
        "statistics": {
            "computed_first_per_seed": True,
            "seed_mean_node_profiles_for_violin": True,
            "three_seed_markers": True,
            "pooled_3xN_inference": False,
            "counterfactuals_run": False,
            "association_analysis_run": False,
            "labels_or_test_metrics_used": False,
        },
        "reproducibility": {
            "exact_command": f"MPLCONFIGDIR=/tmp/mopf_e0c_mpl PYTHONPATH=src conda run --no-capture-output -n yhf_env python scripts/run_mopf_e0c_semantic_retention_heterogeneity.py --device {device}",
            "runtime_seconds": float(time.time() - start_time),
            "output_root": str(output_root),
        },
        "audit": {
            "optimizer_step": False,
            "training_started": False,
            "checkpoint_write": False,
            "final_configuration_modified": False,
            "model_state_digest_checked_before_after_forward": True,
            "source_checkpoint_bytes_checked_before_after_forward": True,
            "graph_or_features_modified": False,
        },
    }
    _write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps({"output_root": str(output_root), "runtime_seconds": manifest["reproducibility"]["runtime_seconds"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
