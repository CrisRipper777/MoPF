#!/usr/bin/env python3
"""Generate correctness-audited Figure 3 V2 mechanism data.

Only existing full/final checkpoints are loaded.  The script never invokes a
training command.  Panel (c) uses a same-checkpoint counterfactual in which
only the semantic-anchor coefficient is set to zero in the recurrence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "outputs" / "figure3_mechanism_v2"
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs" / "cosi_mag_final_benchmark"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
HOPS = (0, 1, 2, 3)
PRIMARY_SEMANTIC_SPACE = "projected_h0_raw"
AUDIT_SEMANTIC_SPACES = ("projected_h0_raw", "input_raw", "learned_metric_internal")
GAP_BINS = 8

PANEL_A_FIELDS = (
    "dataset",
    "seed",
    "semantic_space",
    "relation_weight",
    "semantic_similarity",
    "num_nodes",
    "num_physical_edges",
    "spearman_rho",
    "spearman_pvalue",
    "spearman_status",
)
PANEL_A_GAP_FIELDS = (
    "dataset",
    "seed",
    "bin_index",
    "num_edges",
    "delta_sem_mean",
    "delta_sem_q25",
    "delta_sem_q75",
    "delta_weight_mean",
    "delta_weight_sd",
    "delta_weight_q25",
    "delta_weight_q75",
    "edge_spearman_rho",
    "edge_spearman_pvalue",
    "edge_slope",
    "semantic_space",
)
PANEL_B_FIELDS = (
    "dataset",
    "seed",
    "modality",
    "node_id",
    "degree",
    "raw_compatibility",
    "calibrated_compatibility",
    "delta_compatibility",
    "improved",
    "degree_filter",
    "semantic_space",
    "weight_definition",
)
PANEL_B_AUDIT_FIELDS = (
    "dataset",
    "seed",
    "num_nodes",
    "degree_zero_count",
    "degree_one_count",
    "degree_gt1_count",
    "degree_zero_fraction",
    "degree_one_fraction",
    "degree_gt1_fraction",
    "degree_filter_for_main_panel",
)
PANEL_C_FIELDS = (
    "dataset",
    "seed",
    "checkpoint_variant",
    "propagation_condition",
    "modality",
    "hop",
    "num_nodes",
    "mean_retention",
    "sd_retention_across_nodes",
    "median_retention",
    "retention_q25",
    "retention_q75",
    "anchor_alpha_used",
    "same_checkpoint_counterfactual",
    "hop3_gain_pp_vs_anchor_off",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _checkpoint_path(root: Path, dataset: str, seed: int) -> Path:
    run_id = SEEDS.index(seed) + 1
    return root / "nc" / dataset / "runs_42_43_44" / f"best_run{run_id}.pt"


def _config_path(checkpoint: Path) -> Path:
    return checkpoint.parent / "resolved_config.json"


def _load_checkpoint_model(checkpoint: Path, device: torch.device):
    from src.data import load_mag_data
    from src.models import build_model

    config_path = _config_path(checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f"missing resolved config: {config_path}")
    cfg = OmegaConf.create(json.loads(config_path.read_text(encoding="utf-8")))
    data = load_mag_data(cfg, "nc", int(cfg.seed))
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError(f"checkpoint has no model_state: {checkpoint}")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return cfg, data, model, payload


def _edge_cosine(feature: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(feature.float(), p=2, dim=1, eps=1e-12)
    src, dst = edge_index
    return (normalized[src] * normalized[dst]).sum(dim=1)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite {name}: {value}")
    return value


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, str]:
    if x.size < 3 or y.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None, None, "undefined_constant_or_short_vector"
    result = spearmanr(x, y, nan_policy="raise")
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    if not (np.isfinite(rho) and np.isfinite(pvalue)):
        return None, None, "undefined_nonfinite"
    return rho, pvalue, "ok"


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.ptp(x) == 0.0:
        return None
    slope = float(np.polyfit(x, y, 1)[0])
    return slope if np.isfinite(slope) else None


def _binned_gap_response(delta_sem: np.ndarray, delta_weight: np.ndarray, dataset: str, seed: int) -> list[dict[str, Any]]:
    order = np.argsort(delta_sem, kind="stable")
    chunks = [chunk for chunk in np.array_split(order, GAP_BINS) if chunk.size]
    edge_rho, edge_pvalue, _ = _correlation(delta_sem, delta_weight)
    edge_slope = _linear_slope(delta_sem, delta_weight)
    rows: list[dict[str, Any]] = []
    for bin_index, indices in enumerate(chunks):
        sem = delta_sem[indices]
        weight = delta_weight[indices]
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "bin_index": bin_index,
                "num_edges": int(indices.size),
                "delta_sem_mean": _finite(np.mean(sem), "delta_sem_mean"),
                "delta_sem_q25": _finite(np.quantile(sem, 0.25), "delta_sem_q25"),
                "delta_sem_q75": _finite(np.quantile(sem, 0.75), "delta_sem_q75"),
                "delta_weight_mean": _finite(np.mean(weight), "delta_weight_mean"),
                "delta_weight_sd": _finite(np.std(weight), "delta_weight_sd"),
                "delta_weight_q25": _finite(np.quantile(weight, 0.25), "delta_weight_q25"),
                "delta_weight_q75": _finite(np.quantile(weight, 0.75), "delta_weight_q75"),
                "edge_spearman_rho": "" if edge_rho is None else edge_rho,
                "edge_spearman_pvalue": "" if edge_pvalue is None else edge_pvalue,
                "edge_slope": "" if edge_slope is None else edge_slope,
                "semantic_space": PRIMARY_SEMANTIC_SPACE,
            }
        )
    return rows


def _counterfactual_anchor_off(
    model: Any,
    h0: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> list[torch.Tensor]:
    """Recompute only the propagation recurrence with alpha=0.

    The checkpoint, projected h0, relation weights, normalized operator, and
    every learned parameter remain unchanged.  This is not a retrained model.
    """
    states = [h0]
    current = h0
    for _ in range(int(model.max_order)):
        current = model._propagate_once(current, edge_index, edge_weight)
        states.append(current)
    return states


def _retention_rows(
    *,
    dataset: str,
    seed: int,
    h0: torch.Tensor,
    anchored: list[torch.Tensor],
    anchor_off: list[torch.Tensor],
    anchor_alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conditions = (
        ("anchored", anchored, anchor_alpha, True),
        ("anchor_off_counterfactual", anchor_off, 0.0, True),
    )
    # The explicit loop below keeps the modality label adjacent to the tensor
    # source, which makes the audit CSV self-describing.
    for modality, anchor, anchored_states, off_states in (
        ("text", h0[0], anchored[0], anchor_off[0]),
        ("visual", h0[1], anchored[1], anchor_off[1]),
    ):
        for condition, states, used_alpha, same_checkpoint in conditions:
            selected_states = anchored_states if condition == "anchored" else off_states
            for hop, state in enumerate(selected_states):
                values = _as_numpy(torch.nn.functional.cosine_similarity(state.float(), anchor.float(), dim=-1, eps=1e-12))
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "checkpoint_variant": "cosi_mag_final",
                        "propagation_condition": condition,
                        "modality": modality,
                        "hop": hop,
                        "num_nodes": int(values.size),
                        "mean_retention": _finite(np.mean(values), "mean_retention"),
                        "sd_retention_across_nodes": _finite(np.std(values), "sd_retention_across_nodes"),
                        "median_retention": _finite(np.median(values), "median_retention"),
                        "retention_q25": _finite(np.quantile(values, 0.25), "retention_q25"),
                        "retention_q75": _finite(np.quantile(values, 0.75), "retention_q75"),
                        "anchor_alpha_used": used_alpha,
                        "same_checkpoint_counterfactual": same_checkpoint if condition != "anchored" else False,
                        "hop3_gain_pp_vs_anchor_off": "",
                    }
                )
    by_key = {
        (row["modality"], row["hop"], row["propagation_condition"]): row
        for row in rows
    }
    for modality in MODALITIES:
        anchored_row = by_key[(modality, 3, "anchored")]
        off_row = by_key[(modality, 3, "anchor_off_counterfactual")]
        anchored_row["hop3_gain_pp_vs_anchor_off"] = 100.0 * (
            float(anchored_row["mean_retention"]) - float(off_row["mean_retention"])
        )
    return rows


def _extract_one(
    *, dataset: str, seed: int, checkpoint: Path, device: torch.device
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    cfg, data, model, payload = _load_checkpoint_model(checkpoint, device)
    if data.x_t is None or data.x_i is None:
        raise ValueError("both modality features are required")
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    input_features = {"text": data.x_t.to(device), "visual": data.x_i.to(device)}
    panel_a: list[dict[str, Any]] = []
    panel_a_gap: list[dict[str, Any]] = []
    panel_b: list[dict[str, Any]] = []
    panel_b_audit: list[dict[str, Any]] = []
    panel_c: list[dict[str, Any]] = []

    with torch.inference_mode():
        components = model._analysis_call(x, edge_index)
        h0 = {"text": components["h0_text"], "visual": components["h0_visual"]}
        weights = {"text": components["relation_weight_text"], "visual": components["relation_weight_visual"]}
        primary_semantics = {modality: _edge_cosine(h0[modality], edge_index) for modality in MODALITIES}
        input_semantics = {modality: _edge_cosine(input_features[modality], edge_index) for modality in MODALITIES}
        learned_semantics = {"text": components["semantic_cosine_text"], "visual": components["semantic_cosine_visual"]}
        semantic_spaces = {
            "projected_h0_raw": primary_semantics,
            "input_raw": input_semantics,
            "learned_metric_internal": learned_semantics,
        }
        relation_pairs = (("text", "text"), ("text", "visual"), ("visual", "visual"), ("visual", "text"))
        for semantic_space, semantic_values in semantic_spaces.items():
            for weight_modality, semantic_modality in relation_pairs:
                rho, pvalue, status = _correlation(_as_numpy(weights[weight_modality]), _as_numpy(semantic_values[semantic_modality]))
                panel_a.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "semantic_space": semantic_space,
                        "relation_weight": f"W^{weight_modality[:1].upper()}",
                        "semantic_similarity": f"S^{semantic_modality[:1].upper()}",
                        "num_nodes": int(data.num_nodes),
                        "num_physical_edges": int(edge_index.size(1)),
                        "spearman_rho": "" if rho is None else rho,
                        "spearman_pvalue": "" if pvalue is None else pvalue,
                        "spearman_status": status,
                    }
                )
        delta_sem = _as_numpy(primary_semantics["text"] - primary_semantics["visual"])
        delta_weight = _as_numpy(weights["text"] - weights["visual"])
        panel_a_gap.extend(_binned_gap_response(delta_sem, delta_weight, dataset, seed))

        src, _ = edge_index
        num_nodes = int(data.num_nodes)
        degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        degree_zero = degree == 0
        degree_one = degree == 1
        degree_gt1 = degree > 1
        panel_b_audit.append(
            {
                "dataset": dataset,
                "seed": seed,
                "num_nodes": num_nodes,
                "degree_zero_count": int(degree_zero.sum().item()),
                "degree_one_count": int(degree_one.sum().item()),
                "degree_gt1_count": int(degree_gt1.sum().item()),
                "degree_zero_fraction": _finite(float(degree_zero.float().mean().item()), "degree_zero_fraction"),
                "degree_one_fraction": _finite(float(degree_one.float().mean().item()), "degree_one_fraction"),
                "degree_gt1_fraction": _finite(float(degree_gt1.float().mean().item()), "degree_gt1_fraction"),
                "degree_filter_for_main_panel": "degree > 1",
            }
        )
        valid_nodes = torch.nonzero(degree_gt1, as_tuple=False).flatten()
        for modality in MODALITIES:
            semantic = primary_semantics[modality].float()
            weight = weights[modality].float()
            raw_sum = torch.zeros(num_nodes, dtype=torch.float32, device=device)
            weighted_sum = torch.zeros_like(raw_sum)
            weight_sum = torch.zeros_like(raw_sum)
            raw_sum.index_add_(0, src, semantic)
            weighted_sum.index_add_(0, src, weight * semantic)
            weight_sum.index_add_(0, src, weight)
            raw_compatibility = raw_sum[valid_nodes] / degree[valid_nodes]
            calibrated_compatibility = weighted_sum[valid_nodes] / weight_sum[valid_nodes].clamp_min(1e-12)
            delta = calibrated_compatibility - raw_compatibility
            for node_id, raw_value, calibrated_value, delta_value in zip(
                valid_nodes.cpu().tolist(),
                _as_numpy(raw_compatibility),
                _as_numpy(calibrated_compatibility),
                _as_numpy(delta),
            ):
                delta_value = _finite(delta_value, "delta_compatibility")
                panel_b.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "modality": modality,
                        "node_id": int(node_id),
                        "degree": int(degree[node_id].item()),
                        "raw_compatibility": _finite(raw_value, "raw_compatibility"),
                        "calibrated_compatibility": _finite(calibrated_value, "calibrated_compatibility"),
                        "delta_compatibility": delta_value,
                        "improved": int(delta_value > 0.0),
                        "degree_filter": "degree > 1",
                        "semantic_space": PRIMARY_SEMANTIC_SPACE,
                        "weight_definition": "raw relation_weight_W_m",
                    }
                )

        anchored_states = {"text": components["states_text"], "visual": components["states_visual"]}
        anchor_off_states = {
            "text": _counterfactual_anchor_off(model, h0["text"], components["normalized_edge_index_text"], components["normalized_edge_weight_text"]),
            "visual": _counterfactual_anchor_off(model, h0["visual"], components["normalized_edge_index_visual"], components["normalized_edge_weight_visual"]),
        }
        panel_c.extend(
            _retention_rows(
                dataset=dataset,
                seed=seed,
                h0=(h0["text"], h0["visual"]),
                anchored=(anchored_states["text"], anchored_states["visual"]),
                anchor_off=(anchor_off_states["text"], anchor_off_states["visual"]),
                anchor_alpha=float(cfg.model.multihop_anchor_alpha),
            )
        )

    metadata = {
        "dataset": dataset,
        "seed": seed,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "resolved_config": _relative(_config_path(checkpoint)),
        "checkpoint_selection": payload.get("selection"),
        "best_epoch": payload.get("epoch"),
        "model_name": str(cfg.model.name),
        "anchor_alpha": float(cfg.model.multihop_anchor_alpha),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(edge_index.size(1)),
        "semantic_space_primary": PRIMARY_SEMANTIC_SPACE,
        "weight_definition_panel_b": "raw relation_weight_W_m before normalized propagation operator",
        "panel_c_counterfactual": "same checkpoint, alpha=0 recurrence only",
    }
    del components, model, data, x, edge_index, input_features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return panel_a, panel_a_gap, panel_b, panel_b_audit, panel_c, metadata


def _planned_inputs(root: Path, datasets: list[str], seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            checkpoint = _checkpoint_path(root, dataset, seed)
            item = {"dataset": dataset, "seed": seed, "checkpoint": _relative(checkpoint)}
            missing_fields = [
                name for name, path in (("checkpoint", checkpoint), ("resolved_config", _config_path(checkpoint))) if not path.is_file()
            ]
            if missing_fields:
                item["missing"] = missing_fields
                missing.append(item)
            else:
                present.append(item)
    return present, missing


def _summary_markdown(
    path: Path,
    *,
    panel_a: list[dict[str, Any]],
    panel_a_gap: list[dict[str, Any]],
    panel_b: list[dict[str, Any]],
    panel_b_audit: list[dict[str, Any]],
    panel_c: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    primary = [row for row in panel_a if row["semantic_space"] == PRIMARY_SEMANTIC_SPACE]
    lines = [
        "# Figure 3 V2 audit summary",
        "",
        "No new training was started. All values come from existing full/final checkpoints.",
        "",
        "## Correctness audit",
        "",
        "- **A. Semantic similarity:** primary `S^T_raw`/`S^V_raw` is cosine similarity on the projected `H0` states before the learned diagonal metric. Original input-feature cosine is retained as `input_raw` sensitivity rows; learned metric scores are audit-only and are not used as the primary semantic variable.",
        "- **B. Calibrated compatibility:** primary Panel (b) uses raw relation weights `W^m` before the normalized propagation operator. It does not use `normalized_edge_weight_m`.",
        "- **C. Degree filter:** primary Panel (b) contains only nodes with degree > 1. Degree=1 and degree=0 counts/fractions are reported separately.",
        "- **D. SMP intervention:** Panel (c) compares anchored recurrence with an alpha=0 anchor-off recurrence on the same full checkpoint, same `H0`, same raw `W^m`, and same normalized operator. No retrained ablation checkpoint is used.",
        "",
        "## Panel (a) A1: matched versus cross-modal correlations",
        "",
    ]
    for dataset in DATASETS:
        values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in primary:
            if row["dataset"] == dataset and row["spearman_rho"] != "":
                values[(row["relation_weight"], row["semantic_similarity"])].append(float(row["spearman_rho"]))
        parts = [f"{key[0]} vs {key[1]}={np.mean(value):.3f} ± {np.std(value):.3f}" for key, value in sorted(values.items())]
        lines.append(f"- {dataset}: " + "; ".join(parts))
    lines.extend(["", "## Panel (a) A2: semantic-gap response", ""])
    for dataset in DATASETS:
        grouped = [row for row in panel_a_gap if row["dataset"] == dataset]
        slopes = [float(row["edge_slope"]) for row in grouped if row["edge_slope"] != ""]
        rhos = [float(row["edge_spearman_rho"]) for row in grouped if row["edge_spearman_rho"] != ""]
        lines.append(f"- {dataset}: edge-level rho={np.mean(rhos):.3f}; slope={np.mean(slopes):.4f}." if slopes and rhos else f"- {dataset}: response statistic unavailable.")
    lines.extend(["", "## Panel (b): degree-filtered improvement", ""])
    for dataset in DATASETS:
        audits = [row for row in panel_b_audit if row["dataset"] == dataset]
        if audits:
            degree_one = np.mean([float(row["degree_one_fraction"]) for row in audits])
            gt1 = np.mean([float(row["degree_gt1_count"]) for row in audits])
            lines.append(f"- {dataset}: degree=1 fraction={degree_one:.3f}; mean degree>1 node count={gt1:.1f}.")
        for modality in MODALITIES:
            values = [float(row["delta_compatibility"]) for row in panel_b if row["dataset"] == dataset and row["modality"] == modality]
            if values:
                lines.append(f"  - {modality}: positive fraction={np.mean(np.asarray(values) > 0):.3f}; median Δu={np.median(values):.5f}.")
    lines.extend(["", "## Panel (c): same-checkpoint anchor intervention", ""])
    for dataset in DATASETS:
        for modality in MODALITIES:
            anchored = [float(row["mean_retention"]) for row in panel_c if row["dataset"] == dataset and row["modality"] == modality and row["propagation_condition"] == "anchored" and int(row["hop"]) == 3]
            off = [float(row["mean_retention"]) for row in panel_c if row["dataset"] == dataset and row["modality"] == modality and row["propagation_condition"] == "anchor_off_counterfactual" and int(row["hop"]) == 3]
            if anchored and off:
                lines.append(f"- {dataset}-{modality}: hop-3 gain={100 * (np.mean(anchored) - np.mean(off)):.2f} percentage points.")
    if missing:
        lines.extend(["", "## Missing inputs", ""])
        for item in missing:
            lines.append(f"- {item['dataset']} seed {item['seed']}: {', '.join(item.get('missing', []))}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    seeds = list(dict.fromkeys(args.seeds))
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    present, missing = _planned_inputs(checkpoint_root, datasets, seeds)
    print(f"[figure3-v2-data] planned checkpoints: {len(present) + len(missing)}")
    print(f"[figure3-v2-data] available: {len(present)} | missing: {len(missing)}")
    for item in missing:
        print(f"[figure3-v2-data] missing {item['dataset']} seed={item['seed']}: {item['missing']}", file=sys.stderr)
    if args.dry_run:
        for item in present:
            print(f"[figure3-v2-data] ready {item['dataset']} seed={item['seed']}: {item['checkpoint']}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    requested_device = str(args.device)
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[figure3-v2-data] CUDA unavailable for {requested_device}; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    panel_a: list[dict[str, Any]] = []
    panel_a_gap: list[dict[str, Any]] = []
    panel_b: list[dict[str, Any]] = []
    panel_b_audit: list[dict[str, Any]] = []
    panel_c: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in present:
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        try:
            a, gap, b, b_audit, c, metadata = _extract_one(
                dataset=item["dataset"], seed=int(item["seed"]), checkpoint=checkpoint, device=device
            )
            panel_a.extend(a)
            panel_a_gap.extend(gap)
            panel_b.extend(b)
            panel_b_audit.extend(b_audit)
            panel_c.extend(c)
            analyses.append(metadata)
            print(f"[figure3-v2-data] analyzed {item['dataset']} seed={item['seed']}", flush=True)
        except Exception as exc:
            failure = {**item, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[figure3-v2-data] failed: {failure}", file=sys.stderr, flush=True)

    _write_csv(output_dir / "panel_a_correlation.csv", PANEL_A_FIELDS, panel_a)
    _write_csv(output_dir / "panel_a_gap_response.csv", PANEL_A_GAP_FIELDS, panel_a_gap)
    _write_csv(output_dir / "panel_b_improvement.csv", PANEL_B_FIELDS, panel_b)
    _write_csv(output_dir / "panel_b_degree_audit.csv", PANEL_B_AUDIT_FIELDS, panel_b_audit)
    _write_csv(output_dir / "panel_c_retention_counterfactual.csv", PANEL_C_FIELDS, panel_c)
    all_missing = [*missing, *failures]
    manifest = {
        "figure": "Figure 3 V2",
        "purpose": "Mechanism verification of relation calibration and semantic-preserving propagation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": _relative(Path(__file__)),
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit_sha": _git_value("rev-parse", "HEAD"),
        "git_worktree_status": _git_value("status", "--short"),
        "datasets": datasets,
        "seeds": seeds,
        "requested_device": requested_device,
        "device": str(device),
        "checkpoint_root": _relative(checkpoint_root),
        "correctness_audit": {
            "semantic_similarity_primary": "projected_h0_raw: cosine on H0 before learned metric",
            "semantic_similarity_sensitivity": "input_raw retained in panel_a_correlation.csv",
            "learned_metric_internal": "audit-only; excluded from primary panels",
            "panel_b_weight_definition": "raw relation_weight_W_m, not normalized_edge_weight_m",
            "panel_b_degree_filter": "degree > 1",
            "panel_c_intervention": "same full checkpoint; alpha=0 recurrence only; no retrained ablation",
        },
        "available_inputs": present,
        "missing_inputs": missing,
        "analysis_failures": failures,
        "checkpoint_analyses": analyses,
        "row_counts": {
            "panel_a_correlation": len(panel_a),
            "panel_a_gap_response": len(panel_a_gap),
            "panel_b_improvement": len(panel_b),
            "panel_b_degree_audit": len(panel_b_audit),
            "panel_c_retention_counterfactual": len(panel_c),
        },
        "outputs": {
            "audit_summary": "audit_summary.md",
            "panel_a_correlation": "panel_a_correlation.csv",
            "panel_a_gap_response": "panel_a_gap_response.csv",
            "panel_b_improvement": "panel_b_improvement.csv",
            "panel_b_degree_audit": "panel_b_degree_audit.csv",
            "panel_c_retention_counterfactual": "panel_c_retention_counterfactual.csv",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _summary_markdown(
        output_dir / "audit_summary.md",
        panel_a=panel_a,
        panel_a_gap=panel_a_gap,
        panel_b=panel_b,
        panel_b_audit=panel_b_audit,
        panel_c=panel_c,
        missing=all_missing,
    )
    print(f"[figure3-v2-data] wrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
