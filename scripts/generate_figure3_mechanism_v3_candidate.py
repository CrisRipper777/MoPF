#!/usr/bin/env python3
"""Generate the candidate Panel 3(b) modality-specific allocation analysis.

This script is analysis-only.  It reads existing CoSI-MAG full/final
checkpoints and uses the raw relation weights exposed by ``_analysis_call``.
For every target node, the physical incoming neighbor support is shared by the
text and visual branches; only the raw relation weights differ.  Degree-one
nodes are excluded from the reported allocation statistics.

The propagation-facing convention is used here deliberately: an edge
``src -> dst`` contributes source ``src`` to target ``dst``.  Therefore the
physical neighborhood of node ``i`` is the set of unique ``src`` IDs on edges
whose ``dst == i``.  The normalized propagation operator is never used.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "outputs" / "figure3_mechanism_v3_candidate"
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs" / "cosi_mag_final_benchmark"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
WEIGHT_DEFINITION = "raw relation_weight_W_m"
SUPPORT_DEFINITION = "physical incoming neighbors from edge_index; no self-loop; same support for text/visual"
DEGREE_FILTER = "unique physical incoming degree > 1"
NEIGHBOR_ORIENTATION = "incoming source-to-target: edge src -> dst contributes src to dst"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _checkpoint_path(root: Path, dataset: str, seed: int) -> Path:
    run_id = SEEDS.index(seed) + 1
    return root / "nc" / dataset / "runs_42_43_44" / f"best_run{run_id}.pt"


def _config_path(checkpoint: Path) -> Path:
    return checkpoint.parent / "resolved_config.json"


def _load_checkpoint_model(checkpoint: Path, device: torch.device):
    # Reuse the V2 read-only loader.  The import happens only for real
    # checkpoint analysis, not for self-tests or dry-runs.
    from scripts.generate_figure3_mechanism_v2 import _load_checkpoint_model as loader

    return loader(checkpoint, device)

DETAIL_FIELDS = (
    "dataset",
    "seed",
    "node_id",
    "degree",
    "tv_distance",
    "top_neighbor_text",
    "top_neighbor_visual",
    "top_neighbor_disagree",
    "prob_sum_text",
    "prob_sum_visual",
    "prob_sum_error",
    "physical_neighbor_ids",
    "weight_definition",
    "support_definition",
    "degree_filter",
    "neighbor_orientation",
)

SUMMARY_FIELDS = (
    "dataset",
    "aggregation_level",
    "seed",
    "num_nodes_degree_gt1",
    "mean_tv",
    "median_tv",
    "tv_q25",
    "tv_q75",
    "tv_iqr",
    "tv_sd",
    "tv_min",
    "tv_max",
    "top_neighbor_disagreement_percentage",
    "top_neighbor_disagreement_count",
    "max_probability_sum_error",
    "tv_bounds_ok",
    "degree_filter",
    "support_definition",
    "weight_definition",
    "cross_seed_mean_tv",
    "cross_seed_std_tv",
    "cross_seed_mean_disagreement_percentage",
    "cross_seed_std_disagreement_percentage",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="run toy allocation correctness tests and exit")
    return parser.parse_args()


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


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite {name}: {value}")
    return value


def _node_allocation_metrics(
    neighbor_ids: np.ndarray,
    text_weights: np.ndarray,
    visual_weights: np.ndarray,
    *,
    require_degree_gt1: bool = True,
) -> dict[str, Any] | None:
    """Compute TV/top-neighbor metrics after aligning by physical neighbor ID.

    Duplicate physical edges, if present, are aggregated by neighbor ID before
    normalization.  This makes ``p_ij`` a distribution over physical neighbors
    rather than over duplicated edge records.  Both modalities are required to
    have exactly the same support.
    """
    ids = np.asarray(neighbor_ids, dtype=np.int64).reshape(-1)
    text = np.asarray(text_weights, dtype=np.float64).reshape(-1)
    visual = np.asarray(visual_weights, dtype=np.float64).reshape(-1)
    if ids.size != text.size or ids.size != visual.size:
        raise ValueError("neighbor IDs and both modality weight vectors must have equal length")
    if ids.size == 0:
        return None
    if not (np.isfinite(text).all() and np.isfinite(visual).all()):
        raise ValueError("relation weights contain non-finite values")
    if np.any(text < 0.0) or np.any(visual < 0.0):
        raise ValueError("raw relation weights must be non-negative")

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    degree = int(unique_ids.size)
    if require_degree_gt1 and degree <= 1:
        return None

    text_by_neighbor = np.zeros(degree, dtype=np.float64)
    visual_by_neighbor = np.zeros(degree, dtype=np.float64)
    np.add.at(text_by_neighbor, inverse, text)
    np.add.at(visual_by_neighbor, inverse, visual)
    text_total = float(text_by_neighbor.sum())
    visual_total = float(visual_by_neighbor.sum())
    if text_total <= 0.0 or visual_total <= 0.0:
        raise ValueError("relation weights have a non-positive node-level total")

    p_text = text_by_neighbor / text_total
    p_visual = visual_by_neighbor / visual_total
    sum_text = float(p_text.sum())
    sum_visual = float(p_visual.sum())
    tv = float(0.5 * np.abs(p_text - p_visual).sum())
    top_text = int(unique_ids[int(np.argmax(p_text))])
    top_visual = int(unique_ids[int(np.argmax(p_visual))])
    top_disagree = bool(top_text != top_visual)
    if not np.isclose(sum_text, 1.0, atol=1e-10) or not np.isclose(sum_visual, 1.0, atol=1e-10):
        raise AssertionError("node-level probability vector does not sum to one")
    if tv < -1e-10 or tv > 1.0 + 1e-10:
        raise AssertionError(f"TV distance outside [0, 1]: {tv}")
    return {
        "degree": degree,
        "tv_distance": max(0.0, min(1.0, tv)),
        "top_neighbor_text": top_text,
        "top_neighbor_visual": top_visual,
        "top_neighbor_disagree": top_disagree,
        "prob_sum_text": sum_text,
        "prob_sum_visual": sum_visual,
        "prob_sum_error": max(abs(sum_text - 1.0), abs(sum_visual - 1.0)),
        "physical_neighbor_ids": ",".join(str(int(value)) for value in unique_ids.tolist()),
    }


def _run_self_tests() -> None:
    ids = np.asarray([10, 11], dtype=np.int64)
    equal = _node_allocation_metrics(ids, np.asarray([0.4, 0.6]), np.asarray([0.4, 0.6]))
    assert equal is not None
    assert np.isclose(equal["tv_distance"], 0.0, atol=1e-12)
    assert equal["top_neighbor_text"] == equal["top_neighbor_visual"]
    assert not equal["top_neighbor_disagree"]
    assert np.isclose(equal["prob_sum_text"], 1.0)
    assert np.isclose(equal["prob_sum_visual"], 1.0)

    swapped = _node_allocation_metrics(ids, np.asarray([0.9, 0.1]), np.asarray([0.1, 0.9]))
    assert swapped is not None
    assert swapped["tv_distance"] > 0.0
    assert swapped["top_neighbor_disagree"]
    assert 0.0 <= swapped["tv_distance"] <= 1.0

    assert _node_allocation_metrics(np.asarray([10]), np.asarray([1.0]), np.asarray([1.0])) is None
    try:
        _node_allocation_metrics(np.asarray([10, 11]), np.asarray([1.0]), np.asarray([1.0, 2.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched modality vectors should fail")

    duplicate = _node_allocation_metrics(
        np.asarray([10, 10, 11]), np.asarray([0.2, 0.2, 0.6]), np.asarray([0.4, 0.0, 0.6])
    )
    assert duplicate is not None and duplicate["degree"] == 2
    print("[figure3-v3-data] self-tests passed")


def _planned_inputs(root: Path, datasets: list[str], seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            checkpoint = _checkpoint_path(root, dataset, seed)
            item = {"dataset": dataset, "seed": seed, "checkpoint": _relative(checkpoint)}
            missing_fields = [
                name
                for name, path in (("checkpoint", checkpoint), ("resolved_config", _config_path(checkpoint)))
                if not path.is_file()
            ]
            if missing_fields:
                item["missing"] = missing_fields
                missing.append(item)
            else:
                present.append(item)
    return present, missing


def _extract_one(*, dataset: str, seed: int, checkpoint: Path, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg, data, model, payload = _load_checkpoint_model(checkpoint, device)
    if data.x_t is None or data.x_i is None:
        raise ValueError("both modality features are required")
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    with torch.inference_mode():
        components = model._analysis_call(x, edge_index)
        physical_edge_index = components["physical_edge_index"]
        if physical_edge_index.shape != edge_index.shape or not torch.equal(physical_edge_index, edge_index):
            raise AssertionError("analysis physical_edge_index differs from data edge_index")
        src, dst = physical_edge_index.detach().cpu().long().numpy()
        if np.any(src == dst):
            raise AssertionError("physical edge_index contains self-loops; allocation analysis requires none")
        weights_text = components["relation_weight_text"].detach().float().cpu().numpy()
        weights_visual = components["relation_weight_visual"].detach().float().cpu().numpy()
        if weights_text.shape != weights_visual.shape or weights_text.size != src.size:
            raise AssertionError("relation weights are not aligned with physical edge positions")

        incoming: dict[int, list[int]] = defaultdict(list)
        for edge_position, target in enumerate(dst.tolist()):
            incoming[int(target)].append(edge_position)
        rows: list[dict[str, Any]] = []
        degree_counts = {"degree_zero": 0, "degree_one": 0, "degree_gt1": 0, "degree_other": 0}
        duplicate_edge_count = 0
        probability_sum_errors: list[float] = []
        for node_id in range(int(data.num_nodes)):
            positions = incoming.get(node_id, [])
            neighbor_ids = src[positions]
            degree = int(np.unique(neighbor_ids).size)
            if degree == 0:
                degree_counts["degree_zero"] += 1
            elif degree == 1:
                degree_counts["degree_one"] += 1
            elif degree > 1:
                degree_counts["degree_gt1"] += 1
            else:
                degree_counts["degree_other"] += 1
            duplicate_edge_count += max(0, len(positions) - degree)
            metric = _node_allocation_metrics(
                neighbor_ids,
                weights_text[positions],
                weights_visual[positions],
                require_degree_gt1=True,
            )
            if metric is None:
                continue
            probability_sum_errors.append(float(metric["prob_sum_error"]))
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "node_id": node_id,
                    "degree": metric["degree"],
                    "tv_distance": _finite(metric["tv_distance"], "tv_distance"),
                    "top_neighbor_text": metric["top_neighbor_text"],
                    "top_neighbor_visual": metric["top_neighbor_visual"],
                    "top_neighbor_disagree": int(metric["top_neighbor_disagree"]),
                    "prob_sum_text": _finite(metric["prob_sum_text"], "prob_sum_text"),
                    "prob_sum_visual": _finite(metric["prob_sum_visual"], "prob_sum_visual"),
                    "prob_sum_error": _finite(metric["prob_sum_error"], "prob_sum_error"),
                    "physical_neighbor_ids": metric["physical_neighbor_ids"],
                    "weight_definition": WEIGHT_DEFINITION,
                    "support_definition": SUPPORT_DEFINITION,
                    "degree_filter": DEGREE_FILTER,
                    "neighbor_orientation": NEIGHBOR_ORIENTATION,
                }
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
            "num_nodes": int(data.num_nodes),
            "num_physical_edges": int(src.size),
            "num_rows_degree_gt1": len(rows),
            "degree_counts": degree_counts,
            "duplicate_physical_edge_records": duplicate_edge_count,
            "max_probability_sum_error": max(probability_sum_errors, default=0.0),
            "all_tv_in_bounds": bool(all(0.0 <= float(row["tv_distance"]) <= 1.0 for row in rows)),
            "weight_definition": WEIGHT_DEFINITION,
            "support_definition": SUPPORT_DEFINITION,
            "degree_filter": DEGREE_FILTER,
            "neighbor_orientation": NEIGHBOR_ORIENTATION,
        }
    del components, model, data, x, edge_index, physical_edge_index
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, metadata


def _summary_rows(detail_rows: list[dict[str, Any]], datasets: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_stats: dict[str, list[dict[str, float]]] = defaultdict(list)
    for dataset in datasets:
        for seed in seeds:
            selected = [row for row in detail_rows if row["dataset"] == dataset and int(row["seed"]) == seed]
            if not selected:
                continue
            tv = np.asarray([float(row["tv_distance"]) for row in selected], dtype=np.float64)
            disagree = np.asarray([int(row["top_neighbor_disagree"]) for row in selected], dtype=np.int64)
            stat = {
                "mean_tv": float(np.mean(tv)),
                "disagreement": float(100.0 * np.mean(disagree)),
            }
            seed_stats[dataset].append(stat)
            rows.append(
                {
                    "dataset": dataset,
                    "aggregation_level": "seed",
                    "seed": seed,
                    "num_nodes_degree_gt1": int(tv.size),
                    "mean_tv": stat["mean_tv"],
                    "median_tv": float(np.median(tv)),
                    "tv_q25": float(np.quantile(tv, 0.25)),
                    "tv_q75": float(np.quantile(tv, 0.75)),
                    "tv_iqr": float(np.quantile(tv, 0.75) - np.quantile(tv, 0.25)),
                    "tv_sd": float(np.std(tv)),
                    "tv_min": float(np.min(tv)),
                    "tv_max": float(np.max(tv)),
                    "top_neighbor_disagreement_percentage": stat["disagreement"],
                    "top_neighbor_disagreement_count": int(disagree.sum()),
                    "max_probability_sum_error": float(max(float(row["prob_sum_error"]) for row in selected)),
                    "tv_bounds_ok": int(bool(np.all((tv >= 0.0) & (tv <= 1.0)))),
                    "degree_filter": DEGREE_FILTER,
                    "support_definition": SUPPORT_DEFINITION,
                    "weight_definition": WEIGHT_DEFINITION,
                    "cross_seed_mean_tv": "",
                    "cross_seed_std_tv": "",
                    "cross_seed_mean_disagreement_percentage": "",
                    "cross_seed_std_disagreement_percentage": "",
                }
            )
        pooled = [row for row in detail_rows if row["dataset"] == dataset]
        if not pooled:
            continue
        tv = np.asarray([float(row["tv_distance"]) for row in pooled], dtype=np.float64)
        disagree = np.asarray([int(row["top_neighbor_disagree"]) for row in pooled], dtype=np.int64)
        stats = seed_stats[dataset]
        rows.append(
            {
                "dataset": dataset,
                "aggregation_level": "cross_seed_pooled",
                "seed": "all",
                "num_nodes_degree_gt1": int(tv.size),
                "mean_tv": float(np.mean(tv)),
                "median_tv": float(np.median(tv)),
                "tv_q25": float(np.quantile(tv, 0.25)),
                "tv_q75": float(np.quantile(tv, 0.75)),
                "tv_iqr": float(np.quantile(tv, 0.75) - np.quantile(tv, 0.25)),
                "tv_sd": float(np.std(tv)),
                "tv_min": float(np.min(tv)),
                "tv_max": float(np.max(tv)),
                "top_neighbor_disagreement_percentage": float(100.0 * np.mean(disagree)),
                "top_neighbor_disagreement_count": int(disagree.sum()),
                "max_probability_sum_error": float(max(float(row["prob_sum_error"]) for row in pooled)),
                "tv_bounds_ok": int(bool(np.all((tv >= 0.0) & (tv <= 1.0)))),
                "degree_filter": DEGREE_FILTER,
                "support_definition": SUPPORT_DEFINITION,
                "weight_definition": WEIGHT_DEFINITION,
                "cross_seed_mean_tv": float(np.mean([item["mean_tv"] for item in stats])) if stats else "",
                "cross_seed_std_tv": float(np.std([item["mean_tv"] for item in stats])) if stats else "",
                "cross_seed_mean_disagreement_percentage": float(np.mean([item["disagreement"] for item in stats])) if stats else "",
                "cross_seed_std_disagreement_percentage": float(np.std([item["disagreement"] for item in stats])) if stats else "",
            }
        )
    return rows


def _write_audit_summary(
    path: Path,
    *,
    detail_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    requested_datasets: list[str],
    requested_seeds: list[int],
) -> None:
    lines = [
        "# Figure 3 V3 candidate audit summary",
        "",
        "This candidate uses existing full/final checkpoints only; no training was started.",
        "The current Figure 3 V2 outputs, including the positive-Δu supplementary statistic, were not overwritten.",
        "",
        "## Definition audit",
        "",
        f"- Raw weights: `{WEIGHT_DEFINITION}`; normalized propagation operators are not used.",
        f"- Physical support: `{SUPPORT_DEFINITION}`.",
        f"- Neighbor orientation: `{NEIGHBOR_ORIENTATION}`.",
        f"- Main filter: `{DEGREE_FILTER}`; degree=1 nodes are excluded from detail and summary statistics.",
        "- Text and visual probabilities are normalized on the same physical neighbor IDs before TV and top-neighbor comparisons.",
        "",
        "## Correctness checks",
        "",
        "- Equal toy distributions give TV=0 and identical top neighbors.",
        "- Swapped toy weights give TV>0 and top-neighbor disagreement.",
        "- Per-node probability sums are checked against one; TV bounds are checked against [0, 1].",
        "",
        "## Available inputs",
        "",
        f"- Requested datasets: {', '.join(requested_datasets)}.",
        f"- Requested seeds: {', '.join(str(seed) for seed in requested_seeds)}.",
        f"- Successful checkpoint analyses: {len(analyses)}; detail rows: {len(detail_rows)}.",
    ]
    if analyses:
        lines.extend(["", "### Checkpoint audit", ""])
        for item in analyses:
            counts = item["degree_counts"]
            lines.append(
                f"- {item['dataset']} seed {item['seed']}: {item['num_rows_degree_gt1']} degree>1 nodes; "
                f"degree=1 fraction={counts['degree_one'] / max(item['num_nodes'], 1):.4f}; "
                f"max probability-sum error={item['max_probability_sum_error']:.2e}; "
                f"duplicate edge records={item['duplicate_physical_edge_records']}."
            )
    if summary_rows:
        lines.extend(["", "## Pooled candidate statistics", ""])
        for row in summary_rows:
            if row["aggregation_level"] != "cross_seed_pooled":
                continue
            lines.append(
                f"- {row['dataset']}: median TV={float(row['median_tv']):.6f}; "
                f"IQR=[{float(row['tv_q25']):.6f}, {float(row['tv_q75']):.6f}] "
                f"(width={float(row['tv_iqr']):.6f}); "
                f"top-neighbor disagreement={float(row['top_neighbor_disagreement_percentage']):.2f}%."
            )
            lines.append(
                f"  - Across seeds: mean TV={float(row['cross_seed_mean_tv']):.6f} ± {float(row['cross_seed_std_tv']):.6f}; "
                f"disagreement={float(row['cross_seed_mean_disagreement_percentage']):.2f}% ± "
                f"{float(row['cross_seed_std_disagreement_percentage']):.2f}% (SD across seed-level values)."
            )
    if missing:
        lines.extend(["", "## Missing inputs", ""])
        for item in missing:
            lines.append(f"- {item['dataset']} seed {item['seed']}: {', '.join(item.get('missing', []))}.")
    if failures:
        lines.extend(["", "## Analysis failures", ""])
        for item in failures:
            lines.append(f"- {item['dataset']} seed {item['seed']}: {item['error']}.")
    if not detail_rows:
        lines.extend(["", "No candidate rows were generated. The CSV files are valid but empty; rerun after the listed inputs are available."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.self_test:
        _run_self_tests()
        return 0

    datasets = list(dict.fromkeys(args.datasets))
    seeds = list(dict.fromkeys(args.seeds))
    checkpoint_root = args.checkpoint_root.resolve()
    output_dir = args.output_dir.resolve()
    present, missing = _planned_inputs(checkpoint_root, datasets, seeds)
    print(f"[figure3-v3-data] planned checkpoints: {len(present) + len(missing)}")
    print(f"[figure3-v3-data] available: {len(present)} | missing: {len(missing)}")
    for item in missing:
        print(f"[figure3-v3-data] missing {item['dataset']} seed={item['seed']}: {item['missing']}", file=sys.stderr)
    if args.dry_run:
        for item in present:
            print(f"[figure3-v3-data] ready {item['dataset']} seed={item['seed']}: {item['checkpoint']}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[figure3-v3-data] CUDA unavailable for {device}; falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")

    detail_rows: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in present:
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        try:
            rows, metadata = _extract_one(
                dataset=item["dataset"], seed=int(item["seed"]), checkpoint=checkpoint, device=device
            )
            detail_rows.extend(rows)
            analyses.append(metadata)
            print(f"[figure3-v3-data] analyzed {item['dataset']} seed={item['seed']} ({len(rows)} nodes)", flush=True)
        except Exception as exc:
            failure = {**item, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[figure3-v3-data] failed: {failure}", file=sys.stderr, flush=True)

    summaries = _summary_rows(detail_rows, datasets, seeds)
    _write_csv(output_dir / "panel_b_neighbor_allocation.csv", DETAIL_FIELDS, detail_rows)
    _write_csv(output_dir / "panel_b_neighbor_allocation_summary.csv", SUMMARY_FIELDS, summaries)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": _relative(Path(__file__)),
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "command": " ".join(sys.argv),
        "requested_datasets": datasets,
        "requested_seeds": seeds,
        "device_requested": str(args.device),
        "device_used": str(device),
        "weight_definition": WEIGHT_DEFINITION,
        "support_definition": SUPPORT_DEFINITION,
        "degree_filter": DEGREE_FILTER,
        "neighbor_orientation": NEIGHBOR_ORIENTATION,
        "available_inputs": present,
        "missing_inputs": missing,
        "successful_analyses": analyses,
        "failures": failures,
        "outputs": {
            "detail_csv": "panel_b_neighbor_allocation.csv",
            "summary_csv": "panel_b_neighbor_allocation_summary.csv",
            "audit_summary": "audit_summary.md",
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_audit_summary(
        output_dir / "audit_summary.md",
        detail_rows=detail_rows,
        summary_rows=summaries,
        analyses=analyses,
        missing=missing,
        failures=failures,
        requested_datasets=datasets,
        requested_seeds=seeds,
    )
    print(f"[figure3-v3-data] wrote {len(detail_rows)} detail rows to {output_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
