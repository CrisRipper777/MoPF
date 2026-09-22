#!/usr/bin/env python3
"""Generate final-clean, frozen-checkpoint source data for CoSI-MAG Figure 4.

This script performs inference-only mechanism analysis.  It loads the three
seed-specific checkpoints produced by ``cosi_mag_final`` under the final clean
benchmark, calls ``CoSIMAGFinal.analysis_intervention``, and writes compact
paper-source CSVs.  It never constructs an optimizer, calls ``backward``, or
starts a training/benchmark loop.
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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BENCHMARK_ROOT = ROOT / "outputs" / "cosi_mag_final_benchmark"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "paper_figures" / "figure4_rcmi"
NC_DATASETS = ("Movies", "Grocery")
LP_DATASET = "sports-copurchase"
SEEDS = (42, 43, 44)
INTERACTION_CONDITIONS = (
    ("Query-Collapse", "query_collapse"),
    ("Uniform-Attention", "uniform"),
    ("Interaction-Off", "off"),
)
RELATION_CONDITIONS = (
    ("Relation-Off", "off"),
    ("Relation-Shuffle", "shuffle"),
)
QUANTITIES = ("Text Attn.", "Visual Attn.", "Z", "Logits")
SHUFFLE_SEED = 73129
NUMERIC_ZERO = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate final-clean checkpoint inputs and print the audit without inference.",
    )
    return parser.parse_args()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _checkpoint_path(benchmark_root: Path, task: str, dataset: str, seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Only final benchmark seeds {SEEDS} are supported; got {seed}")
    run_id = SEEDS.index(seed) + 1
    return benchmark_root / task / dataset / "runs_42_43_44" / f"best_run{run_id}.pt"


def _load_manifest(benchmark_root: Path, task: str) -> dict[str, Any]:
    name = "sports-lp.json" if task == "lp" else "nc.json"
    path = benchmark_root / "manifests" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing final-clean manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != "cosi_mag_final":
        raise ValueError(f"Unexpected model in {path}: {payload.get('model')!r}")
    expected_protocol = "unified_full_graph_nc_v1" if task == "nc" else "unified_sampled_lp_v1"
    if payload.get("frozen_task_protocols", {}).get(task) != expected_protocol:
        raise ValueError(f"Unexpected {task} protocol in {path}")
    return payload


def _audit_checkpoints(benchmark_root: Path, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    requested = [("nc", dataset) for dataset in NC_DATASETS] + [("lp", LP_DATASET)]
    manifests = {task: _load_manifest(benchmark_root, task) for task, _ in requested}
    for task, dataset in requested:
        run_dir = benchmark_root / task / dataset / "runs_42_43_44"
        marker = run_dir / "complete.marker"
        if not marker.is_file():
            raise FileNotFoundError(f"Final-clean checkpoint set is incomplete: {marker}")
        manifest = manifests[task]
        if dataset not in manifest.get("datasets", []):
            raise ValueError(f"{dataset} is not listed in final-clean {task} manifest")
        for seed in seeds:
            checkpoint = _checkpoint_path(benchmark_root, task, dataset, seed)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing final-clean checkpoint: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid checkpoint payload: {checkpoint}")
            if int(payload.get("seed", -1)) != seed:
                raise ValueError(f"Checkpoint seed mismatch for {checkpoint}")
            if payload.get("task") != task:
                raise ValueError(f"Checkpoint task mismatch for {checkpoint}")
            config_path = checkpoint.parent / "resolved_config.yaml"
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing resolved config next to {checkpoint}")
            cfg = OmegaConf.load(config_path)
            if str(cfg.model.name) != "cosi_mag_final":
                raise ValueError(f"Unexpected model config in {config_path}")
            metrics = payload.get("metrics", {})
            rows.append(
                {
                    "task": task,
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_path": _relative(checkpoint),
                    "resolved_config_path": _relative(config_path),
                    "model": str(cfg.model.name),
                    "model_version": str(cfg.model.get("version", "")),
                    "protocol": str(cfg.task.protocol_version),
                    "selection": str(payload.get("selection", "")),
                    "epoch": int(payload.get("epoch", -1)),
                    "validation_metric": float(metrics.get("val_acc", metrics.get("val_mrr", float("nan")))),
                    "test_metric": float(metrics.get("test_acc", metrics.get("test_mrr", float("nan")))),
                    "benchmark_commit": manifest.get("git_commit_sha"),
                    "benchmark_worktree_dirty": bool(manifest.get("git_worktree_dirty", True)),
                    "checkpoint_sha256": _sha256(checkpoint),
                }
            )
    if any(row["benchmark_worktree_dirty"] for row in rows):
        raise ValueError("Final-clean benchmark manifest reports a dirty worktree")
    return rows


def _load_context(checkpoint: Path, task: str, device: torch.device):
    from src.data import load_mag_data
    from src.models import LinkPredictor, build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(checkpoint.parent / "resolved_config.yaml")
    data = load_mag_data(cfg, task, int(cfg.seed))
    data_info = payload["data_info"]
    model = build_model(cfg, data_info).to(device).eval()
    model.load_state_dict(payload["model_state"], strict=True)
    head: nn.Module
    projection: nn.Module | None = None
    if task == "nc":
        head = nn.Linear(model.out_dim, int(data.num_classes)).to(device).eval()
        head.load_state_dict(payload["head_state"], strict=True)
    else:
        decoder_cfg = cfg.task.decoder
        proj_dim = int(decoder_cfg.get("proj_dim", 0) or 0)
        if proj_dim > 0:
            projection = nn.Linear(model.out_dim, proj_dim).to(device).eval()
            if payload.get("proj_state") is None:
                raise ValueError(f"LP checkpoint has no projection state: {checkpoint}")
            projection.load_state_dict(payload["proj_state"], strict=True)
        predictor_in_dim = proj_dim if projection is not None else model.out_dim
        head = LinkPredictor(
            in_dim=predictor_in_dim,
            hidden_dim=int(decoder_cfg.hidden_dim),
            num_layers=int(decoder_cfg.num_layers),
            dropout=float(decoder_cfg.dropout),
        ).to(device).eval()
        head.load_state_dict(payload["head_state"], strict=True)
    return cfg, data, model, head, projection


def _call(model: nn.Module, data: Any, device: torch.device, **kwargs: Any) -> dict[str, Any]:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    return model.analysis_intervention(x, edge_index, **kwargs)


def _r_attn(attention: torch.Tensor, max_order: int) -> np.ndarray:
    if attention.ndim != 3 or attention.shape[-1] != max_order + 1:
        raise ValueError(f"Unexpected attention shape: {tuple(attention.shape)}")
    profile = attention.float().mean(dim=1)
    order = torch.arange(max_order + 1, device=attention.device, dtype=profile.dtype)
    order = order / float(max_order)
    values = (profile * order.unsqueeze(0)).sum(dim=-1)
    values = values.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("r_attn is non-finite or outside [0, 1]")
    return values


def _mean_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    value = torch.mean(torch.abs(left.float() - right.float())).item()
    if not np.isfinite(value):
        raise ValueError("Intervention impact is non-finite")
    return float(value)


def _nc_logits(z: torch.Tensor, classifier: nn.Module) -> torch.Tensor:
    return classifier(z)


def _lp_logits(
    z: torch.Tensor,
    predictor: nn.Module,
    projection: nn.Module | None,
    edge_split: Any,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    if projection is not None:
        z = projection(z)
    source = edge_split["source_node"].to(device=device, dtype=torch.long)
    target = edge_split["target_node"].to(device=device, dtype=torch.long)
    target_neg = edge_split["target_node_neg"].to(device=device, dtype=torch.long)
    chunks: list[torch.Tensor] = []
    for start in range(0, int(source.numel()), batch_size):
        stop = min(start + batch_size, int(source.numel()))
        z_src = z[source[start:stop]]
        pos = predictor.score_pairs(z_src, z[target[start:stop]])
        neg = predictor.score_pairs(z_src.unsqueeze(1), z[target_neg[start:stop]])
        chunks.append(torch.cat([pos, neg.reshape(-1)], dim=0).detach().cpu())
    values = torch.cat(chunks, dim=0)
    if not torch.isfinite(values).all():
        raise ValueError("LP scorer logits are non-finite")
    return values


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.quantile(array, [0.25, 0.50, 0.75])
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "n_seeds": int(array.size),
    }


def _run_seed(
    task: str,
    dataset: str,
    seed: int,
    checkpoint: Path,
    device: torch.device,
    panel_a_values: dict[tuple[str, str, int], np.ndarray],
    panel_b_rows: list[dict[str, Any]],
    panel_c_rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
) -> None:
    cfg, data, model, head, projection = _load_context(checkpoint, task, device)
    with torch.inference_mode():
        normal = _call(model, data, device, relation="normal", interaction="normal")
        max_order = int(cfg.model.max_order)
        if task == "nc":
            panel_a_values[(dataset, "text", seed)] = _r_attn(normal["attention_text"], max_order)
            panel_a_values[(dataset, "visual", seed)] = _r_attn(normal["attention_visual"], max_order)

        normal_z = normal["z"]
        shape_rows.append(
            {
                "task": task,
                "dataset": dataset,
                "seed": seed,
                "num_nodes": int(normal_z.shape[0]),
                "hidden_dim": int(normal_z.shape[1]),
                "attention_shape": str(tuple(normal["attention_text"].shape)),
                "message_edges": int(data.edge_index.shape[1]),
            }
        )

        if task == "nc":
            interaction_logits_normal = _nc_logits(normal_z, head)
            relation_normal_logits = interaction_logits_normal
        else:
            relation_normal_logits = _lp_logits(
                normal_z,
                head,
                projection,
                data.edge_split.test,
                device,
            )

        if task == "nc":
            for label, intervention in INTERACTION_CONDITIONS:
                changed = _call(
                    model,
                    data,
                    device,
                    relation="normal",
                    interaction=intervention,
                )
                z_mae = _mean_abs(normal_z, changed["z"])
                panel_b_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "condition": label,
                        "z_mae": z_mae,
                    }
                )

        relation_outputs: dict[str, dict[str, Any]] = {}
        for label, intervention in RELATION_CONDITIONS:
            kwargs: dict[str, Any] = {"relation": intervention, "interaction": "normal"}
            if intervention == "shuffle":
                kwargs["permutation_seed"] = SHUFFLE_SEED
            relation_outputs[label] = _call(model, data, device, **kwargs)

        for label in ("Relation-Off", "Relation-Shuffle"):
            changed = relation_outputs[label]
            relation_logits = (
                _nc_logits(changed["z"], head)
                if task == "nc"
                else _lp_logits(changed["z"], head, projection, data.edge_split.test, device)
            )
            impacts = {
                "Text Attn.": _mean_abs(normal["attention_text"], changed["attention_text"]),
                "Visual Attn.": _mean_abs(normal["attention_visual"], changed["attention_visual"]),
                "Z": _mean_abs(normal_z, changed["z"]),
                "Logits": _mean_abs(relation_normal_logits, relation_logits),
            }
            relation_outputs[label]["_impacts"] = impacts

        for quantity in QUANTITIES:
            off = float(relation_outputs["Relation-Off"]["_impacts"][quantity])
            shuffle = float(relation_outputs["Relation-Shuffle"]["_impacts"][quantity])
            if not np.isfinite(off) or not np.isfinite(shuffle):
                raise ValueError(f"Non-finite relation impact for {dataset}/{seed}/{quantity}")
            if off <= NUMERIC_ZERO:
                raise ValueError(
                    f"Relation-Off impact is zero/near-zero for {dataset}/{seed}/{quantity}; "
                    "ratio is intentionally not stabilized with an arbitrary epsilon."
                )
            ratio = shuffle / off
            excess = 100.0 * (ratio - 1.0)
            panel_c_rows.append(
                {
                    "dataset": "Sports-LP" if task == "lp" else dataset,
                    "seed": seed,
                    "quantity": quantity,
                    "off_impact": off,
                    "shuffle_impact": shuffle,
                    "shuffle_over_off": ratio,
                    "excess_impact_pct": excess,
                }
            )

    del model, head, projection, data
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _build_report_rows(
    panel_a_values: dict[tuple[str, str, int], np.ndarray],
    panel_b_rows: list[dict[str, Any]],
    panel_c_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    panel_a_node_rows: list[dict[str, Any]] = []
    panel_a_summary_rows: list[dict[str, Any]] = []
    for dataset in NC_DATASETS:
        for modality in ("text", "visual"):
            values = [panel_a_values[(dataset, modality, seed)] for seed in SEEDS]
            if not all(array.shape == values[0].shape for array in values):
                raise ValueError(f"Node counts differ across seeds for {dataset}/{modality}")
            averaged = np.stack(values, axis=0).mean(axis=0)
            for node, value in enumerate(averaged):
                panel_a_node_rows.append(
                    {
                        "dataset": dataset,
                        "node": node,
                        "modality": modality,
                        "r_attn_mean_across_seeds": float(value),
                        "n_seeds": len(SEEDS),
                    }
                )
            stats = _summary(averaged.tolist())
            stats["n_seeds"] = len(SEEDS)
            panel_a_summary_rows.append(
                {"dataset": dataset, "modality": modality, "n_nodes": int(averaged.size), **stats}
            )

    panel_b_summary_rows: list[dict[str, Any]] = []
    by_b: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in panel_b_rows:
        by_b[(row["dataset"], row["condition"])].append(float(row["z_mae"]))
    for dataset in NC_DATASETS:
        for condition, _ in INTERACTION_CONDITIONS:
            panel_b_summary_rows.append(
                {"dataset": dataset, "condition": condition, **_summary(by_b[(dataset, condition)])}
            )

    panel_b_audit_rows: list[dict[str, Any]] = []
    for dataset in NC_DATASETS:
        by_seed = {
            seed: {row["condition"]: float(row["z_mae"]) for row in panel_b_rows if row["dataset"] == dataset and row["seed"] == seed}
            for seed in SEEDS
        }
        for seed in SEEDS:
            values = by_seed[seed]
            holds = values["Query-Collapse"] < values["Uniform-Attention"] < values["Interaction-Off"]
            panel_b_audit_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "QC": values["Query-Collapse"],
                    "Uniform": values["Uniform-Attention"],
                    "Off": values["Interaction-Off"],
                    "ordering_holds": bool(holds),
                }
            )

    panel_c_summary_rows: list[dict[str, Any]] = []
    by_c: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in panel_c_rows:
        by_c[(row["dataset"], row["quantity"])].append(row)
    for dataset in ("Movies", "Grocery", "Sports-LP"):
        for quantity in QUANTITIES:
            rows = by_c[(dataset, quantity)]
            excess = [float(row["excess_impact_pct"]) for row in rows]
            off = [float(row["off_impact"]) for row in rows]
            shuffle = [float(row["shuffle_impact"]) for row in rows]
            panel_c_summary_rows.append(
                {
                    "dataset": dataset,
                    "quantity": quantity,
                    "mean_off_impact": float(np.mean(off)),
                    "mean_shuffle_impact": float(np.mean(shuffle)),
                    "mean_excess_impact_pct": float(np.mean(excess)),
                    "median_excess_impact_pct": float(np.median(excess)),
                    "q25_excess_impact_pct": float(np.quantile(excess, 0.25)),
                    "q75_excess_impact_pct": float(np.quantile(excess, 0.75)),
                    "n_seeds": len(rows),
                    "all_seeds_shuffle_gt_off": bool(all(value > 0.0 for value in excess)),
                }
            )
    return panel_a_node_rows, panel_a_summary_rows, panel_b_summary_rows, panel_b_audit_rows, panel_c_summary_rows


def main() -> None:
    args = _parse_args()
    benchmark_root = args.benchmark_root if args.benchmark_root.is_absolute() else ROOT / args.benchmark_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    seeds = tuple(int(seed) for seed in args.seeds)
    if seeds != SEEDS:
        raise ValueError(f"Figure 4 final source requires all seeds {SEEDS}; got {seeds}")
    audit_rows = _audit_checkpoints(benchmark_root, seeds)
    print(json.dumps({"final_clean_checkpoints": audit_rows}, indent=2, ensure_ascii=False))
    if args.audit_only:
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_a_values: dict[tuple[str, str, int], np.ndarray] = {}
    panel_b_rows: list[dict[str, Any]] = []
    panel_c_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []

    for task, dataset in [("nc", dataset) for dataset in NC_DATASETS] + [("lp", LP_DATASET)]:
        for seed in seeds:
            checkpoint = _checkpoint_path(benchmark_root, task, dataset, seed)
            print(f"[figure4-analysis] frozen inference {task}/{dataset}/seed={seed}", flush=True)
            _run_seed(
                task,
                dataset,
                seed,
                checkpoint,
                device,
                panel_a_values,
                panel_b_rows,
                panel_c_rows,
                shape_rows,
            )

    panel_a_node_rows, panel_a_summary_rows, panel_b_summary_rows, panel_b_audit_rows, panel_c_summary_rows = _build_report_rows(
        panel_a_values, panel_b_rows, panel_c_rows
    )
    _write_csv(
        output_dir / "panel_a_node_mean_r_attn.csv",
        ["dataset", "node", "modality", "r_attn_mean_across_seeds", "n_seeds"],
        panel_a_node_rows,
    )
    _write_csv(
        output_dir / "panel_a_summary.csv",
        ["dataset", "modality", "n_nodes", "mean", "std_population", "q25", "median", "q75", "n_seeds"],
        panel_a_summary_rows,
    )
    _write_csv(
        output_dir / "panel_b_seed_z_mae.csv",
        ["dataset", "seed", "condition", "z_mae"],
        panel_b_rows,
    )
    _write_csv(
        output_dir / "panel_b_summary.csv",
        ["dataset", "condition", "mean", "std_population", "q25", "median", "q75", "n_seeds"],
        panel_b_summary_rows,
    )
    _write_csv(
        output_dir / "panel_b_ordering_audit.csv",
        ["dataset", "seed", "QC", "Uniform", "Off", "ordering_holds"],
        panel_b_audit_rows,
    )
    _write_csv(
        output_dir / "panel_c_seed_excess_impact.csv",
        ["dataset", "seed", "quantity", "off_impact", "shuffle_impact", "shuffle_over_off", "excess_impact_pct"],
        panel_c_rows,
    )
    _write_csv(
        output_dir / "panel_c_summary.csv",
        [
            "dataset", "quantity", "mean_off_impact", "mean_shuffle_impact",
            "mean_excess_impact_pct", "median_excess_impact_pct",
            "q25_excess_impact_pct", "q75_excess_impact_pct", "n_seeds",
            "all_seeds_shuffle_gt_off",
        ],
        panel_c_summary_rows,
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": _relative(Path(__file__)),
        "benchmark_root": _relative(benchmark_root),
        "output_dir": _relative(output_dir),
        "model": "cosi_mag_final",
        "seeds": list(seeds),
        "device": str(device),
        "checkpoint_audit": audit_rows,
        "data_shapes": shape_rows,
        "relation_shuffle_seed": SHUFFLE_SEED,
        "relation_zero_guard": NUMERIC_ZERO,
        "lp_protocol": {
            "task_protocol": "unified_sampled_lp_v1",
            "message_graph": "data.edge_index loaded from train split only",
            "scorer_logits_split": "test",
            "scorer": "frozen LinkPredictor plus frozen projection from each seed checkpoint",
        },
        "nc_logit_scope": "all nodes",
        "training_started": False,
        "optimizer_created": False,
        "backward_called": False,
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit_at_generation": _git_value("rev-parse", "HEAD"),
    }
    (output_dir / "final_clean_mechanism_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[figure4-analysis] wrote final-clean source CSVs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
