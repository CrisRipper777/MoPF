#!/usr/bin/env python3
"""Generate source data for Figure 3 mechanism analyses.

This script is read-only with respect to model training: it loads existing NC
checkpoints and frozen dataset features, calls the model's analysis interface,
and writes Figure 3 CSVs under ``outputs/figure3_mechanism``.  Missing
checkpoints are recorded and do not cause an implicit training run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
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

DEFAULT_OUTPUT = ROOT / "outputs" / "figure3_mechanism"
DEFAULT_BENCHMARK_ROOT = ROOT / "outputs" / "cosi_mag_final_benchmark"
DEFAULT_ABLATION_ROOT = ROOT / "outputs" / "cosi_mag_final_ablation"
DATASETS = ("Movies", "Grocery")
SEEDS = (42, 43, 44)
VARIANTS = ("full", "no_semantic_anchor")
MODALITIES = ("text", "visual")
HOPS = (0, 1, 2, 3)

PANEL_A_FIELDS = (
    "dataset",
    "seed",
    "model_variant",
    "num_nodes",
    "num_physical_edges",
    "statistic",
    "relation_weight",
    "semantic_similarity",
    "spearman_rho",
    "spearman_pvalue",
    "spearman_status",
)
PANEL_B_FIELDS = (
    "dataset",
    "seed",
    "model_variant",
    "modality",
    "node_id",
    "degree",
    "raw_compatibility",
    "calibrated_compatibility",
    "delta_compatibility",
)
PANEL_C_FIELDS = (
    "dataset",
    "seed",
    "model_variant",
    "modality",
    "hop",
    "num_nodes",
    "mean_cosine",
    "sd_cosine_across_nodes",
    "median_cosine",
    "cosine_q25",
    "cosine_q75",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--device", default="cpu", help="cpu or cuda device used for analysis")
    parser.add_argument("--dry-run", action="store_true", help="list inputs and missing files without loading data")
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


def _checkpoint_path(
    benchmark_root: Path,
    ablation_root: Path,
    dataset: str,
    seed: int,
    variant: str,
) -> Path:
    run_id = SEEDS.index(seed) + 1
    if variant == "full":
        root = benchmark_root
    else:
        root = ablation_root / variant
    return root / "nc" / dataset / "runs_42_43_44" / f"best_run{run_id}.pt"


def _resolved_config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "resolved_config.json"


def _read_config(checkpoint_path: Path) -> Any:
    config_path = _resolved_config_path(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"missing resolved config next to checkpoint: {config_path}")
    return OmegaConf.create(json.loads(config_path.read_text(encoding="utf-8")))


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    from src.data import load_mag_data
    from src.models import build_model

    cfg = _read_config(checkpoint_path)
    data = load_mag_data(cfg, "nc", int(cfg.seed))
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError(f"checkpoint has no model_state: {checkpoint_path}")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return cfg, data, model, payload


def _edge_cosine(feature: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(feature.float(), p=2, dim=1, eps=1e-12)
    src, dst = edge_index
    return (normalized[src] * normalized[dst]).sum(dim=1)


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite {name}: {value}")
    return value


def _write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, str]:
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None, None, "undefined_constant_or_short_vector"
    result = spearmanr(x, y, nan_policy="raise")
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    if not (np.isfinite(rho) and np.isfinite(pvalue)):
        return None, None, "undefined_nonfinite"
    return rho, pvalue, "ok"


def _raw_modality_features(data: Any) -> dict[str, torch.Tensor]:
    if data.x_t is None or data.x_i is None:
        raise ValueError("both text and visual feature tensors are required")
    return {"text": data.x_t, "visual": data.x_i}


def _extract_one(
    *,
    dataset: str,
    seed: int,
    variant: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg, data, model, payload = _load_checkpoint_model(checkpoint_path, device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    raw_features = {key: value.to(device) for key, value in _raw_modality_features(data).items()}

    with torch.inference_mode():
        # This is the existing read-only analysis interface.  One call supplies
        # both the relation weights and the pre-interaction multi-hop states.
        components = model._analysis_call(x, edge_index)

        raw_sim = {
            modality: _edge_cosine(raw_features[modality], edge_index)
            for modality in MODALITIES
        }
        relation_weights = {
            "text": components["relation_weight_text"],
            "visual": components["relation_weight_visual"],
        }
        states = {
            "text": components["states_text"],
            "visual": components["states_visual"],
        }
        anchors = {
            "text": components["h0_text"],
            "visual": components["h0_visual"],
        }

        src, dst = edge_index
        num_nodes = int(data.num_nodes)
        edge_count = int(edge_index.size(1))
        panel_a_rows: list[dict[str, Any]] = []
        for calibrated_modality, semantic_modality in (
            ("text", "text"),
            ("text", "visual"),
            ("visual", "visual"),
            ("visual", "text"),
        ):
            rho, pvalue, status = _spearman(
                _as_numpy(relation_weights[calibrated_modality]),
                _as_numpy(raw_sim[semantic_modality]),
            )
            panel_a_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model_variant": variant,
                    "num_nodes": num_nodes,
                    "num_physical_edges": edge_count,
                    "statistic": "spearman_rho",
                    "relation_weight": f"W^{calibrated_modality[:1].upper()}",
                    "semantic_similarity": f"S^{semantic_modality[:1].upper()}",
                    "spearman_rho": "" if rho is None else rho,
                    "spearman_pvalue": "" if pvalue is None else pvalue,
                    "spearman_status": status,
                }
            )

        degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        panel_b_rows: list[dict[str, Any]] = []
        for modality in MODALITIES:
            semantic = raw_sim[modality]
            weight = relation_weights[modality].float()
            raw_sum = torch.zeros(num_nodes, dtype=torch.float32, device=device)
            calibrated_sum = torch.zeros_like(raw_sum)
            calibrated_denominator = torch.zeros_like(raw_sum)
            raw_sum.index_add_(0, src, semantic.float())
            calibrated_sum.index_add_(0, src, weight * semantic.float())
            calibrated_denominator.index_add_(0, src, weight)
            valid = degree > 0
            raw_compatibility = raw_sum[valid] / degree[valid]
            calibrated_compatibility = calibrated_sum[valid] / calibrated_denominator[valid].clamp_min(1e-12)
            delta = calibrated_compatibility - raw_compatibility
            valid_nodes = torch.nonzero(valid, as_tuple=False).flatten().cpu().tolist()
            for node_id, raw_value, calibrated_value, delta_value in zip(
                valid_nodes,
                _as_numpy(raw_compatibility),
                _as_numpy(calibrated_compatibility),
                _as_numpy(delta),
            ):
                panel_b_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_variant": variant,
                        "modality": modality,
                        "node_id": int(node_id),
                        "degree": int(degree[node_id].item()),
                        "raw_compatibility": _finite_float(raw_value, "raw_compatibility"),
                        "calibrated_compatibility": _finite_float(calibrated_value, "calibrated_compatibility"),
                        "delta_compatibility": _finite_float(delta_value, "delta_compatibility"),
                    }
                )

        panel_c_rows: list[dict[str, Any]] = []
        # Panel (c) uses the state bank before the cross-order interaction, so
        # the quantity isolates repeated graph propagation plus the anchor.
        for modality in MODALITIES:
            for hop, state in enumerate(states[modality]):
                cosine = torch.nn.functional.cosine_similarity(
                    state.float(), anchors[modality].float(), dim=-1, eps=1e-12
                )
                values = _as_numpy(cosine)
                panel_c_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model_variant": variant,
                        "modality": modality,
                        "hop": hop,
                        "num_nodes": int(values.size),
                        "mean_cosine": _finite_float(np.mean(values), "mean_cosine"),
                        "sd_cosine_across_nodes": _finite_float(np.std(values), "sd_cosine_across_nodes"),
                        "median_cosine": _finite_float(np.median(values), "median_cosine"),
                        "cosine_q25": _finite_float(np.quantile(values, 0.25), "cosine_q25"),
                        "cosine_q75": _finite_float(np.quantile(values, 0.75), "cosine_q75"),
                    }
                )

    checkpoint_meta = {
        "dataset": dataset,
        "seed": seed,
        "model_variant": variant,
        "checkpoint": _relative(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "resolved_config": _relative(_resolved_config_path(checkpoint_path)),
        "checkpoint_selection": payload.get("selection"),
        "best_epoch": payload.get("epoch"),
        "model_name": str(cfg.model.name),
        "anchor_alpha": float(cfg.model.multihop_anchor_alpha),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "zero_degree_nodes": int((degree == 0).sum().item()),
    }
    if variant != "full":
        # Panel (a)/(b) are MRC evidence for the full model.  The
        # no-semantic-anchor checkpoint is intentionally retained only for
        # Panel (c), whose contrast is the SMP ablation.
        panel_a_rows = []
        panel_b_rows = []
    del components, model, data, x, edge_index, raw_features
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return panel_a_rows, panel_b_rows, panel_c_rows, checkpoint_meta


def _planned_inputs(
    benchmark_root: Path,
    ablation_root: Path,
    datasets: list[str],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            for variant in VARIANTS:
                checkpoint = _checkpoint_path(benchmark_root, ablation_root, dataset, seed, variant)
                item = {
                    "dataset": dataset,
                    "seed": seed,
                    "model_variant": variant,
                    "checkpoint": _relative(checkpoint),
                }
                if checkpoint.is_file() and _resolved_config_path(checkpoint).is_file():
                    present.append(item)
                else:
                    item["missing"] = [
                        name for name, path in (
                            ("checkpoint", checkpoint),
                            ("resolved_config", _resolved_config_path(checkpoint)),
                        ) if not path.is_file()
                    ]
                    missing.append(item)
    return present, missing


def _next_commands(
    missing: list[dict[str, Any]],
    *,
    ablation_root: Path,
) -> list[str]:
    commands: list[str] = []
    for item in missing:
        if item["model_variant"] != "no_semantic_anchor":
            continue
        dataset = item["dataset"]
        commands.append(
            "PYTHONPATH=. python scripts/run_cosi_mag_ablation.py "
            f"--variants no_semantic_anchor --datasets {dataset} --device cuda:0"
        )
    return list(dict.fromkeys(commands))


def _write_summary(
    path: Path,
    *,
    panel_a_rows: list[dict[str, Any]],
    panel_b_rows: list[dict[str, Any]],
    panel_c_rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    lines = [
        "# Figure 3 mechanism summary",
        "",
        "This summary is generated from existing checkpoints only; no training is started.",
        "",
        f"- Panel (a) rows: {len(panel_a_rows)} checkpoint-level correlations.",
        f"- Panel (b) rows: {len(panel_b_rows)} node-level compatibility records.",
        f"- Panel (c) rows: {len(panel_c_rows)} checkpoint-level retention summaries.",
        f"- Missing inputs: {len(missing)}.",
        "",
    ]
    if panel_a_rows:
        lines.extend(["## Panel (a)", ""])
        for dataset in sorted({row["dataset"] for row in panel_a_rows}):
            values = {}
            for row in panel_a_rows:
                if row["dataset"] == dataset and row["spearman_rho"] != "":
                    values.setdefault((row["relation_weight"], row["semantic_similarity"]), []).append(
                        float(row["spearman_rho"])
                    )
            parts = [
                f"{left} vs {right}: {np.mean(value):.3f} ± {np.std(value):.3f}"
                for (left, right), value in sorted(values.items())
            ]
            lines.append(f"- {dataset}: " + ("; ".join(parts) if parts else "no defined correlations"))
        lines.append("")
    if panel_b_rows:
        lines.extend(["## Panel (b)", ""])
        for dataset in sorted({row["dataset"] for row in panel_b_rows}):
            for modality in MODALITIES:
                values = [
                    float(row["delta_compatibility"])
                    for row in panel_b_rows
                    if row["dataset"] == dataset and row["modality"] == modality
                ]
                if values:
                    lines.append(
                        f"- {dataset}-{modality}: median Δu={np.median(values):.4f}; "
                        f"positive fraction={np.mean(np.asarray(values) > 0):.3f}."
                    )
        lines.append("")
    if panel_c_rows:
        lines.extend(["## Panel (c)", ""])
        for dataset in sorted({row["dataset"] for row in panel_c_rows}):
            for modality in MODALITIES:
                full = [
                    float(row["mean_cosine"])
                    for row in panel_c_rows
                    if row["dataset"] == dataset and row["modality"] == modality and row["model_variant"] == "full"
                ]
                no_anchor = [
                    float(row["mean_cosine"])
                    for row in panel_c_rows
                    if row["dataset"] == dataset and row["modality"] == modality and row["model_variant"] == "no_semantic_anchor"
                ]
                if full and no_anchor:
                    lines.append(
                        f"- {dataset}-{modality}: full mean retention across hops="
                        f"{np.mean(full):.4f}; w/o anchor={np.mean(no_anchor):.4f}."
                    )
        lines.append("")
    if missing:
        lines.extend(["## Missing inputs", ""])
        for item in missing:
            detail = ", ".join(item.get("missing", [])) or item.get("error", "unspecified failure")
            lines.append(f"- {item['dataset']} seed {item['seed']} {item['model_variant']}: {detail}.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    seeds = list(dict.fromkeys(args.seeds))
    benchmark_root = args.benchmark_root.resolve()
    ablation_root = args.ablation_root.resolve()
    output_dir = args.output_dir.resolve()
    present, missing = _planned_inputs(benchmark_root, ablation_root, datasets, seeds)

    print(f"[figure3-data] output: {output_dir}")
    print(f"[figure3-data] planned checkpoints: {len(present) + len(missing)}")
    print(f"[figure3-data] available checkpoints: {len(present)}")
    print(f"[figure3-data] missing checkpoints/configs: {len(missing)}")
    for item in missing:
        print(
            f"[figure3-data] missing {item['dataset']} seed={item['seed']} "
            f"variant={item['model_variant']}: {', '.join(item['missing'])}",
            file=sys.stderr,
        )
    if args.dry_run:
        for item in present:
            print(
                f"[figure3-data] ready {item['dataset']} seed={item['seed']} "
                f"variant={item['model_variant']}: {item['checkpoint']}"
            )
        for command in _next_commands(missing, ablation_root=ablation_root):
            print(f"[figure3-data] follow-up: {command}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_a_rows: list[dict[str, Any]] = []
    panel_b_rows: list[dict[str, Any]] = []
    panel_c_rows: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    requested_device = str(args.device)
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(
            f"[figure3-data] requested {requested_device}, but CUDA is unavailable; falling back to CPU",
            file=sys.stderr,
        )
        device = torch.device("cpu")
    for item in present:
        checkpoint = Path(item["checkpoint"])
        # The planned path is relative to ROOT for manifest readability.
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        try:
            a_rows, b_rows, c_rows, metadata = _extract_one(
                dataset=item["dataset"],
                seed=int(item["seed"]),
                variant=item["model_variant"],
                checkpoint_path=checkpoint,
                device=device,
            )
            panel_a_rows.extend(a_rows)
            panel_b_rows.extend(b_rows)
            panel_c_rows.extend(c_rows)
            analyses.append(metadata)
            print(
                f"[figure3-data] analyzed {item['dataset']} seed={item['seed']} "
                f"variant={item['model_variant']}",
                flush=True,
            )
        except Exception as exc:  # graceful fallback is part of the data contract
            failure = {**item, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[figure3-data] failed: {failure}", file=sys.stderr, flush=True)

    _write_csv(output_dir / "panel_a_relation_alignment.csv", PANEL_A_FIELDS, panel_a_rows)
    _write_csv(output_dir / "panel_b_neighborhood_compatibility.csv", PANEL_B_FIELDS, panel_b_rows)
    _write_csv(output_dir / "panel_c_semantic_retention.csv", PANEL_C_FIELDS, panel_c_rows)
    next_commands = _next_commands(missing, ablation_root=ablation_root)
    manifest = {
        "figure": "Figure 3",
        "purpose": "Mechanism analysis of relation calibration and semantic preservation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": _relative(Path(__file__)),
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit_sha": _git_value("rev-parse", "HEAD"),
        "git_worktree_status": _git_value("status", "--short"),
        "datasets": datasets,
        "seeds": seeds,
        "requested_device": requested_device,
        "device": str(device),
        "checkpoint_roots": {
            "full": _relative(benchmark_root),
            "no_semantic_anchor": _relative(ablation_root / "no_semantic_anchor"),
        },
        "panel_definitions": {
            "a": "Full-model Spearman rho between MRC relation weights and raw input-feature edge cosine",
            "b": "Full-model node-wise weighted-minus-unweighted incident-neighborhood raw semantic cosine",
            "c": "Mean node cosine between each pre-interaction multi-hop state and h0",
        },
        "available_inputs": present,
        "missing_inputs": missing,
        "analysis_failures": failures,
        "checkpoint_analyses": analyses,
        "row_counts": {
            "panel_a": len(panel_a_rows),
            "panel_b": len(panel_b_rows),
            "panel_c": len(panel_c_rows),
        },
        "next_commands": next_commands,
        "outputs": {
            "panel_a": "panel_a_relation_alignment.csv",
            "panel_b": "panel_b_neighborhood_compatibility.csv",
            "panel_c": "panel_c_semantic_retention.csv",
            "summary": "figure3_mechanism_summary.md",
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_summary(
        output_dir / "figure3_mechanism_summary.md",
        panel_a_rows=panel_a_rows,
        panel_b_rows=panel_b_rows,
        panel_c_rows=panel_c_rows,
        missing=[*missing, *failures],
    )
    print(f"[figure3-data] wrote {output_dir / 'run_manifest.json'}")
    print(f"[figure3-data] wrote {output_dir / 'figure3_mechanism_summary.md'}")
    for command in next_commands:
        print(f"[figure3-data] follow-up: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
