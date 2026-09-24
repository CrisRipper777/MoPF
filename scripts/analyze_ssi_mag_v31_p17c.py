#!/usr/bin/env python3
"""Post-hoc P1.7c sequential attribution analysis.

The analyzer never trains or selects a checkpoint.  It loads validation-
selected NC checkpoints and their saved heads, validates attention/simplex
and node ordering, and writes descriptive performance/mechanism evidence for
B (r2_only) and AB (r1_r2).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_ssi_mag_v31_p17b as p17b  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
CONTROL_TO_VARIANT = {"r2_only": "B", "r1_r2": "AB"}
EPS = 1.0e-12


def _np(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


def _flat(value: Any) -> np.ndarray:
    return _np(value).reshape(-1)


def _safe(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, list):
        return all(_safe(item) for item in value)
    if isinstance(value, dict):
        return all(_safe(item) for item in value.values())
    return True


def _stats(value: Any) -> dict[str, float]:
    return p17b._stats(value)


def _corr(left: Any, right: Any) -> tuple[float, float]:
    a, b = _flat(left), _flat(right)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) <= EPS or np.std(b) <= EPS:
        return math.nan, math.nan
    return float(pearsonr(a, b).statistic), float(spearmanr(a, b).statistic)


def _covariance(term: Any, total: Any) -> float:
    x, y = _flat(term), _flat(total)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    variance = float(np.var(y)) if y.size else 0.0
    if y.size < 3 or variance <= EPS:
        return math.nan
    return float(np.mean((x - x.mean()) * (y - y.mean())) / variance)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_dir(root: Path, dataset: str, seed: int, group: str) -> Path:
    return root / dataset / group / f"seed{seed}"


def _load_run(root: Path, dataset: str, seed: int, group: str, device: torch.device):
    run = _run_dir(root, dataset, seed, group)
    required = [run / name for name in ("best.pt", "complete.marker", "resolved_config.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint artifacts: " + ", ".join(missing))
    marker = json.loads((run / "complete.marker").read_text())
    if marker.get("status") != "complete" or marker.get("task") != "nc" or int(marker.get("seed", -1)) != seed:
        raise ValueError(f"invalid NC completion marker: {run}")
    cfg = OmegaConf.create(json.loads((run / "resolved_config.json").read_text()))
    if str(cfg.task.name) != "nc" or str(cfg.ablation) != "full":
        raise ValueError(f"unexpected task/ablation in {run}")
    data = load_mag_data(cfg, "nc", seed)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    payload = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"checkpoint selection/schema mismatch in {run}")
    model = build_model(cfg, info).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    head.eval()
    return run, cfg, data, model, head


def _data_signature(data) -> tuple[Any, ...]:
    return (
        int(data.num_nodes),
        p17b._tensor_hash(data.x),
        p17b._tensor_hash(data.edge_index),
        p17b._tensor_hash(data.y) if data.y is not None else None,
    )


def _performance(root: Path, label: str, group: str, datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            run = _run_dir(root, dataset, seed, group)
            marker = json.loads((run / "complete.marker").read_text())
            metrics = json.loads((run / "metrics.json").read_text())
            results = json.loads((run / "results.json").read_text())
            if marker.get("status") != "complete" or metrics.get("checkpoint_selection") != "best_val_accuracy":
                raise ValueError(f"performance artifact is not validation-selected: {run}")
            row = {"variant": label, "dataset": dataset, "seed": seed, "source_root": str(root), "selection": metrics.get("checkpoint_selection")}
            for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
                row[key] = float(results[key]["mean"])
            row["best_epoch"] = metrics.get("best_epoch")
            row["runtime_seconds"] = metrics.get("runtime_seconds")
            row["peak_gpu_memory_mib"] = metrics.get("peak_gpu_memory_mib")
            rows.append(row)
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def _sequential_deltas(perf: list[dict[str, Any]], datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    index = {(row["variant"], row["dataset"], int(row["seed"])): row for row in perf}
    pairs = (("Delta_R2", "B", "V3"), ("Delta_R1", "AB", "B"), ("Delta_remove_ref", "V3.1", "AB"))
    metrics = ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1")
    rows: list[dict[str, Any]] = []
    for name, left_label, right_label in pairs:
        for dataset in datasets:
            per_seed = []
            for seed in seeds:
                left, right = index[(left_label, dataset, seed)], index[(right_label, dataset, seed)]
                row = {"row_type": "paired_seed", "comparison": name, "dataset": dataset, "seed": seed, "left_variant": left_label, "right_variant": right_label}
                for metric in metrics:
                    row[f"delta_{metric}"] = left[metric] - right[metric]
                rows.append(row)
                per_seed.append(row)
            summary = {"row_type": "dataset_summary", "comparison": name, "dataset": dataset, "seed": "", "left_variant": left_label, "right_variant": right_label}
            for metric in metrics:
                values = [float(row[f"delta_{metric}"]) for row in per_seed]
                summary[f"delta_{metric}_mean"], summary[f"delta_{metric}_population_std"] = _mean_std(values)
                if metric.startswith("val_"):
                    summary[f"better_{metric}_count"] = sum(value > EPS for value in values)
                    summary[f"worse_{metric}_count"] = sum(value < -EPS for value in values)
                    summary[f"tie_{metric}_count"] = sum(abs(value) <= EPS for value in values)
            rows.append(summary)
        all_row = {"row_type": "all_5_unweighted", "comparison": name, "dataset": "ALL_5_UNWEIGHTED", "seed": "", "left_variant": left_label, "right_variant": right_label}
        for metric in metrics:
            values = [float(row[f"delta_{metric}"]) for row in rows if row.get("row_type") == "paired_seed" and row.get("comparison") == name]
            all_row[f"delta_{metric}_mean"], all_row[f"delta_{metric}_population_std"] = _mean_std(values)
            if metric.startswith("val_"):
                all_row[f"better_{metric}_count"] = sum(value > EPS for value in values)
                all_row[f"worse_{metric}_count"] = sum(value < -EPS for value in values)
                all_row[f"tie_{metric}_count"] = sum(abs(value) <= EPS for value in values)
        rows.append(all_row)
    return rows


def _relation_sensitivity(normal: dict[str, Any], off: dict[str, Any], head: nn.Module, data, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    z, zoff = normal["z"], off["z"]
    with torch.no_grad():
        pred = head(z).argmax(-1)
        pred_off = head(zoff).argmax(-1)
        val = data.val_idx.to(z.device)
        labels = data.y.to(z.device)
        val_acc = float((pred[val] == labels[val]).float().mean().item())
        val_off_acc = float((pred_off[val] == labels[val]).float().mean().item())
        classes = list(range(int(data.num_classes)))
        val_f1 = float(f1_score(labels[val].cpu().numpy(), pred[val].cpu().numpy(), labels=classes, average="macro", zero_division=0))
        val_off_f1 = float(f1_score(labels[val].cpu().numpy(), pred_off[val].cpu().numpy(), labels=classes, average="macro", zero_division=0))
    row = {"variant": variant, "dataset": dataset, "seed": seed, "prediction_flip_rate": float((pred != pred_off).float().mean().item()), "val_acc_delta": val_off_acc - val_acc, "val_macro_f1_delta": val_off_f1 - val_f1}
    row["fused_embedding_mae"] = float((z - zoff).abs().mean().item())
    row["fused_embedding_relative_l2"] = float((z - zoff).norm().item() / (z.norm().item() + EPS))
    row["mean_cosine_similarity"] = float(torch.nn.functional.cosine_similarity(z, zoff, dim=-1).mean().item())
    return row


def _r1_rows(model, normal, off, head, data, variant: str, dataset: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    edge_index = normal["physical_edge_index"]
    nonself = edge_index[0] != edge_index[1]
    rows = []
    for modality in MODALITIES:
        score = normal[f"relation_residual_{modality}"][nonself]
        logmod = torch.log(normal[f"relation_weight_{modality}"][nonself].clamp_min(EPS))
        c = normal[f"c_{modality}"]
        row = {"variant": variant, "dataset": dataset, "seed": seed, "modality": modality, "nonself_edge_count": int(nonself.sum().item()), "beta": float(normal[f"beta_parameter_{modality}"].item()), "score_std": float(score.std(unbiased=False).item()) if score.numel() else math.nan, "score_abs_mean": float(score.abs().mean().item()) if score.numel() else math.nan, "log_modulation_std": float(logmod.std(unbiased=False).item()) if logmod.numel() else math.nan, "log_modulation_abs_mean": float(logmod.abs().mean().item()) if logmod.numel() else math.nan, "attenuation_ratio": float(logmod.std(unbiased=False).item() / (score.std(unbiased=False).item() + EPS)) if score.numel() else math.nan}
        cstats = p17b._stats(c)
        row.update({f"c_{key}": value for key, value in cstats.items()})
        row.update(p17b._operator_metrics(model, edge_index, normal[f"normalized_edge_index_{modality}"], normal[f"normalized_edge_weight_{modality}"], data.num_nodes, data.x.dtype))
        rows.append(row)
    sensitivity = _relation_sensitivity(normal, off, head, data, variant, dataset, seed)
    return rows, sensitivity


def _r2_rows(normal: dict[str, Any], model, variant: str, dataset: str, seed: int) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    rows = []
    vectors = {}
    for modality in MODALITIES:
        p = normal[f"p_{modality}"]
        rho_p = float(getattr(model, f"semantic_rho_p_{modality}").detach().cpu())
        term_p = rho_p * p
        for hop, (change, alpha) in enumerate(zip(normal[f"d_{modality}"], normal[f"alpha_{modality}"], strict=True), start=1):
            st = p17b._stats(alpha)
            iqr = st["q75"] - st["q25"]
            row = {"row_type": "run", "variant": variant, "dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "semantic_bias": float(getattr(model, f"semantic_bias_{modality}")[hop - 1].detach().cpu()), "rho_p": rho_p, "rho_d": float(getattr(model, f"semantic_rho_d_{modality}").detach().cpu()), "alpha_corrected_range_ratio": float((st["q90"] - st["q10"]) / (abs(iqr) + EPS)), "alpha_fraction_lt_0.05": float((alpha < 0.05).float().mean().item()), "alpha_fraction_gt_0.95": float((alpha > 0.95).float().mean().item())}
            row.update({f"alpha_{key}": value for key, value in st.items()})
            row.update({f"p_{key}": value for key, value in p17b._stats(p).items()})
            row.update({f"term_p_{key}": value for key, value in p17b._stats(term_p).items()})
            row.update({f"d_{key}": value for key, value in p17b._stats(change).items()})
            rows.append(row)
            vectors[(modality, hop)] = {"alpha": alpha.detach().cpu(), "term_p": term_p.detach().cpu()}
    return rows, vectors


def _r2_cross_seed(vectors: dict[tuple[str, str, int, int], dict[str, Any]], datasets: tuple[str, ...], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        for variant in ("V3", "B"):
            for modality in MODALITIES:
                for left_index, seed_a in enumerate(seeds):
                    for seed_b in seeds[left_index + 1:]:
                        a = vectors[(dataset, variant, seed_a, 0)][(modality, 0)] if False else None
                        left = vectors[(dataset, variant, seed_a, 1)]
                        right = vectors[(dataset, variant, seed_b, 1)]
                        p, s = _corr(left["term_p"], right["term_p"])
                        rows.append({"row_type": "cross_seed", "variant": variant, "dataset": dataset, "modality": modality, "hop": "all", "seed_a": seed_a, "seed_b": seed_b, "metric": "term_p", "pearson": p, "spearman": s})
                        for hop in (1, 2, 3):
                            left = vectors[(dataset, variant, seed_a, hop)][(modality, hop)]
                            right = vectors[(dataset, variant, seed_b, hop)][(modality, hop)]
                            p, s = _corr(left["alpha"], right["alpha"])
                            rows.append({"row_type": "cross_seed", "variant": variant, "dataset": dataset, "modality": modality, "hop": hop, "seed_a": seed_a, "seed_b": seed_b, "metric": "alpha", "pearson": p, "spearman": s})
    return rows


def _reference_rows(normal: dict[str, Any], model, variant: str, dataset: str, seed: int) -> list[dict[str, Any]]:
    rows = []
    for modality in MODALITIES:
        for hop in range(4):
            ref = normal[f"reference_residual_{modality}"][hop] if f"reference_residual_{modality}" in normal else torch.zeros_like(normal[f"eta_{modality}"][:, hop])
            content = normal[f"delta_content_{modality}"][:, hop]
            relation = normal[f"relation_filter_residual_{modality}"][:, hop]
            eta = normal[f"eta_{modality}"][:, hop]
            row = {"variant": variant, "dataset": dataset, "seed": seed, "modality": modality, "hop": hop, "reference_present": int(variant == "AB"), "reference_scale": float(getattr(model, f"reference_filter_scale_{modality}").detach().cpu()) if hasattr(model, f"reference_filter_scale_{modality}") else 0.0}
            for name, value in (("reference", ref), ("content", content), ("relation", relation), ("eta", eta)):
                row.update({f"{name}_{key}": val for key, val in p17b._stats(value).items()})
                row[f"{name}_covariance_contribution"] = _covariance(value, eta)
            row["three_term_covariance_sum"] = sum(row[f"{name}_covariance_contribution"] for name in ("content", "reference", "relation") if math.isfinite(row[f"{name}_covariance_contribution"]))
            rows.append(row)
    return rows


def _reference_sensitivity(model, normal, off, head, data, dataset: str, seed: int) -> dict[str, Any]:
    row = _relation_sensitivity(normal, off, head, data, "AB", dataset, seed)
    return {"dataset": dataset, "seed": seed, **{key: value for key, value in row.items() if key not in {"variant", "dataset", "seed"}}}


def _validate_lock(path: Path, datasets: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"refusing analysis without provenance lock: {path}")
    lock = json.loads(path.read_text())
    if lock.get("task") != "nc" or lock.get("model") != "ssi_mag_v31_controls" or tuple(lock.get("datasets", [])) != datasets or tuple(int(v) for v in lock.get("seeds", [])) != seeds or set(lock.get("controls", [])) != set(CONTROL_TO_VARIANT):
        raise SystemExit("P1.7c provenance lock does not match the requested B/AB NC set")
    return lock


def _report(path: Path, summary: dict[str, Any], performance: list[dict[str, Any]], deltas: list[dict[str, Any]], r1: list[dict[str, Any]], refs: list[dict[str, Any]], sensitivity: list[dict[str, Any]]) -> None:
    lines = [
        "# SSI-MAG-V3.1 P1.7c Sequential Development Attribution",
        "",
        "This is a development-control report, not a final paper ablation. It loads validation-selected NC checkpoints and saved heads in eval/no-grad mode; it does not train, tune, run LP, add losses, or use test metrics for selection.",
        "",
        "## Integrity",
        "",
        f"- Formal controls: `{summary['expected_control_runs']}` expected, `{summary['loaded_control_runs']}` loaded, `{summary['finite_control_runs']}` finite.",
        f"- Device: `{summary['device']}`; training invoked by analyzer: `false`; LP invoked: `false`.",
        f"- Node ordering verified: `{summary['node_ordering_verified']}`.",
        f"- Attention simplex failures: `{summary['attention_simplex_failures']}`.",
        "",
        "## Sequential performance evidence",
        "",
        "Definitions: `Delta_R2 = B - V3`, `Delta_R1 = AB - B`, `Delta_remove_ref = V3.1 - AB`. Validation is the development comparison; test is descriptive only.",
        "",
    ]
    for row in deltas:
        if row.get("row_type") == "all_5_unweighted":
            lines.append(f"- **{row['comparison']}**: Val Acc `{100*row['delta_val_acc_mean']:.4f} pp`, Val Macro-F1 `{100*row['delta_val_macro_f1_mean']:.4f} pp`; better/worse/tie counts Acc `{row.get('better_val_acc_count')}/{row.get('worse_val_acc_count')}/{row.get('tie_val_acc_count')}`, F1 `{row.get('better_val_macro_f1_count')}/{row.get('worse_val_macro_f1_count')}/{row.get('tie_val_macro_f1_count')}`.")
    lines += [
        "",
        "## R2 mechanism evidence",
        "",
        "The corrected alpha range ratio is defined as `(q90-q10)/(q75-q25+eps)`. Cross-seed Pearson/Spearman values are reported separately and are not interpreted as performance selection criteria. Raw p scale is reported descriptively; a large p is not automatically a failure.",
        "",
        "## R1 mechanism evidence",
        "",
        "Old and new relation scores are on different raw scales and are not ranked by absolute r/a magnitude. The comparable quantities are log modulation, normalized operator perturbation, c_i, and frozen relation-off sensitivity. `attenuation_ratio = std(log(w))/(std(score)+eps)` is diagnostic only.",
        "",
        "## Reference-residual evidence",
        "",
        "AB restores the old reference residual under new R1/new R2. The covariance contribution is `Cov(term, eta)/Var(eta)`; the three-term sum is reported against the full signed eta and need not equal one when global/modality terms contribute.",
        "",
        "## Frozen sensitivity",
        "",
        "Relation-off and reference-off use the same saved NC head and unchanged parameters. These are functional sensitivity diagnostics, not retrained causal ablations.",
        "",
        f"- Relation sensitivity rows: `{len(r1)}`; reference sensitivity rows: `{len(sensitivity)}`.",
        "",
        "## Boundaries",
        "",
        "No final model decision is made here. No LP, paper ablation, auxiliary loss, hyperparameter search, R1 amplitude repair, historical V3 modification, frozen V3.1 modification, or test-based selection was performed.",
        "",
        "See the CSV files in this directory for complete per-seed values.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/ssi_mag_v31_p17c_controls"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssi_mag_v31_p17c_analysis"))
    parser.add_argument("--v3-root", type=Path, default=Path("outputs/ssi_mag_v3_p1_full"))
    parser.add_argument("--v31-root", type=Path, default=Path("outputs/ssi_mag_v31_p17b_full"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    project = ROOT
    input_root = args.input_root if args.input_root.is_absolute() else project / args.input_root
    output = args.output_root if args.output_root.is_absolute() else project / args.output_root
    v3_root = args.v3_root if args.v3_root.is_absolute() else project / args.v3_root
    v31_root = args.v31_root if args.v31_root.is_absolute() else project / args.v31_root
    datasets, seeds = tuple(args.datasets), tuple(args.seeds)
    lock = _validate_lock(input_root / "provenance.lock.json", datasets, seeds)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"requested analysis device unavailable: {device}")
    output.mkdir(parents=True, exist_ok=True)

    performance = []
    performance += _performance(input_root, "B", "B", datasets, seeds)
    performance += _performance(input_root, "AB", "AB", datasets, seeds)
    performance += _performance(v3_root, "V3", "full", datasets, seeds)
    performance += _performance(v31_root, "V3.1", "full", datasets, seeds)
    deltas = _sequential_deltas(performance, datasets, seeds)

    r1_rows, r1_sensitivity = [], []
    r2_rows, ref_rows, ref_sensitivity = [], [], []
    r2_vectors: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    signatures: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    attention_failures: list[str] = []
    finite_count = 0
    loaded_controls = 0

    for dataset in datasets:
        for seed in seeds:
            analyses: dict[str, tuple[Any, Any, Any, dict[str, Any], nn.Module]] = {}
            for label, root, group in (("B", input_root, "B"), ("AB", input_root, "AB"), ("V3", v3_root, "full"), ("V3.1", v31_root, "full")):
                run, cfg, data, model, head = _load_run(root, dataset, seed, group, device)
                x, edge_index = data.x.to(device), data.edge_index.to(device)
                with torch.no_grad():
                    normal = model.analysis(x, edge_index)
                    if label in {"B", "AB"}:
                        relation_off = model.analysis_intervention(x, edge_index, relation="off")
                    else:
                        relation_off = None
                if not _safe(normal) or (relation_off is not None and not _safe(relation_off)):
                    raise ValueError(f"non-finite analysis: {dataset}/seed{seed}/{label}")
                for modality in MODALITIES:
                    try:
                        p17b._validate_attention_simplex(normal[f"attention_{modality}"])
                    except ValueError as exc:
                        attention_failures.append(f"{dataset}/seed{seed}/{label}/{modality}: {exc}")
                        raise
                signatures[dataset].add(_data_signature(data))
                analyses[label] = (cfg, data, model, normal, head)
                if label in {"B", "AB"}:
                    finite_count += 1
                    loaded_controls += 1
                if label == "B":
                    rows, sens = _r1_rows(model, normal, relation_off, head, data, label, dataset, seed)
                    r1_rows.extend(rows); r1_sensitivity.append(sens)
                    part, vectors = _r2_rows(normal, model, label, dataset, seed)
                    r2_rows.extend(part)
                    for hop in (1, 2, 3):
                        for modality in MODALITIES:
                            r2_vectors[(dataset, label, seed, hop)] = {modality: vectors[(modality, hop)]}
                    ref_rows.extend(_reference_rows(normal, model, label, dataset, seed))
                elif label == "AB":
                    rows, sens = _r1_rows(model, normal, relation_off, head, data, label, dataset, seed)
                    r1_rows.extend(rows); r1_sensitivity.append(sens)
                    ref_off = model.analysis_intervention(x, edge_index, reference="off")
                    if not _safe(ref_off):
                        raise ValueError(f"non-finite reference-off analysis: {dataset}/seed{seed}")
                    ref_rows.extend(_reference_rows(normal, model, label, dataset, seed))
                    ref_sensitivity.append(_reference_sensitivity(model, normal, ref_off, head, data, dataset, seed))
                elif label == "V3":
                    part, vectors = _r2_rows(normal, model, label, dataset, seed)
                    r2_rows.extend(part)
                    for hop in (1, 2, 3):
                        for modality in MODALITIES:
                            r2_vectors[(dataset, label, seed, hop)] = {modality: vectors[(modality, hop)]}
                    ref_rows.extend(_reference_rows(normal, model, label, dataset, seed))
                else:
                    ref_rows.extend(_reference_rows(normal, model, label, dataset, seed))
                del cfg, data, model, head, normal, relation_off
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            print(f"OK {dataset} seed={seed}", flush=True)

    cross_rows = []
    for dataset in datasets:
        for variant in ("V3", "B"):
            for modality in MODALITIES:
                for left_index, seed_a in enumerate(seeds):
                    for seed_b in seeds[left_index + 1:]:
                        for hop in (1, 2, 3):
                            left = r2_vectors[(dataset, variant, seed_a, hop)][modality]
                            right = r2_vectors[(dataset, variant, seed_b, hop)][modality]
                            p, s = _corr(left["alpha"], right["alpha"])
                            cross_rows.append({"row_type": "cross_seed", "variant": variant, "dataset": dataset, "modality": modality, "hop": hop, "seed_a": seed_a, "seed_b": seed_b, "metric": "alpha", "pearson": p, "spearman": s})
                        left = r2_vectors[(dataset, variant, seed_a, 1)][modality]
                        right = r2_vectors[(dataset, variant, seed_b, 1)][modality]
                        p, s = _corr(left["term_p"], right["term_p"])
                        cross_rows.append({"row_type": "cross_seed", "variant": variant, "dataset": dataset, "modality": modality, "hop": "all", "seed_a": seed_a, "seed_b": seed_b, "metric": "term_p", "pearson": p, "spearman": s})
    r2_rows.extend(cross_rows)

    node_ordering = {dataset: len(signatures[dataset]) == 1 for dataset in datasets}
    summary = {
        "datasets": list(datasets), "seeds": list(seeds), "expected_control_runs": len(datasets) * len(seeds) * 2,
        "loaded_control_runs": loaded_controls, "finite_control_runs": finite_count,
        "device": str(device), "provenance_lock": str(input_root / "provenance.lock.json"), "lock_git_commit": lock.get("git_commit"),
        "node_ordering_verified": node_ordering, "attention_simplex_failures": attention_failures,
        "performance_rows": len(performance), "sequential_delta_rows": len(deltas), "r1_rows": len(r1_rows), "r2_rows": len(r2_rows), "reference_rows": len(ref_rows),
        "relation_sensitivity_rows": len(r1_sensitivity), "reference_sensitivity_rows": len(ref_sensitivity),
        "training_invoked": False, "lp_invoked": False, "ablation_invoked": False, "test_metrics_used_for_selection_or_decision": False,
        "comparisons": {"Delta_R2": "B - V3", "Delta_R1": "AB - B", "Delta_remove_ref": "V3.1 - AB"},
    }
    _write_csv(output / "p17c_performance.csv", performance)
    _write_csv(output / "p17c_sequential_deltas.csv", deltas)
    _write_csv(output / "p17c_r2_attribution.csv", r2_rows)
    _write_csv(output / "p17c_r1_attribution.csv", r1_rows)
    _write_csv(output / "p17c_reference_attribution.csv", ref_rows)
    _write_csv(output / "p17c_reference_frozen_sensitivity.csv", ref_sensitivity)
    _write_csv(output / "p17c_relation_frozen_sensitivity.csv", r1_sensitivity)
    (output / "p17c_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _report(output / "p17c_report.md", summary, performance, deltas, r1_rows, ref_rows, ref_sensitivity)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
