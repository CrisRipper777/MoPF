#!/usr/bin/env python3
"""Audit completed matched robustness evaluations and build paper summaries.

The replicate hierarchy is fixed: perturbation seeds are averaged within each
model seed first; paper-facing means and population SDs are then computed over
the three model seeds. This script never runs model inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/robustness_analysis"
DATASETS = ("Movies", "Grocery")
MODELS = ("mopf", "wo_relation_calibration", "wo_semantic_anchor", "dip")
MODEL_SEEDS = (42, 43, 44)
PERTURB_SEEDS = (1001, 1002, 1003)
PERTURBATIONS = {
    "random_structural_noise": (0.0, 0.05, 0.10, 0.20, 0.30, 0.40),
    "semantic_conflict_edge_injection": (0.0, 0.05, 0.10, 0.20, 0.30),
}
METRICS = {
    "accuracy": ("accuracy_retention_pct", "test_accuracy"),
    "macro_f1": ("macro_f1_retention_pct", "test_macro_f1"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        row["dataset"], row["perturbation_type"], row["model"],
        int(row["model_seed"]), int(row["perturbation_seed"]), round(float(row["rho"]), 8),
    )


def _finite(row: dict[str, str], field: str) -> bool:
    try:
        return math.isfinite(float(row[field]))
    except (ValueError, TypeError, KeyError):
        return False


def audit_completed_matrix() -> dict[str, Any]:
    audit_path = OUT / "checkpoint_audit_matched.csv"
    clean_path = OUT / "clean_gate_matched.csv"
    raw_path = OUT / "raw_evaluations.csv"
    random_path = OUT / "random_noise/dry_run_plan_matched.csv"
    conflict_path = OUT / "semantic_conflict/dry_run_plan_matched.csv"
    for p in (audit_path, clean_path, raw_path, random_path, conflict_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    audit_rows = read_csv(audit_path)
    clean_rows = read_csv(clean_path)
    raw_rows = read_csv(raw_path)
    plan_rows = read_csv(random_path) + read_csv(conflict_path)
    errors: list[str] = []

    planned = {key(row): row for row in plan_rows}
    evaluated: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in raw_rows:
        k = key(row)
        if k in evaluated:
            errors.append(f"duplicate raw condition: {k}")
        evaluated[k] = row
    if set(evaluated) != set(planned):
        errors.append(
            f"raw/plan key mismatch: missing={len(set(planned)-set(evaluated))}, "
            f"extra={len(set(evaluated)-set(planned))}"
        )
    if len(raw_rows) != 702 or len(plan_rows) != 702:
        errors.append(f"expected 702 raw and planned conditions, got {len(raw_rows)} / {len(plan_rows)}")

    checks = {
        "matrix_complete": set(evaluated) == set(planned) and len(raw_rows) == 702,
        "all_metrics_finite": True,
        "rho0_reused_and_exactly_100": True,
        "nonzero_gpu_runtime_recorded": True,
        "plan_hashes_match": True,
        "checkpoint_hashes_unchanged": True,
        "shared_perturbation_hashes": True,
        "split_sha_matches_within_dataset": True,
        "edge_counts_and_budgets_match": True,
        "pre_evaluation_plan_audit_passed": False,
        "retention_and_clean_baselines_match": True,
    }

    clean_by_path = {str(Path(r["checkpoint_path"]).resolve()): r for r in clean_rows}
    audit_by_path = {str(Path(r["checkpoint_path"]).resolve()): r for r in audit_rows}
    for k, row in evaluated.items():
        planned_row = planned.get(k)
        if planned_row is None:
            continue
        for field in ("checkpoint_sha256", "edge_sha256", "edge_artifact_sha256", "split_sha256", "injected_edge_count"):
            if row.get(field) != planned_row.get(field):
                checks["plan_hashes_match"] = False
                errors.append(f"raw field {field} differs from plan: {k}")
        for field in ("test_accuracy", "test_macro_f1", "clean_accuracy", "clean_macro_f1", "accuracy_retention_pct", "macro_f1_retention_pct"):
            if not _finite(row, field):
                checks["all_metrics_finite"] = False
                errors.append(f"nonfinite {field}: {k}")
        rho = float(row["rho"])
        if rho == 0.0:
            if row.get("clean_row_reused") != "True" or row.get("runtime_source") != "cached_clean_gate":
                checks["rho0_reused_and_exactly_100"] = False
                errors.append(f"rho=0 was not sourced from cache: {k}")
            if float(row["accuracy_retention_pct"]) != 100.0 or float(row["macro_f1_retention_pct"]) != 100.0:
                checks["rho0_reused_and_exactly_100"] = False
                errors.append(f"rho=0 retention is not exactly 100: {k}")
            ckpt = str(Path(row["checkpoint_path"]).resolve())
            gate = clean_by_path.get(ckpt)
            if gate is None or gate.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
                errors.append(f"rho=0 clean-gate provenance mismatch: {k}")
            elif any(
                not np.isclose(float(row[observed]), float(gate[cached]), rtol=0.0, atol=1e-8)
                for observed, cached in (
                    ("clean_accuracy", "measured_accuracy"),
                    ("clean_macro_f1", "measured_macro_f1"),
                    ("test_accuracy", "measured_accuracy"),
                    ("test_macro_f1", "measured_macro_f1"),
                )
            ):
                checks["retention_and_clean_baselines_match"] = False
                errors.append(f"rho=0 raw clean metrics disagree with cached clean gate: {k}")
        else:
            if not _finite(row, "runtime_seconds") or float(row["runtime_seconds"]) <= 0:
                checks["nonzero_gpu_runtime_recorded"] = False
                errors.append(f"missing/nonpositive inference runtime: {k}")
            if not _finite(row, "peak_gpu_memory_mib") or float(row["peak_gpu_memory_mib"]) <= 0:
                checks["nonzero_gpu_runtime_recorded"] = False
                errors.append(f"missing/nonpositive CUDA memory record: {k}")
            expected_acc_ret = 100.0 * float(row["test_accuracy"]) / float(row["clean_accuracy"])
            expected_f1_ret = 100.0 * float(row["test_macro_f1"]) / float(row["clean_macro_f1"])
            if not np.isclose(float(row["accuracy_retention_pct"]), expected_acc_ret, rtol=0.0, atol=1e-10) or not np.isclose(
                float(row["macro_f1_retention_pct"]), expected_f1_ret, rtol=0.0, atol=1e-10
            ):
                checks["retention_and_clean_baselines_match"] = False
                errors.append(f"retention does not match checkpoint-specific clean baseline: {k}")

        ckpt_path = Path(row["checkpoint_path"])
        ckpt_key = str(ckpt_path.resolve())
        audit = audit_by_path.get(ckpt_key)
        if audit is None or audit.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
            checks["checkpoint_hashes_unchanged"] = False
            errors.append(f"checkpoint SHA differs from authoritative audit: {k}")
        elif sha256_file(ckpt_path) != row.get("checkpoint_sha256"):
            checks["checkpoint_hashes_unchanged"] = False
            errors.append(f"checkpoint file SHA changed: {ckpt_path}")

        clean_e = int(row["clean_canonical_edges"])
        injected = int(row["injected_edge_count"])
        if int(row["perturbed_canonical_edges"]) != clean_e + injected:
            checks["edge_counts_and_budgets_match"] = False
            errors.append(f"canonical edge count mismatch: {k}")
        if int(row["perturbed_edge_index_columns"]) != 2 * (clean_e + injected):
            checks["edge_counts_and_budgets_match"] = False
            errors.append(f"directed edge-index column count mismatch: {k}")
        if injected != int(round(rho * clean_e)):
            checks["edge_counts_and_budgets_match"] = False
            errors.append(f"injected edge budget does not match rho rounding: {k}")

    graph_groups: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in raw_rows:
        graph_groups[(row["dataset"], row["perturbation_type"], int(row["perturbation_seed"]), round(float(row["rho"]), 8))].add(row["edge_sha256"])
        split_groups[row["dataset"]].add(row["split_sha256"])
    bad_graphs = [k for k, values in graph_groups.items() if len(values) != 1]
    if bad_graphs:
        checks["shared_perturbation_hashes"] = False
        errors.append(f"shared graph SHA mismatch in {len(bad_graphs)} perturbation conditions")
    if any(len(values) != 1 for values in split_groups.values()):
        checks["split_sha_matches_within_dataset"] = False
        errors.append("split SHA differs within a dataset")

    pre_path = OUT / "robustness_pre_evaluation_audit.json"
    if pre_path.is_file():
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        checks["pre_evaluation_plan_audit_passed"] = pre.get("status") == "PASS" and all(pre.get("checks", {}).values())
    if not checks["pre_evaluation_plan_audit_passed"]:
        errors.append("the pre-evaluation plan/category audit is absent or not PASS")

    if errors:
        checks["overall"] = False
    else:
        checks["overall"] = True
    result = {
        "status": "PASS" if checks["overall"] else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": len(raw_rows),
        "planned_rows": len(plan_rows),
        "rho0_cached_rows": sum(float(r["rho"]) == 0.0 for r in raw_rows),
        "nonzero_cuda_evaluations": sum(float(r["rho"]) > 0.0 for r in raw_rows),
        "checks": checks,
        "errors": errors,
    }
    (OUT / "robustness_post_evaluation_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("Robustness post-evaluation audit failed; do not plot. See robustness_post_evaluation_audit.json")
    return {"raw": raw_rows, "audit": result}


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def average_by_model_seed(raw_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], dict[str, float]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["dataset"], row["perturbation_type"], row["model"], int(row["model_seed"]), round(float(row["rho"]), 8))].append(row)
    result: list[dict[str, Any]] = []
    lookup: dict[tuple[Any, ...], dict[str, float]] = {}
    for key_, values in sorted(grouped.items()):
        ds, perturb, model, seed, rho = key_
        if len(values) != len(PERTURB_SEEDS) or {int(v["perturbation_seed"]) for v in values} != set(PERTURB_SEEDS):
            raise ValueError(f"expected exactly three perturbation seeds for {key_}")
        for metric, (ret_key, absolute_key) in METRICS.items():
            retention = [float(v[ret_key]) for v in values]
            absolute = [100.0 * float(v[absolute_key]) for v in values]
            ret_mean, ret_sd = mean_sd(retention)
            abs_mean, abs_sd = mean_sd(absolute)
            lookup[(ds, perturb, model, seed, rho, metric)] = {
                "retention": ret_mean,
                "retention_pseed_sd": ret_sd,
                "absolute": abs_mean,
                "absolute_pseed_sd": abs_sd,
            }
            result.append({
                "dataset": ds,
                "perturbation_type": perturb,
                "model": model,
                "model_seed": seed,
                "rho": rho,
                "metric": metric,
                "mean_over_perturbation_seeds": ret_mean,
                "sd_over_perturbation_seeds": ret_sd,
                "absolute_mean_over_perturbation_seeds_pct": abs_mean,
                "absolute_sd_over_perturbation_seeds_pct": abs_sd,
                "num_perturbation_seeds": len(values),
            })
    return result, lookup


def aggregate_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["dataset"], row["perturbation_type"], row["model"], row["rho"], row["metric"])].append(row)
    rows = []
    for key_, values in sorted(grouped.items()):
        ds, perturb, model, rho, metric = key_
        ret = [float(v["mean_over_perturbation_seeds"]) for v in values]
        pseed_sd = [float(v["sd_over_perturbation_seeds"]) for v in values]
        mean, sd = mean_sd(ret)
        rows.append({
            "dataset": ds,
            "perturbation_type": perturb,
            "model": model,
            "rho": rho,
            "metric": metric,
            "mean_across_model_seeds": mean,
            "population_sd_across_model_seeds": sd,
            "mean_within_model_seed_perturbation_sd": float(np.mean(pseed_sd)),
            "num_model_seeds": len(values),
            "num_perturbation_seeds_per_model_seed": 3,
            "sd_convention": "population (ddof=0) across model-seed means",
        })
    return rows


def _rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    if len(x) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def make_gap_rows(seed_lookup: dict[tuple[Any, ...], dict[str, float]], perturb: str, comparison: tuple[str, str], name: str) -> list[dict[str, Any]]:
    full_model, ablation_model = comparison
    datasets = DATASETS
    all_rows: list[dict[str, Any]] = []
    for ds in datasets:
        for metric in METRICS:
            rhos = PERTURBATIONS[perturb]
            by_rho: list[tuple[float, list[float]]] = []
            for rho in rhos:
                seed_gaps = [
                    seed_lookup[(ds, perturb, full_model, seed, rho, metric)]["retention"]
                    - seed_lookup[(ds, perturb, ablation_model, seed, rho, metric)]["retention"]
                    for seed in MODEL_SEEDS
                ]
                by_rho.append((rho, seed_gaps))
            means = [float(np.mean(v)) for _, v in by_rho]
            max_index = int(np.argmax(means))
            final_mean = means[-1]
            trend = spearman([rho for rho, _ in by_rho], means)
            for (rho, seed_gaps), mean_gap in zip(by_rho, means):
                sd_gap = float(np.std(seed_gaps, ddof=0))
                row = {
                    "dataset": ds,
                    "perturbation_type": perturb,
                    "gap": name,
                    "metric": metric,
                    "rho": rho,
                    "mean_gap_pp": mean_gap,
                    "population_sd_gap_pp_across_model_seeds": sd_gap,
                    "num_model_seeds": 3,
                    "maximum_gap_pp": means[max_index],
                    "rho_at_maximum_gap": by_rho[max_index][0],
                    "final_rho_gap_pp": final_mean,
                    "final_rho_gap_sign": "positive" if final_mean > 0 else "negative" if final_mean < 0 else "zero",
                    "spearman_rho_rho_vs_gap_descriptive": trend,
                }
                for seed, value in zip(MODEL_SEEDS, seed_gaps):
                    row[f"seed_{seed}_gap_pp"] = value
                all_rows.append(row)
    return all_rows


def make_summary_metrics(seed_lookup: dict[tuple[Any, ...], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    combos = sorted({(k[0], k[1], k[2]) for k in seed_lookup})
    for ds, perturb, model in combos:
        rhos = list(PERTURBATIONS[perturb])
        for metric in METRICS:
            seed_final: list[float] = []
            seed_auc: list[float] = []
            seed_degradation: list[float] = []
            seed_pseed_sd: list[float] = []
            for seed in MODEL_SEEDS:
                vals = [seed_lookup[(ds, perturb, model, seed, rho, metric)]["retention"] for rho in rhos]
                pseed_sd = [seed_lookup[(ds, perturb, model, seed, rho, metric)]["retention_pseed_sd"] for rho in rhos]
                final = vals[-1]
                widths = np.diff(np.asarray(rhos, dtype=float))
                auc = float(np.sum(widths * (np.asarray(vals[:-1]) + np.asarray(vals[1:])) / 2.0) / (rhos[-1] - rhos[0]))
                seed_final.append(final)
                seed_auc.append(auc)
                seed_degradation.append(100.0 - final)
                seed_pseed_sd.append(float(np.mean(pseed_sd)))
            final_m, final_sd = mean_sd(seed_final)
            auc_m, auc_sd = mean_sd(seed_auc)
            deg_m, deg_sd = mean_sd(seed_degradation)
            ps_m, ps_sd = mean_sd(seed_pseed_sd)
            rows.append({
                "dataset": ds,
                "perturbation_type": perturb,
                "model": model,
                "metric": metric,
                "maximum_rho": rhos[-1],
                "final_retention_pct_mean": final_m,
                "final_retention_pct_population_sd": final_sd,
                "normalized_retention_auc_pct_mean": auc_m,
                "normalized_retention_auc_pct_population_sd": auc_sd,
                "absolute_degradation_pp_mean": deg_m,
                "absolute_degradation_pp_population_sd": deg_sd,
                "mean_within_seed_perturbation_sd_pp": ps_m,
                "sd_of_within_seed_perturbation_sd_across_model_seeds": ps_sd,
                "num_model_seeds": 3,
                "sd_convention": "population (ddof=0) across model-seed values",
            })
    return rows


def make_random_conflict(seed_lookup: dict[tuple[Any, ...], dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matched_rhos = (0.05, 0.10, 0.20, 0.30)
    for ds in DATASETS:
        for rho in matched_rhos:
            for metric in METRICS:
                for model in ("mopf", "wo_relation_calibration", "dip"):
                    per_seed = []
                    for seed in MODEL_SEEDS:
                        random = seed_lookup[(ds, "random_structural_noise", model, seed, rho, metric)]["retention"]
                        conflict = seed_lookup[(ds, "semantic_conflict_edge_injection", model, seed, rho, metric)]["retention"]
                        per_seed.append(random - conflict)
                    mean, sd = mean_sd(per_seed)
                    rows.append({
                        "dataset": ds, "rho": rho, "metric": metric,
                        "comparison": "ConflictExtraDamage",
                        "model": model,
                        "mean_pp": mean,
                        "population_sd_pp_across_model_seeds": sd,
                        "num_model_seeds": 3,
                        **{f"seed_{seed}_pp": v for seed, v in zip(MODEL_SEEDS, per_seed)},
                    })
                per_seed = []
                for seed in MODEL_SEEDS:
                    conflict_gap = (
                        seed_lookup[(ds, "semantic_conflict_edge_injection", "mopf", seed, rho, metric)]["retention"]
                        - seed_lookup[(ds, "semantic_conflict_edge_injection", "wo_relation_calibration", seed, rho, metric)]["retention"]
                    )
                    random_gap = (
                        seed_lookup[(ds, "random_structural_noise", "mopf", seed, rho, metric)]["retention"]
                        - seed_lookup[(ds, "random_structural_noise", "wo_relation_calibration", seed, rho, metric)]["retention"]
                    )
                    per_seed.append(conflict_gap - random_gap)
                mean, sd = mean_sd(per_seed)
                rows.append({
                    "dataset": ds, "rho": rho, "metric": metric,
                    "comparison": "CalibrationExtraBenefit",
                    "model": "mopf_vs_wo_relation_calibration",
                    "mean_pp": mean,
                    "population_sd_pp_across_model_seeds": sd,
                    "num_model_seeds": 3,
                    **{f"seed_{seed}_pp": v for seed, v in zip(MODEL_SEEDS, per_seed)},
                })
    return rows


def make_appendix_table(seed_lookup: dict[tuple[Any, ...], dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for ds, perturb in sorted({(k[0], k[1]) for k in seed_lookup}):
        max_rho = PERTURBATIONS[perturb][-1]
        for model in MODELS:
            if not any(k[0] == ds and k[1] == perturb and k[2] == model for k in seed_lookup):
                continue
            for rho, label in ((0.0, "clean"), (max_rho, "maximum")):
                acc = [seed_lookup[(ds, perturb, model, seed, rho, "accuracy")]["absolute"] for seed in MODEL_SEEDS]
                f1 = [seed_lookup[(ds, perturb, model, seed, rho, "macro_f1")]["absolute"] for seed in MODEL_SEEDS]
                acc_m, acc_sd = mean_sd(acc)
                f1_m, f1_sd = mean_sd(f1)
                rows.append({
                    "dataset": ds,
                    "perturbation_type": perturb,
                    "model": model,
                    "rho_label": label,
                    "rho": rho,
                    "accuracy_pct_mean": acc_m,
                    "accuracy_pct_population_sd": acc_sd,
                    "macro_f1_pct_mean": f1_m,
                    "macro_f1_pct_population_sd": f1_sd,
                    "num_model_seeds": 3,
                    "sd_convention": "population (ddof=0) across model-seed values after perturbation-seed averaging",
                })
    return rows


def write_appendix_formats(rows: list[dict[str, Any]]) -> None:
    write_csv(OUT / "absolute_performance_appendix.csv", rows)
    md = [
        "| Dataset | Perturbation | Model | Point | ρ | Accuracy (%) | Macro-F1 (%) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    tex = [r"\begin{tabular}{lllcrrr}", r"\toprule", r"Dataset & Perturbation & Model & Point & $\rho$ & Accuracy & Macro-F1 \\", r"\midrule"]
    for r in rows:
        acc = f"{r['accuracy_pct_mean']:.2f} ± {r['accuracy_pct_population_sd']:.2f}"
        f1 = f"{r['macro_f1_pct_mean']:.2f} ± {r['macro_f1_pct_population_sd']:.2f}"
        acc_tex = f"{r['accuracy_pct_mean']:.2f} \\pm {r['accuracy_pct_population_sd']:.2f}"
        f1_tex = f"{r['macro_f1_pct_mean']:.2f} \\pm {r['macro_f1_pct_population_sd']:.2f}"
        perturb_label = "Random noise" if r["perturbation_type"] == "random_structural_noise" else "Semantic conflict"
        model_tex = str(r["model"]).replace("_", r"\_")
        md.append(f"| {r['dataset']} | {perturb_label} | {r['model']} | {r['rho_label']} | {r['rho']:.2f} | {acc} | {f1} |")
        tex.append(f"{r['dataset']} & {perturb_label} & {model_tex} & {r['rho_label']} & {r['rho']:.2f} & ${acc_tex}$ & ${f1_tex}$ \\")
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "absolute_performance_appendix.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "absolute_performance_appendix.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main() -> None:
    audited = audit_completed_matrix()
    seed_rows, lookup = average_by_model_seed(audited["raw"])
    aggregate = aggregate_rows(seed_rows)
    for perturb, directory in PERTURBATIONS.items():
        subset_seed = [r for r in seed_rows if r["perturbation_type"] == perturb]
        subset_aggregate = [r for r in aggregate if r["perturbation_type"] == perturb]
        write_csv(OUT / directory_name(perturb) / "per_model_seed_summary.csv", subset_seed)
        write_csv(OUT / directory_name(perturb) / "aggregate_summary.csv", subset_aggregate)
    write_csv(OUT / "aggregate_summary.csv", aggregate)
    # Backward-compatible name for the existing plot script interface.
    write_csv(OUT / "robustness_summary.csv", aggregate)

    all_gap_rows = []
    random_gaps = make_gap_rows(lookup, "random_structural_noise", ("mopf", "wo_relation_calibration"), "RelationCalibrationGap")
    random_gaps += make_gap_rows(lookup, "random_structural_noise", ("mopf", "wo_semantic_anchor"), "SemanticAnchorGap")
    conflict_gaps = make_gap_rows(lookup, "semantic_conflict_edge_injection", ("mopf", "wo_relation_calibration"), "RelationCalibrationGap")
    all_gap_rows.extend(random_gaps + conflict_gaps)
    write_csv(OUT / "random_noise/relation_calibration_gap.csv", [r for r in random_gaps if r["gap"] == "RelationCalibrationGap"])
    write_csv(OUT / "random_noise/semantic_anchor_gap.csv", [r for r in random_gaps if r["gap"] == "SemanticAnchorGap"])
    write_csv(OUT / "semantic_conflict/relation_calibration_gap.csv", conflict_gaps)
    write_csv(OUT / "robustness_gaps_all.csv", all_gap_rows)
    write_csv(OUT / "robustness_summary_metrics.csv", make_summary_metrics(lookup))
    write_csv(OUT / "random_vs_conflict_comparison.csv", make_random_conflict(lookup))
    write_appendix_formats(make_appendix_table(lookup))

    completion = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_condition_rows": audited["audit"]["raw_rows"],
        "rho0_cached_rows": audited["audit"]["rho0_cached_rows"],
        "nonzero_cuda_evaluations": audited["audit"]["nonzero_cuda_evaluations"],
        "post_evaluation_audit": "robustness_post_evaluation_audit.json",
        "summary_files": [
            "raw_evaluations.csv",
            "robustness_pre_evaluation_audit.json",
            "robustness_post_evaluation_audit.json",
            "aggregate_summary.csv", "robustness_summary_metrics.csv", "robustness_gaps_all.csv",
            "random_noise/per_model_seed_summary.csv", "random_noise/aggregate_summary.csv",
            "random_noise/relation_calibration_gap.csv", "random_noise/semantic_anchor_gap.csv",
            "semantic_conflict/per_model_seed_summary.csv", "semantic_conflict/aggregate_summary.csv",
            "semantic_conflict/relation_calibration_gap.csv", "random_vs_conflict_comparison.csv",
            "absolute_performance_appendix.csv", "absolute_performance_appendix.md", "absolute_performance_appendix.tex",
            "figure5_robustness.pdf", "figure5_robustness.png", "figure5_robustness.svg", "figure5_robustness.tiff",
            "figure5_robustness_macro_f1.pdf", "figure5_robustness_macro_f1.png",
            "figure5_robustness_alignment.json", "figure5_alignment_layout.json",
            "figure5_collision_audit.json", "figure5_macro_f1_collision_audit.json",
            "../../docs/cosi_mag_robustness_results.md",
        ],
        "sd_convention": "population standard deviation (ddof=0) across model seeds after averaging perturbation seeds within model seed",
    }
    (OUT / "robustness_completion_manifest.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(completion, indent=2))


def directory_name(perturbation: str) -> str:
    return "random_noise" if perturbation == "random_structural_noise" else "semantic_conflict"


if __name__ == "__main__":
    main()
