#!/usr/bin/env python3
"""Aggregate the NC-only final benchmark and generate the paper table."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODELS = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec", "mopf")
DISPLAY_NAMES = {
    "mlp": "MLP",
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "mmgcn": "MMGCN",
    "mgat": "MGAT",
    "dip": "DiP",
    "dgf": "DGF",
    "dmgc": "DMGC",
    "lgmrec": "LGMRec",
    "mopf": "MoPF",
}
PROTOCOL = "unified_full_graph_nc_v1"
METRICS = ("test_accuracy", "test_macro_f1")
ALL_METRICS = ("best_validation_accuracy", "validation_macro_f1", *METRICS)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _stat(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty value list")
    return {
        "mean": float(mean(values)),
        "population_std": float(pstdev(values)),
        "per_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
    }


def _fmt(value: float) -> str:
    return f"{value * 100.0:.2f} $\\pm$ {0.0:.2f}"


def _fmt_stat(stat: dict[str, Any], markdown: bool = False) -> str:
    separator = " ± " if markdown else r" $\pm$ "
    return f"{float(stat['mean']) * 100.0:.2f}{separator}{float(stat['population_std']) * 100.0:.2f}"


def _record_path(output_root: Path, dataset: str, model: str, seed: int) -> Path:
    return output_root / dataset / model / f"seed{seed}" / "run_record.json"


def _load_records(output_root: Path, allow_incomplete: bool) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[str] = []
    for dataset in DATASETS:
        records[dataset] = {}
        for model in MODELS:
            per_seed: dict[str, Any] = {}
            for seed in SEEDS:
                path = _record_path(output_root, dataset, model, seed)
                key = f"{dataset}/{model}/seed{seed}"
                if not path.is_file():
                    missing.append(key)
                    continue
                record = _load(path)
                if record.get("status") != "PASS":
                    missing.append(f"{key} (status={record.get('status')})")
                    continue
                per_seed[str(seed)] = record
            records[dataset][model] = per_seed
    if missing and not allow_incomplete:
        raise RuntimeError("NC benchmark is incomplete:\n" + "\n".join(missing[:40]))
    return records, missing


def _aggregate(records: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for dataset in DATASETS:
        aggregate[dataset] = {}
        for model in MODELS:
            rows = records[dataset][model]
            if len(rows) != len(SEEDS):
                continue
            aggregate[dataset][model] = {
                metric: _stat([float(rows[str(seed)][metric]) for seed in SEEDS])
                for metric in ALL_METRICS
            }
            aggregate[dataset][model]["eval_label_set"] = rows["42"]["eval_label_set"]
            aggregate[dataset][model]["evaluated_class_count"] = rows["42"]["evaluated_class_count"]
            aggregate[dataset][model]["parameter_count"] = rows["42"]["parameter_count"]
    return aggregate


def _rankings(aggregate: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset in DATASETS:
        output[dataset] = {}
        for metric in METRICS:
            available = [model for model in MODELS if model in aggregate[dataset]]
            ordered = sorted(available, key=lambda model: (-aggregate[dataset][model][metric]["mean"], MODELS.index(model)))
            output[dataset][metric] = {
                "best": ordered[0] if ordered else None,
                "second": ordered[1] if len(ordered) > 1 else None,
                "ordered": ordered,
            }
    return output


def _mopf_counts(rankings: dict[str, Any]) -> dict[str, int]:
    best = 0
    top2 = 0
    details = []
    for dataset in DATASETS:
        for metric in METRICS:
            rank = rankings[dataset][metric]
            if rank["best"] == "mopf":
                best += 1
                details.append({"dataset": dataset, "metric": metric, "rank": 1})
            elif rank["second"] == "mopf":
                top2 += 1
                details.append({"dataset": dataset, "metric": metric, "rank": 2})
    return {"best": best, "top2_excluding_best": top2, "top2_total": best + top2, "details": details}


def _load_attempts(output_root: Path) -> dict[str, Any]:
    path = output_root / "attempts.jsonl"
    attempts: dict[str, list[str]] = {}
    if not path.is_file():
        return {"attempt_count": {}, "rerun_jobs": [], "failed_attempts": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = f"{row.get('dataset')}/{row.get('model')}/seed{row.get('seed')}"
        attempts.setdefault(key, []).append(str(row.get("status")))
    rerun = sorted(key for key, statuses in attempts.items() if len(statuses) > 1)
    failed = sorted(key for key, statuses in attempts.items() if any(status in {"failed", "failed_validation"} for status in statuses))
    return {
        "attempt_count": {key: len(value) for key, value in sorted(attempts.items())},
        "rerun_jobs": rerun,
        "failed_attempts": failed,
    }


def _health_summary(records: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    checks: dict[str, int] = {}
    failures: list[str] = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                rows = records[dataset][model]
                record = rows.get(str(seed))
                if record is None:
                    continue
                for key, value in record.get("health_checks", {}).items():
                    checks[key] = checks.get(key, 0) + int(bool(value))
                    if not value:
                        failures.append(f"{dataset}/{model}/seed{seed}: {key}")
    return {"all_checks_pass": not failures, "passed_counts": checks, "failures": failures}


def _anomalies(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    # A transparent descriptive flag, not an exclusion rule.  Three percentage
    # points is used because the final table reports percentage-point spreads.
    output = []
    for dataset in DATASETS:
        for model in MODELS:
            if model not in aggregate[dataset]:
                continue
            for metric in METRICS:
                std_pp = aggregate[dataset][model][metric]["population_std"] * 100.0
                if std_pp > 3.0:
                    output.append({"dataset": dataset, "model": model, "metric": metric, "population_std_percentage_points": std_pp})
    return output


def _markdown(aggregate: dict[str, Any], rankings: dict[str, Any], health: dict[str, Any], attempts: dict[str, Any], anomalies: list[dict[str, Any]], output_root: Path, missing: list[str]) -> str:
    lines = [
        "# Final Node Classification Benchmark (fixed Macro-F1)",
        "",
        "## Protocol",
        "",
        f"- Protocol: `{PROTOCOL}`.",
        "- Task: transductive full-graph node classification; one complete encoder forward per epoch; cross-entropy is computed only on train nodes.",
        "- Checkpoint selection: validation accuracy; test metrics are descriptive and are evaluated once after restoring the selected checkpoint.",
        "- Seeds: `42`, `43`, `44`; all models on a dataset use the same frozen split.",
        "- Fixed Macro-F1: one task-level label set is formed from the valid train + validation + test label union, with invalid labels filtered; the same set is used for every split, model, and seed.",
        "- Internal predecessors `map_mag`, `map_mag_v1`, `map_mag_v2`, and `map_mag_v3` are excluded from the formal ranking.",
        "- LP results are frozen legacy results and were not rerun in this task.",
        "",
        "## Datasets and evaluated labels",
        "",
        "| Dataset | Nodes | Configured classes | Evaluated label set | Evaluated classes | Frozen split | Split SHA-256 |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    preflight_path = output_root / "preflight.json"
    preflight = _load(preflight_path) if preflight_path.is_file() else {"datasets": {}}
    for dataset in DATASETS:
        info = preflight.get("datasets", {}).get(dataset, {})
        lines.append(f"| {dataset} | {info.get('num_nodes', '--')} | {info.get('num_classes', '--')} | `{info.get('eval_labels', '--')}` | {info.get('evaluated_class_count', '--')} | `{info.get('split_path', '--')}` | `{info.get('split_sha256', '--')}` |")

    lines.extend(["", "## Complete per-dataset results", "", "All values are percentages and use mean ± population standard deviation.", ""])
    for dataset in DATASETS:
        lines.extend([
            f"### {dataset}",
            "",
            "| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Seed-level test values (42 / 43 / 44) |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for model in MODELS:
            if model not in aggregate[dataset]:
                continue
            stats = aggregate[dataset][model]
            seed_values = "; ".join(f"{seed}: {stats['test_accuracy']['per_seed'][str(seed)] * 100:.4f} / {stats['test_macro_f1']['per_seed'][str(seed)] * 100:.4f}" for seed in SEEDS)
            lines.append(f"| {DISPLAY_NAMES[model]} | {_fmt_stat(stats['best_validation_accuracy'], True)} | {_fmt_stat(stats['validation_macro_f1'], True)} | {_fmt_stat(stats['test_accuracy'], True)} | {_fmt_stat(stats['test_macro_f1'], True)} | {seed_values} |")
        lines.append("")

    lines.extend(["## Best and second-best", "", "Ranking uses the mean test value among the ten formal models only; standard deviations do not affect ranking.", "", "| Dataset | Metric | Best | Second-best |", "|---|---|---|---|"])
    for dataset in DATASETS:
        for metric in METRICS:
            rank = rankings[dataset][metric]
            best = rank["best"]
            second = rank["second"]
            if best is None:
                continue
            lines.append(f"| {dataset} | {metric.replace('_', ' ').title()} | {DISPLAY_NAMES[best]} ({aggregate[dataset][best][metric]['mean'] * 100:.4f}) | {DISPLAY_NAMES[second]} ({aggregate[dataset][second][metric]['mean'] * 100:.4f}) |")

    mopf = _mopf_counts(rankings)
    lines.extend([
        "",
        "## MoPF paper-facing result",
        "",
        f"MoPF is best on **{mopf['best']} / 10** dataset × metric cells and in the top two on **{mopf['top2_total']} / 10** cells (additional second-place cells: {mopf['top2_excluding_best']}).",
        "",
        "| Dataset | Test Accuracy | Test Macro-F1 |",
        "|---|---:|---:|",
    ])
    for dataset in DATASETS:
        if "mopf" in aggregate[dataset]:
            lines.append(f"| {dataset} | {_fmt_stat(aggregate[dataset]['mopf']['test_accuracy'], True)} | {_fmt_stat(aggregate[dataset]['mopf']['test_macro_f1'], True)} |")

    lines.extend([
        "",
        "## Validity and exceptions",
        "",
        f"- Expected jobs: `{len(DATASETS) * len(MODELS) * len(SEEDS)}`; completed records: `{sum(len(data) for data in aggregate.values()) * len(SEEDS)}`.",
        f"- All recorded health/provenance checks passed: `{health['all_checks_pass']}`.",
        f"- Failed attempts: `{attempts['failed_attempts']}`; rerun jobs: `{attempts['rerun_jobs']}`.",
        f"- High-variance flag threshold: population standard deviation > 3.00 percentage points; flagged cells: `{anomalies}`.",
        f"- Incomplete records at report time: `{missing}`.",
        f"- Machine-readable outputs: `{output_root}`.",
        "",
    ])
    return "\n".join(lines)


def _tex_stat(stat: dict[str, Any], decoration: str | None = None) -> str:
    body = f"{float(stat['mean']) * 100.0:.2f} $\\pm$ {float(stat['population_std']) * 100.0:.2f}"
    if decoration == "best":
        return rf"\textbf{{{body}}}"
    if decoration == "second":
        return rf"\underline{{{body}}}"
    return body


def _tex(aggregate: dict[str, Any], rankings: dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Node classification results on Movies, Toys, Grocery, ele-fashion, and Reddit-S. Test Accuracy and Test Macro-F1 are percentages reported as mean $\pm$ population standard deviation over seeds 42, 43, and 44. All models use the frozen transductive full-graph protocol, validation-accuracy checkpoint selection, and the fixed task-level Macro-F1 label set.}",
        r"\label{tab:nc_main}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        "Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{dataset}}}" for dataset in DATASETS) + r" \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
        " & " + " & ".join(["Acc.", "Macro-F1"] * len(DATASETS)) + r" \\",
        r"\midrule",
    ]
    for model in MODELS:
        if model == "mopf":
            lines.append(r"\midrule")
        cells = [DISPLAY_NAMES[model]]
        for dataset in DATASETS:
            for metric in METRICS:
                stat = aggregate[dataset][model][metric]
                rank = rankings[dataset][metric]
                decoration = "best" if rank["best"] == model else "second" if rank["second"] == model else None
                cells.append(_tex_stat(stat, decoration))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def _preview_tex() -> str:
    return r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\begin{document}
\input{table_nc_main.tex}
\end{document}
"""


def _compile_preview(table_root: Path) -> str:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        return "not_available"
    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "table_nc_preview.tex"]
    completed = subprocess.run(command, cwd=table_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    (table_root / "table_nc_preview.log").write_text(completed.stdout, encoding="utf-8")
    return "passed" if completed.returncode == 0 else f"failed_returncode_{completed.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize final NC benchmark")
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "paper_nc_final_fixed_v1"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    records, missing = _load_records(output_root, args.allow_incomplete)
    aggregate = _aggregate(records)
    if not args.allow_incomplete and any(len(aggregate[dataset]) != len(MODELS) for dataset in DATASETS):
        raise RuntimeError("Cannot generate a formal paper table from incomplete aggregate records")
    rankings = _rankings(aggregate)
    health = _health_summary(records)
    attempts = _load_attempts(output_root)
    anomalies = _anomalies(aggregate)
    summary = {
        "status": "PASS" if not missing and health["all_checks_pass"] else "INCOMPLETE",
        "protocol": PROTOCOL,
        "task": "nc",
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "display_names": DISPLAY_NAMES,
        "seeds": list(SEEDS),
        "fixed_macro_f1": {
            "evaluator": "sklearn f1_score with one task-level labels argument",
            "label_source": "valid train + val + test supervised label union",
            "invalid_label_filter": "labels < 0 or labels >= num_classes excluded",
            "same_label_set_for_all_splits_models_seeds": True,
        },
        "aggregate": aggregate,
        "rankings": rankings,
        "mopf_rank_counts": _mopf_counts(rankings),
        "health": health,
        "attempts": attempts,
        "high_variance_flags": anomalies,
        "missing": missing,
        "results_root": str(output_root),
    }
    _dump(output_root / "summary.json", summary)
    for dataset in DATASETS:
        for model in MODELS:
            if model in aggregate[dataset]:
                _dump(output_root / dataset / model / "summary.json", {"dataset": dataset, "model": model, **aggregate[dataset][model]})

    docs_path = ROOT / "docs" / "nc_benchmark_results_fixed_final.md"
    docs_path.write_text(_markdown(aggregate, rankings, health, attempts, anomalies, output_root, missing), encoding="utf-8")
    table_root = ROOT / "paper" / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    (table_root / "table_nc_main.tex").write_text(_tex(aggregate, rankings), encoding="utf-8")
    (table_root / "table_nc_preview.tex").write_text(_preview_tex(), encoding="utf-8")
    compile_status = _compile_preview(table_root)
    _dump(output_root / "report_generation.json", {"status": summary["status"], "markdown": str(docs_path), "latex": str(table_root / "table_nc_main.tex"), "preview": str(table_root / "table_nc_preview.tex"), "preview_compile": compile_status})
    print(json.dumps({"status": summary["status"], "docs": str(docs_path), "latex": str(table_root / "table_nc_main.tex"), "preview_compile": compile_status}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
