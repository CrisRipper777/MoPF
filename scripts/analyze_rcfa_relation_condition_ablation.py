from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_rcfa_relation_condition_ablation as runner

RESULTS = ROOT / "results/rcfa_relation_condition_ablation"
HISTORICAL_CRSA = ROOT / "results/crsa_iatr_v1/validation_results.csv"
HISTORICAL_FULL = ROOT / "results/crsa_rcfa_v2/validation_results.csv"
E0D_CSV = ROOT / "outputs/e0_empirical_motivation/relation_utilization_bridge/relation_utilization_association.csv"
CRSA_SOURCE = "a529f896a85cc25c17a9a6729f3d691a663abab5"
FULL_SOURCE = "fe2daa7cd299c23fd9c49dc8bc0eae9efef59a94"
METRICS = ("val_acc", "val_macro_f1")
VALIDATION_FIELDS = [
    "dataset", "seed", "best_epoch", "val_acc", "val_macro_f1", "parameter_count",
    "runtime_seconds", "peak_gpu_memory_mib", "source_commit", "test_evaluated",
    "status", "reason", "run_dir", "checkpoint",
]
SUMMARY_FIELDS = [
    "dataset", "n", "mean_val_acc", "std_val_acc", "mean_val_macro_f1", "std_val_macro_f1",
]
COMPARISON_FIELDS = [
    "dataset", "seed", "metric", "crsa_value", "rcfa_no_rse_value", "full_rse_value",
    "no_rse_minus_crsa", "full_minus_no_rse", "full_minus_crsa",
]
STATUS_FIELDS = [
    "mode", "variant", "dataset", "seed", "status", "source_commit", "test_evaluated",
    "run_dir", "checkpoint", "reason",
]


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _metric_key_is_test(key: str) -> bool:
    key = key.lower()
    return key.startswith("test_") and key != "test_evaluated"


def _test_metric_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if _metric_key_is_test(column)]


def _value_rows(rows: list[dict[str, str]], *, variant: str | None = None) -> tuple[dict, list[str]]:
    expected = {(dataset, seed) for dataset in runner.DATASETS for seed in runner.SEEDS}
    selected = [row for row in rows if variant is None or row.get("variant") == variant]
    issues = []
    keys = []
    values = {}
    if len(selected) != 15:
        issues.append(f"expected exactly 15 rows, found {len(selected)}")
    for row in selected:
        try:
            key = (row["dataset"], int(row["seed"]))
            keys.append(key)
        except (KeyError, ValueError):
            issues.append("dataset/seed key is malformed")
            continue
        if row.get("status") != "complete":
            issues.append(f"row {key} is not complete")
        if row.get("test_evaluated", "").lower() != "false":
            issues.append(f"row {key} is not validation-only")
        try:
            metrics = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics.values()):
                raise ValueError
            values[key] = metrics
        except (KeyError, TypeError, ValueError):
            issues.append(f"validation metrics are malformed for {key}")
    if len(set(keys)) != 15 or set(keys) != expected:
        issues.append("dataset/seed rows are not exactly the unique five-by-three matrix")
    return values, list(dict.fromkeys(issues))


def validate_history(path: Path, *, variant: str | None, source: str) -> tuple[dict, dict]:
    rows, columns = _read_csv(path)
    values, issues = _value_rows(rows, variant=variant)
    forbidden = _test_metric_columns(columns)
    if forbidden:
        issues.append(f"Test metric columns found: {forbidden}")
    selected = [row for row in rows if variant is None or row.get("variant") == variant]
    for row in selected:
        if row.get("source_commit") != source:
            issues.append(f"unexpected source commit on {row.get('dataset')}/seed{row.get('seed')}")
    return values, {
        "path": str(path.relative_to(ROOT)), "expected_source_commit": source,
        "rows_selected": len(selected), "source_commits": sorted({r.get("source_commit", "") for r in selected}),
        "test_evaluated_values": sorted({r.get("test_evaluated", "") for r in selected}),
        "test_metric_columns": forbidden, "issues": list(dict.fromkeys(issues)),
    }


def inspect_formal_context(context, manifest: dict[str, dict[str, str]]) -> dict:
    entry = manifest.get(context.run_key, {})
    row = {
        "dataset": context.dataset, "seed": context.seed, "best_epoch": "",
        "val_acc": "", "val_macro_f1": "", "parameter_count": "",
        "runtime_seconds": entry.get("runtime_seconds", ""),
        "peak_gpu_memory_mib": entry.get("peak_gpu_memory_mib", ""),
        "source_commit": entry.get("git_commit", ""), "test_evaluated": "false",
        "status": "missing", "reason": "", "run_dir": str(context.run_dir),
        "checkpoint": str(context.checkpoint),
    }
    if not entry:
        row["reason"] = "formal run absent from manifest"
        return row
    if entry.get("status") != "complete":
        row.update(status=entry.get("status") or "missing", reason=entry.get("error_reason", "run incomplete"))
        return row
    valid, reason, payload = runner.audit_context(context)
    if not valid:
        row.update(status="invalid", reason=reason)
        return row
    if entry.get("git_branch") != runner.BRANCH or entry.get("git_worktree_clean") != "true":
        row.update(status="invalid", reason="branch or clean-worktree provenance mismatch")
        return row
    if entry.get("test_evaluated", "").lower() != "false":
        row.update(status="invalid", test_evaluated=entry.get("test_evaluated"), reason="manifest reports Test evaluation")
        return row

    import torch
    from omegaconf import OmegaConf
    from src.models.crsa_rcfa import Model

    config = _read_json(context.run_dir / "resolved_config.json")
    metrics_file = _read_json(context.run_dir / "metrics.json")
    try:
        checkpoint = torch.load(context.checkpoint, map_location="cpu", weights_only=False)
        data_info = checkpoint["data_info"]
        model = Model(OmegaConf.create(config), data_info)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        head = torch.nn.Linear(model.out_dim, int(data_info["num_classes"]))
        head.load_state_dict(checkpoint["head_state"], strict=True)
        parameter_count += sum(p.numel() for p in head.parameters() if p.requires_grad)
        metrics = payload["metrics"]
        values = {metric: float(metrics[metric]["mean"]) for metric in METRICS}
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
            raise ValueError("validation metric is non-finite or outside [0,1]")
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        row.update(status="invalid", reason=f"strict artifact audit failed: {exc}")
        return row
    row.update(status="complete", best_epoch=metrics_file.get("best_epoch", ""),
               val_acc=values["val_acc"], val_macro_f1=values["val_macro_f1"],
               parameter_count=parameter_count,
               runtime_seconds=metrics_file.get("runtime_seconds", row["runtime_seconds"]),
               peak_gpu_memory_mib=metrics_file.get("peak_gpu_memory_mib", row["peak_gpu_memory_mib"]),
               reason="")
    return row


def summary_rows(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in runner.DATASETS:
        group = [row for row in rows if row["dataset"] == dataset and row["status"] == "complete"]
        if not group:
            continue
        acc = [float(row["val_acc"]) for row in group]
        f1 = [float(row["val_macro_f1"]) for row in group]
        output.append({
            "dataset": dataset, "n": len(group),
            "mean_val_acc": statistics.mean(acc),
            "std_val_acc": statistics.stdev(acc) if len(acc) > 1 else "",
            "mean_val_macro_f1": statistics.mean(f1),
            "std_val_macro_f1": statistics.stdev(f1) if len(f1) > 1 else "",
        })
    return output


def three_way_rows(crsa: dict, no_rse: dict, full: dict) -> list[dict]:
    expected = {(dataset, seed) for dataset in runner.DATASETS for seed in runner.SEEDS}
    if any(set(group) != expected for group in (crsa, no_rse, full)):
        raise ValueError("three-way comparison requires matching 15 dataset/seed pairs")
    rows = []
    for dataset in runner.DATASETS:
        for seed in runner.SEEDS:
            key = (dataset, seed)
            for metric in METRICS:
                c, n, f = crsa[key][metric], no_rse[key][metric], full[key][metric]
                rows.append({
                    "dataset": dataset, "seed": seed, "metric": metric,
                    "crsa_value": c, "rcfa_no_rse_value": n, "full_rse_value": f,
                    "no_rse_minus_crsa": n - c, "full_minus_no_rse": f - n,
                    "full_minus_crsa": f - c,
                })
        for metric in METRICS:
            group = [row for row in rows if row["dataset"] == dataset and row["metric"] == metric]
            means = {field: statistics.mean(float(row[field]) for row in group) for field in (
                "crsa_value", "rcfa_no_rse_value", "full_rse_value", "no_rse_minus_crsa",
                "full_minus_no_rse", "full_minus_crsa",
            )}
            rows.append({"dataset": dataset, "seed": "MEAN", "metric": metric, **means})
    return rows


def _fmt_mean_std(mean, std, scale=100.0) -> str:
    if mean in (None, ""):
        return "n/a"
    if std in (None, ""):
        return f"{scale * float(mean):.2f}"
    return f"{scale * float(mean):.2f} ± {scale * float(std):.2f}"


def _fmt_delta(value) -> str:
    return f"{100 * float(value):+.2f}" if value not in (None, "") else "n/a"


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for pos in order[i:j]:
            ranks[pos] = rank
        i = j
    return ranks


def _corr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def exploratory_e0d(comparison_rows: list[dict]) -> dict:
    if not E0D_CSV.is_file():
        return {"available": False, "reason": "exact E0-D CSV artifact not found; markdown values were not reconstructed"}
    data, columns = _read_csv(E0D_CSV)
    if "rho_partial_degree" not in columns or "dataset" not in columns:
        return {"available": False, "reason": "exact CSV lacks required dataset/rho_partial_degree columns"}
    grouped = defaultdict(list)
    for row in data:
        try:
            if row["dataset"] in runner.DATASETS:
                value = float(row["rho_partial_degree"])
                if math.isfinite(value):
                    grouped[row["dataset"]].append(value)
        except (KeyError, ValueError, TypeError):
            continue
    if any(not grouped[dataset] for dataset in runner.DATASETS):
        return {"available": False, "reason": "CSV does not contain finite rho values for all five datasets"}
    rho = {dataset: statistics.mean(grouped[dataset]) for dataset in runner.DATASETS}
    means = {(row["dataset"], row["metric"]): row for row in comparison_rows if row["seed"] == "MEAN"}
    dataset_rows = []
    correlations = {}
    for metric in METRICS:
        delta = {dataset: float(means[(dataset, metric)]["full_minus_no_rse"])
                 for dataset in runner.DATASETS}
        rho_values = [rho[d] for d in runner.DATASETS]
        delta_values = [delta[d] for d in runner.DATASETS]
        correlations[metric] = {
            "pearson_descriptive": _corr(rho_values, delta_values),
            "spearman_descriptive": _corr(_rank(rho_values), _rank(delta_values)),
            "higher_to_lower_rho_order": sorted(runner.DATASETS, key=lambda d: rho[d], reverse=True),
            "higher_to_lower_delta_order": sorted(runner.DATASETS, key=lambda d: delta[d], reverse=True),
        }
    for dataset in runner.DATASETS:
        dataset_rows.append({
            "dataset": dataset, "mean_degree_controlled_partial_rho": rho[dataset],
            "full_minus_no_rse_accuracy": means[(dataset, "val_acc")]["full_minus_no_rse"],
            "full_minus_no_rse_macro_f1": means[(dataset, "val_macro_f1")]["full_minus_no_rse"],
        })
    return {
        "available": True, "source_csv": str(E0D_CSV.relative_to(ROOT)),
        "n_datasets": 5, "exploratory_only": True,
        "not_statistical_evidence": True, "not_a_paper_claim": True,
        "aggregation": "mean rho_partial_degree across available seed x modality rows within dataset",
        "datasets": dataset_rows, "correlations": correlations,
    }


def _report(preflight: dict | None, audit: dict, result_rows: list[dict],
            summary: list[dict], comparisons: list[dict], e0d: dict,
            manifest_rows: list[dict[str, str]]) -> str:
    complete = [row for row in result_rows if row["status"] == "complete"]
    summary_map = {row["dataset"]: row for row in summary}
    mean_map = {(row["dataset"], row["metric"]): row for row in comparisons if row["seed"] == "MEAN"}
    commits = audit.get("formal_source_commits", [])
    runtimes = [float(row["runtime_seconds"]) for row in complete if row.get("runtime_seconds") not in ("", None)]
    memories = [float(row["peak_gpu_memory_mib"]) for row in complete if row.get("peak_gpu_memory_mib") not in ("", None)]
    failed = [row for row in result_rows if row["status"] != "complete"]
    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for entry in manifest_rows
        for path in (Path(entry.get("run_dir", "")) / "launcher_output.log",
                     Path(entry.get("run_dir", "")) / "main.log") if path.is_file()
    )
    oom_count = len(re.findall(r"out of memory|cuda oom", log_text, re.I))
    nan_count = len(re.findall(r"\b(?:nan|inf)\b", log_text, re.I))
    lines = [
        "# RCFA Explicit Relation-Effect Attribution",
        "",
        "## 1. Research question",
        "",
        "Does passing Stage-I Relation Semantic Effect (RSE) explicitly into the Stage-II RCFA conditioner add value, with RCFA and all Stage-I computation held fixed? The ablation is named **w/o Explicit Relation Effect**: RSE is still computed, but its RCFA conditioner input is zero.",
        "",
        "## 2. Provenance and protocol",
        "",
        f"- Starting source branch/SHA: `crsa_rcfa_v2` / `{audit.get('starting_sha', 'unknown')}`.",
        f"- Experiment branch: `{runner.BRANCH}`.",
        f"- Formal source SHA: `{commits[0] if len(commits) == 1 else 'not uniform/unavailable'}`.",
        f"- Result commit: `{audit.get('results_commit', 'written after analysis')}`.",
        f"- Formal contexts complete: {len(complete)} / 15.",
        "- Matrix: NC, full-graph, AdamW, lr=1e-3, weight decay=1e-4, up to 300 epochs, patience=30, min epoch=30, min delta=1e-4, grad clip=1.0, evaluation every epoch, full inference, no scheduler, K=3, best validation accuracy checkpoint.",
        "- Validation only: `evaluate_test=false`; modality-mask evaluation disabled. No LP context was launched.",
        "",
        "## 3. Implementation-control audit",
        "",
        f"- Preflight status: `{(preflight or {}).get('status', 'missing')}`.",
        f"- Trainable parameters, including the NC head (Full/noRSE): {(preflight or {}).get('parameter_count_including_nc_head_full', 'n/a')} / {(preflight or {}).get('parameter_count_including_nc_head_noRSE', 'n/a')}; model-only counts: {(preflight or {}).get('parameter_count_full', 'n/a')} / {(preflight or {}).get('parameter_count_noRSE', 'n/a')}.",
        f"- Same-seed initialization tensors identical: {(preflight or {}).get('initialization_tensors_identical', 'n/a')}.",
        f"- CRSA states, H0, delta, RSE and captured conditioner arguments passed: {all((preflight or {}).get('stage_i_state_and_effect_checks', {}).values()) if preflight else 'n/a'}.",
        "- The only ablated input is R_k at the RCFA conditioner; noRSE supplies `zeros_like(R_k)`. CRSA propagation and RSE/delta computation remain active.",
        "- Model and YAML source files are frozen at the starting snapshot; preflight records their Git blob hashes.",
        "",
        "## 4. RCFA-noRSE validation results",
        "",
        "Accuracy and Macro-F1 are mean ± sample standard deviation across three seeds, in percent.",
        "",
        "| Dataset | n | Val Accuracy (%) | Val Macro-F1 (%) |",
        "|---|---:|---:|---:|",
    ]
    for dataset in runner.DATASETS:
        row = summary_map.get(dataset)
        lines.append(f"| {dataset} | {row['n']} | {_fmt_mean_std(row['mean_val_acc'], row['std_val_acc'])} | {_fmt_mean_std(row['mean_val_macro_f1'], row['std_val_macro_f1'])} |" if row else f"| {dataset} | 0 | n/a | n/a |")

    for metric, label in (("val_acc", "Accuracy"), ("val_macro_f1", "Macro-F1")):
        lines.extend(["", f"## 5. Three-way comparison — {label}", "",
                      "Values and deltas are percentage points; first three columns are the mean validation scores across seeds.", "",
                      "| Dataset | CRSA | RCFA w/o RSE | Full RCFA+RSE | noRSE−CRSA | Full−noRSE | Full−CRSA |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for dataset in runner.DATASETS:
            row = mean_map.get((dataset, metric))
            if row:
                lines.append("| {} | {:.2f} | {:.2f} | {:.2f} | {} | {} | {} |".format(
                    dataset, 100 * row["crsa_value"], 100 * row["rcfa_no_rse_value"],
                    100 * row["full_rse_value"], _fmt_delta(row["no_rse_minus_crsa"]),
                    _fmt_delta(row["full_minus_no_rse"]), _fmt_delta(row["full_minus_crsa"])))
            else:
                lines.append(f"| {dataset} | n/a | n/a | n/a | n/a | n/a | n/a |")

        lines.extend(["", f"### Paired seed values — {label}", "",
                      "| Dataset | Seed | CRSA | RCFA w/o RSE | Full RCFA+RSE | noRSE−CRSA | Full−noRSE | Full−CRSA |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for dataset in runner.DATASETS:
            for seed in runner.SEEDS:
                row = next((r for r in comparisons if r["dataset"] == dataset and r["seed"] == seed and r["metric"] == metric), None)
                if row:
                    lines.append("| {} | {} | {:.2f} | {:.2f} | {:.2f} | {} | {} | {} |".format(
                        dataset, seed, 100 * row["crsa_value"], 100 * row["rcfa_no_rse_value"],
                        100 * row["full_rse_value"], _fmt_delta(row["no_rse_minus_crsa"]),
                        _fmt_delta(row["full_minus_no_rse"]), _fmt_delta(row["full_minus_crsa"])))
                else:
                    lines.append(f"| {dataset} | {seed} | n/a | n/a | n/a | n/a | n/a | n/a |")

    lines.extend(["", "## 6. Attribution analysis", "",
                  "- **Absorption effect** is `RCFA-noRSE − CRSA`; **explicit RSE effect** is `Full − RCFA-noRSE`; **total Stage-II effect** is `Full − CRSA`. These are paired by dataset and seed.", ""])
    for metric, label in (("val_acc", "Accuracy"), ("val_macro_f1", "Macro-F1")):
        if not mean_map:
            lines.append(f"- {label}: three-way attribution unavailable because a provenance or completeness audit failed.")
            continue
        neg_no = []
        full_better = []
        full_worse = []
        for dataset in runner.DATASETS:
            row = mean_map[(dataset, metric)]
            if row["no_rse_minus_crsa"] < 0:
                neg_no.append(dataset)
            if row["full_minus_no_rse"] > 0:
                full_better.append(dataset)
            elif row["full_minus_no_rse"] < 0:
                full_worse.append(dataset)
        lines.append(f"- {label}: noRSE mean below CRSA on {', '.join(neg_no) if neg_no else 'none'}; explicit RSE mean helps Full on {', '.join(full_better) if full_better else 'none'} and lowers it on {', '.join(full_worse) if full_worse else 'none'}. Inspect the paired tables for seed consistency.")

    lines.extend(["", "## 7. Dataset heterogeneity", "",
                  "Dataset means and seed-level differences are reported separately above. Directions are descriptive; no across-dataset average is used as a substitute for the five per-dataset patterns.", "",
                  "## 8. Optional E0-D exploratory correspondence", ""])
    if e0d.get("available"):
        lines.append(f"Exact artifact: `{e0d['source_csv']}`. Dataset-level mean degree-controlled partial rho is compared with Full−noRSE deltas. This is exploratory only (n=5 datasets), not statistical evidence, and not a paper claim.")
        lines.extend(["", "| Dataset | Mean degree-controlled partial rho | Full−noRSE Accuracy (pp) | Full−noRSE Macro-F1 (pp) |",
                      "|---|---:|---:|---:|"])
        for row in e0d["datasets"]:
            lines.append(f"| {row['dataset']} | {row['mean_degree_controlled_partial_rho']:.3f} | {_fmt_delta(row['full_minus_no_rse_accuracy'])} | {_fmt_delta(row['full_minus_no_rse_macro_f1'])} |")
        for metric, label in (("val_acc", "Accuracy"), ("val_macro_f1", "Macro-F1")):
            c = e0d["correlations"][metric]
            lines.append(f"- {label}: descriptive Pearson={c['pearson_descriptive']:.3f}, Spearman={c['spearman_descriptive']:.3f}; rho order: {', '.join(c['higher_to_lower_rho_order'])}; delta order: {', '.join(c['higher_to_lower_delta_order'])}.")
    else:
        lines.append(e0d.get("reason", "Exact E0-D CSV artifact unavailable; skipped without reconstructing values from markdown."))

    lines.extend(["", "## 9. Engineering audit", "",
                  f"- Complete formal runs: {len(complete)}/15; invalid or incomplete: {len(failed)}.",
                  f"- Total recorded runtime: {sum(runtimes)/3600:.2f} GPU-hours across sequential runs; mean per run: {statistics.mean(runtimes):.1f} s." if runtimes else "- Runtime metadata unavailable.",
                  f"- Maximum recorded peak GPU memory: {max(memories):.1f} MiB." if memories else "- Peak GPU memory metadata unavailable.",
                  f"- OOM signatures found in captured logs: {oom_count}; NaN/Inf tokens found: {nan_count}.",
                  "- Artifact audit requires finite checkpoint tensors and train losses, validation-only metric keys, resolved protocol agreement, and strict model/head checkpoint reload.",
                  "- Test metrics were neither evaluated nor analyzed; no LP runs were included.",
                  "",
                  "## 10. Research conclusion and recommended next step", ""])
    if mean_map:
        component_directions = {}
        for metric, label in (("val_acc", "Accuracy"), ("val_macro_f1", "Macro-F1")):
            mean_rows = [mean_map[(dataset, metric)] for dataset in runner.DATASETS]
            absorption_losses = [runner.DATASETS[i] for i, row in enumerate(mean_rows)
                                 if row["no_rse_minus_crsa"] < 0]
            absorption_stable = [runner.DATASETS[i] for i, row in enumerate(mean_rows)
                                 if all(r["no_rse_minus_crsa"] < 0 for r in comparisons
                                        if r["dataset"] == runner.DATASETS[i] and r["metric"] == metric and r["seed"] != "MEAN")]
            rse_gain = [runner.DATASETS[i] for i, row in enumerate(mean_rows)
                        if row["full_minus_no_rse"] > 0]
            rse_loss = [runner.DATASETS[i] for i, row in enumerate(mean_rows)
                        if row["full_minus_no_rse"] < 0]
            rse_stable_gain, rse_stable_loss = [], []
            for dataset in runner.DATASETS:
                seed_effects = [r["full_minus_no_rse"] for r in comparisons
                                if r["dataset"] == dataset and r["metric"] == metric and r["seed"] != "MEAN"]
                if seed_effects and all(value > 0 for value in seed_effects):
                    rse_stable_gain.append(dataset)
                if seed_effects and all(value < 0 for value in seed_effects):
                    rse_stable_loss.append(dataset)
            component_directions[metric] = (mean_rows, absorption_losses, absorption_stable,
                                            rse_gain, rse_loss, rse_stable_gain, rse_stable_loss)
            lines.append(f"- **{label}:** RCFA-noRSE mean is below CRSA on {', '.join(absorption_losses) if absorption_losses else 'none'}; the decrease is negative in all three seeds on {', '.join(absorption_stable) if absorption_stable else 'none'}. Full-RSE mean is above noRSE on {', '.join(rse_gain) if rse_gain else 'none'}, and below it on {', '.join(rse_loss) if rse_loss else 'none'}; all-seed RSE gains occur on {', '.join(rse_stable_gain) if rse_stable_gain else 'none'}, all-seed losses on {', '.join(rse_stable_loss) if rse_stable_loss else 'none'}.")

        acc_rows, acc_absorption_losses, _, acc_rse_gain, acc_rse_loss = component_directions["val_acc"]
        lines.append(f"- **Q1 — Does RCFA absorption itself cause negative transfer?** On validation accuracy, noRSE is below CRSA on {', '.join(acc_absorption_losses) if acc_absorption_losses else 'none'}; all-seed decreases occur on {', '.join(component_directions['val_acc'][2]) if component_directions['val_acc'][2] else 'none'}. Macro-F1 directions are reported in the preceding line. This isolates absorption without explicit RSE.")
        lines.append(f"- **Q2 — Does explicit RSE conditioning add value?** Its mean accuracy contribution is positive on {', '.join(acc_rse_gain) if acc_rse_gain else 'none'} and negative on {', '.join(acc_rse_loss) if acc_rse_loss else 'none'}; see the preceding line for the Macro-F1 result and the paired tables for seed consistency.")
        dominant_absorption, dominant_rse = [], []
        for dataset in runner.DATASETS:
            row = mean_map[(dataset, "val_acc")]
            if abs(row["no_rse_minus_crsa"]) > abs(row["full_minus_no_rse"]):
                dominant_absorption.append(dataset)
            elif abs(row["full_minus_no_rse"]) > abs(row["no_rse_minus_crsa"]):
                dominant_rse.append(dataset)
        lines.append(f"- **Q3 — What drives the dataset split?** By absolute mean accuracy contrast, the larger component is absorption on {', '.join(dominant_absorption) if dominant_absorption else 'none'} and explicit RSE conditioning on {', '.join(dominant_rse) if dominant_rse else 'none'} (ties omitted). This describes which contrast accounts for more of the observed dataset variation in this matrix.")
        if acc_absorption_losses and acc_rse_gain:
            recommendation = "Retain RCFA as a dataset-conditional Stage-II candidate: the noRSE contrast and/or the explicit RSE contrast changes sign across datasets, so no universal-benefit claim is supported."
        elif len(acc_absorption_losses) >= 3:
            recommendation = "Do not treat the current RCFA absorber as a generally safe Stage II; the noRSE contrast is negative on most datasets, so a later Stage-II absorption redesign is more directly motivated than RSE reliability micro-analysis."
        else:
            recommendation = "Retain the present Stage-II realization for now only where the paired contrasts support it; this round does not establish a universal benefit."
        lines.append(f"- **Q4 — Retain RCFA as Stage II?** {recommendation}")
        if acc_absorption_losses and acc_rse_gain and acc_rse_loss:
            next_step = "Further study relation-effect reliability, because explicit RSE conditioning helps some datasets and hurts others while RCFA-noRSE provides the absorption control."
        elif len(acc_absorption_losses) >= 3:
            next_step = "Reconsider Stage-II absorption before further RSE micro-analysis, because RCFA-noRSE is below CRSA on most datasets."
        else:
            next_step = "Keep the current design as a conditional candidate; this experiment alone does not justify another architecture change."
        lines.append(f"- **Q5 — Next step?** {next_step} No subsequent experiment was started; recommendation is based only on these validation results.")
    else:
        lines.append("Three-way interpretation is withheld because provenance or completeness checks failed; repair the audit before attributing any effect.")
    lines.extend(["", "No automatic scientific PASS/FAIL threshold was applied. The comparison is a controlled decomposition; it does not require a monotonic Full > noRSE > CRSA ordering.", ""])
    return "\n".join(lines)


def analyze() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_rows, manifest_columns = _read_csv(runner.OUTPUT / "formal_run_manifest.csv")
    manifest = {row.get("run_key", ""): row for row in manifest_rows}
    expected = {context.run_key for context in runner.formal_contexts()}
    manifest_issues = []
    if len(manifest_rows) != 15 or set(manifest) != expected:
        manifest_issues.append("formal manifest is not exactly the expected 15 unique contexts")
    if any(column.startswith("test_") and column != "test_evaluated" for column in manifest_columns):
        manifest_issues.append("Test metric fields found in manifest")
    if any(row.get("git_branch") != runner.BRANCH for row in manifest_rows):
        manifest_issues.append("manifest contains a run from another branch")
    if any(row.get("git_worktree_clean") != "true" for row in manifest_rows):
        manifest_issues.append("manifest contains a dirty-source run")
    formal_commits = {row.get("git_commit", "") for row in manifest_rows}
    if len(formal_commits) != 1 or not next(iter(formal_commits), ""):
        manifest_issues.append("formal contexts do not share one source commit")
    if len({row.get("test_evaluated", "").lower() for row in manifest_rows}) != 1 or any(row.get("test_evaluated", "").lower() != "false" for row in manifest_rows):
        manifest_issues.append("manifest does not consistently record validation-only runs")

    result_rows = [inspect_formal_context(context, manifest) for context in runner.formal_contexts()]
    _write_csv(RESULTS / "validation_results.csv", VALIDATION_FIELDS, result_rows)
    status_rows = [{
        "mode": "formal", "variant": runner.VARIANT, "dataset": row["dataset"], "seed": row["seed"],
        "status": row["status"], "source_commit": row["source_commit"],
        "test_evaluated": row["test_evaluated"], "run_dir": row["run_dir"],
        "checkpoint": row["checkpoint"], "reason": row["reason"],
    } for row in result_rows]
    _write_csv(RESULTS / "run_status.csv", STATUS_FIELDS, status_rows)
    summary = summary_rows(result_rows)
    _write_csv(RESULTS / "validation_summary.csv", SUMMARY_FIELDS, summary)

    crsa, crsa_audit = validate_history(HISTORICAL_CRSA, variant="crsa", source=CRSA_SOURCE)
    full, full_audit = validate_history(HISTORICAL_FULL, variant=None, source=FULL_SOURCE)
    no_rse, no_rse_issues = _value_rows(
        [{**row, "status": row["status"], "test_evaluated": row["test_evaluated"]}
         for row in result_rows],
    )
    no_rse_audit = {
        "path": str((RESULTS / "validation_results.csv").relative_to(ROOT)),
        "expected_source_commit": next(iter(formal_commits), ""),
        "rows_selected": len(result_rows), "source_commits": sorted(formal_commits),
        "test_evaluated_values": sorted({row["test_evaluated"] for row in result_rows}),
        "issues": no_rse_issues,
    }
    preflight = _read_json(RESULTS / "preflight_audit.json")
    if not preflight or preflight.get("status") != "passed":
        manifest_issues.append("preflight audit is missing or did not pass")
    if any(row["status"] != "complete" for row in result_rows):
        manifest_issues.append("not all 15 noRSE formal contexts passed artifact audit")
    if len(formal_commits) == 1 and next(iter(formal_commits), ""):
        for row in result_rows:
            if row["source_commit"] != next(iter(formal_commits)):
                manifest_issues.append("a noRSE run's recorded source commit differs from the formal source SHA")
                break

    provenance_ok = not (manifest_issues or crsa_audit["issues"] or full_audit["issues"]
                         or no_rse_audit["issues"])
    comparisons = three_way_rows(crsa, no_rse, full) if provenance_ok else []
    _write_csv(RESULTS / "three_way_comparison.csv", COMPARISON_FIELDS, comparisons)
    e0d = exploratory_e0d(comparisons) if comparisons else {
        "available": False, "reason": "skipped because three-way provenance audit did not pass"
    }
    audit = {
        "status": "valid" if provenance_ok else "invalid",
        "starting_branch": "crsa_rcfa_v2", "starting_sha": "ed2a4174d806d1695c753e8926b386075f77fee2",
        "experiment_branch": runner.BRANCH,
        "formal_source_commits": sorted(formal_commits),
        "formal_source_commit_uniform": len(formal_commits) == 1,
        "formal_contexts_complete": sum(row["status"] == "complete" for row in result_rows),
        "formal_context_count": len(result_rows), "formal_manifest_issues": list(dict.fromkeys(manifest_issues)),
        "crsa_v1": crsa_audit, "full_v2": full_audit, "rcfa_no_rse": no_rse_audit,
        "comparisons_available": provenance_ok,
        "comparison_pairing": "same dataset and seed across all three sources" if provenance_ok else None,
        "test_metrics_accessed": False, "test_labels_used_or_indexed": False, "lp_runs": 0,
        "e0d_exploratory": e0d,
    }
    (RESULTS / "provenance_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    report = _report(preflight, audit, result_rows, summary, comparisons, e0d, manifest_rows)
    (ROOT / "docs/rcfa_relation_condition_ablation_report.md").write_text(report, encoding="utf-8")
    if not provenance_ok:
        print("Three-way comparison withheld; see provenance_audit.json")
    return audit


def main() -> None:
    argparse.ArgumentParser(description="Analyze the explicit RSE conditioning ablation").parse_args()
    audit = analyze()
    print(json.dumps(audit, indent=2))
    if audit["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
