from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from scripts import run_crsa_icsr_nc as runner
from src.models.crsa_icsr import Model

RESULTS = ROOT / "results/crsa_icsr_v3"
HISTORICAL = {
    "crsa": (ROOT / "results/crsa_iatr_v1/validation_results.csv",
             "a529f896a85cc25c17a9a6729f3d691a663abab5", "variant", "crsa"),
    "rcfa_full": (ROOT / "results/crsa_rcfa_v2/validation_results.csv",
                  "fe2daa7cd299c23fd9c49dc8bc0eae9efef59a94", None, None),
    "rcfa_no_rse": (ROOT / "results/rcfa_relation_condition_ablation/validation_results.csv",
                    "d80869c9d8450ff0ece25fa3773c7349d82d50c0", None, None),
}
METRICS = (("val_acc", "Accuracy"), ("val_macro_f1", "Macro-F1"))
EXPECTED_PAIRS = {(d, s) for d in runner.DATASETS for s in runner.SEEDS}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def parse_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def validate_source_rows(name: str, rows: list[dict], expected_commit: str,
                         filter_key: str | None = None, filter_value: str | None = None) -> tuple[dict, dict[tuple[str, int], dict]]:
    selected = [row for row in rows if filter_key is None or row.get(filter_key) == filter_value]
    observed_commits = sorted({str(row.get("source_commit", "")) for row in selected})
    problems = []
    if len(selected) != 15:
        problems.append(f"expected 15 selected rows, found {len(selected)}")
    keyed: dict[tuple[str, int], dict] = {}
    for row in selected:
        try:
            key = (str(row["dataset"]), int(row["seed"]))
        except (KeyError, ValueError, TypeError):
            problems.append("row has missing/invalid dataset or seed")
            continue
        if key in keyed:
            problems.append(f"duplicate pair {key}")
        keyed[key] = row
        if row.get("status") != "complete":
            problems.append(f"{key}: status={row.get('status')!r}")
        if parse_bool(row.get("test_evaluated", "")):
            problems.append(f"{key}: test_evaluated=true")
        if str(row.get("source_commit", "")) != expected_commit:
            problems.append(f"{key}: source commit mismatch")
        for col in ("val_acc", "val_macro_f1"):
            try:
                value = float(row[col])
            except (KeyError, ValueError, TypeError):
                problems.append(f"{key}: missing metric {col}")
                continue
            if not math.isfinite(value) or not 0 <= value <= 1:
                problems.append(f"{key}: invalid metric {col}")
    if set(keyed) != EXPECTED_PAIRS:
        problems.append(f"dataset/seed matrix mismatch: found {sorted(keyed)}")
    audit = {
        "path": str(rows[0].get("_source_path", "")) if rows else "",
        "expected_source_commit": expected_commit,
        "selected_rows": len(selected),
        "source_commits": observed_commits,
        "test_evaluated_false": all(not parse_bool(r.get("test_evaluated", "")) for r in selected),
        "status": "valid" if not problems else "invalid",
        "problems": problems,
    }
    return audit, keyed


def load_historical() -> tuple[dict[str, dict[tuple[str, int], dict]], dict]:
    loaded, audits = {}, {}
    for name, (path, sha, filter_key, filter_value) in HISTORICAL.items():
        if not path.is_file():
            rows = []
            audits[name] = {"path": str(path), "expected_source_commit": sha,
                            "selected_rows": 0, "source_commits": [], "status": "invalid",
                            "problems": ["file is missing"]}
            loaded[name] = {}
            continue
        rows = read_csv(path)
        for row in rows:
            row["_source_path"] = str(path)
        audit, keyed = validate_source_rows(name, rows, sha, filter_key, filter_value)
        audit["path"] = str(path)
        audits[name] = audit
        loaded[name] = keyed
    return loaded, audits


def parameter_count(config: dict, checkpoint: dict) -> int:
    info = checkpoint.get("data_info")
    if not isinstance(info, dict):
        raise ValueError("checkpoint lacks data_info")
    model = Model(OmegaConf.create(config), info)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head = torch.nn.Linear(model.out_dim, int(info["num_classes"]))
    head.load_state_dict(checkpoint["head_state"], strict=True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(
        p.numel() for p in head.parameters() if p.requires_grad
    )


def run_rows() -> tuple[list[dict], list[dict], dict]:
    manifest_path = runner.OUTPUT / "formal_run_manifest.csv"
    manifest = read_csv(manifest_path) if manifest_path.is_file() else []
    formal = [r for r in manifest if r.get("mode") == "formal"]
    by_key = {r.get("run_key"): r for r in formal}
    if len(formal) != 15 or len(by_key) != 15:
        formal_count_problem = f"formal manifest has {len(formal)} rows and {len(by_key)} unique keys; expected 15"
    else:
        formal_count_problem = ""
    validation, status_rows, source_commits = [], [], set()
    source_pairs: dict[tuple[str, int], str] = {}
    for dataset in runner.DATASETS:
        for seed in runner.SEEDS:
            context = runner.RunContext(dataset, seed)
            mrow = by_key.get(context.run_key)
            row = {
                "dataset": dataset, "seed": seed, "best_epoch": "", "val_acc": "",
                "val_macro_f1": "", "parameter_count": "", "runtime_seconds": "",
                "peak_gpu_memory_mib": "", "source_commit": "", "test_evaluated": "false",
                "status": "missing", "reason": "formal context is absent from run manifest",
                "run_dir": str(context.run_dir), "checkpoint": str(context.checkpoint),
            }
            status = {"mode": "formal", "variant": "full", "dataset": dataset, "seed": seed,
                      "status": "missing", "source_commit": "", "test_evaluated": "false",
                      "run_dir": str(context.run_dir), "checkpoint": str(context.checkpoint), "reason": row["reason"]}
            if mrow:
                source = str(mrow.get("git_commit", ""))
                source_pairs[(dataset, seed)] = source
                row.update(source_commit=source, runtime_seconds=mrow.get("runtime_seconds", ""),
                           peak_gpu_memory_mib=mrow.get("peak_gpu_memory_mib", ""),
                           test_evaluated=mrow.get("test_evaluated", ""))
                status.update(source_commit=source, test_evaluated=mrow.get("test_evaluated", ""))
                if mrow.get("status") != "complete":
                    row.update(status=mrow.get("status") or "incomplete",
                               reason=mrow.get("error_reason", "launcher status not complete"))
                    status.update(status=row["status"], reason=row["reason"])
                elif mrow.get("git_branch") != runner.BRANCH:
                    row.update(status="invalid", reason="manifest branch mismatch")
                    status.update(status="invalid", reason=row["reason"])
                elif mrow.get("git_worktree_clean", "").lower() != "true":
                    row.update(status="invalid", reason="formal source was not clean")
                    status.update(status="invalid", reason=row["reason"])
                elif mrow.get("test_evaluated", "").lower() != "false":
                    row.update(status="invalid", reason="manifest reports Test evaluation")
                    status.update(status="invalid", reason=row["reason"])
                else:
                    valid, reason, payload = runner.audit_context(context)
                    if not valid:
                        row.update(status="invalid", reason=reason)
                        status.update(status="invalid", reason=reason)
                    else:
                        try:
                            checkpoint = torch.load(context.checkpoint, map_location="cpu", weights_only=False)
                            config = runner.read_json(context.run_dir / "resolved_config.json")
                            metrics = payload["metrics"]
                            count = parameter_count(config, checkpoint)
                            row.update(
                                best_epoch=payload.get("best_epoch", ""),
                                val_acc=float(metrics["val_acc"]["mean"]),
                                val_macro_f1=float(metrics["val_macro_f1"]["mean"]),
                                parameter_count=count, status="complete", reason="",
                            )
                            status.update(status="complete", reason="")
                        except (OSError, RuntimeError, EOFError, ValueError, KeyError, TypeError) as exc:
                            row.update(status="invalid", reason=f"result extraction failed: {exc}")
                            status.update(status="invalid", reason=row["reason"])
            if row["status"] == "complete":
                source_commits.add(str(row["source_commit"]))
            validation.append(row)
            status_rows.append(status)
    uniform = len(source_commits) == 1 and all(
        source_pairs.get(pair) == next(iter(source_commits), "") for pair in EXPECTED_PAIRS
    )
    audit = {
        "formal_source_commits": sorted(source_commits),
        "formal_source_commit_uniform": uniform,
        "formal_context_count": len(formal),
        "formal_unique_context_count": len(by_key),
        "formal_manifest_problem": formal_count_problem,
        "complete_validation_rows": sum(r["status"] == "complete" for r in validation),
        "all_formal_worktrees_clean": all(
            r.get("git_worktree_clean", "").lower() == "true" for r in formal
        ),
        "test_evaluated_false": all(r.get("test_evaluated", "").lower() == "false" for r in formal),
        "lp_run": False,
        "formal_runs": [
            {"run_key": r.get("run_key"), "dataset": r.get("dataset"), "seed": r.get("seed"),
             "source_commit": r.get("git_commit"), "branch": r.get("git_branch"),
             "worktree_clean": r.get("git_worktree_clean"), "status": r.get("status"),
             "test_evaluated": r.get("test_evaluated")}
            for r in sorted(formal, key=lambda v: v.get("run_key", ""))
        ],
    }
    return validation, status_rows, audit


def grouped(rows: list[dict], field: str) -> dict[str, list[dict]]:
    result = {dataset: [] for dataset in runner.DATASETS}
    for row in rows:
        if row.get("status") == "complete":
            result[row["dataset"]].append(row)
    return result


def summary_rows(rows: list[dict]) -> list[dict]:
    groups = grouped(rows, "val_acc")
    output = []
    for dataset in runner.DATASETS:
        group = groups[dataset]
        acc = [float(r["val_acc"]) for r in group]
        f1 = [float(r["val_macro_f1"]) for r in group]
        output.append({
            "dataset": dataset, "n": len(group),
            "mean_val_acc": statistics.mean(acc) if acc else "",
            "std_val_acc": statistics.stdev(acc) if len(acc) > 1 else "",
            "mean_val_macro_f1": statistics.mean(f1) if f1 else "",
            "std_val_macro_f1": statistics.stdev(f1) if len(f1) > 1 else "",
        })
    return output


def four_way_rows(hist: dict, v3: dict[tuple[str, int], dict]) -> list[dict]:
    output = []
    complete = {}
    for dataset, seed in sorted(EXPECTED_PAIRS):
        values = {}
        for name in ("crsa", "rcfa_no_rse", "rcfa_full"):
            try:
                values[name] = {metric: float(hist[name][(dataset, seed)][metric]) for metric, _ in METRICS}
            except (KeyError, ValueError, TypeError):
                values[name] = {metric: math.nan for metric, _ in METRICS}
        try:
            values["icsr_v3"] = {metric: float(v3[(dataset, seed)][metric]) for metric, _ in METRICS}
        except (KeyError, ValueError, TypeError):
            values["icsr_v3"] = {metric: math.nan for metric, _ in METRICS}
        complete[(dataset, seed)] = values
        for metric, label in METRICS:
            c, nr, rf, iv = (values[name][metric] for name in ("crsa", "rcfa_no_rse", "rcfa_full", "icsr_v3"))
            output.append({
                "dataset": dataset, "seed": seed, "row_type": "seed", "metric": label,
                "crsa": c, "rcfa_no_rse": nr, "rcfa_full": rf, "icsr_v3": iv,
                "rcfa_no_rse_minus_crsa": nr - c, "rcfa_full_minus_crsa": rf - c,
                "icsr_minus_crsa": iv - c, "icsr_minus_rcfa_no_rse": iv - nr,
                "icsr_minus_rcfa_full": iv - rf,
            })
    for dataset in runner.DATASETS:
        for metric, label in METRICS:
            by_arm = {}
            for arm in ("crsa", "rcfa_no_rse", "rcfa_full", "icsr_v3"):
                vals = [complete[(dataset, seed)][arm][metric] for seed in runner.SEEDS]
                by_arm[arm] = statistics.mean(vals) if all(math.isfinite(v) for v in vals) else math.nan
            c, nr, rf, iv = (by_arm[name] for name in ("crsa", "rcfa_no_rse", "rcfa_full", "icsr_v3"))
            output.append({
                "dataset": dataset, "seed": "", "row_type": "dataset_mean", "metric": label,
                "crsa": c, "rcfa_no_rse": nr, "rcfa_full": rf, "icsr_v3": iv,
                "rcfa_no_rse_minus_crsa": nr - c, "rcfa_full_minus_crsa": rf - c,
                "icsr_minus_crsa": iv - c, "icsr_minus_rcfa_no_rse": iv - nr,
                "icsr_minus_rcfa_full": iv - rf,
            })
    return output


def paired_rows(four: list[dict], baseline_key: str, delta_key: str) -> list[dict]:
    rows = []
    for item in four:
        rows.append({
            "dataset": item["dataset"], "seed": item["seed"], "row_type": item["row_type"],
            "metric": item["metric"], "icsr_v3": item["icsr_v3"],
            "baseline": item[baseline_key], "icsr_minus_baseline": item[delta_key],
        })
    return rows


def fmt_metric(x, std=None):
    if x == "" or not math.isfinite(float(x)):
        return "NA"
    if std in (None, ""):
        return f"{100 * float(x):.3f}%"
    return f"{100 * float(x):.3f} ± {100 * float(std):.3f}%"


def fmt_delta(x):
    if x == "" or not math.isfinite(float(x)):
        return "NA"
    return f"{100 * float(x):+.3f} pp"


def build_report(validation: list[dict], summary: list[dict], four: list[dict],
                 history_audit: dict, formal_audit: dict) -> str:
    by_mean = {(r["dataset"], r["metric"]): r for r in four if r["row_type"] == "dataset_mean"}
    summary_by = {r["dataset"]: r for r in summary}
    lines = [
        "# CRSA + ICSR v3 Validation Report", "",
        "## 1. Research question", "",
        "Does an intrinsic-context interaction residual improve on the frozen CRSA carrier, and how does this realization compare with the prior RCFA utility-gating results? This report uses validation metrics only.", "",
        "## 2. Provenance and protocol", "",
        f"- Formal runs: **{formal_audit['complete_validation_rows']} / 15 complete**.",
        f"- Formal source commit(s): `{', '.join(formal_audit['formal_source_commits']) or 'none'}`; uniform: `{formal_audit['formal_source_commit_uniform']}`.",
        "- NC full graph; AdamW, lr 1e-3, weight decay 1e-4; epochs 300, patience 30; min epoch 30, min delta 1e-4; clip 1.0; evaluate each epoch; no scheduler; full inference; K=3.",
        "- Best checkpoint selected by validation accuracy; Test evaluation disabled; LP not run.", "",
        "## 3. Implementation-control audit", "",
        "See `preflight_report.md`. Frozen CRSA parity, identity initialization, descriptor formulas, modality isolation, parameter delta, finite backward, and strict reload were checked on a synthetic graph.",
        "ICSR uses same-modality H0/Ck only; it has no RSE input, other-modality input, or utility gate.", "",
        "## 4. Full-v3 validation results", "",
        "| Dataset | Accuracy mean ± std | Macro-F1 mean ± std | n |", "|---|---:|---:|---:|",
    ]
    for d in runner.DATASETS:
        row = summary_by[d]
        lines.append(f"| {d} | {fmt_metric(row['mean_val_acc'], row['std_val_acc'])} | {fmt_metric(row['mean_val_macro_f1'], row['std_val_macro_f1'])} | {row['n']} |")
    lines += ["", "## 5. Four-way comparison and paired values", ""]
    for metric, label in METRICS:
        lines += [f"### {label}", "",
                  "| Dataset | CRSA | RCFA-noRSE | RCFA-Full | ICSR-v3 | ICSR−CRSA | ICSR−RCFA-Full |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for d in runner.DATASETS:
            row = by_mean[(d, label)]
            lines.append(f"| {d} | {fmt_metric(row['crsa'])} | {fmt_metric(row['rcfa_no_rse'])} | {fmt_metric(row['rcfa_full'])} | {fmt_metric(row['icsr_v3'])} | {fmt_delta(row['icsr_minus_crsa'])} | {fmt_delta(row['icsr_minus_rcfa_full'])} |")
        lines.append("")
    lines += ["Seed-level paired values (scores are %, deltas are percentage points):", "",
              "| Dataset | Seed | Metric | CRSA | RCFA-noRSE | RCFA-Full | ICSR-v3 | ICSR−CRSA | ICSR−RCFA-Full |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in four:
        if row["row_type"] != "seed":
            continue
        vals = [row[k] for k in ("crsa", "rcfa_no_rse", "rcfa_full", "icsr_v3")]
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['metric']} | " + " | ".join(fmt_metric(v) for v in vals) + f" | {fmt_delta(row['icsr_minus_crsa'])} | {fmt_delta(row['icsr_minus_rcfa_full'])} |")
    lines += ["", "## 6. Research interpretation", ""]
    for metric, label in METRICS:
        deltas = {d: by_mean[(d, label)] for d in runner.DATASETS}
        negative = [d for d in runner.DATASETS if float(deltas[d]["icsr_minus_crsa"]) < 0]
        positive = [d for d in runner.DATASETS if float(deltas[d]["icsr_minus_crsa"]) > 0]
        lines.append(f"For {label}, ICSR-v3 is above CRSA on {', '.join(positive) or 'no dataset'} and below CRSA on {', '.join(negative) or 'no dataset'} by dataset mean.")
    lines += ["", "### Q1. How does ICSR-v3 compare with CRSA?", "",
              "Read the per-dataset means and paired seeds above. Dataset-specific changes, rather than a single across-dataset average, are the unit of interpretation.", "",
              "### Q2. Does ICSR reduce RCFA negative transfer on Movies, Toys, and Grocery?", ""]
    for d in ("Movies", "Toys", "Grocery"):
        a = by_mean[(d, "Accuracy")]
        f = by_mean[(d, "Macro-F1")]
        less_negative = float(a["icsr_minus_crsa"]) > float(a["rcfa_full_minus_crsa"])
        lines.append(f"- {d}: ICSR−CRSA Accuracy {fmt_delta(a['icsr_minus_crsa'])} and Macro-F1 {fmt_delta(f['icsr_minus_crsa'])}; RCFA-Full−CRSA Accuracy {fmt_delta(a['rcfa_full_minus_crsa'])}. The ICSR Accuracy change is {'less negative / more positive' if less_negative else 'not less negative'} than RCFA-Full's.")
    lines += ["", "### Q3. Are the ele-fashion and Reddit-S positive trends retained?", ""]
    for d in ("ele-fashion", "Reddit-S"):
        a = by_mean[(d, "Accuracy")]
        lines.append(f"- {d}: ICSR−CRSA Accuracy {fmt_delta(a['icsr_minus_crsa'])}; ICSR−RCFA-Full {fmt_delta(a['icsr_minus_rcfa_full'])}.")
    lines += ["", "### Q4. Is interaction representation healthier than utility gating?", "",
              "The answer is dataset- and metric-specific. Compare the frequency and magnitude of negative ICSR−CRSA changes with RCFA-Full−CRSA, and inspect Macro-F1 alongside Accuracy. This is a descriptive comparison across five datasets and three seeds, without an automatic threshold.", "",
              "### Q5. Should Agreement/Deviation ablations follow?", "",
              "This decision should be based on whether ICSR-v3 reduces systematic negative transfer while keeping seed behavior and the positive-dataset trends coherent. The current report does not launch or pre-commit to those ablations.", "",
              "## 7. Engineering and isolation audit", "",
              "- Test metrics accessed: **NO**.", "- Test labels used/indexed for analysis: **NO**.", "- LP run: **NO**.", "- RSE used inside ICSR: **NO**.", "- Other modality used inside ICSR: **NO**.", "- Stage-II utility gate: **NO**.", "- Four-way results are paired by the same dataset and seed.", "",
              "Historical provenance details are in `provenance_audit.json`; formal run details are in `run_status.csv`.", ""]
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    hist, hist_audit = load_historical()
    invalid_history = [k for k, value in hist_audit.items() if value.get("status") != "valid"]
    validation, status_rows, formal_audit = run_rows()
    provenance = {"historical_sources": hist_audit, "formal": formal_audit,
                  "historical_provenance_valid": not invalid_history,
                  "test_metrics_accessed": False, "test_labels_used": False, "lp_run": False}
    (RESULTS / "provenance_audit.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    write_csv(RESULTS / "run_status.csv",
              ["mode", "variant", "dataset", "seed", "status", "source_commit", "test_evaluated", "run_dir", "checkpoint", "reason"],
              status_rows)
    validation_fields = ["dataset", "seed", "best_epoch", "val_acc", "val_macro_f1", "parameter_count",
                         "runtime_seconds", "peak_gpu_memory_mib", "source_commit", "test_evaluated", "status", "reason",
                         "run_dir", "checkpoint"]
    write_csv(RESULTS / "validation_results.csv", validation_fields, validation)
    summary = summary_rows(validation)
    write_csv(RESULTS / "validation_summary.csv",
              ["dataset", "n", "mean_val_acc", "std_val_acc", "mean_val_macro_f1", "std_val_macro_f1"], summary)
    if invalid_history:
        raise SystemExit("historical provenance validation failed; comparisons were not generated: " + ", ".join(invalid_history))
    if not formal_audit["formal_source_commit_uniform"]:
        raise SystemExit("formal runs do not share one clean source commit; paired analysis withheld")
    if any(r["status"] != "complete" for r in validation):
        raise SystemExit("formal matrix incomplete/invalid; paired analysis withheld")
    v3 = {(r["dataset"], int(r["seed"])): r for r in validation}
    four = four_way_rows(hist, v3)
    four_fields = ["dataset", "seed", "row_type", "metric", "crsa", "rcfa_no_rse", "rcfa_full", "icsr_v3",
                   "rcfa_no_rse_minus_crsa", "rcfa_full_minus_crsa", "icsr_minus_crsa",
                   "icsr_minus_rcfa_no_rse", "icsr_minus_rcfa_full"]
    write_csv(RESULTS / "four_way_comparison.csv", four_fields, four)
    pcrsa = paired_rows(four, "crsa", "icsr_minus_crsa")
    pfull = paired_rows(four, "rcfa_full", "icsr_minus_rcfa_full")
    paired_fields = ["dataset", "seed", "row_type", "metric", "icsr_v3", "baseline", "icsr_minus_baseline"]
    write_csv(RESULTS / "paired_vs_crsa.csv", paired_fields, pcrsa)
    write_csv(RESULTS / "paired_vs_rcfa_full.csv", paired_fields, pfull)
    (ROOT / "docs/crsa_icsr_v3_report.md").write_text(
        build_report(validation, summary, four, hist_audit, formal_audit), encoding="utf-8")
    print(f"validation rows: {len(validation)}; complete: {sum(r['status'] == 'complete' for r in validation)}")
    print(f"formal source commit uniform: {formal_audit['formal_source_commit_uniform']}")
    print(f"report: {ROOT / 'docs/crsa_icsr_v3_report.md'}")


if __name__ == "__main__":
    main()
