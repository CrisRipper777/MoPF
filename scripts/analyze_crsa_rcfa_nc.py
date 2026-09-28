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

import torch
from omegaconf import OmegaConf

from scripts import run_crsa_rcfa_nc as runner

RESULTS = ROOT / "results/crsa_rcfa_v2"
HISTORICAL = ROOT / "results/crsa_iatr_v1/validation_results.csv"
INITIAL_SOURCE = "e26d20ebf2d957bba790e9cbdb8b0c374cbe60b3"
HISTORICAL_SOURCE = "a529f896a85cc25c17a9a6729f3d691a663abab5"
VALIDATION_FIELDS = [
    "dataset", "seed", "best_epoch", "val_acc", "val_macro_f1",
    "parameter_count", "runtime_seconds", "peak_gpu_memory_mib",
    "source_commit", "test_evaluated", "status", "reason", "run_dir", "checkpoint",
]
SUMMARY_FIELDS = [
    "dataset", "n", "mean_val_acc", "std_val_acc",
    "mean_val_macro_f1", "std_val_macro_f1",
]
PAIRED_FIELDS = [
    "dataset", "seed", "metric", "crsa_v1_value", "full_v2_value", "delta",
]
STATUS_FIELDS = [
    "mode", "variant", "dataset", "seed", "status", "source_commit",
    "test_evaluated", "run_dir", "checkpoint", "reason",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def contains_test_key(value) -> bool:
    if isinstance(value, dict):
        return any("test" in str(key).lower() or contains_test_key(child)
                   for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_test_key(child) for child in value)
    return False


def _parameter_count(config: dict, checkpoint: dict) -> int:
    from src.models.crsa_rcfa import Model

    data_info = checkpoint.get("data_info")
    if not isinstance(data_info, dict):
        raise ValueError("checkpoint has no data_info")
    model = Model(OmegaConf.create(config), data_info)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head = torch.nn.Linear(model.out_dim, int(data_info["num_classes"]))
    head.load_state_dict(checkpoint["head_state"], strict=True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(
        p.numel() for p in head.parameters() if p.requires_grad
    )


def _training_log_has_finite_losses(run_dir: Path) -> bool:
    path = run_dir / "train.log"
    if not path.is_file():
        path = run_dir / "main.log"
    if not path.is_file():
        return False
    losses = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Train Loss" not in line:
            continue
        match = re.search(r"Train Loss\s+([^\s]+)", line)
        if match is None:
            return False
        try:
            value = float(match.group(1))
        except ValueError:
            return False
        if not math.isfinite(value):
            return False
        losses.append(value)
    return bool(losses)


def inspect_context(context, manifest_rows: dict[str, dict]) -> dict:
    key = context.run_key
    manifest = manifest_rows.get(key, {})
    row = {
        "mode": context.mode, "variant": context.variant,
        "dataset": context.dataset, "seed": context.seed,
        "status": "missing", "best_epoch": "", "val_acc": "",
        "val_macro_f1": "", "parameter_count": "",
        "runtime_seconds": manifest.get("runtime_seconds", ""),
        "peak_gpu_memory_mib": manifest.get("peak_gpu_memory_mib", ""),
        "source_commit": manifest.get("git_commit", ""),
        "test_evaluated": "false", "reason": "",
        "run_dir": str(context.run_dir), "checkpoint": str(context.checkpoint),
    }
    if not manifest:
        row["reason"] = "formal context is absent from run manifest"
        return row
    if manifest.get("status") != "complete":
        row["status"] = manifest.get("status") or "missing"
        row["reason"] = manifest.get("error_reason", "run did not complete")
        return row
    valid, reason, metrics_payload = runner.audit_context(context)
    if not valid:
        row.update(status="invalid", reason=reason)
        return row
    if manifest.get("git_branch") != "crsa_rcfa_v2":
        row.update(status="invalid", reason="formal run branch is not crsa_rcfa_v2")
        return row
    if manifest.get("git_worktree_clean") != "true":
        row.update(status="invalid", reason="formal source worktree was not clean")
        return row
    if manifest.get("test_evaluated", "").lower() != "false":
        row.update(status="invalid", test_evaluated=manifest.get("test_evaluated"),
                   reason="manifest reports Test evaluation")
        return row
    config = _read_json(context.run_dir / "resolved_config.json")
    metrics_file = _read_json(context.run_dir / "metrics.json")
    marker = _read_json(context.run_dir / "complete.marker")
    try:
        checkpoint = torch.load(context.checkpoint, map_location="cpu", weights_only=False)
        if contains_test_key(metrics_file.get("metrics", {})):
            raise ValueError("Test metric found in metrics.json")
        if contains_test_key(checkpoint.get("metrics", {})):
            raise ValueError("Test metric found in checkpoint metrics")
        if metrics_file.get("best_epoch") != checkpoint.get("epoch"):
            raise ValueError("best epoch differs between metrics and checkpoint")
        count = _parameter_count(config, checkpoint)
        metrics = metrics_payload["metrics"]
        acc = float(metrics["val_acc"]["mean"])
        f1 = float(metrics["val_macro_f1"]["mean"])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (acc, f1)):
            raise ValueError("invalid validation score")
        if marker.get("status") != "complete":
            raise ValueError("completion marker invalid")
        if not _training_log_has_finite_losses(context.run_dir):
            raise ValueError("training log does not contain finite losses")
    except (OSError, RuntimeError, EOFError, ValueError, KeyError, TypeError) as exc:
        row.update(status="invalid", reason=str(exc))
        return row
    row.update(
        status="complete",
        best_epoch=metrics_file.get("best_epoch", ""),
        val_acc=acc,
        val_macro_f1=f1,
        parameter_count=count,
        runtime_seconds=metrics_file.get("runtime_seconds", row["runtime_seconds"]),
        peak_gpu_memory_mib=metrics_file.get(
            "peak_gpu_memory_mib", row["peak_gpu_memory_mib"]
        ),
        source_commit=manifest.get("git_commit", ""),
        reason="",
    )
    return row


def _summary_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            grouped[row["dataset"]].append(row)
    output = []
    for dataset in runner.DATASETS:
        group = grouped.get(dataset, [])
        if not group:
            continue
        acc = [float(row["val_acc"]) for row in group]
        f1 = [float(row["val_macro_f1"]) for row in group]
        output.append({
            "dataset": dataset,
            "n": len(group),
            "mean_val_acc": statistics.mean(acc),
            "std_val_acc": statistics.stdev(acc) if len(acc) > 1 else "",
            "mean_val_macro_f1": statistics.mean(f1),
            "std_val_macro_f1": statistics.stdev(f1) if len(f1) > 1 else "",
        })
    return output


def _preflight_parity_ok() -> tuple[bool, str]:
    path = RESULTS / "preflight_report.md"
    if not path.is_file():
        return False, "preflight_report.md is missing"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"legacy_crsa_parity_max_abs_error:\s*([0-9.eE+-]+)", text)
    if not match:
        return False, "legacy parity maximum error is absent from preflight report"
    error = float(match.group(1))
    if not math.isfinite(error) or error >= 1e-6:
        return False, f"legacy CRSA parity error {error} is not below 1e-6"
    return True, f"legacy CRSA parity error {error:.3e} passed"


def _historical_reference() -> tuple[dict, str]:
    expected = {(dataset, seed) for dataset in runner.DATASETS for seed in runner.SEEDS}
    rows = _read_csv(HISTORICAL)
    crsa = [row for row in rows if row.get("variant") == "crsa"]
    keys = []
    reasons = []
    for row in crsa:
        try:
            keys.append((row["dataset"], int(row["seed"])))
        except (KeyError, ValueError):
            reasons.append("historical dataset/seed is malformed")
    if len(crsa) != 15 or len(set(keys)) != 15 or set(keys) != expected:
        reasons.append(f"expected exactly 15 unique CRSA dataset/seed rows, found {len(crsa)}")
    ref = {}
    for row in crsa:
        try:
            key = (row["dataset"], int(row["seed"]))
            if row.get("status") != "complete":
                reasons.append(f"historical row {key} is not complete")
            if row.get("test_evaluated", "").lower() != "false":
                reasons.append(f"historical row {key} is not marked validation-only")
            if row.get("source_commit") != HISTORICAL_SOURCE:
                reasons.append(f"historical row {key} has unexpected source provenance")
            if any(field in row for field in ("test_acc", "test_macro_f1")):
                reasons.append(f"historical row {key} contains a Test metric")
            ref[key] = {
                "val_acc": float(row["val_acc"]),
                "val_macro_f1": float(row["val_macro_f1"]),
            }
        except (KeyError, TypeError, ValueError):
            reasons.append("historical CRSA validation metrics are malformed")
    if reasons:
        return {}, "; ".join(dict.fromkeys(reasons))
    return ref, "15 validation-only CRSA rows verified at the expected source commit"


def _paired_rows(rows: list[dict], reference: dict) -> tuple[list[dict], str]:
    complete = {(row["dataset"], int(row["seed"])): row for row in rows
                if row["status"] == "complete"}
    if len(complete) != 15:
        return [], "Full-v2 does not have 15 complete contexts"
    source_commits = {row.get("source_commit", "") for row in rows
                      if row["status"] == "complete"}
    if len(source_commits) != 1 or not next(iter(source_commits), ""):
        return [], "Full-v2 formal runs do not share one recorded source commit"
    parity_ok, parity_reason = _preflight_parity_ok()
    if not parity_ok:
        return [], parity_reason
    reference_rows, reference_reason = _historical_reference()
    if not reference_rows:
        return [], reference_reason
    paired = []
    for dataset in runner.DATASETS:
        for seed in runner.SEEDS:
            key = (dataset, seed)
            old = reference_rows[key]
            new = complete[key]
            for metric in ("val_acc", "val_macro_f1"):
                paired.append({
                    "dataset": dataset, "seed": seed, "metric": metric,
                    "crsa_v1_value": old[metric],
                    "full_v2_value": float(new[metric]),
                    "delta": float(new[metric]) - old[metric],
                })
    for dataset in runner.DATASETS:
        for metric in ("val_acc", "val_macro_f1"):
            group = [row for row in paired
                     if row["dataset"] == dataset and row["metric"] == metric]
            paired.append({
                "dataset": dataset, "seed": "MEAN", "metric": metric,
                "crsa_v1_value": statistics.mean(float(row["crsa_v1_value"]) for row in group),
                "full_v2_value": statistics.mean(float(row["full_v2_value"]) for row in group),
                "delta": statistics.mean(float(row["delta"]) for row in group),
            })
    return paired, f"{parity_reason}; {reference_reason}"


def _fmt_mean_std(mean, std) -> str:
    if mean == "" or mean is None:
        return "n/a"
    return f"{100 * float(mean):.2f} ± {100 * float(std):.2f}" if std != "" else f"{100 * float(mean):.2f}"


def _format_report(rows: list[dict], summary: list[dict], paired: list[dict],
                   pairing_status: str, manifest: list[dict[str, str]]) -> str:
    complete = [row for row in rows if row["status"] == "complete"]
    source_commits = sorted({row["source_commit"] for row in complete if row["source_commit"]})
    runtime_by_ds, memory_by_ds = defaultdict(list), defaultdict(list)
    for row in complete:
        if row.get("runtime_seconds") not in ("", None):
            runtime_by_ds[row["dataset"]].append(float(row["runtime_seconds"]))
        if row.get("peak_gpu_memory_mib") not in ("", None):
            memory_by_ds[row["dataset"]].append(float(row["peak_gpu_memory_mib"]))
    summary_map = {row["dataset"]: row for row in summary}
    paired_map = {(row["dataset"], row["metric"]): row for row in paired if row["seed"] == "MEAN"}
    failures = [row for row in rows if row["status"] != "complete"]
    oom = []
    nonfinite = []
    for entry in manifest:
        run_dir = Path(entry.get("run_dir", ""))
        log_paths = [run_dir / "launcher_output.log", run_dir / "main.log"]
        text = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                         for path in log_paths if path.is_file())
        if re.search(r"out of memory|cuda oom", text, re.I):
            oom.append(entry.get("run_key", "unknown"))
        if re.search(r"\b(nan|inf)\b", text, re.I):
            nonfinite.append(entry.get("run_key", "unknown"))

    lines = [
        "# CRSA+RSE+RCFA v2 NC Validation Report",
        "",
        "## 1. Implementation provenance",
        "",
        f"- Source branch: crsa_iatr_v1 at initial snapshot {INITIAL_SOURCE}.",
        "- Experiment branch: crsa_rcfa_v2.",
        f"- Formal source commit: {source_commits[0] if len(source_commits) == 1 else 'not uniform or unavailable'}.",
        f"- Formal contexts complete: {len(complete)} / 15.",
        "- Every formal run is required to record a clean worktree and the same source commit.",
        "",
        "## 2. Preflight correctness audit",
        "",
    ]
    preflight_path = RESULTS / "preflight_report.md"
    if preflight_path.is_file():
        lines.extend(preflight_path.read_text(encoding="utf-8").strip().splitlines())
    else:
        lines.append("Preflight report missing.")
    lines.extend([
        "",
        "## 3. Full-v2 validation results",
        "",
        "| Dataset | n | Validation Accuracy (%) | Validation Macro-F1 (%) |",
        "|---|---:|---:|---:|",
    ])
    for dataset in runner.DATASETS:
        row = summary_map.get(dataset)
        if row:
            lines.append(
                f"| {dataset} | {row['n']} | {_fmt_mean_std(row['mean_val_acc'], row['std_val_acc'])} "
                f"| {_fmt_mean_std(row['mean_val_macro_f1'], row['std_val_macro_f1'])} |"
            )
        else:
            lines.append(f"| {dataset} | 0 | n/a | n/a |")

    lines.extend([
        "",
        "## 4. Paired comparison with CRSA-v1",
        "",
        f"Pairing audit: {pairing_status}. Values are descriptive validation differences; no significance or pass/fail label is assigned.",
        "",
        "| Dataset | Metric | Seed 42 | Seed 43 | Seed 44 | Mean delta |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for dataset in runner.DATASETS:
        for metric, label in (("val_acc", "Accuracy (pp)"), ("val_macro_f1", "Macro-F1 (pp)")):
            values = {
                int(row["seed"]): float(row["delta"])
                for row in paired if row["dataset"] == dataset and row["metric"] == metric
                and row["seed"] != "MEAN"
            }
            mean = paired_map.get((dataset, metric), {}).get("delta")
            if values and mean is not None:
                lines.append(
                    f"| {dataset} | {label} | {100*values.get(42, 0):+.2f} | "
                    f"{100*values.get(43, 0):+.2f} | {100*values.get(44, 0):+.2f} | {100*float(mean):+.2f} |"
                )
            else:
                lines.append(f"| {dataset} | {label} | n/a | n/a | n/a | n/a |")

    directions = defaultdict(list)
    for dataset in runner.DATASETS:
        item = paired_map.get((dataset, "val_acc"))
        if item is not None:
            directions["positive" if float(item["delta"]) > 0 else
                       "negative" if float(item["delta"]) < 0 else "flat"].append(dataset)
    lines.extend([
        "",
        "## 5. Dataset-wise observations",
        "",
        "- Validation-accuracy paired mean direction: "
        + "; ".join(f"{direction}: {', '.join(items)}" for direction, items in directions.items())
        + ("." if directions else "not available."),
        "- Interpret dataset variation descriptively; three seeds do not support a significance claim.",
        "",
        "## 6. Runtime and GPU memory",
        "",
        "| Dataset | Mean runtime (s) | Peak GPU memory (MiB) |",
        "|---|---:|---:|",
    ])
    for dataset in runner.DATASETS:
        runtimes, memories = runtime_by_ds[dataset], memory_by_ds[dataset]
        lines.append(
            f"| {dataset} | {statistics.mean(runtimes):.1f} | {max(memories):.1f} |"
            if runtimes and memories else f"| {dataset} | n/a | n/a |"
        )
    lines.extend([
        "",
        "## 7. Engineering pathology audit",
        "",
        f"- Complete contexts: {len(complete)} / 15; non-complete or invalid contexts: {len(failures)}.",
        f"- OOM signatures in run logs: {len(oom)}" + (f" ({', '.join(oom)})" if oom else "") + ".",
        f"- NaN/Inf signatures in run logs: {len(nonfinite)}" + (f" ({', '.join(nonfinite)})" if nonfinite else "") + ".",
        "- Artifact audit checks finite losses, finite checkpoint tensors, validation-only metric keys, and strict model/head checkpoint loading.",
        "",
        "## 8. Interpretation",
        "",
    ])
    if paired_map:
        vals = [float(paired_map[(dataset, "val_acc")]["delta"]) for dataset in runner.DATASETS]
        positive = sum(value > 0 for value in vals)
        negative = sum(value < 0 for value in vals)
        trend = "positive" if positive == len(vals) else "negative" if negative == len(vals) else "mixed"
        lines.append(
            f"Full-v2 has a {trend} paired mean validation-accuracy pattern: positive on {positive} datasets "
            f"and negative on {negative} datasets. This is descriptive, not evidence of statistical significance."
        )
        lines.append(
            "A formal w/o-relation-condition ablation is "
            + ("worth considering next because Full-v2 shows a consistent positive dataset-level direction."
               if positive == len(runner.DATASETS) else
               "useful only if the team wants to isolate RSE conditioning after reviewing the mixed dataset pattern.")
        )
    else:
        lines.append("The paired comparison is unavailable; no research interpretation is made from incomplete or unverified pairs.")
    lines.extend([
        "",
        "Training uses Train labels and validation selection/reporting only. The resolved configuration sets evaluate_test=false and the analyzer rejects Test metric keys. No link prediction context is in the matrix.",
        "",
    ])
    return "\n".join(lines)


def analyze() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_rows_list = _read_csv(runner.OUTPUT / "formal_run_manifest.csv")
    manifest = {row.get("run_key", ""): row for row in manifest_rows_list}
    contexts = runner.formal_contexts("cuda:1")
    rows = [inspect_context(context, manifest) for context in contexts]
    expected_keys = {context.run_key for context in contexts}
    manifest_keys = [row.get("run_key", "") for row in manifest_rows_list]
    manifest_issues = []
    if len(manifest_rows_list) != 15 or set(manifest_keys) != expected_keys or len(set(manifest_keys)) != 15:
        manifest_issues.append("formal manifest does not contain exactly the 15 unique expected contexts")
    if any(row.get("git_branch") != "crsa_rcfa_v2" for row in manifest_rows_list):
        manifest_issues.append("formal manifest contains a run from another branch")
    if any(row.get("git_worktree_clean") != "true" for row in manifest_rows_list):
        manifest_issues.append("formal manifest contains a dirty-source run")
    recorded_commits = {row.get("git_commit", "") for row in manifest_rows_list}
    if len(recorded_commits) != 1 or not next(iter(recorded_commits), ""):
        manifest_issues.append("formal manifest does not record one common source commit")
    if manifest_issues:
        for row in rows:
            if row["status"] == "complete":
                row["status"] = "invalid"
                row["reason"] = "; ".join(manifest_issues)
    (RESULTS / "formal_matrix_audit.json").write_text(
        json.dumps({"status": "valid" if not manifest_issues else "invalid",
                    "manifest_rows": len(manifest_rows_list),
                    "unique_contexts": len(set(manifest_keys)),
                    "source_commits": sorted(commit for commit in recorded_commits if commit),
                    "issues": manifest_issues}, indent=2),
        encoding="utf-8",
    )
    _write_csv(RESULTS / "formal_run_manifest.csv", runner.MANIFEST_FIELDS, manifest_rows_list)
    _write_csv(RESULTS / "validation_results.csv", VALIDATION_FIELDS, rows)
    summary = _summary_rows(rows)
    _write_csv(RESULTS / "validation_summary.csv", SUMMARY_FIELDS, summary)
    paired, pairing_status = _paired_rows(rows, {})
    _write_csv(RESULTS / "paired_vs_crsa_v1.csv", PAIRED_FIELDS, paired)
    (RESULTS / "paired_comparison_status.json").write_text(
        json.dumps({"status": "available" if paired else "unavailable",
                    "reason": pairing_status}, indent=2),
        encoding="utf-8",
    )
    _write_csv(RESULTS / "run_status.csv", STATUS_FIELDS, rows)
    report = _format_report(rows, summary, paired, pairing_status, manifest_rows_list)
    report_path = ROOT / "docs/crsa_rcfa_v2_nc_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"results={RESULTS}")
    print(f"formal_complete={sum(row['status'] == 'complete' for row in rows)}/15")
    print(f"paired={len(paired)} rows; {pairing_status}")
    return {"rows": rows, "summary": summary, "paired": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and summarize CRSA+RSE+RCFA v2 NC runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("Would audit the fixed 15-context Full-v2 matrix and validation-only artifacts.")
        return
    result = analyze()
    if any(row["status"] != "complete" for row in result["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
