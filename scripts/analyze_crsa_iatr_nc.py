from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from scripts.run_crsa_iatr_nc import (
    DATASETS,
    FLAGS,
    OUTPUT,
    SEEDS,
    VARIANTS,
    formal_contexts,
)
from src.models.crsa_iatr import Model


RESULTS = ROOT / "results/crsa_iatr_v1"
SUMMARY_FIELDS = [
    "variant", "dataset", "n", "mean_val_acc", "std_val_acc",
    "mean_val_macro_f1", "std_val_macro_f1",
]
RUN_FIELDS = [
    "variant", "dataset", "seed", "status", "best_epoch", "val_acc",
    "val_macro_f1", "parameter_count", "runtime_seconds",
    "peak_gpu_memory_mib", "test_evaluated", "source_commit", "run_dir",
    "checkpoint", "reason",
]
DELTA_FIELDS = [
    "contrast", "dataset", "seed", "metric", "delta", "left_value", "right_value",
]


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


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


def contains_test_key(value) -> bool:
    if isinstance(value, dict):
        return any(
            "test" in str(key).lower() or contains_test_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_test_key(child) for child in value)
    return False


def _manifest_contexts(mode: str, manifest_rows: list[dict[str, str]]) -> list[dict]:
    if mode == "formal":
        return [
            {
                "variant": context.variant,
                "dataset": context.dataset,
                "seed": context.seed,
                "run_dir": str(context.run_dir),
                "checkpoint": str(context.checkpoint),
            }
            for context in formal_contexts("cuda:1")
        ]
    contexts = {}
    for row in manifest_rows:
        try:
            variant, dataset, seed_text = row["run_key"].split("/")[-3:]
            seed = int(seed_text.removeprefix("seed"))
        except (KeyError, ValueError):
            continue
        contexts[(variant, dataset, seed)] = {
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "run_dir": row.get("run_dir", ""),
            "checkpoint": row.get("checkpoint", ""),
        }
    return list(contexts.values())


def _metric_mean(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, dict) or "mean" not in value:
        raise ValueError(f"missing validation metric {key}")
    result = float(value["mean"])
    if not math.isfinite(result):
        raise ValueError(f"non-finite validation metric {key}")
    return result


def _parameter_count(config: dict, checkpoint: dict, cache: dict) -> int:
    cfg = OmegaConf.create(config)
    data_info = checkpoint.get("data_info")
    if not isinstance(data_info, dict):
        raise ValueError("checkpoint has no data_info")
    model = Model(cfg, data_info)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    head = torch.nn.Linear(model.out_dim, int(data_info["num_classes"]))
    head.load_state_dict(checkpoint["head_state"], strict=True)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    count += sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
    cache_key = (str(config.get("model", {}).get("variant")), data_info.get("text_dim"),
                 data_info.get("visual_dim"), data_info.get("num_classes"))
    cached = cache.setdefault(cache_key, count)
    if cached != count:
        raise ValueError("parameter count unexpectedly changed within a model context")
    return count


def inspect_context(context: dict, manifest: dict[str, dict], parameter_cache: dict) -> dict:
    variant, dataset, seed = context["variant"], context["dataset"], int(context["seed"])
    key = f"{variant}/{dataset}/seed{seed}"
    manifest_row = manifest.get(key, {})
    run_dir = Path(context.get("run_dir") or manifest_row.get("run_dir") or "")
    checkpoint_path = Path(context.get("checkpoint") or manifest_row.get("checkpoint") or "")
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    config_path = run_dir / "resolved_config.json"
    metrics_path = run_dir / "metrics.json"
    marker_path = run_dir / "complete.marker"
    row = {
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "status": "missing",
        "best_epoch": "",
        "val_acc": "",
        "val_macro_f1": "",
        "parameter_count": "",
        "runtime_seconds": manifest_row.get("runtime_seconds", ""),
        "peak_gpu_memory_mib": manifest_row.get("peak_gpu_memory_mib", ""),
        "test_evaluated": "false",
        "source_commit": manifest_row.get("git_commit", ""),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "reason": "",
    }
    if not all(path.is_file() for path in (config_path, metrics_path, marker_path, checkpoint_path)):
        row["status"] = manifest_row.get("status", "missing") or "missing"
        row["reason"] = "required resolved config, metrics, marker, or checkpoint is absent"
        return row

    config = _read_json(config_path)
    metrics_payload = _read_json(metrics_path)
    marker = _read_json(marker_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError) as exc:
        row.update(status="invalid", reason=f"checkpoint load failed: {exc}")
        return row
    if not all(isinstance(item, dict) for item in (config, metrics_payload, marker, checkpoint)):
        row.update(status="invalid", reason="one or more run artifacts are malformed")
        return row

    task_config = config.get("task", {})
    model_config = config.get("model", {})
    metrics = metrics_payload.get("metrics", {})
    checkpoint_metrics = checkpoint.get("metrics", {})
    try:
        if task_config.get("evaluate_test") is not False:
            raise ValueError("resolved config did not disable Test evaluation")
        if task_config.get("eval_modality_masks", {}).get("enabled") is not False:
            raise ValueError("modality-mask evaluation was enabled")
        if contains_test_key(metrics) or contains_test_key(checkpoint_metrics):
            raise ValueError("Test metric key found in validation artifacts")
        if model_config.get("name") != "crsa_iatr" or model_config.get("variant") != variant:
            raise ValueError("resolved model variant does not match the run context")
        if (model_config.get("use_crsa"), model_config.get("use_iatr")) != FLAGS[variant]:
            raise ValueError("resolved ablation flags do not match the run context")
        if config.get("dataset", {}).get("name") != dataset or int(config.get("seed", -1)) != seed:
            raise ValueError("resolved dataset or seed does not match the run context")
        if metrics_payload.get("task") != "nc" or metrics_payload.get("dataset") != dataset:
            raise ValueError("metrics metadata does not match the run context")
        if metrics_payload.get("seed") != seed or metrics_payload.get("model") != "crsa_iatr":
            raise ValueError("metrics model or seed does not match the run context")
        if marker.get("status") != "complete" or marker.get("task") != "nc":
            raise ValueError("completion marker is invalid")
        if checkpoint.get("task") != "nc" or checkpoint.get("seed") != seed:
            raise ValueError("checkpoint task or seed does not match the run context")
        if checkpoint.get("selection") != "best_val_accuracy":
            raise ValueError("checkpoint was not selected by validation accuracy")
        if checkpoint.get("epoch") != metrics_payload.get("best_epoch"):
            raise ValueError("checkpoint epoch and best epoch disagree")
        val_acc = _metric_mean(metrics, "val_acc")
        val_f1 = _metric_mean(metrics, "val_macro_f1")
        if not 0.0 <= val_acc <= 1.0 or not 0.0 <= val_f1 <= 1.0:
            raise ValueError("validation metric falls outside [0,1]")
        parameter_count = _parameter_count(config, checkpoint, parameter_cache)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        row.update(status="invalid", reason=str(exc))
        return row

    row.update(
        status="complete",
        best_epoch=metrics_payload.get("best_epoch", ""),
        val_acc=val_acc,
        val_macro_f1=val_f1,
        parameter_count=parameter_count,
        runtime_seconds=metrics_payload.get("runtime_seconds", row["runtime_seconds"]),
        peak_gpu_memory_mib=metrics_payload.get(
            "peak_gpu_memory_mib", row["peak_gpu_memory_mib"]
        ),
        source_commit=manifest_row.get("git_commit", ""),
        reason="",
    )
    return row


def _summary_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["status"] == "complete":
            grouped[(row["variant"], row["dataset"])].append(row)
            grouped[(row["variant"], "ALL_DATASETS")].append(row)
    output = []
    for (variant, dataset), group in sorted(grouped.items()):
        acc = [float(row["val_acc"]) for row in group]
        f1 = [float(row["val_macro_f1"]) for row in group]
        output.append({
            "variant": variant,
            "dataset": dataset,
            "n": len(group),
            "mean_val_acc": statistics.mean(acc),
            "std_val_acc": statistics.stdev(acc) if len(acc) > 1 else "",
            "mean_val_macro_f1": statistics.mean(f1),
            "std_val_macro_f1": statistics.stdev(f1) if len(f1) > 1 else "",
        })
    return output


def _paired_rows(rows: list[dict]) -> list[dict]:
    keyed = {
        (row["variant"], row["dataset"], int(row["seed"])): row
        for row in rows if row["status"] == "complete"
    }
    contrasts = [
        ("Full-Base", "full", "base"),
        ("CRSA-Base", "crsa", "base"),
        ("IATR-Base", "iatr", "base"),
        ("Full-CRSA", "full", "crsa"),
        ("Full-IATR", "full", "iatr"),
        ("Interaction-Full-CRSA-IATR+Base", "full", "crsa"),
    ]
    output = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for contrast, left, right in contrasts:
                a = keyed.get((left, dataset, seed))
                b = keyed.get((right, dataset, seed))
                if a is None or b is None:
                    continue
                for metric in ("val_acc", "val_macro_f1"):
                    left_value = float(a[metric])
                    right_value = float(b[metric])
                    if contrast.startswith("Interaction-"):
                        base = keyed.get(("base", dataset, seed))
                        if base is None:
                            continue
                        delta = left_value - right_value - float(
                            keyed[("iatr", dataset, seed)][metric]
                        ) + float(base[metric])
                    else:
                        delta = left_value - right_value
                    output.append({
                        "contrast": contrast,
                        "dataset": dataset,
                        "seed": seed,
                        "metric": metric,
                        "delta": delta,
                        "left_value": left_value,
                        "right_value": right_value,
                    })
    return output


def _fmt(value) -> str:
    if value == "" or value is None:
        return "NA"
    return f"{float(value):.4f}"


def _write_report(path: Path, rows: list[dict], summary: list[dict], mode: str) -> None:
    completed = [row for row in rows if row["status"] == "complete"]
    incomplete = [row for row in rows if row["status"] != "complete"]
    lookup = {(row["variant"], row["dataset"], int(row["seed"])): row for row in completed}
    summary_map = {(row["variant"], row["dataset"]): row for row in summary}
    lines = [
        "# CRSA+IATR NC validation report",
        "",
        f"Mode: {mode}. This report uses validation accuracy and validation Macro-F1 only.",
        "Test evaluation is disabled; no Test metric is reported or used.",
        "",
        f"Completed contexts: {len(completed)} / {len(rows)}.",
        "",
        "## Q1. Full relative to Base",
        "",
        "The values below are descriptive paired-run results. They do not impose a pass/fail threshold.",
        "",
        "| Dataset | Base Val Acc (mean ± SD) | Full Val Acc (mean ± SD) | Full−Base by seed | Base Macro-F1 (mean ± SD) | Full Macro-F1 (mean ± SD) |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for dataset in DATASETS:
        b = summary_map.get(("base", dataset), {})
        f = summary_map.get(("full", dataset), {})
        deltas = []
        for seed in SEEDS:
            br = lookup.get(("base", dataset, seed))
            fr = lookup.get(("full", dataset, seed))
            if br and fr:
                deltas.append(f"{seed}: {_fmt(float(fr['val_acc']) - float(br['val_acc']))}")
        lines.append(
            f"| {dataset} | {_fmt(b.get('mean_val_acc'))} ± {_fmt(b.get('std_val_acc'))} "
            f"| {_fmt(f.get('mean_val_acc'))} ± {_fmt(f.get('std_val_acc'))} "
            f"| {'; '.join(deltas) or 'incomplete'} "
            f"| {_fmt(b.get('mean_val_macro_f1'))} ± {_fmt(b.get('std_val_macro_f1'))} "
            f"| {_fmt(f.get('mean_val_macro_f1'))} ± {_fmt(f.get('std_val_macro_f1'))} |"
        )
    lines.extend(["", "Per-seed validation values:", ""])
    lines.extend(_per_seed_table(lookup))

    for title, variant, question in [
        ("Q2. CRSA-only", "crsa", "This isolates relation interpretation with uniform state readout."),
        ("Q3. IATR-only", "iatr", "This isolates intrinsic-anchor trajectory reconciliation after plain propagation."),
    ]:
        lines.extend(["", f"## {title}", "", question, "", "| Dataset | Val Acc (mean ± SD) | Macro-F1 (mean ± SD) | Seeds present |", "|---|---:|---:|---|"])
        for dataset in DATASETS:
            item = summary_map.get((variant, dataset), {})
            seeds_present = ", ".join(
                str(seed) for seed in SEEDS if (variant, dataset, seed) in lookup
            )
            lines.append(
                f"| {dataset} | {_fmt(item.get('mean_val_acc'))} ± {_fmt(item.get('std_val_acc'))} "
                f"| {_fmt(item.get('mean_val_macro_f1'))} ± {_fmt(item.get('std_val_macro_f1'))} "
                f"| {seeds_present or 'none'} |"
            )

    lines.extend([
        "", "## Q4. Full relative to the single-module variants", "",
        "Paired differences are descriptive within the same dataset and seed.", "",
        "| Dataset | Full−CRSA Val Acc by seed | Full−IATR Val Acc by seed | Full−CRSA Macro-F1 by seed | Full−IATR Macro-F1 by seed |",
        "|---|---|---|---|---|",
    ])
    for dataset in DATASETS:
        cells = []
        for metric in ("val_acc", "val_macro_f1"):
            for single in ("crsa", "iatr"):
                values = []
                for seed in SEEDS:
                    full = lookup.get(("full", dataset, seed))
                    module = lookup.get((single, dataset, seed))
                    if full and module:
                        values.append(f"{seed}: {_fmt(float(full[metric])-float(module[metric]))}")
                cells.append("; ".join(values) or "incomplete")
        lines.append(f"| {dataset} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    lines.extend([
        "", "## Q5. Engineering pathology", "",
        f"Run status: {len(completed)} complete, {sum(row['status'] == 'failed' for row in rows)} failed, "
        f"{sum(row['status'] == 'invalid' for row in rows)} invalid, "
        f"{sum(row['status'] == 'missing' for row in rows)} missing, "
        f"{sum(row['status'] not in {'complete', 'failed', 'invalid', 'missing'} for row in rows)} other.",
    ])
    memory_values = [float(row["peak_gpu_memory_mib"]) for row in completed
                     if row.get("peak_gpu_memory_mib") not in (None, "")]
    runtime_values = [float(row["runtime_seconds"]) for row in completed
                      if row.get("runtime_seconds") not in (None, "")]
    if memory_values:
        lines.append(f"Peak allocated GPU memory: max {max(memory_values):.1f} MiB across completed contexts.")
    if runtime_values:
        lines.append(f"Run time: median {statistics.median(runtime_values):.1f} seconds per completed context.")
    lines.append("No route-collapse diagnostic is part of this experiment; this report does not infer route health from task scores.")
    if incomplete:
        lines.extend(["", "Incomplete or invalid contexts:", ""])
        for row in incomplete:
            lines.append(
                f"- {row['variant']} / {row['dataset']} / seed {row['seed']}: "
                f"{row['status']} — {row['reason']}"
            )
    else:
        lines.append("All expected contexts have valid validation-only artifacts.")
    lines.extend([
        "", "## Protocol and interpretation", "",
        "All reported statistics are descriptive. The across-dataset rows pool context-level observations and are not a substitute for per-dataset results.",
        "No significance threshold or automatic acceptance rule was applied.",
        "No link prediction, robustness, modality-missing, or Test evaluation was run.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _per_seed_table(lookup: dict) -> list[str]:
    lines = [
        "| Dataset | Seed | Base Acc / F1 | CRSA Acc / F1 | IATR Acc / F1 | Full Acc / F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for seed in SEEDS:
            values = []
            for variant in VARIANTS:
                row = lookup.get((variant, dataset, seed))
                values.append(
                    f"{_fmt(row['val_acc'])} / {_fmt(row['val_macro_f1'])}" if row else "NA / NA"
                )
            lines.append(f"| {dataset} | {seed} | " + " | ".join(values) + " |")
    return lines


def analyze(mode: str) -> Path:
    manifest_name = "formal_run_manifest.csv" if mode == "formal" else "smoke_run_manifest.csv"
    manifest_rows = _read_csv(OUTPUT / manifest_name)
    manifest = {row.get("run_key", ""): row for row in manifest_rows}
    contexts = _manifest_contexts(mode, manifest_rows)
    if not contexts:
        raise RuntimeError(f"no {mode} contexts found in {OUTPUT / manifest_name}")
    parameter_cache = {}
    rows = [inspect_context(context, manifest, parameter_cache) for context in contexts]
    out_dir = RESULTS if mode == "formal" else RESULTS / "smoke"
    _write_csv(out_dir / "validation_results.csv", RUN_FIELDS, rows)
    summary = _summary_rows(rows)
    _write_csv(out_dir / "validation_summary.csv", SUMMARY_FIELDS, summary)
    _write_csv(out_dir / "paired_deltas.csv", DELTA_FIELDS, _paired_rows(rows))
    _write_csv(out_dir / "run_status.csv", RUN_FIELDS, rows)
    report_path = (
        ROOT / "docs/crsa_iatr_v1_nc_report.md" if mode == "formal"
        else out_dir / "smoke_report.md"
    )
    _write_report(report_path, rows, summary, mode)
    print(f"Validated {sum(row['status'] == 'complete' for row in rows)} / {len(rows)} contexts")
    print(f"Results: {out_dir}")
    print(f"Report: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze validation-only CRSA+IATR NC runs")
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    args = parser.parse_args()
    analyze(args.mode)


if __name__ == "__main__":
    main()
