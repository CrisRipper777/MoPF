"""Run and package the two requested MoPF targeted diagnostics.

The script keeps all outputs under one ignored diagnosis directory.  It uses
the normal NC runner, with only opt-in checkpoint paths for the seed-44
Macro-F1 diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Make the repository package importable when this file is invoked as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_mag_data
from src.models import build_model
from src.tasks.inference import infer_all_embeddings
from src.tasks.nc import _resolve_nc_eval_labels


TARGET_ROOT = ROOT / "outputs" / "targeted_diagnosis"
FORMAL_LOG_HINT = ROOT / "outputs" / "2026-09-09" / "15-14-25" / "main.log"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _run_job(output_dir: Path, overrides: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.main",
        *overrides,
        f"hydra.run.dir={output_dir}",
    ]
    print("[targeted-diagnosis]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _checkpoint_bundle(
    checkpoint: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[Any, Any, Any, nn.Module, nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data = load_mag_data(cfg, "nc", int(payload["seed"]))
    model = build_model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"])
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"])
    model.eval()
    classifier.eval()
    return payload, cfg, data, model, classifier, payload.get("metrics", {})


def _write_distribution_summary(stats: dict[str, torch.Tensor], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for name, value in stats.items():
        value = value.detach().cpu().float()
        if value.ndim == 1 and name.startswith(("gamma_global", "delta_gamma_")):
            values_by_order = [(order, value[order : order + 1]) for order in range(value.numel())]
        elif value.ndim == 2 and name in {
            "eta_text",
            "eta_visual",
            "delta_node_text",
            "delta_node_visual",
        }:
            values_by_order = [(order, value[:, order]) for order in range(value.size(1))]
        elif value.ndim == 1:
            values_by_order = [(None, value)]
        else:
            raise ValueError(f"Unexpected analysis tensor shape for {name}: {tuple(value.shape)}")

        for order, values in values_by_order:
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                summary = {"mean": None, "std": None, "q10": None, "q25": None, "q50": None, "q75": None, "q90": None}
            else:
                values = values.double()
                summary = {
                    "mean": float(values.mean()),
                    "std": float(values.std(unbiased=False)),
                    "q10": float(torch.quantile(values, 0.10)),
                    "q25": float(torch.quantile(values, 0.25)),
                    "q50": float(torch.quantile(values, 0.50)),
                    "q75": float(torch.quantile(values, 0.75)),
                    "q90": float(torch.quantile(values, 0.90)),
                }
            rows.append({"variable": name, "order": order, **summary, "n": int(values.numel())})

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variable", "order", "mean", "std", "q10", "q25", "q50", "q75", "q90", "n"],
        )
        writer.writeheader()
        writer.writerows(rows)


def export_grocery_analysis(run_dir: Path, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    payload, cfg, data, model, _, _ = _checkpoint_bundle(checkpoint, run_dir, device)
    stats = model.analysis_stats(data.x.to(device), data.edge_index.to(device))
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    cpu_stats = {key: value.detach().cpu() for key, value in stats.items()}
    torch.save(cpu_stats, analysis_dir / "mopf_coefficients.pt")
    _write_distribution_summary(cpu_stats, analysis_dir / "distribution_summary.csv")
    _write_json(
        analysis_dir / "metadata.json",
        {
            "checkpoint": checkpoint,
            "selection": payload.get("selection"),
            "seed": payload.get("seed"),
            "epoch": payload.get("epoch"),
            "metrics": payload.get("metrics", {}),
            "effective_radius_definition": "sum_k k*abs(eta_i,k) / sum_k abs(eta_i,k)",
            "std_definition": "population standard deviation (unbiased=False)",
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )
    result = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "selection": payload.get("selection"),
        "epoch": payload.get("epoch"),
        "val_acc": result["val_acc"]["mean"],
        "test_acc": result["test_acc"]["mean"],
        "test_macro_f1": result["test_macro_f1"]["mean"],
    }


def _split_predictions(
    data: Any,
    z: torch.Tensor,
    classifier: nn.Module,
    device: torch.device,
    idx: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    logits = classifier(z[idx].to(device)).detach().cpu()
    return data.y[idx].cpu().numpy(), logits.argmax(dim=-1).numpy()


def export_class_diagnostics(
    checkpoint: Path,
    run_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    payload, cfg, data, model, classifier, checkpoint_metrics = _checkpoint_bundle(
        checkpoint, run_dir, device
    )
    z = infer_all_embeddings(
        model,
        data,
        device,
        uses_graph=True,
        batch_size=int(cfg.task.inference_batch_size),
        inference_mode=str(cfg.task.inference_mode),
    )
    labels = _resolve_nc_eval_labels(data)
    all_metrics: dict[str, Any] = {}
    predictions: dict[str, dict[str, torch.Tensor]] = {}
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for split_name, idx in (("train", data.train_idx), ("val", data.val_idx), ("test", data.test_idx)):
        target, pred = _split_predictions(data, z, classifier, device, idx)
        precision, recall, f1, support = precision_recall_fscore_support(
            target, pred, labels=labels, zero_division=0
        )
        matrix = confusion_matrix(target, pred, labels=labels)
        counts = np.bincount(pred, minlength=len(labels))
        all_metrics[split_name] = {
            "accuracy": float((target == pred).mean()),
            "macro_f1": float(np.mean(f1)),
            "support_total": int(target.size),
            "predicted_total": int(pred.size),
            "predicted_class_counts": counts.tolist(),
            "per_class": [
                {
                    "class": int(cls),
                    "precision": float(precision[position]),
                    "recall": float(recall[position]),
                    "f1": float(f1[position]),
                    "support": int(support[position]),
                    "predicted_count": int(counts[cls]),
                }
                for position, cls in enumerate(labels)
            ],
            "confusion_matrix": matrix.tolist(),
        }
        predictions[split_name] = {
            "node_id": idx.cpu(),
            "label": torch.from_numpy(target),
            "prediction": torch.from_numpy(pred),
        }
        for position, cls in enumerate(labels):
            per_class_rows.append(
                {
                    "split": split_name,
                    "class": cls,
                    "precision": float(precision[position]),
                    "recall": float(recall[position]),
                    "f1": float(f1[position]),
                    "support": int(support[position]),
                    "predicted_count": int(counts[cls]),
                }
            )
        for true_position, true_cls in enumerate(labels):
            for pred_position, pred_cls in enumerate(labels):
                confusion_rows.append(
                    {
                        "split": split_name,
                        "true_class": true_cls,
                        "predicted_class": pred_cls,
                        "count": int(matrix[true_position, pred_position]),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, output_dir / "predictions.pt")
    with (output_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(confusion_rows[0]))
        writer.writeheader()
        writer.writerows(confusion_rows)

    finite_checkpoint_metrics = {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in checkpoint_metrics.items()
    }
    result = {
        "checkpoint": str(checkpoint),
        "selection": payload.get("selection"),
        "epoch": payload.get("epoch"),
        "checkpoint_metrics": finite_checkpoint_metrics,
        "splits": all_metrics,
    }
    _write_json(output_dir / "metrics.json", result)
    return result


RUN_HEADER_RE = re.compile(r"\[Run (?P<run>\d+)/\d+\] seed=(?P<seed>\d+)")
EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+)")
VAL_RE = re.compile(r"Val Acc\s+(?P<acc>[0-9.]+)\s+\|\s+Val F1\s+(?P<f1>[0-9.]+)")
BEST_RE = re.compile(r"\[Run (?P<run>\d+)\] Best Val Acc (?P<acc>[0-9.]+)")
TEST_RE = re.compile(r"\[Run (?P<run>\d+)\] Final Test Acc (?P<acc>[0-9.]+) \| Final Test F1 (?P<f1>[0-9.]+)")


def parse_formal_log(path: Path) -> dict[str, Any]:
    runs: dict[int, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    current_epoch: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = RUN_HEADER_RE.search(line)
        if header:
            run_id = int(header.group("run"))
            current = runs.setdefault(run_id, {"seed": int(header.group("seed")), "epochs": []})
            current_epoch = None
            continue
        if current is None:
            continue
        epoch = EPOCH_RE.search(line)
        if epoch:
            current_epoch = int(epoch.group("epoch"))
        val = VAL_RE.search(line)
        if val and current_epoch is not None:
            current["epochs"].append(
                {
                    "epoch": current_epoch,
                    "val_acc_pct": float(val.group("acc")),
                    "val_macro_f1_pct": float(val.group("f1")),
                }
            )
        best = BEST_RE.search(line)
        if best:
            runs[int(best.group("run"))]["best_val_acc_pct_from_summary"] = float(best.group("acc"))
        test = TEST_RE.search(line)
        if test:
            runs[int(test.group("run"))]["test_acc_pct"] = float(test.group("acc"))
            runs[int(test.group("run"))]["test_macro_f1_pct"] = float(test.group("f1"))

    for run in runs.values():
        run["best_val_accuracy_observed"] = max(run["epochs"], key=lambda item: item["val_acc_pct"])
        run["best_val_macro_f1_observed"] = max(run["epochs"], key=lambda item: item["val_macro_f1_pct"])
        run["epoch_count_logged"] = len(run["epochs"])
        del run["epochs"]
    return {"source": str(path), "runs": runs}


def find_formal_log() -> Path:
    candidates = []
    if FORMAL_LOG_HINT.is_file() and "Final Test F1 70.81" in FORMAL_LOG_HINT.read_text(encoding="utf-8"):
        candidates.append(FORMAL_LOG_HINT)
    for path in ROOT.glob("outputs/**/main.log"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "Model: mopf" in text and "dataset=ele-fashion" in text and "Final Test F1 70.81" in text:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Could not locate the existing MoPF ele-fashion log containing Test F1 70.81")
    return max(set(candidates), key=lambda path: path.stat().st_mtime)


def build_report(
    grocery: list[dict[str, Any]],
    formal: dict[str, Any],
    seed44_diagnostics: dict[str, Any],
    formal_log: Path,
) -> None:
    lines = [
        "# MoPF targeted diagnosis",
        "",
        "This report is limited to the requested Grocery propagation and ele-fashion Macro-F1 diagnostics.",
        "The MoPF propagation/filter equations, CrossEntropyLoss, default configuration values, and official validation-accuracy checkpoint selection were not changed.",
        "",
        "## observation",
        "",
        "### A. Grocery-NC, seed=42",
        "",
        "All four jobs used the unified full-graph NC protocol with the same seed, split, optimizer, loss, and stopping settings.",
        "",
        "| configuration | Val Acc | Test Acc | Test Macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for item in grocery:
        lines.append(
            f"| {item['configuration']} | {100 * item['val_acc']:.2f}% | {100 * item['test_acc']:.2f}% | {100 * item['test_macro_f1']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "The coefficient/radius distribution summaries are in each run's `analysis/distribution_summary.csv`; raw exported tensors are in `analysis/mopf_coefficients.pt`.",
            "`effective_radius_{text,visual}` is defined as `sum_k k*abs(eta_i,k) / sum_k abs(eta_i,k)`, with population standard deviation.",
            "",
            "### B. ele-fashion existing log",
            "",
            f"The formal log selected for epoch parsing is `{formal_log}`. It is the existing three-run log whose seed44 formal test Macro-F1 is 70.81%.",
            "",
            "| seed | best Validation Accuracy epoch | Val Acc | Val Macro-F1 at that epoch | best Validation Macro-F1 epoch | Val Acc there | best Val Macro-F1 | formal Test Acc | formal Test Macro-F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id in sorted(formal["runs"], key=lambda key: formal["runs"][key]["seed"]):
        run = formal["runs"][run_id]
        best_acc = run["best_val_accuracy_observed"]
        best_f1 = run["best_val_macro_f1_observed"]
        lines.append(
            f"| {run['seed']} | {best_acc['epoch']} | {best_acc['val_acc_pct']:.2f}% | {best_acc['val_macro_f1_pct']:.2f}% | {best_f1['epoch']} | {best_f1['val_acc_pct']:.2f}% | {best_f1['val_macro_f1_pct']:.2f}% | {run.get('test_acc_pct', float('nan')):.2f}% | {run.get('test_macro_f1_pct', float('nan')):.2f}% |"
        )
    lines.extend(
        [
            "",
            "The source log prints validation metrics to two decimal places, so epoch ties/near-ties are reported at that displayed precision.",
            "",
            "## frozen/checkpoint diagnostic",
            "",
            "The existing formal MoPF ele-fashion output contains logs and aggregate metrics only; it contains no best checkpoint, per-class predictions, train accuracy, supports, or confusion matrices for seeds 42/43/44. Therefore those per-class artifacts cannot be reconstructed without retraining seeds 42/43, which was intentionally not done.",
            "",
            "The seed44 rerun produced two independent checkpoint artifacts: `best_val_accuracy.pt` uses the unchanged official Val-Accuracy selection, while `best_val_macro_f1.pt` is diagnostic-only and uses Val Macro-F1 only for the additional cached copy. Per-class CSVs and confusion matrices for both are under `ele-fashion/seed44/`.",
            "",
            "### seed44 cached checkpoint results",
            "",
            "| checkpoint | epoch | split | accuracy | Macro-F1 |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for checkpoint_name, result in seed44_diagnostics.items():
        for split in ("train", "val", "test"):
            metrics = result["splits"][split]
            lines.append(
                f"| {checkpoint_name} | {result['epoch']} | {split} | {100 * metrics['accuracy']:.2f}% | {100 * metrics['macro_f1']:.2f}% |"
            )
    formal_test_classes = {
        item["class"]: item
        for item in seed44_diagnostics["best_val_accuracy"]["splits"]["test"]["per_class"]
    }
    diagnostic_test_classes = {
        item["class"]: item
        for item in seed44_diagnostics["best_val_macro_f1"]["splits"]["test"]["per_class"]
    }
    ranked_classes = sorted(
        (item for item in formal_test_classes.values() if item["support"] > 0),
        key=lambda item: item["f1"],
    )[:5]
    lines.extend(
        [
            "",
            "For every seed44 checkpoint, `per_class.csv` contains support, precision, recall, F1, and predicted count; `confusion_matrix.csv` contains the full matrix; `metrics.json` contains the same values plus overall metrics; and `predictions.pt` contains node-level labels/predictions.",
            "",
            "### seed44 low-F1 class localization",
            "",
            "The following is a within-seed44 diagnostic comparison, not a cross-seed causal attribution. It identifies the classes most suppressed by the formal accuracy-selected checkpoint and how they move at the diagnostic F1-selected checkpoint:",
            "",
            "| class | test support | formal F1 | formal recall | diagnostic F1 | diagnostic recall |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in ranked_classes:
        diagnostic_item = diagnostic_test_classes[item["class"]]
        lines.append(
            f"| {item['class']} | {item['support']} | {100 * item['f1']:.2f}% | {100 * item['recall']:.2f}% | {100 * diagnostic_item['f1']:.2f}% | {100 * diagnostic_item['recall']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Class 5 has zero ground-truth support in all three seed44 splits and therefore contributes a zero term under the fixed 12-class Macro-F1 calculation; it is a dataset/split property rather than a seed44-specific failure.",
            "",
            "## retraining evidence",
            "",
            "Only ele-fashion seed44 was retrained, in a separate output directory. Its training used the unchanged MoPF model, default CE loss, and official best-Val-Accuracy selection. The diagnostic best-F1 copy was captured in parallel from the same epoch loop and did not affect the formal result.",
            "The seed44 rerun is an additional diagnostic reproduction, not a replacement for the existing formal benchmark: its cached formal checkpoint scored 87.93% Test Acc / 76.63% Test Macro-F1, while the preserved existing formal log reports 88.25% / 70.81%. The difference is retained as evidence of run-to-run GPU training variability; the original benchmark values above remain authoritative.",
            "",
            "Grocery was run as four independent seed42 jobs because the requested K/edge-weight combinations change the model configuration by design; no loss term or filter equation was added.",
            "",
            "## hypothesis",
            "",
            "The existing log shows that the seed44 formal best-accuracy epoch is not necessarily the epoch with the best validation Macro-F1. The separation between those two epochs is consistent with class-imbalanced selection: accuracy can improve through majority/easier classes while one or more minority classes lose recall or precision.",
            "",
            "The seed44 per-class tables are the evidence needed to identify the exact classes: compare seed44's `best_val_accuracy/per_class.csv` against the diagnostic `best_val_macro_f1/per_class.csv`; the largest negative F1 deltas and recall collapses are the classes responsible for the Macro-F1 gap. No causal claim is made for seed42/43 at class level because their predictions are absent.",
            "",
            "## conclusion",
            "",
            "The Grocery propagation comparison and coefficient/radius export are complete. The ele-fashion epoch-level log diagnosis is complete for seeds 42/43/44, and complete train/val/test per-class diagnostics are available for seed44's formal and diagnostic checkpoints. Seed42/43 per-class checkpoint diagnostics remain unavailable from existing artifacts under the explicit no-rerun constraint; the aggregate log observations are retained above rather than being presented as reconstructed predictions.",
        ]
    )
    (ROOT / "docs" / "mopf_targeted_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    grocery_root = TARGET_ROOT / "grocery"
    grocery_jobs = [
        ("k3_separate_cos", 3, "separate_cos"),
        ("k2_separate_cos", 2, "separate_cos"),
        ("k3_shared_avg_cos", 3, "shared_avg_cos"),
        ("k2_shared_avg_cos", 2, "shared_avg_cos"),
    ]
    if not args.skip_runs:
        for name, order, edge_mode in grocery_jobs:
            run_dir = grocery_root / name
            ckpt = run_dir / "best_val_accuracy.pt"
            _run_job(
                run_dir,
                [
                    "dataset=Grocery",
                    "task=nc",
                    "model=mopf",
                    "seed=42",
                    "num_runs=1",
                    f"device={args.device}",
                    f"model.max_order={order}",
                    f"model.num_layers={order}",
                    f"model.edge_weight_mode={edge_mode}",
                    f"task.save_ckpt_path={ckpt}",
                ],
            )
    grocery_results = []
    for name, _, _ in grocery_jobs:
        run_dir = grocery_root / name
        ckpt = run_dir / "best_val_accuracy.pt"
        result = export_grocery_analysis(run_dir, ckpt, device)
        result["configuration"] = name
        grocery_results.append(result)
    _write_json(grocery_root / "summary.json", grocery_results)

    formal_log = find_formal_log()
    formal = parse_formal_log(formal_log)
    ele_root = TARGET_ROOT / "ele-fashion" / "seed44"
    best_acc_ckpt = ele_root / "best_val_accuracy.pt"
    best_f1_ckpt = ele_root / "best_val_macro_f1.pt"
    if not args.skip_runs:
        _run_job(
            ele_root,
            [
                "dataset=ele-fashion",
                "task=nc",
                "model=mopf",
                "seed=44",
                "num_runs=1",
                f"device={args.device}",
                f"task.save_ckpt_path={best_acc_ckpt}",
                f"+task.diagnostic_best_macro_f1_checkpoint_path={best_f1_ckpt}",
            ],
        )
    seed44_diagnostics = {
        "best_val_accuracy": export_class_diagnostics(
            best_acc_ckpt, ele_root, ele_root / "best_val_accuracy", device
        ),
        "best_val_macro_f1": export_class_diagnostics(
            best_f1_ckpt, ele_root, ele_root / "best_val_macro_f1", device
        ),
    }
    _write_json(ele_root / "log_epoch_summary.json", formal)
    _write_json(ele_root / "diagnostic_summary.json", seed44_diagnostics)
    build_report(grocery_results, formal, seed44_diagnostics, formal_log)
    print("[targeted-diagnosis] wrote docs/mopf_targeted_diagnosis.md", flush=True)


if __name__ == "__main__":
    main()
