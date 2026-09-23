"""Run the controlled P2 versus P2-clean legacy relation-bias audit."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


DATASETS = ("Movies", "Grocery", "ele-fashion")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_clean(dataset: str, device: str, root: Path, skip_existing: bool) -> Path:
    output_dir = root / f"{dataset}_seed42" / "p2_clean"
    checkpoint = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (skip_existing and metrics_path.is_file() and checkpoint.is_file()):
        overrides = [
            f"dataset={dataset}",
            "task=nc",
            "model=mgsc_mag",
            "model.adaptive_context_gate=true",
            "model.direct_interacted_integration=true",
            "model.use_legacy_relation_order_bias=false",
            "seed=42",
            "num_runs=1",
            f"device={device}",
            f"task.save_ckpt_path={checkpoint}",
            f"hydra.run.dir={output_dir}",
        ]
        subprocess.run([sys.executable, "-m", "src.main", *overrides], cwd=REPO_ROOT, check=True)
    if not metrics_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"missing P2-clean output: {output_dir}")
    return checkpoint


def _cfg(dataset: str, legacy: bool):
    base = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    dataset_cfg = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml")
    task_cfg = OmegaConf.load(REPO_ROOT / "configs" / "task" / "nc.yaml")
    model_cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / "mgsc_mag.yaml")
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "paths": OmegaConf.to_container(base.paths, resolve=False),
            "dataset": dataset_cfg,
            "task": task_cfg,
            "model": model_cfg,
        }
    )
    cfg.model.direct_interacted_integration = True
    cfg.model.use_legacy_relation_order_bias = bool(legacy)
    OmegaConf.resolve(cfg)
    return cfg


def _load(dataset: str, checkpoint: Path, device: torch.device, legacy: bool):
    cfg = _cfg(dataset, legacy)
    data = load_mag_data(cfg, "nc", 42)
    info = {
        "input_dim": int(data.x.shape[1]),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, info).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    classifier.eval()
    return data, model, classifier


@torch.no_grad()
def _audit(dataset: str, model_name: str, checkpoint: Path, device: torch.device, legacy: bool) -> dict:
    data, model, classifier = _load(dataset, checkpoint, device, legacy)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    normal = model.analysis_intervention(x, edge_index, interaction="normal")
    off = model.analysis_intervention(x, edge_index, interaction="off")
    z_normal = normal["z"]
    z_off = off["z"]
    logits_normal = classifier(z_normal)
    logits_off = classifier(z_off)
    result = {
        "dataset": dataset,
        "model": model_name,
        "embedding_mae": float(torch.mean(torch.abs(z_off - z_normal)).item()),
        "logit_mae": float(torch.mean(torch.abs(logits_off - logits_normal)).item()),
        "prediction_flip_rate": float((logits_off.argmax(dim=-1) != logits_normal.argmax(dim=-1)).float().mean().item()),
        "nodes": int(z_normal.size(0)),
        "relation_legacy_enabled": legacy,
    }
    del normal, off, data, model, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _metrics(path: Path, dataset: str, model: str) -> dict:
    payload = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
    values = payload["metrics"]
    def mean(key: str) -> float:
        value = values[key]
        return float(value["mean"] if isinstance(value, dict) else value)
    return {
        "dataset": dataset,
        "model": model,
        "val_accuracy": mean("val_acc"),
        "val_macro_f1": mean("val_macro_f1"),
        "test_accuracy": mean("test_acc"),
        "test_macro_f1": mean("test_macro_f1"),
    }


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    audit_rows = []
    for dataset in DATASETS:
        p2_checkpoint = REPO_ROOT / "outputs/final/mgsc_p2_pilot" / f"{dataset}_nc" / "mgsc_mag_p2" / "best.pt"
        if not p2_checkpoint.is_file():
            raise FileNotFoundError(p2_checkpoint)
        clean_checkpoint = _run_clean(dataset, args.device, root, args.skip_existing)
        p2_metrics = _metrics(p2_checkpoint, dataset, "P2")
        clean_metrics = _metrics(clean_checkpoint, dataset, "P2-clean")
        rows.extend([p2_metrics, clean_metrics])
        audit_rows.extend([
            _audit(dataset, "P2", p2_checkpoint, device, True),
            _audit(dataset, "P2-clean", clean_checkpoint, device, False),
        ])
    _write_csv(root / "summary.csv", rows, ["dataset", "model", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"])
    _write_csv(root / "interaction_off_audit.csv", audit_rows, ["dataset", "model", "embedding_mae", "logit_mae", "prediction_flip_rate", "nodes", "relation_legacy_enabled"])
    by_dataset = {(row["dataset"], row["model"]): row for row in rows}
    clean_deltas = []
    report = [
        "# P2 Legacy Relation-Order Bias Cleanup Control",
        "",
        "P2-clean changes only `use_legacy_relation_order_bias=false`. MRC relation calibration, modality-specific normalized propagation, adaptive gates, direct interacted-state integration, preference, fusion, task head, split, and checkpoint criterion are unchanged. The clean variant is not set as the default.",
        "",
        "| Dataset | P2 test acc/F1 | P2-clean test acc/F1 | Acc delta | F1 delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        p2 = by_dataset[(dataset, "P2")]
        clean = by_dataset[(dataset, "P2-clean")]
        acc_delta = clean["test_accuracy"] - p2["test_accuracy"]
        f1_delta = clean["test_macro_f1"] - p2["test_macro_f1"]
        clean_deltas.append((acc_delta, f1_delta))
        report.append(f"| {dataset} | {p2['test_accuracy']:.4f}/{p2['test_macro_f1']:.4f} | {clean['test_accuracy']:.4f}/{clean['test_macro_f1']:.4f} | {acc_delta:+.4f} | {f1_delta:+.4f} |")
    viable = all(acc >= -0.005 for acc, _ in clean_deltas)
    report.extend([
        "",
        f"Clean architecture viable under the predeclared accuracy criterion (no dataset drops over 0.5 pp): `{viable}`.",
        "",
        "Interaction-off audit embedding MAE:",
        "",
    ])
    for row in audit_rows:
        report.append(f"- {row['dataset']} / {row['model']}: embedding MAE `{float(row['embedding_mae']):.6f}`, logit MAE `{float(row['logit_mae']):.6f}`, flip `{float(row['prediction_flip_rate']):.6f}`")
    report.extend([
        "",
        "This control is descriptive and does not justify deleting the legacy path from the default P2 candidate without broader qualification.",
    ])
    (REPO_ROOT / "docs" / "final" / "relation_cleanup_control.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "audit_rows": len(audit_rows), "clean_viable_accuracy_only": viable}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/relation_cleanup_control"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
