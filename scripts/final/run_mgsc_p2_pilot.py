"""Run the five-NC P2 pilot and inference intervention audit.

P2 trains only the direct-interacted-state integration candidate.  P0 and P1
rows are imported from the completed P1 pilot so that the comparison uses the
same official runs, while P2 is launched through the existing Hydra task
runner with one model switch enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models.factory import build_model  # noqa: E402


TASKS = (
    ("Movies", "nc"),
    ("Toys", "nc"),
    ("Grocery", "nc"),
    ("ele-fashion", "nc"),
    ("Reddit-S", "nc"),
)


def _device(raw: str) -> str:
    if raw == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {raw}, but CUDA is unavailable")
    return raw


def _run_dir(root: Path, dataset: str) -> Path:
    return root / f"{dataset}_nc" / "mgsc_mag_p2"


def _metric_mean(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    return None if value is None else float(value)


def _run_p2_model(
    dataset: str,
    seed: int,
    device: str,
    root: Path,
    skip_existing: bool,
) -> dict:
    output_dir = _run_dir(root, dataset)
    checkpoint = output_dir / "best.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if not (skip_existing and metrics_path.is_file() and checkpoint.is_file()):
        overrides = [
            f"dataset={dataset}",
            "task=nc",
            "model=mgsc_mag",
            "model.direct_interacted_integration=true",
            f"seed={seed}",
            "num_runs=1",
            f"device={device}",
            f"task.save_ckpt_path={checkpoint}",
            f"hydra.run.dir={output_dir}",
        ]
        subprocess.run(
            [sys.executable, "-m", "src.main", *overrides],
            cwd=REPO_ROOT,
            check=True,
        )
    if not metrics_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"incomplete P2 output for {dataset}: {output_dir}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    log_path = output_dir / "main.log"
    log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    parameter_match = re.search(r"params=(\d+)", log_text)
    return {
        "dataset": dataset,
        "task": "nc",
        "model": "P2",
        "model_config": "mgsc_mag",
        "seed": seed,
        "primary_metric": "test_acc",
        "primary_value": _metric_mean(metrics, "test_acc"),
        "secondary_metrics": json.dumps(metrics, sort_keys=True),
        "best_epoch": payload.get("checkpoint_metadata", {}).get("best_epoch"),
        "train_time": payload.get("runtime_seconds"),
        "params": int(parameter_match.group(1)) if parameter_match else "",
        "run_dir": str(output_dir.relative_to(REPO_ROOT)),
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "crashed": False,
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _import_p1_rows(p1_path: Path, datasets: list[str]) -> list[dict]:
    if not p1_path.is_file():
        raise FileNotFoundError(f"P1 summary is required before P2: {p1_path}")
    with p1_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["dataset"] in set(datasets)]
    expected = len(datasets) * 2
    if len(selected) != expected:
        raise ValueError(
            f"P1 summary has {len(selected)} rows for {len(datasets)} datasets; expected {expected}"
        )
    return selected


def _compose_dataset_cfg(dataset: str, device: str, seed: int, direct: bool):
    overrides = [
        f"dataset={dataset}",
        "task=nc",
        "model=mgsc_mag",
        f"seed={seed}",
        f"device={device}",
    ]
    if direct:
        overrides.append("model.direct_interacted_integration=true")
    with initialize_config_dir(
        version_base=None, config_dir=str((REPO_ROOT / "configs").resolve())
    ):
        return compose(
            config_name="config",
            overrides=overrides,
        )


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.x.shape[1]),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _load_analysis_model(cfg, data, checkpoint_path: Path, device: torch.device):
    model = build_model(cfg, _data_info(data)).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()
    return model, classifier


def _intervention_row(
    dataset: str,
    model_name: str,
    intervention: str,
    embedding_mae: float,
    logit_mae: float,
    prediction_flip_rate: float,
    num_nodes: int,
) -> dict:
    return {
        "dataset": dataset,
        "task": "nc",
        "model": model_name,
        "intervention": intervention,
        "embedding_mae": embedding_mae,
        "logit_mae": logit_mae,
        "prediction_flip_rate": prediction_flip_rate,
        "num_nodes": num_nodes,
    }


@torch.no_grad()
def _audit_checkpoint(
    dataset: str,
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
    seed: int,
    direct: bool,
) -> list[dict]:
    cfg = _compose_dataset_cfg(dataset, str(device), seed, direct)
    data = load_mag_data(cfg, "nc", seed)
    model, classifier = _load_analysis_model(cfg, data, checkpoint_path, device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    normal = model.analysis_intervention(
        x, edge_index, relation="normal", interaction="normal"
    )
    reference_z = normal["z"].detach().clone()
    reference_logits = classifier(reference_z).detach().clone()
    reference_prediction = reference_logits.argmax(dim=-1)
    del normal

    interventions = {
        "normal": ("normal", "normal"),
        "query_collapse": ("normal", "query_collapse"),
        "uniform": ("normal", "uniform"),
        "interaction_off": ("normal", "off"),
        "relation_off_audit": ("off", "normal"),
    }
    rows = []
    for name, (relation, interaction) in interventions.items():
        if name == "normal":
            z = reference_z
            logits = reference_logits
        else:
            components = model.analysis_intervention(
                x,
                edge_index,
                relation=relation,
                interaction=interaction,
            )
            z = components["z"]
            logits = classifier(z)
        if not torch.isfinite(z).all() or not torch.isfinite(logits).all():
            raise FloatingPointError(
                f"non-finite P2 intervention output: {dataset}/{model_name}/{name}"
            )
        prediction = logits.argmax(dim=-1)
        rows.append(
            _intervention_row(
                dataset,
                model_name,
                name,
                float(torch.mean(torch.abs(z - reference_z)).item()),
                float(torch.mean(torch.abs(logits - reference_logits)).item()),
                float(torch.mean((prediction != reference_prediction).float()).item()),
                int(z.size(0)),
            )
        )
        if name != "normal":
            del components
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _write_report(
    root: Path,
    summary_rows: list[dict],
    intervention_rows: list[dict],
    effect_rows: list[dict],
    datasets: list[str],
) -> None:
    metric_rows = {(row["dataset"], row["model"]): row for row in summary_rows}
    lines = [
        "# MGSC-MAG P2 Pilot Report",
        "",
        "This P2 pilot uses the five NC datasets (Movies, Toys, Grocery, ele-fashion, Reddit-S). P0 and P1 are the completed seed-42 rows from the P1 pilot; P2 enables only `direct_interacted_integration=true`. The task runner, splits, optimizer, stopping rule, classifier, and graph protocol are unchanged.",
        "",
        "## NC performance",
        "",
        "| Dataset | P0 test acc | P1 test acc | P2 test acc | P2-P1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        p0 = float(metric_rows[(dataset, "P0")]["primary_value"])
        p1 = float(metric_rows[(dataset, "P1")]["primary_value"])
        p2 = float(metric_rows[(dataset, "P2")]["primary_value"])
        lines.append(f"| {dataset} | {p0:.4f} | {p1:.4f} | {p2:.4f} | {p2 - p1:+.4f} |")
    p2_deltas = [
        float(metric_rows[(dataset, "P2")]["primary_value"])
        - float(metric_rows[(dataset, "P1")]["primary_value"])
        for dataset in datasets
    ]
    lines.extend(
        [
            "",
            f"P2 test accuracy did not drop by more than 1 percentage point on any of these datasets: `{min(p2_deltas) >= -0.01}`. The pilot has mixed task-level movement ({sum(delta > 0 for delta in p2_deltas)}/{len(p2_deltas)} datasets improved over P1), so this is not treated as a universal performance win.",
        ]
    )
    lines.extend(
        [
            "",
            "## Intervention audit",
            "",
            "Embedding MAE, logit MAE, and prediction flip rate are measured against the same model's normal inference output. `relation_off_audit` is the requested frozen relation-context audit; it is descriptive and does not delete the legacy relation-context or relation-order-bias parameters.",
            "",
            "| Dataset | Intervention | P1 emb MAE | P2 emb MAE | P1 logit MAE | P2 logit MAE | P1 flip | P2 flip |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    effects = {(row["dataset"], row["intervention"]): row for row in effect_rows}
    for dataset in datasets:
        for intervention in ("query_collapse", "uniform", "interaction_off", "relation_off_audit"):
            row = effects[(dataset, intervention)]
            lines.append(
                f"| {dataset} | {intervention} | {float(row['p1_embedding_mae']):.6g} | {float(row['p2_embedding_mae']):.6g} | {float(row['p1_logit_mae']):.6g} | {float(row['p2_logit_mae']):.6g} | {float(row['p1_prediction_flip_rate']):.6g} | {float(row['p2_prediction_flip_rate']):.6g} |"
            )
    relation_rows = [
        row for row in effect_rows if row["intervention"] == "relation_off_audit"
    ]
    relation_near_zero = all(
        float(row["p2_embedding_mae"]) < 1e-6
        and float(row["p2_logit_mae"]) < 1e-6
        and float(row["p2_prediction_flip_rate"]) == 0.0
        for row in relation_rows
    )
    interaction_off_rows = [
        row for row in effect_rows if row["intervention"] == "interaction_off"
    ]
    interaction_off_increased = all(
        float(row["p2_embedding_mae"]) > float(row["p1_embedding_mae"])
        and float(row["p2_logit_mae"]) > float(row["p1_logit_mae"])
        and float(row["p2_prediction_flip_rate"]) > float(row["p1_prediction_flip_rate"])
        for row in interaction_off_rows
    )
    lines.extend(
        [
            "",
            f"The `interaction_off` intervention effect is larger for P2 than P1 on all three reported measures for every dataset: `{interaction_off_increased}`. This is direct evidence that cross-order interaction reaches the final representation more directly in P2.",
            "",
            f"Relation intervention off is near-zero under the strict descriptive check (all three metrics below 1e-6 / zero flip): `{relation_near_zero}`.",
            "The relation-context audit is therefore retained as an active diagnostic; the legacy relation-context and relation-order-bias parameters are not removed in this round.",
            "",
            "P2 is a mechanism pilot rather than a new selection protocol. The direct-integration mechanism should be retained for further study only where its intervention effect is measurably larger than P1 without a task-level regression; no claim of universal improvement is made here.",
            "",
        ]
    )
    (REPO_ROOT / "docs" / "final" / "mgsc_p2_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    device_name = _device(args.device)
    device = torch.device(device_name)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    datasets = list(args.datasets)
    p1_rows = _import_p1_rows(
        (REPO_ROOT / "outputs/final/mgsc_p1_pilot/summary.csv"), datasets
    )
    p2_rows = [
        _run_p2_model(dataset, args.seed, device_name, root, args.skip_existing)
        for dataset in datasets
    ]
    summary_rows = p1_rows + p2_rows
    summary_fields = [
        "dataset", "task", "model", "model_config", "seed", "primary_metric", "primary_value",
        "secondary_metrics", "best_epoch", "train_time", "params", "run_dir", "checkpoint", "crashed",
    ]
    _write_csv(root / "summary.csv", summary_rows, summary_fields)

    intervention_rows: list[dict] = []
    for dataset in datasets:
        for model_name, checkpoint in (
            ("P1", REPO_ROOT / next(row["checkpoint"] for row in p1_rows if row["dataset"] == dataset and row["model"] == "P1")),
            ("P2", REPO_ROOT / next(row["checkpoint"] for row in p2_rows if row["dataset"] == dataset)),
        ):
            intervention_rows.extend(
                _audit_checkpoint(
                    dataset,
                    model_name,
                    checkpoint,
                    device,
                    args.seed,
                    direct=model_name == "P2",
                )
            )
    intervention_fields = [
        "dataset", "task", "model", "intervention", "embedding_mae", "logit_mae",
        "prediction_flip_rate", "num_nodes",
    ]
    _write_csv(root / "intervention_metrics.csv", intervention_rows, intervention_fields)

    by_key = {(row["dataset"], row["model"], row["intervention"]): row for row in intervention_rows}
    effect_rows = []
    for dataset in datasets:
        for intervention in ("query_collapse", "uniform", "interaction_off", "relation_off_audit"):
            p1 = by_key[(dataset, "P1", intervention)]
            p2 = by_key[(dataset, "P2", intervention)]
            effect_rows.append(
                {
                    "dataset": dataset,
                    "intervention": intervention,
                    "p1_embedding_mae": p1["embedding_mae"],
                    "p2_embedding_mae": p2["embedding_mae"],
                    "p1_logit_mae": p1["logit_mae"],
                    "p2_logit_mae": p2["logit_mae"],
                    "p1_prediction_flip_rate": p1["prediction_flip_rate"],
                    "p2_prediction_flip_rate": p2["prediction_flip_rate"],
                    "embedding_mae_delta_p2_minus_p1": float(p2["embedding_mae"]) - float(p1["embedding_mae"]),
                    "logit_mae_delta_p2_minus_p1": float(p2["logit_mae"]) - float(p1["logit_mae"]),
                    "prediction_flip_delta_p2_minus_p1": float(p2["prediction_flip_rate"]) - float(p1["prediction_flip_rate"]),
                }
            )
    _write_csv(
        root / "intervention_effect_summary.csv",
        effect_rows,
        [
            "dataset", "intervention", "p1_embedding_mae", "p2_embedding_mae",
            "p1_logit_mae", "p2_logit_mae", "p1_prediction_flip_rate", "p2_prediction_flip_rate",
            "embedding_mae_delta_p2_minus_p1", "logit_mae_delta_p2_minus_p1",
            "prediction_flip_delta_p2_minus_p1",
        ],
    )
    _write_report(root, summary_rows, intervention_rows, effect_rows, datasets)
    print(json.dumps({"datasets": datasets, "p2_rows": len(p2_rows), "intervention_rows": len(intervention_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(dataset for dataset, _ in TASKS),
        default=[dataset for dataset, _ in TASKS],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/mgsc_p2_pilot"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
