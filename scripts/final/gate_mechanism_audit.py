"""Audit whether the learned MGSC-MAG gates have node-specific functional use.

This script is inference-only.  It never changes task checkpoints and never
uses a gate intervention during training.  It exports node/order gate
distributions, aligns gates with the model-independent M1 preferred-lambda
probe, and evaluates globalized, shuffled, and fixed-0.9 gate controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import kruskal, spearmanr
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


DATASETS = ("Movies", "Grocery", "ele-fashion", "Toys", "Reddit-S")
M1_DATASETS = ("Movies", "Grocery", "ele-fashion")
MODALITIES = ("text", "visual")
ORDERS = (1, 2, 3)
GATE_INTERVENTIONS = ("normal", "globalized", "shuffled", "fixed_0.9")


def _device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    return device


def _cfg(dataset: str, seed: int):
    base = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    dataset_cfg = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml")
    task_cfg = OmegaConf.load(REPO_ROOT / "configs" / "task" / "nc.yaml")
    model_cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / "mgsc_mag.yaml")
    cfg = OmegaConf.create(
        {
            "seed": int(seed),
            "paths": OmegaConf.to_container(base.paths, resolve=False),
            "dataset": dataset_cfg,
            "task": task_cfg,
            "model": model_cfg,
        }
    )
    cfg.model.direct_interacted_integration = True
    cfg.model.use_legacy_relation_order_bias = True
    OmegaConf.resolve(cfg)
    return cfg


def _info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.x.shape[1]),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _checkpoint(dataset: str, root: Path) -> Path:
    path = root / f"{dataset}_nc" / "mgsc_mag_p2" / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing P2 checkpoint: {path}")
    return path


def _load(dataset: str, checkpoint: Path, device: torch.device, seed: int):
    cfg = _cfg(dataset, seed)
    data = load_mag_data(cfg, "nc", seed)
    model = build_model(cfg, _info(data)).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    classifier.eval()
    return data, model, classifier


def _quantiles(values: torch.Tensor) -> tuple[float, ...]:
    q = torch.tensor((0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95), dtype=torch.float32)
    result = torch.quantile(values.float().cpu(), q)
    return tuple(float(value.item()) for value in result)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _preferred_lambda_map(path: Path, dataset: str, modality: str) -> dict[int, float]:
    losses: dict[tuple[int, float], list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != dataset or row["modality"] != modality:
                continue
            if row["is_isolated"].lower() == "true" or int(row["target"]) < 0:
                continue
            losses[(int(row["node_id"]), float(row["lambda"]))].append(float(row["ce_loss"]))
    by_node: dict[int, dict[float, float]] = defaultdict(dict)
    for (node, lambda_value), values in losses.items():
        by_node[node][lambda_value] = float(np.mean(values))
    preferred = {}
    for node, values in by_node.items():
        preferred[node] = min(values, key=lambda value: (values[value], value))
    return preferred


def _nc_labels(data) -> list[int]:
    indices = [idx for idx in (data.train_idx, data.val_idx, data.test_idx) if idx is not None]
    values = data.y[torch.cat(indices)].detach().cpu()
    return sorted({int(value) for value in values.tolist() if 0 <= int(value) < int(data.num_classes)})


def _split_metrics(logits: torch.Tensor, data, indices: torch.Tensor, labels: list[int]) -> tuple[float, float]:
    indices = indices.to(logits.device)
    target = data.y.to(logits.device)[indices]
    valid = (target >= 0) & (target < int(data.num_classes))
    target = target[valid]
    prediction = logits[indices][valid].argmax(dim=-1)
    if target.numel() == 0:
        return float("nan"), float("nan")
    accuracy = float((prediction == target).float().mean().item())
    macro_f1 = float(
        f1_score(
            target.cpu().numpy(),
            prediction.cpu().numpy(),
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    return accuracy, macro_f1


def _a1_rows(dataset: str, bank: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for modality in MODALITIES:
        values_by_order = bank[modality].unbind(dim=1)
        for order, values in enumerate(values_by_order, start=1):
            p05, p10, p25, p50, p75, p90, p95 = _quantiles(values)
            rows.append(
                {
                    "dataset": dataset,
                    "modality": modality,
                    "order": order,
                    "mean": float(values.mean().item()),
                    "std_across_nodes": float(values.std(unbiased=False).item()),
                    "p05": p05,
                    "p10": p10,
                    "p25": p25,
                    "p50": p50,
                    "p75": p75,
                    "p90": p90,
                    "p95": p95,
                    "min": float(values.min().item()),
                    "max": float(values.max().item()),
                    "saturation_low": float((values < 0.05).float().mean().item()),
                    "saturation_high": float((values > 0.95).float().mean().item()),
                    "nodes": int(values.numel()),
                }
            )
    return rows


def _a2_rows(
    dataset: str,
    bank: dict[str, torch.Tensor],
    data,
    losses_path: Path,
) -> list[dict]:
    rows = []
    for modality in MODALITIES:
        preferred = _preferred_lambda_map(losses_path, dataset, modality)
        gate_mean = bank[modality].mean(dim=1).cpu()
        eligible_nodes = sorted(set(preferred).intersection(set(data.val_idx.cpu().tolist())))
        lambda_values = np.asarray([preferred[node] for node in eligible_nodes], dtype=np.float64)
        gate_values = np.asarray([float(gate_mean[node].item()) for node in eligible_nodes], dtype=np.float64)
        if len(eligible_nodes) < 2:
            rho, rho_p = float("nan"), float("nan")
        else:
            correlation = spearmanr(lambda_values, gate_values)
            rho, rho_p = float(correlation.statistic), float(correlation.pvalue)
        groups: list[tuple[str, np.ndarray]] = []
        for lambda_value in (0.00, 0.25, 0.50, 0.75, 1.00):
            groups.append((f"lambda_{lambda_value:.2f}", gate_values[lambda_values == lambda_value]))
        groups.extend(
            [
                ("low_context_lambda_le_0.25", gate_values[lambda_values <= 0.25]),
                ("high_context_lambda_ge_0.75", gate_values[lambda_values >= 0.75]),
            ]
        )
        nonempty = [values for _, values in groups if values.size]
        if len(nonempty) >= 2:
            kw = kruskal(*nonempty)
            kw_h, kw_p = float(kw.statistic), float(kw.pvalue)
        else:
            kw_h, kw_p = float("nan"), float("nan")
        low = gate_values[lambda_values <= 0.25]
        high = gate_values[lambda_values >= 0.75]
        if low.size and high.size:
            pooled = math.sqrt(
                ((low.size - 1) * float(low.var(ddof=1)) + (high.size - 1) * float(high.var(ddof=1)))
                / max(low.size + high.size - 2, 1)
            )
            effect = (float(high.mean()) - float(low.mean())) / pooled if pooled > 0 else float("nan")
        else:
            effect = float("nan")
        for group_name, values in groups:
            rows.append(
                {
                    "dataset": dataset,
                    "modality": modality,
                    "group": group_name,
                    "count": int(values.size),
                    "proportion": float(values.size / max(len(gate_values), 1)),
                    "mean_gate": float(values.mean()) if values.size else float("nan"),
                    "std_gate": float(values.std()) if values.size else float("nan"),
                    "median_gate": float(np.median(values)) if values.size else float("nan"),
                    "spearman_rho_lambda_vs_gate": rho,
                    "spearman_pvalue": rho_p,
                    "kruskal_h": kw_h,
                    "kruskal_pvalue": kw_p,
                    "high_minus_low_mean_gate": float(high.mean() - low.mean()) if low.size and high.size else float("nan"),
                    "cohens_d_high_vs_low": effect,
                    "eligible_val_nodes": len(gate_values),
                }
            )
    return rows


@torch.no_grad()
def _a3_rows(
    dataset: str,
    data,
    model,
    classifier,
    device: torch.device,
    seed: int,
) -> list[dict]:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    labels = _nc_labels(data)
    normal = model.analysis_intervention(x, edge_index, gate_intervention="normal")
    normal_z = normal["z"].detach().clone()
    normal_logits = classifier(normal_z).detach().clone()
    normal_val_acc, normal_val_f1 = _split_metrics(normal_logits, data, data.val_idx, labels)
    normal_test_acc, normal_test_f1 = _split_metrics(normal_logits, data, data.test_idx, labels)
    normal_prediction = normal_logits.argmax(dim=-1)
    rows = []
    for intervention in GATE_INTERVENTIONS:
        if intervention == "normal":
            z = normal_z
            logits = normal_logits
        else:
            components = model.analysis_intervention(
                x,
                edge_index,
                gate_intervention=intervention,
                gate_permutation_seed=seed + 7000,
            )
            z = components["z"]
            logits = classifier(z)
        val_acc, val_f1 = _split_metrics(logits, data, data.val_idx, labels)
        test_acc, test_f1 = _split_metrics(logits, data, data.test_idx, labels)
        rows.append(
            {
                "dataset": dataset,
                "intervention": intervention,
                "embedding_mae": float(torch.mean(torch.abs(z - normal_z)).item()),
                "logit_mae": float(torch.mean(torch.abs(logits - normal_logits)).item()),
                "prediction_flip_rate": float((logits.argmax(dim=-1) != normal_prediction).float().mean().item()),
                "normal_val_accuracy": normal_val_acc,
                "intervened_val_accuracy": val_acc,
                "delta_val_accuracy": val_acc - normal_val_acc,
                "normal_val_macro_f1": normal_val_f1,
                "intervened_val_macro_f1": val_f1,
                "delta_val_macro_f1": val_f1 - normal_val_f1,
                "normal_test_accuracy": normal_test_acc,
                "intervened_test_accuracy": test_acc,
                "delta_test_accuracy": test_acc - normal_test_acc,
                "normal_test_macro_f1": normal_test_f1,
                "intervened_test_macro_f1": test_f1,
                "delta_test_macro_f1": test_f1 - normal_test_f1,
                "nodes": int(z.size(0)),
            }
        )
        if intervention != "normal":
            del components
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _report(a1: list[dict], a2: list[dict], a3: list[dict]) -> str:
    lines = [
        "# MGSC-MAG Gate Mechanism Audit",
        "",
        "This is an inference-only qualification audit of the existing P2 candidate. No new trainable mechanism, task protocol, split, checkpoint criterion, or early-fusion path was added.",
        "",
        "## A1. Node/order gate distributions",
        "",
        "The `std_across_nodes` column is the standard deviation across nodes for a fixed modality/order. It is distinct from the earlier node-wise standard deviation across propagation orders.",
        "",
    ]
    for dataset in DATASETS:
        rows = [row for row in a1 if row["dataset"] == dataset]
        max_std = max(float(row["std_across_nodes"]) for row in rows)
        max_sat = max(float(row["saturation_low"]) + float(row["saturation_high"]) for row in rows)
        lines.append(f"- **{dataset}**: maximum across-node gate std `{max_std:.6f}`; maximum saturation `{max_sat:.6f}`.")
    lines.extend(
        [
            "",
            "## A2. M1 preferred lambda versus learned gate",
            "",
            "Preferred lambda is the aggregated model-independent descriptive probe from M1, not a ground-truth label. Gate alignment is therefore descriptive and does not establish causal supervision.",
            "",
        ]
    )
    for dataset in M1_DATASETS:
        for modality in MODALITIES:
            rows = [row for row in a2 if row["dataset"] == dataset and row["modality"] == modality and row["group"] in {"low_context_lambda_le_0.25", "high_context_lambda_ge_0.75"}]
            if len(rows) == 2:
                low = next(row for row in rows if row["group"].startswith("low_"))
                high = next(row for row in rows if row["group"].startswith("high_"))
                rho = float(low["spearman_rho_lambda_vs_gate"])
                difference = float(low["high_minus_low_mean_gate"])
                lines.append(
                    f"- **{dataset}/{modality}**: Spearman rho `{rho:.4f}`, low-context mean `{float(low['mean_gate']):.4f}`, high-context mean `{float(high['mean_gate']):.4f}`, high-minus-low `{difference:.4f}`, Cohen's d `{float(low['cohens_d_high_vs_low']):.4f}`."
                )
    lines.extend(
        [
            "",
            "## A3. Functional gate interventions",
            "",
            "Globalized replaces each order's gate by its normal node mean. Shuffled uses an independent fixed node permutation for each modality/order and preserves each gate marginal distribution. Fixed-0.9 replaces every gate value by 0.9. Metrics are measured relative to normal inference from the same checkpoint.",
            "",
        ]
    )
    for dataset in DATASETS:
        rows = [row for row in a3 if row["dataset"] == dataset and row["intervention"] != "normal"]
        strongest = max(rows, key=lambda row: float(row["embedding_mae"]))
        lines.append(
            f"- **{dataset}**: largest embedding MAE `{float(strongest['embedding_mae']):.6f}` under `{strongest['intervention']}`; corresponding logit MAE `{float(strongest['logit_mae']):.6f}` and flip rate `{float(strongest['prediction_flip_rate']):.6f}`."
        )
    lines.extend(
        [
            "",
            "Interpretation: a non-zero shuffled intervention with preserved gate marginals is evidence that node-to-gate correspondence is functionally used. This remains an intervention diagnostic, not a proof that the learned gate recovers a unique causal preference.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    losses_path = args.m1_dir.resolve() / "node_lambda_losses.csv"
    a1_rows: list[dict] = []
    a2_rows: list[dict] = []
    a3_rows: list[dict] = []
    metadata = {"seed": args.seed, "device": str(device), "datasets": list(DATASETS)}
    for dataset in DATASETS:
        data, model, classifier = _load(dataset, _checkpoint(dataset, args.p2_dir.resolve()), device, args.seed)
        normal = model.analysis_intervention(data.x.to(device), data.edge_index.to(device), gate_intervention="normal")
        bank = {
            modality: torch.stack(normal[f"context_gate_{modality}"], dim=1).detach().float().cpu()
            for modality in MODALITIES
        }
        a1_rows.extend(_a1_rows(dataset, bank))
        if dataset in M1_DATASETS:
            a2_rows.extend(_a2_rows(dataset, bank, data, losses_path))
        del normal
        a3_rows.extend(_a3_rows(dataset, data, model, classifier, device, args.seed))
        del data, model, classifier, bank
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(
        output_dir / "gate_node_distribution.csv",
        a1_rows,
        ["dataset", "modality", "order", "mean", "std_across_nodes", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "min", "max", "saturation_low", "saturation_high", "nodes"],
    )
    _write_csv(
        output_dir / "gate_lambda_alignment.csv",
        a2_rows,
        ["dataset", "modality", "group", "count", "proportion", "mean_gate", "std_gate", "median_gate", "spearman_rho_lambda_vs_gate", "spearman_pvalue", "kruskal_h", "kruskal_pvalue", "high_minus_low_mean_gate", "cohens_d_high_vs_low", "eligible_val_nodes"],
    )
    _write_csv(
        output_dir / "gate_intervention_metrics.csv",
        a3_rows,
        ["dataset", "intervention", "embedding_mae", "logit_mae", "prediction_flip_rate", "normal_val_accuracy", "intervened_val_accuracy", "delta_val_accuracy", "normal_val_macro_f1", "intervened_val_macro_f1", "delta_val_macro_f1", "normal_test_accuracy", "intervened_test_accuracy", "delta_test_accuracy", "normal_test_macro_f1", "intervened_test_macro_f1", "delta_test_macro_f1", "nodes"],
    )
    (output_dir / "gate_mechanism_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPO_ROOT / "docs" / "final" / "gate_mechanism_audit.md").write_text(
        _report(a1_rows, a2_rows, a3_rows) + "\n", encoding="utf-8"
    )
    print(json.dumps({"a1_rows": len(a1_rows), "a2_rows": len(a2_rows), "a3_rows": len(a3_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--p2-dir", type=Path, default=Path("outputs/final/mgsc_p2_pilot"))
    parser.add_argument("--m1-dir", type=Path, default=Path("outputs/final/context_demand"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/gate_mechanism_audit"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
