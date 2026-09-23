"""Run inference-only multi-order integration diagnostics for canonical P2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from mgsc_analysis_common import DATASETS, MODALITIES, checkpoint_path, load_checkpoint, resolve_device


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: torch.Tensor) -> tuple[float, float, float, float, float]:
    result = torch.quantile(values.float().cpu(), torch.tensor((0.10, 0.25, 0.50, 0.75, 0.90)))
    return tuple(float(value) for value in result.tolist())


@torch.no_grad()
def _order_rows(dataset: str, analysis: dict) -> list[dict]:
    rows = []
    for modality in MODALITIES:
        eta = analysis[f"eta_{modality}"].detach().float().cpu()
        q = eta.abs() / eta.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        effective = (q * torch.arange(q.size(1), dtype=q.dtype).unsqueeze(0)).sum(dim=-1)
        for order in range(q.size(1)):
            values = q[:, order]
            p10, p25, p50, p75, p90 = quantiles(values)
            rows.append({"dataset": dataset, "modality": modality, "order": order, "mean_contribution": float(values.mean()), "std": float(values.std(unbiased=False)), "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90, "nodes": int(values.numel()), "normalization": "abs_eta/sum_abs_eta"})
        p10, p25, p50, p75, p90 = quantiles(effective)
        rows.append({"dataset": dataset, "modality": modality, "order": "effective_mean", "mean_contribution": float(effective.mean()), "std": float(effective.std(unbiased=False)), "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90, "nodes": int(effective.numel()), "normalization": "sum(order*q), q=abs_eta/sum_abs_eta"})
    return rows


@torch.no_grad()
def _attention_rows(dataset: str, analysis: dict) -> list[dict]:
    rows = []
    for modality in MODALITIES:
        attention = analysis[f"attention_{modality}"].detach().float().cpu()
        average = attention.mean(dim=0)
        for query in range(average.size(0)):
            for key in range(average.size(1)):
                rows.append({"dataset": dataset, "modality": modality, "query_order": query, "key_order": key, "mean_attention": float(average[query, key]), "nodes": int(attention.size(0))})
    return rows


@torch.no_grad()
def _intervention_rows(dataset: str, data, model, classifier, device: torch.device) -> list[dict]:
    x, edge = data.x.to(device), data.edge_index.to(device)
    normal = model.analysis_intervention(x, edge, interaction="normal")
    reference_z = normal["z"].detach().clone()
    reference_logits = classifier(reference_z).detach().clone()
    reference_pred = reference_logits.argmax(dim=-1)
    names = {"normal": "normal", "query_collapse": "query_collapse", "uniform_attention": "uniform", "interaction_off": "off"}
    rows = []
    for name, api_value in names.items():
        if name == "normal":
            z, logits = reference_z, reference_logits
        else:
            out = model.analysis_intervention(x, edge, interaction=api_value)
            z, logits = out["z"], classifier(out["z"])
        rows.append({"dataset": dataset, "intervention": name, "embedding_mae": float((z - reference_z).abs().mean()), "logit_mae": float((logits - reference_logits).abs().mean()), "prediction_flip_rate": float((logits.argmax(dim=-1) != reference_pred).float().mean()), "nodes": int(z.size(0))})
    return rows


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    root = args.output_dir.resolve()
    order_rows, attention_rows, intervention_rows = [], [], []
    for dataset in DATASETS:
        checkpoint = checkpoint_path(dataset, args.seed, args.p2_root.resolve())
        data, model, classifier = load_checkpoint(dataset, args.seed, checkpoint, device)
        analysis = model.analysis_multi_order(data.x.to(device), data.edge_index.to(device))
        order_rows.extend(_order_rows(dataset, analysis))
        attention_rows.extend(_attention_rows(dataset, analysis))
        intervention_rows.extend(_intervention_rows(dataset, data, model, classifier, device))
        del data, model, classifier, analysis
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(root / "effective_order_contribution.csv", order_rows, list(order_rows[0]))
    write_csv(root / "average_attention_matrix.csv", attention_rows, list(attention_rows[0]))
    write_csv(root / "integration_intervention.csv", intervention_rows, list(intervention_rows[0]))
    (root / "manifest.json").write_text(json.dumps({"datasets": list(DATASETS), "seed": args.seed, "normalization": "q_i,k=abs(eta_i,k)/sum_j abs(eta_i,j)", "device": str(device)}, indent=2), encoding="utf-8")
    print(json.dumps({"order_rows": len(order_rows), "attention_rows": len(attention_rows), "intervention_rows": len(intervention_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--p2-root", type=Path, default=Path("outputs/final/final_candidate_nc"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/multi_order_integration_analysis"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
