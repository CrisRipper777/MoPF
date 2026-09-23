"""Run inference-only context-formation diagnostics for canonical P2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mgsc_analysis_common import DATASETS, MODALITIES, REPO_ROOT, checkpoint_path, load_checkpoint, resolve_device


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: torch.Tensor) -> tuple[float, float, float, float, float]:
    result = torch.quantile(values.float().cpu(), torch.tensor((0.10, 0.25, 0.50, 0.75, 0.90)))
    return tuple(float(value) for value in result.tolist())


def _edge_discrepancy_rows(dataset: str, analysis: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    physical = analysis["physical_edge_index"].detach().cpu()
    for modality in MODALITIES:
        semantic = analysis[f"semantic_cosine_{modality}"].detach().float().cpu()
        weights = analysis[f"relation_weight_{modality}"].detach().float().cpu()
        normalized = analysis[f"normalized_edge_weight_{modality}"].detach().float().cpu()
        rows.append({
            "dataset": dataset,
            "modality": modality,
            "physical_edges": int(physical.size(1)),
            "semantic_mean": float(semantic.mean()),
            "semantic_std": float(semantic.std(unbiased=False)),
            "relation_weight_mean": float(weights.mean()),
            "relation_weight_std": float(weights.std(unbiased=False)),
            "normalized_weight_mean": float(normalized.mean()),
            "normalized_weight_std": float(normalized.std(unbiased=False)),
            "semantic_weight_abs_discrepancy": float((semantic - weights).abs().mean()),
            "semantic_weight_pearson": float(np.corrcoef(semantic.numpy(), weights.numpy())[0, 1]) if semantic.numel() > 1 and float(semantic.std()) > 0 and float(weights.std()) > 0 else float("nan"),
        })
    semantic_delta = (analysis["semantic_cosine_text"] - analysis["semantic_cosine_visual"]).abs().float().cpu()
    weight_delta = (analysis["relation_weight_text"] - analysis["relation_weight_visual"]).abs().float().cpu()
    rows.append({
        "dataset": dataset,
        "modality": "text_visual_discrepancy",
        "physical_edges": int(physical.size(1)),
        "semantic_mean": float(semantic_delta.mean()),
        "semantic_std": float(semantic_delta.std(unbiased=False)),
        "relation_weight_mean": float(weight_delta.mean()),
        "relation_weight_std": float(weight_delta.std(unbiased=False)),
        "normalized_weight_mean": float((analysis["normalized_edge_weight_text"] - analysis["normalized_edge_weight_visual"]).abs().float().mean()),
        "normalized_weight_std": float((analysis["normalized_edge_weight_text"] - analysis["normalized_edge_weight_visual"]).abs().float().std(unbiased=False)),
        # This cross-modality row has no within-modality semantic-vs-weight
        # pairing; leave the field empty rather than exporting a NaN.
        "semantic_weight_abs_discrepancy": "",
        "semantic_weight_pearson": float(np.corrcoef(semantic_delta.numpy(), weight_delta.numpy())[0, 1]) if semantic_delta.numel() > 1 and float(semantic_delta.std()) > 0 and float(weight_delta.std()) > 0 else float("nan"),
    })
    return rows


def _neighbor_rows(dataset: str, analysis: dict[str, torch.Tensor], num_nodes: int) -> dict:
    edge_index = analysis["physical_edge_index"].detach().cpu()
    src = edge_index[0]
    text_weight = analysis["relation_weight_text"].detach().float().cpu()
    visual_weight = analysis["relation_weight_visual"].detach().float().cpu()
    tv_values, disagreement = [], []
    for node in range(num_nodes):
        mask = src == node
        if not bool(mask.any()):
            continue
        text = text_weight[mask]
        visual = visual_weight[mask]
        text = text / text.sum().clamp_min(1e-12)
        visual = visual / visual.sum().clamp_min(1e-12)
        tv_values.append(float(0.5 * (text - visual).abs().sum()))
        disagreement.append(int(torch.argmax(text).item() != torch.argmax(visual).item()))
    tv = torch.tensor(tv_values, dtype=torch.float32)
    return {
        "dataset": dataset,
        "nonisolated_nodes": int(tv.numel()),
        "isolated_nodes": int(num_nodes - tv.numel()),
        "tv_distance_mean": float(tv.mean()) if tv.numel() else float("nan"),
        "tv_distance_std": float(tv.std(unbiased=False)) if tv.numel() else float("nan"),
        "tv_distance_p50": float(torch.quantile(tv, 0.5)) if tv.numel() else float("nan"),
        "tv_distance_p90": float(torch.quantile(tv, 0.9)) if tv.numel() else float("nan"),
        "top_neighbor_disagreement": float(np.mean(disagreement)) if disagreement else float("nan"),
    }


@torch.no_grad()
def _gate_rows(dataset: str, analysis: dict, modality: str) -> list[dict]:
    rows = []
    for order, values in enumerate(analysis[f"context_gate_{modality}"], start=1):
        values = values.detach().float().cpu()
        p10, p25, p50, p75, p90 = quantiles(values)
        rows.append({"dataset": dataset, "modality": modality, "order": order, "mean": float(values.mean()), "std_across_nodes": float(values.std(unbiased=False)), "p10": p10, "p25": p25, "p50": p50, "p75": p75, "p90": p90, "saturation_ratio": float(((values < .05) | (values > .95)).float().mean()), "nodes": int(values.numel())})
    return rows


@torch.no_grad()
def _intervention_rows(dataset: str, data, model, classifier, device: torch.device, seed: int) -> list[dict]:
    x, edge = data.x.to(device), data.edge_index.to(device)
    normal = model.analysis_intervention(x, edge, gate_intervention="normal")
    reference_z = normal["z"].detach().clone()
    reference_logits = classifier(reference_z).detach().clone()
    reference_pred = reference_logits.argmax(dim=-1)
    rows = []
    for intervention in ("normal", "globalized", "shuffled"):
        if intervention == "normal":
            z, logits = reference_z, reference_logits
        else:
            out = model.analysis_intervention(x, edge, gate_intervention=intervention, gate_permutation_seed=seed + 7000)
            z, logits = out["z"], classifier(out["z"])
        rows.append({"dataset": dataset, "intervention": intervention, "embedding_mae": float((z - reference_z).abs().mean()), "logit_mae": float((logits - reference_logits).abs().mean()), "prediction_flip_rate": float((logits.argmax(dim=-1) != reference_pred).float().mean()), "nodes": int(z.size(0))})
    return rows


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    root = args.output_dir.resolve()
    p2_root = args.p2_root.resolve()
    relation_rows, neighbor_rows, gate_rows, intervention_rows = [], [], [], []
    for dataset in DATASETS:
        checkpoint = checkpoint_path(dataset, args.seed, p2_root)
        data, model, classifier = load_checkpoint(dataset, args.seed, checkpoint, device)
        x, edge = data.x.to(device), data.edge_index.to(device)
        analysis = model.analysis_multi_order(x, edge)
        relation_analysis = model.analysis_relation_calibration(x, edge)
        relation_rows.extend(_edge_discrepancy_rows(dataset, relation_analysis))
        neighbor_rows.append(_neighbor_rows(dataset, relation_analysis, int(data.num_nodes)))
        for modality in MODALITIES:
            gate_rows.extend(_gate_rows(dataset, analysis, modality))
        intervention_rows.extend(_intervention_rows(dataset, data, model, classifier, device, args.seed))
        del data, model, classifier, analysis
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(root / "relation_response.csv", relation_rows, list(relation_rows[0]))
    write_csv(root / "neighbor_allocation.csv", neighbor_rows, list(neighbor_rows[0]))
    write_csv(root / "adaptive_gate_summary.csv", gate_rows, list(gate_rows[0]))
    write_csv(root / "adaptive_gate_intervention.csv", intervention_rows, list(intervention_rows[0]))
    (root / "manifest.json").write_text(json.dumps({"datasets": list(DATASETS), "seed": args.seed, "checkpoint_root": str(p2_root), "device": str(device)}, indent=2), encoding="utf-8")
    print(json.dumps({"relation_rows": len(relation_rows), "neighbor_rows": len(neighbor_rows), "gate_rows": len(gate_rows), "intervention_rows": len(intervention_rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--p2-root", type=Path, default=Path("outputs/final/final_candidate_nc"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/context_formation_analysis"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
