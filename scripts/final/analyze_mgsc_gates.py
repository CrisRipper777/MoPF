"""Export P1 gate statistics from a saved MGSC-MAG checkpoint."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


def _device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    value = torch.device(raw)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but CUDA is unavailable")
    return value


def _cfg(dataset_name: str, task_name: str, seed: int):
    base = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    dataset = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / f"{dataset_name}.yaml")
    task = OmegaConf.load(REPO_ROOT / "configs" / "task" / f"{task_name}.yaml")
    model = OmegaConf.load(REPO_ROOT / "configs" / "model" / "mgsc_mag.yaml")
    cfg = OmegaConf.create(
        {
            "seed": int(seed),
            "paths": OmegaConf.to_container(base.paths, resolve=False),
            "dataset": dataset,
            "task": task,
            "model": model,
        }
    )
    OmegaConf.resolve(cfg)
    return cfg


def _quantiles(values: torch.Tensor) -> tuple[float, float, float, float, float]:
    quantiles = torch.tensor((0.10, 0.25, 0.50, 0.75, 0.90), dtype=values.dtype)
    result = torch.quantile(values.float(), quantiles)
    return tuple(float(value.item()) for value in result)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_gate_summary(
    checkpoint: Path,
    dataset_name: str,
    task_name: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, Path, Path]:
    cfg = _cfg(dataset_name, task_name, seed)
    data = load_mag_data(cfg, task_name, seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        analysis = model.analysis_multi_order(
            data.x.to(device), data.edge_index.to(device) if data.edge_index is not None else None
        )

    gate_rows: list[dict] = []
    gate_banks: dict[str, torch.Tensor] = {}
    for modality in ("text", "visual"):
        gates = analysis[f"context_gate_{modality}"]
        if not gates:
            raise ValueError("checkpoint did not export adaptive context gates")
        bank = torch.stack([gate.detach().float().cpu() for gate in gates], dim=1)
        gate_banks[modality] = bank
        for order in range(bank.size(1)):
            values = bank[:, order]
            p10, p25, p50, p75, p90 = _quantiles(values)
            gate_rows.append(
                {
                    "dataset": dataset_name,
                    "task": task_name,
                    "seed": seed,
                    "modality": modality,
                    "order": order + 1,
                    "mean": float(values.mean().item()),
                    "std": float(values.std(unbiased=False).item()),
                    "p10": p10,
                    "p25": p25,
                    "p50": p50,
                    "p75": p75,
                    "p90": p90,
                    "saturation_ratio": float(((values < 0.05) | (values > 0.95)).float().mean().item()),
                    "nodes": int(values.numel()),
                }
            )

    node_rows: list[dict] = []
    for modality, bank in gate_banks.items():
        node_std = bank.std(dim=1, unbiased=False)
        node_rows.append(
            {
                "dataset": dataset_name,
                "task": task_name,
                "seed": seed,
                "modality": modality,
                "mean_nodewise_gate_std_across_orders": float(node_std.mean().item()),
                "std_nodewise_gate_std_across_orders": float(node_std.std(unbiased=False).item()),
                "p50_nodewise_gate_std_across_orders": float(torch.quantile(node_std, 0.50).item()),
                "nodes": int(node_std.numel()),
            }
        )

    text_bank = gate_banks["text"]
    visual_bank = gate_banks["visual"]
    dataset_row = {
        "dataset": dataset_name,
        "task": task_name,
        "seed": seed,
        "mean_text_gate": float(text_bank.mean().item()),
        "mean_visual_gate": float(visual_bank.mean().item()),
        "text_visual_mean_gate_difference": float((text_bank.mean() - visual_bank.mean()).abs().item()),
        "max_order_distribution_mean_difference": float(
            torch.cdist(text_bank.mean(dim=0).view(-1, 1), visual_bank.mean(dim=0).view(-1, 1), p=1).max().item()
        ),
        "max_gate_saturation_ratio": max(
            float(row["saturation_ratio"]) for row in gate_rows
        ),
        "mean_nodewise_gate_std_text": node_rows[0]["mean_nodewise_gate_std_across_orders"],
        "mean_nodewise_gate_std_visual": node_rows[1]["mean_nodewise_gate_std_across_orders"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = output_dir / "gate_summary.csv"
    node_path = output_dir / "gate_nodewise_summary.csv"
    dataset_path = output_dir / "gate_dataset_summary.csv"
    fields = [
        "dataset", "task", "seed", "modality", "order", "mean", "std", "p10", "p25", "p50", "p75", "p90",
        "saturation_ratio", "nodes",
    ]
    _write_csv(gate_path, gate_rows, fields)
    _write_csv(
        node_path,
        node_rows,
        [
            "dataset", "task", "seed", "modality", "mean_nodewise_gate_std_across_orders",
            "std_nodewise_gate_std_across_orders", "p50_nodewise_gate_std_across_orders", "nodes",
        ],
    )
    _write_csv(
        dataset_path,
        [dataset_row],
        [
            "dataset", "task", "seed", "mean_text_gate", "mean_visual_gate", "text_visual_mean_gate_difference",
            "max_order_distribution_mean_difference", "max_gate_saturation_ratio", "mean_nodewise_gate_std_text",
            "mean_nodewise_gate_std_visual",
        ],
    )
    return gate_path, node_path, dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", choices=("nc", "lp"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = export_gate_summary(
        args.checkpoint,
        args.dataset,
        args.task,
        args.seed,
        args.output_dir,
        _device(args.device),
    )
    print("\n".join(str(path) for path in paths))
