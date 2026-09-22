#!/usr/bin/env python3
"""Describe label assortativity of the five frozen NC graphs."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.data import load_mag_data


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")


def main() -> int:
    rows = []
    for dataset in DATASETS:
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            cfg = compose(config_name="config", overrides=[f"dataset={dataset}", "task=nc", "model=mlp", "device=cpu"])
        data = load_mag_data(cfg, "nc", 42)
        src, dst = data.edge_index
        valid = (data.y[src] >= 0) & (data.y[dst] >= 0)
        src = src[valid]
        dst = dst[valid]
        same = data.y[src].eq(data.y[dst])
        degree = torch.bincount(src, minlength=int(data.num_nodes)).float()
        rows.append(
            {
                "dataset": dataset,
                "num_nodes": int(data.num_nodes),
                "directed_edges_after_loading": int(data.edge_index.size(1)),
                "mean_out_degree": float(degree.mean()),
                "edge_label_homophily": float(same.float().mean()),
            }
        )
    path = ROOT / "outputs" / "cosi_mag_backbone_diagnostics" / "analysis" / "graph_label_structure.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
