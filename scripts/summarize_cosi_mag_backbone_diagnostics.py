#!/usr/bin/env python3
"""Summarize CoSI backbone probes and MMGCN node-ID diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cosi_mag_backbone_diagnostics"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
BACKBONE_MODES = ("all_plain", "no_graph", "last_hop", "simple_fusion", "early_fusion")
MMGCN_MODES = ("reference_fixed", "zero_fixed", "trainable")
METRICS = ("test_acc", "test_macro_f1")


def checkpoint_dir(family: str, mode: str, dataset: str) -> Path:
    if family == "backbone" and mode == "all_plain":
        return ROOT / "outputs" / "cosi_mag_final_joint_ablation" / "all_plain" / "nc" / dataset / "runs_42_43_44"
    return OUT / family / mode / dataset / "runs_42_43_44"


def values(family: str, mode: str, dataset: str) -> dict[int, dict[str, float]]:
    path = checkpoint_dir(family, mode, dataset)
    result = {}
    for index, seed in enumerate(SEEDS, 1):
        checkpoint = path / f"best_run{index}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload["seed"]) != seed:
            raise ValueError(f"seed mismatch in {checkpoint}")
        result[seed] = {metric: float(payload["metrics"][metric]) for metric in METRICS}
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_family(family: str, modes: tuple[str, ...], reference: str) -> tuple[list[dict], list[dict]]:
    raw = {(mode, dataset): values(family, mode, dataset) for mode in modes for dataset in DATASETS}
    summary = []
    paired = []
    for dataset in DATASETS:
        for mode in modes:
            for metric in METRICS:
                current = [raw[(mode, dataset)][seed][metric] for seed in SEEDS]
                baseline = [raw[(reference, dataset)][seed][metric] for seed in SEEDS]
                deltas = [value - base for value, base in zip(current, baseline)]
                summary.append(
                    {
                        "family": family,
                        "dataset": dataset,
                        "mode": mode,
                        "metric": metric,
                        "mean": float(np.mean(current)),
                        "population_std": float(np.std(current, ddof=0)),
                        "reference_mode": reference,
                        "delta_vs_reference": float(np.mean(deltas)),
                    }
                )
                for seed, value, base, delta in zip(SEEDS, current, baseline, deltas):
                    paired.append(
                        {
                            "family": family,
                            "dataset": dataset,
                            "mode": mode,
                            "metric": metric,
                            "seed": seed,
                            "value": value,
                            "reference_mode": reference,
                            "reference_value": base,
                            "delta": delta,
                        }
                    )
    return summary, paired


def main() -> int:
    backbone_summary, backbone_paired = summarize_family("backbone", BACKBONE_MODES, "all_plain")
    mmgcn_summary, mmgcn_paired = summarize_family("mmgcn", MMGCN_MODES, "reference_fixed")
    analysis = OUT / "analysis"
    write_csv(analysis / "summary.csv", backbone_summary + mmgcn_summary)
    write_csv(analysis / "paired_seed_deltas.csv", backbone_paired + mmgcn_paired)
    (analysis / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": list(DATASETS),
                "seeds": list(SEEDS),
                "backbone_reference": "all_plain",
                "mmgcn_reference": "reference_fixed",
                "uncertainty": "population SD across three training seeds on one fixed split",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
