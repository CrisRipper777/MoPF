#!/usr/bin/env python3
"""Summarize Full, three single ablations, and the five-NC all-plain control."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
METRICS = ("test_acc", "test_macro_f1")
VARIANTS = ("no_mrc", "no_semantic_anchor", "no_rcmi", "all_plain")
LABELS = {
    "full": "Full",
    "no_mrc": "w/o MRC",
    "no_semantic_anchor": "w/o Semantic Anchor",
    "no_rcmi": "w/o RCMI",
    "all_plain": "All Plain",
}
T_CRITICAL_DF2_975 = 4.302652729696142


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "outputs" / "cosi_mag_final_benchmark",
    )
    parser.add_argument(
        "--single-root",
        type=Path,
        default=ROOT / "outputs" / "cosi_mag_final_ablation",
    )
    parser.add_argument(
        "--joint-root",
        type=Path,
        default=ROOT / "outputs" / "cosi_mag_final_joint_ablation",
    )
    return parser.parse_args()


def run_dir(args: argparse.Namespace, variant: str, dataset: str) -> Path:
    if variant == "full":
        return args.benchmark_root / "nc" / dataset / "runs_42_43_44"
    if variant == "all_plain":
        return args.joint_root / "all_plain" / "nc" / dataset / "runs_42_43_44"
    return args.single_root / variant / "nc" / dataset / "runs_42_43_44"


def values(args: argparse.Namespace, variant: str, dataset: str) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    directory = run_dir(args, variant, dataset)
    for run_index, seed in enumerate(SEEDS, 1):
        path = directory / f"best_run{run_index}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"seed mismatch: {path}")
        metrics = payload.get("metrics", {})
        output[seed] = {metric: float(metrics[metric]) for metric in METRICS}
    return output


def exact_sign_flip_pvalue(deltas: list[float]) -> float:
    observed = abs(float(np.mean(deltas)))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        permuted.append(abs(float(np.mean([sign * value for sign, value in zip(signs, deltas)]))))
    return sum(value >= observed - 1e-15 for value in permuted) / len(permuted)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    all_values: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []

    for variant in ("full", *VARIANTS):
        for dataset in DATASETS:
            by_seed = values(args, variant, dataset)
            all_values[(variant, dataset)] = by_seed
            for metric in METRICS:
                metric_values = [by_seed[seed][metric] for seed in SEEDS]
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "label": LABELS[variant],
                        "metric": metric,
                        "mean": float(np.mean(metric_values)),
                        "population_std": float(np.std(metric_values, ddof=0)),
                    }
                )

    for dataset in DATASETS:
        full = all_values[("full", dataset)]
        for variant in VARIANTS:
            current = all_values[(variant, dataset)]
            for metric in METRICS:
                deltas = []
                for seed in SEEDS:
                    delta = current[seed][metric] - full[seed][metric]
                    deltas.append(delta)
                    paired_rows.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "metric": metric,
                            "seed": seed,
                            "full": full[seed][metric],
                            "ablation": current[seed][metric],
                            "delta_ablation_minus_full": delta,
                        }
                    )
                mean = float(np.mean(deltas))
                sample_std = float(np.std(deltas, ddof=1))
                half_width = T_CRITICAL_DF2_975 * sample_std / math.sqrt(len(deltas))
                inference_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "metric": metric,
                        "paired_mean_delta": mean,
                        "paired_sample_std": sample_std,
                        "ci95_low_t_df2": mean - half_width,
                        "ci95_high_t_df2": mean + half_width,
                        "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(deltas),
                        "seeds_ablation_higher": sum(delta > 0 for delta in deltas),
                        "seeds_ablation_lower": sum(delta < 0 for delta in deltas),
                    }
                )

    output_dir = args.joint_root / "analysis"
    write_csv(output_dir / "nc_all_ablation_summary.csv", summary_rows)
    write_csv(output_dir / "nc_paired_seed_deltas.csv", paired_rows)
    write_csv(output_dir / "nc_paired_inference.csv", inference_rows)
    (output_dir / "summary_manifest.json").write_text(
        json.dumps(
            {
                "datasets": list(DATASETS),
                "seeds": list(SEEDS),
                "variants": list(VARIANTS),
                "paired_unit": "same training seed within a fixed dataset split",
                "uncertainty": "95% Student-t CI with df=2",
                "test": "exact two-sided sign-flip randomization test over 2^3 assignments",
                "minimum_attainable_two_sided_p_with_three_nonzero_pairs": 0.25,
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
