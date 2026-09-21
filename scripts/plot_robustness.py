#!/usr/bin/env python3
"""Render Figure 5 from completed checkpoint-only robustness summaries.

This script consumes ``robustness_summary.csv`` written by
``scripts/evaluate_robustness.py``. It does not run inference or perturbation
generation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# Figure contract: retention under random false relations and raw-semantic
# cross-modal conflict; each row answers a distinct stress-test question and
# each dataset column checks the same attack in a fixed representative dataset.
DATASETS = ("Movies", "Grocery")
PERTURBATIONS = (
    ("random_structural_noise", "Random Structural Noise"),
    ("semantic_conflict_edge_injection", "Semantic-Conflict Injection"),
)
MODEL_ORDER = ("mopf", "wo_relation_calibration", "wo_semantic_anchor", "dip")
MODEL_LABELS = {
    "mopf": "Full CoSI-MAG",
    "wo_relation_calibration": "wo_relation_calibration",
    "wo_semantic_anchor": "wo_semantic_anchor",
    "dip": "DiP",
}
MODEL_COLORS = {
    "mopf": "#0F4D92",
    "wo_relation_calibration": "#B64342",
    "wo_semantic_anchor": "#42949E",
    "dip": "#767676",
}
MODEL_MARKERS = {"mopf": "o", "wo_relation_calibration": "s", "wo_semantic_anchor": "^", "dip": "D"}
PANEL_LABELS = (("(a)", "(b)"), ("(c)", "(d)"))


def read_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        if row.get("metric") != "accuracy":
            continue
        parsed.append(
            {
                **row,
                "rho": float(row["rho"]),
                "mean": float(row["mean_across_model_seeds"]),
                "sd": float(row["population_sd_across_model_seeds"]),
            }
        )
    if not parsed:
        raise ValueError(f"No Accuracy-retention rows in {path}")
    expected = {
        (dataset, perturbation, model)
        for dataset in DATASETS
        for perturbation, _ in PERTURBATIONS
        for model in MODEL_ORDER
        if not (perturbation == "semantic_conflict_edge_injection" and model == "wo_semantic_anchor")
    }
    observed = {(r["dataset"], r["perturbation_type"], r["model"]) for r in parsed}
    missing = expected - observed
    if missing:
        raise ValueError(f"Summary is missing expected Figure 5 groups: {sorted(missing)}")
    return parsed


def _subset(
    rows: list[dict[str, Any]], dataset: str, perturbation: str, model: str
) -> list[dict[str, Any]]:
    values = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["perturbation_type"] == perturbation
        and row["model"] == model
    ]
    return sorted(values, key=lambda row: row["rho"])


def build_figure(summary_path: Path, output_root: Path) -> tuple[plt.Figure, list[Path]]:
    """Build an aligned 2×2 quantitative grid of Accuracy-retention curves."""
    rows = read_summary(summary_path)
    output_root.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.0), sharey=True, constrained_layout=False)
    fig.patch.set_facecolor("white")
    for row_idx, (perturbation, perturbation_label) in enumerate(PERTURBATIONS):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")
            ax.axhline(100.0, color="#8A8A8A", linewidth=0.9, linestyle="--", zorder=0)
            model_order = MODEL_ORDER if row_idx == 0 else tuple(
                model for model in MODEL_ORDER if model != "wo_semantic_anchor"
            )
            for model in model_order:
                series = _subset(rows, dataset, perturbation, model)
                if not series:
                    continue
                x = np.asarray([item["rho"] for item in series], dtype=float)
                mean = np.asarray([item["mean"] for item in series], dtype=float)
                sd = np.asarray([item["sd"] for item in series], dtype=float)
                ax.errorbar(
                    x,
                    mean,
                    yerr=sd,
                    color=MODEL_COLORS[model],
                    marker=MODEL_MARKERS[model],
                    markersize=3.8,
                    linewidth=1.5,
                    elinewidth=0.9,
                    capsize=2.0,
                    label=MODEL_LABELS[model],
                    zorder=3 if model == "mopf" else 2,
                )

            series_rhos = sorted(
                {
                    float(item["rho"])
                    for item in rows
                    if item["dataset"] == dataset and item["perturbation_type"] == perturbation
                }
            )
            ax.set_xticks(series_rhos)
            ax.set_xticklabels([f"{100.0 * rho:g}%" for rho in series_rhos])
            ax.set_xlim(-0.012, max(series_rhos) + 0.018)
            ax.set_title(
                f"{PANEL_LABELS[row_idx][col_idx]} {dataset}\n{perturbation_label}",
                loc="left",
                fontsize=8.5,
                fontweight="bold",
                pad=6,
            )
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=7, width=0.7, length=3)
            ax.set_xlabel("Injected physical edges (% of clean support)", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Test Accuracy retention (%)", fontsize=8)

    fig.legend(
        handles=[
            Line2D(
                [0], [0],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                linewidth=1.5,
                markersize=4,
                label=MODEL_LABELS[model],
            )
            for model in MODEL_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=7,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.13, top=0.83, wspace=0.22, hspace=0.36)

    base = output_root / "figure5_robustness"
    # Resolve the active figure-alignment helper from the local user skill
    # installation, without embedding a machine-specific absolute path.
    alignment_module_path = Path.home() / ".codex/skills/nature-figure/scripts"
    if alignment_module_path.is_dir() and str(alignment_module_path) not in sys.path:
        sys.path.insert(0, str(alignment_module_path))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        json_out=base.with_name(base.name + "_alignment.json"),
        overlay_svg=base.with_name(base.name + "_alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        # Parenthesized labels are conventional in the rendered manuscript;
        # the helper's automatic label detector recognizes bare letters only.
        require_panel_labels=False,
        strict=True,
    )
    exports = [base.with_suffix(".pdf"), base.with_suffix(".svg"), base.with_suffix(".png")]
    fig.savefig(exports[0], bbox_inches="tight")
    fig.savefig(exports[1], bbox_inches="tight")
    fig.savefig(exports[2], dpi=600, bbox_inches="tight")
    return fig, exports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/robustness_analysis/robustness_summary.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/robustness_analysis"))
    args = parser.parse_args()
    summary = args.summary if args.summary.is_absolute() else Path.cwd() / args.summary
    output_root = args.output_root if args.output_root.is_absolute() else Path.cwd() / args.output_root
    if not summary.is_file():
        raise FileNotFoundError(
            f"Formal robustness summary is not present: {summary}. "
            "Run scripts/evaluate_robustness.py --execute only when the formal evaluation is authorized."
        )
    _, exports = build_figure(summary, output_root)
    print("Exported Figure 5:")
    for path in exports:
        print(path)


if __name__ == "__main__":
    main()
