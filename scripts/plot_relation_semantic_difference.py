#!/usr/bin/env python3
"""Plot cross-modal relation-layer semantic rank differences from frozen CSV data.

The script reads the requested summary table and the complete Movies rank-plot
sample. It performs schema and finite-value checks before creating any output;
no plotting values are hard-coded.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SUMMARY = Path(
    "/hdd1/DataInHere/YHF/MoPF/outputs/e0_empirical_motivation/"
    "edge_semantic_discrepancy/edge_discrepancy_summary.csv"
)
DEFAULT_MOVIES_SAMPLE = Path(
    "/hdd1/DataInHere/YHF/MoPF/outputs/e0_empirical_motivation/"
    "edge_semantic_discrepancy/Movies/edge_rank_plot_sample.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_figures/relation_semantic_difference"

DATASET_ORDER = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]

TEXT_BLUE = "#2F6FB3"
VISUAL_GREEN = "#3B9960"
SHARED_GRAY = "#4C5560"
INK = "#20252B"
TEXT_HIGH_VISUAL_LOW = "#F2A65A"
VISUAL_HIGH_TEXT_LOW = "#D97B7B"
SUMMARY_BLUE = "#2F6FB3"
SUMMARY_ORANGE = "#C66A3D"

SUMMARY_FIELDS = {
    "dataset",
    "spearman_tv",
    "median_rank_gap",
}
SAMPLE_FIELDS = {"rank_text", "rank_visual"}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _as_float(value: str, *, field: str, row_number: int, path: Path) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"Empty value in {path}, row {row_number}, field {field}")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value in {path}, row {row_number}, field {field}: {value!r}"
        ) from exc
    if not np.isfinite(number):
        raise ValueError(f"Non-finite value in {path}, row {row_number}, field {field}")
    return number


def load_summary(path: Path) -> dict[str, dict[str, Any]]:
    fields, rows = _read_csv(path)
    missing = sorted(SUMMARY_FIELDS.difference(fields))
    if missing:
        raise ValueError(f"Summary file is missing required fields: {missing}")

    by_dataset: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        dataset = row["dataset"].strip()
        if dataset in by_dataset:
            raise ValueError(f"Duplicate dataset {dataset!r} in summary file row {row_number}")
        if dataset not in DATASET_ORDER:
            raise ValueError(
                f"Unexpected dataset {dataset!r}; expected exactly {DATASET_ORDER}"
            )
        by_dataset[dataset] = {
            "dataset": dataset,
            "spearman_tv": _as_float(
                row["spearman_tv"],
                field="spearman_tv",
                row_number=row_number,
                path=path,
            ),
            "median_rank_gap": _as_float(
                row["median_rank_gap"],
                field="median_rank_gap",
                row_number=row_number,
                path=path,
            ),
        }

    if set(by_dataset) != set(DATASET_ORDER):
        missing_datasets = sorted(set(DATASET_ORDER).difference(by_dataset))
        extra_datasets = sorted(set(by_dataset).difference(DATASET_ORDER))
        raise ValueError(
            f"Summary dataset mismatch; missing={missing_datasets}, extra={extra_datasets}"
        )
    return by_dataset


def load_movies_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    fields, rows = _read_csv(path)
    missing = sorted(SAMPLE_FIELDS.difference(fields))
    if missing:
        raise ValueError(f"Movies sample file is missing required fields: {missing}")
    if not rows:
        raise ValueError(f"Movies sample file contains no observations: {path}")

    rank_text = np.asarray(
        [
            _as_float(row["rank_text"], field="rank_text", row_number=i, path=path)
            for i, row in enumerate(rows, start=2)
        ],
        dtype=float,
    )
    rank_visual = np.asarray(
        [
            _as_float(row["rank_visual"], field="rank_visual", row_number=i, path=path)
            for i, row in enumerate(rows, start=2)
        ],
        dtype=float,
    )

    tolerance = 1e-9
    for name, values in (("rank_text", rank_text), ("rank_visual", rank_visual)):
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError(f"{name} values must lie in [0, 1]; observed outside range")
    return rank_text, rank_visual


def _configure_style() -> None:
    # Keep text editable in SVG/PDF and use a publication-scale sans-serif fallback.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )


def _add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    inside: bool = False,
    inside_y: float = 0.98,
) -> None:
    ax.text(
        0.02 if inside else -0.14,
        inside_y if inside else 1.03,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )


def build_figure(
    summary: dict[str, dict[str, Any]],
    rank_text: np.ndarray,
    rank_visual: np.ndarray,
    *,
    title: str,
    output_root: Path,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / "relation_semantic_difference"

    _configure_style()
    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True, facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.20], wspace=0.25)
    ax_rank = fig.add_subplot(outer[0, 0])
    right_grid = outer[0, 1].subgridspec(2, 1, hspace=0.24)
    ax_rho = fig.add_subplot(right_grid[0, 0])
    ax_gap = fig.add_subplot(right_grid[1, 0], sharey=ax_rho)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.11, color=INK)
    fig.text(
        0.5,
        1.035,
        "Low cross-modal rank agreement + non-trivial edge-wise rank gaps",
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="semibold",
        color=INK,
    )

    # Left: all rows from the supplied Movies sample file.
    opposite_regions = [
        ((0.0, 0.75), VISUAL_HIGH_TEXT_LOW),
        ((0.75, 0.0), TEXT_HIGH_VISUAL_LOW),
    ]
    density = ax_rank.hexbin(
        rank_text,
        rank_visual,
        gridsize=55,
        extent=(0.0, 1.0, 0.0, 1.0),
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    density.set_rasterized(True)
    for corner, region_color in opposite_regions:
        ax_rank.add_patch(
            Rectangle(
                corner,
                0.25,
                0.25,
                facecolor=region_color,
                edgecolor="none",
                linewidth=0,
                alpha=0.24,
                zorder=2,
            )
        )
        ax_rank.add_patch(
            Rectangle(
                corner,
                0.25,
                0.25,
                facecolor="none",
                edgecolor=SHARED_GRAY,
                linewidth=1.05,
                linestyle=(0, (3, 2)),
                zorder=4,
            )
        )
    colorbar = fig.colorbar(density, ax=ax_rank, fraction=0.046, pad=0.035)
    colorbar.set_label("Edges per hexbin (log count)", fontsize=7.6, color=INK)
    colorbar.ax.tick_params(labelsize=7.8, colors=INK)

    ax_rank.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=SHARED_GRAY,
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        zorder=5,
    )
    ax_rank.text(
        0.54,
        0.84,
        "Cross-modal rank agreement",
        transform=ax_rank.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=SHARED_GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
        zorder=6,
    )
    ax_rank.annotate(
        "",
        xy=(0.54, 0.54),
        xytext=(0.54, 0.79),
        arrowprops={"arrowstyle": "-|>", "color": SHARED_GRAY, "lw": 0.75},
        zorder=6,
    )
    ax_rank.annotate(
        "",
        xy=(0.85, 0.20),
        xytext=(0.85, 0.23),
        arrowprops={"arrowstyle": "-|>", "color": "#9C4E22", "lw": 0.85},
        zorder=7,
    )
    ax_rank.text(
        0.95,
        0.055,
        "modality-\nspecific\nrelation",
        ha="right",
        va="bottom",
        fontsize=5.4,
        color="#9C4E22",
        zorder=7,
    )
    ax_rank.annotate(
        "",
        xy=(0.18, 0.79),
        xytext=(0.18, 0.82),
        arrowprops={"arrowstyle": "-|>", "color": "#994B4B", "lw": 0.85},
        zorder=7,
    )
    ax_rank.text(
        0.04,
        0.84,
        "modality-\nspecific\nrelation",
        ha="left",
        va="bottom",
        fontsize=5.4,
        color="#994B4B",
        zorder=7,
    )
    ax_rank.set_xlim(0.0, 1.0)
    ax_rank.set_ylim(0.0, 1.0)
    ax_rank.set_aspect("equal", adjustable="box")
    ax_rank.set_xlabel("Text edge-similarity rank", color=TEXT_BLUE)
    ax_rank.set_ylabel("Visual edge-similarity rank", color=VISUAL_GREEN)
    ax_rank.set_xticks(np.linspace(0.0, 1.0, 5))
    ax_rank.set_yticks(np.linspace(0.0, 1.0, 5))
    ax_rank.grid(False)
    ax_rank.set_title("Movies: representative edge-rank relationship", loc="left", pad=4)
    _add_panel_label(ax_rank, "a")

    # Right: aligned summary dot plots, in the exact requested dataset order.
    datasets = DATASET_ORDER
    y = np.arange(len(datasets), dtype=float)
    rho = np.asarray([summary[d]["spearman_tv"] for d in datasets], dtype=float)
    median_gap = np.asarray([summary[d]["median_rank_gap"] for d in datasets], dtype=float)

    rho_span = max(float(np.max(rho) - np.min(rho)), 0.1)
    rho_label_pad = 0.035 * rho_span
    rho_limits = (
        float(np.min(rho)) - 0.12 * rho_span,
        float(np.max(rho)) + 0.20 * rho_span,
    )
    ax_rho.scatter(
        rho,
        y,
        s=42,
        color=SUMMARY_BLUE,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    for y_value, value in zip(y, rho):
        ax_rho.text(
            value + rho_label_pad,
            y_value,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=INK,
            zorder=4,
        )
    ax_rho.axvline(0.0, color=SHARED_GRAY, linestyle=(0, (3, 2)), linewidth=0.8)
    ax_rho.set_xlim(*rho_limits)
    ax_rho.set_ylabel("Dataset")
    ax_rho.set_xlabel("Spearman ρ")
    ax_rho.set_yticks(y)
    ax_rho.set_yticklabels(datasets)
    ax_rho.invert_yaxis()
    ax_rho.grid(False)
    ax_rho.set_title("Spearman ρ", loc="left", pad=4)
    _add_panel_label(ax_rho, "b", inside=True, inside_y=0.88)

    gap_high = float(np.max(median_gap))
    gap_span = max(gap_high - float(np.min(median_gap)), 0.1)
    gap_label_pad = 0.035 * gap_span
    gap_limits = (
        max(0.0, float(np.min(median_gap)) - 0.12 * gap_span),
        gap_high + 0.24 * gap_span,
    )
    ax_gap.scatter(
        median_gap,
        y,
        s=42,
        color=SUMMARY_ORANGE,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    for y_value, value in zip(y, median_gap):
        ax_gap.text(
            value + gap_label_pad,
            y_value,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=INK,
            zorder=4,
        )
    ax_gap.set_xlim(*gap_limits)
    ax_gap.set_xlabel("Median |Text rank − Visual rank|")
    ax_gap.set_yticks(y)
    ax_gap.tick_params(axis="y", labelleft=False)
    ax_gap.grid(False)
    ax_gap.set_title("Median |Text rank − Visual rank|", loc="left", pad=4)
    _add_panel_label(ax_gap, "c", inside=True, inside_y=0.82)

    # Import the skill's render-time alignment gate without making the script
    # depend on a private installation path. Set PYTHONPATH to the skill's
    # scripts directory when running this file.
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import nature-figure audit_panel_alignment.py; add the skill's "
            "scripts directory to PYTHONPATH before running this script."
        ) from exc

    require_matplotlib_panel_alignment(
        fig,
        axes=[ax_rank, ax_rho, ax_gap],
        panel_ids=["a", "b", "c"],
        column_groups=[["b", "c"]],
        json_out=base.with_suffix(".alignment.json"),
        overlay_svg=base.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )

    outputs = [
        base.with_suffix(".png"),
        base.with_suffix(".pdf"),
        base.with_suffix(".svg"),
        base.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[3], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--movies-sample", type=Path, default=DEFAULT_MOVIES_SAMPLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--title",
        default="Cross-modal semantic rank disagreement across graph edges",
        help="Figure title; values and plotted data are still read only from the input CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    rank_text, rank_visual = load_movies_sample(args.movies_sample)
    outputs = build_figure(
        summary,
        rank_text,
        rank_visual,
        title=args.title,
        output_root=args.output_root,
    )
    print(f"Read summary rows: {len(summary)}")
    print(f"Read Movies sample rows: {rank_text.size}")
    print(f"Wrote: {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
