#!/usr/bin/env python3
"""Plot independent candidates for Figure 3(b) V3.

The script intentionally does not touch the existing Figure 3 V2 outputs.  It
exports two standalone candidates and a combined B3 candidate:

* B1: node-level cross-modal total-variation distributions;
* B2: percentage of nodes whose top neighbor differs across modalities;
* B3: B1+B2 in one compact candidate, with the B2 percentage annotated over
  each B1 distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "figure3_mechanism_v3_candidate"
DATASETS = ("Movies", "Grocery")
DATASET_COLORS = {"Movies": "#4C78A8", "Grocery": "#59A14F"}
DATASET_LIGHT = {"Movies": "#AFCBE8", "Grocery": "#B6DCA9"}
INK = "#24272A"
NEUTRAL = "#6F7478"
GRID = "#E7E9EB"
WHITE = "#FFFFFF"
FIGURE_WIDTH_MM = 183.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {key}: {value}")
    return value


def _configure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.size": 7.2,
            "axes.titlesize": 8.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.2,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "lines.solid_capstyle": "round",
        }
    )


def _format_axis(ax: Any, grid_axis: str = "y") -> None:
    ax.tick_params(length=2.4, width=0.6, pad=2.4)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, linestyle=(0, (1.2, 2.0)), alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.08,
        1.045,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def _save_formats(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _alignment_gate(fig: Any, axes: list[Any], output_dir: Path, stem: str) -> None:
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=output_dir / f"{stem}.alignment.json",
        overlay_svg=output_dir / f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
        axes=axes,
    )


def _records(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[row["dataset"]].append(_float(row, "tv_distance"))
    return {dataset: np.asarray(values.get(dataset, []), dtype=float) for dataset in DATASETS}


def _summary_values(summary_rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float]]:
    pooled: dict[str, float] = {}
    seed_values: dict[str, list[float]] = defaultdict(list)
    for row in summary_rows:
        if row.get("aggregation_level") == "cross_seed_pooled":
            pooled[row["dataset"]] = _float(row, "top_neighbor_disagreement_percentage")
        elif row.get("aggregation_level") == "seed":
            seed_values[row["dataset"]].append(_float(row, "top_neighbor_disagreement_percentage"))
    seed_sd = {dataset: float(np.std(seed_values.get(dataset, []))) if seed_values.get(dataset) else 0.0 for dataset in DATASETS}
    return pooled, seed_sd


def _add_empty_message(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=NEUTRAL, fontsize=7.5)
    ax.set_axis_off()


def _draw_tv(ax: Any, records: dict[str, np.ndarray], *, top_disagreement: dict[str, float] | None = None, panel_label: str | None = None) -> None:
    positions = np.arange(len(DATASETS), dtype=float)
    distributions = [records[dataset] for dataset in DATASETS]
    if not any(values.size for values in distributions):
        _add_empty_message(ax, "No allocation data available")
        return

    for position, dataset, values in zip(positions, DATASETS, distributions):
        if not values.size:
            continue
        violin = ax.violinplot(
            [values], positions=[position], widths=0.68, showmeans=False, showmedians=False, showextrema=False
        )
        body = violin["bodies"][0]
        body.set_facecolor(DATASET_LIGHT[dataset])
        body.set_edgecolor(DATASET_COLORS[dataset])
        body.set_linewidth(0.65)
        body.set_alpha(0.88)
        box = ax.boxplot(
            [values],
            positions=[position],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": INK, "linewidth": 1.0},
            whiskerprops={"color": INK, "linewidth": 0.65},
            capprops={"color": INK, "linewidth": 0.65},
            boxprops={"facecolor": WHITE, "edgecolor": INK, "linewidth": 0.65},
        )
        box["boxes"][0].set_alpha(0.96)
        median = float(np.median(values))
        ax.scatter([position], [median], color=DATASET_COLORS[dataset], edgecolor=WHITE, linewidth=0.45, s=14, zorder=5)

    maximum = max(float(np.max(values)) for values in distributions if values.size)
    annotation_extra = 0.0
    if top_disagreement:
        annotation_extra = max(0.018, maximum * 0.24)
    upper = max(0.05, maximum * 1.18 + annotation_extra)
    ax.set_ylim(0.0, upper)
    ax.set_xticks(positions, labels=DATASETS)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylabel("Cross-modal neighbor-allocation divergence (TV)", labelpad=4)
    ax.set_xlabel("Dataset", labelpad=4)
    ax.set_title("Modality-specific neighbor allocation", loc="left", pad=7, fontweight="bold")
    if top_disagreement:
        annotation_y = maximum * 1.10
        for position, dataset in zip(positions, DATASETS):
            if dataset in top_disagreement:
                ax.text(
                    position,
                    annotation_y,
                    f"Top-1 differs: {top_disagreement[dataset]:.1f}%",
                    ha="center",
                    va="top",
                    fontsize=6.4,
                    color=INK,
                )
    _format_axis(ax)
    if panel_label:
        _add_panel_label(ax, panel_label)


def _draw_disagreement(ax: Any, pooled: dict[str, float], seed_sd: dict[str, float], *, panel_label: str | None = None) -> None:
    positions = np.arange(len(DATASETS), dtype=float)
    heights = np.asarray([pooled.get(dataset, np.nan) for dataset in DATASETS], dtype=float)
    errors = np.asarray([seed_sd.get(dataset, 0.0) for dataset in DATASETS], dtype=float)
    mask = np.isfinite(heights)
    if not mask.any():
        _add_empty_message(ax, "No allocation summary available")
        return
    colors = [DATASET_COLORS[dataset] for dataset in DATASETS]
    bars = ax.bar(
        positions[mask],
        heights[mask],
        width=0.56,
        color=np.asarray(colors, dtype=object)[mask],
        edgecolor=INK,
        linewidth=0.45,
        yerr=errors[mask],
        error_kw={"elinewidth": 0.65, "capsize": 1.8, "capthick": 0.65},
        zorder=3,
    )
    for bar, value in zip(bars, heights[mask]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(103.0, float(value) + 2.2),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=INK,
        )
    ax.set_ylim(0.0, 110.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(positions, labels=DATASETS)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_ylabel("Nodes with different top neighbor (%)", labelpad=4)
    ax.set_xlabel("Dataset", labelpad=4)
    ax.set_title("Top-neighbor disagreement", loc="left", pad=7, fontweight="bold")
    _format_axis(ax)
    if panel_label:
        _add_panel_label(ax, panel_label)


def _plot_b1(records: dict[str, np.ndarray], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(95.0 / 25.4, 68.0 / 25.4), constrained_layout=False)
    _draw_tv(ax, records, panel_label="b1")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.19, top=0.86)
    _save_formats(fig, output_dir, "figure3b1_tv_distance")


def _plot_b2(pooled: dict[str, float], seed_sd: dict[str, float], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(95.0 / 25.4, 68.0 / 25.4), constrained_layout=False)
    _draw_disagreement(ax, pooled, seed_sd, panel_label="b2")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.19, top=0.86)
    _save_formats(fig, output_dir, "figure3b2_top_neighbor_disagreement")


def _plot_b3(records: dict[str, np.ndarray], pooled: dict[str, float], seed_sd: dict[str, float], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_MM / 25.4, 70.0 / 25.4), constrained_layout=False)
    _draw_tv(axes[0], records, top_disagreement=pooled, panel_label="b")
    _draw_disagreement(axes[1], pooled, seed_sd)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.83, wspace=0.34)
    _alignment_gate(fig, list(axes), output_dir, "figure3b3_neighbor_allocation")
    _save_formats(fig, output_dir, "figure3b3_neighbor_allocation")


def main() -> int:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir).resolve()
    detail_rows = _read_csv(data_dir / "panel_b_neighbor_allocation.csv")
    summary_rows = _read_csv(data_dir / "panel_b_neighbor_allocation_summary.csv")
    records = _records(detail_rows)
    pooled, seed_sd = _summary_values(summary_rows)

    _configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_b1(records, output_dir)
    _plot_b2(pooled, seed_sd, output_dir)
    _plot_b3(records, pooled, seed_sd, output_dir)

    manifest = {
        "script": str(Path(__file__).resolve().relative_to(ROOT.resolve())),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "candidate_plots": {
            "B1": "figure3b1_tv_distance",
            "B2": "figure3b2_top_neighbor_disagreement",
            "B3": "figure3b3_neighbor_allocation",
        },
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "pooled_top_neighbor_disagreement_percentage": pooled,
        "seed_sd_top_neighbor_disagreement_percentage": seed_sd,
        "no_existing_figure3_v2_outputs_modified": True,
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[figure3-v3-plot] wrote B1/B2/B3 candidates to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
