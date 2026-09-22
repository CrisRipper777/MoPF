#!/usr/bin/env python3
"""Render a compact camera-ready Figure 3 from existing V2/V3 CSV outputs.

This is a plotting-only refinement.  It reuses the existing Figure 3 source
data and definitions, and does not load checkpoints, train models, or modify
the previous Figure 3 outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2_DEFAULT = ROOT / "outputs" / "figure3_mechanism_v2"
V3_DEFAULT = ROOT / "outputs" / "figure3_mechanism_v3_candidate"
OUTPUT_DEFAULT = ROOT / "outputs" / "figure3_final_camera_ready"

DATASETS = ("Movies", "Grocery")
MODALITIES = ("text", "visual")
HOPS = (0, 1, 2, 3)

# Dataset colors are intentionally separate from the paper-wide modality
# language.  These colors are reused in Panels (a) and (b) only.
MOVIES = "#7563A6"
MOVIES_LIGHT = "#C9BDE2"
GROCERY = "#D28A3A"
GROCERY_LIGHT = "#F1C58D"

# Modality colors are fixed across Panel (c).
TEXT_DARK = "#4C78A8"
TEXT_LIGHT = "#86A9CF"
VISUAL_DARK = "#59A14F"
VISUAL_LIGHT = "#91C589"

INK = "#24272A"
NEUTRAL = "#6F7478"
GRID = "#E7E9EB"
WHITE = "#FFFFFF"

FIGURE_WIDTH_IN = 7.2
FIGURE_HEIGHT_IN = 5.2
FIGURE_WIDTH_MM = 182.88
PANEL_A_WIDTH_IN = 4.50
PANEL_B_WIDTH_IN = 6.85
PANEL_C_WIDTH_IN = 6.85
PANEL_HEIGHT_IN = 2.55


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-data-dir", type=Path, default=V2_DEFAULT)
    parser.add_argument("--v3-data-dir", type=Path, default=V3_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


def _load_base_module() -> Any:
    """Reuse the existing final-figure CSV readers and aggregation helpers."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import plot_figure3_final as base

    return base


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.3,
            "axes.titlesize": 8.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.1,
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
    ax.tick_params(axis="x", pad=4.0)
    ax.grid(
        axis=grid_axis,
        color=GRID,
        linewidth=0.4,
        linestyle=(0, (1.2, 2.0)),
        alpha=0.9,
        zorder=0,
    )
    ax.set_axisbelow(True)


def _save_formats(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the declared camera-ready canvas dimensions; do not crop the
    # page with bbox_inches="tight".
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def _add_panel_a_header(fig: Any, ax: Any, *, title_size: float = 8.8, key_offset: float = 0.075) -> None:
    """Place Panel (a)'s title and dataset labels outside the data rectangle."""
    fig.canvas.draw()
    pos = ax.get_position()
    fig.text(
        pos.x0,
        pos.y1 + 0.040,
        "(a) Relation response to semantic discrepancy",
        ha="left",
        va="bottom",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
    )
    fig.text(pos.x0 + 0.020, pos.y1 + key_offset, "Movies", ha="left", va="bottom", fontsize=7.0, fontweight="bold", color=MOVIES)
    fig.text(pos.x0 + 0.105, pos.y1 + key_offset, "Grocery", ha="left", va="bottom", fontsize=7.0, fontweight="bold", color=GROCERY)


def _alignment_gate(
    fig: Any,
    axes: list[Any],
    output_dir: Path,
    stem: str,
    *,
    row_groups: list[list[str]] | None = None,
) -> None:
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    fig.canvas.draw()
    options: dict[str, Any] = {
        "json_out": output_dir / f"{stem}.alignment.json",
        "overlay_svg": output_dir / f"{stem}.alignment.svg",
        "tolerance_pt": 1.5,
        "gutter_tolerance_pt": 1.5,
        "require_panel_labels": False,
        "strict": True,
        "axes": axes,
    }
    if row_groups is not None:
        options["row_groups"] = row_groups
    require_matplotlib_panel_alignment(fig, **options)


def _read_inputs(base: Any, v2_dir: Path, v3_dir: Path) -> dict[str, list[dict[str, str]]]:
    inputs = {
        "a_gap": base._read_csv(v2_dir / "panel_a_gap_response.csv"),
        "c_retention": base._read_csv(v2_dir / "panel_c_retention_counterfactual.csv"),
        "b_detail": base._read_csv(v3_dir / "panel_b_neighbor_allocation.csv"),
        "b_summary": base._read_csv(v3_dir / "panel_b_neighbor_allocation_summary.csv"),
    }
    missing = [name for name, rows in inputs.items() if not rows]
    if missing:
        raise FileNotFoundError("missing required camera-ready inputs: " + ", ".join(missing))
    return inputs


def _draw_panel_a(
    ax: Any,
    base: Any,
    rows: list[dict[str, str]],
    *,
    title: bool = True,
    title_size: float = 8.8,
) -> None:
    points = base._gap_points(rows)
    colors = {"Movies": MOVIES, "Grocery": GROCERY}
    all_x = [item[0] for dataset in DATASETS for item in points.get(dataset, [])]
    all_y = [item[1] for dataset in DATASETS for item in points.get(dataset, [])]
    if not all_x:
        ax.text(0.5, 0.5, "No Panel (a) data", transform=ax.transAxes, ha="center", va="center", color=NEUTRAL)
        ax.set_axis_off()
        return

    for dataset in DATASETS:
        values = points.get(dataset, [])
        if not values:
            continue
        x = np.asarray([item[0] for item in values], dtype=float)
        y = np.asarray([item[1] for item in values], dtype=float)
        error = np.asarray([item[2] for item in values], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=error,
            color=colors[dataset],
            marker="o",
            markersize=3.0,
            linewidth=1.25,
            capsize=1.5,
            capthick=0.6,
            label=dataset,
            zorder=3,
        )

    y_min, y_max = min(all_y), max(all_y)
    y_pad = max(0.01, 0.10 * (y_max - y_min))
    ax.set_xlim(min(all_x) - 0.06, max(all_x) + 0.06)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(0.0, color=NEUTRAL, linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlabel("Text–Visual semantic gap", labelpad=3)
    ax.set_ylabel("Text–Visual relation response", labelpad=3)
    if title:
        ax.set_title("(a) Relation response to semantic discrepancy", loc="left", pad=5, fontsize=title_size)
    _format_axis(ax)


def _draw_panel_b_left(
    ax: Any,
    base: Any,
    rows: list[dict[str, str]],
    *,
    show_title: bool = True,
    show_median: bool = True,
    median_fontsize: float = 6.3,
) -> None:
    records = base._allocation_records(rows)
    palette = {
        "Movies": (MOVIES, MOVIES_LIGHT),
        "Grocery": (GROCERY, GROCERY_LIGHT),
    }
    positions = np.arange(len(DATASETS), dtype=float)
    for position, dataset in zip(positions, DATASETS):
        values = records.get(dataset, np.asarray([], dtype=float))
        if not values.size:
            continue
        violin = ax.violinplot(
            [values],
            positions=[position],
            widths=0.68,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = violin["bodies"][0]
        body.set_facecolor(palette[dataset][1])
        body.set_edgecolor(palette[dataset][0])
        body.set_linewidth(0.7)
        body.set_alpha(0.96)
        box = ax.boxplot(
            [values],
            positions=[position],
            widths=0.17,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": INK, "linewidth": 0.9},
            whiskerprops={"color": INK, "linewidth": 0.6},
            capprops={"color": INK, "linewidth": 0.6},
            boxprops={"facecolor": WHITE, "edgecolor": INK, "linewidth": 0.6},
        )
        box["boxes"][0].set_alpha(0.98)
        median = float(np.median(values))
        ax.scatter([position], [median], color=palette[dataset][0], edgecolor=WHITE, linewidth=0.4, s=15, zorder=5)
        if show_median:
            ax.text(
                -0.48 if dataset == "Movies" else 1.78,
                0.116,
                f"Median=.{int(round(median * 1000)):03d}",
                ha="left" if dataset == "Movies" else "right",
                va="top",
                fontsize=median_fontsize,
                color=INK,
            )
    ax.set_ylim(0.0, 0.12)
    ax.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax.set_xticks(positions, labels=DATASETS)
    ax.set_xlim(-0.65, len(DATASETS) - 0.05)
    ax.set_xlabel("Dataset", labelpad=3)
    ax.set_ylabel("TV divergence", labelpad=3)
    if show_title:
        ax.set_title("TV divergence", loc="left", pad=4, fontsize=7.5)
    _format_axis(ax)


def _draw_panel_b_right(ax: Any, base: Any, rows: list[dict[str, str]], *, show_title: bool = True) -> None:
    pooled, _seed_sd = base._allocation_summary(rows)
    positions = np.arange(len(DATASETS), dtype=float)
    colors = [MOVIES, GROCERY]
    heights = np.asarray([pooled.get(dataset, np.nan) for dataset in DATASETS], dtype=float)
    mask = np.isfinite(heights)
    bars = ax.bar(
        positions[mask],
        heights[mask],
        width=0.56,
        color=np.asarray(colors, dtype=object)[mask],
        edgecolor=INK,
        linewidth=0.5,
        zorder=3,
    )
    for bar, value in zip(bars, heights[mask]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            82.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=INK,
        )
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(positions, labels=DATASETS)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xlabel("Dataset", labelpad=3)
    ax.set_ylabel("Top-neighbor disagreement (%)", labelpad=1, fontsize=7.2)
    if show_title:
        ax.set_title("Top-1 differs", loc="left", pad=4, fontsize=7.5)
    _format_axis(ax)


def _draw_panel_b(
    fig: Any,
    axes: list[Any],
    base: Any,
    detail_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    *,
    group_title: bool,
    show_median: bool = True,
    median_fontsize: float = 6.3,
) -> None:
    _draw_panel_b_left(axes[0], base, detail_rows, show_median=show_median, median_fontsize=median_fontsize)
    _draw_panel_b_right(axes[1], base, summary_rows)
    if group_title:
        fig.text(
            0.08,
            0.965,
            "(b) Modality-specific neighbor allocation",
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="bold",
        )


def _retention_handles() -> list[Any]:
    return [
        Line2D([0], [0], color=TEXT_DARK, linestyle="-", marker="o", linewidth=1.25, markersize=2.8, label="Text anchored"),
        Line2D([0], [0], color=TEXT_LIGHT, linestyle="--", marker="o", linewidth=1.15, markersize=2.8, label="Text anchor-off"),
        Line2D([0], [0], color=VISUAL_DARK, linestyle="-", marker="o", linewidth=1.25, markersize=2.8, label="Visual anchored"),
        Line2D([0], [0], color=VISUAL_LIGHT, linestyle="--", marker="o", linewidth=1.15, markersize=2.8, label="Visual anchor-off"),
    ]


def _draw_panel_c(
    axes: list[Any],
    base: Any,
    rows: list[dict[str, str]],
    *,
    show_facet_titles: bool = True,
    show_gain: bool = True,
) -> list[Any]:
    summary = base._retention_summary(rows)
    colors = {
        ("text", "anchored"): TEXT_DARK,
        ("text", "anchor_off_counterfactual"): TEXT_LIGHT,
        ("visual", "anchored"): VISUAL_DARK,
        ("visual", "anchor_off_counterfactual"): VISUAL_LIGHT,
    }
    for axis, dataset in zip(axes, DATASETS):
        for modality in MODALITIES:
            for condition, linestyle, alpha, linewidth in (
                ("anchored", "-", 1.0, 1.3),
                ("anchor_off_counterfactual", "--", 0.95, 1.15),
            ):
                means: list[float] = []
                spreads: list[float] = []
                for hop in HOPS:
                    mean, spread, count = summary.get((dataset, condition, modality, hop), (np.nan, np.nan, 0))
                    means.append(100.0 * mean if count else np.nan)
                    spreads.append(100.0 * spread if count else np.nan)
                mean_array = np.asarray(means, dtype=float)
                spread_array = np.asarray(spreads, dtype=float)
                mask = np.isfinite(mean_array)
                if not mask.any():
                    continue
                x = np.asarray(HOPS, dtype=float)[mask]
                y = mean_array[mask]
                spread = spread_array[mask]
                color = colors[(modality, condition)]
                axis.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markersize=2.8,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=3,
                )
                axis.fill_between(x, y - spread, y + spread, color=color, alpha=0.08 if condition == "anchored" else 0.045, linewidth=0, zorder=1)
        axis.set_xticks(HOPS, labels=[str(hop) for hop in HOPS])
        axis.set_xlim(-0.05, 3.05)
        axis.set_ylim(80.0, 100.0)
        axis.set_yticks([80, 85, 90, 95, 100])
        axis.set_xlabel("Hop order k", labelpad=3)
        if show_facet_titles:
            axis.set_title(dataset, loc="left", pad=4, fontsize=7.5)
        if show_gain:
            gain_values: list[str] = []
            for modality, short in (("text", "T"), ("visual", "V")):
                anchored = summary.get((dataset, "anchored", modality, 3), (np.nan, np.nan, 0))[0]
                off = summary.get((dataset, "anchor_off_counterfactual", modality, 3), (np.nan, np.nan, 0))[0]
                if np.isfinite(anchored) and np.isfinite(off):
                    gain_values.append(f"{short} {100.0 * (anchored - off):+.1f} pp")
            if gain_values:
                axis.text(
                    0.98,
                    0.965,
                    "Hop-3 gain: " + ", ".join(gain_values),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.4,
                    color=NEUTRAL,
                )
        _format_axis(axis)
    axes[0].set_ylabel("Semantic retention (%)", labelpad=3)
    axes[-1].tick_params(labelleft=False)
    return _retention_handles()


def _write_manifest(output_dir: Path, v2_dir: Path, v3_dir: Path, inputs: dict[str, list[dict[str, str]]]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/plot_figure3_camera_ready.py",
        "figure_size_inches": [FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN],
        "layout": "two rows: (a) left + (b) right; (c) full-width with Movies/Grocery facets",
        "source_data": {
            "v2": str(v2_dir),
            "v3": str(v3_dir),
            "panel_a_gap_rows": len(inputs["a_gap"]),
            "panel_b_detail_rows": len(inputs["b_detail"]),
            "panel_b_summary_rows": len(inputs["b_summary"]),
            "panel_c_retention_rows": len(inputs["c_retention"]),
        },
        "scientific_definitions_unchanged": True,
        "training_started": False,
        "protected_outputs_untouched": [
            "outputs/figure3_mechanism_v2",
            "outputs/figure3_mechanism_v3_candidate",
            "outputs/figure3_final",
        ],
        "colors": {
            "movies": MOVIES,
            "grocery": GROCERY,
            "text_anchored": TEXT_DARK,
            "text_anchor_off": TEXT_LIGHT,
            "visual_anchored": VISUAL_DARK,
            "visual_anchor_off": VISUAL_LIGHT,
        },
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    v2_dir = args.v2_data_dir.resolve()
    v3_dir = args.v3_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _load_base_module()
    inputs = _read_inputs(base, v2_dir, v3_dir)
    _configure_style()

    # Standalone panel (a).
    fig_a, ax_a = plt.subplots(figsize=(PANEL_A_WIDTH_IN, PANEL_HEIGHT_IN))
    fig_a.subplots_adjust(left=0.19, right=0.98, bottom=0.20, top=0.80)
    _draw_panel_a(ax_a, base, inputs["a_gap"], title=False)
    _add_panel_a_header(fig_a, ax_a, key_offset=0.12)
    _alignment_gate(fig_a, [ax_a], output_dir, "figure3a_camera_ready")
    _save_formats(fig_a, output_dir, "figure3a_camera_ready")

    # Standalone panel (b).
    fig_b, axes_b = plt.subplots(1, 2, figsize=(PANEL_B_WIDTH_IN, PANEL_HEIGHT_IN), gridspec_kw={"width_ratios": [1.04, 0.96]})
    fig_b.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.80, wspace=0.25)
    _draw_panel_b(fig_b, list(axes_b), base, inputs["b_detail"], inputs["b_summary"], group_title=True)
    _alignment_gate(fig_b, list(axes_b), output_dir, "figure3b_camera_ready")
    _save_formats(fig_b, output_dir, "figure3b_camera_ready")

    # Standalone panel (c).
    fig_c, axes_c = plt.subplots(1, 2, figsize=(PANEL_C_WIDTH_IN, PANEL_HEIGHT_IN), sharey=True)
    fig_c.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.72, wspace=0.18)
    handles_c = _draw_panel_c(list(axes_c), base, inputs["c_retention"])
    fig_c.text(0.10, 0.965, "(c) Semantic retention under semantic anchoring", ha="left", va="top", fontsize=8.8, fontweight="bold")
    fig_c.legend(handles=handles_c, loc="upper center", bbox_to_anchor=(0.55, 0.885), ncol=4, columnspacing=0.45, handletextpad=0.25, borderaxespad=0.0, fontsize=6.6)
    _alignment_gate(fig_c, list(axes_c), output_dir, "figure3c_camera_ready")
    _save_formats(fig_c, output_dir, "figure3c_camera_ready")

    # Camera-ready two-row composite.
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.18, 1.38],
        height_ratios=[1.0, 1.12],
        hspace=0.66,
        wspace=0.36,
    )
    ax_a_full = fig.add_subplot(outer[0, 0])
    b_inner = outer[0, 1].subgridspec(1, 2, width_ratios=[1.04, 0.96], wspace=0.42)
    ax_b_left = fig.add_subplot(b_inner[0, 0])
    ax_b_right = fig.add_subplot(b_inner[0, 1])
    c_inner = outer[1, :].subgridspec(1, 2, wspace=0.18)
    ax_c_left = fig.add_subplot(c_inner[0, 0])
    ax_c_right = fig.add_subplot(c_inner[0, 1], sharey=ax_c_left)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.90)

    _draw_panel_a(ax_a_full, base, inputs["a_gap"], title=False)
    _draw_panel_b(
        fig,
        [ax_b_left, ax_b_right],
        base,
        inputs["b_detail"],
        inputs["b_summary"],
        group_title=False,
        show_median=True,
        median_fontsize=5.8,
    )
    handles_c_full = _draw_panel_c([ax_c_left, ax_c_right], base, inputs["c_retention"])

    fig.canvas.draw()
    _add_panel_a_header(fig, ax_a_full, title_size=8.1)
    b_pos = ax_b_left.get_position()
    c_pos = ax_c_left.get_position()
    fig.text(b_pos.x0, b_pos.y1 + 0.045, "(b) Modality-specific neighbor allocation", ha="left", va="bottom", fontsize=8.8, fontweight="bold")
    fig.text(c_pos.x0, c_pos.y1 + 0.105, "(c) Semantic retention under semantic anchoring", ha="left", va="bottom", fontsize=8.8, fontweight="bold")
    c_center = 0.5 * (c_pos.x0 + ax_c_right.get_position().x1)
    fig.legend(
        handles=handles_c_full,
        loc="upper center",
        bbox_to_anchor=(c_center, c_pos.y1 + 0.065),
        ncol=4,
        columnspacing=0.45,
        handletextpad=0.25,
        borderaxespad=0.0,
        fontsize=6.6,
    )

    _alignment_gate(
        fig,
        [ax_a_full, ax_b_left, ax_b_right, ax_c_left, ax_c_right],
        output_dir,
        "figure3_final_camera_ready",
        row_groups=[["b", "c"], ["d", "e"]],
    )
    _save_formats(fig, output_dir, "figure3_final_camera_ready")
    _write_manifest(output_dir, v2_dir, v3_dir, inputs)
    print(f"[figure3-camera-ready] wrote camera-ready panels to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
