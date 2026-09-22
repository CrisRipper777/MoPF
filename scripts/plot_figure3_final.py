#!/usr/bin/env python3
"""Render the final manuscript Figure 3 from existing V2/V3 CSV outputs.

The figure is a quantitative, claim-escalating composite:

    (a) edge-level semantic discrepancy -> relation response
    (b) relation response -> modality-specific neighbor allocation
    (c) repeated propagation -> semantic retention with an anchor

This script is plotting-only.  It does not load checkpoints, train models, or
modify the existing Figure 3 V2/V3 outputs.  It reads the already generated
V2 A2/C CSV files and V3 neighbor-allocation CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2_DEFAULT = ROOT / "outputs" / "figure3_mechanism_v2"
V3_DEFAULT = ROOT / "outputs" / "figure3_mechanism_v3_candidate"
OUTPUT_DEFAULT = ROOT / "outputs" / "figure3_final"
DATASETS = ("Movies", "Grocery")
MODALITIES = ("text", "visual")
HOPS = (0, 1, 2, 3)

# The modality language is reserved for Panel (c).  Panel (b) compares
# datasets, so muted purple/orange tones keep dataset identity distinct from
# the global Text/Visual blue/green encoding while matching the paper's softer
# reference palette.
TEXT_BLUE = "#4C78A8"
VISUAL_GREEN = "#59A14F"
MOVIES_NEUTRAL = "#7563A6"
MOVIES_LIGHT = "#C9BDE2"
GROCERY_NEUTRAL = "#D28A3A"
GROCERY_LIGHT = "#F1C58D"
NEUTRAL = "#6F7478"
GRID = "#E7E9EB"
INK = "#24272A"
WHITE = "#FFFFFF"

# Match the wide, horizontal manuscript-figure contract used by Figure 1 V5.
# The larger canvas gives the five quantitative axes enough physical width while
# retaining the same compact journal-scale typography.
FIGURE_WIDTH_MM = 431.8  # 17.0 in
FIGURE_HEIGHT_MM = 121.92  # 4.8 in


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-data-dir", type=Path, default=V2_DEFAULT)
    parser.add_argument("--v3-data-dir", type=Path, default=V3_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
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
    # Keep all final exports editable and consistent with the earlier paper
    # figures: editable SVG text and TrueType text in PDF.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "STIXGeneral", "Times New Roman"]
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.2,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
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
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.4, linestyle=(0, (1.2, 2.0)), alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.15,
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
    # Keep the declared physical canvas intact, matching the Figure 1 V5
    # export contract.  Automatic tight cropping changes the page size and
    # makes the figure less predictable when placed in the manuscript.
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def _alignment_gate(
    fig: Any,
    axes: list[Any],
    output_dir: Path,
    stem: str,
    *,
    row_groups: Iterable[Any] | None = None,
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
        options["row_groups"] = list(row_groups)
    require_matplotlib_panel_alignment(fig, **options)


def _git_value(*args: str) -> str | None:
    try:
        value = subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _add_empty_message(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=NEUTRAL, fontsize=7.0)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Panel (a): V2 A2 semantic-gap response
# ---------------------------------------------------------------------------


def _gap_points(rows: list[dict[str, str]]) -> dict[str, list[tuple[float, float, float]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("semantic_space", "projected_h0_raw") != "projected_h0_raw":
            continue
        grouped[(row["dataset"], int(row["bin_index"]))].append(row)
    output: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (dataset, _bin_index), records in grouped.items():
        x = float(np.mean([_float(row, "delta_sem_mean") for row in records]))
        y_values = np.asarray([_float(row, "delta_weight_mean") for row in records], dtype=float)
        output[dataset].append((x, float(np.mean(y_values)), float(np.std(y_values))))
    for dataset in output:
        output[dataset].sort(key=lambda item: item[0])
    return output


def _draw_panel_a(ax: Any, rows: list[dict[str, str]], *, label: str | None) -> None:
    points = _gap_points(rows)
    colors = {"Movies": TEXT_BLUE, "Grocery": VISUAL_GREEN}
    all_x = [item[0] for dataset in DATASETS for item in points.get(dataset, [])]
    all_y = [item[1] for dataset in DATASETS for item in points.get(dataset, [])]
    if not all_x:
        _add_empty_message(ax, "No Panel (a) data")
        return
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    handles: list[Any] = []
    for dataset in DATASETS:
        values = points.get(dataset, [])
        if not values:
            continue
        x = np.asarray([item[0] for item in values], dtype=float)
        y = np.asarray([item[1] for item in values], dtype=float)
        error = np.asarray([item[2] for item in values], dtype=float)
        color = colors[dataset]
        ax.errorbar(
            x,
            y,
            yerr=error,
            color=color,
            marker="o",
            markersize=2.5,
            linewidth=1.15,
            capsize=1.3,
            capthick=0.55,
            label=dataset,
            zorder=3,
        )
        handles.append(Line2D([0], [0], color=color, marker="o", linewidth=1.15, markersize=2.5, label=dataset))
    x_pad = max(0.12, 0.18 * (x_max - x_min))
    y_pad = max(0.004, 0.14 * (y_max - y_min))
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(0.0, color=NEUTRAL, linewidth=0.55, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Semantic discrepancy (text − visual)", labelpad=3)
    ax.set_ylabel("Weight response (text − visual)", labelpad=3)
    ax.set_title("Relation response to semantic gap", loc="left", pad=16, fontweight="bold")
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, columnspacing=0.55, handletextpad=0.28, borderaxespad=0.0, fontsize=6.5)
    _format_axis(ax)
    if label:
        _add_panel_label(ax, label)


# ---------------------------------------------------------------------------
# Panel (b): V3 modality-specific neighbor allocation
# ---------------------------------------------------------------------------


def _allocation_records(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    records: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        records[row["dataset"]].append(_float(row, "tv_distance"))
    return {dataset: np.asarray(records.get(dataset, []), dtype=float) for dataset in DATASETS}


def _allocation_summary(rows: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float]]:
    pooled: dict[str, float] = {}
    seed_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        dataset = row["dataset"]
        level = row.get("aggregation_level", "")
        if level == "cross_seed_pooled":
            pooled[dataset] = _float(row, "top_neighbor_disagreement_percentage")
        elif level == "seed":
            seed_values[dataset].append(_float(row, "top_neighbor_disagreement_percentage"))
    seed_sd = {
        dataset: float(np.std(seed_values[dataset])) if seed_values.get(dataset) else 0.0
        for dataset in DATASETS
    }
    return pooled, seed_sd


def _draw_panel_b_left(
    ax: Any,
    records: dict[str, np.ndarray],
    *,
    label: str | None,
    show_median_labels: bool = True,
    show_subtitle: bool = True,
) -> None:
    colors = {
        "Movies": (MOVIES_NEUTRAL, MOVIES_LIGHT),
        "Grocery": (GROCERY_NEUTRAL, GROCERY_LIGHT),
    }
    positions = np.arange(len(DATASETS), dtype=float)
    values_list = [records.get(dataset, np.asarray([], dtype=float)) for dataset in DATASETS]
    if not any(values.size for values in values_list):
        _add_empty_message(ax, "No Panel (b) data")
        return
    for position, dataset, values in zip(positions, DATASETS, values_list):
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
        body.set_facecolor(colors[dataset][1])
        body.set_edgecolor(colors[dataset][0])
        body.set_linewidth(0.65)
        body.set_alpha(0.95)
        box = ax.boxplot(
            [values],
            positions=[position],
            widths=0.17,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops={"color": INK, "linewidth": 0.9},
            whiskerprops={"color": INK, "linewidth": 0.55},
            capprops={"color": INK, "linewidth": 0.55},
            boxprops={"facecolor": WHITE, "edgecolor": INK, "linewidth": 0.55},
        )
        box["boxes"][0].set_alpha(0.98)
        median = float(np.median(values))
        ax.scatter([position], [median], color=colors[dataset][0], edgecolor=WHITE, linewidth=0.35, s=12, zorder=5)
        # The label sits in the upper blank part of the compressed 0–0.11
        # scale.  It is a direct median label, not a second statistic.
        if show_median_labels:
            ax.text(position + 0.34, 0.096, f"m={median:.3f}", ha="left", va="top", fontsize=6.0, color=INK)
    ax.set_ylim(0.0, 0.11)
    ax.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax.set_xticks(positions, labels=DATASETS)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xlabel("Dataset", labelpad=3)
    ax.set_ylabel("TV divergence", labelpad=3)
    if show_subtitle:
        ax.set_title("TV divergence", loc="left", pad=4, fontsize=7.2, fontweight="bold")
    _format_axis(ax)
    if label:
        _add_panel_label(ax, label)


def _draw_panel_b_right(
    ax: Any,
    pooled: dict[str, float],
    *,
    seed_sd: dict[str, float] | None = None,
    label: str | None,
) -> None:
    del seed_sd  # Seed variability is reported in the audit, not added to the main bar panel.
    positions = np.arange(len(DATASETS), dtype=float)
    colors = [MOVIES_NEUTRAL, GROCERY_NEUTRAL]
    heights = np.asarray([pooled.get(dataset, np.nan) for dataset in DATASETS], dtype=float)
    mask = np.isfinite(heights)
    if not mask.any():
        _add_empty_message(ax, "No Panel (b) summary")
        return
    bars = ax.bar(
        positions[mask],
        heights[mask],
        width=0.56,
        color=np.asarray(colors, dtype=object)[mask],
        edgecolor=INK,
        linewidth=0.45,
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
    ax.set_ylabel("Top-1 differs (%)", labelpad=3)
    ax.set_title("Top-neighbor disagreement", loc="left", pad=4, fontsize=7.2, fontweight="bold")
    _format_axis(ax)
    if label:
        _add_panel_label(ax, label)


def _draw_panel_b(
    fig: Any,
    axes: list[Any],
    detail_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    *,
    label: str | None,
    show_median_labels: bool = True,
    show_subtitle: bool = True,
) -> None:
    records = _allocation_records(detail_rows)
    pooled, seed_sd = _allocation_summary(summary_rows)
    _draw_panel_b_left(axes[0], records, label=label, show_median_labels=show_median_labels, show_subtitle=show_subtitle)
    _draw_panel_b_right(axes[1], pooled, seed_sd=seed_sd, label=None)
    del fig


# ---------------------------------------------------------------------------
# Panel (c): V2 same-checkpoint semantic-retention counterfactual
# ---------------------------------------------------------------------------


def _retention_summary(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], tuple[float, float, int]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["propagation_condition"], row["modality"], int(row["hop"]))
        grouped[key].append(_float(row, "mean_retention"))
    return {
        key: (float(np.mean(values)), float(np.std(values)), len(values))
        for key, values in grouped.items()
    }


def _retention_gains(summary: dict[tuple[str, str, str, int], tuple[float, float, int]], dataset: str) -> str:
    labels: list[str] = []
    for modality, short in (("text", "T"), ("visual", "V")):
        anchored = summary.get((dataset, "anchored", modality, 3), (np.nan, np.nan, 0))[0]
        off = summary.get((dataset, "anchor_off_counterfactual", modality, 3), (np.nan, np.nan, 0))[0]
        if np.isfinite(anchored) and np.isfinite(off):
            labels.append(f"{short} {100.0 * (anchored - off):+.1f} pp")
    return "; ".join(labels)


def _compact_gain_label(gain: str) -> str:
    values: list[str] = []
    for item in gain.split("; "):
        parts = item.split(" ", 1)
        if len(parts) == 2:
            values.append(parts[1].replace(" pp", ""))
    return "/".join(values) + (" pp" if values else "")


def _retention_handles() -> list[Any]:
    return [
        Line2D([0], [0], color=TEXT_BLUE, linestyle="-", marker="o", linewidth=1.25, markersize=2.4, label="Text, anchored"),
        Line2D([0], [0], color=TEXT_BLUE, linestyle="--", marker="o", linewidth=1.05, markersize=2.4, label="Text, anchor-off"),
        Line2D([0], [0], color=VISUAL_GREEN, linestyle="-", marker="o", linewidth=1.25, markersize=2.4, label="Visual, anchored"),
        Line2D([0], [0], color=VISUAL_GREEN, linestyle="--", marker="o", linewidth=1.05, markersize=2.4, label="Visual, anchor-off"),
    ]


def _draw_panel_c(
    axes: list[Any],
    rows: list[dict[str, str]],
    *,
    label: str | None,
    show_gain_annotation: bool = True,
    show_facet_titles: bool = True,
) -> list[Any]:
    summary = _retention_summary(rows)
    all_values = [_float(row, "mean_retention") for row in rows]
    observed_min = min(all_values) if all_values else 0.80
    observed_max = max(all_values) if all_values else 1.00
    lower = 0.80 if observed_min >= 0.80 else float(np.floor(observed_min * 100.0) / 100.0 - 0.01)
    upper = 1.00 if observed_max <= 1.00 else float(np.ceil(observed_max * 100.0) / 100.0 + 0.01)
    for axis, dataset in zip(axes, DATASETS):
        for modality, color in (("text", TEXT_BLUE), ("visual", VISUAL_GREEN)):
            for condition, linestyle, alpha, width in (
                ("anchored", "-", 1.0, 1.25),
                ("anchor_off_counterfactual", "--", 0.78, 1.05),
            ):
                means: list[float] = []
                spreads: list[float] = []
                for hop in HOPS:
                    mean, spread, count = summary.get((dataset, condition, modality, hop), (np.nan, np.nan, 0))
                    means.append(mean if count else np.nan)
                    spreads.append(spread if count else np.nan)
                means_array = np.asarray(means, dtype=float)
                spreads_array = np.asarray(spreads, dtype=float)
                mask = np.isfinite(means_array)
                if not mask.any():
                    continue
                x = np.asarray(HOPS, dtype=float)[mask]
                y = means_array[mask]
                spread = spreads_array[mask]
                axis.plot(x, y, color=color, linestyle=linestyle, marker="o", markersize=2.4, linewidth=width, alpha=alpha, zorder=3)
                axis.fill_between(x, y - spread, y + spread, color=color, alpha=0.06 if condition == "anchored" else 0.035, linewidth=0, zorder=1)
        axis.set_xticks(HOPS, labels=[str(hop) for hop in HOPS])
        axis.set_xlim(-0.05, 3.05)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("Hop order k", labelpad=3)
        if show_facet_titles:
            axis.set_title(dataset, loc="left", pad=4, fontsize=7.2, fontweight="bold")
        if show_gain_annotation:
            gain = _retention_gains(summary, dataset)
            if gain:
                axis.text(3.0, 0.975, f"T/V {_compact_gain_label(gain)}", ha="right", va="top", fontsize=6.0, color=NEUTRAL)
        _format_axis(axis)
    axes[0].set_ylabel("Semantic retention", labelpad=3)
    axes[-1].tick_params(labelleft=False)
    if label:
        _add_panel_label(axes[0], label)
    return _retention_handles()


def _panel_axes(fig: Any, grid_spec: Any, *, sharey: Any = None) -> tuple[Any, Any]:
    inner = grid_spec.subgridspec(1, 2, wspace=0.25)
    left = fig.add_subplot(inner[0, 0], sharey=sharey)
    right = fig.add_subplot(inner[0, 1], sharey=left if sharey is None else sharey)
    return left, right


def _write_audit_summary(
    path: Path,
    *,
    v2_dir: Path,
    v3_dir: Path,
    output_dir: Path,
    panel_a_rows: list[dict[str, str]],
    panel_a_gap_rows: list[dict[str, str]],
    panel_c_rows: list[dict[str, str]],
    panel_b_rows: list[dict[str, str]],
    panel_b_summary_rows: list[dict[str, str]],
) -> None:
    points = _gap_points(panel_a_gap_rows)
    summary = _retention_summary(panel_c_rows)
    pooled, seed_sd = _allocation_summary(panel_b_summary_rows)
    detail_records = _allocation_records(panel_b_rows)
    lines = [
        "# Figure 3 final audit summary",
        "",
        "No training was started. No formal model, task, data pipeline, benchmark script, or existing Figure 3 V2/V3 output was modified.",
        "",
        "## Figure-level claim",
        "",
        "The figure follows the mechanism chain `edge-level semantic discrepancy -> modality-specific neighbor allocation -> semantic retention during propagation`.",
        "",
        "## Panel definitions",
        "",
        "- **(a) Relation response to semantic discrepancy:** V2 A2 binned response using projected-H0 raw cosine semantic gap `S^T_raw - S^V_raw` and raw relation-weight discrepancy `W^T - W^V`. Points summarize the existing seed-level bins; error bars are SD across seeds.",
        "- **(b) Modality-specific neighbor allocation:** V3 raw relation weights on the same physical incoming neighbor IDs, with degree > 1 only. Left: node-level TV distance; right: pooled percentage of nodes with different top-1 neighbors. The right bars omit error bars for readability; seed SD is recorded below.",
        "- **(c) Semantic retention from anchored propagation:** V2 same-checkpoint counterfactual. Anchored recurrence is compared with the same checkpoint with only the semantic-anchor coefficient set to zero. Shading is SD across seeds.",
        "",
        "## Panel (b) interpretation boundary",
        "",
        "Panel (b) supports: **MRC induces modality-specific neighbor allocation on the same physical neighborhood.** It does not claim that propagation quality necessarily improves; compatibility improvement remains a supplementary statistic in the V2 directory.",
        "",
        "## Key numerical audit",
        "",
    ]
    for dataset in DATASETS:
        values = detail_records.get(dataset, np.asarray([], dtype=float))
        if values.size:
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            lines.append(f"- {dataset} Panel (b): median TV={median:.5f}; IQR=[{q25:.5f}, {q75:.5f}] (n={values.size:,}).")
            lines.append(f"  - Top-neighbor disagreement={pooled.get(dataset, float('nan')):.2f}%; seed SD={seed_sd.get(dataset, float('nan')):.2f} percentage points.")
    for dataset in DATASETS:
        values = points.get(dataset, [])
        if values:
            slopes = [float(row["edge_slope"]) for row in panel_a_gap_rows if row["dataset"] == dataset and row.get("edge_slope", "") != ""]
            rhos = [float(row["edge_spearman_rho"]) for row in panel_a_gap_rows if row["dataset"] == dataset and row.get("edge_spearman_rho", "") != ""]
            lines.append(f"- {dataset} Panel (a): edge Spearman rho={np.mean(rhos):.3f}; response slope={np.mean(slopes):.4f}.")
    for dataset in DATASETS:
        gains = []
        for modality in MODALITIES:
            anchored = summary.get((dataset, "anchored", modality, 3), (np.nan, np.nan, 0))[0]
            off = summary.get((dataset, "anchor_off_counterfactual", modality, 3), (np.nan, np.nan, 0))[0]
            if np.isfinite(anchored) and np.isfinite(off):
                gains.append(f"{modality} {100.0 * (anchored - off):+.2f} pp")
        if gains:
            lines.append(f"- {dataset} Panel (c): hop-3 anchored gain: " + "; ".join(gains) + ".")
    lines.extend(
        [
            "",
            "## Source-data and denominator audit",
            "",
            f"- V2 source directory: `{v2_dir}`; V3 source directory: `{v3_dir}`.",
            f"- Panel (a) rows read: {len(panel_a_rows)} correlation rows; {len(panel_a_gap_rows)} gap-response rows.",
            f"- Panel (b) rows read: {len(panel_b_rows)} node rows; {len(panel_b_summary_rows)} summary rows.",
            f"- Panel (c) rows read: {len(panel_c_rows)} retention rows.",
            "- Panel (b) degree-one nodes are excluded upstream by the V3 generator; no new node or edge filtering is performed here.",
            "- Panel (b) TV values are pooled across the six existing dataset-seed analyses; the seed-level stability values are retained in the V3 summary CSV.",
            "",
            "## Figure design changes",
            "",
            "- Panel (a) keeps the selected V2 A2 semantic-gap response and uses a shorter title.",
            "- Panel (b) uses V3 B3, removes redundant top-neighbor annotations from the TV subplot, uses a compact 0–0.11 TV scale to retain the full observed range, and labels the two medians directly.",
            "- Panel (c) keeps the shared 0.80–1.00 retention scale and moves hop-3 gains into compact facet annotations.",
            "- The complete figure follows the Figure 1 V5-style wide contract (17 × 4.8 in), uses serif typography, tighter inter-panel spacing, and balanced group widths so the five quantitative axes remain readable.",
            "- The static width heuristic may flag the 431.8-mm canvas because it is wider than the default 89/183-mm single/double-column presets; this is intentional and follows the wide reference-figure contract.",
            "",
            "## Outputs",
            "",
            f"- Final output directory: `{output_dir}`.",
            "- SVG/PDF text remains editable; PNG/TIFF are exported at 600 dpi.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    v2_dir = args.v2_data_dir.resolve()
    v3_dir = args.v3_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_a_rows = _read_csv(v2_dir / "panel_a_correlation.csv")
    panel_a_gap_rows = _read_csv(v2_dir / "panel_a_gap_response.csv")
    panel_c_rows = _read_csv(v2_dir / "panel_c_retention_counterfactual.csv")
    panel_b_rows = _read_csv(v3_dir / "panel_b_neighbor_allocation.csv")
    panel_b_summary_rows = _read_csv(v3_dir / "panel_b_neighbor_allocation_summary.csv")
    if not panel_a_gap_rows or not panel_c_rows or not panel_b_rows or not panel_b_summary_rows:
        missing = []
        for name, rows in (
            ("panel_a_gap_response.csv", panel_a_gap_rows),
            ("panel_c_retention_counterfactual.csv", panel_c_rows),
            ("panel_b_neighbor_allocation.csv", panel_b_rows),
            ("panel_b_neighbor_allocation_summary.csv", panel_b_summary_rows),
        ):
            if not rows:
                missing.append(name)
        raise FileNotFoundError("missing required final-figure inputs: " + ", ".join(missing))

    _configure_style()

    # Standalone Panel (a).
    fig_a, ax_a = plt.subplots(figsize=(132.0 / 25.4, 88.0 / 25.4))
    fig_a.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.80)
    _draw_panel_a(ax_a, panel_a_gap_rows, label="a")
    _alignment_gate(fig_a, [ax_a], output_dir, "figure3a_final")
    _save_formats(fig_a, output_dir, "figure3a_final")

    # Standalone Panel (b): the final B3 design.
    fig_b, axes_b = plt.subplots(1, 2, figsize=(158.0 / 25.4, 88.0 / 25.4), gridspec_kw={"width_ratios": [1.04, 0.96]})
    fig_b.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.80, wspace=0.26)
    fig_b.text(0.10, 0.965, "(b) Modality-specific neighbor allocation", ha="left", va="top", fontsize=8.2, fontweight="bold")
    _draw_panel_b(fig_b, list(axes_b), panel_b_rows, panel_b_summary_rows, label=None)
    _alignment_gate(fig_b, list(axes_b), output_dir, "figure3b_final")
    _save_formats(fig_b, output_dir, "figure3b_final")

    # Standalone Panel (c): two equal-y facets.
    fig_c, axes_c = plt.subplots(1, 2, figsize=(158.0 / 25.4, 88.0 / 25.4), sharey=True)
    fig_c.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.80, wspace=0.20)
    fig_c.text(0.10, 0.965, "(c) Semantic retention from anchored propagation", ha="left", va="top", fontsize=8.2, fontweight="bold")
    handles_c = _draw_panel_c(list(axes_c), panel_c_rows, label=None)
    fig_c.legend(handles=handles_c, loc="upper center", bbox_to_anchor=(0.55, 0.895), ncol=4, columnspacing=0.38, handletextpad=0.22, borderaxespad=0.0, fontsize=6.5)
    _alignment_gate(fig_c, list(axes_c), output_dir, "figure3c_final")
    _save_formats(fig_c, output_dir, "figure3c_final")

    # Complete horizontal Figure 3.  The explicit row groups audit the two
    # comparable subplots inside B and C without incorrectly requiring the
    # nested B/C subplots to have the same widths as Panel (a).
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.10, 2.10, 2.05], wspace=0.20)
    ax_a_full = fig.add_subplot(outer[0, 0])
    b_inner = outer[0, 1].subgridspec(1, 2, width_ratios=[1.04, 0.96], wspace=0.26)
    ax_b_left = fig.add_subplot(b_inner[0, 0])
    # B contains two different quantities with intentionally different scales:
    # TV is in [0, 0.11], whereas top-neighbor disagreement is a percentage.
    ax_b_right = fig.add_subplot(b_inner[0, 1])
    c_inner = outer[0, 2].subgridspec(1, 2, wspace=0.20)
    ax_c_left = fig.add_subplot(c_inner[0, 0])
    ax_c_right = fig.add_subplot(c_inner[0, 1], sharey=ax_c_left)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.19, top=0.80)

    _draw_panel_a(ax_a_full, panel_a_gap_rows, label="a")
    _draw_panel_b(
        fig,
        [ax_b_left, ax_b_right],
        panel_b_rows,
        panel_b_summary_rows,
        label=None,
        show_median_labels=False,
        show_subtitle=False,
    )
    handles_c_full = _draw_panel_c(
        [ax_c_left, ax_c_right],
        panel_c_rows,
        label=None,
        show_gain_annotation=False,
        show_facet_titles=False,
    )

    # Group-level titles keep the semantic roles distinct from B/C's internal
    # subplot titles while preserving a common panel-label convention.
    fig.canvas.draw()
    b_pos = ax_b_left.get_position()
    c_pos = ax_c_left.get_position()
    b_records = _allocation_records(panel_b_rows)
    for position, dataset in enumerate(DATASETS):
        values = b_records.get(dataset, np.asarray([], dtype=float))
        if values.size:
            display = ax_b_left.transData.transform((float(position), 0.0))
            figure_xy = fig.transFigure.inverted().transform(display)
            fig.text(float(figure_xy[0]), b_pos.y1 + 0.012, f"m={np.median(values):.3f}", ha="center", va="bottom", fontsize=6.0, color=INK)
    fig.text(b_pos.x0, b_pos.y1 + 0.120, "(b) Modality-specific neighbor allocation", ha="left", va="bottom", fontsize=8.2, fontweight="bold")
    fig.text(c_pos.x0, c_pos.y1 + 0.120, "(c) Semantic retention from anchored propagation", ha="left", va="bottom", fontsize=8.2, fontweight="bold")
    c_summary = _retention_summary(panel_c_rows)
    for axis, dataset in zip((ax_c_left, ax_c_right), DATASETS):
        gain = _retention_gains(c_summary, dataset)
        if gain:
            figure_xy = fig.transFigure.inverted().transform(axis.transData.transform((0.0, 0.0)))
            fig.text(float(figure_xy[0]) + 0.008, c_pos.y1 + 0.018, f"{dataset} | T/V {_compact_gain_label(gain)}", ha="left", va="bottom", fontsize=6.0, color=NEUTRAL)
    fig.legend(handles=handles_c_full, loc="upper center", bbox_to_anchor=(c_pos.x0 + 0.5 * (ax_c_right.get_position().x1 - c_pos.x0), c_pos.y1 + 0.070), ncol=4, columnspacing=0.32, handletextpad=0.20, borderaxespad=0.0, fontsize=6.5)

    _alignment_gate(
        fig,
        [ax_a_full, ax_b_left, ax_b_right, ax_c_left, ax_c_right],
        output_dir,
        "figure3_final",
        row_groups=[["b", "c"], ["d", "e"]],
    )
    _save_formats(fig, output_dir, "figure3_final")

    plot_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT.resolve())),
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "v2_data_dir": str(v2_dir),
        "v3_data_dir": str(v3_dir),
        "output_dir": str(output_dir),
        "figure_contract": {
            "archetype": "quantitative grid",
            "claim": "edge-level semantic discrepancy induces modality-specific neighbor allocation, while semantic anchoring preserves modality semantics across hops",
            "panel_a": "V2 A2 semantic-gap response",
            "panel_b": "V3 B3 raw-weight neighbor allocation; TV plus top-neighbor disagreement",
            "panel_c": "V2 same-checkpoint anchor-off counterfactual",
        },
        "figure_size_inches": [17.0, 4.8],
        "outer_width_ratios": [1.10, 2.10, 2.05],
        "outer_wspace": 0.20,
        "panel_b_scale": {"tv_ylim": [0.0, 0.11], "top_neighbor_ylim": [0.0, 100.0]},
        "source_rows": {
            "panel_a_correlation": len(panel_a_rows),
            "panel_a_gap_response": len(panel_a_gap_rows),
            "panel_b_detail": len(panel_b_rows),
            "panel_b_summary": len(panel_b_summary_rows),
            "panel_c_retention": len(panel_c_rows),
        },
        "exports": [
            "figure3_final.pdf",
            "figure3_final.svg",
            "figure3_final.png",
            "figure3_final.tiff",
            "figure3a_final.pdf",
            "figure3a_final.svg",
            "figure3a_final.png",
            "figure3a_final.tiff",
            "figure3b_final.pdf",
            "figure3b_final.svg",
            "figure3b_final.png",
            "figure3b_final.tiff",
            "figure3c_final.pdf",
            "figure3c_final.svg",
            "figure3c_final.png",
            "figure3c_final.tiff",
        ],
        "protected_outputs_untouched": [
            "outputs/figure3_mechanism_v2",
            "outputs/figure3_mechanism_v3_candidate",
        ],
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(plot_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_audit_summary(
        output_dir / "audit_summary.md",
        v2_dir=v2_dir,
        v3_dir=v3_dir,
        output_dir=output_dir,
        panel_a_rows=panel_a_rows,
        panel_a_gap_rows=panel_a_gap_rows,
        panel_c_rows=panel_c_rows,
        panel_b_rows=panel_b_rows,
        panel_b_summary_rows=panel_b_summary_rows,
    )
    print(f"[figure3-final] wrote final panels and combined figure to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
