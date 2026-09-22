#!/usr/bin/env python3
"""Plot the correctness-audited Figure 3 V2 mechanism analyses.

The script exports both candidate versions of Panel (a).  The combined figure
uses ``--main-panel-a auto`` by default: A2 is selected only when the
semantic-gap response is positive and reasonably consistent across datasets
and seeds; otherwise the more conservative A1 correlation panel is selected.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "figure3_mechanism_v2"
TEXT_BLUE = "#4C78A8"
TEXT_BLUE_LIGHT = "#AFCBE8"
VISUAL_GREEN = "#59A14F"
VISUAL_GREEN_LIGHT = "#B6DCA9"
MISMATCH_GRAY = "#A9ADB1"
NEUTRAL = "#6F7478"
NEUTRAL_LIGHT = "#D7DADC"
INK = "#24272A"
GRID = "#E7E9EB"
DATASETS = ("Movies", "Grocery")
MODALITIES = ("text", "visual")
HOPS = (0, 1, 2, 3)
FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 112.0
PANEL_WIDTH_MM = 62.0
PANEL_HEIGHT_MM = 50.0
PRIMARY_SPACE = "projected_h0_raw"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--main-panel-a", choices=("auto", "a1", "a2"), default="auto")
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
    # Keep SVG text editable and PDF text as TrueType, matching the manuscript
    # figure contract used by the earlier figures.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.2,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
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


def _make_empty_message(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=NEUTRAL, fontsize=7.5)
    ax.set_axis_off()


def _a1_records(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], np.ndarray]:
    records: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("semantic_space") != PRIMARY_SPACE or row.get("spearman_rho", "") == "":
            continue
        records[(row["dataset"], row["relation_weight"], row["semantic_similarity"])].append(
            _float(row, "spearman_rho")
        )
    return {key: np.asarray(value, dtype=float) for key, value in records.items()}


def _panel_a1(ax: Any, rows: list[dict[str, str]], label: str = "a") -> None:
    records = _a1_records(rows)
    x = np.arange(len(DATASETS), dtype=float)
    specs = [
        (("W^T", "S^T"), TEXT_BLUE, "", 0.98, "W^T vs S^T"),
        (("W^T", "S^V"), MISMATCH_GRAY, "", 0.76, "W^T vs S^V"),
        (("W^V", "S^V"), VISUAL_GREEN, "", 0.98, "W^V vs S^V"),
        (("W^V", "S^T"), MISMATCH_GRAY, "", 0.76, "W^V vs S^T"),
    ]
    width = 0.18
    offsets = np.asarray([-1.5, -0.5, 0.5, 1.5]) * width
    observed: list[float] = []
    for offset, (key, color, hatch, alpha, _) in zip(offsets, specs):
        centers: list[float] = []
        errors: list[float] = []
        for dataset in DATASETS:
            values = records.get((dataset, *key), np.asarray([], dtype=float))
            centers.append(float(np.mean(values)) if values.size else np.nan)
            errors.append(float(np.std(values)) if values.size > 1 else 0.0)
            observed.extend(values.tolist())
        center_array = np.asarray(centers, dtype=float)
        error_array = np.asarray(errors, dtype=float)
        mask = np.isfinite(center_array)
        ax.bar(
            x[mask] + offset,
            center_array[mask],
            width=width * 0.82,
            color=color,
            alpha=alpha,
            hatch=hatch,
            edgecolor=INK,
            linewidth=0.45,
            yerr=error_array[mask],
            error_kw={"elinewidth": 0.65, "capsize": 1.8, "capthick": 0.65},
            zorder=3,
        )
    observed_array = np.asarray(observed, dtype=float)
    lower = 0.0 if observed_array.size == 0 or np.min(observed_array) >= 0 else float(np.floor(np.min(observed_array) * 10.0) / 10.0 - 0.05)
    upper = 0.9 if observed_array.size == 0 or np.max(observed_array) <= 0.9 else float(np.ceil(np.max(observed_array) * 10.0) / 10.0)
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xticks(x, labels=DATASETS)
    ax.set_xlim(-0.58, len(DATASETS) - 0.42)
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Spearman rho", labelpad=4)
    ax.set_xlabel("Dataset", labelpad=4)
    ax.set_title("Modality-aligned relation calibration", loc="left", pad=7, fontweight="bold")
    handles = [
        Patch(facecolor=TEXT_BLUE, edgecolor=INK, label="W^T vs S^T", alpha=0.98),
        Patch(facecolor=TEXT_BLUE_LIGHT, edgecolor=INK, label="W^T vs S^V", alpha=0.9),
        Patch(facecolor=VISUAL_GREEN, edgecolor=INK, label="W^V vs S^V", alpha=0.98),
        Patch(facecolor=NEUTRAL_LIGHT, edgecolor=INK, label="W^V vs S^T", alpha=0.9),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.30), ncol=2, columnspacing=0.7, handletextpad=0.8, borderaxespad=0.15)
    _format_axis(ax)
    _add_panel_label(ax, label)


def _gap_summary(rows: list[dict[str, str]]) -> dict[tuple[str, int], tuple[float, float, int]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], int(row["bin_index"]))].append(_float(row, "delta_weight_mean"))
    x_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        x_values[(row["dataset"], int(row["bin_index"]))].append(_float(row, "delta_sem_mean"))
    return {
        key: (float(np.mean(x_values[key])), float(np.std(values)), len(values))
        for key, values in grouped.items()
    }


def _gap_seed_summary(rows: list[dict[str, str]]) -> dict[str, dict[int, tuple[float, float]]]:
    result: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (row["dataset"], int(row["seed"]), int(row["bin_index"]))
        if key in seen:
            continue
        seen.add(key)
        rho = row.get("edge_spearman_rho", "")
        slope = row.get("edge_slope", "")
        if rho != "" and slope != "":
            result[row["dataset"]][int(row["seed"])] = (_float(row, "edge_spearman_rho"), _float(row, "edge_slope"))
    return result


def _a2_clear(rows: list[dict[str, str]]) -> tuple[bool, dict[str, Any]]:
    by_dataset = _gap_seed_summary(rows)
    dataset_stats: dict[str, Any] = {}
    clear = bool(by_dataset)
    for dataset in DATASETS:
        values = list(by_dataset.get(dataset, {}).values())
        rhos = np.asarray([value[0] for value in values], dtype=float)
        slopes = np.asarray([value[1] for value in values], dtype=float)
        dataset_stats[dataset] = {
            "num_seeds": int(len(values)),
            "mean_edge_spearman_rho": None if not len(values) else float(np.mean(rhos)),
            "mean_edge_slope": None if not len(values) else float(np.mean(slopes)),
            "positive_slope_fraction": None if not len(values) else float(np.mean(slopes > 0.0)),
        }
        # A2 is promoted only when both datasets have a positive, non-trivial
        # monotonic response and at least two seed-level observations.
        clear = clear and len(values) >= 2 and float(np.mean(rhos)) >= 0.20 and float(np.mean(slopes)) > 0.0 and float(np.mean(slopes > 0.0)) >= 0.67
    return clear, dataset_stats


def _panel_a2(ax: Any, rows: list[dict[str, str]], label: str = "a") -> None:
    summary = _gap_summary(rows)
    dataset_colors = {"Movies": TEXT_BLUE, "Grocery": VISUAL_GREEN}
    for dataset in DATASETS:
        xs: list[float] = []
        ys: list[float] = []
        yerr: list[float] = []
        for bin_index in sorted(index for current_dataset, index in summary if current_dataset == dataset):
            x_value, spread, _ = summary[(dataset, bin_index)]
            values = [
                _float(row, "delta_weight_mean")
                for row in rows
                if row["dataset"] == dataset and int(row["bin_index"]) == bin_index
            ]
            xs.append(x_value)
            ys.append(float(np.mean(values)))
            yerr.append(spread)
        if xs:
            color = dataset_colors[dataset]
            ax.errorbar(xs, ys, yerr=yerr, color=color, marker="o", markersize=3.2, linewidth=1.35, capsize=1.7, label=dataset, zorder=3)
    all_x = [_float(row, "delta_sem_mean") for row in rows]
    all_y = [_float(row, "delta_weight_mean") for row in rows]
    x_min, x_max = (min(all_x), max(all_x)) if all_x else (-1.0, 1.0)
    y_min, y_max = (min(all_y), max(all_y)) if all_y else (-0.1, 0.1)
    x_pad = max(0.03, 0.08 * (x_max - x_min))
    y_pad = max(0.003, 0.12 * (y_max - y_min))
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax.axvline(0.0, color=NEUTRAL, linewidth=0.6, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Semantic discrepancy  (S^T − S^V)", labelpad=4)
    ax.set_ylabel("Weight discrepancy  (W^T − W^V)", labelpad=4)
    ax.set_title("Modality-aligned relation calibration", loc="left", pad=7, fontweight="bold")
    ax.text(0.03, 0.96, "Movies", transform=ax.transAxes, ha="left", va="top", color=TEXT_BLUE, fontsize=6.8, fontweight="bold")
    ax.text(0.55, 0.96, "Grocery", transform=ax.transAxes, ha="left", va="top", color=VISUAL_GREEN, fontsize=6.8, fontweight="bold")
    _format_axis(ax)
    _add_panel_label(ax, label)


def _panel_b_values(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str], list[float]], dict[tuple[str, str], np.ndarray]]:
    per_seed: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["modality"])
        pooled[key].append(_float(row, "delta_compatibility"))
        per_seed[(row["dataset"], row["modality"], row["seed"])].append(int(row["improved"]))
    fractions: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (dataset, modality, _), values in per_seed.items():
        fractions[(dataset, modality)].append(float(np.mean(values)))
    return fractions, {key: np.asarray(value, dtype=float) for key, value in pooled.items()}


def _panel_b(ax: Any, rows: list[dict[str, str]], label: str = "b") -> None:
    fractions, pooled = _panel_b_values(rows)
    keys = [(dataset, modality) for dataset in DATASETS for modality in MODALITIES]
    positions = np.arange(len(keys), dtype=float)
    heights: list[float] = []
    errors: list[float] = []
    for key in keys:
        values = np.asarray(fractions.get(key, []), dtype=float)
        heights.append(100.0 * float(np.mean(values)) if values.size else np.nan)
        errors.append(100.0 * float(np.std(values)) if values.size > 1 else 0.0)
    colors = [TEXT_BLUE if modality == "text" else VISUAL_GREEN for _, modality in keys]
    bars = ax.bar(
        positions,
        heights,
        width=0.68,
        color=colors,
        alpha=0.9,
        edgecolor=INK,
        linewidth=0.45,
        yerr=errors,
        error_kw={"elinewidth": 0.65, "capsize": 1.8, "capthick": 0.65},
        zorder=3,
    )
    for position, key, height in zip(positions, keys, heights):
        values = pooled.get(key, np.asarray([], dtype=float))
        if not values.size or not np.isfinite(height):
            continue
        median_milli = 1000.0 * float(np.median(values))
        y_text = min(106.0, max(4.0, height + 2.0))
        ax.text(position, y_text, f"m={median_milli:+.2f}e-3", ha="center", va="bottom", fontsize=5.2, color=INK, clip_on=False)
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xticks(positions, labels=[f"{dataset}\n{modality.title()}" for dataset, modality in keys])
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.set_ylim(0.0, 108.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Nodes with Δu > 0 (%)", labelpad=4)
    ax.set_xlabel("Dataset and modality", labelpad=4)
    ax.set_title("Compatibility improvement after calibration", loc="left", pad=7, fontweight="bold")
    _format_axis(ax)
    _add_panel_label(ax, label)


def _panel_c_summary(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], tuple[float, float, int]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["propagation_condition"], row["modality"], int(row["hop"]))].append(_float(row, "mean_retention"))
    return {
        key: (float(np.mean(values)), float(np.std(values)), len(values))
        for key, values in grouped.items()
    }


def _panel_c(
    axs: list[Any],
    rows: list[dict[str, str]],
    label: str = "c",
    legend_anchor: tuple[float, float] = (0.0, 0.0),
    legend_loc: str = "lower left",
) -> None:
    summary = _panel_c_summary(rows)
    all_values = [_float(row, "mean_retention") for row in rows]
    min_value = min(all_values) if all_values else 0.8
    max_value = max(all_values) if all_values else 1.0
    lower = 0.80 if min_value >= 0.80 else float(np.floor(min_value * 100.0) / 100.0 - 0.01)
    upper = 1.00 if max_value <= 1.00 else float(np.ceil(max_value * 100.0) / 100.0 + 0.01)
    for axis, dataset in zip(axs, DATASETS):
        for modality, color in (("text", TEXT_BLUE), ("visual", VISUAL_GREEN)):
            for condition, linestyle, condition_label in (("anchored", "-", "Anchored"), ("anchor_off_counterfactual", "--", "Anchor-off")):
                means: list[float] = []
                spreads: list[float] = []
                valid: list[bool] = []
                for hop in HOPS:
                    mean, spread, count = summary.get((dataset, condition, modality, hop), (np.nan, np.nan, 0))
                    means.append(mean)
                    spreads.append(spread)
                    valid.append(count > 0)
                mean_array = np.asarray(means, dtype=float)
                spread_array = np.asarray(spreads, dtype=float)
                mask = np.isfinite(mean_array) & np.asarray(valid, dtype=bool)
                if not mask.any():
                    continue
                axis.plot(
                    np.asarray(HOPS)[mask],
                    mean_array[mask],
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markersize=3.0,
                    linewidth=1.45 if condition == "anchored" else 1.15,
                    alpha=1.0 if condition == "anchored" else 0.78,
                    zorder=3,
                )
                axis.fill_between(
                    np.asarray(HOPS)[mask],
                    mean_array[mask] - spread_array[mask],
                    mean_array[mask] + spread_array[mask],
                    color=color,
                    alpha=0.08 if condition == "anchored" else 0.04,
                    linewidth=0,
                    zorder=1,
                )
        # Put the hop-3 intervention annotation in the facet title, above the
        # plotted paths, so it cannot be mistaken for a curve or cross one.
        gain_labels: list[str] = []
        for modality, short_label in (("text", "Text"), ("visual", "Visual")):
            anchored = summary.get((dataset, "anchored", modality, 3), (np.nan, np.nan, 0))[0]
            off = summary.get((dataset, "anchor_off_counterfactual", modality, 3), (np.nan, np.nan, 0))[0]
            if np.isfinite(anchored) and np.isfinite(off):
                gain_labels.append(f"{short_label} {100.0 * (anchored - off):+.1f} pp")
        facet_title = dataset if not gain_labels else f"{dataset}  |  hop3: " + "; ".join(gain_labels)
        axis.set_xticks(HOPS, labels=[str(hop) for hop in HOPS])
        axis.set_xlim(-0.05, 3.05)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("Hop order k", labelpad=4)
        axis.set_title(facet_title, loc="left", pad=7, fontsize=6.2, fontweight="bold")
        _format_axis(axis)
    axs[0].set_ylabel("Semantic retention", labelpad=4)
    handles = [
        Line2D([0], [0], color=TEXT_BLUE, linestyle="-", marker="o", linewidth=1.45, markersize=3.0, label="Text, anchored"),
        Line2D([0], [0], color=TEXT_BLUE, linestyle="--", marker="o", linewidth=1.15, markersize=3.0, label="Text, anchor-off"),
        Line2D([0], [0], color=VISUAL_GREEN, linestyle="-", marker="o", linewidth=1.45, markersize=3.0, label="Visual, anchored"),
        Line2D([0], [0], color=VISUAL_GREEN, linestyle="--", marker="o", linewidth=1.15, markersize=3.0, label="Visual, anchor-off"),
    ]
    axs[-1].legend(
        handles=handles,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.15,
    )
    _add_panel_label(axs[0], label)


def _new_axis_panel() -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.23, top=0.83)
    return fig, ax


def main() -> int:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    panel_a_rows = _read_csv(data_dir / "panel_a_correlation.csv")
    panel_a_gap_rows = _read_csv(data_dir / "panel_a_gap_response.csv")
    panel_b_rows = _read_csv(data_dir / "panel_b_improvement.csv")
    panel_c_rows = _read_csv(data_dir / "panel_c_retention_counterfactual.csv")
    if not any((panel_a_rows, panel_a_gap_rows, panel_b_rows, panel_c_rows)):
        print(f"[figure3-v2-plot] no generated CSV data found in {data_dir}", file=sys.stderr)

    a2_clear, a2_stats = _a2_clear(panel_a_gap_rows)
    selected_a = "a2" if args.main_panel_a == "auto" and a2_clear else ("a1" if args.main_panel_a == "auto" else args.main_panel_a)
    selected_reason = "A2 passed the cross-dataset/seed response gate." if selected_a == "a2" else "A1 retained as the conservative direct alignment panel; A2 did not pass the response gate or was not selected."

    # Candidate A1.
    fig_a1, ax_a1 = _new_axis_panel()
    if panel_a_rows:
        _panel_a1(ax_a1, panel_a_rows)
    else:
        _make_empty_message(ax_a1, "No Panel (a) A1 data")
    _alignment_gate(fig_a1, [ax_a1], output_dir, "figure3a_correlation_v2")
    _save_formats(fig_a1, output_dir, "figure3a_correlation_v2")

    # Candidate A2.
    fig_a2, ax_a2 = _new_axis_panel()
    if panel_a_gap_rows:
        _panel_a2(ax_a2, panel_a_gap_rows)
    else:
        _make_empty_message(ax_a2, "No Panel (a) A2 data")
    _alignment_gate(fig_a2, [ax_a2], output_dir, "figure3a_gap_response_v2")
    _save_formats(fig_a2, output_dir, "figure3a_gap_response_v2")

    # Panel (b).
    fig_b, ax_b = _new_axis_panel()
    if panel_b_rows:
        _panel_b(ax_b, panel_b_rows)
    else:
        _make_empty_message(ax_b, "No Panel (b) data")
    _alignment_gate(fig_b, [ax_b], output_dir, "figure3b_compatibility_v2")
    _save_formats(fig_b, output_dir, "figure3b_compatibility_v2")

    # Panel (c), kept as two shared-y facets.
    fig_c, axs_c = plt.subplots(1, 2, figsize=(122 / 25.4, PANEL_HEIGHT_MM / 25.4), sharey=True)
    fig_c.subplots_adjust(left=0.10, right=0.99, bottom=0.21, top=0.74, wspace=0.24)
    if panel_c_rows:
        _panel_c(list(axs_c), panel_c_rows, legend_anchor=(0.5, 1.36), legend_loc="lower center")
    else:
        for axis in axs_c:
            _make_empty_message(axis, "No Panel (c) data")
    axs_c[-1].tick_params(labelleft=False)
    fig_c.suptitle("Semantic retention across propagation depth", fontsize=8.2, fontweight="bold", y=0.91)
    _alignment_gate(fig_c, list(axs_c), output_dir, "figure3c_retention_v2")
    _save_formats(fig_c, output_dir, "figure3c_retention_v2")

    # Combined manuscript figure: top row (a,b), bottom row (c) across width.
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12), hspace=0.70, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    bottom = grid[1, :].subgridspec(1, 2, wspace=0.24)
    ax_c1 = fig.add_subplot(bottom[0, 0])
    ax_c2 = fig.add_subplot(bottom[0, 1], sharey=ax_c1)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.89)
    if selected_a == "a2":
        _panel_a2(ax_a, panel_a_gap_rows)
    else:
        _panel_a1(ax_a, panel_a_rows)
    if panel_b_rows:
        _panel_b(ax_b, panel_b_rows)
    else:
        _make_empty_message(ax_b, "No Panel (b) data")
    if panel_c_rows:
        _panel_c([ax_c1, ax_c2], panel_c_rows)
    else:
        _make_empty_message(ax_c1, "No Panel (c) data")
        _make_empty_message(ax_c2, "No Panel (c) data")
    fig.canvas.draw()
    c_left = ax_c1.get_position()
    c_right = ax_c2.get_position()
    c_center = 0.5 * (c_left.x0 + c_right.x1)
    fig.text(c_center, max(c_left.y1, c_right.y1) + 0.075, "Semantic retention across propagation depth", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax_c2.tick_params(labelleft=False)
    _alignment_gate(fig, [ax_a, ax_b, ax_c1, ax_c2], output_dir, "figure3_mechanism_v2")
    _save_formats(fig, output_dir, "figure3_mechanism_v2")

    plot_manifest = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "main_panel_a_request": args.main_panel_a,
        "main_panel_a_selected": selected_a,
        "main_panel_a_reason": selected_reason,
        "a2_response_gate": a2_stats,
        "panel_rows": {"a_correlation": len(panel_a_rows), "a_gap": len(panel_a_gap_rows), "b": len(panel_b_rows), "c": len(panel_c_rows)},
        "exports": [
            "figure3_mechanism_v2.png",
            "figure3_mechanism_v2.tiff",
            "figure3_mechanism_v2.pdf",
            "figure3_mechanism_v2.svg",
            "figure3a_correlation_v2.png",
            "figure3a_correlation_v2.tiff",
            "figure3a_correlation_v2.pdf",
            "figure3a_correlation_v2.svg",
            "figure3a_gap_response_v2.png",
            "figure3a_gap_response_v2.tiff",
            "figure3a_gap_response_v2.pdf",
            "figure3a_gap_response_v2.svg",
            "figure3b_compatibility_v2.png",
            "figure3b_compatibility_v2.tiff",
            "figure3b_compatibility_v2.pdf",
            "figure3b_compatibility_v2.svg",
            "figure3c_retention_v2.png",
            "figure3c_retention_v2.tiff",
            "figure3c_retention_v2.pdf",
            "figure3c_retention_v2.svg",
        ],
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(plot_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[figure3-v2-plot] selected Panel (a) {selected_a}; wrote exports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
