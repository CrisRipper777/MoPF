#!/usr/bin/env python3
"""Plot Figure 3 mechanism analyses from generated CSV source data."""

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
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "figure3_mechanism"
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
    # Keep editable text in SVG and TrueType text in PDF, matching Figure 1.
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
            "legend.fontsize": 6.8,
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
    ax.text(-0.13, 1.045, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=INK, clip_on=False)


def _save_formats(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _panel_a_records(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[float]]:
    records: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        rho = row.get("spearman_rho", "")
        if rho == "":
            continue
        records[(row["dataset"], row["relation_weight"], row["semantic_similarity"])].append(float(rho))
    return records


def _panel_a(ax: Any, rows: list[dict[str, str]], label: str = "a") -> None:
    records = _panel_a_records(rows)
    x = np.arange(len(DATASETS), dtype=float)
    specs = [
        (("W^T", "S^T"), TEXT_BLUE, "-"),
        (("W^T", "S^V"), MISMATCH_GRAY, "--"),
        (("W^V", "S^V"), VISUAL_GREEN, "-"),
        (("W^V", "S^T"), MISMATCH_GRAY, ":"),
    ]
    width = 0.18
    offsets = np.asarray([-1.5, -0.5, 0.5, 1.5]) * width
    for offset, (key, color, linestyle) in zip(offsets, specs):
        centers = []
        errors = []
        for dataset in DATASETS:
            values = np.asarray(records.get((dataset, *key), []), dtype=float)
            centers.append(np.nan if values.size == 0 else np.mean(values))
            errors.append(np.nan if values.size < 2 else np.std(values))
        centers = np.asarray(centers, dtype=float)
        errors = np.asarray(errors, dtype=float)
        mask = np.isfinite(centers)
        ax.bar(x[mask] + offset, centers[mask], width=width * 0.82, color=color, alpha=0.9 if key in (("W^T", "S^T"), ("W^V", "S^V")) else 0.52, edgecolor=INK, linewidth=0.45, yerr=np.nan_to_num(errors[mask], nan=0.0), error_kw={"elinewidth": 0.65, "capsize": 1.8, "capthick": 0.65}, zorder=3)
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xticks(x, labels=DATASETS)
    ax.set_xlim(-0.58, len(DATASETS) - 0.42)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Spearman ρ", labelpad=4)
    ax.set_xlabel("Dataset", labelpad=4)
    ax.set_title("Modality-aligned relation calibration", loc="left", pad=7, fontweight="bold")
    handles = [
        Line2D([0], [0], color=TEXT_BLUE, marker="s", linestyle="None", markersize=4.2, label="W^T–S^T"),
        Line2D([0], [0], color=VISUAL_GREEN, marker="s", linestyle="None", markersize=4.2, label="W^V–S^V"),
        Line2D([0], [0], color=MISMATCH_GRAY, marker="s", linestyle="None", markersize=4.2, label="Cross-modal"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=1, borderaxespad=0.15, handletextpad=0.45)
    _format_axis(ax)
    _add_panel_label(ax, label)


def _panel_b_values(rows: list[dict[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        values[(row["dataset"], row["modality"])].append(_float(row, "delta_compatibility"))
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def _panel_b(ax: Any, rows: list[dict[str, str]], label: str = "b") -> None:
    values = _panel_b_values(rows)
    keys = [(dataset, modality) for dataset in DATASETS for modality in MODALITIES]
    data = [values.get(key, np.asarray([], dtype=float)) for key in keys]
    positions = np.arange(len(keys), dtype=float)
    valid_data = [value if value.size else np.asarray([np.nan]) for value in data]
    violins = ax.violinplot(valid_data, positions=positions, widths=0.72, bw_method=0.18, showmeans=False, showmedians=False, showextrema=False, points=160)
    for body, (dataset, modality), value in zip(violins["bodies"], keys, data):
        body.set_facecolor(TEXT_BLUE if modality == "text" else VISUAL_GREEN)
        body.set_edgecolor(TEXT_BLUE if modality == "text" else VISUAL_GREEN)
        body.set_alpha(0.56)
        body.set_linewidth(0.55)
    # Overlay explicit quartile/median summaries at known x positions. This
    # avoids relying on collection order after Matplotlib has drawn violins.
    for position, value, (_, modality) in zip(positions, data, keys):
        if value.size:
            q25, median, q75 = np.quantile(value, [0.25, 0.5, 0.75])
            ax.vlines(position, q25, q75, color=INK, linewidth=1.15, zorder=4)
            ax.scatter(position, median, s=18, facecolor="white", edgecolor=INK, linewidth=0.7, zorder=5)
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.7, linestyle=(0, (2, 2)), zorder=2)
    ax.set_xticks(positions, labels=[f"{dataset}\n{modality.title()}" for dataset, modality in keys])
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.set_ylabel("Δu (calibrated − raw)", labelpad=4)
    ax.set_xlabel("Dataset and modality", labelpad=4)
    ax.set_title("Compatibility improvement after calibration", loc="left", pad=7, fontweight="bold")
    _format_axis(ax)
    _add_panel_label(ax, label)


def _panel_c_summary(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], tuple[float, float, int]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model_variant"], row["modality"], int(row["hop"]))].append(_float(row, "mean_cosine"))
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
    for axis, dataset in zip(axs, DATASETS):
        for modality, color in (("text", TEXT_BLUE), ("visual", VISUAL_GREEN)):
            for variant, linestyle, variant_label in (("full", "-", "Full"), ("no_semantic_anchor", "--", "w/o Anchor")):
                means = []
                spreads = []
                valid = []
                for hop in HOPS:
                    mean, spread, count = summary.get((dataset, variant, modality, hop), (np.nan, np.nan, 0))
                    means.append(mean)
                    spreads.append(spread)
                    valid.append(count > 0)
                means_array = np.asarray(means, dtype=float)
                spread_array = np.asarray(spreads, dtype=float)
                mask = np.isfinite(means_array) & np.asarray(valid, dtype=bool)
                if not mask.any():
                    continue
                axis.plot(np.asarray(HOPS)[mask], means_array[mask], color=color, linestyle=linestyle, marker="o", markersize=3.0, linewidth=1.45 if variant == "full" else 1.15, alpha=1.0 if variant == "full" else 0.78, zorder=3)
                axis.fill_between(np.asarray(HOPS)[mask], means_array[mask] - spread_array[mask], means_array[mask] + spread_array[mask], color=color, alpha=0.08 if variant == "full" else 0.04, linewidth=0, zorder=1)
        axis.set_xticks(HOPS, labels=[str(hop) for hop in HOPS])
        axis.set_xlim(-0.05, 3.05)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Hop order k", labelpad=4)
        axis.set_title(dataset, loc="left", pad=7, fontweight="bold")
        _format_axis(axis)
    axs[0].set_ylabel("Semantic retention", labelpad=4)
    handles = [
        Line2D([0], [0], color=TEXT_BLUE, linestyle="-", marker="o", linewidth=1.45, markersize=3.0, label="Text, Full"),
        Line2D([0], [0], color=TEXT_BLUE, linestyle="--", marker="o", linewidth=1.15, markersize=3.0, label="Text, w/o Anchor"),
        Line2D([0], [0], color=VISUAL_GREEN, linestyle="-", marker="o", linewidth=1.45, markersize=3.0, label="Visual, Full"),
        Line2D([0], [0], color=VISUAL_GREEN, linestyle="--", marker="o", linewidth=1.15, markersize=3.0, label="Visual, w/o Anchor"),
    ]
    axs[-1].legend(handles=handles, loc=legend_loc, bbox_to_anchor=legend_anchor, ncol=2, columnspacing=0.8, handletextpad=0.35, borderaxespad=0.15)
    _add_panel_label(axs[0], label)


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


def main() -> int:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or data_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    panel_a_rows = _read_csv(data_dir / "panel_a_relation_alignment.csv")
    panel_b_rows = _read_csv(data_dir / "panel_b_neighborhood_compatibility.csv")
    panel_c_rows = _read_csv(data_dir / "panel_c_semantic_retention.csv")
    if not panel_a_rows and not panel_b_rows and not panel_c_rows:
        print(f"[figure3-plot] no generated CSV data found in {data_dir}", file=sys.stderr)

    # Standalone panel (a).
    fig_a, ax_a = plt.subplots(figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4))
    fig_a.subplots_adjust(left=0.17, right=0.98, bottom=0.22, top=0.83)
    if panel_a_rows:
        _panel_a(ax_a, panel_a_rows)
    else:
        _make_empty_message(ax_a, "No Panel (a) data")
    _save_formats(fig_a, output_dir, "figure3a_relation_calibration")

    # Standalone panel (b).
    fig_b, ax_b = plt.subplots(figsize=(PANEL_WIDTH_MM / 25.4, PANEL_HEIGHT_MM / 25.4))
    fig_b.subplots_adjust(left=0.18, right=0.98, bottom=0.24, top=0.83)
    if panel_b_rows:
        _panel_b(ax_b, panel_b_rows)
    else:
        _make_empty_message(ax_b, "No Panel (b) data")
    _save_formats(fig_b, output_dir, "figure3b_local_compatibility")

    # Standalone panel (c), kept as a two-subplot figure matching the combined panel.
    fig_c, axs_c = plt.subplots(1, 2, figsize=(122 / 25.4, PANEL_HEIGHT_MM / 25.4), sharey=True)
    fig_c.subplots_adjust(left=0.10, right=0.99, bottom=0.21, top=0.74, wspace=0.24)
    if panel_c_rows:
        _panel_c(list(axs_c), panel_c_rows, legend_anchor=(0.5, 1.22), legend_loc="lower center")
    else:
        for axis in axs_c:
            _make_empty_message(axis, "No Panel (c) data")
    _alignment_gate(fig_c, list(axs_c), output_dir, "figure3c_semantic_retention")
    _save_formats(fig_c, output_dir, "figure3c_semantic_retention")

    # Combined Figure 3: equal-width top panels and a full-width bottom panel.
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.12), hspace=0.48, wspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    bottom = grid[1, :].subgridspec(1, 2, wspace=0.24)
    ax_c1 = fig.add_subplot(bottom[0, 0])
    ax_c2 = fig.add_subplot(bottom[0, 1], sharey=ax_c1)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.89)
    if panel_a_rows:
        _panel_a(ax_a, panel_a_rows)
    else:
        _make_empty_message(ax_a, "No Panel (a) data")
    if panel_b_rows:
        _panel_b(ax_b, panel_b_rows)
    else:
        _make_empty_message(ax_b, "No Panel (b) data")
    if panel_c_rows:
        _panel_c([ax_c1, ax_c2], panel_c_rows)
    else:
        _make_empty_message(ax_c1, "No Panel (c) data")
        _make_empty_message(ax_c2, "No Panel (c) data")
    ax_c2.tick_params(labelleft=False)
    _alignment_gate(fig, [ax_a, ax_b, ax_c1, ax_c2], output_dir, "figure3_mechanism")
    _save_formats(fig, output_dir, "figure3_mechanism")

    # Record a small plotting manifest without replacing the data-generation audit.
    plot_manifest = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "panel_rows": {"a": len(panel_a_rows), "b": len(panel_b_rows), "c": len(panel_c_rows)},
        "exports": [
            "figure3_mechanism.png", "figure3_mechanism.tiff", "figure3_mechanism.pdf", "figure3_mechanism.svg",
            "figure3a_relation_calibration.png", "figure3a_relation_calibration.tiff", "figure3a_relation_calibration.pdf", "figure3a_relation_calibration.svg",
            "figure3b_local_compatibility.png", "figure3b_local_compatibility.tiff", "figure3b_local_compatibility.pdf", "figure3b_local_compatibility.svg",
            "figure3c_semantic_retention.png", "figure3c_semantic_retention.tiff", "figure3c_semantic_retention.pdf", "figure3c_semantic_retention.svg",
        ],
    }
    (output_dir / "plot_manifest.json").write_text(json.dumps(plot_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[figure3-plot] wrote combined and standalone exports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
