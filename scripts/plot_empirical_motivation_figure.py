#!/usr/bin/env python3
"""Render Figure 1 panels and a horizontal combined figure from empirical CSVs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "figure1_empirical_motivation"
TEXT_BLUE = "#587FA8"
VISUAL_GREEN = "#65977E"
SHARED_GRAY = "#B8BDC2"
BAR_GRAY = "#B8A78F"
INK = "#24282C"
GRID = "#E5E8EA"
DATASET_ORDER = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
ORDER_DATASETS = ("Movies", "Grocery")
HOPS = (0, 1, 2, 3)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}. Run generate_empirical_motivation_data.py first.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _value(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {key!r} in {path}: {row.get(key)!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite numeric field {key!r} in {path}: {value}")
    return value


def _configure_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "lines.solid_capstyle": "round",
    })


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(-0.14, 1.055, f"({label})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.2, fontweight="bold", color=INK, clip_on=False)


def _format_axis(ax: Any) -> None:
    ax.tick_params(length=2.5, width=0.65, pad=2.4)
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def _panel_a(ax: Any, data_dir: Path) -> None:
    path = data_dir / "relation_discrepancy_summary.csv"
    rows = _read_csv(path)
    by_dataset = {row["dataset"]: row for row in rows}
    missing = [name for name in DATASET_ORDER if name not in by_dataset]
    if missing:
        raise ValueError(f"Relation summary is missing required datasets: {missing}")
    values = np.asarray([
        _value(by_dataset[name], "median_absolute_rank_gap", path) for name in DATASET_ORDER
    ], dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("Median absolute rank gaps must be non-negative")

    x = np.arange(len(DATASET_ORDER))
    bars = ax.bar(x, values, width=0.66, color=BAR_GRAY, edgecolor="none", zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values.max() * 0.06, 0.018),
                f"{value:.2f}", ha="center", va="bottom", fontsize=7.2, color=INK)
    ax.set_xticks(x, labels=list(DATASET_ORDER))
    # A short line break keeps the exact dataset name readable at combined-panel width.
    ax.set_xticklabels(["Movies", "Toys", "Grocery", "ele-\nfashion", "Reddit-S"])
    ax.set_ylabel("Median cross-modal edge-rank gap", labelpad=4)
    ax.set_ylim(0, float(values.max() * 1.23))
    ax.set_title("Relation-level discrepancy", loc="left", pad=9, fontweight="normal")
    _format_axis(ax)


def _panel_b(ax: Any, data_dir: Path) -> None:
    path = data_dir / "semantic_drift_curves.csv"
    rows = _read_csv(path)
    records: dict[tuple[str, str, int], float] = {}
    for row in rows:
        dataset = row["dataset"]
        modality = row["modality"].lower()
        hop = int(row["hop"])
        if modality not in ("text", "visual") or hop not in HOPS:
            continue
        records[(dataset, modality, hop)] = _value(row, "mean_drift_1_minus_cosine", path)
    datasets = sorted({key[0] for key in records})
    if not datasets:
        raise ValueError(f"No usable drift records found in {path}")
    hops = np.asarray(HOPS, dtype=np.float64)
    palette = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}
    for modality in ("text", "visual"):
        expected = [(dataset, modality, hop) for dataset in datasets for hop in HOPS]
        missing = [key for key in expected if key not in records]
        if missing:
            raise ValueError(f"Drift curve is incomplete for {modality}: {missing[:4]}")
        across_dataset = np.asarray([
            [records[(dataset, modality, hop)] for hop in HOPS] for dataset in datasets
        ], dtype=np.float64)
        mean = np.mean(across_dataset, axis=0)
        low, high = np.quantile(across_dataset, [0.25, 0.75], axis=0)
        ax.fill_between(hops, low, high, color=palette[modality], alpha=0.12, linewidth=0, zorder=1)
        ax.plot(hops, mean, color=palette[modality], marker="o", markersize=3.2,
                linewidth=1.55, label=modality.capitalize(), zorder=3)

    ax.set_xticks(hops, labels=[str(hop) for hop in HOPS])
    ax.set_xlim(-0.14, 3.14)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Propagation hop k", labelpad=4)
    ax.set_ylabel("Mean drift: 1 - cosine(H0, Sk)", labelpad=4)
    ax.set_title("Propagation drift", loc="left", pad=9, fontweight="normal")
    ax.text(0.57, 1.14, "Text", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, color=TEXT_BLUE)
    ax.text(0.78, 1.14, "Visual", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, color=VISUAL_GREEN)
    _format_axis(ax)


def _panel_c(ax: Any, data_dir: Path) -> None:
    path = data_dir / "order_utility_heterogeneity.csv"
    rows = _read_csv(path)
    records: dict[tuple[str, str, int], float] = {}
    for row in rows:
        dataset, modality, hop = row["dataset"], row["modality"].lower(), int(row["hop"])
        if dataset in ORDER_DATASETS and modality in ("text", "visual") and hop in HOPS:
            records[(dataset, modality, hop)] = _value(row, "fraction_best", path)
    expected = [(dataset, modality, hop) for dataset in ORDER_DATASETS
                for modality in ("text", "visual") for hop in HOPS]
    missing = [key for key in expected if key not in records]
    if missing:
        raise ValueError(f"Order-utility CSV is incomplete: {missing}")
    for dataset in ORDER_DATASETS:
        for modality in ("text", "visual"):
            mass = np.asarray([records[(dataset, modality, hop)] for hop in HOPS], dtype=np.float64)
            if np.any((mass < 0) | (mass > 1)) or not np.isclose(mass.sum(), 1.0, atol=1e-8):
                raise ValueError(f"Best-order fractions do not sum to one for {dataset} {modality}: {mass}")

    centers = {"Movies": np.arange(4, dtype=np.float64), "Grocery": np.arange(4, dtype=np.float64) + 5.0}
    offsets = {"text": -0.115, "visual": 0.115}
    width = 0.21
    palette = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}
    y_max = max(0.62, max(records.values()) * 1.18)
    for dataset in ORDER_DATASETS:
        for modality in ("text", "visual"):
            heights = [records[(dataset, modality, hop)] for hop in HOPS]
            ax.bar(centers[dataset] + offsets[modality], heights, width=width,
                   color=palette[modality], edgecolor="none", label=modality.capitalize(), zorder=3)
    ax.axvline(4.5, color=SHARED_GRAY, linewidth=0.8, zorder=1)
    ax.set_xlim(-0.55, 8.55)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, np.floor(y_max * 10) / 10 + 0.01, 0.1))
    ax.set_xticks([0, 1, 2, 3, 5, 6, 7, 8], labels=["0", "1", "2", "3", "0", "1", "2", "3"])
    ax.set_ylabel("Fraction of validation nodes", labelpad=4)
    ax.set_title("Multi-order utility", loc="left", pad=9, fontweight="normal")
    ax.text(1.5, -0.19, "Movies", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.2)
    ax.text(6.5, -0.19, "Grocery", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.2)
    ax.text(0.57, 1.14, "Text", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, color=TEXT_BLUE)
    ax.text(0.78, 1.14, "Visual", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, color=VISUAL_GREEN)
    _format_axis(ax)


def _alignment_gate(fig: Any, axes: list[Any], output_dir: Path) -> None:
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if skill_scripts.is_dir() and str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    labels = [
        text.get_text()
        for axis in axes
        for text in axis.texts
        if text.get_text() in {"(a)", "(b)", "(c)"}
    ]
    if set(labels) != {"(a)", "(b)", "(c)"}:
        raise ValueError("Combined figure is missing one or more visible (a)/(b)/(c) panel labels")
    fig.canvas.draw()
    report = require_matplotlib_panel_alignment(
        fig,
        json_out=output_dir / "figure1_empirical_motivation.alignment.json",
        overlay_svg=output_dir / "figure1_empirical_motivation.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        # The visible requested labels include parentheses; the bundled auditor
        # recognizes bare letters only, so verify the visible labels above and
        # use it here for panel geometry and gutter alignment.
        require_panel_labels=False,
        strict=True,
    )
    print(f"[figure1-plot] panel alignment: {report.get('verdict', 'checked')}", flush=True)


def _save_formats(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def _make_standalone(builder: Any, data_dir: Path, output_dir: Path, label: str,
                     stem: str, figsize_mm: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(figsize_mm[0] / 25.4, figsize_mm[1] / 25.4))
    builder(ax, data_dir)
    _add_panel_label(ax, label)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.84)
    _save_formats(fig, output_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to --data-dir; all PNG, PDF and SVG files are written there.")
    args = parser.parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    output_dir = data_dir if args.output_dir is None else (args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    panels = (
        (_panel_a, "a", "figure1a_relation_discrepancy", (96.0, 78.0)),
        (_panel_b, "b", "figure1b_propagation_drift", (92.0, 78.0)),
        (_panel_c, "c", "figure1c_order_heterogeneity", (108.0, 78.0)),
    )
    for builder, label, stem, figsize_mm in panels:
        _make_standalone(builder, data_dir, output_dir, label, stem, figsize_mm)

    width_mm = 180
    height_mm = 88
    fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4, height_mm / 25.4), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.315, top=0.835, wspace=0.43)
    _panel_a(axes[0], data_dir)
    _panel_b(axes[1], data_dir)
    _panel_c(axes[2], data_dir)
    for axis, label in zip(axes, "abc"):
        _add_panel_label(axis, label)
    _alignment_gate(fig, list(axes), output_dir)
    _save_formats(fig, output_dir, "figure1_empirical_motivation")
    print(f"[figure1-plot] wrote standalone panels and combined figure to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
