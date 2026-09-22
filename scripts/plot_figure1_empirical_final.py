#!/usr/bin/env python3
"""Render the final paper-style Figure 1 from existing empirical outputs.

This script is read-only with respect to model checkpoints: it consumes the
already exported relation, drift, and legacy aggregation summaries and does
not start training or load a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "figure1_empirical_motivation"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "figure1_empirical_motivation_final"

# Colors follow the supplied reference figures: muted violet for text,
# blue-green for visual features, restrained gray for shared structure, and a
# small orange accent for cross-modal discrepancy markers.
TEXT_BLUE = "#7569A8"
VISUAL_GREEN = "#4F9B8B"
NEUTRAL = "#B7B8B7"
NEUTRAL_DARK = "#5F6870"
INSET_GRAY = "#A4AAAE"
DISCREPANCY_RED = "#D07A43"
INK = "#202326"
GRID = "#D9DDDE"

DATASET_ORDER = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
HOPS = (0, 1, 2, 3)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _number(row: dict[str, str], key: str, path: Path) -> float:
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
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
        # Keep an explicit publication-safe sans fallback for environments
        # where the preferred serif family is unavailable.
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.2,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.0,
        "axes.labelweight": "bold",
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.75,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "lines.solid_capstyle": "round",
    })


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(-0.14, 1.055, f"({label})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.0, fontweight="bold", color=INK, clip_on=False)


def _format_axis(ax: Any) -> None:
    ax.tick_params(length=2.5, width=0.65, pad=2.5)
    ax.grid(axis="y", color=GRID, linewidth=0.45, linestyle=(0, (1.2, 2.0)),
            alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _load_inset_points(path: Path, dataset: str, max_points: int) -> np.ndarray:
    """Read only one deterministic, bounded sample for the relation inset."""
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != dataset:
                continue
            points.append((_number(row, "text_rank_percentile", path),
                           _number(row, "visual_rank_percentile", path)))
    if not points:
        raise ValueError(f"No edge records for inset dataset {dataset!r} in {path}")
    values = np.asarray(points, dtype=np.float64)
    if len(values) > max_points:
        indices = np.linspace(0, len(values) - 1, max_points, dtype=np.int64)
        values = values[indices]
    return values


def _panel_a(ax: Any, data_dir: Path, inset_dataset: str, inset_points: int) -> None:
    summary_path = data_dir / "relation_discrepancy_summary.csv"
    summary_rows = _read_csv(summary_path)
    summary = {row["dataset"]: row for row in summary_rows}
    missing = [name for name in DATASET_ORDER if name not in summary]
    if missing:
        raise ValueError(f"Relation summary is missing datasets: {missing}")
    values = np.asarray([
        _number(summary[name], "median_absolute_rank_gap", summary_path) for name in DATASET_ORDER
    ], dtype=np.float64)
    rho = np.asarray([_number(summary[name], "spearman_rho", summary_path) for name in DATASET_ORDER])
    x = np.arange(len(DATASET_ORDER), dtype=np.float64)
    ax.bar(x, values, width=0.66, color=NEUTRAL, edgecolor=INK, linewidth=0.45, zorder=3)
    ax.set_xticks(x, labels=["Movies", "Toys", "Grocery", "ele-\nfashion", "Reddit-S"])
    ax.set_ylabel("Median text–visual edge-rank gap", labelpad=4)
    ax.set_ylim(0, float(values.max() * 1.23))
    ax.set_title("Relation-level semantic discrepancy", loc="left", pad=7, fontweight="bold")
    _format_axis(ax)

    # Small supporting relation inset. The main summary remains visually dominant.
    edge_path = data_dir / "relation_discrepancy_edges.csv"
    points = _load_inset_points(edge_path, inset_dataset, inset_points)
    # Keep the inset over the central upper plot area so it remains subordinate
    # to the summary bars.
    inset = ax.inset_axes([0.42, 0.70, 0.35, 0.26])
    inset.scatter(points[:, 0], points[:, 1], s=1.2, color=INSET_GRAY, alpha=0.24,
                  linewidths=0, rasterized=True, zorder=1)
    inset.plot([0, 1], [0, 1], color=NEUTRAL_DARK, lw=0.7, ls=(0, (3, 2)), zorder=2)
    examples = _read_json(data_dir / "relation_discrepancy_examples.json").get("datasets", {})
    for key in ("text_compatible_visual_dissimilar", "visual_compatible_text_dissimilar"):
        rows = examples.get(inset_dataset, {}).get(key, [])
        if rows:
            inset.scatter(
                [_number(item, "text_rank_percentile", data_dir / "relation_discrepancy_examples.json") for item in rows],
                [_number(item, "visual_rank_percentile", data_dir / "relation_discrepancy_examples.json") for item in rows],
                s=9, facecolors="white", edgecolors=DISCREPANCY_RED, linewidths=0.75, zorder=3,
            )
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    # The inset is intentionally axis-free at this compact size; the parent
    # panel and caption define the two rank axes, while the diagonal provides
    # the reference for cross-modal agreement.
    inset.axis("off")


def _panel_b(ax: Any, data_dir: Path) -> None:
    path = data_dir / "semantic_drift_curves.csv"
    rows = _read_csv(path)
    records: dict[tuple[str, str, int], float] = {}
    for row in rows:
        modality, hop = row["modality"].lower(), int(row["hop"])
        if modality in {"text", "visual"} and hop in HOPS:
            records[(row["dataset"], modality, hop)] = _number(row, "mean_drift_1_minus_cosine", path)
    datasets = sorted({key[0] for key in records})
    if not datasets:
        raise ValueError(f"No drift records found in {path}")
    hops = np.asarray(HOPS, dtype=np.float64)
    colors = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}
    for modality in ("text", "visual"):
        matrix = np.asarray([[records[(dataset, modality, hop)] for hop in HOPS] for dataset in datasets])
        center = np.mean(matrix, axis=0)
        lower, upper = np.quantile(matrix, [0.25, 0.75], axis=0)
        ax.fill_between(hops, lower, upper, color=colors[modality], alpha=0.16, linewidth=0, zorder=1)
        ax.plot(hops, center, color=colors[modality], marker="o", markersize=3.1,
                linewidth=1.75, markeredgecolor=colors[modality], markeredgewidth=0.35,
                label=modality.capitalize(), zorder=3)
    ax.set_xticks(hops, labels=[str(hop) for hop in HOPS])
    ax.set_xlim(-0.14, 3.14)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Propagation hop k", labelpad=4)
    ax.set_ylabel("Semantic drift from hop-0 state", labelpad=4)
    ax.set_title("Propagation-level semantic drift", loc="left", pad=7, fontweight="bold")
    ax.text(0.03, 0.98, "Text", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, fontweight="bold", color=TEXT_BLUE)
    ax.text(0.22, 0.98, "Visual", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, fontweight="bold", color=VISUAL_GREEN)
    _format_axis(ax)


def _aggregation_population_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _read_csv(path)
    population: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record_type") != "population" or row.get("modality") not in {"text", "visual"}:
            continue
        population.append({
            "dataset": row["dataset"],
            "modality": row["modality"],
            "mean": _number(row, "effective_order_mean", path),
            "q25": _number(row, "effective_order_q25", path),
            "median": _number(row, "effective_order_median", path),
            "q75": _number(row, "effective_order_q75", path),
            "sd": _number(row, "effective_order_sd", path),
        })
    available = [dataset for dataset in DATASET_ORDER if any(item["dataset"] == dataset for item in population)]
    if not available:
        raise ValueError(f"No population effective-order rows found in {path}")
    return population, available


def _panel_c(ax: Any, data_dir: Path) -> list[str]:
    path = data_dir / "aggregation_heterogeneity_summary.csv"
    population, available = _aggregation_population_rows(path)
    by_key = {(item["dataset"], item["modality"]): item for item in population}
    positions: list[float] = []
    stats: list[dict[str, Any]] = []
    colors: list[str] = []
    modality_offset = {"text": -0.18, "visual": 0.18}
    modality_color = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}
    for index, dataset in enumerate(available):
        for modality in ("text", "visual"):
            item = by_key.get((dataset, modality))
            if item is None:
                continue
            q1, med, q3 = item["q25"], item["median"], item["q75"]
            if not (0 <= q1 <= med <= q3 <= 3.0):
                raise ValueError(f"Invalid effective-order quantiles for {dataset} {modality}: {q1}, {med}, {q3}")
            positions.append(index + modality_offset[modality])
            stats.append({"label": f"{dataset}-{modality}", "whislo": q1, "q1": q1,
                          "med": med, "q3": q3, "whishi": q3, "mean": item["mean"],
                          "fliers": []})
            colors.append(modality_color[modality])

    artists = ax.bxp(stats, positions=positions, widths=0.28, showfliers=False, showmeans=True,
                     patch_artist=True, meanline=False,
                     boxprops={"linewidth": 0.75}, whiskerprops={"linewidth": 0.0},
                     capprops={"linewidth": 0.0}, medianprops={"color": INK, "linewidth": 1.0},
                     meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": INK,
                                "markersize": 3.0, "markeredgewidth": 0.65})
    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.58)
        box.set_edgecolor(INK)
        box.set_linewidth(0.75)
    for mean in artists["means"]:
        mean.set_zorder(4)
    ax.set_xticks(np.arange(len(available)), labels=available)
    ax.set_xlim(-0.58, len(available) - 0.42)
    # The stored population summaries occupy a narrow neighborhood around
    # orders 1.7–1.8. A zoomed y-range makes the IQR and mean differences
    # legible while retaining the order units on the axis.
    ax.set_ylim(1.5, 1.95)
    ax.set_yticks([1.5, 1.7, 1.9])
    ax.set_ylabel("Normalized effective hop order", labelpad=4)
    ax.set_title("Aggregation-level\nheterogeneous\nstructural-context preference", loc="left", pad=7, fontweight="bold")
    # Direct labels avoid a repeated legend and retain the modality encoding.
    ax.text(0.60, 0.98, "Text", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", color=TEXT_BLUE)
    ax.text(0.78, 0.98, "Visual", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", color=VISUAL_GREEN)
    _format_axis(ax)
    return available


def _alignment_gate(fig: Any, axes: list[Any], output_dir: Path) -> None:
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if skill_scripts.is_dir() and str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    visible_labels = {
        text.get_text() for axis in axes for text in axis.texts if text.get_text() in {"(a)", "(b)", "(c)"}
    }
    if visible_labels != {"(a)", "(b)", "(c)"}:
        raise ValueError("The combined figure must contain visible (a), (b), and (c) panel labels")
    fig.canvas.draw()
    report = require_matplotlib_panel_alignment(
        fig,
        json_out=output_dir / "figure1_empirical_final.alignment.json",
        overlay_svg=output_dir / "figure1_empirical_final.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
        axes=axes,
    )
    print(f"[figure1-final] panel alignment: {report.get('verdict', 'checked')}", flush=True)


def _save_formats(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def _standalone(builder: Any, data_dir: Path, output_dir: Path, stem: str, label: str,
                width_mm: float, height_mm: float) -> None:
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    builder(ax, data_dir)
    _add_panel_label(ax, label)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.83)
    _save_formats(fig, output_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inset-dataset", choices=("Movies", "Grocery", "Toys", "ele-fashion", "Reddit-S"),
                        default="Movies")
    parser.add_argument("--inset-points", type=int, default=1200)
    args = parser.parse_args()
    if args.inset_points < 1:
        parser.error("--inset-points must be positive")
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    builders = (
        (lambda ax, data: _panel_a(ax, data, args.inset_dataset, args.inset_points), "a",
         "figure1a_relation_discrepancy_final", 98.0, 80.0),
        (_panel_b, "b", "figure1b_propagation_drift_final", 96.0, 80.0),
        (_panel_c, "c", "figure1c_aggregation_heterogeneity_final", 110.0, 80.0),
    )
    for builder, label, stem, width_mm, height_mm in builders:
        _standalone(builder, data_dir, output_dir, stem, label, width_mm, height_mm)

    figure_width_mm = 180
    figure_height_mm = 88
    fig, axes = plt.subplots(1, 3, figsize=(figure_width_mm / 25.4, figure_height_mm / 25.4),
                             constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.76, wspace=0.44)
    _panel_a(axes[0], data_dir, args.inset_dataset, args.inset_points)
    _panel_b(axes[1], data_dir)
    available = _panel_c(axes[2], data_dir)
    for axis, label in zip(axes, "abc"):
        _add_panel_label(axis, label)
    _alignment_gate(fig, list(axes), output_dir)
    _save_formats(fig, output_dir, "figure1_empirical_final")

    (output_dir / "figure1_empirical_final_manifest.json").write_text(
        json.dumps({
            "figure": "Figure 1. Empirical observations behind CoSI-MAG",
            "panel_a": "Five-dataset median text/visual edge-rank gap with a bounded Movies rank inset",
            "panel_b": "Across-dataset mean of per-dataset mean 1-cosine drift with dataset-level IQR ribbon",
            "panel_c": "Legacy read-only population effective-order IQR summaries and means",
            "panel_c_available_datasets": available,
            "panel_c_missing_datasets": [name for name in DATASET_ORDER if name not in available],
            "panel_c_source": "aggregation_heterogeneity_summary.csv; existing read-only final/full checkpoint summaries",
            "checkpoint_loaded_by_plotter": False,
            "training_started": False,
            "formal_model_files_modified": False,
            "inset_dataset": args.inset_dataset,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[figure1-final] aggregation summary coverage: {', '.join(available)}", flush=True)
    print(f"[figure1-final] wrote final panels and combined figure to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
