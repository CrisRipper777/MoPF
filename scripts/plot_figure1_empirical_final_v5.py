#!/usr/bin/env python3
"""Render the final empirical-motivation Figure 1 (v5).

The plotter consumes frozen empirical CSV outputs and the node-level choices
from the lightweight, model-independent linear probes.  It never loads a
CoSI-MAG checkpoint and never starts a base-model training run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "figure1_empirical_motivation"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "figure1_empirical_motivation_final"

TEXT_BLUE = "#4C78A8"
VISUAL_GREEN = "#59A14F"
DISCREPANCY = "#D78350"
RELATION_MEDIAN = "#D78350"
RELATION_RANGE = "#E6B99B"
NEUTRAL = "#8A8F93"
NEUTRAL_LIGHT = "#D7DADC"
INK = "#24272A"
GRID = "#E7E9EB"

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
HOPS = (0, 1, 2, 3)
MODALITIES = ("text", "visual")
DATASET_LABELS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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
        "axes.linewidth": 0.65,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "lines.solid_capstyle": "round",
    })


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(-0.13, 1.045, f"({label})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.0, fontweight="bold",
            color=INK, clip_on=False)


def _format_axis(ax: Any, grid_axis: str = "y") -> None:
    ax.tick_params(length=2.4, width=0.6, pad=2.4)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.45,
            linestyle=(0, (1.2, 2.0)), alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _save_formats(fig: Any, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=600)
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def _make_figure(width_mm: float, height_mm: float, top: float = 0.82) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=top)
    return fig, ax


def _relation_gap_values(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in _read_csv(path):
        dataset = row.get("dataset", "")
        if dataset not in DATASETS:
            continue
        if row.get("absolute_rank_gap", ""):
            gap = _number(row, "absolute_rank_gap", path)
        else:
            gap = abs(
                _number(row, "text_rank_percentile", path)
                - _number(row, "visual_rank_percentile", path)
            )
        values[dataset].append(100.0 * gap)
    result: dict[str, np.ndarray] = {}
    for dataset in DATASETS:
        if not values.get(dataset):
            raise ValueError(f"No relation edge gaps found for {dataset}")
        result[dataset] = np.asarray(values[dataset], dtype=np.float64)
        if not np.isfinite(result[dataset]).all():
            raise ValueError(f"Non-finite relation edge gaps for {dataset}")
    return result


def _panel_a(ax: Any, data_dir: Path) -> None:
    path = data_dir / "relation_discrepancy_edges.csv"
    values = _relation_gap_values(path)
    stats: list[tuple[float, float, float, float, float]] = []
    for dataset in DATASETS:
        q10, q25, median, q75, q90 = np.quantile(values[dataset], [0.10, 0.25, 0.50, 0.75, 0.90])
        stats.append((float(q10), float(q25), float(median), float(q75), float(q90)))

    y = np.arange(len(DATASETS), dtype=np.float64)
    for yi, (q10, q25, median, q75, q90) in zip(y, stats):
        ax.hlines(yi, q10, q90, color=RELATION_RANGE, linewidth=1.6, alpha=0.85, zorder=2)
        ax.hlines(yi, q25, q75, color=RELATION_RANGE, linewidth=4.2, alpha=0.95, zorder=3)
        ax.scatter(median, yi, s=29, facecolor="white", edgecolor=RELATION_MEDIAN,
                   linewidth=1.15, zorder=4)

    max_value = max(item[-1] for item in stats)
    ax.axvline(0.0, color=NEUTRAL, linewidth=0.65, linestyle=(0, (2, 2)), zorder=1)
    ax.set_yticks(y, labels=DATASET_LABELS)
    ax.invert_yaxis()
    # Leave a narrow margin so the perfect-agreement reference at x=0 is
    # visible as a light dashed line rather than merging with the spine.
    ax.set_xlim(-max(1.0, max_value * 0.035), max(1.0, max_value * 1.08))
    ax.set_xlabel("Text–Visual edge-rank gap (%)", labelpad=4)
    ax.set_title("Relation-level semantic discrepancy", loc="left", pad=7,
                 fontweight="bold")
    _format_axis(ax, grid_axis="x")


def _drift_records(path: Path) -> dict[tuple[str, str, int], float]:
    records: dict[tuple[str, str, int], float] = {}
    for row in _read_csv(path):
        dataset = row.get("dataset", "")
        modality = row.get("modality", "").lower()
        hop = int(row["hop"])
        if dataset in DATASETS and modality in MODALITIES and hop in HOPS:
            records[(dataset, modality, hop)] = 100.0 * _number(
                row, "mean_drift_1_minus_cosine", path
            )
    for dataset in DATASETS:
        for modality in MODALITIES:
            for hop in HOPS:
                if (dataset, modality, hop) not in records:
                    raise ValueError(f"Missing drift record for {dataset}/{modality}/hop {hop}")
    return records


def _modality_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=TEXT_BLUE, marker="o", linewidth=1.6,
               markersize=3.2, label="Text"),
        Line2D([0], [0], color=VISUAL_GREEN, marker="o", linewidth=1.6,
               markersize=3.2, label="Visual"),
    ]


def _panel_b(ax: Any, data_dir: Path) -> None:
    path = data_dir / "semantic_drift_curves.csv"
    records = _drift_records(path)
    hops = np.asarray(HOPS, dtype=np.float64)
    colors = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}
    centers: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        matrix = np.asarray([
            [records[(dataset, modality, hop)] for hop in HOPS] for dataset in DATASETS
        ], dtype=np.float64)
        # Keep the dataset-level trajectories visible as a very light context;
        # the colored mean line remains the only emphasized curve.
        for dataset_curve in matrix:
            ax.plot(hops, dataset_curve, color=NEUTRAL, linewidth=0.65,
                    linestyle=(0, (2, 2)), alpha=0.48, zorder=1)
        center = np.mean(matrix, axis=0)
        lower, upper = np.quantile(matrix, [0.25, 0.75], axis=0)
        centers[modality] = center
        ax.fill_between(hops, lower, upper, color=colors[modality], alpha=0.14,
                        linewidth=0, zorder=1)
        ax.plot(hops, center, color=colors[modality], marker="o", markersize=3.4,
                markeredgecolor=colors[modality], markeredgewidth=0.35,
                linewidth=1.65, zorder=3)

    ymax = max(float(np.max(centers[modality])) for modality in MODALITIES)
    ax.set_xticks(hops, labels=[str(hop) for hop in HOPS])
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(0.0, max(1.0, ymax * 1.20))
    ax.set_xlabel("Propagation hop k", labelpad=4)
    ax.set_ylabel("Representation drift from H₀ (%)", labelpad=4)
    ax.set_title("Propagation-level semantic drift", loc="left", pad=7,
                 fontweight="bold")
    ax.legend(handles=_modality_handles(), loc="upper left", ncol=2,
              borderaxespad=0.15, handletextpad=0.45, columnspacing=0.9)
    _format_axis(ax)


def _node_choice_records(path: Path) -> dict[tuple[str, str], np.ndarray]:
    choices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in _read_csv(path):
        dataset = row.get("dataset", "")
        modality = row.get("modality", "").lower()
        if dataset not in DATASETS or modality not in MODALITIES:
            continue
        choice = int(row["best_order"])
        if choice not in HOPS:
            raise ValueError(f"Invalid best order {choice} in {path}")
        choices[(dataset, modality)].append(choice)
    result: dict[tuple[str, str], np.ndarray] = {}
    for dataset in DATASETS:
        for modality in MODALITIES:
            key = (dataset, modality)
            if not choices.get(key):
                raise ValueError(f"Missing node-level order choices for {dataset}/{modality}")
            values = np.asarray(choices[key], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite node-level order choices for {dataset}/{modality}")
            result[key] = values
    return result


def _panel_c_violin(ax: Any, data_dir: Path) -> None:
    path = data_dir / "order_utility_node_choices.csv"
    choices = _node_choice_records(path)
    x = np.arange(len(DATASETS), dtype=np.float64)
    offsets = {"text": -0.18, "visual": 0.18}
    colors = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}
    for modality in MODALITIES:
        positions = x + offsets[modality]
        data = [choices[(dataset, modality)] for dataset in DATASETS]
        violins = ax.violinplot(
            data, positions=positions, widths=0.30, bw_method=0.14,
            showmeans=False, showmedians=False, showextrema=False, points=160,
        )
        for body in violins["bodies"]:
            body.set_facecolor(colors[modality])
            body.set_edgecolor(colors[modality])
            body.set_alpha(0.54)
            body.set_linewidth(0.55)
        for pos, values in zip(positions, data):
            q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.vlines(pos, q25, q75, color=INK, linewidth=1.15, zorder=4)
            ax.scatter(pos, median, s=21, facecolor="white", edgecolor=INK,
                       linewidth=0.75, zorder=5)

    ax.set_xticks(x, labels=DATASET_LABELS)
    ax.set_xlim(-0.58, len(DATASETS) - 0.42)
    ax.set_ylim(-0.25, 3.25)
    ax.set_yticks(HOPS)
    ax.set_xlabel("Dataset", labelpad=4)
    ax.set_ylabel("Preferred propagation order k*", labelpad=4)
    ax.set_title("Order heterogeneity", loc="left", pad=7, fontweight="bold")
    # Direct labels sit in the small y>3 whitespace and avoid an extra legend
    # box over the discrete violin bodies.
    ax.text(0.03, 0.98, "Text", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", color=TEXT_BLUE)
    ax.text(0.19, 0.98, "Visual", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", color=VISUAL_GREEN)
    _format_axis(ax)


def _summary_fractions(path: Path) -> dict[tuple[str, str, int], float]:
    fractions: dict[tuple[str, str, int], float] = {}
    for row in _read_csv(path):
        dataset = row.get("dataset", "")
        modality = row.get("modality", "").lower()
        if dataset not in DATASETS or modality not in MODALITIES:
            continue
        hop = int(row["hop"])
        if hop not in HOPS:
            continue
        fractions[(dataset, modality, hop)] = _number(row, "fraction_best", path)
    for dataset in DATASETS:
        for modality in MODALITIES:
            total = sum(fractions.get((dataset, modality, hop), 0.0) for hop in HOPS)
            if any((dataset, modality, hop) not in fractions for hop in HOPS):
                raise ValueError(f"Missing stacked fraction for {dataset}/{modality}")
            if not np.isfinite(total) or abs(total - 1.0) > 5e-6:
                raise ValueError(f"Best-order fractions do not sum to one for {dataset}/{modality}: {total}")
    return fractions


def _panel_c_stacked(ax: Any, data_dir: Path) -> None:
    path = data_dir / "order_utility_heterogeneity.csv"
    fractions = _summary_fractions(path)
    x = np.arange(len(DATASETS), dtype=np.float64)
    width = 0.32
    offsets = {"text": -0.19, "visual": 0.19}
    shade_colors = {
        "text": ("#DCE9F6", "#AFCBE8", "#77A7D9", "#3F7FBE"),
        "visual": ("#DCEFD8", "#B6DCA9", "#7FC97F", "#4C9F50"),
    }
    for modality in MODALITIES:
        bottom = np.zeros(len(DATASETS), dtype=np.float64)
        for hop, color in enumerate(shade_colors[modality]):
            heights = 100.0 * np.asarray([
                fractions[(dataset, modality, hop)] for dataset in DATASETS
            ], dtype=np.float64)
            ax.bar(x + offsets[modality], heights, width=width, bottom=bottom,
                   color=color, alpha=1.0, edgecolor="white",
                   linewidth=0.45, zorder=3)
            bottom += heights

    modality_handles = [
        Patch(facecolor=TEXT_BLUE, edgecolor=TEXT_BLUE, linewidth=0.35, label="Text"),
        Patch(facecolor=VISUAL_GREEN, edgecolor=VISUAL_GREEN, linewidth=0.35, label="Visual"),
    ]
    ax.set_xticks(x, labels=DATASET_LABELS)
    ax.set_xlim(-0.62, len(DATASETS) - 0.38)
    # Reserve a compact strip above the 100% bars for the modality key and note;
    # the tick labels remain the requested 0--100% scale.
    ax.set_ylim(0.0, 118.0)
    ax.set_yticks(np.linspace(0, 100, 6))
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"])
    ax.set_xlabel("Dataset", labelpad=4)
    ax.set_ylabel("Validation nodes (%)", labelpad=4)
    ax.set_title("Preferred hop-order heterogeneity", loc="left", pad=7,
                 fontweight="bold")
    ax.tick_params(axis="x", pad=4, labelsize=7.0)
    modality_legend = ax.legend(handles=modality_handles, loc="upper left",
                                bbox_to_anchor=(0.01, 0.965), ncol=2,
                                borderaxespad=0.0, handlelength=0.75,
                                handleheight=0.75, handletextpad=0.55,
                                columnspacing=0.6, borderpad=0.15,
                                fontsize=7.0)
    ax.text(0.99, 0.965, "Darker → higher hop order", transform=ax.transAxes,
            ha="right", va="top", color=NEUTRAL, fontsize=7.0)
    _format_axis(ax)


def _standalone(builder: Any, data_dir: Path, output_dir: Path, stem: str,
                label: str, width_mm: float, height_mm: float, top: float = 0.82) -> None:
    fig, ax = _make_figure(width_mm, height_mm, top=top)
    builder(ax, data_dir)
    _add_panel_label(ax, label)
    _save_formats(fig, output_dir, stem)


def _alignment_gate(fig: Any, axes: list[Any], output_dir: Path) -> None:
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if skill_scripts.is_dir() and str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    fig.canvas.draw()
    report = require_matplotlib_panel_alignment(
        fig,
        json_out=output_dir / "figure1_empirical_final_v5.alignment.json",
        overlay_svg=output_dir / "figure1_empirical_final_v5.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
        axes=axes,
        exemptions=[{
            "panels": ["a", "b", "c"],
            "checks": ["panel-width"],
            "reason": "Intentional unequal width ratios (0.95, 1.10, 1.25) requested for the v5 layout.",
        }],
    )
    print(f"[figure1-v5] panel alignment: {report.get('verdict', 'checked')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    _standalone(_panel_a, data_dir, output_dir,
                "figure1a_relation_discrepancy_final_v5", "a", 132.0, 88.0)
    _standalone(_panel_b, data_dir, output_dir,
                "figure1b_propagation_drift_final_v5", "b", 142.0, 88.0)
    _standalone(_panel_c_stacked, data_dir, output_dir,
                "figure1c_order_heterogeneity_final_v5", "c", 158.0, 88.0)

    fig, axes = plt.subplots(
        1, 3, figsize=(17.0, 4.8), constrained_layout=False,
        gridspec_kw={"width_ratios": (0.95, 1.10, 1.25)},
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.19, top=0.80, wspace=0.20)
    _panel_a(axes[0], data_dir)
    _panel_b(axes[1], data_dir)
    _panel_c_stacked(axes[2], data_dir)
    for axis, label in zip(axes, "abc"):
        _add_panel_label(axis, label)
    _alignment_gate(fig, list(axes), output_dir)
    _save_formats(fig, output_dir, "figure1_empirical_final_v5")

    manifest = {
        "figure": "Figure 1. Empirical observations behind CoSI-MAG (v5)",
        "panel_a": "Per-dataset distribution of 100*abs(text edge-rank - visual edge-rank), shown as terracotta median/IQR and 10-90% whisker",
        "panel_b": "Across-dataset mean of 100*(1-cosine(H0, S_k)) with dataset-level IQR ribbon and light dataset curves",
        "panel_b_line_statistic": "arithmetic mean across the five dataset-level means; IQR ribbon across dataset-level means",
        "panel_c": "Model-independent 100% stacked best-order fractions from identical per-hop linear probes",
        "figure_size_inches": [17.0, 4.8],
        "width_ratios": [0.95, 1.10, 1.25],
        "wspace": 0.20,
        "panel_c_probe_datasets": list(DATASETS),
        "panel_c_default": "100% stacked best-order fractions",
        "panel_c_node_source": "order_utility_node_choices.csv (used to produce the unchanged summary fractions)",
        "panel_c_summary_source": "order_utility_heterogeneity.csv",
        "probe_architecture": "torch.nn.Linear(input_dim, num_classes, bias=True)",
        "probe_optimizer": "Adam(full_batch)",
        "probe_epochs": 100,
        "probe_seed": 42,
        "checkpoint_loaded_by_plotter": False,
        "training_started": False,
        "formal_model_files_modified": False,
        "legacy_eta_data_reserved_for": "Figure 4 mechanism analysis",
    }
    (output_dir / "figure1_empirical_final_v5_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[figure1-v5] wrote v5 panels and combined figure to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
