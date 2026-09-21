#!/usr/bin/env python3
"""Render publication-style Figure 4 from mechanism-verification outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# The figure-alignment auditor is supplied by the active scientific-figure
# workflow. Resolve it from the user's local skill installation without
# baking a machine-specific absolute path into this project script.
_figure_audit_scripts = Path.home() / ".codex/skills/nature-figure/scripts"
if _figure_audit_scripts.is_dir():
    sys.path.insert(0, str(_figure_audit_scripts))
from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/mechanism_verification"
DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
TEXT_BLUE = "#2F6FB3"
VISUAL_GREEN = "#3B9960"
ZERO_GRAY = "#777777"
INK = "#20252B"
MODALITY_COLOR = {"Text": TEXT_BLUE, "Visual": VISUAL_GREEN}
SEED_MARKERS = {42: "o", 43: "s", 44: "^"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required mechanism output is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Mechanism CSV is empty: {path}")
    return rows


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.1,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite value in field {key}: {value}")
    return value


def _plot_seed_summary_panel(
    ax: plt.Axes,
    per_seed: list[dict[str, str]],
    summary: list[dict[str, str]],
    *,
    series: tuple[tuple[str, str, str, str, str], ...],
    ylabel: str,
) -> None:
    x = np.arange(1, len(DATASETS) + 1, dtype=float)
    offsets = np.linspace(-0.13, 0.13, len(series)) if len(series) > 1 else np.zeros(1)
    for series_index, (label, seed_key, color_key, mean_key, _frozen_sd_key) in enumerate(series):
        color = MODALITY_COLOR[color_key]
        offset = offsets[series_index]
        for dataset_index, dataset in enumerate(DATASETS):
            records = [
                row for row in per_seed
                if row["dataset"] == dataset
                and ("modality" not in row or row["modality"] == label)
            ]
            summary_row = next(
                row for row in summary
                if row["dataset"] == dataset
                and ("modality" not in row or row["modality"] == label)
            )
            if len(records) != 3:
                raise ValueError(f"Expected three seed points for {dataset}/{label}, got {len(records)}")
            center = x[dataset_index] + offset
            for seed_position, row in enumerate(sorted(records, key=lambda item: int(item["seed"]))):
                jitter = (seed_position - 1) * 0.032
                ax.scatter(
                    center + jitter,
                    _float(row, seed_key),
                    s=21,
                    marker=SEED_MARKERS[int(row["seed"])],
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.45,
                    alpha=0.9,
                    zorder=4,
                )
            seed_values = np.asarray([_float(row, seed_key) for row in records], dtype=np.float64)
            # Preserve the frozen mean; Figure 4 bars use the paper's
            # population-SD convention, calculated from the three seed rows.
            population_sd = float(np.std(seed_values, ddof=0))
            ax.errorbar(
                center,
                _float(summary_row, mean_key),
                yerr=population_sd,
                fmt="D",
                markersize=5.3,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.6,
                ecolor=color,
                elinewidth=1.15,
                capsize=2.2,
                capthick=1.0,
                zorder=5,
            )
    ax.axhline(0.0, color=ZERO_GRAY, linewidth=0.85, zorder=1)
    ax.set_xlim(0.55, len(DATASETS) + 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=25, ha="right", rotation_mode="anchor", fontsize=6.1)
    for label in ax.get_xticklabels():
        if label.get_text() in {"ele-fashion", "Reddit-S"}:
            label.set_rotation(60)
            label.set_rotation_mode("anchor")
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6E8EA", linewidth=0.55)
    ax.legend(
        handles=[Line2D([], [], color=MODALITY_COLOR[key], marker="D", markersize=5, linewidth=1.0, label=label)
                 for label, _, key, _, _ in series],
        loc="best",
        ncol=1,
        handlelength=1.2,
    )


def _plot_stage3(
    ax: plt.Axes,
    node_path: Path,
    per_seed: list[dict[str, str]],
) -> None:
    with np.load(node_path, allow_pickle=False) as loaded:
        required = {"dataset", "seed", "modality", "node_id", "D"}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"Stage III NPZ missing fields: {sorted(missing)}")
        datasets = loaded["dataset"].astype(str)
        seeds = np.asarray(loaded["seed"], dtype=np.int16)
        modalities = loaded["modality"].astype(str)
        values = np.asarray(loaded["D"], dtype=np.float64)
    if not (datasets.size == seeds.size == modalities.size == values.size):
        raise ValueError("Stage III node-level NPZ arrays have inconsistent lengths")
    if not np.isfinite(values).all() or np.any(values < -1e-8) or np.any(values > 1.0 + 1e-8):
        raise ValueError("Stage III node-level deviations are non-finite or outside [0,1]")

    x = np.arange(1, len(DATASETS) + 1, dtype=float)
    offset = 0.17
    for dataset_index, dataset in enumerate(DATASETS):
        for modality in ("Text", "Visual"):
            mask = (datasets == dataset) & (modalities == modality)
            group = values[mask]
            if group.size == 0:
                raise ValueError(f"No node-level deviations for {dataset}/{modality}")
            position = x[dataset_index] + (-offset if modality == "Text" else offset)
            # Matplotlib's KDE is quadratic in sample size. The NPZ and all
            # summary statistics retain every node; only this rendering uses a
            # deterministic bounded sample for the violin density.
            if group.size > 5000:
                rng = np.random.RandomState(20260920 + 100 * dataset_index + (0 if modality == "Text" else 1))
                violin_values = group[rng.permutation(group.size)[:5000]]
            else:
                violin_values = group
            color = MODALITY_COLOR[modality]
            if violin_values.size > 1 and float(np.ptp(violin_values)) > 1e-10:
                parts = ax.violinplot(
                    violin_values,
                    positions=[position],
                    widths=0.29,
                    points=80,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(INK)
                    body.set_linewidth(0.5)
                    body.set_alpha(0.45)
            median = float(np.median(group))
            ax.scatter(
                [position], [median], s=24, marker="o", facecolor="white",
                edgecolor=INK, linewidth=0.8, zorder=5,
            )
            seed_rows = [row for row in per_seed if row["dataset"] == dataset and row["modality"] == modality]
            if len(seed_rows) != 3:
                raise ValueError(f"Expected three Stage III seed means for {dataset}/{modality}")
            for seed_position, row in enumerate(sorted(seed_rows, key=lambda item: int(item["seed"]))):
                jitter = (seed_position - 1) * 0.034
                ax.scatter(
                    [position + jitter], [_float(row, "mean_D")],
                    s=22, marker=SEED_MARKERS[int(row["seed"])],
                    facecolor=color, edgecolor=INK, linewidth=0.5, zorder=6,
                )
    ax.axhline(0.0, color=ZERO_GRAY, linewidth=0.9, zorder=1)
    ax.set_xlim(0.55, len(DATASETS) + 0.45)
    ax.set_ylim(-0.025, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=25, ha="right", rotation_mode="anchor", fontsize=6.1)
    for label in ax.get_xticklabels():
        if label.get_text() in {"ele-fashion", "Reddit-S"}:
            label.set_rotation(60)
            label.set_rotation_mode("anchor")
    ax.set_ylabel("Reallocation from uniform composition")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6E8EA", linewidth=0.55)


def build_figure(output_root: Path) -> list[Path]:
    stage1 = _read_csv(output_root / "stage1_relation_calibration_per_seed.csv")
    stage1_summary = _read_csv(output_root / "stage1_relation_calibration_summary.csv")
    stage2 = _read_csv(output_root / "stage2_semantic_retention_by_hop.csv")
    stage2_summary = _read_csv(output_root / "stage2_semantic_retention_summary.csv")
    stage3 = _read_csv(output_root / "stage3_adaptive_reallocation_per_seed.csv")
    node_path = output_root / "stage3_adaptive_reallocation_nodes.npz"
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.65), facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.99, top=0.84, bottom=0.31, wspace=0.60)

    _plot_seed_summary_panel(
        axes[0], stage1, stage1_summary,
        series=(
            ("Text-specific", "M_T", "Text", "M_T_mean", "M_T_sd"),
            ("Visual-specific", "M_V", "Visual", "M_V_mean", "M_V_sd"),
        ),
        ylabel="Directional conductance margin",
    )
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    axes[0].set_title("Relation Calibration", loc="left", pad=5, fontsize=8.0, fontweight="bold")
    axes[0].text(-0.13, 1.04, "(a)", transform=axes[0].transAxes, fontweight="bold", fontsize=10, ha="left", va="bottom")

    stage2_endpoint = [row for row in stage2 if int(row["hop"]) == int(row["formal_K"])]
    stage2_endpoint_summary = [row for row in stage2_summary if int(row["hop"]) == int(row["formal_K"])]
    stage2_means = []
    for dataset in DATASETS:
        item = {"dataset": dataset}
        for modality in ("Text", "Visual"):
            summary_row = next(row for row in stage2_endpoint_summary if row["dataset"] == dataset and row["modality"] == modality)
            item[f"{modality}_mean"] = summary_row["retention_gain_formal_K_mean"]
            item[f"{modality}_sd"] = summary_row["retention_gain_formal_K_sd"]
        stage2_means.append(item)
    # Adapt the common two-series renderer to its endpoint summary schema.
    _plot_seed_summary_panel(
        axes[1], stage2_endpoint, stage2_endpoint_summary,
        series=(
            ("Text", "retention_gain_at_formal_K", "Text", "retention_gain_formal_K_mean", "retention_gain_formal_K_sd"),
            ("Visual", "retention_gain_at_formal_K", "Visual", "retention_gain_formal_K_mean", "retention_gain_formal_K_sd"),
        ),
        ylabel="Semantic-retention gain at formal K\n(Δ CKA: anchored − ordinary)",
    )
    # Text/Visual colors are identified by the shared figure legend below;
    # removing this in-panel legend also keeps grid strokes clear of labels.
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    axes[1].yaxis.labelpad = 1
    axes[1].set_title("Semantic Retention", loc="left", pad=5, fontsize=8.0, fontweight="bold")
    axes[1].text(-0.13, 1.04, "(b)", transform=axes[1].transAxes, fontweight="bold", fontsize=10, ha="left", va="bottom")

    _plot_stage3(axes[2], node_path, stage3)
    axes[2].set_title("Adaptive Contribution Reallocation", loc="left", pad=5, fontsize=8.0, fontweight="bold")
    axes[2].text(-0.13, 1.04, "(c)", transform=axes[2].transAxes, fontweight="bold", fontsize=10, ha="left", va="bottom")

    fig.legend(
        handles=[
            Patch(facecolor=TEXT_BLUE, edgecolor=INK, alpha=0.55, label="Text / Text-specific"),
            Patch(facecolor=VISUAL_GREEN, edgecolor=INK, alpha=0.55, label="Visual / Visual-specific"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=INK, markeredgecolor=INK, markersize=3.6, label="Seed results"),
            Line2D([], [], marker="D", linestyle="none", markerfacecolor=INK, markeredgecolor=INK, markersize=4.1, label="Mean ± SD"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=INK, label="Node median"),
        ],
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.035),
        columnspacing=0.8,
        handletextpad=0.4,
        fontsize=5.8,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / "figure4_mechanism_verification"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=output_root / "figure4_mechanism_verification_alignment.json",
        overlay_svg=output_root / "figure4_mechanism_verification_alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        # The visible labels intentionally include parentheses ("(a)", etc.),
        # while the auditor's label detector accepts only a bare lowercase
        # letter. Axis geometry remains fully audited; labels are inspected in
        # the final rendered panel review.
        require_panel_labels=False,
        strict=True,
    )
    paths = [base.with_suffix(".pdf"), base.with_suffix(".png"), base.with_suffix(".svg")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=600, bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    paths = build_figure(args.output_root)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
