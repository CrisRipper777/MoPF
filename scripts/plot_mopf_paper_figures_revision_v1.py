#!/usr/bin/env python3
"""Render publication-facing MoPF Figures 1--3 revision v1.

This script is visualization-only.  It reuses the frozen E0-A--E0-D artifact
loaders from ``plot_mopf_paper_figures.py`` and writes a new revision directory
without touching the previous paper-figure exports.  The only derived display
quantity is the requested E0-B representation shift, ``1 - CKA at formal K``.
No model, checkpoint, label, metric, training, or experiment code is executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import plot_mopf_paper_figures as base


OUTPUT_ROOT = REPO_ROOT / "outputs/paper_figures/revision_v1"
TEXT = base.TEXT
VISUAL = base.VISUAL
SHARED = base.SHARED
LIGHT_SHARED = base.LIGHT_SHARED
ORANGE = base.ORANGE
INK = base.INK
PALE_BLUE = base.PALE_BLUE
PALE_GREEN = base.PALE_GREEN
PALE_GRAY = base.PALE_GRAY
PALE_ORANGE = base.PALE_ORANGE
DATASETS = base.DATASETS
REPRESENTATIVE_DATASETS = base.REPRESENTATIVE_DATASETS
FORMAL_K = base.FORMAL_K
SEEDS = base.SEEDS


def _f(value: Any) -> float:
    return float(value)


def _save_figure(fig: plt.Figure, stem: str, *, dpi: int = 300) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Keep the declared canvas dimensions stable for LaTeX placement and
    # export the same untrimmed canvas in all three formats.
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{stem}.svg", facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=dpi, facecolor="white")
    plt.close(fig)


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = SHARED,
    lw: float = 1.2,
    mutation: float = 11,
    connectionstyle: str = "arc3",
    zorder: int = 3,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = PALE_GRAY,
    edgecolor: str = SHARED,
    fontsize: float = 8.5,
    weight: str = "normal",
    radius: float = 0.02,
    linewidth: float = 0.85,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.009,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=INK,
            zorder=4,
        )
    return patch


def _style() -> None:
    base._setup_style()
    plt.rcParams.update(
        {
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "svg.fonttype": "none",
        }
    )


def _axis_header(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.5, 1.16, f"{label} {title}", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.8, fontweight="bold", color=INK)


def _draw_node_card(ax: plt.Axes, x: float, y: float, *, variant: int) -> None:
    _box(ax, (x, y), 0.22, 0.25, "", facecolor="white", edgecolor=SHARED, linewidth=0.9)
    ax.text(x + 0.11, y + 0.205, f"node {variant}", ha="center", va="center", fontsize=7.4, color=INK)
    ax.text(x + 0.035, y + 0.145, "text", ha="left", va="center", fontsize=6.5, color=TEXT, fontweight="bold")
    ax.plot([x + 0.075, x + 0.195], [y + 0.145, y + 0.145], color=TEXT, linewidth=1.0, alpha=0.65)
    ax.plot([x + 0.075, x + 0.175], [y + 0.118, y + 0.118], color=TEXT, linewidth=1.0, alpha=0.45)
    ax.text(x + 0.035, y + 0.065, "image", ha="left", va="center", fontsize=6.5, color=VISUAL, fontweight="bold")
    ax.add_patch(Rectangle((x + 0.075, y + 0.035), 0.12, 0.055, facecolor=PALE_GREEN, edgecolor=VISUAL, linewidth=0.55))
    ax.add_patch(Circle((x + 0.128, y + 0.063), 0.012, facecolor=VISUAL, edgecolor="none", alpha=0.75))


def _figure1() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"wspace": 0.17})
    fig.subplots_adjust(left=0.025, right=0.985, top=0.82, bottom=0.17)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # (a) Same physical support, different modality-intrinsic compatibility.
    ax = axes[0]
    ax.text(0.5, 1.12, "(a) Relation-level\nsemantic discrepancy", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=INK, linespacing=1.05)
    _draw_node_card(ax, 0.08, 0.42, variant=1)
    _draw_node_card(ax, 0.70, 0.42, variant=2)
    ax.plot([0.30, 0.70], [0.545, 0.545], color=LIGHT_SHARED, linewidth=4.5, solid_capstyle="round", zorder=1)
    _arrow(ax, (0.31, 0.585), (0.69, 0.585), color=TEXT, lw=2.4, mutation=9, connectionstyle="arc3,rad=0.12")
    _arrow(ax, (0.69, 0.505), (0.31, 0.505), color=VISUAL, lw=1.1, mutation=9, connectionstyle="arc3,rad=0.12")
    ax.text(0.50, 0.73, "same physical relation", ha="center", fontsize=7.4, color=SHARED)
    ax.text(0.50, 0.65, "Text compatibility", ha="center", fontsize=7.2, color=TEXT)
    ax.text(0.50, 0.39, "Visual compatibility", ha="center", fontsize=7.2, color=VISUAL)
    ax.text(0.50, 0.105, "Same physical relation,\ndifferent semantic compatibility.", ha="center", fontsize=6.8, color=INK, linespacing=1.05)

    # (b) Four explicit propagation orders without implying universal monotonic loss.
    ax = axes[1]
    ax.text(0.5, 1.12, "(b) Propagation-level\nsemantic drift", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=INK, linespacing=1.05)
    hop_x = np.linspace(0.13, 0.87, 4)
    for idx, x in enumerate(hop_x):
        ax.text(x, 0.77, f"{idx}-hop", ha="center", fontsize=7.3, color=INK)
        ax.add_patch(Circle((x, 0.61), 0.035, facecolor="white", edgecolor=SHARED, linewidth=0.9, zorder=3))
        for n in range(idx):
            angle = (n + 1) * np.pi / (idx + 1)
            nx = x + 0.075 * np.cos(angle)
            ny = 0.61 + 0.075 * np.sin(angle)
            ax.plot([x, nx], [0.61, ny], color=LIGHT_SHARED, linewidth=0.9, zorder=1)
            ax.add_patch(Circle((nx, ny), 0.018, facecolor=LIGHT_SHARED, edgecolor=SHARED, linewidth=0.45, zorder=2))
        for yy, color, phase in ((0.39, TEXT, idx), (0.23, VISUAL, (idx + 1) % 3)):
            widths = [0.31 - 0.025 * phase, 0.21 + 0.02 * phase, 0.13 + 0.015 * ((idx + phase) % 2)]
            ax.add_patch(Rectangle((x - 0.105, yy), widths[0], 0.055, facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.78))
            ax.add_patch(Rectangle((x - 0.105 + widths[0], yy), widths[1], 0.055, facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.58))
            ax.add_patch(Rectangle((x - 0.105 + widths[0] + widths[1], yy), widths[2], 0.055, facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.38))
        if idx < 3:
            _arrow(ax, (x + 0.10, 0.61), (hop_x[idx + 1] - 0.10, 0.61), color=SHARED, lw=0.9, mutation=8)
    ax.text(0.035, 0.415, "T", ha="center", va="center", fontsize=7.3, color=TEXT, fontweight="bold")
    ax.text(0.035, 0.255, "V", ha="center", va="center", fontsize=7.3, color=VISUAL, fontweight="bold")
    ax.text(0.50, 0.08, "Propagation reshapes\nmodality-intrinsic representations.", ha="center", fontsize=6.8, color=INK, linespacing=1.05)

    # (c) Node- and modality-dependent hop profiles.
    ax = axes[2]
    ax.text(0.5, 1.12, "(c) Aggregation-level\nheterogeneity", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=INK, linespacing=1.05)
    profiles_t = [[0.56, 0.25, 0.13, 0.06], [0.14, 0.13, 0.54, 0.19], [0.28, 0.18, 0.18, 0.36]]
    profiles_v = [[0.18, 0.12, 0.52, 0.18], [0.40, 0.27, 0.19, 0.14], [0.08, 0.17, 0.20, 0.55]]
    for idx, x in enumerate((0.22, 0.50, 0.78)):
        ax.text(x, 0.78, f"node {idx + 1}", ha="center", fontsize=7.3, color=INK)
        for yy, values, color, label in ((0.58, profiles_t[idx], TEXT, "T"), (0.39, profiles_v[idx], VISUAL, "V")):
            ax.text(x - 0.105, yy + 0.035, label, ha="right", va="center", fontsize=7.2, color=color, fontweight="bold")
            start = x - 0.095
            for hop, value in enumerate(values):
                alpha = 0.42 + 0.12 * (hop % 3)
                ax.add_patch(Rectangle((start, yy), 0.19 * value, 0.09, facecolor=color, edgecolor="white", linewidth=0.45, alpha=alpha))
                start += 0.19 * value
    ax.text(0.50, 0.24, "hop order: 0   1   2   3", ha="center", fontsize=7.0, color=SHARED)
    ax.text(0.50, 0.08, "Heterogeneous nodes and modalities\ncan utilize propagation orders differently.", ha="center", fontsize=6.6, color=INK, linespacing=1.05)

    fig.text(0.50, 0.035, "Structure-semantic mismatches arise across relation modeling, propagation, and aggregation.", ha="center", fontsize=8.0, color=SHARED)
    _save_figure(fig, "figure1_main")


def _rank_panel(ax: plt.Axes, summary: dict[str, str], sample: list[dict[str, str]]) -> None:
    x = np.asarray([_f(row["rank_text"]) for row in sample])
    y = np.asarray([_f(row["rank_visual"]) for row in sample])
    rank_cmap = LinearSegmentedColormap.from_list("rank_density_gray", ["#F7F9FA", "#7D858C"])
    ax.hexbin(x, y, gridsize=55, mincnt=1, bins="log", cmap=rank_cmap, linewidths=0, alpha=0.82)
    ax.add_patch(Rectangle((0.0, 0.75), 0.25, 0.25, facecolor=LIGHT_SHARED, edgecolor=SHARED, linewidth=0.6, alpha=0.52, zorder=0))
    ax.add_patch(Rectangle((0.75, 0.0), 0.25, 0.25, facecolor=LIGHT_SHARED, edgecolor=SHARED, linewidth=0.6, alpha=0.52, zorder=0))
    ax.plot([0, 1], [0, 1], color=SHARED, linestyle=(0, (3, 2)), linewidth=0.85, zorder=2)
    ax.text(0.055, 0.90, "V high / T low", fontsize=6.6, color=SHARED)
    ax.text(0.78, 0.08, "T high / V low", fontsize=6.6, color=SHARED, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Movies", pad=4, fontsize=9.0, fontweight="normal")
    ax.set_xlabel("Text edge-similarity rank")
    ax.set_ylabel("Visual edge-similarity rank")
    ax.grid(alpha=0.13, linewidth=0.5)
    ax.text(0.98, 0.02, f"plot sample n={len(sample):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color=SHARED)


def _e0a_compact_summary(ax: plt.Axes, summary: dict[str, dict[str, str]]) -> None:
    rho_min, rho_max = -0.10, 0.60
    gap_max = 0.35
    y = np.arange(len(DATASETS))[::-1]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.78, 4.55)
    ax.set_yticks(y, DATASETS)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([])
    ax.set_title("Full-edge summary", pad=4, fontsize=9.0, fontweight="normal")
    ax.axvline(0.55, color=LIGHT_SHARED, linewidth=0.8)
    ax.text(0.27, 4.28, "Spearman $\\rho$", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.text(0.78, 4.28, "Median rank gap", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for tick, xpos in ((-0.1, 0.04), (0.0, 0.10), (0.3, 0.28), (0.6, 0.46)):
        ax.text(xpos, -0.54, f"{tick:.1f}", ha="center", va="top", fontsize=6.3, color=SHARED)
    for tick, xpos in ((0.0, 0.64), (0.1, 0.73), (0.2, 0.82), (0.3, 0.91)):
        ax.text(xpos, -0.54, f"{tick:.1f}", ha="center", va="top", fontsize=6.3, color=SHARED)
    ax.text(0.27, -0.73, "local scale", ha="center", va="top", fontsize=6.1, color=SHARED)
    ax.text(0.78, -0.73, "local scale", ha="center", va="top", fontsize=6.1, color=SHARED)
    for idx, dataset in enumerate(DATASETS):
        yy = y[idx]
        row = summary[dataset]
        rho = _f(row["spearman_tv"])
        gap = _f(row["median_rank_gap"])
        xr = 0.04 + (rho - rho_min) / (rho_max - rho_min) * 0.42
        xg = 0.64 + gap / gap_max * 0.28
        ax.plot([0.04, 0.46], [yy, yy], color=LIGHT_SHARED, linewidth=0.55)
        ax.plot([0.64, 0.92], [yy, yy], color=LIGHT_SHARED, linewidth=0.55)
        ax.scatter(xr, yy, color=SHARED, s=23, zorder=3)
        ax.scatter(xg, yy, color=SHARED, s=23, zorder=3)
        ax.text(min(xr + 0.018, 0.49), yy, f"{rho:.2f}", ha="left", va="center", fontsize=6.4, color=INK)
        ax.text(min(xg + 0.018, 0.96), yy, f"{gap:.2f}", ha="left", va="center", fontsize=6.4, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _formal_k_shift_panel(ax: plt.Axes, e0b: dict[tuple[str, str], dict[str, str]]) -> None:
    x = np.arange(len(DATASETS), dtype=float)
    text_shift = np.asarray([1.0 - _f(e0b[(d, "text")][f"cka_k{FORMAL_K[d]}"]) for d in DATASETS])
    visual_shift = np.asarray([1.0 - _f(e0b[(d, "visual")][f"cka_k{FORMAL_K[d]}"]) for d in DATASETS])
    ax.scatter(x - 0.075, text_shift, color=TEXT, s=30, zorder=4, label="Text")
    ax.scatter(x + 0.075, visual_shift, color=VISUAL, s=30, zorder=4, label="Visual")
    ax.set_xticks(x, [f"{dataset}\nK={FORMAL_K[dataset]}" for dataset in DATASETS], rotation=18, ha="right")
    ax.tick_params(axis="x", pad=4)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Representation shift at formal $K$\n$1 -$ CKA to raw feature")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False, loc="upper left", ncol=2)


def _violin_panel(ax: plt.Axes, seed_means: dict[str, dict[str, np.ndarray]]) -> None:
    positions = np.arange(1, len(DATASETS) + 1, dtype=float)
    text = [seed_means[d]["normalized_order_text"] for d in DATASETS]
    visual = [seed_means[d]["normalized_order_visual"] for d in DATASETS]
    vp_t = ax.violinplot(text, positions=positions - 0.16, widths=0.28, showextrema=False)
    vp_v = ax.violinplot(visual, positions=positions + 0.16, widths=0.28, showextrema=False)
    for body in vp_t["bodies"]:
        body.set_facecolor(TEXT)
        body.set_edgecolor(TEXT)
        body.set_alpha(0.62)
        body.set_linewidth(0.55)
    for body in vp_v["bodies"]:
        body.set_facecolor(VISUAL)
        body.set_edgecolor(VISUAL)
        body.set_alpha(0.62)
        body.set_linewidth(0.55)
    for xi, dataset in zip(positions, DATASETS):
        t_medians = [float(np.median(value)) for value in base._seed_npz_values(dataset, "normalized_order_text")]
        v_medians = [float(np.median(value)) for value in base._seed_npz_values(dataset, "normalized_order_visual")]
        ax.scatter(np.full(3, xi - 0.16), t_medians, color=TEXT, s=8, zorder=4)
        ax.scatter(np.full(3, xi + 0.16), v_medians, color=VISUAL, s=8, zorder=4)
    ax.set_xticks(positions, DATASETS, rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized contribution-weighted\nresponse order")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(
        handles=[
            Line2D([], [], color=TEXT, linewidth=6, alpha=0.62, label="Text"),
            Line2D([], [], color=VISUAL, linewidth=6, alpha=0.62, label="Visual"),
            Line2D([], [], marker="o", color=SHARED, linestyle="None", markersize=3.5, label="Seed medians"),
        ],
        frameon=False,
        loc="upper left",
        ncol=3,
    )


def _partial_summary_panel(ax: plt.Axes, associations: list[dict[str, str]]) -> None:
    positions = np.arange(1, len(DATASETS) + 1, dtype=float)
    for modality, color, offset in (("Text", TEXT, -0.14), ("Visual", VISUAL, 0.14)):
        for xi, dataset in zip(positions, DATASETS):
            rows = [row for row in associations if row["dataset"] == dataset and row["modality"] == modality]
            values = np.asarray([_f(row["rho_partial_degree"]) for row in rows])
            ax.scatter(np.full(values.size, xi + offset), values, color=color, s=14, zorder=3)
            ax.errorbar(xi + offset, values.mean(), yerr=values.std(), fmt="o", color=color, markerfacecolor="white", capsize=2.5, linewidth=0.9, zorder=4)
    ax.axhline(0, color=SHARED, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_xticks(positions, DATASETS, rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Degree-controlled partial Spearman $\\rho$")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(
        handles=[
            Line2D([], [], color=TEXT, marker="o", linewidth=1.2, label="Text"),
            Line2D([], [], color=VISUAL, marker="o", linewidth=1.2, label="Visual"),
            Line2D([], [], marker="o", color=INK, markerfacecolor="white", linestyle="-", markersize=4, label="Mean ± SD"),
        ],
        frameon=False,
        loc="upper left",
        ncol=3,
    )


def _figure2() -> None:
    e0a_summary, e0a_samples = base._load_e0a()
    e0b = base._load_e0b()
    _, e0c_seed_means = base._load_e0c()
    e0d_assoc, _ = base._load_e0d()

    fig = plt.figure(figsize=(7.2, 7.45), facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.72, 1.0], hspace=0.66, left=0.09, right=0.985, top=0.90, bottom=0.09)

    gs_a = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.30)
    ax_a = fig.add_subplot(gs_a[0, 0])
    ax_a_summary = fig.add_subplot(gs_a[0, 1])
    _rank_panel(ax_a, e0a_summary["Movies"], e0a_samples["Movies"])
    _e0a_compact_summary(ax_a_summary, e0a_summary)

    ax_b = fig.add_subplot(outer[1, 0])
    _formal_k_shift_panel(ax_b, e0b)

    gs_c = outer[2].subgridspec(1, 2, width_ratios=[1.16, 0.94], wspace=0.30)
    ax_c = fig.add_subplot(gs_c[0, 0])
    ax_c_summary = fig.add_subplot(gs_c[0, 1])
    _violin_panel(ax_c, e0c_seed_means)
    _partial_summary_panel(ax_c_summary, e0d_assoc)

    fig.text(0.09, 0.935, "(a) Relation-level semantic discrepancy", ha="left", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
    fig.text(0.09, 0.615, "(b) Propagation-level semantic drift", ha="left", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
    fig.text(0.09, 0.325, "(c) Aggregation-level heterogeneity", ha="left", va="bottom", fontsize=10.2, fontweight="bold", color=INK)
    _save_figure(fig, "figure2_main")


def _graph_icon(ax: plt.Axes, x: float, y: float, *, scale: float = 1.0, edge_widths: tuple[float, float, float] = (1.0, 1.0, 1.0), label: str | None = None) -> None:
    nodes = [(x, y + 0.06 * scale), (x + 0.08 * scale, y + 0.13 * scale), (x + 0.16 * scale, y + 0.06 * scale), (x + 0.08 * scale, y - 0.01 * scale)]
    edges = ((0, 1), (1, 2), (1, 3))
    for width, (a, b) in zip(edge_widths, edges):
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]], color=SHARED, linewidth=1.2 * width, alpha=0.75, solid_capstyle="round", zorder=1)
    for nx, ny in nodes:
        ax.add_patch(Circle((nx, ny), 0.024 * scale, facecolor="white", edgecolor=SHARED, linewidth=0.7, zorder=3))
        ax.add_patch(Circle((nx - 0.008 * scale, ny + 0.006 * scale), 0.009 * scale, facecolor=TEXT, edgecolor="none", zorder=4))
        ax.add_patch(Circle((nx + 0.008 * scale, ny - 0.006 * scale), 0.009 * scale, facecolor=VISUAL, edgecolor="none", zorder=4))
    if label:
        ax.text(x + 0.08 * scale, y - 0.055 * scale, label, ha="center", va="top", fontsize=6.4, color=SHARED)


def _profile_bar(ax: plt.Axes, x: float, y: float, width: float, values: list[float], *, color: str, height: float = 0.035) -> None:
    start = x
    for idx, value in enumerate(values):
        ax.add_patch(Rectangle((start, y), width * value, height, facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.48 + 0.13 * (idx % 3), zorder=4))
        start += width * value


def _stage(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str) -> None:
    _box(ax, xy, width, height, "", facecolor=PALE_GRAY, edgecolor=SHARED, radius=0.018, linewidth=0.9)
    ax.text(xy[0] + width / 2, xy[1] + height - 0.045, title, ha="center", va="center", fontsize=8.1, fontweight="bold", color=INK, linespacing=1.0, zorder=4)


def _figure3() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 5.15), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Input multimodal graph.
    _stage(ax, (0.01, 0.16), 0.135, 0.67, "Input\nMultimodal Graph")
    _graph_icon(ax, 0.037, 0.57, scale=0.62, edge_widths=(1.0, 1.0, 1.0), label="shared physical graph")
    _box(ax, (0.028, 0.29), 0.10, 0.07, "Text attributes", facecolor=PALE_BLUE, edgecolor=TEXT, fontsize=7.0)
    _box(ax, (0.028, 0.20), 0.10, 0.07, "Visual attributes", facecolor=PALE_GREEN, edgecolor=VISUAL, fontsize=7.0)

    # Stage I: same topology, modality-specific relation strengths.
    _stage(ax, (0.17, 0.16), 0.19, 0.67, "Stage I ·\nModality-Aware\nRelation Calibration")
    ax.text(0.265, 0.695, "Shared physical support", ha="center", fontsize=6.8, color=SHARED)
    ax.text(0.265, 0.635, "Text-weighted\nrelation graph", ha="center", fontsize=6.4, color=TEXT, fontweight="bold", linespacing=0.95)
    _graph_icon(ax, 0.205, 0.515, scale=0.72, edge_widths=(2.0, 0.75, 1.4), label="$W^T$")
    ax.text(0.265, 0.405, "Visual-weighted\nrelation graph", ha="center", fontsize=6.4, color=VISUAL, fontweight="bold", linespacing=0.95)
    _graph_icon(ax, 0.205, 0.285, scale=0.72, edge_widths=(0.75, 2.0, 0.9), label="$W^V$")
    ax.text(0.265, 0.205, "Semantic relation calibration", ha="center", fontsize=7.0, color=SHARED)

    # Stage II: semantic reference and multi-hop banks.
    _stage(ax, (0.385, 0.16), 0.215, 0.67, "Stage II ·\nSemantic-Preserving\nMulti-Hop Propagation")
    ax.text(0.4925, 0.695, "Semantic reference $H_0$", ha="center", fontsize=6.8, color=SHARED)
    ax.plot([0.415, 0.575], [0.655, 0.655], color=LIGHT_SHARED, linewidth=0.9, zorder=1)
    ax.text(0.402, 0.575, "Text", ha="right", va="center", fontsize=7.0, color=TEXT, fontweight="bold")
    ax.text(0.402, 0.36, "Visual", ha="right", va="center", fontsize=7.0, color=VISUAL, fontweight="bold")
    for row_y, color in ((0.54, TEXT), (0.325, VISUAL)):
        ax.plot([0.415, 0.575], [row_y + 0.035, row_y + 0.035], color=color, alpha=0.22, linewidth=0.8, zorder=1)
        for idx, xx in enumerate((0.42, 0.465, 0.51, 0.555)):
            label = "$S_0$" if idx == 0 else "$S_1$" if idx == 1 else "$S_2$" if idx == 2 else "$S_K$"
            _box(ax, (xx, row_y), 0.036, 0.07, label, facecolor="white", edgecolor=color, fontsize=7.0, radius=0.008, linewidth=0.75)
            if idx < 3:
                _arrow(ax, (xx + 0.038, row_y + 0.035), (xx + 0.043, row_y + 0.035), color=SHARED, lw=0.7, mutation=7)
    ax.text(0.4925, 0.205, "Multi-hop state bank", ha="center", fontsize=7.0, color=SHARED)

    # Stage III: hierarchy, orange cross-stage bridge, mixer and outputs.
    _stage(ax, (0.635, 0.16), 0.235, 0.67, "Stage III ·\nAdaptive Multi-Hop\nComposition")
    ax.text(0.7525, 0.695, "Hierarchical hop preference", ha="center", fontsize=6.8, color=SHARED)
    preference_profiles = [("Global", [0.46, 0.28, 0.16, 0.10]), ("Modality", [0.18, 0.38, 0.27, 0.17]), ("Node", [0.10, 0.16, 0.22, 0.52])]
    for row_idx, (label, values) in enumerate(preference_profiles):
        yy = 0.615 - 0.075 * row_idx
        ax.text(0.675, yy + 0.017, label, ha="right", va="center", fontsize=6.8, color=INK)
        _profile_bar(ax, 0.685, yy, 0.115, values, color=SHARED, height=0.034)
    _box(ax, (0.665, 0.355), 0.175, 0.105, "Transport-Conditioned\nPreference Residual", facecolor=PALE_ORANGE, edgecolor=ORANGE, fontsize=7.0, weight="bold", linewidth=1.1)
    ax.text(0.7525, 0.325, "Local relation condition", ha="center", fontsize=6.8, color=SHARED)
    _box(ax, (0.662, 0.235), 0.181, 0.06, "Base Hop Preference + TCPR", facecolor="white", edgecolor=SHARED, fontsize=7.0)
    _arrow(ax, (0.7525, 0.355), (0.7525, 0.30), color=ORANGE, lw=1.3, mutation=10)
    _box(ax, (0.665, 0.17), 0.18, 0.045, "Final Hop Weights  →  Hop Mixer", facecolor="white", edgecolor=SHARED, fontsize=6.5)
    _arrow(ax, (0.7525, 0.235), (0.7525, 0.215), color=SHARED, lw=0.9, mutation=8)

    # Input streams and ordinary stage arrows.
    _arrow(ax, (0.128, 0.325), (0.17, 0.60), color=TEXT, lw=1.45, mutation=10, connectionstyle="arc3,rad=0.14")
    _arrow(ax, (0.128, 0.235), (0.17, 0.36), color=VISUAL, lw=1.45, mutation=10, connectionstyle="arc3,rad=-0.14")
    _arrow(ax, (0.36, 0.60), (0.385, 0.60), color=SHARED, lw=1.2, mutation=9)
    _arrow(ax, (0.60, 0.60), (0.635, 0.60), color=SHARED, lw=1.2, mutation=9)

    # Dominant cross-stage bridge from Stage I to TCPR.
    _arrow(ax, (0.35, 0.235), (0.665, 0.415), color=ORANGE, lw=2.0, mutation=12, connectionstyle="arc3,rad=-0.14", zorder=5)
    ax.text(0.50, 0.265, "Relation state informs aggregation", ha="center", va="center", fontsize=7.0, color=ORANGE, fontweight="bold", rotation=10, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8}, zorder=6)

    # Output heads.
    ax.text(0.935, 0.77, "Output", ha="center", fontsize=8.5, fontweight="bold")
    _box(ax, (0.895, 0.61), 0.08, 0.065, "$Z^T$", facecolor=PALE_BLUE, edgecolor=TEXT, fontsize=8.0, weight="bold")
    _box(ax, (0.895, 0.51), 0.08, 0.065, "$Z^V$", facecolor=PALE_GREEN, edgecolor=VISUAL, fontsize=8.0, weight="bold")
    _arrow(ax, (0.87, 0.60), (0.895, 0.64), color=TEXT, lw=1.1, mutation=8, connectionstyle="arc3,rad=0.12")
    _arrow(ax, (0.87, 0.46), (0.895, 0.54), color=VISUAL, lw=1.1, mutation=8, connectionstyle="arc3,rad=-0.12")
    _box(ax, (0.887, 0.36), 0.096, 0.065, "Multimodal\nFusion", facecolor=PALE_GRAY, edgecolor=SHARED, fontsize=6.9)
    _arrow(ax, (0.935, 0.51), (0.935, 0.425), color=SHARED, lw=0.9, mutation=8)
    _box(ax, (0.887, 0.25), 0.096, 0.065, "Prediction\nHead", facecolor=PALE_GRAY, edgecolor=SHARED, fontsize=6.9)
    _arrow(ax, (0.935, 0.36), (0.935, 0.315), color=SHARED, lw=0.9, mutation=8)
    ax.text(0.935, 0.185, "NC · Node classification", ha="center", fontsize=6.5, color=INK)
    ax.text(0.935, 0.145, "LP · Link prediction", ha="center", fontsize=6.5, color=INK)

    ax.text(0.50, 0.035, "Blue = Text   Green = Visual   Gray = shared structure   Orange = cross-stage TCPR bridge", ha="center", fontsize=7.5, color=SHARED)
    _save_figure(fig, "figure3_main")


def _write_captions() -> None:
    captions = """# Publication-facing caption drafts\n\n**Figure 1. Three structure–semantic problems in multimodal graphs.** (a) The same shared physical relation can have different modality-intrinsic semantic compatibility for text and visual node attributes. (b) Repeated propagation expands neighborhood context and reshapes the modality-intrinsic representations; the schematic does not imply a universally monotone loss of semantics. (c) Different nodes and modalities can distribute their utilization over propagation orders differently. The figure states the empirical motivation only and does not depict a proposed solution.\n\n**Figure 2. Empirical evidence for the three structure–semantic problems.** (a) The Movies rank–rank panel compares text and visual edge-similarity ranks on the same physical edges; shaded corners mark opposite-quartile relations, while the compact right panel reports full-edge Spearman correlation and median rank gap for all five datasets. (b) Representation shift at the frozen formal K, shown as 1 minus the reported CKA to the raw modality feature, summarizes propagation-induced change for all datasets; K is annotated below each dataset. (c) The left violin plot shows frozen seed-mean node distributions of normalized contribution-weighted response order with seed-level medians, and the right plot shows the degree-controlled partial Spearman association between local relation condition and multi-hop utilization across seeds. Text is blue, Visual is green, and shared structure is gray.\n\n**Figure 3. MoPF architecture.** The input multimodal graph retains a shared physical topology while Text and Visual streams undergo modality-aware relation calibration with different edge strengths. Semantic-preserving multi-hop propagation forms Text and Visual state banks referenced to the raw state. Stage III combines hierarchical hop preferences with a Transport-Conditioned Preference Residual supplied by the Stage-I relation state; the resulting final hop weights mix the state banks into $Z^T$ and $Z^V$. The fused representations support both node classification (NC) and link prediction (LP).\n"""
    (OUTPUT_ROOT / "captions.md").write_text(captions, encoding="utf-8")


def _write_manifest() -> None:
    manifest = {
        "workflow": "publication-facing MoPF Figures 1-3 revision v1",
        "git_branch": base._git("branch", "--show-current"),
        "git_commit": base._git("rev-parse", "HEAD"),
        "source_artifacts": {
            "E0-A": str(base.E0A_ROOT),
            "E0-B": str(base.E0B_ROOT),
            "E0-C": str(base.E0C_ROOT),
            "E0-D": str(base.E0D_ROOT),
        },
        "authoritative_statistics_recomputed": False,
        "e0b_display_transform": "representation shift = 1 - frozen CKA at formal K",
        "training_started": False,
        "checkpoint_loaded": False,
        "labels_used": False,
        "performance_metrics_used": False,
        "model_or_config_modified": False,
        "previous_outputs_preserved": True,
        "style": {
            "background": "white",
            "text": TEXT,
            "visual": VISUAL,
            "shared": SHARED,
            "relation_conditioned_bridge": ORANGE,
            "red_problem_cue_used": False,
            "font_family": "DejaVu Sans",
            "font_sizes_pt": {"base": 8.4, "panel_heading": 10.2, "axis_label": 8.2, "tick": 7.4, "legend": 7.4},
            "pdf_svg_vector_first": True,
            "dense_rank_plot_encoding": "vector hexbin over the fixed E0-A plot sample",
            "png_dpi": 300,
        },
        "figure_dimensions_inches": {
            "figure1_main": [7.2, 2.85],
            "figure2_main": [7.2, 7.45],
            "figure3_main": [13.6, 5.15],
        },
        "outputs": {
            "figure1_main": ["figure1_main.pdf", "figure1_main.svg", "figure1_main.png"],
            "figure2_main": ["figure2_main.pdf", "figure2_main.svg", "figure2_main.png"],
            "figure3_main": ["figure3_main.pdf", "figure3_main.svg", "figure3_main.png"],
            "captions": ["captions.md"],
        },
        "appendix_candidates_preserved": [
            str(REPO_ROOT / "outputs/paper_figures/figure2_appendix.pdf"),
            str(REPO_ROOT / "outputs/paper_figures/figure2_b_cosine_appendix.pdf"),
            str(REPO_ROOT / "outputs/e0_empirical_motivation/edge_semantic_discrepancy/edge_semantic_discrepancy.pdf"),
            str(REPO_ROOT / "outputs/e0_empirical_motivation/semantic_retention/semantic_retention_cka.pdf"),
            str(REPO_ROOT / "outputs/e0_empirical_motivation/semantic_retention/semantic_retention_node_cosine.pdf"),
            str(REPO_ROOT / "outputs/e0_empirical_motivation/multihop_utilization/modality_profile_gap.pdf"),
            str(REPO_ROOT / "outputs/e0_empirical_motivation/relation_utilization_bridge/relation_condition_response_order_quartiles.pdf"),
        ],
    }
    (OUTPUT_ROOT / "revision_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root.resolve()
    _style()
    _figure1()
    _figure2()
    _figure3()
    _write_captions()
    _write_manifest()
    print(f"Wrote publication-facing revision v1 figures to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
