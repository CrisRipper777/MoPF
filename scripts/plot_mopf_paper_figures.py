#!/usr/bin/env python3
"""Render the paper-facing MoPF Figures 1--3 from frozen empirical artifacts.

The script is visualization-only.  Figure 2 reads the completed E0-A--E0-D
CSV/NPZ/JSON artifacts; Figures 1 and 3 are vector-style conceptual diagrams
whose labels are tied to the frozen MoPF-vNext model definition.  No model,
checkpoint, label, metric, training, or experiment code is executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

E0A_ROOT = REPO_ROOT / "outputs/e0_empirical_motivation/edge_semantic_discrepancy"
E0B_ROOT = REPO_ROOT / "outputs/e0_empirical_motivation/semantic_retention"
E0C_ROOT = REPO_ROOT / "outputs/e0_empirical_motivation/multihop_utilization"
E0D_ROOT = REPO_ROOT / "outputs/e0_empirical_motivation/relation_utilization_bridge"
OUTPUT_ROOT = REPO_ROOT / "outputs/paper_figures"
CAPTION_ROOT = OUTPUT_ROOT / "caption_drafts"

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
REPRESENTATIVE_DATASETS = ["Movies", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
FORMAL_K = {"Movies": 3, "Toys": 3, "Grocery": 2, "ele-fashion": 3, "Reddit-S": 3}

TEXT = "#2F6FB3"
VISUAL = "#3B9960"
SHARED = "#737B83"
LIGHT_SHARED = "#D9DEE3"
ORANGE = "#D47B22"
INK = "#20252B"
PALE_BLUE = "#EAF2FB"
PALE_GREEN = "#ECF6EF"
PALE_GRAY = "#F3F5F7"
PALE_ORANGE = "#FFF2E4"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any) -> float:
    return float(value)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, stem: str, *, dpi: int = 320) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_ROOT / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_ROOT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = SHARED, lw: float = 1.3, style: str = "-|>", mutation: float = 12, connectionstyle: str = "arc3") -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, *, facecolor: str = PALE_GRAY, edgecolor: str = SHARED, fontsize: float = 9.0, weight: str = "normal", radius: float = 0.03, **kwargs: Any) -> FancyBboxPatch:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        **kwargs,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
    )
    return box


def _panel_title(ax: plt.Axes, label: str, title: str, subtitle: str | None = None) -> None:
    ax.text(0.0, 1.12, label, transform=ax.transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    ax.text(0.095, 1.12, title, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="left")
    if subtitle:
        ax.text(0.095, 1.065, subtitle, transform=ax.transAxes, fontsize=8.1, color=SHARED, va="bottom", ha="left")


def _load_e0a() -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    summary = {row["dataset"]: row for row in _read_csv(E0A_ROOT / "edge_discrepancy_summary.csv")}
    samples = {dataset: _read_csv(E0A_ROOT / dataset / "edge_rank_plot_sample.csv") for dataset in DATASETS}
    return summary, samples


def _load_e0b() -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(E0B_ROOT / "semantic_retention_summary.csv")
    return {(row["dataset"], row["modality"].lower()): row for row in rows}


def _load_e0c() -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, np.ndarray]]]:
    gaps = _read_csv(E0C_ROOT / "modality_gap_summary.csv")
    gap_by_dataset = {dataset: [row for row in gaps if row["dataset"] == dataset] for dataset in DATASETS}
    seed_means: dict[str, dict[str, np.ndarray]] = {}
    for dataset in DATASETS:
        path = E0C_ROOT / dataset / "seed_mean_node_multihop_utilization.npz"
        with np.load(path, allow_pickle=False) as loaded:
            seed_means[dataset] = {name: loaded[name].copy() for name in loaded.files}
    return gap_by_dataset, seed_means


def _load_e0d() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return (
        _read_csv(E0D_ROOT / "relation_utilization_association.csv"),
        _read_csv(E0D_ROOT / "relation_condition_quartiles.csv"),
    )


def _rank_panel(ax: plt.Axes, dataset: str, summary: dict[str, str], sample: list[dict[str, str]]) -> None:
    x = np.asarray([_f(row["rank_text"]) for row in sample])
    y = np.asarray([_f(row["rank_visual"]) for row in sample])
    points = ax.scatter(x, y, s=1.6, alpha=0.16, color=SHARED, linewidths=0, rasterized=True)
    points.set_rasterized(True)
    ax.add_patch(Rectangle((0.0, 0.75), 0.25, 0.25, facecolor=LIGHT_SHARED, edgecolor=SHARED, linewidth=0.8, alpha=0.55, zorder=0))
    ax.add_patch(Rectangle((0.75, 0.0), 0.25, 0.25, facecolor=LIGHT_SHARED, edgecolor=SHARED, linewidth=0.8, alpha=0.55, zorder=0))
    ax.plot([0, 1], [0, 1], color=SHARED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=2)
    ax.text(0.06, 0.90, "V high / T low", fontsize=7.2, color=SHARED)
    ax.text(0.79, 0.08, "T high / V low", fontsize=7.2, color=SHARED, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{dataset}\n$\\rho$ = {_f(summary['spearman_tv']):.2f}", pad=4)
    ax.set_xlabel("Text edge rank")
    ax.set_ylabel("Visual edge rank")
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.text(0.98, 0.02, f"nplot={len(sample):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=SHARED)


def _e0a_summary_panel(ax: plt.Axes, summary: dict[str, dict[str, str]]) -> None:
    rho = np.asarray([_f(summary[d]["spearman_tv"]) for d in DATASETS])
    y = np.arange(len(DATASETS))
    ax.barh(y, rho, color=LIGHT_SHARED, edgecolor=SHARED, linewidth=0.8, height=0.54)
    for yi, dataset in zip(y, DATASETS):
        row = summary[dataset]
        value = _f(row["spearman_tv"])
        annotation_x = value + 0.015 if value >= 0 else 0.015
        ax.text(annotation_x, yi, f"med gap {_f(row['median_rank_gap']):.2f}", va="center", ha="left", fontsize=7.2, color=INK)
    ax.set_yticks(y, DATASETS)
    ax.invert_yaxis()
    ax.axvline(0, color=SHARED, linewidth=0.8)
    ax.set_xlim(min(-0.1, float(rho.min()) - 0.08), max(0.6, float(rho.max()) + 0.1))
    ax.set_xlabel("Text–visual Spearman $\\rho$")
    ax.set_title("All-dataset summary", pad=4)
    ax.grid(axis="x", alpha=0.18)
    ax.text(0.0, 1.03, "Full-edge statistics", transform=ax.transAxes, fontsize=7.5, color=SHARED, va="bottom")


def _cka_panel(ax: plt.Axes, dataset: str, e0b: dict[tuple[str, str], dict[str, str]]) -> None:
    hops = np.arange(7)
    for modality, color, label in (("text", TEXT, "Text"), ("visual", VISUAL, "Visual")):
        row = e0b[(dataset, modality)]
        values = np.asarray([_f(row[f"cka_k{k}"]) for k in hops])
        ax.plot(hops, values, marker="o", markersize=3.2, linewidth=1.7, color=color, label=label)
    formal_k = FORMAL_K[dataset]
    ax.axvline(formal_k, color=SHARED, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.text(formal_k + 0.08, 0.97, f"K={formal_k}", transform=ax.get_xaxis_transform(), color=SHARED, fontsize=7.2, va="top")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(hops)
    ax.set_xlabel("Propagation hop $k$")
    ax.set_ylabel("Linear CKA to raw feature")
    ax.set_title(dataset, pad=4)
    ax.grid(alpha=0.18)


def _cka_summary_panel(ax: plt.Axes, e0b: dict[tuple[str, str], dict[str, str]]) -> None:
    x = np.arange(len(DATASETS))
    width = 0.34
    text = np.asarray([_f(e0b[(d, "text")][f"cka_k{FORMAL_K[d]}"]) for d in DATASETS])
    visual = np.asarray([_f(e0b[(d, "visual")][f"cka_k{FORMAL_K[d]}"]) for d in DATASETS])
    ax.bar(x - width / 2, text, width, color=TEXT, alpha=0.82, label="Text")
    ax.bar(x + width / 2, visual, width, color=VISUAL, alpha=0.82, label="Visual")
    for xi, dataset in enumerate(DATASETS):
        ax.text(xi, max(text[xi], visual[xi]) + 0.035, f"K={FORMAL_K[dataset]}", ha="center", fontsize=7.0, color=SHARED)
    ax.set_xticks(x, DATASETS, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("CKA at formal $K$")
    ax.set_title("Formal-$K$ summary", pad=4)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper left")


def _cosine_panel(ax: plt.Axes, dataset: str, e0b: dict[tuple[str, str], dict[str, str]]) -> None:
    hops = np.arange(7)
    for modality, color, label in (("text", TEXT, "Text"), ("visual", VISUAL, "Visual")):
        row = e0b[(dataset, modality)]
        mean = np.asarray([_f(row[f"cosine_mean_k{k}"]) for k in hops])
        q25 = np.asarray([_f(row[f"cosine_q25_k{k}"]) for k in hops])
        q75 = np.asarray([_f(row[f"cosine_q75_k{k}"]) for k in hops])
        ax.plot(hops, mean, marker="o", markersize=3.0, linewidth=1.7, color=color, label=label)
        ax.fill_between(hops, q25, q75, color=color, alpha=0.12, linewidth=0)
    formal_k = FORMAL_K[dataset]
    ax.axvline(formal_k, color=SHARED, linestyle=(0, (4, 3)), linewidth=1.0)
    ax.set_xlim(0, 6)
    ax.set_ylim(0.35, 1.03)
    ax.set_xticks(hops)
    ax.set_xlabel("Propagation hop $k$")
    ax.set_ylabel("Mean node-wise cosine")
    ax.set_title(dataset, pad=4)
    ax.grid(alpha=0.18)


def _cosine_summary_panel(ax: plt.Axes, e0b: dict[tuple[str, str], dict[str, str]]) -> None:
    x = np.arange(len(DATASETS))
    width = 0.34
    text = np.asarray([_f(e0b[(d, "text")][f"cosine_mean_k{FORMAL_K[d]}"]) for d in DATASETS])
    visual = np.asarray([_f(e0b[(d, "visual")][f"cosine_mean_k{FORMAL_K[d]}"]) for d in DATASETS])
    ax.bar(x - width / 2, text, width, color=TEXT, alpha=0.82, label="Text")
    ax.bar(x + width / 2, visual, width, color=VISUAL, alpha=0.82, label="Visual")
    ax.set_xticks(x, DATASETS, rotation=25, ha="right")
    ax.set_ylim(0.35, 1.06)
    ax.set_ylabel("Cosine at formal $K$")
    ax.set_title("Formal-$K$ summary", pad=4)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="lower left")


def _seed_npz_values(dataset: str, field: str) -> list[np.ndarray]:
    values = []
    for seed in SEEDS:
        with np.load(E0C_ROOT / dataset / f"seed{seed}" / "node_multihop_utilization.npz", allow_pickle=False) as loaded:
            values.append(loaded[field].copy())
    return values


def _violin_panel(ax: plt.Axes, gap_by_dataset: dict[str, list[dict[str, str]]], seed_means: dict[str, dict[str, np.ndarray]], datasets: list[str] = DATASETS) -> None:
    positions = np.arange(1, len(datasets) + 1, dtype=float)
    text = [seed_means[d]["normalized_order_text"] for d in datasets]
    visual = [seed_means[d]["normalized_order_visual"] for d in datasets]
    vp_t = ax.violinplot(text, positions=positions - 0.16, widths=0.28, showextrema=False)
    vp_v = ax.violinplot(visual, positions=positions + 0.16, widths=0.28, showextrema=False)
    for body in vp_t["bodies"]:
        body.set_facecolor(TEXT); body.set_edgecolor(TEXT); body.set_alpha(0.62); body.set_linewidth(0.7)
    for body in vp_v["bodies"]:
        body.set_facecolor(VISUAL); body.set_edgecolor(VISUAL); body.set_alpha(0.62); body.set_linewidth(0.7)
    for xi, dataset in zip(positions, datasets):
        t_medians = [float(np.median(value)) for value in _seed_npz_values(dataset, "normalized_order_text")]
        v_medians = [float(np.median(value)) for value in _seed_npz_values(dataset, "normalized_order_visual")]
        ax.scatter(np.full(3, xi - 0.16), t_medians, color=TEXT, s=10, zorder=4)
        ax.scatter(np.full(3, xi + 0.16), v_medians, color=VISUAL, s=10, zorder=4)
    ax.set_xticks(positions, datasets, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized contribution-weighted response order")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(handles=[
        Line2D([], [], color=TEXT, linewidth=7, alpha=0.62, label="Text"),
        Line2D([], [], color=VISUAL, linewidth=7, alpha=0.62, label="Visual"),
        Line2D([], [], marker="o", color=SHARED, linestyle="None", markersize=4, label="Seed medians"),
    ], frameon=False, loc="upper left", ncol=3)


def _l1_gap_panel(ax: plt.Axes, gap_by_dataset: dict[str, list[dict[str, str]]], datasets: list[str] = DATASETS) -> None:
    positions = np.arange(1, len(datasets) + 1, dtype=float)
    for xi, dataset in zip(positions, datasets):
        values = np.asarray([_f(row["profile_l1_median"]) for row in gap_by_dataset[dataset]])
        ax.scatter(np.full(values.size, xi), values, color=SHARED, s=24, zorder=3)
        ax.errorbar(xi, values.mean(), yerr=values.std(), fmt="o", color=SHARED, markerfacecolor="white", capsize=3.5, linewidth=1.2, zorder=4)
    ax.set_xticks(positions, datasets, rotation=20, ha="right")
    ax.set_ylabel("Median Text–Visual profile L1 gap")
    ax.set_title("Modality profile gap", pad=4)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=SHARED, linestyle="None", markersize=5, label="Seed values"),
        Line2D([], [], marker="o", color=SHARED, markerfacecolor="white", linestyle="-", markersize=5, label="Mean ± SD"),
    ], frameon=False, loc="upper left")


def _quartile_grocery_panel(ax: plt.Axes, quartiles: list[dict[str, str]], dataset: str = "Grocery") -> None:
    for modality, color, label in (("Text", TEXT, "Text"), ("Visual", VISUAL, "Visual")):
        values = []
        for seed in SEEDS:
            rows = [row for row in quartiles if row["dataset"] == dataset and int(row["seed"]) == seed and row["modality"] == modality]
            rows.sort(key=lambda row: int(row["quartile"][1]))
            values.append([_f(row["order_mean"]) for row in rows])
        array = np.asarray(values, dtype=float)
        x = np.arange(1, 5)
        ax.plot(x, array.mean(axis=0), marker="o", linewidth=1.8, color=color, label=label)
        ax.fill_between(x, array.min(axis=0), array.max(axis=0), color=color, alpha=0.13, linewidth=0)
    ax.set_xticks(np.arange(1, 5), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Local relation-condition quartile")
    ax.set_ylabel("Mean normalized response order")
    ax.set_title("Grocery", pad=4)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="best")


def _partial_summary_panel(ax: plt.Axes, associations: list[dict[str, str]]) -> None:
    positions = np.arange(1, len(DATASETS) + 1, dtype=float)
    for modality, color, offset in (("Text", TEXT, -0.15), ("Visual", VISUAL, 0.15)):
        for xi, dataset in zip(positions, DATASETS):
            rows = [row for row in associations if row["dataset"] == dataset and row["modality"] == modality]
            values = np.asarray([_f(row["rho_partial_degree"]) for row in rows])
            ax.scatter(np.full(values.size, xi + offset), values, color=color, s=19, zorder=3)
            ax.errorbar(xi + offset, values.mean(), yerr=values.std(), fmt="o", color=color, markerfacecolor="white", capsize=3.5, linewidth=1.1, zorder=4)
    ax.axhline(0, color=SHARED, linestyle=(0, (4, 3)), linewidth=0.9)
    ax.set_xticks(positions, DATASETS, rotation=20, ha="right")
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("Degree-controlled partial Spearman $\\rho$")
    ax.set_title("All-dataset bridge summary", pad=4)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(handles=[
        Line2D([], [], color=TEXT, marker="o", linewidth=1.6, label="Text"),
        Line2D([], [], color=VISUAL, marker="o", linewidth=1.6, label="Visual"),
        Line2D([], [], marker="o", color=INK, markerfacecolor="white", linestyle="-", markersize=5, label="Mean ± SD"),
    ], frameon=False, loc="upper left", ncol=3)


def _figure2_main() -> None:
    e0a_summary, e0a_samples = _load_e0a()
    e0b = _load_e0b()
    e0c_gaps, e0c_seed_means = _load_e0c()
    e0d_assoc, e0d_quartiles = _load_e0d()

    fig = plt.figure(figsize=(13.8, 16.8), facecolor="white")
    outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 0.75, 0.75], hspace=0.62, left=0.055, right=0.985, top=0.90, bottom=0.035)

    gs_a = outer[0].subgridspec(1, 4, wspace=0.33, width_ratios=[1, 1, 1, 1.16])
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(4)]
    for ax, dataset in zip(axes_a[:3], REPRESENTATIVE_DATASETS):
        _rank_panel(ax, dataset, e0a_summary[dataset], e0a_samples[dataset])
    _e0a_summary_panel(axes_a[3], e0a_summary)
    _panel_title(axes_a[0], "(a)", "Physical-edge cross-modal semantic discrepancy", "E0-A; representative rank–rank plots plus full-dataset summary")

    gs_b = outer[1].subgridspec(1, 4, wspace=0.33, width_ratios=[1, 1, 1, 1.16])
    axes_b = [fig.add_subplot(gs_b[0, i]) for i in range(4)]
    for ax, dataset in zip(axes_b[:3], REPRESENTATIVE_DATASETS):
        _cka_panel(ax, dataset, e0b)
    _cka_summary_panel(axes_b[3], e0b)
    axes_b[0].legend(frameon=False, loc="upper right")
    _panel_title(axes_b[0], "(b)", "Propagation-induced semantic retention change", "E0-B; linear CKA to the raw modality feature")

    gs_c = outer[2].subgridspec(1, 2, wspace=0.28, width_ratios=[1.45, 0.9])
    axes_c = [fig.add_subplot(gs_c[0, i]) for i in range(2)]
    _violin_panel(axes_c[0], e0c_gaps, e0c_seed_means)
    _l1_gap_panel(axes_c[1], e0c_gaps)
    _panel_title(axes_c[0], "(c)", "Node–modality heterogeneity in multi-hop utilization", "E0-C; seed-mean node profiles with seed-level markers")

    gs_d = outer[3].subgridspec(1, 2, wspace=0.3, width_ratios=[0.9, 1.45])
    axes_d = [fig.add_subplot(gs_d[0, i]) for i in range(2)]
    _quartile_grocery_panel(axes_d[0], e0d_quartiles)
    _partial_summary_panel(axes_d[1], e0d_assoc)
    _panel_title(axes_d[0], "(d)", "Local relation condition vs. multi-hop utilization", "E0-D; Grocery quartile illustration and degree-controlled summary")
    fig.suptitle("Figure 2. Empirical motivation for relation-conditioned multi-hop aggregation", fontsize=14, fontweight="bold", y=0.992)
    _save_figure(fig, "figure2_main")


def _figure2_appendix() -> None:
    e0a_summary, e0a_samples = _load_e0a()
    e0b = _load_e0b()
    e0c_gaps, e0c_seed_means = _load_e0c()
    e0d_assoc, e0d_quartiles = _load_e0d()

    fig = plt.figure(figsize=(15.5, 24.5), facecolor="white")
    outer = fig.add_gridspec(4, 1, height_ratios=[1.65, 1.55, 1.55, 1.15], hspace=0.64, left=0.045, right=0.99, top=0.90, bottom=0.025)

    gs_a = outer[0].subgridspec(2, 3, wspace=0.28, hspace=0.43)
    axes_a = [fig.add_subplot(gs_a[r, c]) for r in range(2) for c in range(3)]
    for ax, dataset in zip(axes_a[:5], DATASETS):
        _rank_panel(ax, dataset, e0a_summary[dataset], e0a_samples[dataset])
    _e0a_summary_panel(axes_a[5], e0a_summary)
    _panel_title(axes_a[0], "(a)", "Physical-edge cross-modal semantic discrepancy", "Complete five-dataset E0-A view")

    gs_b = outer[1].subgridspec(2, 3, wspace=0.28, hspace=0.43)
    axes_b = [fig.add_subplot(gs_b[r, c]) for r in range(2) for c in range(3)]
    for ax, dataset in zip(axes_b[:5], DATASETS):
        _cka_panel(ax, dataset, e0b)
    _cka_summary_panel(axes_b[5], e0b)
    axes_b[0].legend(frameon=False, loc="upper right")
    _panel_title(axes_b[0], "(b)", "Propagation-induced semantic retention change: CKA", "Complete five-dataset E0-B view; dashed line marks formal K")

    gs_cos = outer[2].subgridspec(2, 3, wspace=0.28, hspace=0.43)
    axes_cos = [fig.add_subplot(gs_cos[r, c]) for r in range(2) for c in range(3)]
    for ax, dataset in zip(axes_cos[:5], DATASETS):
        _cosine_panel(ax, dataset, e0b)
    _cosine_summary_panel(axes_cos[5], e0b)
    axes_cos[0].legend(frameon=False, loc="lower left")
    _panel_title(axes_cos[0], "(b, appendix)", "Propagation-induced semantic retention change: node-wise cosine", "Supplementary E0-B cosine view")

    gs_d = outer[3].subgridspec(1, 2, wspace=0.28, width_ratios=[1.45, 1.0])
    axes_d = [fig.add_subplot(gs_d[0, i]) for i in range(2)]
    _violin_panel(axes_d[0], e0c_gaps, e0c_seed_means)
    _partial_summary_panel(axes_d[1], e0d_assoc)
    _panel_title(axes_d[0], "(c,d)", "Utilization heterogeneity and relation–utilization bridge", "Full-dataset E0-C violin and E0-D partial-rho summary")
    fig.suptitle("Figure 2 appendix. Complete empirical-motivation diagnostics", fontsize=14, fontweight="bold", y=0.992)
    _save_figure(fig, "figure2_appendix")

    # A dedicated vector appendix export keeps the cosine diagnostic easy to
    # place independently in a supplementary section.
    fig_cos = plt.figure(figsize=(15.5, 8.0), facecolor="white")
    gs = fig_cos.add_gridspec(2, 3, wspace=0.28, hspace=0.43, left=0.055, right=0.99, top=0.90, bottom=0.12)
    axs = [fig_cos.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax, dataset in zip(axs[:5], DATASETS):
        _cosine_panel(ax, dataset, e0b)
    _cosine_summary_panel(axs[5], e0b)
    axs[0].legend(frameon=False, loc="lower left")
    fig_cos.suptitle("Figure 2(b) appendix. Node-wise cosine retention under ordinary propagation", fontsize=13, fontweight="bold")
    _save_figure(fig_cos, "figure2_b_cosine_appendix")


def _figure1_concept() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.55), gridspec_kw={"wspace": 0.2})
    fig.subplots_adjust(top=0.82, bottom=0.12)
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax = axes[0]
    _panel_title(ax, "(a)", "Relation discrepancy", "One edge; two modality-specific similarities")
    ax.plot([0.25, 0.75], [0.52, 0.52], color=LIGHT_SHARED, linewidth=5, solid_capstyle="round")
    _arrow(ax, (0.27, 0.57), (0.73, 0.57), color=TEXT, lw=2.2, connectionstyle="arc3,rad=0.13")
    _arrow(ax, (0.73, 0.47), (0.27, 0.47), color=VISUAL, lw=2.2, connectionstyle="arc3,rad=0.13")
    for x in (0.25, 0.75):
        ax.add_patch(Circle((x, 0.52), 0.095, facecolor="white", edgecolor=SHARED, linewidth=1.2))
        ax.add_patch(Circle((x, 0.55), 0.038, facecolor=TEXT, edgecolor="none", alpha=0.9))
        ax.add_patch(Circle((x, 0.49), 0.038, facecolor=VISUAL, edgecolor="none", alpha=0.9))
    ax.text(0.5, 0.68, "$s_{ij}^{T}$", color=TEXT, ha="center", fontsize=10)
    ax.text(0.5, 0.33, "$s_{ij}^{V}$", color=VISUAL, ha="center", fontsize=10)
    ax.text(0.5, 0.16, "$s_{ij}^{T} \\neq s_{ij}^{V}$", color=SHARED, ha="center", fontsize=10.5, fontweight="bold")
    ax.text(0.5, 0.08, "shared physical support; modality-intrinsic edge similarity", color=SHARED, ha="center", fontsize=7.8)

    ax = axes[1]
    _panel_title(ax, "(b)", "Propagation drift", "Shared propagation shifts raw modality states")
    _box(ax, (0.05, 0.43), 0.19, 0.18, "raw\n$X^m$", facecolor=PALE_GRAY, edgecolor=SHARED, fontsize=10)
    _arrow(ax, (0.25, 0.52), (0.36, 0.52), color=SHARED, lw=1.6)
    _box(ax, (0.36, 0.43), 0.19, 0.18, "shared\nphysical graph $A$", facecolor=PALE_GRAY, edgecolor=SHARED, fontsize=8.8)
    _arrow(ax, (0.56, 0.52), (0.67, 0.67), color=TEXT, lw=1.6)
    _arrow(ax, (0.56, 0.52), (0.67, 0.37), color=VISUAL, lw=1.6)
    _box(ax, (0.67, 0.58), 0.2, 0.16, "$Q_1^T$", facecolor=PALE_BLUE, edgecolor=TEXT, fontsize=10)
    _box(ax, (0.67, 0.26), 0.2, 0.16, "$Q_1^V$", facecolor=PALE_GREEN, edgecolor=VISUAL, fontsize=10)
    ax.text(0.79, 0.83, "$Q_0^m \\rightarrow Q_1^m \\rightarrow \\cdots$", ha="center", fontsize=9.2, color=SHARED)
    ax.text(0.79, 0.16, "retention is not guaranteed to be modality-aligned", ha="center", fontsize=7.9, color=SHARED)
    ax.annotate("shift", xy=(0.90, 0.66), xytext=(0.90, 0.79), ha="center", color=SHARED, fontsize=8.5, arrowprops={"arrowstyle": "-|>", "color": SHARED, "lw": 1.2})

    ax = axes[2]
    _panel_title(ax, "(c)", "Aggregation heterogeneity", "Node- and modality-specific hop profiles")
    ax.text(0.12, 0.84, "node-specific contribution profiles", fontsize=8.5, color=SHARED)
    starts = [0.20, 0.47, 0.74]
    text_profiles = [[0.55, 0.25, 0.14, 0.06], [0.13, 0.12, 0.55, 0.20], [0.28, 0.18, 0.18, 0.36]]
    visual_profiles = [[0.18, 0.12, 0.52, 0.18], [0.40, 0.27, 0.19, 0.14], [0.08, 0.17, 0.20, 0.55]]
    for idx, x in enumerate(starts):
        ax.text(x, 0.70, f"node {idx + 1}", ha="center", fontsize=8.0, color=INK)
        for y, values, color, label in ((0.58, text_profiles[idx], TEXT, "T"), (0.39, visual_profiles[idx], VISUAL, "V")):
            ax.text(x - 0.09, y + 0.055, label, color=color, fontsize=8, fontweight="bold", ha="right")
            cumulative = 0.0
            for value in values:
                ax.add_patch(Rectangle((x - 0.075 + cumulative * 0.15, y), value * 0.15, 0.10, facecolor=color, edgecolor="white", linewidth=0.25, alpha=0.84))
                cumulative += value
        ax.text(x, 0.26, "$p_{i,k}^{m}$", ha="center", fontsize=7.8, color=SHARED)
    ax.text(0.5, 0.10, "heterogeneous utilization motivates explicit relation-conditioned aggregation", ha="center", fontsize=8.0, color=ORANGE, fontweight="bold")

    # Shared structure and the cross-stage bridge are visually separated from
    # the three empirical observations.
    fig.text(0.333, 0.07, "physical support", color=SHARED, fontsize=8, ha="center")
    fig.text(0.667, 0.07, "cross-stage bridge", color=ORANGE, fontsize=8, ha="center")
    fig.text(0.50, 0.985, "Figure 1. From cross-modal relation mismatch to relation-conditioned multi-hop aggregation", ha="center", va="top", fontsize=13.5, fontweight="bold")
    _save_figure(fig, "figure1_concept")


def _figure3_framework() -> None:
    fig, ax = plt.subplots(figsize=(16.2, 7.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.975, "Figure 3. MoPF framework", ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.935, "Frozen multimodal inputs → learned local semantic graph → anchored multi-hop banks → adaptive composition", ha="center", va="top", fontsize=9.2, color=SHARED)

    # Input streams.
    _box(ax, (0.025, 0.67), 0.12, 0.13, "$X^T$\nFrozen text feature", facecolor=PALE_BLUE, edgecolor=TEXT, fontsize=9.0, weight="bold")
    _box(ax, (0.025, 0.27), 0.12, 0.13, "$X^V$\nFrozen visual feature", facecolor=PALE_GREEN, edgecolor=VISUAL, fontsize=9.0, weight="bold")

    # Stage I.
    _box(ax, (0.19, 0.15), 0.215, 0.70, "", facecolor=PALE_GRAY, edgecolor=SHARED)
    ax.text(0.2975, 0.815, "Stage I · Local semantic graph", ha="center", fontsize=10.5, fontweight="bold")
    _box(ax, (0.215, 0.64), 0.165, 0.10, "modality-specific\nprojection $h^m$", facecolor="white", edgecolor=SHARED, fontsize=8.6)
    _box(ax, (0.215, 0.45), 0.165, 0.12, "$w_{ij}^{m}=f(\\cos(M_m h_i^m, M_m h_j^m))$", facecolor="white", edgecolor=SHARED, fontsize=7.6)
    # Small semantic edge sketch.
    ax.plot([0.245, 0.35], [0.32, 0.32], color=LIGHT_SHARED, linewidth=4.5, solid_capstyle="round")
    ax.add_patch(Circle((0.245, 0.32), 0.035, facecolor=TEXT, edgecolor="white", linewidth=1))
    ax.add_patch(Circle((0.35, 0.32), 0.035, facecolor=VISUAL, edgecolor="white", linewidth=1))
    _arrow(ax, (0.265, 0.35), (0.33, 0.35), color=TEXT, lw=1.6, connectionstyle="arc3,rad=0.2")
    _arrow(ax, (0.33, 0.29), (0.265, 0.29), color=VISUAL, lw=1.6, connectionstyle="arc3,rad=0.2")
    ax.text(0.2975, 0.20, "shared physical support $A$\nmodality-specific edge strength", ha="center", va="center", fontsize=8.0, color=SHARED)

    # Stage II.
    _box(ax, (0.445, 0.15), 0.215, 0.70, "", facecolor=PALE_GRAY, edgecolor=SHARED)
    ax.text(0.5525, 0.815, "Stage II · Multi-hop state banks", ha="center", fontsize=10.5, fontweight="bold")
    _box(ax, (0.47, 0.66), 0.165, 0.10, "$P^m=D_m^{-1/2}(A^m+I)D_m^{-1/2}$", facecolor="white", edgecolor=SHARED, fontsize=7.8)
    ax.text(0.5525, 0.58, "anchored propagation", ha="center", fontsize=8.4, color=SHARED)
    ax.text(0.5525, 0.53, "$S_{k}^{m}=(1-\\alpha)P^m S_{k-1}^{m}+\\alpha H^m$", ha="center", fontsize=8.0, color=INK)
    for row_y, color, label in ((0.38, TEXT, "Text"), (0.24, VISUAL, "Visual")):
        ax.text(0.477, row_y + 0.04, label, color=color, fontsize=8.0, fontweight="bold", ha="right")
        for idx, x in enumerate((0.50, 0.56, 0.62)):
            _box(ax, (x, row_y), 0.045, 0.075, f"$S_{idx}$", facecolor="white", edgecolor=color, fontsize=7.5, radius=0.01)
            if idx < 2:
                _arrow(ax, (x + 0.047, row_y + 0.037), (x + 0.058, row_y + 0.037), color=SHARED, lw=0.8, mutation=8)
    ax.text(0.5525, 0.20, "$\\alpha=0.1$; formal $K$", ha="center", fontsize=7.8, color=SHARED)

    # Stage III.
    _box(ax, (0.70, 0.15), 0.27, 0.70, "", facecolor=PALE_GRAY, edgecolor=SHARED)
    ax.text(0.835, 0.815, "Stage III · Adaptive composition", ha="center", fontsize=10.5, fontweight="bold")
    _box(ax, (0.725, 0.64), 0.22, 0.10, "$\\eta_{i,k}^{C1}=\\gamma_k+\\Delta\\gamma_k^m+\\delta_{i,k}^m$", facecolor="white", edgecolor=SHARED, fontsize=8.0)
    _box(ax, (0.725, 0.47), 0.22, 0.11, "relation-conditioned bridge\n$\\tilde c_i^m \\rightarrow \\tau_{i,k}^m$", facecolor=PALE_ORANGE, edgecolor=ORANGE, fontsize=8.1, weight="bold")
    _arrow(ax, (0.835, 0.64), (0.835, 0.59), color=SHARED, lw=1.1)
    _arrow(ax, (0.835, 0.47), (0.835, 0.41), color=ORANGE, lw=1.5)
    ax.text(0.835, 0.41, "$\\eta_{i,k}^{final}=\\eta_{i,k}^{C1}+\\tau_{i,k}^m$", ha="center", fontsize=8.1, color=ORANGE, fontweight="bold")
    _box(ax, (0.725, 0.27), 0.22, 0.09, "$G_{i,k}^m=\\eta_{i,k}^{final}S_{i,k}^m$", facecolor="white", edgecolor=SHARED, fontsize=8.3)
    _arrow(ax, (0.835, 0.40), (0.835, 0.36), color=SHARED, lw=1.1)
    ax.text(0.835, 0.205, "$p_{i,k}^m \\propto ||G_{i,k}^m||_2$", ha="center", fontsize=8.7, color=INK)
    ax.text(0.835, 0.155, "$z^T,z^V \\rightarrow$ fusion MLP $\\rightarrow$ node prediction", ha="center", fontsize=7.8, color=SHARED)

    # Stream arrows and stage ownership.
    _arrow(ax, (0.145, 0.735), (0.19, 0.735), color=TEXT, lw=1.8)
    _arrow(ax, (0.145, 0.335), (0.19, 0.335), color=VISUAL, lw=1.8)
    _arrow(ax, (0.405, 0.735), (0.445, 0.735), color=SHARED, lw=1.5)
    _arrow(ax, (0.405, 0.335), (0.445, 0.335), color=SHARED, lw=1.5)
    _arrow(ax, (0.66, 0.735), (0.70, 0.735), color=SHARED, lw=1.5)
    _arrow(ax, (0.66, 0.335), (0.70, 0.335), color=SHARED, lw=1.5)
    ax.text(0.5, 0.055, "Blue = Text   Green = Visual   Gray = shared physical structure   Orange = relation-conditioned TCPR bridge", ha="center", fontsize=8.2, color=SHARED)
    _save_figure(fig, "figure3_framework")


def _write_captions() -> None:
    CAPTION_ROOT.mkdir(parents=True, exist_ok=True)
    captions = {
        "figure1_caption.md": """# Figure 1 caption draft\n\n**Figure 1. From cross-modal relation mismatch to relation-conditioned multi-hop aggregation.** (a) A single physical edge may have different modality-intrinsic semantic similarity strengths in the text and visual modalities. (b) Repeated propagation on the shared physical support can shift the modality-intrinsic node states relative to their raw inputs. (c) The predecessor adaptive multi-hop aggregator can assign node- and modality-dependent contribution profiles across propagation orders. Together, these observations motivate making local relation condition available to aggregation, while the figure is conceptual and does not imply causality.\n""",
        "figure2_caption.md": """# Figure 2 caption draft\n\n**Figure 2. Empirical motivation for relation-conditioned multi-hop aggregation.** (a) E0-A rank–rank plots compare text and visual raw feature similarity ranks on the same physical edges; shaded corners mark opposite-quartile relations and the summary panel reports full-edge Spearman correlation. (b) E0-B linear CKA to the raw modality feature across ordinary physical-graph propagation hops; dashed lines mark formal dataset K and the summary reports formal-K CKA. (c) E0-C seed-mean node distributions of normalized contribution-weighted response order, together with the median text–visual contribution-profile L1 gap. (d) E0-D Grocery relation-condition quartiles illustrate the relation–utilization trend, while the summary reports degree-controlled partial Spearman correlations for all datasets and modalities. Colors identify Text and Visual; gray denotes shared structure and orange denotes the relation-conditioned bridge. These analyses are descriptive empirical motivation, not causal evidence or downstream-performance claims.\n""",
        "figure2_appendix_caption.md": """# Figure 2 appendix caption draft\n\n**Figure 2 appendix. Complete empirical-motivation diagnostics.** The appendix retains all five datasets for E0-A rank–rank discrepancy plots, E0-B CKA trajectories, and the E0-C/E0-D summaries; it additionally reports the E0-B node-wise cosine retention trajectories and formal-K summary. All statistics and plotted samples are read from the frozen E0-A–E0-D artifacts; no labels or performance metrics are used.\n""",
        "figure3_caption.md": """# Figure 3 caption draft\n\n**Figure 3. MoPF framework.** Frozen text and visual node features are projected separately and used to construct modality-specific learned semantic edge weights on the shared physical graph. Each modality then forms an anchored multi-hop state bank with fixed formal order K. Stage III combines global, modality, and node-conditioned coefficients; in the final TCPR-equipped model, centered local relation condition contributes the relation-conditioned residual before state contributions are combined and normalized into node–modality multi-hop utilization profiles. The modality outputs are fused for node prediction.\n""",
    }
    for name, text in captions.items():
        (CAPTION_ROOT / name).write_text(text, encoding="utf-8")
    (OUTPUT_ROOT / "caption_drafts.md").write_text(
        "\n".join(f"## {name}\n\n{text}" for name, text in captions.items()),
        encoding="utf-8",
    )


def _write_manifest() -> None:
    manifest = {
        "workflow": "paper-facing MoPF Figures 1-3",
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "HEAD"),
        "source_artifacts": {
            "E0-A": str(E0A_ROOT),
            "E0-B": str(E0B_ROOT),
            "E0-C": str(E0C_ROOT),
            "E0-D": str(E0D_ROOT),
        },
        "representative_datasets": REPRESENTATIVE_DATASETS,
        "appendix_datasets": DATASETS,
        "style": {
            "background": "white",
            "text_color": TEXT,
            "visual_color": VISUAL,
            "shared_color": SHARED,
            "bridge_color": ORANGE,
            "pdf_vector_for_lines_text_patches": True,
            "dense_E0A_scatter_rasterized_inside_PDF": True,
        },
        "no_experiment_execution": True,
        "training_started": False,
        "checkpoint_loaded": False,
        "labels_used": False,
        "performance_metrics_used": False,
        "outputs": {
            "figure1": ["figure1_concept.png", "figure1_concept.pdf"],
            "figure2_main": ["figure2_main.png", "figure2_main.pdf"],
            "figure2_appendix": ["figure2_appendix.png", "figure2_appendix.pdf"],
            "figure2_cosine_appendix": ["figure2_b_cosine_appendix.png", "figure2_b_cosine_appendix.pdf"],
            "figure3": ["figure3_framework.png", "figure3_framework.pdf"],
            "captions": ["caption_drafts.md", "caption_drafts/"],
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figure_generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global OUTPUT_ROOT, CAPTION_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root.resolve()
    CAPTION_ROOT = OUTPUT_ROOT / "caption_drafts"
    _setup_style()
    _figure1_concept()
    _figure2_main()
    _figure2_appendix()
    _figure3_framework()
    _write_captions()
    _write_manifest()
    print(f"Wrote paper figures to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
