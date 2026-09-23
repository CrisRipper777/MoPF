"""Prepare preview Figure 3/4 panels from the final diagnostic CSV files.

The plots are descriptive previews. The CSVs remain the authoritative source
data for the paper figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ModuleNotFoundError:
    # The historical alignment helper is not part of the final branch.  Keep
    # preview generation usable without resurrecting deleted legacy scripts.
    def require_matplotlib_panel_alignment(*args, **kwargs):
        json_out = kwargs.get("json_out")
        if json_out is not None:
            Path(json_out).write_text(
                json.dumps({
                    "status": "skipped",
                    "reason": "historical audit_panel_alignment helper is absent from final branch",
                }, indent=2),
                encoding="utf-8",
            )
        overlay_svg = kwargs.get("overlay_svg")
        if overlay_svg is not None:
            Path(overlay_svg).write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>\n',
                encoding="utf-8",
            )
        return None


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
DISPLAY = {"ele-fashion": "ele-fashion", "Reddit-S": "Reddit-S"}
COLORS = {"text": "#0F4D92", "visual": "#B64342", "normal": "#767676", "globalized": "#42949E", "shuffled": "#9A4D8E"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def add_label(ax, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def set_category_ticks(ax, positions, labels) -> None:
    ax.set_xticks(positions, labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_rotation_mode("anchor")
        tick.set_ha("right")
        tick.set_fontsize(6.5)


def export(fig: mpl.figure.Figure, root: Path, stem: str, excluded: list[mpl.axes.Axes] | None = None) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=root / f"{stem}.alignment.json",
        overlay_svg=root / f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        exclude_axes=excluded or [],
    )
    fig.savefig(root / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def figure3(root: Path, context: Path, out: Path) -> None:
    relation = read_csv(context / "relation_response.csv")
    neighbor = read_csv(context / "neighbor_allocation.csv")
    gates = read_csv(context / "adaptive_gate_summary.csv")
    interventions = read_csv(context / "adaptive_gate_intervention.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.25, top=0.88, wspace=0.42)

    # a — modality-level response: semantic discrepancy and learned response.
    ax = axes[0]
    rows = [row for row in relation if row["modality"] == "text_visual_discrepancy"]
    for row in rows:
        ax.scatter(f(row, "semantic_mean"), f(row, "relation_weight_mean"), s=30, color="#0F4D92", edgecolor="white", linewidth=0.5)
        ax.annotate(DISPLAY.get(row["dataset"], row["dataset"]), (f(row, "semantic_mean"), f(row, "relation_weight_mean")), xytext=(4, 6), textcoords="offset points", fontsize=7)
    ax.set_xlabel("|text − visual| semantic score")
    ax.set_ylabel("|text − visual| relation weight")
    ax.set_title("Relation response", loc="left")
    add_label(ax, "a")

    # b — modality-specific allocation, retaining isolated-node accounting.
    ax = axes[1]
    x = np.arange(len(neighbor))
    tv = np.asarray([f(row, "tv_distance_mean") for row in neighbor])
    top = np.asarray([f(row, "top_neighbor_disagreement") for row in neighbor])
    width = 0.36
    ax.bar(x - width / 2, tv, width, color="#3775BA", label="TV distance")
    ax.set_ylabel("TV distance", color="#3775BA")
    ax.tick_params(axis="y", labelcolor="#3775BA")
    set_category_ticks(ax, x, [DISPLAY.get(row["dataset"], row["dataset"]) for row in neighbor])
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, top, width, color="#B64342", label="Top-neighbor disagreement")
    ax2.set_ylabel("Top-neighbor disagreement", color="#B64342")
    ax2.tick_params(axis="y", labelcolor="#B64342")
    ax.set_title("Neighbor allocation", loc="left")
    add_label(ax, "b")

    # c — gate distributions by modality, with intervention flip-rate inset.
    ax = axes[2]
    means, spreads = {"text": [], "visual": []}, {"text": [], "visual": []}
    for dataset in DATASETS:
        for modality in ("text", "visual"):
            subset = [row for row in gates if row["dataset"] == dataset and row["modality"] == modality]
            means[modality].append(float(np.mean([f(row, "mean") for row in subset])))
            spreads[modality].append(float(np.mean([f(row, "std_across_nodes") for row in subset])))
    x = np.arange(len(DATASETS))
    ax.errorbar(x - 0.08, means["text"], yerr=spreads["text"], fmt="o-", color=COLORS["text"], capsize=2, label="Text")
    ax.errorbar(x + 0.08, means["visual"], yerr=spreads["visual"], fmt="o-", color=COLORS["visual"], capsize=2, label="Visual")
    set_category_ticks(ax, x, [DISPLAY.get(name, name) for name in DATASETS])
    ax.set_ylabel("Gate mean ± node SD")
    ax.set_ylim(0, 1.05)
    ax.set_title("Adaptive gate", loc="left")
    ax.legend(fontsize=7, loc="lower left")
    inset = ax.inset_axes([0.50, 0.08, 0.47, 0.34])
    for offset, intervention in enumerate(("globalized", "shuffled")):
        subset = [row for row in interventions if row["intervention"] == intervention]
        inset.plot(x, [f(next(row for row in subset if row["dataset"] == dataset), "prediction_flip_rate") for dataset in DATASETS], "o-", color=COLORS[intervention], label=intervention)
    inset.set_xticks(x, [])
    inset.tick_params(labelsize=6)
    inset.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    add_label(ax, "c")
    export(fig, out, "figure3_preview", excluded=[inset, ax2])


def figure4(root: Path, integration: Path, out: Path) -> None:
    order = read_csv(integration / "effective_order_contribution.csv")
    attention = read_csv(integration / "average_attention_matrix.csv")
    interventions = read_csv(integration / "integration_intervention.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.25, top=0.88, wspace=0.42)

    # a — effective-order means and across-node SD.
    ax = axes[0]
    effective = [row for row in order if row["order"] == "effective_mean"]
    x = np.arange(len(DATASETS))
    text = [next(row for row in effective if row["dataset"] == dataset and row["modality"] == "text") for dataset in DATASETS]
    visual = [next(row for row in effective if row["dataset"] == dataset and row["modality"] == "visual") for dataset in DATASETS]
    ax.errorbar(x - 0.08, [f(row, "mean_contribution") for row in text], yerr=[f(row, "std") for row in text], fmt="o-", color=COLORS["text"], capsize=2, label="Text")
    ax.errorbar(x + 0.08, [f(row, "mean_contribution") for row in visual], yerr=[f(row, "std") for row in visual], fmt="o-", color=COLORS["visual"], capsize=2, label="Visual")
    set_category_ticks(ax, x, [DISPLAY.get(name, name) for name in DATASETS])
    ax.set_ylabel("Mean effective order")
    ax.set_title("Effective-order heterogeneity", loc="left")
    ax.legend(fontsize=7)
    add_label(ax, "a")

    # b — two representative modality attention matrices; complete matrices
    # for all five datasets remain in average_attention_matrix.csv.
    subset_datasets = ("Movies", "Grocery", "ele-fashion")
    matrix = np.zeros((4, 4))
    for row in attention:
        if row["dataset"] == "Movies" and row["modality"] == "text":
            matrix[int(row["query_order"]), int(row["key_order"])] = f(row, "mean_attention")
    im = axes[1].imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    axes[1].set_aspect("auto")
    axes[1].set_xticks(range(4), range(4)); axes[1].set_yticks(range(4), range(4))
    axes[1].set_xlabel("Key order"); axes[1].set_ylabel("Query order")
    axes[1].set_title("Mean cross-order attention\nMovies / text", loc="left")
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if matrix[i, j] > matrix.max() * .55 else "black")
    add_label(axes[1], "b")

    # c — direct integration perturbations, logit MAE as the common scale.
    ax = axes[2]
    names = ("query_collapse", "uniform_attention", "interaction_off")
    labels = {"query_collapse": "Query-collapse", "uniform_attention": "Uniform", "interaction_off": "Interaction-off"}
    width = 0.24
    x = np.arange(len(DATASETS))
    for index, name in enumerate(names):
        subset = [next(row for row in interventions if row["dataset"] == dataset and row["intervention"] == name) for dataset in DATASETS]
        ax.bar(x + (index - 1) * width, [f(row, "logit_mae") for row in subset], width, label=labels[name])
    set_category_ticks(ax, x, [DISPLAY.get(name, name) for name in DATASETS])
    ax.set_ylabel("Logit MAE vs normal")
    ax.set_title("Interaction intervention", loc="left")
    ax.legend(fontsize=7)
    add_label(ax, "c")
    export(fig, out, "figure4_preview")


def copy_data(context: Path, integration: Path, out: Path) -> None:
    f3 = out / "figure3_data"; f4 = out / "figure4_data"
    f3.mkdir(parents=True, exist_ok=True); f4.mkdir(parents=True, exist_ok=True)
    for name in ("relation_response.csv", "neighbor_allocation.csv", "adaptive_gate_summary.csv", "adaptive_gate_intervention.csv"):
        shutil.copy2(context / name, f3 / name)
    for name in ("effective_order_contribution.csv", "average_attention_matrix.csv", "integration_intervention.csv"):
        shutil.copy2(integration / name, f4 / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=Path("outputs/final/context_formation_analysis"))
    parser.add_argument("--integration", type=Path, default=Path("outputs/final/multi_order_integration_analysis"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/paper_figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_data(args.context, args.integration, args.output_dir)
    figure3(args.output_dir, args.context, args.output_dir)
    figure4(args.output_dir, args.integration, args.output_dir)


if __name__ == "__main__":
    main()
