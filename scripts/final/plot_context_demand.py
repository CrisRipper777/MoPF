"""Render the M1 Contextualization-Demand preview figure.

The plot is intentionally a compact preview rather than a camera-ready figure:
two panels show aggregated preferred-lambda distributions and the third panel
summarizes text/visual disagreement and the descriptive oracle gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL_FIGURE_SCRIPTS = Path("/home/m3/.codex/skills/nature-figure/scripts")
if SKILL_FIGURE_SCRIPTS.is_dir():
    sys.path.insert(0, str(SKILL_FIGURE_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


DATASETS = ("Movies", "Grocery", "ele-fashion")
LAMBDAS = (0.00, 0.25, 0.50, 0.75, 1.00)
PALETTE = {
    "text": "#4C78A8",
    "visual": "#F58518",
    "disagreement": "#54A24B",
    "oracle": "#B279A2",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: str | float | int) -> float:
    return float(value)


def _lambda_key(value: float) -> str:
    return f"{value:.2f}"


def _load_inputs(output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    distributions = _read_csv(output_dir / "preferred_lambda_distribution.csv")
    summary = _read_csv(output_dir / "context_demand_summary.csv")
    disagreement = _read_csv(output_dir / "modality_disagreement.csv")
    distributions = [row for row in distributions if row["scope"] == "aggregated"]
    disagreement = [row for row in disagreement if row["scope"] == "aggregated"]
    return distributions, summary, disagreement


def render(output_dir: Path) -> tuple[Path, Path]:
    """Render PNG/PDF outputs and the required alignment manifest."""
    distributions, summary, disagreement = _load_inputs(output_dir)
    summary_by_key = {(row["dataset"], row["modality"]): row for row in summary}
    disagreement_by_dataset = {row["dataset"]: row for row in disagreement}

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    fig.patch.set_facecolor("white")
    width = 0.14
    x = np.arange(len(DATASETS), dtype=float)
    offsets = (np.arange(len(LAMBDAS)) - 2.0) * width

    for axis, modality, title in (
        (axes[0], "text", "Text"),
        (axes[1], "visual", "Visual"),
    ):
        for lambda_index, lambda_value in enumerate(LAMBDAS):
            heights = []
            for dataset in DATASETS:
                matching = [
                    row
                    for row in distributions
                    if row["dataset"] == dataset
                    and row["modality"] == modality
                    and abs(_f(row["lambda"]) - lambda_value) < 1e-8
                ]
                heights.append(_f(matching[0]["proportion"]) if matching else 0.0)
            axis.bar(
                x + offsets[lambda_index],
                heights,
                width=width * 0.92,
                color=mpl.colors.to_rgba(PALETTE[modality], 0.45 + 0.1 * lambda_index),
                edgecolor="white",
                linewidth=0.3,
                label=f"λ={lambda_value:.2f}",
            )
        axis.set_title(title, pad=7, fontweight="bold")
        axis.set_xticks(x, DATASETS, rotation=25, ha="right")
        for tick in axis.get_xticklabels():
            tick.set_rotation_mode("anchor")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Aggregated preferred proportion")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)

    axes[0].legend(ncol=3, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.27))

    disagreement_values = [
        _f(disagreement_by_dataset.get(dataset, {}).get("disagreement_ratio", 0.0))
        for dataset in DATASETS
    ]
    oracle_values = []
    for dataset in DATASETS:
        modality_rows = [
            summary_by_key[(dataset, modality)] for modality in ("text", "visual")
            if (dataset, modality) in summary_by_key
        ]
        oracle_values.append(
            float(np.mean([_f(row["oracle_relative_gap"]) for row in modality_rows]))
            if modality_rows
            else 0.0
        )

    axis = axes[2]
    axis.bar(x - 0.18, disagreement_values, width=0.32, color=PALETTE["disagreement"], label="T/V disagreement")
    axis.bar(x + 0.18, oracle_values, width=0.32, color=PALETTE["oracle"], label="Mean relative oracle gap")
    axis.set_title("Heterogeneity summary", pad=7, fontweight="bold")
    axis.set_xticks(x, DATASETS, rotation=25, ha="right")
    for tick in axis.get_xticklabels():
        tick.set_rotation_mode("anchor")
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("Ratio / relative CE gap")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
    axis.set_axisbelow(True)
    axis.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            -0.20,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.suptitle(
        "M1 contextualization demand: node-wise preferred local/context mixtures",
        fontsize=9,
        fontweight="bold",
        y=1.03,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure1b_context_demand"
    require_matplotlib_panel_alignment(
        fig,
        json_out=f"{stem}.alignment.json",
        overlay_svg=f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    alignment_payload = json.loads(Path(f"{stem}.alignment.json").read_text(encoding="utf-8"))
    Path(f"{stem}.alignment-layout.json").write_text(
        json.dumps(alignment_payload["layout"], indent=2), encoding="utf-8"
    )
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return Path(f"{stem}.png"), Path(f"{stem}.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/final/context_demand"),
        help="Directory containing M1 CSV outputs and receiving the figure files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    png, pdf = render(args.output_dir)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))
