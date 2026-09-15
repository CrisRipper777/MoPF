#!/usr/bin/env python3
"""Plot seed-mean per-node aggregation-layer utilization distributions.

The violin values are read from each dataset's
``seed_mean_node_multihop_utilization.npz``. The accompanying summary CSV is
used to validate dataset/modality metadata and node counts before rendering.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SUMMARY = Path(
    "/hdd1/DataInHere/YHF/MoPF/outputs/e0_empirical_motivation/"
    "multihop_utilization/multihop_utilization_summary.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_figures/aggregation_structure_heterogeneity"

DATASET_ORDER = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
MODALITIES = {"text", "visual"}
SUMMARY_FIELDS = {
    "dataset",
    "modality",
    "seed",
    "num_nodes",
    "finite_status",
    "all_zero_contribution_profile_count",
}
NPZ_FIELDS = {"normalized_order_text", "normalized_order_visual"}

TEXT_BLUE = "#2F6FB3"
VISUAL_GREEN = "#3B9960"
INK = "#20252B"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _as_float(value: str, *, field: str, row_number: int, path: Path) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"Empty value in {path}, row {row_number}, field {field}")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value in {path}, row {row_number}, field {field}: {value!r}"
        ) from exc
    if not np.isfinite(number):
        raise ValueError(f"Non-finite value in {path}, row {row_number}, field {field}")
    return number


def _as_int(value: str, *, field: str, row_number: int, path: Path) -> int:
    number = _as_float(value, field=field, row_number=row_number, path=path)
    if not number.is_integer():
        raise ValueError(f"Expected an integer in {path}, row {row_number}, field {field}")
    return int(number)


def load_summary_metadata(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    fields, rows = _read_csv(path)
    missing = sorted(SUMMARY_FIELDS.difference(fields))
    if missing:
        raise ValueError(f"Summary file is missing required fields: {missing}")

    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        dataset = row["dataset"].strip()
        modality = row["modality"].strip().lower()
        if dataset not in DATASET_ORDER:
            raise ValueError(f"Unexpected dataset {dataset!r} in row {row_number}")
        if modality not in MODALITIES:
            raise ValueError(f"Unexpected modality {modality!r} in row {row_number}")

        key = (dataset, modality)
        entry = metadata.setdefault(
            key,
            {
                "dataset": dataset,
                "modality": modality,
                "seeds": set(),
                "num_nodes": set(),
            },
        )
        entry["seeds"].add(_as_int(row["seed"], field="seed", row_number=row_number, path=path))
        entry["num_nodes"].add(
            _as_int(row["num_nodes"], field="num_nodes", row_number=row_number, path=path)
        )
        if row["finite_status"].strip().lower() != "true":
            raise ValueError(f"finite_status is not True in {path}, row {row_number}")
        if _as_int(
            row["all_zero_contribution_profile_count"],
            field="all_zero_contribution_profile_count",
            row_number=row_number,
            path=path,
        ) != 0:
            raise ValueError(
                "Summary reports all-zero contribution profiles; refusing to plot an "
                f"unqualified distribution from {path}, row {row_number}"
            )

    expected_keys = {(dataset, modality) for dataset in DATASET_ORDER for modality in MODALITIES}
    if set(metadata) != expected_keys:
        missing_keys = sorted(expected_keys.difference(metadata))
        extra_keys = sorted(set(metadata).difference(expected_keys))
        raise ValueError(f"Summary dataset/modality mismatch; missing={missing_keys}, extra={extra_keys}")
    for entry in metadata.values():
        if len(entry["num_nodes"]) != 1:
            raise ValueError(f"Inconsistent num_nodes for {entry['dataset']}/{entry['modality']}")
    return metadata


def load_seed_mean_distributions(
    npz_root: Path,
    metadata: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], np.ndarray]:
    distributions: dict[tuple[str, str], np.ndarray] = {}
    for dataset in DATASET_ORDER:
        path = npz_root / dataset / "seed_mean_node_multihop_utilization.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Required seed-mean NPZ file does not exist: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            missing = sorted(NPZ_FIELDS.difference(loaded.files))
            if missing:
                raise ValueError(f"NPZ file is missing required fields: {missing}: {path}")
            arrays = {
                "text": np.asarray(loaded["normalized_order_text"], dtype=float),
                "visual": np.asarray(loaded["normalized_order_visual"], dtype=float),
            }

        for modality, values in arrays.items():
            key = (dataset, modality)
            expected_nodes = next(iter(metadata[key]["num_nodes"]))
            if values.ndim != 1 or values.size != expected_nodes:
                raise ValueError(
                    f"{path}: {modality} distribution shape {values.shape} does not match "
                    f"summary num_nodes={expected_nodes}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {modality} distribution contains non-finite values")
            tolerance = 1e-9
            if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
                raise ValueError(f"{path}: {modality} normalized order is outside [0, 1]")
            distributions[key] = values
    return distributions


def _configure_style() -> None:
    # Keep text editable in SVG/PDF and use a publication-scale sans-serif fallback.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.8,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
        }
    )


def build_figure(
    distributions: dict[tuple[str, str], np.ndarray],
    *,
    title: str,
    output_root: Path,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / "aggregation_structure_heterogeneity"
    _configure_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    positions = np.arange(1, len(DATASET_ORDER) + 1, dtype=float)
    offset = 0.18
    violin_width = 0.30
    colors = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}

    for dataset in DATASET_ORDER:
        for modality in ("text", "visual"):
            values = distributions[(dataset, modality)]
            position = positions[DATASET_ORDER.index(dataset)] + (-offset if modality == "text" else offset)
            parts = ax.violinplot(
                values,
                positions=[position],
                widths=violin_width,
                points=100,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(colors[modality])
                body.set_edgecolor(INK)
                body.set_linewidth(0.55)
                body.set_alpha(0.72)
            median = float(np.median(values))
            ax.scatter(
                [position],
                [median],
                s=22,
                marker="o",
                facecolor="white",
                edgecolor=INK,
                linewidth=0.9,
                zorder=4,
            )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.45, len(DATASET_ORDER) + 0.55)
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    ax.set_ylabel("Normalized contribution-weighted response order")
    ax.set_xticks(positions)
    ax.set_xticklabels(DATASET_ORDER, rotation=22, ha="right", rotation_mode="anchor")
    ax.legend(
        handles=[
            Patch(facecolor=TEXT_BLUE, edgecolor=INK, alpha=0.72, label="Text"),
            Patch(facecolor=VISUAL_GREEN, edgecolor=INK, alpha=0.72, label="Visual"),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=INK,
                label="Median",
            ),
        ],
        loc="upper right",
        ncol=3,
        handlelength=1.2,
        columnspacing=1.1,
    )
    ax.tick_params(axis="x", pad=4)
    ax.set_axisbelow(True)

    # Semantic reading cues; these annotations do not alter the distributions.
    ax.text(
        -0.13,
        0.98,
        "More high-order context ↑",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=INK,
        clip_on=False,
    )
    ax.text(
        -0.13,
        0.02,
        "More ego / low-order context ↓",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=INK,
        clip_on=False,
    )

    # Callout 1: Grocery shows a clear Text--Visual displacement.
    ax.annotate(
        "",
        xy=(3.18, 0.82),
        xytext=(2.45, 0.91),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.85},
        zorder=6,
    )
    ax.text(
        2.45,
        0.985,
        "Strong modality difference",
        ha="center",
        va="top",
        fontsize=7.4,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": INK, "linewidth": 0.55, "alpha": 0.90, "pad": 2.0},
        zorder=6,
    )

    # Callout 2: Ele-fashion has broad node-level violin spread.
    ax.annotate(
        "",
        xy=(4.02, 0.44),
        xytext=(4.62, 0.40),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.85},
        zorder=6,
    )
    ax.text(
        4.62,
        0.29,
        "Node-level dispersion",
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": INK, "linewidth": 0.55, "alpha": 0.90, "pad": 2.0},
        zorder=6,
    )

    ax.text(
        0.02,
        0.035,
        "Vertical spread → node heterogeneity\nText–Visual offset → modality heterogeneity",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color=INK,
        bbox={
            "facecolor": "white",
            "edgecolor": INK,
            "linewidth": 0.6,
            "alpha": 0.92,
            "boxstyle": "round,pad=0.35",
        },
        zorder=7,
    )

    fig.tight_layout(pad=1.2)
    outputs = [
        base.with_suffix(".png"),
        base.with_suffix(".pdf"),
        base.with_suffix(".svg"),
        base.with_suffix(".tiff"),
    ]
    fig.savefig(outputs[0], dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[3], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--npz-root",
        type=Path,
        default=None,
        help="Directory containing one dataset subdirectory per seed-mean NPZ; defaults to summary parent.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--title",
        default="Aggregation-layer structural utilization heterogeneity",
        help="Figure title; plotted distributions remain read from the NPZ files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_summary_metadata(args.summary)
    npz_root = args.summary.parent if args.npz_root is None else args.npz_root
    distributions = load_seed_mean_distributions(npz_root, metadata)
    outputs = build_figure(distributions, title=args.title, output_root=args.output_root)
    print(f"Read summary dataset/modality groups: {len(metadata)}")
    print("Read complete seed-mean per-node distributions from normalized_order_text/visual")
    print(f"Wrote: {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
