#!/usr/bin/env python3
"""Plot degree-controlled relation--utilization associations from a real CSV.

The plot shows all three seed-level values for each dataset and modality, with
an overlaid mean +/- sample standard deviation summary. No values are
simulated or manually entered.
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
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = Path(
    "/hdd1/DataInHere/YHF/MoPF/outputs/e0_empirical_motivation/"
    "relation_utilization_bridge/relation_utilization_association.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_figures/relation_utilization_association"

DATASET_ORDER = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
MODALITY_ORDER = ["text", "visual"]
MODALITY_LABELS = {"text": "Text", "visual": "Visual"}
REQUIRED_FIELDS = {
    "dataset",
    "seed",
    "modality",
    "rho_partial_degree",
    "degenerate_partial",
}

TEXT_BLUE = "#2F6FB3"
VISUAL_GREEN = "#3B9960"
INK = "#20252B"


def _read_association_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        missing = sorted(REQUIRED_FIELDS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV is missing required fields: {missing}")
        return list(reader)


def _parse_int(value: str, *, field: str, row_number: int, path: Path) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric value in {path}, row {row_number}, field {field}: {value!r}"
        ) from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"Expected a finite integer in {path}, row {row_number}, field {field}")
    return int(number)


def _parse_float(value: str, *, field: str, row_number: int, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric value in {path}, row {row_number}, field {field}: {value!r}"
        ) from exc
    if not np.isfinite(number):
        raise ValueError(f"Non-finite value in {path}, row {row_number}, field {field}")
    return number


def load_groups(path: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    rows = _read_association_rows(path)
    groups: dict[tuple[str, str], list[tuple[int, float]]] = {}

    for row_number, row in enumerate(rows, start=2):
        dataset = row["dataset"].strip()
        modality = row["modality"].strip().lower()
        if dataset not in DATASET_ORDER:
            raise ValueError(f"Unexpected dataset {dataset!r} in row {row_number}")
        if modality not in MODALITY_ORDER:
            raise ValueError(f"Unexpected modality {modality!r} in row {row_number}")
        if row["degenerate_partial"].strip().lower() != "false":
            raise ValueError(
                "Refusing to plot a degenerate degree-controlled partial association in "
                f"{path}, row {row_number}"
            )

        seed = _parse_int(row["seed"], field="seed", row_number=row_number, path=path)
        rho = _parse_float(
            row["rho_partial_degree"],
            field="rho_partial_degree",
            row_number=row_number,
            path=path,
        )
        groups.setdefault((dataset, modality), []).append((seed, rho))

    expected_keys = {
        (dataset, modality)
        for dataset in DATASET_ORDER
        for modality in MODALITY_ORDER
    }
    if set(groups) != expected_keys:
        missing = sorted(expected_keys.difference(groups))
        extra = sorted(set(groups).difference(expected_keys))
        raise ValueError(f"Dataset/modality mismatch; missing={missing}, extra={extra}")

    for key, entries in groups.items():
        seeds = [seed for seed, _ in entries]
        if len(entries) != 3 or len(set(seeds)) != 3:
            raise ValueError(
                f"Expected exactly three unique seed rows for {key[0]}/{key[1]}, "
                f"found seeds={seeds}"
            )
        groups[key] = sorted(entries, key=lambda item: item[0])
    return groups


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
    groups: dict[tuple[str, str], list[tuple[int, float]]],
    *,
    title: str,
    output_root: Path,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / "relation_utilization_association"
    _configure_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    x_positions = np.arange(1, len(DATASET_ORDER) + 1, dtype=float)
    modality_offset = {"text": -0.16, "visual": 0.16}
    seed_offsets = np.asarray([-0.055, 0.0, 0.055], dtype=float)
    colors = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}
    all_values: list[np.ndarray] = []

    for dataset_index, dataset in enumerate(DATASET_ORDER):
        for modality in MODALITY_ORDER:
            entries = groups[(dataset, modality)]
            values = np.asarray([value for _, value in entries], dtype=float)
            all_values.append(values)
            center = x_positions[dataset_index] + modality_offset[modality]

            ax.scatter(
                center + seed_offsets,
                values,
                s=28,
                color=colors[modality],
                edgecolor="white",
                linewidth=0.65,
                alpha=0.86,
                zorder=3,
            )

            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1))
            ax.errorbar(
                center,
                mean,
                yerr=standard_deviation,
                fmt="o",
                markersize=8.2,
                markerfacecolor=colors[modality],
                markeredgecolor="white",
                markeredgewidth=1.1,
                ecolor=colors[modality],
                elinewidth=1.35,
                capsize=3.2,
                capthick=1.35,
                zorder=5,
            )

    combined = np.concatenate(all_values)
    data_min = float(np.min(combined))
    data_max = float(np.max(combined))
    span = max(data_max - data_min, 0.05)
    lower = data_min - 0.14 * span
    upper = data_max + 0.16 * span
    # Keep the zero-reference line visible even when all observed values have
    # the same sign; these limits are axis margins, not data values.
    lower = min(lower, -0.02)
    upper = max(upper, 0.02)
    ax.set_ylim(lower, upper)
    ax.set_xlim(0.52, len(DATASET_ORDER) + 0.48)
    ax.axhline(0.0, color=INK, linestyle=(0, (4, 3)), linewidth=0.9, alpha=0.75, zorder=1)
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(r"Degree-controlled partial Spearman $\rho$")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(DATASET_ORDER, rotation=22, ha="right", rotation_mode="anchor")
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", markersize=5.6,
                markerfacecolor=TEXT_BLUE, markeredgecolor="white",
                label="Text",
            ),
            Line2D(
                [], [], marker="o", linestyle="none", markersize=5.6,
                markerfacecolor=VISUAL_GREEN, markeredgecolor="white",
                label="Visual",
            ),
            Line2D(
                [], [], marker="o", linestyle="-", color=INK, markersize=6.6,
                markerfacecolor=TEXT_BLUE, markeredgecolor="white",
                label="Mean $\u00b1$ SD",
            ),
        ],
        loc="upper right",
        ncol=3,
        handlelength=1.35,
        columnspacing=1.15,
    )
    ax.tick_params(axis="x", pad=4)

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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--title",
        default="Local relation condition and aggregation utilization association",
        help="Figure title; plotted values remain read from the input CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = load_groups(args.input)
    outputs = build_figure(groups, title=args.title, output_root=args.output_root)
    print(f"Read {sum(len(entries) for entries in groups.values())} seed-level rows")
    print("Plotted field: rho_partial_degree")
    print(f"Wrote: {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
