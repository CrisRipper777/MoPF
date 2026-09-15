#!/usr/bin/env python3
"""Plot Grocery relation-condition quartiles against response order.

All plotted values are read from the supplied CSV. When three seed-level
curves are available, they are shown as faint lines and their arithmetic mean
is overlaid as the main line. The primary figure is a Q1-centered
visualization; the original absolute-order version is also exported as an
appendix figure.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = Path(
    "/hdd1/DataInHere/YHF/MoPF/outputs/e0_empirical_motivation/"
    "relation_utilization_bridge/relation_condition_quartiles.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_figures/relation_condition_quartiles"

DATASET = "Grocery"
QUARTILE_ORDER = ["Q1", "Q2", "Q3", "Q4"]
MODALITY_ORDER = ["text", "visual"]
MODALITY_LABELS = {"text": "Text", "visual": "Visual"}

REQUIRED_FIELDS = {
    "dataset",
    "seed",
    "modality",
    "formal_K",
    "quartile",
    "order_mean",
}

TEXT_BLUE = "#2F6FB3"
VISUAL_GREEN = "#3B9960"
INK = "#20252B"


def _read_rows(path: Path) -> list[dict[str, str]]:
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


def load_grocery_values(path: Path) -> tuple[dict[str, dict[int, np.ndarray]], int]:
    """Return values indexed as modality -> seed -> Q1..Q4 array."""

    rows = _read_rows(path)
    values: dict[tuple[int, str], dict[str, float]] = {}
    formal_ks: set[int] = set()

    for row_number, row in enumerate(rows, start=2):
        if row["dataset"].strip() != DATASET:
            continue
        modality = row["modality"].strip().lower()
        quartile = row["quartile"].strip()
        if modality not in MODALITY_ORDER:
            raise ValueError(f"Unexpected Grocery modality {modality!r} in row {row_number}")
        if quartile not in QUARTILE_ORDER:
            raise ValueError(f"Unexpected Grocery quartile {quartile!r} in row {row_number}")

        seed = _parse_int(row["seed"], field="seed", row_number=row_number, path=path)
        formal_k = _parse_int(row["formal_K"], field="formal_K", row_number=row_number, path=path)
        order_mean = _parse_float(
            row["order_mean"], field="order_mean", row_number=row_number, path=path
        )
        formal_ks.add(formal_k)
        key = (seed, modality)
        if quartile in values.setdefault(key, {}):
            raise ValueError(
                f"Duplicate Grocery row for seed={seed}, modality={modality}, "
                f"quartile={quartile} in {path}"
            )
        values[key][quartile] = order_mean

    if not values:
        raise ValueError(f"No rows for dataset={DATASET!r} found in {path}")
    if formal_ks != {2}:
        raise ValueError(f"Expected Grocery formal_K=2, found {sorted(formal_ks)}")

    seeds = sorted({seed for seed, _ in values})
    if len(seeds) != 3:
        raise ValueError(f"Expected three Grocery seeds, found {seeds}")
    for modality in MODALITY_ORDER:
        for seed in seeds:
            key = (seed, modality)
            if key not in values or set(values[key]) != set(QUARTILE_ORDER):
                found = sorted(values.get(key, {}))
                raise ValueError(
                    f"Incomplete Grocery quartiles for seed={seed}, modality={modality}; "
                    f"found={found}"
                )

    curves: dict[str, dict[int, np.ndarray]] = {modality: {} for modality in MODALITY_ORDER}
    for modality in MODALITY_ORDER:
        for seed in seeds:
            curves[modality][seed] = np.asarray(
                [values[(seed, modality)][quartile] for quartile in QUARTILE_ORDER],
                dtype=float,
            )
    return curves, 2


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
            "xtick.labelsize": 8.0,
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


def q1_center_curves(
    curves: dict[str, dict[int, np.ndarray]],
) -> dict[str, dict[int, np.ndarray]]:
    """Return visualization-only curves centered on each seed's Q1 value."""

    return {
        modality: {
            seed: np.asarray(seed_values, dtype=float) - float(seed_values[0])
            for seed, seed_values in seed_curves.items()
        }
        for modality, seed_curves in curves.items()
    }


def build_figure(
    curves: dict[str, dict[int, np.ndarray]],
    *,
    formal_k: int,
    title: str,
    output_root: Path,
    output_stem: str,
    relative_to_q1: bool,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / output_stem
    _configure_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    x = np.arange(len(QUARTILE_ORDER), dtype=float)
    colors = {"text": TEXT_BLUE, "visual": VISUAL_GREEN}
    all_values: list[np.ndarray] = []
    display_curves = q1_center_curves(curves) if relative_to_q1 else curves

    for modality in MODALITY_ORDER:
        seed_curves = list(display_curves[modality].items())
        for _, seed_values in seed_curves:
            all_values.append(seed_values)
            ax.plot(
                x,
                seed_values,
                color=colors[modality],
                linewidth=1.0,
                alpha=0.24,
                marker="o",
                markersize=3.0,
                markerfacecolor="white",
                markeredgecolor=colors[modality],
                markeredgewidth=0.6,
                zorder=2,
            )

        seed_matrix = np.vstack([seed_values for _, seed_values in seed_curves])
        mean_values = np.mean(seed_matrix, axis=0)
        ax.plot(
            x,
            mean_values,
            color=colors[modality],
            linewidth=2.4,
            marker="o",
            markersize=6.8,
            markerfacecolor=colors[modality],
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=MODALITY_LABELS[modality],
            zorder=4,
        )

    combined = np.concatenate(all_values)
    data_min = float(np.min(combined))
    data_max = float(np.max(combined))
    span = max(data_max - data_min, 0.05)
    ax.set_ylim(data_min - 0.12 * span, data_max + 0.14 * span)
    ax.set_xlim(-0.15, len(QUARTILE_ORDER) - 0.85)
    ax.set_title(title, loc="left", pad=40 if relative_to_q1 else 7, fontweight="bold")
    if relative_to_q1:
        ax.axhline(
            0.0,
            color=INK,
            linestyle=(0, (4, 3)),
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )
        ax.set_xlabel(f"Relation-condition quartile (Grocery; formal K={formal_k})")
        ax.set_ylabel("Change in normalized response order relative to Q1")
        ax.text(
            0.0,
            1.04,
            "More high-order utilization ↑",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=INK,
            fontsize=8.3,
        )
        ax.text(
            0.5,
            -0.23,
            "Q1 = weaker local relation condition\nQ4 = stronger local relation condition",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=INK,
            fontsize=7.8,
            linespacing=1.25,
        )
    else:
        ax.set_xlabel(f"Relation-condition quartile (Grocery; formal K={formal_k})")
        ax.set_ylabel("Mean normalized response order")
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTILE_ORDER)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    legend_kwargs = {
        "handles": [
            Line2D([], [], color=TEXT_BLUE, linewidth=2.4, marker="o", markersize=5.8,
                   markerfacecolor=TEXT_BLUE, markeredgecolor="white", label="Text"),
            Line2D([], [], color=VISUAL_GREEN, linewidth=2.4, marker="o", markersize=5.8,
                   markerfacecolor=VISUAL_GREEN, markeredgecolor="white", label="Visual"),
            Line2D([], [], color=INK, linewidth=1.0, alpha=0.35, marker="o", markersize=3.0,
                   markerfacecolor="white", markeredgecolor=INK, label="Seed-level"),
        ],
        "ncol": 3,
        "handlelength": 1.3,
        "columnspacing": 1.1,
    }
    if relative_to_q1:
        legend_kwargs.update(loc="lower left", bbox_to_anchor=(0.0, 1.08))
    else:
        legend_kwargs["loc"] = "upper left"
    ax.legend(**legend_kwargs)

    if relative_to_q1:
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.80)
    else:
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
        default="Stronger local relation condition is associated with higher-order utilization",
        help="Figure title; plotted values remain read from the input CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curves, formal_k = load_grocery_values(args.input)
    relative_outputs = build_figure(
        curves,
        formal_k=formal_k,
        title=args.title,
        output_root=args.output_root,
        output_stem="relation_condition_quartiles_relative",
        relative_to_q1=True,
    )
    absolute_outputs = build_figure(
        curves,
        formal_k=formal_k,
        title="Original absolute response order (appendix)",
        output_root=args.output_root,
        output_stem="relation_condition_quartiles_absolute_order",
        relative_to_q1=False,
    )
    seed_count = len(next(iter(curves.values())))
    print(f"Read dataset={DATASET}, formal_K={formal_k}, seeds={seed_count}")
    print("Plotted field: order_mean")
    print("Relative transformation: order_mean(Qq) - order_mean(Q1), per seed and modality")
    print(f"Wrote relative figure: {', '.join(str(path) for path in relative_outputs)}")
    print(f"Wrote absolute appendix: {', '.join(str(path) for path in absolute_outputs)}")


if __name__ == "__main__":
    main()
