#!/usr/bin/env python3
"""Plot representation shift at each dataset's formal propagation order.

The plotted values are computed as ``1 - cka_formal_K`` from the supplied
semantic-retention summary CSV. The script validates the requested schema and
formal-K assignments before creating any output.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = Path(
    "/hdd1/DataInHere/YHF/MoPF/outputs/e0_empirical_motivation/"
    "semantic_retention/semantic_retention_summary.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/paper_figures/propagation_semantic_drift"

DATASET_ORDER = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
EXPECTED_FORMAL_K = {
    "Movies": 3,
    "Toys": 3,
    "Grocery": 2,
    "ele-fashion": 3,
    "Reddit-S": 3,
}
EXPECTED_MODALITIES = {"text", "visual"}
REQUIRED_FIELDS = {"dataset", "modality", "formal_K", "cka_formal_K"}

TEXT_BLUE = "#0F6CBD"
VISUAL_GREEN = "#159B63"
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


def load_formal_k_shift(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    fields, rows = _read_csv(path)
    missing = sorted(REQUIRED_FIELDS.difference(fields))
    if missing:
        raise ValueError(f"Input file is missing required fields: {missing}")

    expected_keys = {(dataset, modality) for dataset in DATASET_ORDER for modality in EXPECTED_MODALITIES}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        dataset = row["dataset"].strip()
        modality = row["modality"].strip().lower()
        key = (dataset, modality)
        if dataset not in DATASET_ORDER:
            raise ValueError(f"Unexpected dataset {dataset!r} in row {row_number}")
        if modality not in EXPECTED_MODALITIES:
            raise ValueError(f"Unexpected modality {modality!r} in row {row_number}")
        if key in records:
            raise ValueError(f"Duplicate dataset/modality pair {key!r} in row {row_number}")

        formal_k = _as_int(row["formal_K"], field="formal_K", row_number=row_number, path=path)
        cka_formal = _as_float(
            row["cka_formal_K"],
            field="cka_formal_K",
            row_number=row_number,
            path=path,
        )
        if not 0.0 <= cka_formal <= 1.0:
            raise ValueError(
                f"cka_formal_K must be in [0, 1], observed {cka_formal} in row {row_number}"
            )
        records[key] = {
            "dataset": dataset,
            "modality": modality,
            "formal_K": formal_k,
            "cka_formal_K": cka_formal,
            "representation_shift": 1.0 - cka_formal,
        }

    if set(records) != expected_keys:
        missing_keys = sorted(expected_keys.difference(records))
        extra_keys = sorted(set(records).difference(expected_keys))
        raise ValueError(f"Dataset/modality mismatch; missing={missing_keys}, extra={extra_keys}")

    for dataset in DATASET_ORDER:
        dataset_records = [records[(dataset, modality)] for modality in EXPECTED_MODALITIES]
        observed_k = {record["formal_K"] for record in dataset_records}
        if observed_k != {EXPECTED_FORMAL_K[dataset]}:
            raise ValueError(
                f"formal_K mismatch for {dataset}: observed {sorted(observed_k)}, "
                f"expected {EXPECTED_FORMAL_K[dataset]}"
            )
    return records


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
            "font.size": 9.5,
            "axes.titlesize": 12.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
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
    records: dict[tuple[str, str], dict[str, Any]],
    *,
    title: str,
    output_root: Path,
) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / "propagation_semantic_drift"
    _configure_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    x = np.arange(len(DATASET_ORDER), dtype=float)
    width = 0.34
    text_shift = np.asarray(
        [records[(dataset, "text")]["representation_shift"] for dataset in DATASET_ORDER],
        dtype=float,
    )
    visual_shift = np.asarray(
        [records[(dataset, "visual")]["representation_shift"] for dataset in DATASET_ORDER],
        dtype=float,
    )

    bars_text = ax.bar(
        x - width / 2,
        text_shift,
        width,
        label="Text",
        color=TEXT_BLUE,
        edgecolor=INK,
        linewidth=0.45,
    )
    bars_visual = ax.bar(
        x + width / 2,
        visual_shift,
        width,
        label="Visual",
        color=VISUAL_GREEN,
        edgecolor=INK,
        linewidth=0.45,
    )

    ymax = max(float(np.max(text_shift)), float(np.max(visual_shift)))
    ylim = (0.0, min(1.12, max(1.02, ymax + 0.12)))
    ax.set_ylim(*ylim)
    label_pad = 0.018 * (ylim[1] - ylim[0])
    for bars in (bars_text, bars_visual):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_pad,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=INK,
            )

    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    ax.set_ylabel("Representation shift at formal K (1 - CKA)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{dataset}\nformal K={EXPECTED_FORMAL_K[dataset]}" for dataset in DATASET_ORDER],
        rotation=0,
        rotation_mode="anchor",
    )
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=1.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=4)

    # Presentation-only directional cue; it does not alter the plotted values.
    ax.annotate(
        "",
        xy=(0.975, 0.90),
        xycoords="axes fraction",
        xytext=(0.975, 0.73),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.9},
        zorder=6,
    )
    ax.text(
        0.94,
        0.725,
        "larger = stronger\nrepresentation drift ↑",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
        zorder=6,
    )

    fig.tight_layout(rect=(0.0, 0.085, 1.0, 1.0), pad=1.2)
    fig.text(
        0.5,
        0.018,
        "0 = representation geometry preserved    ─────────    1 = strongly reshaped",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=INK,
    )
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
        default="Propagation-layer semantic drift at formal K",
        help="Figure title; plotted values remain read from the input CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_formal_k_shift(args.input)
    outputs = build_figure(records, title=args.title, output_root=args.output_root)
    print(f"Read rows: {len(records)}")
    print("Computed representation shift as 1 - cka_formal_K")
    print(f"Wrote: {', '.join(str(path) for path in outputs)}")


if __name__ == "__main__":
    main()
