#!/usr/bin/env python3
"""Render the final, three-panel CoSI-MAG Figure 4.

The plotter consumes the final-clean frozen-inference CSVs written by
``analyze_figure4_final_clean.py``. It does not load a checkpoint and never
starts training or benchmark evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "paper_figures" / "figure4_rcmi"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR
NC_DATASETS = ("Movies", "Grocery")
PANEL_C_DATASETS = ("Movies", "Grocery", "Sports-LP")
SEEDS = (42, 43, 44)
INTERVENTIONS = ("Query-Collapse", "Uniform-Attention", "Interaction-Off")
QUANTITIES = ("Text Attn.", "Visual Attn.", "Z", "Logits")

TEXT_BLUE = "#4C78A8"
VISUAL_GREEN = "#59A14F"
MOVIES_BLUE = "#4C78A8"
GROCERY_ORANGE = "#D78350"
SPORTS_PURPLE = "#8172B2"
INK = "#24272A"
NEUTRAL = "#74797D"
GRID = "#E7E9EB"

FIGSIZE = (17.0, 4.8)
WIDTH_RATIOS = (1.0, 1.05, 1.0)
WSPACE = 0.20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Figure 4 source CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str, path: Path) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key!r} in {path}: {row.get(key)!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key!r} in {path}: {value}")
    return value


def _configure_style() -> None:
    # These values follow Figure 1 empirical v5: serif text, modest titles,
    # light grids, thin spines, editable PDF text, and a white canvas.
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
            # Keep an explicit safe fallback declaration for the shared figure
            # preflight while retaining serif as the active publication family.
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.2,
            "axes.labelweight": "normal",
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
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.solid_capstyle": "round",
        }
    )


def _format_axis(ax: Any, grid_axis: str = "y") -> None:
    ax.tick_params(length=2.4, width=0.6, pad=2.3)
    ax.grid(
        axis=grid_axis,
        color=GRID,
        linewidth=0.45,
        linestyle=(0, (1.2, 2.0)),
        alpha=0.9,
        zorder=0,
    )
    ax.set_axisbelow(True)


def _add_panel_label(ax: Any, label: str) -> None:
    # Keep the letter independent from the one-line title, as in Figure 1.
    ax.text(
        -0.12,
        1.045,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def _load_a(data_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    path = data_dir / "panel_a_node_mean_r_attn.csv"
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for row in _read_csv(path):
        dataset = row.get("dataset", "")
        modality = row.get("modality", "")
        if dataset not in NC_DATASETS or modality not in {"text", "visual"}:
            continue
        key = (dataset, int(row["node"]), modality)
        if key in seen:
            raise ValueError(f"Duplicate panel (a) node row: {key}")
        seen.add(key)
        value = _number(row, "r_attn_mean_across_seeds", path)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Panel (a) r_attn outside [0,1]: {value}")
        values[(dataset, modality)].append(value)
    output = {key: np.asarray(item, dtype=np.float64) for key, item in values.items()}
    for dataset in NC_DATASETS:
        for modality in ("text", "visual"):
            if (dataset, modality) not in output or output[(dataset, modality)].size == 0:
                raise ValueError(f"Missing panel (a) values for {dataset}/{modality}")
    return output


def _load_b(data_dir: Path) -> dict[tuple[str, int, str], float]:
    path = data_dir / "panel_b_seed_z_mae.csv"
    values: dict[tuple[str, int, str], float] = {}
    for row in _read_csv(path):
        dataset = row.get("dataset", "")
        condition = row.get("condition", "")
        if dataset not in NC_DATASETS or condition not in INTERVENTIONS:
            continue
        key = (dataset, int(row["seed"]), condition)
        if key in values:
            raise ValueError(f"Duplicate panel (b) row: {key}")
        value = _number(row, "z_mae", path)
        if value < 0.0:
            raise ValueError(f"Negative Z MAE in panel (b): {value}")
        values[key] = value
    for dataset in NC_DATASETS:
        for seed in SEEDS:
            for condition in INTERVENTIONS:
                if (dataset, seed, condition) not in values:
                    raise ValueError(f"Missing panel (b) row: {dataset}/{seed}/{condition}")
    return values


def _load_c(data_dir: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    path = data_dir / "panel_c_seed_excess_impact.csv"
    values: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in _read_csv(path):
        dataset = row.get("dataset", "")
        quantity = row.get("quantity", "")
        if dataset not in PANEL_C_DATASETS or quantity not in QUANTITIES:
            continue
        key = (dataset, int(row["seed"]), quantity)
        if key in values:
            raise ValueError(f"Duplicate panel (c) row: {key}")
        values[key] = {
            "off": _number(row, "off_impact", path),
            "shuffle": _number(row, "shuffle_impact", path),
            "ratio": _number(row, "shuffle_over_off", path),
            "excess": _number(row, "excess_impact_pct", path),
        }
    for dataset in PANEL_C_DATASETS:
        for seed in SEEDS:
            for quantity in QUANTITIES:
                if (dataset, seed, quantity) not in values:
                    raise ValueError(f"Missing panel (c) row: {dataset}/{seed}/{quantity}")
    return values


def _panel_a(ax: Any, values: dict[tuple[str, str], np.ndarray]) -> None:
    positions: list[float] = []
    arrays: list[np.ndarray] = []
    colors: list[str] = []
    for group, dataset in enumerate(NC_DATASETS):
        for offset, modality, color in ((-0.17, "text", TEXT_BLUE), (0.17, "visual", VISUAL_GREEN)):
            positions.append(group + offset)
            arrays.append(values[(dataset, modality)])
            colors.append(color)
    violins = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.29,
        bw_method=0.22,
        points=120,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.48)
        body.set_linewidth(0.55)
    for position, array in zip(positions, arrays):
        q25, median, q75 = np.quantile(array, [0.25, 0.50, 0.75])
        box_width = 0.085
        ax.add_patch(
            Rectangle(
                (position - box_width / 2.0, q25),
                box_width,
                max(float(q75 - q25), 1e-6),
                facecolor="white",
                edgecolor=INK,
                linewidth=0.65,
                zorder=4,
            )
        )
        ax.hlines(
            median,
            position - box_width / 2.0,
            position + box_width / 2.0,
            color=INK,
            linewidth=1.15,
            zorder=5,
        )
    ax.set_xticks([0.0, 1.0], labels=list(NC_DATASETS))
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylabel("Normalized attended order r_attn", labelpad=4)
    ax.set_title("Attended-Order Heterogeneity", loc="left", pad=7, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor=TEXT_BLUE, edgecolor=TEXT_BLUE, alpha=0.48, label="Text"),
            Patch(facecolor=VISUAL_GREEN, edgecolor=VISUAL_GREEN, alpha=0.48, label="Visual"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
        borderaxespad=0.0,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.65,
        borderpad=0.1,
    )
    _format_axis(ax)


def _panel_b(ax: Any, values: dict[tuple[str, int, str], float]) -> None:
    x = np.arange(len(INTERVENTIONS), dtype=float)
    specs = (("Movies", MOVIES_BLUE, "o"), ("Grocery", GROCERY_ORANGE, "D"))
    for dataset, color, marker in specs:
        matrix = np.asarray(
            [[values[(dataset, seed, condition)] for condition in INTERVENTIONS] for seed in SEEDS],
            dtype=np.float64,
        )
        for trajectory in matrix:
            ax.plot(x, trajectory, color=color, linewidth=0.60, alpha=0.30, zorder=1)
        mean = matrix.mean(axis=0)
        ax.plot(
            x,
            mean,
            color=color,
            marker=marker,
            markersize=4.3,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.65,
            zorder=3,
        )
    ax.set_xticks(x, labels=INTERVENTIONS)
    ax.set_xlim(-0.18, 2.18)
    raw_max = max(values.values())
    ax.set_ylim(0.0, max(raw_max * 1.10, 1e-4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_ylabel("Representation shift ΔZ (MAE)", labelpad=4)
    ax.set_title("Interaction Intervention", loc="left", pad=7, fontweight="bold")
    ax.legend(
        handles=[
            Line2D([0], [0], color=MOVIES_BLUE, marker="o", linewidth=1.45, markersize=3.8, label="Movies"),
            Line2D([0], [0], color=GROCERY_ORANGE, marker="D", linewidth=1.45, markersize=3.7, label="Grocery"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
        borderaxespad=0.0,
        handletextpad=0.35,
        columnspacing=0.7,
        borderpad=0.1,
    )
    _format_axis(ax)


def _panel_c(ax: Any, values: dict[tuple[str, int, str], dict[str, float]]) -> None:
    x = np.arange(len(QUANTITIES), dtype=float)
    specs = (("Movies", MOVIES_BLUE, "o"), ("Grocery", GROCERY_ORANGE, "D"), ("Sports-LP", SPORTS_PURPLE, "^"))
    for dataset, color, marker in specs:
        for seed_index, seed in enumerate(SEEDS):
            jitter = (seed_index - 1) * 0.035
            points = [values[(dataset, seed, quantity)]["excess"] for quantity in QUANTITIES]
            ax.scatter(x + jitter, points, s=13, marker=marker, color=color, alpha=0.30, linewidth=0, zorder=2)
        means = [np.mean([values[(dataset, seed, quantity)]["excess"] for seed in SEEDS]) for quantity in QUANTITIES]
        ax.plot(x, means, color=color, marker=marker, linestyle="None", markersize=5.0,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.65, zorder=4)
    all_values = [item["excess"] for item in values.values()]
    minimum, maximum = min(all_values), max(all_values)
    lower = min(0.0, minimum * 1.12 if minimum < 0.0 else 0.0)
    upper = max(0.0, maximum * 1.12 if maximum > 0.0 else 0.0)
    if upper <= lower:
        upper = lower + 1.0
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    ax.set_xticks(x, labels=list(QUANTITIES))
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Excess impact of Shuffle over Off (%)", labelpad=4)
    ax.set_title("Relation-Context Dependency", loc="left", pad=7, fontweight="bold")
    ax.legend(
        handles=[
            Line2D([0], [0], color=MOVIES_BLUE, marker="o", linestyle="None", markersize=4.3, label="Movies"),
            Line2D([0], [0], color=GROCERY_ORANGE, marker="D", linestyle="None", markersize=4.0, label="Grocery"),
            Line2D([0], [0], color=SPORTS_PURPLE, marker="^", linestyle="None", markersize=4.3, label="Sports-LP"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=3,
        borderaxespad=0.0,
        handletextpad=0.35,
        columnspacing=0.65,
        borderpad=0.1,
    )
    _format_axis(ax)


def _alignment_gate(fig: Any, axes: list[Any], output_dir: Path) -> None:
    skill_scripts = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
    if str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=output_dir / "figure4_rcmi_main.alignment.json",
        overlay_svg=output_dir / "figure4_rcmi_main.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
        axes=axes,
        exemptions=[
            {
                "panels": ["a", "b", "c"],
                "checks": ["panel-width"],
                "reason": "Intentional near-equal width ratios [1.0, 1.05, 1.0] for Figure 4.",
            }
        ],
    )


def _save_figure(fig: Any, output_dir: Path) -> None:
    # Only the publication figure is generated; no standalone/appendix panel.
    fig.savefig(output_dir / "figure4_rcmi_main.png", dpi=600)
    fig.savefig(output_dir / "figure4_rcmi_main.tiff", dpi=600)
    fig.savefig(output_dir / "figure4_rcmi_main.pdf")
    fig.savefig(output_dir / "figure4_rcmi_main.svg")
    plt.close(fig)


def _write_report(output_dir: Path, data_dir: Path, seeds: tuple[int, ...], a: dict[tuple[str, str], np.ndarray], b: dict[tuple[str, int, str], float], c: dict[tuple[str, int, str], dict[str, float]]) -> None:
    manifest_path = data_dir / "final_clean_mechanism_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    analysis_device = manifest.get("device", "unspecified")
    lines = [
        "# Figure 4 generation report — final clean RCMI mechanism analysis", "",
        f"Generated at (UTC): `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`", "",
        "## Audit conclusion", "",
        "The authoritative source is the final-clean `cosi_mag_final` benchmark under "
        "`outputs/cosi_mag_final_benchmark/`; the older `rc_iamoc_final_candidate` "
        "checkpoint outputs are not used. No training, optimizer step, backward pass, "
        "or clean-benchmark rerun was performed.", "",
        "Earlier diagnostic heatmaps, attention visualizations, absolute SD/error-bar "
        "prototypes, and mean-ratio relation summaries are not used as正文 Figure 4. "
        "The final figure keeps exactly the three story-aligned panels: adaptivity, "
        "functional interaction, and cross-stage relation-context dependency.", "",
        "## A. Final checkpoint audit", "",
        "| Task | Dataset | Seed | Checkpoint | Model | Protocol | Selection | Epoch |",
        "|---|---|---:|---|---|---|---|---:|",
    ]
    for row in manifest.get("checkpoint_audit", []):
        lines.append(f"| {row['task']} | {row['dataset']} | {row['seed']} | `{row['checkpoint_path']}` | `{row['model']}` | `{row['protocol']}` | {row['selection']} | {row['epoch']} |")
    lines += [
        "", "NC uses `unified_full_graph_nc_v1` with the formal full graph. Sports-LP uses "
        "`unified_sampled_lp_v1`: the train-only message graph is passed to the current "
        "final inference path, while panel (c) LP logits are produced by the frozen "
        "checkpoint projection and `LinkPredictor` on the fixed test edge set.", "",
        "## B. Panel (a): attended-order heterogeneity", "",
        "The node unit is one node after averaging its `r_attn` across seeds 42/43/44. "
        "The violin contains the node distribution; the embedded white box is Q25–Q75 "
        "and the short black line is the median. No population-SD whisker or mean point "
        "is plotted.", "",
        "| Dataset | Modality | Nodes | Q25 | Median | Q75 |", "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in NC_DATASETS:
        for modality in ("text", "visual"):
            q25, median, q75 = np.quantile(a[(dataset, modality)], [0.25, 0.50, 0.75])
            lines.append(f"| {dataset} | {modality.title()} | {a[(dataset, modality)].size} | {q25:.6f} | {median:.6f} | {q75:.6f} |")
    lines += [
        "", "Interpretation boundary: these distributions support node-level dispersion "
        "and Text/Visual differences, including possible dataset-specific reversals. "
        "They do not support a universal claim that one modality always prefers a "
        "particular order range.", "",
        "## C. Panel (b): interaction intervention audit", "",
        "`Z MAE` is measured against the Normal forward. The thin lines are paired "
        "seed trajectories; the emphasized line is the dataset-level arithmetic mean. "
        "The seed audit is:", "",
        "| Dataset | Seed | Query-Collapse | Uniform-Attention | Interaction-Off | Ordering holds |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset in NC_DATASETS:
        for seed in seeds:
            row = {condition: b[(dataset, seed, condition)] for condition in INTERVENTIONS}
            holds = row[INTERVENTIONS[0]] < row[INTERVENTIONS[1]] < row[INTERVENTIONS[2]]
            lines.append(f"| {dataset} | {seed} | {row[INTERVENTIONS[0]]:.8f} | {row[INTERVENTIONS[1]]:.8f} | {row[INTERVENTIONS[2]]:.8f} | {holds} |")
    lines += ["", "Dataset-level means:", "", "| Dataset | Query-Collapse | Uniform-Attention | Interaction-Off |", "|---|---:|---:|---:|"]
    for dataset in NC_DATASETS:
        means = [np.mean([b[(dataset, seed, condition)] for seed in seeds]) for condition in INTERVENTIONS]
        lines.append(f"| {dataset} | {means[0]:.8f} | {means[1]:.8f} | {means[2]:.8f} |")
    lines += [
        "", "The panel supports progressively larger representation perturbations where the "
        "observed ordering holds. It should be described as functional participation in "
        "the representation computation, not as a causal performance claim.", "",
        "## D. Panel (c): relation-context dependency", "",
        "For every seed and quantity, the plotted statistic is computed first as "
        "`100 * (Shuffle impact / Off impact - 1)`, then summarized across seeds. "
        "No epsilon is used when the Relation-Off denominator is near zero; such a case "
        "would stop generation and be reported explicitly.", "",
        "| Dataset | Seed | Quantity | Off | Shuffle | Excess impact (%) |", "|---|---:|---|---:|---:|---:|",
    ]
    for dataset in PANEL_C_DATASETS:
        for seed in seeds:
            for quantity in QUANTITIES:
                row = c[(dataset, seed, quantity)]
                lines.append(f"| {dataset} | {seed} | {quantity} | {row['off']:.8g} | {row['shuffle']:.8g} | {row['excess']:.4f} |")
    lines += ["", "Sports-LP supports `Shuffle > Off` by quantity/seed:"]
    for quantity in QUANTITIES:
        flags = [c[("Sports-LP", seed, quantity)]["excess"] > 0.0 for seed in seeds]
        lines.append(f"- `{quantity}`: `{flags}`")
    lines += [
        "", "## E. Layout and visual settings", "",
        f"- `figsize = {FIGSIZE}` inches; aspect ratio = `{FIGSIZE[0] / FIGSIZE[1]:.4f}:1`.",
        f"- `GridSpec width_ratios = {list((1.0, 1.05, 1.0))}`; `wspace = 0.20`.",
        "- Figure margins follow Figure 1 v5: `left=0.055`, `right=0.99`, `bottom=0.19`, `top=0.80`.",
        "- Serif font family, 8.2 pt panel titles, 7.2 pt axis labels, 7.0 pt ticks/legend; white background and light y-grid.",
        "- Panel letters are separate upper-left `(a)`, `(b)`, `(c)` labels; titles are single-line and aligned.",
        "- Only `figure4_rcmi_main.png`, `.pdf`, `.tiff`, and editable `.svg` are generated by the final plotter. No appendix figure or fourth panel is produced.",
        "", "## F. Reproduction commands", "", "```bash",
        "cd /hdd1/DataInHere/YHF/MoPF_IAMOC",
        f"PYTHONPATH=src conda run --no-capture-output -n yhf_env python scripts/analyze_figure4_final_clean.py --device {analysis_device}",
        "PYTHONPATH=src conda run --no-capture-output -n yhf_env python scripts/plot_figure4_rcmi.py",
        "```", "",
        "The first command is frozen inference-only; `--audit-only` performs the checkpoint/protocol audit without inference. The second command only reads the generated CSVs and renders the figure.", "",
    ]
    (output_dir / "figure4_generation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_metadata(output_dir: Path, data_dir: Path, seeds: tuple[int, ...]) -> None:
    manifest_path = data_dir / "final_clean_mechanism_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    metadata = {
        "figure": "Figure 4. Analysis of Relation-Conditioned Multi-Order Interaction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_data_dir": str(data_dir),
        "authoritative_checkpoint_root": "outputs/cosi_mag_final_benchmark",
        "model": "cosi_mag_final",
        "datasets": {"panel_a": list(NC_DATASETS), "panel_b": list(NC_DATASETS), "panel_c": list(PANEL_C_DATASETS)},
        "seeds": list(seeds),
        "panels": {
            "a": "Node-level r_attn averaged across seeds, Text/Visual violin with embedded IQR and median.",
            "b": "Per-seed frozen Z MAE trajectories for Query-Collapse, Uniform-Attention and Interaction-Off.",
            "c": "Per-seed excess impact 100*(Shuffle/Off-1) for attention, Z and frozen scorer logits.",
        },
        "figure_size_inches": list(FIGSIZE),
        "width_ratios": list(WIDTH_RATIOS),
        "wspace": WSPACE,
        "training_started": False,
        "appendix_figure_written": False,
        "analysis_device": manifest.get("device"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme = """# Figure 4 — final-clean RCMI mechanism analysis

The authoritative source is the final-clean `cosi_mag_final` benchmark. The
directory contains exactly the three-panel正文 figure source and exports; no
training, clean-benchmark rerun, Cloth-LP result, appendix figure, or fourth
panel is used.

Primary exports:

- `figure4_rcmi_main.png`
- `figure4_rcmi_main.pdf`
- `figure4_generation_report.md`

The six `panel_*.csv` files are the final source tables used by the plotter.
`final_clean_mechanism_manifest.json` records checkpoint hashes, protocols,
fixed LP test-edge scorer settings, and inference-only provenance.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else ROOT / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    if seeds != SEEDS:
        raise ValueError(f"Final Figure 4 requires seeds {SEEDS}; got {seeds}")
    a = _load_a(data_dir)
    b = _load_b(data_dir)
    c = _load_c(data_dir)
    if args.audit_only:
        print(json.dumps({"panel_a": {f"{key[0]}/{key[1]}": int(value.size) for key, value in a.items()}, "panel_b_rows": len(b), "panel_c_rows": len(c)}, indent=2))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=False, gridspec_kw={"width_ratios": WIDTH_RATIOS})
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.19, top=0.80, wspace=WSPACE)
    _panel_a(axes[0], a)
    _panel_b(axes[1], b)
    _panel_c(axes[2], c)
    for axis, label in zip(axes, "abc"):
        _add_panel_label(axis, label)
    _alignment_gate(fig, list(axes), output_dir)
    _save_figure(fig, output_dir)
    _write_metadata(output_dir, data_dir, seeds)
    _write_report(output_dir, data_dir, seeds, a, b, c)
    print(f"[figure4-rcmi] wrote final-clean Figure 4 to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
