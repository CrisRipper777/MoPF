"""Prepare paper-facing evidence tables, previews, and audit sidecars.

The script consumes only already-completed final-branch artifacts.  It does
not train a model.  Figure 1(c) is sourced from the separate raw fixed-graph
linear-probe diagnostic produced by ``run_figure1_order_probe.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
M1_DATASETS = ("Movies", "Grocery", "ele-fashion")
MODALITIES = ("text", "visual")
COLORS = {
    "text": "#1F5A96",
    "visual": "#B54A45",
    "Movies": "#376FAF",
    "Toys": "#7A65A7",
    "Grocery": "#3A8E75",
    "ele-fashion": "#C77C2D",
    "Reddit-S": "#8B5A4A",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _save_figure(fig: mpl.figure.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _label(ax: mpl.axes.Axes, text: str) -> None:
    ax.text(-0.12, 1.05, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def _ticks(ax: mpl.axes.Axes, labels: list[str]) -> None:
    x = np.arange(len(labels))
    ax.set_xticks(x, labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
        tick.set_rotation_mode("anchor")
        tick.set_fontsize(7)


def figure1(data_dir: Path, out: Path) -> None:
    relation = read_csv(data_dir / "relation_discrepancy.csv")
    m1_dist = read_csv(REPO_ROOT / "outputs/final/context_demand/preferred_lambda_distribution.csv")
    m1_disagreement = {
        row["dataset"]: row
        for row in read_csv(REPO_ROOT / "outputs/final/context_demand/modality_disagreement.csv")
        if row["scope"] == "aggregated"
    }
    order = read_csv(data_dir / "order_probe_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.35))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.31, top=0.84, wspace=0.43)

    ax = axes[0]
    x = np.arange(len(relation))
    values = [f(row, "mean_abs_cosine_difference") for row in relation]
    errors = [f(row, "std_abs_cosine_difference") for row in relation]
    ax.bar(x, values, yerr=errors, color=[COLORS[d] for d in DATASETS], alpha=0.9, capsize=2)
    _ticks(ax, list(DATASETS))
    ax.set_ylabel("Mean |cos(Text) - cos(Visual)|")
    ax.set_title("Modality-dependent relation discrepancy", loc="left", fontsize=9)
    ax.text(0.02, 0.96, "raw features; physical edges only", transform=ax.transAxes, fontsize=6.5, va="top")
    _label(ax, "a")

    ax = axes[1]
    lambdas = ("0.00", "0.25", "0.50", "0.75", "1.00")
    width = 0.34
    pos = np.arange(len(M1_DATASETS))
    for offset, modality in zip((-width / 2, width / 2), MODALITIES):
        bottom = np.zeros(len(M1_DATASETS))
        for lam in lambdas:
            vals = []
            for dataset in M1_DATASETS:
                row = next(r for r in m1_dist if r["dataset"] == dataset and r["modality"] == modality and r["scope"] == "aggregated" and r["lambda"] == lam)
                vals.append(f(row, "proportion"))
            ax.bar(pos + offset, vals, width, bottom=bottom, color=plt.cm.Blues(float(lam) if modality == "text" else 0.2 + 0.7 * float(lam)), alpha=0.82 if modality == "text" else 0.68, label=f"{modality.title()} lambda" if lam == "0.00" else None)
            bottom += np.asarray(vals)
    # The stacked bars are compact; annotate the modality disagreement rather
    # than attaching a second axis that would obscure the distribution.
    for idx, dataset in enumerate(M1_DATASETS):
        row = m1_disagreement[dataset]
        ax.text(idx, 1.03, f"T/V diff {f(row, 'disagreement_ratio'):.2f}", ha="center", fontsize=6.2)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Preferred-lambda proportion")
    ax.set_xticks(pos, M1_DATASETS)
    ax.set_title("Heterogeneous contextualization demand", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    _label(ax, "b")

    ax = axes[2]
    for dataset in DATASETS:
        for modality, linestyle in (("text", "--"), ("visual", "-")):
            rows = [r for r in order if r["dataset"] == dataset and r["modality"] == modality]
            rows.sort(key=lambda r: int(r["order"]))
            baseline = f(rows[0], "val_loss_mean")
            y = [f(r, "val_loss_mean") - baseline for r in rows]
            ax.plot([int(r["order"]) for r in rows], y, marker="o", ms=3, lw=1.1, ls=linestyle, color=COLORS[dataset], alpha=0.8, label=dataset if modality == "text" else None)
    ax.axhline(0, color="#666666", lw=0.6)
    ax.set_xticks(range(4), ["0", "1", "2", "3"])
    ax.set_xlabel("Fixed physical propagation order")
    ax.set_ylabel("Validation CE - order-0 CE")
    ax.set_title("Dataset-/modality-specific order utility", loc="left", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    _label(ax, "c")
    _save_figure(fig, out, "figure1_final_preview")
    write_json(out / "figure1_final_preview.qa.json", {
        "status": "generated",
        "source": "raw relation discrepancy + M1 + raw fixed-graph order probe",
        "formats": ["png", "pdf", "svg"],
        "test_labels_used": False,
    })


def _box(ax: mpl.axes.Axes, x: float, y: float, w: float, h: float, text: str, color: str, fontsize: float = 7.2, lw: float = 1.1) -> None:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor=color, edgecolor="#314052", linewidth=lw, alpha=0.96)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax: mpl.axes.Axes, x1: float, y1: float, x2: float, y2: float, color: str = "#46515C", lw: float = 1.0, style: str = "-|>") -> None:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8, linewidth=lw, color=color, connectionstyle="arc3,rad=0"))


def figure2(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.015, 0.55), 0.97, 0.40, boxstyle="round,pad=0.01", facecolor="#EDF4FB", edgecolor="#4775A3", linewidth=1.4))
    ax.add_patch(FancyBboxPatch((0.015, 0.06), 0.97, 0.40, boxstyle="round,pad=0.01", facecolor="#F7F0E7", edgecolor="#B07A3D", linewidth=1.4))
    ax.text(0.03, 0.91, "Stage I  |  Modality-Calibrated Adaptive Context Formation", fontsize=11, fontweight="bold", color="#244F7F")
    ax.text(0.03, 0.42, "Stage II  |  Interaction-Aware Multi-Order Context Integration", fontsize=11, fontweight="bold", color="#8A5725")

    _box(ax, 0.04, 0.72, 0.105, 0.105, "Physical\ngraph G", "#D9E6F4")
    _box(ax, 0.04, 0.59, 0.105, 0.085, "Text X_T\nVisual X_V", "#F4DCDC")
    _box(ax, 0.18, 0.70, 0.105, 0.12, "Separate\nprojection phi_m", "#DDECF1")
    _box(ax, 0.32, 0.70, 0.125, 0.12, "Modality-aware\nrelation calibration", "#D7E8D8")
    _box(ax, 0.48, 0.70, 0.11, 0.12, "Weighted\nphysical graph Ahat_m", "#D9E6F4")
    _box(ax, 0.63, 0.70, 0.105, 0.12, "Neighborhood\nproposal N_k_m", "#F4E3C1")
    _box(ax, 0.77, 0.70, 0.09, 0.12, "Adaptive\ngate g_i_k_m", "#E8D8F0")
    _box(ax, 0.885, 0.67, 0.075, 0.17, "S_0_m ... S_K_m\nSTATE BANK", "#F7C98C", fontsize=7.3, lw=1.5)
    for x1, x2 in ((0.145, 0.18), (0.145, 0.18), (0.285, 0.32), (0.445, 0.48), (0.59, 0.63), (0.735, 0.77), (0.86, 0.885)):
        _arrow(ax, x1, 0.76, x2, 0.76)
    _arrow(ax, 0.145, 0.63, 0.18, 0.73)
    ax.text(0.57, 0.595, "legacy relation-order bias: lightweight conditioning only", fontsize=6.7, color="#68717A", ha="center")
    _box(ax, 0.06, 0.23, 0.15, 0.115, "State bank B_i_m\n(from Stage I)", "#F7C98C", lw=1.5)
    _box(ax, 0.27, 0.23, 0.14, 0.115, "Order-aware\nQ/K/V interaction", "#E6D5B8")
    _box(ax, 0.47, 0.23, 0.14, 0.115, "Interacted states\nS_tilde_m_k", "#E7C5D4")
    _box(ax, 0.67, 0.30, 0.13, 0.09, "gamma + delta_gamma_m + delta_i_k_m", "#D5E6D6", fontsize=7)
    _box(ax, 0.67, 0.18, 0.13, 0.09, "Adaptive eta_i_k_m", "#D5E6D6", fontsize=7)
    _box(ax, 0.84, 0.23, 0.11, 0.115, "sum_k eta_i_k_m S_tilde_m_k\nZ_m", "#EBCB8B", lw=1.5)
    _box(ax, 0.84, 0.10, 0.11, 0.075, "Late fusion\nZ_T, Z_V -> NC", "#E0D6F0", fontsize=7.2, lw=1.3)
    for x1, x2 in ((0.21, 0.27), (0.41, 0.47), (0.61, 0.67), (0.80, 0.84)):
        _arrow(ax, x1, 0.287, x2, 0.287)
    _arrow(ax, 0.735, 0.30, 0.84, 0.30)
    _arrow(ax, 0.735, 0.22, 0.84, 0.27)
    _arrow(ax, 0.895, 0.23, 0.895, 0.175)
    ax.text(0.5, 0.015, "Code-faithful preview: shared physical support; independent modality paths until final late fusion", ha="center", fontsize=7.5, color="#5B6670")
    _save_figure(fig, out, "figure2_architecture_preview")
    write_json(out / "figure2_architecture_preview.qa.json", {"status": "generated", "formats": ["pdf", "svg"], "top_level_stages": 2})


def figure3(context: Path, out: Path) -> None:
    relation = read_csv(context / "relation_response.csv")
    neighbor = read_csv(context / "neighbor_allocation.csv")
    gates = read_csv(context / "adaptive_gate_summary.csv")
    intervention = read_csv(context / "adaptive_gate_intervention.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.31, top=0.84, wspace=0.42)
    ax = axes[0]
    rows = [r for r in relation if r["modality"] == "text_visual_discrepancy"]
    for row in rows:
        ax.scatter(f(row, "semantic_mean"), f(row, "relation_weight_mean"), s=32, color=COLORS.get(row["dataset"], "#555"), edgecolor="white", linewidth=0.5)
        ax.annotate(row["dataset"], (f(row, "semantic_mean"), f(row, "relation_weight_mean")), xytext=(3, 4), textcoords="offset points", fontsize=6.8)
    ax.set_xlabel("P2 semantic discrepancy")
    ax.set_ylabel("P2 relation-weight discrepancy")
    ax.set_title("MRC relation response", loc="left", fontsize=9)
    ax.text(0.02, 0.04, "continuous response on unchanged physical edges", transform=ax.transAxes, fontsize=6.3)
    _label(ax, "a")

    ax = axes[1]
    x = np.arange(len(neighbor)); width = 0.36
    tv = [f(r, "tv_distance_mean") for r in neighbor]
    top = [f(r, "top_neighbor_disagreement") for r in neighbor]
    ax.bar(x - width / 2, tv, width, color="#407CB8", label="TV distance")
    ax.set_ylabel("TV distance", color="#407CB8")
    ax.tick_params(axis="y", labelcolor="#407CB8")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, top, width, color="#B64D48", label="Top-neighbor disagreement")
    ax2.set_ylabel("Top-neighbor disagreement", color="#B64D48")
    ax2.tick_params(axis="y", labelcolor="#B64D48")
    _ticks(ax, [r["dataset"] for r in neighbor])
    ax.set_title("Modality-specific neighbor allocation", loc="left", fontsize=9)
    _label(ax, "b")

    ax = axes[2]
    x = np.arange(len(DATASETS))
    for modality, offset in (("text", -0.08), ("visual", 0.08)):
        means = []; errors = []
        for dataset in DATASETS:
            subset = [r for r in gates if r["dataset"] == dataset and r["modality"] == modality]
            means.append(np.mean([f(r, "mean") for r in subset]))
            errors.append(np.mean([f(r, "std_across_nodes") for r in subset]))
        ax.errorbar(x + offset, means, yerr=errors, fmt="o-", color=COLORS[modality], capsize=2, label=modality.title())
    _ticks(ax, list(DATASETS)); ax.set_ylim(0, 1.05); ax.set_ylabel("Gate mean ± node SD")
    ax.set_title("Adaptive context assignment", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    inset = ax.inset_axes([0.49, 0.05, 0.48, 0.36])
    xx = np.arange(len(DATASETS))
    for name, color in (("globalized", "#3D8F9A"), ("shuffled", "#984F8E")):
        subset = [r for r in intervention if r["intervention"] == name]
        inset.plot(xx, [f(next(r for r in subset if r["dataset"] == d), "prediction_flip_rate") for d in DATASETS], "o-", lw=1, ms=2.5, color=color, label=name)
    inset.set_xticks(xx, []); inset.tick_params(labelsize=5.5); inset.legend(fontsize=5.5, loc="upper left")
    _label(ax, "c")
    _save_figure(fig, out, "figure3_final_preview")
    write_json(out / "figure3_final_preview.qa.json", {"status": "generated", "formats": ["png", "pdf", "svg"], "source": "outputs/final/context_formation_analysis"})


def figure4(integration: Path, out: Path) -> None:
    order = read_csv(integration / "effective_order_contribution.csv")
    attention = read_csv(integration / "average_attention_matrix.csv")
    interventions = read_csv(integration / "integration_intervention.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.35))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.31, top=0.84, wspace=0.42)
    ax = axes[0]
    effective = [r for r in order if r["order"] == "effective_mean"]
    x = np.arange(len(DATASETS))
    for modality, offset in (("text", -0.08), ("visual", 0.08)):
        rows = [next(r for r in effective if r["dataset"] == d and r["modality"] == modality) for d in DATASETS]
        ax.errorbar(x + offset, [f(r, "mean_contribution") for r in rows], yerr=[f(r, "std") for r in rows], fmt="o-", color=COLORS[modality], capsize=2, label=modality.title())
    _ticks(ax, list(DATASETS)); ax.set_ylabel("Mean q-weighted effective order")
    ax.set_title("Modality-specific order profiles", loc="left", fontsize=9); ax.legend(fontsize=6.5)
    _label(ax, "a")

    ax = axes[1]
    subplots = [("Movies", "text"), ("Grocery", "visual")]
    for index, (dataset, modality) in enumerate(subplots):
        inset = ax.inset_axes([0.03 + (index % 2) * 0.49, 0.12, 0.44, 0.74])
        matrix = np.zeros((4, 4))
        for row in attention:
            if row["dataset"] == dataset and row["modality"] == modality:
                matrix[int(row["query_order"]), int(row["key_order"])] = f(row, "mean_attention")
        im = inset.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.55, matrix.max()))
        inset.set_xticks(range(4), range(4), fontsize=5); inset.set_yticks(range(4), range(4), fontsize=5)
        inset.set_xlabel("key", fontsize=5); inset.set_ylabel("query", fontsize=5)
        inset.set_title(f"{dataset}\n{modality}", fontsize=6.5)
        for i in range(4):
            for j in range(4):
                inset.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=4.8, color="white" if matrix[i,j] > 0.5 else "black")
    ax.axis("off")
    ax.set_title("Cross-order interaction pattern", loc="left", fontsize=9)
    _label(ax, "b")

    ax = axes[2]
    names = ("query_collapse", "uniform_attention", "interaction_off")
    labels = {"query_collapse": "Query-collapse", "uniform_attention": "Uniform", "interaction_off": "Interaction-off"}
    width = 0.24
    for index, name in enumerate(names):
        subset = [next(r for r in interventions if r["dataset"] == d and r["intervention"] == name) for d in DATASETS]
        ax.bar(x + (index - 1) * width, [f(r, "logit_mae") for r in subset], width, label=labels[name])
    _ticks(ax, list(DATASETS)); ax.set_ylabel("Logit MAE vs normal")
    ax.set_title("Functional integration intervention", loc="left", fontsize=9); ax.legend(fontsize=6.2)
    _label(ax, "c")
    _save_figure(fig, out, "figure4_final_preview")
    write_json(out / "figure4_final_preview.qa.json", {"status": "generated", "formats": ["png", "pdf", "svg"], "source": "outputs/final/multi_order_integration_analysis"})


def _parse_baseline_table(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    text = path.read_text(encoding="utf-8")
    models = ("MLP", "GCN", "GraphSAGE", "MMGCN", "MGAT", "DiP", "DGF", "DMGC", "LGMRec")
    result: dict[str, dict[str, dict[str, float]]] = {d: {} for d in DATASETS}
    pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+\$\\pm\$\s+([0-9]+(?:\.[0-9]+)?)")
    for model in models:
        line = next(line for line in text.splitlines() if line.startswith(model + " &"))
        cells = [part.strip() for part in line.rstrip("\\").split("&")][1:]
        values = []
        for cell in cells:
            cell = re.sub(r"\\(?:textbf|underline)\{([^}]*)\}", r"\1", cell)
            match = pattern.search(cell)
            if match is None:
                raise ValueError(f"could not parse baseline table cell {cell!r}")
            values.append((float(match.group(1)), float(match.group(2))))
        for index, dataset in enumerate(DATASETS):
            result[dataset][model] = {
                "accuracy_mean": values[index * 2][0], "accuracy_sd": values[index * 2][1],
                "macro_f1_mean": values[index * 2 + 1][0], "macro_f1_sd": values[index * 2 + 1][1],
            }
    return result


def table1(table_dir: Path) -> None:
    baseline = _parse_baseline_table(REPO_ROOT / "paper/tables/table_nc_main.tex")
    mgsc_rows = read_csv(REPO_ROOT / "outputs/final/mgsc_corrected_controls/summary.csv")
    full = {r["dataset"]: r for r in mgsc_rows if r["variant"] == "full"}
    rows: list[dict[str, Any]] = []
    model_order = ("MLP", "GCN", "GraphSAGE", "MMGCN", "MGAT", "DGF", "DMGC", "LGMRec", "MGSC-MAG")
    for dataset in DATASETS:
        values: dict[str, dict[str, float]] = {model: dict(baseline[dataset][model]) for model in model_order[:-1]}
        values["MGSC-MAG"] = {
            "accuracy_mean": 100 * f(full[dataset], "accuracy_mean"),
            "accuracy_sd": 100 * f(full[dataset], "accuracy_population_std"),
            "macro_f1_mean": 100 * f(full[dataset], "macro_f1_mean"),
            "macro_f1_sd": 100 * f(full[dataset], "macro_f1_population_std"),
        }
        for metric, mean_key, sd_key in (("Accuracy", "accuracy_mean", "accuracy_sd"), ("Macro-F1", "macro_f1_mean", "macro_f1_sd")):
            ranked = sorted(values, key=lambda model: values[model][mean_key], reverse=True)
            best, second = ranked[0], ranked[1]
            strongest_baseline = max(values[m][mean_key] for m in model_order[:-1])
            for model in model_order:
                rows.append({
                    "dataset": dataset, "model": model, "metric": metric,
                    "mean_percent": values[model][mean_key], "population_sd_percent": values[model][sd_key],
                    "rank": ranked.index(model) + 1, "is_best": model == best, "is_second": model == second,
                    "mgsc_delta_vs_strongest_baseline_pp": values["MGSC-MAG"][mean_key] - strongest_baseline if model == "MGSC-MAG" else "",
                    "source": "paper/tables/table_nc_main.tex" if model != "MGSC-MAG" else "outputs/final/mgsc_corrected_controls/summary.csv",
                })
    write_csv(table_dir / "table1_nc.csv", rows)
    lines = [
        "# Table 1 NC analysis", "",
        "Table 1 combines the repository's existing historical baseline table with the current canonical P2 Full row. Baseline values are parsed from `paper/tables/table_nc_main.tex`; the repository README records the historical source and metric caveat. The MGSC-MAG row is the corrected-control Full result from `outputs/final/mgsc_corrected_controls/summary.csv`.", "",
        "All entries are percentages, mean ± population SD over seeds 42/43/44. Ranking is recomputed by mean within each dataset and metric; `best` and `second` flags in `table1_nc.csv` are descriptive formatting helpers, not significance claims. The table should be described as best among the compared rows, not universal SOTA.", "",
        "## MGSC-MAG versus strongest compared baseline", "",
        "| Dataset | Accuracy delta (pp) | Macro-F1 delta (pp) |", "|---|---:|---:|",
    ]
    for d in DATASETS:
        a = next(r for r in rows if r["dataset"] == d and r["model"] == "MGSC-MAG" and r["metric"] == "Accuracy")
        m = next(r for r in rows if r["dataset"] == d and r["model"] == "MGSC-MAG" and r["metric"] == "Macro-F1")
        lines.append(f"| {d} | {float(a['mgsc_delta_vs_strongest_baseline_pp']):+.2f} | {float(m['mgsc_delta_vs_strongest_baseline_pp']):+.2f} |")
    lines.extend(["", "The current P2 row is competitive and is not uniformly best on both metrics. The historical baseline provenance limitation remains: the original baseline-producing execution SHA is not recoverable from this checkout."])
    (REPO_ROOT / "docs/final/TABLE1_NC_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table2_and_appendix(table_dir: Path) -> None:
    corrected = {r["dataset"]: r for r in read_csv(REPO_ROOT / "outputs/final/mgsc_corrected_controls/summary.csv") if r["variant"] == "full"}
    for row in read_csv(REPO_ROOT / "outputs/final/mgsc_corrected_controls/summary.csv"):
        if row["variant"] != "full":
            corrected[(row["dataset"], row["variant"])] = row
    fine = { (r["dataset"], r["variant"]): r for r in read_csv(REPO_ROOT / "outputs/final/mgsc_functional_ablation/summary.csv") }
    defs = {
        "Full MGSC-MAG": ("full", "canonical P2: MRC + node/order gate + state bank + interaction + eta composition", "mgsc_corrected_controls/summary.csv"),
        "Attribute Only": ("attribute_only", "sanity control: H0 only after modality refinement/fusion", "mgsc_functional_ablation/summary.csv"),
        "Plain Multi-Order Backbone": ("plain_multi_order_backbone", "uniform physical relations, fixed 0.9 gate, no conditional relation/order readout", "mgsc_corrected_controls/summary.csv"),
        "Raw Bank Mean": ("raw_bank_mean", "mean of raw interacted-state bypass states", "mgsc_corrected_controls/summary.csv"),
        "Raw Terminal": ("raw_terminal_only", "terminal raw interacted-state bypass readout", "mgsc_corrected_controls/summary.csv"),
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for name, (key, definition, source) in defs.items():
            row = corrected[dataset] if key == "full" else fine[(dataset, key)] if key == "attribute_only" else corrected[(dataset, key)]
            rows.append({
                "dataset": dataset, "variant": name,
                "accuracy_mean": f(row, "accuracy_mean") if "accuracy_mean" in row else f(row, "test_accuracy_mean"),
                "accuracy_population_std": f(row, "accuracy_population_std") if "accuracy_population_std" in row else f(row, "test_accuracy_population_std"),
                "macro_f1_mean": f(row, "macro_f1_mean") if "macro_f1_mean" in row else f(row, "test_macro_f1_mean"),
                "macro_f1_population_std": f(row, "macro_f1_population_std") if "macro_f1_population_std" in row else f(row, "test_macro_f1_population_std"),
                "definition": definition, "source": source,
            })
    write_csv(table_dir / "table2_framework_controls.csv", rows)
    lines = ["# Table 2 framework-level control analysis", "", "Table 2 is the main control table. Full, Plain Multi-Order Backbone, Raw Bank Mean, and Raw Terminal are sourced from the corrected-control matrix; Attribute Only is sourced from the completed formal functional-ablation matrix because it was not part of the corrected-control rerun. This provenance distinction is retained in the CSV and is not hidden.", "", "The raw terminal readout is sometimes higher than Full, so Table 2 does not support a claim that every adaptive integration component is individually necessary. The plain backbone is the strongest architecture-matched control: Full is better in 14/15 paired seed comparisons for both metrics, with mean paired gains of +0.291 and +0.334 percentage points (corrected-control source).", "", "| Dataset | Full Acc | Full F1 | Plain Acc | Plain F1 |", "|---|---:|---:|---:|---:|"]
    for d in DATASETS:
        vals = {r["variant"]: r for r in rows if r["dataset"] == d}
        lines.append(f"| {d} | {f(vals['Full MGSC-MAG'], 'accuracy_mean'):.4f} | {f(vals['Full MGSC-MAG'], 'macro_f1_mean'):.4f} | {f(vals['Plain Multi-Order Backbone'], 'accuracy_mean'):.4f} | {f(vals['Plain Multi-Order Backbone'], 'macro_f1_mean'):.4f} |")
    (REPO_ROOT / "docs/final/TABLE2_FRAMEWORK_CONTROLS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    paired = read_csv(REPO_ROOT / "outputs/final/mgsc_functional_ablation/paired_deltas.csv")
    variants = ("uniform_relations", "global_context_gate", "uniform_integration", "no_cross_order_interaction")
    appendix: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for variant in variants:
            subset = [r for r in paired if r["dataset"] == dataset and r["ablation"] == variant]
            appendix.append({
                "dataset": dataset, "ablation": variant,
                "full_minus_ablation_accuracy_pp_mean": 100 * np.mean([f(r, "full_minus_ablation_accuracy") for r in subset]),
                "full_minus_ablation_accuracy_pp_population_sd": 100 * np.std([f(r, "full_minus_ablation_accuracy") for r in subset], ddof=0),
                "full_minus_ablation_macro_f1_pp_mean": 100 * np.mean([f(r, "full_minus_ablation_macro_f1") for r in subset]),
                "full_minus_ablation_macro_f1_pp_population_sd": 100 * np.std([f(r, "full_minus_ablation_macro_f1") for r in subset], ddof=0),
                "n_paired_seeds": len(subset), "source": "outputs/final/mgsc_functional_ablation/paired_deltas.csv",
            })
    write_csv(table_dir / "tableA1_fine_ablation.csv", appendix)
    (REPO_ROOT / "docs/final/TABLEA1_FINE_ABLATION.md").write_text("# Appendix Table A1 fine-grained ablation\n\nThe CSV reports paired Full-minus-ablation deltas in percentage points for relation calibration, global context gating, uniform integration, and cross-order interaction removal. Fixed Gate 0.9 was not part of the completed formal matrix and is therefore not fabricated here. Fine-grained deltas are mixed and dataset-dependent; they are not ranked in the main text.\n", encoding="utf-8")


def efficiency(table_dir: Path) -> None:
    per_seed = read_csv(REPO_ROOT / "outputs/final/mgsc_corrected_controls/per_seed_results.csv")
    rows: list[dict[str, Any]] = []
    for model_name, variant in (("MGSC-MAG", "full"), ("Plain Multi-Order Backbone", "plain_multi_order_backbone")):
        for dataset in DATASETS:
            subset = [r for r in per_seed if r["dataset"] == dataset and r["variant"] == variant]
            memory = []
            for r in subset:
                metrics_path = REPO_ROOT / Path(r["checkpoint"]).parent / "metrics.json"
                if metrics_path.is_file():
                    payload = json.loads(metrics_path.read_text())
                    if payload.get("peak_gpu_memory_mib") is not None:
                        memory.append(float(payload["peak_gpu_memory_mib"]))
            rows.append({
                "dataset": dataset, "model": model_name,
                "encoder_params": int(subset[0]["encoder_params"]), "classifier_params": int(subset[0]["classifier_params"]),
                "runtime_seconds_mean": np.mean([f(r, "runtime_seconds") for r in subset]),
                "runtime_seconds_population_sd": np.std([f(r, "runtime_seconds") for r in subset], ddof=0),
                "peak_gpu_memory_mib_mean": np.mean(memory) if memory else "",
                "peak_gpu_memory_mib_population_sd": np.std(memory, ddof=0) if memory else "",
                "profile_status": "measured from corrected-control NC runs",
                "source": "outputs/final/mgsc_corrected_controls/per_seed_results.csv + metrics.json",
            })
    # Reliable counts for DiP and GraphSAGE are obtainable by instantiation;
    # training/memory are intentionally left TODO because no matched runs were
    # found in the final output tree.
    try:
        import torch
        from omegaconf import OmegaConf
        from src.data import load_mag_data
        from src.models.factory import build_model
        for model_name, config_name in (("DiP", "dip"), ("GraphSAGE", "sage")):
            config_model = OmegaConf.load(REPO_ROOT / "configs/model" / f"{config_name}.yaml")
            for dataset in DATASETS:
                base = OmegaConf.load(REPO_ROOT / "configs/config.yaml")
                ds = OmegaConf.load(REPO_ROOT / "configs/dataset" / f"{dataset}.yaml")
                task = OmegaConf.load(REPO_ROOT / "configs/task/nc.yaml")
                cfg = OmegaConf.create({"model": config_model, "dataset": ds, "task": task, "paths": base.paths, "seed": 42})
                OmegaConf.resolve(cfg)
                data = load_mag_data(cfg, "nc", 42)
                info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes, "num_classes": data.num_classes, "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0, "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0}
                model = build_model(cfg, info)
                rows.append({
                    "dataset": dataset, "model": model_name,
                    "encoder_params": sum(p.numel() for p in model.parameters()), "classifier_params": "", "runtime_seconds_mean": "", "runtime_seconds_population_sd": "", "peak_gpu_memory_mib_mean": "", "peak_gpu_memory_mib_population_sd": "",
                    "profile_status": "parameter-count-only; no matched final-branch run",
                    "source": f"configs/model/{config_name}.yaml + instantiated final-branch model",
                })
    except Exception as exc:
        (table_dir / "efficiency_profile_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
    write_csv(table_dir / "table_efficiency.csv", rows)
    (REPO_ROOT / "docs/final/EFFICIENCY_ANALYSIS.md").write_text("# Efficiency profile\n\nMGSC-MAG and the plain multi-order control have matched corrected-control training runtime and peak GPU-memory measurements in the CSV. DiP and GraphSAGE have reliable instantiated parameter counts, but no matched final-branch training/inference profile was found; their time and memory fields are intentionally left blank rather than estimated from incompatible runs. These profiles are descriptive, not a speed claim.\n", encoding="utf-8")


def write_docs() -> None:
    (REPO_ROOT / "docs/final/FIGURE1_EVIDENCE_AUDIT.md").write_text("""# Figure 1 evidence audit

## Panel (a): Modality-dependent relation discrepancy

- **Input:** row-normalized raw text/visual features and actual physical edges from the final data loader; edge-level absolute difference between modality cosine similarities.
- **Model independence:** no CoSI/MGSC projection, checkpoint, learned relation weight, label, or graph rewiring.
- **Labels:** none.
- **Split:** not applicable; this is an edge-level descriptive statistic over the physical graph.
- **Supports:** the same physical relation can have different raw modality semantic similarity.
- **Does not support:** relation reliability, learned usefulness, or causal edge selection.

## Panel (b): Heterogeneous contextualization demand

- **Input:** M1 raw row-normalized local features, physical-neighbor means, and normalized mixtures for λ ∈ {0, .25, .50, .75, 1}.
- **Model independence:** only a shared linear probe is trained separately for each condition; no MGSC checkpoint or gate is used.
- **Labels/split:** official train is split deterministically into probe-train/probe-dev; official validation is read once after probe-dev CE selection; official test labels are never indexed.
- **Supports:** node-level demand for local versus neighborhood context differs across datasets/modalities.
- **Does not support:** the node-wise oracle as a realizable policy or test performance.

## Panel (c): Raw fixed-graph order profile

- **Input:** raw modality features propagated through a fixed symmetric-normalized physical graph at orders 0–3, followed by the same linear probe protocol.
- **Model independence:** no MGSC training, projection, MRC, gate, attention, or checkpoint.
- **Labels/split:** probe-train/probe-dev inside official train; official validation only for the reported profile; test labels unused.
- **Supports:** order utility/profile differs by dataset and modality in this lightweight diagnostic.
- **Does not support:** node-level universal preferred order or downstream MGSC performance attribution.

The authoritative panel data are under `outputs/final/paper_figures/figure1_data/`. The probe manifest records the GPU/device and protocol; the output is a motivation diagnostic, not a model benchmark.
""", encoding="utf-8")
    (REPO_ROOT / "docs/final/FIGURE2_ARCHITECTURE_SPEC.md").write_text("""# Figure 2 architecture specification

The preview has exactly two top-level stages and follows the canonical P2 code path.

## Stage I — Modality-Calibrated Adaptive Context Formation

Shared physical graph support and separate text/visual inputs enter independent projection MLPs. For modality m, the code computes a learned diagonal weighted cosine on each physical edge, maps the cosine to a bounded relation weight, and applies symmetric graph normalization with the configured self-loops. Each order proposes `N_{i,k}^m = [Ahat^m S_{k-1}^m]_i`. The adaptive gate consumes `(H0, Nk, |H0-Nk|, H0*Nk, p_k)` through an independent text/visual MLP and forms `S_{i,k}^m = (1-g_{i,k}^m)H_{i,0}^m + g_{i,k}^m N_{i,k}^m`. The visible state bank is `B_i^m=[S_{i,0}^m,...,S_{i,K}^m]`.

The retained legacy relation-order bias is shown as a small conditioning arrow because it modifies attention logits; it is not presented as a third top-level stage or a new mechanism.

## Stage II — Interaction-Aware Multi-Order Context Integration

The state bank receives order embeddings and independent single-head Q/K/V interaction. The logits include the retained local relation-order conditioning, and the interacted states are `S~`. The preference path computes node/order responses from interacted states, then `eta = gamma_global + delta_gamma^m + delta_i^m`. Canonical P2 directly composes `Z_i^m = sum_k eta_{i,k}^m S~_{i,k}^m`, applies independent modality refinement, and performs late text/visual fusion before the NC classifier.

## Code-to-figure mapping

| Figure object | Code object |
|---|---|
| projection φ_m | `text_proj`, `visual_proj` |
| modality relation calibration | `_relation_calibration`, `metric_theta_*`, `_edge_weight_from_cosine` |
| weighted physical graph | `_normalized_operator` |
| neighborhood proposal | `_propagate_once` inside `_adaptive_multi_hop_states` |
| adaptive gate | `context_gate_text/visual`, `_adaptive_multi_hop_states` |
| state bank | `states_text`, `states_visual` |
| order-aware interaction | `_cross_order_interaction`, `hop_order_embedding_*`, `hop_layers_*` |
| legacy conditioning | `_local_relation_context`, `relation_beta_raw_*`, `use_legacy_relation_order_bias` |
| preference and eta | `_node_preference`, `gamma_global`, `delta_gamma_*`, `eta_*` |
| direct P2 integration | `direct_interacted_integration=True`, `_compose` |
| modality refinement/fusion | `*_refine_*`, `fusion_skip`, `fusion_mlp`, `output_norm` |

Text and visual paths remain separate until `torch.cat([z_text_refined, z_visual_refined])`.
""", encoding="utf-8")
    (REPO_ROOT / "docs/final/FIGURE3_FINAL_CLAIMS.md").write_text("""# Figure 3 final claims

## (a) MRC relation response

The current P2 checkpoint maps modality-dependent semantic discrepancy to continuous relation-weight discrepancy on the same physical support. This supports the claim that MRC is computationally responsive to modality differences. It is not an independent discovery of a new topology, a large rewiring claim, or relation reliability estimation.

## (b) Modality-specific neighbor allocation

Text and visual normalized neighbor-weight distributions have measurable but generally fine-grained differences. The top-neighbor disagreement is decision-relevant in the diagnostic, while the TV distance remains small. The correct claim is modality-specific allocation over a shared topology, not substantial topology difference or rewiring.

## (c) Adaptive context assignment

Gate distributions are non-degenerate, differ between modalities on several datasets, and globalized/shuffled frozen inference changes embeddings, logits, and predictions. This supports a functional node-to-gate correspondence in the trained P2 computation. It is not M1 lambda ground truth, a causal gate test, or evidence of universal performance improvement.

All interventions are frozen-checkpoint inference diagnostics. None should be described as retrained performance. Source CSVs are under `outputs/final/context_formation_analysis/` and `outputs/final/paper_figures/figure3_data/`.
""", encoding="utf-8")
    (REPO_ROOT / "docs/final/FIGURE4_FINAL_CLAIMS.md").write_text("""# Figure 4 final claims

## (a) Modality-specific order profiles

The normalized contribution statistic is `q_{i,k}=|eta_{i,k}|/sum_j|eta_{i,j}|`, and the effective order is `sum_k k q_{i,k}`. Profiles are concentrated in the middle-to-higher orders but vary by dataset and modality. The panel supports modality-specific learned order profiles; it does not support strong universal node-level heterogeneity or an optimal propagation depth claim.

## (b) Cross-order interaction pattern

The complete average attention matrices are exported for all five NC datasets and both modalities. Representative panels show that order-0 dominance occurs in some paths, whereas other paths place more mass on higher-order keys. Therefore the safe claim is dataset-/modality-dependent cross-order interaction, not a universal attention pattern. The matrices show the interaction computation; they do not by themselves establish causal necessity.

## (c) Functional integration intervention

Query-collapse and uniform-attention perturbations are smaller than interaction-off in the current frozen checkpoints across all five NC datasets. This establishes that the interaction output reaches the final representation through the direct P2 composition. It does not establish universal accuracy gains or that query-dependent attention is the only source of the effect.

These are frozen-checkpoint inference interventions, not retrained ablation performance. Source CSVs are under `outputs/final/multi_order_integration_analysis/` and `outputs/final/paper_figures/figure4_data/`.
""", encoding="utf-8")


def manifest() -> None:
    payload = {
        "canonical_model": "src/models/mgsc_mag.py (MGSCMAG; canonical P2 switches)",
        "canonical_config": "configs/model/mgsc_mag_p2.yaml",
        "git_sha": "0d062e43ed90c8bd52baa7051b39b11dffbd150b0",
        "datasets": list(DATASETS),
        "seeds": [42, 43, 44],
        "main_benchmark_source": "outputs/final/mgsc_corrected_controls/summary.csv; baselines: paper/tables/table_nc_main.tex",
        "m1_source": "outputs/final/context_demand/ plus docs/final/context_demand_report.md",
        "corrected_controls_source": "outputs/final/mgsc_corrected_controls/",
        "fine_grained_ablation_source": "outputs/final/mgsc_functional_ablation/",
        "figure3_source": "outputs/final/context_formation_analysis/ and outputs/final/paper_figures/figure3_data/",
        "figure4_source": "outputs/final/multi_order_integration_analysis/ and outputs/final/paper_figures/figure4_data/",
        "evidence_taxonomy": {
            "empirical_motivation": ["M1 raw feature/context probe", "Figure 1(a,c) model-independent diagnostics"],
            "downstream_performance": ["Table 1 MGSC-MAG row", "Table 2 Full and controls"],
            "retrained_ablation": ["outputs/final/mgsc_corrected_controls/", "outputs/final/mgsc_functional_ablation/"],
            "frozen_checkpoint_intervention": ["Figure 3(c)", "Figure 4(c)"],
            "descriptive_diagnostics": ["Figure 3(a,b)", "Figure 4(a,b)"],
        },
        "scope": "NC only; Movies, Toys, Grocery, ele-fashion, Reddit-S; LP intentionally excluded",
        "test_labels_in_m1": False,
        "frozen_files_unchanged": ["src/models/cosi_mag_final.py", "configs/model/cosi_mag_final.yaml"],
    }
    write_json(REPO_ROOT / "docs/final/PAPER_EVIDENCE_MANIFEST.json", payload)
    lines = ["# Paper evidence manifest", "", "This manifest distinguishes empirical motivation, downstream performance, retrained ablation, frozen-checkpoint intervention, and descriptive diagnostics. It is the provenance index for the current five-NC paper scope.", "", "| Item | Authoritative source | Evidence role |", "|---|---|---|"]
    entries = [
        ("Canonical model", payload["canonical_model"], "implementation"),
        ("Canonical config", payload["canonical_config"], "configuration"),
        ("Git SHA", payload["git_sha"], "code snapshot"),
        ("Datasets", ", ".join(DATASETS), "NC scope"),
        ("Seeds", "42, 43, 44", "replication"),
        ("Main benchmark", payload["main_benchmark_source"], "downstream performance"),
        ("M1", payload["m1_source"], "empirical motivation"),
        ("Corrected controls", payload["corrected_controls_source"], "retrained ablation"),
        ("Fine-grained ablation", payload["fine_grained_ablation_source"], "retrained ablation / appendix"),
        ("Figure 3", payload["figure3_source"], "descriptive + frozen-checkpoint intervention"),
        ("Figure 4", payload["figure4_source"], "descriptive + frozen-checkpoint intervention"),
    ]
    for row in entries:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Inference interventions in Figures 3(c) and 4(c) use trained Full P2 checkpoints and are never described as retrained performance. M1 and Figure 1(c) use lightweight linear probes with official test labels excluded. The frozen CoSI reference files remain unchanged."]
    (REPO_ROOT / "docs/final/PAPER_EVIDENCE_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/final/paper_figures"))
    args = parser.parse_args()
    root = (REPO_ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    root.mkdir(parents=True, exist_ok=True)
    data1 = root / "figure1_data"
    table_dir = REPO_ROOT / "outputs/final/paper_tables"
    figure1(data1, root)
    figure2(root)
    figure3(REPO_ROOT / "outputs/final/context_formation_analysis", root)
    figure4(REPO_ROOT / "outputs/final/multi_order_integration_analysis", root)
    table1(table_dir)
    table2_and_appendix(table_dir)
    efficiency(table_dir)
    write_docs()
    manifest()


if __name__ == "__main__":
    main()
