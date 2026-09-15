#!/usr/bin/env python3
"""Generate paper-facing NC and LP LaTeX tables from repository documents.

The numeric cells are parsed from the cited Markdown benchmark tables at run
time. No benchmark values are hard-coded in this generator. Internal MAP-MAG
predecessors are excluded from the paper-facing model set.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
TABLE_ROOT = REPO_ROOT / "paper" / "tables"

NC_SOURCE = DOCS_ROOT / "nc_benchmark_results.md"
LP_SOURCE = DOCS_ROOT / "lp_benchmark_results.md"
NC_FIX_SOURCE = DOCS_ROOT / "nc_macro_f1_metric_fix.md"
PROTOCOL_SOURCE = DOCS_ROOT / "unified_training_evaluation_protocol.md"
FINAL_PROTOCOL_SOURCE = DOCS_ROOT / "mopf_final_evaluation_protocol.md"
CLOTH_SOURCE = REPO_ROOT / "outputs/f1_final_execution/tables/f1_lp_cloth_quasiheldout_paper_table.csv"

NC_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
LP_DATASETS = ["Sports-Copurchase", "Cloth-Copurchase"]
MODELS = ["mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec", "mopf"]
BASELINE_MODELS = MODELS[:-1]
DISPLAY_NAMES = {
    "mlp": "MLP",
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "mmgcn": "MMGCN",
    "mgat": "MGAT",
    "dip": "DiP",
    "dgf": "DGF",
    "dmgc": "DMGC",
    "lgmrec": "LGMRec",
    "mopf": "MoPF",
}

NC_METRICS = ["Test Accuracy", "Test Macro-F1"]
LP_METRICS = ["Test Hits@1", "Test Hits@3", "Test Hits@10", "Test MRR"]


@dataclass(frozen=True)
class Stat:
    mean: float
    spread: float


def _clean_cell(cell: str) -> str:
    cleaned = cell.strip()
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = cleaned.replace("`", "")
    return cleaned.strip()


def _parse_row(line: str) -> list[str]:
    return [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    cells = _parse_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    lines = text.splitlines()
    tables: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(lines) - 1:
        if lines[index].lstrip().startswith("|") and _is_separator(lines[index + 1]):
            header = _parse_row(lines[index])
            rows: list[list[str]] = []
            index += 2
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                row = _parse_row(lines[index])
                if row:
                    rows.append(row)
                index += 1
            tables.append((header, rows))
        else:
            index += 1
    return tables


def _section_after_heading(text: str, heading_prefix: str, level: int) -> str:
    heading_pattern = re.compile(rf"(?m)^{'#' * level} {re.escape(heading_prefix)}.*$")
    match = heading_pattern.search(text)
    if match is None:
        raise ValueError(f"Could not find heading {heading_prefix!r}")
    next_heading = re.search(rf"(?m)^{'#' * level} ", text[match.end() :])
    end = match.end() + next_heading.start() if next_heading else len(text)
    return text[match.end() : end]


def _find_table(section: str, required_headers: set[str]) -> tuple[list[str], list[list[str]]]:
    for header, rows in _markdown_tables(section):
        if required_headers.issubset(header):
            return header, rows
    raise ValueError(f"Could not find a table containing headers {sorted(required_headers)}")


def _parse_stat(cell: str, *, source: Path) -> Stat:
    cleaned = _clean_cell(cell)
    match = re.fullmatch(
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*±\s*"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+))%?",
        cleaned,
    )
    if match is None:
        raise ValueError(f"Expected 'mean ± spread' in {source}: {cell!r}")
    return Stat(float(match.group(1)), float(match.group(2)))


def _table_to_stats(
    table: tuple[list[str], list[list[str]]],
    metrics: list[str],
    *,
    source: Path,
) -> dict[str, dict[str, Stat]]:
    header, rows = table
    positions = {name: header.index(name) for name in metrics}
    model_position = header.index("Model")
    parsed: dict[str, dict[str, Stat]] = {}
    for row in rows:
        if len(row) <= max([model_position, *positions.values()]):
            raise ValueError(f"Short row in {source}: {row!r}")
        model = _clean_cell(row[model_position]).lower()
        parsed[model] = {metric: _parse_stat(row[position], source=source) for metric, position in positions.items()}
    return parsed


def _load_nc() -> dict[str, dict[str, dict[str, Stat]]]:
    text = NC_SOURCE.read_text(encoding="utf-8")
    output: dict[str, dict[str, dict[str, Stat]]] = {}
    required = {"Model", *NC_METRICS}
    for dataset in NC_DATASETS:
        section = _section_after_heading(text, dataset, level=2)
        table = _find_table(section, required)
        parsed = _table_to_stats(table, NC_METRICS, source=NC_SOURCE)
        missing = sorted(set(MODELS).difference(parsed))
        if missing:
            raise ValueError(f"NC {dataset} is missing required models: {missing}")
        output[dataset] = {model: parsed[model] for model in MODELS}
    return output


def _load_lp_sports() -> dict[str, dict[str, Stat]]:
    text = LP_SOURCE.read_text(encoding="utf-8")
    required = {"Model", *LP_METRICS}

    formal_section = _section_after_heading(text, "Formal sports-copurchase LP", level=2)
    formal_table = _find_table(formal_section, {"Model", *LP_METRICS})
    formal_mopf = _table_to_stats(formal_table, LP_METRICS, source=LP_SOURCE)
    if set(formal_mopf) != {"mopf"}:
        raise ValueError(f"Expected only current formal MoPF in formal sports section, found {sorted(formal_mopf)}")

    historical_start = re.search(r"(?m)^## Historical sports-copurchase benchmark", text)
    if historical_start is None:
        raise ValueError("Could not find the historical sports-copurchase table")
    historical_section = text[historical_start.start() :]
    historical_table = _find_table(historical_section, required)
    historical = _table_to_stats(historical_table, LP_METRICS, source=LP_SOURCE)

    combined: dict[str, dict[str, Stat]] = {}
    for model in BASELINE_MODELS:
        if model not in historical:
            raise ValueError(f"Sports LP historical table is missing required baseline {model}")
        combined[model] = historical[model]
    combined["mopf"] = formal_mopf["mopf"]
    return {model: combined[model] for model in MODELS}


def _load_lp_cloth() -> dict[str, dict[str, Stat]]:
    """Load the real three-seed quasi-held-out Cloth LP summary."""

    if not CLOTH_SOURCE.is_file():
        raise FileNotFoundError(f"Required Cloth LP summary does not exist: {CLOTH_SOURCE}")
    metric_columns = {
        "Test Hits@1": ("test_hits@1_mean", "test_hits@1_population_std"),
        "Test Hits@3": ("test_hits@3_mean", "test_hits@3_population_std"),
        "Test Hits@10": ("test_hits@10_mean", "test_hits@10_population_std"),
        "Test MRR": ("test_mrr_mean", "test_mrr_population_std"),
    }
    required_fields = {
        "dataset",
        "model",
        "seeds",
        "seed_count",
        *(column for pair in metric_columns.values() for column in pair),
    }
    with CLOTH_SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Cloth LP summary has no header: {CLOTH_SOURCE}")
        missing = sorted(required_fields.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"Cloth LP summary is missing fields: {missing}")
        rows = list(reader)

    parsed: dict[str, dict[str, Stat]] = {}
    for row in rows:
        if row["dataset"].strip().lower() != "cloth-copurchase":
            raise ValueError(f"Unexpected dataset in Cloth LP summary: {row['dataset']!r}")
        model = row["model"].strip().lower()
        if model not in MODELS:
            raise ValueError(f"Unexpected model in Cloth LP summary: {model!r}")
        if model in parsed:
            raise ValueError(f"Duplicate Cloth LP model row: {model}")
        if row["seeds"].strip() != "42;43;44" or row["seed_count"].strip() != "3":
            raise ValueError(
                "Cloth LP summary is not the expected three-seed record for seeds 42, 43, and 44"
            )
        parsed[model] = {
            metric: Stat(float(row[mean_column]), float(row[std_column]))
            for metric, (mean_column, std_column) in metric_columns.items()
        }

    missing_models = sorted(set(MODELS).difference(parsed))
    if missing_models:
        raise ValueError(f"Cloth LP summary is missing models: {missing_models}")
    return {model: parsed[model] for model in MODELS}


def _rankings(
    values: dict[str, dict[str, Stat]],
    metrics: list[str],
) -> dict[str, tuple[str, str]]:
    rankings: dict[str, tuple[str, str]] = {}
    for metric in metrics:
        ordered = sorted(MODELS, key=lambda model: (-values[model][metric].mean, MODELS.index(model)))
        rankings[metric] = (ordered[0], ordered[1])
    return rankings


def _format_stat(stat: Stat, decoration: str | None = None) -> str:
    body = f"{stat.mean:.2f} $\\pm$ {stat.spread:.2f}"
    if decoration == "best":
        return rf"\textbf{{{body}}}"
    if decoration == "second":
        return rf"\underline{{{body}}}"
    return body


def _decorated_stat(values: dict[str, dict[str, Stat]], model: str, metric: str, rankings: dict[str, tuple[str, str]]) -> str:
    best, second = rankings[metric]
    decoration = "best" if model == best else "second" if model == second else None
    return _format_stat(values[model][metric], decoration)


def _nc_tex(values: dict[str, dict[str, dict[str, Stat]]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Node classification results on the five benchmark datasets. Test Accuracy and Test Macro-F1 are reported as percentages; values are mean $\pm$ population standard deviation over seeds 42, 43, and 44. Checkpoint selection used validation accuracy, and no validation metric is included in this table. The ele-fashion Macro-F1 entries in this draft use the same legacy-evaluator batch for like-with-like comparability; a unified fixed-metric rerun is required before paper submission.}",
        r"\label{tab:nc_main}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        "Model & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{dataset}}}" for dataset in NC_DATASETS) + r" \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
        " & " + " & ".join(["Acc.", "Macro-F1"] * len(NC_DATASETS)) + r" \\",
        r"\midrule",
    ]
    rankings = {dataset: _rankings(values[dataset], NC_METRICS) for dataset in NC_DATASETS}
    for model in MODELS:
        if model == "mopf":
            lines.append(r"\midrule")
        cells = [DISPLAY_NAMES[model]]
        for dataset in NC_DATASETS:
            cells.extend(
                _decorated_stat(values[dataset], model, metric, rankings[dataset])
                for metric in NC_METRICS
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def _lp_tex(
    sports: dict[str, dict[str, Stat]],
    cloth: dict[str, dict[str, Stat]],
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Link prediction results on Sports-Copurchase and Cloth-Copurchase. All entries are test metrics reported as percentages. Sports-Copurchase is the formal $\texttt{unified\_sampled\_lp\_v1}$ benchmark with seeds 42, 43, and 44; values are mean $\pm$ population standard deviation and checkpoint selection uses validation MRR. Cloth-Copurchase values are the real three-seed quasi-held-out extension and are included for completeness, but are not a second formal LP benchmark. Validation MRR is omitted.}",
        r"\label{tab:lp_main}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Model & \multicolumn{4}{c}{Sports-Copurchase} & \multicolumn{4}{c}{Cloth-Copurchase$^{\dagger}$} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r" & H@1 & H@3 & H@10 & MRR & H@1 & H@3 & H@10 & MRR \\",
        r"\midrule",
    ]
    rankings = _rankings(sports, LP_METRICS)
    cloth_rankings = _rankings(cloth, LP_METRICS)
    for model in MODELS:
        if model == "mopf":
            lines.append(r"\midrule")
        cells = [DISPLAY_NAMES[model]]
        cells.extend(_decorated_stat(sports, model, metric, rankings) for metric in LP_METRICS)
        cells.extend(_decorated_stat(cloth, model, metric, cloth_rankings) for metric in LP_METRICS)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\multicolumn{9}{l}{\footnotesize $^{\dagger}$ Cloth-Copurchase is quasi-held-out under the frozen protocol.} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def _read_cloth_status() -> str:
    protocol = FINAL_PROTOCOL_SOURCE.read_text(encoding="utf-8")
    if "current formal LP benchmark is sports-copurchase" not in protocol:
        raise ValueError("Final protocol no longer identifies sports-copurchase as the current formal LP benchmark")
    if "cloth-copurchase remain quasi-held-out" not in protocol:
        raise ValueError("Final protocol no longer marks cloth-copurchase as quasi-held-out")
    if not CLOTH_SOURCE.is_file():
        raise FileNotFoundError(f"Expected real Cloth quasi-held-out summary does not exist: {CLOTH_SOURCE}")
    return (
        "No formal cloth-copurchase LP benchmark result or formal summary was found. "
        "The table now includes the real three-seed values from "
        "outputs/f1_final_execution/tables/f1_lp_cloth_quasiheldout_paper_table.csv "
        "as a separately marked quasi-held-out extension; these values are not formal LP evidence."
    )


def _preview_tex() -> str:
    return r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.6cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\begin{document}

\input{table_nc_main.tex}
\clearpage
\input{table_lp_main.tex}

\end{document}
"""


def _readme(cloth_status: str) -> str:
    return f"""# Paper result tables

Generated by `scripts/generate_paper_result_tables.py`. Numeric cells are parsed from the cited repository documents at generation time; no benchmark values are hard-coded in the generator.

## Sources and scope

1. **NC:** `docs/nc_benchmark_results.md`, the dataset sections `Movies`, `Toys`, `Grocery`, `ele-fashion`, and `Reddit-S`. The table uses only `Test Accuracy` and `Test Macro-F1` from the common historical benchmark table, in percent and formatted to two decimal places as mean ± population standard deviation. Validation Accuracy is not included.
2. **NC metric note:** `docs/nc_macro_f1_metric_fix.md` documents the fixed task-level label set. The current draft intentionally uses the same-batch values in `docs/nc_benchmark_results.md` for baseline comparability. The ele-fashion Macro-F1 baseline values are legacy-evaluator values; **ele-fashion Macro-F1 requires final fixed-metric rerun before paper submission**. Fixed-MoPF and legacy baselines are not mixed for a final claim.
3. **LP Sports-Copurchase:** `docs/lp_benchmark_results.md`. External baseline rows are parsed from the historical sports table; the MoPF row is parsed from the current formal sports section. The protocol is `unified_sampled_lp_v1`, checkpoint selection uses validation MRR, and the current document reports seeds 42/43/44, so the table uses mean ± population standard deviation. No validation metric is placed in the main table.
4. **LP Cloth-Copurchase:** {cloth_status}

## Models

The paper-facing set is exactly: MLP, GCN, GraphSAGE, MMGCN, MGAT, DiP, DGF, DMGC, LGMRec, and MoPF. `map_mag`, `map_mag_v1`, `map_mag_v2`, and `map_mag_v3` are internal predecessors and are excluded from the tables and all best/second-best rankings.

## Ranking and formatting

For every dataset × metric, ranking is recomputed using the mean among the ten included rows only. The largest mean is `\\textbf{{}}`; the second-largest is `\\underline{{}}`. Standard deviations are not used for ranking. Ties, if encountered, are resolved by the declared model order and are not treated as evidence of a meaningful difference. Tables use `table*`, booktabs rules, no vertical lines, and no colored backgrounds. MoPF is separated from the external baselines by `\\midrule`.

## Formality and limitations

- The NC historical external results were later behavior-equivalence certified in the repository, but the historical source lacks an original producing execution SHA; this is a provenance limitation for a final paper table.
- The formal LP benchmark is Sports-Copurchase. Cloth-Copurchase is explicitly quasi-held-out under the final protocol; its real three-seed values are included with a dagger marker for completeness and ranked only within that quasi-held-out table, not presented as a formal benchmark.
- The table values are descriptive benchmark summaries. No significance testing or claim of global SOTA is added here.

## Generated files

- `table_nc_main.tex`
- `table_lp_main.tex`
- `table_results_preview.tex`
- `table_results_preview.pdf` (when `pdflatex` is available)
"""


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    nc_values = _load_nc()
    sports_values = _load_lp_sports()
    cloth_values = _load_lp_cloth()
    cloth_status = _read_cloth_status()

    (TABLE_ROOT / "table_nc_main.tex").write_text(_nc_tex(nc_values), encoding="utf-8")
    (TABLE_ROOT / "table_lp_main.tex").write_text(_lp_tex(sports_values, cloth_values), encoding="utf-8")
    (TABLE_ROOT / "table_results_preview.tex").write_text(_preview_tex(), encoding="utf-8")
    (TABLE_ROOT / "README.md").write_text(_readme(cloth_status), encoding="utf-8")

    print(f"Wrote {TABLE_ROOT / 'table_nc_main.tex'}")
    print(f"Wrote {TABLE_ROOT / 'table_lp_main.tex'}")
    print(f"Wrote {TABLE_ROOT / 'table_results_preview.tex'}")
    print(f"Wrote {TABLE_ROOT / 'README.md'}")
    print(cloth_status)


if __name__ == "__main__":
    main()
