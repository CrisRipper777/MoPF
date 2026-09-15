# MoPF paper-figure revision v1

## Scope

This revision consolidates the existing Figure 1, Figure 2, and Figure 3 into publication-facing main-figure exports. It is a visualization-only pass. The authoritative E0-A/E0-B/E0-C/E0-D statistics were not recomputed or modified, and no model, configuration, checkpoint, training run, F2 analysis, or new empirical/mechanism analysis was executed.

The previous exports under `outputs/paper_figures/` and the complete E0 diagnostic figures remain unchanged. New files are written only under `outputs/paper_figures/revision_v1/`.

## Source audit

The source workflow is [scripts/plot_mopf_paper_figures_revision_v1.py](../scripts/plot_mopf_paper_figures_revision_v1.py). It imports the existing artifact loaders from `scripts/plot_mopf_paper_figures.py`; that earlier plotting script was inspected but not modified in this revision.

| Main panel | Frozen source | Visualization use |
|---|---|---|
| Figure 1(a) | Conceptual | Symbolic multimodal node cards; no dataset case is fabricated. |
| Figure 1(b) | Conceptual | 0-hop through 3-hop schematic; no CKA or monotonic-loss claim is drawn. |
| Figure 1(c) | Conceptual | Symbolic node/modality hop profiles; no solution module is shown. |
| Figure 2(a) | `outputs/e0_empirical_motivation/edge_semantic_discrepancy/` | Movies uses the existing fixed E0-A plot sample; the compact summary reads full-edge `spearman_tv` and `median_rank_gap`. The dense plot is rendered as a vector hexbin. |
| Figure 2(b) | `outputs/e0_empirical_motivation/semantic_retention/semantic_retention_summary.csv` | Plots the requested display transform `1 - cka_kK` at each frozen formal K. No CKA is recomputed. |
| Figure 2(c), left | `outputs/e0_empirical_motivation/multihop_utilization/` | Uses frozen seed-mean node profiles and existing per-seed median markers. |
| Figure 2(c), right | `outputs/e0_empirical_motivation/relation_utilization_bridge/relation_utilization_association.csv` | Uses frozen degree-controlled partial Spearman `rho_partial_degree`, three seed points, and mean ± SD. |
| Figure 3 | Frozen MoPF architecture definition | Architecture diagram only; no checkpoint-derived quantity is plotted. |

The Figure 2 main panel deliberately omits the full CKA trajectories, cosine diagnostics, E0-C profile-gap panel, and E0-D quartile curves. Those complete diagnostics are retained as appendix candidates.

## Publication-facing design

- White background; no embedded `Figure 1.`, `Figure 2.`, or `Figure 3.` titles.
- Text: blue `#2F6FB3`.
- Visual: green/teal `#3B9960`.
- Shared physical structure: gray `#737B83`.
- Relation-conditioned cross-stage interaction: orange `#D47B22`.
- No light-red problem cue was needed.
- Font family: DejaVu Sans.
- Base font: 8.4 pt; panel headings: 10.2 pt; axis labels: 8.2 pt; ticks and legends: 7.4 pt.
- PDF and SVG use the fixed untrimmed canvas; PNG exports use 300 dpi.

Nominal canvas dimensions are:

| Figure | Size |
|---|---:|
| Figure 1 main | 7.2 × 2.85 in |
| Figure 2 main | 7.2 × 7.45 in |
| Figure 3 main | 13.6 × 5.15 in |

## Outputs

- Figure 1: `figure1_main.pdf`, `figure1_main.svg`, `figure1_main.png`.
- Figure 2: `figure2_main.pdf`, `figure2_main.svg`, `figure2_main.png`.
- Figure 3: `figure3_main.pdf`, `figure3_main.svg`, `figure3_main.png`.
- Caption drafts: `captions.md`.
- Reproducibility manifest: `revision_manifest.json`.

All files are in `outputs/paper_figures/revision_v1/`.

Existing appendix candidates preserved include the five-dataset E0-A rank plots, E0-B CKA and cosine plots, the E0-C profile-gap view, and the E0-D full quartile plot.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/mopf_paper_figures_mpl \
PYTHONPATH=src conda run --no-capture-output -n yhf_env \
python scripts/plot_mopf_paper_figures_revision_v1.py
```

Run manifest records branch `vnext` and commit `b080cb8058628eea3b9da708073f72a083d82a6e`.

## Warnings and limitations

1. Figure 1 is a clean symbolic problem statement. Its multimodal node cards are not presented as real item/post examples.
2. Figure 2(b) labels `1 - CKA` as representation shift; it does not mean semantic loss, oversmoothing, information destruction, or performance decline.
3. Figure 2(c) keeps seed-mean node profiles and seed-level markers; it does not pool `3 × N` observations for inference.
4. Figure 3 is an architecture schematic. It visualizes the frozen MoPF mechanism and the orange Stage-I-to-TCPR bridge without replacing the formal Method equations.
5. The captions are drafts for LaTeX integration; captions are not embedded in the image files.
