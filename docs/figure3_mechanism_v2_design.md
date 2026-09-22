# Figure 3 V2: mechanism verification design

## Scope and evidence chain

Figure 3 V2 is restricted to the first two CoSI-MAG stages and follows the
mechanism chain

`edge-level response -> neighborhood-level consequence -> multi-hop propagation behavior`.

The analysis reads existing full/final checkpoints only. It does not train a
new model, create a new model variant, or modify the formal model, task,
data-pipeline, benchmark, or Figure 1 implementation.

The data script is `scripts/generate_figure3_mechanism_v2.py` and the plotting
script is `scripts/plot_figure3_mechanism_v2.py`.

The default datasets are Movies and Grocery, with seeds 42, 43, and 44. The
default checkpoint root is
`outputs/cosi_mag_final_benchmark/nc/{dataset}/runs_42_43_44/`.

## Correctness audit

### A. Definition of raw semantic similarity in Panel (a)

The primary semantic variables `S^T_raw` and `S^V_raw` are edge-wise cosine
similarities of the projected initial states `H_0^T` and `H_0^V`, respectively,
before the learned diagonal metric is applied. This is labelled
`projected_h0_raw` in the generated CSV files.

This choice measures whether the learned relation weights respond to the
modality-specific semantic geometry supplied to the relation calibration
stage, without using the same learned metric score as both input and response.
For auditability, the CSV also retains:

- `input_raw`: cosine on the original text/visual input features;
- `learned_metric_internal`: the model's learned-metric cosine used internally
  to produce `W^T` and `W^V`.

Only `projected_h0_raw` is used for the primary A1 and A2 panels. The latter
two spaces are sensitivity/audit rows, not additional claims in the main
figure.

### B. Definition of calibrated neighborhood compatibility in Panel (b)

The primary calibrated compatibility uses the raw learned relation weights
`W^m = relation_weight_m` directly:

```text
u_i^m(cal) = sum_j W_ij^m S_ij^m(raw) / sum_j W_ij^m
```

It does not use `normalized_edge_weight_m` and therefore does not mix the MRC
test with degree normalization, self-loops, or the propagation operator. The
raw compatibility is the unweighted mean over the original physical one-hop
neighbors. The semantic term is the same projected-H0 raw cosine used by
primary Panel (a).

### C. Degree audit for Panel (b)

The main improvement CSV contains only nodes with physical out-degree greater
than one. Degree-zero and degree-one nodes are not included in the main bar
chart. Their counts and fractions, together with the degree>1 node count, are
written to `panel_b_degree_audit.csv`.

The degree convention is the source endpoint of the physical `edge_index`,
which matches the source-to-target propagation convention used by the formal
model. The audit file makes the exclusion explicit for every dataset and seed.

### D. Same-checkpoint counterfactual for Panel (c)

Panel (c) uses each full checkpoint twice:

1. **Anchored:** the normal recurrence from the checkpoint, with its configured
   `multihop_anchor_alpha`.
2. **Anchor-off counterfactual:** the same checkpoint, same projected `H_0`,
   same calibrated weights, and same normalized propagation operator, but the
   recurrence is recomputed with `alpha=0`.

No `no_semantic_anchor` retrained checkpoint is used as the primary source. For
each hop, retention is the mean node-wise cosine between `S_k^m` and the
initial `H_0^m`. Hop-3 gain is reported in percentage points as anchored minus
anchor-off retention.

## Panel definitions and candidates

### Panel (a), A1: direct modality alignment

For each dataset and seed, the script computes Spearman rho for four explicit
pairings:

```text
W^T vs S^T_raw    W^T vs S^V_raw
W^V vs S^V_raw    W^V vs S^T_raw
```

The output is `panel_a_correlation.csv`. Matched pairs use saturated blue/green
colors; cross-modal pairs remain separate and use lower-saturation light-gray
encoding. The candidate export is
`figure3a_correlation_v2.{png,tiff,pdf,svg}`.

### Panel (a), A2: semantic-gap response

For each physical edge, the script also computes

```text
delta_sem = S^T_raw - S^V_raw
delta_weight = W^T - W^V
```

Edges are split into eight equal-count bins by `delta_sem`. The CSV retains
per-seed edge-level Spearman rho and linear slope, together with binned means
and uncertainty summaries. The candidate export is
`figure3a_gap_response_v2.{png,tiff,pdf,svg}`.

The automatic main-panel rule promotes A2 only if both datasets have at least
two seed-level observations, mean edge-level rho >= 0.20, positive mean slope,
and at least two thirds of available seed-level slopes are positive. Otherwise
the combined figure uses A1 as the more conservative direct alignment panel.
The rule and its statistics are recorded in `plot_manifest.json`.

### Panel (b): neighborhood-level consequence

For each degree>1 node and modality, the script stores

```text
delta_u_i^m = u_i^m(cal) - u_i^m(raw)
```

The main candidate plot is a bar chart of the percentage of nodes with
`delta_u_i^m > 0`, with seed-level standard-deviation error bars. Each bar is
annotated with the pooled median delta in units of `1e-3`. The candidate export
is `figure3b_compatibility_v2.{png,tiff,pdf,svg}`.

### Panel (c): propagation-level semantic retention

Movies and Grocery are shown as two shared-y facets. Each facet contains four
curves: text/visual crossed with anchored/anchor-off. Anchored curves are
solid; anchor-off curves are dashed. Text is blue and visual is green. Hop-3
anchored-minus-anchor-off gains are annotated in percentage points.

The candidate export is `figure3c_retention_v2.{png,tiff,pdf,svg}`.

## Reproducibility and missing-data behavior

Use a dry-run to audit checkpoint availability without loading models:

```bash
PYTHONPATH=. conda run --no-capture-output -n yhf_env \
  python scripts/generate_figure3_mechanism_v2.py --dry-run
```

Generate the CSVs and audit manifest:

```bash
PYTHONPATH=. conda run --no-capture-output -n yhf_env \
  python scripts/generate_figure3_mechanism_v2.py --device cpu
```

Generate both Panel (a) candidates and the combined figure:

```bash
MPLCONFIGDIR=/tmp/figure3_mpl PYTHONPATH=. \
  conda run --no-capture-output -n yhf_env \
  python scripts/plot_figure3_mechanism_v2.py --main-panel-a auto
```

Missing checkpoints or failed analyses are not silently replaced. They are
listed in `run_manifest.json` and `audit_summary.md`; available rows are still
written so that partial audits can be inspected. No training command is
constructed or launched by either script.

## Output contract

All V2 outputs are written under `outputs/figure3_mechanism_v2/`.

Required data/audit files:

- `panel_a_correlation.csv`
- `panel_a_gap_response.csv`
- `panel_b_improvement.csv`
- `panel_b_degree_audit.csv`
- `panel_c_retention_counterfactual.csv`
- `run_manifest.json`
- `audit_summary.md`
- `plot_manifest.json`

The combined publication exports are:

- `figure3_mechanism_v2.png`
- `figure3_mechanism_v2.pdf`
- `figure3_mechanism_v2.svg`

TIFF and alignment-audit sidecars are also emitted for figure QA.
