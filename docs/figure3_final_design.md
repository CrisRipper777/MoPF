# Figure 3 final design

## Figure-level claim

Figure 3 follows one mechanism chain:

`edge-level semantic discrepancy -> modality-specific neighbor allocation -> semantic retention during propagation`

The figure is a quantitative-grid composite. Panel (a) establishes the edge-level response, Panel (b) shows the neighborhood-level consequence, and Panel (c) tests the propagation-level semantic-retention mechanism.

## Panel definitions

### (a) Relation response to semantic discrepancy

This panel keeps the V2 A2 analysis. For each dataset and semantic-gap bin, it plots the existing seed-level summary of:

- x: `delta_sem = S^T_raw - S^V_raw`, where the semantic similarities are cosine similarities on projected `H0` representations;
- y: `delta_weight = W^T - W^V`, using raw learned relation weights;
- error bars: standard deviation across the existing seeds.

The panel therefore tests whether MRC responds monotonically to cross-modal semantic discrepancy. The existing response is positive for both Movies and Grocery.

### (b) Modality-specific neighbor allocation

The final panel uses V3 B3. It uses raw `W^T` and `W^V` on exactly the same physical incoming neighbor IDs, excludes degree-one nodes, and does not use the normalized propagation operator or add self-loops.

The two subplots are:

- **TV divergence:** `TV_i = 0.5 * sum_j |p_ij^T - p_ij^V|`, shown as pooled node-level violin + thin box + median marker. The direct labels `m=...` denote the pooled median. The y-axis is `0–0.11`, which retains the full observed range while reducing unused long-tail space.
- **Top-neighbor disagreement:** pooled percentage of nodes for which `argmax_j p_ij^T != argmax_j p_ij^V`, with values labeled on the bars. Seed-level variability is recorded in `panel_b_neighbor_allocation_summary.csv` and the final audit summary rather than added as error bars.

The supported statement is:

> MRC induces modality-specific neighbor allocation on the same physical neighborhood.

This panel does **not** claim that propagation quality necessarily improves. The earlier positive-Δu compatibility result remains supplementary in the V2 output directory.

Muted purple/orange colors are used for Movies/Grocery in this panel, inspired by the provided reference figures. This keeps the dataset encoding visually distinct while preserving blue/green for the global Text/Visual language used in Panel (c).

### (c) Semantic retention from anchored propagation

This panel keeps the V2 same-checkpoint counterfactual. For each hop `k = 0, 1, 2, 3`, it plots mean cosine similarity between the propagated state and the initial modality representation:

`R_k^m = mean_i cosine(S_{i,k}^m, H_{i,0}^m)`.

Anchored propagation and an anchor-off recurrence (`alpha=0`) use the same full checkpoint and all other parameters unchanged. Text is blue, Visual is green, solid lines are anchored, dashed lines are anchor-off, and shaded regions show standard deviation across seeds. Hop-3 gains are annotated in each dataset facet.

## Layout and export contract

- Main figure: horizontal `(a) | (b: TV + top-1) | (c: Movies + Grocery)` layout.
- Main canvas follows the wide Figure 1 V5-style contract: `17 × 4.8 in`, with outer width ratios `[1.10, 2.10, 2.05]` and compact inter-group spacing. This gives the five quantitative axes comparable physical width while preserving the larger group allocations for the two-part panels.
- Typography uses the same restrained serif hierarchy as Figure 1 V5; standalone Panel (b) is widened to `158 × 88 mm` for a less crowded two-axis layout.
- Exports preserve the declared canvas dimensions; no automatic tight-cropping is applied.
- The generic static width heuristic reports an intentional warning for the `431.8 mm` canvas because it is wider than the default `89/183 mm` presets; the rendered PDF is deliberately matched to the wide reference-figure contract.
- White background, editable SVG text, TrueType PDF text, 600-dpi PNG/TIFF exports.
- Standalone exports use `figure3a_final.*`, `figure3b_final.*`, and `figure3c_final.*`.
- The combined export uses `figure3_final.*`.
- Render-time alignment and PDF collision audit artifacts are written beside the final outputs.

## Reproducibility

The final plotting script only reads existing CSV outputs:

- `outputs/figure3_mechanism_v2/panel_a_gap_response.csv`
- `outputs/figure3_mechanism_v2/panel_c_retention_counterfactual.csv`
- `outputs/figure3_mechanism_v3_candidate/panel_b_neighbor_allocation.csv`
- `outputs/figure3_mechanism_v3_candidate/panel_b_neighbor_allocation_summary.csv`

No checkpoint loading or training occurs in the final plotting step. The generated `plot_manifest.json` and `audit_summary.md` record the source-row counts, definitions, output contract, and key numerical summaries.
