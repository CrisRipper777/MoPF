# Figure 1 empirical observations behind CoSI-MAG (v3)

The v3 renderer is `scripts/plot_figure1_empirical_final_v3.py`. This revision
only changes visual encoding and layout; it reuses the existing empirical CSV
definitions and summary statistics. It does not load a CoSI-MAG checkpoint or
start base-model training.

The figure uses a white canvas, consistent serif typography, thin axes, very
light grid lines, muted blue Text (`#4C78A8`), muted green Visual (`#59A14F`),
terracotta only for legacy discrepancy accents, and neutral gray context
curves. The combined figure is a horizontal 1x3 layout with concise titles:
Relation discrepancy, Propagation drift, and Preferred hop-order heterogeneity.

## Panel (a): Relation discrepancy

For every unique physical edge in `relation_discrepancy_edges.csv`, the source
data provide text and visual rank percentiles. The plotted quantity is

```text
100 * abs(text_rank_percentile - visual_rank_percentile).
```

For each of Movies, Toys, Grocery, ele-fashion, and Reddit-S, the dark blue
dot is the median, the thicker light blue-gray segment is the 25th--75th
percentile interval, and the thinner light blue-gray segment is the
10th--90th percentile spread. The dashed x=0 line marks perfect agreement.

Higher values mean stronger disagreement in the ordering of the same physical
edges across modalities. Lower values mean more similar cross-modal relation
ordering. Spearman rho remains in `relation_discrepancy_summary.csv` for the
caption or text and is not added as a second subplot.

## Panel (b): Propagation drift

For modality `m` and hop `k`, the source metric is the node-wise

```text
1 - cosine(H0^m, S_k^m).
```

The plotted values retain the existing display scaling of 100 times this
quantity. The colored Text and Visual curves are means across the five
dataset-level means, with the light ribbons showing the 25th--75th percentile
across dataset-level means. The five individual dataset curves for each
modality are shown as thin light-gray dashed background trajectories.

Higher values mean a larger shift away from the hop-0 representation. The
Reddit-S post-hop-one flatness is retained from the existing curve and sanity
audit; no additional diagnostic is introduced here.

## Panel (c): Preferred hop-order heterogeneity

The main panel now uses a paired 100% stacked bar chart. Each dataset has two
bars: Text on the left and Visual on the right. Each bar sums to one and gives
the unchanged fraction of validation nodes whose best propagation order is
`k=0`, `k=1`, `k=2`, or `k=3`, read from
`order_utility_heterogeneity.csv`.

The underlying model-independent definition is unchanged. For every dataset,
modality, and hop, identical lightweight linear probes were trained on the
frozen states `S_0^m,...,S_3^m` with the same classifier architecture,
full-batch Adam optimizer, learning rate 0.01, zero weight decay, 100 epochs,
seed 42, and the existing NC train/validation split. For validation node `i`,

```text
k_i^{*,m} = argmin_k CE(f_k^m(S_{i,k}^m), y_i).
```

The panel is model-independent: it does not use RCMI, attention,
relation-conditioned interaction, learned eta, or final CoSI-MAG preference.

Text uses the prescribed blue sequential shades from light (`k=0`) to dark
(`k=3`); Visual uses the corresponding green sequential shades. The compact
in-panel note explains this mapping without combining modality and order into
a crowded six-entry legend. Higher mass at multiple shades means validation
nodes use multiple propagation orders; concentration in one shade would be
consistent with a fixed-order preference.

The old eta-based `aggregation_heterogeneity_summary.csv` and
`aggregation_heterogeneity_examples.json` remain reserved for Figure 4
mechanism analysis. They are not used for the v3 panel (c).

## Inputs, commands, and outputs

The v3 plotter reads:

- `relation_discrepancy_edges.csv` for panel (a);
- `semantic_drift_curves.csv` for panel (b);
- `order_utility_heterogeneity.csv` for panel (c).

The unchanged model-independent probe data were generated for all five NC
datasets with:

```bash
conda run --no-capture-output -n yhf_env python \
  scripts/generate_empirical_motivation_data.py \
  --only-order-utility \
  --order-datasets Movies Toys Grocery ele-fashion Reddit-S
```

Render the v3 figure with:

```bash
python scripts/plot_figure1_empirical_final_v3.py
```

The script writes to `outputs/figure1_empirical_motivation_final/`:

- `figure1_empirical_final_v3.pdf`, `.svg`, `.png`;
- `figure1a_relation_discrepancy_final_v3.pdf`, `.svg`, `.png`;
- `figure1b_propagation_drift_final_v3.pdf`, `.svg`, `.png`;
- `figure1c_order_heterogeneity_final_v3.pdf`, `.svg`, `.png`;
- `figure1_empirical_final_v3_manifest.json` and alignment QA artifacts.

All plotted numeric values are checked for finite values. No formal model,
task, benchmark, configuration, or data-pipeline file is modified.
