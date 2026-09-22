# Figure 3 mechanism design

## Figure-level claim

CoSI-MAG's first two modules perform two linked operations: MRC makes the
learned relation weights modality-aligned and improves local semantic
compatibility, while SMP preserves the initial modality semantics as graph
propagation goes deeper.

The figure is a quantitative mechanism-validation composite. Panel (a) is the
primary relation-calibration evidence, panel (b) is the local compatibility
consequence, and panel (c) is the propagation-depth control/ablation.

## Evidence chain

| Panel | Question | Evidence role | Decisive comparison | Unique inference |
|---|---|---|---|---|
| (a) | Do calibrated relation weights align with the matching modality? | Primary mechanistic evidence | matched vs cross-modal Spearman rho | MRC is modality-aware rather than merely non-uniform |
| (b) | Do those weights improve the semantic compatibility of each node's local neighborhood? | Local consequence / orthogonal validation | calibrated minus raw neighborhood compatibility | the alignment is operational at the neighborhood level |
| (c) | Does the semantic anchor preserve modality states through repeated propagation? | Ablation-based propagation validation | Full vs `no_semantic_anchor` across hop order | SMP reduces semantic drift |

## Statistical definitions

### Panel (a)

For each dataset and full-model checkpoint seed, the data script uses the
model-returned physical-edge relation weights `W^m` and the raw input
feature cosine on the same physical edges `S^m` (or the other modality for the
cross-modal control). The plotted statistic is Spearman rho. The four displayed
quantities are `W^T–S^T`, `W^T–S^V`, `W^V–S^V`, and `W^V–S^T`. Bars are means
across seeds and error bars are population standard deviations across seeds.

The raw semantic similarities are computed from the original frozen feature
arrays, not from the model's projected `h0` states. This preserves the intended
meaning of “raw semantic similarity” and prevents the analysis from measuring a
projection-space quantity as if it were an input-space control.

### Panel (b)

For node `i` and modality `m`, the raw compatibility is the unweighted mean of
the raw modality cosine over incident physical edges. The calibrated
compatibility is the relation-weighted mean using `W^m` on the same edges. The
reported node-level quantity is

```text
Δu_i^m = u_i^m(calibrated) − u_i^m(raw).
```

The graph loader supplies the undirected physical `edge_index`, so both
directions are retained for incident-neighborhood aggregation. Zero-degree
nodes are excluded because both the raw and calibrated denominators are
undefined. The violin distributions pool valid full-model nodes across the
three existing checkpoint seeds; the white dot is the median and the vertical
segment is the interquartile range.

### Panel (c)

For every dataset, seed, variant, modality, and hop `k`, the script computes

```text
r_k^m = mean_i cosine(S_{i,k}^m, H_{i,0}^m).
```

`S_{i,k}^m` is the model's pre-interaction multi-hop state bank and `H_{i,0}^m`
is the corresponding projected initial state returned by the existing
read-only analysis interface. The plot shows means across seeds with population
standard-deviation ribbons. Full uses solid lines; `no_semantic_anchor` uses
dashed lines. This isolates SMP's anchor effect from the later cross-order
interaction and node preference stages.

## Provenance and missing-data policy

The data generator only reads existing checkpoints, resolved configs, dataset
features, and graph edges. It never invokes a training launcher. Missing
checkpoints or analysis failures are retained in `run_manifest.json` and the
summary markdown; corresponding CSVs remain partially populated so the plotter
can render available series and mark absent panels rather than fabricating
values. The manifest also records suggested rerun commands for missing
`no_semantic_anchor` units.

The source-data files are:

- `outputs/figure3_mechanism/panel_a_relation_alignment.csv`
- `outputs/figure3_mechanism/panel_b_neighborhood_compatibility.csv`
- `outputs/figure3_mechanism/panel_c_semantic_retention.csv`

The complete audit record is `outputs/figure3_mechanism/run_manifest.json`, and
the compact trend report is `outputs/figure3_mechanism/figure3_mechanism_summary.md`.
