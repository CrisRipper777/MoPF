# MGSC-MAG stage finalization report

## Scope and decision

The current paper scope in this round is **five NC datasets only**:
Movies, Toys, Grocery, ele-fashion, and Reddit-S. LP, including Sports-LP, is
not discussed or used in this round. The frozen CoSI reference files and the
unified NC protocol were not modified.

The working hypothesis is:

> In a multimodal attributed graph, physical topology defines latent structural
> dependence, but the semantic utility of structural context is not uniform at
> the relation, node/context, or propagation-order granularities.

The current working architecture is canonical P2, MGSC-MAG with
`adaptive_context_gate=true`, `direct_interacted_integration=true`, and
`use_legacy_relation_order_bias=true`.

## A. Architecture data flow

Text and visual features are projected independently to `H0^T` and `H0^V`.
For each modality, MRC computes semantic edge scores, relation weights, and a
normalized graph operator over the unchanged physical topology. Each order
forms a neighborhood proposal from the previous state. A modality-specific
gate MLP uses `(H0, N, |H0-N|, H0*N, p_k)` to form `S_k`; this yields the
modality-specific state bank `[S0,...,SK]`.

Each state receives its order embedding and enters the existing single-head
cross-order attention with retained legacy relation-order bias. The interacted
states are used directly in P2 composition with the existing global,
modality, and node preference coefficients `eta`. Text and visual outputs are
refined separately, then fused only at the final late-fusion stage. No early
fusion or topology change was introduced.

The exact formulas and canonical switches are recorded in
`docs/final/MGSC_P2_ARCHITECTURE.md` and
`configs/model/mgsc_mag_p2.yaml`.

## B. M1 support for heterogeneous contextualization demand

M1 used model-independent row-normalized frozen modality features, physical
neighbors only, no self-loop in the neighborhood mean, no labels, no MRC, and
an internal deterministic train/dev probe split. Official validation nodes
were used only for the predeclared final probe evaluation; test labels were not
used. The node-wise oracle gap is descriptive and is not realizable model or
test performance.

| Dataset | Modality | Global-best λ | Entropy | Non-global ratio | T/V disagreement | Oracle relative gap |
|---|---|---:|---:|---:|---:|---:|
| Movies | Text | 0.00 | 0.7155 | 0.5210 | 0.6509 | 0.0121 |
| Movies | Visual | 0.50 | 0.9081 | 0.8755 | 0.6509 | 0.0390 |
| Grocery | Text | 0.00 | 0.6625 | 0.4996 | 0.6120 | 0.0088 |
| Grocery | Visual | 0.50 | 0.8784 | 0.8952 | 0.6120 | 0.0534 |
| ele-fashion | Text | 0.25 | 0.9455 | 0.8736 | 0.6280 | 0.0530 |
| ele-fashion | Visual | 0.25 | 0.8956 | 0.8892 | 0.6280 | 0.0494 |

At least two datasets satisfy the predeclared support rule, and the collapse
exception is absent. **Gate A: PASS.** This supports entering P1, but it does
not make the oracle policy realizable or establish statistical significance.

## C. Functional ablation

The formal matrix contains 5 datasets × 3 seeds × 7 variants = 105 runs. The
authoritative outputs are in
`outputs/final/mgsc_functional_ablation/`. Full P2 was reused only after code,
split, and effective protocol/config fingerprints matched.

| Dataset | Full Acc ± SD | Full Macro-F1 ± SD |
|---|---:|---:|
| Movies | 0.5640 ± 0.0115 | 0.5064 ± 0.0104 |
| Toys | 0.7990 ± 0.0053 | 0.7736 ± 0.0057 |
| Grocery | 0.8323 ± 0.0053 | 0.7566 ± 0.0067 |
| ele-fashion | 0.8833 ± 0.0019 | 0.7789 ± 0.0045 |
| Reddit-S | 0.9691 ± 0.0026 | 0.9321 ± 0.0035 |

Mean paired deltas are `Full - Ablation`, in percentage points:

| Dataset | Uniform relations Acc/F1 | Global gate Acc/F1 | Terminal only Acc/F1 | Uniform integration Acc/F1 | No cross-order Acc/F1 |
|---|---:|---:|---:|---:|---:|
| Movies | −0.13 / +0.31 | −0.04 / +0.76 | −0.25 / −0.63 | +0.20 / +0.30 | +0.22 / +0.79 |
| Toys | −0.13 / +0.12 | −0.34 / −0.26 | −0.38 / −0.06 | +0.10 / +0.77 | −0.29 / −0.22 |
| Grocery | −0.17 / −0.07 | +0.23 / +0.03 | −0.15 / −0.23 | −0.03 / −0.17 | −0.29 / −0.60 |
| ele-fashion | +0.05 / +0.15 | +0.16 / −0.06 | +0.08 / +0.48 | +0.03 / +0.25 | +0.02 / −0.12 |
| Reddit-S | +0.00 / +0.07 | +0.07 / +0.15 | −0.06 / −0.08 | +0.11 / +0.19 | +0.04 / +0.08 |

The effects are mixed and mostly small. Therefore the ablation matrix supports
the existence and functional use of the paths only in a bounded sense; it does
not support a universal performance ranking of modules. Attribute Only is a
sanity control, not a pure isolation of one innovation; its mean accuracy is
lower than Full on all five datasets.

## D. Mechanism chain

The current evidence supports the following computational chain:

`MRC relation response → modality-specific adaptive state formation →
multi-order state bank → cross-order interaction → direct interacted-state
composition`.

It is a real computation chain because the intervention outputs change the
intermediate/final representations, while the ablations preserve the task
protocol and selectively replace the specified path.

## E. Figure 3 diagnostics: context formation

The source data and preview are under
`outputs/final/context_formation_analysis/` and
`outputs/final/paper_figures/figure3_data/`.

- Relation calibration has nonzero text/visual response differences. For
  Movies/Grocery/ele-fashion, mean semantic-score discrepancy is 0.1913,
  0.1531, and 0.1453; corresponding learned relation-weight discrepancy is
  0.0569, 0.0502, and 0.0446.
- Neighbor allocation differs while physical topology is fixed. Mean TV
  distance is 0.0153, 0.0159, and 0.0082 for those three datasets; top-neighbor
  disagreement is 0.5800, 0.4976, and 0.2984.
- Gate behavior is nontrivial and modality-dependent. For example, Movies has
  mean text/visual gates of approximately 0.328/0.803, while Grocery is
  approximately 0.633/0.590.
- Shuffled gate interventions preserve gate marginals but change embeddings
  and predictions. Embedding MAE / flip rate are Movies 0.0471/0.0323,
  Grocery 0.0567/0.0148, ele-fashion 0.0941/0.0193, with nonzero effects also
  on Toys and Reddit-S.

These results support functional node-to-gate correspondence, not a claim that
the gate is a supervised semantic ground truth or that it recovers a unique
optimal lambda.

## F. Figure 4 diagnostics: multi-order integration

The source data and preview are under
`outputs/final/multi_order_integration_analysis/` and
`outputs/final/paper_figures/figure4_data/`.

Order contributions use the explicitly reported magnitude normalization
`q=abs(eta)/sum(abs(eta))`. Mean effective orders (text / visual) are:

| Dataset | Text | Visual |
|---|---:|---:|
| Movies | 1.743 | 1.803 |
| Toys | 1.818 | 1.979 |
| Grocery | 1.898 | 1.983 |
| ele-fashion | 1.795 | 1.719 |
| Reddit-S | 1.760 | 1.738 |

The attention CSV contains average query/key matrices for both modalities and
all five datasets. Interaction-off changes are larger than uniform-attention
changes in all five datasets: logit MAE is 0.2441/0.0439 on Movies,
0.2084/0.0425 on Toys, 0.3175/0.0545 on Grocery, 0.6609/0.1474 on ele-fashion,
and 0.1324/0.0232 on Reddit-S. This supports direct functional influence of
the interaction residual, without implying a guaranteed accuracy gain.

## G. Prior-init control

The controlled direct-vs-legacy comparison was run on Movies, Grocery, and
ele-fashion at seed 42 only. Direct-minus-legacy accuracy was −0.27, −0.38,
and −0.10 percentage points; Macro-F1 was −2.23, −0.09, and +0.24 points.
Because Movies Macro-F1 moved materially in this single-seed control, the
current evidence does not justify changing the canonical P2 initialization.
Keep `legacy_anchored` in canonical P2; retain `direct` as a future controlled
study only.

## H. Claims allowed and claims not allowed

Supported claims:

- contextualization demand is heterogeneous in the model-independent M1 probe;
- P2 computes modality-specific relation response, adaptive state formation,
  multi-order integration, and late fusion as separate stages;
- gate and interaction interventions have measurable functional effects;
- P2 is a viable unified working architecture under the five-NC protocol.

Claims not supported by this round:

- that semantic similarity is relation reliability;
- that node-wise oracle lambda is realizable performance;
- that every P2 component is necessary or universally improves accuracy;
- that one modality/order is globally optimal;
- that mechanism interventions are causal identification or statistical
  significance tests;
- any LP conclusion.

## I. Freeze recommendation and risks

I recommend freezing the current canonical P2 as the working architecture for
the five-NC paper scope. There is no architecture-search blocker in the
completed matrix. The main caveats are mixed small ablation deltas, the
single-seed prior-init control, and the fact that diagnostic interventions are
inference-only functional audits.

The GPU device node became temporarily unavailable to new processes after the
formal runs; diagnostics were therefore completed on CPU after confirming the
same checkpoint-loading and finite-output paths. This affected execution mode,
not data, checkpoints, or protocol. The final figure previews had no static
source FAIL findings and passed panel-alignment, PDF glyph-size, and rendered-
collision audits. Static preflight retains three preview-oriented warnings:
300-dpi PNG rather than a 600-dpi TIFF submission raster, and a width that
should be revisited for a target journal rather than this preview use.

## J. Frozen files

The following remained unmodified throughout this round:

- `src/models/cosi_mag_final.py`
- `configs/model/cosi_mag_final.yaml`

No commit or push was performed.
