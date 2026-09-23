# Figure 4 claim audit

## Authoritative data

The current Figure 4 data were regenerated from the corrected Full
checkpoints at seed 42:

- `outputs/final/multi_order_integration_analysis/effective_order_contribution.csv`
- `outputs/final/multi_order_integration_analysis/average_attention_matrix.csv`
- `outputs/final/multi_order_integration_analysis/integration_intervention.csv`

Order contribution is explicitly normalized as

\[
q_{i,k}=|\eta_{i,k}|/\sum_j|\eta_{i,j}|,
\qquad
\bar{k}_i=\sum_k kq_{i,k}.
\]

This is a descriptive magnitude normalization; it is not a probability model
for signed coefficients and does not claim causal order selection.

## Panel (a): effective order

| Dataset | Text mean ± SD | Visual mean ± SD |
|---|---:|---:|
| Movies | 1.741 ± 0.002 | 1.802 ± 0.015 |
| Toys | 1.818 ± 0.012 | 1.975 ± 0.003 |
| Grocery | 1.940 ± 0.010 | 1.977 ± 0.012 |
| ele-fashion | 1.778 ± 0.014 | 1.738 ± 0.029 |
| Reddit-S | 1.759 ± 0.007 | 1.738 ± 0.003 |

The distributions are concentrated, with order 2 carrying the largest mean
contribution in every dataset/modality. The strongest visible variation is a
dataset/modality profile difference, especially between text and visual on
Movies, Toys, and ele-fashion. Node-level effective-order SD is modest rather
than broad or multimodal. Therefore the defensible panel claim is
“modality- and dataset-dependent order profiles,” not “strong universal
node-level order heterogeneity.”

## Panel (b): cross-order attention

The average matrices are not identity matrices, and their rows are not
identical. However, the query-collapse intervention has very small effects:
logit MAE ranges from 0.00135 to 0.00642 and prediction flips are below
0.13% across the five datasets. This indicates that the average interaction
pattern is relatively stable under query collapse for this checkpoint, so the
figure should not be described as proving rich node-specific attention
patterns.

The safer wording is “learned cross-order interaction pattern” or
“order-referenced interaction matrix.” The matrix is a faithful diagnostic of
the implemented attention path, not evidence that every node uses a distinct
attention policy.

## Panel (c): functional intervention

Interaction-Off produces substantially larger changes than Uniform-Attention
on every dataset:

| Dataset | Uniform logit MAE | Interaction-Off logit MAE | Off flip rate |
|---|---:|---:|---:|
| Movies | 0.0481 | 0.2682 | 11.70% |
| Toys | 0.0476 | 0.2303 | 3.36% |
| Grocery | 0.0727 | 0.3831 | 4.12% |
| ele-fashion | 0.0457 | 0.3719 | 3.36% |
| Reddit-S | 0.0231 | 0.1327 | 0.20% |

This supports the claim that the cross-order interaction path has direct
functional influence on the final representation and logits. It does not
establish a universal test-accuracy improvement or causal identification.

## Recommended Figure 4 wording

Use a title such as **“Modality-specific order profiles and functional
cross-order interaction”**. Avoid “reliable order,” “optimal order,”
“causal interaction,” and “strong node-level heterogeneity.” Figure 4 is a
mechanism diagnostic, not a replacement for the paired ablation matrix.
