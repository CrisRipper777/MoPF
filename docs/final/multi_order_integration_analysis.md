# Multi-order integration analysis for canonical MGSC-MAG P2

This inference-only analysis uses the seed-42 full P2 best checkpoint on all
five NC datasets. The source tables are under
`outputs/final/multi_order_integration_analysis/`.

## Effective-order heterogeneity

Because `eta` can be signed, the paper-facing normalized contribution is

\[
q_{i,k}=|\eta_{i,k}|/\sum_j|\eta_{i,j}|,
\qquad
\text{effective order}_i=\sum_k kq_{i,k}.
\]

This is a descriptive magnitude normalization, not a probability model for
signed coefficients. The exported table includes the complete order-wise
quantiles and the effective-order summary.

| Dataset | Text mean effective order ± SD | Visual mean effective order ± SD |
|---|---:|---:|
| Movies | 1.741 ± 0.002 | 1.802 ± 0.015 |
| Toys | 1.818 ± 0.012 | 1.975 ± 0.003 |
| Grocery | 1.940 ± 0.010 | 1.977 ± 0.012 |
| ele-fashion | 1.778 ± 0.014 | 1.738 ± 0.029 |
| Reddit-S | 1.759 ± 0.007 | 1.738 ± 0.003 |

These summaries show different modality/dataset order profiles. They are not
evidence that a single propagation depth is optimal for every node.

## Cross-order interaction pattern

`average_attention_matrix.csv` contains the average query-order × key-order
attention matrix for every dataset and modality. The preview uses Movies/text;
the complete matrices for all five datasets are the authoritative source data.
The matrices are computed from the actual P2 interaction path with retained
legacy relation-order bias.

## Functional intervention

The normal representation is compared with:

- `query_collapse`: replace query-specific attention rows by their key-mass
  average;
- `uniform_attention`: use a uniform order-attention matrix;
- `interaction_off`: remove the interaction residual, so interacted states
  equal the original state bank.

| Dataset | Uniform logit MAE | Interaction-off logit MAE | Uniform flip | Interaction-off flip |
|---|---:|---:|---:|---:|
| Movies | 0.0481 | 0.2682 | 0.0206 | 0.1170 |
| Toys | 0.0476 | 0.2303 | 0.0078 | 0.0336 |
| Grocery | 0.0727 | 0.3831 | 0.0067 | 0.0412 |
| ele-fashion | 0.0457 | 0.3719 | 0.0039 | 0.0336 |
| Reddit-S | 0.0231 | 0.1327 | 0.0010 | 0.0020 |

The larger interaction-off changes relative to uniform-attention changes show
that the interaction residual can materially alter the final representation
and logits. This is a functional sensitivity result, not a claim that removing
interaction must lower held-out accuracy on every dataset.

## Bounded conclusion

Together with the functional ablation, the analysis supports a real
multi-order computation chain and provides paper-facing evidence for
dataset/modality-dependent order profiles and direct interacted-state
influence. It does not justify claiming strong universal node-level order
heterogeneity, adding a new interaction module, changing the eta definition,
or claiming that one order or one modality is universally dominant.
