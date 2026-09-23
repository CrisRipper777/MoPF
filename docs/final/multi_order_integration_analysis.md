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
| Movies | 1.743 ± 0.002 | 1.803 ± 0.016 |
| Toys | 1.818 ± 0.011 | 1.979 ± 0.003 |
| Grocery | 1.898 ± 0.007 | 1.983 ± 0.007 |
| ele-fashion | 1.795 ± 0.042 | 1.719 ± 0.046 |
| Reddit-S | 1.760 ± 0.007 | 1.738 ± 0.003 |

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
| Movies | 0.0439 | 0.2441 | 0.0209 | 0.1156 |
| Toys | 0.0425 | 0.2084 | 0.0079 | 0.0313 |
| Grocery | 0.0545 | 0.3175 | 0.0064 | 0.0440 |
| ele-fashion | 0.1474 | 0.6609 | 0.0124 | 0.0433 |
| Reddit-S | 0.0232 | 0.1324 | 0.0009 | 0.0017 |

The larger interaction-off changes relative to uniform-attention changes show
that the interaction residual can materially alter the final representation
and logits. This is a functional sensitivity result, not a claim that removing
interaction must lower held-out accuracy on every dataset.

## Bounded conclusion

Together with the functional ablation, the analysis supports a real
multi-order computation chain and provides paper-facing evidence for
heterogeneous order contributions and direct interacted-state influence. It
does not justify adding a new interaction module, changing the eta definition,
or claiming that one order or one modality is universally dominant.
