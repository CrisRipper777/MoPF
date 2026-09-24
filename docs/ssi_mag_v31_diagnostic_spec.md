# SSI-MAG-V3.1 P1.7b Diagnostic Specification

This document defines descriptive post-hoc diagnostics for future Full V3.1
checkpoints. The analyzer loads a best validation-selected NC checkpoint and
its graph/features in evaluation/no-grad mode. It does not train, select a
checkpoint, read test metrics for a decision, run LP, or run ablations.

Every metric is a diagnostic, not an optimization target. In particular:

- larger operator perturbation is not inherently better;
- larger alpha variance is not inherently better;
- lower attention entropy is not inherently better;
- a larger negative-eta fraction is not inherently better;
- a larger relation residual is not inherently better.

## R1: semantic relation modulation

| Diagnostic question | Metric definition | Valid interpretation | Invalid interpretation |
| --- | --- | --- | --- |
| Does semantic compatibility vary on physical relations? | `s_ij=cos(z_i,z_j)` over original non-self physical edges; mean/std/q10/q50/q90 | Describes relation-space scale and heterogeneity | Does not establish usefulness |
| Is compatibility relative to incident neighborhoods? | `rrel_ij=s_ij-0.5(mu_i+mu_j)`; same quantiles | Describes the centered relation signal | Does not imply centered values should be large |
| Do modalities encode similar relations? | Pearson/Spearman and absolute-difference quantiles on aligned non-self edge support | Describes Text–Visual relation agreement/discrepancy | Does not prove cross-modal interaction |
| Is the scorer saturated? | `a_ij=softsign(w^T u+b)`; quantiles and fractions `|a|>.8/.9` | Flags possible collapse/saturation | A high score is not a quality target |
| Does beta depart from initialization? | `beta=sigmoid(theta_beta)` | Describes learned scalar modulation strength | Near-init beta is not automatically failure |
| Does propagation topology change? | gcn-normalized operator vs raw unit topology, aligned by edge-pair key; MAE/RMSE/relative-L1 | Measures actual normalized-operator perturbation | Larger perturbation is not automatically better |
| Does relation state vary over nodes? | `c_i=mean incident |beta*a|`, with isolated-zero check | Describes node-local adaptation scale | Small amplitude is reported as `relation_residual_small_amplitude`, not called negligible |

Self-loops are excluded from semantic statistics. Explicit input self-loops
have `a_ii=0` and `w_ii=1`; gcn_norm-added self-loops retain the existing
identity policy.

## R2: stabilized semantic reference

| Diagnostic question | Metric definition | Valid interpretation | Invalid interpretation |
| --- | --- | --- | --- |
| Is the stabilized prior numerically scaled? | `p=(w_p^T LN(H0))/(||w_p||_2+eps)`; mean/std/q10/q50/q90 and fractions `|p|>3/>5` | Flags scale explosion or collapse | Does not use sign or tail as a quality score |
| Is the prior functional? | `term_p=rho_p*p`; distribution and cross-seed Pearson/Spearman after ordering verification | Measures the actual prior contribution | Raw p alone is not the functional branch |
| How does structure differ from H0? | `d_k=1-cos(Q_k,H0)` quantiles | Describes hop-wise change | Larger d is not inherently better |
| Is alpha adaptive or saturated? | mean/std/IQR/q10/q50/q90, `(q90-q10)/(abs(mean(alpha))+eps)`, `<.05`/`>.95` fractions | Flags concentration, dispersion, or saturation | Larger dispersion is not inherently better |
| What was learned? | `b_k`, `rho_p`, `rho_d` | Reports learned coefficients | Coefficient magnitude alone is not a performance claim |

## Cross-seed consistency

Before node-wise correlations, the analyzer verifies node count, feature byte
hash, edge-index byte hash, and label byte hash within each dataset. If any
signature differs, node-wise term-p or alpha correlations are not computed.

## Stage-II attention and context change

Attention has shape `[N, query-hop, key-hop]`. The analyzer imports the
corrected P1.5a `attention_diagnostics()` implementation. It computes
per-node/query entropy before aggregation, entropy of the mean matrix, Jensen
gap, node heterogeneity TV, query diversity TV, diagonal mass/excess, mean
attention matrix, and entrywise node standard deviation.

For `TV(p,q)`, the definition is `0.5 * sum_h |p_h-q_h|`.
Node heterogeneity is the mean TV between `A_iq` and `mean_i A_iq` for the
same query q. Query diversity is the within-node mean TV across unordered
query pairs.

For each `k>=1`, context diagnostics report:

- raw change `S_k-S_(k-1)` norm, relative norm, and cosine;
- `C_delta=g_delta W_delta D_k` norm, ratio to `||LN(S_k)||`, and cosine to `LN(S_k)`;
- `C_int=g_int O_k` norm, ratio to `||S_k||`, cosine to `S_k`, and cosine between `S_tilde_k` and `S_k`.

These metrics describe direction and magnitude. Entropy, injection ratio, and
diagonal mass are not ranked as universally desirable.

## Signed filtering

For every hop, the analyzer reports distributions of `gamma`, `DeltaGamma`,
`delta_content`, relation residual, and `eta`, plus negative-eta fraction.
Effective order is the absolute-eta weighted order.

For node-variance attribution, with `v=Var(eta)`:

```text
contribution_content = Cov(delta_content, eta) / v
contribution_relation = Cov(relation_residual, eta) / v
```

Static gamma and DeltaGamma do not enter the node-variance attribution. If
`v` is near zero, the analyzer marks the row `eta_variance_near_zero` and does
not interpret the ratio. Small relation amplitude is reported with the fixed
flag `relation_residual_small_amplitude`; it is not relabeled as negligible.
