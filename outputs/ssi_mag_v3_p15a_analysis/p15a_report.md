# SSI-MAG-V3 P1.5a Corrected Diagnostic Validation

This independent post-hoc analyzer reads only the existing P1 Full best checkpoints. It does not train, run ablations, run LP, read test metrics, select checkpoints, tune hyperparameters, or modify the SSI-MAG-V3 model and the historical P1.5 artifacts.

## Integrity

- Checkpoints loaded: 15/15
- Finite analysis runs: 15/15
- Node ordering verified for cross-seed correlations: True
- Attention shape convention: `[N, query-hop, key-hop]`; observed shapes: `['[15894, 4, 4]', '[16672, 4, 4]', '[17074, 4, 4]', '[20695, 4, 4]', '[97766, 4, 4]']`

## Corrected attention definitions

For each node i and query hop q, normalized entropy is `H(A_iq)/log(K+1)`. Non-uniformity is `1 - mean_iq entropy`. Query-specific diversity is the within-node mean over unordered q<q' pairs of `TV(A_iq,A_iq') = 0.5 * sum_h |A_iqh-A_iq'h|`. Node heterogeneity is `TV(A_iq,Abar_q)` where `Abar_q=mean_i A_iq`. Diagonal mass is `mean_i mean_q A_iqq`; diagonal excess subtracts the uniform value `1/(K+1)`. The old mean-matrix-first entropy and row L1 are retained only as descriptive fields named `old_mean_matrix_*` and `mean_matrix_row_diversity`.

Across run/modality rows, corrected nodewise normalized entropy is 0.521270, while entropy of the mean matrix is 0.888022; mean Jensen gap is 0.366752. Mean node heterogeneity is 0.392988, query diversity is 0.074505, mean-matrix row diversity is 0.015713, and diagonal excess is -0.003526.

Interpretation boundary: these metrics distinguish global, node-dependent, and query-dependent structure, but they are descriptive diagnostics rather than causal necessity evidence. A high old mean-matrix entropy can overestimate nodewise entropy by the Jensen gap; the corrected report does not use off-diagonal mass greater than 0.75 as a diagonal-collapse claim.

## R1 - Relation modulation and operator perturbation

Mean beta is 0.052172; relation-weight mean/CV are 1.004724/0.028403; relation residual abs-mean is 0.645218; local adaptation c mean is 0.033548.
After the same gcn_norm and edge-pair alignment against raw unit physical topology, operator MAE/RMSE/relative-L1 are 0.00159882/0.00290406/0.0147198. These are normalized-operator perturbations, not raw scorer differences.

## Semantic-reference correction

The adaptive-range ratio is corrected to `(q90(alpha)-q10(alpha))/(abs(mean(alpha))+eps)`. The CSV retains alpha standard deviation, IQR, q90-q10, and the corrected ratio. Raw p is not treated as a pathology by its sign alone; the functional branch `term_p=rho_p*p` is reported with its distribution and p saturation fractions.

Mean term_p cross-seed Pearson/Spearman correlations are 0.508899/0.538857; mean alpha cross-seed Pearson/Spearman correlations are 0.582662/0.601793. Mean p saturation fractions are |p|>.90=0.331303, |p|>.95=0.230768, |p|>.99=0.091342.

Cross-seed correlations are computed only after matching dataset node count, feature bytes, edge-index bytes, and labels across seeds. Sign balance is reported descriptively; term_p is the quantity used for functional stability.

## Context-change and interaction direction

Mean context-change injection ratio is 0.162188, with cosine to LayerNorm(S_k) -0.028172. Mean interaction injection ratio is 0.173903, with cosine to S_k 0.051883; cosine(S_tilde_k,S_k) is 0.984096. D norm and gate values are in `p15a_context_injection.csv`.

## Signed filtering and automatic flags

Mean eta negative fraction is 0.243493; mean gate delta/int are 0.164293/0.120487; mean D norm is 14.987875. Filter abs-means for delta_content/reference/relation are 0.031006/0.039799/0.003629; eta std mean is 0.005769. Effective-order summaries are stored per hop row with node-level quantiles. Automatic flags are machine-generated from fixed diagnostic conditions and are not model verdicts.
### Flags

- R1_beta_near_init:Grocery/42/text
- R1_beta_near_init:Grocery/42/visual
- R1_beta_near_init:Grocery/43/text
- R1_beta_near_init:Grocery/43/visual
- R1_beta_near_init:Grocery/44/text
- R1_beta_near_init:Grocery/44/visual
- R1_beta_near_init:Movies/42/text
- R1_beta_near_init:Movies/42/visual
- R1_beta_near_init:Movies/43/text
- R1_beta_near_init:Movies/43/visual
- R1_beta_near_init:Movies/44/text
- R1_beta_near_init:Movies/44/visual
- R1_beta_near_init:Reddit-S/42/text
- R1_beta_near_init:Reddit-S/42/visual
- R1_beta_near_init:Reddit-S/43/text
- R1_beta_near_init:Reddit-S/43/visual
- R1_beta_near_init:Reddit-S/44/text
- R1_beta_near_init:Reddit-S/44/visual
- R1_beta_near_init:Toys/42/text
- R1_beta_near_init:Toys/42/visual
- R1_beta_near_init:Toys/43/text
- R1_beta_near_init:Toys/43/visual
- R1_beta_near_init:Toys/44/text
- R1_beta_near_init:Toys/44/visual
- R1_beta_near_init:ele-fashion/42/text
- R1_beta_near_init:ele-fashion/42/visual
- R1_beta_near_init:ele-fashion/43/text
- R1_beta_near_init:ele-fashion/43/visual
- R1_beta_near_init:ele-fashion/44/text
- R1_beta_near_init:ele-fashion/44/visual
- R2_p_saturation:Movies/42/text/hop1
- R2_p_saturation:Movies/42/text/hop2
- R2_p_saturation:Movies/42/text/hop3
- R2_p_saturation:Movies/44/text/hop1
- R2_p_saturation:Movies/44/text/hop2
- R2_p_saturation:Movies/44/text/hop3
- R3_relation_residual_negligible:Grocery/42/text/hop2
- R3_relation_residual_negligible:Grocery/42/visual/hop0
- R3_relation_residual_negligible:Grocery/42/visual/hop2
- R3_relation_residual_negligible:Grocery/43/text/hop2
- R3_relation_residual_negligible:Grocery/43/visual/hop2
- R3_relation_residual_negligible:Grocery/44/text/hop2
- R3_relation_residual_negligible:Grocery/44/visual/hop1
- R3_relation_residual_negligible:Grocery/44/visual/hop2
- R3_relation_residual_negligible:Movies/42/text/hop2
- R3_relation_residual_negligible:Movies/42/visual/hop2
- R3_relation_residual_negligible:Movies/43/text/hop2
- R3_relation_residual_negligible:Movies/43/visual/hop1
- R3_relation_residual_negligible:Movies/43/visual/hop2
- R3_relation_residual_negligible:Movies/44/text/hop1
- R3_relation_residual_negligible:Movies/44/text/hop2
- R3_relation_residual_negligible:Movies/44/visual/hop1
- R3_relation_residual_negligible:Movies/44/visual/hop2
- R3_relation_residual_negligible:Reddit-S/42/text/hop0
- R3_relation_residual_negligible:Reddit-S/42/text/hop2
- R3_relation_residual_negligible:Reddit-S/42/visual/hop2
- R3_relation_residual_negligible:Reddit-S/43/text/hop2
- R3_relation_residual_negligible:Reddit-S/43/visual/hop2
- R3_relation_residual_negligible:Reddit-S/44/text/hop2
- R3_relation_residual_negligible:Reddit-S/44/visual/hop1
- R3_relation_residual_negligible:Reddit-S/44/visual/hop2
- R3_relation_residual_negligible:Toys/42/text/hop2
- R3_relation_residual_negligible:Toys/42/visual/hop2
- R3_relation_residual_negligible:Toys/43/text/hop2
- R3_relation_residual_negligible:Toys/43/visual/hop2
- R3_relation_residual_negligible:Toys/44/text/hop2
- R3_relation_residual_negligible:Toys/44/visual/hop2
- R3_relation_residual_negligible:ele-fashion/42/text/hop2
- R3_relation_residual_negligible:ele-fashion/42/visual/hop1
- R3_relation_residual_negligible:ele-fashion/42/visual/hop2
- R3_relation_residual_negligible:ele-fashion/43/text/hop1
- R3_relation_residual_negligible:ele-fashion/43/text/hop2
- R3_relation_residual_negligible:ele-fashion/43/visual/hop2
- R3_relation_residual_negligible:ele-fashion/44/text/hop1
- R3_relation_residual_negligible:ele-fashion/44/text/hop2

## Corrected versus unchanged P1.5 conclusions

Corrected: attention entropy, non-uniformity, query diversity, node heterogeneity, diagonal mass, and Jensen gap are now calculated before node aggregation, so the old mean-matrix-first entropy/row-diversity values are not used as node-level evidence. Adaptive-range ratio now has the specified alpha denominator. Functional semantic-reference evidence is based on term_p and cross-seed correlations, not raw p sign alone.

Unchanged: the underlying frozen model, checkpoints, relation/operator quantities, alpha/d/p tensors, context injection tensors, signed eta tensors, and all historical P1.5 output files were not modified. No corrected diagnostic is presented as a validated architecture improvement or causal ablation result.

## Boundary

No training, formal benchmark rerun, ablation, LP experiment, hyperparameter search, test-based decision, loss change, or architecture change was performed.
