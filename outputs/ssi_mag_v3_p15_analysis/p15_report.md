# SSI-MAG-V3 P1.5 Mechanism Attribution Audit

This is a post-hoc diagnosis over the existing 15 P1 Full best checkpoints (5 NC datasets × seeds 42/43/44). No training, ablation, LP execution, hyperparameter search, or test-based model selection was performed. The model forward and training behavior were not modified.

## Checkpoint integrity

- Loaded checkpoints: 15/15
- All analysis tensors finite: 15/15
- Analysis API: `SSIMAGV3.analysis()`
- Test metrics: not read or used

## R1 — Relation semantic discrepancy

Across run-level edge summaries, relation-space Text/Visual semantic compatibility has mean absolute discrepancy 0.282711 and mean Spearman correlation 0.1628. The learned Text/Visual normalized operators differ by relative-L1 0.018206. This is the explicit cosine in the learned relation-projection space; raw H0 cosine is reported separately in `p15_relation_attribution.csv`.
The learned residual has mean Spearman correlation 0.4065 with absolute relation compatibility and 0.2490 with neighborhood-relative compatibility. The mean fraction with |residual|>0.95 is 0.2573, mean beta is 0.052172, and learned-vs-unit normalized-operator relative-L1 is 0.014720.
Diagnosis: if semantic discrepancy is non-negligible while residual alignment is weak and residual saturation is high, the evidence points to a parameterization/utilization problem rather than a failure of the motivation. Small beta and small operator perturbation should not be described as strong structure modulation.

## R2 — Adaptive semantic reference attribution

The mean |p|>0.95 fraction is 0.2308; mean alpha node-wise standard deviation is 0.002254. Mean covariance contributions to adaptive-logit variance are p=0.8460, d=0.1480, c=0.0061; four-term contribution-sum error averages 1.8744e-09. Mean effective-order Spearman correlations are alpha=-0.1813, d=0.1348, c=-0.2878, p=0.0506. Contributions are signed and are computed as Cov(term,total)/Var(total).
The p mean sign is stable on 1/5 datasets for Text and 1/5 for Visual; median-sign stability is 1/5 and 1/5. Near-zero variances are explicitly marked in the CSV. A saturated or seed-unstable p branch should not support a strong node-adaptive semantic-prior claim.

## Stage II — Signed filtering attribution

Mean absolute-term/static-prior ratios are content=0.3985, reference=0.4505, relation=0.0400; mean eta variance is 0.000098. Signed eta covariance contributions are content=0.7272, reference=-0.0059, relation=0.2787; contribution-sum error=5.78781e-08. The full node-variance contributions are in `p15_filter_attribution.csv`.
A signed eta pattern should be described as node-adaptive only when its node variance and branch attribution are materially nonzero and stable across seeds. The report does not collapse signed contributions into magnitudes.

## Context change and cross-hop interaction injection

Mean raw state change norm is 2.225751, relative raw change is 0.222846, and cosine(S_k,S_(k-1)) is 0.970401. Mean context-change injection ratio is 0.162188; cross-hop interaction injection ratio is 0.173903. LayerNorm D norms are not used as mechanism evidence. Attention mean normalized entropy is 0.8880, diagonal mass is 0.2465, and mean per-entry seed std is 0.027374; full matrices are in `p15_attention_matrices.json`.

## Cross-seed consistency

`p15_consistency.csv` reports dataset-level mean/std/CV, sign stability, and near-constant flags for beta, operator perturbation, scorer saturation, semantic discrepancy, alpha/p quantities, gate/injection quantities, and effective order. No seed was removed.

## Evidence-based V3.1 candidates

### Candidate R1 — Relative Semantic Relation Modulation

Verdict: supported as a V3.1 motivation/parameterization candidate, not as an implemented P1.5 change. Relation-space discrepancy is large while the learned-vs-unit operator perturbation is small; the current residual has only partial alignment with semantic and relative compatibility. An explicit relative compatibility term with a raw-topology fallback is therefore justified for a later controlled revision.

### Candidate R2 — Structure-State Adaptive Semantic Reference

Verdict: supported for revision diagnosis. p contributes most adaptive-logit variance, alpha node spread is small, and p mean/median signs are unstable across seeds. A later revision should replace or constrain the saturated/seed-sensitive p branch and test whether d/local relation context can carry the intended node variation.

### Candidate Stage II — Semantic-Query Signed Filtering

Verdict: a plausible Stage-II validation candidate, not yet a demonstrated utility mechanism. Context-change and interaction injections are non-negligible, attention is non-uniform, and the three-seed matrix spread is reported explicitly; a later semantic-query signed filter should be tested only with a matched validation protocol and without changing the P1.5 diagnosis.

## Boundary

This artifact is diagnosis only. No architecture revision, auxiliary loss, retraining, ablation, LP run, or test-based decision was performed.
