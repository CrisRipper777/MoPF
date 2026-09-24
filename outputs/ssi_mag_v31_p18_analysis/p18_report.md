# SSI-MAG-V3.1 P1.8 R1U Post-hoc Report

This report separates performance evidence, mechanism behavior evidence, and frozen functional sensitivity. U is the only newly trained variant; B and AB are provenance-locked P1.7c controls.

## Integrity

- U expected/loaded/finite: 15/15/15.
- Control runs loaded: 30; attention failures: [].
- Device: cpu; analyzer training invoked: False; LP jobs: 0.
- All model selection remains validation Accuracy; test metrics are descriptive only.

## Performance evidence

See p18_performance.csv for per-run metrics and population-standard-deviation summaries. p18_r1_comparison.csv contains paired U-vs-AB and U-vs-B validation/test deltas.
- Predefined performance trigger: PERFORMANCE_REVIEW_TRIGGER; mean U-vs-B validation Accuracy delta -0.1875 pp, Macro-F1 delta -0.1887 pp; simultaneous dataset declines ['Movies', 'Toys', 'Grocery', 'Reddit-S'].
- This is a review trigger, not a final paper-model decision.

## R1 semantic behavior

U reports semantic compatibility, centered relative compatibility, linear scorer terms, bounded softsign score tails, log-weight modulation, weight quantiles/CV/bounds, and local adaptation c in p18_r1_comparison.csv.

## R1 propagation utilization

Normalized operator MAE/RMSE/relative-L1 are aligned by directed edge-pair multisets against the same raw unit-weight topology; self-loop insertion is not compared by position.

## Stage-II sanity

Corrected alpha range ratio is (q90-q10)/(abs(mean(alpha))+eps). Attention metrics use nodewise entropy, query-row total-variation diversity, node heterogeneity, diagonal/off-diagonal mass, and simplex validation. Reference residuals, signed eta, effective order, and exact interaction-off equality are in p18_stage2_sanity.csv.

## Frozen sensitivity

Relation-off and interaction-off keep the trained model and NC head fixed. They are functional sensitivity diagnostics, not causal necessity claims and not retrained ablations.

## Flags

- Automatic descriptive flags: 10. Flags report quantities below fixed diagnostic thresholds; no repair or tuning was performed.
- No final model choice is made here; the open question is whether R1U direct utilization provides enough validated benefit to justify further human review.

## Prohibited work confirmation

No LP, no additional relation variant, no hyperparameter search, no auxiliary loss, no test-based selection, and no retrained ablation was run in this analysis.
