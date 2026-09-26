# Conditional Relational Utility Discovery — Experiment 3

## Scope and protocol

- NC only: Movies, Toys, Grocery, ele-fashion, Reddit-S; seeds 42/43/44.
- C0 Uniform: 15 formal runs; best-validation checkpoints: outputs/cmrf_discovery_v1/reference_uniform/checkpoints/. C1/C2/C3: 45 new formal runs. No LP runs.
- All checkpoints selected by Validation Accuracy under unified_full_graph_nc_v1.
- Test evaluation and test labels were excluded from training, model selection, E3-A/E3-B/E3-C analysis, and conclusions.
- Physical operator uses unit edge weights after removing loops, symmetrizing, adding one loop per node, and symmetric normalization.
- Structural coefficients preserve H0 at exactly 1.0; lambda is fixed at 0.25.

## Controller results

| Variant | Mean Val Acc | Population SD | Mean Val Macro-F1 | Population SD | Mean best epoch |
|---|---:|---:|---:|---:|---:|
| uniform | 0.8107 | 0.1286 | 0.7388 | 0.1380 | 113.4 |
| own_soft | 0.8081 | 0.1354 | 0.7361 | 0.1486 | 93.1 |
| cross_soft | 0.8082 | 0.1349 | 0.7340 | 0.1491 | 101.8 |
| gated_cross_soft | 0.8080 | 0.1336 | 0.7336 | 0.1488 | 96.9 |

### Paired main comparisons

| Comparison | Mean Val Acc Δ (pp) | SD (pp) | Positive seed pairs | Mean Macro-F1 Δ (pp) |
|---|---:|---:|---:|---:|
| OwnSoft - Uniform | -0.254 | 0.777 | 8/15 | -0.275 |
| CrossSoft - OwnSoft | 0.005 | 0.282 | 8/15 | -0.207 |
| GatedCross - OwnSoft | -0.010 | 0.415 | 7/15 | -0.244 |
| GatedCross - CrossSoft | -0.016 | 0.253 | 8/15 | -0.038 |

## E3-A: conditional utility

- 30 dataset/seed/modality summaries and 30 paired utility-predictor runs.
- Mean node utility SD: 0.6050; mean positive-utility fraction: 0.447; mean companion-conditioned sign-flip fraction: 0.064.
- Own validation MSE mean: 0.86648; cross-conditioned MSE mean: 0.86752; mean paired MSE change Cross-Own: 0.00105.

## E3-B: cross-order coupling

- Completed 15 dataset×seed matrices (16 order pairs each). Mean unique conditional Text orders: 1.40; Visual orders: 1.67; mean absolute CE interaction residual: 0.06088.

## E3-C: intervention and controller analysis

- Shuffle rows: 600; cross-zero rows: 60; shuffle-mean Accuracy drop: 0.038 pp; Macro-F1 drop: 0.044 pp; prediction flip rate: 0.0021; mean targeted coefficient change: 0.01495.
- C3 node gate summaries: mean rho=0.7752, mean between-node SD=0.2315; raw node tables contain coefficients, effective relational mass, response contribution ratios, and utility associations.

## Evidence decisions

| Hypothesis | Decision | Datasets supported | Effect size | Counterevidence |
|---|---|---|---|---|
| H1 node/modality structural utility heterogeneity | **STRONG_SUPPORT** | Grocery;Movies;Reddit-S;Toys;ele-fashion | mean node-level utility SD=0.6050 | 0 datasets did not replicate both utility signs in >=2 seeds |
| H2 utility predictable from own relational state | **NO_SUPPORT** | Grocery | median validation MSE reduction vs train-mean=-4.213% | 20 runs failed at least one MSE/Spearman/AUROC criterion |
| H3 companion modality improves utility predictability | **NO_SUPPORT** | Reddit-S | median relative validation MSE reduction Cross-Own=-0.063% | 22 paired runs did not improve both MSE and Spearman |
| H4 cross-order preference coupling | **NO_SUPPORT** | Reddit-S | mean absolute CE interaction residual=0.0609 | 10 matrices did not show multi-order conditional optima on both axes and residual>0.001 |
| H5 own-conditioned soft modulation improves Uniform | **MIXED_SUPPORT** | Reddit-S;ele-fashion | mean Val Acc=-0.254 pp; Macro-F1=-0.275 pp | 7 accuracy seed pairs were non-positive |
| H6 cross-conditioned modulation improves own-only | **MIXED_SUPPORT** | none | paired CrossSoft-OwnSoft and GatedCross-OwnSoft comparisons; see e3c_controller_results.csv | requires H3 predictive gain and at least one cross variant with replicated Val Acc/F1 gain |
| H7 cross-modal controller functionally used under shuffle | **MIXED_SUPPORT** | Grocery | mean Val Acc drop=0.038 pp; mean |coefficient change|=0.01495 | 53 direction-level runs lacked both >0.1pp Acc drop and >0.001 coefficient change |

### Interpretation: neither

Cross-modal support requires replicated cross-over-own utility prediction, replicated cross-conditioned validation gains, and a measurable validation decrease under deterministic companion shuffling. Coefficient changes alone do not establish useful control.

Decision thresholds are descriptive consistency rules across datasets/seeds, not significance tests. Order preferences are summarized over all 15 matrices; no single matrix is treated as a general rule.

## Artifacts

- preflight_report.md
- e3a_conditional_utility.csv
- e3a_utility_predictability.csv
- e3b_cross_order_matrix.csv
- e3b_summary.json
- e3c_controller_results.csv
- e3c_per_run_validation.csv
- e3c_crossmodal_intervention.csv
- e3c_controller_analysis.csv
- e3_decision_matrix.csv
- Raw node-level tables are compressed under outputs/cmrf_discovery_v1/; results CSVs record their row counts and SHA-256 hashes.

No final paper model was designed in this phase.

## Next-stage recommendation

Do not expand the controller/router family from these results. H1 is strong, but own-state utility prediction, companion-state gain, and cross-dataset controller benefit are not supported. First audit the counterfactual utility target's stability and the representation probe's out-of-sample calibration using Train/Validation only; revisit model design only if that signal replicates.
