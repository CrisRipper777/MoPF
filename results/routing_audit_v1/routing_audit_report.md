# Experiment 4 — Adaptive Structural Routing Sufficiency Audit

- Branch/base: routing_audit, starting commit cf938f9fb6cde3b5ba1edd1e2ff5f944e228b15f.
- NC only; all findings use Train + Validation; task.evaluate_test=false.
- Fixed physical topology and E3 C0 unit-weight symmetric normalized operator.
- Phase C: OWN_RELATION_CONTEXT_NOT_SUPPORTED.
- Screening winner: A1_global_simplex.
- Successful formal runs: 66; attempts: 75; unresolved failures: 0; retry successes: 9; nonformal debug smokes excluded: 1.

## A0 utility stability

E3 raw tables were read directly; see a0_utility_stability_report.md and CSV.

## Phase A

| Comparison | Dataset | Mean Δ Val Acc | Mean Δ Macro-F1 | Positive seed pairs |
|---|---|---:|---:|---:|
| A1_global_simplex vs A0_uniform | Movies | -0.0019 | -0.0054 | 0/3 |
| A1_global_simplex vs A0_uniform | Reddit-S | 0.0008 | 0.0004 | 3/3 |
| A1_global_simplex vs A0_uniform | ele-fashion | 0.0019 | 0.0102 | 3/3 |
| A1_global_simplex vs A0_uniform | ALL_SCREEN | 0.0003 | 0.0017 | 6/9 |
| A2_node_flat_simplex vs A0_uniform | Movies | -0.0348 | -0.0828 | 0/3 |
| A2_node_flat_simplex vs A0_uniform | Reddit-S | 0.0040 | 0.0049 | 3/3 |
| A2_node_flat_simplex vs A0_uniform | ele-fashion | 0.0042 | 0.0107 | 3/3 |
| A2_node_flat_simplex vs A0_uniform | ALL_SCREEN | -0.0089 | -0.0224 | 6/9 |
| A3_hierarchical_flat_simplex vs A0_uniform | Movies | -0.0355 | -0.1105 | 0/3 |
| A3_hierarchical_flat_simplex vs A0_uniform | Reddit-S | 0.0043 | 0.0060 | 3/3 |
| A3_hierarchical_flat_simplex vs A0_uniform | ele-fashion | 0.0039 | 0.0102 | 3/3 |
| A3_hierarchical_flat_simplex vs A0_uniform | ALL_SCREEN | -0.0091 | -0.0314 | 6/9 |
| A3_hierarchical_flat_simplex vs A1_global_simplex | Movies | -0.0336 | -0.1050 | 0/3 |
| A3_hierarchical_flat_simplex vs A1_global_simplex | Reddit-S | 0.0035 | 0.0056 | 3/3 |
| A3_hierarchical_flat_simplex vs A1_global_simplex | ele-fashion | 0.0020 | 0.0000 | 2/3 |
| A3_hierarchical_flat_simplex vs A1_global_simplex | ALL_SCREEN | -0.0094 | -0.0331 | 5/9 |
| A3_hierarchical_flat_simplex vs A2_node_flat_simplex | Movies | -0.0007 | -0.0277 | 1/3 |
| A3_hierarchical_flat_simplex vs A2_node_flat_simplex | Reddit-S | 0.0003 | 0.0011 | 2/3 |
| A3_hierarchical_flat_simplex vs A2_node_flat_simplex | ele-fashion | -0.0002 | -0.0005 | 1/3 |
| A3_hierarchical_flat_simplex vs A2_node_flat_simplex | ALL_SCREEN | -0.0002 | -0.0090 | 4/9 |

Node/global frozen interventions are in phase_a_intervention.csv. An A3 gain without a functional residual intervention is interpreted as global calibration.

## Phase B

| Comparison | Dataset | Mean Δ Val Acc | Mean Δ Macro-F1 | Positive seed pairs |
|---|---|---:|---:|---:|
| B1_trajectory vs B0_hierarchical_flat | Movies | 0.0022 | 0.0255 | 2/3 |
| B1_trajectory vs B0_hierarchical_flat | Reddit-S | -0.0006 | -0.0025 | 1/3 |
| B1_trajectory vs B0_hierarchical_flat | ele-fashion | -0.0003 | -0.0004 | 2/3 |
| B1_trajectory vs B0_hierarchical_flat | ALL_SCREEN | 0.0004 | 0.0075 | 5/9 |
| B2_capacity_control vs B1_trajectory | Movies | 0.0031 | 0.0101 | 2/3 |
| B2_capacity_control vs B1_trajectory | Reddit-S | -0.0002 | 0.0010 | 2/3 |
| B2_capacity_control vs B1_trajectory | ele-fashion | -0.0009 | 0.0014 | 1/3 |
| B2_capacity_control vs B1_trajectory | ALL_SCREEN | 0.0007 | 0.0042 | 5/9 |
| B2_relation vs B2_capacity_control | Movies | -0.0068 | -0.0269 | 1/3 |
| B2_relation vs B2_capacity_control | Reddit-S | 0.0002 | -0.0007 | 2/3 |
| B2_relation vs B2_capacity_control | ele-fashion | 0.0003 | -0.0050 | 1/3 |
| B2_relation vs B2_capacity_control | ALL_SCREEN | -0.0021 | -0.0109 | 4/9 |
| B2_relation vs B1_trajectory | Movies | -0.0037 | -0.0168 | 1/3 |
| B2_relation vs B1_trajectory | Reddit-S | -0.0000 | 0.0003 | 1/3 |
| B2_relation vs B1_trajectory | ele-fashion | -0.0005 | -0.0035 | 2/3 |
| B2_relation vs B1_trajectory | ALL_SCREEN | -0.0014 | -0.0067 | 4/9 |

B2 relation mean/shuffle interventions are in phase_b_intervention.csv; parameter counts and paired contrasts are in phase_b_capacity_control.csv.

B2 and its capacity-control parameter balance is recorded as parameter_diff_pct in phase_b_capacity_control.csv.

## Decision matrix

| Hypothesis | Decision | Datasets supported | Paired seeds | Δ Acc | Δ Macro-F1 | Functional intervention | Confirmation (datasets; seeds; Δ Acc; Δ F1) |
|---|---|---:|---:|---:|---:|---|---|
| H1_global_structural_preference | MIXED_SUPPORT | 2 | 6/9 | 0.0003 | 0.0017 | True | 2/5; 7/15; -0.0005; -0.0009 |
| H2_node_routing_beyond_global | MIXED_SUPPORT | 2 | 5/9 | -0.0094 | -0.0331 | True | 2/5; 6/15; -0.0069; -0.0198 |
| H3_trajectory_context_beyond_flat | MIXED_SUPPORT | 1 | 5/9 | 0.0004 | 0.0075 | True | not run for this hypothesis |
| H4_relation_context_beyond_trajectory_capacity | MIXED_SUPPORT | 2 | 4/9 | -0.0021 | -0.0109 | True | not run for this hypothesis |
| H5_relation_context_functionally_used | STRONG_SUPPORT | 3 | 9/9 | -0.0025 | -0.0048 | True | not run for this hypothesis |
| H6_paired_cross_modal_relation_context | NO_SUPPORT | 0 | 0/0 | nan | nan | False | not run for this hypothesis |

## Five-dataset confirmation

New confirmation rows: 12; reused screening rows: 18; Uniform rows reused from E3 C0: 15.

| Variant | Dataset | Mean Val Acc | Mean Macro-F1 | Δ Acc vs Uniform | Δ Macro-F1 vs Uniform |
|---|---|---:|---:|---:|---:|
| global_simplex | Grocery | 0.8303 | 0.7338 | -0.0028 | -0.0121 |
| global_simplex | Movies | 0.5738 | 0.4886 | -0.0019 | -0.0054 |
| global_simplex | Reddit-S | 0.9620 | 0.9215 | 0.0008 | 0.0004 |
| global_simplex | Toys | 0.8070 | 0.7810 | -0.0007 | 0.0022 |
| global_simplex | ele-fashion | 0.8776 | 0.7644 | 0.0019 | 0.0102 |
| hierarchical_flat_simplex | Grocery | 0.8290 | 0.7420 | -0.0041 | -0.0040 |
| hierarchical_flat_simplex | Movies | 0.5402 | 0.3836 | -0.0355 | -0.1105 |
| hierarchical_flat_simplex | Reddit-S | 0.9655 | 0.9271 | 0.0043 | 0.0060 |
| hierarchical_flat_simplex | Toys | 0.8020 | 0.7735 | -0.0056 | -0.0052 |
| hierarchical_flat_simplex | ele-fashion | 0.8797 | 0.7644 | 0.0039 | 0.0102 |
| uniform | Grocery | 0.8331 | 0.7459 | — | — |
| uniform | Movies | 0.5757 | 0.4941 | — | — |
| uniform | Reddit-S | 0.9612 | 0.9211 | — | — |
| uniform | Toys | 0.8077 | 0.7788 | — | — |
| uniform | ele-fashion | 0.8758 | 0.7542 | — | — |

Paired five-dataset aggregate versus Uniform:
- global_simplex: mean Δ Acc=-0.0005, mean Δ Macro-F1=-0.0009, positive dataset means=2/5, positive paired seeds=7/15.
- hierarchical_flat_simplex: mean Δ Acc=-0.0074, mean Δ Macro-F1=-0.0207, positive dataset means=2/5, positive paired seeds=6/15.
- hierarchical_minus_global_simplex: mean Δ Acc=-0.0069, mean Δ Macro-F1=-0.0198, positive dataset means=2/5, positive paired seeds=6/15.
A1_global_simplex passed the predeclared three-dataset screen, but five-dataset confirmation has a negative overall mean and gains on only two dataset means. Treat the screen result as mixed and do not claim a robust cross-dataset improvement.

## Interpretation boundary

The report separates global structural calibration, node adaptation, relation-context-aware adaptation, and paired cross-modal relational conditioning. A skipped Phase C only rejects its predeclared gate, not every possible cross-modal interaction.

No expert bank, signed filter, common/private decomposition, task-aware router, or deeper transformer was added.
