# Experiment 7 — Counterfactual Structural Action Observability & Selective Routing Audit

Screening winner locked before confirmation: **E6_D0**. Screening fits: 27. Confirmation ranker fits: 0. Test metrics: not evaluated.

## Screening gates

| Gate | Status |
|---|---|
| H1_COUNTERFACTUAL_RANKING_OBSERVABLE | FAIL |
| H2_DECISION_RESPONSE_ADDS_INFORMATION | FAIL |
| H3_COUNTERFACTUAL_RANKER_BEATS_DIRECT_CE | FAIL |
| H4_CONFIDENCE_IS_MEANINGFUL | FAIL |
| H5_SELECTIVE_ROUTING_PROMISING | FAIL |

## Final decision matrix

| Question | Status |
|---|---|
| H1_structural_action_preference_is_stable | STRONG_SUPPORT |
| H2_trajectory_evidence_predicts_action_utility | NO_SUPPORT |
| H3_decision_response_adds_observability | NO_SUPPORT |
| H4_counterfactual_ranking_beats_direct_ce | NO_SUPPORT |
| H5_misrouting_concentrates_on_fragile_nodes | MIXED_SUPPORT |
| H6_ranker_confidence_predicts_routing_reliability | NO_SUPPORT |
| H7_selective_uniform_fallback_is_promising | NO_SUPPORT |
| H8_toys_grocery_contrast_is_explained | MIXED_SUPPORT |
| H9_confirmation_supports_selected_mechanism | MIXED_SUPPORT |

## Corrected E6 restart range

Groups: 234; mean 0.00236967, median 0.000665978, P90 0.00940037, max 0.0177898. Grouping is dataset/seed/split/level/chunk_start, then max-minus-min across three restarts. The historical E6 audit was not edited.

## Screening oracle landscape

| dataset | modality | split | margin | gain | gain_positive | uniform_best |
|---|---|---|---|---|---|---|
| Movies | text | train | 0.1385 | 0.1692 | 0.9360 | 0.0640 |
| Movies | text | validation | 0.1925 | 0.2736 | 0.9480 | 0.0520 |
| Movies | visual | train | 0.1994 | 0.2301 | 0.9267 | 0.0733 |
| Movies | visual | validation | 0.2293 | 0.3780 | 0.9371 | 0.0629 |
| Reddit-S | text | train | 0.0145 | 0.0156 | 0.8639 | 0.1361 |
| Reddit-S | text | validation | 0.0406 | 0.0478 | 0.8733 | 0.1267 |
| Reddit-S | visual | train | 0.0168 | 0.0167 | 0.8845 | 0.1155 |
| Reddit-S | visual | validation | 0.0573 | 0.0639 | 0.8868 | 0.1132 |
| ele-fashion | text | train | 0.0564 | 0.0777 | 0.9748 | 0.0252 |
| ele-fashion | text | validation | 0.0797 | 0.1185 | 0.9749 | 0.0251 |
| ele-fashion | visual | train | 0.0509 | 0.0681 | 0.9669 | 0.0331 |
| ele-fashion | visual | validation | 0.0668 | 0.0991 | 0.9675 | 0.0325 |

## Marginal action cross-seed stability

| dataset | modality | best_action_agreement | margin_spearman | gain_spearman | pairwise_ordering |
|---|---|---|---|---|---|
| Movies | text | 0.6256 | 0.6576 | 0.6381 | 0.7660 |
| Movies | visual | 0.5594 | 0.6373 | 0.6291 | 0.7448 |
| Reddit-S | text | 0.6431 | 0.6079 | 0.6495 | 0.7975 |
| Reddit-S | visual | 0.7342 | 0.7534 | 0.7395 | 0.8192 |
| ele-fashion | text | 0.7979 | 0.9118 | 0.9142 | 0.8797 |
| ele-fashion | visual | 0.7539 | 0.8929 | 0.8930 | 0.8525 |

## E6 D0 projected misrouting by oracle-margin quartile

| dataset | modality | margin_quartile | margin | agreement | regret |
|---|---|---|---|---|---|
| Movies | text | 1.0000 | 0.0042 | 0.2263 | 0.2382 |
| Movies | text | 2.0000 | 0.0313 | 0.3798 | 0.3989 |
| Movies | text | 3.0000 | 0.1064 | 0.5305 | 0.4862 |
| Movies | text | 4.0000 | 0.5201 | 0.8231 | 0.3183 |
| Movies | visual | 1.0000 | 0.0072 | 0.2565 | 0.4199 |
| Movies | visual | 2.0000 | 0.0497 | 0.4192 | 0.6144 |
| Movies | visual | 3.0000 | 0.1556 | 0.5434 | 0.7343 |
| Movies | visual | 4.0000 | 0.6448 | 0.7442 | 0.7113 |
| Reddit-S | text | 1.0000 | 0.0000 | 0.0008 | 0.0577 |
| Reddit-S | text | 2.0000 | 0.0000 | 0.0099 | 0.0782 |
| Reddit-S | text | 3.0000 | 0.0000 | 0.6629 | 0.0565 |
| Reddit-S | text | 4.0000 | 0.1102 | 0.8731 | 0.0092 |
| Reddit-S | visual | 1.0000 | 0.0000 | 0.0058 | 0.1487 |
| Reddit-S | visual | 2.0000 | 0.0000 | 0.1191 | 0.1755 |
| Reddit-S | visual | 3.0000 | 0.0000 | 0.8782 | 0.0332 |
| Reddit-S | visual | 4.0000 | 0.1480 | 0.9239 | 0.0297 |
| ele-fashion | text | 1.0000 | 0.0001 | 0.4671 | 0.0192 |
| ele-fashion | text | 2.0000 | 0.0030 | 0.6456 | 0.0462 |
| ele-fashion | text | 3.0000 | 0.0198 | 0.6612 | 0.1024 |
| ele-fashion | text | 4.0000 | 0.2494 | 0.6993 | 0.3300 |
| ele-fashion | visual | 1.0000 | 0.0001 | 0.4093 | 0.0148 |
| ele-fashion | visual | 2.0000 | 0.0020 | 0.5439 | 0.0350 |
| ele-fashion | visual | 3.0000 | 0.0141 | 0.5702 | 0.0916 |
| ele-fashion | visual | 4.0000 | 0.2191 | 0.6787 | 0.2770 |

## Screening rankers: joint deployment

| candidate | dataset | accuracy | macro_f1 | joint_regret |
|---|---|---|---|---|
| R0_trajectory_ranker | Movies | 0.5757 | 0.4942 | 0.5856 |
| R0_trajectory_ranker | Reddit-S | 0.9612 | 0.9212 | 0.0868 |
| R0_trajectory_ranker | ele-fashion | 0.8745 | 0.7497 | 0.1898 |
| R1_capacity_control | Movies | 0.5759 | 0.4930 | 0.5858 |
| R1_capacity_control | Reddit-S | 0.9612 | 0.9211 | 0.0854 |
| R1_capacity_control | ele-fashion | 0.8717 | 0.7506 | 0.1998 |
| R1_decision_ranker | Movies | 0.5759 | 0.4933 | 0.7810 |
| R1_decision_ranker | Reddit-S | 0.9612 | 0.9211 | 0.0897 |
| R1_decision_ranker | ele-fashion | 0.8761 | 0.7548 | 0.2051 |

## Screening rankers: marginal action selection

| candidate | dataset | oracle_agreement | selection_regret |
|---|---|---|---|
| R0_trajectory_ranker | Movies | 0.0576 | 0.3259 |
| R0_trajectory_ranker | Reddit-S | 0.1295 | 0.0566 |
| R0_trajectory_ranker | ele-fashion | 0.0520 | 0.1102 |
| R1_capacity_control | Movies | 0.0633 | 0.3260 |
| R1_capacity_control | Reddit-S | 0.1200 | 0.0558 |
| R1_capacity_control | ele-fashion | 0.1265 | 0.1152 |
| R1_decision_ranker | Movies | 0.1382 | 0.4236 |
| R1_decision_ranker | Reddit-S | 0.1713 | 0.0580 |
| R1_decision_ranker | ele-fashion | 0.1531 | 0.1176 |

## Fragile versus decisive nodes

| candidate | dataset | q1_regret | q4_regret | fragile_q1_regret_higher |
|---|---|---|---|---|
| E6_D0 | Movies | 0.4179 | 0.8221 | False |
| E6_D0 | Reddit-S | 0.1212 | 0.0186 | True |
| E6_D0 | ele-fashion | 0.0193 | 0.4168 | False |
| R0_trajectory_ranker | Movies | 0.0690 | 0.7958 | False |
| R0_trajectory_ranker | Reddit-S | 0.0054 | 0.1968 | False |
| R0_trajectory_ranker | ele-fashion | 0.0024 | 0.3907 | False |
| R1_capacity_control | Movies | 0.0691 | 0.7956 | False |
| R1_capacity_control | Reddit-S | 0.0058 | 0.1921 | False |
| R1_capacity_control | ele-fashion | 0.0061 | 0.3847 | False |
| R1_decision_ranker | Movies | 0.1160 | 0.9055 | False |
| R1_decision_ranker | Reddit-S | 0.0049 | 0.2047 | False |
| R1_decision_ranker | ele-fashion | 0.0034 | 0.4158 | False |

Margin quartiles were descriptive only and did not affect training. Detailed joint accuracy/F1 and oracle-gain-captured fractions are in `phase_a_margin_stratification.csv`.

## Confidence and selective fallback

| dataset | modality | rho | high_q4_regret | low_q1_regret |
|---|---|---|---|---|
| Movies | text | -0.0288 | 0.3676 | 0.3043 |
| Movies | visual | 0.0246 | 0.5983 | 0.5116 |
| Reddit-S | text | -0.0676 | 0.1027 | 0.0345 |
| Reddit-S | visual | -0.0510 | 0.0606 | 0.0956 |
| ele-fashion | text | 0.0587 | 0.2151 | 0.1341 |
| ele-fashion | visual | -0.0016 | 0.1061 | 0.1238 |

| nominal_train_coverage | actual_coverage | accuracy | macro_f1 | mean_joint_regret |
|---|---|---|---|---|
| 0.2500 | 0.2602 | 0.8042 | 0.7231 | 0.3047 |
| 0.5000 | 0.5089 | 0.8042 | 0.7231 | 0.3236 |
| 0.7500 | 0.7566 | 0.8043 | 0.7232 | 0.3433 |
| 1.0000 | 0.9999 | 0.8044 | 0.7231 | 0.3586 |

Coverage thresholds came from Train confidence distributions. The 50% row is the preregistered selective diagnostic; other coverage levels are descriptive.

## Toys versus Grocery confirmation

| dataset | mean_oracle_gain_vs_uniform | mean_oracle_margin | uniform_best_fraction | mean_best_action_stability | mean_pairwise_order_stability | ranker_mean_selection_regret | ranker_oracle_agreement | ranker_confidence_spearman | confidence_regret_status | E6_D0_delta_accuracy_vs_Uniform | E7_ranker_delta_accuracy_vs_Uniform | mean_joint_deployment_regret |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Toys | 0.15552 | 0.10421 | 0.03044 | 0.69572 | 0.81287 | — | — | — | NOT_TESTED: no confirmation ranker was eligible after lock | 0.00032 | — | — |
| Grocery | 0.20679 | 0.12872 | 0.04510 | 0.67929 | 0.82135 | — | — | — | NOT_TESTED: no confirmation ranker was eligible after lock | -0.00020 | — | — |

The winner was E6 D0, so the protocol required frozen C0/D0 confirmation and did not authorize confirmation ranker fits. R0/R1 regret and confidence on Toys/Grocery are therefore NOT_TESTED; no confirmation ranker was loaded.
The contrast differentiates relative oracle headroom and stability; it cannot distinguish low observability from deployment interaction without an eligible confirmation ranker.

## Interpretation

The screening gates did not qualify a learned ranker. The winner remains the reused E6 D0 baseline by validation Accuracy, Macro-F1, regret and consistency among eligible candidates. This is evidence against promoting a new hard-action router from this experiment; it does not identify a better routing objective. Selective fallback also failed its preregistered aggregate/worst-dataset gate.

No Test indices, labels, or metrics were accessed. No C0 checkpoint was modified or trained. E6 source results were left unchanged.
