# IAMOC-v2: Relation-State Conditioned Interaction

## 1. Motivation from IAMOC-v1 Diagnosis

IAMOC-v1 established non-uniform hop attention, query-dependent cross-order interaction, and distinct Text/Visual patterns. Its V3 relation bias was numerically silent: median bias/content-logit RMS ratio was about `1.33e-5`, and relation-off/shuffle barely changed attention or predictions. IAMOC-v2 conditions hop interaction on detached moments of Stage-I calibrated edge weights.

## 2. IAMOC-v2 Design

The model inherits MoPFIAMOC. It computes incident-edge mean and population standard deviation from the existing calibrated modality-specific edge weights, z-normalizes each descriptor graph-wise, and detaches it. Relation state enters only hop-attention key logits. Stage I/II, response bank, fusion, classifier, and NC runner are inherited.

The inherited `forward()` calls `_encode_components()` twice only when `model.ppc_weight > 0` in training mode, then applies `_ppc_raw_loss()` to returned `delta_node_text/visual`. The subclass returns interaction-aware residuals at those keys, so PPC acts on the intended preference profiles without editing task code. Evaluation uses one encode. Total-loss coefficient is `task.loss.aux_weight × model.ppc_weight`; the NC config value is 1.0.

## 3. R0–R3 Definitions

- R0: IAMOC-v1 V2-equivalent, no relation condition, no PPC.
- R1: standardized mean relation state times a centered unit-RMS random order profile; learned sigmoid scale starts at 0.10.
- R2: modality-specific 2→16→K+1 profile MLP; output is centered and unit-RMS per node, with learned scale initialized to 0.10.
- R3: R2 plus the inherited PPC objective.

Every variant uses one interaction layer, one attention head, no FFN, and the same Stage I/II, response composition, fusion, classifier, and full-graph NC protocol. R0 reuses IAMOC-v1 V2 checkpoints.

## 4. PPC Validation-only Sweep

| PPC weight | Mean Val Acc | Mean Val Macro-F1 | Effective coefficient |
|---:|---:|---:|---:|
| 0.001 | 0.7101 | 0.6461 | 0.001 |
| 0.01 | 0.7093 | 0.6451 | 0.01 |
| 0.05 | 0.7123 | 0.6528 | 0.05 |

Selected `0.05` by mean validation Accuracy, using validation Macro-F1 as tie-break. Test evaluation was disabled during the sweep.

## 5. Correctness

- R0 strict state-dict load and forward equivalence: `True`; max difference `0`.
- Trained IAMOC-v1 V2 Movies seed-42 checkpoint equivalence on `16672` nodes: `True`; max difference `2.384e-06` (tolerance `1e-05`).
- R1/R2 descriptors `[N,2]`, biases `[N,K+1]`, finite normalization, order centering, and isolated-node raw moments: passed.
- R2 Relation-Off Stage-I/II invariance: `True`, max difference `1.192e-07`.
- Detached node shuffle: `True`; PPC train/eval behavior: `True`.

## 6. Movies/Grocery Three-Seed Performance

| Dataset | Variant | Metric | Mean | Population SD | Paired seed Δ vs R0 |
|---|---|---|---:|---:|---:|
| Movies | R0 | val_acc | 0.5795 | 0.005893 | 0 |
| Movies | R0 | val_macro_f1 | 0.5086 | 0.0116 | 0 |
| Movies | R0 | test_acc | 0.564 | 0.004176 | 0 |
| Movies | R0 | test_macro_f1 | 0.5045 | 0.005626 | 0 |
| Movies | R1 | val_acc | 0.577 | 0.002641 | -0.0025 |
| Movies | R1 | val_macro_f1 | 0.5053 | 0.009917 | -0.003316 |
| Movies | R1 | test_acc | 0.564 | 0.005534 | 1.987e-08 |
| Movies | R1 | test_macro_f1 | 0.5015 | 0.004315 | -0.002924 |
| Movies | R2 | val_acc | 0.5772 | 0.003023 | -0.0023 |
| Movies | R2 | val_macro_f1 | 0.5062 | 0.01305 | -0.002406 |
| Movies | R2 | test_acc | 0.5588 | 0.004164 | -0.005197 |
| Movies | R2 | test_macro_f1 | 0.5018 | 0.01083 | -0.002629 |
| Movies | R3 | val_acc | 0.5789 | 0.001224 | -0.0005999 |
| Movies | R3 | val_macro_f1 | 0.5137 | 0.00628 | 0.005063 |
| Movies | R3 | test_acc | 0.5588 | 0.001496 | -0.005197 |
| Movies | R3 | test_macro_f1 | 0.4989 | 0.009779 | -0.00555 |
| Grocery | R0 | val_acc | 0.8386 | 0.003581 | 0 |
| Grocery | R0 | val_macro_f1 | 0.7704 | 0.009495 | 0 |
| Grocery | R0 | test_acc | 0.8335 | 0.002272 | 0 |
| Grocery | R0 | test_macro_f1 | 0.7604 | 0.00674 | 0 |
| Grocery | R1 | val_acc | 0.8403 | 0.0002761 | 0.001757 |
| Grocery | R1 | val_macro_f1 | 0.7778 | 0.007372 | 0.007403 |
| Grocery | R1 | test_acc | 0.831 | 0.00435 | -0.00244 |
| Grocery | R1 | test_macro_f1 | 0.7584 | 0.01052 | -0.002014 |
| Grocery | R2 | val_acc | 0.8403 | 0.0009052 | 0.001757 |
| Grocery | R2 | val_macro_f1 | 0.7746 | 0.01138 | 0.004139 |
| Grocery | R2 | test_acc | 0.8282 | 0.004032 | -0.005271 |
| Grocery | R2 | test_macro_f1 | 0.7522 | 0.01353 | -0.008184 |
| Grocery | R3 | val_acc | 0.8388 | 0.001317 | 0.0002928 |
| Grocery | R3 | val_macro_f1 | 0.7718 | 0.01122 | 0.001374 |
| Grocery | R3 | test_acc | 0.8312 | 0.006107 | -0.002245 |
| Grocery | R3 | test_macro_f1 | 0.7518 | 0.0146 | -0.008566 |

Individual matched-seed deltas for each metric are in `analysis/paired_seed_deltas.csv`. The checkpoints are selected on validation Accuracy. Validation metrics guide model judgment; test metrics are descriptive only.

## 7. Relation-Bias Scale

| Dataset | Variant | Modality | Bias RMS | Content RMS | Median ratio | Node profile RMS |
|---|---|---|---:|---:|---:|---:|
| Movies | R1 | text | 0.1021 | 8.641 | 0.01193 | 0.1444 |
| Movies | R1 | visual | 0.1007 | 4.007 | 0.02371 | 0.1424 |
| Movies | R2 | text | 0.09987 | 9.201 | 0.01091 | 0.07454 |
| Movies | R2 | visual | 0.0999 | 4.719 | 0.02286 | 0.06719 |
| Movies | R3 | text | 0.09986 | 9.208 | 0.01109 | 0.07467 |
| Movies | R3 | visual | 0.1006 | 4.478 | 0.0208 | 0.06452 |
| Grocery | R1 | text | 0.104 | 5.134 | 0.02574 | 0.1471 |
| Grocery | R1 | visual | 0.1016 | 6.63 | 0.01632 | 0.1436 |
| Grocery | R2 | text | 0.1013 | 5.401 | 0.01925 | 0.07243 |
| Grocery | R2 | visual | 0.1039 | 5.901 | 0.0176 | 0.06765 |
| Grocery | R3 | text | 0.1011 | 4.774 | 0.01907 | 0.07261 |
| Grocery | R3 | visual | 0.1042 | 5.434 | 0.01947 | 0.07773 |

The relation-bias/content-logit ratios are roughly three orders of magnitude above IAMOC-v1 V3. Per-order variation and calibrated `mu/sigma` distributions are in `analysis/relation_bias.csv`.

### Partial Spearman association

| Dataset | Variant | Modality | Component | Mean rho across seeds | Range |
|---|---|---|---|---:|---:|
| Movies | R1 | text | mu | -0.6259 | [-0.675, -0.595] |
| Movies | R1 | text | sigma | 0.1903 | [0.1428, 0.2608] |
| Movies | R1 | visual | mu | -0.003665 | [-0.1006, 0.06849] |
| Movies | R1 | visual | sigma | 0.02099 | [-0.03408, 0.07774] |
| Movies | R2 | text | mu | -0.5056 | [-0.5713, -0.4615] |
| Movies | R2 | text | sigma | 0.1274 | [0.04134, 0.2061] |
| Movies | R2 | visual | mu | -0.04527 | [-0.177, 0.03381] |
| Movies | R2 | visual | sigma | 0.04211 | [-0.006004, 0.1211] |
| Movies | R3 | text | mu | -0.5101 | [-0.5533, -0.4773] |
| Movies | R3 | text | sigma | 0.1176 | [0.03342, 0.1789] |
| Movies | R3 | visual | mu | -0.01455 | [-0.08008, 0.08943] |
| Movies | R3 | visual | sigma | 0.009776 | [-0.05129, 0.04414] |
| Grocery | R1 | text | mu | -0.2551 | [-0.5441, -0.1079] |
| Grocery | R1 | text | sigma | 0.1095 | [0.01427, 0.2293] |
| Grocery | R1 | visual | mu | 0.1531 | [0.1394, 0.1798] |
| Grocery | R1 | visual | sigma | -0.006392 | [-0.04935, 0.03173] |
| Grocery | R2 | text | mu | -0.1796 | [-0.3996, -0.05951] |
| Grocery | R2 | text | sigma | 0.1265 | [0.05433, 0.2281] |
| Grocery | R2 | visual | mu | 0.1212 | [0.07801, 0.1755] |
| Grocery | R2 | visual | sigma | -0.01826 | [-0.1044, 0.03251] |
| Grocery | R3 | text | mu | -0.2149 | [-0.411, -0.07383] |
| Grocery | R3 | text | sigma | 0.1324 | [0.04321, 0.2266] |
| Grocery | R3 | visual | mu | 0.09471 | [0.06424, 0.1375] |
| Grocery | R3 | visual | sigma | -0.0005607 | [-0.03578, 0.01962] |

These are associations between descriptor components and `r_attn`, controlling `log(1+degree)`; they do not establish causation.

## 8. Relation-Off / Shuffle

| Dataset | Variant | Intervention | Attn MAE Text | Attn MAE Visual | Z MAE | Logit MAE | Probability KL | Flip rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Movies | R1 | off | 0.0129 | 0.01462 | 0.0003195 | 0.000646 | 2.063e-06 | 0.0001999 |
| Movies | R1 | shuffle | 0.01778 | 0.02026 | 0.0004475 | 0.0009014 | 3.779e-06 | 0.0002399 |
| Movies | R2 | off | 0.01876 | 0.01619 | 0.0003268 | 0.0006383 | 1.246e-06 | 0.0003199 |
| Movies | R2 | shuffle | 0.01202 | 0.01018 | 0.0001836 | 0.0003657 | 6.836e-07 | 0.0002199 |
| Movies | R3 | off | 0.01867 | 0.01628 | 0.0005446 | 0.00108 | 4.056e-06 | 0.0004199 |
| Movies | R3 | shuffle | 0.01213 | 0.009488 | 0.0003145 | 0.000627 | 1.8e-06 | 0.0001599 |
| Grocery | R1 | off | 0.0129 | 0.01245 | 0.001555 | 0.003769 | 3.975e-05 | 0.0002733 |
| Grocery | R1 | shuffle | 0.01835 | 0.0172 | 0.002292 | 0.00556 | 7.147e-05 | 0.000449 |
| Grocery | R2 | off | 0.0155 | 0.0156 | 0.001843 | 0.004703 | 1.892e-05 | 0.0003709 |
| Grocery | R2 | shuffle | 0.009377 | 0.008772 | 0.00124 | 0.00325 | 1.938e-05 | 0.0002148 |
| Grocery | R3 | off | 0.01727 | 0.01546 | 0.001529 | 0.003628 | 1.724e-05 | 0.0002538 |
| Grocery | R3 | shuffle | 0.01035 | 0.009965 | 0.001499 | 0.003587 | 3.302e-05 | 0.0002538 |

The CSV also records `r_attn` shift, eta MAE, Stage-I/II tensor differences, validation/test accuracy and Macro-F1 deltas, and split-specific prediction flips.

## 9. Cross-Order Interaction

| Dataset | Variant | Modality | Query-row MAE | Pairwise query JS | r_attn | r_eta | Hop gate | Embedding-off Z MAE | Embedding-off logit MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | R0 | text | 0.005863 | 0.001309 | 0.6428 | 0.5694 | 0.1452 | 0.005306 | 0.0105 |
| Movies | R0 | visual | 0.03194 | 0.01797 | 0.4301 | 0.5859 | 0.179 | 0.005306 | 0.0105 |
| Movies | R1 | text | 0.005336 | 0.0009236 | 0.6343 | 0.5705 | 0.1454 | 0.006642 | 0.01361 |
| Movies | R1 | visual | 0.03032 | 0.01655 | 0.4331 | 0.5869 | 0.1763 | 0.006642 | 0.01361 |
| Movies | R2 | text | 0.00548 | 0.001105 | 0.6418 | 0.5701 | 0.1447 | 0.00418 | 0.008351 |
| Movies | R2 | visual | 0.03009 | 0.01646 | 0.4479 | 0.5854 | 0.1759 | 0.00418 | 0.008351 |
| Movies | R3 | text | 0.005597 | 0.001109 | 0.6454 | 0.5677 | 0.1441 | 0.008493 | 0.01704 |
| Movies | R3 | visual | 0.03281 | 0.01843 | 0.4524 | 0.5841 | 0.1813 | 0.008493 | 0.01704 |
| Grocery | R0 | text | 0.02141 | 0.009185 | 0.5466 | 0.5776 | 0.1794 | 0.01218 | 0.02764 |
| Grocery | R0 | visual | 0.04791 | 0.03534 | 0.5231 | 0.6201 | 0.1893 | 0.01218 | 0.02764 |
| Grocery | R1 | text | 0.02539 | 0.01387 | 0.4463 | 0.5864 | 0.1904 | 0.02884 | 0.06993 |
| Grocery | R1 | visual | 0.05769 | 0.0472 | 0.5122 | 0.6206 | 0.2009 | 0.02884 | 0.06993 |
| Grocery | R2 | text | 0.02594 | 0.01432 | 0.4672 | 0.5902 | 0.1888 | 0.02981 | 0.07641 |
| Grocery | R2 | visual | 0.06142 | 0.05171 | 0.5087 | 0.6269 | 0.2006 | 0.02981 | 0.07641 |
| Grocery | R3 | text | 0.0233 | 0.01215 | 0.5258 | 0.5865 | 0.1818 | 0.02242 | 0.05357 |
| Grocery | R3 | visual | 0.054 | 0.04231 | 0.5127 | 0.6246 | 0.1913 | 0.02242 | 0.05357 |

KL-to-uniform, normalized entropy, max-key mass, node-level query statistics, and `r_attn` seed variance are in `analysis/iamoc_interaction.csv`. Nonzero query-row MAE/JS and embedding-off shifts indicate that query-dependent cross-order interaction remains active.

## 10. Modality Heterogeneity

| Dataset | Variant | r_attn Text−Visual | r_eta Text−Visual | Entropy Text−Visual | Query JS Text−Visual |
|---|---|---:|---:|---:|---:|
| Movies | R0 | 0.2127 | -0.01646 | 0.1528 | -0.01666 |
| Movies | R1 | 0.2012 | -0.01631 | 0.1279 | -0.01563 |
| Movies | R2 | 0.194 | -0.01539 | 0.1424 | -0.01535 |
| Movies | R3 | 0.1929 | -0.01643 | 0.1413 | -0.01732 |
| Grocery | R0 | 0.02353 | -0.04252 | 0.1675 | -0.02615 |
| Grocery | R1 | -0.06589 | -0.03416 | 0.07213 | -0.03333 |
| Grocery | R2 | -0.04152 | -0.03667 | 0.05253 | -0.03739 |
| Grocery | R3 | 0.01317 | -0.03815 | 0.0809 | -0.03015 |

Full per-modality measures are in the interaction and relation-bias CSV files.

## 11. PPC Stability Effect

| Dataset | Variant | Seed | PPC weight | Effective coefficient | Logged PPC raw | Aux contribution | 3-view PPC raw | Text discrepancy | Visual discrepancy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | R2 | 42 | 0 | 0 | NA | 0 | 8.377e-05 | 1.312e-09 | 0.0001675 |
| Movies | R2 | 43 | 0 | 0 | NA | 0 | 5.07e-05 | 1.117e-09 | 0.0001014 |
| Movies | R2 | 44 | 0 | 0 | NA | 0 | 0.0001799 | 6.316e-09 | 0.0003597 |
| Movies | R3 | 42 | 0.05 | 0.05 | 0.0001879 | 9.397e-06 | 0.0002684 | 5.882e-09 | 0.0005367 |
| Movies | R3 | 43 | 0.05 | 0.05 | 5.315e-05 | 2.658e-06 | 8.099e-05 | 3.313e-09 | 0.000162 |
| Movies | R3 | 44 | 0.05 | 0.05 | 9.717e-05 | 4.858e-06 | 0.0001249 | 3.88e-09 | 0.0002497 |
| Grocery | R2 | 42 | 0 | 0 | NA | 0 | 0.000466 | 0.0005794 | 0.0003526 |
| Grocery | R2 | 43 | 0 | 0 | NA | 0 | 1.405e-06 | 1.158e-07 | 2.695e-06 |
| Grocery | R2 | 44 | 0 | 0 | NA | 0 | 0.001717 | 0.001958 | 0.001475 |
| Grocery | R3 | 42 | 0.05 | 0.05 | 5.804e-05 | 2.902e-06 | 6.96e-05 | 0.0001158 | 2.336e-05 |
| Grocery | R3 | 43 | 0.05 | 0.05 | 9.901e-06 | 4.95e-07 | 6.41e-06 | 2.084e-07 | 1.261e-05 |
| Grocery | R3 | 44 | 0.05 | 0.05 | 0.000512 | 2.56e-05 | 0.001511 | 0.002075 | 0.0009464 |


| Dataset | Measure | R2 mean | R3 mean | Relative change R3 vs R2 |
|---|---|---:|---:|---:|
| Movies | stochastic_view_ppc_raw_mean_3 | 0.0001048 | 0.0001581 | 0.5085 |
| Movies | text_profile_discrepancy_mean_3 | 2.915e-09 | 4.358e-09 | 0.4952 |
| Movies | visual_profile_discrepancy_mean_3 | 0.0002096 | 0.0003161 | 0.5085 |
| Movies | text_node_preference_variance_across_seeds | 1.627e-09 | 3.388e-09 | 1.082 |
| Movies | visual_node_preference_variance_across_seeds | 0.0005201 | 0.0006208 | 0.1936 |
| Movies | text_r_attn_variance_across_seeds | 0.0009495 | 0.0009036 | -0.04837 |
| Movies | visual_r_attn_variance_across_seeds | 0.03689 | 0.0409 | 0.1087 |
| Grocery | stochastic_view_ppc_raw_mean_3 | 0.000728 | 0.0005289 | -0.2735 |
| Grocery | text_profile_discrepancy_mean_3 | 0.0008459 | 0.0007304 | -0.1365 |
| Grocery | visual_profile_discrepancy_mean_3 | 0.0006102 | 0.0003275 | -0.4633 |
| Grocery | text_node_preference_variance_across_seeds | 0.002169 | 0.001815 | -0.1633 |
| Grocery | visual_node_preference_variance_across_seeds | 0.001363 | 0.000696 | -0.4893 |
| Grocery | text_r_attn_variance_across_seeds | 0.04143 | 0.03595 | -0.1325 |
| Grocery | visual_r_attn_variance_across_seeds | 0.03963 | 0.04011 | 0.0121 |

PPC effects are mixed across datasets: assess view discrepancy, node-level preference variance, node-level `r_attn` seed variance, and performance together. Do not treat PPC as a stability improvement unless those measures improve consistently.

Node-level preference-profile variance across seeds and effective training auxiliary contribution are also in `analysis/ppc_stability.csv`. A visually smoother attention map is not sufficient evidence of PPC benefit.

## 12. Performance–Mechanism Tradeoff

Interpret validation performance, seed stability, relation intervention effects, bias/content scale, and cross-order diagnostics together. A small validation decrease can be acceptable when relation bias has measurable functional effects and seed stability remains comparable. Near-zero relation bias is not acceptable regardless of narrative fit.

## 13. Recommendation

**Mechanistically Strong Candidate: R1.** Median relation-bias/content ratio `0.01647`; mean paired validation-accuracy change `-0.0003713`; Relation-Shuffle shifts logits more than Relation-Off, query diversity remains nonzero, validation seed variance does not exceed R0 on either dataset, and the validation change is within R0's seed spread. PPC stability evidence remains dataset-dependent. Review per-dataset results before promotion.

The analysis is limited to Movies and Grocery NC with seeds 42/43/44. No LP or other datasets were run.

Analysis artifact counts: `{"association_rows": 72, "intervention_rows": 36, "paired_seed_delta_rows": 72, "performance_rows": 24, "performance_summary_rows": 32, "ppc_rows": 12, "profile_rows": 336, "relation_rows": 36}`.
