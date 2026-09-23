# CSSI P1 Structural Response Report

## 1. Code / protocol audit

This report uses the canonical `all_plain` backbone and the frozen NC protocol `unified_full_graph_nc_v1`. Only validation nodes are used for architecture and hypothesis decisions; test evaluation is disabled in the P1 launcher.

The near-zero utility band is `|u| <= 1.0e-06`. Standard deviations are population standard deviations (`ddof=0`). Degree groups are within-validation rank terciles.

## 2. Exact definition of `all_plain`

The implementation and call-chain audit is recorded in `docs/cssi_all_plain_audit.md`. The response probe subclasses `CoSIMAGAblation` without adding parameters or changing its forward path.

## 3. Mathematical equivalence checks

All `15` dataset/seed runs passed the float32 tolerance `2.0e-05`. Maximum observed errors: `operator_weight_max_abs_error`=0.000e+00; `text_recurrence_max_abs_error`=1.335e-05; `visual_recurrence_max_abs_error`=1.907e-05; `text_response_reconstruction_max_abs_error`=0.000e+00; `visual_response_reconstruction_max_abs_error`=0.000e+00; `text_operator_response_max_abs_error`=1.335e-05; `visual_operator_response_max_abs_error`=1.907e-05; `text_plain_representation_max_abs_error`=2.861e-06; `visual_plain_representation_max_abs_error`=2.861e-06; `leave_one_response_out_base_max_abs_error`=0.000e+00.

The checks cover text/visual normalized-operator equality, the direct `R_k=(P-I)S_{k-1}` residual, response reconstruction, plain-representation equivalence, and the leave-one-response-out base case.

The largest recurrence residual is slightly above `1e-5` only because GPU scatter accumulation is non-associative in float32; the maximum remains below the predeclared `2e-5` runtime tolerance.

## 4. Backbone sanity results

| Variant | Dataset | validation accuracy mean | validation Macro-F1 mean |
|---|---|---:|---:|
| last_hop | Grocery | 0.8081 | 0.7119 |
| last_hop | Movies | 0.5510 | 0.4835 |
| last_hop | Reddit-S | 0.9454 | 0.9011 |
| last_hop | Toys | 0.7965 | 0.7604 |
| last_hop | ele-fashion | 0.8402 | 0.6983 |
| no_graph | Grocery | 0.7881 | 0.7146 |
| no_graph | Movies | 0.5289 | 0.4182 |
| no_graph | Reddit-S | 0.9334 | 0.8828 |
| no_graph | Toys | 0.7581 | 0.7321 |
| no_graph | ele-fashion | 0.8802 | 0.7626 |
| plain | Grocery | 0.8342 | 0.7614 |
| plain | Movies | 0.5781 | 0.5169 |
| plain | Reddit-S | 0.9630 | 0.9263 |
| plain | Toys | 0.8032 | 0.7743 |
| plain | ele-fashion | 0.8780 | 0.7553 |

No test result is used in the following interpretation.

## 5. Structural Response and utility definition

For each frozen best-validation checkpoint, `R_k=S_k-S_{k-1}`. The plain representation is `H0 + 0.75 R1 + 0.50 R2 + 0.25 R3`. Each counterfactual removes one response from one modality, keeps the other modality unchanged, then reruns modality refinement, late fusion, and the original classifier. The reported quantity is frozen-forward functional utility: `u_ce = loss_counterfactual - loss_base` and `u_margin = margin_base - margin_counterfactual`; positive means helpful.

## 6. Per-dataset / modality / order statistics

Detailed machine-readable values are in the selected output directory's `summary.csv`; node-level exports are compressed under `/hdd1/DataInHere/YHF/cssi/outputs/cssi_p1_corrected/node_level/`. The principal aggregate columns are `utility_ce_mean`, `utility_ce_std`, `p_utility_ce_gt0`, `p_utility_ce_lt0`, `response_norm_mean`, and the Pearson/Spearman response-norm correlations.

| Dataset | Modality | Order | N | CE mean | CE std | P(CE>0) | P(CE<0) | Spearman(norm, CE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Grocery | text | 1 | 10245 | 0.032315 | 0.41891 | 0.690 | 0.310 | 0.0659 |
| Grocery | text | 2 | 10245 | 0.0056418 | 0.11764 | 0.593 | 0.383 | 0.0417 |
| Grocery | text | 3 | 10245 | 0.00052152 | 0.021308 | 0.560 | 0.416 | 0.0454 |
| Grocery | visual | 1 | 10245 | 0.18899 | 0.80395 | 0.757 | 0.242 | 0.1109 |
| Grocery | visual | 2 | 10245 | 0.019805 | 0.19638 | 0.635 | 0.343 | 0.0684 |
| Grocery | visual | 3 | 10245 | 0.0021877 | 0.032634 | 0.621 | 0.357 | 0.0497 |
| Movies | text | 1 | 10002 | 0.02418 | 0.48042 | 0.525 | 0.474 | 0.0238 |
| Movies | text | 2 | 10002 | 0.0048188 | 0.10635 | 0.495 | 0.474 | 0.0314 |
| Movies | text | 3 | 10002 | -0.00059459 | 0.020249 | 0.466 | 0.502 | -0.0003 |
| Movies | visual | 1 | 10002 | 0.2466 | 0.92854 | 0.636 | 0.364 | 0.1521 |
| Movies | visual | 2 | 10002 | 0.026464 | 0.21094 | 0.530 | 0.445 | 0.0715 |
| Movies | visual | 3 | 10002 | 0.0019985 | 0.037304 | 0.498 | 0.472 | 0.0551 |
| Reddit-S | text | 1 | 9537 | 0.0025555 | 0.32894 | 0.785 | 0.213 | 0.1854 |
| Reddit-S | text | 2 | 9537 | 7.2655e-10 | 1.5183e-07 | 0.042 | 0.041 | -0.0071 |
| Reddit-S | text | 3 | 9537 | 2.5533e-10 | 1.3059e-07 | 0.036 | 0.036 | -0.0035 |
| Reddit-S | visual | 1 | 9537 | 0.056834 | 0.72881 | 0.740 | 0.234 | 0.3062 |
| Reddit-S | visual | 2 | 9537 | 2.5538e-09 | 1.5977e-07 | 0.047 | 0.040 | -0.0074 |
| Reddit-S | visual | 3 | 9537 | 1.4424e-10 | 1.4264e-07 | 0.040 | 0.036 | 0.0084 |
| Toys | text | 1 | 12417 | 0.024562 | 0.3159 | 0.658 | 0.341 | 0.1003 |
| Toys | text | 2 | 12417 | 0.0039502 | 0.080392 | 0.594 | 0.376 | 0.0762 |
| Toys | text | 3 | 12417 | 0.00037458 | 0.017524 | 0.592 | 0.379 | 0.0813 |
| Toys | visual | 1 | 12417 | 0.12 | 0.53441 | 0.717 | 0.281 | 0.1711 |
| Toys | visual | 2 | 12417 | 0.015281 | 0.12514 | 0.625 | 0.347 | 0.1131 |
| Toys | visual | 3 | 12417 | 0.0026639 | 0.026082 | 0.632 | 0.339 | 0.1192 |
| ele-fashion | text | 1 | 29331 | 0.0028653 | 0.40519 | 0.713 | 0.286 | 0.0912 |
| ele-fashion | text | 2 | 29331 | 0.00021593 | 0.19979 | 0.690 | 0.273 | 0.1744 |
| ele-fashion | text | 3 | 29331 | -0.0015466 | 0.027273 | 0.628 | 0.334 | -0.0299 |
| ele-fashion | visual | 1 | 29331 | 0.0044326 | 0.23183 | 0.565 | 0.432 | 0.0072 |
| ele-fashion | visual | 2 | 29331 | 0.0030619 | 0.096681 | 0.549 | 0.414 | 0.0357 |
| ele-fashion | visual | 3 | 29331 | -2.2245e-05 | 0.01298 | 0.500 | 0.461 | -0.0030 |

## 7. Cross-seed stability

There are 30 dataset × modality × order cells with all three seeds. Their seed-level utility means have the same sign in 20 cells. This is a descriptive stability count, not a success threshold.

## 8. Degree-group analysis

The summary contains low/medium/high within-validation degree-tercile rows for every dataset, seed, modality, and order. Differences should be read from the corresponding `degree_group` rows rather than inferred from the all-node aggregate.

## 9. Response magnitude vs utility

Across the 30 aggregate cells with defined Spearman correlation, the mean correlation is `0.0711` and the median is `0.0605`. This reports association only; it does not fit an adaptation rule.

## 10. CE vs margin consistency

CE/margin sign agreement across aggregate cells has mean `0.8475` and median `0.8991`. Disagreements are retained in the raw and summary outputs.

## 11. Mean versus dispersion

Across the 30 cells, the mean and median of `|mean(u)|/(std(u)+epsilon)` are `0.0642` and `0.0485`. The full cell-level values are in `mean_dispersion.csv`; this separates population-average utility from node-level heterogeneity.

## 12. Effect-size-aware seed stability

| minimum |mu|/sigma | cells | same-sign cells | fraction |
|---:|---:|---:|---:|
| 0.00 | 30 | 20 | 0.667 |
| 0.02 | 21 | 19 | 0.905 |
| 0.05 | 14 | 14 | 1.000 |
| 0.10 | 7 | 7 | 1.000 |

The denominator for each row is restricted to cells whose aggregate effect magnitude meets that threshold; near-zero cells are not treated as equally strong evidence.

## 13. Response magnitude association and modality differences

Within-cell node-level Spearman correlations have mean `0.0711` and median `0.0605`. Across the 30 aggregate cells, Spearman(mean response norm, |mean utility|) is `0.8122`. These answer different questions: node-level prediction within a condition versus utility scale differences across conditions.

| order | modality mean absolute difference | modality median absolute difference |
|---:|---:|---:|
| 1 | 0.10608 | 0.095433 |
| 2 | 0.0099971 | 0.011331 |
| 3 | 0.0016146 | 0.0016662 |

Per-dataset/order modality differences are in `modality_difference.csv`.

## 14. Propagation saturation

A validation cell is flagged near-degenerate when its aggregate median `||R_k||/(||S_{k-1}||+epsilon)` is at most `1e-3`. This is a descriptive flag, not a utility sign threshold.
Flagged cells: Reddit-S/text/order2, Reddit-S/text/order3, Reddit-S/visual/order2, Reddit-S/visual/order3.
The full saturation table is `saturation.csv`; utility signs in flagged cells should not be interpreted as strong evidence of useful response heterogeneity.

## 15. Findings for H1.1–H1.6

- **H1.1 Non-degeneracy:** 30/30 aggregate cells contain both positive and negative CE utility; this directly describes whether response utility is mixed rather than imposing a threshold.
- **H1.2 Node conditionality:** 30/30 cells have nonzero node-level CE dispersion; inspect the exported std and quantiles for its magnitude.
- **H1.3 Modality conditionality:** the mean absolute text/visual utility-mean difference over 15 paired cells is `0.039229`.
- **H1.4 Order conditionality:** the mean within-cell spread across orders over 10 dataset × modality pairs is `0.069775`.
- **H1.5 Magnitude insufficiency:** within-cell response-norm/CE Spearman has mean `0.0711` and median `0.0605`, so magnitude is weak for node-level utility ranking. Across aggregate cells, however, the correlation with `|mean utility|` is `0.8122`; response attenuation can explain utility scale across orders while remaining insufficient to decide node-level helpful versus harmful sign.
- **H1.6 Stability:** same-sign seed means occur in 20/30 complete cells. Cross-dataset repetition is therefore reported explicitly rather than declared from one pooled number.

## 16. Implementation caveats / limitations

This is a frozen-forward functional leave-one-response-out intervention, not a strict causal effect. It uses validation nodes for the decision, does not retrain a response-specific head, and keeps the original classifier and downstream modules fixed. The inherited inactive RCMI/MRC parameters remain in the checkpoint but are bypassed by `all_plain`.

## 17. Recommendation for P2

The follow-up P2 cross-modal evidence probe is reported separately in `docs/cssi_p2_crossmodal_evidence_report.md`; this P1 implementation does not implement or select a P2 mechanism.

## Reproducibility

- Analysis splits: train, validation
- Seeds: 42, 43, 44
- Checkpoint selection: best validation accuracy
- Test evaluation: disabled
