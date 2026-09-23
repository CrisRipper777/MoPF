# CSSI P3 conditional structural-response results

This report is generated from best-validation checkpoints by `scripts/analyze_cssi_p3.py`. All claims use validation nodes only; `task.evaluate_test=false` and no test metric is used.

## 1. Protocol and implementation

The six variants use `unified_full_graph_nc_v1`, the existing NC splits, seeds 42/43/44, and the same inherited refiner/fusion/classifier. The exact architecture is frozen in `docs/cssi_p3_architecture.md`. The low-rank adapter is rank 8 with fixed output scale 1e-3 and zero-initialized up projection; this was added after an early smoke run without the scale showed an exploding correction ratio.

## 2. Validation backbone results

| variant | datasets | seeds | val_acc_mean | val_acc_std | val_f1_mean | val_f1_std |
| --- | --- | --- | --- | --- | --- | --- |
| all_lr | 5 | 3 | 0.8116 | 0.1346 | 0.7503 | 0.1379 |
| mrc_plain | 5 | 3 | 0.8117 | 0.1339 | 0.7492 | 0.1380 |
| same_lr | 5 | 3 | 0.8124 | 0.1325 | 0.7483 | 0.1415 |
| same_lr_mrc | 5 | 3 | 0.8120 | 0.1344 | 0.7478 | 0.1417 |
| same_scalar | 5 | 3 | 0.8120 | 0.1332 | 0.7487 | 0.1409 |
| self_lr | 5 | 3 | 0.8113 | 0.1340 | 0.7461 | 0.1378 |

### Per-dataset validation cells

| dataset | variant | N | val_acc_mean | val_acc_std | val_f1_mean | val_f1_std |
| --- | --- | --- | --- | --- | --- | --- |
| Grocery | all_lr | 3 | 0.8373 | 0.0043 | 0.7711 | 0.0105 |
| Grocery | mrc_plain | 3 | 0.8363 | 0.0026 | 0.7697 | 0.0131 |
| Grocery | same_lr | 3 | 0.8368 | 0.0043 | 0.7711 | 0.0143 |
| Grocery | same_lr_mrc | 3 | 0.8381 | 0.0041 | 0.7738 | 0.0103 |
| Grocery | same_scalar | 3 | 0.8366 | 0.0038 | 0.7701 | 0.0134 |
| Grocery | self_lr | 3 | 0.8357 | 0.0050 | 0.7619 | 0.0127 |
| Movies | all_lr | 3 | 0.5744 | 0.0019 | 0.5130 | 0.0038 |
| Movies | mrc_plain | 3 | 0.5767 | 0.0017 | 0.5142 | 0.0083 |
| Movies | same_lr | 3 | 0.5798 | 0.0044 | 0.5031 | 0.0154 |
| Movies | same_lr_mrc | 3 | 0.5759 | 0.0021 | 0.5047 | 0.0037 |
| Movies | same_scalar | 3 | 0.5782 | 0.0037 | 0.5054 | 0.0081 |
| Movies | self_lr | 3 | 0.5759 | 0.0035 | 0.5111 | 0.0101 |
| Reddit-S | all_lr | 3 | 0.9632 | 0.0022 | 0.9264 | 0.0028 |
| Reddit-S | mrc_plain | 3 | 0.9646 | 0.0024 | 0.9290 | 0.0032 |
| Reddit-S | same_lr | 3 | 0.9632 | 0.0022 | 0.9254 | 0.0047 |
| Reddit-S | same_lr_mrc | 3 | 0.9647 | 0.0023 | 0.9296 | 0.0033 |
| Reddit-S | same_scalar | 3 | 0.9635 | 0.0020 | 0.9270 | 0.0039 |
| Reddit-S | self_lr | 3 | 0.9633 | 0.0021 | 0.9250 | 0.0057 |
| Toys | all_lr | 3 | 0.8039 | 0.0022 | 0.7763 | 0.0034 |
| Toys | mrc_plain | 3 | 0.8032 | 0.0026 | 0.7753 | 0.0046 |
| Toys | same_lr | 3 | 0.8032 | 0.0023 | 0.7750 | 0.0044 |
| Toys | same_lr_mrc | 3 | 0.8027 | 0.0022 | 0.7738 | 0.0046 |
| Toys | same_scalar | 3 | 0.8031 | 0.0026 | 0.7750 | 0.0058 |
| Toys | self_lr | 3 | 0.8027 | 0.0011 | 0.7735 | 0.0030 |
| ele-fashion | all_lr | 3 | 0.8793 | 0.0012 | 0.7647 | 0.0056 |
| ele-fashion | mrc_plain | 3 | 0.8778 | 0.0011 | 0.7578 | 0.0044 |
| ele-fashion | same_lr | 3 | 0.8789 | 0.0017 | 0.7669 | 0.0020 |
| ele-fashion | same_lr_mrc | 3 | 0.8789 | 0.0009 | 0.7570 | 0.0034 |
| ele-fashion | same_scalar | 3 | 0.8786 | 0.0023 | 0.7660 | 0.0089 |
| ele-fashion | self_lr | 3 | 0.8789 | 0.0004 | 0.7590 | 0.0046 |

The per-seed paired rows are in `outputs/cssi_p3/paired_differences.csv`; aggregate means and win/tie/loss are in `outputs/cssi_p3/paired_differences_aggregate.csv`.

## 3. Requested paired comparisons

| comparison | dataset | metric | N | mean | std | median | positive_fraction | zero_fraction | negative_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All-LR-Same-LR | ALL | val_acc | 15 | -0.0007 | 0.0033 | 0.0000 | 0.4000 | 0.2000 | 0.4000 |
| All-LR-Same-LR | ALL | val_macro_f1 | 15 | 0.0020 | 0.0075 | 0.0003 | 0.5333 | 0.0667 | 0.4000 |
| P3-MRC-Plain | ALL | val_acc | 15 | 0.0004 | 0.0023 | 0.0007 | 0.6000 | 0.0000 | 0.4000 |
| P3-MRC-Plain | ALL | val_macro_f1 | 15 | 0.0024 | 0.0110 | 0.0031 | 0.7333 | 0.0000 | 0.2667 |
| Same-LR-MRC-Same-LR | ALL | val_acc | 15 | -0.0003 | 0.0023 | 0.0009 | 0.5333 | 0.0000 | 0.4667 |
| Same-LR-MRC-Same-LR | ALL | val_macro_f1 | 15 | -0.0005 | 0.0070 | -0.0008 | 0.4000 | 0.0000 | 0.6000 |
| Same-LR-Same-Scalar | ALL | val_acc | 15 | 0.0004 | 0.0012 | 0.0007 | 0.6000 | 0.0667 | 0.3333 |
| Same-LR-Same-Scalar | ALL | val_macro_f1 | 15 | -0.0004 | 0.0055 | -0.0014 | 0.4000 | 0.0000 | 0.6000 |
| Same-LR-Self-LR | ALL | val_acc | 15 | 0.0011 | 0.0029 | 0.0003 | 0.6000 | 0.2000 | 0.2000 |
| Same-LR-Self-LR | ALL | val_macro_f1 | 15 | 0.0022 | 0.0078 | 0.0019 | 0.7333 | 0.0000 | 0.2667 |
| all_lr-Plain | ALL | val_acc | 15 | 0.0003 | 0.0026 | 0.0008 | 0.6667 | 0.0000 | 0.3333 |
| all_lr-Plain | ALL | val_macro_f1 | 15 | 0.0035 | 0.0089 | 0.0035 | 0.6667 | 0.0000 | 0.3333 |
| same_lr-Plain | ALL | val_acc | 15 | 0.0011 | 0.0028 | 0.0009 | 0.6667 | 0.0000 | 0.3333 |
| same_lr-Plain | ALL | val_macro_f1 | 15 | 0.0015 | 0.0147 | 0.0022 | 0.6667 | 0.0000 | 0.3333 |
| same_lr_mrc-Plain | ALL | val_acc | 15 | 0.0008 | 0.0029 | 0.0009 | 0.6667 | 0.0000 | 0.3333 |
| same_lr_mrc-Plain | ALL | val_macro_f1 | 15 | 0.0009 | 0.0119 | 0.0020 | 0.7333 | 0.0000 | 0.2667 |
| same_scalar-Plain | ALL | val_acc | 15 | 0.0007 | 0.0026 | 0.0009 | 0.6667 | 0.0000 | 0.3333 |
| same_scalar-Plain | ALL | val_macro_f1 | 15 | 0.0018 | 0.0121 | 0.0033 | 0.6000 | 0.0000 | 0.4000 |
| self_lr-Plain | ALL | val_acc | 15 | 0.0000 | 0.0025 | 0.0004 | 0.6000 | 0.0000 | 0.4000 |
| self_lr-Plain | ALL | val_macro_f1 | 15 | -0.0007 | 0.0096 | -0.0016 | 0.4000 | 0.0000 | 0.6000 |

## 4. Response and controller diagnostics

`response_diagnostics.csv` reports validation response norms, correction norms, correction ratios, correction-response cosine, and zero-correction fractions by dataset, seed, modality, and order. `all_order_attention.csv` reports all-order attention matrices and entropy. `cross_interventions.csv` reports same-LR normal/off/shuffle frozen interventions.
Rows: 540; finite rows: 540; nonzero correction rows: 450.

| modality | diagonal_mass | offdiagonal_mass | attention_entropy |
| --- | --- | --- | --- |
| text | 0.3331 | 0.6669 | 1.0960 |
| visual | 0.3330 | 0.6670 | 1.0868 |

## 5. MRC diagnostics

| variant | dataset | seed | weight_gap_mean | semantic_gap_mean | incident_mean_abs_gap | top_neighbor_disagreement |
| --- | --- | --- | --- | --- | --- | --- |
| mrc_plain | Movies | 42 | 0.0685 | 0.2317 | 1.1833 | 0.5798 |
| same_lr_mrc | Movies | 42 | 0.0678 | 0.2373 | 1.1989 | 0.5800 |
| mrc_plain | Movies | 43 | 0.0676 | 0.2211 | 1.0489 | 0.5816 |
| same_lr_mrc | Movies | 43 | 0.0674 | 0.2170 | 1.0215 | 0.5828 |
| mrc_plain | Movies | 44 | 0.0721 | 0.2282 | 0.9375 | 0.5799 |
| same_lr_mrc | Movies | 44 | 0.0686 | 0.2312 | 0.9956 | 0.5818 |
| mrc_plain | Toys | 42 | 0.0475 | 0.1493 | 0.3357 | 0.4719 |
| same_lr_mrc | Toys | 42 | 0.0475 | 0.1499 | 0.3377 | 0.4731 |
| mrc_plain | Toys | 43 | 0.0470 | 0.1508 | 0.3526 | 0.4738 |
| same_lr_mrc | Toys | 43 | 0.0477 | 0.1509 | 0.3535 | 0.4729 |
| mrc_plain | Toys | 44 | 0.0470 | 0.1519 | 0.3477 | 0.4706 |
| same_lr_mrc | Toys | 44 | 0.0449 | 0.1500 | 0.3416 | 0.4724 |
| mrc_plain | Grocery | 42 | 0.0518 | 0.1524 | 0.5457 | 0.4916 |
| same_lr_mrc | Grocery | 42 | 0.0526 | 0.1554 | 0.5840 | 0.4921 |
| mrc_plain | Grocery | 43 | 0.0510 | 0.1510 | 0.5260 | 0.4962 |
| same_lr_mrc | Grocery | 43 | 0.0503 | 0.1486 | 0.4828 | 0.4968 |
| mrc_plain | Grocery | 44 | 0.0535 | 0.1550 | 0.5674 | 0.4920 |
| same_lr_mrc | Grocery | 44 | 0.0526 | 0.1531 | 0.5374 | 0.4934 |
| mrc_plain | ele-fashion | 42 | 0.0431 | 0.1489 | 0.2772 | 0.2995 |
| same_lr_mrc | ele-fashion | 42 | 0.0446 | 0.1517 | 0.2879 | 0.2994 |
| mrc_plain | ele-fashion | 43 | 0.0451 | 0.1500 | 0.2871 | 0.2994 |
| same_lr_mrc | ele-fashion | 43 | 0.0438 | 0.1472 | 0.2768 | 0.2988 |
| mrc_plain | ele-fashion | 44 | 0.0452 | 0.1503 | 0.2913 | 0.2997 |
| same_lr_mrc | ele-fashion | 44 | 0.0475 | 0.1547 | 0.3089 | 0.2999 |
| mrc_plain | Reddit-S | 42 | 0.0637 | 0.2259 | 1.9885 | 0.5096 |
| same_lr_mrc | Reddit-S | 42 | 0.0641 | 0.2269 | 2.0008 | 0.5099 |
| mrc_plain | Reddit-S | 43 | 0.0626 | 0.2234 | 1.9590 | 0.5111 |
| same_lr_mrc | Reddit-S | 43 | 0.0628 | 0.2238 | 1.9648 | 0.5116 |
| mrc_plain | Reddit-S | 44 | 0.0650 | 0.2287 | 2.0151 | 0.5103 |
| same_lr_mrc | Reddit-S | 44 | 0.0649 | 0.2284 | 2.0121 | 0.5102 |

## 6. Stability and limitations

Training-monitor rows: 90; rows containing textual NaN/Inf: 0. Correction ratio and gradient maxima are in `training_stability.csv`.
Low-rank validation rows with exactly zero correction: 0/360; mean absolute correction norm=0.003409.
All-order attention is close to uniform: mean diagonal mass=0.3330, entropy=1.0914 (log(3)=1.0986).
MRC edge-weight saturation fractions (lower/upper) are 0.0000 on average; projected semantic-gap vs weight-gap Pearson=0.9449.

## 7. Answers to the ten P3 questions

1. Self response adaptation vs Plain: mean paired Δ=0.0000, win/tie/loss fractions=0.600/0.000/0.400 (validation accuracy).
2. Same-order cross-modal conditioning vs Self: mean paired Δ=0.0011, win/tie/loss fractions=0.600/0.200/0.200 (validation accuracy).
3. Low-rank vs scalar gate: mean paired Δ=0.0004, win/tie/loss fractions=0.600/0.067/0.333 (validation accuracy).
4. Directional low-rank transformation: mean paired Δ=0.0004, win/tie/loss fractions=0.600/0.067/0.333 (validation accuracy).
5. All-order context: mean paired Δ=-0.0007, win/tie/loss fractions=0.400/0.200/0.400 (validation accuracy).
6. MRC alone: mean paired Δ=0.0004, win/tie/loss fractions=0.600/0.000/0.400 (validation accuracy).
7. MRC plus response adaptation: mean paired Δ=-0.0003, win/tie/loss fractions=0.533/0.000/0.467 (validation accuracy).
8. Matched cross-modal evidence: normal-minus-cross-off mean validation-accuracy shift=0.0000; normal-minus-cross-shuffle=-0.0000; off/shuffle representation L2 shifts=0.000545/0.000559.
9. Correction direction: mean correction-vs-response cosine=-0.1091; positive (>0.1) rows=0.034, negative (<-0.1) rows=0.226; the remainder is directional/near-orthogonal.
10. Validation-only P4 candidate by mean validation accuracy is `same_lr` (0.8124); this is not a final benchmark decision and uses no test result.
Recommendation: keep `same_lr` as the only conservative P4 follow-up candidate, but do not claim a validated CSSI mechanism yet; its gain is small, cross-off/shuffle changes representations without changing validation decisions, and all-order attention is nearly uniform.

A positive paired difference means the first named variant has higher validation metric. These are functional validation comparisons, not causal effects. No P4 mechanism is implemented by this turn.
