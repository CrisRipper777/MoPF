# CSSI P5 Conditionality Necessity Report

All new experiments are NC-only validation runs under `unified_full_graph_nc_v1`, using existing splits, seeds 42/43/44, `task.evaluate_test=false`, and zero auxiliary-loss weight. No LP or test metric is used.

## Decision: Partial-Go

Same-vs-StaticOrder: Acc Δ=0.0015, Macro-F1 Δ=0.0021; dataset-positive=5/5 and 4/5, seed-positive=0.800 and 0.600.
Same-vs-Self: Acc Δ=0.0001, Macro-F1 Δ=0.0009; dataset-positive=3/5 and 3/5, seed-positive=0.400 and 0.533.
Cross interventions: cross_off: Acc 2/5, Macro-F1 2/5 / cross_shuffle: Acc 1/5, Macro-F1 2/5.
Mean-gate interventions: node_gate_shuffle: 1/5 datasets with both task metrics improved by Normal / train_mean_gate: 2/5 datasets with both task metrics improved by Normal.
5/6 pooled modality/order strata pass the nontrivial-association screen.
Same-Scalar is stronger than Static-Order, but its gain over Self-Scalar is not stable; retain only a cautious node-adaptive structural-response interpretation.
The operational check treats a comparison as stable only when both Accuracy and Macro-F1 have positive aggregate deltas, are positive on at least three of five dataset means, and are positive on at least half of the paired seed rows. This is an audit convention, not a claim of statistical significance.

## 1. Model hierarchy validation metrics

| variant | N | val_acc_mean | val_acc_std | val_macro_f1_mean | val_macro_f1_std |
| --- | --- | --- | --- | --- | --- |
| plain | 15 | 0.8113 | 0.1329 | 0.7468 | 0.1361 |
| same_scalar_mrc | 15 | 0.8129 | 0.1347 | 0.7494 | 0.1425 |
| same_scalar_v1 | 15 | 0.8132 | 0.1333 | 0.7504 | 0.1386 |
| self_scalar | 15 | 0.8131 | 0.1338 | 0.7495 | 0.1408 |
| static_modality | 15 | 0.8122 | 0.1331 | 0.7477 | 0.1390 |
| static_order | 15 | 0.8117 | 0.1334 | 0.7484 | 0.1407 |

### Per-dataset mean ± std

| dataset | variant | N | val_acc_mean | val_acc_std | val_macro_f1_mean | val_macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- |
| Grocery | plain | 3 | 0.8342 | 0.0042 | 0.7614 | 0.0108 |
| Grocery | same_scalar_mrc | 3 | 0.8388 | 0.0037 | 0.7720 | 0.0136 |
| Grocery | same_scalar_v1 | 3 | 0.8376 | 0.0040 | 0.7688 | 0.0111 |
| Grocery | self_scalar | 3 | 0.8394 | 0.0045 | 0.7710 | 0.0194 |
| Grocery | static_modality | 3 | 0.8370 | 0.0054 | 0.7648 | 0.0072 |
| Grocery | static_order | 3 | 0.8363 | 0.0033 | 0.7719 | 0.0116 |
| Movies | plain | 3 | 0.5781 | 0.0018 | 0.5169 | 0.0184 |
| Movies | same_scalar_mrc | 3 | 0.5761 | 0.0019 | 0.5026 | 0.0031 |
| Movies | same_scalar_v1 | 3 | 0.5793 | 0.0056 | 0.5124 | 0.0081 |
| Movies | self_scalar | 3 | 0.5779 | 0.0036 | 0.5069 | 0.0212 |
| Movies | static_modality | 3 | 0.5783 | 0.0005 | 0.5088 | 0.0080 |
| Movies | static_order | 3 | 0.5776 | 0.0026 | 0.5061 | 0.0115 |
| Reddit-S | plain | 3 | 0.9630 | 0.0030 | 0.9263 | 0.0033 |
| Reddit-S | same_scalar_mrc | 3 | 0.9654 | 0.0028 | 0.9285 | 0.0057 |
| Reddit-S | same_scalar_v1 | 3 | 0.9645 | 0.0025 | 0.9280 | 0.0065 |
| Reddit-S | self_scalar | 3 | 0.9645 | 0.0030 | 0.9274 | 0.0052 |
| Reddit-S | static_modality | 3 | 0.9631 | 0.0020 | 0.9260 | 0.0017 |
| Reddit-S | static_order | 3 | 0.9635 | 0.0020 | 0.9272 | 0.0043 |
| Toys | plain | 3 | 0.8032 | 0.0012 | 0.7743 | 0.0034 |
| Toys | same_scalar_mrc | 3 | 0.8042 | 0.0027 | 0.7773 | 0.0039 |
| Toys | same_scalar_v1 | 3 | 0.8041 | 0.0030 | 0.7770 | 0.0075 |
| Toys | self_scalar | 3 | 0.8037 | 0.0022 | 0.7741 | 0.0037 |
| Toys | static_modality | 3 | 0.8030 | 0.0024 | 0.7713 | 0.0030 |
| Toys | static_order | 3 | 0.8026 | 0.0026 | 0.7761 | 0.0038 |
| ele-fashion | plain | 3 | 0.8780 | 0.0008 | 0.7553 | 0.0068 |
| ele-fashion | same_scalar_mrc | 3 | 0.8802 | 0.0005 | 0.7668 | 0.0090 |
| ele-fashion | same_scalar_v1 | 3 | 0.8806 | 0.0018 | 0.7659 | 0.0076 |
| ele-fashion | self_scalar | 3 | 0.8802 | 0.0010 | 0.7680 | 0.0052 |
| ele-fashion | static_modality | 3 | 0.8794 | 0.0010 | 0.7675 | 0.0031 |
| ele-fashion | static_order | 3 | 0.8784 | 0.0023 | 0.7605 | 0.0082 |

## 2. Paired comparisons

| comparison | dataset | seed | metric | delta | level | mean | median | std | positive_fraction | zero_fraction | negative_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| static_order-static_modality | ALL | ALL | val_acc | -0.0005 | all | -0.0005 | 0.0000 | 0.0019 | 0.4667 | 0.0667 | 0.4667 |
| static_order-static_modality | ALL | ALL | val_macro_f1 | 0.0007 | all | 0.0007 | 0.0026 | 0.0079 | 0.6000 | 0.0000 | 0.4000 |
| self_scalar-static_order | ALL | ALL | val_acc | 0.0014 | all | 0.0014 | 0.0017 | 0.0018 | 0.8000 | 0.0000 | 0.2000 |
| self_scalar-static_order | ALL | ALL | val_macro_f1 | 0.0011 | all | 0.0011 | 0.0013 | 0.0062 | 0.5333 | 0.0000 | 0.4667 |
| same_scalar_v1-self_scalar | ALL | ALL | val_acc | 0.0001 | all | 0.0001 | -0.0005 | 0.0018 | 0.4000 | 0.0667 | 0.5333 |
| same_scalar_v1-self_scalar | ALL | ALL | val_macro_f1 | 0.0009 | all | 0.0009 | 0.0013 | 0.0081 | 0.5333 | 0.0000 | 0.4667 |
| same_scalar_v1-static_order | ALL | ALL | val_acc | 0.0015 | all | 0.0015 | 0.0015 | 0.0023 | 0.8000 | 0.0667 | 0.1333 |
| same_scalar_v1-static_order | ALL | ALL | val_macro_f1 | 0.0021 | all | 0.0021 | 0.0024 | 0.0071 | 0.6000 | 0.0000 | 0.4000 |
| same_scalar_mrc-same_scalar_v1 | ALL | ALL | val_acc | -0.0003 | all | -0.0003 | 0.0003 | 0.0023 | 0.6000 | 0.0000 | 0.4000 |
| same_scalar_mrc-same_scalar_v1 | ALL | ALL | val_macro_f1 | -0.0010 | all | -0.0010 | 0.0009 | 0.0091 | 0.5333 | 0.0000 | 0.4667 |
| same_scalar_v1-plain | ALL | ALL | val_acc | 0.0019 | all | 0.0019 | 0.0013 | 0.0031 | 0.7333 | 0.0000 | 0.2667 |
| same_scalar_v1-plain | ALL | ALL | val_macro_f1 | 0.0036 | all | 0.0036 | 0.0037 | 0.0118 | 0.6667 | 0.0000 | 0.3333 |

Per-seed and per-dataset deltas are in `outputs/cssi_p5/paired_differences.csv`; positive delta means the first model is better.

## 3. Same-Scalar gate distribution

| modality | order | gate_mean | gate_std | gate_q10 | gate_q25 | gate_q50 | gate_q75 | gate_q90 | positive_fraction | negative_fraction | node_variation_var_i | order_variation_mean_var_k | modality_difference_mean_abs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 1 | -0.0451 | 0.0078 | -0.0495 | -0.0494 | -0.0492 | -0.0393 | -0.0345 | 0.0389 | 0.9611 | 0.0003 | 0.0001 | 0.0357 |
| text | 2 | -0.0450 | 0.0065 | -0.0478 | -0.0478 | -0.0476 | -0.0472 | -0.0333 | 0.0248 | 0.9752 | 0.0002 | 0.0001 | 0.0362 |
| text | 3 | -0.0469 | 0.0062 | -0.0493 | -0.0493 | -0.0491 | -0.0488 | -0.0358 | 0.0211 | 0.9789 | 0.0002 | 0.0001 | 0.0359 |
| visual | 1 | -0.0113 | 0.0139 | -0.0185 | -0.0183 | -0.0116 | -0.0115 | 0.0086 | 0.4017 | 0.5983 | 0.0005 | 0.0001 | 0.0357 |
| visual | 2 | -0.0124 | 0.0115 | -0.0179 | -0.0178 | -0.0114 | -0.0109 | -0.0021 | 0.3854 | 0.6146 | 0.0004 | 0.0001 | 0.0362 |
| visual | 3 | -0.0129 | 0.0082 | -0.0182 | -0.0181 | -0.0117 | -0.0114 | -0.0111 | 0.3852 | 0.6148 | 0.0002 | 0.0001 | 0.0359 |
`node_variation_var_i` is variance across validation nodes for a fixed modality/order; `order_variation_mean_var_k` is the mean within-node variance across orders; modality difference is paired text-versus-visual absolute gate difference.

## 4. Frozen-forward interventions

| variant | intervention | val_acc | val_macro_f1 | mean_z_l2_shift | mean_z_cosine_to_normal | gate_l1_shift | gate_l2_shift | mean_correction_l2_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same_scalar_mrc | cross_off | 0.8127 | 0.7490 | 0.0433 | 1.0000 | 0.0030 | 0.0061 | 0.0085 |
| same_scalar_mrc | cross_shuffle | 0.8128 | 0.7492 | 0.0497 | 0.9999 | 0.0029 | 0.0065 | 0.0089 |
| same_scalar_mrc | modality_mean_gate | 0.8126 | 0.7488 | 0.0918 | 0.9999 | 0.0064 | 0.0119 | 0.0192 |
| same_scalar_mrc | node_gate_shuffle | 0.8126 | 0.7489 | 0.1073 | 0.9998 | 0.0062 | 0.0132 | 0.0198 |
| same_scalar_mrc | normal | 0.8129 | 0.7494 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | train_mean_gate | 0.8125 | 0.7487 | 0.0967 | 0.9999 | 0.0058 | 0.0112 | 0.0190 |
| same_scalar_v1 | cross_off | 0.8132 | 0.7503 | 0.0608 | 0.9999 | 0.0040 | 0.0080 | 0.0117 |
| same_scalar_v1 | cross_shuffle | 0.8130 | 0.7499 | 0.0614 | 0.9999 | 0.0033 | 0.0074 | 0.0111 |
| same_scalar_v1 | modality_mean_gate | 0.8126 | 0.7502 | 0.0993 | 0.9999 | 0.0068 | 0.0126 | 0.0214 |
| same_scalar_v1 | node_gate_shuffle | 0.8128 | 0.7502 | 0.1177 | 0.9998 | 0.0065 | 0.0138 | 0.0218 |
| same_scalar_v1 | normal | 0.8132 | 0.7504 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | train_mean_gate | 0.8126 | 0.7502 | 0.1053 | 0.9999 | 0.0062 | 0.0118 | 0.0211 |

### Per-dataset means

| variant | dataset | intervention | val_acc | val_macro_f1 | mean_z_l2_shift | mean_z_cosine_to_normal | gate_l1_shift | gate_l2_shift | mean_correction_l2_shift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same_scalar_mrc | Grocery | cross_off | 0.8385 | 0.7713 | 0.0934 | 0.9999 | 0.0068 | 0.0154 | 0.0210 |
| same_scalar_mrc | Grocery | cross_shuffle | 0.8382 | 0.7713 | 0.1328 | 0.9999 | 0.0090 | 0.0209 | 0.0281 |
| same_scalar_mrc | Grocery | modality_mean_gate | 0.8378 | 0.7715 | 0.2223 | 0.9998 | 0.0181 | 0.0339 | 0.0556 |
| same_scalar_mrc | Grocery | node_gate_shuffle | 0.8379 | 0.7712 | 0.2680 | 0.9996 | 0.0184 | 0.0407 | 0.0595 |
| same_scalar_mrc | Grocery | normal | 0.8388 | 0.7720 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | Grocery | train_mean_gate | 0.8381 | 0.7715 | 0.2263 | 0.9998 | 0.0172 | 0.0332 | 0.0562 |
| same_scalar_mrc | Movies | cross_off | 0.5756 | 0.5016 | 0.0708 | 0.9999 | 0.0029 | 0.0062 | 0.0101 |
| same_scalar_mrc | Movies | cross_shuffle | 0.5763 | 0.5023 | 0.0838 | 0.9999 | 0.0030 | 0.0072 | 0.0114 |
| same_scalar_mrc | Movies | modality_mean_gate | 0.5755 | 0.5003 | 0.1988 | 0.9998 | 0.0101 | 0.0189 | 0.0317 |
| same_scalar_mrc | Movies | node_gate_shuffle | 0.5753 | 0.5005 | 0.2368 | 0.9996 | 0.0098 | 0.0209 | 0.0348 |
| same_scalar_mrc | Movies | normal | 0.5761 | 0.5026 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | Movies | train_mean_gate | 0.5750 | 0.5002 | 0.2252 | 0.9998 | 0.0095 | 0.0184 | 0.0337 |
| same_scalar_mrc | Reddit-S | cross_off | 0.9653 | 0.9284 | 0.0253 | 1.0000 | 0.0018 | 0.0032 | 0.0034 |
| same_scalar_mrc | Reddit-S | cross_shuffle | 0.9652 | 0.9283 | 0.0317 | 1.0000 | 0.0025 | 0.0044 | 0.0048 |
| same_scalar_mrc | Reddit-S | modality_mean_gate | 0.9652 | 0.9279 | 0.0316 | 1.0000 | 0.0026 | 0.0046 | 0.0049 |
| same_scalar_mrc | Reddit-S | node_gate_shuffle | 0.9653 | 0.9285 | 0.0313 | 1.0000 | 0.0025 | 0.0044 | 0.0048 |
| same_scalar_mrc | Reddit-S | normal | 0.9654 | 0.9285 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | Reddit-S | train_mean_gate | 0.9652 | 0.9279 | 0.0320 | 1.0000 | 0.0025 | 0.0044 | 0.0049 |
| same_scalar_mrc | Toys | cross_off | 0.8039 | 0.7768 | 0.0270 | 1.0000 | 0.0034 | 0.0058 | 0.0081 |
| same_scalar_mrc | Toys | cross_shuffle | 0.8042 | 0.7773 | 0.0001 | 1.0000 | 0.0000 | 0.0000 | 0.0001 |
| same_scalar_mrc | Toys | modality_mean_gate | 0.8042 | 0.7773 | 0.0025 | 1.0000 | 0.0002 | 0.0004 | 0.0007 |
| same_scalar_mrc | Toys | node_gate_shuffle | 0.8042 | 0.7773 | 0.0002 | 1.0000 | 0.0000 | 0.0000 | 0.0001 |
| same_scalar_mrc | Toys | normal | 0.8042 | 0.7773 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | Toys | train_mean_gate | 0.8042 | 0.7773 | 0.0001 | 1.0000 | 0.0000 | 0.0000 | 0.0001 |
| same_scalar_mrc | ele-fashion | cross_off | 0.8802 | 0.7668 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | ele-fashion | cross_shuffle | 0.8802 | 0.7668 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | ele-fashion | modality_mean_gate | 0.8802 | 0.7671 | 0.0037 | 1.0000 | 0.0010 | 0.0019 | 0.0030 |
| same_scalar_mrc | ele-fashion | node_gate_shuffle | 0.8802 | 0.7668 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | ele-fashion | normal | 0.8802 | 0.7668 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_mrc | ele-fashion | train_mean_gate | 0.8802 | 0.7668 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | Grocery | cross_off | 0.8375 | 0.7696 | 0.1337 | 0.9999 | 0.0089 | 0.0194 | 0.0307 |
| same_scalar_v1 | Grocery | cross_shuffle | 0.8364 | 0.7668 | 0.1613 | 0.9998 | 0.0105 | 0.0239 | 0.0359 |
| same_scalar_v1 | Grocery | modality_mean_gate | 0.8371 | 0.7689 | 0.2311 | 0.9998 | 0.0198 | 0.0367 | 0.0631 |
| same_scalar_v1 | Grocery | node_gate_shuffle | 0.8361 | 0.7675 | 0.2849 | 0.9996 | 0.0203 | 0.0435 | 0.0675 |
| same_scalar_v1 | Grocery | normal | 0.8376 | 0.7688 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | Grocery | train_mean_gate | 0.8368 | 0.7685 | 0.2393 | 0.9998 | 0.0194 | 0.0368 | 0.0654 |
| same_scalar_v1 | Movies | cross_off | 0.5792 | 0.5112 | 0.1041 | 0.9999 | 0.0036 | 0.0079 | 0.0136 |
| same_scalar_v1 | Movies | cross_shuffle | 0.5796 | 0.5116 | 0.1183 | 0.9998 | 0.0038 | 0.0089 | 0.0148 |
| same_scalar_v1 | Movies | modality_mean_gate | 0.5769 | 0.5110 | 0.2309 | 0.9997 | 0.0105 | 0.0198 | 0.0345 |
| same_scalar_v1 | Movies | node_gate_shuffle | 0.5789 | 0.5125 | 0.2737 | 0.9994 | 0.0096 | 0.0208 | 0.0364 |
| same_scalar_v1 | Movies | normal | 0.5793 | 0.5124 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | Movies | train_mean_gate | 0.5772 | 0.5114 | 0.2618 | 0.9997 | 0.0094 | 0.0185 | 0.0356 |
| same_scalar_v1 | Reddit-S | cross_off | 0.9645 | 0.9280 | 0.0517 | 1.0000 | 0.0058 | 0.0102 | 0.0095 |
| same_scalar_v1 | Reddit-S | cross_shuffle | 0.9646 | 0.9281 | 0.0236 | 1.0000 | 0.0021 | 0.0037 | 0.0037 |
| same_scalar_v1 | Reddit-S | modality_mean_gate | 0.9646 | 0.9281 | 0.0249 | 1.0000 | 0.0020 | 0.0037 | 0.0043 |
| same_scalar_v1 | Reddit-S | node_gate_shuffle | 0.9645 | 0.9279 | 0.0245 | 1.0000 | 0.0022 | 0.0039 | 0.0039 |
| same_scalar_v1 | Reddit-S | normal | 0.9645 | 0.9280 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | Reddit-S | train_mean_gate | 0.9646 | 0.9281 | 0.0211 | 1.0000 | 0.0018 | 0.0031 | 0.0032 |
| same_scalar_v1 | Toys | cross_off | 0.8041 | 0.7769 | 0.0143 | 1.0000 | 0.0015 | 0.0027 | 0.0044 |
| same_scalar_v1 | Toys | cross_shuffle | 0.8041 | 0.7770 | 0.0035 | 1.0000 | 0.0002 | 0.0004 | 0.0010 |
| same_scalar_v1 | Toys | modality_mean_gate | 0.8041 | 0.7770 | 0.0051 | 1.0000 | 0.0004 | 0.0008 | 0.0017 |
| same_scalar_v1 | Toys | node_gate_shuffle | 0.8041 | 0.7770 | 0.0052 | 1.0000 | 0.0003 | 0.0006 | 0.0014 |
| same_scalar_v1 | Toys | normal | 0.8041 | 0.7770 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | Toys | train_mean_gate | 0.8041 | 0.7770 | 0.0041 | 1.0000 | 0.0002 | 0.0005 | 0.0012 |
| same_scalar_v1 | ele-fashion | cross_off | 0.8806 | 0.7659 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | ele-fashion | cross_shuffle | 0.8806 | 0.7659 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | ele-fashion | modality_mean_gate | 0.8806 | 0.7659 | 0.0046 | 1.0000 | 0.0012 | 0.0022 | 0.0034 |
| same_scalar_v1 | ele-fashion | node_gate_shuffle | 0.8806 | 0.7659 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | ele-fashion | normal | 0.8806 | 0.7659 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| same_scalar_v1 | ele-fashion | train_mean_gate | 0.8806 | 0.7659 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
The complete dataset/seed table is `outputs/cssi_p5/scalar_interventions.csv`; the table above is a compact dataset-mean view. Every intervention re-runs the frozen refinement, late fusion, and classifier; no logits were edited directly.
Cross-Off zeros the paired projected control after its Linear, so a Linear bias cannot restore paired evidence. Train-Mean-Gate uses train nodes only; Node-Gate-Shuffle permutes gates only within validation nodes.

## 5. Gate–utility functional alignment

| dataset | seed | modality | order | N | spearman_gate_utility_ce | spearman_gate_utility_margin | auroc_gate_predicts_positive_ce | utility_ce_q1_gate_mean | utility_ce_q1_gate_median | utility_ce_q2_gate_mean | utility_ce_q2_gate_median | utility_ce_q3_gate_mean | utility_ce_q3_gate_median | utility_ce_q4_gate_mean | utility_ce_q4_gate_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grocery | ALL | text | 1 | 10245 | -0.0280 | -0.0135 | 0.4871 | -0.0339 | -0.0497 | -0.0332 | -0.0497 | -0.0299 | -0.0497 | -0.0348 | -0.0497 |
| Grocery | ALL | text | 2 | 10245 | -0.0429 | 0.0103 | 0.4898 | -0.0406 | -0.0508 | -0.0383 | -0.0503 | -0.0386 | -0.0509 | -0.0440 | -0.0509 |
| Grocery | ALL | text | 3 | 10245 | -0.0212 | 0.0048 | 0.4978 | -0.0441 | -0.0525 | -0.0409 | -0.0521 | -0.0411 | -0.0526 | -0.0450 | -0.0526 |
| Grocery | ALL | visual | 1 | 10245 | -0.0148 | -0.0601 | 0.5133 | 0.0352 | 0.0507 | 0.0363 | 0.0509 | 0.0353 | 0.0506 | 0.0361 | 0.0506 |
| Grocery | ALL | visual | 2 | 10245 | -0.0087 | 0.0520 | 0.4772 | 0.0345 | 0.0513 | 0.0335 | 0.0513 | 0.0360 | 0.0514 | 0.0314 | 0.0513 |
| Grocery | ALL | visual | 3 | 10245 | -0.0550 | 0.0363 | 0.4871 | 0.0381 | 0.0510 | 0.0379 | 0.0510 | 0.0375 | 0.0510 | 0.0365 | 0.0510 |
| Movies | ALL | text | 1 | 10002 | -0.0160 | 0.0124 | 0.4924 | -0.0533 | -0.0529 | -0.0533 | -0.0529 | -0.0534 | -0.0529 | -0.0533 | -0.0529 |
| Movies | ALL | text | 2 | 10002 | 0.0147 | -0.0201 | 0.5067 | -0.0493 | -0.0492 | -0.0493 | -0.0492 | -0.0493 | -0.0492 | -0.0493 | -0.0492 |
| Movies | ALL | text | 3 | 10002 | -0.0077 | 0.0137 | 0.5028 | -0.0527 | -0.0522 | -0.0527 | -0.0522 | -0.0527 | -0.0522 | -0.0527 | -0.0522 |
| Movies | ALL | visual | 1 | 10002 | -0.0378 | 0.0393 | 0.4769 | -0.0334 | -0.0508 | -0.0352 | -0.0522 | -0.0360 | -0.0522 | -0.0380 | -0.0523 |
| Movies | ALL | visual | 2 | 10002 | -0.0362 | 0.0307 | 0.4861 | -0.0399 | -0.0529 | -0.0425 | -0.0529 | -0.0444 | -0.0529 | -0.0435 | -0.0529 |
| Movies | ALL | visual | 3 | 10002 | -0.0338 | 0.0328 | 0.4837 | -0.0443 | -0.0525 | -0.0471 | -0.0525 | -0.0476 | -0.0525 | -0.0470 | -0.0525 |
| Reddit-S | ALL | text | 1 | 9537 | -0.0784 | -0.0938 | 0.5187 | -0.0446 | -0.0502 | -0.0488 | -0.0502 | -0.0498 | -0.0502 | -0.0482 | -0.0502 |
| Reddit-S | ALL | text | 2 | 9537 | -0.0019 | 0.0008 | 0.5393 | -0.0469 | -0.0500 | -0.0438 | -0.0500 | nan | nan | nan | nan |
| Reddit-S | ALL | text | 3 | 9537 | 0.0165 | -0.0109 | 0.5595 | -0.0469 | -0.0500 | -0.0432 | -0.0500 | nan | nan | nan | nan |
| Reddit-S | ALL | visual | 1 | 9537 | -0.0272 | -0.0838 | 0.5369 | -0.0509 | -0.0504 | -0.0509 | -0.0504 | -0.0511 | -0.0504 | -0.0509 | -0.0504 |
| Reddit-S | ALL | visual | 2 | 9537 | -0.0149 | 0.0024 | 0.4630 | -0.0500 | -0.0500 | -0.0500 | -0.0500 | nan | nan | nan | nan |
| Reddit-S | ALL | visual | 3 | 9537 | -0.0085 | 0.0003 | 0.4736 | -0.0500 | -0.0500 | -0.0500 | -0.0500 | nan | nan | nan | nan |
| Toys | ALL | text | 1 | 12417 | 0.0099 | 0.0108 | 0.5037 | -0.0363 | -0.0493 | -0.0360 | -0.0493 | -0.0354 | -0.0493 | -0.0360 | -0.0493 |
| Toys | ALL | text | 2 | 12417 | 0.0082 | -0.0048 | 0.4988 | -0.0371 | -0.0501 | -0.0366 | -0.0501 | -0.0365 | -0.0501 | -0.0366 | -0.0501 |
| Toys | ALL | text | 3 | 12417 | -0.0065 | 0.0158 | 0.4924 | -0.0371 | -0.0505 | -0.0367 | -0.0505 | -0.0364 | -0.0505 | -0.0366 | -0.0505 |
| Toys | ALL | visual | 1 | 12417 | 0.0080 | -0.0318 | 0.5153 | 0.0505 | 0.0512 | 0.0506 | 0.0512 | 0.0508 | 0.0512 | 0.0507 | 0.0512 |
| Toys | ALL | visual | 2 | 12417 | 0.0138 | -0.0118 | 0.5041 | 0.0514 | 0.0514 | 0.0514 | 0.0514 | 0.0514 | 0.0514 | 0.0514 | 0.0514 |
| Toys | ALL | visual | 3 | 12417 | -0.0126 | -0.0009 | 0.5001 | 0.0512 | 0.0512 | 0.0512 | 0.0512 | 0.0512 | 0.0512 | 0.0512 | 0.0512 |
| ele-fashion | ALL | text | 1 | 29331 | 0.0188 | 0.0041 | 0.5039 | -0.0556 | -0.0557 | -0.0556 | -0.0557 | -0.0555 | -0.0557 | -0.0556 | -0.0557 |
| ele-fashion | ALL | text | 2 | 29331 | 0.0213 | -0.0098 | 0.5042 | -0.0519 | -0.0520 | -0.0519 | -0.0520 | -0.0519 | -0.0520 | -0.0519 | -0.0520 |
| ele-fashion | ALL | text | 3 | 29331 | 0.0129 | 0.0019 | 0.5005 | -0.0554 | -0.0553 | -0.0553 | -0.0553 | -0.0553 | -0.0553 | -0.0553 | -0.0553 |
| ele-fashion | ALL | visual | 1 | 29331 | -0.0254 | 0.0366 | 0.4818 | -0.0565 | -0.0569 | -0.0564 | -0.0569 | -0.0564 | -0.0569 | -0.0566 | -0.0569 |
| ele-fashion | ALL | visual | 2 | 29331 | -0.0494 | 0.0584 | 0.4692 | -0.0547 | -0.0557 | -0.0547 | -0.0557 | -0.0548 | -0.0557 | -0.0549 | -0.0557 |
| ele-fashion | ALL | visual | 3 | 29331 | -0.0223 | 0.0323 | 0.4850 | -0.0566 | -0.0571 | -0.0565 | -0.0571 | -0.0566 | -0.0571 | -0.0567 | -0.0571 |
| ALL | ALL | text | 1 | 71532 | -0.0330 | -0.0090 | 0.5081 | -0.0465 | -0.0529 | -0.0472 | -0.0534 | -0.0490 | -0.0534 | -0.0477 | -0.0529 |
| ALL | ALL | text | 2 | 71532 | 0.1178 | -0.2064 | 0.5141 | -0.0472 | -0.0513 | -0.0460 | -0.0513 | -0.0468 | -0.0500 | -0.0464 | -0.0508 |
| ALL | ALL | text | 3 | 71532 | 0.0629 | -0.1203 | 0.4552 | -0.0490 | -0.0527 | -0.0493 | -0.0540 | -0.0477 | -0.0500 | -0.0492 | -0.0527 |
| ALL | ALL | visual | 1 | 71532 | -0.2187 | 0.2263 | 0.4095 | -0.0011 | 0.0075 | -0.0170 | -0.0521 | -0.0422 | -0.0543 | -0.0238 | -0.0543 |
| ALL | ALL | visual | 2 | 71532 | -0.0768 | 0.0857 | 0.4322 | -0.0103 | -0.0513 | -0.0171 | -0.0526 | -0.0379 | -0.0500 | -0.0200 | -0.0526 |
| ALL | ALL | visual | 3 | 71532 | -0.1159 | 0.1259 | 0.4148 | -0.0069 | -0.0517 | -0.0294 | -0.0500 | -0.0313 | -0.0542 | -0.0218 | -0.0536 |
| ALL | ALL | ALL | 0 | 429192 | -0.0663 | 0.0463 | 0.4404 | -0.0267 | -0.0513 | -0.0342 | -0.0525 | -0.0427 | -0.0505 | -0.0347 | -0.0524 |
P1 utility is treated only as a cross-check. Because it comes from a different Plain checkpoint than P4 Same-Scalar, these correlations are cross-checkpoint functional associations, not supervised targets or causal effects.

## 6. MRC

| variant | dataset | seed | raw_semantic_gap_mean | raw_weight_gap_pearson | neighbor_allocation_tv_divergence | top_neighbor_disagreement | incident_mass_abs_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| same_scalar_mrc | Movies | 42 | 0.2391 | 0.9517 | 0.0143 | 0.5799 | 1.1797 |
| same_scalar_mrc | Movies | 43 | 0.2258 | 0.9480 | 0.0166 | 0.5839 | 1.0802 |
| same_scalar_mrc | Movies | 44 | 0.2278 | 0.9249 | 0.0196 | 0.5812 | 0.9506 |
| same_scalar_mrc | Toys | 42 | 0.1491 | 0.9445 | 0.0139 | 0.4724 | 0.3421 |
| same_scalar_mrc | Toys | 43 | 0.1509 | 0.9463 | 0.0131 | 0.4731 | 0.3468 |
| same_scalar_mrc | Toys | 44 | 0.1517 | 0.9473 | 0.0125 | 0.4713 | 0.3462 |
| same_scalar_mrc | Grocery | 42 | 0.1526 | 0.9401 | 0.0161 | 0.4937 | 0.5324 |
| same_scalar_mrc | Grocery | 43 | 0.1502 | 0.9376 | 0.0161 | 0.4964 | 0.4990 |
| same_scalar_mrc | Grocery | 44 | 0.1556 | 0.9399 | 0.0169 | 0.4912 | 0.5714 |
| same_scalar_mrc | ele-fashion | 42 | 0.1508 | 0.9542 | 0.0083 | 0.2996 | 0.3006 |
| same_scalar_mrc | ele-fashion | 43 | 0.1455 | 0.9535 | 0.0084 | 0.2995 | 0.2830 |
| same_scalar_mrc | ele-fashion | 44 | 0.1487 | 0.9539 | 0.0085 | 0.3014 | 0.3022 |
| same_scalar_mrc | Reddit-S | 42 | 0.2269 | 0.9431 | 0.0126 | 0.5077 | 2.0267 |
| same_scalar_mrc | Reddit-S | 43 | 0.2228 | 0.9441 | 0.0124 | 0.5084 | 1.9954 |
| same_scalar_mrc | Reddit-S | 44 | 0.2227 | 0.9447 | 0.0126 | 0.5088 | 1.9344 |
Raw semantic gaps use cosine(H0) before the learned diagonal metric. `incident_mass_abs_gap` is the incident relation-weight total difference, not an incident mean.

## 7. H1-style conditionality findings

- **H1.1 / non-degeneracy:** Same-Scalar gates are not exactly one-signed across all strata: text is predominantly negative (mean positive fraction 0.0283), while visual retains both signs (mean positive fraction 0.3908; mean negative fraction 0.6092). This establishes gate sign heterogeneity, not useful task conditionality by itself.
- **H1.2 / node conditionality:** Gate dispersion across nodes is measurable but compressed: the largest fixed-modality/order node variance is 0.000480; gate shuffling changes representations more clearly than it changes validation decisions. Evidence for functionally important node conditionality is weak.
- **H1.3 / modality conditionality:** Text-versus-visual paired gate difference averages 0.0359, so the controller distinguishes modalities at a coarse level. The hierarchy result does not show that paired cross-modal evidence is needed for the task gain.
- **H1.4 / order conditionality:** Mean within-node order variance is at most 0.000113 in the aggregate. Static-Order is close to Static-Modality, while Same-Scalar's advantage over Static-Order is small; order-specific calibration is plausible but not sufficient evidence for a conditional interaction mechanism.
- **H1.5 / utility alignment:** Aggregate cross-check values are Spearman(gate, CE utility)=-0.0663, Spearman(gate, margin utility)=0.0463, and AUROC=0.4404. Stratum-level signs are mixed; gate magnitude alone is not a reliable utility selector.
- **H1.6 / stability:** Same-Scalar is above Static-Order on the aggregate (Accuracy Δ=0.0015, Macro-F1 Δ=0.0021), but its difference from Self-Scalar is near zero (Accuracy Δ=0.0001, Macro-F1 Δ=0.0009) and is not stable under the stated audit rule. Cross-modal and gate-intervention task counts are shown below.

### Task-level intervention reproducibility

| intervention | datasets_normal_better_acc | datasets_normal_better_macro_f1 | datasets_normal_better_both |
| --- | --- | --- | --- |
| node_gate_shuffle | 2 | 2 | 1 |
| train_mean_gate | 2 | 2 | 2 |
| modality_mean_gate | 2 | 1 | 1 |
For Cross-Off and Cross-Shuffle, the screen counts how many of the five dataset means have Normal better than the intervention for each metric.
| intervention | datasets_normal_better_acc | datasets_normal_better_macro_f1 |
| --- | --- | --- |
| cross_off | 2 | 2 |
| cross_shuffle | 1 | 2 |

## 8. Go / Partial-Go / No-Go interpretation

**Partial-Go.**
- Same-vs-StaticOrder: Acc Δ=0.0015, Macro-F1 Δ=0.0021; dataset-positive=5/5 and 4/5, seed-positive=0.800 and 0.600.
- Same-vs-Self: Acc Δ=0.0001, Macro-F1 Δ=0.0009; dataset-positive=3/5 and 3/5, seed-positive=0.400 and 0.533.
- Cross interventions: cross_off: Acc 2/5, Macro-F1 2/5 / cross_shuffle: Acc 1/5, Macro-F1 2/5.
- Mean-gate interventions: node_gate_shuffle: 1/5 datasets with both task metrics improved by Normal / train_mean_gate: 2/5 datasets with both task metrics improved by Normal.
- 5/6 pooled modality/order strata pass the nontrivial-association screen.
- Same-Scalar is stronger than Static-Order, but its gain over Self-Scalar is not stable; retain only a cautious node-adaptive structural-response interpretation.
The evidence supports at most a lightweight scalar or node-adaptive structural-response calibration story. It does not justify presenting same-order cross-modal conditioning as a necessary core mechanism. Same-Scalar+MRC is mixed and slightly below Same-Scalar in the aggregate, so MRC should remain optional analysis rather than a mandatory module.

## 9. Caveats and implementation boundary

P5 is validation-only and uses the existing split/checkpoint protocol; no test metric, LP run, dataset-specific tuning, BRSM change, attention, router, or auxiliary loss was introduced. P1 utility alignment is a cross-checkpoint functional association because P1 Plain and P4 Same-Scalar are different frozen checkpoints; it is not a causal estimate or supervised gate target. The interventions are frozen-forward functional audits, not causal effects.
The scalar hierarchy is implemented independently in `src/models/cssi_scalar_probe.py`; historical `cosi_mag_final.py`, P1/P2/P3/P4 reference implementations, and their training logic were not modified.
