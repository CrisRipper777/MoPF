# CSSI-v1 P4 BRSM validation results

All results are validation-only best-checkpoint evaluations under `unified_full_graph_nc_v1`, using the existing splits and seeds 42/43/44. `task.evaluate_test=false`; no test metric is read or used. P1 Plain is the historical reference.

## 1. Architecture and protocol

CSSI-v1 uses a row-orthonormal rank-8 basis, `a=tanh(f(C))`, and `delta R=lambda * Bbar^T(a*Bbar R)`. `lambda=lambda_max*sigmoid(theta)`, with lambda_max=0.2 and lambda initialized near 0.05. The output contains no affine response bias or fixed 1e-3 scale. Same-order cross-modal control uses target, paired-other, product, and absolute-difference blocks; Cross-Off zeros the paired projected control after its Linear, so the paired contribution is exactly zero while the target path remains.

## 2. Validation metrics

| variant | N | val_acc_mean | val_acc_std | macro_f1_mean | macro_f1_std |
| --- | --- | --- | --- | --- | --- |
| same_brsm | 15 | 0.8126 | 0.1329 | 0.7487 | 0.1382 |
| same_brsm_mrc | 15 | 0.8129 | 0.1342 | 0.7485 | 0.1423 |
| same_scalar_v1 | 15 | 0.8132 | 0.1333 | 0.7504 | 0.1386 |
| self_brsm | 15 | 0.8122 | 0.1335 | 0.7464 | 0.1413 |

### Per-dataset mean ± std

| dataset | variant | N | val_acc_mean | val_acc_std | macro_f1_mean | macro_f1_std |
| --- | --- | --- | --- | --- | --- | --- |
| Grocery | same_brsm | 3 | 0.8365 | 0.0047 | 0.7676 | 0.0114 |
| Grocery | same_brsm_mrc | 3 | 0.8368 | 0.0031 | 0.7671 | 0.0121 |
| Grocery | same_scalar_v1 | 3 | 0.8376 | 0.0040 | 0.7688 | 0.0111 |
| Grocery | self_brsm | 3 | 0.8347 | 0.0059 | 0.7616 | 0.0188 |
| Movies | same_brsm | 3 | 0.5796 | 0.0052 | 0.5133 | 0.0118 |
| Movies | same_brsm_mrc | 3 | 0.5775 | 0.0071 | 0.5035 | 0.0086 |
| Movies | same_scalar_v1 | 3 | 0.5793 | 0.0056 | 0.5124 | 0.0081 |
| Movies | self_brsm | 3 | 0.5782 | 0.0055 | 0.5051 | 0.0107 |
| Reddit-S | same_brsm | 3 | 0.9646 | 0.0016 | 0.9286 | 0.0025 |
| Reddit-S | same_brsm_mrc | 3 | 0.9654 | 0.0033 | 0.9294 | 0.0058 |
| Reddit-S | same_scalar_v1 | 3 | 0.9645 | 0.0025 | 0.9280 | 0.0065 |
| Reddit-S | self_brsm | 3 | 0.9649 | 0.0018 | 0.9288 | 0.0028 |
| Toys | same_brsm | 3 | 0.8037 | 0.0036 | 0.7715 | 0.0002 |
| Toys | same_brsm_mrc | 3 | 0.8041 | 0.0031 | 0.7785 | 0.0069 |
| Toys | same_scalar_v1 | 3 | 0.8041 | 0.0030 | 0.7770 | 0.0075 |
| Toys | self_brsm | 3 | 0.8042 | 0.0022 | 0.7759 | 0.0051 |
| ele-fashion | same_brsm | 3 | 0.8787 | 0.0012 | 0.7625 | 0.0081 |
| ele-fashion | same_brsm_mrc | 3 | 0.8808 | 0.0009 | 0.7640 | 0.0072 |
| ele-fashion | same_scalar_v1 | 3 | 0.8806 | 0.0018 | 0.7659 | 0.0076 |
| ele-fashion | self_brsm | 3 | 0.8788 | 0.0009 | 0.7605 | 0.0054 |

## 3. Paired comparisons

| comparison | dataset | metric | N | mean | std | median | positive_fraction | zero_fraction | negative_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same_brsm-same_scalar_v1 | ALL | val_acc | 15 | -0.0006 | 0.0014 | -0.0006 | 0.2000 | 0.1333 | 0.6667 |
| same_brsm-same_scalar_v1 | ALL | val_macro_f1 | 15 | -0.0017 | 0.0081 | -0.0015 | 0.4000 | 0.0000 | 0.6000 |
| same_brsm-self_brsm | ALL | val_acc | 15 | 0.0004 | 0.0014 | 0.0003 | 0.6667 | 0.0667 | 0.2667 |
| same_brsm-self_brsm | ALL | val_macro_f1 | 15 | 0.0023 | 0.0083 | 0.0002 | 0.5333 | 0.0667 | 0.4000 |
| same_brsm_mrc-same_brsm | ALL | val_acc | 15 | 0.0003 | 0.0020 | 0.0000 | 0.4667 | 0.1333 | 0.4000 |
| same_brsm_mrc-same_brsm | ALL | val_macro_f1 | 15 | -0.0002 | 0.0070 | -0.0009 | 0.4667 | 0.0000 | 0.5333 |
| self_brsm-Plain | ALL | val_acc | 15 | 0.0009 | 0.0032 | 0.0006 | 0.6667 | 0.0000 | 0.3333 |
| self_brsm-Plain | ALL | val_macro_f1 | 15 | -0.0005 | 0.0120 | 0.0016 | 0.6667 | 0.0000 | 0.3333 |

Per-seed paired rows are in `outputs/cssi_p4/paired_differences.csv`; dataset-level rows are retained there as well.

## 4. BRSM mechanism diagnostics

Rows=360; finite=360; all validation response/correction rows are grouped by dataset, seed, modality, and order. `lambda` is global per modality/order, while amplitude statistics are node-level.

| variant | modality | order | lambda | correction_ratio_mean | correction_cosine_mean | amplitude_mean | amplitude_std | amplitude_positive_fraction | amplitude_negative_fraction | amplitude_node_dispersion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same_brsm | text | 1 | 0.0529 | 0.0168 | -0.0664 | -0.1571 | 0.8798 | 0.4231 | 0.5769 | 1.0807 |
| same_brsm | text | 2 | 0.0494 | 0.0161 | -0.0586 | -0.1868 | 0.8813 | 0.4085 | 0.5915 | 1.0882 |
| same_brsm | text | 3 | 0.0522 | 0.0176 | -0.0490 | -0.1590 | 0.8818 | 0.4224 | 0.5776 | 0.9996 |
| same_brsm | visual | 1 | 0.0530 | 0.0147 | 0.0053 | 0.0654 | 0.7942 | 0.5345 | 0.4655 | 1.3076 |
| same_brsm | visual | 2 | 0.0512 | 0.0149 | 0.0106 | 0.0351 | 0.7918 | 0.5194 | 0.4806 | 1.1910 |
| same_brsm | visual | 3 | 0.0523 | 0.0158 | 0.0200 | 0.0593 | 0.7842 | 0.5315 | 0.4685 | 1.1558 |
| same_brsm_mrc | text | 1 | 0.0534 | 0.0169 | -0.0654 | -0.1306 | 0.8891 | 0.4357 | 0.5643 | 1.1799 |
| same_brsm_mrc | text | 2 | 0.0503 | 0.0162 | -0.0760 | -0.1704 | 0.8827 | 0.4157 | 0.5843 | 1.1939 |
| same_brsm_mrc | text | 3 | 0.0527 | 0.0174 | -0.0689 | -0.1402 | 0.8856 | 0.4311 | 0.5689 | 1.1169 |
| same_brsm_mrc | visual | 1 | 0.0535 | 0.0152 | 0.0217 | 0.1446 | 0.7835 | 0.5731 | 0.4269 | 1.3561 |
| same_brsm_mrc | visual | 2 | 0.0520 | 0.0151 | 0.0153 | 0.1150 | 0.7818 | 0.5583 | 0.4417 | 1.3028 |
| same_brsm_mrc | visual | 3 | 0.0528 | 0.0158 | 0.0186 | 0.1344 | 0.7708 | 0.5676 | 0.4324 | 1.2243 |
| same_scalar_v1 | text | 1 | 0.0522 | 0.0487 | -0.9220 | -0.8608 | 0.1501 | 0.0389 | 0.9611 | 0.1143 |
| same_scalar_v1 | text | 2 | 0.0506 | 0.0472 | -0.9502 | -0.8904 | 0.1273 | 0.0248 | 0.9752 | 0.0821 |
| same_scalar_v1 | text | 3 | 0.0521 | 0.0487 | -0.9517 | -0.8975 | 0.1200 | 0.0211 | 0.9789 | 0.0723 |
| same_scalar_v1 | visual | 1 | 0.0524 | 0.0516 | -0.1959 | -0.1963 | 0.2686 | 0.4017 | 0.5983 | 0.1897 |
| same_scalar_v1 | visual | 2 | 0.0521 | 0.0513 | -0.2285 | -0.2299 | 0.2216 | 0.3854 | 0.6146 | 0.1484 |
| same_scalar_v1 | visual | 3 | 0.0523 | 0.0516 | -0.2239 | -0.2300 | 0.1581 | 0.3852 | 0.6148 | 0.1011 |
| self_brsm | text | 1 | 0.0528 | 0.0160 | -0.0648 | -0.1979 | 0.8145 | 0.4004 | 0.5996 | 1.0005 |
| self_brsm | text | 2 | 0.0497 | 0.0155 | -0.0551 | -0.2221 | 0.8033 | 0.3880 | 0.6120 | 0.9522 |
| self_brsm | text | 3 | 0.0523 | 0.0168 | -0.0500 | -0.2111 | 0.8068 | 0.3935 | 0.6065 | 0.8632 |
| self_brsm | visual | 1 | 0.0531 | 0.0139 | 0.0082 | 0.0747 | 0.6939 | 0.5410 | 0.4590 | 1.5228 |
| self_brsm | visual | 2 | 0.0516 | 0.0142 | 0.0125 | 0.0490 | 0.6718 | 0.5292 | 0.4708 | 1.3738 |
| self_brsm | visual | 3 | 0.0523 | 0.0149 | 0.0196 | 0.0712 | 0.6623 | 0.5408 | 0.4592 | 1.3460 |
Same-BRSM has a genuinely mixed node-level controller response: mean positive/negative amplitude fractions are 0.473/0.527, with mean node dispersion 1.137.
A caveat is that Self-BRSM text control is heterogeneous rather than uniformly helpful: mean amplitude=-0.210, negative fraction=0.606, and node dispersion=0.939; the dataset-level negative fraction ranges from 0.471 to 0.680.
The scalar control is also strongly suppressive on average: mean amplitude=-0.551, mean correction cosine=-0.579; this is an observed diagnostic, not a constraint imposed on BRSM.

## 5. Cross-modal interventions

| intervention | val_acc | val_macro_f1 | mean_z_l2_shift | mean_z_cosine_to_normal | mean_modulation_vector_shift | mean_correction_l2_shift |
| --- | --- | --- | --- | --- | --- | --- |
| normal | 0.8126 | 0.7487 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| off | 0.8123 | 0.7484 | 0.1234 | 0.9999 | 1.0107 | 0.0187 |
| shuffle | 0.8120 | 0.7484 | 0.1503 | 0.9998 | 1.0159 | 0.0211 |
Cross-Off uses an exact zero paired-control path; Cross-Shuffle permutes the other modality's node tokens over all nodes. Representation changes with unchanged task decisions are reported as such.

## 6. MRC diagnostics with strict raw semantic gap

| variant | dataset | seed | raw_semantic_gap_mean | raw_weight_gap_pearson | neighbor_allocation_tv_divergence | top_neighbor_disagreement | incident_mass_abs_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| same_brsm_mrc | Movies | 42 | 0.2287 | 0.9429 | 0.0165 | 0.5816 | 1.0506 |
| same_brsm_mrc | Movies | 43 | 0.2300 | 0.9399 | 0.0172 | 0.5804 | 1.0648 |
| same_brsm_mrc | Movies | 44 | 0.2249 | 0.9160 | 0.0207 | 0.5825 | 0.9743 |
| same_brsm_mrc | Toys | 42 | 0.1512 | 0.9466 | 0.0134 | 0.4712 | 0.3424 |
| same_brsm_mrc | Toys | 43 | 0.1513 | 0.9472 | 0.0128 | 0.4716 | 0.3434 |
| same_brsm_mrc | Toys | 44 | 0.1497 | 0.9449 | 0.0134 | 0.4683 | 0.3416 |
| same_brsm_mrc | Grocery | 42 | 0.1553 | 0.9403 | 0.0164 | 0.4941 | 0.5750 |
| same_brsm_mrc | Grocery | 43 | 0.1528 | 0.9424 | 0.0157 | 0.4957 | 0.5125 |
| same_brsm_mrc | Grocery | 44 | 0.1541 | 0.9379 | 0.0163 | 0.4912 | 0.5418 |
| same_brsm_mrc | ele-fashion | 42 | 0.1546 | 0.9512 | 0.0088 | 0.2980 | 0.3126 |
| same_brsm_mrc | ele-fashion | 43 | 0.1554 | 0.9576 | 0.0087 | 0.2984 | 0.3176 |
| same_brsm_mrc | ele-fashion | 44 | 0.1518 | 0.9553 | 0.0084 | 0.2983 | 0.3006 |
| same_brsm_mrc | Reddit-S | 42 | 0.1888 | 0.9438 | 0.0104 | 0.5075 | 1.4986 |
| same_brsm_mrc | Reddit-S | 43 | 0.2251 | 0.9359 | 0.0125 | 0.5047 | 2.0285 |
| same_brsm_mrc | Reddit-S | 44 | 0.1883 | 0.9414 | 0.0106 | 0.5103 | 1.5059 |
`raw_semantic_gap` is computed from cosine(H0_i,H0_j) before the learned diagonal metric. The old P3 label `incident_mean_abs_gap` is not used; the quantity is named `incident_mass_abs_gap` because it compares incident weight totals.

## 7. Stability

Training monitor rows=60, textual NaN/Inf rows=0. Controller/basis/lambda gradient norms and correction ratios are in `outputs/cssi_p4/controller_diagnostics.csv` and `outputs/cssi_p4/training_stability.csv`.
| variant | controller_gradient_norm_first | basis_gradient_norm_first | lambda_gradient_norm_first | mean_lambda_text_first | mean_lambda_text_last | mean_lambda_visual_first | mean_lambda_visual_last | mean_response_correction_ratio_first | max_response_correction_ratio_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| same_brsm | 0.0012 | 0.0014 | 0.0001 | 0.0500 | 0.0523 | 0.0500 | 0.0532 | 0.0027 | 0.0187 |
| same_brsm_mrc | 0.0012 | 0.0014 | 0.0000 | 0.0500 | 0.0532 | 0.0500 | 0.0539 | 0.0027 | 0.0176 |
| same_scalar_v1 | 0.0009 | 0.0000 | 0.0000 | 0.0500 | 0.0523 | 0.0500 | 0.0527 | 0.0034 | 0.0537 |
| self_brsm | 0.0014 | 0.0018 | 0.0001 | 0.0500 | 0.0524 | 0.0500 | 0.0533 | 0.0037 | 0.0179 |

## 8. Required P4 answers

1. P3 gradient starvation: yes at initialization; the zero up projection blocks conditioner and response-down gradients, and the fixed 1e-3 scale makes the path under-active.
2. BRSM correction: the transformation removes that structural zero-gradient/output-bias issue and enforces zero-response and norm-bound properties; whether it improves task metrics is answered empirically below.
3. self_brsm-Plain: mean paired Δ=0.000881; win/tie/loss=0.667/0.000/0.333.
4. same_brsm-self_brsm: mean paired Δ=0.000438; win/tie/loss=0.667/0.067/0.267.
5. same_brsm-same_scalar_v1: mean paired Δ=-0.000587; win/tie/loss=0.200/0.133/0.667.
6. same_brsm_mrc-same_brsm: mean paired Δ=0.000323; win/tie/loss=0.467/0.133/0.400.
7. Cross-Off/Shuffle: mean normal-minus-off accuracy=0.000283; normal-minus-shuffle accuracy=0.000566; modulation-vector shifts off/shuffle=1.010733/1.015934; correction shifts=0.018714/0.021131.
8. Model freeze decision: do not freeze a final CSSI mechanism from P4. BRSM is a stable candidate and fixes the P3 structural initialization problem, but Self-BRSM does not improve Macro-F1, Same-BRSM gains over Self-BRSM are small, BRSM is below the scalar control on both aggregate Accuracy and Macro-F1, MRC is mixed, and cross interventions change modulation/representations much more than task decisions.

Positive paired difference means the first named model is better. All utility/mechanism statements are frozen-forward functional diagnostics, not causal effects.
