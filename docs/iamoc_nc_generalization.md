# IAMOC Core Generalization Across Five NC Datasets

## 1. Protocol

Node classification was evaluated on Movies, Toys, Grocery, ele-fashion, and Reddit-S with seeds 42/43/44. Runs use the frozen full-graph NC protocol (`unified_full_graph_nc_v1`), 300 maximum epochs, AdamW, validation-accuracy checkpoint selection, and the existing data splits. Reported seed summaries use population standard deviation. No tuning or test-driven model changes were used.

Existing Movies/Grocery V0/V1/V2 runs were reused only after checking the saved resolved configuration against the frozen formal MoPF and NC task configuration, the variant settings, seed, split path and pre-run split-file timestamp, protocol, checkpoint-selection rule, metrics, and checkpoint. Their source files remain in `outputs/iamoc_v1/`. New runs and all new artifacts are under `outputs/iamoc_nc_generalization/`; resolved split paths and current SHA-256 checksums are in `analysis/split_manifest.csv`.

## 2. 2×2 Factorial Design

| Variant | TCPR | Hop interaction | Definition |
|---|---|---|---|
| V00 | Off | Off | MoPF, `use_transport_residual=false` |
| V0 | On | Off | Formal MoPF |
| V1 | On | On | IAMOC, one interaction layer, `relation_conditioning=output` |
| V2 | Off | On | IAMOC, one interaction layer, `relation_conditioning=none` |

## 3. Five-Dataset Main Results

Values are mean ± population SD across the three seeds; best epochs are listed in seed order 42/43/44.

| Dataset | Variant | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Best epoch (42/43/44) |
|---|---|---:|---:|---:|---:|---|
| Movies | V00 | 0.5761 ± 0.0013 | 0.4897 ± 0.0106 | 0.5561 ± 0.0025 | 0.4918 ± 0.0044 | 67,84,81 |
| Movies | V0 | 0.5762 ± 0.0024 | 0.4984 ± 0.0076 | 0.5582 ± 0.0044 | 0.4988 ± 0.0048 | 63,74,66 |
| Movies | V1 | 0.5779 ± 0.0023 | 0.5053 ± 0.0048 | 0.5612 ± 0.0016 | 0.4956 ± 0.0064 | 81,76,75 |
| Movies | V2 | 0.5795 ± 0.0059 | 0.5086 ± 0.0116 | 0.5640 ± 0.0042 | 0.5045 ± 0.0056 | 71,78,90 |
| Toys | V00 | 0.8043 ± 0.0010 | 0.7736 ± 0.0013 | 0.8030 ± 0.0042 | 0.7716 ± 0.0050 | 46,55,62 |
| Toys | V0 | 0.8041 ± 0.0011 | 0.7755 ± 0.0010 | 0.8022 ± 0.0050 | 0.7734 ± 0.0045 | 46,42,52 |
| Toys | V1 | 0.8043 ± 0.0004 | 0.7752 ± 0.0034 | 0.7975 ± 0.0068 | 0.7668 ± 0.0075 | 43,46,54 |
| Toys | V2 | 0.8033 ± 0.0014 | 0.7748 ± 0.0080 | 0.7995 ± 0.0054 | 0.7702 ± 0.0115 | 44,43,48 |
| Grocery | V00 | 0.8370 ± 0.0025 | 0.7682 ± 0.0141 | 0.8293 ± 0.0029 | 0.7534 ± 0.0148 | 80,68,83 |
| Grocery | V0 | 0.8377 ± 0.0015 | 0.7639 ± 0.0087 | 0.8305 ± 0.0047 | 0.7514 ± 0.0135 | 71,65,85 |
| Grocery | V1 | 0.8375 ± 0.0040 | 0.7677 ± 0.0132 | 0.8328 ± 0.0043 | 0.7595 ± 0.0094 | 101,82,71 |
| Grocery | V2 | 0.8386 ± 0.0036 | 0.7704 ± 0.0095 | 0.8335 ± 0.0023 | 0.7604 ± 0.0067 | 119,81,84 |
| ele-fashion | V00 | 0.8808 ± 0.0014 | 0.7611 ± 0.0062 | 0.8805 ± 0.0012 | 0.7659 ± 0.0072 | 85,116,191 |
| ele-fashion | V0 | 0.8811 ± 0.0008 | 0.7679 ± 0.0087 | 0.8816 ± 0.0020 | 0.7731 ± 0.0109 | 92,200,132 |
| ele-fashion | V1 | 0.8826 ± 0.0011 | 0.7717 ± 0.0091 | 0.8828 ± 0.0005 | 0.7778 ± 0.0047 | 142,118,206 |
| ele-fashion | V2 | 0.8820 ± 0.0006 | 0.7678 ± 0.0006 | 0.8826 ± 0.0008 | 0.7749 ± 0.0009 | 174,120,131 |
| Reddit-S | V00 | 0.9646 ± 0.0029 | 0.9276 ± 0.0041 | 0.9657 ± 0.0019 | 0.9272 ± 0.0061 | 83,74,70 |
| Reddit-S | V0 | 0.9642 ± 0.0031 | 0.9279 ± 0.0053 | 0.9653 ± 0.0042 | 0.9268 ± 0.0094 | 39,73,65 |
| Reddit-S | V1 | 0.9657 ± 0.0025 | 0.9286 ± 0.0037 | 0.9666 ± 0.0015 | 0.9287 ± 0.0046 | 123,118,60 |
| Reddit-S | V2 | 0.9651 ± 0.0025 | 0.9279 ± 0.0035 | 0.9663 ± 0.0021 | 0.9290 ± 0.0065 | 100,83,56 |

## 4. Paired Seed Deltas

Each paired value is the first variant minus the second at the same seed. The table shows test metrics; all four metrics and all seed-level values are in `outputs/iamoc_nc_generalization/analysis/paired_seed_deltas.csv`.

| Dataset | Comparison | Test Acc Δ mean ± SD; positive seeds | Test Macro-F1 Δ mean ± SD; positive seeds |
|---|---|---:|---:|
| Movies | A_hop_without_TCPR (V2-V00) | 0.0079 ± 0.0017; 3/3 | 0.0126 ± 0.0042; 3/3 |
| Movies | B_hop_with_TCPR (V1-V0) | 0.0030 ± 0.0030; 2/3 | -0.0032 ± 0.0023; 0/3 |
| Movies | C_TCPR_without_hop (V0-V00) | 0.0021 ± 0.0029; 2/3 | 0.0070 ± 0.0073; 2/3 |
| Movies | D_TCPR_with_hop (V1-V2) | -0.0028 ± 0.0026; 1/3 | -0.0089 ± 0.0044; 0/3 |
| Toys | A_hop_without_TCPR (V2-V00) | -0.0035 ± 0.0013; 0/3 | -0.0013 ± 0.0107; 1/3 |
| Toys | B_hop_with_TCPR (V1-V0) | -0.0047 ± 0.0037; 0/3 | -0.0066 ± 0.0078; 1/3 |
| Toys | C_TCPR_without_hop (V0-V00) | -0.0008 ± 0.0009; 1/3 | 0.0019 ± 0.0026; 2/3 |
| Toys | D_TCPR_with_hop (V1-V2) | -0.0020 ± 0.0028; 1/3 | -0.0034 ± 0.0072; 1/3 |
| Grocery | A_hop_without_TCPR (V2-V00) | 0.0042 ± 0.0007; 3/3 | 0.0070 ± 0.0082; 2/3 |
| Grocery | B_hop_with_TCPR (V1-V0) | 0.0023 ± 0.0009; 3/3 | 0.0080 ± 0.0062; 2/3 |
| Grocery | C_TCPR_without_hop (V0-V00) | 0.0012 ± 0.0018; 2/3 | -0.0020 ± 0.0014; 0/3 |
| Grocery | D_TCPR_with_hop (V1-V2) | -0.0007 ± 0.0021; 1/3 | -0.0009 ± 0.0051; 1/3 |
| ele-fashion | A_hop_without_TCPR (V2-V00) | 0.0020 ± 0.0019; 3/3 | 0.0091 ± 0.0071; 2/3 |
| ele-fashion | B_hop_with_TCPR (V1-V0) | 0.0012 ± 0.0021; 2/3 | 0.0047 ± 0.0156; 2/3 |
| ele-fashion | C_TCPR_without_hop (V0-V00) | 0.0010 ± 0.0009; 2/3 | 0.0073 ± 0.0044; 3/3 |
| ele-fashion | D_TCPR_with_hop (V1-V2) | 0.0002 ± 0.0011; 2/3 | 0.0029 ± 0.0050; 2/3 |
| Reddit-S | A_hop_without_TCPR (V2-V00) | 0.0006 ± 0.0027; 2/3 | 0.0018 ± 0.0055; 2/3 |
| Reddit-S | B_hop_with_TCPR (V1-V0) | 0.0013 ± 0.0038; 2/3 | 0.0019 ± 0.0072; 2/3 |
| Reddit-S | C_TCPR_without_hop (V0-V00) | -0.0004 ± 0.0025; 2/3 | -0.0005 ± 0.0034; 2/3 |
| Reddit-S | D_TCPR_with_hop (V1-V2) | 0.0002 ± 0.0006; 2/3 | -0.0003 ± 0.0021; 2/3 |

The factorial effects below are calculated for each seed and metric: Hop main effect = ½[(V2−V00)+(V1−V0)], TCPR main effect = ½[(V0−V00)+(V1−V2)], and interaction = (V1−V0)−(V2−V00). These are descriptive decompositions; they are not statistical causal estimates.

Each summary reports the mean and median over the 15 dataset-seed effects, the number of positive dataset means, and the number of positive paired seed effects. CSV files retain all four metrics.

## 5. Hop Main Effect

| Metric | Effect | Mean Δ | Median Δ | + datasets / 5 | + seeds / 15 |
|---|---|---:|---:|---:|---:|
| test_acc | hop_main_effect | 0.00145 | 0.00264 | 4/5 | 10/15 |
| test_macro_f1 | hop_main_effect | 0.00340 | 0.00579 | 4/5 | 10/15 |

## 6. TCPR Main Effect

| Metric | Mean Δ | Median Δ | + datasets / 5 | + seeds / 15 |
|---|---:|---:|---:|---:|
| test_acc | -0.00020 | -0.00016 | 2/5 | 6/15 |
| test_macro_f1 | 0.00030 | 0.00118 | 1/5 | 8/15 |

## 7. Interaction Effect

| Metric | Mean Δ | Median Δ | + datasets / 5 | + seeds / 15 |
|---|---:|---:|---:|---:|
| test_acc | -0.00163 | -0.00117 | 1/5 | 3/15 |
| test_macro_f1 | -0.00485 | -0.00521 | 2/5 | 4/15 |

Per-dataset and per-seed decompositions for all four reported metrics are in `factorial_seed_effects.csv` and `factorial_summary.csv`; cross-dataset counts are in `factorial_across_datasets.csv`.

## 8. Mechanism Generalization

All 15 V2 best checkpoints were analyzed by inference only. KL is KL(attended order mass || uniform); normalized entropy divides entropy by log(number of orders). Query-row MAE and pairwise query JS compare query-order distributions within each node. Entries below average per-run node means, node standard deviations, and node quartiles across the three seeds.

| Dataset | Modality | KL(p‖uniform) | Normalized entropy | Max-key mass | Query-row MAE | Pairwise query JS | r_attn mean ± node SD | r_attn Q25 / median / Q75 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Movies | text | 0.1546 | 0.8885 | 0.3806 | 0.0059 | 0.0013 | 0.6428 ± 0.1034 | 0.5810 / 0.6395 / 0.7076 |
| Movies | visual | 0.3665 | 0.7357 | 0.5583 | 0.0319 | 0.0180 | 0.4301 ± 0.2402 | 0.2394 / 0.4428 / 0.6051 |
| Toys | text | 0.0905 | 0.9347 | 0.3526 | 0.0080 | 0.0019 | 0.5821 ± 0.0964 | 0.5260 / 0.5860 / 0.6473 |
| Toys | visual | 0.2050 | 0.8521 | 0.4460 | 0.0132 | 0.0044 | 0.3865 ± 0.1574 | 0.2783 / 0.4183 / 0.5012 |
| Grocery | text | 0.2528 | 0.8176 | 0.4789 | 0.0214 | 0.0092 | 0.5466 ± 0.1946 | 0.4086 / 0.5600 / 0.6975 |
| Grocery | visual | 0.4851 | 0.6501 | 0.6194 | 0.0479 | 0.0353 | 0.5231 ± 0.2878 | 0.2853 / 0.5360 / 0.7723 |
| ele-fashion | text | 0.2932 | 0.7885 | 0.5231 | 0.0213 | 0.0092 | 0.6104 ± 0.2116 | 0.4852 / 0.6229 / 0.7625 |
| ele-fashion | visual | 0.5316 | 0.6166 | 0.6463 | 0.0350 | 0.0190 | 0.5238 ± 0.2742 | 0.3146 / 0.5311 / 0.7470 |
| Reddit-S | text | 0.1359 | 0.9020 | 0.3709 | 0.0088 | 0.0023 | 0.4944 ± 0.1299 | 0.4066 / 0.5140 / 0.5981 |
| Reddit-S | visual | 0.1759 | 0.8731 | 0.3852 | 0.0134 | 0.0063 | 0.4917 ± 0.1560 | 0.4117 / 0.5348 / 0.6012 |

Order-embedding norm divided by mean propagation-state norm is averaged across seeds for each order:

| Dataset | Modality | Order 0 | Order 1 | Order 2 | Order 3 |
|---|---|---:|---:|---:|---:|
| Movies | text | 0.0167 | 0.0326 | 0.0362 | 0.0378 |
| Movies | visual | 0.0530 | 0.0600 | 0.0632 | 0.0691 |
| Toys | text | 0.0161 | 0.0257 | 0.0282 | 0.0298 |
| Toys | visual | 0.0244 | 0.0266 | 0.0260 | 0.0276 |
| Grocery | text | 0.0567 | 0.0664 | 0.0537 | 0.0544 |
| Grocery | visual | 0.0770 | 0.0837 | 0.0851 | 0.1253 |
| ele-fashion | text | 0.0467 | 0.0555 | 0.0629 | 0.0820 |
| ele-fashion | visual | 0.0746 | 0.0901 | 0.0816 | 0.1216 |
| Reddit-S | text | 0.0319 | 0.0351 | 0.0360 | 0.0362 |
| Reddit-S | visual | 0.0436 | 0.0528 | 0.0548 | 0.0552 |

## 9. Modality Heterogeneity

| Dataset | Mean-attention matrix MAE | Text − Visual mean r_attn | Text hop gate | Visual hop gate |
|---|---:|---:|---:|---:|
| Movies | 0.14489 | +0.21272 | 0.1452 | 0.1790 |
| Toys | 0.11952 | +0.19558 | 0.1248 | 0.1318 |
| Grocery | 0.07544 | +0.02353 | 0.1794 | 0.1893 |
| ele-fashion | 0.11611 | +0.08663 | 0.1499 | 0.1737 |
| Reddit-S | 0.01913 | +0.00264 | 0.1541 | 0.1526 |

Order-embedding norms, mean state norms, per-order ratios, and pairwise embedding cosine matrices are in `mechanism_order_embeddings.csv` and `results.json`. Per-node values are retained in compressed `mechanism_node_metrics.csv.gz`; average attention matrices are in `mean_attention_matrices.csv`.

## 10. OrderEmbedding-Off

This inference-only intervention removes the learned order embedding from V2 attention inputs by a temporary forward hook. Metrics are off minus normal for accuracy/F1; probability KL is KL(p_normal‖p_off). Repeated Stage-I/II outputs were checked at max absolute tolerance 1e-04; the largest observed difference was 5.341e-05, consistent with float32 sparse propagation roundoff. No checkpoint state is written or changed.

| Dataset | Attention MAE text / visual | Eta MAE text / visual | Z MAE | Logit MAE | Probability KL | Test flip rate | Test Acc Δ | Test Macro-F1 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 1.365e-02 / 7.771e-02 | 8.080e-06 / 5.426e-03 | 5.306e-03 | 1.050e-02 | 3.308e-04 | 0.0049 | +0.00050 | -0.00030 |
| Toys | 1.067e-02 / 1.772e-02 | 1.435e-04 / 9.088e-05 | 9.057e-05 | 1.660e-04 | 3.330e-08 | 0.0000 | +0.00000 | +0.00000 |
| Grocery | 5.977e-02 / 1.426e-01 | 4.536e-03 / 1.016e-02 | 1.218e-02 | 2.764e-02 | 7.935e-04 | 0.0041 | -0.00049 | -0.00012 |
| ele-fashion | 1.014e-01 / 1.611e-01 | 4.230e-04 / 1.187e-02 | 5.706e-03 | 1.323e-02 | 4.191e-05 | 0.0004 | -0.00013 | -0.00016 |
| Reddit-S | 9.606e-03 / 6.371e-02 | 9.221e-04 / 1.569e-03 | 1.012e-03 | 2.243e-03 | 1.734e-05 | 0.0003 | -0.00010 | -0.00012 |

## 11. Interim Scientific Interpretation

**A. IAMOC core across datasets.** Test-accuracy mean deltas for Hop Interaction were positive in 4/5 datasets without TCPR and 4/5 with TCPR. The corresponding test Macro-F1 positive-dataset counts are 4/5 and 3/5. This is a broad but non-universal mean gain, with Toys negative on both accuracy comparisons; seed-level variation is shown above.

**B. TCPR after Hop Interaction.** For V1−V2, TCPR had positive mean test-accuracy deltas in 2/5 datasets and positive Macro-F1 deltas in 1/5; the pooled 15-pair means were -0.00101 for accuracy and -0.00213 for Macro-F1. Without Hop Interaction (V0−V00), the corresponding pooled means were +0.00062 and +0.00273. Thus these results do not show a consistent added TCPR benefit once Hop Interaction is present. The factorial TCPR test-accuracy effect is -0.00020; interaction effect is -0.00163.

**C. Mechanism transfer.** Non-uniform attended order mass was present by the descriptive KL/entropy criteria in 30/30 dataset-seed-modality profiles; query-row MAE and pairwise query JS were nonzero in 30/30 profiles. Text and Visual mean attention matrices differed in 15/15 dataset-seed checkpoints, though the size of that gap varied by dataset. OrderEmbedding-Off changed Z or logits above 1e-8 in 5/5 dataset-level summaries; the detailed effect sizes and test metrics are in Section 10.

The design supports comparisons within the chosen datasets and seeds. The factorial terms are descriptive decompositions, and no statistical causal claim is made. Dataset-level signs should be read together with paired seed deltas and population SDs rather than as universal model guarantees.

## Machine-readable outputs

`analysis/results.json` contains the complete grid, paired deltas, factorial effects, mechanism summaries, and OrderEmbedding-Off results. CSV files in the same directory provide main results, seed-level deltas, decompositions, order-embedding diagnostics, modality heterogeneity, and the compressed node table.
