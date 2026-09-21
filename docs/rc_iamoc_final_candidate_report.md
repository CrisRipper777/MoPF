# RC-IAMOC Final Candidate Consolidation

## Scope and protocol

R1 denotes Relation-Conditioned Interaction-Aware Multi-Order Composition (RC-IAMOC). The frozen R1 configuration uses one single-head hop-interaction layer, order embeddings, rank-1 relation state, relation scale initialization 0.10, `relation_conditioning=none`, and `ppc_weight=0`. The model's inherited Stage I/II, fusion, and NC classifier settings match the completed R1 configuration. `relation_conditioning=none` disables the relation output path; it leaves the inherited transport residual used by this model intact.

The added training runs cover Toys, ele-fashion, and Reddit-S with seeds 42/43/44. Movies and Grocery R1 checkpoints were reused. Formal V0 and IAMOC core V2/R0 baselines were reused from the existing five-dataset NC grid and its audited legacy Movies/Grocery runs. All seed summaries below use population standard deviation. No LP or hyperparameter search was run.

## A. Five-dataset NC generalization

Test Accuracy and Test Macro-F1 are reported as mean ± population SD across seeds. Validation metrics, epoch and source paths are retained in the result CSVs.

| Dataset | Model | Test Accuracy | Test Macro-F1 |
|---|---|---:|---:|
| Movies | Formal V0 | 0.5582 ± 0.0044 | 0.4988 ± 0.0048 |
| Movies | IAMOC core V2/R0 | 0.5640 ± 0.0042 | 0.5045 ± 0.0056 |
| Movies | R1 | 0.5640 ± 0.0055 | 0.5015 ± 0.0043 |
| Grocery | Formal V0 | 0.8305 ± 0.0047 | 0.7514 ± 0.0135 |
| Grocery | IAMOC core V2/R0 | 0.8335 ± 0.0023 | 0.7604 ± 0.0067 |
| Grocery | R1 | 0.8310 ± 0.0043 | 0.7584 ± 0.0105 |
| Toys | Formal V0 | 0.8022 ± 0.0050 | 0.7734 ± 0.0045 |
| Toys | IAMOC core V2/R0 | 0.7995 ± 0.0054 | 0.7702 ± 0.0115 |
| Toys | R1 | 0.7987 ± 0.0060 | 0.7745 ± 0.0059 |
| ele-fashion | Formal V0 | 0.8816 ± 0.0020 | 0.7731 ± 0.0109 |
| ele-fashion | IAMOC core V2/R0 | 0.8826 ± 0.0008 | 0.7749 ± 0.0009 |
| ele-fashion | R1 | 0.8822 ± 0.0023 | 0.7733 ± 0.0052 |
| Reddit-S | Formal V0 | 0.9653 ± 0.0042 | 0.9268 ± 0.0094 |
| Reddit-S | IAMOC core V2/R0 | 0.9663 ± 0.0021 | 0.9290 ± 0.0065 |
| Reddit-S | R1 | 0.9666 ± 0.0017 | 0.9294 ± 0.0056 |

Validation Accuracy and Macro-F1 are included as a separate checkpoint-selection view:

| Dataset | Model | Val Accuracy | Val Macro-F1 |
|---|---|---:|---:|
| Movies | Formal V0 | 0.5762 ± 0.0024 | 0.4984 ± 0.0076 |
| Movies | IAMOC core V2/R0 | 0.5795 ± 0.0059 | 0.5086 ± 0.0116 |
| Movies | R1 | 0.5770 ± 0.0026 | 0.5053 ± 0.0099 |
| Grocery | Formal V0 | 0.8377 ± 0.0015 | 0.7639 ± 0.0087 |
| Grocery | IAMOC core V2/R0 | 0.8386 ± 0.0036 | 0.7704 ± 0.0095 |
| Grocery | R1 | 0.8403 ± 0.0003 | 0.7778 ± 0.0074 |
| Toys | Formal V0 | 0.8041 ± 0.0011 | 0.7755 ± 0.0010 |
| Toys | IAMOC core V2/R0 | 0.8033 ± 0.0014 | 0.7748 ± 0.0080 |
| Toys | R1 | 0.8050 ± 0.0011 | 0.7803 ± 0.0046 |
| ele-fashion | Formal V0 | 0.8811 ± 0.0008 | 0.7679 ± 0.0087 |
| ele-fashion | IAMOC core V2/R0 | 0.8820 ± 0.0006 | 0.7678 ± 0.0006 |
| ele-fashion | R1 | 0.8819 ± 0.0009 | 0.7668 ± 0.0068 |
| Reddit-S | Formal V0 | 0.9642 ± 0.0031 | 0.9279 ± 0.0053 |
| Reddit-S | IAMOC core V2/R0 | 0.9651 ± 0.0025 | 0.9279 ± 0.0035 |
| Reddit-S | R1 | 0.9653 ± 0.0023 | 0.9281 ± 0.0042 |

Paired seed deltas are R1 minus the named baseline. Positive/neutral/negative cell counts classify each of the 10 dataset × test-metric mean deltas against a neutral band of ±1 baseline population SD; the per-seed deltas remain available in `performance/paired_seed_deltas.csv`.

| Baseline | Positive | Neutral | Negative |
|---|---:|---:|---:|
| Formal V0 | 1 | 9 | 0 |
| IAMOC core V2/R0 | 0 | 8 | 2 |

| Dataset | Baseline | Metric | Seed 42 Δ | Seed 43 Δ | Seed 44 Δ | Mean Δ ± population SD | Class |
|---|---|---|---:|---:|---:|---:|---|
| Movies | Formal V0 | val_acc | +0.0024 | -0.0015 | +0.0015 | +0.0008 ± 0.0017 | neutral |
| Movies | Formal V0 | val_macro_f1 | +0.0224 | -0.0030 | +0.0013 | +0.0069 ± 0.0111 | neutral |
| Movies | Formal V0 | test_acc | +0.0030 | +0.0069 | +0.0075 | +0.0058 ± 0.0020 | positive |
| Movies | Formal V0 | test_macro_f1 | +0.0041 | +0.0025 | +0.0016 | +0.0028 ± 0.0010 | neutral |
| Grocery | Formal V0 | val_acc | +0.0038 | +0.0003 | +0.0038 | +0.0026 ± 0.0017 | positive |
| Grocery | Formal V0 | val_macro_f1 | +0.0146 | +0.0027 | +0.0246 | +0.0139 ± 0.0089 | positive |
| Grocery | Formal V0 | test_acc | +0.0006 | +0.0026 | -0.0015 | +0.0006 ± 0.0017 | neutral |
| Grocery | Formal V0 | test_macro_f1 | +0.0160 | +0.0143 | -0.0096 | +0.0069 ± 0.0117 | neutral |
| Toys | Formal V0 | val_acc | +0.0007 | -0.0005 | +0.0024 | +0.0009 ± 0.0012 | neutral |
| Toys | Formal V0 | val_macro_f1 | +0.0054 | -0.0011 | +0.0100 | +0.0048 ± 0.0045 | positive |
| Toys | Formal V0 | test_acc | -0.0065 | -0.0041 | +0.0000 | -0.0035 ± 0.0027 | neutral |
| Toys | Formal V0 | test_macro_f1 | -0.0004 | -0.0049 | +0.0085 | +0.0011 ± 0.0056 | neutral |
| ele-fashion | Formal V0 | val_acc | +0.0020 | +0.0008 | -0.0007 | +0.0007 ± 0.0011 | neutral |
| ele-fashion | Formal V0 | val_macro_f1 | +0.0195 | -0.0100 | -0.0130 | -0.0012 ± 0.0146 | neutral |
| ele-fashion | Formal V0 | test_acc | +0.0050 | -0.0002 | -0.0031 | +0.0006 ± 0.0034 | neutral |
| ele-fashion | Formal V0 | test_macro_f1 | +0.0216 | -0.0139 | -0.0071 | +0.0002 ± 0.0154 | neutral |
| Reddit-S | Formal V0 | val_acc | +0.0025 | +0.0000 | +0.0006 | +0.0010 ± 0.0011 | neutral |
| Reddit-S | Formal V0 | val_macro_f1 | +0.0032 | -0.0048 | +0.0021 | +0.0002 ± 0.0035 | neutral |
| Reddit-S | Formal V0 | test_acc | +0.0050 | +0.0035 | -0.0047 | +0.0013 ± 0.0043 | neutral |
| Reddit-S | Formal V0 | test_macro_f1 | +0.0108 | +0.0058 | -0.0087 | +0.0026 ± 0.0083 | neutral |
| Movies | IAMOC core V2/R0 | val_acc | +0.0030 | -0.0042 | -0.0063 | -0.0025 ± 0.0040 | neutral |
| Movies | IAMOC core V2/R0 | val_macro_f1 | +0.0146 | -0.0099 | -0.0147 | -0.0033 ± 0.0128 | neutral |
| Movies | IAMOC core V2/R0 | test_acc | -0.0060 | +0.0024 | +0.0036 | +0.0000 ± 0.0043 | neutral |
| Movies | IAMOC core V2/R0 | test_macro_f1 | +0.0005 | +0.0018 | -0.0111 | -0.0029 ± 0.0058 | neutral |
| Grocery | IAMOC core V2/R0 | val_acc | +0.0009 | -0.0020 | +0.0064 | +0.0018 ± 0.0035 | neutral |
| Grocery | IAMOC core V2/R0 | val_macro_f1 | +0.0056 | -0.0017 | +0.0182 | +0.0074 ± 0.0082 | neutral |
| Grocery | IAMOC core V2/R0 | test_acc | +0.0009 | -0.0029 | -0.0053 | -0.0024 ± 0.0025 | negative |
| Grocery | IAMOC core V2/R0 | test_macro_f1 | +0.0107 | -0.0041 | -0.0126 | -0.0020 ± 0.0097 | neutral |
| Toys | IAMOC core V2/R0 | val_acc | +0.0031 | +0.0019 | +0.0002 | +0.0018 ± 0.0012 | positive |
| Toys | IAMOC core V2/R0 | val_macro_f1 | +0.0146 | -0.0001 | +0.0018 | +0.0054 ± 0.0065 | neutral |
| Toys | IAMOC core V2/R0 | test_acc | -0.0027 | -0.0017 | +0.0017 | -0.0009 ± 0.0019 | neutral |
| Toys | IAMOC core V2/R0 | test_macro_f1 | +0.0133 | -0.0022 | +0.0017 | +0.0043 ± 0.0066 | neutral |
| ele-fashion | IAMOC core V2/R0 | val_acc | -0.0003 | +0.0016 | -0.0018 | -0.0002 ± 0.0014 | neutral |
| ele-fashion | IAMOC core V2/R0 | val_macro_f1 | +0.0078 | -0.0010 | -0.0099 | -0.0010 ± 0.0072 | negative |
| ele-fashion | IAMOC core V2/R0 | test_acc | +0.0003 | +0.0016 | -0.0031 | -0.0004 ± 0.0020 | neutral |
| ele-fashion | IAMOC core V2/R0 | test_macro_f1 | +0.0065 | -0.0033 | -0.0081 | -0.0016 ± 0.0061 | negative |
| Reddit-S | IAMOC core V2/R0 | val_acc | +0.0006 | +0.0000 | +0.0000 | +0.0002 ± 0.0003 | neutral |
| Reddit-S | IAMOC core V2/R0 | val_macro_f1 | +0.0012 | -0.0000 | -0.0006 | +0.0002 ± 0.0007 | neutral |
| Reddit-S | IAMOC core V2/R0 | test_acc | +0.0009 | -0.0003 | +0.0000 | +0.0002 ± 0.0005 | neutral |
| Reddit-S | IAMOC core V2/R0 | test_macro_f1 | +0.0019 | -0.0007 | -0.0001 | +0.0004 ± 0.0011 | neutral |

## B. R1 parameter-learning audit

The initialization vector is reconstructed from the recorded fixed seed and initialization standard deviation. The effective profile is `center(beta_raw) / RMS(center(beta_raw))`, matching the R1 forward calculation. Relative L2 change is computed on that effective profile; raw-parameter change and both raw/profile cosine values are also in the JSON. Relation scales are sigmoid-transformed parameters.

| Dataset | Modality | Median cosine(initial, final) | Median effective-profile relative L2 change | Relation scale initial → final (mean over seeds) | Mean across-seed final-profile cosine |
|---|---|---:|---:|---:|---:|
| Movies | Text | 0.5603 | 0.9378 | 0.1000 → 0.1021 | 0.9961 |
| Movies | Visual | 0.5095 | 0.9905 | 0.1000 → 0.1007 | 0.7072 |
| Grocery | Text | 0.4157 | 1.0810 | 0.1000 → 0.1040 | 0.8534 |
| Grocery | Visual | 0.2255 | 1.2446 | 0.1000 → 0.1016 | 0.3105 |
| Toys | Text | 0.8631 | 0.5232 | 0.1000 → 0.1017 | 0.9011 |
| Toys | Visual | 0.6939 | 0.7825 | 0.1000 → 0.0988 | 0.8185 |
| ele-fashion | Text | 0.4427 | 1.0557 | 0.1000 → 0.1025 | 0.9758 |
| ele-fashion | Visual | -0.0452 | 1.4458 | 0.1000 → 0.1034 | 0.8097 |
| Reddit-S | Text | 0.7516 | 0.7048 | 0.1000 → 0.1025 | 0.9063 |
| Reddit-S | Visual | 0.3469 | 1.1429 | 0.1000 → 0.0992 | 0.8978 |

Using the declared learning diagnostic (median effective-profile relative L2 change > 0.10 and median initial/final profile cosine < 0.99), beta profile learned: **True**. Initial/final raw vectors and normalized profiles are in `parameters/beta_learning_audit.json`; order-wise CSV rows are in `parameters/beta_profiles.csv`.

## C. Hop-interaction functional interventions

Every intervention is inference-only and reloads the unchanged best checkpoint. Interaction-Off sets only the local hop gate to zero; Uniform-Attention replaces each query's key distribution with 1/(K+1); Query-Collapse broadcasts each node's query-mean key-mass profile across queries. Thus Query-Collapse retains the node-specific key-mass profile while removing query-dependent cross-order interaction. Attention and representation shifts use all nodes; test metrics use the frozen test split.

| Dataset | Intervention | Attention MAE (Text / Visual) | η MAE (Text / Visual) | Z MAE | Logit MAE | KL(Pnormal‖Pcondition) | Prediction flip (test) | Test Acc Δ | Test Macro-F1 Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | Interaction-Off | 8.764e-08 / 3.905e-08 | 0.002867 / 0.02529 | 0.0242 | 0.04904 | 0.003263 | 0.02189 | -0.001599 | -0.0006536 |
| Movies | Uniform-Attention | 0.09569 / 0.1608 | 7.871e-06 / 0.007817 | 0.00886 | 0.0182 | 0.001389 | 0.007396 | -0.001399 | -0.001174 |
| Movies | Query-Collapse | 0.005336 / 0.03032 | 3.108e-07 / 0.001765 | 0.002002 | 0.00404 | 0.000144 | 0.001399 | -9.995e-05 | +6.11e-05 |
| Grocery | Interaction-Off | 4.541e-08 / 4.799e-08 | 0.02908 / 0.02818 | 0.04685 | 0.1139 | 0.008657 | 0.01542 | -0.004002 | -0.004076 |
| Grocery | Uniform-Attention | 0.1727 / 0.2148 | 0.01674 / 0.01619 | 0.02825 | 0.06825 | 0.004745 | 0.01025 | -0.001659 | -0.002103 |
| Grocery | Query-Collapse | 0.02539 / 0.05769 | 0.002867 / 0.005634 | 0.00933 | 0.02241 | 0.0009193 | 0.003123 | -0.001074 | -0.0008265 |
| Toys | Interaction-Off | 4.059e-08 / 3.27e-08 | 0.002778 / 0.002775 | 0.002049 | 0.003948 | 1.399e-05 | 0.001289 | +0.0003221 | +0.0001924 |
| Toys | Uniform-Attention | 0.09398 / 0.1006 | 0.0002292 / 0.0002066 | 0.0002384 | 0.0004394 | 1.03e-06 | 8.053e-05 | +8.053e-05 | +0.0001045 |
| Toys | Query-Collapse | 0.01129 / 0.01307 | 4.191e-05 / 2.688e-05 | 7.15e-05 | 0.0001301 | 2.283e-07 | 0 | +0 | +0 |
| ele-fashion | Interaction-Off | 5.065e-08 / 3.881e-08 | 0.001999 / 0.01908 | 0.008979 | 0.02101 | 0.0001024 | 0.0007501 | -5.682e-05 | -0.0002607 |
| ele-fashion | Uniform-Attention | 0.1224 / 0.1843 | 0.0001823 / 0.006518 | 0.003433 | 0.008539 | 7.498e-05 | 0.0001932 | -1.136e-05 | -6.353e-05 |
| ele-fashion | Query-Collapse | 0.01535 / 0.03401 | 1.512e-05 / 0.001177 | 0.000478 | 0.001126 | 4.441e-06 | 5.682e-05 | +3.409e-05 | +0.0001035 |
| Reddit-S | Interaction-Off | 4.082e-08 / 4.905e-08 | 0.0164 / 0.009579 | 0.01043 | 0.0229 | 0.0002692 | 0.001258 | -0.0002097 | -0.0006438 |
| Reddit-S | Uniform-Attention | 0.09913 / 0.1145 | 0.002059 / 0.003297 | 0.002478 | 0.00559 | 0.0001153 | 0.0009437 | -0.0004194 | -0.0009054 |
| Reddit-S | Query-Collapse | 0.01014 / 0.01586 | 0.0001591 / 0.0002063 | 0.0001826 | 0.0003967 | 3.529e-06 | 0.0002097 | +0.0001049 | -1.121e-05 |

Normal attention structure averaged over seeds (node means are calculated before the seed summary):

| Dataset | Modality | KL to uniform | Normalized entropy | Max key mass | Query-row MAE | Pairwise query JS | r_attn mean ± node SD |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | Text | 0.1509 | 0.8911 | 0.3746 | 0.005336 | 0.0009236 | 0.6343 ± 0.1086 |
| Movies | Visual | 0.3283 | 0.7632 | 0.5348 | 0.03032 | 0.01655 | 0.4331 ± 0.2241 |
| Grocery | Text | 0.3803 | 0.7257 | 0.551 | 0.02539 | 0.01387 | 0.4463 ± 0.2114 |
| Grocery | Visual | 0.4803 | 0.6536 | 0.6214 | 0.05769 | 0.0472 | 0.5122 ± 0.2894 |
| Toys | Text | 0.1305 | 0.9059 | 0.3862 | 0.01129 | 0.003603 | 0.5863 ± 0.1342 |
| Toys | Visual | 0.1686 | 0.8784 | 0.426 | 0.01307 | 0.004238 | 0.4394 ± 0.1621 |
| ele-fashion | Text | 0.2271 | 0.8362 | 0.4751 | 0.01535 | 0.005009 | 0.6045 ± 0.1901 |
| ele-fashion | Visual | 0.3959 | 0.7144 | 0.5792 | 0.03401 | 0.01701 | 0.6146 ± 0.2382 |
| Reddit-S | Text | 0.1711 | 0.8766 | 0.4099 | 0.01014 | 0.002958 | 0.4750 ± 0.1493 |
| Reddit-S | Visual | 0.2354 | 0.8302 | 0.4305 | 0.01586 | 0.007898 | 0.4681 ± 0.1788 |

Direct Query-Collapse vs Uniform-Attention contrast (Uniform minus Query-Collapse for test metrics):

| Dataset | Attention MAE (Text / Visual) | η MAE (Text / Visual) | Z MAE | Logit MAE | KL(Pcollapse‖Puniform) | Test flip rate | Test Acc Δ | Test Macro-F1 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.09558 / 0.1553 | 7.764e-06 / 0.007467 | 0.008778 | 0.01804 | 0.001393 | 0.007596 | -0.001299 | -0.001235 |
| Grocery | 0.1688 / 0.1992 | 0.01613 / 0.01435 | 0.02504 | 0.06053 | 0.004007 | 0.008492 | -0.0005857 | -0.001277 |
| Toys | 0.09255 / 0.09934 | 0.0002099 / 0.0002041 | 0.0002154 | 0.000402 | 6.833e-07 | 8.053e-05 | +8.053e-05 | +0.0001045 |
| ele-fashion | 0.121 / 0.179 | 0.000182 / 0.006238 | 0.003325 | 0.008289 | 7.055e-05 | 0.0001818 | -4.546e-05 | -0.000167 |
| Reddit-S | 0.09833 / 0.1116 | 0.002037 / 0.003226 | 0.002423 | 0.005468 | 0.0001118 | 0.000734 | -0.0005243 | -0.0008942 |

Normal attention non-uniformity and query dependence are summarized in `mechanism/normal_attention_mechanism.csv`. Normal vs Query-Collapse measures the value of query-dependent cross-order interaction. Query-Collapse vs Uniform-Attention is separately recorded with `reference=Query-Collapse`; it measures the value of node-specific hop preference, because collapse keeps each node's averaged key mass. Performance changes are reported alongside tensor shifts, so attention changes alone are not interpreted as performance causation.

## D. Relation functional interventions

Relation-Off removes the explicit rank-1 relation bias. Relation-Shuffle applies the same fixed node permutation to the detached text and visual relation states while preserving the learned parameters and other computations. MAEs and KL are Normal-relative; `r_attn shift` is signed Normal minus intervention, with absolute node-mean shift also stored. Accuracy/F1 deltas are intervention minus Normal on test nodes.

| Dataset | Condition | Attention MAE (Text / Visual) | r_attn abs shift (Text / Visual) | η MAE (Text / Visual) | Z MAE | Logit MAE | Probability KL | Flip rate (test) | Test Acc Δ | Test Macro-F1 Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | Relation-Off | 0.0129 / 0.01462 | 0.01564 / 0.008842 | 3.547e-07 / 0.0003263 | 0.0003195 | 0.000646 | 2.063e-06 | 0.0002999 | +0 | +4.01e-05 |
| Movies | Relation-Shuffle | 0.01777 / 0.02018 | 0.0213 / 0.01235 | 4.558e-07 / 0.0004432 | 0.0004456 | 0.0008979 | 4.111e-06 | 0.0004998 | +0 | +0.0001047 |
| Grocery | Relation-Off | 0.0129 / 0.01245 | 0.01983 / 0.01404 | 0.0008797 / 0.001011 | 0.001555 | 0.003769 | 3.975e-05 | 0.0005857 | -3.701e-17 | +7.126e-05 |
| Grocery | Relation-Shuffle | 0.01848 / 0.01723 | 0.02831 / 0.01972 | 0.001249 / 0.001412 | 0.002276 | 0.005508 | 6.571e-05 | 0.0006833 | -9.761e-05 | +2.304e-05 |
| Toys | Relation-Off | 0.0137 / 0.0144 | 0.01208 / 0.008975 | 1.079e-05 / 9.756e-06 | 1.015e-05 | 1.902e-05 | 1.956e-09 | 0 | +0 | +0 |
| Toys | Relation-Shuffle | 0.01923 / 0.02013 | 0.01694 / 0.01273 | 1.456e-05 / 1.394e-05 | 1.374e-05 | 2.57e-05 | 2.021e-09 | 0 | +0 | +0 |
| ele-fashion | Relation-Off | 0.01383 / 0.01515 | 0.01891 / 0.02232 | 8.271e-06 / 0.0006408 | 0.0002965 | 0.0007348 | 1.154e-06 | 0 | +0 | +0 |
| ele-fashion | Relation-Shuffle | 0.01873 / 0.02062 | 0.02585 / 0.03045 | 1.128e-05 / 0.0009353 | 0.000452 | 0.001116 | 1.617e-06 | 0 | +0 | +0 |
| Reddit-S | Relation-Off | 0.01604 / 0.01313 | 0.01889 / 0.01606 | 0.0001198 / 5.516e-05 | 9.061e-05 | 0.0001988 | 4.705e-07 | 0 | +0 | +0 |
| Reddit-S | Relation-Shuffle | 0.02239 / 0.01834 | 0.02617 / 0.02249 | 0.000171 / 7.4e-05 | 0.0001246 | 0.0002735 | 4.739e-07 | 0 | +0 | +0 |

| Dataset | Quantity | Modality | Off impact | Shuffle impact | Shuffle / Off |
|---|---|---|---:|---:|---:|
| Movies | attention | text | 0.0129 | 0.01777 | 1.377 |
| Movies | attention | visual | 0.01462 | 0.02018 | 1.38 |
| Movies | Z_mae | fused | 0.0003195 | 0.0004456 | 1.395 |
| Movies | logit_mae | fused | 0.000646 | 0.0008979 | 1.39 |
| Grocery | attention | text | 0.0129 | 0.01848 | 1.432 |
| Grocery | attention | visual | 0.01245 | 0.01723 | 1.383 |
| Grocery | Z_mae | fused | 0.001555 | 0.002276 | 1.463 |
| Grocery | logit_mae | fused | 0.003769 | 0.005508 | 1.461 |
| Toys | attention | text | 0.0137 | 0.01923 | 1.403 |
| Toys | attention | visual | 0.0144 | 0.02013 | 1.398 |
| Toys | Z_mae | fused | 1.015e-05 | 1.374e-05 | 1.354 |
| Toys | logit_mae | fused | 1.902e-05 | 2.57e-05 | 1.351 |
| ele-fashion | attention | text | 0.01383 | 0.01873 | 1.355 |
| ele-fashion | attention | visual | 0.01515 | 0.02062 | 1.361 |
| ele-fashion | Z_mae | fused | 0.0002965 | 0.000452 | 1.524 |
| ele-fashion | logit_mae | fused | 0.0007348 | 0.001116 | 1.519 |
| Reddit-S | attention | text | 0.01604 | 0.02239 | 1.395 |
| Reddit-S | attention | visual | 0.01313 | 0.01834 | 1.397 |
| Reddit-S | Z_mae | fused | 9.061e-05 | 0.0001246 | 1.375 |
| Reddit-S | logit_mae | fused | 0.0001988 | 0.0002735 | 1.375 |

A Shuffle impact above Off means the correct node-to-relation-state assignment carries functional information for that measured output. This remains an inference intervention result and does not establish training-time or downstream causal benefit by itself.

## E. Figure-ready data and diagnostic plots

Figure data use all nodes and all three seeds. Figure 1 reports mean Text/Visual query-by-key attention matrices for all datasets. Figure 2 has per-node `r_attn` values. Figure 3 contains relation-state quartile rows and attended-order profiles. Figure 4 contains Normal/Off/Shuffle attention, Z and logit shifts. Figure 5 contains Normal/Query-Collapse/Uniform/Interaction-Off degradation and performance deltas.

Quick diagnostic plots:

- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure1_mean_hop_attention.png`
- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure2_node_r_attn_distributions.png`
- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure3_relation_state_quartiles.png`
- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure4_relation_intervention_shifts.png`
- `outputs/rc_iamoc_final_candidate/figures/diagnostics/figure5_hop_intervention_degradation.png`

Text-vs-Visual and between-dataset matrix distances are in `figures/figure1_text_visual_dataset_contrasts.csv`. All plotted source rows are CSV files alongside the figure data.

## F. Candidate decision

**FINAL ARCHITECTURE CANDIDATE**

- Learned beta profile: True (median effective-profile relative L2 change 0.9401; median initial/final cosine 0.5581).
- Non-uniform hop attention across all runs and modalities: True.
- Query-Collapse changed Z or logits beyond numerical tolerance in 15/15 runs.
- Relation-Off changed Z or logits beyond numerical tolerance in 15/15 runs.
- Shuffle impact exceeded Off in 20/20 dataset/quantity ratios; the 'usually' criterion is True.
- Text/Visual and between-dataset attention profiles differ under the declared matrix-MAE diagnostic: True.
- R1 negative cells beyond the baseline-SD band: 0/10 vs Formal V0 and 2/10 vs core V2/R0.

Interpretation: the neutral band is descriptive and based on the baseline's three-seed population SD; it is not a hypothesis test. The intervention contrasts establish sensitivity of the frozen forward computation to the specified functional changes, and do not imply that a plotted attention pattern alone improves NC performance.

## G. Reproducibility files

All newly generated results are stored under `outputs/rc_iamoc_final_candidate/`; the nine additional run directories record their exact resolved configuration and checkpoint. `final_candidate_audit.json` contains the combined decision evidence. Existing IAMOC v1/v2 and generalization trees were read only.
