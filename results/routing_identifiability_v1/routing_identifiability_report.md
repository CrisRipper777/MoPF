# Experiment 6 - Structural Routing Identifiability & Joint Utility Audit

## Scope and protocol

Branch screening winner lock: `D0_independent`. C0 reload preflight covered 9/9 screening runs; maximum Validation Acc delta=5.96e-08, Macro-F1 delta=0. Screening and oracle diagnostics used Train/Validation only. No Test labels or metrics were used. Toys/Grocery were loaded only after the screening lock.

Formal router fits: 15 (9 D0 screening + 6 confirmation; D1=0). C0 retraining runs: 0.

## Phase A - canonicalization and teacher scale

E5 five-action outputs were mapped to the S0-S3 simplex. Both prescribed redundant routing vectors map to alpha_U=(0.25,0.25,0.25,0.25). Across the saved checks, maximum embedding error=4.68e-07 and logit error=3.34e-07, both below 1e-6.

| T | Mean entropy | Preference strength | Hard ranking invariant | JS(qT || q1) |
|---:|---:|---:|---:|---:|
| 0.10 | 1.3117 | 0.18497 | 0.9999 | 0.05113 |
| 0.25 | 1.4547 | 0.09617 | 1.0000 | 0.01753 |
| 0.50 | 1.5311 | 0.04865 | 1.0000 | 0.00370 |
| 1.00 | 1.5761 | 0.02072 | 1.0000 | 0.00000 |

Mean preference strength remains small at T=1 and increases as temperature falls; hard rankings remain nearly unchanged. Temperature was diagnostic only and did not affect training.

## Phase B - joint discrete landscape

Gate: **JOINT_UTILITY_WEAK_OR_MIXED**. The marginal pair regret was positive in 9/9 dataset-seeds and all 3 dataset means, but relative regret was 0.0448 (pre-registered materiality threshold 0.10); reproducible interaction criterion passed for 1/3 datasets.

| Dataset | Marginal-pair regret | Joint gain vs uniform | Relative regret | Marginal pair = joint best | Mean |I| | Mean interaction energy |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.01513 | 0.33693 | 0.0450 | 0.753 | 0.08522 | 0.0701 |
| ele-fashion | 0.00322 | 0.12345 | 0.0261 | 0.843 | 0.04365 | 0.1038 |
| Reddit-S | 0.00328 | 0.02253 | 0.1776 | 0.891 | 0.02600 | 0.0840 |

Across Train nodes, signed residual=0.00415, mean absolute residual=0.05162, RMS=0.07396, P90 |I|=0.10899. Mean interaction energy was 0.0860; 29.8% of nodes exceeded 0.1 and 6.9% exceeded 0.25.

Across-seed Train stability means: joint-best-pair agreement=0.462, text/visual best-response agreement=0.667/0.686, joint-gain Spearman=0.754, interaction-energy Spearman=0.410.

## Phase C - continuous simplex oracle

These are label-informed Train/Validation upper-bound diagnostics, not deployable router scores. The optimizer sanity check passed: mean positive violation=6.65e-08, maximum=1.14e-05.

| Solution | Train CE | Train Acc upper bound | Train Macro-F1 upper bound | Validation CE | Validation Acc upper bound | Validation Macro-F1 upper bound |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0.31343 | 0.8961 | 0.8424 | 0.65593 | 0.8042 | 0.7231 |
| best_joint_discrete | 0.15246 | 0.9519 | 0.9175 | 0.37248 | 0.8828 | 0.8212 |
| marginal_continuous_pair | 0.15069 | 0.9534 | 0.9198 | 0.37131 | 0.8829 | 0.8225 |
| joint_continuous | 0.14614 | 0.9553 | 0.9238 | 0.36521 | 0.8856 | 0.8258 |

Mean Train CE gain of joint continuous over best discrete=0.00632; by dataset: Movies 0.01613, Reddit-S 0.00148, ele-fashion 0.00135. Positive by the 1e-4 audit cutoff in 9/9 dataset-seeds. Joint continuous improved over the marginal continuous pair by 0.00455 mean CE.

| Modality | Entropy | Nearest-vertex L1 | L1 to uniform | Active orders | Top-2 mass | Near vertex (<0.10) | >=2 active | >=3 active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text | 0.1162 | 0.1028 | 1.4101 | 1.181 | 0.9808 | 87.4% | 10.9% | 4.6% |
| visual | 0.1278 | 0.1155 | 1.4004 | 1.199 | 0.9804 | 85.4% | 12.9% | 4.5% |

Across-seed Train alpha stability: cosine=0.764, JS=0.163, order-rank agreement=0.739; distance-to-uniform Spearman=0.274. Mean restart loss range=0.02839, P90=0.03772; report this restart sensitivity when interpreting oracle alpha.

## Phase D - canonical router and confirmation

D0 gate: **CANONICAL_ROUTER_PROMISING**. Mean paired screening deltas were Acc=0.00171, Macro-F1=0.00337; positive Acc seeds=8/9; positive dataset means=3/3. Zero-init embedding error and checkpoint reload alpha error were both 0 for all runs.

| Dataset | Mean delta Acc | Mean delta Macro-F1 | Positive seeds |
|---|---:|---:|---:|
| Movies | 0.00070 | 0.00044 | 2/3 |
| Reddit-S | 0.00304 | 0.00352 | 3/3 |
| ele-fashion | 0.00140 | 0.00614 | 3/3 |

| Intervention | Delta Acc vs normal | Delta Macro-F1 | Prediction flip | Mean |delta alpha| |
|---|---:|---:|---:|---:|
| train_mean_alpha | -0.00089 | -0.00123 | 0.00252 | 0.01379 |
| node_shuffle_10_seed_mean | -0.00130 | -0.00160 | 0.00389 | 0.01890 |
| uniform_alpha | -0.00171 | -0.00337 | 0.01205 | 0.05594 |

D1 was **not tested** because Phase B was JOINT_UTILITY_WEAK_OR_MIXED; no D1-vs-capacity-control result exists.

The screening lock selected **D0_independent** before loading untouched datasets. Confirmation completed on all 6 Toys/Grocery seed pairs: mean delta Acc=0.00006, Macro-F1=0.00008, positive Acc pairs=3/6.

| Confirmation dataset | Mean delta Acc | Mean delta Macro-F1 | Positive seeds |
|---|---:|---:|---:|
| Grocery | -0.00020 | 0.00004 | 0/3 |
| Toys | 0.00032 | 0.00012 | 3/3 |

Confirmation node-shuffle intervention mean delta Acc=-0.00003, Macro-F1=-0.00003, prediction flip=0.000154; this effect was small.

## Final decision matrix

| Hypothesis | Status | Evidence |
|---|---|---|
| H1_e5_routing_nonidentifiability_confirmed | STRONG_SUPPORT | Two prescribed distinct five-action vectors map to alpha_U; maximum representation error=4.68e-07, logit error=3.34e-07. |
| H2_marginal_modality_utility_is_sufficient | STRONG_SUPPORT | Material joint-utility gate=JOINT_UTILITY_WEAK_OR_MIXED; relative joint regret=0.04478. |
| H3_joint_multimodal_structural_utility_is_material | NO_SUPPORT | Joint gate requires relative regret >=0.10 and reproducible interaction; observed 0.04478 and 1/3 reproducible datasets. |
| H4_discrete_order_selection_is_sufficient | MIXED_SUPPORT | Mean continuous CE gain over discrete=0.006322; near-vertex fraction by dataset={'Movies': 0.7967276483721552, 'Reddit-S': 0.873636744966443, 'ele-fashion': 0.9217482966524034}. |
| H5_continuous_filtering_has_extra_headroom | MIXED_SUPPORT | Continuous gain by dataset={'Movies': 0.016133979583779934, 'Reddit-S': 0.0014822907590616998, 'ele-fashion': 0.0013486492680385334}; positive dataset-seeds=9/9; non-vertex fraction by dataset={'Movies': 0.20327235162784485, 'Reddit-S': 0.12636325503355705, 'ele-fashion': 0.07825170334759657}. |
| H6_continuous_oracle_filter_is_cross_seed_stable | MIXED_SUPPORT | Train alpha cosine=0.7645, JS=0.1631, rank agreement=0.7387. |
| H7_canonical_frozen_router_improves_uniform | STRONG_SUPPORT | {"mean_delta_acc": 0.0017128255632188586, "mean_delta_macro_f1": 0.003370394921682834, "mean_node_shuffle_prediction_flip": 0.003892946108761761, "node_shuffle_functional_effect": true, "pass": true, "positive_dataset_count": 3, "positive_seed_count": "8/9"} |
| H8_joint_router_improves_independent_router | NOT_TESTED | D1 skipped because the Phase B joint-utility gate was weak/mixed. |
| H9_untouched_confirmation | MIXED_SUPPORT | D0 confirmation on Toys/Grocery: 3/6 positive seed pairs; mean delta Acc=6.346e-05, Macro-F1=7.992e-05. |

## Interpretation and next step

E6 supports independent canonical structural routing at the screening gate and does not support a material joint text-visual routing requirement at the pre-registered threshold. The continuous label oracle finds extra CE headroom, especially on Movies, but its filters are often near simplex vertices and only moderately stable across seeds; that does not establish that a deployable continuous filter is needed. Untouched confirmation is heterogeneous (Toys positive, Grocery negative) with only 3/6 positive seed pairs and small mean gains. Treat the router as a promising screening result, not a broadly confirmed final model. The next useful step is targeted replication of independent routing across more seeds, with particular attention to the Movies versus Grocery split, while retaining C0 Uniform as the conservative fallback. Do not build D1 or distill the oracle from this evidence.

No prohibited model components, C0 retraining, Test-based selection, Test metrics, or topology changes were used.
