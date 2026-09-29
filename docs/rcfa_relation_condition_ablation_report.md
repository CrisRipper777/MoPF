# RCFA Explicit Relation-Effect Attribution

## 1. Research question

Does passing Stage-I Relation Semantic Effect (RSE) explicitly into the Stage-II RCFA conditioner add value, with RCFA and all Stage-I computation held fixed? The ablation is named **w/o Explicit Relation Effect**: RSE is still computed, but its RCFA conditioner input is zero.

## 2. Provenance and protocol

- Starting source branch/SHA: `crsa_rcfa_v2` / `ed2a4174d806d1695c753e8926b386075f77fee2`.
- Experiment branch: `exp/rcfa_relation_condition_ablation`.
- Formal source SHA: `d80869c9d8450ff0ece25fa3773c7349d82d50c0`.
- Formal contexts complete: 15 / 15.
- Matrix: NC, full-graph, AdamW, lr=1e-3, weight decay=1e-4, up to 300 epochs, patience=30, min epoch=30, min delta=1e-4, grad clip=1.0, evaluation every epoch, full inference, no scheduler, K=3, best validation accuracy checkpoint.
- Validation only: `evaluate_test=false`; modality-mask evaluation disabled. No LP context was launched.

## 3. Implementation-control audit

- Preflight status: `passed`.
- Trainable parameters, including the NC head (Full/noRSE): 1226187 / 1226187; model-only counts: 1225416 / 1225416.
- Same-seed initialization tensors identical: True.
- CRSA states, H0, delta, RSE and captured conditioner arguments passed: True.
- Formal Full-v2/noRSE parameter-count parity: passed across 15/15 paired contexts.
- The only ablated input is R_k at the RCFA conditioner; noRSE supplies `zeros_like(R_k)`. CRSA propagation and RSE/delta computation remain active.
- Model and YAML source files are frozen at the starting snapshot; preflight records their Git blob hashes.

## 4. RCFA-noRSE validation results

Accuracy and Macro-F1 are mean ± sample standard deviation across three seeds, in percent.

| Dataset | n | Val Accuracy (%) | Val Macro-F1 (%) |
|---|---:|---:|---:|
| Movies | 3 | 56.28 ± 0.20 | 46.44 ± 1.56 |
| Toys | 3 | 80.16 ± 0.06 | 77.50 ± 0.44 |
| Grocery | 3 | 83.19 ± 0.36 | 75.07 ± 1.58 |
| ele-fashion | 3 | 88.00 ± 0.20 | 76.95 ± 0.52 |
| Reddit-S | 3 | 96.54 ± 0.45 | 92.68 ± 0.35 |

## 5. Three-way comparison — Accuracy

Values and deltas are percentage points; first three columns are the mean validation scores across seeds.

| Dataset | CRSA | RCFA w/o RSE | Full RCFA+RSE | noRSE−CRSA | Full−noRSE | Full−CRSA |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 57.37 | 56.28 | 56.27 | -1.09 | -0.01 | -1.10 |
| Toys | 80.54 | 80.16 | 80.14 | -0.39 | -0.02 | -0.40 |
| Grocery | 83.54 | 83.19 | 83.18 | -0.35 | -0.01 | -0.36 |
| ele-fashion | 87.78 | 88.00 | 88.01 | +0.22 | +0.01 | +0.23 |
| Reddit-S | 95.90 | 96.54 | 96.55 | +0.64 | +0.01 | +0.65 |

### Paired seed values — Accuracy

| Dataset | Seed | CRSA | RCFA w/o RSE | Full RCFA+RSE | noRSE−CRSA | Full−noRSE | Full−CRSA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 56.96 | 56.27 | 56.12 | -0.69 | -0.15 | -0.84 |
| Movies | 43 | 57.14 | 56.09 | 56.24 | -1.05 | +0.15 | -0.90 |
| Movies | 44 | 58.01 | 56.48 | 56.45 | -1.53 | -0.03 | -1.56 |
| Toys | 42 | 80.48 | 80.09 | 80.24 | -0.39 | +0.14 | -0.24 |
| Toys | 43 | 80.67 | 80.19 | 80.19 | -0.48 | +0.00 | -0.48 |
| Toys | 44 | 80.48 | 80.19 | 80.00 | -0.29 | -0.19 | -0.48 |
| Grocery | 42 | 83.48 | 83.02 | 82.93 | -0.47 | -0.09 | -0.56 |
| Grocery | 43 | 83.75 | 82.96 | 83.13 | -0.79 | +0.18 | -0.61 |
| Grocery | 44 | 83.40 | 83.60 | 83.48 | +0.20 | -0.12 | +0.09 |
| ele-fashion | 42 | 87.47 | 88.16 | 88.08 | +0.69 | -0.07 | +0.61 |
| ele-fashion | 43 | 87.80 | 87.78 | 87.70 | -0.02 | -0.08 | -0.10 |
| ele-fashion | 44 | 88.08 | 88.07 | 88.25 | -0.01 | +0.17 | +0.16 |
| Reddit-S | 42 | 95.75 | 96.04 | 95.97 | +0.28 | -0.06 | +0.22 |
| Reddit-S | 43 | 96.07 | 96.92 | 96.95 | +0.85 | +0.03 | +0.88 |
| Reddit-S | 44 | 95.88 | 96.67 | 96.73 | +0.79 | +0.06 | +0.85 |

## 5. Three-way comparison — Macro-F1

Values and deltas are percentage points; first three columns are the mean validation scores across seeds.

| Dataset | CRSA | RCFA w/o RSE | Full RCFA+RSE | noRSE−CRSA | Full−noRSE | Full−CRSA |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 48.86 | 46.44 | 46.46 | -2.42 | +0.02 | -2.40 |
| Toys | 78.00 | 77.50 | 77.37 | -0.50 | -0.13 | -0.63 |
| Grocery | 75.28 | 75.07 | 74.50 | -0.21 | -0.57 | -0.78 |
| ele-fashion | 75.87 | 76.95 | 75.93 | +1.08 | -1.02 | +0.06 |
| Reddit-S | 91.73 | 92.68 | 92.51 | +0.95 | -0.17 | +0.77 |

### Paired seed values — Macro-F1

| Dataset | Seed | CRSA | RCFA w/o RSE | Full RCFA+RSE | noRSE−CRSA | Full−noRSE | Full−CRSA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 42 | 49.09 | 46.07 | 46.50 | -3.01 | +0.43 | -2.58 |
| Movies | 43 | 47.04 | 45.09 | 45.22 | -1.95 | +0.13 | -1.82 |
| Movies | 44 | 50.44 | 48.15 | 47.65 | -2.29 | -0.51 | -2.80 |
| Toys | 42 | 77.79 | 77.26 | 77.12 | -0.53 | -0.14 | -0.67 |
| Toys | 43 | 78.28 | 78.01 | 77.99 | -0.27 | -0.02 | -0.29 |
| Toys | 44 | 77.92 | 77.23 | 76.99 | -0.69 | -0.24 | -0.93 |
| Grocery | 42 | 77.12 | 76.70 | 75.83 | -0.41 | -0.87 | -1.28 |
| Grocery | 43 | 75.65 | 74.97 | 73.20 | -0.68 | -1.76 | -2.45 |
| Grocery | 44 | 73.08 | 73.54 | 74.46 | +0.46 | +0.92 | +1.38 |
| ele-fashion | 42 | 74.64 | 77.53 | 76.06 | +2.89 | -1.47 | +1.42 |
| ele-fashion | 43 | 75.87 | 76.53 | 75.28 | +0.66 | -1.25 | -0.59 |
| ele-fashion | 44 | 77.10 | 76.79 | 76.44 | -0.31 | -0.35 | -0.66 |
| Reddit-S | 42 | 92.01 | 92.42 | 92.22 | +0.41 | -0.20 | +0.22 |
| Reddit-S | 43 | 91.37 | 93.08 | 92.90 | +1.71 | -0.18 | +1.53 |
| Reddit-S | 44 | 91.83 | 92.54 | 92.40 | +0.71 | -0.14 | +0.57 |

## 6. Attribution analysis

- **Absorption effect** is `RCFA-noRSE − CRSA`; **explicit RSE effect** is `Full − RCFA-noRSE`; **total Stage-II effect** is `Full − CRSA`. These are paired by dataset and seed.

- Accuracy: noRSE mean below CRSA on Movies, Toys, Grocery; explicit RSE mean helps Full on ele-fashion, Reddit-S and lowers it on Movies, Toys, Grocery. Inspect the paired tables for seed consistency.
- Macro-F1: noRSE mean below CRSA on Movies, Toys, Grocery; explicit RSE mean helps Full on Movies and lowers it on Toys, Grocery, ele-fashion, Reddit-S. Inspect the paired tables for seed consistency.

## 7. Dataset heterogeneity

Dataset means and seed-level differences are reported separately above. Directions are descriptive; no across-dataset average is used as a substitute for the five per-dataset patterns.

## 8. Optional E0-D exploratory correspondence

Exact artifact: `outputs/e0_empirical_motivation/relation_utilization_bridge/relation_utilization_association.csv`. Dataset-level mean degree-controlled partial rho is compared with Full−noRSE deltas. This is exploratory only (n=5 datasets), not statistical evidence, and not a paper claim.

| Dataset | Mean degree-controlled partial rho | Full−noRSE Accuracy (pp) | Full−noRSE Macro-F1 (pp) |
|---|---:|---:|---:|
| Movies | 0.265 | -0.01 | +0.02 |
| Toys | 0.342 | -0.02 | -0.13 |
| Grocery | 0.311 | -0.01 | -0.57 |
| ele-fashion | 0.399 | +0.01 | -1.02 |
| Reddit-S | 0.879 | +0.01 | -0.17 |
- Accuracy: descriptive Pearson=0.762, Spearman=0.700; rho order: Reddit-S, ele-fashion, Toys, Grocery, Movies; delta order: Reddit-S, ele-fashion, Grocery, Movies, Toys.
- Macro-F1: descriptive Pearson=0.114, Spearman=-0.500; rho order: Reddit-S, ele-fashion, Toys, Grocery, Movies; delta order: Movies, Toys, Reddit-S, Grocery, ele-fashion.

## 9. Engineering audit

- Complete formal runs: 15/15; invalid or incomplete: 0.
- Total recorded runtime: 0.81 GPU-hours across sequential runs; mean per run: 194.8 s.
- Maximum recorded peak GPU memory: 10007.1 MiB.
- OOM signatures found in captured logs: 0; NaN/Inf tokens found: 0.
- Focused tests: 14 passed across `tests/test_crsa_rcfa.py` and `tests/test_rcfa_relation_condition_ablation.py`.
- Artifact audit requires finite checkpoint tensors and train losses, validation-only metric keys, resolved protocol agreement, and strict model/head checkpoint reload.
- Test metrics were neither evaluated nor analyzed; no LP runs were included.

## 10. Research conclusion and recommended next step

- **Accuracy:** RCFA-noRSE mean is below CRSA on Movies, Toys, Grocery; the decrease is negative in all three seeds on Movies, Toys. Full-RSE mean is above noRSE on ele-fashion, Reddit-S, and below it on Movies, Toys, Grocery; all-seed RSE gains occur on none, all-seed losses on none.
- **Macro-F1:** RCFA-noRSE mean is below CRSA on Movies, Toys, Grocery; the decrease is negative in all three seeds on Movies, Toys. Full-RSE mean is above noRSE on Movies, and below it on Toys, Grocery, ele-fashion, Reddit-S; all-seed RSE gains occur on none, all-seed losses on Toys, ele-fashion, Reddit-S.
- **Q1 — Does RCFA absorption itself cause negative transfer?** Yes, descriptively on the dataset means: RCFA-noRSE accuracy is below CRSA on Movies, Toys, Grocery and macro-F1 is below on Movies, Toys, Grocery. The accuracy decrease is same-direction across all three seeds on Movies, Toys; macro-F1 is same-direction on Movies, Toys. This identifies an absorption-side loss independent of explicit RSE conditioning.
- **Q2 — Does explicit RSE conditioning add value?** Mean accuracy deltas are near zero (largest absolute dataset-mean Full−noRSE delta 0.016 pp) and seed directions are mixed. Macro-F1 decreases on Toys, Grocery, ele-fashion, Reddit-S and increases only on Movies; the decreases are same-direction across all three seeds on Toys, ele-fashion, Reddit-S. The observed results do not support a reliable general task benefit from explicit RSE conditioning.
- **Q3 — What drives the dataset split?** Accuracy's larger absolute mean component is absorption on Movies, Toys, Grocery, ele-fashion, Reddit-S; explicit RSE is smaller on every dataset. The absorption contrast itself is negative on Movies/Toys/Grocery and positive on ele-fashion/Reddit-S, matching the broad split; explicit RSE contributes almost no dataset-mean accuracy change. Macro-F1 shows an additional explicit-conditioning decrease on most datasets.
- **Q4 — Retain RCFA as Stage II?** Do not retain this exact absorber as a universal Stage-II design: noRSE underperforms CRSA on three datasets and improves on two. It remains a dataset-conditional candidate for ele-fashion and Reddit-S, while Movies, Toys, and Grocery show absorption-side losses.
- **Q5 — Next step?** Prioritize changing Stage-II absorption. The noRSE−CRSA contrast is the larger accuracy component on all five datasets, while mean Full−noRSE accuracy changes are near zero and explicit conditioning lowers Macro-F1 on four datasets. Do not begin relation-effect reliability analysis from these results alone; no follow-up experiment was started.

No automatic scientific PASS/FAIL threshold was applied. The comparison is a controlled decomposition; it does not require a monotonic Full > noRSE > CRSA ordering.
