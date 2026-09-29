# CRSA + ICSR v3 Validation Report

## 1. Research question

Does an intrinsic-context interaction residual improve on the frozen CRSA carrier, and how does this realization compare with the prior RCFA utility-gating results? This report uses validation metrics only.

## 2. Provenance and protocol

- Formal runs: **15 / 15 complete**.
- Formal source commit(s): `61eb33ea01c84a469c7a77cd1d78827507fb7c47`; uniform: `True`.
- NC full graph; AdamW, lr 1e-3, weight decay 1e-4; epochs 300, patience 30; min epoch 30, min delta 1e-4; clip 1.0; evaluate each epoch; no scheduler; full inference; K=3.
- Best checkpoint selected by validation accuracy; Test evaluation disabled; LP not run.

## 3. Implementation-control audit

See `preflight_report.md`. Frozen CRSA parity, identity initialization, descriptor formulas, modality isolation, parameter delta, finite backward, and strict reload were checked on a synthetic graph.
ICSR uses same-modality H0/Ck only; it has no RSE input, other-modality input, or utility gate.

## 4. Full-v3 validation results

| Dataset | Accuracy mean ± std | Macro-F1 mean ± std | n |
|---|---:|---:|---:|
| Movies | 55.479 ± 0.590% | 45.036 ± 1.425% | 3 |
| Toys | 80.035 ± 0.432% | 76.675 ± 0.390% | 3 |
| Grocery | 82.343 ± 0.356% | 72.389 ± 1.413% | 3 |
| ele-fashion | 87.757 ± 0.071% | 75.917 ± 0.266% | 3 |
| Reddit-S | 96.498 ± 0.429% | 92.800 ± 0.226% | 3 |

## 5. Four-way comparison and paired values

### Accuracy

| Dataset | CRSA | RCFA-noRSE | RCFA-Full | ICSR-v3 | ICSR−CRSA | ICSR−RCFA-Full |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 57.369% | 56.279% | 56.269% | 55.479% | -1.890 pp | -0.790 pp |
| Toys | 80.543% | 80.156% | 80.140% | 80.035% | -0.507 pp | -0.105 pp |
| Grocery | 83.543% | 83.192% | 83.182% | 82.343% | -1.201 pp | -0.839 pp |
| ele-fashion | 87.784% | 88.002% | 88.009% | 87.757% | -0.027 pp | -0.252 pp |
| Reddit-S | 95.900% | 96.540% | 96.550% | 96.498% | +0.598 pp | -0.052 pp |

### Macro-F1

| Dataset | CRSA | RCFA-noRSE | RCFA-Full | ICSR-v3 | ICSR−CRSA | ICSR−RCFA-Full |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 48.856% | 46.437% | 46.456% | 45.036% | -3.820 pp | -1.421 pp |
| Toys | 77.998% | 77.500% | 77.368% | 76.675% | -1.324 pp | -0.693 pp |
| Grocery | 75.283% | 75.071% | 74.499% | 72.389% | -2.893 pp | -2.110 pp |
| ele-fashion | 75.868% | 76.950% | 75.927% | 75.917% | +0.049 pp | -0.010 pp |
| Reddit-S | 91.732% | 92.678% | 92.506% | 92.800% | +1.067 pp | +0.294 pp |

Seed-level paired values (scores are %, deltas are percentage points):

| Dataset | Seed | Metric | CRSA | RCFA-noRSE | RCFA-Full | ICSR-v3 | ICSR−CRSA | ICSR−RCFA-Full |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Grocery | 42 | Accuracy | 83.485% | 83.016% | 82.928% | 82.167% | -1.318 pp | -0.761 pp |
| Grocery | 42 | Macro-F1 | 77.117% | 76.703% | 75.833% | 72.909% | -4.208 pp | -2.924 pp |
| Grocery | 43 | Accuracy | 83.748% | 82.958% | 83.133% | 82.108% | -1.640 pp | -1.025 pp |
| Grocery | 43 | Macro-F1 | 75.649% | 74.967% | 73.204% | 70.790% | -4.859 pp | -2.414 pp |
| Grocery | 44 | Accuracy | 83.397% | 83.602% | 83.485% | 82.753% | -0.644 pp | -0.732 pp |
| Grocery | 44 | Macro-F1 | 73.081% | 73.543% | 74.460% | 73.469% | +0.388 pp | -0.992 pp |
| Movies | 42 | Accuracy | 56.959% | 56.269% | 56.119% | 55.039% | -1.920 pp | -1.080 pp |
| Movies | 42 | Macro-F1 | 49.085% | 46.074% | 46.503% | 44.116% | -4.970 pp | -2.388 pp |
| Movies | 43 | Accuracy | 57.139% | 56.089% | 56.239% | 55.249% | -1.890 pp | -0.990 pp |
| Movies | 43 | Macro-F1 | 47.040% | 45.086% | 45.221% | 44.315% | -2.726 pp | -0.906 pp |
| Movies | 44 | Accuracy | 58.008% | 56.479% | 56.449% | 56.149% | -1.860 pp | -0.300 pp |
| Movies | 44 | Macro-F1 | 50.441% | 48.151% | 47.645% | 46.677% | -3.764 pp | -0.969 pp |
| Reddit-S | 42 | Accuracy | 95.753% | 96.036% | 95.974% | 96.036% | +0.283 pp | +0.063 pp |
| Reddit-S | 42 | Macro-F1 | 92.006% | 92.420% | 92.223% | 92.929% | +0.923 pp | +0.706 pp |
| Reddit-S | 43 | Accuracy | 96.068% | 96.917% | 96.949% | 96.886% | +0.818 pp | -0.063 pp |
| Reddit-S | 43 | Macro-F1 | 91.365% | 93.075% | 92.899% | 92.932% | +1.567 pp | +0.033 pp |
| Reddit-S | 44 | Accuracy | 95.879% | 96.666% | 96.729% | 96.571% | +0.692 pp | -0.157 pp |
| Reddit-S | 44 | Macro-F1 | 91.826% | 92.537% | 92.395% | 92.539% | +0.713 pp | +0.143 pp |
| Toys | 42 | Accuracy | 80.478% | 80.092% | 80.237% | 80.285% | -0.193 pp | +0.048 pp |
| Toys | 42 | Macro-F1 | 77.793% | 77.259% | 77.119% | 76.499% | -1.294 pp | -0.620 pp |
| Toys | 43 | Accuracy | 80.672% | 80.188% | 80.188% | 79.536% | -1.136 pp | -0.652 pp |
| Toys | 43 | Macro-F1 | 78.285% | 78.013% | 77.993% | 76.403% | -1.882 pp | -1.590 pp |
| Toys | 44 | Accuracy | 80.478% | 80.188% | 79.995% | 80.285% | -0.193 pp | +0.290 pp |
| Toys | 44 | Macro-F1 | 77.917% | 77.229% | 76.991% | 77.122% | -0.795 pp | +0.131 pp |
| ele-fashion | 42 | Accuracy | 87.471% | 88.156% | 88.084% | 87.798% | +0.327 pp | -0.286 pp |
| ele-fashion | 42 | Macro-F1 | 74.639% | 77.530% | 76.057% | 76.063% | +1.424 pp | +0.006 pp |
| ele-fashion | 43 | Accuracy | 87.798% | 87.777% | 87.696% | 87.798% | +0.000 pp | +0.102 pp |
| ele-fashion | 43 | Macro-F1 | 75.867% | 76.530% | 75.280% | 75.609% | -0.257 pp | +0.329 pp |
| ele-fashion | 44 | Accuracy | 88.084% | 88.074% | 88.248% | 87.675% | -0.409 pp | -0.573 pp |
| ele-fashion | 44 | Macro-F1 | 77.099% | 76.791% | 76.443% | 76.079% | -1.020 pp | -0.364 pp |

## 6. Research interpretation

### Q1. How does ICSR-v3 compare with CRSA?

The Accuracy mean is lower on Movies (-1.890 pp), Toys (-0.507 pp), Grocery (-1.201 pp), and ele-fashion (-0.027 pp), and higher on Reddit-S (+0.598 pp). For Macro-F1, the corresponding changes are Movies -3.820 pp, Toys -1.324 pp, Grocery -2.893 pp, ele-fashion +0.049 pp, and Reddit-S +1.067 pp.

Seed direction is consistent on Movies and Toys: all three seeds are lower than CRSA for both metrics. Grocery Accuracy is lower in all three seeds; Grocery Macro-F1 is lower in two of three. Reddit-S is higher in all three seeds for both metrics. ele-fashion is near neutral and its seed direction is mixed.

### Q2. Does ICSR reduce RCFA negative transfer on Movies, Toys, and Grocery?

No. Against CRSA, ICSR-v3 is lower on all three datasets for Accuracy means and is less healthy for Macro-F1, especially Grocery. RCFA-Full−CRSA Accuracy means were Movies -1.100 pp, Toys -0.403 pp, and Grocery -0.361 pp; ICSR−CRSA was more negative on each. ICSR-v3 also trails RCFA-Full in the three dataset mean Accuracy comparisons.

### Q3. Are the ele-fashion and Reddit-S positive trends retained?

Reddit-S retains a consistent positive change over CRSA (+0.598 pp Accuracy and +1.067 pp Macro-F1, with all three seeds positive); its mean is approximately tied with RCFA-Full Accuracy and higher on Macro-F1. ele-fashion is effectively neutral relative to CRSA (+0.049 pp Macro-F1 and -0.027 pp Accuracy), without a consistent seed-level gain, so its earlier positive trend is not clearly retained.

### Q4. Is interaction representation healthier than utility gating?

Not across the full dataset split. ICSR-v3 is more negative than RCFA-Full relative to CRSA on Movies, Toys, and Grocery, particularly for Macro-F1 on Grocery. It retains the Reddit-S gain and is close to neutral on ele-fashion. Thus this ICSR realization does not provide a general remedy for Stage-II negative transfer, although its Reddit-S behavior is favorable. These are descriptive results from five datasets and three seeds; no significance test or automatic threshold is applied.

### Q5. Should Agreement/Deviation ablations follow?

Based only on this round, do not prioritize the Agreement/Deviation ablation matrix yet. Full ICSR-v3 did not improve the CRSA baseline consistently and was lower on Movies, Toys, and Grocery; the component ablations would not resolve that attribution cleanly. Keep the current result as evidence about this realization and decide separately whether Stage II should be revised before spending the next experiment budget. No next-stage run is started here.

## 7. Engineering and isolation audit

- Test metrics accessed: **NO**.
- Test labels used/indexed for analysis: **NO**.
- LP run: **NO**.
- RSE used inside ICSR: **NO**.
- Other modality used inside ICSR: **NO**.
- Stage-II utility gate: **NO**.
- Four-way results are paired by the same dataset and seed.
- Total measured formal runtime: 2554.7 s across 15 runs (mean 170.3 s/run).
- Peak GPU allocation: 8532.3 MiB (ele-fashion seed 42).
- Formal OOM/NaN/Inf log markers: 0; all run losses/checkpoint tensors finite and strict checkpoint reload passed.

Historical provenance details are in `provenance_audit.json`; formal run details are in `run_status.csv` and `formal_run_manifest.csv`.
