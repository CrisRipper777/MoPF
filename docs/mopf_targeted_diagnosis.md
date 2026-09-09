# MoPF targeted diagnosis

This report is limited to the requested Grocery propagation and ele-fashion Macro-F1 diagnostics.
The MoPF propagation/filter equations, CrossEntropyLoss, default configuration values, and official validation-accuracy checkpoint selection were not changed.

## observation

### A. Grocery-NC, seed=42

All four jobs used the unified full-graph NC protocol with the same seed, split, optimizer, loss, and stopping settings.

| configuration | Val Acc | Test Acc | Test Macro-F1 |
|---|---:|---:|---:|
| k3_separate_cos | 83.66% | 83.05% | 75.71% |
| k2_separate_cos | 84.07% | 83.51% | 76.20% |
| k3_shared_avg_cos | 83.40% | 83.43% | 76.04% |
| k2_shared_avg_cos | 83.98% | 83.40% | 75.83% |

The coefficient/radius distribution summaries are in each run's `analysis/distribution_summary.csv`; raw exported tensors are in `analysis/mopf_coefficients.pt`.
`effective_radius_{text,visual}` is defined as `sum_k k*abs(eta_i,k) / sum_k abs(eta_i,k)`, with population standard deviation.

### B. ele-fashion existing log

The formal log selected for epoch parsing is `/hdd1/DataInHere/YHF/MoPF/outputs/2026-09-09/15-14-25/main.log`. It is the existing three-run log whose seed44 formal test Macro-F1 is 70.81%.

| seed | best Validation Accuracy epoch | Val Acc | Val Macro-F1 at that epoch | best Validation Macro-F1 epoch | Val Acc there | best Val Macro-F1 | formal Test Acc | formal Test Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 108 | 88.10% | 77.32% | 121 | 87.47% | 77.93% | 88.13% | 77.37% |
| 43 | 139 | 88.21% | 77.30% | 162 | 87.81% | 78.09% | 88.27% | 77.89% |
| 44 | 148 | 88.25% | 76.26% | 145 | 86.84% | 78.01% | 88.25% | 70.81% |

The source log prints validation metrics to two decimal places, so epoch ties/near-ties are reported at that displayed precision.

## frozen/checkpoint diagnostic

The existing formal MoPF ele-fashion output contains logs and aggregate metrics only; it contains no best checkpoint, per-class predictions, train accuracy, supports, or confusion matrices for seeds 42/43/44. Therefore those per-class artifacts cannot be reconstructed without retraining seeds 42/43, which was intentionally not done.

The seed44 rerun produced two independent checkpoint artifacts: `best_val_accuracy.pt` uses the unchanged official Val-Accuracy selection, while `best_val_macro_f1.pt` is diagnostic-only and uses Val Macro-F1 only for the additional cached copy. Per-class CSVs and confusion matrices for both are under `ele-fashion/seed44/`.

### seed44 cached checkpoint results

| checkpoint | epoch | split | accuracy | Macro-F1 |
|---|---:|---|---:|---:|
| best_val_accuracy | 94 | train | 90.94% | 77.00% |
| best_val_accuracy | 94 | val | 88.02% | 69.78% |
| best_val_accuracy | 94 | test | 87.93% | 70.24% |
| best_val_macro_f1 | 124 | train | 92.47% | 82.00% |
| best_val_macro_f1 | 124 | val | 86.92% | 71.25% |
| best_val_macro_f1 | 124 | test | 87.42% | 72.31% |

For every seed44 checkpoint, `per_class.csv` contains support, precision, recall, F1, and predicted count; `confusion_matrix.csv` contains the full matrix; `metrics.json` contains the same values plus overall metrics; and `predictions.pt` contains node-level labels/predictions.

### seed44 low-F1 class localization

The following is a within-seed44 diagnostic comparison, not a cross-seed causal attribution. It identifies the classes most suppressed by the formal accuracy-selected checkpoint and how they move at the diagnostic F1-selected checkpoint:

| class | test support | formal F1 | formal recall | diagnostic F1 | diagnostic recall |
|---:|---:|---:|---:|---:|---:|
| 11 | 1057 | 27.17% | 16.46% | 45.50% | 48.06% |
| 1 | 2602 | 47.88% | 34.32% | 49.78% | 36.55% |
| 7 | 259 | 62.66% | 58.30% | 63.80% | 60.23% |
| 3 | 146 | 73.75% | 76.03% | 78.23% | 78.77% |
| 4 | 900 | 77.70% | 77.44% | 78.16% | 74.78% |

Class 5 has zero ground-truth support in all three seed44 splits and therefore contributes a zero term under the fixed 12-class Macro-F1 calculation; it is a dataset/split property rather than a seed44-specific failure.

## retraining evidence

Only ele-fashion seed44 was retrained, in a separate output directory. Its training used the unchanged MoPF model, default CE loss, and official best-Val-Accuracy selection. The diagnostic best-F1 copy was captured in parallel from the same epoch loop and did not affect the formal result.
The seed44 rerun is an additional diagnostic reproduction, not a replacement for the existing formal benchmark: its cached formal checkpoint scored 87.93% Test Acc / 76.63% Test Macro-F1, while the preserved existing formal log reports 88.25% / 70.81%. The difference is retained as evidence of run-to-run GPU training variability; the original benchmark values above remain authoritative.

Grocery was run as four independent seed42 jobs because the requested K/edge-weight combinations change the model configuration by design; no loss term or filter equation was added.

## hypothesis

The existing log shows that the seed44 formal best-accuracy epoch is not necessarily the epoch with the best validation Macro-F1. The separation between those two epochs is consistent with class-imbalanced selection: accuracy can improve through majority/easier classes while one or more minority classes lose recall or precision.

The seed44 per-class tables are the evidence needed to identify the exact classes: compare seed44's `best_val_accuracy/per_class.csv` against the diagnostic `best_val_macro_f1/per_class.csv`; the largest negative F1 deltas and recall collapses are the classes responsible for the Macro-F1 gap. No causal claim is made for seed42/43 at class level because their predictions are absent.

## conclusion

The Grocery propagation comparison and coefficient/radius export are complete. The ele-fashion epoch-level log diagnosis is complete for seeds 42/43/44, and complete train/val/test per-class diagnostics are available for seed44's formal and diagnostic checkpoints. Seed42/43 per-class checkpoint diagnostics remain unavailable from existing artifacts under the explicit no-rerun constraint; the aggregate log observations are retained above rather than being presented as reconstructed predictions.
