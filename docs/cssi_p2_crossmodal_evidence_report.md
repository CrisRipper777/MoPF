# CSSI P2 Cross-Modal Structural Evidence Report

## Scope and leakage controls

This report reuses the 15 frozen Plain best-validation checkpoints. The graph encoder and NC classifier are not retrained. Probe fitting uses train-node descriptors and evaluation uses validation-node descriptors; test nodes and test labels are never loaded.

Each target dataset/seed/modality/order is fit independently with StandardScaler + L2 Logistic Regression (`C=1`) for sign classification and StandardScaler + Ridge (`alpha=1`) for utility regression.

## Response descriptor

Each response uses `[log1p(||R||), ||R||/(||S_prev||+epsilon), cos(R,H0), cos(R,S_prev), cos(S_prev,H0)]`. Non-finite descriptor values are converted to zero and the audit records finite matrices.

## P1 corrected CE sign

The corrected convention is `u_ce = loss_remove - loss_base`; positive means removal increases loss and the response is helpful. Margin remains `u_margin = margin_base - margin_remove`. P1 corrected outputs are in `outputs/cssi_p1_corrected/` and the full report is `docs/cssi_p1_corrected_report.md`.

P1 corrected aggregate CE/margin sign agreement has mean `0.8475` across 30 validation cells.

## Probe definitions

- `own`: target modality descriptors for orders 1/2/3.
- `paired_same_order`: Own plus the paired modality's target-order descriptor.
- `paired_all_orders`: Own plus all three paired-modality descriptors.
- `shuffled_all_orders`: same dimension as Paired-AllOrders, with paired modality node correspondence independently permuted within train and validation.

## Primary validation comparison at |u| > 1e-5

The compact table reports mean validation metrics over target cells and seeds. Full per-cell results and threshold sensitivities are in `probe_results.csv` and `summary.csv`.

| Task / metric | Own | Same | All | Shuffled | Delta same | Delta all | Delta match |
|---|---:|---:|---:|---:|---:|---:|---:|
| classification / auroc | 0.5706 | 0.5783 | 0.5819 | 0.5687 | 0.0076 | 0.0036 | 0.0132 |
| classification / auprc | 0.6790 | 0.6851 | 0.6878 | 0.6774 | 0.0061 | 0.0026 | 0.0104 |
| classification / balanced_accuracy | 0.5516 | 0.5569 | 0.5595 | 0.5504 | 0.0053 | 0.0027 | 0.0091 |
| regression / spearman | 0.0457 | 0.0476 | 0.0477 | 0.0432 | 0.0019 | 0.0001 | 0.0045 |
| regression / mae | 0.0957 | 0.0960 | 0.0962 | 0.0960 | 0.0003 | 0.0002 | 0.0002 |
| regression / r2 | 0.0086 | 0.0106 | 0.0121 | 0.0083 | 0.0020 | 0.0016 | 0.0038 |

For AUROC, AUPRC, balanced accuracy, Spearman, and R2, positive delta is better. For MAE, lower is better, so a positive MAE delta is not an improvement.

## Per-cell/seed stability of the primary AUROC deltas

At `|u|>1e-5`, 78/90 target-cell/seed fits are valid; the remaining cases are degenerate train or validation sign classes, primarily in propagation-saturated cells.

| comparison | positive delta fits | valid fits | fraction |
|---|---:|---:|---:|
| same | 62 | 78 | 0.795 |
| all | 57 | 78 | 0.731 |
| match | 73 | 78 | 0.936 |

The target-cell/seed details are in `probe_results.csv`; these fractions show whether the pooled deltas repeat across checkpoints rather than only reflecting one global fit.

## Threshold sensitivity of AUROC deltas

| threshold | Delta same | Delta all | Delta match | valid target cells |
|---:|---:|---:|---:|---:|
| 1e-06 | 0.00691 | 0.00734 | 0.01109 | 30 |
| 1e-05 | 0.00762 | 0.00364 | 0.01316 | 26 |
| 1e-04 | 0.00854 | 0.00394 | 0.01493 | 26 |

## Near-zero sensitivity

Classification results are reported at `1e-6`, `1e-5`, and `1e-4`; samples within the threshold are excluded. Regression reports all samples and additional non-near-zero sensitivities. Cells with insufficient classes or samples are marked degenerate rather than forced into a metric.

## Utility correspondence

- same_node: diagonal mean/median `0.2528`/`0.2640`, off-diagonal mean/max `0.1167`/`0.3217`, all-cell mean `0.1620`.
- shuffled_node: diagonal mean/median `0.0017`/`0.0007`, off-diagonal mean/max `-0.0011`/`0.0408`, all-cell mean `-0.0002`.

The correspondence matrix is descriptive only; utility is never included as a probe feature.

## Decision against Cases A/B/C

Case B requires same-order gains over Own, All approximately equal to Same, and matched All gains over Shuffled. Case C additionally requires a stable increment from Same to All. The conclusion below is based on the complete per-cell/seed table, threshold sensitivity, and shuffled control, not on one pooled score.

**Result:** Provisional Case C-like on pooled AUROC: All exceeds Same and Shuffled. Positive-fit fractions are Same 0.795, All-over-Same 0.731, and All-over-Shuffled 0.936; the effect is not universal, so do not implement full co-reasoning without a pre-registered same-order control and per-cell confirmation.

The recommended next model direction is therefore determined by the per-cell stability table: do not implement cross-order interaction unless the All-over-Same and All-over-Shuffled gains are repeated across datasets and seeds. Saturated cells must be excluded from strong architectural claims.

## P1 saturation caveat

P1 uses a median normalized-response threshold of `1e-3`; 4 aggregate cells are flagged near-degenerate. Their utility signs should not be treated as strong evidence of meaningful response heterogeneity.

## Limitations

P1 utility is a frozen-forward functional leave-one-response-out intervention, not a strict causal effect. P2 probes establish predictive association only; they do not implement a proposed CSSI mechanism or establish causality.
