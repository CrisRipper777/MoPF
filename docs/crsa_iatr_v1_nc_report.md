# CRSA+IATR NC validation report

Mode: formal. This report uses validation accuracy and validation Macro-F1 only.
Test evaluation is disabled; no Test metric is reported or used.

Completed contexts: 60 / 60.
Source commit(s): a529f896a85cc25c17a9a6729f3d691a663abab5

## Q1. Full relative to Base

The values below are descriptive paired-run results. They do not impose a pass/fail threshold.

| Dataset | Base Val Acc (mean ± SD) | Full Val Acc (mean ± SD) | Full−Base by seed | Base Macro-F1 (mean ± SD) | Full Macro-F1 (mean ± SD) |
|---|---:|---:|---|---:|---:|
| Movies | 0.5748 ± 0.0046 | 0.5549 ± 0.0021 | 42: -0.0165; 43: -0.0192; 44: -0.0240 | 0.4823 ± 0.0144 | 0.4512 ± 0.0152 |
| Toys | 0.8074 ± 0.0008 | 0.7917 ± 0.0021 | 42: -0.0143; 43: -0.0167; 44: -0.0162 | 0.7801 ± 0.0025 | 0.7595 ± 0.0071 |
| Grocery | 0.8316 ± 0.0035 | 0.8196 ± 0.0044 | 42: -0.0070; 43: -0.0120; 44: -0.0170 | 0.7331 ± 0.0118 | 0.7213 ± 0.0175 |
| ele-fashion | 0.8756 ± 0.0006 | 0.8769 ± 0.0007 | 42: 0.0002; 43: 0.0024; 44: 0.0015 | 0.7463 ± 0.0055 | 0.7484 ± 0.0047 |
| Reddit-S | 0.9611 ± 0.0026 | 0.9631 ± 0.0024 | 42: 0.0009; 43: 0.0003; 44: 0.0047 | 0.9213 ± 0.0045 | 0.9204 ± 0.0044 |

Per-seed validation values:

| Dataset | Seed | Base Acc / F1 | CRSA Acc / F1 | IATR Acc / F1 | Full Acc / F1 |
|---|---:|---:|---:|---:|---:|
| Movies | 42 | 0.5726 / 0.4776 | 0.5696 / 0.4909 | 0.5477 / 0.4311 | 0.5561 / 0.4584 |
| Movies | 43 | 0.5717 / 0.4709 | 0.5714 / 0.4704 | 0.5474 / 0.4358 | 0.5525 / 0.4337 |
| Movies | 44 | 0.5801 / 0.4986 | 0.5801 / 0.5044 | 0.5462 / 0.4472 | 0.5561 / 0.4615 |
| Toys | 42 | 0.8082 / 0.7801 | 0.8048 / 0.7779 | 0.7920 / 0.7624 | 0.7939 / 0.7676 |
| Toys | 43 | 0.8065 / 0.7826 | 0.8067 / 0.7828 | 0.7876 / 0.7622 | 0.7898 / 0.7540 |
| Toys | 44 | 0.8074 / 0.7777 | 0.8048 / 0.7792 | 0.7881 / 0.7519 | 0.7913 / 0.7570 |
| Grocery | 42 | 0.8281 / 0.7429 | 0.8348 / 0.7712 | 0.8179 / 0.7341 | 0.8211 / 0.7411 |
| Grocery | 43 | 0.8351 / 0.7363 | 0.8375 / 0.7565 | 0.8275 / 0.7357 | 0.8231 / 0.7078 |
| Grocery | 44 | 0.8316 / 0.7201 | 0.8340 / 0.7308 | 0.8193 / 0.7220 | 0.8146 / 0.7151 |
| ele-fashion | 42 | 0.8759 / 0.7423 | 0.8747 / 0.7464 | 0.8778 / 0.7490 | 0.8761 / 0.7520 |
| ele-fashion | 43 | 0.8749 / 0.7439 | 0.8780 / 0.7587 | 0.8751 / 0.7529 | 0.8773 / 0.7501 |
| ele-fashion | 44 | 0.8758 / 0.7526 | 0.8808 / 0.7710 | 0.8781 / 0.7561 | 0.8774 / 0.7431 |
| Reddit-S | 42 | 0.9594 / 0.9190 | 0.9575 / 0.9201 | 0.9591 / 0.9208 | 0.9604 / 0.9238 |
| Reddit-S | 43 | 0.9641 / 0.9265 | 0.9607 / 0.9137 | 0.9635 / 0.9232 | 0.9645 / 0.9220 |
| Reddit-S | 44 | 0.9597 / 0.9185 | 0.9588 / 0.9183 | 0.9626 / 0.9147 | 0.9645 / 0.9154 |

## Q2. CRSA-only

This isolates relation interpretation with uniform state readout.

| Dataset | Val Acc (mean ± SD) | Macro-F1 (mean ± SD) | Seeds present |
|---|---:|---:|---|
| Movies | 0.5737 ± 0.0056 | 0.4886 ± 0.0171 | 42, 43, 44 |
| Toys | 0.8054 ± 0.0011 | 0.7800 ± 0.0026 | 42, 43, 44 |
| Grocery | 0.8354 ± 0.0018 | 0.7528 ± 0.0204 | 42, 43, 44 |
| ele-fashion | 0.8778 ± 0.0031 | 0.7587 ± 0.0123 | 42, 43, 44 |
| Reddit-S | 0.9590 ± 0.0016 | 0.9173 ± 0.0033 | 42, 43, 44 |

## Q3. IATR-only

This isolates intrinsic-anchor trajectory reconciliation after plain propagation.

| Dataset | Val Acc (mean ± SD) | Macro-F1 (mean ± SD) | Seeds present |
|---|---:|---:|---|
| Movies | 0.5471 ± 0.0008 | 0.4380 ± 0.0083 | 42, 43, 44 |
| Toys | 0.7892 ± 0.0024 | 0.7588 ± 0.0060 | 42, 43, 44 |
| Grocery | 0.8216 ± 0.0052 | 0.7306 ± 0.0075 | 42, 43, 44 |
| ele-fashion | 0.8770 ± 0.0016 | 0.7527 ± 0.0036 | 42, 43, 44 |
| Reddit-S | 0.9617 ± 0.0023 | 0.9196 ± 0.0044 | 42, 43, 44 |

## Q4. Full relative to the single-module variants

Paired differences are descriptive within the same dataset and seed.

| Dataset | Full−CRSA Val Acc by seed | Full−IATR Val Acc by seed | Full−CRSA Macro-F1 by seed | Full−IATR Macro-F1 by seed |
|---|---|---|---|---|
| Movies | 42: -0.0135; 43: -0.0189; 44: -0.0240 | 42: 0.0084; 43: 0.0051; 44: 0.0099 | 42: -0.0324; 43: -0.0367; 44: -0.0429 | 42: 0.0273; 43: -0.0021; 44: 0.0143 |
| Toys | 42: -0.0109; 43: -0.0169; 44: -0.0135 | 42: 0.0019; 43: 0.0022; 44: 0.0031 | 42: -0.0104; 43: -0.0288; 44: -0.0222 | 42: 0.0051; 43: -0.0081; 44: 0.0051 |
| Grocery | 42: -0.0138; 43: -0.0143; 44: -0.0193 | 42: 0.0032; 43: -0.0044; 44: -0.0047 | 42: -0.0301; 43: -0.0487; 44: -0.0157 | 42: 0.0070; 43: -0.0279; 44: -0.0069 |
| ele-fashion | 42: 0.0014; 43: -0.0007; 44: -0.0035 | 42: -0.0016; 43: 0.0021; 44: -0.0007 | 42: 0.0056; 43: -0.0086; 44: -0.0279 | 42: 0.0030; 43: -0.0028; 44: -0.0130 |
| Reddit-S | 42: 0.0028; 43: 0.0038; 44: 0.0057 | 42: 0.0013; 43: 0.0009; 44: 0.0019 | 42: 0.0037; 43: 0.0083; 44: -0.0028 | 42: 0.0030; 43: -0.0012; 44: 0.0008 |

## Q5. Engineering pathology

Run status: 60 complete, 0 failed, 0 invalid, 0 missing, 0 other.
Peak allocated GPU memory: max 21953.0 MiB across completed contexts.
Run time: median 85.3 seconds per completed context.
No route-collapse diagnostic is part of this experiment; this report does not infer route health from task scores.
All expected contexts have valid validation-only artifacts.

## Protocol and interpretation

All reported statistics are descriptive. The across-dataset rows pool context-level observations and are not a substitute for per-dataset results.
No significance threshold or automatic acceptance rule was applied.
No link prediction, robustness, modality-missing, or Test evaluation was run.

## Model v1 freeze recommendation

Freeze this implementation and its 60-run artifacts as the evaluated Model v1 snapshot: all contexts completed, and the run audit found no NaN, OOM, or invalid checkpoint pathology.
The validation evidence does not support a general claim that Full improves over Base: Full is lower on Movies, Toys, and Grocery for every paired seed, while it is slightly higher on ele-fashion and Reddit-S.
Keep this version reproducible and treat any subsequent design change as a separately versioned experiment.
