# CRSA+IATR NC validation report

Mode: smoke. This report uses validation accuracy and validation Macro-F1 only.
Test evaluation is disabled; no Test metric is reported or used.

Completed contexts: 6 / 6.
Source commit(s): dc5a43955162428837154181757dfbe91e9faad3

## Q1. Full relative to Base

The values below are descriptive paired-run results. They do not impose a pass/fail threshold.

| Dataset | Base Val Acc (mean ± SD) | Full Val Acc (mean ± SD) | Full−Base by seed | Base Macro-F1 (mean ± SD) | Full Macro-F1 (mean ± SD) |
|---|---:|---:|---|---:|---:|
| Movies | 0.3293 ± NA | 0.3293 ± NA | 42: 0.0000 | 0.0248 ± NA | 0.0248 ± NA |
| Toys | NA ± NA | NA ± NA | incomplete | NA ± NA | NA ± NA |
| Grocery | NA ± NA | NA ± NA | incomplete | NA ± NA | NA ± NA |
| ele-fashion | NA ± NA | 0.4221 ± NA | incomplete | NA ± NA | 0.1254 ± NA |
| Reddit-S | NA ± NA | 0.1595 ± NA | incomplete | NA ± NA | 0.0295 ± NA |

Per-seed validation values:

| Dataset | Seed | Base Acc / F1 | CRSA Acc / F1 | IATR Acc / F1 | Full Acc / F1 |
|---|---:|---:|---:|---:|---:|
| Movies | 42 | 0.3293 / 0.0248 | 0.3293 / 0.0248 | 0.3293 / 0.0248 | 0.3293 / 0.0248 |
| Movies | 43 | NA / NA | NA / NA | NA / NA | NA / NA |
| Movies | 44 | NA / NA | NA / NA | NA / NA | NA / NA |
| Toys | 42 | NA / NA | NA / NA | NA / NA | NA / NA |
| Toys | 43 | NA / NA | NA / NA | NA / NA | NA / NA |
| Toys | 44 | NA / NA | NA / NA | NA / NA | NA / NA |
| Grocery | 42 | NA / NA | NA / NA | NA / NA | NA / NA |
| Grocery | 43 | NA / NA | NA / NA | NA / NA | NA / NA |
| Grocery | 44 | NA / NA | NA / NA | NA / NA | NA / NA |
| ele-fashion | 42 | NA / NA | NA / NA | NA / NA | 0.4221 / 0.1254 |
| ele-fashion | 43 | NA / NA | NA / NA | NA / NA | NA / NA |
| ele-fashion | 44 | NA / NA | NA / NA | NA / NA | NA / NA |
| Reddit-S | 42 | NA / NA | NA / NA | NA / NA | 0.1595 / 0.0295 |
| Reddit-S | 43 | NA / NA | NA / NA | NA / NA | NA / NA |
| Reddit-S | 44 | NA / NA | NA / NA | NA / NA | NA / NA |

## Q2. CRSA-only

This isolates relation interpretation with uniform state readout.

| Dataset | Val Acc (mean ± SD) | Macro-F1 (mean ± SD) | Seeds present |
|---|---:|---:|---|
| Movies | 0.3293 ± NA | 0.0248 ± NA | 42 |
| Toys | NA ± NA | NA ± NA | none |
| Grocery | NA ± NA | NA ± NA | none |
| ele-fashion | NA ± NA | NA ± NA | none |
| Reddit-S | NA ± NA | NA ± NA | none |

## Q3. IATR-only

This isolates intrinsic-anchor trajectory reconciliation after plain propagation.

| Dataset | Val Acc (mean ± SD) | Macro-F1 (mean ± SD) | Seeds present |
|---|---:|---:|---|
| Movies | 0.3293 ± NA | 0.0248 ± NA | 42 |
| Toys | NA ± NA | NA ± NA | none |
| Grocery | NA ± NA | NA ± NA | none |
| ele-fashion | NA ± NA | NA ± NA | none |
| Reddit-S | NA ± NA | NA ± NA | none |

## Q4. Full relative to the single-module variants

Paired differences are descriptive within the same dataset and seed.

| Dataset | Full−CRSA Val Acc by seed | Full−IATR Val Acc by seed | Full−CRSA Macro-F1 by seed | Full−IATR Macro-F1 by seed |
|---|---|---|---|---|
| Movies | 42: 0.0000 | 42: 0.0000 | 42: 0.0000 | 42: 0.0000 |
| Toys | incomplete | incomplete | incomplete | incomplete |
| Grocery | incomplete | incomplete | incomplete | incomplete |
| ele-fashion | incomplete | incomplete | incomplete | incomplete |
| Reddit-S | incomplete | incomplete | incomplete | incomplete |

## Q5. Engineering pathology

Run status: 6 complete, 0 failed, 0 invalid, 0 missing, 0 other.
Peak allocated GPU memory: max 21922.4 MiB across completed contexts.
Run time: median 1.9 seconds per completed context.
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
