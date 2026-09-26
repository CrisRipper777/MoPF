# Figure 5 Robustness Results

## 1. Evaluation completeness

The complete matched-split checkpoint-only matrix contains **702/702 planned conditions**: 432 random structural-noise rows and 270 semantic-conflict rows. This includes 126 cached clean rows at `rho=0` and 576 nonzero CUDA inference runs. The clean rows reuse the verified matched clean gate; no clean inference was repeated. No model was trained, fine-tuned, or modified.

The random-noise schedule was `rho = 0, 0.05, 0.10, 0.20, 0.30, 0.40`. The semantic-conflict schedule was `rho = 0, 0.05, 0.10, 0.20, 0.30`. Model seeds 42/43/44 and perturbation seeds 1001/1002/1003 were all present.

## 2. Matched-split / perturbation audit

The pre-evaluation audit and post-evaluation audit both passed. The audits verified condition completeness, cached `rho=0` retention exactly equal to 100%, unchanged checkpoint SHA256 values, shared test split within each dataset, identical perturbation graph hashes across compared models/seeds, requested integer edge budgets, nested perturbation budgets, and the frozen raw-feature category and balance rules for semantic-conflict edges. Perturbation lists and candidate pools were not regenerated.

The matched split SHA256 values were `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211` (Movies) and `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e` (Grocery). Clean canonical edge counts were 80,401 and 71,131, respectively. The post-evaluation machine-readable audit is [robustness_post_evaluation_audit.json](../outputs/robustness_analysis/robustness_post_evaluation_audit.json); the pre-evaluation category and plan audit is [robustness_pre_evaluation_audit.json](../outputs/robustness_analysis/robustness_pre_evaluation_audit.json).

All paper-facing means and SDs use the following hierarchy: average the three perturbation seeds within each model seed, then compute the mean and **population SD (ddof=0)** across the three model seeds. Perturbation-seed variability is retained separately in the summaries. No p-values were computed.

## 3. Random Structural Noise results

At the maximum noise level (`rho=0.40`), Full CoSI-MAG retained 97.95 ± 0.82% Accuracy and 96.07 ± 0.80% Macro-F1 on Movies, and 96.23 ± 0.28% Accuracy and 95.31 ± 0.13% Macro-F1 on Grocery. Values are checkpoint-specific ratios to each checkpoint's own clean result.

The Accuracy retention curves generally declined as more random edges were injected. For Full CoSI-MAG, Movies changed from 100% to 97.95%; Grocery changed from 100% to 96.23%. The relation-calibration gap was negative at Movies `rho=0.05` (-0.31 pp), approximately zero at `rho=0.10`, and positive at larger ratios. It was positive from `rho=0.05` in Grocery, peaked at `rho=0.30`, then narrowed slightly at `rho=0.40`. Thus, the relation-calibration advantage at stronger noise was present in both datasets, but it did not widen monotonically across all levels.

The semantic-anchor ablation retained more Accuracy than Full CoSI-MAG at every nonzero noise ratio on both datasets. Macro-F1 was mixed: Full was slightly higher than the ablation at the Movies maximum ratio, while the ablation retained more on Grocery. DiP was below Full at the Movies maximum ratio, but slightly above Full for both metrics on Grocery.

## 4. Semantic-Conflict results

At the maximum conflict level (`rho=0.30`), Full CoSI-MAG retained 98.33 ± 0.05% Accuracy and 96.85 ± 0.17% Macro-F1 on Movies, and 97.53 ± 0.26% Accuracy and 97.24 ± 0.57% Macro-F1 on Grocery.

Mean Full-minus-w/o-Relation-Calibration gaps at the final ratio were positive on both datasets and both metrics: Movies, +0.44 ± 0.18 pp Accuracy and +0.69 ± 0.89 pp Macro-F1; Grocery, +0.36 ± 0.25 pp Accuracy and +0.40 ± 0.97 pp Macro-F1. The largest mean Accuracy gap was +0.44 pp at `rho=0.30` on Movies and +0.51 pp at `rho=0.20` on Grocery. For Macro-F1, the largest mean gap occurred at `rho=0.20` in both datasets (+0.84 pp and +0.69 pp, respectively), before narrowing at the final ratio.

Full CoSI-MAG out-retained DiP on both metrics for Movies at `rho=0.30`; on Grocery, DiP was 0.02 pp above Full on Accuracy while Full was 0.10 pp above DiP on Macro-F1. These are small differences and are descriptive only.

## 5. Full vs w/o Relation Calibration

The table gives the final-ratio gap (Full minus ablation) and the maximum mean gap over the full `rho` schedule. Gaps are retention percentage points; SD is population SD across paired model-seed gaps. Spearman values are descriptive summaries across the few scheduled `rho` levels.

| Perturbation | Dataset | Metric | Final gap (mean ± SD) | Maximum mean gap (at rho) | Spearman rho vs gap |
|---|---|---|---:|---:|---:|
| Random noise | Movies | Accuracy | +0.78 ± 0.98 | +0.78 (0.40) | 0.83 |
| Random noise | Movies | Macro-F1 | +1.01 ± 1.90 | +1.01 (0.40) | 0.94 |
| Random noise | Grocery | Accuracy | +0.64 ± 0.37 | +0.71 (0.30) | 0.94 |
| Random noise | Grocery | Macro-F1 | +0.51 ± 0.57 | +0.73 (0.30) | 0.71 |
| Semantic conflict | Movies | Accuracy | +0.44 ± 0.18 | +0.44 (0.30) | 0.90 |
| Semantic conflict | Movies | Macro-F1 | +0.69 ± 0.89 | +0.84 (0.20) | 0.80 |
| Semantic conflict | Grocery | Accuracy | +0.36 ± 0.25 | +0.51 (0.20) | 0.90 |
| Semantic conflict | Grocery | Macro-F1 | +0.40 ± 0.97 | +0.69 (0.20) | 0.80 |

At high corruption, these summaries favor Full CoSI-MAG over the relation-calibration ablation in all four dataset/perturbation combinations. Seed-level variation is substantial for some Movies and Macro-F1 gaps, so the result should be described as a consistent mean direction at the tested final ratios rather than as a statistically established effect.

## 6. Full vs w/o Semantic Anchor

This comparison is defined only for random structural noise; the ablation was not part of the semantic-conflict matrix. At `rho=0.40`, Full-minus-ablation Accuracy retention was -0.07 ± 0.48 pp on Movies and -0.56 ± 0.74 pp on Grocery. For Macro-F1 it was +0.05 ± 0.58 pp on Movies and -1.05 ± 0.98 pp on Grocery. The semantic-anchor robustness benefit is therefore not supported consistently by this experiment.

## 7. Full vs DiP

At the maximum random-noise ratio, Full CoSI-MAG retained more than DiP on Movies by 1.26 pp Accuracy and 3.34 pp Macro-F1. On Grocery, DiP was slightly higher by 0.16 pp Accuracy and 0.39 pp Macro-F1. Under maximum semantic conflict, Full was higher than DiP on Movies by 0.91 pp Accuracy and 1.39 pp Macro-F1; on Grocery, Full was 0.02 pp lower on Accuracy and 0.10 pp higher on Macro-F1. The comparator result varies by dataset and metric.

## 8. Random vs Semantic-Conflict comparison

`ConflictExtraDamage = random retention - conflict retention`; positive values mean semantic-conflict injection was more damaging than random noise at the matched ratio. The values below are for Full CoSI-MAG Accuracy and are mean ± population SD across model seeds.

| Dataset | rho=0.05 | rho=0.10 | rho=0.20 | rho=0.30 |
|---|---:|---:|---:|---:|
| Movies | -0.57 ± 0.25 | -0.58 ± 0.49 | -0.23 ± 0.65 | +0.12 ± 0.74 |
| Grocery | -0.08 ± 0.07 | -0.24 ± 0.35 | -0.61 ± 0.23 | -0.46 ± 0.33 |

Conflict injection was therefore less damaging than random noise for Full CoSI-MAG at the first three matched Movies ratios and all four Grocery ratios; Movies reversed slightly at `rho=0.30`. The direction is not universal across models either; all model-specific comparisons are retained in [random_vs_conflict_comparison.csv](../outputs/robustness_analysis/random_vs_conflict_comparison.csv).

`CalibrationExtraBenefit = relation-calibration gap under conflict - relation-calibration gap under random noise`. For Accuracy, this was +0.62, +0.11, +0.01, and -0.31 pp across the four matched ratios on Movies; on Grocery it was +0.00, -0.24, -0.06, and -0.35 pp. This does not support a general claim that relation calibration is especially beneficial under semantic conflict.

## 9. Final-ratio retention and AUC

Each cell reports final-ratio retention mean ± population SD / normalized trapezoidal retention AUC mean ± population SD. The AUC is divided by the `rho` interval, so a flat 100% curve has AUC 100. These diagnostics supplement, and do not replace, the plotted full curves.

| Dataset | Condition | Model | Accuracy final / AUC (%) | Macro-F1 final / AUC (%) |
|---|---|---|---:|---:|
| Movies | Random noise | CoSI-MAG | 97.95 ± 0.82 / 98.79 ± 0.58 | 96.07 ± 0.80 / 98.18 ± 0.53 |
| Movies | Random noise | w/o Relation Calibration | 97.18 ± 0.26 / 98.45 ± 0.13 | 95.06 ± 1.11 / 97.69 ± 0.76 |
| Movies | Random noise | w/o Semantic Anchor | 98.02 ± 0.79 / 99.00 ± 0.62 | 96.02 ± 0.68 / 97.92 ± 0.96 |
| Movies | Random noise | DiP | 96.70 ± 0.72 / 98.09 ± 0.37 | 92.73 ± 1.58 / 95.82 ± 0.97 |
| Grocery | Random noise | CoSI-MAG | 96.23 ± 0.28 / 97.86 ± 0.21 | 95.31 ± 0.13 / 97.16 ± 0.02 |
| Grocery | Random noise | w/o Relation Calibration | 95.58 ± 0.09 / 97.36 ± 0.10 | 94.80 ± 0.67 / 96.64 ± 0.57 |
| Grocery | Random noise | w/o Semantic Anchor | 96.79 ± 1.01 / 98.12 ± 0.67 | 96.37 ± 0.93 / 97.66 ± 0.65 |
| Grocery | Random noise | DiP | 96.39 ± 0.47 / 97.89 ± 0.28 | 95.70 ± 0.43 / 97.35 ± 0.05 |
| Movies | Semantic conflict | CoSI-MAG | 98.33 ± 0.05 / 99.28 ± 0.20 | 96.85 ± 0.17 / 98.74 ± 0.15 |
| Movies | Semantic conflict | w/o Relation Calibration | 97.89 ± 0.23 / 99.00 ± 0.27 | 96.16 ± 1.04 / 98.20 ± 0.65 |
| Movies | Semantic conflict | DiP | 97.42 ± 0.52 / 98.65 ± 0.29 | 95.46 ± 1.14 / 97.68 ± 0.72 |
| Grocery | Semantic conflict | CoSI-MAG | 97.53 ± 0.26 / 98.62 ± 0.18 | 97.24 ± 0.57 / 98.60 ± 0.25 |
| Grocery | Semantic conflict | w/o Relation Calibration | 97.16 ± 0.13 / 98.31 ± 0.13 | 96.83 ± 0.71 / 98.26 ± 0.54 |
| Grocery | Semantic conflict | DiP | 97.55 ± 0.29 / 98.61 ± 0.16 | 97.14 ± 0.22 / 98.21 ± 0.20 |

The absolute Accuracy and Macro-F1 at clean and maximum `rho` are available in [CSV](../outputs/robustness_analysis/absolute_performance_appendix.csv), [Markdown](../outputs/robustness_analysis/absolute_performance_appendix.md), and [LaTeX](../outputs/robustness_analysis/absolute_performance_appendix.tex).

## 10. Important reversals / anomalies

- Semantic-conflict edges were usually less damaging than random edges at matched ratios, with only a small positive Full-Accuracy conflict extra-damage estimate on Movies at `rho=0.30`.
- The Full-minus-w/o-Semantic-Anchor Accuracy gap was negative at all nonzero random-noise levels in both datasets. This is a direct reversal of a simple robustness-benefit expectation for the anchor.
- Full CoSI-MAG did not beat DiP uniformly. DiP slightly exceeded Full on Grocery random-noise final retention for both metrics, and on Grocery semantic-conflict Accuracy.
- The relation-calibration gap was not monotone at every ratio. It peaked at `rho=0.30` rather than `0.40` in Grocery random noise and semantic conflict; Movies semantic-conflict Macro-F1 also peaked at `rho=0.20`.
- Some paired gap SDs are large relative to their means, especially Movies random-noise Macro-F1 and semantic-conflict Macro-F1. With three model seeds, these are descriptive summaries, not inferential evidence.

## 11. Paper-safe claims

For the evaluated matched-split checkpoints and prepared perturbation schedules, Full CoSI-MAG had higher **mean final-ratio retention** than the w/o-Relation-Calibration model on both datasets and both metrics, across random noise and semantic conflict. The advantage varied in size and sometimes in sign at lower noise levels. This supports a bounded statement that relation calibration is associated with improved retention at stronger tested corruption levels in these settings.

## 12. Claims not supported

These results do not support universal robustness superiority, a consistent robustness benefit from the semantic anchor, monotonic widening of the relation-calibration gap, or a general claim that semantic-conflict injection is more harmful than random noise or that calibration is especially beneficial for conflict edges. They do not identify causal effects of real-world noisy edges and do not establish adversarial or certified robustness. No statistical significance claim is made.

## 13. Recommended Figure 5 caption

**Figure 5 | Checkpoint-specific retention under matched structural perturbations.** (a,b) Random structural-noise injection on Movies and Grocery; (c,d) balanced semantic-conflict edge injection on the same matched test splits. The x-axis gives injected physical edges as a fraction of the clean canonical edge set. Accuracy retention is normalized to each checkpoint's own clean test Accuracy, with the clean reference fixed at 100%. Curves show the mean across model seeds after averaging the three perturbation seeds within each model seed; error bars show population SD across model seeds. Semantic-anchor ablations are included only in the random-noise panels. The experiment compares the specified prepared perturbations and does not establish adversarial or certified robustness.

## 14. Recommended paper paragraph

Under the matched-split perturbation protocol, Full CoSI-MAG retained more Accuracy and Macro-F1 than the w/o-Relation-Calibration variant at the maximum tested corruption level on both Movies and Grocery, although the size of this mean advantage varied across datasets, metrics, and perturbation ratios. The comparison with w/o-Semantic-Anchor and DiP was mixed: removing the anchor yielded higher Accuracy retention under random noise, and DiP slightly exceeded Full on selected Grocery endpoints. Semantic-conflict injections were not uniformly more damaging than random noise, and the relation-calibration gap was not consistently larger under conflict. These checkpoint-only results therefore support a qualified association between relation calibration and retention under stronger tested graph corruption, without establishing universal or adversarial robustness.
