# CoSI-MAG Core Story Ablation — Final Audit and Scientific Analysis

Analysis protocol: `cosi_mag_core_story_ablation_v1`. Generated from the completed formal matrix; no runs were launched, changed, or excluded.

## 1. Executive Summary

- Completeness: planned 63, complete 63, incomplete 0, missing 0; all official checker gates passed, including collision/duplicate check.
- Frozen provenance: branch `vnext`, commit `3549c8c39e651d716519af37f1cbfeb9da20aeb3`; all 63 run manifests agree with the current HEAD and one another.
- Run health: 63/63 passed; semantic audit: 63/63 passed.
- Largest positive Full-minus-ablation drop: `wo_adaptive_composition`, cloth-copurchase Hits@10 = +1.295 pp.
- Largest reversal (ablation exceeds Full mean): `wo_adaptive_composition`, ele-fashion Macro-F1 = -0.333 pp.
- All test-result comparisons are descriptive. The Full PDFs contain means and SDs, not matched seed-level values; no paired tests or 3/3 seed-drop statements are made.

## 2. Completeness and Provenance Audit

The complete official checker output is preserved at `outputs/core_story_ablation_analysis/completeness_checker_stdout.txt`. It reports `manifest_mismatch=PASS`, `config_mismatch=PASS`, `git_commit_mismatch=PASS`, `metric_schema=PASS`, and `duplicate_collision=PASS`.

All run manifests report commit `3549c8c39e651d716519af37f1cbfeb9da20aeb3` on `vnext`. Per-run configs and split pointers/hashes were compared across the three variants; only the top-level ablation identity and run-specific checkpoint path differ. Dataset graph/feature/self-loop config paths match across variants. Raw-uniform support preservation is also a code-level invariant of the frozen implementation; the run artifacts do not serialize a per-run edge-index hash.

Selection metadata agrees at manifest, metrics, and checkpoint level: NC uses `best_val_accuracy`; LP uses `best_val_mrr`. See `provenance_audit.csv`, `ablation_semantics_audit.csv`, and `run_health.csv` for every run.

Formal K verified from each resolved config and manifest: Movies=3, Toys=3, Grocery=2, ele-fashion=3, Reddit-S=3, sports-copurchase=3, cloth-copurchase=3.

Run-health summary: Best epoch min/median/max=16/62/190; runtime min/median/max=10.7/30.3/16488.2 s; peak GPU memory available for 63/63 runs, min/median/max=1715.1/3083.5/8993.7 MiB.

## 3. Full Reference

Full values were read only from the two user-designated PDFs. The analyzer did not use F1/F2 output metrics or infer Full values from the ablations.

- NC: `paper/tables/NC.pdf`, page 1, CoSI-MAG row; SHA-256 `e5147149f60668783b8f0993f946f3a798fd8371b284353af6b7c48bd9e7cd1f`.
- LP: `paper/tables/LP.pdf`, page 1, CoSI-MAG row; SHA-256 `718a58df13eb3380f81985b1373af78c581258b7d4b4c26f9c29f9c53a887b47`.
- Spread convention: NC and Sports Full SD are identified as population SD in their PDFs; Cloth Full SD is reproduced as printed, while its convention is not separately explicit in the LP caption. Ablation SDs are sample SD across the three formal seed-level run values.
- The Full references provide summary mean ± SD only. Seed pairing is unknown, so relative-drop tables contain no paired-seed tests.

## 4. Complete NC Results

All five datasets and all three variants are in `dataset_summary.csv`; raw seed values are in `per_seed_results.csv`. `appendix_nc.md` and `.tex` include the Full row and every formal NC dataset.

## 5. Complete LP Results

Sports and Cloth are retained separately in `dataset_summary.csv` and `appendix_lp.md`/`.tex`. The supplied LP PDF describes Sports as the formal benchmark and Cloth as a completeness entry; the current 63-run ablation scope includes both as requested.

## 6. Full-Relative Drops

`full_relative_drops.csv` reports `Full mean − ablation mean` in percentage points and relative drop as an auxiliary percentage. Positive means Full has the higher mean; negative means the ablation has the higher mean. Values inherit the two-decimal precision printed in the reference PDFs.

The heatmap is a wide diagnostic visualization (14 metric columns), not a manuscript-production figure. It encodes point differences only; uncertainty conventions and seed-level variation are documented in the tables, and matched Full seed values are unavailable. Final PDF QA found no text collisions or clipping.

Largest positive: wo_adaptive_composition on cloth-copurchase Hits@10 (+1.295 pp; relative 2.37%). Largest reverse: wo_adaptive_composition on ele-fashion Macro-F1 (-0.333 pp; relative -0.43%).

## 7. Stage-I Analysis — Relation-Level Semantic Discrepancy

- **Stage I — Modality-Aware Relation Calibration (`wo_relation_calibration`): Mixed / contradicted on some tasks.** Dataset pattern: positive `Grocery;ele-fashion`, mixed `Movies;Toys;Reddit-S`, reverse `sports-copurchase;cloth-copurchase`. Metric-cell directions: 7/18 positive (Full higher), 0 zero, 11/18 reverse; task positive shares NC=0.70, LP=0.00; median absolute drop=0.148 pp; median |drop|/quadrature of reported Full and ablation SDs=0.20; positive cells with drop at least that combined SD scale=0/18. These ratios are descriptive scale comparisons, not tests or standard errors. Dataset sign class requires all metrics in that dataset to share the sign. All comparisons are descriptive mean differences; no paired-seed tests are supported by the Full source.

- Full − ablation mean drops in pp (NC; positive means Full higher): Movies [Acc +0.248, F1 -0.245]; Toys [Acc -0.113, F1 +0.015]; Grocery [Acc +0.179, F1 +0.723]; ele-fashion [Acc +0.028, F1 +0.162]; Reddit-S [Acc +0.108, F1 -0.148].
- Full − ablation mean drops in pp (LP; positive means Full higher): sports-copurchase [MRR -0.084, H@1 -0.063, H@3 -0.208, H@10 -0.124]; cloth-copurchase [MRR -0.148, H@1 -0.085, H@3 -0.188, H@10 -0.235].

Across NC, 7/10 metric cells have Full above raw-uniform; across LP, 0/8 do. Grocery is the clearest NC positive case (Accuracy +0.179 pp; Macro-F1 +0.723 pp), while both LP datasets reverse on every metric.

A1 asks whether replacing modality-specific calibrated physical conductances with raw uniform conductances reduces downstream performance. Any dataset reversal is retained below; this does not establish that calibration is universally necessary.

## 8. Stage-II Analysis — Propagation-Induced Semantic Drift

- **Stage II — Semantic-Preserving Multi-Hop Propagation (`wo_semantic_anchor`): Moderately supported / dataset-dependent.** Dataset pattern: positive `Movies;Grocery;ele-fashion;sports-copurchase;cloth-copurchase`, mixed `Toys`, reverse `Reddit-S`. Metric-cell directions: 15/18 positive (Full higher), 0 zero, 3/18 reverse; task positive shares NC=0.70, LP=1.00; median absolute drop=0.130 pp; median |drop|/quadrature of reported Full and ablation SDs=0.24; positive cells with drop at least that combined SD scale=4/18. These ratios are descriptive scale comparisons, not tests or standard errors. Dataset sign class requires all metrics in that dataset to share the sign. All comparisons are descriptive mean differences; no paired-seed tests are supported by the Full source.

- Full − ablation mean drops in pp (NC; positive means Full higher): Movies [Acc +0.068, F1 +0.238]; Toys [Acc -0.137, F1 +0.020]; Grocery [Acc +0.511, F1 +0.762]; ele-fashion [Acc +0.018, F1 +0.106]; Reddit-S [Acc -0.123, F1 -0.332].
- Full − ablation mean drops in pp (LP; positive means Full higher): sports-copurchase [MRR +0.800, H@1 +0.647, H@3 +1.010, H@10 +1.118]; cloth-copurchase [MRR +0.086, H@1 +0.122, H@3 +0.023, H@10 +0.117].

Full is higher in 7/10 NC and 8/8 LP metric cells. Sports shows the largest concentrated Stage-II pattern (MRR +0.800 pp; Hits@10 +1.118 pp); Reddit-S reverses on both NC metrics.

This is descriptive consistency evidence for the semantic-preservation motivation, not proof that semantic drift causes downstream error.

## 9. Stage-III Analysis — Heterogeneous Multi-Hop Utilization

- **Stage III — Adaptive Multi-Order Composition (`wo_adaptive_composition`): Moderately supported / dataset-dependent.** Dataset pattern: positive `Movies;Toys;Grocery;sports-copurchase;cloth-copurchase`, mixed `ele-fashion`, reverse `Reddit-S`. Metric-cell directions: 15/18 positive (Full higher), 0 zero, 3/18 reverse; task positive shares NC=0.70, LP=1.00; median absolute drop=0.544 pp; median |drop|/quadrature of reported Full and ablation SDs=0.83; positive cells with drop at least that combined SD scale=7/18. These ratios are descriptive scale comparisons, not tests or standard errors. Dataset sign class requires all metrics in that dataset to share the sign. All comparisons are descriptive mean differences; no paired-seed tests are supported by the Full source.

- Full − ablation mean drops in pp (NC; positive means Full higher): Movies [Acc +0.628, F1 +0.702]; Toys [Acc +0.201, F1 +0.090]; Grocery [Acc +0.921, F1 +1.199]; ele-fashion [Acc +0.164, F1 -0.333]; Reddit-S [Acc -0.060, F1 -0.322].
- Full − ablation mean drops in pp (LP; positive means Full higher): sports-copurchase [MRR +0.553, H@1 +0.659, H@3 +0.535, H@10 +0.177]; cloth-copurchase [MRR +0.614, H@1 +0.371, H@3 +0.824, H@10 +1.295].

Full is higher in 7/10 NC and 8/8 LP metric cells. On the prespecified examples, w/o adaptive composition drops Movies by 0.628/0.702 pp (Accuracy/Macro-F1), Grocery by 0.921/1.199 pp, and Sports MRR/Hits@10 by 0.553/0.177 pp. Reddit-S reverses on both NC metrics; ele-fashion Macro-F1 also reverses.

A3 is the story-level comparison of adaptive composition against a uniform response-bank mean. Its positive and reverse effects are compared directly with A1/A2 in the CSV and heatmap; no dataset is selected post hoc.

## 10. Representative Main-Text Table

Representative datasets were fixed in advance (Movies, Grocery, Sports). The table bolds the actual highest mean in each column; Full is not automatically bolded.

# Main-paper representative table candidate

### Fixed representative datasets

| Variant | Movies Acc. | Movies Macro-F1 | Grocery Acc. | Grocery Macro-F1 | Sports MRR | Sports Hits@10 |
|---|---:|---:|---:|---:|---:|---:|
| Full CoSI-MAG | **56.36 ± 0.45** | 50.03 ± 0.32 | **83.42 ± 0.30** | **76.18 ± 0.44** | 37.48 ± 0.34 | 73.04 ± 0.38 |
| w/o Relation Calibration | 56.11 ± 0.66 | **50.28 ± 1.60** | 83.24 ± 0.68 | 75.46 ± 1.61 | **37.56 ± 0.37** | **73.16 ± 0.62** |
| w/o Semantic Anchor | 56.29 ± 0.69 | 49.79 ± 1.13 | 82.91 ± 0.59 | 75.42 ± 1.72 | 36.68 ± 0.34 | 71.92 ± 0.43 |
| w/o Adaptive Composition | 55.73 ± 0.78 | 49.33 ± 0.92 | 82.50 ± 0.34 | 74.98 ± 1.42 | 36.93 ± 0.41 | 72.86 ± 0.67 |

Test metrics are descriptive. Values are mean ± SD: ablation SD is sample SD (ddof=1) across seeds 42–44; Full SD is copied as printed in the supplied PDF (population SD for NC and Sports). No paired-seed inference.

## 11. Empirical Observation → Design → Downstream Ablation

- E0-A → Modality-Aware Relation Calibration → `wo_relation_calibration`.
- E0-B → Semantic-Preserving Multi-Hop Propagation → `wo_semantic_anchor`.
- E0-C → Adaptive Multi-Order Composition → `wo_adaptive_composition`.
- E0-D is not treated as a fourth independent challenge. Relation-state modulation of composition is a finer-grained mechanism question, outside the Core Table interpretation.

## 12. Reversals and Anomalies

- `wo_relation_calibration` reversals (ablation mean > Full mean): Movies Macro-F1 -0.245 pp; Toys Accuracy -0.113 pp; Reddit-S Macro-F1 -0.148 pp; sports-copurchase MRR -0.084 pp; sports-copurchase Hits@1 -0.063 pp; sports-copurchase Hits@3 -0.208 pp; sports-copurchase Hits@10 -0.124 pp; cloth-copurchase MRR -0.148 pp; cloth-copurchase Hits@1 -0.085 pp; cloth-copurchase Hits@3 -0.188 pp; cloth-copurchase Hits@10 -0.235 pp.
- `wo_semantic_anchor` reversals (ablation mean > Full mean): Toys Accuracy -0.137 pp; Reddit-S Accuracy -0.123 pp; Reddit-S Macro-F1 -0.332 pp.
- `wo_adaptive_composition` reversals (ablation mean > Full mean): ele-fashion Macro-F1 -0.333 pp; Reddit-S Accuracy -0.060 pp; Reddit-S Macro-F1 -0.322 pp.

Reversals are not removed or redefined. Seed variability is reported separately; because Full has no seed-level values, differences between means are not paired observations.

## 13. Paper-Safe Claims

### Results facts

- Stage I — Modality-Aware Relation Calibration (`wo_relation_calibration`): positive `Grocery;ele-fashion`, mixed `Movies;Toys;Reddit-S`, reverse `sports-copurchase;cloth-copurchase`. Metric-cell directions: 7/18 positive (Full higher), 0 zero, 11/18 reverse; task positive shares NC=0.70, LP=0.00; median absolute drop=0.148 pp; median |drop|/quadrature of reported Full and ablation SDs=0.20; positive cells with drop at least that combined SD scale=0/18. These ratios are descriptive scale comparisons, not tests or standard errors. Dataset sign class requires all metrics in that dataset to share the sign. All comparisons are descriptive mean differences; no paired-seed tests are supported by the Full source.
- Stage II — Semantic-Preserving Multi-Hop Propagation (`wo_semantic_anchor`): positive `Movies;Grocery;ele-fashion;sports-copurchase;cloth-copurchase`, mixed `Toys`, reverse `Reddit-S`. Metric-cell directions: 15/18 positive (Full higher), 0 zero, 3/18 reverse; task positive shares NC=0.70, LP=1.00; median absolute drop=0.130 pp; median |drop|/quadrature of reported Full and ablation SDs=0.24; positive cells with drop at least that combined SD scale=4/18. These ratios are descriptive scale comparisons, not tests or standard errors. Dataset sign class requires all metrics in that dataset to share the sign. All comparisons are descriptive mean differences; no paired-seed tests are supported by the Full source.
- Stage III — Adaptive Multi-Order Composition (`wo_adaptive_composition`): positive `Movies;Toys;Grocery;sports-copurchase;cloth-copurchase`, mixed `ele-fashion`, reverse `Reddit-S`. Metric-cell directions: 15/18 positive (Full higher), 0 zero, 3/18 reverse; task positive shares NC=0.70, LP=1.00; median absolute drop=0.544 pp; median |drop|/quadrature of reported Full and ablation SDs=0.83; positive cells with drop at least that combined SD scale=7/18. These ratios are descriptive scale comparisons, not tests or standard errors. Dataset sign class requires all metrics in that dataset to share the sign. All comparisons are descriptive mean differences; no paired-seed tests are supported by the Full source.

### Safe interpretation

- Stage I — Modality-Aware Relation Calibration: The effects are mixed: mean reversals occur on sports-copurchase, cloth-copurchase, so the component's downstream benefit is not universal across tasks/datasets.
- Stage II — Semantic-Preserving Multi-Hop Propagation: Removing the component lowers mean test scores consistently on Movies, Grocery, ele-fashion, sports-copurchase, cloth-copurchase, while outcomes vary elsewhere (Toys, Reddit-S); the benefit is dataset-dependent.
- Stage III — Adaptive Multi-Order Composition: Removing the component lowers mean test scores consistently on Movies, Toys, Grocery, sports-copurchase, cloth-copurchase, while outcomes vary elsewhere (ele-fashion, Reddit-S); the benefit is dataset-dependent.

### Avoid

- Do not claim all components are indispensable or that each module improves every dataset.
- Do not call mean differences statistically significant, use paired tests, or claim all three seeds drop; the Full references are aggregate-only.
- Do not state that semantic drift causes errors; the anchor ablation is downstream evidence consistent with a motivation, not causal identification.

## 14. Need for Progressive Decomposition

- Stage II and Stage III tie on all-positive dataset count (five each) and both are moderately supported/dataset-dependent. Stage III is descriptively stronger in magnitude relative to reported spread: median absolute gap 0.544 pp and 7 positive-effect cells reach at least the quadrature of the two reported SDs, versus 0.130 pp and 4 positive-effect cells for Stage II. This is a descriptive ranking, not significance.
- Weakest evidence is Stage I — Modality-Aware Relation Calibration (Stage I): only 7/18 metric cells favor Full, both LP datasets reverse on all four metrics, and none of its positive gaps reaches the quadrature of the reported SDs. The relation-calibration benefit is not established as a general downstream effect by this contrast.
- The Core ablations support Stage II and especially Stage III at a moderate, dataset-dependent level, but do not support a universal three-stage necessity claim. Stage I needs more qualification.
- The clearest reversal is A1 on both LP datasets (all eight metric cells favor raw-uniform ablation over Full); additional reversals include A1 on Toys Accuracy and Reddit-S Macro-F1, A2 on Reddit-S (both metrics) and Toys Accuracy, and A3 on Reddit-S (both metrics). These should be discussed, not removed.
- If distinguishing learned calibration from merely semantic/fixed calibration is central, Stage I is the most useful candidate for a prospective Raw Uniform → Fixed Semantic Calibration → Learned Semantic Calibration progression. This report does not reuse historical F2 result values.
- For Stage III, the current uniform-vs-adaptive result gives the story-level contrast. A global-only progression is not necessary before mechanism analysis; consider it only if the paper needs to attribute gains specifically to global, modality, node, or relation-conditioned terms.
- Recommendation: first perform mechanism analysis of dataset/task regimes and the observed A1 reversals; do not immediately rerun. If a specific Stage-I distinction remains necessary for the paper claim, preregister the fixed-calibration contrast. No new experiment is launched here.

## 15. Recommended Next Experiments

No experiments are authorized or started by this report. If a follow-up is later approved, preregister the specific contrast and retain all datasets/seeds; do not reuse old F2 metrics as formal Core Story results.

## Generated artifacts

- `outputs/core_story_ablation_analysis/completeness_checker_stdout.txt`
- `outputs/core_story_ablation_analysis/provenance_audit.csv`
- `outputs/core_story_ablation_analysis/ablation_semantics_audit.csv`
- `outputs/core_story_ablation_analysis/run_health.csv`
- `outputs/core_story_ablation_analysis/full_reference_values.csv`
- `outputs/core_story_ablation_analysis/per_seed_results.csv`
- `outputs/core_story_ablation_analysis/dataset_summary.csv`
- `outputs/core_story_ablation_analysis/full_relative_drops.csv`
- `outputs/core_story_ablation_analysis/stage_evidence_summary.csv`
- `outputs/core_story_ablation_analysis/main_table.md` / `.tex`
- `outputs/core_story_ablation_analysis/appendix_nc.md` / `.tex`; `appendix_lp.md` / `.tex`
- `outputs/core_story_ablation_analysis/all_datasets_table.tex` (one table float with NC and LP panels)
- `outputs/core_story_ablation_analysis/core_story_drop_heatmap.pdf` / `.png` / `.svg`
- `outputs/core_story_ablation_analysis/plot_core_story_drop_heatmap.py`
- `outputs/core_story_ablation_analysis/core_story_drop_heatmap.alignment.json`
- `outputs/core_story_ablation_analysis/core_story_drop_heatmap.collision-audit.json` (0 findings; no collision overlay was needed)
- `docs/cosi_mag_core_story_ablation_analysis.md`
