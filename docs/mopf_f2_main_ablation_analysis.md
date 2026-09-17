# MoPF F2 main ablation analysis

> Scope: formal result audit, aggregation, and exploratory scientific diagnosis only. No model, training, hyperparameter, ablation-definition, or interaction-training changes were made.

## 1. Completeness audit

The expected main matrix contains 108 runs: 5 NC datasets × 6 variants × 3 seeds plus 1 LP dataset × 6 variants × 3 seeds. Full MoPF is explicitly reused from the audited F1 formal artifacts under `outputs/f1_final_execution/`; the five ablation variants are read from `outputs/f2_ablation/`.

Main status: 108/108 complete; 0 missing or inconsistent.

No missing, unreadable, non-finite, failed/incomplete, seed-mismatched, checkpoint-selection-mismatched, or cross-variant configuration-inconsistent run was found in the main matrix.

Out-of-scope residuals were retained and not used: 3 complete interaction artifact(s) and 1 incomplete artifact(s). These are recorded in `f2_main_completeness.csv` and do not enter any table or diagnosis.

- `nc/Grocery/wo_anchor_tcpr/seed42`: `EXTRA_COMPLETE` — out of scope.
- `nc/Movies/wo_anchor_tcpr/seed42`: `EXTRA_COMPLETE` — out of scope.
- `nc/Toys/wo_anchor_tcpr/seed42`: `EXTRA_COMPLETE` — out of scope.
- `nc/ele-fashion/wo_anchor_tcpr/seed42`: `EXTRA_INCOMPLETE` — missing_required_artifact.

The reported standard deviation is the population standard deviation across the three pre-specified seeds (ddof=0). The independent experimental unit is a dataset × seed run; no significance terminology or p-values are used.

## 2. Main table

The requested candidate table is generated as `outputs/f2_ablation/f2_main_paper_table.md` and `.tex`. Sports MRR denotes test MRR. Bold marks the best mean in each displayed column; an ablation is not suppressed when it exceeds Full.

| Variant | NC Avg Acc | NC Avg Macro-F1 | Sports MRR | H@1 | H@10 |
|---|---:|---:|---:|---:|---:|
| Full MoPF | 0.8071 ± 0.0015 | 0.7432 ± 0.0035 | 0.3748 ± 0.0034 | 0.2156 ± 0.0039 | 0.7304 ± 0.0038 |
| w/o learned semantic calibration | 0.8074 ± 0.0024 | 0.7465 ± 0.0049 | 0.3720 ± 0.0014 | 0.2131 ± 0.0012 | 0.7264 ± 0.0029 |
| w/o semantic anchor | 0.8084 ± 0.0015 | 0.7456 ± 0.0045 | 0.3718 ± 0.0055 | 0.2134 ± 0.0023 | 0.7242 ± 0.0146 |
| w/o TCPR | 0.8055 ± 0.0019 | 0.7435 ± 0.0046 | 0.3780 ± 0.0054 | 0.2180 ± 0.0040 | 0.7364 ± 0.0086 |
| w/o node adaptation | 0.8068 ± 0.0012 | 0.7445 ± 0.0035 | 0.3765 ± 0.0017 | 0.2178 ± 0.0019 | 0.7319 ± 0.0018 |
| w/o modality adaptation | 0.8068 ± 0.0020 | 0.7453 ± 0.0025 | 0.3812 ± 0.0024 | 0.2196 ± 0.0019 | 0.7423 ± 0.0038 |

NC averages are arithmetic averages of the five dataset-level seed means; node-level pooling was not used. The complete per-dataset candidate is `outputs/f2_ablation/f2_appendix_candidate_tables.md`.

## 3. Per-component diagnosis

Classification rubric used here: Strongly supported requires positive average NC and LP drops, broad NC cell coverage, and mostly same-direction paired seeds; Moderately supported allows positive contribution with mixed coverage; Weak / dataset-dependent is localized or mixed; Not supported means the average directions do not support a contribution. These are evidence labels for this audit, not statistical significance claims.

| Component | Mean NC drop across 10 NC cells | Mean LP drop across 4 test metrics | NC positive-cell coverage | Paired positive fraction | Audit label |
|---|---:|---:|---:|---:|---|
| w/o learned semantic calibration | -0.00181 | 0.00299 | 20% | 44% | Weak / dataset-dependent |
| w/o semantic anchor | -0.00181 | 0.00365 | 30% | 38% | Weak / dataset-dependent |
| w/o TCPR | 0.00069 | -0.00410 | 40% | 47% | Weak / dataset-dependent |
| w/o node adaptation | -0.00050 | -0.00189 | 60% | 42% | Not supported |
| w/o modality adaptation | -0.00090 | -0.00813 | 40% | 36% | Not supported |

### A1 — learned semantic calibration (E0-A)

- Movies: Accuracy drop -0.00000, Macro-F1 drop 0.00018; paired directions 1/3 and 2/3. E0-A mean rank gap=0.2916, Spearman TV=0.2039.
- Toys: Accuracy drop 0.00081, Macro-F1 drop -0.00184; paired directions 2/3 and 1/3. E0-A mean rank gap=0.3013, Spearman TV=0.1573.
- Grocery: Accuracy drop -0.00010, Macro-F1 drop -0.00502; paired directions 1/3 and 1/3. E0-A mean rank gap=0.2976, Spearman TV=0.1641.
- ele-fashion: Accuracy drop -0.00234, Macro-F1 drop -0.00862; paired directions 0/3 and 0/3. E0-A mean rank gap=0.2302, Spearman TV=0.4848.
- Reddit-S: Accuracy drop -0.00000, Macro-F1 drop -0.00115; paired directions 1/3 and 1/3. E0-A mean rank gap=0.3410, Spearman TV=-0.0169.

Descriptive reading: the calibration-removal drops are compared with E0-A rank-gap and TV-association summaries, but no correlation or causal claim is made. Reddit-S has the largest E0-A rank-gap severity but does not show a positive downstream calibration-removal drop, whereas ele-fashion has the strongest raw TV association but shows reverse-direction NC differences. Thus these results do not support a simple monotone discrepancy-to-drop pattern.

### A2 — semantic anchor (E0-B)

- Movies: Accuracy drop -0.00150, Macro-F1 drop -0.00305; E0-B mean CKA formal-K drop=0.8348, mean cosine drop=0.0929.
- Toys: Accuracy drop 0.00089, Macro-F1 drop 0.00044; E0-B mean CKA formal-K drop=0.8150, mean cosine drop=0.0746.
- Grocery: Accuracy drop -0.00332, Macro-F1 drop -0.00590; E0-B mean CKA formal-K drop=0.9112, mean cosine drop=0.0662.
- ele-fashion: Accuracy drop -0.00067, Macro-F1 drop 0.00027; E0-B mean CKA formal-K drop=0.7360, mean cosine drop=0.1000.
- Reddit-S: Accuracy drop -0.00189, Macro-F1 drop -0.00335; E0-B mean CKA formal-K drop=0.2857, mean cosine drop=0.0650.

Anchor removal does not overall hurt the NC summary (its Full-relative NC drops are negative), while it does lower the sports LP metrics on average. The E0-B retention quantities are empirical severity descriptors and do not establish that drift caused either downstream pattern.

### A3 — TCPR (E0-D)

- Movies: Accuracy drop 0.00800, Macro-F1 drop -0.00013, Accuracy paired direction 3/3; E0-D mean absolute partial rho=0.2650.
- Toys: Accuracy drop -0.00064, Macro-F1 drop -0.00303, Accuracy paired direction 1/3; E0-D mean absolute partial rho=0.3417.
- Grocery: Accuracy drop -0.00010, Macro-F1 drop -0.00173, Accuracy paired direction 2/3; E0-D mean absolute partial rho=0.3114.
- ele-fashion: Accuracy drop -0.00058, Macro-F1 drop 0.00255, Accuracy paired direction 1/3; E0-D mean absolute partial rho=0.3990.
- Reddit-S: Accuracy drop 0.00136, Macro-F1 drop 0.00119, Accuracy paired direction 3/3; E0-D mean absolute partial rho=0.8791.

TCPR is not a stable all-task drop: its NC average drop is positive, but its LP average drop is negative, and only a minority of NC cells are positive. E0-D association strength is used only for qualitative comparison; this is not a causal mediation test.

### A4/A5 — node and modality adaptation (E0-C)

- Movies: w/o node adaptation drops Accuracy/Macro-F1 by 0.00110/-0.00261; w/o modality adaptation by 0.00080/-0.00028. E0-C profile-L1=0.0872, order-gap=0.0112.
- Toys: w/o node adaptation drops Accuracy/Macro-F1 by 0.00105/0.00057; w/o modality adaptation by 0.00056/-0.00013. E0-C profile-L1=0.2454, order-gap=0.0854.
- Grocery: w/o node adaptation drops Accuracy/Macro-F1 by -0.00225/-0.00360; w/o modality adaptation by -0.00088/-0.00451. E0-C profile-L1=0.4658, order-gap=0.2092.
- ele-fashion: w/o node adaptation drops Accuracy/Macro-F1 by 0.00017/-0.00227; w/o modality adaptation by -0.00045/-0.00687. E0-C profile-L1=0.1052, order-gap=0.0101.
- Reddit-S: w/o node adaptation drops Accuracy/Macro-F1 by 0.00147/0.00139; w/o modality adaptation by 0.00136/0.00142. E0-C profile-L1=0.1148, order-gap=0.0239.

Grocery has the largest E0-C heterogeneity descriptor but both adaptation removals improve rather than hurt its NC summary; Toys shows small positive drops for node adaptation and mixed effects for modality adaptation. Movies and ele-fashion provide lower-heterogeneity contrasts with dataset-specific reversals. The comparison remains descriptive and exploratory.

## 4. Per-dataset observations

- **Movies**: mean of the two NC drops by ablation: w/o TCPR 0.0039, w/o modality adaptation 0.0003, w/o learned semantic calibration 0.0001, w/o node adaptation -0.0008, w/o semantic anchor -0.0023.
- **Toys**: mean of the two NC drops by ablation: w/o node adaptation 0.0008, w/o semantic anchor 0.0007, w/o modality adaptation 0.0002, w/o learned semantic calibration -0.0005, w/o TCPR -0.0018.
- **Grocery**: mean of the two NC drops by ablation: w/o TCPR -0.0009, w/o learned semantic calibration -0.0026, w/o modality adaptation -0.0027, w/o node adaptation -0.0029, w/o semantic anchor -0.0046.
- **ele-fashion**: mean of the two NC drops by ablation: w/o TCPR 0.0010, w/o semantic anchor -0.0002, w/o node adaptation -0.0010, w/o modality adaptation -0.0037, w/o learned semantic calibration -0.0055.
- **Reddit-S**: mean of the two NC drops by ablation: w/o node adaptation 0.0014, w/o modality adaptation 0.0014, w/o TCPR 0.0013, w/o learned semantic calibration -0.0006, w/o semantic anchor -0.0026.
- **sports-copurchase Test MRR**: ablation gains relative to Full are retained: w/o TCPR 0.00321 gain, w/o node adaptation 0.00170 gain, w/o modality adaptation 0.00637 gain.
- **sports-copurchase Hits@1**: ablation gains relative to Full are retained: w/o TCPR 0.00234 gain, w/o node adaptation 0.00220 gain, w/o modality adaptation 0.00402 gain.
- **sports-copurchase Hits@3**: ablation gains relative to Full are retained: w/o TCPR 0.00483 gain, w/o node adaptation 0.00217 gain, w/o modality adaptation 0.01030 gain.
- **sports-copurchase Hits@10**: ablation gains relative to Full are retained: w/o TCPR 0.00601 gain, w/o node adaptation 0.00149 gain, w/o modality adaptation 0.01182 gain.

## 5. Seed stability

Across paired rows, 4 are 3/3 same-direction drops (strong seed consistency under the audit rule), 40 are 1/3 (weak evidence), and 16 are 0/3 in the reverse direction. Rows with 2/3 are mixed direction.

Largest observed seed standard-deviation flags:
- sports-copurchase / wo_semantic_anchor test_hits@10 std=0.0146
- Grocery / wo_learned_semantic_calibration Macro-F1 std=0.0144
- Grocery / wo_semantic_anchor Macro-F1 std=0.0132
- Grocery / wo_tcpr Macro-F1 std=0.0129
- Grocery / wo_node_adaptation Macro-F1 std=0.0124
- Reddit-S / wo_modality_adaptation Macro-F1 std=0.0109
- Grocery / full Macro-F1 std=0.0102
- Reddit-S / wo_node_adaptation Macro-F1 std=0.0102

The paired-drop file preserves the three seed differences, their mean/std/median, and the positive count; it should be preferred over a comparison of two unpaired means when discussing consistency.

## 6. Relation to E0 empirical studies

The E0 alignment file is `outputs/f2_ablation/f2_e0_alignment_summary.csv`. The comparisons used are:
- E0-A: mean/median/q90 rank gaps and Spearman text–visual association for relation-level semantic discrepancy.
- E0-B: mean CKA and cosine retention drops at formal K across text and visual modalities for propagation-induced drift.
- E0-C: seed-averaged modality profile L1 and order-gap measures for utilization heterogeneity.
- E0-D: seed- and modality-averaged absolute partial rho for relation-condition/utilization association.

These are qualitative alignments only. The analysis does not fit or report a correlation, does not claim mechanism identification, and does not convert E0 severity into a causal prediction.

## 7. Anomalies

No main-run missing artifact, non-finite metric, failed/incomplete marker, seed mismatch, checkpoint-selection mismatch, runtime-crash pattern, duplicate main output, or cross-variant configuration mismatch was detected.
The out-of-scope `wo_anchor_tcpr` residuals are an audit anomaly in the directory, not evidence for a main result and not used in any aggregation.
A negative Full-relative drop means the ablation exceeds Full; such values are preserved in the CSV files and should not be absolute-valued.

## 8. Interaction-phase recommendation

This audit alone does not authorize or start interaction training. A cautious interaction phase is worth considering only after the residual out-of-scope artifacts and any paper-table review are closed; the main matrix itself is complete if Section 1 reports 108/108.
Recommended exploratory interactions, ranked by the paired component evidence available here, are:
1. semantic anchor × TCPR (`wo_anchor_tcpr`): tests whether propagation stabilization and relation-conditioned utilization are complementary.
2. node adaptation × TCPR (`wo_node_tcpr`): tests whether node heterogeneity and relation-utilization conditioning overlap.
3. modality adaptation × TCPR (`wo_modality_tcpr`): tests whether modality-specific residuals and TCPR provide complementary handling of utilization differences.
These are hypotheses for a later, explicitly pre-registered interaction design—not conclusions from the partial residual `wo_anchor_tcpr` artifacts already present.

## 9. Paper-writing-safe claims

- **w/o learned semantic calibration**: Weak / dataset-dependent; report the mean drops, dataset coverage, and paired seed directions together. Avoid saying that the component is necessary or causally responsible.
- **w/o semantic anchor**: Weak / dataset-dependent; report the mean drops, dataset coverage, and paired seed directions together. Avoid saying that the component is necessary or causally responsible.
- **w/o TCPR**: Weak / dataset-dependent; report the mean drops, dataset coverage, and paired seed directions together. Avoid saying that the component is necessary or causally responsible.
- **w/o node adaptation**: Not supported; report the mean drops, dataset coverage, and paired seed directions together. Avoid saying that the component is necessary or causally responsible.
- **w/o modality adaptation**: Not supported; report the mean drops, dataset coverage, and paired seed directions together. Avoid saying that the component is necessary or causally responsible.
- It is safe to state that the formal ablations provide descriptive evidence of component contribution under the fixed F2 protocol, with effects varying across datasets and metrics.
- It is safe to state when Full exceeds an ablation on 3/3 seeds; call this strong seed consistency under the present audit rule, not statistical significance.
- It is not safe to claim that E0-A/B/C/D empirically prove the corresponding downstream mechanism; they motivate qualitative alignment only.
- Any ablation improvement over Full must remain visible as a negative Full-relative drop and be discussed as a dataset/task-specific result.

### Final answers

1. **Most stable downstream contributions:** none meets the strong cross-task stability rule; w/o TCPR is only the relative leader at 47% positive paired directions; the seed and task reversals mean this is not strong evidence of a uniformly necessary mechanism.
2. **Smaller contribution but stronger mechanism evidence:** No component meets a separate small-effect/strong-direction rule in this audit.
3. **Ablation improvements:** w/o learned semantic calibration, w/o semantic anchor, w/o TCPR, w/o node adaptation, w/o modality adaptation
4. **Implementation suspicion:** none is warranted from completeness/config/runtime checks alone; investigate only any anomalies explicitly listed above rather than changing code based on outcome.
5. **Interaction phase:** not started; potentially worthwhile only after review of this audit, with the pre-specified exploratory scope above.
6. **Three interactions:** semantic anchor × TCPR, node adaptation × TCPR, and modality adaptation × TCPR.

Generated products: `f2_main_completeness.csv`, `f2_nc_full_results.csv`, `f2_lp_full_results.csv`, `f2_main_summary.csv`, `f2_seed_paired_drops.csv`, `f2_e0_alignment_summary.csv`, the paper-table candidates, and the diagnosis-only heatmap under `outputs/f2_ablation/`.
