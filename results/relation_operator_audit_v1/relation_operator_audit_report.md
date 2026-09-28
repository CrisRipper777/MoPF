# P0 — Relation-Conditioned Semantic Operator Audit

Completed formal contexts: **60/60**. No formal runs are launched by this analyzer.

## Registered question and controls

On the same unit-weight, symmetrically normalized physical graph, does relation-conditioned semantic transformation improve NC over plain physical propagation or scalar semantic weighting? All variants use independent Text/Visual projectors, K=3, uniform four-state readout, and CMRF plain fusion.
Only Validation Accuracy selects checkpoints. Macro-F1 is reported. `task.evaluate_test=false`; the analyzer checks checkpoint and metrics payloads for test fields and reads only Train/Validation labels. The data loader reads the dataset's full CPU label tensor as required by its format, while the NC training path moves only Train targets to the device.

## Paired comparisons

| Contrast | Metric | Mean Δ | Population SD | Positive pairs | Positive dataset means |
|---|---|---:|---:|---:|---:|
| relation_expert_minus_global_expert | val_acc | -0.004066 | 0.005047 | 3/15 (15 complete) | 1/5 |
| relation_expert_minus_global_expert | val_macro_f1 | -0.009965 | 0.014930 | 4/15 (15 complete) | 2/5 |
| relation_expert_minus_scalar_weight | val_acc | -0.003384 | 0.003537 | 3/15 (15 complete) | 1/5 |
| relation_expert_minus_scalar_weight | val_macro_f1 | -0.012149 | 0.015337 | 4/15 (15 complete) | 0/5 |
| relation_expert_minus_plain | val_acc | -0.001493 | 0.003933 | 7/15 (15 complete) | 2/5 |
| relation_expert_minus_plain | val_macro_f1 | -0.002994 | 0.015738 | 8/15 (15 complete) | 2/5 |
| global_expert_minus_plain | val_acc | 0.002574 | 0.003305 | 11/15 (15 complete) | 4/5 |
| global_expert_minus_plain | val_macro_f1 | 0.006971 | 0.012210 | 10/15 (15 complete) | 4/5 |

## Descriptive gates

- **H1_relation_conditioning_beyond_capacity** — `GATE_NOT_MET`. A3-A2: >=3/5 positive dataset means; >=9/15 positive seed pairs; mean Accuracy >0; mean Macro-F1 >=0
- **H2_transformation_vs_scalar_weighting** — `GATE_NOT_MET`. A3-A1 aggregate Accuracy/F1 >=0; >=3/5 dataset Accuracy means >=0; review catastrophic transfer and mechanism evidence descriptively
- **H3_relation_conditioning_functionally_used** — `FUNCTIONAL_EFFECT_OBSERVED`. Validation-only routing shuffle, modality-global route, and zero residual; compare task metrics, flips, message change, and routing change
- **expert_collapse_diagnostic** — `EXPERT_COLLAPSE_OBSERVED`. Flag if any expert load >90%, pairwise expert-output cosine >0.98, or mean-expert intervention has negligible Validation effect across measured contexts

H2's phrase “catastrophic negative transfer” has no numeric cutoff in the frozen request. The report exposes the most negative per-dataset Accuracy delta for review and does not silently add a threshold. A3-vs-A1 cannot be promoted from metrics alone; correction/routing diagnostics must also be examined.

## Mechanism and expert diagnostics

Validation-only frozen interventions recorded: 315 rows. Targeted similarity-quartile edge interventions: included.
`relation_diagnostics.csv` uses cosine similarity in projected H0 space. For each dataset/seed/modality, quartiles and edge assignments are frozen from the plain A0 best checkpoint; Q25/Q50/Q75 use physical Train–Train non-self edges, and A1/A3 are measured on the identical A0-defined physical Validation-incident edge groups. `similarity_reference=plain_A0_H0`. Test labels are not used.
A3 residual dominance is summarized by fractions with |Δm|/(|h|+ε)>1, >2, and cos(h,h+Δm)<0 for each quartile and hop.
expert_specialization.csv contains mean load, routing entropy, top-1 fractions, and pairwise output cosine on Train-node inputs at hop 1 (H0), hop 2 (C1), and hop 3 (C2). Collapse diagnostic: EXPERT_COLLAPSE_OBSERVED. No balancing/diversity regularizer is added.
Collapse cosine rows by exact context and hop: global_expert/Movies/seed42/text/hop1:0.987655; global_expert/Movies/seed42/text/hop1:0.989385; global_expert/Movies/seed42/text/hop2:0.988788; global_expert/Movies/seed42/text/hop2:0.990648; global_expert/Movies/seed42/text/hop3:0.984191; relation_expert/Movies/seed42/text/hop1:0.985937; relation_expert/Movies/seed42/text/hop1:0.992852; relation_expert/Movies/seed42/text/hop2:0.985584; relation_expert/Movies/seed42/text/hop2:0.994596; relation_expert/Movies/seed42/text/hop3:0.993389; global_expert/Movies/seed43/text/hop2:0.981323; global_expert/Movies/seed43/text/hop2:0.981733; relation_expert/Movies/seed43/text/hop1:0.981984; relation_expert/Movies/seed43/text/hop2:0.982875; global_expert/Movies/seed44/text/hop1:0.984415; global_expert/Movies/seed44/text/hop1:0.981619; global_expert/Movies/seed44/text/hop1:0.981173; global_expert/Movies/seed44/text/hop1:0.984572; global_expert/Movies/seed44/text/hop2:0.981488; global_expert/Movies/seed44/text/hop2:0.982684; global_expert/Movies/seed44/text/hop2:0.986264

## Parameter counts

Parameter-count rows: 20. Counts distinguish the Text+Visual model, NC classifier head, and their sum. A2/A3 parity is guarded by a unit test.

## Run integrity

Every completed run is checked for a resolved config with test evaluation disabled and for absence of test metrics in the checkpoint and run metrics. The formal launcher requires a clean worktree and writes branch, git SHA, exact command, deterministic run/checkpoint paths, runtime, peak GPU allocation, and failure reason to its manifest.
