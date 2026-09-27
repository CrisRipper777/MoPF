# P0 — Relation-Conditioned Semantic Operator Audit

Completed formal contexts: **0/60**. No formal runs are launched by this analyzer.

## Registered question and controls

On the same unit-weight, symmetrically normalized physical graph, does relation-conditioned semantic transformation improve NC over plain physical propagation or scalar semantic weighting? All variants use independent Text/Visual projectors, K=3, uniform four-state readout, and CMRF plain fusion.
Only Validation Accuracy selects checkpoints. Macro-F1 is reported. `task.evaluate_test=false`; the analyzer checks checkpoint and metrics payloads for test fields and reads only Train/Validation labels. The data loader reads the dataset's full CPU label tensor as required by its format, while the NC training path moves only Train targets to the device.

## Paired comparisons

| Contrast | Metric | Mean Δ | Population SD | Positive pairs | Positive dataset means |
|---|---|---:|---:|---:|---:|
| relation_expert_minus_global_expert | val_acc | — | — | 0/15 (0 complete) | 0/5 |
| relation_expert_minus_global_expert | val_macro_f1 | — | — | 0/15 (0 complete) | 0/5 |
| relation_expert_minus_scalar_weight | val_acc | — | — | 0/15 (0 complete) | 0/5 |
| relation_expert_minus_scalar_weight | val_macro_f1 | — | — | 0/15 (0 complete) | 0/5 |
| relation_expert_minus_plain | val_acc | — | — | 0/15 (0 complete) | 0/5 |
| relation_expert_minus_plain | val_macro_f1 | — | — | 0/15 (0 complete) | 0/5 |
| global_expert_minus_plain | val_acc | — | — | 0/15 (0 complete) | 0/5 |
| global_expert_minus_plain | val_macro_f1 | — | — | 0/15 (0 complete) | 0/5 |

## Descriptive gates

- **H1_relation_conditioning_beyond_capacity** — `PENDING`. A3-A2: >=3/5 positive dataset means; >=9/15 positive seed pairs; mean Accuracy >0; mean Macro-F1 >=0
- **H2_transformation_vs_scalar_weighting** — `PENDING`. A3-A1 aggregate Accuracy/F1 >=0; >=3/5 dataset Accuracy means >=0; review catastrophic transfer and mechanism evidence descriptively
- **H3_relation_conditioning_functionally_used** — `PENDING`. Validation-only routing shuffle, modality-global route, and zero residual; compare task metrics, flips, message change, and routing change
- **expert_collapse_diagnostic** — `PENDING`. Flag if any expert load >90%, pairwise expert-output cosine >0.98, or mean-expert intervention has negligible Validation effect across measured contexts

H2's phrase “catastrophic negative transfer” has no numeric cutoff in the frozen request. The report exposes the most negative per-dataset Accuracy delta for review and does not silently add a threshold. A3-vs-A1 cannot be promoted from metrics alone; correction/routing diagnostics must also be examined.

## Mechanism and expert diagnostics

Validation-only frozen interventions recorded: 0 rows. Targeted similarity-quartile edge interventions: not run (optional flag not supplied).
`relation_diagnostics.csv` uses cosine similarity in projected H0 space. Quartile boundaries are computed from physical edges whose endpoints are both Train nodes; reported edges are physical edges incident to at least one Validation node. Test labels are not used.
expert_specialization.csv contains mean load, routing entropy, top-1 fractions, and pairwise output cosine on the same Train-node inputs. Collapse diagnostic: PENDING. No balancing/diversity regularizer is added.

## Parameter counts

Parameter-count rows: 20. Counts distinguish the Text+Visual model, NC classifier head, and their sum. A2/A3 parity is guarded by a unit test.

## Run integrity

Every completed run is checked for a resolved config with test evaluation disabled and for absence of test metrics in the checkpoint and run metrics. The formal launcher writes the exact command, git SHA, deterministic run/checkpoint paths, runtime, peak GPU allocation, and failure reason to its manifest.
