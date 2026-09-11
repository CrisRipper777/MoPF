# MoPF Final Evaluation Protocol

Status: **FINAL / pre-registered after architecture freeze**

This protocol applies the frozen model in
`docs/mopf_vnext_final_method_freeze.md`. It does not authorize model
selection, architecture changes, hyperparameter search, or post-hoc tuning.

## Frozen common settings

- Seeds are `42, 43, 44` for every core model and dataset.
- The same repository splits, evaluator, label-space construction, and
  checkpoint-selection rule are used for MoPF and every fairly runnable
  baseline.
- NC checkpoint selection is by validation accuracy. Validation Macro-F1 is
  reported as a secondary validation statistic; test metrics are descriptive
  and are never used for selection.
- The formal NC training protocol remains hidden dimension 256, dropout 0.2,
  AdamW with learning rate `1e-3`, weight decay `1e-4`, 300 maximum epochs,
  validation every epoch, patience 30, gradient clipping 1.0, and the existing
  scheduler/evaluator implementation.
- The formal LP protocol remains the existing `unified_sampled_lp_v1`
  protocol: sampled link prediction, the repository's fixed splits and
  filtered negatives, the existing neighbor-sampling settings, and full-graph
  exact inference at evaluation. The LP decoder and evaluator are unchanged.

## Formal order policy

| Dataset | Formal `K` |
|---|---:|
| Movies | 3 |
| Toys | 3 |
| Grocery | 2 |
| ele-fashion | 3 |
| Reddit-S | 3 |

Any newly added dataset or task uses `K=3` unless its already-established
baseline protocol explicitly specifies another value. This decision is made
before inspecting results; per-dataset K selection is prohibited.

## Formal benchmark scope

The formal NC benchmark is exactly Movies, Toys, Grocery, ele-fashion, and
Reddit-S. The current formal LP benchmark is sports-copurchase under the
repository's unified sampled LP protocol. Every formal comparison includes
MoPF and the already implemented baselines that can be run under the same
splits, seeds, input features, optimization budget, and evaluator.

Datasets or tasks not selected by the U1–U3 protocol are marked
**quasi-held-out** in tables, logs, and figures. For example, repository LP
presets such as books-lp or cloth-copurchase remain quasi-held-out unless a
separate protocol explicitly promotes them before result inspection.

## No-selection and no-tuning rules

The following are frozen and cannot be changed in response to any validation
or test result: U1 conductance mode and temperature, U2 state/response mode
and alpha, U3 TCPR, formal K, hidden size, fusion, loss, optimizer budget,
filter rank, evaluator, data splits, negative sampling, and seed set.

No temperature, alpha, TCPR profile, filter rank, hidden size, fusion choice,
loss, or per-dataset setting may be tuned from the final evaluation results.
If an additional seed is required, it is added to every core model and every
formal dataset, never only to a favorable model.

## Reporting rules

Report per-seed results, mean, and population standard deviation for validation
and test metrics. Mark test values as descriptive. Keep quasi-held-out results
separate from formal results. Mechanism and counterfactual diagnostics are
supporting descriptive evidence, not a new selection gate. Any implementation
failure is reported and repaired by rerunning the same pre-registered job;
the protocol itself is not revised to accommodate a result.

Checkpoint compatibility must be checked before evaluation: U1 freeze, U2-C1,
and U3-B1 checkpoints must load strictly, and final MoPF evaluation must use
the U3-B1-compatible formal configuration.
