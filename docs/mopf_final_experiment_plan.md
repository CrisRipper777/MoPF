# MoPF Final Experiment Plan

Status: **FROZEN PLAN — F1/F2 not started**

This is the execution plan after the final method and evaluation protocol
freeze. It is intentionally a plan only; no experiment in this document may
modify the frozen architecture or select a new model.

## F1 — Formal benchmark completion

Run the final MoPF and fair baseline comparison under the protocol in
`docs/mopf_final_evaluation_protocol.md`.

### Scope

- NC: Movies, Toys, Grocery, ele-fashion, Reddit-S, with fixed
  `K=3,3,2,3,3`, respectively.
- LP: the current formal sports-copurchase benchmark under
  `unified_sampled_lp_v1`.
- Seeds: 42, 43, and 44 for every core model and dataset.
- Baselines: all already implemented baselines that satisfy the same data,
  split, seed, optimization, and evaluator contract.
- Any previously unselected dataset/task is labeled quasi-held-out and is not
  mixed into the formal aggregate.

### Required artifacts

For each job, retain the resolved configuration, exact seed, checkpoint path,
validation selection metric, per-seed validation/test metrics, and evaluator
version/provenance. Aggregate only after all core model jobs for the relevant
dataset/task are complete. Test metrics are descriptive and cannot alter the
frozen model.

## F2 — Frozen ablation and mechanism package

Run only the following pre-locked comparisons against Full MoPF. All other
ablations are out of scope.

| Variant | Exact change from Full MoPF |
|---|---|
| Full MoPF | Final U1 + anchored cumulative U2-C1 + TCPR U3-B1 |
| w/o U1 semantic transport calibration | Replace `learned_diag_cos` with the pre-U1 `separate_cos` conductance path; downstream settings unchanged |
| w/o U2 semantic anchor | Use ordinary state propagation with cumulative responses; U1/U3 and all downstream settings unchanged |
| w/o U3 TCPR | `use_transport_residual=false`; U1/U2 and hierarchy unchanged |
| w/o node personalization | `use_node_residual=false`; all other final settings unchanged |
| w/o modality residual | `use_modality_residual=false`; all other final settings unchanged |

The minimal interaction set is fixed to the three mechanism-driven pairs:

1. w/o U2 semantic anchor + w/o U3 TCPR;
2. w/o node personalization + w/o U3 TCPR;
3. w/o modality residual + w/o U3 TCPR.

F2 uses the same formal datasets/tasks, seeds, splits, checkpoint-selection
rule, and descriptive test policy as F1. It does not introduce a new loss,
router, attention, MoE, projection, fusion, classifier, evaluator, or LP
decoder.

## Frozen figure plan

Only four mechanism figures are authorized:

- **Figure A:** modality conductance distributions/contrast on the fixed
  physical support;
- **Figure B:** anchored-state semantic retention and response contribution;
- **Figure C:** conductance versus response order;
- **Figure D:** TCPR beta profiles and the TransportShuffle diagnostic.

No additional mechanism figure or post-hoc explanatory ablation may be added
without reopening the frozen protocol, which is not part of F1/F2.

## Execution boundary

F0 stops after documentation, provenance, compatibility, and test audits.
F1 and F2 remain pending and must start only from the frozen configuration and
this plan. After F0, architecture design is closed.
