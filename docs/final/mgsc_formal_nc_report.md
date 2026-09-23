# MGSC-MAG Formal NC Qualification

## Scope and protocol

This qualification covers the five NC datasets `Movies`, `Toys`, `Grocery`,
`ele-fashion`, and `Reddit-S`, with seeds 42, 43, and 44. Sports-LP is
intentionally excluded from this round. P0 is the frozen CoSI-MAG reference;
P1 enables adaptive context-state formation with direct interacted-state
integration disabled; P2 enables the same gate and enables direct interacted-
state integration.

All three variants use the same unified NC runner, official split, optimizer,
early stopping, classifier, graph preprocessing, and seed protocol. The
frozen reference files and task protocol were not modified. Exact code/config
and split fingerprints are recorded in
`outputs/final/final_candidate_nc/qualification_metadata.json` and each run's
`qualification_metadata.json`.

## Test performance

Values are mean ± population standard deviation over the three seeds.

| Dataset | P0 accuracy | P1 accuracy | P2 accuracy | P0 Macro-F1 | P1 Macro-F1 | P2 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.5638 ± 0.0021 | 0.5633 ± 0.0013 | 0.5640 ± 0.0115 | 0.5000 ± 0.0053 | 0.4980 ± 0.0122 | 0.5064 ± 0.0104 |
| Toys | 0.8009 ± 0.0060 | 0.8015 ± 0.0064 | 0.7990 ± 0.0053 | 0.7745 ± 0.0073 | 0.7751 ± 0.0066 | 0.7736 ± 0.0057 |
| Grocery | 0.8304 ± 0.0038 | 0.8354 ± 0.0044 | 0.8323 ± 0.0053 | 0.7516 ± 0.0136 | 0.7637 ± 0.0032 | 0.7566 ± 0.0067 |
| ele-fashion | 0.8828 ± 0.0007 | 0.8843 ± 0.0011 | 0.8833 ± 0.0019 | 0.7793 ± 0.0030 | 0.7798 ± 0.0023 | 0.7789 ± 0.0045 |
| Reddit-S | 0.9666 ± 0.0014 | 0.9698 ± 0.0028 | 0.9692 ± 0.0025 | 0.9280 ± 0.0049 | 0.9350 ± 0.0033 | 0.9321 ± 0.0035 |

The machine-readable complete results are in
`outputs/final/final_candidate_nc/per_seed_results.csv` and
`outputs/final/final_candidate_nc/summary.csv`.

## Paired mean deltas

The following deltas are computed from paired seed runs and reported as
percentage-point units here.

| Dataset | P1−P0 accuracy | P2−P0 accuracy | P2−P1 accuracy | P1−P0 Macro-F1 | P2−P0 Macro-F1 | P2−P1 Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | -0.05 pp | +0.02 pp | +0.07 pp | -0.20 pp | +0.64 pp | +0.84 pp |
| Toys | +0.06 pp | -0.19 pp | -0.25 pp | +0.06 pp | -0.09 pp | -0.15 pp |
| Grocery | +0.51 pp | +0.20 pp | -0.31 pp | +1.21 pp | +0.50 pp | -0.71 pp |
| ele-fashion | +0.15 pp | +0.05 pp | -0.11 pp | +0.05 pp | -0.05 pp | -0.10 pp |
| Reddit-S | +0.33 pp | +0.26 pp | -0.06 pp | +0.70 pp | +0.41 pp | -0.29 pp |

The unrounded values are in
`outputs/final/final_candidate_nc/paired_deltas.csv`.

## P1 decision

**P1: PASS for continued study.**

- All 15 P1 runs completed without training failure or non-finite metrics.
- No paired P1 test-accuracy drop exceeded 1 percentage point; the worst
  seed-level drop was Movies seed 44 at -0.21 pp.
- The existing seed-42 gate analysis shows non-trivial node/order behavior,
  non-identical text/visual behavior, and non-saturated gate distributions.
- The inference-only audit in
  `docs/final/gate_mechanism_audit.md` shows that globalized, shuffled, and
  fixed-0.9 gates change representations and logits. The shuffled condition
  preserves gate marginals while changing node correspondence, so this is
  functional evidence rather than a claim of causal ground truth.

P1 is not interpreted as a universal performance improvement: its mean
accuracy change is positive on four datasets and slightly negative on Movies,
while the paired results vary by seed.

## P2 decision

P2 is **qualified as a viable candidate mechanism**, not as a universally
better model. Its mean test accuracy is within -0.19 pp of P0 on every
dataset, improves over P0 on Movies, Grocery, ele-fashion, and Reddit-S, and
is slightly lower on Toys. The direct-interaction audit from the current P2
checkpoint shows larger interaction-off embedding/logit changes and prediction
flip rates than the corresponding P1 audit on all five NC datasets.

This supports retaining P2 for the next research round while keeping claims
modest: the evidence establishes functional use of the adaptive states and
direct integration, not that P2 dominates P0 on every dataset or seed.

## Controlled cleanup diagnostic

The separate P2-clean control switches only
`use_legacy_relation_order_bias=false` for seed 42 on Movies, Grocery, and
ele-fashion. It satisfies the predeclared accuracy criterion (no drop over
0.5 pp): test-accuracy deltas are +0.42 pp, -0.32 pp, and +0.22 pp,
respectively. Grocery Macro-F1 drops by 1.06 pp, so the clean path remains a
diagnostic and is **not** made the default. Details are in
`docs/final/relation_cleanup_control.md` and
`outputs/final/relation_cleanup_control/`.

## Freeze recommendation and limitations

For the next round, freeze the current P2 candidate with
`use_legacy_relation_order_bias=true` as the working MGSC-MAG candidate, while
keeping `cosi_mag_final.py` and `cosi_mag_final.yaml` as the frozen reference.
Do not treat the node-wise oracle gap, preferred lambda, or gate alignment as
supervised labels or test-time mechanisms. Sports-LP was not run by scope
decision and remains unqualified in this report.
