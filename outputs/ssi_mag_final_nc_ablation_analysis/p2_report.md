# P2 Final NC Ablation Report

The canonical architecture remains frozen as P1.8 U. This report describes the pre-registered Full plus seven NC variants; it does not reopen architecture search.

## Integrity

- Formal jobs: 120/120; finite runs: 120/120; errors: []
- Protocol: `unified_full_graph_nc_v1`; LP jobs: 0; test used for selection: `False`.
- Provenance commit: `c3aa85a8221794af0b9f8830d1f910c6e0fdd482`.

## Performance

`p2_table_validation.csv` is the development-facing validation table. `p2_table_test.csv` is descriptive paper output evaluated only at validation-selected checkpoints.
Means and population standard deviations are across three seeds per dataset. The all-dataset summaries are unweighted means of five dataset means; no pooled node/dataset uncertainty is used.

## Paired interpretation

`p2_paired_deltas.csv` reports same-seed ablation minus Full deltas, including 15-run better/worse/tie counts. No p-values or significance claims are made.
Each variant is interpreted only against its pre-registered question. A positive ablation delta is reported descriptively and does not trigger redesign or tuning.

## Full versus historical P1.8 U

This is an equivalence/development sanity comparison only. Historical U is not a P2 selection input.

## Boundary

No LP, hyperparameter search, auxiliary loss, test-based tuning, seed deletion, post-hoc ablation, or architecture redesign was performed.

