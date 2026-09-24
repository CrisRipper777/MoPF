# SSI-MAG-V3.1 P1.7b Decision Packet

Decision status: **PASS_GUARDRAIL**

This packet separates performance evidence, mechanism behavior evidence, and frozen functional sensitivity. It does not choose a final paper model.

## 1. Performance evidence

- Formal Full NC runs: 15/15 loaded; finite: 15.
- Checkpoint selection is validation Accuracy; test metrics are final descriptive outputs only.

## 2. Seed stability

- Cross-seed ordering failures: `[]`.
- See `p17b_performance_summary.csv` for population standard deviations.

## 3–5. Mechanism behavior

- R1/R2/Stage-II quantities are reported in the diagnostic CSVs, including scorer dynamic-logit attribution and attention simplex validation.
- Flags are descriptive review triggers, not automatic mechanism failures.

## 6. Frozen functional sensitivity

- `relation=off` and `interaction=off` use unchanged model parameters and the same saved NC head; these are sensitivity diagnostics, not retrained causal ablations.

## 7. V3 vs V3.1 comparison

- Paired performance provenance: `verified: matching datasets/seeds, NC, full-graph, unified_full_graph_nc_v1, validation-Accuracy checkpoint selection`.
- Test deltas are descriptive and were not used for guardrail decisions.

## 8. Guardrail status

- `PASS_GUARDRAIL`.
- Validation Accuracy mean delta: `0.043459335962932055 pp`; Validation Macro-F1 mean delta: `0.060906515532251904 pp`.
- Simultaneous validation declines: `['Toys', 'Grocery']`.

## 9. Open questions

- Any saturation, collapse, or negligible-effect flag requires human audit and is not repaired here.
- No LP, retrained ablation, hyperparameter search, auxiliary loss, or test-based architecture selection was run.
