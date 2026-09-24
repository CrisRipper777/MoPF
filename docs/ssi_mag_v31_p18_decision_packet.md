# SSI-MAG-V3.1 P1.8 Decision Packet

Decision status: OPEN REVIEW — no final paper-model decision

## Scope

P1.8 trains exactly one independent direct-utilization R1U variant (U) under the frozen full-graph NC protocol. P1.7c B/AB are read-only controls.

## Evidence separation

- Performance evidence: outputs/ssi_mag_v31_p18_analysis/p18_performance.csv and paired rows in p18_r1_comparison.csv.
- Mechanism behavior evidence: R1 semantic scorer/weight/operator diagnostics, Stage-II alpha/attention/signed-filter diagnostics.
- Frozen functional sensitivity: relation-off and interaction-off with the same saved NC head; not retrained causal ablations.

## Predefined trigger

U-vs-B validation guardrail status: PERFORMANCE_REVIEW_TRIGGER; mean Accuracy delta -0.1875 pp; mean Macro-F1 delta -0.1887 pp; simultaneous declines ['Movies', 'Toys', 'Grocery', 'Reddit-S'].
This trigger only requests review and does not decide the final paper model.

## Boundary

No LP, no extra relation variants, no tuning, no auxiliary loss, no test-based selection, and no additional ablation training were performed.
