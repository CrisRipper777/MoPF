# MoPF M1-D — PDC Functional Verification

This audit reuses the M1-C `v1_pdc` best-Validation-Accuracy NC checkpoints. No NC training or checkpoint mutation was performed.

- Checkpoints audited: 15 (5 datasets × 3 seeds)
- Device: cuda:1
- PDC follow-up decision: **continue**
- Formal-forward re-evaluation max absolute difference: `4.46e-6` (CUDA sparse-reduction repeat-rounding; the all-ones helper routes directly to learned rho)
- Eta-change vs delta-change max absolute error: `5.96e-8`

## Frozen task-stream evidence

| Dataset | fused relative L2 | all-node logit relative L2 | alignment gain Val Macro-F1 | alignment gain Test Macro-F1 |
|---|---:|---:|---:|---:|
| Movies | 0.000463989 | 0.000383522 | -0.000145378 | 9.93949e-05 |
| Toys | 8.18713e-06 | 5.50656e-06 | 0 | 0 |
| Grocery | 0.000205981 | 0.000171005 | 9.24805e-07 | 0.000101623 |
| ele-fashion | 0.000268264 | 0.000196518 | -2.05336e-05 | -0.000119244 |
| Reddit-S | 1.85126e-05 | 1.46658e-05 | 0 | 9.02969e-06 |

## Interpretation

`PDC-On - intervention` is used for downstream deltas. Positive values mean that closing the PDC path lowers the metric. Node-shuffle alignment gain is `shuffle_drop_on - shuffle_drop_off`; positive values mean that the learned PDC path makes node/profile alignment more task-functional.

The authoritative machine-readable artifact is `m1d_master_summary.json`; each run also contains path-utilization, stream-change, and alignment JSON files.
