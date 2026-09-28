# CRSA+RSE+RCFA v2 Preflight Report

Audit device: cuda:1.

legacy_crsa_parity_max_abs_error: 0.0000000000e+00
rse_decomposition_max_abs_error: 2.9907096177e-07
identity_init_max_abs_error: 0.0000000000e+00
initial_gate_deviation_from_one: 0.0000000000e+00
modality_isolation_max_abs_error: 0.0000000000e+00
forward_finite: true
backward_finite: true
smoke_contexts_complete: 5/5
smoke_peak_gpu_memory_mib: 11887.191
smoke_checkpoint_strict_load: true
test_accessed: false
lp_run: false
other_modality_inside_rcfa_gate: false
initial_smoke_oom_observed_and_resolved: true

The first full-graph ele-fashion smoke exceeded available memory before backward completed. Exact per-edge-chunk checkpointing was added to CRSA message/RSE computation; the retried smoke completed with the same CRSA/RSE/RCFA equations and full graph. The final smoke manifest records each successful context and peak allocated GPU memory.
