# CRSA+RSE+RCFA v2 NC Validation Report

## 1. Implementation provenance

- Source branch: crsa_iatr_v1 at initial snapshot e26d20ebf2d957bba790e9cbdb8b0c374cbe60b3.
- Experiment branch: crsa_rcfa_v2.
- Formal source commit: fe2daa7cd299c23fd9c49dc8bc0eae9efef59a94.
- Formal contexts complete: 15 / 15.
- Every formal run is required to record a clean worktree and the same source commit.

## 2. Preflight correctness audit

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

## 3. Full-v2 validation results

| Dataset | n | Validation Accuracy (%) | Validation Macro-F1 (%) |
|---|---:|---:|---:|
| Movies | 3 | 56.27 ± 0.17 | 46.46 ± 1.21 |
| Toys | 3 | 80.14 ± 0.13 | 77.37 ± 0.55 |
| Grocery | 3 | 83.18 ± 0.28 | 74.50 ± 1.31 |
| ele-fashion | 3 | 88.01 ± 0.28 | 75.93 ± 0.59 |
| Reddit-S | 3 | 96.55 ± 0.51 | 92.51 ± 0.35 |

## 4. Paired comparison with CRSA-v1

Pairing audit: legacy CRSA parity error 0.000e+00 passed; 15 validation-only CRSA rows verified at the expected source commit. Values are descriptive validation differences; no significance or pass/fail label is assigned.

| Dataset | Metric | Seed 42 | Seed 43 | Seed 44 | Mean delta |
|---|---|---:|---:|---:|---:|
| Movies | Accuracy (pp) | -0.84 | -0.90 | -1.56 | -1.10 |
| Movies | Macro-F1 (pp) | -2.58 | -1.82 | -2.80 | -2.40 |
| Toys | Accuracy (pp) | -0.24 | -0.48 | -0.48 | -0.40 |
| Toys | Macro-F1 (pp) | -0.67 | -0.29 | -0.93 | -0.63 |
| Grocery | Accuracy (pp) | -0.56 | -0.61 | +0.09 | -0.36 |
| Grocery | Macro-F1 (pp) | -1.28 | -2.45 | +1.38 | -0.78 |
| ele-fashion | Accuracy (pp) | +0.61 | -0.10 | +0.16 | +0.23 |
| ele-fashion | Macro-F1 (pp) | +1.42 | -0.59 | -0.66 | +0.06 |
| Reddit-S | Accuracy (pp) | +0.22 | +0.88 | +0.85 | +0.65 |
| Reddit-S | Macro-F1 (pp) | +0.22 | +1.53 | +0.57 | +0.77 |

## 5. Dataset-wise observations

- Validation-accuracy paired mean direction: negative: Movies, Toys, Grocery; positive: ele-fashion, Reddit-S.
- Interpret dataset variation descriptively; three seeds do not support a significance claim.

## 6. Runtime and GPU memory

| Dataset | Mean runtime (s) | Peak GPU memory (MiB) |
|---|---:|---:|
| Movies | 140.9 | 4328.4 |
| Toys | 110.2 | 3587.3 |
| Grocery | 110.9 | 4078.0 |
| ele-fashion | 445.6 | 11900.0 |
| Reddit-S | 130.7 | 6496.2 |

## 7. Engineering pathology audit

- Complete contexts: 15 / 15; non-complete or invalid contexts: 0.
- OOM signatures in run logs: 0.
- NaN/Inf signatures in run logs: 0.
- Artifact audit checks finite losses, finite checkpoint tensors, validation-only metric keys, and strict model/head checkpoint loading.

## 8. Interpretation

Full-v2 has a mixed paired mean validation-accuracy pattern: positive on 2 datasets and negative on 3 datasets. This is descriptive, not evidence of statistical significance.
A formal w/o-relation-condition ablation is worth running as a targeted attribution follow-up: it directly tests whether RSE input explains the mixed dataset pattern. This recommendation is for mechanism isolation, not an assumption of improved mean performance; no ablation is started here.

Training uses Train labels and validation selection/reporting only. The resolved configuration sets evaluate_test=false and the analyzer rejects Test metric keys. No link prediction context is in the matrix.
