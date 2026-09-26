# Conditional Relational Utility Discovery: preflight

- Branch: `cmrf_discovery`
- Starting HEAD: `e0f93dea44f8081f14ddf6ce2a232ba843338216` (user-authorized current base)
- Scope: NC only; datasets Movies, Toys, Grocery, ele-fashion, Reddit-S; training seeds 42, 43, 44
- Protocol: `unified_full_graph_nc_v1`; Train/Validation only; `task.evaluate_test=false`
- Planned formal training: C0 = 15 runs; C1/C2/C3 = 45 new runs; total = 60.
- LP is excluded. Test labels/metrics are not used for model selection or analysis.
- Physical operator follows baseline `multi_order_bank`: remove loops, symmetrize, add one self-loop per node, coalesce, unit weights, symmetric normalization. Text/Visual share the same operator.
- Forbidden semantic pruning/reweighting, graph rewiring, old MoPF mechanisms, alignment losses, cross-attention, hard experts and hard selectors are absent.
- C0 uniform response identity max absolute error: float32 `2.161e-07`; CPU float64 `4.441e-16`.
- Adaptive initialization max output difference versus C0 after copying shared weights: {"own_soft": 0.0, "cross_soft": 0.0, "gated_cross_soft": 0.0} (each required to be <1e-6).
- C0 best-validation checkpoints: `outputs/cmrf_discovery_v1/reference_uniform/checkpoints/`; reconstructed states use the saved checkpoint and analyze() API.
- Output roots: `outputs/cmrf_discovery_v1/` and `results/cmrf_discovery_v1/`; no existing `u*` or full benchmark outputs are touched.
- The NC runner's validation label universe is restricted to Train+Validation whenever test evaluation is disabled.
- Focused CMRF tests: 8 passed. Full repository tests: 237 passed.

The identity audited is
`(S0+S1+S2+S3)/4 = S0 + 0.75 R1 + 0.50 R2 + 0.25 R3`,
where `Rk=Sk-S(k-1)`. CMRF fixes the H0 coefficient to 1 and bounds adaptive
response adjustments using lambda=0.25. All controller output heads are zero
initialized; gated cross residuals are zero at initialization.
