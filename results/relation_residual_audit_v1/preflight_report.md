# P0-R — Preflight Audit

## Repository state

- Base branch: `relation_operator_diagnostics`
- Base commit: `f953b2b3a1c49aad49705a5998c83dc149a7dcbb`
- Base was clean and synchronized with `origin`; implementation is isolated on `relation_residual_audit`.
- Existing `relation_operator_audit_v1` and `relation_operator_diagnostics_v1` outputs are read-only for this phase.

## Existing model and NC protocol

- Effective physical operator: remove input self-loops, symmetrize with `to_undirected`, add one self-loop per node, coalesce coordinates, use unit weights, then apply symmetric normalization `P = D^-1/2 (A + I) D^-1/2`.
- Sparse coordinates use row=target and column=source. Thus `P @ C` aggregates source-node messages into target nodes.
- P0's modality projectors are independent Text and Visual `Linear → LayerNorm → ReLU → Dropout` blocks. Its fusion is `Linear(512→256) → ReLU → Dropout → Linear(256→256)`.
- Propagation uses `C0=H0`, three hops (`max_order=3`), and uniform readout `(C0+C1+C2+C3)/4`; output dimension is 256.
- The NC task performs one full-graph forward per epoch, computes cross-entropy only for Train nodes, evaluates Validation each epoch, and selects by Validation Accuracy while reporting Macro-F1. The formal launcher must pass `task.evaluate_test=false`; then Macro-F1 labels use Train+Validation only and the NC runner skips Test evaluation. The loader still materializes the dataset label tensor in host memory but the training/evaluation path does not index Test labels.
- Checkpoints contain `task`, `seed`, `selection`, `epoch`, validation-only `metrics`, `model_state`, `head_state`, and `data_info`. The existing analyzer strict-loads model and head states.

## Frozen empirical evidence and design mapping

- P0-D reports that removing A2's shared expert transformation reduces Validation Accuracy in 45/45 scopes and Macro-F1 in 43/45. Replacing its global route with a uniform mixture keeps every scope's Accuracy within 0.01 of normal (mean route ΔCE about 0.00135). This motivates a single shared semantic operator and removes the need for global expert routing.
- P0-D shrinkage found an intermediate `0 < λ < 1` with lower Validation CE in 41/45 A3 contexts. P0-R therefore retains a global shared transformation but limits edge specialization to low-rank residual adapters.
- Targeted effects were heterogeneous and stronger on directly touched nodes. Residual adaptation remains testable rather than being removed outright.
- P0-D structural Spearman summaries show a weak positive association of route deviation with `||D_i-D_j||` (mean rho about 0.104; positive in 27/30 modality-contexts) and a weak negative association with `cos(D_i,D_j)` (mean rho about -0.077; negative in 28/30). These descriptive results motivate the R2-vs-R1 comparison; they do not establish causal necessity.

## Implementation boundaries

- R0 consolidates to a modality-specific `256→256→256` shared bottleneck operator.
- R0G, R1, and R2 use identical trainable relation encoders, NULL-context vectors, three `256→32→256` residual adapters, and routers. R0G applies the computed mean edge route; R1 routes from endpoint attributes plus learned NULL context; R2 adds differentiable `D=P_nbr H0-H0` to the same context pathway.
- Self-loop messages remain the normalized identity path and are excluded from relation encoding, shared operators, and residual adapters. Relation features/routes are computed once from H0 (and R2 D) and reused for all three hops.
- No formal 60-run training, Test evaluation, auxiliary objective, topology modification, or changes to historical P0/P0-D result directories are part of this implementation pass.


## Smoke and implementation verification

- CUDA training smoke completed without formal runs: Movies seed 42 for all four variants at one epoch; Reddit-S seed 42 for R2 at one epoch. Peak allocation/runtime pairs (MiB / seconds): Movies R0 4,388 / 1.39, R0G 7,986 / 2.00, R1 7,986 / 1.91, R2 8,021 / 2.04; Reddit-S R2 13,633 / 2.45.
- R2 no-label forward completed with finite outputs for all five datasets. Maximum observed forward-only allocation was 2,233 MiB on ele-fashion (97,766 nodes; 399,172 non-self coordinates). `data_forward_smoke.csv` records all cases.
- Model trainable parameter counts are R0=855,040 and R0G=R1=R2=1,090,248 on Movies, Toys, Grocery, and Reddit-S; ele-fashion is R0=723,968 and R0G=R1=R2=959,176. Classification-head counts are recorded separately in `parameter_counts.csv`.
- Before the first optimizer step, mean no-adaptation probability across eight modality summaries from the four residual-model smoke contexts was 0.5026; post-smoke best-checkpoint mean was about 0.5035. These are engineering-only observations.
- The fused CUDA SDPA path failed on an initial 65,536-edge attention batch. The model now asks MultiheadAttention for its (discarded) attention weights to select the general attention path, and caps relation-encoder batches at 8,192 edges (`min(edge_chunk_size, 8192)`). The architecture/equations and persistent `[E,4]` route storage are unchanged. This is the sole execution-kernel deviation and is documented in `docs/relation_residual_audit_design.md`.
- Focused P0-R and historical P0/P0-D regression suites passed 55 tests. Formal dry-run generated 60 unique commands; no formal training or Test evaluation was performed.

- Smoke manifests now include `git_worktree_clean=false`; their commit field is the recorded base HEAD `f953b2b3a1c49aad49705a5998c83dc149a7dcbb`. The P0-R code was an uncommitted worktree during engineering smoke. Formal launches require a clean committed `relation_residual_audit` branch and record its final SHA.
