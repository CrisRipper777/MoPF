# P0-R — Shared-Prior Contextual Relation Adaptation

Analysis mode: `smoke`; loaded P0-R checkpoints: 5.
This analyzer uses only Train/Validation labels. Every resolved config and checkpoint is required to have `task.evaluate_test=false` and no Test metrics. No Test labels or metrics are used.
All checkpoint comparisons use strict state loading. Within-P0-R paired comparisons share dataset and seed. Previous P0 comparisons are labeled `cross-experiment paired descriptive comparison`; their CE values are unavailable from the historical validation table.
Structural context magnitude quartiles use the Train–Train distribution of `||D_i-D_j||`; semantic-similarity quartiles use frozen plain A0 H0 and the same Train–Train thresholds used by P0-D.

## Formal status

Formal scientific decisions remain PENDING until all 60 formal contexts are analyzed. Smoke metrics are engineering checks only.

## Main paired comparisons

- `R0G-R0` (1 contexts): mean ΔAccuracy=+0.0000; mean ΔMacro-F1=+0.0000; mean ΔCE=-0.0366.
- `R1-R0G` (1 contexts): mean ΔAccuracy=+0.0000; mean ΔMacro-F1=+0.0000; mean ΔCE=-0.0000.
- `R2-R1` (1 contexts): mean ΔAccuracy=+0.0000; mean ΔMacro-F1=+0.0000; mean ΔCE=-0.0000.
- `R2-R0` (1 contexts): mean ΔAccuracy=+0.0000; mean ΔMacro-F1=+0.0000; mean ΔCE=-0.0366.
- `R0-old_plain` (1 contexts): mean ΔAccuracy=-0.2442; mean ΔMacro-F1=-0.4570; CE not present in historical comparison table.
- `R0-old_global_expert` (1 contexts): mean ΔAccuracy=-0.2475; mean ΔMacro-F1=-0.4680; CE not present in historical comparison table.
- `R2-old_relation_expert` (2 contexts): mean ΔAccuracy=-0.5027; mean ΔMacro-F1=-0.6418; CE not present in historical comparison table.
- `R2-old_global_expert` (2 contexts): mean ΔAccuracy=-0.5078; mean ΔMacro-F1=-0.6549; CE not present in historical comparison table.

## Frozen interventions and mechanisms

- `shared_prior` zero-shared: mean ΔCE +0.0173; Validation Accuracy declined in 0/1 contexts.
- `global_residual` zero-shared: mean ΔCE +0.0178; Validation Accuracy declined in 0/1 contexts.
- `attribute_residual` zero-shared: mean ΔCE +0.0178; Validation Accuracy declined in 0/1 contexts.
- `context_residual` zero-shared: mean ΔCE +0.0124; Validation Accuracy declined in 0/2 contexts.
- Zero-residual: mean ΔCE +0.0010; Accuracy declined in 0/4 contexts.
- `context_null`: mean ΔCE +0.0000; Accuracy declined in 0/2 contexts.
- `context_shuffle`: mean ΔCE -0.0000; Accuracy declined in 0/2 contexts.
- `global_route`: mean ΔCE +0.0000; Accuracy declined in 0/3 contexts.
- Context-sensitivity summaries: 64 semantic × structural strata.

## R2 shrinkage

R2 intermediate λ beat λ=1 CE in 0/6 contexts; best CE λ was 1.0 in 1 contexts and 0.75 in 2. Historical A3 reported 41/45 contexts with an intermediate advantage.

## Parameter parity and mechanism interpretation

R0G/R1/R2 trainable model parameter parity holds in 6/6 paired dataset-seed records.
No-adaptation initialization check: mean π0 epoch1=0.5026; mean π0 post_smoke=0.5035. The first value is measured during the epoch-1 forward before its optimizer step; the post-smoke value uses the best Validation-Accuracy checkpoint.
The training_mechanism_statistics.csv.gz file preserves rr_ mechanism summaries from each logged training epoch; mechanism_statistics.csv reports frozen best-checkpoint evaluation. Both include per-modality/per-hop shared and residual magnitudes, no-adaptation route, adapter usage, routing entropy/top-1 fractions, and adapter-output cosine. Context sensitivity compares real D routes with learned-NULL routes while keeping checkpoint parameters fixed.
Acceptance summaries are written to `acceptance_decisions.csv`. The protocol gives no numeric cutoffs for qualitative terms such as 'markedly' or 'noticeably', so these rows remain descriptive and require scientific adjudication rather than an invented pass/fail threshold.
No P0-R formal training is started by the analyzer. No auxiliary objectives, structural supervision, or test-set selection are used.

## Engineering smoke evidence

No-label R2 forward: 5/5 dataset contexts had finite output; maximum measured allocation 2233.1 MiB.
One-epoch CUDA training smoke: 5 completed contexts; maximum runtime 2.45 s and peak allocation 13633.2 MiB across the recorded runs.
Smoke provenance: HEAD f953b2b3a1c49aad49705a5998c83dc149a7dcbb; git_worktree_clean=false.
CUDA attention execution note: an initial 65,536-edge fused SDPA batch failed with an invalid kernel configuration. Relation attention now requests (and discards) MHA attention weights to select the general MHA path, with an internal edge batch cap of min(edge_chunk_size, 8,192). This changes chunk execution, not model equations.
