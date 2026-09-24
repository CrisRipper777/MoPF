# P3 final-S experiment specification

## Scope and freeze

P3 is a code-preparation and provenance-planning stage. It does not start an NC or LP training job. The final architecture remains the frozen implementation in src/models/ssi_mag_scd.py; its P2.1 provenance hash is:

5b00505db5c504e9e6c57006673bd84c18e71504fbe13f97e0fe12df41c49132

The P3 launchers refuse formal execution if this source hash changes. They also require branch V3, a clean worktree, and HEAD == origin/V3 before creating an immutable formal lock.

P2.1/P2.1a Full-S NC outputs are reused from outputs/ssi_mag_scd_p21_nc/; they are not rerun.

## S-specific ablations

The new ssi_mag_scd_ablation model uses the same retained S graph for NC and LP. It adds only the following explicit, paper-facing branches:

| Variant | Definition | Claim tested |
| --- | --- | --- |
| no_semantic_conductance | Unit relation weights on the raw physical support, c=0, and zero relation response residual; semantic states and the remaining S path are retained. | Whether learned semantic conductance and relation formation are needed. |
| fixed_restart | For every modality/node/hop, alpha=0.10; Q=P S_{k-1}, S_k=.90Q+.10H_0; the reference bank is the actual xi=[1,.1,.1,.1]. | Whether adaptive restart adds evidence beyond a fixed restart. |
| terminal_context_only | Form the complete S_0...S_K bank, set earlier response coefficients to zero, set terminal reference residual to zero, and use Z=eta_K S_K with terminal global, content, and relation terms. This is a composite baseline, not a causal single-variable intervention. | Whether full retained multi-context utilization matters beyond a terminal representation. |
| global_response_only | Retain the complete state bank but use eta=gamma+DeltaGamma; content, reference, and relation response residuals are zero. | Whether a global order response explains the formation-conditioned response. |
| no_formation_conditioning | Retain state formation and use eta=gamma+DeltaGamma+content; reference and relation response residuals are zero. | Whether reference/relation formation conditioning contributes beyond content response. |

Historical P1/P2 ablation names and their behavior are unchanged.

## Formal matrix

NC uses Movies, Toys, Grocery, ele-fashion, and Reddit-S with seeds 42/43/44.

- Existing Full-S reference: 15 reused jobs.
- S-specific ablations: 5 variants x 5 datasets x 3 seeds = 75 jobs.
- Generic controls: PPR-style and GPR-style x 5 datasets x 3 seeds = 30 jobs.

LP uses the current sampled pipeline, sports-copurchase and books-lp, with seeds 42/43/44.

- Main comparison: Full-S, generic PPR-style, and generic GPR-style x 2 datasets x 3 seeds = 18 jobs.
- LP ablations: the five S-specific variants on sports-copurchase x 3 seeds = 15 jobs.
- No LP ablations are planned for books-lp.

The LP sampler remains num_neighbors=[5,5,5], with positive message-edge masking, filtered negatives, validation-MRR checkpoint selection, and test metrics reported only after selection. No test metric is used for checkpoint selection or a decision gate.

## Required future artifacts

Each completed formal run must contain best.pt, metrics.json, results.json, resolved_config.json, and complete.marker. NC completion requires finite val_acc, val_macro_f1, test_acc, and test_macro_f1. LP completion requires finite val_mrr, test_mrr, test_hits@1, test_hits@3, and test_hits@10.

Each suite root records model-source/config/task/data-config hashes, dataset/seed/variant coverage, the protocol, LP neighbors, selection metric, and test_used_for_selection=false. Dry-runs write only provenance.plan.json; they never create a formal lock or invoke src.main.

## Decision boundary

P3 does not automatically enter P2.2. The eventual response decision is limited to FREEZE_S or P22_RESPONSE_PILOT_TRIGGERED, using the P2.1a performance gate and the formation-conditioned mechanism report together. A single small metric or isolated run is not sufficient to trigger P2.2.
