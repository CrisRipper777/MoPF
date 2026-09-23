# Figure 3 final claims

## (a) MRC relation response

The current P2 checkpoint maps modality-dependent semantic discrepancy to continuous relation-weight discrepancy on the same physical support. This supports the claim that MRC is computationally responsive to modality differences. It is not an independent discovery of a new topology, a large rewiring claim, or relation reliability estimation.

## (b) Modality-specific neighbor allocation

Text and visual normalized neighbor-weight distributions have measurable but generally fine-grained differences. The top-neighbor disagreement is decision-relevant in the diagnostic, while the TV distance remains small. The correct claim is modality-specific allocation over a shared topology, not substantial topology difference or rewiring.

## (c) Adaptive context assignment

Gate distributions are non-degenerate, differ between modalities on several datasets, and globalized/shuffled frozen inference changes embeddings, logits, and predictions. This supports a functional node-to-gate correspondence in the trained P2 computation. It is not M1 lambda ground truth, a causal gate test, or evidence of universal performance improvement.

All interventions are frozen-checkpoint inference diagnostics. None should be described as retrained performance. Source CSVs are under `outputs/final/context_formation_analysis/` and `outputs/final/paper_figures/figure3_data/`.
