# MGSC-MAG story decision

## Decision

Freeze `mgsc_mag_p2` as the only working architecture for the current paper
scope: five NC datasets (Movies, Toys, Grocery, ele-fashion, Reddit-S).
Sports-LP and all LP claims are excluded from this round.

## Evidence status

1. M1 passed Gate A on Movies, Grocery, and ele-fashion. It supports
   heterogeneous intrinsic-versus-neighborhood contextualization demand, but
   its node-wise oracle is descriptive only.
2. Corrected controls completed 5 datasets × 3 seeds × 4 controls, with 60
   metrics, 20 summary rows, and 45 paired Full-minus-control rows.
3. Full is better than the strict plain backbone in 14/15 paired comparisons
   for both accuracy and Macro-F1, but the mean advantages are only 0.291 pp
   and 0.334 pp.
4. Raw terminal and raw bank readouts are close and sometimes better, so no
   universal necessity claim is supported.
5. Context and interaction interventions change embeddings/logits, confirming
   that the mechanism chain is computationally active.

## Story tier

The project is **Tier B**: a mechanism-supported unified framework with
non-dominant performance gains. The paper can argue for explicit modeling of
multi-granular structure-semantic context, not for a claim that each module is
individually indispensable.

## Canonical data flow

Independent text/visual projection → modality-specific MRC relation response
→ adaptive node/order context states → modality-specific state bank →
cross-order interaction → eta-weighted direct interacted-state composition →
independent refinement → late fusion → NC classifier.

No early fusion, topology rewiring, semantic-neighbor augmentation, new loss,
MoE, or additional architecture module is part of the frozen working story.

## Prior initialization

The current seed-42 prior-init control does not justify changing the legacy
anchored initialization. Direct-minus-legacy accuracy is −0.120, −0.029, and
−0.048 percentage points on Movies, Grocery, and ele-fashion; Macro-F1 is
−1.535, −0.301, and +0.477 points. Keep the canonical legacy mode and report
the direct mode as a controlled diagnostic only.

## Remaining risks

- The mechanism diagnostics are inference interventions, not causal tests.
- The prior-init comparison has one seed per mode/dataset.
- The ablation deltas are mixed and small; they should be shown, not ranked or
  selectively summarized.
- Figure 4 shows concentrated order profiles, not strong universal node-level
  heterogeneity.

There is no architecture blocker for freezing P2 within the declared five-NC
scope.
