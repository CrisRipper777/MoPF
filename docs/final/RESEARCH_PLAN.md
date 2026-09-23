# MGSC-MAG Research Plan

## Frozen reference

- `src/models/cosi_mag_final.py`
- `configs/model/cosi_mag_final.yaml`

These files are frozen for this research round and must not be modified.

## Current hypothesis

Multigranular structure-semantic context modeling: in a multimodal attributed
graph, physical topology defines latent structural dependence, but the semantic
utility of structural context is not uniform across relation, node/context, or
propagation-order granularities.

The core representation under study is a modality-specific multi-order context
state bank, with a node-local modality state at order 0 and
structure-semantic context states at higher structural radii.

## Current empirical question

Do nodes and modalities exhibit heterogeneous intrinsic-context preferences,
measured by the preferred mixture between node-local modality semantics and
physical-neighborhood context?

## Candidate mechanism

Adaptive context-state formation: learn a modality-specific, node- and
propagation-order-dependent gate between the intrinsic state and the propagated
context state.

## Candidate second upgrade

Direct interacted-state integration: use cross-order interacted states directly
when composing the final modality representation, while holding the preference,
fusion, classifier, and MRC protocols fixed.

## Experimental sequence and decision status

- **M1 — Contextualization-demand empirical study:** first priority; NC only on
  Movies, Grocery, and ele-fashion.
- **Gate A:** decide whether heterogeneous contextualization demand is supported.
- **P1 — Adaptive context-state formation:** implement only if Gate A is PASS or
  WEAK-BUT-PROMISING under the predeclared engineering criteria.
- **P2 — Direct interacted-state integration:** test only if P1 passes.
- **Frozen:** the current strongest reference remains frozen throughout.

Current stage status: **M1 complete; P1 and P2 formally qualified on five NC
datasets; frozen reference remains the reference model**.

Decision record: M1 Gate A **PASS**; P1 **PASS** on the five-NC formal matrix;
P2 is retained as a viable candidate based on the five-NC formal matrix and
inference interventions, without claiming universal improvement. Sports-LP was
excluded from this round by scope decision. The controlled P2-clean relation
bias diagnostic is viable by accuracy but is not the default because Grocery
Macro-F1 drops.

## Integrity constraints

- Text and visual modalities propagate independently until final late fusion.
- No early fusion, semantic-neighbor augmentation, labels in M1 probe features,
  MRC, or test-label-based mechanism selection.
- The unified NC/LP task protocols and the frozen reference are not changed.
