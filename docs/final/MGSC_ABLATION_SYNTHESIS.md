# MGSC-MAG ablation synthesis

## Scope

This synthesis combines the existing fine-grained functional matrix
(`outputs/final/mgsc_functional_ablation/`) with the corrected raw-state and
plain-backbone controls (`outputs/final/mgsc_corrected_controls/`). The
fine-grained A1-A6 matrix was not rerun in this phase. All conclusions are
restricted to Movies, Toys, Grocery, ele-fashion, and Reddit-S NC; LP is not
part of the current paper scope.

## Evidence from the two control families

The earlier fine-grained matrix showed small, mixed Full-minus-ablation
changes for MRC, globalized context gates, terminal-state readout, uniform
integration, and no interaction. Attribute Only was lower on all five
datasets, but it is a sanity control rather than a pure single-module
ablation. Those results do not support a module ranking.

The stricter controls provide the cleaner architectural comparison:

| Comparison against Full | Mean accuracy delta | Mean Macro-F1 delta | Interpretation |
|---|---:|---:|---|
| raw terminal | −0.036 pp | −0.094 pp | raw terminal state is often competitive; no universal need for Stage-II readout |
| raw bank mean | +0.271 pp | +0.288 pp | adaptive eta/interacted readout gives a small average advantage, not a dominant one |
| plain backbone | +0.291 pp | +0.334 pp | Full is better in 14/15 paired comparisons, but the gap is small relative to the retained backbone |

The plain backbone removes the conditional components and naturally has fewer
parameters. It retains separate modality propagation, all four raw orders,
refinement, late fusion, and the same NC head. Thus its near-Full performance
cannot be explained by secretly retaining the removed modules.

## What can be concluded

- MRC, adaptive state formation, state-bank construction, interaction, and
  integration are present as separate executable paths.
- The paths are functionally live: inference interventions change embeddings
  and logits, and Full is modestly but consistently better than the strict
  plain backbone in paired means.
- The mechanism is not performance-dominant in the strong sense. Raw terminal
  and raw bank readouts remain close, with raw terminal better on Toys,
  Grocery, ele-fashion, and Reddit-S accuracy means.
- Dataset and modality dependence is real, so a bounded multi-granular claim
  is more defensible than a universal component-necessity claim.

## Story tier

**Tier B — mechanism-supported, performance-non-dominant.**

Tier A would require a clear and stable component-specific performance
advantage; the paired controls do not provide that. Tier C would require the
mechanism chain to be inactive or unsupported; the gate/intervention and
interaction-off diagnostics contradict that. Tier B matches both facts:
P2 is a coherent, functional unified framework, while the benchmark gains
are modest and mixed.

## Writing boundaries

Allowed: “P2 models modality-specific relation response, adaptive context
states, multi-order state banks, and interaction-aware integration as a
single executable chain.”

Not allowed: “every component is necessary,” “adaptive integration always
improves accuracy,” “semantic similarity is relation reliability,” or a
ranking of MRC, gating, state-bank, integration, and interaction by importance.
The node-wise M1 oracle remains descriptive only and is not a realizable model.

## Freeze recommendation

Freeze canonical P2 as the sole working architecture for the five-NC paper
scope. Do not add another trainable architecture module or launch another
architecture search based on these small differences. Keep the legacy
relation-order bias in the canonical implementation; its cleanup remains a
future controlled study, not a change for this round.
