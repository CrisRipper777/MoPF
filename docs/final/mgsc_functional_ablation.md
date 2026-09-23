# MGSC-MAG functional ablation report

Scope is node classification on Movies, Toys, Grocery, ele-fashion, and
Reddit-S with seeds 42/43/44. Sports-LP and all other LP experiments are out
of scope for this round. The official split, unified NC protocol, optimizer,
early stopping, checkpoint rule, classifier, preprocessing, and metrics were
unchanged. `full` is canonical P2; its formal runs were reused only after
matching the effective model configuration, split hash, protocol, and code
fingerprints.

The authoritative files are
`outputs/final/mgsc_functional_ablation/per_seed_results.csv`,
`summary.csv`, and `paired_deltas.csv`. Summary uncertainty is population SD
over the three seeds. Paired deltas are `Full - Ablation` for the same
dataset/seed, so positive values favour Full.

## Full P2 reference

| Dataset | Accuracy mean ± SD | Macro-F1 mean ± SD |
|---|---:|---:|
| Movies | 0.5640 ± 0.0115 | 0.5064 ± 0.0104 |
| Toys | 0.7990 ± 0.0053 | 0.7736 ± 0.0057 |
| Grocery | 0.8323 ± 0.0053 | 0.7566 ± 0.0067 |
| ele-fashion | 0.8833 ± 0.0019 | 0.7789 ± 0.0045 |
| Reddit-S | 0.9691 ± 0.0026 | 0.9321 ± 0.0035 |

## Paired mean deltas in percentage points

The following table reports the mean of the three paired deltas. It is a
compact view; the seed-level values and population variation are retained in
`paired_deltas.csv`.

| Dataset | Uniform relations Acc/F1 | Global context gate Acc/F1 | Terminal state only Acc/F1 | Uniform integration Acc/F1 | No cross-order interaction Acc/F1 |
|---|---:|---:|---:|---:|---:|
| Movies | −0.13 / +0.31 | −0.04 / +0.76 | −0.25 / −0.63 | +0.20 / +0.30 | +0.22 / +0.79 |
| Toys | −0.13 / +0.12 | −0.34 / −0.26 | −0.38 / −0.06 | +0.10 / +0.77 | −0.29 / −0.22 |
| Grocery | −0.17 / −0.07 | +0.23 / +0.03 | −0.15 / −0.23 | −0.03 / −0.17 | −0.29 / −0.60 |
| ele-fashion | +0.05 / +0.15 | +0.16 / −0.06 | +0.08 / +0.48 | +0.03 / +0.25 | +0.02 / −0.12 |
| Reddit-S | +0.00 / +0.07 | +0.07 / +0.15 | −0.06 / −0.08 | +0.11 / +0.19 | +0.04 / +0.08 |

These small mixed-sign changes are consistent with the frozen decision that
P1/P2 differences are normal experiment variation. The table is not used to
rank modules.

## What each controlled removal says

1. **MRC relation calibration (`uniform_relations`)**: mean accuracy changes
   are small and mixed across datasets; the result does not support a claim
   that relation calibration is uniformly necessary for NC performance.

2. **Node-adaptive gate (`global_context_gate`)**: accuracy changes are also
   small and mixed. The diagnostic supports functional node variation, but the
   ablation alone does not establish a universal accuracy gain from that
   variation.

3. **Terminal state only**: retaining only the terminal interacted state does
   not consistently reduce accuracy or Macro-F1. The state-bank mechanism is
   therefore supported as a representational/diagnostic hypothesis, not as a
   universally required performance component under this protocol.

4. **Uniform integration**: replacing learned coefficients by equal order
   weights produces mixed, mostly sub-percentage-point paired changes. This
   bounds the performance claim: learned integration is not shown to dominate
   uniformly across all five NC datasets.

5. **No cross-order interaction**: disabling order interaction likewise gives
   small mixed changes. The separate intervention diagnostics below show that
   the interaction path can change embeddings and logits, but that functional
   sensitivity is not identical to a guaranteed benchmark gain.

6. **Attribute only**: this is a sanity control, not a pure ablation of one
   innovation. It removes structural contextualization and gives lower mean
   performance on all five datasets, but it must not be interpreted as
   isolating one particular module.

No conclusion here uses test results to tune an ablation or to alter the
protocol. No ablation received a special hidden dimension, optimizer, seed, or
early-stopping rule.

## Integrity status

The A0/full path has the same parameter keys and shapes as canonical MGSC-MAG
and passes the exact forward smoke comparison to the base MGSC implementation
in eval mode. All seven formal variants pass finite NC forward/backward and
isolated-node checks. The model/task protocol remains outside the ablation
switch. Uniform relations use exactly one on every physical edge before the
normalization operator; global gates are node-invariant but modality/order
specific; terminal and uniform integration use their specified state paths;
no-cross-order uses identity interaction.

## Decision

The functional matrix does not justify architecture search or a module ranking.
It is compatible with freezing canonical P2 as the working architecture while
making bounded, mechanism-oriented claims in the paper. Performance claims
should report the mixed paired deltas rather than implying that every P2
component is necessary on every dataset.
