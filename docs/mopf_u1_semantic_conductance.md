# MoPF-vNext U1 — Modality-Conditioned Semantic Conductance

This report covers NC only: S0 Frozen Separate Cosine and S1 Multi-Perspective Semantic Conductance.
LP, sports-copurchase, the decoder, sampler, and LP protocol are intentionally excluded.

- Frozen reference behavior commit: `9b03e6fbeb73d888050833391d9859964335b9ea`
- vNext commit at analysis time: `4fe97ce6d726f2243d8ae58667348a0d0508966a`
- Test status: `clean`
- Candidate decision: **Conditional candidate**

## Downstream comparison

The authoritative numerical table is `outputs/u1_semantic_conductance/u1_master_table.csv`;
the JSON contains the per-seed records, population standard deviations, and S1-minus-S0 values.
Checkpoint selection is best validation accuracy only. No test metric is used for selection.

| Dataset | S0 Val Acc | S1 Val Acc | S1-S0 Val Acc | S0 Test Acc | S1 Test Acc |
|---|---:|---:|---:|---:|---:|
| Movies | 0.570786 | 0.571186 | +0.000400 | 0.560020 | 0.561819 |
| Toys | 0.803012 | 0.803817 | +0.000805 | 0.793509 | 0.792462 |
| Grocery | 0.836701 | 0.836798 | +0.000098 | 0.836115 | 0.835237 |
| ele-fashion | 0.880809 | 0.879956 | -0.000852 | 0.881987 | 0.881464 |
| Reddit-S | 0.962986 | 0.963301 | +0.000315 | 0.964664 | 0.963930 |

## Mechanism interpretation

The conductance distributions, modality gaps, normalized operator differences, neighborhood entropy,
label-aware edge summaries, and perspective diagnostics are reported without a preferred direction.
For heterophilous or non-label-homogeneous graphs, same-label conductance is not assumed to be higher.

## Candidate rationale

S1 validation remained within the declared 0.01 same-band tolerance on 5/5 datasets, and all completed runs passed finite/bounds/pathology checks=True. Structural conductance/operator change was detected on 5/5 datasets, but the predefined nonzero perspective-specialization threshold was met on only 0/5 datasets. The resulting classification is therefore Conditional candidate; it does not claim statistical significance, and entropy, operator distance, and perspective similarity are interpreted as mechanism diagnostics rather than monotone objectives.

The decision is a screening label, not a statistical significance claim.
