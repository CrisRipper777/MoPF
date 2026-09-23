# Context-formation analysis for canonical MGSC-MAG P2

This is an inference-only analysis of the full P2 best checkpoint for each of
the five NC datasets, using seed 42. It does not use test labels for model or
mechanism selection, and it does not treat the M1 preferred lambda as gate
ground truth. The source tables are under
`outputs/final/context_formation_analysis/`.

## Relation response

`relation_response.csv` summarizes physical edges before graph normalization.
For each modality it reports the semantic cosine score, the learned relation
weight, and their edge-level discrepancy/correlation. The
`text_visual_discrepancy` rows report the absolute text/visual difference in
semantic scores and in learned relation weights. This is a response audit of
the current MRC calibration; it is not a claim that semantic similarity is
relation reliability.

For the three datasets requested for the primary relation view:

| Dataset | Mean semantic T/V discrepancy | Mean relation-weight T/V discrepancy | Edge-level correlation |
|---|---:|---:|---:|
| Movies | 0.1827 | 0.0559 | 0.9478 |
| Grocery | 0.1511 | 0.0514 | 0.9365 |
| ele-fashion | 0.1457 | 0.0433 | 0.9568 |

The same table is exported for Toys and Reddit-S. The high correlation is
expected in part because the calibrated weight is a monotonic function of the
semantic score; the nonzero modality discrepancies demonstrate that the two
modality paths are not identical.

## Modality-specific neighbor allocation

For each non-isolated node, raw modality-specific relation weights are
normalized over that node's physical neighbors. `TV distance` is one half of
the L1 distance between the text and visual neighbor distributions, and
top-neighbor disagreement compares their highest-weight neighbor. No semantic
neighbors were added.

| Dataset | Non-isolated nodes | Isolated nodes | Mean TV distance | Top-neighbor disagreement |
|---|---:|---:|---:|---:|
| Movies | 16,672 | 0 | 0.0162 | 0.5777 |
| Toys | 20,685 | 10 | 0.0129 | 0.4705 |
| Grocery | 17,074 | 0 | 0.0166 | 0.4982 |
| ele-fashion | 97,763 | 3 | 0.0080 | 0.2982 |
| Reddit-S | 15,894 | 0 | 0.0116 | 0.5097 |

The top-neighbor statistic is more sensitive than mean TV distance because many
small reweightings can change an argmax. It should be read as modality-specific
allocation disagreement, not as a claim of different topology.

## Adaptive gate distributions and interventions

`adaptive_gate_summary.csv` reports mean, population node SD, and quantiles for
each modality/order. As examples, the mean gate over orders is approximately
0.305 for text and 0.789 for visual on Movies; it is approximately 0.635 and
0.860 on Grocery, and 0.661 and 0.644 on ele-fashion. Thus the gate behavior
is modality- and dataset-dependent rather than a fixed 0.9 rule.

The intervention table uses the same checkpoint and compares each result to
normal inference:

- `globalized`: replace each order's node gates by that order's node mean;
- `shuffled`: permute node assignments independently by modality/order while
  preserving the gate marginal distribution;
- metrics: embedding MAE, logit MAE, and prediction flip rate.

| Dataset | Globalized embedding MAE | Shuffled embedding MAE | Globalized flip | Shuffled flip |
|---|---:|---:|---:|---:|
| Movies | 0.0392 | 0.0516 | 0.0275 | 0.0326 |
| Toys | 0.0279 | 0.0369 | 0.0065 | 0.0092 |
| Grocery | 0.0645 | 0.0794 | 0.0141 | 0.0169 |
| ele-fashion | 0.0587 | 0.0717 | 0.0134 | 0.0162 |
| Reddit-S | 0.0380 | 0.0514 | 0.0033 | 0.0042 |

The nonzero shuffled changes, with preserved marginal gate distributions, are
consistent with functional use of node-to-gate correspondence. They do not
prove that the gate recovers a unique causal or optimal context preference.
The isolated nodes in Toys and ele-fashion remain finite and are excluded from
neighbor-allocation statistics; gate inference itself remains defined.

## Bounded conclusion

The current diagnostics support a real computational chain from modality-
specific relation calibration to modality-specific context formation. They
support node/modality/order heterogeneity as a functional property of P2, but
they do not establish a supervised semantic target for the gate, nor do they
imply universal performance gains from every adaptive component.
