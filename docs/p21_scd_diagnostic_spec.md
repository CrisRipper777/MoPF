# P2.1 diagnostic specification

This document registers the diagnostics before formal SCD NC training. All
diagnostics are descriptive mechanism checks; none is a causal claim and none
uses Test metrics for architecture selection.

## R1 — semantic conductance

For each dataset, seed, and modality, report same-edge text/visual semantic
compatibility discrepancy using Pearson, Spearman, and MAD. Report the
distribution of `a` (mean, standard deviation, q10/q50/q90, and fractions
`|a|>.8` and `|a|>.9`), relation weights (mean, standard deviation, CV, q01,
q99), normalized-operator versus raw-unit operator MAE/RMSE/relative-L1, and
the distribution of local relation state `c`. Operator perturbation magnitude
is not assigned a desired direction.

## R2 — semantic feedback restart

For every modality and hop, report `p`, `rho_p*p`, `d`, `rho_d*d`, `alpha`, and
the parameters `b`, `rho_p`, `rho_d`. Report alpha IQR, q10/q50/q90,
`(q90-q10)/(abs(mean)+eps)`, low/high saturation fractions, and Pearson and
Spearman correlations of `d` with `alpha` and `rho_d*d` with `alpha`. These
correlations describe feedback behavior and are not causal evidence.

## R3 — formation-conditioned signed response

For every modality and hop, report `gamma`, `DeltaGamma`, `delta_content`,
`delta_ref`, `delta_rel`, and signed `eta`, including mean, standard deviation,
absolute mean, q10/q50/q90, negative fraction, and effective order. Report
node-level covariance-over-variance attribution for content, reference, and
relation terms when `Var(eta)>eps`; their theoretical sum is checked within a
numerical tolerance.

## Chain and efficiency

`p21_feedback_chain.csv` keeps the evidence as three separate links:

```text
R1 semantic discrepancy -> conductance dispersion
R2 semantic drift d     -> restart alpha / term_d
R3 c / alpha-state      -> eta variance contribution
```

`p21_efficiency.csv` compares P2 Full, P2 `no_cross_hop_interaction`, and
clean S on trainable/total parameter count, runtime, peak GPU memory, best
epoch, and a fixed-evaluation forward latency when available. P2
`no_cross_hop_interaction` still computes dead attention and therefore is not
the clean runtime reference.

## Decision gate

S versus P2 `no_cross_hop_interaction` is judged using validation accuracy and
validation Macro-F1, mechanism health, and efficiency. A five-dataset mean
drop below `-0.30` percentage points in either validation metric, or at least
three datasets with a simultaneous clear drop in both, is
`REVIEW_REQUIRED`. Otherwise, if mechanisms are healthy and efficiency
improves, the status is `SIMPLIFICATION_PASS`. Test metrics are descriptive
only.
