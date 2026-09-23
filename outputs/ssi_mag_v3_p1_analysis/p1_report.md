# SSI-MAG-V3 P1 Full NC mechanism health check

- Repository branch / commit: `V3` / `1057bfc623efa6db8ce3b422a8dabef67c939743`
- Input root: `outputs/ssi_mag_v3_p1_full`
- Protocol: `model=ssi_mag_v3`, `ablation=full`, NC only, unified full-graph, seeds 42/43/44.
- Completed runs analyzed: 15/15. No retraining was performed by this analyzer.
- Checkpoint selection used validation Accuracy; test metrics are descriptive final metrics only.
- LP jobs: 0; ablation jobs: 0; hyperparameter search: 0.

## Performance

| Dataset | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | best epoch | trained epochs |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.5736 ± 0.0024 | 0.4936 ± 0.0075 | 0.5657 ± 0.0025 | 0.4952 ± 0.0052 | 65.3333 ± 14.0554 | 95.3333 ± 14.0554 |
| Toys | 0.8045 ± 0.0005 | 0.7778 ± 0.0034 | 0.7995 ± 0.0080 | 0.7731 ± 0.0056 | 57.3333 ± 2.8674 | 87.3333 ± 2.8674 |
| Grocery | 0.8378 ± 0.0043 | 0.7693 ± 0.0080 | 0.8298 ± 0.0059 | 0.7520 ± 0.0073 | 78.6667 ± 2.6247 | 108.6667 ± 2.6247 |
| ele-fashion | 0.8813 ± 0.0006 | 0.7621 ± 0.0014 | 0.8821 ± 0.0010 | 0.7707 ± 0.0049 | 120.0000 ± 10.7083 | 150.0000 ± 10.7083 |
| Reddit-S | 0.9656 ± 0.0025 | 0.9266 ± 0.0042 | 0.9684 ± 0.0021 | 0.9309 ± 0.0030 | 57.3333 ± 24.8507 | 87.3333 ± 24.8507 |
| ALL | 0.8126 ± 0.1311 | 0.7459 ± 0.1402 | 0.8091 ± 0.1346 | 0.7444 ± 0.1405 | 75.7333 ± 27.1967 | 105.7333 ± 27.1967 |

## Mechanism health snapshot (population mean over available seeds)

| Dataset | beta T | beta V | operator rel-L1 T | operator rel-L1 V | alpha k1 std T | alpha k1 std V | g_delta T | g_int T | relation-off fused rel-L2 | interaction-off fused rel-L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.0524 | 0.0515 | 0.0184 | 0.0124 | 0.0004 | 0.0028 | 0.1401 | 0.0994 | 0.0063 | 0.2168 |
| Toys | 0.0517 | 0.0507 | 0.0150 | 0.0131 | 0.0018 | 0.0017 | 0.1473 | 0.1023 | 0.0036 | 0.1623 |
| Grocery | 0.0530 | 0.0512 | 0.0194 | 0.0132 | 0.0019 | 0.0024 | 0.1641 | 0.1238 | 0.0051 | 0.2598 |
| ele-fashion | 0.0543 | 0.0540 | 0.0160 | 0.0195 | 0.0030 | 0.0048 | 0.1985 | 0.1429 | 0.0073 | 0.1877 |
| Reddit-S | 0.0514 | 0.0514 | 0.0106 | 0.0097 | 0.0018 | 0.0024 | 0.1610 | 0.1018 | 0.0062 | 0.1033 |
| ALL | 0.0526 | 0.0518 | 0.0159 | 0.0136 | 0.0018 | 0.0028 | 0.1622 | 0.1140 | 0.0057 | 0.1860 |

High-dimensional Q/S quantiles use a deterministic uniformly strided sample capped at 1,000,000 values to bound analyzer memory; full-tensor mean/std/min/max are retained.
Train loss is recovered from the existing train log at the validation-selected best epoch; the framework does not log validation loss (val_loss_logged=0).

## Mechanism evidence

R1 reports learned beta, relation residual/weight distributions, local adaptation, and normalized-operator perturbation against the same physical topology with unit edge weights and the same self-loop policy.
R2 reports per-hop alpha and d, semantic prior p, and learned bias/rho parameters. Alpha saturation flags use alpha < 0.05 or > 0.95.
R3 reports context-change/interactions gates, D norms, attention entropy, diagonal/off-diagonal mass, and signed eta/effective-order statistics.
Attention row diversity is the mean pairwise distance across query rows of the same node: mean L1/L2 distance over all unordered row pairs.

## Frozen functional sensitivity

relation_off and interaction_off are post-hoc frozen interventions evaluated with the same saved NC classifier/head. They are sensitivity diagnostics, not retrained causal ablations.
The per-run CSV contains embedding MAE/relative-L2/cosine changes, prediction flip rates, and validation/test Accuracy and Macro-F1 deltas.

## Automatic flags

- `Movies/seed42`: R1_text_beta_near_init;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Movies/seed43`: R1_text_beta_near_init;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Movies/seed44`: R1_text_beta_near_init;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Toys/seed42`: R1_text_beta_near_init;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Toys/seed43`: R1_text_beta_near_init;R2_text_alpha_near_fixed;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Toys/seed44`: R1_text_beta_near_init;R2_text_alpha_near_fixed;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Grocery/seed42`: R1_text_beta_near_init;R2_text_alpha_near_fixed;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Grocery/seed43`: R1_text_beta_near_init;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Grocery/seed44`: R1_text_beta_near_init;R2_text_alpha_near_fixed;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `ele-fashion/seed42`: R1_text_beta_near_init;R1_visual_beta_near_init
- `ele-fashion/seed43`: R1_text_beta_near_init;R1_visual_beta_near_init
- `ele-fashion/seed44`: R1_text_beta_near_init;R1_visual_beta_near_init
- `Reddit-S/seed42`: R1_text_beta_near_init;R2_text_alpha_near_fixed;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Reddit-S/seed43`: R1_text_beta_near_init;R1_visual_beta_near_init;R2_visual_alpha_near_fixed
- `Reddit-S/seed44`: R1_text_beta_near_init;R2_text_alpha_near_fixed;R1_visual_beta_near_init;R2_visual_alpha_near_fixed

## Historical comparison

| Dataset | Method | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |
|---|---|---:|---:|---:|---:|
| Movies | SSI-MAG-V3 Full | 0.5736 | 0.4936 | 0.5657 | 0.4952 |
| Movies | CoSI-MAG Final | 0.5751 | 0.4996 | 0.5650 | 0.5015 |
| Movies | All Plain | 0.5744 | 0.5133 | 0.5621 | 0.5059 |
| Toys | SSI-MAG-V3 Full | 0.8045 | 0.7778 | 0.7995 | 0.7731 |
| Toys | CoSI-MAG Final | 0.8043 | 0.7793 | 0.7937 | 0.7682 |
| Toys | All Plain | 0.8033 | 0.7723 | 0.7953 | 0.7670 |
| Grocery | SSI-MAG-V3 Full | 0.8378 | 0.7693 | 0.8298 | 0.7520 |
| Grocery | CoSI-MAG Final | 0.8391 | 0.7847 | 0.8356 | 0.7608 |
| Grocery | All Plain | 0.8346 | 0.7711 | 0.8338 | 0.7601 |
| ele-fashion | SSI-MAG-V3 Full | 0.8813 | 0.7621 | 0.8821 | 0.7707 |
| ele-fashion | CoSI-MAG Final | 0.8823 | 0.7672 | 0.8821 | 0.7746 |
| ele-fashion | All Plain | 0.8797 | 0.7640 | 0.8786 | 0.7732 |
| Reddit-S | SSI-MAG-V3 Full | 0.9656 | 0.9266 | 0.9684 | 0.9309 |
| Reddit-S | CoSI-MAG Final | 0.9637 | 0.9324 | 0.9639 | 0.9239 |
| Reddit-S | All Plain | 0.9612 | 0.9278 | 0.9619 | 0.9184 |
| ALL | SSI-MAG-V3 Full | 0.8126 | 0.7459 | 0.8091 | 0.7444 |
| ALL | CoSI-MAG Final | 0.8129 | 0.7526 | 0.8081 | 0.7458 |
| ALL | All Plain | 0.8106 | 0.7497 | 0.8063 | 0.7449 |

Historical rows are descriptive only; no baseline was rerun and no test metric was used for model selection.

## Interpretation boundary

Performance evidence, mechanism behavior evidence, and frozen functional sensitivity are reported as separate evidence types. This report does not declare a mechanism failed or propose an unvalidated fix.
