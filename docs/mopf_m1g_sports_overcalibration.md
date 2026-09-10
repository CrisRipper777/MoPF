# MoPF M1-G — PDC Cross-Task Over-Calibration Diagnosis

Frozen commit: `c90a7f716ed85417109d662d02f6dafea41d536b`. Only Current/C3 seeds 43/44 were newly trained; seed42 was reused from M1-F.

## Three-seed training

| Variant | Val MRR | Test MRR | Test H@1 | Test H@3 | Test H@10 |
|---|---:|---:|---:|---:|---:|
| S0_current | 0.404718 ± 0.002066 | 0.374546 ± 0.001722 | 0.217260 ± 0.001265 | 0.428158 ± 0.003606 | 0.726370 ± 0.002273 |
| S3_pdc_v2_full | 0.399857 ± 0.005315 | 0.370908 ± 0.005221 | 0.213767 ± 0.004011 | 0.427516 ± 0.007503 | 0.718672 ± 0.007630 |

C3 − Current per-seed Val/Test MRR:
- seed42: -0.008550 / -0.007136
- seed43: -0.000632 / +0.001154
- seed44: -0.005401 / -0.004931
- mean: -0.004861 / -0.003638

## Frozen C3 interventions

| Condition | Val MRR gain over PDC-On | Test MRR gain over PDC-On | Positive Val seeds |
|---|---:|---:|---:|
| F0_pdc_on | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | 0/3 |
| F1_all_pdc_off | +0.021120 ± 0.001185 | +0.015121 ± 0.000395 | 3/3 |
| F2_text_pdc_off | +0.002244 ± 0.001388 | +0.001828 ± 0.001309 | 3/3 |
| F3_visual_pdc_off | +0.021112 ± 0.001613 | +0.016357 ± 0.001301 | 3/3 |
| F4_order0_off | +0.004311 ± 0.003315 | +0.002609 ± 0.002470 | 2/3 |
| F5_order1_off | +0.004476 ± 0.003068 | +0.002738 ± 0.002296 | 3/3 |
| F6_order2_off | +0.017007 ± 0.002903 | +0.012041 ± 0.001723 | 3/3 |
| F7_order3_off | +0.010525 ± 0.001707 | +0.008149 ± 0.001972 | 3/3 |

## Frozen global-strength sweep

alpha_star_global = **0.125**, selected only by mean Validation MRR.

| alpha | Mean Val MRR | Val std | Mean Test MRR |
|---:|---:|---:|---:|
| 0.000 | 0.420977 | 0.005450 | 0.386030 |
| 0.125 | 0.421038 | 0.005441 | 0.386701 |
| 0.250 | 0.418555 | 0.004791 | 0.385277 |
| 0.500 | 0.411344 | 0.004764 | 0.379739 |
| 0.750 | 0.404518 | 0.004478 | 0.374580 |
| 1.000 | 0.399857 | 0.005315 | 0.370908 |

## Modality scale grid

alpha_t_star = **0.5**, alpha_v_star = **0.0** (validation-only selection).

The heatmap source table is in `m1g_master_table.csv`; visual attenuation is assessed by comparing the validation surface across alpha_t/alpha_v, not by Test MRR.

## Sampled training-context vs full inference

Ratios are matched by global node ID using only sampled edge-label endpoints as target/root nodes; support-only nodes are excluded. Each C3 checkpoint uses 100 sampled batches and no optimizer step.

| Modality | Order | Median sampled/full ratio | Log-ratio abs. error | Spearman | Branch cosine | State cosine | Delta RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 0 | 0.9645 | 0.2524 | 0.8699 | 0.9047 | 1.0000 | 0.0129 |
| text | 1 | 0.9627 | 0.4510 | 0.8480 | 0.8597 | 0.9672 | 0.0307 |
| text | 2 | 0.8266 | 0.4981 | 0.5553 | 0.6423 | 0.9998 | 0.0620 |
| text | 3 | 0.9286 | 0.4861 | 0.6736 | 0.7411 | 0.9998 | 0.0338 |
| visual | 0 | 0.8770 | 0.3756 | 0.8307 | 0.8617 | 1.0000 | 0.0249 |
| visual | 1 | 0.8787 | 0.5062 | 0.7863 | 0.8143 | 1.0000 | 0.0039 |
| visual | 2 | 0.7591 | 0.5573 | 0.5383 | 0.6850 | 0.9996 | 0.1126 |
| visual | 3 | 0.7707 | 0.6284 | 0.5479 | 0.6819 | 0.9976 | 0.0435 |

## NC vs LP scale comparison

The table is an analysis-only comparison to the existing M1-E C3 aggregates; cross-task differences are not treated as causal evidence.

## Node-shuffle and order0

Active-path alignment gain (Val MRR) mean = +0.045960, positive seeds = 3/3. Strong alignment indicates functional dependence, not better model quality.
Order0-on minus order0-off Val MRR mean = -0.004311; positive seeds = 1/3.

## Final M1-G diagnosis

**Category B: PDC requires scale balancing**

an interior global alpha is materially better than alpha=1 on mean validation MRR while functional/alignment evidence remains nonzero.

This is a diagnostic category, not a final model declaration. No PDC-v3 or scale-balanced implementation was added in M1-G; no statistical significance claim is made from three training seeds.
