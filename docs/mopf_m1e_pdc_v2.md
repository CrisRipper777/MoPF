# MoPF M1-E — PDC-v2 Structural Refinement

This study retrains C0/C1/C2/C3 under one full-graph NC protocol. Sports is intentionally not run in M1-E.

- Runs: 60 (4 variants × 5 datasets × 3 seeds)
- Git commit: `f2f110d714db0a98e33c3e32e2618ee71207ccf9`

## Candidate recommendations

| Candidate | Recommendation |
|---|---|
| C1_pdc_v1 | **Keep** |
| C2_pdc_v2_sep | **Strong candidate** |
| C3_pdc_v2_full | **Strong candidate** |

## NC validation and active-path evidence

| Dataset | C0 Val Acc | C1 fused rel. L2 | C2 fused rel. L2 | C3 fused rel. L2 |
|---|---:|---:|---:|---:|
| Movies | 0.575585 | 0.000458382 | 0.00106389 | 0.00151401 |
| Toys | 0.802529 | 1.10776e-05 | 1.75506e-05 | 5.10156e-05 |
| Grocery | 0.840117 | 7.37934e-05 | 0.00102943 | 0.000776593 |
| ele-fashion | 0.881184 | 0.000347769 | 0.00220318 | 0.00579908 |
| Reddit-S | 0.964978 | 1.81527e-05 | 5.38277e-06 | 2.58335e-05 |

`active_path_alignment_gain` is computed per shared permutation seed as `shuffle_drop_on - shuffle_drop_off`; aggregates include population standard deviation, positive training-seed count, and paired 10-permutation standard error. The machine-readable `m1e_master_summary.json` is authoritative.
For the PDC-v2 comparison gate, `stronger than C1` requires a 10% mean fused/logit relative-change margin and the same direction in at least 2/3 training seeds; the summary retains the per-seed values.

## C3 order0 analysis

| Dataset | mean rho0 text | mean rho0 visual | fused rel. L2 | logit rel. L2 | mean Val Acc delta | mean Test Acc delta |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0.0441501 | -0.0145583 | 0.000852914 | 0.000682129 | 0.000199954 | 0 |
| Toys | -0.0194601 | -0.0464368 | 4.26904e-05 | 3.46754e-05 | 8.05457e-05 | 0 |
| Grocery | 0.0341642 | 0.0210801 | 0.00069213 | 0.00057684 | 0 | 9.76125e-05 |
| ele-fashion | -0.0202335 | -0.0572784 | 3.98309e-06 | 2.71093e-06 | 0 | 0 |
| Reddit-S | -0.0105194 | -0.0187618 | 1.19776e-05 | 9.81377e-06 | 0 | 0 |

The order0-off deltas are PDC-On minus order0-off; ele-fashion is included explicitly because nonzero learned rho0 does not by itself imply a large downstream effect.

No final MoPF claim is made by this study.
