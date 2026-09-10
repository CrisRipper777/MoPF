# MoPF M1-F — PDC-v2 Cross-Task Transfer Validation

Only sports-copurchase LP seed=42 was run. Seeds 43/44 were not started.

## Training comparison

| Variant | Conditioner | Val MRR | Test MRR | Hits@1 | Hits@3 | Hits@10 | Best epoch | Params | Peak GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0_current | `absolute` | 0.402570 | 0.372659 | 0.216796 | 0.423462 | 0.723278 | 44 | 997069 | 16846.0 |
| S1_pdc_v1 | `pdc` | 0.395698 | 0.364033 | 0.208885 | 0.416566 | 0.709352 | 22 | 997077 | 17660.0 |
| S2_pdc_v2_sep | `pdc_v2_sep` | 0.393901 | 0.362712 | 0.208029 | 0.413680 | 0.707481 | 22 | 1003243 | unavailable |
| S3_pdc_v2_full | `pdc_v2_full` | 0.394020 | 0.365523 | 0.210301 | 0.418999 | 0.708925 | 22 | 1005301 | unavailable |

## Frozen intervention

| Variant | F0 Val MRR | F1 All-Off Val MRR drop | F1 Test MRR drop | Fused rel. L2 | Val score rel. L2 |
|---|---:|---:|---:|---:|---:|
| S1_pdc_v1 | 0.395698 | -0.019879 | -0.019106 | 0.110521 | 0.13232 |
| S2_pdc_v2_sep | 0.393901 | -0.021026 | -0.017451 | 0.112509 | 0.11446 |
| S3_pdc_v2_full | 0.394020 | -0.020059 | -0.014773 | 0.115593 | 0.120655 |

## Node-shuffle counterfactual

| Variant | Val MRR gain mean | Pop. std over permutations | Paired permutation SE |
|---|---:|---:|---:|
| S1_pdc_v1 | +0.0417417 | 0.00188742 | 0.000596853 |
| S2_pdc_v2_sep | +0.0384951 | 0.00196807 | 0.000622357 |
| S3_pdc_v2_full | +0.0321537 | 0.00207082 | 0.00065485 |

## C3 order0

- rho0 text=-0.334135, visual=+0.000230158
- fused relative L2=0.0130928; validation link-score relative L2=0.0118383
- Val MRR delta (PDC-On minus order0-off)=+0.000342298; Test MRR delta=+0.000695091
- order0 functionally weak: **False**

## C2 vs C3

- C3 minus C2 Val MRR: +0.000119
- C3 minus C2 Test MRR: +0.002811
- C3 minus C2 fused relative L2: +0.00308443
- The order0 branch is evaluated separately; it is not inferred from C3's overall score.

## Historical PDC-v1 reference

The following values are from the prior M1-D run and are not mixed into the formal S0–S3 comparison: Current seed42 Val/Test MRR 0.404909/0.373686; PDC-v1 seed42 0.402646/0.372636.

## Cross-task labels and expansion recommendation

- S1_pdc_v1: **viable**
- S2_pdc_v2_sep: **promising**
- S3_pdc_v2_full: **promising**

Recommendation field: **Expand C3**. This is a recommendation only; no seed43/44 run was performed.

Training comparison, frozen intervention, and node-shuffle counterfactuals are separate evidence types.
Peak GPU memory is marked unavailable for recovered S2/S3 runs because the original parent launcher exited after child completion; no value was inferred.

No final MoPF claim is made by this study.
