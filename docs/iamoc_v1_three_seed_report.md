# IAMOC v1 Three-Seed Exploratory Report (NC complete; LP pending)

## 1. Implementation Summary

IAMOC adds per-modality, within-node attention over the Stage-II propagation-order states before the inherited low-rank node scorer. The scorer still uses the inherited projectors and order vectors. The learned coefficients compose the original Stage-II response bank; attention outputs do not replace structural responses.

## 2. Baseline Invariance

Stage I relation calibration, Stage II propagation, fusion, task heads, split handling, optimizer, checkpoint selection, and evaluation remain inherited from the formal MoPF implementation. The synthetic equivalence audit loads all shared MoPF parameters and checks exact output equality with hop interaction disabled.

## 3. Variant Definitions

- V0: formal CoSI-MAG / `mopf`.
- V1: one-layer IAMOC with output TCPR conditioning.
- V2: one-layer IAMOC without relation conditioning.
- V3: one-layer IAMOC with relation bias in hop attention and no output TCPR.
- V4: two-layer IAMOC with relation bias in hop attention and no output TCPR.

## 4. Correctness Tests

Synthetic baseline equivalence, K=2/K=3 forward/backward, V2 relation isolation, V3 relation-bias/no-output-TCPR, absolute-conditioner fail-fast, config parity, and protected-file checks passed. The five Movies-NC seed-42 one-epoch smoke runs completed with checkpoints and results artifacts. Attention export and V3 relation interventions were exercised from those smoke checkpoints. Five partial sports LP attempts were interrupted before producing checkpoints; they are itemized in `outputs/iamoc_v1/analysis/aborted_attempts.json` and excluded from the result grid.

## 5. Three-Seed Main Results

### Movies

| Variant | Best epoch (seed:epoch) | val_acc mean ± std | test_acc mean ± std | test_macro_f1 mean ± std |
|---|---:|---:|---:|---:|
| V0 | 42:63; 43:74; 44:66 | 0.5762 ± 0.0024 (n=3) | 0.5582 ± 0.0044 (n=3) | 0.4988 ± 0.0048 (n=3) |
| V1 | 42:81; 43:76; 44:75 | 0.5779 ± 0.0023 (n=3) | 0.5612 ± 0.0016 (n=3) | 0.4956 ± 0.0064 (n=3) |
| V2 | 42:71; 43:78; 44:90 | 0.5795 ± 0.0059 (n=3) | 0.5640 ± 0.0042 (n=3) | 0.5045 ± 0.0056 (n=3) |
| V3 | 42:80; 43:70; 44:75 | 0.5756 ± 0.0012 (n=3) | 0.5581 ± 0.0046 (n=3) | 0.4973 ± 0.0056 (n=3) |
| V4 | 42:121; 43:63; 44:75 | 0.5806 ± 0.0017 (n=3) | 0.5648 ± 0.0025 (n=3) | 0.5037 ± 0.0021 (n=3) |

### Grocery

| Variant | Best epoch (seed:epoch) | val_acc mean ± std | test_acc mean ± std | test_macro_f1 mean ± std |
|---|---:|---:|---:|---:|
| V0 | 42:71; 43:65; 44:85 | 0.8377 ± 0.0015 (n=3) | 0.8305 ± 0.0047 (n=3) | 0.7514 ± 0.0135 (n=3) |
| V1 | 42:101; 43:82; 44:71 | 0.8375 ± 0.0040 (n=3) | 0.8328 ± 0.0043 (n=3) | 0.7595 ± 0.0094 (n=3) |
| V2 | 42:119; 43:81; 44:84 | 0.8386 ± 0.0036 (n=3) | 0.8335 ± 0.0023 (n=3) | 0.7604 ± 0.0067 (n=3) |
| V3 | 42:86; 43:75; 44:150 | 0.8398 ± 0.0018 (n=3) | 0.8308 ± 0.0036 (n=3) | 0.7559 ± 0.0087 (n=3) |
| V4 | 42:92; 43:95; 44:155 | 0.8382 ± 0.0020 (n=3) | 0.8304 ± 0.0038 (n=3) | 0.7565 ± 0.0074 (n=3) |

### sports-copurchase

| Variant | Best epoch (seed:epoch) | val_mrr mean ± std | test_mrr mean ± std | test_hits@1 mean ± std | test_hits@3 mean ± std | test_hits@10 mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| V0 | — | — | — | — | — | — |
| V1 | — | — | — | — | — | — |
| V2 | — | — | — | — | — | — |
| V3 | — | — | — | — | — | — |
| V4 | — | — | — | — | — | — |

## 6. Paired Seed Comparison

Paired values are Variant(seed) minus V0(seed), with no seed exclusion:

- `Movies/V1` `val_acc`: seed 42: +0.0039, seed 43: -0.0006, seed 44: +0.0018
- `Movies/V1` `val_macro_f1`: seed 42: +0.0174, seed 43: +0.0040, seed 44: -0.0007
- `Movies/V1` `test_acc`: seed 42: +0.0048, seed 43: +0.0054, seed 44: -0.0012
- `Movies/V1` `test_macro_f1`: seed 42: -0.0063, seed 43: -0.0023, seed 44: -0.0010
- `Movies/V2` `val_acc`: seed 42: -0.0006, seed 43: +0.0027, seed 44: +0.0078
- `Movies/V2` `val_macro_f1`: seed 42: +0.0078, seed 43: +0.0068, seed 44: +0.0160
- `Movies/V2` `test_acc`: seed 42: +0.0090, seed 43: +0.0045, seed 44: +0.0039
- `Movies/V2` `test_macro_f1`: seed 42: +0.0036, seed 43: +0.0008, seed 44: +0.0127
- `Movies/V3` `val_acc`: seed 42: +0.0024, seed 43: -0.0021, seed 44: -0.0021
- `Movies/V3` `val_macro_f1`: seed 42: +0.0038, seed 43: +0.0136, seed 44: +0.0140
- `Movies/V3` `test_acc`: seed 42: +0.0009, seed 43: -0.0009, seed 44: -0.0003
- `Movies/V3` `test_macro_f1`: seed 42: -0.0040, seed 43: -0.0049, seed 44: +0.0045
- `Movies/V4` `val_acc`: seed 42: +0.0096, seed 43: +0.0042, seed 44: -0.0006
- `Movies/V4` `val_macro_f1`: seed 42: +0.0252, seed 43: +0.0128, seed 44: +0.0042
- `Movies/V4` `test_acc`: seed 42: +0.0048, seed 43: +0.0135, seed 44: +0.0015
- `Movies/V4` `test_macro_f1`: seed 42: +0.0080, seed 43: -0.0017, seed 44: +0.0086
- `Grocery/V1` `val_acc`: seed 42: +0.0029, seed 43: +0.0009, seed 44: -0.0044
- `Grocery/V1` `val_macro_f1`: seed 42: +0.0119, seed 43: -0.0009, seed 44: +0.0004
- `Grocery/V1` `test_acc`: seed 42: +0.0020, seed 43: +0.0035, seed 44: +0.0015
- `Grocery/V1` `test_macro_f1`: seed 42: +0.0115, seed 43: +0.0133, seed 44: -0.0007
- `Grocery/V2` `val_acc`: seed 42: +0.0029, seed 43: +0.0023, seed 44: -0.0026
- `Grocery/V2` `val_macro_f1`: seed 42: +0.0089, seed 43: +0.0043, seed 44: +0.0063
- `Grocery/V2` `test_acc`: seed 42: -0.0003, seed 43: +0.0056, seed 44: +0.0038
- `Grocery/V2` `test_macro_f1`: seed 42: +0.0053, seed 43: +0.0185, seed 44: +0.0030
- `Grocery/V3` `val_acc`: seed 42: +0.0032, seed 43: +0.0020, seed 44: +0.0012
- `Grocery/V3` `val_macro_f1`: seed 42: +0.0213, seed 43: -0.0018, seed 44: +0.0218
- `Grocery/V3` `test_acc`: seed 42: -0.0035, seed 43: +0.0006, seed 44: +0.0041
- `Grocery/V3` `test_macro_f1`: seed 42: -0.0003, seed 43: +0.0120, seed 44: +0.0016
- `Grocery/V4` `val_acc`: seed 42: +0.0003, seed 43: +0.0012, seed 44: +0.0000
- `Grocery/V4` `val_macro_f1`: seed 42: +0.0098, seed 43: +0.0020, seed 44: -0.0005
- `Grocery/V4` `test_acc`: seed 42: -0.0023, seed 43: +0.0000, seed 44: +0.0020
- `Grocery/V4` `test_macro_f1`: seed 42: +0.0056, seed 43: +0.0130, seed 44: -0.0033
- `sports-copurchase/V1` `val_mrr`: no paired completed run
- `sports-copurchase/V1` `test_mrr`: no paired completed run
- `sports-copurchase/V1` `test_hits@1`: no paired completed run
- `sports-copurchase/V1` `test_hits@3`: no paired completed run
- `sports-copurchase/V1` `test_hits@10`: no paired completed run
- `sports-copurchase/V2` `val_mrr`: no paired completed run
- `sports-copurchase/V2` `test_mrr`: no paired completed run
- `sports-copurchase/V2` `test_hits@1`: no paired completed run
- `sports-copurchase/V2` `test_hits@3`: no paired completed run
- `sports-copurchase/V2` `test_hits@10`: no paired completed run
- `sports-copurchase/V3` `val_mrr`: no paired completed run
- `sports-copurchase/V3` `test_mrr`: no paired completed run
- `sports-copurchase/V3` `test_hits@1`: no paired completed run
- `sports-copurchase/V3` `test_hits@3`: no paired completed run
- `sports-copurchase/V3` `test_hits@10`: no paired completed run
- `sports-copurchase/V4` `val_mrr`: no paired completed run
- `sports-copurchase/V4` `test_mrr`: no paired completed run
- `sports-copurchase/V4` `test_hits@1`: no paired completed run
- `sports-copurchase/V4` `test_hits@3`: no paired completed run
- `sports-copurchase/V4` `test_hits@10`: no paired completed run

## 7. Stability

Completed runs: 30 / 45. Missing or failed final runs: 15. Final runs with NaN/Inf log tokens: 0. 5 uncheckpointed LP attempts are listed separately and excluded. Per-run status, best epoch, and return code are in the machine-readable run summary and each run's `run_record.json`.

## 8. Efficiency

| Dataset | Variant | Encoder params | Extra vs V0 | Total trainable params | Runtime s mean ± std | Peak memory MiB mean ± std |
|---|---|---:|---:|---:|---:|---:|
| Movies | V0 | 996692.0 | 0.0 | 1001832.0 ± 0.0 | 17.4 ± 0.8 | 2136.7 ± 0.0 |
| Movies | V1 | 1394520.0 | 397828.0 | 1399660.0 ± 0.0 | 22.7 ± 0.5 | 3042.6 ± 0.0 |
| Movies | V2 | 1394520.0 | 397828.0 | 1399660.0 ± 0.0 | 23.7 ± 1.6 | 3042.6 ± 0.0 |
| Movies | V3 | 1394520.0 | 397828.0 | 1399660.0 ± 0.0 | 22.9 ± 0.8 | 3042.8 ± 0.0 |
| Movies | V4 | 1790296.0 | 793604.0 | 1795436.0 ± 0.0 | 31.6 ± 6.0 | 3839.9 ± 0.0 |
| Grocery | V0 | 996692.0 | 0.0 | 1001832.0 ± 0.0 | 20.6 ± 1.6 | 2052.8 ± 0.0 |
| Grocery | V1 | 1394520.0 | 397828.0 | 1399660.0 ± 0.0 | 27.0 ± 2.8 | 2981.8 ± 0.0 |
| Grocery | V2 | 1394520.0 | 397828.0 | 1399660.0 ± 0.0 | 29.8 ± 4.0 | 2981.8 ± 0.0 |
| Grocery | V3 | 1394520.0 | 397828.0 | 1399660.0 ± 0.0 | 29.6 ± 6.5 | 2981.9 ± 0.0 |
| Grocery | V4 | 1790296.0 | 793604.0 | 1795436.0 ± 0.0 | 41.1 ± 12.3 | 3789.4 ± 0.0 |
| sports-copurchase | V0 | — | — | — | — | — |
| sports-copurchase | V1 | — | — | — | — | — |
| sports-copurchase | V2 | — | — | — | — | — |
| sports-copurchase | V3 | — | — | — | — | — |
| sports-copurchase | V4 | — | — | — | — | — |

## 9. Hop Interaction Mechanism

Per-checkpoint Text/Visual mean matrices, per-node profiles, and numerical summaries are under `outputs/iamoc_v1/analysis/`. The matrices and node values are also available as CSV for plotting.

## 10. Node-Level Heterogeneity

- Movies V1 seed 42: Text r_attn mean/std=0.6106/0.0974; Visual=0.4486/0.2293.
- Movies V1 seed 43: Text r_attn mean/std=0.6479/0.0966; Visual=0.4312/0.2241.
- Movies V1 seed 44: Text r_attn mean/std=0.6513/0.1078; Visual=0.4755/0.2661.
- Movies V2 seed 42: Text r_attn mean/std=0.6274/0.1112; Visual=0.3978/0.2261.
- Movies V2 seed 43: Text r_attn mean/std=0.6516/0.0949; Visual=0.4516/0.2228.
- Movies V2 seed 44: Text r_attn mean/std=0.6494/0.1041; Visual=0.4409/0.2718.
- Movies V3 seed 42: Text r_attn mean/std=0.6182/0.1056; Visual=0.4823/0.2274.
- Movies V3 seed 43: Text r_attn mean/std=0.6441/0.0981; Visual=0.4356/0.2174.
- Movies V3 seed 44: Text r_attn mean/std=0.6505/0.1068; Visual=0.4750/0.2658.
- Movies V4 seed 42: Text r_attn mean/std=0.5637/0.0581; Visual=0.3600/0.2442.
- Movies V4 seed 43: Text r_attn mean/std=0.5696/0.0879; Visual=0.4849/0.1724.
- Movies V4 seed 44: Text r_attn mean/std=0.5742/0.0708; Visual=0.4142/0.2030.
- Grocery V1 seed 42: Text r_attn mean/std=0.5244/0.2615; Visual=0.5267/0.2957.
- Grocery V1 seed 43: Text r_attn mean/std=0.6078/0.1357; Visual=0.5384/0.3190.
- Grocery V1 seed 44: Text r_attn mean/std=0.5477/0.1367; Visual=0.4875/0.2690.
- Grocery V2 seed 42: Text r_attn mean/std=0.5061/0.2545; Visual=0.5575/0.2861.
- Grocery V2 seed 43: Text r_attn mean/std=0.6263/0.1323; Visual=0.5099/0.3182.
- Grocery V2 seed 44: Text r_attn mean/std=0.5075/0.1970; Visual=0.5019/0.2590.
- Grocery V3 seed 42: Text r_attn mean/std=0.5571/0.2073; Visual=0.4286/0.2708.
- Grocery V3 seed 43: Text r_attn mean/std=0.6065/0.1223; Visual=0.5411/0.3129.
- Grocery V3 seed 44: Text r_attn mean/std=0.3329/0.2773; Visual=0.4917/0.2533.
- Grocery V4 seed 42: Text r_attn mean/std=0.5735/0.1592; Visual=0.5236/0.2216.
- Grocery V4 seed 43: Text r_attn mean/std=0.6485/0.1180; Visual=0.5510/0.2619.
- Grocery V4 seed 44: Text r_attn mean/std=0.3188/0.2611; Visual=0.5365/0.2657.

## 11. Modality-Level Heterogeneity

- Movies V1 seed 42: mean attention matrix MAE Text-vs-Visual=0.123819; mean key-mass MAE=0.123819; entropy delta=+0.189370.
- Movies V1 seed 43: mean attention matrix MAE Text-vs-Visual=0.143322; mean key-mass MAE=0.143322; entropy delta=+0.147849.
- Movies V1 seed 44: mean attention matrix MAE Text-vs-Visual=0.132360; mean key-mass MAE=0.132360; entropy delta=+0.248350.
- Movies V2 seed 42: mean attention matrix MAE Text-vs-Visual=0.153697; mean key-mass MAE=0.153697; entropy delta=+0.183037.
- Movies V2 seed 43: mean attention matrix MAE Text-vs-Visual=0.131859; mean key-mass MAE=0.131859; entropy delta=+0.135326.
- Movies V2 seed 44: mean attention matrix MAE Text-vs-Visual=0.149106; mean key-mass MAE=0.149106; entropy delta=+0.317287.
- Movies V3 seed 42: mean attention matrix MAE Text-vs-Visual=0.109157; mean key-mass MAE=0.109157; entropy delta=+0.153576.
- Movies V3 seed 43: mean attention matrix MAE Text-vs-Visual=0.134956; mean key-mass MAE=0.134957; entropy delta=+0.126949.
- Movies V3 seed 44: mean attention matrix MAE Text-vs-Visual=0.133974; mean key-mass MAE=0.133974; entropy delta=+0.254919.
- Movies V4 seed 42: mean attention matrix MAE Text-vs-Visual=0.140408; mean key-mass MAE=0.140408; entropy delta=+0.379505.
- Movies V4 seed 43: mean attention matrix MAE Text-vs-Visual=0.057647; mean key-mass MAE=0.057647; entropy delta=+0.086023.
- Movies V4 seed 44: mean attention matrix MAE Text-vs-Visual=0.107971; mean key-mass MAE=0.107971; entropy delta=+0.195485.
- Grocery V1 seed 42: mean attention matrix MAE Text-vs-Visual=0.060127; mean key-mass MAE=0.043420; entropy delta=+0.142926.
- Grocery V1 seed 43: mean attention matrix MAE Text-vs-Visual=0.112019; mean key-mass MAE=0.112019; entropy delta=+0.417701.
- Grocery V1 seed 44: mean attention matrix MAE Text-vs-Visual=0.066006; mean key-mass MAE=0.051212; entropy delta=+0.291367.
- Grocery V2 seed 42: mean attention matrix MAE Text-vs-Visual=0.067273; mean key-mass MAE=0.057370; entropy delta=+0.122761.
- Grocery V2 seed 43: mean attention matrix MAE Text-vs-Visual=0.113553; mean key-mass MAE=0.113553; entropy delta=+0.394102.
- Grocery V2 seed 44: mean attention matrix MAE Text-vs-Visual=0.045507; mean key-mass MAE=0.031445; entropy delta=+0.179865.
- Grocery V3 seed 42: mean attention matrix MAE Text-vs-Visual=0.094867; mean key-mass MAE=0.094867; entropy delta=+0.246296.
- Grocery V3 seed 43: mean attention matrix MAE Text-vs-Visual=0.112907; mean key-mass MAE=0.112907; entropy delta=+0.414751.
- Grocery V3 seed 44: mean attention matrix MAE Text-vs-Visual=0.128397; mean key-mass MAE=0.128397; entropy delta=-0.219021.
- Grocery V4 seed 42: mean attention matrix MAE Text-vs-Visual=0.033786; mean key-mass MAE=0.033135; entropy delta=+0.108043.
- Grocery V4 seed 43: mean attention matrix MAE Text-vs-Visual=0.064578; mean key-mass MAE=0.063941; entropy delta=+0.228243.
- Grocery V4 seed 44: mean attention matrix MAE Text-vs-Visual=0.132246; mean key-mass MAE=0.132246; entropy delta=-0.129510.

## 12. Relation-Bias Mechanism

V3 relation-context associations with attended order use partial Spearman correlation controlling for `log(1 + physical degree)`. These are associations, not causal effects.

- Movies seed 42 text: partial Spearman=-0.5032171442071958 ; relation quartile mean r_attn=Q1 0.6860 (n=4168), Q2 0.6244 (n=4168), Q3 0.5939 (n=4167), Q4 0.5687 (n=4169)
- Movies seed 42 visual: partial Spearman=-0.04056039096131565 ; relation quartile mean r_attn=Q1 0.4784 (n=4168), Q2 0.4908 (n=4168), Q3 0.4976 (n=4168), Q4 0.4624 (n=4168)
- Movies seed 43 text: partial Spearman=-0.5015365921249707 ; relation quartile mean r_attn=Q1 0.7084 (n=4168), Q2 0.6497 (n=4168), Q3 0.6246 (n=4168), Q4 0.5936 (n=4168)
- Movies seed 43 visual: partial Spearman=-0.02522259698400699 ; relation quartile mean r_attn=Q1 0.4448 (n=4168), Q2 0.4358 (n=4168), Q3 0.4169 (n=4168), Q4 0.4449 (n=4168)
- Movies seed 44 text: partial Spearman=-0.5059382957835244 ; relation quartile mean r_attn=Q1 0.7172 (n=4168), Q2 0.6614 (n=4168), Q3 0.6291 (n=4168), Q4 0.5942 (n=4168)
- Movies seed 44 visual: partial Spearman=-0.07229078277885995 ; relation quartile mean r_attn=Q1 0.4973 (n=4168), Q2 0.4814 (n=4168), Q3 0.4593 (n=4168), Q4 0.4619 (n=4168)
- Grocery seed 42 text: partial Spearman=-0.17097679181561717 ; relation quartile mean r_attn=Q1 0.5945 (n=4269), Q2 0.5620 (n=4268), Q3 0.5524 (n=4268), Q4 0.5197 (n=4269)
- Grocery seed 42 visual: partial Spearman=0.07756045828371928 ; relation quartile mean r_attn=Q1 0.4230 (n=4269), Q2 0.3925 (n=4268), Q3 0.4293 (n=4268), Q4 0.4699 (n=4269)
- Grocery seed 43 text: partial Spearman=-0.2602888599064223 ; relation quartile mean r_attn=Q1 0.6391 (n=4269), Q2 0.6201 (n=4268), Q3 0.6027 (n=4268), Q4 0.5641 (n=4269)
- Grocery seed 43 visual: partial Spearman=0.1265646592593578 ; relation quartile mean r_attn=Q1 0.4880 (n=4269), Q2 0.5140 (n=4268), Q3 0.5502 (n=4268), Q4 0.6123 (n=4269)
- Grocery seed 44 text: partial Spearman=0.01974524322164632 ; relation quartile mean r_attn=Q1 0.3377 (n=4269), Q2 0.3386 (n=4268), Q3 0.3365 (n=4268), Q4 0.3187 (n=4269)
- Grocery seed 44 visual: partial Spearman=0.09226389926047482 ; relation quartile mean r_attn=Q1 0.4709 (n=4269), Q2 0.4542 (n=4268), Q3 0.5029 (n=4268), Q4 0.5388 (n=4269)

## 13. Relation Intervention

Normal, Relation-Off, and fixed-seed Relation-Shuffle interventions are evaluated for V3 checkpoints; interventions affect the attention bias only. Deltas below are relative to Normal; mean |ΔA| compares each intervention's per-modality mean attention matrix with Normal.

- Movies seed 42 off: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=7.36e-08/2.52e-06.
- Movies seed 42 shuffle: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=7.45e-08/2.58e-06.
- Movies seed 43 off: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=1.03e-07/1.12e-06.
- Movies seed 43 shuffle: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=1.01e-07/1.13e-06.
- Movies seed 44 off: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=1.68e-07/3.17e-06.
- Movies seed 44 shuffle: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=1.66e-07/3.24e-06.
- Grocery seed 42 off: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=6.01e-07/2.10e-06.
- Grocery seed 42 shuffle: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=5.95e-07/2.03e-06.
- Grocery seed 43 off: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=5.49e-08/9.58e-07.
- Grocery seed 43 shuffle: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=4.94e-08/9.75e-07.
- Grocery seed 44 off: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=1.89e-06/5.22e-06.
- Grocery seed 44 shuffle: Δtest_acc=+0.0000000, Δtest_macro-F1=+0.0000000; mean |ΔA| Text/Visual=1.91e-06/5.30e-06.

## 14. Shallow Design

V3 and V4 are compared on matched seeds in the main tables and efficiency section. Per-checkpoint gate values, attention entropy, order profiles, and mean interaction matrices are in the mechanism JSON/CSV artifacts.

## 15. Overall Assessment

NC-only interim assessment (all 30 NC runs complete; LP remains pending). The paired deltas below are descriptive across three seeds and do not establish significance.
- V0: reference model; its NC test Accuracy and Macro-F1 are the paired baselines in Section 5.
- V1: Movies ΔAcc=+0.0030, ΔMacro-F1=-0.0032; Grocery ΔAcc=+0.0023, ΔMacro-F1=+0.0080; mean max |hop gate|=0.1781.
- V2: Movies ΔAcc=+0.0058, ΔMacro-F1=+0.0057; Grocery ΔAcc=+0.0030, ΔMacro-F1=+0.0089; mean max |hop gate|=0.1846.
- V3: Movies ΔAcc=-0.0001, ΔMacro-F1=-0.0015; Grocery ΔAcc=+0.0004, ΔMacro-F1=+0.0044; mean max |hop gate|=0.1861, mean max |active relation-bias gate|=0.1126.
- V4: Movies ΔAcc=+0.0066, ΔMacro-F1=+0.0049; Grocery ΔAcc=-0.0001, ΔMacro-F1=+0.0051; mean max |hop gate|=0.1741, mean max |active relation-bias gate|=0.1227.
- Mechanism/stability: all completed NC runs have finite logs and all hop gates are nonzero; V3/V4 active relation-bias gates are also nonzero. However, V3 Relation-Off/Shuffle changed the mean attention matrix by at most 5.30e-06 MAE and changed neither test Accuracy nor Macro-F1 (maximum absolute metric shift 0.00e+00); this is weak functional evidence for a consequential relation-bias effect.
- Efficiency/story: Section 8 reports all variants. V3 versus V4 shows whether the added interaction layer buys NC performance at added cost, but the LP comparison is still needed before judging the requested shallow-design story or making a full-grid recommendation.

## 16. Recommendation

D. insufficient evidence — the requested run grid is incomplete; no scientific conclusion is assigned to missing cells.
