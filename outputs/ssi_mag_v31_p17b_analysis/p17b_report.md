# SSI-MAG-V3.1 P1.7b Full NC Benchmark and Mechanism Report

本报告基于冻结 provenance 下的 15 个 SSI-MAG-V3.1 Full NC best checkpoints。训练使用 `cuda:0`；post-hoc analyzer 使用 CPU、eval/no-grad。Test 只作为 validation-selected checkpoint 的最终描述性结果，未用于调参、选模型、选 seed 或 guardrail。

## 1. Integrity and protocol

- Branch/commit: `V3` / `e913a4eae834fdb8cb92709b1f488a05b050e295`
- Model/config SHA256: `358f35e5274f227ddd7699396454fcb2eda4cfe8598779a5be8ed85215534da4` / `6bec458b786b5e08d0d04363785a2b82c9b6b73ab381dbe176f118dfbad08ac1`
- Task: NC only; model `ssi_mag_v31`; variant `full`; protocol `unified_full_graph_nc_v1`; datasets Movies/Toys/Grocery/ele-fashion/Reddit-S; seeds 42/43/44.
- Formal jobs: `15/15` loaded, finite `15/15`; node-order guard all passed; attention simplex failures `0`.
- LP jobs: 0; ablation jobs: 0; hyperparameter search: 0; auxiliary loss added: 0.

## 2. Performance evidence

| Dataset | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Best epoch | Trained epochs | Runtime s | Peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.5763 ± 0.0006 | 0.5017 ± 0.0010 | 0.5631 ± 0.0020 | 0.4976 ± 0.0087 | 55.0000 | 85.0000 | 31.2034 | 3688.0225 |
| Toys | 0.8033 ± 0.0018 | 0.7725 ± 0.0006 | 0.8008 ± 0.0046 | 0.7672 ± 0.0042 | 48.3333 | 78.3333 | 33.0388 | 3971.8398 |
| Grocery | 0.8372 ± 0.0008 | 0.7633 ± 0.0107 | 0.8319 ± 0.0031 | 0.7582 ± 0.0095 | 72.3333 | 102.3333 | 38.6580 | 3608.3506 |
| ele-fashion | 0.8819 ± 0.0005 | 0.7656 ± 0.0009 | 0.8817 ± 0.0005 | 0.7724 ± 0.0033 | 118.3333 | 148.3333 | 176.0223 | 17545.1450 |
| Reddit-S | 0.9663 ± 0.0012 | 0.9293 ± 0.0054 | 0.9673 ± 0.0019 | 0.9303 ± 0.0037 | 72.3333 | 102.3333 | 36.9246 | 4374.5928 |
| ALL | 0.8130 ± 0.1304 | 0.7465 ± 0.1377 | 0.8090 ± 0.1352 | 0.7451 ± 0.1394 | 73.2667 | 103.2667 | 63.1694 | 6637.5901 |

Population std is over the three fixed seeds. No run was removed.

## 3. Paired V3 versus V3.1 performance (descriptive)

| Dataset | Δ Val Acc | Δ Val Macro-F1 | Δ Test Acc | Δ Test Macro-F1 | V3.1 better / V3 better / ties (Val Acc) |
|---|---:|---:|---:|---:|---:|
| Movies | 0.2699 pp | 0.8149 pp | -0.2599 pp | 0.2439 pp | 3 / 0 / 0 |
| Toys | -0.1289 pp | -0.5352 pp | 0.1369 pp | -0.5862 pp | 0 / 3 / 0 |
| Grocery | -0.0586 pp | -0.5947 pp | 0.2147 pp | 0.6169 pp | 1 / 1 / 1 |
| ele-fashion | 0.0614 pp | 0.3470 pp | -0.0398 pp | 0.1705 pp | 3 / 0 / 0 |
| Reddit-S | 0.0734 pp | 0.2725 pp | -0.1153 pp | -0.0676 pp | 2 / 1 / 0 |
| ALL | 0.0435 pp | 0.0609 pp | -0.0127 pp | 0.0755 pp | 9 / 5 / 1 |

Paired provenance was verified against same datasets/seeds, NC, full-graph mode, protocol and validation-Accuracy selection. The five-dataset unweighted validation mean deltas are Accuracy `+0.0435 pp` and Macro-F1 `+0.0609 pp`; this comparison is descriptive only.

## 4. R1 relation modulation

| Modality | β mean | score std | dynamic-logit std | weight CV | operator relative-L1 | c mean |
|---|---:|---:|---:|---:|---:|---:|
| text | 0.0528 | 0.0266 | 0.0300 | 0.0014 | 0.0011 | 0.0026 |
| visual | 0.0525 | 0.0231 | 0.0253 | 0.0012 | 0.0012 | 0.0026 |

- Same-edge text/visual compatibility correlations and absolute-difference quantiles are in `p17b_relation_diagnostics.csv` (the `text_vs_visual` rows).
- R1 is non-degenerate in some modality/dataset combinations, while beta remains close to 0.05 in 26/30 modality-run rows and Movies text has near-negligible weight CV; these are reported flags, not automatic failure claims.
- Operator perturbation is measured after the same `gcn_norm` and self-loop policy against raw unit physical topology; no edge support was rewired.

## 5. R2 adaptive semantic reference

| Modality | α mean (k1/k2/k3) | α std (k1/k2/k3) | p mean / std | term_p std | d mean (k1/k2/k3) |
|---|---:|---:|---:|---:|---:|
| text | 0.1242 / 0.1241 / 0.1155 | 0.0103 / 0.0103 / 0.0097 | 1.2110 / 2.0488 | 0.0951 | 0.0910 / 0.1027 / 0.1156 |
| visual | 0.1106 / 0.1102 / 0.1042 | 0.0125 / 0.0125 / 0.0117 | -0.5365 / 2.4234 | 0.1197 | 0.1054 / 0.1192 / 0.1370 |

- Alpha shows finite node/hop dispersion rather than exact fixed 0.1, with no reported alpha 0/1 saturation flags. R2 p-tail flags occur in 30 hop rows and are reported as scale observations; no repair or retuning was performed.

## 6. Stage-II context interaction and signed filtering

| Modality | Nodewise norm entropy | Node heterogeneity | Query diversity | g_delta | g_int | D norm | S_tilde cosine | eta std | negative eta fraction | effective order mean / node std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text | 0.4870 | 0.4203 | 0.0600 | 0.1578 | 0.1160 | 15.1296 | 0.9849 | 0.0073 | 0.2369 | 1.5534 / 0.0111 |
| visual | 0.5853 | 0.3557 | 0.0729 | 0.1605 | 0.1226 | 14.7077 | 0.9848 | 0.0033 | 0.1529 | 1.6645 / 0.0065 |

- Attention simplex validation passed for all 30 modality-run matrices. Entropy/nonuniformity, node heterogeneity and query diversity are reported descriptively; no global-mixture claim is inferred from one scalar.
- Signed filtering retained negative eta mass and node-varying effective order. Covariance attribution is valid because eta variance was nonzero in the reported rows; relation residual small-amplitude flags are explicitly named `R3_relation_residual_small_amplitude` and are not automatic invalidity judgments.

## 7. Frozen functional sensitivity

| Intervention | Mean fused relative-L2 | Mean flip rate | Mean Val Acc Δ | Mean Val Macro-F1 Δ |
|---|---:|---:|---:|---:|
| relation_off | 0.0006 | 0.0001 | 0.0000 | 0.0000 |
| interaction_off | 0.1829 | 0.0459 | -0.0099 | -0.0097 |

`relation=off` changes fused embeddings very little on average; `interaction=off` produces materially larger fused relative-L2 changes and prediction flips. These are frozen functional sensitivity diagnostics using the same saved NC head, not retrained causal ablations.

## 8. Flags and guardrail

- Automatic flags: `172` total; see `p17b_flags.csv`. Main groups: R1 beta-near-init 26, R1 relation-weight-CV-negligible 3, R2 p-scale-tail 30, R3 relation-residual-small-amplitude 113.
- Guardrail: **PASS_GUARDRAIL**. Validation-only trigger thresholds were not crossed: five-dataset mean deltas are above -0.5 pp and simultaneous validation Accuracy+Macro-F1 declines occur on 2/5 datasets.
- This status does not declare a final paper model and does not suppress any mechanism flags.

## 9. Boundary and open questions

- No LP, retrained ablation, hyperparameter search, architecture change, auxiliary loss, test-based architecture selection, or seed deletion was performed.
- P1.7c control plan was generated because the guardrail passed, but controls were not implemented, trained, or launched.
- Any future architecture decision must separately audit the flagged R1/R2/R3 quantities; this report proposes no unvalidated repair.
