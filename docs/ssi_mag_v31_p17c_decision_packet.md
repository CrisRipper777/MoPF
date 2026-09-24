# SSI-MAG-V3.1 P1.7c Sequential Development Attribution Decision Packet

本文件是 development-control 事实包，不是最终论文 ablation，也不对最终模型作选择。所有 performance、mechanism、frozen sensitivity 证据分开解释。

## 1. Scope and provenance

- Branch: `V3`; formal benchmark provenance commit: `2642cd39fb458ba4b2ece3fbf025251b333254fa`.
- Model/config: `ssi_mag_v31_controls` / `configs/model/ssi_mag_v31_controls.yaml`; model SHA256 `482d76137caffb2031ff3c391bf7f64538157b086be83c732120d1249b699e8c`; config SHA256 `6636ed5e845e4d21ce0a646ccef82da6a684c3d7c4da08778d7a6f3f41a15f1a`.
- Protocol: `unified_full_graph_nc_v1`; task `nc`; training device `cuda:0`; validation Accuracy selects checkpoint/early stopping; test is descriptive only.
- Jobs: `5 datasets × 3 seeds × 2 controls = 30/30`; controls `B=r2_only`, `AB=r1_r2`; LP jobs `0`.
- Historical V3 and frozen V3.1 checkpoints were reused; `ssi_mag_v3.py` and `ssi_mag_v31.py` were not modified.
- Post-hoc analyzer loads saved best checkpoint plus saved NC head in eval/no-grad. GPU analysis was attempted, but the largest node-level intervention exceeded 23.7 GiB; the final complete analysis used CPU and did not retrain.

## 2. Sequential definitions

- `Delta_R2 = B - V3`: old R1 + new R2 + old reference residual versus historical V3.
- `Delta_R1 = AB - B`: new R1 versus old R1 under the same new R2 and old reference residual.
- `Delta_remove_ref = V3.1 - AB`: reference-residual removal under new R1 + new R2.

## 3. Performance evidence

Values are seed mean ± population standard deviation, shown as percentages. The development comparisons below use validation metrics; test deltas are included only descriptively.

| Dataset | B Val Acc | B Val Macro-F1 | B Test Acc | B Test Macro-F1 | AB Val Acc | AB Val Macro-F1 | AB Test Acc | AB Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 57.718 ± 0.353 | 50.599 ± 0.385 | 55.402 ± 0.480 | 49.657 ± 1.630 | 57.468 ± 0.385 | 50.547 ± 1.004 | 56.202 ± 0.163 | 49.910 ± 0.772 |
| Toys | 80.656 ± 0.257 | 77.964 ± 0.662 | 79.898 ± 0.681 | 77.209 ± 0.814 | 80.438 ± 0.211 | 77.403 ± 0.407 | 80.060 ± 0.532 | 76.990 ± 0.135 |
| Grocery | 84.090 ± 0.108 | 76.693 ± 0.706 | 83.084 ± 0.455 | 75.853 ± 0.693 | 83.826 ± 0.156 | 76.399 ± 1.552 | 83.026 ± 0.470 | 75.728 ± 1.628 |
| ele-fashion | 88.091 ± 0.144 | 76.373 ± 0.367 | 88.166 ± 0.103 | 76.991 ± 0.479 | 88.299 ± 0.107 | 77.073 ± 0.287 | 88.338 ± 0.148 | 77.743 ± 0.370 |
| Reddit-S | 96.718 ± 0.142 | 93.156 ± 0.485 | 96.823 ± 0.118 | 93.164 ± 0.161 | 96.624 ± 0.146 | 92.814 ± 0.486 | 96.770 ± 0.233 | 92.894 ± 0.350 |

Best epoch is retained per run in `p17c_performance.csv`; model selection never reads test metrics.

| Comparison | 5-dataset unweighted validation delta Acc | validation delta Macro-F1 | Val Acc better/worse/tie | Val F1 better/worse/tie | descriptive test Acc delta | descriptive test F1 delta |
|---|---:|---:|---:|---:|---:|---:|
| Delta_R2 | +0.1980 pp | +0.3685 pp | 8/6/1 | 9/6/0 | -0.2358 pp | +0.1357 pp |
| Delta_R1 | -0.1235 pp | -0.1098 pp | 5/10/0 | 7/8/0 | +0.2045 pp | +0.0779 pp |
| Delta_remove_ref | -0.0311 pp | -0.1978 pp | 5/10/0 | 6/9/0 | +0.0187 pp | -0.1381 pp |

### Validation paired dataset means

| Dataset | Delta_R2 Acc / F1 (pp) | Delta_R1 Acc / F1 (pp) | Delta_remove_ref Acc / F1 (pp) |
|---|---:|---:|---:|
| Movies | +0.3599 / +1.2391 | -0.2499 / -0.0514 | +0.1600 / -0.3728 |
| Toys | +0.2013 / +0.1807 | -0.2174 / -0.5614 | -0.1127 / -0.1545 |
| Grocery | +0.3123 / -0.2366 | -0.2635 / -0.2943 | -0.1074 / -0.0638 |
| ele-fashion | -0.0409 / +0.1584 | +0.2080 / +0.7001 | -0.1057 / -0.5115 |
| Reddit-S | +0.1573 / +0.5007 | -0.0944 / -0.3418 | +0.0105 / +0.1136 |

Interpretation boundary: these are descriptive development-control comparisons, not retrained causal ablations and not a basis for test-driven selection.

## 4. R2 mechanism evidence: B versus V3

The analyzer reports alpha mean/std, corrected range ratio `(q90-q10)/(abs(mean(alpha))+eps)`, raw `p`, normalized `term_p`, and `d` per modality/hop. Raw p scale is descriptive only.

| Variant | Modality | alpha mean h1/h2/h3 | alpha std h1/h2/h3 | corrected range ratio h1/h2/h3 | term_p node std h1/h2/h3 |
|---|---|---|---|---|---|
| V3 | text | 0.104445/0.104963/0.097461 | 0.001771/0.001773/0.001657 | 0.044027/0.043924/0.044547 | 0.018074/0.018074/0.018074 |
| V3 | visual | 0.101088/0.100804/0.097104 | 0.002818/0.002822/0.002684 | 0.071634/0.071874/0.072327 | 0.030846/0.030846/0.030846 |
| B | text | 0.113808/0.114443/0.106336 | 0.008075/0.008124/0.007530 | 0.187872/0.187686/0.189714 | 0.079946/0.079946/0.079946 |
| B | visual | 0.103818/0.103395/0.099784 | 0.012470/0.012473/0.011870 | 0.310374/0.310486/0.312733 | 0.128653/0.128653/0.128653 |

- Alpha saturation: for both V3 and B, every run/hop/modality has fraction `<0.05 = 0` and fraction `>0.95 = 0`; no 0/1 saturation was observed.
- B has visibly larger alpha node dispersion than V3 (mean alpha std across run rows: `0.010090` versus `0.002254`) and larger term_p node std (`0.104299` versus `0.024460`). The corrected range ratio is now `(q90-q10)/(abs(mean(alpha))+eps)`; its run-row mean is `0.249811` for B versus `0.058055` for V3.
- Cross-seed mean Pearson/Spearman for alpha: V3 text `0.563/0.584`, visual `0.603/0.620`; B text `0.411/0.413`, visual `0.726/0.706`. For term_p: V3 text `0.419/0.459`, visual `0.599/0.618`; B text `0.406/0.407`, visual `0.725/0.704`. These are stability descriptors, not performance criteria.
- Fact-level answer for R2: B-V3 is accompanied by stronger learned node-level alpha/term_p dispersion without saturation. The performance delta is reported separately above.

## 5. R1 mechanism evidence: AB versus B

Raw old/new scorer values are not ranked directly because the mechanisms have different score semantics and scales. Comparable diagnostics are `log(w)` modulation, normalized operator perturbation, c, and frozen relation-off sensitivity.

| Variant | Modality | beta | score std | log(w) std | attenuation ratio | c mean | operator relative-L1 |
|---|---|---:|---:|---:|---:|---:|---:|
| B | text | 0.052599 | 0.614139 | 0.032483 | 0.052599 | 0.035451 | 0.016965 |
| B | visual | 0.052204 | 0.504660 | 0.026525 | 0.052204 | 0.034125 | 0.015099 |
| AB | text | 0.053112 | 0.031606 | 0.001739 | 0.053112 | 0.002535 | 0.001157 |
| AB | visual | 0.052310 | 0.024989 | 0.001322 | 0.052310 | 0.002278 | 0.001066 |

| Relation-off frozen intervention | Prediction flip rate | fused relative-L2 | mean cosine | Val Acc delta (pp) | Val Macro-F1 delta (pp) |
|---|---:|---:|---:|---:|---:|
| B | 0.001360 | 0.006054 | 0.999978 | -0.0102 | -0.0128 |
| AB | 0.000133 | 0.000523 | 1.000000 | -0.0120 | -0.0090 |

- B old R1 has nontrivial normalized operator perturbation (relative-L1 mean text/visual `0.016965/0.015099`) and larger relation-off sensitivity.
- AB new R1 has finite scorer variation, but log(w) and normalized operator perturbation are much smaller (relative-L1 `0.001157/0.001066`), and relation-off frozen sensitivity is correspondingly small. This is a reported functional-use difference, not an architectural repair decision.
- The attenuation ratio is approximately beta for both controls (`~0.052–0.053`), so the new scorer’s small score scale remains visible after beta modulation; no ratio optimization was performed.

## 6. Reference-residual evidence: AB versus V3.1

For AB, `reference_residual = s_ref * centered([1, alpha_1, alpha_2, alpha_3])`. For fixed modality/hop, gamma + DeltaGamma is node-constant. The CSV reports per-hop node statistics and `Cov(term, eta)/Var(eta)` for content/reference/relation; a numeric assertion checks that the three-term sum is 1 within `1e-4` whenever Var(eta)>eps.

| Modality | Hop | reference std | reference abs mean | reference covariance contribution | content contribution | relation contribution | 3-term sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| text | 0 | 0.000670 | 0.085395 | 0.209224 | 0.727716 | 0.063060 | 1.000000 |
| text | 1 | 0.000243 | 0.028145 | 0.016132 | 0.983154 | 0.000714 | 1.000000 |
| text | 2 | 0.000245 | 0.028126 | 0.014040 | 0.981062 | 0.004897 | 1.000000 |
| text | 3 | 0.000182 | 0.029124 | -0.006978 | 1.002895 | 0.004083 | 1.000000 |
| visual | 0 | 0.001212 | 0.071138 | 0.098295 | 0.903091 | -0.001386 | 1.000000 |
| visual | 1 | 0.000444 | 0.023434 | -0.011888 | 1.011521 | 0.000367 | 1.000000 |
| visual | 2 | 0.000434 | 0.023545 | 0.038549 | 0.954469 | 0.006982 | 1.000000 |
| visual | 3 | 0.000334 | 0.024159 | 0.006508 | 0.980765 | 0.012727 | 1.000000 |

| Reference-off frozen intervention on AB | Prediction flip rate | fused relative-L2 | mean cosine | Val Acc delta (pp) | Val Macro-F1 delta (pp) |
|---|---:|---:|---:|---:|---:|
| AB reference off | 0.009641 | 0.040836 | 0.999146 | -0.1083 | +0.0047 |

- Reference-off changes fused embeddings more than relation-off on average (`relative-L2 0.040836` versus B `0.006054` and AB `0.000523`), with mean prediction flip rate `0.009641`. This is frozen functional sensitivity, not causal necessity.
- The reference covariance contribution is hop/modality dependent: it is most visible at hop 0 and smaller at later hops on average, while content explains most eta variance at hops 1–3. The exact per-run values are in the CSV.
- `Delta_remove_ref = V3.1 - AB` is the performance-side description of removing this branch under new R1/new R2; it must not be conflated with the frozen intervention.

## 7. Automatic flags and non-conclusions

- FLAG (reported, not repaired): new-R1 `log(w)` and normalized operator perturbation are much smaller than old-R1 B; relation-off frozen sensitivity is correspondingly small.
- FLAG (reported, not repaired): learned beta remains near initialization (`0.05`) in both controls; scorer scale and propagation effect must be read together.
- NO FLAG: alpha saturation at 0 or 1 was not observed.
- NO FLAG: R2 alpha/term_p node dispersion is not collapsed to the V3 level; B is more heterogeneous.
- No statement here labels any mechanism a final failure or proves causal necessity; no repair, retuning, auxiliary loss, new R1, `w=exp(a)`, or `c=mean|a|` path was introduced.

## 8. Reproducibility and boundaries

- Formal lock: `outputs/ssi_mag_v31_p17c_controls/provenance.lock.json`; manifest: `outputs/ssi_mag_v31_p17c_controls/manifest.json`.
- Analysis outputs: `outputs/ssi_mag_v31_p17c_analysis/`.
- No LP jobs, paper ablation, hyperparameter search, auxiliary loss, test-based model selection, baseline rerun, historical V3 modification, or frozen V3.1 modification was performed.
- The decision packet reports the three requested facts—`B-V3`, `AB-B`, and `V3.1-AB`—without selecting a final model.
