# LP Benchmark Results

F1-B1 的最终汇总见 [mopf_f1_final_benchmark_results.md](mopf_f1_final_benchmark_results.md)。
本文件保留 F1-B fresh LP 结果；sports 为 formal，cloth 为 quasi-held-out，二者不混合
平均，且所有 Test 指标均为 descriptive only。

## Final F1-B authoritative results

本节是当前冻结协议下的正式结果。结果生成日期为 `2026-09-14`，对应
method freeze SHA `4ddbd6918ceebadc25eed2694e1b463f9aac87f4`，正式
`configs/model/mopf.yaml` SHA256 为
`1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a`。

- 统一协议：`unified_sampled_lp_v1`。
- Seeds：`42, 43, 44`。
- 训练：sampled link prediction；评估：full-graph exact inference。
- checkpoint 只按 validation MRR 选择；Test 指标仅作描述，不参与选择或调参。
- 数值单位为百分比，格式为 `mean ± population std`。
- F1-B 48/48 个计划 run 已完成；`failed=0`、`missing=0`、`incomplete=0`、`duplicate=0`。

正式 sports LP 与 cloth quasi-held-out LP 严格分开。F1-B 主表先保留冻结版
MoPF 的结果；历史 sports baseline 的 producing SHA 虽未记录，但 F1-C 已在
文档末尾完成 behavior-equivalence certification，并据此生成独立的最终
formal external comparison。

## Formal sports-copurchase LP

当前 F1-B formal sports LP 完成了 MoPF 的 3 个 seed：

| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|
| mopf | **40.5950 ± 0.2371** | **37.4791 ± 0.3360** | **21.5611 ± 0.3899** | **43.0091 ± 0.4423** | **73.0441 ± 0.3769** |

F1-A 曾因历史 source provenance 不完整将外部 sports baseline 标为
`RERUN_REQUIRED`；F1-C 已按行为等价规则完成复核。上表仍保留 F1-B 的 MoPF
主结果，包含外部 baseline 的最终 formal comparison 见下方 F1-C 表；不把
历史 internal `map_mag_v3` 或旧版 `mopf` 计入外部排名。

## Quasi-held-out cloth-copurchase LP

cloth 是未参与 U1/U2/U3 选择的 quasi-held-out extension，不得与 formal
sports LP 合并平均。该表包含 MoPF 和 9 个 external baseline，各运行
3 个 seed。

| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|
| mlp | 21.2479 ± 0.2152 | 18.9668 ± 0.2017 | 9.0236 ± 0.1510 | 19.1235 ± 0.2124 | 39.4065 ± 0.4626 |
| gcn | 23.2875 ± 0.1624 | 20.9900 ± 0.1172 | 11.0351 ± 0.1021 | 21.9186 ± 0.1359 | 41.0537 ± 0.7002 |
| sage | 27.6355 ± 0.1767 | 24.6818 ± 0.1284 | 12.7090 ± 0.0962 | 26.1489 ± 0.1188 | 50.3954 ± 0.2247 |
| mmgcn | 29.5478 ± 0.0803 | 26.2543 ± 0.0398 | 13.7166 ± 0.0316 | 28.3123 ± 0.1046 | 53.3201 ± 0.1659 |
| mgat | 28.4310 ± 0.1654 | 25.2154 ± 0.1343 | 13.1594 ± 0.0821 | 26.9263 ± 0.2334 | 51.2648 ± 0.2997 |
| dip | 29.1923 ± 0.8636 | 25.7576 ± 0.8178 | 13.2880 ± 0.6200 | 27.8389 ± 1.0326 | 52.7814 ± 1.3475 |
| dgf | 29.3892 ± 0.1720 | 25.9258 ± 0.1337 | 13.7932 ± 0.1358 | 27.8040 ± 0.1399 | 51.9602 ± 0.2377 |
| dmgc | 26.4087 ± 0.1246 | 23.3502 ± 0.1081 | 11.5177 ± 0.0666 | 24.6966 ± 0.0846 | 48.6312 ± 0.2445 |
| lgmrec | 24.7819 ± 0.1950 | 22.1648 ± 0.1988 | 11.4688 ± 0.2151 | 23.3268 ± 0.2761 | 44.2805 ± 0.2865 |
| mopf | **31.1114 ± 0.4978** | **27.9103 ± 0.3502** | **15.4326 ± 0.1858** | **30.4852 ± 0.4481** | **54.5655 ± 0.7671** |

### Cloth result analysis

MoPF 在 cloth 的五个报告指标上均为最高均值：Test MRR、Hits@1、Hits@3、
Hits@10 以及 Val MRR。相对于各指标第二名，MoPF 的均值优势分别为
`+1.6560`、`+1.6394`、`+2.1729`、`+1.2454` 和 `+1.5636` 个百分点。
这些是 quasi-held-out 泛化结果，不应被表述为 formal sports benchmark
上的结论。

## F1-B artifacts and audit

- [F1-B completion audit](../outputs/f1_final_execution/f1b_completion.json)
- [F1-B machine-readable summary](../outputs/f1_final_execution/f1b_summary.json)
- [F1-B Markdown summary](../outputs/f1_final_execution/f1b_summary.md)
- [F1-B execution plan](../outputs/f1_final_execution/f1b_plan.csv)
- [F1-B formal output root](../outputs/f1_final_execution/)

每个 fresh run 均通过 required marker、checkpoint、metrics、resolved config
和 provenance 检查；所有已加载结果均为 finite，且 validation-only selection
guard 与 frozen provenance audit 均通过。

## Historical sports-copurchase benchmark（F1-C source audit）

以下内容保留早期 `outputs/lp_benchmark/` 的 11 模型 benchmark，便于复核
历史 source。F1-C 已证明其中 9 个 external baseline 在当前 frozen formal
protocol 下行为等价，因此它们进入最终 external comparison；`map_mag_v3`
仍是 historical internal model，旧版 `mopf` 仍是 pre-U3-B1 reference，二者
不进入 formal main table。该历史实验的数值同样为百分比，格式为
`mean ± population std`。

### 历史实验设置

- 数据集：`sports-copurchase`
- 模型：11 个模型（MAP-MAG 仅保留 `map_mag_v3`，另含 `mopf`）
- Seeds：`42, 43, 44`
- LP 协议：`unified_sampled_lp_v1`
- 训练方式：sampled link prediction
- Neighbor sampling：两跳 `[5, 5]`
- Training negative：每条正边 1 个 filtered negative
- LP projection dimension：`128`
- Inference：full-graph exact inference

| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|
| mlp | 27.0313 ± 0.3456 | 24.9860 ± 0.3515 | 11.7553 ± 0.3136 | 26.3206 ± 0.3361 | 55.3863 ± 0.7039 |
| gcn | 34.2503 ± 0.1534 | 32.0852 ± 0.1421 | 17.7701 ± 0.1798 | 35.4431 ± 0.0548 | 64.1105 ± 0.2315 |
| sage | 35.1167 ± 0.2361 | 32.5921 ± 0.2369 | 17.1010 ± 0.1557 | 36.4062 ± 0.2627 | 69.0535 ± 0.4137 |
| mmgcn | 37.6103 ± 0.2629 | 34.6295 ± 0.2975 | 18.4223 ± 0.3270 | 39.4854 ± 0.2589 | **72.4543 ± 0.3007** |
| mgat | 35.3487 ± 0.1798 | 32.6118 ± 0.1141 | 17.1660 ± 0.0504 | 36.2895 ± 0.2576 | 69.1132 ± 0.6134 |
| dip | 38.1210 ± 0.1706 | 35.1427 ± 0.2141 | 18.9025 ± 0.3937 | 40.2855 ± 0.3614 | 72.2806 ± 0.5189 |
| dgf | 37.3452 ± 0.4084 | 34.1880 ± 0.3075 | 18.5746 ± 0.2540 | 38.5053 ± 0.4546 | 70.4372 ± 0.4431 |
| dmgc | 35.6925 ± 1.7976 | 32.9330 ± 1.7656 | 17.5420 ± 1.5683 | 37.3346 ± 2.4270 | 68.0503 ± 2.3289 |
| lgmrec | 35.6221 ± 0.0433 | 32.9235 ± 0.0323 | 18.1630 ± 0.0832 | 36.7074 ± 0.1397 | 66.2515 ± 0.0874 |
| map_mag_v3 | 39.8790 ± 0.2599 | 36.8362 ± 0.2255 | 20.9642 ± 0.1827 | 42.4683 ± 0.3127 | 71.8583 ± 0.4453 |
| mopf | **40.1393 ± 0.2652** | **37.1078 ± 0.2304** | **21.4061 ± 0.1400** | **42.5823 ± 0.3009** | 72.1888 ± 0.4717 |

历史结果路径：`../outputs/lp_benchmark/sports-copurchase/`。

F1-C 最终 formal sports comparison（9 个 certified external baseline + F1-B
冻结版 MoPF）位于：

- `outputs/f1_final_execution/tables/f1_lp_sports_final_comparison.csv`
- `outputs/f1_final_execution/tables/f1_lp_sports_final_comparison_paper_table.csv`
- `outputs/f1_final_execution/f1c_historical_baseline_certification.csv`
- `docs/mopf_f1c_historical_baseline_certification.md`

相对于 strongest eligible baseline，MoPF 的 mean delta 为：Val MRR `+2.4740`
pp、Test MRR `+2.3364` pp、Test Hits@1 `+2.6586` pp、Test Hits@3 `+2.7237`
pp、Test Hits@10 `+0.5898` pp。5/5 指标为最高均值；Test 指标仍仅作
descriptive reporting，不代表显著性或全局 SOTA。
