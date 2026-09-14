# MoPF-vNext F1 Final Benchmark Results

本文件是 F1-B1 与 F1-C 的最终汇总。F1-C 对历史 external baseline 执行了
behavior-equivalence certification，并只进行了审计、重分类、指标聚合、表格
生成和文档更新；没有重新训练、架构修改、超参数调优、评估器修改或数据划分
修改。结果生成日期：`2026-09-14`。

## 1. Protocol

- Method-freeze SHA：`4ddbd6918ceebadc25eed2694e1b463f9aac87f4`。
- Formal `configs/model/mopf.yaml` SHA256：
  `1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a`。
- NC：`Movies`、`Toys`、`Grocery`、`ele-fashion`、`Reddit-S`，每个数据集
  使用 seeds `42, 43, 44`，协议为 `unified_full_graph_nc_v1`。
- LP：sports 使用 `unified_sampled_lp_v1` 的 formal split；cloth 使用同一
  冻结协议下的 quasi-held-out split。sports 与 cloth 不计算混合平均。
- 所有表均报告 3-seed mean 与 population standard deviation，单位为百分比。
- checkpoint 只按 validation 指标选择：NC 为 validation accuracy，LP 为
  validation MRR。Test 指标全部是 descriptive only，不参与选择、调参或
  protocol decisions。
- F1-B0 预先记录的 reporting policy 已执行：NC 主表使用 F1-A 审计通过的
  U3-B1 `REUSE_EXACT`；F1-B fresh NC 作为 reproducibility check。sports 和
  cloth 使用 F1-B fresh execution 结果。

Completion audit 结果为 48/48：`failed=0`、`missing=0`、`incomplete=0`、
`duplicate=0`。F1-C 又逐 seed 审计了 135 个历史 external NC records 与 27 个
历史 external sports records；它们均通过行为等价认证并进入下面的正式比较。
历史输出未记录 producing execution SHA，故这些结果是
`CERTIFIED_BEHAVIOR_EQUIVALENT_REUSE`，不是伪造 SHA 的 `REUSE_EXACT`。

## 2. NC main results

### 2.1 Primary table: U3-B1 `REUSE_EXACT`

| Dataset | Val Accuracy | Val Macro-F1 | Test Accuracy (descriptive) | Test Macro-F1 (descriptive) |
|---|---:|---:|---:|---:|
| Movies | 57.3585 ± 0.3325 | 49.7301 ± 0.4627 | 55.9120 ± 0.6791 | 49.7164 ± 0.9877 |
| Toys | 80.4784 ± 0.1685 | 77.4748 ± 0.2070 | 79.4556 ± 0.0932 | 76.4430 ± 0.2556 |
| Grocery | 83.8458 ± 0.1810 | 78.0387 ± 0.2118 | 83.5725 ± 0.3538 | 76.1701 ± 0.6276 |
| ele-fashion | 88.0638 ± 0.1066 | 76.3623 ± 0.3231 | 88.1362 ± 0.1293 | 77.0687 ± 0.4738 |
| Reddit-S | 96.2147 ± 0.0646 | 93.0371 ± 0.0357 | 96.3511 ± 0.2283 | 92.0161 ± 0.4779 |

### 2.2 Fresh F1-B NC reproducibility check

| Dataset | Val Accuracy | Val Macro-F1 | Test Accuracy (descriptive) | Test Macro-F1 (descriptive) |
|---|---:|---:|---:|---:|
| Movies | 57.6785 ± 0.3472 | 49.9429 ± 0.9556 | 55.8721 ± 0.1979 | 49.9310 ± 0.3709 |
| Toys | 80.4301 ± 0.0395 | 77.2573 ± 0.0358 | 80.1401 ± 0.4784 | 77.0464 ± 0.2522 |
| Grocery | 84.0996 ± 0.1805 | 77.6816 ± 0.6317 | 82.8990 ± 0.0414 | 75.1120 ± 1.0243 |
| ele-fashion | 88.0434 ± 0.0823 | 76.2246 ± 0.4404 | 88.1248 ± 0.1390 | 76.9414 ± 0.5838 |
| Reddit-S | 96.4349 ± 0.3252 | 92.7554 ± 0.4708 | 96.5083 ± 0.3704 | 92.5926 ± 0.8549 |

Fresh-minus-reuse mean differences were retained without choosing the numerically
higher source. The mean differences, in percentage points, are available in
[`f1_nc_fresh_vs_reuse.csv`](../outputs/f1_final_execution/f1_nc_fresh_vs_reuse.csv),
and all 15 seed-level differences are in
[`f1_nc_per_seed_fresh_vs_reuse.csv`](../outputs/f1_final_execution/f1_nc_per_seed_fresh_vs_reuse.csv).
The largest absolute seed-level difference is 2.1734 percentage points (Grocery,
seed 43, Test Macro-F1); this is an audit finding, not a basis for source selection.

F1-A 原先因历史 producing SHA 等字段未记录而将 135 个 external NC cells 标为
`RERUN_REQUIRED`。F1-C 按行为等价规则复核后，将它们全部重分类为
`CERTIFIED_BEHAVIOR_EQUIVALENT_REUSE`：split、features、baseline model source/config、
训练协议、有效 evaluator 语义和 validation-only checkpoint selection 均等价。
因此下面的 final comparison 可以报告 strongest eligible baseline，但不做
post-hoc model selection。

相对于每个数据集的 strongest eligible baseline，MoPF 的 test mean delta 为：

| Dataset | Test Accuracy strongest baseline | MoPF delta (pp) | Test Macro-F1 strongest baseline | MoPF delta (pp) |
|---|---:|---:|---:|---:|
| Movies | lgmrec 55.5222 | +0.3898 | lgmrec 49.2728 | +0.4437 |
| Toys | lgmrec 79.1818 | +0.2738 | lgmrec 76.5190 | -0.0760 |
| Grocery | dip 83.6506 | -0.0781 | dip 75.8024 | +0.3677 |
| ele-fashion | dip 88.0430 | +0.0932 | dip 77.3445 | -0.2757 |
| Reddit-S | dip 96.4454 | -0.0944 | dip 92.3912 | -0.3751 |

MoPF is best on 4/10 of these NC test comparisons and top-2 on 6/10. These are
direct table comparisons only; they do not imply significance or global SOTA.

Machine-readable outputs:

- [`f1_nc_full_results.csv`](../outputs/f1_final_execution/tables/f1_nc_full_results.csv)
- [`f1_nc_paper_table.csv`](../outputs/f1_final_execution/tables/f1_nc_paper_table.csv)
- [`f1_nc_final_comparison.csv`](../outputs/f1_final_execution/tables/f1_nc_final_comparison.csv)
- [`f1_nc_final_comparison_paper_table.csv`](../outputs/f1_final_execution/tables/f1_nc_final_comparison_paper_table.csv)
- [`f1c_comparative_statistics.csv`](../outputs/f1_final_execution/f1c_comparative_statistics.csv)

## 3. Sports formal LP results

The formal sports result is MoPF on `sports-copurchase`, with seeds `42, 43, 44`.

| Model | Val MRR | Test MRR (descriptive) | Test Hits@1 (descriptive) | Test Hits@3 (descriptive) | Test Hits@10 (descriptive) |
|---|---:|---:|---:|---:|---:|
| MoPF | 40.5950 ± 0.2371 | 37.4791 ± 0.3360 | 21.5611 ± 0.3899 | 43.0091 ± 0.4423 | 73.0441 ± 0.3769 |

F1-A 原先将 27 个 external sports LP cells 标为 `RERUN_REQUIRED`。F1-C 完成行为
等价认证后，9 个 external models 均可正式复用。相对于 strongest eligible
baseline，MoPF 的 sports mean delta 为：Val MRR `+2.4740` pp、Test MRR
`+2.3364` pp、Test Hits@1 `+2.6586` pp、Test Hits@3 `+2.7237` pp、Test Hits@10
`+0.5898` pp；5/5 指标为最高均值。Test 仍全部是 descriptive only。

历史 `map_mag_v3` 与旧版 historical `mopf` 保持 internal/old reference，未进入
formal external main table。

[`f1_lp_sports_full_results.csv`](../outputs/f1_final_execution/tables/f1_lp_sports_full_results.csv)
and
[`f1_lp_sports_paper_table.csv`](../outputs/f1_final_execution/tables/f1_lp_sports_paper_table.csv)
contain the F1-B MoPF-only table. The F1-C final comparison is in
[`f1_lp_sports_final_comparison.csv`](../outputs/f1_final_execution/tables/f1_lp_sports_final_comparison.csv)
and
[`f1_lp_sports_final_comparison_paper_table.csv`](../outputs/f1_final_execution/tables/f1_lp_sports_final_comparison_paper_table.csv).

## 4. Cloth quasi-held-out LP results

Cloth is a quasi-held-out extension, not a second formal sports benchmark. It is
reported separately and is never mixed with sports in a formal average.

| Model | Val MRR | Test MRR (descriptive) | Test Hits@1 (descriptive) | Test Hits@3 (descriptive) | Test Hits@10 (descriptive) |
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
| MoPF | **31.1114 ± 0.4978** | **27.9103 ± 0.3502** | **15.4326 ± 0.1858** | **30.4852 ± 0.4481** | **54.5655 ± 0.7671** |

All ten cloth models have complete, comparable F1-B provenance. Relative to the
strongest eligible baseline for each metric, MoPF's absolute mean deltas are:

| Metric | Strongest eligible baseline | MoPF delta (percentage points) |
|---|---|---:|
| Val MRR | mmgcn | +1.5635 |
| Test MRR (descriptive) | mmgcn | +1.6560 |
| Test Hits@1 (descriptive) | dgf | +1.6394 |
| Test Hits@3 (descriptive) | mmgcn | +2.1729 |
| Test Hits@10 (descriptive) | mmgcn | +1.2453 |

Within this eligible quasi-held-out cloth table, MoPF is best on all five reported
metrics and is top-2 on all five. This is a statement about this table only; it is
not a claim of global SOTA or statistical significance.

[`f1_lp_cloth_quasiheldout_full_results.csv`](../outputs/f1_final_execution/tables/f1_lp_cloth_quasiheldout_full_results.csv)
and
[`f1_lp_cloth_quasiheldout_paper_table.csv`](../outputs/f1_final_execution/tables/f1_lp_cloth_quasiheldout_paper_table.csv)
contain the full and paper-facing versions.

## 5. Stability

All reported groups contain exactly seeds `42, 43, 44`; no unfavorable seed was
removed. Completion audit found no failed, missing, incomplete, or duplicate final
run. The final formal logs contain no detected traceback, CUDA OOM, non-finite,
overflow, or equivalent failure marker, and all aggregate metrics are finite.

The reported standard deviations are population standard deviations over the three
seeds. The primary NC largest standard deviation is Movies Test Macro-F1
(`0.9877` percentage points); Grocery Test Macro-F1 is `0.6276` percentage points.
For the fresh NC reproducibility check, Grocery Test Macro-F1 is `1.0243` percentage
points. These are reported as observed stability, not hidden or filtered.

On cloth, the largest baseline variability is dip's Test Hits@10 standard deviation
(`1.3475` percentage points). MoPF's corresponding value is `0.7671` percentage
points; this does not justify a significance claim. Fresh-vs-reuse NC differences
were checked both per seed and by mean, with no post-hoc source or model selection.

## 6. Efficiency

[`f1_efficiency.csv`](../outputs/f1_final_execution/f1_efficiency.csv) records the
actual comparable items available in F1-B logs: parameter count and the observed
final-attempt log span. Parameter counts were recorded for every fresh run. Peak GPU
memory was not recorded in the F1-B logs and is therefore left blank rather than
estimated. The log spans are observational run spans, not a controlled throughput
benchmark, and should not be interpreted as a hardware-normalized speed ranking.

Representative parameter counts:

| Scope | Models / datasets | Parameter count |
|---|---|---:|
| NC | MoPF on Movies, Reddit-S | 1,001,832 |
| NC | MoPF on Toys | 1,001,318 |
| NC | MoPF on Grocery | 999,763 |
| NC | MoPF on ele-fashion | 868,704 |
| LP | MoPF | 997,589 |
| LP | mlp / gcn / sage / mmgcn | 460,673 / 526,977 / 920,193 / 871,041 |
| LP | mgat / dip | 16,573,657 / 7,800,809 |
| LP | dgf / dmgc / lgmrec | 173,057 / 346,243 / 254,977 |

## 7. Provenance

The F1-B audit contains 63 passing records: 48 fresh F1-B runs plus 15 F1-A
`REUSE_EXACT` U3-B1 NC runs. F1-C adds 162 certified historical external seed
records: 135 NC and 27 sports. Every historical record was checked for seed,
dataset, model, resolved configuration, freeze target, source snapshot, evaluator,
split, and validation-only checkpoint selection. Sports records are marked
`formal`; cloth records are marked `quasi-held-out`.

Fresh F1-B records use execution SHA
`f958ee77cdd4b74dec8fd50c6b3621386d2f04f8`; the reused U3-B1 records use producing
SHA `615832d7c8afd04fba2958e535ae25c30f1c2f9a`. All records match the method-freeze
SHA and formal model configuration hash above. The machine-readable audit is in
[`f1b1_provenance_audit.csv`](../outputs/f1_final_execution/f1b1_provenance_audit.csv)
and [`f1b1_provenance_audit.json`](../outputs/f1_final_execution/f1b1_provenance_audit.json).

The historical source manifest does not contain an original producing execution
SHA; this remains an explicit provenance limitation. It does not invalidate the
behavior-equivalence certification, because the old source snapshot and effective
protocol checks are recorded in
[`f1c_historical_baseline_certification.csv`](../outputs/f1_final_execution/f1c_historical_baseline_certification.csv)
and its JSON companion.

## 8. Limitations / exceptions

- Historical external NC and sports values are now formal-eligible by F1-C
  behavior-equivalence certification. Their original producing execution SHA was
  not recorded, so the decision is certified reuse rather than exact SHA reuse.
- Cloth is quasi-held-out and must not be described as the formal sports benchmark
  or combined with sports in an average.
- NC has two retained sources by policy: U3-B1 `REUSE_EXACT` is primary, while the
  fresh F1-B run is reproducibility evidence. The fresh source was not selected
  because it was higher or lower.
- Test metrics are descriptive only. No significance tests were run, so “best” is
  used only for directly comparable tables and does not imply significance.
- Peak GPU memory was unavailable in the logs; efficiency reporting is limited to
  parameter counts and observed log spans.
- F1-B1/F1-C made no architecture change and performed no hyperparameter tuning.

## F2 gate

**F1-C CLOSED. PASS — Proceed F2.** Formal NC is complete, all 135 external NC
seed records are equivalence-certified, formal sports LP is complete, and all 27
external sports seed records are equivalence-certified. F2 was not started here;
this document is the hard stop for final result aggregation.
