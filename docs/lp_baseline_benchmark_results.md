# LP Benchmark Results

当前文档记录 `sports-copurchase` 的 LP benchmark 结果。`cloth-copurchase` 将在实验完成后补充到同一文档。

## 实验协议

- Task：Link Prediction（LP）
- Protocol：`unified_sampled_lp_v1`
- 数据集：`sports-copurchase`
- Baseline：GCN、SAGE、MMGCN、MGAT、DGF、DMGC、LGMRec、DiP
- 自有模型：`cosi_mag_final`
- Runs：3 runs，seeds 42、43、44；split/base seed 为 42
- Neighbor sampling：`task.num_neighbors=[5, 5, 5]`
- Checkpoint selection：每个 run 按 validation MRR 选择最佳 checkpoint，再报告对应 test 指标
- 表中数值：百分比，格式为 `mean ± population std`

Baseline launcher summary 显示 8/8 个模型均正常完成，失败数为 0。结果由各模型的 `results.json` 汇总得到。

## Sports-copurchase 结果

表格按 Test MRR 从高到低排列。

| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|
| **CoSI-MAG (ours)** | **40.6393 ± 0.3046** | **37.4191 ± 0.2420** | **21.5397 ± 0.1778** | **42.9610 ± 0.2602** | 72.9247 ± 0.5021 |
| DiP | 39.0227 ± 0.3021 | 35.9931 ± 0.3211 | 19.6269 ± 0.2721 | 41.4606 ± 0.3757 | 73.3364 ± 0.5368 |
| MMGCN | 38.4891 ± 0.2012 | 35.4425 ± 0.2145 | 19.1262 ± 0.2868 | 40.4369 ± 0.2244 | **73.4361 ± 0.3746** |
| DGF | 37.9203 ± 0.2101 | 34.7595 ± 0.2880 | 19.0504 ± 0.2262 | 39.2324 ± 0.4558 | 71.1776 ± 0.2861 |
| GCN | 36.3845 ± 0.9659 | 33.9779 ± 0.8972 | 19.1636 ± 0.5354 | 38.0438 ± 1.3251 | 67.2413 ± 1.7781 |
| SAGE | 35.5443 ± 0.2435 | 32.8774 ± 0.1700 | 17.3371 ± 0.0931 | 36.8686 ± 0.3243 | 69.3511 ± 0.3559 |
| DMGC | 35.6493 ± 1.2308 | 32.7484 ± 1.2682 | 17.0948 ± 1.3787 | 37.0281 ± 1.6732 | 68.8914 ± 0.4393 |
| MGAT | 35.5936 ± 0.2416 | 32.6101 ± 0.1605 | 17.1803 ± 0.0734 | 36.4089 ± 0.3473 | 68.8121 ± 0.5588 |
| LGMRec | 34.6917 ± 0.5467 | 32.1925 ± 0.4336 | 17.6311 ± 0.3039 | 35.6543 ± 0.6519 | 65.1752 ± 0.6890 |

### 简要比较

- Baseline 中 Test MRR 最高的是 DiP：`35.9931 ± 0.3211`。
- `cosi_mag_final` 的 Test MRR 为 `37.4191 ± 0.2420`，相对最佳 baseline DiP 高 **1.4260 个百分点**。
- `cosi_mag_final` 在 Test Hits@1 和 Test Hits@3 上最高；Test Hits@10 最高的是 MMGCN（`73.4361 ± 0.3746`）。
- 以上比较是描述性统计，不构成显著性检验结论。

## 结果路径

Baseline 结果根目录：`outputs/lp_baseline_benchmark/sports-copurchase/`

- Baseline 汇总：[summaries/sports-copurchase.json](../outputs/lp_baseline_benchmark/summaries/sports-copurchase.json)
- Baseline manifest：[manifests/sports-copurchase.json](../outputs/lp_baseline_benchmark/manifests/sports-copurchase.json)
- 自有模型结果：[cosi_mag_final results.json](../outputs/cosi_mag_final_benchmark/lp/sports-copurchase/runs_42_43_44/results.json)

各模型的详细输出位于：

```text
outputs/lp_baseline_benchmark/sports-copurchase/<model>/runs_42_43_44/
```

## Cloth-copurchase

待实验完成后补充。
