# CoSI-MAG Final Benchmark Results

统计日期：2026-09-22

## 1. 实验设置

- 模型：`cosi_mag_final`，`ablation=full`
- Run seeds：42、43、44
- Split seed：42；同一数据集的三个 run 共用同一份 split
- NC 协议：`unified_full_graph_nc_v1`
- LP 协议：`unified_sampled_lp_v1`
- 汇总方式：3 runs 的均值 ± population standard deviation
- 表中分类与链路预测指标均以百分数表示

冻结配置 SHA-256：

| 配置 | SHA-256 |
| --- | --- |
| `configs/model/cosi_mag_final.yaml` | `caea19cfb4a24529fb9242c7c0aa8fda7ffdc37bb54220eee8c7c1e236a4fd36` |
| `configs/task/nc.yaml` | `8e7e5b580f2aea3112fdd20076f57ea16c143ec57730abd5ed094634c9068e75` |
| `configs/task/lp.yaml` | `180cc515bfd1f95ccb76e1eae661e1b35a6984ead760e5ddfa10ea7a6ba3dfcb` |

## 2. 主结果

### 2.1 节点分类

| Dataset | Highest Valid Acc | Valid Macro-F1 | Test Acc | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Movies | 57.51 ± 0.26 | 49.96 ± 0.87 | 56.50 ± 0.44 | 50.15 ± 0.69 |
| Toys | 80.43 ± 0.03 | 77.93 ± 0.58 | 79.37 ± 0.23 | 76.82 ± 0.84 |
| Grocery | 83.91 ± 0.16 | 78.47 ± 0.71 | 83.56 ± 0.05 | 76.08 ± 0.32 |
| ele-fashion | 88.23 ± 0.18 | 76.72 ± 0.51 | 88.21 ± 0.14 | 77.46 ± 0.56 |
| Reddit-S | 96.37 ± 0.05 | 93.24 ± 0.13 | 96.39 ± 0.10 | 92.39 ± 0.21 |

### 2.2 链路预测

| Dataset | Highest Valid MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sports-copurchase | 40.64 ± 0.30 | 37.42 ± 0.24 | 21.54 ± 0.18 | 42.96 ± 0.26 | 72.92 ± 0.50 |
| cloth-copurchase | 31.96 ± 0.21 | 28.48 ± 0.14 | 15.70 ± 0.16 | 31.25 ± 0.21 | 55.88 ± 0.26 |

## 3. 逐 seed 结果

### 3.1 节点分类

| Dataset | Seed | Best epoch | Valid Acc | Valid Macro-F1 | Test Acc | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Movies | 42 | 74 | 57.65 | 50.07 | 56.70 | 49.22 |
| Movies | 43 | 59 | 57.14 | 48.84 | 56.91 | 50.36 |
| Movies | 44 | 98 | 57.74 | 50.98 | 55.89 | 50.87 |
| Toys | 42 | 41 | 80.38 | 77.13 | 79.05 | 75.68 |
| Toys | 43 | 68 | 80.45 | 78.15 | 79.44 | 77.09 |
| Toys | 44 | 65 | 80.45 | 78.50 | 79.61 | 77.70 |
| Grocery | 42 | 113 | 84.01 | 78.60 | 83.51 | 76.27 |
| Grocery | 43 | 96 | 83.69 | 77.54 | 83.54 | 75.63 |
| Grocery | 44 | 96 | 84.04 | 79.26 | 83.63 | 76.35 |
| ele-fashion | 42 | 96 | 87.98 | 76.16 | 88.02 | 76.77 |
| ele-fashion | 43 | 184 | 88.40 | 77.39 | 88.35 | 78.13 |
| ele-fashion | 44 | 145 | 88.31 | 76.60 | 88.25 | 77.47 |
| Reddit-S | 42 | 123 | 96.45 | 93.38 | 96.51 | 92.44 |
| Reddit-S | 43 | 81 | 96.32 | 93.27 | 96.41 | 92.61 |
| Reddit-S | 44 | 110 | 96.35 | 93.06 | 96.26 | 92.11 |

### 3.2 链路预测

| Dataset | Seed | Best epoch | Valid MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sports-copurchase | 42 | 18 | 40.55 | 37.38 | 21.58 | 43.00 | 72.30 |
| sports-copurchase | 43 | 62 | 41.05 | 37.73 | 21.74 | 43.26 | 73.53 |
| sports-copurchase | 44 | 44 | 40.32 | 37.14 | 21.31 | 42.62 | 72.94 |
| cloth-copurchase | 42 | 56 | 32.16 | 28.63 | 15.92 | 31.47 | 55.76 |
| cloth-copurchase | 43 | 62 | 32.04 | 28.51 | 15.63 | 31.33 | 56.24 |
| cloth-copurchase | 44 | 64 | 31.67 | 28.30 | 15.54 | 30.96 | 55.63 |

## 4. 运行资源

`runtime` 是每个 dataset 三个 runs 的整个进程时长，包含训练、验证、最终测试和结果导出。峰值显存来自 `torch.cuda.max_memory_allocated`。

| Task | Dataset | Device | Runtime | Peak GPU memory | Parameters |
| --- | --- | --- | ---: | ---: | ---: |
| NC | Movies | `cuda:0` | 1 min 31 s | 2.61 GiB | 1,399,660 |
| NC | Toys | `cuda:0` | 59 s | 2.70 GiB | 1,399,146 |
| NC | Grocery | `cuda:0` | 1 min 19 s | 2.52 GiB | 1,399,660 |
| NC | ele-fashion | `cuda:0` | 6 min 7 s | 11.66 GiB | 1,266,532 |
| NC | Reddit-S | `cuda:0` | 1 min 38 s | 3.24 GiB | 1,399,660 |
| LP | sports-copurchase | `cuda:1` | 4 h 0 min 48 s | 4.61 GiB | 1,395,417 |
| LP | cloth-copurchase | `cuda:0` | 21 h 2 min 8 s | 7.23 GiB | 1,395,417 |

各运行组记录的 runtime 总和为 25 h 14 min 29 s。NC 参数量包含分类头；LP 参数量包含共享投影与预测头，因此会随任务或类别数变化。

## 5. 完整性与数值审计

审计结果：7/7 dataset 运行组通过，21/21 checkpoint 通过。

- 每组均存在 `complete.marker`、`results.json`、`metrics.json`、resolved config、训练日志和三个 run checkpoint。
- `best.pt` 均正确指向 `best_run3.pt`。
- checkpoint seeds 均依次为 42、43、44，任务、dataset、选择指标和最佳 epoch 元数据一致。
- 21 个 checkpoint 的模型、任务头和投影层张量均为有限值。
- 当前 7 组 `main.log` / `train.log` 未发现非有限 train loss 或 validation metric。
- launcher 的完成状态审计对 7 组均返回 `PASS`。
- 最终 sports-copurchase 结果为 Test MRR 37.42 ± 0.24，不包含此前异常运行产生的 NaN loss 或 100% 假验证指标。

运行来源存在两个 Git commit：Movies、Toys、Grocery、ele-fashion、Reddit-S 和 sports-copurchase 的 manifest 记录为 `e5c479c093bd8f7f6277820c99a5e3124aafd7b2`；cloth-copurchase 记录为 `c11aa1416b159f851cd3f71b78e25de2725ae6bc`。两批运行记录的模型与 NC/LP task 配置 SHA-256 完全一致，汇总时按相同冻结配置处理。

## 6. 原始结果位置

- NC：[`outputs/cosi_mag_final_benchmark/nc/`](../outputs/cosi_mag_final_benchmark/nc/)
- LP：[`outputs/cosi_mag_final_benchmark/lp/`](../outputs/cosi_mag_final_benchmark/lp/)
- 统一训练评估协议：[`docs/unified_training_evaluation_protocol.md`](unified_training_evaluation_protocol.md)
- Benchmark launcher 说明：[`docs/cosi_mag_final_benchmark_launcher.md`](cosi_mag_final_benchmark_launcher.md)

主结果表直接来自各运行组的 `results.json`；逐 seed 指标和最佳 epoch 来自 `best_run1.pt`、`best_run2.pt`、`best_run3.pt`。
