# LP Benchmark Results

## 实验设置

- 数据集：`sports-copurchase`
- 模型：13 个实现模型
- 追加模型：`mopf`（详见文末新增结果与训练检查）
- Seed：`42`
- LP 协议：`unified_sampled_lp_v1`
- 训练方式：sampled link prediction
- Neighbor sampling：两跳 `[5, 5]`
- Training negative：每条正边 1 个 filtered negative
- LP projection dimension：`128`
- Inference：full-graph exact inference
- 表中数值：百分比（`results.json` 中的 `[0, 1]` 数值乘以 100）
- 本实验为单 seed，因此没有跨 seed 方差；表中记录 seed 42 的结果
- 原始结果目录：`outputs/full_benchmark/lp/sports-copurchase/`

本次共完成 `1 × 13 = 13` 个 LP benchmark jobs。结果由各模型目录下的
`seed42_runs1/results.json` 汇总而来。`map_mag_v3` 的 LP 运行使用
`map_mag_v3_lp` 配置 preset。

在原 benchmark 内容基础上追加 MoPF 的 1 个 `num_runs=1` job；MoPF 结果来自
本报告新增的 2026-09-09 输出目录。

## 结果总览

| 指标 | 最优模型 | 结果 |
|---|---|---:|
| Validation MRR | `map_mag_v3` | **40.4279** |
| Test MRR | `map_mag_v3` | **37.3798** |
| Test Hits@1 | `map_mag_v3` | **21.2840** |
| Test Hits@3 | `map_mag_v3` | **43.0252** |
| Test Hits@10 | `dip` | **73.6295** |

## 详细结果

| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|
| mlp | 27.0150 | 25.0620 | 11.8889 | 26.3224 | 55.3711 |
| gcn | 33.0181 | 30.8291 | 17.1331 | 33.3921 | 61.7833 |
| sage | 35.0372 | 32.5724 | 16.9807 | 36.5301 | 69.0375 |
| mmgcn | 38.1364 | 35.0268 | 18.7582 | 40.0663 | 72.7768 |
| mgat | 35.4767 | 32.5305 | 16.8685 | 36.4419 | 69.4064 |
| dip | 39.1335 | 35.7429 | 19.2500 | 41.2584 | **73.6295** |
| dgf | 37.6213 | 34.4381 | 18.6887 | 38.9651 | 70.9459 |
| dmgc | 16.6131 | 15.7282 | 5.6772 | 14.9253 | 37.2170 |
| lgmrec | 35.3194 | 32.7383 | 18.2584 | 36.2975 | 65.3703 |
| map_mag | 37.7181 | 35.0568 | 19.7231 | 39.7268 | 69.7351 |
| map_mag_v1 | 37.7321 | 34.8233 | 19.1564 | 39.7616 | 70.0612 |
| map_mag_v2 | 38.5795 | 35.7234 | 20.2015 | 40.6009 | 70.8123 |
| map_mag_v3 | **40.4279** | **37.3798** | **21.2840** | **43.0252** | 73.6241 |
| mopf | 40.2610 | 37.0492 | 21.2226 | 42.5013 | 72.5390 |

## 结果路径

- [LP benchmark outputs](../outputs/full_benchmark/lp/sports-copurchase/)
- [Benchmark manifest](../outputs/full_benchmark/benchmark_manifest.json)

## MoPF 新增结果与训练检查

### 运行配置与数据规模

本节追加整理运行目录 `outputs/2026-09-09/14-53-42` 的完整日志、resolved
config 和 `results.json`。这是 `seed=42`、`num_runs=1` 的单次运行，设备为
`cuda:0`，模型与 decoder 共 `997,069` 个参数。

- 协议：`unified_sampled_lp_v1`，`training_mode=sampled`，Loader 为
  `LinkNeighborLoader`，`subgraph_type=bidirectional`。
- 图规模：`50,250` 个节点、`603,696` 条 message edges；输入为
  `X=(50,250,1024)`，text/visual 各 `512` 维。
- 固定 LP split：train `356,202` 条正边；validation `45,557` 条正边、每条
  `150` 个负例；test `37,413` 条正边、每条 `150` 个负例。
- 训练负采样为 global filtered、每条正边 `1` 个负例；正 message-edge
  masking 后端为 `global_eid`。
- 配置中的 `num_neighbors=[5,5]` 在运行时按 `num_layers=3` 实际解析为
  `[5,5,5]`；日志明确记录了该三跳采样设置。
- 优化器为 Adam，`lr=1e-3`，`weight_decay=1e-5`，`batch_size=2048`，
  `grad_clip=1.0`；最多 150 epoch，`patience=10`，每 2 个 epoch 做一次验证。
- 训练使用 sampled LP；评估使用 `inference_mode=full`，预加载完整节点嵌入
  后做 exact full-graph inference，嵌入形状为 `(50,250,128)`。

### MoPF 结果

以下结果直接来自 `outputs/2026-09-09/14-53-42/results.json`，已将 `[0,1]`
数值转换为百分比：

| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|
| mopf (seed 42) | 40.2610 | 37.0492 | 21.2226 | 42.5013 | 72.5390 |

日志中的四舍五入值为 Val MRR `40.26`、Test MRR `37.05`、Hits@1/3/10
`21.22 / 42.50 / 72.54`，与 `results.json` 一致。MoPF 在本次单 seed 运行中
没有超过表中 `map_mag_v3` 的 MRR、Hits@1 或 Hits@3，也没有超过 `dip` 的
Hits@10；这只是性能比较，不影响训练协议判断。

### 训练过程与稳定性判断

结论：从日志可观测信号看，这次 MoPF 的 sports-copurchase LP 训练正常完成，
没有发现数值爆炸、异常中断或协议执行错误。

1. 训练从 epoch 1 持续到 epoch 66，并在 `patience=10` 后正常 early stopping；
   最佳 validation MRR 为 `40.26`（epoch 46），随后恢复 best checkpoint 做最终
   full-graph exact test。
2. Train loss 从 `0.1104` 降至 `0.0261`，整体下降趋势清晰，仅有很小的后期
   随机波动；没有出现发散或塌缩。
3. Validation MRR 在 `36.34–40.26` 范围内波动，并在 epoch 46 达峰；采样训练下
   验证指标的这种波动是正常的，且 early-stopping 计数与配置一致。
4. 每个 epoch 的 positive message edges removed 约为 `3.81–3.83×10^5`，
   没有出现边掩码数量异常归零或失控；训练负采样、邻居采样和 full-graph
   inference 模式都被日志明确打印。
5. 日志中没有 `NaN`、`Inf`、`non-finite`、`OOM`、`Traceback`、`Exception`、
   `ERROR` 或 `failed`；结果文件正常写出，单 seed 的各项 std 均为零。

需要保留的实验性限制是：这是单 seed，无法估计跨 seed 方差；Validation MRR
`40.2610` 到 Test MRR `37.0492` 存在约 `3.21` 个百分点的泛化差距，但当前日志
没有证据表明这是优化或数值故障。正式性能结论应在相同 split、negative
sampling 和 decoder 协议下补充更多 seeds。

原始结果文件：`outputs/2026-09-09/14-53-42/results.json`。
