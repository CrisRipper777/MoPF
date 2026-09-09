# Grocery-NC: K=2 validation selection

## 结论

Grocery 三 seed 的 validation-only 选择结果支持 K=2：K=2 的 mean Validation Accuracy 为 **83.78 ± 0.20%**，高于现有正式 K=3 三 seed 结果的 **83.65 ± 0.11%**。因此本轮记录：**Grocery selected K=2 by validation performance**。

这只是 Grocery 的选择记录；`configs/model/mopf.yaml` 的全局默认 `max_order=3` 未修改，MoPF 其它数据集仍使用默认 K=3。

## 实验协议

- 数据集/任务/模型：Grocery / NC / MoPF
- seeds：42、43、44（`seed=42, num_runs=3`）
- 图：full graph
- optimizer：AdamW，`lr=1e-3`，`weight_decay=1e-4`
- 训练：300 epochs，patience=30
- checkpoint selection：Validation Accuracy
- edge weight：`separate_cos`
- K=2 仅通过 Hydra override `model.max_order=2`；resolved config 同时显示 `model.num_layers=2`
- split、classifier、CE loss、filter、semantic graph 和其它训练设置保持不变
- 汇总标准差采用 population std（ddof=0）

K=2 的独立输出位于 [`outputs/grocery_k_selection/k2_separate`](../outputs/grocery_k_selection/k2_separate)。resolved config 可核对上述 override；K=3 的现有正式三 seed 输出位于 [`outputs/2026-09-09/14-47-21`](../outputs/2026-09-09/14-47-21)。

## seed42 exploratory diagnosis

定向诊断阶段的 seed42 结果用于提出候选 K，不用于替代三 seed 选择：

| configuration | Val Acc | Test Acc | Test Macro-F1 |
|---|---:|---:|---:|
| K=3 + separate_cos | 83.66% | 83.05% | 75.71% |
| K=2 + separate_cos | 84.07% | 83.51% | 76.20% |

来源为 [`docs/mopf_targeted_diagnosis.md`](mopf_targeted_diagnosis.md) 中的 Grocery seed42 diagnosis。由于 GPU 训练存在运行间随机性，后续正式三 seed run 的 seed42 不要求与这次 exploratory run 数值逐位相同。

## three-seed validation-based hyperparameter selection

K=2 本轮正式验证的每个 seed：

| seed | Best Val Acc | Test Acc | Test Macro-F1 |
|---:|---:|---:|---:|
| 42 | 83.98% | 83.28% | 75.70% |
| 43 | 83.84% | 82.99% | 75.58% |
| 44 | 83.51% | 83.34% | 75.82% |
| **mean ± std** | **83.78 ± 0.20%** | **83.20 ± 0.15%** | **75.70 ± 0.10%** |

对照的现有正式 K=3 三 seed 结果：

| seed | Best Val Acc | Test Acc | Test Macro-F1 |
|---:|---:|---:|---:|
| 42 | 83.78% | 83.07% | 75.42% |
| 43 | 83.66% | 83.34% | 75.63% |
| 44 | 83.51% | 82.93% | 74.81% |
| **mean ± std** | **83.65 ± 0.11%** | **83.11 ± 0.17%** | **75.29 ± 0.35%** |

K 的选择只看 mean Validation Accuracy：K=2 比 K=3 高 **0.13 个百分点**。Test Accuracy 和 Test Macro-F1 只在选择完成后报告，未参与 K 的选择。

## final test reporting

选择完成后，K=2 的 final test reporting 为 Test Accuracy **83.20 ± 0.15%**、Test Macro-F1 **75.70 ± 0.10%**。这些 test 数值不构成调参依据，也不改变全局 MoPF 默认 K=3。

