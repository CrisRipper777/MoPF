# CoSI-MAG VNext 第三轮最小结构验证

## 范围与口径

本轮从基准提交 `3396f603fdd4594b1e10ced5452f1b083edd4dcd` 建立独立分支 `exp/cosi-mag-vnext-r3-minimal-design`。原 VNext 实现与 CoSI Final 冻结代码未修改；新增变体位于 `src/models/cosi_mag_vnext_r3.py` 及三个薄入口模块中。

所有新训练使用 `unified_full_graph_nc_v1`、已有 seed-42 split、seed 42/43/44、AdamW、lr `1e-3`、weight decay `1e-4`、最多 300 epochs、full graph、每 epoch 验证、patience 30、`early_stop_min_epoch=30`、`early_stop_min_delta=1e-4`、grad clip 1.0、无 scheduler。统一以 validation Accuracy 选择 checkpoint，`evaluate_test=false`。加载数据后立即把 test labels 设为 `-100`；训练、指标和类别集合只读取 train/validation labels。新模型 Macro-F1 的标签集合为 train/validation 中出现的类别，本数据三个 split 中均覆盖全部配置类别。

固定 checkpoint 分析读取原 pilot manifest 中九个 `best_val_accuracy` checkpoint，并逐项核对 checkpoint SHA、seed-42 split SHA、保存 epoch 与 validation Accuracy/Macro-F1。原模型验证指标重算均与 checkpoint 相符。round2 的固定 checkpoint intervention 与 GCN/SAGE 结果作为已有验证数据引用，不重新训练；CoSI MAG Final 数字是历史 validation 参考，配置不同，不能视为本轮受控消融。

训练资源记录为训练/逐 epoch 验证及 checkpoint 序列化的墙钟时间，排除数据加载和训练后 checkpoint 重载核验。峰值显存为 CUDA max allocated，设备 `cuda:1`。参数量包含 NC 分类头。

## 变体定义

| 变体 | 传播算子 | 节点关系上下文 q | 其余结构 |
|---|---|---|---|
| VNext-U | 物理邻接均匀权重，经 GCN 对称归一化并加一个 self-loop | 保留 CRC、重新训练 | 保留原 Chebyshev bank、RCMI、结构语义残差和 late fusion |
| VNext-UQ | 同上 | 删除 CRC/q；重新训练无 q 条件 RCMI | 保留跨阶 attention、逐阶 eta、结构残差和 late fusion；条件 attention bias、q 输入 eta scorer、q 输入 semantic gate 移除 |
| VNext-UA | 同 VNext-U | 保留 CRC、重新训练 | 以 anchored multi-hop bank 替换 Chebyshev bank，其余与 U 相同 |

UA 的固定 anchor 为 `alpha=0.1`，不增加可训练参数：

```text
B0 = H0
Bk = alpha * H0 + (1 - alpha) * P_uniform * B(k-1),  k=1..3
Dk = Bk_interacted - B0_interacted
U  = sum_k eta_k * Dk
Y  = H0 + sigmoid(Wg [H0, q] + bg) * U
```

初始化沿用 VNext：eta prior 对应 `[0.2, 0.0, 0.0]`，条件 eta scorer 最后一层权重/偏置为零；attention gate 初值 0.1；semantic gate 权重为零、偏置为 `logit(0.1)`。alpha 是固定配置值，不是参数。U 与 UA 的总可训练参数均为 1,282,347；UQ 为 1,258,913，减少 23,434 个参数（含分类头）。

## 新变体 validation 结果

下表为三个 seed 的均值 ± population standard deviation。逐 seed 的 Accuracy、Macro-F1、最佳 epoch、时间、显存、参数量、checkpoint SHA 和 split SHA 见机器结果。

| 数据集 | 变体 | Val Acc (%) | Val Macro-F1 (%) | 最佳 epoch 均值 | 时间均值 (s) | 峰值显存 (MiB) | 参数量 |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | U | 55.07 ± 0.26 | 46.58 ± 0.91 | 57.7 | 19.2 | 3,138.4 | 1,282,347 |
| Movies | UQ | 54.30 ± 0.54 | 46.26 ± 0.91 | 61.0 | 18.5 | 1,605.7 | 1,258,913 |
| Movies | UA | 55.60 ± 0.13 | 47.81 ± 1.52 | 62.7 | 26.0 | 3,138.8 | 1,282,347 |
| Grocery | U | 83.55 ± 0.23 | 76.74 ± 0.40 | 105.7 | 38.1 | 3,006.6 | 1,282,347 |
| Grocery | UQ | 83.25 ± 0.29 | 76.23 ± 0.53 | 74.7 | 23.0 | 1,634.7 | 1,258,913 |
| Grocery | UA | 83.76 ± 0.19 | 77.84 ± 0.30 | 89.7 | 41.5 | 3,006.6 | 1,282,347 |
| Reddit-S | U | 96.21 ± 0.08 | 92.33 ± 0.13 | 61.0 | 34.5 | 4,107.2 | 1,282,347 |
| Reddit-S | UQ | 96.10 ± 0.08 | 92.18 ± 0.41 | 109.0 | 32.8 | 1,539.9 | 1,258,913 |
| Reddit-S | UA | 96.21 ± 0.08 | 92.44 ± 0.09 | 89.3 | 45.9 | 4,107.2 | 1,282,347 |

已保存 validation confusion matrix 和逐 seed per-class F1。U 的验证集持续薄弱类别包括 Movies class 16（F1 0–0.043，support 38）、Grocery class 4（0.156–0.316，support 54）、Reddit-S class 13（0.429–0.511，support 31）。Movies seed 42 中 class 16 主要被预测为 1/10/2；Grocery class 4 主要混淆到 14/7/16；Reddit-S class 13 主要混淆到 15/6/11。它们对 Macro-F1 的影响大于对 Accuracy 的影响；此处是 validation confusion 的观察，不是对历史 CoSI Final 逐类别差异的因果解释。

## 与已有结果的比较

| 数据集 | 指标 | VNext 固定 checkpoint (R2) | CoSI MAG Final 历史参考 | U | UQ | UA | GCN | SAGE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | Acc | 55.09 ± 0.03 | 57.51 ± 0.26 | 55.07 ± 0.26 | 54.30 ± 0.54 | 55.60 ± 0.13 | 52.71 ± 0.16 | 52.65 ± 2.23 |
| Movies | Macro-F1 | 46.55 ± 1.53 | 49.96 ± 0.87 | 46.58 ± 0.91 | 46.26 ± 0.91 | 47.81 ± 1.52 | 43.86 ± 1.14 | 40.32 ± 11.63 |
| Grocery | Acc | 83.35 ± 0.37 | 83.91 ± 0.16 | 83.55 ± 0.23 | 83.25 ± 0.29 | 83.76 ± 0.19 | 80.65 ± 0.13 | 83.28 ± 0.17 |
| Grocery | Macro-F1 | 76.38 ± 0.69 | 78.47 ± 0.71 | 76.74 ± 0.40 | 76.23 ± 0.53 | 77.84 ± 0.30 | 72.53 ± 1.00 | 76.67 ± 0.36 |
| Reddit-S | Acc | 96.23 ± 0.08 | 96.37 ± 0.05 | 96.21 ± 0.08 | 96.10 ± 0.08 | 96.21 ± 0.08 | 94.25 ± 0.15 | 95.32 ± 0.08 |
| Reddit-S | Macro-F1 | 92.23 ± 0.15 | 93.24 ± 0.13 | 92.33 ± 0.13 | 92.18 ± 0.41 | 92.44 ± 0.09 | 90.60 ± 0.07 | 91.74 ± 0.14 |

新变体与 VNext 固定 checkpoint 共用 seed-42 split 和 validation 指标口径；但前者是本轮 fresh training，后者是原 checkpoint。CoSI Final 数字沿用 R2 已记录的历史参考。GCN/SAGE 使用各自标准 model config。模型 config 不同，CoSI Final/GCN/SAGE 只用于背景比较，不能解释为单因素架构消融。完整逐 seed paired delta 和来源记录见 `comparison_references.json`。

## 固定 checkpoint 机制诊断

### 投影前后的边相似度

每个 checkpoint 按有向物理边计算 cosine；原图为无向时正反边均计入。机器结果保存每个 dataset/seed/modality/表示的全分布分位数（0、1、5、25、50、75、95、99、100%）、节点邻域方差分布，以及同一节点不同邻居相似度的 RMS pairwise gap。表中是三个 seed 的节点/边统计再取均值，`gap/global std` 表示邻居间区分度相对全图边相似度标准差。

| 数据集 | 模态 | raw → pre-ReLU → post-ReLU 边 cosine 中位数 | post-ReLU 邻域方差均值 | post-ReLU gap/global std |
|---|---|---|---:|---:|
| Movies | Text | 0.974 → 0.725 → 0.800 | 0.00908 | 0.986 |
| Movies | Visual | 0.623 → 0.496 → 0.607 | 0.00690 | 0.922 |
| Grocery | Text | 0.977 → 0.478 → 0.592 | 0.01074 | 1.039 |
| Grocery | Visual | 0.630 → 0.377 → 0.526 | 0.01192 | 0.952 |
| Reddit-S | Text | 0.967 → 0.566 → 0.645 | 0.00589 | 0.871 |
| Reddit-S | Visual | 0.719 → 0.800 → 0.854 | 0.00644 | 0.665 |

投影显著改变模态相似性结构，但 post-ReLU 后同一节点不同邻居仍有可测区分度；Reddit-S Visual 的相对邻居差异低于其他组合。此现象本身不能说明它是否与分类错误相关。

### q、关系算子、attention 与结构残差

固定 checkpoint 上将 q 行按固定 seed `20261013` 打乱，传播 bank、学习到的 P、输入、模型和分类头均保持不变。下表是逐 seed validation 变化的均值：

| 数据集 | Logit RMS 差 | 预测变化率 | 置信度绝对变化 | Acc Δ (pp) | Macro-F1 Δ (pp) |
|---|---:|---:|---:|---:|---:|
| Movies | 0.0417 | 0.930% | 0.594% | -0.070 | -0.138 |
| Grocery | 0.0773 | 0.732% | 0.611% | -0.117 | -0.030 |
| Reddit-S | 0.0407 | 0.147% | 0.162% | -0.021 | +0.030 |

q 扰动仅改变很少的 validation 预测。与此同时，fresh-trained U 相对 UQ 的三 seed 均值在三个数据集上均略好（Macro-F1 +0.16 至 +0.51 pp）。这是小幅、方向一致的观察；3 个 seed 不足以支持显著性结论。

R2 中学习 P 与 uniform P 的相对 Frobenius 差异很小：

| 数据集 | `P_text` vs uniform | `P_visual` vs uniform | `P_text` vs `P_visual` |
|---|---:|---:|---:|
| Movies | 0.25% | 0.40% | 0.65% |
| Grocery | 0.24% | 0.25% | 0.43% |
| Reddit-S | 0.31% | 1.43% | 1.12% |

R2 固定 checkpoint 将 P 替换为 uniform 后，三数据集 Acc/Macro-F1 变化均不超过 0.12 pp；原 relation scorer 的中心化 edge-output RMS 为 Movies 0.0025、Grocery 0.0054、Reddit-S 0.0026，输出整体变化有限。U 仍保留 CRC 以生成并训练 q，因此这轮能判断 learned **propagation weights** 的价值，不能单独断言整个 CRC scorer 对 q 没有价值。

原 VNext bank 的结构响应、每阶 eta 和 attention/残差贡献如下；eta node std 与饱和率按模态和阶数分别统计，表格是三个 seed 均值。饱和定义为 `|eta| >= 0.45`，eta 有界于 `[-0.5,0.5]`。

| 数据集 | 模态 | `||B3-B1||F/||B1||F` | attention 差分 / 原始结构 RMS | eta1 std / 饱和 | eta2 std / 饱和 | eta3 std / 饱和 | gated U/H0 节点比例中位数 |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | Text | 0.761 | 0.024 | 0.054 / 0.03% | 0.120 / 0.66% | 0.115 / 0.65% | 0.105 |
| Movies | Visual | 0.978 | 0.065 | 0.026 / 8.68% | 0.075 / 8.68% | 0.076 / 6.54% | 0.376 |
| Grocery | Text | 1.088 | 0.042 | 0.044 / 6.90% | 0.110 / 1.71% | 0.110 / 1.16% | 0.321 |
| Grocery | Visual | 1.161 | 0.071 | 0.019 / 61.92% | 0.066 / 29.45% | 0.065 / 24.00% | 0.539 |
| Reddit-S | Text | 0.005 | 0.027 | 0.018 / 2.22% | 0.061 / 0.53% | 0.034 / 0.00% | 0.353 |
| Reddit-S | Visual | 0.016 | 0.046 | 0.026 / 1.18% | 0.071 / 1.18% | 0.043 / 0.20% | 0.191 |

R2 原 checkpoint 上仅关闭最终 gated residual，Acc/Macro-F1 分别下降：Movies `4.07/6.17 pp`、Grocery `8.71/7.77 pp`、Reddit-S `3.74/4.55 pp`。关闭跨阶 attention 更新的影响小得多：Acc `0.01–0.13 pp`、Macro-F1 约 `-0.28 至 +0.01 pp`；其差分贡献只有原始结构差异 RMS 的约 2.4–7.1%。所以高阶结构经 eta 加权后形成的最终残差很重要，但 attention 更新的直接增量偏小。Grocery Visual 的 eta1 饱和率较高；Chebyshev `B3-B1` 响应在数据集间差异也很大，提醒我们不应把当前未锚定高阶递推认定为稳定主干。

## 五个架构问题的结论

以下区分本轮观测事实与解释：

1. **Learned relation weights 是否值得保留？**
   - **已验证：** 学习到的传播算子与 uniform operator 的差异仅 0.24–1.43%；在九个原 checkpoint 上换成 uniform 几乎不改变 validation 指标。fresh-trained U 与旧 VNext 表现相近。
   - **建议：** 不保留当前可学习的边权传播分支作为默认主干，采用 uniform propagation。CRC 产生 q 的关系统计保留到 q 消融完成进一步复核为止；本轮没有单独关闭 scorer 同时保留 q 的干预。

2. **节点关系上下文 q 是否值得保留？**
   - **已验证：** 固定模型 shuffle q 后预测变化率很低；重新训练的 U 在三个数据集 Macro-F1 均略高于 UQ，提升 0.16–0.51 pp，Accuracy 提升 0.11–0.77 pp。
   - **建议：** 暂时保留 q，但将其视为小幅增益、低置信度机制；不应仅凭固定 checkpoint 的 q shuffle 认定 q 完全无效。

3. **Chebyshev 是否优于简单 anchored multi-hop？**
   - **已验证：** 参数量相同的 UA 相对 U，在 Movies/Grocery Macro-F1 分别高 1.23/1.10 pp，在 Reddit-S 高 0.11 pp；Accuracy 分别高 0.53/0.20/0.00 pp。
   - **建议：** 本轮没有证据表明 Chebyshev 更优。下一版最小候选优先 anchored bank（alpha 0.1），但 3-seed 结果仍需外部审查，不据此宣称普遍优势。

4. **当前多阶残差是否能够作为稳定主干？**
   - **已验证：** 关闭结构残差导致三个数据集均出现明显下降，说明该路径是当前有效模型的重要部分；UA 保留同一 RCMI/残差，在本轮达到相同或略好结果。Grocery Visual eta 高阶饱和明显，原 bank 的 `B3-B1` 比例跨数据集变化大。
   - **建议：** 保留逐阶 eta 加权残差作为主干机制，但以 anchored bank 作为更简单的传播候选，并继续审查 eta 饱和；不能把当前 Chebyshev 高阶 bank 的数值稳定性当作已证实事实。

5. **下一步是否增加显式跨模态消息交互？**
   - **已验证：** q shuffle 与 learned-vs-uniform P 的固定 checkpoint 影响有限；本轮没有实现或测试显式跨模态消息传递。
   - **建议/推断：** 当前证据不足以支持增加新消息交互，先不增加。若后续确需研究，应独立加入单一、可关闭的跨模态消息路径并保持同一验证协议。本轮不作该架构修改。

### 对 VNext 低于 CoSI Final 的解释边界

**观察到的事实：** VNext 相对历史 CoSI Final 的 validation Macro-F1 差距约为 Movies 3.41 pp、Grocery 2.09 pp、Reddit-S 1.01 pp。关系传播权重与 uniform 很接近，关系 scorer 边间输出变化也小；跨阶 attention 贡献较小，类别 F1 错误集中在少数持续困难类别。UA 在 Movies/Grocery 提升明显但没有完全抹平历史差距。

**可能解释（推断，不是因果结论）：** 当前 VNext 的复杂 relation-weighted propagation 没有学出明显不同于 uniform 的算子，q 对逐节点预测影响有限，跨阶 attention 贡献弱；同时 Chebyshev 高阶响应和 eta 饱和存在数据集差异。这些机制复杂度可能没有转成足够的类别区分能力，令少数类 F1 偏低。现有结果不能证明这些因素是 VNext 低于 CoSI Final 的唯一原因；CoSI Final 是不同模型配置的历史对照，缺少本轮逐类别配对消融。

## 复现与输出

主要执行命令：

```bash
conda run -n yhf_env python -m py_compile \
  src/models/cosi_mag_vnext_r3.py \
  src/models/cosi_mag_vnext_u.py \
  src/models/cosi_mag_vnext_uq.py \
  src/models/cosi_mag_vnext_ua.py \
  scripts/cosi_mag_vnext_round3.py
conda run -n yhf_env python scripts/cosi_mag_vnext_round3.py diagnostics
conda run -n yhf_env python scripts/cosi_mag_vnext_round3.py train
conda run -n yhf_env python scripts/cosi_mag_vnext_round3.py compare
```

报告和小型机器结果：

- `reports/cosi_mag_vnext/round3/variant_validation_metrics.csv`：27 个逐 seed validation、epoch、资源、参数与 checkpoint SHA。
- `reports/cosi_mag_vnext/round3/variant_provenance.json`：split/checkpoint/config 来源、任务超参数、汇总指标。
- `reports/cosi_mag_vnext/round3/fixed_checkpoint_diagnostics.json`：九个原 checkpoint 的关系、响应、q、eta、attention、残差诊断与 hash。
- `reports/cosi_mag_vnext/round3/feature_similarity.csv`、`q_perturbation.csv`：可直接分析的特征和 q 干预数据。
- `reports/cosi_mag_vnext/round3/variant_per_class_f1.csv`、`variant_confusion_matrices.json`：逐 seed per-class 与 validation confusion。
- `reports/cosi_mag_vnext/round3/comparison_references.json`：新变体、paired VNext、历史 CoSI Final 和 GCN/SAGE 对照。

训练 checkpoint 和日志保存在 ignored 路径 `outputs/cosi_mag_vnext_round3/runs/`，没有纳入提交；其绝对路径及 SHA 记录在 provenance 中。新模型代码只增加独立变体，未更改公共任务协议、冻结模型实现或历史实验产物。本轮结束后停止模型修改，等待外部审查。
