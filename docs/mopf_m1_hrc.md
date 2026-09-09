# MoPF M1-A — Hierarchical Residual Centering Candidate Test

本报告记录 M1-A 的单候选实验：Hierarchical Residual Centering（HRC）。所有实验均为 seed=42、NC full-graph training、`separate_cos`；Movies/ele-fashion 使用 K=3，Grocery 使用已经由 validation 选择的 K=2。没有运行 LP、三 seed 或正式 benchmark 更新。

## 1. Motivation from M0

M0 审计显示，部分 dataset/order 的 `delta_node` 存在明显 shared offset，同时也存在真实的 node-wise variation。例如 M0 的 Movies/ele-fashion 高阶 residual 均值明显非零，而 Movies visual、ele-fashion order 1 等位置仍有可观的节点标准差。因此本轮只检验一个 identifiability candidate：把 shared offset 推回 hierarchical global/modality 层，同时保留 zero-mean node personalization。

## 2. HRC definition

对当前训练节点集合计算：

```text
mu_m,k = mean_i(delta_node_i,k^m)
L_hrc = 1 / (2 * (K + 1)) * sum_m sum_k mu_m,k^2
aux_loss = hrc_weight * L_hrc
loss = task_loss + task.loss.aux_weight * aux_loss
```

`task.loss.aux_weight` 保持 1.0。HRC 不对前向的 `delta_node` 做 hard centering；有效系数仍严格为 `gamma_global + delta_gamma_m + delta_node_i^m`。配置新增 `model.hrc_weight`，默认精确为 `0.0`。当 `use_node_residual=false` 或 `hrc_weight=0` 时，HRC contribution 为精确零。

## 3. Implementation integrity

- HRC 复用了 `_encode_components()` 生成的 raw `delta_node_text/visual`，没有改动 semantic graph、polynomial bank、gamma parameterization、node residual generator、fusion、classifier 或 checkpoint selection。
- NC full-graph 训练循环只额外向 MoPF 提供 `train_idx`，因此均值确实在当前训练节点上计算；runtime index 不进入 checkpoint state dict。
- 默认权重关闭时 embedding 与当前 MoPF forward 等价；新增单测覆盖 A–F，并额外验证 training-node index。M1-A 专项 `tests/test_mopf.py` 共 14 个测试通过。
- 每个 best Validation Accuracy checkpoint 均重导了 M0 所需的 `gamma_global`、`delta_gamma`、`delta_node`、`eta`、effective radius，以及 modality/order mean/std。
- 完整 pytest：`116 passed, 1 warning`。

对应实现与分析脚本：[`src/models/mopf.py`](../src/models/mopf.py)、[`src/tasks/nc.py`](../src/tasks/nc.py)、[`configs/model/mopf.yaml`](../configs/model/mopf.yaml)、[`scripts/run_mopf_hrc_analysis.py`](../scripts/run_mopf_hrc_analysis.py)。

## 4. Performance

以下均为单个 seed42 run；格式为 Val Acc / Val Macro-F1 / Test Acc / Test Macro-F1。

| Dataset | hrc_weight | Best epoch | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 0 | 92 | 57.02% | 50.29% | 54.42% | 48.65% |
| Movies | 1 | 88 | 57.41% | 49.42% | 54.84% | 49.59% |
| Movies | 10 | 67 | 57.41% | 50.35% | 55.56% | 49.17% |
| ele-fashion | 0 | 131 | 88.15% | 76.17% | 88.21% | 76.64% |
| ele-fashion | 1 | 111 | 88.15% | 76.99% | 88.14% | 77.53% |
| ele-fashion | 10 | 199 | 88.35% | 77.58% | 88.51% | 78.47% |
| Grocery K=2 | 0 | 63 | 83.60% | 76.44% | 83.54% | 76.06% |
| Grocery K=2 | 1 | 139 | 83.89% | 78.13% | 83.13% | 75.31% |
| Grocery K=2 | 10 | 125 | 83.69% | 78.23% | 83.10% | 75.79% |

性能证据总体为：Movies 与 ele-fashion 的 λ=1/10 保持或提高 validation/test accuracy，ele-fashion λ=10 四项均为该数据集候选中最好；Grocery validation 保持同一 band，但 test accuracy 和 test Macro-F1 相对 λ=0 有轻微下降。由于只有一个 seed，不能把这些差异当成稳健 benchmark gain。

## 5. Shared-offset reduction

下表是训练节点上两种 modality、全部 order 的 HRC raw loss，即各 `mu_m,k^2` 的平均；不是 test 指标。

| Dataset | λ=0 | λ=1 | λ=10 | λ=10 相对 λ=0 |
|---|---:|---:|---:|---:|
| Movies | 0.00337285 | 0.00122638 | 0.00004628 | -98.6% |
| ele-fashion | 0.00444772 | 0.00022535 | 0.00000393 | -99.9% |
| Grocery K=2 | 0.00150258 | 0.00048217 | 0.00002086 | -98.6% |

每个 modality/order 的绝对训练节点均值也显著下降：最大值由 Movies `0.07175→0.00979`、ele-fashion `0.08559→0.00346`、Grocery `0.06393→0.00594`（λ=0→10）。这直接支持 HRC 的 identifiability 目标。

## 6. Node-variation preservation

HRC 惩罚的是均值平方，不是 `std(delta_node)`。实验中 λ=10 没有把 node variation 压到零：各 dataset 的最大训练节点 std 为 Movies `0.03421→0.03485`、ele-fashion `0.02611→0.05845`、Grocery `0.00472→0.06081`（λ=0→10）。新增 E 单测使用 zero-mean but varied residual，得到 raw HRC loss 约等于 0，证明公式本身不惩罚合法 zero-mean diversity。

同时，这项结果也暴露风险：Grocery 和 ele-fashion 某些 order 的 std 增长较大。因此 HRC 成功去掉 shared offset，但不能单独保证 node profile 的幅度校准。

逐 modality/order 的 A/B/C/D/E 统计位于每个 run 的 [`hrc_metrics.csv`](../outputs/m1_hrc/Movies/lambda10/hrc_metrics.csv)；C 指标为 `abs(mean)/(std+1e-8)`，仅作描述性诊断。

## 7. Hierarchical effect transfer

以最高阶作为代表（完整 order 明细见 `hrc_metrics.csv`），训练节点统计如下：

| Dataset / modality | mean(delta_node) λ=0 → 10 | delta_gamma λ=0 → 10 | mean(eta) λ=0 → 10 |
|---|---:|---:|---:|
| Movies / text, k=3 | -0.07175 → -0.00535 | -0.07434 → -0.06721 | -0.22298 → -0.13621 |
| Movies / visual, k=3 | -0.06976 → -0.00979 | -0.06632 → -0.05039 | -0.21296 → -0.12380 |
| ele-fashion / text, k=3 | -0.06281 → -0.00134 | -0.05584 → -0.07921 | -0.18026 → -0.17379 |
| ele-fashion / visual, k=3 | -0.07336 → -0.00072 | -0.06879 → -0.12120 | -0.20375 → -0.21511 |
| Grocery / text, k=2 | -0.06025 → -0.00594 | -0.05667 → -0.12628 | 0.61290 → 0.55132 |
| Grocery / visual, k=2 | 0.00947 → -0.00186 | 0.01800 → -0.00636 | 0.75729 → 0.67530 |

`delta_gamma` 确实随 node mean 的消失而重新调整，方向和幅度具有 dataset/modality/order 依赖性；`eta` 没有精确守恒。这支持“shared offset 的归属发生转移”的描述性解释，但不足以声称 exact conservation 或因果 transfer。

## 8. Effective-profile preservation

effective radius 定义为 `sum_k k*abs(eta_i,k) / sum_k abs(eta_i,k)`。均值如下：

| Dataset | λ | text radius | visual radius |
|---|---:|---:|---:|
| Movies | 0 / 1 / 10 | 1.551 / 1.542 / 1.552 | 1.594 / 1.574 / 1.540 |
| ele-fashion | 0 / 1 / 10 | 1.514 / 1.518 / 1.524 | 1.514 / 1.554 / 1.537 |
| Grocery K=2 | 0 / 1 / 10 | 1.349 / 1.178 / 1.276 | 1.654 / 1.498 / 1.527 |

Movies 与 ele-fashion 的 effective radius 基本稳定；Grocery 的 text radius 变化约 0.17，说明 HRC 在 K=2 的 Grocery 上会重新塑造有效 profile。所有 checkpoint 的完整 eta 分布、radius 分布和结构分组统计都已保存在 M0-compatible artifacts 中。

## 9. Dataset-specific behavior

- Movies：HRC λ=1/10 将 text 的 shared offset 明显拆出，同时 visual variation 保持；λ=10 的 Test Acc 最好，但 Test Macro-F1 略低于 λ=1。
- ele-fashion：HRC 效果最一致。λ=10 将两种 modality 的均值压到约 `0.001` 量级，并保持/增加 node variation；四项性能均不降，Test Macro-F1 达到 78.47%。
- Grocery K=2：shared-offset reduction 清晰，但 text profile std 和 effective radius 对 λ 更敏感；validation 指标改善，test 指标相对 λ=0 轻微下降，说明泛化和 profile calibration 仍需多 seed 验证。

## 10. Whether HRC should enter the final MoPF

本轮不应把 HRC 直接写入最终默认 MoPF。机制证据强：三个 dataset 都出现近 99% 的 raw shared-offset reduction，且 zero-mean variation 没有被公式直接惩罚；但证据仍只有 seed42，Grocery test 有轻微损失，ele-fashion/Grocery 的 node/eta std 在 λ=10 下明显增大，Grocery effective profile 也有较大移动。因此当前最稳妥的决策是保留候选，等待多 seed、profile stability 和 test 泛化复核。

## 11. Recommended hrc_weight if supported

本轮不提供最终部署权重。若后续多 seed 验证继续支持 HRC，`hrc_weight=10` 是优先复核候选：它在三个 dataset 上都把 shared offset 压得最充分，并在 Movies/ele-fashion 上保持良好性能；但 Grocery 的 profile 变化意味着不能直接将 10 设为最终默认值。

## 12. Remaining issues

- 当前实验严格是 seed42，尚不能估计跨 seed 方差或显著性。
- HRC 只约束 training-node mean；validation/test 节点的 residual mean 可能不同，需要 split-wise audit。
- HRC 不约束 node residual std 或 eta amplitude，ele-fashion/Grocery 的 std 增长需要后续 stability check。
- λ=0 的重复实验受 GPU 数值/训练路径影响，不能仅凭单次 λ=0 与 M0 历史 run 做 bitwise performance 对照；代码等价性由单测保证。
- 未测试 HRC 对 LP 的影响，符合本轮禁止运行 LP 的范围。

## Final decision

**Keep HRC as analysis-only candidate**

