# CoSI-MAG VNext 第二轮研究验证

日期：2026-10-12（Asia/Shanghai）

## 1. 范围与数据使用

审查分支为 `exp/cosi-mag-vnext-crc-cheb-rcmi`，审查基准提交为 `9a4d136590a83546cd2544f434076edb07cca907`。本轮没有修改 VNext 的模型定义，也没有重新训练九个 VNext checkpoint。所有 checkpoint 仍是上一轮按 validation Accuracy 选出的 best checkpoint。本轮只在验证节点上比较固定模型的前向干预，并另外训练 GCN/SAGE 验证基线。

数据加载后，脚本立即将 test 节点标签覆盖为忽略值；训练与所有新指标只使用 train/validation 标签。用于 Macro-F1 的 20 类集合只从 train/validation 节点标签得到。九个 checkpoint 的 validation Accuracy 与 Macro-F1 均与保存的 validation 指标复现到浮点精度（最大绝对差 `1.11e-16`）；原始前向和诊断重构输出的最大差为 `0`。审计偏差：检查旧 CoSI Final 报告时，一条宽范围文档读取命令意外输出了含 Test 列的历史结果表。我没有转录、计算、比较或据此选择模型；本轮新增训练、干预、类别统计及 checkpoint 选择均没有使用 test 指标。该意外读取作为本轮流程偏差保留记录。

所有数据集使用 split seed 42，NC 协议为 `unified_full_graph_nc_v1`。数据与 checkpoint 的绝对路径、SHA-256、best epoch 和选择规则见 [`round2/checkpoint_provenance.json`](../reports/cosi_mag_vnext/round2/checkpoint_provenance.json)。split 摘要如下：

| 数据集 | Split 文件 | SHA-256 |
| --- | --- | --- |
| Movies | `/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt` | `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211` |
| Grocery | `/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt` | `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e` |
| Reddit-S | `/hdd1/DataInHere/YHF/data/MAGB_split/Reddit-S_nc_seed42_train0.6_val0.2.pt` | `2154e82ef061de9d3c018aa19674ea6e2220953f272bedebb1559396e43750a5` |

## 2. 固定 checkpoint 路径干预

对每个 checkpoint 先缓存其 learned `P_text`、`P_visual`、`q` 和 Chebyshev banks。干预不再次调用 CRC 来替换缓存中的其他量：

- **uniform operator / keep q：** 用原始物理图、均匀边权和一个 self-loop 构造两个归一化算子；保留 checkpoint 的 `q`。
- **shuffle q / keep operator：** 只按固定 seed `20261012` 打乱 `q` 的节点行；传播算子和 Chebyshev banks 保持原样。
- **attention update off：** 只将 `omega * attention_output` 置零，再用原始 `q` 和 learned banks 计算后续 `eta` 与 `U`。
- **structural residual off：** 仍计算 attention、`eta`、`U` 和 gate，只将最终应用的 `g * U` 置零。

没有训练或更新任何参数。下表为三个 seeds 的均值 ± population 标准差；差值以百分点计，正值表示相对同一 checkpoint 原始输出提高。每个 seed、每个条件的完整 Accuracy、Macro-F1 和差值见 [`round2/intervention_metrics.csv`](../reports/cosi_mag_vnext/round2/intervention_metrics.csv) 和 JSON 汇总。

| 数据集 | 固定 checkpoint 条件 | Δ Accuracy (pp) | Δ Macro-F1 (pp) |
| --- | --- | ---: | ---: |
| Movies | uniform operator / keep q | +0.000 ± 0.000 | +0.001 ± 0.001 |
|  | shuffle q / keep operator | +0.040 ± 0.051 | −0.022 ± 0.078 |
|  | attention update off | −0.130 ± 0.037 | −0.278 ± 0.209 |
|  | structural residual off | −4.069 ± 0.954 | −6.175 ± 1.323 |
| Grocery | uniform operator / keep q | −0.020 ± 0.028 | −0.057 ± 0.081 |
|  | shuffle q / keep operator | −0.088 ± 0.072 | −0.143 ± 0.076 |
|  | attention update off | −0.127 ± 0.050 | −0.283 ± 0.227 |
|  | structural residual off | −8.707 ± 0.739 | −7.774 ± 0.896 |
| Reddit-S | uniform operator / keep q | +0.000 ± 0.000 | −0.002 ± 0.002 |
|  | shuffle q / keep operator | +0.000 ± 0.051 | −0.000 ± 0.140 |
|  | attention update off | −0.010 ± 0.015 | +0.009 ± 0.032 |
|  | structural residual off | −3.743 ± 0.044 | −4.547 ± 0.390 |

**观察：** learned 边权算子替换为 uniform 后，三个数据集的验证指标几乎不变；打乱 `q` 也只有很小影响。关闭 attention 更新影响较小，关闭最终结构残差则在三个数据集上都造成明显下降。这说明当前 checkpoint 的预测依赖 Chebyshev 结构残差，但 learned CRC 边重加权与 RCMI 的关系条件对验证预测贡献有限。固定 checkpoint 干预不能证明重新训练后这些机制一定无用，也不能解释相对 CoSI Final 的全部差距。

## 3. 关系算子、滤波与条件路径

### 3.1 传播算子差异

`P_text` 与 uniform 的差以 `||P_text-P_uniform||_F / ||P_uniform||_F` 计算；Visual 同理。Text 与 Visual 的差以 `||P_text-P_visual||_F / ||(P_text+P_visual)/2||_F` 计算。用稀疏 COO 运算合并重复坐标后计算 Frobenius 范数，没有创建稠密 `N×N` 矩阵。

| 数据集 | Text vs uniform | Visual vs uniform | Text vs Visual |
| --- | ---: | ---: | ---: |
| Movies | 0.252% ± 0.118% | 0.401% ± 0.140% | 0.651% ± 0.238% |
| Grocery | 0.238% ± 0.159% | 0.250% ± 0.102% | 0.427% ± 0.235% |
| Reddit-S | 0.311% ± 0.184% | 1.429% ± 0.243% | 1.118% ± 0.248% |

除 Reddit-S 的 Visual 算子外，两种 learned 算子与 uniform 的相对 Frobenius 差均低于 0.5%。Reddit-S Visual 的差仍只有约 1.4%。这与 uniform 算子干预的验证结果一致。

### 3.2 Chebyshev 响应

下表为 `B_k = T_k(P)H0` 的各阶响应 RMS，数值为三个 checkpoint 的均值，阶数顺序为 `k=0,1,2,3`。完整每阶 Pearson 相关矩阵保存在 [`round2/mechanism_diagnostics.json`](../reports/cosi_mag_vnext/round2/mechanism_diagnostics.json)。相关性为将节点和特征维展平后计算的 Pearson 相关；表中“阶间相关”是每个模态六个非对角阶对的均值，“跨模态相关”逐阶给出。

| 数据集 | 模态 | RMS（k=0,1,2,3） | 平均阶间相关 |
| --- | --- | --- | ---: |
| Movies | Text | 0.743, 0.625, 0.675, 0.651 | 0.623 |
|  | Visual | 0.704, 0.539, 0.610, 0.567 | 0.360 |
| Grocery | Text | 0.697, 0.531, 0.580, 0.562 | 0.315 |
|  | Visual | 0.705, 0.518, 0.582, 0.559 | 0.233 |
| Reddit-S | Text | 0.697, 0.594, 0.697, 0.594 | 0.714 |
|  | Visual | 0.744, 0.683, 0.744, 0.683 | 0.850 |

各阶响应并非独立。尤其 Reddit-S 的同模态阶响应相关较高，`k=0/2` 与 `k=1/3` 的 RMS 也相近；多阶 bank 可能包含较强冗余。跨模态同阶相关（`k=0,1,2,3`）分别为：Movies `−0.001, 0.061, 0.145, 0.153`；Grocery `−0.002, 0.131, 0.221, 0.210`；Reddit-S `−0.002, −0.002, −0.002, −0.002`。

### 3.3 Attention、eta 与结构残差

Attention 增量比率定义为 `RMS(omega * attention_output)`（仅 `k=1..K`）除以原始 `RMS(B_k-B_0)`。同时列出将 `k=0` 更新从高阶更新中扣除后的差分比率，以反映其实际改变 `D[k]` 的幅度。

| 数据集 | 模态 | attention 增量 / 原始结构差异 | 差分 attention 增量 / 原始结构差异 | eta 节点标准差（阶 1,2,3） |
| --- | --- | ---: | ---: | --- |
| Movies | Text | 10.7% | 2.4% | 0.0538, 0.1204, 0.1154 |
|  | Visual | 10.2% | 6.5% | 0.0255, 0.0750, 0.0756 |
| Grocery | Text | 8.8% | 4.2% | 0.0435, 0.1103, 0.1101 |
|  | Visual | 9.2% | 7.1% | 0.0194, 0.0661, 0.0650 |
| Reddit-S | Text | 8.4% | 2.7% | 0.0183, 0.0613, 0.0337 |
|  | Visual | 14.6% | 4.6% | 0.0255, 0.0708, 0.0428 |

`eta` 在节点间有一定变化，但 attention 对高阶差异的有效改变量较小。这与关闭 attention 后 Macro-F1 只小幅变化一致。`q` 的节点中心化 RMS 为 Movies `0.266`、Grocery `0.200`、Reddit-S `0.187`，上下文确实随节点变化；不过固定算子下打乱 `q` 的预测变化很小，说明这些变化没有显著传到当前预测结果。

最终有效残差的逐节点比例定义为 `RMS(g*U)_i / RMS(H0)_i`。下表给出逐 checkpoint 经验分位数后再对 seed 求均值；完整 0%、25%、50%、75%、90%、95%、99%、100% 分位数均在 JSON 中。

| 数据集 | 模态 | 中位数 | 90% 分位 | 95% 分位 |
| --- | --- | ---: | ---: | ---: |
| Movies | Text | 0.105 | 0.229 | 0.278 |
|  | Visual | 0.376 | 0.552 | 0.596 |
| Grocery | Text | 0.321 | 0.513 | 0.561 |
|  | Visual | 0.539 | 0.723 | 0.767 |
| Reddit-S | Text | 0.353 | 0.500 | 0.548 |
|  | Visual | 0.191 | 0.334 | 0.401 |

### 3.4 验证集类别错误

混淆矩阵按每个 seed 单独保存；每类 F1 也提供逐 seed 值和跨 seed 均值。下表列出 VNext 每个数据集均值 F1 最低的类别，以及跨三个 seeds 汇总混淆矩阵中最常见的若干错误方向。混淆计数是三个同一 split 预测矩阵的计数和，类别 F1 则是逐 seed F1 的均值。

| 数据集 | 低 F1 类别（验证支持；F1 均值 ± 标准差） | 主要错误方向 |
| --- | --- | --- |
| Movies | class 16（38；0.015 ± 0.021），class 9（37；0.191 ± 0.079），class 5（82；0.297 ± 0.027），class 15（47；0.324 ± 0.055） | class 0→1：420/1065；class 1→0：259/3294；class 2→1：258/1089。class 16 平均每个 checkpoint 仅被预测约 4 次。 |
| Grocery | class 4（54；0.271 ± 0.019），class 2（26；0.448 ± 0.037），class 6（14；0.492 ± 0.052），class 5（168；0.674 ± 0.014） | class 4→14：44/162；class 9→12：54/678；class 16→12：59/645。低支持类 6 的 seed 间波动相对更大。 |
| Reddit-S | class 13（31；0.446 ± 0.038），class 10（53；0.695 ± 0.018），class 3（84；0.844 ± 0.009） | class 13→15：19/93；class 10→3：46/159；class 15→6：20/474。 |

这些结果显示 Movies 的 class 16/9 和 Grocery 的 class 4/2 等类别召回较弱，能解释 VNext 自身 Macro-F1 低于 Accuracy 所暗示的整体表现。当前没有 CoSI Final 在同一 checkpoint 口径下的 per-class 预测，因此不能将这些错误类别直接归因为 VNext 与 CoSI Final 之间的 Macro-F1 差距。

## 4. GCN / SAGE 验证基线

运行 GCN 与 SAGE，各数据集 seeds 42/43/44，共 18 次。运行时使用 `yhf_env` 与 `cuda:1`，并复用现有的 `_run_single_nc` 单次 NC 训练/验证实现。为避免读取 test 标签，数据载入后 test 节点标签即被屏蔽，`task.evaluate_test=false`。训练为 full graph，checkpoint 按 validation Accuracy 选择。

训练设置：AdamW，学习率 `1e-3`，weight decay `1e-4`，最多 300 epochs，每 epoch 验证，patience 30、最小 epoch 30、最小 Accuracy 提升 `1e-4`，gradient clip 1.0。GCN 与 SAGE 均为 3 层、hidden dim 256、dropout 0.2、ReLU、BatchNorm。为避免现有 `run_nc` 汇总器在 `evaluate_test=false` 时仍从 test 标签解析 Macro-F1 类别集合，脚本直接调用同一训练/验证实现 `_run_single_nc`，并仅用 train/validation 标签得到 20 类集合；该集合与完整任务类别一致。下表的均值和标准差基于三个 run，单位为百分比。

| 数据集 | 模型 | Validation Accuracy | Validation Macro-F1 |
| --- | --- | ---: | ---: |
| Movies | GCN | 52.71 ± 0.16 | 43.86 ± 1.14 |
|  | SAGE | 52.65 ± 2.23 | 40.32 ± 11.63 |
|  | CoSI-MAG VNext | 55.09 ± 0.03 | 46.55 ± 1.53 |
|  | CoSI-MAG Final（历史） | 57.51 ± 0.26 | 49.96 ± 0.87 |
| Grocery | GCN | 80.65 ± 0.13 | 72.53 ± 1.00 |
|  | SAGE | 83.28 ± 0.17 | 76.67 ± 0.36 |
|  | CoSI-MAG VNext | 83.35 ± 0.37 | 76.38 ± 0.69 |
|  | CoSI-MAG Final（历史） | 83.91 ± 0.16 | 78.47 ± 0.71 |
| Reddit-S | GCN | 94.25 ± 0.16 | 90.60 ± 0.07 |
|  | SAGE | 95.32 ± 0.08 | 91.75 ± 0.14 |
|  | CoSI-MAG VNext | 96.23 ± 0.08 | 92.23 ± 0.15 |
|  | CoSI-MAG Final（历史） | 96.37 ± 0.05 | 93.24 ± 0.13 |

逐 seed 的 best epoch、指标、训练/验证耗时、显存和 checkpoint/config 哈希见 [`round2/baseline_metrics.csv`](../reports/cosi_mag_vnext/round2/baseline_metrics.csv) 与 JSON。每次时间是 `_run_single_nc` 的实测墙钟时间，包含模型/分类头/优化器构建、训练、验证和 checkpoint 保存，不含数据加载、配置读取及用于参数计数的临时模型构建；完整日志和 resolved configs 在 `outputs/cosi_mag_vnext_round2/baselines/{Movies,Grocery,Reddit-S}/{gcn,sage}/seed{42,43,44}/`。

逐 seed 的单次运行记录使用“best epoch；Accuracy/Macro-F1 (%)；秒”格式：

| 数据集 | 模型 | Seed 42 | Seed 43 | Seed 44 |
| --- | --- | --- | --- | --- |
| Movies | GCN | 82；52.759/43.138；9.0 | 81；52.490/42.979；8.3 | 132；52.879/45.467；12.4 |
| Movies | SAGE | 82；54.199/49.982；9.8 | 28；49.490/23.953；5.1 | 101；54.259/47.013；11.4 |
| Grocery | GCN | 98；80.644/72.547；9.5 | 100；80.820/73.746；9.8 | 76；80.498/71.286；8.1 |
| Grocery | SAGE | 99；83.397/76.974；10.8 | 109；83.045/76.167；11.7 | 91；83.397/76.859；10.0 |
| Reddit-S | GCN | 38；94.432/90.670；5.6 | 60；94.055/90.510；7.3 | 42；94.275/90.631；5.9 |
| Reddit-S | SAGE | 85；95.282/91.775；12.1 | 118；95.439/91.899；15.5 | 126；95.250/91.561；16.3 |

GCN/SAGE 配置差异需要保留在解读中：两者使用单路 256 维、3 层 BatchNorm/ReLU 图编码；VNext 是双模态独立投影、LayerNorm、CRC、Chebyshev bank 和关系条件残差；历史 CoSI Final 使用独立配置中的 learned diagonal cosine edge metric、rank-4 filtering、anchored/cumulative multihop、global prior 和单层单头 hop interaction。三类模型共用任务配置 `configs/task/nc.yaml`（SHA-256 `8e7e5b580f2aea3112fdd20076f57ea16c143ec57730abd5ed094634c9068e75`）、数据 split、run seeds 和 validation Accuracy checkpoint 选择，但模型结构与归纳偏置不同，因此 GCN/SAGE 是基线核对而非 VNext 组件的受控消融。

## 5. 对性能差距的判断与最小下一步

三组数据中 VNext 均高于本轮 GCN 的 Macro-F1；在 Grocery 与 SAGE 接近，在 Reddit-S 高于 SAGE，但仍低于历史 CoSI Final。VNext 相比 CoSI Final 的 Macro-F1 差分别为 Movies `−3.41` pp、Grocery `−2.09` pp、Reddit-S `−1.01` pp。Movies 的 Accuracy 差也达到 `−2.42` pp。当前证据支持以下有限结论：

1. learned CRC 边权没有实质改变归一化算子，固定 checkpoint 使用 uniform 算子也不改变预测。它目前不像是性能差距的主要来源。
2. `q` 本身存在节点变化，但将 `q` 打乱几乎不改变预测。RCMI 中的关系条件路径可能被其他特征路径绕过，或关系上下文对分类任务信息不足。
3. attention 更新的结构差异幅度较小，关闭后影响也小；当前最有影响的路径是最终 `g*U` 结构残差。
4. Movies/Grocery 的 per-class 结果显示多个类别被系统性混淆或极少被预测。只按 Accuracy 选 checkpoint 可能与 Macro-F1 目标存在偏差，但本轮没有重新选 checkpoint，也不能据此断言这是相对 CoSI Final 差距的原因。

最小的下一步结构验证建议是只将 learned `P_text/P_visual` 改为 uniform `P`，保留 `q`、MMF、attention 和 `U`，使用相同 seeds/split 重新训练并比较 validation 指标。它能检验是否可以删除几乎不起作用的边重加权分支，不应预设性能会提高。再根据该结果决定是否单独检验关系条件路径；本轮不实施这些改动。

## 6. 统计口径与复现自审

- **checkpoint 来源：** 九个模型和分类头都从 round-one `run_manifest.json` 指向的 best-validation checkpoint 加载；逐文件 SHA-256、选择指标和 epoch 已保存。没有使用其他轮次或 test 选出的 checkpoint。
- **split：** 每个数据集的三次训练都使用相同 seed-42 split；实际加载路径与上一轮 manifest 的路径及 SHA-256 一致。
- **标签与指标：** train/validation 标签集合包含 20 个类别；Macro-F1、混淆矩阵和 per-class F1 只索引 validation nodes。baseline 训练中 test 标签已屏蔽，且 test evaluation 关闭。
- **模型选择：** GCN/SAGE 每 epoch 在 validation 上评估，仍按 Accuracy 选择 best checkpoint，与已有协议一致。没有基于 Macro-F1 或 test 指标重新选择模型。
- **干预隔离：** uniform 算子干预保留原始 `q`；`q` shuffle 保留 learned 算子及 Chebyshev bank；attention 与结构残差开关只归零对应更新路径。原始路径被逐 checkpoint 重构，最大绝对误差为 0。
- **差异定义：** operator 差使用稀疏 Frobenius 范数；response Pearson 使用展平后的节点/特征向量；attention 同时报告完整增量及扣除 order-0 后对 `D[k]` 的增量；eta 标准差逐 order、逐模态跨节点计算；有效残差比例逐节点计算后取分位数。
- **运行代码：** [`scripts/cosi_mag_vnext_round2.py`](../scripts/cosi_mag_vnext_round2.py)。本轮命令：

```bash
conda run -n yhf_env python scripts/cosi_mag_vnext_round2.py diagnose
conda run -n yhf_env python scripts/cosi_mag_vnext_round2.py baselines
```

## 7. 产物

结构化结果在 `/hdd1/DataInHere/YHF/MoPF_IAMOC/reports/cosi_mag_vnext/round2/`：

- `intervention_metrics.json` / `intervention_metrics.csv`
- `mechanism_diagnostics.json`
- `per_class_f1.json` / `per_class_f1.csv`
- `validation_confusion_matrices.json`
- `baseline_metrics.json` / `baseline_metrics.csv`
- `checkpoint_provenance.json` / `checkpoint_provenance.csv`

本轮训练日志、resolved configs 和 GCN/SAGE best-validation checkpoint 保存在 `/hdd1/DataInHere/YHF/MoPF_IAMOC/outputs/cosi_mag_vnext_round2/baselines/`，不提交到 Git。

## 8. 未解决问题

- 没有 CoSI Final 的同口径 per-class 输出或 checkpoint 干预结果，不能将类别混淆与两模型间的性能差距建立因果关系。
- 只有三个 run seeds，所有 run 使用同一 split；没有跨 split 或统计显著性验证。
- Movies 的 SAGE 三 seed 结果波动较大，seed 43 Macro-F1 仅 23.95%，应结合 per-run checkpoint 与日志看待其均值。
- 该结果仅覆盖 NC validation，不涉及测试集或 LP 性能；本轮建议的简化方案尚未训练验证。
