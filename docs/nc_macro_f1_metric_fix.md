# NC Macro-F1 fixed evaluation label set

## 问题与修复

legacy NC evaluator 调用 `sklearn.metrics.f1_score` 时没有传入 `labels`。在 `labels=None` 的 dynamic-label 行为下，sklearn 会根据当前 split 的真实标签和预测标签构造平均集合。如果模型偶然预测出一个在监督任务中没有 ground-truth support 的类别，该 phantom class 会进入 Macro-F1 分母，并贡献一个零分项；因此分母和分数会随预测偶然变化，而不是只反映有效监督类别的性能。

修复位于 [`src/tasks/nc.py`](../src/tasks/nc.py)：

1. `_resolve_nc_eval_labels(data)` 收集 `train_idx + val_idx + test_idx` 对应的监督标签。
2. 过滤负标签/缺失标签以及超出 `[0, num_classes)` 的无效 id。
3. 对有效 id 去重、排序，形成一个 task-level stable label set。
4. train/val/test 的每次 Macro-F1 都显式传入同一个 `eval_labels`。
5. Accuracy、classifier 输出维度、CrossEntropyLoss、checkpoint selection、split 和 MoPF forward 均未改变。

如果模型预测了不属于 `eval_labels` 的 phantom class，该预测不会创建新的 Macro-F1 类别，但对应真实类别仍会因错误预测产生 false negative，继续受到惩罚。

每个 NC run 开始时日志打印：

```text
NC valid evaluation labels: [...]
NC number of evaluated classes: X
```

ele-fashion 实际监督 label set 为 **`[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]`**，共 **11** 类；class 5 在 train/val/test 的监督并集中没有出现，因此不被硬编码加入评估分母。

`scripts/run_targeted_diagnosis.py` 的离线 per-class 导出也复用该 helper，并正确处理非连续 class id。

## Regression tests

新增 [`tests/test_nc_metrics.py`](../tests/test_nc_metrics.py)，覆盖：

- y_true 只有 class 0/1、预测出现 class 2：class 2 不进入分母，但对真实类造成的 false negative 仍计入；
- `num_classes` 大于监督任务实际出现的类别数时不加入 phantom class；
- 某类缺失于 test 但存在于 train/val 时，三个 split 使用同一 union label set；
- 普通连续类别数据的 fixed 结果与 legacy Macro-F1 一致；
- Accuracy 不受 fixed label set 影响。

完整测试命令为 `conda run --no-capture-output -n yhf_env python -m pytest -q`，结果为 **109 passed, 1 warning**。MoPF 相关 forward 测试也通过。

## ele-fashion MoPF three-seed validation

本轮 fixed-metric 输出位于 [`outputs/nc_metric_fix_validation/ele-fashion/mopf`](../outputs/nc_metric_fix_validation/ele-fashion/mopf)，配置仍为正式 MoPF：K=3、`separate_cos`、rank=4、global/modality/node filter 全开启、full graph、AdamW、Validation Accuracy checkpoint selection。

### legacy metric 与 fixed metric

旧正式三 seed 输出使用 legacy dynamic-label metric：[`outputs/2026-09-09/15-14-25`](../outputs/2026-09-09/15-14-25)。新输出使用 stable 11-class label set：

| metric | legacy mean ± std | fixed mean ± std | fixed - legacy |
|---|---:|---:|---:|
| Val Accuracy | 88.19 ± 0.06% | 88.01 ± 0.08% | -0.17 pp |
| Test Accuracy | 88.22 ± 0.06% | 88.12 ± 0.12% | -0.10 pp |
| Test Macro-F1 | 75.36 ± 3.22% | 76.65 ± 0.14% | +1.29 pp |

Accuracy 不是逐位相同，因为 fixed 结果是独立重新训练的三 seed GPU run；但仍保持原有水平，且 metric fix 没有改动训练目标或 checkpoint selection。Accuracy 的小幅差异属于独立运行的随机性，不是 Macro-F1 公式改变造成的训练路径改变。

每个 seed 的记录如下：

| seed | legacy Val Acc | legacy Test Acc | legacy Test Macro-F1 | fixed Val Acc | fixed Test Acc | fixed Test Macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 88.10% | 88.13% | 77.37% | 88.11% | 88.18% | 76.71% |
| 43 | 88.21% | 88.27% | 77.89% | 88.00% | 88.23% | 76.79% |
| 44 | 88.25% | 88.25% | 70.81% | 87.92% | 87.95% | 76.45% |

fixed 三 seed 的 Macro-F1 为 **76.71/76.79/76.45%**，population std **0.14 个百分点**；legacy 的 seed44 为 **70.81%**，三 seed std 为 **3.22 个百分点**。因此本轮结果强烈支持此前 seed44 大方差主要包含 evaluation label-set artifact：在不改变 MoPF、loss、optimizer 或 validation-accuracy selection 的情况下，固定 task-level label set 后，11 类 Macro-F1 回到稳定范围。由于 fixed 结果是重新训练所得，该结论应表述为有力验证/支持，而非从同一 checkpoint 的严格 counterfactual 证明。

## 论文表格注意事项

旧 ele-fashion baseline 的 Macro-F1 使用 legacy dynamic-label metric。若要制作严格可比的论文最终表格，应统一用 fixed metric 重新产生相关 baseline 结果。本轮没有自动重跑 13 个 baseline；不得把 fixed-MoPF F1 与 legacy-baseline F1 混在同一张正式排名表中并据此标注最佳。

