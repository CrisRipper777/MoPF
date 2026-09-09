# MoPF Mechanism Audit (M0)

本审计只做 mechanism diagnosis 和 evidence export。五个 NC 数据集均使用 seed=42，并从该 run 的 best Validation Accuracy checkpoint 导出；Grocery 使用上一阶段 validation-selected K=2，其余数据集使用正式 K=3。没有修改 MoPF forward、task loss、optimizer、semantic graph、fusion、checkpoint selection 或默认 K，也没有实现任何 regularizer。

所有原始输出位于 [`outputs/mechanism_audit`](../outputs/mechanism_audit)。每个 `dataset/seed42/` 目录包含：

- `coefficients.pt`：raw `gamma_global`、modality residual、node residual、`gamma_text/visual`、`eta_text/visual` 和 effective radius；
- `distribution_summary.csv`：逐 modality/逐 order 的 mean、population std、min/max、q10/q25/median/q75/q90、mean absolute value；
- `node_profiles.csv`：每个节点的 degree、local label homophily、完整 eta/delta-node profiles；
- `modality_summary.csv`：modality-level coefficients、profile distance 和 effective-radius 汇总；
- `structure_profile_summary.csv`：degree 与 homophily rank-quartile 下的 profile means/std；
- `audit_summary.json`：checkpoint、配置、核心统计和 correlation 摘要。

## 1. Observations

### Checkpoint and task configuration

| Dataset | K | Best epoch | Val Acc | Test Acc | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Movies | 3 | 63 | 57.14% | 56.10% | 50.17% |
| Toys | 3 | 60 | 80.38% | 79.51% | 76.71% |
| Grocery | 2 | 88 | 83.84% | 82.90% | 74.92% |
| ele-fashion | 3 | 135 | 88.13% | 88.36% | 77.59% |
| Reddit-S | 3 | 62 | 96.23% | 96.38% | 92.36% |

这些是机制审计 run 的 checkpoint reporting，不是新的三-seed benchmark。所有配置都记录了 full-graph NC、AdamW、`lr=1e-3`、`weight_decay=1e-4`、300 epochs、patience=30 和 validation-accuracy selection。

### Core coefficient observations

`gamma_text[k] = gamma_global[k] + delta_gamma_text[k]`，`gamma_visual[k] = gamma_global[k] + delta_gamma_visual[k]`；`eta` 再加上对应的 node residual。五个数据集的 `gamma_global`、modality-level profile 和所有节点张量均保存在 `coefficients.pt`，没有做 rescale。

global profile 在各数据集都不是全零；text/visual 的 modality corrections 也都不是全零。它们不是简单的同一共享 profile。例如：

- Movies：`gamma_text=[0.2620, 0.0573, 0.6083, -0.1185]`，`gamma_visual=[0.2388, 0.1109, 0.6420, -0.0988]`；
- Grocery K=2：`gamma_text=[0.2448, 0.0502, 0.6310]`，`gamma_visual=[0.1604, 0.0482, 0.7258]`；
- ele-fashion：`gamma_text=[0.2700, 0.0192, 0.5826, -0.1237]`，`gamma_visual=[0.2780, 0.0210, 0.5649, -0.1406]`。

## 2. Dataset comparison

### Modality profile distance

这里的 distance 是每个节点 `||eta_text[i,:] - eta_visual[i,:]||_2`，只作为 analysis statistic，不进入模型或 loss。

| Dataset | mean | std | q25 | median | q75 |
|---|---:|---:|---:|---:|---:|
| Movies | 0.13463 | 0.01716 | 0.12221 | 0.13234 | 0.14445 |
| Toys | 0.14893 | 0.00134 | 0.14818 | 0.14919 | 0.14990 |
| Grocery K=2 | 0.23443 | 0.00436 | 0.23153 | 0.23336 | 0.23696 |
| ele-fashion | 0.06189 | 0.01117 | 0.05382 | 0.05767 | 0.06568 |
| Reddit-S | 0.04716 | 0.00440 | 0.04503 | 0.04845 | 0.05039 |

Grocery 的 text/visual coefficient profile separation最大，Reddit-S 最小；Toys 的 distance mean 不小但 dispersion 很小，说明其节点间 text/visual separation 近似共享。distance 大小本身不被解释为模型更好。

### Effective radius

effective radius 是 analysis-only statistic：`sum_k k*abs(eta_i,k) / sum_k abs(eta_i,k)`。下表为 mean ± population std，括号为 q25–q75。

| Dataset | Text radius | Visual radius |
|---|---:|---:|
| Movies | 1.5507 ± 0.0012 (1.5504–1.5515) | 1.5571 ± 0.0281 (1.5378–1.5666) |
| Toys | 1.5429 ± 0.0017 (1.5419–1.5443) | 1.7258 ± 0.0013 (1.7250–1.7268) |
| Grocery K=2 | 1.2429 ± 0.0025 (1.2413–1.2435) | 1.6043 ± 0.0114 (1.5952–1.6117) |
| ele-fashion | 1.5259 ± 0.0079 (1.5183–1.5331) | 1.5240 ± 0.0079 (1.5180–1.5305) |
| Reddit-S | 1.5342 ± 0.0068 (1.5329–1.5378) | 1.5635 ± 0.0026 (1.5628–1.5632) |

## 3. Hierarchical identifiability

node residual 的 mean 与 std 是识别层级分工的关键。由于 global 和 modality residual 对所有节点都是常数，`std_i(eta_i,k^m) = std_i(delta_node_i,k^m)`；因此 eta 的节点 std 不会被常数项人为放大。

最高阶的代表性结果如下。ratio 为 `abs(mean(delta_node))/abs(delta_gamma_m)`，用于比较 node shared offset 与 modality correction 的相对大小，不是显著性检验。

| Dataset/order | modality | delta_gamma | mean(delta_node) | std(delta_node) | ratio |
|---|---|---:|---:|---:|---:|
| Movies/3 | text | -0.06164 | -0.05854 | 0.00092 | 0.95 |
| Movies/3 | visual | -0.04197 | -0.04478 | 0.00460 | 1.07 |
| Toys/3 | text | -0.02436 | -0.02531 | 0.00085 | 1.04 |
| Toys/3 | visual | 0.02158 | 0.01154 | 0.00041 | 0.53 |
| Grocery/2 | text | -0.08388 | -0.08710 | 0.00300 | 1.04 |
| Grocery/2 | visual | 0.01089 | -0.00419 | 0.00024 | 0.38 |
| ele-fashion/3 | text | -0.05837 | -0.06576 | 0.00070 | 1.13 |
| ele-fashion/3 | visual | -0.07526 | -0.08016 | 0.00050 | 1.07 |
| Reddit-S/3 | text | -0.05000 | -0.04619 | 0.00582 | 0.92 |
| Reddit-S/3 | visual | -0.03709 | -0.03514 | 0.00030 | 0.95 |

这说明 node residual 在多个 dataset/modality/order 中含有与 modality correction 同量级的 shared offset。严格表述为：**node residual contains a strong shared offset**；这说明 hierarchical parameterization 存在 identifiability pressure，但单凭该统计不能宣称实现是 bug。

## 4. Node personalization

`node_profiles.csv` 导出了每个节点完整的 `eta_text`、`eta_visual`、`delta_node_text` 和 `delta_node_visual` profile。下表报告每个 dataset、每个 modality 跨 order 的最大 `std_i(delta_node)`；相同数值也适用于 `eta` 的 std。

| Dataset | Text max std (order) | Visual max std (order) |
|---|---:|---:|
| Movies | 0.00092 (3) | 0.02064 (0) |
| Toys | 0.00154 (1) | 0.00060 (1) |
| Grocery K=2 | 0.00300 (2) | 0.00630 (0) |
| ele-fashion | 0.01616 (1) | 0.02956 (1) |
| Reddit-S | 0.00582 (3) | 0.00175 (0) |

因此 node-wise personalization **确实存在**，但不是所有 dataset/order 都同样强：ele-fashion order1、Movies visual order0、Grocery visual order0 和 Reddit-S text order3 有明显的节点差异；Toys 整体以及 Movies text 多数 order 更接近 shared correction。结论是“heterogeneous and dataset-dependent”，不是“所有节点都高度个性化”。

## 5. Modality personalization

每个 dataset 的 `gamma_text` 和 `gamma_visual` 已保存并在上文给出代表例；per-node profile distance 的完整分布在 `modality_summary.csv`。结果显示：

- Movies visual 的 node-level profile variation 明显高于 text；
- ele-fashion 两个 modality 在 order1 都有较强 node variability，但 mean profile 仍相近；
- Grocery 的 modality-level order2 差异很大，且 visual 的 effective radius 高于 text；
- Toys 的 text/visual distance 的 std 仅 0.00134，说明跨节点 profile separation 很稳定；
- Reddit-S 平均 profile distance 最小，但 text order3 仍有比 visual 更强的 node variability。

这些是 descriptive profile differences，不证明某个 modality 或 distance 更优。

## 6. High-order behavior

### Highest-order coefficients

| Dataset/order | gamma_global[K] | delta_gamma_text[K] | delta_gamma_visual[K] | text delta mean/std | visual delta mean/std |
|---|---:|---:|---:|---:|---:|
| Movies/3 | -0.05686 | -0.06164 | -0.04197 | -0.05854 / 0.00092 | -0.04478 / 0.00460 |
| Toys/3 | 0.01379 | -0.02436 | 0.02158 | -0.02531 / 0.00085 | 0.01154 / 0.00041 |
| Grocery/2 | 0.71490 | -0.08388 | 0.01089 | -0.08710 / 0.00300 | -0.00419 / 0.00024 |
| ele-fashion/3 | -0.06532 | -0.05837 | -0.07526 | -0.06576 / 0.00070 | -0.08016 / 0.00050 |
| Reddit-S/3 | -0.03948 | -0.05000 | -0.03709 | -0.04619 / 0.00582 | -0.03514 / 0.00030 |

在 K=3 的 Movies、ele-fashion、Reddit-S 中，最高阶 global coefficient 明显不为 0，modality corrections 也明显不为 0；Toys 的 global K coefficient 接近 0，但 modality/node corrections 仍不为 0。Grocery 使用 K=2，order2 是其最高阶且接近 MAP prior 的主阶，不能与 K=3 的最高阶直接按“应为 0”比较。

因此当前证据不支持直接删掉最高阶。它可能包含真实高阶传播，也可能部分承担 modality-level correction；仅凭单个 best checkpoint 无法区分“有用高阶项”和“无必要漂移”。

## 7. Potential degeneracies

1. **Node residual shared-offset degeneracy.** 多个最高阶及其它 order 中，`abs(mean(delta_node))` 与 `abs(delta_gamma)` 同量级，而 node std 很小。node layer 可能同时承担 personalization 和 modality correction。
2. **Profile near-collapse in subsets.** Toys 的 node std 普遍较小，Movies text 也很小；这与 ele-fashion order1 和 Movies visual 的明显 variability 并存。
3. **High-order nonzero drift candidate.** K=3 多数数据集的 K coefficient 非零，但本审计没有训练轨迹、跨 seed 或 counterfactual K=3/去高阶性能证据，不能把非零直接称为 unnecessary drift。
4. **Structure dependence is inconsistent.** degree quartile 的 eta group range 最大约为 Movies visual 0.00789、ele-fashion visual 0.01458；homophily quartile 最大约为 ele-fashion visual 0.01085、Reddit-S text 0.00643。相关方向跨 dataset/order 改变，不能作统一因果解释。

`structure_profile_summary.csv` 同时给出 degree/homophily quartile 下每个 modality/order 的 eta 与 delta-node mean/std，以及 `audit_summary.json` 中的 correlation。分组使用 rank quartiles；local label homophily 只使用有效标签的连接邻居，未作为模型输入。

## 8. Evidence-supported regularization candidates

### A. node-residual centering — **Supported**

支持证据是 shared offset 的重复出现：在 Movies、Toys、Grocery text、ele-fashion 和 Reddit-S 的最高阶 text/visual 中，node residual mean 与 modality residual 大致同量级，而 std 往往只有 0.0002–0.0058；Movies visual/Grocery visual 等处则存在同时可观测的 node variability。也就是说，centering 有明确的 identifiability 目标：将 shared offset 留给 modality/global 层，让 node 层更专注于 residual personalization。

“Supported”只表示该机制候选有统计动机，不表示尚可直接加入训练；下一阶段仍需受控 ablation 验证是否改善性能或稳定性。

### B. weak global-prior anchoring — **Weakly supported**

Movies、ele-fashion、Reddit-S 的 K=3 `gamma_global[K]` 分别为 -0.05686、-0.06532、-0.03948，且对应 modality corrections 也非零；这说明 learned filter 已明显偏离初始 MAP prior 的零高阶项。另一方面，Toys 的 global K coefficient 仅 0.01379，Grocery K=2 的 0.71490 是最高主阶，不应作为“漂移”证据。当前只有单个 best checkpoint，无法证明非零高阶是不必要的，也没有证明 anchoring 会带来收益，因此只评为 Weakly supported。

### C. profile anti-collapse regularization — **Weakly supported**

描述性证据支持“部分 profile 接近 collapse”：Toys 的最大 node std 仅 text 0.00154、visual 0.00060，Movies text 最大仅 0.00092；但 ele-fashion order1 的 text/visual std 达到 0.01616/0.02956，Movies visual order0 达到 0.02064。当前证据说明 diversity 是 dataset/order-dependent，而没有说明低 diversity 一定损害性能；人为推高 profile distance 还可能破坏合法的 shared filter。因此只能作为弱候选，不能据此实现正则项。

## 9. Things NOT supported by current evidence

- 不支持宣称 node residual 非零就等于真实 node personalization；shared mean/std 对比显示其中相当部分可能是 shared offset。
- 不支持直接删除 K=3 最高阶或把所有高阶非零判定为 unnecessary drift。
- 不支持把 modality profile distance 大解释为更好的模型或更强的机制。
- 不支持 degree/homophily 与传播 profile 之间的统一因果结论；相关方向跨 dataset/order 不一致。
- 不支持仅凭 seed42、单一 best checkpoint 决定最终 regularization；A/B/C 都需要后续受控 ablation、保持相同 checkpoint protocol，并至少扩展到三 seed。
- 本轮没有实现任何 regularizer，也没有修改默认 K、MoPF forward、loss 或 baseline。

