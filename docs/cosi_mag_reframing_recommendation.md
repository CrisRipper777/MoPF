# CoSI-MAG 论文叙事重构与消融设计建议

## 1. 先给结论

当前论文最需要改变的不是措辞，而是**贡献层级**。

原叙事把 MRC、Semantic Anchor、RCMI 设为三个并列且分别必要的创新模块，但已有代码审计和 5-NC 实验表明：

- 单独关闭 MRC 后，归一化传播算子变化很小；
- `alpha=0` 与带 anchor 的多项式空间基本等价，自由阶数组合可以吸收这种换基；
- uniform multi-order composition 已经非常强，现有 RCMI 的额外增益较小；
- 真正稳定的性能来源是：模态输入适配、规范化图扩散、显式保留 `h0...hK`，以及让分类器同时使用本征语义和多跳上下文。

因此，建议保留 **CoSI-MAG / Continuous Structure–Semantic Interaction** 总名称，但把核心故事改为：

> **CoSI-MAG 将图上下文化建模为一条从 modality-intrinsic semantics 到 multi-hop structural semantics 的显式 context trajectory，而不是将多次传播压缩成单个末层状态。模型在轨迹形成阶段进行模态与关系校准，在轨迹扩展阶段保留本征语义和不同结构半径，在轨迹读取阶段联合利用多个传播阶数。**

这套叙事把已证实有效的强主干放进了方法机制内部，同时不把 Linear、GCN normalization、residual MLP 等成熟算子单独包装成创新。

---

## 2. 当前叙事中需要调整的地方

### 2.1 当前中心命题过度依赖三个弱增量模块

当前中心命题是 structural context 的效用随 relation、depth 和 local relation context 变化，对应 MRC、SMP、RCMI。问题在于，三个现象都成立，并不代表当前三个增量实现都对最终性能具有同等贡献。

尤其需要删除或降级以下表述：

- “移除 MRC、semantic anchor 或 RCMI 均会造成稳定性能下降”；
- “relation context 是最终性能的主要来源”；
- “semantic anchor 提供了普通多阶传播无法表达的能力”；
- “复杂 residual fusion 是模型高性能的重要原因”。

这些表述与现有实验不一致，会让审稿人轻易通过 `unit edges + alpha=0 + uniform order mean` 反例推翻主线。

### 2.2 当前 SMP 定义得过窄

当前 SMP 几乎等同于 restart recurrence。这样一来，`w/o Semantic Anchor` 只把 `alpha` 设为 0，却仍保留完整 `h0...hK` 状态库。由于 anchored basis 与 ordinary monomial basis 在 `alpha<1` 时存在可逆变换，这个消融没有真正移除多阶表达能力。

建议把“语义保持”从一个标量 restart 扩展为一个完整功能：

> **显式保留从 `h0` 到 `hK` 的语义—结构上下文轨迹，使本征语义不会只能依靠末层残差被间接恢复。**

Anchor 只是该功能内部的稳定器，不再承担一个独立主要贡献。

### 2.3 当前 RCMI 定义得过窄又过强

当前 `w/o RCMI` 用 uniform mean 代替 adaptive composition，但 uniform mean 本身已经使用了完整 `h0...hK`。因此该消融测试的是“复杂自适应读取比简单读取强多少”，并没有测试“多阶上下文读取是否必要”。

建议区分：

1. **Multi-order trajectory composition**：是否保留并联合读取多个阶数；
2. **Adaptive composition refinement**：是否使用全局、模态、节点和 relation-conditioned 系数。

前者是强而稳定的核心机制；后者是精细校准。

---

## 3. 从相关论文中应学习的包装方式

### 3.1 OptiMAG：先冻结一个冲突，再提出满足明确条件的机制

OptiMAG 先把问题压缩成 explicit graph 与 modality-induced semantic structure 的 **structural-semantic conflict**，随后提出三个设计要求，再用 UOT 同时回应这些要求。它的贡献分成 problem formalization、method 和 validation，而不是把 cosine、PPR、KL、Sinkhorn 分别称作创新。[OptiMAG 原文](https://arxiv.org/html/2601.22856)

CoSI-MAG 应学习这种层级：创新应是“显式建模 context trajectory 及其连续控制”，而不是 diagonal cosine、restart、attention 和 residual MLP 的并列组合。

### 3.2 RoleMAG：经验观察与功能路径一一对应

RoleMAG 先用专门的 empirical study 验证三种邻居角色，再让 shared、complementary、heterophily 三条传播路径分别对应一个观察。其消融不是任意关掉一个操作，而是构造 Shared-only、w/o role routing、w/o complementary expert 等会破坏一项设计原则的变体。[RoleMAG 原文](https://arxiv.org/html/2604.12271)

CoSI-MAG 应采用同样的 “observation → representation object → functional path → ablation” 结构：

- 图上下文有用，但不能只使用单一末层；
- `h0...hK` 构成从本征语义到结构语义的轨迹；
- trajectory construction 和 trajectory composition 分别成为可以整体移除的功能路径。

### 3.3 NSG-MoE：承认结构核心本身很强，把高级模块作为扩展

NSG-MoE 把 node splitting 和 graph rewiring 作为结构核心，把 MoE 作为增强模块。它明确报告没有 MoE 的 NSG 仍然很强，并据此说明结构重新定义本身有效，而不是试图隐藏强基础结构。[NSG-MoE 原文](https://arxiv.org/html/2602.00067)

这与 CoSI-MAG 当前情况最相似。建议明确承认 plain context trajectory 已经很强，并将 MRC、anchor 和 relation-conditioned interaction 描述为对该核心表示的校准。这样反而更可信。

### 3.4 SMGFM：围绕一个“不可逆混合前要保留语义角色”的中心原则

SMGFM 的写法不是罗列 Chebyshev filter、router 和 losses，而是围绕“structure-induced semantics 与 modality-intrinsic semantics 在融合前应被区分”这一原则组织。其 ablation 还分别检查 binding、alignment、preservation 对应的诊断量。[SMGFM 原文](https://arxiv.org/html/2606.12867)

CoSI-MAG 可以形成空间域中的互补定位：SMGFM 在 graph-frequency domain 中区分语义角色；CoSI-MAG 在 propagation-order domain 中显式保留从 intrinsic 到 structural context 的轨迹。

### 3.5 DiP：消融的是完整信息路径

DiP 将方法压缩为 intra-modal diffusion pathway 与 inter-modal aggregation pathway，其 ablation 直接移除 Local、Global、pseudo nodes 或跨模态 pseudo-node communication，因此每一行都破坏一条完整信息路径，而不是只修改一个容易被补偿的标量。[DiP 原文](https://arxiv.org/html/2603.09258)

CoSI-MAG 的主消融也应移除“图上下文扩展路径”或“多阶轨迹读取路径”，而不是只测试 `alpha=0`。

### 3.6 GMoPE：挑战与机制数量严格对应

GMoPE 的组织方式是先明确跨域适配中的少数几个挑战，再让 prompt experts、structure-aware routing 和 anti-collapse constraint 对应这些挑战。其可借鉴点是 challenge-to-mechanism mapping，而不是 MoE 本身。[GMoPE 原文](https://arxiv.org/html/2511.03251)

对 CCF-C 级会议，CoSI-MAG 最好只保留两个核心挑战和两个核心功能阶段，避免三个大模块下再分十余个子机制。

---

## 4. 推荐的新中心命题

### 4.1 一句话问题定义

> **Existing multimodal graph models often treat graph contextualization as a terminal transformation: repeated propagation is compressed into a final state, making modality-intrinsic semantics and contexts at different structural radii no longer directly accessible to downstream prediction.**

中文含义：现有模型通常把图上下文化视为得到末层表示的过程；经过多次传播后，节点自身语义和不同结构半径的信息被压缩到单个最终状态中。

这比“结构上下文在三个阶段中持续变化”更具体，也能被现有实验直接验证。

### 4.2 一句话方法思想

> **CoSI-MAG represents graph contextualization as an explicit modality-conditioned context trajectory, calibrates how the trajectory is formed on the physical graph, retains intrinsic-to-multi-hop states, and composes the trajectory rather than reading only its endpoint.**

### 4.3 与主要基线的清晰区别

| 方法 | 主要表示对象 | 是否显式保留 `h0...hK` | 论文定位 |
|---|---|---:|---|
| MMGCN | 两个模态的串行末层状态 | 否 | 模态分支上的普通串行传播 |
| DiP | pseudo-node 动态路径的最终节点状态 | 否 | 自适应局部/全局路由 |
| OptiMAG | 结构与语义空间的对齐正则 | 否 | 训练期结构—语义对齐 |
| RoleMAG | shared/complementary/heterophilous neighbor roles | 否 | 邻居参与方式建模 |
| SMGFM | graph-frequency band tokens | 是另一种分解 | 频域语义角色分解与预训练 |
| **CoSI-MAG** | modality-specific intrinsic-to-structural context trajectory | **是** | 固定 support 上的多阶轨迹形成与读取 |

建议不要再把“我们也是双流传播”作为主要差异。现有 Early Fusion 实验表明，只要模态专属投影保留，后续双路扩散本身没有稳定优势。

---

## 5. 推荐的方法包装

建议正文采用两个核心阶段，另加标准预测头。这样比三个平级缩写更集中。

## 5.1 Stage I：Modality-Calibrated Context Trajectory Construction

可用缩写 **MCTC**，中文为“模态校准的上下文轨迹构造”。该阶段包含：

1. text/image 各自的 projection、normalization 和 nonlinear adaptation；
2. fixed physical support 上的 modality-specific relation calibration；
3. self-loop symmetric normalization；
4. 从 `S0=H0` 到 `SK` 的逐阶传播；
5. semantic restart 作为轨迹稳定器；
6. 显式输出整个 `S0...SK`，而不是只输出 `SK`。

该阶段的主要对象是：

\[
\mathcal T_i^m=(S_{i,0}^m,S_{i,1}^m,\ldots,S_{i,K}^m),
\]

称为 **modality-specific context trajectory**。

这里应明确：

- projection、cosine、GCN normalization 和 restart 都不是单独的新算子；
- 创新点在于把它们组织成一个保留 intrinsic-to-structural continuum 的表示接口；
- MRC 控制 trajectory 在哪些关系上形成；
- restart 控制 trajectory 远端状态偏离 `h0` 的速度；
- `h0...hK` 的显式保留是该阶段最关键的输出契约。

## 5.2 Stage II：Multi-Order Context Trajectory Composition

可用缩写 **MCTC-Readout** 或更简洁的 **MOC**，中文为“多阶上下文轨迹组合”。该阶段包含：

1. 把不同阶状态作为具有 order identity 的 context tokens；
2. 建模不同阶状态的互补与冗余；
3. 从 global、modality 和 node 三个层次生成组合系数；
4. relation context 作为一个轻量条件变量修正读取过程；
5. 组合原始轨迹状态并进行模态内 refinement；
6. 最后进行 late multimodal fusion。

最重要的叙事变化是：

> **该阶段的核心功能是“读取完整轨迹”，relation-conditioned attention 是读取器内部的一种校准机制。**

因此，uniform mean 是 MOC 的简单版本，当前 RCMI 是其 adaptive version。这样即使 adaptive version 的增益不大，论文的主要机制仍由 Last-Hop 实验有力支撑。

## 5.3 Late Fusion 的地位

Late residual fusion 保留为实现细节，不列为主要贡献。Simple Fusion 和 Early Fusion 的结果说明复杂融合不是主体来源。正文只需说明在获得两个模态的 trajectory-aware representations 后使用标准 residual fusion。

---

## 6. 推荐的 Introduction 故事线

### Paragraph 1：MAG 中存在两类非等价信息

图拓扑提供结构上下文，预训练文本/视觉特征提供节点本征语义。二者互补，但反复图传播会逐渐把前者写入后者。

### Paragraph 2：指出“terminal-state contextualization”假设

多数方法虽然采用模态分支、attention、rewiring 或 dynamic routing，最终仍把传播过程压缩成一个末层状态。这样会导致：

- `h0` 的强预训练语义只能通过残差间接保留；
- 不同结构半径的上下文无法被下游头直接比较；
- 节点与模态之间的最佳传播半径差异难以利用。

### Paragraph 3：给出三条经验观察，但重新排序

建议 Figure 1 改为：

1. **Graph context is useful**：`h0-only` 明显弱于完整轨迹，说明不能放弃图；
2. **The endpoint is insufficient**：`hK-only` 明显弱于完整轨迹，说明不能只保留末阶；
3. **Trajectory formation is conditional**：edge semantic discrepancy 和 preferred-order heterogeneity 表明轨迹还需按模态、节点和关系环境校准。

前两项直接对应当前最强实验结果，第三项再引出 MRC 和 adaptive readout。

### Paragraph 4：提出 Context Trajectory 视角

图上下文化不是从输入到末层的黑盒变换，而是一条从 intrinsic semantics 到 increasingly structural contexts 的轨迹。理想模型需要同时控制其形成和读取。

### Paragraph 5：引出两个阶段

- MCTC 构建并保留 modality-specific context trajectory；
- MOC 联合读取轨迹中的多个阶数，并用关系环境进行轻量校准。

### Paragraph 6：贡献

建议改为两项方法贡献加一项实验贡献：

1. **Context-trajectory perspective.** 将 MAG 图上下文化重新表述为从本征语义到多跳结构语义的显式轨迹，指出仅使用末层状态会不可逆地压缩不同结构半径的信息。
2. **Trajectory construction and composition.** 提出在固定 physical support 上进行模态校准的轨迹构造，并通过 multi-order composition 联合利用 intrinsic-to-structural contexts；relation calibration、semantic restart 和 relation-conditioned interaction 分别控制轨迹形成、漂移和读取。
3. **Trajectory-oriented evaluation.** 通过 intrinsic-only、terminal-only、order-bank、relation/operator 和 composition controls，验证图上下文、轨迹保留及自适应校准分别贡献了什么。

---

## 7. 重新设计消融：先测试功能，再测试内部实现

### 7.1 正文主消融

正文只放能回答核心故事的功能级变体：

| 变体 | 精确定义 | 回答的问题 | 当前证据 |
|---|---|---|---|
| Full CoSI-MAG | 完整模型 | 总体效果 | benchmark |
| Intrinsic-only | 只使用 `h0`，保留相同 projection/refinement/fusion/head | 图上下文是否需要 | 平均 −3.364 Acc / −4.959 F1，相对 All Plain |
| Terminal-only | 只使用 `hK`，相同深度和 operator | 显式轨迹是否优于末层状态 | 平均 −2.477 / −4.099 |
| Plain Trajectory | unit edges、`alpha=0`、`h0...hK` uniform composition | 高级校准之外，轨迹核心能保留多少性能 | 相对 Full 仅小幅下降 |
| Shared-input Trajectory | 原始模态先融合，再用参数量匹配的单一 projector 和单路轨迹 | 模态专属输入适配是否必要 | **需要新增实验** |

其中前两个变体已经提供真实且明显的下降，是新故事最关键的证据。

`Early Fusion` 现有版本保留了两个独立 projector，因此不能用于声称“模态专属初始化无效”；它只能说明投影后的两条线性图扩散路径可以交换和合并。

### 7.2 Appendix 或第二张小表：轨迹构造内部消融

建议新增：

1. `w/o h0 in trajectory`：只组合 `h1...hK`；
2. `w/o self-loop`：测试每阶传播中的本节点保留；
3. `row-mean operator`：把 symmetric normalization 换成 MMGCN 风格 mean aggregation；
4. `shared unit operator`：现有 w/o MRC；
5. `raw cosine operator` 与 `learned diagonal cosine operator`；
6. `alpha=0` 与 `alpha=0.1`，名称写作 `w/o restart`，不要写 `w/o SMP`。

这些实验区分“轨迹本身的价值”和“轨迹如何形成”。

### 7.3 Appendix 或第三张小表：轨迹读取阶梯

建议使用逐步增加能力的 readout ladder：

1. `hK only`；
2. uniform `mean(h0...hK)`；
3. learned global order coefficients；
4. global + modality-specific coefficients；
5. node-adaptive coefficients without cross-order interaction；
6. cross-order interaction without relation context；
7. full relation-conditioned composition。

这张表将诚实展示：大增益来自从 single endpoint 到 multi-order bank，后续 adaptive mechanisms 提供较小、数据集相关的改进。

### 7.4 不建议继续采用的消融名称

- 不要把 `alpha=0` 称为 `w/o semantic-preserving propagation`；
- 不要把 uniform mean 称为完全 `w/o multi-order utilization`；
- 不要把 unit edges 称为 `w/o modality-aware contextualization`，因为两个模态 projection 和状态流仍然保留；
- 不要为了得到更大下降而同时删掉无关的 hidden dimension、训练轮次或分类头容量。

消融变体的名称必须准确描述被破坏的**功能**。下降应来自信息路径被移除，而不是容量被任意削弱。

---

## 8. 推荐的实验执行顺序

### 第一优先级：确认新主线

在 5 个 NC 数据集、seeds 42/43/44 上运行：

1. Full；
2. Intrinsic-only；
3. Terminal-only；
4. `h1...hK` without `h0`；
5. Plain Trajectory；
6. Shared-input Trajectory。

目标是确认三个命题：图有用、末层不够、模态输入适配是否必要。

Full、Intrinsic-only、Terminal-only 和 Plain Trajectory 已有可复用结果；第一轮只需新增 `h1...hK without h0` 与 Shared-input Trajectory，共 **2 variants × 5 datasets × 3 seeds = 30 次 NC 训练**。建议先完成这 30 次，再决定是否扩展读取器阶梯；此阶段无需运行 LP。

### 第二优先级：拆解 readout

运行 global、modality、node、interaction、relation 五级读取器。若 full relation conditioning 仍无稳定增益，就把 relation conditioning 降为实现细节或机制分析，不把它列入摘要贡献。

### 第三优先级：解释为什么超过 MMGCN/DiP

增加两个 matched controls：

1. 使用 CoSI projector 和 symmetric operator，但只输出末层；
2. 使用相同参数预算的两层 learnable GCN，但显式 concat/mean 每层状态。

这样可以区分增益来自 operator、参数化方式，还是显式 order retention。

### 统计报告

- 固定 split 下至少报告 paired seeds；
- 当前 `n=3` 主要作描述性比较；
- 正文报告均值和标准差，附录给逐 seed 差值；
- 不要求所有数据集每个小机制都下降；主机制应表现出跨数据集稳定趋势。

---

## 9. 推荐的 Figure 与 Table 结构

### Figure 1：Why a context trajectory?

- (a) Intrinsic-only vs full trajectory：graph context utility；
- (b) Terminal-only vs full trajectory：endpoint insufficiency；
- (c) preferred order / modality discrepancy：conditional trajectory utilization。

### Figure 2：Context trajectory framework

绘制每个模态的横向轨迹：

\[
H_0^m\rightarrow S_1^m\rightarrow S_2^m\rightarrow S_3^m,
\]

并突出：

- relation calibration 作用于边和 operator；
- semantic restart 贯穿轨迹扩展；
- 所有状态均进入 trajectory composer；
- 最终再进行 multimodal fusion。

视觉重点应是“整条轨迹被保留并读取”，而不是三个独立方框。

### Table 1：总体性能

保持 NC/LP benchmark。

### Table 2：Functional ablation

Full、Intrinsic-only、Terminal-only、Plain Trajectory、Shared-input Trajectory。

### Table 3 或 Appendix：Component refinement

MRC、restart、global/modality/node/relation-conditioned composer 的细分。

### Figure 3：Mechanism evidence

优先保留能直接服务新主线的分析：

- 每阶对 `h0` 的 drift；
- terminal-only 的少数类或节点分层损失；
- 不同节点的最优阶数分布；
- learned order weights 与局部同质性/度/置信度的关系。

当前 relation-shuffle 若仍几乎不改变预测，应移到 Appendix 或删除，避免正文用弱路径支撑强 claim。

---

## 10. 标题候选

优先推荐：

1. **CoSI-MAG: Modality-Calibrated Context Trajectories for Multimodal Attributed Graphs**
2. **CoSI-MAG: Preserving and Composing Multi-Order Context in Multimodal Attributed Graphs**
3. **CoSI-MAG: Continuous Structure–Semantic Contextualization over Propagation Orders**

若希望保留现有标题，可以使用：

> **CoSI-MAG: Continuous Structure–Semantic Interaction for Multimodal Attributed Graph Learning**

但摘要第一段必须立刻把 continuous 解释为“context trajectory formation and composition”，不要继续把它主要解释成 MRC→SMP→RCMI 三个小模块的连续调用。

---

## 11. 最终建议

### 建议采用的低风险方案

保持当前 Full 模型和 benchmark 不变，仅重构贡献层级：

- context trajectory 是核心表示；
- MRC、restart、RCMI 是 formation/composition 中的 refinement；
- 主消融使用 Intrinsic-only 和 Terminal-only；
- micro ablation 诚实报告 refinements 的小幅、数据集相关增益。

这一方案与现有结果一致，不需要为了故事重新选择 checkpoint，也不会被审稿人通过 All Plain 反例推翻。

### 不建议的方案

继续把三个旧模块包装为同等必要，然后把 projection、multi-order bank 或 fusion 一并从 `w/o MRC / w/o RCMI` 中删除，以制造更大下降。这会让消融定义与模块名称不一致，容易被认为是 capacity ablation，而不是 mechanism ablation。

### 如果坚持让 relation conditioning 成为主要创新

则需要修改模型，使 relation context 直接、足量地影响最终 composition，例如直接进入 `eta` 或让 interacted states 参与最终组合，并重新运行全部 benchmark。此时应视为新模型版本，不能继续沿用当前 frozen Full 结果。

当前最稳妥且最有说服力的论文故事是：

> **Graph context is useful, but its endpoint is insufficient. CoSI-MAG preserves the entire modality-conditioned context trajectory and composes intrinsic and multi-hop structural semantics for downstream prediction.**
