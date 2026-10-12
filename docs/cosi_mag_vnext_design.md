# CoSI-MAG VNext Design

## Research question and architecture

CoSI-MAG VNext asks whether a multimodal attributed graph encoder can use agreement and disagreement between text and visual semantics to calibrate the observed physical relations, then use that local relation context to choose how each modality combines graph orders. The model keeps text and visual representations separate through graph filtering and order interaction, and fuses them only at the end:

`Text/Visual Projection → CRC → MMF → RCMI → Late Fusion`

CRC (Cross-Modal Relation Context) assigns bounded positive weights to existing physical edges and summarizes each node's local relation evidence. MMF (Modality-Conditioned Multi-Order Filtering) builds a fixed Chebyshev bank over each weighted normalized adjacency. RCMI (Relation-Conditioned Multi-Order Interaction) attends across order tokens and combines order differences through node-conditioned signed coefficients. Late fusion preserves the CoSI concatenation residual MLP.

The graph is not reconstructed: CRC never creates a candidate edge, deletes a non-self physical edge, or reads labels. This is a transductive encoder over the graph supplied to `forward`. For LP that graph is the sampled train-only message graph after positive-edge masking.

## Notation and input projection

Let `N` be the number of nodes, `E` the number of directed physical edge records, and `d=256` the hidden width. The repository's feature order is `[text, visual]`.

Independent projection MLPs preserve the existing CoSI input projection:

\[
H^0_t=\operatorname{Dropout}(\operatorname{ReLU}(\operatorname{Norm}(X_tW_t+b_t))),\qquad
H^0_v=\operatorname{Dropout}(\operatorname{ReLU}(\operatorname{Norm}(X_vW_v+b_v))).
\]

Each output is `[N, 256]`. Text and visual projection parameters are independent.

## Stage I: Cross-Modal Relation Context (CRC)

For each supplied non-self edge `(i,j)`, compute cosine similarities in the projected modality spaces:

\[
s^t_{ij}=\cos(H^0_t[i],H^0_t[j]),\quad
s^v_{ij}=\cos(H^0_v[i],H^0_v[j]),
\]
\[
e_{ij}=[s^t_{ij},s^v_{ij},|s^t_{ij}-s^v_{ij}|,s^t_{ij}s^v_{ij}].
\]

A shared scorer `Linear(4,32) → GELU → Linear(32,32) → GELU → Linear(32,2)` produces `(a,b)`. Its last layer is zero initialized. The edge weights are

\[
w^t_{ij}=1+0.3\tanh(a+b),\qquad w^v_{ij}=1+0.3\tanh(a-b).
\]

Thus the initial physical-edge weights equal one, and finite scores give weights in `[0.7,1.3]`, bounded strictly above zero. The evidence and shared scorer are invariant to swapping `i,j`, so reciprocal records receive the same score. Edge weights remain in the autograd graph.

For each modality, PyG `gcn_norm` constructs the source-to-target COO form of

\[
P_m=D_m^{-1/2}(A\odot W_m+I)D_m^{-1/2}.
\]

Input self-loops are removed before scoring and exactly one self-loop per node is inserted for propagation. Self-loops do not contribute to relation descriptors. On symmetric physical graphs the resulting operator is symmetric; directed inputs do not receive an extra symmetrization step. Repeated directed records retain multiedge semantics inside this module; the standard repository loaders preprocess and coalesce their graphs before the model sees them.

The eight local descriptor values for node `i` are outgoing-neighborhood means of `s_text`, `s_visual`, `abs(s_text-s_visual)`, `w_text`, and `w_visual`; population standard deviations of the two weights; and `log1p(degree)`. On an undirected graph, this is the usual local neighborhood. Empty neighborhoods have an all-zero descriptor. Standard deviations use centered sums of squares with a small variance floor, avoiding an infinite derivative at zero variance. A small MLP followed by LayerNorm maps the descriptor to `q_i ∈ R^32`. There is no graph-wide z-score and no `detach()` in the training path.

CRC returns `P_text`, `P_visual`, `q`, and physical-edge statistics. `Model.relation_diagnostics(x, edge_index)` exports compact cosine, edge-weight, coefficient, gate, and structural-residual summaries without labels.

## Stage II: Modality-Conditioned Multi-Order Filtering (MMF)

For each modality, use the normalized adjacency `P_m` directly:

\[
B^m_0=H^0_m,\qquad B^m_1=P_mH^0_m,\qquad
B^m_k=2P_mB^m_{k-1}-B^m_{k-2}.
\]

The default maximum order is `K=3`, so `bank_m=stack(B^m_0,…,B^m_3)` has shape `[N,4,256]`. This is the Chebyshev recurrence on `P`; it is not PyG `ChebConv`'s scaled-Laplacian recurrence. It needs no explicit eigendecomposition. A dense-matrix reference test checks the sparse recurrence on a small graph.

This fixed Chebyshev bank is a chosen basis. It is not claimed to be an optimal or signal-adaptive basis.

## Stage III: Relation-Conditioned Multi-Order Interaction (RCMI)

Each modality has its own order embedding `E_m ∈ R^{4×256}`. Shared LayerNorm, query and key projections (`256→64`), and value projection (`256→256`) are used across both modalities. For order tokens, only Q/K receive order embeddings:

\[
Q^m=\operatorname{LN}(bank_m+E_m)W_Q,\quad
K^m=\operatorname{LN}(bank_m+E_m)W_K,\quad
V^m=bank_mW_V.
\]

The relation-order bias has separate per-modality maps `Linear(32,4)` and is broadcast over query orders:

\[
L^m_{k\ell}=Q^m_k(K^m_\ell)^T/\sqrt{64}+b^m_\ell(q_i),\quad
A^m=\operatorname{softmax}_{\ell}(L^m),
\]
\[
\widetilde B^m=bank_m+\omega_m A^mV^m.
\]

Attention dropout is zero in the initial configuration; `ω_m` is a learned sigmoid scalar initialized to `0.1`. Values use the original `B_k` and one shared transform, with no order-specific value bias.

The structural differences are

\[
D^m_k=\widetilde B^m_k-\widetilde B^m_0,\quad k=1,2,3.
\]

A shared order prior, modality-specific order bias, and modality-specific node-conditioned scorer produce

\[
\eta^m_{ik}=0.5\tanh(p_k+c^m_k+f_m([D^m_{ik},q_i])).
\]

The output layer of `f_m` is zero initialized, while `p_1=atanh(0.4)` and `p_2=p_3=0`, giving initial coefficients approximately `(0.2,0,0)`. The coefficients are signed and bounded in `(-0.5,0.5)`. The structural residual and node-wise semantic gate are

\[
U^m_i=\sum_{k=1}^{3}\eta^m_{ik}D^m_{ik},\qquad
g^m_i=\sigma(G_m[H^0_m[i],q_i]),
\]

where the last gate layer has zero weights and a bias initialized to `logit(0.1)`. Then

\[
Y_m=H^0_m+g_m\odot U_m,\qquad
Z_m=\operatorname{LayerNorm}(Y_m+\operatorname{RefineMLP}_m(Y_m)).
\]

The two modality refinement MLPs are independent.

## Late fusion and task contract

The final embedding is

\[
Z=\operatorname{LayerNorm}(W_s[Z_t\Vert Z_v]+\operatorname{MLP}([Z_t\Vert Z_v])).
\]

`Z` has shape `[N,256]`. `forward(x, edge_index)` returns `(Z, None, None, scalar_zero, {})`; `out_dim=256`. The model exposes `requires_full_lp_sampler_depth=True`. `inference` performs the same full-graph computation in evaluation mode; it does not divide relation normalization or global edge scoring into independent node batches.

## Initialization and sharing summary

| Component | Sharing | Initialization |
| --- | --- | --- |
| Text/visual projection | Independent | Standard Linear/Norm initialization |
| CRC scorer | Shared by edge and modality | Final Linear weight and bias zero; weights start at one |
| Relation context MLP | Shared across modalities | Standard Linear initialization, then LayerNorm |
| Chebyshev recurrence | No learned order-specific filter | Fixed recurrence, `K=3` |
| Q/K/value | Shared across modalities and orders | Standard Linear initialization |
| Order embeddings | Independent by modality | Normal with standard deviation `0.02` |
| Relation-order bias | Independent by modality | Standard Linear initialization |
| Eta prior | Shared by modality | `eta_1≈0.2`, `eta_2=eta_3=0` |
| Conditional eta scorer | Independent by modality, shared across orders | Output layer zero initialized |
| Attention residual scalar | Independent by modality | Sigmoid value `0.1` |
| Semantic gate | Independent by modality | Zero final weights, sigmoid value `0.1` |
| Refine MLP and late fusion | Independent modality refine; shared fusion | Standard Linear initialization |

## Differences from CoSI-MAG Final

VNext is implemented independently and does not inherit `CoSIMAGFinal`. Final uses learned diagonal modality metrics for edge cosine calibration, anchored cumulative multi-hop states, a learned global order prior, and separate modality attention blocks. VNext uses a shared four-evidence cross-modal edge scorer, a differentiable eight-value local context, the fixed Chebyshev recurrence on normalized `P`, shared Q/K/V across modalities, and relation-conditioned signed residual coefficients. No old CoSI parameter name or checkpoint is reused. The public task contract and final concat residual fusion remain compatible.

## Relationship to related mechanisms

- [ReCoG: Reciprocal Co-Evolution for Multimodal Graph Learning](https://arxiv.org/abs/2608.22786) jointly refines multimodal graph structure and representations. VNext shares the broad motivation of using cross-modal semantic evidence to assess relations, but preserves the supplied physical topology and only changes positive edge weights; it has no graph refiner, edge proposal, or coupled cross-modal message passing.
- [UniFilter](https://arxiv.org/abs/2405.12474) builds a universal polynomial basis to adapt filtering across homophily and heterophily. VNext uses a fixed Chebyshev bank and learns downstream node-conditioned mixtures; it does not adapt the polynomial basis itself.
- [OptBasisGNN](https://arxiv.org/abs/2302.12432) addresses a graph- and signal-dependent optimal polynomial basis. VNext neither solves that optimization nor claims its fixed Chebyshev bank is optimal.
- [HeroFilter](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5d570ed1708bbe19cb60f7a7aff60575-Abstract-Conference.html) adapts filtering to varying spectral responses in heterophilic relations. VNext conditions a mixture over a small polynomial order bank on local multimodal context; it does not estimate or adapt frequency-wise spectral responses.
- [MvCGE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7315c16422558ae81d65a812723d2cec-Abstract-Conference.html) uses routed collaborative graph experts and explicit load-equilibrium and view-discrepancy losses. VNext keeps two fixed modality streams and has neither expert routing nor those auxiliary losses.
- The recurrence follows the Chebyshev polynomial filtering family introduced for localized spectral graph convolutions; see [Defferrard et al.](https://arxiv.org/abs/1606.09375). The operator and recurrence specified here are the direct `T_k(P)` recurrence above.

These are mechanism-level comparisons, not claims of equivalent objectives, datasets, or empirical results.
