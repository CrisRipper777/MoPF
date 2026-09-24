# SSI-MAG-V3.1 Architecture Design Freeze

Status: architecture/design audit only. This document freezes one V3.1 candidate; it does not implement V3.1.

Repository state audited: branch `V3`, commit `8c263cd5b5224fe4226cd2f8b56b9773ebe3a17f` at audit start. The audited evidence is the existing P1 Full NC result and the P1.5/P1.5a post-hoc analyses. No model forward, configuration, split, task protocol, loss, checkpoint, or historical P1/P1.5 artifact was changed.

## 1. Diagnostic validation before design

The P1.5a implementation was checked directly, including its CSV/JSON outputs:

- attention is `[N, query-hop, key-hop]`, observed as `[N,4,4]`;
- entropy is computed per node/query row before aggregation;
- query diversity is the within-node mean over unordered query pairs of `0.5 * ||A_iq-A_iq'||_1`;
- node heterogeneity is `0.5 * ||A_iq-mean_i(A_iq)||_1`;
- diagonal excess is `diag_mass - 1/(K+1)`;
- `alpha_range_ratio` is `(q90(alpha)-q10(alpha))/(abs(mean(alpha))+eps)`;
- cross-seed semantic-prior consistency uses `term_p=rho_p*p`, not raw `p` sign.

No new P1.5a metric bug was found. The earlier P1.5 mean-matrix-first attention summary and adaptive-range denominator issue are treated as corrected historical diagnostics, not as V3.1 evidence.

## 2. Frozen paper-level story

The paper-level story is:

> A shared physical topology does not imply a shared structural context across modalities. The usefulness of the resulting multi-hop structural contexts is node-dependent.

The model is a two-stage framework:

1. **Stage I — Modality-Specific Context Formation**: each modality interprets the same physical topology in its own semantic space, forms progressive structural-semantic contexts, and explicitly retains all hop states.
2. **Stage II — Node-Adaptive Context Utilization**: each node interacts over retained hop contexts and applies signed, node-adaptive context coefficients before composition.

The old `MRC -> SMP -> RCMI` module-stack wording is not the primary story. It may be used only as implementation history.

## 3. Evidence separation

Evidence is kept in separate categories in `p16_evidence_matrix.csv`:

- **Empirical evidence**: observed distributions, heterogeneity, cross-seed consistency, and branch magnitudes.
- **Mechanism behavior evidence**: what the frozen tensors and formulas actually do.
- **Functional sensitivity**: frozen relation-off or interaction-off changes using the saved head; these are not retrained causal ablations.
- **Performance evidence**: descriptive P1 validation/test results only; no test metric was used for a decision.

The corrected P1.5a aggregate shows nodewise attention entropy `0.521270` versus mean-matrix entropy `0.888022`, Jensen gap `0.366752`, node heterogeneity `0.392988`, query diversity `0.074505`, and diagonal excess `-0.003526`. Thus attention is strongly node-dependent, has smaller query-dependent variation, and is not diagonally collapsed under the corrected definition.

P1 Full has five-dataset unweighted validation Accuracy `0.812566` and Macro-F1 `0.745884`; test values `0.809106` and `0.744392` are descriptive only. The protocol-matched P1 comparison is not a uniform performance win, so V3.1 is justified as a parameterization simplification and mechanism test, not as a claimed performance improvement.

## 4. Unique SSI-MAG-V3.1 candidate architecture

Let `m` denote Text or Visual. All tensors below are modality-specific until the final late fusion.

### Stage I: Modality-Specific Context Formation

#### 4.1 Projection and physical topology

`H0_i^m = ProjectionMLP_m(X_i^m)`.

The physical edge support is unchanged. V3.1 adds no edges, removes no edges, uses no semantic kNN graph, and performs no cross-modal propagation.

#### 4.2 R1 — Semantic-Grounded Relative Relation Modulation

This is the single R1 revision frozen for the P1.7 pilot.

`h_i^m = LayerNorm(H0_i^m)`

`z_i^m = W_rel^m h_i^m`, with `W_rel^m: hidden_dim -> relation_rank`, `bias=False`.

Use the undirected incident physical neighborhood for the endpoint statistic:

`s_ij^m = cosine(z_i^m,z_j^m)`

`mu_i^m = mean_{j in N_i^und} s_ij^m`, with `mu_i=0` for an isolated node.

`rrel_ij^m = s_ij^m - 0.5*(mu_i^m+mu_j^m)`

`u_ij^m = [s_ij^m, rrel_ij^m, abs(rrel_ij^m)]`

`a_ij^m = softsign((w_rel^m)^T u_ij^m + b_rel^m)`

where `softsign(x)=x/(1+abs(x))`. The scorer is one modality-specific 3-to-1 affine map, not a large MLP. The descriptor is symmetric under edge reversal. Edge scoring remains chunked.

`beta^m = sigmoid(theta_beta^m)`, initialized to `0.05`.

`w_ij^m = exp(beta^m*a_ij^m)`.

As `beta -> 0`, every physical edge weight is exactly one and the normalized operator is the raw physical-topology operator under the unchanged self-loop policy.

`c_i^m = mean_{j in N_i^und} abs(beta^m*a_ij^m)`, with `c_i=0` for isolated nodes.

Candidate scorer parameters `w_rel,b_rel` are zero-initialized so the initial relation interpretation is neutral while gradients remain available. This is an initialization choice requiring P1.7 validation, not an observed result.

#### 4.3 R2 — Stabilized semantic reference

Retain a learned semantic direction but remove the saturating `tanh`:

`h_i^m = LayerNorm(H0_i^m)`

`p_i^m = ((w_p^m)^T h_i^m)/(||w_p^m||_2 + eps)`.

For `k=1,...,K`:

`S0^m=H0^m`

`Q_k^m=P^m S_{k-1}^m`

`d_ik^m=1-cosine(Q_ik^m,H0_i^m)`

`logit(alpha_ik^m) = logit(alpha0) + b_k^m + rho_p^m p_i^m + rho_d^m d_ik^m`

`S_k^m=(1-alpha_k^m)Q_k^m + alpha_k^m H0^m`.

The explicit `rho_c^m c_i^m` term is removed. R1 already changes `P^m` and therefore `Q_k,d_k`; `c_i` is retained for Stage-II relation-state conditioning. Initialization remains `b_k=0`, `rho_p=0`, `rho_d=0`, and `alpha0=0.10`, hence `alpha_ik=0.10` at initialization.

#### 4.4 Explicit multi-hop preservation

The complete bank `{S0,S1,...,SK}` is retained. These states are the outcomes of progressive structure-semantic contextualization under increasing receptive fields. “Trajectory” is not a headline claim.

### Stage II: Node-Adaptive Context Utilization

#### 4.5 Context-change encoding

`D0=0`; for `k>=1`, `D_k=LayerNorm(S_k-S_{k-1})`.

`C_delta,k = g_delta^m W_delta^m D_k`

`T_k=LayerNorm(S_k)+C_delta,k+order_embedding_k`.

Keep one scalar gate with `g_delta=tanh(theta_delta)` and initialization `g_delta=0.10`. The P1.5a injection ratio is nonzero, so this remains core.

#### 4.6 Node-Adaptive Cross-Hop Context Interaction

Keep exactly one layer, one head, and `K+1` tokens per node:

`A_i=softmax(Q_i K_i^T/sqrt(hidden_dim))`

`O_i=A_i V_i`

`S_tilde_k=S_k+g_int^m O_k`, with `g_int=tanh(theta_int)` initialized at `0.10`.

The formal claim is **Node-Adaptive Cross-Hop Context Interaction**: different nodes form different mixtures over retained multi-hop contexts. The claim is not strong pairwise hop-dependency reasoning. The interacted state, not the original state, enters final composition.

#### 4.7 Signed context utilization

Retain the global and modality order prior:

`gamma_k + DeltaGamma_k`.

Retain the current low-rank content scorer, reading `S_tilde_k`:

`delta_content,ik = LowRankScorer(S_tilde_ik)`.

Remove the explicit Stage-II reference residual `s_ref * centered([1,alpha1,...,alphaK])`. Its observed node standard deviation is tiny and its covariance contribution is near zero; its dominant behavior is a fixed order pattern already represented by the global/modality order prior.

Retain relation-state conditioning:

`beta_hat_rel = standardize(center(relation_order_profile))`

`relation_residual_ik = s_rel^m * c_i^m * beta_hat_rel,k^m`.

`eta_ik = gamma_k + DeltaGamma_k^m + delta_content,ik^m + relation_residual_ik^m`.

`eta` remains signed and unbounded; no softmax is introduced.

Final modality composition is:

`Z_i^m = sum_k eta_ik^m * S_tilde_ik^m`.

Keep the per-modality residual refinement MLP/LayerNorm and the existing concat-residual late fusion. Text and Visual have no data dependency before this final fusion.

## 5. Why the simplification is one framework

R1 and R2 define how each modality forms contexts from the shared topology. Preservation makes the full context bank available. Stage II then decides, per node, how to use that bank through cross-hop interaction and signed composition. Removing the fixed reference residual avoids injecting the same order-shaped signal a second time. Retaining `c_i` only where it carries relation state into utilization gives the same information a single coherent path:

`shared physical topology + heterogeneous modality semantics -> modality-specific relation interpretation -> progressive context formation -> explicit preservation -> node-adaptive interaction -> signed utilization -> late fusion`.

This is a context-formation-to-context-utilization framework, not a collection of independent module names.

## 6. Initialization, fallback, and complexity audit

- R1 `beta=0.05`; `beta->0` is the exact unit physical-topology fallback; `w_rel,b_rel=0` gives a neutral initial candidate scorer.
- R2 `alpha0=0.10`, `b=0`, `rho_p=0`, `rho_d=0`; initialization is exactly alpha `0.10`.
- `g_delta=0.10` and `g_int=0.10`; order embeddings remain zero-initialized as in V3.
- `gamma_global` keeps the current restart/order prior; `DeltaGamma=0`; node content vectors and relation order profile keep their current small random initialization; `s_rel=0.10`.
- No auxiliary loss is used.
- Current V3 model parameter count from the saved checkpoint is `1,538,028` excluding the NC classifier head. The candidate removes two 16-dimensional projection biases, replaces the two 561-parameter relation MLPs with two 4-parameter affine softsign scorers, removes two `rho_c` scalars, and removes two reference scales: `1,536,878` parameters, a reduction of `1,150` (`0.0748%`).
- No new attention layer/head, graph construction, PLM/VLM fine-tuning, or encoder branch is introduced. The candidate is compatible with the RTX3090/24GB requirement subject to the P1.7 memory pilot.

## 7. Evidence-chain requirements for P1.7

Each mechanism must be evaluated as Problem -> Mechanism -> Expected Functional Effect -> Diagnostic Metric -> Existing Evidence -> Missing Evidence. The complete matrix is in `outputs/ssi_mag_v3_p16_design/p16_evidence_matrix.csv`.

The most important missing evidence is matched P1.7 training evidence for the single R1 revision, the single R2 revision, and their combined candidate. P1.5 frozen interventions are sensitivity diagnostics only and cannot establish retrained causal necessity.

## 8. Freeze decisions

The unique candidate decisions are in `p16_component_decisions.csv`:

- R1: `REVISE` to Semantic-Grounded Relative Relation Modulation.
- R2: `REVISE` to Stabilized Semantic Prior and remove explicit `rho_c c`.
- Multi-hop preservation, context-change encoding, cross-hop interaction, direct interacted content, global/modality priors, content scoring, relation-state conditioning, and late fusion: `KEEP`.
- Stage-II reference residual: `REMOVE`.
- Auxiliary loss, cross-modal propagation, graph rewiring, and semantic-edge construction: `REMOVE`.
- Trajectory and strong pairwise-hop-dependency wording: `DEMOTE_FROM_CORE_CLAIM`.

## Boundary

This is a design freeze only. No V3.1 executable model was added. No training, ablation, LP experiment, split/protocol change, hyperparameter search, auxiliary loss, or test-based selection was performed.
