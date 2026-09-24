# SSI-MAG-V3.1 Candidate Specification

This file is the single candidate specification for the future P1.7 pilot. It is not executable V3.1 code.

## Scope and invariants

- Task: NC only; full-graph protocol; existing splits and validation-selected checkpoints.
- Modalities: Text and Visual remain independent until final concat-residual late fusion.
- `hidden_dim=256`, `max_order=num_layers=3`, one cross-hop layer and one head.
- Shared physical edge support; no graph rewiring, semantic kNN, edge addition/deletion, or early fusion.
- No auxiliary loss, PLM/VLM fine-tuning, or new encoder branch.

## Candidate data flow

`X^m -> H0^m -> relative semantic relation interpretation -> P^m -> {S0^m,...,SK^m} -> node-wise cross-hop interaction -> signed context utilization -> Z^m -> per-modality refinement -> late fusion`.

## Exact formulas

For each modality `m`:

`H0_i^m = ProjectionMLP_m(X_i^m)`.

### R1: Semantic-Grounded Relative Relation Modulation

`h_i^m=LN(H0_i^m)`

`z_i^m=W_rel^m h_i^m`, with `W_rel^m` mapping 256 to 16 and no bias.

`s_ij^m=cos(z_i^m,z_j^m)`.

Using the undirected incident physical neighborhood:

`mu_i^m=mean_{j in N_i^und}s_ij^m`; isolated `mu_i=0`.

`rrel_ij^m=s_ij^m-0.5*(mu_i^m+mu_j^m)`.

`u_ij^m=[s_ij^m,rrel_ij^m,abs(rrel_ij^m)]`.

`a_ij^m=softsign((w_rel^m)^T u_ij^m+b_rel^m)` where `softsign(x)=x/(1+abs(x))`.

`beta^m=sigmoid(theta_beta^m)`, initialized at `0.05`.

`w_ij^m=exp(beta^m*a_ij^m)` and `P^m=gcn_norm(edge_index,w^m)` with the current self-loop policy.

`c_i^m=mean_{j in N_i^und}abs(beta^m*a_ij^m)`, isolated `c_i=0`.

The descriptor and scorer are symmetric; edge scoring is chunked. `beta->0` exactly restores unit physical edge weights.

### R2: Stabilized semantic reference

`h_i^m=LN(H0_i^m)` and `p_i^m=((w_p^m)^T h_i^m)/(||w_p^m||_2+eps)`.

`S0^m=H0^m`.

For `k=1,...,K`:

`Q_k^m=P^m S_{k-1}^m`

`d_ik^m=1-cos(Q_ik^m,H0_i^m)`

`logit(alpha_ik^m)=logit(0.10)+b_k^m+rho_p^m p_i^m+rho_d^m d_ik^m`

`S_k^m=(1-alpha_k^m)Q_k^m+alpha_k^m H0^m`.

The explicit `rho_c c_i` term is removed. Initialize `b=0`, `rho_p=0`, `rho_d=0`, giving alpha exactly `0.10`.

### Stage-I preservation

Retain `{S0,S1,...,SK}` as the formal context bank.

### Stage-II context change and interaction

`D0=0`; `D_k=LN(S_k-S_{k-1})` for `k>=1`.

`T_k=LN(S_k)+g_delta W_delta D_k+order_embedding_k`, with `g_delta=tanh(theta_delta)` initialized at `0.10`.

For one attention layer/head per node:

`A_i=softmax(Q_iK_i^T/sqrt(256))` and `O_i=A_iV_i`.

`S_tilde_k=S_k+g_int O_k`, with `g_int=tanh(theta_int)` initialized at `0.10`.

The claim is Node-Adaptive Cross-Hop Context Interaction: nodes form different mixtures over retained contexts.

### Signed utilization and composition

`delta_content,ik=LowRankScorer(S_tilde_ik)`.

`beta_hat_rel=standardize(center(relation_order_profile))`.

`relation_residual_ik=s_rel^m c_i^m beta_hat_rel,k^m`, initialized with `s_rel=0.10`.

`eta_ik=gamma_k+DeltaGamma_k^m+delta_content,ik^m+relation_residual_ik^m`.

No softmax is used. The Stage-II reference residual is absent.

`Z_i^m=sum_k eta_ik^m S_tilde_ik^m`.

Per-modality residual refinement and existing concat-residual late fusion are unchanged.

## Decisions requiring P1.7 validation

R1 and R2 are candidate revisions, not validated improvements. P1.7 must measure validation performance, training stability, operator perturbation, relation-state behavior, alpha dispersion/saturation, term_p stability, attention node adaptivity, signed eta behavior, and peak memory. Test metrics are recorded only after validation selection and never used to choose a setting.
