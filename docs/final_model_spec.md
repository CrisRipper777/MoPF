# SSI-MAG Final Model Specification

Status: frozen canonical implementation, P1.9. The final architecture is P1.8 variant U. This document specifies the function implemented by `src/models/ssi_mag_final.py`; `ssi_mag_v31_r1u.py` remains the historical development pilot and is not modified.

## Naming

- Stage I: Modality-Specific Context Formation
  - R1: Semantic-Grounded Relation Modulation
  - R2: Adaptive Semantic Reference
  - retained multi-hop contexts
- Stage II: Node-Adaptive Context Utilization
  - Context-Change Encoding
  - Cross-Hop Context Interaction
  - Formation-Conditioned Signed Filtering
- Final block: per-modality residual refinement and concat-residual late fusion

Text and visual streams are independent until late fusion. There is no early fusion, propagation-stage cross-modal attention, graph rewiring, added semantic edges, or deleted physical edges.

## Stage I: context formation

For modality `m`, projection produces `H0^m = ProjectionMLP(X^m)`. A relation projection is applied to `LayerNorm(H0^m)`:

`R_i^m = Linear(LayerNorm(H0_i^m), relation_rank)`.

For every original physical edge `(i,j)`, the symmetric semantic descriptor is

`u_ij^m = [cos(R_i^m,R_j^m), r_ij^m, |r_ij^m|]`,

where

`mu_i^m = mean_{j in N_i} cos(R_i^m,R_j^m)` and
`r_ij^m = cos(R_i^m,R_j^m) - 0.5(mu_i^m + mu_j^m)`.

The descriptor is scored in chunks by a modality-specific linear scorer followed by softsign:

`a_ij^m = softsign(f_rel^m(u_ij^m))`.

Self-loops have `a_ii=0`. The relation weight is used directly:

`w_ij^m = exp(a_ij^m)`.

There is no beta parameter. The normalized operator `P^m` is `gcn_norm` applied to the unchanged physical support and these weights. The local relation state is

`c_i^m = mean_{j in N_i, j != i} |a_ij^m|`,

with isolated nodes exactly zero.

For `k=1,...,K`,

`Q_k^m = P^m S_{k-1}^m`,

`d_ik^m = 1 - cosine(Q_ik^m,H0_i^m)`,

`p_i^m = (LayerNorm(H0_i^m) dot v_p^m) / (||v_p^m||_2 + eps)`,

`alpha_ik^m = sigmoid(logit(alpha0) + b_k^m + rho_p^m p_i^m + rho_d^m d_ik^m)`,

`S_0^m=H_0^m`, and

`S_k^m=(1-alpha_ik^m)Q_k^m + alpha_ik^m H_0^m`.

The explicit retained context bank is `[S_0^m,...,S_K^m]`.

## Stage II: context utilization

`D_0^m=0` and, for `k>=1`,

`D_k^m = LayerNorm(S_k^m-S_{k-1}^m)`.

The context-change token is

`T_k^m = LayerNorm(S_k^m) + g_delta^m W_delta^m(D_k^m) + order_embedding_k^m`,

where `g_delta=tanh(theta_delta)`.

One layer and one head of node-local Q/K/V attention operates only over the `K+1` hop tokens. It produces `O_k^m`, and

`S_tilde_k^m = S_k^m + g_int^m O_k^m`,

where `g_int=tanh(theta_int)`. The interaction-off intervention sets `g_int=0` at analysis time and therefore returns `S_tilde=S` exactly.

The formation-conditioned signed coefficient is

`eta_ik^m = gamma_k + DeltaGamma_k^m + delta_content,ik^m + delta_ref,ik^m + delta_rel,ik^m`,

where `delta_content` reads `S_tilde`,

`delta_ref = s_ref^m (a_i,k^m - mean_q a_i,q^m)`,

and

`delta_rel = s_rel^m c_i^m beta_hat_rel,k^m`.

Coefficients are signed and unbounded; no softmax is used. Final modality composition is strictly

`Z_i^m = sum_k eta_ik^m S_tilde_ik^m`.

Each modality then receives its residual refinement MLP and LayerNorm. Only the refined text and visual outputs are concatenated in the final residual fusion block.

## Claim boundary

- R1 supports the context-formation mechanism and demonstrably perturbs the normalized propagation operator; it is not claimed to be the main performance driver.
- R2 provides node-specific semantic-reference adaptation; no claim is made that it improves cross-seed stability for every modality.
- Cross-hop interaction is a major functionally active utilization mechanism in frozen intervention measurements.
- Attention has node heterogeneity substantially larger than query diversity in the available diagnostics; it is not described as strong pairwise-hop reasoning.
- Signed filtering permits suppression and cancellation. A larger negative-eta fraction is not interpreted as inherently better.
- Performance evidence, mechanism behavior, frozen sensitivity, and retrained ablation evidence are separate evidence types.
