# MoPF-vNext Final Method Freeze

Status: **FINAL — architecture design officially closed**

This document freezes the MoPF-vNext method after U3-B. The frozen model and
configuration are implemented by U3-B commit
`b4d93160aab9957a3de73a05aa71311ce21e6680` (the finalization commit adds only
this provenance documentation and the companion protocol/plan documents).
The SHA256 of `configs/model/mopf.yaml` is
`1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a`.

## Frozen architecture

The final model is exactly U1 + U2-C1 + U3-B1:

1. **U1 — Modality-Adaptive Semantic Transport**, formally **Modality-
   Adaptive Semantic Conductance**.
2. **U2 — Modality-Semantic Anchored Multi-Hop Propagation**.
3. **U3 — Transport-Conditioned Node–Modality Adaptive Composition**, using
   the concrete **Transport-Conditioned Preference Residual (TCPR)**.

The formal configuration is:

| Field | Frozen value |
|---|---|
| `edge_weight_mode` | `learned_diag_cos` |
| `edge_weight_temperature` | `0.35` |
| `multihop_state_mode` | `anchored` |
| `multihop_response_mode` | `cumulative` |
| `multihop_anchor_alpha` | `0.1` |
| `use_transport_residual` | `true` |
| `use_modality_residual` | `true` |
| `use_node_residual` | `true` |
| `fusion_mode` | `concat_residual_mlp` |
| `hrc_weight` / `ppc_weight` | `0.0` / `0.0` |

The existing `map_prior_restart` and `map_prior_order` fields remain in the
configuration and source for historical checkpoint compatibility. In formal
method prose, they are called **shallow restart filter initialization**.

## Formal equations

For modality `$m$`, the learned diagonal metric is

\[
a^m = \operatorname{softplus}(\theta^m) / \operatorname{mean}
(\operatorname{softplus}(\theta^m)),
\qquad
s^m_{ij}=\cos(h_i^m\odot a^m,h_j^m\odot a^m).
\]

On the original sparse physical support, the modality-specific conductance is

\[
w^m_{ij}=w_{\min}+(1-w_{\min})
\sigma(s^m_{ij}/T),
\qquad w_{\min}=0.1,\quad T=0.35,
\]

and `A_hat^m` is the corresponding sparse normalized operator. No dense
node-by-node adjacency is formed.

Let `$H_0^m=h^m$` and let `$K$` be the dataset-level fixed formal order. The
anchored state bank is

\[
S_0^m=H_0^m,\qquad
S_k^m=(1-\alpha)\hat A^mS_{k-1}^m+\alpha H_0^m,
\quad k=1,\ldots,K,
\]

with fixed \(\alpha=0.1\). The formal response bank is cumulative,
\(R_k^m=S_k^m\), and the modality output is

\[
Z_i^m=\sum_{k=0}^{K}\eta_{i,k}^mR_{i,k}^m.
\]

The frozen hierarchical coefficient remains

\[
\eta_{i,k}^m=\gamma_k+\Delta\gamma_k^m+\delta_{i,k}^m+\tau_{i,k}^m,
\]

where the existing node term is the rank-4 absolute-state conditioner

\[
\delta_{i,k}^m=\frac{1}{4}
\left\langle\tanh(W_k^mS_{i,k}^m),v_k^m\right\rangle.
\]

TCPR computes the mean incident U1 conductance on the raw physical support,
centers it across nodes, and detaches the resulting scalar context:

\[
\widetilde c_i^m=c_i^m-\operatorname{mean}_j(c_j^m),
\qquad
\beta_k^m=\theta_{\mathrm{transport},k}^m-
\operatorname{mean}_r(\theta_{\mathrm{transport},r}^m),
\qquad
\tau_{i,k}^m=\widetilde c_i^m\beta_k^m.
\]

The two transport profiles are zero-initialized, so the final model is
functionally initialization-equivalent to U2-C1 before training.

## Fixed order, parameters, and complexity

| Dataset | `K` | Final trainable parameters (MoPF encoder + NC head) |
|---|---:|---:|
| Movies | 3 | 1,001,832 |
| Toys | 3 | 1,001,318 |
| Grocery | 2 | 999,763 |
| ele-fashion | 3 | 868,704 |
| Reddit-S | 3 | 1,001,832 |

The counts include the NC linear classifier head; the LP decoder is task-
provided and unchanged. Relative to the frozen separate-cosine encoder:

- U1 adds two learned diagonal metric vectors of length `d=256`, exactly
  **+512 trainable scalars**.
- U2 adds **0 trainable parameters**. Anchoring, cumulative coordinates, and
  the fixed alpha change the state/response semantics and initialization
  coordinates only.
- U3 adds two zero-initialized transport order profiles, exactly
  **+2(K+1)** scalars: +8 for K=3 and +6 for Grocery.

The sparse implementation has time complexity `O(K|E|d)`, with modality
constants suppressed, and explicit-bank memory `O(KNd+|E|)`. TCPR adds
linear `O(|E|+NK)` scalar-context work. There is no dense `N×N` adjacency,
eigendecomposition, or additional asymptotic path.

## Compatibility and provenance closure

Strict CPU loading passed for all 15 U1 freeze checkpoints, all 15 U2-C1
checkpoints, and all 15 U3-B1 checkpoints, including each NC head. The final
formal model loads U3-B1 without a historical compatibility break. The U3-B1
master artifacts are in
`outputs/u3b_transport_conditioned_composition/`, including the master
summary, table, counterfactuals, and transport associations.

The following historical candidates are retained only as source/checkpoint
provenance or diagnostics and are not formal architecture choices: U2
differential response, PDC, HRC/PPC, and other conductance/response paths.
The final configuration selects neither differential response nor any PDC,
HRC, PPC, router, attention, MoE, auxiliary loss, fusion redesign, classifier
change, evaluator change, nor LP decoder change.

## Terminology freeze

Formal writing must use the names above. The paper must not use the historical
labels `Hop-wise Structural Innovation`, `Distinctive Differential Response`,
`effective radius`, `MAP prior`, `long-range shortcut`, `oversquashing
solution`, or `oversmoothing solution`. The compatibility field names may
remain in code, but paper-facing prose must use “shallow restart filter
initialization”.

## Freeze boundary

No model selection remains after this document. U1, U2, and U3 are immutable;
future evaluation is descriptive and uses the pre-registered quasi-held-out
markers described in
`docs/mopf_final_evaluation_protocol.md` and
`docs/mopf_final_experiment_plan.md`. Architecture design is officially
closed at this boundary.
