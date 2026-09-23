# MGSC-MAG P2 architecture

This document records the canonical working architecture for the current
five-dataset node-classification study. The canonical configuration is
[`configs/model/mgsc_mag_p2.yaml`](../../configs/model/mgsc_mag_p2.yaml).
The frozen CoSI reference remains unchanged.

## Stage I: Modality-Calibrated Adaptive Context Formation

Text and visual attributes are projected independently:

\[
H_0^m = \phi_m(X^m), \qquad m\in\{T,V\}.
\]

For each physical edge `(i,j)`, the modality-specific relation calibration
computes a semantic score `s_ij^m`, converts it to a relation weight `w_ij^m`,
and constructs the normalized modality-specific graph operator `A-hat^m`.
The physical topology is shared; only the edge weighting is modality-specific.

At order `k`, the neighborhood proposal is

\[
N_{i,k}^m = [\hat A^m S_{k-1}^m]_i.
\]

The adaptive context gate is a scalar per node, modality, and order:

\[
g_{i,k}^m = \operatorname{sigmoid}\left(
f_g^m(H_{i,0}^m, N_{i,k}^m,
|H_{i,0}^m-N_{i,k}^m|,
H_{i,0}^m\odot N_{i,k}^m,p_k)\right),
\]

where `p_k` is a learnable order embedding and the text and visual gate MLPs
are independent. The context state is

\[
S_{i,k}^m=(1-g_{i,k}^m)H_{i,0}^m+g_{i,k}^mN_{i,k}^m.
\]

Thus the modality-specific multi-order context state bank is

\[
B_i^m=[S_{i,0}^m,S_{i,1}^m,\ldots,S_{i,K}^m],
\]

with `K=3` in the canonical configuration. Text and visual propagation remain
completely separate through this stage.

## Stage II: Interaction-Aware Multi-Order Context Integration

Each order state receives its order embedding before within-node cross-order
interaction:

\[
T_k^m=S_k^m+p_k.
\]

The implementation applies single-head order attention using the projected
order tokens and the retained legacy relation-order bias:

\[
A=\operatorname{softmax}(QK^\top/\sqrt d + \text{relation-order bias}),
\]

and produces interacted states `\tilde S_k^m`. In canonical P2 the final
composition directly uses these interacted states:

\[
\eta_{i,k}^m=\gamma_k+\Delta\gamma_k^m+\Delta_{i,k}^m,
\qquad
Z_i^m=\sum_k\eta_{i,k}^m\tilde S_{i,k}^m.
\]

The two modality representations `Z^T` and `Z^V` are refined and fused only
at the final late-fusion stage before the NC classifier. There is no early
fusion, semantic-neighbor augmentation, topology rewiring, auxiliary loss, or
new interaction module in this canonicalization.

## Canonical switches

The working P2 setting is explicitly:

```yaml
adaptive_context_gate: true
direct_interacted_integration: true
use_legacy_relation_order_bias: true
```

The legacy relation-order bias is retained for the functional ablations and
for comparability with the formal P2 runs. The P2 model is a candidate working
architecture, not a modification of the frozen `cosi_mag_final` reference.
