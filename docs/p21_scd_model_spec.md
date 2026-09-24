# P2.1 Clean Semantic-Controlled Diffusion model specification

## Scope

`ssi_mag_scd` is the clean simplified S model for P2.1. It is derived from
the current Final model by removing Context-Change Encoding and Cross-Hop
Interaction. P2.1 is a simplification and verification round; it introduces
no new attention, gate, auxiliary loss, hyperparameter search, LP experiment,
or Test-driven model choice.

The retained graph is modality-separated until the existing late fusion:

```text
x_m -> H0_m -> semantic relation conductance P_m
             -> semantic-feedback restart states S_0...S_K
             -> formation-conditioned signed response eta
             -> Z_m -> existing refinement -> existing late fusion
```

## Retained formulas

For each modality, the model retains exactly the Final R1/R2/R3 formulas.

- `z_i = W_rel LN(H0_i)` and `s_ij = cos(z_i,z_j)`;
- incident non-self `mu_i`, centered residual `rrel_ij`, and
  `u_ij=[s_ij,rrel_ij,|rrel_ij|]`;
- `a_ij = softsign(f_rel(u_ij))`, `a_ii=0`, `w_ij=exp(a_ij)`;
- `P = GCNNorm(E,w)`;
- `c_i` is the incident non-self mean of `|a_ij|`, with isolated nodes set to
  zero;
- `Q_k=P S_{k-1}`, `d_k=1-cos(Q_k,H0)`;
- `alpha_k=sigmoid(logit(0.1)+b_k+rho_p p+rho_d d_k)`;
- `S_k=(1-alpha_k)Q_k+alpha_k H0`, retaining the full state bank;
- `eta_k=gamma_k+DeltaGamma_k+delta_content_k+delta_ref_k+delta_rel_k`;
- `Z=sum_k eta_k S_k`.

The existing modality refinement and late fusion are unchanged.

## Removed graph and parameters

The clean instance has no trainable or runtime instances of:

| path | historical names |
|---|---|
| Context-Change Encoding | `context_delta_norm_{text,visual}`, `context_delta_proj_{text,visual}`, `theta_delta_{text,visual}` |
| hop-order token encoding | `hop_order_embedding_{text,visual}` |
| Cross-Hop Interaction | `hop_layers_{text,visual}` (`norm`, `query`, `key`, `value`), `theta_int_{text,visual}` |

The constructor first executes the historical Final construction and then
deletes these components. This preserves active-parameter initialization under
the same global seed; the deleted keys are not part of the clean
`state_dict()`. The clean forward never creates `D`, tokens, Q/K/V attention,
attention output, or interaction correction. `S_used` is the retained `S`
state bank and is exposed only as an analysis label.

The public flags are:

```text
context_change_present = false
cross_hop_interaction_present = false
```
