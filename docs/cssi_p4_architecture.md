# CSSI-v1: Bounded Response Subspace Modulation

CSSI-v1 is a validation-stage replacement for the P3 response transformation. It preserves the `all_plain` backbone, the independent text/visual projectors, explicit `S0...S3`, unit physical edges when MRC is disabled, optional historical diagonal-cosine MRC, response tokens, modality-specific refinement, late fusion, and the existing NC head.

## Response transformation

For each modality, a trainable `r × d` parameter (`r=8`) is converted on every forward pass to `Bbar` using reduced QR on its transpose. Thus `Bbar Bbarᵀ` is numerically the identity. For `h=Bbar R`, the BRSM correction is:

```text
a = tanh(f(C))
delta_R = lambda_k^m Bbar^T (a ⊙ h)
lambda_k^m = lambda_max sigmoid(theta_k^m)
```

`lambda_max=0.2` and `lambda_init=0.05`. There is no fixed `1e-3` output multiplier, no affine output bias, and `R=0` gives exactly `delta_R=0`. Orthonormality and the bound `||delta_R|| <= lambda ||R||` are tested at runtime and in `tests/test_cssi_v1.py`.

The conditioner output projection receives an initialization-only gain of 4.0. This prevents the rank-8 projection in a 256-dimensional hidden space from producing a universal first-epoch ratio near `1e-3`; it is not a forward output scale and does not alter the BRSM bound.

The scalar control baseline uses the same global bounded strength but applies `delta_R = g R`, where `g=lambda tanh(f(C))`. It has no zero-initialized global gate.

## Conditioning

Self-BRSM uses the target response token only. Same-BRSM uses modality-specific control projections and an MLP over `[q_target, q_other, q_target⊙q_other, |q_target-q_other|]`. The other modality controls the target response only; it is not directly injected into the target content representation. Cross-Off zeros `q_other` after its projection, preventing a Linear bias from reintroducing paired evidence. Cross-Shuffle permutes the other modality's node tokens.

No all-order attention, auxiliary loss, learned hop coefficients, or new MRC implementation is used.

## Variants

| Variant | Relation construction | Controller | Response transform |
| --- | --- | --- | --- |
| `self_brsm` | unit physical edges | self-only | BRSM |
| `same_scalar_v1` | unit physical edges | same-order cross-modal | bounded scalar |
| `same_brsm` | unit physical edges | same-order cross-modal | BRSM |
| `same_brsm_mrc` | historical diagonal-cosine MRC | same-order cross-modal | BRSM |

All P4 runs use `unified_full_graph_nc_v1`, existing NC splits, seeds 42/43/44, `task.evaluate_test=false`, and `task.loss.aux_weight=0.0`.
