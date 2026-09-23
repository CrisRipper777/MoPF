# CSSI P3 conditional structural-response adaptation

This document freezes the implementation used by `src/models/cssi_v0.py`. It
is a new P3 model family; the historical `cosi_mag_final.py`,
`cosi_mag_ablation.py`, and P1/P2 response-probe implementations are not
modified.

## Backbone

The two modalities have independent projectors, `H0_text = phi_text(X_text)`
and `H0_visual = phi_visual(X_visual)`. The default P3 backbone uses unit edge
weights in both modalities, adds self loops, and applies the same symmetric
GCN operator

`P = D^(-1/2) (A + I) D^(-1/2)`.

With `multihop_anchor_alpha=0` the model materializes `S0, S1, S2, S3` and
`Sk = P S(k-1)`. Responses are `Rk = Sk - S(k-1)`, and the plain state is

`Zbase_m = 1/4 sum_k Sk_m = H0_m + .75 R1_m + .50 R2_m + .25 R3_m`.

The inherited modality-specific refinement, late residual fusion, and NC
linear head are always applied after this representation. P3 does not add a
utility predictor or utility-supervised loss.

## Response tokens and conditioners

For each modality and order, the response token is

`T(m,k) = LN(WH H0_m + WS S(k-1)_m + WR R(k)_m + e_k)`.

The `self` conditioner uses only the target modality's tokens. The `same`
conditioner uses only the same-order token from the other modality. Each
modality has its own projection into the common control space; its controller
input is

`[Qtarget, Qother, Qtarget * Qother, |Qtarget - Qother|]`, followed by a
modality-specific MLP. The `all`
conditioner uses scaled dot-product attention over the other modality's three
response orders. The `none` setting is the frozen no-adapter control.

## Adapters and variants

For each response, the scalar adapter emits a scalar gate and the low-rank
adapter emits `U diag(g(T)) V R`. `U` is initialized to zero, and all
corrections are additionally multiplied by the fixed numerical scale
`response_correction_scale=1e-3`. The scalar output scale is zero initialized.
Thus every enabled adapter starts exactly at the plain representation while
remaining trainable. There are no learned hop coefficients or routers.

The six P3 variants are:

| Variant | MRC | Conditioner | Adapter |
|---|---:|---|---|
| `mrc_plain` | yes | none | none |
| `self_lr` | no | self | low-rank, rank 8 |
| `same_scalar` | no | same-order | scalar |
| `same_lr` | no | same-order | low-rank, rank 8 |
| `all_lr` | no | all-order attention | low-rank, rank 8 |
| `same_lr_mrc` | yes | same-order | low-rank, rank 8 |

`mrc_plain` and `same_lr_mrc` reuse the inherited diagonal cosine relation
calibration. Unit-operator variants return exactly one for every physical
edge before symmetric normalization. No final-logit intervention is used in
the functional analysis; counterfactuals recompute the frozen refiner,
fusion, and classifier.

## Stability guard and interpretation

The first smoke run without a correction scale produced an exploding
correction ratio. The fixed `1e-3` output scale was added before formal P3
training and is recorded in every resolved config. It is an optimization
stability parameter, not a learned clipping rule: no response, gradient, or
logit clipping was added beyond the pre-existing training protocol's gradient
clip. The analysis reports correction ratios, finite-value checks, gradient
norms, low-rank collapse, and MRC saturation so this guard remains auditable.
