# MoPF-vNext U2-C0 — State–Response Decoupling and Differential-Basis Integration Audit

Frozen integration, algebra, and code-path audit only. No formal NC retraining, U2-C full run, LP run, alpha selection, K tuning, or downstream redesign was performed.

## Terminology correction

The earlier U2-B phrase `Hop-wise Structural Innovation` is refined to `Hop-wise Differential Structural Response` (逐跳差分结构响应). `D_k` is the representation change induced by one additional propagation step; it is not exact k-hop unique information, orthogonal innovation, or newly observed k-hop nodes.

## Algebra and architecture

For `q=1-alpha`, the audit verifies `S_k=q P S_{k-1}+alpha H`, `D_k=S_k-S_{k-1}`, `D_k=q^k P^(k-1)(P-I)H`, and `D_k^B3=q^k D_k^B1`. Semantic anchoring resides in cumulative states `S_k`; the differential values do not retain an additive alpha H term at every hop.

The opt-in prototype uses `S_k -> node/modality coefficient conditioner` and `D_k -> response filter values`. Default `cumulative` delegates to the historical propagation bank and remains the formal default; historical PDC modes are explicitly rejected with `anchored_differential`.

## Audit result

- Formal checkpoints: 15 U1-T T2 checkpoints; formal K values are unchanged.
- Maximum state reconstruction error: `1.7763568394002505e-15` absolute, `1.9581354972563268e-17` relative L2.
- Maximum B3/B1 scaling error: `1.7053025658242404e-13` absolute, `1.0419626863357139e-13` relative L2.
- Maximum closed-form error: `1.829647544582258e-13` absolute, `8.429351364712361e-14` relative L2.
- Basis-equivalent initialization: `True` on toy and real data paths.
- Anchor conditioner path nontrivial: `True`.
- Parameter count unchanged: `True`; alpha trainable: `False`.

## Corrected mechanism attribution

Differencing is the dominant source of response-coordinate distinctiveness. Semantic anchoring improves ego-semantic retention of cumulative states. B3 combines complementary, largely separable roles; the anchored differential directions are scaled B1 directions rather than a new joint response geometry.

## U2-C gate: **A. Proceed to U2-C: state–response integration is mathematically and numerically valid**

The gate is based only on mathematical validity, numerical validity, initialization equivalence, parameter invariance, and the frozen anchor-conditioner functional path. It is not a downstream performance claim.

Authoritative summary: `/hdd1/DataInHere/YHF/MoPF/outputs/u2c0_state_response_integration_audit/u2c0_master_summary.json`
