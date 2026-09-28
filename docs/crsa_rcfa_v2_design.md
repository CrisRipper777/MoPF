# CRSA + RSE + RCFA v2 design

## Research question

The model studies structure–attribute semantic interaction in multimodal attributed graphs. Physical connectivity specifies where nodes interact; modality-specific attributes interpret those relations, and a node then decides how much of each resulting structure-induced semantic change to absorb.

> Interpret how structure acts, then condition how its semantic effect is absorbed.

## Stage I: CRSA

Contextual Relational Semantic Adaptation (CRSA) answers:

> What does a physical relation semantically mean for modality *m*?

For text and visual features independently, the unchanged v1 path projects the modality to `H0`, computes the non-self local structural response `D = P_nbr H0 - H0`, interprets physical edges from `H0` and `D`, and reuses each edge route over three propagation steps. A non-self message is `C_j + E_sh(C_j) + ΔE_ij(C_j)`. The self-loop stays on its direct normalized physical path. Topology is fixed.

The CRSA modules, projectors, normalized graph construction, routing, operators, adapters, and plain fusion retain their v1 definitions and initialization stream (`seed + 17011`).

## Relation Semantic Effect

At hop `k`, the functional effect on a non-self edge is `e_ij,k = M_ij,k - C_j,k-1 = E_sh(C_j,k-1) + ΔE_ij(C_j,k-1)`. Its receiver aggregation with the exact CRSA normalized non-self edge weights is `R_i,k = Σ_j P_ij e_ij,k`.

RSE answers:

> How did that interpretation functionally alter structural transfer?

It is computed inside the same edge-chunk loop as the adapted message, has no separate learnable module, and remains attached to autograd. Since the self path is unchanged, `C_k = P C_k-1 + R_k` up to floating-point accumulation error.

## Structure-induced semantic change

`Δ_k = C_k - C_k-1` answers:

> What semantic change was induced by the k-th relation-aware propagation?

This is a per-step difference used by the absorber. It does not choose a neighborhood range or compete with other propagation steps.

## Stage II: RCFA

Relation-Conditioned Feature-wise Absorption (RCFA) answers:

> Given the receiver semantics and relational semantic effect, which components of that induced semantic change should be absorbed?

For each modality and node, the shared-across-hops conditioner consumes `[H0, Δ_k, R_k]` and outputs a 256-dimensional gate `g_k = 2 sigmoid(MLP([H0, Δ_k, R_k]))`. Text and visual conditioners have independent parameters. The final linear layer is zero-initialized, so the initial gate is exactly one. The correction uses fixed coefficients `ω = (0.75, 0.50, 0.25)`:

`B = mean([H0, C1, C2, C3])`

`Z = B + Σ_k ω_k ((g_k - 1) ⊙ Δ_k)`.

Thus initial `Z` equals the existing CRSA carrier `B`. The weights are the algebraic expansion of the uniform state mean and are not learned. There is no hop softmax, hop competition, transform of `Δ_k`, output normalization, or post-gate projection.

The `use_relation_condition=false` implementation keeps the same conditioner and parameters while replacing `R_k` with zeros. It is reserved for a later ablation and is excluded from the formal matrix.

## Full computation path

```text
Physical Connectivity
        ↓
Modality-specific Relation Interpretation
        ↓
Relation Semantic Effect
        ↓
Structure-induced Semantic Change
        ↓
Relation-conditioned Feature-wise Absorption
        ↓
Modality Representation
        ↓
Plain Multimodal Fusion
```

Text and visual each complete `CRSA → RSE → RCFA` without access to the companion modality. Their final `Z_text` and `Z_visual` enter the existing plain fusion MLP. Intermediate states are the carrier for progressive structure-induced changes; multi-order context retention is not the contribution.

## Code map

- `src/models/crsa_rcfa.py`: unchanged CRSA path, in-loop differentiable RSE, exact node-chunked RCFA, analysis interface.
- `configs/model/crsa_rcfa.yaml`: fixed dimensions and exact chunk sizes.
- `tests/test_crsa_rcfa.py`: legacy parity, RSE identity, RCFA identity, isolation, gradients, ablation path, label-free interface, strict reload.
- `scripts/run_crsa_rcfa_nc.py`: smoke and 15-context validation-only NC launcher.
- `scripts/analyze_crsa_rcfa_nc.py`: artifact audit and validation-only summaries/paired descriptions.

The formal protocol fixes `K=3` on all five NC datasets, AdamW (`1e-3`, `1e-4`), 300 epochs, patience 30, validation-accuracy checkpoint selection, and no test evaluation or link prediction.
