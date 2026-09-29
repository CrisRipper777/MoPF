# CRSA + ICSR v3 Design

## Research framing

The model keeps Stage I fixed as CRSA: modality-specific attributes interpret how physical graph relations act during propagation. Stage II then represents interactions between each node's intrinsic semantics and its own relation-interpreted contextual semantics. The design question is whether a small agreement/deviation interaction residual can improve the stable CRSA carrier without predicting structural utility.

## Stage I — CRSA

For each modality, the frozen CRSA implementation constructs projected intrinsic features `H0`, local structural response, relation routes, and contextual states `C1:C3`. The physical graph operator, symmetric normalization, direct self-loop path, shared relation operator, three residual adapters, route reuse, edge chunking, checkpointing, and plain multimodal fusion retain their frozen computation and initialization stream (`seed + 17011`).

`C_k` denotes relation-interpreted contextual semantics progressively induced by structural propagation. The stable carrier is `B = (H0 + C1 + C2 + C3) / 4`; it preserves semantic content while Stage II contributes only an interaction residual.

## Stage II — ICSR

For modality `m` and hop `k`, parameter-free layer normalization gives `h = LN(H0)` and `c_k = LN(C_k)`. Semantic agreement is `A_k = h ⊙ c_k`, a feature-wise interaction descriptor rather than a cosine similarity. Semantic deviation is `D_k = c_k - h`; deviation can encode complementary contextual information and is not assumed to be noise.

The concatenated descriptor `[A_k || D_k]` passes through one shared-per-modality `Linear(512, 32) → GELU → Linear(32, 256)` adapter. Text and visual adapters are independent, and each modality's adapter is shared over all three hops. The final layer is zero initialized, so at initialization every interaction residual is exactly zero. The output is `Z = B + (U1 + U2 + U3) / 3`.

ICSR receives only same-modality `H0` and `C_k`. It does not use RSE, relation route values, the other modality, labels, hop selection, or a utility gate. Normal training does not accumulate RSE diagnostics. Full-node descriptors are materialized only through `analyze()`.

## Interpretation boundary

Agreement and deviation provide complementary descriptors of intrinsic-context interaction. This design does not claim to identify complementary neighbors, solve heterophily, guarantee reliable context, or establish causal interaction. Whether the residual improves downstream representations is determined by the fixed validation-only experiment.
