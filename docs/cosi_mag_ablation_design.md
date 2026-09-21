# CoSI-MAG Framework-Level Ablation Design

This document specifies the three paper ablations built on the frozen
`CoSIMAGFinal` encoder. The ablation module subclasses the final model so the
`full` path keeps the original parameter names, initialization, and forward
operations. The final model, task/data loaders, and model factory remain
unchanged.

## Variants

| Paper row | Mode | Removed stage |
|---|---|---|
| CoSI-MAG | `full` | None; used for equivalence checks only |
| w/o MRC (Uniform Relations) | `no_mrc` | Modality-Aware Relation Calibration and its two downstream signals |
| w/o Semantic Anchor | `no_semantic_anchor` | Semantic-preserving anchor in multi-hop propagation |
| w/o RCMI (Uniform Composition) | `no_rcmi` | Relation-Conditioned Multi-Order Interaction |

No TCPR, IAMOC, PPC, R2, or order-embedding ablations are implemented.

## Dependency handling

### w/o MRC

For each physical edge, `relation_weight_text` and `relation_weight_visual`
are exactly one. MRC does not evaluate its learned metric or cosine path in
this mode. Both modalities therefore use the same unweighted normalized
physical propagation operator in SMP.

The local relation context vectors are explicitly filled with zero for every
node before RCMI. This removes the MRC-to-RCMI relation-conditioned bias while
leaving content-based cross-order interaction and adaptive composition active.
The context is not estimated from uniform weights or z-scored. Isolated nodes
therefore receive zero context as well; they cannot acquire a synthetic
relation context from the graph-wide descriptor normalization.

### w/o Semantic Anchor

The runner sets `model.multihop_anchor_alpha=0.0`, and model construction
rejects any other value for this mode. The state recurrence is therefore

\[
S_0 = H_0, \qquad S_k = \hat A S_{k-1}.
\]

The alpha value is fixed before the inherited constructor initializes
`gamma_global`. Its basis conversion consequently uses alpha zero, which is
the ordinary monomial basis. This keeps the intended global prior aligned with
the no-anchor state basis instead of retaining the anchored initialization.
The other variants require alpha 0.1.

### w/o RCMI

After MRC and SMP produce the order states, each modality is pooled as

\[
Z_i^m = \frac{1}{K+1}\sum_{k=0}^{K} S_{i,k}^m.
\]

This branch then applies the same modality refinement MLPs and norms, fusion
skip, fusion MLP, and output norm as the full model. It does not evaluate order
embeddings, Q/K/V projections, cross-order attention, local relation context,
relation beta/scale, hop gates, node preference, `gamma_global`,
`delta_gamma`, or `eta`. The inherited RCMI parameters remain in the checkpoint
schema for a stable model family, but do not participate in this branch's
function. Perturbation and backward checks cover those parameters.

## Frozen protocol

The model config copies the final model's frozen hyperparameters and changes
only its name/version and the default `ablation_mode: full`. Supported model
modes are `full`, `no_mrc`, `no_semantic_anchor`, and `no_rcmi`.

The launcher has exactly three variants and three datasets:

- Movies, NC
- Grocery, NC
- sports-copurchase, LP

Each dataset-variant pair is one launcher unit. It launches `src.main` once
with base/split seed 42 and `num_runs=3`, so the task runner uses run seeds
42/43/44 without changing the dataset split. There are 9 launcher units and
27 training runs in the formal matrix. Sampled LP inherits
`requires_full_lp_sampler_depth=True`; with `K=3`, `[5, 5]` is extended to
`[5, 5, 5]` for all three variants.

The output tree is isolated under
`outputs/cosi_mag_final_ablation/<variant>/`. The full CoSI-MAG row is reused
from `outputs/cosi_mag_final_benchmark/`; it is not retrained. The summarizer
checks model configuration, task protocol, dataset config and split path,
split-file SHA256, base seed, run seeds, and run count before combining
results. It reports test metrics as mean ± population standard deviation in
their native [0, 1] scale, and writes paired per-seed deltas as
`ablation - CoSI-MAG`. It performs no significance test.

## Correctness answers

1. **Does MRC removal cut both MRC → SMP and MRC → RCMI?** Yes. SMP receives
   unit physical-edge weights for both modalities, and RCMI receives explicit
   all-node zero relation context.
2. **Can isolated nodes retain fake relation context?** No. `no_mrc` sets both
   context vectors directly to zero, including isolated nodes.
3. **Does the semantic-anchor ablation account for the gamma initialization
   basis coupling?** Yes. Alpha is zero during construction, so
   `gamma_global` is initialized in the ordinary monomial basis.
4. **Is `no_rcmi` disconnected from every RCMI parameter?** Yes. It pools
   states before refinement and does not call the RCMI computations. Tests
   perturb the RCMI parameter groups and check output invariance plus absent or
   zero gradients.
5. **Does LP retain `[5, 5, 5]` sampling?** Yes, through the inherited full
   sampler-depth capability at `K=3`.
6. **Is `full` mathematically equivalent to the final model?** The full mode
   delegates to the parent implementation unchanged. With the same state
   dict, CPU tests require exact equality for relation weights, normalized
   operators, every state, attention, eta, modality outputs, and fused output.

## Verification performed

The targeted CPU correctness suite covers full equivalence, synthetic graphs
with an isolated node, MRC-path perturbations, semantic-anchor recurrence and
prior basis, RCMI parameter disconnection and gradients, and LP sampler depth.
The launcher dry-run prints all nine units and reports zero `src.main`
invocations. No formal Movies, Grocery, or Sports training was started.
