# IAMOC v1 Implementation Notes

## Scope and formal-model audit

The active branch is `exp/iamoc-v1`. Formal MoPF Stage III is implemented in
`src/models/mopf.py::_encode_components`, with the node preference scorer in
`_node_residuals`, coefficient construction in `_effective_coefficients`, and
response composition in `_compose_responses`.

The inherited flow is:

1. `text_proj` and `visual_proj` form modality-specific `H_0`.
2. `_semantic_edge_weights` calibrates modality-specific edge weights.
3. `_normalized_operator` and `_build_multihop_banks` form the anchored
   multi-hop states and the formal response bank.
4. `_node_residuals` scores each order independently with the inherited
   low-rank projectors and order vectors.
5. `_transport_context` forms a centered local conductance context.
6. `_effective_coefficients` combines global, modality, node, and (when
   enabled) TCPR terms; `_compose_responses` combines the unchanged response
   bank.
7. The inherited modality refiners and concat-residual MLP fusion produce the
   task-facing embedding.

The formal configuration uses `learned_diag_cos`, anchored/cumulative states,
anchor alpha `0.1`, trainable global filters, modality/node/transport
residuals, adaptive composition, and `concat_residual_mlp` fusion. The original
node-specific residual scorer sees each order separately, without explicit
cross-order interaction.

## IAMOC implementation

`src/models/mopf_iamoc.py` subclasses `MoPF`. It reuses the parent projection,
relation calibration, graph operator, propagation, transport context,
response composition, fusion, forward contract, and inference contract. Its
only training-path change is the input to the existing node-specific scorer:

```text
per-modality states [N, K+1, d]
  -> learned order embedding (interaction branch only)
  -> shallow within-node hop attention
  -> gated residual interaction states
  -> inherited low-rank node scorer
  -> eta over the original Stage-II response bank
```

Each modality has independent attention parameters and a scalar gate initialized
to `tanh(theta)=0.10`. Attention is single-head scaled dot-product attention
with pre-LayerNorm and no Transformer FFN. Order embeddings and attention
outputs never replace the structural response bank.

`relation_conditioning` supports the three requested locations:

- `output`: inherited TCPR remains in eta; relation context does not enter
  attention.
- `none`: relation context is excluded from attention and eta.
- `attention`: centered relation context times the centered transport order
  profile is added to each key-order logit; eta omits TCPR.

`analysis_hop_attention` is an analysis-only API. It captures attention and
supports `normal`, `off`, and fixed-seed `shuffle` relation-bias interventions.
The normal training default keeps `export_hop_attention: false`, so the
node-level attention tensor is not retained in normal component outputs.

## Variants and run commands

The launcher maps V0–V4 exactly as described in the experiment brief. Full runs
default to Movies NC, Grocery NC, sports-copurchase LP, seeds 42/43/44, and
`cuda:0`. Each run has a separate directory below `outputs/iamoc_v1/` and stores
the complete command/config/status in `run_record.json`.

For IAMOC LP runs, the launcher explicitly resolves the sampler to `[5, 5, 5]`.
The frozen LP task runner extends `[5, 5]` to three hops only for the exact
model name `mopf`; setting the same effective depth for `mopf_iamoc` preserves
the formal three-order sampled message-passing neighborhood without modifying
task code.

```bash
conda run --no-capture-output -n yhf_env python scripts/run_iamoc_v1.py --dry-run
conda run --no-capture-output -n yhf_env python scripts/run_iamoc_v1.py --smoke --device cuda:0
conda run --no-capture-output -n yhf_env python scripts/run_iamoc_v1.py --resume --device cuda:0
```

The smoke option is limited to Movies NC, seed 42, one epoch per variant. A
completed run is skipped only when `--resume` is supplied and its checkpoint,
completion marker, resolved config, metrics, and results are present.

## Correctness and mechanism analysis

Run the synthetic correctness checks before training:

```bash
conda run --no-capture-output -n yhf_env python scripts/analyze_iamoc_v1.py --correctness-only
```

Checkpoint analysis exports, per node, key-mass order profiles, entropy,
effective attended order, effective eta order, relation context, and physical
degree. It also writes Text/Visual mean attention matrices and calculates
partial Spearman association after controlling for `log(1 + physical degree)`.
V3 receives the normal, relation-off, and fixed-seed relation-shuffle
inference-only interventions.

```bash
conda run --no-capture-output -n yhf_env python scripts/analyze_iamoc_v1.py --device cuda:0 --summarize
```

All attention and node-level evidence is written as CSV plus JSON under
`outputs/iamoc_v1/analysis/`; training does not save full node-level attention.

## Frozen behavior

The implementation does not edit formal MoPF, its model configuration or
factory, task runners, datasets/splits, classifier, LP decoder, training or
validation checkpoint protocol, evaluation protocol, or existing ablation
definitions. Outputs remain under `outputs/iamoc_v1/`. No Git push is performed.
