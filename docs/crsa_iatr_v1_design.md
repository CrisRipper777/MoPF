# CRSA+IATR v1 design and experiment protocol

## Research objective

This experiment evaluates a two-stage structure–attribute interaction model built around the principle “interpret structure before absorbing structure.” CRSA contextualizes physical relations using endpoint attributes and local structural response. IATR then encodes the relation-aware propagation trajectory and lets each node's intrinsic embedding query that trajectory.

The only scientific factors are CRSA and IATR. No additional routing, gate, hop predictor, topology change, auxiliary loss, or multimodal attention is introduced.

## Stage I: CRSA

Text and visual embeddings are projected independently to 256 dimensions. For each modality, the local structural response is computed once from the non-self normalized physical operator as D = P_nbr H0 − H0. A relation encoder sees the two endpoint attributes and their structural responses, then predicts a four-way route. The route is reused through all three propagation steps.

Messages retain the frozen P0-R R2 form: source state plus one shared 256→256→256 bottleneck transform plus a routed residual from three independent 256→32→256 adapters. Physical self-loops use the direct normalized self path. Text and visual branches have independent parameters. The topology and normalized operator construction match the frozen P0-R implementation.

## Stage II: IATR

For each modality, the three propagation states and increments form three ordered tokens. A fixed one-layer Transformer encoder uses dimension 256, four heads, FFN 512, GELU, and dropout 0.2. H0 does not enter trajectory self-attention. Instead, it supplies one protected anchor query over the encoded three-token trajectory, followed by the specified residual normalization and 256→512→256 FFN.

IATR node chunking is exact because attention is restricted to the three tokens within a node; there is no attention across nodes. Full-graph propagation remains unchunked with respect to nodes. CRSA relation encoding and edge message updates use exact edge chunks.

## Variants

| Variant | CRSA | IATR | Modality readout |
|---|---:|---:|---|
| base | off | off | Uniform mean of H0, C1, C2, C3 |
| crsa | on | off | Uniform mean of H0, C1, C2, C3 |
| iatr | off | on | Intrinsic-anchor trajectory reconciliation |
| full | on | on | Intrinsic-anchor trajectory reconciliation |

The text/visual projectors and plain 512→256→256 fusion are shared in definition across all variants. Inactive factor modules are not instantiated. Factor-specific initialization uses isolated deterministic random streams so common initialization and the NC head stay aligned across ablations.

## Training protocol

- Task: full-graph node classification only.
- Datasets: Movies, Toys, Grocery, ele-fashion, Reddit-S.
- Seeds: 42, 43, 44.
- Variants: base, crsa, iatr, full; 60 contexts total.
- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4.
- Maximum epochs: 300; patience 30; minimum epoch 30; minimum delta 1e-4.
- Gradient clipping: 1.0; validation each epoch; no scheduler.
- Checkpoint: best validation accuracy.
- Test evaluation and modality-mask evaluation: disabled.

Smoke runs are stored under a separate outputs and results namespace and are engineering checks only. They do not enter the formal 60-context analysis.

## Validation and interpretation

The analyzer reads validation summaries and checkpoints to verify the saved model parameter count. It checks that Test evaluation was disabled and that no Test metric key was saved. It does not evaluate a model or access node labels. Reports include each dataset and seed, mean and sample standard deviation, paired descriptive differences, runtime, memory, and context status. No pass/fail threshold or significance cutoff is used.

No LP, books-nc, robustness, modality-missing experiment, noise experiment, hyperparameter sweep, or Test evaluation belongs to this stage.
