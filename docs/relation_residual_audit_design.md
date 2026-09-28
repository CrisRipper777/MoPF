# P0-R design and execution record

## Research scope

P0-R tests whether the useful shared semantic transformation identified in P0-D can be consolidated into one operator, while moving relation-specific capacity into small residual adapters. R2 adds the same local structural response used in P0-D. This phase implements the fixed design and engineering checks only; it does not launch the formal 60-run matrix.

The four variants are:

| ID | Config value | Shared operator | Route/adapters | Relation evidence |
|---|---|---:|---|---|
| R0 | `shared_prior` | yes | none | none |
| R0G | `global_residual` | yes | three adapters, mean edge route | endpoints + learned NULL context |
| R1 | `attribute_residual` | yes | three adapters, per-edge route | endpoints + learned NULL context |
| R2 | `context_residual` | yes | three adapters, per-edge route | endpoints + differentiable structural response |

R0G, R1, and R2 have identical trainable model parameter counts for each dataset. Their routing paths all remain present; R0G computes each edge route before taking its mean across physical non-self edges.

## Frozen common model

Text and Visual each use `Linear → LayerNorm → ReLU → Dropout` to produce 256-dimensional H0. The physical operator removes input loops, makes the graph undirected, adds one loop per node, coalesces coordinates, assigns unit weights, then applies symmetric normalization. COO rows are targets and columns are sources, so sparse multiplication aggregates source messages into targets.

Each modality has its own shared operator:

`E_sh(h) = U_sh GELU(V_sh h)`, with dimensions `256 → 256 → 256`.

The up projection uses normal initialization with standard deviation `1e-3` and zero bias. R0 applies only this shared transformation to non-self messages. R0G/R1/R2 add a weighted sum of three modality-specific `256 → 32 → 256` residual adapters. Their up projections use the same small initialization. The fourth route entry is fixed no-adaptation: it has no module and contributes exactly zero.

For R2, each modality computes `D = P_nbr H0 - H0` once per forward, where `P_nbr` is the normalized operator with self-loop coordinates removed. This calculation keeps autograd enabled and the resulting D is reused to route all three hops. R1/R0G use learned target/source NULL vectors in those same context-token positions. No topology, edge weights, or labels enter D.

For each non-self edge, the contextual encoder builds four 64-dimensional evidence tokens: target intrinsic, source intrinsic, target context, source context. Role and evidence-kind embeddings distinguish them. A learned 64-dimensional relation query attends to those tokens with one 4-head MultiheadAttention block, residual LayerNorm, and a `64 → 128 → 64` FFN. A linear coordination head produces four route logits; its bias initializes no-adaptation to `log(3)` and adapter logits to zero. There is no extra scalar gate.

At each of three hops, self-loop messages copy the prior node state using their normalized physical edge weight and bypass relation encoding, the shared operator, and adapters. Non-self messages apply the shared transformation and optional residual mixture. Each modality reuses its H0/D-derived route across all hops. Readout is the uniform average of H0 and C1–C3. Text and Visual readouts concatenate into the frozen `Linear(512→256) → ReLU → Dropout → Linear(256→256)` fusion block.

The only task objective is NC cross-entropy on Train nodes. The fixed formal protocol is full-graph training, AdamW, `lr=1e-3`, `weight_decay=1e-4`, 300 epochs, patience 30, minimum epoch 30, gradient clip 1.0, Validation-Accuracy checkpoint selection, Validation Accuracy/Macro-F1 reporting, and `task.evaluate_test=false`.

## Memory behavior and implementation note

Only the `[E,4]` route matrix is retained across hops. Evidence tokens and adapter outputs are built per edge chunk; the implementation does not materialize persistent `[E,5,64]` or `[E,4,256]` tensors. `edge_chunk_size` bounds graph work. A CUDA smoke attempt at a 65,536-edge attention batch encountered a PyTorch fused SDPA invalid-configuration error. To keep the specified MHA architecture while selecting the general MHA path, the attention call requests attention weights (the weights are discarded) and relation encoding applies an 8,192-edge internal cap. Thus the configured chunk size is still an upper bound, while relation attention uses `min(edge_chunk_size, 8192)`. This changes kernel/chunk execution only, not the relation equations or parameterization.

## Diagnostics

The model returns `rr_` mechanism summaries for each modality and hop: shared and residual relative magnitudes, distribution quantiles, residual threshold fractions, no-adaptation route distribution, adapter usage, routing entropy, top-1 fractions, and pairwise adapter-output cosine. `src/tasks/common.py` accepts these statistics additively and transfers batched scalar summaries to CPU.

The analyzer writes validation-only normal metrics, paired comparisons, interventions I1–I5, node-level validation CE utility, R2 context sensitivity, R2 shrinkage, parameter counts, and mechanism statistics. training_mechanism_statistics.csv.gz preserves the rr_ summaries logged at every training epoch; mechanism_statistics.csv reports frozen best-checkpoint reevaluation. Context quartiles use Train–Train thresholds; semantic similarity uses frozen plain A0 H0 and the P0-D binning helper. Historical P0 comparisons are explicitly descriptive across experiment families. The raw node table is gzip-compressed and ignored by Git.

`acceptance_decisions.csv` presents H-R0 through H-R3 evidence. It does not assign pass/fail based on invented cutoffs: the prompt defines qualitative terms such as “markedly” and “noticeably” but no numeric thresholds. Formal rows remain pending until the complete matrix is analyzed and then require evidence-based scientific adjudication. Smoke rows are always marked `SMOKE_ONLY`.

## Verification performed

- P0-R CUDA training smoke: Movies, seed 42, one epoch for R0/R0G/R1/R2; Reddit-S, seed 42, one epoch for R2. These are engineering checks, not scientific results.
- Full-graph R2 no-label forward on all five datasets, seed 42: all outputs finite. The largest graph was ele-fashion (97,766 nodes; 399,172 directed non-self coordinates) with about 2,233 MiB peak allocated GPU memory for that forward.
- Training smoke maximum observed peak: Reddit-S R2, 13,633 MiB, 2.45 seconds. Movies R0/R0G/R1/R2 took 1.39/2.00/1.91/2.04 seconds and used 4,388/7,986/7,986/8,021 MiB respectively. Times and memory are one-epoch smoke measurements only.
- Initial no-adaptation route mean across the four residual-model smoke contexts and two modalities: `π0=0.5026` (8 modality summaries), recorded before the first optimizer step. The post-smoke checkpoint mean across those summaries was about 0.5035. These values confirm conservative initialization and do not support a training-mechanism conclusion.
- Model parameters: Movies/Toys/Grocery/Reddit-S R0=855,040 and each residual variant=1,090,248; ele-fashion R0=723,968 and each residual variant=959,176. R0G=R1=R2 exactly for every dataset. NC head counts depend on class count and are tabulated separately.
- 55 focused P0-R plus historical P0/P0-D tests passed. The launcher dry-run produced exactly 60 unique formal commands. No formal run was started and no Test metric was evaluated.

Machine-readable smoke tables are in `results/relation_residual_audit_v1/`. The smoke run manifest and checkpoints remain under the ignored `outputs/relation_residual_audit_v1/` namespace. Smoke provenance records branch `relation_residual_audit`, HEAD `f953b2b3a1c49aad49705a5998c83dc149a7dcbb`, and `git_worktree_clean=false`: the P0-R implementation was still an uncommitted worktree during the engineering smoke. The formal launcher requires the final implementation to be committed and the worktree clean, then records the resulting SHA.

## Commands

One-epoch CUDA engineering smoke example:

```bash
python scripts/run_relation_residual_audit.py --mode smoke --device cuda:1 --dataset Movies --variants shared_prior global_residual attribute_residual context_residual --seed 42 --epochs 1
```

Formal launch command for a later authorized phase (not run here):

```bash
python scripts/run_relation_residual_audit.py --mode formal --device cuda:1
```

Formal frozen analyzer after all 60 runs complete:

```bash
python scripts/analyze_relation_residual_audit.py --mode formal --device cuda:1
```

Smoke analyzer command used for this record:

```bash
python scripts/analyze_relation_residual_audit.py --mode smoke --device cuda:1 --datasets Movies Reddit-S --seeds 42 --variants shared_prior global_residual attribute_residual context_residual
```
