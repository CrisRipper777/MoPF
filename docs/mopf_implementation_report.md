# MoPF implementation report

## Scope and changed files

The final MoPF encoder is implemented directly without altering MAP-v1,
MAP-v2, MAP-v3, DiP, baseline encoders, data splits, negative sampling, LP
decoder, or LP evaluation.  The changed/new files are:

- `src/models/mopf.py` — new `MoPF` encoder and `Model = MoPF` export.
- `configs/model/mopf.yaml` — default full-MoPF configuration.
- `tests/test_mopf.py` — encoder, filtering, sparse graph, backward,
  inference, empty-graph, and LP sampler tests.
- `src/tasks/lp.py` — a narrow MoPF-only sampler-depth adaptation.  Existing
  encoders retain their prior LP sampler behavior.
- `docs/mopf_implementation_report.md` — this report.

`factory.py` needs no registry change: its existing dynamic import resolves
`src.models.mopf` from `model=mopf`.

## Final data flow

```text
x_text ─ Projection_t ─ h_t ─ semantic A_t ─ Ahat_t ─ [H_t^0 ... H_t^K]
                                                    └─ eta_t ─ Z_t ─ refine_t

x_visual ─ Projection_v ─ h_v ─ semantic A_v ─ Ahat_v ─ [H_v^0 ... H_v^K]
                                                       └─ eta_v ─ Z_v ─ refine_v

concat(refine_t, refine_v) ── skip + fusion MLP ── LayerNorm ── z
```

The two modalities are not fused before their individual projection,
semantic-graph construction, polynomial propagation, and personalized
filtering are complete.  The public encoder contract is unchanged:
`z, None, None, aux_loss, aux_info = model(x, edge_index)`.  `aux_loss` is an
exact zero scalar and `aux_info` is empty in this first performance-oriented
version.

## Formula-to-code mapping

| Component | Formula / behavior | Code |
| --- | --- | --- |
| Independent projections | `Linear -> LayerNorm -> ReLU -> Dropout` independently for text and visual input | `ProjectionMLP`, `src/models/mopf.py:12-25`; instantiation at `143-144` |
| Semantic graph weights | `w_m = w_min + (1-w_min) sigmoid(cos_m / T)` on original sparse edges only | cosine and weighting at `212-234`; three modes at `236-263` |
| Separate normalized operators | `Ahat_t = gcn_norm(A_t)`, `Ahat_v = gcn_norm(A_v)` | `265-291`, invoked separately at `371-383` |
| Polynomial bank | `H_0^m=h_m`, `H_{k+1}^m=Ahat_m H_k^m`, retaining `0..K` | sparse scatter step `293-308`, bank `310-322`, construction `384-385` |
| Global prior | default `[0.15, 0.1275, 0.7225, 0]`; general MAP restart expansion | parameter `146-150`; initializer `180-194` |
| Modality residual | `gamma_m = gamma_global + delta_gamma_m` with zero-initialized `delta_gamma_m` | `151-152`, `340-353` |
| Node low-rank residual | `delta_i,k^m = tanh(W_k^m H_i,k^m) dot v_k^m / r` | projectors/vectors `154-165`, residual calculation `324-338` |
| Personalized filter | `eta_i,k^m = gamma_global,k + delta_gamma_m,k + delta_i,k^m`; `Z_i^m=sum_k eta_i,k^m H_i,k^m` | coefficient composition `340-353`, basis sum `355-360`, calls `393-396` |
| Modality refinement | `LayerNorm(Z_m + MLP_m(Z_m))` | `167-170`, `398-403` |
| Late fusion | `LayerNorm(Linear([Z_t;Z_v]) + MLP([Z_t;Z_v]))` | `172-174`, `404-407` |

No softmax, sigmoid, non-negativity constraint, or sum-to-one constraint is
applied to `gamma_global`, modality residuals, node residuals, or `eta`.

## Sparse semantic graphs and numerical behavior

`separate_cos` is the default and creates independent text and visual weights.
`shared_avg_cos` uses the average cosine to give both modalities the same
weights; `raw_uniform` gives both all-one weights.  All modes retain the input
`edge_index` cardinality and never construct an `N x N` adjacency or kNN
edges.  `gcn_norm` and sparse `scatter` are used for every propagation step.

Cosines are protected with `nan_to_num`; normalized weights and the final
embedding are also made finite.  An empty graph is valid: with default self
loops its polynomial bases retain each node's own feature; with self-loops
disabled, a missing normalized edge set yields zero bases after `H_0`.

The default propagation bank holds four `[N, hidden_dim]` bases per modality
for `K=3`, as permitted for the requested implementation.  No trainable
parameter has a node-count dimension; the node-specific behavior is generated
from rank-4 projections and `[K+1, rank]` vectors.

## Configuration and LP integration

`configs/model/mopf.yaml` uses the requested defaults: hidden size 256,
dropout 0.2, `max_order=num_layers=3`, MAP prior restart 0.15/order 2,
self-loops, `separate_cos`, weight floor 0.1, temperature 2.0, rank 4, and
all three hierarchical filter levels enabled.

`global_filter_trainable=false` freezes the MAP-style prior.  Disabling the
modality/node switches replaces the corresponding contribution by exact zero;
it does not select a different model class.

For NC, the unchanged task configuration selects full-graph training.  For
LP, MoPF is still trained by the unchanged sampled `LinkNeighborLoader` path,
with its existing filtered negatives, positive-message-edge masking, decoder,
and evaluation.  The only task-code change is `src/tasks/lp.py:332-359`:
MoPF's explicit three-order bank asks the existing `resolve_num_neighbors()`
helper to extend the task default `[5, 5]` to `[5, 5, 5]`.  Non-MoPF encoders
continue to receive the existing unmodified resolver behavior.

`inference()` at `src/models/mopf.py:437-452` performs exact full-graph
forward propagation, so LP full-graph evaluation and NC full inference use
the same MoPF computation as evaluation-mode `forward()`.

## Parameter counts

The encoder parameter count depends only on input modality dimensions, hidden
size, propagation order, and filter rank; it is independent of the number of
nodes and edges.

- 512-dimensional text + 512-dimensional visual inputs: **865,100** encoder
  parameters (ele-fashion and sports-copurchase).
- 768-dimensional text + 768-dimensional visual inputs: **996,172** encoder
  parameters (Movies).

Smoke-run logs include task heads: Movies NC model+classifier **1,001,312**,
ele-fashion NC model+classifier **868,184**, and sports-copurchase LP
model+projection+decoder **997,069**.

## Validation results

Executed in the repository's `yhf_env` with `python -m pytest`:

```text
python -m pytest tests/test_mopf.py -q  -> 7 passed
python -m pytest -q                     -> 104 passed, 1 upstream PyG deprecation warning
```

`tests/test_mopf.py` covers factory construction; shape/finite contract;
zero auxiliary loss; exact K=3 gamma prior; zero modality and node residual
initialization; initial `eta` equality to the MAP prior; sparse semantic
weight shapes; the two ablation edge modes; finite gradients for all requested
filter parameters; exact inference equivalence; `num_layers=3`; residual
switches; empty graph handling; and `[5,5] -> [5,5,5]` LP resolution.

## CUDA smoke tests

All requested one-epoch smoke tests completed on `cuda:0` (RTX 3090), without
numerical exceptions or non-finite outputs.

| Command target | Result |
| --- | --- |
| Movies NC | train loss 2.7993; val accuracy 32.93%; test accuracy 32.92%; test macro-F1 2.48% |
| ele-fashion NC | train loss 2.6729; val accuracy 43.38%; test accuracy 42.11%; test macro-F1 11.03% |
| sports-copurchase LP (`max_train_batches=2`) | train loss 0.7025; val MRR 6.48%; test MRR 5.82%; H@1/3/10 = 0.84% / 3.36% / 12.26% |

The LP log explicitly reports the resolved training neighbor sampler as
`[5, 5, 5]`.  These are smoke results only, not benchmark claims.

## Difference from MAP-MAG v3

MoPF is a separate module.  Unlike MAP-MAG v3, it has no reliability gate or
feature scaling, modality router, modality-balance loss, prototype path/loss,
conflict channel, self-residual gate, high-pass branch, MoE, cross-attention,
new decoder, or task loss.  It also does not use MAP-v3's restart recurrence:
it explicitly constructs the generalized polynomial bases and learns signed
global/modality/node coefficients over them.  Its fusion is only the requested
late concatenate-residual MLP.

## Readiness

The implementation is ready to start formal seed-42 downstream performance
experiments.  The full test suite and all requested NC/LP smoke paths pass;
no numerical instability was observed.  The next run should be the normal
seed-42 training schedule, not an expanded benchmark or analysis-script
campaign.
