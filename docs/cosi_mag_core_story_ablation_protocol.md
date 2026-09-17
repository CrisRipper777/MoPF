# CoSI-MAG Core Story Ablation Protocol

Status: implementation and smoke-test freeze. This document defines the new
paper-facing Core Story ablation suite; it does not contain performance
results and it does not authorize formal training in this implementation
step.

## 1. Scientific motivation

CoSI-MAG is presented as one Continuous Structure–Semantic Interaction story
with three stages:

1. **Stage I — Modality-Aware Relation Calibration**: the shared physical
   support receives modality-specific semantic conductances.
2. **Stage II — Semantic-Preserving Multi-Hop Propagation**: each modality
   propagates through an anchored state bank while retaining intrinsic
   modality semantics.
3. **Stage III — Adaptive Multi-Order Composition**: the response bank is
   composed with learned global, modality, node, and relation-conditioned
   preferences.

The new ablations remove one stage-level mechanism at a time. Full CoSI-MAG
is an external fixed reference and is not retrained here. Historical F2
outputs remain on disk but are not inputs to this suite.

## 2. Core variants

| Variant | Removed mechanism | Exact retained path |
|---|---|---|
| `wo_relation_calibration` | Stage I modality-aware relation calibration | Raw uniform conductance on the original physical support; Stage II/III and downstream paths unchanged |
| `wo_semantic_anchor` | Stage II semantic anchor | Existing historical ordinary-state + cumulative-response implementation; Stage I/III unchanged |
| `wo_adaptive_composition` | Stage III adaptive hop composition | Same cumulative response bank, composed by an arithmetic mean; all refinement/fusion/head paths unchanged |

The old `wo_learned_semantic_calibration` remains a legacy F2 variant. Its
meaning is still `learned_diag_cos -> separate_cos`; it is not aliased to
`wo_relation_calibration`.

## 3. Mathematical definitions

For the physical edge set (E), Full uses the learned diagonal metric and

\[
w_{ij}^m=w_{\min}+(1-w_{\min})\sigma(s_{ij}^m/T),
\qquad m\in\{\text{text},\text{visual}\}.
\]

### A1: `wo_relation_calibration`

The ablation resolves `edge_weight_mode=raw_uniform`, so

\[
w_{ij}^{\text{text}}=w_{ij}^{\text{visual}}=1
\quad\text{for every observed }(i,j)\in E.
\]

The same physical support, self-loop policy, GCN normalization, K, anchored
response bank, refinement, fusion, head, and task protocol are retained. The
incident-conductance mean is one for every connected node and for isolated
nodes after graph-mean filling; hence its centered transport context is zero.
No replacement relation condition is introduced.

### A2: `wo_semantic_anchor`

The resolver forces

\[
S_k^m=\hat A^mS_{k-1}^m,
\qquad k\ge 1,
\]

while the response mode remains cumulative, (R_k^m=S_k^m). Relation
calibration, K, adaptive composition, refinement, fusion, and task protocol
are unchanged. This is the existing historical `wo_semantic_anchor` key and
is reused without changing its semantics.

### A3: `wo_adaptive_composition`

The same response bank (R_0^m,\ldots,R_K^m) is retained, but the final
modality representation is

\[
Z_i^m=\frac{1}{K+1}\sum_{k=0}^{K}R_{i,k}^m.
\]

This explicit `composition_mode=uniform` branch bypasses
`gamma_global`, modality residual Δγ, node residual δ, and
relation-conditioned residual τ for the final modality representation.
The adaptive parameters may remain instantiated for state-dict compatibility;
they are not used by this final composition.

## 4. Code/config mapping

| Story element | Implementation |
|---|---|
| Central resolver | `src/ablation.py`: `AblationSpec`, `CORE_STORY_ABLATIONS`, `effective_edge_weight_mode`, `effective_composition_mode`, `effective_multihop_modes` |
| A1 override | `wo_relation_calibration` has `edge_weight_override="raw_uniform"`; legacy `learned_relation_calibration` is not overloaded |
| A2 override | `effective_multihop_modes` forces `ordinary`; `_build_multihop_banks` retains cumulative responses |
| A3 override | `composition_mode="uniform"`; `MoPF._compose_responses` takes the exact response-bank mean |
| Stage I | `MoPF._semantic_edge_weights`, `_normalized_operator`, `_mean_incident_conductance`, `_transport_context` in `src/models/mopf.py` |
| Stage II | `MoPF._propagation_bank`, `_build_multihop_banks` in `src/models/mopf.py` |
| Stage III coefficients | `MoPF._effective_coefficients` and `MoPF._filter_bases`; adaptive Full path is preserved |
| Final aggregation | `MoPF._encode_components`: compose modality streams, apply modality refinement, fusion skip/MLP, and output normalization |
| Base model config | `configs/model/mopf.yaml`, with `composition_mode: adaptive` for Full |
| Formal K map | `src/formal_protocol.py` and matching explicit runner overrides |
| Formal runner | `scripts/run_core_story_ablation.sh`, sequential and never schedules Full |
| Completeness gate | `scripts/check_core_story_ablation.py` |
| Smoke runner | `scripts/run_core_story_ablation_smoke.py` |

## 5. Invariant-test and Full-regression results

The complete repository test suite passed:

```text
209 passed, 6 warnings
```

The Core Story targeted suite plus the existing MoPF/F2 regression tests
passed 63/63 cases. The new tests cover raw-unit weights, identical
text/visual normalized operators, unchanged physical support, zero centered
relation context, ordinary/cumulative A2 banks without H0 reinjection,
exact A3 response means, finite forward/backward behavior, and shape
compatibility.

The fixed-input, fixed-initial-state-dict Full regression compares the
current Full path with the pre-Core-Story baseline
`97b57a02884d7ce3441b5500d000c41042e19fa3:src/models/mopf.py`, so the gate
remains meaningful after the implementation is committed. The following
maximum absolute differences were all exactly zero:

```text
projected h_text, h_visual             0.0
calibrated edge/operator tensors       0.0
state and response banks               0.0
eta_text, eta_visual                   0.0
z_text, z_visual                        0.0
fused z                                0.0
classifier logits                      0.0
```

State-dict keys are identical and the historical state dict loads strictly
into the current Full model. No formal training is permitted if this gate
fails.

## 6. Per-dataset formal protocol

The repository-authoritative task protocols are retained exactly:

| Task | Protocol | Training | Selection | Reported test metrics |
|---|---|---|---|---|
| NC | `unified_full_graph_nc_v1` | Full graph, AdamW, lr `1e-3`, weight decay `1e-4`, hidden `256`, dropout `0.2`, max 300 epochs, patience 30, validation every epoch, gradient clip 1.0 | best validation Accuracy | Accuracy and Macro-F1 |
| LP | `unified_sampled_lp_v1` | Sampled LinkNeighborLoader, Adam, lr `1e-3`, weight decay `1e-5`, hidden `256`, dropout `0.2`, max 150 epochs, patience 10, filtered negatives, configured neighbor sampling, exact full-graph inference | best validation MRR | MRR, Hits@1, Hits@3, Hits@10 |

The formal implementation freezes the existing split sources, loss, evaluator,
negative sampling, edge masking, decoder, and inference settings. Only the
named Core Story mechanism changes.

## 7. Confirmed formal K

| Task | Dataset | K | Authority |
|---|---|---:|---|
| NC | Movies | 3 | final resolved MoPF configuration and final protocol |
| NC | Toys | 3 | final resolved MoPF configuration and final protocol |
| NC | Grocery | 2 | final resolved MoPF configuration and final protocol |
| NC | ele-fashion | 3 | final resolved MoPF configuration and final protocol |
| NC | Reddit-S | 3 | final resolved MoPF configuration and final protocol |
| LP | sports-copurchase | 3 | `outputs/f1_final_execution/lp_formal/sports-copurchase/mopf/seed42/resolved_config.yaml` |
| LP | cloth-copurchase | 3 | `outputs/f1_final_execution/lp_quasi_held_out/cloth-copurchase/mopf/seed42/resolved_config.yaml`; promoted to this user-requested Core Story scope |

`Grocery` is the only K=2 dataset. The new runner passes both
`model.max_order=K` and `model.num_layers=K` explicitly for all seven datasets,
including both LP datasets.

## 8. Smoke-test results

The smoke runner executed only six cases: Movies NC and sports-copurchase LP,
each with seed 42 and each of the three Core Story variants. It used one epoch,
one training batch, and `evaluate_test=false`.

```text
6/6 passed
NC: train/backward/validation/checkpoint/manifest/metrics passed
LP: train/backward/validation/checkpoint/manifest/metrics passed
formal_training_launched: false
```

Smoke outputs are isolated under `outputs/core_story_ablation_smoke/` and are
not formal results.

## 9. Exact 63-run matrix

Each cell below means three variants × seeds 42, 43, 44.

| Task | Dataset | K | Runs |
|---|---|---:|---:|
| NC | Movies | 3 | 9 |
| NC | Toys | 3 | 9 |
| NC | Grocery | 2 | 9 |
| NC | ele-fashion | 3 | 9 |
| NC | Reddit-S | 3 | 9 |
| **NC subtotal** |  |  | **45** |
| LP | sports-copurchase | 3 | 9 |
| LP | cloth-copurchase | 3 | 9 |
| **LP subtotal** |  |  | **18** |
| **Total** |  |  | **63** |

Variants are exactly `wo_relation_calibration`, `wo_semantic_anchor`, and
`wo_adaptive_composition`. Full is not a matrix row.

## 10. Exact manual commands

Run these manually in order after committing the implementation. The commands
below are provided only; they were not executed as formal training.

### Step A — dry-run all 63

```bash
conda run --no-capture-output -n yhf_env bash scripts/run_core_story_ablation.sh \
  --task all \
  --datasets Movies,Toys,Grocery,ele-fashion,Reddit-S,sports-copurchase,cloth-copurchase \
  --seeds 42,43,44 \
  --variants wo_relation_calibration,wo_semantic_anchor,wo_adaptive_composition \
  --device cuda:0 \
  --require-clean-git \
  --dry-run
```

### Step B — NC, 45 runs

```bash
conda run --no-capture-output -n yhf_env bash scripts/run_core_story_ablation.sh \
  --task nc \
  --datasets Movies,Toys,Grocery,ele-fashion,Reddit-S \
  --seeds 42,43,44 \
  --device cuda:0 \
  --require-clean-git
```

### Step C — NC completeness

```bash
conda run --no-capture-output -n yhf_env python scripts/check_core_story_ablation.py \
  --task nc \
  --datasets Movies,Toys,Grocery,ele-fashion,Reddit-S \
  --seeds 42,43,44
```

### Step D — sports LP, 9 runs

```bash
conda run --no-capture-output -n yhf_env bash scripts/run_core_story_ablation.sh \
  --task lp \
  --datasets sports-copurchase \
  --seeds 42,43,44 \
  --device cuda:0 \
  --require-clean-git
```

### Step E — sports completeness

```bash
conda run --no-capture-output -n yhf_env python scripts/check_core_story_ablation.py \
  --task lp \
  --datasets sports-copurchase \
  --seeds 42,43,44
```

### Step F — cloth LP, 9 runs

```bash
conda run --no-capture-output -n yhf_env bash scripts/run_core_story_ablation.sh \
  --task lp \
  --datasets cloth-copurchase \
  --seeds 42,43,44 \
  --device cuda:0 \
  --require-clean-git
```

### Step G — final 63-run completeness

```bash
conda run --no-capture-output -n yhf_env python scripts/check_core_story_ablation.py \
  --task all \
  --datasets Movies,Toys,Grocery,ele-fashion,Reddit-S,sports-copurchase,cloth-copurchase \
  --seeds 42,43,44
```

The runner is sequential by design. Add `--skip-existing` only when the
corresponding directory has already passed the artifact checks.

## 11. Output structure

```text
outputs/core_story_ablation/
  nc/<dataset>/<variant>/seed42/
  nc/<dataset>/<variant>/seed43/
  nc/<dataset>/<variant>/seed44/
  lp/<dataset>/<variant>/seed42/
  lp/<dataset>/<variant>/seed43/
  lp/<dataset>/<variant>/seed44/
```

Every run must contain `resolved_config.yaml`, `resolved_config.json`,
`ablation_manifest.json`, `train.log`, `metrics.json`, `best.pt`, and
`complete.marker`. The manifest
records effective and configured stage settings, K, alpha, hidden size,
dropout, optimizer, lr, weight decay, epochs, patience, batch/sampling
configuration, split source, checkpoint selection, task protocol, branch,
and commit SHA. `metrics.json` records the checkpoint reference, best epoch,
runtime, seed, identity, and task metric schema.

## 12. Completeness-gate criteria

`check_core_story_ablation.py` reports planned, complete, incomplete, and
missing runs, plus duplicate/collision runs. A formal matrix is releasable
only when:

- the selected plan has exactly 45 NC, 18 LP, and 63 total runs;
- every planned directory is complete and no planned path collides;
- no unplanned manifest identity or duplicate identity exists;
- every manifest has the required effective-stage fields and exact identity;
- resolved configs match the frozen dataset K and task protocol;
- metrics contain exactly the expected finite mean/std schema;
- checkpoint paths point to the run's `best.pt` and selection is NC Accuracy or
  LP MRR as appropriate;
- all run manifests carry one expected branch and one expected commit SHA;
- no Full directory is required or accepted by this checker.

## 13. Full-reference policy

Full CoSI-MAG is not retrained, and no Full value is copied or inferred from
`outputs/f2_ablation/` or any historical output. Post-run analysis may accept
an external reference table with this schema only:

```text
task,dataset,metric,full_mean,full_std,source_note
```

Mean/std-only Full references do not imply paired seed observations.

## 14. Known caveats

- The current worktree is intentionally modified by this implementation;
  commit it before using `--require-clean-git` for formal execution.
- Existing repository prose classifies Cloth LP as quasi-held-out. The present
  user-requested Core Story scope explicitly promotes it into the new 63-run
  ablation matrix; its established resolved MoPF protocol confirms K=3.
- Smoke metrics intentionally omit test metrics (`evaluate_test=false`) and
  must never be aggregated as formal results.
- The checker validates protocol/configuration and provenance, but it does not
  analyze performance or make paper conclusions.
