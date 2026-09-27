# P0 — Relation-Conditioned Semantic Operator Audit

## 1. Scientific question

On a fixed real physical graph, does transforming neighbor messages according to the intrinsic Text/Visual relation improve MAG node classification over ordinary physical propagation or semantic scalar weighting? The controlled contrast is message weighting versus message transformation. Physical edges always carry messages; semantic information can interpret them but does not delete connectivity.

This P0 does not include hop routers, trajectory attention, relation-context co-evolution, cross-modal routing, auxiliary losses, or expert regularization.

## 2. Shared frozen backbone

Each modality uses an independent projector copied in structure from `cmrf_probe`:

`Linear(input_dim, 256) → LayerNorm(256) → ReLU → Dropout(0.2)`.

The graph is canonicalized once per model forward exactly as in CMRF: remove self-loops, undirect, add one self-loop per node, coalesce, use unit edge weights, and apply symmetric normalization. Sparse coordinate `(row, col)` means `P[row=target, col=source]`. The custom edge path gathers `C[source]`, multiplies by the coefficient at `(target, source)`, then accumulates at `target`.

All variants use `K=3`, `C0=H0`, and uniform readout `(C0+C1+C2+C3)/4`. Text/Visual fusion is the CMRF `plain_fusion`: `Linear(512,256) → ReLU → Dropout(0.2) → Linear(256,256)`.

## 3. Variants

### A0 — `plain`

For each modality and hop, `Ck = sparse.mm(P, C{k-1})`. This is the reference and is tested against CMRF's `composition_mode=uniform` at state, readout, and fused-output levels with maximum error below `1e-6`.

### A1 — `scalar_weight`

A modality-specific node reducer maps 256 to 64. For directed physical coordinate `(i,j)`, the pair descriptor is `[u_i, u_j, u_i-u_j, u_i⊙u_j]` with shape 256. The relation MLP is `Linear(256,128) → GELU → Linear(128,64) → LayerNorm(64)`. A zero-initialized four-head projection produces `s_ij = mean_h(2 sigmoid(l_ijh))`, exactly 1 at initialization. Non-self message is `s_ij C_j`; self-loop messages bypass this module.

### A2 — `global_expert`

Each modality has the same relation encoder, query, four learned 64D prototypes, and four experts as A3. Relation states are mean-pooled deterministically over non-self physical edges. The mean token produces one modality-global softmax mixture shared by all edges.

### A3 — `relation_expert`

A modality-specific relation token `r_ij` is computed from H0 once per forward and reused for all three hops. `q_ij=Wq r_ij`; routing is `softmax(q_ij p_r / sqrt(64))`. Each expert is a residual bottleneck operator `Linear(256,64) → GELU → Linear(64,256)`. The message is `C_j + sum_r pi_ijr E_r(C_j)`. The physical base message remains present for every edge.

A2 and A3 instantiate the same modules in the same dimensions; their trainable parameter counts are exactly equal and a regression test enforces this.

## 4. Initialization and edge memory

A1's last scalar layer has zero weights and bias, so all edge scalars start at one. Expert down-projections use PyTorch's standard Linear initialization; each final up-projection uses normal standard deviation `1e-3` and zero bias. Small, nonzero expert outputs leave a gradient path into routing and prototypes without replacing the stable physical base path at initialization.

`edge_chunk_size` defaults to 65,536. Relation MLP/router work is checkpointed chunk-by-chunk and only the small scalar or E×4 routing controls remain for the three hops. Each propagation chunk builds temporary expert activations of shape `[E_chunk,4,256]`, applies normalized weights, and scatter-adds to target nodes. No full `[E,4,256]` tensor is constructed or retained. Node feature states remain `[N,256]` per hop.

Self edges are split from non-self edges after normalization; each self message is exactly `C_i` before applying its normalized physical coefficient and never enters a relation encoder, scalar head, or expert.

## 5. NC protocol and test isolation

The launcher uses `configs/task/nc.yaml` and explicitly records these settings: `unified_full_graph_nc_v1`, full graph, hidden 256, max order 3, 300 maximum epochs, AdamW, learning rate `1e-3`, weight decay `1e-4`, patience 30, minimum early-stop epoch 30, gradient clip 1.0, Validation Accuracy checkpoint selection, Validation Accuracy and Macro-F1 reporting, `num_runs=1`, and one of seeds 42/43/44. The config's `early_stop_min_delta=1e-4` is retained.

Every launcher command sets `task.evaluate_test=false`. The NC training path transfers only Train labels during full-graph training; Macro-F1's fixed label set uses Train/Validation labels. The analyzer checks resolved configs, run metrics, and checkpoint payloads for test evaluation/metrics. Mechanism work indexes only Train and Validation nodes/edges; it does not inspect test labels.

Checkpoints retain the repository payload format: model and classifier states, data info, selection name, best epoch, and validation metrics. Each formal context has a deterministic run directory and its own checkpoint.

## 6. Launch and resume behavior

The formal matrix is 4 variants × 5 datasets × 3 seeds = 60 contexts. Successful contexts with a checkpoint, completion marker, resolved config, and test-disabled metrics are skipped on later invocations. Failed contexts record a reason and can be retried. Smoke artifacts live under `outputs/relation_operator_audit_v1/smoke/` and never count as formal runs.

Exact formal launch command:

```bash
python scripts/run_relation_operator_audit.py --mode formal --device cuda:0
```

Dry-run the planned commands without starting jobs:

```bash
python scripts/run_relation_operator_audit.py --mode formal --device cuda:0 --dry-run
```

One-context smoke command:

```bash
python scripts/run_relation_operator_audit.py --mode smoke --dataset Movies --variant relation_expert --seed 42 --epochs 1 --device cuda:1
```

## 7. Metrics and analysis

The analyzer creates `results/relation_operator_audit_v1/` with the formal manifest, validation table, parameter counts, paired comparisons, intervention table, expert specialization, semantic relation diagnostics, decision matrix, preflight report, and report. Paired comparisons report mean delta, population SD, positive paired seeds out of 15, positive dataset means out of 5, and per-dataset mean deltas for A3−A2, A3−A1, A3−A0, and A2−A0.

Frozen A3 interventions are per-edge route shuffle, replacement by modality-global mean routing, and zero expert residual with the base physical path retained. Mean-expert replacement is also recorded. The analysis records Validation metrics, prediction flips, routing changes, and relative message changes.

Semantic quartiles use cosine similarity between projected H0 endpoints. Boundaries come only from physical edges whose endpoints are both in Train; diagnostics summarize edges incident to at least one Validation node. A1 reports scalar-weight mean, median, P10, and P90. A3 reports relative correction size, message-direction cosine, and expert usage per quartile/hop. Optional matched-size random edge-group controls are available with `--targeted-edge-interventions`.

Expert reports include average load, routing entropy, top-1 fraction, pairwise expert-output cosine on shared Train inputs, and mean-expert intervention impact. No regularizer is added. An observed usage above 90% or pairwise output cosine above 0.98 is flagged as EXPERT_COLLAPSE_OBSERVED.

## 8. Tests

The dedicated test file covers CMRF equivalence, dense `P@H` orientation, self-loop bypass, A2/A3 parameter parity and common-token equivalence, chunk invariance, A3 gradient paths, safe initialization, identical topology, and test-disabled NC metrics/checkpoint fields.

Run the dedicated tests:

```bash
/home/m3/miniconda3/envs/yhf_env/bin/python -m pytest tests/test_relation_operator_audit.py -q
```

Relevant existing regression tests:

```bash
/home/m3/miniconda3/envs/yhf_env/bin/python -m pytest tests/test_cmrf_probe.py tests/test_routing_audit.py tests/test_nc_metrics.py tests/test_paper_nc_runner.py -q
```

## 9. Limitations and explicit audit notes

- The current dataset loader still reads the dataset's full label tensor into CPU memory; the frozen NC protocol uses only Train labels for loss and Train/Validation labels for selection/metrics, and the P0 full-graph training path no longer transfers test label values to GPU.
- Chunk checkpointing trades additional backward recomputation for bounded edge activation memory. `edge_chunk_size` can be lowered if Reddit-S smoke reveals memory pressure; hidden size, expert count, K, and protocol remain fixed.
- The H2 wording “catastrophic negative transfer” has no numerical cutoff in the request. The report exposes per-dataset deltas for explicit review rather than introducing an unregistered threshold.
- The optional targeted edge intervention is a Validation diagnostic, not a causal estimate of per-edge utility.
- Formal training is deliberately not started as part of implementation, unit tests, and smoke validation.
## 10. Implementation verification performed

- Dedicated P0 tests: **13 passed**.
- Combined P0 + CMRF + historical routing-audit + NC metric/launcher tests: **42 passed**. Only upstream PyG/checkpoint deprecation warnings were emitted.
- CUDA smoke: Movies, all four variants, seed 42, one epoch each on `cuda:1`; all full-graph train/validation/checkpoint paths completed. Peak allocated memory: plain 462.6 MiB, scalar_weight 839.6 MiB, global_expert 1605.9 MiB, relation_expert 1613.0 MiB.
- CUDA smoke: Reddit-S relation_expert, seed 42, one epoch, `edge_chunk_size=65536`; completed without OOM, peak allocated memory 1612.3 MiB.
- The smoke logs, resolved configs, metrics, and checkpoints are under `outputs/relation_operator_audit_v1/smoke/`; they are not included in formal aggregation.
- Current parameter counts are materialized for all five dataset feature shapes in `results/relation_operator_audit_v1/parameter_counts.csv`. For Movies: model-only counts are 591,872 (plain), 707,848 (scalar_weight), and 980,736 each (global_expert and relation_expert). Dataset-specific head counts are listed separately in that CSV.
- The analyzer was run before formal training and reports **0/60** completed formal contexts with all H1–H3 decisions `PENDING`. No formal job was started.

Full formal training remains a separate user-triggered action using the command in Section 6.
