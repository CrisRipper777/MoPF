# CoSI-MAG VNext Implementation Report

Date: 2026-10-12 (Asia/Shanghai)

## Repository and implementation scope

- Worktree: `/hdd1/DataInHere/YHF/MoPF_IAMOC`.
- Branch: `exp/cosi-mag-vnext-crc-cheb-rcmi` tracking `origin/exp/cosi-mag-vnext-crc-cheb-rcmi`.
- Starting branch commit and target remote branch base: `d5a06103be0a27952ef5890e2839a36eb83f580a`, the specified CoSI Final SHA. The remote target was fetched and confirmed at that SHA before implementation.
- Implementation source commit used for the correctness record and pilot runs: `c5a901634da8ddd6bed1c4ad01b37b67326a3da4`. The final delivery commit adds only this commit provenance to the report and run manifest.
- Project data path: `/hdd1/DataInHere/YHF/data`. `CLAUDE.md` mentions the historical `MAG_baseline/` path; configs and actual loaders use the data path above.
- Runtime: `conda run -n yhf_env`; Python 3.12.13, PyTorch 2.4.0+cu121, PyG 2.7.0, `torch_sparse` 0.6.18+pt24cu121, Hydra 1.3.2, NumPy 2.4.3, scikit-learn 1.8.0, DGL 2.4.0+cu121. Pilot ran on RTX 3090 `cuda:1`.
- The base shell has incompatible/older Python packages and is not used for project commands. `nvidia-smi` could not initialize NVML; run metrics use PyTorch's CUDA memory accounting.

## Added files

- `src/models/cosi_mag_vnext.py`: independent `Model(cfg, data_info)`, modality projectors, end-to-end encoder, output fusion, task interface, and `relation_diagnostics`.
- `src/models/cosi_relation.py`: CRC relation scoring, stable local descriptors, weighted `gcn_norm` operators, and compact summaries.
- `src/models/cosi_filter.py`: sparse normalized-adjacency multiplication and direct Chebyshev recurrence.
- `src/models/cosi_interaction.py`: shared Q/K/V cross-order attention, relation biases, signed order residuals, semantic gates, and modality refinement.
- `configs/model/cosi_mag_vnext.yaml`: exposed architecture, widths, dropout and initialization settings.
- `tests/test_cosi_mag_vnext.py`: 13 synthetic correctness tests.
- `docs/cosi_mag_vnext_design.md`: equations, scope, initialization, and related-work boundaries.
- `docs/cosi_mag_vnext_implementation_report.md` and `docs/cosi_mag_vnext_pilot_report.md`.
- `scripts/collect_cosi_mag_vnext_reports.py`: collects completed run metrics and checkpoint diagnostics into small JSON/CSV artifacts.
- `reports/cosi_mag_vnext/`: environment, correctness, pilot, mechanism and run manifests plus two CSV summaries.

No existing model, config, task runner, dataset loader, protocol document, checkpoint or historical result was edited.

## Model size and implementation map

- Encoder parameters: **1,277,207** (all trainable).
- Movies/Grocery/Reddit-S NC encoder plus linear classifier: **1,282,347** parameters.
- Sports-Copurchase LP encoder plus projection and decoder: **1,278,104** parameters.
- `Model.forward` calls `ProjectionMLP`, CRC, `chebyshev_bank` twice, `RelationConditionedMultiOrderInteraction.forward_modality` for each modality, then the CoSI concat residual late fusion.
- `Model.inference` performs one evaluation-mode full forward. `batch_size` is accepted for interface compatibility; independent node chunks would change global edge weights and normalization.
- The model returns `(z, None, None, zero_scalar, {})`, `z:[N,256]`, `out_dim=256`, and `requires_full_lp_sampler_depth=True`.

## Correctness and regression tests

| Area | Result | Evidence |
| --- | --- | --- |
| CPU forward/backward, shapes, finite values | PASS | Synthetic graph test |
| CRC last-layer zero initialization and unit weights | PASS | Synthetic relation test |
| Reciprocal edge symmetry, positive weight bounds, symmetric P on an undirected graph | PASS | Synthetic relation test |
| Relation-score and `q` gradient paths | PASS | Finite nonzero last-layer gradient; scorer hidden layer receives gradient after an update |
| Sparse Chebyshev recurrence | PASS | Matches a dense matrix reference on a 4-node graph |
| Isolated node and self-loop handling | PASS | Descriptor is zero; order differences are zero |
| Empty graph, one node, duplicate edges, integer input, float64 input | PASS | Synthetic edge-case test |
| Initial `eta`, attention residual gate and semantic gate | PASS | Direct initialization assertions |
| Eval `forward` versus `inference`; state-dict save/load | PASS | Outputs agree within `1e-6` |
| NC task checkpoint save/reload | PASS | Smoke runs exported best-validation checkpoints; the report collector loaded all nine pilot checkpoints and executed a diagnostic forward from each |
| LP sampler depth and reverse positive-edge removal | PASS | Global-eid synthetic test; actual LP smoke removed 72 message edges before model forward |
| Legacy CoSI and LP/protocol regressions | PASS | 57 relevant existing tests |
| Combined final targeted test run | PASS | 70 passed, 0 failed |

Commands:

```text
conda run -n yhf_env python -m pytest -q tests/test_cosi_mag_vnext.py
conda run -n yhf_env python -m pytest -q tests/test_cosi_mag_ablation.py tests/test_cosi_mag_final_stability.py tests/test_lp_sampling.py tests/test_inference_equivalence.py tests/test_protocol.py
conda run -n yhf_env python -m pytest -q tests/test_cosi_mag_vnext.py tests/test_cosi_mag_ablation.py tests/test_cosi_mag_final_stability.py tests/test_lp_sampling.py tests/test_inference_equivalence.py tests/test_protocol.py
```

The first gradient test exposed a real numerical issue: `sqrt(0)` for a constant-neighborhood standard deviation gave an infinite derivative and NaN CRC gradients. The variance now has a small floor before square root, with zero derivative below the floor. The tests then passed. The `eta` initialization was also checked against `eta=.5*tanh(p)` and set so the first coefficient starts at `0.2`.

The full repository test suite was not run; the 57 existing tests above cover the old CoSI Final/ablation, inference equivalence, LP sampling and task protocols.

## NC and LP integration

- NC remains on `unified_full_graph_nc_v1`: one full graph forward per epoch, labels used only for train-node loss, validation accuracy for checkpoint selection. No NC runner change was needed.
- LP remains on `unified_sampled_lp_v1`: train-only message graph and existing `LinkNeighborLoader`. In `src/tasks/lp.py`, neighbor sampling happens first; then the current positive supervision edges and both directions are removed on CPU with `global_eid`; only then is the batch sent to the device and `model.forward` called. CRC sees only that supplied, masked batch graph. VNext has no full-graph relation cache.
- `requires_full_lp_sampler_depth=True` makes the existing sampler resolve to `[5,5,5]` for `max_order=3`.
- Validation uses the existing full-graph train-only message graph and full inference. Sampled-subgraph degree normalization is local and is not claimed to equal full-graph propagation.
- Existing negative sampling, official validation/test negatives, decoder, rank calculation and checkpoint selection code were untouched.

## Smoke and pilot resource use

The smoke runs and full pilot runs are described in the [pilot report](cosi_mag_vnext_pilot_report.md). The completed full NC runs took 17.2–32.3 seconds each. PyTorch recorded peak CUDA allocation of 4,180 MiB for Movies, 3,938 MiB for Grocery, and 5,855 MiB for Reddit-S. The Sports-Copurchase LP smoke completed one epoch with two train batches and full validation inference; it reported training loss `0.6916`, validation MRR `5.63%`, and 72 removed positive message edges. Smoke scores are not performance results.

## Independent code self-review

1. **Math/code agreement:** CRC uses the four specified pairwise evidence terms and exact bounded weights. Local descriptors exclude loops. MMF uses direct `T_k(P)` recurrence. RCMI uses order embeddings only in Q/K, original bank states in shared V, `D_k=interacted_k-interacted_0`, signed eta, a node gate, refinement residual and concat late fusion.
2. **Legacy parameters:** none are inherited or referenced; VNext has no old checkpoint key compatibility requirement.
3. **Unused branches:** both modalities traverse projection, CRC operators, all polynomial orders, relation bias, attention, eta, semantic gate, refinement and fusion. Tests exercise first-layer CRC gradients after the zero-initialized output layer updates.
4. **Autograd:** no `detach()` exists in the normal VNext forward path. Diagnostics are deliberately `no_grad`; LP graph selection/masking is task data handling before the model.
5. **Loops and duplicates:** self-loops are stripped before relation scoring and local statistics, then one identity loop is inserted by `gcn_norm`. Standard loaders preprocess to unique edge indices; direct duplicate COO records remain additive multiedges.
6. **LP leakage:** reviewed the current `global_eid` path in `src/tasks/lp.py`; both orientations and duplicate instances of current positive labels are removed before model forward. Model state contains no full graph or relation cache.
7. **Train/eval:** dropout modules follow the module mode; eval `forward` and `inference` agree. Inference intentionally executes the same complete graph computation.
8. **Validation/test:** all pilot runs set `task.evaluate_test=false`; no test metric was read or used for selection.
9. **Literature claims:** comparisons in the design document are limited to mechanism distinctions. The fixed Chebyshev basis is not described as optimal/adaptive.
10. **Scope:** only new model/config/test/report/collector files were added. Old CoSI Final code, config, task protocols and historical results remain untouched.

## Limitations and unresolved work

- CRC edge weights remain tightly concentrated around one on all three pilot datasets. The scorer is differentiable and trained, but the measured edge-weight spread suggests its reweighting effect is currently modest.
- The raw structural residual can be large relative to `H0`; the learned semantic gates reduce it, but this behavior needs a more careful stability/ablation study before making causal claims.
- Only one small sampled LP smoke was run; no full LP training or LP pilot was run.
- No formal ablations or test-set evaluation were run. The current pilot is a validation-only candidate screen, not a paper result.
- Full inference is exact relative to this model's supplied graph but is global; larger graphs may need a separately verified exact or memory-aware path.
- There is no VNext historical checkpoint compatibility by design.

Machine-readable evidence and output locations are listed in `reports/cosi_mag_vnext/` and `run_manifest.json`.
