# CoSI-MAG Final Implementation Cleanup

Date: 2026-09-21  
Branch: `exp/iamoc-v2`

## Audit

The frozen encoder inheritance chain is `MoPFIAMOCV2 → MoPFIAMOC → MoPF → nn.Module`. `build_model` dynamically imports `src.models.${cfg.model.name}`, so `model=cosi_mag_final` works without changing the factory.

The NC and LP runners call `model(x, edge_index)` and consume the five-value result `(z, _, _, aux_loss, aux_info)`. Their classifier and link decoder live outside the encoder. The runners do not call `regularization_loss`; the layerwise inference path calls `inference`, which the final model also provides. The existing R1 settings return a zero auxiliary loss and do one encoder forward because both auxiliary weights are zero.

The final implementation keeps the active operations and learned values: modality projections; learned positive diagonal metrics; global and modality order coefficients; low-rank node preference; order embeddings; single-head Q/K/V attention; hop gates; rank-1 relation-to-order profiles and scales; modality refinement; and late fusion. In R1, `theta_transport_text`, `theta_transport_visual`, `theta_relation_bias_text`, and `theta_relation_bias_visual` do not affect the output. The former are discarded by the V2 attention override, and the latter are bypassed by that same override. `pdc_theta_text` and `pdc_theta_visual` are inert buffers for the absolute node conditioner. The zeroed PPC/HRC switches create no state-dict parameters and add no loss path.

The final class is standalone. It keeps the effective parameter names so valid tensors copy directly, and contains only anchored cumulative propagation, one interaction layer, attention-level relation conditioning, signed adaptive coefficients, and the original late fusion. The final Stage III preference uses interacted states; composition continues to use the original Stage II state bank. Normal forward computes only the incident mean relation context; incident standard deviation is calculated only by the analysis API.

## State-dict audit and conversion

| Model | Trainable parameters | State-dict keys |
|---|---:|---:|
| Frozen R1 source | 1,394,530 | 81 |
| CoSI-MAG final | 1,394,520 | 75 |

The 10 removed trainable scalar values are in four parameters. Two additional removed keys are buffers:

- `theta_transport_text` — 4 values
- `theta_transport_visual` — 4 values
- `theta_relation_bias_text` — 1 value
- `theta_relation_bias_visual` — 1 value
- `pdc_theta_text` — buffer, 4 values
- `pdc_theta_visual` — buffer, 4 values

All 75 retained keys map by identical name and retain their original tensor values. Conversion checks that the only dropped keys are the six listed above, then strict-loads the final model. For Movies and Grocery seed 42, the audit found zero missing final keys and zero unexpected retained keys. The converted checkpoints are independent files under `outputs/cosi_mag_final/checkpoints/`; source checkpoints were not overwritten.

Fresh construction with the same seed also produced identical values for all 75 shared state-dict keys, including the anchored global-prior initialization (`[0.05555556, 0.05246913, 0.89197534, 0]`).

The final state dict has no transport, old relation-gate, PDC, profile-branch, PPC, or HRC keys. All retained parameters participate in the final forward path.

Retained namespaces include `text_proj.*`, `visual_proj.*`, `metric_theta_text/visual`, `gamma_global`, `delta_gamma_text/visual`, `node_proj_text/visual.*`, `node_vector_text/visual`, `hop_layers_text/visual.0.{norm,query,key,value}.*`, `hop_order_embedding_text/visual`, `theta_hop_gate_text/visual`, `relation_beta_raw_text/visual`, `theta_relation_scale_text/visual`, modality refinement, and fusion. The legacy unused set is four trainable parameters (10 scalar values) plus two buffers (8 values).

## Equivalence verification

The verifier compared the same frozen parameters and input features/edges for Movies and Grocery seed 42. It reports the shared-input Stage III replay separately from full independent recomputation. Sparse support indices match exactly.

| Dataset | Device | Shared Stage III max absolute error | Recomputed sparse-stage max absolute error | Full forward max absolute error |
|---|---|---:|---:|---:|
| Movies | CPU deterministic | 0 | 0 | 0 |
| Grocery | CPU deterministic | 0 | 0 | 0 |
| Movies | CUDA 0 | 0 | 1.21e-5 | 1.21e-5 |
| Grocery | CUDA 0 | 0 | 1.31e-5 | 1.31e-5 |

CPU acceptance was `max_abs <= 1e-7` and `relative_l2 <= 1e-6`. CUDA acceptance was `max_abs <= 5e-5` and `relative_l2 <= 5e-5`, consistent with the sparse-reduction variation observed in prior diagnostics. The largest observed CUDA error is below that fixed tolerance. The reports include per-tensor max absolute, mean absolute, and relative L2 error for H0, calibrated edge weights, normalized operators, every multi-hop state, local relation contexts, normalized beta profiles, attention, interacted states, node preferences, eta, modality outputs, fused output, and classifier logits.

- CPU report: `outputs/cosi_mag_final/verification/equivalence_cpu.json`
- CUDA report: `outputs/cosi_mag_final/verification/equivalence_cuda.json`
- Combined acceptance report: `outputs/cosi_mag_final/verification/equivalence.json`

## Task and analysis smoke checks

Both task runners completed one epoch on CUDA 0 and saved/reloaded a best checkpoint:

| Task | Smoke run | Setting | Validation metric |
|---|---|---|---:|
| NC | Movies, seed 42 | 1 full-graph epoch | Accuracy 0.3293 |
| LP | sports-copurchase, seed 42 | 1 epoch, 1 training batch | MRR 0.0666 |

The LP run used the existing sampled `LinkNeighborLoader`, pair decoder, and edge masking path. Test evaluation was disabled for both smoke runs. These metrics only confirm runner execution; they are not benchmark results.

Analysis smoke checks on each saved task checkpoint exercised `analysis_relation_calibration`, `analysis_multi_order`, all 12 combinations of relation (`normal/off/shuffle`) and interaction (`normal/query_collapse/uniform/off`), and `inference`. Outputs were finite, the module training modes were restored, and no parameter or buffer changed. Strict state-dict round-trip matched every tensor exactly. The GPU output comparison differed by at most `9.54e-7`, within the `5e-5` sparse-reduction tolerance.

- NC log and metrics: `outputs/cosi_mag_final/smoke/nc/seed_42/run/`
- NC analysis JSON: `outputs/cosi_mag_final/smoke/nc/seed_42/analysis_smoke.json`
- LP log and metrics: `outputs/cosi_mag_final/smoke/lp/seed_42/run/`
- LP analysis JSON: `outputs/cosi_mag_final/smoke/lp/seed_42/analysis_smoke.json`

## Git audit

The protected model/config, factory, task, dataset, split, decoder, and training files have no diff. `git status --short` still shows 52 documentation deletions that were present before this cleanup, plus the three pre-existing untracked candidate files; those were left untouched. The new untracked source files are `src/models/cosi_mag_final.py`, `configs/model/cosi_mag_final.yaml`, `scripts/convert_r1_to_cosi_mag_final.py`, `scripts/verify_cosi_mag_final_equivalence.py`, `scripts/smoke_cosi_mag_final.py`, and this report. `git diff --stat` therefore reports only the pre-existing 52 documentation deletions: 7,387 deleted lines. Smoke and converted checkpoints are under the ignored `outputs/cosi_mag_final/` tree.

## Acceptance

A. Mathematical equivalence: **PASS**  
B. Final state dict free of legacy branches: **PASS**  
C. NC task compatibility: **PASS**  
D. LP task compatibility: **PASS**  
E. Analysis API compatibility: **PASS**

No full training runs or paper ablations were started. Cleanup stops here for review.
