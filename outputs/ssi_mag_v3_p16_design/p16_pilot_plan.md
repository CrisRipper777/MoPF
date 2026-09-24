# P1.7 SSI-MAG-V3.1 Pilot Plan

Design only. This plan is not executed in P1.6.

## Objective

Implement and test the single candidate in `p16_v3_1_spec.md` while preserving the existing NC protocol. The first formal candidate is:

`R1-B Semantic-Grounded Relative Relation Modulation + R2-B Stabilized Semantic Prior + removed Stage-II reference residual`.

P1 Full checkpoints and results are reused descriptively; no V3 checkpoint is converted into a V3.1 checkpoint.

## Required pilot order

1. Implement V3.1 in an independent model/config path. Do not overwrite `ssi_mag_v3.py`, its config, historical CoSI-MAG Final, or V3 outputs.
2. Run V3.1 Full on Movies, Toys, Grocery, ele-fashion, and Reddit-S with seeds 42/43/44.
3. If both revisions need attribution, run development controls with the same protocol:
   - V3 reference;
   - V3 + R1-B only;
   - V3 + R2-B only, including removal of explicit `rho_c c`;
   - V3.1 Full with both revisions and removed reference residual.
4. These controls are development pilots, not paper ablations. No LP jobs are included.

## Frozen protocol

- `task=nc`, unified full-graph NC;
- train split only for optimization;
- validation Accuracy for checkpoint selection/early stopping;
- test Accuracy/Macro-F1 recorded only once at the selected checkpoint;
- seeds `42,43,44`;
- no test-based hyperparameter/model selection;
- no auxiliary loss in the first pilot;
- full graph and existing dataset splits unchanged.

## Metrics

### Performance guardrails

- validation Accuracy and Macro-F1 per dataset/seed;
- descriptive test Accuracy and Macro-F1 after validation selection;
- best epoch, trained epochs, train/validation loss if available;
- NaN/Inf, wall-clock, and peak GPU memory.

### R1 health

- beta and relation residual/weight distributions;
- semantic compatibility and relative compatibility distributions;
- normalized operator MAE/RMSE/relative-L1 against raw unit physical topology with edge-pair alignment;
- local `c_i` mean/std/quantiles and isolated-node zeros;
- relation-off frozen sensitivity only as functional evidence.

### R2 health

- alpha mean/std/IQR/q90-q10/corrected range ratio and saturation;
- d and normalized `p`/`term_p` distributions;
- cross-seed term_p and alpha correlations after node-order verification;
- learned bias/rho values and branch scale.

### Stage-II health

- D norm and context-change injection magnitude/direction;
- attention shape, nodewise entropy, Jensen gap, node heterogeneity, query diversity, diagonal excess;
- interaction injection and S_tilde-to-S cosine;
- gamma/DeltaGamma/content/relation eta branch distributions;
- signed eta fraction and effective order/radius.

## Guardrails and stopping rule

Pause further freezing if V3.1 relative to the existing V3 reference shows either:

- a systematic decrease greater than `0.5` percentage points in five-dataset mean validation Accuracy or Macro-F1; or
- a majority of datasets with both validation metrics clearly lower.

Small performance movement is acceptable if training stability is preserved and the mechanism behaves according to its design responsibility. A guardrail is a stop/review criterion, not a tuning loop.

## Resource constraint

Use `cuda:0` where available and record peak memory. The implementation must fit an RTX3090 24GB on all five NC datasets. Relation scoring must remain chunked; Stage II must remain one layer/one head; no PLM/VLM fine-tuning or graph construction is allowed.

## Deliverables for P1.7

- independent V3.1 model/config and unit tests;
- manifest identifying dataset/seed/control and checkpoint provenance;
- validation/test performance summary with population standard deviation;
- R1/R2/Stage-II mechanism CSVs and post-hoc report;
- explicit separation of performance evidence, mechanism behavior, and frozen functional sensitivity;
- no claim of causal necessity from frozen interventions.

## Boundary

P1.7 is a future pilot. P1.6 performs none of these runs, introduces no model file, and makes no test-based architecture selection.
