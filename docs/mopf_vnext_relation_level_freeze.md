# MoPF-vNext Relation-Level Freeze

## Selected relation module

The selected relation-level module is **Calibrated R1 — Modality-Adaptive Semantic Conductance**:

`s_ij^m = cos(w^m ⊙ h_i^m, w^m ⊙ h_j^m)`

`w^m = softplus(theta^m) / mean(softplus(theta^m))`

`c_ij^m = c_min + (1-c_min) sigmoid(s_ij^m / 0.35)`

The scalar `tau=0.35` is global, fixed, and shared by both modalities and all datasets.

## Rejected alternatives

- R0 ordinary separate cosine: retained as the stable fallback/ablation, but not selected because R1 was performance-safe and its learned metric was non-identity.
- R2 multi-perspective metric: rejected because collapse-to-mean frozen interventions did not establish sufficient functional effect across the five datasets.
- Frozen calibration candidates `tau=1.5` and `tau=1.0`: rejected because mechanism amplification did not reach the 2x gate.
- Frozen calibration candidates `tau=0.75`, `tau=0.5`, `tau=0.35`, and `tau=0.25` passed Phase A; `tau=0.75` and `tau=0.35` were the mild/strong retraining candidates. The final selection is `tau=0.35`.

## Validation evidence

Phase B compared each candidate against T0 (the existing R1 `tau=2.0` runs) using validation-first selection:

- `tau=0.35`: unweighted five-dataset mean Validation Accuracy delta `+0.0137 pp`.
- Largest dataset mean Validation Accuracy drop: Movies `-0.1700 pp`.
- Functionally Amplified: 5/5 datasets.
- Functionally Material: 4/5 datasets.
- Temperature-off frozen audit active: 5/5 datasets.
- Test metrics are descriptive only and were not used for checkpoint, candidate, or final selection.

## Frozen functional evidence

Phase A used all 15 U1-R R1 best checkpoints across the seven-temperature grid. Learned and identity streams were both evaluated at every tau. Raw semantic scores were invariant within `<1e-7`; model state and checkpoint bytes were unchanged by analysis overrides. Edge support, GCN-normalization support, finite/bounded conductance, prediction stability, and saturation gates passed for the selected path.

Functional evidence is reported continuously and in three levels: Numerically Active, Functionally Amplified, and Functionally Material. The authoritative details are in `outputs/u1t_semantic_conductance_calibration/u1t_master_summary.json`.

## Exact provenance

- Exact reproducible implementation commit: `329216e` (`freeze U1-T calibrated semantic conductance`).
- U1-R source: `outputs/u1r_semantic_metric_resolution/u1r_master_summary.json`.
- Frozen reference tag: `mopf-v0-frozen`.

## U2 input definition

U2 receives the fixed relation module above. It must not change the metric form or temperature. No U2 work is started by this stage.
