# MoPF-vNext U1-T — Semantic Conductance Calibration

NC-only controlled temperature calibration. The R1 metric form is frozen; only the global fixed conductance temperature was varied.

- Experiment commit: `26fe1c79e2d0260b43cb09d9c20ebe56f21f40c3`
- Public reproducible commit: `26fe1c79e2d0260b43cb09d9c20ebe56f21f40c3`
- U1-R source: `/hdd1/DataInHere/YHF/MoPF/outputs/u1r_semantic_metric_resolution/u1r_master_summary.json`
- Phase A passing temperatures: `[0.75, 0.5, 0.35, 0.25]`
- Final decision: **Select Calibrated R1 with tau = 0.35**

## Phase A diagnosis

Every R1 checkpoint was evaluated with learned and identity metric streams at each temperature. Raw semantic scores were required to remain invariant to temperature, and all state/checkpoint hashes were required to remain unchanged.

Candidate ranking uses validation drift and continuous mechanism effects only; Test metrics are descriptive and are not used for selection.

## Functional activity levels

- Numerically Active: effect above numerical-noise screening.
- Functionally Amplified: at least 2x the tau=2 metric effect.
- Functionally Material: amplified and max fused/logit effect at least 1e-4.

The authoritative detailed evidence is in `u1t_master_summary.json`.

## Phase B

Phase B was run only for temperatures that passed the Phase-A gate. The retrained frozen identity and temperature-off audits test whether sharpening remains in the inference path after co-adaptation.

No U2, LP, propagation-bank, personalized-filter, fusion, or auxiliary-loss changes were started.
