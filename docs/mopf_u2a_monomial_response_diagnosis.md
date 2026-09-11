# MoPF-vNext U2-A — Monomial Structural Response Bank Diagnosis

This is an inference-only diagnosis of the frozen U1-T Calibrated R1 relation module. No model was trained or fine-tuned, no checkpoint was written, and no propagation/filter/fusion/classifier code was changed.

- Analysis commit: `50db75ffb755ad512d39beea972155c5f0f47606`
- Source U1-T summary: `/hdd1/DataInHere/YHF/MoPF/outputs/u1t_semantic_conductance_calibration/u1t_master_summary.json`
- Source checkpoints: `15` final T2 checkpoints, datasets `Movies, Toys, Grocery, ele-fashion, Reddit-S`, seeds `(42, 43, 44)`
- Frozen relation: `edge_weight_mode=learned_diag_cos`, `tau=0.35`
- CKA subset: deterministic seed `20260911`, maximum `20000` nodes, identical across all orders

## Scientific scope

For each modality, the analysis extracts the exact raw response bank `R_0=H_0`, `R_k=P^k H_0` from the frozen projection, semantic conductance, and GCN-normalized operator. Frobenius cosine, feature-space linear CKA, normalized response Gram conditioning, incremental structural response novelty, norm/variance evolution, operator-native structural variation, sparse symmetry, and the conditional Dirichlet-energy diagnostic are reported.

Monomial `{1, x, ..., x^K}` and any complete degree-K polynomial basis `{phi_0(x), ..., phi_K(x)}` span the same polynomial subspace under non-degenerate conditions. Therefore a future orthogonal-polynomial study concerns response-coordinate quality, redundancy, conditioning, structural-response separation, and optimization suitability—not larger degree-K expressivity or a larger receptive field.

## Dataset-level screening

- Strong evidence datasets by modality: `{'text': 5, 'visual': 5}`
- Text+visual aggregate Moderate-or-Strong datasets: `5/5`
- Strong/Moderate/Weak labels are screening classifications from the prescribed thresholds, not theoretical claims.

## Operator prerequisite

- Classification counts: `{'near_symmetric': 30, 'approximately_symmetric': 0, 'materially_asymmetric': 0}`
- Spectral statuses: `{'estimated_eigsh_symmetric_part': 24, 'approximate_rayleigh_power': 6}`
- Interpretation: supports_clean_symmetric spectral interpretation

## U2-B gate

**A. Proceed to U2-B: monomial response redundancy/conditioning is empirically supported**

Rationale: The gate uses continuous response diagnostics and the prescribed screening thresholds; it does not treat any one metric as a theorem. Monomial and complete degree-K polynomial bases span the same non-degenerate polynomial subspace, so any future U2-B study concerns coordinate quality, conditioning, separation, and optimisation suitability rather than larger span.

## Pending attribution ablation

The final-paper pending 2x2 relation attribution remains registered and was not run in U2-A: A `separate_cos,tau=2`; B `learned_diag_cos,tau=2`; C `separate_cos,tau=0.35`; D `learned_diag_cos,tau=0.35`. C remains non-blocking for U2 but must not be forgotten.

## Artifacts

- Authoritative JSON: `/hdd1/DataInHere/YHF/MoPF/outputs/u2a_monomial_response_diagnosis/u2a_master_summary.json`
- Flat CSV: `/hdd1/DataInHere/YHF/MoPF/outputs/u2a_monomial_response_diagnosis/u2a_master_table.csv`
- Per-stage records: `/hdd1/DataInHere/YHF/MoPF/outputs/u2a_monomial_response_diagnosis/per_checkpoint`

U2-A hard stop: no Jacobi, Chebyshev, U2-B implementation, U3/U4 work, auxiliary loss, LP, retraining, or formal propagation-bank changes were performed.
