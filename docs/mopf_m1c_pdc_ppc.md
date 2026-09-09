# MoPF M1-C — Propagation Profile Calibration Study

## Scope and protocol

This candidate study evaluates PDC (Propagation-Discrepancy Calibration), PPC (Propagation Profile Consistency), and their combinations. It runs six variants × five NC datasets × three training seeds (42, 43, 44), with Movies/Toys/ele-fashion/Reddit-S at K=3 and Grocery at validation-selected K=2.

The existing full-graph NC protocol is retained: AdamW, learning rate `1e-3`, weight decay `1e-4`, 300 maximum epochs, patience 30, and best-checkpoint selection by Validation Accuracy. Macro-F1 uses the fixed repaired label protocol. HRC remains disabled at `hrc_weight=0` throughout.

The formal default file `configs/model/mopf.yaml` was not changed. Candidate settings are passed only as experiment overrides; V0 is absolute conditioning with PPC disabled.

## PPC scale calibration

The global no-update seed-42 calibration is stored in `outputs/m1c_pdc_ppc/ppc_scale_calibration.json`:

- `R = 3.16704e+08`
- `lambda_low = 1.58352e+06`
- `lambda_high = 6.33407e+06`

The probe values are retained in the JSON. Because the current model initializes `node_vector_text/visual` to zero, the initial PPC raw loss was below epsilon for every dataset: **calibration pathology detected = true**. The prescribed epsilon formula was used exactly; the scale was not silently replaced with a per-dataset or hand-tuned weight.

## Downstream and frozen summary

The table gives the unweighted mean Test/Validation Accuracy across the five dataset means. The node-shuffle column is the mean Test Acc drop relative to Original, aggregated first over 10 permutations and then over the three training seeds.

| Variant | mean Val Acc | mean Test Acc | mean Node-shuffle drop | Detected pathologies |
|---|---:|---:|---:|---|
| v0_current | 81.221% | 80.814% | +0.009 pp | none recorded |
| v1_pdc | 81.244% | 80.770% | +0.038 pp | none recorded |
| v2_ppc_low | 81.313% | 80.819% | +0.000 pp | centered node profile collapse on Movies, Toys, Grocery, ele-fashion, Reddit-S |
| v3_ppc_high | 81.114% | 80.574% | +0.000 pp | centered node profile collapse on Movies, Toys, Grocery, ele-fashion, Reddit-S |
| v4_pdc_ppc_low | 81.261% | 80.836% | +0.000 pp | centered node profile collapse on Movies, Toys, Grocery, ele-fashion, Reddit-S |
| v5_pdc_ppc_high | 81.094% | 80.703% | +0.000 pp | centered node profile collapse on Movies, Toys, Grocery, ele-fashion, Reddit-S |

All four downstream metrics, per-seed values, population standard deviations, mechanism diagnostics, and frozen Macro-F1 drops are in the master JSON. The CSV is a compact browsing table only.

## Downstream evidence

Downstream metrics are reported descriptively. Differences around 0.1–0.3 percentage points in a single run are not interpreted as module gains. Candidate selection is not based on Test metrics.

## Mechanism evidence

PDC exports learned `rho_text[k]` and `rho_visual[k]`, discrepancy magnitudes, centered profile magnitudes, discrepancy/profile Spearman statistics, shared-offset ratios, mean absolute node residuals, and centered node standard deviations. Nonzero rho is evidence that the candidate was used by optimization, not proof that it improved the mechanism.

## Stability evidence

PPC exports 20-view pairwise centered-profile MSE and per-node profile cosine for both modalities. Cross-seed stability uses centered profiles from the three best checkpoints and reports flattened cosine similarity plus normalized RMSE for 42–43, 42–44, and 43–44. Profile RMS below `1e-6` is marked N/A rather than interpreted as numerical stability.

## Frozen functional evidence

For every dataset/variant/seed, Node-mean and independent text/visual Node-shuffle counterfactuals are evaluated with fixed permutation seeds 0–9. The full Val/Test Acc and Macro-F1 drops, including within-seed shuffle mean/std and across-training-seed aggregates, are included in the master JSON.

## Candidate decisions

- **PDC: Keep for further study**. This is based on rho movement, discrepancy/profile diagnostics, profile pathologies, downstream validation band, and node-shuffle evidence jointly.
- **PPC: Reject**. This is based on stochastic MSE/cosine movement, cross-seed stability, centered-profile preservation, and downstream validation band jointly.
- **Combined: Reject**. This requires evidence that the two candidate mechanisms are compatible; it is not a request to maximize Test Accuracy.

These are candidate-study decisions, not a final MoPF declaration. No formal default has been changed and no new loss is enabled by default.

## Unsupported claims

This study does not support universal claims that PDC or PPC is necessary for every dataset, causal claims from descriptive Spearman/cross-seed correlations, claims that a lower stochastic MSE alone proves better propagation, or claims that frozen counterfactual drops equal retraining ablation effects. The master JSON is the authoritative external-analysis artifact.
