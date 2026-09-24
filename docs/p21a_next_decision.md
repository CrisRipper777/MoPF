# P2.1a Final Response Evidence Audit

Final decision: **FREEZE_S**.

## Performance gate

S minus P2 `no_cross_hop_interaction` five-dataset paired validation mean: Accuracy 0.0042 pp; Macro-F1 -0.2688 pp.
The dataset-level simultaneous-decline count is 2/5; the count trigger requires at least 3/5.
Test metrics were retained as descriptive rows only and were not used for this decision.

## Formation-conditioned response

Formation covariance is reference plus relation only. The audited collapse-candidate fraction is 0.0000; majority collapse = False.
Existing retrained controls are present: no_formation_conditioning and global_filter_only, with baseline support = True.

## Mechanism health

R1 reports semantic discrepancy nondegeneracy, finite relation scores, finite operator perturbation, and c-node variation without treating larger perturbation as better.
R2 reports alpha finiteness/saturation/adaptive range and d-to-alpha Pearson/Spearman correlations; correlation magnitude has no hard threshold.
R3 covariance identity assertions were applied whenever Var(eta) exceeded epsilon.

## Boundary

No training jobs, new architectures, LP, SHAR/FSCC, tuning, or test-based selection were used.

Final architecture candidate = S.
Next: generic diffusion baselines, robustness, final paper ablation, and LP. No new response mechanism is implemented.
