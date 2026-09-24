# SSI-MAG-V3.1 P1.7c Sequential Development Attribution

This is a development-control report, not a final paper ablation. It loads validation-selected NC checkpoints and saved heads in eval/no-grad mode; it does not train, tune, run LP, add losses, or use test metrics for selection.

## Integrity

- Formal controls: `30` expected, `30` loaded, `30` finite.
- Device: `cpu`; training invoked by analyzer: `false`; LP invoked: `false`.
- Node ordering verified: `{'Movies': True, 'Toys': True, 'Grocery': True, 'ele-fashion': True, 'Reddit-S': True}`.
- Attention simplex failures: `[]`.

## Sequential performance evidence

Definitions: `Delta_R2 = B - V3`, `Delta_R1 = AB - B`, `Delta_remove_ref = V3.1 - AB`. Validation is the development comparison; test is descriptive only.

- **Delta_R2**: Val Acc `0.1980 pp`, Val Macro-F1 `0.3685 pp`; better/worse/tie counts Acc `8/6/1`, F1 `9/6/0`.
- **Delta_R1**: Val Acc `-0.1235 pp`, Val Macro-F1 `-0.1098 pp`; better/worse/tie counts Acc `5/10/0`, F1 `7/8/0`.
- **Delta_remove_ref**: Val Acc `-0.0311 pp`, Val Macro-F1 `-0.1978 pp`; better/worse/tie counts Acc `5/10/0`, F1 `6/9/0`.

## R2 mechanism evidence

The corrected alpha range ratio is defined as `(q90-q10)/(q75-q25+eps)`. Cross-seed Pearson/Spearman values are reported separately and are not interpreted as performance selection criteria. Raw p scale is reported descriptively; a large p is not automatically a failure.

## R1 mechanism evidence

Old and new relation scores are on different raw scales and are not ranked by absolute r/a magnitude. The comparable quantities are log modulation, normalized operator perturbation, c_i, and frozen relation-off sensitivity. `attenuation_ratio = std(log(w))/(std(score)+eps)` is diagnostic only.

## Reference-residual evidence

B and AB restore the old reference residual under their respective R1/new R2 controls. The covariance contribution is `Cov(term, eta)/Var(eta)`; the three-term sum is reported against the full signed eta and need not equal one when global/modality terms contribute.

## Frozen sensitivity

Relation-off and reference-off use the same saved NC head and unchanged parameters. These are functional sensitivity diagnostics, not retrained causal ablations.

- Relation sensitivity rows: `60`; reference sensitivity rows: `15`.

## Boundaries

No final model decision is made here. No LP, paper ablation, auxiliary loss, hyperparameter search, R1 amplitude repair, historical V3 modification, frozen V3.1 modification, or test-based selection was performed.

See the CSV files in this directory for complete per-seed values.
