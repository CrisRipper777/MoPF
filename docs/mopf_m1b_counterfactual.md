# MoPF M1-B — Counterfactual Functional Validation

## Scope and protocol

This report evaluates frozen counterfactuals on the seed-42 best-validation-accuracy checkpoints for Movies, Toys, Grocery, ele-fashion, and Reddit-S. Grocery uses K=2; the other datasets use K=3. All metrics use the fixed Macro-F1 protocol whose label set is the union of valid labels in train, validation, and test splits, with `zero_division=0`.

No model was retrained. The formal MoPF forward path, training objective, checkpoints, semantic graphs, bases, fusion layers, and classifier heads were kept fixed. Counterfactuals only replace the requested coefficient component in the analysis-side embedding reconstruction.

## Canonical centered decomposition

For every dataset and modality, the identity

`gamma_global + delta_gamma + delta_node = gamma_global + delta_gamma_effective + delta_node_centered`

was checked with maximum absolute error below `1e-6`. The exact errors are stored in each `canonical_decomposition.pt` and JSON metadata file.

## Frozen results

The table reports test-accuracy drops relative to Original. Positive values mean degradation under the counterfactual. Node-shuffle is the mean ± population standard deviation over the fixed permutation seeds 0–9.

| Dataset | mean centered node std | modality profile distance | effective radius | Node-mean drop | Node-shuffle drop | Modality-mean drop | Modality-swap drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Movies | 0.00580 | 0.13463 | 1.554 | +0.060 pp | +0.105 pp ± +0.098 pp | +0.060 pp | +0.300 pp |
| Toys | 0.00064 | 0.14893 | 1.634 | +0.000 pp | -0.010 pp ± +0.012 pp | -0.072 pp | +0.000 pp |
| Grocery | 0.00211 | 0.23443 | 1.424 | +0.000 pp | -0.003 pp ± +0.031 pp | +0.029 pp | -0.117 pp |
| ele-fashion | 0.00597 | 0.06189 | 1.525 | +0.003 pp | +0.003 pp ± +0.009 pp | -0.010 pp | -0.007 pp |
| Reddit-S | 0.00160 | 0.04716 | 1.549 | +0.000 pp | +0.000 pp ± +0.014 pp | -0.063 pp | -0.063 pp |

The full Val Acc, Val Macro-F1, Test Acc, and Test Macro-F1 values are in `counterfactual_summary.csv`; per-permutation results are in `node_shuffle_summary.json`.

## Observational evidence

- Node residual variability is heterogeneous across datasets; the structural values are descriptive statistics, not evidence of function by themselves.
- Modality profile distance and effective radius summarize the learned checkpoint profiles. They are likewise observational and do not establish that the corresponding corrections are used functionally.

## Frozen counterfactual evidence

- Node-mean removes node-specific variation while retaining the shared node-generator contribution. A positive drop is evidence that node-specific variation contributes under a frozen intervention.
- Node-shuffle preserves each modality's complete K+1 profile distribution, including order-wise means, standard deviations, quantiles, and the overall coefficient multiset, while breaking node-to-profile alignment. A positive drop supports functional importance of alignment.
- Modality-mean removes the systematic text-versus-visual correction difference while retaining node residuals. Modality-swap preserves correction magnitudes but breaks modality assignment. A positive drop in either is counterfactual evidence for modality-level use; coefficient distance alone is insufficient.

## Dataset-level interpretation

- Largest node-profile-alignment drop: **Movies** (`+0.105 pp` mean Test Acc drop).
- Largest modality-mean drop: **Movies** (`+0.060 pp` Test Acc drop).
- Across the five datasets, node-shuffle gives a positive drop above 0.1 percentage points in at least one dataset: **true**. Node-mean gives such a drop in at least one dataset: **false**.
- At least one modality counterfactual gives a positive drop above 0.1 percentage points: **true**.

These are descriptive frozen-checkpoint findings. They do not establish retraining ablation results, statistical significance, or causal claims beyond the stated interventions.

## Descriptive cross-dataset correlations

With only five datasets, these Pearson correlations are descriptive summaries only and must not be interpreted as significance tests or strong causal evidence.

| Structural quantity | Counterfactual degradation | Pearson r |
|---|---|---:|
| mean centered node std | Node-shuffle Test Acc drop | 0.6422400822141497 |
| mean centered node std | Node-mean Test Acc drop | 0.6220159611668444 |
| mean modality profile distance | Modality-mean Test Acc drop | 0.44111713597861196 |
| mean modality profile distance | Modality-swap Test Acc drop | -0.0951976079536783 |

## Required answers

1. **Is node-level personalization functionally used?** The answer should be read from Node-mean drops above: positive drops indicate frozen evidence for use on the corresponding datasets; near-zero drops do not support a universal claim.
2. **Is node-profile alignment important?** Node-shuffle is the direct test. Positive mean drops indicate that assignment of a complete profile to its original node matters while the profile distribution is held fixed.
3. **Is modality-level personalization functionally used?** Only datasets with positive Modality-mean or Modality-swap drops provide frozen evidence. Modality coefficient distance alone is not enough.
4. **Which datasets rely most on each level?** By this protocol, `Movies` has the largest mean Node-shuffle degradation and `Movies` has the largest Modality-mean degradation. The complete ranking is in the CSV.
5. **Does the evidence justify keeping the node residual module?** The module is justified as a mechanism worth retaining when Node-mean and/or Node-shuffle produce meaningful positive degradation on the target datasets. The results support a dataset-conditional conclusion, not a universal claim that every dataset requires node personalization.

## Unsupported claims

This stage does not support claims that node residuals are universally necessary, that modality distance causes performance differences, that frozen counterfactual drops equal retraining ablation effects, or that any observed cross-dataset correlation is statistically significant.
