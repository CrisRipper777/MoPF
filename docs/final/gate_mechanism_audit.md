# MGSC-MAG Gate Mechanism Audit

This is an inference-only qualification audit of the existing P2 candidate. No new trainable mechanism, task protocol, split, checkpoint criterion, or early-fusion path was added.

## A1. Node/order gate distributions

The `std_across_nodes` column is the standard deviation across nodes for a fixed modality/order. It is distinct from the earlier node-wise standard deviation across propagation orders.

- **Movies**: maximum across-node gate std `0.126057`; maximum saturation `0.011036`.
- **Grocery**: maximum across-node gate std `0.164681`; maximum saturation `0.004744`.
- **ele-fashion**: maximum across-node gate std `0.197308`; maximum saturation `0.010310`.
- **Toys**: maximum across-node gate std `0.134348`; maximum saturation `0.088765`.
- **Reddit-S**: maximum across-node gate std `0.133047`; maximum saturation `0.106329`.

## A2. M1 preferred lambda versus learned gate

Preferred lambda is the aggregated model-independent descriptive probe from M1, not a ground-truth label. Gate alignment is therefore descriptive and does not establish causal supervision.

- **Movies/text**: Spearman rho `0.0225`, low-context mean `0.3405`, high-context mean `0.3441`, high-minus-low `0.0036`, Cohen's d `0.0292`.
- **Movies/visual**: Spearman rho `0.0087`, low-context mean `0.8135`, high-context mean `0.8150`, high-minus-low `0.0015`, Cohen's d `0.0220`.
- **Grocery/text**: Spearman rho `0.0600`, low-context mean `0.6748`, high-context mean `0.6984`, high-minus-low `0.0236`, Cohen's d `0.1567`.
- **Grocery/visual**: Spearman rho `0.1892`, low-context mean `0.8721`, high-context mean `0.8805`, high-minus-low `0.0084`, Cohen's d `0.3504`.
- **ele-fashion/text**: Spearman rho `0.1625`, low-context mean `0.6269`, high-context mean `0.6830`, high-minus-low `0.0561`, Cohen's d `0.3012`.
- **ele-fashion/visual**: Spearman rho `0.0177`, low-context mean `0.6988`, high-context mean `0.7003`, high-minus-low `0.0014`, Cohen's d `0.0100`.

## A3. Functional gate interventions

Globalized replaces each order's gate by its normal node mean. Shuffled uses an independent fixed node permutation for each modality/order and preserves each gate marginal distribution. Fixed-0.9 replaces every gate value by 0.9. Metrics are measured relative to normal inference from the same checkpoint.

- **Movies**: largest embedding MAE `0.123370` under `fixed_0.9`; corresponding logit MAE `0.203747` and flip rate `0.076356`.
- **Grocery**: largest embedding MAE `0.082983` under `fixed_0.9`; corresponding logit MAE `0.154959` and flip rate `0.024950`.
- **ele-fashion**: largest embedding MAE `0.098943` under `fixed_0.9`; corresponding logit MAE `0.236866` and flip rate `0.026829`.
- **Toys**: largest embedding MAE `0.075951` under `fixed_0.9`; corresponding logit MAE `0.128460` and flip rate `0.020488`.
- **Reddit-S**: largest embedding MAE `0.060835` under `fixed_0.9`; corresponding logit MAE `0.127090` and flip rate `0.006669`.

Interpretation: a non-zero shuffled intervention with preserved gate marginals is evidence that node-to-gate correspondence is functionally used. This remains an intervention diagnostic, not a proof that the learned gate recovers a unique causal preference.

