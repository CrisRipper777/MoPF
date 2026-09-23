# M1 Contextualization-Demand Report

## Experimental definition

M1 evaluates whether node-local modality semantics and physical-neighborhood context have heterogeneous utility in the official NC splits of Movies, Grocery, and ele-fashion. For each modality, frozen raw features are row-wise L2-normalized, a simple mean over actual non-self-loop physical neighbors is formed, and five normalized mixtures are evaluated: `C(lambda) = normalize((1-lambda)H + lambda N)` for lambda in `{0.00, 0.25, 0.50, 0.75, 1.00}`.

The only learned component is `Linear(feature_dim, num_classes)`. Every modality/lambda/seed uses the same AdamW hyperparameters, CE objective, deterministic 90/10 stratified split inside official train, and probe-dev CE early stopping. Official validation is evaluated once after probe checkpoint selection. Official test labels are not indexed, used for training, used for model selection, or used in any M1 statistic.

Physical degree zero nodes are retained in the node-level loss export but are explicitly excluded from preferred-lambda primary statistics. The node-wise oracle is descriptive only: it is not a realizable model and is not test performance.

## Core statistics

| Dataset | Modality | Global-best lambda | Preference entropy | Non-global ratio | T/V disagreement | Mean abs T/V lambda difference | Oracle relative CE gap | Eligible val nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | text | 0.00 | 0.7155 | 0.5210 | 0.6509 | 0.4432 | 0.0121 | 3334 |
| Movies | visual | 0.50 | 0.9081 | 0.8755 | 0.6509 | 0.4432 | 0.0390 | 3334 |
| Grocery | text | 0.00 | 0.6625 | 0.4996 | 0.6120 | 0.4322 | 0.0088 | 3415 |
| Grocery | visual | 0.50 | 0.8784 | 0.8952 | 0.6120 | 0.4322 | 0.0534 | 3415 |
| ele-fashion | text | 0.25 | 0.9455 | 0.8736 | 0.6280 | 0.3580 | 0.0530 | 9776 |
| ele-fashion | visual | 0.25 | 0.8956 | 0.8892 | 0.6280 | 0.3580 | 0.0494 | 9776 |

Global-best lambda minimizes mean official-validation CE across seeds for the indicated modality. Preference entropy and non-global ratio are computed from aggregated node-wise CE across seeds. The T/V disagreement is computed on simultaneously valid, non-isolated validation nodes.

## Dataset decisions and Gate A

The predeclared engineering support rule is: a dataset is supported when at least one modality has non-global preference ratio ≥ 0.25, normalized preference entropy ≥ 0.35, and node-wise oracle relative CE gap ≥ 0.01. Gate A is PASS when at least two of three datasets are supported and no dataset has both modalities collapsing to the same lambda with >90% of eligible nodes.

**Gate A: PASS**

Supported datasets: Movies, Grocery, ele-fashion. Collapse violation datasets: none.

The secondary descriptive weak-evidence count is 3 dataset(s); this count does not relax the PASS thresholds. The implementation decision is therefore **enter P1**.

If Gate A is not sufficient to justify P1, the recommended fallback is relation-calibrated multi-order state-bank modeling, without an adaptive gate search in this round.

## Stability and probe fallbacks

Seed-wise distributions, entropy, and modality disagreement are available in `stability.csv`; node-level losses are in `node_lambda_losses.csv`; global probe metrics are in `global_probe_metrics.csv`; and the degree-stratified descriptive analysis is in `preferred_lambda_vs_degree.csv`.

Deterministic probe split fallback records are stored in `outputs/final/context_demand/probe_split_fallbacks.json`. Fallbacks only occur when a class has fewer than two official training examples and therefore cannot contribute one example to both probe-train and probe-dev.

## Interpretation boundary

These results test contextualization demand using frozen modality features and a linear probe. They do not establish that an adaptive neural gate improves NC/LP performance, and they do not justify interpreting semantic similarity as relation reliability. Any node-wise oracle gap is a descriptive motivation upper bound, not a realizable model or test result.
