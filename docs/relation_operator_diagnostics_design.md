# P0-D — Global Prior & Relation Personalization Diagnostics

## Scope

This frozen-checkpoint analysis follows `relation_operator_audit` and writes only to `results/relation_operator_diagnostics_v1/`. It reads the existing `global_expert` A2 and `relation_expert` A3 checkpoints (5 datasets × 3 seeds per variant). Checkpoints are loaded strictly through the existing analyzer. It does not change model forward code, checkpoint parameters, training computation, or training outputs.

Only Train and Validation indices and labels are used. Validation metrics index `data.y[data.val_idx]`; the fixed Macro-F1 class list is resolved with `include_test=False`. Test labels and metrics are never evaluated. Quantile thresholds use Train–Train physical edges. The source dataset loader materializes its label tensor in host memory as part of its established API, but this analyzer never indexes its Test portion.

## D1 — Global operator necessity

For each frozen A2 checkpoint, `_prepare_frozen` computes the current projector outputs, canonical sparse physical operator, and exact checkpoint global route. `_forward_routes` applies the unchanged expert modules and original three-hop propagation with supplied routes. Regression tests compare the result with the model's ordinary `analyze` and `forward` outputs, and strict-load a state-identical clone.

The analyzer evaluates ZERO, UNIFORM, and checkpoint TOP1 routes in Text-only, Visual-only, and both-modality scopes. It evaluates all four diagnostic single-expert and drop-one routes separately per modality. These are validation diagnostics; best-single selection is an oracle and must not be treated as a deployment policy. Node-level compressed output preserves normal/intervention logits, CE, true-class margins, predictions, and flips.

`ΔCE = CE(intervention) - CE(normal)`, so positive values mean the frozen normal mechanism reduced CE. Margin utility is `margin(normal) - margin(intervention)`, with positive values meaning normal improved the true-class margin. The descriptive D1 decision uses a declared 0.01 nats/node CE tolerance and 0.01 absolute Accuracy/Macro-F1 tolerance to label single-operator sufficiency; these are interpretation thresholds, not statistical tests.

## D2-A — Global shrinkage

For each A3 modality, compute the global prior as the arithmetic mean route over every non-self physical edge. Use

`route(lambda) = global_mean + lambda * (route - global_mean)`.

The λ=1 path returns the original route tensor exactly. The λ=0 path broadcasts the same checkpoint-specific global mean over all edges. Text-only, Visual-only, and both-modality scopes are evaluated at λ ∈ {0, 0.25, 0.5, 0.75, 1}. Per-context and dataset-aggregate summaries include best λ by mean Validation CE and Validation Accuracy. `λ=0` is not compared as if it were a separately trained A2 checkpoint.

## D2-B — Edge-targeted route-to-global

For each modality and A3 checkpoint, assign Validation-related physical edges to A0 semantic-similarity Q1–Q4 and A3 personalization-deviation D1–D4. Both quartile cutoffs are computed from Train–Train edges. The personalization measure is `L1(route - global_mean)`.

A target intervention replaces only selected edges' route by the A3 global mean route. Every other edge route is checked to remain bit-identical. Each target group receives 20 matched random control groups with equal edge counts, sampled without replacement from all non-self physical edges after excluding the selected target edge set. This control pool allows D1/D4 targets to match personalization magnitude; the intervention still scores only on Validation labels, and no Test labels or metrics select edges. Matching bins are train-defined quartiles of log source degree, log target degree, and personalization magnitude. Matching relaxes in order from three dimensions, to two, to one, then unrestricted sampling when a stratum does not contain enough candidates. The fallback rate, candidate count, duplicates, overlap, standardized mean differences, and quartile-bin total variation are recorded for every repeat.

Node effects are measured for all Validation nodes and for directly touched Validation nodes (at least one selected edge incident to the node). `d2_node_utility_concentration.csv` summarizes per-target touched-node coverage, mean absolute ΔCE, harmed/improved/near-zero fractions, and absolute margin utility for both populations. The report compares target mean node ΔCE with the empirical 20-control distribution and reports aggregate matching balance. The report includes target percentile, two-sided empirical extremeness, random mean and spread, and excess CE utility divided by mean absolute and relative message change. Directly-touched control populations with zero Validation nodes are recorded with `n_repeats=0` and `n_empty_touched_controls=20`; their all-Validation-node distributions still contain all 20 controls. These are descriptive diagnostics, not formal significance tests.

## Structural-response descriptive audit

Use A0 H0 and the canonical symmetric-normalized physical operator with self-loop coordinates removed to calculate `G = P_nbr H0` and `D = G - H0`. For each Validation-related physical edge and modality, preserve `||D_i||`, `||D_j||`, `||D_i-D_j||`, `cos(D_i,D_j)`, `cos(H0_i,H0_j)`, A0 semantic similarity, and A3 route deviation. Summaries include Spearman correlations, Kruskal–Wallis group comparisons across D1–D4, and a Mann–Whitney comparison of high- versus lower-deviation edges within semantic Q1. These edge-dependent rank statistics are exploratory and do not establish causal necessity.

## Outputs

- `d1_global_operator_interventions.csv`, `d1_node_utility_summary.csv`, `d1_decision.csv`
- `d1_node_utility.csv.gz` — node logits and utility for the full D1 intervention matrix
- `d2_shrinkage_sweep.csv`, `d2_shrinkage_node_utility.csv.gz`
- `d2_targeted_personalization.csv`, `d2_targeted_node_utility.csv.gz`
- `d2_random_control_summary.csv`, `d2_matching_quality.csv`, `d2_node_utility_concentration.csv`
- `d2_structural_context_descriptives.csv`, `d2_structural_context_edge_rows.csv.gz`
- `d2_decision.csv`, `relation_operator_diagnostics_report.md`

The gzip node and edge tables are listed in `.gitignore` to keep a multi-gigabyte raw artifact out of Git. Summary CSVs and the final report remain tracked. The full analyzer can regenerate all artifacts from frozen checkpoints with:

```bash
/home/m3/miniconda3/envs/yhf_env/bin/python scripts/analyze_relation_operator_diagnostics.py --device cuda:1 --random-repeats 20
```

No P0-R, Stage II, model architecture update, auxiliary loss, or new training run belongs to this phase.

The completed result bundle can refresh node concentration and report summaries from the compressed target-node artifact without loading checkpoints or using a GPU:

```bash
/home/m3/miniconda3/envs/yhf_env/bin/python scripts/analyze_relation_operator_diagnostics.py --refresh-existing-results
```
