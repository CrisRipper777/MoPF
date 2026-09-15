# E0-C — Node–Modality Multi-Hop Utilization Heterogeneity

## Scope and definition

This is an analysis-only, pre-TCPR diagnosis over the frozen U2-C C1
best-validation checkpoints used by U3-A. It covers Movies, Toys, Grocery,
ele-fashion, and Reddit-S with seeds 42, 43, and 44. No new training,
counterfactual, association analysis, or performance analysis was run.

For each frozen predecessor forward, the analysis used the existing
`src.analysis.u3a.contribution_profile` implementation. For each node,
modality, and order, it forms `G = eta * S`, `g = ||G||_2`, and
`p = g / sum(g)`. The primary order statistic is the normalized
contribution-weighted response order

\[
r_i^m = \sum_k (k/K) p_{i,k}^m,
\]

where `K` is the formal dataset order (`3, 3, 2, 3, 3`, respectively).
Raw contribution-weighted order and normalized contribution entropy are also
retained in the artifacts. Text–visual differences use absolute order gap,
full-profile L1 distance, and the Jensen–Shannon implementation already used
by U3-A.

The C1 configuration was checked at runtime for every checkpoint:

`learned_diag_cos`, temperature `0.35`, anchored/cumulative multi-hop state,
anchor alpha `0.1`, formal `K`, and `use_transport_residual=false`.
Thus TCPR was disabled. The existing loader supplied the raw frozen modality
features and graph; the classifier head, labels, predictions, and metrics were
not used by this analysis.

## Core results

The table reports descriptive means across the three seeds for the node-level
order distributions. The bracketed values are the minimum–maximum of the
three seed-specific medians for the modality-gap statistics; seed rows remain
separate in the CSV and were not pooled as independent observations.

| Dataset | K | Nodes | Text order mean | Text IQR / q90–q10 | Visual order mean | Visual IQR / q90–q10 | Median absolute order gap | Median profile L1 | Median profile JS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 3 | 16,672 | 0.537 | 0.052 / 0.090 | 0.538 | 0.049 / 0.084 | 0.009 [0.009, 0.010] | 0.083 [0.079, 0.086] | 0.004 [0.002, 0.006] |
| Toys | 3 | 20,695 | 0.562 | 0.030 / 0.052 | 0.648 | 0.011 / 0.020 | 0.085 [0.082, 0.089] | 0.243 [0.214, 0.275] | 0.023 [0.020, 0.030] |
| Grocery | 2 | 17,074 | 0.651 | 0.074 / 0.123 | 0.861 | 0.042 / 0.074 | 0.208 [0.191, 0.237] | 0.464 [0.440, 0.510] | 0.037 [0.036, 0.040] |
| ele-fashion | 3 | 97,766 | 0.518 | 0.053 / 0.097 | 0.512 | 0.051 / 0.098 | 0.009 [0.007, 0.010] | 0.103 [0.078, 0.130] | 0.002 [0.001, 0.005] |
| Reddit-S | 3 | 15,894 | 0.553 | 0.013 / 0.024 | 0.576 | 0.009 / 0.018 | 0.023 [0.010, 0.032] | 0.113 [0.090, 0.135] | 0.006 [0.003, 0.010] |

The complete q10/q25/q75/q90, standard deviation, entropy, raw order, and
per-seed modality-gap values are in
`multihop_utilization_summary.csv` and `modality_gap_summary.csv`.

## Answers to the empirical questions

1. **How wide are the normalized response-order distributions?**

   Node-level width is dataset- and modality-dependent. Grocery is widest in
   both modalities by IQR (Text 0.074, Visual 0.042 on the three-seed
   descriptive average). Movies and ele-fashion have intermediate width near
   0.05 IQR. Reddit-S is narrowest (Text 0.013, Visual 0.009), while Toys is
   especially narrow for Visual (0.011 IQR) but broader for Text (0.030 IQR).

2. **Is node heterogeneity stable across seeds?**

   The qualitative width pattern is reasonably stable for Movies and
   ele-fashion. Grocery remains broad in all seeds, although its order means
   vary materially across seeds (Text 0.545–0.712; Visual 0.784–0.910).
   Toys has a stable, sizeable text–visual separation but its absolute order
   distributions shift across seeds. Reddit-S remains narrow in every seed,
   while its cross-modal gap increases from seed 42 to seeds 43/44. These are
   descriptive seed checks, not pooled inferential tests.

3. **Do Text and Visual have systematically different utilization
   distributions?**

   The difference is strong for Grocery and clear for Toys: the three-seed
   mean normalized order is 0.651 vs. 0.861 for Grocery and 0.562 vs. 0.648
   for Toys (Text vs. Visual). Reddit-S shows a smaller separation (0.553 vs.
   0.576). Movies (0.537 vs. 0.538) and ele-fashion (0.518 vs. 0.512) are
   close in their marginal order means. These are directionally descriptive
   utilization differences and are not claims that one modality is better.

4. **Do order gap, L1, and JS agree?**

   Yes at the dataset level. All three statistics are nonzero in every
   dataset/seed row. Their broad ranking is consistent: Grocery has the
   largest discrepancy, Toys is next, Reddit-S is intermediate, and Movies
   and ele-fashion are the smallest or near-smallest. The absolute scale of
   the three statistics differs, so they should not be treated as interchangeable
   effect sizes.

5. **Which datasets have the strongest or weakest heterogeneity?**

   Grocery is the strongest combined node- and modality-heterogeneity case:
   it has the widest order distributions and the largest order/profile gaps.
   Toys is also a strong modality-heterogeneity case, despite its relatively
   narrow Visual node distribution. Movies is closest to a shared Text/Visual
   order profile. Ele-fashion also has a small order gap, although its full
   profile L1 is somewhat larger than Movies. Reddit-S has the narrowest
   node-level order distributions and a smaller, seed-sensitive modality gap.

6. **Is there a near-collapse dataset/modality?**

   There is no all-zero contribution profile. In the narrower descriptive
   sense of a nearly shared node-level normalized-order distribution, Reddit-S
   Visual (IQR 0.009 on average) and Toys Visual (IQR 0.011) are the closest
   to concentration around a common order. This does not imply that their
   full contribution vectors are identical; the Toys profile L1 gap remains
   substantial.

7. **Does this support the current motivation?**

   The data support a qualified motivation, but not a universal claim that
   every dataset and modality exhibits a large personalization effect.
   Movies and ele-fashion have small median order gaps, while Toys Visual and
   Reddit-S have relatively narrow node-level order distributions. These
   cases limit the strength of an across-the-board statement.

## Claim level

The strongest defensible choice is **B**:

> Multi-hop utilization is heterogeneous across nodes and modalities, but the
> strength of personalization is dataset- and modality-dependent.

This statement reflects the broad nonzero node-level distributions and
text–visual profile discrepancies, while preserving the clear variation in
magnitude across datasets and seeds. Claim A is too strong because the
heterogeneity is not uniformly large, and claim C is too weak because Grocery
and Toys show substantial and repeatable discrepancies.

## Audit and reproducibility

- Branch: `vnext`.
- Commit: `b080cb8058628eea3b9da708073f72a083d82a6e`.
- Source: the 15 U2-C C1 best-validation checkpoints resolved by
  `scripts/run_mopf_u3a_node_modality_preference_diagnosis.py::_source_records`
  and cross-checked against the authoritative U3-A source manifest.
- Every source checkpoint SHA matched before and after its forward export;
  every model state digest matched before and after forward.
- All 30 modality rows and all 15 modality-gap rows were finite. No
  contribution profile was all-zero. The maximum nonzero-profile probability
  sum error was at most `2.3841858e-07`.
- Raw feature paths, shapes, dtypes, finite status, and zero-norm counts are
  recorded in each `seed*/summary.json`. All reported raw modality features
  were finite and had zero zero-norm nodes.
- The standard project loader was reused. It materializes the normal dataset
  object, but this script does not access labels, split indices, predictions,
  or validation/test metrics.
- No optimizer step, backward pass, training run, counterfactual, association
  analysis, or checkpoint write was performed. No final model configuration
  file was modified.
- CUDA was unavailable inside `yhf_env` (`torch.cuda.is_available()` was
  false), despite the host GPUs being visible to `nvidia-smi`; the successful
  run therefore used CPU. This changes execution device only, not the frozen
  model, definitions, or statistics.
- Runtime was approximately 42 seconds for the 15 analysis-only forward
  exports.

## Files and command

New code and report:

- `scripts/run_mopf_e0c_semantic_retention_heterogeneity.py`
- `docs/mopf_e0c_multihop_utilization_heterogeneity.md`

Successful command:

```bash
MPLCONFIGDIR=/tmp/mopf_e0c_mpl PYTHONPATH=src conda run --no-capture-output -n yhf_env \
  python scripts/run_mopf_e0c_semantic_retention_heterogeneity.py --device cpu
```

Outputs are under
`outputs/e0_empirical_motivation/multihop_utilization/`:

- `multihop_utilization_summary.csv`
- `modality_gap_summary.csv`
- `normalized_response_order_violin.png` / `.pdf`
- `modality_profile_gap.png` / `.pdf`
- `run_manifest.json`
- For each dataset: `summary.json`, three `seed*/summary.json` files,
  three `seed*/node_multihop_utilization.npz` files, and
  `seed_mean_node_multihop_utilization.npz`.

## Limitations

E0-C is a pre-TCPR learned-model diagnosis, not a method-independent dataset
property. It shows that the frozen predecessor actually learns heterogeneous
multi-hop utilization, but it does not establish that such heterogeneity is
model-independent or causal. It also does not establish any relationship to
accuracy, Macro-F1, semantic drift, graph degree, conductance, or downstream
performance; those analyses are outside E0-C.
