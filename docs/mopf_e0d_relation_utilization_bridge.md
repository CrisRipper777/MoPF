# E0-D — Local Relation Condition vs. Multi-Hop Utilization

## Scope and fixed definitions

This is an analysis-only pre-TCPR bridge diagnosis over the same 15 frozen
U2-C C1 best-validation checkpoints used by E0-C: five NC datasets and seeds
42, 43, and 44. No new training, TCPR intervention, transport intervention,
label analysis, validation/test metric, or F2 run was performed.

The Stage-I local relation weights were taken from the frozen predecessor
forward as `components["edges"]["w_t"]` and
`components["edges"]["w_v"]`. Loader edge indices were canonicalized to
unique physical relations `(min(i,j), max(i,j))`; self-loops were excluded.
For each node and modality,

\[
c_i^m = |N(i)|^{-1} \sum_{j \in N(i)} w_{ij}^m.
\]

The aggregation-side outcome was not redefined. E0-C's saved
`contribution_profile_text/visual` and normalized response-order arrays were
loaded and independently reproduced from
`src.analysis.u3a.contribution_profile`:

\[
p_{i,k}^m = \frac{\|\eta_{i,k}^m S_{i,k}^m\|_2}
{\sum_r \|\eta_{i,r}^m S_{i,r}^m\|_2}, \qquad
r_i^m = \sum_k (k/K)p_{i,k}^m.
\]

The primary association is partial Spearman: average-rank `c` and average-
rank `r` were separately regressed on average-rank `log1p(unique degree)`,
then Pearson correlation was computed between the two residual vectors. This
definition was fixed before inspecting results. Raw-versus-centered rank
agreement was audited separately.

The C1 configuration was checked for every checkpoint:
`learned_diag_cos`, temperature `0.35`, anchored/cumulative state,
`alpha=0.1`, formal dataset `K`, and `use_transport_residual=false`.
TCPR was disabled.

## Primary association results

Values are means across the three seed-specific cells; brackets show the
seed-level minimum–maximum. All 30 partial correlations are positive.

| Dataset | Modality | Raw rho | Degree-controlled partial rho | Degree–order rho | Degree–condition rho | Partial sign across seeds |
|---|---|---:|---:|---:|---:|---|
| Movies | Text | 0.216 [0.200, 0.240] | 0.281 [0.268, 0.303] | 0.520 | -0.046 | 3/3 positive |
| Movies | Visual | 0.204 [0.191, 0.219] | 0.249 [0.233, 0.266] | 0.363 | -0.077 | 3/3 positive |
| Toys | Text | 0.274 [0.237, 0.298] | 0.360 [0.322, 0.379] | 0.422 | -0.118 | 3/3 positive |
| Toys | Visual | 0.243 [0.234, 0.249] | 0.324 [0.321, 0.327] | 0.432 | -0.109 | 3/3 positive |
| Grocery | Text | 0.255 [0.243, 0.267] | 0.337 [0.329, 0.343] | 0.433 | -0.109 | 3/3 positive |
| Grocery | Visual | 0.255 [0.237, 0.269] | 0.286 [0.262, 0.304] | 0.361 | -0.031 | 3/3 positive |
| ele-fashion | Text | 0.389 [0.386, 0.395] | 0.444 [0.440, 0.446] | 0.413 | -0.035 | 3/3 positive |
| ele-fashion | Visual | 0.304 [0.298, 0.315] | 0.354 [0.349, 0.361] | 0.379 | -0.061 | 3/3 positive |
| Reddit-S | Text | 0.703 [0.683, 0.717] | 0.868 [0.854, 0.888] | -0.564 | 0.024 | 3/3 positive |
| Reddit-S | Visual | 0.851 [0.845, 0.858] | 0.890 [0.886, 0.896] | -0.324 | -0.028 | 3/3 positive |

Across the 30 primary cells, the median absolute partial rho is `0.342` and
the mean absolute partial rho is `0.439`. The positive-sign fraction is
`1.000`; there are no negative or zero partial-rho cells. The complete
seed-level table, including sample counts, is in
`relation_utilization_association.csv`.

## Answers to the empirical questions

1. **Raw association.** Local relation condition and normalized response
   order have positive raw associations in every dataset/modality/seed cell.
   The weakest raw cells are Movies Visual (minimum 0.191) and Movies Text
   (minimum 0.200); the strongest are Reddit-S Visual (0.845–0.858) and
   Reddit-S Text (0.683–0.717).

2. **After controlling physical degree.** The association remains in every
   cell and is stronger than the raw association in all 30 cells. Partial rho
   ranges from `0.233` (Movies Visual, seed 43) to `0.896` (Reddit-S Visual,
   seed 43). Thus the observed relation is not reduced to zero by the fixed
   degree control.

3. **Dataset, modality, and seed stability.** The sign is completely stable
   across all three seeds for every dataset and modality. Magnitudes vary:
   Reddit-S is strongest, Movies is weakest, and Toys/Grocery/ele-fashion are
   intermediate. Seed variation is small for ele-fashion and Reddit-S Visual,
   and larger but still same-sign for Toys Text, Grocery Visual, and Reddit-S
   Text.

4. **Quartile agreement.** The quartile table and figure show increasing mean
   response order from the low-condition to high-condition quartiles for 27 of
   30 seed-specific dataset/modality series. The only repeated mild exception
   is Movies Text, where Q4 is slightly below Q3 while remaining above Q1.
   The mean Q4−Q1 order differences are positive for every dataset and
   modality: approximately 0.018/0.016 for Movies Text/Visual, 0.015/0.005
   for Toys, 0.030/0.018 for Grocery, 0.035/0.027 for ele-fashion, and
   0.016/0.016 for Reddit-S. This is directionally consistent with the
   positive partial correlations, without imposing a monotonicity gate.

5. **Strongest and weakest cells.** Reddit-S has the strongest association in
   both modalities (mean partial rho 0.868 Text, 0.890 Visual). Movies Visual
   is weakest (0.249), followed by Movies Text (0.281) and Grocery Visual
   (0.286). The strongest association is therefore not simply the dataset
   with the largest E0-C modality gap; E0-D measures a distinct bridge
   relation.

6. **Role of degree.** Degree does not explain most of the association under
   the fixed partial-correlation definition. The mean absolute
   degree–condition rho is about `0.064` (maximum absolute value `0.132`),
   and partial rho is larger than raw rho in every cell. Degree–order
   association is positive for Movies, Toys, Grocery, and ele-fashion but
   negative for Reddit-S; this variation further argues against treating
   degree as a universal explanation.

7. **Sign reversal or null cases.** There is no sign reversal across seeds or
   modalities in the primary partial association and no near-zero primary
   cell. The small Movies Text Q3-to-Q4 downturn is a non-monotonic quartile
   detail, not a sign reversal or null association.

## Claim level

The strongest defensible choice is **A**:

> Local relation condition is consistently associated with learned multi-hop
> utilization across datasets and modalities, and the association remains
> after controlling for physical degree.

This claim is supported by 30/30 positive partial-rho cells, same-sign
behavior across all three seeds, and positive relation-condition quartile
separation in every dataset/modality. The claim should still be read together
with the substantial effect-size variation across datasets and modalities.

## Physical-edge and numerical audit

- Unique physical edge counts were: Movies 80,401; Toys 56,701; Grocery
  71,131; ele-fashion 199,586; Reddit-S 141,540.
- Loader graph records were symmetric directed storage for every dataset;
  missing reverse relation count was zero in all 15 checkpoint exports.
- No self-loop records were included. Physical degree is the unique physical
  neighbor count; Toys had 10 isolated nodes and ele-fashion had 3 isolated
  nodes per seed. All other nodes were connected.
- Reverse-weight symmetry was within the fixed `1e-6` tolerance everywhere.
  The largest observed max absolute forward/reverse difference was
  `1.1920929e-07` for both modalities; mean differences were at machine-level
  numerical noise.
- E0-C contribution profiles and normalized response orders were reproduced
  exactly in this CPU rerun: maximum absolute difference was `0.0` for every
  dataset/seed/modality artifact checked.
- All saved node statistics were finite. Each association sample count equals
  the number of connected nodes for its dataset/seed.
- The same 15 C1 checkpoint paths and SHA values as U3-A/E0-C were used.
  Checkpoint SHA and model-state digest were identical before and after each
  forward.
- Branch: `vnext`; commit:
  `b080cb8058628eea3b9da708073f72a083d82a6e`.
- Device: CPU. CUDA initialization was unavailable inside `yhf_env`, although
  host GPUs were visible to `nvidia-smi`; this changes execution device only.
- No optimizer step, backward pass, training, checkpoint write, label access,
  validation/test metric access, or transport intervention was performed.
  No final model configuration was modified.

## Files and command

New code and report:

- `scripts/run_mopf_e0d_relation_utilization_bridge.py`
- `docs/mopf_e0d_relation_utilization_bridge.md`

Successful command:

```bash
MPLCONFIGDIR=/tmp/mopf_e0d_mpl PYTHONPATH=src \
conda run --no-capture-output -n yhf_env \
python scripts/run_mopf_e0d_relation_utilization_bridge.py --device cpu
```

Outputs are under
`outputs/e0_empirical_motivation/relation_utilization_bridge/`:

- `relation_utilization_association.csv`
- `relation_condition_quartiles.csv`
- `relation_utilization_master_summary.json`
- `relation_condition_response_order_quartiles.png` / `.pdf`
- `partial_correlation_summary.png` / `.pdf`
- Per dataset/seed `node_relation_utilization_bridge.npz` and `summary.json`

## Limitation

This is a pre-TCPR learned-model association, not causal evidence. It
motivates explicit relation-conditioned aggregation, but does not by itself
establish that TCPR improves downstream prediction. Functional causality is
tested separately by TransportOff / TransportShuffle / ModalitySwap; those
interventions were not run as part of E0-D.
