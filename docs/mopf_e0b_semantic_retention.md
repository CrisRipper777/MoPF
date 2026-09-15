# MoPF E0-B — Semantic Retention under Ordinary Physical-Graph Propagation

## Scope and scientific question

E0-B is a pure analysis-only, method-independent study of whether repeated
propagation on the shared physical graph changes the modality-intrinsic
semantics of frozen text and visual node features.

The five formal NC datasets were analyzed: Movies, Toys, Grocery, ele-fashion,
and Reddit-S. Formal `K` is 3, 3, 2, 3, and 3 respectively. Every dataset was
also evaluated through `K_analysis=6`.

The existing E0-A data path was reused through
`src.data.load_mag_data`, with the same raw `data.x_t` and `data.x_i` feature
sources and the same loader physical graph support. The U2-A validated
`_linear_cka` and `deterministic_node_subset` definitions were reused. No
model, checkpoint, learned projection, semantic relation weight, label, or
training path was used.

## Fixed propagation definition

Let `A` be the shared physical graph returned by the NC loader before analysis
self-loops. The configured graph is already undirected and has no self-loops.
The analysis used the existing symmetric oriented representation of `A`
directly; it did not call `to_undirected` or add reverse edges a second time.
Every physical edge received weight 1. Exactly one self-loop per node was then
added for the operator only:

```text
A_tilde = A + I
D_tilde = degree(A_tilde)
P = D_tilde^(-1/2) A_tilde D_tilde^(-1/2)
Q_0^m = X^m
Q_k^m = P Q_(k-1)^m,  k = 1,...,6
```

The operator was implemented as a sparse COO matrix using this formula
directly, not through a learned or alternative normalization. Physical degree
and connected-node membership were computed from `A` before the analysis
self-loops.

## Population, features, and sampling

| Dataset | Nodes | Connected | Isolated | Connected fraction | Physical edges | CKA sample |
|---|---:|---:|---:|---:|---:|---:|
| Movies | 16,672 | 16,672 | 0 | 1.000000 | 80,401 | 16,672 |
| Toys | 20,695 | 20,685 | 10 | 0.999517 | 56,701 | 20,000 |
| Grocery | 17,074 | 17,074 | 0 | 1.000000 | 71,131 | 17,074 |
| ele-fashion | 97,766 | 97,763 | 3 | 0.999969 | 199,586 | 20,000 |
| Reddit-S | 15,894 | 15,894 | 0 | 1.000000 | 141,540 | 15,894 |

CKA was computed only on deterministic connected-node subsets, with seed
`20260914` and maximum size 20,000. The actual node IDs are saved in each
dataset's `cka_node_ids.npy`. All-node cosine means are retained as audit
statistics, while the primary cosine summaries use connected nodes.

The raw input feature sources and shapes are identical to E0-A:

| Dataset | Text / visual shapes | Feature source |
|---|---|---|
| Movies | `(16672, 768)` / `(16672, 768)` | `Movies_roberta_base_512_mean.npy` / `Movies_openai_clip-vit-large-patch14.npy` |
| Toys | `(20695, 768)` / `(20695, 768)` | `Toys_roberta_base_512_mean.npy` / `Toys_openai_clip-vit-large-patch14.npy` |
| Grocery | `(17074, 768)` / `(17074, 768)` | `Grocery_roberta_base_256_mean.npy` / `Grocery_openai_clip-vit-large-patch14.npy` |
| ele-fashion | `(97766, 512)` / `(97766, 512)` | `clip_feat.pt`, slices `[0:512]` / `[512:1024]` |
| Reddit-S | `(15894, 768)` / `(15894, 768)` | `RedditS_roberta_base_100_mean.npy` / `RedditS_openai_clip-vit-large-patch14.npy` |

All feature matrices were finite. Zero-norm raw nodes were zero for both
modalities in all five datasets. For any zero-norm state/reference pair, the
node-wise cosine implementation assigns 0.0 and records the special-node
count; no node is silently removed.

## CKA and node-wise cosine results

`cka_drop_*` and `cosine_drop_*` are hop-0 value minus the value at the stated
hop. The full hop-by-hop values, retention ratios, quantiles, all-node means,
and special-node counts are in `semantic_retention_summary.csv` and each
dataset's `summary.json`.

| Dataset | Modality | CKA @ formal K | CKA drop @ K | CKA @ 6 | CKA drop @ 6 | CKA slope | Cosine @ formal K | Cosine drop @ K | Cosine @ 6 | Cosine drop @ 6 | Cosine slope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | Text | 0.017971 | 0.982029 | 0.010863 | 0.989137 | -0.108349 | 0.985446 | 0.014554 | 0.983067 | 0.016933 | -0.002283 |
| Movies | Visual | 0.312349 | 0.687651 | 0.237721 | 0.762279 | -0.098030 | 0.828847 | 0.171153 | 0.794057 | 0.205943 | -0.028317 |
| Toys | Text | 0.018202 | 0.981798 | 0.011785 | 0.988215 | -0.108020 | 0.989126 | 0.010874 | 0.987026 | 0.012974 | -0.001802 |
| Toys | Visual | 0.351757 | 0.648243 | 0.282733 | 0.717267 | -0.091303 | 0.861576 | 0.138424 | 0.830445 | 0.169555 | -0.023801 |
| Grocery | Text | 0.011794 | 0.988206 | 0.003044 | 0.996956 | -0.107784 | 0.991101 | 0.008899 | 0.986961 | 0.013039 | -0.001873 |
| Grocery | Visual | 0.165801 | 0.834199 | 0.055613 | 0.944387 | -0.110836 | 0.876408 | 0.123592 | 0.810596 | 0.189404 | -0.027519 |
| ele-fashion | Text | 0.379180 | 0.620820 | 0.417934 | 0.582066 | -0.064109 | 0.891707 | 0.108293 | 0.852753 | 0.147247 | -0.023798 |
| ele-fashion | Visual | 0.148742 | 0.851258 | 0.167109 | 0.832891 | -0.093432 | 0.908260 | 0.091740 | 0.881086 | 0.118914 | -0.018849 |
| Reddit-S | Text | 0.581606 | 0.418394 | 0.581606 | 0.418394 | -0.044828 | 0.986595 | 0.013405 | 0.986595 | 0.013405 | -0.001436 |
| Reddit-S | Visual | 0.847091 | 0.152909 | 0.847091 | 0.152909 | -0.016383 | 0.883377 | 0.116623 | 0.883377 | 0.116623 | -0.012495 |

Sanity checks gave `CKA(Q_0, X)=1.0` and mean valid-node
`cosine(Q_0[i], X[i])=1.0` for every dataset and modality. All 7 states for
both modalities were finite in every dataset.

## Objective answers

### 1. Does repeated propagation systematically change CKA relative to raw features?

Yes at the endpoint level. Every dataset/modality pair has a CKA value below
1 at formal `K` and at `K=6`, with negative linear slope over hops 0–6. The
endpoint change is therefore present across all ten cases. However, this does
not mean every intermediate step is strictly decreasing.

### 2. How much does it change within formal K?

At formal `K`, CKA drops range from 0.152909 (Reddit-S visual) to 0.988206
(Grocery text). Mean node-wise cosine drops range from 0.008899 (Grocery
text) to 0.171153 (Movies visual). Thus CKA detects a much larger feature-space
relationship change than the average node-wise cosine in several cases; the
two statistics should not be treated as interchangeable.

### 3. Does the trend continue through the K=6 stress range?

For Movies, Toys, and Grocery, both CKA and mean node-wise cosine are lower at
6 than at formal K. For ele-fashion, cosine continues to decrease, but CKA
rises after its low point at hop 1/3 and is higher at hop 6 than at formal K,
while remaining below hop 0. Reddit-S reaches an almost exact plateau after
hop 1 for both metrics. The stress range therefore confirms overall endpoint
shift but not universal continued stepwise dilution after formal K.

### 4. Do Text and Visual have different trajectories?

Yes. In Movies, Toys, and Grocery, text CKA falls to roughly 0.01–0.02 by
formal K, whereas visual CKA remains around 0.17–0.35. In ele-fashion, visual
CKA shifts more at formal K than text CKA. In Reddit-S, text CKA is lower than
visual CKA but both plateau after hop 1. Cosine shows a different modality
ordering: visual cosine decreases more in Movies, Toys, Grocery, and Reddit-S,
while text decreases slightly more in ele-fashion.

### 5. Do CKA and node-wise cosine agree in direction?

Yes for the endpoint direction: all ten cases have positive drops at formal K
and K=6, and negative hop-0-to-6 slopes for both metrics. They disagree in
magnitude and trajectory detail. CKA is especially sensitive to the
feature-space change in MAGB text features, while cosine remains near 1 in
those same cases.

### 6. Are there clearly non-monotonic dataset/modality cases?

Yes. Ele-fashion is clearly non-monotonic in CKA for both modalities. Its
trajectories are:

```text
Text:   1.000000, 0.324491, 0.655610, 0.379180, 0.481710, 0.387013, 0.417934
Visual: 1.000000, 0.132322, 0.361012, 0.148742, 0.207167, 0.150531, 0.167109
```

Toys visual also has a smaller hop-1-to-hop-2 CKA increase. Reddit-S is best
described as a post-hop-1 plateau, with tiny floating-point variation rather
than a meaningful monotonic trajectory. Mean node-wise cosine is strictly
decreasing at every step for Movies, Toys, Grocery, and ele-fashion; Reddit-S
decreases at hop 1 and then remains flat.

### 7. Strongest and weakest semantic-shift cases

The answer depends on the metric and modality, so it is reported without
combining them into a new score:

- By formal-K CKA drop, Grocery text is strongest (0.988206), ele-fashion
  visual is next (0.851258), and Reddit-S visual is weakest (0.152909).
- By formal-K cosine drop, Movies visual is strongest (0.171153), followed by
  Toys visual (0.138424); Grocery text is weakest (0.008899).
- By K=6 CKA drop, Grocery text is strongest (0.996956), while Reddit-S
  visual remains weakest (0.152909).
- By K=6 cosine drop, Movies visual is strongest (0.205943), while Grocery
  text is weakest among the reported cases (0.013039).

### 8. Does any result fail to support the current Figure 1(b) motivation?

No dataset is an outright counterexample to the broad motivation that ordinary
physical propagation can shift modality-intrinsic features: every endpoint
shows a CKA and cosine change from hop 0. The qualification is that ele-fashion
does not show a monotonic CKA trajectory and Reddit-S largely plateaus after
hop 1. Therefore a claim of universal step-by-step dilution, equal modality
behavior, or performance degradation would not be supported by this analysis.

## Most defensible statement

**B. “Repeated physical-graph propagation generally shifts
modality-intrinsic representations, while the magnitude and trajectory are
dataset- and modality-dependent.”**

Option A is too strong if “consistently reduces” is read as strict decline at
every hop: ele-fashion has clear CKA rebounds and Reddit-S plateaus. Option C
is too weak because endpoint CKA and cosine shifts occur in all five datasets
and both modalities.

This report does not interpret the result as oversmoothing, does not claim a
performance decrease, and does not use test performance to explain any
trajectory. It also does not establish that an anchor or any MoPF mechanism
is effective.

## Audit and reproducibility

Run provenance:

- Git branch: `vnext`
- Git commit: `b080cb8058628eea3b9da708073f72a083d82a6e`
- Device: `cuda:0`
- CKA sampling seed: `20260914`
- CKA maximum: `20,000` connected nodes
- Analysis hops: `0,...,6`
- Propagation: fixed uniform physical `A`, one analysis self-loop/node,
  symmetric `D_tilde^(-1/2)(A+I)D_tilde^(-1/2)` normalization
- Runtime: approximately 11.39 seconds total; exact per-dataset times are in
  `run_manifest.json`

The loader call uses the existing NC loader and therefore loads the standard
NC data object, but E0-B does not access labels or split indices in its
calculations. No label value was used. No model was instantiated, no optimizer
step occurred, no checkpoint was read or written, and no final model/config or
formal protocol was modified. The fixed propagation definition and thresholds
were not changed after inspecting results.

## Files and command

Added:

- `scripts/run_mopf_e0b_semantic_retention.py`
- `docs/mopf_e0b_semantic_retention.md`

Command:

```bash
conda run --no-capture-output -n yhf_env \
  python scripts/run_mopf_e0b_semantic_retention.py \
  --device cuda:0 \
  --sampling-seed 20260914 \
  --max-cka-nodes 20000 \
  --output-root outputs/e0_empirical_motivation/semantic_retention
```

Outputs:

- `outputs/e0_empirical_motivation/semantic_retention/semantic_retention_summary.csv`
- `outputs/e0_empirical_motivation/semantic_retention/semantic_retention_cka.png`
- `outputs/e0_empirical_motivation/semantic_retention/semantic_retention_cka.pdf`
- `outputs/e0_empirical_motivation/semantic_retention/semantic_retention_node_cosine.png`
- `outputs/e0_empirical_motivation/semantic_retention/semantic_retention_node_cosine.pdf`
- `outputs/e0_empirical_motivation/semantic_retention/run_manifest.json`
- Each dataset directory: `summary.json`, `node_semantic_retention.npz`,
  and `cka_node_ids.npy`

## Warnings and limitations

- This is a descriptive propagation audit, not a confidence interval,
  significance, or causal analysis.
- CKA uses a deterministic connected-node subset capped at 20,000; node-wise
  cosine summaries use all connected nodes and separately retain all-node means.
- The physical support is taken from the existing loader's preprocessed
  `edge_index`, then used directly as the already-undirected `A`; no independent
  graph-file parser was introduced.
- A zero-norm state/reference pair is assigned cosine 0.0 and counted
  explicitly. No such raw zero-norm nodes occurred in these five datasets.
- CKA and node-wise cosine quantify different aspects of representation change;
  their magnitudes should not be compared as if they were the same quantity.
- The artifacts are intended for later conditional analysis, but no later F2
  performance result was used to redefine this E0-B analysis.

E0-C, E0-D, and F2 were not started.
