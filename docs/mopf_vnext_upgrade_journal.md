# MoPF-vNext Upgrade Journal

This document tracks the controlled upgrade process from the frozen Original MoPF toward MoPF-vNext. It records the planned roadmap, experimental stages, code audits, mechanism diagnoses, candidate decisions, and stopping criteria. The document is intended to be updated after every upgrade stage.

## 0. Governing principles

The vNext process follows five rules:

1. Freeze the validated Original MoPF and never move the reference behavior.
2. Change one modeling axis at a time so that downstream and mechanism effects remain attributable.
3. Prefer mechanisms with an explicit scientific object and a dedicated diagnostic, rather than complexity for its own sake.
4. Do not use Test metrics for checkpoint or candidate selection.
5. LP is postponed until the NC architecture is nearly finalized.

Planned roadmap:

`F0 Freeze -> U1 Semantic Conductance -> U2 Orthogonal Structural Response Bank -> U3 Hierarchical Node-Modality Transfer Function -> U4 Complementary Fusion -> U5 Targeted Regularization only if a concrete pathology remains -> LP cross-task validation`

---

## 1. F0 — Freeze Original MoPF

### 1.1 Goal

Create a stable, immutable control for all later architecture upgrades.

### 1.2 Frozen identity

- Last pre-manifest behavior commit: `1f8ebf127435a5d88b3f9d09118c87e704ad5ea5`
- Freeze/reference manifest commit: `9b03e6fbeb73d888050833391d9859964335b9ea`
- Formal tag: `mopf-v0-frozen`
- Development branch: `vnext`
- Frozen test status: `143 passed, 1 warning`

The distinction above is intentional: `1f8ebf...` is the last commit containing the frozen model behavior; `9b03e6...` adds the manifest and is the tagged freeze point from which `vnext` was created.

### 1.3 Formal MoPF-v0 architecture

Frozen formal settings:

- `edge_weight_mode=separate_cos`
- `node_conditioner_mode=absolute`
- `hrc_weight=0`
- `ppc_weight=0`
- `use_modality_residual=true`
- `use_node_residual=true`
- `fusion_mode=concat_residual_mlp`

Historical PDC-v1/PDC-v2, HRC, and PPC implementations remain in the repository only for analysis/development and are not part of formal MoPF-v0.

### 1.4 NC protocol

Datasets: Movies, Toys, Grocery, ele-fashion, Reddit-S.

Seeds: 42/43/44.

Orders: Movies/Toys/ele-fashion/Reddit-S use `K=3`; Grocery uses validation-selected `K=2`.

Checkpoint selection: best Validation Accuracy. Macro-F1 uses the repaired fixed label-set protocol.

### 1.5 F0 decision

**Status: Frozen reference established.**

All subsequent upgrades are evaluated relative to this behavior.

---

## 2. U1 — Modality-Conditioned Semantic Conductance

### 2.1 Scientific question

Does a single physical edge carry the same semantic transport strength in different modalities?

The frozen model uses one ordinary cosine score per modality on the original graph support. U1 tests whether a learned multi-perspective semantic metric can provide a richer modality-conditioned conductance while leaving graph support, graph normalization, propagation bank, personalized filter, fusion, and task loss unchanged.

### 2.2 Variants

**S0 — Frozen Separate Cosine**

`separate_cos`, identical to MoPF-v0.

**S1 — Multi-Perspective Semantic Conductance**

For each modality and each of four perspectives, S1 learns a positive feature-wise metric vector and computes weighted cosine similarity on the original edges. The four perspective scores are uniformly averaged before the same sigmoid conductance transform and existing `gcn_norm` path.

The metric vectors are initialized with `softplus(theta)=1`, so S1 begins numerically equivalent to ordinary cosine.

### 2.3 Experimental scope

- 5 NC datasets
- S0/S1
- seeds 42/43/44
- 30 formal training runs
- LP excluded

Initialization equivalence passed exactly on all five datasets: the aggregate S1 metric score equals S0 ordinary cosine at initialization (reported mean and maximum absolute differences are zero). All 30 runs passed finite/bounds/pathology checks. The test suite after U1 reports `148 passed`.

### 2.4 Downstream result

Mean S1-minus-S0 changes across three seeds:

| Dataset | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |
|---|---:|---:|---:|---:|
| Movies | +0.040 pp | -0.262 pp | +0.180 pp | +0.340 pp |
| Toys | +0.081 pp | +0.381 pp | -0.105 pp | +0.122 pp |
| Grocery | +0.010 pp | +0.294 pp | -0.088 pp | -0.066 pp |
| ele-fashion | -0.085 pp | -0.119 pp | -0.052 pp | -0.088 pp |
| Reddit-S | +0.031 pp | +0.079 pp | -0.073 pp | -0.166 pp |

The unweighted mean across dataset means is approximately `+0.015 pp` Validation Accuracy, `+0.075 pp` Validation Macro-F1, `-0.028 pp` Test Accuracy, and `+0.028 pp` Test Macro-F1. U1 is therefore performance-neutral at the screening scale; it does not provide evidence of a robust downstream gain or loss.

### 2.5 Mechanism result reported by the experiment

The automatic U1 summary reports:

- structural conductance/operator change detected: 5/5 datasets;
- conductance collapse/pathology: none;
- perspective specialization threshold met: 0/5 datasets;
- automatic screening decision: `Conditional candidate`.

The scalar text-vs-visual conductance/operator summaries change only modestly from S0 to S1. For example, the mean absolute text-visual conductance gap changes by roughly -4.4% on Movies and within about +/-1.4% on the other four datasets; the within-model normalized text-vs-visual operator-distance scalar changes by roughly -2.35% on Movies and within about +/-1.16% elsewhere.

### 2.6 Code-and-data audit: exact permutation-symmetry lock

The current S1 implementation initializes all four perspective metric vectors **exactly identically** and combines them through a uniform mean. Each perspective has the same architecture and the task loss is permutation-symmetric in the perspective index.

At initialization:

`w_1 = w_2 = w_3 = w_4`

and every perspective receives the same gradient. With identical optimizer state, equality is preserved throughout training.

The authoritative U1 master JSON confirms this prediction rather than merely suggesting it. Across all S1 checkpoints inspected:

- perspective-score mean absolute deviation from the four-perspective aggregate is exactly zero;
- across-perspective weight standard deviation is exactly zero;
- pairwise perspective score correlations are numerically one (up to floating-point roundoff).

Therefore:

**0/5 perspective specialization is a consequence of the present parameterization and cannot be interpreted as evidence that the datasets intrinsically do not require multiple semantic perspectives.**

The current S1 is functionally equivalent at inference to evaluating one learned modality-specific diagonal cosine metric four times and averaging identical copies.

### 2.7 What U1 has actually tested

The scientifically defensible interpretation of S1 is currently:

**learned modality-adaptive anisotropic semantic metric**

rather than:

**genuine multi-perspective semantic conductance**.

The learned diagonal metrics do move away from the identity, but only mildly. Across S1 runs the representative per-dimension metric weights typically deviate from one by roughly 1–2% on average, with observed extrema in the approximate range 0.95–1.07. Thus the model learns a small anisotropic rescaling instead of leaving the metric exactly at ordinary cosine.

This is compatible with the performance-neutral result: relation-level geometry can be refined without destabilizing the existing propagation/filtering pipeline.

### 2.8 Important limitation of the current `structural_change` criterion

The automatic `structural_change` flag does **not** directly compute the operator difference between S0 and S1 on a common hidden representation. It checks whether scalar summaries such as the S1 text-vs-visual conductance gap or the S1 text-vs-visual normalized-operator distance differ from their S0 counterparts by more than `1e-5`.

Because S0 and S1 are separately trained models, this criterion mixes metric effects with ordinary co-adaptation of the projection/filter/fusion parameters. It is useful as a screening indicator, but it is not a frozen causal test of semantic-metric utilization.

A stronger structuralization audit must hold the trained S1 checkpoint and hidden representations fixed, then replace the learned metric with identity weights and measure:

- edge-score change;
- conductance change;
- normalized-operator change;
- representation/logit change;
- Validation/Test change under the same checkpoint.

This frozen intervention should be added to U1-R.

### 2.9 Additional implementation-audit findings

1. The diagnostic field named `weight_pairwise_cosine_similarity` is computed with the generic Pearson-correlation helper, not a cosine-similarity formula. This is a reporting bug. It does not affect the symmetry-lock conclusion because identical perspectives produce 1 under either measure, but it must be fixed before future specialization analysis.
2. `_multi_perspective_cosine_values` recomputes `hidden * metric_weight` for the full node matrix inside every edge chunk. This is mathematically correct but unnecessarily expensive. The U1 data reflect the cost: S1 adds only 2,048 parameters (about 0.2% of the ~1M-parameter NC model) yet the unweighted mean training time across dataset means rises from about 41.2 s to 67.2 s. This overhead is especially difficult to justify while all four perspectives are identical.
3. Neighborhood-entropy aggregation includes zero-degree nodes as entropy zero through `np.bincount(..., minlength=num_nodes)`. Future selectivity analysis should report non-isolated-node entropy separately or use it as the primary statistic.
4. The conductance-bound diagnostic hard-codes `0.1`; it should read `model.edge_weight_min` so later hyperparameter variants remain valid.

### 2.10 U1 revised decision

**Status: performance-safe structuralization candidate, but current four-perspective formulation is not acceptable as the final vNext module.**

The automatic `Conditional candidate` label remains valid as bookkeeping, but the scientific decision is split:

- performance preservation: **supported**;
- learned modality-specific anisotropic metric: **plausible and worth retaining for direct testing**;
- causal utilization of the learned metric: **not yet established by the current S0/S1 comparison**;
- genuine multi-perspective specialization: **not tested because of exact symmetry lock**.

Do not promote current S1 directly into U2.

### 2.11 Required next step before U2: U1-R / U1.1

Run a compact symmetry-resolution and causal-utilization study:

- `R0`: Frozen S0 ordinary separate cosine;
- `R1`: one learned modality-specific diagonal semantic metric;
- `R2`: four genuinely symmetry-broken perspectives initialized as small zero-mean perturbations around the all-ones metric, with no diversity/orthogonality loss.

R2 must satisfy two conditions simultaneously: individual perspectives are non-identical from step 0, while their aggregate initial edge score remains highly aligned with ordinary cosine. The initialization perturbation should be small enough to preserve the frozen model's starting geometry.

Each trained R1/R2 checkpoint should also receive a frozen **identity-metric-off** intervention to isolate the functional contribution of learned semantic geometry from co-adaptation elsewhere in the network.

Decision rule:

- if R1 preserves performance and the frozen identity intervention shows real operator/representation/logit dependence, retain a simple **Modality-Adaptive Semantic Conductance** formulation;
- if R2 additionally develops reproducible, nontrivial specialization without harming validation performance, retain the multi-perspective form;
- if R2 re-collapses to one metric after symmetry is genuinely broken, reject the extra perspective capacity and use R1;
- if neither R1 nor R2 shows frozen functional utilization, revert to S0 and do not carry semantic-metric decoration forward.

No U2 response-bank modification should begin until this relation-level operator is selected.

---

## 3. Current candidate state

| Component | Current status |
|---|---|
| Frozen MoPF-v0 | Reference, immutable |
| U1 S1 current four-perspective implementation | Performance-safe but structurally symmetry-locked; not final |
| Learned modality-adaptive diagonal semantic metric | Primary low-risk U1-R candidate |
| Genuine multi-perspective semantic conductance | Requires symmetry-broken R2 test |
| Frozen metric-utilization evidence | Missing; required in U1-R |
| U2 Orthogonal Response Bank | Not started |
| U3 Hierarchical Transfer Function upgrade | Not started |
| U4 Complementary Fusion | Not started |
| U5 Targeted regularization | Not started; only allowed after a concrete pathology is identified |
| LP validation | Deferred until NC architecture is nearly finalized |

---

## 4. Next update

The next journal update should append the U1-R design, code audit, three-seed NC results, frozen metric-utilization diagnostics, specialization diagnostics, efficiency comparison, and the final relation-level structuralization decision. Only after that decision should the selected operator be frozen and U2 begin.

---

## 5. U1-R — Semantic Metric Resolution and Frozen Functional Audit

U1-R completed the required symmetry-resolution and causal-utilization study
before any U2 work. The scope remained NC only; LP, sports-copurchase, the
decoder, sampler, and LP protocol were not run or modified.

### 5.1 Controls and implementation

- R0 reused the 15 U1 S0 checkpoints after a strict frozen-tag equivalence
  audit. The current R0 path matched `mopf-v0-frozen` exactly on a deterministic
  six-node audit (`max_abs_difference = 0.0`) and registered no metric
  parameters.
- R1 uses one learned positive diagonal metric per modality, normalized by its
  mean plus epsilon.
- R2 uses four positive mean-normalized perspectives with deterministic
  zero-mean RMS-0.01 symmetry-breaking initialization and no diversity loss.
- All five dataset initialization audits passed: R1 retained ordinary-cosine
  initialization equivalence, while R2 preserved aggregate alignment and made
  perspectives and scores non-identical.

### 5.2 Formal results

- 30 new runs were completed: R1/R2 × five datasets × seeds 42/43/44.
- R0 reused the 15 U1 checkpoints, giving 45 per-run records in the U1-R
  summary.
- The unified full-graph NC protocol used AdamW, hidden dimension 256, a
  300-epoch cap, patience 30, and best validation accuracy checkpoint
  selection. Test metrics were not used for selection.
- Same-checkpoint frozen interventions covered identity metrics, stream-specific
  identity, R2 collapse-to-mean, edge scores, conductance, normalized
  operators, propagation banks, modality and fused representations, logits,
  probabilities, predictions, and downstream metrics.
- Conductance pathology checks were finite and within configured bounds for all
  completed R1/R2 runs.

### 5.3 Decision

- R1 passed the relation-level screening criteria on all five datasets:
  validation same-band, normalized metric non-identity, and repeated frozen
  functional change.
- R2 showed repeated perspective specialization, but the predefined
  collapse-to-mean functional-change criterion was not met on any dataset.
- Decision: **Select R1 — Modality-Adaptive Semantic Conductance**.

This is a relation-level structuralization decision only. It is not a final
MoPF claim or a statistical-significance claim. U2 must use R1 as the selected
relation-level candidate, and all U1-R artifacts remain immutable.

Authoritative U1-R artifacts:

- `outputs/u1r_semantic_metric_resolution/u1r_master_summary.json`
- `outputs/u1r_semantic_metric_resolution/u1r_master_table.csv`
- `outputs/u1r_semantic_metric_resolution/initialization_audit.json`
- `outputs/u1r_semantic_metric_resolution/r0_equivalence_audit.json`
- `docs/mopf_u1r_semantic_metric_resolution.md`

## U1-R Independent Post-Audit

The independent post-audit preserves the U1-R historical decision while tightening the interpretation of frozen functional evidence:

- R2 is rejected: perspective specialization did not produce a sufficient functional effect under the collapse-to-mean frozen intervention on all five datasets.
- R1 is preferred: the single Modality-Adaptive Semantic Metric is non-identity, performance-safe, and is the only relation-level candidate carried into U1-T.
- The previous `functional=true` threshold was too permissive for a scientific materiality claim.
- R1 frozen learned-vs-identity effects are mainly in the `1e-5`–`1e-4` scale.
- The conductance transform was therefore treated as the suspected attenuation bottleneck.
- U1-T was introduced to test one global fixed conductance temperature without changing the frozen R1 metric form or any downstream module.

## U1-T — Semantic Conductance Calibration

U1-T kept the R1 metric frozen:

`s_ij^m = cos(w^m ⊙ h_i^m, w^m ⊙ h_j^m)`, with `w^m = softplus(theta^m) / mean(softplus(theta^m))`.

Only the global fixed scalar `edge_weight_temperature=tau` was varied, shared by both modalities and all datasets. Frozen temperature overrides were routed through an explicit analysis-only `temperature_override` argument; model state and checkpoint bytes were verified unchanged.

### Phase A — Frozen Temperature Diagnosis

- Source: all 15 U1-R R1 best-validation checkpoints.
- Grid: `2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25`.
- Raw learned and identity semantic scores were invariant across tau within `<1e-7`.
- All runs were finite, retained edge support and the GCN-normalization support family, and showed no prediction or conductance collapse.
- `tau=1.5` and `1.0` did not reach 2x mechanism amplification.
- `tau=0.75, 0.5, 0.35, 0.25` passed the Phase-A validation/mechanism gate.
- Candidate-Mild: `tau=0.75`.
- Candidate-Strong: `tau=0.35`.

### Phase B — Selected Temperature Retraining

Both candidates were retrained under the unchanged full-graph NC protocol, 5 datasets × 3 seeds each. Each new best checkpoint received a learned-vs-identity audit at its selected tau and a learned tau-vs-2.0 temperature-off audit.

The final candidate comparison was validation-first. `tau=0.35` had an unweighted five-dataset validation-accuracy delta of `+0.0137 pp` versus T0; its largest dataset mean drop was `-0.1700 pp` on Movies. It reached Functionally Amplified on 5/5 datasets, Functionally Material on 4/5, and the temperature-off audit remained active on all 5 datasets.

### U1-T relation-level decision

**Select Calibrated R1 with tau = 0.35.**

The frozen relation module is now:

`edge_weight_mode=learned_diag_cos`

`c_ij^m = c_min + (1-c_min) sigmoid(s_ij^m / 0.35)`

R0 and R2 code paths remain available for ablations. `mopf-v0-frozen` was not modified. U2 and all propagation-bank, personalized-filter, fusion, auxiliary-loss, and LP changes remain prohibited until a separately authorized stage.

## U2 — Structural Response Representation

U1 learns modality-conditioned transport geometry. U2 represents the structural
responses induced by that geometry. U3 will adaptively combine those responses.

## U2-A — Monomial Response Bank Diagnosis

U2-A diagnosed the frozen monomial response bank without training or changing
the formal model:

`R_0^m = H_0^m`, `R_k^m = P^m R_{k-1}^m`, for the frozen U1-T Calibrated R1
module (`edge_weight_mode=learned_diag_cos`, `tau=0.35`). The source was only
the 15 final U1-T Phase-B T2 best-validation checkpoints (5 datasets × 3
seeds), with `K={Movies:3, Toys:3, Grocery:2, ele-fashion:3, Reddit-S:3}`.

The analysis extracted the exact raw bank before any response normalization,
filter, node conditioner, refinement, fusion, or classifier. It measured
Frobenius cosine, feature-space linear CKA, normalized-response Gram spectrum
and conditioning, incremental structural response novelty, norm evolution,
node-feature variance, operator-native structural variation, sparse operator
symmetry, and conditional Dirichlet energy. For `ele-fashion`, CKA used the
same deterministic 20,000-node subset for every order; smaller graphs used all
nodes. No `N×N` kernel matrix was constructed.

The authoritative artifacts are:

- `outputs/u2a_monomial_response_diagnosis/u2a_master_summary.json`
- `outputs/u2a_monomial_response_diagnosis/u2a_master_table.csv`
- `docs/mopf_u2a_monomial_response_diagnosis.md`

### U2-A findings

- Strong monomial-redundancy screening evidence was present in all 5/5
  datasets for both text and visual, with the same classification on 3/3
  seeds in every dataset/modality cell.
- High-order Frobenius cosine means were `0.9758–1.0000`; high-order CKA means
  were `0.8904–1.0000`. Normalized effective rank was `0.288–0.459`, while
  Gram condition numbers ranged from about `112` to `1.1×10^7`.
- Final-order novelty means were `0.0012–0.2231`, providing continuous
  evidence that later orders add little incremental structural response,
  especially on Reddit-S.
- Final response norms and node-feature variances generally decayed relative
  to order 0; final structural-variation ratios were `0.0007–0.2172`. This
  supports progressive smoothing/collapse as a mechanism, while retaining the
  dataset- and modality-specific magnitudes in the JSON rather than asserting
  a universal monotonic law.
- Text and visual behavior was similar but not identical: for example, text
  had higher final-order cosine than visual on all five dataset means, while
  visual had higher final Gram condition number on Reddit-S and text had the
  higher condition number on the other four datasets.

### Operator and spectral prerequisite

All 30 operator audits (5 datasets × 3 seeds × 2 modalities) were classified
`near_symmetric`; asymmetry ratios were approximately `2.57×10^-8` to
`3.33×10^-8`. Sparse eigsh estimates (with deterministic Rayleigh fallback
where needed) had maximum absolute eigenvalue in
`[0.9998856, 1.0000003]`, with no spectral-range failure. The report therefore
marks the clean symmetric spectral prerequisite as resolved for a future
orthogonal-polynomial study.

The GPU smoke path exposed repeat-call atomic scatter drift up to a few
`10^-6` on this operator, above the required `<1e-6` equivalence gate. The
authoritative 15-checkpoint audit consequently ran on CPU using the same
formal recurrence and obtained `max_abs_difference=0.0` for every checkpoint.
All checkpoint SHA256 values and model state digests were unchanged.

### U2-A gate

**Proceed to U2-B: monomial response redundancy/conditioning is empirically supported.**

This is the U2-B gate only; U2-A itself is now stopped. The pending final-paper
2×2 relation-attribution ablation remains registered: A `separate_cos,tau=2`,
B `learned_diag_cos,tau=2`, C `separate_cos,tau=0.35`, D
`learned_diag_cos,tau=0.35`. It was not run here and does not block U2.

## U2-B — Semantic-Preserving Distinctive Multi-Hop Propagation

U2-B screened frozen response-coordinate mechanisms only. It used the 15 final
U1-T Phase-B T2 best-validation checkpoints (5 datasets × 3 seeds), the frozen
`learned_diag_cos` relation with `tau=0.35`, and the formal dataset depths
`K={Movies:3, Toys:3, Grocery:2, ele-fashion:3, Reddit-S:3}`. The analysis
stress-tested `K=1..6` without changing any formal training configuration,
checkpoint, optimizer, LP path, or model source. No training step was run.

The four analysis variants were:

- B0: current cumulative states, `S_0=H_0`, `S_k=P S_{k-1}`, `C_k=S_k`;
- B1: ordinary propagation with hop innovations,
  `C_0=S_0`, `C_k=S_k-S_{k-1}`;
- B2: semantic anchoring,
  `S_k=(1-alpha)P S_{k-1}+alpha H_0`, `C_k=S_k`, with
  `alpha={0.05,0.1,0.2}`;
- B3: anchored propagation plus hop innovations, using the same alpha grid.

Innovation reconstruction was audited on CPU in float64 with tolerance
`1e-6`; the maximum reconstruction error across all 15 checkpoints,
modalities, variants, and stress depths was `0.0`. B0 formal equivalence to
the original response bank also had maximum absolute error `0.0`. Checkpoint
SHA256 values and model-state digests were unchanged.

### U2-B findings

- Semantic anchoring was effective for retention: B2 passed on all 5/5
  datasets with all 3/3 seeds direction-consistent. At K=6, semantic-retention
  CKA for B2 was `0.5902` at the main `alpha=0.1`, versus `0.4615` for B0.
  The sensitivity grid was monotone in the expected direction: alpha `0.05`,
  `0.1`, and `0.2` gave K=6 retention CKA means `0.5256`, `0.5902`, and
  `0.7105`, respectively. Anchoring alone did not solve redundancy: B2 still
  had K=6 high-hop cosine `0.9999` and novelty `0.0023`.
- Hop innovation was effective for distinctiveness: B1 passed on all 5/5
  datasets with all 3/3 seeds direction-consistent. At K=6, B1 reduced
  high-hop Frobenius cosine from the B0 value `0.9994` to `0.6527`, reduced
  high-hop CKA from `0.9979` to `0.7468`, and raised final novelty from
  `0.0044` to `0.1121`. Its final response magnitude ratio was `0.0213`,
  above the vanishing threshold `1e-4`.
- B3 jointly passed on all 5/5 datasets with all 3/3 seeds
  direction-consistent. At the main alpha `0.1` and K=6, it retained the B2
  semantic CKA `0.5902`, while retaining B1-like distinctiveness
  (`0.6527` high-hop cosine, `0.7468` high-hop CKA, `0.1128` novelty). The
  final innovation magnitude ratio was `0.0113`, also above `1e-4`.
- The factorial effects separate the mechanisms: anchoring mainly improved
  semantic-retention CKA (`+0.0998` main effect), while innovation mainly
  reduced high-hop cosine (`-0.7687`), reduced high-hop CKA (`-0.5045`), and
  increased novelty (`+0.5633`). The interaction was small relative to the
  innovation main effect, supporting the combined B3 interpretation.
- The K stress test showed cumulative B0 becoming progressively redundant:
  high-hop cosine increased from `0.9936` at K=3 to `0.9994` at K=6, while
  novelty decreased from `0.0857` to `0.0044`. B1 and B3 retained a distinct
  innovation signal at K=6. Text and visual responses had different
  magnitudes and retention offsets by dataset, but the qualitative mechanism
  direction was consistent across both modalities; the result is not a
  single-modality artifact.

### U2-B gate and decision

**Decision: A — proceed to U2-C with B3, main `alpha=0.1`.** The language is
restricted to mitigating propagation-induced semantic dilution, reducing
multi-hop response redundancy, preserving modality-specific semantics, and
maintaining distinctive structural evidence. It does not claim to solve
oversmoothing or oversquashing, and no downstream U2-C change was implemented
in U2-B.

Orthogonal response construction remains a fallback only if the authorized
U2-C study finds that B3 is insufficient. The pending relation-level 2×2
attribution (`separate_cos/tau2`, `learned_diag_cos/tau2`,
`separate_cos/tau0.35`, `learned_diag_cos/tau0.35`) remains registered and was
not run here.

Authoritative U2-B artifacts:

- `outputs/u2b_semantic_preserving_multihop_screening/u2b_master_summary.json`
- `outputs/u2b_semantic_preserving_multihop_screening/u2b_master_table.csv`
- `docs/mopf_u2b_semantic_preserving_multihop_screening.md`

## U2-C0 — State–Response Decoupling and Differential-Basis Integration Audit

U2-C0 was the final integration/algebra/code-path audit before any authorized
U2-C retraining. It was frozen and inference-only: no NC retraining, LP run,
checkpoint write, downstream redesign, alpha selection, or K tuning was
performed. The audit consumed all 15 U1-T Phase-B T2 best-validation
checkpoints with the frozen `learned_diag_cos` relation and `tau=0.35`, using
only the formal dataset depths
`K={Movies:3, Toys:3, Grocery:2, ele-fashion:3, Reddit-S:3}`.

Repository provenance was checked on branch `vnext` with a clean worktree at
the start (`HEAD=c1fe198`, U2-B). `git switch vnext` succeeded and the local
branch matched `origin/vnext`; `git fetch origin` and `git pull --ff-only origin
vnext` were attempted but both reported the real network failure
`ssh: Could not resolve hostname github.com: Temporary failure in name
resolution`. No synchronization success was inferred from that failure.

### Terminology correction

The earlier U2-B phrase **Hop-wise Structural Innovation** is refined for
mathematical accuracy to **Hop-wise Differential Structural Response**
(逐跳差分结构响应). For `D_k=S_k-S_{k-1}`, the formal interpretation is the
representation change induced by one additional propagation step. It is not
exact k-hop unique information, independent new structural information,
orthogonal innovation, or newly observed k-hop nodes. Earlier U2-B results are
retained, with this wording correction applied to their interpretation.

### Algebraic identities

With `q=1-alpha`, the audit verified in CPU float64:

`S_k=q P S_{k-1}+alpha H`

`D_k=S_k-S_{k-1}`

`D_k=q^k P^(k-1)(P-I)H`

and therefore:

`D_k^B3=q^k D_k^B1`.

Across every dataset, seed, modality, and formal K, the maximum state
reconstruction error was `1.7764e-15` absolute and `1.9581e-17` relative L2;
the maximum closed-form error was `1.8296e-13` absolute and `8.4294e-14`
relative L2; and the maximum B3/B1 scaling error was `1.7053e-13` absolute
and `1.0420e-13` relative L2. The alpha-to-zero limit also passed exactly at
the recorded float64 precision.

The anchor term cancels in the differential values. Semantic anchoring is
represented in cumulative states `S_k`; `D_k` retains the propagation
difference and the `q^k` scaling, not an additive `alpha H` term at each hop.

### State–response decoupling design

The opt-in internal `multihop_mode=anchored_differential` implements:

- `S_k` as the state bank supplied to the node/modality coefficient
  conditioner;
- `D_k` as the response bank supplied to the response filter;
- explicit `states_text`, `states_visual`, `responses_text`,
  `responses_visual`, `eta_text`, `eta_visual`, `z_text`, and `z_visual`
  outputs;
- backward-compatible `bases_*` aliases referring to filter response values.

The historical `cumulative` mode delegates to the existing
`_propagation_bank()` path and remains the default because
`configs/model/mopf.yaml` was not changed. `anchored_differential` is allowed
only with `node_conditioner_mode=absolute`; historical PDC modes fail
explicitly in this prototype path, while cumulative PDC behavior remains
compatible.

### Basis-equivalent coefficients and initialization

The reusable transform supports `[K+1]` and `[N,K+1]` coefficients:

`tilde_eta_0=sum_k eta_k`,

`tilde_eta_j=q^(-j) sum_{k=j}^K eta_k`.

The exact inverse was implemented and round-trip checked in float64. For the
historical `gamma_global` prior, the prototype first constructs the unchanged
monomial prior and then applies the alpha `0.1` transform. Formal K maximum
`q^(-K)` was `1.23457` for Grocery (`K=2`) and `1.37174` for the K=3
datasets; no coefficient explosion occurred.

Current cumulative and anchored-differential models had identical parameter
counts (`996684` total and trainable parameters in the initialization audit),
identical state-key sets, and fixed non-trainable alpha. Toy-graph and real
dataset initialization comparisons passed the required relative-L2 `<1e-6`
criterion for `z_text`, `z_visual`, refined outputs, and fused `z`. The
largest recorded max-absolute difference was `3.8147e-6`; it was retained,
not hidden or accepted by a relaxed tolerance, and is attributed to float32
recurrence/filter accumulation order.

For all frozen learned checkpoints, transformed effective eta coefficients
reconstructed the original filter with maximum relative L2 `1.4601e-7` and
maximum absolute difference `5.7736e-5`. The absolute residual is recorded as
the corresponding float32 checkpoint/accumulation-order effect; the relative
criterion passed without tolerance inflation.

### Anchor functional path

The anchor was verified to enter a nontrivial functional path through
`S_k -> node residual -> eta`. The audit compared current monomial states and
anchored states using identical learned node projectors, node vectors, gamma,
and modality residuals, then composed both eta sets over the same monomial
values. The path was not machine-zero across the 15 checkpoints and both
modalities. This is a frozen mechanism diagnostic, not a task-performance
claim.

### Runtime and formal-integrity audit

The CUDA smoke path ran both cumulative and anchored-differential toy forward/
backward passes with finite loss and finite gradients, with the same-order
`O(K|E|d)` sparse path and no dense `N×N` allocation or eigendecomposition.
All 15 checkpoint bytes and model-state digests were unchanged. The formal
`mopf.yaml` relation, `tau=0.35`, default cumulative behavior, LP/fusion/task
paths, and frozen baseline provenance were unchanged.

### U2-C0 gate

**Decision: A — Proceed to U2-C: state–response integration is mathematically
and numerically valid.**

This gate is based only on algebraic validity, coordinate-transform validity,
initialization equivalence, parameter invariance, numerical integrity, and a
nontrivial frozen anchor-conditioner path. It does not claim downstream
superiority. The next stage must still use the corrected terminology and must
not describe `D_k` as unique or orthogonal k-hop information.

Authoritative U2-C0 artifacts:

- `outputs/u2c0_state_response_integration_audit/u2c0_master_summary.json`
- `outputs/u2c0_state_response_integration_audit/u2c0_master_table.csv`
- `docs/mopf_u2c0_state_response_integration_audit.md`
