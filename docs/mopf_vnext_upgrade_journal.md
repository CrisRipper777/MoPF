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

Initialization equivalence passed on all five datasets. All 30 runs passed finite/bounds/pathology checks. The current test suite after U1 reports `148 passed`.

### 2.4 Downstream result

Mean Validation Accuracy difference, S1 minus S0:

| Dataset | S1-S0 Val Acc |
|---|---:|
| Movies | +0.040 pp |
| Toys | +0.081 pp |
| Grocery | +0.010 pp |
| ele-fashion | -0.085 pp |
| Reddit-S | +0.031 pp |

The effect is therefore performance-neutral at the scale relevant to this screening stage. The U1 report also records mixed, very small Test changes rather than a consistent gain.

### 2.5 Mechanism result reported by the experiment

- structural conductance/operator change detected: 5/5 datasets;
- conductance collapse/pathology: none;
- perspective specialization threshold met: 0/5 datasets;
- automatic screening decision: `Conditional candidate`.

### 2.6 Code-audit diagnosis: exact permutation-symmetry lock

The current S1 implementation initializes all four perspective metric vectors **exactly identically** and combines them through a uniform mean. Each perspective has the same architecture and the loss is permutation-symmetric in the perspective index.

Consequently, at initialization:

`w_1 = w_2 = w_3 = w_4`

and therefore every perspective receives the same gradient. With the same optimizer state, the equality is preserved throughout training (up to irrelevant numerical noise). The four perspectives are thus structurally unable to spontaneously specialize under the current parameterization.

This changes the interpretation of the U1 result:

**0/5 perspective specialization is expected from the implementation and cannot be used as evidence that the datasets do not need multiple semantic perspectives.**

The current S1 is functionally equivalent, at inference, to repeatedly evaluating one learned modality-specific diagonal cosine metric and averaging four identical copies. The useful signal observed in U1 is therefore better interpreted as evidence for a **learned modality-adaptive anisotropic semantic metric**, not yet for a genuine multi-perspective metric.

### 2.7 Additional code-audit findings

1. The diagnostic field named `weight_pairwise_cosine_similarity` is currently computed with a Pearson-correlation helper rather than an actual cosine-similarity formula. This is a reporting bug and should be corrected before using weight-similarity evidence in a paper.
2. `_multi_perspective_cosine_values` recomputes `hidden * metric_weight` for the full node matrix inside every edge chunk. This is mathematically correct but unnecessarily expensive. A chunk-local weighted gather, or one weighted node matrix per perspective, would avoid repeated full-node multiplication.
3. Neighborhood-entropy aggregation currently includes zero-degree nodes as entropy zero through `np.bincount(..., minlength=num_nodes)`. For a selectivity interpretation, non-isolated-node statistics should be reported separately or used as the primary value.
4. The conductance-bound diagnostic hard-codes `0.1`; it should read `model.edge_weight_min` if the diagnostic is to remain valid for later variants.

None of these issues invalidates the reported downstream S0/S1 numbers. The first issue above, however, invalidates the intended claim that U1 has already tested learnable multi-perspective specialization.

### 2.8 U1 revised decision

**Status: U1 mechanism not yet resolved. Do not promote the current S1 as the vNext structuralization module.**

The automatic `Conditional candidate` label is acceptable as an experiment bookkeeping label, but the scientific interpretation is more specific:

- performance preservation: supported;
- learned modality-specific conductance change: supported;
- genuine multi-perspective specialization: **not tested successfully because of symmetry lock**.

### 2.9 Required next step before U2

Run a small **U1-R / U1.1 symmetry-resolution study** before moving to the orthogonal response bank.

Recommended comparison:

- `R0`: Frozen S0 ordinary separate cosine;
- `R1`: one learned modality-specific diagonal semantic metric;
- `R2`: four symmetry-broken perspectives initialized as small zero-mean perturbations around the all-ones metric, with no diversity/orthogonality loss.

The R2 initialization must remain close to ordinary cosine at the aggregate-score level, but individual perspective vectors must be non-identical from step 0. If R2 still converges back to one metric across datasets, then multi-perspective structure is empirically redundant and U1 should intentionally collapse to the simpler R1 formulation. If R2 shows stable specialization without hurting validation performance, retain the genuine multi-perspective version.

No U2/Jacobi-bank modification should begin until this question is resolved.

---

## 3. Current candidate state

| Component | Current status |
|---|---|
| Frozen MoPF-v0 | Reference, immutable |
| U1 S1 current four-perspective implementation | Conditional bookkeeping candidate; scientifically unresolved due symmetry lock |
| Learned modality-adaptive diagonal semantic metric | Supported as a plausible low-risk interpretation of the U1 result |
| Genuine multi-perspective semantic conductance | Requires U1-R symmetry-broken test |
| U2 Orthogonal Response Bank | Not started |
| U3 Hierarchical Transfer Function upgrade | Not started |
| U4 Complementary Fusion | Not started |
| U5 Targeted regularization | Not started; only allowed after a concrete pathology is identified |
| LP validation | Deferred until NC architecture is nearly finalized |

---

## 4. Next update

The next journal update should append the U1-R design, implementation audit, three-seed NC results, mechanism diagnostics, and the final structuralization decision. The journal should then freeze the selected relation-level operator before U2 begins.
