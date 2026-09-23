# MGSC-MAG stage finalization report

## Scope and frozen reference

The current paper scope in this round is node classification on Movies, Toys,
Grocery, ele-fashion, and Reddit-S only. Sports-LP and all other LP work are
not discussed. The frozen CoSI reference and the unified NC protocol were not
modified:

- `src/models/cosi_mag_final.py`
- `configs/model/cosi_mag_final.yaml`

The working architecture is canonical P2 in
`configs/model/mgsc_mag_p2.yaml`.

## A. Unified hypothesis

Physical topology defines latent structural dependence, but the semantic
utility of structural context is not uniform at relation, node/context, or
propagation-order granularity. MGSC-MAG therefore separates modality-specific
context-state formation from adaptive multi-order integration.

## B. P2 data flow

Text and visual features are projected independently. MRC produces separate
modality relation responses and normalized graph operators over the unchanged
physical topology. Each order forms a neighborhood proposal; an independent
text/visual gate uses `(H0, N, |H0-N|, H0*N, p_k)` to form `S_k`. The state bank
is `[S0,...,SK]`. Order-embedded states enter the existing cross-order
attention with retained legacy relation-order bias. P2 composes the interacted
states with the existing `eta` coefficients, refines each modality separately,
and fuses only at the final late-fusion stage.

## C. M1 and contextualization demand

M1 used frozen row-normalized modality features, physical neighbors only, no
self-loop in the neighborhood mean, no labels, no MRC, and deterministic
internal train/dev probe splits. Test labels were never used. Gate A passed on
Movies, Grocery, and ele-fashion under the predeclared thresholds. The
node-wise oracle gap is a descriptive motivation upper bound, not a realizable
model or test performance.

## D. Corrected controls

The formal corrected matrix completed 60 runs. Full mean test results were:

| Dataset | Accuracy ± population SD | Macro-F1 ± population SD |
|---|---:|---:|
| Movies | 0.5643 ± 0.0052 | 0.5026 ± 0.0101 |
| Toys | 0.8013 ± 0.0061 | 0.7741 ± 0.0067 |
| Grocery | 0.8331 ± 0.0054 | 0.7581 ± 0.0106 |
| ele-fashion | 0.8831 ± 0.0007 | 0.7812 ± 0.0048 |
| Reddit-S | 0.9691 ± 0.0026 | 0.9320 ± 0.0037 |

Full-minus-control paired means in percentage points were:

| Dataset | Raw terminal Acc/F1 | Raw bank Acc/F1 | Plain backbone Acc/F1 |
|---|---:|---:|---:|
| Movies | +0.480 / +0.812 | +0.610 / −0.063 | +0.080 / +0.075 |
| Toys | −0.354 / −0.405 | +0.153 / +0.303 | +0.540 / +0.544 |
| Grocery | −0.098 / −0.479 | +0.420 / +0.120 | +0.390 / +0.345 |
| ele-fashion | −0.167 / −0.237 | +0.014 / +0.758 | +0.214 / +0.462 |
| Reddit-S | −0.042 / −0.162 | +0.157 / +0.321 | +0.231 / +0.243 |

The strict plain backbone is close to Full despite removing conditional
parameters; raw terminal is sometimes better. Thus the mechanism is real but
not performance-dominant. This is a Tier-B result, not a module ranking.

## E. Mechanism chain

The current implementation forms a real computation chain:

`MRC relation response → adaptive state formation → multi-order state bank →
cross-order interaction → direct interacted-state composition`.

The corrected C0 path is exactly equivalent to canonical MGSC in CPU/CUDA
smoke checks. Raw controls bypass the Stage-II readout as specified, and the
plain control uses exact unit pre-normalization relations and fixed gates.

## F. Figure 3 context diagnostics

Current source data are in `outputs/final/context_formation_analysis/` and
`outputs/final/paper_figures/figure3_data/`. On Movies/Grocery/ele-fashion,
mean text/visual semantic-score discrepancies are 0.183/0.151/0.146 and
corresponding relation-weight discrepancies are 0.056/0.051/0.043. Mean
neighbor TV distances are 0.0162/0.0166/0.0080, with top-neighbor
disagreement 0.578/0.498/0.298. Gate interventions are nonzero on all five
datasets; shuffled-gate embedding MAE ranges from 0.037 to 0.079 and flip
rates from 0.004 to 0.033. These support functional modality/node dependence,
not gate ground truth or causal identification.

## G. Figure 4 integration diagnostics

Current source data are in `outputs/final/multi_order_integration_analysis/`
and `outputs/final/paper_figures/figure4_data/`. Contributions use
`q=abs(eta)/sum(abs(eta))`. Mean effective order is concentrated near order 2,
with text/visual means: Movies 1.741/1.802, Toys 1.818/1.975, Grocery
1.940/1.977, ele-fashion 1.778/1.738, Reddit-S 1.759/1.738. The safe claim
is modality/dataset-dependent order profiles, not strong universal node-level
heterogeneity. Interaction-Off logit MAE is 0.268/0.230/0.383/0.372/0.133
from Movies through Reddit-S, larger than Uniform-Attention in every case;
this establishes direct functional influence without claiming universal
accuracy gains.

## H. Prior-init cleanup diagnostic

Current seed-42 direct-minus-legacy accuracy is −0.120, −0.029, and −0.048
percentage points for Movies, Grocery, and ele-fashion. Macro-F1 changes are
−1.535, −0.301, and +0.477 points. The single-seed evidence does not justify
changing canonical P2; keep legacy-anchored initialization.

## I. Claims and freeze decision

Supported: heterogeneous contextualization demand; a modality-separated,
multi-granular P2 computation chain; measurable gate and interaction effects;
and P2 as a viable five-NC working architecture.

Not supported: semantic similarity as relation reliability, oracle lambda as a
realizable policy, universal module necessity, universal accuracy improvement,
causal mechanism claims, or any LP conclusion.

Recommendation: freeze canonical P2 as the sole working architecture for this
five-NC paper scope. No architecture-search blocker remains, but all claims
should retain the bounded Tier-B language.

## J. Reproducibility and tests

Authoritative corrected-control files are under
`outputs/final/mgsc_corrected_controls/`; the runner manifest records the
dataset, seed, control, pairing, and population-SD protocol. Relevant tests
pass: 17 passed, including CPU finite/backward/integrity tests; CUDA C0 max
absolute difference is 0.0. Full `pytest tests/` remains blocked at
collection by six historical tests importing deleted legacy scripts; those
scripts were not restored.

## Paper evidence finalization

1. **Figure 1 design and audit.** Figure 1(a) uses raw feature cosine
   discrepancy on actual physical edges; Figure 1(b) reuses the leakage-safe
   M1 preferred-lambda distributions; Figure 1(c) uses the new raw fixed-graph
   multi-order linear-probe diagnostic. No MGSC checkpoint is used in the
   motivation panels, and official test labels are not used. The panel-level
   audit is in `docs/final/FIGURE1_EVIDENCE_AUDIT.md`.

2. **Figure 2 architecture.** The architecture preview contains only the two
   top-level stages, keeps text/visual propagation separate, and maps the
   plotted objects to the current code in
   `docs/final/FIGURE2_ARCHITECTURE_SPEC.md`.

3. **Figure 3 claims.** Current P2 diagnostics support continuous MRC
   response, small but measurable modality-specific neighbor allocation, and
   nontrivial functional gate effects. They do not support relation
   reliability, topology rewiring, gate causality, or M1 lambda as gate
   ground truth. See `docs/final/FIGURE3_FINAL_CLAIMS.md`.

4. **Figure 4 claims.** The safe interpretation is modality-/dataset-specific
   order profiles and computationally active cross-order integration. The
   interaction-off intervention is larger than uniform-attention in all five
   NC datasets, but this is not a causal or universal accuracy claim. See
   `docs/final/FIGURE4_FINAL_CLAIMS.md`.

5. **Table 1.** The current Full P2 row is competitive and is ranked best among
   the compared rows on both metrics in the generated descriptive table. The
   MGSC-MAG margin over the strongest historical baseline ranges from +0.39 to
   +0.95 percentage points in Accuracy and +0.48 to +1.31 points in Macro-F1.
   Historical baseline provenance remains a limitation; no universal SOTA
   claim is made.

6. **Table 2.** The strict plain multi-order control is close to Full, with
   Full better in 14/15 paired comparisons for both metrics. Raw Terminal is
   sometimes higher than Full, so the table explicitly prevents a universal
   necessity claim. Attribute Only is retained as a sanity control and not
   treated as a pure innovation ablation.

7. **Appendix ablation.** Table A1 reports paired deltas for Uniform Relations,
   Global Context Gate, Uniform Integration, and No Cross-Order Interaction.
   The signs are mixed and dataset-dependent; no module ranking is reported.
   Fixed Gate 0.9 is not reported because no formal final-branch result was
   found.

8. **Efficiency.** Matched MGSC and plain-control training runtime and peak
   GPU memory are exported. DiP and GraphSAGE have reliable instantiated
   parameter counts only; time and memory are left blank because no matched
   final-branch run exists.

9. **Terminology.** The frozen vocabulary and prohibited interpretations are
   recorded in `docs/final/PAPER_TERMINOLOGY.md`.

10. **Contributions.** The paper uses exactly three contributions: the
    empirical/problem perspective, the unified two-stage framework, and the
    evidence-oriented five-NC evaluation. They are recorded in
    `docs/final/PAPER_CONTRIBUTIONS.md`.

11. **Blueprint.** The complete section-by-section outline and code-faithful
    canonical P2 equations are in `docs/final/PAPER_V3_BLUEPRINT.md`.

12. **Remaining TODOs.** Before submission, rerun or behavior-equivalence
    certify the historical baseline table under the final fixed Macro-F1
    evaluator, record a producing execution SHA if recoverable, and replace
    preview plots with the journal's final typography. No new architecture
    module is required.

13. **Evidence inconsistency.** The main residual inconsistency is provenance:
    Table 1 historical baselines come from the existing paper table, Table 2
    Attribute Only comes from the formal functional-ablation matrix, while the
    corrected-control Full/plain/raw rows come from the corrected matrix.
    This is exposed in the machine-readable `source` fields and should remain
    explicit in the manuscript. Frozen inference interventions are never
    described as retrained performance.
