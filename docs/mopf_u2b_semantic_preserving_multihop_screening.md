# MoPF-vNext U2-B — Semantic-Preserving Distinctive Multi-Hop Propagation

Frozen mechanism screening only. No optimizer step, checkpoint write, formal config change, LP change, or U2-C implementation was performed.

- Source: `/hdd1/DataInHere/YHF/MoPF/outputs/u1t_semantic_conductance_calibration/u1t_master_summary.json`; 15 final U1-T T2 best-validation checkpoints
- Frozen relation: `learned_diag_cos`, `tau=0.35`
- Variants: B0 current cumulative, B1 hop innovation, B2 semantic anchor (`alpha=0.1`), B3 anchor + innovation
- Stress: `K=1..6`; analysis-only, formal training K unchanged

## Scientific interpretation

U2-B separates semantic preservation from hop distinctiveness. B1 tests innovation coordinates alone; B2 tests fixed semantic anchoring alone; B3 tests their combination. Semantic retention is a mechanism diagnostic, not a performance claim.

## Stress summary

Across K=1..6, B0's cumulative-state semantic retention and response redundancy are reported alongside B1/B2/B3. The authoritative tables retain per-dataset/per-modality/per-seed values; no stress depth was used to modify formal training. At K=6, B0 high-hop cosine mean was 0.9994421415197671, while B3(alpha=0.1) was 0.6526731781355977; B0 final novelty mean was 0.004439941709337388, while B3 was 0.112753260247445.

## Factorial interpretation

The JSON reports anchor main effects, innovation main effects, and anchor×innovation interaction for semantic retention, redundancy, novelty, and response magnitude at each dataset's formal K. These are mechanism effect sizes, not task-performance effects.

## Decision

**A. Proceed to U2-C with B3**

The language is restricted to mitigating propagation-induced semantic dilution, alleviating multi-hop response redundancy, preserving modality-specific semantics, and maintaining distinctive structural evidence. It does not claim to solve oversmoothing or oversquashing.

## Fallback and pending work

Orthogonal response construction remains a fallback path only if B3 retains high redundancy, low novelty, or vanishing higher-hop innovation. The pending relation-level 2×2 attribution (`separate_cos/tau2`, `learned_diag_cos/tau2`, `separate_cos/tau0.35`, `learned_diag_cos/tau0.35`) remains registered and was not run.

Authoritative summary: `/hdd1/DataInHere/YHF/MoPF/outputs/u2b_semantic_preserving_multihop_screening/u2b_master_summary.json`
