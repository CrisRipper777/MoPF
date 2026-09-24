# Final Architecture Evidence Matrix

Canonical final architecture: P1.8 variant U. P1.8 artifacts are the empirical motivation and frozen functional evidence; formal retrained paper ablations are pre-registered but not run in P1.9.

| Component | Empirical motivation | Mechanism behavior evidence | Frozen intervention evidence | Retrained ablation evidence | Boundary |
|---|---|---|---|---|---|
| R1 Semantic-Grounded Relation Modulation | P1.8 U learned nonzero semantic scores and direct weights | Text/Visual score non-degeneracy, weight CV, `c`, and normalized operator perturbation are in `p18_r1_comparison.csv` | relation-off changes fused embeddings with small but nonzero sensitivity | `no_relation_modulation` is pre-registered; 0 jobs run | supporting context-formation mechanism, not main performance driver |
| R2 Adaptive Semantic Reference | alpha moved from initialization without saturation | alpha dispersion, hop statistics, bias/rho parameters, and p statistics are in `p18_stage2_sanity.csv` | reference-off is an available canonical analysis intervention | `fixed_semantic_reference` is pre-registered; 0 jobs run | no universal cross-seed stability claim |
| Retained contexts | explicit `S0...SK` is part of final function | per-hop Q/S/D and effective-order diagnostics are exported | terminal/interaction interventions are frozen only where defined | `last_context_only` is pre-registered; 0 jobs run | no hidden early-context path in the ablation definition |
| Context-change encoding | P1.8 Stage-II diagnostics expose nonzero gate/injection | D norms and injection terms are exported | no retrained causal claim | `no_context_change` pre-registered; 0 jobs run | order embedding remains active |
| Cross-hop interaction | P1.8 attention is nonuniform | entropy, diagonal mass, node heterogeneity, and injection diagnostics | interaction-off produces exact `S_tilde=S` and material frozen sensitivity | `no_cross_hop_interaction` pre-registered; 0 jobs run | major functional utilization mechanism, not strong pairwise-hop reasoning |
| Signed filtering | hierarchical signed composition is part of U | negative eta fraction and reference/relation residuals are exported | frozen sensitivity is not causal necessity | global-only and no-formation-conditioning pre-registered; 0 jobs run | negative fraction is descriptive, not a quality target |
| Performance | U/B/V3 development results | P1.8 performance CSV and paired controls | frozen interventions use same NC head | formal 105-job matrix not run | test remains descriptive only |

Evidence categories are kept separate: empirical motivation, mechanism behavior, frozen intervention, retrained ablation, and task performance.
