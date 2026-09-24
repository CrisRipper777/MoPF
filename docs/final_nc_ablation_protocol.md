# Final NC Ablation Protocol (Pre-registered)

Status: frozen before formal training. This protocol is descriptive and uses the same unified full-graph NC task as the canonical model.

Matrix: 7 ablations x 5 datasets (`Movies`, `Toys`, `Grocery`, `ele-fashion`, `Reddit-S`) x seeds (`42`, `43`, `44`) = 105 NC jobs. Selection uses validation Accuracy only. Test metrics are final descriptive metrics and never control model selection. No LP jobs are part of this matrix.

All ablations use the same hidden dimension, initialization policy, optimizer/task configuration, split, NC classifier protocol, and late-fusion interface as Full.

| Claim | Ablation | Exact definition | Expected diagnostic change | Metrics | Interpretation boundary |
|---|---|---|---|---|---|
| Relation modulation contributes to context formation | `no_relation_modulation` | `w=1` on the unchanged physical support and `c=0`; all later stages remain active | normalized operator perturbation and relation-state bridge vanish | validation Accuracy/Macro-F1; test descriptive only; operator metrics | isolates relation modulation, not the value of the graph itself |
| Adaptive semantic reference matters | `fixed_semantic_reference` | `alpha_ik=0.1` for every node and hop; propagation and Stage II remain active | alpha dispersion and learned R2 parameter effects vanish | validation Accuracy/Macro-F1 | does not remove propagation or retained contexts |
| Retained multi-hop bank is used | `last_context_only` (Terminal Context Only) | composite baseline: only `S_K` enters final composition; cross-hop interaction is off; multi-hop-centered reference residual is absent; earlier retained contexts do not directly enter output | terminal-context representation replaces full retained-context utilization | validation Accuracy/Macro-F1 | not a single-variable causal ablation; asks whether full retained-context utilization improves over terminal context only |
| Context change encoding matters | `no_context_change` | `D` injection is exactly zero; order embedding, attention, and all other paths remain | context-change injection and its gate contribution vanish | validation Accuracy/Macro-F1 | order identity and cross-hop interaction remain |
| Cross-hop interaction matters | `no_cross_hop_interaction` | `S_tilde=S`; signed filtering remains active | attention injection and interaction-off sensitivity vanish | validation Accuracy/Macro-F1 | isolates utilization interaction, not all Stage-II filtering |
| Global signed prior is sufficient | `global_filter_only` | `eta=gamma+DeltaGamma`; content/reference/relation residuals are zero; coefficients remain signed and unnormalized | node/context-dependent filter variability vanishes | validation Accuracy/Macro-F1 | does not mean signed filtering is removed |
| Stage-I formation state conditions Stage-II filtering | `no_formation_conditioning` | `delta_ref=delta_rel=0`; content, gamma, and DeltaGamma remain | reference/relation filter residuals vanish while content remains | validation Accuracy/Macro-F1 | isolates explicit cross-stage conditioning only |

Protocol amendment made before any formal ablation training: the matrix now includes Full plus the seven pre-registered variants, 8 x 5 x 3 = 120 NC jobs. Historical P1.8 U is equivalence/development reference only and is not a P2 model-selection input.
