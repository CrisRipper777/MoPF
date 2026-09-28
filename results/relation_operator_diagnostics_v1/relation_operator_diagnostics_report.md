# P0-D — Global Prior & Relation Personalization Diagnostics

## Scope and data integrity

This report analyzes only the 15 frozen A2 `global_expert` and 15 frozen A3 `relation_expert` checkpoints. It does not train or update models. Checkpoints are loaded with `strict=True` by the existing checkpoint loader. All reported task metrics index only Validation labels; fixed Macro-F1 labels come from Train and Validation. Train–Train edges define diagnostic quantile thresholds. No Test metrics or Test labels are evaluated.
A0 semantic quartiles use each frozen plain checkpoint's projected H0 and the existing canonical non-self physical edge ordering. A3 shrinkage λ=0 uses the global-mean route of that A3 checkpoint and must not be interpreted as a separately trained A2 result.
Matched random controls use 20 no-replacement samples per target context from all non-self physical edges excluding the target set. Sampling first matches the joint quartile of log source degree, log target degree, and personalization magnitude; it then relaxes to two dimensions, one dimension, and finally unrestricted matching when a stratum lacks enough candidates. Every relaxation level and distribution balance is recorded. Across 4800 matched repeats, 60 used relaxed strata; 15360/78724320 selected control edges (0.0195%) were relaxed. Mean absolute SMDs were source degree 0.0280, target degree 0.0311, and personalization magnitude 0.0199. Empirical percentiles are descriptive, not formal significance tests.
Structural response is `D_i = P_nbr H0_i - H0_i`, where `P_nbr` is the frozen A0 symmetric-normalized physical operator with its self-loop coordinates removed. Structural correlations and rank tests are descriptive; edge dependence prevents causal interpretation.

## D1 — Global operator necessity

ZERO caused positive mean ΔCE and positive mean margin utility in 22/45 modality-scope contexts. Decision: **GLOBAL_TRANSFORMATION_NOT_CONSISTENTLY_NECESSARY**.
Multiple-operator decision: **SINGLE_SHARED_OPERATOR_SUFFICIENT** (22/30 modality-contexts pass the declared close-to-normal rule). This is a descriptive discovery decision. See `d1_global_operator_interventions.csv` and node-level logits/utility in `d1_node_utility.csv.gz`.
Expert IDs are reported within Text or Visual modality and are not interpreted as cross-modality semantic matches. `best single` is a diagnostic oracle selected on Validation CE, not a deployable selection policy.

## D2 — Relation-specific personalization

At least one intermediate shrinkage λ had lower Validation CE than λ=1 in 41/45 dataset × seed × modality-scope contexts. Decision: **FULL_PERSONALIZATION_APPEARS_OVERSTRONG**.
D4 route-to-global removal improves mean CE in 10/30 target contexts and improves Validation Accuracy in 3 contexts. D4 target-vs-matched-random CE excess is positive in 18/30 population summaries. Q1 target CE excess is positive in 16/30 population summaries. Read target percentile, random spread, and message-normalized effects in `d2_random_control_summary.csv`; these values are descriptive rankings rather than significance tests.
Node utility is concentrated on directly touched nodes: across 240 target interventions, mean absolute ΔCE was 0.0478 for touched Validation nodes and 0.0271 across all Validation nodes; touched-node mean absolute utility was higher in 240/240 cases. Selected edges touched 58.6% of Validation nodes on average. Harmed fractions (ΔCE>0) were 46.8% touched vs 38.2% all, while improved fractions (ΔCE<0) were 39.9% vs 34.0%. See `d2_node_utility_concentration.csv` and the per-node raw file. When a matched control has no directly touched Validation nodes, the touched-control summary reports zero valid repeats and the empty count explicitly; all-node controls still retain all requested repeats.
Structural-response edge rows: 150 Spearman summaries; group and Q1 comparisons are in `d2_structural_context_descriptives.csv`. Decision: **DESCRIPTIVE_ONLY**. These results support descriptive assessment of structural context only.

## Explicit decisions

- **D1_A_global_transformation_necessity** — `GLOBAL_TRANSFORMATION_NOT_CONSISTENTLY_NECESSARY`. ZERO has mean ΔCE>0 and mean margin utility>0 in a majority of modality scopes; ΔCE=CE(intervention)-CE(normal).
- **D1_B_multiple_global_operators** — `SINGLE_SHARED_OPERATOR_SUFFICIENT`. A modality-context passes when diagnostic best single, TOP1, and UNIFORM each have mean ΔCE≤0.01 and best-single Validation Accuracy/Macro-F1 are within 0.01 of NORMAL. This is a descriptive discovery rule; expert IDs are modality-specific.
- **D2_Q1_full_personalization_strength** — `FULL_PERSONALIZATION_APPEARS_OVERSTRONG`. Counts modality-scope contexts where at least one intermediate λ has lower mean Validation CE than λ=1.
- **D2_Q2_strong_deviation_utility** — `DESCRIPTIVE_TARGET_VS_MATCHED_RANDOM`. Positive CE excess means removing personalization from D4 is more harmful than removing it from matched random edges; report matched-control percentile alongside this count.
- **D2_Q3_semantic_Q1_specialized_utility** — `DESCRIPTIVE_TARGET_VS_MATCHED_RANDOM`. Q1 route-to-global utility is compared against 20 degree- and deviation-matched control edge sets sampled from the full physical edge population, excluding the target group.
- **D2_Q4_node_utility_concentration** — `UTILITY_STRONGER_ON_DIRECTLY_TOUCHED_NODES`. Compares mean absolute node ΔCE for directly touched Validation nodes with all Validation nodes; this is a descriptive concentration check.
- **D2_Q5_structural_response_association** — `DESCRIPTIVE_ONLY`. Edge-level Spearman correlations and exploratory Kruskal–Wallis/Mann–Whitney summaries are descriptive; edges are dependent and these are not causal tests.

## Runtime

Elapsed analysis time: 3008.7 s. Peak allocated GPU memory recorded by PyTorch: 3042.6 MiB.

## Boundaries

No new training was performed. Test metrics and Test labels were not evaluated. P0-R, Stage II, architecture changes, and training computation changes were not implemented.
