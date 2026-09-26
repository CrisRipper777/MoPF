# Experiment 5 — Utility-Aligned Structural Routing Audit

- Branch: utility_routing_audit; HEAD: 12e52f948aab8944dc71eb342d091ef3266a53f3; starting commit: 12e52f948aab8944dc71eb342d091ef3266a53f3.
- NC only; screen results use Movies, ele-fashion, Reddit-S (seeds 42/43/44).
- No C0 retraining; all routing objectives use Train labels only. Validation is used for selection/audit; Test is not indexed.
- Toys and Grocery were not loaded until screening_winner_lock.json existed.
- Frozen winner: C0_Uniform (C0_uniform); safe routing=False.
- Phase B objective gate: UTILITY_ALIGNED_OBJECTIVE_NOT_SUPPORTED.

## C0 reuse and action landscape

See preflight_report.md for all 9 checkpoint reproduction checks. Raw action rows are gzip-compressed under outputs/utility_routing_audit_v1/action_landscape/.

Action best frequencies, preference entropy/strength, oracle CE gain and label-oracle accuracy upper bound are in action_landscape_summary.csv; cross-seed hard-action agreement, soft-q JS/cosine and preference-strength Spearman are in action_stability.csv.

The label-oracle accuracy is an upper bound and is not a deployable result.

## Phase B

| Variant | Dataset | Mean Val Acc | Mean Macro-F1 | Mean Routing Regret | Excess Loss vs AU |
|---|---|---:|---:|---:|---:|
| B0_frozen_direct_ce | Movies | 0.5765 | 0.4964 | 0.3319 | 0.0061 |
| B0_frozen_direct_ce | Reddit-S | 0.9636 | 0.9237 | 0.0489 | -0.0069 |
| B0_frozen_direct_ce | ele-fashion | 0.8774 | 0.7606 | 0.1005 | -0.0083 |
| B1_flat_utility_kd | Movies | 0.5759 | 0.4943 | 0.3284 | 0.0026 |
| B1_flat_utility_kd | Reddit-S | 0.9619 | 0.9221 | 0.0549 | -0.0009 |
| B1_flat_utility_kd | ele-fashion | 0.8760 | 0.7548 | 0.1078 | -0.0010 |
| B2_weighted_utility_kd | Movies | 0.5762 | 0.4946 | 0.3278 | 0.0020 |
| B2_weighted_utility_kd | Reddit-S | 0.9632 | 0.9225 | 0.0491 | -0.0067 |
| B2_weighted_utility_kd | ele-fashion | 0.8775 | 0.7613 | 0.1007 | -0.0081 |

Paired routing-regret and objective gates are in phase_b_routing_regret.csv and the screening decision matrix. Node shuffles, train-mean probabilities and AU fallback are in phase_b_interventions.csv.

B0 vs E4 historical comparison: {"status": "MIXED_SUPPORT", "positive_paired_seeds": "3/9", "mean_B0_minus_E4_acc_effect": 0.010695669386121986, "mean_B0_minus_E4_f1_effect": 0.03519528525502326, "note": "E4 B0 HierarchicalFlat vs C0 and E5 B0 frozen Direct CE vs C0, paired on the same screening dataset/seed; descriptive historical contrast."}.

Preference weighting gate: false.

## Phase C

NOT_TESTED because its predeclared Phase B gate did not pass.

## Phase D

NOT_TESTED because its predeclared Phase B gate did not pass.

## Phase E conservative routing

Selection/audit: {"selected_variant": "B2_weighted_utility_kd", "phase": "phase_b", "conservative_personalization_supported": false, "movies_negative_transfer_reduced": false, "positive_dataset_gains_retained": false, "selected_by": "highest screening mean Val Acc among utility-aligned candidates; regret tie-break"}. Per-run normal/safe/C0 values and regret are in phase_e_safe_routing.csv.

## Screening decision matrix

| Hypothesis | Status | Evidence |
|---|---|---|
| H1_action_landscape_nontrivial | MIXED_SUPPORT | {"mean_preference_strength": 0.02588419538612161, "mean_oracle_ce_gain": 0.16348185949027533, "mean_oracle_acc_gain": 0.044085005919138553, "diverse_action_count": 6.0} |
| H2_frozen_routing_reduces_coadaptation_problem | MIXED_SUPPORT | {"status": "MIXED_SUPPORT", "positive_paired_seeds": "3/9", "mean_B0_minus_E4_acc_effect": 0.010695669386121974, "mean_B0_minus_E4_f1_effect": 0.035195285255023266, "note": "E4 B0 HierarchicalFlat vs C0 and E5 B0 frozen Direct CE vs C0, paired on the same screening dataset/seed; descriptive historical contrast."} |
| H3_utility_distillation_improves_routing_alignment | MIXED_SUPPORT | {"status": "UTILITY_ALIGNED_OBJECTIVE_NOT_SUPPORTED", "supported": false, "candidate_gates": [{"candidate": "B1_flat_utility_kd", "datasets_lower_regret": 1, "positive_regret_seed_pairs": "3/9", "mean_delta_regret": 0.0032880219320456185, "mean_delta_acc": -0.0012137691179911296, "mean_delta_macro_f1": -0.0031787772511826765, "mean_delta_regret_by_dataset": {"Movies": -0.003515938917795817, "Reddit-S": 0.005998712033033371, "ele-fashion": 0.0073812926808993025}, "mean_delta_acc_by_dataset": {... |
| H4_preference_weighting_helps_selective_personalization | MIXED_SUPPORT | {"status": "UTILITY_ALIGNED_OBJECTIVE_NOT_SUPPORTED", "supported": false, "preference_weighting_promising": false, "B2_nonnegative_acc_datasets": 1, "Movies_delta_acc_B1": -0.0005999008814493815, "Movies_delta_acc_B2": -0.00029995044072469074} |
| H5_decision_response_improves_utility_routing | NOT_TESTED | Phase B gate failed. |
| H6_relation_context_benefits_from_utility_alignment | NOT_TESTED | Phase B gate failed; Phase D is conditional. |
| H7_conservative_personalization_reduces_negative_transfer | NO_SUPPORT | {"selected_variant": "B2_weighted_utility_kd", "phase": "phase_b", "conservative_personalization_supported": false, "movies_negative_transfer_reduced": false, "positive_dataset_gains_retained": false, "selected_by": "highest screening mean Val Acc among utility-aligned candidates; regret tie-break"} |
| H8_untouched_dataset_confirmation | NOT_TESTED | {"reason": "C0 Uniform was locked as incumbent; no learned router confirmation candidate."} |

## Five-dataset winner summary (after confirmation)

These are per-dataset Validation metrics for the locked winner; the cross-dataset aggregate did not affect winner selection.

| Dataset | Paired seeds | Mean Val Acc | Mean Macro-F1 |
|---|---:|---:|---:|
| Grocery | 3 | 0.8331 | 0.7459 |
| Movies | 3 | 0.5757 | 0.4941 |
| Reddit-S | 3 | 0.9612 | 0.9211 |
| Toys | 3 | 0.8077 | 0.7788 |
| ele-fashion | 3 | 0.8758 | 0.7542 |

## Untouched Toys/Grocery confirmation

The winner lock was written before any Toys/Grocery features, labels, checkpoint metrics, or predictions were loaded.

| Dataset | Paired seeds | Mean Δ Val Acc | Mean Δ Macro-F1 | Mean Δ Routing Regret | Interpretation |
|---|---:|---:|---:|---:|---|
| Grocery | 3 | 0.0000 | 0.0000 | N/A | NOT_TESTED: locked winner is C0 identity |
| Toys | 3 | 0.0000 | 0.0000 | N/A | NOT_TESTED: locked winner is C0 identity |

## Interpretation

Results distinguish frozen direct task routing, unweighted utility distillation, preference-weighted distillation, decision-response evidence, corrected physical-edge relation evidence, and conservative AU fallback. Conditional phases not run are marked NOT_TESTED in the decision matrix.

This audit does not implement MoE experts, signed filters, cross-modal routing, common/private decomposition, topology changes, or test-set evaluation.
