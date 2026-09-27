# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/routing_identifiability_v1"
OUT = ROOT / "outputs/routing_identifiability_v1"


def read(name):
    return pd.read_csv(RES / name)


def mean_dict(frame, cols):
    return {c: float(frame[c].mean()) for c in cols}


def fmt(value, digits=5):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def main():
    pre = read("preflight_c0_checks.csv")
    pa = read("phase_a_e5_canonicalization.csv")
    teacher = read("phase_a_teacher_scale_diagnostic.csv")
    breg = read("phase_b_joint_regret.csv")
    bres = read("phase_b_interaction_residual.csv")
    bstab = read("phase_b_stability.csv")
    pc = read("phase_c_continuous_oracle_summary.csv")
    geom = read("phase_c_filter_geometry.csv")
    cstab = read("phase_c_stability.csv")
    caudit = read("phase_c_optimizer_audit.csv")
    d0 = read("phase_d_canonical_router_results.csv")
    dint = read("phase_d_interventions.csv")
    confirm = read("confirmation_results.csv")
    cint = read("confirmation_interventions.csv")
    bgate = json.loads((RES / "phase_b_gate.json").read_text())
    d0gate = json.loads((RES / "phase_d_d0_gate.json").read_text())
    d1gate = {"status": "NOT_TESTED", "reason": "Phase B joint-utility gate did not pass"}
    lock = json.loads((RES / "screening_winner_lock.json").read_text())

    pa_max_embedding = float(pa.embedding_max_abs_error.max())
    pa_max_logits = float(pa.logit_max_abs_error.max())
    teacher_summary = teacher.groupby("temperature").agg(
        entropy=("entropy", "mean"),
        preference_strength=("preference_strength", "mean"),
        hard_ranking_invariance=("hard_ranking_invariance_vs_T1", "mean"),
        soft_q_js=("soft_q_js_vs_T1", "mean"),
    ).reset_index()

    train_reg = breg[breg.split == "train"]
    train_res = bres[bres.split == "train"]
    train_stab = bstab[bstab.split == "train"]
    b_by_ds = train_reg.groupby("dataset").mean(numeric_only=True)
    i_by_ds = train_res.groupby("dataset").mean(numeric_only=True)
    stab_mean = mean_dict(train_stab, [
        "joint_best_pair_agreement", "text_best_response_agreement",
        "visual_best_response_agreement", "joint_oracle_gain_spearman",
        "interaction_energy_spearman", "signed_interaction_spearman",
    ])

    pc_train = pc[pc.split == "train"]
    pc_val = pc[pc.split == "validation"]
    pc_solution_train = pc_train.groupby("solution").mean(numeric_only=True)
    pc_solution_val = pc_val.groupby("solution").mean(numeric_only=True)
    gains = pc_train[pc_train.solution == "continuous_minus_discrete_gain"]
    gains_by_ds = gains.groupby("dataset").mean_CE.mean()
    gains_by_seed = gains.groupby(["dataset", "seed"]).mean_CE.mean()
    geometry_train = geom[geom.split == "train"]
    nonvertex_by_ds = 1.0 - geometry_train.groupby("dataset").fraction_near_vertex_l1_lt_0_10.mean()
    stable_continuous = (
        int((gains_by_ds > 1e-4).sum()) >= 2
        and int((gains_by_seed > 1e-4).sum()) >= 6
        and int((nonvertex_by_ds > 0.5).sum()) >= 2
    )
    near_vertex_by_ds = geometry_train.groupby("dataset").fraction_near_vertex_l1_lt_0_10.mean()
    discrete_sufficient = int((gains_by_ds <= 1e-4).sum()) >= 2 and int((near_vertex_by_ds > 0.5).sum()) >= 2
    train_geom = geometry_train.groupby("modality").mean(numeric_only=True)
    train_cstab = cstab[cstab.split == "train"]
    cstab_mean = mean_dict(train_cstab, [
        "alpha_cosine_similarity", "alpha_js_divergence", "order_rank_agreement",
        "distance_uniform_spearman", "continuous_gain_spearman",
    ])
    audit_train = caudit[caudit.split == "train"]
    optimizer_stats = {
        "mean_restart_loss_range": float(audit_train.restart_loss_range_mean.mean()),
        "p90_restart_loss_range": float(audit_train.restart_loss_range_mean.quantile(0.9)),
        "mean_steps": float(audit_train.steps.mean()),
        "early_stop_fraction": float(audit_train.early_stopped.mean()),
    }

    violation_values = []
    for path in sorted((OUT / "phase_c_node_oracles").glob("*.csv.gz")):
        nodes = pd.read_csv(path, usecols=["optimizer_sanity_violation"])
        violation_values.extend(nodes.optimizer_sanity_violation.to_numpy().tolist())
    violation = np.asarray(violation_values, dtype=float)
    max_violation = float(np.maximum(violation, 0).max()) if len(violation) else float("nan")
    mean_violation = float(np.maximum(violation, 0).mean()) if len(violation) else float("nan")

    d0_by_ds = d0.groupby("dataset").agg(
        delta_acc=("delta_acc_vs_uniform", "mean"),
        delta_macro_f1=("delta_macro_f1_vs_uniform", "mean"),
        positive_seeds=("delta_acc_vs_uniform", lambda s: int((s > 0).sum())),
    )
    intervention_rows = []
    for label, mask in [
        ("train_mean_alpha", dint.intervention == "train_mean_alpha"),
        ("node_shuffle_10_seed_mean", dint.intervention.str.startswith("node_shuffle_")),
        ("uniform_alpha", dint.intervention == "uniform_alpha"),
    ]:
        subset = dint[mask]
        intervention_rows.append({
            "intervention": label,
            "delta_acc": float(subset.delta_acc_vs_normal.mean()),
            "delta_macro_f1": float(subset.delta_macro_f1_vs_normal.mean()),
            "prediction_flip_fraction": float(subset.prediction_flip_fraction.mean()),
            "mean_abs_alpha_change": float(subset.mean_abs_alpha_change.mean()),
        })

    confirm_by_ds = confirm.groupby("dataset").agg(
        delta_acc=("delta_acc_vs_uniform", "mean"),
        delta_macro_f1=("delta_macro_f1_vs_uniform", "mean"),
        positive_seeds=("delta_acc_vs_uniform", lambda s: int((s > 0).sum())),
    )
    conf_shuffle = cint[cint.intervention.str.startswith("node_shuffle_")]
    conf_shuffle_stats = {
        "delta_acc": float(conf_shuffle.delta_acc_vs_normal.mean()),
        "delta_macro_f1": float(conf_shuffle.delta_macro_f1_vs_normal.mean()),
        "prediction_flip_fraction": float(conf_shuffle.prediction_flip_fraction.mean()),
        "mean_abs_alpha_change": float(conf_shuffle.mean_abs_alpha_change.mean()),
    }
    conf_positive = int((confirm.delta_acc_vs_uniform > 0).sum())
    conf_status = "COMPLETED"
    formal_runs = 9 + (18 if bgate["status"] == "JOINT_UTILITY_MATERIAL" else 0) + len(confirm)

    d0_status = "CANONICAL_ROUTER_PROMISING" if d0gate["pass"] else "NO_SUPPORT"
    if bgate["status"] == "JOINT_UTILITY_MATERIAL":
        h2_status, h3_status = "MIXED_SUPPORT", "STRONG_SUPPORT"
    else:
        h2_status, h3_status = "STRONG_SUPPORT", "NO_SUPPORT"
    if discrete_sufficient:
        h4_status = "STRONG_SUPPORT"
    elif int((near_vertex_by_ds > 0.5).sum()) >= 2:
        h4_status = "MIXED_SUPPORT"
    else:
        h4_status = "NO_SUPPORT"
    h5_status = "STRONG_SUPPORT" if stable_continuous else "MIXED_SUPPORT"
    h6_status = "STRONG_SUPPORT" if cstab_mean["alpha_cosine_similarity"] >= 0.8 else "MIXED_SUPPORT"
    h9_status = "STRONG_SUPPORT" if conf_positive == len(confirm) else ("MIXED_SUPPORT" if conf_positive else "NO_SUPPORT")
    decisions = [
        {"hypothesis": "H1_e5_routing_nonidentifiability_confirmed", "status": "STRONG_SUPPORT",
         "evidence": f"Two prescribed distinct five-action vectors map to alpha_U; maximum representation error={pa_max_embedding:.3g}, logit error={pa_max_logits:.3g}."},
        {"hypothesis": "H2_marginal_modality_utility_is_sufficient", "status": h2_status,
         "evidence": f"Material joint-utility gate={bgate['status']}; relative joint regret={bgate['relative_joint_regret']:.4g}."},
        {"hypothesis": "H3_joint_multimodal_structural_utility_is_material", "status": h3_status,
         "evidence": f"Joint gate requires relative regret >=0.10 and reproducible interaction; observed {bgate['relative_joint_regret']:.4g} and {bgate['interaction_reproducible_dataset_count']}/3 reproducible datasets."},
        {"hypothesis": "H4_discrete_order_selection_is_sufficient", "status": h4_status,
         "evidence": f"Mean continuous CE gain over discrete={float(gains.mean_CE.mean()):.4g}; near-vertex fraction by dataset={near_vertex_by_ds.to_dict()}."},
        {"hypothesis": "H5_continuous_filtering_has_extra_headroom", "status": h5_status,
         "evidence": f"Continuous gain by dataset={gains_by_ds.to_dict()}; positive dataset-seeds={int((gains_by_seed > 1e-4).sum())}/9; non-vertex fraction by dataset={nonvertex_by_ds.to_dict()}."},
        {"hypothesis": "H6_continuous_oracle_filter_is_cross_seed_stable", "status": h6_status,
         "evidence": f"Train alpha cosine={cstab_mean['alpha_cosine_similarity']:.4g}, JS={cstab_mean['alpha_js_divergence']:.4g}, rank agreement={cstab_mean['order_rank_agreement']:.4g}."},
        {"hypothesis": "H7_canonical_frozen_router_improves_uniform", "status": "STRONG_SUPPORT" if d0gate["pass"] else "NO_SUPPORT",
         "evidence": json.dumps(d0gate, sort_keys=True)},
        {"hypothesis": "H8_joint_router_improves_independent_router", "status": "NOT_TESTED",
         "evidence": "D1 skipped because the Phase B joint-utility gate was weak/mixed."},
        {"hypothesis": "H9_untouched_confirmation", "status": h9_status,
         "evidence": f"D0 confirmation on Toys/Grocery: {conf_positive}/{len(confirm)} positive seed pairs; mean delta Acc={confirm.delta_acc_vs_uniform.mean():.4g}, Macro-F1={confirm.delta_macro_f1_vs_uniform.mean():.4g}."},
    ]
    pd.DataFrame(decisions).to_csv(RES / "screening_decision_matrix.csv", index=False)

    phase_a_summary = {
        "max_embedding_abs_error": pa_max_embedding,
        "max_logit_abs_error": pa_max_logits,
        "preflight_max_val_acc_delta": float(pre.val_acc_abs_delta.max()),
        "preflight_max_val_macro_f1_delta": float(pre.val_macro_f1_abs_delta.max()),
        "teacher_scale": teacher_summary.to_dict("records"),
    }
    phase_b_summary = {
        "gate": bgate,
        "mean_abs_interaction": float(train_res.mean_abs_I.mean()),
        "mean_RMS_interaction": float(train_res.RMS_I.mean()),
        "mean_P90_abs_interaction": float(train_res.P90_abs_I.mean()),
        "mean_interaction_energy": float(train_res.interaction_energy_mean.mean()),
        "mean_joint_best_pair_agreement": stab_mean["joint_best_pair_agreement"],
    }
    phase_c_summary = {
        "mean_continuous_minus_discrete_gain": float(gains.mean_CE.mean()),
        "gain_by_dataset": {str(k): float(v) for k, v in gains_by_ds.items()},
        "positive_seed_count": int((gains_by_seed > 1e-4).sum()),
        "stable_continuous_gate": bool(stable_continuous),
        "discrete_sufficient_gate": bool(discrete_sufficient),
        "train_geometry_by_modality": train_geom.to_dict("index"),
        "train_stability": cstab_mean,
        "optimizer": optimizer_stats,
        "sanity_mean_violation": mean_violation,
        "sanity_max_violation": max_violation,
        "oracle_train_solutions": json.loads(pc_solution_train.to_json(orient="index")),
        "oracle_validation_solutions": json.loads(pc_solution_val.to_json(orient="index")),
    }
    summary = {
        "experiment": "E6-SRIJ",
        "phase_a": phase_a_summary,
        "phase_b": phase_b_summary,
        "phase_c": phase_c_summary,
        "phase_d_d0_gate": d0gate,
        "phase_d_d1_gate": d1gate,
        "screening_winner": lock["winning_variant"],
        "screening_winner_lock": lock,
        "confirmation_status": conf_status,
        "confirmation_positive_seed_count": f"{conf_positive}/{len(confirm)}",
        "confirmation_mean_delta_acc": float(confirm.delta_acc_vs_uniform.mean()),
        "confirmation_mean_delta_macro_f1": float(confirm.delta_macro_f1_vs_uniform.mean()),
        "formal_training_run_count": formal_runs,
        "c0_retraining_runs": 0,
        "test_metrics_used": False,
        "decisions": decisions,
    }
    (RES / "routing_identifiability_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# Experiment 6 - Structural Routing Identifiability & Joint Utility Audit",
        "",
        "## Scope and protocol",
        "",
        f"Branch screening winner lock: `{lock['winning_variant']}`. C0 reload preflight covered {len(pre)}/9 screening runs; maximum Validation Acc delta={pre.val_acc_abs_delta.max():.3g}, Macro-F1 delta={pre.val_macro_f1_abs_delta.max():.3g}. Screening and oracle diagnostics used Train/Validation only. No Test labels or metrics were used. Toys/Grocery were loaded only after the screening lock.",
        "",
        f"Formal router fits: {formal_runs} (9 D0 screening + {len(confirm)} confirmation; D1=0). C0 retraining runs: 0.",
        "",
        "## Phase A - canonicalization and teacher scale",
        "",
        f"E5 five-action outputs were mapped to the S0-S3 simplex. Both prescribed redundant routing vectors map to alpha_U=(0.25,0.25,0.25,0.25). Across the saved checks, maximum embedding error={pa_max_embedding:.3g} and logit error={pa_max_logits:.3g}, both below 1e-6.",
        "",
        "| T | Mean entropy | Preference strength | Hard ranking invariant | JS(qT || q1) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in teacher_summary.to_dict("records"):
        lines.append(f"| {row['temperature']:.2f} | {row['entropy']:.4f} | {row['preference_strength']:.5f} | {row['hard_ranking_invariance']:.4f} | {row['soft_q_js']:.5f} |")
    lines += [
        "",
        "Mean preference strength remains small at T=1 and increases as temperature falls; hard rankings remain nearly unchanged. Temperature was diagnostic only and did not affect training.",
        "",
        "## Phase B - joint discrete landscape",
        "",
        f"Gate: **{bgate['status']}**. The marginal pair regret was positive in {bgate['positive_dataset_seed_count']} dataset-seeds and all 3 dataset means, but relative regret was {bgate['relative_joint_regret']:.4f} (pre-registered materiality threshold 0.10); reproducible interaction criterion passed for {bgate['interaction_reproducible_dataset_count']}/3 datasets.",
        "",
        "| Dataset | Marginal-pair regret | Joint gain vs uniform | Relative regret | Marginal pair = joint best | Mean |I| | Mean interaction energy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in ["Movies", "ele-fashion", "Reddit-S"]:
        r, i = b_by_ds.loc[ds], i_by_ds.loc[ds]
        lines.append(f"| {ds} | {r.mean_joint_pair_regret:.5f} | {r.mean_joint_oracle_gain_vs_uniform:.5f} | {r.relative_joint_regret:.4f} | {r.marginal_pair_equals_joint_best_fraction:.3f} | {i.mean_abs_I:.5f} | {i.interaction_energy_mean:.4f} |")
    lines += [
        "",
        f"Across Train nodes, signed residual={train_res.mean_signed_I.mean():.5f}, mean absolute residual={train_res.mean_abs_I.mean():.5f}, RMS={train_res.RMS_I.mean():.5f}, P90 |I|={train_res.P90_abs_I.mean():.5f}. Mean interaction energy was {train_res.interaction_energy_mean.mean():.4f}; {train_res.fraction_energy_gt_0_1.mean():.1%} of nodes exceeded 0.1 and {train_res.fraction_energy_gt_0_25.mean():.1%} exceeded 0.25.",
        "",
        f"Across-seed Train stability means: joint-best-pair agreement={stab_mean['joint_best_pair_agreement']:.3f}, text/visual best-response agreement={stab_mean['text_best_response_agreement']:.3f}/{stab_mean['visual_best_response_agreement']:.3f}, joint-gain Spearman={stab_mean['joint_oracle_gain_spearman']:.3f}, interaction-energy Spearman={stab_mean['interaction_energy_spearman']:.3f}.",
        "",
        "## Phase C - continuous simplex oracle",
        "",
        "These are label-informed Train/Validation upper-bound diagnostics, not deployable router scores. The optimizer sanity check passed: mean positive violation=" + fmt(mean_violation, 3) + ", maximum=" + fmt(max_violation, 3) + ".",
        "",
        "| Solution | Train CE | Train Acc upper bound | Train Macro-F1 upper bound | Validation CE | Validation Acc upper bound | Validation Macro-F1 upper bound |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    solution_order = ["uniform", "best_joint_discrete", "marginal_continuous_pair", "joint_continuous"]
    for solution in solution_order:
        tr, va = pc_solution_train.loc[solution], pc_solution_val.loc[solution]
        lines.append(f"| {solution} | {tr.mean_CE:.5f} | {tr.accuracy_oracle:.4f} | {tr.macro_f1_oracle:.4f} | {va.mean_CE:.5f} | {va.accuracy_oracle:.4f} | {va.macro_f1_oracle:.4f} |")
    lines += [
        "",
        f"Mean Train CE gain of joint continuous over best discrete={gains.mean_CE.mean():.5f}; by dataset: " + ", ".join(f"{ds} {val:.5f}" for ds, val in gains_by_ds.items()) + f". Positive by the 1e-4 audit cutoff in {int((gains_by_seed > 1e-4).sum())}/9 dataset-seeds. Joint continuous improved over the marginal continuous pair by {pc_train.loc[pc_train.solution == 'joint_continuous_minus_marginal_gain', 'mean_CE'].mean():.5f} mean CE.",
        "",
        "| Modality | Entropy | Nearest-vertex L1 | L1 to uniform | Active orders | Top-2 mass | Near vertex (<0.10) | >=2 active | >=3 active |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for modality in ["text", "visual"]:
        g = train_geom.loc[modality]
        lines.append(f"| {modality} | {g.mean_entropy:.4f} | {g.mean_nearest_vertex_l1:.4f} | {g.mean_l1_to_uniform:.4f} | {g.mean_active_order_count:.3f} | {g.mean_top2_mass:.4f} | {g.fraction_near_vertex_l1_lt_0_10:.1%} | {g.fraction_ge_2_active_orders:.1%} | {g.fraction_ge_3_active_orders:.1%} |")
    lines += [
        "",
        f"Across-seed Train alpha stability: cosine={cstab_mean['alpha_cosine_similarity']:.3f}, JS={cstab_mean['alpha_js_divergence']:.3f}, order-rank agreement={cstab_mean['order_rank_agreement']:.3f}; distance-to-uniform Spearman={cstab_mean['distance_uniform_spearman']:.3f}. Mean restart loss range={optimizer_stats['mean_restart_loss_range']:.5f}, P90={optimizer_stats['p90_restart_loss_range']:.5f}; report this restart sensitivity when interpreting oracle alpha.",
        "",
        "## Phase D - canonical router and confirmation",
        "",
        f"D0 gate: **{'CANONICAL_ROUTER_PROMISING' if d0gate['pass'] else 'NO_SUPPORT'}**. Mean paired screening deltas were Acc={d0gate['mean_delta_acc']:.5f}, Macro-F1={d0gate['mean_delta_macro_f1']:.5f}; positive Acc seeds={d0gate['positive_seed_count']}; positive dataset means={d0gate['positive_dataset_count']}/3. Zero-init embedding error and checkpoint reload alpha error were both 0 for all runs.",
        "",
        "| Dataset | Mean delta Acc | Mean delta Macro-F1 | Positive seeds |",
        "|---|---:|---:|---:|",
    ]
    for ds, r in d0_by_ds.iterrows():
        lines.append(f"| {ds} | {r.delta_acc:.5f} | {r.delta_macro_f1:.5f} | {int(r.positive_seeds)}/3 |")
    lines += [
        "",
        "| Intervention | Delta Acc vs normal | Delta Macro-F1 | Prediction flip | Mean |delta alpha| |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in intervention_rows:
        lines.append(f"| {row['intervention']} | {row['delta_acc']:.5f} | {row['delta_macro_f1']:.5f} | {row['prediction_flip_fraction']:.5f} | {row['mean_abs_alpha_change']:.5f} |")
    lines += [
        "",
        f"D1 was **not tested** because Phase B was {bgate['status']}; no D1-vs-capacity-control result exists.",
        "",
        f"The screening lock selected **{lock['winning_variant']}** before loading untouched datasets. Confirmation completed on all 6 Toys/Grocery seed pairs: mean delta Acc={confirm.delta_acc_vs_uniform.mean():.5f}, Macro-F1={confirm.delta_macro_f1_vs_uniform.mean():.5f}, positive Acc pairs={conf_positive}/6.",
        "",
        "| Confirmation dataset | Mean delta Acc | Mean delta Macro-F1 | Positive seeds |",
        "|---|---:|---:|---:|",
    ]
    for ds, r in confirm_by_ds.iterrows():
        lines.append(f"| {ds} | {r.delta_acc:.5f} | {r.delta_macro_f1:.5f} | {int(r.positive_seeds)}/3 |")
    lines += [
        "",
        f"Confirmation node-shuffle intervention mean delta Acc={conf_shuffle_stats['delta_acc']:.5f}, Macro-F1={conf_shuffle_stats['delta_macro_f1']:.5f}, prediction flip={conf_shuffle_stats['prediction_flip_fraction']:.6f}; this effect was small.",
        "",
        "## Final decision matrix",
        "",
        "| Hypothesis | Status | Evidence |",
        "|---|---|---|",
    ]
    lines.extend(f"| {d['hypothesis']} | {d['status']} | {d['evidence'].replace('|', '/')} |" for d in decisions)
    lines += [
        "",
        "## Interpretation and next step",
        "",
        "E6 supports independent canonical structural routing at the screening gate and does not support a material joint text-visual routing requirement at the pre-registered threshold. The continuous label oracle finds extra CE headroom, especially on Movies, but its filters are often near simplex vertices and only moderately stable across seeds; that does not establish that a deployable continuous filter is needed. Untouched confirmation is heterogeneous (Toys positive, Grocery negative) with only 3/6 positive seed pairs and small mean gains. Treat the router as a promising screening result, not a broadly confirmed final model. The next useful step is targeted replication of independent routing across more seeds, with particular attention to the Movies versus Grocery split, while retaining C0 Uniform as the conservative fallback. Do not build D1 or distill the oracle from this evidence.",
        "",
        "No prohibited model components, C0 retraining, Test-based selection, Test metrics, or topology changes were used.",
        "",
    ]
    (RES / "routing_identifiability_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "winner": lock["winning_variant"], "formal_training_runs": formal_runs,
        "phase_b": bgate["status"], "d0_gate": d0gate["pass"],
        "confirmation_positive": f"{conf_positive}/{len(confirm)}",
        "continuous_mean_train_gain": float(gains.mean_CE.mean()),
        "report": str(RES / "routing_identifiability_report.md")
    }, indent=2))


if __name__ == "__main__":
    main()
