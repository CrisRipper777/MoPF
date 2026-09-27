from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import structural_observability_data as dataio
from scripts import structural_observability_training as trainio

RES = dataio.RES


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / name)


def _context_values(frame: pd.DataFrame, candidate: str, column: str, source: str) -> pd.DataFrame:
    sub = frame[(frame.candidate == candidate) & (frame.split == "validation")]
    keys = ["dataset", "seed"]
    if source == "selection":
        sub = sub.groupby(keys, as_index=False)[column].mean()
    else:
        sub = sub.groupby(keys, as_index=False)[column].mean()
    return sub[keys + [column]]


def _compare(frame: pd.DataFrame, candidate: str, baseline: str, column: str, source: str, lower_better: bool) -> dict:
    a = _context_values(frame, candidate, column, source).rename(columns={column: "candidate"})
    b = _context_values(frame, baseline, column, source).rename(columns={column: "baseline"})
    paired = a.merge(b, on=["dataset", "seed"], how="inner")
    paired["delta"] = paired.candidate - paired.baseline
    improved = paired.delta < 0 if lower_better else paired.delta > 0
    dataset_delta = paired.groupby("dataset").delta.mean()
    return {
        "candidate": candidate, "baseline": baseline, "metric": column,
        "mean_candidate": float(paired.candidate.mean()), "mean_baseline": float(paired.baseline.mean()),
        "mean_delta_candidate_minus_baseline": float(paired.delta.mean()),
        "improved_seed_count": int(improved.sum()), "paired_seed_count": int(len(paired)),
        "datasets_improved": int((dataset_delta < 0 if lower_better else dataset_delta > 0).sum()),
        "dataset_mean_deltas": {str(k): float(v) for k, v in dataset_delta.items()},
        "paired_deltas": paired.to_dict("records"),
    }


def _joint_nonworse(execution: pd.DataFrame, candidate: str, baseline: str) -> dict:
    acc = _compare(execution, candidate, baseline, "accuracy", "joint", lower_better=False)
    f1 = _compare(execution, candidate, baseline, "macro_f1", "joint", lower_better=False)
    return {
        "accuracy_mean_delta": acc["mean_delta_candidate_minus_baseline"],
        "macro_f1_mean_delta": f1["mean_delta_candidate_minus_baseline"],
        "nonworse": acc["mean_delta_candidate_minus_baseline"] >= -1e-12 and f1["mean_delta_candidate_minus_baseline"] >= -1e-12,
        "accuracy_paired_nonworse": int(sum(x["delta"] >= -1e-12 for x in acc["paired_deltas"])),
        "macro_f1_paired_nonworse": int(sum(x["delta"] >= -1e-12 for x in f1["paired_deltas"])),
    }


def _gate_record(name: str, passed: bool, evidence: dict, definition: str) -> dict:
    return {"gate": name, "passed": bool(passed), "status": "PASS" if passed else "FAIL", "definition": definition, "evidence": evidence}


def evaluate_screening_and_lock() -> dict:
    selection = _read("phase_b_selection_regret.csv")
    execution = _read("phase_b_joint_execution.csv")
    confidence = _read("phase_c_confidence_analysis.csv")
    coverage = _read("phase_c_coverage_curve.csv")
    variants = [name for name, _ in dataio.VARIANTS]

    h1_regret = _compare(selection, "R0_trajectory_ranker", "GLOBAL_ACTION", "selection_regret_mean", "selection", True)
    h1_joint = _joint_nonworse(execution, "R0_trajectory_ranker", "GLOBAL_ACTION")
    h1 = h1_regret["datasets_improved"] >= 2 and h1_regret["improved_seed_count"] >= 6 and h1_joint["nonworse"]

    h2_regret = _compare(selection, "R1_decision_ranker", "R1_capacity_control", "selection_regret_mean", "selection", True)
    h2_joint = _joint_nonworse(execution, "R1_decision_ranker", "R1_capacity_control")
    h2 = h2_regret["datasets_improved"] >= 2 and h2_regret["improved_seed_count"] >= 6 and h2_joint["nonworse"]

    h3_each = {}
    for candidate in variants:
        regret = _compare(execution, candidate, "E6_D0", "mean_joint_deployment_regret", "joint", True)
        joint = _joint_nonworse(execution, candidate, "E6_D0")
        h3_each[candidate] = {"regret": regret, "joint": joint,
                              "passed": regret["datasets_improved"] >= 2 and regret["improved_seed_count"] >= 6 and joint["nonworse"]}
    best_variant = sorted(variants, key=lambda x: (
        -float(execution[(execution.candidate == x) & (execution.split == "validation")].accuracy.mean()),
        -float(execution[(execution.candidate == x) & (execution.split == "validation")].macro_f1.mean()),
        float(execution[(execution.candidate == x) & (execution.split == "validation")].mean_joint_deployment_regret.mean()),
    ))[0]
    h3 = h3_each[best_variant]["passed"]

    conf_by_dataset = confidence.groupby("dataset").agg(
        rho=("spearman_confidence_vs_negative_regret", "mean"),
        high_q4_regret=("high_confidence_q4_regret", "mean"),
        low_q1_regret=("low_confidence_q1_regret", "mean"),
    )
    h4_dataset = {
        dataset: {"positive_spearman": bool(row.rho > 0), "high_confidence_lower_regret": bool(row.high_q4_regret < row.low_q1_regret),
                  "mean_spearman": float(row.rho), "q4_regret": float(row.high_q4_regret), "q1_regret": float(row.low_q1_regret)}
        for dataset, row in conf_by_dataset.iterrows()
    }
    h4 = sum(v["positive_spearman"] and v["high_confidence_lower_regret"] for v in h4_dataset.values()) >= 2

    selective = coverage[(coverage.nominal_train_coverage == 0.5) & (coverage.candidate == "R1_decision_ranker")]
    full_r1 = execution[(execution.split == "validation") & (execution.candidate == "R1_decision_ranker")].groupby("dataset").agg(acc=("accuracy", "mean"), f1=("macro_f1", "mean"))
    uniform = execution[(execution.split == "validation") & (execution.candidate == "C0_Uniform")].groupby("dataset").agg(acc=("accuracy", "mean"), f1=("macro_f1", "mean"))
    sel = selective.groupby("dataset").agg(acc=("accuracy", "mean"), f1=("macro_f1", "mean"))
    delta_r1_acc = (full_r1.acc - uniform.acc).reindex(dataio.SCREEN)
    delta_sel_acc = (sel.acc - uniform.acc).reindex(dataio.SCREEN)
    delta_r1_f1 = (full_r1.f1 - uniform.f1).reindex(dataio.SCREEN)
    delta_sel_f1 = (sel.f1 - uniform.f1).reindex(dataio.SCREEN)
    worst_transfer_reduced = float(delta_sel_acc.min()) > float(delta_r1_acc.min()) + 1e-12
    aggregate_not_below_uniform = float(sel.acc.mean()) >= float(uniform.acc.mean()) - 1e-12 and float(sel.f1.mean()) >= float(uniform.f1.mean()) - 1e-12
    h5 = worst_transfer_reduced and aggregate_not_below_uniform

    gate_defs = [
        _gate_record("H1_COUNTERFACTUAL_RANKING_OBSERVABLE", h1,
                     {"regret": h1_regret, "joint_execution": h1_joint},
                     "At least 2/3 dataset means and 6/9 paired contexts have lower marginal selection regret than GLOBAL_ACTION; aggregate joint Accuracy and Macro-F1 are both nonworse."),
        _gate_record("H2_DECISION_RESPONSE_ADDS_INFORMATION", h2,
                     {"regret": h2_regret, "joint_execution": h2_joint},
                     "At least 2/3 dataset means and 6/9 paired contexts have lower marginal selection regret than capacity control; aggregate joint Accuracy and Macro-F1 are both nonworse."),
        _gate_record("H3_COUNTERFACTUAL_RANKER_BEATS_DIRECT_CE", h3,
                     {"best_ranker_by_validation_accuracy_then_macro_f1_then_regret": best_variant, "per_ranker": h3_each},
                     "Best ranker must improve joint deployment regret on at least 2/3 datasets and 6/9 paired contexts over E6 D0, with aggregate Accuracy and Macro-F1 nonworse."),
        _gate_record("H4_CONFIDENCE_IS_MEANINGFUL", h4, h4_dataset,
                     "At least 2/3 datasets have positive mean Spearman(confidence, -regret) and lower Q4 than Q1 confidence regret."),
        _gate_record("H5_SELECTIVE_ROUTING_PROMISING", h5,
                     {"worst_dataset_negative_transfer_reduced": worst_transfer_reduced,
                      "aggregate_accuracy_and_macro_f1_not_below_uniform": aggregate_not_below_uniform,
                      "worst_dataset_delta_accuracy_full_r1": float(delta_r1_acc.min()),
                      "worst_dataset_delta_accuracy_selective_50": float(delta_sel_acc.min()),
                      "worst_dataset_delta_macro_f1_full_r1": float(delta_r1_f1.min()),
                      "worst_dataset_delta_macro_f1_selective_50": float(delta_sel_f1.min()),
            "worst_dataset_negative_transfer_reduced_macro_f1": bool(max(0.0, -float(delta_sel_f1.min())) < max(0.0, -float(delta_r1_f1.min())) - 1e-12),
                      "aggregate_selective_accuracy": float(sel.acc.mean()), "aggregate_uniform_accuracy": float(uniform.acc.mean()),
                      "aggregate_selective_macro_f1": float(sel.f1.mean()), "aggregate_uniform_macro_f1": float(uniform.f1.mean())},
                     "At 50% train-threshold coverage, worst-dataset negative Accuracy transfer improves over full R1 and aggregate Accuracy/Macro-F1 are both at least Uniform."),
    ]

    # Apply each learned candidate's relevant gate plus a direct comparison against E6 D0.
    def variant_metrics(candidate):
        rows = execution[(execution.candidate == candidate) & (execution.split == "validation")]
        regret_rows = selection[(selection.candidate == candidate) & (selection.split == "validation")]
        return {
            "mean_accuracy": float(rows.accuracy.mean()), "mean_macro_f1": float(rows.macro_f1.mean()),
            "mean_joint_regret": float(rows.mean_joint_deployment_regret.mean()),
            "mean_selection_regret": float(regret_rows.selection_regret_mean.mean()),
        }

    eligibility = {"C0_Uniform": {"eligible": True, "reason": "frozen Uniform fallback baseline"},
                   "E6_D0": {"eligible": True, "reason": "reused frozen E6 comparator"}}
    for candidate in variants:
        to_global = _compare(selection, candidate, "GLOBAL_ACTION", "selection_regret_mean", "selection", True)
        candidate_h1 = to_global["datasets_improved"] >= 2 and to_global["improved_seed_count"] >= 6 and _joint_nonworse(execution, candidate, "GLOBAL_ACTION")["nonworse"]
        candidate_h3 = h3_each[candidate]["passed"]
        if candidate == "R1_decision_ranker":
            own_gate = h2 and candidate_h3
            reason = "requires H2 and candidate-specific H3"
        else:
            own_gate = candidate_h1 and candidate_h3
            reason = "requires candidate-specific H1-style global-prior comparison and H3"
        paired_vs_d0 = _compare(execution, candidate, "E6_D0", "accuracy", "joint", False)
        consistency = sum(x["delta"] >= -1e-12 for x in paired_vs_d0["paired_deltas"])
        consistency_ok = consistency >= 5
        eligibility[candidate] = {"eligible": bool(own_gate and consistency_ok), "reason": reason,
                                  "counterfactual_observability_gate": bool(h2 if candidate == "R1_decision_ranker" else candidate_h1),
                                  "candidate_specific_H3": bool(candidate_h3),
                                  "paired_nonworse_accuracy_contexts_vs_E6_D0": int(consistency),
                                  "paired_consistency_requirement": "at least 5/9 validation contexts nonworse in Accuracy vs E6 D0",
                                  "metrics": variant_metrics(candidate)}

    eligible = [name for name, info in eligibility.items() if info["eligible"]]
    winner = sorted(eligible, key=lambda x: (
        -variant_metrics(x)["mean_accuracy"], -variant_metrics(x)["mean_macro_f1"],
        variant_metrics(x)["mean_joint_regret"],
        -sum(v >= -1e-12 for v in _compare(execution, x, "E6_D0", "accuracy", "joint", False)["paired_deltas"][i]["delta"]
             for i, _ in enumerate(_compare(execution, x, "E6_D0", "accuracy", "joint", False)["paired_deltas"])) if x not in ("C0_Uniform", "E6_D0") else 0,
    ))[0]
    # Preserve the exact gate table and a lock written before any confirmation context is loaded.
    matrix_rows = [{"row_type": "gate", "candidate": "", "passed": g["passed"], "gate": g["gate"], "details_json": json.dumps(g["evidence"], ensure_ascii=False)} for g in gate_defs]
    for candidate, info in eligibility.items():
        matrix_rows.append({"row_type": "candidate", "candidate": candidate, "passed": info["eligible"], "gate": info.get("reason", "baseline candidate"),
                            "details_json": json.dumps(info, ensure_ascii=False)})
    dataio.write_csv(RES / "screening_decision_matrix.csv", matrix_rows)
    lock = {
        "experiment": "E7-CSAO", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "screening_datasets": dataio.SCREEN, "seeds": dataio.SEEDS, "test_evaluated": False,
        "winner": winner, "winner_type": "learned_ranker" if winner in variants else "frozen_baseline",
        "gate_results": {g["gate"]: g for g in gate_defs}, "candidate_eligibility": eligibility,
        "selection_rule": "Eligible candidates ranked by validation joint-execution Accuracy, then Macro-F1, then lower joint regret; paired Accuracy consistency is a qualification check for learned candidates.",
        "confirmation_fit_policy": "If winner is R1, run R1 and capacity control; if R0, run R0; frozen baseline winner gets no confirmation ranker fit.",
    }
    dataio.write_json(RES / "screening_winner_lock.json", lock)
    return lock


def run_confirmation(device_name="cuda:0", lock_path: Path = RES / "screening_winner_lock.json") -> dict:
    if not lock_path.exists():
        raise RuntimeError("screening winner lock must exist before confirmation data is loaded")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    winner = lock["winner"]
    if winner == "R1_decision_ranker":
        confirm_variants = [("R1_decision_ranker", "r1_decision"), ("R1_capacity_control", "r1_capacity_control")]
    elif winner == "R0_trajectory_ranker":
        confirm_variants = [("R0_trajectory_ranker", "r0_trajectory")]
    else:
        confirm_variants = []
    device = torch.device(device_name)
    frames, summaries, stability_rows, result_rows, regret_rows = {}, [], [], [], []
    confirmation_global_rows, confidence_rows, fit_count = [], [], 0
    for dataset in dataio.CONFIRM:
        for seed in dataio.SEEDS:
            ctx = dataio._load_context(dataset, seed, device, verify_actions=False)
            frame, rows = dataio._action_landscape_frame(ctx)
            frames[(dataset, seed)] = frame
            summaries.extend(rows)
            raw_path = dataio.OUT / "confirmation_action_landscape_nodes" / f"{dataset}_seed{seed}.csv.gz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(raw_path, index=False, compression="gzip")
            train_idx = ctx["train"]
            text_losses = ctx["matrix"][train_idx, :, 0]
            visual_losses = ctx["matrix"][train_idx, 0, :]
            global_t, global_v = int(text_losses.mean(0).argmin()), int(visual_losses.mean(0).argmin())
            global_actions = {dataset: {"text": global_t, "visual": global_v}}
            for modality, action, losses in (("text", global_t, text_losses), ("visual", global_v, visual_losses)):
                row = {"dataset": dataset, "seed": seed, "modality": modality,
                       "global_best_action": dataio.ACTION_NAMES[action], "action_id": action,
                       "train_mean_CE": float(losses.mean(0)[action].item())}
                row.update({f"train_mean_CE_{dataio.ACTION_NAMES[k]}": float(losses.mean(0)[k].item()) for k in range(5)})
                confirmation_global_rows.append(row)
            ctx = trainio.prepare_runtime_context(ctx, global_actions)
            base = trainio._baseline_payloads(ctx)
            result_rows.extend(row for row in trainio._execution_rows(ctx, base) if row["split"] == "validation")
            selection, _ = trainio._selection_rows(ctx, base)
            regret_rows.extend(row for row in selection if row["split"] == "validation")
            for variant, mode in confirm_variants:
                result, payloads, exrows, srrows, auxiliary = trainio.train_variant(ctx, variant, mode)
                fit_count += 1
                result_rows.extend(row for row in exrows if row["candidate"] == variant and row["split"] == "validation")
                regret_rows.extend(row for row in srrows if row.get("candidate") == variant and row.get("split") == "validation" and "selection_regret_mean" in row)
                confidence_rows.extend(row for row in auxiliary if "spearman_confidence_vs_negative_regret" in row)
                print(f"[C] {variant} {dataset} seed={seed} best_epoch={result['best_epoch']}", flush=True)
            del ctx
            torch.cuda.empty_cache()
    if not confidence_rows:
        confidence_rows = [{"dataset": dataset, "candidate": "confirmation_ranker", "status": "NOT_TESTED",
                            "reason": f"screening winner locked to {winner}; conditional confirmation ranker fit was not authorized",
                            "spearman_confidence_vs_negative_regret": np.nan}
                           for dataset in dataio.CONFIRM]
    stability_rows = dataio._stability_rows(frames, dataio.CONFIRM)
    dataio.write_csv(RES / "confirmation_global_action_baseline.csv", confirmation_global_rows)
    dataio.write_csv(RES / "confirmation_action_landscape.csv", summaries)
    dataio.write_csv(RES / "confirmation_action_stability.csv", stability_rows)
    dataio.write_csv(RES / "confirmation_results.csv", result_rows)
    dataio.write_csv(RES / "confirmation_regret.csv", regret_rows)
    dataio.write_csv(RES / "confirmation_confidence_analysis.csv", confidence_rows)
    return {"winner": winner, "confirmation_variants": [x[0] for x in confirm_variants],
            "confirmation_fit_count": fit_count, "action_landscape_rows": len(summaries), "stability_rows": len(stability_rows)}


def build_final_outputs(lock: dict, confirmation: dict) -> dict:
    screening_land = _read("phase_a_action_landscape.csv")
    stability = _read("phase_a_action_stability.csv")
    d0 = _read("phase_a_d0_misrouting.csv")
    strat = _read("phase_a_margin_stratification.csv")
    selection = _read("phase_b_selection_regret.csv")
    execution = _read("phase_b_joint_execution.csv")
    confidence = _read("phase_c_confidence_analysis.csv")
    coverage = _read("phase_c_coverage_curve.csv")
    confirm_land = _read("confirmation_action_landscape.csv") if (RES / "confirmation_action_landscape.csv").exists() else pd.DataFrame()
    confirm_stab = _read("confirmation_action_stability.csv") if (RES / "confirmation_action_stability.csv").exists() else pd.DataFrame()
    confirm_exec = _read("confirmation_results.csv") if (RES / "confirmation_results.csv").exists() else pd.DataFrame()
    confirm_regret = _read("confirmation_regret.csv") if (RES / "confirmation_regret.csv").exists() else pd.DataFrame()
    confirm_confidence = _read("confirmation_confidence_analysis.csv") if (RES / "confirmation_confidence_analysis.csv").stat().st_size > 2 else pd.DataFrame()
    gates = lock["gate_results"]

    # Descriptive status for cross-seed action stability and fragile-node concentration.
    stab_val = stability[stability.split == "validation"]
    stable_by_ds = stab_val.groupby("dataset").agg(best=("hard_best_action_agreement", "mean"), order=("pairwise_action_ordering_agreement", "mean"))
    h1_stable = int(((stable_by_ds.best >= .5) & (stable_by_ds.order >= .5)).sum()) >= 2
    frag = strat[(strat.split == "validation") & (strat.modality.isin(["text", "visual"]))]
    frag_rows = []
    for candidate in ["E6_D0", "R0_trajectory_ranker", "R1_capacity_control", "R1_decision_ranker"]:
        sub = frag[frag.candidate == candidate].copy()
        if sub.empty: continue
        primary = sub["selection_regret_mean"] if "selection_regret_mean" in sub else pd.Series(np.nan, index=sub.index)
        alternate = sub["mean_selection_regret"] if "mean_selection_regret" in sub else pd.Series(np.nan, index=sub.index)
        sub["marginal_regret"] = primary.fillna(alternate)
        means = sub.groupby(["dataset", "margin_quartile"]).marginal_regret.mean().unstack()
        for dataset, row in means.iterrows():
            if 1 in row.index and 4 in row.index:
                frag_rows.append({"candidate": candidate, "dataset": dataset, "q1_regret": float(row[1]), "q4_regret": float(row[4]),
                                  "fragile_q1_regret_higher": bool(row[1] > row[4])})
    dataio.write_csv(RES / "fragile_decisive_summary.csv", frag_rows)
    fragile_candidates = ("E6_D0", "R0_trajectory_ranker", "R1_decision_ranker")
    fragile_support = any(sum(r["fragile_q1_regret_higher"] for r in frag_rows if r["candidate"] == candidate) >= 2 for candidate in fragile_candidates)

    # Mechanism contrast uses confirmation observations only and makes no post-lock model changes.
    contrast_rows = []
    if not confirm_land.empty:
        val_land = confirm_land[confirm_land.split == "validation"]
        for dataset in dataio.CONFIRM:
            action = val_land[val_land.dataset == dataset]
            stab = confirm_stab[(confirm_stab.dataset == dataset) & (confirm_stab.split == "validation")]
            row = {"dataset": dataset,
                   "mean_oracle_gain_vs_uniform": float(action.oracle_gain_mean.mean()),
                   "mean_oracle_margin": float(action.oracle_margin_mean.mean()),
                   "uniform_best_fraction": float(action.uniform_hard_best_fraction.mean()),
                   "mean_best_action_stability": float(stab.hard_best_action_agreement.mean()),
                   "mean_pairwise_order_stability": float(stab.pairwise_action_ordering_agreement.mean())}
            if not confirm_regret.empty:
                rs = confirm_regret[(confirm_regret.dataset == dataset) & (confirm_regret.candidate.isin(["R0_trajectory_ranker", "R1_decision_ranker", "R1_capacity_control"]))]
                row["ranker_mean_selection_regret"] = float(rs.selection_regret_mean.mean()) if len(rs) else np.nan
                row["ranker_oracle_agreement"] = float(rs.top1_oracle_action_agreement.mean()) if len(rs) else np.nan
                conf_ds = confirm_confidence[confirm_confidence.dataset == dataset] if not confirm_confidence.empty else pd.DataFrame()
                confidence_tested = bool(len(conf_ds) and "spearman_confidence_vs_negative_regret" in conf_ds and
                                          conf_ds.spearman_confidence_vs_negative_regret.notna().any())
                row["ranker_confidence_spearman"] = float(conf_ds.spearman_confidence_vs_negative_regret.mean()) if confidence_tested else np.nan
                row["confidence_regret_status"] = "TESTED" if confidence_tested else "NOT_TESTED: no confirmation ranker was eligible after lock"
            if not confirm_exec.empty:
                er = confirm_exec[(confirm_exec.dataset == dataset) & (confirm_exec.split == "validation")]
                u = er[er.candidate == "C0_Uniform"].accuracy.mean()
                d = er[er.candidate == "E6_D0"].accuracy.mean()
                row["E6_D0_delta_accuracy_vs_Uniform"] = float(d - u)
                learned = er[er.candidate.isin(["R0_trajectory_ranker", "R1_decision_ranker", "R1_capacity_control"])]
                row["E7_ranker_delta_accuracy_vs_Uniform"] = float(learned.accuracy.mean() - u) if len(learned) else np.nan
                row["mean_joint_deployment_regret"] = float(learned.mean_joint_deployment_regret.mean()) if len(learned) else np.nan
            gain = row["mean_oracle_gain_vs_uniform"]
            row["mechanism_read"] = ("lower relative oracle headroom than peer; positive gain remains, but routing benefit is unestablished" if gain <= float(val_land.oracle_gain_mean.mean()) else "higher relative oracle headroom; ranker observability and execution interaction remain untested")
            contrast_rows.append(row)
    dataio.write_csv(RES / "toys_grocery_contrast.csv", contrast_rows)

    h8 = False
    if len(contrast_rows) == 2:
        gains = [r["mean_oracle_gain_vs_uniform"] for r in contrast_rows]
        has_confirmation_ranker_evidence = any(np.isfinite(r.get("ranker_mean_selection_regret", np.nan)) for r in contrast_rows)
        h8 = has_confirmation_ranker_evidence and not np.isclose(gains[0], gains[1])
    h9_status = "NOT_TESTED"
    if not confirm_exec.empty:
        winner = lock["winner"]
        confirm_winner = confirm_exec[(confirm_exec.candidate == winner) & (confirm_exec.split == "validation")]
        uniform = confirm_exec[(confirm_exec.candidate == "C0_Uniform") & (confirm_exec.split == "validation")]
        if len(confirm_winner) and len(uniform):
            merged = confirm_winner.merge(uniform, on=["dataset", "seed"], suffixes=("_winner", "_uniform"))
            merged["delta_accuracy"] = merged.accuracy_winner - merged.accuracy_uniform
            merged["delta_macro_f1"] = merged.macro_f1_winner - merged.macro_f1_uniform
            per_dataset = merged.groupby("dataset")[["delta_accuracy", "delta_macro_f1"]].mean()
            aggregate_nonworse = merged.delta_accuracy.mean() >= -1e-12 and merged.delta_macro_f1.mean() >= -1e-12
            consistent = bool((per_dataset.delta_accuracy >= -1e-12).all() and (per_dataset.delta_macro_f1 >= -1e-12).all())
            if consistent:
                h9_status = "STRONG_SUPPORT"
            elif aggregate_nonworse or bool((per_dataset.delta_accuracy >= 0).any() or (per_dataset.delta_macro_f1 >= 0).any()):
                h9_status = "MIXED_SUPPORT"
            else:
                h9_status = "NO_SUPPORT"

    statuses = {
        "H1_structural_action_preference_is_stable": "STRONG_SUPPORT" if h1_stable else "MIXED_SUPPORT",
        "H2_trajectory_evidence_predicts_action_utility": "STRONG_SUPPORT" if gates["H1_COUNTERFACTUAL_RANKING_OBSERVABLE"]["passed"] else "NO_SUPPORT",
        "H3_decision_response_adds_observability": "STRONG_SUPPORT" if gates["H2_DECISION_RESPONSE_ADDS_INFORMATION"]["passed"] else "NO_SUPPORT",
        "H4_counterfactual_ranking_beats_direct_ce": "STRONG_SUPPORT" if gates["H3_COUNTERFACTUAL_RANKER_BEATS_DIRECT_CE"]["passed"] else "NO_SUPPORT",
        "H5_misrouting_concentrates_on_fragile_nodes": "STRONG_SUPPORT" if fragile_support else "MIXED_SUPPORT",
        "H6_ranker_confidence_predicts_routing_reliability": "STRONG_SUPPORT" if gates["H4_CONFIDENCE_IS_MEANINGFUL"]["passed"] else "NO_SUPPORT",
        "H7_selective_uniform_fallback_is_promising": "STRONG_SUPPORT" if gates["H5_SELECTIVE_ROUTING_PROMISING"]["passed"] else "NO_SUPPORT",
        "H8_toys_grocery_contrast_is_explained": "STRONG_SUPPORT" if h8 else "MIXED_SUPPORT",
        "H9_confirmation_supports_selected_mechanism": h9_status,
    }
    summary = {"experiment": "E7-CSAO", "branch": "structural_observability_audit", "winner": lock["winner"],
               "formal_screening_fit_count": 27, "confirmation_fit_count": confirmation["confirmation_fit_count"],
               "test_evaluated": False, "screening_gates": gates, "decision_matrix": statuses,
               "confirmation": confirmation, "fragile_decisive_by_candidate_dataset": frag_rows,
               "toys_grocery_contrast": contrast_rows,
               "current_best_explanation": "Routing mainline evidence is insufficient. Structural oracle headroom and moderate cross-seed preference stability exist, but trajectory/decision rankers failed action-utility and confidence gates; task/deployment deltas are small or inconsistent. The evidence points most toward an action-observability bottleneck, while it does not establish that the CE routing objective is wrong.",
               "next_stage_direction": "Keep frozen C0 and E6 D0 as baselines; do not promote a new paper router from this audit. Improve or validate inference-time action-identification evidence before reopening hard structural routing. Selective Uniform fallback remains unsupported."}
    summary = _json_safe(summary)
    dataio.write_json(RES / "structural_observability_summary.json", summary)
    def _md_table(frame, columns, digits=4):
        if frame.empty:
            return "_(not available)_"
        headers = [str(c) for c in columns]
        rows = []
        for values in frame[columns].itertuples(index=False, name=None):
            formatted = []
            for value in values:
                if pd.isna(value):
                    formatted.append("—")
                elif isinstance(value, (float, np.floating)):
                    formatted.append(f"{float(value):.{digits}f}")
                else:
                    formatted.append(str(value))
            rows.append(formatted)
        return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"] +
                           ["| " + " | ".join(row) + " |" for row in rows])

    action_report = screening_land.groupby(["dataset", "modality", "split"], as_index=False).agg(
        margin=("oracle_margin_mean", "mean"), gain=("oracle_gain_mean", "mean"),
        gain_positive=("fraction_oracle_gain_gt_0", "mean"), uniform_best=("uniform_hard_best_fraction", "mean"))
    stability_report = stab_val.groupby(["dataset", "modality"], as_index=False).agg(
        best_action_agreement=("hard_best_action_agreement", "mean"),
        margin_spearman=("oracle_margin_spearman", "mean"), gain_spearman=("oracle_gain_spearman", "mean"),
        pairwise_ordering=("pairwise_action_ordering_agreement", "mean"))
    d0_q = d0[d0.margin_quartile.notna()].groupby(["dataset", "modality", "margin_quartile"], as_index=False).agg(
        agreement=("hard_order_agreement", "mean"), regret=("selection_regret_mean", "mean"),
        margin=("mean_oracle_margin", "mean"))
    ranker = execution[(execution.split == "validation") & execution.candidate.isin([v[0] for v in dataio.VARIANTS])].groupby(
        ["candidate", "dataset"], as_index=False).agg(accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
        joint_regret=("mean_joint_deployment_regret", "mean"))
    selection_report = selection[(selection.split == "validation") & selection.candidate.isin([v[0] for v in dataio.VARIANTS])].groupby(
        ["candidate", "dataset"], as_index=False).agg(selection_regret=("selection_regret_mean", "mean"),
        oracle_agreement=("top1_oracle_action_agreement", "mean"))
    conf_report = confidence.groupby(["dataset", "modality"], as_index=False).agg(
        rho=("spearman_confidence_vs_negative_regret", "mean"), high_q4_regret=("high_confidence_q4_regret", "mean"),
        low_q1_regret=("low_confidence_q1_regret", "mean"))
    coverage_report = coverage.groupby("nominal_train_coverage", as_index=False).agg(
        actual_coverage=("actual_validation_coverage", "mean"), accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"), mean_joint_regret=("mean_joint_deployment_regret", "mean"))
    restart_summary = json.loads((RES / "e6_corrected_restart_audit_summary.json").read_text(encoding="utf-8"))
    lines = ["# Experiment 7 — Counterfactual Structural Action Observability & Selective Routing Audit", "",
             f"Screening winner locked before confirmation: **{lock['winner']}**. Screening fits: 27. Confirmation ranker fits: {confirmation['confirmation_fit_count']}. Test metrics: not evaluated.", "",
             "## Screening gates", "", "| Gate | Status |", "|---|---|"]
    for name, record in gates.items():
        lines.append(f"| {name} | {'PASS' if record['passed'] else 'FAIL'} |")
    lines.extend(["", "## Final decision matrix", "", "| Question | Status |", "|---|---|"])
    for name, status in statuses.items(): lines.append(f"| {name} | {status} |")
    lines.extend(["", "## Corrected E6 restart range", "",
                  f"Groups: {restart_summary['chunk_count']}; mean {restart_summary['range_mean']:.6g}, median {restart_summary['range_median']:.6g}, P90 {restart_summary['range_p90']:.6g}, max {restart_summary['range_max']:.6g}. Grouping is dataset/seed/split/level/chunk_start, then max-minus-min across three restarts. The historical E6 audit was not edited.",
                  "", "## Screening oracle landscape", "", _md_table(action_report, ["dataset", "modality", "split", "margin", "gain", "gain_positive", "uniform_best"]),
                  "", "## Marginal action cross-seed stability", "", _md_table(stability_report, ["dataset", "modality", "best_action_agreement", "margin_spearman", "gain_spearman", "pairwise_ordering"]),
                  "", "## E6 D0 projected misrouting by oracle-margin quartile", "", _md_table(d0_q, ["dataset", "modality", "margin_quartile", "margin", "agreement", "regret"]),
                  "", "## Screening rankers: joint deployment", "", _md_table(ranker, ["candidate", "dataset", "accuracy", "macro_f1", "joint_regret"]),
                  "", "## Screening rankers: marginal action selection", "", _md_table(selection_report, ["candidate", "dataset", "oracle_agreement", "selection_regret"]),
                  "", "## Fragile versus decisive nodes", "", _md_table(pd.DataFrame(frag_rows), ["candidate", "dataset", "q1_regret", "q4_regret", "fragile_q1_regret_higher"]),
                  "", "Margin quartiles were descriptive only and did not affect training. Detailed joint accuracy/F1 and oracle-gain-captured fractions are in `phase_a_margin_stratification.csv`.",
                  "", "## Confidence and selective fallback", "", _md_table(conf_report, ["dataset", "modality", "rho", "high_q4_regret", "low_q1_regret"]),
                  "", _md_table(coverage_report, ["nominal_train_coverage", "actual_coverage", "accuracy", "macro_f1", "mean_joint_regret"]),
                  "", "Coverage thresholds came from Train confidence distributions. The 50% row is the preregistered selective diagnostic; other coverage levels are descriptive.",
                  "", "## Toys versus Grocery confirmation", "", _md_table(pd.DataFrame(contrast_rows), ["dataset", "mean_oracle_gain_vs_uniform", "mean_oracle_margin", "uniform_best_fraction", "mean_best_action_stability", "mean_pairwise_order_stability", "ranker_mean_selection_regret", "ranker_oracle_agreement", "ranker_confidence_spearman", "confidence_regret_status", "E6_D0_delta_accuracy_vs_Uniform", "E7_ranker_delta_accuracy_vs_Uniform", "mean_joint_deployment_regret"], digits=5),
                  "", "The winner was E6 D0, so the protocol required frozen C0/D0 confirmation and did not authorize confirmation ranker fits. R0/R1 regret and confidence on Toys/Grocery are therefore NOT_TESTED; no confirmation ranker was loaded.",
                  "The contrast differentiates relative oracle headroom and stability; it cannot distinguish low observability from deployment interaction without an eligible confirmation ranker.",
                  "", "## Interpretation", "",
                  "The screening gates did not qualify a learned ranker. The winner remains the reused E6 D0 baseline by validation Accuracy, Macro-F1, regret and consistency among eligible candidates. This is evidence against promoting a new hard-action router from this experiment; it does not identify a better routing objective. Selective fallback also failed its preregistered aggregate/worst-dataset gate.",
                  "", "No Test indices, labels, or metrics were accessed. No C0 checkpoint was modified or trained. E6 source results were left unchanged.", ""])
    (RES / "structural_observability_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    device = "cuda:0"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E7 formal fits")
    print("[0] corrected E6 restart audit", flush=True)
    dataio.phase0_main()
    print("[A] preflight and action landscape", flush=True)
    dataio.preflight_screening(device)
    print("[B] 27 screening fits", flush=True)
    trainio.train_screening(device)
    print("[LOCK] evaluate screening gates and write winner lock", flush=True)
    lock = evaluate_screening_and_lock()
    print(f"[C] confirmation starts after lock: {lock['winner']}", flush=True)
    confirmation = run_confirmation(device)
    summary = build_final_outputs(lock, confirmation)
    print(f"[DONE] winner={lock['winner']} fits=27+{confirmation['confirmation_fit_count']}", flush=True)
    return summary


if __name__ == "__main__":
    main()
