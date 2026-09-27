from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_routing_identifiability as common
from scripts import run_e6_srij as e6
from scripts import finish_e6_srij as phase_c_impl

OUT, RES = e6.OUT, e6.RES
DATASETS, SEEDS = e6.DATASETS, e6.SEEDS


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_b(device):
    frames, summaries, residuals, regrets = [], [], [], []
    for ds in DATASETS:
        for seed in SEEDS:
            frame, sm, rr, rg = e6.phase_b_one(ds, seed, device)
            frame["dataset"], frame["seed"] = ds, seed
            frames.append(frame); summaries.extend(sm); residuals.extend(rr); regrets.extend(rg)
            print(f"[Phase B] {ds} seed={seed}", flush=True)
            torch.cuda.empty_cache()
    write_csv(RES / "phase_b_joint_action_summary.csv", summaries)
    write_csv(RES / "phase_b_interaction_residual.csv", residuals)
    write_csv(RES / "phase_b_joint_regret.csv", regrets)
    stability = []
    for ds in DATASETS:
        for split in ("train", "validation"):
            by_seed = [f.loc[(f.dataset == ds) & (f.split == split)].set_index("node_id") for f in frames]
            for i, j in __import__("itertools").combinations(range(3), 2):
                q = by_seed[i].join(by_seed[j], lsuffix="_a", rsuffix="_b", how="inner")
                if q.empty: continue
                stability.append({"dataset": ds, "split": split, "seed_a": SEEDS[i], "seed_b": SEEDS[j],
                                  "common_nodes": len(q),
                                  "joint_best_pair_agreement": ((q.best_k_a == q.best_k_b) & (q.best_l_a == q.best_l_b)).mean(),
                                  "text_best_response_agreement": np.mean([np.mean(q[f"best_text_response_v{k}_a"] == q[f"best_text_response_v{k}_b"]) for k in range(4)]),
                                  "visual_best_response_agreement": np.mean([np.mean(q[f"best_visual_response_t{k}_a"] == q[f"best_visual_response_t{k}_b"]) for k in range(4)]),
                                  "joint_oracle_gain_spearman": e6.spearman_safe(q.joint_oracle_gain_vs_uniform_a, q.joint_oracle_gain_vs_uniform_b),
                                  "interaction_energy_spearman": e6.spearman_safe(q.interaction_energy_ratio_a, q.interaction_energy_ratio_b),
                                  "signed_interaction_spearman": e6.spearman_safe(q.mean_signed_interaction_a, q.mean_signed_interaction_b)})
    write_csv(RES / "phase_b_stability.csv", stability)
    tr = pd.DataFrame(regrets).query("split == 'train'")
    signed_train = pd.DataFrame(residuals).query("split == 'train'")
    dsmean = tr.groupby("dataset").mean_joint_pair_regret.mean()
    npositive = int((tr.mean_joint_pair_regret > 0).sum())
    mreg, mgain = float(tr.mean_joint_pair_regret.mean()), float(tr.mean_joint_oracle_gain_vs_uniform.mean())
    relative = mreg / max(mgain, 1e-12)
    st = pd.DataFrame(stability).query("split == 'train'")
    consistent = 0
    for ds in DATASETS:
        per_seed = signed_train.loc[signed_train.dataset == ds, "mean_signed_I"].to_numpy()
        corr = st.loc[st.dataset == ds, "signed_interaction_spearman"].dropna()
        same = len(per_seed) == 3 and (np.all(per_seed >= 0) or np.all(per_seed <= 0))
        consistent += int(same and not corr.empty and corr.mean() > 0)
    passed = int((dsmean > 0).sum()) >= 2 and npositive >= 6 and relative >= .10 and consistent >= 2
    gate = {"status": "JOINT_UTILITY_MATERIAL" if passed else "JOINT_UTILITY_WEAK_OR_MIXED",
            "positive_dataset_count": int((dsmean > 0).sum()), "positive_dataset_seed_count": f"{npositive}/9",
            "relative_joint_regret": relative, "mean_joint_pair_regret": mreg,
            "mean_joint_oracle_gain_vs_uniform": mgain, "interaction_reproducible_dataset_count": consistent,
            "gate_split": "Train primary; Validation diagnostics reported separately"}
    write_json(RES / "phase_b_gate.json", gate)
    return gate


def router_gate(rows, interventions, variant):
    frame = pd.DataFrame([r for r in rows if r["variant"] == variant]).set_index(["dataset", "seed"])
    byds = frame.delta_acc_vs_uniform.groupby(level="dataset").mean()
    positive = int((frame.delta_acc_vs_uniform > 0).sum())
    sh = pd.DataFrame([r for r in interventions if r["variant"] == variant and r["intervention"].startswith("node_shuffle_")])
    effect = bool(not sh.empty and sh.prediction_flip_fraction.mean() > 0)
    passed = int((byds > 0).sum()) >= 2 and positive >= 6 and frame.delta_acc_vs_uniform.mean() > 0 and frame.delta_macro_f1_vs_uniform.mean() >= 0 and effect
    return {"pass": bool(passed), "positive_dataset_count": int((byds > 0).sum()),
            "positive_seed_count": f"{positive}/9", "mean_delta_acc": float(frame.delta_acc_vs_uniform.mean()),
            "mean_delta_macro_f1": float(frame.delta_macro_f1_vs_uniform.mean()),
            "node_shuffle_functional_effect": effect,
            "mean_node_shuffle_prediction_flip": float(sh.prediction_flip_fraction.mean()) if not sh.empty else 0.0}


def run_screening(devices, joint_gate):
    d0runs, d0rows, interventions = [], [], []
    for i, (ds, seed) in enumerate(( (d, s) for d in DATASETS for s in SEEDS )):
        run = phase_c_impl.train_router_one(ds, seed, devices[i % len(devices)], "independent")
        d0runs.append(run); d0rows.append(run["row"]); interventions.extend(phase_c_impl.router_interventions(run))
        print(f"[D0] {ds} seed={seed}", flush=True)
        torch.cuda.empty_cache()
    write_csv(RES / "phase_d_canonical_router_results.csv", d0rows)
    write_csv(RES / "phase_d_interventions.csv", interventions)
    d0gate = router_gate(d0rows, interventions, "D0_independent")
    d1rows, caprows, d1inter = [], [], []
    d1gate = {"status": "NOT_TESTED", "reason": "Phase B joint-utility gate did not pass"}
    if joint_gate["status"] == "JOINT_UTILITY_MATERIAL":
        for i, (ds, seed) in enumerate(((d, s) for d in DATASETS for s in SEEDS)):
            device = devices[i % len(devices)]
            jr = phase_c_impl.train_router_one(ds, seed, device, "joint")
            cr = phase_c_impl.train_router_one(ds, seed, device, "capacity_control")
            d1rows.append(jr["row"]); caprows.append(cr["row"])
            d1inter.extend(phase_c_impl.router_interventions(jr, include_companion=True))
            print(f"[D1] {ds} seed={seed}", flush=True)
            torch.cuda.empty_cache()
        write_csv(RES / "phase_d_joint_router_results.csv", d1rows)
        write_csv(RES / "phase_d_joint_capacity_control.csv", caprows)
        write_csv(RES / "phase_d_joint_interventions.csv", d1inter)
        abs_gate = router_gate(d1rows, d1inter, "D1_joint")
        j = pd.DataFrame(d1rows).set_index(["dataset", "seed"])
        c = pd.DataFrame(caprows).set_index(["dataset", "seed"])
        delta = j.delta_acc_vs_uniform - c.delta_acc_vs_uniform
        dd = delta.rename("delta").reset_index()
        byds = dd.groupby("dataset").delta.mean()
        nd = int((byds > 0).sum()); ns = int((dd.delta > 0).sum())
        sh = pd.DataFrame([r for r in d1inter if r["intervention"].startswith("shuffle_visual_companion") or r["intervention"].startswith("shuffle_text_companion")])
        functional = bool(not sh.empty and sh.prediction_flip_fraction.mean() > 0)
        passed = abs_gate["pass"] and nd >= 2 and ns >= 6 and delta.mean() > 0 and functional
        d1gate = {"status": "CANONICAL_JOINT_ROUTER_PROMISING" if passed else "JOINT_ROUTER_NOT_SUPPORTED",
                  "pass": bool(passed), "absolute_D0_style_gate": abs_gate,
                  "positive_dataset_count_vs_capacity": nd, "positive_seed_count_vs_capacity": f"{ns}/9",
                  "mean_delta_acc_vs_capacity": float(delta.mean()), "companion_shuffle_functional_effect": functional,
                  "mean_companion_shuffle_prediction_flip": float(sh.prediction_flip_fraction.mean()) if not sh.empty else 0.0}
        write_json(RES / "phase_d_joint_gate.json", d1gate)
    write_json(RES / "phase_d_d0_gate.json", d0gate)
    candidates = [{"candidate": "C0_Uniform", "mean_val_acc": float(np.mean([r["c0_uniform_val_acc"] for r in d0rows])),
                   "mean_val_macro_f1": float(np.mean([r["c0_uniform_val_macro_f1"] for r in d0rows]))}]
    if d0gate["pass"]:
        candidates.append({"candidate": "D0_independent", "mean_val_acc": np.mean([r["val_acc"] for r in d0rows]), "mean_val_macro_f1": np.mean([r["val_macro_f1"] for r in d0rows])})
    if d1gate.get("pass"):
        candidates.append({"candidate": "D1_joint", "mean_val_acc": np.mean([r["val_acc"] for r in d1rows]), "mean_val_macro_f1": np.mean([r["val_macro_f1"] for r in d1rows])})
    winner = sorted(candidates, key=lambda r: (r["mean_val_acc"], r["mean_val_macro_f1"]), reverse=True)[0]["candidate"]
    lock = {"experiment": "E6-SRIJ", "winning_variant": winner,
            "selection": "Validation Accuracy primary, Macro-F1 secondary; candidate gates include paired seed consistency and functional intervention",
            "candidate_summary": candidates, "phase_b_joint_gate": joint_gate,
            "phase_d_independent_gate": d0gate, "phase_d_joint_gate": d1gate,
            "confirmation_loaded_before_lock": False, "test_metrics": None}
    write_json(RES / "screening_winner_lock.json", lock)
    print(f"[winner lock] {winner}", flush=True)
    confirmation = []
    if winner in ("D0_independent", "D1_joint"):
        mode = "independent" if winner == "D0_independent" else "joint"
        confirm_int = []
        for i, (ds, seed) in enumerate(((d, s) for d in ("Toys", "Grocery") for s in SEEDS)):
            run = phase_c_impl.train_router_one(ds, seed, devices[i % len(devices)], mode)
            row = dict(run["row"]); row["winner"] = winner; confirmation.append(row)
            its = phase_c_impl.router_interventions(run, include_companion=(mode == "joint"))
            confirm_int.extend(its)
            print(f"[confirmation] {ds} seed={seed}", flush=True)
            torch.cuda.empty_cache()
        write_csv(RES / "confirmation_results.csv", confirmation)
        write_csv(RES / "confirmation_analysis.csv", [
            {"dataset": r["dataset"], "seed": r["seed"], "winner": winner,
             "delta_acc": r["delta_acc_vs_uniform"], "delta_macro_f1": r["delta_macro_f1_vs_uniform"],
             "positive_seed": r["delta_acc_vs_uniform"] > 0} for r in confirmation])
        write_csv(RES / "confirmation_interventions.csv", confirm_int)
    else:
        write_json(RES / "confirmation_status.json", {"status": "NOT_TESTED_ROUTER", "reason": "C0 Uniform won; Toys/Grocery were not loaded"})
    return d0rows, interventions, d0gate, d1rows, caprows, d1inter, d1gate, lock, confirmation


def final_report(a_stats, b_gate, c_stats, d0gate, d1gate, lock, confirmation):
    pc = pd.read_csv(RES / "phase_c_continuous_oracle_summary.csv")
    geom = pd.read_csv(RES / "phase_c_filter_geometry.csv")
    gains = pc.loc[pc.solution == "continuous_minus_discrete_gain"].copy()
    by_seed_gain = gains.groupby(["dataset", "seed"]).mean_CE.mean()
    by_dataset_gain = gains.groupby("dataset").mean_CE.mean()
    nonvertex = geom.groupby("dataset").fraction_ge_2_active_orders.mean()
    continuous_pass = int((by_dataset_gain > 1e-4).sum()) >= 2 and int((by_seed_gain > 1e-4).sum()) >= 6 and int((nonvertex > .5).sum()) >= 2
    near_vertex = int((nonvertex < .5).sum()) >= 2
    stab = pd.read_csv(RES / "phase_c_stability.csv")
    alpha_stable = bool(not stab.empty and stab.alpha_cosine_similarity.mean() >= .8)
    confirmation_status = "NOT_TESTED_ROUTER" if not confirmation else "COMPLETED"
    decisions = [
        {"hypothesis": "H1_e5_routing_nonidentifiability_confirmed", "status": "STRONG_SUPPORT", "evidence": "The two prescribed distinct five-action vectors map exactly to alpha_U; representation and logits are checked against E5."},
        {"hypothesis": "H2_marginal_modality_utility_is_sufficient", "status": "STRONG_SUPPORT" if b_gate["status"] != "JOINT_UTILITY_MATERIAL" else "NO_SUPPORT", "evidence": f"relative joint regret={b_gate['relative_joint_regret']:.6g}; gate={b_gate['status']}"},
        {"hypothesis": "H3_joint_multimodal_structural_utility_is_material", "status": "STRONG_SUPPORT" if b_gate["status"] == "JOINT_UTILITY_MATERIAL" else "NO_SUPPORT", "evidence": json.dumps(b_gate)},
        {"hypothesis": "H4_discrete_order_selection_is_sufficient", "status": "STRONG_SUPPORT" if near_vertex and not continuous_pass else "MIXED_SUPPORT", "evidence": f"datasets with <50% multi-order filters={int((nonvertex < .5).sum())}/3; continuous gate={continuous_pass}"},
        {"hypothesis": "H5_continuous_filtering_has_extra_headroom", "status": "STRONG_SUPPORT" if continuous_pass else "NO_SUPPORT", "evidence": f"mean CE gain by dataset={by_dataset_gain.to_dict()}"},
        {"hypothesis": "H6_continuous_oracle_filter_is_cross_seed_stable", "status": "STRONG_SUPPORT" if alpha_stable else "MIXED_SUPPORT", "evidence": f"mean alpha cosine={stab.alpha_cosine_similarity.mean() if len(stab) else float('nan'):.6g}"},
        {"hypothesis": "H7_canonical_frozen_router_improves_uniform", "status": "STRONG_SUPPORT" if d0gate["pass"] else "NO_SUPPORT", "evidence": json.dumps(d0gate)},
        {"hypothesis": "H8_joint_router_improves_independent_router", "status": "NOT_TESTED" if d1gate.get("status") == "NOT_TESTED" else ("STRONG_SUPPORT" if d1gate.get("pass") else "NO_SUPPORT"), "evidence": json.dumps(d1gate)},
        {"hypothesis": "H9_untouched_confirmation", "status": "NOT_TESTED" if not confirmation else ("STRONG_SUPPORT" if all(r["delta_acc_vs_uniform"] > 0 for r in confirmation) else "MIXED_SUPPORT"), "evidence": confirmation_status if not confirmation else f"{sum(r['delta_acc_vs_uniform'] > 0 for r in confirmation)}/{len(confirmation)} positive seed pairs"},
    ]
    write_csv(RES / "screening_decision_matrix.csv", decisions)
    summary = {"experiment": "E6-SRIJ", "phase_a": a_stats, "phase_b": b_gate,
               "phase_c": {k: v for k, v in c_stats.items() if k != "filter_geometry"},
               "phase_d_d0_gate": d0gate, "phase_d_d1_gate": d1gate,
               "screening_winner": lock["winning_variant"], "confirmation_status": confirmation_status,
               "formal_training_run_count": 9 + (18 if b_gate["status"] == "JOINT_UTILITY_MATERIAL" else 0) + len(confirmation),
               "c0_retraining_runs": 0, "test_metrics_used": False, "decisions": decisions}
    write_json(RES / "routing_identifiability_summary.json", summary)
    lines = ["# Experiment 6 — Structural Routing Identifiability & Joint Utility Audit", "",
             "Canonical action basis is S0–S3; alpha is a four-simplex and alpha_U=(0.25,0.25,0.25,0.25).",
             "C0 was frozen. NC screening used Movies, ele-fashion, Reddit-S and seeds 42–44. Train/Validation only; no Test labels were indexed.",
             "", "## Final decision matrix", "", "| Hypothesis | Status | Evidence |", "|---|---|---|"]
    lines.extend(f"| {r['hypothesis']} | {r['status']} | {str(r['evidence']).replace('|', '/')} |" for r in decisions)
    lines += ["", "## Gates and winner", "", f"Phase B: {b_gate['status']}; relative regret={b_gate['relative_joint_regret']:.6g}.",
              f"D0 gate: {'pass' if d0gate['pass'] else 'fail'}; D1: {d1gate.get('status')}.",
              f"Screening winner: {lock['winning_variant']}.",
              f"Untouched Toys/Grocery confirmation: {confirmation_status}.",
              "", "## Scope", "", "Continuous and discrete oracle results use Train/Validation labels for diagnosis and are not deployable router scores. The report preserves the requested oracle upper-bound interpretation.",
              "No prohibited model components, Test-based selection, C0 retraining, or topology changes were used.", ""]
    (RES / "routing_identifiability_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("E6-SRIJ requires CUDA")
    for name in args.devices:
        if not name.startswith("cuda"): raise RuntimeError(f"CUDA device required: {name}")
    OUT.mkdir(parents=True, exist_ok=True); RES.mkdir(parents=True, exist_ok=True)
    preflight = common.run_preflight(args.devices[0])
    a_stats = e6.phase_a(args.devices[0])
    b_gate = run_b(args.devices[0])
    # The oracle remains a diagnostic and cannot feed D0/D1 training.
    torch.manual_seed(20260927)
    c_stats = phase_c_impl.phase_c(args.devices[0])
    d0rows, d0int, d0gate, d1rows, caprows, d1int, d1gate, lock, confirmation = run_screening(args.devices, b_gate)
    final_report(a_stats, b_gate, c_stats, d0gate, d1gate, lock, confirmation)
    print(json.dumps({"winner": lock["winning_variant"], "phase_b": b_gate["status"],
                      "d0_pass": d0gate["pass"], "d1": d1gate["status"],
                      "formal_training_runs": 9 + (18 if b_gate["status"] == "JOINT_UTILITY_MATERIAL" else 0) + len(confirmation)}, indent=2))


if __name__ == "__main__":
    main()
