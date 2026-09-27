from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_routing_identifiability as common
from scripts import run_utility_routing_audit as e5
from src.models.canonical_routing_audit import (
    CanonicalRoutingAudit, canonical_mix, e5_probabilities_to_alpha,
)
from src.models.utility_routing_audit import (
    UtilityRouter, action_bank, expected_mixture, safe_probabilities, uniform_state_mix,
)

OUT = ROOT / "outputs/routing_identifiability_v1"
RES = ROOT / "results/routing_identifiability_v1"
DATASETS, SEEDS = common.SCREEN, common.SEEDS
VARIANTS = common.VARIANTS


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def get_c0(dataset, seed, device):
    return common._load_c0(dataset, seed, torch.device(device))


def phase_a_one(dataset, seed, device):
    cache = e5._move_cache(torch.load(e5.cache_path(dataset, seed), map_location="cpu", weights_only=False), torch.device(device))
    st, sv = cache["states_text"], cache["states_visual"]
    allowed, train, val, y = cache["allowed_idx"], cache["train_idx"], cache["val_idx"], cache["y_allowed"]
    labels = sorted(int(v) for v in torch.unique(y).cpu().tolist())
    _, _, _, c0, head = get_c0(dataset, seed, device)
    arows, erows, trows, irows = [], [], [], []
    for variant in VARIANTS:
        saved = torch.load(e5.checkpoint_path("phase_b", variant, dataset, seed), map_location="cpu", weights_only=False)
        router = UtilityRouter(cache["hidden_dim"], cache["num_classes"], saved["mode"]).to(device)
        router.load_state_dict(saved["router_state"], strict=True)
        router.eval()
        with torch.no_grad():
            out = e5.router_forward(router, cache)
            pt, pv = out["prob_text"], out["prob_visual"]
            at, av = e5_probabilities_to_alpha(pt), e5_probabilities_to_alpha(pv)
            e5t, e5v = expected_mixture(action_bank(st), pt), expected_mixture(action_bank(sv), pv)
            cat, cav = canonical_mix(st, at), canonical_mix(sv, av)
            ee = max((e5t[allowed] - cat[allowed]).abs().max().item(), (e5v[allowed] - cav[allowed]).abs().max().item())
            le = (head(c0._fuse(e5t[allowed], e5v[allowed])) - head(c0._fuse(cat[allowed], cav[allowed]))).abs().max().item()
            if ee >= 1e-6 or le >= 1e-6:
                raise RuntimeError(f"E5 to alpha mismatch {dataset}/{seed}/{variant}: {ee:g}, {le:g}")
            for modality, alpha in (("text", at[allowed]), ("visual", av[allowed])):
                for split, ix in (("train", train), ("validation", val)):
                    a = alpha[ix].clamp_min(1e-12)
                    ent = -(a * a.log()).sum(-1)
                    active = (a > .10).sum(-1)
                    nearest = torch.cdist(a, torch.eye(4, device=a.device), p=1).min(-1).values
                    vals = {
                        "mean_entropy": ent.mean().item(),
                        "node_variance": a.var(0, unbiased=False).mean().item(),
                        "mean_l1_to_uniform": (a - .25).abs().sum(-1).mean().item(),
                        "mean_nearest_vertex_l1": nearest.mean().item(),
                        "mean_active_orders_gt_0_10": active.float().mean().item(),
                        "mean_top2_mass": a.topk(2, dim=-1).values.sum(-1).mean().item(),
                    }
                    arows.append({
                        "variant": variant, "dataset": dataset, "seed": seed, "modality": modality,
                        "split": split, "embedding_max_abs_error": ee, "logit_max_abs_error": le,
                        **{f"mean_alpha_s{k}": a[:, k].mean().item() for k in range(4)},
                        **{f"std_alpha_s{k}": a[:, k].std(unbiased=False).item() for k in range(4)}, **vals,
                    })
                    erows.append({"variant": variant, "dataset": dataset, "seed": seed, "modality": modality, "split": split, **vals})
            for modality, ce in (("text", cache["action_ce_text"][allowed]), ("visual", cache["action_ce_visual"][allowed])):
                q1 = torch.softmax(-ce, -1)
                for split, ix in (("train", train), ("validation", val)):
                    for temp in (1., .5, .25, .10):
                        q = torch.softmax(-ce[ix] / temp, -1).clamp_min(1e-12)
                        base = q1[ix].clamp_min(1e-12)
                        mid = (q + base) * .5
                        js = .5 * ((q * (q.log() - mid.log())).sum(-1) + (base * (base.log() - mid.log())).sum(-1))
                        h = -(q * q.log()).sum(-1)
                        trows.append({"variant": variant, "dataset": dataset, "seed": seed, "modality": modality, "split": split,
                                      "temperature": temp, "entropy": h.mean().item(),
                                      "preference_strength": (1 - h / math.log(5)).mean().item(),
                                      "hard_ranking_invariance_vs_T1": (ce[ix].argmin(-1) == (ce[ix] / temp).argmin(-1)).float().mean().item(),
                                      "soft_q_js_vs_T1": js.mean().item()})
            st_a, sv_a = [s[allowed] for s in st], [s[allowed] for s in sv]
            normal = head(c0._fuse(canonical_mix(st_a, at[allowed]), canonical_mix(sv_a, av[allowed])))[val]
            nacc, nf1, npred = common._metric_pair(normal, y[val], labels)
            specs = [("normal", at[allowed], av[allowed])]
            mt, mv = at[allowed].clone(), av[allowed].clone()
            mt[val], mv[val] = at[allowed][train].mean(0), av[allowed][train].mean(0)
            specs.append(("train_mean_alpha", mt, mv))
            specs.append(("e5_safe_routing", e5_probabilities_to_alpha(safe_probabilities(pt))[allowed], e5_probabilities_to_alpha(safe_probabilities(pv))[allowed]))
            specs.append(("uniform_alpha", torch.full_like(mt, .25), torch.full_like(mv, .25)))
            for j in range(10):
                gen = torch.Generator(device="cpu").manual_seed(seed * 10007 + j)
                perm = torch.randperm(val.numel(), generator=gen).to(device)
                qt, qv = at[allowed].clone(), av[allowed].clone()
                qt[val], qv[val] = at[allowed][val[perm]], av[allowed][val[perm]]
                specs.append((f"node_shuffle_{j}", qt, qv))
            for name, qt, qv in specs:
                logits = head(c0._fuse(canonical_mix(st_a, qt), canonical_mix(sv_a, qv)))[val]
                acc, f1, pred = common._metric_pair(logits, y[val], labels)
                irows.append({"variant": variant, "dataset": dataset, "seed": seed, "intervention": name,
                              "val_acc": acc, "val_macro_f1": f1, "delta_acc_vs_normal": acc - nacc,
                              "delta_macro_f1_vs_normal": f1 - nf1,
                              "prediction_flip_fraction": (pred != npred).float().mean().item(),
                              "mean_abs_alpha_change": .5 * ((qt[val] - at[allowed][val]).abs().mean().item() + (qv[val] - av[allowed][val]).abs().mean().item())})
    pflat = e5_probabilities_to_alpha(torch.full((1, 5), .2, device=device))
    pau = e5_probabilities_to_alpha(torch.tensor([[0., 0., 0., 0., 1.]], device=device))
    if not torch.equal(pflat, pau) or not torch.equal(pflat, torch.full_like(pflat, .25)):
        raise RuntimeError("redundant E5 vectors do not map exactly to alpha_U")
    return arows, erows, trows, irows, {"dataset": dataset, "seed": seed, "embedding_max_error": ee,
                                      "logit_max_error": le, "redundant_examples_alpha": pflat.cpu().tolist()}


def phase_a(device):
    buckets = [[], [], [], [], []]
    for ds in DATASETS:
        for seed in SEEDS:
            result = phase_a_one(ds, seed, device)
            for bucket, rows in zip(buckets, result[:4]): bucket.extend(rows)
            buckets[4].append(result[4])
            print(f"[A] {ds} seed={seed}", flush=True)
    names = ["phase_a_e5_canonicalization.csv", "phase_a_e5_effective_filter_analysis.csv",
             "phase_a_teacher_scale_diagnostic.csv", "phase_a_e5_interventions.csv"]
    for name, rows in zip(names, buckets[:4]): write_csv(RES / name, rows)
    write_json(RES / "phase_a_canonicalization_checks.json", buckets[4])
    max_e = max(x["embedding_max_error"] for x in buckets[4])
    max_l = max(x["logit_max_error"] for x in buckets[4])
    t = pd.DataFrame(buckets[2]).query("split == 'validation' and temperature == 1.0")
    (RES / "phase_a_report.md").write_text(
        "# Phase A — E5 canonicalization\n\n"
        "Mapped p to four-order alpha with alpha_k = p_k + pU/4. Both redundant examples map exactly to alpha_U.\n\n"
        f"Maximum embedding error: {max_e:.9g}; maximum logit error: {max_l:.9g}.\n"
        f"Mean T=1 Validation teacher preference strength: {t.preference_strength.mean():.6g}.\n"
        "Teacher temperatures are diagnostic only and were not used for training. See Phase A CSVs.\n", encoding="utf-8")
    return {"max_embedding_error": max_e, "max_logit_error": max_l}


def phase_b_one(dataset, seed, device):
    dev = torch.device(device)
    _, data, _, c0, head = get_c0(dataset, seed, dev)
    allowed, y, train, val = common._allowed_split(data, dev)
    with torch.no_grad():
        analysis = c0.analyze(data.x.to(dev), data.edge_index.to(dev))
        st0, sv0 = analysis["S_text"], analysis["S_visual"]
        st, sv = [x[allowed] for x in st0], [x[allowed] for x in sv0]
        ut, uv = uniform_state_mix(st), uniform_state_mix(sv)
        n, chunk = len(y), 32768
        joint = np.empty((n, 4, 4), np.float32)
        lt_u, lu_v, luu = np.empty((n, 4), np.float32), np.empty((n, 4), np.float32), np.empty(n, np.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n); sl = slice(start, end); yy = y[sl]
            for k in range(4):
                logits = head(c0._fuse(st[k][sl], uv[sl]))
                lt_u[sl, k] = F.cross_entropy(logits, yy, reduction="none").cpu().numpy()
                for l in range(4):
                    logits = head(c0._fuse(st[k][sl], sv[l][sl]))
                    joint[sl, k, l] = F.cross_entropy(logits, yy, reduction="none").cpu().numpy()
            for l in range(4):
                logits = head(c0._fuse(ut[sl], sv[l][sl]))
                lu_v[sl, l] = F.cross_entropy(logits, yy, reduction="none").cpu().numpy()
            logits = head(c0._fuse(ut[sl], uv[sl]))
            luu[sl] = F.cross_entropy(logits, yy, reduction="none").cpu().numpy()
    marginal_hat = lt_u[:, :, None] + lu_v[:, None, :] - luu[:, None, None]
    residual = joint - marginal_hat
    grand = joint.mean((1, 2), keepdims=True)
    rowm, colm = joint.mean(2, keepdims=True), joint.mean(1, keepdims=True)
    component = joint - rowm - colm + grand
    denom = ((joint - grand) ** 2).sum((1, 2))
    energy = (component ** 2).sum((1, 2)) / np.maximum(denom, 1e-12)
    flatarg = joint.reshape(n, 16).argmin(1)
    bestk, bestl = flatarg // 4, flatarg % 4
    km, lm = lt_u.argmin(1), lu_v.argmin(1)
    ix = np.arange(n)
    marginal_loss = joint[ix, km, lm]
    best_loss = joint[ix, bestk, bestl]
    regret, gain = marginal_loss - best_loss, luu - best_loss
    best_response_t = joint.argmin(1)
    best_response_v = joint.argmin(2)
    global_ids = allowed.detach().cpu().numpy()
    splits = np.array(["train"] * len(train) + ["validation"] * len(val))
    frame = pd.DataFrame({"node_id": global_ids, "split": splits, "uniform_loss": luu,
                          "k_marginal": km, "l_marginal": lm, "best_k": bestk, "best_l": bestl,
                          "marginal_pair_loss": marginal_loss, "joint_best_loss": best_loss,
                          "joint_pair_regret": regret, "joint_oracle_gain_vs_uniform": gain,
                          "interaction_energy_ratio": energy,
                          "mean_signed_interaction": residual.mean((1, 2)),
                          "mean_abs_interaction": np.abs(residual).mean((1, 2)),
                          "rms_interaction": np.sqrt((residual ** 2).mean((1, 2))),
                          "p90_abs_interaction": np.quantile(np.abs(residual), .9, axis=(1, 2)),
                          "marginal_pair_equals_joint_best": ((km == bestk) & (lm == bestl)),
                          "best_text_order_changes_with_visual": (best_response_t.min(1) != best_response_t.max(1)),
                          "best_visual_order_changes_with_text": (best_response_v.min(1) != best_response_v.max(1))})
    for k in range(4):
        frame[f"L_t{k}_U"] = lt_u[:, k]
        frame[f"L_U_v{ k }"] = lu_v[:, k]
        frame[f"best_text_response_v{k}"] = best_response_t[:, k]
        frame[f"best_visual_response_t{k}"] = best_response_v[:, k]
        for l in range(4):
            frame[f"L_{k}_{l}"] = joint[:, k, l]
            frame[f"I_{k}_{l}"] = residual[:, k, l]
    outpath = OUT / "phase_b_node_landscape" / f"{dataset}_seed{seed}.csv.gz"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outpath, index=False, compression="gzip")
    # Compact cache for Phase C; raw tensors stay under outputs/.
    cp = OUT / "phase_b_cache" / f"{dataset}_seed{seed}.npz"
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, joint=joint, lt_u=lt_u, lu_v=lu_v, luu=luu, node_id=global_ids, split=splits)
    summary, residual_rows, regret_rows = [], [], []
    for split in ("train", "validation"):
        sub = frame.loc[frame.split == split]
        if sub.empty: continue
        for k in range(4):
            for l in range(4):
                summary.append({"dataset": dataset, "seed": seed, "split": split, "text_order": k,
                                "visual_order": l, "mean_CE": sub[f"L_{k}_{l}"].mean(),
                                "mean_signed_interaction": sub[f"I_{k}_{l}"].mean(),
                                "mean_abs_interaction": sub[f"I_{k}_{l}"].abs().mean()})
        residual_rows.append({"dataset": dataset, "seed": seed, "split": split,
                              "mean_signed_I": sub.mean_signed_interaction.mean(),
                              "mean_abs_I": sub.mean_abs_interaction.mean(),
                              "RMS_I": sub.rms_interaction.mean(), "P90_abs_I": sub.p90_abs_interaction.mean(),
                              "interaction_energy_mean": sub.interaction_energy_ratio.mean(),
                              "interaction_energy_median": sub.interaction_energy_ratio.median(),
                              "interaction_energy_P90": sub.interaction_energy_ratio.quantile(.9),
                              "fraction_energy_gt_0_1": (sub.interaction_energy_ratio > .1).mean(),
                              "fraction_energy_gt_0_25": (sub.interaction_energy_ratio > .25).mean()})
        regret_rows.append({"dataset": dataset, "seed": seed, "split": split,
                            "mean_joint_pair_regret": sub.joint_pair_regret.mean(),
                            "mean_joint_oracle_gain_vs_uniform": sub.joint_oracle_gain_vs_uniform.mean(),
                            "relative_joint_regret": sub.joint_pair_regret.mean() / max(sub.joint_oracle_gain_vs_uniform.mean(), 1e-12),
                            "marginal_pair_equals_joint_best_fraction": sub.marginal_pair_equals_joint_best.mean(),
                            "best_text_changes_fraction": sub.best_text_order_changes_with_visual.mean(),
                            "best_visual_changes_fraction": sub.best_visual_order_changes_with_text.mean()})
    return frame, summary, residual_rows, regret_rows


def spearman_safe(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return float(spearmanr(a, b).statistic)


def phase_b(device):
    frames, summary, residual, regret = [], [], [], []
    for ds in DATASETS:
        for seed in SEEDS:
            result = phase_b_one(ds, seed, device)
            frames.append(result[0]); summary.extend(result[1]); residual.extend(result[2]); regret.extend(result[3])
            print(f"[B] {ds} seed={seed}", flush=True)
    write_csv(RES / "phase_b_joint_action_summary.csv", summary)
    write_csv(RES / "phase_b_interaction_residual.csv", residual)
    write_csv(RES / "phase_b_joint_regret.csv", regret)
    stability = []
    for ds in DATASETS:
        for split in ("train", "validation"):
            fs = [f.loc[(f.dataset == ds) & (f.split == split)].set_index("node_id") for f in frames]
            for i, j in combinations(range(3), 2):
                common = fs[i].join(fs[j], lsuffix="_a", rsuffix="_b", how="inner")
                if common.empty: continue
                stability.append({"dataset": ds, "split": split, "seed_a": SEEDS[i], "seed_b": SEEDS[j],
                                  "common_nodes": len(common),
                                  "joint_best_pair_agreement": ((common.best_k_a == common.best_k_b) & (common.best_l_a == common.best_l_b)).mean(),
                                  "text_best_response_agreement": np.mean([np.mean(common[f"best_text_response_v{k}_a"] == common[f"best_text_response_v{k}_b"]) for k in range(4)]),
                                  "visual_best_response_agreement": np.mean([np.mean(common[f"best_visual_response_t{k}_a"] == common[f"best_visual_response_t{k}_b"]) for k in range(4)]),
                                  "joint_oracle_gain_spearman": spearman_safe(common.joint_oracle_gain_vs_uniform_a, common.joint_oracle_gain_vs_uniform_b),
                                  "interaction_energy_spearman": spearman_safe(common.interaction_energy_ratio_a, common.interaction_energy_ratio_b),
                                  "signed_interaction_spearman": spearman_safe(common.mean_signed_interaction_a, common.mean_signed_interaction_b)})
    write_csv(RES / "phase_b_stability.csv", stability)
    train_reg = pd.DataFrame(regret).query("split == 'train'")
    dsmeans = train_reg.groupby("dataset").mean_joint_pair_regret.mean()
    positives = int((train_reg.mean_joint_pair_regret > 0).sum())
    mean_regret = train_reg.mean_joint_pair_regret.mean()
    mean_gain = train_reg.mean_joint_oracle_gain_vs_uniform.mean()
    relative = mean_regret / max(mean_gain, 1e-12)
    st = pd.DataFrame(stability).query("split == 'train'")
    stable_ds = 0
    for ds in DATASETS:
        sub = st.loc[st.dataset == ds]
        signed = train_reg.loc[train_reg.dataset == ds].mean_signed_I.to_numpy()
        same_sign = len(signed) == 3 and (np.all(signed >= 0) or np.all(signed <= 0))
        corr = sub.signed_interaction_spearman.dropna().mean() if not sub.empty else np.nan
        if same_sign and np.isfinite(corr) and corr > 0: stable_ds += 1
    gate = {"status": "JOINT_UTILITY_MATERIAL" if (int((dsmeans > 0).sum()) >= 2 and positives >= 6 and relative >= .10 and stable_ds >= 2) else "JOINT_UTILITY_WEAK_OR_MIXED",
            "positive_dataset_count": int((dsmeans > 0).sum()), "positive_dataset_seed_count": f"{positives}/9",
            "relative_joint_regret": relative, "mean_joint_pair_regret": mean_regret,
            "mean_joint_oracle_gain_vs_uniform": mean_gain, "interaction_reproducible_dataset_count": stable_ds,
            "gate_split": "Train primary; Validation diagnostics reported separately"}
    write_json(RES / "phase_b_gate.json", gate)
    return gate
