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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_routing_identifiability as common
from scripts import run_e6_srij as e6
from scripts import run_utility_routing_audit as e5
from src.models.canonical_routing_audit import CanonicalRoutingAudit, canonical_mix
from src.models.utility_routing_audit import uniform_state_mix

OUT, RES = e6.OUT, e6.RES
DATASETS, SEEDS = e6.DATASETS, e6.SEEDS


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def execute_mix(states, alpha):
    """Differentiable four-state sum; deliberately avoids the uniform fast path."""
    result = states[0] * alpha[:, :1]
    for order in range(1, 4):
        result = result + states[order] * alpha[:, order:order + 1]
    return result


def _losses(c0, head, st, sv, y, at, av, mode):
    if mode == "text":
        at = torch.softmax(at, -1)
        av = torch.full_like(at, .25)
    elif mode == "visual":
        av = torch.softmax(av, -1)
        at = torch.full_like(av, .25)
    else:
        at, av = torch.softmax(at, -1), torch.softmax(av, -1)
    zt, zv = execute_mix(st, at), execute_mix(sv, av)
    logits = head(c0._fuse(zt, zv))
    return F.cross_entropy(logits, y, reduction="none"), at, av, logits


def _optimize_chunk(c0, head, st, sv, y, mode, initial_t, initial_v, seed):
    n, device = len(y), y.device
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    best_loss = torch.full((n,), float("inf"), device=device)
    best_t = torch.full((n, 4), .25, device=device)
    best_v = torch.full((n, 4), .25, device=device)
    restart_rows = []
    for restart in range(3):
        if restart == 0:
            u_t, u_v = torch.zeros((n, 4), device=device), torch.zeros((n, 4), device=device)
        elif restart == 1:
            u_t, u_v = torch.full((n, 4), -12., device=device), torch.full((n, 4), -12., device=device)
            if mode in ("text", "joint"):
                u_t.scatter_(1, initial_t[:, None], 12.)
            if mode in ("visual", "joint"):
                u_v.scatter_(1, initial_v[:, None], 12.)
        else:
            rand_t = torch.distributions.Dirichlet(torch.ones(4)).sample((n,)).log()
            rand_v = torch.distributions.Dirichlet(torch.ones(4)).sample((n,)).log()
            u_t, u_v = rand_t.to(device), rand_v.to(device)
        u_t = torch.nn.Parameter(u_t)
        u_v = torch.nn.Parameter(u_v)
        params = [u_t] if mode == "text" else ([u_v] if mode == "visual" else [u_t, u_v])
        opt = torch.optim.Adam(params, lr=.05)
        local_best = torch.full((n,), float("inf"), device=device)
        local_t, local_v = torch.full((n, 4), .25, device=device), torch.full((n, 4), .25, device=device)
        prev = float("inf"); stale = 0; steps = 0
        for step in range(151):
            opt.zero_grad(set_to_none=True)
            loss, at, av, _ = _losses(c0, head, st, sv, y, u_t, u_v, mode)
            with torch.no_grad():
                improved = loss < local_best
                local_best[improved] = loss[improved]
                local_t[improved] = at[improved]
                local_v[improved] = av[improved]
            current = float(loss.mean().detach().item())
            if step == 150 or stale >= 10:
                steps = step
                break
            loss.mean().backward()
            opt.step()
            steps = step + 1
            rel = (prev - current) / max(abs(prev), 1e-12) if math.isfinite(prev) else float("inf")
            stale = stale + 1 if rel < 1e-6 else 0
            prev = current
        take = local_best < best_loss
        best_loss[take], best_t[take], best_v[take] = local_best[take], local_t[take], local_v[take]
        restart_rows.append({"restart": restart, "mean_best_loss": float(local_best.mean().item()),
                             "steps": steps, "early_stopped": steps < 150})
    return best_loss, best_t, best_v, restart_rows


def optimize_split(c0, head, st, sv, y, discrete_t, discrete_v, mode, seed, chunk_size=4096):
    losses, alphas_t, alphas_v, audits = [], [], [], []
    for start in range(0, len(y), chunk_size):
        end = min(start + chunk_size, len(y)); sl = slice(start, end)
        result = _optimize_chunk(c0, head, [x[sl] for x in st], [x[sl] for x in sv], y[sl], mode,
                                 torch.as_tensor(discrete_t[sl], device=y.device, dtype=torch.long),
                                 torch.as_tensor(discrete_v[sl], device=y.device, dtype=torch.long),
                                 seed + start)
        losses.append(result[0].detach().cpu().numpy())
        alphas_t.append(result[1].detach().cpu().numpy())
        alphas_v.append(result[2].detach().cpu().numpy())
        audits.extend({**r, "chunk_start": start, "chunk_size": end - start} for r in result[3])
    return np.concatenate(losses), np.concatenate(alphas_t), np.concatenate(alphas_v), audits


def evaluate_mix(c0, head, st, sv, y, at, av, device):
    with torch.no_grad():
        at, av = torch.as_tensor(at, device=device), torch.as_tensor(av, device=device)
        logits = head(c0._fuse(execute_mix(st, at), execute_mix(sv, av)))
        loss = F.cross_entropy(logits, y, reduction="none").cpu().numpy()
        pred = logits.argmax(-1).cpu().numpy()
        truth = y.cpu().numpy()
    labels = sorted(int(x) for x in np.unique(truth))
    f1 = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score
    return loss, float(np.mean(pred == truth)), float(f1(truth, pred, labels=labels, average="macro", zero_division=0))


def phase_c_one(dataset, seed, device):
    dev = torch.device(device)
    _, data, _, c0, head = e6.get_c0(dataset, seed, dev)
    allowed, y, train, val = common._allowed_split(data, dev)
    with torch.no_grad():
        analysis = c0.analyze(data.x.to(dev), data.edge_index.to(dev))
        st_all, sv_all = analysis["S_text"], analysis["S_visual"]
        st, sv = [x[allowed] for x in st_all], [x[allowed] for x in sv_all]
    saved = np.load(OUT / "phase_b_cache" / f"{dataset}_seed{seed}.npz")
    joint, lt_u, lu_v, luu = saved["joint"], saved["lt_u"], saved["lu_v"], saved["luu"]
    node_id, split = saved["node_id"], saved["split"]
    raw, summaries, audits, geometries = [], [], [], []
    for split_name in ("train", "validation"):
        ids = np.flatnonzero(split == split_name)
        if not len(ids): continue
        sub_st, sub_sv = [x[ids] for x in st], [x[ids] for x in sv]
        sub_y = y[ids]
        jl = joint[ids]
        best_flat = jl.reshape(len(ids), 16).argmin(1)
        best_t, best_v = best_flat // 4, best_flat % 4
        km, lm = lt_u[ids].argmin(1), lu_v[ids].argmin(1)
        # C1 text marginal; C2 visual marginal; C4 joint simplex oracle.
        c1 = optimize_split(c0, head, sub_st, sub_sv, sub_y, km, lm, "text", seed * 1009 + 1)
        c2 = optimize_split(c0, head, sub_st, sub_sv, sub_y, km, lm, "visual", seed * 1009 + 2)
        c4 = optimize_split(c0, head, sub_st, sub_sv, sub_y, best_t, best_v, "joint", seed * 1009 + 4)
        c3loss, c3acc, c3f1 = evaluate_mix(c0, head, sub_st, sub_sv, sub_y, c1[1], c2[2], dev)
        c1loss = c1[0]; c2loss = c2[0]; c4loss = c4[0]
        discrete = jl[np.arange(len(ids)), best_t, best_v]
        violation = c4loss - discrete
        mean_violation = max(float(np.maximum(violation, 0).mean()), 0.0)
        max_violation = max(float(np.maximum(violation, 0).max()), 0.0)
        if mean_violation >= 1e-5 or max_violation >= 1e-3:
            raise RuntimeError(f"continuous oracle sanity failure {dataset}/{seed}/{split_name}: mean={mean_violation}, max={max_violation}")
        for label, result in (("C1_text_marginal", c1), ("C2_visual_marginal", c2), ("C4_joint", c4)):
            for audit in result[3]:
                audits.append({"dataset": dataset, "seed": seed, "split": split_name, "level": label, **audit,
                               "restart_loss_range_mean": float(np.ptp([a["mean_best_loss"] for a in result[3]]))})
        y_np = sub_y.cpu().numpy()
        base_labels = sorted(int(x) for x in np.unique(y_np))
        f1fn = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score
        for label, loss, at, av in (("marginal_continuous_pair", c3loss, c1[1], c2[2]),
                                    ("joint_continuous", c4loss, c4[1], c4[2])):
            _, acc, f1 = evaluate_mix(c0, head, sub_st, sub_sv, sub_y, at, av, dev)
            summaries.append({"dataset": dataset, "seed": seed, "split": split_name, "solution": label,
                              "mean_CE": float(np.mean(loss)), "accuracy_oracle": acc, "macro_f1_oracle": f1})
        _, c1acc, c1f1 = evaluate_mix(c0, head, sub_st, sub_sv, sub_y, c1[1], np.full_like(c1[2], .25), dev)
        _, c2acc, c2f1 = evaluate_mix(c0, head, sub_st, sub_sv, sub_y, np.full_like(c2[1], .25), c2[2], dev)
        summaries.extend([
            {"dataset": dataset, "seed": seed, "split": split_name, "solution": "uniform", "mean_CE": float(luu[ids].mean()), "accuracy_oracle": float(np.mean((head(c0._fuse(uniform_state_mix(sub_st), uniform_state_mix(sub_sv))).argmax(-1).cpu().numpy()) == y_np)), "macro_f1_oracle": np.nan},
            {"dataset": dataset, "seed": seed, "split": split_name, "solution": "best_joint_discrete", "mean_CE": float(discrete.mean()), "accuracy_oracle": np.nan, "macro_f1_oracle": np.nan},
            {"dataset": dataset, "seed": seed, "split": split_name, "solution": "C1_text_marginal", "mean_CE": float(c1loss.mean()), "accuracy_oracle": c1acc, "macro_f1_oracle": c1f1},
            {"dataset": dataset, "seed": seed, "split": split_name, "solution": "C2_visual_marginal", "mean_CE": float(c2loss.mean()), "accuracy_oracle": c2acc, "macro_f1_oracle": c2f1},
        ])
        geometry_rows = []
        for modality, alpha in (("text", c4[1]), ("visual", c4[2])):
            a = np.clip(alpha, 1e-12, 1.0)
            ent = -(a * np.log(a)).sum(1)
            near = np.abs(a[:, :, None] - np.eye(4)[None, :, :]).sum(1).min(1)
            active = (a > .10).sum(1)
            top2 = np.sort(a, axis=1)[:, -2:].sum(1)
            geom = {"dataset": dataset, "seed": seed, "split": split_name, "modality": modality,
                    "mean_entropy": float(ent.mean()), "mean_nearest_vertex_l1": float(near.mean()),
                    "mean_l1_to_uniform": float(np.abs(a - .25).sum(1).mean()),
                    "mean_active_order_count": float(active.mean()), "mean_top2_mass": float(top2.mean()),
                    "fraction_near_vertex_l1_lt_0_10": float((near < .10).mean()),
                    "fraction_ge_2_active_orders": float((active >= 2).mean()),
                    "fraction_ge_3_active_orders": float((active >= 3).mean())}
            geometries.append(geom); geometry_rows.append(geom)
        node_rows = pd.DataFrame({"dataset": dataset, "seed": seed, "split": split_name, "node_id": node_id[ids],
                                  "L_uniform": luu[ids], "L_best_discrete": discrete,
                                  "L_marginal_continuous_pair": c3loss, "L_joint_continuous": c4loss,
                                  "continuous_minus_discrete_gain": discrete - c4loss,
                                  "joint_continuous_minus_marginal_gain": c3loss - c4loss,
                                  "joint_oracle_gain": luu[ids] - c4loss,
                                  "optimizer_sanity_violation": violation})
        for modality, alpha in (("text", c4[1]), ("visual", c4[2])):
            for k in range(4): node_rows[f"alpha_{modality}_s{k}"] = alpha[:, k]
        raw.append(node_rows)
        # Per-run effect-size summary and optimizer audit.
        summaries.extend([
            {"dataset": dataset, "seed": seed, "split": split_name, "solution": "continuous_minus_discrete_gain", "mean_CE": float((discrete - c4loss).mean()), "accuracy_oracle": np.nan, "macro_f1_oracle": np.nan},
            {"dataset": dataset, "seed": seed, "split": split_name, "solution": "joint_continuous_minus_marginal_gain", "mean_CE": float((c3loss - c4loss).mean()), "accuracy_oracle": np.nan, "macro_f1_oracle": np.nan},
        ])
    raw_frame = pd.concat(raw, ignore_index=True)
    path = OUT / "phase_c_node_oracles" / f"{dataset}_seed{seed}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True); raw_frame.to_csv(path, index=False, compression="gzip")
    return raw_frame, summaries, geometries, audits


def phase_c(device):
    frames, summaries, geometries, audits = [], [], [], []
    for ds in DATASETS:
        for seed in SEEDS:
            f, s, g, a = phase_c_one(ds, seed, device)
            frames.append(f); summaries.extend(s); geometries.extend(g); audits.extend(a)
            print(f"[C] {ds} seed={seed}", flush=True)
    write_csv(RES / "phase_c_continuous_oracle_summary.csv", summaries)
    write_csv(RES / "phase_c_filter_geometry.csv", geometries)
    write_csv(RES / "phase_c_optimizer_audit.csv", audits)
    stab = []
    allraw = pd.concat(frames, ignore_index=True)
    for ds in DATASETS:
        for split in ("train", "validation"):
            per_seed = [allraw.loc[(allraw.dataset == ds) & (allraw.split == split) & (allraw.seed == seed)].set_index("node_id") for seed in SEEDS]
            for i, j in combinations(range(3), 2):
                q = per_seed[i].join(per_seed[j], lsuffix="_a", rsuffix="_b", how="inner")
                if q.empty: continue
                for modality in ("text", "visual"):
                    aa = q[[f"alpha_{modality}_s{k}_a" for k in range(4)]].to_numpy()
                    ab = q[[f"alpha_{modality}_s{k}_b" for k in range(4)]].to_numpy()
                    cos = np.sum(aa * ab, 1) / np.maximum(np.linalg.norm(aa, axis=1) * np.linalg.norm(ab, axis=1), 1e-12)
                    mid = .5 * (aa + ab)
                    js = .5 * (np.sum(aa * np.log(np.maximum(aa, 1e-12) / np.maximum(mid, 1e-12)), 1) + np.sum(ab * np.log(np.maximum(ab, 1e-12) / np.maximum(mid, 1e-12)), 1))
                    stab.append({"dataset": ds, "split": split, "modality": modality, "seed_a": SEEDS[i], "seed_b": SEEDS[j],
                                 "common_nodes": len(q), "alpha_cosine_similarity": float(cos.mean()),
                                 "alpha_js_divergence": float(js.mean()),
                                 "order_rank_agreement": float(np.mean(np.argmax(aa, 1) == np.argmax(ab, 1))),
                                 "distance_uniform_spearman": spearman(np.abs(aa - .25).sum(1), np.abs(ab - .25).sum(1)),
                                 "continuous_gain_spearman": spearman(q.joint_oracle_gain_a, q.joint_oracle_gain_b)})
    write_csv(RES / "phase_c_stability.csv", stab)
    violation = allraw.optimizer_sanity_violation.clip(lower=0)
    return {"mean_violation": float(violation.mean()), "max_violation": float(violation.max()),
            "stable_pair_count": len(stab), "filter_geometry": geometries}
