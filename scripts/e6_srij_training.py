from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts import run_routing_identifiability as common
from scripts import run_utility_routing_audit as e5
from src.models.canonical_routing_audit import CanonicalRoutingAudit, canonical_mix

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/routing_identifiability_v1"


def train_only_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, train_idx: torch.Tensor):
    """Training objective restricted to Train rows; Validation is selection-only."""
    return F.cross_entropy(logits[train_idx], labels[train_idx])


def train_router_one(dataset: str, seed: int, device: str, mode: str = "independent") -> dict:
    dev = torch.device(device)
    _, data, _, c0, head = common._load_c0(dataset, seed, dev)
    allowed, y, train, val = common._allowed_split(data, dev)
    with torch.no_grad():
        analysis = c0.analyze(data.x.to(dev), data.edge_index.to(dev))
        # Match C0's full-graph fusion before slicing Train/Validation rows.
        # Fusing only the subset can change GEMM rounding on large graphs.
        alpha_u_full = torch.full((data.num_nodes, 4), .25, device=dev)
        init = c0._fuse(canonical_mix(analysis["S_text"], alpha_u_full),
                        canonical_mix(analysis["S_visual"], alpha_u_full))
        init_error = float((init - analysis["fused_z"]).abs().max().item())
        st = [s[allowed] for s in analysis["S_text"]]
        sv = [s[allowed] for s in analysis["S_visual"]]
        alpha_u = torch.full((len(y), 4), .25, device=dev)
        if init_error >= 1e-6:
            raise RuntimeError(f"zero-init D0 does not reproduce C0: {dataset}/{seed}: {init_error}")
    runname = {"independent":"D0_independent", "joint":"D1_joint", "capacity_control":"D1_capacity_control"}[mode]
    checkpoint = OUTPUT / "checkpoints" / runname / f"{dataset}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted(int(v) for v in torch.unique(y).cpu().tolist())
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = CanonicalRoutingAudit(256, mode).to(dev)
        model.load_state_dict(saved["router_state"], strict=True)
        best_epoch, history = int(saved["best_epoch"]), saved.get("history", [])
    else:
        torch.manual_seed(seed + {"independent": 21031, "joint": 21041, "capacity_control": 21051}[mode])
        model = CanonicalRoutingAudit(256, mode).to(dev)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_acc, best_state, best_epoch, patience = -1.0, None, 0, 30
        history = []
        for epoch in range(1, 201):
            model.train(); optimizer.zero_grad(set_to_none=True)
            out = model(st, sv)
            logits = head(c0._fuse(canonical_mix(st, out["alpha_text"]), canonical_mix(sv, out["alpha_visual"])))
            loss = train_only_cross_entropy(logits, y, train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                measured = model(st, sv)
                val_logits = head(c0._fuse(canonical_mix(st, measured["alpha_text"]), canonical_mix(sv, measured["alpha_visual"])))[val]
                acc, f1, _ = common._metric_pair(val_logits, y[val], labels)
            history.append({"epoch": epoch, "train_ce": float(loss.item()), "val_acc": acc, "val_macro_f1": f1})
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 30
            else:
                patience -= 1
            if epoch >= 30 and patience <= 0:
                break
        if best_state is None:
            raise RuntimeError(f"no best-validation checkpoint: {dataset}/{seed}/{mode}")
        model.load_state_dict(best_state, strict=True)
        torch.save({"mode": mode, "dataset": dataset, "seed": seed, "best_epoch": best_epoch,
                    "router_state": best_state, "history": history,
                    "c0_sha256": e5.sha256(e5.c0_checkpoint(dataset, seed)),
                    "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
                    "selection": "best Validation Accuracy", "test_metrics": None}, checkpoint)
        directory = OUTPUT / "runs" / runname / dataset / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
        (directory / "complete.marker").write_text("complete\n", encoding="utf-8")
    model.eval()
    with torch.no_grad(): out = model(st, sv)
    val_logits = head(c0._fuse(canonical_mix(st, out["alpha_text"]), canonical_mix(sv, out["alpha_visual"])))[val]
    acc, f1, pred = common._metric_pair(val_logits, y[val], labels)
    baseline_logits = head(c0._fuse(canonical_mix(st, alpha_u), canonical_mix(sv, alpha_u)))[val]
    baseline_acc, baseline_f1, baseline_pred = common._metric_pair(baseline_logits, y[val], labels)
    reload_model = CanonicalRoutingAudit(256, mode).to(dev)
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(reloaded["router_state"], strict=True); reload_model.eval()
    with torch.no_grad(): reload_out = reload_model(st, sv)
    reload_error = max((reload_out["alpha_text"] - out["alpha_text"]).abs().max().item(),
                       (reload_out["alpha_visual"] - out["alpha_visual"]).abs().max().item())
    if reload_error >= 1e-7:
        raise RuntimeError(f"router checkpoint reload mismatch {dataset}/{seed}/{mode}: {reload_error}")
    row = {"variant": runname, "dataset": dataset, "seed": seed, "val_acc": acc, "val_macro_f1": f1,
           "c0_uniform_val_acc": baseline_acc, "c0_uniform_val_macro_f1": baseline_f1,
           "delta_acc_vs_uniform": acc - baseline_acc, "delta_macro_f1_vs_uniform": f1 - baseline_f1,
           "best_epoch": best_epoch, "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
           "zero_init_embedding_error": init_error, "checkpoint_reload_alpha_error": reload_error,
           "test_metrics": None, "test_labels_indexed": False}
    return {"dataset": dataset, "seed": seed, "mode": mode, "model": model, "c0": c0, "head": head,
            "y": y, "train": train, "val": val, "states_text": st, "states_visual": sv,
            "output": out, "prediction": pred, "baseline_prediction": baseline_pred,
            "row": row, "labels": labels, "checkpoint": checkpoint}


def router_interventions(run: dict, include_companion: bool = False) -> list[dict]:
    model, c0, head = run["model"], run["c0"], run["head"]
    st, sv, y = run["states_text"], run["states_visual"], run["y"]
    train, val, labels = run["train"], run["val"], run["labels"]
    with torch.no_grad(): normal = model(st, sv)
    at, av = normal["alpha_text"].clone(), normal["alpha_visual"].clone()
    specs = [("normal", at, av, False)]
    mt, mv = at.clone(), av.clone()
    mt[val], mv[val] = at[train].mean(0), av[train].mean(0)
    specs.append(("train_mean_alpha", mt, mv, False))
    for i in range(10):
        gen = torch.Generator(device="cpu").manual_seed(run["seed"] * 10007 + i)
        perm = torch.randperm(val.numel(), generator=gen).to(val.device)
        qt, qv = at.clone(), av.clone()
        qt[val], qv[val] = at[val[perm]], av[val[perm]]
        specs.append((f"node_shuffle_{i}", qt, qv, False))
    specs.append(("uniform_alpha", torch.full_like(at, .25), torch.full_like(av, .25), False))
    if include_companion and run["mode"] == "joint":
        ct, cv = normal["context_text"], normal["context_visual"]
        for i in range(10):
            gen = torch.Generator(device="cpu").manual_seed(run["seed"] * 20011 + i)
            perm = torch.randperm(val.numel(), generator=gen).to(val.device)
            shuffle_v, shuffle_t = cv.clone(), ct.clone()
            shuffle_v[val], shuffle_t[val] = cv[val[perm]], ct[val[perm]]
            with torch.no_grad():
                out_t = model(st, sv, companion_visual=shuffle_v)
                out_v = model(st, sv, companion_text=shuffle_t)
            specs.append((f"shuffle_visual_companion_for_text_{i}", out_t["alpha_text"], av, True))
            specs.append((f"shuffle_text_companion_for_visual_{i}", at, out_v["alpha_visual"], True))
    rows = []
    for name, qt, qv, is_companion in specs:
        with torch.no_grad():
            logits = head(c0._fuse(canonical_mix(st, qt), canonical_mix(sv, qv)))[val]
        acc, f1, pred = common._metric_pair(logits, y[val], labels)
        rows.append({"variant": {"independent":"D0_independent", "joint":"D1_joint", "capacity_control":"D1_capacity_control"}[run["mode"]],
                     "dataset": run["dataset"], "seed": run["seed"], "intervention": name,
                     "val_acc": acc, "val_macro_f1": f1,
                     "delta_acc_vs_normal": acc - run["row"]["val_acc"],
                     "delta_macro_f1_vs_normal": f1 - run["row"]["val_macro_f1"],
                     "prediction_flip_fraction": float((pred != run["prediction"]).float().mean().item()),
                     "mean_abs_alpha_change": .5 * ((qt[val] - at[val]).abs().mean().item() + (qv[val] - av[val]).abs().mean().item()),
                     "companion_shuffle_only": is_companion, "test_metrics": None})
    return rows
