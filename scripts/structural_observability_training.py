from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from scripts import structural_observability_data as dataio
from src.models.canonical_routing_audit import CanonicalRoutingAudit, canonical_mix
from src.models.structural_action_ranker import (
    ACTION_NAMES, StructuralActionRanker, execute_hard_action, hard_action_bank,
    pairwise_ranking_loss, select_hard_action, weighted_pairwise_accuracy,
)

ROOT = Path(__file__).resolve().parents[1]
OUT, RES = dataio.OUT, dataio.RES
VARIANTS = dataio.VARIANTS


def _metrics(logits, labels, label_set):
    pred = logits.argmax(-1)
    acc = float((pred == labels).float().mean().item())
    f1 = float(f1_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy(),
                        labels=label_set, average="macro", zero_division=0))
    ce = float(F.cross_entropy(logits, labels).item())
    return {"accuracy": acc, "macro_f1": f1, "mean_CE": ce, "prediction": pred}


def _action_loss_views(ctx):
    matrix = ctx["matrix"]
    return matrix[:, :, 0], matrix[:, 0, :]


def global_action_choices(datasets=None) -> dict[str, dict[str, int]]:
    choices = {}
    rows = []
    datasets = dataio.SCREEN if datasets is None else list(datasets)
    for dataset in datasets:
        text_parts, visual_parts = [], []
        for seed in dataio.SEEDS:
            saved = np.load(dataio.E6_OUT / "phase_b_cache" / f"{dataset}_seed{seed}.npz", allow_pickle=False)
            matrix = dataio.action_matrix_from_e6(saved)
            train = saved["split"] == "train"
            text_parts.append(matrix[train, :, 0])
            visual_parts.append(matrix[train, 0, :])
        text_losses = np.concatenate(text_parts)
        visual_losses = np.concatenate(visual_parts)
        at, av = int(text_losses.mean(0).argmin()), int(visual_losses.mean(0).argmin())
        choices[dataset] = {"text": at, "visual": av}
        for modality, action, losses in (("text", at, text_losses), ("visual", av, visual_losses)):
            row = {"dataset": dataset, "modality": modality,
                   "global_best_action": ACTION_NAMES[action], "action_id": action,
                   "mean_train_CE": float(losses.mean(0)[action])}
            row.update({f"mean_train_CE_{ACTION_NAMES[k]}": float(losses.mean(0)[k]) for k in range(5)})
            rows.append(row)
    baseline_name = "global_action_baseline.csv" if datasets == dataio.SCREEN else "confirmation_global_action_baseline.csv"
    dataio.write_csv(RES / baseline_name, rows)
    return choices


def prepare_runtime_context(ctx: dict, global_actions: dict[str, dict[str, int]]) -> dict:
    device = ctx["device"]
    bank_t = hard_action_bank(ctx["states_text"])
    bank_v = hard_action_bank(ctx["states_visual"])
    d0_model, d0 = dataio._d0_model(ctx)
    alpha_t, alpha_v = d0["alpha_text"], d0["alpha_visual"]
    d0_action_t, d0_action_v = alpha_t.argmax(-1) + 1, alpha_v.argmax(-1) + 1
    global_t = torch.full((len(ctx["y"]),), global_actions[ctx["dataset"]]["text"], device=device, dtype=torch.long)
    global_v = torch.full((len(ctx["y"]),), global_actions[ctx["dataset"]]["visual"], device=device, dtype=torch.long)
    signatures_t, signatures_v = dataio.compute_decision_signatures(
        ctx["c0"], ctx["head"], ctx["states_text"], ctx["states_visual"]
    )
    return {
        **ctx, "bank_text": bank_t, "bank_visual": bank_v, "d0_model": d0_model, "d0_output": d0,
        "d0_action_text": d0_action_t, "d0_action_visual": d0_action_v,
        "global_action_text": global_t, "global_action_visual": global_v,
        "signatures_text": signatures_t, "signatures_visual": signatures_v,
    }


def _logits_for_actions(ctx, action_t, action_v, indices=None):
    if indices is None:
        indices = torch.arange(len(ctx["y"]), device=ctx["device"])
    return ctx["head"](ctx["c0"]._fuse(
        ctx["bank_text"][indices, action_t[indices]],
        ctx["bank_visual"][indices, action_v[indices]],
    ))


def _baseline_payloads(ctx):
    n, device = len(ctx["y"]), ctx["device"]
    zeros = torch.zeros(n, device=device, dtype=torch.long)
    candidates = {}
    for name, at, av, kind in [
        ("C0_Uniform", zeros, zeros, "uniform"),
        ("GLOBAL_ACTION", ctx["global_action_text"], ctx["global_action_visual"], "hard"),
        ("E6_D0_projected", ctx["d0_action_text"], ctx["d0_action_visual"], "hard"),
    ]:
        logits = _logits_for_actions(ctx, at, av)
        candidates[name] = {
            "actions_text": at, "actions_visual": av, "logits": logits,
            "scores_text": None, "scores_visual": None,
            "confidence_text": None, "confidence_visual": None, "execution": kind,
        }
    with torch.no_grad():
        zt = canonical_mix(ctx["states_text"], ctx["d0_output"]["alpha_text"])
        zv = canonical_mix(ctx["states_visual"], ctx["d0_output"]["alpha_visual"])
        d0_logits = ctx["head"](ctx["c0"]._fuse(zt, zv))
    candidates["E6_D0"] = {
        "actions_text": ctx["d0_action_text"], "actions_visual": ctx["d0_action_visual"],
        "logits": d0_logits, "scores_text": None, "scores_visual": None,
        "confidence_text": None, "confidence_visual": None, "execution": "soft_alpha_baseline",
    }
    return candidates


def _node_regret(ctx, actions_t, actions_v):
    matrix = ctx["matrix"]
    rows = torch.arange(len(ctx["y"]), device=ctx["device"])
    loss_t = matrix[rows, actions_t, 0]
    loss_v = matrix[rows, 0, actions_v]
    best_t, best_v = matrix[:, :, 0].min(1).values, matrix[:, 0, :].min(1).values
    joint = matrix[rows, actions_t, actions_v]
    joint_best = matrix.reshape(len(ctx["y"]), 25).min(1).values
    return loss_t - best_t, loss_v - best_v, joint - joint_best, joint_best


def _execution_rows(ctx, payloads) -> list[dict]:
    rows = []
    oracle_joint = ctx["matrix"].reshape(len(ctx["y"]), 25).min(1).values
    for candidate, payload in payloads.items():
        acts_t, acts_v = payload["actions_text"], payload["actions_visual"]
        actual_joint = ctx["matrix"][torch.arange(len(ctx["y"]), device=ctx["device"]), acts_t, acts_v]
        if candidate == "E6_D0" and payload.get("execution") == "soft_alpha_baseline":
            actual_joint = F.cross_entropy(payload["logits"], ctx["y"], reduction="none")
        joint_regret = actual_joint - oracle_joint
        for split, ids in (("train", ctx["train"]), ("validation", ctx["val"])):
            values = _metrics(payload["logits"][ids], ctx["y"][ids], ctx["labels"])
            rows.append({
                "candidate": candidate, "dataset": ctx["dataset"], "seed": ctx["seed"], "split": split,
                "val_acc": values["accuracy"] if split == "validation" else np.nan,
                "accuracy": values["accuracy"], "macro_f1": values["macro_f1"], "mean_CE": values["mean_CE"],
                "mean_joint_deployment_regret": float(joint_regret[ids].mean().item()),
                "median_joint_deployment_regret": float(joint_regret[ids].median().item()),
                "p90_joint_deployment_regret": float(torch.quantile(joint_regret[ids], .9).item()),
                "oracle_gain_captured_fraction": float(
                    ((ctx["matrix"][ids, 0, 0] - actual_joint[ids]) /
                     (ctx["matrix"][ids, 0, 0] - oracle_joint[ids]).clamp_min(1e-12)).mean().item()
                ),
                "test_metrics": None,
            })
    return rows


def _selection_rows(ctx, payloads) -> tuple[list[dict], list[dict]]:
    selection, pairwise = [], []
    losses_t, losses_v = _action_loss_views(ctx)
    for candidate, payload in payloads.items():
        for modality, losses, actions, score_key, confidence_key in [
            ("text", losses_t, payload["actions_text"], "scores_text", "confidence_text"),
            ("visual", losses_v, payload["actions_visual"], "scores_visual", "confidence_visual"),
        ]:
            scores = payload[score_key]
            confidence = payload[confidence_key]
            for split, ids in (("train", ctx["train"]), ("validation", ctx["val"])):
                selected = actions[ids]
                local_losses = losses[ids]
                oracle = local_losses.argmin(1)
                regret = local_losses[torch.arange(len(ids), device=ctx["device"]), selected] - local_losses.min(1).values
                row = {
                    "candidate": candidate, "dataset": ctx["dataset"], "seed": ctx["seed"],
                    "split": split, "modality": modality,
                    "top1_oracle_action_agreement": float((selected == oracle).float().mean().item()),
                    "selection_regret_mean": float(regret.mean().item()),
                    "selection_regret_median": float(regret.median().item()),
                    "selection_regret_p90": float(torch.quantile(regret, .9).item()),
                    "oracle_gain_captured_fraction": float(
                        ((local_losses[:, 0] - local_losses[torch.arange(len(ids), device=ctx["device"]), selected]) /
                         (local_losses[:, 0] - local_losses.min(1).values).clamp_min(1e-12)).mean().item()
                    ),
                    "selected_action": ACTION_NAMES[int(torch.mode(selected).values.item())],
                }
                for aid, action in enumerate(ACTION_NAMES):
                    row[f"selected_fraction_{action}"] = float((selected == aid).float().mean().item())
                selection.append(row)
                if scores is not None:
                    pairwise.append({
                        "candidate": candidate, "dataset": ctx["dataset"], "seed": ctx["seed"],
                        "split": split, "modality": modality,
                        "weighted_pairwise_accuracy": float(weighted_pairwise_accuracy(scores[ids], local_losses).item()),
                        "top1_oracle_action_agreement": row["top1_oracle_action_agreement"],
                    })
    return selection, pairwise


def _prediction_frame(ctx, candidate: str, payload: dict) -> pd.DataFrame:
    frame = pd.read_csv(OUT / "phase_a_action_landscape_nodes" / f"{ctx['dataset']}_seed{ctx['seed']}.csv.gz")
    text = frame[frame.modality == "text"].set_index("node_id")
    visual = frame[frame.modality == "visual"].set_index("node_id")
    ids = ctx["node_id"]
    at = payload["actions_text"].detach().cpu().numpy()
    av = payload["actions_visual"].detach().cpu().numpy()
    logits = payload["logits"].detach()
    pred = logits.argmax(-1).cpu().numpy()
    correct = (logits.argmax(-1) == ctx["y"]).detach().cpu().numpy()
    if candidate == "E6_D0":
        actual_joint = F.cross_entropy(logits, ctx["y"], reduction="none").cpu().numpy()
    else:
        actual_joint = ctx["matrix"][torch.arange(len(ctx["y"]), device=ctx["device"]),
                                      payload["actions_text"], payload["actions_visual"]].detach().cpu().numpy()
    best_joint = ctx["matrix"].reshape(len(ctx["y"]), 25).min(1).values.detach().cpu().numpy()
    rows = []
    for i, node in enumerate(ids):
        trow, vrow = text.loc[int(node)], visual.loc[int(node)]
        margin_joint = 0.5 * (float(trow.oracle_margin) + float(vrow.oracle_margin))
        rows.append({
            "candidate": candidate, "dataset": ctx["dataset"], "seed": ctx["seed"],
            "split": ctx["split"][i], "node_id": int(node), "predicted_class": int(pred[i]),
            "joint_prediction_correct": bool(correct[i]), "selected_action_text": ACTION_NAMES[int(at[i])],
            "selected_action_visual": ACTION_NAMES[int(av[i])],
            "oracle_action_text": trow.oracle_best_action, "oracle_action_visual": vrow.oracle_best_action,
            "oracle_margin_text": float(trow.oracle_margin), "oracle_margin_visual": float(vrow.oracle_margin),
            "oracle_gain_text": float(trow.oracle_gain_vs_uniform), "oracle_gain_visual": float(vrow.oracle_gain_vs_uniform),
            "margin_quartile_text": int(trow.margin_quartile), "margin_quartile_visual": int(vrow.margin_quartile),
            "joint_oracle_margin": margin_joint,
            "joint_deployment_regret": float(actual_joint[i] - best_joint[i]),
            **({f"score_text_{ACTION_NAMES[j]}": float(payload["scores_text"][i, j].item()) for j in range(5)}
               if payload["scores_text"] is not None else {}),
            **({f"score_visual_{ACTION_NAMES[j]}": float(payload["scores_visual"][i, j].item()) for j in range(5)}
               if payload["scores_visual"] is not None else {}),
            "confidence_text": float(payload["confidence_text"][i, 0].item()) if payload["confidence_text"] is not None else np.nan,
            "confidence_visual": float(payload["confidence_visual"][i, 0].item()) if payload["confidence_visual"] is not None else np.nan,
            "confidence_uniform_text": float(payload["confidence_text"][i, 1].item()) if payload["confidence_text"] is not None else np.nan,
            "confidence_uniform_visual": float(payload["confidence_visual"][i, 1].item()) if payload["confidence_visual"] is not None else np.nan,
        })
    return pd.DataFrame(rows)


def _quartile_rows(ctx, candidate, payload):
    frame = _prediction_frame(ctx, candidate, payload)
    matrix = ctx["matrix"]
    rows = []
    local_by_node = {int(node): i for i, node in enumerate(ctx["node_id"])}
    frame_by_node = frame.set_index("node_id")
    losses = {
        "text": matrix[:, :, 0],
        "visual": matrix[:, 0, :],
    }
    for modality in ("text", "visual"):
        qcol = f"margin_quartile_{modality}"
        acol = f"selected_action_{modality}"
        ocol = f"oracle_action_{modality}"
        for split in ("train", "validation"):
            part = frame[frame.split == split]
            ids_full = np.flatnonzero(ctx["split"] == split)
            part = part.set_index("node_id")
            local = torch.as_tensor(ids_full, device=ctx["device"])
            for q in range(1, 5):
                node_ids = frame.loc[(frame.split == split) & (frame[qcol] == q), "node_id"].to_numpy()
                if not len(node_ids):
                    continue
                local_ids = np.asarray([local_by_node[int(node)] for node in node_ids], dtype=np.int64)
                ix = torch.as_tensor(local_ids, device=ctx["device"], dtype=torch.long)
                selected_names = frame_by_node.loc[node_ids, acol].to_numpy()
                actions = torch.as_tensor([ACTION_NAMES.index(name) for name in selected_names], device=ctx["device"])
                lmat = losses[modality][ix]
                oracle = lmat.argmin(1)
                regret = lmat[torch.arange(len(ix), device=ctx["device"]), actions] - lmat.min(1).values
                gain_denom = (lmat[:, 0] - lmat.min(1).values).clamp_min(1e-12)
                captured = (lmat[:, 0] - lmat[torch.arange(len(ix), device=ctx["device"]), actions]) / gain_denom
                selected_label = frame_by_node.loc[node_ids, acol].to_numpy()
                oracle_label = frame_by_node.loc[node_ids, ocol].to_numpy()
                rows.append({
                    "candidate": candidate, "dataset": ctx["dataset"], "seed": ctx["seed"],
                    "split": split, "modality": modality, "margin_quartile": q, "node_count": len(ix),
                    "mean_oracle_margin": float(frame_by_node.loc[node_ids, f"oracle_margin_{modality}"].mean()),
                    "top1_oracle_action_agreement": float(np.mean(selected_label == oracle_label)),
                    "mean_selection_regret": float(regret.mean().item()),
                    "mean_oracle_gain_captured_fraction": float(captured.mean().item()),
                })
    # Joint outcomes are stratified by mean of the two marginal oracle margins.
    for split in ("train", "validation"):
        part = frame[frame.split == split].copy()
        part["joint_margin_quartile"] = dataio._stratified_quartile(part.joint_oracle_margin.to_numpy())
        for q in range(1, 5):
            rows_q = part[part.joint_margin_quartile == q]
            node_ids = rows_q.node_id.to_numpy()
            if not len(node_ids):
                continue
            local_ids = np.asarray([local_by_node[int(node)] for node in node_ids], dtype=np.int64)
            ix = torch.as_tensor(local_ids, device=ctx["device"], dtype=torch.long)
            labels = ctx["y"][ix]
            pred = torch.as_tensor(np.array(rows_q.predicted_class.to_numpy(), copy=True), device=ctx["device"], dtype=torch.long)
            f1 = float(f1_score(labels.cpu().numpy(), pred.cpu().numpy(), labels=ctx["labels"], average="macro", zero_division=0))
            joint_at = torch.as_tensor([ACTION_NAMES.index(t) for t in rows_q.selected_action_text], device=ctx["device"], dtype=torch.long)
            joint_av = torch.as_tensor([ACTION_NAMES.index(v) for v in rows_q.selected_action_visual], device=ctx["device"], dtype=torch.long)
            joint_loss = ctx["matrix"][ix, joint_at, joint_av]
            if candidate == "E6_D0":
                joint_loss = F.cross_entropy(payload["logits"][ix], labels, reduction="none")
            best = ctx["matrix"][ix].reshape(len(ix), 25).min(1).values
            uniform = ctx["matrix"][ix, 0, 0]
            captured = ((uniform - joint_loss) / (uniform - best).clamp_min(1e-12)).mean()
            rows.append({
                "candidate": candidate, "dataset": ctx["dataset"], "seed": ctx["seed"],
                "split": split, "modality": "joint", "margin_quartile": q, "node_count": len(ix),
                "mean_oracle_margin": float(rows_q.joint_oracle_margin.mean()),
                "joint_accuracy": float((pred == labels).float().mean().item()),
                "joint_macro_f1": f1, "mean_joint_deployment_regret": float((joint_loss - best).mean().item()),
                "mean_oracle_gain_captured_fraction": float(captured.item()),
            })
    return rows, frame


def _confidence_rows(ctx, payload):
    rows = []
    losses_t, losses_v = _action_loss_views(ctx)
    for modality, scores, conf, actions, losses in [
        ("text", payload["scores_text"], payload["confidence_text"], payload["actions_text"], losses_t),
        ("visual", payload["scores_visual"], payload["confidence_visual"], payload["actions_visual"], losses_v),
    ]:
        ids = ctx["val"]
        local_conf = conf[ids, 0].detach().cpu().numpy()
        local_uniform_conf = conf[ids, 1].detach().cpu().numpy()
        local_actions = actions[ids]
        local_losses = losses[ids]
        oracle = local_losses.argmin(1)
        regret = local_losses[torch.arange(len(ids), device=ctx["device"]), local_actions] - local_losses.min(1).values
        q = dataio._stratified_quartile(local_conf)
        high = regret[torch.as_tensor(q == 4, device=ctx["device"])]
        low = regret[torch.as_tensor(q == 1, device=ctx["device"])]
        rows.append({
            "dataset": ctx["dataset"], "seed": ctx["seed"], "modality": modality,
            "mean_confidence": float(np.mean(local_conf)),
            "mean_confidence_vs_uniform": float(np.mean(local_uniform_conf)),
            "spearman_confidence_vs_negative_regret": dataio._spearman(local_conf, -regret.detach().cpu().numpy()),
            "high_confidence_q4_regret": float(high.mean().item()),
            "low_confidence_q1_regret": float(low.mean().item()),
            "high_confidence_q4_oracle_agreement": float((local_actions[torch.as_tensor(q == 4, device=ctx["device"])] == oracle[torch.as_tensor(q == 4, device=ctx["device"])]).float().mean().item()),
            "low_confidence_q1_oracle_agreement": float((local_actions[torch.as_tensor(q == 1, device=ctx["device"])] == oracle[torch.as_tensor(q == 1, device=ctx["device"])]).float().mean().item()),
        })
    return rows


def _save_variant_checkpoint(model, ctx, variant, best_epoch, history, score_reload_error):
    path = OUT / "checkpoints" / variant / f"{ctx['dataset']}_seed{ctx['seed']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant, "mode": model.mode, "dataset": ctx["dataset"], "seed": ctx["seed"],
        "hidden_dim": ctx["states_text"][0].shape[-1], "best_epoch": int(best_epoch),
        "router_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "c0_sha256": ctx["c0_sha256"], "trainable_parameter_count": model.trainable_parameter_count,
        "selection": "Validation joint-execution Accuracy primary, Macro-F1 secondary",
        "checkpoint_reload_score_error": float(score_reload_error), "test_metrics": None,
    }
    torch.save(payload, path)
    run_dir = OUT / "runs" / variant / ctx["dataset"] / f"seed{ctx['seed']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    (run_dir / "complete.marker").write_text("complete\n", encoding="utf-8")
    return path


def train_variant(ctx, variant: str, mode: str) -> tuple[dict, dict, list, list, list]:
    device = ctx["device"]
    signature_t = ctx["signatures_text"] if mode == "r1_decision" else None
    signature_v = ctx["signatures_visual"] if mode == "r1_decision" else None
    seed_offset = {"R0_trajectory_ranker": 81011, "R1_capacity_control": 81017, "R1_decision_ranker": 81023}[variant]
    torch.manual_seed(ctx["seed"] + seed_offset)
    model = StructuralActionRanker(ctx["states_text"][0].shape[-1], mode).to(device)
    losses_t, losses_v = _action_loss_views(ctx)
    best_key, best_state, best_epoch, patience = (-1.0, -1.0), None, 0, 30
    history = []
    run_dir = OUT / "runs" / variant / ctx["dataset"] / f"seed{ctx['seed']}"
    checkpoint_path = OUT / "checkpoints" / variant / f"{ctx['dataset']}_seed{ctx['seed']}.pt"
    marker = run_dir / "complete.marker"
    if marker.exists() and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (saved.get("dataset") != ctx["dataset"] or int(saved.get("seed", -1)) != ctx["seed"]
                or saved.get("variant") != variant or saved.get("mode") != mode
                or saved.get("c0_sha256") != ctx["c0_sha256"]):
            raise RuntimeError(f"completed fit checkpoint identity mismatch: {variant}/{ctx['dataset']}/{ctx['seed']}")
        best_state = saved["router_state"]
        best_epoch = int(saved["best_epoch"])
        history_path = run_dir / "training_history.csv"
        history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
        model.load_state_dict(best_state, strict=True)
        model.eval()
        with torch.no_grad():
            out = model(ctx["states_text"], ctx["states_visual"], signature_t, signature_v)
            at, av = select_hard_action(out["scores_text"]), select_hard_action(out["scores_visual"])
            metric = _metrics(_logits_for_actions(ctx, at, av)[ctx["val"]], ctx["y"][ctx["val"]], ctx["labels"])
        best_key = (metric["accuracy"], metric["macro_f1"])
        print(f"[RESUME] {variant} {ctx['dataset']} seed={ctx['seed']} checkpoint verified", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(1, 201):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(ctx["states_text"], ctx["states_visual"], signature_t, signature_v)
            loss_t = pairwise_ranking_loss(output["scores_text"], losses_t, ctx["train"])
            loss_v = pairwise_ranking_loss(output["scores_visual"], losses_v, ctx["train"])
            loss = loss_t + loss_v
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                out = model(ctx["states_text"], ctx["states_visual"], signature_t, signature_v)
                actions_t = select_hard_action(out["scores_text"])
                actions_v = select_hard_action(out["scores_visual"])
                logits = _logits_for_actions(ctx, actions_t, actions_v)
                metric = _metrics(logits[ctx["val"]], ctx["y"][ctx["val"]], ctx["labels"])
            key = (metric["accuracy"], metric["macro_f1"])
            history.append({
                "epoch": epoch, "train_pairwise_loss_text": float(loss_t.item()),
                "train_pairwise_loss_visual": float(loss_v.item()), "train_pairwise_loss": float(loss.item()),
                "val_joint_accuracy": metric["accuracy"], "val_joint_macro_f1": metric["macro_f1"],
                "val_joint_CE": metric["mean_CE"],
            })
            if key > best_key:
                best_key, best_epoch = key, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                patience = 30
            else:
                patience -= 1
            if epoch >= 30 and patience <= 0:
                break
        if best_state is None:
            raise RuntimeError(f"No validation-selected model: {variant}/{ctx['dataset']}/{ctx['seed']}")
        model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        expected = model(ctx["states_text"], ctx["states_visual"], signature_t, signature_v)
    reload_model = StructuralActionRanker(ctx["states_text"][0].shape[-1], mode).to(device)
    reload_model.load_state_dict(best_state, strict=True)
    reload_model.eval()
    with torch.no_grad():
        reloaded = reload_model(ctx["states_text"], ctx["states_visual"], signature_t, signature_v)
    reload_error = max(
        float((expected["scores_text"] - reloaded["scores_text"]).abs().max().item()),
        float((expected["scores_visual"] - reloaded["scores_visual"]).abs().max().item()),
    )
    if reload_error >= 1e-7:
        raise RuntimeError(f"ranker checkpoint reload mismatch {variant}/{ctx['dataset']}/{ctx['seed']}: {reload_error}")
    path = _save_variant_checkpoint(model, ctx, variant, best_epoch, history, reload_error)
    with torch.no_grad():
        final = model(ctx["states_text"], ctx["states_visual"], signature_t, signature_v)
        actions_t = select_hard_action(final["scores_text"])
        actions_v = select_hard_action(final["scores_visual"])
        logits = _logits_for_actions(ctx, actions_t, actions_v)
    payloads = {
        **_baseline_payloads(ctx),
        variant: {
            "actions_text": actions_t, "actions_visual": actions_v, "logits": logits,
            "scores_text": final["scores_text"], "scores_visual": final["scores_visual"],
            "confidence_text": final["confidence_text"], "confidence_visual": final["confidence_visual"],
            "execution": "hard_action",
        },
    }
    ranker = payloads[variant]
    result = {
        "variant": variant, "dataset": ctx["dataset"], "seed": ctx["seed"],
        "val_accuracy": best_key[0], "val_macro_f1": best_key[1],
        "val_mean_CE": float(F.cross_entropy(ranker["logits"][ctx["val"]], ctx["y"][ctx["val"]]).item()),
        "best_epoch": best_epoch, "trainable_parameter_count": model.trainable_parameter_count,
        "checkpoint_reload_score_error": reload_error, "checkpoint": str(path.relative_to(ROOT)),
        "test_metrics": None,
    }
    execution_rows = _execution_rows(ctx, {name: payload for name, payload in payloads.items() if name in (
        "C0_Uniform", "GLOBAL_ACTION", "E6_D0", "E6_D0_projected", variant
    )})
    selection_rows, pairwise_rows = _selection_rows(ctx, payloads)
    # Shared baselines are emitted once by train_screening; retain this fit only.
    selection_rows = [row for row in selection_rows if row["candidate"] == variant]
    pairwise_rows = [row for row in pairwise_rows if row["candidate"] == variant]
    confidence_rows = _confidence_rows(ctx, ranker) if variant == "R1_decision_ranker" else []
    coverage_rows = _coverage_from_payloads(ctx, ranker, payloads["C0_Uniform"]) if variant == "R1_decision_ranker" else []
    strat_rows, node_frame = _quartile_rows(ctx, variant, ranker)
    raw_path = OUT / "ranker_node_predictions" / f"{ctx['dataset']}_seed{ctx['seed']}_{variant}.csv.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    node_frame.to_csv(raw_path, index=False, compression="gzip")
    return result, payloads, execution_rows, selection_rows + pairwise_rows, confidence_rows + coverage_rows + strat_rows


def confidence_coverage_thresholds(train_confidence, coverages=(0.25, 0.50, 0.75, 1.00)) -> dict[float, float]:
    values = np.asarray(train_confidence, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("train confidence must be a nonempty one-dimensional array")
    if any(not 0 < float(target) <= 1 for target in coverages):
        raise ValueError("coverage targets must lie in (0, 1]")
    return {float(target): float(np.quantile(values, 1 - float(target))) for target in coverages}


def _coverage_from_payloads(ctx, ranker, uniform_payload):
    train_conf = torch.minimum(ranker["confidence_text"][ctx["train"], 0], ranker["confidence_visual"][ctx["train"], 0]).detach().cpu().numpy()
    val_conf = torch.minimum(ranker["confidence_text"][ctx["val"], 0], ranker["confidence_visual"][ctx["val"], 0])
    metrics_u = _metrics(uniform_payload["logits"][ctx["val"]], ctx["y"][ctx["val"]], ctx["labels"])
    rows = []
    thresholds = confidence_coverage_thresholds(train_conf)
    oracle_joint = ctx["matrix"].reshape(len(ctx["y"]), 25).min(1).values
    for target in (.25, .50, .75, 1.00):
        threshold = thresholds[target]
        take = val_conf >= threshold
        at, av = ranker["actions_text"][ctx["val"]].clone(), ranker["actions_visual"][ctx["val"]].clone()
        at[~take], av[~take] = 0, 0
        logits = ctx["head"](ctx["c0"]._fuse(ctx["bank_text"][ctx["val"], at], ctx["bank_visual"][ctx["val"], av]))
        metric = _metrics(logits, ctx["y"][ctx["val"]], ctx["labels"])
        actual = ctx["matrix"][ctx["val"], at, av]
        reg = actual - oracle_joint[ctx["val"]]
        rows.append({
            "candidate": "R1_decision_ranker", "dataset": ctx["dataset"], "seed": ctx["seed"],
            "nominal_train_coverage": target, "train_confidence_threshold": threshold,
            "actual_validation_coverage": float(take.float().mean().item()),
            "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"], "mean_CE": metric["mean_CE"],
            "delta_acc_vs_uniform": metric["accuracy"] - metrics_u["accuracy"],
            "delta_macro_f1_vs_uniform": metric["macro_f1"] - metrics_u["macro_f1"],
            "mean_joint_deployment_regret": float(reg.mean().item()),
            "mean_joint_regret_routed": float(reg[take].mean().item()) if take.any() else np.nan,
            "mean_joint_regret_fallback": float(reg[~take].mean().item()) if (~take).any() else np.nan,
        })
    return rows


def train_screening(device_name: str = "cuda:0") -> dict:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E7-CSAO router fits require a CUDA device")
    global_actions = global_action_choices()
    ranker_rows, execution_rows, selection_rows, pairwise_rows = [], [], [], []
    confidence_rows, coverage_rows, stratification_rows = [], [], []
    baseline_seen = set()
    for dataset in dataio.SCREEN:
        for seed in dataio.SEEDS:
            ctx = prepare_runtime_context(
                dataio._load_context(dataset, seed, device, verify_actions=False), global_actions
            )
            baseline_payload = _baseline_payloads(ctx)
            # Emit shared baselines once per dataset/seed, not once per variant.
            if (dataset, seed) not in baseline_seen:
                execution_rows.extend(_execution_rows(ctx, baseline_payload))
                base_sel, _ = _selection_rows(ctx, baseline_payload)
                selection_rows.extend(base_sel)
                for candidate, payload in baseline_payload.items():
                    qrows, node_frame = _quartile_rows(ctx, candidate, payload)
                    stratification_rows.extend(qrows)
                    raw_path = OUT / "ranker_node_predictions" / f"{dataset}_seed{seed}_{candidate}.csv.gz"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    node_frame.to_csv(raw_path, index=False, compression="gzip")
                baseline_seen.add((dataset, seed))
            signatures_ready = ctx
            for variant, mode in VARIANTS:
                # Signatures are label-free and reused for both R1 variants' context.
                result, payloads, exrows, srrows, auxiliary = train_variant(signatures_ready, variant, mode)
                ranker_rows.append(result)
                execution_rows.extend(row for row in exrows if row["candidate"] == variant)
                # The helper returns selection and pairwise rows concatenated.
                selection_rows.extend(row for row in srrows if row.get("candidate") == variant and "selection_regret_mean" in row)
                pairwise_rows.extend(row for row in srrows if "weighted_pairwise_accuracy" in row)
                confidence_rows.extend(row for row in auxiliary if "spearman_confidence_vs_negative_regret" in row)
                coverage_rows.extend(row for row in auxiliary if "nominal_train_coverage" in row)
                stratification_rows.extend(row for row in auxiliary if "margin_quartile" in row and "oracle_margin_spearman" not in row)
                print(f"[B] {variant} {dataset} seed={seed} best_epoch={result['best_epoch']}", flush=True)
                torch.cuda.empty_cache()
            del ctx
    write_csv = dataio.write_csv
    write_csv(dataio.RES / "phase_b_ranker_results.csv", ranker_rows)
    write_csv(dataio.RES / "phase_b_joint_execution.csv", execution_rows)
    write_csv(dataio.RES / "phase_b_selection_regret.csv", selection_rows)
    write_csv(dataio.RES / "phase_b_pairwise_metrics.csv", pairwise_rows)
    cap = build_capacity_comparison(ranker_rows, selection_rows, execution_rows)
    write_csv(dataio.RES / "phase_b_capacity_control.csv", cap)
    write_csv(dataio.RES / "phase_c_confidence_analysis.csv", confidence_rows)
    write_csv(dataio.RES / "phase_c_coverage_curve.csv", coverage_rows)
    base = pd.read_csv(dataio.RES / "phase_a_margin_stratification.csv")
    ranked = pd.DataFrame(stratification_rows)
    combined = pd.concat([base, ranked], ignore_index=True, sort=False)
    write_csv(dataio.RES / "phase_a_margin_stratification.csv", combined.to_dict("records"))
    return {
        "ranker_rows": ranker_rows, "execution_rows": execution_rows,
        "selection_rows": selection_rows, "pairwise_rows": pairwise_rows,
        "confidence_rows": confidence_rows, "coverage_rows": coverage_rows,
        "stratification_rows": stratification_rows,
    }


def build_capacity_comparison(ranker_rows, selection_rows, execution_rows):
    rankers = pd.DataFrame(ranker_rows)
    reg = pd.DataFrame(selection_rows)
    ex = pd.DataFrame(execution_rows)
    output = []
    for dataset in dataio.SCREEN:
        for seed in dataio.SEEDS:
            def ranker_value(name, column):
                sub = rankers[(rankers.dataset == dataset) & (rankers.seed == seed) & (rankers.variant == name)]
                return float(sub.iloc[0][column]) if len(sub) else np.nan
            def regret(name):
                sub = reg[(reg.dataset == dataset) & (reg.seed == seed) & (reg.candidate == name) & (reg.split == "validation")]
                return float(sub.selection_regret_mean.mean()) if len(sub) else np.nan
            output.append({
                "dataset": dataset, "seed": seed,
                "R1_joint_accuracy": ranker_value("R1_decision_ranker", "val_accuracy"),
                "capacity_joint_accuracy": ranker_value("R1_capacity_control", "val_accuracy"),
                "delta_joint_accuracy_R1_minus_capacity": ranker_value("R1_decision_ranker", "val_accuracy") - ranker_value("R1_capacity_control", "val_accuracy"),
                "R1_selection_regret": regret("R1_decision_ranker"),
                "capacity_selection_regret": regret("R1_capacity_control"),
                "delta_selection_regret_capacity_minus_R1": regret("R1_capacity_control") - regret("R1_decision_ranker"),
                "parameter_count_R1": ranker_value("R1_decision_ranker", "trainable_parameter_count"),
                "parameter_count_capacity": ranker_value("R1_capacity_control", "trainable_parameter_count"),
            })
    return output


def write_screening_raw_placeholder():
    pass


def main():
    raise RuntimeError("Call preflight_screening then train_screening from the experiment driver")
