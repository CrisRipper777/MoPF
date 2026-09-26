from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, roc_auc_score

from src.data import load_mag_data
from src.models.cmrf_probe import Model
from src.tasks.nc import _resolve_nc_eval_labels

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
VARIANTS = ["uniform", "own_soft", "cross_soft", "gated_cross_soft"]
OUTPUT = ROOT / "outputs/cmrf_discovery_v1"
RESULTS = ROOT / "results/cmrf_discovery_v1"
CHECKPOINTS = OUTPUT / "checkpoints"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_cfg(dataset: str, seed: int, mode: str):
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.dataset = OmegaConf.load(ROOT / f"configs/dataset/{dataset}.yaml")
    cfg.task = OmegaConf.load(ROOT / "configs/task/nc.yaml")
    cfg.model = OmegaConf.load(ROOT / "configs/model/cmrf_probe.yaml")
    cfg.seed = int(seed)
    cfg.num_runs = 1
    cfg.device = str(DEVICE)
    cfg.model.composition_mode = mode
    # Offline analysis does not use Hydra's run directory/time interpolation.
    cfg.pop("hydra", None)
    OmegaConf.resolve(cfg)
    return cfg


def load_run(dataset: str, seed: int, variant: str):
    if variant == "uniform":
        path = OUTPUT / "reference_uniform" / "checkpoints" / f"{dataset}_seed{seed}.pt"
        if not path.is_file():
            path = CHECKPOINTS / variant / f"{dataset}_seed{seed}.pt"
    else:
        path = CHECKPOINTS / variant / f"{dataset}_seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing best-validation checkpoint: {path}")
    cfg = make_cfg(dataset, seed, variant)
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = Model(cfg, data_info).to(DEVICE)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(DEVICE)
    classifier.load_state_dict(checkpoint["head_state"])
    model.eval()
    classifier.eval()
    with torch.no_grad():
        x = data.x.to(DEVICE)
        edge_index = data.edge_index.to(DEVICE)
        analysis = model.analyze(x, edge_index)
    train_idx = data.train_idx.to(DEVICE)
    val_idx = data.val_idx.to(DEVICE)
    allowed = torch.cat([data.train_idx.cpu(), data.val_idx.cpu()])
    eval_labels = _resolve_nc_eval_labels(data, include_test=False)
    context = {
        "cfg": cfg,
        "data": data,
        "model": model,
        "classifier": classifier,
        "analysis": analysis,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "allowed_idx": allowed,
        "eval_labels": eval_labels,
        "checkpoint": checkpoint,
    }
    return context


def node_ce(classifier, fused: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(classifier(fused), labels, reduction="none")


def node_splits(data, selected_cpu: torch.Tensor) -> list[str]:
    train = set(data.train_idx.tolist())
    return ["train" if int(i) in train else "validation" for i in selected_cpu.tolist()]


def save_gzip_table(frame: pd.DataFrame, path: Path) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return sha256(path), len(frame)


def conditional_utility_for_run(ctx: dict) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    model, classifier, analysis, data = (
        ctx["model"], ctx["classifier"], ctx["analysis"], ctx["data"]
    )
    selected_cpu = ctx["allowed_idx"]
    selected = selected_cpu.to(DEVICE)
    # Labels are read only for Train and Validation rows.
    labels = data.y[selected_cpu].to(DEVICE)
    st, sv = analysis["S_text"], analysis["S_visual"]
    rt, rv = analysis["R_text"], analysis["R_visual"]
    zu_t = analysis["H0_text"] + 0.75 * rt[0] + 0.50 * rt[1] + 0.25 * rt[2]
    zu_v = analysis["H0_visual"] + 0.75 * rv[0] + 0.50 * rv[1] + 0.25 * rv[2]
    z_t0, z_v0 = st[0], sv[0]
    with torch.no_grad():
        ce_full = node_ce(classifier, model._fuse(zu_t[selected], zu_v[selected]), labels)
        ce_t0vu = node_ce(classifier, model._fuse(z_t0[selected], zu_v[selected]), labels)
        ce_utv0 = node_ce(classifier, model._fuse(zu_t[selected], z_v0[selected]), labels)
        ce_t0v0 = node_ce(classifier, model._fuse(z_t0[selected], z_v0[selected]), labels)
        g_t_vu = ce_t0vu - ce_full
        g_v_tu = ce_utv0 - ce_full
        g_t_v0 = ce_t0v0 - ce_utv0
        g_v_t0 = ce_t0v0 - ce_t0vu
        delta_t = g_t_vu - g_t_v0
        delta_v = g_v_tu - g_v_t0
        sign_flip_t = g_t_vu * g_t_v0 < 0
        sign_flip_v = g_v_tu * g_v_t0 < 0
    frame = pd.DataFrame(
        {
            "dataset": data.name,
            "seed": int(ctx["checkpoint"]["seed"]),
            "node_id": selected_cpu.numpy(),
            "split": node_splits(data, selected_cpu),
            "label": labels.cpu().numpy(),
            "CE_full": ce_full.cpu().numpy(),
            "CE_T0V_U": ce_t0vu.cpu().numpy(),
            "CE_T_UV0": ce_utv0.cpu().numpy(),
            "CE_T0V0": ce_t0v0.cpu().numpy(),
            "G_T_given_VU": g_t_vu.cpu().numpy(),
            "G_V_given_TU": g_v_tu.cpu().numpy(),
            "G_T_given_V0": g_t_v0.cpu().numpy(),
            "G_V_given_T0": g_v_t0.cpu().numpy(),
            "cross_condition_delta_T": delta_t.cpu().numpy(),
            "cross_condition_delta_V": delta_v.cpu().numpy(),
            "sign_flip_T": sign_flip_t.cpu().numpy(),
            "sign_flip_V": sign_flip_v.cpu().numpy(),
        }
    )
    # Expand only the permitted Train/Validation targets back onto their original
    # node ids. Test positions stay neutral placeholders and are never indexed by
    # either predictor fit or validation reporting.
    def expand_allowed(values: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(data.num_nodes, dtype=values.dtype, device=values.device)
        full[selected] = values
        return full

    targets = {
        "G_T_given_VU": expand_allowed(g_t_vu),
        "G_V_given_TU": expand_allowed(g_v_tu),
        "G_T_given_V0": expand_allowed(g_t_v0),
        "G_V_given_T0": expand_allowed(g_v_t0),
        "cross_condition_delta_T": expand_allowed(delta_t),
        "cross_condition_delta_V": expand_allowed(delta_v),
    }
    return frame, targets


def summarize_conditional(frame: pd.DataFrame, node_path: Path, digest: str, row_count: int):
    rows = []
    for modality, target, delta, flip in (
        ("Text", "G_T_given_VU", "cross_condition_delta_T", "sign_flip_T"),
        ("Visual", "G_V_given_TU", "cross_condition_delta_V", "sign_flip_V"),
    ):
        values = frame[target].to_numpy()
        rows.append(
            {
                "dataset": frame.dataset.iloc[0],
                "seed": int(frame.seed.iloc[0]),
                "target_modality": modality,
                "n_train": int((frame.split == "train").sum()),
                "n_validation": int((frame.split == "validation").sum()),
                "utility_mean": float(np.mean(values)),
                "utility_std": float(np.std(values, ddof=0)),
                "utility_median": float(np.median(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "negative_fraction": float(np.mean(values < 0)),
                "cross_condition_delta_mean": float(frame[delta].mean()),
                "cross_condition_delta_std": float(frame[delta].std(ddof=0)),
                "sign_flip_fraction": float(frame[flip].mean()),
                "raw_table": str(node_path.relative_to(ROOT)),
                "raw_sha256": digest,
                "raw_row_count": row_count,
            }
        )
    return rows


class UtilityRegressor(nn.Module):
    """Equal-size own/cross utility predictor over H0,R1,R2,R3 tokens."""

    def __init__(self, hidden_dim: int, latent_dim: int = 32, controller_dim: int = 64):
        super().__init__()
        self.text_encoder = nn.ModuleList(
            nn.Linear(hidden_dim, latent_dim) for _ in range(4)
        )
        self.visual_encoder = nn.ModuleList(
            nn.Linear(hidden_dim, latent_dim) for _ in range(4)
        )
        qdim = latent_dim * 4
        self.predictor = nn.Sequential(
            nn.Linear(2 * qdim, controller_dim),
            nn.ReLU(),
            nn.Linear(controller_dim, 1),
        )

    @staticmethod
    def _encode(tokens, modules):
        return torch.cat(
            [torch.relu(layer(token)) for layer, token in zip(modules, tokens)], dim=-1
        )

    def forward(
        self, tokens_text, tokens_visual, target_modality: str, cross: bool,
        node_idx: torch.Tensor | None = None,
    ):
        if node_idx is not None:
            tokens_text = [token[node_idx] for token in tokens_text]
            tokens_visual = [token[node_idx] for token in tokens_visual]
        q_text = self._encode(tokens_text, self.text_encoder)
        q_visual = self._encode(tokens_visual, self.visual_encoder)
        if target_modality == "Text":
            own, companion = q_text, q_visual
        else:
            own, companion = q_visual, q_text
        second = companion if cross else torch.zeros_like(own)
        return self.predictor(torch.cat([own, second], dim=-1)).squeeze(-1)


def safe_spearman(target: np.ndarray, pred: np.ndarray) -> float:
    if len(target) < 2 or np.std(target) == 0 or np.std(pred) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(scipy_stats.spearmanr(target, pred).statistic)


def utility_predictability_for_target(
    ctx: dict, targets: dict[str, torch.Tensor], target_modality: str
) -> dict:
    analysis, data = ctx["analysis"], ctx["data"]
    train_idx = ctx["train_idx"]
    val_idx = ctx["val_idx"]
    if target_modality == "Text":
        target = targets["G_T_given_VU"]
    else:
        target = targets["G_V_given_TU"]
    train_target = target[train_idx]
    mean = train_target.mean()
    std = train_target.std(unbiased=False).clamp_min(1e-8)
    standardized = (target - mean) / std
    tokens_text = [analysis["H0_text"], *analysis["R_text"]]
    tokens_visual = [analysis["H0_visual"], *analysis["R_visual"]]
    train_labels = standardized[train_idx]
    metrics = {}
    parameter_counts = {}
    for cross, name in ((False, "own"), (True, "cross")):
        devices = [DEVICE.index] if DEVICE.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            if DEVICE.type == "cuda":
                torch.cuda.manual_seed_all(0)
            predictor = UtilityRegressor(hidden_dim=ctx["model"].hidden_dim).to(DEVICE)
        parameter_counts[name] = sum(p.numel() for p in predictor.parameters())
        optimizer = torch.optim.AdamW(
            predictor.parameters(), lr=1e-3, weight_decay=1e-4
        )
        predictor.train()
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            pred = predictor(
                tokens_text, tokens_visual, target_modality, cross, node_idx=train_idx
            )
            loss = F.huber_loss(pred, train_labels)
            loss.backward()
            optimizer.step()
        predictor.eval()
        with torch.no_grad():
            val_pred = predictor(
                tokens_text, tokens_visual, target_modality, cross, node_idx=val_idx
            ) * std + mean
        true = target[val_idx].detach().cpu().numpy()
        predicted = val_pred.detach().cpu().numpy()
        positive = true > 0
        negative = true < 0
        try:
            sign_auc = float(roc_auc_score(positive.astype(np.int8), predicted))
        except ValueError:
            sign_auc = float("nan")
        metrics[name] = {
            "val_mse": float(np.mean((predicted - true) ** 2)),
            "val_mae": float(np.mean(np.abs(predicted - true))),
            "val_spearman": safe_spearman(true, predicted),
            "sign_auroc": sign_auc,
            "positive_negative_pred_gap": (
                float(predicted[positive].mean() - predicted[negative].mean())
                if positive.any() and negative.any()
                else float("nan")
            ),
            "positive_negative_true_gap": (
                float(true[positive].mean() - true[negative].mean())
                if positive.any() and negative.any()
                else float("nan")
            ),
            "validation_positive_fraction": float(positive.mean()),
        }
    base_mse = float(np.mean((target[val_idx].detach().cpu().numpy() - float(mean)) ** 2))
    own, cross = metrics["own"], metrics["cross"]
    return {
        "dataset": data.name,
        "seed": int(ctx["checkpoint"]["seed"]),
        "target_modality": target_modality,
        "probe_seed": 0,
        "epochs": 200,
        "target_train_mean": float(mean),
        "target_train_std": float(std),
        "validation_mean_predictor_mse": base_mse,
        "own_parameter_count": parameter_counts["own"],
        "cross_parameter_count": parameter_counts["cross"],
        **{f"own_{k}": v for k, v in own.items()},
        **{f"cross_{k}": v for k, v in cross.items()},
        "cross_minus_own_val_mse": cross["val_mse"] - own["val_mse"],
        "cross_minus_own_val_mae": cross["val_mae"] - own["val_mae"],
        "cross_minus_own_val_spearman": cross["val_spearman"] - own["val_spearman"],
        "cross_minus_own_sign_auroc": cross["sign_auroc"] - own["sign_auroc"],
        "cross_minus_own_pred_gap": (
            cross["positive_negative_pred_gap"] - own["positive_negative_pred_gap"]
        ),
    }


def run_e3a() -> None:
    node_dir = OUTPUT / "node_utility"
    node_dir.mkdir(parents=True, exist_ok=True)
    conditional_rows, predict_rows = [], []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"E3-A {dataset} seed={seed}: loading C0 and calculating frozen headroom", flush=True)
            ctx = load_run(dataset, seed, "uniform")
            frame, targets = conditional_utility_for_run(ctx)
            node_path = node_dir / f"{dataset}_seed{seed}.csv.gz"
            digest, row_count = save_gzip_table(frame, node_path)
            conditional_rows.extend(
                summarize_conditional(frame, node_path, digest, row_count)
            )
            for modality in ("Text", "Visual"):
                print(f"E3-A predictor {dataset} seed={seed} modality={modality}", flush=True)
                predict_rows.append(
                    utility_predictability_for_target(ctx, targets, modality)
                )
            del ctx, frame, targets
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(conditional_rows).to_csv(
        RESULTS / "e3a_conditional_utility.csv", index=False
    )
    pd.DataFrame(predict_rows).to_csv(
        RESULTS / "e3a_utility_predictability.csv", index=False
    )


def fit_linear_probe(
    features: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
    eval_labels: list[int],
    *,
    lr: float,
    weight_decay: float,
) -> dict[str, float]:
    devices = [DEVICE.index] if DEVICE.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        if DEVICE.type == "cuda":
            torch.cuda.manual_seed_all(0)
        probe = nn.Linear(features.size(1), num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=lr, weight_decay=weight_decay
    )
    targets = y.to(DEVICE)
    for _ in range(200):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(features[train_idx]), targets[train_idx])
        loss.backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        logits = probe(features[val_idx])
        val_loss = F.cross_entropy(logits, targets[val_idx]).item()
        pred = logits.argmax(dim=-1).cpu().numpy()
    true = targets[val_idx].detach().cpu().numpy()
    return {
        "val_accuracy": float(np.mean(pred == true)),
        "val_macro_f1": float(
            f1_score(
                true, pred, labels=eval_labels, average="macro", zero_division=0
            )
        ),
        "val_ce": float(val_loss),
    }


def run_e3b() -> None:
    rows = []
    summaries = []
    for dataset in DATASETS:
        for seed in SEEDS:
            print(f"E3-B {dataset} seed={seed}: fitting 16 order-pair probes", flush=True)
            ctx = load_run(dataset, seed, "uniform")
            data, analysis = ctx["data"], ctx["analysis"]
            st, sv = analysis["S_text"], analysis["S_visual"]
            y = data.y.to(DEVICE)
            train_idx, val_idx = ctx["train_idx"], ctx["val_idx"]
            allowed_y = y[torch.cat([train_idx, val_idx])]
            eval_labels = sorted(
                int(v) for v in torch.unique(allowed_y).detach().cpu().tolist()
            )
            num_classes = int(data.num_classes)
            matrix = np.zeros((4, 4), dtype=np.float64)
            for kt in range(4):
                for kv in range(4):
                    features = torch.cat([st[kt], sv[kv]], dim=-1)
                    metrics = fit_linear_probe(
                        features, y, train_idx, val_idx, num_classes, eval_labels,
                        lr=1e-2, weight_decay=1e-4,
                    )
                    matrix[kt, kv] = metrics["val_ce"]
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "k_text": kt,
                            "k_visual": kv,
                            **metrics,
                            "probe_seed": 0,
                            "epochs": 200,
                        }
                    )
                    del features
            unimodal = {}
            for modality, states in (("text", st), ("visual", sv)):
                scores = []
                for order in range(4):
                    metrics = fit_linear_probe(
                        states[order], y, train_idx, val_idx, num_classes, eval_labels,
                        lr=1e-2, weight_decay=1e-4,
                    )
                    scores.append(metrics["val_ce"])
                    unimodal[f"{modality}_k{order}"] = metrics
                unimodal[f"best_{modality}_order"] = int(np.argmin(scores))
            best_t_given_v = [int(np.argmin(matrix[:, kv])) for kv in range(4)]
            best_v_given_t = [int(np.argmin(matrix[kt, :])) for kt in range(4)]
            interaction = np.zeros_like(matrix)
            for kt in range(4):
                for kv in range(4):
                    interaction[kt, kv] = (
                        matrix[kt, 0] + matrix[0, kv] - matrix[0, 0] - matrix[kt, kv]
                    )
            joint = np.unravel_index(int(np.argmin(matrix)), matrix.shape)
            diagonal = min(range(4), key=lambda k: matrix[k, k])
            record = {
                "dataset": dataset,
                "seed": seed,
                "val_ce_matrix": matrix.tolist(),
                "best_text_order_given_visual_order": best_t_given_v,
                "best_visual_order_given_text_order": best_v_given_t,
                "number_unique_best_text_orders": len(set(best_t_given_v)),
                "number_unique_best_visual_orders": len(set(best_v_given_t)),
                "ce_interaction_residual_matrix": interaction.tolist(),
                "mean_abs_ce_interaction_residual": float(np.mean(np.abs(interaction))),
                "joint_best_pair": [int(joint[0]), int(joint[1])],
                "joint_best_val_ce": float(matrix[joint]),
                "diagonal_best_pair": [int(diagonal), int(diagonal)],
                "diagonal_best_val_ce": float(matrix[diagonal, diagonal]),
                "independent_unimodal_best_pair": [
                    unimodal["best_text_order"], unimodal["best_visual_order"]
                ],
                "unimodal_best_val_ce": {
                    "text": float(unimodal[f"text_k{unimodal['best_text_order']}"]["val_ce"]),
                    "visual": float(unimodal[f"visual_k{unimodal['best_visual_order']}"]["val_ce"]),
                },
                "unimodal_probe_metrics": unimodal,
            }
            summaries.append(record)
            del ctx, st, sv
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "e3b_cross_order_matrix.csv", index=False)
    dataset_summary = {}
    for dataset in DATASETS:
        current = [r for r in summaries if r["dataset"] == dataset]
        dataset_summary[dataset] = {
            "n_seeds": len(current),
            "unique_text_order_counts": [
                r["number_unique_best_text_orders"] for r in current
            ],
            "unique_visual_order_counts": [
                r["number_unique_best_visual_orders"] for r in current
            ],
            "mean_abs_interaction_residual": float(
                np.mean([r["mean_abs_ce_interaction_residual"] for r in current])
            ),
            "joint_best_pairs": [r["joint_best_pair"] for r in current],
            "diagonal_best_pairs": [r["diagonal_best_pair"] for r in current],
            "independent_unimodal_best_pairs": [
                r["independent_unimodal_best_pair"] for r in current
            ],
        }
    summary = {
        "protocol": {
            "encoder": "frozen C0 best-validation checkpoint",
            "fit": "Train nodes only",
            "report": "Validation only",
            "test_used": False,
            "probe": "Linear(2*hidden_dim,num_classes)",
            "optimizer": "AdamW",
            "lr": 1e-2,
            "weight_decay": 1e-4,
            "epochs": 200,
            "probe_seed": 0,
            "independent_unimodal_probes": "4 linear probes per modality per dataset/seed",
        },
        "n_matrices": len(summaries),
        "aggregate_mean_unique_best_text_orders": float(
            np.mean([r["number_unique_best_text_orders"] for r in summaries])
        ),
        "aggregate_mean_unique_best_visual_orders": float(
            np.mean([r["number_unique_best_visual_orders"] for r in summaries])
        ),
        "aggregate_mean_abs_ce_interaction_residual": float(
            np.mean([r["mean_abs_ce_interaction_residual"] for r in summaries])
        ),
        "datasets": dataset_summary,
        "per_dataset_seed": summaries,
    }
    (RESULTS / "e3b_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )


def read_run_metrics(variant: str, dataset: str, seed: int) -> dict:
    run_dir = OUTPUT / "runs" / variant / dataset / f"seed{seed}"
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    values = payload["metrics"]
    return {
        "val_acc": float(values["val_acc"]["mean"]),
        "val_macro_f1": float(values["val_macro_f1"]["mean"]),
        "best_epoch": payload.get("best_epoch"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "peak_gpu_memory_mib": payload.get("peak_gpu_memory_mib"),
    }


def run_main_comparisons() -> pd.DataFrame:
    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            by_variant = {
                variant: read_run_metrics(variant, dataset, seed)
                for variant in VARIANTS
            }
            for variant, values in by_variant.items():
                records.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "variant": variant,
                        **values,
                    }
                )
    metrics = pd.DataFrame(records)
    pairs = [
        ("own_soft", "uniform", "OwnSoft - Uniform"),
        ("cross_soft", "own_soft", "CrossSoft - OwnSoft"),
        ("gated_cross_soft", "own_soft", "GatedCross - OwnSoft"),
        ("gated_cross_soft", "cross_soft", "GatedCross - CrossSoft"),
    ]
    rows = []
    for dataset in [*DATASETS, "ALL"]:
        sample = metrics if dataset == "ALL" else metrics[metrics.dataset == dataset]
        for left, right, label in pairs:
            a = sample[sample.variant == left].set_index(["dataset", "seed"])
            b = sample[sample.variant == right].set_index(["dataset", "seed"])
            pair_ids = sorted(set(a.index) & set(b.index))
            acc_diff = np.asarray(
                [100 * (a.loc[i, "val_acc"] - b.loc[i, "val_acc"]) for i in pair_ids]
            )
            f1_diff = np.asarray(
                [100 * (a.loc[i, "val_macro_f1"] - b.loc[i, "val_macro_f1"]) for i in pair_ids]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": label,
                    "left_variant": left,
                    "right_variant": right,
                    "n_paired_seeds": int(len(pair_ids)),
                    "mean_val_acc_pp": float(np.mean(acc_diff)),
                    "population_sd_val_acc_pp": float(np.std(acc_diff, ddof=0)),
                    "positive_acc_seed_pairs": int(np.sum(acc_diff > 0)),
                    "positive_acc_seed_pairs_denominator": int(len(acc_diff)),
                    "mean_val_macro_f1_pp": float(np.mean(f1_diff)),
                    "population_sd_val_macro_f1_pp": float(np.std(f1_diff, ddof=0)),
                    "positive_macro_f1_seed_pairs": int(np.sum(f1_diff > 0)),
                    "positive_macro_f1_seed_pairs_denominator": int(len(f1_diff)),
                }
            )
    result = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS / "e3c_controller_results.csv", index=False)
    metrics.to_csv(RESULTS / "e3c_per_run_validation.csv", index=False)
    return result


def classification_metrics(logits, labels, eval_labels):
    pred = logits.argmax(dim=-1)
    true = labels
    return {
        "acc": float((pred == true).float().mean().item()),
        "macro_f1": float(
            f1_score(
                true.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                labels=eval_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "pred": pred,
    }


def run_cross_interventions() -> pd.DataFrame:
    rows = []
    for variant in ("cross_soft", "gated_cross_soft"):
        for dataset in DATASETS:
            for seed in SEEDS:
                print(f"E3-C intervention {variant} {dataset} seed={seed}", flush=True)
                ctx = load_run(dataset, seed, variant)
                model, classifier, analysis, data = (
                    ctx["model"], ctx["classifier"], ctx["analysis"], ctx["data"]
                )
                val_idx = ctx["val_idx"]
                val_labels = data.y[data.val_idx.cpu()].to(DEVICE)
                normal = model.recompose(analysis)
                with torch.no_grad():
                    base = classification_metrics(
                        classifier(normal["fused_z"][val_idx]), val_labels,
                        ctx["eval_labels"],
                    )
                base_pred = base.pop("pred")
                for target_modality in ("Text", "Visual"):
                    other_key = "q_visual" if target_modality == "Text" else "q_text"
                    focal_key = "a_text" if target_modality == "Text" else "a_visual"
                    rho_key = "rho_V2T" if target_modality == "Text" else "rho_T2V"
                    q_other = analysis[other_key]
                    scenarios = []
                    for shuffle_seed in range(10):
                        generator = torch.Generator(device="cpu").manual_seed(
                            20260926 + int(seed) * 100 + shuffle_seed
                        )
                        permutation = torch.randperm(q_other.size(0), generator=generator)
                        scenarios.append(
                            (
                                "cross_shuffle",
                                shuffle_seed,
                                q_other[permutation.to(DEVICE)],
                            )
                        )
                    scenarios.append(("cross_zero", -1, torch.zeros_like(q_other)))
                    for intervention, shuffle_seed, replacement in scenarios:
                        changed = model.intervene_companion(
                            analysis,
                            target_modality=target_modality,
                            companion_q=replacement,
                        )
                        with torch.no_grad():
                            metric = classification_metrics(
                                classifier(changed["fused_z"][val_idx]),
                                val_labels,
                                ctx["eval_labels"],
                            )
                        pred = metric.pop("pred")
                        coeff_delta = (
                            changed[focal_key][val_idx]
                            - normal[focal_key][val_idx]
                        ).abs().mean().item()
                        rho_delta = rho_mean = rho_std_delta = float("nan")
                        if normal[rho_key] is not None and changed[rho_key] is not None:
                            rho_base = normal[rho_key][val_idx]
                            rho_changed = changed[rho_key][val_idx]
                            rho_delta = (rho_base - rho_changed).abs().mean().item()
                            rho_mean = rho_changed.mean().item() - rho_base.mean().item()
                            rho_std_delta = rho_changed.std(unbiased=False).item() - rho_base.std(unbiased=False).item()
                        rows.append(
                            {
                                "variant": variant,
                                "dataset": dataset,
                                "seed": seed,
                                "target_modality": target_modality,
                                "intervention": intervention,
                                "shuffle_seed": shuffle_seed,
                                "validation_acc": metric["acc"],
                                "validation_macro_f1": metric["macro_f1"],
                                "normal_validation_acc": base["acc"],
                                "normal_validation_macro_f1": base["macro_f1"],
                                "acc_drop_pp": 100 * (base["acc"] - metric["acc"]),
                                "macro_f1_drop_pp": 100 * (base["macro_f1"] - metric["macro_f1"]),
                                "prediction_flip_rate": float((pred != base_pred).float().mean().item()),
                                "mean_abs_coefficient_change": float(coeff_delta),
                                "rho_mean_abs_change": float(rho_delta),
                                "rho_mean_shift": float(rho_mean),
                                "rho_std_shift": float(rho_std_delta),
                            }
                        )
                del ctx
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS / "e3c_crossmodal_intervention.csv", index=False)
    return result


def correlation(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(scipy_stats.spearmanr(x[valid], y[valid]).statistic)


def export_one_controller(variant: str, dataset: str, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    ctx = load_run(dataset, seed, variant)
    data, analysis = ctx["data"], ctx["analysis"]
    allowed_cpu = ctx["allowed_idx"]
    allowed = allowed_cpu.to(DEVICE)
    utility_path = OUTPUT / "node_utility" / f"{dataset}_seed{seed}.csv.gz"
    utility = pd.read_csv(utility_path)
    utility = utility.set_index("node_id").loc[allowed_cpu.numpy()].reset_index()
    frame = utility[["dataset", "seed", "node_id", "split", "label"]].copy()
    controller_rows = []
    modality_coefficients = {}
    for modality, h0_key, r_key, a_key, rho_key, utility_key, delta_key in (
        (
            "Text", "H0_text", "R_text", "a_text", "rho_V2T",
            "G_T_given_VU", "cross_condition_delta_T",
        ),
        (
            "Visual", "H0_visual", "R_visual", "a_visual", "rho_T2V",
            "G_V_given_TU", "cross_condition_delta_V",
        ),
    ):
        coeff = analysis[a_key][allowed]
        modality_coefficients[modality] = coeff.detach().cpu().numpy()
        responses = [item[allowed] for item in analysis[r_key]]
        h0 = analysis[h0_key][allowed]
        masses = torch.stack(
            [
                (coeff[:, k : k + 1].abs() * responses[k].abs()).sum(dim=-1)
                for k in range(3)
            ],
            dim=-1,
        )
        rel_mass = masses.sum(dim=-1)
        h0_mass = h0.abs().sum(dim=-1)
        contribution_ratio = rel_mass / (rel_mass + h0_mass).clamp_min(1e-12)
        coeff_np = coeff.detach().cpu().numpy()
        mass_np = masses.detach().cpu().numpy()
        frame[f"a_{modality.lower()}_1"] = coeff_np[:, 0]
        frame[f"a_{modality.lower()}_2"] = coeff_np[:, 1]
        frame[f"a_{modality.lower()}_3"] = coeff_np[:, 2]
        frame[f"relational_mass_{modality.lower()}"] = rel_mass.detach().cpu().numpy()
        frame[f"response_contribution_ratio_{modality.lower()}"] = contribution_ratio.detach().cpu().numpy()
        for k in range(3):
            frame[f"response_mass_{modality.lower()}_{k+1}"] = mass_np[:, k]
        if analysis[rho_key] is not None:
            rho_values = analysis[rho_key][allowed].detach().cpu().numpy()
            direction = "V2T" if modality == "Text" else "T2V"
            for k in range(3):
                frame[f"rho_{direction}_{k+1}"] = rho_values[:, k]
        frame[utility_key] = utility[utility_key].to_numpy()
        frame[delta_key] = utility[delta_key].to_numpy()
        modality_record = {
            "variant": variant,
            "dataset": dataset,
            "seed": seed,
            "modality": modality,
            "n_nodes": int(len(frame)),
            "effective_relational_mass_mean": float(rel_mass.mean().item()),
            "effective_relational_mass_std": float(rel_mass.std(unbiased=False).item()),
            "response_contribution_ratio_mean": float(contribution_ratio.mean().item()),
            "response_contribution_ratio_std": float(contribution_ratio.std(unbiased=False).item()),
            "utility_spearman": correlation(
                rel_mass.detach().cpu().numpy(), utility[utility_key].to_numpy()
            ),
            "cross_condition_delta_spearman": correlation(
                rel_mass.detach().cpu().numpy(), utility[delta_key].to_numpy()
            ),
        }
        for k in range(3):
            values = coeff_np[:, k]
            modality_record[f"a{k+1}_mean"] = float(np.mean(values))
            modality_record[f"a{k+1}_std"] = float(np.std(values, ddof=0))
            modality_record[f"a{k+1}_q05"] = float(np.quantile(values, 0.05))
            modality_record[f"a{k+1}_q95"] = float(np.quantile(values, 0.95))
            modality_record[f"a{k+1}_spearman_utility"] = correlation(
                values, utility[utility_key].to_numpy()
            )
            modality_record[f"a{k+1}_spearman_cross_condition_delta"] = correlation(
                values, utility[delta_key].to_numpy()
            )
        if analysis[rho_key] is not None:
            rho = analysis[rho_key][allowed].detach().cpu().numpy()
            modality_record["rho_mean"] = float(np.mean(rho))
            modality_record["rho_std"] = float(np.std(rho, ddof=0))
        else:
            modality_record["rho_mean"] = float("nan")
            modality_record["rho_std"] = float("nan")
        controller_rows.append(modality_record)
    mean_difference = np.abs(
        modality_coefficients["Text"].mean(0)
        - modality_coefficients["Visual"].mean(0)
    ).mean()
    for row in controller_rows:
        row["text_visual_mean_coefficient_distance"] = float(mean_difference)
    output_path = (
        OUTPUT / "controller_analysis" / variant / f"{dataset}_seed{seed}.csv.gz"
    )
    digest, row_count = save_gzip_table(frame, output_path)
    for row in controller_rows:
        row["raw_table"] = str(output_path.relative_to(ROOT))
        row["raw_sha256"] = digest
        row["raw_row_count"] = row_count
    del ctx, utility
    return frame[["node_id", "a_text_1", "a_text_2", "a_text_3",
                  "a_visual_1", "a_visual_2", "a_visual_3"]], controller_rows


def add_cross_seed_consistency(records: list[dict]) -> None:
    lookup = {}
    for variant in ("own_soft", "cross_soft", "gated_cross_soft"):
        for dataset in DATASETS:
            frames = {}
            for seed in SEEDS:
                path = OUTPUT / "controller_analysis" / variant / f"{dataset}_seed{seed}.csv.gz"
                frames[seed] = pd.read_csv(path)
            for modality in ("text", "visual"):
                for order in range(1, 4):
                    column = f"a_{modality}_{order}"
                    pair_values = {seed: [] for seed in SEEDS}
                    for left, right in combinations(SEEDS, 2):
                        joined = frames[left][["node_id", column]].merge(
                            frames[right][["node_id", column]],
                            on="node_id", suffixes=("_l", "_r"),
                        )
                        rho = correlation(
                            joined[f"{column}_l"], joined[f"{column}_r"]
                        )
                        pair_values[left].append(rho)
                        pair_values[right].append(rho)
                    for seed in SEEDS:
                        values = [v for v in pair_values[seed] if np.isfinite(v)]
                        lookup[(variant, dataset, seed, modality, order)] = (
                            float(np.mean(values)) if values else float("nan")
                        )
    for row in records:
        values = [
            lookup.get(
                (row["variant"], row["dataset"], row["seed"], row["modality"], order),
                float("nan"),
            )
            for order in range(1, 4)
        ]
        finite = [v for v in values if np.isfinite(v)]
        row["cross_seed_consistency_mean_spearman"] = (
            float(np.mean(finite)) if finite else float("nan")
        )
        for order, value in enumerate(values, 1):
            row[f"a{order}_cross_seed_spearman"] = value


def run_controller_analysis() -> pd.DataFrame:
    records = []
    for variant in ("own_soft", "cross_soft", "gated_cross_soft"):
        for dataset in DATASETS:
            for seed in SEEDS:
                print(f"E3-C controller analysis {variant} {dataset} seed={seed}", flush=True)
                _, rows = export_one_controller(variant, dataset, seed)
                records.extend(rows)
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
    add_cross_seed_consistency(records)
    frame = pd.DataFrame(records)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS / "e3c_controller_analysis.csv", index=False)
    return frame


def get_comparison(main: pd.DataFrame, comparison: str, dataset="ALL"):
    part = main[
        (main.dataset == dataset) & (main.comparison == comparison)
    ]
    return None if part.empty else part.iloc[0].to_dict()


def build_decisions() -> tuple[pd.DataFrame, dict]:
    conditional = pd.read_csv(RESULTS / "e3a_conditional_utility.csv")
    prediction = pd.read_csv(RESULTS / "e3a_utility_predictability.csv")
    order_summary = json.loads(
        (RESULTS / "e3b_summary.json").read_text(encoding="utf-8")
    )["per_dataset_seed"]
    main = pd.read_csv(RESULTS / "e3c_controller_results.csv")
    interventions = pd.read_csv(RESULTS / "e3c_crossmodal_intervention.csv")

    h1_by_seed = []
    for (dataset, seed), group in conditional.groupby(["dataset", "seed"]):
        supported = bool(
            (
                (group.utility_std > 0.01)
                & (group.positive_fraction >= 0.05)
                & (group.negative_fraction >= 0.05)
            ).any()
        )
        h1_by_seed.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "supported": supported,
                "utility_std": float(group.utility_std.mean()),
            }
        )
    h1_df = pd.DataFrame(h1_by_seed)
    h1_datasets = sorted(
        ds for ds, group in h1_df.groupby("dataset")
        if int(group.supported.sum()) >= 2
    )
    h1_seed_rate = float(h1_df.supported.mean())
    h1_status = (
        "STRONG_SUPPORT" if len(h1_datasets) >= 4 and h1_seed_rate >= 0.8
        else "MIXED_SUPPORT" if len(h1_datasets) >= 2
        else "NO_SUPPORT"
    )

    own = prediction.copy()
    own["supported"] = (
        (own.own_val_mse < own.validation_mean_predictor_mse)
        & (own.own_val_spearman > 0)
        & (own.own_sign_auroc > 0.55)
    )
    h2_datasets = sorted(
        ds for ds, group in own.groupby("dataset")
        if int(group.supported.sum()) >= 4
    )
    h2_seed_rate = float(own.supported.mean())
    h2_relative_mse_gain = float(
        np.nanmedian(
            (own.validation_mean_predictor_mse - own.own_val_mse)
            / np.maximum(own.validation_mean_predictor_mse, 1e-12)
        )
    )
    h2_status = (
        "STRONG_SUPPORT" if len(h2_datasets) >= 4 and h2_seed_rate >= 0.65
        else "MIXED_SUPPORT" if len(h2_datasets) >= 2
        else "NO_SUPPORT"
    )

    prediction["paired_cross_better"] = (
        (prediction.cross_minus_own_val_mse < 0)
        & (prediction.cross_minus_own_val_spearman > 0)
    )
    h3_datasets = sorted(
        ds for ds, group in prediction.groupby("dataset")
        if int(group.paired_cross_better.sum()) >= 4
    )
    h3_seed_rate = float(prediction.paired_cross_better.mean())
    h3_mse_gain = float(
        np.nanmedian(
            (prediction.own_val_mse - prediction.cross_val_mse)
            / np.maximum(prediction.own_val_mse, 1e-12)
        )
    )
    h3_status = (
        "STRONG_SUPPORT" if len(h3_datasets) >= 4 and h3_seed_rate >= 0.65
        else "MIXED_SUPPORT" if len(h3_datasets) >= 2
        else "NO_SUPPORT"
    )

    h4_seed = []
    for row in order_summary:
        supported = (
            row["number_unique_best_text_orders"] >= 2
            and row["number_unique_best_visual_orders"] >= 2
            and row["mean_abs_ce_interaction_residual"] > 0.001
        )
        h4_seed.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "supported": bool(supported),
                "interaction": row["mean_abs_ce_interaction_residual"],
            }
        )
    h4_df = pd.DataFrame(h4_seed)
    h4_datasets = sorted(
        ds for ds, group in h4_df.groupby("dataset")
        if int(group.supported.sum()) >= 2
    )
    h4_seed_rate = float(h4_df.supported.mean())
    h4_effect = float(h4_df.interaction.mean())
    h4_status = (
        "STRONG_SUPPORT" if len(h4_datasets) >= 4 and h4_seed_rate >= 0.65
        else "MIXED_SUPPORT" if len(h4_datasets) >= 2
        else "NO_SUPPORT"
    )

    h5_global = get_comparison(main, "OwnSoft - Uniform")
    h5_datasets = sorted(
        ds for ds in DATASETS
        if (
            (r := get_comparison(main, "OwnSoft - Uniform", ds)) is not None
            and r["mean_val_acc_pp"] > 0
            and r["mean_val_macro_f1_pp"] > 0
        )
    )
    h5_status = (
        "STRONG_SUPPORT"
        if h5_global["mean_val_acc_pp"] > 0
        and h5_global["positive_acc_seed_pairs"] >= 12
        and h5_global["positive_macro_f1_seed_pairs"] >= 10
        and len(h5_datasets) >= 4
        else "MIXED_SUPPORT"
        if h5_global["mean_val_acc_pp"] > 0 or len(h5_datasets) >= 2
        else "NO_SUPPORT"
    )

    cross_candidates = [
        ("CrossSoft - OwnSoft", "cross_soft"),
        ("GatedCross - OwnSoft", "gated_cross_soft"),
    ]
    h6_perf = []
    for label, variant in cross_candidates:
        global_row = get_comparison(main, label)
        ds_rows = [get_comparison(main, label, dataset) for dataset in DATASETS]
        supported_ds = sum(
            r is not None
            and r["mean_val_acc_pp"] > 0
            and r["mean_val_macro_f1_pp"] > 0
            for r in ds_rows
        )
        performance_strong = (
            global_row["mean_val_acc_pp"] > 0
            and global_row["positive_acc_seed_pairs"] >= 12
            and supported_ds >= 4
        )
        h6_perf.append(
            {
                "variant": variant,
                "comparison": label,
                "mean_acc_pp": global_row["mean_val_acc_pp"],
                "mean_macro_f1_pp": global_row["mean_val_macro_f1_pp"],
                "positive_seed_pairs": global_row["positive_acc_seed_pairs"],
                "datasets_supported": int(supported_ds),
                "performance_strong": bool(performance_strong),
            }
        )
    h6_supported = [row for row in h6_perf if row["performance_strong"]]
    h6_datasets = sorted(
        {
            ds for item in h6_supported for ds in DATASETS
            if (
                (r := get_comparison(main, item["comparison"], ds)) is not None
                and r["mean_val_acc_pp"] > 0
                and r["mean_val_macro_f1_pp"] > 0
            )
        }
    )
    h6_status = (
        "STRONG_SUPPORT"
        if h3_status == "STRONG_SUPPORT" and h6_supported
        else "MIXED_SUPPORT"
        if h3_status != "NO_SUPPORT" or any(r["mean_acc_pp"] > 0 for r in h6_perf)
        else "NO_SUPPORT"
    )

    shuffled = interventions[interventions.intervention == "cross_shuffle"]
    per_direction = (
        shuffled.groupby(["variant", "dataset", "seed", "target_modality"])
        .agg(
            acc_drop_pp=("acc_drop_pp", "mean"),
            macro_f1_drop_pp=("macro_f1_drop_pp", "mean"),
            flip_rate=("prediction_flip_rate", "mean"),
            coefficient_change=("mean_abs_coefficient_change", "mean"),
        )
        .reset_index()
    )
    per_direction["supported"] = (
        (per_direction.acc_drop_pp > 0.1)
        & (per_direction.coefficient_change > 1e-3)
    )
    h7_datasets = sorted(
        ds for ds, group in per_direction.groupby("dataset")
        if int(group.supported.sum()) >= 3
    )
    h7_seed_rate = float(per_direction.supported.mean())
    h7_effect = float(per_direction.acc_drop_pp.mean())
    h7_coeff_effect = float(per_direction.coefficient_change.mean())
    h7_status = (
        "STRONG_SUPPORT"
        if len(h7_datasets) >= 4 and h7_seed_rate >= 0.5 and h7_effect > 0.1
        else "MIXED_SUPPORT"
        if len(h7_datasets) >= 2 or h7_coeff_effect > 1e-3
        else "NO_SUPPORT"
    )

    evidence = [
        {
            "hypothesis": "H1 node/modality structural utility heterogeneity",
            "status": h1_status,
            "datasets_supported": ";".join(h1_datasets),
            "seed_consistency": f"{int(h1_df.supported.sum())}/15 dataset-seeds",
            "effect_size": f"mean node-level utility SD={h1_df.utility_std.mean():.4f}",
            "counterevidence": f"{5-len(h1_datasets)} datasets did not replicate both utility signs in >=2 seeds",
            "criteria": "per dataset-seed, at least one modality: utility SD>0.01 and positive/negative fractions each >=0.05",
        },
        {
            "hypothesis": "H2 utility predictable from own relational state",
            "status": h2_status,
            "datasets_supported": ";".join(h2_datasets),
            "seed_consistency": f"{int(own.supported.sum())}/{len(own)} target-modality runs; {h2_seed_rate:.1%}",
            "effect_size": f"median validation MSE reduction vs train-mean={h2_relative_mse_gain:.3%}",
            "counterevidence": f"{len(own)-int(own.supported.sum())} runs failed at least one MSE/Spearman/AUROC criterion",
            "criteria": "own MSE below train-mean baseline, validation Spearman>0 and sign AUROC>0.55",
        },
        {
            "hypothesis": "H3 companion modality improves utility predictability",
            "status": h3_status,
            "datasets_supported": ";".join(h3_datasets),
            "seed_consistency": f"{int(prediction.paired_cross_better.sum())}/{len(prediction)} paired target runs; {h3_seed_rate:.1%}",
            "effect_size": f"median relative validation MSE reduction Cross-Own={h3_mse_gain:.3%}",
            "counterevidence": f"{len(prediction)-int(prediction.paired_cross_better.sum())} paired runs did not improve both MSE and Spearman",
            "criteria": "equal parameter counts; Cross MSE lower and Cross Spearman higher, replicated in >=4/6 rows per dataset",
        },
        {
            "hypothesis": "H4 cross-order preference coupling",
            "status": h4_status,
            "datasets_supported": ";".join(h4_datasets),
            "seed_consistency": f"{int(h4_df.supported.sum())}/15 dataset-seed matrices; {h4_seed_rate:.1%}",
            "effect_size": f"mean absolute CE interaction residual={h4_effect:.4f}",
            "counterevidence": f"{15-int(h4_df.supported.sum())} matrices did not show multi-order conditional optima on both axes and residual>0.001",
            "criteria": ">=2 unique conditional best orders on both axes and mean abs interaction residual>0.001",
        },
        {
            "hypothesis": "H5 own-conditioned soft modulation improves Uniform",
            "status": h5_status,
            "datasets_supported": ";".join(h5_datasets),
            "seed_consistency": f"{int(h5_global['positive_acc_seed_pairs'])}/15 positive accuracy pairs; {int(h5_global['positive_macro_f1_seed_pairs'])}/15 positive Macro-F1 pairs",
            "effect_size": f"mean Val Acc={h5_global['mean_val_acc_pp']:.3f} pp; Macro-F1={h5_global['mean_val_macro_f1_pp']:.3f} pp",
            "counterevidence": f"{15-int(h5_global['positive_acc_seed_pairs'])} accuracy seed pairs were non-positive",
            "criteria": "strong requires positive mean Acc, >=12/15 positive Acc pairs, >=10/15 positive F1 pairs, and >=4 datasets positive on both",
        },
        {
            "hypothesis": "H6 cross-conditioned modulation improves own-only",
            "status": h6_status,
            "datasets_supported": ";".join(h6_datasets),
            "seed_consistency": json.dumps(h6_perf, ensure_ascii=False),
            "effect_size": "paired CrossSoft-OwnSoft and GatedCross-OwnSoft comparisons; see e3c_controller_results.csv",
            "counterevidence": "requires H3 predictive gain and at least one cross variant with replicated Val Acc/F1 gain",
            "criteria": "H3 strong plus at least one cross variant positive mean Acc, >=12/15 positive Acc pairs, and >=4/5 datasets positive on Acc/F1",
        },
        {
            "hypothesis": "H7 cross-modal controller functionally used under shuffle",
            "status": h7_status,
            "datasets_supported": ";".join(h7_datasets),
            "seed_consistency": f"{int(per_direction.supported.sum())}/{len(per_direction)} direction-level paired runs; {h7_seed_rate:.1%}",
            "effect_size": f"mean Val Acc drop={h7_effect:.3f} pp; mean |coefficient change|={h7_coeff_effect:.5f}",
            "counterevidence": f"{len(per_direction)-int(per_direction.supported.sum())} direction-level runs lacked both >0.1pp Acc drop and >0.001 coefficient change",
            "criteria": "shuffle-averaged Acc drop>0.1pp and targeted coefficient change>0.001, replicated across >=4 datasets",
        },
    ]
    decisions = pd.DataFrame(evidence)
    RESULTS.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(RESULTS / "e3_decision_matrix.csv", index=False)

    if (
        h3_status == "STRONG_SUPPORT"
        and h6_status == "STRONG_SUPPORT"
        and h7_status == "STRONG_SUPPORT"
    ):
        overall = "Cross-Modal Conditional Relational Filtering"
    elif h5_status == "STRONG_SUPPORT":
        overall = "only Context-Conditioned Modality-Specific Relational Filtering"
    else:
        overall = "neither"
    return decisions, {
        "H1": h1_status, "H2": h2_status, "H3": h3_status, "H4": h4_status,
        "H5": h5_status, "H6": h6_status, "H7": h7_status,
        "overall_interpretation": overall,
        "support_details": {
            "H1_datasets": h1_datasets, "H2_datasets": h2_datasets,
            "H3_datasets": h3_datasets, "H4_datasets": h4_datasets,
            "H5_datasets": h5_datasets, "H6_datasets": h6_datasets,
            "H7_datasets": h7_datasets,
        },
    }


def write_final_report(
    main: pd.DataFrame,
    interventions: pd.DataFrame,
    controller: pd.DataFrame,
    decisions: pd.DataFrame,
    evidence_summary: dict,
) -> None:
    per_run = pd.read_csv(RESULTS / "e3c_per_run_validation.csv")
    agg = (
        per_run.groupby("variant")
        .agg(
            val_acc_mean=("val_acc", "mean"),
            val_acc_sd=("val_acc", lambda x: np.std(x, ddof=0)),
            val_f1_mean=("val_macro_f1", "mean"),
            val_f1_sd=("val_macro_f1", lambda x: np.std(x, ddof=0)),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .reindex(VARIANTS)
    )
    lines = [
        "# Conditional Relational Utility Discovery — Experiment 3",
        "",
        "## Scope and protocol",
        "",
        "- NC only: Movies, Toys, Grocery, ele-fashion, Reddit-S; seeds 42/43/44.",
        "- C0 Uniform: 15 formal runs, with best-validation checkpoints under outputs/cmrf_discovery_v1/reference_uniform/checkpoints/. C1/C2/C3: 45 new formal runs. No LP runs.",
        "- All checkpoints selected by Validation Accuracy under unified_full_graph_nc_v1.",
        "- Test evaluation and test labels were excluded from training, model selection, E3-A/E3-B/E3-C analysis, and conclusions.",
        "- Physical operator uses unit edge weights after removing loops, symmetrizing, adding one loop per node, and symmetric normalization.",
        "- Structural coefficients preserve H0 at exactly 1.0; lambda is fixed at 0.25.",
        "",
        "## Controller results",
        "",
        "| Variant | Mean Val Acc | Population SD | Mean Val Macro-F1 | Population SD | Mean best epoch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, row in agg.iterrows():
        lines.append(
            f"| {variant} | {row.val_acc_mean:.4f} | {row.val_acc_sd:.4f} | "
            f"{row.val_f1_mean:.4f} | {row.val_f1_sd:.4f} | {row.best_epoch_mean:.1f} |"
        )
    lines += [
        "",
        "### Paired main comparisons",
        "",
        "| Comparison | Mean Val Acc Δ (pp) | SD (pp) | Positive seed pairs | Mean Macro-F1 Δ (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    all_pairs = main[main.dataset == "ALL"]
    for _, row in all_pairs.iterrows():
        lines.append(
            f"| {row.comparison} | {row.mean_val_acc_pp:.3f} | "
            f"{row.population_sd_val_acc_pp:.3f} | "
            f"{int(row.positive_acc_seed_pairs)}/{int(row.positive_acc_seed_pairs_denominator)} | "
            f"{row.mean_val_macro_f1_pp:.3f} |"
        )
    lines += ["", "## E3-A: conditional utility", ""]
    cond = pd.read_csv(RESULTS / "e3a_conditional_utility.csv")
    pred = pd.read_csv(RESULTS / "e3a_utility_predictability.csv")
    lines.append(
        f"- {len(cond)} dataset/seed/modality summaries and {len(pred)} paired utility-predictor runs."
    )
    lines.append(
        f"- Mean node utility SD: {cond.utility_std.mean():.4f}; mean positive-utility fraction: "
        f"{cond.positive_fraction.mean():.3f}; mean companion-conditioned sign-flip fraction: "
        f"{cond.sign_flip_fraction.mean():.3f}."
    )
    lines.append(
        f"- Own validation MSE mean: {pred.own_val_mse.mean():.5f}; cross-conditioned MSE mean: "
        f"{pred.cross_val_mse.mean():.5f}; mean paired MSE change Cross-Own: "
        f"{pred.cross_minus_own_val_mse.mean():.5f}."
    )
    lines += ["", "## E3-B: cross-order coupling", ""]
    order = json.loads((RESULTS / "e3b_summary.json").read_text(encoding="utf-8"))
    lines.append(
        f"- Completed {order['n_matrices']} dataset×seed matrices (16 order pairs each). "
        f"Mean unique conditional Text orders: {order['aggregate_mean_unique_best_text_orders']:.2f}; "
        f"Visual orders: {order['aggregate_mean_unique_best_visual_orders']:.2f}; "
        f"mean absolute CE interaction residual: {order['aggregate_mean_abs_ce_interaction_residual']:.5f}."
    )
    lines += ["", "## E3-C: intervention and controller analysis", ""]
    shuffle = interventions[interventions.intervention == "cross_shuffle"]
    zero = interventions[interventions.intervention == "cross_zero"]
    lines.append(
        f"- Shuffle rows: {len(shuffle)}; cross-zero rows: {len(zero)}; "
        f"shuffle-mean Accuracy drop: {shuffle.acc_drop_pp.mean():.3f} pp; "
        f"Macro-F1 drop: {shuffle.macro_f1_drop_pp.mean():.3f} pp; "
        f"prediction flip rate: {shuffle.prediction_flip_rate.mean():.4f}; "
        f"mean targeted coefficient change: {shuffle.mean_abs_coefficient_change.mean():.5f}."
    )
    gate = controller[controller.variant == "gated_cross_soft"]
    lines.append(
        f"- C3 node gate summaries: mean rho={gate.rho_mean.mean():.4f}, "
        f"mean between-node SD={gate.rho_std.mean():.4f}; raw node tables contain coefficients, "
        "effective relational mass, response contribution ratios, and utility associations."
    )
    lines += ["", "## Evidence decisions", ""]
    lines.append("| Hypothesis | Decision | Datasets supported | Effect size | Counterevidence |")
    lines.append("|---|---|---|---|---|")
    for _, row in decisions.iterrows():
        lines.append(
            f"| {row.hypothesis} | **{row.status}** | {row.datasets_supported or 'none'} | "
            f"{row.effect_size} | {row.counterevidence} |"
        )
    lines += [
        "",
        f"### Interpretation: {evidence_summary['overall_interpretation']}",
        "",
        "Cross-modal support requires replicated cross-over-own utility prediction, replicated cross-conditioned validation gains, and a measurable validation decrease under deterministic companion shuffling. Coefficient changes alone do not establish useful control.",
        "",
        "Decision thresholds are descriptive consistency rules across datasets/seeds, not significance tests. Order preferences are summarized over all 15 matrices; no single matrix is treated as a general rule.",
        "",
        "## Artifacts",
        "",
        "- preflight_report.md",
        "- e3a_conditional_utility.csv",
        "- e3a_utility_predictability.csv",
        "- e3b_cross_order_matrix.csv",
        "- e3b_summary.json",
        "- e3c_controller_results.csv",
        "- e3c_per_run_validation.csv",
        "- e3c_crossmodal_intervention.csv",
        "- e3c_controller_analysis.csv",
        "- e3_decision_matrix.csv",
        "- Raw node-level tables are compressed under outputs/cmrf_discovery_v1/; results CSVs record their row counts and SHA-256 hashes.",
        "",
        "No final paper model was designed in this phase.",
        "",
        "## Next-stage recommendation",
        "",
        "Do not expand the controller/router family from these results. H1 is strong, but own-state utility prediction, companion-state gain, and cross-dataset controller benefit are not supported. First audit the counterfactual utility target's stability and the representation probe's out-of-sample calibration using Train/Validation only; revisit model design only if that signal replicates.",
        "",
    ]
    (RESULTS / "experiment3_report.md").write_text("\n".join(lines), encoding="utf-8")
    import subprocess
    markers = list(OUTPUT.glob("runs/*/*/seed*/complete.marker"))
    summary = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "formal_run_count": len(markers),
        "formal_runs_expected": 60,
        "formal_runs_by_variant": {
            variant: sum(1 for p in OUTPUT.glob(f"runs/{variant}/*/seed*/complete.marker"))
            for variant in VARIANTS
        },
        "test_used": False,
        "lp_runs": 0,
        "controller_mean_metrics": agg.reset_index().to_dict(orient="records"),
        "main_comparisons_all": all_pairs.to_dict(orient="records"),
        "mean_shuffle_effects": {
            "acc_drop_pp": float(shuffle.acc_drop_pp.mean()),
            "macro_f1_drop_pp": float(shuffle.macro_f1_drop_pp.mean()),
            "prediction_flip_rate": float(shuffle.prediction_flip_rate.mean()),
            "mean_abs_coefficient_change": float(shuffle.mean_abs_coefficient_change.mean()),
        },
        "mean_cross_zero_effects": {
            "acc_drop_pp": float(zero.acc_drop_pp.mean()),
            "macro_f1_drop_pp": float(zero.macro_f1_drop_pp.mean()),
            "prediction_flip_rate": float(zero.prediction_flip_rate.mean()),
            "mean_abs_coefficient_change": float(zero.mean_abs_coefficient_change.mean()),
        },
        "hypotheses": evidence_summary,
        "next_stage_recommendation": "Do not expand the router family; audit utility-target stability and out-of-sample probe calibration on Train/Validation before any new model design.",
        "results_files": sorted(str(p.relative_to(ROOT)) for p in RESULTS.iterdir() if p.is_file()),
    }
    (RESULTS / "experiment3_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )


def run_e3c() -> None:
    main = run_main_comparisons()
    interventions = run_cross_interventions()
    controller = run_controller_analysis()
    decisions, evidence_summary = build_decisions()
    write_final_report(main, interventions, controller, decisions, evidence_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["e3a", "e3b", "e3c"], required=True)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    args = parser.parse_args()
    global DEVICE
    requested = args.devices[0]
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    DEVICE = torch.device(requested)
    torch.set_num_threads(1)
    if args.phase == "e3a":
        run_e3a()
    elif args.phase == "e3b":
        run_e3b()
    else:
        run_e3c()


if __name__ == "__main__":
    main()
