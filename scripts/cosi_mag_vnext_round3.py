from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, f1_score

from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _run_single_nc
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters


REPORT = ROOT / "reports/cosi_mag_vnext/round3"
OUTPUT = ROOT / "outputs/cosi_mag_vnext_round3"
PILOT_MANIFEST = ROOT / "reports/cosi_mag_vnext/run_manifest.json"
ROUND2_PROVENANCE = ROOT / "reports/cosi_mag_vnext/round2/checkpoint_provenance.json"
ROUND2_INTERVENTIONS = ROOT / "reports/cosi_mag_vnext/round2/intervention_metrics.json"
ROUND2_BASELINES = ROOT / "reports/cosi_mag_vnext/round2/baseline_metrics.json"
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
VARIANTS = ("cosi_mag_vnext_u", "cosi_mag_vnext_uq", "cosi_mag_vnext_ua")
Q_SHUFFLE_SEED = 20261013
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item()) if value.numel() else 0.0


def _quantile_summary(value: torch.Tensor) -> dict:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "mean": 0.0, "std_population": 0.0, "quantiles": {str(q): 0.0 for q in QUANTILES}}
    vals = torch.quantile(flat, torch.tensor(QUANTILES, device=flat.device)).cpu().tolist()
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std_population": float(flat.std(unbiased=False).item()),
        "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, vals)},
    }


def _ratio_distribution(numerator: torch.Tensor, denominator: torch.Tensor) -> dict:
    num = numerator.float().square().mean(dim=-1).sqrt()
    den = denominator.float().square().mean(dim=-1).sqrt().clamp_min(1e-12)
    ratio = num / den
    return {"node_ratio": _quantile_summary(ratio), "fraction_above_one": float((ratio > 1.0).float().mean().item())}


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    a, b = left.float().reshape(-1), right.float().reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return float(((a * b).sum() / denom).item()) if denom > 0 else 0.0


def _load_manifests():
    pilot = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    provenance = json.loads(ROUND2_PROVENANCE.read_text(encoding="utf-8"))
    intervention = json.loads(ROUND2_INTERVENTIONS.read_text(encoding="utf-8"))
    baseline = json.loads(ROUND2_BASELINES.read_text(encoding="utf-8"))
    expected = {(d, s) for d in DATASETS for s in SEEDS}
    runs = {(r["dataset"], int(r["seed"])): r for r in pilot["runs"]}
    if set(runs) != expected:
        raise RuntimeError(f"pilot checkpoint manifest mismatch: missing={expected-set(runs)}")
    prov = {(r["dataset"], int(r["seed"])): r for r in provenance["runs"]}
    if set(prov) != expected:
        raise RuntimeError(f"round-two provenance mismatch: missing={expected-set(prov)}")
    return pilot, runs, provenance, prov, intervention, baseline


def _data_info(data) -> dict:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def _load_data_cache(runs: dict) -> dict:
    cache = {}
    for dataset in DATASETS:
        source = runs[(dataset, 42)]
        cfg = OmegaConf.create(source["resolved_config"])
        cfg.task.evaluate_test = False
        cfg.dataset.nc_split_path = source["split_path"]
        data = load_mag_data(cfg, task_name="nc", seed=42)
        # Mask immediately; every subsequent label-dependent operation is train/val-only.
        data.y = data.y.clone()
        data.y[data.test_idx] = -100
        allowed = torch.cat((data.train_idx.cpu(), data.val_idx.cpu()))
        labels = sorted(int(v) for v in torch.unique(data.y[allowed]).tolist() if int(v) >= 0)
        if not labels:
            raise RuntimeError(f"{dataset}: no labels in train/validation")
        if sha256(Path(source["split_path"])) != source["split_sha256"]:
            raise RuntimeError(f"{dataset}: split SHA-256 mismatch")
        if str(Path(data.info["nc_split_path"]).resolve()) != str(Path(source["split_path"]).resolve()):
            raise RuntimeError(f"{dataset}: loaded a split different from the recorded seed-42 split")
        cache[dataset] = (cfg, data, labels)
    return cache


def _val_metrics(logits: torch.Tensor, data, labels: list[int]) -> tuple[dict, dict]:
    val = data.val_idx.cpu()
    target = data.y[val].cpu().long()
    pred = logits[val.to(logits.device)].argmax(dim=-1).cpu().long()
    cm = confusion_matrix(target.numpy(), pred.numpy(), labels=labels)
    f1 = f1_score(target.numpy(), pred.numpy(), labels=labels, average=None, zero_division=0)
    metric = {
        "val_acc": float((target == pred).float().mean().item()),
        "val_macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=labels, average="macro", zero_division=0)),
        "val_count": int(target.numel()),
    }
    classes = [
        {"class_id": int(label), "support": int(cm[i].sum()), "predicted_count": int(cm[:, i].sum()), "f1": float(f1[i])}
        for i, label in enumerate(labels)
    ]
    return metric, {"labels": labels, "matrix": cm.tolist(), "per_class": classes}


def _compose_with_q(model, components, q):
    text = model.interaction.forward_modality(
        components["h0_text"], components["bank_text"], q, "text"
    )
    visual = model.interaction.forward_modality(
        components["h0_visual"], components["bank_visual"], q, "visual"
    )
    fused = torch.cat((text["z"], visual["z"]), dim=-1)
    z = model.output_norm(model.fusion_skip(fused) + model.fusion_mlp(fused))
    return {"z": z, "text": text, "visual": visual}


def _edge_similarity_stats(features: torch.Tensor, edge_index: torch.Tensor) -> dict:
    src, dst = edge_index
    similarity = F.cosine_similarity(features[src], features[dst], dim=-1, eps=1e-8)
    n = int(features.size(0))
    degree = torch.bincount(src, minlength=n).to(features.dtype)
    sums = torch.zeros(n, device=features.device, dtype=features.dtype)
    sums.index_add_(0, src, similarity)
    means = sums / degree.clamp_min(1)
    centered = similarity - means[src]
    sq = torch.zeros_like(sums)
    sq.index_add_(0, src, centered.square())
    variances = sq / degree.clamp_min(1)
    active = degree > 0
    multi = degree > 1
    # Exact RMS pairwise gap over distinct neighbors, computed from local variance:
    # mean_(j<k)(s_ij-s_ik)^2 = 2*n/(n-1) * population_variance_i.
    gap = torch.zeros_like(variances)
    gap[multi] = torch.sqrt(
        2.0 * degree[multi] / (degree[multi] - 1.0) * variances[multi].clamp_min(0.0)
    )
    global_std = similarity.float().std(unbiased=False).clamp_min(1e-12)
    return {
        "directed_physical_edge_cosine": _quantile_summary(similarity),
        "node_neighborhood_variance": _quantile_summary(variances[active]),
        "neighbor_pairwise_rms_gap": _quantile_summary(gap[multi]),
        "neighbor_pairwise_rms_gap_relative_to_global_edge_std": _quantile_summary(gap[multi] / global_std),
        "nodes_with_neighbors": int(active.sum().item()),
        "nodes_with_two_or_more_neighbors": int(multi.sum().item()),
        "global_edge_cosine_std": float(global_std.item()),
    }


def _relation_score_stats(model, relation: dict) -> dict:
    crc = model.crc
    cosine_text, cosine_visual = relation["cosine_text"], relation["cosine_visual"]
    evidence = torch.stack(
        (cosine_text, cosine_visual, (cosine_text - cosine_visual).abs(), cosine_text * cosine_visual),
        dim=-1,
    )
    scores = crc.relation_scorer(evidence)
    weights = torch.stack((relation["weight_text"], relation["weight_visual"]), dim=-1)
    return {
        "raw_relation_scorer_channels": [_quantile_summary(scores[:, i]) for i in range(scores.size(1))],
        "raw_relation_scorer_centered_edge_rms": _rms(scores - scores.mean(dim=0, keepdim=True)),
        "modality_edge_weight_std": [float(weights[:, i].std(unbiased=False).item()) for i in range(2)],
        "modality_edge_weight_quantiles": [_quantile_summary(weights[:, i]) for i in range(2)],
    }


def _diagnose_original_checkpoints() -> tuple[list[dict], list[dict], list[dict]]:
    _, runs, _, provenance, _, _ = _load_manifests()
    cache = _load_data_cache(runs)
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    mechanisms, feature_rows, q_rows = [], [], []
    checkpoint_records = []
    for dataset in DATASETS:
        _, data, labels = cache[dataset]
        for seed in SEEDS:
            run, prov = runs[(dataset, seed)], provenance[(dataset, seed)]
            checkpoint_path = Path(run["checkpoint_path"])
            if not checkpoint_path.is_file() or sha256(checkpoint_path) != prov["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint missing/hash mismatch: {checkpoint_path}")
            cfg = OmegaConf.create(run["resolved_config"])
            cfg.task.evaluate_test = False
            cfg.seed, cfg.num_runs, cfg.device = seed, 1, str(device)
            set_seed(seed)
            model = build_model(cfg, _data_info(data)).to(device).eval()
            classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device).eval()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("selection") != "best_val_accuracy" or int(checkpoint.get("seed", -1)) != seed:
                raise RuntimeError(f"unexpected checkpoint metadata: {checkpoint_path}")
            model.load_state_dict(checkpoint["model_state"])
            classifier.load_state_dict(checkpoint["head_state"])
            x, edges = data.x.to(device), data.edge_index.to(device)
            with torch.no_grad():
                components = model._encode_components(x, edges)
                original_z = components["z"]
                rebuilt_z = model(x, edges)[0]
                if float((original_z - rebuilt_z).abs().max().item()) > 1e-5:
                    raise RuntimeError(f"model forward reconstruction mismatch: {dataset}/{seed}")
                original_logits = classifier(original_z)
                original_metric, confusion = _val_metrics(original_logits, data, labels)
                saved_metric = checkpoint["metrics"]
                if abs(original_metric["val_acc"] - float(saved_metric["val_acc"])) > 1e-6:
                    raise RuntimeError(f"checkpoint validation accuracy changed: {dataset}/{seed}")
                if abs(original_metric["val_macro_f1"] - float(saved_metric["val_macro_f1"])) > 1e-6:
                    raise RuntimeError(f"checkpoint validation macro-F1 changed: {dataset}/{seed}")
                q = components["relation"]["q"]
                generator = torch.Generator(device="cpu").manual_seed(Q_SHUFFLE_SEED)
                permutation = torch.randperm(data.num_nodes, generator=generator).to(device)
                q_shuffled = q[permutation]
                shuffled = _compose_with_q(model, components, q_shuffled)
                val_idx = data.val_idx.to(device)
                original_val_logits = original_logits[val_idx]
                shuffled_logits = classifier(shuffled["z"])
                shuffled_val_logits = shuffled_logits[val_idx]
                original_prob = original_val_logits.softmax(dim=-1)
                shuffled_prob = shuffled_val_logits.softmax(dim=-1)
                conf_change = shuffled_prob.max(dim=-1).values - original_prob.max(dim=-1).values
                q_diag = {
                    "dataset": dataset,
                    "seed": seed,
                    "q_shuffle_seed": Q_SHUFFLE_SEED,
                    "q_node_centered_rms": _rms(q - q.mean(dim=0, keepdim=True)),
                    "q_mean_node_norm": float(q.norm(dim=-1).mean().item()),
                    "val_logits_rms_delta": _rms(shuffled_val_logits - original_val_logits),
                    "val_prediction_change_rate": float((shuffled_val_logits.argmax(-1) != original_val_logits.argmax(-1)).float().mean().item()),
                    "val_confidence_delta_mean": float(conf_change.mean().item()),
                    "val_confidence_delta_mean_abs": float(conf_change.abs().mean().item()),
                    "val_acc_delta_pp": 100.0 * (_val_metrics(shuffled_logits, data, labels)[0]["val_acc"] - original_metric["val_acc"]),
                    "val_macro_f1_delta_pp": 100.0 * (_val_metrics(shuffled_logits, data, labels)[0]["val_macro_f1"] - original_metric["val_macro_f1"]),
                }
                q_rows.append(q_diag)
                edge_index = components["relation"]["physical_edge_index"]
                representations = {
                    "raw_pretrained": (data.x_t.to(device), data.x_i.to(device)),
                    "pre_relu_projection": (
                        model.text_proj.net[1](model.text_proj.net[0](data.x_t.to(device))),
                        model.visual_proj.net[1](model.visual_proj.net[0](data.x_i.to(device))),
                    ),
                    "post_relu_projection": (components["h0_text"], components["h0_visual"]),
                }
                for representation, pair in representations.items():
                    for modality, feature in zip(("text", "visual"), pair):
                        stats = _edge_similarity_stats(feature, edge_index)
                        feature_rows.append({"dataset": dataset, "seed": seed, "representation": representation, "modality": modality, **stats})
                mode_mechanism = {
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": prov["checkpoint_sha256"],
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "split_path": prov["split_path"],
                    "split_sha256": prov["split_sha256"],
                    "val_acc_pct": original_metric["val_acc"] * 100.0,
                    "val_macro_f1_pct": original_metric["val_macro_f1"] * 100.0,
                    "q_perturbation": q_diag,
                    "relation_scorer": _relation_score_stats(model, components["relation"]),
                    "modalities": {},
                }
                for modality in ("text", "visual"):
                    bank = components[f"bank_{modality}"]
                    state = components[modality]
                    attention_delta = state["interacted_bank"] - bank
                    differential_delta = attention_delta[:, 1:] - attention_delta[:, :1]
                    raw_difference = bank[:, 1:] - bank[:, :1]
                    eta_stats = {}
                    for order in range(state["eta"].size(1)):
                        eta = state["eta"][:, order]
                        eta_stats[str(order + 1)] = {
                            "mean": float(eta.mean().item()),
                            "node_std": float(eta.std(unbiased=False).item()),
                            "saturation_fraction_abs_ge_0_45": float((eta.abs() >= 0.45).float().mean().item()),
                        }
                    mode_mechanism["modalities"][modality] = {
                        "bank_order_rms": [_rms(bank[:, order]) for order in range(bank.size(1))],
                        "bank_order_correlations": [[_pearson(bank[:, i], bank[:, j]) for j in range(bank.size(1))] for i in range(bank.size(1))],
                        "b3_minus_b1_over_b1_frobenius": float((bank[:, 3] - bank[:, 1]).norm().item() / bank[:, 1].norm().clamp_min(1e-12).item()),
                        "eta_by_order": eta_stats,
                        "attention_increment_rms": _rms(attention_delta),
                        "attention_order_difference_rms": _rms(differential_delta),
                        "attention_order_difference_over_raw_structure_rms": _rms(differential_delta) / max(_rms(raw_difference), 1e-12),
                        "raw_structure_difference_rms": _rms(raw_difference),
                        "structural_residual_rms": _rms(state["structural_residual"]),
                        "effective_gated_residual_rms": _rms(state["semantic_gate"] * state["structural_residual"]),
                        "effective_residual_to_h0": _ratio_distribution(state["semantic_gate"] * state["structural_residual"], components[f"h0_{modality}"]),
                        "ungated_residual_to_h0": _ratio_distribution(state["structural_residual"], components[f"h0_{modality}"]),
                    }
                mechanisms.append(mode_mechanism)
                checkpoint_records.append({
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": prov["checkpoint_sha256"],
                    "checkpoint_selection": checkpoint["selection"],
                    "epoch": checkpoint["epoch"],
                    "split_path": prov["split_path"],
                    "split_sha256": prov["split_sha256"],
                    "val_count": original_metric["val_count"],
                    "class_ids_train_val_only": labels,
                })
            del model, classifier, checkpoint, components
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[diagnostics] {dataset} seed={seed} complete", flush=True)
    return mechanisms, feature_rows, q_rows


def _logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"r3.{path.parent.name}.{path.stem}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _train_variants() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    pilot, runs, _, _, _, _ = _load_manifests()
    cache = _load_data_cache(runs)
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    metric_rows, confusion_rows, class_rows, provenance_rows = [], [], [], []
    for dataset in DATASETS:
        base_cfg, data, labels = cache[dataset]
        expected_split = runs[(dataset, 42)]["split_sha256"]
        for model_name in VARIANTS:
            model_config_path = ROOT / "configs/model" / f"{model_name}.yaml"
            for seed in SEEDS:
                source = runs[(dataset, seed)]
                if source["split_sha256"] != expected_split:
                    raise RuntimeError(f"{dataset}/{seed}: source manifest has inconsistent split hashes")
                cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
                cfg.model = OmegaConf.load(model_config_path)
                cfg.seed, cfg.num_runs, cfg.device = seed, 1, str(device)
                cfg.dataset.nc_split_path = source["split_path"]
                cfg.task.evaluate_test = False
                cfg.task.eval_modality_masks.enabled = False
                run_dir = OUTPUT / "runs" / dataset / model_name / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = run_dir / "best_val.pt"
                cfg.task.save_ckpt_path = str(checkpoint_path)
                cfg.paths.output_root = str(run_dir)
                resolved = OmegaConf.to_container(cfg, resolve=True)
                (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                log = _logger(run_dir / "train.log")
                # Parameter counting is outside the timed trainer and RNG is reset by _run_single_nc.
                set_seed(seed)
                model = build_model(cfg, _data_info(data))
                encoder_parameters = count_parameters(model)
                head_parameters = int(model.out_dim) * int(data.num_classes) + int(data.num_classes)
                del model
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                    torch.cuda.init()
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                result = _run_single_nc(cfg, data, device, log, 0, None, labels)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    peak_memory_mib = float(torch.cuda.max_memory_allocated(device) / 1024**2)
                else:
                    peak_memory_mib = None
                runtime = time.perf_counter() - start
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if checkpoint.get("selection") != "best_val_accuracy" or bool(cfg.task.evaluate_test):
                    raise RuntimeError(f"bad checkpoint selection or test evaluation enabled: {run_dir}")
                if checkpoint.get("epoch") is None:
                    raise RuntimeError(f"missing selected epoch in checkpoint: {run_dir}")
                # Verify serialized selected checkpoint on the same masked validation labels.
                set_seed(seed)
                model = build_model(cfg, _data_info(data)).to(device).eval()
                classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device).eval()
                model.load_state_dict(checkpoint["model_state"])
                classifier.load_state_dict(checkpoint["head_state"])
                with torch.no_grad():
                    logits = classifier(model(data.x.to(device), data.edge_index.to(device))[0])
                    metric, confusion = _val_metrics(logits, data, labels)
                if abs(metric["val_acc"] - float(result["val_acc"])) > 1e-6 or abs(metric["val_macro_f1"] - float(result["val_macro_f1"])) > 1e-6:
                    raise RuntimeError(f"saved best checkpoint does not reproduce training metrics: {dataset}/{model_name}/{seed}")
                for row in confusion["per_class"]:
                    class_rows.append({"dataset": dataset, "model": model_name, "seed": seed, **row})
                confusion_rows.append({"dataset": dataset, "model": model_name, "seed": seed, **confusion})
                metric_rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "split_seed": 42,
                    "split_sha256": expected_split,
                    "protocol_version": str(cfg.task.protocol_version),
                    "selection": "best_val_accuracy",
                    "best_epoch": int(checkpoint["epoch"]),
                    "val_acc_pct": metric["val_acc"] * 100.0,
                    "val_macro_f1_pct": metric["val_macro_f1"] * 100.0,
                    "runtime_seconds": runtime,
                    "peak_gpu_memory_mib": peak_memory_mib,
                    "encoder_parameters": encoder_parameters,
                    "classifier_parameters": head_parameters,
                    "total_trainable_parameters": encoder_parameters + head_parameters,
                    "anchor_alpha": float(cfg.model.get("anchor_alpha", 0.1)) if model_name.endswith("_ua") else None,
                    "evaluate_test": False,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256(checkpoint_path),
                })
                provenance_rows.append({
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "source_model": "cosi_mag_vnext",
                    "source_run_manifest": str(PILOT_MANIFEST.resolve()),
                    "source_resolved_config_sha256": source["resolved_config_sha256"],
                    "variant_config": str(model_config_path.relative_to(ROOT)),
                    "variant_config_sha256": sha256(model_config_path),
                    "split_path": source["split_path"],
                    "split_sha256": expected_split,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": sha256(checkpoint_path),
                    "selection": checkpoint["selection"],
                    "epoch": checkpoint["epoch"],
                    "evaluate_test": False,
                    "validation_label_universe": labels,
                    "task_hyperparameters": {
                        key: OmegaConf.to_container(cfg.task.get(key)) if isinstance(cfg.task.get(key), (dict, list)) else cfg.task.get(key)
                        for key in ("protocol_version", "training_mode", "optimizer", "epochs", "lr", "weight_decay", "batch_size", "inference_mode", "inference_batch_size", "eval_every", "patience", "early_stop_min_epoch", "early_stop_min_delta", "grad_clip", "scheduler", "evaluate_test")
                    },
                })
                log.info("Saved and reloaded best validation checkpoint; val_acc=%.8f val_macro_f1=%.8f", metric["val_acc"], metric["val_macro_f1"])
                log.handlers.clear()
                del model, classifier, checkpoint, logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[train] {dataset}/{model_name}/seed{seed}: acc={metric['val_acc']*100:.4f} f1={metric['val_macro_f1']*100:.4f} epoch={metric_rows[-1]['best_epoch']} runtime={runtime:.1f}s", flush=True)
    return metric_rows, confusion_rows, class_rows, provenance_rows


def _aggregate_metrics(rows: list[dict]) -> dict:
    aggregate = {}
    for dataset in DATASETS:
        aggregate[dataset] = {}
        for model in VARIANTS:
            subset = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            aggregate[dataset][model] = {
                "runs": len(subset),
                "val_acc_pct_mean": mean(r["val_acc_pct"] for r in subset),
                "val_acc_pct_population_std": pstdev(r["val_acc_pct"] for r in subset),
                "val_macro_f1_pct_mean": mean(r["val_macro_f1_pct"] for r in subset),
                "val_macro_f1_pct_population_std": pstdev(r["val_macro_f1_pct"] for r in subset),
                "best_epoch_mean": mean(r["best_epoch"] for r in subset),
                "runtime_seconds_mean": mean(r["runtime_seconds"] for r in subset),
                "peak_gpu_memory_mib_mean": mean(float(r["peak_gpu_memory_mib"]) for r in subset if r["peak_gpu_memory_mib"] is not None) if subset and subset[0]["peak_gpu_memory_mib"] is not None else None,
                "parameter_count": int(subset[0]["total_trainable_parameters"]) if subset else None,
            }
    return aggregate


def run(mode: str) -> None:
    pilot, _, _, _, r2_intervention, r2_baselines = _load_manifests()
    REPORT.mkdir(parents=True, exist_ok=True)
    if mode == "compare":
        with (REPORT / "variant_validation_metrics.csv").open(encoding="utf-8", newline="") as handle:
            metrics = list(csv.DictReader(handle))
        if len(metrics) != len(DATASETS) * len(VARIANTS) * len(SEEDS):
            raise RuntimeError(f"expected 27 variant runs, found {len(metrics)}")
        aggregate = _aggregate_metrics([
            {**r, **{k: float(r[k]) for k in ("val_acc_pct", "val_macro_f1_pct", "best_epoch", "runtime_seconds")}}
            for r in metrics
        ])
        existing_vnext = [
            {k: row[k] for k in ("dataset", "seed", "val_acc_pct", "val_macro_f1_pct", "checkpoint_path")}
            for row in r2_intervention["runs"] if row["condition"] == "original"
        ]
        vnext_lookup = {
            (row["dataset"], int(row["seed"])): row
            for row in existing_vnext
        }
        paired_deltas = []
        for row in metrics:
            reference = vnext_lookup[(row["dataset"], int(row["seed"]))]
            paired_deltas.append({
                "dataset": row["dataset"],
                "model": row["model"],
                "seed": int(row["seed"]),
                "delta_val_acc_pp_vs_fixed_vnext": float(row["val_acc_pct"]) - float(reference["val_acc_pct"]),
                "delta_val_macro_f1_pp_vs_fixed_vnext": float(row["val_macro_f1_pct"]) - float(reference["val_macro_f1_pct"]),
            })
        paired_delta_summary = {}
        for dataset in DATASETS:
            paired_delta_summary[dataset] = {}
            for model_name in VARIANTS:
                subset = [r for r in paired_deltas if r["dataset"] == dataset and r["model"] == model_name]
                paired_delta_summary[dataset][model_name] = {
                    key: mean(r[key] for r in subset)
                    for key in ("delta_val_acc_pp_vs_fixed_vnext", "delta_val_macro_f1_pp_vs_fixed_vnext")
                }
        historical = r2_baselines["comparison_references_percent"]["CoSI_MAG_Final_historical"]
        variant_provenance_path = REPORT / "variant_provenance.json"
        variant_provenance = json.loads(variant_provenance_path.read_text(encoding="utf-8"))
        variant_provenance["original_vnext_implementation_commit"] = pilot.get("implementation_commit")
        variant_provenance["pilot_manifest_sha256"] = sha256(PILOT_MANIFEST)
        variant_provenance["aggregate"] = aggregate
        write_json(variant_provenance_path, variant_provenance)
        write_json(REPORT / "comparison_references.json", {
            "test_labels_used_for_this_round": False,
            "variant_results": aggregate,
            "variant_per_seed": [
                {k: row[k] for k in ("dataset", "model", "seed", "val_acc_pct", "val_macro_f1_pct", "best_epoch")}
                for row in metrics
            ],
            "paired_deltas_vs_fixed_vnext": paired_deltas,
            "paired_delta_mean_vs_fixed_vnext": paired_delta_summary,
            "vnext_fixed_checkpoint_validation_from_round2": existing_vnext,
            "vnext_run_manifest_source": str(PILOT_MANIFEST.resolve()),
            "round2_baselines": r2_baselines["aggregate"],
            "historical_cosi_mag_final_validation_reference": historical,
            "historical_comparability_note": r2_baselines["comparison_references_percent"].get("comparability_note"),
            "interpretation_limit": "VNext variants are fresh controlled retrains on the shared seed-42 split. Historical CoSI Final and round-two VNext/GNN references have different model configurations and are context, not a controlled ablation. Macro-F1 is computed over train/validation-observed classes for new runs.",
        })
        print(f"[done] comparisons={REPORT / 'comparison_references.json'}", flush=True)
        return
    diagnostics = {}
    metrics, confusions, classes, provenance = [], [], [], []
    checkpoint_records = []
    feature_rows, q_rows = [], []
    if mode == "diagnostics":
        mechanisms, feature_rows, q_rows = _diagnose_original_checkpoints()
        diagnostics = {"fixed_checkpoint_mechanisms": mechanisms, "feature_similarity": feature_rows, "q_perturbation": q_rows}
        # Include checkpoint hashes and split metadata from the generated mechanism rows.
        checkpoint_records = [
            {k: row[k] for k in ("dataset", "seed", "checkpoint_path", "checkpoint_sha256", "checkpoint_epoch", "split_path", "split_sha256", "val_acc_pct", "val_macro_f1_pct")}
            for row in mechanisms
        ]
        write_json(REPORT / "fixed_checkpoint_diagnostics.json", {
            "base_commit": "3396f603fdd4594b1e10ced5452f1b083edd4dcd",
            "original_vnext_implementation_commit": pilot.get("implementation_commit"),
            "pilot_manifest_sha256": sha256(PILOT_MANIFEST),
            "round2_provenance_sha256": sha256(ROUND2_PROVENANCE),
            "test_labels_used": False,
            "split": "existing seed-42 NC split; test labels masked immediately after load",
            "checkpoint_selection": "original fixed best_val_accuracy checkpoints; no retraining or reselection",
            "definitions": {
                "neighbor_pairwise_rms_gap": "RMS similarity difference over unordered distinct neighbors, computed exactly from within-neighborhood population variance",
                "relative_neighbor_pairwise_rms_gap": "per-node neighbor-pairwise RMS gap divided by global physical-edge cosine standard deviation",
                "q_perturbation": f"permute q rows with fixed seed {Q_SHUFFLE_SEED}; propagation banks, learned relation operators, features, model and head are held fixed; metrics use validation nodes only",
                "attention_contribution": "attention increment and its order-differential component; denominator is RMS(B_k-B_0), k=1..K",
                "eta_saturation": "fraction with |eta_k| >= 0.45, i.e. within 10% of the tanh bound 0.5, reported separately for each order and modality",
                "residual_ratio": "per-node RMS of raw U or gated U divided by per-node RMS of the original projected modality representation H0",
                "B3_B1_ratio": "Frobenius norm(B3-B1)/Frobenius norm(B1)",
            },
            "fixed_checkpoint_records": checkpoint_records,
            **diagnostics,
        })
        write_csv(REPORT / "feature_similarity.csv", feature_rows)
        write_csv(REPORT / "q_perturbation.csv", q_rows)
    if mode == "train":
        metrics, confusions, classes, provenance = _train_variants()
        write_csv(REPORT / "variant_validation_metrics.csv", metrics)
        write_csv(REPORT / "variant_per_class_f1.csv", classes)
        write_json(REPORT / "variant_confusion_matrices.json", {
            "test_labels_used": False,
            "split_seed": 42,
            "label_universe": "classes present in train or validation labels only, after masking test labels",
            "matrices": confusions,
        })
        write_json(REPORT / "variant_provenance.json", {
            "base_commit": "3396f603fdd4594b1e10ced5452f1b083edd4dcd",
            "original_vnext_implementation_commit": pilot.get("implementation_commit"),
            "pilot_manifest_sha256": sha256(PILOT_MANIFEST),
            "protocol_version": "unified_full_graph_nc_v1",
            "evaluate_test": False,
            "test_labels_used": False,
            "selection": "best validation Accuracy; strict improvement by configured early_stop_min_delta",
            "split_seed": 42,
            "source_pilot_manifest": str(PILOT_MANIFEST.resolve()),
            "runs": provenance,
            "aggregate": _aggregate_metrics(metrics),
        })
    print(f"[done] mode={mode} reports={REPORT}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("diagnostics", "train", "compare"))
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
