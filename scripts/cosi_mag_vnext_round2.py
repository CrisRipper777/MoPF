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
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, f1_score
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from src.data import load_mag_data
from src.models import build_model
from src.tasks.nc import _run_single_nc
from src.utils.summary import count_parameters


REPORT_ROOT = ROOT / "reports/cosi_mag_vnext/round2"
PILOT_ROOT = ROOT / "reports/cosi_mag_vnext/run_manifest.json"
SPLIT_ROOT = Path("/hdd1/DataInHere/YHF/data/MAGB_split")
DATASETS = ("Movies", "Grocery", "Reddit-S")
SEEDS = (42, 43, 44)
CONDITIONS = (
    "original",
    "uniform_operator_keep_q",
    "shuffle_q_keep_operator",
    "attention_update_off",
    "structural_residual_off",
)
Q_SHUFFLE_SEED = 20261012
PILOT_BASELINES = {
    "Movies": {"acc": 55.09, "macro_f1": 46.55},
    "Grocery": {"acc": 83.35, "macro_f1": 76.38},
    "Reddit-S": {"acc": 96.23, "macro_f1": 92.23},
}
HISTORICAL_COSI_FINAL = {
    "Movies": {"acc_mean": 57.51, "acc_std": 0.26, "macro_f1_mean": 49.96, "macro_f1_std": 0.87},
    "Grocery": {"acc_mean": 83.91, "acc_std": 0.16, "macro_f1_mean": 78.47, "macro_f1_std": 0.71},
    "Reddit-S": {"acc_mean": 96.37, "acc_std": 0.05, "macro_f1_mean": 93.24, "macro_f1_std": 0.13},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_runs() -> tuple[dict, dict[tuple[str, int], dict]]:
    manifest = json.loads(PILOT_ROOT.read_text(encoding="utf-8"))
    rows = {(row["dataset"], int(row["seed"])): row for row in manifest["runs"]}
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if set(rows) != expected:
        raise RuntimeError(f"pilot manifest run keys mismatch: missing={expected - set(rows)}")
    return manifest, rows


def config_from_run(run: dict):
    return OmegaConf.create(run["resolved_config"])


def data_info(data) -> dict:
    return {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0,
    }


def mask_test_labels(data) -> list[int]:
    """Mask test labels immediately; all metric labels come from train/validation."""
    if data.test_idx is None:
        raise RuntimeError(f"{data.name}: no test index available for label masking")
    data.y = data.y.clone()
    data.y[data.test_idx] = -100
    allowed_idx = torch.cat((data.train_idx.cpu(), data.val_idx.cpu()))
    labels = sorted(int(v) for v in torch.unique(data.y[allowed_idx]).tolist() if int(v) >= 0)
    if not labels:
        raise RuntimeError(f"{data.name}: no train/validation labels")
    return labels


def load_dataset_cache(rows: dict[tuple[str, int], dict]):
    cache = {}
    for dataset in DATASETS:
        run = rows[(dataset, 42)]
        cfg = config_from_run(run)
        cfg.task.evaluate_test = False
        data = load_mag_data(cfg, task_name="nc", seed=42)
        labels = mask_test_labels(data)
        split_path = Path(data.info["nc_split_path"])
        if str(split_path.resolve()) != str(Path(run["split_path"]).resolve()):
            raise RuntimeError(f"{dataset}: loaded split path differs from pilot manifest")
        actual_split_hash = sha256(split_path)
        if actual_split_hash != run["split_sha256"]:
            raise RuntimeError(f"{dataset}: split hash differs from pilot manifest")
        cache[dataset] = (cfg, data, labels)
    return cache


def _metric_summary(logits: torch.Tensor, data, labels: list[int]) -> tuple[dict, dict]:
    val_idx = data.val_idx.cpu()
    target = data.y[val_idx].cpu().long()
    pred = logits[val_idx.to(logits.device)].argmax(dim=-1).cpu().long()
    matrix = confusion_matrix(target.numpy(), pred.numpy(), labels=labels)
    f1 = f1_score(target.numpy(), pred.numpy(), labels=labels, average=None, zero_division=0)
    result = {
        "val_acc": float((target == pred).float().mean().item()),
        "val_macro_f1": float(f1_score(target.numpy(), pred.numpy(), labels=labels, average="macro", zero_division=0)),
        "val_count": int(target.numel()),
    }
    class_result = [
        {
            "class_id": int(label),
            "support": int(matrix[index].sum()),
            "predicted_count": int(matrix[:, index].sum()),
            "f1": float(f1[index]),
        }
        for index, label in enumerate(labels)
    ]
    return result, {"labels": labels, "matrix": matrix.tolist(), "per_class": class_result}


def _rms(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.float().square().mean().sqrt().item())


def _pearson_flat(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.float().reshape(-1)
    b = right.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denominator = torch.sqrt(a.square().sum() * b.square().sum())
    if denominator <= 0:
        return 0.0
    return float((a * b).sum().div(denominator).item())


def _operator_sparse(index: torch.Tensor, weight: torch.Tensor, n: int) -> torch.Tensor:
    # torch sparse matrices are row=target, col=source, matching scatter messages.
    matrix_index = torch.stack((index[1], index[0]), dim=0)
    return torch.sparse_coo_tensor(matrix_index, weight, (n, n), device=weight.device).coalesce()


def _linear_sparse_norm(items: list[tuple[float, torch.Tensor]]) -> float:
    indices = []
    values = []
    shape = None
    device = None
    for scale, matrix in items:
        matrix = matrix.coalesce()
        indices.append(matrix.indices())
        values.append(matrix.values() * scale)
        shape = matrix.shape
        device = matrix.device
    combined = torch.sparse_coo_tensor(torch.cat(indices, dim=1), torch.cat(values), shape, device=device).coalesce()
    return float(combined.values().float().square().sum().sqrt().item())


def operator_differences(relation: dict, h_text: torch.Tensor, h_visual: torch.Tensor) -> dict:
    physical_edge_index = relation["physical_edge_index"]
    num_nodes = int(h_text.size(0))
    ones = torch.ones(physical_edge_index.size(1), device=h_text.device, dtype=h_text.dtype)
    uniform_index, uniform_weight = gcn_norm(
        physical_edge_index,
        edge_weight=ones,
        num_nodes=num_nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=h_text.dtype,
    )
    if uniform_weight is None:
        uniform_weight = torch.ones(uniform_index.size(1), device=h_text.device, dtype=h_text.dtype)
    uniform = _operator_sparse(uniform_index, uniform_weight, num_nodes)
    learned_text = _operator_sparse(relation["P_text_edge_index"], relation["P_text_edge_weight"], num_nodes)
    learned_visual = _operator_sparse(relation["P_visual_edge_index"], relation["P_visual_edge_weight"], num_nodes)
    uniform_norm = _linear_sparse_norm([(1.0, uniform)])
    midpoint_norm = _linear_sparse_norm([(0.5, learned_text), (0.5, learned_visual)])
    return {
        "relative_frobenius_text_vs_uniform": _linear_sparse_norm([(1.0, learned_text), (-1.0, uniform)]) / max(uniform_norm, 1e-12),
        "relative_frobenius_visual_vs_uniform": _linear_sparse_norm([(1.0, learned_visual), (-1.0, uniform)]) / max(uniform_norm, 1e-12),
        "relative_frobenius_text_vs_visual": _linear_sparse_norm([(1.0, learned_text), (-1.0, learned_visual)]) / max(midpoint_norm, 1e-12),
        "frobenius_norm_uniform": uniform_norm,
        "frobenius_norm_learned_text": _linear_sparse_norm([(1.0, learned_text)]),
        "frobenius_norm_learned_visual": _linear_sparse_norm([(1.0, learned_visual)]),
        "uniform_nnz": int(uniform._nnz()),
        "learned_text_nnz": int(learned_text._nnz()),
        "learned_visual_nnz": int(learned_visual._nnz()),
        "uniform_edge_index": uniform_index,
        "uniform_edge_weight": uniform_weight,
    }


def _run_modality(
    interaction,
    h0: torch.Tensor,
    bank: torch.Tensor,
    q: torch.Tensor,
    modality: str,
    *,
    attention_enabled: bool,
    residual_enabled: bool,
) -> dict:
    order_embedding = getattr(interaction, f"order_embedding_{modality}")
    token_input = interaction.token_norm(bank + order_embedding.unsqueeze(0))
    query = interaction.query(token_input)
    key = interaction.key(token_input)
    value = interaction.value(bank)
    logits = torch.matmul(query, key.transpose(-1, -2)) * interaction.attention_scale
    relation_bias = interaction.condition_order_bias[modality](q)
    logits = logits + relation_bias.unsqueeze(1)
    attention = torch.softmax(logits, dim=-1)
    attention_output = torch.matmul(interaction.attention_dropout(attention), value)
    attention_gate = torch.sigmoid(interaction.attention_gate_logit[modality])
    attention_increment = attention_gate * attention_output
    if not attention_enabled:
        attention_increment = torch.zeros_like(attention_increment)
    interacted = bank + attention_increment

    differences = interacted[:, 1:, :] - interacted[:, :1, :]
    condition = q.unsqueeze(1).expand(-1, interaction.max_order, -1)
    conditioned = torch.cat((differences, condition), dim=-1)
    conditional = interaction.conditional_score[modality](conditioned).squeeze(-1)
    eta = 0.5 * torch.tanh(
        interaction.shared_order_prior.unsqueeze(0)
        + interaction.modality_order_bias[modality].unsqueeze(0)
        + conditional
    )
    structural_residual = (eta.unsqueeze(-1) * differences).sum(dim=1)
    semantic_gate = torch.sigmoid(interaction.semantic_gate[modality](torch.cat((h0, q), dim=-1)))
    applied_residual = semantic_gate * structural_residual if residual_enabled else torch.zeros_like(structural_residual)
    y = h0 + applied_residual
    z = interaction.refine_norm[modality](y + interaction.refine[modality](y))
    return {
        "z": z,
        "eta": eta,
        "semantic_gate": semantic_gate,
        "structural_residual": structural_residual,
        "applied_residual": applied_residual,
        "attention_increment": attention_increment,
        "attention": attention,
        "raw_differences": bank[:, 1:, :] - bank[:, :1, :],
        "updated_differences": differences,
    }


def _compose(model, components: dict, q: torch.Tensor, bank_text: torch.Tensor, bank_visual: torch.Tensor, *, attention_enabled: bool, residual_enabled: bool) -> dict:
    text = _run_modality(model.interaction, components["h0_text"], bank_text, q, "text", attention_enabled=attention_enabled, residual_enabled=residual_enabled)
    visual = _run_modality(model.interaction, components["h0_visual"], bank_visual, q, "visual", attention_enabled=attention_enabled, residual_enabled=residual_enabled)
    fused = torch.cat((text["z"], visual["z"]), dim=-1)
    z = model.output_norm(model.fusion_skip(fused) + model.fusion_mlp(fused))
    return {"z": z, "text": text, "visual": visual}


def _ratio_distribution(numerator: torch.Tensor, denominator: torch.Tensor) -> dict:
    numerator = numerator.float().square().mean(dim=-1).sqrt()
    denominator = denominator.float().square().mean(dim=-1).sqrt().clamp_min(1e-12)
    ratios = numerator / denominator
    probabilities = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    quantiles = torch.tensor(probabilities, device=ratios.device)
    q_values = torch.quantile(ratios, quantiles)
    return {
        "mean": float(ratios.mean().item()),
        "std": float(ratios.std(unbiased=False).item()),
        "fraction_above_one": float((ratios > 1.0).float().mean().item()),
        "quantiles": {str(q): float(v) for q, v in zip(probabilities, q_values.cpu().tolist())},
    }


def _mechanism_summary(components: dict, relation: dict, learned: dict, composed: dict) -> dict:
    output = {
        "operator_differences": {key: value for key, value in learned.items() if not isinstance(value, torch.Tensor)},
        "q_node_variation": {
            "mean_node_norm": float(relation["q"].norm(dim=-1).mean().item()),
            "node_centered_rms": _rms(relation["q"] - relation["q"].mean(dim=0, keepdim=True)),
            "mean_per_dimension_node_std": float(relation["q"].std(dim=0, unbiased=False).mean().item()),
            "mean_per_node_feature_std": float(relation["q"].std(dim=1, unbiased=False).mean().item()),
        },
        "modalities": {},
    }
    for modality in ("text", "visual"):
        bank = components[f"bank_{modality}"]
        state = composed[modality]
        correlations = [[_pearson_flat(bank[:, i], bank[:, j]) for j in range(bank.size(1))] for i in range(bank.size(1))]
        raw_d = bank[:, 1:, :] - bank[:, :1, :]
        delta = state["attention_increment"]
        differential_delta = delta[:, 1:, :] - delta[:, :1, :]
        original_d_rms = _rms(raw_d)
        output["modalities"][modality] = {
            "chebyshev_order_rms": {str(order): _rms(bank[:, order, :]) for order in range(bank.size(1))},
            "chebyshev_order_pearson_correlation": correlations,
            "attention_increment_rms": _rms(delta),
            "attention_increment_to_raw_structural_difference_rms": _rms(delta[:, 1:, :]) / max(original_d_rms, 1e-12),
            "differential_attention_increment_rms": _rms(differential_delta),
            "differential_attention_increment_to_raw_structural_difference_rms": _rms(differential_delta) / max(original_d_rms, 1e-12),
            "raw_structural_difference_rms": original_d_rms,
            "eta_by_order": {
                str(order + 1): {
                    "mean": float(state["eta"][:, order].mean().item()),
                    "node_std": float(state["eta"][:, order].std(unbiased=False).item()),
                }
                for order in range(state["eta"].size(1))
            },
            "semantic_gate_mean": float(state["semantic_gate"].mean().item()),
            "effective_residual_to_h0_node_ratio": _ratio_distribution(state["applied_residual"], components[f"h0_{modality}"]),
            "raw_residual_to_h0_node_ratio": _ratio_distribution(state["structural_residual"], components[f"h0_{modality}"]),
        }
    output["chebyshev_same_order_text_visual_pearson"] = {
        str(order): _pearson_flat(components["bank_text"][:, order], components["bank_visual"][:, order])
        for order in range(components["bank_text"].size(1))
    }
    return output


@torch.no_grad()
def diagnose() -> None:
    _, runs = read_runs()
    cache = load_dataset_cache(runs)
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    metrics_rows = []
    mechanism_runs = []
    class_runs = []
    confusion_runs = []
    checkpoint_rows = []

    for dataset in DATASETS:
        _base_cfg, data, labels = cache[dataset]
        x_device = data.x.to(device)
        edge_device = data.edge_index.to(device)
        for seed in SEEDS:
            run = runs[(dataset, seed)]
            cfg = config_from_run(run)
            cfg.task.evaluate_test = False
            model = build_model(cfg, data_info(data)).to(device)
            classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
            ckpt_path = Path(run["checkpoint_path"])
            if not ckpt_path.is_file():
                raise FileNotFoundError(ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if int(checkpoint.get("seed", -1)) != seed or checkpoint.get("selection") != "best_val_accuracy":
                raise RuntimeError(f"unexpected checkpoint metadata for {dataset} seed {seed}")
            model.load_state_dict(checkpoint["model_state"])
            classifier.load_state_dict(checkpoint["head_state"])
            model.eval()
            classifier.eval()

            forward_z = model(x_device, edge_device)[0]
            components = model._encode_components(x_device, edge_device)
            original_z = components["z"]
            relation = components["relation"]
            learned_ops = operator_differences(relation, components["h0_text"], components["h0_visual"])
            uniform_text_bank = model_module_cheb(components["h0_text"], learned_ops["uniform_edge_index"], learned_ops["uniform_edge_weight"], model.max_order)
            uniform_visual_bank = model_module_cheb(components["h0_visual"], learned_ops["uniform_edge_index"], learned_ops["uniform_edge_weight"], model.max_order)

            canonical = _compose(model, components, relation["q"], components["bank_text"], components["bank_visual"], attention_enabled=True, residual_enabled=True)
            max_reconstruction_error = float((canonical["z"] - components["z"]).abs().max().item())
            repeated_forward_error = float((components["z"] - forward_z).abs().max().item())
            if max_reconstruction_error > 1e-6:
                raise RuntimeError(f"diagnostic reconstruction mismatch {dataset}/{seed}: {max_reconstruction_error}")
            if repeated_forward_error > 1e-4:
                raise RuntimeError(f"repeated eval forward is unstable for {dataset}/{seed}: {repeated_forward_error}")

            generator = torch.Generator(device="cpu").manual_seed(Q_SHUFFLE_SEED)
            permutation = torch.randperm(data.num_nodes, generator=generator).to(device)
            q_shuffled = relation["q"][permutation]
            scenarios = {
                "original": original_z,
                "uniform_operator_keep_q": _compose(model, components, relation["q"], uniform_text_bank, uniform_visual_bank, attention_enabled=True, residual_enabled=True)["z"],
                "shuffle_q_keep_operator": _compose(model, components, q_shuffled, components["bank_text"], components["bank_visual"], attention_enabled=True, residual_enabled=True)["z"],
                "attention_update_off": _compose(model, components, relation["q"], components["bank_text"], components["bank_visual"], attention_enabled=False, residual_enabled=True)["z"],
                "structural_residual_off": _compose(model, components, relation["q"], components["bank_text"], components["bank_visual"], attention_enabled=True, residual_enabled=False)["z"],
            }
            condition_summaries = {}
            original_metric = None
            original_classes = None
            for condition in CONDITIONS:
                logits = classifier(scenarios[condition])
                metric, classes = _metric_summary(logits, data, labels)
                if condition == "original":
                    original_metric = metric
                    original_classes = classes
                    checkpoint_val = checkpoint["metrics"]
                    if abs(metric["val_acc"] - float(checkpoint_val["val_acc"])) > 1e-6:
                        raise RuntimeError(f"validation accuracy mismatch vs saved checkpoint {dataset}/{seed}")
                    if abs(metric["val_macro_f1"] - float(checkpoint_val["val_macro_f1"])) > 1e-6:
                        raise RuntimeError(f"validation Macro-F1 mismatch vs saved checkpoint {dataset}/{seed}")
                    # Round-one F1 used the full supervised label universe. All labels
                    # used here are train/validation only; a mismatch is reported below.
                delta_acc = (metric["val_acc"] - (original_metric or metric)["val_acc"]) * 100.0
                delta_f1 = (metric["val_macro_f1"] - (original_metric or metric)["val_macro_f1"]) * 100.0
                condition_summaries[condition] = {**metric, "delta_acc_pp": delta_acc, "delta_macro_f1_pp": delta_f1}
                metrics_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "condition": condition,
                    "val_acc_pct": metric["val_acc"] * 100.0,
                    "val_macro_f1_pct": metric["val_macro_f1"] * 100.0,
                    "delta_acc_pp": delta_acc,
                    "delta_macro_f1_pp": delta_f1,
                    "checkpoint_path": str(ckpt_path.resolve()),
                })
                if condition == "original":
                    confusion_runs.append({"dataset": dataset, "seed": seed, **classes})
                    for row in classes["per_class"]:
                        class_runs.append({"dataset": dataset, "seed": seed, **row})
            diagnostics = _mechanism_summary(components, relation, learned_ops, canonical)
            diagnostics["checkpoint_selection_metric"] = "validation_accuracy"
            diagnostics["saved_checkpoint_val_macro_f1"] = float(checkpoint["metrics"]["val_macro_f1"])
            diagnostics["train_val_label_universe"] = labels
            diagnostics["reconstruction_max_abs_error"] = max_reconstruction_error
            diagnostics["repeated_forward_max_abs_error"] = repeated_forward_error
            diagnostics["q_shuffle_seed"] = Q_SHUFFLE_SEED
            mechanism_runs.append({"dataset": dataset, "seed": seed, "checkpoint_path": str(ckpt_path.resolve()), **diagnostics})
            checkpoint_rows.append({
                "dataset": dataset,
                "seed": seed,
                "checkpoint_path": str(ckpt_path.resolve()),
                "checkpoint_sha256": sha256(ckpt_path),
                "checkpoint_selection": checkpoint["selection"],
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_saved_val_acc": float(checkpoint["metrics"]["val_acc"]),
                "checkpoint_saved_val_macro_f1": float(checkpoint["metrics"]["val_macro_f1"]),
                "split_path": run["split_path"],
                "split_sha256": run["split_sha256"],
                "val_count": int(data.val_idx.numel()),
                "class_ids_from_train_val_only": labels,
            })
            del model, classifier, checkpoint, components, relation, scenarios, original_z
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[diagnose] {dataset} seed={seed} done", flush=True)

    acc_summary = {}
    for dataset in DATASETS:
        rows = [row for row in metrics_rows if row["dataset"] == dataset and row["condition"] == "original"]
        acc_summary[dataset] = {
            "train_val_label_universe_only": True,
            "val_acc_percent_mean": mean(row["val_acc_pct"] for row in rows),
            "val_acc_percent_population_std": pstdev(row["val_acc_pct"] for row in rows),
            "val_macro_f1_percent_mean": mean(row["val_macro_f1_pct"] for row in rows),
            "val_macro_f1_percent_population_std": pstdev(row["val_macro_f1_pct"] for row in rows),
            "round1_pilot_reference_percent": PILOT_BASELINES[dataset],
            "historical_cosi_mag_final_validation_percent": HISTORICAL_COSI_FINAL[dataset],
        }
    intervention_summary = {}
    for dataset in DATASETS:
        intervention_summary[dataset] = {}
        for condition in CONDITIONS:
            rows = [row for row in metrics_rows if row["dataset"] == dataset and row["condition"] == condition]
            intervention_summary[dataset][condition] = {
                key: {"mean": mean(float(row[key]) for row in rows), "population_std": pstdev(float(row[key]) for row in rows)}
                for key in ("val_acc_pct", "val_macro_f1_pct", "delta_acc_pp", "delta_macro_f1_pp")
            }
    class_aggregate = {}
    for dataset in DATASETS:
        class_aggregate[dataset] = []
        for label in sorted({row["class_id"] for row in class_runs if row["dataset"] == dataset}):
            rows = [row for row in class_runs if row["dataset"] == dataset and row["class_id"] == label]
            class_aggregate[dataset].append({
                "class_id": label,
                "support": rows[0]["support"],
                "predicted_count_mean": mean(row["predicted_count"] for row in rows),
                "f1_mean": mean(row["f1"] for row in rows),
                "f1_population_std": pstdev(row["f1"] for row in rows),
            })
    write_json(REPORT_ROOT / "intervention_metrics.json", {
        "units": "percent; deltas are percentage points from the same checkpoint's original output",
        "checkpoint_selection": "validation_accuracy; unchanged fixed checkpoints; no retraining",
        "test_labels_used": False,
        "metric_label_universe": "unique labels in training and validation nodes only; test-node labels were masked before any metric computation",
        "interventions": {
            "uniform_operator_keep_q": "replace both learned P operators with symmetrically normalized uniform physical adjacency plus one self-loop; preserve checkpoint q",
            "shuffle_q_keep_operator": f"permute q rows using fixed seed {Q_SHUFFLE_SEED}; preserve learned P and Chebyshev banks",
            "attention_update_off": "set only omega_m * attention_output contribution to zero; recompute downstream eta/U using unchanged q and learned P",
            "structural_residual_off": "preserve attention, q, P, eta, and gate; set only the applied final gated U contribution to zero",
        },
        "aggregate": intervention_summary,
        "runs": [
            {"dataset": row["dataset"], "seed": row["seed"], "condition": row["condition"], "val_acc_pct": row["val_acc_pct"], "val_macro_f1_pct": row["val_macro_f1_pct"], "delta_acc_pp": row["delta_acc_pp"], "delta_macro_f1_pp": row["delta_macro_f1_pp"], "checkpoint_path": row["checkpoint_path"]}
            for row in metrics_rows
        ],
    })
    write_csv(REPORT_ROOT / "intervention_metrics.csv", metrics_rows)
    write_json(REPORT_ROOT / "mechanism_diagnostics.json", {
        "units": "relative operator differences, RMS ratios, eta/q/gate statistics are dimensionless; response RMS uses projected feature units",
        "definitions": {
            "operator_relative_frobenius_text_visual_denominator": "Frobenius norm of the mean of learned P_text and P_visual",
            "operator_relative_frobenius_vs_uniform_denominator": "Frobenius norm of the uniform normalized operator",
            "chebyshev_correlation": "Pearson correlation of flattened node-by-feature responses for each order pair",
            "attention_increment_ratio": "RMS of omega*attention_output on orders 1..K divided by RMS of raw B_k-B_0; also reports the differential increment after subtracting order-0 attention update",
            "residual_ratio_distribution": "per-node RMS(g*U) divided by per-node RMS(H0), with empirical quantiles",
            "eta_node_std": "population standard deviation across nodes separately for each order and modality",
        },
        "runs": mechanism_runs,
    })
    write_json(REPORT_ROOT / "per_class_f1.json", {
        "units": "per-class F1 in [0,1]; confusion matrix cells are validation node counts",
        "test_labels_used": False,
        "label_universe": "training and validation labels only",
        "per_seed": class_runs,
        "mean_across_seeds": class_aggregate,
        "confusion_matrices_validation_only": confusion_runs,
    })
    write_json(REPORT_ROOT / "validation_confusion_matrices.json", {
        "test_labels_used": False,
        "split_seed": 42,
        "matrices": confusion_runs,
    })
    write_csv(REPORT_ROOT / "per_class_f1.csv", class_runs)
    write_json(REPORT_ROOT / "checkpoint_provenance.json", {
        "branch": "exp/cosi-mag-vnext-crc-cheb-rcmi",
        "review_base_commit": "9a4d136590a83546cd2544f434076edb07cca907",
        "model_code_commit": "c5a901634da8ddd6bed1c4ad01b37b67326a3da4",
        "split_seed": 42,
        "runs": checkpoint_rows,
        "validation_reproduction": acc_summary,
    })
    write_csv(REPORT_ROOT / "checkpoint_provenance.csv", checkpoint_rows)


def model_module_cheb(x, edge_index, edge_weight, max_order):
    from src.models.cosi_filter import chebyshev_bank
    return chebyshev_bank(x, edge_index, edge_weight, max_order)


def _run_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"cosi_round2.{path.parent.name}.{path.stem}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def run_baselines() -> None:
    _, runs = read_runs()
    cache = load_dataset_cache(runs)
    device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in DATASETS:
        base_cfg, data, labels = cache[dataset]
        for model_name in ("gcn", "sage"):
            model_cfg_path = ROOT / f"configs/model/{model_name}.yaml"
            for seed in SEEDS:
                cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
                cfg.model = OmegaConf.load(model_cfg_path)
                cfg.seed = seed
                cfg.num_runs = 1
                cfg.device = str(device)
                cfg.dataset.nc_split_path = str((SPLIT_ROOT / f"{dataset}_nc_seed42_train0.6_val0.2.pt").resolve())
                cfg.task.evaluate_test = False
                cfg.task.epochs = 300
                cfg.task.lr = 1e-3
                cfg.task.weight_decay = 1e-4
                cfg.task.patience = 30
                cfg.task.early_stop_min_epoch = 30
                cfg.task.early_stop_min_delta = 1e-4
                cfg.task.grad_clip = 1.0
                cfg.task.batch_size = 1024
                cfg.task.eval_every = 1
                cfg.task.inference_mode = "full"
                cfg.task.inference_batch_size = 4096
                run_dir = ROOT / f"outputs/cosi_mag_vnext_round2/baselines/{dataset}/{model_name}/seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = run_dir / "best_val.pt"
                cfg.task.save_ckpt_path = str(ckpt_path)
                logger = _run_logger(run_dir / "train.log")
                logger.info("Round-two baseline run; test labels are masked and test evaluation is disabled.")
                logger.info("Model config: %s", model_cfg_path)
                cfg_path = run_dir / "resolved_config.yaml"
                cfg_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
                info = data_info(data)
                probe = build_model(cfg, info)
                params = count_parameters(probe)
                del probe
                if device.type == "cuda":
                    torch.cuda.init()
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                metrics = _run_single_nc(cfg, data, device, logger, 0, None, labels)
                runtime = time.perf_counter() - start
                peak_mib = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if checkpoint["seed"] != seed or checkpoint["selection"] != "best_val_accuracy":
                    raise RuntimeError(f"baseline checkpoint provenance mismatch: {ckpt_path}")
                row = {
                    "dataset": dataset,
                    "model": model_name,
                    "seed": seed,
                    "split_seed": 42,
                    "split_path": str(SPLIT_ROOT / f"{dataset}_nc_seed42_train0.6_val0.2.pt"),
                    "split_sha256": sha256(SPLIT_ROOT / f"{dataset}_nc_seed42_train0.6_val0.2.pt"),
                    "protocol_version": str(cfg.task.protocol_version),
                    "checkpoint_selection": checkpoint["selection"],
                    "best_epoch": checkpoint["epoch"],
                    "val_acc": float(metrics["val_acc"]),
                    "val_macro_f1": float(metrics["val_macro_f1"]),
                    "runtime_seconds_train_eval": runtime,
                    "peak_gpu_memory_mib": peak_mib,
                    "model_parameters_encoder_only": params,
                    "device": str(device),
                    "evaluate_test": False,
                    "checkpoint_path": str(ckpt_path.resolve()),
                    "checkpoint_sha256": sha256(ckpt_path),
                    "config_path": str(cfg_path.resolve()),
                    "config_sha256": sha256(cfg_path),
                }
                write_json(run_dir / "metrics.json", row)
                rows.append(row)
                print(f"[baseline] {dataset} {model_name} seed={seed}: acc={row['val_acc']*100:.3f}% f1={row['val_macro_f1']*100:.3f}% epoch={row['best_epoch']} time={runtime:.1f}s", flush=True)
                logger.handlers.clear()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    grouped = {}
    for dataset in DATASETS:
        grouped[dataset] = {}
        for model_name in ("gcn", "sage"):
            model_rows = [row for row in rows if row["dataset"] == dataset and row["model"] == model_name]
            grouped[dataset][model_name] = {
                "runs": len(model_rows),
                "val_acc_pct_mean": mean(row["val_acc"] * 100.0 for row in model_rows),
                "val_acc_pct_population_std": pstdev(row["val_acc"] * 100.0 for row in model_rows),
                "val_macro_f1_pct_mean": mean(row["val_macro_f1"] * 100.0 for row in model_rows),
                "val_macro_f1_pct_population_std": pstdev(row["val_macro_f1"] * 100.0 for row in model_rows),
            }
    write_json(REPORT_ROOT / "baseline_metrics.json", {
        "per_run_metric_unit": "val_acc and val_macro_f1 are fractions in [0,1]; aggregate fields are percentages",
        "runner_entrypoint": "existing src.tasks.nc._run_single_nc called directly; this avoids the run_nc wrapper's test-label-based Macro-F1 class-universe discovery",
        "metric_label_universe": "unique train/validation labels only; all 20 configured classes were present",
        "protocol_version": "unified_full_graph_nc_v1",
        "model_configs": {name: {"path": f"configs/model/{name}.yaml", "sha256": sha256(ROOT / f"configs/model/{name}.yaml")} for name in ("gcn", "sage")},
        "task_config": {"path": "configs/task/nc.yaml", "sha256": sha256(ROOT / "configs/task/nc.yaml")},
        "split_seed": 42,
        "run_seeds": list(SEEDS),
        "test_metrics_used": False,
        "test_labels_masked_before_training": True,
        "settings": {"optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "epochs_max": 300, "patience": 30, "early_stop_min_epoch": 30, "early_stop_min_delta": 0.0001, "grad_clip": 1.0, "full_graph": True, "eval_every": 1, "selection": "validation_accuracy"},
        "aggregate": grouped,
        "runs": rows,
        "comparison_references_percent": {
            "VNext_round1": PILOT_BASELINES,
            "CoSI_MAG_Final_historical": HISTORICAL_COSI_FINAL,
            "comparability_note": "same dataset, seed-42 split SHA, NC protocol, and task config hash; GCN/SAGE use configs/model/{gcn,sage}.yaml, VNext uses configs/model/cosi_mag_vnext.yaml, and historical CoSI Final uses configs/model/cosi_mag_final.yaml. These are different model configurations, not controlled architectural ablations.",
        },
    })
    csv_rows = [{
        "dataset": row["dataset"], "model": row["model"], "seed": row["seed"], "best_epoch": row["best_epoch"],
        "val_acc_pct": f"{row['val_acc']*100:.6f}", "val_macro_f1_pct": f"{row['val_macro_f1']*100:.6f}",
        "runtime_seconds_train_eval": f"{row['runtime_seconds_train_eval']:.3f}", "peak_gpu_memory_mib": row["peak_gpu_memory_mib"],
        "split_sha256": row["split_sha256"], "checkpoint_path": row["checkpoint_path"],
    } for row in rows]
    write_csv(REPORT_ROOT / "baseline_metrics.csv", csv_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("diagnose", "baselines"))
    args = parser.parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "diagnose":
        diagnose()
    else:
        run_baselines()


if __name__ == "__main__":
    main()
