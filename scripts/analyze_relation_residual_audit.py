from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_relation_operator_audit as old_audit  # noqa: E402
from scripts import analyze_relation_operator_diagnostics as old_diag  # noqa: E402
from scripts.run_relation_residual_audit import DATASETS, SEEDS, VARIANTS, OUTPUT  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models.relation_residual_audit import Model  # noqa: E402
from src.tasks.nc import _resolve_nc_eval_labels  # noqa: E402

RESULTS = ROOT / "results/relation_residual_audit_v1"
EPS = 1e-6
INTERVENTION_FIELDS = [
    "variant", "dataset", "seed", "intervention", "val_acc", "val_macro_f1",
    "mean_val_ce", "delta_val_acc", "delta_val_macro_f1", "mean_delta_ce",
    "median_delta_ce", "p10_delta_ce", "p90_delta_ce", "harmed_fraction",
    "improved_fraction", "near_zero_fraction", "mean_margin_utility",
    "prediction_flip_rate", "test_evaluated",
]
NODE_FIELDS = [
    "variant", "dataset", "seed", "intervention", "node_id", "label", "normal_ce",
    "intervention_ce", "delta_ce", "normal_margin", "intervention_margin",
    "margin_utility", "normal_prediction", "intervention_prediction", "prediction_flip",
]
SHRINK_FIELDS = [
    "dataset", "seed", "modality_scope", "lambda", "val_acc", "val_macro_f1",
    "mean_val_ce", "mean_delta_ce_vs_lambda1", "harmed_fraction", "improved_fraction",
    "near_zero_fraction", "prediction_flip_rate", "best_lambda_by_mean_ce",
    "best_lambda_by_val_accuracy", "test_evaluated",
]
COMPARISON_FIELDS = [
    "comparison", "comparison_scope", "dataset", "seed", "left_variant", "right_variant",
    "delta_val_acc", "delta_val_macro_f1", "delta_mean_val_ce", "test_evaluated",
]
CONTEXT_FIELDS = [
    "dataset", "seed", "modality", "semantic_similarity_quartile",
    "structural_response_magnitude_quartile", "n_edges", "mean_route_l1_change",
    "median_route_l1_change", "p90_route_l1_change",
]
PARAM_FIELDS = [
    "variant", "dataset", "seed", "model_trainable_params", "head_trainable_params",
    "total_trainable_params",
]
NORMAL_FIELDS = [
    "variant", "dataset", "seed", "best_epoch", "val_acc", "val_macro_f1",
    "mean_val_ce", "runtime_seconds", "peak_gpu_memory_mib", "test_evaluated",
]
STAT_FIELDS = ["phase", "variant", "dataset", "seed", "statistic", "value"]
TRAIN_STAT_FIELDS = ["variant", "dataset", "seed", "epoch", "statistic", "value"]
ACCEPTANCE_FIELDS = [
    "hypothesis", "scientific_question", "status", "contexts_analyzed",
    "contexts_expected", "evidence_summary", "adjudication_note",
]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _run_paths(mode: str, variant: str, dataset: str, seed: int) -> tuple[Path, Path]:
    if mode == "formal":
        run_dir = OUTPUT / "runs" / variant / dataset / f"seed{seed}"
        checkpoint_path = OUTPUT / "checkpoints" / variant / dataset / f"seed{seed}.pt"
    else:
        run_dir = OUTPUT / "smoke" / variant / dataset / f"seed{seed}"
        checkpoint_path = OUTPUT / "smoke/checkpoints" / variant / dataset / f"seed{seed}.pt"
    return run_dir, checkpoint_path


def _load_context(mode: str, variant: str, dataset: str, seed: int, device: torch.device):
    run_dir, checkpoint_path = _run_paths(mode, variant, dataset, seed)
    config_path, metrics_path = run_dir / "resolved_config.json", run_dir / "metrics.json"
    marker = run_dir / "complete.marker"
    if not (config_path.is_file() and metrics_path.is_file() and marker.is_file() and checkpoint_path.is_file()):
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config.get("task", {}).get("evaluate_test") is not False:
        raise RuntimeError(f"Test evaluation must be disabled in {config_path}")
    def contains_test_metric_key(value) -> bool:
        if isinstance(value, dict):
            return any("test" in str(key).lower() or contains_test_metric_key(child)
                       for key, child in value.items())
        if isinstance(value, (list, tuple)):
            return any(contains_test_metric_key(child) for child in value)
        return False

    if contains_test_metric_key(metrics_payload.get("metrics", {})):
        raise RuntimeError(f"Test metric key found in {metrics_path}")
    if contains_test_metric_key(payload.get("metrics", {})):
        raise RuntimeError(f"Test metric key found in {checkpoint_path}")
    if (payload.get("task") != "nc" or payload.get("seed") != seed
            or payload.get("selection") != "best_val_accuracy"):
        raise RuntimeError(f"checkpoint metadata mismatch in {checkpoint_path}")
    if config.get("model", {}).get("residual_variant") != variant:
        raise RuntimeError(f"checkpoint variant mismatch in {config_path}")
    cfg = OmegaConf.create(config)
    data = load_mag_data(cfg, "nc", seed)
    model = Model(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    head = torch.nn.Linear(model.out_dim, int(payload["data_info"]["num_classes"])).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    head.eval()
    eval_labels = _resolve_nc_eval_labels(data, include_test=False)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    return {
        "cfg": cfg, "data": data, "model": model, "head": head, "payload": payload,
        "eval_labels": eval_labels, "x": x, "edge_index": edge_index,
        "metrics_payload": metrics_payload, "run_dir": run_dir,
    }


def _evaluate(head, z: torch.Tensor, data, eval_labels: list[int], device: torch.device) -> dict:
    val_idx = data.val_idx.to(device)
    labels = data.y[data.val_idx].to(device)
    logits = head(z[val_idx])
    ce = F.cross_entropy(logits, labels, reduction="none")
    pred = logits.argmax(dim=-1)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    margin = true_logits - other.max(dim=-1).values
    return {
        "logits": logits, "labels": labels, "ce": ce, "predictions": pred,
        "margin": margin,
        "acc": float((pred == labels).float().mean().item()),
        "macro_f1": float(f1_score(
            labels.detach().cpu().numpy(), pred.detach().cpu().numpy(),
            labels=eval_labels, average="macro", zero_division=0,
        )),
        "mean_ce": float(ce.mean().item()),
        "val_idx": val_idx,
    }


def _utility_summary(normal: dict, changed: dict) -> dict:
    delta = changed["ce"] - normal["ce"]
    margin_utility = normal["margin"] - changed["margin"]
    q = torch.quantile(delta, delta.new_tensor([0.1, 0.5, 0.9]))
    return {
        "mean_delta_ce": float(delta.mean().item()),
        "median_delta_ce": float(q[1].item()),
        "p10_delta_ce": float(q[0].item()),
        "p90_delta_ce": float(q[2].item()),
        "harmed_fraction": float((delta > EPS).float().mean().item()),
        "improved_fraction": float((delta < -EPS).float().mean().item()),
        "near_zero_fraction": float((delta.abs() <= EPS).float().mean().item()),
        "mean_margin_utility": float(margin_utility.mean().item()),
        "prediction_flip_rate": float((normal["predictions"] != changed["predictions"]).float().mean().item()),
    }


def _write_node_rows(writer, variant: str, dataset: str, seed: int, intervention: str,
                     normal: dict, changed: dict) -> None:
    delta = changed["ce"] - normal["ce"]
    margin_utility = normal["margin"] - changed["margin"]
    labels = normal["labels"].detach().cpu().tolist()
    node_ids = normal["val_idx"].detach().cpu().tolist()
    normal_ce, changed_ce = normal["ce"].detach().cpu().tolist(), changed["ce"].detach().cpu().tolist()
    normal_margin = normal["margin"].detach().cpu().tolist()
    changed_margin = changed["margin"].detach().cpu().tolist()
    delta_values, margin_values = delta.detach().cpu().tolist(), margin_utility.detach().cpu().tolist()
    normal_pred = normal["predictions"].detach().cpu().tolist()
    changed_pred = changed["predictions"].detach().cpu().tolist()
    for i, node_id in enumerate(node_ids):
        writer.writerow({
            "variant": variant, "dataset": dataset, "seed": seed,
            "intervention": intervention, "node_id": node_id, "label": labels[i],
            "normal_ce": normal_ce[i], "intervention_ce": changed_ce[i],
            "delta_ce": delta_values[i], "normal_margin": normal_margin[i],
            "intervention_margin": changed_margin[i], "margin_utility": margin_values[i],
            "normal_prediction": normal_pred[i], "intervention_prediction": changed_pred[i],
            "prediction_flip": int(normal_pred[i] != changed_pred[i]),
        })


def _intervention_specs(variant: str, seed: int) -> list[tuple[str, dict]]:
    specs = [("zero_shared", {"shared_enabled": False})]
    if variant != "shared_prior":
        specs.append(("zero_residual", {"residual_enabled": False}))
    if variant == "context_residual":
        specs.extend([
            ("context_null", {"context_mode": "null"}),
            ("context_shuffle", {"context_mode": "shuffle", "context_seed": seed * 1009 + 73}),
        ])
    if variant in {"attribute_residual", "context_residual"}:
        specs.append(("global_route", {"route_mode": "global"}))
    return specs


def _context_sensitivity_rows(
    dataset: str, seed: int, model: Model, x: torch.Tensor, edge_index: torch.Tensor,
    data, device: torch.device,
) -> list[dict]:
    real = model.route_analysis(x, edge_index, context_mode="real")
    null = model.route_analysis(x, edge_index, context_mode="null")
    target, source = real["target"], real["source"]
    rows = []
    train_nodes = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_nodes[data.train_idx.to(device)] = True
    train_edges = train_nodes[target] & train_nodes[source]
    # Use frozen plain A0 H0 and P0-D's Train-Train similarity quartiles.
    _, a0_data, a0_model, _, _, _, a0_x, a0_edges = old_audit._load_checkpoint_context(
        "plain", dataset, seed, device
    )
    a0 = a0_model.analyze(a0_x, a0_edges)
    a0_target, a0_source, _, _, _, _ = a0_model._operator_edges(
        a0_model._get_operator(a0_edges, a0_x.size(0), a0_x.dtype)
    )
    if not torch.equal(target, a0_target) or not torch.equal(source, a0_source):
        raise RuntimeError(f"P0-R and frozen A0 physical edge order differs for {dataset}/seed{seed}")
    for modality in ("text", "visual"):
        semantic = old_diag._make_semantic_bins(
            a0_model, a0[f"H0_{modality}"], a0_data, target, source, modality
        )
        d = real[f"D_{modality}"]
        structural_magnitude = (d[target] - d[source]).norm(dim=-1)
        structural_thresholds, _ = old_diag._quantile_bins(structural_magnitude, train_edges)
        semantic_bins_all = torch.bucketize(semantic["similarity"], semantic["thresholds"])
        structural_bins = torch.bucketize(structural_magnitude, structural_thresholds)
        shift = (real[f"routes_{modality}"] - null[f"routes_{modality}"]).abs().sum(dim=-1)
        for semantic_q in range(4):
            for structural_q in range(4):
                selected = (semantic_bins_all == semantic_q) & (structural_bins == structural_q)
                values = shift[selected]
                if not values.numel():
                    continue
                q = torch.quantile(values, values.new_tensor([0.5, 0.9]))
                rows.append({
                    "dataset": dataset, "seed": seed, "modality": modality,
                    "semantic_similarity_quartile": f"Q{semantic_q + 1}",
                    "structural_response_magnitude_quartile": f"S{structural_q + 1}",
                    "n_edges": int(values.numel()), "mean_route_l1_change": float(values.mean().item()),
                    "median_route_l1_change": float(q[0].item()),
                    "p90_route_l1_change": float(q[1].item()),
                })
    del a0_model, a0_data, a0_x, a0_edges, a0
    return rows


def _normal_row(variant: str, dataset: str, seed: int, values: dict, loaded: dict) -> dict:
    payload = loaded["metrics_payload"]
    return {
        "variant": variant, "dataset": dataset, "seed": seed,
        "best_epoch": loaded["payload"].get("epoch"),
        "val_acc": values["acc"], "val_macro_f1": values["macro_f1"],
        "mean_val_ce": values["mean_ce"],
        "runtime_seconds": payload.get("runtime_seconds"),
        "peak_gpu_memory_mib": payload.get("peak_gpu_memory_mib"),
        "test_evaluated": False,
    }


def _training_mechanism_stats(run_dir: Path) -> list[tuple[int, dict[str, float]]]:
    log_path = run_dir / "main.log"
    if not log_path.is_file():
        return []
    import re
    epoch = None
    records = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = re.search(r"Epoch\s+(\d+)", line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
        if "Aux rr_" not in line or epoch is None:
            continue
        values = {
            key: float(value) for key, value in re.findall(
                r"(rr_[A-Za-z0-9_]+) (-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)",
                line,
            )
        }
        if values:
            records.append((epoch, values))
    return records


def _append_comparisons(normal_metrics: list[dict]) -> list[dict]:
    by_key = {(r["variant"], r["dataset"], int(r["seed"])): r for r in normal_metrics}
    rows = []
    within = [("R0G-R0", "global_residual", "shared_prior"),
              ("R1-R0G", "attribute_residual", "global_residual"),
              ("R2-R1", "context_residual", "attribute_residual"),
              ("R2-R0", "context_residual", "shared_prior")]
    for dataset in DATASETS:
        for seed in SEEDS:
            for name, left_variant, right_variant in within:
                left = by_key.get((left_variant, dataset, seed))
                right = by_key.get((right_variant, dataset, seed))
                if left is None or right is None:
                    continue
                rows.append({
                    "comparison": name, "comparison_scope": "within_p0r",
                    "dataset": dataset, "seed": seed, "left_variant": left_variant,
                    "right_variant": right_variant,
                    "delta_val_acc": left["val_acc"] - right["val_acc"],
                    "delta_val_macro_f1": left["val_macro_f1"] - right["val_macro_f1"],
                    "delta_mean_val_ce": left["mean_val_ce"] - right["mean_val_ce"],
                    "test_evaluated": False,
                })
    old_rows = list(csv.DictReader((ROOT / "results/relation_operator_audit_v1/validation_results.csv").open()))
    old_map = {(r["variant"], r["dataset"], int(r["seed"])): r for r in old_rows}
    cross = [("R0-old_plain", "shared_prior", "plain"),
             ("R0-old_global_expert", "shared_prior", "global_expert"),
             ("R2-old_relation_expert", "context_residual", "relation_expert"),
             ("R2-old_global_expert", "context_residual", "global_expert")]
    for (variant, dataset, seed), current in by_key.items():
        for name, left_variant, old_variant in cross:
            if variant != left_variant:
                continue
            old = old_map.get((old_variant, dataset, seed))
            if old is None:
                continue
            rows.append({
                "comparison": name,
                "comparison_scope": "cross-experiment paired descriptive comparison",
                "dataset": dataset, "seed": seed, "left_variant": left_variant,
                "right_variant": f"old_{old_variant}",
                "delta_val_acc": current["val_acc"] - float(old["val_acc"]),
                "delta_val_macro_f1": current["val_macro_f1"] - float(old["val_macro_f1"]),
                "delta_mean_val_ce": None, "test_evaluated": False,
            })
    return rows


def _aggregate_shrinkage(rows: list[dict]) -> list[dict]:
    aggregates = []
    for scope in ("text_only", "visual_only", "both"):
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            subset = [r for r in rows if r["modality_scope"] == scope and float(r["lambda"]) == lam]
            if not subset:
                continue
            aggregates.append({
                "dataset": "ALL", "seed": "ALL", "modality_scope": scope, "lambda": lam,
                **{key: float(np.mean([float(r[key]) for r in subset])) for key in (
                    "val_acc", "val_macro_f1", "mean_val_ce", "mean_delta_ce_vs_lambda1",
                    "harmed_fraction", "improved_fraction", "near_zero_fraction",
                    "prediction_flip_rate",
                )},
                "best_lambda_by_mean_ce": "", "best_lambda_by_val_accuracy": "",
                "test_evaluated": False,
            })
    for scope in ("text_only", "visual_only", "both"):
        subset = [r for r in aggregates if r["modality_scope"] == scope]
        if not subset:
            continue
        best_ce = min(subset, key=lambda r: r["mean_val_ce"])["lambda"]
        best_acc = max(subset, key=lambda r: r["val_acc"])["lambda"]
        for row in subset:
            row["best_lambda_by_mean_ce"] = best_ce
            row["best_lambda_by_val_accuracy"] = best_acc
    return aggregates



def _acceptance_rows(mode: str, normal_rows: list[dict], comparisons: list[dict],
                    intervention_rows: list[dict], shrink_rows: list[dict],
                    context_rows: list[dict], mechanism_rows: list[dict]) -> list[dict]:
    expected_keys = {(variant, dataset, seed) for variant in VARIANTS
                     for dataset in DATASETS for seed in SEEDS}
    observed_keys = {(row["variant"], row["dataset"], int(row["seed"]))
                     for row in normal_rows}
    formal_complete = mode == "formal" and observed_keys == expected_keys

    def mean(rows, field):
        values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
        return float(np.mean(values)) if values else float("nan")

    def fmt(value):
        return "NA" if not np.isfinite(value) else f"{value:+.4f}"

    def paired(name):
        return [row for row in comparisons if row["comparison"] == name]

    def interventions(name, variant=None):
        return [row for row in intervention_rows if row["intervention"] == name
                and (variant is None or row["variant"] == variant)]

    r0_zero = interventions("zero_shared", "shared_prior")
    r0_old_plain = paired("R0-old_plain")
    r0_old_global = paired("R0-old_global_expert")
    r0_shared_mag = [row for row in mechanism_rows
                     if row["variant"] == "shared_prior" and "rho_shared_mean" in row["statistic"]]
    r1_pair = paired("R1-R0G")
    r1_zero = interventions("zero_residual", "attribute_residual")
    r1_global = interventions("global_route", "attribute_residual")
    r2_pair = paired("R2-R1")
    r2_null = interventions("context_null", "context_residual")
    r2_shuffle = interventions("context_shuffle", "context_residual")
    r2_route = [row for row in context_rows]
    r2_residual_mag = [row for row in mechanism_rows
                       if row["variant"] == "context_residual" and "rho_adapt_mean" in row["statistic"]]
    r2_shrink_raw = [row for row in shrink_rows if row["dataset"] != "ALL"]
    r2_scopes = {(row["dataset"], row["seed"], row["modality_scope"])
                 for row in r2_shrink_raw}
    r2_intermediate = 0
    for key in r2_scopes:
        rows = [row for row in r2_shrink_raw
                if (row["dataset"], row["seed"], row["modality_scope"]) == key]
        full = next((row for row in rows if float(row["lambda"]) == 1.0), None)
        intermediate = [row for row in rows if 0.0 < float(row["lambda"]) < 1.0]
        if full and intermediate and min(float(row["mean_val_ce"]) for row in intermediate) < float(full["mean_val_ce"]):
            r2_intermediate += 1
    r2_old_a3 = paired("R2-old_relation_expert")

    specs = [
        ("H-R0", "Can one independently trained shared operator replace the old global expert bank?",
         len([row for row in normal_rows if row["variant"] == "shared_prior"]), len(DATASETS) * len(SEEDS),
         (f"zero-shared: mean ΔCE={fmt(mean(r0_zero, 'mean_delta_ce'))}, Accuracy decreased "
          f"in {sum(row['delta_val_acc'] < 0 for row in r0_zero)}/{len(r0_zero)} contexts; "
          f"R0−old plain mean ΔAccuracy={fmt(mean(r0_old_plain, 'delta_val_acc'))}, "
          f"R0−old A2 mean ΔAccuracy={fmt(mean(r0_old_global, 'delta_val_acc'))}; "
          f"mean shared-branch rho={fmt(mean(r0_shared_mag, 'value'))}.")),
        ("H-R1", "Does attribute-conditioned residual specialization add value over a shared global residual route?",
         len(r1_pair), len(DATASETS) * len(SEEDS),
         (f"R1−R0G mean ΔAccuracy={fmt(mean(r1_pair, 'delta_val_acc'))}, "
          f"ΔMacro-F1={fmt(mean(r1_pair, 'delta_val_macro_f1'))}, ΔCE={fmt(mean(r1_pair, 'delta_mean_val_ce'))}; "
          f"zero-residual mean ΔCE={fmt(mean(r1_zero, 'mean_delta_ce'))}; "
          f"global-route mean ΔCE={fmt(mean(r1_global, 'mean_delta_ce'))}.")),
        ("H-R2", "Does local structural response improve relation interpretation beyond endpoint attributes?",
         len(r2_pair), len(DATASETS) * len(SEEDS),
         (f"R2−R1 mean ΔAccuracy={fmt(mean(r2_pair, 'delta_val_acc'))}, "
          f"ΔMacro-F1={fmt(mean(r2_pair, 'delta_val_macro_f1'))}, ΔCE={fmt(mean(r2_pair, 'delta_mean_val_ce'))}; "
          f"Context→NULL mean ΔCE={fmt(mean(r2_null, 'mean_delta_ce'))} "
          f"({sum(row['delta_val_acc'] < 0 for row in r2_null)}/{len(r2_null)} Accuracy declines); "
          f"shuffle mean ΔCE={fmt(mean(r2_shuffle, 'mean_delta_ce'))}; "
          f"mean real-vs-NULL route L1={fmt(mean(r2_route, 'mean_route_l1_change'))} across "
          f"{len(r2_route)} strata; mean adapter rho={fmt(mean(r2_residual_mag, 'value'))}.")),
        ("H-R3", "Does the shared prior reduce P0-D's over-personalization pattern?",
         len(r2_scopes), len(DATASETS) * len(SEEDS) * 3,
         (f"Intermediate λ beat λ=1 CE in {r2_intermediate}/{len(r2_scopes)} modality-scope contexts; "
          f"R2−old A3 mean ΔAccuracy={fmt(mean(r2_old_a3, 'delta_val_acc'))}, "
          f"ΔMacro-F1={fmt(mean(r2_old_a3, 'delta_val_macro_f1'))}. "
          "Historical A3 reference: 41/45 contexts had an intermediate-λ CE advantage.")),
    ]
    if mode == "smoke":
        status = "SMOKE_ONLY"
    elif formal_complete:
        status = "PENDING_ADJUDICATION"
    else:
        status = "PENDING_FORMAL_60_CONTEXTS"
    note = (
        "Descriptive evidence only. The protocol does not define numeric cutoffs for terms such as "
        "'markedly', 'noticeably', or 'systematically'; retain these summaries for scientific review "
        "instead of inventing pass/fail thresholds."
    )
    return [{
        "hypothesis": hypothesis, "scientific_question": question, "status": status,
        "contexts_analyzed": analyzed, "contexts_expected": expected,
        "evidence_summary": evidence, "adjudication_note": note,
    } for hypothesis, question, analyzed, expected, evidence in specs]


def _write_report(mode: str, normal_rows: list[dict], comparisons: list[dict],
                  intervention_rows: list[dict], shrink_rows: list[dict],
                  context_rows: list[dict], parameter_rows: list[dict],
                  mechanism_rows: list[dict], initial_mechanism_rows: list[dict]) -> None:
    expected = 60 if mode == "formal" else None
    complete = len(normal_rows)
    complete_formal = mode == "formal" and complete == expected
    lines = [
        "# P0-R — Shared-Prior Contextual Relation Adaptation", "",
        f"Analysis mode: `{mode}`; loaded P0-R checkpoints: {complete}.",
        "This analyzer uses only Train/Validation labels. Every resolved config and checkpoint is required to have `task.evaluate_test=false` and no Test metrics. No Test labels or metrics are used.",
        "All checkpoint comparisons use strict state loading. Within-P0-R paired comparisons share dataset and seed. Previous P0 comparisons are labeled `cross-experiment paired descriptive comparison`; their CE values are unavailable from the historical validation table.",
        "Structural context magnitude quartiles use the Train–Train distribution of `||D_i-D_j||`; semantic-similarity quartiles use frozen plain A0 H0 and the same Train–Train thresholds used by P0-D.",
        "", "## Formal status", "",
        ("All 60 formal contexts are available." if complete_formal else
         "Formal scientific decisions remain PENDING until all 60 formal contexts are analyzed. Smoke metrics are engineering checks only."),
        "", "## Main paired comparisons", "",
    ]
    for name in ("R0G-R0", "R1-R0G", "R2-R1", "R2-R0",
                 "R0-old_plain", "R0-old_global_expert", "R2-old_relation_expert", "R2-old_global_expert"):
        subset = [r for r in comparisons if r["comparison"] == name]
        if not subset:
            lines.append(f"- `{name}`: no matched analyzed contexts yet.")
            continue
        lines.append(
            f"- `{name}` ({len(subset)} contexts): mean ΔAccuracy={np.mean([r['delta_val_acc'] for r in subset]):+.4f}; "
            f"mean ΔMacro-F1={np.mean([r['delta_val_macro_f1'] for r in subset]):+.4f}; "
            + (f"mean ΔCE={np.mean([r['delta_mean_val_ce'] for r in subset if r['delta_mean_val_ce'] is not None]):+.4f}."
               if any(r["delta_mean_val_ce"] is not None for r in subset) else "CE not present in historical comparison table.")
        )
    lines += ["", "## Frozen interventions and mechanisms", ""]
    for variant in VARIANTS:
        subset = [r for r in intervention_rows if r["variant"] == variant and r["intervention"] == "zero_shared"]
        if subset:
            lines.append(
                f"- `{variant}` zero-shared: mean ΔCE {np.mean([r['mean_delta_ce'] for r in subset]):+.4f}; "
                f"Validation Accuracy declined in {sum(r['delta_val_acc'] < 0 for r in subset)}/{len(subset)} contexts."
            )
    residual_zero = [r for r in intervention_rows if r["intervention"] == "zero_residual"]
    if residual_zero:
        lines.append(
            f"- Zero-residual: mean ΔCE {np.mean([r['mean_delta_ce'] for r in residual_zero]):+.4f}; "
            f"Accuracy declined in {sum(r['delta_val_acc'] < 0 for r in residual_zero)}/{len(residual_zero)} contexts."
        )
    for intervention in ("context_null", "context_shuffle", "global_route"):
        subset = [r for r in intervention_rows if r["intervention"] == intervention]
        if subset:
            lines.append(
                f"- `{intervention}`: mean ΔCE {np.mean([r['mean_delta_ce'] for r in subset]):+.4f}; "
                f"Accuracy declined in {sum(r['delta_val_acc'] < 0 for r in subset)}/{len(subset)} contexts."
            )
    lines.append(f"- Context-sensitivity summaries: {len(context_rows)} semantic × structural strata.")
    lines += ["", "## R2 shrinkage", ""]
    raw_shrink = [r for r in shrink_rows if r["dataset"] != "ALL"]
    scopes = {(r["dataset"], r["seed"], r["modality_scope"]) for r in raw_shrink}
    advantage = 0
    lambda1_best = lambda075_best = 0
    for key in scopes:
        local = [r for r in raw_shrink if (r["dataset"], r["seed"], r["modality_scope"]) == key]
        full = next(r for r in local if float(r["lambda"]) == 1.0)
        advantage += int(min(float(r["mean_val_ce"]) for r in local if 0.0 < float(r["lambda"]) < 1.0) < full["mean_val_ce"])
        best = min(local, key=lambda r: r["mean_val_ce"])
        lambda1_best += int(float(best["lambda"]) == 1.0)
        lambda075_best += int(float(best["lambda"]) == 0.75)
    lines.append(
        f"R2 intermediate λ beat λ=1 CE in {advantage}/{len(scopes)} contexts; best CE λ was 1.0 in "
        f"{lambda1_best} contexts and 0.75 in {lambda075_best}. Historical A3 reported 41/45 contexts with an intermediate advantage."
    )
    lines += ["", "## Parameter parity and mechanism interpretation", ""]
    if parameter_rows:
        by_context = defaultdict(dict)
        for row in parameter_rows:
            by_context[(row["dataset"], row["seed"])][row["variant"]] = int(row["model_trainable_params"])
        parity = [all(group.get(v) == group.get("global_residual") for v in
                      ("attribute_residual", "context_residual"))
                  for group in by_context.values() if "global_residual" in group]
        lines.append(f"R0G/R1/R2 trainable model parameter parity holds in {sum(parity)}/{len(parity)} paired dataset-seed records.")
    if mode == "smoke" and initial_mechanism_rows:
        route_summary = []
        for phase, source_rows in (("epoch1", initial_mechanism_rows), ("post_smoke", mechanism_rows)):
            values = [float(row["value"]) for row in source_rows if row["statistic"].endswith("pi0_mean")]
            if values:
                route_summary.append(f"mean π0 {phase}={np.mean(values):.4f}")
        lines.append("No-adaptation initialization check: " + "; ".join(route_summary) + ". The first value is measured during the epoch-1 forward before its optimizer step; the post-smoke value uses the best Validation-Accuracy checkpoint.")
    lines += [
        "The training_mechanism_statistics.csv.gz file preserves rr_ mechanism summaries from each logged training epoch; mechanism_statistics.csv reports frozen best-checkpoint evaluation. Both include per-modality/per-hop shared and residual magnitudes, no-adaptation route, adapter usage, routing entropy/top-1 fractions, and adapter-output cosine. Context sensitivity compares real D routes with learned-NULL routes while keeping checkpoint parameters fixed.",
        "Acceptance summaries are written to `acceptance_decisions.csv`. The protocol gives no numeric cutoffs for qualitative terms such as 'markedly' or 'noticeably', so these rows remain descriptive and require scientific adjudication rather than an invented pass/fail threshold.",
        "No P0-R formal training is started by the analyzer. No auxiliary objectives, structural supervision, or test-set selection are used.",
    ]
    data_smoke_path = RESULTS / "data_forward_smoke.csv"
    if data_smoke_path.is_file():
        data_smoke = list(csv.DictReader(data_smoke_path.open("r", newline="", encoding="utf-8")))
        finite_count = sum(str(row.get("finite_output", "")).lower() == "true" for row in data_smoke)
        peaks = [float(row["peak_gpu_memory_mib"]) for row in data_smoke
                 if row.get("peak_gpu_memory_mib") not in (None, "")]
        lines += ["", "## Engineering smoke evidence", "",
                  f"No-label R2 forward: {finite_count}/{len(data_smoke)} dataset contexts had finite output; "
                  + (f"maximum measured allocation {max(peaks):.1f} MiB." if peaks else "peak memory not recorded.")]
    manifest_path = OUTPUT / "smoke_run_manifest.csv"
    if manifest_path.is_file():
        manifest = list(csv.DictReader(manifest_path.open("r", newline="", encoding="utf-8")))
        complete_runs = [row for row in manifest if row.get("status") in {"complete", "already_complete"}]
        runtimes = [float(row["runtime_seconds"]) for row in complete_runs
                    if row.get("runtime_seconds") not in (None, "")]
        peaks = [float(row["peak_gpu_memory_mib"]) for row in complete_runs
                 if row.get("peak_gpu_memory_mib") not in (None, "")]
        if complete_runs:
            lines.append(f"One-epoch CUDA training smoke: {len(complete_runs)} completed contexts; "
                         f"maximum runtime {max(runtimes):.2f} s and peak allocation "
                         f"{max(peaks):.1f} MiB across the recorded runs.")
            commits = sorted({row.get("git_commit", "") for row in complete_runs if row.get("git_commit")})
            clean_states = sorted({row.get("git_worktree_clean", "unknown") for row in complete_runs})
            lines.append(f"Smoke provenance: HEAD {', '.join(commits) or 'unknown'}; "
                         f"git_worktree_clean={', '.join(clean_states)}.")
    lines.append(
        "CUDA attention execution note: an initial 65,536-edge fused SDPA batch failed with an invalid kernel configuration. "
        "Relation attention now requests (and discards) MHA attention weights to select the general MHA path, "
        "with an internal edge batch cap of min(edge_chunk_size, 8,192). This changes chunk execution, not model equations."
    )
    (RESULTS / "relation_residual_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen P0-R NC mechanism and intervention analyzer")
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--variants", nargs="*", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    torch.use_deterministic_algorithms(True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    normal_rows, intervention_rows, shrink_rows, context_rows = [], [], [], []
    parameter_rows, mechanism_rows, smoke_initial_mechanism_rows = [], [], []
    training_mechanism_rows = []
    node_path = RESULTS / "intervention_node_utility.csv.gz"
    with gzip.open(node_path, "wt", newline="", encoding="utf-8", compresslevel=6) as node_handle:
        node_writer = csv.DictWriter(node_handle, fieldnames=NODE_FIELDS, lineterminator="\n")
        node_writer.writeheader()
        for dataset in args.datasets:
            for seed in args.seeds:
                for variant in args.variants:
                    loaded = _load_context(args.mode, variant, dataset, seed, device)
                    if loaded is None:
                        continue
                    model, head, data = loaded["model"], loaded["head"], loaded["data"]
                    with torch.no_grad():
                        normal_analysis = model.analyze(loaded["x"], loaded["edge_index"])
                        normal = _evaluate(head, normal_analysis["fused_z"], data, loaded["eval_labels"], device)
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    h_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
                    parameter_rows.append({
                        "variant": variant, "dataset": dataset, "seed": seed,
                        "model_trainable_params": n_params, "head_trainable_params": h_params,
                        "total_trainable_params": n_params + h_params,
                    })
                    normal_row = _normal_row(variant, dataset, seed, normal, loaded)
                    normal_rows.append(normal_row)
                    stats = normal_analysis["mechanism_stats"]
                    for key, value in stats.items():
                        mechanism_rows.append({
                            "phase": "best_checkpoint_eval", "variant": variant,
                            "dataset": dataset, "seed": seed, "statistic": key,
                            "value": float(value.detach().cpu().item()),
                        })
                    for epoch, epoch_stats in _training_mechanism_stats(loaded["run_dir"]):
                        for key, value in epoch_stats.items():
                            training_mechanism_rows.append({
                                "variant": variant, "dataset": dataset, "seed": seed,
                                "epoch": epoch, "statistic": key, "value": value,
                            })
                            if args.mode == "smoke" and epoch == 1:
                                smoke_initial_mechanism_rows.append({
                                    "phase": "epoch1_forward_pre_optimizer_step", "variant": variant,
                                    "dataset": dataset, "seed": seed, "statistic": key, "value": value,
                                })
                    for name, kwargs in _intervention_specs(variant, seed):
                        with torch.no_grad():
                            changed_analysis = model.analyze(
                                loaded["x"], loaded["edge_index"], **kwargs
                            )
                            changed = _evaluate(head, changed_analysis["fused_z"], data,
                                                loaded["eval_labels"], device)
                        summary = _utility_summary(normal, changed)
                        row = {
                            "variant": variant, "dataset": dataset, "seed": seed,
                            "intervention": name, "val_acc": changed["acc"],
                            "val_macro_f1": changed["macro_f1"], "mean_val_ce": changed["mean_ce"],
                            "delta_val_acc": changed["acc"] - normal["acc"],
                            "delta_val_macro_f1": changed["macro_f1"] - normal["macro_f1"],
                            **summary, "test_evaluated": False,
                        }
                        intervention_rows.append(row)
                        _write_node_rows(node_writer, variant, dataset, seed, name, normal, changed)
                    if variant == "context_residual":
                        context_rows.extend(_context_sensitivity_rows(
                            dataset, seed, model, loaded["x"], loaded["edge_index"], data, device
                        ))
                        for scope in ("text_only", "visual_only", "both"):
                            for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
                                kwargs = {
                                    "shrinkage_text": lam if scope in {"text_only", "both"} else 1.0,
                                    "shrinkage_visual": lam if scope in {"visual_only", "both"} else 1.0,
                                }
                                with torch.no_grad():
                                    analysis = model.analyze(
                                        loaded["x"], loaded["edge_index"], **kwargs
                                    )
                                    changed = _evaluate(head, analysis["fused_z"], data,
                                                        loaded["eval_labels"], device)
                                summary = _utility_summary(normal, changed)
                                shrink_rows.append({
                                    "dataset": dataset, "seed": seed, "modality_scope": scope,
                                    "lambda": lam, "val_acc": changed["acc"],
                                    "val_macro_f1": changed["macro_f1"], "mean_val_ce": changed["mean_ce"],
                                    "mean_delta_ce_vs_lambda1": summary["mean_delta_ce"],
                                    "harmed_fraction": summary["harmed_fraction"],
                                    "improved_fraction": summary["improved_fraction"],
                                    "near_zero_fraction": summary["near_zero_fraction"],
                                    "prediction_flip_rate": summary["prediction_flip_rate"],
                                    "best_lambda_by_mean_ce": "", "best_lambda_by_val_accuracy": "",
                                    "test_evaluated": False,
                                })
                        for scope in ("text_only", "visual_only", "both"):
                            local = [r for r in shrink_rows if r["dataset"] == dataset and r["seed"] == seed
                                     and r["modality_scope"] == scope]
                            best_ce = min(local, key=lambda r: r["mean_val_ce"])["lambda"]
                            best_acc = max(local, key=lambda r: r["val_acc"])["lambda"]
                            for row in local:
                                row["best_lambda_by_mean_ce"] = best_ce
                                row["best_lambda_by_val_accuracy"] = best_acc
                    del loaded, model, head, data, normal_analysis
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    if not normal_rows:
        raise RuntimeError(f"No complete {args.mode} P0-R checkpoint contexts found")
    comparisons = _append_comparisons(normal_rows)
    raw_shrink = list(shrink_rows)
    shrink_rows.extend(_aggregate_shrinkage(raw_shrink))
    acceptance_rows = _acceptance_rows(
        args.mode, normal_rows, comparisons, intervention_rows, shrink_rows,
        context_rows, mechanism_rows,
    )
    _write_csv(RESULTS / "acceptance_decisions.csv", acceptance_rows, ACCEPTANCE_FIELDS)
    _write_csv(RESULTS / "normal_validation_results.csv", normal_rows, NORMAL_FIELDS)
    _write_csv(RESULTS / "intervention_results.csv", intervention_rows, INTERVENTION_FIELDS)
    _write_csv(RESULTS / "r2_shrinkage_sweep.csv", shrink_rows, SHRINK_FIELDS)
    _write_csv(RESULTS / "paired_comparisons.csv", comparisons, COMPARISON_FIELDS)
    _write_csv(RESULTS / "context_sensitivity.csv", context_rows, CONTEXT_FIELDS)
    parameter_path = RESULTS / "parameter_counts.csv"
    if args.mode == "smoke" and parameter_path.is_file():
        existing = list(csv.DictReader(parameter_path.open("r", newline="", encoding="utf-8")))
        existing_all = [row for row in existing if str(row.get("seed")) == "ALL"]
        by_key = {(row["variant"], row["dataset"], str(row["seed"])): row
                  for row in existing_all}
        for row in parameter_rows:
            by_key[(row["variant"], row["dataset"], str(row["seed"]))] = row
        parameter_rows = list(by_key.values())
    _write_csv(parameter_path, parameter_rows, PARAM_FIELDS)
    _write_csv(RESULTS / "mechanism_statistics.csv", mechanism_rows, STAT_FIELDS)
    training_stat_path = RESULTS / "training_mechanism_statistics.csv.gz"
    with gzip.open(training_stat_path, "wt", newline="", encoding="utf-8", compresslevel=6) as training_handle:
        training_writer = csv.DictWriter(
            training_handle, fieldnames=TRAIN_STAT_FIELDS, lineterminator="\n"
        )
        training_writer.writeheader()
        training_writer.writerows(training_mechanism_rows)
    if args.mode == "smoke":
        _write_csv(RESULTS / "smoke_initial_mechanism_statistics.csv",
                   smoke_initial_mechanism_rows, STAT_FIELDS)
    _write_report(args.mode, normal_rows, comparisons, intervention_rows, shrink_rows,
                  context_rows, parameter_rows, mechanism_rows, smoke_initial_mechanism_rows)
    print(f"Analyzed {len(normal_rows)} P0-R checkpoints in {args.mode} mode: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
