#!/usr/bin/env python3
"""Checkpoint-only robustness evaluation for prepared perturbation plans.

Formal execution is deliberately opt-in via ``--execute``. This program never
creates an optimizer and never calls backward/train. It first verifies clean
inference for every participating checkpoint and only then evaluates nonzero
perturbations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_robustness_perturbations import (  # noqa: E402
    audit_clean_graph,
    edge_index_from_canonical,
    sha256_edges,
    sha256_file,
)
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from src.tasks.inference import infer_all_embeddings  # noqa: E402
from src.tasks.nc import _evaluate_split, _resolve_nc_eval_labels  # noqa: E402


CLEAN_ATOL = 1e-6
EVAL_BATCH_SIZE = 4096


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_injected_edges(output_root: Path, row: dict[str, str]) -> np.ndarray:
    path = (output_root / row["edge_artifact"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if row.get("edge_artifact_sha256") and sha256_file(path) != row["edge_artifact_sha256"]:
        raise ValueError(f"perturbation artifact SHA256 mismatch: {path}")
    if path.suffix == ".npy":
        edges = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as archive:
            edges = archive["edges"]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if row.get("injected_edge_count") and len(edges) != int(row["injected_edge_count"]):
        raise ValueError(f"injected-edge count mismatch for {path}")
    if row.get("edge_sha256") and sha256_edges(edges) != row["edge_sha256"]:
        raise ValueError(f"canonical injected-edge SHA256 mismatch for {path}")
    return edges


def _load_checkpoint_context(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[Any, nn.Module, nn.Module, list[int], dict[str, Any]]:
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    data = load_mag_data(cfg, "nc", int(cfg.seed))
    if data.edge_index is None or data.y is None or data.test_idx is None:
        raise ValueError(f"incomplete NC data for {checkpoint_path}")
    audit_clean_graph(data.edge_index, int(data.num_nodes))
    data_info = {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()
    eval_labels = _resolve_nc_eval_labels(data)
    state = {
        "cfg": cfg,
        "uses_graph": str(cfg.model.name).lower() != "mlp",
        "inference_mode": str(cfg.task.get("inference_mode", "full")),
        "model_seed": int(cfg.seed),
    }
    return data, model, classifier, eval_labels, state


@torch.no_grad()
def _run_metrics(
    data: Any,
    model: nn.Module,
    classifier: nn.Module,
    eval_labels: list[int],
    state: dict[str, Any],
    device: torch.device,
    edge_index: torch.Tensor,
) -> dict[str, float]:
    inference_data = replace(data, edge_index=edge_index)
    z = infer_all_embeddings(
        model,
        inference_data,
        device,
        bool(state["uses_graph"]),
        EVAL_BATCH_SIZE,
        str(state["inference_mode"]),
    )
    metrics = _evaluate_split(
        classifier,
        z,
        data.y,
        data.test_idx,
        device,
        EVAL_BATCH_SIZE,
        eval_labels,
    )
    return {"accuracy": float(metrics["acc"]), "macro_f1": float(metrics["macro_f1"])}


def _metric_matches(observed: float, expected: float) -> bool:
    return bool(np.isclose(float(observed), float(expected), rtol=0.0, atol=CLEAN_ATOL))


def clean_checkpoint_check(
    checkpoint_path: Path,
    config_path: Path,
    expected_accuracy: float,
    expected_macro_f1: float,
    device: torch.device,
) -> dict[str, float]:
    """One clean inference used as the gate before any robustness perturbation."""
    data, model, classifier, eval_labels, state = _load_checkpoint_context(
        checkpoint_path, config_path, device
    )
    measured = _run_metrics(
        data, model, classifier, eval_labels, state, device, data.edge_index
    )
    if not _metric_matches(measured["accuracy"], expected_accuracy):
        raise RuntimeError(
            f"clean accuracy mismatch at {checkpoint_path}: "
            f"measured={measured['accuracy']:.10f}, expected={expected_accuracy:.10f}, atol={CLEAN_ATOL}"
        )
    if not _metric_matches(measured["macro_f1"], expected_macro_f1):
        raise RuntimeError(
            f"clean Macro-F1 mismatch at {checkpoint_path}: "
            f"measured={measured['macro_f1']:.10f}, expected={expected_macro_f1:.10f}, atol={CLEAN_ATOL}"
        )
    return measured


def _audit_row(audit_rows: list[dict[str, str]], checkpoint_path: str) -> dict[str, str]:
    matches = [row for row in audit_rows if str(Path(row["checkpoint_path"]).resolve()) == str(Path(checkpoint_path).resolve())]
    if len(matches) != 1:
        raise ValueError(f"checkpoint has {len(matches)} audit rows: {checkpoint_path}")
    return matches[0]


def _load_clean_baselines(
    rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    checkpoint_paths = sorted({str(Path(row["checkpoint_path"]).resolve()) for row in rows})
    baselines: dict[str, dict[str, float]] = {}
    for checkpoint_path in checkpoint_paths:
        audit = _audit_row(audit_rows, checkpoint_path)
        baselines[checkpoint_path] = clean_checkpoint_check(
            Path(checkpoint_path),
            Path(audit["config_path"]),
            float(audit["clean_test_accuracy"]),
            float(audit["clean_test_macro_f1"]),
            device,
        )
        print(
            f"[clean PASS] {audit['dataset']}/{audit['model']}/seed{audit['model_seed']} "
            f"Acc={baselines[checkpoint_path]['accuracy']:.8f} "
            f"Macro-F1={baselines[checkpoint_path]['macro_f1']:.8f}",
            flush=True,
        )
    return baselines


def _evaluate_nonzero_rows(
    rows: list[dict[str, str]],
    output_root: Path,
    audit_rows: list[dict[str, str]],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows_by_checkpoint: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if float(row["rho"]) > 0.0:
            rows_by_checkpoint[str(Path(row["checkpoint_path"]).resolve())].append(row)

    output: list[dict[str, Any]] = []
    for checkpoint_path in sorted(rows_by_checkpoint):
        audit = _audit_row(audit_rows, checkpoint_path)
        data, model, classifier, eval_labels, state = _load_checkpoint_context(
            Path(checkpoint_path), Path(audit["config_path"]), device
        )
        clean_physical = audit_clean_graph(data.edge_index, int(data.num_nodes))
        clean_keys = set(map(tuple, clean_physical.tolist()))
        for row in sorted(
            rows_by_checkpoint[checkpoint_path],
            key=lambda item: (
                item["perturbation_type"],
                int(item["perturbation_seed"]),
                float(item["rho"]),
            ),
        ):
            injected = _load_injected_edges(output_root, row)
            if len(injected) and (
                np.any(injected[:, 0] >= injected[:, 1])
                or len(np.unique(injected, axis=0)) != len(injected)
                or any(tuple(pair) in clean_keys for pair in injected.tolist())
            ):
                raise ValueError(f"invalid injected edge list: {row['edge_artifact']}")
            perturbed_edge_index = edge_index_from_canonical(data.edge_index, injected)
            # Assert that the loader's clean edge columns remain an exact prefix.
            if not torch.equal(perturbed_edge_index[:, : data.edge_index.size(1)], data.edge_index):
                raise AssertionError("perturbation changed clean edge_index ordering/content")
            measured = _run_metrics(
                data,
                model,
                classifier,
                eval_labels,
                state,
                device,
                perturbed_edge_index,
            )
            baseline = {
                "accuracy": float(row["expected_clean_accuracy"]),
                "macro_f1": float(row["expected_clean_macro_f1"]),
            }
            output.append(_result_row(row, measured, baseline, cached_clean=False))
            print(
                f"[eval] {row['dataset']}/{row['model']}/seed{row['model_seed']} "
                f"{row['perturbation_type']} pseed={row['perturbation_seed']} rho={row['rho']} "
                f"Acc={measured['accuracy']:.6f} F1={measured['macro_f1']:.6f}",
                flush=True,
            )
        del data, model, classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def _result_row(
    plan: dict[str, str],
    measured: dict[str, float],
    baseline: dict[str, float],
    cached_clean: bool,
) -> dict[str, Any]:
    return {
        **plan,
        "test_accuracy": measured["accuracy"],
        "test_macro_f1": measured["macro_f1"],
        "clean_accuracy": baseline["accuracy"],
        "clean_macro_f1": baseline["macro_f1"],
        "accuracy_retention_pct": 100.0 * measured["accuracy"] / baseline["accuracy"],
        "macro_f1_retention_pct": 100.0 * measured["macro_f1"] / baseline["macro_f1"],
        "clean_row_reused": cached_clean,
    }


def _aggregate(raw_rows: list[dict[str, Any]], output_root: Path) -> None:
    """Aggregate perturbation replicates before model-seed mean/population SD."""
    first: dict[tuple[str, str, str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            row["dataset"],
            row["perturbation_type"],
            row["model"],
            int(row["model_seed"]),
            float(row["rho"]),
        )
        first[key].append(row)
    model_seed_rows: list[dict[str, Any]] = []
    for key, values in first.items():
        dataset, perturbation_type, model, model_seed, rho = key
        for metric, retention_key in (
            ("accuracy", "accuracy_retention_pct"),
            ("macro_f1", "macro_f1_retention_pct"),
        ):
            model_seed_rows.append(
                {
                    "dataset": dataset,
                    "perturbation_type": perturbation_type,
                    "model": model,
                    "model_seed": model_seed,
                    "rho": rho,
                    "metric": metric,
                    "mean_over_perturbation_seeds": float(np.mean([v[retention_key] for v in values])),
                    "num_perturbation_seeds": len(values),
                }
            )
    summary_groups: dict[tuple[str, str, str, float, str], list[float]] = defaultdict(list)
    for row in model_seed_rows:
        summary_groups[(row["dataset"], row["perturbation_type"], row["model"], row["rho"], row["metric"])].append(
            float(row["mean_over_perturbation_seeds"])
        )
    summary_rows = []
    for key, values in sorted(summary_groups.items()):
        dataset, perturbation_type, model, rho, metric = key
        summary_rows.append(
            {
                "dataset": dataset,
                "perturbation_type": perturbation_type,
                "model": model,
                "rho": rho,
                "metric": metric,
                "mean_across_model_seeds": float(np.mean(values)),
                "population_sd_across_model_seeds": float(np.std(values, ddof=0)),
                "num_model_seeds": len(values),
            }
        )
    _write_rows(output_root / "robustness_model_seed_means.csv", model_seed_rows)
    _write_rows(output_root / "robustness_summary.csv", summary_rows)
    summary_lookup = {
        (r["dataset"], r["perturbation_type"], r["model"], r["rho"], r["metric"]): r
        for r in summary_rows
    }
    gap_rows = []
    for dataset, perturbation_type in sorted({(r["dataset"], r["perturbation_type"]) for r in summary_rows}):
        rhos = sorted({r["rho"] for r in summary_rows if r["dataset"] == dataset and r["perturbation_type"] == perturbation_type})
        for rho in rhos:
            for metric in ("accuracy", "macro_f1"):
                full = summary_lookup[(dataset, perturbation_type, "mopf", rho, metric)]["mean_across_model_seeds"]
                for ablation, name in (("wo_relation_calibration", "RelationCalibrationGap"), ("wo_semantic_anchor", "AnchorGap")):
                    if perturbation_type == "semantic_conflict_edge_injection" and ablation == "wo_semantic_anchor":
                        continue
                    key = (dataset, perturbation_type, ablation, rho, metric)
                    if key not in summary_lookup:
                        continue
                    value = summary_lookup[key]["mean_across_model_seeds"]
                    gap_rows.append(
                        {
                            "dataset": dataset,
                            "perturbation_type": perturbation_type,
                            "rho": rho,
                            "metric": metric,
                            "gap": name,
                            "full_minus_ablation_retention_pp": full - value,
                        }
                    )
    _write_rows(output_root / "mechanism_gaps.csv", gap_rows)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/robustness_analysis")
    parser.add_argument("--perturbation-type", choices=("all", "random_noise", "semantic_conflict"), default="all")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--execute", action="store_true", help="required to run the formal checkpoint-only matrix")
    parser.add_argument("--check-clean", nargs=3, metavar=("DATASET", "MODEL", "SEED"), help="evaluate one checkpoint on the clean graph only")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    audit_path = output_root / "checkpoint_audit.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(f"run prepare_robustness_perturbations.py first: {audit_path}")
    audit_rows = _read_csv(audit_path)
    device = torch.device(args.device)

    if args.check_clean:
        dataset, model, seed_text = args.check_clean
        seed = int(seed_text)
        matches = [r for r in audit_rows if r["dataset"] == dataset and r["model"] == model and int(r["model_seed"]) == seed]
        if len(matches) != 1:
            raise ValueError(f"expected one checkpoint audit row, found {len(matches)}")
        row = matches[0]
        measured = clean_checkpoint_check(
            Path(row["checkpoint_path"]),
            Path(row["config_path"]),
            float(row["clean_test_accuracy"]),
            float(row["clean_test_macro_f1"]),
            device,
        )
        print(json.dumps({"status": "PASS", "dataset": dataset, "model": model, "seed": seed, **measured}, indent=2))
        return

    if not args.execute:
        raise SystemExit("Refusing inference without --execute; use --check-clean for one clean checkpoint.")

    plan_paths = []
    if args.perturbation_type in ("all", "random_noise"):
        plan_paths.append(output_root / "random_noise/dry_run_plan.csv")
    if args.perturbation_type in ("all", "semantic_conflict"):
        plan_paths.append(output_root / "semantic_conflict/dry_run_plan.csv")
    plan_rows = [row for path in plan_paths for row in _read_csv(path)]
    if not plan_rows:
        raise ValueError("selected dry-run plan has zero rows")

    # Gate all checkpoints globally before evaluating any nonzero perturbation.
    baselines = _load_clean_baselines(plan_rows, audit_rows, device)
    clean_raw = []
    for row in plan_rows:
        if float(row["rho"]) == 0.0:
            ckpt = str(Path(row["checkpoint_path"]).resolve())
            clean_raw.append(
                _result_row(
                    row,
                    baselines[ckpt],
                    baselines[ckpt],
                    cached_clean=True,
                )
            )
    nonzero = _evaluate_nonzero_rows(plan_rows, output_root, audit_rows, device)
    raw_rows = clean_raw + nonzero
    _write_rows(output_root / "raw_evaluations.csv", raw_rows)
    _aggregate(raw_rows, output_root)
    print(
        f"Completed {len(raw_rows)} condition rows; clean baselines reused per checkpoint, "
        f"nonzero graph evaluations={len(nonzero)}. Results: {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
