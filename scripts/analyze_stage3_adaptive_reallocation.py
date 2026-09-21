#!/usr/bin/env python3
"""Regenerate only Stage III same-response-bank adaptive reallocation outputs.

This is an offline checkpoint analysis. It loads the selected Full NC
checkpoints, extracts each frozen cumulative response bank and learned eta,
and writes only the three Stage-III reallocation data files plus a provenance
record. It does not evaluate, train, or rewrite Stages I/II.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_mechanism_verification import (  # noqa: E402
    DATASETS,
    FORMAL_K,
    SEEDS,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    _check_same_data_config,
    _model_state_sha256,
    _read_checkpoint_run,
    _sha256_file,
    _stage3_reallocation_rows,
    _summarize_stage3_reallocation,
    _validate_formal_model,
    _write_csv,
    _write_npz,
)
from src.data import load_mag_data  # noqa: E402
from src.data.loaders import resolve_path  # noqa: E402
from src.models.factory import build_model  # noqa: E402


FROZEN_STAGE12_FILES = (
    "stage1_relation_calibration_per_seed.csv",
    "stage1_relation_calibration_summary.csv",
    "stage2_semantic_retention_by_hop.csv",
    "stage2_semantic_retention_summary.csv",
)


def _read_sha_snapshot(output_root: Path) -> dict[str, str]:
    paths = [output_root / name for name in FROZEN_STAGE12_FILES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen Stage I/II outputs are missing: {missing}")
    return {path.name: _sha256_file(path) for path in paths}


def run_stage3_reallocation(
    checkpoint_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_root = checkpoint_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage12_before = _read_sha_snapshot(output_root)
    rows: list[dict[str, Any]] = []
    node_parts: list[dict[str, np.ndarray]] = []
    checkpoint_records: list[dict[str, Any]] = []
    data_sources: dict[str, dict[str, Any]] = {}

    for dataset in DATASETS:
        configs: dict[int, Any] = {}
        payloads: dict[int, dict[str, Any]] = {}
        checkpoints: dict[int, tuple[Path, str]] = {}
        first_config = None
        for seed in SEEDS:
            config, payload, checkpoint_path, checkpoint_sha, _run_info = _read_checkpoint_run(
                checkpoint_root, dataset, seed
            )
            if first_config is None:
                first_config = config
            else:
                _check_same_data_config(first_config, config, dataset)
            configs[seed] = config
            payloads[seed] = payload
            checkpoints[seed] = (checkpoint_path, checkpoint_sha)
        assert first_config is not None

        data = load_mag_data(first_config, "nc", SEEDS[0])
        x = data.x.to(device=device, dtype=torch.float32).contiguous()
        edge_index = data.edge_index.to(device=device, dtype=torch.long).contiguous()
        if int(x.size(0)) != int(data.num_nodes):
            raise RuntimeError(f"{dataset}: feature and graph node counts do not match")
        graph_path = first_config.dataset.get("graph_path") or first_config.dataset.get("edge_path")
        if graph_path is None:
            raise ValueError(f"{dataset}: config does not identify its graph/edge input")
        data_sources[dataset] = {
            "graph_input_path": str(resolve_path(graph_path)),
            "num_nodes": int(data.num_nodes),
            "loader_directed_edges": int(edge_index.size(1)),
            "formal_K": FORMAL_K[dataset],
        }

        for seed in SEEDS:
            config = configs[seed]
            payload = payloads[seed]
            checkpoint_path, checkpoint_sha_before = checkpoints[seed]
            model = build_model(config, payload["data_info"])
            model.load_state_dict(payload["model_state"], strict=True)
            model.to(device)
            model.eval()
            model_config = _validate_formal_model(model, dataset)
            model_sha_before = _model_state_sha256(model)

            with torch.inference_mode():
                components = model._encode_components(x, edge_index)
                rows.extend(_stage3_reallocation_rows(dataset, seed, components, node_parts))

            model_sha_after = _model_state_sha256(model)
            checkpoint_sha_after = _sha256_file(checkpoint_path)
            if model_sha_after != model_sha_before:
                raise RuntimeError(f"{dataset}/seed{seed}: model state changed during Stage-III analysis")
            if checkpoint_sha_after != checkpoint_sha_before:
                raise RuntimeError(f"Checkpoint changed during Stage-III analysis: {checkpoint_path}")
            checkpoint_records.append({
                "dataset": dataset,
                "seed": seed,
                "formal_K": FORMAL_K[dataset],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha_before,
                "checkpoint_sha256_unchanged": checkpoint_sha_after == checkpoint_sha_before,
                "checkpoint_selection": payload["selection"],
                "model_config": model_config,
                "model_state_sha256_before": model_sha_before,
                "model_state_sha256_after": model_sha_after,
                "model_state_unchanged": model_sha_after == model_sha_before,
            })
            del components, model, payload
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del x, edge_index, data, payloads
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = _summarize_stage3_reallocation(rows)
    _write_csv(output_root / "stage3_adaptive_reallocation_per_seed.csv", rows)
    _write_csv(output_root / "stage3_adaptive_reallocation_summary.csv", summary)
    _write_npz(output_root / "stage3_adaptive_reallocation_nodes.npz", node_parts)

    stage12_after = _read_sha_snapshot(output_root)
    if stage12_before != stage12_after:
        raise RuntimeError("A frozen Stage I/II output changed during Stage-III regeneration")
    audit = {
        "analysis": "Stage-III adaptive contribution reallocation only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "selected_checkpoint_root": str(checkpoint_root),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS),
        "formal_K": FORMAL_K,
        "device": str(device),
        "checkpoints": checkpoint_records,
        "data_sources": data_sources,
        "metric_definitions": {
            "adaptive": "g_adapt_i,k=||eta_i,k*S_i,k||_2; p_adapt_i,k=g_adapt_i,k/sum_r g_adapt_i,r",
            "uniform_same_bank": "eta_uniform_k=1/(K+1); p_uniform_i,k=||S_i,k||_2/sum_r ||S_i,r||_2 from the exact same cumulative response bank",
            "primary": "D_i=0.5*sum_k |p_adapt_i,k-p_uniform_i,k|",
            "diagnostic_only": "q_i,k=|eta_i,k|/sum_r|eta_i,r|; C_i=0.5*sum_k |q_i,k-1/(K+1)|",
            "threshold_fractions": [0.01, 0.05, 0.10],
            "replicate_unit": "seed; node distributions are descriptive and are not significance replicates",
            "summary_sd": "population SD across seeds (ddof=0)",
        },
        "outputs": [
            "stage3_adaptive_reallocation_per_seed.csv",
            "stage3_adaptive_reallocation_summary.csv",
            "stage3_adaptive_reallocation_nodes.npz",
        ],
        "sanity_checks": {
            "adaptive_and_uniform_profiles_sum_to_one": all(
                float(row["p_adapt_sum_max_error"]) <= 2e-5
                and float(row["p_uniform_sum_max_error"]) <= 2e-12
                for row in rows
            ),
            "D_in_unit_interval": all(0.0 <= float(row["D_min"]) <= float(row["D_max"]) <= 1.0 for row in rows),
            "formal_cumulative_response_bank_used": all(int(row["formal_K"]) == FORMAL_K[row["dataset"]] for row in rows),
            "model_state_unchanged": all(item["model_state_unchanged"] for item in checkpoint_records),
            "checkpoint_files_unchanged": all(item["checkpoint_sha256_unchanged"] for item in checkpoint_records),
            "frozen_stage1_stage2_files_unchanged": stage12_before == stage12_after,
            "no_training_or_optimizer_created": True,
        },
        "frozen_stage1_stage2_sha256": stage12_after,
    }
    audit_path = output_root / "stage3_adaptive_reallocation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def _git_value(*args: str) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {args.device}")
    if args.device.startswith("cuda"):
        index = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested unavailable CUDA device {args.device}")
    torch.set_num_threads(args.cpu_threads)
    result = run_stage3_reallocation(
        args.checkpoint_root, args.output_root, torch.device(args.device)
    )
    print(json.dumps({"status": "complete", "sanity_checks": result["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
