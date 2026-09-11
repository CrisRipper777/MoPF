"""Run the U3-B degree-confound preflight over the frozen U3-A C1 checkpoints.

This script is intentionally analysis-only.  It does not import a TCPR model
variant and does not modify the formal config, checkpoints, or training path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.u3a import contribution_profile  # noqa: E402
from src.analysis.u3b import association_gate, centered_transport_context, partial_spearman  # noqa: E402
from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402
from scripts.run_mopf_u1_semantic_conductance import (  # noqa: E402
    DATASETS,
    K_BY_DATASET,
    SEEDS,
    _fixed_split_override,
)


OUTPUT_ROOT = ROOT / "outputs" / "u3b_transport_conditioned_composition"
U2C_SUMMARY = ROOT / "outputs" / "u2c_formal_factorial_training" / "u2c_master_summary.json"
FORMAL_MODEL_CONFIG = ROOT / "configs" / "model" / "mopf.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
        + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _compose_c1_cfg(dataset: str, seed: int, device: str):
    k = K_BY_DATASET[dataset]
    overrides = [
        f"dataset={dataset}", "task=nc", "model=mopf", f"seed={seed}", f"device={device}",
        "model.edge_weight_mode=learned_diag_cos", "model.edge_weight_temperature=0.35",
        f"model.max_order={k}", f"model.num_layers={k}", "model.num_metric_perspectives=4",
        "model.metric_init_seed=20260910", "model.metric_init_noise_std=0.01",
        "model.multihop_state_mode=anchored", "model.multihop_response_mode=cumulative",
        "model.multihop_anchor_alpha=0.1",
    ]
    split_override = _fixed_split_override(dataset)
    if split_override is not None:
        overrides.append(f"dataset.nc_split_path={split_override}")
    transport_override = (
        "model.use_transport_residual=false"
        if "use_transport_residual:" in FORMAL_MODEL_CONFIG.read_text(encoding="utf-8")
        else "+model.use_transport_residual=false"
    )
    overrides.append(transport_override)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="config", overrides=overrides)


def _source_records() -> list[dict[str, Any]]:
    summary = json.loads(U2C_SUMMARY.read_text(encoding="utf-8"))
    rows = [row for row in summary["records"] if row["variant"] == "C1"]
    rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), SEEDS.index(int(row["seed"]))))
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    actual = {(row["dataset"], int(row["seed"])) for row in rows}
    if actual != expected:
        raise RuntimeError(f"Expected exactly 15 C1 records, got {sorted(actual)}")
    for row in rows:
        if not Path(row["checkpoint"]).is_file():
            raise FileNotFoundError(row["checkpoint"])
    return rows


def _data_info(data: Any) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double()
    return {
        "mean": float(values.mean().item()) if values.numel() else 0.0,
        "std": float(values.std(unbiased=False).item()) if values.numel() else 0.0,
        "min": float(values.min().item()) if values.numel() else 0.0,
        "max": float(values.max().item()) if values.numel() else 0.0,
    }


def _run_one(record: dict[str, Any], device: str, output_root: Path, resume: bool) -> dict[str, Any]:
    path = output_root / "preflight" / f"{record['dataset']}_seed{record['seed']}.json"
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(record["checkpoint"])
    sha_before = _sha256(checkpoint)
    cfg = _compose_c1_cfg(record["dataset"], int(record["seed"]), device)
    data = load_mag_data(cfg, "nc", int(record["seed"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg, payload["data_info"] or _data_info(data))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        components = model._encode_components(x, edge_index)
        contexts = {}
        profiles = {}
        for modality, weight_key in (("text", "w_t"), ("visual", "w_v")):
            contexts[modality] = centered_transport_context(
                edge_index,
                components["edges"][weight_key],
                data.num_nodes,
            )
            profiles[modality] = contribution_profile(
                components[f"eta_{modality}"], components[f"states_{modality}"]
            )

        associations = []
        for modality in ("text", "visual"):
            factors = contexts[modality]
            outcome = profiles[modality]["response_order"]
            result = partial_spearman(
                outcome,
                factors["mean_conductance"],
                torch.log1p(factors["degree"]),
            )
            associations.append({
                "dataset": record["dataset"],
                "seed": int(record["seed"]),
                "modality": modality,
                "outcome": "response_order",
                "predictor": "mean_incident_conductance",
                **result,
                "conductance_mean": float(factors["mean_conductance"].mean().item()),
                "conductance_std": float(factors["mean_conductance"].std(unbiased=False).item()),
                "log_degree_mean": float(torch.log1p(factors["degree"]).mean().item()),
            })
        delta_order = profiles["text"]["response_order"] - profiles["visual"]["response_order"]
        conductance_gap = contexts["text"]["mean_conductance"] - contexts["visual"]["mean_conductance"]
        gap_result = partial_spearman(
            delta_order,
            conductance_gap,
            torch.log1p(contexts["text"]["degree"]),
        )
        associations.append({
            "dataset": record["dataset"],
            "seed": int(record["seed"]),
            "modality": "text_minus_visual",
            "outcome": "delta_response_order",
            "predictor": "conductance_gap",
            **gap_result,
            "conductance_gap_mean": float(conductance_gap.mean().item()),
            "conductance_gap_std": float(conductance_gap.std(unbiased=False).item()),
        })
        output = {
            "dataset": record["dataset"],
            "seed": int(record["seed"]),
            "formal_K": K_BY_DATASET[record["dataset"]],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256_before": sha_before,
            "contexts": {
                modality: {
                    "centered_mean": value["centered_mean"],
                    "graph_mean_conductance": value["graph_mean_conductance"],
                    "mean_conductance_distribution": _distribution(value["mean_conductance"]),
                    "centered_distribution": _distribution(value["centered"]),
                    "degree_distribution": _distribution(value["degree"]),
                }
                for modality, value in contexts.items()
            },
            "contexts_all_finite": bool(
                all(
                    torch.isfinite(value[key]).all()
                    for value in contexts.values()
                    for key in ("mean_conductance", "centered", "degree")
                )
            ),
            "response_order": {
                modality: profiles[modality]["response_order"].cpu().tolist()
                for modality in ("text", "visual")
            },
            "associations": associations,
        }
    del data, model, components
    after = _sha256(checkpoint)
    output["checkpoint_sha256_after"] = after
    output["checkpoint_bytes_unchanged"] = bool(sha_before == after)
    _dump(path, output)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[U3-B preflight] {args.device} unavailable; using cpu", flush=True)
        args.device = "cpu"
    output_root = args.output_root
    records = _source_records()
    rows = []
    for index, record in enumerate(records, start=1):
        print(f"[U3-B preflight] {index}/{len(records)} {record['dataset']} seed={record['seed']} {args.device}", flush=True)
        rows.append(_run_one(record, str(args.device), output_root, not args.no_resume))
    rows.sort(key=lambda row: (DATASETS.index(row["dataset"]), SEEDS.index(int(row["seed"]))))
    associations = [association for row in rows for association in row["associations"]]
    main_rows = [row for row in associations if row["predictor"] == "mean_incident_conductance"]
    gap_rows = [row for row in associations if row["predictor"] == "conductance_gap"]
    gate = association_gate(main_rows)
    gap_signs = [int(row["partial_sign"]) for row in gap_rows if int(row["partial_sign"])]
    gap_direction = max(sum(sign > 0 for sign in gap_signs), sum(sign < 0 for sign in gap_signs)) / max(len(gap_signs), 1)
    gate["cross_modal_gap"] = {
        "valid_count": len(gap_rows),
        "dominant_sign_fraction": float(gap_direction),
        "median_abs_partial_rho": float(np.median([abs(float(row["partial_rho"])) for row in gap_rows])) if gap_rows else 0.0,
        "rows": gap_rows,
    }
    gate["proceed_to_tcpr"] = bool(gate["passed"])
    summary = {
        "metadata": {
            "stage": "MoPF-vNext U3-B degree-confound preflight",
            "scope": "frozen U2-C C1 best-validation checkpoints",
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "formal_K_by_dataset": dict(K_BY_DATASET),
            "conductance_definition": "U3-A mean incident conductance on raw physical support using w_t/w_v",
            "partial_spearman_definition": "rank-transform + linear residualization on rank(log_degree) + Pearson residual correlation",
            "uses_labels": False,
            "no_training": True,
            "formal_config_modified": False,
        },
        "git_provenance": {"head": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"), "status": _git("status", "--short")},
        "validation": {
            "checkpoint_count": len(rows),
            "all_checkpoint_bytes_unchanged": all(row["checkpoint_bytes_unchanged"] for row in rows),
            "max_centered_context_mean_abs": max(abs(float(row["contexts"][modality]["centered_mean"])) for row in rows for modality in ("text", "visual")),
            "all_finite": bool(all(row["contexts_all_finite"] for row in rows)),
        },
        "associations": associations,
        "preflight_gate": gate,
        "per_checkpoint": rows,
        "artifacts": {
            "preflight_summary": str(output_root / "u3b_preflight_summary.json"),
            "transport_associations": str(output_root / "u3b_transport_associations.csv"),
            "per_checkpoint": str(output_root / "preflight"),
        },
    }
    _dump(output_root / "u3b_preflight_summary.json", summary)
    _write_csv(output_root / "u3b_transport_associations.csv", associations)
    print(json.dumps({"proceed_to_tcpr": gate["proceed_to_tcpr"], "main_gate": gate["passed"], "median_abs_partial_rho": gate["median_abs_partial_rho"], "dominant_sign_fraction": gate["overall_dominant_sign_fraction"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
