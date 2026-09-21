#!/usr/bin/env python3
"""Compare converted final encoders with the frozen rank-1 checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.convert_r1_to_cosi_mag_final import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    OLD_MODEL_CONFIG,
    FINAL_MODEL_CONFIG,
    build_cfg,
    convert_state_dict,
    extract_state,
)
from src.data import load_mag_data  # noqa: E402
from src.models.factory import build_model  # noqa: E402


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_reference": list(reference.shape),
            "shape_candidate": list(candidate.shape),
            "shape_match": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "relative_l2_error": None,
        }
    if reference.numel() == 0:
        return {
            "shape_reference": list(reference.shape),
            "shape_candidate": list(candidate.shape),
            "shape_match": True,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "relative_l2_error": 0.0,
        }
    ref = reference.detach().to(dtype=torch.float64, device="cpu")
    got = candidate.detach().to(dtype=torch.float64, device="cpu")
    difference = (ref - got).abs()
    relative = torch.linalg.vector_norm(ref - got) / torch.linalg.vector_norm(ref).clamp_min(1e-30)
    return {
        "shape_reference": list(reference.shape),
        "shape_candidate": list(candidate.shape),
        "shape_match": True,
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "relative_l2_error": float(relative.item()),
    }


def add_comparison(
    rows: dict[str, dict[str, Any]],
    label: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    category: str,
) -> None:
    rows[label] = {
        "category": category,
        **tensor_metrics(reference, candidate),
    }


def state_list(
    components: dict[str, Any], key: str, modality: str
) -> list[torch.Tensor]:
    value = components.get(key)
    if value is None:
        raise KeyError(f"missing {key} in old encoder components")
    return value


def compare_components(
    old: dict[str, Any],
    final: dict[str, Any],
    old_model,
    final_model,
    classifier_state: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    full: dict[str, dict[str, Any]] = {}
    shared: dict[str, dict[str, Any]] = {}

    simple = (
        ("H0_text", old["h_text"], final["h0_text"], "dense"),
        ("H0_visual", old["h_visual"], final["h0_visual"], "dense"),
        ("w_text", old["edges"]["w_t"], final["relation_weight_text"], "sparse_stage"),
        ("w_visual", old["edges"]["w_v"], final["relation_weight_visual"], "sparse_stage"),
        ("semantic_cosine_text", old["edges"]["cos_t"], final["semantic_cosine_text"], "sparse_stage"),
        ("semantic_cosine_visual", old["edges"]["cos_v"], final["semantic_cosine_visual"], "sparse_stage"),
        ("metric_weights_text", old["edges"]["metric_weights_text"], final["metric_weights_text"], "dense"),
        ("metric_weights_visual", old["edges"]["metric_weights_visual"], final["metric_weights_visual"], "dense"),
        ("Ahat_weights_text", old["norm_t_weight"], final["normalized_edge_weight_text"], "sparse_stage"),
        ("Ahat_weights_visual", old["norm_v_weight"], final["normalized_edge_weight_visual"], "sparse_stage"),
        ("local_relation_context_text", old["relation_state_text"][:, 0], final["local_relation_context_text"], "sparse_stage"),
        ("local_relation_context_visual", old["relation_state_visual"][:, 0], final["local_relation_context_visual"], "sparse_stage"),
        (
            "beta_hat_text",
            (old_model.relation_beta_raw_text - old_model.relation_beta_raw_text.mean())
            / (old_model.relation_beta_raw_text - old_model.relation_beta_raw_text.mean())
            .square()
            .mean()
            .sqrt()
            .clamp_min(old_model.relation_descriptor_eps),
            final["beta_hat_text"],
            "dense",
        ),
        (
            "beta_hat_visual",
            (old_model.relation_beta_raw_visual - old_model.relation_beta_raw_visual.mean())
            / (old_model.relation_beta_raw_visual - old_model.relation_beta_raw_visual.mean())
            .square()
            .mean()
            .sqrt()
            .clamp_min(old_model.relation_descriptor_eps),
            final["beta_hat_visual"],
            "dense",
        ),
        (
            "relation_scale_text",
            torch.sigmoid(old_model.theta_relation_scale_text),
            final["relation_scale_text"],
            "dense",
        ),
        (
            "relation_scale_visual",
            torch.sigmoid(old_model.theta_relation_scale_visual),
            final["relation_scale_visual"],
            "dense",
        ),
        (
            "hop_gate_text",
            torch.tanh(old_model.theta_hop_gate_text),
            final["hop_gate_text"],
            "dense",
        ),
        (
            "hop_gate_visual",
            torch.tanh(old_model.theta_hop_gate_visual),
            final["hop_gate_visual"],
            "dense",
        ),
        ("hop_attention_text", old["hop_attention_text"], final["attention_text"], "stage3"),
        ("hop_attention_visual", old["hop_attention_visual"], final["attention_visual"], "stage3"),
        ("delta_text", old["delta_node_text"], final["delta_text"], "stage3"),
        ("delta_visual", old["delta_node_visual"], final["delta_visual"], "stage3"),
        ("eta_text", old["eta_text"], final["eta_text"], "stage3"),
        ("eta_visual", old["eta_visual"], final["eta_visual"], "stage3"),
        ("Z_text", old["z_text"], final["z_text"], "stage3"),
        ("Z_visual", old["z_visual"], final["z_visual"], "stage3"),
        ("fused_Z", old["z"], final["z"], "fusion"),
    )
    for label, reference, candidate, category in simple:
        add_comparison(full, label, reference, candidate, category)

    for modality, old_norm_index, final_norm_index in (
        ("text", old["norm_t_index"], final["normalized_edge_index_text"]),
        ("visual", old["norm_v_index"], final["normalized_edge_index_visual"]),
    ):
        shared_index = torch.equal(old_norm_index, final_norm_index)
        full[f"Ahat_edge_index_{modality}"] = {
            "category": "sparse_stage",
            "exact_match": bool(shared_index),
            **tensor_metrics(old_norm_index, final_norm_index),
        }

    for modality in ("text", "visual"):
        old_states = old[f"states_{modality}"]
        final_states = final[f"states_{modality}"]
        old_interacted = old[f"conditioned_states_{modality}"]
        final_interacted = final[f"interacted_states_{modality}"]
        if len(old_states) != len(final_states) or len(old_interacted) != len(final_interacted):
            raise AssertionError(f"multi-order bank length mismatch for {modality}")
        for order, (reference, candidate) in enumerate(zip(old_states, final_states)):
            add_comparison(
                full,
                f"S_{order}_{modality}",
                reference,
                candidate,
                "sparse_stage",
            )
        for order, (reference, candidate) in enumerate(zip(old_interacted, final_interacted)):
            add_comparison(
                full,
                f"S_tilde_{order}_{modality}",
                reference,
                candidate,
                "stage3",
            )

    # Feed both implementations identical Stage-I/II tensors. This isolates
    # the final attention, preference, composition, and fusion computation.
    shared_z: dict[str, torch.Tensor] = {}
    for modality in ("text", "visual"):
        old_context = old_model._relation_state_used[modality][:, 0]
        old_states = old[f"states_{modality}"]
        interacted, attention = final_model._cross_order_interaction(
            old_states,
            modality,
            old_context,
            relation_intervention="normal",
            interaction_intervention="normal",
            capture_attention=True,
        )
        delta = final_model._node_preference(interacted, modality)
        modality_residual = getattr(final_model, f"delta_gamma_{modality}")
        eta = final_model.gamma_global.unsqueeze(0) + modality_residual.unsqueeze(0) + delta
        z_modality = final_model._compose(old_states, eta)
        refined = getattr(final_model, f"{modality}_refine_norm")(
            z_modality + getattr(final_model, f"{modality}_refine_mlp")(z_modality)
        )
        shared_z[modality] = refined

        old_attention = old[f"hop_attention_{modality}"]
        add_comparison(
            shared,
            f"attention_{modality}",
            old_attention,
            attention,
            "exact_shared_forward",
        )
        for order, (reference, candidate) in enumerate(
            zip(old[f"conditioned_states_{modality}"], interacted)
        ):
            add_comparison(
                shared,
                f"S_tilde_{order}_{modality}",
                reference,
                candidate,
                "exact_shared_forward",
            )
        add_comparison(
            shared,
            f"delta_{modality}",
            old[f"delta_node_{modality}"],
            delta,
            "exact_shared_forward",
        )
        add_comparison(
            shared,
            f"eta_{modality}",
            old[f"eta_{modality}"],
            eta,
            "exact_shared_forward",
        )
        add_comparison(
            shared,
            f"Z_{modality}",
            old[f"z_{modality}"],
            z_modality,
            "exact_shared_forward",
        )
        add_comparison(
            shared,
            f"Z_refined_{modality}",
            old[f"z_{modality}_refined"],
            refined,
            "exact_shared_forward",
        )

    fused_input = torch.cat((shared_z["text"], shared_z["visual"]), dim=-1)
    shared_fused = final_model.output_norm(
        final_model.fusion_skip(fused_input) + final_model.fusion_mlp(fused_input)
    )
    add_comparison(shared, "fused_Z", old["z"], shared_fused, "exact_shared_forward")
    if classifier_state is not None:
        classifier = nn.Linear(final_model.out_dim, int(classifier_state["weight"].size(0)))
        classifier.load_state_dict(classifier_state, strict=True)
        classifier = classifier.to(shared_fused.device).eval()
        add_comparison(
            shared,
            "classifier_logits",
            classifier(old["z"]),
            classifier(shared_fused),
            "exact_shared_forward",
        )
    return full, shared


def all_pass(
    rows: dict[str, dict[str, Any]], max_abs: float, relative: float
) -> bool:
    return all(
        row.get("shape_match", False)
        and row.get("max_abs_error", float("inf")) <= max_abs
        and row.get("relative_l2_error", float("inf")) <= relative
        for row in rows.values()
    )


def evaluate_dataset(
    dataset: str,
    seed: int,
    device: torch.device,
    checkpoint_root: Path,
    converted_root: Path,
    *,
    deterministic: bool,
) -> dict[str, Any]:
    source_path = checkpoint_root / dataset / "R1" / f"seed_{seed}" / "best.pt"
    converted_path = converted_root / dataset / f"seed_{seed}" / "best.pt"
    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    converted_payload = torch.load(converted_path, map_location="cpu", weights_only=False)
    data_info = source_payload["data_info"]

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"dataset={dataset}",
                "task=nc",
                "model=mopf_iamoc_v2",
                f"seed={seed}",
                "num_runs=1",
                "ablation=full",
            ],
        )
    data = load_mag_data(cfg, "nc", seed)
    actual_data_info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0,
    }
    if actual_data_info != data_info:
        raise AssertionError(
            f"checkpoint/data metadata mismatch for {dataset}: "
            f"{data_info} != {actual_data_info}"
        )

    old_cfg = build_cfg(
        OLD_MODEL_CONFIG,
        "mopf_iamoc_v2",
        {
            "relation_state_mode": "rank1",
            "relation_conditioning": "none",
            "ppc_weight": 0.0,
            "hrc_weight": 0.0,
            "hop_interaction_layers": 1,
            "hop_interaction_heads": 1,
            "hop_interaction_dropout": 0.0,
        },
    )
    final_cfg = build_cfg(FINAL_MODEL_CONFIG, "cosi_mag_final")
    old_model = build_model(old_cfg, data_info).to(device)
    old_model.load_state_dict(extract_state(source_payload), strict=True)
    final_model = build_model(final_cfg, data_info).to(device)
    final_state, mapping = convert_state_dict(
        extract_state(source_payload), set(final_model.state_dict())
    )
    final_model.load_state_dict(final_state, strict=True)
    if final_state.keys() != converted_payload["model_state"].keys():
        raise AssertionError("saved converted checkpoint key set differs from in-memory conversion")
    for key, value in final_state.items():
        if not torch.equal(value, converted_payload["model_state"][key]):
            raise AssertionError(f"converted checkpoint changed parameter value: {key}")

    old_model.eval()
    final_model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    with torch.no_grad():
        old_components = old_model._encode_components(
            x,
            edge_index,
            _capture_hop_attention=True,
            _capture_relation_diagnostics=True,
        )
        final_components = final_model._encode_components(
            x, edge_index, capture_attention=True
        )
        full, shared = compare_components(
            old_components,
            final_components,
            old_model,
            final_model,
            source_payload.get("head_state"),
        )
    if deterministic:
        torch.use_deterministic_algorithms(False)

    cpu_like = device.type == "cpu"
    sparse_abs_tolerance = 1e-7 if cpu_like else 5e-5
    sparse_rel_tolerance = 1e-6 if cpu_like else 5e-5
    exact_abs_tolerance = 1e-7
    exact_rel_tolerance = 1e-6
    sparse_rows = {
        key: row for key, row in full.items() if row.get("category") == "sparse_stage"
    }
    non_sparse_rows = {
        key: row for key, row in full.items() if row.get("category") != "sparse_stage"
    }
    return {
        "dataset": dataset,
        "seed": seed,
        "device": str(device),
        "source_checkpoint": str(source_path),
        "converted_checkpoint": str(converted_path),
        "parameter_mapping_count": len(mapping),
        "exact_shared_forward": {
            "max_abs_tolerance": exact_abs_tolerance,
            "relative_l2_tolerance": exact_rel_tolerance,
            "pass": all_pass(shared, exact_abs_tolerance, exact_rel_tolerance),
            "comparisons": shared,
        },
        "recomputed_sparse_stage": {
            "max_abs_tolerance": sparse_abs_tolerance,
            "relative_l2_tolerance": sparse_rel_tolerance,
            "pass": all_pass(sparse_rows, sparse_abs_tolerance, sparse_rel_tolerance),
            "comparisons": sparse_rows,
        },
        "full_forward": {
            "max_abs_tolerance": sparse_abs_tolerance,
            "relative_l2_tolerance": sparse_rel_tolerance,
            "pass": all_pass(full, sparse_abs_tolerance, sparse_rel_tolerance),
            "comparisons": full,
        },
        "dense_and_stage3_forward": {
            "max_abs_tolerance": sparse_abs_tolerance,
            "relative_l2_tolerance": sparse_rel_tolerance,
            "pass": all_pass(non_sparse_rows, sparse_abs_tolerance, sparse_rel_tolerance),
            "comparisons": non_sparse_rows,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["Movies", "Grocery"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--converted-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/cosi_mag_final/verification/equivalence.json",
    )
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    device_names = args.devices or (["cuda:0"] if torch.cuda.is_available() else ["cpu"])
    reports = []
    failed = False
    for device_name in device_names:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device_name}")
        if args.deterministic:
            torch.set_num_threads(1)
        for dataset in args.datasets:
            report = evaluate_dataset(
                dataset,
                args.seed,
                device,
                args.checkpoint_root,
                args.converted_root,
                deterministic=args.deterministic,
            )
            reports.append(report)
            passed = all(
                report[key]["pass"]
                for key in (
                    "exact_shared_forward",
                    "recomputed_sparse_stage",
                    "full_forward",
                    "dense_and_stage3_forward",
                )
            )
            failed |= not passed
            print(
                f"{dataset} seed={args.seed} device={device}: "
                f"shared={report['exact_shared_forward']['pass']} "
                f"sparse={report['recomputed_sparse_stage']['pass']} "
                f"full={report['full_forward']['pass']}"
            )

    payload = {
        "model": "cosi_mag_final",
        "comparison": "same frozen parameters and inputs; independent full forward plus shared Stage-I/II replay",
        "reports": reports,
        "overall_pass": not failed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
