#!/usr/bin/env python3
"""Exercise final-model analysis APIs and checkpoint round-trip on a graph slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models.factory import build_model  # noqa: E402


def data_info_for(data) -> dict[str, int]:
    return {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.size(1)) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.size(1)) if data.x_i is not None else 0,
    }


def tensors_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensors_finite(item) for item in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Movies")
    parser.add_argument("--task", choices=("nc", "lp"), default="nc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/cosi_mag_final/smoke/analysis_smoke.json",
    )
    parser.add_argument("--analysis-nodes", type=int, default=256)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device is unavailable: {device}")

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"dataset={args.dataset}",
                f"task={args.task}",
                "model=cosi_mag_final",
                f"seed={args.seed}",
                "num_runs=1",
                f"device={device}",
            ],
        )
    data = load_mag_data(cfg, args.task, args.seed)
    model = build_model(cfg, data_info_for(data)).to(device)
    loaded_checkpoint = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload.get("state_dict", payload))
        model.load_state_dict(state, strict=True)
        loaded_checkpoint = str(args.checkpoint)

    num_nodes = min(int(args.analysis_nodes), int(data.num_nodes))
    x = data.x[:num_nodes].to(device)
    edge_index = data.edge_index
    keep = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    edge_index = edge_index[:, keep].to(device)

    model.train()
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    modes_before = [module.training for module in model.modules()]
    calibration = model.analysis_relation_calibration(x, edge_index)
    multi_order = model.analysis_multi_order(x, edge_index)
    interventions = {}
    for relation in ("normal", "off", "shuffle"):
        for interaction in ("normal", "query_collapse", "uniform", "off"):
            result = model.analysis_intervention(
                x,
                edge_index,
                relation=relation,
                interaction=interaction,
                permutation_seed=args.seed,
            )
            if not tensors_finite(result):
                raise FloatingPointError(
                    f"non-finite analysis output: relation={relation}, interaction={interaction}"
                )
            interventions[f"{relation}/{interaction}"] = {
                "finite": True,
                "attention_text_shape": list(result["attention_text"].shape),
                "attention_visual_shape": list(result["attention_visual"].shape),
                "fused_shape": list(result["z"].shape),
            }

    after = model.state_dict()
    changed = [
        key
        for key, value in after.items()
        if not torch.equal(value.detach().cpu(), before[key])
    ]
    modes_after = [module.training for module in model.modules()]
    if changed:
        raise AssertionError(f"analysis APIs changed model state: {changed}")
    if modes_before != modes_after:
        raise AssertionError("analysis APIs did not restore module training modes")
    if not tensors_finite(calibration) or not tensors_finite(multi_order):
        raise FloatingPointError("non-finite analysis output")

    model.eval()
    with torch.no_grad():
        before_roundtrip = model(x, edge_index)[0]
        inference_output = model.inference(x, edge_index, device=device)
    inference_max_error = float(
        (before_roundtrip.detach().cpu() - inference_output).abs().max().item()
    )
    inference_tolerance = 1e-7 if device.type == "cpu" else 5e-5
    inference_within_tolerance = torch.allclose(
        before_roundtrip.detach().cpu(),
        inference_output,
        atol=inference_tolerance,
        rtol=inference_tolerance,
    )
    if not inference_within_tolerance:
        raise AssertionError(
            f"inference API output error {inference_max_error} exceeds "
            f"{inference_tolerance}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_name("analysis_roundtrip.pt")
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    restored = build_model(cfg, data_info_for(data)).to(device).eval()
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(checkpoint_payload["model_state"], strict=True)
    with torch.no_grad():
        after_roundtrip = restored(x, edge_index)[0]
    max_roundtrip_error = float(
        (before_roundtrip - after_roundtrip).abs().max().detach().cpu().item()
    )
    output_exact = torch.equal(before_roundtrip, after_roundtrip)
    output_tolerance = 1e-7 if device.type == "cpu" else 5e-5
    output_close = torch.allclose(
        before_roundtrip,
        after_roundtrip,
        atol=output_tolerance,
        rtol=output_tolerance,
    )
    if not output_close:
        raise AssertionError(
            f"checkpoint round-trip output error {max_roundtrip_error} exceeds "
            f"{output_tolerance}"
        )

    report = {
        "dataset": args.dataset,
        "task": args.task,
        "seed": args.seed,
        "device": str(device),
        "loaded_checkpoint": loaded_checkpoint,
        "analysis_nodes": num_nodes,
        "analysis_edges": int(edge_index.size(1)),
        "analysis_api_finite": True,
        "supported_interventions": interventions,
        "analysis_parameter_or_buffer_changes": changed,
        "training_modes_restored": modes_before == modes_after,
        "inference_api": {
            "within_tolerance": inference_within_tolerance,
            "max_abs_error_vs_forward": inference_max_error,
            "tolerance": inference_tolerance,
        },
        "checkpoint_roundtrip": {
            "strict_load": True,
            "exact_state_dict_match": all(
                torch.equal(value.detach().cpu(), restored.state_dict()[key].detach().cpu())
                for key, value in model.state_dict().items()
            ),
            "exact_output_match": output_exact,
            "output_within_tolerance": output_close,
            "output_tolerance": output_tolerance,
            "max_abs_error": max_roundtrip_error,
            "checkpoint_path": str(checkpoint_path),
        },
        "all_pass": True,
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
