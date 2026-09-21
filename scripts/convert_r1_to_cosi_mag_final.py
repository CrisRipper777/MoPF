#!/usr/bin/env python3
"""Convert frozen rank-1 encoder checkpoints to the final model namespace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import build_model  # noqa: E402


OLD_MODEL_CONFIG = ROOT / "configs/model/mopf_iamoc_v2.yaml"
FINAL_MODEL_CONFIG = ROOT / "configs/model/cosi_mag_final.yaml"
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs/iamoc_v2"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/cosi_mag_final/checkpoints"
DISCARDED_KEYS = {
    "theta_transport_text",
    "theta_transport_visual",
    "theta_relation_bias_text",
    "theta_relation_bias_visual",
    "pdc_theta_text",
    "pdc_theta_visual",
}


def build_cfg(model_config: Path, model_name: str, overrides: dict[str, Any] | None = None):
    model = OmegaConf.load(model_config)
    model.name = model_name
    if overrides:
        for key, value in overrides.items():
            model[key] = value
    cfg = OmegaConf.create({"ablation": "full", "model": model})
    OmegaConf.resolve(cfg)
    return cfg


def extract_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("checkpoint does not contain a model state dictionary")
    return state


def convert_state_dict(
    source: dict[str, torch.Tensor], final_keys: set[str]
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    mapping = {key: key for key in source if key in final_keys}
    dropped = set(source) - set(mapping)
    missing = final_keys - set(mapping.values())
    unexpected = set(mapping.values()) - final_keys
    if missing:
        raise KeyError(f"missing final keys: {sorted(missing)}")
    if unexpected:
        raise KeyError(f"unexpected retained final keys: {sorted(unexpected)}")
    if dropped != DISCARDED_KEYS:
        raise KeyError(
            "unexpected source keys were dropped or expected legacy keys are absent: "
            f"dropped={sorted(dropped)}, expected={sorted(DISCARDED_KEYS)}"
        )
    converted = {target: source[source_key] for source_key, target in mapping.items()}
    return converted, mapping


def convert_one(
    checkpoint_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite converted checkpoint: {output_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("expected checkpoint payload to be a mapping")
    source = extract_state(payload)
    data_info = payload.get("data_info")
    if not isinstance(data_info, dict) or "input_dim" not in data_info:
        raise KeyError("checkpoint metadata must contain data_info.input_dim")

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
    old_model = build_model(old_cfg, data_info)
    old_model.load_state_dict(source, strict=True)
    final_model = build_model(final_cfg, data_info)
    final_template = final_model.state_dict()
    converted, mapping = convert_state_dict(source, set(final_template))
    final_model.load_state_dict(converted, strict=True)

    source_parameter_count = sum(parameter.numel() for parameter in old_model.parameters())
    final_parameter_count = sum(parameter.numel() for parameter in final_model.parameters())
    if final_parameter_count > source_parameter_count:
        raise AssertionError("final model has more parameters than the source model")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_payload = {
        **{key: value for key, value in payload.items() if key not in {"model_state", "state_dict"}},
        "model": "cosi_mag_final",
        "model_state": converted,
        "conversion_audit": {
            "source_model": "mopf_iamoc_v2",
            "source_relation_state_mode": "rank1",
            "source_checkpoint": str(checkpoint_path.resolve()),
            "mapping": mapping,
            "dropped_keys": sorted(set(source) - set(converted)),
            "missing_final_keys": [],
            "unexpected_retained_final_keys": [],
            "source_state_dict_key_count": len(source),
            "final_state_dict_key_count": len(converted),
            "source_trainable_parameter_count": source_parameter_count,
            "final_trainable_parameter_count": final_parameter_count,
            "strict_load": True,
        },
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(converted_payload, temporary)
    temporary.replace(output_path)
    return converted_payload["conversion_audit"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["Movies", "Grocery"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = []
    if args.checkpoint:
        if not args.output:
            parser.error("--output is required when --checkpoint is supplied")
        jobs.append((args.checkpoint, args.output))
    else:
        for dataset in args.datasets:
            for seed in args.seeds:
                checkpoint = (
                    args.checkpoint_root / dataset / "R1" / f"seed_{seed}" / "best.pt"
                )
                output = args.output_root / dataset / f"seed_{seed}" / "best.pt"
                jobs.append((checkpoint, output))

    audits = []
    for checkpoint, output in jobs:
        audit = convert_one(checkpoint, output, overwrite=args.overwrite)
        audits.append(audit)
        print(
            json.dumps(
                {"output": str(output), **audit}, indent=2, ensure_ascii=False
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
