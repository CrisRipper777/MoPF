#!/usr/bin/env python3
"""Check the completed P5 smoke checkpoints before formal runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.data import load_mag_data
from src.models.cssi_scalar_probe import CSSIScalarProbe


DATASETS = ("Movies", "Grocery", "Reddit-S")
VARIANTS = ("static_modality", "static_order", "self_scalar", "same_scalar_mrc")


def _load(root: Path, variant: str, dataset: str, device: torch.device):
    run = root / "runs" / variant / dataset / "seed_42"
    cfg = OmegaConf.create(json.loads((run / "resolved_config.json").read_text()))
    payload = torch.load(root / "checkpoints" / variant / dataset / "seed_42.pt", map_location="cpu", weights_only=False)
    data = load_mag_data(cfg, "nc", 42)
    model = CSSIScalarProbe(cfg, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return run, data, model


def _logged_values(path: Path, key: str) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.rsplit("Resolved config:", 1)[-1]
    values = []
    for raw in re.findall(rf"{key} ([^\s|]+)", text):
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p5"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    rows = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            run, data, model = _load(root, variant, dataset, device)
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            with torch.no_grad():
                normal = model.analysis_components(x, edge_index, cross_intervention="normal")
                off = model.analysis_components(x, edge_index, cross_intervention="off")
                gates = [normal["scalar_gate_text"], normal["scalar_gate_visual"]]
                max_gate = max(float(g.abs().max()) for g in gates)
                off_control = max(
                    float(off["control_q_visual_for_text"].abs().max()),
                    float(off["control_q_text_for_visual"].abs().max()),
                )
                finite = bool(torch.isfinite(normal["z"]).all())
            model.zero_grad(set_to_none=True)
            z, _, _, _, _ = model(x, edge_index)
            z[data.train_idx.to(device)].square().mean().backward()
            if variant in {"static_modality", "static_order"}:
                gradients = [model.static_gate_theta.grad]
            elif variant == "self_scalar":
                gradients = [p.grad for n, p in model.named_parameters() if "scalar_controller" in n]
            else:
                gradients = [p.grad for n, p in model.named_parameters() if "same_controller" in n or "scalar_controller" in n]
            gradient_nonzero = bool(gradients) and all(g is not None for g in gradients) and any(float(g.abs().sum()) > 0.0 for g in gradients if g is not None)
            ratios = _logged_values(run / "main.log", "mean_response_correction_ratio")
            max_ratios = _logged_values(run / "main.log", "max_response_correction_ratio")
            rows.append({
                "variant": variant,
                "dataset": dataset,
                "seed": 42,
                "max_gate_abs": max_gate,
                "gate_bound": 0.1 if variant in {"static_modality", "static_order"} else 0.2,
                "cross_off_control_max_abs": off_control,
                "finite_forward": finite,
                "controller_gradient_nonzero": gradient_nonzero,
                "initial_correction_ratio": ratios[0] if ratios else float("nan"),
                "max_logged_correction_ratio": max(max_ratios) if max_ratios else float("nan"),
                "log_nan_inf": bool(re.search(r"(?<![A-Za-z0-9_])(nan|inf)(?![A-Za-z0-9_])", (run / "main.log").read_text(errors="replace"), re.IGNORECASE)),
            })
            del data, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    output = root / "smoke_checks.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    bad = [row for row in rows if row["max_gate_abs"] > row["gate_bound"] + 1e-6 or row["cross_off_control_max_abs"] > 1e-7 or not row["finite_forward"] or not row["controller_gradient_nonzero"] or row["log_nan_inf"]]
    print(json.dumps({"rows": len(rows), "bad_rows": len(bad), "output": str(output), "bad": bad}, indent=2))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
