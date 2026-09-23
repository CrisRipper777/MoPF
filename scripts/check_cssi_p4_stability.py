#!/usr/bin/env python3
"""Run the required CSSI-v1 stability checks on the three-dataset smoke set."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from src.data import load_mag_data
from src.models.cssi_v1 import CSSIV1


DATASETS = ("Movies", "Grocery", "Reddit-S")
VARIANTS = ("self_brsm", "same_scalar_v1", "same_brsm", "same_brsm_mrc")


def _run(root: Path, variant: str, dataset: str) -> Path:
    return root / "runs" / variant / dataset / "seed_42"


def _load(root: Path, variant: str, dataset: str, device: torch.device):
    run = _run(root, variant, dataset)
    checkpoint_path = root / "checkpoints" / variant / dataset / "seed_42.pt"
    cfg = OmegaConf.create(json.loads((run / "resolved_config.json").read_text(encoding="utf-8")))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = load_mag_data(cfg, "nc", 42)
    model = CSSIV1(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return data, model


def _log_values(path: Path, key: str) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    # A forced smoke rerun may append to the existing Hydra log.  Keep only
    # the latest process so the reported "first" value is the current run.
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
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p4"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            data, model = _load(root, variant, dataset, device)
            with torch.no_grad():
                components = model.analysis_components(data.x.to(device), data.edge_index.to(device))
                basis_errors = []
                bound_excess = []
                zero_errors = []
                for modality in ("text", "visual"):
                    basis = components[f"response_basis_{modality}"]
                    basis_errors.append(float((basis @ basis.transpose(0, 1) - torch.eye(basis.size(0), device=device)).abs().max()))
                    for response in components[f"responses_{modality}"]:
                        zero_correction = model._adapt_responses(
                            [torch.zeros_like(response)],
                            components[f"control_{modality}"][:, :1], modality,
                        )[0][0]
                        zero_errors.append(float(zero_correction.abs().max()))
                    for response, correction in zip(components[f"responses_{modality}"], components[f"corrections_{modality}"]):
                        ratio = correction.norm(dim=-1) / response.norm(dim=-1).clamp_min(1e-8)
                        bound_excess.append(float((ratio - model.lambda_max).max()))
            log_path = _run(root, variant, dataset) / "main.log"
            controller = _log_values(log_path, "controller_gradient_norm")
            basis_grad = _log_values(log_path, "basis_gradient_norm")
            lambda_grad = _log_values(log_path, "lambda_gradient_norm")
            correction_ratio = _log_values(log_path, "mean_response_correction_ratio")
            max_ratio = _log_values(log_path, "max_response_correction_ratio")
            text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            rows.append({
                "variant": variant, "dataset": dataset, "seed": 42,
                "max_orthonormality_error": max(basis_errors),
                "max_zero_response_correction": max(zero_errors),
                "max_bound_excess": max(bound_excess),
                "finite_forward": bool(all(math.isfinite(value) for value in basis_errors + zero_errors + bound_excess)),
                "controller_gradient_first": controller[0] if controller else float("nan"),
                "basis_gradient_first": basis_grad[0] if basis_grad else float("nan"),
                "lambda_gradient_first": lambda_grad[0] if lambda_grad else float("nan"),
                "correction_ratio_first": correction_ratio[0] if correction_ratio else float("nan"),
                "max_logged_correction_ratio": max(max_ratio) if max_ratio else float("nan"),
                "log_nan_inf": bool(re.search(r"(?<![A-Za-z0-9_])(nan|inf)(?![A-Za-z0-9_])", text, re.IGNORECASE)),
            })
            del data, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    output = root / "smoke_checks.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
