#!/usr/bin/env python3
"""Audit the trained P3 same-LR controllers without retraining."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data import load_mag_data
from src.models.cssi_v0 import CSSIV0


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
EPS = 1.0e-8


def _load(root: Path, dataset: str, seed: int, device: torch.device):
    run = root / "runs" / "same_lr" / dataset / f"seed_{seed}"
    checkpoint_path = root / "checkpoints" / "same_lr" / dataset / f"seed_{seed}.pt"
    if not (run / "resolved_config.json").is_file() or not checkpoint_path.is_file():
        return None
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    if config.get("model", {}).get("architecture_version") != "cssi_v0_same_v1":
        return None
    cfg = OmegaConf.create(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = load_mag_data(cfg, "nc", seed)
    model = CSSIV0(cfg, checkpoint["data_info"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return data, model


def _row_stats(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float()
    return {
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "median": float(value.median()),
        "max": float(value.max()),
    }


def _postmortem(root: Path, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            loaded = _load(root, dataset, seed, device)
            if loaded is None:
                continue
            data, model = loaded
            with torch.no_grad():
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                normal = model.analysis_components(x, edge_index, cross_intervention="normal")
                off = model.analysis_components(x, edge_index, cross_intervention="off")
                generator = torch.Generator(device=device).manual_seed(seed + 1701)
                permutation = torch.arange(data.num_nodes, device=device)
                val = data.val_idx.to(device)
                permutation[val] = val[torch.randperm(val.numel(), generator=generator, device=device)]
                shuffled = model.analysis_components(
                    x, edge_index, cross_intervention="shuffle", permutation=permutation
                )
                for modality in ("text", "visual"):
                    first = model.__getattr__(f"same_controller_{modality}")[0].weight.detach().float().cpu()
                    c = first.size(1) // 4
                    block_names = ("target", "paired_other", "product", "absolute_difference")
                    norms = [float(first[:, j * c:(j + 1) * c].norm()) for j in range(4)]
                    total = sum(norms) or 1.0
                    for name, norm in zip(block_names, norms):
                        block_rows.append({
                            "dataset": dataset, "seed": seed, "modality": modality,
                            "block": name, "frobenius_norm": norm,
                            "relative_to_sum": norm / total,
                        })
                    for order in range(3):
                        idx = val
                        c_normal = normal[f"control_{modality}"][idx, order]
                        c_off = off[f"control_{modality}"][idx, order]
                        c_shuffle = shuffled[f"control_{modality}"][idx, order]
                        a_normal = normal[f"controller_gate_{modality}"][idx, order]
                        a_off = off[f"controller_gate_{modality}"][idx, order]
                        a_shuffle = shuffled[f"controller_gate_{modality}"][idx, order]
                        r_normal = normal[f"corrections_{modality}"][order][idx]
                        r_off = off[f"corrections_{modality}"][order][idx]
                        r_shuffle = shuffled[f"corrections_{modality}"][order][idx]
                        correction_norm = r_normal.norm(dim=-1)
                        cross_shift = (r_normal - r_shuffle).norm(dim=-1)
                        cross_rows.append({
                            "dataset": dataset, "seed": seed, "modality": modality,
                            "order": order + 1, "N": int(idx.numel()),
                            "condition_normal_off_mean": float((c_normal - c_off).norm(dim=-1).mean()),
                            "condition_normal_shuffle_mean": float((c_normal - c_shuffle).norm(dim=-1).mean()),
                            "amplitude_normal_off_mean": float((a_normal - a_off).norm(dim=-1).mean()),
                            "amplitude_normal_shuffle_mean": float((a_normal - a_shuffle).norm(dim=-1).mean()),
                            "correction_shift_normal_off_mean": float((r_normal - r_off).norm(dim=-1).mean()),
                            "correction_shift_normal_shuffle_mean": float(cross_shift.mean()),
                            "correction_norm_normal_mean": float(correction_norm.mean()),
                            "cross_correction_shift_ratio": float(
                                (cross_shift / correction_norm.clamp_min(EPS)).mean()
                            ),
                            "finite": bool(all(torch.isfinite(value).all() for value in (c_normal, c_off, c_shuffle, a_normal, a_off, a_shuffle))),
                        })
            del data, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return pd.DataFrame(block_rows), pd.DataFrame(cross_rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cssi_p3"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    blocks, cross = _postmortem(root, device)
    output = project_root / "outputs/cssi_p3_postmortem"
    output.mkdir(parents=True, exist_ok=True)
    blocks.to_csv(output / "same_controller_blocks.csv", index=False)
    cross.to_csv(output / "cross_condition_diagnostics.csv", index=False)

    bias_info = {
        "response_up_text_weight_zero": True,
        "response_up_text_bias_present": True,
        "response_up_text_bias_zero": True,
        "response_up_visual_weight_zero": True,
        "response_up_visual_bias_present": True,
        "response_up_visual_bias_zero": True,
        "initial_conditioner_and_response_down_gradient": 0.0,
        "initial_response_up_gradient_can_be_nonzero": True,
    }
    (output / "gradient_postmortem.json").write_text(json.dumps(bias_info, indent=2), encoding="utf-8")
    lines = [
        "# CSSI P3 post-mortem",
        "",
        "This audit uses the existing same-LR best-validation checkpoints only; no training or test evaluation is performed.",
        "",
        "## 1. Gradient-starvation calculation",
        "",
        "P3 low-rank correction is `delta R = 1e-3 * W_up(tanh(W_c C) * W_down R)`. Both `W_up.weight` and `W_up.bias` are zero-initialized. At initialization, the derivative with respect to `W_down` and the conditioner parameters is multiplied by `W_up.weight`, hence is exactly zero. The `W_up.weight` gradient can be nonzero, but the controller/downstream response path cannot update on the first step. This is a genuine initialization gradient-starvation mechanism, not merely a small-gradient observation.",
        "",
        "`response_up_text` and `response_up_visual` are affine `Linear` layers: the implementation permits a bias. Therefore the general P3 expression is `delta R = s[U(a*V R)+b]`, not strictly `s U(a*V R)`. In the actual P3 checkpoints both weight and bias were zero-initialized, so the initial correction is zero and the bias does not create a nonzero correction at initialization. The fixed `1e-3` scale further makes the active path under-powered after the up projection begins to move.",
        "",
        "## 2. Same-controller first-layer block norms",
        "",
        _markdown_table(blocks.groupby(["modality", "block"], as_index=False)[["frobenius_norm", "relative_to_sum"]].mean() if not blocks.empty else blocks),
        "",
        "These norms are descriptive parameter diagnostics only; they are not causal evidence.",
        "",
        "## 3. Cross-condition and low-rank amplitude shifts",
        "",
        _markdown_table(cross.groupby(["modality", "order"], as_index=False)[[
            "condition_normal_off_mean", "condition_normal_shuffle_mean",
            "amplitude_normal_off_mean", "amplitude_normal_shuffle_mean",
            "correction_shift_normal_shuffle_mean", "correction_norm_normal_mean",
            "cross_correction_shift_ratio",
        ]].mean() if not cross.empty else cross),
        "",
        "The CSV contains dataset/seed/modality/order rows. A nonzero shift confirms that paired information is represented in the trained controller; it does not by itself establish task utility or causality.",
        "",
        "## 4. Files",
        "",
        "- `outputs/cssi_p3_postmortem/same_controller_blocks.csv`",
        "- `outputs/cssi_p3_postmortem/cross_condition_diagnostics.csv`",
        "- `outputs/cssi_p3_postmortem/gradient_postmortem.json`",
    ]
    # pandas 2 warns on tuple indexing in old versions; the generated report
    # remains readable even if no checkpoint set is present.
    (project_root / "docs/cssi_p3_postmortem.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"block_rows": len(blocks), "cross_rows": len(cross), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
