"""Shared checkpoint/data loading helpers for the final MGSC diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import load_mag_data  # noqa: E402
from src.models import build_model  # noqa: E402


DATASETS = ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
MODALITIES = ("text", "visual")


def resolve_device(raw: str) -> torch.device:
    device = torch.device("cuda:0" if raw == "auto" and torch.cuda.is_available() else ("cpu" if raw == "auto" else raw))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    return device


def canonical_cfg(dataset: str, seed: int):
    base = OmegaConf.load(REPO_ROOT / "configs" / "config.yaml")
    dataset_cfg = OmegaConf.load(REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml")
    task_cfg = OmegaConf.load(REPO_ROOT / "configs" / "task" / "nc.yaml")
    model_cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / "mgsc_mag_p2.yaml")
    cfg = OmegaConf.create(
        {
            "seed": int(seed),
            "paths": OmegaConf.to_container(base.paths, resolve=False),
            "dataset": dataset_cfg,
            "task": task_cfg,
            "model": model_cfg,
        }
    )
    OmegaConf.resolve(cfg)
    return cfg


def checkpoint_path(dataset: str, seed: int, root: Path) -> Path:
    candidates = (
        root / f"{dataset}_seed{seed}" / "P2" / "best.pt",
        root / f"{dataset}_seed{seed}" / "full" / "best.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "missing formal P2/full checkpoint; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_checkpoint(dataset: str, seed: int, checkpoint: Path, device: torch.device):
    cfg = canonical_cfg(dataset, seed)
    data = load_mag_data(cfg, "nc", seed)
    data_info = {
        "input_dim": int(data.x.shape[1]),
        "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes),
        "text_dim": int(data.x_t.shape[1]) if data.x_t is not None else 0,
        "visual_dim": int(data.x_i.shape[1]) if data.x_i is not None else 0,
    }
    model = build_model(cfg, data_info).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = torch.nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()
    return data, model, classifier
