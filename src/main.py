from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.ablation import build_ablation_manifest, cfg_ablation
from src.data import load_mag_data
from src.tasks import run_lp, run_nc
from src.utils.device import get_device
from src.utils.logging import setup_logger


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log_data_info(logger, data, model_name: str) -> None:
    logger.info("Dataset: %s | Source: %s | Task: %s", data.name, data.source, data.task)
    logger.info("Model: %s", model_name)
    logger.info("X: %s | dtype=%s", tuple(data.x.shape), data.x.dtype)
    if data.x_i is not None or data.x_t is not None:
        x_i_shape = tuple(data.x_i.shape) if data.x_i is not None else None
        x_t_shape = tuple(data.x_t.shape) if data.x_t is not None else None
        logger.info("X_i: %s | X_t: %s", x_i_shape, x_t_shape)
    logger.info("Graph edge_index: %s | num_nodes=%d | num_edges=%d", tuple(data.edge_index.shape), data.num_nodes, data.num_edges)
    if data.y is not None:
        logger.info("Labels: shape=%s | num_classes=%s", tuple(data.y.shape), data.num_classes)
    if data.train_idx is not None:
        logger.info(
            "NC split: train=%d | val=%d | test=%d",
            int(data.train_idx.numel()),
            int(data.val_idx.numel()),
            int(data.test_idx.numel()),
        )
    if data.edge_split is not None:
        logger.info(
            "LP split: train=%d | valid=%d x %d neg | test=%d x %d neg",
            int(data.edge_split.train["source_node"].numel()),
            int(data.edge_split.valid["source_node"].numel()),
            int(data.edge_split.valid["target_node_neg"].size(1)),
            int(data.edge_split.test["source_node"].numel()),
            int(data.edge_split.test["target_node_neg"].size(1)),
        )
    for key, value in data.info.items():
        logger.info("%s: %s", key, value)
        if key.endswith("split_path"):
            split_path = Path(str(value))
            if split_path.is_file():
                logger.info("%s_sha256: %s", key, _sha256_file(split_path))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    started_at = time.perf_counter()
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = setup_logger(output_dir, cfg.logging.level)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    ablation = cfg_ablation(cfg)
    logger.info("Ablation: %s", ablation.name)

    device = get_device(str(cfg.device))
    torch_threads = cfg.task.get("torch_threads")
    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    logger.info("Device: %s", device)

    data = load_mag_data(cfg, str(cfg.task.name), int(cfg.seed))
    _log_data_info(logger, data, str(cfg.model.name))
    manifest = build_ablation_manifest(
        cfg,
        dataset=str(data.name),
        task=str(cfg.task.name),
        seed=int(cfg.seed),
        split_source=(
            data.info.get(f"{str(cfg.task.name)}_split_path")
            or data.info.get("edge_split_path")
            or data.info.get("node_split_path")
        ),
        project_root=Path(__file__).resolve().parents[1],
    )
    with (output_dir / "ablation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Saved ablation manifest: %s", output_dir / "ablation_manifest.json")

    if int(cfg.task.epochs) <= 0:
        logger.info("task.epochs <= 0, stopping after data loading/split preparation")
        results = {}
    elif str(cfg.task.name) == "nc":
        results = run_nc(cfg, data, device, logger, output_dir)
    elif str(cfg.task.name) == "lp":
        results = run_lp(cfg, data, device, logger, output_dir)
    else:
        raise ValueError(f"Unsupported task: {cfg.task.name}")

    serializable = {key: {"mean": value[0], "std": value[1]} for key, value in results.items()}
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved results: %s", output_dir / "results.json")

    checkpoint_path = cfg.task.get("save_ckpt_path")
    checkpoint_metadata = {}
    if checkpoint_path:
        checkpoint = Path(str(checkpoint_path))
        if int(cfg.num_runs) > 1:
            checkpoint = checkpoint.with_name(
                f"{checkpoint.stem}_run{int(cfg.num_runs)}{checkpoint.suffix}"
            )
        # F2 launches use num_runs=1. For legacy multi-run invocations, the
        # aggregate launcher already owns the per-run checkpoint convention.
        if checkpoint.is_file():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                checkpoint_metadata = {
                    "selection": payload.get("selection"),
                    "best_epoch": payload.get("epoch"),
                }

    runtime_seconds = time.perf_counter() - started_at
    peak_gpu_memory_mib = None
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        try:
            peak_gpu_memory_mib = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        except (RuntimeError, ValueError):
            peak_gpu_memory_mib = None
    metrics_payload = {
        "task": str(cfg.task.name),
        "dataset": str(data.name),
        "model": str(cfg.model.name),
        "ablation": ablation.name,
        "seed": int(cfg.seed),
        "selection_metric": manifest["evaluation_metric"],
        "checkpoint_selection": manifest["checkpoint_selection"],
        "best_epoch": checkpoint_metadata.get("best_epoch"),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_metadata": checkpoint_metadata,
        "metrics": serializable,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    if checkpoint_path and Path(str(checkpoint_path)).is_file():
        with (output_dir / "complete.marker").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "complete",
                    "task": str(cfg.task.name),
                    "dataset": str(data.name),
                    "ablation": ablation.name,
                    "seed": int(cfg.seed),
                },
                f,
                indent=2,
            )
        logger.info("Saved metrics and completion marker")
    # Persist a genuinely resolved configuration for reproducibility audits.
    # The Hydra internal config may retain `${...}` interpolation markers.
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(resolved_config, resolve=True), encoding="utf-8"
    )
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2, ensure_ascii=False)
    main_log = output_dir / "main.log"
    if main_log.is_file():
        shutil.copyfile(main_log, output_dir / "train.log")


if __name__ == "__main__":
    main()
