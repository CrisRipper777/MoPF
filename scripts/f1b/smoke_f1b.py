#!/usr/bin/env python3
"""Run only the required short F1-B smoke checks.

The four task smoke jobs use one epoch and at most one training batch.  The
DiP check constructs the cloth LP encoder and executes one forward pass; it
does not enter the training loop.  All artifacts are isolated below
``outputs/f1b_smoke`` and are never part of the formal completion plan.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = PROJECT_ROOT / "outputs" / "f1b_smoke"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _device(value: str) -> str:
    value = str(value)
    if value.isdigit():
        return f"cuda:{value}"
    if value == "cpu" or value.startswith("cuda:"):
        return value
    raise ValueError("device must be an integer, cuda:<index>, or cpu")


def _run_group(group: str, device: str) -> dict:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "f1b" / "run_group.py"),
        "--group",
        group,
        "--gpu-id",
        device,
        "--smoke",
        "--resume",
    ]
    print(f"[SMOKE] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return {"kind": group, "device": device, "exit_code": int(result.returncode), "command": command}


def _run_dip_forward(device: str) -> dict:
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    # These imports intentionally remain inside the smoke function so --help
    # and command inspection do not require the ML runtime to initialize.
    from src.data import load_mag_data
    from src.models import build_model

    torch.manual_seed(42)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is unavailable")
    torch.set_num_threads(2)
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=cloth-copurchase",
                "task=lp",
                "model=dip",
                "seed=42",
                "num_runs=1",
                f"device={device}",
            ],
        )
    OmegaConf.resolve(cfg)
    data = load_mag_data(cfg, "lp", 42)
    info = {
        "input_dim": data.input_dim,
        "num_nodes": data.num_nodes,
        "num_classes": data.num_classes,
        "text_dim": int(data.x_t.shape[1]),
        "visual_dim": int(data.x_i.shape[1]),
    }
    model = build_model(cfg, info).to(device)
    model.eval()
    with torch.no_grad():
        z, _, _, aux_loss, _ = model(data.x.to(device), data.edge_index.to(device))
    payload = {
        "kind": "dip_cloth_lp_constructor_forward",
        "device": device,
        "dataset": "cloth-copurchase",
        "task": "lp",
        "model": "dip",
        "seed": 42,
        "input_shape": list(data.x.shape),
        "edge_shape": list(data.edge_index.shape),
        "output_shape": list(z.shape),
        "output_finite": bool(torch.isfinite(z).all().item()),
        "aux_loss_finite": bool(torch.isfinite(aux_loss).item()),
    }
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    (SMOKE_ROOT / "smoke_dip_cloth_forward.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    del z, model, data
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run required one-epoch F1-B smoke checks only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nc-device", default="0")
    parser.add_argument("--lp-device", default="1")
    parser.add_argument("--baseline-device", default="0")
    parser.add_argument("--dip-device", default="1")
    args = parser.parse_args()

    checks = [
        _run_group("mopf-nc", _device(args.nc_device)),
        _run_group("mopf-lp", _device(args.lp_device)),
        _run_group("cloth-baselines-lp", _device(args.baseline_device)),
    ]
    dip_error = None
    dip_payload = None
    try:
        dip_payload = _run_dip_forward(_device(args.dip_device))
        print(f"[SMOKE] DiP forward: {dip_payload['output_shape']} finite={dip_payload['output_finite']}", flush=True)
    except Exception as exc:  # keep a machine-readable failure report
        dip_error = repr(exc)
        print(f"[SMOKE-FAIL] DiP constructor/forward: {dip_error}", flush=True)

    payload = {
        "generated_at": _now(),
        "scope": "smoke_only",
        "formal_training_executed": False,
        "task_checks": checks,
        "dip_check": dip_payload,
        "dip_error": dip_error,
        "all_passed": all(item["exit_code"] == 0 for item in checks) and dip_error is None,
    }
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    report = SMOKE_ROOT / "smoke_report.json"
    report.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {report}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
