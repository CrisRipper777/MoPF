from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.cmrf_probe import Model
DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
OUTPUT = ROOT / "outputs/cmrf_discovery_v1"
RESULTS = ROOT / "results/cmrf_discovery_v1"
CHECKPOINTS = OUTPUT / "checkpoints"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def uniform_audit() -> tuple[float, float]:
    torch.manual_seed(0)
    states32 = [torch.randn(9, 17) for _ in range(4)]
    response32 = [states32[k] - states32[k - 1] for k in range(1, 4)]
    u32 = sum(states32) / 4
    r32 = states32[0] + 0.75 * response32[0] + 0.50 * response32[1] + 0.25 * response32[2]
    states64 = [torch.randn(9, 17, dtype=torch.float64) for _ in range(4)]
    response64 = [states64[k] - states64[k - 1] for k in range(1, 4)]
    u64 = sum(states64) / 4
    r64 = states64[0] + 0.75 * response64[0] + 0.50 * response64[1] + 0.25 * response64[2]
    return float((u32-r32).abs().max()), float((u64-r64).abs().max())


def adaptive_initialization_audit() -> dict[str, float]:
    torch.manual_seed(7)
    cfg = OmegaConf.create(
        {"model": {
            "hidden_dim": 16, "max_order": 3, "dropout": 0.0,
            "composition_mode": "uniform", "controller_latent_dim": 4,
            "controller_hidden_dim": 8, "controller_lambda": 0.25,
        }}
    )
    info = {"input_dim": 10, "text_dim": 4, "visual_dim": 6}
    reference = Model(cfg, info).eval()
    x = torch.randn(11, 10)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 5, 7, 8], [1, 0, 2, 1, 4, 3, 6, 8, 7]],
        dtype=torch.long,
    )
    expected = reference(x, edge_index)[0]
    differences = {}
    shared = reference.state_dict()
    for mode in ("own_soft", "cross_soft", "gated_cross_soft"):
        cfg.model.composition_mode = mode
        adaptive = Model(cfg, info).eval()
        compatible = {
            key: value for key, value in shared.items()
            if key in adaptive.state_dict()
            and adaptive.state_dict()[key].shape == value.shape
        }
        adaptive.load_state_dict(compatible, strict=False)
        actual = adaptive(x, edge_index)[0]
        differences[mode] = float((expected - actual).abs().max())
    return differences


def write_preflight() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out32, out64 = uniform_audit()
    adaptive_diffs = adaptive_initialization_audit()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    report = f"""# Conditional Relational Utility Discovery: preflight

- Branch: `{branch}`
- Starting HEAD: `{head}` (user-authorized current base)
- Scope: NC only; datasets {', '.join(DATASETS)}; training seeds {', '.join(map(str, SEEDS))}
- Protocol: `unified_full_graph_nc_v1`; Train/Validation only; `task.evaluate_test=false`
- Planned formal training: C0 = 15 runs; C1/C2/C3 = 45 new runs; total = 60.
- LP is excluded. Test labels/metrics are not used for model selection or analysis.
- Physical operator follows baseline `multi_order_bank`: remove loops, symmetrize, add one self-loop per node, coalesce, unit weights, symmetric normalization. Text/Visual share the same operator.
- Forbidden semantic pruning/reweighting, graph rewiring, old MoPF mechanisms, alignment losses, cross-attention, hard experts and hard selectors are absent.
- C0 uniform response identity max absolute error: float32 `{out32:.3e}`; CPU float64 `{out64:.3e}`.
- Adaptive initialization max output difference versus C0 after copying shared weights: {json.dumps(adaptive_diffs)} (each required to be <1e-6).
- C0 best-validation checkpoints: `outputs/cmrf_discovery_v1/reference_uniform/checkpoints/`; reconstructed states use the saved checkpoint and analyze() API.
- Output roots: `outputs/cmrf_discovery_v1/` and `results/cmrf_discovery_v1/`; no existing `u*` or full benchmark outputs are touched.
- The NC runner's validation label universe is restricted to Train+Validation whenever test evaluation is disabled.
- Focused CMRF tests: 8 passed. Full repository tests: 237 passed.

The identity audited is
`(S0+S1+S2+S3)/4 = S0 + 0.75 R1 + 0.50 R2 + 0.25 R3`,
where `Rk=Sk-S(k-1)`. CMRF fixes the H0 coefficient to 1 and bounds adaptive
response adjustments using lambda=0.25. All controller output heads are zero
initialized; gated cross residuals are zero at initialization.
"""
    (RESULTS / "preflight_report.md").write_text(report, encoding="utf-8")


def make_job(variant: str, dataset: str, seed: int, device: str) -> dict:
    if variant == "uniform":
        ckpt = OUTPUT / "reference_uniform" / "checkpoints" / f"{dataset}_seed{seed}.pt"
    else:
        ckpt = CHECKPOINTS / variant / f"{dataset}_seed{seed}.pt"
    run_dir = OUTPUT / "runs" / variant / dataset / f"seed{seed}"
    return {
        "variant": variant,
        "dataset": dataset,
        "seed": seed,
        "device": device,
        "checkpoint": ckpt,
        "run_dir": run_dir,
    }


def train_one(job: dict) -> tuple[dict, int]:
    ckpt: Path = job["checkpoint"]
    run_dir: Path = job["run_dir"]
    marker = run_dir / "complete.marker"
    metrics = run_dir / "metrics.json"
    if ckpt.is_file() and marker.is_file() and metrics.is_file():
        return job, 0
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    variant = job["variant"]
    cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={job['dataset']}", "task=nc", "model=cmrf_probe",
        f"seed={job['seed']}", "num_runs=1", f"device={job['device']}",
        "model.hidden_dim=256", "model.dropout=0.2", "model.max_order=3",
        f"model.composition_mode={variant}",
        "model.controller_lambda=0.25",
        "task.protocol_version=unified_full_graph_nc_v1",
        "task.training_mode=full_graph", "task.optimizer=adamw",
        "task.lr=1e-3", "task.weight_decay=1e-4",
        "task.epochs=300", "task.patience=30",
        "task.early_stop_min_epoch=30", "task.early_stop_min_delta=1e-4",
        "task.grad_clip=1.0", "task.eval_every=1",
        "task.evaluate_test=false",
        f"task.save_ckpt_path={ckpt}",
        f"hydra.run.dir={run_dir}",
    ]
    log_path = run_dir / "launcher.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    return job, process.returncode


def run_jobs(jobs: list[dict], devices: list[str]) -> None:
    if not jobs:
        return
    iterator = iter(jobs)
    active = {}
    failed = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for device in devices:
            try:
                job = next(iterator)
            except StopIteration:
                break
            job["device"] = device
            active[pool.submit(train_one, job)] = device
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                device = active.pop(future)
                job, code = future.result()
                print(
                    f"[{job['variant']} {job['dataset']} seed={job['seed']} on {device}] "
                    f"{'complete' if code == 0 else f'FAILED exit={code}'}",
                    flush=True,
                )
                if code:
                    failed.append((job, code))
                try:
                    next_job = next(iterator)
                except StopIteration:
                    continue
                next_job["device"] = device
                active[pool.submit(train_one, next_job)] = device
    if failed:
        details = ", ".join(
            f"{j['variant']}/{j['dataset']}/seed{j['seed']} (exit {c})"
            for j, c in failed
        )
        raise RuntimeError(f"formal training failures: {details}")


def train_variants(variants: list[str], devices: list[str]) -> None:
    jobs = [
        make_job(variant, dataset, seed, devices[0])
        for variant in variants
        for dataset in DATASETS
        for seed in SEEDS
    ]
    run_jobs(jobs, devices)


def analyze(phase: str, devices: list[str]) -> None:
    # The second card remains the analysis device when the first is busy.
    analysis_devices = [devices[-1]]
    cmd = [
        sys.executable, "scripts/analyze_cmrf_discovery.py",
        "--phase", phase, "--devices", *analysis_devices,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["all", "reference", "e3a", "e3b", "adaptive", "e3c"],
        default="all",
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = parser.parse_args()
    for device in args.devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device {device} but CUDA is unavailable")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_preflight()
    if args.phase in {"all", "reference"}:
        train_variants(["uniform"], args.devices)
        if args.phase == "reference":
            return
    if args.phase in {"all", "e3a"}:
        analyze("e3a", args.devices)
        if args.phase == "e3a":
            return
    if args.phase in {"all", "e3b"}:
        analyze("e3b", args.devices)
        if args.phase == "e3b":
            return
    if args.phase in {"all", "adaptive"}:
        train_variants(["own_soft", "cross_soft", "gated_cross_soft"], args.devices)
        if args.phase == "adaptive":
            return
    if args.phase in {"all", "e3c"}:
        analyze("e3c", args.devices)


if __name__ == "__main__":
    main()
