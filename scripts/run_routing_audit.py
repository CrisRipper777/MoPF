from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCREEN = ["Movies", "ele-fashion", "Reddit-S"]
ALL_DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
SEEDS = [42, 43, 44]
OUTPUT = ROOT / "outputs/routing_audit_v1"
RESULTS = ROOT / "results/routing_audit_v1"


def model_ckpt(phase, variant, dataset, seed):
    return OUTPUT / "checkpoints" / phase / variant / f"{dataset}_seed{seed}.pt"


def run_dir(phase, variant, dataset, seed):
    return OUTPUT / "runs" / phase / variant / dataset / f"seed{seed}"


def record_job(job, state, return_code=0):
    manifest = OUTPUT / "run_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    exists = manifest.is_file()
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["phase", "variant", "dataset", "seed", "device", "state", "return_code"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "phase": job["phase"], "variant": job["variant"],
            "dataset": job["dataset"], "seed": job["seed"],
            "device": job.get("device", ""), "state": state,
            "return_code": return_code,
        })


def make_job(phase, variant, dataset, seed):
    ckpt = model_ckpt(phase, variant, dataset, seed)
    directory = run_dir(phase, variant, dataset, seed)
    return {
        "phase": phase, "variant": variant, "dataset": dataset, "seed": seed,
        "checkpoint": ckpt, "run_dir": directory,
    }


def train_one(job):
    ckpt, directory = job["checkpoint"], job["run_dir"]
    marker = directory / "complete.marker"
    metrics = directory / "metrics.json"
    if ckpt.is_file() and marker.is_file() and metrics.is_file():
        return job, 0, "reused_existing"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.main",
        f"dataset={job['dataset']}", "task=nc", "model=routing_audit",
        f"seed={job['seed']}", "num_runs=1", f"device={job['device']}",
        "model.hidden_dim=256", "model.dropout=0.2", "model.max_order=3",
        "model.context_dim=128", "model.latent_dim=32",
        f"model.routing_variant={job['variant']}",
        "task.protocol_version=unified_full_graph_nc_v1",
        "task.training_mode=full_graph", "task.optimizer=adamw",
        "task.lr=1e-3", "task.weight_decay=1e-4",
        "task.epochs=300", "task.patience=30",
        "task.early_stop_min_epoch=30", "task.early_stop_min_delta=1e-4",
        "task.grad_clip=1.0", "task.eval_every=1",
        "task.evaluate_test=false", "task.loss.aux_weight=0.0",
        f"task.save_ckpt_path={ckpt}",
        f"hydra.run.dir={directory}",
    ]
    log_path = directory / "launcher.log"
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=False, env=env,
        )
    if proc.returncode == 0 and not (ckpt.is_file() and marker.is_file() and metrics.is_file()):
        return job, 97, "incomplete_artifacts"
    return job, proc.returncode, "trained" if proc.returncode == 0 else "failed"


def run_jobs(jobs, devices):
    if not jobs:
        return
    active, failures = {}, []
    iterator = iter(jobs)
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
                job, code, state = future.result()
                record_job(job, state, code)
                print(
                    f"[{job['phase']} {job['variant']} {job['dataset']} seed={job['seed']} "
                    f"{device}] {state}" + ("" if code == 0 else f" exit={code}"),
                    flush=True,
                )
                if code:
                    failures.append((job, code))
                try:
                    next_job = next(iterator)
                except StopIteration:
                    continue
                next_job["device"] = device
                active[pool.submit(train_one, next_job)] = device
    if failures:
        details = ", ".join(
            f"{job['phase']}/{job['variant']}/{job['dataset']}/seed{job['seed']}={code}"
            for job, code in failures
        )
        raise RuntimeError(f"routing audit training failures: {details}")


def train_variants(phase, variants, datasets=SCREEN, reuse_complete=True, devices=None):
    jobs = []
    for variant in variants:
        for dataset in datasets:
            for seed in SEEDS:
                job = make_job(phase, variant, dataset, seed)
                if reuse_complete and (
                    job["checkpoint"].is_file()
                    and (job["run_dir"] / "complete.marker").is_file()
                    and (job["run_dir"] / "metrics.json").is_file()
                ):
                    record_job(job, "reused_existing")
                    continue
                jobs.append(job)
    run_jobs(jobs, devices)


def call_analyzer(phase, device):
    cmd = [
        sys.executable, "scripts/analyze_routing_audit.py",
        "--phase", phase, "--device", device,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _phase_c_job_plan(devices):
    return [make_job("phase_c", "cross_relation", ds, seed) for ds in SCREEN for seed in SEEDS]


def confirmation_variants(summary):
    winner = summary.get("winning_candidate")
    name_map = {
        "A1_global_simplex": "global_simplex",
        "A2_node_flat_simplex": "node_flat_simplex",
        "A3_hierarchical_flat_simplex": "hierarchical_flat_simplex",
        "B1_trajectory": "trajectory",
        "B2_relation": "relation",
        "C1_cross_relation": "cross_relation",
    }
    variants = ["global_simplex", "hierarchical_flat_simplex"]
    if winner:
        variants.append(name_map.get(winner, winner))
    if winner == "C1_cross_relation":
        variants.append("relation")
    return list(dict.fromkeys(variants))


def train_confirmation(summary, devices):
    variants = confirmation_variants(summary)
    phase_for = {
        "global_simplex": "phase_a",
        "node_flat_simplex": "phase_a",
        "hierarchical_flat_simplex": "phase_a",
        "trajectory": "phase_b",
        "relation": "phase_b",
        "cross_relation": "phase_c",
    }
    jobs = []
    for variant in variants:
        prior_phase = phase_for[variant]
        for dataset in ALL_DATASETS:
            for seed in SEEDS:
                if dataset in SCREEN:
                    # The selected screening checkpoint is reused.
                    continue
                job = make_job("confirmation", variant, dataset, seed)
                job["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
                jobs.append(job)
    # Confirmation uses the same model family and protocol; run path/checkpoint
    # names remain separate from discovery screening.
    run_jobs(jobs, devices)


def check_branch():
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if branch != "routing_audit":
        raise RuntimeError(f"refusing to train outside routing_audit (current branch={branch})")
    if head != "cf938f9fb6cde3b5ba1edd1e2ff5f944e228b15f":
        # This runner can be restarted after its own commits only if that policy
        # is explicitly updated. At initial start the required pinned base is exact.
        raise RuntimeError(f"expected pinned starting HEAD, found {head}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this experiment")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "a0", "phase_a", "phase_b", "phase_c", "confirmation", "finish"], default="all")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = parser.parse_args()
    check_branch()
    for device in args.devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.phase == "finish":
        subprocess.run(
            [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "finalize"],
            cwd=ROOT, check=True,
        )
        summary_path = RESULTS / "routing_audit_summary.json"
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        c_results = RESULTS / "phase_c_results.csv"
        if summary_data.get("phase_c_triggered") and not c_results.is_file():
            train_variants("phase_c", ["cross_relation"], devices=args.devices)
            call_analyzer("phase_c", args.devices[0])
            subprocess.run(
                [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "finalize"],
                cwd=ROOT, check=True,
            )
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_data.get("winning_candidate"):
            train_confirmation(summary_data, args.devices)
            subprocess.run(
                [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "confirmation"],
                cwd=ROOT, check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "finalize"],
                cwd=ROOT, check=True,
            )
        subprocess.run(
            [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "report"],
            cwd=ROOT, check=True,
        )
        return

    if args.phase in {"all", "a0"}:
        call_analyzer("a0", args.devices[0])
        if args.phase == "a0":
            return
    if args.phase in {"all", "phase_a"}:
        train_variants(
            "phase_a",
            ["global_simplex", "node_flat_simplex", "hierarchical_flat_simplex"],
            devices=args.devices,
        )
        call_analyzer("phase_a", args.devices[0])
        if args.phase == "phase_a":
            return
    if args.phase in {"all", "phase_b"}:
        train_variants(
            "phase_b", ["trajectory", "relation", "capacity_control"],
            devices=args.devices,
        )
        call_analyzer("phase_b", args.devices[0])
        if args.phase == "phase_b":
            return
    if args.phase in {"all", "phase_c", "confirmation"}:
        summary = None
        if args.phase != "confirmation":
            summary = subprocess.run(
                [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "finalize"],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            summary_path = RESULTS / "routing_audit_summary.json"
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            if args.phase == "phase_c":
                if not summary_data.get("phase_c_triggered"):
                    print("OWN_RELATION_CONTEXT_NOT_SUPPORTED; Phase C gate failed.", flush=True)
                    return
                train_variants("phase_c", ["cross_relation"], devices=args.devices)
                call_analyzer("phase_c", args.devices[0])
                # Re-evaluate H6 and possible C1 confirmation eligibility.
                subprocess.run(
                    [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "finalize"],
                    cwd=ROOT, check=True,
                )
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            elif args.phase == "all":
                if summary_data.get("phase_c_triggered"):
                    train_variants("phase_c", ["cross_relation"], devices=args.devices)
                    call_analyzer("phase_c", args.devices[0])
                    subprocess.run(
                        [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "finalize"],
                        cwd=ROOT, check=True,
                    )
                    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_data.get("winning_candidate"):
                    train_confirmation(summary_data, args.devices)
                    subprocess.run(
                        [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "confirmation"],
                        cwd=ROOT, check=True,
                    )
                    subprocess.run(
                        [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "finalize"],
                        cwd=ROOT, check=True,
                    )
        else:
            summary_data = json.loads(
                (RESULTS / "routing_audit_summary.json").read_text(encoding="utf-8")
            )
            train_confirmation(summary_data, args.devices)
            subprocess.run(
                [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "confirmation"],
                cwd=ROOT, check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "finalize"],
                cwd=ROOT, check=True,
            )
        subprocess.run(
            [sys.executable, "scripts/analyze_routing_audit.py", "--phase", "report"],
            cwd=ROOT, check=True,
        )


if __name__ == "__main__":
    main()

