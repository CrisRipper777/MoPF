"""Run and summarize the sports-copurchase LP baseline benchmark.

The benchmark is intentionally explicit about its model list.  The
``configs/model`` directory also contains MAP-MAG ablation presets; only the
stable ``map_mag_v3`` preset is included here.  ``mopf`` is included as the
additional requested model.

The launcher starts one Hydra process per model.  Each process executes
``num_runs=3`` with ``seed=42``, which makes the task runner use seeds 42, 43,
and 44.  Jobs can be assigned to independent devices and are safe to resume:
an existing ``results.json`` is treated as completed unless ``--no-resume`` is
given.

Run from the project root:

    python scripts/run_sports_copurchase_lp_benchmark.py \
        --devices cuda:0 cuda:1

The generated report is ``docs/lp_benchmark_results.md`` by default.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_MODEL_DIR = PROJECT_ROOT / "configs" / "model"
DATASET = "sports-copurchase"
TASK = "lp"

# These are the baseline model configs currently documented by the project,
# plus the requested MoPF model.
# Keep MAP-MAG family selection explicit: v1/v2/full/lp are not additional
# baseline rows for this benchmark.
BASELINE_MODELS = (
    "mlp",
    "gcn",
    "sage",
    "mmgcn",
    "mgat",
    "dip",
    "dgf",
    "dmgc",
    "lgmrec",
    "map_mag_v3",
    "mopf",
)

METRICS = (
    ("val_mrr", "Val MRR"),
    ("test_mrr", "Test MRR"),
    ("test_hits@1", "Test Hits@1"),
    ("test_hits@3", "Test Hits@3"),
    ("test_hits@10", "Test Hits@10"),
)


@dataclass(frozen=True)
class Job:
    index: int
    model: str
    model_config: str
    base_seed: int
    num_runs: int
    run_dir: Path


@dataclass(frozen=True)
class JobResult:
    index: int
    model: str
    model_config: str
    device: str
    status: str
    returncode: int | None
    seconds: float
    run_dir: str
    log_path: str
    message: str = ""


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the sports-copurchase LP baseline benchmark and write its Markdown report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="Independent device lanes; at most one child job runs on each device.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_MODELS,
        default=list(BASELINE_MODELS),
        help="Subset of baseline model configs to run.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="First seed; with the default num-runs this yields 42, 43, and 44.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per model.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "lp_benchmark",
        help="Directory for per-model Hydra outputs and launcher metadata.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "lp_benchmark_results.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Run a model even when its results.json already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop assigning new jobs after the first failed child process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest and print commands without launching training.",
    )
    args, extra_overrides = parser.parse_known_args()
    if extra_overrides and extra_overrides[0] == "--":
        extra_overrides = extra_overrides[1:]
    return args, extra_overrides


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.num_runs < 1:
        parser.error("--num-runs must be >= 1")
    if not args.devices:
        parser.error("at least one --devices value is required")
    if len(set(args.devices)) != len(args.devices):
        parser.error("duplicate devices would create unsafe same-device concurrency")

    missing = [
        model
        for model in dict.fromkeys(args.models)
        if not (CONFIG_MODEL_DIR / f"{model}.yaml").is_file()
    ]
    if missing:
        parser.error(
            "missing model config(s): "
            + ", ".join(missing)
            + f" under {CONFIG_MODEL_DIR}"
        )


def _selected_models(args: argparse.Namespace) -> list[str]:
    return list(dict.fromkeys(args.models))


def _absolute_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_jobs(args: argparse.Namespace) -> list[Job]:
    output_root = _absolute_path(args.output_root)
    jobs: list[Job] = []
    for index, model in enumerate(_selected_models(args), start=1):
        jobs.append(
            Job(
                index=index,
                model=model,
                model_config=model,
                base_seed=args.base_seed,
                num_runs=args.num_runs,
                run_dir=(
                    output_root
                    / DATASET
                    / model
                    / f"seed{args.base_seed}_runs{args.num_runs}"
                ),
            )
        )
    return jobs


def _build_command(
    job: Job,
    device: str,
    args: argparse.Namespace,
    extra_overrides: Iterable[str],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.main",
        f"dataset={DATASET}",
        f"task={TASK}",
        f"model={job.model_config}",
        f"seed={job.base_seed}",
        f"num_runs={job.num_runs}",
        f"device={device}",
        f"hydra.run.dir={job.run_dir}",
    ]
    command.extend(extra_overrides)
    return command


def _format_command(command: Iterable[str]) -> str:
    return " ".join(
        repr(part) if any(ch in part for ch in " \t\n") else part
        for part in command
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _manifest_payload(
    jobs: list[Job],
    args: argparse.Namespace,
    extra_overrides: list[str],
) -> dict[str, object]:
    return {
        "script": Path(__file__).name,
        "project_root": str(PROJECT_ROOT),
        "dataset": DATASET,
        "task": TASK,
        "models": [job.model for job in jobs],
        "base_seed": args.base_seed,
        "num_runs": args.num_runs,
        "seeds": list(range(args.base_seed, args.base_seed + args.num_runs)),
        "devices": list(args.devices),
        "extra_overrides": list(extra_overrides),
        "jobs": [
            {
                "index": job.index,
                "model": job.model,
                "model_config": job.model_config,
                "base_seed": job.base_seed,
                "num_runs": job.num_runs,
                "run_dir": str(job.run_dir),
            }
            for job in jobs
        ],
    }


def _run_one_job(
    job: Job,
    device: str,
    args: argparse.Namespace,
    extra_overrides: list[str],
    print_lock: threading.Lock,
) -> JobResult:
    start = time.monotonic()
    log_path = job.run_dir / "launcher.log"
    result_path = job.run_dir / "results.json"

    if not args.no_resume and result_path.is_file():
        with print_lock:
            print(
                f"[skip] [{job.index}] {job.model}: {result_path} already exists",
                flush=True,
            )
        return JobResult(
            index=job.index,
            model=job.model,
            model_config=job.model_config,
            device=device,
            status="skipped",
            returncode=0,
            seconds=0.0,
            run_dir=str(job.run_dir),
            log_path=str(log_path),
            message="results.json already exists",
        )

    job.run_dir.mkdir(parents=True, exist_ok=True)
    command = _build_command(job, device, args, extra_overrides)
    with print_lock:
        print(f"[start] [{job.index}] {job.model} on {device}", flush=True)
        print(f"        {_format_command(command)}", flush=True)

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    returncode: int | None = None
    message = ""
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = int(completed.returncode)
        status = "completed" if returncode == 0 else "failed"
        if status == "failed":
            message = "child process failed; inspect launcher.log and main.log"
    except OSError as exc:
        status = "failed"
        message = f"could not launch child process: {exc}"

    seconds = time.monotonic() - start
    with print_lock:
        marker = "done" if status == "completed" else "FAIL"
        print(
            f"[{marker}] [{job.index}] {job.model} on {device} "
            f"({seconds / 60.0:.1f} min)",
            flush=True,
        )
        if message:
            print(f"       {message}: {log_path}", flush=True)

    return JobResult(
        index=job.index,
        model=job.model,
        model_config=job.model_config,
        device=device,
        status=status,
        returncode=returncode,
        seconds=seconds,
        run_dir=str(job.run_dir),
        log_path=str(log_path),
        message=message,
    )


def _run_parallel(
    jobs: list[Job],
    args: argparse.Namespace,
    extra_overrides: list[str],
) -> list[JobResult]:
    pending: queue.Queue[Job | None] = queue.Queue()
    for job in jobs:
        pending.put(job)
    for _ in args.devices:
        pending.put(None)

    print_lock = threading.Lock()
    stop_assigning = threading.Event()
    results: list[JobResult] = []
    results_lock = threading.Lock()

    def worker(device: str) -> None:
        while not stop_assigning.is_set():
            job = pending.get()
            if job is None:
                return
            result = _run_one_job(job, device, args, extra_overrides, print_lock)
            with results_lock:
                results.append(result)
            if args.fail_fast and result.status == "failed":
                stop_assigning.set()
                return

    threads = [
        threading.Thread(target=worker, args=(device,), name=f"worker-{device}")
        for device in args.devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if stop_assigning.is_set():
        assigned = {result.index for result in results}
        for job in jobs:
            if job.index not in assigned:
                results.append(
                    JobResult(
                        index=job.index,
                        model=job.model,
                        model_config=job.model_config,
                        device="",
                        status="not_started",
                        returncode=None,
                        seconds=0.0,
                        run_dir=str(job.run_dir),
                        log_path=str(job.run_dir / "launcher.log"),
                        message="not started because --fail-fast stopped scheduling",
                    )
                )
    return sorted(results, key=lambda result: result.index)


def _read_metric(result_path: Path, key: str) -> tuple[float, float] | None:
    try:
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        item = payload[key]
        mean = float(item["mean"])
        std = float(item["std"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not math.isfinite(mean) or not math.isfinite(std):
        return None
    return mean, std


def _percent_text(metric: tuple[float, float] | None) -> str:
    if metric is None:
        return "—"
    mean, std = metric
    return f"{mean * 100.0:.4f} ± {std * 100.0:.4f}"


def _best_means(
    job_results: list[JobResult],
) -> dict[str, float]:
    best: dict[str, float] = {}
    for job_result in job_results:
        if job_result.status not in {"completed", "skipped"}:
            continue
        for key, _ in METRICS:
            metric = _read_metric(Path(job_result.run_dir) / "results.json", key)
            if metric is not None:
                best[key] = max(best.get(key, float("-inf")), metric[0])
    return best


def _format_report_metric(
    metric: tuple[float, float] | None,
    key: str,
    best: dict[str, float],
) -> str:
    if metric is None:
        return "—"
    value = _percent_text(metric)
    if math.isclose(metric[0], best.get(key, float("-inf")), rel_tol=0.0, abs_tol=1e-12):
        return f"**{value}**"
    return value


def _report_path(path: Path, report_path: Path) -> str:
    relative = os.path.relpath(path, report_path.parent)
    return relative.replace(os.sep, "/")


def _write_report(
    report_path: Path,
    jobs: list[Job],
    job_results: list[JobResult],
    args: argparse.Namespace,
) -> None:
    report_path = _absolute_path(report_path)
    result_by_model = {result.model: result for result in job_results}
    best = _best_means(job_results)
    seeds = list(range(args.base_seed, args.base_seed + args.num_runs))
    completed = sum(
        result.status in {"completed", "skipped"} for result in job_results
    )
    failed = sum(result.status == "failed" for result in job_results)
    not_started = sum(result.status == "not_started" for result in job_results)

    lines = [
        "# LP Benchmark Results",
        "",
        "## 实验设置",
        "",
        f"- 数据集：`{DATASET}`",
        f"- 模型：{len(jobs)} 个模型（MAP-MAG 仅保留 `map_mag_v3`，另含 `mopf`）",
        f"- Seeds：`{', '.join(map(str, seeds))}`",
        "- LP 协议：`unified_sampled_lp_v1`",
        "- 训练方式：sampled link prediction",
        "- Neighbor sampling：两跳 `[5, 5]`（模型特殊深度由项目 runner 解析）",
        "- Training negative：每条正边 1 个 filtered negative",
        "- LP projection dimension：`128`",
        "- Inference：full-graph exact inference",
        "- 表中数值：百分比；格式为 `mean ± population std`",
        f"- Job 状态：{completed}/{len(jobs)} 已完成，{failed} 失败，{not_started} 未启动",
        f"- 原始结果根目录：`{_report_path(_absolute_path(args.output_root), report_path)}/`",
        "",
        "普通 baseline 的配置直接对应 `configs/model/` 中的同名 YAML；"
        "MAP-MAG 系列仅运行 `map_mag_v3.yaml`，不纳入 v1/v2/full/lp preset。",
        "",
        "## 结果总览",
        "",
        "| 指标 | 最优模型 | 结果 |",
        "|---|---|---:|",
    ]

    for key, label in METRICS:
        candidates = []
        for model in [job.model for job in jobs]:
            result = result_by_model.get(model)
            if result is None or result.status not in {"completed", "skipped"}:
                continue
            metric = _read_metric(Path(result.run_dir) / "results.json", key)
            if metric is not None:
                candidates.append((metric[0], model, metric))
        if candidates:
            _, model, metric = max(candidates, key=lambda item: item[0])
            lines.append(f"| {label} | `{model}` | **{_percent_text(metric)}** |")
        else:
            lines.append(f"| {label} | — | — |")

    lines.extend(
        [
            "",
            "## 详细结果",
            "",
            "| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for job in jobs:
        result = result_by_model.get(job.model)
        if result is None or result.status not in {"completed", "skipped"}:
            cells = ["—"] * len(METRICS)
        else:
            cells = [
                _format_report_metric(
                    _read_metric(Path(result.run_dir) / "results.json", key),
                    key,
                    best,
                )
                for key, _ in METRICS
            ]
        lines.append(f"| {job.model} | " + " | ".join(cells) + " |")

    lines.extend(["", "## 结果路径", ""])
    for job in jobs:
        result_path = job.run_dir / "results.json"
        log_path = job.run_dir / "launcher.log"
        lines.append(
            f"- `{job.model}`：[{_report_path(result_path, report_path)}]"
            f"({_report_path(result_path, report_path)})；"
            f"[launcher.log]({_report_path(log_path, report_path)})"
        )

    lines.extend(
        [
            "",
            "## 运行说明",
            "",
            "本报告由 `scripts/run_sports_copurchase_lp_benchmark.py` 自动生成。"
            "若有失败 job，表中保留 `—`，请先检查对应 `launcher.log`，"
            "修复后重新执行同一命令即可续跑并刷新本报告。",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    args, extra_overrides = _parse_args()
    _validate_args(args, parser)
    jobs = _build_jobs(args)
    if not jobs:
        parser.error("no jobs selected")

    output_root = _absolute_path(args.output_root)
    report_path = _absolute_path(args.report)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "benchmark_manifest.json",
        _manifest_payload(jobs, args, extra_overrides),
    )

    print(
        f"Benchmark plan: {len(jobs)} jobs | dataset={DATASET} | "
        f"seeds={args.base_seed}-{args.base_seed + args.num_runs - 1} | "
        f"devices={','.join(args.devices)}",
        flush=True,
    )
    if extra_overrides:
        print(f"Extra Hydra overrides: {' '.join(extra_overrides)}", flush=True)

    if args.dry_run:
        for offset, job in enumerate(jobs):
            device = args.devices[offset % len(args.devices)]
            print(
                f"[{job.index}/{len(jobs)}] {job.model} on {device}: "
                f"{_format_command(_build_command(job, device, args, extra_overrides))}",
                flush=True,
            )
        print(f"Manifest: {output_root / 'benchmark_manifest.json'}", flush=True)
        return 0

    job_results = _run_parallel(jobs, args, extra_overrides)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": len(jobs),
        "completed": sum(result.status == "completed" for result in job_results),
        "skipped": sum(result.status == "skipped" for result in job_results),
        "failed": sum(result.status == "failed" for result in job_results),
        "not_started": sum(result.status == "not_started" for result in job_results),
        "results": [asdict(result) for result in job_results],
    }
    _write_json(output_root / "benchmark_summary.json", summary)
    _write_report(report_path, jobs, job_results, args)

    print(
        "Benchmark finished: "
        f"completed={summary['completed']} skipped={summary['skipped']} "
        f"failed={summary['failed']} not_started={summary['not_started']}",
        flush=True,
    )
    print(f"Summary: {output_root / 'benchmark_summary.json'}", flush=True)
    print(f"Report: {report_path}", flush=True)
    return 1 if summary["failed"] or summary["not_started"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
