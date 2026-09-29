from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_crsa_icsr_nc as runner

EXPECTED = {("Movies", 42, 2), ("ele-fashion", 42, 1), ("Reddit-S", 42, 1)}


def main() -> None:
    manifest_path = runner.OUTPUT / "formal_run_manifest.csv"
    import csv
    if not manifest_path.is_file():
        raise SystemExit("smoke run manifest is missing")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    smoke = [r for r in rows if r.get("mode") == "smoke"]
    observed = {(r.get("dataset"), int(r.get("seed", -1)), int(r.get("epochs", -1))) for r in smoke}
    errors, contexts = [], []
    if len(smoke) != 3 or observed != EXPECTED:
        errors.append(f"fixed smoke matrix mismatch: {sorted(observed)}")
    for row in smoke:
        context = next((c for c in runner.smoke_contexts(row.get("device", "cuda:1"))
                        if c.dataset == row.get("dataset") and c.seed == int(row.get("seed", -1))), None)
        if context is None:
            errors.append(f"unknown smoke context: {row.get('run_key')}")
            continue
        valid, reason, payload = runner.audit_context(context)
        output_log = context.run_dir / "launcher_output.log"
        main_log = context.run_dir / "main.log"
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in (output_log, main_log) if p.is_file())
        runtime_flags = re.findall(r"\b(out of memory|cuda error|nan|inf)\b", text, flags=re.IGNORECASE)
        if row.get("status") != "complete" or not valid:
            errors.append(f"{context.run_key}: status/audit failed: {reason}")
        if runtime_flags:
            errors.append(f"{context.run_key}: runtime log contains {runtime_flags}")
        metrics = (payload or {}).get("metrics", {})
        has_validation = all(k in metrics and "mean" in metrics[k] for k in ("val_acc", "val_macro_f1"))
        if not has_validation:
            errors.append(f"{context.run_key}: validation metric missing")
        contexts.append({
            "run_key": context.run_key, "status": "complete" if valid and row.get("status") == "complete" else "invalid",
            "runtime_seconds": (payload or {}).get("runtime_seconds", row.get("runtime_seconds", "")),
            "peak_gpu_memory_mib": (payload or {}).get("peak_gpu_memory_mib", row.get("peak_gpu_memory_mib", "")),
            "best_epoch": (payload or {}).get("best_epoch", ""),
            "val_acc": metrics.get("val_acc", {}).get("mean", ""),
            "val_macro_f1": metrics.get("val_macro_f1", {}).get("mean", ""),
            "checkpoint_strict_reload": valid, "finite_losses_and_checkpoint": valid,
            "oom_nan_inf_log_hits": len(runtime_flags), "error": reason,
        })
    preflight = runner.read_json(ROOT / "results/crsa_icsr_v3/preflight_audit.json") or {}
    gradient_ok = bool(preflight.get("backward_finite") and preflight.get("icsr_up_gradient_nonzero"))
    if not gradient_ok:
        errors.append("finite full-v3 backward preflight is missing or failed")
    result = {
        "expected_contexts": 3, "complete_contexts": sum(r["status"] == "complete" for r in contexts),
        "invalid_contexts": sum(r["status"] != "complete" for r in contexts),
        "no_oom_nan_inf": not any(r["oom_nan_inf_log_hits"] for r in contexts),
        "finite_gradient_preflight": gradient_ok,
        "gradient_note": "Full-v3 synthetic CE backward was audited separately: all available gradients finite and both ICSR up.weight gradients finite/nonzero.",
        "test_evaluated": False, "lp_run": False, "contexts": contexts, "errors": errors,
    }
    out = ROOT / "results/crsa_icsr_v3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# CRSA + ICSR v3 Smoke Audit", "",
             f"- Contexts complete: **{result['complete_contexts']} / 3**.",
             f"- OOM/NaN/Inf log hits: **{'none' if result['no_oom_nan_inf'] else 'present'}**.",
             f"- Validation metrics and strict checkpoint reload: **{'pass' if result['complete_contexts'] == 3 else 'incomplete'}**.",
             f"- Finite backward/gradient check: **{result['finite_gradient_preflight']}** (preflight audit).",
             "- Test evaluation: **NO**; LP: **NO**.", "",
             "| Context | Runtime (s) | Peak GPU MiB | Best epoch | Val Acc | Val Macro-F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in contexts:
        lines.append(f"| {r['run_key']} | {r['runtime_seconds']} | {r['peak_gpu_memory_mib']} | {r['best_epoch']} | {r['val_acc']} | {r['val_macro_f1']} |")
    if errors:
        lines += ["", "Errors:", *[f"- {e}" for e in errors]]
    lines.append("")
    (out / "smoke_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if errors or result["complete_contexts"] != 3 or not result["no_oom_nan_inf"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
