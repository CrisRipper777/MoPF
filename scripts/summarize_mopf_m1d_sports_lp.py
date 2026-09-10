"""Summarize the automatically triggered M1-D sports LP follow-up."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "m1d_sports_lp"
M1D_SUMMARY = ROOT / "outputs" / "m1d_pdc_functional" / "m1d_master_summary.json"


def _load_result(variant: str) -> dict:
    path = OUTPUT_ROOT / variant / "results.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: value["mean"] for key, value in payload.items()}


def main() -> None:
    current = _load_result("v0_current_seed42")
    pdc = _load_result("v1_pdc_seed42")
    metrics = (
        "val_mrr",
        "test_mrr",
        "test_hits@1",
        "test_hits@3",
        "test_hits@10",
    )
    delta = {metric: pdc[metric] - current[metric] for metric in metrics}
    summary = {
        "stage": "M1-D",
        "follow_up": "sports-copurchase LP",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "seed": 42,
            "training_runs": 1,
            "checkpoint_selection": "best Validation MRR",
            "training_mode": "sampled",
            "num_neighbors": [5, 5, 5],
            "negative_sampling": "one filtered negative per positive",
            "positive_message_edge_mask_backend": "global_eid",
            "test_used_for_selection": False,
        },
        "runs": {
            "Current": {
                "variant": "v0_current",
                "conditioner": "absolute",
                "output_dir": str(OUTPUT_ROOT / "v0_current_seed42"),
                "metrics": current,
            },
            "PDC": {
                "variant": "v1_pdc",
                "conditioner": "pdc",
                "output_dir": str(OUTPUT_ROOT / "v1_pdc_seed42"),
                "metrics": pdc,
            },
        },
        "PDC_minus_Current": delta,
        "interpretation": "Single-seed LP follow-up; descriptive only and not a basis for expanding to seeds 43/44.",
    }
    output_path = OUTPUT_ROOT / "sports_lp_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if M1D_SUMMARY.is_file():
        m1d = json.loads(M1D_SUMMARY.read_text(encoding="utf-8"))
        decision = m1d.setdefault("continue_decision", {})
        decision["sports_lp_triggered_by_this_audit"] = True
        m1d["sports_lp_follow_up"] = summary
        M1D_SUMMARY.write_text(
            json.dumps(m1d, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    report = ROOT / "docs" / "mopf_m1d_sports_lp.md"
    lines = [
        "# MoPF M1-D — sports-copurchase LP follow-up",
        "",
        "This is the automatically triggered single-seed follow-up after the NC frozen functional audit. Checkpoint selection uses Validation MRR; Test is reporting-only.",
        "",
        "| Variant | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("Current", "PDC"):
        run = summary["runs"][name]["metrics"]
        lines.append(
            f"| {name} | {run['val_mrr']:.6f} | {run['test_mrr']:.6f} | {run['test_hits@1']:.6f} | {run['test_hits@3']:.6f} | {run['test_hits@10']:.6f} |"
        )
    lines.extend(
        [
            "",
            "PDC minus Current: "
            + ", ".join(f"{metric}={delta[metric]:+.6f}" for metric in metrics)
            + ".",
            "",
            "Only seed=42 was run, as required by M1-D.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)
    print(json.dumps(delta, indent=2))


if __name__ == "__main__":
    main()
