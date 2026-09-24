#!/usr/bin/env python3
"""Plan or resume the P3 final-S NC ablation matrix; no training on dry-run."""

from __future__ import annotations

import argparse

from scd_p3_launcher_common import (
    add_common_args,
    make_jobs,
    provenance,
    resolve_devices,
    root_path,
    run_formal,
    write_dry_run_plan,
)

DATASETS = ["Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S"]
VARIANTS = [
    ("no_semantic_conductance", "ssi_mag_scd_ablation", "no_semantic_conductance"),
    ("fixed_restart", "ssi_mag_scd_ablation", "fixed_restart"),
    ("terminal_context_only", "ssi_mag_scd_ablation", "terminal_context_only"),
    ("global_response_only", "ssi_mag_scd_ablation", "global_response_only"),
    ("no_formation_conditioning", "ssi_mag_scd_ablation", "no_formation_conditioning"),
]
PROTOCOL = "unified_full_graph_nc_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        dataset_choices=DATASETS,
        default_datasets=DATASETS,
        default_output="outputs/scd_p3_nc_ablation",
    )
    args = parser.parse_args()
    datasets, seeds = list(args.datasets), list(args.seeds)
    output = root_path(args.output_root)
    jobs = make_jobs(
        output,
        task="nc",
        datasets=datasets,
        seeds=seeds,
        devices=resolve_devices(args),
        variants=VARIANTS,
    )
    payload = provenance(
        suite="p3_nc_s_ablation",
        task="nc",
        protocol=PROTOCOL,
        jobs=jobs,
        output=output,
        selection_metric="val_acc",
        checkpoint_selection="best_val_accuracy",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        write_dry_run_plan(
            output=output,
            suite="p3_nc_s_ablation",
            protocol=PROTOCOL,
            jobs=jobs,
            provenance_payload=payload,
        )
        print(f"Dry-run planned {len(jobs)} NC jobs; training started: false")
        return 0
    return run_formal(
        output=output,
        suite="p3_nc_s_ablation",
        protocol=PROTOCOL,
        jobs=jobs,
        provenance_payload=payload,
    )


if __name__ == "__main__":
    raise SystemExit(main())
