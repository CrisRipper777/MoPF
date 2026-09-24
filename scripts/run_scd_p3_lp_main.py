#!/usr/bin/env python3
"""Plan or resume the P3 sampled-LP main comparison."""

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

DATASETS = ["sports-copurchase", "books-lp"]
VARIANTS = [
    ("S", "ssi_mag_scd", "full"),
    ("generic_ppr", "ssi_mag_generic_ppr", "full"),
    ("generic_gpr", "ssi_mag_generic_gpr", "full"),
]
PROTOCOL = "unified_sampled_lp_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(
        parser,
        dataset_choices=DATASETS,
        default_datasets=DATASETS,
        default_output="outputs/scd_p3_lp_main",
    )
    args = parser.parse_args()
    output = root_path(args.output_root)
    jobs = make_jobs(
        output,
        task="lp",
        datasets=list(args.datasets),
        seeds=list(args.seeds),
        devices=resolve_devices(args),
        variants=VARIANTS,
    )
    payload = provenance(
        suite="p3_lp_main",
        task="lp",
        protocol=PROTOCOL,
        jobs=jobs,
        output=output,
        selection_metric="val_mrr",
        checkpoint_selection="best_val_mrr",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        write_dry_run_plan(
            output=output,
            suite="p3_lp_main",
            protocol=PROTOCOL,
            jobs=jobs,
            provenance_payload=payload,
        )
        print(f"Dry-run planned {len(jobs)} LP jobs; neighbors: [5, 5, 5]; training started: false")
        return 0
    return run_formal(
        output=output,
        suite="p3_lp_main",
        protocol=PROTOCOL,
        jobs=jobs,
        provenance_payload=payload,
    )


if __name__ == "__main__":
    raise SystemExit(main())
