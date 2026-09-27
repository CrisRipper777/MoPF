"""Run Phase D and the gated confirmation after Phases A–C are complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_e6_srij_final  # installs the audited E5 Phase A path
from scripts import finish_e6_srij as phase_c_impl
from scripts import e6_srij_training as training
from scripts import e6_srij_finalize as finalize

phase_c_impl.train_router_one = training.train_router_one
phase_c_impl.router_interventions = training.router_interventions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    args = parser.parse_args()
    gate = json.loads((ROOT / "results/routing_identifiability_v1/phase_b_gate.json").read_text())
    result = finalize.run_screening(args.devices, gate)
    print(json.dumps({"d0_gate": result[2], "d1_gate": result[6],
                      "winner": result[7]["winning_variant"],
                      "confirmation_run_count": len(result[8])}, indent=2))


if __name__ == "__main__": main()
