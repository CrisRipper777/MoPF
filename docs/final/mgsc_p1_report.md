# MGSC-MAG P1 Pilot Report

P1 compares the frozen CoSI-MAG reference (P0) with the adaptive context-state formation candidate on the five NC datasets under the same task runner, data splits, seed, optimizer, early stopping, classifier, and graph protocol. Sports-LP is outside this scope. The old relation-context bias and relation-order bias remain in place.

**P1 decision: PASS**

- No training crash: `True`
- No primary metric drop greater than 1 percentage point: `True`
- Gate behavior non-trivial: `True`
- Text/Visual gate difference: `True`
- Gate saturation check: `True`

Primary metric deltas (P1 − P0):

- Movies-nc: -0.001499
- Toys-nc: -0.004832
- Grocery-nc: 0.000879
- ele-fashion-nc: -0.001807
- Reddit-S-nc: 0.003146

Detailed performance is in `outputs/final/mgsc_p1_pilot/summary.csv`; detailed gate order distributions are stored in each P1 run's `gate_analysis/` directory and summarized in `outputs/final/mgsc_p1_pilot/gate_summary.csv`.

P1 does not by itself establish that the mechanism is beneficial beyond this pilot. P2 remains conditional on the predeclared P1 gate.
