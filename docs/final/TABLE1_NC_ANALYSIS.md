# Table 1 NC analysis

Table 1 combines the repository's existing historical baseline table with the current canonical P2 Full row. Baseline values are parsed from `paper/tables/table_nc_main.tex`; the repository README records the historical source and metric caveat. The MGSC-MAG row is the corrected-control Full result from `outputs/final/mgsc_corrected_controls/summary.csv`.

All entries are percentages, mean ± population SD over seeds 42/43/44. Ranking is recomputed by mean within each dataset and metric; `best` and `second` flags in `table1_nc.csv` are descriptive formatting helpers, not significance claims. The table should be described as best among the compared rows, not universal SOTA.

## MGSC-MAG versus strongest compared baseline

| Dataset | Accuracy delta (pp) | Macro-F1 delta (pp) |
|---|---:|---:|
| Movies | +0.84 | +1.06 |
| Toys | +0.95 | +0.89 |
| Grocery | +0.43 | +0.48 |
| ele-fashion | +0.39 | +1.31 |
| Reddit-S | +0.92 | +1.31 |

The current P2 row is competitive and is not uniformly best on both metrics. The historical baseline provenance limitation remains: the original baseline-producing execution SHA is not recoverable from this checkout.
