# Table 2 framework-level control analysis

Table 2 is the main control table. Full, Plain Multi-Order Backbone, Raw Bank Mean, and Raw Terminal are sourced from the corrected-control matrix; Attribute Only is sourced from the completed formal functional-ablation matrix because it was not part of the corrected-control rerun. This provenance distinction is retained in the CSV and is not hidden.

The raw terminal readout is sometimes higher than Full, so Table 2 does not support a claim that every adaptive integration component is individually necessary. The plain backbone is the strongest architecture-matched control: Full is better in 14/15 paired seed comparisons for both metrics, with mean paired gains of +0.291 and +0.334 percentage points (corrected-control source).

| Dataset | Full Acc | Full F1 | Plain Acc | Plain F1 |
|---|---:|---:|---:|---:|
| Movies | 0.5643 | 0.5026 | 0.5635 | 0.5018 |
| Toys | 0.8013 | 0.7741 | 0.7959 | 0.7687 |
| Grocery | 0.8331 | 0.7581 | 0.8292 | 0.7547 |
| ele-fashion | 0.8831 | 0.7812 | 0.8809 | 0.7766 |
| Reddit-S | 0.9691 | 0.9320 | 0.9668 | 0.9295 |
