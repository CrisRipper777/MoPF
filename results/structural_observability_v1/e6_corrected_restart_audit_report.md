# Corrected E6 optimizer restart audit

Recomputed within-chunk restart ranges from the historical E6 audit. Each group is dataset, seed, split, oracle level, and chunk_start; the range is max(mean_best_loss) minus min(mean_best_loss) across the three restarts.

Chunks audited: 234. The E6 source CSV and E6 report were not modified.

| Statistic | Within-chunk restart range |
|---|---:|
| Mean | 0.0023696671 |
| Median | 0.0006659776 |
| P90 | 0.0094003677 |
| Maximum | 0.017789781 |

This corrected statistic isolates restart variation within the same chunk and does not mix differences among chunks, splits, or oracle levels.
