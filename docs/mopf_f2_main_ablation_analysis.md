 # MoPF F2 main ablation analysis
 
 Status: **BLOCKED at source-artifact completeness gate**
 
 Audit date: 2026-09-15
 
 ## 1. Completeness audit
 
 The requested formal source directory `outputs/f2_ablation/` is absent from the shared workspace. The audit therefore records every preregistered expected run as missing rather than substituting nearby outputs.
 
 Expected scope:
 
 - NC: 5 datasets × 6 variants × 3 seeds = 90 runs.
 - LP: 1 dataset × 6 variants × 3 seeds = 18 runs.
 - Total expected runs: **108**.
 - Located under the requested source directory: **0/108**.
 
 All 108 rows are recorded in [f2_main_completeness.csv](/hdd1/DataInHere/YHF/MoPF/outputs/f2_main_completeness.csv). For every row, metrics/config/checkpoint and downstream integrity checks are marked `not_audited`, because no F2 run directory exists to inspect. This is an explicit missing-artifact finding, not an assumption that the runs passed.
 
 The nearby timestamped output directories and existing F1/other mechanism directories were not accepted as F2 evidence because they do not provide the requested frozen F2 variant × dataset × seed structure. In particular, no directory or file named `f2_ablation` was found in the workspace.
 
 ## 2. Main table
 
 Not computable. No F2 `metrics.json` records are available for any variant, dataset, or seed.
 
 The following requested files are intentionally **not generated with empty/NaN values**, because that would look like a completed quantitative analysis:
 
 - `f2_nc_full_results.csv`
 - `f2_lp_full_results.csv`
 - `f2_main_summary.csv`
 - `f2_seed_paired_drops.csv`
 - `f2_performance_drop_heatmap.pdf`
 - `f2_performance_drop_heatmap.png`
 
 ## 3. Per-component diagnosis
 
 Not computable. Average drops, per-dataset drops, paired seed deltas, direction consistency, and variance require the missing Full and ablation metrics.
 
 ## 4. Per-dataset observations
 
 Not computable for Movies, Toys, Grocery, ele-fashion, Reddit-S, or sports-copurchase. No dataset-level F2 comparison was inferred from other experiments.
 
 ## 5. Seed stability
 
 Not auditable. No F2 seed-level records are present, so weak/strong seed-consistency labels, seed reversals, single-seed dominance, collapse, and anomalies cannot be assigned.
 
 ## 6. Relation to E0 empirical studies
 
 The requested descriptive comparisons to E0-A, E0-B, E0-C, and E0-D are deferred. It would be scientifically unsafe to claim qualitative alignment without the F2 outcome deltas.
 
 ## 7. Anomalies
 
 The primary anomaly is an artifact/provenance mismatch: the user-specified F2 result directory is absent, while the repository documentation still describes F2 as pending. This prevents distinguishing missing transfer/mount state from an unexecuted or unexported F2 result package.
 
 No implementation crash can be diagnosed from the absent directory. Nearby logs were not treated as F2 logs.
 
 ## 8. Interaction-phase recommendation
 
 No interaction experiment should be started. The interaction phase remains **not assessable** until the F2 source artifacts pass completeness and configuration audits.
 
 ## 9. Paper-writing-safe claims
 
 At this checkpoint, the only safe claim is:
 
 > The F2 main-ablation result package was not available at the specified path during audit; therefore no quantitative component-contribution or interaction-phase conclusion was drawn.
 
 Do not report component support labels, performance drops, seed consistency, ablation gains, implementation concerns, or interaction priorities from this workspace state.
 
 ## Required next action
 
 Make the completed F2 artifact directory available at:
 
 `MoPF/outputs/f2_ablation/`
 
 with one auditable directory per dataset × variant × seed containing at least `metrics.json`, resolved config, checkpoint-selection evidence, and runtime log. Then rerun this audit from the source artifacts. No training or interaction run is authorized by this report.
 
