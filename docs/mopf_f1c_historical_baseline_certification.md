# MoPF F1-C Historical Baseline Equivalence Certification

Status: **COMPLETE — historical external baseline reuse certified by effective behavior equivalence**  
Audit date: `2026-09-14`  
Audit code SHA: `c29d692a4a97c7cfd06e044d35006392d21f2083`

## Scope and hard boundary

F1-C performed completion/provenance audits, source equivalence checks, metric aggregation, table generation, and documentation only. No external baseline was retrained, and no architecture, training, evaluator, or split file was modified. The historical NC source is the existing sibling-repository output at `/hdd1/DataInHere/YHF/MAP/MAP/outputs/full_benchmark/nc`; the historical sports source is `/hdd1/DataInHere/YHF/MoPF/outputs/lp_benchmark/sports-copurchase`.

The nine eligible external baselines are `mlp`, `gcn`, `sage`, `mmgcn`, `mgat`, `dip`, `dgf`, `dmgc`, and `lgmrec`. `map_mag*` outputs remain `historical_internal_model` and are excluded from formal main tables. The historical old `mopf` sports output is also excluded because it predates U3-B1.

## Equivalence decision

All `135` external NC seed records (`5 datasets × 9 models × 3 seeds`) and all `27` external sports seed records (`1 dataset × 9 models × 3 seeds`) are `CERTIFIED_BEHAVIOR_EQUIVALENT_REUSE`. The old producing execution SHA was not recorded in the historical output manifest; this is explicitly retained as an exception rather than fabricated. The historical NC output timestamps predate the `79a5b70` commit, so that revision is used only as a content-matching source snapshot, not asserted as the producing commit. Reuse is certified from the recorded resolved configuration/logs, the snapshot, current-source comparison, data/split hashes, and effective evaluator semantics.

### A–H checks

- Split: recorded paths and SHA256 match the current formal files. Historical NC uses the fixed `seed42` split for all three model seeds, matching the F1-B primary U3-B1 `REUSE_EXACT` policy.
- Features: dataset config and feature paths are unchanged; the old and current dataset config files are byte-identical for the audited baseline scope.
- Model behavior: each external model source and config is byte-identical between old snapshot `79a5b70` and the current repository.
- Training: resolved common NC/LP task settings match the frozen protocol, including optimizer, budget, sampling, inference, and validation cadence.
- Evaluator: LP baseline path is unchanged. NC current code uses an explicit label set, but its effective set equals the inferred set on every formal split:
- `Movies`: effective labels `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; every train/val/test split has the complete effective set: `True`.
- `Toys`: effective labels `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]`; every train/val/test split has the complete effective set: `True`.
- `Grocery`: effective labels `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; every train/val/test split has the complete effective set: `True`.
- `ele-fashion`: effective labels `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]`; every train/val/test split has the complete effective set: `True`.
- `Reddit-S`: effective labels `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`; every train/val/test split has the complete effective set: `True`.
- Checkpoint selection: NC uses validation accuracy and LP uses validation MRR. Test metrics appear after the validation-selected checkpoint and are descriptive only.
- Seed safety: successful aggregate logs contain exactly seeds `42, 43, 44`; no seed was removed. The historical sports MLP log contains one superseded partial attempt before a complete final attempt; only the complete final attempt backing `results.json` is certified.

## Final comparisons

The final NC comparison contains `45 external cells plus 5 MoPF primary cells`; the final sports comparison contains `9 external models plus MoPF`. Metric-wise strongest-baseline deltas are in `outputs/f1_final_execution/f1c_comparative_statistics.csv`. They are descriptive comparisons across the pre-registered eligible set, not post-hoc model selection and not significance tests.

| Task | Dataset | Metric | MoPF % | Strongest baseline | Baseline % | Delta pp | MoPF best | MoPF top-2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nc | Movies | test_acc | 55.9120 | lgmrec | 55.5222 | +0.3898 | 1 | 1 |
| nc | Movies | test_macro_f1 | 49.7164 | lgmrec | 49.2728 | +0.4437 | 1 | 1 |
| nc | Toys | test_acc | 79.4556 | lgmrec | 79.1818 | +0.2738 | 1 | 1 |
| nc | Toys | test_macro_f1 | 76.4430 | lgmrec | 76.5190 | -0.0760 | 0 | 0 |
| nc | Grocery | test_acc | 83.5725 | dip | 83.6506 | -0.0781 | 0 | 1 |
| nc | Grocery | test_macro_f1 | 76.1701 | dip | 75.8024 | +0.3677 | 1 | 1 |
| nc | ele-fashion | test_acc | 88.1362 | dip | 88.0430 | +0.0932 | 1 | 1 |
| nc | ele-fashion | test_macro_f1 | 77.0687 | dip | 77.3445 | -0.2757 | 0 | 0 |
| nc | Reddit-S | test_acc | 96.3511 | dip | 96.4454 | -0.0944 | 0 | 1 |
| nc | Reddit-S | test_macro_f1 | 92.0161 | dip | 92.3912 | -0.3751 | 0 | 1 |
| lp | sports-copurchase | val_mrr | 40.5950 | dip | 38.1210 | +2.4740 | 1 | 1 |
| lp | sports-copurchase | test_mrr | 37.4791 | dip | 35.1427 | +2.3364 | 1 | 1 |
| lp | sports-copurchase | test_hits@1 | 21.5611 | dip | 18.9025 | +2.6586 | 1 | 1 |
| lp | sports-copurchase | test_hits@3 | 43.0091 | dip | 40.2855 | +2.7237 | 1 | 1 |
| lp | sports-copurchase | test_hits@10 | 73.0441 | mmgcn | 72.4543 | +0.5898 | 1 | 1 |

Cloth remains unchanged from F1-B: it is a separate `cloth-copurchase` quasi-held-out table and is not combined with formal sports averages.

## Provenance artifacts

- Certification CSV: [`f1c_historical_baseline_certification.csv`](../outputs/f1_final_execution/f1c_historical_baseline_certification.csv)
- Certification JSON: [`f1c_historical_baseline_certification.json`](../outputs/f1_final_execution/f1c_historical_baseline_certification.json)
- NC full comparison: [`f1_nc_final_comparison.csv`](../outputs/f1_final_execution/tables/f1_nc_final_comparison.csv)
- NC paper table: [`f1_nc_final_comparison_paper_table.csv`](../outputs/f1_final_execution/tables/f1_nc_final_comparison_paper_table.csv)
- Sports full comparison: [`f1_lp_sports_final_comparison.csv`](../outputs/f1_final_execution/tables/f1_lp_sports_final_comparison.csv)
- Sports paper table: [`f1_lp_sports_final_comparison_paper_table.csv`](../outputs/f1_final_execution/tables/f1_lp_sports_final_comparison_paper_table.csv)

## Limitations

- Historical source manifests do not contain an original producing execution SHA, so this is equivalence-certified reuse rather than exact SHA reuse.
- Historical NC `results.json` omits validation Macro-F1. The final machine-readable NC table retains it reconstructed from the successful log's two-decimal validation display; paper-facing NC conclusions use test metrics plus the recorded validation accuracy, and the precision limitation is explicit.
- Test metrics are descriptive only. Cloth is quasi-held-out. No claim of SOTA, statistical significance, or global superiority is made without direct table support.

## F1 status and gate

**F1-C CLOSED.** The nine external NC baselines and nine external sports baselines are certified for formal comparison. Formal NC and sports results are complete and provenance-valid under the equivalence policy.

**F2 gate: PASS — Proceed F2.** F2 remains a separate future phase and was not started here.
