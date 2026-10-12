# CoSI-MAG VNext Pilot Report

Date: 2026-10-12 (Asia/Shanghai)

## Scope and selection protocol

This is a validation-only initial pilot, not a paper result. Three full-graph NC datasets were trained with seeds 42, 43 and 44. All seeds reused the seed-42 node split. Model selection used validation accuracy under `unified_full_graph_nc_v1`; `task.evaluate_test=false` for every run. No test metrics were read or used for selection.

Resolved model configuration: `configs/model/cosi_mag_vnext.yaml` (SHA-256 `40d651f8b283cbb407aaa84eb9b0a6a38f5a7fd9d605f3def674befd5b40eb67`). NC task configuration: `configs/task/nc.yaml` (SHA-256 `8e7e5b580f2aea3112fdd20076f57ea16c143ec57730abd5ed094634c9068e75`). Per-run resolved snapshots, their hashes and absolute paths are in [`run_manifest.json`](../reports/cosi_mag_vnext/run_manifest.json).

Principal settings: independent 256-dimensional text and visual projectors; CRC hidden/context widths 32 and edge-weight scale 0.3; Chebyshev maximum order 3; RCMI Q/K width 64; zero attention dropout; initial residual attention gate 0.1; initial eta `(0.2,0,0)`; semantic gate 0.1; concat residual late fusion. Training used AdamW, learning rate `1e-3`, weight decay `1e-4`, maximum 300 epochs, validation every epoch, patience 30 after epoch 30, gradient clipping at 1.0. NC classifier selection followed the existing runner.

The seed-42 splits were:

| Dataset | Split path | SHA-256 |
| --- | --- | --- |
| Movies | `/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt` | `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211` |
| Grocery | `/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt` | `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e` |
| Reddit-S | `/hdd1/DataInHere/YHF/data/MAGB_split/Reddit-S_nc_seed42_train0.6_val0.2.pt` | `2154e82ef061de9d3c018aa19674ea6e2220953f272bedebb1559396e43750a5` |

## Completed NC runs

All nine runs produced a completion marker, results file and a best-validation checkpoint. Values below are percentages. Best checkpoints were selected by validation accuracy; the accompanying Macro-F1 is from the same selected checkpoint. Runtime and peak GPU allocation are from the task runner's `metrics.json`.

| Dataset | Seed | Best epoch | Validation Accuracy | Validation Macro-F1 | Runtime (s) | Peak GPU (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Movies | 42 | 58 | 55.07 | 48.68 | 22.60 | 4,180 |
| Movies | 43 | 36 | 55.07 | 45.13 | 17.24 | 4,180 |
| Movies | 44 | 57 | 55.13 | 45.85 | 22.30 | 4,180 |
| Grocery | 42 | 81 | 83.51 | 76.55 | 26.91 | 3,938 |
| Grocery | 43 | 57 | 82.84 | 75.47 | 21.15 | 3,938 |
| Grocery | 44 | 83 | 83.69 | 77.13 | 27.27 | 3,938 |
| Reddit-S | 42 | 76 | 96.32 | 92.17 | 31.68 | 5,855 |
| Reddit-S | 43 | 36 | 96.23 | 92.43 | 19.60 | 5,855 |
| Reddit-S | 44 | 82 | 96.13 | 92.09 | 32.33 | 5,855 |

Population mean ± standard deviation across the three completed seeds:

| Dataset | Validation Accuracy | Validation Macro-F1 | Historical CoSI Final Accuracy | Historical CoSI Final Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Movies | 55.09 ± 0.03 | 46.55 ± 1.53 | 57.51 ± 0.26 | 49.96 ± 0.87 |
| Grocery | 83.35 ± 0.37 | 76.38 ± 0.69 | 83.91 ± 0.16 | 78.47 ± 0.71 |
| Reddit-S | 96.23 ± 0.08 | 92.23 ± 0.15 | 96.37 ± 0.05 | 93.24 ± 0.13 |

The historical values are from [`cosi_mag_final_benchmark_results.md`](cosi_mag_final_benchmark_results.md). They are reasonably comparable validation references: the NC protocol and `configs/task/nc.yaml` hash match, the historical report specifies split seed 42, and this pilot used the same split files (hashes above). Both use validation-accuracy checkpoint selection. Model configurations differ as intended. This comparison does not include test metrics.

## Smoke runs

- Movies NC, seed 42, 2 epochs, full-graph training: completed and wrote a checkpoint.
- Grocery NC, seed 42, 2 epochs, full-graph training: completed and wrote a checkpoint.
- Sports-Copurchase LP, seed 42, 1 epoch with `train_pos_per_epoch=128`, `batch_size=64`, and `max_train_batches=2`: completed two sampled training batches using `LinkNeighborLoader` with three hops. Global-eid positive-edge masking removed 72 sampled message edges before model forward. Full validation inference completed with MRR 5.63%, Hits@1 0.99%, Hits@3 3.47%, Hits@10 12.20%. This is only a one-epoch plumbing smoke and must not be interpreted as an LP performance result.

Smoke outputs:

- `/hdd1/DataInHere/YHF/MoPF_IAMOC/outputs/cosi_mag_vnext_smoke/nc/Movies/`
- `/hdd1/DataInHere/YHF/MoPF_IAMOC/outputs/cosi_mag_vnext_smoke/nc/Grocery/`
- `/hdd1/DataInHere/YHF/MoPF_IAMOC/outputs/cosi_mag_vnext_smoke/lp/sports/`

Full pilot outputs, including logs, resolved configs and checkpoints, are under `/hdd1/DataInHere/YHF/MoPF_IAMOC/outputs/cosi_mag_vnext_pilot/nc/`. The encoder has 1,277,207 parameters; each NC run has 1,282,347 parameters including the classifier. Runtime ranged from 17.2 to 32.3 seconds per run; peak allocation was 3,938–5,855 MiB.

## Mechanism diagnostics

Diagnostics were computed from each validation-selected checkpoint using only node features and the task message graph. The CRC reports distinguish modality responses, but mean edge-weight dispersion is very small:

| Dataset | Mean text cosine | Mean visual cosine | Mean absolute cosine gap | Mean text edge weight (edge std) | Mean visual edge weight (edge std) | Mean text gate | Mean visual gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Movies | 0.766 | 0.604 | 0.204 | 0.995 (0.0003) | 1.009 (0.0015) | 0.184 | 0.334 |
| Grocery | 0.601 | 0.549 | 0.157 | 0.995 (0.0005) | 1.001 (0.0031) | 0.321 | 0.432 |
| Reddit-S | 0.648 | 0.809 | 0.225 | 1.007 (0.0003) | 1.031 (0.0015) | 0.426 | 0.455 |

The first two cosine columns and the final two gate columns are ordered text/visual. Mean eta across orders 1–3 was `0.106/0.366` for Movies, `0.300/0.418` for Grocery, and `0.351/0.302` for Reddit-S (text/visual). Mean raw structural-residual RMS relative to `H0` was `0.470/0.997`, `0.818/1.187`, and `0.724/0.478`, respectively. These are descriptive summaries, not causal evidence that a component improves prediction.

The edge weights remain close to one despite nonzero cross-modal cosine gaps, so CRC's current learned reweighting is modest. The eta and gate values vary more, but that alone does not show that relation context caused the validation results. Per-run values are in [`mechanism_summary.json`](../reports/cosi_mag_vnext/mechanism_summary.json) and [`mechanism_summary.csv`](../reports/cosi_mag_vnext/mechanism_summary.csv).

## Assessment

The candidate is numerically stable in the exercised settings, runs through full-graph NC and sampled LP interfaces, and performs close to CoSI Final on Grocery and Reddit-S validation accuracy. It is lower on Movies and all three validation Macro-F1 comparisons. With no test evaluation and no ablations, these results do not support a claim of improvement. The strongest reason to keep the implementation as a review candidate is that the required cross-modal relation context and order-conditioned residual path are implemented and testable; the strongest concern is that CRC edge weights barely vary and may contribute little beyond ordinary normalized adjacency. No performance-driven architecture changes were made after this pilot.

## Machine-readable summaries

- [`pilot_metrics.json`](../reports/cosi_mag_vnext/pilot_metrics.json): per-seed results and population summaries.
- [`pilot_metrics.csv`](../reports/cosi_mag_vnext/pilot_metrics.csv): per-seed tabular results.
- [`mechanism_summary.json`](../reports/cosi_mag_vnext/mechanism_summary.json) and [`mechanism_summary.csv`](../reports/cosi_mag_vnext/mechanism_summary.csv): relation and residual diagnostics.
- [`run_manifest.json`](../reports/cosi_mag_vnext/run_manifest.json): run configs, split hashes, outputs and checkpoints.
- [`environment_manifest.json`](../reports/cosi_mag_vnext/environment_manifest.json) and [`correctness_summary.json`](../reports/cosi_mag_vnext/correctness_summary.json): runtime and verification records.
