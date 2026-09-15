# Final Node Classification Benchmark (fixed Macro-F1)

## Protocol

- Protocol: `unified_full_graph_nc_v1`.
- Task: transductive full-graph node classification; one complete encoder forward per epoch; cross-entropy is computed only on train nodes.
- Checkpoint selection: validation accuracy; test metrics are descriptive and are evaluated once after restoring the selected checkpoint.
- Seeds: `42`, `43`, `44`; all models on a dataset use the same frozen split.
- Fixed Macro-F1: one task-level label set is formed from the valid train + validation + test label union, with invalid labels filtered; the same set is used for every split, model, and seed.
- Internal predecessors `map_mag`, `map_mag_v1`, `map_mag_v2`, and `map_mag_v3` are excluded from the formal ranking.
- LP results are frozen legacy results and were not rerun in this task.

## Datasets and evaluated labels

| Dataset | Nodes | Configured classes | Evaluated label set | Evaluated classes | Frozen split | Split SHA-256 |
|---|---:|---:|---|---:|---|---|
| Movies | 16672 | 20 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]` | 20 | `/hdd1/DataInHere/YHF/data/MAGB_split/Movies_nc_seed42_train0.6_val0.2.pt` | `9089d5eb1d5a3edeae46bd9296e4f85815fccf8e489452bc87a2126a88546211` |
| Toys | 20695 | 18 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]` | 18 | `/hdd1/DataInHere/YHF/data/MAGB_split/Toys_nc_seed42_train0.6_val0.2.pt` | `cc0633526581d9281d0a92fa50d989309f8b81b982305409d3034f10a28e41d2` |
| Grocery | 17074 | 20 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]` | 20 | `/hdd1/DataInHere/YHF/data/MAGB_split/Grocery_nc_seed42_train0.6_val0.2.pt` | `3ca4581592fcb705b42672b18412375d33bf6109c72e4714786066878a54d58e` |
| ele-fashion | 97766 | 12 | `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]` | 11 | `/hdd1/DataInHere/YHF/data/ele-fashion/split.pt` | `e48e74afd502df5bc5e4d638e611fc4fbfdecee461aa920f3f47a6446826c52f` |
| Reddit-S | 15894 | 20 | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]` | 20 | `/hdd1/DataInHere/YHF/data/MAGB_split/Reddit-S_nc_seed42_train0.6_val0.2.pt` | `2154e82ef061de9d3c018aa19674ea6e2220953f272bedebb1559396e43750a5` |

## Complete per-dataset results

All values are percentages and use mean ± population standard deviation.

### Movies

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Seed-level test values (42 / 43 / 44) |
|---|---:|---:|---:|---:|---|
| MLP | 51.13 ± 0.21 | 35.68 ± 2.51 | 50.91 ± 0.15 | 36.84 ± 2.22 | 42: 51.0045 / 34.1388; 43: 50.7046 / 36.8012; 44: 51.0345 / 39.5835 |
| GCN | 53.03 ± 0.09 | 43.41 ± 0.72 | 53.22 ± 0.51 | 45.30 ± 0.14 | 42: 53.7931 / 45.1278; 43: 53.3133 / 45.3093; 44: 52.5637 / 45.4764 |
| GraphSAGE | 52.81 ± 2.31 | 40.97 ± 11.80 | 52.50 ± 1.93 | 40.63 ± 10.84 | 42: 53.1334 / 47.7349; 43: 49.8951 / 25.3172; 44: 54.4828 / 48.8449 |
| MMGCN | 55.04 ± 0.09 | 46.39 ± 1.98 | 55.25 ± 1.00 | 48.00 ± 0.75 | 42: 56.6417 / 47.7562; 43: 54.7526 / 49.0137; 44: 54.3628 / 47.2182 |
| MGAT | 19.81 ± 4.50 | 5.62 ± 2.18 | 20.09 ± 4.60 | 5.97 ± 2.13 | 42: 23.3283 / 7.3712; 43: 13.5832 / 2.9558; 44: 23.3583 / 7.5719 |
| DiP | 56.30 ± 0.17 | 47.64 ± 1.02 | 55.07 ± 0.18 | 47.28 ± 0.56 | 42: 55.3223 / 48.0039; 43: 54.9325 / 47.1961; 44: 54.9625 / 46.6287 |
| DGF | 51.95 ± 0.15 | 29.07 ± 4.02 | 51.35 ± 0.28 | 31.34 ± 3.19 | 42: 51.0045 / 29.3693; 43: 51.3643 / 35.8363; 44: 51.6942 / 28.8005 |
| DMGC | 46.61 ± 1.15 | 21.31 ± 1.36 | 46.55 ± 0.57 | 21.73 ± 1.56 | 42: 47.1964 / 20.4150; 43: 45.8171 / 20.8576; 44: 46.6267 / 23.9265 |
| LGMRec | 55.91 ± 0.11 | 48.49 ± 0.77 | 55.59 ± 0.30 | 49.20 ± 0.09 | 42: 55.3823 / 49.2464; 43: 56.0120 / 49.2778; 44: 55.3823 / 49.0736 |
| MoPF | 57.20 ± 0.19 | 49.84 ± 0.74 | 56.19 ± 0.52 | 49.49 ± 0.71 | 42: 55.9820 / 48.9124; 43: 56.9115 / 50.4895; 44: 55.6822 / 49.0546 |

### Toys

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Seed-level test values (42 / 43 / 44) |
|---|---:|---:|---:|---:|---|
| MLP | 76.15 ± 0.23 | 73.83 ± 0.31 | 74.30 ± 0.29 | 71.84 ± 0.38 | 42: 73.9067 / 71.3058; 43: 74.4141 / 71.9982; 44: 74.5832 / 72.2040 |
| GCN | 79.46 ± 0.14 | 76.07 ± 0.27 | 78.77 ± 0.22 | 75.25 ± 0.44 | 42: 78.5214 / 74.6317; 43: 79.0529 / 75.6617; 44: 78.7388 / 75.4420 |
| GraphSAGE | 79.04 ± 0.50 | 76.34 ± 0.74 | 77.75 ± 0.31 | 75.19 ± 0.22 | 42: 77.4583 / 75.0372; 43: 77.6033 / 75.0327; 44: 78.1831 / 75.4993 |
| MMGCN | 79.31 ± 0.11 | 76.11 ± 0.15 | 78.36 ± 0.25 | 75.33 ± 0.43 | 42: 78.1831 / 74.9875; 43: 78.7147 / 75.9378; 44: 78.1831 / 75.0750 |
| MGAT | 39.64 ± 8.04 | 24.76 ± 6.69 | 39.70 ± 7.81 | 24.90 ± 6.52 | 42: 38.1010 / 19.4529; 43: 49.9638 / 34.0651; 44: 31.0220 / 21.1703 |
| DiP | 80.74 ± 0.03 | 77.99 ± 0.28 | 79.18 ± 0.36 | 76.61 ± 0.26 | 42: 79.1012 / 76.5433; 43: 79.6569 / 76.9554; 44: 78.7871 / 76.3297 |
| DGF | 79.31 ± 0.04 | 75.83 ± 0.45 | 78.35 ± 0.22 | 74.95 ± 0.58 | 42: 78.3039 / 74.1361; 43: 78.1107 / 75.3310; 44: 78.6422 / 75.3887 |
| DMGC | 72.84 ± 1.91 | 69.43 ± 1.93 | 71.49 ± 1.83 | 67.83 ± 1.44 | 42: 72.2880 / 67.3784; 43: 73.2302 / 69.7812; 44: 68.9539 / 66.3306 |
| LGMRec | 80.30 ± 0.10 | 77.19 ± 0.11 | 79.18 ± 0.02 | 76.52 ± 0.13 | 42: 79.1979 / 76.6505; 43: 79.1979 / 76.3344; 44: 79.1496 / 76.5721 |
| MoPF | 80.36 ± 0.09 | 77.46 ± 0.31 | 79.64 ± 0.20 | 76.99 ± 0.20 | 42: 79.7052 / 76.8396; 43: 79.3670 / 76.8561; 44: 79.8502 / 77.2669 |

### Grocery

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Seed-level test values (42 / 43 / 44) |
|---|---:|---:|---:|---:|---|
| MLP | 78.62 ± 0.17 | 71.65 ± 0.55 | 78.86 ± 0.48 | 69.78 ± 1.18 | 42: 78.9165 / 71.0013; 43: 78.2430 / 68.1793; 44: 79.4143 / 70.1716 |
| GCN | 80.61 ± 0.08 | 72.82 ± 0.47 | 81.27 ± 0.12 | 71.95 ± 0.46 | 42: 81.1127 / 72.1402; 43: 81.4056 / 71.3157; 44: 81.2884 / 72.3860 |
| GraphSAGE | 83.41 ± 0.10 | 76.83 ± 0.46 | 82.61 ± 0.23 | 74.35 ± 0.42 | 42: 82.5769 / 74.5521; 43: 82.8990 / 74.7291; 44: 82.3426 / 73.7719 |
| MMGCN | 81.67 ± 0.38 | 73.57 ± 0.89 | 82.00 ± 0.11 | 71.70 ± 1.31 | 42: 82.0498 / 73.3818; 43: 82.1083 / 71.5382; 44: 81.8448 / 70.1792 |
| MGAT | 27.53 ± 2.18 | 13.40 ± 2.93 | 27.43 ± 2.61 | 13.13 ± 2.87 | 42: 28.4627 / 9.8150; 43: 29.9854 / 16.8090; 44: 23.8360 / 12.7804 |
| DiP | 83.61 ± 0.21 | 77.97 ± 0.15 | 83.30 ± 0.26 | 75.69 ± 0.47 | 42: 82.9575 / 75.4473; 43: 83.3382 / 75.2707; 44: 83.6018 / 76.3444 |
| DGF | 80.73 ± 0.09 | 68.85 ± 0.47 | 81.47 ± 0.10 | 69.63 ± 0.11 | 42: 81.6105 / 69.7664; 43: 81.4348 / 69.4984; 44: 81.3763 / 69.6178 |
| DMGC | 73.10 ± 2.27 | 61.00 ± 1.88 | 72.22 ± 2.86 | 60.08 ± 2.84 | 42: 76.0176 / 62.7916; 43: 71.5373 / 61.2817; 44: 69.1069 / 56.1526 |
| LGMRec | 83.18 ± 0.11 | 77.24 ± 0.30 | 82.88 ± 0.08 | 75.33 ± 0.21 | 42: 82.9868 / 75.3550; 43: 82.8697 / 75.0597; 44: 82.7818 / 75.5840 |
| MoPF | 83.61 ± 0.12 | 77.49 ± 0.42 | 83.42 ± 0.30 | 76.18 ± 0.44 | 42: 83.2211 / 76.3594; 43: 83.1918 / 75.5697; 44: 83.8360 / 76.6080 |

### ele-fashion

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Seed-level test values (42 / 43 / 44) |
|---|---:|---:|---:|---:|---|
| MLP | 87.71 ± 0.04 | 75.91 ± 0.32 | 87.92 ± 0.04 | 76.81 ± 0.31 | 42: 87.8725 / 77.2455; 43: 87.9304 / 76.5292; 44: 87.9680 / 76.6552 |
| GCN | 85.42 ± 0.09 | 72.18 ± 0.48 | 85.21 ± 0.10 | 73.02 ± 0.23 | 42: 85.2915 / 73.3350; 43: 85.0699 / 72.9455; 44: 85.2574 / 72.7783 |
| GraphSAGE | 87.70 ± 0.07 | 75.72 ± 0.81 | 87.75 ± 0.26 | 76.55 ± 0.93 | 42: 87.4872 / 75.3529; 43: 88.0975 / 77.6093; 44: 87.6577 / 76.6748 |
| MMGCN | 87.65 ± 0.03 | 75.71 ± 0.52 | 87.55 ± 0.05 | 76.30 ± 0.30 | 42: 87.6031 / 76.7226; 43: 87.4770 / 76.0550; 44: 87.5759 / 76.1314 |
| MGAT | 72.85 ± 4.72 | 35.23 ± 7.68 | 72.58 ± 4.62 | 35.42 ± 7.72 | 42: 76.9349 / 39.1881; 43: 66.1780 / 24.6609; 44: 74.6164 / 42.4080 |
| DiP | 88.05 ± 0.05 | 76.24 ± 1.05 | 87.99 ± 0.28 | 77.13 ± 1.31 | 42: 87.6747 / 76.2892; 43: 88.3635 / 78.9805; 44: 87.9373 / 76.1077 |
| DGF | 86.88 ± 0.05 | 71.91 ± 0.28 | 86.86 ± 0.04 | 72.63 ± 0.43 | 42: 86.9110 / 73.0911; 43: 86.8224 / 72.7495; 44: 86.8326 / 72.0504 |
| DMGC | 86.50 ± 0.23 | 67.20 ± 4.99 | 86.42 ± 0.25 | 67.66 ± 4.52 | 42: 86.5905 / 70.1726; 43: 86.0586 / 61.3143; 44: 86.6008 / 71.4790 |
| LGMRec | 85.82 ± 0.18 | 70.40 ± 1.05 | 85.68 ± 0.31 | 71.29 ± 1.52 | 42: 85.8916 / 72.5329; 43: 85.2336 / 69.1433; 44: 85.9052 / 72.1797 |
| MoPF | 88.04 ± 0.03 | 76.50 ± 0.34 | 88.03 ± 0.10 | 76.99 ± 0.65 | 42: 87.8895 / 76.2390; 43: 88.1146 / 76.9210; 44: 88.0873 / 77.8157 |

### Reddit-S

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Seed-level test values (42 / 43 / 44) |
|---|---:|---:|---:|---:|---|
| MLP | 92.72 ± 0.20 | 87.63 ± 0.27 | 92.82 ± 0.01 | 87.28 ± 0.03 | 42: 92.8279 / 87.2358; 43: 92.7965 / 87.3061; 44: 92.8279 / 87.2900 |
| GCN | 94.26 ± 0.13 | 90.43 ± 0.17 | 93.52 ± 0.08 | 88.88 ± 0.20 | 42: 93.6143 / 89.1620; 43: 93.5200 / 88.6900; 44: 93.4256 / 88.8005 |
| GraphSAGE | 95.31 ± 0.07 | 91.76 ± 0.05 | 94.84 ± 0.05 | 90.45 ± 0.17 | 42: 94.9041 / 90.6606; 43: 94.7782 / 90.2502; 44: 94.8411 / 90.4334 |
| MMGCN | 95.86 ± 0.26 | 92.40 ± 0.52 | 95.67 ± 0.15 | 91.25 ± 0.39 | 42: 95.5646 / 91.0663; 43: 95.8792 / 91.7877; 44: 95.5646 / 90.8860 |
| MGAT | 77.73 ± 5.55 | 67.55 ± 6.21 | 77.90 ± 6.50 | 67.67 ± 7.03 | 42: 85.1840 / 75.7852; 43: 79.1129 / 68.5950; 44: 69.3929 / 58.6281 |
| DiP | 96.23 ± 0.05 | 92.73 ± 0.15 | 96.43 ± 0.05 | 92.42 ± 0.38 | 42: 96.3825 / 91.9067; 43: 96.4140 / 92.5810; 44: 96.5083 / 92.7869 |
| DGF | 96.06 ± 0.12 | 91.48 ± 0.31 | 95.98 ± 0.08 | 90.57 ± 0.55 | 42: 95.8792 / 90.2038; 43: 96.0679 / 91.3388; 44: 96.0050 / 90.1531 |
| DMGC | 93.03 ± 0.19 | 88.41 ± 0.19 | 92.79 ± 0.31 | 86.83 ± 0.32 | 42: 93.0481 / 86.9952; 43: 92.9538 / 87.1066; 44: 92.3561 / 86.3877 |
| LGMRec | 95.91 ± 0.13 | 92.43 ± 0.35 | 95.99 ± 0.04 | 91.89 ± 0.13 | 42: 95.9421 / 91.7103; 43: 96.0365 / 92.0368; 44: 96.0050 / 91.9154 |
| MoPF | 96.21 ± 0.06 | 93.02 ± 0.03 | 96.37 ± 0.26 | 92.11 ± 0.53 | 42: 96.0050 / 91.3598; 43: 96.5398 / 92.5420; 44: 96.5712 / 92.4136 |

## Best and second-best

Ranking uses the mean test value among the ten formal models only; standard deviations do not affect ranking.

| Dataset | Metric | Best | Second-best |
|---|---|---|---|
| Movies | Test Accuracy | MoPF (56.1919) | LGMRec (55.5922) |
| Movies | Test Macro F1 | MoPF (49.4855) | LGMRec (49.1993) |
| Toys | Test Accuracy | MoPF (79.6408) | DiP (79.1818) |
| Toys | Test Macro F1 | MoPF (76.9875) | DiP (76.6095) |
| Grocery | Test Accuracy | MoPF (83.4163) | DiP (83.2992) |
| Grocery | Test Macro F1 | MoPF (76.1790) | DiP (75.6875) |
| ele-fashion | Test Accuracy | MoPF (88.0305) | DiP (87.9918) |
| ele-fashion | Test Macro F1 | DiP (77.1258) | MoPF (76.9919) |
| Reddit-S | Test Accuracy | DiP (96.4349) | MoPF (96.3720) |
| Reddit-S | Test Macro F1 | DiP (92.4249) | MoPF (92.1051) |

## MoPF paper-facing result

MoPF is best on **7 / 10** dataset × metric cells and in the top two on **10 / 10** cells (additional second-place cells: 3).

| Dataset | Test Accuracy | Test Macro-F1 |
|---|---:|---:|
| Movies | 56.19 ± 0.52 | 49.49 ± 0.71 |
| Toys | 79.64 ± 0.20 | 76.99 ± 0.20 |
| Grocery | 83.42 ± 0.30 | 76.18 ± 0.44 |
| ele-fashion | 88.03 ± 0.10 | 76.99 ± 0.65 |
| Reddit-S | 96.37 ± 0.26 | 92.11 ± 0.53 |

## Validity and exceptions

- Expected jobs: `150`; completed records: `150`.
- All recorded health/provenance checks passed: `True`.
- Failed attempts: `[]`; rerun jobs: `[]`.
- High-variance flag threshold: population standard deviation > 3.00 percentage points; flagged cells: `[{'dataset': 'Movies', 'model': 'sage', 'metric': 'test_macro_f1', 'population_std_percentage_points': 10.838922427911731}, {'dataset': 'Movies', 'model': 'mgat', 'metric': 'test_accuracy', 'population_std_percentage_points': 4.6009808664158065}, {'dataset': 'Movies', 'model': 'dgf', 'metric': 'test_macro_f1', 'population_std_percentage_points': 3.1910854019695485}, {'dataset': 'Toys', 'model': 'mgat', 'metric': 'test_accuracy', 'population_std_percentage_points': 7.814718730323128}, {'dataset': 'Toys', 'model': 'mgat', 'metric': 'test_macro_f1', 'population_std_percentage_points': 6.521269905923122}, {'dataset': 'ele-fashion', 'model': 'mgat', 'metric': 'test_accuracy', 'population_std_percentage_points': 4.622333732524857}, {'dataset': 'ele-fashion', 'model': 'mgat', 'metric': 'test_macro_f1', 'population_std_percentage_points': 7.719865776271193}, {'dataset': 'ele-fashion', 'model': 'dmgc', 'metric': 'test_macro_f1', 'population_std_percentage_points': 4.515383311675093}, {'dataset': 'Reddit-S', 'model': 'mgat', 'metric': 'test_accuracy', 'population_std_percentage_points': 6.503820213571817}, {'dataset': 'Reddit-S', 'model': 'mgat', 'metric': 'test_macro_f1', 'population_std_percentage_points': 7.034866536098816}]`.
- Incomplete records at report time: `[]`.
- Machine-readable outputs: `/hdd1/DataInHere/YHF/MoPF/outputs/paper_nc_final_fixed_v1`.
