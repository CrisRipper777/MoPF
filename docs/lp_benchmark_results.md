# LP Benchmark Results

## 实验设置

- 数据集：`sports-copurchase`
- 模型：11 个模型（MAP-MAG 仅保留 `map_mag_v3`，另含 `mopf`）
- Seeds：`42, 43, 44`
- LP 协议：`unified_sampled_lp_v1`
- 训练方式：sampled link prediction
- Neighbor sampling：两跳 `[5, 5]`（模型特殊深度由项目 runner 解析）
- Training negative：每条正边 1 个 filtered negative
- LP projection dimension：`128`
- Inference：full-graph exact inference
- 表中数值：百分比；格式为 `mean ± population std`
- Job 状态：11/11 已完成，0 失败，0 未启动
- 原始结果根目录：`../outputs/lp_benchmark/`

普通 baseline 的配置直接对应 `configs/model/` 中的同名 YAML；MAP-MAG 系列仅运行 `map_mag_v3.yaml`，不纳入 v1/v2/full/lp preset。

## 结果总览

| 指标 | 最优模型 | 结果 |
|---|---|---:|
| Val MRR | `mopf` | **40.1393 ± 0.2652** |
| Test MRR | `mopf` | **37.1078 ± 0.2304** |
| Test Hits@1 | `mopf` | **21.4061 ± 0.1400** |
| Test Hits@3 | `mopf` | **42.5823 ± 0.3009** |
| Test Hits@10 | `mmgcn` | **72.4543 ± 0.3007** |

## 详细结果

| Model | Val MRR | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 |
|---|---:|---:|---:|---:|---:|
| mlp | 27.0313 ± 0.3456 | 24.9860 ± 0.3515 | 11.7553 ± 0.3136 | 26.3206 ± 0.3361 | 55.3863 ± 0.7039 |
| gcn | 34.2503 ± 0.1534 | 32.0852 ± 0.1421 | 17.7701 ± 0.1798 | 35.4431 ± 0.0548 | 64.1105 ± 0.2315 |
| sage | 35.1167 ± 0.2361 | 32.5921 ± 0.2369 | 17.1010 ± 0.1557 | 36.4062 ± 0.2627 | 69.0535 ± 0.4137 |
| mmgcn | 37.6103 ± 0.2629 | 34.6295 ± 0.2975 | 18.4223 ± 0.3270 | 39.4854 ± 0.2589 | **72.4543 ± 0.3007** |
| mgat | 35.3487 ± 0.1798 | 32.6118 ± 0.1141 | 17.1660 ± 0.0504 | 36.2895 ± 0.2576 | 69.1132 ± 0.6134 |
| dip | 38.1210 ± 0.1706 | 35.1427 ± 0.2141 | 18.9025 ± 0.3937 | 40.2855 ± 0.3614 | 72.2806 ± 0.5189 |
| dgf | 37.3452 ± 0.4084 | 34.1880 ± 0.3075 | 18.5746 ± 0.2540 | 38.5053 ± 0.4546 | 70.4372 ± 0.4431 |
| dmgc | 35.6925 ± 1.7976 | 32.9330 ± 1.7656 | 17.5420 ± 1.5683 | 37.3346 ± 2.4270 | 68.0503 ± 2.3289 |
| lgmrec | 35.6221 ± 0.0433 | 32.9235 ± 0.0323 | 18.1630 ± 0.0832 | 36.7074 ± 0.1397 | 66.2515 ± 0.0874 |
| map_mag_v3 | 39.8790 ± 0.2599 | 36.8362 ± 0.2255 | 20.9642 ± 0.1827 | 42.4683 ± 0.3127 | 71.8583 ± 0.4453 |
| mopf | **40.1393 ± 0.2652** | **37.1078 ± 0.2304** | **21.4061 ± 0.1400** | **42.5823 ± 0.3009** | 72.1888 ± 0.4717 |

## 结果路径

- `mlp`：[../outputs/lp_benchmark/sports-copurchase/mlp/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/mlp/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/mlp/seed42_runs3/launcher.log)
- `gcn`：[../outputs/lp_benchmark/sports-copurchase/gcn/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/gcn/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/gcn/seed42_runs3/launcher.log)
- `sage`：[../outputs/lp_benchmark/sports-copurchase/sage/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/sage/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/sage/seed42_runs3/launcher.log)
- `mmgcn`：[../outputs/lp_benchmark/sports-copurchase/mmgcn/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/mmgcn/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/mmgcn/seed42_runs3/launcher.log)
- `mgat`：[../outputs/lp_benchmark/sports-copurchase/mgat/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/mgat/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/mgat/seed42_runs3/launcher.log)
- `dip`：[../outputs/lp_benchmark/sports-copurchase/dip/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/dip/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/dip/seed42_runs3/launcher.log)
- `dgf`：[../outputs/lp_benchmark/sports-copurchase/dgf/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/dgf/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/dgf/seed42_runs3/launcher.log)
- `dmgc`：[../outputs/lp_benchmark/sports-copurchase/dmgc/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/dmgc/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/dmgc/seed42_runs3/launcher.log)
- `lgmrec`：[../outputs/lp_benchmark/sports-copurchase/lgmrec/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/lgmrec/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/lgmrec/seed42_runs3/launcher.log)
- `map_mag_v3`：[../outputs/lp_benchmark/sports-copurchase/map_mag_v3/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/map_mag_v3/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/map_mag_v3/seed42_runs3/launcher.log)
- `mopf`：[../outputs/lp_benchmark/sports-copurchase/mopf/seed42_runs3/results.json](../outputs/lp_benchmark/sports-copurchase/mopf/seed42_runs3/results.json)；[launcher.log](../outputs/lp_benchmark/sports-copurchase/mopf/seed42_runs3/launcher.log)

## 运行说明

本报告由 `scripts/run_sports_copurchase_lp_benchmark.py` 自动生成。若有失败 job，表中保留 `—`，请先检查对应 `launcher.log`，修复后重新执行同一命令即可续跑并刷新本报告。
