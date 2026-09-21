# CoSI-MAG Final NC + LP Benchmark Launcher

本文档记录 `cosi_mag_final` 冻结配置下的正式 NC/LP benchmark 启动方式。launcher 每个 dataset 调用一次 `python -m src.main`，通过一次 `num_runs=3` 完成 seeds 42、43、44。不重写训练逻辑，也不覆盖此前逐 seed 运行的结果目录。

此前的 `.../<dataset>/seed_42`、`seed_43`、`seed_44` 目录保持原样；它们是分开的主程序调用，不属于这次共享 seed-42 split 的三-run 结果组，分析时不要与新目录合并。

## Benchmark 范围

| Suite | Task | Datasets | Run seeds | `src.main` calls | Training runs |
| --- | --- | --- | --- | ---: | ---: |
| NC | `nc` | Movies、Toys、Grocery、ele-fashion、Reddit-S | 42、43、44 | 5 | 15 |
| Sports LP | `lp` | sports-copurchase | 42、43、44 | 1 | 3 |
| Cloth LP | `lp` | cloth-copurchase | 42、43、44 | 1 | 3 |
| **Total** |  |  |  | **7** | **21** |

正式模型统一为 `model=cosi_mag_final`，每个 dataset 进程使用 `seed=42`、`num_runs=3` 和 `ablation=full`。`src.main` 用 base seed 42 加载数据和 split 一次；NC/LP runner 随后按 `cfg.seed + run_id` 训练 seeds 42、43、44。因此三个训练 seed 共用 seed 42 对应的 split，run seed 只控制初始化、dropout 和训练采样。

launcher 保留 `--seeds` 参数用于明确记录 seed 集合，但冻结 benchmark 只接受 `42 43 44`。launcher 只显式设置 dataset、task、model、seed、num_runs、device、ablation、checkpoint 路径和确定性 Hydra 输出目录；不会覆盖冻结 task protocol 或训练参数。

## 冻结配置审计

NC 使用 `unified_full_graph_nc_v1`，LP 使用 `unified_sampled_lp_v1`。每次启动前，launcher 校验配置中的 protocol 与 final model 核心值，并打印、保存包含 Git branch/commit、三个配置文件 SHA256、suite、dataset、task、base/split seed 42、run seeds 42–44、device 和 final model 配置的 manifest。suite manifest 位于 `outputs/cosi_mag_final_benchmark/manifests/`，每个 dataset 运行组另存 `benchmark_manifest.json`。

| Final model key | Value |
| --- | ---: |
| `hidden_dim` | 256 |
| `max_order` | 3 |
| `multihop_anchor_alpha` | 0.1 |
| `edge_weight_min` | 0.1 |
| `edge_weight_temperature` | 0.35 |
| `filter_rank` | 4 |
| `hop_interaction_layers` | 1 |
| `hop_interaction_heads` | 1 |
| `relation_bias_init` | 0.10 |

## 从仓库根目录运行的三条正式命令

先用下列 dry-run 命令检查路径与 Hydra overrides。dry-run 不会启动 `src.main`；期望分别打印 5、1、1 条命令。

```bash
python scripts/run_cosi_mag_final_benchmark.py --suite nc --device cuda:0 --seeds 42 43 44 --dry-run
python scripts/run_cosi_mag_final_benchmark.py --suite sports-lp --device cuda:0 --seeds 42 43 44 --dry-run
python scripts/run_cosi_mag_final_benchmark.py --suite cloth-lp --device cuda:0 --seeds 42 43 44 --dry-run
```

正式训练由用户手动启动，下面是三条正式命令：

```bash
python scripts/run_cosi_mag_final_benchmark.py --suite nc --device cuda:0 --seeds 42 43 44 --resume
python scripts/run_cosi_mag_final_benchmark.py --suite sports-lp --device cuda:0 --seeds 42 43 44 --resume
python scripts/run_cosi_mag_final_benchmark.py --suite cloth-lp --device cuda:0 --seeds 42 43 44 --resume
```

如需使用其他设备，可替换 `--device`，例如 `cuda:1`。默认 seeds 为 `42 43 44`。不使用 tmux；每条命令由用户自行选择前台或后台运行方式。

## 输出目录

```text
outputs/cosi_mag_final_benchmark/
  manifests/
    nc.json
    sports-lp.json
    cloth-lp.json
  nc/<dataset>/runs_42_43_44/
  lp/<dataset>/runs_42_43_44/
```

每个 dataset 运行组由 `src.main` 写入 `main.log`、`train.log`、`results.json`、`metrics.json`、`resolved_config.yaml`、`resolved_config.json`、`ablation_manifest.json` 和 `complete.marker`。三个独立 checkpoint 为 `best_run1.pt`、`best_run2.pt`、`best_run3.pt`，其 metadata seeds 分别为 42、43、44。`best.pt` 是指向 `best_run3.pt` 的相对符号链接，用于兼容 `src.main` 当前的 completion-marker 检查。

## Resume 策略

`--resume` 仅在 `complete.marker`、`metrics.json`、`results.json`、`best.pt`、`resolved_config.json` 和三个 run checkpoint 同时存在且审计通过时跳过整个 dataset 运行组。metrics/resolved config 必须匹配 `cosi_mag_final`、task、dataset、base seed 42 和 `num_runs=3`；三个 checkpoint 的 metadata seeds 必须依次为 42、43、44。冻结模型值、task protocol，以及日志中是否出现 NaN/Inf train loss 或 validation metric 也会核验。目录存在但不完整或配置不匹配时，launcher 会发出 warning，保留原目录并重新运行该组的全部三个 seed；不会删除目录。

正常完成的单元不会因为 Test 指标较低而自动重跑。允许用户重新运行的原因限于 process crash、NaN/Inf、缺失输出、损坏 checkpoint 或 protocol/config mismatch。launcher 不按测试分数重试或筛选 seed。

`src/models/cosi_mag_final.py` 导入时会过滤 PyTorch 2.4.0 在 activation checkpoint 反向重算中产生的 `torch.cpu.amp.autocast` 弃用告警。因此普通 `python src/main.py ...` 和 launcher 启动都适用。过滤只匹配这条 `FutureWarning`，不会隐藏其他 warning，也不改变训练计算。

## Dry-run 验收

dry-run 期望启动命令数：NC 5 条、Sports LP 1 条、Cloth LP 1 条，共 7 个进程；每个进程执行 3 runs，因此正式训练总数仍为 21 runs。正式 benchmark summary 留待全部实验由用户运行完成后再生成。
