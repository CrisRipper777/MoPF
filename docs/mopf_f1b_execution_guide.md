# MoPF-vNext F1-B0 Final Benchmark Execution Guide

Status: **PREPARED — no formal full training executed in F1-B0**

This guide freezes the execution boundary after F1-A. The method-freeze SHA is
`4ddbd6918ceebadc25eed2694e1b463f9aac87f4`, and the frozen
`configs/model/mopf.yaml` SHA256 is
`1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a`.
The scripts do not alter `src/models`, training semantics, evaluation
semantics, splits, the LP decoder, or the evaluator.

## 1. Exact scripts and commands

Run from the project root with `yhf_env` active:

```bash
conda activate yhf_env

# Fresh frozen MoPF NC: 15 jobs.
bash scripts/f1b/run_mopf_nc.sh 0

# MoPF LP: 3 formal sports jobs + 3 quasi-held-out cloth jobs.
bash scripts/f1b/run_mopf_lp.sh 1

# External baseline cloth LP: 27 jobs in two deterministic shards.
bash scripts/f1b/run_cloth_baselines_lp.sh 0 shard0
bash scripts/f1b/run_cloth_baselines_lp.sh 1 shard1
```

The scripts call the actual repository entry point once per seed:

```text
python -m src.main dataset=<dataset> task=<nc|lp> model=<model>
  seed=<42|43|44> num_runs=1 device=<explicit device>
  hydra.run.dir=outputs/f1_final_execution/<independent run directory>
```

The final MoPF NC command includes only the preregistered dataset order
override (`K=3`, except Grocery `K=2`, with `model.num_layers=K`). All other
frozen values come from the checked SHA-pinned config. Baselines retain their
documented native dimensions, depth, dropout, and model-level optimizer
presets while sharing the frozen task protocol.

## 2. GPU allocation

GPU selection is explicit and no launcher automatically occupies both cards.
The recommended manual layout is:

| Device | Work |
|---|---|
| `cuda:0` | MoPF NC, then baseline cloth `shard0` |
| `cuda:1` | MoPF LP, then baseline cloth `shard1` |

The first two groups may be started manually in parallel. Shards should not
share a GPU. If `CUDA_VISIBLE_DEVICES=1` is used, it remaps the visible card,
so use script device `0` inside that environment. Otherwise pass the physical
device directly, for example `run_mopf_lp.sh 1`.

## 3. Expected counts and scope

| Group | Count | Scope |
|---|---:|---|
| MoPF NC | 15 | formal; Movies/Toys/Grocery/ele-fashion/Reddit-S × 42/43/44 |
| MoPF sports LP | 3 | formal `unified_sampled_lp_v1` |
| MoPF cloth LP | 3 | quasi-held-out only |
| External cloth baseline LP | 27 | quasi-held-out only; 9 models × 3 seeds |
| **Fresh total** | **48** | — |

The exact baseline shard membership is fixed: `shard0 = mlp, gcn, sage,
mmgcn, mgat` (15 jobs) and `shard1 = dip, dgf, dmgc, lgmrec` (12 jobs).

The prior F1-A baseline audit found that the 135 external baseline NC cells
and 27 external baseline sports-LP cells are `RERUN_REQUIRED`, not formally
reusable, because complete source SHA/evaluator/checkpoint-selection
provenance was unavailable. F1-B0 does not rerun those cells. The existing 15
U3-B1 MoPF NC cells are `REUSE_EXACT`.

## 4. MoPF NC: reuse and fresh policies

The recommended scientific policy is Option A: reuse the exact U3-B1 NC
results already audited by F1-A from:

```text
outputs/u3b_transport_conditioned_composition/runs/<dataset>/B1_TCPR/seed<seed>
```

The reuse plan launches no training:

```bash
python scripts/f1b/run_group.py --group mopf-nc --nc-policy reuse --dry-run
```

Option B is the requested final-execution consistency rerun:

```bash
bash scripts/f1b/run_mopf_nc.sh 0
```

Fresh NC is not scientifically necessary because F1-A already certified the
15 exact U3-B1 runs. It is operationally useful if one wants all final
outputs, logs, and per-run provenance under one F1-B root. Fresh results must
not be used to reselect architecture or hyperparameters. Any final report
must declare one policy consistently and retain the original-vs-fresh
difference audit rather than choosing the more favorable values.

With NC reuse, the number of new full runs is 33 (`6 + 27`), not 48.

## 5. Output paths and per-run files

Formal output root:

```text
outputs/f1_final_execution/
├── nc/<dataset>/mopf/seed<seed>/
├── lp_formal/sports-copurchase/mopf/seed<seed>/
└── lp_quasi_held_out/cloth-copurchase/<model>/seed<seed>/
```

Each fresh completed run must contain:

```text
resolved_config.yaml  command.txt       git_sha.txt
seed.txt              device.txt        formal_config_sha256.txt
provenance.json       training.log      launcher.log
best.pt               results.json      metrics.json
complete.marker
```

`run_status.csv` at the root records `task`, `dataset`, `model`, `seed`,
`status`, `exit_code`, `output_dir`, `start_time`, and `end_time` for each
attempt. Every run stores the method-freeze SHA, execution HEAD, formal config
SHA256, task, dataset, model, seed, scope, and exact resolved command.

## 6. Resume and failure behavior

`SKIP_COMPLETE` is allowed only when `complete.marker`, valid machine-readable
`metrics.json`, and a non-empty `best.pt` all exist. Partial logs or a bare
`results.json` do not count. A partial run is marked `INCOMPLETE` and requires
one of:

```bash
# Same frozen command, rerun in place. No optimizer-state continuation exists.
bash scripts/f1b/run_mopf_nc.sh 0 --resume

# Explicit in-place rerun without deleting existing files.
bash scripts/f1b/run_mopf_nc.sh 0 --overwrite
```

Each run has an independent output directory and launcher log. A failing run
therefore does not overwrite or cancel the results of other completed runs.

## 7. Smoke test boundary

No complete F1-B budget is authorized in this preparation phase. The required
short smoke suite is:

```bash
python scripts/f1b/smoke_f1b.py \
  --nc-device 0 --lp-device 1 --baseline-device 0 --dip-device 1
```

It runs one MoPF NC dataset/seed, one MoPF sports LP seed, one MoPF cloth LP
seed, and one simple MLP cloth LP seed with one epoch and at most one
training batch. It also constructs and forwards DiP on cloth LP without
entering a training loop. All outputs are isolated under `outputs/f1b_smoke/`.

## 8. Completion check

Prepare the authoritative fresh 48-entry plan without training:

```bash
python scripts/f1b/run_group.py --group all --dry-run
```

Then run:

```bash
python scripts/f1b/check_f1b_completion.py
```

The checker reads `outputs/f1_final_execution/f1b_plan.csv` and writes:

```text
outputs/f1_final_execution/f1b_completion.json
outputs/f1_final_execution/f1b_completion.csv
```

It reports expected, completed/reused, failed, missing, incomplete, and
duplicate runs separately for MoPF NC, sports formal LP, and cloth
quasi-held-out LP.

## 9. Summary preparation

After the relevant jobs have completed:

```bash
python scripts/f1b/summarize_f1b.py
```

This produces `f1b_summary.json` and `f1b_summary.md` with per-seed metrics,
mean ± population standard deviation, formal sports tables, quasi-held-out
cloth tables, MoPF NC tables, baseline rows, provenance checks, and the
validation-only selection guard. It intentionally does not write a final
experimental conclusion. Cloth must remain separate from formal sports LP in
all later reporting.
