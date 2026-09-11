# F1-B0 frozen benchmark execution guide

This directory contains the execution preparation for MoPF-vNext Final
Evaluation F1-B. It does not change model behavior, task semantics, splits,
the evaluator, or the LP decoder. The formal method freeze is
`4ddbd6918ceebadc25eed2694e1b463f9aac87f4`; the frozen
`configs/model/mopf.yaml` SHA256 is
`1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a`.

Activate the project environment before running a full job:

```bash
conda activate yhf_env
```

## Exact entry points

The shell scripts accept a physical GPU ID as their first argument. They call
the repository's real entry point, `python -m src.main`, once per seed with
`num_runs=1` and independent output directories.

```bash
# 15 fresh final MoPF NC jobs: five datasets × seeds 42/43/44.
bash scripts/f1b/run_mopf_nc.sh 0

# 6 final MoPF LP jobs: sports formal + cloth quasi-held-out.
bash scripts/f1b/run_mopf_lp.sh 1

# 27 cloth baseline LP jobs, split deterministically as 15 + 12.
bash scripts/f1b/run_cloth_baselines_lp.sh 0 shard0
bash scripts/f1b/run_cloth_baselines_lp.sh 1 shard1
```

The external cloth baseline order is fixed and native settings are taken from
the existing model configs:

```text
shard0: mlp, gcn, sage, mmgcn, mgat       (5 × 3 = 15)
shard1: dip, dgf, dmgc, lgmrec            (4 × 3 = 12)
```

To print the complete fresh plan, without launching anything:

```bash
DRY_RUN=1 python scripts/f1b/run_group.py --group all
# or:
python scripts/f1b/run_group.py --group all --dry-run
```

The plan must report exactly 48 new full runs. Individual dry-runs report 15,
6, 15, and 12 entries for the commands above. No free-form Hydra override is
accepted by the launcher, so frozen protocol values cannot be changed from a
shell command.

## MoPF NC reuse versus fresh reevaluation

Option A — recommended scientific reuse: the 15 U3-B1 NC runs already marked
`REUSE_EXACT` by F1-A can be used directly from
`outputs/u3b_transport_conditioned_composition/runs/<dataset>/B1_TCPR/seed<seed>`.
This launches no training:

```bash
python scripts/f1b/run_group.py --group mopf-nc --nc-policy reuse --dry-run
```

Option B — requested final-execution consistency: run the 15 fresh frozen NC
jobs with `run_mopf_nc.sh`. These are reevaluations, not architecture or
hyperparameter selection runs. The later summary must retain both the
original U3-B1 reused values and the fresh values; it must use one declared
policy consistently rather than choosing better numbers.

With exact NC reuse, the new full-run count is 33 instead of 48 (6 MoPF LP +
27 cloth baselines).

## GPU allocation

The launcher never discovers or occupies all GPUs automatically. Pass one
explicit device per process. A practical sequential layout is:

```text
GPU 0: run_mopf_nc.sh 0, then cloth baseline shard0
GPU 1: run_mopf_lp.sh 1, then cloth baseline shard1
```

The two first groups may be started manually in parallel because they use
independent output directories. Do not start two shards on the same GPU. If
using `CUDA_VISIBLE_DEVICES=1`, the visible device is remapped and the script
argument should normally be `0` (or omit the variable and pass physical `1`).
This explicit layout also leaves the GPU choice under the user's control when
moving on after a seed-43 job.

## Output and provenance

Formal outputs are created only below:

```text
outputs/f1_final_execution/
  nc/<dataset>/mopf/seed<seed>/
  lp_formal/sports-copurchase/mopf/seed<seed>/
  lp_quasi_held_out/cloth-copurchase/<model>/seed<seed>/
```

Each completed fresh run contains `resolved_config.yaml`, `command.txt`,
`git_sha.txt`, `seed.txt`, `device.txt`, `provenance.json`, `training.log`,
`best.pt`, `results.json`, `metrics.json`, and `complete.marker`. The launcher
also maintains `run_status.csv` with task, dataset, model, seed, status, exit
code, output directory, and start/end times. `execution_head` is recorded at
launch time; the method freeze SHA remains the immutable F0 SHA above.

An existing run is skipped only when all three conditions hold: a valid
`complete.marker`, valid machine-readable `metrics.json`, and a non-empty
`best.pt`. A partial directory is reported as `INCOMPLETE` and is not silently
skipped. Use `--resume` to rerun it in place, or `--overwrite` to explicitly
rerun it in place without deleting existing files. The current training code
does not implement optimizer-state continuation, so `--resume` means a fresh
same-protocol rerun of that incomplete seed.

## Scope warning

Every cloth result is labeled `quasi-held-out`; cloth is never formal LP and
must never be averaged with sports-copurchase in an unmarked aggregate. Sports
LP is the formal LP table. Cloth is a separate “Quasi-Held-Out LP
Generalization” table.

F1-A audited the existing baseline results as follows: the 135 external NC
cells and 27 external sports-LP cells are `RERUN_REQUIRED`, because complete
source SHA, evaluator, and checkpoint-selection provenance was not available.
They are not silently promoted to formal reuse, and F1-B0 does not rerun them;
only cloth baseline LP is in the requested F1-B execution plan.

## Smoke checks

F1-B0 smoke checks are intentionally short and isolated from formal outputs:

```bash
python scripts/f1b/smoke_f1b.py \
  --nc-device 0 --lp-device 1 --baseline-device 0 --dip-device 1
```

This runs one MoPF NC dataset/seed, one MoPF sports LP seed, one MoPF cloth LP
seed, one simple `mlp` cloth LP seed, each with one epoch and at most one
training batch, plus a DiP cloth constructor/forward check with no training
loop. Artifacts go to `outputs/f1b_smoke/` and are not formal results.

## Completion and summary

Generate the 48-entry plan before executing full jobs, then check completion:

```bash
python scripts/f1b/run_group.py --group all --dry-run
python scripts/f1b/check_f1b_completion.py
```

The checker reads the formal plan and writes
`outputs/f1_final_execution/f1b_completion.json` and
`outputs/f1_final_execution/f1b_completion.csv`. It separately reports MoPF
NC, sports formal LP, and cloth quasi-held-out LP, including expected,
completed/reused, failed, missing, incomplete, and duplicate counts.

Prepare per-seed and mean ± population-standard-deviation tables only after
the relevant runs exist:

```bash
python scripts/f1b/summarize_f1b.py
```

This writes `f1b_summary.json` and `f1b_summary.md` without making a final
conclusion. It verifies the frozen provenance fields, keeps cloth separate,
and records that test metrics are descriptive and never used for selection.
