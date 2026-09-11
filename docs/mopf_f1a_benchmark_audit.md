# MoPF F1-A Benchmark Matrix and Result-Reuse Audit

Status: **COMPLETE — no training or benchmark execution performed**

This audit freezes the eligibility and reuse boundary before F1-B. The source
of truth is method-freeze SHA
`4ddbd6918ceebadc25eed2694e1b463f9aac87f4`, branch `vnext`, with formal
configuration SHA256
`1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a`.
`git pull --ff-only origin vnext` was up to date and the worktree was clean at
audit capture. No model, training, evaluator, or split source was changed.

## Formal scope

- NC: Movies (`K=3`), Toys (`K=3`), Grocery (`K=2`), ele-fashion (`K=3`),
  and Reddit-S (`K=3`).
- LP: sports-copurchase under `unified_sampled_lp_v1`.
- Seeds: `42, 43, 44`.
- Final MoPF: `learned_diag_cos`, temperature `0.35`, anchored cumulative
  propagation, alpha `0.1`, TCPR enabled.
- Quasi-held-out backlog: books-lp, cloth-copurchase, and every other
  unregistered dataset/task. None was run in F1-A.

## Implemented model inventory and roles

All 16 files in `configs/model/*.yaml` and their corresponding source modules
were enumerated. Constructor-only checks passed for every config on all five
formal NC feature shapes and the sports-copurchase LP feature shape.

| Config | Role | Formal treatment | Native-setting note |
|---|---|---|---|
| `mlp` | `formal_external_baseline` | Formal | hidden 256, 2 layers, dropout 0.5, no graph |
| `gcn` | `formal_external_baseline` | Formal | hidden 256, 3 layers, dropout 0.2 |
| `sage` | `formal_external_baseline` | Formal | hidden 256, 3 layers, dropout 0.2 |
| `mmgcn` | `formal_external_baseline` | Formal | hidden 128, 2 layers, node-id branch |
| `mgat` | `formal_external_baseline` | Formal | hidden 128, 1 layer, node-id branch |
| `dip` | `formal_external_baseline` | Formal | `d_model=q_dim=256`, `n_q=8`, `mp_hops=3` |
| `dgf` | `formal_external_baseline` | Formal | hidden 64, 10 layers, native lr/weight decay |
| `dmgc` | `formal_external_baseline` | Formal | hidden 128, 1 layer, native lr/weight decay |
| `lgmrec` | `formal_external_baseline` | Formal | hidden 128, 3 layers, `hyper_num=64` |
| `mopf` | `final_MoPF` | Formal | frozen U1 + U2-C1 + U3-B1 |
| `map_mag` | `historical_internal_model` | Internal only | predecessor/preset |
| `map_mag_v1` | `historical_internal_model` | Internal only | predecessor/preset |
| `map_mag_v2` | `historical_internal_model` | Internal only | predecessor/preset |
| `map_mag_v3` | `historical_internal_model` | Internal only | predecessor/preset |
| `map_mag_v3_full` | `historical_internal_model` | Internal only | predecessor/preset |
| `map_mag_v3_lp` | `historical_internal_model` | Internal only | predecessor/preset |

No `task-specific_baseline` or `unsupported_under_frozen_protocol` model was
found. The MAP family is not automatically promoted merely because its config
exists or because an older table contained one of its presets.

## Eligibility audit

For the nine external baselines, every NC dataset is
`eligible_with_documented_native_setting`; final MoPF is `eligible` on every
NC dataset. These classifications share the frozen multimodal features,
repository split, seed set, full-graph NC path, fixed-label evaluator,
validation-accuracy checkpoint selection, and descriptive test policy. Native
model dimensions, depth, dropout, normalization, and explicitly documented
model-level optimizer presets remain in each model config.

For LP, the same nine baselines and final MoPF all implement the required
forward path and are constructor-compatible with sports-copurchase. They are
eligible under the shared `unified_sampled_lp_v1` settings: train-only message
graph, filtered negatives, `[5,5]` neighbor sampling, shared 128-dimensional
Hadamard MLP decoder, full-graph exact evaluation, validation-MRR selection,
and seeds 42/43/44. A model's native full-graph preference is overridden by
the frozen sampled LP task protocol where applicable.

All six MAP configs are `not_fairly_comparable` for this formal table because
they are historical/internal predecessors rather than registered external
baselines. The complete per-model × dataset records are in
`outputs/f1_formal_benchmark/f1a_benchmark_audit.json`; the compact matrix is
`outputs/f1_formal_benchmark/f1_benchmark_matrix.csv`.

## Result reuse audit

The inventory contains all **361 config-backed `results.json` files** found
under `outputs/`; each row records source, config hash, split path/hash when
available, protocol, selection rule, metrics, and reuse decision. Unknown
producing SHA or evaluator provenance is never silently reused.

- **15 seed-level runs** from U3-B1 (five NC datasets × three seeds) are
  `REUSE_EXACT`. Their configs match the formal config, and U1-freeze,
  U2-C1, and U3-B1 checkpoint compatibility was independently confirmed by
  strict model+head loading.
- The old `outputs/lp_benchmark/` result set has 11 aggregate result files.
  The nine external-baseline files and the old MoPF LP file are
  `RERUN_REQUIRED`: the source SHA/evaluator provenance is incomplete, and
  the old MoPF config is pre-U3-B1. The MAP LP result is internal-only.
- Earlier MoPF NC/U1/U2 and diagnostic outputs are retained as provenance or
  marked as superseded; none can replace the final U3-B1 result.

The full source inventory is
`outputs/f1_formal_benchmark/f1_reuse_inventory.csv`.

## Exact job manifest

The formal matrix has 10 registered models × 6 task/dataset cells × 3 seeds =
180 seed-level jobs. The manifest additionally exposes the six internal MAP
configs for a complete 288-row audit boundary:

| Status | Seed-level jobs | Meaning |
|---|---:|---|
| `REUSE_EXACT` | 15 | Existing final U3-B1 NC runs |
| `RERUN_REQUIRED` | 165 | 135 external NC + 27 external LP + 3 final MoPF LP |
| `NOT_ELIGIBLE` | 108 | Six internal MAP configs × six cells × three seeds |

The authoritative manifest is
`outputs/f1_formal_benchmark/f1_job_manifest.csv`. Rerun jobs prefer
`cuda:0` for NC and `cuda:1` for LP; this is a scheduling preference only and
does not alter the protocol.

## Tag and gate

`mopf-vnext-final` does not exist. Only the following recommendation is
recorded; no tag was created or pushed:

```bash
git tag -a mopf-vnext-final 4ddbd6918ceebadc25eed2694e1b463f9aac87f4 -m "Freeze MoPF-vNext final architecture and evaluation protocol"
```

F1-B gate: **PASS**. The freeze SHA is verified, formal datasets and K are
locked, baseline eligibility is resolved, all 361 discovered result files are
classified, no ambiguous reuse remains, and the exact rerun manifest exists.
F1-A now hard-stops: no training, benchmark execution, F2 ablation, tuning, or
architecture modification was performed.

## Artifacts

- `outputs/f1_formal_benchmark/f1a_benchmark_audit.json`
- `outputs/f1_formal_benchmark/f1_job_manifest.csv`
- `outputs/f1_formal_benchmark/f1_reuse_inventory.csv`
- `outputs/f1_formal_benchmark/f1_benchmark_matrix.csv`
- `docs/mopf_f1_table_schema.md`
