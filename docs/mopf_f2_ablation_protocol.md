# MoPF F2 formal ablation protocol

## 1. Scientific purpose

F2 isolates the frozen MoPF mechanisms that correspond to the frozen E0
empirical studies.  Each ablation changes only the named mechanism; dataset,
split, seed, model width, optimizer, training budget, checkpoint selection and
evaluation protocol remain those of the frozen Full configuration.

This document defines the runnable protocol and its audit boundaries.  It does
not report F2 performance results: formal F2 training has intentionally not
been launched in this implementation phase.

## 2. Variant definitions

The single source of truth is `src/ablation.py`.  The table below is both the
human-readable audit table and the exact Boolean semantics recorded in every
`ablation_manifest.json`.

| Variant | Learned relation calibration | Anchor | Global | Modality | Node | TCPR |
|---|---:|---:|---:|---:|---:|---:|
| A0 `full` | ON | ON | ON | ON | ON | ON |
| A1 `wo_learned_semantic_calibration` | OFF | ON | ON | ON | ON | ON |
| A2 `wo_semantic_anchor` | ON | OFF | ON | ON | ON | ON |
| A3 `wo_tcpr` | ON | ON | ON | ON | ON | OFF |
| A4 `wo_node_adaptation` | ON | ON | ON | ON | OFF | ON |
| A5 `wo_modality_adaptation` | ON | ON | ON | OFF | ON | ON |
| I1 `wo_anchor_tcpr` | ON | OFF | ON | ON | ON | OFF |
| I2 `wo_node_tcpr` | ON | ON | ON | ON | OFF | OFF |
| I3 `wo_modality_tcpr` | ON | ON | ON | OFF | ON | OFF |

The formal main phase contains A1--A5.  The interaction phase contains I1--I3.
Full is not implicitly retrained by the launcher; it is reused from the
audited F1 formal artifacts when the reuse conditions in Section 6 hold.

## 3. Exact code/config mapping

The frozen Full configuration in `configs/model/mopf.yaml` is:

```text
edge_weight_mode: learned_diag_cos
multihop_state_mode: anchored
multihop_response_mode: cumulative
multihop_anchor_alpha: 0.1
global_filter_trainable: true
use_modality_residual: true
use_node_residual: true
use_transport_residual: true
hidden_dim: 256
edge_weight_temperature: 0.35
```

The top-level `configs/config.yaml` field `ablation: full` is the unified CLI
interface.  A run can select a frozen variant with, for example,
`ablation=wo_tcpr`.

| Mechanism | Source location | Frozen config field/default | Ablation operation |
|---|---|---|---|
| Learned semantic calibration | `MoPF.__init__`; `_semantic_edge_weights` in `src/models/mopf.py` | `model.edge_weight_mode=learned_diag_cos` | A1 resolves the effective mode to `separate_cos`; physical support and modality-specific cosine weighting remain active. |
| Semantic anchor | `effective_multihop_modes` in `src/ablation.py`; `_build_multihop_banks` in `src/models/mopf.py` | `model.multihop_state_mode=anchored`, `model.multihop_anchor_alpha=0.1` | A2 changes only the state bank to `ordinary`; response mode remains `cumulative`, K and downstream aggregation remain unchanged. |
| Global hop preference | `gamma_global` initialization and `_effective_coefficients` in `src/models/mopf.py` | `model.global_filter_trainable=true` | Always ON for A0--I3. No F2 variant removes it. |
| Modality residual | `self.use_modality_residual` and `delta_gamma_text/visual` in `_effective_coefficients` | `model.use_modality_residual=true` | A5/I3 set the effective modality residual to disabled; global and node terms remain. |
| Node residual | `self.use_node_residual` and `_node_residuals` | `model.use_node_residual=true` | A4/I2 set the effective node residual to disabled; global, modality and (where specified) TCPR terms remain. |
| TCPR | `_transport_context`, `theta_transport_text/visual`, and `_effective_coefficients` | `model.use_transport_residual=true` | A3/I1/I2/I3 disable the transport-conditioned term `c_tilde_i^m * beta_k^m`; other coefficient terms remain. |

`src/ablation.py` centralizes the catalog, effective switches and manifest
construction.  `src/models/mopf.py` consumes the resolver for both NC and LP;
there are no variant-specific model classes.  `src/main.py` writes the
manifest, metrics wrapper, resolved config and completion marker.  The LP
checkpoint now records the same `best_val_mrr` selection metadata as NC's
checkpoint path.

## 4. Audit table and invariant checks

Every completed F2 run must contain:

```text
ablation_manifest.json
metrics.json
resolved_config.yaml
train.log
best.pt
complete.marker
```

The manifest contains the variant flags in Section 2 plus effective edge and
multi-hop modes, alpha, K, hidden dimension, learning rate, weight decay,
optimizer, epoch budget, patience, temperature, batch size, neighbor/negative
sampling settings, evaluation metric, checkpoint selection, inference
protocol, dataset, task, seed, split source and git branch/commit.

`scripts/check_f2_ablation_plan.py` prints the Section 2 audit table before
the dataset × variant × seed matrix, so the table can be captured with the
preflight log as a machine-readable audit record.

For every fixed task/dataset/seed context, the plan checker compares the
common protocol fields across variants.  The intended invariant fields are:

```text
dataset split, seed, hidden_dim, optimizer, learning_rate, weight_decay,
max_epochs, patience, K, alpha, temperature, batch_size,
negative_sampling, evaluation_metric, checkpoint_selection,
inference_protocol, protocol_version
```

The only intended effective-mode changes are those in Section 2.  `alpha` is
still recorded for anchor-off runs so the frozen protocol remains auditable;
it is inactive when the anchor is OFF.

## 5. Dataset and seed matrix

Formal NC datasets are `Movies`, `Toys`, `Grocery`, `ele-fashion` and
`Reddit-S`.  Formal LP is `sports-copurchase`.  Seeds are 42, 43 and 44.
`cloth-copurchase` is excluded from F2.

The frozen NC propagation order is K=2 for `Grocery` and K=3 for the other
four NC datasets.  LP uses the formal sampled-LP protocol and K=3.  The
launcher passes the same K to `model.max_order` and `model.num_layers` for NC.

Run counts are:

| Phase | New variants | Dataset-task cells | Seeds | Expected new runs |
|---|---:|---:|---:|---:|
| Main | 5 | 6 | 3 | 90 |
| Interaction | 3 | 6 | 3 | 54 |
| Full reuse candidate | 0 | 6 | 3 | 18 existing artifacts |
| Full rerun contingency | 1 | 6 | 3 | 18 additional runs |

The expected F2 output tree is:

```text
outputs/f2_ablation/
  nc/{Movies,Toys,Grocery,ele-fashion,Reddit-S}/{variant}/seed{42,43,44}/
  lp/sports-copurchase/{variant}/seed{42,43,44}/
```

The launcher does not write into `outputs/f1_final_execution/` or the prior
U3-B output roots.

## 6. Full reuse decision

The current audit supports `full_rerun_required=false` for F2 planning,
subject to the pre-run checks being rerun on the actual server state.  The
evidence is:

1. `outputs/f1_final_execution/f1b_completion.json` reports 48/48 completed
   F1 records with zero failed, missing, incomplete or duplicate records.  The
   formal F2 scope has all 18 required Full artifact directories (15 NC + 3
   sports LP) with `metrics.json`, `best.pt`, `resolved_config.yaml` and
   `complete.marker`.
2. The F1 formal resolved configs use the same frozen MoPF effective settings:
   learned diagonal semantic calibration, anchored cumulative propagation,
   alpha 0.1, Full global/modality/node/TCPR terms, hidden dimension 256, and
   the same task-specific selection metrics (`val_acc` for NC and `val_mrr`
   for LP).  Their split paths and loader schema are the formal ones recorded
   in the configs and logs: seed-specific MAGB NC/LP split files where used,
   and the fixed `node_split_path` for `ele-fashion` NC.
3. `tests/test_f2_ablation.py::test_historical_head_full_path_matches_current_full_path`
   loads the pre-change `HEAD` model state into the current explicit Full
   model and compares the same input/checkpoint path.  Representation and
   auxiliary output max-absolute differences are zero; the same classifier
   gives zero logits max-absolute difference and identical synthetic
   validation/test accuracy.  This is a code-path regression test, not a new
   formal training run.

This decision does not claim that the newly written bookkeeping files existed
in F1; they are generated for new F2 runs.  It establishes that adding the
unified ablation interface does not alter the Full numerical path.  If the
F1 artifact, split, selection or evaluator audit fails, the safe contingency
is to set `full_rerun_required=true` and run the extra 18 Full jobs explicitly;
the provided launcher never does that implicitly.

## 7. Smoke-test results

The smoke matrix was executed with one NC dataset (`Movies`), one LP dataset
(`sports-copurchase`), seed 42, one epoch and one training batch.  It covered
Full plus all five main and three interaction variants: 18 cases total.

```text
status=PASS
total_cases=18
passed_cases=18
failed_cases=0
formal_training_launched=false
interaction_training_launched=false
```

The machine-readable summary is
`outputs/f2_ablation_smoke/smoke_summary.json` and the row-level summary is
`outputs/f2_ablation_smoke/smoke_summary.csv`.  The smoke outputs were checked
for train/backward completion, finite payloads, checkpoint writing, distinct
paths, manifest identity/flags, and copied logs/configs.  Smoke metrics are
not performance evidence.

## 8. Official run commands

Run from the MoPF project root.  First inspect the plan; this command never
launches training:

```bash
python scripts/check_f2_ablation_plan.py --phase main --task all --seeds 42,43,44
python scripts/check_f2_ablation_plan.py --phase interaction --task all --seeds 42,43,44
```

Recommended order on a single GPU is: main NC sequentially, main LP
sequentially, audit all main outputs, then run the interaction preflight and
interaction phase only after the main phase is complete.

Main NC:

```bash
bash scripts/run_f2_ablation.sh \
  --phase main \
  --task nc \
  --datasets Movies,Toys,Grocery,ele-fashion,Reddit-S \
  --seeds 42,43,44 \
  --device cuda:0 \
  --skip-existing
```

Main LP:

```bash
bash scripts/run_f2_ablation.sh \
  --phase main \
  --task lp \
  --datasets sports-copurchase \
  --seeds 42,43,44 \
  --device cuda:1 \
  --skip-existing
```

After the main completeness audit, interaction:

```bash
bash scripts/run_f2_ablation.sh \
  --phase interaction \
  --task all \
  --seeds 42,43,44 \
  --device cuda:0 \
  --skip-existing
```

Dry-run examples:

```bash
bash scripts/run_f2_ablation.sh --phase main --task nc --datasets Movies --seeds 42 --dry-run
bash scripts/run_f2_ablation.sh --phase all --task all --seeds 42 --dry-run
```

The launcher supports `--datasets`, `--seeds`, `--task`, `--device`/`--gpu`,
`--skip-existing`, `--resume`, `--overwrite` and `--dry-run`.  Existing
incomplete directories stop the launcher unless `--resume` or `--overwrite`
is explicit.  No command in this document has been used to launch the formal
90-run or 54-run matrix.

## 9. Expected output structure

Each F2 run is written to a unique
`{task}/{dataset}/{variant}/seed{seed}` directory.  `metrics.json` has a
uniform outer schema containing task, dataset, model, ablation, seed,
selection metric, checkpoint selection, best epoch, runtime, peak GPU memory
when available, checkpoint reference, checkpoint metadata and the task metric
mapping.  NC records validation and test Accuracy/Macro-F1 when test
evaluation is enabled.  LP records Val MRR, Test MRR and Test Hits@1/3/10
when test evaluation is enabled.

## 10. Known caveats and paper-safe interpretation

- The independent experimental unit for F2 summaries is a complete
  dataset-seed run; the planned replication count is three seeds per
  dataset/variant.  `mean ± std` must retain the three seed values and state
  the chosen standard-deviation convention.
- Three seeds are a small replication set.  Report effect sizes and direction
  counts; do not add undefined significance terminology or p values.
- A Full-minus-ablation difference is descriptive.  A negative difference
  means the ablation is higher than Full and must remain visible.  A one-seed
  directional result should be labelled weak evidence; 3/3 same-direction
  results can be labelled strong seed consistency without calling them
  statistically significant.
- Any relation to E0-A/B/C/D is exploratory/descriptive.  A drop aligned with
  discrepancy, drift, heterogeneity or association strength does not by itself
  establish causality or mechanism.
- `--resume` reruns an incomplete directory in place; the current entrypoint
  does not provide optimizer-state continuation.  Use it only when a clean
  restart in the same directory is intended.  `--overwrite` also reuses the
  target path and should be used only after inspecting its contents.
- Smoke uses one epoch and one batch and is only an execution invariant test;
  it cannot support component rankings or paper claims.
- The F1 Full artifacts are eligible for reuse only while their exact split,
  checkpoint selection and evaluator provenance remain unchanged.  Do not
  silently substitute a different checkpoint or remove an inconvenient seed.
- No interaction training is started by this protocol.  Any interaction
  phase requires a fresh preflight and complete main-phase audit first.
