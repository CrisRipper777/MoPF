# CoSI-MAG Robustness Protocol (Figure 5)

## Scope and scientific questions

Figure 5 is a checkpoint-only stress test of the inference-time message graph. It asks whether clean-trained models retain test performance as the message graph receives (i) generic false physical relations and (ii) nonedge relations that are supported by one raw modality and opposed by the other. No model is retrained, fine-tuned, optimized, or selected using perturbed test performance. Model parameters and the trained classifier stay fixed; only `edge_index` changes.

The fixed datasets are Movies (`K=3`) and Grocery (`K=2`), with model seeds 42, 43 and 44. The two perturbation regimes are intentionally separate: random structural noise probes generic structural unreliability, whereas semantic-conflict injection creates a controlled cross-modal discrepancy that directly stresses modality-aware relation calibration.

## Checkpoint families

| Model in plan | Checkpoint source | Selection and evaluation evidence |
|---|---|---|
| Full CoSI-MAG (`mopf`) | `outputs/paper_nc_final_fixed_v1/{Movies,Grocery}/mopf/seed{42,43,44}/best.pt` | User-frozen Figure 4 family; NC, `unified_full_graph_nc_v1`, full-graph training/inference, best validation accuracy, fixed NC split, saved model and head states, passing runner health checks. |
| `wo_relation_calibration` | `outputs/core_story_ablation/nc/{Movies,Grocery}/wo_relation_calibration/seed{42,43,44}/best.pt` | Formal Core Story outputs; manifests mark relation calibration ineffective; NC protocol and best-validation-accuracy checkpoint selection. |
| `wo_semantic_anchor` | `outputs/core_story_ablation/nc/{Movies,Grocery}/wo_semantic_anchor/seed{42,43,44}/best.pt` | Formal Core Story outputs; manifests mark semantic anchoring ineffective; NC protocol and best-validation-accuracy checkpoint selection. |
| DiP | `outputs/paper_nc_final_fixed_v1/{Movies,Grocery}/dip/seed{42,43,44}/best.pt` | Unique matching target-dataset NC family under `unified_full_graph_nc_v1`, fixed split, validation-accuracy selection and passing health checks. The unrelated cloth-copurchase DiP output is an LP run and is excluded. |

The machine-readable audit is `outputs/robustness_analysis/checkpoint_audit.csv`; it records resolved configurations, formal `K`, metrics, split provenance, graph and raw-feature paths, checkpoint/config hashes and available Git provenance. It verifies that every checkpoint config for a dataset points to the same physical graph, raw modality features and edge-direction/self-loop convention. The split audit found a formal-family difference: Full and DiP use the fixed seed-42 NC split for all model seeds, whereas the Core Story ablations use the split corresponding to each model seed. As required, each checkpoint is evaluated on its own original split and its own clean metric; no split is substituted. This means retention is within-checkpoint, but cross-model comparisons are not fully paired on identical test nodes for seeds 43/44. The Core Story manifests record commit `3549c8c39e651d716519af37f1cbfeb9da20aeb3`. The selected `paper_nc_final_fixed_v1` Full and DiP run records do not record a Git commit, so their audit rows say `not_recorded`; their path, resolved config, fixed split, record health checks and SHA256 hashes identify the selected artifacts. DiP provenance is unambiguous within the requested target NC scope: each dataset × seed has exactly one matching checkpoint family. The only other discovered DiP checkpoint family is `outputs/f1_final_execution/lp_quasi_held_out/cloth-copurchase/dip/`, which is LP on a different dataset and is excluded.

Each checkpoint is gated on a clean-graph re-evaluation before any nonzero perturbation is evaluated. Clean accuracy and Macro-F1 must match the checkpoint record to absolute tolerance `1e-6`; otherwise evaluation stops before perturbed conditions.

## Clean graph and edge representation

The official NC data loader supplies the clean `edge_index`. The preparation audit canonicalizes support as unique undirected non-self pairs `(u,v)` with `u<v`. For Movies and Grocery, it verifies that the loader graph has no self loops or duplicate oriented columns and has exactly two directed columns per canonical physical edge. To construct a perturbed graph, the evaluator preserves the complete clean `edge_index` prefix and appends both orientations of every injected canonical pair. Thus one requested injected physical edge adds exactly two directed columns.

The loader audit found Movies with 16,672 nodes, 80,401 clean canonical edges and 160,802 `edge_index` columns; Grocery has 17,074 nodes, 71,131 clean canonical edges and 142,262 columns. At the maximum random-noise ratio (`rho=0.40`), the injected budgets are 32,160 and 28,452 edges, respectively; the perturbed supports therefore contain 112,561 / 99,583 canonical edges and 225,122 / 199,166 directed columns. Budget rounding is fixed as `floor(rho * |E| + 0.5)`.

## A. Random Structural Noise

For each dataset and perturbation seed 1001, 1002 or 1003, the generator samples one ordered list of unique random nonedges large enough for `rho=0.40`. Every pair is canonical, non-self, absent from clean physical support and sampled independently of labels, predictions and model internals. Requested conditions use prefixes of the same ordered list, so the injected supports are nested. The exact persisted graph artifact is shared across every model and model seed at a given dataset × perturbation seed × `rho`.

The preregistered ratios are `0.00, 0.05, 0.10, 0.20, 0.30, 0.40`. `random_noise/perturbation_capacity.csv` records the clean support, available nonedges, requested counts and maximum-ratio artifact; `random_noise/dry_run_plan.csv` identifies each future model evaluation without running it.

## B. Semantic-Conflict Edge Injection

### Candidate pool and ranking

Before any perturbed model evaluation, the generator draws a deterministic pool of **1,000,000 unique nonedges per dataset × perturbation seed**. This is about 41.5× the largest requested total semantic-conflict budget (Movies, `rho=0.30`); the resulting smaller modality categories are separately capacity-audited. The pool size was fixed from the largest preregistered requested edge budget, without using model performance. For each sampled pair it computes cosine similarity on the original frozen raw text and visual features. It reuses the chunked cosine and average-tie empirical-rank functions from `scripts/run_mopf_e0a_edge_semantic_discrepancy.py`; it does not use projected features, learned conductance, propagated embeddings, labels or predictions.

Within each candidate pool, average-tie empirical ranks are scaled by pool size to `[0,1]`. Categories are frozen as:

- `C_T`: `r_T >= 0.75` and `r_V <= 0.25`.
- `C_V`: `r_V >= 0.75` and `r_T <= 0.25`.

The candidate pool, cosine values, ranks and category codes are retained under `semantic_conflict/candidate_pools/` for replay. Candidate-capacity counts and feature provenance are in `semantic_conflict/candidate_capacity.csv`.

### Injection and feasible schedule

For each dataset × perturbation seed, the generator deterministically permutes each category and constructs a nested order alternating categories so every prefix differs by at most one edge between `C_T` and `C_V`. An odd budget puts its extra edge in a class with available capacity; ties are resolved deterministically. The maximum balanced capacity is `2*min(|C_T|,|C_V|) + 1[|C_T| != |C_V|]`.

The preregistered ratios are `0.00, 0.05, 0.10, 0.20, 0.30`. If `0.30` is not feasible in every Movies/Grocery × perturbation-seed condition, the largest common feasible ratio from that fixed set is used. All six dataset × perturbation-seed pools support `rho=0.30`; the full schedule is retained. Movies requires 24,120 injected edges at `rho=0.30` and Grocery requires 21,339. Observed candidate capacities are:

| Dataset | Perturbation seed | `|C_T|` | `|C_V|` | Maximum balanced capacity | Maximum feasible `rho` |
|---|---:|---:|---:|---:|---:|
| Movies | 1001 | 57,105 | 57,342 | 114,211 | 0.30 |
| Movies | 1002 | 56,931 | 57,739 | 113,863 | 0.30 |
| Movies | 1003 | 56,827 | 57,889 | 113,655 | 0.30 |
| Grocery | 1001 | 55,705 | 57,760 | 111,411 | 0.30 |
| Grocery | 1002 | 55,530 | 57,502 | 111,061 | 0.30 |
| Grocery | 1003 | 56,188 | 57,410 | 112,377 | 0.30 |

Category thresholds and candidate-pool size are not tuned. The selected schedule and all per-condition counts are written to the capacity CSV and manifest before any inference.

## Planned evaluation matrix and reporting

Random Structural Noise includes Full CoSI-MAG, `wo_relation_calibration`, `wo_semantic_anchor` and DiP. Semantic-Conflict Injection includes Full CoSI-MAG, `wo_relation_calibration` and DiP. Each includes three model seeds and three perturbation seeds. The plan retains one row for every dataset × model × model seed × perturbation seed × ratio. `rho=0` values are reused from one numerically verified clean inference per checkpoint because the clean graph is identical across perturbation seeds; repeated raw-plan rows are marked as cached clean baselines. The evaluator gates all checkpoints globally before starting any nonzero perturbation.

For every condition, the saved endpoints are absolute test Accuracy and Macro-F1. Retention is `100 * Metric(rho) / Metric(0)`, calculated separately for both metrics; the main Figure 5 curve uses Accuracy retention and Macro-F1 retention is secondary. Within each dataset × model × model seed × `rho`, retention is first averaged over the three perturbation seeds. The resulting three model-seed means are summarized as mean ± population standard deviation (`ddof=0`). The independent replication unit is the trained model seed (`n=3`); perturbation seeds characterize graph-draw variability and do not create nine independent trained runs. No p-values are planned.

The secondary RelationCalibrationGap is the aggregated Full minus `wo_relation_calibration` retention curve for both perturbations. AnchorGap is Full minus `wo_semantic_anchor` for random structural noise. Trends are reported as observed and are not assumed to be monotone.

The dry-run plan contains **702** matrix rows: 432 random-noise rows plus 270 semantic-conflict rows. The complete conflict schedule through `rho=0.30` is feasible. The evaluator performs one clean gate per checkpoint (24 clean inference calls), then 576 nonzero graph conditions; it reuses each verified clean output for the repeated `rho=0` rows. Thus there are 702 retained condition rows and 600 distinct model inference calls if the full evaluator is run. The generated manifest and plan CSVs contain the exact post-capacity count. No formal matrix has been run in this preparation stage.

## Figure 5 layout

Use a 2×2 quantitative grid: rows are Random Structural Noise and Semantic-Conflict Injection; columns are Movies and Grocery. Label panels (a)–(d) in row-major order. Plot Accuracy retention against `rho`; distinguish model families with consistent colors/markers and show the three model-seed means with mean ± population SD across those seeds. Share the y scale and add a neutral 100% reference line. Use the preregistered x range for each attack (0–0.40 for random noise; 0–0.30 for semantic conflict). Macro-F1 retention belongs in the appendix. Captions should state `n=3 model seeds`, that perturbation-seed outcomes were averaged within model seed, the population-SD convention, and that the graph changes only at inference. In the semantic-conflict row, `wo_semantic_anchor` is not included by design. Avoid significance stars and claims of monotonicity.

## Execution commands (for a later authorized formal run)

Preparation only:

```bash
PYTHONPATH=src conda run --no-capture-output -n yhf_env \
  python scripts/prepare_robustness_perturbations.py --device cpu
```

One clean-only check for a single checkpoint:

```bash
PYTHONPATH=src conda run --no-capture-output -n yhf_env \
  python scripts/evaluate_robustness.py --check-clean Movies mopf 42 --device cpu
```

The formal matrix is deliberately opt-in and was **not** run here:

```bash
PYTHONPATH=src conda run --no-capture-output -n yhf_env \
  python scripts/evaluate_robustness.py --execute --perturbation-type all --device cpu
```

After formal evaluation has produced `robustness_summary.csv`, the publication figure can be rendered with:

```bash
PYTHONPATH=src conda run --no-capture-output -n yhf_env \
  python scripts/plot_robustness.py \
  --summary outputs/robustness_analysis/robustness_summary.csv \
  --output-root outputs/robustness_analysis
```

## Limitations

This design tests inference-time addition of physical relations, not edge deletion, weight corruption, distribution shift in features, or adaptation through retraining. Semantic-conflict ranks are empirical within a fixed random candidate pool, not ranks over every possible nonedge. Results cover the two preregistered representative datasets and three model seeds; small seed counts limit precision. The protocol evaluates robustness of saved clean-trained systems and does not establish that the perturbation process represents every real deployment failure mode.
