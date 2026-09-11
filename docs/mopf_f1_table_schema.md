# MoPF F1 Pre-Registered Table Schema

Status: **FROZEN before F1 results**

This schema fixes the main tables before any F1-B result is inspected. It uses
method-freeze SHA `4ddbd6918ceebadc25eed2694e1b463f9aac87f4` and the protocol in
`docs/mopf_final_evaluation_protocol.md`.

## NC main table

One row per registered model and formal dataset. Each numeric entry is the
mean ± population standard deviation over seeds `42, 43, 44`; retain the
three per-seed values in the accompanying artifact.

| Model | Dataset | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Seeds | Test status |
|---|---|---:|---:|---:|---:|---|---|
| registered model | Movies/Toys/Grocery/ele-fashion/Reddit-S | mean ± pop. std | mean ± pop. std | mean ± pop. std | mean ± pop. std | 42/43/44 | descriptive |

The formal NC datasets and orders are fixed at Movies=3, Toys=3, Grocery=2,
ele-fashion=3, and Reddit-S=3. Validation Accuracy is the checkpoint
selection metric. Validation Macro-F1 is secondary; both test columns are
descriptive and cannot select a model or alter the protocol.

The registered external rows are `mlp`, `gcn`, `sage`, `mmgcn`, `mgat`, `dip`,
`dgf`, `dmgc`, and `lgmrec`, plus final `mopf`. MAP-family rows are internal
only and must remain outside the formal external aggregate.

## LP main table

One row per registered model on sports-copurchase under
`unified_sampled_lp_v1`. Every entry is mean ± population standard deviation
over seeds `42, 43, 44`.

| Model | Val MRR | Val Hits@1 | Val Hits@3 | Val Hits@10 | Test MRR | Test Hits@1 | Test Hits@3 | Test Hits@10 | Seeds | Test status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| registered model | mean ± pop. std | mean ± pop. std | mean ± pop. std | mean ± pop. std | mean ± pop. std | mean ± pop. std | mean ± pop. std | mean ± pop. std | 42/43/44 | descriptive |

Validation MRR is the fixed checkpoint-selection and primary validation
metric. The test metrics are descriptive only. The train-only message graph,
filtered negatives, `[5,5]` sampling, shared decoder, full-graph exact
evaluation, and pessimistic-tie evaluator are fixed by the protocol and may
not be changed after results are seen.

## Reporting constraints

Report per-seed values, mean, population standard deviation, parameter count,
best epoch, training time, peak memory, resolved configuration, split
provenance, and protocol version. Keep quasi-held-out results separate and do
not add metrics, datasets, baselines, or selection rules in response to F1
outcomes.
