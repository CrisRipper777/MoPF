# CSSI P0-B: LP protocol audit

Audit scope: branch `cssi`, current checkout, current configuration files, and
read-only inspection of available formal output manifests outside this
checkout. No LP job was launched in this round.

## Current source configuration

`configs/task/lp.yaml` currently contains:

```yaml
protocol_version: unified_sampled_lp_v1
num_neighbors: [5, 5, 5]
subgraph_type: bidirectional
batch_size: 2048
```

The LP runner resolves this list to the model's required depth in
`src/tasks/common.py:131-152`. For a three-layer encoder it retains all three
entries. The current `max_order=3` model configurations therefore have three
sampled neighborhood levels available for three explicit propagation steps.

## Historical resolved-config evidence

The current checkout's `outputs/` contains no formal LP runs. The available
formal output tree at `/hdd1/DataInHere/YHF/MoPF_IAMOC/outputs` was inspected
read-only. The following existing resolved configs all identify themselves as
`unified_sampled_lp_v1` but contain only two neighbor levels:

- `outputs/cosi_mag_final_benchmark/lp/sports-copurchase/runs_42_43_44/resolved_config.yaml`
- `outputs/cosi_mag_final_lp_stability/Movies/seed_42/run/resolved_config.yaml`
- `outputs/lp_baseline_benchmark/sports-copurchase/gcn/runs_42_43_44/resolved_config.yaml`

Each has:

```yaml
protocol_version: unified_sampled_lp_v1
num_neighbors:
- 5
- 5
```

The same two-hop value appears in the corresponding LP benchmark output
families discovered during the audit. Thus the historical record is mixed:
the source config is now `[5,5,5]`, while existing formal-looking `v1` results
include `[5,5]`. It would be incorrect to rewrite those historical manifests
or silently pool their metrics with three-hop runs.

## Documentation correction

The old two-hop description in the unified protocol document and README has
been corrected to state three-hop `[5,5,5]` sampling and alignment with
`max_order=3`. The current source config remains unchanged because it already
has the corrected list. No LP protocol version was retroactively reassigned.

## Required future action

Use `unified_sampled_lp_v2` for the corrected three-hop protocol once LP is
revisited. At that point, either regenerate the affected benchmark results or
keep the historical two-hop `v1` results in a separate, explicitly labelled
comparison. This P1 round does not run or rename LP experiments.
