# Frozen MoPF-v0 reference

This document freezes the formal MoPF behavior before the vNext semantic
conductance study. The historical PDC-v1/PDC-v2, HRC, and PPC code remains
available for analysis and development, but none of it is part of the formal
architecture frozen here.

## Freeze identity

- Frozen behavior commit SHA: `1f8ebf127435a5d88b3f9d09118c87e704ad5ea5`
- Repository state before the manifest: clean (`git status --short` was empty)
- Freeze verification command: `conda run --no-capture-output -n yhf_env python -m pytest -q`
- Current pytest result: **143 passed, 1 warning** (PyG deprecation warning)
- Formal tag: `mopf-v0-frozen`
- vNext development branch: `vnext`

The SHA above is the last pre-manifest commit containing the frozen model
behavior. The freeze commit adds this reference manifest and is tagged
`mopf-v0-frozen`; the `vnext` branch is created from that tagged freeze point.

## Formal Frozen MoPF-v0 configuration

The formal NC model is `model=mopf` with the following resolved settings:

```yaml
hidden_dim: 256
dropout: 0.2
norm: layernorm
max_order: 3
num_layers: 3
map_prior_restart: 0.15
map_prior_order: 2
diffusion_add_self_loops: true
edge_weight_mode: separate_cos
edge_weight_min: 0.1
edge_weight_temperature: 2.0
filter_rank: 4
global_filter_trainable: true
node_conditioner_mode: absolute
use_modality_residual: true
use_node_residual: true
hrc_weight: 0.0
ppc_weight: 0.0
fusion_mode: concat_residual_mlp
export_aux_stats: false
export_node_aux: false
export_edge_aux: false
```

The conductance is computed independently for text and visual modalities as
`0.1 + 0.9 * sigmoid(cos(h_i^m, h_j^m) / 2.0)` on the original graph edges,
then passed through the existing `gcn_norm` path. The propagation bank,
hierarchical filter, modality/node residuals, refinement, and late fusion are
unchanged. **PDC / HRC / PPC are NOT part of formal MoPF-v0.**

## Parameter counts

Parameter counts below include the MoPF encoder and the NC linear classifier
(`num_classes` differs by dataset); they count trainable parameters only.
The S0 encoder count is independent of node and edge count. S1 adds 4
perspectives per modality and is therefore reported after its implementation
in the vNext benchmark manifest.

| Dataset | Text dim | Visual dim | Classes | Frozen NC parameter count |
|---|---:|---:|---:|---:|
| Movies | 768 | 768 | 20 | 1,001,312 |
| Toys | 768 | 768 | 18 | 1,000,798 |
| Grocery | 768 | 768 | 20 | 1,001,312 |
| ele-fashion | 512 | 512 | 12 | 868,184 |
| Reddit-S | 768 | 768 | 20 | 1,001,312 |

## NC protocol

The frozen protocol is `unified_full_graph_nc_v1`:

- datasets: Movies, Toys, Grocery, ele-fashion, Reddit-S;
- full-graph training, one encoder forward per epoch;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- 300 maximum epochs, validation every epoch;
- patience `30`, earliest early stopping epoch `30`, minimum delta `1e-4`;
- gradient clipping at max norm `1.0`;
- dropout `0.2`, hidden dimension `256`, K/max order `3`;
- seeds `42`, `43`, `44`;
- Macro-F1 uses the fixed label set resolved from the union of valid
  train/validation/test labels, with `zero_division=0`;
- checkpoint selection uses **best validation accuracy only**;
- test accuracy and test Macro-F1 are computed once after restoring that
  validation-selected checkpoint and never select a model.

Dataset-specific K for the vNext NC study is:

| Dataset | K |
|---|---:|
| Movies | 3 |
| Toys | 3 |
| Grocery | 2 |
| ele-fashion | 3 |
| Reddit-S | 3 |

## Existing reference benchmark locations

The previous formal MoPF NC benchmark is documented in
`docs/nc_benchmark_results.md`, with run artifacts under
`outputs/2026-09-09/` (see the per-dataset paths listed in that report).
Those historical results are references only. The U1 study must retrain S0
and S1 together and must not splice old results into its authoritative table.

LP is intentionally excluded from the vNext architecture-screening stage.
LP will be used only after the NC architecture is nearly finalized. No LP
experiment, sports-copurchase run, LP protocol change, decoder change, or
sampler change is authorized by this stage.
