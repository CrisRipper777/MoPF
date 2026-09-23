# MGSC-MAG corrected architecture controls

## Scope and protocol

This study is the final architecture-control phase for the five NC datasets:
Movies, Toys, Grocery, ele-fashion, and Reddit-S. Sports-LP and all other LP
experiments are out of scope. Each control uses seeds 42/43/44 and the same
official split, full-graph NC runner, optimizer, early stopping,
checkpoint criterion, classifier, preprocessing, and evaluation metrics.

The canonical configuration is [`configs/model/mgsc_mag_p2.yaml`](../../configs/model/mgsc_mag_p2.yaml):

```yaml
adaptive_context_gate: true
direct_interacted_integration: true
use_legacy_relation_order_bias: true
```

The corrected-control implementation is
[`src/models/mgsc_mag_corrected_controls.py`](../../src/models/mgsc_mag_corrected_controls.py),
with the runner in
[`scripts/final/run_mgsc_corrected_controls.py`](../../scripts/final/run_mgsc_corrected_controls.py).
The previous fine-grained A1-A6 matrix was not rerun in this phase.

## Controls

- `full`: canonical P2; C0 delegates to the canonical MGSC path exactly.
- `raw_terminal_only`: forms the complete Stage-I state bank, then reads raw
  `S_K`; it bypasses cross-order interaction and all eta coefficients.
- `raw_bank_mean`: forms the complete raw state bank and reads the arithmetic
  mean of `S_0,...,S_K`; it bypasses all Stage-II readout.
- `plain_multi_order_backbone`: keeps modality-specific projection,
  multi-order propagation, refinement, late fusion, and the task head, but
  uses unit physical-edge weights before normalization, fixed `g=0.9`, raw
  bank mean, no cross-order interaction, and no preference coefficients.
  Its lower parameter count is natural because the removed conditional
  parameters are not retained.

The raw controls do not call the interaction path. The plain control does not
contain learned MRC, gate, interaction, or preference parameters. Text and
visual paths remain separate until late fusion in every control.

## Formal results

Values are mean ± population SD over three seeds. Accuracy and Macro-F1 are
test metrics evaluated only after validation checkpoint selection.

| Dataset | Full accuracy | Full Macro-F1 |
|---|---:|---:|
| Movies | 0.5643 ± 0.0052 | 0.5026 ± 0.0101 |
| Toys | 0.8013 ± 0.0061 | 0.7741 ± 0.0067 |
| Grocery | 0.8331 ± 0.0054 | 0.7581 ± 0.0106 |
| ele-fashion | 0.8831 ± 0.0007 | 0.7812 ± 0.0048 |
| Reddit-S | 0.9691 ± 0.0026 | 0.9320 ± 0.0037 |

The following are paired means in percentage points, defined as
`Full - control` for the same dataset and seed. Positive values favour Full.

| Dataset | Raw terminal Acc/F1 | Raw bank mean Acc/F1 | Plain backbone Acc/F1 |
|---|---:|---:|---:|
| Movies | +0.480 / +0.812 | +0.610 / −0.063 | +0.080 / +0.075 |
| Toys | −0.354 / −0.405 | +0.153 / +0.303 | +0.540 / +0.544 |
| Grocery | −0.098 / −0.479 | +0.420 / +0.120 | +0.390 / +0.345 |
| ele-fashion | −0.167 / −0.237 | +0.014 / +0.758 | +0.214 / +0.462 |
| Reddit-S | −0.042 / −0.162 | +0.157 / +0.321 | +0.231 / +0.243 |

Across all 15 paired dataset-seed comparisons, the mean deltas are:

| Control | Accuracy | Macro-F1 | Full-better count, accuracy | Full-better count, Macro-F1 |
|---|---:|---:|---:|---:|
| raw_terminal_only | −0.036 pp | −0.094 pp | 6/15 | 5/15 |
| raw_bank_mean | +0.271 pp | +0.288 pp | 11/15 | 10/15 |
| plain_multi_order_backbone | +0.291 pp | +0.334 pp | 14/15 | 14/15 |

These are small, mixed effects. They do not justify ranking modules or
claiming that every Stage-II operation is necessary for every dataset.

## Parameter and integrity checks

`full`, `raw_terminal_only`, and `raw_bank_mean` have the same trainable
parameter count as Full. The plain backbone has 987,904 encoder parameters
versus 1,528,026 on Movies/Toys/Grocery/Reddit-S, a 35.35% reduction; on
ele-fashion the reduction is 38.66%. The classifier and late-fusion/refinement
capacity are unchanged within each dataset.

The C0 state-dict shape/key test passes. Maximum eval forward difference is
`0.0` on the CUDA check (target `<=1e-5`) and `<=1e-6` on CPU. Integrity tests
also verify unit pre-normalization relation weights, node-invariant plain
gates, raw terminal/bank readouts, identity interaction placeholders,
finite isolated-node behavior, and finite backward gradients.

The earlier formal P2 checkpoints failed the current code fingerprint gate,
so the corrected Full rows were trained rather than silently reused. The
authoritative outputs are:

- `outputs/final/mgsc_corrected_controls/per_seed_results.csv`
- `outputs/final/mgsc_corrected_controls/summary.csv`
- `outputs/final/mgsc_corrected_controls/paired_deltas.csv`
- `outputs/final/mgsc_corrected_controls/parameter_counts.csv`
- `outputs/final/mgsc_corrected_controls/manifest.json`

## Decision

The strict controls support a bounded Tier-B interpretation: the canonical
P2 path is a real, functional mechanism chain and has a small, mostly
consistent advantage over the parameter-reduced plain backbone, but raw
state readouts are often comparable and sometimes better. Canonical P2 is
therefore retained as the working architecture for the five-NC paper scope,
without a new architecture-search phase.
