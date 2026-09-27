# Relation-Conditioned Semantic Operator Audit — Preflight

## Repository state

- Repository: `CrisRipper777/MoPF`
- Base branch: `structural_observability_audit`
- Base HEAD after `git pull --ff-only`: `fc558c86584679047c3dcf5ba5261bf8bf45d7aa`
- Base was clean and synchronized with `origin/structural_observability_audit`.
- Work proceeds on the isolated `relation_operator_audit` branch.
- Historical E1–E7 result directories are treated as read-only.

## Frozen NC behavior

- `Model(cfg, data_info)` is built through `src.models.factory.build_model`.
- NC calls `forward(x, edge_index)` and expects `(z, _, _, aux_loss, aux_info)`; `out_dim` sizes the linear classifier. `inference(...)` returns CPU embeddings.
- A model with `requires_full_graph_training = True` forces full-graph training. Each epoch performs one full-graph forward and CE only on `train_idx`; Validation is evaluated each configured interval. The best checkpoint is chosen by Validation Accuracy; Macro-F1 is reported alongside it.
- With `task.evaluate_test=false`, NC resolves the Macro-F1 label set from Train and Validation, skips test evaluation and modality-mask test diagnostics, and returns no test metrics. Checkpoint metrics therefore contain only validation fields. Before this P0, the full `data.y` vector was also copied to the GPU in full-graph training despite only Train targets being used. The P0 implementation changes this path to move only `data.y[train_idx]` to the device. Dataset loading still reads the label tensor into CPU memory as part of the repository's existing data format.
- NC checkpoint payload: `task`, `seed`, `selection`, `epoch`, `metrics`, CPU-cloned `model_state`, `head_state`, and `data_info`.

## Frozen model and graph details

- `cmrf_probe` Text and Visual projectors are each `Linear → LayerNorm → ReLU → Dropout`; its shared plain fusion is `Linear(2d,d) → ReLU → Dropout → Linear(d,d)`.
- The physical operator removes existing loops, calls `to_undirected`, adds one loop per node, coalesces coordinates, assigns unit weights, and symmetrically normalizes. Sparse coordinate row is target and column is source, so `torch.sparse.mm(P, C)` aggregates source messages into targets.
- CMRF uniform composition applies `H0 + .75(C1-C0) + .50(C2-C1) + .25(C3-C2)`, algebraically equal to `(C0+C1+C2+C3)/4`.
- Movies, Toys, Grocery, ele-fashion, and Reddit-S all expose NC splits and the complete physical graph through their configured loaders. The new P0 model pins K=3 independently of historical dataset-specific settings.
- Existing E6/E7 code stores isolated checkpoints and histories/metrics, completion markers, and analysis tables/reports. E7 uses a phased winner-lock protocol; P0 does not reuse that selection flow.

## Pre-coding decisions

- Keep all existing CMRF/routing-audit modules and historical outputs untouched.
- New artifacts use only `relation_operator_audit_v1` names.
- Use the frozen NC task configuration and explicit `evaluate_test=false` on every launcher command.
- The requested qualitative H2 “catastrophic negative transfer” rule has no numeric cutoff. The analyzer reports per-dataset deltas and leaves that judgment visible rather than silently inventing a threshold.
- No formal P0 training is authorized in this implementation pass; expected matrix size is 4 variants × 5 datasets × 3 seeds = 60 contexts.
## Implementation checks added after preflight

- The only existing framework change is in `src/tasks/nc.py`: full-graph training now transfers just Train targets to the device. Validation continues to index only validation targets; test evaluation remains skipped when disabled.
- Dedicated and relevant regression tests passed (13 dedicated; 42 combined).
- CUDA smoke completed for the four Movies variants and Reddit-S A3. Peak memory and model-only parameter counts are recorded in `docs/relation_operator_audit_design.md` and `parameter_counts.csv`.
- The formal launcher matrix is verified at 60 distinct run/checkpoint paths; formal execution was not started.
