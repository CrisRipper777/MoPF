# Figure 4 final claims

## (a) Modality-specific order profiles

The normalized contribution statistic is `q_{i,k}=|eta_{i,k}|/sum_j|eta_{i,j}|`, and the effective order is `sum_k k q_{i,k}`. Profiles are concentrated in the middle-to-higher orders but vary by dataset and modality. The panel supports modality-specific learned order profiles; it does not support strong universal node-level heterogeneity or an optimal propagation depth claim.

## (b) Cross-order interaction pattern

The complete average attention matrices are exported for all five NC datasets and both modalities. Representative panels show that order-0 dominance occurs in some paths, whereas other paths place more mass on higher-order keys. Therefore the safe claim is dataset-/modality-dependent cross-order interaction, not a universal attention pattern. The matrices show the interaction computation; they do not by themselves establish causal necessity.

## (c) Functional integration intervention

Query-collapse and uniform-attention perturbations are smaller than interaction-off in the current frozen checkpoints across all five NC datasets. This establishes that the interaction output reaches the final representation through the direct P2 composition. It does not establish universal accuracy gains or that query-dependent attention is the only source of the effect.

These are frozen-checkpoint inference interventions, not retrained ablation performance. Source CSVs are under `outputs/final/multi_order_integration_analysis/` and `outputs/final/paper_figures/figure4_data/`.
