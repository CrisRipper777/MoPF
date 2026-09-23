# Paper evidence manifest

This manifest distinguishes empirical motivation, downstream performance, retrained ablation, frozen-checkpoint intervention, and descriptive diagnostics. It is the provenance index for the current five-NC paper scope.

| Item | Authoritative source | Evidence role |
|---|---|---|
| Canonical model | src/models/mgsc_mag.py (MGSCMAG; canonical P2 switches) | implementation |
| Canonical config | configs/model/mgsc_mag_p2.yaml | configuration |
| Git SHA | 0d062e43ed90c8bd52baa7051b39b11dffbd150b0 | code snapshot |
| Datasets | Movies, Toys, Grocery, ele-fashion, Reddit-S | NC scope |
| Seeds | 42, 43, 44 | replication |
| Main benchmark | outputs/final/mgsc_corrected_controls/summary.csv; baselines: paper/tables/table_nc_main.tex | downstream performance |
| M1 | outputs/final/context_demand/ plus docs/final/context_demand_report.md | empirical motivation |
| Corrected controls | outputs/final/mgsc_corrected_controls/ | retrained ablation |
| Fine-grained ablation | outputs/final/mgsc_functional_ablation/ | retrained ablation / appendix |
| Figure 3 | outputs/final/context_formation_analysis/ and outputs/final/paper_figures/figure3_data/ | descriptive + frozen-checkpoint intervention |
| Figure 4 | outputs/final/multi_order_integration_analysis/ and outputs/final/paper_figures/figure4_data/ | descriptive + frozen-checkpoint intervention |

Inference interventions in Figures 3(c) and 4(c) use trained Full P2 checkpoints and are never described as retrained performance. M1 and Figure 1(c) use lightweight linear probes with official test labels excluded. The frozen CoSI reference files remain unchanged.
