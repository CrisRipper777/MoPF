# CoSI-MAG Figure 4 Mechanism Verification Protocol

Status: checkpoint-only offline analysis. No Full model, training protocol, or ablation definition was changed.

## Selected Full checkpoint set

The user selected `outputs/paper_nc_final_fixed_v1` for Figure 4. The 15 paths, checkpoint SHA256 values, model configs, validation-accuracy checkpoint-selection metadata, run-record health checks, and stored final test metrics are in `outputs/mechanism_verification/checkpoint_audit.csv`. This selected set does not reproduce every frozen NC table mean and standard deviation after two-decimal rounding; the discrepancy is retained in the audit and manifest and was not used to replace the selected checkpoints.

Formal K is Movies=3, Toys=3, Grocery=2, ele-fashion=3, Reddit-S=3. Seeds 42, 43, 44 are the replicate unit. Figure 4 mean±SD error bars use population SD across seeds (ddof=0), matching the frozen NC/LP tables. Existing Stage-I/II raw seed rows and summary CSVs are frozen and were not rewritten; Figure error bars compute population SD directly from the seed rows. No tests are conducted over individual edges or nodes.

## Figure 4(a): Modality-specific relation calibration

Physical edges are unique undirected non-self edges from the NC loader, using `canonicalize_edges`. Raw text and visual edge cosine similarities are computed from the frozen input feature matrices before learned projections. Edge ranks use E0-A's average-tie empirical CDF (`scripts/run_mopf_e0a_edge_semantic_discrepancy.py::_average_rank_cdf`): average 1-based rank divided by the number of physical edges. Frozen categories are E_T: r_T>=0.75 and r_V<=0.25; E_V: r_V>=0.75 and r_T<=0.25. No category threshold is fitted.

Stage-I conductances are read from `MoPF.conductance_stats` on canonical physical edges, which returns raw learned conductance separately from normalized edge weights. Self-loops and GCN normalization are excluded from these reported conductances. M_T=mean_E_T(w_T-w_V), M_V=mean_E_V(w_V-w_T), and M_overall=(M_T+M_V)/2. Edges are measurements, not independent replicates.

## Figure 4(b): Semantic retention under multi-hop propagation

For each selected checkpoint and modality, H0 and Ahat come from that checkpoint's eval-mode `_encode_components` path. The actual frozen model recurrence is S_0=H0 and S_k=(1-alpha)Ahat S_(k-1)+alpha H0, alpha=0.1. The ordinary counterfactual starts from the exact same H0 and applies the exact same Ahat with Q_0=H0 and Q_k=Ahat Q_(k-1). Only the recurrence differs. The initial task text displayed a minus sign before alpha H0; the trained model implementation and frozen method document both use plus, so this analysis follows the trained Full path.

Linear CKA is centered over all graph nodes and uses the feature-space formula ||X_c^T Y_c||_F^2 / (||X_c^T X_c||_F ||Y_c^T Y_c||_F), accumulated in float64 row chunks. No N-by-N Gram matrix or node subsampling is used. Retention gain is CKA(S_K,H0)-CKA(Q_K,H0); it is a functional comparison and is not interpreted as a cause of classification performance.

## Figure 4(c): Adaptive contribution reallocation

The adaptive contribution profile reuses `src.analysis.u3a.contribution_profile`: G_adapt_i,k=eta_i,k S_i,k, g_adapt_i,k=||G_adapt_i,k||_2, and p_adapt_i,k=g_adapt_i,k/sum_r g_adapt_i,r. The uniform-composition counterfactual uses the exact same checkpoint response bank S_i,k, sets eta_uniform=1/(K+1), and computes p_uniform_i,k=||S_i,k||_2/sum_r ||S_i,r||_2. The primary statistic is D_i=0.5 sum_k |p_adapt_i,k-p_uniform_i,k|. Consequently, unequal response norms alone do not count as adaptive reallocation. The optional coefficient-only diagnostic defines q_i,k=|eta_i,k|/sum_r|eta_i,r| and C_i=0.5 sum_k |q_i,k-1/(K+1)|; C is not the primary Figure-4 metric. E0-C used all graph nodes; this analysis also uses all nodes. A zero denominator in either profile stops analysis rather than introducing another normalization.

The node-level NPZ and summary statistics retain all graph nodes. For the violin density rendering only, the plotting script uses a deterministic sample of at most 5,000 nodes per dataset and modality to bound KDE cost; all-node medians and seed-level mean markers are computed without that rendering subsample. Node and edge distributions are mechanism visualizations, not independent experimental replicates. Cross-dataset magnitude ranking is not claimed because formal K differs for Grocery.

## Provenance

The machine-readable run manifest is `outputs/mechanism_verification/mechanism_verification_manifest.json`; the isolated Stage-III rerun audit is `outputs/mechanism_verification/stage3_adaptive_reallocation_audit.json`. Stage III was regenerated without rewriting Stage-I/II outputs. Source definitions reused: E0-A raw cosine, average-tie empirical ranks and canonical physical support; E0-C `contribution_profile`; the frozen `MoPF` conductance/operator/state/response helpers. Outputs are the per-seed CSVs, seed summaries, node-level NPZ, and Figure 4 PDF/PNG/SVG under `outputs/mechanism_verification/`.
