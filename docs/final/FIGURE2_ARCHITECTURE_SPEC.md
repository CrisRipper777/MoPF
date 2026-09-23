# Figure 2 architecture specification

The preview has exactly two top-level stages and follows the canonical P2 code path.

## Stage I — Modality-Calibrated Adaptive Context Formation

Shared physical graph support and separate text/visual inputs enter independent projection MLPs. For modality m, the code computes a learned diagonal weighted cosine on each physical edge, maps the cosine to a bounded relation weight, and applies symmetric graph normalization with the configured self-loops. Each order proposes `N_{i,k}^m = [Ahat^m S_{k-1}^m]_i`. The adaptive gate consumes `(H0, Nk, |H0-Nk|, H0*Nk, p_k)` through an independent text/visual MLP and forms `S_{i,k}^m = (1-g_{i,k}^m)H_{i,0}^m + g_{i,k}^m N_{i,k}^m`. The visible state bank is `B_i^m=[S_{i,0}^m,...,S_{i,K}^m]`.

The retained legacy relation-order bias is shown as a small conditioning arrow because it modifies attention logits; it is not presented as a third top-level stage or a new mechanism.

## Stage II — Interaction-Aware Multi-Order Context Integration

The state bank receives order embeddings and independent single-head Q/K/V interaction. The logits include the retained local relation-order conditioning, and the interacted states are `S~`. The preference path computes node/order responses from interacted states, then `eta = gamma_global + delta_gamma^m + delta_i^m`. Canonical P2 directly composes `Z_i^m = sum_k eta_{i,k}^m S~_{i,k}^m`, applies independent modality refinement, and performs late text/visual fusion before the NC classifier.

## Code-to-figure mapping

| Figure object | Code object |
|---|---|
| projection φ_m | `text_proj`, `visual_proj` |
| modality relation calibration | `_relation_calibration`, `metric_theta_*`, `_edge_weight_from_cosine` |
| weighted physical graph | `_normalized_operator` |
| neighborhood proposal | `_propagate_once` inside `_adaptive_multi_hop_states` |
| adaptive gate | `context_gate_text/visual`, `_adaptive_multi_hop_states` |
| state bank | `states_text`, `states_visual` |
| order-aware interaction | `_cross_order_interaction`, `hop_order_embedding_*`, `hop_layers_*` |
| legacy conditioning | `_local_relation_context`, `relation_beta_raw_*`, `use_legacy_relation_order_bias` |
| preference and eta | `_node_preference`, `gamma_global`, `delta_gamma_*`, `eta_*` |
| direct P2 integration | `direct_interacted_integration=True`, `_compose` |
| modality refinement/fusion | `*_refine_*`, `fusion_skip`, `fusion_mlp`, `output_norm` |

Text and visual paths remain separate until `torch.cat([z_text_refined, z_visual_refined])`.
