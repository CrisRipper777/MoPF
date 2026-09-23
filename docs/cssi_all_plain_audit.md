# CSSI P0-A: `all_plain` audit

Audit scope: branch `cssi`, current checkout, with no modification to
`src/models/cosi_mag_final.py`, the formal baseline implementation, or the NC
runner.

## Conclusion

`ablation_mode=all_plain` in `CoSIMAGAblation` is a suitable strong plain
backbone for the Structural Response hypothesis. Its effective forward path
is unit physical edge weights, one shared symmetric GCN operator, ordinary
three-step propagation, uniform order pooling, and the original modality
refinement plus late fusion. It does not enter MRC, local relation context,
RCMI, learned order composition, or relation-conditioned attention.

The inherited RCMI/MRC parameter tensors remain present in the module and in
the optimizer/checkpoint state dict, but the all-plain branch does not read
them in forward. This is an implementation-overhead caveat, not a hidden
functional path.

## Actual call chain

For NC, the runtime chain is:

```text
src.main.main
  -> src.data.load_mag_data
  -> src.tasks.nc.run_nc
  -> _run_single_nc
  -> src.models.factory.build_model
  -> src.models.cosi_mag_ablation.Model
  -> CoSIMAGAblation.forward
  -> CoSIMAGAblation._encode_components
  -> CoSIMAGAblation._encode_without_rcmi   # all_plain branch
```

The canonical model config is
`configs/model/cosi_mag_ablation.yaml` with
`ablation_mode: all_plain` and `multihop_anchor_alpha: 0.0`.

The decisive branch is `src/models/cosi_mag_ablation.py:102-190`:

1. `_split_features` separates the concatenated `[text, visual]` input.
2. `text_proj` and `visual_proj` independently produce `h_text` and
   `h_visual`.
3. The overridden `_relation_calibration` returns one tensor of ones per
   physical edge for each modality.
4. `_normalized_operator` calls PyG `gcn_norm` with `add_self_loops=True`,
   `improved=False`, and those unit edge weights.
5. `_multi_hop_states` retains `[S0, S1, S2, S3]` for each modality.
6. Each state bank is pooled with `torch.stack(...).mean(dim=1)`.
7. The original modality-specific residual MLP/LayerNorm blocks and
   `fusion_skip + fusion_mlp` plus output LayerNorm are still applied.

The NC runner (`src/tasks/nc.py`) trains the model output with the unchanged
linear classifier, selects the best validation-accuracy checkpoint, restores
both model and head state, and only then optionally evaluates test. The P1
launcher sets `task.evaluate_test=false`; this does not alter the formal NC
runner.

## Mathematical mapping

Let `P` be the operator returned by `gcn_norm` for the physical edge index,
unit edge weights, and added self-loops. The branch implements:

```text
H0_T = text_proj(X_T),   H0_V = visual_proj(X_V)
S0_m = H0_m
S_k_m = P S_{k-1}_m,      k = 1,2,3
Z_plain_m = mean(S0_m, S1_m, S2_m, S3_m)
```

Because the Text and Visual calls pass the same edge index, node count, and
unit edge-weight vector into `gcn_norm`, the normalized operator is strictly
identical in index and weight tensors. The response probe and unit tests assert
this equality.

## Parameters that participate in `all_plain` forward

- `text_proj` and `visual_proj` weights, biases, normalization, and dropout
  configuration (dropout is inactive in `eval()` and active during training).
- `max_order` and `multihop_anchor_alpha`; the canonical alpha is exactly
  `0.0`.
- `diffusion_add_self_loops`, which is `true` in the canonical config.
- The physical `edge_index`, node count, and the two all-ones physical edge
  weight tensors.
- `text_refine_mlp`, `visual_refine_mlp`, their LayerNorms,
  `fusion_skip`, `fusion_mlp`, and `output_norm`.

## Parameters/configuration inherited but bypassed or only validated

The following are constructed by `CoSIMAGFinal` for state-dict compatibility,
but are not read by the all-plain forward branch:

- learned metric parameters `metric_theta_text/visual` and the semantic edge
  calibration controls (`edge_weight_min`, `edge_weight_temperature`, and
  the metric path's `eps`);
- `gamma_global`, `delta_gamma_text/visual`, `node_proj_*`, and
  `node_vector_*` (global/modality/node order preference);
- `hop_order_embedding_*`, `hop_layers_*`, `theta_hop_gate_*`,
  `relation_beta_raw_*`, and `theta_relation_scale_*` (RCMI, Q/K/V
  cross-order attention, order embedding, relation-conditioned bias, and
  learned relation order profiles);
- `global_prior_restart/order`, `filter_rank`, relation initialization
  controls, and hop-gate initialization controls. They initialize or shape
  bypassed tensors, but do not affect the computed `Z_plain` in this branch;
- `multihop_state` and `multihop_response` are checked against the frozen
  `anchored/cumulative` vocabulary, while alpha `0.0` makes the actual state
  recurrence ordinary.

The inherited methods `_local_relation_context`, `_cross_order_interaction`,
`_node_preference`, and `_compose` are not called by the all-plain
`_encode_without_rcmi` implementation. `relation_permutation`, relation and
interaction interventions, and attention capture are likewise not consumed by
this branch.

`metric_init_seed` is stored by the constructor but has no runtime use in the
current `CoSIMAGFinal` implementation. It is therefore not an effective
all-plain forward parameter either.

## Difference from `cosi_mag_backbone_probe`

`src/models/cosi_mag_backbone_probe.py` is not the canonical plain reference.
Its `probe_mode=no_graph` uses `H0` as the selected representation and its
`probe_mode=last_hop` uses `S3`; they are controls. Both currently compute the
unit-weight state bank internally, but that internal computation does not make
either probe equivalent to the canonical `all_plain` output. The probe's
`simple_fusion` and `early_fusion` modes are outside this P1 protocol and are
not run.

## Hidden-path checks

The audit tests cover:

- all-plain state bank and uniform order mean;
- unit physical edge weights;
- exact Text/Visual normalized operator equality;
- alpha-zero recurrence;
- response and plain-representation reconstruction;
- invariance to bypassed RCMI/order-preference parameters (through the
  existing ablation tests).

No learned relation calibration, anchor contribution, RCMI interaction,
learned order prior, or relation-conditioned bias is effective in `all_plain`.

## Suitability decision

Yes: `CoSIMAGAblation(ablation_mode=all_plain)` is an appropriate no-MRC,
no-anchor, no-RCMI strong plain backbone for P1 Structural Response analysis.
The response analysis must use the post-pooling modality representations and
then rerun the frozen refinement, late fusion, and classifier; it must not edit
logits directly. That independent path is implemented in
`src/models/cssi_response_probe.py`.
