# P3 generic diffusion controls

These are conceptual PPR-style and GPR-style controls, not official APPNP or GPR-GNN reproductions. Both use the same modality projection, late refinement, and late fusion scaffolding as S, while removing SCD-specific relation, semantic-feedback, adaptive-response, formation, and attention modules.

## Generic PPR-style

For each modality:

\[
S_0=H_0,\qquad S_k=0.9 P S_{k-1}+0.1H_0,\qquad Z=S_K.
\]

P is the raw physical unit-weight operator after the configured normalized graph construction. K=3. There is no learned semantic conductance, node-dependent response, modality-specific adaptive restart, formation conditioning, or hop attention.

## Generic GPR-style

For each modality:

\[
S_0=H_0,\qquad S_k=P S_{k-1},\qquad
Z=\sum_{k=0}^{K}\gamma_k^{(m)}S_k.
\]

The gamma bank is modality-specific, globally shared over nodes, signed, and learnable. It is not a node-dependent response mechanism. K=3, with four coefficients per modality. Initialization is the frozen-S-compatible prior with restart r=0.15 and prior order 2: gamma=[0.15, 0.1275, 0.7225, 0.0]. The coefficients remain unconstrained after initialization; there is no softmax or clipping.

## Parameter accounting

For the shipped configuration with LayerNorm projections, hidden width H=256, and total input width D=D_text+D_visual, the exact generic counts are:

- Generic PPR-style: D*H + 9*H^2 + 19*H.
- Generic GPR-style: D*H + 9*H^2 + 19*H + 2*(K+1), which is PPR plus 8 scalars at K=3.

The accounting categories are fixed:

| Control | Parameters beyond the input projections | SCD-specific relation/semantic/response/attention parameters |
| --- | --- | --- |
| Generic PPR-style | modality refinement MLPs, refinement norms, late fusion skip/MLP, output norm | none |
| Generic GPR-style | Generic PPR-style shared scaffold plus 2*(K+1) global gamma scalars | none |
| Full S | frozen S implementation, reported separately | present by design |

Thus GPR adds exactly eight scalar coefficients at K=3 relative to the same generic PPR scaffold. total_parameters, requires_grad_parameters, and functionally_active_parameters are kept as distinct fields; no generic control counts a dead SCD module.

## Compatibility contract

All three controls expose the same forward, inference, and analysis interfaces. They use requires_full_lp_sampler_depth=True, so the current LP authority resolves [5,5,5] for K=3. A synthetic sampled-like forward/backward finite check is part of the P3 test suite; it does not load a real dataset or start training.
