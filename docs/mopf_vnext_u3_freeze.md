# MoPF-vNext U3 Freeze

## Selected candidate

**B — C1 + Transport-Conditioned Preference Residual (TCPR).**

The formal model keeps the U2-C C1 semantics:

- `edge_weight_mode=learned_diag_cos`
- `edge_weight_temperature=0.35`
- `multihop_state_mode=anchored`
- `multihop_response_mode=cumulative`
- `multihop_anchor_alpha=0.1`
- `use_transport_residual=true`

Formal K remains `Movies=3`, `Toys=3`, `Grocery=2`, `ele-fashion=3`, and
`Reddit-S=3`.

## TCPR semantics

For each modality, the model computes the U3-A mean incident conductance on
the raw physical support, centers it across nodes, and detaches that scalar
context. With a zero-initialized learned order profile,

`beta_k^m = theta_k^m - mean_r(theta_r^m)`

and

`tau_i,k^m = c_tilde_i^m beta_k^m`.

The effective coefficient is the existing C1 coefficient plus `tau`. TCPR
adds exactly `2*(K+1)` scalar parameters and introduces no router, attention,
MoE, normalization, projection, loss, fusion, classifier, evaluator, or LP
change.

## Selection evidence

- Degree-controlled preflight retained the conductance–response association:
  all 30 dataset×modality×seed main rows had the same partial-Spearman sign;
  median absolute partial rho was `0.3417`. The text-minus-visual
  conductance-gap relation was also direction-consistent.
- Initialization equivalence passed with relative L2 `0` on toy and Movies
  audits for eta, modality outputs, fused output, and logits.
- Five-dataset equal-weight Validation Accuracy changed by `+0.0436 pp`;
  Validation Macro-F1 changed by `+0.0212 pp`. Test metrics are descriptive
  only.
- TransportOff, TransportShuffle, and TransportModalitySwap all produced
  measurable frozen functional changes on all five datasets. TransportShuffle
  exceeded TransportOff at the dataset-mean level on all five datasets, and
  modality-context swap was measurable on all five datasets.
- The original hierarchy remained functional: B1 `NoNode`, `NoModality`, and
  `NodeShuffle` diagnostics were stable on all five datasets.
- TCPR did not dominate node personalization: the maximum transport/node
  residual L2 ratio was `0.1241`, and the median was `0.0439`.

## Formal stop

U3 is formally frozen at C1 + TCPR. No second U3 candidate, router,
attention, MoE, auxiliary loss, U1/U2 modification, LP extension, or fusion
redesign is authorized by this stage.

## Provenance

- U3-B master summary: `outputs/u3b_transport_conditioned_composition/u3b_master_summary.json`
- U3-B master table: `outputs/u3b_transport_conditioned_composition/u3b_master_table.csv`
- U3-B counterfactuals: `outputs/u3b_transport_conditioned_composition/u3b_counterfactuals.csv`
- U3-B associations: `outputs/u3b_transport_conditioned_composition/u3b_transport_associations.csv`
- U3-B report: `docs/mopf_u3b_transport_conditioned_composition.md`
- Formal config hash before selection: `65e39cce2358e6e06646ac0dd781abaa78f0dc1f9c6f9954e55337ff21417448`
- Formal config hash after selection: `1e29aa0f7141bbeeb16c695ba294358f59441f75b0d92fc7ffb55f560f7d140a`
