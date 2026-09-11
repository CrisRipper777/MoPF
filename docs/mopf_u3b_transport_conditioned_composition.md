# MoPF-vNext U3-B — Transport-Conditioned Hierarchical Composition

Controlled candidate test over the frozen U2-C C1 NC protocol. Test metrics are descriptive only.

- Preflight degree-control gate: **PASS**; median |partial rho| = `0.3417`, dominant sign fraction = `1.000`.
- Initialization equivalence: **PASS**.
- Performance-safe gate: **PASS**; five-dataset Val Acc delta = `+0.0436 pp`.
- Mechanism gate: **PASS**; TransportOff, TransportShuffle, and TransportModalitySwap support `5/5`, `5/5`, and `5/5` datasets.
- Hierarchy preservation: **PASS**; B1 NoNode, NoModality, and NodeShuffle were stable on `5/5`, `5/5`, and `5/5` datasets.

## TCPR definition

`c_tilde_i^m = mean_incident_conductance_i^m - mean_i(mean_incident_conductance_i^m)`; `beta_k^m = theta_k^m - mean_k(theta_k^m)`; `tau_i,k^m = c_tilde_i^m beta_k^m`. The context is detached, and the only added trainable values are `theta_transport_text/visual` with shape `[K+1]` and zero initialization.

## Final decision: B. Select C1 + TCPR

TCPR is validation-safe and passes the predefined functional mechanism gate, including node-context shuffle sensitivity, while the original modality/node hierarchy remains explicitly represented in the canonical and counterfactual diagnostics.

Authoritative machine-readable details are in `u3b_master_summary.json`; per-run counterfactuals, beta profiles, canonical decomposition, and hierarchy diagnostics are retained there.
