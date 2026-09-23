# Paper v3 blueprint

This is a writing outline, not a full manuscript. It is constrained by the canonical P2 implementation and the evidence manifest.

## 1. Title candidates

- Semantically Conditioned Multi-Order Contextualization for Multimodal Attributed Graphs
- Multi-Granular Structure-Semantic Context Modeling for Multimodal Attributed Graphs
- Modality-Calibrated Context-State Formation and Multi-Order Integration in MAGs

## 2. Abstract logical skeleton

1. MAGs combine modality-intrinsic semantics with physical relational structure.
2. A physical edge does not determine one modality-independent semantic contribution, and context utility varies over nodes, modalities, and structural ranges.
3. M1 and a raw fixed-graph order probe establish this motivation without MGSC checkpoints.
4. MGSC-MAG implements modality-calibrated context-state formation followed by interaction-aware multi-order context-state integration.
5. Five-NC results, corrected controls, and frozen-checkpoint interventions show a viable and computationally active framework with modest but mostly consistent gains over a strong plain multi-order backbone.
6. The conclusion states bounded claims and limitations.

## 3. Introduction paragraph plan

1. Define MAGs and the tension between physical graph support and multimodal attributes.
2. Explain why one shared edge response is insufficient: modalities can disagree on the same physical relation.
3. Explain why one global propagation/readout rule is insufficient: contextualization demand and order profiles vary.
4. State the hypothesis and recommended sentence: “Physical topology defines possible structural dependencies, whereas multimodal semantics determine how structural context is formed and utilized.”
5. Present the unified solution at two levels: formation and utilization.
6. Preview evidence: model-independent motivation, benchmark/control results, and mechanism diagnostics.
7. State exactly three contributions from `PAPER_CONTRIBUTIONS.md`.

## 4. Related-work positioning

Organize prior work by (i) attribute-aware graph learning, (ii) multimodal graph encoders, (iii) relation/edge calibration, and (iv) multi-hop or order-aware graph representations. Position MGSC-MAG around the separation of modality-specific context-state formation from multi-order context-state utilization. Do not claim that prior methods are universally inadequate; describe the missing combination under the current MAG hypothesis.

## 5. Problem formulation

Let a MAG be `G=(V,E,X^T,X^V,Y)`, where `E` is the observed physical topology and `X^T`, `X^V` are text and visual attributes. The NC objective predicts `Y` on the official train/validation/test protocol. The model keeps the two modality paths separate until late fusion.

## 6. Method overview

The canonical data flow is:

`(G, X^T, X^V) → independent projections → modality-specific relation calibration → separate physical graph operators → adaptive context states → B^m → cross-order interaction → eta-weighted interacted-state composition → independent refinement → late fusion → NC classifier`.

There are exactly two top-level stages: Stage I, Modality-Calibrated Adaptive Context Formation; and Stage II, Interaction-Aware Multi-Order Context Integration.

## 7. Stage I: modality-calibrated adaptive context formation

### Input and projection

For each modality `m ∈ {T,V}`, the code computes

`H_0^m = phi_m(X^m)`,

where `phi_m` is the independent `ProjectionMLP` (`Linear → norm → ReLU → Dropout`).

### MRC semantic metric and relation weights

The learned diagonal metric is

`d_theta^m = softplus(theta_m) / mean(softplus(theta_m))`.

For a physical edge `(i,j)`, the code computes the weighted cosine

`r_ij^m = cos(H_{0,i}^m ⊙ d_theta^m, H_{0,j}^m ⊙ d_theta^m)`.

The bounded relation weight is

`a_ij^m = a_min + (1-a_min) sigmoid(r_ij^m / temperature)`.

Only observed physical edges receive these weights. The graph operator `Ahat^m` is the symmetric GCN normalization of the weighted physical graph with the configured diffusion self-loops.

### Neighborhood proposal, adaptive gate, and state bank

At order `k`,

`N_{i,k}^m = [Ahat^m S_{k-1}^m]_i`.

The independent modality gate is

`g_{i,k}^m = sigmoid(f_g^m(H_{i,0}^m, N_{i,k}^m, |H_{i,0}^m-N_{i,k}^m|, H_{i,0}^m⊙N_{i,k}^m, p_k))`,

where `p_k` is the learnable order embedding used by the context-gate MLP. The state is

`S_{i,k}^m = (1-g_{i,k}^m)H_{i,0}^m + g_{i,k}^m N_{i,k}^m`.

This is an adaptive context-state recurrence; it is not the fixed-alpha anchored recurrence. The state bank is

`B_i^m = [S_{i,0}^m,S_{i,1}^m,...,S_{i,K}^m]`.

## 8. Stage II: interaction-aware multi-order context integration

### Order tokens and Q/K/V

The code adds a learned hop/order embedding `e_k^m` to each state:

`T_k^m = S_k^m + e_k^m`.

After layer normalization,

`Q=T W_Q`, `K=T W_K`, `V=T W_V`,

and the single-head attention logits are

`L_{i,q,k}^m = Q_{i,q}^m (K_{i,k}^m)^T / sqrt(d) + b_{i,k}^m`.

The retained legacy relation-order conditioning is lightweight:

`b_{i,k}^m = rho_m c_i^m beta_hat_k^m`,

where `c_i^m` is the standardized local relation context, `beta_hat_k^m` is the centered/normalized learned order profile, and `rho_m=sigmoid(theta_relation_scale_m)`. In code it is broadcast across the query-order axis and is omitted only when the explicit relation intervention is off.

The attention is `A=softmax(L)` unless an inference intervention replaces it. The interaction output and interacted state are

`I_{i,q}^m = sum_k A_{i,q,k}^m V_{i,k}^m`,

`S~_{i,q}^m = S_{i,q}^m + tanh(theta_hop_gate^m) I_{i,q}^m`.

### Preference and integration

The node/order preference response is computed from the interacted state:

`delta_{i,k}^m = (tanh(W_k^m S~_{i,k}^m) · v_k^m) / filter_rank`.

The final coefficient is

`eta_{i,k}^m = gamma_global,k + delta_gamma_k^m + delta_{i,k}^m`.

Canonical P2 directly composes the interacted states:

`Z_i^m = sum_k eta_{i,k}^m S~_{i,k}^m`.

Each modality then receives its own residual refinement,

`Z_ref^m = LayerNorm(Z^m + MLP_m(Z^m))`,

and only then are text and visual representations concatenated and passed through the final residual fusion MLP and output normalization. The NC classifier is task-facing and unchanged.

The canonical `gamma_global` initialization still uses the code's legacy `_monomial_to_anchored_cumulative(_make_global_prior(), alpha)` transform. The direct-prior version is a controlled diagnostic only and is not the canonical model.

## 9. Training objective

The NC training objective is the unchanged task-runner cross-entropy on official train nodes:

`L_NC = CE(classifier(Z_i), y_i)`.

The current encoder returns zero auxiliary loss; no contrastive, auxiliary, or extra regularization objective is introduced.

## 10. Experiments

- Scope: Movies, Toys, Grocery, ele-fashion, Reddit-S; NC only.
- Seeds: 42, 43, 44.
- Metrics: Accuracy and Macro-F1, mean ± population SD.
- Protocol: official split, shared optimizer/early stopping/checkpoint selection/classifier/task runner.
- Main comparison: historical baseline rows plus current canonical P2 Full.
- Framework controls: Full, Attribute Only, Plain Multi-Order Backbone, Raw Bank Mean, Raw Terminal.
- Appendix fine ablations: Uniform Relations, Global Context Gate, Uniform Integration, No Cross-Order Interaction; Fixed Gate 0.9 is omitted when no formal result exists.

## 11. Research questions

- RQ1: Do physical relations and contextualization demand differ across modalities and data contexts?
- RQ2: Does a unified relation-calibrated adaptive context-state/state-bank framework remain competitive under a strict architecture-matched control?
- RQ3: Are adaptive context assignment and cross-order integration computationally active in frozen-checkpoint interventions?

## 12. Table and figure placement

- Figure 1: empirical motivation; introduce before the method.
- Figure 2: two-stage architecture; immediately after the hypothesis.
- Table 1: overall NC benchmark.
- Table 2: framework-level controls.
- Figure 3: context-formation diagnostics.
- Figure 4: multi-order integration diagnostics.
- Appendix Table A1: fine-grained functional ablation.

## 13. Contributions

Use the exactly-three formulation in `PAPER_CONTRIBUTIONS.md`.

## 14. Limitations

The study is limited to five NC datasets in this round. Baseline table provenance is historical and lacks the original producing execution SHA. Fine-grained ablation deltas are mixed and modest. Prior-init control is single-seed on three datasets. Inference interventions are functional diagnostics, not causal identification. The raw order probe and M1 motivate heterogeneity but do not establish a universally optimal order policy.

## 15. Claims discipline

### Claims supported

- modality-dependent raw relation discrepancy on shared physical edges;
- heterogeneous contextualization demand in the M1 linear-probe study;
- dataset-/modality-dependent raw fixed-graph order profiles;
- a code-realized two-stage P2 computation chain;
- competitive five-NC results and mostly positive paired differences over the strict plain backbone;
- nonzero frozen-checkpoint effects of context gates and cross-order integration.

### Claims not supported

Do not claim universal superiority, that every module is indispensable, that semantic similarity is relation reliability, that an oracle preferred lambda is realizable, that the interventions are causal, or that one order is universally optimal. Do not discuss LP in this paper round.
