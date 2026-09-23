# Paper story v3

## Opening hypothesis

“Multimodal attributed graphs combine modality-intrinsic semantics with physical relational structure. However, a physical edge does not uniquely determine its semantic contribution to each modality, and the usefulness of neighborhood context varies across nodes, modalities, and structural ranges. Therefore graph contextualization in MAGs should be modeled as: (1) semantically conditioned context-state formation; (2) multi-order context-state utilization.”

Physical topology defines possible structural dependencies, whereas multimodal semantics determine how structural context is formed and utilized.

## Positioning

The paper is positioned as a **mechanism-supported unified framework with modest but mostly consistent gains over a strong architecture-matched multi-order backbone**. The evidence chain is:

1. Model-independent diagnostics show that the same physical relations can differ in raw text/visual semantics, contextualization demand is heterogeneous in M1, and fixed physical propagation order profiles vary by dataset and modality.
2. MGSC-MAG turns that observation into one MAG contextualization framework: modality-specific relation calibration and adaptive context-state formation feed a modality-specific multi-order context state bank, followed by interaction-aware integration and late fusion.
3. The five-NC benchmark and corrected controls show a competitive P2 row and mostly positive paired differences against the strict plain multi-order backbone, while the mixed raw readouts prevent a universal module-necessity claim.
4. Frozen-checkpoint interventions verify that gates and cross-order interaction are computationally active. They are functional diagnostics, not causal tests or retrained performance.

The narrative is therefore about conditional contextualization in multimodal graphs, not about adding three unrelated attention tricks and not about universal superiority.

## Evidence order

Figure 1 motivates the problem independently of MGSC. Figure 2 introduces the two-stage computation. Table 1 establishes the benchmark context and Table 2 establishes the framework-level control comparison. Figure 3 diagnoses context formation; Figure 4 diagnoses multi-order integration. Appendix Table A1 reports fine-grained functional removals without ranking them.

## Scope and boundaries

The current paper scope is NC on Movies, Toys, Grocery, ele-fashion, and Reddit-S with seeds 42/43/44. LP is outside this round. Text and visual propagation remain separate until late fusion. The frozen CoSI reference and the unified NC protocol remain unchanged.
