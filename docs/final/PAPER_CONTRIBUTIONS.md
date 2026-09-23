# Paper contributions

The paper should state exactly three contributions:

1. **Empirical/problem perspective.** We show that a shared physical relation can have modality-dependent semantic discrepancy, contextualization demand varies across datasets, modalities, and nodes in a model-independent probe, and fixed physical propagation order profiles differ across dataset/modality paths. These observations are unified as conditional formation and utilization of structural context in MAGs, rather than presented as three independent problems.

2. **Unified framework.** We introduce MGSC-MAG, which combines modality-specific relation calibration with adaptive context-state formation, a modality-specific multi-order context state bank, cross-order interaction, and interaction-aware integration before late multimodal fusion. The contribution is the coherent computation chain, not a collection of unrelated modules.

3. **Evidence-oriented evaluation.** We evaluate five NC datasets under a shared strong benchmark protocol, compare against an architecture-matched plain multi-order backbone and framework-level controls, and provide frozen-checkpoint mechanism diagnostics for context assignment and cross-order integration. The controls are reported without claiming that every component is indispensable.
