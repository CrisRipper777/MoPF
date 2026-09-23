# Figure 1 evidence audit

## Panel (a): Modality-dependent relation discrepancy

- **Input:** row-normalized raw text/visual features and actual physical edges from the final data loader; edge-level absolute difference between modality cosine similarities.
- **Model independence:** no CoSI/MGSC projection, checkpoint, learned relation weight, label, or graph rewiring.
- **Labels:** none.
- **Split:** not applicable; this is an edge-level descriptive statistic over the physical graph.
- **Supports:** the same physical relation can have different raw modality semantic similarity.
- **Does not support:** relation reliability, learned usefulness, or causal edge selection.

## Panel (b): Heterogeneous contextualization demand

- **Input:** M1 raw row-normalized local features, physical-neighbor means, and normalized mixtures for λ ∈ {0, .25, .50, .75, 1}.
- **Model independence:** only a shared linear probe is trained separately for each condition; no MGSC checkpoint or gate is used.
- **Labels/split:** official train is split deterministically into probe-train/probe-dev; official validation is read once after probe-dev CE selection; official test labels are never indexed.
- **Supports:** node-level demand for local versus neighborhood context differs across datasets/modalities.
- **Does not support:** the node-wise oracle as a realizable policy or test performance.

## Panel (c): Raw fixed-graph order profile

- **Input:** raw modality features propagated through a fixed symmetric-normalized physical graph at orders 0–3, followed by the same linear probe protocol.
- **Model independence:** no MGSC training, projection, MRC, gate, attention, or checkpoint.
- **Labels/split:** probe-train/probe-dev inside official train; official validation only for the reported profile; test labels unused.
- **Supports:** order utility/profile differs by dataset and modality in this lightweight diagnostic.
- **Does not support:** node-level universal preferred order or downstream MGSC performance attribution.

The authoritative panel data are under `outputs/final/paper_figures/figure1_data/`. The probe manifest records the GPU/device and protocol; the output is a motivation diagnostic, not a model benchmark.
