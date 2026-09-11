# MoPF-vNext U3-A — Node–Modality Multi-Hop Preference Diagnosis

Analysis-only diagnosis over the frozen U2-C C1 checkpoints. No model was trained, no formal architecture was changed, and Test metrics are descriptive only.

- Runtime commit: `95101aec162769a7c9746298660fd53967e6b3ca`
- Source checkpoints: `15` C1 best-validation checkpoints
- Checkpoint bytes unchanged: `True`
- Canonical eta reconstruction max abs: `3.725e-09`

## Evidence questions

- Q1 modality preference: **True** across `5/5` datasets.
- Q2 node personalization: **True** across `5/5` datasets.
- Q3 node–preference alignment by NodeShuffle: `5/5` datasets.
- Q4 state association: see the per-seed Spearman table; effect sizes and seed consistency are reported rather than p-values alone.

## U3-A gate

**A. Both modality and node adaptation are functionally supported**

Proceed to U3-B with the C1 three-level hierarchy retained; test only minimal mechanism-driven composition changes, beginning with contribution/state-conditioned coefficient composition and no new router by default.

Full per-checkpoint coefficient statistics, contribution profiles, entropy, counterfactuals, associations, and quartile stratification are in the authoritative JSON and CSV artifacts.
