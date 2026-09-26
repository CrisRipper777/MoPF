# E5 C0 reuse and action-landscape preflight

Screening C0 checks completed: 9/9. No Toys/Grocery C0 checkpoint or split data was loaded before the winner lock.

All route supervision/audit rows are Train or Validation only. Test indices/labels were never indexed. Frozen C0 uses the stored projectors, normalized physical propagation, plain fusion, and classifier.

## C0 reproduction

| Dataset | Seed | Embedding max abs. error | Probability max abs. error | Stored/reloaded Val Acc | Stored/reloaded Val Macro-F1 | Forward <1e-6 | Metrics <1e-6 |
|---|---:|---:|---:|---:|---:|---|---|
| Movies | 42 | 0 | 0 | 0.573185/0.573185 | 0.489836/0.489836 | True | True |
| Movies | 43 | 0 | 0 | 0.577684/0.577684 | 0.501133/0.501133 | True | True |
| Movies | 44 | 0 | 0 | 0.576185/0.576185 | 0.491287/0.491287 | True | True |
| Reddit-S | 42 | 0 | 0 | 0.959107/0.959107 | 0.917636/0.917636 | True | True |
| Reddit-S | 43 | 0 | 0 | 0.964140/0.964140 | 0.926534/0.926534 | True | True |
| Reddit-S | 44 | 0 | 0 | 0.960365/0.960365 | 0.919180/0.919180 | True | True |
| ele-fashion | 42 | 0 | 0 | 0.874911/0.874911 | 0.752275/0.752275 | True | True |
| ele-fashion | 43 | 0 | 0 | 0.876547/0.876547 | 0.752800/0.752800 | True | True |
| ele-fashion | 44 | 0 | 0 | 0.875831/0.875831 | 0.757388/0.757388 | True | True |

## Action-space/headroom summary

See action_landscape_summary.csv for per-dataset/seed/split/modality action frequencies, entropy, preference strength, oracle CE gain, and label-oracle accuracy upper bound. The upper bound is not an achievable router score.

See action_stability.csv for split-matched cross-seed hard-action agreement, soft-teacher JS/cosine, and preference-strength Spearman.

## Data provenance

- Frozen C0 source: outputs/cmrf_discovery_v1/reference_uniform/checkpoints/.
- The five action embeddings share C0's frozen plain fusion and classifier; modality-specific action landscapes hold the opposite modality at AU.
- Train teachers use softmax(-CE) without temperature and preference strength 1-H(q)/log(5). Validation teachers occur only in audit columns and never enter the training objective.
- Raw action rows and tensor caches are under outputs/utility_routing_audit_v1/; no E3/E4 output was overwritten.
