# MoPF-vNext U2 Freeze

## Selected variant

**C1 — Semantic Anchor only**.

Formal configuration: `multihop_state_mode=anchored`,
`multihop_response_mode=cumulative`, `multihop_anchor_alpha=0.1`.

## Evidence and rejected alternatives

- C0 was the current cumulative baseline and reused the 15 U1-T T2
  best-validation checkpoints after the explicit/legacy cumulative equivalence
  audit.
- C1 passed the validation performance-safe gate and the trained Anchor
  mechanism gate. Anchor-off frozen interventions changed the learned eta,
  fused representation, and logits on all five datasets.
- C2 and C3 showed lower trained cross-hop absolute cosine/CKA redundancy and
  higher novelty/effective rank, but both exhibited strong coefficient
  compensation: multiple orders exceeded the matched cumulative coefficient
  magnitude by more than 10x. Differential candidates therefore failed the
  formal mechanism gate.
- Test metrics are final descriptive results only and were not used for
  checkpoint, variant, or hyperparameter selection.

## Formal semantics

For C1, `S_0=H` and `S_k=(1-alpha) P S_(k-1) + alpha H`; the conditioner
receives `S_k` and the response bank is the cumulative `S_k`. The C1 basis
initialization uses the exact triangular solve defined by `S=A M`, so the
initial filter function is equivalent to C0 within the required relative-L2
gate.

The formal K policy remains `Movies=3`, `Toys=3`, `Grocery=2`,
`ele-fashion=3`, `Reddit-S=3`. No alpha tuning, K tuning, LP, fusion change,
new loss, or U3 work is included in this freeze.

## Provenance

- Implementation/config freeze commit: `22d66d7` (`complete U2-C formal factorial verification`)
- U2-C artifacts: `outputs/u2c_formal_factorial_training/u2c_master_summary.json`
- U2-C report: `docs/mopf_u2c_formal_factorial_training.md`

## Next-stage interface

Any later U3 work must consume C1 as the frozen U2 NC interface and must not
silently re-enable C2/C3 differential responses or use their test metrics for
selection.
