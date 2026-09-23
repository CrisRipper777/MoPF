# Prior-initialization controlled diagnostic

This diagnostic compares the two initializations of `gamma_global` while
keeping the P2 architecture, NC protocol, split, seed 42, optimizer, early
stopping, and task head unchanged. It is limited to Movies, Grocery, and
ele-fashion. The canonical P2 model was not changed.

- `legacy_anchored`: current parent initialization using the
  monomial-to-anchored-cumulative basis transform;
- `direct`: initialize `gamma_global` directly from `_make_global_prior()`.

| Dataset | Direct − legacy accuracy | Direct − legacy Macro-F1 |
|---|---:|---:|
| Movies | −0.12 pp | −1.53 pp |
| Grocery | −0.03 pp | −0.30 pp |
| ele-fashion | −0.05 pp | +0.48 pp |

The accuracy changes are small, but the Movies Macro-F1 decrease is not
negligible for a single seed. Therefore this control does not justify changing
the canonical P2 initialization to `direct` in the current round. A future
multi-seed prior study could revisit the basis choice; no such choice was made
using test results here.

Authoritative files:
`outputs/final/prior_init_control/summary.csv` and `deltas.csv`.
