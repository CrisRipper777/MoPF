# MGSC-MAG P2 Pilot Report

This P2 pilot uses the five NC datasets (Movies, Toys, Grocery, ele-fashion, Reddit-S). P0 and P1 are the completed seed-42 rows from the P1 pilot; P2 enables only `direct_interacted_integration=true`. The task runner, splits, optimizer, stopping rule, classifier, and graph protocol are unchanged.

## NC performance

| Dataset | P0 test acc | P1 test acc | P2 test acc | P2-P1 |
|---|---:|---:|---:|---:|
| Movies | 0.5688 | 0.5673 | 0.5700 | +0.0027 |
| Toys | 0.7951 | 0.7903 | 0.7900 | -0.0002 |
| Grocery | 0.8360 | 0.8369 | 0.8413 | +0.0044 |
| ele-fashion | 0.8840 | 0.8822 | 0.8824 | +0.0001 |
| Reddit-S | 0.9638 | 0.9670 | 0.9657 | -0.0013 |

P2 test accuracy did not drop by more than 1 percentage point on any of these datasets: `True`. The pilot has mixed task-level movement (3/5 datasets improved over P1), so this is not treated as a universal performance win.

## Intervention audit

Embedding MAE, logit MAE, and prediction flip rate are measured against the same model's normal inference output. `relation_off_audit` is the requested frozen relation-context audit; it is descriptive and does not delete the legacy relation-context or relation-order-bias parameters.

| Dataset | Intervention | P1 emb MAE | P2 emb MAE | P1 logit MAE | P2 logit MAE | P1 flip | P2 flip |
|---|---|---:|---:|---:|---:|---:|---:|
| Movies | query_collapse | 0.000676195 | 0.00142562 | 0.00138273 | 0.00262487 | 0.000419866 | 0.00071977 |
| Movies | uniform | 0.00251226 | 0.0224797 | 0.00516427 | 0.0411727 | 0.00107965 | 0.0191939 |
| Movies | interaction_off | 0.00767162 | 0.136912 | 0.0154911 | 0.229041 | 0.00407869 | 0.105566 |
| Movies | relation_off_audit | 7.38534e-05 | 0.000780083 | 0.000150726 | 0.00143442 | 0 | 0.000659789 |
| Toys | query_collapse | 2.80999e-05 | 0.00152973 | 5.56664e-05 | 0.00280938 | 0 | 0.000773134 |
| Toys | uniform | 0.000121709 | 0.0201815 | 0.000236667 | 0.0369678 | 4.83208e-05 | 0.00715149 |
| Toys | interaction_off | 0.0011573 | 0.118136 | 0.00228712 | 0.186273 | 0.000531529 | 0.0282194 |
| Toys | relation_off_audit | 8.43903e-06 | 0.00072257 | 1.68654e-05 | 0.00132584 | 0 | 0.000241604 |
| Grocery | query_collapse | 0.00940692 | 0.00237736 | 0.0221934 | 0.00456166 | 0.00140565 | 0.000527117 |
| Grocery | uniform | 0.0197208 | 0.0272599 | 0.0465201 | 0.0532986 | 0.00462692 | 0.00673539 |
| Grocery | interaction_off | 0.036038 | 0.169181 | 0.084656 | 0.315356 | 0.00527117 | 0.0452149 |
| Grocery | relation_off_audit | 0.00119643 | 0.000463695 | 0.00284147 | 0.000937052 | 0.000351412 | 0.000117137 |
| ele-fashion | query_collapse | 5.05959e-05 | 0.000512689 | 9.72918e-05 | 0.00123142 | 0 | 7.15995e-05 |
| ele-fashion | uniform | 0.000423704 | 0.0144123 | 0.000811471 | 0.033882 | 3.06855e-05 | 0.00272078 |
| ele-fashion | interaction_off | 0.00238467 | 0.132074 | 0.00477907 | 0.305836 | 0.000347769 | 0.0389706 |
| ele-fashion | relation_off_audit | 3.35886e-05 | 0.00052779 | 6.77997e-05 | 0.00125933 | 1.02285e-05 | 5.11425e-05 |
| Reddit-S | query_collapse | 1.26505e-05 | 0.000687602 | 2.49262e-05 | 0.00135561 | 0 | 0 |
| Reddit-S | uniform | 3.01772e-05 | 0.0117797 | 5.96463e-05 | 0.0233518 | 0 | 0.000755002 |
| Reddit-S | interaction_off | 0.000756264 | 0.0752867 | 0.00144374 | 0.132448 | 6.29168e-05 | 0.00207626 |
| Reddit-S | relation_off_audit | 4.31543e-06 | 0.000901144 | 8.51517e-06 | 0.00186071 | 0 | 6.29168e-05 |

The `interaction_off` intervention effect is larger for P2 than P1 on all three reported measures for every dataset: `True`. This is direct evidence that cross-order interaction reaches the final representation more directly in P2.

Relation intervention off is near-zero under the strict descriptive check (all three metrics below 1e-6 / zero flip): `False`.
The relation-context audit is therefore retained as an active diagnostic; the legacy relation-context and relation-order-bias parameters are not removed in this round.

P2 is a mechanism pilot rather than a new selection protocol. The direct-integration mechanism should be retained for further study only where its intervention effect is measurably larger than P1 without a task-level regression; no claim of universal improvement is made here.
