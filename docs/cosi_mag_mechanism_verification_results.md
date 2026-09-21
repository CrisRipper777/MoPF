# CoSI-MAG Figure 4 Mechanism Verification Results

Checkpoint-only offline analysis using the user-selected `paper_nc_final_fixed_v1` Full checkpoint set. Results are descriptive. The selected checkpoint aggregate mismatch to the frozen NC table remains recorded in `checkpoint_audit.csv`; no checkpoints were substituted. Figure 4 mean±SD error bars use population SD across seeds (ddof=0), matching the frozen NC/LP tables. Stage-I/II raw seed rows and summary CSVs remain unchanged; displayed SDs below are recomputed from those seed rows only for presentation.

## (a) Modality-Specific Relation Calibration

### A. Raw numerical results

| Dataset | M_T mean ± SD | M_V mean ± SD | M_overall mean ± SD | E_T / E_V edges |
|---|---:|---:|---:|---:|
| Movies | 0.13079 ± 0.00395 | 0.02913 ± 0.01062 | 0.07996 ± 0.00438 | 3356 / 3445 |
| Toys | 0.09558 ± 0.00097 | 0.08580 ± 0.00428 | 0.09069 ± 0.00203 | 2230 / 3086 |
| Grocery | 0.10827 ± 0.00346 | 0.08244 ± 0.00345 | 0.09536 ± 0.00266 | 3186 / 3614 |
| ele-fashion | 0.12477 ± 0.00422 | 0.06754 ± 0.00121 | 0.09615 ± 0.00258 | 4633 / 2415 |
| Reddit-S | 0.06928 ± 0.00258 | 0.11902 ± 0.00625 | 0.09415 ± 0.00243 | 6306 / 10805 |

### B. Cross-dataset consistency

Text-specific margins were positive in 15/15 dataset-seed cases; visual-specific margins were positive in 15/15 cases.

### C. Reversals / exceptions

No non-positive directional seed-level margins. Relative margin magnitude reverses in Reddit-S (M_V > M_T), while both directional margins remain positive.

### D. Interpretation

Positive M_T and M_V indicate that learned raw physical-edge conductances favor the corresponding modality on the method-independent raw-feature relation sets. Any non-positive result is retained as a reversal.

### E. Safe paper claim

Across all 15 dataset-seed cases, both directional margins were positive (M_T: 15/15; M_V: 15/15); report these as checkpoint-level calibration evidence.

### F. Claims not supported

This panel does not establish that conductance calibration improves test performance, that every individual edge is correctly calibrated, or that the relation sets are causal explanations.

## (b) Semantic Retention under Multi-Hop Propagation

### A. Raw numerical results

| Dataset | Text ΔCKA mean ± SD at K | Visual ΔCKA mean ± SD at K |
|---|---:|---:|
| Movies | 0.12665 ± 0.00958 | 0.12472 ± 0.00549 |
| Toys | 0.11305 ± 0.00121 | 0.09857 ± 0.00042 |
| Grocery | 0.11787 ± 0.00260 | 0.11456 ± 0.00127 |
| ele-fashion | 0.09126 ± 0.00470 | 0.12171 ± 0.00225 |
| Reddit-S | 0.05574 ± 0.00149 | 0.02319 ± 0.00042 |

### B. Cross-dataset consistency

Retention gain was positive in 30/30 dataset-modality-seed cases.

### C. Reversals / exceptions

No non-positive seed-level endpoint gains. Visual retention gain exceeds Text in ele-fashion; the modality ordering is not uniform across datasets.

### D. Interpretation

The comparison isolates the recurrence on a checkpoint's shared H0 and learned normalized operator. Positive gain means the anchored state has higher linear CKA with H0 at the formal endpoint than its ordinary counterfactual.

### E. Safe paper claim

At formal K, the anchored recurrence had higher CKA with H0 than the same-checkpoint ordinary counterfactual in all 30/30 dataset-modality-seed cases; describe this as a functional effect of the recurrence.

### F. Claims not supported

Higher CKA is not evidence that semantic retention causes better classification performance, and this comparison does not establish a universal benefit outside the selected checkpoints and formal datasets.

## (c) Adaptive Contribution Reallocation

### A. Raw numerical results

The previous statistic compared the adaptive contribution profile directly with a flat `1/(K+1)` profile. That confounded learned coefficient composition with unequal response norms: even uniform coefficients can yield a non-flat effective contribution profile. The revised primary statistic compares adaptive and uniform coefficients on the exact same Full-checkpoint response bank, so response-norm heterogeneity is held fixed.

| Dataset | Modality | Mean D across seeds ± population SD | Median D | Q1 | Q3 | IQR | Fraction D>0.01 | Fraction D>0.05 | Fraction D>0.10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | Text | 0.33743 ± 0.05385 | 0.33897 | 0.32499 | 0.35022 | 0.02523 | 1.00000 | 1.00000 | 1.00000 |
| Movies | Visual | 0.36185 ± 0.04132 | 0.36068 | 0.34862 | 0.37239 | 0.02376 | 1.00000 | 1.00000 | 1.00000 |
| Toys | Text | 0.57179 ± 0.05925 | 0.57246 | 0.56342 | 0.58056 | 0.01714 | 1.00000 | 1.00000 | 1.00000 |
| Toys | Visual | 0.65016 ± 0.01830 | 0.64912 | 0.64254 | 0.65713 | 0.01458 | 1.00000 | 1.00000 | 1.00000 |
| Grocery | Text | 0.37084 ± 0.01016 | 0.37104 | 0.35775 | 0.38374 | 0.02599 | 1.00000 | 1.00000 | 1.00000 |
| Grocery | Visual | 0.59503 ± 0.01307 | 0.59475 | 0.58598 | 0.60387 | 0.01788 | 1.00000 | 1.00000 | 1.00000 |
| ele-fashion | Text | 0.28123 ± 0.01598 | 0.27917 | 0.27067 | 0.29095 | 0.02028 | 1.00000 | 1.00000 | 1.00000 |
| ele-fashion | Visual | 0.27356 ± 0.02305 | 0.27082 | 0.26291 | 0.28154 | 0.01863 | 1.00000 | 1.00000 | 1.00000 |
| Reddit-S | Text | 0.38504 ± 0.08795 | 0.38480 | 0.37958 | 0.39032 | 0.01073 | 1.00000 | 1.00000 | 1.00000 |
| Reddit-S | Visual | 0.42772 ± 0.06536 | 0.42909 | 0.42473 | 0.43214 | 0.00741 | 1.00000 | 1.00000 | 1.00000 |

The seed-level mean, median, quartiles, IQR, and threshold fractions are summarized across seeds; Q1/Q3 are the means of the per-seed node-population quartiles. Seed is the replicate unit. The optional coefficient-only C diagnostic is present in the per-seed CSV/NPZ but is not used in the figure or primary interpretation.

### B. Cross-dataset consistency

For all 30 dataset×modality×seed populations, every node had D>0.10. Seed-level mean D values are descriptive; no cross-dataset magnitude ranking is claimed because Grocery uses formal K=2 while the other datasets use K=3.

### C. Reversals / exceptions

No dataset-modality has near-zero reallocation under the prespecified D>0.01 threshold: the fraction above 0.01 is 1.0 in every seed. This is a node-distribution description, not a node-level significance result.

### D. Interpretation

D=0 means the learned coefficients leave the effective order-contribution distribution unchanged relative to uniform coefficients on the same response bank. Larger D means stronger contribution reallocation beyond what response norms alone would produce. The synthetic QA case with uniform eta and an unequal-norm response bank returns D=0 within numerical tolerance.

### E. Safe paper claim

For the selected Full checkpoints, learned coefficients reallocate effective contribution relative to the same-bank uniform-coefficient counterfactual: all 30 seed-level node populations had fraction D>0.10 equal to 1.0. Report this as descriptive mechanism evidence, without ranking datasets by D.

### F. Claims not supported

This result does not establish that larger reallocation is optimal, improves test performance, or generalizes beyond these runs. Nodes are not treated as independent model replicates, and no node-level significance tests are performed.
