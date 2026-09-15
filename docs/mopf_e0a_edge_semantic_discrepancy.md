# MoPF E0-A — Physical-edge Cross-modal Semantic Discrepancy

## Scope

E0-A is an inference-only, method-independent empirical analysis. For each
physical edge, it computes cosine similarity directly from the frozen input
text and visual features shared by the models:

```text
sim_text(i,j)   = cosine(x_text[i], x_text[j])
sim_visual(i,j) = cosine(x_visual[i], x_visual[j])
```

The analysis covers the five formal NC datasets: Movies, Toys, Grocery,
ele-fashion, and Reddit-S. `sports-copurchase` was not added because its
current configuration exposes only the LP edge split rather than a full
physical-edge NC graph through the standard NC loader. This does not change
the definition or interpretation of the five formal NC datasets.

The existing `src.data.load_mag_data` pipeline was reused. Its configured
undirected graph support was canonicalized as unique
`(min(i,j), max(i,j))` pairs. Self-loops were excluded. Thus, an edge present
as both `(i,j)` and `(j,i)` contributes once.

## Rank normalization and statistics

For each dataset and modality independently, similarities were converted to
empirical CDF ranks as `average_rank / num_physical_edges`, with average ranks
for ties. The opposite-quartile conditions were fixed as strict inequalities:

- text high / visual low: `rank_text > 0.75` and `rank_visual < 0.25`;
- visual high / text low: `rank_visual > 0.75` and `rank_text < 0.25`.

Spearman correlation is the Pearson correlation of the two average-rank
arrays. Pearson correlation of the raw cosine similarities is also reported.
All correlation and gap statistics use the complete physical-edge set. For
the figure only, each dataset was sampled with seed `20260914`, capped at
50,000 edges.

## Core results

### Correlation and rank-gap statistics

| Dataset | Nodes | Physical edges | Spearman text–visual | Pearson raw cosine | Mean rank gap | Median rank gap | q25 | q75 | q90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Movies | 16,672 | 80,401 | 0.203916 | 0.169479 | 0.291553 | 0.246241 | 0.109290 | 0.436064 | 0.614370 |
| Toys | 20,695 | 56,701 | 0.157253 | 0.108971 | 0.301320 | 0.258161 | 0.114196 | 0.451623 | 0.631082 |
| Grocery | 17,074 | 71,131 | 0.164085 | 0.139172 | 0.297604 | 0.252042 | 0.106606 | 0.450549 | 0.631124 |
| ele-fashion | 97,766 | 199,586 | 0.484795 | 0.460502 | 0.230190 | 0.189014 | 0.084809 | 0.335235 | 0.491109 |
| Reddit-S | 15,894 | 141,540 | -0.016890 | -0.015890 | 0.341045 | 0.309752 | 0.146915 | 0.506665 | 0.674198 |

### Opposite-quartile fractions

| Dataset | Text high / visual low | Visual high / text low |
|---|---:|---:|
| Movies | 0.041741 (4.174%) | 0.042848 (4.285%) |
| Toys | 0.039329 (3.933%) | 0.054426 (5.443%) |
| Grocery | 0.044791 (4.479%) | 0.050808 (5.081%) |
| ele-fashion | 0.023213 (2.321%) | 0.012100 (1.210%) |
| Reddit-S | 0.044553 (4.455%) | 0.076339 (7.634%) |

## Objective answers to the E0-A questions

1. The text–visual edge-semantic Spearman correlations are 0.203916
   (Movies), 0.157253 (Toys), 0.164085 (Grocery), 0.484795 (ele-fashion),
   and -0.016890 (Reddit-S).

2. The median rank gap is nonzero in all five datasets: 0.189014–0.309752.
   Four datasets are around 0.25 or higher, while ele-fashion is lower but
   still clearly nonzero. Mean rank gaps range from 0.230190 to 0.341045.

3. Both opposite-quartile edge types occur in every dataset. The smaller of
   the two reported fractions is 1.210% (ele-fashion visual-high/text-low);
   the other datasets show roughly 3.9%–7.6% per direction. This is evidence
   that the observation is not confined to a single selected dataset.

4. No dataset is an obvious null case under these descriptive criteria.
   `ele-fashion` is the least discrepant of the five: it has the highest
   correlation, the smallest median gap, and the smallest opposite-quartile
   fractions. Nevertheless, its median gap remains nonzero and both
   opposite-quartile regions are populated. `Reddit-S` is the strongest
   disagreement case by rank gap and has near-zero negative correlation.

5. **Highest defensible level for this five-dataset analysis: A (Strong), with
   scope qualification.** Cross-modal semantic relation disagreement is
   observed across all five datasets. The qualification is important: the
   magnitude is heterogeneous, this is not a significance test, and the
   result does not establish that either modality is better for downstream
   prediction. A more conservative restatement also follows directly from
   the data: physical-edge semantic utility is not perfectly aligned across
   modalities and modality-specific discrepancies are present.

## Feature and graph provenance

The loader and feature paths recorded below are also present in each
dataset's `summary.json`:

| Dataset | Loader/source | Text feature source | Visual feature source | Feature shapes (text / visual) |
|---|---|---|---|---|
| Movies | `load_mag_data`, MAGB | `/hdd1/DataInHere/YHF/data/Movies/TextFeature/Movies_roberta_base_512_mean.npy` | `/hdd1/DataInHere/YHF/data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy` | `(16672, 768)` / `(16672, 768)` |
| Toys | `load_mag_data`, MAGB | `/hdd1/DataInHere/YHF/data/Toys/TextFeature/Toys_roberta_base_512_mean.npy` | `/hdd1/DataInHere/YHF/data/Toys/ImageFeature/Toys_openai_clip-vit-large-patch14.npy` | `(20695, 768)` / `(20695, 768)` |
| Grocery | `load_mag_data`, MAGB | `/hdd1/DataInHere/YHF/data/Grocery/TextFeature/Grocery_roberta_base_256_mean.npy` | `/hdd1/DataInHere/YHF/data/Grocery/ImageFeature/Grocery_openai_clip-vit-large-patch14.npy` | `(17074, 768)` / `(17074, 768)` |
| ele-fashion | `load_mag_data`, MM-Graph | `/hdd1/DataInHere/YHF/data/ele-fashion/clip_feat.pt`, slice `[0:512]` | `/hdd1/DataInHere/YHF/data/ele-fashion/clip_feat.pt`, slice `[512:1024]` | `(97766, 512)` / `(97766, 512)` |
| Reddit-S | `load_mag_data`, MAGB | `/hdd1/DataInHere/YHF/data/Reddit-S/TextFeature/RedditS_roberta_base_100_mean.npy` | `/hdd1/DataInHere/YHF/data/Reddit-S/ImageFeature/RedditS_openai_clip-vit-large-patch14.npy` | `(15894, 768)` / `(15894, 768)` |

The configured graphs have `make_undirected=true` and `add_self_loops=false`.
After loader preprocessing, the directed edge-index counts were exactly twice
the canonical physical-edge counts for all five datasets; duplicate
undirected representations were therefore removed before similarity
calculation. No self-loop was included in the analysis.

## Audit status

The run was performed at git commit
`b080cb8058628eea3b9da708073f72a083d82a6e` on branch `vnext`.

- Training was not started.
- No optimizer step was executed.
- No model or MoPF checkpoint was loaded or written.
- No learned relation weight, learned embedding, `learned_diag_cos`, or
  checkpoint-derived representation was used.
- No label, including test label, was used to construct similarity, ranks,
  thresholds, or plots.
- No final MoPF architecture/configuration/protocol was modified.
- The 0.25 and 0.75 thresholds were fixed before examining the output.

## Files and command

Modified/added source and documentation:

- `scripts/run_mopf_e0a_edge_semantic_discrepancy.py`
- `docs/mopf_e0a_edge_semantic_discrepancy.md`

Run command:

```bash
conda run --no-capture-output -n yhf_env \
  python scripts/run_mopf_e0a_edge_semantic_discrepancy.py \
  --device cuda:0 \
  --output-root outputs/e0_empirical_motivation/edge_semantic_discrepancy
```

Outputs:

- `outputs/e0_empirical_motivation/edge_semantic_discrepancy/edge_discrepancy_summary.csv`
- `outputs/e0_empirical_motivation/edge_semantic_discrepancy/edge_semantic_discrepancy.png`
- `outputs/e0_empirical_motivation/edge_semantic_discrepancy/edge_semantic_discrepancy.pdf`
- `outputs/e0_empirical_motivation/edge_semantic_discrepancy/run_manifest.json`
- Per dataset: `<dataset>/summary.json`
- Per dataset: `<dataset>/edge_rank_plot_sample.csv` (50,000 rows each)

## Warnings and limitations

- The analysis is descriptive and reports no confidence intervals, bootstrap
  uncertainty, or hypothesis tests.
- Rank normalization is performed separately within each dataset and modality;
  raw cosine scales should not be compared across datasets.
- Physical edges are the unique undirected support recovered from the existing
  loader's preprocessed `edge_index`. This follows the frozen dataset config,
  but does not independently re-parse the graph files outside the project
  loader.
- The plotted density is based on the fixed 50,000-edge sample when a dataset
  has more edges; all CSV summary statistics remain full-edge statistics.
- The result supports an empirical relation-disagreement observation only. It
  does not select a model, tune a threshold, or imply downstream performance
  gains for either modality.

E0-B, E0-C, E0-D, and F2 were not started.
