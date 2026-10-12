# CoSI-MAG VNext Environment Audit

Audit date: 2026-10-12 (Asia/Shanghai)

## Repository and working tree

- Project root: `/hdd1/DataInHere/YHF/MoPF_IAMOC` (the parent workspace contains several unrelated projects).
- Branch: `exp/cosi-mag-vnext-crc-cheb-rcmi`, tracking `origin/exp/cosi-mag-vnext-crc-cheb-rcmi`.
- Remote target and benchmark base both resolve to `d5a06103be0a27952ef5890e2839a36eb83f580a`; the target branch had no prior commits beyond that base at audit time.
- The initial worktree was clean. `git fetch origin` first failed under the restricted shell because DNS was unavailable; the explicitly authorized fetch then succeeded.
- Repository instructions: `CLAUDE.md` describes the MAG task interfaces, while its historical `MAG_baseline/` working-directory reference is stale for this checkout. Actual configs resolve data under `/hdd1/DataInHere/YHF/data`.

## Runtime and resources

- Project environment `yhf_env`: Python 3.12.13, PyTorch 2.4.0+cu121, PyG 2.7.0, torch_sparse 0.6.18+pt24cu121, Hydra 1.3.2, OmegaConf 2.3.0, NumPy 2.4.3, scikit-learn 1.8.0, DGL 2.4.0+cu121.
- CUDA is available inside `yhf_env`. Two NVIDIA GeForce RTX 3090 devices are visible through PyTorch; at audit, GPU 0 had about 0.75 GiB free and GPU 1 about 23.4 GiB free. `nvidia-smi`/NVML is unavailable in this session, so memory will be measured through PyTorch where possible.
- Host memory: 125 GiB total, 110 GiB available. Workspace filesystem: about 133 GiB free (89% used).
- The base shell is not the project runtime: it has PyTorch 2.5.1, PyG 2.0.1, no Hydra, and a NumPy/scikit-learn ABI error. All project checks and runs must use `conda run -n yhf_env`.

## Protocol and data availability

- NC runner supports full-graph encoder training and selects checkpoints by validation accuracy; `infer_all_embeddings` calls the model's full forward for `inference_mode=full` and the model-specific `inference` for `layerwise`.
- LP uses train-only message edges with `LinkNeighborLoader`; current positive supervision edges and reverse directions are removed from sampled `edge_index` on CPU before moving the batch to the model/device. Sampler depth honors `requires_full_lp_sampler_depth`.
- The loader concatenates MAGB features in `[text, visual]` order; MM-Graph joint features are split in the same order. LP graph construction uses train positives only and does not add loader self-loops.
- Required Movies, Grocery, Reddit-S, and Sports-Copurchase files and seed-42 splits are present. The three NC split SHA-256 values and Sports-Copurchase LP split SHA-256 are recorded in the VNext run manifest.
- Existing unified protocols are `unified_full_graph_nc_v1` and `unified_sampled_lp_v1`; smoke runs must be clearly marked as development runs.

## Initial constraints

- Run on `cuda:1` for headroom; collect peak allocated memory with PyTorch.
- Keep generated datasets, checkpoints, and full logs under ignored/local `outputs/`; only small manifests, summaries, code, and reports belong in Git.
- No model code had been changed when this audit was written.
