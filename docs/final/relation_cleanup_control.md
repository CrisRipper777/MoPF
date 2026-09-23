# P2 Legacy Relation-Order Bias Cleanup Control

P2-clean changes only `use_legacy_relation_order_bias=false`. MRC relation calibration, modality-specific normalized propagation, adaptive gates, direct interacted-state integration, preference, fusion, task head, split, and checkpoint criterion are unchanged. The clean variant is not set as the default.

| Dataset | P2 test acc/F1 | P2-clean test acc/F1 | Acc delta | F1 delta |
|---|---:|---:|---:|---:|
| Movies | 0.5700/0.4997 | 0.5742/0.5090 | +0.0042 | +0.0093 |
| Grocery | 0.8413/0.7660 | 0.8381/0.7554 | -0.0032 | -0.0106 |
| ele-fashion | 0.8824/0.7741 | 0.8846/0.7842 | +0.0022 | +0.0100 |

Clean architecture viable under the predeclared accuracy criterion (no dataset drops over 0.5 pp): `True`.

Interaction-off audit embedding MAE:

- Movies / P2: embedding MAE `0.136912`, logit MAE `0.229041`, flip `0.105566`
- Movies / P2-clean: embedding MAE `0.148213`, logit MAE `0.262046`, flip `0.115703`
- Grocery / P2: embedding MAE `0.169181`, logit MAE `0.315356`, flip `0.045215`
- Grocery / P2-clean: embedding MAE `0.167212`, logit MAE `0.311383`, flip `0.048085`
- ele-fashion / P2: embedding MAE `0.132074`, logit MAE `0.305836`, flip `0.038971`
- ele-fashion / P2-clean: embedding MAE `0.156474`, logit MAE `0.396348`, flip `0.036004`

This control is descriptive and does not justify deleting the legacy path from the default P2 candidate without broader qualification.
