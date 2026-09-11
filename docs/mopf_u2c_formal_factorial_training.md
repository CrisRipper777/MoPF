# MoPF-vNext U2-C — Formal Factorial Training Verification

NC-only formal factorial training under the frozen U1-T relation (`learned_diag_cos`, `tau=0.35`). Test metrics are descriptive and were not used for checkpoint, variant, or hyperparameter selection.

- Runtime commit: `fbd6962125467d7d4276deff420c150af4a84696`
- Remote sync: `failed: git fetch/pull could not resolve github.com in this environment`
- Initialization gate: **True**
- Formal protocol: hidden 256, dropout 0.2, AdamW, lr 1e-3, weight decay 1e-4, 300 epochs, patience 30, best Validation Accuracy.

## Downstream summary

| Variant | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 |
|---|---:|---:|---:|---:|
| C0 | 0.811962 ± 0.131162 | 0.752452 ± 0.139252 | 0.806942 ± 0.136826 | 0.743735 ± 0.138661 |
| C1 | 0.811487 ± 0.130841 | 0.749074 ± 0.138923 | 0.807652 ± 0.134258 | 0.744107 ± 0.135487 |
| C2 | 0.811648 ± 0.131439 | 0.748897 ± 0.140633 | 0.806652 ± 0.135842 | 0.741342 ± 0.140489 |
| C3 | 0.812203 ± 0.129820 | 0.748282 ± 0.139148 | 0.808160 ± 0.133450 | 0.743083 ± 0.135955 |

## Factorial effects

- Anchor effect (equal-dataset mean): Val Acc `0.000040`, Val Macro-F1 `-0.001996`.
- Differential effect (equal-dataset mean): Val Acc `0.000201`, Val Macro-F1 `-0.002173`.
- Interaction (equal-dataset mean): Val Acc `0.001030`, Val Macro-F1 `0.002763`.

## Mechanism gates

- Anchor mechanism after training: **True**; datasets passing `5/5`; frozen anchor-off intervention datasets `5`.
- Differential mechanism after training: **False**; datasets passing `5/5`.
- Eta compensation pathology: **True**; multi-hop contribution collapse: **False**.

## Final decision

**C1**

The decision uses validation metrics and mechanism evidence only. Differential coordinates change the organization/conditioning of the same degree-K propagation span; they do not expand the receptive field or claim exact k-hop unique information.
