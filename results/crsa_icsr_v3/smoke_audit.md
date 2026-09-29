# CRSA + ICSR v3 Smoke Audit

- Contexts complete: **3 / 3**.
- OOM/NaN/Inf log hits: **none**.
- Validation metrics and strict checkpoint reload: **pass**.
- Finite backward/gradient check: **True** (preflight audit).
- Test evaluation: **NO**; LP: **NO**.

| Context | Runtime (s) | Peak GPU MiB | Best epoch | Val Acc | Val Macro-F1 |
|---|---:|---:|---:|---:|---:|
| smoke/full/Movies/seed42 | 2.5109172998927534 | 3181.20068359375 | 1 | 0.3293341398239136 | 0.02477436823104693 |
| smoke/full/Reddit-S/seed42 | 2.3596379209775478 | 4622.30908203125 | 1 | 0.21201635897159576 | 0.07491838507358865 |
| smoke/full/ele-fashion/seed42 | 3.670037634903565 | 8523.31494140625 | 1 | 0.5338038206100464 | 0.1683838308091812 |
