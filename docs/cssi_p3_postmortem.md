# CSSI P3 post-mortem

This audit uses the existing same-LR best-validation checkpoints only; no training or test evaluation is performed.

## 1. Gradient-starvation calculation

P3 low-rank correction is `delta R = 1e-3 * W_up(tanh(W_c C) * W_down R)`. Both `W_up.weight` and `W_up.bias` are zero-initialized. At initialization, the derivative with respect to `W_down` and the conditioner parameters is multiplied by `W_up.weight`, hence is exactly zero. The `W_up.weight` gradient can be nonzero, but the controller/downstream response path cannot update on the first step. This is a genuine initialization gradient-starvation mechanism, not merely a small-gradient observation.

`response_up_text` and `response_up_visual` are affine `Linear` layers: the implementation permits a bias. Therefore the general P3 expression is `delta R = s[U(a*V R)+b]`, not strictly `s U(a*V R)`. In the actual P3 checkpoints both weight and bias were zero-initialized, so the initial correction is zero and the bias does not create a nonzero correction at initialization. The fixed `1e-3` scale further makes the active path under-powered after the up projection begins to move.

## 2. Same-controller first-layer block norms

| modality | block | frobenius_norm | relative_to_sum |
| --- | --- | --- | --- |
| text | absolute_difference | 2.389639 | 0.254413 |
| text | paired_other | 2.343234 | 0.249519 |
| text | product | 2.321033 | 0.247165 |
| text | target | 2.337516 | 0.248903 |
| visual | absolute_difference | 2.372130 | 0.252153 |
| visual | paired_other | 2.350064 | 0.249813 |
| visual | product | 2.335365 | 0.248249 |
| visual | target | 2.349911 | 0.249785 |

These norms are descriptive parameter diagnostics only; they are not causal evidence.

## 3. Cross-condition and low-rank amplitude shifts

| modality | order | condition_normal_off_mean | condition_normal_shuffle_mean | amplitude_normal_off_mean | amplitude_normal_shuffle_mean | correction_shift_normal_shuffle_mean | correction_norm_normal_mean | cross_correction_shift_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 1 | 9.517196 | 1.090729 | 0.214104 | 0.052790 | 0.000005 | 0.008658 | 0.010982 |
| text | 2 | 10.161129 | 0.878948 | 0.215884 | 0.049145 | 0.000035 | 0.003423 | 0.002713 |
| text | 3 | 10.185917 | 0.819373 | 0.216533 | 0.046962 | 0.000000 | 0.002072 | 0.001168 |
| visual | 1 | 11.285612 | 1.303959 | 0.222823 | 0.049463 | 0.000215 | 0.003508 | 0.026661 |
| visual | 2 | 12.132001 | 0.906188 | 0.203612 | 0.024327 | 0.000044 | 0.001388 | 0.003760 |
| visual | 3 | 12.062534 | 0.950154 | 0.202655 | 0.022168 | 0.000011 | 0.000841 | 0.002025 |

The CSV contains dataset/seed/modality/order rows. A nonzero shift confirms that paired information is represented in the trained controller; it does not by itself establish task utility or causality.

## 4. Files

- `outputs/cssi_p3_postmortem/same_controller_blocks.csv`
- `outputs/cssi_p3_postmortem/cross_condition_diagnostics.csv`
- `outputs/cssi_p3_postmortem/gradient_postmortem.json`
