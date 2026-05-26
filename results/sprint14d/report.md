# Sprint 14D — S11.x universalité scale CIFAR-10

MLP CIFAR-10 (3072d input), 8 seeds.

## Aggregate

| condition | acc | surp_init | surp_final | steps→0.05 |
|---|---|---|---|---|
| none_to_self | 0.325±0.017 | 0.2586±0.0670 | 0.0076±0.0026 | 12.5±4.2 |
| world_to_self | 0.164±0.033 | 0.0385±0.0192 | 0.0093±0.0074 | 7.0±12.2 |
| fresh_self | 0.333±0.010 | 0.0053±0.0024 | 0.0041±0.0006 | 0.0±0.0 |

## Verdict — S11.x tient sur scale CIFAR-10 ?

- none → self  : surp_init = 0.2586
- world → self : surp_init = 0.0385
- **Ratio speedup** : 6.7×

  Sprint 11.3 MNIST avait ratio 103×.
  → **AFFAIBLI** : effet présent mais diminué