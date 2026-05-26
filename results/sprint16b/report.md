# Sprint 16B — S11.x scale CIFAR-100

CNN bigger (32, 64, 128). 100 classes. Drift noise=0.05.

## Aggregate

| condition | acc | surp_init | surp/drift ratio | speedup vs none |
|---|---|---|---|---|
| none | 0.0995±0.0026 | 0.32415±0.14507 | 130.67±58.48 | 1.0× |
| world | 0.0416±0.0020 | 0.03982±0.00584 | 16.05±2.36 | 8.1× |
| fresh_self | 0.0951±0.0072 | 0.00631±0.00155 | 2.54±0.62 | 51.4× |

## Verdict scale

- S11.3 MNIST : 103×, S15C MNIST drift-controlled : 74×, S14D CIFAR-10 MLP : 6.7×
- **S16B CIFAR-100 CNN drift-controlled** : **8.1×**
  → Magnitude réduite mais directionnelle robuste