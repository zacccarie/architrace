# Sprint 18C — S11.x cross-architecture vision

## Aggregate par (arch, condition)

| arch | condition | acc | surp_init | speedup vs none |
|---|---|---|---|---|
| resnet | none | 0.409±0.032 | 0.30239±0.08854 | 1.0× |
| resnet | world | 0.328±0.023 | 0.02355±0.01008 | 12.8× |
| resnet | fresh_self | 0.370±0.021 | 0.02115±0.01572 | 14.3× |
| vit | none | 0.345±0.004 | 0.10520±0.04038 | 1.0× |
| vit | world | 0.256±0.016 | 0.14610±0.00972 | 0.7× |
| vit | fresh_self | 0.342±0.006 | 0.01032±0.00362 | 10.2× |
| mixer | none | 0.393±0.008 | 0.05956±0.01315 | 1.0× |
| mixer | world | 0.280±0.013 | 0.01691±0.00712 | 3.5× |
| mixer | fresh_self | 0.406±0.014 | 0.00452±0.00017 | 13.2× |

## Verdict cross-arch

- **resnet** : world→self speedup = **12.8×**
- **vit** : world→self speedup = **0.7×**
- **mixer** : world→self speedup = **3.5×**

Rappel : S11.3 MNIST MLP = 103×, S15C drift-norm = 74×, S16B CNN CIFAR-100 = 8.1×