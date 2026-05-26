# Sprint 14B — Lesion sur CNN (CIFAR-10)

## Aggregate

| lesion | acc | surp_init | surp_final | surp_mean | surp_max |
|---|---|---|---|---|---|
| **no_lesion** | 0.330±0.004 | 0.0039±0.0029 | 0.0043±0.0012 | 0.0032±0.0007 | 0.0071±0.0022 |
| **freeze_predictor** | 0.330±0.006 | 0.0039±0.0029 | 0.0036±0.0011 | 0.0030±0.0005 | 0.0065±0.0013 |
| **freeze_trunk** | 0.307±0.005 | 0.0039±0.0029 | 0.0009±0.0003 | 0.0012±0.0004 | 0.0040±0.0028 |
| **freeze_fusion** | 0.325±0.007 | 0.0039±0.0029 | 0.0024±0.0009 | 0.0018±0.0005 | 0.0052±0.0021 |
| **freeze_all_but_predictor** | 0.298±0.010 | 0.0039±0.0029 | 0.0003±0.0001 | 0.0006±0.0003 | 0.0039±0.0029 |

## Verdict

- **no_lesion surp_final** : 0.0043

- **freeze_predictor** : 0.0036 (ratio 0.82× no_lesion)
- **freeze_trunk** : 0.0009 (ratio 0.20× no_lesion)
- **freeze_fusion** : 0.0024 (ratio 0.56× no_lesion)
- **freeze_all_but_predictor** : 0.0003 (ratio 0.07× no_lesion)

- **freeze_all_but_predictor** : 0.0003
  → Predictor seul suffit pour maintenir self-coherence. Le reste est redondant.