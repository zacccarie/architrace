# Sprint 13C — Causal lesion test

## Aggregate

| lesion | acc | surp_init | surp_final | surp_mean | surp_max |
|---|---|---|---|---|---|
| **no_lesion** | 0.861±0.010 | 0.0097±0.0024 | 0.0126±0.0025 | 0.0112±0.0015 | 0.0151±0.0027 |
| **freeze_predictor** | 0.863±0.007 | 0.0097±0.0024 | 0.0105±0.0023 | 0.0095±0.0015 | 0.0122±0.0020 |
| **freeze_trunk** | 0.828±0.010 | 0.0097±0.0024 | 0.0066±0.0019 | 0.0076±0.0016 | 0.0128±0.0033 |
| **freeze_both** | 0.828±0.010 | 0.0097±0.0024 | 0.0108±0.0046 | 0.0102±0.0038 | 0.0129±0.0040 |

## Verdict

- **no_lesion** : surp 0.0097 → 0.0126
- **freeze_predictor** : surp 0.0097 → 0.0105
  → **CIRCUIT DISTRIBUÉ** : trunk peut maintenir self-coherence même avec predictor gelé. Mécanisme distribué.

- **freeze_trunk** acc impact : 0.861 → 0.828 (Δ=-0.033)