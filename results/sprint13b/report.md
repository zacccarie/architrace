# Sprint 13B — Compositional pre-training

## Aggregate

| condition | acc | surp_init Phase 2 | steps→0.05 | steps→0.02 |
|---|---|---|---|---|
| none | 0.873±0.005 | 1.8230±0.7663 | 27.2±2.9 | 40.0±4.4 |
| self | 0.879±0.004 | 0.0124±0.0090 | 0.0±0.0 | 0.2±0.4 |
| world | 0.884±0.005 | 2.0626±0.2521 | 35.8±9.3 | 54.2±5.8 |
| world_and_self | 0.862±0.007 | 0.0203±0.0172 | 0.0±0.0 | 14.0±24.2 |
| world_self_mutual | 0.870±0.004 | 0.0118±0.0039 | 0.0±0.0 | 0.0±0.0 |

## Verdict

- none → self : 1.8230 (baseline)
- self → self : 0.0124
- world → self : 2.0626
- world+self → self : 0.0203
- world+self+mutual → self : 0.0118

  → **SATURATION** : compositions ≈ single (0.0118 ≈ 0.0124). Substrate déjà saturé par 1 source. Redondance Spinoza modes confirmée.