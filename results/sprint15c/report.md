# Sprint 15C — S11.x avec drift égalisé

Noise injecté = 0.05 → drift target uniforme across conditions.

## Aggregate

| condition | acc | surp_init Phase 2 | drift_inj | surp/drift ratio |
|---|---|---|---|---|
| none | 0.852±0.014 | 1.41291±0.51578 | 0.00253±0.00000 | 559.466±204.232 |
| world | 0.724±0.028 | 0.01902±0.00459 | 0.00253±0.00000 | 7.533±1.818 |
| fresh_self | 0.856±0.011 | 0.01258±0.00284 | 0.00253±0.00000 | 4.982±1.126 |

## Verdict

- **none → self** : surp_init=1.41291, ratio=559.466
- **world → self** : surp_init=0.01902, ratio=7.533
- **Ratio brut speedup (surp)** : 74.3×
- **Ratio normalized (surp/drift)** : 74.3×

  Sprint 11.3 MNIST sans noise : ratio 103×.
  → **EFFET ROBUSTE** : exaptation persiste même avec drift égalisé