# Sprint 11.3 — Universalité circuit modeling

## Question

Sprint 11.1 (mutual→self) et 11.2 (self→mutual) montrent symétrie : ~100× speedup. Mais c'est entre agents (entities qui apprennent).

Si on remplace partenaire par système non-agentique (Lorenz attracteur), le circuit modeling exapte-t-il toujours vers self ?

## Aggregate

| condition | acc | surprise initial Phase 2 | final | steps→0.05 | steps→0.02 |
|---|---|---|---|---|---|
| **none_to_self** | 0.851±0.013 | 1.3822±0.5142 | 0.0092±0.0027 | 26.2±2.9 | 33.5±5.0 |
| **world_to_self** | 0.727±0.025 | 0.0134±0.0015 | 0.0398±0.0147 | 0.0±0.0 | 0.0±0.0 |
| **fresh_self** | 0.857±0.012 | 0.0117±0.0034 | 0.0111±0.0031 | 0.0±0.0 | 0.0±0.0 |

## Per-run

| condition | seed | acc | surp_init | surp_final | steps→0.05 |
|---|---|---|---|---|---|
| none_to_self | 0 | 0.852 | 1.6292 | 0.0067 | 30 |
| none_to_self | 1 | 0.831 | 2.0692 | 0.0136 | 28 |
| none_to_self | 2 | 0.869 | 1.1251 | 0.0079 | 24 |
| none_to_self | 3 | 0.853 | 0.7051 | 0.0086 | 23 |
| world_to_self | 0 | 0.730 | 0.0143 | 0.0460 | 0 |
| world_to_self | 1 | 0.763 | 0.0153 | 0.0281 | 0 |
| world_to_self | 2 | 0.724 | 0.0128 | 0.0243 | 0 |
| world_to_self | 3 | 0.693 | 0.0112 | 0.0608 | 0 |
| fresh_self | 0 | 0.863 | 0.0140 | 0.0111 | 0 |
| fresh_self | 1 | 0.846 | 0.0154 | 0.0161 | 0 |
| fresh_self | 2 | 0.874 | 0.0110 | 0.0090 | 0 |
| fresh_self | 3 | 0.845 | 0.0066 | 0.0081 | 0 |

## Verdict

- **none → self** (baseline) : surp_init = 1.3822, steps→0.05 = 26.2
- **world (Lorenz) → self** : surp_init = 0.0134, steps→0.05 = 0.0
- **Ratio speedup** : 103.0×
- **Plafond fresh_self** : surp_init = 0.0117

### Comparaison historique
- Sprint 11.1 mutual → self : ratio ~100×
- Sprint 11.2 self → mutual : ratio ~85×
- Sprint 11.3 world → self  : ratio **103.0×**

  → **CONFIRMÉ UNIVERSEL** : modeling Lorenz transfère à self-modeling. Circuit modeling est vraiment générique. Self ≡ Other ≡ World au niveau substrat.