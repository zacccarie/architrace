# Sprint 11.2 — Symétrie exaptation self → mutual

## Hypothèse

Sprint 11.1 : **mutual → self** = exaptation parfaite (100× speedup).
Question : direction inverse aussi ?

- Si symétrique : "any modeling helps any modeling" (banalisé)
- Si asymétrique (self → mutual NE marche PAS) : primauté évolutionnaire sociale

## Aggregate (mean ± std sur seeds)

| condition | acc moy | mutual_surprise initial Phase 2 | final | steps→0.05 | steps→0.02 |
|---|---|---|---|---|---|
| **none_to_mutual** | 0.855±0.006 | 1.4732±0.5750 | 0.0092±0.0006 | 26.2±2.9 | 34.2±2.9 |
| **self_to_mutual** | 0.863±0.004 | 0.0172±0.0097 | 0.0116±0.0010 | 0.0±0.0 | 1.2±1.6 |
| **fresh_mutual** | 0.863±0.004 | 0.0121±0.0040 | 0.0118±0.0010 | 0.0±0.0 | 0.0±0.0 |

## Per-run

| condition | seed | acc_a | acc_b | mutual_surp_init | final | steps→0.05 |
|---|---|---|---|---|---|---|
| none_to_mutual | 0 | 0.857 | 0.860 | 1.5415 | 0.0090 | 27 |
| none_to_mutual | 1 | 0.844 | 0.848 | 2.2542 | 0.0084 | 30 |
| none_to_mutual | 2 | 0.870 | 0.851 | 1.4651 | 0.0095 | 26 |
| none_to_mutual | 3 | 0.857 | 0.855 | 0.6320 | 0.0099 | 22 |
| self_to_mutual | 0 | 0.867 | 0.861 | 0.0294 | 0.0129 | 0 |
| self_to_mutual | 1 | 0.855 | 0.862 | 0.0060 | 0.0104 | 0 |
| self_to_mutual | 2 | 0.876 | 0.861 | 0.0096 | 0.0122 | 0 |
| self_to_mutual | 3 | 0.862 | 0.858 | 0.0238 | 0.0109 | 0 |
| fresh_mutual | 0 | 0.866 | 0.858 | 0.0187 | 0.0132 | 0 |
| fresh_mutual | 1 | 0.856 | 0.866 | 0.0083 | 0.0107 | 0 |
| fresh_mutual | 2 | 0.874 | 0.865 | 0.0096 | 0.0123 | 0 |
| fresh_mutual | 3 | 0.865 | 0.855 | 0.0120 | 0.0112 | 0 |

## Verdict — symétrie ou asymétrie ?

- **none → mutual (baseline)** : surprise initial = 1.4732, steps→0.05 = 26.2
- **self → mutual (test)** : surprise initial = 0.0172, steps→0.05 = 0.0
- **Δ initial surprise** (self - none) : -1.4560
- **Δ steps→0.05** (self - none) : -26.2

### Comparaison avec Sprint 11.1 (mutual → self)

Sprint 11.1 : none→self surprise initial = 1.496 ; mutual→self = 0.015 → ratio 100×

- **Plafond (fresh_mutual continuous)** : surprise initial = 0.0121
- **Ratio none→mutual / self→mutual** : 85.7× (comparable au 100× de Sprint 11.1 ?)

  → **SYMÉTRIE** : self→mutual accélère autant que mutual→self. Any modeling helps any modeling. Finding banalisé : pas de primauté du social.