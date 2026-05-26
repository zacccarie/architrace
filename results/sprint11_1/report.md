# Sprint 11.1 — Internalisation Mead (other → self exaptation)

## Hypothèse

Pré-entraînement mutual modeling → self-coherence émerge plus vite en Phase 2 que pré-entraînement no-modeling. Si vrai, preuve directe d'**exaptation** du circuit other-modeling vers self-modeling (Mead).

## Aggregate (mean ± std sur agents A et B × seeds)

| condition | acc | surprise initial Phase 2 | surprise final | steps→0.05 | steps→0.02 |
|---|---|---|---|---|---|
| **none_to_self** | 0.855±0.007 | 1.4963±0.6935 | 0.0087±0.0021 | 25.9±2.9 | 33.0±3.4 |
| **mutual_to_self** | 0.861±0.005 | 0.0149±0.0102 | 0.0112±0.0029 | 0.0±0.0 | 1.5±2.2 |
| **fresh_self** | 0.865±0.006 | 0.0132±0.0070 | 0.0124±0.0032 | 0.0±0.0 | 0.1±0.3 |

## Per-run

| condition | seed | agent | acc | surp_init | surp_50 | surp_final | steps→0.05 |
|---|---|---|---|---|---|---|---|
| none_to_self | 0 | A | 0.857 | 1.3491 | 0.0064 | 0.0054 | 27 |
| none_to_self | 0 | B | 0.856 | 1.7433 | 0.0126 | 0.0126 | 26 |
| none_to_self | 1 | A | 0.844 | 2.6682 | 0.0135 | 0.0092 | 28 |
| none_to_self | 1 | B | 0.847 | 2.2105 | 0.0086 | 0.0064 | 31 |
| none_to_self | 2 | A | 0.864 | 1.0628 | 0.0083 | 0.0101 | 24 |
| none_to_self | 2 | B | 0.853 | 1.7331 | 0.0088 | 0.0089 | 27 |
| none_to_self | 3 | A | 0.853 | 0.7362 | 0.0104 | 0.0087 | 23 |
| none_to_self | 3 | B | 0.865 | 0.4668 | 0.0092 | 0.0084 | 21 |
| mutual_to_self | 0 | A | 0.868 | 0.0061 | 0.0097 | 0.0084 | 0 |
| mutual_to_self | 0 | B | 0.860 | 0.0310 | 0.0143 | 0.0137 | 0 |
| mutual_to_self | 1 | A | 0.850 | 0.0292 | 0.0155 | 0.0171 | 0 |
| mutual_to_self | 1 | B | 0.866 | 0.0092 | 0.0091 | 0.0085 | 0 |
| mutual_to_self | 2 | A | 0.863 | 0.0061 | 0.0093 | 0.0118 | 0 |
| mutual_to_self | 2 | B | 0.856 | 0.0051 | 0.0094 | 0.0103 | 0 |
| mutual_to_self | 3 | A | 0.863 | 0.0229 | 0.0131 | 0.0114 | 0 |
| mutual_to_self | 3 | B | 0.864 | 0.0094 | 0.0076 | 0.0083 | 0 |
| fresh_self | 0 | A | 0.866 | 0.0178 | 0.0098 | 0.0127 | 0 |
| fresh_self | 0 | B | 0.867 | 0.0270 | 0.0126 | 0.0199 | 0 |
| fresh_self | 1 | A | 0.858 | 0.0173 | 0.0145 | 0.0121 | 0 |
| fresh_self | 1 | B | 0.868 | 0.0075 | 0.0085 | 0.0085 | 0 |
| fresh_self | 2 | A | 0.875 | 0.0109 | 0.0125 | 0.0128 | 0 |
| fresh_self | 2 | B | 0.854 | 0.0027 | 0.0155 | 0.0110 | 0 |
| fresh_self | 3 | A | 0.860 | 0.0124 | 0.0093 | 0.0094 | 0 |
| fresh_self | 3 | B | 0.869 | 0.0099 | 0.0139 | 0.0127 | 0 |

## Verdict

- **Baseline (none→self)** : initial surprise=1.4963, steps→0.05=26
- **Pré-mutual (mutual→self)** : initial surprise=0.0149, steps→0.05=0
- **Δ initial surprise** : -1.4814 (négatif = mutual démarre avec moins de surprise)
- **Δ steps→0.05** : -25.9 (négatif = mutual converge plus vite)

  → **EXAPTATION CONFIRMÉE** : pré-mutual accélère self-coherence. Mead validé : other-modeling circuit se redirige vers self-modeling.

- **Plafond (fresh_self continuous)** : initial surprise=0.0132
- mutual→self atteint surprise initial Phase 2 vs fresh_self : 0.0149 vs 0.0132