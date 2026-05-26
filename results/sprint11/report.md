# Sprint 11 — Dyadic mutual modeling

## Aggregate par condition (mean ± std)

| condition | α | acc_a | acc_b | surp_a | surp_b | alignment | transfer_a | transfer_b |
|---|---|---|---|---|---|---|---|---|
| **none** | 0.0 | 0.867±0.009 | 0.860±0.007 | 1.7935±0.4550 | 2.1232±0.3158 | — | 0.611±0.007 | 0.616±0.009 |
| **mutual** | 1.0 | 0.867±0.009 | 0.861±0.006 | 0.0106±0.0021 | 0.0139±0.0089 | — | 0.626±0.000 | 0.618±0.007 |
| **mutual_strong** | 2.0 | 0.862±0.012 | 0.861±0.003 | 0.0112±0.0058 | 0.0112±0.0069 | — | 0.629±0.004 | 0.620±0.011 |

## Per-run

| cond | seed | acc_a | acc_b | surp_a | surp_b | align | transfer_a | transfer_b |
|---|---|---|---|---|---|---|---|---|
| none | 0 | 0.862 | 0.862 | 1.7803 | 1.9724 | — | 0.606 | 0.628 |
| none | 1 | 0.860 | 0.867 | 2.3572 | 2.5626 | — | 0.607 | 0.612 |
| none | 2 | 0.879 | 0.851 | 1.2430 | 1.8345 | — | 0.621 | 0.608 |
| mutual | 0 | 0.867 | 0.861 | 0.0123 | 0.0203 | — | 0.627 | 0.621 |
| mutual | 1 | 0.855 | 0.868 | 0.0076 | 0.0199 | — | 0.626 | 0.624 |
| mutual | 2 | 0.878 | 0.854 | 0.0119 | 0.0013 | — | 0.626 | 0.608 |
| mutual_strong | 0 | 0.864 | 0.861 | 0.0190 | 0.0181 | — | 0.635 | 0.631 |
| mutual_strong | 1 | 0.847 | 0.865 | 0.0052 | 0.0139 | — | 0.626 | 0.623 |
| mutual_strong | 2 | 0.876 | 0.858 | 0.0093 | 0.0017 | — | 0.627 | 0.605 |

## Verdict — hypothèse dyadique

- **Baseline (α=0)** : acc=0.864, transfer=0.614
- **Mutual (α=1)** : acc=0.864 (Δ=+0.000), transfer=0.622 (Δ=+0.008)
  → **CONFIRMÉ** : mutual modeling améliore transfer (+0.8%)

- Pas assez de samples alignment pour comparer