# Sprint 8C — observers hiérarchiques

## Per-run

| n_lvls | seed | acc | levels SSVG | inter-level corr |
|---|---|---|---|---|
| 1 | 0 | 0.850 | L0:0.0093 | — |
| 1 | 1 | 0.841 | L0:0.0295 | — |
| 1 | 2 | 0.847 | L0:0.0147 | — |
| 2 | 0 | 0.844 | L0:0.0097 / L1:2.5601 | 0↔1=0.95 |
| 2 | 1 | 0.835 | L0:0.0161 / L1:0.2052 | 0↔1=0.98 |
| 2 | 2 | 0.848 | L0:0.0135 / L1:0.0556 | 0↔1=0.97 |
| 3 | 0 | 0.842 | L0:0.0104 / L1:2.6034 / L2:0.0284 | 0↔1=0.95 / 1↔2=0.51 |
| 3 | 1 | 0.839 | L0:0.0250 / L1:0.0959 / L2:0.0266 | 0↔1=0.97 / 1↔2=0.98 |
| 3 | 2 | 0.848 | L0:0.0226 / L1:0.0536 / L2:3.0925 | 0↔1=0.97 / 1↔2=0.96 |

## Aggregate par niveau profondeur

| level | mean SSVG final | std SSVG final | converge ? |
|---|---|---|---|
| L0 | 0.0168±0.0069 | -0.0602 init delta | YES |
| L1 | 0.9290±1.1698 | +0.8720 init delta | NO (stagnant) |
| L2 | 1.0492±1.4448 | +0.9372 init delta | NO (stagnant) |

## Verdict

- L0 SSVG : 0.0769 → 0.0168
- L1 SSVG : 0.0570 → 0.9290
- L0 % drop : 78.2%  ·  L1 % drop : -1531.1%
- **L1 stagne** : récursion saturée, méta-méta-info marginal