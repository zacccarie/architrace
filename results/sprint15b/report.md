# Sprint 15B — Décomposition self-surprise

Self-surprise = MSE(pred, target). DRIFT = MSE(target_t, target_t-1). PRED_QUALITY = 1 - surprise/drift.

Si pred_quality > 0 → predictor mieux qu'identity. Si ≈ 0 → predictor = identity. Si < 0 → pire.

## Aggregate

| condition | acc | surprise | drift | pred_quality |
|---|---|---|---|---|
| **no_lesion** | 0.864±0.005 | 0.01113±0.00272 | 0.00000±0.00000 | -535790.184±584435.180 |
| **freeze_predictor** | 0.861±0.005 | 0.00923±0.00161 | 0.00000±0.00000 | -616797.954±466308.491 |
| **freeze_trunk** | 0.830±0.011 | 0.00621±0.00152 | 0.00000±0.00000 | -466370.340±376789.680 |

## Verdict — artéfact ou vrai distribution ?

- **no_lesion** : drift=0.00000, pred_quality=-535790.184
- **freeze_predictor** : drift=0.00000 (ratio 0.62× no_lesion), pred_quality=-616797.954 (Δ=-81007.770)
  → **VRAI IMPACT** : pred_quality dégrade (-81007.770). Mécanisme perturbed.
- **freeze_trunk** : drift=0.00000 (ratio 0.77× no_lesion), pred_quality=-466370.340 (Δ=+69419.844)