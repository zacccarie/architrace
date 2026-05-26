# Sprint 15A — LLM two-halves design

Two-halves : pred(first_half) → second_half. Gradient flows through trunk via two paths.

## Aggregate

| α | val_perplexity | final_surp | Δ pp vs α=0 |
|---|---|---|---|
| 0.0 | 33.559±1.332 | 0.2366±0.0702 | +0.000 |
| 0.1 | 33.560±1.332 | 0.0010±0.0001 | +0.001 |
| 0.5 | 33.548±1.342 | 0.0008±0.0001 | -0.011 |
| 1.0 | 33.543±1.339 | 0.0007±0.0000 | -0.016 |
| 2.0 | 33.519±1.332 | 0.0007±0.0000 | -0.040 |

## Verdict

- **Baseline α=0** : 33.559
- **Best α=2.0** : 33.519 (Δ=-0.040)
  → **NEUTRE** : effet trop petit pour cette tâche