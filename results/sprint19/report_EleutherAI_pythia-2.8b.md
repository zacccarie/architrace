# Sprint 19 — LLM scale 1B+ via LoRA

Model : `EleutherAI/pythia-2.8b`

Max steps : 40

## Aggregate

| α | val_pp | surp init→final | params total / trainable |
|---|---|---|---|
| 0.0 | 22.890±0.000 | 0.4101→0.4520 | 2,783,073,280 / 7,950,464 (Δ pp = +0.000) |
| 1.0 | 23.694±0.000 | 0.2263→0.0854 | 2,783,073,280 / 7,950,464 (Δ pp = +0.804) |

## Verdict scale-up

- Model size : **2783M params**
- Baseline α=0 : 22.890
- Best α=0.0 : 22.890 (Δ=+0.000)
  → **NEUTRE** : effet pas visible sur cette taille, peut-être encore trop petit.