# Sprint 19 — LLM scale 1B+ via LoRA

Model : `EleutherAI/pythia-1.4b`

Max steps : 40

## Aggregate

| α | val_pp | surp init→final | params total / trainable |
|---|---|---|---|
| 0.0 | 25.515±0.000 | 0.8746→0.9119 | 1,419,366,400 / 4,788,352 (Δ pp = +0.000) |
| 1.0 | 26.583±0.000 | 0.3583→0.0884 | 1,419,366,400 / 4,788,352 (Δ pp = +1.068) |

## Verdict scale-up

- Model size : **1419M params**
- Baseline α=0 : 25.515
- Best α=0.0 : 25.515 (Δ=+0.000)
  → **NEUTRE** : effet pas visible sur cette taille, peut-être encore trop petit.