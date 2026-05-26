# Sprint 19 — LLM scale 1B+ via LoRA

Model : `EleutherAI/pythia-6.9b`

Max steps : 40

## Aggregate

| α | val_pp | surp init→final | params total / trainable |
|---|---|---|---|
| 0.0 | 21.483±0.000 | 0.4426→0.4903 | 6,869,884,928 / 12,718,208 (Δ pp = +0.000) |
| 1.0 | 21.915±0.000 | 0.1899→0.0686 | 6,869,884,928 / 12,718,208 (Δ pp = +0.433) |

## Verdict scale-up

- Model size : **6870M params**
- Baseline α=0 : 21.483
- Best α=0.0 : 21.483 (Δ=+0.000)
  → **NEUTRE** : effet pas visible sur cette taille, peut-être encore trop petit.