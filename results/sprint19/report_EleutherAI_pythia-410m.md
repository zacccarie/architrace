# Sprint 19 — LLM scale 1B+ via LoRA

Model : `EleutherAI/pythia-410m`

Max steps : 60

## Aggregate

| α | val_pp | surp init→final | params total / trainable |
|---|---|---|---|
| 0.0 | 30.036±0.000 | 2.5406→2.2940 | 407,693,312 / 2,396,288 (Δ pp = +0.000) |
| 1.0 | 32.379±0.000 | 0.7351→0.1213 | 407,693,312 / 2,396,288 (Δ pp = +2.343) |

## Verdict scale-up

- Model size : **408M params**
- Baseline α=0 : 30.036
- Best α=0.0 : 30.036 (Δ=+0.000)
  → **NEUTRE** : effet pas visible sur cette taille, peut-être encore trop petit.