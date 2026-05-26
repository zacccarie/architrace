# Sprint 20B — FULL fine-tune (compare LoRA)

Model : `EleutherAI/pythia-410m`

Mode : full fine-tune (no LoRA)

LR : 5e-06

## Aggregate

| α | val_pp | surp init→final | trainable |
|---|---|---|---|
| 0.0 | 23.633±0.000 | 1.8574→1.8424 | 405,371,008 (Δ pp = +0.000) |
| 1.0 | 25.739±0.000 | 0.7831→0.0957 | 405,371,008 (Δ pp = +2.106) |

## Comparaison LoRA (Sprint 19)

| Model | LoRA Δ pp | Full Δ pp | Interpretation |
|---|---|---|---|
| EleutherAI/pythia-410m | +2.34 | +0.000 | **Full < LoRA** : capacité d'allocation libre = clé |