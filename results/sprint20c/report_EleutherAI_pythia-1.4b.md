# Sprint 20C — Per-token InfoNCE contrastive

Model : `EleutherAI/pythia-1.4b`

Mode : LoRA + InfoNCE per-token (vs two-halves MSE Sprint 19)

## Aggregate

| α | val_pp | surp init→final | Δ pp vs α=0 |
|---|---|---|---|
| 0.0 | 20.602±0.979 | 6.784→6.759 | +0.000 |
| 0.1 | 20.495±0.962 | 6.144→4.342 | -0.107 |

## Comparaison MSE two-halves (Sprint 19)

| Model | MSE Δ pp | InfoNCE Δ pp | Verdict signal-alignment |
|---|---|---|---|