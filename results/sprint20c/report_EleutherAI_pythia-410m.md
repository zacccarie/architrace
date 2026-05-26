# Sprint 20C — Per-token InfoNCE contrastive

Model : `EleutherAI/pythia-410m`

Mode : LoRA + InfoNCE per-token (vs two-halves MSE Sprint 19)

## Aggregate

| α | val_pp | surp init→final | Δ pp vs α=0 |
|---|---|---|---|
| 0.0 | 27.572±0.000 | 6.688→6.676 | +0.000 |
| 0.1 | 27.478±0.000 | 6.321→4.709 | -0.094 |
| 1.0 | 29.467±0.000 | 6.280→4.440 | +1.896 |

## Comparaison MSE two-halves (Sprint 19)

| Model | MSE Δ pp | InfoNCE Δ pp | Verdict signal-alignment |
|---|---|---|---|
| EleutherAI/pythia-410m | +2.34 | +1.896 | Pas de différence : scale-problem fondamental |