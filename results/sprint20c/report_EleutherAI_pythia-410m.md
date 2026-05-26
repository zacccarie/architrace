# Sprint 20C — Per-token InfoNCE contrastive

Model : `EleutherAI/pythia-410m`

Mode : LoRA + InfoNCE per-token (vs two-halves MSE Sprint 19)

## Aggregate

| α | val_pp | surp init→final | Δ pp vs α=0 |
|---|---|---|---|
| 0.0 | 27.480±0.974 | 6.650→6.653 | +0.000 |
| 0.05 | 27.354±0.904 | 6.277→4.634 | -0.126 |
| 0.1 | 27.342±0.822 | 6.276→4.622 | -0.138 |
| 0.2 | 27.488±0.879 | 6.264→4.578 | +0.008 |
| 0.3 | 27.482±0.853 | 6.258→4.537 | +0.002 |
| 0.5 | 27.967±0.723 | 6.242→4.480 | +0.487 |

## Comparaison MSE two-halves (Sprint 19)

| Model | MSE Δ pp | InfoNCE Δ pp | Verdict signal-alignment |
|---|---|---|---|