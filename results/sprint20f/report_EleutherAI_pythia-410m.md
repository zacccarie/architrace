# Sprint 20F — LLM whole-model world warmup

Model : `EleutherAI/pythia-410m`

α : 0.1  World steps : 100

## Aggregate par condition

| condition | val_pp | surp init→final | Δ pp vs none |
|---|---|---|---|
| none | 27.480±0.974 | 6.650→6.653 | +0.000 |
| fresh | 27.342±0.822 | 6.276→4.622 | -0.138 |
| world_full | 28.006±0.968 | 6.323→4.594 | +0.526 |

## Verdict whole-model world warmup

- world_full init surp : 6.323
- fresh init surp : 6.276
- **Speedup init surp : 0.99×**
- world_full val_pp : 28.006
- fresh val_pp : 27.342
- **Δ pp (world_full vs fresh) : +0.664**

→ Pas de bénéfice net. LLM substrate vraiment incompatible avec exaptation S11.x.

## Comparaison S20D (predictor-only warmup)

| condition | S20D Δ pp (410M) | S20F Δ pp |
|---|---|---|
| none | 0.000 | +0.000 |
| fresh | -0.138 | -0.138 |
| world (predictor-only S20D / whole-model S20F) | -0.607 | +0.526 |