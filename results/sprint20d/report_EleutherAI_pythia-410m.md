# Sprint 20D — World→Self exaptation (LLM)

Model : `EleutherAI/pythia-410m`

α : 0.1 (sweet spot Sprint 20C)

## Aggregate par condition

| condition | val_pp | surp init→final | Δ pp vs none |
|---|---|---|---|
| none | 27.480±0.974 | 6.650→6.653 | +0.000 |
| fresh | 27.342±0.822 | 6.276→4.622 | -0.138 |
| world | 26.873±1.013 | 6.130→4.554 | -0.607 |

## Verdict exaptation LLM

- world init surp : 6.130
- fresh init surp : 6.276
- **Speedup init surp : 1.02×**
- world val_pp : 26.873
- fresh val_pp : 27.342
- **Δ pp (world vs fresh) : -0.469**
  → Pas d'exaptation détectée (LLM = exception comme ViT)

## Comparaison historique exaptation

| Substrat | Speedup |
|---|---|
| S11.3 MLP MNIST | 103× |
| S15C MNIST drift-norm | 74× |
| S16B CNN CIFAR-100 | 8.1× |
| S18C ResNet/Mixer | 8-12× |
| S18C ViT | 0.7× (inverse) |
| **S20D LLM pythia-410m** | **1.02×** |