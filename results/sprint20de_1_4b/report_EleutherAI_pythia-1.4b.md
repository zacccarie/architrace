# Sprint 20D — World→Self exaptation (LLM)

Model : `EleutherAI/pythia-1.4b`

α : 0.1 (sweet spot Sprint 20C)

## Aggregate par condition

| condition | val_pp | surp init→final | Δ pp vs none |
|---|---|---|---|
| none | 20.602±0.979 | 6.784→6.759 | +0.000 |
| fresh | 20.495±0.962 | 6.144→4.342 | -0.107 |
| world | 20.618±0.919 | 6.047→4.352 | +0.016 |
| gaussian | 20.719±0.936 | 5.633→4.188 | +0.117 |

## Verdict exaptation LLM

- world init surp : 6.047
- fresh init surp : 6.144
- **Speedup init surp : 1.02×**
- world val_pp : 20.618
- fresh val_pp : 20.495
- **Δ pp (world vs fresh) : +0.123**
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