# Sprint 20G — LLM-on-world-task + InfoNCE

Model : `EleutherAI/pythia-410m`

Task : next Lorenz coord regression (LLM via inputs_embeds)

Aux  : InfoNCE per-token sur hidden states

## Aggregate par α

| α | val_mse | task init→final | surp init→final | Δ mse vs α=0 |
|---|---|---|---|---|
| 0.0 | 0.02030±0.00744 | 1.283→0.0284 | 6.767→6.548 | +0.00000 |
| 0.1 | 0.03620±0.01871 | 1.286→0.0455 | 5.545→4.279 | +0.01590 |
| 1.0 | 0.15854±0.05555 | 1.255→0.1254 | 5.498→4.245 | +0.13824 |

## Comparaison régimes substrats

| Substrat | Task | Aux loss effet (Δ relative) |
|---|---|---|
| Vision MLP/CNN | Image class | Exapte 8-100× speedup |
| ViT | Image class | Inverse 0.7× |
| LLM text S20C | Next-token | -0.5% perplexity (marginal) |
| LLM warmup S20F | Text after Lorenz LoRA | +2% perplexity (degrade) |
| **LLM world task S20G** | Lorenz regression | **+0.0% MSE** (α=0.0 best) |