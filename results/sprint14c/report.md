# Sprint 14C — LLM self-coherence (fixed)

Fix bug Sprint 12C : self-coherence gradient flows through shared backbone via cached prev pred.

## Aggregate

| α | val_perplexity | lm_loss | self_surprise | Δ pp vs α=0 |
|---|---|---|---|---|
| 0.0 | 29.20±0.29 | 3.356±0.001 | 0.0231±0.0057 | +0.000 |
| 0.1 | 29.19±0.29 | 3.356±0.001 | 0.0232±0.0058 | -0.001 |
| 0.5 | 29.19±0.29 | 3.356±0.001 | 0.0232±0.0058 | -0.000 |
| 1.0 | 29.19±0.29 | 3.356±0.001 | 0.0232±0.0057 | -0.001 |
| 2.0 | 29.19±0.29 | 3.356±0.001 | 0.0232±0.0058 | -0.001 |
| 5.0 | 29.19±0.29 | 3.356±0.001 | 0.0231±0.0058 | -0.001 |

## Verdict

- **Baseline α=0** : val_perplexity = 29.20
- **Best α=0.1** : val_perplexity = 29.19 (Δ=-0.001)
  → **ÉQUIVALENT** : self-coherence neutre sur LM toy task