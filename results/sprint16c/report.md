# Sprint 16C — DistilGPT-2 + auxiliary self-coherence

Fine-tune last 2 transformer blocks + heads on tiny corpus.

## Aggregate

| α | val_perplexity | final_surp | Δ pp vs α=0 |
|---|---|---|---|
| 0.0 | 2.799±0.814 | 18.3630±10.9876 | +0.000 |
| 0.5 | 10.620±10.581 | 0.5478±0.1925 | +7.821 |
| 1.0 | 8.016±6.372 | 0.2439±0.0312 | +5.217 |
| 2.0 | 6.148±0.556 | 0.1445±0.0063 | +3.350 |

## Verdict

- **Baseline α=0** : val_pp = 2.799
- **Best α=0.0** : val_pp = 2.799 (Δ=+0.000)
  → **NEUTRE** : effet pas détecté