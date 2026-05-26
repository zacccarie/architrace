# Sprint 17B — LLM pretrain from scratch + aux loss

TinyGPT 4 layers 128d ~3M params. From scratch. Aux loss intégrée dès step 1.

## Aggregate

| α | val_perplexity | final_surp | params | Δ pp vs α=0 |
|---|---|---|---|---|
| 0.0 | 1.105±0.001 | 0.0831±0.0275 | 830,127 | +0.000 |
| 0.1 | 1.106±0.001 | 0.0021±0.0001 | 830,127 | +0.001 |
| 0.5 | 1.108±0.002 | 0.0017±0.0001 | 830,127 | +0.004 |
| 1.0 | 1.110±0.001 | 0.0014±0.0000 | 830,127 | +0.005 |
| 2.0 | 1.113±0.002 | 0.0012±0.0000 | 830,127 | +0.008 |

## Verdict

- **Baseline α=0** : val_pp = 1.105
- **Best α=0.0** : val_pp = 1.105 (Δ=+0.000)

Comparaison avec S16C (DistilGPT-2 fine-tune) : α=1 → val_pp +5.22 (DÉGRADATION).
Ici from-scratch α=0.0 → val_pp +0.000
  → **NEUTRE** : effet pas détecté