# Sprint 18B — LLM recipe per-token + real text

Per-token self-coherence + real text (20 newsgroups) + α curriculum warmup.

## Aggregate

| α | val_perplexity | final_surp | Δ pp vs α=0 |
|---|---|---|---|
| 0.0 | 10.205±0.158 | 0.4374±0.0188 | +0.000 |
| 0.1 | 10.215±0.156 | 0.4783±0.0195 | +0.009 |
| 0.5 | 10.652±0.114 | 0.4100±0.0651 | +0.447 |
| 1.0 | 12.151±0.460 | 0.5245±0.1171 | +1.945 |

## Verdict

- **Baseline α=0** : val_pp = 10.205
- **Best α=0.0** : val_pp = 10.205 (Δ=+0.000)
  → **NEUTRE** : effet pas détecté même avec améliorations