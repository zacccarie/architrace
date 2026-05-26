# Sprint 12C — LLM self-coherence recipe

## Setup

TinyLM : embedding(64) + 2-layer Transformer + LM head + state_predictor.

Synthetic sequences markov-structured (4 clusters × 16 tokens).

Task : next-token prediction + auxiliary self-coherence (α).

## Aggregate

| α | lm_loss | val_perplexity | self_surprise | Δ perplexity vs α=0 |
|---|---|---|---|---|
| 0.0 | 3.356±0.001 | 29.17±0.26 | 0.0403±0.0028 | +0.000 |
| 0.1 | 3.356±0.001 | 29.17±0.26 | 0.0041±0.0011 | -0.001 |
| 0.5 | 3.356±0.001 | 29.17±0.26 | 0.0041±0.0011 | -0.001 |
| 1.0 | 3.356±0.001 | 29.17±0.26 | 0.0041±0.0011 | -0.001 |
| 2.0 | 3.356±0.001 | 29.17±0.26 | 0.0041±0.0011 | -0.001 |

## Per-run

| α | seed | lm_loss | val_perplexity | self_surprise |
|---|---|---|---|---|
| 0.0 | 0 | 3.355 | 28.89 | 0.0414 |
| 0.0 | 1 | 3.355 | 29.51 | 0.0364 |
| 0.0 | 2 | 3.358 | 29.12 | 0.0430 |
| 0.1 | 0 | 3.355 | 28.89 | 0.0056 |
| 0.1 | 1 | 3.355 | 29.51 | 0.0037 |
| 0.1 | 2 | 3.358 | 29.12 | 0.0030 |
| 0.5 | 0 | 3.355 | 28.89 | 0.0056 |
| 0.5 | 1 | 3.355 | 29.51 | 0.0037 |
| 0.5 | 2 | 3.358 | 29.12 | 0.0030 |
| 1.0 | 0 | 3.355 | 28.89 | 0.0056 |
| 1.0 | 1 | 3.355 | 29.51 | 0.0037 |
| 1.0 | 2 | 3.358 | 29.12 | 0.0030 |
| 2.0 | 0 | 3.355 | 28.89 | 0.0056 |
| 2.0 | 1 | 3.355 | 29.51 | 0.0037 |
| 2.0 | 2 | 3.358 | 29.12 | 0.0030 |

## Verdict

- **Baseline α=0** : val_perplexity = 29.17
- **Best α = 0.1** : val_perplexity = 29.17 (Δ=-0.001)
  → **ÉQUIVALENT** : self-coherence n'aide ni ne nuit LM dans ce setup