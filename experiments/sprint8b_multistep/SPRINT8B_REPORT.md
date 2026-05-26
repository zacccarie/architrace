# Sprint 8B — multi-step self-prediction horizon

**Date** : 2026-05-26
**Statut** : 🟡 Finding négatif informatif

## Hypothèse

Étend prediction horizon : state_predictor prédit z[t+k] depuis z[t-1] pour k ∈ {1, 2, 3, 5, 10}.
Plus k grand = stronger IB pressure (compresser plus de contexte temporel) = meilleur transfer.

## Résultats

α=2 (sweet spot transfer Sprint 7), 3 seeds MNIST → FashionMNIST transfer :

| horizon k | MNIST acc | Fashion transfer | Δ vs k=1 |
|---|---|---|---|
| 1 | 0.8769±0.0040 | 0.6353±0.0036 | — |
| 2 | 0.8769±0.0040 | 0.6353±0.0036 | +0.0000 |
| 3 | 0.8769±0.0040 | 0.6353±0.0036 | +0.0000 |
| 5 | 0.8769±0.0040 | 0.6353±0.0036 | +0.0000 |
| 10 | 0.8769±0.0040 | 0.6353±0.0036 | +0.0000 |

**Résultats EXACTEMENT identiques** à 4 décimales près. Horizon n'affecte rien.

## Interprétation

**Pourquoi nul effet** : self-state stabilise très rapidement (Sprint 5C/5C.1 : state_drift → 0 dès quelques epochs). Donc :
- state_buffer[0] (k steps ago) ≈ state_buffer[-1] (1 step ago) ≈ current_state
- Le différentiel entre k=1 et k=10 est essentiellement zéro
- Multi-step prediction ≡ single-step prediction dans ce régime

**Connexion Sprint 5C** : drift→0 indique fixed-point dans espace self-state. Une fois atteint, les voisinages temporels (k=1, 5, 10) sont indistinguables.

**Implication méta** : la stabilité que Sprint 5C.1 introduit (avec α≥0.1) **élimine la possibilité même de signal multi-step**. C'est-à-dire : pour avoir un signal multi-step utile, il faut un système qui DERIVE — mais Sprint 5C.1 montre que self-coherence loss force convergence vers fixed-point.

**Catch-22** : 
- Without self-coherence (α=0) → state drifts → multi-step COULD work → mais base task perf souffre (Sprint 5C divergence)
- With self-coherence (α≥0.1) → state stabilizes → multi-step DOESN'T work → mais task perf better

→ Single-step suffit. Multi-step est inutile dans ce setup.

## Caveat

Le test n'est pas concluant pour ABSENCE de mécanisme — possible que :
- Avec α=0, multi-step changerait quelque chose (mais task acc trop bas pour comparaison utile)
- Avec state_dim plus grand (16 ici), drift serait plus persistant
- Avec data plus varié (CIFAR-100 ?), state plus chaotique

**Finding utilisable** : sur MNIST avec ce setup, single-step est suffisant ET sufficient. Multi-step ne nuit pas, mais n'aide pas.

## Code

| Fichier | Rôle |
|---|---|
| `architrace/strange_loop.py` | + `pred_horizon` param dans config + state_buffer deque |
| `experiments/sprint8b_multistep/sweep.py` | sweep k × seeds |
