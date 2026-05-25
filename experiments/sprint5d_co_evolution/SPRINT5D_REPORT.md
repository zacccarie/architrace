# Sprint 5D — Co-évolution observer × observed

**Date** : 2026-05-26
**Statut** : 🟡 Mécanique validée, dynamique sous-stimulée sur MNIST

## Setup

- **Observed** = VG-MLP, init [20,20], trigger via φ_id dynamoscope drop
- **Observer** = MutableVGCNNObserver, init channels=(4,8), trigger via SSVG plateau
- Boucle :
  ```
  pour chaque batch :
      VG-MLP forward + grad step
      render arch → image 64×64×3
      observer SSVG step (predict z[t] from z[t-1])
      observer encode → trace buffer
      if φ_id drop → mutation VG-MLP
      if SSVG plateau → mutation observer
  ```

## Résultats (MNIST × 3 seeds)

| seed | acc | observed muts | observer muts | observed topo | observer topo | phase-lock dist |
|---|---|---|---|---|---|---|
| 0 | 0.906 | 3 (3 acc) | 0 | [56, 20] | [4, 8] | — |
| 1 | 0.889 | 2 (1 acc) | 1 | [32, 20] | [8, 8] | 144 steps |
| 2 | 0.892 | 2 (1 acc) | 0 | [32, 20] | [4, 8] | — |

**Mean acc** : 0.896 ± 0.007

## Findings

### Mécanique fonctionne
- VG-MLP grandit comme dans Sprint 5B (2-3 mutations / 10 epochs)
- MutableVGCNNObserver capable de mutation (testé seed 1 : conv0 widen [4,8] → [8,8])
- Pipeline rendu+encode+SSVG step+trigger sans bug

### Observer sous-stimulé
- 2/3 seeds : observer **n'a pas grandi** du tout
- Sa SSVG loss converge SOUS le plateau threshold (5e-5) sans déclencher widen
- Observer 570 params (init) = capacité suffisante pour prédire trace MNIST observed à 4-niveaux de précision

### Phase-lock absent
- Seed 1 : observer mut step 300, observed muts step 132 + 420
- Distance min entre mutations : 132 steps → **pas de couplage temporel observable**
- Hypothèse phase-lock co-évolution non confirmée sur cette task

## Interprétation

**MNIST trop simple pour stresser observer**. Le réseau observed reste petit (~30k params), ses dynamiques de poids restent dans le régime que le micro observer (570p) peut capturer linéairement. Le SSVG converge vite et reste stable.

**Pour observer faire grandir** :
- Task plus dure (CIFAR-100) → observed plus complexe → trace plus riche → SSVG plus difficile
- OU augmenter plateau detection sensibilité
- OU forcer observer à des prédictions plus longues (multi-step rollout sur trace)

**Implication conceptuelle** :
La capacité **minimale d'auto-observation** est très basse. 570 params suffisent pour suivre la croissance d'un MLP 16k-45k params. C'est consistent avec Sprint 3 (random-init features géométriques riches). L'observateur n'a pas besoin de scale.

## Co-évolution : test redirigé vers Sprint 6

Pour vraiment voir co-évolution coupled, besoin task plus exigeante :
- Sprint 6 sur CIFAR-100 : 100 classes, VG-MLP devra grandir plus, trace plus complexe
- Devrait stresser observer + forcer plus de mutations des deux côtés

## Question ouverte

Si observer reste petit pendant que observed grandit, qu'est-ce que ça dit du **rapport observateur / observé** ?

Possibilité 1 : observateur minimal suffit pour systèmes simples → conscience-de-soi cheap (alignée Sprint 5C.1).

Possibilité 2 : observer indépendant ne capture pas vraiment la complexité de observed → mesure superficielle.

Pour discriminer : comparer φ_id calculé via observer mutable vs observer fixe taille moyenne. Si φ_id équivalent → poss 1. Si différent → poss 2.

→ Sprint 5D.1 candidat futur.

## Code livré

| Fichier | Rôle |
|---|---|
| `architrace/mutable_observer.py` | MutableVGCNNObserver avec widen + SSVG self-training |
| `experiments/sprint5d_co_evolution/coevolve.py` | runner co-évolution |
| `results/sprint5d/coevolution_report.md` | tableau résultats |

## Prochaine étape

Sprint 6 — CIFAR-100 robustness. Replication Sprint 2 + Sprint 5C.1 + ce Sprint 5D sur task harder pour activer co-évolution coupled.
