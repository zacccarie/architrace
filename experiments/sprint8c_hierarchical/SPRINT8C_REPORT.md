# Sprint 8C — observers hiérarchiques

**Date** : 2026-05-26
**Statut** : ✅ Finding paradoxal — récursion d'observation = redondance

## Hypothèse

Empiler des observers (L1 observe L0, L2 observe L1, ...) ajoute-t-il info méta-méta ?

## Setup
- L0 : MutableVGCNNObserver observe `render(VG-MLP)` (Sprint 5B)
- L1 : MutableVGCNNObserver observe `render(L0)`
- L2 : MutableVGCNNObserver observe `render(L1)`
- Chaque observer SSVG-train sur sa séquence d'images

## Résultats (3 niveaux × 3 seeds × MNIST)

| niveau | SSVG init → final | % drop | Converge ? |
|---|---|---|---|
| **L0** (observe VG-MLP) | 0.077 → 0.017 | -78% | ✅ |
| **L1** (observe L0) | 0.057 → **0.929** | +1531% | ❌ |
| **L2** (observe L1) | 0.140 → 1.05 | instable | ❌ |

| paire niveaux | Pearson corr encodings |
|---|---|
| L0 ↔ L1 | **0.95-0.98** (forte) |
| L1 ↔ L2 | 0.51-0.98 |

Final observed acc constant ~0.84-0.85 (hiérarchie n'affecte pas observed).

## Interprétation paradoxale

**Le paradoxe** :
- L1 ne peut **PAS** prédire son input (SSVG explose × 16)
- Mais L1 encode une trajectoire **quasi-identique** à L0 (corr 0.95)
- → Information **structurelle** persiste, mais prédictibilité **disparaît**

**Mécanisme probable** :
1. L0 a une trajectoire d'encodage qui varie avec entraînement de VG-MLP
2. L1 observe L0 qui CHANGE simultanément avec son propre apprentissage
3. Cible mobile de L1 = sa propre input dépend de ses propres weights (à un niveau de méta de plus)
4. L1 ne peut pas converger : ses weights changent → input change → cible bouge
5. Mais L1 reflète passivement la structure de son input (corrélation haute) sans la prédire

**Conclusion forte** :
> **Récursion d'observation = redondance structurelle + perte de prédictibilité.**
> Empiler des observers ne crée pas d'information méta-méta, juste des copies dégradées.

## Lien théorique

**Gödel-Hofstadter renforcé** : système ne peut pas auto-modéliser récursivement. À chaque niveau de méta supplémentaire, prédictibilité s'effondre — même si la structure visible persiste.

**Différence avec Sprint 5C** :
- Sprint 5C : auto-référence INTERNE (système contient sa propre représentation comme partie de son input)
- Sprint 8C : observation EXTERNE empilée
- Les deux montrent **incohérence prédictive** à l'auto-référence/méta-observation
- Sprint 5C : self-surprise diverge (auto-modèle faux)
- Sprint 8C : higher-level SSVG diverge (méta-observer pas convergent)

**Lien Wolfram** : computational irreducibility — observer la trajectoire d'un système ne permet pas de la prédire sans la simuler. Empiler des observers ne crée pas de raccourci.

## Implication pratique

Pour analyse dynamoscope d'un système qui pousse :
- **Une observation suffit** (L0 = direct render → VG-CNN encode)
- Pas la peine d'empiler des observers
- Mécanisme d'auto-observation est **plat** dans sa puissance, pas hiérarchique

## Synthèse Sprint 5D + 8C

| Phénomène | Sprint | Finding |
|---|---|---|
| Observer mutable | 5D | rarement mute sur MNIST, capacité minimale suffit |
| Observer hierarchical | 8C | redondance inter-niveaux, méta-méta nul |
| Self-reference interne | 5C | auto-modèle diverge par défaut |
| Self-reference contrainte | 5C.1 | cheap quand explicit |

**Pattern unifié** : structure d'observation peut être minimale et plate. Empiler ou complexifier ne crée pas d'info supplémentaire dans ce régime de tâches.

## Code

| Fichier | Rôle |
|---|---|
| `experiments/sprint8c_hierarchical/hierarchical.py` | runner 3 niveaux × 3 seeds |
| `results/sprint8c/report.md` | tableau résultats |
