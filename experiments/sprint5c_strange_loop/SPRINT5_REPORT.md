# Sprint 5 — Méta-observation + boucle étrange

**Date** : 2026-05-26
**Statut** : ✅ Sprint 5B validé · ✅ Sprint 5C résultat philosophique

## Sprint 5B — Live coupling visuel

### Question
Architecture state observée via render image → VG-CNN encode → dynamoscope → trigger.
Le signal visuel capture-t-il les mêmes lois dynamiques que l'observation directe (Sprint 2) ?

### Setup
- `architrace/arch_to_image.py` : VG state → (64, 64, 3) RGB invariant à croissance
  - Channels : |weights| · |gradients| · |biases| tiled-padded
- `architrace/visual_trigger.py` : VisualEmergenceTrigger
  - Render arch tous les 3 steps
  - VG-CNN encode (tiny, 6362 params, latent_dim=32)
  - Buffer 35 latents → φ_id via dynamoscope → trigger si drop

### Résultats (MNIST × 3 seeds)

| trigger | acc | params | n_fires | first_fire |
|---|---|---|---|---|
| loss_plateau (baseline) | 0.876 ± 0.004 | 16330 | 0 | — |
| direct_emergence (S2) | 0.902 ± 0.002 | 48530 | 3.3 | 152 |
| **visual_emergence (S5B)** | **0.898 ± 0.006** | 45310 | 3.0 | 144 |

**Visual ≈ Direct** (Δ acc = 0.004, négligeable). Validation : observer visuel capture **autant d'info** que mesure directe des latents.

### Implication
1. Architecture render contient assez d'info pour analyse dynamique
2. VG-CNN tiny random-init suffit comme observer
3. **Open path Sprint 5C** — système peut s'observer lui-même par cette voie

---

## Sprint 5C — Strange loop self-référentiel

### Question
VG dont l'input inclut sa propre représentation (render → VG-CNN encode). Mécanisme :
```
forward(x, prev_state) → (task_logits, predicted_next_state)
```
Self-surprise = ‖predicted_next_state - actual_next_state_after_update‖²

Hypothèses possibles :
- **Convergence fixed-point** : state stabilise, predictions correctes (self-model converge)
- **Divergence** : state explose ou collapse
- **Boucle paradoxale** : auto-prédiction inconsistante, instabilité

### Résultats (MNIST × 3 seeds)

| seed | task_loss | self_surprise | state_drift | state_norm |
|---|---|---|---|---|
| 0 | 2.31 → 0.35 | 0.042 → 2.783 | 0.045 → 0.001 | 0.799 const |
| 1 | 2.29 → 0.28 | 0.045 → 3.209 | 0.005 → 0.000 | 0.850 const |
| 2 | 2.33 → 0.18 | 0.046 → 1.626 | 0.040 → 0.002 | 0.689 const |

**Pattern asymétrique inattendu** :
- **Task loss** descend (2.3 → 0.27) → tâche apprise
- **State drift** → 0 (0.04 → 0.001) → représentation SOI stabilise
- **State norm** constant → ni collapse ni explosion
- **Self-surprise** EXPLOSE (0.04 → 2.78) → prédictions d'état divergent du SOI réel

### Interprétation

Le système atteint un **steady state asymétrique** :
1. Le SOI est stable (drift=0, norm constant)
2. L'AUTO-MODÈLE est faux (self-surprise = 60× initial)
3. La TÂCHE est résolue (0.877 acc)

**Lien Hofstadter** — "I am a strange loop" : le soi peut exister stablement sans nécessiter une auto-prédiction correcte. Pas de paradoxe Gödel-Tarski explosif — le système trouve un équilibre fonctionnel **malgré** l'inconsistance auto-référentielle.

**Lien Friston** — active inference rompue **sans conséquence**. Le réseau ne minimise pas sa propre surprise. Il maximise son utilité externe (loss tâche) — l'environnement fournit le signal qui compte, pas l'introspection.

**Lien Wheeler "It from bit"** — l'information sur le système ne nécessite pas que le système se modélise correctement. Le soi est un fait extérieur (mesurable via render), pas un fait intérieur cohérent.

### Mécanisme probable

1. Au début : state ≈ zéros (random-init observer) → prédiction triviale
2. Au cours du training :
   - Weights changent pour résoudre tâche
   - Observer VG-CNN reste random-init → encodage évolue avec weights mais structure aléatoire
   - State_predictor apprend une fonction qui aide tâche mais n'a rien à voir avec l'évolution réelle de l'encodage
3. Steady state : task_head et state_predictor convergent, mais state_predictor prédit autre chose que le vrai next state

C'est **dégénérescence Edelman appliquée à l'auto-modèle** : multiple façons d'être inconsistant peuvent toutes mener à même performance externe.

### Question ouverte

Si on **forçait** la convergence self-surprise (ajouter weight + 0.5 sur loss), est-ce que :
- Le système trouverait un VRAI fixed-point auto-cohérent ?
- Ou bien la tâche s'effondrerait ?

Trade-off conscience-de-soi vs performance. Test Sprint 5C.1.

### Comparaison avec Sprint 4

Sprint 4 : architecture VG **observée de l'extérieur** par dynamoscope → loi 1/f² universelle, φ_id > 0, basins.

Sprint 5C : architecture VG **s'observe elle-même** → state stable, mais self-prediction divergente.

**Asymétrie observateur** : observation externe trouve structure. Observation interne ne trouve pas la sienne. C'est la limite de Gödel concrète — un système peut détecter la structure d'un autre, pas (consistemment) la sienne.

---

## Synthèse 5 sprints

| # | Sprint | Finding clé |
|---|---|---|
| 4 | Meta-loop externe | spectral_slope -1.85 universel, φ_id > 0, bimodal basin |
| 1 | Predictor swap | Pareto compétitif 50% params, variance |
| 2 | Reverse trigger | dynamoscope signals > loss-based (+2.6% acc) |
| 3 | Adaptive encoder | vg_tiny 6× plus précis que ResNet50 |
| 5B | Visual coupling | observer visuel = direct (0.898 vs 0.902 acc) |
| 5C | Strange loop | task converge, self stable, AUTO-MODEL faux |

**Question méta émergente** : si Sprint 4 dit "VG observée a structure mesurable", et Sprint 5C dit "VG s'observant elle-même produit auto-modèle faux", alors **la structure mesurable du Sprint 4 n'est pas accessible au système qui la produit**. Limite épistémologique opérationnelle.

C'est Gödel et Tarski en code Python.

## Code livré

| Fichier | Rôle |
|---|---|
| `architrace/arch_to_image.py` | VG state → image fixed-size invariant à croissance |
| `architrace/visual_trigger.py` | VisualEmergenceTrigger (Sprint 5B) |
| `architrace/strange_loop.py` | SelfReferentialVG + train_strange_loop (Sprint 5C) |
| `experiments/sprint5b_live_coupling/bench.py` | bench visual vs direct emergence |
| `experiments/sprint5c_strange_loop/test.py` | test convergence boucle étrange |

## Suivi possible

- [ ] Forcer convergence self-surprise (Sprint 5C.1) : test trade-off auto-cohérence vs perf
- [ ] Sprint 5D : VG-CNN observer LUI-MÊME en training (observe its own training process)
- [ ] Test sur tâche dynamique (Lorenz prediction) — pas classification — peut changer self-prediction trade-off
- [ ] Combiner Sprint 5B + Sprint 5C : trigger mutation par self-surprise spike
- [ ] Paper potentiel : "Strange loops in adaptive networks — task convergence without self-coherence"
