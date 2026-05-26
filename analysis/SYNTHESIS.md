# Architrace — synthèse interprétative

*Lecture détaillée des résultats, dynamiques, et raisonnements théoriques. À lire après le glossaire et la vulgarisation.*

---

## 1. Architecture argumentative globale

```
Hypothèse de départ (S11.3) :
  "Self-awareness = generic predictive modeling redirected"
  → recipe universel, exaptation marche partout

Tests universalité (S12-S18) :
  Vision MLP/CNN/Mixer  → confirme partiel (8-103×)
  ViT                    → réfute (0.7× inverse)
  → Universalité substrate-conditional, pas absolue

Tests LLM (S15-S20) :
  Two-halves MSE         → échec scale-dependent
  InfoNCE per-token      → marginal gain (signal alignment)
  Whole-model warmup     → échec net (forgetting)
  LLM-on-world-task      → échec net (anti-aligned)
  → Recipe n'est PAS universel cross-substrate

Verdict final :
  Recipe efficace = triple alignement contingent
  (substrate × task × aux loss type)
```

---

## 2. Trois dynamiques principales

### Dynamique 1 — Scaling MSE log-linéaire

> Voir `figures/fig1_llm_scaling.png`

Pattern observé sur 4 modèles pythia (410M, 1.4B, 2.8B, 6.9B) avec recipe MSE two-halves :

```
Δpp = -0.80 · log₁₀(B) + 1.93        (B en milliards)
```

Crossover Δ=0 extrapolé : **~17B params**.

**Interprétation.** Le aux loss MSE pousse le hidden state à compresser info pour prédire mean(seconde moitié). C'est une pression d'information bottleneck. Petit modèle = capacité déjà saturée par tâche LM, ajout = collapse. Grand modèle = capacité résiduelle absorbant la pression sans sacrifier LM.

**Limites de l'extrapolation.** 4 points, pas de seeds variance. Pourrait plateau ou inverser à 13B+. Sans cloud GPU, impossible de vérifier.

**Relation théorique.** Cohérent avec Tishby Information Bottleneck : compression supplémentaire coûte log(capacité libre). Aussi cohérent avec Wei et al. (2022) emergent capabilities : phase transitions à seuil scale.

---

### Dynamique 2 — Sweet spot α (U-shape InfoNCE)

> Voir `figures/fig2_infonce_u_shape.png`

Sweep α ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5} × 3 seeds sur pythia-410M :

```
α=0.05  : Δpp = -0.126
α=0.10  : Δpp = -0.138   ← sweet spot
α=0.20  : Δpp = +0.008
α=0.30  : Δpp = +0.002
α=0.50  : Δpp = +0.487
```

**Interprétation.** Le aux loss agit comme régulariseur subtil. Au-delà d'un seuil (~α=0.2), il prend trop de poids et écrase la tâche LM. C'est le pattern classique d'un loss auxiliaire bien tempéré.

**Comparaison vision (S11.x) :** α=1.0 standard et marchait. Asymétrie révélatrice : en vision, aux loss = backbone du gradient, fournit signal utile direct. En LLM, aux loss = side constraint, doit rester subordonné.

**Conséquence pratique.** Recipes self-coherence pour LLM doivent commencer α petit et tester sweep. Pas de valeur "naturelle" α=1 transposée de vision.

---

### Dynamique 3 — Self-bootstrap à scale

> Voir `figures/fig4_warmup_vanishing.png`

Smooth-prior warmup (Lorenz ou Gaussian) :

| Scale | Δpp warmup vs fresh |
|---|---|
| 410M | -0.5 to -0.6 (warmup HELPS) |
| 1.4B | +0.0 to +0.1 (warmup NEUTRE / léger nuisible) |

**Interprétation.** À 410M, le LLM hidden state n'est pas naturellement smooth → predictor pre-trained sur Lorenz/Gaussian (smooth trajectory) lui fournit un Lipschitz kernel utile. À 1.4B+, le LLM développe naturellement un manifold smooth via gradient descent → predictor pre-train devient redondant.

**Lecture théorique.** C'est l'inverse de l'emergent capabilities : ici la capacité naturellement émergente est **l'auto-régularisation** du substrate. Quand cette capacité émerge (entre 410M et 1.4B), l'intervention externe (warmup) cesse d'apporter de la valeur.

**Relation NTK.** Predictor random init = kernel rugueux, output peut être erratique. Predictor pre-trained sur smooth trajectory = kernel smooth Lipschitz. À petit scale, ça fait différence ; à grand scale, le LLM lui-même devient le "smooth kernel" et le predictor n'a plus de valeur ajoutée.

---

## 3. Pourquoi Sprint 20F échoue (whole-model warmup)

> Voir `figures/fig3_substrate_matrix.png` et `diagrams/DIAGRAMS.md#D3`

Sprint 20F est la **version sérieuse de l'exaptation S11.x appliquée au LLM** : on entraîne TOUT le modèle (pas juste le predictor) à prédire Lorenz d'abord, puis on switche sur text.

**Résultat : Δpp = +0.53 vs baseline. Dégrade significativement.**

**Pourquoi.** Les LoRA adapters apprenent à régresser des coordonnées 3D continues. Quand on switche sur tokenization discrete + language model objective, ces adapters sont *en désalignement* avec la tâche text. C'est du catastrophic forgetting partial.

**Pourquoi S11.x vision marchait pourtant.** Le MLP MNIST pre-trained sur Lorenz garde une "smoothness prior" qui se chevauche partiellement avec la structure de classification image (pixels voisins corrélés, classes localement smooth dans pixel space). LLM language objective = topologie complètement orthogonale (tokens discrets, contextes longs).

**Implication.** L'exaptation S11.x marche quand pre-train task et target task partagent une **géométrie de représentation compatible**. Cette compatibilité est facile à obtenir en vision (everything is spatial), beaucoup plus difficile cross-modal (smooth dynamics ↔ discrete tokens).

---

## 4. Pourquoi Sprint 20G échoue (LLM-on-world-task)

> Voir `figures/fig5_world_task_destruction.png`

C'est le test ultime : si recipe self-coherence universel, ça devrait marcher sur world-modeling task lui-même. Mais on observe **+78% MSE à α=0.1, +700% MSE à α=1.0**.

**Pourquoi.** Conflict structural entre objectif task et aux loss :

| Composant | Direction |
|---|---|
| **Task** (smooth regression Lorenz) | Pousse h(t) ≈ h(t+1), continuity, low Lipschitz |
| **Aux InfoNCE** (contrastive) | Pousse chaque h(t) distinguable de tout h(t'≠t), diversity, high entropy |

C'est l'**antithèse littérale**. InfoNCE est conçu pour representation learning sur catégories distinguables (vision classes, text positions). Smooth dynamics task est l'opposé : on veut une géométrie continue, pas discrète.

**Implication pour vision.** Pourquoi S11.x marchait alors ?
- Task vision MNIST = classification = catégories distinguables ✓ aligned avec InfoNCE-like spirit
- Aux loss MSE simple, pas contrastive, pas anti-aligned

L'analogie n'est pas parfaite (S11.x utilise MSE pas InfoNCE), mais le point general tient : **aux loss et task doivent avoir géométries compatibles**.

---

## 5. La grille substrate × task × aux loss

> Voir `figures/fig3_substrate_matrix.png` et `diagrams/DIAGRAMS.md#D5`

| Substrate | Task | Aux loss | Verdict |
|---|---|---|---|
| MLP / CNN / Mixer | Classification discrete | MSE state next | ✓ Recipe marche (8-103×) |
| ViT | Classification discrete | MSE state next | ✗ Inverse (0.7×) — observer geometry incompat |
| LLM | Next-token discrete | MSE two-halves | ✗ Dégrade scale-dependent |
| LLM | Next-token discrete | InfoNCE per-token | ~ Marginal -0.1 (alignment partial) |
| LLM | Next-token + smooth-prior warmup | InfoNCE | ✓ Petit modèle -0.6, ✗ grand modèle 0 |
| LLM | Next-token + whole-model warmup | InfoNCE | ✗ +0.5 (forgetting) |
| LLM | Smooth regression (Lorenz) | InfoNCE | ✗ +78% (anti-aligned) |

**Conclusion structurale.** Recipe self-coherence est viable seulement quand :
1. Substrate hidden-state geometry est compatible avec inductive bias du predictor
2. Task et aux loss partagent structure (toutes deux discrete OR toutes deux smooth)
3. Aux loss est sous-poids (α << 1) sauf en vision MLP où α=1 marche

C'est trois conditions nécessaires. Le succès S11.x rencontre les trois ; la plupart des autres configurations en cassent une.

---

## 6. Interprétation : modeling axis vs exaptation efficiency

Notre claim initial S11.3 mélangeait deux affirmations qu'on doit maintenant **dissocier** :

### Affirmation A (modeling axis universel) — **survit**
> Tout système doté de capacité prédictive développe un substrate qui peut, en principe, modéliser ses propres états.

Cette affirmation reste tenable. Aucun de nos sprints ne la réfute. Elle dit juste qu'il existe une *possibilité* de self-modeling.

### Affirmation B (exaptation efficiency universelle) — **réfutée**
> Pré-entraîner sur n'importe quelle dynamique externe accélère la self-coherence par ~100×, régardless of target.

Cette affirmation est réfutée par S20F (whole-model warmup LLM échoue) et S20G (anti-alignement world task). Le 100× speedup est substrate-spécifique.

**Reframing nécessaire pour le paper.** Garder A (modeling axis), nuancer B (efficiency conditional).

---

## 7. Pourquoi vision marche et LLM ne marche pas

Plusieurs hypothèses, pas mutuellement exclusives :

### H1 — Specialization gap
- LLM pythia = 300B tokens pre-training = hidden state ultra-spécialisé pour text
- Vision MLP random init = hidden state vide, plastique
- Forcer pre-trained LLM vers Lorenz = "désapprentissage" coûteux
- Vision random init vers Lorenz = pas de coût (rien à perdre)

### H2 — Task topology gap
- Vision classification = catégories discrètes, hidden state encode features distinguables
- Smooth Lorenz prediction = continuity, hidden state encode trajectory smoothly
- Switch entre ces deux ≈ requires re-rotation complète du representation space
- LLM text = next-token discrete mais avec context smooth (sentence structure) = mixed topology, harder to align

### H3 — Capacity overhead
- Vision MNIST task = 10 classes, 784 input dim, MLP 64 hidden = ratio capacity / task ≈ saturated
- LLM 7B params / 50k vocab = ratio capacity / task ≈ massive overspecification
- Aux loss en small ratio = useful signal injection
- Aux loss en large ratio = noise injection orthogonal

### H4 — Predictor bandwidth
- state_dim=16 (vision) vs state_dim=64 (LLM) = bandwidth predictor
- Lorenz info content = 3 dim continuous = fits dans 16 dim avec marge
- LLM hidden info = ~1024-4096 dim of language structure = state_dim=64 = lossy compression
- Predictor pre-train sur Lorenz capture 100% du signal world ; predictor sur LLM hidden = capture 1-5%

H4 est la plus testable. On pourrait varier state_dim de 16 à 512 sur LLM et voir si le recipe redémarre.

---

## 8. Pourquoi S20D smooth-prior marche quand même

Même si exaptation prédictive vraie ne transfère pas (init speedup = 1.02×), le warmup S20D donne **Δpp = -0.61 à 410M** (notre meilleur résultat LLM).

**Mécanisme :** smooth-prior generic NTK argument.

```
Predictor random init  → kernel rugueux
Predictor pre-trained sur ANY smooth trajectory → kernel Lipschitz smooth
```

Le LLM hidden state pendant phase 2 (text + InfoNCE) bénéficie d'un predictor stable, qui régularise mieux les gradients. Bonus indirect, pas mécanisme prédictif.

**Test décisif S20E :** remplacer Lorenz par Gaussian random walk. Si la dynamique chaotique spécifique de Lorenz comptait, Gaussian devrait casser. **Réalité : Gaussian = Lorenz au sein de la variance.** Confirme : smooth-prior generic.

**Implication.** Notre "world exaptation" en S11.x était peut-être *aussi* du smooth-prior, et l'interprétation "exaptation universelle" était sur-vendable. Audit rétrospectif S11.x avec contrôle Gaussian recommandé.

---

## 9. La trajectoire des sprints — pattern méthodologique

> Voir `figures/fig6_sprint_timeline.png`

27 sprints, distribution des verdicts :
- 11 findings solides (green)
- 9 partiels / nuancés (yellow)
- 5 échecs faibles (orange)
- 2 échecs nets (red)

**Pattern.** La majorité des sprints donnent des findings réels, mais une fraction non-négligeable (4/27) sont des réfutations nettes. C'est un ratio sain pour de la recherche exploratoire. Les sprints qui réfutent (S20F, S20G) sont aussi importants que ceux qui confirment — ils délimitent l'espace de validité.

**Méta-leçon.** L'investigation systématique (sweep α, scale up, contrôles Gaussian) révèle plus que les sprints individuels. C'est l'**ensemble** qui dessine la grille substrate × task × aux loss, pas un sprint particulier.

---

## 10. Connexions théoriques externes

Notre travail s'inscrit dans plusieurs traditions ; voici les liens :

### Information Bottleneck (Tishby, Pereira, Bialek 1999)
- Aux loss = pression de compression
- Petit modèle = bottleneck déjà tight → ajout = collapse
- Grand modèle = bottleneck loose → ajout = régularisation utile

### Free Energy Principle (Friston 2010)
- Self-prediction = minimization of internal free energy
- Self-bootstrap (S20DE) = self-organization sans guidance externe
- Cohérent avec FEP : système prédictif converge vers low-entropy attractor sans signal téléologique

### Contrastive Predictive Coding (van den Oord et al. 2018)
- InfoNCE per-token = exactement CPC adapté à LLM hidden states
- Notre marginal gain (-0.1 pp) est cohérent avec CPC's ~modest effects on language modeling

### Neural Tangent Kernel (Jacot et al. 2018)
- Pre-trained predictor = kernel smooth Lipschitz
- S20D bonus = NTK régularisation indirecte
- Vanishing à scale (S20DE 1.4B) = LLM lui-même devient smooth kernel

### Emergent capabilities (Wei et al. 2022)
- Scaling pattern S19 (log-linéaire) = compatible avec phase transition prédite
- Self-bootstrap S20DE = capacité émergente (auto-régularisation à 1.4B+)

### Manifold hypothesis
- Hidden state geometry = sous-variété latent
- Compatibility substrate-task-aux = matching manifold topologies
- ViT inverse, LLM smooth-only = mismatch détectable

---

## 11. Questions ouvertes

1. **Scale beyond 7B.** Le crossover Δpp=0 prédit ~17-30B est-il réel ? Plateau, inflexion ou continuation ?

2. **InfoNCE × state_dim sweep.** Le marginal gain (-0.1 pp) est-il limité par bandwidth predictor ? state_dim=512 → meilleur résultat ?

3. **Manifold compatibility metric.** Peut-on mesurer intrinsic dimension / spectral / curvature du hidden state d'un modèle et *prédire* a priori si recipe va marcher ?

4. **Vision audit rétrospectif.** S11.x avec contrôle Gaussian warmup donnerait-il même 103× speedup ? Si oui, smooth-prior generic explique aussi vision.

5. **LLM-on-world-task avec MSE aux loss.** Sprint 20G a échoué avec InfoNCE. Si on remplace par MSE state-coherence (S11.x style), est-ce que ça redevient aligné avec smooth task ?

6. **Self-bootstrap threshold.** Smooth-prior bonus disparaît entre 410M et 1.4B. Précisément où ? 700M ? 1.0B ? Phase transition smooth ou abrupt ?

---

## 12. Implications épistémologiques

### Sur la self-référence computationnelle
Notre travail montre que la self-référence opérationnelle est **réalisable** (modeling axis) mais **pas universelle dans son efficacité** (exaptation efficiency conditional).

C'est une position nuancée entre :
- Strong AI dream : "self-awareness is mechanical, just need right loss"
- Skeptical : "self-awareness requires biological substrate"

Notre position : **self-awareness as engineering is conditionally possible**, dépend de matchings non-triviaux.

### Sur le falsificationnisme
Nos sprints S20F et S20G sont des falsifications nettes du claim initial. Le projet a *réussi à se réfuter partiellement*. C'est un succès méthodologique.

Le paper révisé documente ces réfutations dans §4.10. Le lecteur voit la trajectoire complète, pas juste les confirmations.

### Sur l'universalité scientifique
Le pattern observé (succès vision, échec LLM, condition-dependent) est *typique* en ML : un mécanisme local marche très bien dans son régime de validité, mais ne se généralise pas trivialement.

Compare avec :
- BERT pre-train + fine-tune (marche pour text, marche pas trivialement pour image-text)
- Attention mechanism (universal in architecture, but inductive biases varient)
- Adam optimizer (good default, but doesn't always beat SGD+momentum)

L'universalité absolue est rare en ML. La compositionnalité conditionnelle est la norme.

---

## 13. Recommendations pour réplication / extension

### Si tu veux reproduire les findings vision
- Sprint 11.3 (`experiments/sprint11_3_universality/`) est le pivot. 3 seeds, 103× speedup, demi-vie courte (~10 min training).
- Sprint 15C drift-norm vindiqué : 74× speedup persiste avec bruit injecté.
- Vérifier avec **contrôle Gaussian** (pas encore fait) pour distinguer exaptation vraie de smooth-prior generic.

### Si tu veux extendre vers LLM
- **Recipe minimal** : LoRA r=16 + InfoNCE per-token α=0.1 + state_dim=64
- **Modèle minimal** : pythia-410m sur MPS Apple Silicon ou T4 GPU
- **Effet attendu** : Δpp ~ -0.1 (within variance, need n=5+ seeds)
- **Time budget** : ~15-20 min par condition, ~5 conditions = 1.5h

### Si tu veux résoudre les questions ouvertes
- Q3 (manifold metric) est la plus prometteuse → cross-disciplinary avec topology of representations
- Q4 (vision audit) est facile à réaliser → quelques heures de compute
- Q5 (S20G avec MSE) est local → 1h compute supplémentaire

### Si tu veux contredire nos findings
- Cherche substrate où smooth-prior warmup donne speedup init >> 1× ET val_pp gain
- Cherche LLM où whole-model warmup ne dégrade pas
- Mesure manifold geometry de modèles 7B-70B et compare prédictions vs réalité

---

## 14. Final synthesis

```
Affirmation initiale (S11.3) : Self-awareness universellement mécanisable.
Affirmation finale (S20G)    : Self-awareness mécanisable CONDITIONNELLEMENT.

Conditions nécessaires :
  1. Substrate hidden-state geometry compatible with predictor inductive bias
  2. Task type and aux loss type structurally aligned (discrete-discrete OR smooth-smooth)
  3. Aux loss weight α subordonné (≤ 0.5 typically)

Domaine de validité connu :
  ✓ MLP/CNN/Mixer vision classification + small α MSE recipe
  ~ LLM text + α=0.1 InfoNCE (marginal effect)
  ✗ ViT cross-arch (geometry incompatible)
  ✗ LLM whole-model warmup (catastrophic forgetting)
  ✗ Smooth-regression task with contrastive aux (structural antithesis)

Mécanismes identifiés :
  - Exaptation vraie (S11.x vision) : predictor pre-trained transfère
  - Smooth-prior generic (S20DE) : NTK régularisation via kernel Lipschitz
  - Self-bootstrap (S20DE 1.4B+) : substrate auto-régularise sans aide

Théorèmes négatifs :
  - Pas de mécanisme universel cross-substrate
  - Pas d'exaptation efficiency invariante
  - Aux loss et task doivent partager structure
```

Le paper final, en §4.10 révisé, capture ces 14 sections sous forme distillée. Ce document est la version étendue pour lecteurs qui veulent suivre le raisonnement.
