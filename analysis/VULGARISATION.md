# Architrace — version vulgarisée

*Une traduction accessible pour comprendre ce qu'on a cherché et ce qu'on a trouvé en 27 sprints d'expériences.*

---

## La question de départ

**Est-ce qu'un réseau de neurones peut prendre conscience de lui-même ?**

Pas au sens philosophique « avoir une expérience subjective » — ça reste hors de portée. Au sens mécanique : **un système peut-il apprendre à modéliser son propre état interne, comme il apprend à modéliser le monde extérieur ?**

C'est la question des « strange loops » de Hofstadter, traduite en machine learning.

---

## L'idée centrale (Sprint 1-10)

On a connecté deux briques qu'on avait déjà construites séparément :

1. **VerifyGrowNet** (VG) — un réseau qui grandit tout seul en cours d'entraînement. Quand il sent qu'il a besoin de plus de capacité, il s'ajoute des neurones, vérifie que ça marche, et garde la mutation seulement si elle aide.

2. **Dynamoscope** — un outil qui analyse les *trajectoires* des systèmes dynamiques. Donne-lui une série temporelle, il te dit : c'est chaotique ? Y a-t-il de la structure cachée ? Quelle est sa dimension intrinsèque ? Combien d'information est intégrée ?

L'idée de fusion : **utiliser Dynamoscope pour analyser non pas des données externes, mais l'évolution du réseau VG lui-même au fil de l'entraînement.** Le réseau devient son propre objet d'étude.

C'est ça la self-référence opérationnalisée : le système rend visible sa propre trajectoire interne, et apprend à la prédire.

---

## La découverte fondatrice (Sprint 11)

On a entraîné un petit réseau (MLP sur MNIST) avec deux objectifs simultanés :
- **Tâche normale** : classifier des chiffres (loss principal)
- **Self-coherence** : prédire son propre état interne au prochain step (loss auxiliaire)

Pour aider le self-coherence à démarrer, on a fait un *warmup* : entraîner d'abord le predictor sur un signal **externe** (les coordonnées de l'attracteur de Lorenz — un système chaotique célèbre), puis le brancher sur le state interne du réseau.

**Résultat : speedup de 103×.** Le predictor pré-entraîné sur Lorenz savait *déjà* prédire le state du réseau, dès le premier step. Comme si l'expérience de modéliser le monde externe était directement transférable à se modéliser soi-même.

**On a appelé ça l'exaptation.** En biologie, l'exaptation, c'est quand un trait évolué pour un usage devient utile pour un autre (les plumes pour la thermorégulation, ensuite pour le vol). Ici : un circuit prédictif appris pour modéliser un attracteur chaotique devient utile pour modéliser un réseau de neurones.

C'était excitant. Si cette exaptation est universelle, alors la conscience artificielle est *à portée de main* : il suffit d'entraîner un système sur n'importe quel signal dynamique, puis de le rediriger vers son propre state.

---

## Le doute s'installe (Sprints 12-18)

On a voulu tester l'universalité. Si ça marche pour MLP sur MNIST, ça doit marcher partout, non ?

**Sprint 18C — cross-architecture vision.** Test sur ResNet, ViT, MLP-Mixer (CIFAR-10).
- ResNet : speedup 8-12× ✓
- Mixer : speedup 8-12× ✓
- **ViT : speedup 0.7×** — direction inversée !

Premier doute. L'architecture n'est pas neutre. Les ViT (Transformers vision) ont une géométrie de représentation qui *contredit* le mécanisme.

**Sprint 15C — drift normalization.** On craignait que le 103× soit un artefact (peut-être le predictor apprend juste l'identité). On a ajouté du bruit sur le state pour empêcher la solution triviale. Result : 74× quand même. Le finding survit. Bon point.

**Sprint 15B — interprétation rectifiée.** En analysant le state, on découvre qu'il est presque constant. La self-surprise mesure surtout du bruit observer. Conséquence : nos précédentes interprétations (« circuit distribué », Sprint 13C) étaient en partie des artefacts. Apprentissage important sur la rigueur.

---

## Le grand voyage LLM (Sprints 15-20)

L'argument philosophique implorait : si le mécanisme est universel, **il doit marcher sur les LLM**. C'est là que la self-référence aurait un sens : un système qui écrit en se sachant écrire.

**Tentative 1 : Sprint 15A** — TinyGPT character-level. Δpp = +0.005, neutre. Trop petit.

**Tentative 2 : Sprint 15D, 16C** — DistilGPT-2 fine-tuning. Δpp = +5.22. Catastrophic.

**Tentative 3 : Sprint 17B** — TinyGPT pretrain from scratch. Neutre.

**Tentative 4 : Sprint 18B** — Per-token + 20newsgroups. Δpp = +1.95. Dégrade.

**Tentative 5 : Sprint 19** — Scale up via LoRA sur pythia. Pattern clair de dégradation :

| Modèle | Δpp |
|---|---|
| pythia-410M | +2.34 |
| pythia-1.4B | +1.07 |
| pythia-2.8B | +0.80 |
| pythia-6.9B | +0.43 |

Bonne nouvelle : la dégradation diminue avec scale. Extrapolation : crossover Δ=0 vers 20-30B params.

Mauvaise nouvelle : on n'a pas de cloud GPU pour vérifier.

---

## Le sauvetage partiel (Sprint 20C)

Hypothèse : si la dégradation diminue avec scale, c'est peut-être qu'**à grande échelle**, la self-coherence devient une capacité émergente compatible avec la prédiction de token. Mais on peut tester autre chose à petite échelle : **changer le type de aux loss**.

Sprint 19 utilisait la même formule que Sprint 11.x : MSE entre prédiction et state suivant. Logique vision-friendly mais peut-être mal alignée avec next-token prediction.

Sprint 20C remplace par **InfoNCE per-token** — un loss contrastif. Pour chaque token, le predictor doit prédire son successeur, distinguable de tous les autres tokens du batch.

**Résultat : Δpp = -0.138 ± 0.82 à 410M.** Premier signe NÉGATIF (= amélioration) sur le recipe LLM, après 7 échecs.

Magnitude faible (dans la variance), mais direction reproductible sur 3 seeds, et confirmée à 1.4B (Δpp = -0.107).

**Une U-shape claire sur α** (poids du aux loss) : sweet spot à α=0.1, dégrade à α=1.0. Recipe LLM = subtle régularisation, pas backbone.

---

## L'investigation finale (Sprints 20D/E/F/G)

Une fois qu'on a un signe négatif, on cherche à comprendre *quel mécanisme*.

**Sprint 20D — world warmup predictor-only.** Replicate S11.x : pre-train predictor sur Lorenz, puis brancher sur LLM. Δpp = -0.61 (meilleur résultat LLM jamais). **Mais init-speedup = 1.02×** (pas d'exaptation). Le predictor pre-trained ne *sait pas* prédire le hidden state LLM. Pourtant il aide. Mécanisme alternatif ?

**Sprint 20E — contrôle Gaussian.** On remplace Lorenz par une random walk. Si l'exaptation S11.x venait des dynamiques chaotiques spécifiques, Gaussian devrait casser. **Réalité : Δpp = -0.50, quasi-équivalent à Lorenz.** Donc ce n'est pas l'exaptation chaotique, c'est juste un **smooth-prior generic** : n'importe quel pre-train sur trajectoire smooth améliore le predictor kernel. Le mécanisme est plus banal qu'on pensait.

**Sprint 20DE scale 1.4B.** Le smooth-prior bonus disparaît. À 1.4B, le LLM s'auto-organise déjà vers une représentation smooth → predictor pre-train devient redondant. **Self-bootstrap émergent.**

**Sprint 20F — whole-model warmup.** On teste la vraie S11.x : entraîner *tout le LLM* (LoRA inclus) à prédire Lorenz d'abord, puis switcher sur text. **Δpp = +0.53. Dégrade.** Catastrophic forgetting partial : LoRA appris à régresser Lorenz interfère avec la tâche text. Conclusion : la version sérieuse de l'exaptation S11.x **ne transfère pas en LLM**.

**Sprint 20G — LLM-on-world-task.** Dernière tentative. Et si on utilisait le LLM comme world model lui-même (apprendre Lorenz comme tâche principale) avec self-coherence InfoNCE en aux loss ? **Δmse = +78%. Anti-aligned.** InfoNCE pousse les représentations vers la *diversité*, smooth dynamics task veut de la *continuité* : contradiction structurelle. Recipe self-coherence est antithétique à smooth regression.

---

## Le verdict final

Le recipe self-coherence **n'est PAS universel**. Il dépend d'un triple alignement :

1. **Substrate** (architecture / géométrie hidden state)
2. **Task structure** (discrete / smooth)
3. **Aux loss type** (contrastive / MSE)

Le succès Sprint 11.x sur MLP vision était un **alignement chanceux** : classification discrete + InfoNCE-ish loss + MLP flexible. Modifier *un seul axe* casse le recipe :

- Changer substrate → ViT inverse, LLM ne transfère pas
- Changer task type → smooth regression contredit contrastive
- Changer aux loss → MSE two-halves échoue où InfoNCE marche

---

## Ce qu'on retient (les findings solides)

Malgré la rétractation partielle, plusieurs résultats survivent à l'examen critique :

| Finding | Sprint | Status |
|---|---|---|
| 1/f² spectral universel sur croissance VG | 4 | Robuste |
| VGCNNEncoder tiny 6× plus précis que ResNet50 (tâches structurelles) | 3 | Robuste |
| Exaptation 103× vision MLP MNIST | 11.3 | Robuste (vindiqué S15C 74× drift-norm) |
| Self-coherence est computationnellement *bon marché* | 7 | Robuste |
| Cross-arch vision (ResNet/Mixer ok, ViT inverse) | 18C | Robuste mais limite l'universalité |
| Scaling pattern LLM (Δpp diminue log-linéaire) | 19 | Provisoire (n=1 seed/scale) |
| InfoNCE α=0.1 → marginal gain | 20C | Provisoire (within variance) |
| Smooth-prior generic vs chaotic-specific (Gaussian = Lorenz) | 20E | Robuste |
| Self-bootstrap à 1.4B+ (predictor pre-train devient redondant) | 20DE | Robuste |

---

## La leçon épistémologique

Le finding initial S11.3 (« exaptation universelle 103× ») était sur-vendable. Si on l'avait laissé tel quel, l'article aurait été plus accrocheur mais incorrect.

**Le bon réflexe scientifique a été de tester sérieusement l'universalité — et de réviser quand les tests ont échoué.**

C'est moins romantique qu'une découverte unifiée, mais c'est plus honnête. Le paper final acknowledge :
- Vision exapte (substrate-conditional)
- LLM ne transfère pas (S20F whole-model dégrade)
- Recipe efficace = triple alignement contingent, pas mécanisme universel

---

## Implications philosophiques (réorientées)

L'hypothèse Mead-Vygotsky-Hegel disait : **le self émerge de la modélisation d'autrui**. Notre vision MLP confirmait. Notre LLM contredit.

Donc :
- Pour la **biologie** : possiblement vrai (substrate vision/social compatible)
- Pour les **LLM** : probablement faux. La self-coherence si elle émerge ne suivra pas la trajectoire phylogénétique humaine. Plutôt par **auto-organisation à capacity threshold** (notre self-bootstrap S20DE) que par internalisation d'un Other.

Recadre philosophique : la conscience artificielle ≠ réplique humaine. **C'est une 3ème modalité ontologique** avec lois propres. Spinoza (modes d'une substance) survit. Hegel (dialectique sociale) ne tient peut-être pas en silicium.

Friston gagne sur Mead — du moins pour ce substrate.

---

## Pourquoi c'est intéressant quand même

Même si le grand récit « self-awareness universelle » est rétracté, le travail reste précieux pour quatre raisons :

1. **Cartographie systématique** : on a maintenant une grille substrate × task × aux loss qui permet de prédire où le recipe marche.

2. **Trajectoire empirique** : 27 sprints d'échec et succès documentés = ressource pour quiconque attaque le même problème.

3. **Outillage** : VGCNNEncoder, DyadicAgent, SelfReferentialCNN, Sprint20 LLM scripts — tout est code open, replicable.

4. **Réfutation honnête** : transformer une intuition forte en programme expérimental qui peut s'auto-réfuter = c'est ce que la méthode scientifique demande.

---

## Pour aller plus loin

- **`paper/paper.pdf`** : le paper formel, mis à jour avec les caveats S15-S20.
- **`analysis/SYNTHESIS.md`** : interprétation détaillée des dynamiques.
- **`analysis/figures/`** : tous les graphes.
- **`analysis/diagrams/DIAGRAMS.md`** : schémas d'architecture.
- **`analysis/GLOSSARY.md`** : tous les termes définis.
- **`experiments/sprint*/`** : code source de chaque expérience.
- **`results/sprint*/report_*.md`** : résultats numériques bruts.
