# autoresearch PLAYBOOK — architrace

⚠️ **architrace n'est PAS un terrain optimal pour autoresearch.** Lis CONTEXT.md d'abord pour comprendre pourquoi.

## Cible courante (si on doit insister)

**Sprint 3.1 extended bench** — comparaison fair encoders pour reconstruction systèmes dynamiques.

Wrapper : `autoresearch/wrapper.py sprint3_1`
Métrique : `delta_corr_dim` (delta corr_dim vs ground truth, lower better)
Direction : **lower_better**

## Already established (DO NOT re-prove)

11 findings ROBUSTES, ne PAS toucher :

1. **Slope spectrale 1/f²** (β ≈ -1.85 ± 0.13) universel inter-seed (S4)
2. **VGCNN tiny 498p 6× > ResNet50 23.5M** sur tâches structurelles (S3)
3. **Exaptation 103× vision MLP MNIST** (S11.3, S15C 74× persistant)
4. **Self-coherence computationnellement bon marché** (S7)
5. **Cross-arch vision** : ResNet/Mixer OK, ViT inverse 0.7× (S18C)
6. **Smooth-prior generic** (Gaussian ≈ Lorenz au sein variance, S20E)
7. **Self-bootstrap émerge à 1.4B+** (S20DE)
8. **RQA DET = LAM = 0.999** sur trajectoire architecture VG (S4)
9. **Lyapunov ≈ 0.04** sur évolution VG (S4)
10. **DMD rank 2-3** dans snapshot 58D (S4)
11. **Bimodal basin factor 6×** cluster separation (S4)

**Ne tente pas** de "battre" ces résultats. Ils sont vrais.

## Already refuted (DO NOT re-try)

- ❌ Exaptation universelle 100% — confirmé conditionnel substrate × task × aux loss
- ❌ MSE two-halves sur LLM — dégrade scale-dependent (S15D, S19)
- ❌ Whole-model warmup LLM (S20F)
- ❌ LLM-on-world-task + InfoNCE (S20G : Δmse +78%)
- ❌ Sprint 13C "circuit distribué" — artefact partial (S15B révision)
- ❌ InfoNCE α=0.1 LLM — within variance

## Questions ouvertes (cibles potentielles)

| # | Question | Cheap pour autoresearch ? |
|---|---|---|
| Q1 | Crossover Δpp=0 à ~17B params réel ? | ❌ cloud GPU |
| Q2 | InfoNCE × state_dim sweep | ✓ ~1h mais via M3, pas architrace |
| Q3 | Métrique a priori manifold compat | ✓ **M3 cible — pas architrace** |
| Q4 | Vision audit rétrospectif Gaussian control | ✓ heures |
| Q5 | LLM-on-world-task MSE aux loss | ✓ 1h via annexe M7 |
| Q6 | Self-bootstrap threshold 410M-1.4B | ❌ cloud GPU |

## Pourquoi autoresearch limité ici

1. **Sprint 3.1 run ≥ 30 min** (5 encoders × 3 systèmes × seeds). Budget 5min fixe inadapté.
2. **Question = "inductive bias real?"** qualitative, pas "optimize X" numérique.
3. **Findings robustes déjà nombreux** → marge upside faible.
4. **Vraie suite = metamorph** (M3 attaque Q3, M7 annexe attaque Q5).

## Si tu insistes sur architrace

Best edit space : `architrace/triggers.py` (Trigger classes, plateau detection) ou `architrace/predictors.py` (predictor heads). Petits changements possibles. Mais ROI faible.

## Recommandation forte

**Ne lance pas loop architrace.** Bascule sur metamorph M3 (voir `metamorph/autoresearch/`). Plus tractable, métrique propre, gradient clair.

architrace = **read for context, don't iterate**. Le paper draft est la sortie naturelle, pas plus d'expé automatisées.
