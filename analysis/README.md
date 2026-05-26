# Architrace — analysis directory

Documentation interprétative et visualisations pour les 27 sprints d'expériences.

---

## Index des fichiers

### Documents

| Fichier | Public visé | Lire en premier ? |
|---|---|---|
| **[VULGARISATION.md](VULGARISATION.md)** | Non-experts, lecteurs généraux | OUI si tu découvres le projet |
| **[GLOSSARY.md](GLOSSARY.md)** | Référence — tous termes, modèles, variables, notations | Consultation à la demande |
| **[SYNTHESIS.md](SYNTHESIS.md)** | Experts ML/AI, peer review | OUI si tu veux comprendre les dynamiques en profondeur |
| **[diagrams/DIAGRAMS.md](diagrams/DIAGRAMS.md)** | Schémas architecture (Mermaid) | Complément visuel à SYNTHESIS |

### Figures (`figures/`)

| Fichier | Contenu |
|---|---|
| `fig1_llm_scaling.png` | Scaling MSE Δpp vs log(params), crossover extrapolé ~17B |
| `fig2_infonce_u_shape.png` | Sweep α InfoNCE, sweet spot α=0.1 |
| `fig3_substrate_matrix.png` | Heatmap verdict par substrat × tâche |
| `fig4_warmup_vanishing.png` | Smooth-prior bonus 410M vs 1.4B (disparition à scale) |
| `fig5_world_task_destruction.png` | Sprint 20G : aux loss DÉGRADE smooth regression |
| `fig6_sprint_timeline.png` | Timeline des 27 sprints avec verdicts |
| `fig7_mse_vs_infonce.png` | Comparaison MSE two-halves vs InfoNCE per-token |
| `fig8_exaptation_speedup.png` | Speedup init surp par substrat (log-scale) |

Régénérer toutes : `.venv/bin/python figures/generate_figures.py`

---

## Parcours recommandés

### Parcours 1 — Découverte (lecteur curieux non-technique)
1. [VULGARISATION.md](VULGARISATION.md) (10 min)
2. `figures/fig6_sprint_timeline.png` (visualisation rapide)
3. `figures/fig3_substrate_matrix.png` (grille verdict)

### Parcours 2 — Audit critique (peer reviewer)
1. [SYNTHESIS.md](SYNTHESIS.md) sections 1-5 (verdict structuré)
2. `figures/fig1_llm_scaling.png` + `fig5_world_task_destruction.png` (key data)
3. [GLOSSARY.md](GLOSSARY.md) §4 (notations mathématiques)
4. [SYNTHESIS.md](SYNTHESIS.md) sections 6-12 (interprétation théorique)

### Parcours 3 — Réplication / extension (ML practitioner)
1. [GLOSSARY.md](GLOSSARY.md) §1-3 (modèles, variables, hyperparamètres)
2. [diagrams/DIAGRAMS.md](diagrams/DIAGRAMS.md) D1-D3 (architecture)
3. [SYNTHESIS.md](SYNTHESIS.md) §13 (recommendations réplication)
4. Code source dans `../experiments/sprint*/`

### Parcours 4 — Philosophie / épistémologie
1. [VULGARISATION.md](VULGARISATION.md) sections finales (implications philosophiques)
2. [SYNTHESIS.md](SYNTHESIS.md) §6, §10, §12 (modeling axis, théories, épistémologie)
3. `../paper/paper.pdf` (formulation académique)

---

## Key findings en une page

```
SOLIDES
  S03    : VGCNNEncoder tiny (~500 params) bat ResNet50 sur tâches structurelles (6×)
  S04    : 1/f² spectral universel sur croissance VG (SOC signature)
  S11.3  : Exaptation 103× vision MLP MNIST (vindiqué par S15C 74× drift-norm)
  S18C   : Cross-architecture vision works (ResNet/Mixer 8-12×) MAIS ViT inverse (0.7×)
  S19    : Pattern de scaling LLM monotone log-linéaire, demi-vie ~2.5× params
  S20C   : InfoNCE per-token α=0.1 = premier Δpp NÉGATIF en LLM (rescue partial)
  S20E   : Gaussian ≈ Lorenz → exaptation S11.x = smooth-prior generic, pas chaotic-specific
  S20DE  : Self-bootstrap émergent entre 410M et 1.4B (predictor pre-train devient redondant)

RÉFUTÉS / NUANCÉS
  S15B   : Drift ≈ 0, self-surprise = observer noise (partial artifact in earlier sprints)
  S20F   : Whole-model world warmup en LLM = catastrophic forgetting partial (+0.53 pp)
  S20G   : InfoNCE anti-aligned avec smooth regression task (+78% MSE)
  
VERDICT FINAL
  Recipe self-coherence n'est PAS universel.
  Triple alignement nécessaire : substrate × task × aux loss.
  Domaine de validité connu = vision classification + α petit + MSE/InfoNCE compatible.
```

---

## Pour citer

```bibtex
@misc{architrace2026,
  author       = {Zaccarie Aglas},
  title        = {Architrace: Self-Reference Operationalized in Adaptive Neural Architectures},
  year         = {2026},
  month        = may,
  note         = {Workshop draft, 27 sprints, paper.pdf in repo}
}
```

---

## Métadonnées techniques

- **Hardware** : Apple M-series Silicon, MPS device, 32 GB RAM
- **Software** : Python 3.13, PyTorch 2.x, transformers 5.9, peft 0.19, matplotlib 3.x
- **Repo** : https://github.com/zacccarie/architrace (private)
- **Build paper** : `cd paper && tectonic paper.tex`
- **Contact** : hello@zaccarie.fr
