# autoresearch CONTEXT — architrace

⚠️ **architrace n'est pas une cible autoresearch optimale.** Lis ceci avant de tenter.

## What architrace est

Repo fusion `adaptive-nn` × `dynamoscope`. 27 sprints (Sprint 1 → Sprint 20+). Paper draft en cours (`paper/`).

**Verdict final** : self-modeling **possible** (modeling axis universel) mais exaptation efficiency **conditionnelle** au triple alignement substrate × task × aux loss.

**Trajectoire honnête** : le repo a démarré sur affirmation forte Sprint 11.3 (103× speedup universel) puis **s'est auto-réfuté partiellement** au fil des sprints. C'est sa rigueur scientifique principale.

## Pourquoi ce n'est PAS autoresearch-friendly

### 1. Question qualitative
architrace cherche à prouver/réfuter "**inductive bias > scale**". Réponse = "conditionnel". Pas un nombre à maximiser.

### 2. Métrique multi-axe
Chaque sprint a sa métrique : Δpp (delta percent points), corr_dim error, Lyapunov, φ_id. Pas de métrique unique transversale.

### 3. Run lourd
Sprint 3.1 bench = 5 encoders × 3 systèmes × N seeds. ~30 min minimum. Budget autoresearch 5 min = mismatch.

### 4. Findings déjà nombreux
11 robustes + 6 réfutés + 6 ouvertes. La récolte facile est faite. Marge upside = faible.

### 5. Suite naturelle = metamorph
Les 6 questions ouvertes Q1-Q6 sont reprises dans metamorph (M3 attaque Q3, M7 annexe attaque Q5). C'est metamorph qui itère, pas architrace.

## Ce que architrace est pour autoresearch

- **Read for context** : findings memory `architrace_findings_solid_vs_open` = base de référence
- **Source de data** pour M3 metamorph (les Δpp viennent de sprints architrace)
- **Paper draft** = sortie naturelle (ai-scientist peut polish, pas autoresearch)

## Ce que architrace n'est PAS

- Pas un terrain à boucler
- Pas une métrique simple à optimiser
- Pas un repo à modifier automatiquement (paper en finalisation = code stable préférable)

## Si autoresearch insiste

Best minimal target : tweak `architrace/triggers.py` (plateau detection thresholds, Lyapunov triggers). Petits changements, faible portée.

**Ne touche pas** :
- `paper/` (en finalisation)
- `experiments/sprint*/` historiques (résultats publiés/à publier)
- `architrace/triggers.py:Trigger` interface (utilisé partout)
- Calculs métriques (`bottleneck_layer`, `eval_val`)

## Recommandation

**Skip architrace autoresearch loop.** Mets ton compute sur :
1. **metamorph M3** (Q3 directement)
2. **dynamoscope paper** (ai-scientist write)
3. **nanoGPT autoresearch** (Karpathy strict, infra test)

architrace = lis, cite, base le travail sur. Itère sur `metamorph` à la place.
