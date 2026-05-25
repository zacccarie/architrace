# Sprint 4 — Architecture-as-Trajectory : VerifyGrowNet × Dynamoscope

**Date** : 2026-05-26
**Statut** : Pipeline end-to-end validé

## Problème

VerifyGrowNet (VG) modifie sa propre architecture en cours d'entraînement (widen layers, prune neurons, freeze).
Cette **séquence de mutations + drift des poids** constitue intrinsèquement un système dynamique.

Question :
> *La trajectoire d'architecture VG dans son espace de features (largeurs, rangs effectifs, normes de gradient, etc.) montre-t-elle une structure dynamique mesurable ?*

## Approche

1. **Extraction features** : chaque snapshot architecture → vecteur fixe 58D
   - Par couche (8 slots × 6 features) : width · mean|w| · spectral_radius · effective_rank · mean|grad| · mean|bias|
   - Globaux (10) : params · depth · val_loss · train_loss · step · mutations · accepted · rejected · frozen · distill_remaining

2. **Bridge dynamoscope** : nouveau endpoint `POST /api/inject_latents` accepte trace N×D, projette UMAP 3D, branche pipeline analyses existante.

3. **Analyses lancées sur cache injecté** :
   - `dynamics_native` (Lyapunov, RQA, corr_dim sur 58D natif)
   - `topology_native` (persistent homology H0/H1)
   - `sindy` (équations différentielles sparses)
   - `emergence` (effective information ladder, φ_id)
   - `spectral` (DMD/Koopman, slope, multiscale entropy)
   - `causal` (Granger/TE/CCM)
   - `/api/compare` (UMAP partagé inter-runs)

4. **Run** : 5 seeds × MNIST × 4 epochs × VG `[20,20] → max [128,128]`. 87-88 snapshots/run.

## Résultats clés (5 seeds, MNIST)

### Universalité spectrale
| metric | mean ± std |
|---|---|
| spectral_slope | **-1.85 ± 0.13** |
| DMD rank | 2.4 ± 0.5 |

→ Loi 1/f² **quasi-invariante inter-seed**. Architecture VG = système low-rank (2-3 modes propres). Mémoire long-terme caractéristique.

### Déterminisme + stabilité contractive
| metric | mean ± std |
|---|---|
| Lyapunov 58D | 0.04 ± 0.06 |
| RQA DET | 0.999 ± 0.002 |
| RQA LAM | 0.999 ± 0.001 |
| corr_dim 58D | 0.56 ± 0.33 |

→ Trajectoire **purement déterministe + laminaire** (DET ≈ LAM ≈ 1.0). Lyapunov ≈ 0 → faiblement expansive, pas de chaos. Dimension de corrélation < 1 → trajectoire quasi-1D dans son espace 58D.

### Émergence détectable
| metric | mean ± std |
|---|---|
| EI micro | 2.42 ± 0.29 |
| φ_id (Hoel) | **0.19 ± 0.29** |
| optimal_scale | 2.6 ± 2.4 |

→ `interpretation: macro states are more predictable than micro` sur tous les seeds.
**Coarse-graining gagne** = émergence causale présente. Variance φ_id large = signal sensible aux conditions init.

### Bassins d'attraction multiples
Distances pairwise inter-seed (UMAP partagé) :

| pair | mean_pair_dist | centroid_dist |
|---|---|---|
| s0-s3 | 0.018 | 0.002 |
| s0-s4 | 0.056 | 0.025 |
| s3-s4 | 0.057 | 0.025 |
| s1-s2 | 0.056 | 0.030 |
| s0-s1 | 0.373 | 0.346 |
| s1-s3 | 0.350 | 0.281 |
| s2-s3 | 0.265 | 0.243 |

→ **Distance intra-cluster ~0.05, inter-cluster ~0.30** → **factor 6× separation**.
Clusters :
- Cluster A = {s0, s3, s4} — mutations acceptées
- Cluster B = {s1, s2} — pending (verify n'a pas conclu en temps)

L'**outcome de la mutation determine le bassin**. Même accuracy finale (0.857 ± 0.008) mais topologies différentes ([36,20] vs [24,24]).

### SINDy
| metric | mean ± std |
|---|---|
| R² | 0.76 ± 0.15 |
| n_active terms | 24 / 30 lib |

→ Dynamics modélisables par équations sparses ordre 2 dans projection 3D, R² > 0.7 sur tous seeds.

## Interprétation

1. **L'architecture VG est un système dynamique de basse dimension intrinsèque** (rank DMD 2-3, corr_dim < 1) embarqué dans un espace de features 58D.

2. **Loi 1/f² universelle** : `spectral_slope = -1.85` quasi-invariant inter-seed. Pattern consistent avec systèmes auto-organisés (criticalité auto-organisée, drift gradient, mémoire long terme).

3. **Émergence causale mesurable** : φ_id > 0 partout, optimal_scale > 1 → coarse-graining augmente la prédictibilité. Architecture macroscopique (topologie) plus informative que micro (poids individuels).

4. **Bassins multiples séparés par mutations** : la décision verify-then-keep d'une mutation = bifurcation. Système possède plusieurs attracteurs équivalents en performance.

5. **Stabilité contractive** : pas de chaos, Lyapunov ≈ 0. Pousse de réseau = système dissipatif convergent.

## Lien obsessions Zac

- **Émergence** : φ_id > 0 + "macro plus prédictible que micro" = définition opérationnelle de l'émergence à la Hoel. Architecture VG = laboratoire d'émergence in-silico.
- **Loi 1/f²** : signature systèmes auto-organisés (DDLab Wuensche, Wolfram, criticalité).
- **Bassins d'attraction** : DDLab garden-of-Eden states + bassins atteignables. Mutations VG = chemins dans paysage d'architectures, attracteurs visibles.
- **Information substrat** : Wheeler "It from bit" — architecture = forme de l'information, mesurable comme dynamique.

## Code livré

| Fichier | Rôle |
|---|---|
| `adaptive-nn/arch_snapshot.py` | Extracteur features 58D depuis VG state |
| `adaptive-nn/verifygrow_nn.py` | Hooks snapshot dans training loop (pre/post mutation + ticks) |
| `adaptive-nn/vg_run_to_trace.py` | CLI runner multi-seed → trace JSON |
| `adaptive-nn/vg_dynamoscope_pipeline.py` | Pipeline inject + analyses cascade |
| `adaptive-nn/vg_compare_seeds.py` | Aggregate metrics inter-seed → markdown table |
| `dynamoscope/backend/main.py` | Endpoint `POST /api/inject_latents` + `GET /api/inject_meta/{cache_key}` |

## Suivi — Sprint 4.1 (follow-ups)

- [ ] Frontend dynamoscope : bouton "Load injected trace" + dropdown cache_keys
- [ ] Coloration trajectoire par tag (tick / pre_widen / accept / reject / final)
- [ ] Vue mode "architecture phase space" avec axes interprétables (depth × params × val_loss)
- [ ] Scaling : run 50+ seeds pour statistiques robustes sur φ_id et bassins
- [ ] Varier dataset (CIFAR-100, synth) pour tester universalité spectral_slope
- [ ] Sprint 1 (P3 prune cycle) + Sprint 4 ensemble : trajectoire avec mutations widen + prune

## Options restantes Sprint

- ✅ Option 4 (architecture-as-trajectory) — **validé pipeline complet**
- ⬜ Option 1 (VG remplace prediction.py)
- ⬜ Option 2 (surprise dynamoscope → trigger VG)
- ⬜ Option 3 (VG comme encoder adaptatif)
