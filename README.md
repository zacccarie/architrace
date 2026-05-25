# architrace

Fusion VerifyGrowNet (adaptive-nn) × Dynamoscope. Architecture vivante × dynamique mesurable.

## Pitch

Réseaux qui font pousser leur propre architecture (VG) + outil d'analyse de trajectoires latentes (dynamoscope). Quatre coutures fertiles :

| # | Couture | Statut | Finding |
|---|---|---|---|
| 4 | **Meta-loop** : architecture VG = trajectoire dans dynamoscope. Lyapunov / SINDy / Koopman sur évolution architecture. | ✅ | spectral_slope -1.85 universel, φ_id émergence, bimodal basin (factor 6× sep) |
| 1 | **Predictor swap** : VG remplace `prediction.py` LatentMLP. Growth triggered par surprise rollout n-step. | 🟡 POC | Pareto compétitif à 50% params, variance élevée |
| 2 | **Reverse trigger** : signaux dynamoscope (φ_id, Lyapunov, surprise) déclenchent mutations VG, remplacent plateau detection naïf. | ✅ | emergence/lyapunov +2.6% acc vs loss_plateau (0/3 fires sur baseline) |
| 3 | **Adaptive encoder** : VG-CNN remplace ResNet/DINOv2 fixe dans encoder dynamoscope. | ✅ | **vg_tiny 498p 6× plus précis que ResNet50 23.5M** sur corr_dim Lorenz |

## Pourquoi cette fusion

VerifyGrowNet et Dynamoscope ancrent dans le même paradigme : *predictive coding* (Rao-Ballard, Friston, JEPA). VG utilise la surprise interne comme signal de croissance (SSVG). Dynamoscope mesure la surprise rollout sur trajectoires. Les deux outils découpent le même substrat avec des couteaux différents.

Au-delà de l'ingénierie, question scientifique : **l'architecture d'un réseau qui s'auto-modifie obéit-elle à des lois dynamiques mesurables ?** Sprint 4 répond oui — loi 1/f² universelle, φ_id > 0, bassins d'attraction multiples.

## Architecture du repo

```
architrace/
├── architrace/              # package Python
│   ├── snapshot.py          # extracteur features architecture → vec 58D
│   ├── bridge.py            # injection vers dynamoscope + helpers analyses
│   ├── trajectory.py        # API haut-niveau (run + analyse en 1 call)
│   └── predictor_vg.py      # option 1 : VG comme predictor rollout
├── experiments/
│   ├── sprint4_meta_loop/   # validation pipeline meta-loop (5 seeds MNIST)
│   └── sprint1_predictor_swap/  # option 1 swap LatentMLP → VG
├── results/                 # outputs analyses
├── paper/                   # paper.tex (compilation des 4 options)
└── tests/
```

## Dépendances

- `adaptive-nn` à `/Users/zac/claude-code-projects/adaptive-nn`
- `dynamoscope` à `/Users/zac/claude-code-projects/dynamoscope`

Pas de fork — repos originaux intacts. architrace = couche d'intégration au-dessus.

## Stack

PyTorch · FastAPI (réutilise dynamoscope server) · NumPy · UMAP · scikit-learn · ripser · pysindy-style sparse regression

## Insights Sprint 4

- spectral_slope = -1.85 ± 0.13 (universel inter-seed)
- φ_id = 0.19, optimal_scale > 1 → émergence causale détectée
- RQA DET = LAM = 0.999 → trajectoire déterministe + laminaire
- Lyapunov ≈ 0.04 → stable contractive
- DMD rank 2-3 → low-rank intrinsèque dans 58D
- Bimodal basin (factor 6× cluster separation)

Détails : `experiments/sprint4_meta_loop/SPRINT4_REPORT.md`
