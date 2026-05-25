# Sprint 1 — VG predictor vs LatentMLP fixe

**Date** : 2026-05-26
**Statut** : POC validé, refinement nécessaire avant intégration dynamoscope production

## Question

Le `LatentMLP` de `dynamoscope/backend/prediction.py` peut-il être remplacé par un `ResidualVG` (VerifyGrowNet à croissance disciplinée) ?
La forme résiduelle `z[t+1] = z[t] + VG(z[t])` permet à VG d'apprendre la dynamique du système.

Si oui, deux gains attendus :
1. **Moins de params** : VG démarre petit, grandit seulement si gain mesurable
2. **Auto-tuning** : pas de choix manuel de `hidden`, le réseau trouve sa taille

## Architecture

```python
class ResidualVG(VerifyGrowNet):
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = self._act(layer(h))
        delta = self.head(h)
        return x + delta  # résidu pour stabilité
```

**Trigger growth** : plateau sur **rollout n-step error** (horizon=5) calculé sur slice validation temporelle (20% derniers points). Différent de VG classique qui plateau sur train loss.

**Verify-then-keep** : compare rollout error horizon=5 avant vs après distillation post-mutation. Reject si gain < `val_tol=1e-4`.

## Bench

3 systèmes × 3 seeds × n=500 points :
- Lorenz (3D ODE chaotique)
- Rössler (3D ODE chaos funnel)
- Van der Pol (limit cycle, Takens-embedded en 3D)

Compare contre `LatentMLP` (**uncapped**, version corrigée pour low-dim) avec hidden ∈ {32, 128, 256}.

### Résultats per-system (mean ± std)

| system | MLP-s (227p) | MLP-m (899p) | MLP-l (1795p) | VG | VG params |
|---|---|---|---|---|---|
| **lorenz** | 0.794±0.034 | 0.904±0.008 | 0.920±0.010 | 0.887±0.032 | **483±45** |
| **rossler** | 0.794±0.032 | 0.878±0.018 | 0.900±0.015 | 0.820±0.145 | **430±92** |
| **van_der_pol** | 0.300±0.056 | 0.556±0.022 | 0.675±0.013 | 0.557±0.271 | **515±157** |

Métrique = predictability = 1 - mean_err/baseline_err. 1.0 parfait, 0 = identity baseline.

### Lectures clés

**1. Pareto compétitif** — VG sur 50-55% des params de MLP-medium pour performance comparable.

| system | MLP-m perf | VG perf | params ratio | dominance |
|---|---|---|---|---|
| lorenz | 0.904 (899p) | 0.887 (483p) | 0.54× | ⚖ équivalent |
| rossler | 0.878 (899p) | 0.820 (430p) | 0.48× | ⚖ légèrement under |
| van_der_pol | 0.556 (899p) | 0.557 (515p) | 0.57× | ✅ égal sur 57% params |

**2. Coups d'éclat occasionnels** — sur certains seeds VG explose les baselines :

| run | VG | MLP-medium | MLP-large | verdict |
|---|---|---|---|---|
| lorenz seed=1 | 0.919 (515p) | 0.892 (899p) | 0.916 (1795p) | VG bat MLP-l, 28% params |
| rossler seed=1 | 0.927 (547p) | 0.880 (899p) | 0.921 (1795p) | VG bat MLP-l, 30% params |
| van_der_pol seed=0 | **0.826 (707p)** | 0.525 (899p) | 0.657 (1795p) | **VG +25% vs MLP-l à 40% params** |

**3. Variance VG élevée** — verify-then-keep accepte des mutations qui ne s'avèrent pas bonnes long-terme :

| system | VG std | Worst seed |
|---|---|---|
| van_der_pol | **0.271** | seed=2 : 0.186 (collapse) |
| rossler | 0.145 | seed=2 : 0.614 |
| lorenz | 0.032 | seed=0 : 0.844 |

L'instabilité augmente avec la difficulté du système. Lorenz très propre (std=0.032), Van der Pol (limit cycle Takens-embedded) très variable.

## Découverte secondaire — bug dans `LatentMLP` original

```python
class LatentMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        h = min(hidden, dim)   # ← THROTTLE
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            nn.GELU(),
            nn.Linear(h, dim),
        )
```

Le cap `h = min(hidden, dim)` était silent killer sur systèmes low-dim :
- Pour Lorenz (dim=3) : h=3 quelle que soit la valeur passée → modèle 24 params au lieu de 899
- Predictability ≈ 0 sur tous systèmes 3D

En production dynamoscope, latents = encoder ResNet/ViT/DINOv2/CLIP → dim ∈ [384, 2048] → cap jamais hit. Mais sur traces architecture VG (58D Sprint 4) ou traces synthetiques low-dim, ce cap réduit silencieusement la capacité.

**Recommandation** : retirer le cap dans `dynamoscope/backend/prediction.py:29` ou rendre explicite (`hidden_actual = max(hidden, 2*dim)` par ex).

## Verdict Sprint 1

VG predictor est **POC viable** mais **pas drop-in production-ready** :

✅ **Force** : params 50% inférieurs pour performance équivalente sur Lorenz/Rossler
✅ **Force** : sur certaines conditions init bat les modèles fixes les plus gros
❌ **Faiblesse** : variance élevée — verify-then-keep trop permissif sur task prediction
❌ **Faiblesse** : test sur dim=3 uniquement — pas validé sur dim=384-2048 cas réel

## Suivi — Sprint 1.1 (refinements)

- [ ] **Verify plus strict** : require gain rollout > k×std(buf) au lieu de simple val_tol absolu
- [ ] **Multi-horizon verify** : rollout error à horizons {3, 5, 10} weighted
- [ ] **Test sur dim haute** : injecter trace 58D Sprint 4 ou latents video réels
- [ ] **GradMax alignement** (P4) : VG init du widen aligné gradient ∂rollout_err
- [ ] **Submit fix to dynamoscope** : LatentMLP cap explicit ou retiré

## Suivi — Sprint 1.2 (intégration)

- [ ] Endpoint dynamoscope `/api/rollout/{cache_key}?predictor=vg` flag opt-in
- [ ] Tab `predict` frontend toggle MLP/VG, ghost ground-truth overlay
- [ ] Persist VG mutation log dans réponse JSON pour viz

## Suivi — Sprint 1.3 (paper figure)

- [ ] Plot Pareto params × predictability sur 5 systèmes × 10 seeds
- [ ] Ablation : VG sans verify, VG sans distill, VG sans GradMax → confirmer triage Sprint 4 ablation_results.json

## Code livré

| Fichier | Rôle |
|---|---|
| `architrace/predictor_vg.py` | `ResidualVG`, `train_vg_predictor`, `fit_vg_predictor`, `rollout_vg` |
| `experiments/sprint1_predictor_swap/bench.py` | bench VG vs MLP uncapped sur systèmes synthétiques |
| `results/sprint1/bench_report.md` | rapport multi-seed |
| `results/sprint1/bench_results.json` | données brutes |

## Options Sprint suivantes

- ✅ Option 4 (meta-loop) — validé Sprint 4
- 🟡 Option 1 (predictor swap) — **POC validé ce sprint, refinements identifiés**
- ⬜ Option 2 (reverse trigger : dynamoscope → VG)
- ⬜ Option 3 (VG comme encoder adaptatif)
