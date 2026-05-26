# Glossaire architrace — termes, modèles, variables, notations

Document exhaustif. Toute notion utilisée dans les 27 sprints (architrace, paper, expériences) est définie ici.

---

## 1. Modèles construits par nous

### 1.1 Composants fondamentaux

| Nom | Fichier | Rôle |
|---|---|---|
| **VG** (VerifyGrowNet) | `predictor_vg.py` | Réseau adaptatif à croissance disciplinée. Mutation via verify-then-keep (Sprint 1 baseline) |
| **ResidualVG** | `predictor_vg.py` | Replacement de `prediction.py` dans Dynamoscope avec architecture VG résiduelle (Sprint 1) |
| **VGCNNEncoder** | `encoder_vg.py` | Backbone CNN tiny (~500 params) pour encoder images architecturales. 6× plus précis que ResNet50 sur tâches structurelles (Sprint 3) |
| **MutableVGCNNObserver** | `mutable_observer.py` | Variante VGCNNEncoder dont la topologie peut grandir au runtime |

### 1.2 Agents self-référentiels

| Nom | Fichier | Architecture body | Use case |
|---|---|---|---|
| **SelfReferentialVG** | `strange_loop.py` | MLP avec fusion `(input, prev_state) → (logits, predicted_state)` | Sprint 5C/11.x MNIST baseline |
| **SelfReferentialCNN** | `self_referential_cnn.py` | CNN simple avec fusion idem | Sprint 16B CIFAR-100 |
| **ResNetSelfAgent** | `multi_arch_agents.py` | ResNet (3 BasicBlocks) | Sprint 18C cross-arch CIFAR-10 |
| **ViTSelfAgent** | `multi_arch_agents.py` | Vision Transformer (3 layers, 4 heads, patch=4) | Sprint 18C |
| **MixerSelfAgent** | `multi_arch_agents.py` | MLP-Mixer (3 blocks, patch=4) | Sprint 18C |
| **DyadicAgent** | `dyadic.py` | Deux SelfReferentialVG s'observant mutuellement | Sprint 11.2 symétrie self/other |

### 1.3 LLM-side (Sprints 14-20)

| Nom | Fichier | Description |
|---|---|---|
| **PerTokenGPT** | `sprint18b_llm_per_token/per_token.py` | GPT from scratch (128 d_model, 4 heads, 4 layers) avec self-coherence per-token |
| **TinyGPT** | `sprint17b_llm_pretrain/pretrain.py` | Très petit GPT (0.83M params) pour test pretrain-then-tune |
| **scale_1b.py** | `sprint19_llm_scale_1b/` | Wrapper LoRA autour HuggingFace causal LMs (pythia / GPT-2 family) |
| **contrastive.py** | `sprint20c_contrastive/` | LoRA + InfoNCE per-token aux loss |
| **world_exaptation.py** | `sprint20d_world_exaptation/` | Phase 1 predictor-only sur Lorenz, phase 2 LLM text + InfoNCE |
| **llm_world_model.py** | `sprint20f_llm_world_model/` | Phase 1 LLM entier sur Lorenz via inputs_embeds bypass, phase 2 text |
| **world_task.py** | `sprint20g_world_task/` | LLM entraîné directement comme world model (Lorenz regression) + InfoNCE |

---

## 2. Modèles externes (benchmarks / comparaisons)

### 2.1 Pré-entraînés HuggingFace

| Nom | Params | Provenance | Sprint(s) utilisé |
|---|---|---|---|
| **DistilGPT-2** | 82M | HF `distilgpt2` (knowledge-distilled GPT-2) | S16C fine-tune (échec) |
| **EleutherAI/pythia-410m** | 405M | Pythia suite, decoder-only | S19, S20A-G |
| **EleutherAI/pythia-1.4b** | 1.4B | Pythia | S19, S20C, S20DE |
| **EleutherAI/pythia-2.8b** | 2.8B | Pythia | S19, S20A |
| **EleutherAI/pythia-6.9b** | 6.9B | Pythia | S19 ext |
| **gpt2-large** | 774M | OpenAI GPT-2 large (fallback non utilisé) | S19 candidat fallback |

### 2.2 Comparateurs visuels

| Nom | Params | Rôle |
|---|---|---|
| **ResNet50** | 25M | Baseline visuel ImageNet. Comparé contre VGCNNEncoder sur tâches structurelles (Sprint 3 : VG tiny 6× plus précis sur analyse de poids) |

### 2.3 Outils externes

| Nom | Rôle |
|---|---|
| **Dynamoscope** | Outil interne (ailleurs dans projects/) pour analyse trajectoires : Lyapunov, RQA, spectral, SINDy. Backend appelé via HTTP par `bridge.py` |
| **PyPhi** | Calcul IIT φ (non utilisé — notre `φ_id` est approximation) |
| **peft** | HuggingFace : LoRA adapters fine-tuning |
| **transformers** | HuggingFace : loading pythia/GPT-2 |
| **sklearn fetch_20newsgroups** | Corpus texte pour fine-tune LLM |
| **torchvision CIFAR-10/100/MNIST** | Datasets vision |

---

## 3. Variables, hyperparamètres, constantes

### 3.1 Variables d'architecture

| Symbole | Sens | Typique |
|---|---|---|
| `d_model` | Dimension du hidden state d'un transformer ou MLP body | 64 (toy), 128 (PerTokenGPT), 1024 (pythia-410M), 2048 (pythia-1.4B), 2560 (pythia-2.8B), 4096 (pythia-6.9B) |
| `state_dim` | Dimension de la self-state (output de `state_proj`) | 16 (vision), 32 (S19), 64 (S20C-G) |
| `n_heads` | Nombre d'attention heads | 4-32 selon scale |
| `n_layers` | Nombre de couches transformer | 3-32 |
| `vocab_size` | Taille vocabulaire tokenizer | ~50k (pythia), ~50k (GPT-2), variable (char-level) |
| `seq_len` | Longueur séquence d'entrée | 128 (LLM), 32×32 patches (vision) |
| `patch_size` | Taille patch ViT/Mixer | 4 |
| `r` (LoRA) | Rang adapter LoRA | 16 |
| `lora_alpha` | Scaling LoRA | 32 (donc effective scaling = α/r = 2) |
| `lora_dropout` | Dropout adapter | 0.1 |

### 3.2 Variables d'entraînement

| Symbole | Sens | Typique |
|---|---|---|
| **α** (alpha, `weight`) | Poids du aux loss (self-coherence) | 0.0 (baseline), 0.05-0.5 (sweep S20C), 0.1 (sweet spot), 1.0 (S11.x vision), 2.0 |
| `lr` | Learning rate AdamW | 1e-3 (vision), 3e-4 (LLM LoRA), 5e-6 (LLM full ft) |
| `weight_decay` | L2 regularization | 1e-4 |
| `BATCH` (batch size) | Taille batch | 4 (LLM), 16-32 (vision) |
| `max_steps` | Nb steps phase 2 | 40-100 |
| `world_steps` | Nb steps phase 1 (warmup) | 100-300 |
| `temperature` (T, τ) | Température InfoNCE | 0.1 |
| `inject_noise` | Bruit ajouté au state pour drift normalization S15C | 0.05 |
| `PHASE1_EPOCHS` | Epochs phase 1 (vision) | 3 |
| `PHASE2_EPOCHS` | Epochs phase 2 (vision) | 2 |

### 3.3 Métriques de sortie

| Symbole | Sens | Unité |
|---|---|---|
| **val_pp** (perplexity) | exp(val loss) sur validation set LLM | sans unité, ↓ better |
| **val_mse** | Mean squared error sur validation (regression) | sans unité, ↓ better |
| **val acc / final_acc** | Accuracy classification | ∈ [0, 1] |
| **surp** (surprise) | Valeur instantanée du aux loss (MSE ou InfoNCE) | sans unité |
| **init_surp** | Moyenne surp sur 10 premiers steps | indicateur exaptation |
| **final_surp** | Moyenne surp sur 10 derniers steps | indicateur convergence |
| **Δ pp** | `val_pp(α=w) - val_pp(α=0)`, scalar | ↓ better |
| **Δ mse** | Idem en MSE | ↓ better |
| **speedup** | `init_surp(fresh) / init_surp(world)` | >1 = exaptation positive |
| **lm_loss** | Cross-entropy LM loss | logarithm, ↓ better |
| **task_loss** | Loss principal (CE classification ou MSE regression) | ↓ better |
| **train_time** | Wall-clock time | secondes |
| **n_params** | Total params modèle | comptés via `sum(p.numel())` |
| **n_trainable** | Params modifiés en training (LoRA actif uniquement) | ≪ n_params en LoRA |

### 3.4 Dataset constants

| Variable | Valeur | Sprint |
|---|---|---|
| `n_train` (LLM 20news) | 400 | S19+ |
| `n_val` (LLM 20news) | 80 | S19+ |
| `n_train` (Lorenz seqs) | 400 | S20G |
| `n_val` (Lorenz seqs) | 80 | S20G |
| `n_train` (vision CIFAR-10) | 4000 | S18C |
| `n_val` (vision CIFAR-10) | 1500 | S18C |
| `text corpus tokens` | ~26000 (20news 80 docs) | S19+ |
| `lorenz n_points` | 2000-4000 | S20D/F/G |
| `lorenz dt` | 0.01 | S20 |

---

## 4. Notations mathématiques

### 4.1 Self-coherence aux loss (vision S11.x, two-halves)

```
surp = ||predictor(z_t) − z_{t+1}.detach()||²

avec z_t = self_state() à step t
     predictor = MLP(state_dim → state_dim)
```

`detach()` empêche le gradient de remonter par le target → on évite la trivial solution `predictor(x)=x` qui shorts-circuit le signal.

### 4.2 Two-halves MSE recipe (S19, échec)

```
h ∈ R^{B×T×d_model}    hidden state output transformer
half = T // 2
sf = state_proj(h[:, :half].mean(dim=1))      # mean first half
ss = state_proj(h[:, half:].mean(dim=1))      # mean second half
pred = predictor(sf)
surp = ||pred − ss||²
total_loss = lm_loss + α · surp
```

Compresse l'info de la première moitié de séquence pour prédire la moyenne de la seconde. Empiriquement = signal mal aligné avec next-token prediction.

### 4.3 InfoNCE per-token (S20C, succès marginal)

```
h ∈ R^{B×T×d_model}
s = state_proj(h)                  # (B, T, state_dim) projection
ŝ = predictor(s)                   # (B, T, state_dim) prédiction

pred_flat   = ŝ[:, :-1].reshape(N, state_dim)   N = B(T-1)
target_flat = s[:, 1:].detach().reshape(N, state_dim)

p = pred_flat / ||pred_flat||₂
t = target_flat / ||target_flat||₂

logits = p · tᵀ / τ                  # (N, N) similarity matrix
labels = arange(N)                    # diagonale = positifs
surp = CrossEntropy(logits, labels)
```

Contrastive : chaque `ŝ[b, t]` doit être plus similaire à `s[b, t+1]` qu'à tout autre `s[b', t']` du batch. Température τ contrôle sharpness.

### 4.4 World→Self exaptation speedup

```
init_surp_fresh = mean(surp[0:10] avec predictor random init)
init_surp_world = mean(surp[0:10] avec predictor pre-trained sur world)
speedup = init_surp_fresh / init_surp_world
```

> 1 = predictor pré-entraîné sait mieux prédire self → exaptation. < 1 = inverse (cas ViT). ≈ 1 = pas d'exaptation (cas LLM).

### 4.5 Perplexité

```
perplexity = exp(loss_LM) = exp(-1/N · Σ log p(y_i | x_i))
```

↓ better. Mesure standard pour LLM. Sensibilité : variance ~1 sur petit corpus 26k tokens.

### 4.6 Drift normalization (S15C)

```
current_noisy = current_state + ε·noise         ε = 0.05
surp = ||predictor(prev_state) − current_noisy||²
```

Le bruit injecté empêche que self-coherence triviale = `predictor(x)=x` se forme via paramètres figés. C'est ce qui vindique S11.x (le bonus reste 74× même avec drift normalization).

### 4.7 Mesures dynamiques (Dynamoscope)

| Métrique | Formule (informelle) |
|---|---|
| `lambda_lyap` | Plus grand exposant Lyapunov, mesure chaos |
| `spectral_slope` | Pente de la densité spectrale en log-log, ~-2 = 1/f² universal |
| `phi_id` | Approximation Φ d'IIT, mesure d'intégration informationnelle |
| `rqa_det` | Recurrence quantification — déterminisme |
| `rqa_lam` | Idem — laminarité |
| `sindy_complexity` | Nombre de termes non-nuls dans découverte SINDy |

### 4.8 Lorenz attractor (test world)

```
dx/dt = σ(y - x)         σ = 10
dy/dt = x(ρ - z) - y     ρ = 28
dz/dt = xy - βz          β = 8/3
dt = 0.01
n_points = 2000-4000
```

Standardisation : `coords = (coords − mean) / std`. Padding au state_dim si nécessaire (3 coords + zeros).

---

## 5. Datasets

| Nom | Type | Taille | Usage sprint(s) |
|---|---|---|---|
| **MNIST** | 28×28 grayscale, 10 classes | 60k train + 10k val | Sprints baseline vision (5C, 7, 11.x, 15C) |
| **FashionMNIST** | 28×28, 10 classes | 60k+10k | Transfer test (Sprint 6) |
| **CIFAR-10** | 32×32 RGB, 10 classes | 50k+10k | Sprint 18C cross-arch |
| **CIFAR-100** | 32×32 RGB, 100 classes | 50k+10k | Sprint 16B scale test |
| **20 newsgroups** (sklearn) | Texte newsgroups Usenet | 11k docs | LLM Sprints (15D, 18B, 19, 20) — sous-échantillon 80 docs ~26k tokens |
| **Lorenz attractor** | 3D trajectory continu | n_points custom | Sprint 11.3 world, S20D/F/G |

---

## 6. Termes théoriques cités

| Terme | Provenance | Définition courte |
|---|---|---|
| **Strange loop** | Hofstadter, *I Am a Strange Loop* (2007) | Système qui se référence lui-même via niveaux hiérarchiques imbriqués |
| **Exaptation** | Gould & Vrba (1982) | Trait évolutif détourné de sa fonction originelle vers une autre |
| **Information Bottleneck (IB)** | Tishby, Pereira, Bialek (1999) | Principe : optimal representation = compress input while preserving output info |
| **Free Energy Principle (FEP)** | Friston (2010) | Brains minimize variational free energy = prediction error |
| **Mead-Vygotsky-Hegel** | Sociogenèse de la conscience | Self emerges via internalisation du regard d'autrui |
| **Spinozan modes** | Spinoza, *Ethica* | Une substance, infinis modes. Self/other/world = modes différents d'une seule computation |
| **Whitehead process** | Whitehead, *Process and Reality* | Identité = trajectoire, pas substance |
| **Gödel-Tarski** | Logique : truth ≠ provability | Observateur récursif ne capture pas tout son objet |
| **Self-organized criticality** | Bak, Tang, Wiesenfeld (1987) | Systèmes naturellement → état critique, signature 1/f² spectral |
| **Neural Tangent Kernel (NTK)** | Jacot, Gabriel, Hongler (2018) | MLP wide limit = kernel method ; smoothness du kernel ↔ training stability |
| **CPC (Contrastive Predictive Coding)** | van den Oord et al. (2018) | InfoNCE on temporal predictions → representation learning |
| **SimCLR** | Chen et al. (2020) | Contrastive learning augmentation views → visual representation |
| **Emergent capabilities** | Wei et al. (2022) | Capacités absentes en petit modèle, apparues à scale (phase transition) |
| **In-context learning** | Brown et al. (2020, GPT-3) | Few-shot via prompt sans fine-tune |
| **Chain-of-thought** | Wei et al. (2022) | Génération raisonnement intermédiaire → meilleur perf tâches complexes |
| **Manifold hypothesis** | ML folklore | Données high-dim concentrent sur sous-variété low-dim |

---

## 7. Verdicts qualitatifs (couleur des sprints)

| Code | Sens |
|---|---|
| **green / major** | Finding solide, reproductible, magnitude significative |
| **yellow / partial** | Finding partiel, nuancé, ou avec caveat |
| **orange / fail** | Échec faible / inconclusif |
| **red / fail** | Échec net, hypothèse réfutée |
| **rescue** | Sprint qui sauve un finding précédent (S20C rescue S19) |
| **reframe** | Sprint qui change l'interprétation (S20E reframe S20D) |

---

## 8. Notation des sprints

Format : `S{n}[{lettre}]` ou `Sprint {n}[{lettre}]`.

Exemple : `S15C` = Sprint 15, sous-expérience C (drift normalization vindication).

Lettres : A=initial, B/C/D=variations, parfois sous-numérotation `S11.3` (Sprint 11 phase 3).

---

## 9. Comment lire un report sprint

Chaque sprint produit deux fichiers dans `results/sprint{n}/` :

| Fichier | Contenu |
|---|---|
| `results_*.json` | Dictionnaire python `{rows: [...], args: {...}}` avec une entrée par (condition × seed) |
| `report_*.md` | Markdown auto-généré : tableau aggregate + verdict |

Le verdict en bas du report applique un seuil binaire qui peut masquer des nuances. Toujours lire la table aggregate (mean ± std) plutôt que le verdict pour interprétation rigoureuse.
