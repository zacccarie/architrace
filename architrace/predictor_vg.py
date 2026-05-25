"""
architrace.predictor_vg — Option 1 : VG comme rollout predictor pour dynamoscope.

Remplace `LatentMLP` (fixe) par `ResidualVG` (croissance disciplinée).
Trigger growth : plateau sur **rollout n-step error** au lieu de single-step val loss.

Drop-in compatible avec backend/prediction.py :
    train_vg_predictor(latents, epochs=, hidden=, ...) -> (model, mean, std, history)
    rollout_vg(latents, coords_3d, start_idx, horizon, ...) -> dict (même schema)
    fit_vg_predictor(latents, ...) -> dict (même schema)

Question scientifique : VG bat-il MLP fixe sur rollout error / params trade-off ?
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Imports VG depuis adaptive-nn (sys.path setup par __init__.py)
from verifygrow_nn import VerifyGrowNet, _surgeryAdam, bottleneck_layer  # type: ignore


# =============================================================================
# Modèle résiduel
# =============================================================================
class ResidualVG(VerifyGrowNet):
    """z[t+1] = z[t] + VG(z[t]). Predictor résiduel mutable.

    Hérite mutation logic de VerifyGrowNet (widen_layer + prune_layer + GradMax).
    Override forward pour ajout résiduel.

    Note : VG.head doit avoir out_dim == in_dim pour que résiduel matche.
    """
    def __init__(self, dim: int, init_hidden: int = 32, init_depth: int = 2,
                 activation: str = 'gelu'):
        super().__init__(
            in_dim=dim, out_dim=dim,
            init_hidden=init_hidden, init_depth=init_depth,
            activation=activation,
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self._act(layer(h))
        delta = self.head(h)
        return x + delta  # résidu


# =============================================================================
# Config training
# =============================================================================
@dataclass
class VGPredictorConfig:
    """Hyperparams VG predictor. Tunés pour sequence latent prediction."""
    epochs: int = 200
    lr: float = 1e-3
    init_hidden: int = 32
    init_depth: int = 2
    max_hidden: int = 256
    grow_k: int = 16
    val_frac: float = 0.2     # split temporel pour rollout verify
    rollout_horizon: int = 5  # horizon n-step pour verify
    plateau_window: int = 30
    plateau_eps: float = 5e-4
    cooldown: int = 60
    verify_extra: int = 15
    distill_steps: int = 20
    distill_alpha: float = 0.3
    val_tol: float = 1e-4
    max_rejections: int = 3
    log_every: int = 20
    weight_decay: float = 1e-4
    use_gradmax: bool = True
    use_verify: bool = True
    use_distill: bool = True


# =============================================================================
# Rollout error eval
# =============================================================================
@torch.no_grad()
def _rollout_error(
    model: ResidualVG, val_z: torch.Tensor, horizon: int, mean: torch.Tensor, std: torch.Tensor,
) -> float:
    """Erreur moyenne n-step rollout sur slice validation.

    Pour chaque point i de val (sauf horizon derniers) :
      z = val[i]
      pour h in 0..horizon : z = model(z)  (en espace standardisé)
      compare z final vs val[i+horizon]
    Retourne mean L2 error en espace dé-standardisé.
    """
    model.eval()
    n = val_z.shape[0]
    if n <= horizon:
        return float('inf')
    n_starts = n - horizon
    errs = []
    for i in range(n_starts):
        z = val_z[i:i+1]
        z_n = (z - mean) / std
        for _ in range(horizon):
            z_n = model(z_n)
        z_pred = z_n * std + mean
        err = (z_pred - val_z[i+horizon:i+horizon+1]).norm().item()
        errs.append(err)
    model.train()
    return float(np.mean(errs)) if errs else float('inf')


# =============================================================================
# Training loop avec verify sur rollout horizon (pas single-step)
# =============================================================================
def train_vg_predictor(
    latents: np.ndarray,
    cfg: VGPredictorConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Entraîne ResidualVG sur trajectoire latente avec growth via rollout error.

    Pipeline :
    1. Split temporel latents → train (80%) + val (20%)
    2. Standardize sur train mean/std
    3. ResidualVG init (small) avec dim = latent_dim
    4. Loop : minimise MSE single-step sur train
    5. Plateau detection sur rollout horizon error val → tentative widen
    6. Verify : compare rollout err avant/après mutation post distillation
    7. Accept si gain > val_tol sinon rollback

    Retourne dict avec model, mean, std, loss_curve, mutations, snapshots.
    """
    cfg = cfg or VGPredictorConfig()
    n, d = latents.shape
    if n < 20:
        raise ValueError(f"Need >=20 frames, got {n}")

    # Split temporel
    n_val = max(cfg.rollout_horizon + 5, int(n * cfg.val_frac))
    n_train = n - n_val
    train_z = torch.from_numpy(latents[:n_train]).float().to(device)
    val_z = torch.from_numpy(latents[n_train:]).float().to(device)

    # Standardize sur train uniquement
    mean = train_z.mean(0, keepdim=True)
    std = train_z.std(0, keepdim=True).clamp_min(1e-6)
    x = train_z[:-1]
    y = train_z[1:]
    x_n = (x - mean) / std
    y_n = (y - mean) / std

    model = ResidualVG(
        dim=d, init_hidden=cfg.init_hidden, init_depth=cfg.init_depth,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loss_buf: deque = deque(maxlen=cfg.plateau_window)
    last_mut = -cfg.cooldown
    rejections = 0
    frozen = False
    teacher: ResidualVG | None = None
    distill_remaining = 0
    pending: dict | None = None
    mutations: list[dict] = []
    history: list[dict] = []

    if verbose:
        init_err = _rollout_error(model, val_z, cfg.rollout_horizon, mean, std)
        print(f'[init]  topology={model.topology()}  params={model.num_params()}  '
              f'rollout_err={init_err:.4f}')

    for step in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_n)
        task_loss = ((pred - y_n) ** 2).mean()
        loss = task_loss

        # Distillation burst après mutation
        if teacher is not None and distill_remaining > 0:
            with torch.no_grad():
                t_pred = teacher(x_n)
            distill_loss = ((pred - t_pred) ** 2).mean()
            loss = loss + cfg.distill_alpha * distill_loss
            distill_remaining -= 1
            if distill_remaining == 0:
                teacher = None

        loss.backward()
        opt.step()
        loss_buf.append(task_loss.item())

        if step % cfg.log_every == 0:
            history.append({
                'step': step,
                'loss': task_loss.item(),
                'params': model.num_params(),
                'topology': model.topology(),
            })

        # Tentative mutation
        if (not frozen and pending is None and teacher is None
                and len(loss_buf) == cfg.plateau_window
                and step - last_mut >= cfg.cooldown):
            arr = torch.tensor(list(loss_buf))
            half = cfg.plateau_window // 2
            slope = (arr[-half:].mean() - arr[:half].mean()).abs().item()

            if slope < cfg.plateau_eps:
                snap = copy.deepcopy(model) if (cfg.use_verify or cfg.use_distill) else None
                err_before = _rollout_error(model, val_z, cfg.rollout_horizon, mean, std) if cfg.use_verify else None

                target = bottleneck_layer(model)
                if model.layers[target].out_features >= cfg.max_hidden:
                    cands = [i for i, l in enumerate(model.layers)
                             if l.out_features < cfg.max_hidden]
                    if not cands:
                        frozen = True
                        if verbose:
                            print(f'[step {step:4d}] FROZEN — all layers at max')
                        continue
                    target = cands[0]

                model.widen_layer(target, cfg.grow_k, opt, use_gradmax=cfg.use_gradmax)
                if cfg.use_distill and snap is not None:
                    teacher = snap.eval()
                    distill_remaining = cfg.distill_steps
                last_mut = step

                if cfg.use_verify:
                    pending = {
                        'step': step, 'target': target,
                        'err_before': err_before,
                        'snapshot': snap,
                        'verify_step': step + cfg.distill_steps + cfg.verify_extra,
                    }
                    status = 'pending'
                else:
                    pending = None
                    status = 'no_verify'
                mutations.append({
                    'step': step, 'action': f'widen L{target} +{cfg.grow_k}',
                    'err_before': err_before, 'err_after': None, 'gain': None,
                    'status': status, 'params_after': model.num_params(),
                })
                if verbose:
                    eb = f'err_before={err_before:.4f}' if err_before is not None else 'no_verify'
                    print(f'[step {step:4d}] TRY widen L{target} +{cfg.grow_k}  '
                          f'{eb}  params={model.num_params()}')

        # Verification post-distill
        if cfg.use_verify and pending is not None and step >= pending['verify_step']:
            err_after = _rollout_error(model, val_z, cfg.rollout_horizon, mean, std)
            gain = pending['err_before'] - err_after
            mutations[-1]['err_after'] = err_after
            mutations[-1]['gain'] = gain

            if gain > cfg.val_tol:
                rejections = 0
                mutations[-1]['status'] = 'accepted'
                if verbose:
                    print(f'[step {step:4d}] ✓ ACCEPT  gain={gain:+.4f}  '
                          f'err={err_after:.4f}  params={model.num_params()}')
            else:
                snap = pending['snapshot']
                model.layers = snap.layers
                model.head = snap.head
                opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                        weight_decay=cfg.weight_decay)
                rejections += 1
                mutations[-1]['status'] = 'rejected'
                mutations[-1]['params_after'] = model.num_params()
                if verbose:
                    print(f'[step {step:4d}] ✗ REJECT  gain={gain:+.4f}  '
                          f'rejections={rejections}')
                if rejections >= cfg.max_rejections:
                    frozen = True
                    if verbose:
                        print(f'[step {step:4d}] FROZEN — {rejections} rejections')

            pending = None
            loss_buf.clear()

    final_err = _rollout_error(model, val_z, cfg.rollout_horizon, mean, std)

    return {
        'model': model.eval(),
        'mean': mean,
        'std': std,
        'history': history,
        'mutations': mutations,
        'final_train_loss': history[-1]['loss'] if history else None,
        'final_rollout_err': final_err,
        'final_params': model.num_params(),
        'final_topology': model.topology(),
        'frozen': frozen,
        'accepted': sum(1 for m in mutations if m['status'] == 'accepted'),
        'rejected': sum(1 for m in mutations if m['status'] == 'rejected'),
    }


# =============================================================================
# Pipeline complet (fit + surprise + predictability)
# =============================================================================
def fit_vg_predictor(
    latents: np.ndarray,
    cfg: VGPredictorConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Drop-in replacement pour `prediction.fit_predictor`.

    Même schema sortie : loss_curve, surprise, predictability, ... + extras VG-spécifiques.
    """
    res = train_vg_predictor(latents, cfg=cfg, device=device, verbose=verbose)
    model, mean, std = res['model'], res['mean'], res['std']

    # Pleins latents pour surprise per-frame (pas juste train)
    n, d = latents.shape
    x_all = torch.from_numpy(latents[:-1]).float().to(device)
    y_all = torch.from_numpy(latents[1:]).float().to(device)
    x_n = (x_all - mean) / std

    with torch.no_grad():
        pred_n = model(x_n)
        pred = pred_n * std + mean
        err = ((pred - y_all) ** 2).sum(dim=1).sqrt().cpu().numpy()

    baseline_err = np.linalg.norm(latents[1:] - latents[:-1], axis=1)
    surprise = np.concatenate([[0.0], err]).astype(np.float32)
    baseline_surprise = np.concatenate([[0.0], baseline_err]).astype(np.float32)

    predictability = float(1.0 - min(1.0, err.mean() / max(baseline_err.mean(), 1e-9)))

    return {
        'loss_curve': [h['loss'] for h in res['history']],
        'final_loss': float(res['final_train_loss']) if res['final_train_loss'] else None,
        'mean_error': float(err.mean()),
        'std_error': float(err.std()),
        'max_error': float(err.max()),
        'mean_baseline': float(baseline_err.mean()),
        'predictability': predictability,
        'surprise': surprise.tolist(),
        'baseline_surprise': baseline_surprise.tolist(),
        'epochs': cfg.epochs if cfg else VGPredictorConfig().epochs,
        # VG-specific
        'final_topology': res['final_topology'],
        'final_params': res['final_params'],
        'final_rollout_err': res['final_rollout_err'],
        'mutations': res['mutations'],
        'accepted_mutations': res['accepted'],
        'rejected_mutations': res['rejected'],
        'frozen': res['frozen'],
    }


# =============================================================================
# Rollout (drop-in pour prediction.rollout)
# =============================================================================
def _project_via_knn(
    pred_arr: np.ndarray, all_latents: np.ndarray, coords_3d: np.ndarray, k: int = 5,
) -> list[list[float]]:
    """Réutilise approche prediction.py — kNN cosine softmax."""
    all_norm = all_latents / np.maximum(np.linalg.norm(all_latents, axis=1, keepdims=True), 1e-9)
    pred_norm = pred_arr / np.maximum(np.linalg.norm(pred_arr, axis=1, keepdims=True), 1e-9)
    k_eff = min(k, all_latents.shape[0])
    out: list[list[float]] = []
    for p in pred_norm:
        sims = all_norm @ p
        top_k = np.argpartition(-sims, k_eff - 1)[:k_eff]
        w = np.exp(sims[top_k] * 10)
        w /= w.sum()
        coord = (w[:, None] * coords_3d[top_k]).sum(axis=0)
        out.append(coord.tolist())
    return out


def rollout_vg(
    latents: np.ndarray,
    coords_3d: np.ndarray,
    start_idx: int,
    horizon: int = 20,
    cfg: VGPredictorConfig | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Drop-in replacement pour prediction.rollout — utilise VG sous le capot."""
    n, d = latents.shape
    if not (0 <= start_idx < n - 1):
        raise ValueError(f'start_idx {start_idx} out of [0, {n-1})')
    horizon = max(1, min(horizon, n - start_idx - 1, 100))

    res = train_vg_predictor(latents, cfg=cfg, device=device, verbose=verbose)
    model, mean, std = res['model'], res['mean'], res['std']

    z = torch.from_numpy(latents[start_idx:start_idx+1]).float().to(device)
    z_n = (z - mean) / std
    pred_latents: list[np.ndarray] = []
    with torch.no_grad():
        cur = z_n.clone()
        for _ in range(horizon):
            cur = model(cur)
            denorm = (cur * std + mean).cpu().numpy()[0]
            pred_latents.append(denorm)
    pred_arr = np.stack(pred_latents, axis=0)

    ghost_3d = _project_via_knn(pred_arr, latents, coords_3d)

    actual_latents = latents[start_idx+1 : start_idx+1+horizon]
    step_errors = []
    for i in range(min(len(actual_latents), len(pred_arr))):
        err = float(np.linalg.norm(pred_arr[i] - actual_latents[i]))
        step_errors.append(err)

    return {
        'start_idx': int(start_idx),
        'horizon': int(horizon),
        'ghost_3d': ghost_3d,
        'step_errors': step_errors,
        'n_actual_compared': len(step_errors),
        # VG extras
        'vg_final_topology': res['final_topology'],
        'vg_final_params': res['final_params'],
        'vg_accepted': res['accepted'],
        'vg_rejected': res['rejected'],
        'vg_mutations': res['mutations'],
    }
