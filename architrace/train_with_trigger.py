"""
architrace.train_with_trigger — wrapper training VG avec trigger pluggable.

Reprend logique verifygrow_nn.train mais factorise plateau detection
en trigger interchangeable (voir architrace.triggers).

Aussi : maintient buffer latents (activations dernière couche) pour
triggers basés dynamoscope (Emergence, Lyapunov).
"""
from __future__ import annotations

import copy
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from verifygrow_nn import VerifyGrowNet, eval_val, bottleneck_layer  # type: ignore

from .triggers import Trigger, TrainHistory


def _capture_latents(model, batch_x: torch.Tensor) -> np.ndarray:
    """Snapshot activations dernière couche cachée (pré-head) sur batch.

    Returns mean over batch dim → (last_hidden_dim,) ndarray.
    Permet à dynamoscope d'analyser évolution représentation latente.
    """
    model.eval()
    with torch.no_grad():
        h = batch_x
        for layer in model.layers:
            h = model._act(layer(h))
        z = h.mean(dim=0).cpu().numpy().astype(np.float32)  # mean batch
    model.train()
    return z


@torch.no_grad()
def _rollout_err(model, val_loader, task: str, horizon: int = 5, max_batches: int = 5) -> float:
    """n-step rollout error proxy : si task = classification, mesure
    erreur "imaginaire" en faisant n itérations head→logits (pas trajectoire vraie).

    Pour task prediction (régression), ce serait rollout vrai. Pour classif,
    on utilise val_loss comme proxy (rollout n'a pas de sens).
    """
    if task == 'classification':
        return eval_val(model, val_loader, task=task, max_batches=max_batches)['loss']
    # Regression : approche rollout naive sur batches
    model.eval()
    errs = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        pred = model(x)
        errs.append(F.mse_loss(pred, y).item())
    model.train()
    return float(np.mean(errs)) if errs else float('inf')


def train_with_trigger(
    model: VerifyGrowNet,
    train_loader,
    val_loader,
    trigger: Trigger,
    *,
    lr: float = 1e-3,
    epochs: int = 8,
    task: str = 'classification',
    grow_k: int = 16,
    max_hidden: int = 256,
    distill_steps: int = 150,
    distill_alpha: float = 0.5,
    verify_extra: int = 30,
    val_tol: float = 2e-3,
    max_rejections: int = 3,
    log_every: int = 50,
    latents_buf_size: int = 200,
    rollout_buf_size: int = 50,
    use_verify: bool = True,
    use_distill: bool = True,
    verbose: bool = False,
) -> dict:
    """Training avec trigger pluggable. Reproduit verifygrow_nn.train mais :
    1. Plateau detection remplacé par trigger.should_grow(history)
    2. Maintient buffers latents + rollout_err pour triggers dynamoscope
    3. Same verify-then-keep + distill machinery
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_buf: deque = deque(maxlen=200)
    rollout_buf: deque = deque(maxlen=rollout_buf_size)
    latents_buf: list[np.ndarray] = []
    val_loss_history: list[float] = []
    last_mut_step = -10000
    rejections = 0
    frozen = False
    teacher: Optional[VerifyGrowNet] = None
    distill_remaining = 0
    pending: Optional[dict] = None
    mutations: list[dict] = []
    history: list[dict] = []
    trigger_log: list[dict] = []
    step = 0

    for ep in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            pred = model(x)
            if task == 'classification':
                task_loss = F.cross_entropy(pred, y)
            else:
                task_loss = F.mse_loss(pred, y)
            loss = task_loss

            if teacher is not None and distill_remaining > 0:
                with torch.no_grad():
                    t_pred = teacher(x)
                if task == 'classification':
                    d_loss = F.kl_div(F.log_softmax(pred, -1), F.softmax(t_pred, -1),
                                      reduction='batchmean')
                else:
                    d_loss = F.mse_loss(pred, t_pred)
                loss = loss + distill_alpha * d_loss
                distill_remaining -= 1
                if distill_remaining == 0:
                    teacher = None

            loss.backward()
            opt.step()
            loss_buf.append(task_loss.item())

            # Buffers pour triggers : capture latents fréquent pour atteindre window
            if step % 3 == 0:
                latents_buf.append(_capture_latents(model, x))
                if len(latents_buf) > latents_buf_size:
                    latents_buf[:] = latents_buf[-latents_buf_size:]

            if step % log_every == 0:
                v = eval_val(model, val_loader, task=task, max_batches=5)
                val_loss_history.append(v['loss'])
                history.append({
                    'step': step, 'loss': task_loss.item(),
                    'val_loss': v['loss'], 'val_acc': v.get('acc', 0.0),
                    'params': model.num_params(), 'topology': model.topology(),
                })
                rollout_buf.append(v['loss'])  # proxy : val_loss roll

            # === Trigger evaluation ===
            if (not frozen and pending is None and teacher is None):
                hist = TrainHistory(
                    step=step,
                    loss_buffer=loss_buf,
                    val_loss=val_loss_history[-1] if val_loss_history else float('inf'),
                    val_loss_history=val_loss_history,
                    rollout_err_buffer=rollout_buf,
                    latents_buffer=latents_buf,
                    step_last_mutation=last_mut_step,
                )
                fired, reason = trigger.should_grow(hist)

                if fired:
                    trigger_log.append({'step': step, 'reason': reason})
                    snap = copy.deepcopy(model) if (use_verify or use_distill) else None
                    val_before = eval_val(model, val_loader, task=task,
                                          max_batches=8) if use_verify else None

                    target = bottleneck_layer(model)
                    if model.layers[target].out_features >= max_hidden:
                        cands = [i for i, l in enumerate(model.layers)
                                 if l.out_features < max_hidden]
                        if not cands:
                            frozen = True
                            if verbose:
                                print(f'[step {step:5d}] FROZEN — all at max')
                            step += 1
                            continue
                        target = cands[0]

                    model.widen_layer(target, grow_k, opt, use_gradmax=True)
                    # Reset latents_buf : dim activations a changé après widen
                    latents_buf.clear()
                    if use_distill and snap is not None:
                        teacher = snap.eval()
                        distill_remaining = distill_steps
                    last_mut_step = step

                    if use_verify:
                        pending = {
                            'step': step, 'target': target,
                            'val_before': val_before['loss'],
                            'snapshot': snap,
                            'verify_step': step + distill_steps + verify_extra,
                        }
                        status = 'pending'
                    else:
                        pending = None
                        status = 'no_verify'
                    mutations.append({
                        'step': step, 'action': f'widen L{target} +{grow_k}',
                        'trigger_reason': reason,
                        'val_before': val_before['loss'] if val_before else None,
                        'val_after': None, 'gain': None, 'status': status,
                        'params_after': model.num_params(),
                    })
                    if verbose:
                        print(f'[step {step:5d}] TRY widen L{target}  '
                              f'trigger="{reason[:60]}"  '
                              f'params={model.num_params()}')

            # === verification ===
            if use_verify and pending is not None and step >= pending['verify_step']:
                val_after = eval_val(model, val_loader, task=task, max_batches=8)
                gain = pending['val_before'] - val_after['loss']
                mutations[-1]['val_after'] = val_after['loss']
                mutations[-1]['gain'] = gain

                if gain > val_tol:
                    rejections = 0
                    mutations[-1]['status'] = 'accepted'
                    if verbose:
                        print(f'[step {step:5d}] ✓ ACCEPT  gain={gain:+.4f}  '
                              f'val={val_after["loss"]:.4f}')
                else:
                    snap = pending['snapshot']
                    model.layers = snap.layers
                    model.head = snap.head
                    opt = torch.optim.Adam(model.parameters(), lr=lr)
                    # Reset latents_buf : rollback shape
                    latents_buf.clear()
                    rejections += 1
                    mutations[-1]['status'] = 'rejected'
                    mutations[-1]['params_after'] = model.num_params()
                    if verbose:
                        print(f'[step {step:5d}] ✗ REJECT  gain={gain:+.4f}  '
                              f'rej={rejections}')
                    if rejections >= max_rejections:
                        frozen = True
                pending = None
                loss_buf.clear()

            step += 1

    final = eval_val(model, val_loader, task=task)
    return {
        'final_params': model.num_params(),
        'final_topology': model.topology(),
        'final_val_loss': final['loss'],
        'final_val_acc': final['acc'],
        'mutations': mutations,
        'trigger_log': trigger_log,
        'history': history,
        'frozen': frozen,
        'accepted': sum(1 for m in mutations if m['status'] == 'accepted'),
        'rejected': sum(1 for m in mutations if m['status'] == 'rejected'),
        'trigger_name': trigger.name,
    }
