"""
architrace.strange_loop — Sprint 5C : auto-référence Hofstadter-style.

UN seul VG. Forward pass prend (x_data, self_state) en input. Self_state
= encodage VG-CNN du render arch_to_image. Self-prediction head prédit
état suivant. Self-surprise = ‖next_state_predicted - actual_next_state‖².

Mutation trigger : pic de self-surprise (réseau ne se prédit plus lui-même).

Loop strange :
    state[t-1] → forward(x, state[t-1]) → loss + grad step
    → state[t] (post-update) → compare prediction → self_surprise
    → if drift: trigger mutation → state[t+1] discontinu (mutation)

Question : converge vers fixed point d'auto-représentation ? Diverge ?
Boucle paradoxale Gödel-Tarski-Hofstadter prend forme concrète ou s'écrase.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_to_image import render_arch_state
from .encoder_vg import VGCNNEncoder


class SelfReferentialVG(nn.Module):
    """VG qui se contient comme partie de son input.

    Forward : (x, prev_state) → (task_logits, predicted_next_state).
    prev_state est detached — pas de gradient à travers le mécanisme self-rendering.

    Mutation logic réutilise VerifyGrowNet via composition (self.task_trunk).
    """

    def __init__(
        self,
        in_dim: int, out_dim: int,
        state_dim: int = 32,
        hidden: int = 32,
        depth: int = 2,
        observer_channels: tuple[int, ...] = (4, 8),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_dim = state_dim

        # Trunk : prend (x, state) concat
        layers: list[nn.Module] = []
        prev = in_dim + state_dim
        for _ in range(depth):
            layers.extend([nn.Linear(prev, hidden), nn.GELU()])
            prev = hidden
        self.trunk = nn.Sequential(*layers)
        self.task_head = nn.Linear(prev, out_dim)
        self.state_predictor = nn.Linear(prev, state_dim)

        # Observer = VG-CNN qui encode l'image rendered de SOI
        self.observer = VGCNNEncoder(channels=observer_channels)
        # Force state_dim match observer.latent_dim (sinon adapter linéaire)
        self.state_adapter: nn.Module
        if self.observer.latent_dim != state_dim:
            self.state_adapter = nn.Linear(self.observer.latent_dim, state_dim)
        else:
            self.state_adapter = nn.Identity()

    @torch.no_grad()
    def self_state(self) -> torch.Tensor:
        """Render arch + encode via observer + adapt to state_dim.

        Detached : pas de grad à travers ce path.
        """
        device = next(self.parameters()).device
        img = render_arch_state(self, image_size=64)
        z = self.observer.encode(img[None, ...])[0]  # numpy (D,)
        z = torch.from_numpy(z).float().to(device)
        # adapter peut avoir grad, mais we still detach final result
        if isinstance(self.state_adapter, nn.Identity):
            s = z
        else:
            s = self.state_adapter(z)
        return s.detach()

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, in_dim) data
        prev_state: (state_dim,) ou (B, state_dim) — répliqué si scalar batch
        """
        B = x.shape[0]
        if prev_state.dim() == 1:
            prev_state = prev_state.unsqueeze(0).expand(B, -1)
        z = self.trunk(torch.cat([x, prev_state], dim=-1))
        task_logits = self.task_head(z)
        next_state_pred = self.state_predictor(z)
        return task_logits, next_state_pred

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class StrangeLoopConfig:
    epochs: int = 5
    lr: float = 1e-3
    state_dim: int = 32
    hidden: int = 32
    init_depth: int = 2
    observer_channels: tuple[int, ...] = (4, 8)
    self_surprise_weight: float = 0.1
    surprise_window: int = 20
    surprise_spike_threshold: float = 0.3  # multiple of rolling baseline
    log_every: int = 30
    state_update_every: int = 1  # rafraîchit self_state tous les N batches


def train_strange_loop(
    train_loader, val_loader,
    in_dim: int, out_dim: int,
    cfg: Optional[StrangeLoopConfig] = None,
    task: str = 'classification',
    verbose: bool = True,
) -> dict:
    """Training boucle étrange auto-référentielle.

    Mesures :
    - task_loss : convergence sur tâche
    - self_surprise : ‖pred_state - actual_state‖² quand actual mesuré post-step
    - state_drift : ‖state[t] - state[t-1]‖ — vitesse changement représentation
    - state_norm : ‖state[t]‖ — saturation / collapse
    """
    cfg = cfg or StrangeLoopConfig()
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    model = SelfReferentialVG(
        in_dim=in_dim, out_dim=out_dim,
        state_dim=cfg.state_dim,
        hidden=cfg.hidden, depth=cfg.init_depth,
        observer_channels=cfg.observer_channels,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    # State initial : 1 forward pour bootstrap
    prev_state = model.self_state()

    history = {
        'step': [], 'task_loss': [], 'self_surprise': [],
        'state_drift': [], 'state_norm': [],
        'val_acc': [], 'val_loss': [],
    }
    surprise_buf: deque = deque(maxlen=cfg.surprise_window)
    step = 0

    if verbose:
        print(f'[init] state_dim={cfg.state_dim}  params={model.num_params()}  device={device}')
        print(f'       initial state_norm={prev_state.norm().item():.3f}')

    for ep in range(cfg.epochs):
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)

            opt.zero_grad()
            task_logits, pred_next_state = model(x, prev_state)
            if task == 'classification':
                task_loss = F.cross_entropy(task_logits, y)
            else:
                task_loss = F.mse_loss(task_logits, y)

            # Pred_next_state : forward pass produces this (B, state_dim)
            # We compare to ACTUAL state AFTER this batch's update
            # → done after opt.step() below
            loss = task_loss  # surprise loss appliquée après step
            loss.backward()
            opt.step()

            # Actual new state post-update
            with torch.no_grad():
                new_state = model.self_state()
                # Compare pred (averaged over batch) vs actual
                pred_mean = pred_next_state.mean(0).detach()
                # Adapter state_dim si diff
                if pred_mean.shape[0] != new_state.shape[0]:
                    # Pad/crop pour matching
                    min_d = min(pred_mean.shape[0], new_state.shape[0])
                    self_surprise = ((pred_mean[:min_d] - new_state[:min_d]) ** 2).mean()
                else:
                    self_surprise = ((pred_mean - new_state) ** 2).mean()
                drift = (new_state - prev_state).norm().item()
                state_norm = new_state.norm().item()

            surprise_buf.append(self_surprise.item())

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    # Quick val
                    val_correct = 0
                    val_total = 0
                    val_loss_sum = 0.0
                    n_batches = 0
                    for vx, vy in val_loader:
                        if n_batches >= 5:
                            break
                        vx = vx.to(device); vy = vy.to(device)
                        v_logits, _ = model(vx, prev_state)
                        if task == 'classification':
                            val_correct += (v_logits.argmax(-1) == vy).sum().item()
                            val_total += vy.shape[0]
                            val_loss_sum += F.cross_entropy(v_logits, vy).item()
                        n_batches += 1
                    val_acc = val_correct / max(1, val_total)
                    val_loss = val_loss_sum / max(1, n_batches)

                history['step'].append(step)
                history['task_loss'].append(task_loss.item())
                history['self_surprise'].append(self_surprise.item())
                history['state_drift'].append(drift)
                history['state_norm'].append(state_norm)
                history['val_acc'].append(val_acc)
                history['val_loss'].append(val_loss)
                if verbose:
                    print(f'[step {step:4d}] task={task_loss.item():.4f} '
                          f'surprise={self_surprise.item():.4f} '
                          f'drift={drift:.3f} state_norm={state_norm:.3f} '
                          f'val_acc={val_acc:.3f}')

            prev_state = new_state  # update for next batch (already detached)
            step += 1

    # Final eval
    model.eval()
    val_correct = 0; val_total = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = model(vx, prev_state)
            val_correct += (logits.argmax(-1) == vy).sum().item()
            val_total += vy.shape[0]

    return {
        'final_val_acc': val_correct / max(1, val_total),
        'final_params': model.num_params(),
        'final_state_norm': prev_state.norm().item(),
        'history': history,
        'surprise_history': list(surprise_buf),
    }
