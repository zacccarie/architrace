"""
architrace.dyadic — Sprint 11 : conscience dyadique.

Hypothèse Vygotsky-style : la conscience-de-soi est relationnelle, pas
individuelle. Émerge dans l'interaction entre deux systèmes, pas en isolation.

Setup :
- Agent A et Agent B sont deux SelfReferentialVG-like
- A modélise B (predicted_B_state)
- B modélise A (predicted_A_state)
- AUCUN agent ne se modélise lui-même
- Loss = task_A + task_B + α * (surprise_A + surprise_B)

Question : mutual modeling produit-il quelque chose que ni self-modeling
(Sprint 5C.1) ni absence (α=0) ne produit ?
- Plus stable ?
- Meilleur transfer ?
- Alignement états A ↔ B ?
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


class DyadicAgent(nn.Module):
    """Un agent dyadique : trunk + task_head + partner_state_predictor.

    Forward prend (x, partner_state) → (task_logits, predicted_partner_next_state).
    """

    def __init__(self, in_dim: int, out_dim: int, state_dim: int = 16,
                 hidden: int = 32, depth: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.state_dim = state_dim

        layers: list[nn.Module] = []
        prev = in_dim + state_dim
        for _ in range(depth):
            layers.extend([nn.Linear(prev, hidden), nn.GELU()])
            prev = hidden
        self.trunk = nn.Sequential(*layers)
        self.task_head = nn.Linear(prev, out_dim)
        # Predit l'état du PARTENAIRE, pas le sien
        self.partner_predictor = nn.Linear(prev, state_dim)

    def forward(self, x: torch.Tensor, partner_state: torch.Tensor):
        B = x.shape[0]
        if partner_state.dim() == 1:
            partner_state = partner_state.unsqueeze(0).expand(B, -1)
        z = self.trunk(torch.cat([x, partner_state], dim=-1))
        task_logits = self.task_head(z)
        pred_partner = self.partner_predictor(z)
        return task_logits, pred_partner

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class DyadicConfig:
    epochs: int = 5
    lr: float = 1e-3
    state_dim: int = 16
    hidden: int = 32
    depth: int = 2
    mutual_weight: float = 1.0  # α : poids du mutual prediction loss
    observer_channels: tuple[int, ...] = (4, 8)
    log_every: int = 30


def _agent_state(agent: DyadicAgent, observer: VGCNNEncoder,
                 image_size: int = 64) -> torch.Tensor:
    """Render agent's arch → encode via observer → state_dim vector."""
    img = render_arch_state(agent, image_size=image_size)
    z = observer.encode(img[None, ...])[0]
    z_t = torch.from_numpy(z).float().to(next(agent.parameters()).device)
    # Pad/crop to state_dim
    if z_t.shape[0] < agent.state_dim:
        pad = torch.zeros(agent.state_dim - z_t.shape[0], device=z_t.device)
        z_t = torch.cat([z_t, pad])
    elif z_t.shape[0] > agent.state_dim:
        z_t = z_t[:agent.state_dim]
    return z_t.detach()


def train_dyadic(
    train_loader_a, val_loader_a,
    train_loader_b, val_loader_b,
    in_dim_a: int, out_dim_a: int,
    in_dim_b: int, out_dim_b: int,
    cfg: Optional[DyadicConfig] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> dict:
    """Entraîne 2 agents A et B avec mutual modeling.

    A solve task A et modélise B's state.
    B solve task B et modélise A's state.

    Returns: dict avec final accs, alignment, surprise levels.
    """
    cfg = cfg or DyadicConfig()
    device = device or torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    agent_a = DyadicAgent(in_dim_a, out_dim_a, cfg.state_dim, cfg.hidden, cfg.depth).to(device)
    agent_b = DyadicAgent(in_dim_b, out_dim_b, cfg.state_dim, cfg.hidden, cfg.depth).to(device)

    # Shared observer (économise params, simplifie)
    observer = VGCNNEncoder(channels=cfg.observer_channels, input_size=64)

    opt = torch.optim.AdamW(
        list(agent_a.parameters()) + list(agent_b.parameters()),
        lr=cfg.lr, weight_decay=1e-4,
    )

    # Initial states
    state_a = _agent_state(agent_a, observer)
    state_b = _agent_state(agent_b, observer)

    history = {
        'step': [], 'task_a_loss': [], 'task_b_loss': [],
        'surprise_a': [], 'surprise_b': [],
        'alignment_ab': [], 'val_acc_a': [], 'val_acc_b': [],
        'state_a_norm': [], 'state_b_norm': [],
    }
    step = 0

    iter_b = iter(train_loader_b)
    for ep in range(cfg.epochs):
        for x_a, y_a in train_loader_a:
            try:
                x_b, y_b = next(iter_b)
            except StopIteration:
                iter_b = iter(train_loader_b)
                x_b, y_b = next(iter_b)

            x_a = x_a.to(device); y_a = y_a.to(device)
            x_b = x_b.to(device); y_b = y_b.to(device)

            # Current states BEFORE forward (targets for partner predictors)
            current_state_a = _agent_state(agent_a, observer)
            current_state_b = _agent_state(agent_b, observer)

            opt.zero_grad()
            # A forward avec partner state = B's previous state
            logits_a, pred_b = agent_a(x_a, state_b)
            # B forward avec partner state = A's previous state
            logits_b, pred_a = agent_b(x_b, state_a)

            task_loss_a = F.cross_entropy(logits_a, y_a)
            task_loss_b = F.cross_entropy(logits_b, y_b)

            # Surprise : A predicts B's CURRENT state ; B predicts A's CURRENT state
            pred_b_mean = pred_b.mean(0)
            pred_a_mean = pred_a.mean(0)

            def _match(p, q):
                if p.shape[0] != q.shape[0]:
                    m = min(p.shape[0], q.shape[0])
                    return ((p[:m] - q[:m]) ** 2).mean()
                return ((p - q) ** 2).mean()

            surprise_a = _match(pred_b_mean, current_state_b)  # A surprised by B's actual
            surprise_b = _match(pred_a_mean, current_state_a)

            loss = task_loss_a + task_loss_b + cfg.mutual_weight * (surprise_a + surprise_b)
            loss.backward()
            opt.step()

            # Update states for next iteration
            state_a = current_state_a
            state_b = current_state_b

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    # Quick val
                    va_correct = 0; va_total = 0; vb_correct = 0; vb_total = 0
                    for i, (vx, vy) in enumerate(val_loader_a):
                        if i >= 3: break
                        vx = vx.to(device); vy = vy.to(device)
                        vl, _ = agent_a(vx, state_b)
                        va_correct += (vl.argmax(-1) == vy).sum().item()
                        va_total += vy.shape[0]
                    for i, (vx, vy) in enumerate(val_loader_b):
                        if i >= 3: break
                        vx = vx.to(device); vy = vy.to(device)
                        vl, _ = agent_b(vx, state_a)
                        vb_correct += (vl.argmax(-1) == vy).sum().item()
                        vb_total += vy.shape[0]

                    # Alignment : Pearson corr between state_a and state_b
                    sa = current_state_a.cpu().numpy()
                    sb = current_state_b.cpu().numpy()
                    min_d = min(len(sa), len(sb))
                    if sa[:min_d].std() > 1e-9 and sb[:min_d].std() > 1e-9:
                        alignment = float(np.corrcoef(sa[:min_d], sb[:min_d])[0, 1])
                    else:
                        alignment = 0.0

                history['step'].append(step)
                history['task_a_loss'].append(task_loss_a.item())
                history['task_b_loss'].append(task_loss_b.item())
                history['surprise_a'].append(surprise_a.item())
                history['surprise_b'].append(surprise_b.item())
                history['alignment_ab'].append(alignment)
                history['val_acc_a'].append(va_correct / max(1, va_total))
                history['val_acc_b'].append(vb_correct / max(1, vb_total))
                history['state_a_norm'].append(float(current_state_a.norm()))
                history['state_b_norm'].append(float(current_state_b.norm()))

                if verbose:
                    print(f'[step {step:4d}] task_a={task_loss_a.item():.3f} '
                          f'task_b={task_loss_b.item():.3f} '
                          f'surp_a={surprise_a.item():.4f} surp_b={surprise_b.item():.4f} '
                          f'align={alignment:.3f} '
                          f'val_a={va_correct/max(1,va_total):.3f} '
                          f'val_b={vb_correct/max(1,vb_total):.3f}')
            step += 1

    # Final eval
    def _eval(agent, loader, partner_state):
        agent.eval()
        c = 0; t = 0
        with torch.no_grad():
            for vx, vy in loader:
                vx = vx.to(device); vy = vy.to(device)
                logits, _ = agent(vx, partner_state)
                c += (logits.argmax(-1) == vy).sum().item()
                t += vy.shape[0]
        return c / max(1, t)

    final_acc_a = _eval(agent_a, val_loader_a, state_b)
    final_acc_b = _eval(agent_b, val_loader_b, state_a)

    return {
        'agent_a': agent_a, 'agent_b': agent_b,
        'state_a_final': state_a, 'state_b_final': state_b,
        'final_acc_a': final_acc_a, 'final_acc_b': final_acc_b,
        'final_surprise_a': history['surprise_a'][-1] if history['surprise_a'] else None,
        'final_surprise_b': history['surprise_b'][-1] if history['surprise_b'] else None,
        'final_alignment': history['alignment_ab'][-1] if history['alignment_ab'] else None,
        'mean_alignment_late': float(np.mean(history['alignment_ab'][-5:])) if len(history['alignment_ab']) >= 5 else None,
        'history': history,
    }
