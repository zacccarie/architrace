"""
architrace.mutable_observer — Sprint 5D : observer VG-CNN mutable.

VGCNNEncoder fixe → MutableVGCNNObserver qui peut widen ses convs.
Trigger growth : plateau sur sa propre SSVG loss (predict next state in trace).

Co-évolution : observed VG-MLP grandit ↔ observer VG-CNN grandit.
Question : couplent leurs croissances ? Phase-lock ?
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from verifygrow_cnn import VerifyGrowCNN, _surgeryAdam  # type: ignore


class MutableVGCNNObserver(nn.Module):
    """VG-CNN encoder mutable + state predictor pour SSVG self-training.

    Méthodes :
    - encode(frames) → latents : interface dynamoscope compatible
    - train_ssvg_step(prev_img, curr_img) : 1 step SSVG (predict z[t] from z[t-1])
    - try_grow(opt) : tente widen si SSVG plateau, verify-then-keep
    """

    def __init__(
        self,
        in_ch: int = 3,
        channels: tuple[int, ...] = (4, 8),
        input_size: int = 64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
        self.input_size = input_size
        self.backbone = VerifyGrowCNN(
            in_ch=in_ch, channels=channels, num_classes=10,  # head ignoré
        )
        # Probe latent_dim via dry forward
        with torch.no_grad():
            x_probe = torch.zeros(1, in_ch, input_size, input_size)
            feat = self._features(x_probe)
            self.latent_dim = int(feat.shape[1])
        # State predictor : z[t-1] → z_pred[t]
        self.predictor = nn.Linear(self.latent_dim, self.latent_dim)
        self.to(self.device)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.backbone.convs:
            h = conv(h)
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Compatible interface dynamoscope encoder."""
        n = frames.shape[0]
        out = []
        for i in range(0, n, batch_size):
            chunk = frames[i:i + batch_size]
            t = torch.from_numpy(chunk).float().permute(0, 3, 1, 2).to(self.device)
            if t.shape[-1] != self.input_size:
                t = F.interpolate(t, size=(self.input_size,) * 2, mode='bilinear', align_corners=False)
            out.append(self._features(t).float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def train_ssvg_step(
        self, prev_img: np.ndarray, curr_img: np.ndarray,
        opt: torch.optim.Optimizer,
    ) -> float:
        """1 step SSVG : minimise ‖pred(z[t-1]) - z[t]‖² sur prev,curr image pair.

        Returns: loss valeur.
        """
        prev_t = torch.from_numpy(prev_img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        curr_t = torch.from_numpy(curr_img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        if prev_t.shape[-1] != self.input_size:
            prev_t = F.interpolate(prev_t, size=(self.input_size,) * 2, mode='bilinear', align_corners=False)
            curr_t = F.interpolate(curr_t, size=(self.input_size,) * 2, mode='bilinear', align_corners=False)

        z_prev = self._features(prev_t)
        with torch.no_grad():
            z_curr = self._features(curr_t)
        z_pred = self.predictor(z_prev)
        loss = F.mse_loss(z_pred, z_curr)

        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def topology(self) -> list[int]:
        return [c.out_ch for c in self.backbone.convs]

    @torch.no_grad()
    def try_grow(
        self,
        opt: torch.optim.Optimizer,
        loss_buf: deque,
        plateau_window: int = 30,
        plateau_eps: float = 5e-4,
        grow_k: int = 4,
        max_channels: int = 64,
    ) -> Optional[dict]:
        """Tentative widen conv si SSVG loss plateau.

        Returns dict mutation info ou None si pas plateau.
        """
        if len(loss_buf) < plateau_window:
            return None
        arr = torch.tensor(list(loss_buf))
        half = plateau_window // 2
        slope = (arr[-half:].mean() - arr[:half].mean()).abs().item()
        if slope >= plateau_eps:
            return None

        # Cible : conv le plus étroit
        convs = self.backbone.convs
        target = min(range(len(convs)), key=lambda i: convs[i].out_ch)
        if convs[target].out_ch >= max_channels:
            return None

        old_topo = self.topology()
        old_latent_dim = self.latent_dim
        self.backbone.widen_conv(target, grow_k, opt, use_gradmax=True)

        # Update latent_dim si dernier conv → impacte predictor
        new_latent_dim = self.backbone.convs[-1].out_ch
        if new_latent_dim != old_latent_dim:
            # Expand predictor in/out
            old_pred = self.predictor
            new_pred = nn.Linear(new_latent_dim, new_latent_dim).to(self.device)
            # Init : copy old weights, pad zeros
            with torch.no_grad():
                new_pred.weight.data.zero_()
                new_pred.weight.data[:old_latent_dim, :old_latent_dim] = old_pred.weight.data
                new_pred.bias.data[:old_latent_dim] = old_pred.bias.data
                new_pred.bias.data[old_latent_dim:] = 0.0
            self.predictor = new_pred
            self.latent_dim = new_latent_dim
            # Reset opt state for predictor (param refs changed)
            _surgeryAdam(opt, old_pred, new_pred)

        new_topo = self.topology()
        loss_buf.clear()
        return {
            'old_topology': old_topo,
            'new_topology': new_topo,
            'target_conv': target,
            'grow_k': grow_k,
            'old_latent_dim': old_latent_dim,
            'new_latent_dim': self.latent_dim,
        }
