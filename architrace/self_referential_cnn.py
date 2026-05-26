"""
architrace.self_referential_cnn — Sprint 13A : SelfReferentialVG version CNN.

Scaling validation Sprint 5C.1 sur architecture convolutive + CIFAR-10.

Architecture :
- Conv trunk : Conv-Pool × 2 + global avg pool
- State injection : concat self-state à flattened features avant heads
- Task head + state_predictor

Self-state vector toujours via render arch + VGCNNEncoder observer (shared).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_to_image import render_arch_state
from .encoder_vg import VGCNNEncoder


class SelfReferentialCNN(nn.Module):
    """CNN avec self-state conditioning + state_predictor.

    forward(images, prev_state) → (task_logits, predicted_next_state)
    """

    def __init__(self, in_ch: int = 3, num_classes: int = 10,
                 channels: tuple[int, ...] = (16, 32, 64),
                 state_dim: int = 16,
                 hidden: int = 64,
                 observer_channels: tuple[int, ...] = (4, 8)):
        super().__init__()
        self.state_dim = state_dim
        # Conv trunk
        layers = []
        prev = in_ch
        for c in channels:
            layers.extend([
                nn.Conv2d(prev, c, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            prev = c
        self.trunk = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # After GAP : (B, channels[-1])
        feat_dim = channels[-1]
        # Concat with state, project
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + state_dim, hidden),
            nn.GELU(),
        )
        self.task_head = nn.Linear(hidden, num_classes)
        self.state_predictor = nn.Linear(hidden, state_dim)

        # Observer for self-state
        self.observer = VGCNNEncoder(channels=observer_channels, input_size=64)
        # State adapter si latent_dim != state_dim
        if self.observer.latent_dim != state_dim:
            self.state_adapter = nn.Linear(self.observer.latent_dim, state_dim)
        else:
            self.state_adapter = nn.Identity()

    @torch.no_grad()
    def self_state(self) -> torch.Tensor:
        """Render arch + encode observer + adapter."""
        device = next(self.parameters()).device
        img = render_arch_state(self, image_size=64)
        z = self.observer.encode(img[None, ...])[0]
        z_t = torch.from_numpy(z).float().to(device)
        if isinstance(self.state_adapter, nn.Identity):
            s = z_t
        else:
            s = self.state_adapter(z_t)
        return s.detach()

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor):
        B = x.shape[0]
        feat = self.gap(self.trunk(x)).flatten(1)  # (B, feat_dim)
        if prev_state.dim() == 1:
            prev_state = prev_state.unsqueeze(0).expand(B, -1)
        z = self.fusion(torch.cat([feat, prev_state], dim=-1))
        task = self.task_head(z)
        pred = self.state_predictor(z)
        return task, pred

    def num_params(self) -> int:
        # Hors observer (qui est fixe random-init via VGCNNEncoder.backbone)
        observer_params = set(id(p) for p in self.observer.backbone.parameters())
        return sum(p.numel() for p in self.parameters() if id(p) not in observer_params)
