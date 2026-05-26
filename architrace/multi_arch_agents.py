"""
architrace.multi_arch_agents — Sprint 18C : 3 archs vision avec self-state input.

ResNetSelfAgent, ViTSelfAgent, MLPMixerSelfAgent.
Tous : x_image + prev_state → (task_logits, predicted_next_state).
Compatible test S11.x exaptation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch_to_image import render_arch_state
from .encoder_vg import VGCNNEncoder


class _BaseSelfAgent(nn.Module):
    """Interface commune : forward(x, prev_state) → (task, pred_state)."""
    state_dim: int

    @torch.no_grad()
    def self_state(self) -> torch.Tensor:
        device = next(self.parameters()).device
        img = render_arch_state(self, image_size=64)
        z = self.observer.encode(img[None, ...])[0]
        z_t = torch.from_numpy(z).float().to(device)
        if z_t.shape[0] < self.state_dim:
            z_t = torch.cat([z_t, torch.zeros(self.state_dim - z_t.shape[0], device=device)])
        elif z_t.shape[0] > self.state_dim:
            z_t = z_t[:self.state_dim]
        return z_t.detach()


# =============================================================================
# ResNet style
# =============================================================================
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetSelfAgent(_BaseSelfAgent):
    def __init__(self, in_ch=3, num_classes=10, state_dim=16, hidden=64,
                 observer_channels=(4, 8)):
        super().__init__()
        self.state_dim = state_dim
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(32, 32)
        self.layer2 = BasicBlock(32, 64, stride=2)
        self.layer3 = BasicBlock(64, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        feat_dim = 128
        self.fusion = nn.Sequential(nn.Linear(feat_dim + state_dim, hidden), nn.GELU())
        self.task_head = nn.Linear(hidden, num_classes)
        self.state_predictor = nn.Linear(hidden, state_dim)
        self.observer = VGCNNEncoder(channels=observer_channels, input_size=64)

    def forward(self, x, prev_state):
        B = x.shape[0]
        z = self.layer3(self.layer2(self.layer1(self.stem(x))))
        feat = self.gap(z).flatten(1)
        if prev_state.dim() == 1:
            prev_state = prev_state.unsqueeze(0).expand(B, -1)
        h = self.fusion(torch.cat([feat, prev_state], dim=-1))
        return self.task_head(h), self.state_predictor(h)


# =============================================================================
# ViT style
# =============================================================================
class ViTSelfAgent(_BaseSelfAgent):
    def __init__(self, in_ch=3, num_classes=10, image_size=32, patch_size=4,
                 d_model=64, n_heads=4, n_layers=3, state_dim=16, hidden=64,
                 observer_channels=(4, 8)):
        super().__init__()
        self.state_dim = state_dim
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_ch, d_model, patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                            dim_feedforward=d_model * 2,
                                            batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fusion = nn.Sequential(nn.Linear(d_model + state_dim, hidden), nn.GELU())
        self.task_head = nn.Linear(hidden, num_classes)
        self.state_predictor = nn.Linear(hidden, state_dim)
        self.observer = VGCNNEncoder(channels=observer_channels, input_size=64)

    def forward(self, x, prev_state):
        B = x.shape[0]
        z = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, n_patches, d)
        cls_tok = self.cls.expand(B, -1, -1)
        z = torch.cat([cls_tok, z], dim=1) + self.pos
        z = self.transformer(z)
        feat = z[:, 0]  # CLS
        if prev_state.dim() == 1:
            prev_state = prev_state.unsqueeze(0).expand(B, -1)
        h = self.fusion(torch.cat([feat, prev_state], dim=-1))
        return self.task_head(h), self.state_predictor(h)


# =============================================================================
# MLP-Mixer style
# =============================================================================
class MixerBlock(nn.Module):
    def __init__(self, n_tokens, d_model, hidden_token=64, hidden_channel=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mix = nn.Sequential(
            nn.Linear(n_tokens, hidden_token), nn.GELU(),
            nn.Linear(hidden_token, n_tokens),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, hidden_channel), nn.GELU(),
            nn.Linear(hidden_channel, d_model),
        )

    def forward(self, x):
        # x : (B, n_tokens, d_model)
        y = self.norm1(x).transpose(1, 2)  # (B, d, n_tokens)
        y = self.token_mix(y).transpose(1, 2)
        x = x + y
        y = self.norm2(x)
        x = x + self.channel_mix(y)
        return x


class MixerSelfAgent(_BaseSelfAgent):
    def __init__(self, in_ch=3, num_classes=10, image_size=32, patch_size=4,
                 d_model=64, n_layers=3, state_dim=16, hidden=64,
                 observer_channels=(4, 8)):
        super().__init__()
        self.state_dim = state_dim
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_ch, d_model, patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            MixerBlock(n_patches, d_model) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(nn.Linear(d_model + state_dim, hidden), nn.GELU())
        self.task_head = nn.Linear(hidden, num_classes)
        self.state_predictor = nn.Linear(hidden, state_dim)
        self.observer = VGCNNEncoder(channels=observer_channels, input_size=64)

    def forward(self, x, prev_state):
        B = x.shape[0]
        z = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        feat = z.mean(dim=1)
        if prev_state.dim() == 1:
            prev_state = prev_state.unsqueeze(0).expand(B, -1)
        h = self.fusion(torch.cat([feat, prev_state], dim=-1))
        return self.task_head(h), self.state_predictor(h)
