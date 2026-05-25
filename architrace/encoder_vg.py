"""
architrace.encoder_vg — Sprint 3 : VG-CNN comme encoder adaptatif pour dynamoscope.

Drop-in pour ResNet50Encoder dans dynamoscope/backend/encoder.py.
Interface : name · latent_dim · encode(frames: (N,H,W,3)) -> (N, latent_dim).

Pretrain self-supervised possible (SSVG predictive coding) — sinon random-init
features (souvent suffisant pour analyse trajectoire dynamique).

Question : VG-CNN à ~10k params peut-il extraire des features assez riches
pour que dynamoscope retrouve les mêmes lois dynamiques que ResNet50 (25M params) ?
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from verifygrow_cnn import VerifyGrowCNN  # type: ignore


class VGCNNEncoder:
    """Wrapper compatible interface dynamoscope encoder.

    Penultimate features (avant FC head) = latent vector pour analyse trajectoire.
    Convertit (N, H, W, 3) [0,1] RGB → (N, latent_dim) float32.
    """

    name = "vg_cnn"

    def __init__(
        self,
        in_ch: int = 3,
        channels: tuple[int, ...] = (8, 16, 32),
        device: Optional[torch.device] = None,
        pretrained_path: Optional[str] = None,
        input_size: int = 64,
    ):
        self.device = device or torch.device(
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.input_size = input_size
        # VG-CNN avec head temporaire (sera ignorée pour extraction features)
        self.backbone = VerifyGrowCNN(
            in_ch=in_ch, channels=channels, num_classes=10,
        ).to(self.device)
        if pretrained_path:
            sd = torch.load(pretrained_path, map_location=self.device)
            self.backbone.load_state_dict(sd, strict=False)
        self.backbone.eval()
        # Pré-calcule latent_dim = nombre channels après dernier conv (global avg pool)
        self.latent_dim = self._probe_latent_dim()

    def _probe_latent_dim(self) -> int:
        """Forward 1 frame fake pour découvrir output dim avant FC."""
        with torch.no_grad():
            x = torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
            feat = self._extract_features(x)
            return int(feat.shape[1])

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward backbone jusqu'à global avg pool (avant FC classifier)."""
        h = x
        for conv in self.backbone.convs:
            h = conv(h)
        # Global avg pool spatial → (B, C)
        h = torch.nn.functional.adaptive_avg_pool2d(h, 1).flatten(1)
        return h

    @torch.no_grad()
    def encode(self, frames: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Encode (N, H, W, 3) RGB float [0,1] → (N, latent_dim) float32.

        Auto-resize si frame shape != input_size.
        """
        n = frames.shape[0]
        out: list[np.ndarray] = []
        for i in range(0, n, batch_size):
            chunk = frames[i : i + batch_size]
            t = torch.from_numpy(chunk).float().permute(0, 3, 1, 2).to(self.device)
            # Resize si nécessaire
            if t.shape[-1] != self.input_size or t.shape[-2] != self.input_size:
                t = torch.nn.functional.interpolate(
                    t, size=(self.input_size, self.input_size),
                    mode='bilinear', align_corners=False,
                )
            feats = self._extract_features(t).float().cpu().numpy()
            out.append(feats)
        return np.concatenate(out, axis=0)

    def num_params(self) -> int:
        """Total params modèle (utilisé pour comparaison taille)."""
        return sum(p.numel() for p in self.backbone.parameters())

    def topology(self) -> list[int]:
        """Topologie channels conv (utile pour log mutations)."""
        return self.backbone.topology()


# =============================================================================
# Self-supervised pre-training : predictive coding sur séquences
# =============================================================================
class FrameSequencePredictor(nn.Module):
    """Encoder VG-CNN + head linéaire prédisant z[t+1] depuis z[t].

    Train loss : ‖z_pred[t+1] - z_true[t+1]‖²
    Apprend des features qui capturent **dynamique temporelle** (vs sémantique ImageNet).
    Aligné Friston / Rao-Ballard / JEPA.
    """

    def __init__(self, encoder: VGCNNEncoder):
        super().__init__()
        self.encoder = encoder
        d = encoder.latent_dim
        # Prédicteur résiduel z[t+1] = z[t] + delta
        self.predictor = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(),
            nn.Linear(d * 2, d),
        )

    def forward(self, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_t, z_t_plus_1_pred)."""
        z_t = self.encoder._extract_features(x_t)
        z_pred = z_t + self.predictor(z_t)
        return z_t, z_pred


def pretrain_encoder_ssvg(
    encoder: VGCNNEncoder,
    frame_sequence: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """Self-supervised : entraîne encoder à prédire frame suivante.

    Args:
        frame_sequence: (N, H, W, 3) float [0,1] séquence temporelle continue
        epochs: training epochs
    Returns:
        dict avec loss_curve, final_loss
    """
    device = device or encoder.device
    encoder.backbone.train()
    model = FrameSequencePredictor(encoder).to(device)

    n = frame_sequence.shape[0]
    if n < batch_size + 2:
        raise ValueError(f'sequence too short: {n}')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []

    for ep in range(epochs):
        # Sample batches de pairs (t, t+1) consécutives
        perm = np.random.permutation(n - 1)
        ep_losses = []
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            x_t = frame_sequence[idx]
            x_t1 = frame_sequence[idx + 1]

            t = torch.from_numpy(x_t).float().permute(0, 3, 1, 2).to(device)
            t1 = torch.from_numpy(x_t1).float().permute(0, 3, 1, 2).to(device)

            if t.shape[-1] != encoder.input_size:
                t = torch.nn.functional.interpolate(t, size=(encoder.input_size,)*2, mode='bilinear', align_corners=False)
                t1 = torch.nn.functional.interpolate(t1, size=(encoder.input_size,)*2, mode='bilinear', align_corners=False)

            _, z_pred = model(t)
            with torch.no_grad():
                z_true_next = encoder._extract_features(t1).detach()
            loss = ((z_pred - z_true_next) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())

        avg = float(np.mean(ep_losses))
        losses.append(avg)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f'[ep {ep:3d}] ssvg loss={avg:.5f}')

    encoder.backbone.eval()
    return {
        'loss_curve': losses,
        'final_loss': losses[-1],
        'epochs': epochs,
        'n_params': encoder.num_params(),
    }
