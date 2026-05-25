"""
architrace.arch_to_image — Sprint 5 : rend l'état architecture VG en image fixed-size.

Permet à VG-CNN encoder d'observer VG-MLP architecture comme stimulus visuel.
Invariant à croissance : output toujours (H, W, 3) même si layers grow.

Encoding :
- Channel R : magnitudes |weights| de tous layers tiled puis pad/crop 64x64
- Channel G : magnitudes |gradients| (si disponibles) tiled
- Channel B : magnitudes |biases| tiled

Méta-loop : Sprint 4 mesurait architecture via feature vec 58D direct.
Sprint 5B mesure via render image → VG-CNN encode. Test : encoder visuel
capture-t-il les mêmes lois dynamiques que feature vec direct ?
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _to_square_grid(arr: np.ndarray, target_size: int) -> np.ndarray:
    """Tile/pad/crop 1D ou 2D array vers grille target_size × target_size.

    Si flat n_elements > target², subsample uniforme.
    Si < target², pad zéros.
    """
    if arr.ndim == 1:
        n = arr.shape[0]
        side = int(math.ceil(math.sqrt(n)))
        padded = np.zeros(side * side, dtype=np.float32)
        padded[:n] = arr
        arr = padded.reshape(side, side)

    # arr is now 2D
    h, w = arr.shape
    # Resize to target_size via nearest neighbor (simple, deterministic)
    if h == target_size and w == target_size:
        return arr.astype(np.float32)

    # Pad to square first
    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.float32)
    sq[:h, :w] = arr
    # Resample to target
    idx_h = np.linspace(0, side - 1, target_size).astype(int)
    idx_w = np.linspace(0, side - 1, target_size).astype(int)
    return sq[idx_h[:, None], idx_w[None, :]].astype(np.float32)


def _normalize_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize per-channel pour clamp [0, 1] (préserve relative magnitude)."""
    lo = x.min()
    hi = x.max()
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + eps)


def render_arch_state(
    model: nn.Module,
    image_size: int = 64,
    include_gradients: bool = True,
) -> np.ndarray:
    """VG model state → (image_size, image_size, 3) RGB float [0, 1].

    Stratégie :
    - Concatène |weight| de TOUS Linear/Conv layers en 1 flat array
    - Reshape en grille → channel R
    - Idem |grad| → channel G (zeros si pas de grad)
    - Idem |bias| → channel B
    - Normalize per-channel

    Invariant à grow : output toujours (H, W, 3), magnitude relative préservée.
    """
    w_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight.detach().abs().cpu().numpy().flatten()
            w_parts.append(w)
            if include_gradients and module.weight.grad is not None:
                g = module.weight.grad.detach().abs().cpu().numpy().flatten()
                g_parts.append(g)
            else:
                g_parts.append(np.zeros_like(w))
            if module.bias is not None:
                b = module.bias.detach().abs().cpu().numpy().flatten()
                b_parts.append(b)

    if not w_parts:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)

    w_flat = np.concatenate(w_parts)
    g_flat = np.concatenate(g_parts)
    b_flat = np.concatenate(b_parts) if b_parts else np.zeros(1, dtype=np.float32)

    r = _to_square_grid(w_flat, image_size)
    g = _to_square_grid(g_flat, image_size)
    b = _to_square_grid(b_flat, image_size)

    r = _normalize_channel(r)
    g = _normalize_channel(g)
    b = _normalize_channel(b)

    return np.stack([r, g, b], axis=-1).astype(np.float32)


def render_arch_sequence(
    snapshots: list,
    image_size: int = 64,
) -> np.ndarray:
    """Series de snapshots (modèles ou états picklés) → (N, H, W, 3) séquence frames.

    Utile pour offline : pre-collecter trace mutations, render après, encode batch.
    """
    frames = np.zeros((len(snapshots), image_size, image_size, 3), dtype=np.float32)
    for i, snap in enumerate(snapshots):
        frames[i] = render_arch_state(snap, image_size=image_size)
    return frames
