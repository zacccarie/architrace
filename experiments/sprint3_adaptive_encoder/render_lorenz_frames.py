"""
Sprint 3 — génère séquence de frames depuis trajectoire Lorenz.

Lorenz attracteur projeté sur plan (x,y) → point lumineux animé sur fond noir.
Frames 64x64 RGB. Ground truth dynamique = Lorenz chaos.
"""
from __future__ import annotations
import numpy as np


def render_lorenz_frames(
    n_frames: int = 300, frame_size: int = 64, dt: float = 0.01,
    sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0,
    trail_length: int = 10, particle_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Génère séquence Lorenz : trajectoire 3D + frames rendered 2D projection.

    Returns:
        frames (N, H, W, 3) float [0,1]
        coords3d (N, 3) raw Lorenz xyz pour ground truth
    """
    # Intégrer Lorenz
    x, y, z = 0.1, 0.0, 0.0
    coords = np.zeros((n_frames, 3), dtype=np.float32)
    for i in range(n_frames):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt; y += dy * dt; z += dz * dt
        coords[i] = [x, y, z]

    # Normalise xy sur grille pixel
    xy = coords[:, :2]
    mn = xy.min(0)
    mx = xy.max(0)
    span = np.maximum(mx - mn, 1e-6)
    pad = 4
    grid = (xy - mn) / span * (frame_size - 2 * pad) + pad
    grid = grid.astype(int)
    z_normalized = (coords[:, 2] - coords[:, 2].min()) / max(coords[:, 2].max() - coords[:, 2].min(), 1e-6)

    # Render frames avec trail
    frames = np.zeros((n_frames, frame_size, frame_size, 3), dtype=np.float32)
    for i in range(n_frames):
        # Trail : particules plus anciennes plus fade
        for t_back in range(min(trail_length, i + 1)):
            idx = i - t_back
            cx, cy = grid[idx]
            intensity = (1.0 - t_back / trail_length) ** 2
            # Color : Z détermine couleur (cyan→magenta)
            z_v = z_normalized[idx]
            r = z_v
            g = 1.0 - 0.5 * z_v
            b = 1.0
            # Particule disque
            for dx_ in range(-particle_radius, particle_radius + 1):
                for dy_ in range(-particle_radius, particle_radius + 1):
                    if dx_*dx_ + dy_*dy_ <= particle_radius*particle_radius:
                        px = cx + dx_; py = cy + dy_
                        if 0 <= px < frame_size and 0 <= py < frame_size:
                            frames[i, py, px, 0] = max(frames[i, py, px, 0], intensity * r)
                            frames[i, py, px, 1] = max(frames[i, py, px, 1], intensity * g)
                            frames[i, py, px, 2] = max(frames[i, py, px, 2], intensity * b)

    return frames, coords


if __name__ == '__main__':
    frames, coords = render_lorenz_frames(n_frames=200, frame_size=64)
    print(f'frames: {frames.shape}  range=[{frames.min():.3f}, {frames.max():.3f}]')
    print(f'coords: {coords.shape}  Lorenz range x=[{coords[:,0].min():.2f}, {coords[:,0].max():.2f}]')

    # Save preview PNG si PIL dispo
    try:
        from PIL import Image
        # Sample 8 frames uniformly
        idxs = np.linspace(0, 199, 8).astype(int)
        strip = np.concatenate([frames[i] for i in idxs], axis=1)
        Image.fromarray((strip * 255).astype(np.uint8)).save('/tmp/lorenz_preview.png')
        print('preview → /tmp/lorenz_preview.png')
    except ImportError:
        pass
