"""
Sprint 3.1 — 3 systèmes dynamiques pour bench encoder extended.

- Lorenz attractor (3D chaotic)
- Rössler attractor (3D chaotic, plus simple)
- Double pendulum (4D, chaotique au-dessus seuil énergie)
"""
from __future__ import annotations
import numpy as np


def integrate_lorenz(n=2000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3,
                      x0=0.1, y0=0.0, z0=0.0):
    """Lorenz : 3D coords, chaotic."""
    coords = np.zeros((n, 3), dtype=np.float32)
    x, y, z = x0, y0, z0
    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt; y += dy * dt; z += dz * dt
        coords[i] = [x, y, z]
    return coords


def integrate_rossler(n=2000, dt=0.02, a=0.2, b=0.2, c=5.7,
                       x0=1.0, y0=1.0, z0=1.0):
    """Rössler : 3D coords, chaotic (simpler attractor than Lorenz)."""
    coords = np.zeros((n, 3), dtype=np.float32)
    x, y, z = x0, y0, z0
    for i in range(n):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt; y += dy * dt; z += dz * dt
        coords[i] = [x, y, z]
    return coords


def integrate_double_pendulum(n=2000, dt=0.01, g=9.81,
                                L1=1.0, L2=1.0, m1=1.0, m2=1.0,
                                th1_0=1.5, th2_0=2.0):
    """Double pendulum : 4D state (θ1, ω1, θ2, ω2). Chaotic.

    On exporte 3D : (θ1, θ2, ω1) car projection 2D-render utilise (θ1, θ2).
    """
    th1, th2 = th1_0, th2_0
    w1, w2 = 0.0, 0.0
    coords = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        # Equations of motion (Lagrangian dynamics)
        delta = th2 - th1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
        dw1 = (m2 * L1 * w1 ** 2 * np.sin(delta) * np.cos(delta)
               + m2 * g * np.sin(th2) * np.cos(delta)
               + m2 * L2 * w2 ** 2 * np.sin(delta)
               - (m1 + m2) * g * np.sin(th1)) / den1
        den2 = (L2 / L1) * den1
        dw2 = (-m2 * L2 * w2 ** 2 * np.sin(delta) * np.cos(delta)
               + (m1 + m2) * g * np.sin(th1) * np.cos(delta)
               - (m1 + m2) * L1 * w1 ** 2 * np.sin(delta)
               - (m1 + m2) * g * np.sin(th2)) / den2
        w1 += dw1 * dt; w2 += dw2 * dt
        th1 += w1 * dt; th2 += w2 * dt
        coords[i] = [th1, th2, w1]
    return coords


SYSTEMS = {
    'lorenz': integrate_lorenz,
    'rossler': integrate_rossler,
    'double_pendulum': integrate_double_pendulum,
}


def render_trajectory(coords: np.ndarray, frame_size: int = 64,
                       trail_length: int = 10, particle_radius: int = 2):
    """Trajectoire 3D → frames 64x64 RGB. Projection (col0, col1) sur grille.

    Coords[:, 2] détermine couleur (cyan→magenta).
    """
    n = coords.shape[0]
    xy = coords[:, :2]
    mn = xy.min(0); mx = xy.max(0)
    span = np.maximum(mx - mn, 1e-6)
    pad = 4
    grid = (xy - mn) / span * (frame_size - 2 * pad) + pad
    grid = grid.astype(int)
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    z_norm = (coords[:, 2] - z_min) / max(z_max - z_min, 1e-6)

    frames = np.zeros((n, frame_size, frame_size, 3), dtype=np.float32)
    for i in range(n):
        for t_back in range(min(trail_length, i + 1)):
            idx = i - t_back
            cx, cy = grid[idx]
            intensity = (1.0 - t_back / trail_length) ** 2
            z_v = z_norm[idx]
            r, g, b = z_v, 1.0 - 0.5 * z_v, 1.0
            for dx_ in range(-particle_radius, particle_radius + 1):
                for dy_ in range(-particle_radius, particle_radius + 1):
                    if dx_ * dx_ + dy_ * dy_ <= particle_radius ** 2:
                        px, py = cx + dx_, cy + dy_
                        if 0 <= px < frame_size and 0 <= py < frame_size:
                            frames[i, py, px, 0] = max(frames[i, py, px, 0], intensity * r)
                            frames[i, py, px, 1] = max(frames[i, py, px, 1], intensity * g)
                            frames[i, py, px, 2] = max(frames[i, py, px, 2], intensity * b)
    return frames


def generate_system(name: str, n_frames: int = 200, seed: int = 0):
    """Génère coords + frames pour un système. Seed change conditions initiales."""
    rng = np.random.RandomState(seed)
    if name == 'lorenz':
        x0 = 0.1 + rng.uniform(-0.05, 0.05)
        y0 = rng.uniform(-0.05, 0.05)
        coords = integrate_lorenz(n=n_frames, x0=x0, y0=y0)
    elif name == 'rossler':
        x0 = 1.0 + rng.uniform(-0.1, 0.1)
        y0 = 1.0 + rng.uniform(-0.1, 0.1)
        coords = integrate_rossler(n=n_frames, x0=x0, y0=y0)
    elif name == 'double_pendulum':
        th1 = 1.5 + rng.uniform(-0.1, 0.1)
        th2 = 2.0 + rng.uniform(-0.1, 0.1)
        coords = integrate_double_pendulum(n=n_frames, th1_0=th1, th2_0=th2)
    else:
        raise ValueError(name)
    frames = render_trajectory(coords, frame_size=64)
    return frames, coords


if __name__ == '__main__':
    for name in SYSTEMS:
        frames, coords = generate_system(name, n_frames=200, seed=0)
        print(f'{name:20s} : frames {frames.shape}  coords range '
              f'[{coords.min():.2f}, {coords.max():.2f}]')
