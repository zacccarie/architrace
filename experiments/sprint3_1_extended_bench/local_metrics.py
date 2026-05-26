"""
Sprint 3.1 — métriques dynamiques locales (sans Dynamoscope HTTP).

- Lyapunov : finite-difference estimateur depuis trajectoire latente
- Correlation dim : Grassberger-Procaccia
- Spectral slope : FFT + linear fit log-log power spectrum
"""
from __future__ import annotations
import numpy as np


def lyapunov_estimator(traj: np.ndarray, dt: float = 1.0,
                        min_steps: int = 5, max_steps: int = 50) -> float:
    """Estimateur Lyapunov via divergence de trajectoires voisines.

    Pour chaque point, trouve son plus proche voisin, suit comment leur
    distance évolue. Pente moyenne log-distance vs t = exposant Lyapunov.
    """
    n = traj.shape[0]
    if n < max_steps + 10:
        return 0.0
    # Trouver plus proche voisin pour chaque point (excluant temporal neighbors)
    log_divergences = []
    n_samples = min(50, n - max_steps - 1)
    sample_idx = np.linspace(0, n - max_steps - 2, n_samples).astype(int)
    for i in sample_idx:
        # Distances aux autres points (skip ±5 temporal neighbors)
        skip = 5
        candidates = np.concatenate([np.arange(0, max(0, i - skip)),
                                      np.arange(min(n, i + skip + 1), n - max_steps)])
        if len(candidates) == 0:
            continue
        d = np.linalg.norm(traj[candidates] - traj[i], axis=1)
        if len(d) == 0:
            continue
        j = candidates[np.argmin(d)]
        d0 = np.linalg.norm(traj[j] - traj[i])
        if d0 < 1e-9:
            continue
        # Suivre la divergence pendant max_steps
        log_d = []
        for k in range(min_steps, max_steps):
            if i + k >= n or j + k >= n:
                break
            dk = np.linalg.norm(traj[i + k] - traj[j + k])
            if dk < 1e-12:
                continue
            log_d.append(np.log(dk / d0))
        if len(log_d) >= 5:
            # Slope = Lyapunov
            x = np.arange(min_steps, min_steps + len(log_d)) * dt
            slope = np.polyfit(x, log_d, 1)[0]
            log_divergences.append(slope)
    if not log_divergences:
        return 0.0
    return float(np.mean(log_divergences))


def correlation_dim(traj: np.ndarray, n_pairs: int = 500) -> float:
    """Estimateur Grassberger-Procaccia dimension de corrélation.

    C(r) ~ r^D pour petits r. Fit log-log donne D.
    """
    n = traj.shape[0]
    if n < 50:
        return 0.0
    rng = np.random.RandomState(42)
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    idx_pairs = []
    for _ in range(n_pairs):
        i, j = rng.randint(n), rng.randint(n)
        if i != j:
            idx_pairs.append((i, j))
    if not idx_pairs:
        return 0.0
    dists = np.array([np.linalg.norm(traj[i] - traj[j]) for i, j in idx_pairs])
    dists = dists[dists > 0]
    if len(dists) < 20:
        return 0.0
    # Choisir 8 valeurs r entre 5% et 50% du max
    rs = np.quantile(dists, np.linspace(0.05, 0.5, 8))
    log_r = []
    log_c = []
    for r in rs:
        c = float(np.sum(dists < r)) / len(dists)
        if c > 0:
            log_r.append(np.log(r))
            log_c.append(np.log(c))
    if len(log_r) < 4:
        return 0.0
    slope, _ = np.polyfit(log_r, log_c, 1)
    return float(slope)


def spectral_slope(traj: np.ndarray) -> float:
    """Pente log-log densité spectrale moyenne (sur toutes les dims).

    1/f² (slope ≈ -2) = bruit Brownian, signature dynamique chaotic typique.
    """
    n, d = traj.shape
    if n < 32:
        return 0.0
    slopes = []
    for dim in range(d):
        sig = traj[:, dim] - traj[:, dim].mean()
        # Welch-ish périodogramme simple
        spectrum = np.abs(np.fft.rfft(sig)) ** 2
        freqs = np.fft.rfftfreq(n)
        # Skip DC + Nyquist edge
        mask = (freqs > 1e-3) & (freqs < 0.4) & (spectrum > 0)
        if mask.sum() < 8:
            continue
        log_f = np.log(freqs[mask])
        log_p = np.log(spectrum[mask])
        slope, _ = np.polyfit(log_f, log_p, 1)
        slopes.append(slope)
    if not slopes:
        return 0.0
    return float(np.mean(slopes))


def analyze(traj: np.ndarray) -> dict:
    """Toutes métriques locales."""
    return {
        'lyapunov': lyapunov_estimator(traj),
        'corr_dim': correlation_dim(traj),
        'spectral_slope': spectral_slope(traj),
    }


if __name__ == '__main__':
    # Test sur Lorenz
    from dynamical_systems import generate_system
    frames, coords = generate_system('lorenz', n_frames=500, seed=0)
    m = analyze(coords)
    print('Lorenz ground truth :', m)
    # Test sur Gaussian noise
    rng = np.random.RandomState(0)
    noise = rng.randn(500, 3).astype(np.float32)
    m2 = analyze(noise)
    print('Gaussian noise     :', m2)
