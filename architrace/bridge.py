"""
architrace.bridge — client HTTP propre pour dynamoscope.

Remplace les helpers urllib éparpillés dans Sprint 4 scripts.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from . import DYNAMOSCOPE_URL


class DynamoscopeError(Exception):
    """Erreur côté dynamoscope (HTTP non-2xx ou network)."""


@dataclass
class InjectResult:
    cache_key: str
    n: int
    latent_dim: int
    coords_3d: list[list[float]]
    perceptual_velocity: list[float]
    euclidean_velocity: list[float]
    reducer: str
    raw: dict


class DynamoscopeClient:
    """Client HTTP pour dynamoscope. Default URL via env / config.

    Usage :
        client = DynamoscopeClient()
        result = client.inject_latents(cache_key='vg_run_0', latents=trace, ...)
        dyn = client.dynamics_native(cache_key)
        sindy = client.sindy(coords_3d, order=2, threshold=0.05)
    """

    def __init__(self, base_url: str = DYNAMOSCOPE_URL, timeout: float = 180.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    # ---------- low-level ----------
    def _post(self, path: str, body: Optional[dict] = None) -> dict:
        url = f'{self.base_url}{path}'
        data = json.dumps(body).encode() if body is not None else b''
        headers = {'Content-Type': 'application/json'} if body is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='replace')[:500]
            raise DynamoscopeError(f'HTTP {e.code} {path}: {msg}') from e

    def _get(self, path: str) -> dict:
        url = f'{self.base_url}{path}'
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='replace')[:500]
            raise DynamoscopeError(f'HTTP {e.code} {path}: {msg}') from e

    # ---------- health ----------
    def health(self) -> dict:
        return self._get('/api/health')

    # ---------- inject (architrace-specific) ----------
    def inject_latents(
        self,
        cache_key: str,
        latents: list[list[float]],
        var_names: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        steps: Optional[list[int]] = None,
        reducer: str = 'umap',
        meta: Optional[dict] = None,
        name: Optional[str] = None,
    ) -> InjectResult:
        body: dict[str, Any] = {
            'cache_key': cache_key,
            'latents': latents,
            'reducer': reducer,
        }
        if var_names is not None:
            body['var_names'] = var_names
        if tags is not None:
            body['tags'] = tags
        if steps is not None:
            body['steps'] = steps
        if meta is not None:
            body['meta'] = meta
        if name is not None:
            body['name'] = name
        r = self._post('/api/inject_latents', body)
        return InjectResult(
            cache_key=r['cache_key'],
            n=r['n'],
            latent_dim=r['latent_dim'],
            coords_3d=r['coords'],
            perceptual_velocity=r.get('perceptual_velocity', []),
            euclidean_velocity=r.get('euclidean_velocity', []),
            reducer=r.get('reducer', reducer),
            raw=r,
        )

    def inject_meta(self, cache_key: str) -> dict:
        return self._get(f'/api/inject_meta/{cache_key}')

    # ---------- analyses native (haute-dim) ----------
    def dynamics_native(self, cache_key: str) -> dict:
        return self._post(f'/api/dynamics_native/{cache_key}')

    def topology_native(self, cache_key: str, max_dim: int = 1) -> dict:
        return self._post(f'/api/topology_native/{cache_key}?max_dim={max_dim}')

    # ---------- analyses sur coords 3D ----------
    def dynamics(self, coords: list[list[float]]) -> dict:
        return self._post('/api/dynamics', {'coords': coords})

    def sindy(
        self, coords: list[list[float]],
        dt: float = 1.0, order: int = 2, threshold: float = 0.05,
        var_names: Optional[list[str]] = None,
    ) -> dict:
        body = {'coords': coords, 'dt': dt, 'order': order, 'threshold': threshold}
        if var_names:
            body['var_names'] = var_names
        return self._post('/api/sindy', body)

    def emergence(self, coords: list[list[float]]) -> dict:
        return self._post('/api/emergence', {'coords': coords})

    def spectral(self, coords: list[list[float]]) -> dict:
        return self._post('/api/spectral', {'coords': coords})

    def causal(self, coords: list[list[float]]) -> dict:
        return self._post('/api/causal', {'coords': coords})

    # ---------- multi-trace ----------
    def compare(
        self, key1: str, key2: str, reducer: str = 'umap',
        n_neighbors: int = 15, min_dist: float = 0.1, metric: str = 'cosine',
    ) -> dict:
        path = (f'/api/compare?key1={key1}&key2={key2}&reducer={reducer}'
                f'&n_neighbors={n_neighbors}&min_dist={min_dist}&metric={metric}')
        return self._get(path)

    # ---------- batch helper ----------
    def all_analyses(self, cache_key: str, coords_3d: list[list[float]]) -> dict:
        """Lance toutes analyses standard sur un cache. Erreurs trappées dans dict result."""
        out: dict[str, Any] = {}
        for name, fn in [
            ('dynamics_native', lambda: self.dynamics_native(cache_key)),
            ('topology_native', lambda: self.topology_native(cache_key)),
            ('dynamics_3d', lambda: self.dynamics(coords_3d)),
            ('sindy', lambda: self.sindy(coords_3d, var_names=['u1', 'u2', 'u3'])),
            ('emergence', lambda: self.emergence(coords_3d)),
            ('spectral', lambda: self.spectral(coords_3d)),
            ('causal', lambda: self.causal(coords_3d)),
        ]:
            try:
                out[name] = fn()
            except DynamoscopeError as e:
                out[name] = {'error': str(e)}
        return out
