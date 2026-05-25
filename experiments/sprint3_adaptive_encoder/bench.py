"""
Sprint 3 — bench encoder VG-CNN vs ResNet50 sur séquence Lorenz rendered.

Question : VG-CNN à ~6k params extrait-il des features assez riches pour que
dynamoscope retrouve les mêmes lois dynamiques (Lyapunov ≈ 0.9, corr_dim ≈ 2)
que ResNet50 (25M params) ?

Procédure :
1. Génère séquence Lorenz rendered en frames 64×64
2. Encode avec VG-CNN random-init (~6k params)
3. Encode avec ResNet50 (25M params)
4. Inject latents dans dynamoscope, calcule analyses (dynamics, sindy, emergence)
5. Compare métriques + corrélation des trajectoires
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import architrace
from architrace.encoder_vg import VGCNNEncoder
from architrace.bridge import DynamoscopeClient

from render_lorenz_frames import render_lorenz_frames

# Dynamoscope encoder direct
import sys as _sys
_sys.path.insert(0, str(architrace.DYNAMOSCOPE_PATH))
from backend.encoder import ResNet50Encoder


def bench_encoder(name: str, encoder, frames: np.ndarray, n_params: int,
                  client: DynamoscopeClient, seed: int) -> dict:
    """Encode + inject + analyses pour un encoder donné."""
    t0 = time.time()
    latents = encoder.encode(frames)
    enc_time = time.time() - t0

    cache_key = f'sprint3_{name}_s{seed}'
    inj = client.inject_latents(
        cache_key=cache_key,
        latents=latents.tolist(),
        var_names=[f'd{i}' for i in range(latents.shape[1])],
        meta={'encoder': name, 'n_params': n_params},
        name=f'Lorenz-{name}',
    )

    analyses = client.all_analyses(cache_key, inj.coords_3d)

    return {
        'encoder': name,
        'n_params': n_params,
        'enc_time': enc_time,
        'latent_dim': int(latents.shape[1]),
        'n_frames': int(latents.shape[0]),
        'cache_key': cache_key,
        'coords_3d_sample_first': inj.coords_3d[0],
        'coords_3d_sample_last': inj.coords_3d[-1],
        'analyses': {
            k: {kk: vv for kk, vv in v.items()
                if not isinstance(vv, list) or len(str(vv)) < 200}
            for k, v in analyses.items() if isinstance(v, dict)
        },
        # Key metrics extracted
        'lyap_native': analyses.get('dynamics_native', {}).get('lyapunov'),
        'corr_dim_native': analyses.get('dynamics_native', {}).get('correlation_dim'),
        'rqa_DET_native': analyses.get('dynamics_native', {}).get('rqa', {}).get('DET'),
        'sindy_r2': analyses.get('sindy', {}).get('r2'),
        'phi_id': analyses.get('emergence', {}).get('phi_id_approx'),
        'spectral_slope': analyses.get('spectral', {}).get('spectral_slope'),
        'dmd_rank': analyses.get('spectral', {}).get('dmd', {}).get('rank'),
    }


def run_one(seed: int, n_frames: int, client) -> list[dict]:
    """Bench les 2 encoders sur même séquence Lorenz."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    frames, lorenz_coords = render_lorenz_frames(n_frames=n_frames, frame_size=64)

    rows = []

    # === VG-CNN random-init (3 sizes) ===
    for tag, channels in [
        ('vg_tiny', (4, 8)),
        ('vg_small', (8, 16, 32)),
        ('vg_medium', (16, 32, 64)),
    ]:
        torch.manual_seed(seed)
        enc = VGCNNEncoder(channels=channels)
        n_params = enc.num_params()
        print(f'  [{tag}] params={n_params}  latent_dim={enc.latent_dim}')
        r = bench_encoder(tag, enc, frames, n_params, client, seed)
        rows.append(r)
        print(f'    enc_time={r["enc_time"]:.2f}s  lyap={r["lyap_native"]:.3f}  '
              f'corr_dim={r["corr_dim_native"]:.3f}  sindy_r2={r["sindy_r2"]:.3f}  '
              f'phi_id={r["phi_id"]:.3f}')

    # === ResNet50 baseline ===
    enc = ResNet50Encoder()
    n_params = sum(p.numel() for p in enc.model.parameters())
    print(f'  [resnet50] params={n_params}  latent_dim={enc.latent_dim}')
    r = bench_encoder('resnet50', enc, frames, n_params, client, seed)
    rows.append(r)
    print(f'    enc_time={r["enc_time"]:.2f}s  lyap={r["lyap_native"]:.3f}  '
          f'corr_dim={r["corr_dim_native"]:.3f}  sindy_r2={r["sindy_r2"]:.3f}  '
          f'phi_id={r["phi_id"]:.3f}')

    # Add ground truth via raw Lorenz coords (no encoder)
    # cache_key for raw coords (3D, will be passed-through)
    raw_cache = f'sprint3_lorenz_raw_s{seed}'
    raw_inj = client.inject_latents(
        cache_key=raw_cache,
        latents=lorenz_coords.tolist(),
        var_names=['x', 'y', 'z'],
        meta={'encoder': 'raw_lorenz', 'n_params': 0},
        name='Lorenz-raw',
    )
    raw_analyses = client.all_analyses(raw_cache, raw_inj.coords_3d)
    rows.append({
        'encoder': 'raw_lorenz',
        'n_params': 0,
        'enc_time': 0.0,
        'latent_dim': 3,
        'n_frames': n_frames,
        'cache_key': raw_cache,
        'lyap_native': raw_analyses.get('dynamics_native', {}).get('lyapunov'),
        'corr_dim_native': raw_analyses.get('dynamics_native', {}).get('correlation_dim'),
        'rqa_DET_native': raw_analyses.get('dynamics_native', {}).get('rqa', {}).get('DET'),
        'sindy_r2': raw_analyses.get('sindy', {}).get('r2'),
        'phi_id': raw_analyses.get('emergence', {}).get('phi_id_approx'),
        'spectral_slope': raw_analyses.get('spectral', {}).get('spectral_slope'),
        'dmd_rank': raw_analyses.get('spectral', {}).get('dmd', {}).get('rank'),
    })
    print(f'  [raw_lorenz] ground truth')
    print(f'    lyap={rows[-1]["lyap_native"]:.3f}  '
          f'corr_dim={rows[-1]["corr_dim_native"]:.3f}  '
          f'sindy_r2={rows[-1]["sindy_r2"]:.3f}  '
          f'phi_id={rows[-1]["phi_id"]:.3f}')

    return rows


def render_md(all_rows: list[dict]) -> str:
    md = ['# Sprint 3 — VG-CNN vs ResNet50 encoder bench\n']
    md.append('Trajectoire Lorenz rendered en frames 64×64 RGB, encodée → dynamoscope analyses.\n')

    cols = ['encoder', 'seed', 'params', 'latent_dim', 'enc_time',
            'lyap', 'corr_dim', 'RQA_DET', 'sindy_r2', 'phi_id', 'spectral_slope']
    md.append('| ' + ' | '.join(cols) + ' |')
    md.append('|' + '|'.join(['---'] * len(cols)) + '|')

    for r in all_rows:
        md.append(
            f'| {r["encoder"]} | {r.get("seed", "—")} | '
            f'{r["n_params"]:,} | {r["latent_dim"]} | '
            f'{r["enc_time"]:.2f}s | '
            f'{r["lyap_native"]:.3f} | '
            f'{r["corr_dim_native"]:.3f} | '
            f'{(r.get("rqa_DET_native") or 0):.3f} | '
            f'{(r.get("sindy_r2") or 0):.3f} | '
            f'{(r.get("phi_id") or 0):.3f} | '
            f'{(r.get("spectral_slope") or 0):.3f} |'
        )

    md.append('\n## Verdict\n')

    # Group by encoder
    by_enc = {}
    for r in all_rows:
        by_enc.setdefault(r['encoder'], []).append(r)

    raw = by_enc.get('raw_lorenz', [])
    if raw:
        gt = raw[0]
        md.append(f'- **Ground truth (raw 3D Lorenz)** : Lyap={gt["lyap_native"]:.3f}  '
                  f'corr_dim={gt["corr_dim_native"]:.3f}  '
                  f'sindy_r2={gt["sindy_r2"]:.3f}  '
                  f'spectral_slope={(gt.get("spectral_slope") or 0):.3f}')

        for enc_name in ['vg_tiny', 'vg_small', 'vg_medium', 'resnet50']:
            runs = by_enc.get(enc_name, [])
            if not runs:
                continue
            r = runs[0]
            d_lyap = abs((r["lyap_native"] or 0) - gt["lyap_native"])
            d_corr = abs((r["corr_dim_native"] or 0) - gt["corr_dim_native"])
            md.append(f'- **{enc_name}** ({r["n_params"]:,} params) — Lyap Δ={d_lyap:.3f}, '
                      f'corr_dim Δ={d_corr:.3f}, sindy_r2={(r.get("sindy_r2") or 0):.3f}')

    return '\n'.join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='0,1')
    ap.add_argument('--n-frames', type=int, default=200)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint3')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 3 encoder bench ===')
    print(f'Seeds: {seeds}  n_frames: {args.n_frames}')

    client = DynamoscopeClient()
    print(f'Dynamoscope: {client.health()}')

    all_rows = []
    for seed in seeds:
        print(f'\n[seed {seed}]')
        rows = run_one(seed, args.n_frames, client)
        for r in rows:
            r['seed'] = seed
        all_rows.extend(rows)

    md = render_md(all_rows)
    (args.out / 'bench_results.json').write_text(json.dumps({'rows': all_rows}, default=float, indent=2))
    (args.out / 'bench_report.md').write_text(md)

    print('\n' + '=' * 70)
    print(md)
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
