"""
Sprint 3.1 — bench extended : 5 encoders × 3 systèmes × 5 seeds.

Encoders :
- vg_tiny  (498 params)
- vg_small (6362 params)
- vg_medium (24234 params)
- resnet50_pretrained (23.5M, ImageNet weights)
- resnet50_random (23.5M, random init = FAIR comparator)

Systèmes :
- Lorenz
- Rössler
- Double pendulum

5 seeds × 3 systems × 5 encoders = 75 configs. ~3 min total.

Output : tableau aggregate avec mean±std, comparaison fair.
"""
from __future__ import annotations
import argparse
import json
import sys as _sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import architrace  # noqa
from architrace.encoder_vg import VGCNNEncoder

_sys.path.insert(0, str(Path(__file__).parent))
from dynamical_systems import generate_system, SYSTEMS
from local_metrics import analyze


class ResNet50Wrapper:
    """ResNet50 features extractor — pretrained ou random-init.

    Frames 64x64 → upsample 224x224 → resnet50 → 2048-d features (avgpool).
    """
    def __init__(self, pretrained: bool = True, seed: int = 0):
        import torchvision.models as models
        torch.manual_seed(seed)
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            # Random init via seed setting
            self.model = models.resnet50(weights=None)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.latent_dim = 2048

    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> np.ndarray:
        """(N, 64, 64, 3) → (N, 2048)."""
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (N, 3, 64, 64)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Normalisation ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        feats = self.model(x)
        return feats.numpy()


def bench_one(encoder_name: str, encoder, frames: np.ndarray,
               coords_gt: np.ndarray, n_params: int) -> dict:
    t0 = time.time()
    latents = encoder.encode(frames)
    enc_time = time.time() - t0
    # Analyse les latents
    m = analyze(latents)
    # Aussi calculer ground truth pour reference
    return {
        'encoder': encoder_name,
        'n_params': n_params,
        'latent_dim': int(latents.shape[1]),
        'enc_time': enc_time,
        'lyap': m['lyapunov'],
        'corr_dim': m['corr_dim'],
        'spectral_slope': m['spectral_slope'],
    }


def run_seed(system: str, seed: int, n_frames: int) -> tuple[list[dict], dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    frames, coords = generate_system(system, n_frames=n_frames, seed=seed)
    gt = analyze(coords)

    rows = []

    # VG-CNN variants
    for tag, channels in [
        ('vg_tiny', (4, 8)),
        ('vg_small', (8, 16, 32)),
        ('vg_medium', (16, 32, 64)),
    ]:
        torch.manual_seed(seed)
        enc = VGCNNEncoder(channels=channels)
        n_params = enc.num_params()
        r = bench_one(tag, enc, frames, coords, n_params)
        rows.append(r)

    # ResNet50 pretrained (le test original Sprint 3)
    enc = ResNet50Wrapper(pretrained=True, seed=seed)
    r = bench_one('resnet50_pretrained', enc, frames, coords, enc.num_params())
    rows.append(r)

    # ResNet50 random-init (FAIR comparator)
    enc = ResNet50Wrapper(pretrained=False, seed=seed)
    r = bench_one('resnet50_random', enc, frames, coords, enc.num_params())
    rows.append(r)

    return rows, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', type=str, default='lorenz,rossler,double_pendulum')
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4')
    ap.add_argument('--n-frames', type=int, default=200)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint3_1')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    systems = [s.strip() for s in args.systems.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 3.1 — encoder bench extended ===')
    print(f'Systems: {systems}  Seeds: {seeds}  n_frames: {args.n_frames}\n')

    all_rows = []
    all_gt = {}
    t_total = time.time()
    for system in systems:
        print(f'\n--- {system} ---')
        all_gt[system] = []
        for seed in seeds:
            rows, gt = run_seed(system, seed, args.n_frames)
            for r in rows:
                r['system'] = system
                r['seed'] = seed
            all_rows.extend(rows)
            all_gt[system].append(gt)
            top = max(rows, key=lambda r: -abs(r['corr_dim'] - gt['corr_dim']))
            print(f'  seed={seed}  GT corr_dim={gt["corr_dim"]:.3f}  '
                  f'GT lyap={gt["lyapunov"]:.3f}')
            for r in rows:
                d_corr = abs(r['corr_dim'] - gt['corr_dim'])
                d_lyap = abs(r['lyap'] - gt['lyapunov'])
                print(f'    {r["encoder"]:25s}  '
                       f'corr_dim={r["corr_dim"]:.3f} (Δ {d_corr:.3f})  '
                       f'lyap={r["lyap"]:.3f} (Δ {d_lyap:.3f})')
    elapsed = time.time() - t_total
    print(f'\nTotal compute : {elapsed:.0f}s')

    # Aggregate
    md = ['# Sprint 3.1 — bench encoder extended (fair comparison)\n']
    md.append(f'Systems : {", ".join(systems)}\n')
    md.append(f'Seeds   : {len(seeds)} ({seeds[0]}-{seeds[-1]})\n')
    md.append(f'n_frames : {args.n_frames}\n')
    md.append('Compute total : {:.0f}s\n'.format(elapsed))

    md.append('\n## Aggregate (mean ± std) sur 5 seeds\n')
    md.append('| system | encoder | params | Δ corr_dim | Δ lyap | enc_time (s) |')
    md.append('|---|---|---|---|---|---|')

    for system in systems:
        gt_corr = np.mean([g['corr_dim'] for g in all_gt[system]])
        gt_lyap = np.mean([g['lyapunov'] for g in all_gt[system]])
        md.append(f'| **{system}** GT | — | — | corr_dim={gt_corr:.3f} | lyap={gt_lyap:.3f} | — |')
        for enc_name in ['vg_tiny', 'vg_small', 'vg_medium',
                          'resnet50_pretrained', 'resnet50_random']:
            rs = [r for r in all_rows if r['system'] == system and r['encoder'] == enc_name]
            if not rs:
                continue
            d_corr = np.array([abs(r['corr_dim'] - all_gt[system][i]['corr_dim'])
                               for i, r in enumerate(rs)])
            d_lyap = np.array([abs(r['lyap'] - all_gt[system][i]['lyapunov'])
                               for i, r in enumerate(rs)])
            enc_times = [r['enc_time'] for r in rs]
            md.append(
                f'| {system} | {enc_name} | {rs[0]["n_params"]:,} | '
                f'{d_corr.mean():.3f}±{d_corr.std():.3f} | '
                f'{d_lyap.mean():.3f}±{d_lyap.std():.3f} | '
                f'{np.mean(enc_times):.2f} |'
            )

    md.append('\n## Comparaison FAIR : ResNet50 pretrained vs random\n')
    md.append('| system | vg_tiny | resnet50_pretrained | resnet50_random | Verdict |')
    md.append('|---|---|---|---|---|')
    for system in systems:
        gt_corr = np.mean([g['corr_dim'] for g in all_gt[system]])
        vt = [r for r in all_rows if r['system'] == system and r['encoder'] == 'vg_tiny']
        rp = [r for r in all_rows if r['system'] == system and r['encoder'] == 'resnet50_pretrained']
        rr = [r for r in all_rows if r['system'] == system and r['encoder'] == 'resnet50_random']
        if not (vt and rp and rr):
            continue
        d_vt = np.mean([abs(r['corr_dim'] - all_gt[system][i]['corr_dim'])
                         for i, r in enumerate(vt)])
        d_rp = np.mean([abs(r['corr_dim'] - all_gt[system][i]['corr_dim'])
                         for i, r in enumerate(rp)])
        d_rr = np.mean([abs(r['corr_dim'] - all_gt[system][i]['corr_dim'])
                         for i, r in enumerate(rr)])
        # Verdict : qui est meilleur ?
        scores = [('vg_tiny', d_vt), ('resnet50_pre', d_rp), ('resnet50_rand', d_rr)]
        best = min(scores, key=lambda x: x[1])
        verdict = f'**{best[0]}** best (Δ={best[1]:.3f})'
        md.append(f'| {system} | {d_vt:.3f} | {d_rp:.3f} | {d_rr:.3f} | {verdict} |')

    md.append('\n## Lecture honnête\n')
    md.append('- Si **vg_tiny < resnet50_random** : random-init resnet50 = fair comparator')
    md.append('  battu par vg_tiny → claim "inductive bias > scale" tient.')
    md.append('- Si **vg_tiny ≈ resnet50_random** : la différence Sprint 3 original venait')
    md.append('  du PRETRAINING (ImageNet bias), pas du SIZE. Claim "tiny bat 25M" inexact.')
    md.append('- Si **vg_tiny > resnet50_random** (worse): vg_tiny était lucky single-seed,')
    md.append('  finding rétracté à étendre échantillon.')

    (args.out / 'bench_extended_results.json').write_text(
        json.dumps({'rows': all_rows, 'gt': all_gt, 'args': vars(args)},
                   default=str, indent=2)
    )
    (args.out / 'bench_extended_report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
