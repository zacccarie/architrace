"""
Sprint 1 — bench predictor VG vs LatentMLP fixe.

Sur systèmes synthétiques (Lorenz, Rössler, Van der Pol, Hénon) — ground truth.
Mesures :
- Predictability ratio (1 - mean_err / baseline)
- Final rollout error (n-step horizon=5)
- Params utilisés (VG peut être bien plus petit)
- Step errors curve (Lyapunov-like)
- Wallclock

Question : à params égaux, VG bat-il MLP fixe ? À performance égale, est-il plus petit ?
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
# Path setup via architrace __init__
import architrace  # noqa
from architrace.predictor_vg import fit_vg_predictor, VGPredictorConfig

from backend.systems import lorenz, rossler, van_der_pol, henon, logistic_map
import torch.nn as nn


class UncappedMLP(nn.Module):
    """LatentMLP sans cap h=min(hidden,dim). Pour comparaison équitable low-dim.

    z[t+1] = z[t] + Linear(d,h) → GELU → Linear(h,d).
    """
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.net(x)


def fit_uncapped_mlp(latents: np.ndarray, epochs: int = 300, lr: float = 1e-3,
                    hidden: int = 128, device: str = 'cpu') -> dict:
    """Drop-in equivalent fit_predictor but with uncapped hidden."""
    n, d = latents.shape
    x = torch.from_numpy(latents[:-1]).float().to(device)
    y = torch.from_numpy(latents[1:]).float().to(device)
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True).clamp_min(1e-6)
    x_n = (x - mean) / std
    y_n = (y - mean) / std

    model = UncappedMLP(d, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_n)
        loss = ((pred - y_n) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        pred_n = model(x_n)
        pred = pred_n * std + mean
        err = ((pred - y) ** 2).sum(dim=1).sqrt().cpu().numpy()
    baseline_err = np.linalg.norm(latents[1:] - latents[:-1], axis=1)
    predictability = float(1.0 - min(1.0, err.mean() / max(baseline_err.mean(), 1e-9)))
    n_params = sum(p.numel() for p in model.parameters())
    return {
        'predictability': predictability,
        'mean_error': float(err.mean()),
        'final_loss': float(losses[-1]),
        'n_params': n_params,
    }


fit_mlp_predictor = fit_uncapped_mlp


SYSTEMS = {
    'lorenz': lambda n: lorenz(n=n, dt=0.01),
    'rossler': lambda n: rossler(n=n, dt=0.05),
    'van_der_pol': lambda n: van_der_pol(n=n, dt=0.05),
    'henon': lambda n: henon(n=n),
    'logistic': lambda n: logistic_map(n=n),
}


def bench_one(system_name: str, n_points: int, seed: int) -> dict:
    """Compare VG predictor vs LatentMLP fixe sur 1 système, 1 seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    sys_data = SYSTEMS[system_name](n_points)
    latents = np.array(sys_data['raw_coords'], dtype=np.float32)
    d = latents.shape[1]

    # === Baseline MLP fixe ===
    # 3 tailles : small, medium, large
    mlp_variants = [
        ('mlp_small', 32),
        ('mlp_medium', 128),
        ('mlp_large', 256),
    ]
    mlp_results = {}
    for tag, hidden in mlp_variants:
        torch.manual_seed(seed)
        t0 = time.time()
        r = fit_mlp_predictor(latents, epochs=300, hidden=hidden, lr=1e-3)
        elapsed = time.time() - t0
        mlp_results[tag] = {
            'hidden': hidden,
            'params': r['n_params'],
            'predictability': r['predictability'],
            'mean_error': r['mean_error'],
            'final_loss': r['final_loss'],
            'elapsed': elapsed,
        }

    # === VG predictor ===
    torch.manual_seed(seed)
    cfg = VGPredictorConfig(
        epochs=300,
        init_hidden=8, init_depth=2, max_hidden=128, grow_k=8,
        plateau_window=25, plateau_eps=1e-3, cooldown=40, verify_extra=10,
        rollout_horizon=5, distill_steps=15,
    )
    t0 = time.time()
    r = fit_vg_predictor(latents, cfg=cfg)
    elapsed = time.time() - t0
    vg_result = {
        'final_topology': r['final_topology'],
        'final_params': r['final_params'],
        'predictability': r['predictability'],
        'mean_error': r['mean_error'],
        'final_rollout_err': r['final_rollout_err'],
        'accepted': r['accepted_mutations'],
        'rejected': r['rejected_mutations'],
        'frozen': r['frozen'],
        'elapsed': elapsed,
    }

    return {
        'system': system_name,
        'seed': seed,
        'n_points': n_points,
        'dim': d,
        'mlp_variants': mlp_results,
        'vg': vg_result,
    }


def aggregate(rows: list[dict]) -> dict:
    """Aggregate by system."""
    by_system: dict[str, list] = {}
    for r in rows:
        by_system.setdefault(r['system'], []).append(r)

    summary = {}
    for sys_name, runs in by_system.items():
        pred_mlp_s = [r['mlp_variants']['mlp_small']['predictability'] for r in runs]
        pred_mlp_m = [r['mlp_variants']['mlp_medium']['predictability'] for r in runs]
        pred_mlp_l = [r['mlp_variants']['mlp_large']['predictability'] for r in runs]
        pred_vg = [r['vg']['predictability'] for r in runs]
        params_vg = [r['vg']['final_params'] for r in runs]
        params_mlp_m = [r['mlp_variants']['mlp_medium']['params'] for r in runs]

        def stat(xs):
            xs = [x for x in xs if x is not None]
            if not xs:
                return None
            return {'mean': float(np.mean(xs)), 'std': float(np.std(xs)),
                    'min': float(min(xs)), 'max': float(max(xs))}

        summary[sys_name] = {
            'n_seeds': len(runs),
            'mlp_small_pred': stat(pred_mlp_s),
            'mlp_medium_pred': stat(pred_mlp_m),
            'mlp_large_pred': stat(pred_mlp_l),
            'vg_pred': stat(pred_vg),
            'vg_params': stat(params_vg),
            'mlp_medium_params': stat(params_mlp_m),
            'vg_acc_mutations': stat([r['vg']['accepted'] for r in runs]),
        }
    return summary


def render_md(rows: list[dict], summary: dict) -> str:
    """Markdown table : VG vs MLP par système."""
    md = []
    md.append('# Sprint 1 — VG predictor vs LatentMLP fixe\n')
    md.append('## Per-system summary (mean ± std sur seeds)\n')
    md.append('| system | n_seeds | MLP-s pred | MLP-m pred | MLP-l pred | VG pred | VG params | MLP-m params | VG mutations |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for sys_name, s in summary.items():
        def fmt(stat_dict, key='mean', dec=3):
            if not stat_dict:
                return '—'
            v = stat_dict.get('mean', 0)
            std = stat_dict.get('std', 0)
            return f'{v:.{dec}f}±{std:.{dec}f}'

        md.append(
            f'| {sys_name} | {s["n_seeds"]} | '
            f'{fmt(s["mlp_small_pred"])} | '
            f'{fmt(s["mlp_medium_pred"])} | '
            f'{fmt(s["mlp_large_pred"])} | '
            f'{fmt(s["vg_pred"])} | '
            f'{fmt(s["vg_params"], dec=0)} | '
            f'{fmt(s["mlp_medium_params"], dec=0)} | '
            f'{fmt(s["vg_acc_mutations"], dec=1)} |'
        )

    md.append('\n## Per-run detail\n')
    md.append('| system | seed | MLP-s pred | MLP-m pred | MLP-l pred | VG pred | VG params | VG topo |')
    md.append('|---|---|---|---|---|---|---|---|')
    for r in rows:
        m = r['mlp_variants']
        v = r['vg']
        md.append(
            f'| {r["system"]} | {r["seed"]} | '
            f'{m["mlp_small"]["predictability"]:.3f} ({m["mlp_small"]["params"]}p) | '
            f'{m["mlp_medium"]["predictability"]:.3f} ({m["mlp_medium"]["params"]}p) | '
            f'{m["mlp_large"]["predictability"]:.3f} ({m["mlp_large"]["params"]}p) | '
            f'{v["predictability"]:.3f} | '
            f'{v["final_params"]} | '
            f'{v["final_topology"]} |'
        )

    # Verdict
    md.append('\n## Verdict per system\n')
    for sys_name, s in summary.items():
        if not s['vg_pred'] or not s['mlp_medium_pred']:
            continue
        vg = s['vg_pred']['mean']
        mlp_s = s['mlp_small_pred']['mean']
        mlp_m = s['mlp_medium_pred']['mean']
        mlp_l = s['mlp_large_pred']['mean']
        vg_p = s['vg_params']['mean']
        mlp_p = s['mlp_medium_params']['mean']

        winner = 'VG' if vg > mlp_m else 'MLP-medium'
        ratio = vg_p / mlp_p if mlp_p > 0 else float('inf')
        md.append(
            f'- **{sys_name}** : pred VG={vg:.3f} vs MLP-m={mlp_m:.3f} '
            f'→ **{winner}** wins. VG uses {ratio:.2f}× params of MLP-medium. '
            f'(MLP-s={mlp_s:.3f}, MLP-l={mlp_l:.3f})'
        )

    return '\n'.join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', type=str, default='lorenz,rossler,van_der_pol')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint1')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    systems = [s.strip() for s in args.systems.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 1 bench ===')
    print(f'Systems: {systems}  Seeds: {seeds}  N points: {args.n}')

    rows = []
    for sys_name in systems:
        for seed in seeds:
            print(f'\n[{sys_name} seed={seed}] ...')
            t0 = time.time()
            r = bench_one(sys_name, args.n, seed)
            elapsed = time.time() - t0
            rows.append(r)
            v = r['vg']
            mm = r['mlp_variants']['mlp_medium']
            print(f'  MLP-m: pred={mm["predictability"]:.3f}  '
                  f'params={mm["params"]}  ({mm["elapsed"]:.1f}s)')
            print(f'  VG:    pred={v["predictability"]:.3f}  '
                  f'params={v["final_params"]}  acc={v["accepted"]}  '
                  f'topo={v["final_topology"]}  ({v["elapsed"]:.1f}s)')

    summary = aggregate(rows)
    md = render_md(rows, summary)

    (args.out / 'bench_results.json').write_text(
        json.dumps({'rows': rows, 'summary': summary}, default=float, indent=2)
    )
    (args.out / 'bench_report.md').write_text(md)

    print('\n' + '=' * 70)
    print(md)
    print(f'\nResults → {args.out}')


if __name__ == '__main__':
    main()
