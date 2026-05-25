"""
Sprint 5C.1 — sweep self_surprise_weight α.

Hypothèse :
    α=0 (baseline) : task converge, self-modèle diverge (Sprint 5C original)
    α=1 : pression égale task vs self-coherence → trade-off
    α >> 1 : self-coherence priorisée → task acc devrait s'effondrer ?

Mesure quantitative coût conscience-de-soi.

Output : pareto curve task_acc vs self_surprise_final.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.strange_loop import train_strange_loop, StrangeLoopConfig


def load_mnist(n_train=6000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def run(seed: int, weight: float) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    cfg = StrangeLoopConfig(
        epochs=5, lr=1e-3, state_dim=16, hidden=32, init_depth=2,
        observer_channels=(4, 8),
        self_surprise_weight=weight,
        log_every=30,
    )
    res = train_strange_loop(tr, va, in_dim=784, out_dim=10, cfg=cfg, verbose=False)

    h = res['history']
    surprise_curve = h['self_surprise']

    return {
        'weight': weight,
        'seed': seed,
        'final_acc': res['final_val_acc'],
        'final_params': res['final_params'],
        'task_loss_init': h['task_loss'][0] if h['task_loss'] else None,
        'task_loss_final': h['task_loss'][-1] if h['task_loss'] else None,
        'surprise_init': surprise_curve[0] if surprise_curve else None,
        'surprise_final': surprise_curve[-1] if surprise_curve else None,
        'surprise_min': min(surprise_curve) if surprise_curve else None,
        'surprise_max': max(surprise_curve) if surprise_curve else None,
        'drift_final': h['state_drift'][-1] if h['state_drift'] else None,
        'state_norm_final': h['state_norm'][-1] if h['state_norm'] else None,
        'val_acc_trajectory': h['val_acc'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.1,0.5,1.0,2.0,5.0,10.0')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint5c1')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w.strip()) for w in args.weights.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 5C.1 — self_surprise_weight sweep ===')
    print(f'Weights: {weights}  Seeds: {seeds}')

    rows = []
    for w in weights:
        print(f'\n--- weight α={w} ---')
        for seed in seeds:
            r = run(seed, w)
            rows.append(r)
            print(f'  seed={seed}: acc={r["final_acc"]:.3f}  '
                  f'surprise {r["surprise_init"]:.3f}→{r["surprise_final"]:.4f}  '
                  f'(min={r["surprise_min"]:.4f})  '
                  f'drift={r["drift_final"]:.4f}  norm={r["state_norm_final"]:.3f}')

    # Aggregate by weight
    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 5C.1 — self_surprise weight sweep\n']
    md.append('## Trade-off task accuracy vs self-coherence\n')
    md.append('| α (weight) | n_seeds | acc | surprise_final | drift_final | state_norm |')
    md.append('|---|---|---|---|---|---|')
    summary = {}
    for w in sorted(by_w.keys()):
        runs = by_w[w]
        accs = [r['final_acc'] for r in runs]
        surprises = [r['surprise_final'] for r in runs]
        drifts = [r['drift_final'] for r in runs]
        norms = [r['state_norm_final'] for r in runs]
        summary[w] = {
            'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'surprise_mean': float(np.mean(surprises)), 'surprise_std': float(np.std(surprises)),
            'drift_mean': float(np.mean(drifts)),
            'norm_mean': float(np.mean(norms)),
        }
        md.append(
            f'| {w} | {len(runs)} | '
            f'{np.mean(accs):.3f}±{np.std(accs):.3f} | '
            f'{np.mean(surprises):.4f}±{np.std(surprises):.4f} | '
            f'{np.mean(drifts):.4f} | '
            f'{np.mean(norms):.3f} |'
        )

    md.append('\n## Per-run\n')
    md.append('| α | seed | acc | surprise L→F | drift | norm |')
    md.append('|---|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["weight"]} | {r["seed"]} | {r["final_acc"]:.3f} | '
            f'{r["surprise_init"]:.3f}→{r["surprise_final"]:.4f} | '
            f'{r["drift_final"]:.4f} | {r["state_norm_final"]:.3f} |'
        )

    md.append('\n## Interprétation\n')

    # Identifie best acc + best self-coherence
    best_acc_w = max(summary, key=lambda w: summary[w]['acc_mean'])
    best_coh_w = min(summary, key=lambda w: summary[w]['surprise_mean'])
    acc_at_zero = summary[0.0]['acc_mean'] if 0.0 in summary else None
    surprise_at_zero = summary[0.0]['surprise_mean'] if 0.0 in summary else None

    md.append(f'- **Best accuracy** : α={best_acc_w} → acc={summary[best_acc_w]["acc_mean"]:.3f}')
    md.append(f'- **Best self-coherence** : α={best_coh_w} → surprise={summary[best_coh_w]["surprise_mean"]:.4f}')
    if acc_at_zero is not None:
        md.append(f'- **α=0 baseline** : acc={acc_at_zero:.3f}, surprise={surprise_at_zero:.4f}')

    # Test : forcer self-coherence dégrade-t-elle task ?
    high_w_accs = [summary[w]['acc_mean'] for w in summary if w >= 1.0]
    if high_w_accs:
        max_high = max(high_w_accs)
        if acc_at_zero is not None:
            delta = max_high - acc_at_zero
            md.append(f'- **Force self-coherence (α≥1)** : best acc={max_high:.3f} '
                      f'vs α=0 baseline={acc_at_zero:.3f} → Δ={delta:+.3f}')
            if delta < -0.02:
                md.append(f'- **VERDICT** : forcer auto-cohérence **coûte** {-delta*100:.1f}% acc')
            elif delta > 0.02:
                md.append(f'- **VERDICT** : forcer auto-cohérence **aide** task ({delta*100:.1f}%)')
            else:
                md.append(f'- **VERDICT** : auto-cohérence sans impact significatif sur task')

    (args.out / 'sweep_results.json').write_text(
        json.dumps({'rows': rows, 'summary': summary}, default=float, indent=2))
    (args.out / 'sweep_report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
