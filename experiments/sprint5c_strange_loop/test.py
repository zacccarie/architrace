"""
Sprint 5C — test convergence boucle étrange auto-référentielle.

Question : VG qui se contient comme input converge vers fixed-point ?
Diverge ? Boucle paradoxale prend forme ou s'écrase ?

Mesures :
- task_acc finale (le réseau apprend-il toujours la tâche ?)
- self_surprise dynamics (descend = stabilité auto-modèle ; spike = drift)
- state_drift over time (vitesse changement représentation soi)
- state_norm (saturation / collapse)
- Compare avec baseline VG non-référentiel sur même tâche
"""
from __future__ import annotations
import argparse
import json
import time
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


def run_one(seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    cfg = StrangeLoopConfig(
        epochs=5, lr=1e-3, state_dim=16, hidden=32, init_depth=2,
        observer_channels=(4, 8),
        log_every=30,
    )
    t0 = time.time()
    res = train_strange_loop(
        tr, va, in_dim=784, out_dim=10, cfg=cfg, verbose=False,
    )
    elapsed = time.time() - t0

    h = res['history']

    # Analyse dynamics
    surprise_curve = h['self_surprise']
    drift_curve = h['state_drift']
    state_norm_curve = h['state_norm']

    surprise_trend = (np.mean(surprise_curve[-5:]) - np.mean(surprise_curve[:5])
                      if len(surprise_curve) >= 10 else 0)
    drift_trend = (np.mean(drift_curve[-5:]) - np.mean(drift_curve[:5])
                   if len(drift_curve) >= 10 else 0)

    return {
        'seed': seed,
        'elapsed': elapsed,
        'final_acc': res['final_val_acc'],
        'final_params': res['final_params'],
        'final_state_norm': res['final_state_norm'],
        'task_loss_init': h['task_loss'][0] if h['task_loss'] else None,
        'task_loss_final': h['task_loss'][-1] if h['task_loss'] else None,
        'surprise_init': surprise_curve[0] if surprise_curve else None,
        'surprise_final': surprise_curve[-1] if surprise_curve else None,
        'surprise_trend': float(surprise_trend),
        'surprise_max': float(max(surprise_curve)) if surprise_curve else None,
        'drift_init': drift_curve[0] if drift_curve else None,
        'drift_final': drift_curve[-1] if drift_curve else None,
        'drift_trend': float(drift_trend),
        'state_norm_init': state_norm_curve[0] if state_norm_curve else None,
        'state_norm_final': state_norm_curve[-1] if state_norm_curve else None,
        'val_acc_curve': h['val_acc'],
        'val_loss_curve': h['val_loss'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint5c')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 5C — strange loop self-referential ===')
    print(f'Seeds: {seeds}')

    rows = []
    for seed in seeds:
        print(f'\n[seed {seed}] ...')
        r = run_one(seed)
        rows.append(r)
        print(f'  acc={r["final_acc"]:.3f}  params={r["final_params"]}  '
              f'({r["elapsed"]:.1f}s)')
        print(f'  task_loss: {r["task_loss_init"]:.3f} → {r["task_loss_final"]:.3f}')
        print(f'  self_surprise: {r["surprise_init"]:.4f} → {r["surprise_final"]:.4f}  '
              f'trend={r["surprise_trend"]:+.4f}  max={r["surprise_max"]:.4f}')
        print(f'  state_drift: {r["drift_init"]:.3f} → {r["drift_final"]:.3f}  '
              f'trend={r["drift_trend"]:+.3f}')
        print(f'  state_norm: {r["state_norm_init"]:.3f} → {r["state_norm_final"]:.3f}')

    # Aggregate
    accs = [r['final_acc'] for r in rows]
    surprise_finals = [r['surprise_final'] for r in rows]
    drift_finals = [r['drift_final'] for r in rows]
    state_norms = [r['state_norm_final'] for r in rows]

    md = ['# Sprint 5C — Strange loop self-référence\n']
    md.append('## Verdict global\n')
    md.append(f'- N runs: {len(rows)}')
    md.append(f'- **Final val_acc** : {np.mean(accs):.3f} ± {np.std(accs):.3f}')
    md.append(f'- **Final self_surprise** : {np.mean(surprise_finals):.4f} ± {np.std(surprise_finals):.4f}')
    md.append(f'- **Final state_drift** : {np.mean(drift_finals):.3f} ± {np.std(drift_finals):.3f}')
    md.append(f'- **Final state_norm** : {np.mean(state_norms):.3f} ± {np.std(state_norms):.3f}')

    md.append('\n## Per-run\n')
    md.append('| seed | acc | params | task L→F | surprise L→F | drift L→F | norm L→F |')
    md.append('|---|---|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["seed"]} | {r["final_acc"]:.3f} | {r["final_params"]} | '
            f'{r["task_loss_init"]:.2f}→{r["task_loss_final"]:.2f} | '
            f'{r["surprise_init"]:.3f}→{r["surprise_final"]:.3f} | '
            f'{r["drift_init"]:.2f}→{r["drift_final"]:.2f} | '
            f'{r["state_norm_init"]:.2f}→{r["state_norm_final"]:.2f} |'
        )

    md.append('\n## Interprétation dynamique\n')
    surprise_trends = [r['surprise_trend'] for r in rows]
    drift_trends = [r['drift_trend'] for r in rows]
    if all(t < 0 for t in surprise_trends):
        md.append('- **Self-surprise descend systématiquement** → réseau converge vers '
                  'modèle stable de lui-même (auto-prédiction améliore avec entraînement)')
    elif all(t > 0 for t in surprise_trends):
        md.append('- **Self-surprise augmente** → mécanisme divergent, réseau '
                  'perd capacité auto-prédiction au fil training')
    else:
        md.append('- **Self-surprise trend mixte** : pas de tendance claire inter-seed')

    if all(d < 0 for d in drift_trends):
        md.append('- **State drift décroît** → représentation soi se stabilise '
                  '(fixed-point convergence ?)')
    elif all(d > 0 for d in drift_trends):
        md.append('- **State drift croît** → représentation soi continue d\'évoluer '
                  '(pas de fixed-point dans cette window)')

    if np.mean(accs) > 0.85:
        md.append(f'- **Tâche apprise malgré boucle étrange** : acc moyenne {np.mean(accs):.3f}. '
                  'Auto-référence n\'empêche pas convergence task.')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
