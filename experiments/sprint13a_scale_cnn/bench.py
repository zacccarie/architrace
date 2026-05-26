"""
Sprint 13A — scale validation CNN sur CIFAR-10.

Reproduit Sprint 5C.1 (α sweep self-coherence) sur :
- Architecture : SelfReferentialCNN (proper CNN, pas MLP flatten)
- Task : CIFAR-10 (real images, pas MNIST)
- N=8 seeds (vs 3 précédent)

Si finding tient sur arch + task plus sérieux → robust scale-up.
Sinon → MNIST-specific.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.self_referential_cnn import SelfReferentialCNN


def load_cifar10(n_train=8000, n_val=2000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tr = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    va = datasets.CIFAR10('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def run(weight: float, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_ds, va_ds = load_cifar10()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = SelfReferentialCNN(
        in_ch=3, num_classes=10,
        channels=(16, 32, 64), state_dim=16, hidden=64,
        observer_channels=(4, 8),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    prev_state = model.self_state()
    surprise_history = []
    task_history = []

    t0 = time.time()
    for ep in range(6):
        for x, y in tr:
            x = x.to(device); y = y.to(device)
            current = model.self_state()
            opt.zero_grad()
            logits, pred = model(x, prev_state)
            task = F.cross_entropy(logits, y)
            pred_mean = pred.mean(0)
            if pred_mean.shape[0] != current.shape[0]:
                mm = min(pred_mean.shape[0], current.shape[0])
                surp = ((pred_mean[:mm] - current[:mm]) ** 2).mean()
            else:
                surp = ((pred_mean - current) ** 2).mean()
            loss = task + weight * surp
            loss.backward()
            opt.step()
            prev_state = current
            surprise_history.append(surp.item())
            task_history.append(task.item())
    train_time = time.time() - t0

    model.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in va:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = model(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    val_acc = c / max(1, t)

    return {
        'weight': weight,
        'seed': seed,
        'val_acc': val_acc,
        'final_task_loss': float(np.mean(task_history[-30:])),
        'final_surprise': float(np.mean(surprise_history[-30:])),
        'surprise_init': float(np.mean(surprise_history[:30])),
        'train_time': train_time,
        'params': model.num_params(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.1,0.5,1.0,2.0,5.0')
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4,5,6,7')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint13a')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 13A — Scale validation CNN/CIFAR-10 ===')
    print(f'Weights: {weights}  Seeds (n={len(seeds)}): {seeds}\n')

    rows = []
    for w in weights:
        print(f'--- α={w} ---')
        for seed in seeds:
            r = run(w, seed)
            rows.append(r)
            print(f'  seed={seed}  acc={r["val_acc"]:.4f}  '
                  f'surp {r["surprise_init"]:.4f}→{r["final_surprise"]:.4f}  '
                  f'({r["train_time"]:.1f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 13A — Scale validation CNN/CIFAR-10\n']
    md.append('Setup : SelfReferentialCNN (channels 16-32-64, ~30k params), '
              f'CIFAR-10, {len(seeds)} seeds, 6 epochs.\n')
    md.append('## Aggregate\n')
    md.append('| α | val_acc | surprise final | Δ acc vs α=0 | t-stat (vs α=0) |')
    md.append('|---|---|---|---|---|')

    zero_accs = by_w.get(0.0, [])
    zero_mean = float(np.mean([r['val_acc'] for r in zero_accs])) if zero_accs else 0
    zero_std = float(np.std([r['val_acc'] for r in zero_accs])) if zero_accs else 1

    for w in sorted(by_w):
        runs = by_w[w]
        accs = [r['val_acc'] for r in runs]
        sps = [r['final_surprise'] for r in runs]
        delta = np.mean(accs) - zero_mean
        # Simple t-stat approx
        if w != 0.0 and len(zero_accs) > 1 and len(runs) > 1:
            pooled_var = (np.var(accs) + np.var([r['val_acc'] for r in zero_accs])) / 2
            t_stat = delta / max(np.sqrt(2 * pooled_var / len(runs)), 1e-9)
        else:
            t_stat = 0
        md.append(
            f'| {w} | {np.mean(accs):.4f}±{np.std(accs):.4f} | '
            f'{np.mean(sps):.4f}±{np.std(sps):.4f} | {delta:+.4f} | {t_stat:+.2f} |'
        )

    md.append('\n## Per-run\n')
    md.append('| α | seed | val_acc | surp_final | params |')
    md.append('|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["weight"]} | {r["seed"]} | {r["val_acc"]:.4f} | '
            f'{r["final_surprise"]:.4f} | {r["params"]} |'
        )

    md.append('\n## Verdict\n')
    accs_by_w = {w: float(np.mean([r['val_acc'] for r in by_w[w]])) for w in by_w}
    best_w = max(accs_by_w, key=accs_by_w.get)
    delta_best = accs_by_w[best_w] - zero_mean
    md.append(f'- **Baseline α=0** : {zero_mean:.4f}')
    md.append(f'- **Best α={best_w}** : {accs_by_w[best_w]:.4f} (Δ={delta_best:+.4f})')

    if delta_best > 0.01:
        md.append(f'  → **CONFIRMÉ SCALE-UP** : self-coherence cheap finding tient sur CNN/CIFAR-10 (+{delta_best*100:.1f}%)')
    elif delta_best > 0.005:
        md.append(f'  → **CONFIRMÉ FAIBLEMENT** : gain modeste mais positif sur scale-up')
    elif delta_best < -0.005:
        md.append(f'  → **INVERSÉ** : self-coherence DÉGRADE sur CNN/CIFAR-10')
    else:
        md.append(f'  → **NEUTRE** : finding ne réplique pas franchement sur scale-up')

    # Identifie regression α>seuil
    high_w_accs = [(w, a) for w, a in accs_by_w.items() if w >= 2.0]
    if high_w_accs:
        max_high = max(a for _, a in high_w_accs)
        if max_high < zero_mean - 0.01:
            md.append(f'  → α≥2 nuit (dépassement). Sweet spot étroit.')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
