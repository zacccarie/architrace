"""
Sprint 8B — sweep prediction horizon k pour self-coherence.

Hypothèse : horizon plus long = stronger IB pressure (must compress
more temporal context into state_dim) = better transfer.

Sweep k ∈ {1, 2, 3, 5, 10} × α=2 (sweet spot transfer Sprint 7).
Mesure : MNIST acc + Fashion transfer.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.strange_loop import (
    SelfReferentialVG, StrangeLoopConfig,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'sprint7_transfer'))
from transfer import load_mnist, load_fashion, linear_probe


def run_one(horizon: int, weight: float, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    cfg = StrangeLoopConfig(
        epochs=5, lr=1e-3, state_dim=16, hidden=32, init_depth=2,
        observer_channels=(4, 8),
        self_surprise_weight=weight,
        pred_horizon=horizon,
        log_every=30,
    )

    # Build + train inline (réplique du Sprint 7 mais avec horizon)
    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = SelfReferentialVG(
        in_dim=784, out_dim=10,
        state_dim=cfg.state_dim,
        hidden=cfg.hidden, depth=cfg.init_depth,
        observer_channels=cfg.observer_channels,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    prev_state = model.self_state()
    from collections import deque
    state_buffer: deque = deque(maxlen=horizon + 1)
    state_buffer.append(prev_state)

    t0 = time.time()
    for ep in range(cfg.epochs):
        for x, y in tr:
            x = x.to(device); y = y.to(device)
            current_state = model.self_state()
            if horizon > 1 and len(state_buffer) >= horizon:
                input_state = state_buffer[0]
            else:
                input_state = prev_state
            opt.zero_grad()
            task_logits, pred_state = model(x, input_state)
            task_loss = F.cross_entropy(task_logits, y)
            pred_mean = pred_state.mean(0)
            if pred_mean.shape[0] != current_state.shape[0]:
                m = min(pred_mean.shape[0], current_state.shape[0])
                surp = ((pred_mean[:m] - current_state[:m]) ** 2).mean()
            else:
                surp = ((pred_mean - current_state) ** 2).mean()
            loss = task_loss + weight * surp
            loss.backward()
            opt.step()
            prev_state = current_state
            state_buffer.append(current_state)
    pretrain_time = time.time() - t0

    # MNIST acc
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in va:
            x = x.to(device); y = y.to(device)
            logits, _ = model(x, prev_state)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.shape[0]
    mnist_acc = correct / max(1, total)

    # Transfer FashionMNIST
    tr_f, va_f = load_fashion()
    trf = DataLoader(tr_f, batch_size=128, shuffle=True, num_workers=0)
    vaf = DataLoader(va_f, batch_size=256, num_workers=0)
    t0 = time.time()
    probe = linear_probe(model, trf, vaf, out_dim=10, epochs=8, lr=5e-3, device=device)
    probe_time = time.time() - t0

    return {
        'horizon': horizon,
        'weight': weight,
        'seed': seed,
        'mnist_acc': mnist_acc,
        'fashion_transfer_acc': probe['transfer_acc'],
        'pretrain_time': pretrain_time,
        'probe_time': probe_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizons', type=str, default='1,2,3,5,10')
    ap.add_argument('--weight', type=float, default=2.0)
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint8b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    horizons = [int(h.strip()) for h in args.horizons.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 8B — multi-step horizon sweep ===')
    print(f'horizons={horizons}  α={args.weight}  seeds={seeds}\n')

    rows = []
    for h in horizons:
        print(f'--- horizon k={h} ---')
        for seed in seeds:
            r = run_one(h, args.weight, seed)
            rows.append(r)
            print(f'  seed={seed}  mnist={r["mnist_acc"]:.4f}  '
                  f'transfer={r["fashion_transfer_acc"]:.4f}  '
                  f'({r["pretrain_time"]:.1f}s + {r["probe_time"]:.1f}s)')

    by_h = {}
    for r in rows:
        by_h.setdefault(r['horizon'], []).append(r)

    md = ['# Sprint 8B — multi-step horizon sweep\n']
    md.append(f'α={args.weight} (sweet spot transfer Sprint 7)\n')
    md.append('## Aggregate\n')
    md.append('| horizon k | MNIST acc | Fashion transfer | Δ vs k=1 |')
    md.append('|---|---|---|---|')
    h1_transfer = None
    if 1 in by_h:
        h1_transfer = float(np.mean([r['fashion_transfer_acc'] for r in by_h[1]]))
    for h in sorted(by_h):
        runs = by_h[h]
        mn = [r['mnist_acc'] for r in runs]
        ft = [r['fashion_transfer_acc'] for r in runs]
        delta = (np.mean(ft) - h1_transfer) if h1_transfer is not None else 0
        md.append(
            f'| k={h} | {np.mean(mn):.4f}±{np.std(mn):.4f} | '
            f'{np.mean(ft):.4f}±{np.std(ft):.4f} | {delta:+.4f} |'
        )

    md.append('\n## Per-run\n')
    md.append('| k | seed | MNIST | transfer |')
    md.append('|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["horizon"]} | {r["seed"]} | '
            f'{r["mnist_acc"]:.4f} | {r["fashion_transfer_acc"]:.4f} |'
        )

    md.append('\n## Verdict\n')
    if h1_transfer is not None:
        transfers = {h: float(np.mean([r['fashion_transfer_acc'] for r in by_h[h]])) for h in by_h}
        best_h = max(transfers, key=transfers.get)
        delta = transfers[best_h] - h1_transfer
        md.append(f'- **k=1 baseline** : transfer = {h1_transfer:.4f}')
        md.append(f'- **Best horizon k={best_h}** : transfer = {transfers[best_h]:.4f} '
                  f'(Δ={delta:+.4f})')
        if delta > 0.005:
            md.append(f'  → **CONFIRMÉ** : multi-step prédiction améliore transfer')
        elif delta < -0.005:
            md.append(f'  → **INVERSÉ** : single-step suffit, multi-step nuit')
        else:
            md.append(f'  → **ÉQUIVALENT** : horizon n\'affecte pas significativement')

    (args.out / 'sweep_results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'sweep_report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
