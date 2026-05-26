"""
Sprint 14A — Lésion fine par layer trunk.

S13C montrait circuit distribué (freeze predictor seul ne détruit pas surp).
Question : quelles couches du trunk contribuent le plus ?

Phase 1 : train α=1 (3 epochs).
Phase 2 : freeze SEULEMENT 1 layer trunk (rest unfrozen), 2 epochs.
Mesure surp_final vs no_lesion baseline.

Si lesion layer i fait exploser surp → layer i critique pour modeling.
Si toutes équivalentes → distribution uniforme.
Si une fait beaucoup plus que les autres → concentration.
"""
from __future__ import annotations
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.strange_loop import SelfReferentialVG


def load_mnist(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def get_trunk_linears(model: SelfReferentialVG) -> list[nn.Linear]:
    """Liste des Linear dans trunk en ordre."""
    return [m for m in model.trunk if isinstance(m, nn.Linear)]


def train_phase(model, train_loader, opt, weight, epochs, device):
    prev_state = model.self_state()
    for ep in range(epochs):
        for x, y in train_loader:
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
            (task + weight * surp).backward()
            opt.step()
            prev_state = current
    return prev_state


def phase2_lesion(model, train_loader, val_loader, weight, epochs, device,
                  lesion_layer_idx: int = -1) -> dict:
    """lesion_layer_idx = -1 (no_lesion), 0, 1, ... pour freeze layer i trunk."""
    linears = get_trunk_linears(model)
    if lesion_layer_idx >= 0 and lesion_layer_idx < len(linears):
        for p in linears[lesion_layer_idx].parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)

    prev_state = model.self_state()
    history = {'surp_smoothed': []}
    buf = deque(maxlen=20)

    for ep in range(epochs):
        for x, y in train_loader:
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
            (task + weight * surp).backward()
            opt.step()
            prev_state = current
            buf.append(surp.item())
            history['surp_smoothed'].append(float(np.mean(buf)))

    model.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = model(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    return {'final_acc': c / max(1, t), 'history': history}


def run(lesion_idx: int, seed: int, depth: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = SelfReferentialVG(in_dim=784, out_dim=10, state_dim=16, hidden=32,
                              depth=depth, observer_channels=(4, 8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_phase(model, tr, opt, weight=1.0, epochs=3, device=device)
    res = phase2_lesion(model, tr, va, weight=1.0, epochs=2, device=device,
                        lesion_layer_idx=lesion_idx)
    sm = res['history']['surp_smoothed']
    return {
        'lesion_idx': lesion_idx,
        'seed': seed,
        'depth': depth,
        'final_acc': res['final_acc'],
        'surp_init': float(sm[0]) if sm else None,
        'surp_final': float(sm[-1]) if sm else None,
        'surp_mean': float(np.mean(sm)),
        'surp_max': float(max(sm)) if sm else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth', type=int, default=3, help='Trunk depth (number of layers)')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint14a')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    # Lesion indices : -1 (no_lesion), 0, 1, ..., depth-1
    lesion_indices = [-1] + list(range(args.depth))

    print(f'=== Sprint 14A — Layer-by-layer lesion ===')
    print(f'Trunk depth={args.depth}  Lesion indices: {lesion_indices}  Seeds: {seeds}\n')

    rows = []
    for idx in lesion_indices:
        label = 'no_lesion' if idx == -1 else f'layer_{idx}'
        print(f'--- {label} ---')
        for seed in seeds:
            r = run(idx, seed, args.depth)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                  f'surp_init={r["surp_init"]:.4f}  '
                  f'final={r["surp_final"]:.4f}  '
                  f'max={r["surp_max"]:.4f}')

    md = ['# Sprint 14A — Layer-by-layer lesion\n']
    md.append(f'Trunk depth={args.depth}. Freeze 1 layer Phase 2, mesure self-coherence impact.\n')
    md.append('## Aggregate\n')
    md.append('| lesion | acc | surp_init | surp_final | surp_mean | surp_max |')
    md.append('|---|---|---|---|---|---|')

    by_idx = {}
    for r in rows:
        by_idx.setdefault(r['lesion_idx'], []).append(r)

    for idx in lesion_indices:
        if idx not in by_idx:
            continue
        runs = by_idx[idx]
        def m(k, dec=4):
            vs = [r[k] for r in runs if r.get(k) is not None]
            return f'{np.mean(vs):.{dec}f}±{np.std(vs):.{dec}f}'
        label = 'no_lesion' if idx == -1 else f'layer_{idx}'
        md.append(
            f'| **{label}** | {m("final_acc", 3)} | {m("surp_init")} | '
            f'{m("surp_final")} | {m("surp_mean")} | {m("surp_max")} |'
        )

    md.append('\n## Verdict\n')
    no_lesion_final = np.mean([r['surp_final'] for r in by_idx.get(-1, [])])
    md.append(f'- **no_lesion** surp_final : {no_lesion_final:.4f}\n')

    impacts = {}
    for idx in lesion_indices:
        if idx < 0 or idx not in by_idx:
            continue
        lesion_final = np.mean([r['surp_final'] for r in by_idx[idx]])
        impact = lesion_final / max(no_lesion_final, 1e-9)
        impacts[idx] = impact
        md.append(f'- **layer_{idx}** surp_final : {lesion_final:.4f} (ratio {impact:.2f}× no_lesion)')

    if impacts:
        max_layer = max(impacts, key=impacts.get)
        min_layer = min(impacts, key=impacts.get)
        spread = max(impacts.values()) / max(min(impacts.values()), 1e-9)
        md.append(f'\n- **Layer le plus critique** : {max_layer} (ratio {impacts[max_layer]:.2f}×)')
        md.append(f'- **Layer le moins critique** : {min_layer} (ratio {impacts[min_layer]:.2f}×)')
        md.append(f'- **Spread max/min** : {spread:.2f}×')

        if spread > 3:
            md.append(f'\n  → **CONCENTRATION** : circuit principalement dans layer(s) critique(s)')
        elif spread < 1.5:
            md.append(f'\n  → **DISTRIBUTION UNIFORME** : toutes layers équivalentes pour self-coherence')
        else:
            md.append(f'\n  → **DISTRIBUTION HÉTÉROGÈNE** : modérément concentré')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
