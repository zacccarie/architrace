"""
Sprint 14B — Lesion sur CNN.

Reproduit S13C sur SelfReferentialCNN. Test si circuit distribué tient
sur architecture convolutive (différente structure : conv + GAP + fusion).
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
from architrace.self_referential_cnn import SelfReferentialCNN


def load_cifar10(n_train=6000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tr = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    va = datasets.CIFAR10('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


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
                  freeze_part: str) -> dict:
    """Lesion variants : no_lesion, freeze_predictor, freeze_trunk, freeze_fusion, freeze_all_but_predictor."""
    # Default : everything trainable
    for p in model.parameters():
        p.requires_grad = True

    if freeze_part == 'freeze_predictor':
        for p in model.state_predictor.parameters():
            p.requires_grad = False
    elif freeze_part == 'freeze_trunk':
        for p in model.trunk.parameters():
            p.requires_grad = False
    elif freeze_part == 'freeze_fusion':
        for p in model.fusion.parameters():
            p.requires_grad = False
    elif freeze_part == 'freeze_all_but_predictor':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.state_predictor.parameters():
            p.requires_grad = True
    elif freeze_part != 'no_lesion':
        raise ValueError(freeze_part)

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


def run(lesion: str, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_ds, va_ds = load_cifar10()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = SelfReferentialCNN(in_ch=3, num_classes=10,
                                channels=(16, 32, 64), state_dim=16, hidden=64,
                                observer_channels=(4, 8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Phase 1 : entraînement α=1 (3 epochs)
    train_phase(model, tr, opt, weight=1.0, epochs=3, device=device)
    # Phase 2 : avec lesion (2 epochs)
    res = phase2_lesion(model, tr, va, weight=1.0, epochs=2, device=device,
                        freeze_part=lesion)
    sm = res['history']['surp_smoothed']
    return {
        'lesion': lesion,
        'seed': seed,
        'final_acc': res['final_acc'],
        'surp_init': float(sm[0]) if sm else None,
        'surp_final': float(sm[-1]) if sm else None,
        'surp_mean': float(np.mean(sm)),
        'surp_max': float(max(sm)) if sm else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lesions', type=str,
                    default='no_lesion,freeze_predictor,freeze_trunk,freeze_fusion,freeze_all_but_predictor')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint14b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lesions = [l.strip() for l in args.lesions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 14B — Lesion sur CNN (CIFAR-10) ===')
    print(f'Lesions: {lesions}  Seeds: {seeds}\n')

    rows = []
    for lesion in lesions:
        print(f'--- {lesion} ---')
        for seed in seeds:
            r = run(lesion, seed)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                  f'surp_init={r["surp_init"]:.4f}  '
                  f'final={r["surp_final"]:.4f}  '
                  f'max={r["surp_max"]:.4f}')

    md = ['# Sprint 14B — Lesion sur CNN (CIFAR-10)\n']
    md.append('## Aggregate\n')
    md.append('| lesion | acc | surp_init | surp_final | surp_mean | surp_max |')
    md.append('|---|---|---|---|---|---|')

    by_l = {}
    for r in rows:
        by_l.setdefault(r['lesion'], []).append(r)

    for lesion in lesions:
        if lesion not in by_l:
            continue
        runs = by_l[lesion]
        def m(k, dec=4):
            vs = [r[k] for r in runs if r.get(k) is not None]
            return f'{np.mean(vs):.{dec}f}±{np.std(vs):.{dec}f}'
        md.append(
            f'| **{lesion}** | {m("final_acc", 3)} | {m("surp_init")} | '
            f'{m("surp_final")} | {m("surp_mean")} | {m("surp_max")} |'
        )

    md.append('\n## Verdict\n')
    if 'no_lesion' in by_l:
        n_final = np.mean([r['surp_final'] for r in by_l['no_lesion']])
        md.append(f'- **no_lesion surp_final** : {n_final:.4f}\n')
        for lesion in lesions:
            if lesion == 'no_lesion' or lesion not in by_l:
                continue
            l_final = np.mean([r['surp_final'] for r in by_l[lesion]])
            ratio = l_final / max(n_final, 1e-9)
            md.append(f'- **{lesion}** : {l_final:.4f} (ratio {ratio:.2f}× no_lesion)')

        # Special analysis
        if 'freeze_all_but_predictor' in by_l:
            isolated_final = np.mean([r['surp_final'] for r in by_l['freeze_all_but_predictor']])
            md.append(f'\n- **freeze_all_but_predictor** : {isolated_final:.4f}')
            if isolated_final > n_final * 2:
                md.append('  → Predictor seul ne suffit pas. Trunk/fusion essentiels.')
            elif isolated_final < n_final * 1.5:
                md.append('  → Predictor seul suffit pour maintenir self-coherence. Le reste est redondant.')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
