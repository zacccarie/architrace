"""
Sprint 16B — S11.x scale CIFAR-100 + bigger CNN.

SelfReferentialCNN avec channels (32, 64, 128) sur CIFAR-100 (100 classes).
Test exaptation S11.x avec drift contrôlé (S15C noise injection).

Conditions :
- none → self : baseline
- world (Lorenz) → self : test
- fresh_self : plafond

Mesure : surp_init Phase 2 normalisé par drift injecté, ratio speedup.
"""
from __future__ import annotations
import argparse
import json
import sys as _sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.self_referential_cnn import SelfReferentialCNN
from architrace.encoder_vg import VGCNNEncoder

_sys.path.insert(0, str(architrace.DYNAMOSCOPE_PATH))
from backend.systems import lorenz  # type: ignore


def load_cifar100(n_train=8000, n_val=2000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tr = datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    va = datasets.CIFAR100('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def phase_world_lorenz_cnn(model: SelfReferentialCNN, opt, epochs, device):
    """CNN modèle pour pred Lorenz. Input dummy zeros, state porté par prev_state."""
    lz = lorenz(n=2000, dt=0.01)
    coords = np.array(lz['raw_coords'], dtype=np.float32)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    state_dim = model.state_dim
    padded = np.zeros((len(coords), state_dim), dtype=np.float32)
    padded[:, :3] = coords
    padded_t = torch.from_numpy(padded).to(device)
    # Image dummy (B, 3, 32, 32)
    zero_img = torch.zeros(1, 3, 32, 32, device=device)
    for ep in range(epochs):
        perm = np.random.permutation(len(padded) - 1)
        for batch_start in range(0, len(perm), 64):
            batch_idx = perm[batch_start:batch_start + 64]
            for i in batch_idx:
                opt.zero_grad()
                _, pred = model(zero_img, padded_t[i])
                loss = ((pred.mean(0)[:3] - padded_t[i + 1][:3]) ** 2).mean()
                loss.backward()
                opt.step()


def phase_self_with_drift_inject(model, train_loader, val_loader,
                                  opt, weight, epochs, device, inject_noise):
    prev_state = model.self_state()
    history = {'surp_smoothed': [], 'drift_inj': []}
    buf = deque(maxlen=20)
    rng = torch.Generator(device=device).manual_seed(42)

    for ep in range(epochs):
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            current_raw = model.self_state()
            noise = torch.randn(current_raw.shape, device=device, generator=rng) * inject_noise
            current_noisy = current_raw + noise

            opt.zero_grad()
            logits, pred = model(x, prev_state)
            task = F.cross_entropy(logits, y)
            pred_mean = pred.mean(0)
            if pred_mean.shape[0] != current_noisy.shape[0]:
                mm = min(pred_mean.shape[0], current_noisy.shape[0])
                surp = ((pred_mean[:mm] - current_noisy[:mm]) ** 2).mean()
            else:
                surp = ((pred_mean - current_noisy) ** 2).mean()
            (task + weight * surp).backward()
            opt.step()
            prev_state = current_noisy
            buf.append(surp.item())
            history['surp_smoothed'].append(float(np.mean(buf)))
            history['drift_inj'].append((noise ** 2).mean().item())

    model.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = model(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    return {'final_acc': c / max(1, t), 'history': history}


def run(condition: str, seed: int, inject_noise: float) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tr_ds, va_ds = load_cifar100()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = SelfReferentialCNN(in_ch=3, num_classes=100,
                                channels=(32, 64, 128), state_dim=16, hidden=128,
                                observer_channels=(4, 8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    PHASE1_EPOCHS = 4
    PHASE2_EPOCHS = 2

    t0 = time.time()
    if condition == 'none':
        prev_state = model.self_state()
        for ep in range(PHASE1_EPOCHS):
            for x, y in tr:
                x = x.to(device); y = y.to(device)
                opt.zero_grad()
                logits, _ = model(x, prev_state)
                F.cross_entropy(logits, y).backward()
                opt.step()
    elif condition == 'world':
        phase_world_lorenz_cnn(model, opt, PHASE1_EPOCHS, device)
    elif condition == 'fresh_self':
        phase_self_with_drift_inject(model, tr, va, opt,
                                      weight=1.0, epochs=PHASE1_EPOCHS, device=device,
                                      inject_noise=inject_noise)
    else:
        raise ValueError(condition)
    p1_time = time.time() - t0

    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    res = phase_self_with_drift_inject(model, tr, va, opt2,
                                        weight=1.0, epochs=PHASE2_EPOCHS, device=device,
                                        inject_noise=inject_noise)
    p2_time = time.time() - t0

    sm = res['history']['surp_smoothed']
    drifts = res['history']['drift_inj']
    return {
        'condition': condition,
        'seed': seed,
        'final_acc': res['final_acc'],
        'surp_init': float(sm[0]) if sm else None,
        'surp_final': float(sm[-1]) if sm else None,
        'mean_drift_inj': float(np.mean(drifts)),
        'p1_time': p1_time,
        'p2_time': p2_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str, default='none,world,fresh_self')
    ap.add_argument('--noise', type=float, default=0.05)
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint16b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 16B — S11.x scale CIFAR-100 ===')
    print(f'CNN channels (32, 64, 128). Conditions: {conditions}  Seeds: {seeds}\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run(cond, seed, args.noise)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.4f}  '
                  f'surp_init={r["surp_init"]:.5f}  '
                  f'final={r["surp_final"]:.5f}  '
                  f'drift_inj={r["mean_drift_inj"]:.5f}  '
                  f'({r["p1_time"]:.0f}+{r["p2_time"]:.0f}s)')

    md = ['# Sprint 16B — S11.x scale CIFAR-100\n']
    md.append(f'CNN bigger (32, 64, 128). 100 classes. Drift noise={args.noise}.\n')
    md.append('## Aggregate\n')
    md.append('| condition | acc | surp_init | surp/drift ratio | speedup vs none |')
    md.append('|---|---|---|---|---|')

    by_c = {}
    for r in rows:
        by_c.setdefault(r['condition'], []).append(r)

    none_surp = float(np.mean([r['surp_init'] for r in by_c.get('none', [])])) if 'none' in by_c else 0
    for cond in conditions:
        if cond not in by_c:
            continue
        runs = by_c[cond]
        accs = [r['final_acc'] for r in runs]
        surps = [r['surp_init'] for r in runs]
        drifts = [r['mean_drift_inj'] for r in runs]
        ratios = [s / max(d, 1e-9) for s, d in zip(surps, drifts)]
        speedup = none_surp / max(np.mean(surps), 1e-9) if none_surp else 0
        md.append(
            f'| {cond} | {np.mean(accs):.4f}±{np.std(accs):.4f} | '
            f'{np.mean(surps):.5f}±{np.std(surps):.5f} | '
            f'{np.mean(ratios):.2f}±{np.std(ratios):.2f} | '
            f'{speedup:.1f}× |'
        )

    md.append('\n## Verdict scale\n')
    if 'world' in by_c and 'none' in by_c:
        w_surp = np.mean([r['surp_init'] for r in by_c['world']])
        n_surp = np.mean([r['surp_init'] for r in by_c['none']])
        speedup = n_surp / max(w_surp, 1e-9)
        md.append(f'- S11.3 MNIST : 103×, S15C MNIST drift-controlled : 74×, S14D CIFAR-10 MLP : 6.7×')
        md.append(f'- **S16B CIFAR-100 CNN drift-controlled** : **{speedup:.1f}×**')
        if speedup > 30:
            md.append('  → Magnitude tient à scale CNN/CIFAR-100')
        elif speedup > 5:
            md.append('  → Magnitude réduite mais directionnelle robuste')
        else:
            md.append('  → Effet collapse à scale, vrai problème robustesse')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
