"""
Sprint 14D — Validation S11.x universalité à grande échelle.

S11.3 montrait world→self ratio 103× speedup sur MLP MNIST n=4.
Test ici : MLP CIFAR-10 + n=8 seeds. Plus dur et plus de stats.

3 conditions Phase 1 (3 epochs) + Phase 2 self (2 epochs) :
- none → self (baseline)
- world (Lorenz) → self (TEST universalité)
- self → self (plafond)

Si ratio speedup tient sur scale-up → S11.3 robuste.
Si ratio s'effondre → MNIST-specific.
"""
from __future__ import annotations
import argparse
import json
import sys as _sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.dyadic import DyadicAgent, _agent_state
from architrace.encoder_vg import VGCNNEncoder

_sys.path.insert(0, str(architrace.DYNAMOSCOPE_PATH))
from backend.systems import lorenz  # type: ignore


def load_cifar10_flat(n_train=5000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    va = datasets.CIFAR10('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def phase_world_lorenz(agent, opt, epochs, device, state_dim, in_dim):
    lz = lorenz(n=2000, dt=0.01)
    coords = np.array(lz['raw_coords'], dtype=np.float32)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    padded = np.zeros((len(coords), state_dim), dtype=np.float32)
    padded[:, :3] = coords
    padded_t = torch.from_numpy(padded).to(device)
    zero_x = torch.zeros(1, in_dim, device=device)
    for ep in range(epochs):
        perm = np.random.permutation(len(padded) - 1)
        for batch_start in range(0, len(perm), 64):
            batch_idx = perm[batch_start:batch_start + 64]
            for i in batch_idx:
                prev_state = padded_t[i]
                target_state = padded_t[i + 1]
                opt.zero_grad()
                _, pred = agent(zero_x, prev_state)
                pred_mean = pred.mean(0)
                loss = ((pred_mean[:3] - target_state[:3]) ** 2).mean()
                loss.backward()
                opt.step()


def phase_self_track(agent, observer, train_loader, val_loader, opt, weight, epochs, device):
    prev_state = _agent_state(agent, observer)
    history = {'surp_smoothed': []}
    buf = deque(maxlen=20)
    for ep in range(epochs):
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            current = _agent_state(agent, observer)
            opt.zero_grad()
            logits, pred = agent(x, prev_state)
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
    agent.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = agent(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    return {'final_acc': c / max(1, t), 'history': history}


def run(condition: str, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tr_ds, va_ds = load_cifar10_flat()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    state_dim = 16
    in_dim = 3072  # CIFAR-10 flattened
    agent = DyadicAgent(in_dim, 10, state_dim=state_dim, hidden=64, depth=3).to(device)
    observer = VGCNNEncoder(channels=(4, 8), input_size=64)

    opt_p1 = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    PHASE1_EPOCHS = 3
    PHASE2_EPOCHS = 2

    if condition == 'none_to_self':
        prev_state = _agent_state(agent, observer)
        for ep in range(PHASE1_EPOCHS):
            for x, y in tr:
                x = x.to(device); y = y.to(device)
                opt_p1.zero_grad()
                logits, _ = agent(x, prev_state)
                F.cross_entropy(logits, y).backward()
                opt_p1.step()
    elif condition == 'world_to_self':
        phase_world_lorenz(agent, opt_p1, PHASE1_EPOCHS, device, state_dim, in_dim)
    elif condition == 'fresh_self':
        phase_self_track(agent, observer, tr, va, opt_p1,
                         weight=1.0, epochs=PHASE1_EPOCHS, device=device)
    else:
        raise ValueError(condition)

    opt_p2 = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    res = phase_self_track(agent, observer, tr, va, opt_p2,
                            weight=1.0, epochs=PHASE2_EPOCHS, device=device)

    sm = res['history']['surp_smoothed']
    return {
        'condition': condition,
        'seed': seed,
        'final_acc': res['final_acc'],
        'phase2_surp_init': float(sm[0]) if sm else None,
        'phase2_surp_final': float(sm[-1]) if sm else None,
        'phase2_steps_to_005': next((i for i, v in enumerate(sm) if v < 0.05), len(sm)),
        'phase2_steps_to_002': next((i for i, v in enumerate(sm) if v < 0.02), len(sm)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str, default='none_to_self,world_to_self,fresh_self')
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4,5,6,7')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint14d')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 14D — S11.x universalité scale-up CIFAR-10 ===')
    print(f'Conditions: {conditions}  Seeds (n={len(seeds)}): {seeds}\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run(cond, seed)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                  f'surp_init={r["phase2_surp_init"]:.4f}  '
                  f'steps→0.05={r["phase2_steps_to_005"]}')

    md = ['# Sprint 14D — S11.x universalité scale CIFAR-10\n']
    md.append(f'MLP CIFAR-10 (3072d input), {len(seeds)} seeds.\n')
    md.append('## Aggregate\n')
    md.append('| condition | acc | surp_init | surp_final | steps→0.05 |')
    md.append('|---|---|---|---|---|')

    by_c = {}
    for r in rows:
        by_c.setdefault(r['condition'], []).append(r)

    summary = {}
    for cond in conditions:
        if cond not in by_c:
            continue
        runs = by_c[cond]
        def m(k, dec=4):
            vs = [r[k] for r in runs if r.get(k) is not None]
            return f'{np.mean(vs):.{dec}f}±{np.std(vs):.{dec}f}'
        summary[cond] = {
            'acc': float(np.mean([r['final_acc'] for r in runs])),
            'surp_init': float(np.mean([r['phase2_surp_init'] for r in runs if r['phase2_surp_init'] is not None])),
            'steps_005': float(np.mean([r['phase2_steps_to_005'] for r in runs])),
        }
        md.append(
            f'| {cond} | {m("final_acc", 3)} | {m("phase2_surp_init")} | '
            f'{m("phase2_surp_final")} | {m("phase2_steps_to_005", 1)} |'
        )

    md.append('\n## Verdict — S11.x tient sur scale CIFAR-10 ?\n')
    if 'none_to_self' in summary and 'world_to_self' in summary:
        n = summary['none_to_self']
        w = summary['world_to_self']
        ratio = n['surp_init'] / max(w['surp_init'], 1e-9)
        md.append(f'- none → self  : surp_init = {n["surp_init"]:.4f}')
        md.append(f'- world → self : surp_init = {w["surp_init"]:.4f}')
        md.append(f'- **Ratio speedup** : {ratio:.1f}×')
        md.append(f'\n  Sprint 11.3 MNIST avait ratio 103×.')
        if ratio > 30:
            md.append(f'  → **CONFIRMÉ ROBUSTE** : universalité tient sur CIFAR-10')
        elif ratio > 5:
            md.append(f'  → **AFFAIBLI** : effet présent mais diminué')
        else:
            md.append(f'  → **NE TIENT PAS** : finding était MNIST-spécifique')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows, 'summary': summary}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
