"""
Sprint 15C — S11.x avec drift injecté pour égaliser baseline.

Sprint 15B révèle drift→0. Donc self-surprise mesure surtout bruit observer.
Pour avoir une mesure propre du prediction skill, INJECTER du drift contrôlé :
ajouter noise au target_state pour forcer drift fixe across conditions.

Si exaptation tient AVEC drift égalisé → finding S11.x robuste.
Si exaptation s'effondre → finding était artéfact drift comparison.
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


def load_mnist(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def phase_world(agent, opt, epochs, device, state_dim, in_dim):
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
                opt.zero_grad()
                _, pred = agent(zero_x, padded_t[i])
                loss = ((pred.mean(0)[:3] - padded_t[i + 1][:3]) ** 2).mean()
                loss.backward()
                opt.step()


def phase_self_with_drift_inject(agent, observer, train_loader, val_loader,
                                  opt, weight, epochs, device,
                                  inject_noise: float):
    """Phase 2 with controlled drift via target noise injection."""
    prev_state = _agent_state(agent, observer)
    history = {'surp_smoothed': [], 'drift_obs': [], 'drift_inject': []}
    buf = deque(maxlen=20)
    rng = torch.Generator(device=device).manual_seed(42)

    for ep in range(epochs):
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            current_raw = _agent_state(agent, observer)
            # Inject Gaussian noise to target → forces drift = noise scale
            noise = torch.randn(current_raw.shape, device=device, generator=rng) * inject_noise
            current_noisy = current_raw + noise

            opt.zero_grad()
            logits, pred = agent(x, prev_state)
            task = F.cross_entropy(logits, y)
            pred_mean = pred.mean(0)
            if pred_mean.shape[0] != current_noisy.shape[0]:
                mm = min(pred_mean.shape[0], current_noisy.shape[0])
                surp = ((pred_mean[:mm] - current_noisy[:mm]) ** 2).mean()
            else:
                surp = ((pred_mean - current_noisy) ** 2).mean()
            (task + weight * surp).backward()
            opt.step()

            with torch.no_grad():
                drift_obs = ((current_raw - prev_state) ** 2).mean().item()
                drift_inj = (noise ** 2).mean().item()
            prev_state = current_noisy
            buf.append(surp.item())
            history['surp_smoothed'].append(float(np.mean(buf)))
            history['drift_obs'].append(drift_obs)
            history['drift_inject'].append(drift_inj)

    agent.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = agent(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    return {'final_acc': c / max(1, t), 'history': history}


def run(condition: str, seed: int, inject_noise: float) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)
    state_dim = 16
    agent = DyadicAgent(784, 10, state_dim=state_dim, hidden=32, depth=2).to(device)
    observer = VGCNNEncoder(channels=(4, 8), input_size=64)
    opt_p1 = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    PHASE1_EPOCHS = 3
    PHASE2_EPOCHS = 2

    if condition == 'none':
        prev_state = _agent_state(agent, observer)
        for ep in range(PHASE1_EPOCHS):
            for x, y in tr:
                x = x.to(device); y = y.to(device)
                opt_p1.zero_grad()
                logits, _ = agent(x, prev_state)
                F.cross_entropy(logits, y).backward()
                opt_p1.step()
    elif condition == 'world':
        phase_world(agent, opt_p1, PHASE1_EPOCHS, device, state_dim, 784)
    elif condition == 'fresh_self':
        # Self pretraining with drift injection too
        phase_self_with_drift_inject(agent, observer, tr, va, opt_p1,
                                      weight=1.0, epochs=PHASE1_EPOCHS, device=device,
                                      inject_noise=inject_noise)
    else:
        raise ValueError(condition)

    opt_p2 = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    res = phase_self_with_drift_inject(agent, observer, tr, va, opt_p2,
                                        weight=1.0, epochs=PHASE2_EPOCHS, device=device,
                                        inject_noise=inject_noise)
    sm = res['history']['surp_smoothed']
    return {
        'condition': condition,
        'seed': seed,
        'inject_noise': inject_noise,
        'final_acc': res['final_acc'],
        'phase2_surp_init': float(sm[0]) if sm else None,
        'phase2_surp_final': float(sm[-1]) if sm else None,
        'mean_drift_obs': float(np.mean(res['history']['drift_obs'])),
        'mean_drift_inject': float(np.mean(res['history']['drift_inject'])),
        'phase2_steps_to_below_drift': next(
            (i for i, v in enumerate(sm)
             if v < float(np.mean(res['history']['drift_inject'])) * 0.5),
            len(sm)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str, default='none,world,fresh_self')
    ap.add_argument('--noise', type=float, default=0.05)
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint15c')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 15C — S11.x avec drift contrôlé (noise={args.noise}) ===\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run(cond, seed, args.noise)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                  f'surp_init={r["phase2_surp_init"]:.5f}  '
                  f'drift_obs={r["mean_drift_obs"]:.5f}  '
                  f'drift_inj={r["mean_drift_inject"]:.5f}')

    md = ['# Sprint 15C — S11.x avec drift égalisé\n']
    md.append(f'Noise injecté = {args.noise} → drift target uniforme across conditions.\n')
    md.append('## Aggregate\n')
    md.append('| condition | acc | surp_init Phase 2 | drift_inj | surp/drift ratio |')
    md.append('|---|---|---|---|---|')

    by_c = {}
    for r in rows:
        by_c.setdefault(r['condition'], []).append(r)

    for cond in conditions:
        if cond not in by_c:
            continue
        runs = by_c[cond]
        def m(k, dec=5):
            vs = [r[k] for r in runs]
            return f'{np.mean(vs):.{dec}f}±{np.std(vs):.{dec}f}'
        surps = [r['phase2_surp_init'] for r in runs]
        drifts = [r['mean_drift_inject'] for r in runs]
        ratios = [s / max(d, 1e-9) for s, d in zip(surps, drifts)]
        md.append(
            f'| {cond} | {m("final_acc", 3)} | {m("phase2_surp_init")} | '
            f'{m("mean_drift_inject")} | {np.mean(ratios):.3f}±{np.std(ratios):.3f} |'
        )

    md.append('\n## Verdict\n')
    if 'none' in by_c and 'world' in by_c:
        n_surp = np.mean([r['phase2_surp_init'] for r in by_c['none']])
        w_surp = np.mean([r['phase2_surp_init'] for r in by_c['world']])
        n_ratio = np.mean([r['phase2_surp_init'] / max(r['mean_drift_inject'], 1e-9) for r in by_c['none']])
        w_ratio = np.mean([r['phase2_surp_init'] / max(r['mean_drift_inject'], 1e-9) for r in by_c['world']])
        speedup_surp = n_surp / max(w_surp, 1e-9)
        speedup_ratio = n_ratio / max(w_ratio, 1e-9)
        md.append(f'- **none → self** : surp_init={n_surp:.5f}, ratio={n_ratio:.3f}')
        md.append(f'- **world → self** : surp_init={w_surp:.5f}, ratio={w_ratio:.3f}')
        md.append(f'- **Ratio brut speedup (surp)** : {speedup_surp:.1f}×')
        md.append(f'- **Ratio normalized (surp/drift)** : {speedup_ratio:.1f}×')
        md.append(f'\n  Sprint 11.3 MNIST sans noise : ratio 103×.')
        if speedup_ratio > 30:
            md.append('  → **EFFET ROBUSTE** : exaptation persiste même avec drift égalisé')
        elif speedup_ratio > 5:
            md.append('  → **EFFET PARTIEL** : exaptation diminue mais reste mesurable')
        else:
            md.append('  → **EFFET ÉFFONDRE** : finding 11.x était artéfact drift baseline différent')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
