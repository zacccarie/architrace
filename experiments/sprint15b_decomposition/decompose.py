"""
Sprint 15B — Décomposition self-surprise en drift × pred_quality.

Self-surprise = ||pred_state - current_state||²
Mais current_state = f(weights). Si weights stables → drift faible → surprise basse trivialement.

DRIFT = ||current_state[t] - current_state[t-1]||
PRED_QUALITY = corrélation pred[t] vs actual change

Decomposition :
- surprise(t) = ||pred - target||² (notre métrique)
- drift(t)   = ||target[t] - target[t-1]||² (changement intrinsèque)
- pred_qual(t) = surprise(t) / max(drift(t), ε) — relative à drift

Si lesion freeze trunk : drift baisse trivialement → surprise baisse → pred_qual peut rester haut.
Direct test artéfact S13C.
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


def train_with_decomposition(model, train_loader, val_loader,
                              opt, weight, epochs, device,
                              freeze_predictor=False, freeze_trunk=False):
    if freeze_predictor:
        for p in model.state_predictor.parameters():
            p.requires_grad = False
    if freeze_trunk:
        for p in model.trunk.parameters():
            p.requires_grad = False
    if freeze_predictor or freeze_trunk:
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)

    prev_state = model.self_state()
    prev_prev_state = prev_state.clone()
    history = {'surprise': [], 'drift': [], 'pred_quality': []}

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

            # Décomposition
            with torch.no_grad():
                drift = ((current - prev_state) ** 2).mean().item()
                # Pred quality : how much of the change does pred catch
                # If drift ≈ 0, no change to predict → surprise ≈ 0 trivially
                # Pred quality = 1 - (surprise / drift) capped to [0, 1]
                # If pred_quality > 0 → pred better than identity
                # If pred_quality ≈ 0 → pred = identity
                # If pred_quality < 0 → pred worse than identity
                identity_err = drift  # baseline : predict no change
                pred_quality = 1.0 - surp.item() / max(identity_err, 1e-9) if identity_err > 1e-9 else 0.0

            history['surprise'].append(surp.item())
            history['drift'].append(drift)
            history['pred_quality'].append(pred_quality)

            prev_prev_state = prev_state
            prev_state = current

    model.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = model(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    return {'final_acc': c / max(1, t), 'history': history}


def run(condition: str, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = SelfReferentialVG(in_dim=784, out_dim=10, state_dim=16, hidden=32,
                              depth=2, observer_channels=(4, 8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Phase 1 : train α=1, 3 epochs
    train_with_decomposition(model, tr, va, opt, weight=1.0, epochs=3, device=device)

    # Phase 2 : avec/sans lesion, 2 epochs
    if condition == 'no_lesion':
        res = train_with_decomposition(model, tr, va, opt,
                                        weight=1.0, epochs=2, device=device)
    elif condition == 'freeze_predictor':
        res = train_with_decomposition(model, tr, va, opt,
                                        weight=1.0, epochs=2, device=device,
                                        freeze_predictor=True)
    elif condition == 'freeze_trunk':
        res = train_with_decomposition(model, tr, va, opt,
                                        weight=1.0, epochs=2, device=device,
                                        freeze_trunk=True)
    else:
        raise ValueError(condition)

    h = res['history']
    return {
        'condition': condition,
        'seed': seed,
        'final_acc': res['final_acc'],
        'mean_surprise': float(np.mean(h['surprise'])),
        'mean_drift': float(np.mean(h['drift'])),
        'mean_pred_quality': float(np.mean(h['pred_quality'])),
        'final_surprise': float(np.mean(h['surprise'][-30:])),
        'final_drift': float(np.mean(h['drift'][-30:])),
        'final_pred_quality': float(np.mean(h['pred_quality'][-30:])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str,
                    default='no_lesion,freeze_predictor,freeze_trunk')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint15b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 15B — Décomposition drift × pred_quality ===\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run(cond, seed)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                  f'surp={r["final_surprise"]:.5f}  '
                  f'drift={r["final_drift"]:.5f}  '
                  f'pred_qual={r["final_pred_quality"]:.3f}')

    md = ['# Sprint 15B — Décomposition self-surprise\n']
    md.append('Self-surprise = MSE(pred, target). DRIFT = MSE(target_t, target_t-1). '
              'PRED_QUALITY = 1 - surprise/drift.\n')
    md.append('Si pred_quality > 0 → predictor mieux qu\'identity. '
              'Si ≈ 0 → predictor = identity. Si < 0 → pire.\n')

    md.append('## Aggregate\n')
    md.append('| condition | acc | surprise | drift | pred_quality |')
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
        md.append(
            f'| **{cond}** | {m("final_acc", 3)} | {m("final_surprise")} | '
            f'{m("final_drift")} | {m("final_pred_quality", 3)} |'
        )

    md.append('\n## Verdict — artéfact ou vrai distribution ?\n')
    if 'no_lesion' in by_c:
        nl = by_c['no_lesion']
        nl_drift = float(np.mean([r['final_drift'] for r in nl]))
        nl_pq = float(np.mean([r['final_pred_quality'] for r in nl]))
        md.append(f'- **no_lesion** : drift={nl_drift:.5f}, pred_quality={nl_pq:.3f}')

        for cond in ['freeze_predictor', 'freeze_trunk']:
            if cond in by_c:
                runs = by_c[cond]
                c_drift = float(np.mean([r['final_drift'] for r in runs]))
                c_pq = float(np.mean([r['final_pred_quality'] for r in runs]))
                drift_ratio = c_drift / max(nl_drift, 1e-9)
                pq_delta = c_pq - nl_pq
                md.append(f'- **{cond}** : drift={c_drift:.5f} (ratio {drift_ratio:.2f}× '
                          f'no_lesion), pred_quality={c_pq:.3f} (Δ={pq_delta:+.3f})')

                if drift_ratio < 0.5 and abs(pq_delta) < 0.1:
                    md.append(f'  → **ARTÉFACT CONFIRMÉ** : drift baisse mais '
                              f'pred_quality identique. Surprise basse triviale.')
                elif pq_delta < -0.2:
                    md.append(f'  → **VRAI IMPACT** : pred_quality dégrade ({pq_delta:+.3f}). '
                              f'Mécanisme perturbed.')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
