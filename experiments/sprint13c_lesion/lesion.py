"""
Sprint 13C — Causal lesion test.

Question : circuit modeling localisé dans state_predictor ou distribué dans trunk ?

Setup :
- Train avec α=1 (self-coherence engagé)
- Lesion : freeze state_predictor weights après training
- Test : self-surprise persiste-t-elle (circuit localisé) ou émerge ailleurs (distribué) ?

Variations :
- Lesion state_predictor seul
- Lesion trunk seul
- Lesion les deux
- Compare avec no-lesion baseline

Si freeze state_predictor → surprise reste basse → circuit RÉ-EXPRESSION possible (distribué)
Si freeze state_predictor → surprise explode → circuit LOCALISÉ
"""
from __future__ import annotations
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
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


def train_phase1(model, train_loader, opt, weight, epochs, device):
    """Phase 1 : train avec self-coherence."""
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


def train_phase2_with_lesion(model, train_loader, val_loader, weight, epochs, device,
                              freeze_predictor: bool, freeze_trunk: bool) -> dict:
    """Phase 2 : continue avec partial freeze."""
    # Apply lesion
    for p in model.state_predictor.parameters():
        p.requires_grad = not freeze_predictor
    for p in model.trunk.parameters():
        p.requires_grad = not freeze_trunk

    # New opt sur unfrozen params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)

    prev_state = model.self_state()
    history = {'surp_smoothed': [], 'task_loss': []}
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
            history['task_loss'].append(task.item())

    model.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = model(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    return {'final_acc': c / max(1, t), 'history': history}


def run_condition(lesion_type: str, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = SelfReferentialVG(in_dim=784, out_dim=10, state_dim=16, hidden=32,
                              depth=2, observer_channels=(4, 8)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Phase 1 : entraînement normal α=1 (3 epochs)
    train_phase1(model, tr, opt, weight=1.0, epochs=3, device=device)

    # Phase 2 : avec lesion appropriate
    if lesion_type == 'no_lesion':
        freeze_p, freeze_t = False, False
    elif lesion_type == 'freeze_predictor':
        freeze_p, freeze_t = True, False
    elif lesion_type == 'freeze_trunk':
        freeze_p, freeze_t = False, True
    elif lesion_type == 'freeze_both':
        freeze_p, freeze_t = True, True
    else:
        raise ValueError(lesion_type)

    res = train_phase2_with_lesion(model, tr, va, weight=1.0, epochs=2,
                                    device=device,
                                    freeze_predictor=freeze_p, freeze_trunk=freeze_t)

    sm = res['history']['surp_smoothed']
    return {
        'lesion_type': lesion_type,
        'seed': seed,
        'final_acc': res['final_acc'],
        'surp_init_phase2': float(sm[0]) if sm else None,
        'surp_at_50': float(sm[min(50, len(sm) - 1)]) if sm else None,
        'surp_final': float(sm[-1]) if sm else None,
        'surp_mean_phase2': float(np.mean(sm)),
        'surp_max_phase2': float(max(sm)) if sm else None,
        'final_task_loss': float(np.mean(res['history']['task_loss'][-20:])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lesions', type=str,
                    default='no_lesion,freeze_predictor,freeze_trunk,freeze_both')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint13c')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lesions = [l.strip() for l in args.lesions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 13C — Causal lesion test ===')
    print(f'Phase 1 : 3 epochs α=1. Phase 2 : 2 epochs avec lesion.\n')

    rows = []
    for lesion in lesions:
        print(f'--- {lesion} ---')
        for seed in seeds:
            r = run_condition(lesion, seed)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                  f'surp_init={r["surp_init_phase2"]:.4f}  '
                  f'final={r["surp_final"]:.4f}  '
                  f'mean={r["surp_mean_phase2"]:.4f}  '
                  f'max={r["surp_max_phase2"]:.4f}')

    md = ['# Sprint 13C — Causal lesion test\n']
    md.append('## Aggregate\n')
    md.append('| lesion | acc | surp_init | surp_final | surp_mean | surp_max |')
    md.append('|---|---|---|---|---|---|')

    by_l = {}
    for r in rows:
        by_l.setdefault(r['lesion_type'], []).append(r)

    for lesion in lesions:
        if lesion not in by_l:
            continue
        runs = by_l[lesion]
        def m(k, dec=4):
            vs = [r[k] for r in runs if r.get(k) is not None]
            return f'{np.mean(vs):.{dec}f}±{np.std(vs):.{dec}f}'
        md.append(
            f'| **{lesion}** | {m("final_acc", 3)} | {m("surp_init_phase2")} | '
            f'{m("surp_final")} | {m("surp_mean_phase2")} | {m("surp_max_phase2")} |'
        )

    md.append('\n## Verdict\n')
    if 'no_lesion' in by_l and 'freeze_predictor' in by_l:
        n_init = np.mean([r['surp_init_phase2'] for r in by_l['no_lesion']])
        n_final = np.mean([r['surp_final'] for r in by_l['no_lesion']])
        f_init = np.mean([r['surp_init_phase2'] for r in by_l['freeze_predictor']])
        f_final = np.mean([r['surp_final'] for r in by_l['freeze_predictor']])

        md.append(f'- **no_lesion** : surp {n_init:.4f} → {n_final:.4f}')
        md.append(f'- **freeze_predictor** : surp {f_init:.4f} → {f_final:.4f}')

        if f_final > n_final * 3:
            md.append('  → **CIRCUIT LOCALISÉ dans state_predictor** : freezing détruit '
                      'self-coherence. Pas de ré-expression possible.')
        elif f_final < n_final * 1.5:
            md.append('  → **CIRCUIT DISTRIBUÉ** : trunk peut maintenir self-coherence '
                      'même avec predictor gelé. Mécanisme distribué.')
        else:
            md.append('  → **DISTRIBUTION PARTIELLE** : predictor frozen perturbe '
                      'mais ne détruit pas. Mécanisme principalement (mais non exclusivement) localisé.')

    if 'no_lesion' in by_l and 'freeze_trunk' in by_l:
        n_acc = np.mean([r['final_acc'] for r in by_l['no_lesion']])
        ft_acc = np.mean([r['final_acc'] for r in by_l['freeze_trunk']])
        md.append(f'\n- **freeze_trunk** acc impact : {n_acc:.3f} → {ft_acc:.3f} (Δ={ft_acc-n_acc:+.3f})')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
