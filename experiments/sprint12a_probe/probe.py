"""
Sprint 12A — Probe interpretability circuit modeling.

Compare état interne de SelfReferentialVG entraîné avec :
- α=0 (pas de self-coherence)
- α=1 (sweet spot Sprint 5C.1)
- α=2 (transfer optimum Sprint 7)

Mesures :
- Spectral analysis state_predictor weight matrix (singular values, effective rank)
- Activation statistics trunk last layer
- Self_state distribution (norm, variance, entropy)
- Probe : peut-on lire la topologie depuis state_predictor weights ?

Question : quelle est la signature interprétable du "modeling" ?
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
from architrace.strange_loop import SelfReferentialVG, StrangeLoopConfig


def load_mnist(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def train_model(weight: float, seed: int, device) -> tuple:
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    cfg = StrangeLoopConfig(epochs=5, lr=1e-3, state_dim=16, hidden=32,
                            init_depth=2, observer_channels=(4, 8),
                            self_surprise_weight=weight, log_every=40)
    model = SelfReferentialVG(in_dim=784, out_dim=10,
                              state_dim=cfg.state_dim,
                              hidden=cfg.hidden, depth=cfg.init_depth,
                              observer_channels=cfg.observer_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    prev_state = model.self_state()

    for ep in range(cfg.epochs):
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

    return model, va, prev_state


def probe_model(model: SelfReferentialVG, val_loader, prev_state, device) -> dict:
    """Extraction des signatures internes."""
    model.eval()

    # 1. State_predictor weight analysis
    sp_weight = model.state_predictor.weight.detach().cpu().numpy()
    U, S, Vt = np.linalg.svd(sp_weight, full_matrices=False)
    # Effective rank via entropy
    p = S / S.sum()
    p_safe = np.where(p > 1e-12, p, 1e-12)
    eff_rank = float(np.exp(-(p_safe * np.log(p_safe)).sum()))
    spectral_radius = float(S[0])
    cond_number = float(S[0] / max(S[-1], 1e-12))

    # 2. Trunk last layer activations on probe batch
    activations = []
    self_states = []
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            # Manual forward to capture intermediate
            B = x.shape[0]
            ps = prev_state.unsqueeze(0).expand(B, -1)
            z = model.trunk(torch.cat([x, ps], dim=-1))
            activations.append(z.cpu().numpy())
            self_states.append(model.self_state().cpu().numpy())
            logits = model.task_head(z)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.shape[0]

    acts = np.concatenate(activations, axis=0)
    # Activation statistics
    act_mean_abs = float(np.abs(acts).mean())
    act_std = float(acts.std())
    act_sparsity = float((acts == 0).mean())
    # Effective rank activations
    Us, Ss, Vts = np.linalg.svd(acts - acts.mean(0), full_matrices=False)
    p_a = Ss / Ss.sum()
    p_a_safe = np.where(p_a > 1e-12, p_a, 1e-12)
    act_eff_rank = float(np.exp(-(p_a_safe * np.log(p_a_safe)).sum()))

    # Self_state stats
    states = np.stack(self_states)
    state_norm = float(np.linalg.norm(states, axis=1).mean())
    state_std = float(states.std())

    return {
        'val_acc': correct / max(1, total),
        'sp_eff_rank': eff_rank,
        'sp_spectral_radius': spectral_radius,
        'sp_condition_number': cond_number,
        'sp_singular_values_top5': [float(s) for s in S[:5]],
        'act_mean_abs': act_mean_abs,
        'act_std': act_std,
        'act_sparsity': act_sparsity,
        'act_eff_rank': act_eff_rank,
        'state_norm': state_norm,
        'state_std': state_std,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,1.0,2.0,5.0')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint12a')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    print(f'=== Sprint 12A — Probe interpretability ===')
    print(f'Weights: {weights}  Seeds: {seeds}\n')

    rows = []
    for w in weights:
        print(f'--- α={w} ---')
        for seed in seeds:
            model, va, ps = train_model(w, seed, device)
            probe = probe_model(model, va, ps, device)
            probe.update({'weight': w, 'seed': seed})
            rows.append(probe)
            print(f'  seed={seed}  acc={probe["val_acc"]:.3f}  '
                  f'sp_eff_rank={probe["sp_eff_rank"]:.2f}  '
                  f'sp_spec={probe["sp_spectral_radius"]:.3f}  '
                  f'sp_cond={probe["sp_condition_number"]:.1f}  '
                  f'act_eff_rank={probe["act_eff_rank"]:.2f}  '
                  f'state_norm={probe["state_norm"]:.3f}')

    # Aggregate
    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 12A — Probe interpretability\n']
    md.append('## Internal signatures par α\n')
    md.append('| α | acc | sp_eff_rank | sp_spectral | sp_cond | act_eff_rank | state_norm |')
    md.append('|---|---|---|---|---|---|---|')
    for w in sorted(by_w):
        runs = by_w[w]
        def m(k, dec=3):
            vs = [r[k] for r in runs]
            return f'{np.mean(vs):.{dec}f}±{np.std(vs):.{dec}f}'
        md.append(
            f'| {w} | {m("val_acc")} | {m("sp_eff_rank", 2)} | '
            f'{m("sp_spectral_radius")} | {m("sp_condition_number", 1)} | '
            f'{m("act_eff_rank", 2)} | {m("state_norm")} |'
        )

    md.append('\n## Per-run\n')
    md.append('| α | seed | acc | sp_eff_rank | sp_spectral | sp_cond | top5 SV |')
    md.append('|---|---|---|---|---|---|---|')
    for r in rows:
        sv5 = ', '.join(f'{s:.2f}' for s in r['sp_singular_values_top5'])
        md.append(
            f'| {r["weight"]} | {r["seed"]} | {r["val_acc"]:.3f} | '
            f'{r["sp_eff_rank"]:.2f} | {r["sp_spectral_radius"]:.3f} | '
            f'{r["sp_condition_number"]:.1f} | {sv5} |'
        )

    md.append('\n## Interprétation\n')
    if 0.0 in by_w and 1.0 in by_w:
        r0 = by_w[0.0]; r1 = by_w[1.0]
        delta_rank = np.mean([r['sp_eff_rank'] for r in r1]) - np.mean([r['sp_eff_rank'] for r in r0])
        delta_spec = np.mean([r['sp_spectral_radius'] for r in r1]) - np.mean([r['sp_spectral_radius'] for r in r0])
        delta_actr = np.mean([r['act_eff_rank'] for r in r1]) - np.mean([r['act_eff_rank'] for r in r0])

        md.append(f'- **Δ state_predictor eff_rank (α=1 vs α=0)** : {delta_rank:+.2f}')
        md.append(f'- **Δ state_predictor spectral_radius** : {delta_spec:+.3f}')
        md.append(f'- **Δ activation eff_rank** : {delta_actr:+.2f}')

        if delta_rank > 1:
            md.append('  → Self-coherence augmente rang effectif state_predictor → utilise plus de dimensions')
        elif delta_rank < -1:
            md.append('  → Self-coherence DIMINUE rang state_predictor → compression IB confirmée')

        if delta_actr < -1:
            md.append('  → Activations trunk plus compressées avec self-coherence (low-rank features)')
        elif delta_actr > 1:
            md.append('  → Activations trunk plus dispersées avec self-coherence')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
