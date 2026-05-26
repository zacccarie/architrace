"""
Sprint 11 — bench dyadic vs self vs none.

Conditions :
- none : 2 agents indépendants, α=0 (baseline)
- self : chaque agent self-models (Sprint 5C.1 style, repliqué dans dyadic frame)
- mutual : chaque agent modélise l'autre (Sprint 11 nouveauté)
- mutual_strong : mutual avec α=2 (test sensibilité)

Mesures :
- task acc A, task acc B
- alignment final A↔B (Pearson corr états)
- surprise levels A et B
- transfer quality (probe sur task C)
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
from architrace.dyadic import DyadicAgent, DyadicConfig, train_dyadic, _agent_state
from architrace.encoder_vg import VGCNNEncoder


def load_mnist(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def load_fashion(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.FashionMNIST('./data', train=True, download=True, transform=tf)
    va = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def linear_probe(agent: DyadicAgent, transfer_train, transfer_val,
                 out_dim: int, device: torch.device, epochs=8, lr=5e-3):
    """Probe sur trunk d'agent. Trunk frozen, new head."""
    for p in agent.parameters():
        p.requires_grad = False
    last_out = None
    for m in agent.trunk.modules():
        if isinstance(m, nn.Linear):
            last_out = m.out_features
    head = nn.Linear(last_out, out_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    zero_state = torch.zeros(agent.state_dim, device=device)

    @torch.no_grad()
    def _trunk(x):
        return agent.trunk(torch.cat([x, zero_state.unsqueeze(0).expand(x.shape[0], -1)], dim=-1))

    for ep in range(epochs):
        for x, y in transfer_train:
            x = x.to(device); y = y.to(device)
            logits = head(_trunk(x))
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    c = 0; t = 0
    with torch.no_grad():
        for x, y in transfer_val:
            x = x.to(device); y = y.to(device)
            c += (head(_trunk(x)).argmax(-1) == y).sum().item()
            t += y.shape[0]
    return c / max(1, t)


def run_condition(condition: str, seed: int) -> dict:
    """3 conditions : none (α=0), mutual (cross-modeling α=1), mutual_strong (α=2).

    Setup : 2 agents apprennent MNIST (deux instances avec seeds diff pour diversité initiale).
    Transfer test : FashionMNIST.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    if condition == 'none':
        weight = 0.0
    elif condition == 'mutual':
        weight = 1.0
    elif condition == 'mutual_strong':
        weight = 2.0
    else:
        raise ValueError(condition)

    # Deux instances MNIST avec seeds différents
    torch.manual_seed(seed); np.random.seed(seed)
    tr_a, va_a = load_mnist()
    torch.manual_seed(seed + 1000); np.random.seed(seed + 1000)
    tr_b, va_b = load_mnist()
    tr_a_loader = DataLoader(tr_a, batch_size=128, shuffle=True, num_workers=0)
    va_a_loader = DataLoader(va_a, batch_size=256, num_workers=0)
    tr_b_loader = DataLoader(tr_b, batch_size=128, shuffle=True, num_workers=0)
    va_b_loader = DataLoader(va_b, batch_size=256, num_workers=0)

    torch.manual_seed(seed); np.random.seed(seed)
    cfg = DyadicConfig(
        epochs=5, lr=1e-3,
        state_dim=16, hidden=32, depth=2,
        mutual_weight=weight,
        observer_channels=(4, 8),
        log_every=40,
    )

    t0 = time.time()
    res = train_dyadic(
        tr_a_loader, va_a_loader,
        tr_b_loader, va_b_loader,
        in_dim_a=784, out_dim_a=10,
        in_dim_b=784, out_dim_b=10,
        cfg=cfg, device=device, verbose=False,
    )
    train_time = time.time() - t0

    # Transfer probe : utilise agent A pour FashionMNIST transfer
    tr_f, va_f = load_fashion()
    trf = DataLoader(tr_f, batch_size=128, shuffle=True, num_workers=0)
    vaf = DataLoader(va_f, batch_size=256, num_workers=0)
    t0 = time.time()
    transfer_a = linear_probe(res['agent_a'], trf, vaf, out_dim=10, device=device)
    transfer_b = linear_probe(res['agent_b'], trf, vaf, out_dim=10, device=device)
    probe_time = time.time() - t0

    return {
        'condition': condition,
        'weight': weight,
        'seed': seed,
        'final_acc_a': res['final_acc_a'],
        'final_acc_b': res['final_acc_b'],
        'final_surprise_a': res['final_surprise_a'],
        'final_surprise_b': res['final_surprise_b'],
        'final_alignment': res['final_alignment'],
        'mean_alignment_late': res['mean_alignment_late'],
        'transfer_a': transfer_a,
        'transfer_b': transfer_b,
        'train_time': train_time,
        'probe_time': probe_time,
        'history_alignments': res['history']['alignment_ab'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str, default='none,mutual,mutual_strong')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint11')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 11 — Dyadic mutual modeling ===')
    print(f'Conditions: {conditions}  Seeds: {seeds}')
    print(f'Test : conscience-de-soi relationnelle vs individuelle\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run_condition(cond, seed)
            rows.append(r)
            al = r["mean_alignment_late"]
            al_s = f'{al:.3f}' if al is not None else '—'
            print(f'  seed={seed}  acc_a={r["final_acc_a"]:.3f}  acc_b={r["final_acc_b"]:.3f}  '
                  f'surp_a={r["final_surprise_a"]:.4f}  surp_b={r["final_surprise_b"]:.4f}  '
                  f'align_late={al_s}  '
                  f'transfer={r["transfer_a"]:.3f}/{r["transfer_b"]:.3f}  '
                  f'({r["train_time"]:.1f}s)')

    # Aggregate
    md = ['# Sprint 11 — Dyadic mutual modeling\n']
    md.append('## Aggregate par condition (mean ± std)\n')
    md.append('| condition | α | acc_a | acc_b | surp_a | surp_b | alignment | transfer_a | transfer_b |')
    md.append('|---|---|---|---|---|---|---|---|---|')

    by_cond = {}
    for r in rows:
        by_cond.setdefault(r['condition'], []).append(r)

    for cond in conditions:
        if cond not in by_cond:
            continue
        runs = by_cond[cond]
        def fmt(key, dec=3):
            vals = [r[key] for r in runs if r.get(key) is not None]
            if not vals:
                return '—'
            return f'{np.mean(vals):.{dec}f}±{np.std(vals):.{dec}f}'
        w = runs[0]['weight']
        md.append(
            f'| **{cond}** | {w} | '
            f'{fmt("final_acc_a")} | {fmt("final_acc_b")} | '
            f'{fmt("final_surprise_a", dec=4)} | {fmt("final_surprise_b", dec=4)} | '
            f'{fmt("mean_alignment_late")} | '
            f'{fmt("transfer_a")} | {fmt("transfer_b")} |'
        )

    md.append('\n## Per-run\n')
    md.append('| cond | seed | acc_a | acc_b | surp_a | surp_b | align | transfer_a | transfer_b |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        al = r["mean_alignment_late"]
        al_s = f'{al:.3f}' if al is not None else '—'
        md.append(
            f'| {r["condition"]} | {r["seed"]} | '
            f'{r["final_acc_a"]:.3f} | {r["final_acc_b"]:.3f} | '
            f'{r["final_surprise_a"]:.4f} | {r["final_surprise_b"]:.4f} | '
            f'{al_s} | '
            f'{r["transfer_a"]:.3f} | {r["transfer_b"]:.3f} |'
        )

    md.append('\n## Verdict — hypothèse dyadique\n')
    if 'none' in by_cond and 'mutual' in by_cond:
        baseline_acc = np.mean([r['final_acc_a'] for r in by_cond['none']]
                               + [r['final_acc_b'] for r in by_cond['none']])
        baseline_transfer = np.mean([r['transfer_a'] for r in by_cond['none']]
                                    + [r['transfer_b'] for r in by_cond['none']])
        mutual_acc = np.mean([r['final_acc_a'] for r in by_cond['mutual']]
                             + [r['final_acc_b'] for r in by_cond['mutual']])
        mutual_transfer = np.mean([r['transfer_a'] for r in by_cond['mutual']]
                                  + [r['transfer_b'] for r in by_cond['mutual']])

        delta_acc = mutual_acc - baseline_acc
        delta_transfer = mutual_transfer - baseline_transfer

        md.append(f'- **Baseline (α=0)** : acc={baseline_acc:.3f}, transfer={baseline_transfer:.3f}')
        md.append(f'- **Mutual (α=1)** : acc={mutual_acc:.3f} (Δ={delta_acc:+.3f}), '
                  f'transfer={mutual_transfer:.3f} (Δ={delta_transfer:+.3f})')
        if delta_transfer > 0.005:
            md.append(f'  → **CONFIRMÉ** : mutual modeling améliore transfer (+{delta_transfer*100:.1f}%)')
        elif delta_transfer < -0.005:
            md.append(f'  → **INVERSÉ** : mutual modeling dégrade transfer')
        else:
            md.append(f'  → **NEUTRE** : mutual modeling n\'affecte pas significativement transfer')

    # Alignment finding
    if 'none' in by_cond and 'mutual' in by_cond:
        n_al = [abs(r['mean_alignment_late']) for r in by_cond['none'] if r['mean_alignment_late'] is not None]
        m_al = [abs(r['mean_alignment_late']) for r in by_cond['mutual'] if r['mean_alignment_late'] is not None]
        if not n_al or not m_al:
            md.append('\n- Pas assez de samples alignment pour comparer')
            (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
            (args.out / 'report.md').write_text('\n'.join(md))
            print('\n' + '=' * 70)
            print('\n'.join(md))
            print(f'\n→ {args.out}')
            return
        none_align = np.mean(n_al)
        mutual_align = np.mean(m_al)
        md.append(f'\n- **Alignment A↔B** : none={none_align:.3f}, mutual={mutual_align:.3f}')
        if mutual_align > none_align + 0.05:
            md.append('  → **États A et B se synchronisent davantage avec mutual modeling**')
        elif mutual_align < none_align - 0.05:
            md.append('  → **Mutual modeling crée plus de divergence (rôles différenciés)**')
        else:
            md.append('  → Pas de synchronisation accrue')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
