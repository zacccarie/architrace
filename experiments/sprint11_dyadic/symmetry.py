"""
Sprint 11.2 — test symétrie exaptation.

Sprint 11.1 a montré : mutual → self exapte parfaitement (100× speedup).
Mais : self → mutual exapte-t-il aussi ?

Si OUI symétrique : "any dynamical modeling pre-training helps any other"
  → finding banalisé, pas de primauté du social
Si NON : self pré-entraîné ne transfère PAS à mutual → asymétrie
  → primauté évolutionnaire du social confirmée

Conditions :
- none → mutual (baseline) : Phase 1 α=0, Phase 2 mutual α=1
- self → mutual (TEST)      : Phase 1 self α=1, Phase 2 mutual α=1
- fresh_mutual (plafond)    : mutual continuous

Mesure Phase 2 : mutual_surprise initial + steps→0.05.
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
from architrace.dyadic import DyadicAgent, _agent_state
from architrace.encoder_vg import VGCNNEncoder


def load_mnist(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def phase_self(agent: DyadicAgent, observer: VGCNNEncoder,
               train_loader, opt, weight: float, epochs: int, device):
    """Self-modeling solo : agent prédit son propre état."""
    prev_state = _agent_state(agent, observer)
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
            loss = task + weight * surp
            loss.backward()
            opt.step()
            prev_state = current
    return prev_state


def phase_mutual(agent_a, agent_b, observer, train_a, train_b, opt,
                 weight: float, epochs: int, device, track: bool = False):
    """Mutual modeling : A modélise B, B modélise A. Track histoire si demandé."""
    state_a = _agent_state(agent_a, observer)
    state_b = _agent_state(agent_b, observer)
    iter_b = iter(train_b)
    history = {'step': [], 'surp_a': [], 'surp_b': [], 'surp_smoothed': []}
    surp_buf = deque(maxlen=20)
    step = 0
    for ep in range(epochs):
        for x_a, y_a in train_a:
            try:
                x_b, y_b = next(iter_b)
            except StopIteration:
                iter_b = iter(train_b)
                x_b, y_b = next(iter_b)
            x_a = x_a.to(device); y_a = y_a.to(device)
            x_b = x_b.to(device); y_b = y_b.to(device)

            cur_a = _agent_state(agent_a, observer)
            cur_b = _agent_state(agent_b, observer)

            opt.zero_grad()
            logits_a, pred_b = agent_a(x_a, state_b)
            logits_b, pred_a = agent_b(x_b, state_a)
            task_a = F.cross_entropy(logits_a, y_a)
            task_b = F.cross_entropy(logits_b, y_b)

            def m(p, q):
                if p.shape[0] != q.shape[0]:
                    mm = min(p.shape[0], q.shape[0])
                    return ((p[:mm] - q[:mm]) ** 2).mean()
                return ((p - q) ** 2).mean()

            surp_a = m(pred_b.mean(0), cur_b)
            surp_b = m(pred_a.mean(0), cur_a)
            loss = task_a + task_b + weight * (surp_a + surp_b)
            loss.backward()
            opt.step()
            state_a = cur_a
            state_b = cur_b

            if track:
                avg_surp = (surp_a.item() + surp_b.item()) / 2
                surp_buf.append(avg_surp)
                history['step'].append(step)
                history['surp_a'].append(surp_a.item())
                history['surp_b'].append(surp_b.item())
                history['surp_smoothed'].append(float(np.mean(surp_buf)))
            step += 1
    return state_a, state_b, history


def run_condition(condition: str, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_a_ds, va_a_ds = load_mnist()
    torch.manual_seed(seed + 1000); np.random.seed(seed + 1000)
    tr_b_ds, va_b_ds = load_mnist()
    tr_a = DataLoader(tr_a_ds, batch_size=128, shuffle=True, num_workers=0)
    tr_b = DataLoader(tr_b_ds, batch_size=128, shuffle=True, num_workers=0)
    va_a = DataLoader(va_a_ds, batch_size=256, num_workers=0)
    va_b = DataLoader(va_b_ds, batch_size=256, num_workers=0)

    torch.manual_seed(seed); np.random.seed(seed)
    state_dim = 16
    agent_a = DyadicAgent(784, 10, state_dim=state_dim, hidden=32, depth=2).to(device)
    agent_b = DyadicAgent(784, 10, state_dim=state_dim, hidden=32, depth=2).to(device)
    observer = VGCNNEncoder(channels=(4, 8), input_size=64)

    PHASE1_EPOCHS = 3
    PHASE2_EPOCHS = 2

    t0 = time.time()
    # === Phase 1 ===
    if condition == 'self_to_mutual':
        # Phase 1 : chaque agent en self-modeling SOLO α=1
        opt_a = torch.optim.AdamW(agent_a.parameters(), lr=1e-3, weight_decay=1e-4)
        phase_self(agent_a, observer, tr_a, opt_a, weight=1.0,
                   epochs=PHASE1_EPOCHS, device=device)
        opt_b = torch.optim.AdamW(agent_b.parameters(), lr=1e-3, weight_decay=1e-4)
        phase_self(agent_b, observer, tr_b, opt_b, weight=1.0,
                   epochs=PHASE1_EPOCHS, device=device)
    elif condition == 'none_to_mutual':
        # Phase 1 : training α=0 mutual structure (pas de modeling pressure)
        opt_joint = torch.optim.AdamW(
            list(agent_a.parameters()) + list(agent_b.parameters()),
            lr=1e-3, weight_decay=1e-4,
        )
        phase_mutual(agent_a, agent_b, observer, tr_a, tr_b, opt_joint,
                     weight=0.0, epochs=PHASE1_EPOCHS, device=device)
    elif condition == 'fresh_mutual':
        # Phase 1 : déjà mutual α=1
        opt_joint = torch.optim.AdamW(
            list(agent_a.parameters()) + list(agent_b.parameters()),
            lr=1e-3, weight_decay=1e-4,
        )
        phase_mutual(agent_a, agent_b, observer, tr_a, tr_b, opt_joint,
                     weight=1.0, epochs=PHASE1_EPOCHS, device=device)
    else:
        raise ValueError(condition)
    phase1_time = time.time() - t0

    # === Phase 2 : MUTUAL modeling ===
    t0 = time.time()
    opt_joint_p2 = torch.optim.AdamW(
        list(agent_a.parameters()) + list(agent_b.parameters()),
        lr=1e-3, weight_decay=1e-4,
    )
    _, _, history = phase_mutual(
        agent_a, agent_b, observer, tr_a, tr_b, opt_joint_p2,
        weight=1.0, epochs=PHASE2_EPOCHS, device=device, track=True,
    )
    phase2_time = time.time() - t0

    # Eval final
    def _eval(agent, loader, partner_state):
        agent.eval()
        c = 0; t = 0
        with torch.no_grad():
            for vx, vy in loader:
                vx = vx.to(device); vy = vy.to(device)
                logits, _ = agent(vx, partner_state)
                c += (logits.argmax(-1) == vy).sum().item()
                t += vy.shape[0]
        return c / max(1, t)

    state_a_final = _agent_state(agent_a, observer)
    state_b_final = _agent_state(agent_b, observer)
    acc_a = _eval(agent_a, va_a, state_b_final)
    acc_b = _eval(agent_b, va_b, state_a_final)

    # Analyse Phase 2 trajectory
    sm = history['surp_smoothed']
    return {
        'condition': condition,
        'seed': seed,
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
        'final_acc_a': acc_a,
        'final_acc_b': acc_b,
        'phase2_surprise_initial': float(sm[0]) if sm else None,
        'phase2_surprise_at_50steps': float(sm[min(50, len(sm)-1)]) if sm else None,
        'phase2_surprise_final': float(sm[-1]) if sm else None,
        'phase2_steps_to_below_005': next((i for i, v in enumerate(sm) if v < 0.05), len(sm)),
        'phase2_steps_to_below_002': next((i for i, v in enumerate(sm) if v < 0.02), len(sm)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str,
                    default='none_to_mutual,self_to_mutual,fresh_mutual')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint11_2')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 11.2 — Symétrie exaptation self→mutual ===')
    print(f'Conditions: {conditions}  Seeds: {seeds}')
    print(f'Test : direction inverse de Sprint 11.1\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run_condition(cond, seed)
            rows.append(r)
            print(f'  seed={seed}  acc_a={r["final_acc_a"]:.3f}/b={r["final_acc_b"]:.3f}  '
                  f'mutual_surp_init={r["phase2_surprise_initial"]:.4f}  '
                  f'final={r["phase2_surprise_final"]:.4f}  '
                  f'steps→0.05={r["phase2_steps_to_below_005"]}  '
                  f'({r["phase1_time"]:.1f}s+{r["phase2_time"]:.1f}s)')

    # Aggregate
    md = ['# Sprint 11.2 — Symétrie exaptation self → mutual\n']
    md.append('## Hypothèse\n')
    md.append('Sprint 11.1 : **mutual → self** = exaptation parfaite (100× speedup).\n'
              'Question : direction inverse aussi ?\n')
    md.append('- Si symétrique : "any modeling helps any modeling" (banalisé)\n'
              '- Si asymétrique (self → mutual NE marche PAS) : primauté évolutionnaire sociale\n')

    md.append('## Aggregate (mean ± std sur seeds)\n')
    md.append('| condition | acc moy | mutual_surprise initial Phase 2 | final | steps→0.05 | steps→0.02 |')
    md.append('|---|---|---|---|---|---|')

    by_cond = {}
    for r in rows:
        by_cond.setdefault(r['condition'], []).append(r)

    summary = {}
    for cond in conditions:
        if cond not in by_cond:
            continue
        runs = by_cond[cond]
        def fmt(key, dec=4):
            vals = [r[key] for r in runs if r.get(key) is not None]
            if not vals:
                return '—'
            return f'{np.mean(vals):.{dec}f}±{np.std(vals):.{dec}f}'
        accs = [(r['final_acc_a'] + r['final_acc_b']) / 2 for r in runs]
        summary[cond] = {
            'acc_mean': float(np.mean(accs)),
            'surp_init_mean': float(np.mean([r['phase2_surprise_initial'] for r in runs if r['phase2_surprise_initial'] is not None])),
            'surp_final_mean': float(np.mean([r['phase2_surprise_final'] for r in runs if r['phase2_surprise_final'] is not None])),
            'steps_to_005_mean': float(np.mean([r['phase2_steps_to_below_005'] for r in runs])),
            'steps_to_002_mean': float(np.mean([r['phase2_steps_to_below_002'] for r in runs])),
        }
        md.append(
            f'| **{cond}** | {np.mean(accs):.3f}±{np.std(accs):.3f} | '
            f'{fmt("phase2_surprise_initial")} | '
            f'{fmt("phase2_surprise_final")} | '
            f'{fmt("phase2_steps_to_below_005", dec=1)} | '
            f'{fmt("phase2_steps_to_below_002", dec=1)} |'
        )

    md.append('\n## Per-run\n')
    md.append('| condition | seed | acc_a | acc_b | mutual_surp_init | final | steps→0.05 |')
    md.append('|---|---|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["condition"]} | {r["seed"]} | '
            f'{r["final_acc_a"]:.3f} | {r["final_acc_b"]:.3f} | '
            f'{r["phase2_surprise_initial"]:.4f} | '
            f'{r["phase2_surprise_final"]:.4f} | '
            f'{r["phase2_steps_to_below_005"]} |'
        )

    md.append('\n## Verdict — symétrie ou asymétrie ?\n')
    if 'none_to_mutual' in summary and 'self_to_mutual' in summary:
        n = summary['none_to_mutual']
        s = summary['self_to_mutual']

        md.append(f'- **none → mutual (baseline)** : surprise initial = {n["surp_init_mean"]:.4f}, '
                  f'steps→0.05 = {n["steps_to_005_mean"]:.1f}')
        md.append(f'- **self → mutual (test)** : surprise initial = {s["surp_init_mean"]:.4f}, '
                  f'steps→0.05 = {s["steps_to_005_mean"]:.1f}')

        delta_init = s['surp_init_mean'] - n['surp_init_mean']
        delta_steps = s['steps_to_005_mean'] - n['steps_to_005_mean']
        md.append(f'- **Δ initial surprise** (self - none) : {delta_init:+.4f}')
        md.append(f'- **Δ steps→0.05** (self - none) : {delta_steps:+.1f}')

        # Compare avec Sprint 11.1
        md.append('\n### Comparaison avec Sprint 11.1 (mutual → self)\n')
        md.append('Sprint 11.1 : none→self surprise initial = 1.496 ; mutual→self = 0.015 → ratio 100×')

        if 'fresh_mutual' in summary:
            f = summary['fresh_mutual']
            ratio_inverse = n['surp_init_mean'] / max(s['surp_init_mean'], 1e-9)
            md.append(f'\n- **Plafond (fresh_mutual continuous)** : surprise initial = {f["surp_init_mean"]:.4f}')
            md.append(f'- **Ratio none→mutual / self→mutual** : {ratio_inverse:.1f}× '
                      f'(comparable au 100× de Sprint 11.1 ?)')

            if ratio_inverse > 10:
                md.append(f'\n  → **SYMÉTRIE** : self→mutual accélère autant que mutual→self. '
                          f'Any modeling helps any modeling. Finding banalisé : '
                          f'pas de primauté du social.')
            elif ratio_inverse < 2:
                md.append(f'\n  → **ASYMÉTRIE FORTE** : self→mutual N\'accélère PAS mutual. '
                          f'**Primauté évolutionnaire du social confirmée** : '
                          f'le circuit other-modeling produit self-coherence (Sprint 11.1) '
                          f'mais l\'inverse n\'est PAS vrai. Direction matters.')
            else:
                md.append(f'\n  → **ASYMÉTRIE PARTIELLE** : self→mutual aide mais moins que mutual→self. '
                          f'Direction matters mais avec quelque transfert.')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows, 'summary': summary}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
