"""
Sprint 11.1 — test internalisation Mead-style.

Hypothèse : circuit "modéliser autrui" se réutilise comme "modéliser soi-même".
Pré-entraînement mutual modeling devrait accélérer self-modeling subséquent.

Setup :
- Phase 1 (3 epochs) : 2 agents A et B
  - condition `mutual` : A modélise B, B modélise A (α=1)
  - condition `none` : α=0 (pas de modeling)
  - condition `fresh_self` : déjà self-modeling (contrôle baseline)
- Phase 2 (2 epochs) : chaque agent SOLO en self-modeling (α=1)

Mesures Phase 2 :
- Trajectoire self_surprise (descend-elle plus vite chez pré-mutual ?)
- Self_surprise initial (premier batch Phase 2)
- Self_surprise final (dernier batch)
- Steps pour descendre sous seuil convergence (0.05)
"""
from __future__ import annotations
import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


def phase1_mutual(
    agent_a: DyadicAgent, agent_b: DyadicAgent, observer: VGCNNEncoder,
    train_a, train_b, opt: torch.optim.Optimizer, weight: float, epochs: int,
    device: torch.device,
):
    """Phase 1 : mutual modeling (α>0) ou no modeling (α=0)."""
    state_a = _agent_state(agent_a, observer)
    state_b = _agent_state(agent_b, observer)
    iter_b = iter(train_b)
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
    return state_a, state_b


def phase2_self(
    agent: DyadicAgent, observer: VGCNNEncoder,
    train_loader, val_loader,
    opt: torch.optim.Optimizer, weight: float, epochs: int,
    device: torch.device,
) -> dict:
    """Phase 2 SOLO : agent self-models lui-même.

    Use partner_predictor pour predict SA propre state. Input = own previous state.
    """
    prev_state = _agent_state(agent, observer)
    history = {
        'step': [], 'task_loss': [], 'self_surprise': [],
        'self_surprise_smoothed': [],
    }
    surp_buf: deque = deque(maxlen=20)
    step = 0

    for ep in range(epochs):
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            current_state = _agent_state(agent, observer)
            opt.zero_grad()
            logits, pred_state = agent(x, prev_state)  # input prev_state = SOI
            task_loss = F.cross_entropy(logits, y)
            pred_mean = pred_state.mean(0)
            if pred_mean.shape[0] != current_state.shape[0]:
                mm = min(pred_mean.shape[0], current_state.shape[0])
                surp = ((pred_mean[:mm] - current_state[:mm]) ** 2).mean()
            else:
                surp = ((pred_mean - current_state) ** 2).mean()
            loss = task_loss + weight * surp
            loss.backward()
            opt.step()
            prev_state = current_state
            surp_buf.append(surp.item())

            history['step'].append(step)
            history['task_loss'].append(task_loss.item())
            history['self_surprise'].append(surp.item())
            history['self_surprise_smoothed'].append(float(np.mean(surp_buf)))
            step += 1

    # Final eval
    agent.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = agent(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    final_acc = c / max(1, t)
    return {'history': history, 'final_acc': final_acc}


def run_condition(condition: str, seed: int) -> dict:
    """3 conditions : mutual→self, none→self, fresh_self.

    fresh_self = self-modeling dès Phase 1 (contrôle).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Setup
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
    opt = torch.optim.AdamW(list(agent_a.parameters()) + list(agent_b.parameters()),
                            lr=1e-3, weight_decay=1e-4)

    PHASE1_EPOCHS = 3
    PHASE2_EPOCHS = 2

    t0 = time.time()
    # === Phase 1 ===
    if condition == 'mutual_to_self':
        phase1_mutual(agent_a, agent_b, observer, tr_a, tr_b, opt,
                      weight=1.0, epochs=PHASE1_EPOCHS, device=device)
    elif condition == 'none_to_self':
        phase1_mutual(agent_a, agent_b, observer, tr_a, tr_b, opt,
                      weight=0.0, epochs=PHASE1_EPOCHS, device=device)
    elif condition == 'fresh_self':
        # Phase 1 = déjà self-modeling
        # Pour A : utilise son propre state comme input (not partner's)
        # Implémenté en faisant phase2_self directement pour les deux agents
        opt_a = torch.optim.AdamW(agent_a.parameters(), lr=1e-3, weight_decay=1e-4)
        phase2_self(agent_a, observer, tr_a, va_a, opt_a, weight=1.0,
                    epochs=PHASE1_EPOCHS, device=device)
        opt_b = torch.optim.AdamW(agent_b.parameters(), lr=1e-3, weight_decay=1e-4)
        phase2_self(agent_b, observer, tr_b, va_b, opt_b, weight=1.0,
                    epochs=PHASE1_EPOCHS, device=device)
    else:
        raise ValueError(condition)
    phase1_time = time.time() - t0

    # === Phase 2 : SOLO self-modeling ===
    t0 = time.time()
    # Optimizers séparés pour la phase solo
    opt_a = torch.optim.AdamW(agent_a.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_b = torch.optim.AdamW(agent_b.parameters(), lr=1e-3, weight_decay=1e-4)
    res_a = phase2_self(agent_a, observer, tr_a, va_a, opt_a,
                        weight=1.0, epochs=PHASE2_EPOCHS, device=device)
    res_b = phase2_self(agent_b, observer, tr_b, va_b, opt_b,
                        weight=1.0, epochs=PHASE2_EPOCHS, device=device)
    phase2_time = time.time() - t0

    # Analyse trajectoire self_surprise phase 2
    def analyse(history):
        sm = history['self_surprise_smoothed']
        if not sm:
            return {}
        return {
            'phase2_surprise_initial': float(sm[0]),
            'phase2_surprise_at_50steps': float(sm[min(50, len(sm)-1)]),
            'phase2_surprise_final': float(sm[-1]),
            'phase2_steps_to_below_005': next(
                (i for i, v in enumerate(sm) if v < 0.05), len(sm)
            ),
            'phase2_steps_to_below_002': next(
                (i for i, v in enumerate(sm) if v < 0.02), len(sm)
            ),
        }

    return {
        'condition': condition,
        'seed': seed,
        'agent_a': {
            'final_acc': res_a['final_acc'],
            **analyse(res_a['history']),
        },
        'agent_b': {
            'final_acc': res_b['final_acc'],
            **analyse(res_b['history']),
        },
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str,
                    default='none_to_self,mutual_to_self,fresh_self')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint11_1')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 11.1 — Internalisation Mead test ===')
    print(f'Conditions: {conditions}  Seeds: {seeds}')
    print(f'Hypothèse : pré-mutual exapte vers self → self-coherence émerge plus vite\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run_condition(cond, seed)
            rows.append(r)
            sa = r['agent_a']; sb = r['agent_b']
            print(f'  seed={seed}  acc_a={sa["final_acc"]:.3f}/b={sb["final_acc"]:.3f}  '
                  f'surp_init_a={sa.get("phase2_surprise_initial", 0):.4f}  '
                  f'final_a={sa.get("phase2_surprise_final", 0):.4f}  '
                  f'steps_to_0.05_a={sa.get("phase2_steps_to_below_005", "?")}  '
                  f'({r["phase1_time"]:.1f}s+{r["phase2_time"]:.1f}s)')

    # Aggregate
    md = ['# Sprint 11.1 — Internalisation Mead (other → self exaptation)\n']
    md.append('## Hypothèse\n')
    md.append('Pré-entraînement mutual modeling → self-coherence émerge plus vite en Phase 2 '
              'que pré-entraînement no-modeling. Si vrai, preuve directe d\'**exaptation** '
              'du circuit other-modeling vers self-modeling (Mead).\n')

    md.append('## Aggregate (mean ± std sur agents A et B × seeds)\n')
    md.append('| condition | acc | surprise initial Phase 2 | surprise final | steps→0.05 | steps→0.02 |')
    md.append('|---|---|---|---|---|---|')

    by_cond = {}
    for r in rows:
        by_cond.setdefault(r['condition'], []).extend([r['agent_a'], r['agent_b']])

    summary = {}
    for cond in conditions:
        if cond not in by_cond:
            continue
        agents = by_cond[cond]
        def fmt(key, dec=4):
            vals = [a.get(key) for a in agents if a.get(key) is not None]
            if not vals:
                return '—'
            return f'{np.mean(vals):.{dec}f}±{np.std(vals):.{dec}f}'
        summary[cond] = {
            'acc_mean': float(np.mean([a['final_acc'] for a in agents])),
            'surp_init_mean': float(np.mean([a.get('phase2_surprise_initial', 0) for a in agents])),
            'surp_final_mean': float(np.mean([a.get('phase2_surprise_final', 0) for a in agents])),
            'steps_to_005_mean': float(np.mean([a.get('phase2_steps_to_below_005', 999) for a in agents])),
            'steps_to_002_mean': float(np.mean([a.get('phase2_steps_to_below_002', 999) for a in agents])),
        }
        md.append(
            f'| **{cond}** | {fmt("final_acc", dec=3)} | {fmt("phase2_surprise_initial")} | '
            f'{fmt("phase2_surprise_final")} | {fmt("phase2_steps_to_below_005", dec=1)} | '
            f'{fmt("phase2_steps_to_below_002", dec=1)} |'
        )

    md.append('\n## Per-run\n')
    md.append('| condition | seed | agent | acc | surp_init | surp_50 | surp_final | steps→0.05 |')
    md.append('|---|---|---|---|---|---|---|---|')
    for r in rows:
        for tag, a in [('A', r['agent_a']), ('B', r['agent_b'])]:
            md.append(
                f'| {r["condition"]} | {r["seed"]} | {tag} | {a["final_acc"]:.3f} | '
                f'{a.get("phase2_surprise_initial", 0):.4f} | '
                f'{a.get("phase2_surprise_at_50steps", 0):.4f} | '
                f'{a.get("phase2_surprise_final", 0):.4f} | '
                f'{a.get("phase2_steps_to_below_005", "?")} |'
            )

    md.append('\n## Verdict\n')
    if 'none_to_self' in summary and 'mutual_to_self' in summary:
        n = summary['none_to_self']
        m = summary['mutual_to_self']

        md.append(f'- **Baseline (none→self)** : initial surprise={n["surp_init_mean"]:.4f}, '
                  f'steps→0.05={n["steps_to_005_mean"]:.0f}')
        md.append(f'- **Pré-mutual (mutual→self)** : initial surprise={m["surp_init_mean"]:.4f}, '
                  f'steps→0.05={m["steps_to_005_mean"]:.0f}')

        delta_init = m['surp_init_mean'] - n['surp_init_mean']
        delta_steps = m['steps_to_005_mean'] - n['steps_to_005_mean']

        md.append(f'- **Δ initial surprise** : {delta_init:+.4f} '
                  f'(négatif = mutual démarre avec moins de surprise)')
        md.append(f'- **Δ steps→0.05** : {delta_steps:+.1f} '
                  f'(négatif = mutual converge plus vite)')

        if delta_init < -0.001 or delta_steps < -5:
            md.append(f'\n  → **EXAPTATION CONFIRMÉE** : pré-mutual accélère self-coherence. '
                      f'Mead validé : other-modeling circuit se redirige vers self-modeling.')
        elif delta_init > 0.005 or delta_steps > 10:
            md.append(f'\n  → **INTERFERENCE** : pré-mutual nuit à self-modeling subséquent. '
                      f'Mécanismes other vs self ne sont PAS compatibles.')
        else:
            md.append(f'\n  → **NEUTRE** : pré-mutual et pré-none équivalents pour self subséquent. '
                      f'Other-modeling ne s\'internalise pas (contradiction Mead simple), '
                      f'mais ne nuit pas non plus.')

    # Compare avec fresh_self (contrôle plafond)
    if 'fresh_self' in summary:
        f = summary['fresh_self']
        md.append(f'\n- **Plafond (fresh_self continuous)** : initial surprise={f["surp_init_mean"]:.4f}')
        if 'mutual_to_self' in summary:
            m = summary['mutual_to_self']
            md.append(f'- mutual→self atteint surprise initial Phase 2 vs fresh_self : '
                      f'{m["surp_init_mean"]:.4f} vs {f["surp_init_mean"]:.4f}')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows, 'summary': summary}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
