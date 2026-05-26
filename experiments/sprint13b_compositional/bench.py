"""
Sprint 13B — pre-training compositionnel.

Hypothèse : si circuit modeling universel (S11.3), alors pre-training sur
MULTIPLE cibles (world + self + other) devrait empiler les bénéfices.

OU au contraire : équivalent à single (substrate déjà polyvalent).

Conditions Phase 1 (5 epochs) puis Phase 2 self solo (2 epochs) :
- none → self                 : baseline
- self → self                 : single source (S5C.1)
- world → self                : single source (S11.3)
- world+self → self           : composition 2 sources
- world+self+mutual → self    : composition 3 sources (avec partenaire)

Mesure Phase 2 : self-surprise initial + steps→0.05 + task acc.

Si compositions > single → cumul empirique, circuit absorbe multi-cibles.
Si plateau (= single) → substrate saturé par 1 source, redondance.
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


def _lorenz_padded(state_dim: int, n: int = 1000) -> np.ndarray:
    lz = lorenz(n=n, dt=0.01)
    coords = np.array(lz['raw_coords'], dtype=np.float32)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    padded = np.zeros((n, state_dim), dtype=np.float32)
    padded[:, :3] = coords
    return padded


def phase1_step(agent: DyadicAgent, observer, x, y, opt,
                target_state: torch.Tensor, input_state: torch.Tensor,
                weight: float, device, task_weight: float = 1.0):
    """1 step training avec target_state explicit (peut être self/other/world)."""
    opt.zero_grad()
    logits, pred = agent(x, input_state)
    if task_weight > 0:
        task = F.cross_entropy(logits, y) * task_weight
    else:
        task = torch.tensor(0.0, device=device)
    pred_mean = pred.mean(0)
    if pred_mean.shape[0] != target_state.shape[0]:
        mm = min(pred_mean.shape[0], target_state.shape[0])
        surp = ((pred_mean[:mm] - target_state[:mm]) ** 2).mean()
    else:
        surp = ((pred_mean - target_state) ** 2).mean()
    loss = task + weight * surp
    loss.backward()
    opt.step()


def phase_self_track(agent, observer, train_loader, val_loader, opt,
                     weight: float, epochs: int, device) -> dict:
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
            loss = task + weight * surp
            loss.backward()
            opt.step()
            prev_state = current
            buf.append(surp.item())
            history['surp_smoothed'].append(float(np.mean(buf)))
    # eval
    agent.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = agent(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    return {'final_acc': c / max(1, t), 'history': history}


def run_condition(condition: str, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    # Partner pour mutual condition
    torch.manual_seed(seed + 1000); np.random.seed(seed + 1000)
    tr_partner_ds, _ = load_mnist()
    tr_partner = DataLoader(tr_partner_ds, batch_size=128, shuffle=True, num_workers=0)

    torch.manual_seed(seed); np.random.seed(seed)
    state_dim = 16
    agent = DyadicAgent(784, 10, state_dim=state_dim, hidden=32, depth=2).to(device)
    observer = VGCNNEncoder(channels=(4, 8), input_size=64)

    PHASE1_EPOCHS = 5
    PHASE2_EPOCHS = 2

    # Lorenz data ready
    lorenz_pad = _lorenz_padded(state_dim, n=2000)
    lorenz_t = torch.from_numpy(lorenz_pad).to(device)
    zero_x = torch.zeros(1, agent.in_dim, device=device)

    # Partner pour mutual
    if 'mutual' in condition:
        torch.manual_seed(seed + 1000)
        partner = DyadicAgent(784, 10, state_dim=state_dim, hidden=32, depth=2).to(device)
        torch.manual_seed(seed); np.random.seed(seed)

    opt = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    # === Phase 1 ===
    t0 = time.time()
    prev_state = _agent_state(agent, observer)
    lorenz_idx = 0
    partner_state = None
    if 'mutual' in condition:
        partner_state = _agent_state(partner, observer)
        opt_partner = torch.optim.AdamW(partner.parameters(), lr=1e-3, weight_decay=1e-4)
        partner_iter = iter(tr_partner)

    for ep in range(PHASE1_EPOCHS):
        for x, y in tr:
            x = x.to(device); y = y.to(device)
            current_self = _agent_state(agent, observer)

            if condition == 'none':
                # Task only, no modeling
                opt.zero_grad()
                logits, _ = agent(x, prev_state)
                F.cross_entropy(logits, y).backward()
                opt.step()
            elif condition == 'self':
                # Single source : self
                phase1_step(agent, observer, x, y, opt,
                            target_state=current_self, input_state=prev_state,
                            weight=1.0, device=device)
            elif condition == 'world':
                # Single source : Lorenz
                lz_prev = lorenz_t[lorenz_idx % (len(lorenz_t) - 1)]
                lz_target = lorenz_t[(lorenz_idx + 1) % (len(lorenz_t) - 1)]
                lorenz_idx += 1
                phase1_step(agent, observer, zero_x, torch.zeros(1, dtype=torch.long, device=device),
                            opt, target_state=lz_target, input_state=lz_prev,
                            weight=1.0, device=device, task_weight=0)
                # Also train task without modeling (sinon task non apprise)
                opt.zero_grad()
                logits, _ = agent(x, prev_state)
                F.cross_entropy(logits, y).backward()
                opt.step()
            elif condition == 'world_and_self':
                # Composition : alternate world + self
                if lorenz_idx % 2 == 0:
                    lz_prev = lorenz_t[lorenz_idx % (len(lorenz_t) - 1)]
                    lz_target = lorenz_t[(lorenz_idx + 1) % (len(lorenz_t) - 1)]
                    phase1_step(agent, observer, zero_x, torch.zeros(1, dtype=torch.long, device=device),
                                opt, target_state=lz_target, input_state=lz_prev,
                                weight=1.0, device=device, task_weight=0)
                else:
                    phase1_step(agent, observer, x, y, opt,
                                target_state=current_self, input_state=prev_state,
                                weight=1.0, device=device)
                lorenz_idx += 1
            elif condition == 'world_self_mutual':
                # 3 sources : world, self, mutual
                cycle = lorenz_idx % 3
                if cycle == 0:
                    lz_prev = lorenz_t[lorenz_idx % (len(lorenz_t) - 1)]
                    lz_target = lorenz_t[(lorenz_idx + 1) % (len(lorenz_t) - 1)]
                    phase1_step(agent, observer, zero_x, torch.zeros(1, dtype=torch.long, device=device),
                                opt, target_state=lz_target, input_state=lz_prev,
                                weight=1.0, device=device, task_weight=0)
                elif cycle == 1:
                    phase1_step(agent, observer, x, y, opt,
                                target_state=current_self, input_state=prev_state,
                                weight=1.0, device=device)
                else:
                    # Mutual : agent predicts partner
                    cur_partner = _agent_state(partner, observer)
                    try:
                        px, py = next(partner_iter)
                    except StopIteration:
                        partner_iter = iter(tr_partner)
                        px, py = next(partner_iter)
                    px = px.to(device); py = py.to(device)
                    phase1_step(agent, observer, x, y, opt,
                                target_state=cur_partner, input_state=partner_state,
                                weight=1.0, device=device)
                    # Train partner too
                    phase1_step(partner, observer, px, py, opt_partner,
                                target_state=current_self, input_state=prev_state,
                                weight=1.0, device=device)
                    partner_state = cur_partner
                lorenz_idx += 1
            else:
                raise ValueError(condition)
            prev_state = current_self
    phase1_time = time.time() - t0

    # === Phase 2 : self-modeling solo ===
    opt2 = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    res = phase_self_track(agent, observer, tr, va, opt2,
                            weight=1.0, epochs=PHASE2_EPOCHS, device=device)
    phase2_time = time.time() - t0

    sm = res['history']['surp_smoothed']
    return {
        'condition': condition,
        'seed': seed,
        'final_acc': res['final_acc'],
        'phase2_surp_init': float(sm[0]) if sm else None,
        'phase2_surp_at_50': float(sm[min(50, len(sm) - 1)]) if sm else None,
        'phase2_surp_final': float(sm[-1]) if sm else None,
        'phase2_steps_to_005': next((i for i, v in enumerate(sm) if v < 0.05), len(sm)),
        'phase2_steps_to_002': next((i for i, v in enumerate(sm) if v < 0.02), len(sm)),
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str,
                    default='none,self,world,world_and_self,world_self_mutual')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint13b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 13B — compositional pre-training ===')
    print(f'Conditions: {conditions}  Seeds: {seeds}\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run_condition(cond, seed)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                  f'surp_init={r["phase2_surp_init"]:.4f}  '
                  f'steps→0.05={r["phase2_steps_to_005"]}')

    md = ['# Sprint 13B — Compositional pre-training\n']
    md.append('## Aggregate\n')
    md.append('| condition | acc | surp_init Phase 2 | steps→0.05 | steps→0.02 |')
    md.append('|---|---|---|---|---|')

    by_c = {}
    for r in rows:
        by_c.setdefault(r['condition'], []).append(r)

    for cond in conditions:
        if cond not in by_c:
            continue
        runs = by_c[cond]
        def m(k, dec=4):
            vs = [r[k] for r in runs if r.get(k) is not None]
            return f'{np.mean(vs):.{dec}f}±{np.std(vs):.{dec}f}'
        md.append(
            f'| {cond} | {m("final_acc", 3)} | {m("phase2_surp_init")} | '
            f'{m("phase2_steps_to_005", 1)} | {m("phase2_steps_to_002", 1)} |'
        )

    md.append('\n## Verdict\n')
    inits = {c: float(np.mean([r['phase2_surp_init'] for r in by_c[c]])) for c in by_c}

    md.append(f'- none → self : {inits.get("none", 0):.4f} (baseline)')
    md.append(f'- self → self : {inits.get("self", 0):.4f}')
    md.append(f'- world → self : {inits.get("world", 0):.4f}')
    md.append(f'- world+self → self : {inits.get("world_and_self", 0):.4f}')
    md.append(f'- world+self+mutual → self : {inits.get("world_self_mutual", 0):.4f}')

    single_best = min(inits.get("self", 1), inits.get("world", 1))
    composition_best = min(inits.get("world_and_self", 1), inits.get("world_self_mutual", 1))

    if composition_best < single_best - 0.005:
        md.append(f'\n  → **CUMUL BÉNÉFIQUE** : compositions ({composition_best:.4f}) '
                  f'< single ({single_best:.4f}). Multi-source exapte mieux que single.')
    elif composition_best > single_best + 0.005:
        md.append(f'\n  → **INTERFERENCE** : compositions ({composition_best:.4f}) '
                  f'> single ({single_best:.4f}). Multi-source nuit.')
    else:
        md.append(f'\n  → **SATURATION** : compositions ≈ single ({composition_best:.4f} ≈ {single_best:.4f}). '
                  f'Substrate déjà saturé par 1 source. Redondance Spinoza modes confirmée.')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
