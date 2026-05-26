"""
Sprint 11.3 — test universalité circuit modeling.

Sprint 11.1 : mutual → self exapte (100× speedup)
Sprint 11.2 : self → mutual exapte aussi (85× speedup) — symétrie

Question finale : un système NON-AGENTIQUE comme cible exapte-t-il aussi ?
Test : pré-entraîner agent à modéliser une trajectoire Lorenz, puis switch self.

Conditions :
- none  → self : baseline (rappel S11.1)
- world → self : TEST (Phase 1 = Lorenz modeling, Phase 2 = self)
- mutual → self : rappel S11.1 (autre agent)
- fresh_self : plafond

Si world → self exapte aussi → confirmation circuit modeling UNIVERSEL.
Si non → asymétrie agentique vs non-agentique révélée.
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

# Lorenz generator
_sys.path.insert(0, str(architrace.DYNAMOSCOPE_PATH))
from backend.systems import lorenz  # type: ignore


def load_mnist(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def phase_world_lorenz(agent: DyadicAgent, opt, epochs: int, device, weight: float = 1.0):
    """Phase 1 — agent apprend à prédire Lorenz next state from prev state.

    x = zeros (pas de tâche supervisée).
    prev_state = lorenz current (padded to state_dim).
    target = lorenz next.
    Loss = MSE pred vs target.

    Trunk apprend dynamique non-agentique. État SOI n'intervient pas.
    """
    lz = lorenz(n=2000, dt=0.01)
    coords = np.array(lz['raw_coords'], dtype=np.float32)  # (N, 3)
    # Normalize pour stabilité
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    n = coords.shape[0]
    state_dim = agent.state_dim

    # Pad Lorenz 3D to state_dim
    padded = np.zeros((n, state_dim), dtype=np.float32)
    padded[:, :3] = coords
    padded_t = torch.from_numpy(padded).to(device)

    zero_x = torch.zeros(1, agent.in_dim, device=device)

    for ep in range(epochs):
        # Random shuffle pairs
        perm = np.random.permutation(n - 1)
        for batch_start in range(0, len(perm), 64):
            batch_idx = perm[batch_start:batch_start + 64]
            # Process minibatch
            for i in batch_idx:
                prev_state = padded_t[i]
                target_state = padded_t[i + 1]
                opt.zero_grad()
                _, pred = agent(zero_x, prev_state)
                pred_mean = pred.mean(0)
                # Loss seulement sur dims significatives (premières 3)
                loss = weight * ((pred_mean[:3] - target_state[:3]) ** 2).mean()
                loss.backward()
                opt.step()


def phase_self(agent: DyadicAgent, observer, train_loader, val_loader,
               opt, weight: float, epochs: int, device, track: bool = True) -> dict:
    """Phase self-modeling. Track surprise trajectory."""
    prev_state = _agent_state(agent, observer)
    history = {'step': [], 'self_surprise': [], 'self_surprise_smoothed': []}
    surp_buf = deque(maxlen=20)
    step = 0
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
            if track:
                surp_buf.append(surp.item())
                history['step'].append(step)
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
    return {'history': history, 'final_acc': c / max(1, t)}


def run_condition(condition: str, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    state_dim = 16
    agent = DyadicAgent(784, 10, state_dim=state_dim, hidden=32, depth=2).to(device)
    observer = VGCNNEncoder(channels=(4, 8), input_size=64)

    PHASE1_EPOCHS = 3
    PHASE2_EPOCHS = 2

    t0 = time.time()
    opt_p1 = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    if condition == 'none_to_self':
        # Phase 1 : task seul (pas de modeling)
        prev_state = _agent_state(agent, observer)
        for ep in range(PHASE1_EPOCHS):
            for x, y in tr:
                x = x.to(device); y = y.to(device)
                opt_p1.zero_grad()
                logits, _ = agent(x, prev_state)
                F.cross_entropy(logits, y).backward()
                opt_p1.step()
    elif condition == 'world_to_self':
        # Phase 1 : modeling Lorenz
        phase_world_lorenz(agent, opt_p1, epochs=PHASE1_EPOCHS, device=device)
    elif condition == 'fresh_self':
        # Phase 1 = déjà self-modeling
        phase_self(agent, observer, tr, va, opt_p1,
                   weight=1.0, epochs=PHASE1_EPOCHS, device=device, track=False)
    else:
        raise ValueError(condition)
    phase1_time = time.time() - t0

    # === Phase 2 : self-modeling ===
    t0 = time.time()
    opt_p2 = torch.optim.AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)
    res = phase_self(agent, observer, tr, va, opt_p2,
                     weight=1.0, epochs=PHASE2_EPOCHS, device=device, track=True)
    phase2_time = time.time() - t0

    sm = res['history']['self_surprise_smoothed']
    return {
        'condition': condition,
        'seed': seed,
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
        'final_acc': res['final_acc'],
        'phase2_surprise_initial': float(sm[0]) if sm else None,
        'phase2_surprise_at_50steps': float(sm[min(50, len(sm)-1)]) if sm else None,
        'phase2_surprise_final': float(sm[-1]) if sm else None,
        'phase2_steps_to_below_005': next((i for i, v in enumerate(sm) if v < 0.05), len(sm)),
        'phase2_steps_to_below_002': next((i for i, v in enumerate(sm) if v < 0.02), len(sm)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', type=str,
                    default='none_to_self,world_to_self,fresh_self')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint11_3')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 11.3 — Universalité circuit modeling ===')
    print(f'Conditions: {conditions}  Seeds: {seeds}')
    print(f'Test : world (Lorenz) → self exapte-t-il comme mutual → self ?\n')

    rows = []
    for cond in conditions:
        print(f'--- {cond} ---')
        for seed in seeds:
            r = run_condition(cond, seed)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.3f}  '
                  f'surp_init={r["phase2_surprise_initial"]:.4f}  '
                  f'final={r["phase2_surprise_final"]:.4f}  '
                  f'steps→0.05={r["phase2_steps_to_below_005"]}  '
                  f'({r["phase1_time"]:.1f}s+{r["phase2_time"]:.1f}s)')

    by_cond = {}
    for r in rows:
        by_cond.setdefault(r['condition'], []).append(r)

    md = ['# Sprint 11.3 — Universalité circuit modeling\n']
    md.append('## Question\n')
    md.append('Sprint 11.1 (mutual→self) et 11.2 (self→mutual) montrent symétrie : '
              '~100× speedup. Mais c\'est entre agents (entities qui apprennent).\n')
    md.append('Si on remplace partenaire par système non-agentique (Lorenz attracteur), '
              'le circuit modeling exapte-t-il toujours vers self ?\n')

    md.append('## Aggregate\n')
    md.append('| condition | acc | surprise initial Phase 2 | final | steps→0.05 | steps→0.02 |')
    md.append('|---|---|---|---|---|---|')

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
        summary[cond] = {
            'acc_mean': float(np.mean([r['final_acc'] for r in runs])),
            'surp_init_mean': float(np.mean([r['phase2_surprise_initial'] for r in runs if r['phase2_surprise_initial'] is not None])),
            'surp_final_mean': float(np.mean([r['phase2_surprise_final'] for r in runs if r['phase2_surprise_final'] is not None])),
            'steps_005_mean': float(np.mean([r['phase2_steps_to_below_005'] for r in runs])),
        }
        md.append(
            f'| **{cond}** | {fmt("final_acc", dec=3)} | {fmt("phase2_surprise_initial")} | '
            f'{fmt("phase2_surprise_final")} | {fmt("phase2_steps_to_below_005", dec=1)} | '
            f'{fmt("phase2_steps_to_below_002", dec=1)} |'
        )

    md.append('\n## Per-run\n')
    md.append('| condition | seed | acc | surp_init | surp_final | steps→0.05 |')
    md.append('|---|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["condition"]} | {r["seed"]} | {r["final_acc"]:.3f} | '
            f'{r["phase2_surprise_initial"]:.4f} | '
            f'{r["phase2_surprise_final"]:.4f} | '
            f'{r["phase2_steps_to_below_005"]} |'
        )

    md.append('\n## Verdict\n')
    if 'none_to_self' in summary and 'world_to_self' in summary:
        n = summary['none_to_self']
        w = summary['world_to_self']
        ratio = n['surp_init_mean'] / max(w['surp_init_mean'], 1e-9)

        md.append(f'- **none → self** (baseline) : surp_init = {n["surp_init_mean"]:.4f}, '
                  f'steps→0.05 = {n["steps_005_mean"]:.1f}')
        md.append(f'- **world (Lorenz) → self** : surp_init = {w["surp_init_mean"]:.4f}, '
                  f'steps→0.05 = {w["steps_005_mean"]:.1f}')
        md.append(f'- **Ratio speedup** : {ratio:.1f}×')

        if 'fresh_self' in summary:
            f = summary['fresh_self']
            md.append(f'- **Plafond fresh_self** : surp_init = {f["surp_init_mean"]:.4f}')

        # Compare avec S11.1 (mutual→self ratio 100×) et S11.2 (self→mutual 85×)
        md.append(f'\n### Comparaison historique')
        md.append(f'- Sprint 11.1 mutual → self : ratio ~100×')
        md.append(f'- Sprint 11.2 self → mutual : ratio ~85×')
        md.append(f'- Sprint 11.3 world → self  : ratio **{ratio:.1f}×**')

        if ratio > 30:
            md.append(f'\n  → **CONFIRMÉ UNIVERSEL** : modeling Lorenz transfère à self-modeling. '
                      f'Circuit modeling est vraiment générique. Self ≡ Other ≡ World au niveau substrat.')
        elif ratio > 5:
            md.append(f'\n  → **TRANSFER PARTIEL** : world modeling aide mais moins que mutual/self. '
                      f'Pas de circuit complètement universel. Agentique vs non-agentique = différence.')
        else:
            md.append(f'\n  → **PAS DE TRANSFER** : world modeling ne transfère pas à self. '
                      f'Self/other (agentique) et world (non-agentique) = circuits DIFFÉRENTS. '
                      f'Mead/Vygotsky réhabilité au sens strict : c\'est bien spécifiquement '
                      f'le modeling AGENTS qui exapte vers self-modeling.')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows, 'summary': summary}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
