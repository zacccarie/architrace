"""
Sprint 2 — bench triggers : qui déclenche le mieux la croissance VG ?

Compare :
- LossPlateauTrigger    (baseline = comportement VG classique)
- SurpriseTrigger       (rollout error growth)
- EmergenceTrigger      (φ_id drop, via dynamoscope)
- LyapunovLocalTrigger  (Lyapunov local sur latents)
- CompositeTrigger      (vote pondéré loss + emergence + lyap)

Sur MNIST × 3 seeds. Mesure :
- Timing premier trigger fire
- Total triggers fired
- Acceptance rate (verify-then-keep)
- Final val_acc / params
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import architrace  # noqa: path setup
from architrace.bridge import DynamoscopeClient, DynamoscopeError
from architrace.train_with_trigger import train_with_trigger
from architrace.triggers import (
    LossPlateauTrigger, SurpriseTrigger, EmergenceTrigger,
    LyapunovLocalTrigger, CompositeTrigger, TriggerWeight,
)
from verifygrow_nn import VerifyGrowNet  # type: ignore


def load_mnist(n_train=6000, n_val=1500):
    """N_train suffisant pour ~470 steps en 10 epochs (window=35 doit fill)."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def build_trigger(name: str, client=None) -> object:
    # window >= 30 obligatoire pour emergence/lyapunov (dynamoscope min 30 points)
    if name == 'loss_plateau':
        return LossPlateauTrigger(window=30, eps=3e-2, cooldown=30)
    if name == 'surprise':
        return SurpriseTrigger(window=4, growth_threshold=0.005, cooldown=30)
    if name == 'emergence':
        t = EmergenceTrigger(window=35, phi_drop_threshold=-0.01,
                             ema_alpha=0.4, cooldown=30, client=client)
        t.eval_every = 15
        return t
    if name == 'lyapunov':
        t = LyapunovLocalTrigger(window=35, lyap_threshold=0.05,
                                 cooldown=30, client=client)
        t.eval_every = 15
        return t
    if name == 'composite':
        e = EmergenceTrigger(window=35, phi_drop_threshold=-0.01,
                             ema_alpha=0.4, cooldown=10, client=client)
        e.eval_every = 15
        ly = LyapunovLocalTrigger(window=35, lyap_threshold=0.05,
                                  cooldown=10, client=client)
        ly.eval_every = 15
        return CompositeTrigger(
            triggers=[
                TriggerWeight(LossPlateauTrigger(window=30, eps=3e-2, cooldown=10), weight=0.5),
                TriggerWeight(e, weight=1.0),
                TriggerWeight(ly, weight=0.7),
            ],
            threshold=1.0, cooldown=30,
        )
    raise ValueError(f'unknown trigger {name}')


def run_one(trigger_name: str, seed: int, client=None, verbose=False) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = VerifyGrowNet(in_dim=784, out_dim=10, init_hidden=20, init_depth=2)
    trigger = build_trigger(trigger_name, client=client)

    t0 = time.time()
    res = train_with_trigger(
        model, tr, va, trigger,
        epochs=10, grow_k=12, max_hidden=128, log_every=15,
        distill_steps=20, verify_extra=10,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    first_fire = res['trigger_log'][0]['step'] if res['trigger_log'] else None

    return {
        'trigger': trigger_name,
        'seed': seed,
        'elapsed': elapsed,
        'final_acc': res['final_val_acc'],
        'final_params': res['final_params'],
        'final_topology': res['final_topology'],
        'n_fires': len(res['trigger_log']),
        'first_fire_step': first_fire,
        'accepted': res['accepted'],
        'rejected': res['rejected'],
        'frozen': res['frozen'],
        'trigger_log': res['trigger_log'],
        'mutations': [{k: v for k, v in m.items() if k != 'snapshot'} for m in res['mutations']],
    }


def aggregate(rows: list[dict]) -> dict:
    by_trigger: dict[str, list] = {}
    for r in rows:
        by_trigger.setdefault(r['trigger'], []).append(r)

    summary = {}
    for name, runs in by_trigger.items():
        def stat(key):
            vals = [r[key] for r in runs if r.get(key) is not None]
            if not vals:
                return None
            return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

        summary[name] = {
            'n_seeds': len(runs),
            'final_acc': stat('final_acc'),
            'final_params': stat('final_params'),
            'n_fires': stat('n_fires'),
            'first_fire': stat('first_fire_step'),
            'accepted': stat('accepted'),
            'rejected': stat('rejected'),
            'elapsed': stat('elapsed'),
        }
    return summary


def render_md(rows, summary) -> str:
    md = ['# Sprint 2 — Triggers VG bench\n']

    md.append('## Per-trigger summary (mean ± std sur seeds)\n')
    md.append('| trigger | seeds | final_acc | final_params | n_fires | first_fire_step | accepted | rejected |')
    md.append('|---|---|---|---|---|---|---|---|')
    for name, s in summary.items():
        def fmt(stat, dec=3):
            if not stat:
                return '—'
            return f'{stat["mean"]:.{dec}f}±{stat["std"]:.{dec}f}'
        md.append(
            f'| **{name}** | {s["n_seeds"]} | '
            f'{fmt(s["final_acc"])} | '
            f'{fmt(s["final_params"], dec=0)} | '
            f'{fmt(s["n_fires"], dec=1)} | '
            f'{fmt(s["first_fire"], dec=0)} | '
            f'{fmt(s["accepted"], dec=1)} | '
            f'{fmt(s["rejected"], dec=1)} |'
        )

    md.append('\n## Per-run detail\n')
    md.append('| trigger | seed | acc | params | topo | fires | first | acc/rej | reasons (first 2) |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        reasons = '; '.join(t['reason'][:30] for t in r['trigger_log'][:2])
        md.append(
            f'| {r["trigger"]} | {r["seed"]} | '
            f'{r["final_acc"]:.3f} | {r["final_params"]} | '
            f'{r["final_topology"]} | {r["n_fires"]} | '
            f'{r["first_fire_step"] or "—"} | '
            f'{r["accepted"]}/{r["rejected"]} | '
            f'{reasons} |'
        )

    md.append('\n## Verdict\n')
    accs = {n: s['final_acc']['mean'] for n, s in summary.items() if s['final_acc']}
    params = {n: s['final_params']['mean'] for n, s in summary.items() if s['final_params']}
    if accs:
        best_acc = max(accs, key=accs.get)
        md.append(f'- **Meilleur final_acc** : `{best_acc}` ({accs[best_acc]:.3f})')
    if params:
        smallest = min(params, key=params.get)
        md.append(f'- **Plus petit modèle** : `{smallest}` ({params[smallest]:.0f} params)')
    return '\n'.join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--triggers', type=str,
                    default='loss_plateau,surprise,emergence,lyapunov,composite')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint2')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    triggers = [t.strip() for t in args.triggers.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 2 trigger bench ===')
    print(f'Triggers: {triggers}  Seeds: {seeds}')

    needs_client = any(t in {'emergence', 'lyapunov', 'composite'} for t in triggers)
    client = None
    if needs_client:
        try:
            client = DynamoscopeClient()
            client.health()
            print('Dynamoscope: OK')
        except DynamoscopeError as e:
            print(f'WARNING: dynamoscope unavailable, fallback no_client for emergence/lyapunov: {e}')
            client = None

    rows = []
    for trig in triggers:
        for seed in seeds:
            print(f'\n[{trig} seed={seed}] ...')
            r = run_one(trig, seed, client=client, verbose=False)
            rows.append(r)
            print(f'  acc={r["final_acc"]:.3f}  params={r["final_params"]}  '
                  f'topo={r["final_topology"]}  fires={r["n_fires"]}  '
                  f'acc={r["accepted"]} rej={r["rejected"]}  '
                  f'first={r["first_fire_step"]}  ({r["elapsed"]:.1f}s)')

    summary = aggregate(rows)
    md = render_md(rows, summary)

    (args.out / 'bench_results.json').write_text(json.dumps({
        'rows': rows, 'summary': summary,
    }, default=float, indent=2))
    (args.out / 'bench_report.md').write_text(md)

    print('\n' + '=' * 70)
    print(md)
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
