"""
Sprint 5B — bench visual emergence trigger vs direct emergence.

Compare :
- direct_emergence (Sprint 2)  : φ_id sur activations latentes mean batch
- visual_emergence (Sprint 5B) : φ_id sur encodage VG-CNN de render(arch state)

Hypothèse : signal visuel = même qualité (capture lois dynamiques architecture)
mais médié par image space. Si oui, validation que :
1. Architecture state contient assez d'info pour analyse dynamique
2. VG-CNN observer suffisant (random-init)
3. Open path Sprint 5C (self-référence)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import architrace  # noqa
from architrace.bridge import DynamoscopeClient
from architrace.encoder_vg import VGCNNEncoder
from architrace.train_with_trigger import train_with_trigger
from architrace.triggers import (
    EmergenceTrigger, LossPlateauTrigger,
)
from architrace.visual_trigger import VisualEmergenceTrigger
from verifygrow_nn import VerifyGrowNet  # type: ignore


def load_mnist(n_train=6000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def build_trigger(name: str, client, observer=None):
    if name == 'loss_plateau':
        return LossPlateauTrigger(window=30, eps=3e-2, cooldown=30)
    if name == 'direct_emergence':
        t = EmergenceTrigger(window=35, phi_drop_threshold=-0.01,
                             ema_alpha=0.4, cooldown=30, client=client)
        t.eval_every = 15
        return t
    if name == 'visual_emergence':
        return VisualEmergenceTrigger(
            observer_encoder=observer,
            window=35, phi_drop_threshold=-0.01,
            ema_alpha=0.4, cooldown=30,
            capture_every=3, eval_every=15,
            image_size=64, client=client,
        )
    raise ValueError(name)


def run_one(trigger_name: str, seed: int, client, observer):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = VerifyGrowNet(in_dim=784, out_dim=10, init_hidden=20, init_depth=2)
    trigger = build_trigger(trigger_name, client, observer)

    t0 = time.time()
    res = train_with_trigger(
        model, tr, va, trigger,
        epochs=10, grow_k=12, max_hidden=128, log_every=15,
        distill_steps=20, verify_extra=10,
        verbose=False,
    )
    elapsed = time.time() - t0

    visual_phi_history = []
    if hasattr(trigger, 'phi_history'):
        visual_phi_history = list(trigger.phi_history)

    return {
        'trigger': trigger_name,
        'seed': seed,
        'elapsed': elapsed,
        'final_acc': res['final_val_acc'],
        'final_params': res['final_params'],
        'final_topology': res['final_topology'],
        'n_fires': len(res['trigger_log']),
        'first_fire': res['trigger_log'][0]['step'] if res['trigger_log'] else None,
        'accepted': res['accepted'],
        'rejected': res['rejected'],
        'phi_history_len': len(visual_phi_history),
        'phi_history_min': min(visual_phi_history) if visual_phi_history else None,
        'phi_history_max': max(visual_phi_history) if visual_phi_history else None,
        'trigger_reasons': [t['reason'][:60] for t in res['trigger_log']],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--triggers', type=str,
                    default='loss_plateau,direct_emergence,visual_emergence')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint5b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    triggers = [t.strip() for t in args.triggers.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 5B — visual emergence trigger ===')
    print(f'Triggers: {triggers}  Seeds: {seeds}')

    client = DynamoscopeClient()
    print(f'Dynamoscope: {client.health()["device"]}')

    # Observer encoder (VG-CNN tiny, partagé entre seeds pour cohérence)
    observer = VGCNNEncoder(channels=(8, 16, 32))
    print(f'Visual observer: VG-CNN tiny, {observer.num_params()} params, '
          f'latent_dim={observer.latent_dim}')

    rows = []
    for trig in triggers:
        for seed in seeds:
            print(f'\n[{trig} seed={seed}] ...')
            r = run_one(trig, seed, client, observer)
            rows.append(r)
            print(f'  acc={r["final_acc"]:.3f}  params={r["final_params"]}  '
                  f'topo={r["final_topology"]}  fires={r["n_fires"]}  '
                  f'first={r["first_fire"]}  acc={r["accepted"]} rej={r["rejected"]}')
            if r['phi_history_len']:
                print(f'  phi range: [{r["phi_history_min"]:.3f}, {r["phi_history_max"]:.3f}]  '
                      f'({r["phi_history_len"]} samples)')

    # Aggregate
    summary = {}
    for r in rows:
        n = r['trigger']
        summary.setdefault(n, []).append(r)

    md = ['# Sprint 5B — Visual emergence trigger\n']
    md.append('## Summary (mean ± std)\n')
    md.append('| trigger | acc | params | n_fires | first_fire | accept/reject |')
    md.append('|---|---|---|---|---|---|')
    for name, runs in summary.items():
        accs = [r['final_acc'] for r in runs]
        params = [r['final_params'] for r in runs]
        fires = [r['n_fires'] for r in runs]
        firsts = [r['first_fire'] for r in runs if r['first_fire']]
        accepts = [r['accepted'] for r in runs]
        rejects = [r['rejected'] for r in runs]
        md.append(
            f'| **{name}** | '
            f'{np.mean(accs):.3f}±{np.std(accs):.3f} | '
            f'{np.mean(params):.0f}±{np.std(params):.0f} | '
            f'{np.mean(fires):.1f}±{np.std(fires):.1f} | '
            f'{(np.mean(firsts) if firsts else float("nan")):.0f} | '
            f'{np.mean(accepts):.1f}/{np.mean(rejects):.1f} |'
        )

    md.append('\n## Per-run\n')
    md.append('| trigger | seed | acc | params | topo | fires | first | phi_range |')
    md.append('|---|---|---|---|---|---|---|---|')
    for r in rows:
        phi_r = (f'[{r["phi_history_min"]:.3f}, {r["phi_history_max"]:.3f}]'
                 if r['phi_history_len'] else '—')
        md.append(
            f'| {r["trigger"]} | {r["seed"]} | '
            f'{r["final_acc"]:.3f} | {r["final_params"]} | '
            f'{r["final_topology"]} | {r["n_fires"]} | '
            f'{r["first_fire"] or "—"} | {phi_r} |'
        )

    (args.out / 'bench_results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'bench_report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
