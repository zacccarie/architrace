"""
Sprint 6 — réplication findings sur CIFAR-100 (100 classes, task plus dure).

Tests :
1. Sprint 2 reverse trigger : emergence/lyapunov > loss_plateau ?
2. Sprint 5C.1 self_surprise sweep : conscience-cheap finding persiste ?
3. Sprint 5D : observer pousse-t-il quand observed plus complexe ?

CIFAR-100 = 100 classes vs MNIST 10. Devrait forcer plus de mutations,
amplifier différences entre triggers.
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
from architrace.train_with_trigger import train_with_trigger
from architrace.triggers import EmergenceTrigger, LossPlateauTrigger
from architrace.strange_loop import train_strange_loop, StrangeLoopConfig
from verifygrow_nn import VerifyGrowNet  # type: ignore


def load_cifar100(n_train=8000, n_val=2000):
    """CIFAR-100 flattened pour MLP."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # 3072 dim
    ])
    tr = datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    va = datasets.CIFAR100('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


# =============================================================================
# Sprint 2 replication
# =============================================================================
def run_trigger_bench(trigger_name: str, seed: int, client) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_ds, va_ds = load_cifar100()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    if trigger_name == 'loss_plateau':
        trigger = LossPlateauTrigger(window=40, eps=3e-2, cooldown=40)
    elif trigger_name == 'emergence':
        trigger = EmergenceTrigger(window=35, phi_drop_threshold=-0.01,
                                   ema_alpha=0.4, cooldown=40, client=client)
        trigger.eval_every = 20
    else:
        raise ValueError(trigger_name)

    model = VerifyGrowNet(in_dim=3072, out_dim=100, init_hidden=64, init_depth=2)
    t0 = time.time()
    res = train_with_trigger(
        model, tr, va, trigger,
        epochs=8, grow_k=24, max_hidden=256, log_every=20,
        distill_steps=30, verify_extra=15, val_tol=5e-4,
        verbose=False,
    )
    elapsed = time.time() - t0

    return {
        'mode': 'trigger',
        'trigger': trigger_name,
        'seed': seed,
        'elapsed': elapsed,
        'final_acc': res['final_val_acc'],
        'final_params': res['final_params'],
        'final_topology': res['final_topology'],
        'n_fires': len(res['trigger_log']),
        'accepted': res['accepted'],
        'rejected': res['rejected'],
        'first_fire': res['trigger_log'][0]['step'] if res['trigger_log'] else None,
    }


# =============================================================================
# Sprint 5C.1 replication
# =============================================================================
def run_strangeloop(weight: float, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_ds, va_ds = load_cifar100()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    cfg = StrangeLoopConfig(
        epochs=4, lr=1e-3, state_dim=32, hidden=64, init_depth=2,
        observer_channels=(8, 16),
        self_surprise_weight=weight,
        log_every=40,
    )
    t0 = time.time()
    res = train_strange_loop(tr, va, in_dim=3072, out_dim=100, cfg=cfg, verbose=False)
    elapsed = time.time() - t0

    h = res['history']
    surprise = h['self_surprise']

    return {
        'mode': 'strangeloop',
        'weight': weight,
        'seed': seed,
        'elapsed': elapsed,
        'final_acc': res['final_val_acc'],
        'final_params': res['final_params'],
        'task_loss_init': h['task_loss'][0] if h['task_loss'] else None,
        'task_loss_final': h['task_loss'][-1] if h['task_loss'] else None,
        'surprise_init': surprise[0] if surprise else None,
        'surprise_final': surprise[-1] if surprise else None,
        'surprise_max': max(surprise) if surprise else None,
        'state_norm_final': h['state_norm'][-1] if h['state_norm'] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['triggers', 'strangeloop', 'both'], default='both')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--weights', type=str, default='0.0,0.5,1.0,5.0')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint6')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    weights = [float(w.strip()) for w in args.weights.split(',')]

    print(f'=== Sprint 6 — CIFAR-100 robustness ===')

    all_rows = []

    if args.mode in ('triggers', 'both'):
        print(f'\n--- Sprint 2 replication on CIFAR-100 ---')
        client = DynamoscopeClient()
        for trig in ['loss_plateau', 'emergence']:
            for seed in seeds:
                print(f'[{trig} seed={seed}] ...')
                r = run_trigger_bench(trig, seed, client)
                all_rows.append(r)
                print(f'  acc={r["final_acc"]:.4f}  params={r["final_params"]}  '
                      f'fires={r["n_fires"]} acc={r["accepted"]} rej={r["rejected"]}  '
                      f'topo={r["final_topology"]}  ({r["elapsed"]:.1f}s)')

    if args.mode in ('strangeloop', 'both'):
        print(f'\n--- Sprint 5C.1 replication on CIFAR-100 ---')
        for w in weights:
            for seed in seeds:
                print(f'[strangeloop α={w} seed={seed}] ...')
                r = run_strangeloop(w, seed)
                all_rows.append(r)
                print(f'  acc={r["final_acc"]:.4f}  params={r["final_params"]}  '
                      f'surprise {r["surprise_init"]:.3f}→{r["surprise_final"]:.4f}  '
                      f'({r["elapsed"]:.1f}s)')

    # ------- Report -------
    md = ['# Sprint 6 — CIFAR-100 robustness\n']

    # Sprint 2 replication
    trig_rows = [r for r in all_rows if r['mode'] == 'trigger']
    if trig_rows:
        md.append('## Sprint 2 replication\n')
        md.append('| trigger | seed | acc | params | topo | fires | acc/rej |')
        md.append('|---|---|---|---|---|---|---|')
        for r in trig_rows:
            md.append(
                f'| {r["trigger"]} | {r["seed"]} | {r["final_acc"]:.4f} | '
                f'{r["final_params"]} | {r["final_topology"]} | '
                f'{r["n_fires"]} | {r["accepted"]}/{r["rejected"]} |'
            )
        by_trig = {}
        for r in trig_rows:
            by_trig.setdefault(r['trigger'], []).append(r)
        md.append('\n### Aggregate\n')
        md.append('| trigger | acc mean±std | params mean | fires mean | accepted mean |')
        md.append('|---|---|---|---|---|')
        for t, runs in by_trig.items():
            accs = [r['final_acc'] for r in runs]
            params = [r['final_params'] for r in runs]
            fires = [r['n_fires'] for r in runs]
            accepts = [r['accepted'] for r in runs]
            md.append(
                f'| **{t}** | {np.mean(accs):.4f}±{np.std(accs):.4f} | '
                f'{np.mean(params):.0f} | {np.mean(fires):.1f} | {np.mean(accepts):.1f} |'
            )

    # Sprint 5C.1 replication
    sl_rows = [r for r in all_rows if r['mode'] == 'strangeloop']
    if sl_rows:
        md.append('\n## Sprint 5C.1 replication — self_surprise weight sweep\n')
        md.append('| α | seed | acc | surprise L→F | params |')
        md.append('|---|---|---|---|---|')
        for r in sl_rows:
            md.append(
                f'| {r["weight"]} | {r["seed"]} | {r["final_acc"]:.4f} | '
                f'{r["surprise_init"]:.3f}→{r["surprise_final"]:.4f} | {r["final_params"]} |'
            )
        by_w = {}
        for r in sl_rows:
            by_w.setdefault(r['weight'], []).append(r)
        md.append('\n### Aggregate\n')
        md.append('| α | acc mean±std | surprise final mean |')
        md.append('|---|---|---|')
        for w in sorted(by_w):
            runs = by_w[w]
            accs = [r['final_acc'] for r in runs]
            surps = [r['surprise_final'] for r in runs]
            md.append(
                f'| {w} | {np.mean(accs):.4f}±{np.std(accs):.4f} | '
                f'{np.mean(surps):.4f}±{np.std(surps):.4f} |'
            )

    md.append('\n## Verdict\n')

    if trig_rows:
        by_trig = {}
        for r in trig_rows:
            by_trig.setdefault(r['trigger'], []).append(r['final_acc'])
        if 'loss_plateau' in by_trig and 'emergence' in by_trig:
            lp_mean = np.mean(by_trig['loss_plateau'])
            em_mean = np.mean(by_trig['emergence'])
            delta = em_mean - lp_mean
            md.append(f'- **Sprint 2 finding on CIFAR-100** : emergence={em_mean:.4f} vs '
                      f'loss_plateau={lp_mean:.4f} → Δ={delta:+.4f}')
            if delta > 0.005:
                md.append(f'  → **REPLIQUÉ** : dynamoscope trigger bat loss-based ({delta*100:+.1f}%)')
            elif delta < -0.005:
                md.append(f'  → **INVERSÉ** : loss-based bat dynamoscope sur task harder')
            else:
                md.append(f'  → **ÉQUIVALENT** sur task harder')

    if sl_rows:
        by_w = {}
        for r in sl_rows:
            by_w.setdefault(r['weight'], []).append(r['final_acc'])
        if 0.0 in by_w and 1.0 in by_w:
            zero_acc = np.mean(by_w[0.0])
            one_acc = np.mean(by_w[1.0])
            delta = one_acc - zero_acc
            md.append(f'- **Sprint 5C.1 finding on CIFAR-100** : α=0 acc={zero_acc:.4f} vs '
                      f'α=1 acc={one_acc:.4f} → Δ={delta:+.4f}')
            if delta > 0.005:
                md.append(f'  → **REPLIQUÉ** : auto-cohérence aide ({delta*100:+.1f}%)')
            elif delta < -0.005:
                md.append(f'  → **INVERSÉ** : auto-cohérence coûte sur task harder')
            else:
                md.append(f'  → **NEUTRE** : auto-cohérence n\'affecte pas significativement')

    (args.out / 'results.json').write_text(json.dumps({'rows': all_rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
