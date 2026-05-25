"""
Sprint 4 — runner : entraîne VG, sauvegarde trace architecture (NxD vectors)
au format injection dynamoscope.

Usage :
    python vg_run_to_trace.py --dataset mnist --seed 42 --epochs 4 --out ./traces/
    python vg_run_to_trace.py --dataset synth --seeds 0,1,2,3,4 --epochs 3

Sortie JSON par run :
    {
        cache_key, run_id, dataset, seed, epochs,
        latents: NxD (D=58),
        var_names: [str],
        tags: [str],          # tick/pre_widen/post_widen/accept/reject/final
        steps: [int],
        mutations: [...],
        meta: {final_acc, final_params, final_topology, ...}
    }
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from arch_snapshot import var_names, snapshot_to_dict
from verifygrow_nn import VerifyGrowNet, TrainConfig, train


DEFAULT_OUT = Path(__file__).parent.parent / "dynamoscope" / "cache" / "vg_traces"


# =============================================================================
# Loaders datasets
# =============================================================================
def _load_mnist(n_train=8000, n_val=1500, root='./data'):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    va = datasets.MNIST(root, train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val)), 784, 10, 'classification'


def _load_cifar10(n_train=6000, n_val=1200, root='./data'):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    va = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val)), 3072, 10, 'classification'


def _load_synth(n=2000, in_dim=32, out_dim=4, seed=0):
    """Tâche synthétique reproductible : classification linéaire bruitée."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(in_dim, out_dim, generator=g)
    x = torch.randn(n, in_dim, generator=g)
    logits = x @ W + 0.3 * torch.randn(n, out_dim, generator=g)
    y = logits.argmax(-1)
    ds = TensorDataset(x, y)
    split = int(n * 0.75)
    return Subset(ds, range(split)), Subset(ds, range(split, n)), in_dim, out_dim, 'classification'


DATASETS = {'mnist': _load_mnist, 'cifar10': _load_cifar10, 'synth': _load_synth}


# =============================================================================
# Run single
# =============================================================================
def run_single(dataset: str, seed: int, epochs: int, snapshot_every: int,
               grow_k: int, init_hidden: int, init_depth: int,
               max_hidden: int, verbose: bool = False) -> dict:
    """Lance 1 run VG + retourne dict trace prêt pour injection."""
    torch.manual_seed(seed)

    if dataset == 'synth':
        tr_ds, va_ds, in_dim, out_dim, task = _load_synth(seed=seed)
    else:
        tr_ds, va_ds, in_dim, out_dim, task = DATASETS[dataset]()

    bs = 64 if dataset == 'synth' else 128
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = VerifyGrowNet(
        in_dim=in_dim, out_dim=out_dim,
        init_hidden=init_hidden, init_depth=init_depth,
    )
    # Config tunée pour DÉCLENCHER des mutations même sur runs courts.
    # Cooldown court + plateau_eps tolérant + plateau_window modéré.
    cfg = TrainConfig(
        epochs=epochs,
        plateau_window=30 if dataset == 'synth' else 50,
        plateau_eps=2e-2,
        cooldown=40 if dataset == 'synth' else 60,
        distill_steps=20 if dataset == 'synth' else 40,
        verify_extra=8,
        val_tol=1e-3,
        max_rejections=4,
        grow_k=grow_k,
        max_hidden=max_hidden,
        log_every=20 if dataset == 'synth' else 50,
        snapshot_every=snapshot_every,
        snapshot_mutations=True,
    )

    t0 = time.time()
    result = train(model, tr_loader, va_loader, cfg, task=task, verbose=verbose)
    elapsed = time.time() - t0

    snapshots = result['snapshots']
    latents = [s.to_vec() for _, s in snapshots]
    tags = [t for t, _ in snapshots]
    steps = [s.step for _, s in snapshots]
    snaps_dict = [snapshot_to_dict(s) for _, s in snapshots]

    run_id = hashlib.sha1(
        f'{dataset}_{seed}_{epochs}_{int(time.time())}'.encode()
    ).hexdigest()[:12]
    cache_key = f'vg_{dataset}_s{seed}_{run_id}'

    return {
        'cache_key': cache_key,
        'run_id': run_id,
        'dataset': dataset,
        'seed': seed,
        'epochs': epochs,
        'elapsed_sec': elapsed,
        'n_snapshots': len(latents),
        'dim': len(latents[0]) if latents else 0,
        'latents': latents,
        'var_names': var_names(),
        'tags': tags,
        'steps': steps,
        'snapshots_full': snaps_dict,
        'mutations': result['mutations'],
        'meta': {
            'final_val_acc': result['final_val_acc'],
            'final_val_loss': result['final_val_loss'],
            'final_params': result['final_params'],
            'final_topology': result['final_topology'],
            'accepted': result['accepted'],
            'rejected': result['rejected'],
            'frozen': result['frozen'],
            'init_hidden': init_hidden,
            'init_depth': init_depth,
            'grow_k': grow_k,
            'max_hidden': max_hidden,
        },
    }


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description='VG → architecture trace pour dynamoscope.')
    ap.add_argument('--dataset', choices=list(DATASETS.keys()), default='synth')
    ap.add_argument('--seeds', type=str, default='0',
                    help='Comma-separated seeds (ex: 0,1,2)')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--snapshot-every', type=int, default=10)
    ap.add_argument('--grow-k', type=int, default=8)
    ap.add_argument('--init-hidden', type=int, default=16)
    ap.add_argument('--init-depth', type=int, default=2)
    ap.add_argument('--max-hidden', type=int, default=128)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    print(f'=== VG → trace · dataset={args.dataset} · seeds={seeds} · epochs={args.epochs} ===')

    written = []
    for seed in seeds:
        print(f'\n[seed {seed}] run...')
        trace = run_single(
            args.dataset, seed, args.epochs, args.snapshot_every,
            args.grow_k, args.init_hidden, args.init_depth, args.max_hidden,
            verbose=args.verbose,
        )
        out_file = args.out / f'{trace["cache_key"]}.json'
        out_file.write_text(json.dumps(trace, default=float))
        written.append(out_file)
        m = trace['meta']
        print(f'[seed {seed}] saved {out_file.name}')
        print(f'  snapshots={trace["n_snapshots"]}  dim={trace["dim"]}  '
              f'elapsed={trace["elapsed_sec"]:.1f}s')
        print(f'  final_topology={m["final_topology"]}  acc={m["final_val_acc"]:.3f}  '
              f'params={m["final_params"]}  '
              f'accepted={m["accepted"]} rejected={m["rejected"]}')

    print(f'\n{len(written)} traces written → {args.out}')


if __name__ == '__main__':
    main()
