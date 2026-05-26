"""
Sprint 10 — real-world transfer : MNIST → CIFAR-10/CIFAR-100.

Amplification Sprint 7 (qui testait FashionMNIST = trop proche).
CIFAR différent : couleur, structure visuelle, classes différentes.
Test si self-coherence (α=2) transfere VRAIMENT à task radicalement différent.

Caveat : input dims différents — pad MNIST 28×28 → 32×32×3 pour matcher CIFAR.
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
from architrace.strange_loop import SelfReferentialVG, StrangeLoopConfig


# Dimension commune : on traitera tout en 3072 (32×32×3) en répliquant MNIST en 3 channels + pad
INPUT_DIM = 3072


def load_mnist_as_cifar_shape(n_train=6000, n_val=1500):
    """MNIST padded 28→32 + dupliqué sur 3 channels → 3072 flat."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.Pad(2),  # 28×28 → 32×32
        transforms.ToTensor(),  # (1, 32, 32)
        transforms.Lambda(lambda x: x.expand(3, -1, -1).reshape(-1)),  # → (3072,)
    ])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def load_cifar10_flat(n_train=4000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    va = datasets.CIFAR10('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def load_cifar100_flat(n_train=4000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    va = datasets.CIFAR100('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def pretrain_mnist(weight: float, seed: int, device: torch.device) -> SelfReferentialVG:
    """Pretrain SelfReferentialVG sur MNIST padded en 3072-D avec α=weight."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_ds, va_ds = load_mnist_as_cifar_shape()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)

    cfg = StrangeLoopConfig(
        epochs=5, lr=1e-3, state_dim=32, hidden=64, init_depth=2,
        observer_channels=(8, 16),
        self_surprise_weight=weight,
        log_every=40,
    )

    model = SelfReferentialVG(
        in_dim=INPUT_DIM, out_dim=10,
        state_dim=cfg.state_dim,
        hidden=cfg.hidden, depth=cfg.init_depth,
        observer_channels=cfg.observer_channels,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    prev_state = model.self_state()

    for ep in range(cfg.epochs):
        for x, y in tr:
            x = x.to(device); y = y.to(device)
            current_state = model.self_state()
            opt.zero_grad()
            task_logits, pred_state = model(x, prev_state)
            task_loss = F.cross_entropy(task_logits, y)
            pred_mean = pred_state.mean(0)
            if pred_mean.shape[0] != current_state.shape[0]:
                m = min(pred_mean.shape[0], current_state.shape[0])
                surp = ((pred_mean[:m] - current_state[:m]) ** 2).mean()
            else:
                surp = ((pred_mean - current_state) ** 2).mean()
            loss = task_loss + weight * surp
            loss.backward()
            opt.step()
            prev_state = current_state

    return model, prev_state


def transfer_probe(
    model: SelfReferentialVG, prev_state: torch.Tensor,
    transfer_train, transfer_val, out_dim: int, device: torch.device,
    epochs: int = 8, lr: float = 5e-3,
) -> float:
    """Linear probe sur transfer task. Trunk frozen, head only train."""
    for p in model.parameters():
        p.requires_grad = False

    last_out = None
    for m in model.trunk.modules():
        if isinstance(m, nn.Linear):
            last_out = m.out_features
    new_head = nn.Linear(last_out, out_dim).to(device)
    opt = torch.optim.AdamW(new_head.parameters(), lr=lr, weight_decay=1e-4)

    prev_state_zero = torch.zeros(model.state_dim, device=device)

    @torch.no_grad()
    def _trunk(x):
        return model.trunk(torch.cat([
            x, prev_state_zero.unsqueeze(0).expand(x.shape[0], -1)
        ], dim=-1))

    for ep in range(epochs):
        for x, y in transfer_train:
            x = x.to(device); y = y.to(device)
            z = _trunk(x)
            logits = new_head(z)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    correct = 0; total = 0
    with torch.no_grad():
        for x, y in transfer_val:
            x = x.to(device); y = y.to(device)
            pred = new_head(_trunk(x)).argmax(-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return correct / max(1, total)


def run_one(weight: float, seed: int, transfer_task: str) -> dict:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    t0 = time.time()
    model, prev_state = pretrain_mnist(weight, seed, device)
    pretrain_time = time.time() - t0

    # MNIST acc (sanity)
    model.eval()
    tr_ds, va_ds = load_mnist_as_cifar_shape()
    va = DataLoader(va_ds, batch_size=256, num_workers=0)
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in va:
            x = x.to(device); y = y.to(device)
            logits, _ = model(x, prev_state)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.shape[0]
    mnist_acc = correct / max(1, total)

    # Transfer
    if transfer_task == 'cifar10':
        tr_t, va_t = load_cifar10_flat()
        out_dim = 10
    elif transfer_task == 'cifar100':
        tr_t, va_t = load_cifar100_flat()
        out_dim = 100
    else:
        raise ValueError(transfer_task)
    tr_t_loader = DataLoader(tr_t, batch_size=128, shuffle=True, num_workers=0)
    va_t_loader = DataLoader(va_t, batch_size=256, num_workers=0)

    t0 = time.time()
    transfer_acc = transfer_probe(model, prev_state, tr_t_loader, va_t_loader, out_dim, device)
    probe_time = time.time() - t0

    return {
        'weight': weight,
        'seed': seed,
        'transfer_task': transfer_task,
        'mnist_acc': mnist_acc,
        'transfer_acc': transfer_acc,
        'pretrain_time': pretrain_time,
        'probe_time': probe_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,1.0,2.0,5.0')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--tasks', type=str, default='cifar10,cifar100')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint10')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w.strip()) for w in args.weights.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    tasks = [t.strip() for t in args.tasks.split(',')]

    print(f'=== Sprint 10 — real-world transfer ===')
    print(f'α ∈ {weights}  seeds {seeds}  transfer tasks {tasks}\n')

    rows = []
    for task in tasks:
        print(f'\n*** Transfer target: {task} ***')
        for w in weights:
            for seed in seeds:
                r = run_one(w, seed, task)
                rows.append(r)
                print(f'  α={w} seed={seed} → mnist={r["mnist_acc"]:.4f}  '
                      f'{task}_transfer={r["transfer_acc"]:.4f}  '
                      f'({r["pretrain_time"]:.1f}s+{r["probe_time"]:.1f}s)')

    md = ['# Sprint 10 — real-world transfer\n']
    md.append('MNIST (padded 32×32×3) pretrain → CIFAR-10 / CIFAR-100 linear probe.\n')

    by = {}
    for r in rows:
        key = (r['transfer_task'], r['weight'])
        by.setdefault(key, []).append(r)

    md.append('## Aggregate\n')
    md.append('| transfer task | α | MNIST acc | transfer acc | Δ vs α=0 |')
    md.append('|---|---|---|---|---|')
    baselines = {}
    for task in tasks:
        if (task, 0.0) in by:
            baselines[task] = float(np.mean([r['transfer_acc'] for r in by[(task, 0.0)]]))
    for task in tasks:
        for w in sorted({k[1] for k in by if k[0] == task}):
            runs = by[(task, w)]
            mn = [r['mnist_acc'] for r in runs]
            tr = [r['transfer_acc'] for r in runs]
            delta = (np.mean(tr) - baselines.get(task, np.mean(tr)))
            md.append(
                f'| {task} | {w} | {np.mean(mn):.4f}±{np.std(mn):.4f} | '
                f'{np.mean(tr):.4f}±{np.std(tr):.4f} | {delta:+.4f} |'
            )

    md.append('\n## Per-run\n')
    md.append('| task | α | seed | MNIST | transfer |')
    md.append('|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["transfer_task"]} | {r["weight"]} | {r["seed"]} | '
            f'{r["mnist_acc"]:.4f} | {r["transfer_acc"]:.4f} |'
        )

    md.append('\n## Verdict\n')
    for task in tasks:
        if task not in baselines:
            continue
        all_w = sorted({k[1] for k in by if k[0] == task})
        accs = {w: float(np.mean([r['transfer_acc'] for r in by[(task, w)]])) for w in all_w}
        best_w = max(accs, key=accs.get)
        delta = accs[best_w] - baselines[task]
        md.append(f'- **{task}** : best α={best_w} → {accs[best_w]:.4f} '
                  f'(baseline α=0: {baselines[task]:.4f}, Δ={delta:+.4f})')
        if delta > 0.005:
            md.append(f'  → **CONFIRMÉ** : self-coherence transfer +{delta*100:.1f}%')
        elif delta < -0.005:
            md.append(f'  → **INVERSÉ** : self-coherence dégrade transfer')
        else:
            md.append(f'  → **NEUTRE**')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
