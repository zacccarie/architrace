"""
Sprint 7 — self-coherence transfer learning.

Hypothèse robuste à tester :
    self-prediction loss → features plus généralisables car force structures
    inter-couches utiles indépendamment du task signal.

Protocole :
1. Train SelfReferentialVG sur MNIST × {α=0, α=1}
2. Freeze TRUNK uniquement (state_predictor + observer pas concernés)
3. Replace task_head par linéaire frais
4. Train head sur FashionMNIST avec trunk frozen
5. Compare transfer acc

Si α=1 transfère mieux → self-modeling crée représentation transferable.
Si égal → self-modeling = regularization task-specific.
Si α=1 transfère moins bien → self-modeling = overfit à dynamiques weights MNIST.
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
from architrace.strange_loop import (
    SelfReferentialVG, StrangeLoopConfig, train_strange_loop,
)


def load_mnist(n_train=6000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def load_fashion(n_train=4000, n_val=1500):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.FashionMNIST('./data', train=True, download=True, transform=tf)
    va = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def linear_probe(
    pretrained_model: SelfReferentialVG,
    transfer_train_loader, transfer_val_loader,
    out_dim: int = 10,
    epochs: int = 10,
    lr: float = 5e-3,
    device: torch.device = torch.device('cpu'),
) -> dict:
    """Freeze trunk + observer + state_predictor. Train new linear head sur task transfer.

    Note : on doit aussi geler prev_state path. Pour simplifier, on utilise
    UN état prev_state fixe (zeros) pour transfer, isolant le trunk uniquement.
    """
    pretrained_model.eval()
    # Freeze EVERYTHING dans pretrained
    for p in pretrained_model.parameters():
        p.requires_grad = False

    # Get trunk output dim (depends on architecture)
    # Trunk takes (x, state) → output passed to task_head and state_predictor
    # We need to mimic forward without using task_head
    # Find trunk last Linear out_features
    last_linear_out = None
    for m in pretrained_model.trunk.modules():
        if isinstance(m, nn.Linear):
            last_linear_out = m.out_features
    if last_linear_out is None:
        raise ValueError('no Linear in trunk')

    # New head
    new_head = nn.Linear(last_linear_out, out_dim).to(device)
    opt = torch.optim.AdamW(new_head.parameters(), lr=lr, weight_decay=1e-4)

    # Fixed zero prev_state for transfer (isole trunk)
    prev_state_zero = torch.zeros(pretrained_model.state_dim, device=device)

    @torch.no_grad()
    def _trunk_features(x_batch):
        z = pretrained_model.trunk(torch.cat([
            x_batch,
            prev_state_zero.unsqueeze(0).expand(x_batch.shape[0], -1),
        ], dim=-1))
        return z

    # Train head
    for ep in range(epochs):
        for x, y in transfer_train_loader:
            x = x.to(device); y = y.to(device)
            z = _trunk_features(x)
            logits = new_head(z)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Eval
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in transfer_val_loader:
            x = x.to(device); y = y.to(device)
            z = _trunk_features(x)
            pred = new_head(z).argmax(-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]

    return {
        'transfer_acc': correct / max(1, total),
        'last_linear_out': last_linear_out,
    }


def run_one(weight: float, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # === Phase 1 : pretrain sur MNIST avec self-coherence weight = α ===
    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    cfg = StrangeLoopConfig(
        epochs=5, lr=1e-3, state_dim=16, hidden=32, init_depth=2,
        observer_channels=(4, 8),
        self_surprise_weight=weight,
        log_every=30,
    )
    t0 = time.time()
    pre = train_strange_loop(tr, va, in_dim=784, out_dim=10, cfg=cfg, verbose=False)
    pretrain_time = time.time() - t0

    # === Phase 2 : transfer FashionMNIST ===
    # Recreate model architecture to load — actually train_strange_loop returns ephemeral
    # Need to keep model. Let me recompute via re-training (but same seed → same final).
    # Actually train_strange_loop doesn't return model — refactor needed. Re-implement here.
    # SHORTCUT : retrain just to get model handle. Sub-optimal but functional.

    # Re-create model exact same as train_strange_loop
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SelfReferentialVG(
        in_dim=784, out_dim=10,
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

    # MNIST val acc post-train
    model.eval()
    mnist_correct = 0; mnist_total = 0
    with torch.no_grad():
        for x, y in va:
            x = x.to(device); y = y.to(device)
            logits, _ = model(x, prev_state)
            mnist_correct += (logits.argmax(-1) == y).sum().item()
            mnist_total += y.shape[0]
    mnist_acc = mnist_correct / max(1, mnist_total)

    # === Transfer FashionMNIST ===
    tr_f, va_f = load_fashion()
    tr_floader = DataLoader(tr_f, batch_size=128, shuffle=True, num_workers=0)
    va_floader = DataLoader(va_f, batch_size=256, num_workers=0)

    t0 = time.time()
    probe_res = linear_probe(
        model, tr_floader, va_floader,
        out_dim=10, epochs=8, lr=5e-3, device=device,
    )
    probe_time = time.time() - t0

    return {
        'weight': weight,
        'seed': seed,
        'mnist_acc': mnist_acc,
        'fashion_transfer_acc': probe_res['transfer_acc'],
        'trunk_out_dim': probe_res['last_linear_out'],
        'pretrain_time': pretrain_time,
        'probe_time': probe_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.1,0.5,1.0,2.0')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint7')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w.strip()) for w in args.weights.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print(f'=== Sprint 7 — self-coherence transfer learning ===')
    print(f'Pretrain MNIST × α={weights} × seed={seeds}, transfer FashionMNIST\n')

    rows = []
    for w in weights:
        print(f'\n--- α={w} ---')
        for seed in seeds:
            r = run_one(w, seed)
            rows.append(r)
            print(f'  seed={seed}  mnist={r["mnist_acc"]:.4f}  '
                  f'fashion_transfer={r["fashion_transfer_acc"]:.4f}  '
                  f'(pretrain {r["pretrain_time"]:.1f}s + probe {r["probe_time"]:.1f}s)')

    # Aggregate by weight
    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 7 — self-coherence transfer learning\n']
    md.append('Pretrain MNIST avec self_surprise weight α, freeze trunk, '
              'linear probe FashionMNIST.\n')
    md.append('## Aggregate\n')
    md.append('| α | n_seeds | MNIST acc | Fashion transfer acc | Δ vs α=0 |')
    md.append('|---|---|---|---|---|')

    zero_transfer = None
    if 0.0 in by_w:
        zero_transfer = np.mean([r['fashion_transfer_acc'] for r in by_w[0.0]])

    summary = {}
    for w in sorted(by_w.keys()):
        runs = by_w[w]
        mn = [r['mnist_acc'] for r in runs]
        ft = [r['fashion_transfer_acc'] for r in runs]
        delta = (np.mean(ft) - zero_transfer) if zero_transfer is not None else 0
        summary[w] = {
            'mnist_mean': float(np.mean(mn)), 'mnist_std': float(np.std(mn)),
            'fashion_mean': float(np.mean(ft)), 'fashion_std': float(np.std(ft)),
            'delta_vs_zero': float(delta),
        }
        md.append(
            f'| {w} | {len(runs)} | '
            f'{np.mean(mn):.4f}±{np.std(mn):.4f} | '
            f'{np.mean(ft):.4f}±{np.std(ft):.4f} | '
            f'{delta:+.4f} |'
        )

    md.append('\n## Per-run\n')
    md.append('| α | seed | MNIST | Fashion transfer |')
    md.append('|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["weight"]} | {r["seed"]} | '
            f'{r["mnist_acc"]:.4f} | {r["fashion_transfer_acc"]:.4f} |'
        )

    md.append('\n## Verdict\n')
    if zero_transfer is not None:
        best_w = max(summary, key=lambda w: summary[w]['fashion_mean'])
        best_acc = summary[best_w]['fashion_mean']
        delta = best_acc - zero_transfer
        md.append(f'- **Baseline α=0** : transfer = {zero_transfer:.4f}')
        md.append(f'- **Best α={best_w}** : transfer = {best_acc:.4f} (Δ={delta:+.4f})')
        if delta > 0.005:
            md.append(f'  → **CONFIRMÉ** : self-coherence améliore transfer (+{delta*100:.1f}%)')
        elif delta < -0.005:
            md.append(f'  → **INVERSÉ** : self-coherence DÉGRADE transfer')
        else:
            md.append(f'  → **ÉQUIVALENT** : self-coherence n\'affecte pas transfer')

    (args.out / 'transfer_results.json').write_text(
        json.dumps({'rows': rows, 'summary': summary}, default=float, indent=2))
    (args.out / 'transfer_report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
