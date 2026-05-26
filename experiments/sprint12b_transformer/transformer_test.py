"""
Sprint 12B — Cross-architecture validation : Transformer + self-coherence.

Test si finding Sprint 5C.1 (α=1 = sweet spot self-coherence) tient sur
architecture différente. MLP → Tiny Transformer.
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
from architrace.encoder_vg import VGCNNEncoder
from architrace.arch_to_image import render_arch_state


class TinyTransformerAgent(nn.Module):
    """Mini-transformer pour MNIST flattened + self-state token.

    Input : x (B, 784) reshape en (B, 49, 16) tokens 16-dim.
    Self-state ajouté comme token supplémentaire (1, state_dim).
    Sortie : logits classification + pred next self-state.
    """
    def __init__(self, n_tokens=49, token_dim=16, state_dim=16,
                 n_heads=4, n_layers=2, n_classes=10):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.state_dim = state_dim
        # Project state to token_dim if mismatch
        self.state_proj = nn.Linear(state_dim, token_dim) if state_dim != token_dim else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads,
            dim_feedforward=token_dim * 2, batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.task_head = nn.Linear(token_dim, n_classes)
        self.state_predictor = nn.Linear(token_dim, state_dim)

    def forward(self, x, prev_state):
        B = x.shape[0]
        # x: (B, 784) → (B, 49, 16)
        tokens = x.view(B, self.n_tokens, self.token_dim)
        # State token
        if prev_state.dim() == 1:
            state_tok = self.state_proj(prev_state).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        else:
            state_tok = self.state_proj(prev_state).unsqueeze(1)
        # Concat state token to sequence
        seq = torch.cat([state_tok, tokens], dim=1)  # (B, 50, 16)
        z = self.encoder(seq)
        # Take state token output for state pred, mean pooling for task
        state_out = z[:, 0]  # (B, token_dim)
        pooled = z[:, 1:].mean(dim=1)
        return self.task_head(pooled), self.state_predictor(state_out)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def load_mnist(n_train=4000, n_val=1000):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('./data', train=True, download=True, transform=tf)
    va = datasets.MNIST('./data', train=False, download=True, transform=tf)
    return Subset(tr, range(n_train)), Subset(va, range(n_val))


def _agent_state(model, observer, device, image_size=64):
    img = render_arch_state(model, image_size=image_size)
    z = observer.encode(img[None, ...])[0]
    z_t = torch.from_numpy(z).float().to(device)
    if z_t.shape[0] < model.state_dim:
        z_t = torch.cat([z_t, torch.zeros(model.state_dim - z_t.shape[0], device=device)])
    elif z_t.shape[0] > model.state_dim:
        z_t = z_t[:model.state_dim]
    return z_t.detach()


def run(weight: float, seed: int) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    tr_ds, va_ds = load_mnist()
    tr = DataLoader(tr_ds, batch_size=128, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, num_workers=0)

    model = TinyTransformerAgent(n_tokens=49, token_dim=16, state_dim=16,
                                  n_heads=4, n_layers=2).to(device)
    observer = VGCNNEncoder(channels=(4, 8), input_size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    prev_state = _agent_state(model, observer, device)

    surprise_history = []
    t0 = time.time()
    for ep in range(5):
        for x, y in tr:
            x = x.to(device); y = y.to(device)
            current = _agent_state(model, observer, device)
            opt.zero_grad()
            logits, pred = model(x, prev_state)
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
            surprise_history.append(surp.item())

    train_time = time.time() - t0

    model.eval()
    c = 0; t = 0
    with torch.no_grad():
        for vx, vy in va:
            vx = vx.to(device); vy = vy.to(device)
            logits, _ = model(vx, prev_state)
            c += (logits.argmax(-1) == vy).sum().item()
            t += vy.shape[0]
    final_acc = c / max(1, t)

    return {
        'weight': weight,
        'seed': seed,
        'final_acc': final_acc,
        'params': model.num_params(),
        'surprise_init': float(np.mean(surprise_history[:10])),
        'surprise_final': float(np.mean(surprise_history[-10:])),
        'train_time': train_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.5,1.0,2.0,5.0')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint12b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    print(f'=== Sprint 12B — Transformer + self-coherence ===')
    print(f'Weights: {weights}  Seeds: {seeds}\n')

    rows = []
    for w in weights:
        print(f'--- α={w} ---')
        for seed in seeds:
            r = run(w, seed)
            rows.append(r)
            print(f'  seed={seed}  acc={r["final_acc"]:.4f}  '
                  f'surp {r["surprise_init"]:.3f}→{r["surprise_final"]:.4f}  '
                  f'({r["train_time"]:.1f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 12B — Transformer architecture validation\n']
    md.append('## Aggregate (mean ± std)\n')
    md.append('| α | acc | surprise final | Δ acc vs α=0 |')
    md.append('|---|---|---|---|')
    zero_acc = None
    if 0.0 in by_w:
        zero_acc = float(np.mean([r['final_acc'] for r in by_w[0.0]]))
    for w in sorted(by_w):
        runs = by_w[w]
        accs = [r['final_acc'] for r in runs]
        sps = [r['surprise_final'] for r in runs]
        delta = (np.mean(accs) - zero_acc) if zero_acc is not None else 0
        md.append(
            f'| {w} | {np.mean(accs):.4f}±{np.std(accs):.4f} | '
            f'{np.mean(sps):.4f}±{np.std(sps):.4f} | {delta:+.4f} |'
        )

    md.append('\n## Per-run\n')
    md.append('| α | seed | acc | params | surp final |')
    md.append('|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["weight"]} | {r["seed"]} | {r["final_acc"]:.4f} | '
            f'{r["params"]} | {r["surprise_final"]:.4f} |'
        )

    md.append('\n## Verdict\n')
    if zero_acc is not None:
        accs_by_w = {w: float(np.mean([r['final_acc'] for r in by_w[w]])) for w in by_w}
        best_w = max(accs_by_w, key=accs_by_w.get)
        delta = accs_by_w[best_w] - zero_acc
        md.append(f'- **Best α = {best_w}** : acc={accs_by_w[best_w]:.4f} (Δ={delta:+.4f})')
        if delta > 0.005:
            md.append(f'  → **CONFIRMÉ sur Transformer** : self-coherence cheap finding réplique')
        else:
            md.append(f'  → **ARCHITECTURE-DÉPENDANT** : finding ne réplique pas sur Transformer')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
