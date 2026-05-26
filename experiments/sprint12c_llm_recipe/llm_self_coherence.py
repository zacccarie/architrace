"""
Sprint 12C — Recette LLM self-coherence.

Tiny LLM décodeur autoregressif + auxiliary self-coherence loss.

Setup minimaliste :
- TinyLM : embedding + 2-layer Transformer decoder + LM head
- Self-state = mean over sequence des activations dernière couche
- Auxiliary loss : prédire self-state[t] depuis self-state[t-1] via head linéaire
- Compare α=0 vs α=1 sur tâche language modeling sur séquences synthétiques

Question : self-coherence loss améliore-t-il LM perplexity ?
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


class TinyLM(nn.Module):
    """LM décodeur autoregressif simple + self-coherence head."""
    def __init__(self, vocab_size=64, d_model=64, n_heads=4,
                 n_layers=2, max_len=32, state_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.state_dim = state_dim

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Self-state extracted from mean pooling last layer
        self.state_proj = nn.Linear(d_model, state_dim)
        self.state_predictor = nn.Linear(state_dim, state_dim)

    def forward(self, x_ids, prev_self_state=None):
        B, T = x_ids.shape
        z = self.embed(x_ids) + self.pos[:, :T]
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x_ids.device) * float('-inf'), diagonal=1)
        z = self.transformer(z, mask=mask)
        # Self-state from mean pooling
        self_state = self.state_proj(z.mean(dim=1))  # (B, state_dim)
        # Predict next self-state from previous
        if prev_self_state is not None:
            if prev_self_state.dim() == 1:
                prev_self_state = prev_self_state.unsqueeze(0).expand(B, -1)
            pred_self_state = self.state_predictor(prev_self_state)
        else:
            pred_self_state = None
        return self.lm_head(z), self_state, pred_self_state


def generate_synthetic_sequences(n_samples=2000, seq_len=32, vocab_size=64, seed=0):
    """Séquences synthétiques avec structure : digraphes répétitifs.

    Vocab divisé en 4 clusters de 16 tokens. Chaque sequence reste majoritairement
    dans 1 cluster, avec parfois transitions. Apprenable structure markovienne.
    """
    rng = np.random.RandomState(seed)
    seqs = np.zeros((n_samples, seq_len), dtype=np.int64)
    for i in range(n_samples):
        cluster = rng.randint(4)
        for t in range(seq_len):
            # 80% stay in cluster, 20% switch
            if rng.random() < 0.2:
                cluster = rng.randint(4)
            token = cluster * 16 + rng.randint(16)
            seqs[i, t] = token
    return torch.from_numpy(seqs)


def run(weight: float, seed: int, device) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)

    seqs = generate_synthetic_sequences(n_samples=2000, seq_len=32, seed=seed)
    val_seqs = generate_synthetic_sequences(n_samples=400, seq_len=32, seed=seed + 100)

    BATCH = 32
    n_batches = len(seqs) // BATCH

    model = TinyLM(vocab_size=64, d_model=64, state_dim=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    prev_state = torch.zeros(model.state_dim, device=device)
    history = {'task_loss': [], 'self_surprise': []}

    t0 = time.time()
    for ep in range(8):
        perm = torch.randperm(len(seqs))
        for b in range(n_batches):
            batch_idx = perm[b * BATCH:(b + 1) * BATCH]
            batch = seqs[batch_idx].to(device)
            # LM : predict next token (shift)
            x = batch[:, :-1]
            y = batch[:, 1:]
            opt.zero_grad()
            logits, self_state, pred_self = model(x, prev_state)
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1))
            # Self-coherence : pred_self should match current self_state mean
            target_state = self_state.mean(0).detach()
            if pred_self is not None:
                pred_mean = pred_self.mean(0)
                surp = ((pred_mean - target_state) ** 2).mean()
            else:
                surp = torch.zeros(1, device=device)
            loss = lm_loss + weight * surp
            loss.backward()
            opt.step()
            prev_state = target_state
            history['task_loss'].append(lm_loss.item())
            history['self_surprise'].append(surp.item())
    train_time = time.time() - t0

    # Eval val perplexity
    model.eval()
    with torch.no_grad():
        val_x = val_seqs[:, :-1].to(device)
        val_y = val_seqs[:, 1:].to(device)
        logits, _, _ = model(val_x, prev_state)
        val_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                    val_y.reshape(-1))
        val_perplexity = float(torch.exp(val_loss))

    return {
        'weight': weight,
        'seed': seed,
        'final_lm_loss': float(np.mean(history['task_loss'][-50:])),
        'final_self_surprise': float(np.mean(history['self_surprise'][-50:])),
        'val_perplexity': val_perplexity,
        'train_time': train_time,
        'params': sum(p.numel() for p in model.parameters()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.1,0.5,1.0,2.0')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint12c')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    print(f'=== Sprint 12C — LLM self-coherence recipe ===')
    print(f'TinyLM (64-dim, 2 layers) + self-coherence aux loss')
    print(f'Weights: {weights}  Seeds: {seeds}\n')

    rows = []
    for w in weights:
        print(f'--- α={w} ---')
        for seed in seeds:
            r = run(w, seed, device)
            rows.append(r)
            print(f'  seed={seed}  lm_loss={r["final_lm_loss"]:.3f}  '
                  f'val_perplexity={r["val_perplexity"]:.2f}  '
                  f'surp={r["final_self_surprise"]:.4f}  ({r["train_time"]:.1f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 12C — LLM self-coherence recipe\n']
    md.append('## Setup\n')
    md.append('TinyLM : embedding(64) + 2-layer Transformer + LM head + state_predictor.\n')
    md.append('Synthetic sequences markov-structured (4 clusters × 16 tokens).\n')
    md.append('Task : next-token prediction + auxiliary self-coherence (α).\n')

    md.append('## Aggregate\n')
    md.append('| α | lm_loss | val_perplexity | self_surprise | Δ perplexity vs α=0 |')
    md.append('|---|---|---|---|---|')
    zero_pp = None
    if 0.0 in by_w:
        zero_pp = float(np.mean([r['val_perplexity'] for r in by_w[0.0]]))
    for w in sorted(by_w):
        runs = by_w[w]
        def m(k, dec=3):
            vs = [r[k] for r in runs]
            return f'{np.mean(vs):.{dec}f}±{np.std(vs):.{dec}f}'
        delta = (np.mean([r['val_perplexity'] for r in runs]) - zero_pp) if zero_pp else 0
        md.append(
            f'| {w} | {m("final_lm_loss")} | {m("val_perplexity", 2)} | '
            f'{m("final_self_surprise", 4)} | {delta:+.3f} |'
        )

    md.append('\n## Per-run\n')
    md.append('| α | seed | lm_loss | val_perplexity | self_surprise |')
    md.append('|---|---|---|---|---|')
    for r in rows:
        md.append(
            f'| {r["weight"]} | {r["seed"]} | {r["final_lm_loss"]:.3f} | '
            f'{r["val_perplexity"]:.2f} | {r["final_self_surprise"]:.4f} |'
        )

    md.append('\n## Verdict\n')
    if zero_pp is not None:
        pps = {w: float(np.mean([r['val_perplexity'] for r in by_w[w]])) for w in by_w}
        best_w = min(pps, key=pps.get)
        delta = pps[best_w] - zero_pp
        md.append(f'- **Baseline α=0** : val_perplexity = {zero_pp:.2f}')
        md.append(f'- **Best α = {best_w}** : val_perplexity = {pps[best_w]:.2f} (Δ={delta:+.3f})')
        if delta < -0.05:
            md.append(f'  → **RECETTE CONFIRMÉE** : self-coherence améliore LM (perplexity ↓)')
        elif delta > 0.05:
            md.append(f'  → **RECETTE ÉCHOUE** : self-coherence nuit à LM')
        else:
            md.append(f'  → **ÉQUIVALENT** : self-coherence n\'aide ni ne nuit LM dans ce setup')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
