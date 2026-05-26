"""
Sprint 15A — LLM avec two-halves design (grad backprop dans trunk garanti).

Architecture corrigée :
- z = trunk(seq)  # (B, T, d)
- state_first  = pool(z[:, :T/2])      # (B, state_dim) GRAD activé
- state_second = pool(z[:, T/2:])      # (B, state_dim) GRAD activé
- pred_second  = predictor(state_first)
- surprise = MSE(pred_second, state_second)  # gradient à TRAVERS trunk via DEUX paths

Tâche : Markov mixture HARDER (8 clusters × 16 tokens, transitions non-uniformes).
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


class TwoHalvesLM(nn.Module):
    def __init__(self, vocab_size=128, d_model=64, n_heads=4, n_layers=2,
                 max_len=64, state_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.state_dim = state_dim
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.state_proj = nn.Linear(d_model, state_dim)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
        )

    def forward(self, x_ids):
        B, T = x_ids.shape
        z = self.embed(x_ids) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, device=x_ids.device) * float('-inf'), diagonal=1)
        z = self.transformer(z, mask=mask)
        half = T // 2
        state_first = self.state_proj(z[:, :half].mean(dim=1))   # GRAD
        state_second = self.state_proj(z[:, half:].mean(dim=1))  # GRAD
        pred_second = self.predictor(state_first)
        return self.lm_head(z), state_first, state_second, pred_second


def generate_harder_sequences(n_samples=2000, seq_len=64, vocab_size=128, seed=0):
    """Markov mixture plus dure : 8 clusters × 16 tokens, transitions structurées."""
    rng = np.random.RandomState(seed)
    # Transition matrix between clusters : not uniform
    n_clusters = 8
    T_mat = rng.dirichlet(np.ones(n_clusters) * 0.3, size=n_clusters)
    seqs = np.zeros((n_samples, seq_len), dtype=np.int64)
    for i in range(n_samples):
        cluster = rng.randint(n_clusters)
        for t in range(seq_len):
            if rng.random() < 0.15:
                cluster = rng.choice(n_clusters, p=T_mat[cluster])
            seqs[i, t] = cluster * 16 + rng.randint(16)
    return torch.from_numpy(seqs)


def run(weight: float, seed: int, device) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    seqs = generate_harder_sequences(n_samples=2000, seq_len=64, vocab_size=128, seed=seed)
    val_seqs = generate_harder_sequences(n_samples=400, seq_len=64, vocab_size=128, seed=seed + 100)

    model = TwoHalvesLM(vocab_size=128, d_model=64, max_len=64, state_dim=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    BATCH = 32
    n_batches = len(seqs) // BATCH

    history = {'lm_loss': [], 'surprise': []}
    t0 = time.time()
    for ep in range(8):
        perm = torch.randperm(len(seqs))
        for b in range(n_batches):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            batch = seqs[idx].to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]
            opt.zero_grad()
            logits, sf, ss, pred_second = model(x)
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            # Surprise : pred from first half should match second half
            # GRAD flows through trunk via BOTH state_first AND state_second
            surp = ((pred_second - ss) ** 2).mean()
            loss = lm_loss + weight * surp
            loss.backward()
            opt.step()
            history['lm_loss'].append(lm_loss.item())
            history['surprise'].append(surp.item())
    train_time = time.time() - t0

    model.eval()
    val_loss_sum = 0; n = 0
    with torch.no_grad():
        for b in range(len(val_seqs) // BATCH):
            batch = val_seqs[b * BATCH:(b + 1) * BATCH].to(device)
            x = batch[:, :-1]; y = batch[:, 1:]
            logits, _, _, _ = model(x)
            val_loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            y.reshape(-1)).item()
            n += 1
    val_loss = val_loss_sum / max(1, n)
    val_pp = float(np.exp(val_loss))

    return {
        'weight': weight,
        'seed': seed,
        'final_lm_loss': float(np.mean(history['lm_loss'][-30:])),
        'final_surprise': float(np.mean(history['surprise'][-30:])),
        'init_surprise': float(np.mean(history['surprise'][:30])),
        'val_perplexity': val_pp,
        'train_time': train_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.1,0.5,1.0,2.0,5.0')
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4,5')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint15a')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    print(f'=== Sprint 15A — LLM two-halves design ===')
    print(f'Markov mixture 8 clusters × 16 tokens, harder task. n={len(seeds)} seeds.\n')

    rows = []
    for w in weights:
        print(f'--- α={w} ---')
        for seed in seeds:
            r = run(w, seed, device)
            rows.append(r)
            print(f'  seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'surp {r["init_surprise"]:.4f}→{r["final_surprise"]:.4f}  '
                  f'({r["train_time"]:.1f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 15A — LLM two-halves design\n']
    md.append('Two-halves : pred(first_half) → second_half. Gradient flows through trunk via two paths.\n')
    md.append('## Aggregate\n')
    md.append('| α | val_perplexity | final_surp | Δ pp vs α=0 |')
    md.append('|---|---|---|---|')
    zero_pp = float(np.mean([r['val_perplexity'] for r in by_w.get(0.0, [])])) if 0.0 in by_w else 0
    for w in sorted(by_w):
        runs = by_w[w]
        pps = [r['val_perplexity'] for r in runs]
        sps = [r['final_surprise'] for r in runs]
        delta = (np.mean(pps) - zero_pp) if zero_pp else 0
        md.append(
            f'| {w} | {np.mean(pps):.3f}±{np.std(pps):.3f} | '
            f'{np.mean(sps):.4f}±{np.std(sps):.4f} | {delta:+.3f} |'
        )

    md.append('\n## Verdict\n')
    if zero_pp:
        pps = {w: float(np.mean([r['val_perplexity'] for r in by_w[w]])) for w in by_w}
        best_w = min(pps, key=pps.get)
        delta = pps[best_w] - zero_pp
        md.append(f'- **Baseline α=0** : {zero_pp:.3f}')
        md.append(f'- **Best α={best_w}** : {pps[best_w]:.3f} (Δ={delta:+.3f})')
        if delta < -0.2:
            md.append(f'  → **RECETTE OPÉRATIONNELLE** : self-coherence améliore LM (Δpp={delta:.3f})')
        elif delta > 0.2:
            md.append(f'  → **DÉGRADE** : self-coherence nuit LM ({delta:+.3f})')
        else:
            md.append(f'  → **NEUTRE** : effet trop petit pour cette tâche')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
