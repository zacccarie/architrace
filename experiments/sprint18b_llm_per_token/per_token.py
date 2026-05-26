"""
Sprint 18B — LLM recipe per-token sur real text (20 newsgroups).

Améliorations vs S15A/S15D/S17B :
- Per-token self-coherence : chaque z[t] prédit z[t+1] (autoregressive)
- Real text from sklearn 20 newsgroups (vrais articles avec long-range deps)
- Curriculum α : warmup linéaire au lieu de constant

Hypothèse : per-token aux loss + harder text révèle bénéfice mesurable.
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


class PerTokenGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 max_len=128, state_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Per-token state extractor : project each hidden state
        self.state_proj = nn.Linear(d_model, state_dim)
        # Per-token predictor : state_t → state_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
        )

    def forward(self, x_ids):
        B, T = x_ids.shape
        z = self.embed(x_ids) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, device=x_ids.device) * float('-inf'), diagonal=1)
        z = self.transformer(z, mask=mask)
        z = self.ln_f(z)
        lm_logits = self.lm_head(z)
        # Per-token state (B, T, state_dim)
        states = self.state_proj(z)
        # Predictor : map state_t to state_{t+1}
        pred_states = self.predictor(states)  # (B, T, state_dim)
        # Targets for surprise : states shifted by 1
        return lm_logits, states, pred_states


def prepare_data(seq_len=128, n_train=3000, n_val=500, seed=0):
    """20 newsgroups char-level."""
    from sklearn.datasets import fetch_20newsgroups
    print('Loading 20 newsgroups...')
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    text = ' '.join(data.data[:200])  # First 200 docs
    # Filter chars
    text = ''.join(c if 32 <= ord(c) < 127 else ' ' for c in text)
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    ids = np.array([c2i[c] for c in text], dtype=np.int64)
    print(f'Corpus : {len(text)} chars, vocab {len(chars)}')

    rng = np.random.RandomState(seed)
    starts_train = rng.randint(0, len(ids) - seq_len - 1, size=n_train)
    starts_val = rng.randint(0, len(ids) - seq_len - 1, size=n_val)
    train = np.stack([ids[s:s + seq_len] for s in starts_train])
    val = np.stack([ids[s:s + seq_len] for s in starts_val])
    return torch.from_numpy(train), torch.from_numpy(val), len(chars)


_DATA_CACHE = {}


def get_data(seed):
    if seed not in _DATA_CACHE:
        _DATA_CACHE[seed] = prepare_data(seed=seed)
    return _DATA_CACHE[seed]


def run(weight: float, seed: int, device, alpha_warmup: bool = True) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    train_seqs, val_seqs, vocab_size = get_data(seed)

    model = PerTokenGPT(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=4,
                       max_len=128, state_dim=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    BATCH = 16
    n_batches = len(train_seqs) // BATCH

    history = {'lm_loss': [], 'surp': []}
    t0 = time.time()
    total_steps = 6 * n_batches
    step = 0
    for ep in range(6):
        model.train()
        perm = torch.randperm(len(train_seqs))
        for b in range(n_batches):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            batch = train_seqs[idx].to(device)
            x = batch[:, :-1]; y = batch[:, 1:]
            opt.zero_grad()
            logits, states, pred_states = model(x)
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            # Per-token surprise : pred state_t should match state_{t+1}
            # states (B, T_in, sd), pred_states (B, T_in, sd)
            # Compare pred_states[:, :-1] vs states[:, 1:].detach()
            target_next = states[:, 1:].detach()  # (B, T-1, sd)
            pred_curr = pred_states[:, :-1]       # (B, T-1, sd)
            surp = ((pred_curr - target_next) ** 2).mean()
            # Curriculum α
            if alpha_warmup:
                cur_alpha = weight * min(1.0, step / max(1, total_steps // 4))
            else:
                cur_alpha = weight
            (lm_loss + cur_alpha * surp).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            history['lm_loss'].append(lm_loss.item())
            history['surp'].append(surp.item())
            step += 1
    train_time = time.time() - t0

    model.eval()
    val_loss_sum = 0; n = 0
    with torch.no_grad():
        for b in range(len(val_seqs) // BATCH):
            batch = val_seqs[b * BATCH:(b + 1) * BATCH].to(device)
            x = batch[:, :-1]; y = batch[:, 1:]
            logits, _, _ = model(x)
            val_loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                            y.reshape(-1)).item()
            n += 1
    val_loss = val_loss_sum / max(1, n)
    val_pp = float(np.exp(val_loss))

    return {
        'weight': weight,
        'seed': seed,
        'val_perplexity': val_pp,
        'final_lm_loss': float(np.mean(history['lm_loss'][-30:])),
        'final_surp': float(np.mean(history['surp'][-30:])),
        'init_surp': float(np.mean(history['surp'][:30])),
        'train_time': train_time,
        'vocab_size': vocab_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.1,0.5,1.0,2.0')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint18b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 18B — LLM per-token + real text ===')
    print(f'20newsgroups, per-token surprise, α warmup. Weights: {weights}  Seeds: {seeds}\n')

    rows = []
    for w in weights:
        print(f'--- α={w} ---')
        for seed in seeds:
            r = run(w, seed, device)
            rows.append(r)
            print(f'  seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'surp {r["init_surp"]:.3f}→{r["final_surp"]:.4f}  '
                  f'vocab={r["vocab_size"]}  ({r["train_time"]:.0f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 18B — LLM recipe per-token + real text\n']
    md.append('Per-token self-coherence + real text (20 newsgroups) + α curriculum warmup.\n')
    md.append('## Aggregate\n')
    md.append('| α | val_perplexity | final_surp | Δ pp vs α=0 |')
    md.append('|---|---|---|---|')
    zero_pp = float(np.mean([r['val_perplexity'] for r in by_w.get(0.0, [])])) if 0.0 in by_w else 0
    for w in sorted(by_w):
        runs = by_w[w]
        pps = [r['val_perplexity'] for r in runs]
        sps = [r['final_surp'] for r in runs]
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
        md.append(f'- **Baseline α=0** : val_pp = {zero_pp:.3f}')
        md.append(f'- **Best α={best_w}** : val_pp = {pps[best_w]:.3f} (Δ={delta:+.3f})')
        if delta < -0.5:
            md.append(f'  → **RECIPE OPERATIONAL** : per-token + real text révèle gain.')
        elif delta < -0.05:
            md.append(f'  → **GAIN FAIBLE** : direction OK')
        elif delta > 0.5:
            md.append(f'  → **DÉGRADE** : design encore inadéquat')
        else:
            md.append(f'  → **NEUTRE** : effet pas détecté même avec améliorations')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
