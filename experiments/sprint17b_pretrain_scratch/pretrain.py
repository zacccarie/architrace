"""
Sprint 17B — LLM pretrain from scratch + self-coherence aux loss.

Hypothèse : S16C ÉCHEC parce que pretrained → aux loss déstabilise.
Test : pretrain from scratch, aux loss présente depuis step 1.

Architecture : TinyGPT (4 layers, 128d, 4 heads, ~3M params).
Tâche : char-level LM sur Wikipedia-style structured text.
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


CORPUS = """
The history of philosophy is a continuous dialogue across millennia. Plato
proposed that reality consists of eternal forms perceived only by reason.
Aristotle countered that knowledge derives from sensory experience refined
by logic. The medieval scholastics synthesized Greek thought with Christian
theology, producing the elaborate metaphysics of Aquinas and Scotus.

The early modern period witnessed the rise of rationalism in Descartes,
Spinoza, and Leibniz, who sought certain knowledge through pure reason.
Against them, the British empiricists Locke, Berkeley, and Hume argued
that all knowledge originates in experience. Kant attempted a grand
synthesis, claiming that mind structures experience through innate
categories of understanding.

The nineteenth century brought new perspectives. Hegel proposed history
as the unfolding of absolute spirit through dialectical contradictions.
Marx inverted this idealism into a materialist analysis of class struggle.
Nietzsche radically questioned all traditional values, proclaiming that
God is dead and humans must create their own meaning. Existentialists
followed, exploring authenticity, anxiety, and freedom.

In the twentieth century, analytic philosophy emerged in Britain and
America, focusing on language and logic. Wittgenstein revolutionized
the field twice with different views of meaning. Phenomenology developed
in Germany with Husserl and Heidegger, then spread to France with
Sartre, Merleau-Ponty, and others. The structuralists and post-structuralists
challenged stable meaning and identity.

Contemporary philosophy fragments into many specialized subfields. Some
explore consciousness, others ethics in technology, others questions of
identity, others the foundations of mathematics. Cross-disciplinary
work proliferates with cognitive science, neuroscience, and artificial
intelligence raising new philosophical questions about mind, agency,
and the nature of intelligence itself.
""" * 6  # Repeat to make decent corpus


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 max_len=128, state_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.state_dim = state_dim
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Self-coherence components
        self.state_proj = nn.Linear(d_model, state_dim)
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
        half = T // 2
        state_first = self.state_proj(z[:, :half].mean(dim=1))
        state_second = self.state_proj(z[:, half:].mean(dim=1))
        pred_second = self.predictor(state_first)
        return lm_logits, state_first, state_second, pred_second


def prepare_data(seq_len=128, n_train=4000, n_val=500, seed=0):
    chars = sorted(set(CORPUS))
    c2i = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    ids = np.array([c2i[c] for c in CORPUS], dtype=np.int64)

    rng = np.random.RandomState(seed)
    starts_train = rng.randint(0, len(ids) - seq_len - 1, size=n_train)
    starts_val = rng.randint(0, len(ids) - seq_len - 1, size=n_val)
    train = np.stack([ids[s:s + seq_len] for s in starts_train])
    val = np.stack([ids[s:s + seq_len] for s in starts_val])
    return torch.from_numpy(train), torch.from_numpy(val), vocab_size


def run(weight: float, seed: int, device) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    train_seqs, val_seqs, vocab_size = prepare_data(seq_len=128, seed=seed)

    model = TinyGPT(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=4,
                    max_len=128, state_dim=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    BATCH = 16
    n_batches = len(train_seqs) // BATCH

    history = {'lm_loss': [], 'surp': [], 'val_pp_curve': []}
    t0 = time.time()
    for ep in range(6):
        model.train()
        perm = torch.randperm(len(train_seqs))
        for b in range(n_batches):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            batch = train_seqs[idx].to(device)
            x = batch[:, :-1]; y = batch[:, 1:]
            opt.zero_grad()
            logits, sf, ss, pred_second = model(x)
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            surp = ((pred_second - ss) ** 2).mean()
            (lm_loss + weight * surp).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            history['lm_loss'].append(lm_loss.item())
            history['surp'].append(surp.item())

        # Quick val between epochs
        model.eval()
        with torch.no_grad():
            vx = val_seqs[:BATCH * 5].to(device)[:, :-1]
            vy = val_seqs[:BATCH * 5].to(device)[:, 1:]
            vlogits, _, _, _ = model(vx)
            v_loss = F.cross_entropy(vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1))
            history['val_pp_curve'].append(float(torch.exp(v_loss)))

    train_time = time.time() - t0

    # Final eval
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
        'val_perplexity': val_pp,
        'final_lm_loss': float(np.mean(history['lm_loss'][-30:])),
        'final_surp': float(np.mean(history['surp'][-30:])),
        'init_surp': float(np.mean(history['surp'][:30])),
        'val_pp_curve': history['val_pp_curve'],
        'train_time': train_time,
        'params': sum(p.numel() for p in model.parameters()),
        'vocab_size': vocab_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.1,0.5,1.0,2.0')
    ap.add_argument('--seeds', type=str, default='0,1,2,3')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint17b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 17B — LLM pretrain from scratch + aux loss ===')
    print(f'TinyGPT 4 layers 128d ~3M params. Weights: {weights}  Seeds: {seeds}\n')

    rows = []
    for w in weights:
        print(f'--- α={w} ---')
        for seed in seeds:
            r = run(w, seed, device)
            rows.append(r)
            print(f'  seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'curve={[round(p, 2) for p in r["val_pp_curve"]]}  '
                  f'surp {r["init_surp"]:.3f}→{r["final_surp"]:.4f}  '
                  f'({r["train_time"]:.0f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 17B — LLM pretrain from scratch + aux loss\n']
    md.append('TinyGPT 4 layers 128d ~3M params. From scratch. Aux loss intégrée dès step 1.\n')
    md.append('## Aggregate\n')
    md.append('| α | val_perplexity | final_surp | params | Δ pp vs α=0 |')
    md.append('|---|---|---|---|---|')
    zero_pp = float(np.mean([r['val_perplexity'] for r in by_w.get(0.0, [])])) if 0.0 in by_w else 0
    for w in sorted(by_w):
        runs = by_w[w]
        pps = [r['val_perplexity'] for r in runs]
        sps = [r['final_surp'] for r in runs]
        delta = (np.mean(pps) - zero_pp) if zero_pp else 0
        md.append(
            f'| {w} | {np.mean(pps):.3f}±{np.std(pps):.3f} | '
            f'{np.mean(sps):.4f}±{np.std(sps):.4f} | {runs[0]["params"]:,} | {delta:+.3f} |'
        )

    md.append('\n## Verdict\n')
    if zero_pp:
        pps = {w: float(np.mean([r['val_perplexity'] for r in by_w[w]])) for w in by_w}
        best_w = min(pps, key=pps.get)
        delta = pps[best_w] - zero_pp
        md.append(f'- **Baseline α=0** : val_pp = {zero_pp:.3f}')
        md.append(f'- **Best α={best_w}** : val_pp = {pps[best_w]:.3f} (Δ={delta:+.3f})')

        # Compare avec S16C où fine-tune dégradait
        md.append(f'\nComparaison avec S16C (DistilGPT-2 fine-tune) : '
                  f'α=1 → val_pp +5.22 (DÉGRADATION).')
        md.append(f'Ici from-scratch α={best_w} → val_pp {"+%.3f" % delta}')

        if delta < -0.5:
            md.append(f'  → **RECIPE OPÉRATIONNELLE pretrain from scratch** : '
                      f'aux loss aide quand intégrée dès step 1.')
        elif delta < -0.1:
            md.append(f'  → **Amélioration faible** : direction OK, magnitude modeste')
        elif delta > 0.5:
            md.append(f'  → **DÉGRADE encore** : recipe pas fonctionnelle même from scratch')
        else:
            md.append(f'  → **NEUTRE** : effet pas détecté')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
