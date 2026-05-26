"""
Sprint 16C — Real LLM (DistilGPT-2) + auxiliary self-coherence.

DistilGPT-2 (~82M params pretrained). Fine-tune sur tiny text avec
auxiliary self-coherence loss. Compare α=0 vs α>0 perplexity.

Si recipe LLM scale aide vrai modèle → recipe productionnable.
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


SAMPLE_TEXT = """
The history of artificial intelligence began in antiquity with myths of
mechanical men and reasoning machines. The dream of creating thinking
machines persisted through centuries of philosophical speculation and
mathematical innovation. Alan Turing proposed the famous test in 1950
that bears his name, asking whether machines could exhibit behavior
indistinguishable from humans. The field formally began at Dartmouth in
1956 where John McCarthy coined the term artificial intelligence.

Neural networks emerged as a major paradigm in the 1980s with backpropagation.
Deep learning revolutionized the field starting around 2012 with AlexNet.
Transformer architectures introduced in 2017 enabled large language models.
GPT, BERT, and their successors demonstrated remarkable capabilities.
The race for ever larger models continues, raising questions about
intelligence, consciousness, and the nature of understanding.

Some researchers argue that scaling alone will produce general intelligence.
Others believe new architectural innovations are needed. The debate continues.
Meanwhile, models grow larger, capabilities expand, and applications
proliferate across every domain of human activity. The future remains uncertain.
"""


class GPT2WithSelfCoherence(nn.Module):
    """Wrapper around DistilGPT2 adding self-coherence auxiliary head."""
    def __init__(self, state_dim=64):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained('distilgpt2')
        d_model = self.lm.config.n_embd  # 768
        self.state_dim = state_dim
        self.state_proj = nn.Linear(d_model, state_dim)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2), nn.GELU(),
            nn.Linear(state_dim * 2, state_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.lm.transformer(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = out.last_hidden_state  # (B, T, 768)
        # LM logits
        lm_logits = self.lm.lm_head(hidden)
        # Two-halves state
        T = hidden.shape[1]
        half = T // 2
        state_first = self.state_proj(hidden[:, :half].mean(dim=1))
        state_second = self.state_proj(hidden[:, half:].mean(dim=1))
        pred_second = self.predictor(state_first)
        return lm_logits, state_first, state_second, pred_second


def prepare_tokens(model, text, seq_len=64, n_samples=200):
    enc = model.tokenizer(text, return_tensors='pt', truncation=False)
    ids = enc['input_ids'][0]  # (N,)
    # Random crops
    seqs = []
    rng = np.random.RandomState(0)
    if len(ids) <= seq_len + 1:
        # Repeat text if too short
        ids = torch.cat([ids] * (seq_len * n_samples // len(ids) + 2))
    for _ in range(n_samples):
        start = rng.randint(0, len(ids) - seq_len - 1)
        seqs.append(ids[start:start + seq_len])
    return torch.stack(seqs)


def run(weight: float, seed: int, device) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    model = GPT2WithSelfCoherence(state_dim=64).to(device)

    train_seqs = prepare_tokens(model, SAMPLE_TEXT * 5, seq_len=64, n_samples=300)
    val_seqs = prepare_tokens(model, SAMPLE_TEXT * 5, seq_len=64, n_samples=60)

    # Freeze most of GPT-2 to fine-tune fast — only last 2 layers + heads
    for p in model.lm.transformer.parameters():
        p.requires_grad = False
    # Unfreeze last 2 transformer blocks
    n_blocks = len(model.lm.transformer.h)
    for blk in model.lm.transformer.h[-2:]:
        for p in blk.parameters():
            p.requires_grad = True
    # Train state_proj + predictor too
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=1e-4)

    BATCH = 8
    n_batches = len(train_seqs) // BATCH
    history = {'lm_loss': [], 'surp': []}
    t0 = time.time()
    for ep in range(3):
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
            opt.step()
            history['lm_loss'].append(lm_loss.item())
            history['surp'].append(surp.item())
    train_time = time.time() - t0

    # Eval
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
        'final_lm_loss': float(np.mean(history['lm_loss'][-20:])),
        'final_surp': float(np.mean(history['surp'][-20:])),
        'init_surp': float(np.mean(history['surp'][:20])),
        'train_time': train_time,
        'n_trainable': sum(p.numel() for p in trainable),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='0.0,0.5,1.0,2.0')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint16c')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 16C — DistilGPT-2 + self-coherence ===')
    print(f'Fine-tune last 2 blocks + heads. Weights: {weights}  Seeds: {seeds}\n')

    rows = []
    for w in weights:
        print(f'--- α={w} ---')
        for seed in seeds:
            r = run(w, seed, device)
            rows.append(r)
            print(f'  seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'surp {r["init_surp"]:.4f}→{r["final_surp"]:.4f}  '
                  f'({r["train_time"]:.1f}s, {r["n_trainable"]:,} trainable)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 16C — DistilGPT-2 + auxiliary self-coherence\n']
    md.append('Fine-tune last 2 transformer blocks + heads on tiny corpus.\n')
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
            md.append(f'  → **CONFIRMÉ SUR VRAI LLM** : recipe scale (Δpp={delta:.3f})')
        elif delta < -0.1:
            md.append(f'  → **AMÉLIORATION MARGINALE** : direction OK')
        elif delta > 0.5:
            md.append(f'  → **DÉGRADE** : aux loss perturbe fine-tune')
        else:
            md.append(f'  → **NEUTRE** : effet pas détecté')

    (args.out / 'results.json').write_text(json.dumps({'rows': rows}, default=float, indent=2))
    (args.out / 'report.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
