"""
Sprint 20B — Full fine-tune vs LoRA pour LLM recipe self-coherence.

Hypothèse : LoRA limite l'intervention au rank-16 adapter. Full fine-tune
permet :
- (A) plus de pathways disruptés → recipe pire
- (B) plus de liberté pour allouer subspace dédié → recipe mieux

Test sur pythia-410m et pythia-1.4b (cap mémoire MPS 32GB).
Compare Δ pp vs Sprint 19 LoRA results :
  410M LoRA : +2.34
  1.4B LoRA : +1.07

Si full Δ < LoRA Δ → (B) confirmé, capacité d'allocation = clé
Si full Δ > LoRA Δ → (A) confirmé, LoRA isolait dégradation
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


def prepare_data(tokenizer, seq_len=128, n_train=400, n_val=80, seed=0):
    from sklearn.datasets import fetch_20newsgroups
    print('Loading 20 newsgroups subset...')
    data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    text = ' '.join(data.data[:80])
    enc = tokenizer(text, return_tensors='pt', truncation=False)
    ids = enc['input_ids'][0]
    print(f'Corpus : {len(ids)} tokens')
    rng = np.random.RandomState(seed)
    starts_train = rng.randint(0, len(ids) - seq_len - 1, size=n_train)
    starts_val = rng.randint(0, len(ids) - seq_len - 1, size=n_val)
    train = torch.stack([ids[s:s + seq_len] for s in starts_train])
    val = torch.stack([ids[s:s + seq_len] for s in starts_val])
    return train, val


def run(model_name: str, weight: float, seed: int, device,
        max_steps: int = 40, lr: float = 5e-5) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f'Loading {model_name} (FULL fine-tune)...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Full fine-tune : pas de LoRA, tous params trainable
    # Note : fp32 pour MPS stability (fp16 grads buggy MPS Apple)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    cfg = model.config

    # Self-coherence head
    d_model = cfg.hidden_size
    state_dim = 32
    state_proj = nn.Linear(d_model, state_dim).to(device, dtype=torch.float32)
    predictor = nn.Sequential(
        nn.Linear(state_dim, state_dim * 2), nn.GELU(),
        nn.Linear(state_dim * 2, state_dim),
    ).to(device, dtype=torch.float32)

    model.to(device)
    n_trainable_base = sum(p.numel() for p in model.parameters())
    n_trainable_head = sum(p.numel() for p in state_proj.parameters()) + \
                       sum(p.numel() for p in predictor.parameters())
    print(f'trainable (full) : {n_trainable_base:,} base + {n_trainable_head:,} head')

    train_seqs, val_seqs = prepare_data(tokenizer, seq_len=128, n_train=400, n_val=80,
                                         seed=seed)

    trainable = list(model.parameters()) + list(state_proj.parameters()) + \
                list(predictor.parameters())
    # Lower LR for full fine-tune (vs LoRA 3e-4)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    BATCH = 2  # smaller batch for full fine-tune memory
    history = {'lm_loss': [], 'surp': []}
    t0 = time.time()
    step = 0
    while step < max_steps:
        perm = torch.randperm(len(train_seqs))
        for b in range(0, len(perm), BATCH):
            if step >= max_steps:
                break
            idx = perm[b:b + BATCH]
            batch = train_seqs[idx].to(device)
            x = batch[:, :-1]; y = batch[:, 1:]
            opt.zero_grad()
            out = model(input_ids=x, output_hidden_states=True)
            logits = out.logits
            hidden = out.hidden_states[-1]
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       y.reshape(-1))
            T = hidden.shape[1]
            half = T // 2
            sf = state_proj(hidden[:, :half].mean(dim=1))
            ss = state_proj(hidden[:, half:].mean(dim=1))
            pred_second = predictor(sf)
            surp = ((pred_second - ss) ** 2).mean()
            loss = lm_loss + weight * surp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            history['lm_loss'].append(lm_loss.item())
            history['surp'].append(surp.item())
            step += 1
            if step % 10 == 0:
                print(f'    step {step}/{max_steps}  lm_loss={lm_loss.item():.3f}  '
                      f'surp={surp.item():.4f}')
    train_time = time.time() - t0

    model.eval()
    val_loss_sum = 0; n = 0
    with torch.no_grad():
        for b in range(0, len(val_seqs), BATCH):
            batch = val_seqs[b:b + BATCH].to(device)
            x = batch[:, :-1]; y = batch[:, 1:]
            out = model(input_ids=x)
            val_loss_sum += F.cross_entropy(
                out.logits.reshape(-1, out.logits.size(-1)),
                y.reshape(-1)
            ).item()
            n += 1
    val_loss = val_loss_sum / max(1, n)
    val_pp = float(np.exp(val_loss))

    return {
        'model_name': model_name,
        'mode': 'full_finetune',
        'weight': weight,
        'seed': seed,
        'val_perplexity': val_pp,
        'final_lm_loss': float(np.mean(history['lm_loss'][-10:])),
        'final_surp': float(np.mean(history['surp'][-10:])),
        'init_surp': float(np.mean(history['surp'][:10])),
        'train_time': train_time,
        'n_trainable': n_trainable_base + n_trainable_head,
        'lr': lr,
        'max_steps': max_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='EleutherAI/pythia-410m')
    ap.add_argument('--weights', type=str, default='0.0,1.0')
    ap.add_argument('--seeds', type=str, default='0')
    ap.add_argument('--max-steps', type=int, default=40)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint20b')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 20B — FULL fine-tune (vs LoRA Sprint 19) ===')
    print(f'Model: {args.model}  Weights: {weights}  Seeds: {seeds}  '
          f'Steps: {args.max_steps}  LR: {args.lr}\n')

    rows = []
    for w in weights:
        print(f'\n--- α={w} ---')
        for seed in seeds:
            r = run(args.model, w, seed, device, args.max_steps, args.lr)
            rows.append(r)
            print(f'  ✓ seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'surp {r["init_surp"]:.4f}→{r["final_surp"]:.4f}  '
                  f'trainable={r["n_trainable"]:,}  ({r["train_time"]:.0f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 20B — FULL fine-tune (compare LoRA)\n']
    md.append(f'Model : `{args.model}`\n')
    md.append(f'Mode : full fine-tune (no LoRA)\n')
    md.append(f'LR : {args.lr}\n')
    md.append('## Aggregate\n')
    md.append('| α | val_pp | surp init→final | trainable |')
    md.append('|---|---|---|---|')
    zero_pp = float(np.mean([r['val_perplexity'] for r in by_w.get(0.0, [])])) \
        if 0.0 in by_w else 0
    for w in sorted(by_w):
        runs = by_w[w]
        pps = [r['val_perplexity'] for r in runs]
        i_sps = [r['init_surp'] for r in runs]
        f_sps = [r['final_surp'] for r in runs]
        delta = (np.mean(pps) - zero_pp) if zero_pp else 0
        md.append(
            f'| {w} | {np.mean(pps):.3f}±{np.std(pps):.3f} | '
            f'{np.mean(i_sps):.4f}→{np.mean(f_sps):.4f} | '
            f'{runs[0]["n_trainable"]:,} (Δ pp = {delta:+.3f}) |'
        )

    md.append('\n## Comparaison LoRA (Sprint 19)\n')
    md.append('| Model | LoRA Δ pp | Full Δ pp | Interpretation |')
    md.append('|---|---|---|---|')
    lora_deltas = {
        'EleutherAI/pythia-410m': 2.34,
        'EleutherAI/pythia-1.4b': 1.07,
    }
    if args.model in lora_deltas and zero_pp:
        pps = {w: float(np.mean([r['val_perplexity'] for r in by_w[w]])) for w in by_w}
        best_w = min(pps, key=pps.get)
        full_delta = pps[best_w] - zero_pp
        lora_delta = lora_deltas[args.model]
        if full_delta < lora_delta - 0.2:
            interp = '**Full < LoRA** : capacité d\'allocation libre = clé'
        elif full_delta > lora_delta + 0.2:
            interp = '**Full > LoRA** : LoRA isolait la dégradation'
        else:
            interp = 'Pas de différence claire (variance dominante)'
        md.append(f'| {args.model} | +{lora_delta:.2f} | {full_delta:+.3f} | {interp} |')

    (args.out / f'results_{args.model.replace("/", "_")}.json').write_text(
        json.dumps({'rows': rows, 'args': vars(args)}, default=str, indent=2)
    )
    (args.out / f'report_{args.model.replace("/", "_")}.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
