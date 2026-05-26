"""
Sprint 20C — Per-token contrastive aux loss (InfoNCE) au lieu two-halves MSE.

Hypothèse Sprint 20B : two-halves MSE compresse info NON pertinente pour LM.
Fix : aux loss alignée avec next-token via InfoNCE per-token.

Design :
- h_t hidden state per-token
- s_t = state_proj(h_t)
- pred_t = predictor(s_t) : prédit s_{t+1}
- target_t = s_{t+1}.detach()
- InfoNCE : pred_t · target_t (pos) vs pred_t · target_{t'} (neg) sur batch×time

Si recipe = signal-alignment problem, InfoNCE → Δ pp négatif ou neutre.
Si recipe = scale problem fundamental, InfoNCE → Δ pp toujours positif (~+2 pour 410M).
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


def info_nce_loss(pred, target, temperature=0.1):
    """InfoNCE per-token.
    pred (N, d), target (N, d) where N = B*T-1.
    Diagonale = positifs, off-diag = negatifs.
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.t() / temperature  # (N, N)
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)


def run(model_name: str, weight: float, seed: int, device,
        max_steps: int = 80, temperature: float = 0.1) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f'Loading {model_name} LoRA...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    cfg = base.config

    if 'pythia' in model_name.lower():
        target_modules = ['query_key_value', 'dense']
    else:
        target_modules = ['c_attn', 'c_proj']

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.1, bias='none', task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)

    d_model = cfg.hidden_size
    state_dim = 64  # bigger for contrastive
    state_proj = nn.Linear(d_model, state_dim).to(device, dtype=torch.float32)
    predictor = nn.Sequential(
        nn.Linear(state_dim, state_dim * 2), nn.GELU(),
        nn.Linear(state_dim * 2, state_dim),
    ).to(device, dtype=torch.float32)

    model.to(device)
    model.print_trainable_parameters()

    train_seqs, val_seqs = prepare_data(tokenizer, seq_len=128, n_train=400, n_val=80,
                                         seed=seed)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(state_proj.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)

    BATCH = 4
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
            lm_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       y.reshape(-1))
            # Per-token contrastive
            hidden_fp = hidden.float()
            states = state_proj(hidden_fp)         # (B, T, sd)
            pred_states = predictor(states)        # (B, T, sd)
            # Targets : state shifted by 1
            target = states[:, 1:].detach()        # (B, T-1, sd)
            pred = pred_states[:, :-1]             # (B, T-1, sd)
            # Flatten over batch×time pour negatifs
            pred_flat = pred.reshape(-1, state_dim)
            target_flat = target.reshape(-1, state_dim)
            surp = info_nce_loss(pred_flat, target_flat, temperature)
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
                out.logits.float().reshape(-1, out.logits.size(-1)),
                y.reshape(-1)
            ).item()
            n += 1
    val_loss = val_loss_sum / max(1, n)
    val_pp = float(np.exp(val_loss))

    return {
        'model_name': model_name,
        'mode': 'lora_infonce_per_token',
        'weight': weight,
        'seed': seed,
        'temperature': temperature,
        'val_perplexity': val_pp,
        'final_lm_loss': float(np.mean(history['lm_loss'][-10:])),
        'final_surp': float(np.mean(history['surp'][-10:])),
        'init_surp': float(np.mean(history['surp'][:10])),
        'train_time': train_time,
        'max_steps': max_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='EleutherAI/pythia-410m')
    ap.add_argument('--weights', type=str, default='0.0,0.1,1.0')
    ap.add_argument('--seeds', type=str, default='0')
    ap.add_argument('--max-steps', type=int, default=80)
    ap.add_argument('--temperature', type=float, default=0.1)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint20c')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 20C — Per-token InfoNCE contrastive ===')
    print(f'Model: {args.model}  Weights: {weights}  Seeds: {seeds}  '
          f'Steps: {args.max_steps}  T={args.temperature}\n')

    rows = []
    for w in weights:
        print(f'\n--- α={w} ---')
        for seed in seeds:
            r = run(args.model, w, seed, device, args.max_steps, args.temperature)
            rows.append(r)
            print(f'  ✓ seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'surp {r["init_surp"]:.3f}→{r["final_surp"]:.3f}  '
                  f'({r["train_time"]:.0f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 20C — Per-token InfoNCE contrastive\n']
    md.append(f'Model : `{args.model}`\n')
    md.append(f'Mode : LoRA + InfoNCE per-token (vs two-halves MSE Sprint 19)\n')
    md.append('## Aggregate\n')
    md.append('| α | val_pp | surp init→final | Δ pp vs α=0 |')
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
            f'{np.mean(i_sps):.3f}→{np.mean(f_sps):.3f} | '
            f'{delta:+.3f} |'
        )

    md.append('\n## Comparaison MSE two-halves (Sprint 19)\n')
    md.append('| Model | MSE Δ pp | InfoNCE Δ pp | Verdict signal-alignment |')
    md.append('|---|---|---|---|')
    mse_deltas = {
        'EleutherAI/pythia-410m': 2.34,
        'EleutherAI/pythia-1.4b': 1.07,
    }
    if args.model in mse_deltas and zero_pp:
        pps_by_w = {w: float(np.mean([r['val_perplexity'] for r in by_w[w]])) for w in by_w}
        # Compare α=1.0 specifically (not min)
        if 1.0 in pps_by_w:
            infonce_delta = pps_by_w[1.0] - zero_pp
            mse_delta = mse_deltas[args.model]
            if infonce_delta < mse_delta - 0.5:
                verdict = '**Alignment matters** : InfoNCE >> MSE'
            elif infonce_delta < 0:
                verdict = '**RECIPE FONCTIONNE** : Δ négatif sur 410M'
            elif abs(infonce_delta - mse_delta) < 0.5:
                verdict = 'Pas de différence : scale-problem fondamental'
            else:
                verdict = '**InfoNCE pire** : signal-alignment empire'
            md.append(f'| {args.model} | +{mse_delta:.2f} | {infonce_delta:+.3f} | {verdict} |')

    (args.out / f'results_{args.model.replace("/", "_")}.json').write_text(
        json.dumps({'rows': rows, 'args': vars(args)}, default=str, indent=2)
    )
    (args.out / f'report_{args.model.replace("/", "_")}.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
