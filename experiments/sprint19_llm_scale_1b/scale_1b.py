"""
Sprint 19 — LLM recipe scale 1B+ via LoRA.

Modèles cibles (essai progressif) :
- pythia-410m (4-layer head test rapide)
- pythia-1.4b (cible 1B+)
- gpt2-large 774M (fallback)

Setup :
- LoRA adapters sur attention layers
- Aux self-coherence head sur top
- Tiny corpus (20 newsgroups subset)
- Train 1-2 epochs, batch petit

Hypothèse : à scale, aux loss régularisation devient utile (vs S16C/S17B fail).
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
    text = ' '.join(data.data[:80])  # 80 docs ~ enough for short test
    enc = tokenizer(text, return_tensors='pt', truncation=False)
    ids = enc['input_ids'][0]
    print(f'Corpus : {len(ids)} tokens')
    rng = np.random.RandomState(seed)
    starts_train = rng.randint(0, len(ids) - seq_len - 1, size=n_train)
    starts_val = rng.randint(0, len(ids) - seq_len - 1, size=n_val)
    train = torch.stack([ids[s:s + seq_len] for s in starts_train])
    val = torch.stack([ids[s:s + seq_len] for s in starts_val])
    return train, val


def run(model_name: str, weight: float, seed: int, device, max_steps: int = 80) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f'Loading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in fp16 to fit memory
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    # LoRA config — target attention projections
    cfg = base.config
    # Adapt target_modules per arch
    if 'pythia' in model_name.lower() or 'gpt-neox' in model_name.lower():
        target_modules = ['query_key_value', 'dense']
    else:  # GPT-2 family
        target_modules = ['c_attn', 'c_proj']

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.1, bias='none', task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)

    # Add self-coherence head (fp32 for stability)
    d_model = cfg.hidden_size
    state_dim = 32
    state_proj = nn.Linear(d_model, state_dim).to(device, dtype=torch.float32)
    predictor = nn.Sequential(
        nn.Linear(state_dim, state_dim * 2), nn.GELU(),
        nn.Linear(state_dim * 2, state_dim),
    ).to(device, dtype=torch.float32)

    model.to(device)
    model.print_trainable_parameters()

    # Data
    train_seqs, val_seqs = prepare_data(tokenizer, seq_len=128, n_train=400, n_val=80, seed=seed)

    # Optimizer : LoRA params + aux head
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
            hidden = out.hidden_states[-1]  # (B, T, d) fp16
            lm_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                        y.reshape(-1))
            # Two-halves state (cast hidden to fp32 for aux head)
            T = hidden.shape[1]
            half = T // 2
            hidden_fp = hidden.float()
            sf = state_proj(hidden_fp[:, :half].mean(dim=1))
            ss = state_proj(hidden_fp[:, half:].mean(dim=1))
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

    # Eval val perplexity
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

    n_params = sum(p.numel() for p in base.parameters())
    n_trainable = sum(p.numel() for p in trainable)

    return {
        'model_name': model_name,
        'weight': weight,
        'seed': seed,
        'val_perplexity': val_pp,
        'final_lm_loss': float(np.mean(history['lm_loss'][-10:])),
        'final_surp': float(np.mean(history['surp'][-10:])),
        'init_surp': float(np.mean(history['surp'][:10])),
        'train_time': train_time,
        'n_params_total': n_params,
        'n_trainable': n_trainable,
        'max_steps': max_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='EleutherAI/pythia-410m',
                    help='HuggingFace model name. pythia-410m, pythia-1.4b, gpt2-large')
    ap.add_argument('--weights', type=str, default='0.0,1.0')
    ap.add_argument('--seeds', type=str, default='0,1')
    ap.add_argument('--max-steps', type=int, default=80)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint19')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 19 — LLM scale 1B+ via LoRA ===')
    print(f'Model: {args.model}  Weights: {weights}  Seeds: {seeds}  Steps: {args.max_steps}\n')

    rows = []
    for w in weights:
        print(f'\n--- α={w} ---')
        for seed in seeds:
            r = run(args.model, w, seed, device, args.max_steps)
            rows.append(r)
            print(f'  ✓ seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'surp {r["init_surp"]:.4f}→{r["final_surp"]:.4f}  '
                  f'params total/trainable={r["n_params_total"]:,}/{r["n_trainable"]:,}  '
                  f'({r["train_time"]:.0f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 19 — LLM scale 1B+ via LoRA\n']
    md.append(f'Model : `{args.model}`\n')
    md.append(f'Max steps : {args.max_steps}\n')
    md.append('## Aggregate\n')
    md.append('| α | val_pp | surp init→final | params total / trainable |')
    md.append('|---|---|---|---|')
    zero_pp = float(np.mean([r['val_perplexity'] for r in by_w.get(0.0, [])])) if 0.0 in by_w else 0
    for w in sorted(by_w):
        runs = by_w[w]
        pps = [r['val_perplexity'] for r in runs]
        i_sps = [r['init_surp'] for r in runs]
        f_sps = [r['final_surp'] for r in runs]
        delta = (np.mean(pps) - zero_pp) if zero_pp else 0
        md.append(
            f'| {w} | {np.mean(pps):.3f}±{np.std(pps):.3f} | '
            f'{np.mean(i_sps):.4f}→{np.mean(f_sps):.4f} | '
            f'{runs[0]["n_params_total"]:,} / {runs[0]["n_trainable"]:,} '
            f'(Δ pp = {delta:+.3f}) |'
        )

    md.append('\n## Verdict scale-up\n')
    if zero_pp:
        pps = {w: float(np.mean([r['val_perplexity'] for r in by_w[w]])) for w in by_w}
        best_w = min(pps, key=pps.get)
        delta = pps[best_w] - zero_pp
        n_params = rows[0]['n_params_total']
        md.append(f'- Model size : **{n_params/1e6:.0f}M params**')
        md.append(f'- Baseline α=0 : {zero_pp:.3f}')
        md.append(f'- Best α={best_w} : {pps[best_w]:.3f} (Δ={delta:+.3f})')
        if delta < -0.1:
            md.append(f'  → **RECIPE FONCTIONNE À SCALE** : Δ{delta:.3f} sur {n_params/1e6:.0f}M params.')
            md.append(f'  Hypothèse "need scale" CONFIRMÉE.')
        elif delta > 0.5:
            md.append(f'  → **DÉGRADE encore** : recipe pas viable même à {n_params/1e6:.0f}M.')
        else:
            md.append(f'  → **NEUTRE** : effet pas visible sur cette taille, peut-être encore trop petit.')

    (args.out / f'results_{args.model.replace("/", "_")}.json').write_text(
        json.dumps({'rows': rows, 'args': vars(args)}, default=str, indent=2)
    )
    (args.out / f'report_{args.model.replace("/", "_")}.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
