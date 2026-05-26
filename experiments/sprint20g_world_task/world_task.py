"""
Sprint 20G — LLM on world-modeling task + InfoNCE self-coherence aux loss.

Différent S20F : world = TÂCHE PRINCIPALE, pas warmup.
LLM entraîné à prédire next Lorenz coord, avec InfoNCE per-token sur
hidden states comme aux loss.

Hypothèse : hidden state pendant Lorenz task ENCODE trajectoire smooth.
Self-coherence aux loss = structurally aligned. Forte synergie attendue.

Compare avec :
- S20C LLM-on-text : Δ ≈ -0.1 (marginal)
- S20F LLM warmup-then-text : Δ +0.5 (dégrade)
- S20G LLM-on-world : ? (test)
"""
from __future__ import annotations
import argparse
import json
import sys as _sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import architrace  # noqa

_sys.path.insert(0, str(architrace.DYNAMOSCOPE_PATH))
from backend.systems import lorenz  # type: ignore


def make_lorenz_seqs(seq_len=128, n_train=400, n_val=80, seed=0):
    lz = lorenz(n=4000, dt=0.01)
    coords = np.array(lz['raw_coords'], dtype=np.float32)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
    rng = np.random.RandomState(seed)
    starts_tr = rng.randint(0, len(coords) - seq_len - 1, size=n_train)
    starts_va = rng.randint(0, len(coords) - seq_len - 1, size=n_val)
    tr = np.stack([coords[s:s + seq_len] for s in starts_tr])
    va = np.stack([coords[s:s + seq_len] for s in starts_va])
    return torch.from_numpy(tr), torch.from_numpy(va)


def info_nce_loss(pred, target, temperature=0.1):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.t() / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)


def run(model_name: str, weight: float, seed: int, device,
        max_steps: int = 100, temperature: float = 0.1,
        state_dim: int = 64) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    print(f'\n[α={weight}/seed={seed}] Loading {model_name} LoRA...')
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    cfg = base.config
    target_modules = ['query_key_value', 'dense'] if 'pythia' in model_name.lower() \
                     else ['c_attn', 'c_proj']
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=target_modules,
        lora_dropout=0.1, bias='none', task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)
    model.to(device)

    d_model = cfg.hidden_size
    coord_proj = nn.Linear(3, d_model).to(device, dtype=torch.float32)
    coord_head = nn.Linear(d_model, 3).to(device, dtype=torch.float32)
    state_proj = nn.Linear(d_model, state_dim).to(device, dtype=torch.float32)
    predictor = nn.Sequential(
        nn.Linear(state_dim, state_dim * 2), nn.GELU(),
        nn.Linear(state_dim * 2, state_dim),
    ).to(device, dtype=torch.float32)

    tr_seqs, va_seqs = make_lorenz_seqs(seq_len=128, n_train=400, n_val=80, seed=seed)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(coord_proj.parameters()) + list(coord_head.parameters())
    trainable += list(state_proj.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)

    BATCH = 4
    history = {'task_loss': [], 'surp': []}
    t0 = time.time()
    step = 0
    while step < max_steps:
        perm = torch.randperm(len(tr_seqs))
        for b in range(0, len(perm), BATCH):
            if step >= max_steps:
                break
            idx = perm[b:b + BATCH]
            seqs = tr_seqs[idx].to(device)  # (B, T, 3)
            opt.zero_grad()
            emb_fp32 = coord_proj(seqs)
            emb = emb_fp32.to(dtype=torch.float16)
            out = model(inputs_embeds=emb, output_hidden_states=True)
            hidden = out.hidden_states[-1].float()
            pred_coord = coord_head(hidden)
            task_loss = ((pred_coord[:, :-1] - seqs[:, 1:]) ** 2).mean()
            # InfoNCE per-token sur hidden state
            states = state_proj(hidden)
            pred_states = predictor(states)
            target = states[:, 1:].detach()
            pred = pred_states[:, :-1]
            pred_flat = pred.reshape(-1, state_dim)
            target_flat = target.reshape(-1, state_dim)
            surp = info_nce_loss(pred_flat, target_flat, temperature)
            loss = task_loss + weight * surp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            history['task_loss'].append(task_loss.item())
            history['surp'].append(surp.item())
            step += 1
            if step % 25 == 0:
                print(f'    step {step}/{max_steps}  '
                      f'task_mse={task_loss.item():.4f}  '
                      f'surp={surp.item():.3f}')
    train_time = time.time() - t0

    # Val MSE
    model.eval()
    val_mse_sum = 0.0; n = 0
    with torch.no_grad():
        for b in range(0, len(va_seqs), BATCH):
            seqs = va_seqs[b:b + BATCH].to(device)
            emb = coord_proj(seqs).to(dtype=torch.float16)
            out = model(inputs_embeds=emb)
            hidden = out.hidden_states[-1].float() if hasattr(out, 'hidden_states') \
                     and out.hidden_states is not None else None
            # Re-run with hidden states
            out2 = model(inputs_embeds=emb, output_hidden_states=True)
            hidden = out2.hidden_states[-1].float()
            pred_coord = coord_head(hidden)
            mse = ((pred_coord[:, :-1] - seqs[:, 1:]) ** 2).mean().item()
            val_mse_sum += mse
            n += 1
    val_mse = val_mse_sum / max(1, n)

    return {
        'model_name': model_name,
        'weight': weight,
        'seed': seed,
        'val_mse': val_mse,
        'final_task_loss': float(np.mean(history['task_loss'][-10:])),
        'init_task_loss': float(np.mean(history['task_loss'][:10])),
        'final_surp': float(np.mean(history['surp'][-10:])),
        'init_surp': float(np.mean(history['surp'][:10])),
        'train_time': train_time,
        'max_steps': max_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='EleutherAI/pythia-410m')
    ap.add_argument('--weights', type=str, default='0.0,0.1,1.0')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--max-steps', type=int, default=100)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint20g')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights = [float(w) for w in args.weights.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 20G — LLM on world task + InfoNCE ===')
    print(f'Model: {args.model}  Weights: {weights}  Seeds: {seeds}  '
          f'Steps: {args.max_steps}\n')

    rows = []
    for w in weights:
        print(f'\n--- α={w} ---')
        for seed in seeds:
            r = run(args.model, w, seed, device, args.max_steps)
            rows.append(r)
            print(f'  ✓ α={w}/seed={seed}  val_mse={r["val_mse"]:.5f}  '
                  f'task {r["init_task_loss"]:.3f}→{r["final_task_loss"]:.4f}  '
                  f'surp {r["init_surp"]:.3f}→{r["final_surp"]:.3f}  '
                  f'({r["train_time"]:.0f}s)')

    by_w = {}
    for r in rows:
        by_w.setdefault(r['weight'], []).append(r)

    md = ['# Sprint 20G — LLM-on-world-task + InfoNCE\n']
    md.append(f'Model : `{args.model}`\n')
    md.append('Task : next Lorenz coord regression (LLM via inputs_embeds)\n')
    md.append('Aux  : InfoNCE per-token sur hidden states\n')
    md.append('## Aggregate par α\n')
    md.append('| α | val_mse | task init→final | surp init→final | Δ mse vs α=0 |')
    md.append('|---|---|---|---|---|')
    zero_mse = float(np.mean([r['val_mse'] for r in by_w.get(0.0, [])])) \
        if 0.0 in by_w else 0
    for w in sorted(by_w):
        runs = by_w[w]
        mses = [r['val_mse'] for r in runs]
        i_ts = [r['init_task_loss'] for r in runs]
        f_ts = [r['final_task_loss'] for r in runs]
        i_sps = [r['init_surp'] for r in runs]
        f_sps = [r['final_surp'] for r in runs]
        delta = (np.mean(mses) - zero_mse) if zero_mse else 0
        md.append(
            f'| {w} | {np.mean(mses):.5f}±{np.std(mses):.5f} | '
            f'{np.mean(i_ts):.3f}→{np.mean(f_ts):.4f} | '
            f'{np.mean(i_sps):.3f}→{np.mean(f_sps):.3f} | '
            f'{delta:+.5f} |'
        )

    md.append('\n## Comparaison régimes substrats\n')
    md.append('| Substrat | Task | Aux loss effet (Δ relative) |')
    md.append('|---|---|---|')
    md.append('| Vision MLP/CNN | Image class | Exapte 8-100× speedup |')
    md.append('| ViT | Image class | Inverse 0.7× |')
    md.append('| LLM text S20C | Next-token | -0.5% perplexity (marginal) |')
    md.append('| LLM warmup S20F | Text after Lorenz LoRA | +2% perplexity (degrade) |')
    if zero_mse:
        best_w = min(by_w, key=lambda w: float(np.mean([r['val_mse'] for r in by_w[w]])))
        best_mse = float(np.mean([r['val_mse'] for r in by_w[best_w]]))
        rel = (best_mse - zero_mse) / zero_mse * 100
        md.append(f'| **LLM world task S20G** | Lorenz regression | **{rel:+.1f}% MSE** (α={best_w} best) |')

    (args.out / f'results_{args.model.replace("/", "_")}.json').write_text(
        json.dumps({'rows': rows, 'args': vars(args)}, default=str, indent=2)
    )
    (args.out / f'report_{args.model.replace("/", "_")}.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
