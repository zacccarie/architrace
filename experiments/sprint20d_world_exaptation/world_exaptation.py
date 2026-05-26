"""
Sprint 20D — World→Self exaptation pour LLM recipe.

S11.x finding : predictor warmed sur world model (Lorenz) → speedup sur self.
- S11.3 MLP MNIST : 103×
- S15C drift-norm : 74×
- S18C ResNet/Mixer : 8-12×
- S18C ViT : 0.7× (inverse — exception)

Question : exaptation universelle survive jusqu'en LLM ?

Design :
  Phase 1 (World) : LLM + state_proj frozen, predictor seul trained sur
                    Lorenz next-step prediction. 200 steps MSE.
  Phase 2 (Self)  : LoRA unfreeze base + InfoNCE α=0.1 per-token,
                    predictor warmed reuse OR fresh re-init.

3 conditions :
  - none       : fresh predictor + α=0 baseline (sanity)
  - fresh      : fresh predictor + α=0.1 (= Sprint 20C result)
  - world      : Lorenz-warmed predictor + α=0.1
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
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.t() / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)


def build_predictor(state_dim, device):
    return nn.Sequential(
        nn.Linear(state_dim, state_dim * 2), nn.GELU(),
        nn.Linear(state_dim * 2, state_dim),
    ).to(device, dtype=torch.float32)


def phase_world(predictor, state_dim, device, steps=200, seed=0,
                source: str = 'lorenz'):
    """Pre-train predictor sur next-step prediction de source.

    source='lorenz' : chaotic determinist (S11.x baseline)
    source='gaussian' : random walk, x_{t+1} = x_t + N(0, σ). Control non-chaotic.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    if source == 'lorenz':
        lz = lorenz(n=2000, dt=0.01)
        coords = np.array(lz['raw_coords'], dtype=np.float32)
        coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
        padded = np.zeros((len(coords), state_dim), dtype=np.float32)
        padded[:, :3] = coords
    elif source == 'gaussian':
        # Random walk : continuous trajectory mais non-déterministe
        n = 2000
        rng = np.random.RandomState(seed)
        steps_walk = rng.randn(n, state_dim).astype(np.float32) * 0.01
        padded = np.cumsum(steps_walk, axis=0)
        padded = (padded - padded.mean(0)) / (padded.std(0) + 1e-6)
    else:
        raise ValueError(f'unknown source {source}')
    padded_t = torch.from_numpy(padded).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=3e-4, weight_decay=1e-4)
    BATCH = 32
    for step in range(steps):
        idx = torch.randint(0, len(padded) - 1, (BATCH,))
        x = padded_t[idx]
        y = padded_t[idx + 1]
        opt.zero_grad()
        pred = predictor(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0:
            print(f'    [world/{source}] step {step+1}/{steps}  mse={loss.item():.4f}')
    return predictor


def run(model_name: str, condition: str, seed: int, device,
        max_steps: int = 80, weight: float = 0.1,
        temperature: float = 0.1, state_dim: int = 64) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f'\n[{condition}/seed={seed}] Loading {model_name} LoRA...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    state_proj = nn.Linear(d_model, state_dim).to(device, dtype=torch.float32)
    predictor = build_predictor(state_dim, device)

    # Phase 1 World (conditions 'world' = lorenz, 'gaussian' = control)
    if condition == 'world':
        print('[phase 1 world] Pre-training predictor on Lorenz...')
        phase_world(predictor, state_dim, device, steps=200, seed=seed,
                    source='lorenz')
    elif condition == 'gaussian':
        print('[phase 1 gaussian] Pre-training predictor on random walk...')
        phase_world(predictor, state_dim, device, steps=200, seed=seed,
                    source='gaussian')

    # Conditions :
    #   none  : α=0, fresh predictor (sanity baseline)
    #   fresh : α=weight, fresh predictor
    #   world : α=weight, world-warmed predictor (from phase_world)
    if condition == 'none':
        active_weight = 0.0
    else:
        # fresh, world, gaussian all use α=weight
        active_weight = weight

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
            hidden_fp = hidden.float()
            states = state_proj(hidden_fp)
            pred_states = predictor(states)
            target = states[:, 1:].detach()
            pred = pred_states[:, :-1]
            pred_flat = pred.reshape(-1, state_dim)
            target_flat = target.reshape(-1, state_dim)
            surp = info_nce_loss(pred_flat, target_flat, temperature)
            loss = lm_loss + active_weight * surp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            history['lm_loss'].append(lm_loss.item())
            history['surp'].append(surp.item())
            step += 1
            if step % 20 == 0:
                print(f'    [phase 2] step {step}/{max_steps}  '
                      f'lm={lm_loss.item():.3f}  surp={surp.item():.3f}')
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
        'condition': condition,
        'weight_used': active_weight,
        'seed': seed,
        'val_perplexity': val_pp,
        'final_lm_loss': float(np.mean(history['lm_loss'][-10:])),
        'final_surp': float(np.mean(history['surp'][-10:])),
        'init_surp': float(np.mean(history['surp'][:10])),
        'train_time': train_time,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='EleutherAI/pythia-410m')
    ap.add_argument('--conditions', type=str, default='none,fresh,world')
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--max-steps', type=int, default=80)
    ap.add_argument('--weight', type=float, default=0.1)
    ap.add_argument('--out', type=Path,
                    default=Path(__file__).parent.parent.parent / 'results' / 'sprint20d')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conds = [c.strip() for c in args.conditions.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'=== Sprint 20D — World→Self exaptation pour LLM ===')
    print(f'Model: {args.model}  Conditions: {conds}  Seeds: {seeds}  '
          f'Steps: {args.max_steps}  α={args.weight}\n')

    rows = []
    for cond in conds:
        print(f'\n--- {cond} ---')
        for seed in seeds:
            r = run(args.model, cond, seed, device, args.max_steps, args.weight)
            rows.append(r)
            print(f'  ✓ {cond}/seed={seed}  val_pp={r["val_perplexity"]:.3f}  '
                  f'surp {r["init_surp"]:.3f}→{r["final_surp"]:.3f}  '
                  f'({r["train_time"]:.0f}s)')

    by_c = {}
    for r in rows:
        by_c.setdefault(r['condition'], []).append(r)

    md = ['# Sprint 20D — World→Self exaptation (LLM)\n']
    md.append(f'Model : `{args.model}`\n')
    md.append(f'α : {args.weight} (sweet spot Sprint 20C)\n')
    md.append('## Aggregate par condition\n')
    md.append('| condition | val_pp | surp init→final | Δ pp vs none |')
    md.append('|---|---|---|---|')
    none_pp = float(np.mean([r['val_perplexity'] for r in by_c.get('none', [])])) \
        if 'none' in by_c else 0
    for cond in conds:
        if cond not in by_c:
            continue
        runs = by_c[cond]
        pps = [r['val_perplexity'] for r in runs]
        i_sps = [r['init_surp'] for r in runs]
        f_sps = [r['final_surp'] for r in runs]
        delta = (np.mean(pps) - none_pp) if none_pp else 0
        md.append(
            f'| {cond} | {np.mean(pps):.3f}±{np.std(pps):.3f} | '
            f'{np.mean(i_sps):.3f}→{np.mean(f_sps):.3f} | '
            f'{delta:+.3f} |'
        )

    md.append('\n## Verdict exaptation LLM\n')
    if 'world' in by_c and 'fresh' in by_c:
        world_pp = float(np.mean([r['val_perplexity'] for r in by_c['world']]))
        fresh_pp = float(np.mean([r['val_perplexity'] for r in by_c['fresh']]))
        world_init = float(np.mean([r['init_surp'] for r in by_c['world']]))
        fresh_init = float(np.mean([r['init_surp'] for r in by_c['fresh']]))
        speedup = fresh_init / max(world_init, 1e-9)
        md.append(f'- world init surp : {world_init:.3f}')
        md.append(f'- fresh init surp : {fresh_init:.3f}')
        md.append(f'- **Speedup init surp : {speedup:.2f}×**')
        md.append(f'- world val_pp : {world_pp:.3f}')
        md.append(f'- fresh val_pp : {fresh_pp:.3f}')
        md.append(f'- **Δ pp (world vs fresh) : {world_pp - fresh_pp:+.3f}**')
        if world_pp < fresh_pp - 0.1 and speedup > 1.5:
            md.append('  → **EXAPTATION CONFIRMÉE EN LLM** : finding universel survie scale')
        elif speedup > 1.5:
            md.append('  → Speedup init OK, mais pas gain val_pp final')
        else:
            md.append('  → Pas d\'exaptation détectée (LLM = exception comme ViT)')

    md.append('\n## Comparaison historique exaptation\n')
    md.append('| Substrat | Speedup |')
    md.append('|---|---|')
    md.append('| S11.3 MLP MNIST | 103× |')
    md.append('| S15C MNIST drift-norm | 74× |')
    md.append('| S16B CNN CIFAR-100 | 8.1× |')
    md.append('| S18C ResNet/Mixer | 8-12× |')
    md.append('| S18C ViT | 0.7× (inverse) |')
    if 'world' in by_c and 'fresh' in by_c:
        sp = fresh_init / max(world_init, 1e-9)
        md.append(f'| **S20D LLM pythia-410m** | **{sp:.2f}×** |')

    (args.out / f'results_{args.model.replace("/", "_")}.json').write_text(
        json.dumps({'rows': rows, 'args': vars(args)}, default=str, indent=2)
    )
    (args.out / f'report_{args.model.replace("/", "_")}.md').write_text('\n'.join(md))

    print('\n' + '=' * 70)
    print('\n'.join(md))
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
